#include "defs.h"
#include "dma_transpose_node.h"
#include "fcd_ops_utils.h"
#include "habana_nodes.h"
#include "log_manager.h"
#include "node_factory.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "settable.h"
#include "split_strategies.h"
#include "synapse_common_types.h"
#include "transpose_permutation.h"
#include "transpose_utils.h"
#include "broadcast_node.h"
#include "types.h"
#include "utils.h"
#include <memory>
#include <queue>
#include <stdint.h>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "broadcast_node_creator.h"
#include "compilation_hal_reader.h"
#include "transpose_strategies.h"

BroadcastNodeCreator::BroadcastNodeCreator(const NodePtr& node, const unsigned cacheLineSizeInBytes)
: m_input(node->getInput(0)),
  m_shapeTensor(node->getInput(1)),
  m_output(node->getOutput(0)),
  m_name(node->getNodeName()),
  m_cacheLineSizeInBytes(cacheLineSizeInBytes),
  m_physicalBroadcastEngineType(CompilationHalReader::getHalReader()->getBroadcastEngine())
{
    // we aggregate sequences of broadcasted dims, example:
    // [1, 1, 1, X, 1] -> [B1, B2, B3, X, B4] will create 2 Broadcast params:
    // first:  dim start = 0, dim end = 2, total size = B1 * B2 * B3
    // second: dim start = 4, dim end = 4, total size = B4
    Settable<BroadcastParams> current;
    for (unsigned dim = 0; dim <= m_input->getDim(); ++dim)
    {
        if (dim < m_input->getDim() && isTrivialDim(m_input, dim) && !isTrivialDim(m_output, dim))
        {
            if (!current.is_set())
            {
                current            = BroadcastParams();
                current->dimStart  = dim;
                current->totalSize = 1;
            }

            current->dimEnd = dim;
            current->totalSize *= m_output->getSizeInElements(dim);

            m_broadcastedDims.insert(dim);
        }
        else if (current.is_set())
        {
            m_params.push_back(current.value());
            current.unset();
        }
    }
}

bool BroadcastNodeCreator::isTrivialDim(const TensorPtr& t, const unsigned dim)
{
    return t->getSizeInElements(dim) == 1 && t->getMinimalSizeInElements(dim) == 1;
}

TransposePermutationArray BroadcastNodeStrategy::TransposeStrategy::createShiftPermutation(const unsigned tensorDim,
                                                                                           const unsigned axis) const
{
    TransposePermutationArray permutation(tensorDim);
    for (unsigned dim = 0; dim < tensorDim; ++dim)
    {
        permutation[dim] = (TransposePermutationDim)((dim + axis + 1) % tensorDim);
    }
    return permutation;
}

NodePtr BroadcastNodeStrategy::BaseStrategy::createTwoDimTranspose(const TensorPtr& input,
                                                                   std::string_view suffix) const
{
    HB_ASSERT(input->getDim() == 2, "cannot create 2 dim transpose for tensor of dim {}", input->getDim());

    bool                      isShape = input->isShapeTensor();
    TransposePermutationArray twoDimsPermutation(2);
    twoDimsPermutation[0] = (TransposePermutationDim)1;
    twoDimsPermutation[1] = (TransposePermutationDim)0;

    synTransposeParamsNDims twoDimsTransposeParams = permutationToParams(twoDimsPermutation);
    TensorPtr          transposeOutput        = getTensorAfterTranspose(*input, twoDimsPermutation);

    NodePtr transposeNode = NodeFactory::createNode(
        {input},
        {transposeOutput},
        &twoDimsTransposeParams,
        (isShape) ? NodeFactory::transposedShapeNodeTypeName : NodeFactory::transposeNodeTypeName,
        fmt::format("{}/transpose{}{}", m_name, isShape ? "_shape" : "", suffix));
    return transposeNode;
}

// we define shift transpose to be transpose with cyclic permutation
// that mean: for all dims, (permutation[dim] + 1) % tensor dim == (permutation[dim + 1]) % tensor dim,
// such transpose create with: reshape the n-dim input into 2-dim -> transpose -> reshape back to n-dim
// example [A, B, C, D, E], axis 2:
// [A, B, C, D, E] -> [A * B * C, D * E] -> [D * E, A * B * C] -> [D, E, A, B, C].
// the function input that called "tensor" can be the input or the output to the transpose sequence
NodeList BroadcastNodeStrategy::TransposeStrategy::createShiftTransposeSequence(const TensorPtr& tensor,
                                                                                unsigned         axis,
                                                                                bool             fromInput,
                                                                                std::string_view suffix) const
{
    NodeList ret;
    unsigned tensorDim = tensor->getDim();

    TransposePermutationArray permutation    = createShiftPermutation(tensorDim, axis);
    TransposePermutationArray invPermutation = inversePermutation(permutation);

    // in case that the output is provided we need the inverse permutation
    if (!fromInput)
    {
        std::swap(permutation, invPermutation);
        HB_ASSERT(tensorDim >= axis + 2, "transpose strategy applied on the SCD, which is not supported");
        // update the axis to fit the inverse permutation
        axis = tensorDim - axis - 2;
    }
    TensorPtr input  = (fromInput) ? tensor : getTensorAfterTranspose(*tensor, invPermutation);
    TensorPtr output = (fromInput) ? getTensorAfterTranspose(*tensor, permutation) : tensor;

    // first Reshape:
    auto flattenNodes = FcdOpsUtils::createFlattenByReshapeNode(input, axis, m_name);
    for (const NodePtr& node : flattenNodes)
    {
        node->setName(fmt::format("{}{}", node->getNodeName(), suffix));
    }
    const TensorPtr& flattenOutput = flattenNodes.back()->getOutput(0);
    ret.insert(ret.end(), flattenNodes.begin(), flattenNodes.end());

    // 2-dim Transpose:
    NodePtr transposeNode = createTwoDimTranspose(flattenOutput, suffix);
    ret.push_back(transposeNode);

    // second Reshape:
    TensorVector inputs = {transposeNode->getOutput(0)};
    if (m_shapeTensor != nullptr)
    {
        if (fromInput)
        {
            // "tensor" is the input, so need to create transposed shape tensor for
            // the second reshape
            NodePtr extractShape       = FcdOpsUtils::createExtractShapeNode(tensor);
            NodePtr transposeShapeNode = FcdOpsUtils::createTransposedShape(extractShape->getOutput(0), permutation);
            ret.push_back(extractShape);
            ret.push_back(transposeShapeNode);
            inputs.push_back(transposeShapeNode->getOutput(0));
        }
        else
        {
            // "tensor" is the output, so the original shape tensor fit to the second
            // transpose
            inputs.push_back(m_shapeTensor);
        }
    }
    NodePtr reshapeNode = NodeFactory::createNode(inputs,
                                                  {output},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  fmt::format("{}/reshape{}", m_name, suffix));
    ret.push_back(reshapeNode);
    return ret;
}

// step and axis are default, the ends of dim 0 is the original size, and the ends of dim 1 is the remainder
SliceNode::SliceNodeStaticParams
BroadcastNodeStrategy::PhysicalBroadcastStrategy::createSliceParams(const unsigned remainder)
{
    SliceNode::SliceNodeStaticParams sliceParams;
    memset(&sliceParams, 0, sizeof(sliceParams));
    sliceParams.steps[0] = 1;
    sliceParams.steps[1] = 1;
    sliceParams.ends[0]  = m_output->getSizeInElements(0);
    sliceParams.ends[1]  = remainder;
    return sliceParams;
}

// duplicate tensor on dim with concat node, example: [X, Y, Z], dim 1 -> [X, 2Y, Z]
// with high probabilty it will create only one additional memcopy since one of the inputs will be aliased to the output
NodePtr BroadcastNodeStrategy::PhysicalBroadcastStrategy::duplicateTensor(const TensorPtr& tensor,
                                                                          const unsigned   dim) const
{
    SizeArray outputSizes = tensor->getAllSizesInElements();
    outputSizes[dim] *= 2;
    TensorShape shape(tensor->getDim(), outputSizes);
    TensorPtr   output = std::make_shared<Tensor>(shape, tensor->getElementType());

    synConcatenateParams concatParams {.axis = dim};
    return NodeFactory::createNode({tensor, tensor},
                                   {output},
                                   &concatParams,
                                   NodeFactory::concatenateNodeInternalTypeName,
                                   fmt::format("{}/duplicate_{}", m_name, tensor->getName()));
}

void BroadcastNodeStrategy::PhysicalBroadcastStrategy::performWithConcatsAndSlice()
{
    HB_ASSERT(m_output->getDim() == 2, "this function should be called only with 2-dim broadcast node");

    unsigned  broadcastDim  = m_output->getDim() - 1;
    unsigned  broadcastSize = m_output->getDenseSizeInElements() / m_input->getDenseSizeInElements();
    TensorPtr currentInput  = m_input;

    // while the input size is less or equal to the half of the broadcast size
    // we need to duplicate the input with concat node
    while (currentInput->getSizeInElements(broadcastDim) * 2 <= broadcastSize)
    {
        NodePtr concatNode = duplicateTensor(currentInput, broadcastDim);
        m_nodesExceptBroadcasts.push_back(concatNode);
        currentInput = concatNode->getOutput(0);
    }

    unsigned remainder = broadcastSize - currentInput->getSizeInElements(broadcastDim);
    if (remainder != 0)  // broadcast size is not power of 2
    {
        // create slice node from the last concat output to the remainder and concatenate them together
        auto           sliceParams            = createSliceParams(remainder);
        TensorPtr      sliceOutput            = currentInput->clone(false, false, false);
        auto      sizes                  = currentInput->getAllSizesInElements();
        sizes[m_output->getDim() - 1]    = remainder;
        sliceOutput->reshape(m_output->getDim(), sizes.data(), nullptr);

        // create slice node
        NodePtr sliceNode = NodeFactory::createNode({currentInput},
                                                    {sliceOutput},
                                                    &sliceParams,
                                                    NodeFactory::sliceNodeTypeName,
                                                    fmt::format("{}/slice", m_name));
        m_nodesExceptBroadcasts.push_back(sliceNode);

        // create concat node
        synConcatenateParams concatParams {.axis = broadcastDim};
        NodePtr              concatNode = NodeFactory::createNode({currentInput, sliceOutput},
                                                     {m_output},
                                                     &concatParams,
                                                     NodeFactory::concatenateNodeInternalTypeName,
                                                     fmt::format("{}/final_concat", m_name));
        m_nodesExceptBroadcasts.push_back(concatNode);
    }
    else
    {
        m_nodesExceptBroadcasts.back()->replaceOutput(0, m_output);
    }
}

std::string BroadcastNodeStrategy::PhysicalBroadcastStrategy::getTpcBroadcastGuid()
{
    return fmt::format("broadcast_non_fcd_{}", getDtypeSuffixFromSynDataType(m_input->getElementType()));
}

void BroadcastNodeStrategy::PhysicalBroadcastStrategy::extractNodes()
{
    HB_ASSERT(!m_output->isDynamicDim(m_output->getDim() - 1), "dynamic shape of broadcasted dim is not supported");
    HB_ASSERT(m_output->getDim() <= 2, "physical broadcast must have at most 2 dims");

    // if is dynamic shape we perform the broadcast with sequence of concats that duplicate the input
    if (m_output->isDynamicShape())
    {
        performWithConcatsAndSlice();
    }
    else if (m_physicalEngineType == HabanaDeviceType::DEVICE_TPC)
    {
        static constexpr ns_BroadcastNonFcd::Params params = {.axis = 1};
        NodePtr tpcBroadcast = NodeFactory::createNode({m_input}, {m_output}, &params, getTpcBroadcastGuid(), m_name);
        m_nodesExceptBroadcasts.push_back(tpcBroadcast);
    }
    else if (m_physicalEngineType == HabanaDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL)
    {
        NodePtr dmaBroadcast =
            NodePtr(new DMABroadcastNode({m_input}, {m_output}, fmt::format("{}/DmaBroadcast", m_name)));
        HB_ASSERT(dmaBroadcast->validateNode(), "{}: node validation failed", dmaBroadcast->getNodeName());
        m_nodesExceptBroadcasts.push_back(dmaBroadcast);
    }
    else
    {
        HB_ASSERT(false, "engine type: {} is not supported for physical broadcast", m_physicalEngineType);
    }
}

std::string BroadcastNodeStrategy::TpcConstantKernelStrategy::getGuid()
{
    return fmt::format("constant_{}", getDtypeSuffixFromSynDataType(m_input->getElementType()));
}

void BroadcastNodeStrategy::TpcConstantKernelStrategy::extractNodes()
{
    HB_ASSERT(m_input->getDenseSizeInElements() == 1, "constant kernel support only input with one element");
    NodePtr constantNode = NodeFactory::createNode({m_input}, {m_output}, nullptr, getGuid(), m_name);
    m_nodesExceptBroadcasts.push_back(constantNode);
}

void BroadcastNodeStrategy::IdentityStrategy::extractNodes()
{
    HB_ASSERT(m_input->getShape() == m_output->getShape(), "input and output shape mismatch");
    NodePtr identityNode =
        NodeFactory::createNode({m_input}, {m_output}, nullptr, NodeFactory::identityNodeTypeName, m_name);
    m_nodesExceptBroadcasts.push_back(identityNode);
}

unsigned BroadcastNodeCreator::getSizeBeforeFirstBroadcastedDim() const
{
    uint64_t ret = 1;
    for (unsigned dim = 0; dim < m_params[0].dimStart; ++dim)
    {
        ret *= m_input->getSizeInElements(dim);
    }
    HB_ASSERT(ret <= std::numeric_limits<unsigned>::max(), "got overflow for unsigned");
    return ret;
}

bool BroadcastNodeCreator::isFcdBroadcast() const
{
    return m_params[0].dimStart == 0;
}

bool BroadcastNodeCreator::isLinearBroadcast() const
{
    return m_params.size() == 1 && m_params[0].dimEnd == (m_output->getDim() - 1);
}

bool BroadcastNodeCreator::existsTrivialDim() const
{
    for (unsigned dim = 0; dim < m_output->getDim(); ++dim)
    {
        if (isTrivialDim(m_output, dim)) return true;
    }
    return false;
}

bool BroadcastNodeCreator::isGoodUtilization(unsigned fcdSize) const
{
    fcdSize *= m_input->getElementSizeInBytes();
    if (fcdSize % m_cacheLineSizeInBytes == 0) return true;
    unsigned fullUtilization = fcdSize + m_cacheLineSizeInBytes - (fcdSize % m_cacheLineSizeInBytes);
    return (static_cast<float>(fcdSize) / fullUtilization) > BroadcastNodeCreator::goodUtilization;
}

void BroadcastNodeStrategy::BaseStrategy::createNewBroadcast(const TensorVector& inputs,
                                                             const TensorVector& outputs,
                                                             std::string_view    suffix)
{
    m_broadcastNodes.push_back(NodeFactory::createNode(inputs,
                                                       outputs,
                                                       nullptr,
                                                       NodeFactory::broadcastNodeTypeName,
                                                       fmt::format("{}{}", m_name, suffix)));
}

void BroadcastNodeStrategy::TransposeStrategy::extractNodes()
{
    NodeList beforeNodes = createShiftTransposeSequence(m_input, m_dim, true /* from input */, "_before");
    NodeList afterNodes  = createShiftTransposeSequence(m_output, m_dim, false /* from input */, "_after");

    const TensorPtr& broadcastIn  = beforeNodes.back()->getOutput(0);
    const TensorPtr& broadcastOut = afterNodes.front()->getInput(0);

    TensorVector inputs = {broadcastIn};
    // if shape tensor exists we need to transpose him too
    if (m_shapeTensor != nullptr)
    {
        TransposePermutationArray permutation        = createShiftPermutation(broadcastOut->getDim(), m_dim);
        NodePtr                   transposeShapeNode = FcdOpsUtils::createTransposedShape(m_shapeTensor, permutation);
        m_nodesExceptBroadcasts.push_back(transposeShapeNode);
        inputs.push_back(transposeShapeNode->getOutput(0));
    }

    m_nodesExceptBroadcasts.splice(m_nodesExceptBroadcasts.end(), beforeNodes);
    // create new broadcast
    createNewBroadcast(inputs, {broadcastOut}, "/transposed");
    m_nodesExceptBroadcasts.splice(m_nodesExceptBroadcasts.end(), afterNodes);
}

void BroadcastNodeStrategy::SqueezeStrategy::extractNodes()
{
    const unsigned dims = m_input->getDim();

    DimVector trivialDims;
    // SCD to FCD order, that because if we squeeze dim all the dims after shift by 1, so we want to handle them first
    for (int dim = dims - 1; dim >= 0; --dim)
    {
        if (BroadcastNodeCreator::isTrivialDim(m_output, dim))
        {
            trivialDims.push_back(dim);
        }
    }

    TensorPtr input  = m_input;
    TensorPtr output = m_output;
    TensorPtr shape  = m_shapeTensor;
    // create squeeze and expand dim nodes
    for (const unsigned dim : trivialDims)
    {
        synAxisParams params = {.axis = dim};
        // create squeeze node for input
        TensorPtr newInput    = createSqueezedTensor(input, dim);
        NodePtr   squeezeNode = NodeFactory::createNode({input},
                                                      {newInput},
                                                      &params,
                                                      NodeFactory::squeezeNodeTypeName,
                                                      fmt::format("{}/squeeze_dim_{}", m_name, dim));
        m_nodesExceptBroadcasts.push_back(squeezeNode);
        input = newInput;

        // create expand dim node for output
        TensorPtr newOutput  = createSqueezedTensor(output, dim);
        NodePtr   expandNode;
        expandNode = NodeFactory::createNode({newOutput},
                                             {output},
                                             &params,
                                             NodeFactory::expandDimsNodeTypeName,
                                             fmt::format("{}/expand_dim_{}", m_name, dim));
        m_nodesExceptBroadcasts.push_back(expandNode);
        output = newOutput;
        if (shape != nullptr)
        {
            TensorPtr newShape         = createSqueezedTensor(shape, dim);
            NodePtr   squeezeShapeNode = NodeFactory::createNode({shape},
                                                               {newShape},
                                                               &params,
                                                               NodeFactory::squeezeShapeNodeTypeName,
                                                               fmt::format("{}/squeeze_shape_dim_{}", m_name, dim));
            m_nodesExceptBroadcasts.push_back(squeezeShapeNode);
            shape = newShape;
        }
    }

    TensorVector inputs = {input};
    if (shape != nullptr)
    {
        inputs.push_back(shape);
    }

    // create new broadcast
    createNewBroadcast(inputs, {output}, "/squeezed");
}

void BroadcastNodeStrategy::ExpandDimStrategy::extractNodes()
{
    TensorPtr input = m_input;
    for (unsigned dim = m_input->getDim(); dim < m_output->getDim(); ++dim)
    {
        TensorPtr expandOutput = createExpandedTensor(input, dim);
        // create expand dim node
        synExpandDimsParams params     = {.axis = dim};
        NodePtr             expandNode = NodeFactory::createNode({input},
                                                     {expandOutput},
                                                     &params,
                                                     NodeFactory::expandDimsNodeTypeName,
                                                     fmt::format("{}/expand_dim_for_input_{}", m_name, dim));
        m_nodesExceptBroadcasts.push_back(expandNode);
        input = expandOutput;
    }

    TensorVector inputs = {input};
    if (m_shapeTensor != nullptr)
    {
        inputs.push_back(m_shapeTensor);
    }
    createNewBroadcast(inputs, {m_output});
}

void BroadcastNodeStrategy::SliceStrategy::extractNodes()
{
    HB_ASSERT(m_shapeTensor != nullptr, "slice strategy should applied only on dynamic broadcast");

    auto newMinSizes = m_output->getAllMinimalSizesInElements();
    auto newMaxSizes = m_output->getAllSizesInElements();

    // set min broadcast size to be equal to max broadcast size
    newMinSizes[m_output->getDim() - 1] = newMaxSizes[m_output->getDim() - 1];

    TensorPtr broadcastOutput = m_output->clone(false, false, false);
    broadcastOutput->reshape(m_output->getDim(), newMaxSizes.data(), nullptr, newMinSizes.data());

    createNewBroadcast({m_input}, {broadcastOutput}, "/fixed_broadcasted_dim");

    // create slice node
    SliceNode::SliceNodeStaticParams sliceDynamicParams;
    memset(&sliceDynamicParams, 0, sizeof(sliceDynamicParams));
    for (unsigned dim = 0; dim < m_output->getDim(); ++dim)
    {
        sliceDynamicParams.ends[dim]  = m_output->getSizeInElements(dim);
        sliceDynamicParams.steps[dim] = 1;
    }

    NodePtr sliceNode = NodeFactory::createNode({broadcastOutput, m_shapeTensor},
                                                {m_output},
                                                &sliceDynamicParams,
                                                NodeFactory::sliceNodeTypeName,
                                                fmt::format("{}/slice", m_name));
    m_nodesExceptBroadcasts.push_back(sliceNode);
}

// example: if the broadcast is from [A,B,1,C,1] to [A,B,B1,C,B2]
// then the output of flattened broadcast is [A*B*C,B1*B2]
TensorPtr BroadcastNodeStrategy::FlattenStrategy::createBroadcastOutputAfterFlatten() const
{
    // calculate the new broadcast sizes
    SizeArray broadcastOutputSizes;
    SizeArray broadcastOutputMinSizes;
    broadcastOutputSizes[0]    = m_input->getDenseSizeInElements();
    broadcastOutputSizes[1]    = m_output->getDenseSizeInElements() / m_input->getDenseSizeInElements();
    broadcastOutputMinSizes[0] = m_input->getMinimalElements();
    if (m_input->getMinimalElements() == 0)
    {
        broadcastOutputMinSizes[1] = 0;
    }
    else
    {
        broadcastOutputMinSizes[1] = m_output->getMinimalElements() / m_input->getMinimalElements();
    }

    TensorPtr broadcastOutput = m_output->clone(false, false, false);
    broadcastOutput->reshape(2, broadcastOutputSizes.data(), nullptr, broadcastOutputMinSizes.data());
    return broadcastOutput;
}

TensorPtr BroadcastNodeStrategy::FlattenStrategy::createReshapeShapeTensor(const TransposePermutationArray& permutation)
{
    TensorPtr reshapeShapeTensor = nullptr;
    // if transpose is needed after reshape, we need to transpose the original shape tensor
    if (!isSameDataMemOrder(*m_shapeTensor, permutation))
    {
        NodePtr transposeShapeNode = FcdOpsUtils::createTransposedShape(m_shapeTensor, permutation);
        reshapeShapeTensor         = transposeShapeNode->getOutput(0);
        m_nodesExceptBroadcasts.push_back(transposeShapeNode);
    }
    else
    {
        reshapeShapeTensor = m_shapeTensor;
    }
    return reshapeShapeTensor;
}

TensorPtr BroadcastNodeStrategy::FlattenStrategy::createBroadcastShapeTensor(const TensorPtr& outputShape,
                                                                             const unsigned   numDimsThatNotBroadcasted)
{
    TensorPtr broadcastShapeTensor = nullptr;
    if (numDimsThatNotBroadcasted != 0)
    {
        // create flatten shape node
        NodePtr flattenShape = FcdOpsUtils::createFlattenShapeNode(outputShape, numDimsThatNotBroadcasted - 1);
        m_nodesExceptBroadcasts.push_back(flattenShape);
        broadcastShapeTensor = flattenShape->getOutput(0);
    }
    else
    {
        // in case that we perform broadcast on all tensor dims the output shape should be [1, tensor dense size]
        // but it's not supported in our flatten node (but supported in onnx).
        // the solution is to perform flaten that his output shape is [tensor dense size, 1] and transpose after it.
        // Example: broadcast from [1,1,1] -> [4,8,2], so the flatten shape will be: [4,8,2] (F)-> [64,1] (T)-> [1,64]

        NodePtr flattenShape = FcdOpsUtils::createFlattenShapeNode(outputShape, m_input->getDim() - 1);
        m_nodesExceptBroadcasts.push_back(flattenShape);

        NodePtr transposeShapeNode = createTwoDimTranspose(flattenShape->getOutput(0), "_for_all_dims_broadcast");
        m_nodesExceptBroadcasts.push_back(transposeShapeNode);
        broadcastShapeTensor = transposeShapeNode->getOutput(0);
    }
    return broadcastShapeTensor;
}

void BroadcastNodeStrategy::FlattenStrategy::extractNodes()
{
    // create the permutation for the transpose node
    TransposePermutationArray permutation;
    unsigned                  dimsThatNotBroadcastedIndex = 0;
    unsigned                  dimsThatBroadcastedIndex    = m_output->getDim() - m_broadcastedDims.size();
    for (unsigned dim = 0; dim < m_output->getDim(); ++dim)
    {
        if (m_broadcastedDims.count(dim) == 1)
        {
            permutation.push_back((TransposePermutationDim)(dimsThatBroadcastedIndex));
            ++dimsThatBroadcastedIndex;
        }
        else
        {
            permutation.push_back((TransposePermutationDim)(dimsThatNotBroadcastedIndex));
            ++dimsThatNotBroadcastedIndex;
        }
    }

    TransposePermutationArray invPermutation = inversePermutation(permutation);

    // create the first Reshape
    auto             flattenNodes  = FcdOpsUtils::createFlattenByReshapeNode(m_input, m_input->getDim() - 1, m_name);
    const TensorPtr& flattenOutput = flattenNodes.back()->getOutput(0);
    m_nodesExceptBroadcasts.insert(m_nodesExceptBroadcasts.end(), flattenNodes.begin(), flattenNodes.end());

    TensorPtr broadcastOutput = createBroadcastOutputAfterFlatten();

    TensorVector broadcastInputs = {flattenOutput};
    TensorVector reshapeInputs   = {broadcastOutput};
    // create the shape tensors for the new broadcast and the second reshape
    if (m_shapeTensor != nullptr)
    {
        TensorPtr reshapeShapeTensor = createReshapeShapeTensor(invPermutation);
        reshapeInputs.push_back(reshapeShapeTensor);

        TensorPtr broadcastShapeTensor = createBroadcastShapeTensor(reshapeShapeTensor, dimsThatNotBroadcastedIndex);
        broadcastInputs.push_back(broadcastShapeTensor);
    }

    // create new flattened broadcast node
    createNewBroadcast(broadcastInputs, {broadcastOutput}, "/optimized");

    // transpose is not needed if input and output have same data memory order (is linear broadcast)
    if (isSameDataMemOrder(*m_output, invPermutation))
    {
        // create second reshape node
        NodePtr reshapeNode = NodeFactory::createNode(reshapeInputs,
                                                      {m_output},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      fmt::format("{}/reshape", m_name));
        m_nodesExceptBroadcasts.push_back(reshapeNode);
    }
    else
    {
        // create second reshape node
        TensorPtr reshapeOutput = getTensorAfterTranspose(*m_output, invPermutation);
        NodePtr   reshapeNode   = NodeFactory::createNode(reshapeInputs,
                                                      {reshapeOutput},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      fmt::format("{}/reshape", m_name));
        m_nodesExceptBroadcasts.push_back(reshapeNode);

        // create transpose node
        synTransposeParamsNDims transposeParams = permutationToParams(permutation);
        NodePtr                 transposeNode   = NodeFactory::createNode({reshapeOutput},
                                                        {m_output},
                                                        &transposeParams,
                                                        NodeFactory::transposeNodeTypeName,
                                                        fmt::format("{}/transpose", m_name));
        m_nodesExceptBroadcasts.push_back(transposeNode);
    }
}

void BroadcastNodeStrategy::SplitStrategy::extractNodes()
{
    HB_ASSERT(!m_output->isDynamicShape(), "spilt strategy not supported for dynamic broadcast");

    // create first Broadcast
    SizeArray outputSizes                    = m_input->getAllSizesInElements();
    outputSizes[m_broadcastDimSequenceStart] = m_splitBy;
    TensorShape shape                        = TensorShape(m_input->getDim(), outputSizes);
    TensorPtr   firstBroadcastOuput          = std::make_shared<Tensor>(shape, m_input->getElementType());

    createNewBroadcast({m_input}, {firstBroadcastOuput}, "/first_broadcast");

    // create first Reshape
    outputSizes[m_broadcastDimSequenceStart - 1] *= m_splitBy;
    outputSizes[m_broadcastDimSequenceStart] = 1;
    shape                                    = TensorShape(m_input->getDim(), outputSizes);
    TensorPtr firstReshapeOuput              = std::make_shared<Tensor>(shape, m_input->getElementType());
    NodePtr   firstReshapeNode =
        NodeFactory::createNode({firstBroadcastOuput},
                                {firstReshapeOuput},
                                nullptr,
                                NodeFactory::reshapeNodeTypeName,
                                fmt::format("{}/first_reshape_{}", m_name, firstBroadcastOuput->getId()));
    m_nodesExceptBroadcasts.push_back(firstReshapeNode);

    // create second Broadcast
    outputSizes = m_output->getAllSizesInElements();
    outputSizes[m_broadcastDimSequenceStart - 1] *= m_splitBy;
    for (unsigned dim = m_broadcastDimSequenceStart + 1; dim <= m_broadcastDimSequenceEnd; ++dim)
    {
        outputSizes[dim] = 1;
    }
    outputSizes[m_broadcastDimSequenceStart] = m_broadcastDimSequenceTotalSize / m_splitBy;
    shape                                    = TensorShape(m_output->getDim(), outputSizes);
    TensorPtr secondBroadcastOuput           = std::make_shared<Tensor>(shape, m_output->getElementType());

    createNewBroadcast({firstReshapeOuput}, {secondBroadcastOuput}, "/second_broadcast");

    // create second Reshape Node
    NodePtr secondReshapeNode =
        NodeFactory::createNode({secondBroadcastOuput},
                                {m_output},
                                nullptr,
                                NodeFactory::reshapeNodeTypeName,
                                fmt::format("{}/second_reshape_{}", m_name, secondBroadcastOuput->getId()));
    m_nodesExceptBroadcasts.push_back(secondReshapeNode);
}

// create new input and output for the broadcast node, it's a static function because we need to know
// if the strategy is redundant before we create the new nodes
std::pair<TensorPtr, TensorPtr>
BroadcastNodeStrategy::PreTransposeStrategy::creteNewTensors(const TensorPtr& input,
                                                             const TensorPtr& output,
                                                             unsigned         splitBy,
                                                             unsigned         lastBroadcastedDim)
{
    SizeArray broadcastInSizes;
    SizeArray broadcastOutSizes;
    broadcastInSizes.fill(1);
    broadcastOutSizes.fill(1);

    unsigned sizeUntilLastBroadcastedDim = 1;
    unsigned dim                         = 0;
    for (; dim <= lastBroadcastedDim; ++dim)
    {
        broadcastInSizes[dim]  = input->getSizeInElements(dim);
        broadcastOutSizes[dim] = output->getSizeInElements(dim);
        sizeUntilLastBroadcastedDim *= broadcastInSizes[dim];
    }

    // update the new shape
    broadcastInSizes[dim]      = splitBy;
    broadcastOutSizes[dim]     = splitBy;
    broadcastInSizes[dim + 1]  = input->getDenseSizeInElements() / (sizeUntilLastBroadcastedDim * splitBy);
    broadcastOutSizes[dim + 1] = input->getDenseSizeInElements() / (sizeUntilLastBroadcastedDim * splitBy);

    TensorPtr broadcastInput  = input->clone(false, false, false);
    TensorPtr broadcastOutput = output->clone(false, false, false);

    broadcastInput->reshape(dim + 2, broadcastInSizes.data(), nullptr);
    broadcastOutput->reshape(dim + 2, broadcastOutSizes.data(), nullptr);
    return {broadcastInput, broadcastOutput};
}

bool BroadcastNodeStrategy::PreTransposeStrategy::isRedundant(const TensorPtr& input,
                                                              const TensorPtr& output,
                                                              unsigned         splitBy,
                                                              unsigned         lastBroadcastedDim)
{
    auto newTensors = creteNewTensors(input, output, splitBy, lastBroadcastedDim);
    return newTensors.first->compareGeometry(*input);
}

void BroadcastNodeStrategy::PreTransposeStrategy::extractNodes()
{
    HB_ASSERT(!m_output->isDynamicShape(), "pre transpose strategy not supported for dynamic broadcast");
    HB_ASSERT(m_output->getDim() < MAX_DIMENSIONS_NUM,
              "pre transpose strategy not supported {}D tensors",
              m_output->getDim());

    // create new broadcast tensors
    auto newTensors = creteNewTensors(m_input, m_output, m_splitBy, m_lastBroadcastedDim);
    // create Reshape before
    NodePtr reshapeBeforeNode = NodeFactory::createNode({m_input},
                                                        {newTensors.first},
                                                        nullptr,
                                                        NodeFactory::reshapeNodeTypeName,
                                                        fmt::format("{}/reshape_before", m_name));
    m_nodesExceptBroadcasts.push_back(reshapeBeforeNode);

    // create Broadcast Node
    createNewBroadcast({newTensors.first}, {newTensors.second}, "/first_broadcast");

    // create Reshape after
    NodePtr reshapeAfterNode = NodeFactory::createNode({newTensors.second},
                                                       {m_output},
                                                       nullptr,
                                                       NodeFactory::reshapeNodeTypeName,
                                                       fmt::format("{}/reshape_after", m_name));
    m_nodesExceptBroadcasts.push_back(reshapeAfterNode);
}
BroadcastNodeCreator::StrategyPtr BroadcastNodeCreator::tryToUseSplitStrategy(const unsigned effectiveFcd,
                                                                              const unsigned totalSize) const
{
    auto splitBy = findBestSplit(effectiveFcd, totalSize, "Split");
    if (!splitBy.has_value()) return nullptr;
    return StrategyPtr(new BroadcastNodeStrategy::SplitStrategy(m_input,
                                                                m_output,
                                                                m_shapeTensor,
                                                                m_name,
                                                                splitBy.value(),
                                                                m_params[0].dimStart,
                                                                m_params[0].dimEnd,
                                                                m_params[0].totalSize));
}
// find the split size that is enough for good utilization,
// or if such a split doesn't exist, return the maximal split
// if wither the split or the total size left are 1 - the split is not relevant
std::optional<unsigned>
BroadcastNodeCreator::findBestSplit(const unsigned effectiveFcd, unsigned totalSize, std::string_view strategy) const
{
    std::optional<unsigned> ret     = std::nullopt;
    unsigned                splitBy = 1;

    // right now we try only to split by powers of 2, since it's the most common case.
    // we may get improvement if we will iterate over ordered prime factorization of the total size,
    // the above suggestion extended our case since 2 is the smallest prime
    while ((totalSize % 2 == 0) && !isGoodUtilization(effectiveFcd * splitBy))
    {
        totalSize /= 2;
        splitBy *= 2;
    }
    if (splitBy != 1 && totalSize != 1)
    {
        if (!isGoodUtilization(effectiveFcd * splitBy))
        {
            LOG_WARN(BROADCAST_NODE_CREATOR, "found {} strategy to {} with low utilization", strategy, m_name);
        }
        ret = splitBy;
    }
    return ret;
}

NodeList BroadcastNodeCreator::createBroadcast(const NodePtr& node, const unsigned cacheLineSizeInBytes)
{
    NodeList            ret;
    std::queue<NodePtr> broadcastNodes({node});
    unsigned            counter = 0;
    // when we apply a strategy on a broadcast node it may return another broadcast node(s), so we extract all
    // those nodes until we reach final strategies
    while (!broadcastNodes.empty())
    {
        const NodePtr&       broadcastNode = broadcastNodes.front();
        BroadcastNodeCreator creator(broadcastNode, cacheLineSizeInBytes);

        // find the strategy that is most relevant to current broadcast
        // [CID: 40085] False positive - coverity ignores std::map and std::set default c'tor
        auto strategy = creator.findWinningStrategy();
        strategy->extractNodes();
        ret.splice(ret.end(), strategy->getExtractedNodesExceptBroadcasts());

        // since new broadcast may be created we add them to the broadcast node queue
        for (const NodePtr& broadcastNode : strategy->getExtractedBroadcastNodes())
        {
            broadcastNodes.push(broadcastNode);
        }
        LOG_TRACE(BROADCAST_NODE_CREATOR, "applied {} for {}", strategy->strategyName(), broadcastNode->getNodeName());

        broadcastNodes.pop();
        ++counter;
    }
    LOG_TRACE(BROADCAST_NODE_CREATOR, "applied {} strategies for {}", counter, node->getNodeName());
    return ret;
}

// Our first optimization is replace the broadcast with TPC constant kernel, this possible in static shape, on devices
// that perform physical broadcast on the TPC (Gaudi3).
// At first we prepare the tensors if they are not in same dimension, or if there are trivial dims (size 1)
// and then we apply shape manipulation strategies until the shape is fit to physical broadcast
// in the decisions tree:
//
//                     +--------------------------------------------------------------------------+
//                     |broadcast is at most dim 2, and there is only 1 broadcasted dim on the SCD|
//                     +---------------------------------+----------------------------------------+
//                                                       |
//                                           yes (1)     |
//                                 +---------------------+
//                                 |                     |
//                                 |                     |
//                                 |                     |
//                  +--------------v-------------+       | no (2)
//                  |the broadcast dim is dynamic|       |
//                  +----------------+-----------+       |
//                                   |                   |
//                       yes (3)     |                   |
//             +---------------------+                   |
//             |                     |                   |
//             |                     |                   |
//             |                     |                   |
//  +----------v-------+             | no (4)            |
//  |use slice strategy|             |                   |
//  +------------------+             |                   |
//                                   |                   |
//                 +-----------------v---------------+   |
//                 |use physical broadcast,          |   |
//                 |(in gaudi3 we also may use split |   |
//                 |strategy to increase utilization)|   |
//                 +---------------------------------+   |
//                                                       |
//                                                       |
//                                  +--------------------v--------------------+
//                                  |all the broadcasted dims are outside dims|
//                                  +--------------------+--------------------+
//                                                       |
//                                                       |
//                                           yes (5)     |
//                                 +---------------------+
//                                 |                     |
//                                 |                     |
//                                 |                     |
//     +---------------------------v----------------+    | no (6)
//     |use flatten strategy without transpose after|    |
//     +--------------------------------------------+    |
//                                                       |
//                                                       |
//                                  +--------------------v----------------------------+
//                                  |there is good utilization for the transpose after|
//                                  +--------------------+----------------------------+
//                                                       |
//                                           yes (7)     |
//                                 +---------------------+
//                                 |                     |
//                                 |                     |
//        +------------------------v----------------+    | no (8)
//        |use flatten strategy with transpose after|    |
//        +-----------------------------------------+    |
//                                                       |
//                                                       |
//                                               +-------v--------+
//                                               |is FCD broadcast|
//                                               +-------+--------+
//                                           yes (9)     |
//                                 +---------------------+
//                                 |                     |
//                                 |                     |
// +-------------------------------v-----+               | no (10)
// |can flatten the tensor around axis   |               |
// |that both sides have good utilization|               |
// +-----------------------+-------------+               |
//             yes (11)    |                             |
//   +---------------------+                             |
//   |                     |       +---------------------v-----------------------+
//   |                     |       |can split the first broadcasted dims sequence|
//   |                     |       +---------------------+-----------------------+
//   |                     |                             |
//   |                     |  no (12)                    |          yes (13)
//   |                     +-----------+                 +----------------------------+
//   |                                 |                 |                            |
//   |                                 |                 |                            |
//   |                                 |                 |                            |
//   |      +--------------------------v-------+         | no (14)                    |
//   |      |can multiple all dims after last  |         |                            |
//   |      |broadcasted dim, and split to dims|         |                  +---------v--------+
//   |      |that fit to transpose strategy    |         |                  |use split strategy|
//   |      +--------------------------+-------+         |                  +------------------+
//   |                                 |                 |
//   |                                 |        no (15)  |
//   |                                 +-----------------+-------------------------+
//   |                                 |                                           |
//   |                                 |                                           |
//   |                                 | yes (16)                                  |
//   |                                 |                                           |
//   |                                 |                                           |
//   |                                 |                                           |
//   |                                 |                                           |
//   |     +---------------------------v--------------------+                      |
//   |     |use pre transpose strategy with good utilization|                      |
//   |     +------------------------------------------------+                      |
//   |                                                         +-------------------v---------------------+
//   |                                                         |use flatten strategy with low utilization|
//   |                                                         +-----------------------------------------+
//   |
//   |
//   |
//   +---------------------------+
//                               |
//                               |
//   +---------------------------v----------------+
//   |use transpose strategy with good utilization|
//   +--------------------------------------------+

std::shared_ptr<BroadcastNodeStrategy::BaseStrategy> BroadcastNodeCreator::findWinningStrategy()
{
    using namespace BroadcastNodeStrategy;

    // In case that broadcast node input contains only one element, and it should be performed on the TPC engine,
    // it can be replaced with constant kernel which is optimized to this specific case.
    if (isRunOnTpc() && m_input->getDenseSizeInElements() == 1 && !m_output->isDynamicShape())
    {
        return StrategyPtr(new TpcConstantKernelStrategy(m_input, m_output, m_shapeTensor, m_name));
    }

    // first validate that input and output have same dimension, and expand input dim if not
    if (m_input->getDim() != m_output->getDim())
    {
        return StrategyPtr(new ExpandDimStrategy(m_input, m_output, m_shapeTensor, m_name));
    }

    // broadcasted dims not exists - replace with identity node
    if (m_params.empty())
    {
        return StrategyPtr(new IdentityStrategy(m_input, m_output, m_shapeTensor, m_name));
    }

    // if there are trivial dims (size 1), get rid of them
    if (existsTrivialDim())
    {
        return StrategyPtr(new SqueezeStrategy(m_input, m_output, m_shapeTensor, m_name));
    }

    // when this part is reached it's mean that input and output are in same dimension, and there aren't trivial dims.
    // the comments below points to the above flow graph

    if (m_params.size() == 1 && m_input->getDim() <= 2 && m_params[0].dimStart == m_input->getDim() - 1)
    {
        // (1)
        if (m_output->isDynamicDim(m_params[0].dimStart))
        {
            // (3)
            return StrategyPtr(new SliceStrategy(m_input, m_output, m_shapeTensor, m_name));
        }
        // To increase the utilization of TPC broadcast, we try to split it into two broadcast, the first is small
        // and inefficient, and the second is large but efficient.
        if (isRunOnTpc() && !m_output->isDynamicShape())
        {
            const auto& strategy = tryToUseSplitStrategy(m_input->getSizeInElements(0), m_params[0].totalSize);
            if (strategy)
            {
                return strategy;
            }
        }
        // (4)
        return StrategyPtr(
            new PhysicalBroadcastStrategy(m_input, m_output, m_shapeTensor, m_name, m_physicalBroadcastEngineType));
    }

    // (2)
    if (isLinearBroadcast())
    {
        // (5)
        return StrategyPtr(new FlattenStrategy(m_input, m_output, m_shapeTensor, m_name, m_broadcastedDims));
    }
    // the effective FCD is the fcd size during the transpose in the flatten strategy,
    // so if it fcd broadcast we need to take the first broadcast sequence size,
    // if not we need to calculate the dense size before first broadcasted dim
    unsigned fcdDuringTheTranspose = isFcdBroadcast() ? m_params[0].totalSize : getSizeBeforeFirstBroadcastedDim();

    // (6)
    if (isGoodUtilization(fcdDuringTheTranspose))
    {
        // (7)
        return StrategyPtr(new FlattenStrategy(m_input, m_output, m_shapeTensor, m_name, m_broadcastedDims));
    }

    // (8)
    if (isFcdBroadcast())
    {
        // (9)
        unsigned lastBroadcastedDim          = m_params.back().dimEnd;
        unsigned sizeUntilLastBroadcastedDim = 1;
        for (unsigned dim = 0; dim <= lastBroadcastedDim; ++dim)
        {
            sizeUntilLastBroadcastedDim *= m_input->getSizeInElements(dim);
        }
        unsigned oldFcdSize = sizeUntilLastBroadcastedDim;                     // dense size before
        unsigned newFcdSize = m_input->getDenseSizeInElements() / oldFcdSize;  // dense size after
        for (unsigned dim = lastBroadcastedDim + 1; dim < m_input->getDim(); ++dim)
        {
            oldFcdSize *= m_input->getSizeInElements(dim);
            newFcdSize /= m_input->getSizeInElements(dim);
            if (isGoodUtilization(oldFcdSize) && isGoodUtilization(newFcdSize))  // found fully utilized transpose
            {
                // (11)
                return StrategyPtr(new TransposeStrategy(m_input, m_output, m_shapeTensor, m_name, dim));
            }
        }

        // (12)
        if (!m_output->isDynamicShape() && m_output->getDim() < MAX_DIMENSIONS_NUM)
        {
            auto splitBy = findBestSplit(sizeUntilLastBroadcastedDim,
                                         m_input->getDenseSizeInElements() / sizeUntilLastBroadcastedDim,
                                         "PreTranspose");

            // check if the strategy is redundant to avoid infinity loop
            if (splitBy.has_value() &&
                !PreTransposeStrategy::isRedundant(m_input, m_output, splitBy.value(), lastBroadcastedDim))
            {
                // (15)
                return StrategyPtr(new PreTransposeStrategy(m_input,
                                                            m_output,
                                                            m_shapeTensor,
                                                            m_name,
                                                            splitBy.value(),
                                                            lastBroadcastedDim));
            }
        }
    }
    // (10)
    else if (!m_output->isDynamicShape())
    {
        const auto& strategy = tryToUseSplitStrategy(fcdDuringTheTranspose, m_params[0].totalSize);
        if (strategy)
        {
            return strategy;
        }
    }
    // (14)
    LOG_WARN(BROADCAST_NODE_CREATOR, "{}: found only strategy with low utilization, use FlattenStrategy", m_name);
    return StrategyPtr(new FlattenStrategy(m_input, m_output, m_shapeTensor, m_name, m_broadcastedDims));
}
