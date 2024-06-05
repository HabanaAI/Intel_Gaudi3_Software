#include "slice_node.h"

#include "compilation_hal_reader.h"
#include "fcd_ops_utils.h"
#include "h2d_tensors.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "slice_logical_node.h"
#include "synapse_common_types.h"
#include "transpose_utils.h"
#include "types_exception.h"

using SliceNodeStaticParams = SliceNode::SliceNodeStaticParams;

SliceNode::SliceNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     std::string_view    name,
                     eNodeType           type,
                     ShapeFuncID         sifFunction)
: MultiNode(inputs, outputs, name, type, sifFunction)
{
}

void SliceNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    m_params = getSliceParams(userParams, userParamsSize, m_inputs, getSlicedTensor(), getUnslicedTensor(), m_name);
    BaseClass::setParams(userParams, userParamsSize);
}

bool SliceNode::RunOnCpu()
{
    return false;
}

bool SliceNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

void SliceNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

void SliceNode::permuteParams(const PermutationVector& inputPermutations)
{
    HB_ASSERT(!inputPermutations.empty(), "Expecting non-empty input permutation container");
    const gc::Permutation& perm      = inputPermutations[0];
    const auto             inputRank = getInput(0)->getDim();
    perm.permuteShape(m_params.starts.data(), inputRank);
    perm.permuteShape(m_params.steps.data(), inputRank);
    perm.permuteShape(m_params.ends.data(), inputRank);
}

bool SliceNode::findDim(unsigned& dim)
{
    const TensorPtr& input = getInput(TENSOR_IFM);

    for (int i = input->getDim() - 1; i > 0; --i)
    {
        // Find a dim that has no slice and we can use it to transpose
        if (!isStepOnDim(i))
        {
            dim = i;
            return true;
        }
    }
    return false;
}

SliceNodeStaticParams SliceNode::expandDims() const
{
    unsigned            inputDim = getInput(TENSOR_IFM)->getDim();
    SliceNodeStaticParams afterExpandParams = {};

    for (unsigned i = inputDim; i > 0; i--)
    {
        afterExpandParams.starts[i] = m_params.starts[i - 1];
        afterExpandParams.ends[i]   = m_params.ends[i - 1];
        afterExpandParams.steps[i]  = m_params.steps[i - 1];
    }
    afterExpandParams.starts[0] = 0;
    afterExpandParams.ends[0]   = 1;
    afterExpandParams.steps[0]  = 1;

    return afterExpandParams;
}

SliceNodeStaticParams SliceNode::expandParamsFor64bitOperands() const
{
    SliceNodeStaticParams newParams = expandDims();
    newParams.ends[0]             = 2;
    return newParams;
}

SliceNodeStaticParams SliceNode::swapSteps(unsigned axis1, unsigned axis2)
{
    SliceNodeStaticParams afterTransposeParams = m_params;

    // replace steps, starts and ends of axis1 with steps, starts and ends from the axis2
    std::swap(afterTransposeParams.steps[axis1], afterTransposeParams.steps[axis2]);
    std::swap(afterTransposeParams.starts[axis1], afterTransposeParams.starts[axis2]);
    std::swap(afterTransposeParams.ends[axis1], afterTransposeParams.ends[axis2]);

    return afterTransposeParams;
}

NodePtr SliceNode::addTransposeNode(const TensorPtr& tensor, unsigned dimToReplace, bool TransposeBefore)
{
    std::string_view addTransposeName = TransposeBefore ? "_before" : "_after";

    NSizeArray transposeSize = tensor->getAllNSizesInElements();
    std::swap(transposeSize[DIM_C], transposeSize[dimToReplace]);

    TensorPtr transposeTensor = tensor->clone(false, false);
    transposeTensor->resetAliasing();
    transposeTensor->setName(fmt::format("{}_transpose{}", tensor->getName(), addTransposeName));
    transposeTensor->reshape(tensor->getDim(), transposeSize.data(), nullptr);

    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim = tensor->getDim();
    for (int i = 0; i < HABANA_DIM_MAX; i++)
    {
        transposeParams.permutation[i] = i;
    }

    std::swap(transposeParams.permutation[0], transposeParams.permutation[dimToReplace]);

    TensorPtr transposeInput  = TransposeBefore ? tensor : transposeTensor;
    TensorPtr transposeOutput = TransposeBefore ? transposeTensor : tensor;

    NodePtr transposeNode =
        NodeFactory::createNode({transposeInput},
                                {transposeOutput},
                                &transposeParams,
                                NodeFactory::transposeNodeTypeName,
                                fmt::format("{}_transpose{}", transposeTensor->getName(), addTransposeName));

    return transposeNode;
}

static NodePtr createTransposedShape(const TensorPtr& shape, unsigned dimToReplace, unsigned dim)
{
    TransposePermutationArray permutationArray;
    permutationArray = getIdentityPermutation(dim);
    HB_ASSERT(dimToReplace < permutationArray.size(), "{} is out of range", dimToReplace);
    permutationArray[dimToReplace] = (TransposePermutationDim)DIM_C;
    permutationArray[DIM_C]        = (TransposePermutationDim)dimToReplace;

    synTransposeParams params;
    params.tensorDim = dim;
    memcpy(
        params.permutation,
        permutationArray.data(),
        std::min(sizeof(params.permutation), permutationArray.size() * sizeof(TransposePermutationArray::value_type)));

    NSizeArray transposedTensorMaxSizes = shape->getNSizesInElements();
    NSizeArray transposedTensorMinSizes = shape->getNMinimalSizesInElements();
    transposedTensorMaxSizes            = applyPermutationOnSizes(transposedTensorMaxSizes, permutationArray);
    transposedTensorMinSizes            = applyPermutationOnSizes(transposedTensorMinSizes, permutationArray);

    TensorPtr shapeOut                  = shape->cloneGeometry();
    shapeOut->setTensorInWorkspace();
    shapeOut->setShapeTensor(OUTPUT_DESCRIBING_SHAPE_TENSOR);
    shapeOut->setName(fmt::format("{}_transpose", shape->getName()));
    shapeOut->reshape(dim, transposedTensorMaxSizes.data(), nullptr, transposedTensorMinSizes.data());
    return NodeFactory::createNode({shape},
                                   {shapeOut},
                                   &params,
                                   NodeFactory::transposedShapeNodeTypeName,
                                   fmt::format("{}_transpose", shape->getName()));
}

void SliceNode::extractIntoExpandSequence(NodeList& allNodes)
{
    TensorPtr output = getOutput(TENSOR_OFM);
    TensorVector inputs;

    synExpandDimsParams expandDimsParams;
    expandDimsParams.axis = 0;
    for (const TensorPtr& input : getDataInputs())
    {
        TensorPtr sliceInput = createExpandedTensor(input, 0 /* dim */);
        NodePtr   expandNode = NodeFactory::createNode({input},
                                                     {sliceInput},
                                                     &expandDimsParams,
                                                     sizeof(expandDimsParams),
                                                     NodeFactory::expandDimsNodeTypeName,
                                                     fmt::format("{}_expand", m_name));
        allNodes.push_back(expandNode);
        inputs.push_back(sliceInput);
    }

    for (unsigned i = getFirstShapeIndex(); i < m_inputs.size(); i++)
    {
        // need to expand the tensor with a single '0' for STARTS tensor, otherwise regular expand dims.
        auto [expandedShapeTensor, expandShapeNode] =
            (i == STARTS_TENSOR) ? expandShapeTensorWithValue(m_inputs[i], 0 /* dim */, 0 /* fill value */)
                                 : expandTensor(m_inputs[i], 0);
        allNodes.push_back(expandShapeNode);
        inputs.push_back(expandedShapeTensor);
    }

    SliceNodeStaticParams sliceParams  = expandDims();
    TensorPtr           sliceOutput  = createExpandedTensor(output, 0 /* dim */);
    NodePtr             logicalSlice = getSliceNode(inputs, sliceOutput, sliceParams);
    disableFcdExpansion(*dynamic_cast<SliceNode*>(logicalSlice.get()));
    NodePtr shrinkNode = NodeFactory::createNode({sliceOutput},
                                                 {output},
                                                 &expandDimsParams,
                                                 sizeof(expandDimsParams),
                                                 NodeFactory::squeezeNodeTypeName,
                                                 fmt::format("{}_squeeze", m_name));
    allNodes.push_back(logicalSlice);
    allNodes.push_back(shrinkNode);
}

unsigned SliceNode::countDimsWithSlice() const
{
    const TensorPtr& unslicedTensor   = getUnslicedTensor();
    int              numDimsWithSlice = 0;

    for (int i = 0; i < unslicedTensor->getDim(); ++i)
    {
        if (isSliceOnDim(i, unslicedTensor->getSizeInElements(i)))
        {
            numDimsWithSlice++;
        }
    }
    return numDimsWithSlice;
}

void SliceNode::addShiftTransposes(NodeList& allNodes)
{
    const TensorPtr& input  = getInput(TENSOR_IFM);
    const TensorPtr& output = getOutput(TENSOR_OFM);
    unsigned         dim    = input->getDim();

    const auto nodeName = fmt::format("{}/slice_optimization", m_name);

    FcdOpsUtils::ShiftTransposesForFcdOpsResults shiftTransposes =
        FcdOpsUtils::createOppositeShiftTransposes(*CompilationHalReader::getHalReader().get(), nodeName, m_inputs, {output}, 0);

    unsigned     newFcdDim = shiftTransposes.newFcdDim;
    TensorVector inputs    = shiftTransposes.newInputs;

    SliceNodeStaticParams sliceParams;

    for (int i = 0; i < dim; ++i)
    {
        // Create new slice params that fits to the shift transpose permutation
        // Example:
        //      If the shift transpose was: [0,1,2,3,4] -> [3,4,0,1,2] , then newFcdDim = 3
        //      So we also need to rotate left the slice params by newFcdDim = 3
        sliceParams.starts[i] = m_params.starts[(newFcdDim + i) % dim];
        sliceParams.steps[i]  = m_params.steps[(newFcdDim + i) % dim];
        sliceParams.ends[i]   = m_params.ends[(newFcdDim + i) % dim];
    }

    allNodes.splice(allNodes.end(), shiftTransposes.newNodes);
    m_name               = fmt::format("{}/transposed", m_name);
    NodePtr logicalSlice = getSliceNode(inputs, shiftTransposes.newOutputs[0], sliceParams);
    disableFcdExpansion(*dynamic_cast<SliceNode*>(logicalSlice.get()));
    allNodes.push_back(logicalSlice);
}

void SliceNode::addFcdTransposes(unsigned axisToReplaceWith, NodeList& allNodes)
{
    TensorVector inputs;

    for (const TensorPtr& input : getDataInputs())
    {
        NodePtr transposeBefore = addTransposeNode(input, axisToReplaceWith, true);
        inputs.push_back(transposeBefore->getOutput(0));
        allNodes.push_back(transposeBefore);
    }

    for (unsigned i = getFirstShapeIndex(); i < m_inputs.size(); i++)
    {
        NodePtr transposeShape = createTransposedShape(m_inputs[i], axisToReplaceWith, m_inputs[i]->getDim());
        allNodes.push_back(transposeShape);
        inputs.push_back(transposeShape->getOutput(0));
    }

    SliceNodeStaticParams sliceParams    = swapSteps(0, axisToReplaceWith);
    NodePtr             transposeAfter = addTransposeNode(m_outputs[0], axisToReplaceWith, false);
    NodePtr             logicalSlice = getSliceNode(inputs, transposeAfter->getInput(0), sliceParams);
    disableFcdExpansion(*dynamic_cast<SliceNode*>(logicalSlice.get()));

    allNodes.push_back(logicalSlice);
    allNodes.push_back(transposeAfter);
}

// Returns the total dma effort (in bytes) needed for executing this SliceFcdNode
uint64_t SliceNode::getSliceExpectedCost() const
{
    const TensorPtr& slicedTensor         = getSlicedTensor();
    const TensorPtr& unslicedTensor       = getUnslicedTensor();
    TSize            cacheLineSizeInBytes = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    TSize            fcdSize              = 1;
    for (unsigned i = 0; i < slicedTensor->getDim(); i++)
    {
        if (isStepOnDim(i)) break;
        TSize dimSize = unslicedTensor->getSizeInElements(i);
        if (isSliceOnDim(i, dimSize))
        {
            fcdSize *= (m_params.ends[i] - m_params.starts[i]);
            break;
        }
        fcdSize *= dimSize;
    }

    fcdSize *= slicedTensor->getElementSizeInBytes();
    return FcdOpsUtils::calculateExpectedCost(*slicedTensor, cacheLineSizeInBytes, fcdSize);
}

bool SliceNode::isBeneficialToTranspose() const
{
    HB_ASSERT(CompilationHalReader::isHalReaderSet(), "HAL reader is not set");
    if (!m_enableFcdExpansion) return false;
    if (!isFcdSliceNode(getUnslicedTensor())) return false;
    uint64_t                                     sliceExpectedCost = getSliceExpectedCost();
    FcdOpsUtils::ShiftTransposesForFcdOpsResults res;
    findBestDimForTranspose(*CompilationHalReader::getHalReader().get(), 0, getDataInputs(), {getOutput(TENSOR_OFM)}, res);
    // If slice is on fcd only and input tensor has multiple dimensions, can use shiftTransposes optimization
    if (GCFG_ENABLE_SLICE_FCD_OPTIMIZATION.value() && canTranspose() && res.expectedCost < sliceExpectedCost &&
        (getInput(TENSOR_IFM)->getDim() > 1) && (countDimsWithSlice() == 1))
    {
        return true;
    }
    return false;
}

bool SliceNode::shouldExtractToLogicalSlice() const
{
    if (isRedundant()) return false;              // will turn into identity
    if (isDynamicSlice()) return false;           // involves dynamic dma nodes
    if (is64BitOperands()) return false;          // involves logical reinterpret nodes
    if (isStepOnDim(0)) return false;             // must transpose otherwise we get strides on FCd
    if (isBeneficialToTranspose()) return false;  // might want to transpose

    return true;
}

NodeList SliceNode::extractSliceFcdNodes()
{
    LOG_DEBUG(SLICE_NODE, "\nExtracting nodes for: {}", m_name);
    NodeList         allNodes;
    const TensorPtr& input  = getInput(TENSOR_IFM);
    const TensorPtr& output = getOutput(TENSOR_OFM);
    LOG_TRACE(SLICE_NODE,
              "Input: sizes = {}, strides = {} ; Output: sizes = {}, strides = {}",
              input->getDimSizesStr(),
              input->getStridesStr(),
              output->getDimSizesStr(),
              output->getStridesStr());

    if (isBeneficialToTranspose())
    {
        LOG_DEBUG(SLICE_NODE, "Call addShiftTransposes optimization");
        addShiftTransposes(allNodes);
    }
    else if (isStepOnDim(0))  // must transpose anyway
    {
        LOG_TRACE(SLICE_NODE, "Slice on fcd is strided, steps = {}", m_params.steps[0]);
        unsigned axisToReplaceWith = 0;
        if (!canTranspose() || !findDim(axisToReplaceWith))  // try to find dim to transpose with fcd
        {
            LOG_WARN(HABANA_NODE,
                     "Slice ({}): there must be at least one dimension with size 1. Using non-optimal solution",
                     m_name);
            LOG_TRACE(SLICE_NODE, "Slice on all dims. Call extractIntoExpandSequence");
            HB_ASSERT(input->getDim() < Tensor::c_tensorMaxNDim, "Can't expand tensor");
            extractIntoExpandSequence(allNodes);
        }
        else
        {
            LOG_TRACE(HABANA_NODE,
                      "Slice ({}): swaping dimensions 1 and {} so that FCD steps will be 1",
                      getNodeName(),
                      axisToReplaceWith);
            LOG_TRACE(SLICE_NODE,
                      "Swaping dimensions 1 and {} so that FCD steps will be 1. Call addFcdTransposes",
                      axisToReplaceWith);
            addFcdTransposes(axisToReplaceWith, allNodes);
        }
    }
    else  // slice_size != original_dim_size
    {
        LOG_TRACE(SLICE_NODE,
                  "Slice on fcd is not strided, start = {}, end = {}",
                  m_params.starts[0],
                  m_params.ends[0]);
        LOG_DEBUG(SLICE_NODE, "Keep slice on fcd");
        disableFcdExpansion(*this);
    }
    return allNodes;
}

SifNodeParams SliceNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_params;
}

size_t SliceNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_params);
}

bool SliceNode::isFcdSliceNode(const TensorPtr& tensor) const
{
    // Strides on FCD isn't supported so we will add transpose node before and after
    return isSliceOnDim(0, tensor->getSizeInElements(0));
}

bool SliceNode::isSliceOnDim(unsigned dim, TSize dimSizeInElements) const
{
    return (isStepOnDim(dim) || m_params.ends[dim] - m_params.starts[dim] != dimSizeInElements);
}

bool SliceNode::isStepOnDim(unsigned dim) const
{
    if (m_params.steps[dim] > 1 && m_params.steps[dim] < m_params.ends[dim])
        return true;

    if (m_inputs.size() > STEPS_TENSOR)
    {
        if (m_inputs[H2D_TENSOR]->isHost2DeviceTensor())
        {
            synDynamicSliceDmaH2dTensor* h2dData = reinterpret_cast<synDynamicSliceDmaH2dTensor*>(m_inputs[H2D_TENSOR]->getHostMinData());
            HB_ASSERT(h2dData, "Slice node: H2D input tensor data is null");
            return h2dData->steps[dim] > 1;
        }
        else
        {
            return m_inputs[STEPS_TENSOR]->getMinimalSizeInElements(dim) > 1;
        }
    }

    return false;
}

SliceNodeStaticParams SliceNode::getDefaultSliceParams(const TensorPtr& unsliced)
{
    SliceNodeStaticParams defaultParams = {};
    for (auto i = 0; i < unsliced->getDim(); ++i)
    {
        defaultParams.starts[i] = 0;
        defaultParams.ends[i]   = unsliced->getSizeInElements(i);
        defaultParams.steps[i]  = 1;
    }
    return defaultParams;
}

synSliceParamsV2 SliceNode::userParams2NDimParams(UserParams userParams, unsigned userParamsSize)
{
    synSliceParamsV2 ndimParams {{0}};
    HB_ASSERT_PTR(userParams);
    if (userParamsSize == sizeof(synSliceParams))
    {
        // deprecated (non ndim) params
        const auto oldParams = *(synSliceParams*)userParams;
        for (unsigned i = 0; i < MAX_DIMENSIONS_NUM; i++)
        {
            ndimParams.axes[i]   = oldParams.axes[i];
            ndimParams.starts[i] = oldParams.starts[i];
            ndimParams.ends[i]   = oldParams.ends[i];
            ndimParams.steps[i]  = oldParams.steps[i];
        }
    }
    else if (userParamsSize == sizeof(synSliceParamsNDims))
    {
        // deprecated (ndim) params
        const auto oldParams = *(synSliceParamsNDims*)userParams;
        for (unsigned i = 0; i < HABANA_DIM_MAX; i++)
        {
            ndimParams.axes[i]   = oldParams.axes[i];
            ndimParams.starts[i] = oldParams.starts[i];
            ndimParams.ends[i]   = oldParams.ends[i];
            ndimParams.steps[i]  = oldParams.steps[i];
        }
    }
    else
    {
        HB_ASSERT(userParamsSize == sizeof(synSliceParamsV2),
                  "Expecting userParamsSize {} == sizeof(synSliceParamsV2) {}",
                  userParamsSize,
                  sizeof(synSliceParamsV2));
        ndimParams = *(synSliceParamsV2*)userParams;
    }
    return ndimParams;
}

SliceNodeStaticParams SliceNode::getStaticParamsFromDynamicTensors(const TensorVector& inputs,
                                                                   const TensorPtr&    sliced,
                                                                   const TensorPtr&    unsliced)
{
    SliceNodeStaticParams params {};
    if (inputs[H2D_TENSOR]->isHost2DeviceTensor())
    {
        synDynamicSliceDmaH2dTensor* h2dData = reinterpret_cast<synDynamicSliceDmaH2dTensor*>(inputs[H2D_TENSOR]->getHostMaxData());
        HB_ASSERT(h2dData, "Slice node: H2D input tensor data is null");
        std::copy(h2dData->steps, h2dData->steps + h2dData->dims, params.steps.begin());
        std::copy(h2dData->starts, h2dData->starts + h2dData->dims, params.starts.begin());
        for (unsigned i = 0; i < h2dData->dims; i++)
        {
            params.ends[i] = std::min(sliced->getSizeInElements(i) * params.steps[i] + params.starts[i],
                                      unsliced->getSizeInElements(i));
        }
    }
    else
    {
        for (unsigned i = 0; i < inputs[INPUT_TENSOR]->getDim(); i++)
        {
            params.starts[i] = inputs[STARTS_TENSOR]->getSizeInElements(i);
            params.steps[i]  = inputs[STEPS_TENSOR]->getSizeInElements(i);
            params.ends[i]   = std::min(sliced->getSizeInElements(i) * params.steps[i] + params.starts[i],
                                      unsliced->getSizeInElements(i));
        }
    }
    return params;
}

SliceNodeStaticParams SliceNode::getSliceParams(UserParams          userParams,
                                                unsigned            userParamsSize,
                                                const TensorVector& inputs,
                                                const TensorPtr&    sliced,
                                                const TensorPtr&    unsliced,
                                                const std::string&  name)
{
    SliceNodeStaticParams params;
    HB_ASSERT_PTR(sliced);
    HB_ASSERT_PTR(unsliced);
    if (userParamsSize == sizeof(SliceNodeStaticParams))
    {
        params = *(SliceNodeStaticParams*)userParams;
        return params;
    }
    else if (inputs.size() == SliceNode::MAX_NUM_INPUTS || (inputs.size() > SliceNode::H2D_TENSOR && inputs[H2D_TENSOR]->isHost2DeviceTensor()))
    {
        // Convert parameter tensors to node parameters
        // Assign default params, the right values will be assigned in extracting
        params                    = getDefaultSliceParams(sliced);
    }
    else
    {
        if (userParams == nullptr)
        {
            LOG_ERR(HABANA_NODE, "SliceNode userParams is null");
            throw InvalidNodeParamsException(name, "userParams");
        }
        const auto userNdimParams = userParams2NDimParams(userParams, userParamsSize);
        params                    = getDefaultSliceParams(unsliced);
        for (auto i = 0; i < unsliced->getDim(); ++i)
        {
            const auto& axis  = userNdimParams.axes[i];
            const auto& start = userNdimParams.starts[i];
            const auto& step  = userNdimParams.steps[i];
            const auto& end   = userNdimParams.ends[i];

            // supporting old flow that allowed onnx style parameters in which a
            // zero size tensor is a corner case where these parameters makes sense
            bool useDefaultParams = (axis == 0) && (start == 0) && (end == 0) && (sliced->getSizeInElements(0) != 0);
            if (!useDefaultParams)
            {
                params.starts[axis] = start;
                params.steps[axis]  = step;
                params.ends[axis]   = end;
            }
        }
    }
    LOG_TRACE(HABANA_NODE,
              "SliceNode name - {}, params - starts={}, ends={}, steps={}",
              name,
              toString(params.starts.data(), params.starts.data() + unsliced->getDim(), ','),
              toString(params.ends.data(), params.ends.data() + unsliced->getDim(), ','),
              toString(params.steps.data(), params.steps.data() + unsliced->getDim(), ','));
    return params;
}

bool SliceNode::validateSliceInExtraction(const TensorPtr& unsliced, const TensorPtr& sliced) const
{
     // Support for both old and new api.
    if (m_inputs.size() > H2D_TENSOR && m_inputs[H2D_TENSOR]->isHost2DeviceTensor())
    {
        if (hasShapeTensor() && !m_inputs[SHAPE_TENSOR]->isShapeTensor())
        {
            LOG_ERR(HABANA_NODE, "Slice node {}: Invalid inputs, expecting shape tensor at index {}", m_name, SHAPE_TENSOR);
            return false;
        }
        if (hasShapeTensor() && m_inputs[SHAPE_TENSOR]->getDim() != m_inputs[0]->getDim())
        {
            LOG_ERR(HABANA_NODE, "Slice node {}: Shape tensor must have the same rank as input tensor", m_name);
            return false;
        }
        synDynamicSliceDmaH2dTensor* h2dData = reinterpret_cast<synDynamicSliceDmaH2dTensor*>(m_inputs[H2D_TENSOR]->getHostMaxData());
        HB_ASSERT(h2dData, "Slice node {}: H2D input tensor data is null", m_name);
        if (h2dData->dims != m_inputs[0]->getDim())
        {
            LOG_ERR(HABANA_NODE, "Slice node {}: Number of starts/steps must be the same as input tensor's rank", m_name);
            return false;
        }
    }
    else
    {
        for (unsigned i = getFirstShapeIndex(); i < m_inputs.size(); i++)
        {
            if (!m_inputs[i]->isShapeTensor())
            {
                LOG_ERR(HABANA_NODE, "Slice node {}: Invalid inputs, expecting shape tensor at index {}", m_name, i);
                return false;
            }
            if (m_inputs[i]->getDim() != m_inputs[0]->getDim())
            {
                LOG_ERR(HABANA_NODE, "Slice node {}: Invalid shape tensor dimensionality at index {}", m_name, i);
                return false;
            }
        }
    }

    if (unsliced->isZeroSizedDataTensor())
    {
        if (!sliced->isZeroSizedDataTensor())
        {
            LOG_ERR(HABANA_NODE,
                    "Slice: detected zero-sized {}, expected zero-sized {}",
                    unsliced->getName(),
                    sliced->getName());
            return false;
        }
        return MultiNode::validateNode();  // don't validate any further
    }

    unsigned    dim          = unsliced->getDim();
    const auto& aliasedShape = sliced->getAllNSizesInElements();
    const auto& realShape    = unsliced->getAllNSizesInElements();
    bool        dynamicSlice = isDynamicSlice();

    if (unsliced->getDim() != sliced->getDim())
    {
        LOG_ERR(HABANA_NODE,
                "{}: Node {}:{} - Unsliced tensor \"{}\" dims {} != {} dims of sliced tensor \"{}\"",
                HLLOG_FUNC,
                getNodeTypeStr(),
                getNodeName(),
                unsliced->getName(),
                unsliced->getDim(),
                sliced->getName(),
                sliced->getDim());
        return false;
    }

    HB_ASSERT(!dynamicSlice || dim <= SYN_MAX_TENSOR_DIM, "node: {}. ndim dynamic slice is not allowed", m_name);

    for (unsigned i = 0; i < dim; i++)
    {
        if (m_params.ends[i] > realShape[i])
        {
            LOG_ERR(HABANA_NODE,
                    "Slice ({}): invalid ends param, got {} on axis {} which is greater than the unsliced operand "
                    "dimension {}",
                    m_name,
                    m_params.ends[i],
                    i,
                    aliasedShape[i]);
            return false;
        }

        if (m_params.starts[i] > m_params.ends[i])
        {
            LOG_ERR(HABANA_NODE,
                    "Slice ({}): invalid starts and ends params, slice starts in {} on axis {} which is greater than it's end "
                    "at {}",
                    m_name,
                    m_params.starts[i],
                    i,
                    m_params.ends[i]);
            return false;
        }

        if (m_params.ends[i] != m_params.starts[i] &&
            (m_params.steps[i] == 0 ||
             aliasedShape[i] != ceil((m_params.ends[i] - m_params.starts[i]) / double(m_params.steps[i]))))
        {
            LOG_ERR(HABANA_NODE,
                    "Slice ({}): sliced operand shape does not match params on axis {} (start {} end {} step {} dim {})",
                    m_name,
                    i,
                    m_params.starts[i],
                    m_params.ends[i],
                    m_params.steps[i],
                    aliasedShape[i]);
            return false;
        }
    }
    return true;
}


bool SliceNode::validateSlice(const TensorPtr& unsliced, const TensorPtr& sliced) const
{
    if (m_inputs.size() < 1 || m_inputs.size() > MAX_NUM_INPUTS || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Slice node {}: invalid number of operands", m_name);
        return false;
    }
    return MultiNode::validateNode();
}

bool SliceNode::isDynamicSlice() const
{
    if (m_inputs.size() > H2D_TENSOR && isDynamicSliceDmaH2DTensorDynamic(m_inputs[H2D_TENSOR]))
        return true;
    if (m_inputs.size() == MAX_NUM_INPUTS)
        return m_inputs[STARTS_TENSOR]->isDynamicShape() || m_inputs[STEPS_TENSOR]->isDynamicShape();
    return false;
}

NodeList SliceNode::extractDynamicSlice()
{
    bool             isFwdNode = isFwd();
    TensorPtr        newSliced = getSlicedTensor()->clone(false, false, false);
    const TensorPtr& dmaIn     = isFwdNode ? newSliced : getSlicedTensor();
    const TensorPtr& dmaOut    = isFwdNode ? getSlicedTensor() : newSliced;

    const TensorPtr& connectingTensor = isFwdNode ? dmaIn : dmaOut;
    connectingTensor->setConnectAtomicNodes();


    NodePtr dma = NodeFactory::createNode({dmaIn, m_inputs[STEPS_TENSOR], m_inputs[STARTS_TENSOR]},
                                          {dmaOut},
                                          &isFwdNode,
                                          NodeFactory::getDynamicSliceMemcpyNodeGUID(),
                                          fmt::format("{}_dynamic_slice", m_name));

    // save original size to dynamic dma node for later use (validation during patching)
    dynamic_cast<NodeWithParentInfo*>(dma.get())->setParentInfo(getUnslicedTensor()->getTotalSizeInBytes());

    NodePtr slice = getLogicalNode(getUnslicedTensor(), newSliced, m_params);
    HB_ASSERT_PTR(slice);
    return NodeList{slice, dma};
}

/*  sequence:  [X,Y,Z] -> slice -> [x,y,z]
    turns into:
    [X,Y,Z](64b) -> reinterpret-> [X,Y,Z*2](32b)-> reshape-> [X,Y,Z,2] -> slice -> [x,y,z,2]->
                    reshape-> [x,y,z*2](32b)-> reinterpret-> [x,y,z] (64b)
*/
NodeList SliceNode::extractCast64To32BitNodes()
{
    NodeList     retval;
    TensorVector sliceInputs(m_inputs.size());
    // arbitrary choice of 32u since mem move ops are agnostic to signedness
    constexpr synDataType dtype = syn_type_uint32;

    // cast to 32b + reshape data tensors
    for (unsigned i = 0; i < getFirstShapeIndex(); i++)
    {
        auto [newInput, reinterpretInNode, reshapeInNode] = reinterpret64BitTensor(getInput(i), true, dtype);
        retval.emplace_back(std::move(reinterpretInNode));
        retval.emplace_back(std::move(reshapeInNode));
        sliceInputs[i] = std::move(newInput);
    }
    auto [newOutput, reinterpretOutNode, reshapeOutNode] = reinterpret64BitTensor(getOutput(0), false, dtype);

    // transform shape tensor parameters accordingly
    for (unsigned i = getFirstShapeIndex(); i < m_inputs.size(); i++)
    {
        NodePtr   expandNode;
        TensorPtr expandedTensor;
        switch (i)
        {
            case SliceNode::SHAPE_TENSOR:  // shape [N,H,W,C] (64 bit) turns into [N,H,W,C,2] (32 bit)
                std::tie(expandedTensor, expandNode) = expandShapeTensorWithValue(getInput(i), 0, /* fillValue */ 2);
                break;
            case SliceNode::STEPS_TENSOR:  // expand STEPS tensor with '1'
                std::tie(expandedTensor, expandNode) = expandTensor(getInput(i), 0);
                break;
            case SliceNode::STARTS_TENSOR:  // expand STARTS tensor with '0'
                std::tie(expandedTensor, expandNode) = expandShapeTensorWithValue(getInput(i), 0, /* fillValue */ 0);
                break;
            default:
                HB_ASSERT(0, "{}: unexpected shape tensor index", __FUNCTION__);
        }
        sliceInputs[i] = expandedTensor;
        retval.push_back(expandNode);
    }

    // create new transformed slice node
    const NodePtr newSliceNode = getSliceNode(sliceInputs, newOutput, expandParamsFor64bitOperands());
    retval.push_back(reshapeOutNode);
    retval.push_back(reinterpretOutNode);
    retval.push_back(newSliceNode);
    return retval;
}

bool SliceNode::isRedundantSlice(const TensorPtr& unsliced, const TensorPtr& sliced, const SliceNodeStaticParams& p)
{
    if (!sliced->compareGeometry(*unsliced)) return false;

    for (unsigned i = 0; i < unsliced->getDim(); i++)
    {
        if (p.steps[i] != 1) return false;
        if (p.ends[i] != unsliced->getSizeInElements(i)) return false;
        if (p.starts[i] != 0) return false;
    }

    return true;
}

bool SliceNode::isRedundant() const
{
    if (isDynamicShape() || isDynamicSlice()) return false;
    return isRedundantSlice(getUnslicedTensor(), getSlicedTensor(), m_params);
}

NodeList SliceNode::extractToIdentity()
{
    const TensorPtr& in  = isFwd() ? getUnslicedTensor() : getSlicedTensor();
    const TensorPtr& out = isFwd() ? getSlicedTensor() : getUnslicedTensor();

    NodePtr identity = NodeFactory::createNode({in}, {out}, nullptr, NodeFactory::identityNodeTypeName, m_name);
    return {identity};
}

NodePtr SliceNode::convertH2DToShape()
{
    synDynamicSliceDmaH2dTensor* inData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(m_inputs[H2D_TENSOR]->getHostMaxData());
    HB_ASSERT(inData, "input data host address is null");

    TensorPtr stepsTensor = std::make_shared<Tensor>(inData->dims, inData->steps, syn_type_uint32);
    stepsTensor->setShapeTensor(synTensorType::SHAPE_TENSOR);
    stepsTensor->setName(fmt::format("{}_steps", m_name));

    TensorPtr startsTensor = std::make_shared<Tensor>(inData->dims, inData->starts, syn_type_uint32);
    startsTensor->setShapeTensor(synTensorType::SHAPE_TENSOR);
    startsTensor->setName(fmt::format("{}_starts", m_name));

    const TensorPtr& h2dTensor = m_inputs[H2D_TENSOR];

    NodePtr convert = NodeFactory::createNode({h2dTensor},
                                              {stepsTensor, startsTensor},
                                              nullptr,
                                              NodeFactory::sliceConversionNodeTypeName,
                                              fmt::format("{}_convert", m_name));
    convert->inferOutputsSizes(synDeviceTypeInvalid, false);  // device type not needed
    return convert;
}

bool SliceNode::isDataMovementMultiNode() const
{
    // if the slice will be extracted into identity, we do want to to happen as soon as possible
    return !isRedundant() && !shouldExtractToLogicalSlice();
}

NodeList SliceNode::extractNodes()
{
    if (m_inputs.size() == SliceNode::MAX_NUM_INPUTS || (m_inputs.size() > SliceNode::H2D_TENSOR && m_inputs[H2D_TENSOR]->isHost2DeviceTensor()))
    {
        // Convert parameter tensors to node parameters
        m_params = getStaticParamsFromDynamicTensors(m_inputs, getSlicedTensor(), getUnslicedTensor());
        validateSliceInExtraction(getUnslicedTensor(), getSlicedTensor());
    }
    if (m_inputs.size() > H2D_TENSOR && m_inputs[H2D_TENSOR]->isHost2DeviceTensor())
    {
        NodePtr convertNode = convertH2DToShape();
        NodePtr sliceNode   = getSliceNode(
            {m_inputs[INPUT_TENSOR], m_inputs[SHAPE_TENSOR], convertNode->getOutput(0), convertNode->getOutput(1)},
            m_outputs[TENSOR_OFM],
            m_params);
        return {convertNode, sliceNode};
    }

    if (isRedundant())
    {
        return extractToIdentity();
    }

    if (is64BitOperands())
    {
        return extractCast64To32BitNodes();
    }

    if (m_enableFcdExpansion && isFcdSliceNode(getUnslicedTensor()))
    {
        return extractSliceFcdNodes();
    }

    return {};
}
