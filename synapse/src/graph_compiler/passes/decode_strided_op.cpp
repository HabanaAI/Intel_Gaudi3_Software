#include "decode_strided_op.h"

#include "node_factory.h"
#include "strided_insert_node.h"
#include "strided_view_node.h"
#include "strided_op_node_utils.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpNodeInfo
///////////////////////////////////////////////////////////////////////////////////////////////////

struct StridedOpNodeInfo
{
    StridedOpNodeInfo(const TensorVector&       nodeInputs,
                      const TensorVector&       nodeOutputs,
                      const synStridedOpParams& nodeParams,
                      std::string_view          nodeName)
    : output(nodeOutputs[0]), params(nodeParams), name(nodeName)
    {
        inputs[0] = nodeInputs[0];
        if (nodeInputs.size() > 1)
        {
            inputs[1] = nodeInputs[1];
            numInputs = 2;
        }
    }

    // we only deal with static shape compilation for non LAZY mode.
    // So strided_view will have a single input operand and strided insert will have two.
    // In case we have additional shape tensors for graph mode compilation we exclude them
    // from the remaining strided view after the decoding.
    // Strided view and strided insert fill the parameters based on the shape tensors as part of node creation so
    // dropping them for the static graph mode compilation should be o.k.
    static constexpr unsigned             MAX_NUM_INPUTS = 2;
    std::array<TensorPtr, MAX_NUM_INPUTS> inputs         = {};
    TensorPtr                             output;
    synStridedOpParams                    params    = {};
    unsigned                              numInputs = 1;
    std::string_view                      name;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedNodeSubOpDecoder
///////////////////////////////////////////////////////////////////////////////////////////////////

void StridedNodeSubOpDecoder::fixupStridedViewZeroStrides()
{
    const TensorPtr& output      = m_nodeInfo.output;
    NSizeArray       outputSizes = output->getAllNSizesInElements();
    unsigned         dims        = output->getDim();
    for (unsigned i = 0; i < dims; i++)
    {
        if (m_nodeInfo.params.strides[i] == 0 && outputSizes[i] == 1)
        {
            // get rid of 0 strides on trivial dimensions
            m_nodeInfo.params.strides[i] = (i == 0) ? 1 : m_nodeInfo.params.strides[i - 1];
        }
    }
}

bool StridedNodeSubOpDecoder::canDropStridedView() const
{
    if (m_node->getNodeType() != Node::TYPE_STRIDED_VIEW) return false;
    if (m_nodeInfo.params.baseOffset > 0) return false;
    const TensorPtr& input      = m_nodeInfo.inputs[0];
    const TensorPtr& output     = m_nodeInfo.output;
    unsigned         outputDims = output->getDim();
    unsigned         inputDims  = input->getDim();
    if (outputDims != inputDims) return false;
    const NSizeArray& outputSizes  = output->getAllNSizesInElements();
    const NSizeArray& inputSizes   = input->getAllNSizesInElements();
    NStrideArray      inputStrides = input->getNStridesInElements();
    for (int i = 0; i < inputDims; i++)
    {
        if (inputSizes[i] != outputSizes[i]) return false;
        if (inputSizes[i] > 1 && inputStrides[i] != m_nodeInfo.params.strides[i]) return false;
    }
    LOG_DEBUG(STRIDED_OP_DECODE, "dropped strided_view {} after decoding", m_nodeInfo.name);
    return true;
}

const TensorPtr& StridedNodeSubOpDecoder::getViewTensor() const
{
    return m_node->getNodeType() == Node::TYPE_STRIDED_VIEW ? m_nodeInfo.output
                                                            : m_nodeInfo.inputs[StridedInsertNode::INSERT_TENSOR];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpBroadcastDecoder
///////////////////////////////////////////////////////////////////////////////////////////////////

bool StridedOpBroadcastDecoder::canExtract() const
{
    if (m_node->getNodeType() != Node::TYPE_STRIDED_VIEW) return false;
    const TensorPtr&  output      = m_nodeInfo.output;
    const NSizeArray& outputSizes = output->getAllNSizesInElements();
    unsigned          dims        = output->getDim();
    for (unsigned i = 0; i < dims; i++)
    {
        if (m_nodeInfo.params.strides[i] == 0 && outputSizes[i] > 1) return true;
    }
    return false;
}

void StridedOpBroadcastDecoder::updateStridedOpNodeInfoParams(const TensorPtr& broadcastInput)
{
    // update new strided view node output to reference the broadcast input
    m_nodeInfo.output = broadcastInput;
    // update strides with a value of 0 in case we also had a strided view encoding broadcast.
    // we can't fix the strides in case those are permuted as the 0s need to be replaced with
    // the previous stride, so in this case we postpone the adjustment to StridedOpTransposeDecoder.
    if (!StridedOpTransposeDecoder(m_node, m_nodeInfo).canExtract())
    {
        fixupStridedViewZeroStrides();
    }
}

std::pair<NodePtr, bool> StridedOpBroadcastDecoder::extract()
{
    const TensorPtr& output              = m_nodeInfo.output;
    NSizeArray       broadcastInputSizes = output->getAllNSizesInElements();
    unsigned         dims                = output->getDim();
    for (unsigned i = 0; i < dims; i++)
    {
        if (m_nodeInfo.params.strides[i] == 0 && broadcastInputSizes[i] > 1)
        {
            // update boradcast dim on input to trivial dim
            broadcastInputSizes[i] = 1;
        }
    }

    TensorPtr broadcastInput = output->clone(false /*copyAddresses*/, false /*copyData*/);
    broadcastInput->reshape(dims, broadcastInputSizes.data());

    NodePtr broadcast = NodeFactory::createInternalNode({broadcastInput},
                                                        {output},
                                                        nullptr,
                                                        NodeFactory::broadcastNodeTypeName,
                                                        fmt::format("decoded_broadcast_{}", m_nodeInfo.name));

    updateStridedOpNodeInfoParams(broadcastInput);

    LOG_DEBUG(STRIDED_OP_DECODE,
              "decoded broadcast {} from strided_view node {}",
              broadcast->getNodeName(),
              m_nodeInfo.name);

    bool shouldDropStridedView = canDropStridedView();
    if (shouldDropStridedView)
    {
        broadcast->replaceInput(0, m_nodeInfo.inputs[0]);
    }

    return {broadcast, shouldDropStridedView};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpTransposeDecoder
///////////////////////////////////////////////////////////////////////////////////////////////////

bool StridedOpTransposeDecoder::canExtract() const
{
    // TODO (SW-161081)
    // transpose engine does not support all data types and may require
    // wrap with casts, which need to go through cguid for ndim, so we
    // currently do not support ndim extraction.
    // For strided view the transpose is for the output.
    // Currently blocked by SW-161082.
    const TensorPtr& t = getViewTensor();
    if (t->getDim() > gcapi::MAX_TENSOR_DIM) return false;

    const NSizeArray&      sizes = t->getAllNSizesInElements();
    unsigned               dims  = t->getDim();
    std::optional<TStride> prevStride;
    for (unsigned i = 0; i < dims; i++)
    {
        if (sizes[i] == 1) continue;
        if (prevStride.has_value() && m_nodeInfo.params.strides[i] < *prevStride) return true;
        prevStride = m_nodeInfo.params.strides[i];
    }
    return false;
}

NodePtr StridedOpTransposeDecoder::createTransposeNode(const TensorPtr&          origTensor,
                                                       const TensorPtr&          newTensor,
                                                       const StrideAndDimVector& strideAndDimVec,
                                                       bool                      isStridedView) const
{
    TensorVector            transposeInputs;
    TensorVector            transposeOutputs;
    synTransposeParamsNDims transposeParams = {};
    unsigned                dims            = origTensor->getDim();
    transposeParams.tensorDim               = dims;
    if (isStridedView)
    {
        for (unsigned i = 0; i < dims; i++)
        {
            transposeParams.permutation[strideAndDimVec[i].second] = i;
        }
        transposeInputs.push_back(newTensor);
        transposeOutputs.push_back(origTensor);
        m_nodeInfo.output = newTensor;
    }
    else
    {
        for (unsigned i = 0; i < dims; i++)
        {
            transposeParams.permutation[i] = strideAndDimVec[i].second;
        }
        transposeInputs.push_back(origTensor);
        transposeOutputs.push_back(newTensor);
        m_nodeInfo.inputs[StridedInsertNode::INSERT_TENSOR] = newTensor;
    }
    return NodeFactory::createInternalNode(transposeInputs,
                                           transposeOutputs,
                                           &transposeParams,
                                           NodeFactory::transposeNodeTypeName,
                                           fmt::format("decoded_transpose_{}", m_nodeInfo.name));
}

void StridedOpTransposeDecoder::updateStridedOpNodeInfoParams(const TensorPtr&                 newTensor,
                                                              const TransposePermutationArray& inversePermutationArray,
                                                              bool                             isStridedView)
{
    unsigned dims = newTensor->getDim();
    if (isStridedView)
    {
        m_nodeInfo.output = newTensor;
    }
    else
    {
        m_nodeInfo.inputs[StridedInsertNode::INSERT_TENSOR] = newTensor;
    }
    NStrideArray origStrides = {};
    std::copy_n(std::begin(m_nodeInfo.params.strides), dims, origStrides.begin());
    applyPermutation(origStrides.data(), inversePermutationArray, m_nodeInfo.params.strides);

    // update strides with a value of 0 in case we also had a strided view encoding broadcast
    if (isStridedView)
    {
        fixupStridedViewZeroStrides();
    }
}

std::pair<NodePtr, bool> StridedOpTransposeDecoder::extract()
{
    bool             isStridedView = m_node->getNodeType() == Node::TYPE_STRIDED_VIEW;
    const TensorPtr& origTensor    = getViewTensor();

    StrideAndDimVector strideAndDimVec;
    const NSizeArray&  sizes = origTensor->getAllNSizesInElements();
    unsigned           dims  = origTensor->getDim();
    for (unsigned i = 0; i < dims; i++)
    {
        strideAndDimVec.emplace_back(m_nodeInfo.params.strides[i], i);
    }
    std::stable_sort(strideAndDimVec.begin(), strideAndDimVec.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    TransposePermutationArray inversePermutationArray(dims);
    for (unsigned i = 0; i < dims; i++)
    {
        inversePermutationArray[i] = static_cast<TransposePermutationDim>(strideAndDimVec[i].second);
    }

    TensorPtr newTensor = getTensorAfterTranspose(*origTensor, inversePermutationArray);
    NodePtr   transpose = createTransposeNode(origTensor, newTensor, strideAndDimVec, isStridedView);
    updateStridedOpNodeInfoParams(newTensor, inversePermutationArray, isStridedView);

    LOG_DEBUG(STRIDED_OP_DECODE,
              "decoded transpose {} from strided_view node {}",
              transpose->getNodeName(),
              m_nodeInfo.name);

    bool shouldDropStridedView = canDropStridedView();
    if (shouldDropStridedView)
    {
        transpose->replaceInput(0, m_nodeInfo.inputs[0]);
    }

    return {transpose, shouldDropStridedView};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpReshapeDecoder
///////////////////////////////////////////////////////////////////////////////////////////////////

const TensorPtr& StridedOpReshapeDecoder::getReshapeInputTensor() const
{
    return m_node->getNodeType() == Node::TYPE_STRIDED_VIEW ? m_nodeInfo.inputs[0]
                                                            : m_nodeInfo.inputs[StridedInsertNode::INSERT_TENSOR];
}

bool StridedOpReshapeDecoder::canExtract() const
{
    if (m_nodeInfo.params.baseOffset != 0) return false;
    const TensorPtr& stridedOpInput = getReshapeInputTensor();
    const TensorPtr& outputTensor   = m_nodeInfo.output;
    if (stridedOpInput->getDenseSizeInElements() != outputTensor->getDenseSizeInElements())
    {
        return false;
    }
    return StridedOpUtils::isDenseStridedOpParams(m_nodeInfo.params, stridedOpInput);
}

std::pair<NodePtr, bool> StridedOpReshapeDecoder::extract()
{
    const TensorPtr& stridedOpInput = getReshapeInputTensor();
    NodePtr          reshape        = NodeFactory::createInternalNode({stridedOpInput},
                                                      {m_nodeInfo.output},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      fmt::format("decoded_reshape_{}", m_nodeInfo.name));

    LOG_DEBUG(STRIDED_OP_DECODE,
              "replaced strided node {} after decoding with reshape {}",
              m_nodeInfo.name,
              reshape->getNodeName());
    return {reshape, true};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpDecoderStrategies
///////////////////////////////////////////////////////////////////////////////////////////////////

StridedOpDecoderStrategies::StridedOpDecoderStrategies(const NodePtr& node, StridedOpNodeInfo& nodeInfo)
: m_broadcastDecoder(node, nodeInfo),
  m_transposeDecoder(node, nodeInfo),
  m_reshapeDecoder(node, nodeInfo),
  m_subOpDecoders {&m_broadcastDecoder, &m_transposeDecoder, &m_reshapeDecoder}
{
}

inline const std::array<StridedNodeSubOpDecoder*, StridedOpDecoderStrategies::DECODER_STRATEGY_COUNT>&
StridedOpDecoderStrategies::getDecoderStrategies() const
{
    // ordering matters. We wish to decode broadcast first to fix 0 strides with corresponding non trivial output dim.
    // then decode permutations and finally reshape. Changing the ordering can result in failure to decode some of the
    // sub operations.
    return m_subOpDecoders;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// StridedOpDecoder
///////////////////////////////////////////////////////////////////////////////////////////////////

bool StridedOpDecoder::canExtract(const NodePtr& node)
{
    if (!GCFG_ENABLE_STRIDED_OP_DECODING.value()) return false;
    if (node->getNodeType() != Node::TYPE_STRIDED_VIEW && node->getNodeType() != Node::TYPE_STRIDED_INSERT)
    {
        return false;
    }
    return !node->isDynamicShape();
}

const synStridedOpParams* StridedOpDecoder::getStridedOpParams(const NodePtr& node)
{
    switch (node->getNodeType())
    {
        case Node::TYPE_STRIDED_VIEW:
            return &static_cast<StridedViewNode*>(node.get())->getParams();
        case Node::TYPE_STRIDED_INSERT:
            return &static_cast<StridedInsertNode*>(node.get())->getParams();
        default:
            HB_ASSERT(false, "StridedOpDecoder extract called for non supported node type");
            return nullptr;
    }
}

NodeVector StridedOpDecoder::extract(const NodePtr& node, bool changeInPlace)
{
    const synStridedOpParams* params = getStridedOpParams(node);
    StridedOpNodeInfo         nodeInfo {node->getInputs(), node->getOutputs(), *params, node->getNodeName()};

    NodeVector                 extractedNodes;
    StridedOpDecoderStrategies strategies(node, nodeInfo);
    for (StridedNodeSubOpDecoder* subOpDecoder : strategies.getDecoderStrategies())
    {
        if (subOpDecoder->canExtract())
        {
            auto [subOp, dropOriginalStridedView] = subOpDecoder->extract();
            extractedNodes.push_back(std::move(subOp));
            if (dropOriginalStridedView) return extractedNodes;
        }
    }
    if (!extractedNodes.empty())
    {
        // if we got here it means we still need to add the strided op
        NodePtr remainingStridedOp = node;
        if (!changeInPlace)
        {
            remainingStridedOp = node->clone();
            remainingStridedOp->removeDataInputsFromIndex(nodeInfo.numInputs);
        }
        remainingStridedOp->replaceInput(0, nodeInfo.inputs[0]);
        if (remainingStridedOp->getNumInputs() > 1)
        {
            remainingStridedOp->replaceInput(1, nodeInfo.inputs[1]);
        }
        remainingStridedOp->replaceOutput(0, nodeInfo.output);
        remainingStridedOp->setParams(&nodeInfo.params, sizeof(nodeInfo.params));
        extractedNodes.push_back(std::move(remainingStridedOp));
    }
    // if we get here with an empty extractedNodes it means no decoding took place
    return extractedNodes;
}