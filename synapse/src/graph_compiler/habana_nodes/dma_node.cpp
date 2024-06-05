#include "dma_node.h"

#include "habana_graph.h"
#include "hal_reader/hal_reader.h"

DMANode::DMANode(const TensorVector& input,
                 const TensorVector& output,
                 std::string_view    name,
                 DMA_TYPE            dmaType,
                 ShapeFuncID         sifId)
: Node(input, output, name, Node::TYPE_DMA, sifId), m_dmaType(dmaType)
{
}

DMANode::DMANode(const TensorPtr& t, std::string_view name, DMA_TYPE dmaType, ShapeFuncID sifId)
: DMANode(isTensorInputForDMANode(dmaType) ? TensorVector({t}) : TensorVector(),
          isTensorInputForDMANode(dmaType) ? TensorVector() : TensorVector({t}),
          name,
          dmaType,
          sifId)
{
}

DMANode::DMANode(const TensorPtr& input,
                 const TensorPtr& output,
                 std::string_view name,
                 DMA_TYPE         dmaType,
                 ShapeFuncID      sifId)
: DMANode(TensorVector({input}), TensorVector({output}), name, dmaType, sifId)
{
}

HabanaDeviceType DMANode::getNodeDeviceType() const
{
    switch (m_dmaType)
    {
        case DMA_TYPE_UPSTREAM:
        case DMA_TYPE_INTERMEDIATES:
            return DEVICE_DMA_DEVICE_HOST;

        case DMA_TYPE_DOWNSTREAM:
            return DEVICE_DMA_HOST_DEVICE;

        case DMA_TYPE_INTERNAL:
        case DMA_TYPE_PREFETCH_STATIC_TENSORS:
            return DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL;

        default:
            HB_ASSERT(false, "Unsupported DMA node type");
            return LAST_HABANA_DEVICE;
    }
}

bool DMANode::validateNode() const
{
    const auto& nodeName = getNodeName();
    SET_TEMP_LOG_CONTEXT(nodeName);

    const TensorPtr& src = getInput(TENSOR_IFM);
    const TensorPtr& dst = getOutput(TENSOR_OFM);

    switch (m_dmaType)
    {
        case DMA_TYPE_DOWNSTREAM:
        case DMA_TYPE_PREFETCH_STATIC_TENSORS:
            if (src)
            {
                LOG_ERR(HABANA_NODE, "dma type expects only one tensor while two are passed");
                return false;
            }
            break;
        case DMA_TYPE_UPSTREAM:
        case DMA_TYPE_INTERMEDIATES:
            if (dst)
            {
                LOG_ERR(HABANA_NODE, "dma type expects only one tensor while two are passed");
                return false;
            }
            break;
        case DMA_TYPE_PREFETCH_ACTIVATIONS:
        case DMA_TYPE_SPILL:
        case DMA_TYPE_INTERNAL:
            if (isMemset() && src && !src->isShapeTensor())
            {
                LOG_ERR(HABANA_NODE, "DMA_TYPE_INTERNAL Memset cannot be created with non-shape input tensor");
                return false;
            }
            if (!dst)
            {
                LOG_ERR(HABANA_NODE, "DMA_TYPE_INTERNAL destination not provided");
                return false;
            }
            if (!isMemset() && !src)
            {
                LOG_ERR(HABANA_NODE, "DMA_TYPE_INTERNAL memcpy source not found");
                return false;
            }
            break;
        default:
            LOG_ERR(HABANA_NODE, "DMA type not supported");
            return false;
            break;
    }
    if (m_dmaType == DMA_TYPE_INVALID)
    {
        return false;
    }

    if (src && !src->isShapeTensor() && dst)
    {
        bool isTransposeNode = isTranspose();
        if (!isTransposeNode && !isBroadcast())
        {
            CHECK_RET_FALSE(src->compareGeometry(*dst), "{}: geometry of src and dst differs", nodeName);
        }
        else if (isTransposeNode)
        {
            CHECK_RET_FALSE(src->getDenseSizeInElements() == dst->getDenseSizeInElements(),
                            "{}: src and dst element count differ",
                            nodeName);
        }
        CHECK_RET_FALSE(src->getElementSizeInBits() == dst->getElementSizeInBits(),
                        "{}: src and dst element size differ",
                        nodeName);
    }

    return Node::validateNode();
}

bool DMANode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (isLinearDma())
    {
        return true;
    }
    return g.getHALReader()->isNonLinearDmaSupported();
}

NodePtr DMANode::clone() const
{
    return NodePtr(new DMANode(*this));
}

bool DMANode::isROIDynamic(const NodeROI* roi) const
{
    // All Rois are executed for Memset
    if (isMemset()) return false;
    else
    {
        return BaseClass::isROIDynamic(roi);
    }
}

void DMANode::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;

    Node::print();

    switch (m_dmaType)
    {
        case DMA_TYPE_UPSTREAM:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_inputs.empty() ? getInput(TENSOR_IFM)->getTensorOffset() : 0);
            LOG_DEBUG(GRAPH_DATA, "      To:   {}", !m_inputs.empty() ? getInput(TENSOR_IFM)->getAddress() : "");
            break;
        case DMA_TYPE_DOWNSTREAM:
            LOG_DEBUG(GRAPH_DATA, "      From: {}", !m_outputs.empty() ? getOutput(TENSOR_OFM)->getAddress() : "");
            LOG_DEBUG(GRAPH_DATA,
                      "      To: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getTensorOffset() : 0);
            break;
        case DMA_TYPE_INTERMEDIATES:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_inputs.empty() ? getInput(TENSOR_IFM)->getTensorOffset() : 0);
            LOG_DEBUG(GRAPH_DATA, "      To:   {}", !m_inputs.empty() ? getInput(TENSOR_IFM)->getAddress() : "");
            break;
        case DMA_TYPE_PREFETCH_STATIC_TENSORS:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getDramOffset() : 0);
            LOG_DEBUG(GRAPH_DATA,
                      "      To: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getTensorOffset() : 0);
            break;
        case DMA_TYPE_SPILL:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_inputs.empty() ? getInput(TENSOR_IFM)->getTensorOffset() : 0);
            LOG_DEBUG(GRAPH_DATA,
                      "      To: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getTensorOffset() : 0);
            break;
        case DMA_TYPE_PREFETCH_ACTIVATIONS:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_inputs.empty() ? getInput(TENSOR_IFM)->getTensorOffset() : 0);
            LOG_DEBUG(GRAPH_DATA,
                      "      To: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getTensorOffset() : 0);
            break;
        case DMA_TYPE_INTERNAL:
            LOG_DEBUG(GRAPH_DATA,
                      "      From: 0x{:x}",
                      !m_inputs.empty() ? getInput(TENSOR_IFM)->getTensorOffset() : 0);
            LOG_DEBUG(GRAPH_DATA,
                      "      To: 0x{:x}",
                      !m_outputs.empty() ? getOutput(TENSOR_OFM)->getTensorOffset() : 0);
            break;
        default:
            LOG_DEBUG(GRAPH_DATA, "");
            break;
    }
}

bool DMANode::isLinearDma() const
{
    TensorPtr src = !m_inputs.empty() ? getInput(TENSOR_IFM) : nullptr;
    TensorPtr dst = !m_outputs.empty() ? getOutput(TENSOR_OFM) : nullptr;

    if (isTranspose())
    {
        return false;
    }

    if (getDynamicMemoryOpType() == DMA_OP_DYNAMIC_STRIDE)
    {
        return false;
    }

    if (src && !src->isDenseLayout())
    {
        return false;
    }
    if (dst && !dst->isDenseLayout())
    {
        return false;
    }
    return !isDynamicShape();
}

bool DMANode::isTensorTwoDimensionalStrided(TensorPtr t) const
{
    unsigned size1   = t->getSizeInElements(1);
    uint64_t stride0 = t->getStrideInElements(1);

    if (t->getSizeInBytes(0) >= chunkSizeInBytes())
    {
        LOG_INFO(HABANA_NODE, "{} {}: dim 0 size is larger than chunk size", HLLOG_FUNC, getNodeName());
        return false;
    }

    if (t->getStrideInElements(2) != stride0 * size1)
    {
        LOG_INFO(HABANA_NODE, "{} {}: dim 1 is strided.", HLLOG_FUNC, getNodeName());
        return false;
    }

    for (unsigned dim = 2; dim < Tensor::c_tensorMaxDim; ++dim)
    {
        if (t->getSizeInElements(dim) != 1)
        {
            LOG_INFO(HABANA_NODE, "{} {}: dim {} is not 1.", HLLOG_FUNC, getNodeName(), dim);
            return false;
        }
        if (t->getStrideInElements(dim + 1) != stride0 * size1)
        {
            LOG_INFO(HABANA_NODE, "{} {}: there is a stride at dim {}", HLLOG_FUNC, getNodeName(), dim);
            return false;
        }
    }
    return true;
}

bool DMANode::isNodeTwoDimensionalStrided() const
{
    TensorPtr src = !m_inputs.empty() ? getInput(TENSOR_IFM) : nullptr;
    TensorPtr dst = !m_outputs.empty() ? getOutput(TENSOR_OFM) : nullptr;

    if (getDynamicMemoryOpType() == DMA_OP_DYNAMIC_STRIDE)
    {
        return false;
    }
    if (src && !isTensorTwoDimensionalStrided(src))
    {
        return false;
    }
    if (dst && !isTensorTwoDimensionalStrided(dst))
    {
        return false;
    }
    return true;
}

DimVector DMANode::getSplitDimensionsOrder()
{
    // Unless someone finds something better- split according to dimensions order. 0, 1, 2....
    unsigned splitDims = 0;
    if (getOutputs().size())
    {
        splitDims = getOutput(0)->getDim();
    }
    else
    {
        LOG_ERR(HABANA_NODE, "Split DMA- invalid node with no outputs");
        HB_ASSERT(false, "Split DMA- invalid node with no outputs");
    }
    // for DMA we split all dims in order
    DimVector splitDimsOrder;
    for (unsigned i = 0; i < splitDims; i++)
    {
        splitDimsOrder.push_back(splitDims - 1 - i);
    }
    return splitDimsOrder;
}

NodeROI DMANode::generateRoi() const
{
    NodeROI   fullRoi;
    TensorPtr t = !m_inputs.empty() ? getInput(TENSOR_IFM) : getOutput(TENSOR_OFM);

    if (isLinearDma())
    {
        fullRoi.size[0] = BITS_PER_BYTE * t->getTotalSizeInBytes() / t->getElementSizeInBits();
        std::fill(fullRoi.size + 1, fullRoi.size + ARRAY_SIZE(fullRoi.size), 1);
    }
    else if (!isMemset() && !isTranspose() && !isBroadcast() && !isDynamicShape())
    {
        // in case of strided memcpy with static shape we try to increase the size of the FCD
        // by aggregating the first dense dims
        generateRoiForStridedMemcpy(fullRoi);
    }
    else
    {
        // This is the general case for a strided DMA.
        // For now set ROI as full. to split to at least 16KB chunks we need to modify it here.
        t->getAllSizesInElements(fullRoi.size, ARRAY_SIZE(fullRoi.size));
    }
    return fullRoi;
}

bool DMANode::isMemset() const
{
    return false;
}

DMA_OP_TYPE DMANode::getOpType() const
{
    return DMA_OP_TYPE::DMA_OP_COPY;
}

uint64_t DMANode::parallelLevel() const
{
    return m_parallelLevel;
}

void DMANode::setParallelLevel(uint64_t pLevel)
{
    m_parallelLevel = pLevel;
}

uint64_t DMANode::dispatcherIndex() const
{
    return m_dispatcherIndex;
}

void DMANode::setDispatcherIndex(uint64_t pLevel)
{
    m_dispatcherIndex = pLevel;
}

uint64_t DMANode::chunkSizeInBytes() const
{
    return GCFG_DMA_CHUNK_SIZE.value() * (parallelLevel() > 1 ? 1 : 2);
}

std::string_view DMANode::getEngineTypeStr() const
{
    return "DMA";
}

// for strided memcpy that is not dynamic shape, we want to aggregate all first dense dims
// to increase the utilization of the memcpy, Example (bf16):
// the utilization of memcpy with sizes [2, 32, X], and strides [2, 4, 200, 200 * X] is 4/128 = 0.03125
// but if we aggregate the first 2 dims we get a memcpy
// with sizes [64, X], and strides [2, 200, 200 * X] and the utilization is 128/128 = 1
void DMANode::generateRoiForStridedMemcpy(NodeROI& roi) const
{
    const TensorPtr& input  = m_inputs[0];
    const TensorPtr& output = m_outputs[0];

    uint64_t newFcdSizeInElements = 1;  // the aggregated FCD size
    unsigned tensorDim            = input->getDim();

    unsigned dim = 0;
    // aggregated the first dims that have trivial strides (dense in memory), both in the input and output
    for (; dim < tensorDim; ++dim)
    {
        newFcdSizeInElements *= input->getSizeInElements(dim);
        if (newFcdSizeInElements != input->getStrideInElements(dim + 1) ||
            newFcdSizeInElements != output->getStrideInElements(dim + 1))
            break;
    }

    // add additional data to the roi, needed for updating the strides in the tensors roi's
    roi.additionalData = std::make_shared<unsigned>(std::move(dim));

    // set the roi size of the fcd to be the aggregated size
    roi.size[0] = newFcdSizeInElements;

    // update the rest of the sizes to be like in the original sizes after the aggregated FCD
    for (unsigned i = dim + 1; i < tensorDim; ++i)
    {
        roi.size[i - dim] = input->getSizeInElements(i);
    }

    // set 1 to the rest of the dims
    std::fill(roi.size + (tensorDim - dim), roi.size + Tensor::c_tensorMaxNDim, 1);
}