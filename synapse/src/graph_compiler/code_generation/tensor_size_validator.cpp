#include "tensor_size_validator.h"
#include "synapse_common_types.h"
#include "tpc_node.h"

bool TensorSizeValidator::validateTensors(const HabanaGraph& graph) const
{
    if (GCFG_DISABLE_TENSOR_SIZE_VALIDATION.value())
    {
        LOG_DEBUG(GC, "CodeGen tensor validation - disabled by GCFG");
        return true;
    }

    LOG_DEBUG(GC, "CodeGen validating tensors");
    for (const auto& node : graph.getNodes())
    {
        // no need to validate since it doesn't get to the HW
        if (node->isLogicalOperation())
        {
            continue;
        }
        std::list<NodeROI>& rois = graph.getCodeGenerator()->getPhysicalRois(node);

        if (graph.runsOnTPC(node))
        {
            // when GC support program TPC in IRF44 mode as per SW-118395 need a change here to activate the IRF44 validation
            LOG_TRACE(GC, " validating TPC Node");
            const TPCNode* tpcNode = dynamic_cast<const TPCNode*>(node.get());
            HB_ASSERT(tpcNode != nullptr, "invalid node type");

            const bool isIRF44 = tpcNode->is44bitMode();

            if (!validateTPCNodeTensors(node->getInputs(), isIRF44)) return false;
            if (!validateTPCNodeTensors(node->getOutputs(), isIRF44)) return false;
        }
        else if (node->isDma())
        {
            LOG_TRACE(GC, " validating DMA Node");
            const uint64_t maxRegVal = m_halReader->getMaxRegValForDma();
            for (const NodeROI& roi : rois)
            {
                if (!validateDMANodeTensors(roi.inputRois, maxRegVal)) return false;
                if (!validateDMANodeTensors(roi.outputRois, maxRegVal)) return false;
            }
        }
        else
        {
            LOG_TRACE(GC, " validating MME Node");
            if (!validateMMENodeTensors(node->getInputs())) return false;
            if (!validateMMENodeTensors(node->getOutputs())) return false;
        }
    }
    LOG_DEBUG(GC, "CodeGen validating tensors passed ");
    return true;
}

bool TensorSizeValidator::validateTensor(const TensorPtr&    tensor,
                                         const NSizeArray&   sizes,
                                         const NStrideArray& strides,
                                         HabanaDeviceType    engineType,
                                         bool                isIRF44 /* relevant only for tpc engine type */) const
{
    if (tensor->isZeroSizedDataTensor() || tensor->isShapeTensor()) return true;
    if (engineType == HabanaDeviceType::DEVICE_TPC)
    {
        if (m_halReader->isAsicSupportIRF44Mode())
        {
            return validateTensorIRF44ModeForTPC(tensor, sizes.data(), strides.data(), isIRF44);
        }
        else
        {
            return validateTensorForTPC(tensor, sizes.data(), strides.data());
        }
    }
    else if (engineType == HabanaDeviceType::DEVICE_MME)
    {
        const uint64_t maxRegVal = m_halReader->getMaxRegValForMME(tensor->getElementSizeInBytes());
        return validatePerSizeAndStrideForMME(tensor, sizes.data(), strides.data(), maxRegVal);
    }
    else if (engineType == HabanaDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL)
    {
        const uint64_t maxRegVal = m_halReader->getMaxRegValForDma();
        return validatePerSizeAndStrideForDMA(tensor, sizes, strides.data(), maxRegVal);
    }
    else
    {
        return true;
    }
}

bool TensorSizeValidator::validateTensor(const NodePtr&      node,
                                         const TensorPtr&    tensor,
                                         const NSizeArray&   sizes,
                                         const NStrideArray& strides,
                                         HabanaDeviceType    engineType) const
{
    bool isIRF44 = false;
    if (engineType == HabanaDeviceType::LAST_HABANA_DEVICE)
    {
        engineType = node->getNodeDeviceType();
    }
    if (engineType == HabanaDeviceType::DEVICE_TPC)
    {
        const TPCNode* tpcNode = dynamic_cast<const TPCNode*>(node.get());

        // If we can't cast node to tpcNode, assume it's not in IRF44 mode
        if (tpcNode != nullptr)
        {
            isIRF44 = tpcNode->is44bitMode();
        }
    }
    return validateTensor(tensor, sizes, strides, engineType, isIRF44);
}

bool TensorSizeValidator::validateDMANodeTensors(const TensorROIVector& tensorRois, const uint64_t maxRegVal) const
{
    bool res = true;
    for (const auto& tensorRoi : tensorRois)
    {
        const TensorROILayout& layout = tensorRoi.getLayout();
        res &= validatePerSizeAndStrideForDMA(tensorRoi.m_parentTensor,
                                              tensorRoi.getLayout().m_size,
                                              tensorRoi.getStridesWithFcdDim().data(),
                                              maxRegVal);
    }
    return res;
}

bool TensorSizeValidator::validateMMENodeTensors(const TensorVector& tensors) const
{
    bool res = true;
    for (const auto& tensor : tensors)
    {
        if (tensor == nullptr || tensor->isShapeTensor() || tensor->isControlEdge())
        {
            continue;
        }
        const uint64_t maxRegVal = m_halReader->getMaxRegValForMME(tensor->getElementSizeInBytes());
        res &= validatePerSizeAndStrideForMME(tensor, maxRegVal);
    }
    return res;
}

bool TensorSizeValidator::validateTPCNodeTensors(const TensorVector& tensors, bool isIRF44) const
{
    bool res = true;
    for (const auto& tensor : tensors)
    {
        if (tensor == nullptr || tensor->isShapeTensor() || tensor->isControlEdge())
        {
            continue;
        }
        if (m_halReader->isAsicSupportIRF44Mode())
        {
            res &= validateTensorIRF44ModeForTPC(tensor, isIRF44);
        }
        else
        {
            res &= validateTensorForTPC(tensor);
        }
    }
    return res;
}

bool TensorSizeValidator::validatePerSizeAndStrideForMME(const TensorPtr& tensor,
                                                         const TSize*     sizes,
                                                         const TStride*   strides,
                                                         const uint64_t   maxRegVal) const
{
    bool isDegeneratedDim = true;
    for (int dim = tensor->getDim() - 1; dim >= 0; --dim)
    {
        const uint64_t offset = sizes[dim] * tensor->calcStrideInElements(strides[dim]);
        isDegeneratedDim      = isDegeneratedDim && sizes[dim] == 1;
        if (offset > maxRegVal && !isDegeneratedDim)
        {
            logTensorSizeOverflow(tensor, dim, offset);
            return false;
        }
    }
    return true;
}

bool TensorSizeValidator::validatePerSizeAndStrideForMME(const TensorPtr& tensor, const uint64_t maxRegVal) const
{
    return validatePerSizeAndStrideForMME(tensor,
                                          tensor->getAllNSizesInElements().data(),
                                          tensor->getNStridesInBytes(),
                                          maxRegVal);
}

bool TensorSizeValidator::validatePerSizeAndStrideForDMA(const TensorPtr&  tensor,
                                                         const NSizeArray& sizes,
                                                         const TStride*    strides,
                                                         const uint64_t    maxRegVal) const
{
    for (unsigned dim = 0; dim < tensor->getDim(); ++dim)
    {
        const uint64_t offset = (sizes[dim] - 1) * strides[dim];
        if (offset > maxRegVal)
        {
            logTensorSizeOverflow(tensor, sizes[dim], strides[dim], dim, offset);
            return false;
        }
    }
    return true;
}

bool TensorSizeValidator::validateTensorIRF44ModeForTPC(const TensorPtr& tensor,
                                                        const bool       supportIRF44 /*= false*/) const
{
    return validateTensorIRF44ModeForTPC(tensor,
                                         tensor->getAllNSizesInElements().data(),
                                         tensor->getNStridesInBytes(),
                                         supportIRF44);
}

bool TensorSizeValidator::validateTensorIRF44ModeForTPC(const TensorPtr& tensor,
                                                        const TStride*   sizes,
                                                        const TStride*   strides,
                                                        const bool       supportIRF44 /*= false*/) const
{
    const unsigned tensorDim   = tensor->getDim();
    const unsigned extendedDim = tensor->getIndexOfMaxNonDegenerateStride();
    const unsigned is64BitElementSize = tensor->is64BitElementSize() ? 2 : 1; // single 64b elements will be treated as 2 32b elements

    const uint64_t maxRegValForCordXStride =
        supportIRF44 ? m_halReader->getMaxRegValForSigned44BitMode() : m_halReader->getDefaultMaxRegVal();
    const uint64_t maxRegValForExtendedDimTPC = m_halReader->getMaxRegValForExtendedDimInTPC();
    const uint64_t maxRegValForAfterTheShift  = m_halReader->getMaxRegValForSigned44BitMode();
    const uint64_t logElementSizeInBytes      = log2(tensor->getElementSizeInBytes()/is64BitElementSize);
    bool           tensorExceedsRegister      = false;
    for (unsigned dim = 0; dim < tensorDim; ++dim)
    {
        TSize size = sizes[dim];
        size       = (size == 0) ? 0 : size - 1;  // max coordinate in TPC in AGU is size-1 (except ZST)

        const uint64_t offset = size * tensor->calcStrideInElements(strides[dim]) * is64BitElementSize;
        // uses the bigger restriction
        if (dim == 1 || dim == extendedDim)
        {
            const bool shiftedOffsetBiggerRestriction = (offset << logElementSizeInBytes) > maxRegValForExtendedDimTPC;
            if (dim == 1 && dim != extendedDim)
            {
                tensorExceedsRegister = shiftedOffsetBiggerRestriction || ((!supportIRF44) && (offset > maxRegValForCordXStride));
            }
            if (dim == extendedDim)
            {
                tensorExceedsRegister = shiftedOffsetBiggerRestriction;
            }
        }
        else
        {
            const bool shiftedOffsetDefaultRestriction = (offset << logElementSizeInBytes) > maxRegValForAfterTheShift;
            if (dim == 0)
            {
                tensorExceedsRegister = shiftedOffsetDefaultRestriction;
            }
            else
            {
                tensorExceedsRegister = (shiftedOffsetDefaultRestriction || offset > maxRegValForCordXStride);
            }
        }

        if (tensorExceedsRegister)
        {
            logTensorSizeOverflow(tensor, dim, offset);
            return false;
        }
    }
    return true;
}

// gaudi1 and greco doens't support IRF44 Mode hence have this special implementation
bool TensorSizeValidator::validateTensorForTPC(const TensorPtr& tensor) const
{
    return validateTensorForTPC(tensor,
                                tensor->getAllNSizesInElements().data(),
                                tensor->getNStridesInElements().data());
}

bool TensorSizeValidator::validateTensorForTPC(const TensorPtr& tensor,
                                               const TSize*     sizes,
                                               const TStride*   strides) const
{
    const unsigned tensorDim   = tensor->getDim();
    const unsigned extendedDim = tensor->getIndexOfMaxNonDegenerateStride();
    const unsigned is64BitElementSize = tensor->is64BitElementSize() ? 2 : 1; // single 64b elements will be treated as 2 32b elements

    const uint64_t maxRegVal                   = m_halReader->getDefaultMaxRegVal();
    const uint64_t maxRegValForExtendedDimTPC  = m_halReader->getMaxRegValForExtendedDimInTPC();
    const uint64_t logElementSizeInBytes      = log2(tensor->getElementSizeInBytes()/is64BitElementSize);
    uint64_t       offsetLSum                  = 0;
    for (unsigned dim = 0; dim < tensorDim; ++dim)
    {
        const bool     isExtendedDim = (dim == extendedDim);
        TSize          size          = sizes[dim];
        size = (size == 0) ? 0 : size - 1;  // max coordinate in TPC in AGU is size-1 (except ZST)
        TStride        elementStride = tensor->calcStrideInElements(strides[dim]);
        const uint64_t offset        = (size * elementStride * is64BitElementSize) << logElementSizeInBytes;

        if (isExtendedDim && dim != 0)
        {
            if (offset > maxRegValForExtendedDimTPC)
            {
                logTensorSizeOverflow(tensor, dim, offset);
                return false;
            }
        }
        else
        {
            if (dim != 0) offsetLSum += offset;  //dim0 is clamped seperately from dims {1,2,3,4}-H
            if ((offset > maxRegVal) || (offsetLSum > maxRegVal))
            {
                logTensorSizeOverflow(tensor, dim, offset);
                return false;
            }
        }
    }
    return true;
}

void TensorSizeValidator::logTensorSizeOverflow(const TensorPtr& tensor, unsigned dim, const uint64_t offset) const
{
    logTensorSizeOverflow(tensor, tensor->getSizeInElements(dim), tensor->getStrideInElements(dim), dim, offset);
}

void TensorSizeValidator::logTensorSizeOverflow(const TensorPtr& tensor,
                                                TSize            size,
                                                TStride          stride,
                                                unsigned         dim,
                                                const uint64_t   offset) const
{
    if (!log_level_at_least(synapse::LogManager::LogType::GC, m_logLevel)) return;
    const auto errMsg = fmt::format("The tensor {} in dim {} size is {} and stride is {}. it's element "
                                    "size in bytes is {}, which leads "
                                    "to offset exceeds the tensor register",
                                    tensor->getName(),
                                    dim,
                                    size,
                                    stride,
                                    tensor->getElementSizeInBytes());
    SYN_LOG_TYPE(GC, m_logLevel, "{}", errMsg);
    tensor->debugPrint();
}
