#include "api.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "syn_singleton.hpp"
#include "data_type_utils.h"
#include "defenders.h"
#include "section_handle.hpp"

extern const uint32_t DUMMY_GRAPH_ID;
const uint32_t        DUMMY_GRAPH_ID = UINT32_MAX;

void synSingleton::_getTensorScaleZp(const synTensorDescriptor& pDescriptor, double& scale, double& zp)
{
    scale = 1;
    zp    = 0;

    for (synQuantizationParams params : pDescriptor.m_quantizationParams)
    {
        if (params.m_qDataType == pDescriptor.m_dataType)
        {
            scale = params.m_scale;
            zp    = params.m_zp;
        }
    }
}

/*
***************************************************************************************************
*   @brief Creates a Tensor memory object
*
*   @param pDescriptor  [in]   A descriptor to define the required tensor.
*   @param pStatus      [out]  An optional argument for return code.
*   @return                    A handle to the newly created tensor
***************************************************************************************************
*/
synTensor synSingleton::createTensor(synTensorDescriptor* pDescriptor,
                                     synStatus*           pStatus,
                                     bool                 isOutput, /*=false*/
                                     bool                 isInput,  /*=false*/
                                     bool                 isStaticParam)            /*=false*/
{
    if (pDescriptor == nullptr)
    {
        LOG_ERR(SYN_API, "{} was given nullptr tensor descriptor as input", HLLOG_FUNC);
        *pStatus = synInvalidArgument;
        return nullptr;
    }

    double scale = 1;
    double zp    = 0;

    for (synQuantizationParams params : pDescriptor->m_quantizationParams)
    {
        if (params.m_qDataType == pDescriptor->m_dataType)
        {
            scale = params.m_scale;
            zp    = params.m_zp;
        }
    }

    LOG_DEBUG(
        SYN_API,
        "synSingleton::createTensor {} ({}), Data type {}, dims {}, max sizes ({}) min sizes ({}){}, scale={}, zp={} "
        " type={} {}{}{}{}{}{}{}{}",
        (pDescriptor->m_name != nullptr) ? pDescriptor->m_name : "",
        pDescriptor->m_ptr,
        getStringFromSynDataType(pDescriptor->m_dataType),
        pDescriptor->m_dims,
        toString(pDescriptor->m_sizes, pDescriptor->m_sizes + pDescriptor->m_dims, ','),
        toString(pDescriptor->m_minSizes, pDescriptor->m_minSizes + pDescriptor->m_dims, ','),
        (pDescriptor->m_batchPos != INVALID_BATCH_POS) ? ", batchPos=" + std::to_string(pDescriptor->m_batchPos) : "",
        scale,
        zp,
        pDescriptor->m_tensorType,
        pDescriptor->m_isWeights ? "(weights)" : "",
        pDescriptor->m_isQuantized ? "(quantized)" : "",
        isStaticParam ? "(static_param)" : "",
        isInput ? "(input)" : "",
        isOutput ? "(output)" : "",
        pDescriptor->m_enablePerChannelQuant ? "(per channel enabled)" : "",
        pDescriptor->m_isSparsityWeights ? "(sparsity weights tensor)" : "",
        pDescriptor->m_dynamicRange.isSet ? ", drange=(" + std::to_string(pDescriptor->m_dynamicRange.min) + ", " +
                                                std::to_string(pDescriptor->m_dynamicRange.max) + ")"
                                          : "");

    *pStatus = synSuccess;

    uint32_t currGraphId = DUMMY_GRAPH_ID;  // TODO - need to find graphID or remove this API

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    // not in cache
    if (!m_graphEntries.doTensorExistByPtr(currGraphId, pDescriptor->m_ptr))
    {
        TensorPtr t = m_graphEntries.createTensor(pDescriptor, isOutput, isInput, pStatus, nullptr, currGraphId);
        if (*pStatus != synSuccess)
        {
            return nullptr;
        }
        if (pDescriptor->m_isWeights)
        {
            t->setAsWeights();
        }
        if (isStaticParam)
        {
            t->setAsStaticParam();
            t->setProp(synTensorPropHostPtr);
        }
        if (pDescriptor->m_isQuantized)
        {
            t->setAsDataTypeMatchData();
        }

        if (pDescriptor->m_isSparsityWeights)
        {
            t->setAsSparsityWeights();
        }
        if (pDescriptor->m_enablePerChannelQuant)
        {
            if (!t->isStaticParam())
            {
                LOG_WARN(SYN_API,
                         "{}, Cannot set per channel quant for {} tensor, Only static tensors supported",
                         HLLOG_FUNC,
                         t->getName());
                t->setPerChannelQuant(false);
            }
            else
            {
                t->setPerChannelQuant(true);
            }
        }

        synTensor synTen = reinterpret_cast<synTensor>(t.get());
        // Add tensor and its shared_ptr to the map
        m_graphEntries.setTensorSharedPtr(currGraphId, t);
        for (auto params : pDescriptor->m_quantizationParams)
        {
            if (params.m_qDataType != syn_type_na)
            {
                t->setQuantizationParams(params);
            }
        }

        if (pDescriptor->m_enablePerChannelQuant)
        {
            for (auto params : pDescriptor->m_perChannelQuantizationParams)
            {
                if (params.m_qDataType != syn_type_na)
                {
                    t->setQuantizationParams(params);
                }
            }
        }

        if (pDescriptor->m_dynamicRange.isSet && !t->setDynamicRange(pDescriptor->m_dynamicRange))
        {
            LOG_ERR(SYN_API,
                    "synSingleton::createTensor {}: Invalid Dynamic Range: min={}, max={} ",
                    (pDescriptor->m_name != nullptr) ? pDescriptor->m_name : "",
                    pDescriptor->m_dynamicRange.min,
                    pDescriptor->m_dynamicRange.max);
        }

        t->bind(pDescriptor->m_ptr, false);
        t->setProp(synTensorPropFpQuantMetadata);
        t->setProp(synTensorPropFlags);
        t->setProp(synTensorPropDynamicRange);
        t->setProp(synTensorPropGeometryMin);
        t->setProp(synTensorPropGeometryMax);
        t->setProp(synTensorPropGeometryDim);
        t->setProp(synTensorPropDeviceLayout);

        m_graphEntries.setTensorByPtr(currGraphId, pDescriptor->m_ptr, t);

        // The same buffer, which is the key for the above map, might be also of a Tensor in m_deletedTensors.
        // That is OK.
        // A Tensor in m_deletedTensors will not use its "data".
        // In addition upon Synapse destruction, we validate "data" buffer prior of deleting it (on Tensor class)
        // We could delete it upon destroy, but the user might want to use that buffer... so we will not.

        return synTen;
    }
    auto itr = m_graphEntries.getTensorByPtr(currGraphId, pDescriptor->m_ptr);
    LOG_WARN(SYN_API, "Tensor {} was already mapped", itr->getName());
    return reinterpret_cast<synTensor>(itr.get());
}

#define ASSERT_RET_AND_DESTROY_TENSOR()                                                                                \
    {                                                                                                                  \
        if (ret != synSuccess)                                                                                         \
        {                                                                                                              \
            m_graphEntries.destroyTensor(*synTen);                                                                     \
            return ret;                                                                                                \
        }                                                                                                              \
    }

synStatus synSingleton::createTensor(const synTensorDescriptor* pDescriptor,
                                     synTensor*                 synTen,
                                     synSectionHandle           sectionHandle,
                                     uint64_t                   synSectionOffset,
                                     bool                       isConst)
{
    VERIFY_IS_NULL_POINTER(SYN_API, synTen, "Tensor");

    if (pDescriptor == nullptr)
    {
        return synInvalidArgument;
    }

    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)sectionHandle];

    const bool isPersistentTensor = (sectionPtr != nullptr) && sectionPtr->getPersistent();
    double     scale, zp;
    _getTensorScaleZp(*pDescriptor, scale, zp);

    LOG_DEBUG_T(SYN_API,
                "synSingleton::createTensor {}, is-persistent={}, Data type {}, dims {}, Max Shape ({}) Min Shape ({})"
                "{}, scale={}, zp={} type={} {}{}{}{}{}{}{}{}",
                (pDescriptor->m_name != nullptr) ? pDescriptor->m_name : "",
                isPersistentTensor,
                getStringFromSynDataType(pDescriptor->m_dataType),
                pDescriptor->m_dims,
                toString(pDescriptor->m_sizes, pDescriptor->m_sizes + pDescriptor->m_dims, ','),
                toString(pDescriptor->m_minSizes, pDescriptor->m_minSizes + pDescriptor->m_dims, ','),
                (pDescriptor->m_batchPos != INVALID_BATCH_POS) ? ", batchPos=" + std::to_string(pDescriptor->m_batchPos)
                                                               : "",
                scale,
                zp,
                pDescriptor->m_tensorType,
                pDescriptor->m_isWeights ? "(weights)" : "",
                pDescriptor->m_isQuantized ? "(quantized)" : "",
                pDescriptor->m_isStatic ? "(static)" : "",
                pDescriptor->m_isInput ? "(input)" : "",
                pDescriptor->m_isOutput ? "(output)" : "",
                pDescriptor->m_enablePerChannelQuant ? "(per channel enabled)" : "",
                pDescriptor->m_isSparsityWeights ? "(sparsity weights tensor)" : "",
                pDescriptor->m_dynamicRange.isSet ? ", drange=(" + std::to_string(pDescriptor->m_dynamicRange.min) +
                                                        ", " + std::to_string(pDescriptor->m_dynamicRange.max) + ")"
                                                  : "");

    InternalGraphHandle graphHandle;
    graphHandle.graphId = DUMMY_GRAPH_ID;
    if (sectionPtr)
    {
        graphHandle.graphId = sectionPtr->getGraphID();
    }

    // route this API to the new API
    // tensor creation
    synStatus ret = synSuccess;
    ret           = createTensor(synTen, &graphHandle, pDescriptor->m_tensorType, pDescriptor->m_name);
    if (ret == synNameIsAlreadyUsed) return synSuccess;
    if (ret != synSuccess) return ret;

    // THIS IS A WORKAROUND because old tests rely on transferring pointers to tensors that aren't static.
    if (pDescriptor->m_ptr)
    {
        TensorPtr tensorPtr = m_graphEntries.findTensor(*synTen);
        tensorPtr->setDataUnsafe((char*)pDescriptor->m_ptr);
    }

    // workaround - some tests still rely on isOutput
    if (pDescriptor->m_isOutput)
    {
        TensorPtr tensorPtr = m_graphEntries.findTensor(*synTen);
        tensorPtr->enforceOutput(true);
    }

    // section and host prt
    if (sectionPtr)
    {
        ret = tensorAssignToSectionInternal(*synTen, sectionHandle, sectionPtr, synSectionOffset);
        ASSERT_RET_AND_DESTROY_TENSOR()
    }
    if (isConst || pDescriptor->m_isStatic || pDescriptor->m_tensorType == HOST_SHAPE_TENSOR ||
        pDescriptor->m_tensorType == HOST_TO_DEVICE_TENSOR)
    {
        synDataType bufferType;
        if (pDescriptor->m_tensorType == HOST_SHAPE_TENSOR || pDescriptor->m_tensorType == HOST_TO_DEVICE_TENSOR ||
            pDescriptor->m_isQuantized)
        {
            bufferType = pDescriptor->m_dataType;
        }
        else
        {
            bufferType = syn_type_float;
        }

        ret = tensorSetHostPtr(*synTen, pDescriptor->m_ptr, 0, bufferType, false);
        ASSERT_RET_AND_DESTROY_TENSOR()
    }

    // geometry
    synTensorGeometryExt geometry;
    memset(geometry.sizes, 0, sizeof(geometry.sizes));
    geometry.dims = pDescriptor->m_dims;

    for (unsigned i = 0; i < SYN_MAX_TENSOR_DIM; i++)
    {
        geometry.sizes[i] = (TSize)pDescriptor->m_sizes[i];
    }
    ret           = tensorSetGeometryExt(*synTen, &geometry, synGeometrySizes);
    ASSERT_RET_AND_DESTROY_TENSOR()

    // min geometry if available
    unsigned zeros[SYN_MAX_TENSOR_DIM] = {0};
    if (memcmp(pDescriptor->m_minSizes, zeros, sizeof(pDescriptor->m_minSizes)))
    {
        for (unsigned i = 0; i < SYN_MAX_TENSOR_DIM; i++)
        {
           geometry.sizes[i] = (TSize)pDescriptor->m_minSizes[i];
        }
        geometry.dims = pDescriptor->m_dims;
        ret           = tensorSetGeometryExt(*synTen, &geometry, synGeometryMinSizes);
        ASSERT_RET_AND_DESTROY_TENSOR()
    }

    // device layout
    synTensorDeviceLayout deviceLayout;
    memset(deviceLayout.strides, 0, sizeof(deviceLayout.strides));
    deviceLayout.deviceDataType = pDescriptor->m_dataType;
    memcpy(deviceLayout.strides, pDescriptor->m_strides, sizeof(pDescriptor->m_strides));
    // if strides are not supported, ignore the user reset to default strides
    if (!((pDescriptor->m_strides[0] == 0 ||
           pDescriptor->m_strides[0] == pDescriptor->m_sizes[0] * dataTypeSizeInBytes(pDescriptor->m_dataType, true)) &&
          (pDescriptor->m_strides[1] == 0 ||
           pDescriptor->m_strides[1] == pDescriptor->m_sizes[1] * pDescriptor->m_strides[0]) &&
          (pDescriptor->m_strides[2] == 0 ||
           pDescriptor->m_strides[2] == pDescriptor->m_sizes[2] * pDescriptor->m_strides[1]) &&
          (pDescriptor->m_strides[3] == 0 ||
           pDescriptor->m_strides[3] == pDescriptor->m_sizes[3] * pDescriptor->m_strides[2]) &&
          (pDescriptor->m_strides[4] == 0 ||
           pDescriptor->m_strides[4] == pDescriptor->m_sizes[4] * pDescriptor->m_strides[3])))
    {
        LOG_WARN(SYN_API, "Strides in a user-created tensor are not supported, using default strides.");
        memset(deviceLayout.strides, 0, sizeof(deviceLayout.strides));
    }
    ret = tensorSetDeviceLayout(*synTen, &deviceLayout);
    ASSERT_RET_AND_DESTROY_TENSOR()

    if (GCFG_GAUDI_DEMO.value())
    {
        TensorExUserContext tmpCtx;
        tmpCtx.strides = nullptr;
        uint64_t s[Tensor::c_tensorMaxDim];
        tmpCtx.strides = s;
        for (unsigned i = 0; i < Tensor::c_tensorMaxDim; ++i)
        {
            tmpCtx.strides[i] = pDescriptor->m_strides[i];
        }
        TensorPtr tensorPtr = m_graphEntries.findTensor(*synTen);
        tensorPtr->setUserContext(tmpCtx);
    }

    // quant metadata
    synQuantMetadata quantMetadata;
    for (auto params : pDescriptor->m_quantizationParams)
    {
        if (params.m_qDataType != syn_type_na)
        {
            quantMetadata.dataType = params.m_qDataType;
            synQuantZPScale zpScale;
            zpScale.zp                = params.m_zp;
            zpScale.scale             = params.m_scale;
            quantMetadata.zpScales    = &zpScale;
            quantMetadata.numZPScales = 1;
            ret = tensorSetQuantizationData(*synTen, SYN_QUANT_METADATA, &quantMetadata, sizeof(quantMetadata));
            ASSERT_RET_AND_DESTROY_TENSOR()
        }
    }
    if (pDescriptor->m_enablePerChannelQuant)
    {
        for (auto params : pDescriptor->m_perChannelQuantizationParams)
        {
            if (params.m_qDataType != syn_type_na)
            {
                quantMetadata.dataType   = params.m_qDataType;
                synQuantZPScale* zpScale = new synQuantZPScale[params.m_numChannels];
                for (int i = 0; i < params.m_numChannels; i++)
                {
                    zpScale[i].zp    = params.m_pcZps[i];
                    zpScale[i].scale = params.m_pcScales[i];
                }
                quantMetadata.zpScales    = zpScale;
                quantMetadata.numZPScales = params.m_numChannels;
                ret = tensorSetQuantizationData(*synTen, SYN_QUANT_METADATA, &quantMetadata, sizeof(quantMetadata));
                delete[] zpScale;
                ASSERT_RET_AND_DESTROY_TENSOR()
            }
        }
    }

    // quant fp metadata
    synFpQuantMetadata fpQuantMetadata;
    for (auto params : pDescriptor->m_quantizationParams)
    {
        if (params.m_qDataType != syn_type_na)
        {
            fpQuantMetadata.dataType = params.m_qDataType;
            synFpQuantParam fpQuantParams;
            fpQuantParams.scale              = params.m_scale;
            fpQuantParams.expBias            = params.m_expBias;
            fpQuantMetadata.fpQuantParams    = &fpQuantParams;
            fpQuantMetadata.numFpQuantParams = 1;
            ret = tensorSetQuantizationData(*synTen, SYN_FP_QUANT_METADATA, &fpQuantMetadata, sizeof(fpQuantMetadata));
            ASSERT_RET_AND_DESTROY_TENSOR()
        }
    }
    if (pDescriptor->m_enablePerChannelQuant)
    {
        for (auto params : pDescriptor->m_perChannelQuantizationParams)
        {
            if (params.m_qDataType != syn_type_na)
            {
                fpQuantMetadata.dataType       = params.m_qDataType;
                synFpQuantParam* fpQuantParams = new synFpQuantParam[params.m_numChannels];
                for (int i = 0; i < params.m_numChannels; i++)
                {
                    fpQuantParams[i].scale   = params.m_pcScales[i];
                    fpQuantParams[i].expBias = params.m_pcExpBias[i];
                }
                fpQuantMetadata.fpQuantParams    = fpQuantParams;
                fpQuantMetadata.numFpQuantParams = params.m_numChannels;
                ret                              = tensorSetQuantizationData(*synTen,
                                                SYN_FP_QUANT_METADATA,
                                                &fpQuantMetadata,
                                                sizeof(fpQuantMetadata));
                delete[] fpQuantParams;
                ASSERT_RET_AND_DESTROY_TENSOR()
            }
        }
    }

    // quant flags
    synQuantFlags quantFlags;
    quantFlags.isSparsifiedWeights   = pDescriptor->m_isSparsityWeights;
    quantFlags.enablePerChannelQuant = pDescriptor->m_enablePerChannelQuant;
    quantFlags.isWeights             = pDescriptor->m_isWeights;
    ret = tensorSetQuantizationData(*synTen, SYN_QUANT_FLAGS, &quantFlags, sizeof(quantFlags));
    ASSERT_RET_AND_DESTROY_TENSOR()

    // dynamic range
    if (pDescriptor->m_dynamicRange.isSet)
    {
        synQuantDynamicRange quantDynamicRange;
        quantDynamicRange.max = pDescriptor->m_dynamicRange.max;
        quantDynamicRange.min = pDescriptor->m_dynamicRange.min;
        ret =
            tensorSetQuantizationData(*synTen, SYN_QUANT_DYNAMIC_RANGE, &quantDynamicRange, sizeof(quantDynamicRange));
        ASSERT_RET_AND_DESTROY_TENSOR()
    }

    return synSuccess;
}

synStatus
synSingleton::createTensor(synTensor* tensor, synGraphHandle graph, synTensorType type, const char* tensorName)
{
    std::string tensorNameStr(tensorName ? tensorName : "");
    LOG_DEBUG_T(SYN_API,
                "{}: name: {}, type: {}, graph ID: {}",
                HLLOG_FUNC,
                tensorName ? tensorName : "nullptr",
                synTensorType2Txt(type),
                graph->graphId);
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");

    TensorPtr createTensorPtr = std::make_shared<Tensor>(type, graph->graphId, tensorName);
    createTensorPtr->setProp(synTensorPropType);
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        // If the user provided a name, enforce its uniqueness
        if (tensorName != nullptr)
        {
            if (m_graphEntries.doTensorExistByName(graph->graphId, tensorNameStr))
            {
                LOG_WARN(SYN_API, "Tensor {} was already mapped", tensorName);
                *tensor =
                    reinterpret_cast<synTensor>(m_graphEntries.getTensorByName(graph->graphId, tensorNameStr).get());
                return synNameIsAlreadyUsed;
            }
            else
            {
                m_graphEntries.setTensorByName(graph->graphId, tensorNameStr, createTensorPtr);
                createTensorPtr->setProp(synTensorPropName);
            }
        }
        m_graphEntries.setTensorSharedPtr(graph->graphId, createTensorPtr);
    }
    *tensor = reinterpret_cast<synTensor>(createTensorPtr.get());

    synMemoryDescriptor memDesc(false);
    createTensorPtr->setMemoryDescriptor(memDesc);
    createTensorPtr->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);

    LOG_DEBUG_T(SYN_API,
                "{}: created tensor with name: {}, id: {}, synTensor 0x{:x}, graph: 0x{:x}",
                HLLOG_FUNC,
                createTensorPtr->getName(),
                createTensorPtr->getId(),
                TO64(*tensor),
                TO64(graph));
    return synSuccess;
}

#define GET_TENSOR_UNSAFE() Tensor* t = reinterpret_cast<Tensor*>((tensor))

#define FIND_TENSOR_AND_LOCK_GRAPHS()                                                                                  \
    TensorPtr                    tensorPtr;                                                                            \
    std::unique_lock<std::mutex> guard(m_graphsMutex);                                                                 \
    {                                                                                                                  \
        Tensor* t = reinterpret_cast<Tensor*>(tensor);                                                                 \
        if (t == nullptr)                                                                                              \
        {                                                                                                              \
            LOG_ERR(SYN_API, "{}: tensor handle not initialized.", HLLOG_FUNC);                                        \
            return synObjectNotInitialized;                                                                            \
        }                                                                                                              \
        tensorPtr = m_graphEntries.findTensor(tensor);                                                                 \
        if (tensorPtr == nullptr)                                                                                      \
        {                                                                                                              \
            LOG_ERR(SYN_API, "{}: tensor not found in graph.", HLLOG_FUNC);                                            \
            return synFail;                                                                                            \
        }                                                                                                              \
    }

#define IS_SFG_SUPPORTED(t, isExternal)                                                                                \
    {                                                                                                                  \
        if (isExternal)                                                                                                \
        {                                                                                                              \
            const uint32_t graphId = (t)->getGraphID();                                                                \
            if (m_graphEntries.getDeviceTypeByGraphId(graphId) == synDeviceGaudi)                                      \
            {                                                                                                          \
                LOG_ERR(SYN_API,                                                                                       \
                        "{}: tensor {} can't set to external, SFG is not supported by device Gaudi",                   \
                        HLLOG_FUNC,                                                                                    \
                        (t)->getName());                                                                               \
            }                                                                                                          \
        }                                                                                                              \
    }

#define IS_PROPS_LOCKED(t)                                                                                             \
    {                                                                                                                  \
        if ((t)->isPropsLocked())                                                                                      \
        {                                                                                                              \
            LOG_ERR(SYN_API,                                                                                           \
                    "{}: tensor {} already added to the graph and cannot be changed",                                  \
                    HLLOG_FUNC,                                                                                        \
                    (t)->getName());                                                                                   \
            return synObjectAlreadyInitialized;                                                                        \
        }                                                                                                              \
    }

synStatus synSingleton::tensorSetSectionOffset(synTensor tensor, uint64_t byteOffset)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_DEBUG_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, section ID: {}, byte offset: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                t->getMemorySectionID(),
                byteOffset);

    auto sectionPtr = t->getSectionHandle();

    VERIFY_IS_NULL_POINTER(SYN_API, sectionPtr, "SectionPtr");

    if (!sectionPtr->getRMW() && sectionPtr->getPersistent())
    {
        t->setMemorySectionOffset(byteOffset);
    }
    else if (!sectionPtr->getPersistent())
    {
        auto& nonPersistentSectionInfo = t->getTensorAnnotation().nonPersistentSectionInfo;
        nonPersistentSectionInfo.offsetFromBase.set(byteOffset);
    }
    else if (sectionPtr->getRMW() && sectionPtr->getPersistent())
    {
        LOG_ERR(SYN_API, "{}: Persistent RMW tensors are not supported", HLLOG_FUNC);
        return synUnsupported;
    }
    return synSuccess;
}

synStatus synSingleton::tensorAssignToSection(synTensor tensor, synSectionHandle section, uint64_t byteOffset)
{
    auto sectionPtr = m_sectionHndlSlopMap[(SMHandle)section];

    VERIFY_IS_NULL_POINTER(SYN_API, sectionPtr, "SectionPtr");

    return tensorAssignToSectionInternal(tensor, section, sectionPtr, byteOffset);
}

synStatus synSingleton::tensorAssignToSectionInternal(synTensor              tensor,
                                                      synSectionHandle       sectionHandle,
                                                      SlotMapItemSptr<InternalSectionHandle> sectionPtr,
                                                      uint64_t               byteOffset)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_TRACE_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, section ID: {}, byte offset: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                sectionPtr->getID(),
                byteOffset);

    if (t->getGraphID() != sectionPtr->getGraphID())
    {
        LOG_ERR(SYN_API, "{}: section graph ID does not match the tensors graph ID", HLLOG_FUNC);
        return synInvalidArgument;
    }
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        if (!_validateSectionForGraph(sectionPtr.get()))
        {
            LOG_ERR(SYN_API, "{}: Invalid section used in tensor '{}'", HLLOG_FUNC, t->getName());
            return synInvalidArgument;
        }
        _sectionLockAndSetID(sectionPtr.get());
    }
    return tensorUpdateSectionInfoInternal(tensor, sectionHandle, sectionPtr, byteOffset);
}

synStatus synSingleton::tensorUpdateSectionInfoInternal(synTensor              tensor,
                                                        synSectionHandle       sectionHandle,
                                                        SlotMapItemSptr<InternalSectionHandle> sectionPtr,
                                                        uint64_t               byteOffset)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    GET_TENSOR_UNSAFE();

    if (!sectionPtr->getRMW() && sectionPtr->getPersistent())
    {
        synMemoryDescriptor memDesc(sectionPtr->getPersistent());
        t->setMemoryDescriptor(memDesc);
        t->setMemorySectionID(sectionPtr->getID());
        t->setMemorySectionOffset(byteOffset);
    }
    else if (sectionPtr->getRMW() && !sectionPtr->getPersistent())
    {
        t->setTensorInSram();
        auto& nonPersistentSectionInfo = t->getTensorAnnotation().nonPersistentSectionInfo;
        nonPersistentSectionInfo.sectionId.set(sectionPtr->getID());
        nonPersistentSectionInfo.offsetFromBase.set(byteOffset);
    }
    else if (!sectionPtr->getRMW() && !sectionPtr->getPersistent())
    {
        t->setTensorInWorkspace();
        auto& nonPersistentSectionInfo = t->getTensorAnnotation().nonPersistentSectionInfo;
        nonPersistentSectionInfo.sectionId.set(sectionPtr->getID());
        nonPersistentSectionInfo.offsetFromBase.set(byteOffset);
    }
    else if (sectionPtr->getRMW() && sectionPtr->getPersistent())
    {
        LOG_ERR(SYN_API, "{}: Persistent RMW tensors are not supported", HLLOG_FUNC);
        return synUnsupported;
    }

    t->setSectionHandle(sectionPtr);
    t->setProp(synTensorPropSection);
    t->setProp(synTensorPropSectionOffset);
    return synSuccess;
}

synStatus
synSingleton::tensorSetHostPtr(synTensor tensor, void* hostPtr, uint64_t size, synDataType dataType, bool copyBuffer)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, hostPtr, "HostPtr");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_TRACE_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, host ptr: {}, size: {}, data type: {}, copy buffer: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                hostPtr,
                size,
                getStringFromSynDataType(dataType),
                copyBuffer);

    t->setTensorBuffer(hostPtr, size, dataType, copyBuffer);

    synMemoryDescriptor memDesc(false);

    if (!t->inConstSection() && t->getTensorType() != HOST_TO_DEVICE_TENSOR)
    {
        // since tensor was set to persistent in tensorAssignToSection - we do not want to override it
        t->setMemoryDescriptor(memDesc);
    }

    if (t->getTensorType() != HOST_TO_DEVICE_TENSOR)
    {
        if (!t->inConstSection())
        {
            LOG_WARN(SYN_API,
                     "Const tenosr {} should be in a const section. size {} bytes.",
                     t->getName(),
                     t->getBufferSizeInBytes());
            if (t->getBufferSizeInBytes() > GCFG_MAX_CONST_TENSOR_SIZE_BYTES.value())
            {
                LOG_ERR(SYN_API,
                        "Const tenosr {} should be in a const section. Tensor is too big. Actual size {} bytes is "
                        "grater than limit size of {} bytes.",
                        t->getName(),
                        t->getBufferSizeInBytes(),
                        GCFG_MAX_CONST_TENSOR_SIZE_BYTES.value());
                return synInvalidArgument;
            }
            // if section is const - do not override tensor section id
            t->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
        }
        t->setAsStaticParam();
        t->setProp(synTensorPropHostPtr);
        FIND_TENSOR_AND_LOCK_GRAPHS();
        m_graphEntries.setTensorByPtr(tensorPtr->getGraphID(), hostPtr, tensorPtr);
    }

    return synSuccess;
}

synStatus synSingleton::tensorSetQuantizationData(synTensor               tensor,
                                                  synQuantizationProperty prop,
                                                  void*                   propVal,
                                                  uint64_t                propSize)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, propVal, "PropVal");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_TRACE_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, prop: {}, prop val: {}, prop size: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                prop,
                propVal,
                propSize);

    switch (prop)
    {
        case SYN_QUANT_DYNAMIC_RANGE:
        {
            HB_ASSERT(propSize == sizeof(synQuantDynamicRange), "invalid prop size");
            synQuantDynamicRange* userDrange = (synQuantDynamicRange*)propVal;
            DynamicRange          drange;
            drange.min   = userDrange->min;
            drange.max   = userDrange->max;
            drange.isSet = true;
            if (!t->setDynamicRange(drange))
            {
                LOG_ERR(SYN_API,
                        "{}: tensor {} invalid Dynamic Range: min={}, max={} ",
                        HLLOG_FUNC,
                        t->getName(),
                        drange.min,
                        drange.max);
                return synInvalidArgument;
            }

            LOG_TRACE_T(SYN_API,
                        "{}: tensor ID: {}, tensor name: {}, SYN_QUANT_DYNAMIC_RANGE: min: {} max {}",
                        HLLOG_FUNC,
                        t->getId(),
                        t->getName(),
                        drange.min,
                        drange.max);

            t->setProp(synTensorPropDynamicRange);
            break;
        }
        case SYN_QUANT_PC_DYNAMIC_RANGE:
        {
            HB_ASSERT(propSize == sizeof(synPerChannelDynamicRange), "invalid prop size");
            synPerChannelDynamicRange* userDrange = (synPerChannelDynamicRange*)propVal;
            HB_ASSERT_PTR(userDrange->ranges);
            if (!t->setPerChannelDynamicRange(*userDrange))
            {
                LOG_ERR(SYN_API, "{}: tensor {} invalid Dynamic Range", HLLOG_FUNC, t->getName());
                return synInvalidArgument;
            }

            LOG_TRACE_T(SYN_API,
                        "{}: tensor ID: {}, tensor name: {}, SYN_QUANT_PC_DYNAMIC_RANGE",
                        HLLOG_FUNC,
                        t->getId(),
                        t->getName());

            t->setProp(synTensorPropPCDynamicRange);
            break;
        }
        case SYN_QUANT_METADATA:
        {
            HB_ASSERT(propSize == sizeof(synQuantMetadata), "invalid prop size");
            synQuantMetadata* userMetadata = (synQuantMetadata*)propVal;
            HB_ASSERT_PTR(userMetadata->zpScales);
            t->setQuantizationParams(*userMetadata);
            t->setProp(synTensorPropQuantMetadata);
            break;
        }
        case SYN_FP_QUANT_METADATA:
        {
            HB_ASSERT(propSize == sizeof(synFpQuantMetadata), "invalid prop size");
            synFpQuantMetadata* userMetadata = (synFpQuantMetadata*)propVal;
            HB_ASSERT_PTR(userMetadata->fpQuantParams);
            t->setQuantizationParams(*userMetadata);
            t->setProp(synTensorPropFpQuantMetadata);
            break;
        }
        case SYN_QUANT_FLAGS:
        {
            HB_ASSERT(propSize == sizeof(synQuantFlags), "invalid prop size");
            synQuantFlags* userFlags = (synQuantFlags*)propVal;
            t->setPerChannelQuant(userFlags->enablePerChannelQuant);
            if (userFlags->isSparsifiedWeights)
            {
                t->setAsSparsityWeights();
            }
            if (userFlags->isWeights)
            {
                t->setAsWeights();
            }
            t->setProp(synTensorPropFlags);
            break;
        }
        default:
        {
            LOG_ERR(SYN_API, "{}: Invalid prop", HLLOG_FUNC);
            return synInvalidArgument;
        }
    }

    return synSuccess;
}

synStatus synSingleton::tensorSetIsExternal(synTensor tensor, bool isExternal)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    GET_TENSOR_UNSAFE();
    IS_SFG_SUPPORTED(t, isExternal);
    IS_PROPS_LOCKED(t);
    LOG_DEBUG_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, as external {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                isExternal);

    t->setTensorAsExternal(isExternal);
    return synSuccess;
}

synStatus
synSingleton::tensorSetGeometry(synTensor tensor, const synTensorGeometry* geometry, synGeometryType geometryType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, geometry, "Geometry");
    VERIFY_IS_NULL_POINTER(SYN_API, geometry->sizes, "Sizes");
    LOG_TRACE_T(SYN_API, "{}", HLLOG_FUNC);

    synTensorGeometryExt geometry64bit;
    for (unsigned i = 0; i < HABANA_DIM_MAX; i++)
    {
        geometry64bit.sizes[i] = (TSize)geometry->sizes[i];
    }
    geometry64bit.dims = geometry->dims;

    return tensorSetGeometryExt(tensor, &geometry64bit, geometryType);
}

synStatus
synSingleton::tensorSetGeometryExt(synTensor tensor, const synTensorGeometryExt* geometry, synGeometryType geometryType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, geometry, "Geometry");
    VERIFY_IS_NULL_POINTER(SYN_API, geometry->sizes, "Sizes");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_DEBUG_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, dims: {}, shape: {}, geometry type: {}, tensor 0x{:x}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                geometry->dims,
                arrayToString(geometry->sizes, ','),
                geometryType,
                TO64(tensor));

    TSize sizes[HABANA_DIM_MAX];
    castNcopy(sizes, geometry->sizes, HABANA_DIM_MAX);
    t->setGeometry(geometry->dims, sizes, geometryType);
    return synSuccess;
}

synStatus synSingleton::tensorSetPermutation(synTensor tensor, const synTensorPermutation* permutation)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, permutation, "Permutation");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_DEBUG_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, dims: {}, tensor: 0x{:x}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                permutation->dims,
                TO64(tensor));

    DimVector permVector;
    if (permutation->dims == 0)
    {
        LOG_DEBUG(SYN_GRAPH,
                  "{}: tensor {} got permutation with dims 0 - unsetting permutation",
                  HLLOG_FUNC,
                  t->getName());
        t->unsetPermutation();
        return synSuccess;
    }
    for (unsigned i = 0; i < permutation->dims; i++)
    {
        permVector.push_back(permutation->permutation[i]);
    }
    gc::Permutation gcPermutation(permVector);

    if (!t->setPermutation(gcPermutation))
    {
        LOG_ERR(SYN_API, "{}: tensor {} invalid permutation", HLLOG_FUNC, t->getName());
        return synInvalidArgument;
    }

    t->setProp(synTensorPropPermutation);

    return synSuccess;
}

synStatus synSingleton::tensorSetDeviceDataType(synTensor tensor, synDataType deviceDataType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_DEBUG_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, device data type: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                getStringFromSynDataType(deviceDataType));

    t->setElementTypeWithoutStrides(deviceDataType);

    t->setProp(synTensorPropDeviceDataType);
    return synSuccess;
}

synStatus synSingleton::tensorSetDeviceLayout(synTensor tensor, const synTensorDeviceLayout* layout)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, layout, "Layout");
    synTensorDeviceFullLayout fullLayout;
    fullLayout.deviceDataType = layout->deviceDataType;
    memcpy(fullLayout.strides + 1, layout->strides, sizeof(layout->strides));
    unsigned zeros[HABANA_DIM_MAX] = {0};
    if (memcmp(layout->strides, zeros, sizeof(layout->strides)) == 0)
    {
        fullLayout.strides[0] = 0;
    }
    else
    {
        fullLayout.strides[0] = dataTypeSizeInBytes(layout->deviceDataType, true);
    }
    return tensorSetDeviceFullLayout(tensor, &fullLayout);
}

synStatus synSingleton::tensorSetDeviceFullLayout(synTensor tensor, const synTensorDeviceFullLayout* layout)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, layout, "Layout");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    LOG_TRACE_T(SYN_API,
                "{}: tensor ID: {}, tensor name: {}, device data type: {}, strides: {}",
                HLLOG_FUNC,
                t->getId(),
                t->getName(),
                getStringFromSynDataType(layout->deviceDataType),
                arrayToString(layout->strides, ','));

    uint32_t* strides               = (uint32_t*)layout->strides;
    unsigned  zeros[HABANA_DIM_MAX] = {0};
    // if strides is zeros, pass nullptr to calculate strides in tensor
    if (memcmp(layout->strides, zeros, sizeof(layout->strides)) == 0)
    {
        strides = nullptr;
    }
    else
    {
        LOG_DEBUG(SYN_GRAPH, "{}: Trying to set non-zero strides in Tensor {}", HLLOG_FUNC, t->getName());
        // TODO SW-86687: change to ERROR and return synInvalidArgument once this is removed from all tests and infra
    }

    synDataType layoutDataType = layout->deviceDataType;
    t->setDeviceLayout(layoutDataType, strides);

    t->setProp(synTensorPropDeviceLayout);
    return synSuccess;
}

#define VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, ptr, propType)                                                           \
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor")                                                                  \
    VERIFY_IS_NULL_POINTER(SYN_API, ptr, "property")                                                                   \
    GET_TENSOR_UNSAFE();                                                                                               \
    if (!t->isPropSet(propType))                                                                                       \
    {                                                                                                                  \
        LOG_WARN(SYN_API, "{}: Prop {} is not set in tensor. ", HLLOG_FUNC, propType);                                 \
        return synObjectNotInitialized;                                                                                \
    }

synStatus synSingleton::tensorSetAllowPermutation(synTensor tensor, int8_t allow)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor")
    GET_TENSOR_UNSAFE();
    t->getTensorAnnotation().memory.allowPermutation = allow;
    return synSuccess;
}

synStatus synSingleton::tensorGetAllowPermutation(synTensor tensor, int8_t* allow)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor")
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Out_allow_permutation")
    GET_TENSOR_UNSAFE();
    *allow = t->getTensorAnnotation().memory.allowPermutation;
    return synSuccess;
}

synStatus synSingleton::tensorGetSection(synTensor tensor, synSectionHandle* section, uint64_t* byteOffset)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, section, synTensorPropSection);
    VERIFY_IS_NULL_POINTER(SYN_API, byteOffset, "size")
    // TODO add in Tensor class protection against getting a section that was deleted by user (synSectionDestroy API)
    const auto& internalSection = t->getSectionHandle();
    *section = internalSection->getSectionHandle();
    if (*section == nullptr)
    {
        LOG_DEBUG(SYN_GRAPH, "Tensor {} has no section.", t->getName());
        *byteOffset = 0;
        return synSuccess;
    }
    *byteOffset = t->getMemorySectionOffset();
    return synSuccess;
}

synStatus synSingleton::tensorGetHostPtr(synTensor tensor, void** hostPtr, uint64_t* size, synDataType* dataType)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, hostPtr, synTensorPropHostPtr)
    VERIFY_IS_NULL_POINTER(SYN_API, size, "size")
    VERIFY_IS_NULL_POINTER(SYN_API, dataType, "dataType")
    *size     = t->getBufferSizeInBytes();
    *dataType = t->getBufferDataType();
    *hostPtr  = t->getAddress();
    return synSuccess;
}

synStatus synSingleton::tensorGetQuantizationData(synTensor               tensor,
                                                  synQuantizationProperty prop,
                                                  void*                   propVal,
                                                  uint64_t                propSize)
{
    synTensorProperty tensorProp = quantizationPropertyToTensorProperty(prop);
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, propVal, tensorProp)
    switch (prop)
    {
        case SYN_QUANT_DYNAMIC_RANGE:
        {
            if (propSize != sizeof(synQuantDynamicRange))
            {
                LOG_ERR(SYN_API, "{}, Invalid synQuantDynamicRange prop size", HLLOG_FUNC);
                return synFail;
            }
            synQuantDynamicRange* ptrReturnedDynamicRange = (synQuantDynamicRange*)propVal;
            ptrReturnedDynamicRange->min                  = t->getDynamicRange().min;
            ptrReturnedDynamicRange->max                  = t->getDynamicRange().max;
            break;
        }
        case SYN_QUANT_PC_DYNAMIC_RANGE:
        {
            auto     pcDynamicRange = t->getPerChannelDynamicRange();
            unsigned numChannels    = pcDynamicRange.numChannels;

            // validate property size
            if (propSize != sizeof(synPerChannelDynamicRange) &&
                (propSize != sizeof(synPerChannelDynamicRange) + numChannels * sizeof(synQuantDynamicRange)))
            {
                LOG_ERR(SYN_API, "{}, Invalid synPerChannelDynamicRange prop size", HLLOG_FUNC);
                return synFail;
            }

            synPerChannelDynamicRange* ptrReturnedDynamicRange = (synPerChannelDynamicRange*)propVal;

            if (ptrReturnedDynamicRange->ranges == nullptr)
            {
                ptrReturnedDynamicRange->numChannels = numChannels;
                LOG_TRACE(SYN_API, "Number of channels for tensor {} is {}", t->getName(), numChannels);
            }
            else
            {
                HB_ASSERT(ptrReturnedDynamicRange->numChannels == numChannels, "numChannels is expected to be equal to tensor's number of channels");
                // validate again that size is correct
                if (propSize != sizeof(synPerChannelDynamicRange) + numChannels * sizeof(synQuantDynamicRange))
                {
                    LOG_ERR(SYN_API, "{}, Invalid synPerChannelDynamicRange prop size, should be struct plus allocated array size", HLLOG_FUNC);
                    return synFail;
                }
                // copy from tensor QuantizationParams to synFpQuantParam array
                for (unsigned i = 0; i < numChannels; i++)
                {
                    ptrReturnedDynamicRange->ranges[i] = pcDynamicRange.ranges[i];
                }
            }

            break;
        }
        case SYN_QUANT_METADATA:
        {
            /*
             * validate property size is either just struct size (in case no synQuantZPScale array was allocated)
             * or the struct size + size of allocated synQuantZPScale array
             */
            if (propSize != sizeof(synQuantMetadata) &&
                (propSize !=
                 sizeof(synQuantMetadata) + t->getQuantizationParams().m_numChannels * sizeof(synQuantZPScale)))
            {
                LOG_ERR(SYN_API, "{}, Invalid synQuantMetadata prop size", HLLOG_FUNC);
                return synFail;
            }
            synQuantMetadata* ptrReturnedQuantMetadata = (synQuantMetadata*)propVal;
            synDataType       userGivenDataType        = ptrReturnedQuantMetadata->dataType;
            if (!t->isQuantizationParamsExist(userGivenDataType))
            {
                LOG_ERR(SYN_API,
                        "{}: Tensor {} does not have quantization metadata for data type {}",
                        HLLOG_FUNC,
                        t->getName(),
                        getStringFromSynDataType(userGivenDataType));
                return synInvalidArgument;
            }

            auto     tensorQuantInfo = t->getQuantizationParams(userGivenDataType);
            unsigned numChannels     = tensorQuantInfo.m_numChannels;
            if (ptrReturnedQuantMetadata->zpScales == nullptr)
            {
                ptrReturnedQuantMetadata->numZPScales = numChannels;
                LOG_TRACE(SYN_API, "Number of channels for tensor {} is {}", t->getName(), numChannels);
            }
            else
            {
                HB_ASSERT(ptrReturnedQuantMetadata->numZPScales == numChannels,
                          "numZPScales is expected to be equal to tensor's number of channels");
                // validate again that size is correct
                if (propSize !=
                    sizeof(synQuantMetadata) + t->getQuantizationParams().m_numChannels * sizeof(synQuantZPScale))
                {
                    LOG_ERR(SYN_API,
                            "{}, Invalid synQuantMetadata prop size, should be struct plus allocated array size",
                            HLLOG_FUNC);
                    return synFail;
                }
                // copy from tensor QuantizationParams to synQuantZPScale array
                for (unsigned i = 0; i < tensorQuantInfo.m_numChannels; i++)
                {
                    ptrReturnedQuantMetadata->zpScales[i] = {tensorQuantInfo.zp(i), tensorQuantInfo.scale(i)};
                }
            }
            break;
        }
        case SYN_FP_QUANT_METADATA:
        {
            /*
             * validate property size is either just struct size (in case no synFpQuantParam array was allocated)
             * or the struct size + size of allocated synFpQuantParam array
             */
            if (propSize != sizeof(synFpQuantMetadata) &&
                (propSize !=
                 sizeof(synFpQuantMetadata) + t->getQuantizationParams().m_numChannels * sizeof(synFpQuantParam)))
            {
                LOG_ERR(SYN_API, "{}, Invalid synFpQuantMetadata prop size", HLLOG_FUNC);
                return synFail;
            }
            synFpQuantMetadata* ptrReturnedQuantMetadata = (synFpQuantMetadata*)propVal;
            synDataType         userGivenDataType        = ptrReturnedQuantMetadata->dataType;
            if (!t->isQuantizationParamsExist(userGivenDataType))
            {
                LOG_ERR(SYN_API,
                        "{}: Tensor {} does not have quantization metadata for data type {}",
                        HLLOG_FUNC,
                        t->getName(),
                        getStringFromSynDataType(userGivenDataType));
                return synInvalidArgument;
            }

            auto     tensorQuantInfo = t->getQuantizationParams(userGivenDataType);
            unsigned numChannels     = tensorQuantInfo.m_numChannels;
            if (ptrReturnedQuantMetadata->fpQuantParams == nullptr)
            {
                ptrReturnedQuantMetadata->numFpQuantParams = numChannels;
                LOG_TRACE(SYN_API, "Number of channels for tensor {} is {}", t->getName(), numChannels);
            }
            else
            {
                HB_ASSERT(ptrReturnedQuantMetadata->numFpQuantParams == numChannels,
                          "numFpQuantParams is expected to be equal to tensor's number of channels");
                // validate again that size is correct
                if (propSize !=
                    sizeof(synFpQuantMetadata) + t->getQuantizationParams().m_numChannels * sizeof(synFpQuantParam))
                {
                    LOG_ERR(SYN_API,
                            "{}, Invalid synFpQuantMetadata prop size, should be struct plus allocated array size",
                            HLLOG_FUNC);
                    return synFail;
                }
                // copy from tensor QuantizationParams to synFpQuantParam array
                for (unsigned i = 0; i < tensorQuantInfo.m_numChannels; i++)
                {
                    ptrReturnedQuantMetadata->fpQuantParams[i] = {tensorQuantInfo.scale(i), tensorQuantInfo.expBias(i)};
                }
            }
            break;
        }
        case SYN_QUANT_FLAGS:
        {
            if (propSize != sizeof(synQuantFlags))
            {
                LOG_ERR(SYN_API, "{}, Invalid synQuantFlags prop size", HLLOG_FUNC);
                return synFail;
            }
            synQuantFlags* ptrReturnedQuantFlags         = (synQuantFlags*)propVal;
            ptrReturnedQuantFlags->enablePerChannelQuant = t->isPerChannelQuant();
            ptrReturnedQuantFlags->isSparsifiedWeights   = t->isSparsityWeights();
            ptrReturnedQuantFlags->isWeights             = t->isWeights();
            break;
        }
        default:
            HB_ASSERT(false, "Unknown synQuantizationProperty {}", tensorProp);
            return synInvalidArgument;
    }
    return synSuccess;
}

synStatus
synSingleton::tensorGetGeometryExt(const synTensor tensor, synTensorGeometryExt* geometry, synGeometryType geometryType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor")
    VERIFY_IS_NULL_POINTER(SYN_API, geometry, "Geometry");
    GET_TENSOR_UNSAFE();
    switch (geometryType)
    {
        case synGeometryMinSizes:
        {
            if (!t->isPropSet(synTensorPropGeometryMin))
            {
                LOG_ERR(SYN_API, "{}: Prop synGeometryMinSizes unset for tensor \"{}\"", HLLOG_FUNC, t->getName());
                return synObjectNotInitialized;
            }
            TSize sizes[Tensor::c_tensorMaxNDim];
            t->getAllMinimalSizesInElements(sizes, Tensor::c_tensorMaxNDim);
            castNcopy(geometry->sizes, sizes, Tensor::c_tensorMaxNDim);
            geometry->dims = t->getDim();
            break;
        }
        case synGeometryMaxSizes:
        {
            if (!t->isPropSet(synTensorPropGeometryMax))
            {
                LOG_ERR(SYN_API, "{}: Prop synGeometryMaxSizes unset for tensor \"{}\"", HLLOG_FUNC, t->getName());
                return synObjectNotInitialized;
            }
            TSize sizes[Tensor::c_tensorMaxNDim];
            t->getAllSizesInElements(sizes, Tensor::c_tensorMaxNDim);
            castNcopy(geometry->sizes, sizes, Tensor::c_tensorMaxNDim);
            geometry->dims = t->getDim();
            break;
        }
        case synGeometryDims:
        {
            if (!t->isPropSet(synTensorPropGeometryDim))
            {
                LOG_ERR(SYN_API, "{}: Prop synGeometryDims unset for tensor \"{}\"", HLLOG_FUNC, t->getName());
                return synObjectNotInitialized;
            }
            geometry->dims = t->getDim();
            break;
        }
        default:
            LOG_ERR(SYN_API, "{}: Unknown synGeometryType", HLLOG_FUNC);
            return synInvalidArgument;
    }

    return synSuccess;
}

synStatus
synSingleton::tensorGetGeometry(const synTensor tensor, synTensorGeometry* geometry, synGeometryType geometryType)
{
    VERIFY_IS_NULL_POINTER(SYN_API, geometry, "Geometry");
    GET_TENSOR_UNSAFE();
    synTensorGeometryExt geometry64bit;
    synStatus            status = tensorGetGeometryExt(tensor, &geometry64bit, geometryType);
    if (status != synSuccess) return status;

    for (unsigned i = 0; i < HABANA_DIM_MAX; i++)
    {
        if (geometry64bit.sizes[i] > std::numeric_limits<uint32_t>::max())
        {
            LOG_WARN(SYN_API,
                     "{}: overflow detected when using 32bit geometry version for tensor \"{}\"",
                     HLLOG_FUNC,
                     t->getName());
        }
        geometry->sizes[i] = geometry64bit.sizes[i];
    }
    geometry->dims = geometry64bit.dims;

    return synSuccess;
}

synStatus synSingleton::tensorGetIsExternal(const synTensor tensor, bool* isExternal)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    VERIFY_IS_NULL_POINTER(SYN_API, isExternal, "isExternal");
    GET_TENSOR_UNSAFE();
    IS_PROPS_LOCKED(t);
    *isExternal = t->getTensorIsExternal();
    return synSuccess;
}

synStatus synSingleton::tensorGetPermutation(const synTensor tensor, synTensorPermutation* permutation)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, permutation, synTensorPropPermutation);

    const std::optional<gc::Permutation>& perm = t->getPermutation();
    if (!perm)
    {
        LOG_ERR(SYN_API, "{}: Permutation is not set for tensor \"{}\"", HLLOG_FUNC, t->getName());
        return synObjectNotInitialized;
    }
    permutation->dims = perm->size();
    for (unsigned i = 0; i < permutation->dims; i++)
    {
        permutation->permutation[i] = perm->getValues()[i];
    }
    return synSuccess;
}

synStatus synSingleton::tensorGetDeviceDataType(synTensor tensor, synDataType* deviceDataType)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, deviceDataType, synTensorPropDeviceLayout)
    *deviceDataType = t->getElementType();
    return synSuccess;
}

synStatus synSingleton::tensorGetDeviceLayout(const synTensor tensor, synTensorDeviceLayout* deviceLayout)
{
    VERIFY_IS_NULL_POINTER(SYN_API, deviceLayout, "Layout");
    synTensorDeviceFullLayout fullLayout;
    synStatus                 status = tensorGetDeviceFullLayout(tensor, &fullLayout);
    if (status != synSuccess)
    {
        return status;
    }
    memcpy(deviceLayout->strides, fullLayout.strides + 1, sizeof(deviceLayout->strides));
    deviceLayout->deviceDataType = fullLayout.deviceDataType;
    return synSuccess;
}

synStatus synSingleton::tensorGetDeviceFullLayout(const synTensor tensor, synTensorDeviceFullLayout* deviceLayout)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, deviceLayout, synTensorPropDeviceLayout);
    TStride arrayTStride[HABANA_DIM_MAX];
    t->getAllStridesInBytes(arrayTStride, HABANA_DIM_MAX);
    castNcopy(deviceLayout->strides, arrayTStride, HABANA_DIM_MAX);
    deviceLayout->deviceDataType = t->getElementType();

    return synSuccess;
}

synStatus synSingleton::tensorGetName(const synTensor tensor, const uint64_t size, char* name)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, name, synTensorPropName)
    size_t tensorNameSizeInBytes = t->getName().size() + 1;
    if (size < tensorNameSizeInBytes)
    {
        LOG_ERR(SYN_API, "{}: buffer size {} for name is too small.", HLLOG_FUNC, size);
        return synInvalidArgument;
    }
    memcpy(name, t->getName().c_str(), tensorNameSizeInBytes);
    return synSuccess;
}

synStatus synSingleton::tensorGetType(const synTensor tensor, synTensorType* type)
{
    VERIFY_NULL_GET_TENSOR_UNSAFE(tensor, type, synTensorPropType)
    *type = t->getTensorType();
    return synSuccess;
}

/* Temporary API - For internal use */
synStatus synSingleton::setTensorDeviceAddr(synTensor tensor, uint64_t addr)
{
    LOG_SINGLETON_API();

    std::unique_lock<std::mutex> guard(m_graphsMutex);
    TensorPtr                    t = m_graphEntries.findTensor(tensor);
    if (t == nullptr)
    {
        return synObjectNotInitialized;
    }

    LOG_TRACE(SYN_API, "{} setting tensor {} addr to 0x{:x}", HLLOG_FUNC, t->getName(), addr);

    std::shared_ptr<HalReader> pHalReader;
    synStatus                  status = DeviceManager::getHalReader(synDeviceGaudi, pHalReader);
    if (status != synSuccess)
    {
        return status;
    }

    uint64_t sramBaseAddr = (uint64_t)pHalReader->getSRAMBaseAddr();
    if (addr >= sramBaseAddr)
    {
        t->setSramOffset(addr);
    }
    else
    {
        t->setDramOffset(addr);
    }

    return synSuccess;
}

bool synSingleton::isTensorValidForGraph(const TensorPtr tensor, uint32_t graphId)
{
    // A persistent tensor has to be attached to a graph - make sure it is the correct graph
    // If new tensor creation API was used, workspace tensors should also have a valid graph id
    if ((tensor->getGraphID() != DUMMY_GRAPH_ID) && (tensor->getGraphID() != graphId))
    {
        LOG_ERR(SYN_API,
                "{}: Tensor {} does not belong to graph Expected :{} Actual :{}",
                HLLOG_FUNC,
                tensor->getName(),
                tensor->getGraphID(),
                graphId);
        return false;
    }

    return true;
}

/*
***************************************************************************************************
*   @brief Unbind & Destroys Tensor
*
*   @param tensor   [in]  Tensor to delete from memory.
***************************************************************************************************
*/
synStatus synSingleton::destroyTensor(synTensor tensor)
{
    VERIFY_IS_NULL_POINTER(SYN_API, tensor, "Tensor");
    TensorPtr tensorPtr;
    // we delay the actual destruction of the Tensor to the unprotected scope
    // once the shared pointer reference count reaches 0.
    {
        std::unique_lock<std::mutex> guard(m_graphsMutex);
        tensorPtr = m_graphEntries.findTensor(tensor);
        if (!tensorPtr)
        {
            LOG_TRACE(SYN_API,
                      "{}: Tensor 0x{:x} not found, it was probably already destroyed",
                      HLLOG_FUNC,
                      (uint64_t)tensor);
            return synSuccess;
        }
        if (!m_graphEntries.destroyTensor(tensor))
        {
            return synFail;
        }
    }
    return synSuccess;
}
