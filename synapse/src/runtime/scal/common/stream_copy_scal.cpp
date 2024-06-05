#include "stream_copy_scal.hpp"
#include "defs.h"
#include "device/device_mem_alloc.hpp"
#include "runtime/scal/common/entities/scal_stream_copy_interface.hpp"
#include "scal_event.hpp"
#include "global_statistics.hpp"
#include "profiler_api.hpp"

QueueCopyScal::QueueCopyScal(const BasicQueueInfo&    rBasicQueueInfo,
                             ScalStreamCopyInterface* pScalStream,
                             DevMemoryAllocInterface& rDevMemoryAlloc)
: QueueBaseScalCommon(rBasicQueueInfo, pScalStream), m_rDevMemoryAlloc(rDevMemoryAlloc)
{
}

synStatus QueueCopyScal::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
#ifdef DISABLE_SYNC_ON_DEV
    synchronizeStream(streamHandle);
    return synSuccess;
#endif
    ScalEvent& rScalEvent = dynamic_cast<ScalEvent&>(rEventInterface);
    rScalEvent.clearState();
    rScalEvent.pStreamIfScal = this;

    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->eventRecord(true, rScalEvent);
}

synStatus
QueueCopyScal::eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());

#ifdef DISABLE_SYNC_ON_DEV
    return synSuccess;
#endif

    const ScalEvent& tmpScalEvent = dynamic_cast<const ScalEvent&>(rEventInterface);
    // defend against overriding the event while we're working on it
    ScalEvent rcScalEvent = tmpScalEvent;

    ScalLongSyncObject longSo = rcScalEvent.longSo;
    if (rcScalEvent.isOnHclStream())
    {
        longSo.m_index       = rcScalEvent.hclSyncInfo.long_so_index;
        longSo.m_targetValue = rcScalEvent.hclSyncInfo.targetValue;
    }

    synStatus status;
    {
        std::lock_guard<std::timed_mutex> lock(m_userOpLock);
        status = m_scalStream->longSoWaitOnDevice(longSo, true);
    }

    return status;
}

synStatus QueueCopyScal::query()
{
    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->longSoWaitForLast(true, 0, __FUNCTION__);
}

synStatus QueueCopyScal::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    return waitForLastLongSo(isUserRequest);
}

/*
 ***************************************************************************************************
 *   @brief memcopy() sends memcpy request to device. Input is operation and a vector of requests.
 *          addr in the request is already translated to device memory values.
 *
 *   @param  operation
 *   @param  memcpyParams - a vector of src/dst/size
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueCopyScal::memcopy(internalMemcopyParams& memcpyParams,
                                 internalDmaDir         direction,
                                 bool                   isUserRequest,
                                 QueueInterface*        pPreviousStream,
                                 const uint64_t         overrideMemsetVal,
                                 bool                   inspectCopiedContent,
                                 SpRecipeProgramBuffer* pRecipeProgramBuffer,
                                 uint8_t                apiId)
{
    STAT_GLBL_START(streamCopyMemCopyDuration);

    if (memcpyParams.empty())
    {
        LOG_DEBUG(SYN_STREAM, "{}: Got empty memCopyParams", HLLOG_FUNC);
        return synSuccess;
    }

    LOG_TRACE(SYN_STREAM,
              "memcpy. First src/dst/size {:x}/{:x}/{:x} dir {} userReq {} apiId {}",
              memcpyParams[0].src,
              memcpyParams[0].dst,
              memcpyParams[0].size,
              direction,
              isUserRequest,
              apiId);

    if (pRecipeProgramBuffer != nullptr)
    {
        LOG_ERR(SYN_STREAM, "{} Recipe Program-Buffer activities should have been done by the caller", HLLOG_FUNC);
        return synFail;
    }

    bool hasDataToCopy = false;
    for (auto& single : memcpyParams)
    {
        uint64_t* translateAddr;
        bool      needTranslate;

        if (single.size > 0)
        {
            hasDataToCopy = true;
        }

        switch (direction)
        {
            case MEMCOPY_HOST_TO_DRAM:
            case MEMCOPY_HOST_TO_SRAM:
            {
                translateAddr = &single.src;
                needTranslate = true;
                break;
            }
            case MEMCOPY_DRAM_TO_HOST:
            case MEMCOPY_SRAM_TO_HOST:
            {
                translateAddr = &single.dst;
                needTranslate = true;
                break;
            }
            case MEMCOPY_DRAM_TO_DRAM:
            {
                needTranslate = false;
                break;
            }
            default:
            {
                return synInvalidArgument;
            }
        }

        if (needTranslate)
        {
            uint64_t       virtualAddr     = 0;
            eMappingStatus translateStatus = m_rDevMemoryAlloc.getDeviceVirtualAddress(isUserRequest,
                                                                                       (void*)(*translateAddr),
                                                                                       single.size,
                                                                                       &virtualAddr,
                                                                                       nullptr);

            if (translateStatus != HATVA_MAPPING_STATUS_FOUND)
            {
                LOG_ERR(SYN_STREAM,
                        "Can not translate to addr. Addr {:x} direction {} status {}",
                        *translateAddr,
                        direction,
                        translateStatus);
                return synMappingNotFound;
            }
            *translateAddr = virtualAddr;
        }
    }

    if (!hasDataToCopy)
    {
        LOG_DEBUG(SYN_STREAM, "{}: Got 0 total size", HLLOG_FUNC);
        return synSuccess;
    }

    internalStreamType internalType = getInternalStreamType(direction);

    if (internalType != m_basicQueueInfo.queueType)  // got to the wrong stream
    {
        LOG_ERR_T(SYN_STREAM, "Wrong dma stream, expected {} actual {}", m_basicQueueInfo.queueType, internalType);
        return synInvalidArgument;
    }

    STAT_GLBL_START(streamCopyMutexDuration);
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    STAT_GLBL_COLLECT_TIME(streamCopyMutexDuration, globalStatPointsEnum::streamCopyMutexDuration);
    STAT_GLBL_START(streamCopyOperationDuration);

    ScalLongSyncObject longSo;
    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};

    synStatus status = m_scalStream->memcopy(m_scalStream->getResourceType(),
                                             memcpyParams,
                                             isUserRequest,
                                             true,
                                             apiId,
                                             longSo,
                                             overrideMemsetVal,
                                             memcopySyncInfo);

    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "SCAL memcopy failed with status", status);
        return status;
    }

    ProfilerApi::setHostProfilerApiId(apiId);

    STAT_GLBL_COLLECT_TIME(streamCopyOperationDuration, globalStatPointsEnum::streamCopyOperationDuration);
    STAT_GLBL_COLLECT_TIME(streamCopyMemCopyDuration, globalStatPointsEnum::streamCopyMemCopyDuration);

    return synSuccess;
}

synStatus QueueCopyScal::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                         std::vector<tensor_info_t>& tensorInfoArray) const
{
    LOG_ERR(SYN_API, "Unsupported stream type for getLastTensorArray: {}", getBasicQueueInfo().queueType);
    return synFail;
}

/*
 ***************************************************************************************************
 *   @brief getMappedMemorySize() returns the size of mapped memory used by this stream
 *
 *   The size returned is of the global-hbm memory size (which is always 0), not arc-shared memory
 *
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueCopyScal::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = 0;
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief getDirection() gets the direction (UP/DONW) based on internal dir
 *
 *   @param  internal dir
 *   @return UP/DOWN
 *
 ***************************************************************************************************
 */
internalStreamType QueueCopyScal::getInternalStreamType(internalDmaDir dir)
{
    internalStreamType ret = INTERNAL_STREAM_TYPE_NUM;
    switch (dir)
    {
        case MEMCOPY_HOST_TO_DRAM:
        case MEMCOPY_HOST_TO_SRAM:
        {
            ret = INTERNAL_STREAM_TYPE_DMA_DOWN_USER;
            break;
        }
        case MEMCOPY_DRAM_TO_HOST:
        case MEMCOPY_SRAM_TO_HOST:
        {
            ret = INTERNAL_STREAM_TYPE_DMA_UP;
            break;
        }
        case MEMCOPY_DRAM_TO_DRAM:
        {
            ret = INTERNAL_STREAM_TYPE_DEV_TO_DEV;
            break;
        }
        default:
        {
            HB_ASSERT(false, "direction not supported yet. dir {}", dir);
        }
    }
    return ret;
}