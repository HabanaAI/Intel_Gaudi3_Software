#include "scal_streams_container.hpp"

#include "defs.h"
#include "define_synapse_common.hpp"
#include "device_info_interface.hpp"

#include "log_manager.h"

#include "scal_completion_group.hpp"
#include "scal_completion_group_direct_mode.hpp"
#include "scal_stream_compute.hpp"
#include "scal_streams_fences.hpp"
#include "scal_streams_monitors.hpp"
#include "syn_event_dispatcher.hpp"

#include "utils.h"

#include <memory>

using namespace common;

static const uint32_t completionInitValue = std::numeric_limits<uint32_t>::max();

// Mapping between a cluster's type to its name is expected to be stable
static const std::map<unsigned, std::string> clustersName
{
    {SCAL_MME_COMPUTE_GROUP        , "mme"},
    {SCAL_TPC_COMPUTE_GROUP        , "compute_tpc"},
    {SCAL_EDMA_COMPUTE_GROUP       , "compute_edma"},
    {SCAL_RTR_COMPUTE_GROUP        , "rotator"},
    {SCAL_PDMA_TX_CMD_GROUP        , "pdma_tx"},
    {SCAL_PDMA_TX_DATA_GROUP       , "pdma_tx"},
    {SCAL_PDMA_RX_GROUP            , "pdma_rx"},
    {SCAL_PDMA_DEV2DEV_DEBUG_GROUP , "pdma_dev2dev_debug"},
    {SCAL_PDMA_RX_DEBUG_GROUP      , "pdma_rx_debug"},
    {SCAL_CME_GROUP                , "cme"}
};

// NOTE: changes to the vector below effects test stream_restriction_ASIC
// streamId is used also for streamId to the pdma.
constexpr StreamsSetInfo constStreamsSetsInfo[(uint8_t)ResourceStreamType::AMOUNT] {
    /* USER_DMA_UP,        */ {"pdma_rx", 3},
    /* USER_DMA_DOWN,      */ {"pdma_tx", 3},
    /* USER_DEV_TO_DEV,    */ {"pdma_device2device", 3},
    /* SYNAPSE_DMA_UP,     */ {""},
    /* SYNAPSE_DMA_DOWN,   */ {"pdma_tx_commands", 3},
    /* SYNAPSE_DEV_TO_DEV, */ {""},
    /* COMPUTE,            */ {"compute", 3},
    /* AMOUNT              */
};

ScalStreamsContainer::ScalStreamsContainer(const common::DeviceInfoInterface* deviceInfoInterface)
: m_scalArcFwConfigHandle(nullptr),
  m_apDeviceInfoInterface(deviceInfoInterface)
{
}

ScalStreamsContainer::~ScalStreamsContainer()
{
    for (auto& singleStreamSet : m_streamsResources)
    {
        for (auto& singleResource : singleStreamSet.scalStreamsInfo)
        {
            delete singleResource.pScalCompletionGroup;
            singleResource.pScalCompletionGroup = nullptr;

            delete singleResource.pScalStream;
            singleResource.pScalStream = nullptr;
        }
    }
}

/*
 ***************************************************************************************************
 *   @brief createDeviceStreams() creates all the scal streams including their completion groups.
 *          It also calls a function to populate the table m_clusterTypeCompletionsAmountDB used to know
 *          how many completion signals are in each engine group (needed for sync)
 *
 *   @param  devHndl - device handle
 *   @param  mpHostShared - memory pool (arc shared). Needed by the stream for cyclic buffer
 *   @return status
 *
 *   This function create all the streams based on a pre-defined table streamsResources[]
 *
 ***************************************************************************************************
 */
synStatus ScalStreamsContainer::createStreams(scal_handle_t                devHndl,
                                              ScalMemoryPool&              memoryPoolHostShared,
                                              ScalStreamsMonitors&         streamsMonitors,
                                              ScalStreamsFences&           streamsFences)
{
    LOG_INFO(SYN_STREAM, "");

    if (devHndl == nullptr)
    {
        LOG_ERR_T(SYN_STREAM, "devHndl is nullptr");
        return synFail;
    }

    if (!initStreamsResources(devHndl))
    {
        return synFail;
    }

    std::unique_lock<std::mutex> lck(m_mutex);

    bool res = updateNumOfCompletions(devHndl);
    if (res == false)
    {
        return synFail;
    }

    FenceIdType globalFenceId      = 0;
    bool        isGlobalFenceIdSet = false;
    synStatus   status             = synSuccess;
    for (ResourceStreamType type = ResourceStreamType::FIRST;
         type < ResourceStreamType::AMOUNT;
         type = (ResourceStreamType)((uint32_t)type + 1))
    {
        for (unsigned i = 0; i < getResourcesAmount(type); i++)
        {
            StreamModeType streamModeType = StreamModeType::SCHEDULER;
            if (!getResourcesStreamMode(streamModeType, type))
            {
                return synFail;
            }
            if (!isGlobalFenceIdSet && (streamModeType == StreamModeType::SCHEDULER))
            {
                streamsFences.getFenceId(globalFenceId);
                isGlobalFenceIdSet = true;
            }

            synStatus status = createSingleScalResource(m_streamsResources[(uint8_t)type].scalStreamsInfo[i],
                                                        type,
                                                        streamModeType,
                                                        i,
                                                        devHndl,
                                                        memoryPoolHostShared,
                                                        streamsMonitors,
                                                        streamsFences,
                                                        globalFenceId);
            if (status != synSuccess)
            {
                releaseAllDeviceStreamsL();
                return status;
            }
        }
    }

    if (!createCompoundResources())
    {
        LOG_ERR(SYN_STREAM, "Failed to create compound resources");
        releaseAllDeviceStreamsL();
        return synFail;
    }

    if (getResourcesAmount(ResourceStreamType::COMPUTE) != 0)
    {
        status = initComputeStream();
        if (status != synSuccess)
        {
            releaseAllDeviceStreamsL();
        }
    }

    return status;
}

/*
 ***************************************************************************************************
 *   @brief getFreeStream() finds an unused scal stream from the given type
 *          Called when the user requests a new stream
 *
 *   @param  resourceType - requested resource type
 *   @return stream and complition group. Both are nullptr if not found
 *
 ***************************************************************************************************
 */
bool ScalStreamsContainer::getFreeStream(ResourceStreamType resourceType, StreamAndIndex& streamInfo)
{
    if ((resourceType == ResourceStreamType::SYNAPSE_DMA_UP) ||
        (resourceType == ResourceStreamType::SYNAPSE_DMA_DOWN) ||
        (resourceType == ResourceStreamType::SYNAPSE_DEV_TO_DEV))
    {
        LOG_ERR(SYN_STREAM,
                     "Resource ({}) cannot be acquired seperately from the Compute resource",
                     (uint8_t)resourceType);
        return false;
    }

    std::unique_lock<std::mutex> lck(m_mutex);

    LOG_TRACE(SYN_STREAM, "Checking if any stream (resourceType = {}) is free", (uint8_t)resourceType);

    // Check if we have a free one
    for (unsigned i = 0; i < getResourcesAmount(resourceType); i++)
    {
        if (updateResourceUsage(resourceType, i, true))  // found an unused one
        {
            streamInfo.pStream = getScalStream(resourceType, i);
            streamInfo.idx     = i;
            return true;
        }
    }

    LOG_WARN(SYN_STREAM,
                  "No free streams for type {}. There are {} of this type",
                  (uint8_t)resourceType,
                  getResourcesAmount(resourceType));
    return false;
}

ScalStreamBaseInterface*
ScalStreamsContainer::getFreeComputeResources(const ComputeCompoundResources*& pCompoundResourceInfo)
{
    std::unique_lock<std::mutex> lck(m_mutex);

    LOG_TRACE(SYN_STREAM, "Checking if any compute stream is free");

    // Check if we have a free one
    for (unsigned i = 0; i < getResourcesAmount(ResourceStreamType::COMPUTE); i++)
    {
        if (updateResourceUsage(ResourceStreamType::COMPUTE, i, true)) // found an unused one
        {
            pCompoundResourceInfo = &m_computeCompoundResources[i];
            return getScalStream(ResourceStreamType::COMPUTE, i);
        }
    }

    LOG_WARN(SYN_STREAM,
                  "No free Compute streams. There is a total {} of this type",
                  getResourcesAmount(ResourceStreamType::COMPUTE));
    return nullptr;
}

ScalStreamBaseInterface* ScalStreamsContainer::debugGetCreatedStream(ResourceStreamType type)
{
    return getScalStream(type, 0);
}

ScalStreamBaseInterface*
ScalStreamsContainer::debugGetCreatedComputeResources(const ComputeCompoundResources*& pComputeCompoundResources)
{
    if (getResourcesAmount(ResourceStreamType::COMPUTE) == 0)
    {
        return nullptr;
    }

    pComputeCompoundResources = &m_computeCompoundResources[0];
    return getScalStream(ResourceStreamType::COMPUTE, 0);
}

/*
 ***************************************************************************************************
 *   @brief releaseStream() returns a stream to be unused
 *          Called when the user destroys a stream
 *
 *   @param  scalStream - stream to mark as unused
 *   @return synStatus - fail if scalStream is not found
 *
 ***************************************************************************************************
 */
synStatus ScalStreamsContainer::releaseStream(ScalStreamBaseInterface* pScalStream)
{
    std::unique_lock<std::mutex> lck(m_mutex);

    if (pScalStream == nullptr)  // This can happen if we try to release after an error
    {
        LOG_ERR_T(SYN_STREAM, "Can not release scalStream of nullptr");
        return synSuccess;
    }

    for (ResourceStreamType type = ResourceStreamType::FIRST;
         type < ResourceStreamType::AMOUNT;
         type = (ResourceStreamType)((uint32_t)type + 1))
    {
        for (int i = 0; i < getResourcesAmount(type); i++)
        {
            if (getScalStream(type, i) == pScalStream)
            {
                if (!updateResourceUsage(type, i, false))
                {
                    LOG_ERR(SYN_STREAM, "stream {:x} is already free", TO64(pScalStream));
                    return synFail;
                }

                return synSuccess;
            } // if (found stream)
        } // for(i)
    } // for(type)

    LOG_ERR_T(SYN_API, "scalStream not found {:x}", TO64(pScalStream));
    return synFail;
}

synStatus ScalStreamsContainer::releaseComputeResources(ScalStreamBaseInterface* pComputeScalStream)
{
    std::unique_lock<std::mutex> lck(m_mutex);

    if (pComputeScalStream == nullptr)
    {
        LOG_ERR_T(SYN_STREAM,
                  "Got nullptr for compute resource ({:#x})",
                  TO64(pComputeScalStream));
        return synSuccess;
    }

    bool releaseRxResource      = (getResourcesAmount(ResourceStreamType::SYNAPSE_DMA_UP) != 0);
    bool releaseTxResource      = (getResourcesAmount(ResourceStreamType::SYNAPSE_DMA_DOWN) != 0);
    bool releaseDev2DevResource = (getResourcesAmount(ResourceStreamType::SYNAPSE_DEV_TO_DEV) != 0);

    ResourceStreamType resourceType = ResourceStreamType::COMPUTE;
    for (int i = 0; i < getResourcesAmount(resourceType); i++)
    {
        if (getScalStream(resourceType, i) == pComputeScalStream)
        {
            // "Master" resource release
            if (!updateResourceUsage(resourceType, i, false))
            {
                LOG_ERR(SYN_STREAM, "stream {:x} is already free", TO64(pComputeScalStream));
                return synFail;
            }

            // "Slave" resources are allocated together, and hence no need to check for release failure
            if (releaseRxResource)
            {
                updateResourceUsage(ResourceStreamType::SYNAPSE_DMA_UP, i, false);
            }
            if (releaseTxResource)
            {
                updateResourceUsage(ResourceStreamType::SYNAPSE_DMA_DOWN, i, false);
            }
            if (releaseDev2DevResource)
            {
                updateResourceUsage(ResourceStreamType::SYNAPSE_DEV_TO_DEV, i, false);
            }

            return synSuccess;
        } // if (found stream)
    } // for(i)

    LOG_ERR_T(SYN_API, "Compute scalStream was not found {:x}", TO64(pComputeScalStream));
    return synFail;
}

/*
 ***************************************************************************************************
 *   @brief releaseAllDeviceStreams() takes the look and releases all the streams. This is called
 *          during device release
 *
 *   @param  None
 *   @return synStatus
 *
 ***************************************************************************************************
 */
synStatus ScalStreamsContainer::releaseAllDeviceStreams()
{
    std::unique_lock<std::mutex> lck(m_mutex);
    return releaseAllDeviceStreamsL();
}

unsigned ScalStreamsContainer::getResourcesAmount(ResourceStreamType type)
{
    if (type >= ResourceStreamType::AMOUNT)
    {
        LOG_ERR(SYN_STREAM, "Invalid resource-type {}", (uint8_t)type);
        return 0;
    }

    return m_streamsResources[(uint8_t)type].scalStreamsInfo.size();
}

bool ScalStreamsContainer::getResourcesStreamMode(StreamModeType& streamModeType, ResourceStreamType type)
{
    if (type >= ResourceStreamType::AMOUNT)
    {
        LOG_ERR(SYN_STREAM, "Invalid resource-type {}", (uint8_t)type);
        return false;
    }

    streamModeType = m_streamsResources[(uint8_t)type].streamModeType;
    return true;
}

ScalStreamBaseInterface* ScalStreamsContainer::createStream(scal_handle_t            devHndl,
                                                            ScalMemoryPool&          memoryPoolHostShared,
                                                            ScalStreamsMonitors&     streamsMonitors,
                                                            ScalStreamsFences&       streamsFences,
                                                            StreamModeType           queueType,
                                                            ResourceStreamType       resourceType,
                                                            unsigned                 streamIndex,
                                                            const std::string&       streamName,
                                                            ScalCompletionGroupBase* pCompletionGroup,
                                                            FenceIdType              globalFenceId) const
{
    uint64_t syncMonitorId = 0;
    synStatus status = streamsMonitors.getMonitorId(syncMonitorId);
    if (status != synSuccess)
    {
        return nullptr;
    }

    ScalStreamBaseInterface* pStream = nullptr;

    switch (queueType)
    {
        case StreamModeType::SCHEDULER:
        {
            FenceIdType fenceId           = -1;
            FenceIdType fenceIdForCompute = -1;
            streamsFences.getFenceId(fenceId);

            if (resourceType == ResourceStreamType::COMPUTE)
            {
                if (m_streamsResources[(uint8_t)ResourceStreamType::SYNAPSE_DMA_DOWN].scalStreamsInfo.size() != 0)
                {
                    streamsFences.getFenceId(fenceIdForCompute);
                }
                else
                {
                    fenceIdForCompute = fenceId;
                }
            }

            ScalStreamCtorInfo scalStreamCtorInfo {{.name                 = streamName,
                                                    .mpHostShared         = memoryPoolHostShared,
                                                    .devHndl              = devHndl,
                                                    .devStreamInfo        = nullptr,
                                                    .deviceInfoInterface  = nullptr,
                                                    .pScalCompletionGroup = pCompletionGroup,
                                                    .devType              = synDeviceTypeInvalid,
                                                    .fenceId              = fenceId,
                                                    .fenceIdForCompute    = fenceIdForCompute,
                                                    .streamIdx            = streamIndex,
                                                    .syncMonitorId        = (unsigned)syncMonitorId,
                                                    .resourceType         = resourceType},
                                                   .globalFenceId = globalFenceId};

            if (resourceType == ResourceStreamType::COMPUTE)
            {
                pStream = createComputeStream(&scalStreamCtorInfo);
            }
            else
            {
                pStream = createCopySchedulerModeStream(&scalStreamCtorInfo);
            }

        }
        break;

        case StreamModeType::DIRECT:
        {
            ScalStreamCtorInfoBase scalStreamCtorInfo {.name                 = streamName,
                                                       .mpHostShared         = memoryPoolHostShared,
                                                       .devHndl              = devHndl,
                                                       .devStreamInfo        = nullptr,
                                                       .deviceInfoInterface  = nullptr,
                                                       .pScalCompletionGroup = pCompletionGroup,
                                                       .devType              = synDeviceTypeInvalid,
                                                       .fenceId              = 0,
                                                       .fenceIdForCompute    = 0,
                                                       .streamIdx            = streamIndex,
                                                       .syncMonitorId        = (unsigned)syncMonitorId,
                                                       .resourceType         = resourceType};
            pStream = createCopyDirectModeStream(&scalStreamCtorInfo);
        }
        break;
    }

    return pStream;
}

ScalCompletionGroupBase* ScalStreamsContainer::createCompletionGroup(scal_handle_t      devHndl,
                                                                     StreamModeType     queueType,
                                                                     const std::string& completionGroupName) const
{
    ScalCompletionGroupBase* pCompletionGroup = nullptr;

    switch (queueType)
    {
        case StreamModeType::SCHEDULER:
            pCompletionGroup = new ScalCompletionGroup(devHndl,
                                                       completionGroupName);
        break;

        case StreamModeType::DIRECT:
            pCompletionGroup = new ScalCompletionGroupDirectMode(devHndl,
                                                                 completionGroupName,
                                                                 m_apDeviceInfoInterface);
        break;
    }

    return pCompletionGroup;
}

/*
 *******************************************************************************************************************
 *   @brief updateNumOfCompletions() fills a table with the number of completion signals per engine group
 *
 *   @param  devHandle - device handle
 *   @return bool - true - OK, false - failed
 *
 *******************************************************************************************************************
 */
bool ScalStreamsContainer::updateNumOfCompletions(scal_handle_t devHandle)
{
    m_deviceStreamInfo.clusterTypeCompletionsAmountDB.fill(completionInitValue);

    const auto& resourcesClusters = getResourcesClusters();

    for (ResourceStreamType resourceType = ResourceStreamType::FIRST;
         resourceType < ResourceStreamType::AMOUNT;
         resourceType = (ResourceStreamType)((uint32_t)resourceType + 1))
    {
        // Ignore resources which have a non Scheduler-Mode streams
        StreamModeType streamModeType = StreamModeType::SCHEDULER;
        if ((getResourcesAmount(resourceType) != 0) &&
            ((getResourcesStreamMode(streamModeType, resourceType) == false) ||
             (streamModeType != StreamModeType::SCHEDULER)))
        {
            continue;
        }

        const ResourceClusters& currentResourceClusters = resourcesClusters.at(resourceType);
        // 1) check the primary clusters' set (of current resource)
        const std::vector<unsigned>* pResourceClustersType = &currentResourceClusters.primaryClusterType;
        bool status = updateClustersSetNumOfCompletion(devHandle, *pResourceClustersType);

        // 2) In case primary clusters' set failed, try the secondary
        if (!status)
        {
            LOG_DEBUG(SYN_STREAM, "Secondary clusters group is being used");

            pResourceClustersType = &currentResourceClusters.secondaryClusterType;
            status = updateClustersSetNumOfCompletion(devHandle, *pResourceClustersType);
        }

        // 3) In case both primary and secondary failed, abort
        if (!status)
        {
            return false;
        }

        // Set resource information
        setResourceInfo(resourceType, *pResourceClustersType);
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief releaseAllDeviceStreamsL() removes all the scal streams and completion groups
 *          Called when releasing the device or if there was any error during createDeviceStreams
 *
 *   @param  None
 *   @return synStatus
 *
 *   NOTE: it is assumed that all the streams are idle. This is done by waiting for the stream to be
 *         idle before destroying the user stream
 *
 ***************************************************************************************************
 */
synStatus ScalStreamsContainer::releaseAllDeviceStreamsL()
{
    synStatus returnedStatus = synSuccess;

    for (ResourceStreamType resourceType = ResourceStreamType::FIRST;
         resourceType < ResourceStreamType::AMOUNT;
         resourceType = (ResourceStreamType)((uint32_t)resourceType + 1))
    {
        for (unsigned i = 0; i < getResourcesAmount(resourceType); i++)
        {
            SingleScalStreamInfo& scalStreamInfo = m_streamsResources[(uint8_t)resourceType].scalStreamsInfo[i];

            delete scalStreamInfo.pScalStream;
            scalStreamInfo.pScalStream = nullptr;

            delete scalStreamInfo.pScalCompletionGroup;
            scalStreamInfo.pScalCompletionGroup = nullptr;
        } // for(i)
    } // for(resourceType)

    return returnedStatus;
}

bool ScalStreamsContainer::getStreamSetInfo(const StreamsSetInfo*& pStreamSetInfo,
                                            ResourceStreamType     resourceType) const
{
    if (resourceType >= ResourceStreamType::AMOUNT)
    {
        LOG_ERR(SYN_STREAM, "Invalid resource-type {}", (uint8_t)resourceType);
        return false;
    }

    pStreamSetInfo = &constStreamsSetsInfo[(uint8_t)resourceType];
    return true;
}

bool ScalStreamsContainer::createStreamResource(const std::string& streamSetName,
                                                ResourceStreamType resourceType,
                                                unsigned           streamIndex)
{
    SingleScalStreamInfo scalStreamInfo {.streamName = fmt::format("{}{}", streamSetName, streamIndex),
                                         .completionName =
                                             fmt::format("{}_completion_queue{}", streamSetName, streamIndex),
                                         .streamIdx = streamIndex};

    m_streamsResources[(uint8_t)resourceType].scalStreamsInfo.push_back(scalStreamInfo);

    return true;
}

bool ScalStreamsContainer::updateResourceUsage(ResourceStreamType type, unsigned index, bool isUsed)
{
    if (!isValidStreamSetIndex(type, index))
    {
        return false;
    }

    if (m_streamsResources[(uint8_t)type].scalStreamsInfo[index].used == isUsed)
    {
        return false;
    }

    m_streamsResources[(uint8_t)type].scalStreamsInfo[index].used = isUsed;
    return true;
}

ScalStreamBaseInterface* ScalStreamsContainer::getScalStream(ResourceStreamType type, unsigned index)
{
    if (!isValidStreamSetIndex(type, index))
    {
        return nullptr;
    }

    return m_streamsResources[(uint8_t)type].scalStreamsInfo[index].pScalStream;
}

synStatus ScalStreamsContainer::initComputeStream()
{
    // init the global fence
    if (m_streamsResources[(uint8_t)ResourceStreamType::COMPUTE].scalStreamsInfo.size() != 0)
    {
        SingleResourceTypeInfo& computeResource      = m_streamsResources[(uint8_t)ResourceStreamType::COMPUTE];
        SingleScalStreamInfo&   computeFirstResource = computeResource.scalStreamsInfo[0];
        if (computeResource.streamModeType != StreamModeType::SCHEDULER)
        {
            LOG_ERR(SYN_STREAM, "Compute stream is not a direct-mode stream");
            return synFail;
        }

        ScalStreamCompute* pStream = reinterpret_cast<ScalStreamCompute*>(computeFirstResource.pScalStream);
        if (pStream != nullptr)
        {
            // we must send this command immidiately (==> send = true !!!)
            // since we don't know if the first compute stream
            // will send any other command later, this stream could be idle
            pStream->addGlobalFenceInc(true);
        }
    }
    return synSuccess;
}

bool ScalStreamsContainer::initStreamsResources(scal_handle_t& devHndl)
{
    for (ResourceStreamType resourceType = ResourceStreamType::FIRST;
         resourceType < ResourceStreamType::AMOUNT;
         resourceType = (ResourceStreamType)((uint32_t)resourceType + 1))
    {
        const StreamsSetInfo* pStreamSetInfo;
        bool status = getStreamSetInfo(pStreamSetInfo, resourceType);
        if ((!status) || (pStreamSetInfo == nullptr))
        {
            return false;
        }

        if (pStreamSetInfo->m_streamsSetName != "")
        {
            const std::string streamName {pStreamSetInfo->m_streamsSetName};

            unsigned       streamsAmount  = 0;
            StreamModeType streamModeType = StreamModeType::SCHEDULER;
            if (!initSingleStreamsSetResources(streamsAmount, streamModeType, streamName, devHndl))
            {
                return false;
            }

            if (streamsAmount == 0)
            {
                // SYN_DMA_DOWN had been a Schduler-Mode cluster, and will become a Direct-Mode stream
                // Hence, it is expected
                // In case we will fail to create any of the two, createStreams will fail
                if (resourceType != ResourceStreamType::SYNAPSE_DMA_DOWN)
                {
                    LOG_ERR(SYN_STREAM, "Failed to get StreamSet handle for name {}", pStreamSetInfo->m_streamsSetName);
                }

                continue;
            }

            m_streamsResources[(uint8_t)resourceType].streamModeType = streamModeType;
            if (pStreamSetInfo->m_amount > streamsAmount)
            {
                LOG_DEBUG(SYN_STREAM, "Streams-set {} supports fewer streams ({}) than required ({})",
                               pStreamSetInfo->m_streamsSetName, streamsAmount, pStreamSetInfo->m_amount);
            }
            else
            {
                streamsAmount = pStreamSetInfo->m_amount;
            }

            for (unsigned streamIndex = 0; streamIndex < streamsAmount; streamIndex++)
            {
                if (!createStreamResource(streamName, resourceType, streamIndex))
                {
                    return false;
                }
            }
        }
    }

    return true;
}

bool ScalStreamsContainer::initSingleStreamsSetResources(unsigned&          streamsAmount,
                                                         StreamModeType&    streamModeType,
                                                         const std::string& streamSetName,
                                                         scal_handle_t&     devHndl)
{
    scal_streamset_handle_t streamSetHandle;
    int rtn = scal_get_streamset_handle_by_name(devHndl, streamSetName.c_str(), &streamSetHandle);
    if (rtn != SCAL_SUCCESS)
    {
        streamsAmount = 0;
        LOG_DEBUG(SYN_STREAM, "Failed to get StreamSet handle for name {}", streamSetName);
        return true;
    }

    scal_streamset_info_t streamSetInfo;
    rtn = scal_streamset_get_info(streamSetHandle, &streamSetInfo);
    if (rtn != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "Failed to get info for StreamSet {}", streamSetName);
        return false;
    }

    streamModeType = streamSetInfo.isDirectMode ? StreamModeType::DIRECT : StreamModeType::SCHEDULER;
    streamsAmount  = streamSetInfo.streamsAmount;

    return true;
}

synStatus ScalStreamsContainer::createSingleScalResource(SingleScalStreamInfo&        scalStreamInfo,
                                                         ResourceStreamType           resourceType,
                                                         StreamModeType               streamModeType,
                                                         unsigned                     streamIndex,
                                                         scal_handle_t                devHndl,
                                                         ScalMemoryPool&              memoryPoolHostShared,
                                                         ScalStreamsMonitors&         streamsMonitors,
                                                         ScalStreamsFences&           streamsFences,
                                                         FenceIdType                  globalFenceId)
{
    const std::string& streamName          = scalStreamInfo.streamName;
    const std::string& completionGroupName = scalStreamInfo.completionName;

    // create completion-group
    ScalCompletionGroupBase* pCompletionGroup = createCompletionGroup(devHndl, streamModeType, completionGroupName);
    if (pCompletionGroup == nullptr)
    {
        LOG_ERR(SYN_STREAM, "Failed to create completion-group {}", completionGroupName);
        return synFail;
    }
    //
    synStatus status = pCompletionGroup->init();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Failed to init completion-group {:x} status {}", TO64(pCompletionGroup), status);
        return status;
    }

    // create stream
    ScalStreamBaseInterface* pStream = createStream(devHndl,
                                                    memoryPoolHostShared,
                                                    streamsMonitors,
                                                    streamsFences,
                                                    streamModeType,
                                                    resourceType,
                                                    streamIndex,
                                                    streamName,
                                                    pCompletionGroup,
                                                    globalFenceId);
    if (pStream == nullptr)
    {
        LOG_ERR(SYN_STREAM, "Failed to create scal-stream {}", streamName);
        return synFail;
    }

    ScalStreamBase* pStreamBase = dynamic_cast<ScalStreamBase*>(pStream);
    if (pStreamBase == nullptr)
    {
        LOG_ERR(SYN_STREAM, "Failed to create scal-stream {}", streamName);
        return synFail;
    }

    status = pStreamBase->init();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "Failed to init scalStream {:x} status {}", TO64(pStream), status);
    }

    scalStreamInfo.pScalStream          = pStream;
    scalStreamInfo.pScalCompletionGroup = pCompletionGroup;

    if ((streamIndex == 0) && (pCompletionGroup->isForceOrdered()))
    {
        m_deviceStreamInfo.resourcesInfo[(uint8_t)resourceType].targetVal++;
    }

    return status;
}

bool ScalStreamsContainer::isValidStreamSetIndex(ResourceStreamType type, unsigned index)
{
    if (type >= ResourceStreamType::AMOUNT)
    {
        LOG_ERR(SYN_STREAM, "Invalid resource-type {}", (uint8_t)type);
        return false;
    }

    if (index < m_streamsResources[(uint8_t)type].scalStreamsInfo.size())
    {
        return true;
    }

    LOG_ERR(SYN_STREAM, "Invalid stream-index {} for resource-type {}", index, (uint8_t)type);
    return false;
}

bool ScalStreamsContainer::updateClustersSetNumOfCompletion(scal_handle_t devHandle,
                                                            const std::vector<unsigned>& clustersSet)
{
    // PDMA requires to be defined
    bool shouldAbortOnFailure = (clustersSet.size() == 1);

    for (const unsigned& clusterType : clustersSet)
    {
        // In case info for given cluster already set, continue
        if (m_deviceStreamInfo.clusterTypeCompletionsAmountDB[clusterType] != completionInitValue)
        {
            continue;
        }

        if (clusterType >= clustersName.size())
        {
            LOG_ERR(SYN_STREAM, "Invalid clusterType {}", clusterType);
            return false;
        }

        const std::string& clusterName = clustersName.at(clusterType);
        int rtn = updateSingleClustersNumOfCompletions(devHandle, clusterType, clusterName);
        if ((rtn != SCAL_SUCCESS) &&
            ((shouldAbortOnFailure) || (rtn != SCAL_NOT_FOUND)))
        {
            return false;
        }
    }

    return true;
}

int ScalStreamsContainer::updateSingleClustersNumOfCompletions(scal_handle_t      devHandle,
                                                               uint32_t           clusterType,
                                                               const std::string& clusterName)
{
    if (clusterType >= m_deviceStreamInfo.clusterTypeCompletionsAmountDB.size())
    {
        LOG_ERR(SYN_STREAM, "Invalid clusterType {}", clusterType);
        return SCAL_FAILURE;
    }
    scal_cluster_handle_t cluster;
    int rtn = scal_get_cluster_handle_by_name(devHandle, clusterName.c_str(), &cluster);
    if (rtn != SCAL_SUCCESS)
    {
        if (rtn == SCAL_NOT_FOUND)
        {
            LOG_DEBUG(SYN_STREAM, "Cluster {} was not found", clusterName);
        }
        else
        {
            LOG_INFO(SYN_STREAM, "Failed to get cluster handle for name {}", clusterName);
        }

        return rtn;
    }

    scal_cluster_info_t info;
    rtn = scal_cluster_get_info(cluster, &info);
    if (rtn != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "Failed to get cluster info for name {}", clusterName);
        return rtn;
    }

    m_deviceStreamInfo.clusterTypeCompletionsAmountDB[clusterType] = info.numCompletions;
    LOG_INFO(SYN_STREAM,
                  "clusterType {} engine grp {} has {} engines",
                  clusterType,
                  clusterName,
                  m_deviceStreamInfo.clusterTypeCompletionsAmountDB[clusterType]);

    return rtn;
}

void ScalStreamsContainer::setResourceInfo(ResourceStreamType           resourceType,
                                           const std::vector<unsigned>& resourceClustersType)
{
    auto& completions = m_deviceStreamInfo.clusterTypeCompletionsAmountDB;

    EngineGrpArr& engineGrpArr = m_deviceStreamInfo.resourcesInfo[(uint8_t)resourceType].engineGrpArr;
    uint32_t&     targetVal    = m_deviceStreamInfo.resourcesInfo[(uint8_t)resourceType].targetVal;

    engineGrpArr.numEngineGroups = 0;
    targetVal = 0;
    for (auto singleClusterType : resourceClustersType)
    {
        if (completions[singleClusterType] != -1)
        {
            if (engineGrpArr.numEngineGroups >= ARRAY_SIZE(engineGrpArr.eng))
            {
                LOG_ERR(SYN_STREAM,
                        "{}: Number of completions exceeded {} for resourceType {}",
                        HLLOG_FUNC,
                        ARRAY_SIZE(engineGrpArr.eng),
                        resourceType);
                break;
            }

            engineGrpArr.eng[engineGrpArr.numEngineGroups++] = singleClusterType;
            targetVal += completions[singleClusterType];
        }
    }
    if (targetVal == 0)
    {
        targetVal = -1; // if target is not set, set it to (-1)
    }

    LOG_INFO(SYN_STREAM,
             "set resourceType {} numEngineGroups {} Engines {:#x} {:#x} {:#x} {:#x} target {:#x}",
             resourceType,
             engineGrpArr.numEngineGroups,
             engineGrpArr.eng[0],
             engineGrpArr.eng[1],
             engineGrpArr.eng[2],
             engineGrpArr.eng[3],
             targetVal);
}

bool ScalStreamsContainer::createCompoundResources()
{
    bool acquireRxResource      = (getResourcesAmount(ResourceStreamType::SYNAPSE_DMA_UP) != 0);
    bool acquireTxResource      = (getResourcesAmount(ResourceStreamType::SYNAPSE_DMA_DOWN) != 0);
    bool acquireDev2DevResource = (getResourcesAmount(ResourceStreamType::SYNAPSE_DEV_TO_DEV) != 0);

    unsigned computeResourceAmount = getResourcesAmount(ResourceStreamType::COMPUTE);
    if (GCFG_GAUDI3_SINGLE_DIE_CHIP.value())
    {
        // on Gaudi3 single die, a special json override is used, that deletes some pdma channels (so all fit in 1 die)
        // compute resources must also be aligned
        computeResourceAmount = getResourcesAmount(ResourceStreamType::SYNAPSE_DMA_DOWN);
    }
    m_computeCompoundResources.resize(computeResourceAmount);

    for (unsigned streamIndex = 0; streamIndex < computeResourceAmount; streamIndex++)
    {
        ScalStreamBaseInterface* pComputeStream = getScalStream(ResourceStreamType::COMPUTE, streamIndex);
        ScalStreamBaseInterface* pRxCommandsStream =
            acquireRxResource ? getScalStream(ResourceStreamType::SYNAPSE_DMA_UP, streamIndex) : pComputeStream;
        ScalStreamBaseInterface* pTxCommandsStream =
            acquireTxResource ? getScalStream(ResourceStreamType::SYNAPSE_DMA_DOWN, streamIndex) : pComputeStream;
        ScalStreamBaseInterface* pDev2DevCommandsStream =
            acquireDev2DevResource ? getScalStream(ResourceStreamType::SYNAPSE_DEV_TO_DEV, streamIndex)
                                   : pComputeStream;

        if ((pComputeStream == nullptr) ||
            (pRxCommandsStream == nullptr) ||
            (pTxCommandsStream == nullptr) ||
            (pDev2DevCommandsStream == nullptr))
        {
            LOG_ERR(SYN_STREAM, "Mismatch between resource amount and actual amount of resources");
            releaseAllDeviceStreamsL();
            return false;
        }

        m_computeCompoundResources[streamIndex].m_pRxCommandsStream = (ScalStreamCopyInterface*)pRxCommandsStream;
        m_computeCompoundResources[streamIndex].m_pTxCommandsStream = (ScalStreamCopyInterface*)pTxCommandsStream;
        m_computeCompoundResources[streamIndex].m_pDev2DevCommandsStream =
            (ScalStreamCopyInterface*)pDev2DevCommandsStream;
        m_computeCompoundResources[streamIndex].m_streamIndex            = streamIndex;
    }

    return true;
}
