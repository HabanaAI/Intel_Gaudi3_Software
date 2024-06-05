#include "queue_compute_utils.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "log_manager.h"
#include "defenders.h"
#include "global_statistics.hpp"
#include "profiler_api.hpp"
#include "utils.h"
#include "basic_queue_info.hpp"

std::atomic<uint64_t> QueueComputeUtils::launchSeqId;

// 128-Bytes alignment required for the data for best-performance
const uint64_t QueueComputeUtils::REQUIRED_DATA_BITS_ALIGNMENT = 7;
const uint64_t QueueComputeUtils::REQUIRED_DATA_ALIGNMENT      = (1 << REQUIRED_DATA_BITS_ALIGNMENT);

synStatus QueueComputeUtils::prepareLaunch(const BasicQueueInfo&         rBasicStreamHandle,
                                           const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                           const uint32_t                enqueueTensorsAmount,
                                           const synRecipeHandle         pRecipeHandle,
                                           EventWithMappedTensorDB&      events,
                                           uint32_t                      flags)
{
    STAT_GLBL_START(devicePrepareLaunch);

    if (flags & SYN_FLAGS_TENSOR_NAME)
    {
        RecipeTensorsInfo& recipeTensorsInfo = pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeTensorInfo;
        HB_ASSERT(recipeTensorsInfo.m_isTensorName2idxInit == true, "m_tensorName2idxMap is not initalized");

        for (uint32_t tensorIndex = 0; tensorIndex < enqueueTensorsAmount; tensorIndex++)
        {
            synLaunchTensorInfoExt* enqueueTensorsInfoWithIds = const_cast<synLaunchTensorInfoExt*>(enqueueTensorsInfo);
            synStatus            status = recipeTensorsInfo.tensorRetrieveId(enqueueTensorsInfo[tensorIndex].tensorName,
                                                                  &enqueueTensorsInfoWithIds[tensorIndex].tensorId);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_RECIPE,
                        "{}: tensorRetrieveId for tensorName {} failed",
                        HLLOG_FUNC,
                        enqueueTensorsInfo->tensorName);
                return synInvalidArgument;
            }
        }
    }

    if (enqueueTensorsAmount != 0)
    {
        VERIFY_IS_NULL_POINTER(SYN_API, enqueueTensorsInfo, "enqueueTensorsInfo");
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(SYN_API))
    {
        logLaunchTensors(enqueueTensorsInfo, enqueueTensorsAmount, HLLOG_LEVEL_INFO, flags);
    }

    LOG_TRACE_T(SYN_API,
                "{} stream {} recipe id 0x{:x}",
                HLLOG_FUNC,
                rBasicStreamHandle.getDescription(),
                pRecipeHandle->recipeSeqNum);

    if (rBasicStreamHandle.queueType != INTERNAL_STREAM_TYPE_COMPUTE)
    {
        LOG_WARN(SYN_API, "{}: used wrong stream type {}", HLLOG_FUNC, rBasicStreamHandle.queueType);
        return synInvalidArgument;
    }

    if (LOG_LEVEL_AT_LEAST_TRACE(SYN_API))
    {
        for (auto const& event : events)
        {
            LOG_TRACE_T(SYN_API, "{}: enqueue with event {}", HLLOG_FUNC, event->toString());
        }
    }

    STAT_GLBL_COLLECT_TIME(devicePrepareLaunch, globalStatPointsEnum::devicePrepareLaunch);
    return synSuccess;
}

uint64_t QueueComputeUtils::getAlignedWorkspaceAddress(uint64_t workspaceAddress)
{
    return round_to_multiple(workspaceAddress, REQUIRED_DATA_ALIGNMENT);
}

bool QueueComputeUtils::isRecipeEmpty(InternalRecipeHandle& rRecipeHandle)
{
    return (rRecipeHandle.basicRecipeHandle.recipe->node_nr == 0 || rRecipeHandle.basicRecipeHandle.recipe->blobs_nr == 0);
}

void QueueComputeUtils::logLaunchTensors(const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                         const uint32_t                enqueueTensorsAmount,
                                         int                           logLevel,
                                         uint32_t                      flags)
{
    for (uint32_t i = 0; i < enqueueTensorsAmount; i++)
    {
        const synLaunchTensorInfoExt& info          = enqueueTensorsInfo[i];
        bool                          enqueueByName = flags & SYN_FLAGS_TENSOR_NAME;

        HLLOG_TYPED(SYN_API, logLevel,
                    "Tensor {} id {:x} name {} isEnqueueByName {} type {} addr 0x{:x} sizes 0x{:x} 0x{:x} 0x{:x} "
                    "0x{:x} 0x{:x}",
                    i,
                    info.tensorId,
                    (info.tensorName) ? info.tensorName : "",
                    enqueueByName,
                    synTensorType2Txt(info.tensorType),
                    info.pTensorAddress,
                    info.tensorSize[0],
                    info.tensorSize[1],
                    info.tensorSize[2],
                    info.tensorSize[3],
                    info.tensorSize[4]);
    }
}
