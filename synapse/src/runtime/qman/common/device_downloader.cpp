#include "device_downloader.hpp"
#include "defs.h"
#include "event_triggered_logger.hpp"
#include "global_statistics.hpp"
#include "profiler_api.hpp"
#include "runtime/qman/common/recipe_program_buffer.hpp"
#include "runtime/common/queues/queue_interface.hpp"
#include "habana_global_conf_runtime.h"

DeviceDownloader::DeviceDownloader(QueueInterface& rStreamCopy) : m_rStreamCopy(rStreamCopy) {}

synStatus DeviceDownloader::downloadProgramCodeBuffer(uint64_t               recipeId,
                                                      QueueInterface*        pPreviousStream,
                                                      internalMemcopyParams& rMemcpyParams,
                                                      uint64_t               hostBufferSize) const
{
    LOG_TRACE(SYN_PROG_DWNLD, "{}: Download program code to device", HLLOG_FUNC);

    if (rMemcpyParams.size() == 0)
    {
        LOG_TRACE(SYN_PROG_DWNLD, "Nothing to download");
        return synSuccess;
    }

    PROFILER_COLLECT_TIME()
    STAT_GLBL_START(downloadProgramCodeBlobs);

    SpRecipeProgramBuffer spRecipeProgramCodeBuffer =
        std::make_shared<RecipeProgramBuffer>(recipeId, (char*)rMemcpyParams[0].src, hostBufferSize, false);
    if (GCFG_ENABLE_MAPPING_IN_STREAM_COPY.value())
    {
        spRecipeProgramCodeBuffer->setMappingInformation("program-code");
    }

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES);

    const synStatus status = m_rStreamCopy.memcopy(rMemcpyParams,
                                                   MEMCOPY_HOST_TO_DRAM,
                                                   false,
                                                   pPreviousStream,
                                                   0 /* overrideMemsetVal */,
                                                   true /* inspectCopiedContent */,
                                                   &spRecipeProgramCodeBuffer,
                                                   0);

    ETL_ADD_LOG_TRACE(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                      logId,
                      SYN_PROG_DWNLD,
                      "Downloaded program code (status {})",
                      status);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_PROG_DWNLD, "{}: Failed to copy program-code to device", HLLOG_FUNC);
        STAT_EXIT_NO_COLLECT();
        return status;
    }

    PROFILER_MEASURE_TIME("downloadProgramCodeBlobs")
    STAT_GLBL_COLLECT_TIME(downloadProgramCodeBlobs, globalStatPointsEnum::downloadProgramCodeBlobs);
    return synSuccess;
}

synStatus DeviceDownloader::downloadProgramDataBuffer(QueueInterface*        pPreviousStream,
                                                      internalMemcopyParams& rMemcpyParams,
                                                      SpRecipeProgramBuffer* pRecipeProgramDataBuffer) const
{
    PROFILER_COLLECT_TIME()

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES);

    if ((pRecipeProgramDataBuffer != nullptr) && (GCFG_ENABLE_MAPPING_IN_STREAM_COPY.value()))
    {
        (*pRecipeProgramDataBuffer)->setMappingInformation("program-data");
    }

    const synStatus status = m_rStreamCopy.memcopy(rMemcpyParams,
                                                   MEMCOPY_HOST_TO_DRAM,
                                                   false,
                                                   pPreviousStream,
                                                   0 /* overrideMemsetVal */,
                                                   false /* inspectCopiedContent */,
                                                   pRecipeProgramDataBuffer,
                                                   0);

    ETL_ADD_LOG_T_DEBUG(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                       logId,
                       SYN_PROG_DWNLD,
                       "Downloaded program data (status {})",
                       status);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_PROG_DWNLD, "{}: Failed to copy to device", HLLOG_FUNC);
        return status;
    }

    PROFILER_MEASURE_TIME("downPDToDev")

    return status;
}
