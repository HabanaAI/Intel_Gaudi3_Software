#include "scal_monitor_direct_mode.hpp"

#include "platform/gaudi2/utils.cpp"

// #include "defs.h"
// #include "synapse_common_types.h"
// #include "types.h"

// #include "runtime/scal/common/entities/scal_completion_group.hpp"

#include "runtime/scal/common/entities/device_info_interface.hpp"

#include "log_manager.h"

#include "scal.h"

ScalMonitorDirectMode::ScalMonitorDirectMode(MonitorIdType                      id,
                                             const common::DeviceInfoInterface& deviceInfoInterface,
                                             scal_handle_t                      deviceHandle)
: ScalMonitorBase(id, deviceInfoInterface, deviceHandle), m_fenceCounterAddress(0)
{
}

synStatus ScalMonitorDirectMode::init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo)
{
    if (pStreamInfo == nullptr)
    {
        LOG_ERR(SYN_DM_STREAM, "{}: error got nullptr for stream-info", HLLOG_FUNC);
        return synFail;
    }

    ScalMonitorBase::init(cgInfo, pStreamInfo);

    if (!cgInfo.isDirectMode)
    {
        LOG_ERR(SYN_DM_STREAM, "{}: error completion-group is not managed directly", HLLOG_FUNC);
        return synFail;
    }

    m_fenceCounterAddress = pStreamInfo->fenceCounterAddress;

    return synSuccess;
}

bool ScalMonitorDirectMode::getPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData, FenceIdType fenceId) const
{
    union sync_object_update
    {
        struct
        {
            uint32_t value    :16;
            uint32_t reserved :13;
            uint32_t mode     :3;
        } so_update;
        uint32_t raw;
    };

    // Increments SOBJ (Fence) by 1
    sync_object_update sobjUpdate;
    sobjUpdate.raw = 0;
    //
    sobjUpdate.so_update.value = 1;
    sobjUpdate.so_update.mode  = 1; // add operation

    payloadAddress = m_fenceCounterAddress;
    payloadData    = sobjUpdate.raw;

    return true;
}
