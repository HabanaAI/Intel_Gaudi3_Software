#include "scal_monitor.hpp"

#include "defs.h"
#include "synapse_common_types.h"
#include "types.h"

#include "runtime/scal/common/entities/scal_completion_group.hpp"

#include "log_manager.h"

#include "runtime/scal/common/entities/device_info_interface.hpp"

#include "scal.h"

// TODO - utils infra
#define upper_32_bits(n)  ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n)  ((uint32_t)(n))

ScalMonitor::ScalMonitor(MonitorIdType                      id,
                         const common::DeviceInfoInterface& deviceInfoInterface,
                         scal_handle_t                      deviceHandle)
: ScalMonitorBase(id, deviceInfoInterface, deviceHandle)
{
}

synStatus ScalMonitor::init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo)
{
    ScalMonitorBase::init(cgInfo, nullptr);

    if (cgInfo.isDirectMode)
    {
        LOG_ERR(SYN_DEVICE, "{}: error completion-group is not managed by-scheduler", HLLOG_FUNC);
        return synFail;
    }

    m_coreHandle = cgInfo.scheduler_handle;

    return synSuccess;
}

bool ScalMonitor::getPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData, FenceIdType fenceId) const
{
    scal_control_core_infoV2_t coreInfo;

    int rc = scal_control_core_get_infoV2(m_coreHandle, &coreInfo);
    if (rc != 0)
    {
        LOG_ERR(SYN_DEVICE, "{}: error scal_control_core_get_info {}", HLLOG_FUNC, rc);
        return false;
    }

    if (m_deviceInfoInterface.isQueueSelectorMaskSupported())
    {
        payloadAddress = m_deviceInfoInterface.getQueueSelectorMaskCounterAddress(coreInfo.idx, fenceId);
        payloadData    = m_deviceInfoInterface.getQueueSelectorMaskCounterValue(common::QueueSelectorOperation::ADD, 1);
    }
    else
    {
        payloadAddress = coreInfo.dccm_message_queue_address;
        payloadData    = m_deviceInfoInterface.getSchedMonExpFenceCommand(fenceId);
    }

    LOG_TRACE(SYN_DEVICE,
                   "scheduler {} {} payloadAddress {:#x} payloadData {:#x} {}",
                   coreInfo.idx,
                   coreInfo.name,
                   payloadAddress,
                   payloadData,
                   m_deviceInfoInterface.isQueueSelectorMaskSupported());

    return true;
}
