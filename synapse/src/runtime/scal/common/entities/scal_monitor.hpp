#pragma once

#include "scal_monitor_base.hpp"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

namespace common
{
class DeviceInfoInterface;
}

class ScalMonitor : public ScalMonitorBase
{
public:
    ScalMonitor(MonitorIdType                      id,
                const common::DeviceInfoInterface& deviceInfoInterface,
                scal_handle_t                      deviceHandle);

    virtual synStatus init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo) override;

    virtual ~ScalMonitor() = default;

    virtual bool getPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData, FenceIdType fenceId) const override;

protected:
    scal_core_handle_t m_coreHandle;
};