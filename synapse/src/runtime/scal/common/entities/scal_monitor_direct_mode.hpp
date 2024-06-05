#pragma once

#include "scal_monitor_base.hpp"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

namespace common
{
class DeviceInfoInterface;
}

class ScalMonitorDirectMode : public ScalMonitorBase
{
public:
    ScalMonitorDirectMode(MonitorIdType                      id,
                          const common::DeviceInfoInterface& deviceInfoInterface,
                          scal_handle_t                      deviceHandle);

    virtual ~ScalMonitorDirectMode() = default;

    virtual synStatus init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo) override;

    virtual bool getPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData, FenceIdType fenceId) const override;

private:
    uint64_t m_fenceCounterAddress;
};