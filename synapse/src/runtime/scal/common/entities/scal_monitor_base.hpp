#pragma once

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

struct ScalLongSyncObject;

namespace common
{
class DeviceInfoInterface;
}

class ScalMonitorBase
{
public:
    ScalMonitorBase(MonitorIdType                      id,
                    const common::DeviceInfoInterface& deviceInfoInterface,
                    scal_handle_t                      deviceHandle);

    virtual ~ScalMonitorBase() = default;

    virtual synStatus init(const scal_completion_group_infoV2_t& cgInfo, scal_stream_info_t* pStreamInfo);

    bool getConfigRegsForLongSO(FenceIdType           fenceId,
                                uint8_t&              numRegs,
                                MonitorAddressesType& addr,
                                MonitorValuesType&    value) const;

    void getArmRegsForLongSO(const ScalLongSyncObject& rLongSo,
                             const FenceIdType         fenceId,
                             uint8_t&                  numRegs,
                             MonitorAddressesType&     addr,
                             MonitorValuesType&        value);

    virtual bool getPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData, FenceIdType fenceId) const = 0;

    uint64_t getSyncMgrId() { return m_syncMgrId; };

protected:
    uint64_t m_syncMgrId;

    const common::DeviceInfoInterface& m_deviceInfoInterface;

private:
    MonitorIdType m_monitorId;
    unsigned      m_soGroupIdx;
    scal_handle_t m_deviceHandle;
    uint64_t      m_lastLongSOtargetValue;
};