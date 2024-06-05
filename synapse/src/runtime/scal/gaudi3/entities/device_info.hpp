#pragma once

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

#include "runtime/scal/common/entities/device_info_interface.hpp"

namespace gaudi3
{
class DeviceInfo : public common::DeviceInfoInterface
{
public:
    DeviceInfo()          = default;
    virtual ~DeviceInfo() = default;

    // Get SOBJ's Register-Address
    virtual uint64_t getSosAddr(unsigned dcore, unsigned sos) const override;

    virtual uint32_t getSosValue(unsigned value, bool isLongSo, bool isTraceEvict, bool isIncrement) const override;

    // Get Monitor's Register-Value
    virtual uint32_t getMonitorConfigRegisterValue(bool     isLongSob,
                                                   bool     isUpperLongSoBits,
                                                   bool     isCqEnabled,
                                                   unsigned writesNum,
                                                   unsigned sobjGroup) const override;

    virtual uint32_t getMonitorArmRegisterValue(unsigned longSobTargetValue,
                                                bool     isEqualOperation,
                                                bool     maskValue,
                                                unsigned sobjId) const override;

    virtual uint32_t getSchedMonExpFenceCommand(FenceIdType fenceId) const override;

    virtual uint32_t getStaticComputeEcbListBufferSize() const override;

    virtual uint32_t getDynamicComputeEcbListBufferSize() const override;

    virtual uint32_t getQmanEngineGroupTypeCount() const override;

    virtual uint32_t getSchedPdmaCommandsTransferMaxParamCount() const override;
    virtual uint64_t getSchedPdmaCommandsTransferMaxCopySize() const override
    {
        return std::numeric_limits<uint32_t>::max();
    }

    virtual bool isQueueSelectorMaskSupported() const override { return true; };
    virtual uint64_t getQueueSelectorMaskCounterAddress(unsigned smIndex, FenceIdType fenceId) const override;
    virtual uint64_t getQueueSelectorMaskCounterValue(common::QueueSelectorOperation operation,
                                                      unsigned                       value) const override;

    virtual uint64_t getMsixAddrress() const override;
    virtual uint32_t getMsixUnexpectedInterruptValue() const override;

    virtual uint16_t getNumArcCpus() const override;

protected:
    // Get SM-Base address
    virtual uint64_t getSmBase(unsigned dcoreId) const override;
    uint64_t getArcAcpEng(unsigned smIndex) const;

    // Get Monitor-Register's SM-Offset (relative to SM-Base)
    virtual uint64_t getMonitorPayloadAddrHighRegisterSmOffset(MonitorIdType monitorId) const override;
    virtual uint64_t getMonitorPayloadAddrLowRegisterSmOffset(MonitorIdType monitorId) const override;
    virtual uint64_t getMonitorPayloadDataRegisterSmOffset(MonitorIdType monitorId) const override;
    virtual uint64_t getMonitorConfigRegisterSmOffset(MonitorIdType monitorId) const override;
    virtual uint64_t getMonitorArmRegisterSmOffset(MonitorIdType monitorId) const override;
};
}  // namespace gaudi3