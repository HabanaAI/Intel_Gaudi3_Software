#pragma once

#include "runtime/scal/common/infra/scal_types.hpp"

namespace common
{
enum class QueueSelectorOperation
{
    SET,
    ADD
};

class DeviceInfoInterface
{
public:

    DeviceInfoInterface()          = default;
    virtual ~DeviceInfoInterface() = default;

    // Get SOBJ's Register-Address
    virtual uint64_t getSosAddr(unsigned dcore, unsigned sos) const = 0;

    // Get SOBJ's Register-Value
    virtual uint32_t getSosValue(unsigned value, bool isLongSo, bool isTraceEvict, bool isIncrement) const = 0;

    // Get Monitor-Register's SM-Offset (relative to SM-Base)
    virtual uint64_t getMonitorPayloadAddrHighRegisterSmOffset(MonitorIdType monitorId) const = 0;
    virtual uint64_t getMonitorPayloadAddrLowRegisterSmOffset(MonitorIdType monitorId) const  = 0;
    virtual uint64_t getMonitorPayloadDataRegisterSmOffset(MonitorIdType monitorId) const     = 0;
    virtual uint64_t getMonitorConfigRegisterSmOffset(MonitorIdType monitorId) const          = 0;
    virtual uint64_t getMonitorArmRegisterSmOffset(MonitorIdType monitorId) const             = 0;

    // Get Monitor's Register-Address
    virtual uint64_t getMonitorPayloadAddrHighRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const;
    virtual uint64_t getMonitorPayloadAddrLowRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const;
    virtual uint64_t getMonitorPayloadDataRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const;
    virtual uint64_t getMonitorConfigRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const;
    virtual uint64_t getMonitorArmRegisterAddress(uint64_t dcoreId, MonitorIdType monitorId) const;

    // Get Monitor's Register-Value
    virtual uint32_t getMonitorConfigRegisterValue(bool     isLongSob,
                                                   bool     isUpperLongSoBits,
                                                   bool     isCqEnabled,
                                                   unsigned writesNum,
                                                   unsigned sobjGroup) const = 0;

    virtual uint32_t getMonitorArmRegisterValue(unsigned longSobTargetValue,
                                                bool     isEqualOperation,
                                                bool     maskValue,
                                                unsigned sobjId) const = 0;

    virtual uint32_t getSchedMonExpFenceCommand(FenceIdType fenceId) const = 0;

    virtual bool isQueueSelectorMaskSupported() const = 0;
    virtual uint64_t getQueueSelectorMaskCounterAddress(unsigned smIndex, FenceIdType fenceId) const = 0;
    virtual uint64_t getQueueSelectorMaskCounterValue(QueueSelectorOperation operation, unsigned value) const = 0;

    virtual uint32_t getStaticComputeEcbListBufferSize() const = 0;

    virtual uint32_t getDynamicComputeEcbListBufferSize() const = 0;

    virtual uint32_t getQmanEngineGroupTypeCount() const = 0;

    virtual uint32_t getSchedPdmaCommandsTransferMaxParamCount() const = 0;
    virtual uint64_t getSchedPdmaCommandsTransferMaxCopySize() const   = 0;

    virtual bool isDummyMessageRequiredForMonitorForLongSO() const { return false; };

    virtual uint64_t getMsixAddrress() const = 0;
    virtual uint32_t getMsixUnexpectedInterruptValue() const = 0;

    virtual uint16_t getNumArcCpus() const = 0;

private:
    // Get SM-Base address
    virtual uint64_t getSmBase(unsigned dcoreId) const = 0;
};
}  // namespace common