#pragma once

#include "graph_compiler/queue_command_factory.h"

namespace gaudi2
{

class QueueCommandFactory : public ::QueueCommandFactory
{
public:
    static QueueCommandFactory& instance();

    virtual ~QueueCommandFactory();

    virtual QueueCommandPtr getDmaDramToSram(deviceAddrOffset  dramPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID) const override;

    virtual QueueCommandPtr getDmaSramToDram(deviceAddrOffset  dramPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID) const override;

    virtual QueueCommandPtr getCpDma(deviceAddrOffset  addrOffset,
                                     uint64_t          size,
                                     uint32_t          predicate = 0) const override;

    virtual QueueCommandPtr getLoadDesc(void*             desc,
                                        unsigned          descSize,
                                        unsigned          descOffset,
                                        HabanaDeviceType  device,
                                        unsigned          deviceID = 0,
                                        uint32_t          predicate = 0) const override;

    virtual QueueCommandPtr getWriteManyRegisters(unsigned        firstRegOffset,
                                                  unsigned        count,
                                                  const uint32_t* values,
                                                  uint32_t        predicate = 0) const override;

    virtual QueueCommandPtr getExecute(HabanaDeviceType type,
                                       unsigned         deviceID  = 0,
                                       uint32_t         predicate = 0,
                                       uint32_t         value     = 0x1) const override;

    virtual QueueCommandPtr getSfgInc(unsigned sigOutValue) const override;

    virtual QueueCommandPtr getSfgInit(unsigned sigOutValue) const override;

    virtual QueueCommandPtr getResetSobs(unsigned target, unsigned totalNumEngs) const override;

    virtual QueueCommandPtr getDynamicExecute(QueueCommandPtr execute, std::vector<QueueCommandPtr>& signalSemaphore, BypassType enableBypass) const override;

    virtual QueueCommandPtr getSetupAndArmCommand(std::vector<QueueCommandPtr>& setupAndArm) const override;

    virtual QueueCommandPtr getMonitorArm(SyncObjectManager::SyncId  syncObj,
                                          SyncObjectManager::SyncId  mon,
                                          MonitorOp                  operation,
                                          unsigned                   value,
                                          Settable<uint8_t>          mask) const override;

    virtual QueueCommandPtr getWaitForSemaphore(SyncObjectManager::SyncId  syncObj,
                                                SyncObjectManager::SyncId  mon,
                                                MonitorOp                  operation,
                                                unsigned                   value,
                                                Settable<uint8_t>          mask,
                                                WaitID                     waitID = ID_0,
                                                unsigned                   fenceValue = 1) const override;

    virtual QueueCommandPtr getSignalSemaphoreWithPredicate(SyncObjectManager::SyncId  which,
                                                            int16_t                    value,
                                                            uint32_t                   predicate,
                                                            int                        operation = 0,
                                                            int                        barriers = ALL_BARRIERS) const override;

    virtual QueueCommandPtr getSignalSemaphore(SyncObjectManager::SyncId  which,
                                               int16_t                    value,
                                               int                        operation = 0,
                                               int                        barriers = ALL_BARRIERS) const override;

    virtual QueueCommandPtr getMonitorSetup(SyncObjectManager::SyncId mon,
                                            WaitID                    waitID,
                                            HabanaDeviceType          deviceType,
                                            unsigned                  deviceID,
                                            uint32_t                  value,
                                            unsigned                  streamID,
                                            uint32_t                  predicate     = 0,
                                            bool                      incSyncObject = false) const override;

    virtual QueueCommandPtr getMonitorSetup(SyncObjectManager::SyncId mon,
                                            SyncObjectManager::SyncId syncId,
                                            uint32_t                  value,
                                            uint32_t                  predicate     = 0,
                                            bool                      incSyncObject = false) const override;

    virtual QueueCommandPtr getWriteReg64(unsigned  addrBaseRegIndex,
                                          uint64_t  addrOffset,
                                          unsigned  targetRegisterInBytes,
                                          bool      writeTargetLow = true,
                                          bool      writeTargetHigh = true,
                                          uint32_t  predicate = 0) const override;

    virtual QueueCommandPtr getFence(WaitID waitID, unsigned int targetValue, uint32_t predicate = 0) const override;

    virtual QueueCommandPtr getInvalidateTPCCaches(uint32_t predicate = 0) const override;

    virtual QueueCommandPtr getUploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate = 0) const override;

    virtual unsigned        getCpDmaCmdSize() const override;

    virtual unsigned        getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID, const NodePtr& n = nullptr) const override;

    virtual unsigned        getRegForBaseAddress(unsigned regIndex) const override;

    virtual QueueCommandPtr getSuspend(unsigned cyclesToWait) const override;

    virtual QueueCommandPtr getNop() const override;

    virtual QueueCommandPtr getQmanDelay() const override;

private:
    QueueCommandFactory();
};

} // namespace gaudi2
