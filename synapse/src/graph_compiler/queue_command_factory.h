#pragma once

#include "types.h"
#include "utils.h"
#include "sync/sync_types.h"
#include "sync/sync_object_manager.h"
#include "habana_device_types.h"
#include "infra/settable.h"

namespace gc_recipe {class generic_packets_container;}
class QueueCommand;
using QueueCommandPtr = std::unique_ptr<QueueCommand>;

class QueueCommandFactory
{
public:
    virtual ~QueueCommandFactory() {};

    virtual QueueCommandPtr getDmaHostToDram(const char*       hostPtr,
                                             deviceAddrOffset  dramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const
    {
        HB_ASSERT(false, "Unexpected func call");
        return nullptr;
    }

    virtual QueueCommandPtr getDmaHostToDram(gc_recipe::generic_packets_container&  container,
                                             deviceAddrOffset                       dramPtr,
                                             uint64_t                               size,
                                             bool                                   wrComplete,
                                             uint16_t                               contextID = 0) const
    {
        HB_ASSERT(false, "Unexpected func call");
        return nullptr;
    }

    virtual QueueCommandPtr getDmaDramToHost(char*             hostPtr,
                                             deviceAddrOffset  dramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const
    {
        HB_ASSERT(false, "Unexpected func call");
        return nullptr;
    }

    virtual QueueCommandPtr getDmaHostToSram(const char*       hostPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const
    {
        HB_ASSERT(false, "Unexpected func call");
        return nullptr;
    }

    virtual QueueCommandPtr getDmaSramToHost(char*             hostPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const
    {
        HB_ASSERT(false, "Unexpected func call");
        return nullptr;
    }

    virtual QueueCommandPtr getDmaDramToSram(deviceAddrOffset  dramPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const = 0;

    virtual QueueCommandPtr getDmaSramToDram(deviceAddrOffset  dramPtr,
                                             deviceAddrOffset  sramPtr,
                                             uint64_t          size,
                                             bool              wrComplete,
                                             uint16_t          contextID = 0) const = 0;

    virtual QueueCommandPtr getCpDma(deviceAddrOffset  addrOffset,
                                     uint64_t          size,
                                     uint32_t          predicate = 0) const = 0;

    virtual QueueCommandPtr getLoadDesc(void*             desc,
                                        unsigned          descSize,
                                        unsigned          descOffset,
                                        HabanaDeviceType  device,
                                        unsigned          deviceID = 0,
                                        uint32_t          predicate = 0) const = 0;

    virtual QueueCommandPtr getWriteManyRegisters(unsigned        firstRegOffset,
                                                  unsigned        count,
                                                  const uint32_t* values,
                                                  uint32_t        predicate = 0) const = 0;

    virtual QueueCommandPtr getExecute(HabanaDeviceType type,
                                       unsigned         deviceID = 0,
                                       uint32_t         predicate = 0,
                                       uint32_t         value = 0x1) const = 0;

    virtual QueueCommandPtr getExecute(HabanaDeviceType type,
                                       unsigned         deviceID,
                                       uint32_t         predicate,
                                       uint32_t         value,
                                       bool             engineBarrier) const
    {
        HB_ASSERT(!engineBarrier, "No implementation found to enable engine barrier");
        return getExecute(type, deviceID, predicate, value, engineBarrier);
    }

    virtual QueueCommandPtr getDmaClearCtrl(HabanaDeviceType type,
                                            unsigned         deviceID = 0,
                                            uint32_t         predicate = 0,
                                            uint32_t         value = 0x0) const {return nullptr;}

    virtual QueueCommandPtr getSfgInc(unsigned sigOutValue) const { return nullptr; }

    virtual QueueCommandPtr getSfgInit(unsigned sigOutValue) const { return nullptr; }

    virtual QueueCommandPtr getResetSobs(unsigned target, unsigned totalNumEngs) const { return nullptr; }

    virtual QueueCommandPtr getMcidRollover(unsigned target) const { return nullptr; }

    virtual QueueCommandPtr getDynamicExecute(QueueCommandPtr               execute,
                                              std::vector<QueueCommandPtr>& signalSemaphore,
                                              BypassType                    enableBypass) const = 0;

    virtual QueueCommandPtr getSetupAndArmCommand(std::vector<QueueCommandPtr>& setupAndArm) const = 0;

    virtual QueueCommandPtr getMonitorArm(SyncObjectManager::SyncId  syncObj,
                                          SyncObjectManager::SyncId  mon,
                                          MonitorOp                  operation,
                                          unsigned                   value,
                                          Settable<uint8_t>          mask) const = 0;

    virtual QueueCommandPtr getWaitForSemaphore(SyncObjectManager::SyncId syncObj,
                                                SyncObjectManager::SyncId mon,
                                                MonitorOp                 operation,
                                                unsigned                  value,
                                                Settable<uint8_t>         mask,
                                                WaitID                    waitID,
                                                unsigned                  fenceValue = 1) const = 0;

    virtual QueueCommandPtr getSignalSemaphore(SyncObjectManager::SyncId  which,
                                               int16_t                    value,
                                               int                        operation = 0,
                                               int                        barriers = ALL_BARRIERS) const = 0;

    virtual QueueCommandPtr getSignalSemaphoreWithPredicate(SyncObjectManager::SyncId  which,
                                                            int16_t                    value,
                                                            uint32_t                   predicate,
                                                            int                        operation = 0,
                                                            int                        barriers = ALL_BARRIERS) const = 0;

    virtual QueueCommandPtr getMonitorSetup(SyncObjectManager::SyncId mon,
                                            WaitID                    waitID,
                                            HabanaDeviceType          deviceType,
                                            unsigned                  deviceID,
                                            uint32_t                  value,
                                            unsigned                  streamID,
                                            uint32_t                  predicate     = 0,
                                            bool                      incSyncObject = false) const = 0;

    virtual QueueCommandPtr getMonitorSetup(SyncObjectManager::SyncId mon,
                                            SyncObjectManager::SyncId syncId,
                                            uint32_t                  value,
                                            uint32_t                  predicate     = 0,
                                            bool                      incSyncObject = false) const = 0;

    virtual QueueCommandPtr getWriteReg64(unsigned  addrBaseRegIndex,
                                          uint64_t  addrOffset,
                                          unsigned  targetRegisterInBytes,
                                          bool      writeTargetLow = true,
                                          bool      writeTargetHigh = true,
                                          uint32_t  predicate = 0) const = 0;

    virtual QueueCommandPtr getFence(WaitID waitID, unsigned int targetValue, uint32_t predicate = 0) const = 0;

    virtual QueueCommandPtr getInvalidateTPCCaches(uint32_t predicate = 0) const = 0;

    virtual QueueCommandPtr getUploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate = 0) const = 0;

    virtual QueueCommandPtr getSuspend(unsigned cyclesToWait) const = 0;

    virtual QueueCommandPtr getNop() const { return nullptr; }

    virtual QueueCommandPtr getWaitForQmanMutex(uint32_t predicate = 0) const { return nullptr; }

    virtual QueueCommandPtr getQmanDelay() const { return nullptr; }

    virtual unsigned getCpDmaCmdSize() const = 0;

    virtual unsigned getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID, const NodePtr& n = nullptr) const = 0;

    virtual unsigned getRegForBaseAddress(unsigned regIndex) const = 0;
};
