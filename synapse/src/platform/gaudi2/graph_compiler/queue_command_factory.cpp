#include <defs.h>
#include "platform/gaudi2/graph_compiler/queue_command_factory.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "block_data.h"
#include "gaudi2/gaudi2_packets.h"
#include "define_synapse_common.hpp"


namespace gaudi2
{

QueueCommandFactory& QueueCommandFactory::instance()
{
    // Singleton implementation - static variable is created only once.
    thread_local static QueueCommandFactory onlyOneInstance;
    return onlyOneInstance;
}

QueueCommandFactory::QueueCommandFactory()
{
}

QueueCommandFactory::~QueueCommandFactory()
{
}

QueueCommandPtr QueueCommandFactory::getDmaDramToSram(
                  deviceAddrOffset  dramPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    return std::make_unique<DmaDramToSram>(dramPtr, sramPtr, size, wrComplete, contextID);
}

QueueCommandPtr QueueCommandFactory::getDmaSramToDram(
                  deviceAddrOffset  dramPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    return std::make_unique<DmaSramToDram>(dramPtr, sramPtr, size, wrComplete, contextID);
}

QueueCommandPtr QueueCommandFactory::getCpDma(
                  deviceAddrOffset  addrOffset,
                  uint64_t          size,
                  uint32_t          predicate) const
{
    return std::make_unique<CpDma>(addrOffset, size, predicate);
}

QueueCommandPtr QueueCommandFactory::getLoadDesc(
                  void*             desc,
                  unsigned          descSize,
                  unsigned          descOffset,
                  HabanaDeviceType  device,
                  unsigned          deviceID,
                  uint32_t          predicate) const
{
    return std::make_unique<LoadDesc>(desc, descSize, descOffset, device, deviceID, predicate);
}

QueueCommandPtr QueueCommandFactory::getWriteManyRegisters(unsigned        firstRegOffset,
                                                           unsigned        count,
                                                           const uint32_t* values,
                                                           uint32_t        predicate) const
{
    return std::make_unique<WriteManyRegisters>(firstRegOffset, count, values, predicate);
}

QueueCommandPtr QueueCommandFactory::getExecute(HabanaDeviceType type,
                                                unsigned         deviceID,
                                                uint32_t         predicate,
                                                uint32_t         value) const
{
    return std::make_unique<Execute>(type, deviceID, predicate, value);
}

QueueCommandPtr QueueCommandFactory::getMonitorArm(
                  SyncObjectManager::SyncId  syncObj,
                  SyncObjectManager::SyncId  mon,
                  MonitorOp                  operation,
                  unsigned                   value,
                  Settable<uint8_t>          mask) const
{
    return std::make_unique<MonitorArm>(syncObj, mon, operation, value, mask);
}

QueueCommandPtr QueueCommandFactory::getWaitForSemaphore(
                  SyncObjectManager::SyncId  syncObj,
                  SyncObjectManager::SyncId  mon,
                  MonitorOp                  operation,
                  unsigned                   value,
                  Settable<uint8_t>          mask,
                  WaitID                     waitID,
                  unsigned                   fenceValue) const
{
    return std::make_unique<WaitForSemaphore>(syncObj, mon, operation, value, mask, waitID, fenceValue);
}

QueueCommandPtr QueueCommandFactory::getSignalSemaphoreWithPredicate(SyncObjectManager::SyncId  which,
                                                                     int16_t                    value,
                                                                     uint32_t                   predicate,
                                                                     int                        operation,
                                                                     int                        barriers) const
{
    return std::make_unique<SignalSemaphoreWithPredicate>(which, value, predicate, operation, barriers);
}

QueueCommandPtr QueueCommandFactory::getSignalSemaphore(
                  SyncObjectManager::SyncId  which,
                  int16_t                    value,
                  int                        operation,
                  int                        barriers) const
{
    return std::make_unique<SignalSemaphore>(which, value, operation, barriers);
}

QueueCommandPtr QueueCommandFactory::getMonitorSetup(SyncObjectManager::SyncId mon,
                                                     WaitID                    waitID,
                                                     HabanaDeviceType          deviceType,
                                                     unsigned                  deviceID,
                                                     uint32_t                  value,
                                                     unsigned                  streamID,
                                                     uint32_t                  predicate,
                                                     bool                      incSyncObject) const
{
    return std::make_unique<MonitorSetup>(mon, waitID, deviceType, deviceID, value, streamID, predicate);
}

QueueCommandPtr QueueCommandFactory::getMonitorSetup(SyncObjectManager::SyncId mon,
                                                     SyncObjectManager::SyncId syncId,
                                                     uint32_t                  value,
                                                     uint32_t                  predicate,
                                                     bool                      incSyncObject) const
{
    return std::make_unique<MonitorSetup>(mon, syncId, value, predicate);
}

QueueCommandPtr QueueCommandFactory::getFence(WaitID waitID, unsigned int targetValue, uint32_t predicate) const
{
    return std::make_unique<Fence>(waitID, targetValue);  // not using yet predicate
}

QueueCommandPtr QueueCommandFactory::getInvalidateTPCCaches(uint32_t predicate) const
{
    return std::make_unique<InvalidateTPCCaches>(predicate);
}

QueueCommandPtr QueueCommandFactory::getUploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate) const
{
    return std::make_unique<UploadKernelsAddr>(low, high, predicate);
}

unsigned QueueCommandFactory::getCpDmaCmdSize() const
{
    return sizeof(packet_cp_dma);
}

unsigned QueueCommandFactory::getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID, const NodePtr& n) const
{
    return gaudi2::getRegForLoadDesc(type, deviceID);
}

unsigned QueueCommandFactory::getRegForBaseAddress(unsigned regIndex) const
{
    return gaudi2::getRegForBaseAddress(regIndex);
}

QueueCommandPtr QueueCommandFactory::getSuspend(unsigned cyclesToWait) const
{
    return std::make_unique<Suspend>(ID_1, cyclesToWait);
}

QueueCommandPtr QueueCommandFactory::getNop() const
{
    return std::make_unique<Nop>();
}

QueueCommandPtr QueueCommandFactory::getSfgInc(unsigned sigOutValue) const
{
    return std::make_unique<SFGCmd>(sigOutValue);
}

QueueCommandPtr QueueCommandFactory::getSfgInit(unsigned sigOutValue) const
{
    return std::make_unique<SFGInitCmd>(sigOutValue);
}

QueueCommandPtr QueueCommandFactory::getResetSobs(unsigned target, unsigned totalNumEngs) const
{
    return std::make_unique<ResetSobs>(target, totalNumEngs);
}

QueueCommandPtr QueueCommandFactory::getDynamicExecute(QueueCommandPtr execute, std::vector<QueueCommandPtr>& signalSemaphore, BypassType enableBypass) const
{
    std::vector<std::shared_ptr<Gaudi2QueueCommand>> commandVector;

    commandVector.push_back(static_unique_pointer_cast<Gaudi2QueueCommand>(std::move(execute)));
    for (auto& cmd : signalSemaphore)
    {
        commandVector.push_back(static_unique_pointer_cast<Gaudi2QueueCommand>(std::move(cmd)));
    }

    return std::make_unique<DynamicExecute>(commandVector);
}

QueueCommandPtr QueueCommandFactory::getSetupAndArmCommand(std::vector<QueueCommandPtr>& setupAndArm) const
{
    HB_ASSERT(0, "SetupAndArm is not implemented for gaudi2");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getWriteReg64(unsigned  addrBaseRegIndex,
                                                   uint64_t  addrOffset,
                                                   unsigned  targetRegisterInBytes,
                                                   bool      writeTargetLow,
                                                   bool      writeTargetHigh,
                                                   uint32_t  predicate) const
{
    return std::make_unique<WriteReg64>(addrBaseRegIndex,
                                        addrOffset,
                                        targetRegisterInBytes,
                                        writeTargetLow,
                                        writeTargetHigh,
                                        predicate);
}

QueueCommandPtr QueueCommandFactory::getQmanDelay() const
{
    return std::make_unique<QmanDelay>();
}

} // namespace gaudi2
