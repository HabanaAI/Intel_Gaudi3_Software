#include <defs.h>
#include "platform/gaudi/graph_compiler/queue_command_factory.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "gaudi/gaudi_packets.h"
#include "define_synapse_common.hpp"


namespace gaudi
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

QueueCommandPtr QueueCommandFactory::getDmaHostToDram(
                  const char*       hostPtr,
                  deviceAddrOffset  dramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    HB_ASSERT(false, "Unexpected func call");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getDmaHostToDram(
                  gc_recipe::generic_packets_container&  container,
                  deviceAddrOffset                       dramPtr,
                  uint64_t                               size,
                  bool                                   wrComplete,
                  uint16_t                               contextID) const
{
    HB_ASSERT(false, "Unexpected func call");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getDmaDramToHost(
                  char*             hostPtr,
                  deviceAddrOffset  dramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    HB_ASSERT(false, "Unexpected func call");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getDmaHostToSram(
                  const char*       hostPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    HB_ASSERT(false, "Unexpected func call");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getDmaSramToHost(
                  char*             hostPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID) const
{
    HB_ASSERT(false, "Unexpected func call");
    return nullptr;
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
    return std::make_unique<MonitorSetup>(mon, waitID, deviceType, deviceID, value, streamID, predicate, incSyncObject);
}

QueueCommandPtr QueueCommandFactory::getMonitorSetup(SyncObjectManager::SyncId mon,
                                                     SyncObjectManager::SyncId syncId,
                                                     uint32_t                  value,
                                                     uint32_t                  predicate,
                                                     bool                      incSyncObject) const
{
    return std::make_unique<MonitorSetup>(mon, syncId, value, predicate, incSyncObject);
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
    return gaudi::getRegForLoadDesc(type, deviceID);
}

unsigned QueueCommandFactory::getRegForBaseAddress(unsigned regIndex) const
{
    HB_ASSERT(false, "Unexpected func call, gaudi1 does not support qman base address register");
    return 0;
}

QueueCommandPtr QueueCommandFactory::getSuspend(unsigned cyclesToWait) const
{
    return std::make_unique<Suspend>(ID_1, cyclesToWait);
}

QueueCommandPtr QueueCommandFactory::getNop() const
{
    return std::make_unique<Nop>();
}

QueueCommandPtr QueueCommandFactory::getDynamicExecute(QueueCommandPtr execute, std::vector<QueueCommandPtr>& signalSemaphore, BypassType enableBypass) const
{
    std::vector<std::shared_ptr<GaudiQueueCommand>> commandVector;

    commandVector.push_back(static_unique_pointer_cast<GaudiQueueCommand>(std::move(execute)));
    for (auto& cmd : signalSemaphore)
    {
        commandVector.push_back(static_unique_pointer_cast<GaudiQueueCommand>(std::move(cmd)));
    }

    return std::make_unique<DynamicExecute>(commandVector, enableBypass);
}

QueueCommandPtr QueueCommandFactory::getSetupAndArmCommand(std::vector<QueueCommandPtr>& setupAndArm) const
{
    std::vector<std::shared_ptr<GaudiQueueCommand>> commandVector;

    for (auto& cmd : setupAndArm)
    {
        commandVector.push_back(static_unique_pointer_cast<GaudiQueueCommand>(std::move(cmd)));
    }

    return std::make_unique<SetupAndArm>(commandVector);
}

QueueCommandPtr QueueCommandFactory::getWriteReg64(unsigned  addrBaseRegIndex,
                                                   uint64_t  addrOffset,
                                                   unsigned  targetRegisterInBytes,
                                                   bool      writeTargetLow,
                                                   bool      writeTargetHigh,
                                                   uint32_t  predicate) const
{
    HB_ASSERT(false, "Unexpected func call, gaudi1 does not support WriteReg64 (wreg64) command");
    return nullptr;
}

} // namespace gaudi
