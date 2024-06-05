#include <defs.h>
#include "platform/gaudi3/graph_compiler/queue_command_factory.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "block_data.h"
#include "gaudi3/gaudi3_packets.h"
#include "define_synapse_common.hpp"
#include "habana_nodes/mme_node.h"

namespace gaudi3
{
QueueCommandFactory& QueueCommandFactory::instance()
{
    // Singleton implementation - static variable is created only once.
    thread_local static QueueCommandFactory onlyOneInstance;
    return onlyOneInstance;
}

QueueCommandFactory::QueueCommandFactory() {}

QueueCommandFactory::~QueueCommandFactory() {}

QueueCommandPtr QueueCommandFactory::getDmaDramToSram(deviceAddrOffset dramPtr,
                                                      deviceAddrOffset sramPtr,
                                                      uint64_t         size,
                                                      bool             wrComplete,
                                                      uint16_t         contextID) const
{
    HB_ASSERT(0, "getDmaDramToSram is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getDmaSramToDram(deviceAddrOffset dramPtr,
                                                      deviceAddrOffset sramPtr,
                                                      uint64_t         size,
                                                      bool             wrComplete,
                                                      uint16_t         contextID) const
{
    HB_ASSERT(0, "getDmaSramToDram is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getCpDma(deviceAddrOffset addrOffset, uint64_t size, uint32_t predicate) const
{
    HB_ASSERT(0, "getCpDma is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getLoadDesc(void*            desc,
                                                 unsigned         descSize,
                                                 unsigned         descOffset,
                                                 HabanaDeviceType device,
                                                 unsigned         deviceID,
                                                 uint32_t         predicate) const
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

QueueCommandPtr
QueueCommandFactory::getExecute(HabanaDeviceType type, unsigned deviceID, uint32_t predicate, uint32_t value) const
{
    return std::make_unique<Execute>(type, deviceID, predicate, value);
}

QueueCommandPtr QueueCommandFactory::getMonitorArm(SyncObjectManager::SyncId syncObj,
                                                   SyncObjectManager::SyncId mon,
                                                   MonitorOp                 operation,
                                                   unsigned                  value,
                                                   Settable<uint8_t>         mask) const
{
    HB_ASSERT(0, "getMonitorArm is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getWaitForSemaphore(SyncObjectManager::SyncId syncObj,
                                                         SyncObjectManager::SyncId mon,
                                                         MonitorOp                 operation,
                                                         unsigned                  value,
                                                         Settable<uint8_t>         mask,
                                                         WaitID                    waitID,
                                                         unsigned                  fenceValue) const
{
    HB_ASSERT(0, "getWaitForSemaphore is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getSignalSemaphoreWithPredicate(SyncObjectManager::SyncId which,
                                                                     int16_t                   value,
                                                                     uint32_t                  predicate,
                                                                     int                       operation,
                                                                     int                       barriers) const
{
    HB_ASSERT(0, "getSignalSemaphoreWithPredicate is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getSignalSemaphore(SyncObjectManager::SyncId which,
                                                        int16_t                   value,
                                                        int                       operation,
                                                        int                       barriers) const
{
    HB_ASSERT(0, "getSignalSemaphore is not implemented for gaudi3");
    return nullptr;
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
    HB_ASSERT(0, "getMonitorSetup is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getMonitorSetup(SyncObjectManager::SyncId mon,
                                                     SyncObjectManager::SyncId syncId,
                                                     uint32_t                  value,
                                                     uint32_t                  predicate,
                                                     bool                      incSyncObject) const
{
    HB_ASSERT(0, "getMonitorSetup is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getFence(WaitID waitID, unsigned int targetValue, uint32_t predicate) const
{
    return std::make_unique<Fence>(waitID, targetValue);  // TODO not using yet predicate
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
    HB_ASSERT(0, "getCpDmaCmdSize is not implemented for gaudi3");
    return 0;
}

unsigned QueueCommandFactory::getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID, const NodePtr& n) const
{
    return gaudi3::getRegForLoadDesc(type, MmeNode::isDmaOperation(n));
}

unsigned QueueCommandFactory::getRegForBaseAddress(unsigned regIndex) const
{
    return gaudi3::getRegForBaseAddress(regIndex);
}

QueueCommandPtr QueueCommandFactory::getSuspend(unsigned cyclesToWait) const
{
    return std::make_unique<Suspend>(ID_1, cyclesToWait);
}

QueueCommandPtr QueueCommandFactory::getNop() const
{
    return std::make_unique<Nop>();
}

QueueCommandPtr QueueCommandFactory::getResetSobs(unsigned target, unsigned totalNumEngs) const
{
    return std::make_unique<ResetSobs>(target, totalNumEngs);
}

QueueCommandPtr QueueCommandFactory::getMcidRollover(unsigned target) const
{
    return std::make_unique<McidRollover>(target);
}

QueueCommandPtr QueueCommandFactory::getDynamicExecute(QueueCommandPtr               execute,
                                                       std::vector<QueueCommandPtr>& signalSemaphore,
                                                       BypassType                    enableBypass) const
{
    HB_ASSERT(0, "getDynamicExecute is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getSetupAndArmCommand(std::vector<QueueCommandPtr>& setupAndArm) const
{
    HB_ASSERT(0, "SetupAndArm is not implemented for gaudi3");
    return nullptr;
}

QueueCommandPtr QueueCommandFactory::getWriteReg64(unsigned addrBaseRegIndex,
                                                   uint64_t addrOffset,
                                                   unsigned targetRegisterInBytes,
                                                   bool     writeTargetLow,
                                                   bool     writeTargetHigh,
                                                   uint32_t predicate) const
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

}  // namespace gaudi3
