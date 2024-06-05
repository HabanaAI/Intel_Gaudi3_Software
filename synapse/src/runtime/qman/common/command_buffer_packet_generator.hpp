#pragma once

#include "define_synapse_common.hpp"
#include "defs.h"
#include "utils.h"

#include <memory>
#include <deque>

namespace CommonPktGen
{
class BasePacket;
}

namespace generic
{
typedef std::deque<uint32_t> CommonQmansIdDB;

struct masterSlavesArbitration
{
    uint32_t        masterQmanId;
    CommonQmansIdDB slaveQmansId;
};

// Master-Slave arbitrators DB
struct MasterSlabeArbitrationInfo
{
    //  QMANs
    masterSlavesArbitration masterSlaveArbQmans;

    // Specific flags
    bool isArbByPriority;
};
typedef std::deque<MasterSlabeArbitrationInfo> masterSlaveArbitrationInfoDB;

enum eArbitrationPriority
{
    ARB_PRIORITY_NORMAL = 0x1,
    ARB_PRIORITY_HIGH   = 0x2
};
class CommandBufferPktGenerator
{
public:
    CommandBufferPktGenerator();
    virtual ~CommandBufferPktGenerator() = default;

    virtual uint32_t getQmanId(uint64_t engineId) const = 0;

    // The caller needs to pass mapped addresses
    virtual synStatus generateLinDmaPacket(char*&         pPacket,
                                           uint64_t&      packetSize,
                                           uint64_t       srcAddress,
                                           uint64_t       dstAddress,
                                           uint32_t       size,
                                           internalDmaDir direction,
                                           uint32_t       contextId  = 0,
                                           bool           isMemset   = false,
                                           uint32_t       engBarrier = 1) const = 0;

    virtual uint64_t getLinDmaPacketSize() = 0;

    virtual synStatus generateNopPacket(char*& pPacket, uint64_t& packetSize, uint32_t barriers = 0) const = 0;

    virtual uint64_t getArbitrationCommandSize() = 0;

    virtual synStatus generateCpDma(char*&   pPacket,
                                    uint32_t tsize,
                                    uint32_t upperCp,
                                    uint32_t engBarrier,
                                    uint32_t msgBarrier,
                                    uint64_t addr,
                                    uint32_t predicate)
    {
        return synUnsupported;
    };

    virtual void generateDefaultCpDma(char*& pPacket, uint32_t tsize, uint64_t addr)
    {
        HB_ASSERT(false, "CPDMA is unsupported");
    };

    virtual uint64_t getCpDmaSize() = 0;

    virtual uint64_t getFenceSetPacketCommandSize() = 0;

    virtual uint64_t getFenceClearPacketCommandSize() = 0;

    virtual synStatus generateSlaveArbitratorConfigCommand(char*&   pPackets,
                                                           uint32_t slaveId,
                                                           uint32_t uSlaveQmanId,
                                                           uint32_t uMasterQmanId) const;

    virtual void getArbitratorDisableConfigCommandSize(uint64_t& commandSize) {}

    virtual void getMasterArbitratorBasicConfigCommandSize(uint64_t& commandSize) {}

    virtual void getMasterSingleSlaveArbitratorConfigCommandSize(uint64_t& commandSize) {}

    virtual void getSlaveArbitratorConfigCommandSize(uint64_t& commandSize) {}

    virtual synStatus generateArbitratorDisableConfigCommand(char*& pPackets, uint32_t qmanId) const;

    virtual synStatus generateMasterArbitratorConfigCommand(char*&                           pPackets,
                                                            generic::masterSlavesArbitration masterSlaveIds,
                                                            bool                             isByPriority) const;

    virtual synStatus
    generateArbitrationCommand(char*&                        pPacket,
                               uint64_t&                     packetSize,
                               bool                          priorityRelease,
                               generic::eArbitrationPriority priority = generic::ARB_PRIORITY_NORMAL) const;

    virtual synStatus generateFenceCommand(char*&    pPacket,
                                           uint64_t& packetSize,
                                           uint32_t  engBarrier = 1,
                                           uint32_t  regBarrier = 1,
                                           uint32_t  msgBarrier = 1) const
    {
        // currently unused
        HB_ASSERT(false, "generateFenceCommand synUnsupported");

        return synUnsupported;
    }

    virtual synStatus generateFenceClearCommand(char*&    pPacket,
                                                uint64_t& packetSize,
                                                uint64_t  streamId,
                                                uint32_t  engBarrier = 1,
                                                uint32_t  regBarrier = 1,
                                                uint32_t  msgBarrier = 1) const
    {
        // currently unused
        HB_ASSERT(false, "generateFenceClearCommand synUnsupported");

        return synUnsupported;
    }

    virtual uint64_t getSignalCommandSize() { return 0; }

    virtual uint64_t getWaitCommandSize() { return 0; }

    virtual void getLoadAndExecCommandSize(uint64_t& commandSize) {}

    virtual synStatus generateLoadPredicateCommand(char*& pPackets, uint64_t srcAddr) const { return synUnsupported; }

    virtual synStatus generateSignalCommand(char*&    pPackets,
                                            uint64_t& commandSize,
                                            uint32_t  which,
                                            int16_t   value,
                                            uint16_t  operation,
                                            uint32_t  barriers) const
    {
        return synUnsupported;
    }

    virtual synStatus generateWaitCommand(char*&    pPackets,
                                          uint64_t& commandSize,
                                          uint32_t  waitQueueId,
                                          uint32_t  monitorObjId,
                                          uint32_t  syncObjId,
                                          int16_t   syncObjValue,
                                          uint16_t  operation,
                                          uint32_t  barriers) const
    {
        return synUnsupported;
    }

    virtual bool isDmaDownArbitrationRequired() = 0;

    virtual uint64_t getCoeffTableConfigCommandSize() { return 0; }

    virtual synStatus
    generateCoeffTableConfigCommands(char*& pPackets, ptrToInt tableBaseAddr, uint32_t singleCmdSize) const
    {
        return synSuccess;
    };
    virtual synStatus
    generateCoeffTableConfigCommand(char*& pPackets, uint64_t tpcBaseAddr, ptrToInt tableBaseAddr) const
    {
        return synSuccess;
    };

    static CommandBufferPktGenerator* getCommandBufferPktGenerator(synDeviceType deviceType);

protected:
    synStatus _generatePacket(char*&                    pPacket,
                              uint64_t&                 bufferSize,
                              CommonPktGen::BasePacket& packetGen,
                              bool                      shouldIncrementPointer = false) const;
};
}  // namespace generic
