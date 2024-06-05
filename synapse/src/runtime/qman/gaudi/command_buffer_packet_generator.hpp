#pragma once

#include "runtime/qman/common/command_buffer_packet_generator.hpp"

#include <memory>
#include <atomic>
#include <mutex>

namespace gaudi
{
class CommandBufferPktGenerator : public generic::CommandBufferPktGenerator
{
public:
    static CommandBufferPktGenerator* getInstance();

    virtual ~CommandBufferPktGenerator();

    virtual uint32_t getQmanId(uint64_t engineId) const override;

    // The caller needs to pass mapped addresses
    virtual synStatus generateLinDmaPacket(char*&         pPacket,
                                           uint64_t&      packetSize,
                                           uint64_t       srcAddress,
                                           uint64_t       dstAddress,
                                           uint32_t       size,
                                           internalDmaDir direction,
                                           uint32_t       contextId  = 0,
                                           bool           isMemSet   = false,
                                           uint32_t       engBarrier = 1) const override;

    virtual synStatus generateNopPacket(char*& pPacket, uint64_t& packetSize, uint32_t barriers = 0) const override;

    synStatus generateSignalCommand(char*&    pPackets,
                                    uint64_t& commandSize,
                                    uint32_t  which,
                                    int16_t   value,
                                    uint16_t  operation,
                                    uint32_t  barriers) const;

    synStatus generateWaitCommand(char*&    pPackets,
                                  uint64_t& commandSize,
                                  uint32_t  waitQueueId,
                                  uint32_t  monitorObjId,
                                  uint32_t  syncObjId,
                                  int16_t   syncObjValue,
                                  uint16_t  operation,
                                  uint32_t  barriers) const;

    synStatus generateResetSyncObjectsCommand(char*&   pPackets,
                                              uint32_t firstSyncObjId,
                                              uint32_t numOfSyncObjects,
                                              uint32_t barriers) const;

    synStatus generateArbitratorDisableConfigCommand(char*& pPackets, uint32_t qmanId) const;

    synStatus generateMasterArbitratorConfigCommand(char*&                           pPackets,
                                                    generic::masterSlavesArbitration masterSlaveIds,
                                                    bool                             isByPriority) const;

    synStatus generateSlaveArbitratorConfigCommand(char*&   pPackets,
                                                   uint32_t slaveId,
                                                   uint32_t uSlaveQmanId,
                                                   uint32_t uMasterQmanId) const;

    synStatus generateLoadPredicateCommand(char*& pPackets, uint64_t srcAddr) const;

    synStatus generateFenceCommand(char*&    pPacket,
                                   uint64_t& packetSize,
                                   uint32_t  engBarrier = 1,
                                   uint32_t  regBarrier = 1,
                                   uint32_t  msgBarrier = 1) const;

    synStatus generateFenceClearCommand(char*&    pPacket,
                                        uint64_t& packetSize,
                                        uint64_t  streamId,
                                        uint32_t  engBarrier = 1,
                                        uint32_t  regBarrier = 1,
                                        uint32_t  msgBarrier = 1) const;

    virtual uint64_t getLinDmaPacketSize() override;

    void getNopPacketSize(uint64_t& packetSize);

    uint64_t getSignalCommandSize();

    uint64_t getWaitCommandSize();

    virtual uint64_t getFenceSetPacketCommandSize() override;

    virtual uint64_t getFenceClearPacketCommandSize() override;

    void getSingleResetSyncObjectsCommandSize(uint64_t& commandSize);

    virtual uint64_t getArbitrationCommandSize() override;

    void getArbitratorDisableConfigCommandSize(uint64_t& commandSize);

    void getMasterArbitratorBasicConfigCommandSize(uint64_t& commandSize);

    void getMasterSingleSlaveArbitratorConfigCommandSize(uint64_t& commandSize);

    void getSlaveArbitratorConfigCommandSize(uint64_t& commandSize);

    virtual synStatus generateCpDma(char*&   pPacket,
                                    uint32_t tsize,
                                    uint32_t upperCp,
                                    uint32_t engBarrier,
                                    uint32_t msgBarrier,
                                    uint64_t addr,
                                    uint32_t predicate) override;

    void generateDefaultCpDma(char*& pPacket, uint32_t tsize, uint64_t addr) override;

    virtual uint64_t getCpDmaSize() override;

    void      getLoadAndExecCommandSize(uint64_t& commandSize);
    synStatus generateArbitrationCommand(char*&                        pPacket,
                                         uint64_t&                     packetSize,
                                         bool                          priorityRelease,
                                         generic::eArbitrationPriority priority = generic::ARB_PRIORITY_NORMAL) const;

    virtual bool isDmaDownArbitrationRequired() override { return true; }

    virtual uint64_t getCoeffTableConfigCommandSize() override;

    virtual synStatus
    generateCoeffTableConfigCommands(char*& pPackets, ptrToInt tableBaseAddr, uint32_t singleCmdSize) const override;

    virtual synStatus
    generateCoeffTableConfigCommand(char*& pPackets, uint64_t tpcBaseAddr, ptrToInt tableBaseAddr) const override;

private:
    typedef enum MonitorRegisterType
    {
        MONITOR_REGISTER_PAYLOAD_LOW_ADDRESS,
        MONITOR_REGISTER_PAYLOAD_HIGH_ADDRESS,
        MONITOR_REGISTER_PAYLOAD_DATA,
        MONITOR_REGISTER_PAYLOAD_MONITOR_ARM
    } MonitorRegisterType;

    CommandBufferPktGenerator();

    uint64_t _getSyncObjectAddress(unsigned syncObjId) const;

    uint64_t _getMonitorRegisterAddress(unsigned monitoObjId, MonitorRegisterType registerType) const;

    synStatus _generateArbitratorBaseAddressConfiguration(char*& pPackets, uint32_t qmanId) const;

    uint64_t _getArbitratorBaseAddressConfigCommandSize();

    static std::shared_ptr<CommandBufferPktGenerator> m_pInstance;
    static std::mutex                                 s_mutex;

    uint64_t m_linDmaCommandSize;
    uint64_t m_msgLongPacketSize;
    uint64_t m_signalCommandSize;
    uint64_t m_waitCommandSize;
    uint64_t m_resetSyncObjectsCommandSize;
    uint64_t m_arbitrationPacketCommandSize;
    uint64_t m_arbitratorBaseAddressCommandSize;
    uint64_t m_arbitratorDisableCommandSize;
    uint64_t m_masterArbitratorBasicCommandSize;
    uint64_t m_masterArbitratorSingleSlaveCommandSize;
    uint64_t m_slaveArbitratorCommandSize;
    std::atomic<uint64_t> m_cpDmaCommandSize;
    uint64_t m_loadAndExecCommandSize;
    uint64_t m_fencePacketSize;
    uint64_t m_fenceClearPacketSize;
    uint64_t m_coeffTableConfPacketSize;
};
}  // namespace gaudi
