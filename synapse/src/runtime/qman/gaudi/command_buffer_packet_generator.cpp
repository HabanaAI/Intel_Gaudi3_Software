#include "command_buffer_packet_generator.hpp"

#include "defs.h"
#include "generate_packet.hpp"
#include "synapse_runtime_logging.h"
#include "utils.h"

#include "platform/gaudi/utils.hpp"

#include "graph_compiler/sync/sync_types.h"

#include "define_synapse_common.hpp"

#include "gaudi/gaudi.h"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "gaudi/asic_reg_structs/sync_mngr_regs.h"
#include "gaudi/asic_reg_structs/qman_regs.h"
#include "coeff_table_configuration_manager.hpp"
#include <limits>

using namespace gaudi;

typedef generic::CommandBufferPktGenerator Parent;

static const uint64_t INVALID_QMAN_BASE_ADDRESS = std::numeric_limits<uint64_t>::max();

// For configuring the QMANs' arbitrator
static const uint64_t QMANS_QM_BASE_ADDRESS[GAUDI_ENGINE_ID_SIZE] = {mmDMA0_QM_BASE,  mmDMA1_QM_BASE,
                                                                     mmDMA2_QM_BASE,  mmDMA3_QM_BASE,
                                                                     mmDMA4_QM_BASE,  mmDMA5_QM_BASE,
                                                                     mmDMA6_QM_BASE,  mmDMA7_QM_BASE,

                                                                     mmMME0_QM_BASE,  INVALID_QMAN_BASE_ADDRESS,
                                                                     mmMME2_QM_BASE,  INVALID_QMAN_BASE_ADDRESS,

                                                                     mmTPC0_QM_BASE,  mmTPC1_QM_BASE,
                                                                     mmTPC2_QM_BASE,  mmTPC3_QM_BASE,
                                                                     mmTPC4_QM_BASE,  mmTPC5_QM_BASE,
                                                                     mmTPC6_QM_BASE,  mmTPC7_QM_BASE,

                                                                     mmNIC0_QM0_BASE, mmNIC0_QM1_BASE,
                                                                     mmNIC1_QM0_BASE, mmNIC1_QM1_BASE,
                                                                     mmNIC2_QM0_BASE, mmNIC2_QM1_BASE,
                                                                     mmNIC3_QM0_BASE, mmNIC3_QM1_BASE,
                                                                     mmNIC4_QM0_BASE, mmNIC4_QM1_BASE};

// For configuring the coeff table for TPCs
static const uint64_t TPCS_CFG_BASE_ADDRESS[] = {mmTPC0_CFG_BASE,
                                                 mmTPC1_CFG_BASE,
                                                 mmTPC2_CFG_BASE,
                                                 mmTPC3_CFG_BASE,
                                                 mmTPC4_CFG_BASE,
                                                 mmTPC5_CFG_BASE,
                                                 mmTPC6_CFG_BASE,
                                                 mmTPC7_CFG_BASE};

static const uint64_t ARB_BASIC_BASE_LOW_CFG_REG_OFFSET  = offsetof(block_qman, arb_base_lo);
static const uint64_t ARB_BASIC_BASE_HIGH_CFG_REG_OFFSET = offsetof(block_qman, arb_base_hi);
static const uint64_t ARB_BASIC_CONFIG_REG_OFFSET        = offsetof(block_qman, arb_cfg_0);
static const uint64_t ARB_MASTER_CHOISE_PUSH_REG_OFFSET  = offsetof(block_qman, arb_mst_choise_push_ofst);
static const uint64_t ARB_SLAVE_CHOISE_PUSH_REG_OFFSET   = offsetof(block_qman, arb_choise_q_push);
static const uint64_t ARB_MASTER_CREDIT_INC_REG_OFFSET   = offsetof(block_qman, arb_mst_cred_inc);
static const uint64_t ARB_SLAVE_CREDIT_INC_REG_OFFSET    = offsetof(block_qman, arb_slv_master_inc_cred_ofst);
static const uint64_t ARB_MASTER_SLAVE_MASK_REG_OFFSET   = offsetof(block_qman, arb_mst_slave_en);
static const uint64_t ARB_SLAVE_ID_REG_OFFSET            = offsetof(block_qman, arb_slv_id);

static const uint64_t RUN_TIME_SYNC_MANAGER_BASE_ADDRESS = mmSYNC_MNGR_GLBL_E_N_BASE;
static const uint64_t SYNC_MGR_OBJ_BASE_ADDR =
    RUN_TIME_SYNC_MANAGER_BASE_ADDRESS + offsetof(block_sync_mngr, sync_mngr_objs);

static const uint32_t NUM_OF_MONITOR_OBJ = 512;
static const uint32_t NUM_OF_SYNC_OBJ    = 2048;

static const uint32_t LAST_MONITOR_OBJ_ID = NUM_OF_MONITOR_OBJ - 1;
static const uint32_t LAST_SYNC_OBJ_ID    = NUM_OF_SYNC_OBJ - 1;

static const uint32_t MAX_SLAVE_PER_MASTER = 32;

#define VERIFY_IS_NULL_POINTER(logger_name, pointer, name)                                                             \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(logger_name, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                       \
        return synFail;                                                                                                \
    }

std::shared_ptr<CommandBufferPktGenerator> CommandBufferPktGenerator::m_pInstance = nullptr;
std::mutex                                 CommandBufferPktGenerator::s_mutex;

CommandBufferPktGenerator* CommandBufferPktGenerator::getInstance()
{
    if (m_pInstance == nullptr)
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        {
            if (m_pInstance == nullptr)
            {
                m_pInstance.reset(new CommandBufferPktGenerator());
            }
        }
    }

    return m_pInstance.get();
}

CommandBufferPktGenerator::CommandBufferPktGenerator()
: Parent(),
  m_linDmaCommandSize(0),
  m_msgLongPacketSize(0),
  m_signalCommandSize(0),
  m_waitCommandSize(0),
  m_resetSyncObjectsCommandSize(0),
  m_arbitrationPacketCommandSize(0),
  m_arbitratorBaseAddressCommandSize(0),
  m_arbitratorDisableCommandSize(0),
  m_masterArbitratorBasicCommandSize(0),
  m_masterArbitratorSingleSlaveCommandSize(0),
  m_slaveArbitratorCommandSize(0),
  m_cpDmaCommandSize(0),
  m_loadAndExecCommandSize(0),
  m_fencePacketSize(0),
  m_fenceClearPacketSize(0),
  m_coeffTableConfPacketSize(0)
{
}

uint32_t CommandBufferPktGenerator::getQmanId(uint64_t engineId) const
{
    const unsigned numOfStreamsInQman = 4;

    if (engineId <= GAUDI_QUEUE_ID_DMA_1_3)
    {
        return ((uint64_t)GAUDI_ENGINE_ID_DMA_0 + (engineId / numOfStreamsInQman));
    }

    if (engineId >= GAUDI_QUEUE_ID_TPC_0_0)
    {
        return ((uint64_t)GAUDI_ENGINE_ID_TPC_0 + ((engineId - GAUDI_QUEUE_ID_TPC_0_0) / numOfStreamsInQman));
    }

    if (engineId >= GAUDI_QUEUE_ID_MME_0_0)
    {
        return ((uint64_t)GAUDI_ENGINE_ID_MME_0 + ((engineId - GAUDI_QUEUE_ID_MME_0_0) / numOfStreamsInQman));
    }

    return ((uint64_t)GAUDI_ENGINE_ID_DMA_2 + ((engineId - GAUDI_QUEUE_ID_DMA_2_0) / numOfStreamsInQman));
}

CommandBufferPktGenerator::~CommandBufferPktGenerator() {}

synStatus CommandBufferPktGenerator::generateLinDmaPacket(char*&         pPacket,
                                                          uint64_t&      packetSize,
                                                          uint64_t       srcAddress,
                                                          uint64_t       dstAddress,
                                                          uint32_t       size,
                                                          internalDmaDir direction,
                                                          uint32_t       contextId,
                                                          bool           isMemset,
                                                          uint32_t       engBarrier) const
{
    if (likely(pPacket != nullptr))
    {
        gaudi::GenLinDma::generateLinDma(pPacket,
                                         size,
                                         engBarrier,
                                         1 /*msgBarrier*/,
                                         direction,
                                         srcAddress,
                                         dstAddress,
                                         (contextId >> 8) & 0xFF /*dstContextIdHigh*/,
                                         contextId & 0xFF /*dstContextIdLow*/,
                                         0 /*wrComplete*/,
                                         0 /*transpose*/,
                                         0 /*dataType*/,
                                         isMemset /*memSet*/,
                                         0 /*compress*/,
                                         0 /*decompress*/);

        return synSuccess;
    }

    gaudi::GenLinDma dmaPacketGen(size,
                                  engBarrier,
                                  1 /*msgBarrier*/,
                                  direction,
                                  srcAddress,
                                  dstAddress,
                                  0 /*dstContextIdHigh*/,
                                  0 /*dstContextIdLow*/,
                                  0 /*wrComplete*/,
                                  0 /*transpose*/,
                                  0 /*dataType*/,
                                  isMemset /*memSet*/,
                                  0 /*compress*/,
                                  0 /*decompress*/);

    return _generatePacket(pPacket, packetSize, dmaPacketGen);
}

synStatus CommandBufferPktGenerator::generateNopPacket(char*& pPacket, uint64_t& packetSize, uint32_t barriers) const
{
    if (likely(pPacket != nullptr))
    {
        gaudi::GenNop::generatePacket(pPacket,
                                      (barriers & ENGINE_BARRIER) ? 1 : 0,
                                      (barriers & REGISTER_BARRIER) ? 1 : 0,
                                      (barriers & MESSAGE_BARRIER) ? 1 : 0);
        return synSuccess;
    }

    gaudi::GenNop nopPacketGen((barriers & ENGINE_BARRIER) ? 1 : 0,
                               (barriers & REGISTER_BARRIER) ? 1 : 0,
                               (barriers & MESSAGE_BARRIER) ? 1 : 0);

    return _generatePacket(pPacket, packetSize, nopPacketGen);
}

synStatus CommandBufferPktGenerator::generateSignalCommand(char*&    pPackets,
                                                           uint64_t& commandSize,
                                                           uint32_t  which,
                                                           int16_t   value,
                                                           uint16_t  operation,
                                                           uint32_t  barriers) const
{
    if (value < 0)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid sync-value {} ", HLLOG_FUNC, value);
        return synInvalidArgument;
    }

    uint64_t address = _getSyncObjectAddress(which);

    // Cannot use the struct in gaudi_packets...
    typedef union SyncObjectUpdate
    {
        struct
        {
            uint32_t sync_value : 15;
            uint32_t reserved   : 15;
            uint32_t te         : 1;
            uint32_t mode       : 1;
        };
        uint32_t control;
    } SyncObjectUpdate;
    SyncObjectUpdate syncObjectUpdate;
    syncObjectUpdate.sync_value = value;
    syncObjectUpdate.te         = 0;
    syncObjectUpdate.mode       = (operation != 0);

    if (likely(pPackets != nullptr))
    {
        gaudi::GenMsgLong::generatePacket(pPackets,
                                          syncObjectUpdate.control,
                                          0,  // set
                                          1,  // (barriers & ENGINE_BARRIER)   ? 1 : 0,
                                          1,  // (barriers & REGISTER_BARRIER) ? 1 : 0,
                                          1,  // (barriers & MESSAGE_BARRIER)  ? 1 : 0,
                                          address);
        return synSuccess;
    }

    gaudi::GenMsgLong signalPacketGen(syncObjectUpdate.control,
                                      0,  // set
                                      1,  // (barriers & ENGINE_BARRIER)   ? 1 : 0,
                                      1,  // (barriers & REGISTER_BARRIER) ? 1 : 0,
                                      1,  // (barriers & MESSAGE_BARRIER)  ? 1 : 0,
                                      address);

    return _generatePacket(pPackets, commandSize, signalPacketGen);  // check
}

synStatus CommandBufferPktGenerator::generateWaitCommand(char*&    pPackets,
                                                         uint64_t& commandSize,
                                                         uint32_t  waitQueueId,
                                                         uint32_t  monitorObjId,
                                                         uint32_t  syncObjId,
                                                         int16_t   syncObjValue,
                                                         uint16_t  operation,
                                                         uint32_t  barriers) const
{
    if (waitQueueId >= GAUDI_QUEUE_ID_SIZE)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid wait queue-ID {}", HLLOG_FUNC, waitQueueId);
        return synInvalidArgument;
    }

    if (monitorObjId > LAST_MONITOR_OBJ_ID)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid monitor-obj ID {}", HLLOG_FUNC, monitorObjId);
        return synInvalidArgument;
    }

    if (syncObjId > LAST_SYNC_OBJ_ID)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid sync-obj ID {}", HLLOG_FUNC, syncObjId);
        return synInvalidArgument;
    }

    if (syncObjValue < 0)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid sync-value {} for sync-obj ID {}", HLLOG_FUNC, syncObjValue, syncObjId);
        return synInvalidArgument;
    }

    const uint32_t engineBarrier   = 1;  // (barriers & ENGINE_BARRIER)   ? 1 : 0;
    const uint32_t registerBarrier = 1;  // (barriers & REGISTER_BARRIER) ? 1 : 0;
    const uint32_t messageBarrier  = 1;  // (barriers & MESSAGE_BARRIER)  ? 1 : 0;

    static const WaitID   signalingWaitId  = ID_0;
    static const uint32_t fenceTargetValue = 1;

    uint64_t fenceOffset = getCPFenceOffset((gaudi_queue_id)waitQueueId, signalingWaitId);

    ptrToInt fenceOffsetPtrToInt;
    fenceOffsetPtrToInt.u64 = fenceOffset;

    char*    pPacketOffset   = pPackets;
    uint64_t registerAddress = 0;
    // Monitor Setup
    //    Config Low-Address of the sync-obj:
    registerAddress = _getMonitorRegisterAddress(monitorObjId, MONITOR_REGISTER_PAYLOAD_LOW_ADDRESS);

    gaudi::GenMsgLong::generatePacket(pPacketOffset,
                                      fenceOffsetPtrToInt.u32[0],
                                      0,  // set
                                      engineBarrier,
                                      registerBarrier,
                                      messageBarrier,
                                      registerAddress,
                                      true);

    //    Config High-Address of the sync-obj:
    registerAddress = _getMonitorRegisterAddress(monitorObjId, MONITOR_REGISTER_PAYLOAD_HIGH_ADDRESS);
    gaudi::GenMsgLong::generatePacket(pPacketOffset,
                                      fenceOffsetPtrToInt.u32[1],
                                      0,  // set
                                      1,  // 0,
                                      1,  // 0,
                                      1,
                                      registerAddress,
                                      true);

    //    Config data for the sync-obj:
    registerAddress = _getMonitorRegisterAddress(monitorObjId, MONITOR_REGISTER_PAYLOAD_DATA);
    gaudi::GenMsgLong::generatePacket(pPacketOffset,
                                      fenceTargetValue,
                                      0,  // set
                                      1,  // 0,
                                      1,  // 0,
                                      1,
                                      registerAddress,
                                      true);

    // Cannot use the struct in gaudi_packets...
    typedef union MonitorArmRegister
    {
        struct
        {
            uint32_t sync_group_id : 8;
            uint32_t mask          : 8;
            uint32_t mode          : 1;
            uint32_t sync_value    : 15;
        };
        uint32_t control;
    } MonitorArmRegister;
    MonitorArmRegister monitorArmRegister;
    monitorArmRegister.sync_group_id = syncObjId / 8;
    monitorArmRegister.mask          = ~(1 << (syncObjId & 0x7));
    monitorArmRegister.mode          = operation;  // GEQ
    monitorArmRegister.sync_value    = syncObjValue;

    registerAddress = _getMonitorRegisterAddress(monitorObjId, MONITOR_REGISTER_PAYLOAD_MONITOR_ARM);
    gaudi::GenMsgLong::generatePacket(pPacketOffset,
                                      monitorArmRegister.control,
                                      0,  // set
                                      1,  // 0,
                                      1,  // 0,
                                      1,
                                      registerAddress,
                                      true);

    gaudi::GenFence::generatePacket(pPacketOffset, fenceTargetValue, fenceTargetValue, signalingWaitId, 1, 1, 1);

    return synSuccess;
}

synStatus CommandBufferPktGenerator::generateResetSyncObjectsCommand(char*&   pPackets,
                                                                     uint32_t firstSyncObjId,
                                                                     uint32_t numOfSyncObjects,
                                                                     uint32_t barriers) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Reset sync-obj packets");

    char* pTmpPackets = pPackets;  // As generate packet increments the pointer

    for (uint32_t syncObjId = firstSyncObjId; syncObjId < numOfSyncObjects; syncObjId++)
    {
        uint64_t address = _getSyncObjectAddress(syncObjId);

        GenMsgLong::generatePacket(pTmpPackets,
                                   0,
                                   0,  // Set
                                   (barriers & ENGINE_BARRIER) ? 1 : 0,
                                   (barriers & REGISTER_BARRIER) ? 1 : 0,
                                   (barriers & MESSAGE_BARRIER) ? 1 : 0,
                                   address,
                                   true);
    }

    return synSuccess;
}

synStatus CommandBufferPktGenerator::generateLoadPredicateCommand(char*& pPackets, uint64_t srcAddr) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Reset load and exec packets");

    char* pTmpPackets = pPackets;  // As generate packet increments the pointer

    GenLoadAndExecute::generatePacket(pTmpPackets,
                                      0x1,  // isLoad,
                                      0x0,  // isDst,
                                      0x0,  // isExecute,
                                      0x0,  // isEType,
                                      0x0,  // engBarrier,
                                      0x1,  // msgBarrier,
                                      srcAddr,
                                      true);

    return synSuccess;
}

synStatus CommandBufferPktGenerator::generateArbitrationCommand(
    char*&                        pPacket,
    uint64_t&                     packetSize,
    bool                          priorityRelease,
    generic::eArbitrationPriority priority /* = ARB_PRIORITY_NORMAL */) const
{
    HB_ASSERT((uint8_t)priority < std::numeric_limits<uint8_t>::max(), "Invalid priority value");

    if (pPacket)
    {
        gaudi::GenArbitrationPoint::generateArbitrationPoint(pPacket, (uint8_t)priority, priorityRelease);

        return synSuccess;
    }

    gaudi::GenArbitrationPoint arbPacketGen((uint8_t)priority, priorityRelease);

    return _generatePacket(pPacket, packetSize, arbPacketGen);
}

synStatus CommandBufferPktGenerator::generateArbitratorDisableConfigCommand(char*& pPackets, uint32_t qmanId) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Arbitration-disable command buffer");

    if (qmanId >= GAUDI_ENGINE_ID_SIZE)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid qman-id {} ", HLLOG_FUNC, qmanId);
        return synFail;
    }

    char* pTmpPackets = pPackets;  // As generate packet increments the pointer

    if (_generateArbitratorBaseAddressConfiguration(pTmpPackets, qmanId) != synSuccess)
    {
        return synFail;
    }

    uint64_t qmanBaseAddress = QMANS_QM_BASE_ADDRESS[qmanId];
    HB_ASSERT((qmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address (config register)");
    uint64_t configRegister = qmanBaseAddress + ARB_BASIC_CONFIG_REG_OFFSET;

    // Basic (disable) configuration
    qman::reg_arb_cfg_0 arbitratorConfiguration = {};
    arbitratorConfiguration.en                  = false;

    GenMsgLong::generatePacket(pTmpPackets,
                               arbitratorConfiguration._raw,
                               0,  // Set
                               1,  // eng-barrier
                               1,  // reg-barrier
                               1,  // msg-barrier
                               configRegister,
                               true);

    return synSuccess;
}

synStatus
CommandBufferPktGenerator::generateMasterArbitratorConfigCommand(char*&                           pPackets,
                                                                 generic::masterSlavesArbitration masterSlaveIds,
                                                                 bool                             isByPriority) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Arbitrator master-configuration command buffer");

    uint32_t masterQmanId = masterSlaveIds.masterQmanId;
    if (masterQmanId >= GAUDI_ENGINE_ID_SIZE)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid (master) qman-id {} ", HLLOG_FUNC, masterQmanId);
        return synFail;
    }

    const generic::CommonQmansIdDB& slaveQmansId = masterSlaveIds.slaveQmansId;
    if (slaveQmansId.size() >= MAX_SLAVE_PER_MASTER)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Invalid number of slaves ({}) for master qman-id {} ",
                HLLOG_FUNC,
                slaveQmansId.size(),
                masterQmanId);
        return synFail;
    }

    char* pTmpPackets = pPackets;  // As generate packet increments the pointer

    if (_generateArbitratorBaseAddressConfiguration(pTmpPackets, masterQmanId) != synSuccess)
    {
        return synFail;
    }

    uint64_t masterQmanBaseAddress = QMANS_QM_BASE_ADDRESS[masterQmanId];
    HB_ASSERT((masterQmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address");

    // Basic (master) configuration
    uint64_t configRegister = masterQmanBaseAddress + ARB_BASIC_CONFIG_REG_OFFSET;

    qman::reg_arb_cfg_0 arbitratorConfiguration = {};
    UNUSED(isByPriority);                   // In case RR is needed to be supported the WRR registers
                                            // are also needed to be set
    arbitratorConfiguration.type      = 0;  // Priority-bases arbitrator
    arbitratorConfiguration.is_master = 1;
    arbitratorConfiguration.en        = true;

    GenMsgLong::generatePacket(pTmpPackets,
                               arbitratorConfiguration._raw,
                               0,  // Set
                               1,  // eng-barrier
                               1,  // reg-barrier
                               1,  // msg-barrier
                               configRegister,
                               true);

    // Reset slave-mask variable
    qman::reg_arb_mst_slave_en slavesMask = {};
    slavesMask._raw                       = 0;

    uint32_t slaveId = 0;
    // Arbitration-choise configuration
    for (auto singleQmanId : slaveQmansId)
    {
        if (singleQmanId >= GAUDI_ENGINE_ID_SIZE)
        {
            LOG_ERR(SYN_STREAM, "{}: Invalid (slave) qman-id {} ", HLLOG_FUNC, masterQmanId);
            return synFail;
        }

        slavesMask.val |= (1 << slaveId);

        configRegister = masterQmanBaseAddress + ARB_MASTER_CHOISE_PUSH_REG_OFFSET +
                         sizeof(qman::reg_arb_mst_choise_push_ofst) * slaveId;

        qman::reg_arb_mst_choise_push_ofst masterChoisePushConfig = {};
        uint64_t                           slaveQmanBaseAddress   = QMANS_QM_BASE_ADDRESS[singleQmanId];
        HB_ASSERT((slaveQmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address");
        masterChoisePushConfig.val = slaveQmanBaseAddress - CFG_BASE + ARB_SLAVE_CHOISE_PUSH_REG_OFFSET;

        GenMsgLong::generatePacket(pTmpPackets,
                                   masterChoisePushConfig._raw,
                                   0,  // Set
                                   1,  // eng-barrier
                                   1,  // reg-barrier
                                   1,  // msg-barrier
                                   configRegister,
                                   true);

        slaveId++;
    }

    // Slave mask configuration
    configRegister = masterQmanBaseAddress + ARB_MASTER_SLAVE_MASK_REG_OFFSET;
    GenMsgLong::generatePacket(pTmpPackets,
                               slavesMask._raw,
                               0,  // Set
                               1,  // eng-barrier
                               1,  // reg-barrier
                               1,  // msg-barrier
                               configRegister,
                               true);

    return synSuccess;
}

synStatus CommandBufferPktGenerator::generateSlaveArbitratorConfigCommand(char*&   pPackets,
                                                                          uint32_t slaveId,
                                                                          uint32_t uSlaveQmanId,
                                                                          uint32_t uMasterQmanId) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Arbitrator slave-configuration command buffer");
    gaudi_engine_id slaveQmanId  = (gaudi_engine_id)uSlaveQmanId;
    gaudi_engine_id masterQmanId = (gaudi_engine_id)uMasterQmanId;

    if (slaveQmanId >= GAUDI_ENGINE_ID_SIZE)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid (slave) qman-id {} ", HLLOG_FUNC, slaveQmanId);
        return synFail;
    }

    if (masterQmanId >= GAUDI_ENGINE_ID_SIZE)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid (master) qman-id {} ", HLLOG_FUNC, masterQmanId);
        return synFail;
    }

    if (slaveId >= MAX_SLAVE_PER_MASTER - 1)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Invalid master's slave-index ({}) for master qman-id {} ",
                HLLOG_FUNC,
                slaveId,
                masterQmanId);
        return synFail;
    }

    char* pTmpPackets = pPackets;  // As generate packet increments the pointer

    if (_generateArbitratorBaseAddressConfiguration(pTmpPackets, slaveQmanId) != synSuccess)
    {
        return synFail;
    }

    uint64_t slaveQmanBaseAddress = QMANS_QM_BASE_ADDRESS[slaveQmanId];
    HB_ASSERT((slaveQmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address");

    // Basic (slave) configuration
    uint64_t            configRegister          = slaveQmanBaseAddress + ARB_BASIC_CONFIG_REG_OFFSET;
    qman::reg_arb_cfg_0 arbitratorConfiguration = {};
    arbitratorConfiguration.type                = 0;  // Priority-bases arbitrator
    arbitratorConfiguration.is_master           = 0;
    arbitratorConfiguration.en                  = true;

    gaudi::GenMsgLong::generatePacket(pTmpPackets,
                                      arbitratorConfiguration._raw,
                                      0,  // Set
                                      1,  // eng-barrier
                                      1,  // reg-barrier
                                      1,  // msg-barrier
                                      configRegister,
                                      true);
    // Slave-ID configuration
    configRegister                     = slaveQmanBaseAddress + ARB_SLAVE_ID_REG_OFFSET;
    qman::reg_arb_slv_id slaveIdConfig = {};
    slaveIdConfig.val                  = slaveId;

    gaudi::GenMsgLong::generatePacket(pTmpPackets,
                                      slaveIdConfig._raw,
                                      0,  // Set
                                      1,  // eng-barrier
                                      1,  // reg-barrier
                                      1,  // msg-barrier
                                      configRegister,
                                      true);

    // Increment-credit configuration
    configRegister = slaveQmanBaseAddress + ARB_SLAVE_CREDIT_INC_REG_OFFSET;
    qman::reg_arb_slv_master_inc_cred_ofst masterIncCreditConfig = {};
    uint64_t                               masterQmanBaseAddress = QMANS_QM_BASE_ADDRESS[masterQmanId];
    HB_ASSERT((masterQmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address");
    masterIncCreditConfig.val = masterQmanBaseAddress - CFG_BASE + ARB_MASTER_CREDIT_INC_REG_OFFSET;

    gaudi::GenMsgLong::generatePacket(pTmpPackets,
                                      masterIncCreditConfig._raw,
                                      0,  // Set
                                      1,  // eng-barrier
                                      1,  // reg-barrier
                                      1,  // msg-barrier
                                      configRegister,
                                      true);

    return synSuccess;
}

uint64_t CommandBufferPktGenerator::getLinDmaPacketSize()
{
    if (m_linDmaCommandSize != 0)
    {
        return m_linDmaCommandSize;
    }

    m_linDmaCommandSize = gaudi::GenLinDma::packetSize();
    return m_linDmaCommandSize;
}

void CommandBufferPktGenerator::getNopPacketSize(uint64_t& packetSize)
{
    gaudi::GenNop nopPacketGen(0, 0, 0);
    packetSize = nopPacketGen.getPacketSize();
}

uint64_t CommandBufferPktGenerator::getSignalCommandSize()
{
    if (m_signalCommandSize != 0)
    {
        return m_signalCommandSize;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    m_signalCommandSize = m_msgLongPacketSize;
    return m_signalCommandSize;
}

uint64_t CommandBufferPktGenerator::getFenceClearPacketCommandSize()
{
    if (m_fenceClearPacketSize != 0)
    {
        return m_fenceClearPacketSize;
    }
    if (m_msgLongPacketSize != 0)
    {
        m_fenceClearPacketSize = m_msgLongPacketSize;
        return m_fenceClearPacketSize;
    }

    m_fenceClearPacketSize = m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    return m_fenceClearPacketSize;
}

uint64_t CommandBufferPktGenerator::getFenceSetPacketCommandSize()
{
    if (m_fencePacketSize != 0)
    {
        return m_fencePacketSize;
    }
    // Fence
    m_fencePacketSize = gaudi::GenFence::packetSize();
    return m_fencePacketSize;
}

uint64_t CommandBufferPktGenerator::getWaitCommandSize()
{
    if (m_waitCommandSize != 0)
    {
        return m_waitCommandSize;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // Monitor Setup
    //    Config Low-Address of the sync-obj:
    m_waitCommandSize = m_msgLongPacketSize;
    //    Config High-Address of the sync-obj:
    m_waitCommandSize += m_msgLongPacketSize;
    //    Config data for the sync-obj:
    m_waitCommandSize += m_msgLongPacketSize;

    // Arm Monitor
    m_waitCommandSize += m_msgLongPacketSize;

    // Fence
    unsigned fencePacketSize = gaudi::GenFence::packetSize();

    m_waitCommandSize += fencePacketSize;

    return m_waitCommandSize;
}

void CommandBufferPktGenerator::getSingleResetSyncObjectsCommandSize(uint64_t& commandSize)
{
    if (m_resetSyncObjectsCommandSize != 0)
    {
        commandSize = m_resetSyncObjectsCommandSize;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    m_resetSyncObjectsCommandSize = m_msgLongPacketSize;
    commandSize                   = m_resetSyncObjectsCommandSize;
}

void CommandBufferPktGenerator::getLoadAndExecCommandSize(uint64_t& commandSize)
{
    if (m_loadAndExecCommandSize != 0)
    {
        commandSize = m_loadAndExecCommandSize;
        return;
    }

    commandSize = m_loadAndExecCommandSize = gaudi::GenLoadAndExecute::packetSize();
}

uint64_t CommandBufferPktGenerator::getArbitrationCommandSize()
{
    if (m_arbitrationPacketCommandSize != 0)
    {
        return m_arbitrationPacketCommandSize;
    }

    m_arbitrationPacketCommandSize = gaudi::GenArbitrationPoint::packetSize();

    return m_arbitrationPacketCommandSize;
}

void CommandBufferPktGenerator::getArbitratorDisableConfigCommandSize(uint64_t& commandSize)
{
    if (m_arbitratorDisableCommandSize != 0)
    {
        commandSize = m_arbitratorDisableCommandSize;
        return;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // Arbitration config-base setting
    m_arbitratorDisableCommandSize = _getArbitratorBaseAddressConfigCommandSize();
    // Basic (master) configuration
    m_arbitratorDisableCommandSize += m_msgLongPacketSize;

    commandSize = m_arbitratorDisableCommandSize;
}

void CommandBufferPktGenerator::getMasterArbitratorBasicConfigCommandSize(uint64_t& commandSize)
{
    if (m_masterArbitratorBasicCommandSize != 0)
    {
        commandSize = m_masterArbitratorBasicCommandSize;
        return;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // Arbitration config-base setting
    m_masterArbitratorBasicCommandSize = _getArbitratorBaseAddressConfigCommandSize();
    // Basic (master) configuration
    m_masterArbitratorBasicCommandSize += m_msgLongPacketSize;
    // Slave mask configuration
    m_masterArbitratorBasicCommandSize += m_msgLongPacketSize;

    commandSize = m_masterArbitratorBasicCommandSize;
}

void CommandBufferPktGenerator::getMasterSingleSlaveArbitratorConfigCommandSize(uint64_t& commandSize)
{
    if (m_masterArbitratorSingleSlaveCommandSize != 0)
    {
        commandSize = m_masterArbitratorSingleSlaveCommandSize;
        return;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // Arbitration-choise configuration
    m_masterArbitratorSingleSlaveCommandSize = m_msgLongPacketSize;

    commandSize = m_masterArbitratorSingleSlaveCommandSize;
}

void CommandBufferPktGenerator::getSlaveArbitratorConfigCommandSize(uint64_t& commandSize)
{
    if (m_slaveArbitratorCommandSize != 0)
    {
        commandSize = m_slaveArbitratorCommandSize;
        return;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // Arbitration config-base setting
    m_slaveArbitratorCommandSize = _getArbitratorBaseAddressConfigCommandSize();
    // Basic (slave) configuration
    m_slaveArbitratorCommandSize += m_msgLongPacketSize;
    // Slave-ID configuration
    m_slaveArbitratorCommandSize += m_msgLongPacketSize;
    // Increment-credit configuration
    m_slaveArbitratorCommandSize += m_msgLongPacketSize;

    commandSize = m_slaveArbitratorCommandSize;
}

synStatus CommandBufferPktGenerator::generateCpDma(char*&   pPacket,
                                                   uint32_t tsize,
                                                   uint32_t upperCp,
                                                   uint32_t engBarrier,
                                                   uint32_t msgBarrier,
                                                   uint64_t addr,
                                                   uint32_t predicate)
{
    gaudi::GenCpDma::generateCpDma(pPacket, tsize, engBarrier, msgBarrier, addr, predicate);
    return synSuccess;
}

void CommandBufferPktGenerator::generateDefaultCpDma(char*& pPacket, uint32_t tsize, uint64_t addr)
{
    gaudi::GenCpDma::generateDefaultCpDma(pPacket, tsize, addr);
}

uint64_t CommandBufferPktGenerator::getCpDmaSize()
{
    if (m_cpDmaCommandSize != 0)
    {
        return m_cpDmaCommandSize;
    }

    m_cpDmaCommandSize = gaudi::GenCpDma::packetSize();
    return m_cpDmaCommandSize;
}

uint64_t CommandBufferPktGenerator::_getSyncObjectAddress(unsigned syncObjId) const
{
    HB_ASSERT(syncObjId < NUM_OF_SYNC_OBJ, "syncObjId overflow {}", syncObjId);

    return SYNC_MGR_OBJ_BASE_ADDR + varoffsetof(block_sob_objs, sob_obj[syncObjId]);
}

uint64_t CommandBufferPktGenerator::_getMonitorRegisterAddress(unsigned            monitoObjId,
                                                               MonitorRegisterType registerType) const
{
    HB_ASSERT(monitoObjId < NUM_OF_MONITOR_OBJ, "monitoObjId overflow {}", monitoObjId);

    uint64_t registerOffset = 0;

    switch (registerType)
    {
        case MONITOR_REGISTER_PAYLOAD_LOW_ADDRESS:
            registerOffset = varoffsetof(block_sob_objs, mon_pay_addrl[monitoObjId]);
            break;

        case MONITOR_REGISTER_PAYLOAD_HIGH_ADDRESS:
            registerOffset = varoffsetof(block_sob_objs, mon_pay_addrh[monitoObjId]);
            break;

        case MONITOR_REGISTER_PAYLOAD_DATA:
            registerOffset = varoffsetof(block_sob_objs, mon_pay_data[monitoObjId]);
            break;

        case MONITOR_REGISTER_PAYLOAD_MONITOR_ARM:
            registerOffset = varoffsetof(block_sob_objs, mon_arm[monitoObjId]);
            break;
    }

    return SYNC_MGR_OBJ_BASE_ADDR + registerOffset;
}

synStatus CommandBufferPktGenerator::_generateArbitratorBaseAddressConfiguration(char*& pPackets, uint32_t qmanId) const
{
    HB_ASSERT_PTR(pPackets);
    HB_ASSERT((qmanId < GAUDI_ENGINE_ID_SIZE), "Invalid qman-id");

    ptrToInt configBaseAddress;
    configBaseAddress.u64 = CFG_BASE;

    uint64_t qmanBaseAddress = QMANS_QM_BASE_ADDRESS[qmanId];
    HB_ASSERT((qmanBaseAddress != INVALID_QMAN_BASE_ADDRESS), "Invalid QMAN base-address");
    uint64_t                           configRegister = qmanBaseAddress + ARB_BASIC_BASE_LOW_CFG_REG_OFFSET;
    gaudi::GenMsgLong::generatePacket(pPackets,
                                      configBaseAddress.u32[0],
                                      0,  // Set
                                      1,  // eng-barrier
                                      1,  // reg-barrier
                                      1,  // msg-barrier
                                      configRegister,
                                      true);
    configRegister = qmanBaseAddress + ARB_BASIC_BASE_HIGH_CFG_REG_OFFSET;
    gaudi::GenMsgLong::generatePacket(pPackets,
                                      configBaseAddress.u32[1],
                                      0,  // Set
                                      1,  // eng-barrier
                                      1,  // reg-barrier
                                      1,  // msg-barrier
                                      configRegister,
                                      true);

    return synSuccess;
}

uint64_t CommandBufferPktGenerator::_getArbitratorBaseAddressConfigCommandSize()
{
    if (m_arbitratorBaseAddressCommandSize != 0)
    {
        return m_arbitratorBaseAddressCommandSize;
    }

    if (m_msgLongPacketSize == 0)
    {
        m_msgLongPacketSize = gaudi::GenMsgLong::packetSize();
    }

    // High-address
    m_arbitratorBaseAddressCommandSize = m_msgLongPacketSize;
    // Low-address
    m_arbitratorBaseAddressCommandSize += m_msgLongPacketSize;

    return m_arbitratorBaseAddressCommandSize;
}

synStatus CommandBufferPktGenerator::generateFenceCommand(char*&    pPacket,
                                                          uint64_t& packetSize,
                                                          uint32_t  engBarrier,
                                                          uint32_t  regBarrier,
                                                          uint32_t  msgBarrier) const
{
    if (likely(pPacket != nullptr))
    {
        gaudi::GenFence::generatePacket(pPacket, 1, 1, (uint32_t)ID_0, engBarrier, regBarrier, msgBarrier);
        return synSuccess;
    }
    gaudi::GenFence fencePacket(1, 1, ID_0, engBarrier, regBarrier, msgBarrier);
    return _generatePacket(pPacket, packetSize, fencePacket);
}

synStatus CommandBufferPktGenerator::generateFenceClearCommand(char*&    pPacket,
                                                               uint64_t& packetSize,
                                                               uint64_t  streamId,
                                                               uint32_t  engBarrier,
                                                               uint32_t  regBarrier,
                                                               uint32_t  msgBarrier) const

{
    const WaitID   signalingWaitId = ID_0;
    uint64_t       fenceOffset     = gaudi::getCPFenceOffset((gaudi_queue_id)streamId, signalingWaitId);
    const uint32_t SET_OPCODE      = 0x0;

    if (likely(pPacket != nullptr))
    {
        gaudi::GenMsgLong::generatePacket(pPacket,
                                          1,  // Value
                                          SET_OPCODE,
                                          engBarrier,
                                          regBarrier,
                                          msgBarrier,
                                          fenceOffset);
        return synSuccess;
    }

    gaudi::GenMsgLong msgLongFenceClear(1,  // Value
                                        SET_OPCODE,
                                        engBarrier,
                                        regBarrier,
                                        msgBarrier,
                                        fenceOffset);

    return _generatePacket(pPacket, packetSize, msgLongFenceClear);
}

uint64_t CommandBufferPktGenerator::getCoeffTableConfigCommandSize()
{
    if (m_coeffTableConfPacketSize != 0)
    {
        return m_coeffTableConfPacketSize;
    }

    m_coeffTableConfPacketSize = gaudi::GenMsgLong::packetSize() * TPC_COEFF_TABLE_ADDR_REGS_NUM;
    return m_coeffTableConfPacketSize;
}

synStatus CommandBufferPktGenerator::generateCoeffTableConfigCommands(char*&   pPackets,
                                                                      ptrToInt tableBaseAddr,
                                                                      uint32_t singleCmdSize) const
{
    synStatus status       = synSuccess;
    char*     pCurrPackets = pPackets;
    uint32_t  tpcNum       = 0;

    for (uint64_t tpcBaseAddr : TPCS_CFG_BASE_ADDRESS)
    {
        status = generateCoeffTableConfigCommand(pCurrPackets, tpcBaseAddr, tableBaseAddr);
        if (status != synSuccess)
        {
            break;
        }

        pCurrPackets += singleCmdSize;
        LOG_TRACE(SYN_API, "Generate MSG Long to configure coeff table addr for TPC{}", tpcNum);
        tpcNum++;
    }

    return status;
}

synStatus CommandBufferPktGenerator::generateCoeffTableConfigCommand(char*&   pPackets,
                                                                     uint64_t tpcBaseAddr,
                                                                     ptrToInt tableBaseAddr) const
{
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pPackets, "Coeff table configuration command buffer");

    char* pTmpPackets = pPackets;

    for (unsigned table = 0, reg = 0; reg < TPC_COEFF_TABLE_ADDR_REGS_NUM; table++, reg += 2)
    {
        uint64_t configRegister = tpcBaseAddr + TPC_COEFF_TABLE_BASE_ADDRESS_REGS[reg];
        ptrToInt tableAddr;
        tableAddr.u64 = tableBaseAddr.u64 + TPC_COEFF_TABLE_BASE_ADDRESS[table];

        GenMsgLong::generatePacket(pTmpPackets,
                                   tableAddr.u32[0],
                                   0,  // Set
                                   1,  // eng-barrier
                                   1,  // reg-barrier
                                   1,  // msg-barrier
                                   configRegister,
                                   true);

        configRegister = tpcBaseAddr + TPC_COEFF_TABLE_BASE_ADDRESS_REGS[reg + 1];
        GenMsgLong::generatePacket(pTmpPackets,
                                   tableAddr.u32[1],
                                   0,  // Set
                                   1,  // eng-barrier
                                   1,  // reg-barrier
                                   1,  // msg-barrier
                                   configRegister,
                                   true);
    }
    return synSuccess;
}
