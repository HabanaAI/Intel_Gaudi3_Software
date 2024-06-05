#include <gtest/gtest.h>

#include "hal_reader/gaudi1/hal_reader.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "platform/gaudi/graph_compiler/sync/sync_conventions.h"
#include "gaudi/gaudi_packets.h"
#include "gaudi/asic_reg_structs/sob_objs_regs.h"
#include "utils.h"
#include "../utils.hpp"
#include "graph_compiler/compilation_hal_reader.h"
#include "node_factory.h"

using namespace gaudi;

class GaudiCommandToBinaryTest : public ::testing::Test
{
public:
    static void validateAddressFieldInfoSetContent(const AddressFieldInfoSet& AfciSet, const std::map<uint64_t, uint32_t>& targetToNewOffset);
};

void GaudiCommandToBinaryTest::validateAddressFieldInfoSetContent(const AddressFieldInfoSet& AfciSet, const std::map<uint64_t, uint32_t>& targetToNewOffset)
{
    ASSERT_EQ(AfciSet.size(), targetToNewOffset.size());

    for(auto &singleAddressFieldsInfoPair : AfciSet)
    {
        // We assume that the test assigned different target address to each address field info
        auto currentMapping = targetToNewOffset.find(singleAddressFieldsInfoPair.second->getTargetAddress());
        ASSERT_TRUE(targetToNewOffset.end() != currentMapping);

        // Compare updated address field info
        ASSERT_EQ(currentMapping->second, singleAddressFieldsInfoPair.first);
        ASSERT_EQ(currentMapping->second, singleAddressFieldsInfoPair.second->getFieldIndexOffset());
    }
}


TEST_F(GaudiCommandToBinaryTest, write_reg)
{
    const unsigned REG_OFFSET = 8;
    const unsigned VALUE = 3;
    const std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}};

    packet_wreg32 result = {0};
    BasicFieldsContainerInfo addrContainer;
    WriteRegister wRegCommand(REG_OFFSET, VALUE);


    std::pair<uint64_t, uint32_t> singleTargetToOffset = *targetToOffset.begin();
    addrContainer.addAddressEngineFieldInfo(nullptr,
                                     getMemorySectionNameForMemoryID(0),
                                     0,
                                     singleTargetToOffset.first,
                                     singleTargetToOffset.second,
                                     FIELD_MEMORY_TYPE_DRAM);

    wRegCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(wRegCommand.GetBinarySize(), sizeof(packet_wreg32));
    ASSERT_EQ(wRegCommand.writeInstruction(&result), sizeof(packet_wreg32));

    ASSERT_EQ(result.value, VALUE);
    ASSERT_EQ(result.reg_offset, REG_OFFSET);
    ASSERT_EQ(result.opcode, PACKET_WREG_32);
    ASSERT_EQ(result.pred, 0);
    ASSERT_EQ(result.reg_barrier, 1);
    ASSERT_EQ(result.eng_barrier, 0);
    ASSERT_EQ(result.msg_barrier, 1);


    wRegCommand.prepareFieldInfos();
    AddressFieldInfoSet afci = wRegCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet();
    validateAddressFieldInfoSetContent(afci, targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, eb_padding)
{
    const unsigned NUM_OF_PADDING = 30;
    const unsigned REG_OFFSET     = getRegForEbPadding();
    const unsigned VALUE          = 0;

    packet_wreg32 result[NUM_OF_PADDING] = {0};
    EbPadding     ebPaddingCommand(NUM_OF_PADDING);

    ASSERT_EQ(ebPaddingCommand.GetBinarySize(), sizeof(packet_wreg32) * NUM_OF_PADDING);
    ASSERT_EQ(ebPaddingCommand.writeInstruction(&result), sizeof(packet_wreg32) * NUM_OF_PADDING);

    for (unsigned i = 0; i < NUM_OF_PADDING; i++)
    {
        ASSERT_EQ(result[i].value, VALUE);
        ASSERT_EQ(result[i].reg_offset, REG_OFFSET);
        ASSERT_EQ(result[i].opcode, PACKET_WREG_32);
        ASSERT_EQ(result[i].pred, 0);
        ASSERT_EQ(result[i].reg_barrier, 0);
        ASSERT_EQ(result[i].eng_barrier, 0);
        ASSERT_EQ(result[i].msg_barrier, 0);
    }
}

TEST_F(GaudiCommandToBinaryTest, nop)
{
    packet_nop result = {0};
    Nop nopCommand;

    ASSERT_EQ(nopCommand.GetBinarySize(), sizeof(packet_nop));
    ASSERT_EQ(nopCommand.writeInstruction(&result), sizeof(packet_nop));

    ASSERT_EQ(result.opcode, PACKET_NOP);
    ASSERT_EQ(result.msg_barrier, 1);
    ASSERT_EQ(result.reg_barrier, 1);
    ASSERT_EQ(result.eng_barrier, 0);
}

TEST_F(GaudiCommandToBinaryTest, cp_dma)
{
    const deviceAddrOffset ADDR_PTR = 0x1000;
    const uint64_t SIZE = 0x103;
    const uint64_t DRAM_BASE = 0x1414;
    const uint32_t PREDICATE = 3;
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}, {0x1001, 1}};

    packet_cp_dma result = {0};
    CpDma cpDmaCommand(ADDR_PTR, SIZE, DRAM_BASE, PREDICATE);
    BasicFieldsContainerInfo addrContainer;

    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second += 2;
    }
    cpDmaCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(cpDmaCommand.GetBinarySize(), sizeof(packet_cp_dma));
    ASSERT_EQ(cpDmaCommand.writeInstruction(&result), sizeof(packet_cp_dma));

    ASSERT_EQ(result.opcode, PACKET_CP_DMA);
    ASSERT_EQ(result.tsize, SIZE);
    ASSERT_EQ(result.pred, PREDICATE);
    ASSERT_EQ(result.eng_barrier, 0x0);
    ASSERT_EQ(result.reg_barrier, 0x1);
    ASSERT_EQ(result.msg_barrier, 0x1);
    ASSERT_EQ(result.src_addr, ADDR_PTR);

    cpDmaCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(cpDmaCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, lin_dma)
{
    const deviceAddrOffset SRC_ADDR = 0x10000000;
    const deviceAddrOffset DST_ADDR = 0xf0000000;
    const uint64_t SIZE = 0x10300;
    const bool SET_ENG_BARRIER = true;
    const bool IS_MEMSET = false;
    const bool WR_COMPLETE = true;
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0},
                                                   {0x1001, 1},
                                                   {0x1002, 2},
                                                   {0x1003, 3}};

    packet_lin_dma result = {0};
    DmaDeviceInternal linDmaCommand(SRC_ADDR, false, DST_ADDR, false,
                                    SIZE, SET_ENG_BARRIER, IS_MEMSET, WR_COMPLETE);

    BasicFieldsContainerInfo addrContainer;

    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second += 2;
    }
    linDmaCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(linDmaCommand.GetBinarySize(), sizeof(packet_lin_dma));
    ASSERT_EQ(linDmaCommand.writeInstruction(&result), sizeof(packet_lin_dma));

    ASSERT_EQ(result.tsize, SIZE);

    ASSERT_EQ(result.wr_comp_en, WR_COMPLETE ? 1 : 0);
    ASSERT_EQ(result.dtype, 0);
    ASSERT_EQ(result.lin, 1);
    ASSERT_EQ(result.mem_set, IS_MEMSET);
    ASSERT_EQ(result.compress, 0);
    ASSERT_EQ(result.decompress, 0);
    ASSERT_EQ(result.reserved, 0);
    ASSERT_EQ(result.context_id_low, 0);
    ASSERT_EQ(result.opcode, PACKET_LIN_DMA);
    ASSERT_EQ(result.eng_barrier, SET_ENG_BARRIER);
    ASSERT_EQ(result.reg_barrier, 1);
    ASSERT_EQ(result.msg_barrier, 1);

    ASSERT_EQ(result.src_addr, SRC_ADDR);

    ASSERT_EQ(result.dst_addr_ctx_id_raw, DST_ADDR);

    linDmaCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(linDmaCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, monitor_setup)
{
    const unsigned NUM_OF_SHORT_PACKETS = 3;

    const SyncObjectManager::SyncId MON = 7;
    const WaitID WAIT_ID = ID_2;
    const HabanaDeviceType HABANA_DEVICE_TYPE = DEVICE_DMA_HOST_DEVICE;
    const unsigned DEVICE_ID = 40;
    const uint32_t VALUE = 0x94939291;
    const unsigned STREAM_ID = 0x40414243;
    const uint64_t ADDR = getCPFenceOffset(HABANA_DEVICE_TYPE, DEVICE_ID, WAIT_ID, STREAM_ID);
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0},
                                                   {0x1001, 1}};

    packet_msg_short result[3];
    memset(&result, 0, sizeof(result));
    MonitorSetup monitorSetupCommand(MON, WAIT_ID, HABANA_DEVICE_TYPE,
                                     DEVICE_ID, VALUE, STREAM_ID);
    ptrToInt p;
    p.u64 = ADDR;

    BasicFieldsContainerInfo addrContainer;
    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second =
                (singleTargetToNewOffset.second == 1 ? 2 : singleTargetToNewOffset.second);
    }
    monitorSetupCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(monitorSetupCommand.GetBinarySize(), 3 * sizeof(packet_msg_short));
    ASSERT_EQ(monitorSetupCommand.writeInstruction(&result), 3 * sizeof(packet_msg_short));

    unsigned monitorBlockBase = offsetof(block_sob_objs, mon_pay_addrl);
    for(unsigned i =  0; i < NUM_OF_SHORT_PACKETS; i++)
    {
        ASSERT_EQ(result[i].weakly_ordered, 0);
        ASSERT_EQ(result[i].no_snoop, 0);
        ASSERT_EQ(result[i].op, 0);
        ASSERT_EQ(result[i].base, 0);
        ASSERT_EQ(result[i].opcode, PACKET_MSG_SHORT);
        ASSERT_EQ(result[i].eng_barrier, 0);
        ASSERT_EQ(result[i].reg_barrier, 1);
        ASSERT_EQ(result[i].msg_barrier, 0);
    }

    ASSERT_EQ(result[0].msg_addr_offset, offsetof(block_sob_objs, mon_pay_addrl[MON]) - monitorBlockBase);
    ASSERT_EQ(result[1].msg_addr_offset, offsetof(block_sob_objs, mon_pay_addrh[MON]) - monitorBlockBase);
    ASSERT_EQ(result[2].msg_addr_offset, offsetof(block_sob_objs, mon_pay_data[MON]) - monitorBlockBase);

    ASSERT_EQ(result[0].value, p.u32[0]);
    ASSERT_EQ(result[1].value, p.u32[1]);
    ASSERT_EQ(result[2].value, VALUE);

    monitorSetupCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(monitorSetupCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, fence)
{
    const WaitID WAIT_ID = ID_1;
    const unsigned VALUE = 0x98;

    std::vector<packet_fence> result;
    Fence fenceCommand(WAIT_ID, VALUE);

    unsigned numPkts = (VALUE / 0xF) + 1;
    unsigned currFenceAggValue = VALUE;
    result.resize(numPkts);

    ASSERT_EQ(fenceCommand.writeInstruction(result.data()), result.size() * sizeof(packet_fence));
    ASSERT_EQ(fenceCommand.GetBinarySize(), result.size() * sizeof(packet_fence));

    for (unsigned i = 0 ; i < numPkts ; ++i)
    {
        if (i != numPkts - 1)
        {
            ASSERT_EQ(result[i].dec_val, 0xF);
            ASSERT_EQ(result[i].target_val, currFenceAggValue);
            currFenceAggValue -= 0xF;
        }
        else
        {
            ASSERT_EQ(result[i].dec_val, currFenceAggValue);
            ASSERT_EQ(result[i].target_val, currFenceAggValue);
            currFenceAggValue = 0;
        }

        ASSERT_EQ(result[i].id, WAIT_ID);

        ASSERT_EQ(result[i].pred, 0);
        ASSERT_EQ(result[i].opcode, PACKET_FENCE);
        ASSERT_EQ(result[i].eng_barrier, 0x0);
        ASSERT_EQ(result[i].reg_barrier, 0x1);
        ASSERT_EQ(result[i].msg_barrier, 0x0);
    }

    fenceCommand.prepareFieldInfos();
}

TEST_F(GaudiCommandToBinaryTest, wait)
{
    const WaitID WAIT_ID            = ID_2;
    const unsigned WAIT_CYCLES      = 42;
    const unsigned INCREMENT_VALUE  = 7;

    packet_wait result;
    Wait waitCommand(WAIT_ID, WAIT_CYCLES, INCREMENT_VALUE);

    ASSERT_EQ(waitCommand.writeInstruction((void*)&result), sizeof(packet_wait));
    ASSERT_EQ(waitCommand.GetBinarySize(), sizeof(packet_wait));

    ASSERT_EQ(result.num_cycles_to_wait, WAIT_CYCLES);
    ASSERT_EQ(result.inc_val, INCREMENT_VALUE);
    ASSERT_EQ(result.id, WAIT_ID);

    ASSERT_EQ(result.opcode, PACKET_WAIT);
    ASSERT_EQ(result.eng_barrier, 0x0);
    ASSERT_EQ(result.reg_barrier, 0x1);
    ASSERT_EQ(result.msg_barrier, 0x0);

    waitCommand.prepareFieldInfos();
}

TEST_F(GaudiCommandToBinaryTest, suspend)
{
    const WaitID WAIT_ID =              ID_3;
    const unsigned WAIT_CYCLES =        1984;
    const unsigned INCREMENT_VALUE =    5;

    const unsigned OUTPUT_SIZE = sizeof(packet_wait) + sizeof(packet_fence);
    char* pResult = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));

    Suspend suspendCommand(WAIT_ID, WAIT_CYCLES, INCREMENT_VALUE);

    ASSERT_EQ(suspendCommand.writeInstruction(pResult), OUTPUT_SIZE);
    ASSERT_EQ(suspendCommand.GetBinarySize(), OUTPUT_SIZE);

    unsigned currentResultOffset = 0;

    /****** Check wait part ******/
    auto* waitPacket = reinterpret_cast<packet_wait*>(pResult + currentResultOffset);

    ASSERT_EQ(waitPacket->num_cycles_to_wait, WAIT_CYCLES);
    ASSERT_EQ(waitPacket->inc_val, INCREMENT_VALUE);
    ASSERT_EQ(waitPacket->id, WAIT_ID);

    ASSERT_EQ(waitPacket->opcode, PACKET_WAIT);
    ASSERT_EQ(waitPacket->eng_barrier, 0x0);
    ASSERT_EQ(waitPacket->reg_barrier, 0x1);
    ASSERT_EQ(waitPacket->msg_barrier, 0x0);

    currentResultOffset += sizeof(packet_wait);

    /****** Check fence part ******/
    unsigned currFenceAggValue = INCREMENT_VALUE;
    unsigned numFencePkts = 1;

    auto* fencePackets = reinterpret_cast<packet_fence*>(pResult + currentResultOffset);
    for (unsigned i = 0 ; i < numFencePkts ; ++i)
    {
        if (i != numFencePkts - 1)
        {
            ASSERT_EQ(fencePackets[i].dec_val, 0xF);
            ASSERT_EQ(fencePackets[i].target_val, currFenceAggValue);
            currFenceAggValue -= 0xF;
        }
        else
        {
            ASSERT_EQ(fencePackets[i].dec_val, currFenceAggValue);
            ASSERT_EQ(fencePackets[i].target_val, currFenceAggValue);
            currFenceAggValue = 0;
        }

        ASSERT_EQ(fencePackets[i].id, WAIT_ID);
        ASSERT_EQ(fencePackets[i].pred, 0);
        ASSERT_EQ(fencePackets[i].opcode, PACKET_FENCE);
        ASSERT_EQ(fencePackets[i].eng_barrier, 0x0);
        ASSERT_EQ(fencePackets[i].reg_barrier, 0x1);
        ASSERT_EQ(fencePackets[i].msg_barrier, 0x0);
    }

    suspendCommand.prepareFieldInfos();
}

TEST_F(GaudiCommandToBinaryTest, upload_kernels_addr)
{
    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    const unsigned NUM_OF_PACKETS = 3;
    ptrToInt ADDR;
    ADDR.u64 = 0x1234567887654321;
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0},
                                                   {0x1001, 1}};

    packet_wreg32 result[NUM_OF_PACKETS];
    memset(&result, 0, sizeof(result));
    UploadKernelsAddr uploadKernelsAddrCommand(ADDR.u32[0], ADDR.u32[1]);

    BasicFieldsContainerInfo addrContainer;
    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second =
                (singleTargetToNewOffset.second == 1 ? 2 : singleTargetToNewOffset.second);
    }
    uploadKernelsAddrCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(uploadKernelsAddrCommand.GetBinarySize(), 3 * sizeof(packet_wreg32));
    ASSERT_EQ(uploadKernelsAddrCommand.writeInstruction(&result), 3 * sizeof(packet_wreg32));

    for (unsigned i = 0; i < NUM_OF_PACKETS; i++)
    {
        ASSERT_EQ(result[i].pred, DEFAULT_PREDICATE);
        ASSERT_EQ(result[i].opcode, PACKET_WREG_32);
        ASSERT_EQ(result[i].reg_barrier, 1);
        ASSERT_EQ(result[i].msg_barrier, 0);
    }

    ASSERT_EQ(result[0].reg_offset, offsetof(block_tpc, icache_base_adderess_low));
    ASSERT_EQ(result[0].value, ADDR.u32[0] & GaudiHalReader::instance(synDeviceGaudi)->getPrefetchAlignmentMask());
    ASSERT_EQ(result[0].eng_barrier, 0);

    ASSERT_EQ(result[1].reg_offset, offsetof(block_tpc, icache_base_adderess_high));
    ASSERT_EQ(result[1].value, ADDR.u32[1]);
    ASSERT_EQ(result[1].eng_barrier, 0);

    tpc::reg_tpc_cmd cmd;
    cmd._raw                 = 0;
    cmd.icache_prefetch_64kb = 1;
    ASSERT_EQ(result[2].reg_offset, offsetof(block_tpc, tpc_cmd));
    ASSERT_EQ(result[2].value, cmd._raw);
    ASSERT_EQ(result[2].eng_barrier, 1);
    uploadKernelsAddrCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(uploadKernelsAddrCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, write_many_registers)
{
    const unsigned FIRST_REG_OFFSET = 0x3;
    const unsigned COUNT_32BIT = 8;
    uint32_t VALUES[COUNT_32BIT] = {0x11111111, 0x22222222, 0x33333333, 0x44444444,
                                    0x55555555, 0x66666666, 0x77777777, 0x88888888};
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0},
                                                   {0x1001, 5},
                                                   {0x1002, 7}};

    const unsigned OUTPUT_SIZE = sizeof(packet_wreg32) * 2 + sizeof(packet_wreg_bulk) + (COUNT_32BIT-2) * sizeof(uint32_t);
    char* pOutput = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));
    WriteManyRegisters writeManyRegistersCommand(FIRST_REG_OFFSET, COUNT_32BIT, VALUES);

    BasicFieldsContainerInfo addrContainer;
    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second =
                (singleTargetToNewOffset.second == 0 ?
                 0 : singleTargetToNewOffset.second + 3);
    }
    writeManyRegistersCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(writeManyRegistersCommand.GetBinarySize(), OUTPUT_SIZE);
    ASSERT_EQ(writeManyRegistersCommand.writeInstruction(pOutput), OUTPUT_SIZE);

    unsigned currentBufferOffset = 0;
    unsigned currentRegIndex = 0;

    auto* firstWriteReg = reinterpret_cast<packet_wreg32*>(pOutput + currentBufferOffset);
    ASSERT_EQ(firstWriteReg->value, VALUES[currentRegIndex]);
    ASSERT_EQ(firstWriteReg->reg_offset, FIRST_REG_OFFSET + currentRegIndex);
    ASSERT_EQ(firstWriteReg->opcode, PACKET_WREG_32);
    ASSERT_EQ(firstWriteReg->pred, 0);
    ASSERT_EQ(firstWriteReg->eng_barrier, 0);
    ASSERT_EQ(firstWriteReg->reg_barrier, 1);
    ASSERT_EQ(firstWriteReg->msg_barrier, 1);
    currentRegIndex++;
    currentBufferOffset += sizeof(packet_wreg32);

    auto* writeBulk = reinterpret_cast<packet_wreg_bulk*>(pOutput + currentBufferOffset);
    ASSERT_EQ(writeBulk->size64, (COUNT_32BIT-2)/2);
    ASSERT_EQ(writeBulk->pred, DEFAULT_PREDICATE);
    ASSERT_EQ(writeBulk->reg_offset, FIRST_REG_OFFSET + currentRegIndex * sizeof(uint32_t));
    ASSERT_EQ(writeBulk->opcode, PACKET_WREG_BULK);
    ASSERT_EQ(writeBulk->eng_barrier, 0);
    ASSERT_EQ(writeBulk->reg_barrier, 1);
    ASSERT_EQ(writeBulk->msg_barrier, 1);

    for (; currentRegIndex < COUNT_32BIT - 1; currentRegIndex += 2)
    {
        ptrToInt currentVal;
        currentVal.u32[0] = VALUES[currentRegIndex];
        currentVal.u32[1] = VALUES[currentRegIndex + 1];
        ASSERT_EQ(writeBulk->values[currentRegIndex/2], currentVal.u64);
    }

    currentBufferOffset += sizeof(packet_wreg_bulk) + (COUNT_32BIT-2) * sizeof(uint32_t);

    auto* secondWriteReg = reinterpret_cast<packet_wreg32*>(pOutput + currentBufferOffset);
    ASSERT_EQ(secondWriteReg->value, VALUES[currentRegIndex]);
    ASSERT_EQ(secondWriteReg->reg_offset, FIRST_REG_OFFSET + currentRegIndex * sizeof(uint32_t));
    ASSERT_EQ(secondWriteReg->opcode, PACKET_WREG_32);
    ASSERT_EQ(secondWriteReg->pred, 0);
    ASSERT_EQ(secondWriteReg->eng_barrier, 0);
    ASSERT_EQ(secondWriteReg->reg_barrier, 1);
    ASSERT_EQ(secondWriteReg->msg_barrier, 1);

    writeManyRegistersCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(writeManyRegistersCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, monitor_arm)
{
    const SyncObjectManager::SyncId SYNC_OBJ = 5;
    const SyncObjectManager::SyncId MON = 7;
    const MonitorOp OPERATION = MONITOR_SO_OP_GREQ;
    const unsigned VALUE = 50;
    const Settable<uint8_t> MASK(20);

    unsigned monitorBlockBase = offsetof(block_sob_objs, mon_pay_addrl);

    uint8_t syncMask;
    unsigned syncGroupId;
    if (MASK.is_set())
    {
        syncMask     = ~(MASK.value());
        syncGroupId  = SYNC_OBJ;
    }
    else
    {
        syncMask     = ~(static_cast<uint8_t>(0x1U << (SYNC_OBJ % 8)));
        syncGroupId  = SYNC_OBJ / 8;
    }

    packet_msg_short result = {0};
    MonitorArm monitorArmCommand(SYNC_OBJ, MON, OPERATION, VALUE, MASK);

    ASSERT_EQ(monitorArmCommand.GetBinarySize(), sizeof(packet_msg_short));
    ASSERT_EQ(monitorArmCommand.writeInstruction(&result), sizeof(packet_msg_short));

    ASSERT_EQ(result.mon_arm_register.sync_group_id, syncGroupId);
    ASSERT_EQ(result.mon_arm_register.mask, syncMask);
    ASSERT_EQ(result.mon_arm_register.mode, OPERATION);
    ASSERT_EQ(result.mon_arm_register.sync_value, VALUE);
    ASSERT_EQ(result.msg_addr_offset, offsetof(block_sob_objs, mon_arm[MON]) - monitorBlockBase);
    ASSERT_EQ(result.weakly_ordered, 0);
    ASSERT_EQ(result.no_snoop, 0);
    ASSERT_EQ(result.op, 0);
    ASSERT_EQ(result.base, 0);
    ASSERT_EQ(result.opcode, PACKET_MSG_SHORT);
    ASSERT_EQ(result.eng_barrier, 0);
    ASSERT_EQ(result.reg_barrier, 1);
    ASSERT_EQ(result.msg_barrier, 0);
}

TEST_F(GaudiCommandToBinaryTest, wait_for_semaphore)
{
    const SyncObjectManager::SyncId SYNC_OBJ = 5;
    const SyncObjectManager::SyncId MON = 7;
    const MonitorOp OPERATION = MONITOR_SO_OP_GREQ;
    const unsigned VALUE = 50;
    const Settable<uint8_t> MASK(20);
    const WaitID WAIT_ID = ID_1;
    const unsigned FENCE_VALUE = 0x98;

    unsigned currentResultOffset = 0;

    unsigned numFencePkts = (FENCE_VALUE / 0xF) + 1;
    const unsigned OUTPUT_SIZE = sizeof(packet_msg_short) + numFencePkts * sizeof(packet_fence);
    char* pResult = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));
    WaitForSemaphore waitForSemaphoreCommand(SYNC_OBJ, MON, OPERATION, VALUE, MASK, WAIT_ID, FENCE_VALUE);

    ASSERT_EQ(waitForSemaphoreCommand.writeInstruction(pResult + currentResultOffset), OUTPUT_SIZE);
    ASSERT_EQ(waitForSemaphoreCommand.GetBinarySize(), OUTPUT_SIZE);


    /****** Check arm monitor part ******/
    unsigned monitorBlockBase = offsetof(block_sob_objs, mon_pay_addrl);

    uint8_t syncMask;
    unsigned syncGroupId;
    if (MASK.is_set())
    {
        syncMask     = ~(MASK.value());
        syncGroupId  = SYNC_OBJ;
    }
    else
    {
        syncMask     = ~(static_cast<uint8_t>(0x1U << (SYNC_OBJ % 8)));
        syncGroupId  = SYNC_OBJ / 8;
    }

    auto* armMonitorPacket = reinterpret_cast<packet_msg_short*>(pResult + currentResultOffset);

    ASSERT_EQ(armMonitorPacket->mon_arm_register.sync_group_id, syncGroupId);
    ASSERT_EQ(armMonitorPacket->mon_arm_register.mask, syncMask);
    ASSERT_EQ(armMonitorPacket->mon_arm_register.mode, OPERATION);
    ASSERT_EQ(armMonitorPacket->mon_arm_register.sync_value, VALUE);
    ASSERT_EQ(armMonitorPacket->msg_addr_offset, offsetof(block_sob_objs, mon_arm[MON]) - monitorBlockBase);
    ASSERT_EQ(armMonitorPacket->weakly_ordered, 0);
    ASSERT_EQ(armMonitorPacket->no_snoop, 0);
    ASSERT_EQ(armMonitorPacket->op, 0);
    ASSERT_EQ(armMonitorPacket->base, 0);
    ASSERT_EQ(armMonitorPacket->opcode, PACKET_MSG_SHORT);
    ASSERT_EQ(armMonitorPacket->eng_barrier, 0);
    ASSERT_EQ(armMonitorPacket->reg_barrier, 1);
    ASSERT_EQ(armMonitorPacket->msg_barrier, 0);

    currentResultOffset += sizeof(packet_msg_short);

    /****** Check fence part ******/
    unsigned currFenceAggValue = FENCE_VALUE;
    auto* fencePackets = reinterpret_cast<packet_fence*>(pResult + currentResultOffset);
    for (unsigned i = 0 ; i < numFencePkts ; ++i)
    {
        if (i != numFencePkts - 1)
        {
            ASSERT_EQ(fencePackets[i].dec_val, 0xF);
            ASSERT_EQ(fencePackets[i].target_val, currFenceAggValue);
            currFenceAggValue -= 0xF;
        }
        else
        {
            ASSERT_EQ(fencePackets[i].dec_val, currFenceAggValue);
            ASSERT_EQ(fencePackets[i].target_val, currFenceAggValue);
            currFenceAggValue = 0;
        }

        ASSERT_EQ(fencePackets[i].id, WAIT_ID);
        ASSERT_EQ(fencePackets[i].pred, 0);
        ASSERT_EQ(fencePackets[i].opcode, PACKET_FENCE);
        ASSERT_EQ(fencePackets[i].eng_barrier, 0x0);
        ASSERT_EQ(fencePackets[i].reg_barrier, 0x1);
        ASSERT_EQ(fencePackets[i].msg_barrier, 0x0);
    }

    waitForSemaphoreCommand.prepareFieldInfos();
}

TEST_F(GaudiCommandToBinaryTest, signal_semaphore)
{
    const SyncObjectManager::SyncId WHICH = 6;
    int16_t VALUE = 102;
    int                             OPERATION = SYNC_OP_ADD;
    packet_msg_short result = {0};
    SignalSemaphore signalSemaphoreCommand(WHICH, VALUE, OPERATION);

    ASSERT_EQ(signalSemaphoreCommand.GetBinarySize(), sizeof(packet_msg_short));
    ASSERT_EQ(signalSemaphoreCommand.writeInstruction(&result), sizeof(packet_msg_short));

    ASSERT_EQ(result.so_upd.sync_value, VALUE);
    ASSERT_EQ(result.so_upd.te, false);
    ASSERT_EQ(result.so_upd.mode, (OPERATION == SYNC_OP_ADD));
    ASSERT_EQ(result.msg_addr_offset, WHICH * sizeof(uint32_t));
    ASSERT_EQ(result.weakly_ordered, 0);
    ASSERT_EQ(result.no_snoop, 0);
    ASSERT_EQ(result.op, 0);
    ASSERT_EQ(result.base, 1);
    ASSERT_EQ(result.opcode, PACKET_MSG_SHORT);
    ASSERT_EQ(result.eng_barrier, (ALL_BARRIERS & ENGINE_BARRIER) ? 1 : 0);
    ASSERT_EQ(result.reg_barrier, (ALL_BARRIERS & REGISTER_BARRIER) ? 1 : 0);
    ASSERT_EQ(result.msg_barrier, (ALL_BARRIERS & MESSAGE_BARRIER) ? 1 : 0);
}


TEST_F(GaudiCommandToBinaryTest, signal_semaphore_with_predicate)
{
    const SyncObjectManager::SyncId WHICH = 6;
    int16_t VALUE = 102;
    int                             OPERATION = SYNC_OP_ADD;
    int PREDICATE = 31;
    packet_msg_long result = {0};
    packet_msg_short parsedResult{0};
    SignalSemaphoreWithPredicate signalSemaphoreCommand(WHICH, VALUE, PREDICATE, OPERATION);

    ASSERT_EQ(signalSemaphoreCommand.GetBinarySize(), sizeof(packet_msg_long));
    ASSERT_EQ(signalSemaphoreCommand.writeInstruction(&result), sizeof(packet_msg_long));

    parsedResult.value = result.value;
    ASSERT_EQ(parsedResult.so_upd.sync_value, VALUE);
    ASSERT_EQ(parsedResult.so_upd.te, false);
    ASSERT_EQ(parsedResult.so_upd.mode, (OPERATION == SYNC_OP_ADD));
    ASSERT_EQ(result.addr, getSyncObjectAddress(WHICH));
    ASSERT_EQ(result.weakly_ordered, 0);
    ASSERT_EQ(result.no_snoop, 0);
    ASSERT_EQ(result.op, 0);
    ASSERT_EQ(result.opcode, PACKET_MSG_LONG);
    ASSERT_EQ(result.eng_barrier, (ALL_BARRIERS & ENGINE_BARRIER) ? 1 : 0);
    ASSERT_EQ(result.reg_barrier, (ALL_BARRIERS & REGISTER_BARRIER) ? 1 : 0);
    ASSERT_EQ(result.msg_barrier, (ALL_BARRIERS & MESSAGE_BARRIER) ? 1 : 0);
}


TEST_F(GaudiCommandToBinaryTest, increment_fence)
{
    const HabanaDeviceType HABANA_DEVICE_TYPE = DEVICE_DMA_HOST_DEVICE;
    const unsigned DEVICE_ID = 40;
    const WaitID WAIT_ID = ID_2;
    const unsigned STREAM_ID = 0x40414243;
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}, {0x1001, 1}};

    /*IncrementFence(HabanaDeviceType deviceType, unsigned deviceID, WaitID waitID, unsigned streamID, uint32_t predicate = DEFAULT_PREDICATE);*/

    packet_msg_long result = {0};
    IncrementFence incrementFenceCommand(HABANA_DEVICE_TYPE, DEVICE_ID, WAIT_ID, STREAM_ID);
    BasicFieldsContainerInfo addrContainer;

    for (auto &singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                         getMemorySectionNameForMemoryID(0),
                                         0,
                                         singleTargetToNewOffset.first,
                                         singleTargetToNewOffset.second,
                                         FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second += 2;
    }
    incrementFenceCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(incrementFenceCommand.GetBinarySize(), sizeof(packet_cp_dma));
    ASSERT_EQ(incrementFenceCommand.writeInstruction(&result), sizeof(packet_cp_dma));

    ASSERT_EQ(result.opcode, PACKET_MSG_LONG);
    ASSERT_EQ(result.value, 1);
    ASSERT_EQ(result.pred, DEFAULT_PREDICATE);
    ASSERT_EQ(result.op, 0);
    ASSERT_EQ(result.msg_barrier, 0);
    ASSERT_EQ(result.reg_barrier, 1);
    ASSERT_EQ(result.eng_barrier, 0);
    ASSERT_EQ(result.addr, getCPFenceOffset(HABANA_DEVICE_TYPE, DEVICE_ID, WAIT_ID, STREAM_ID));

    incrementFenceCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(incrementFenceCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
                                       targetToOffset);
}

TEST_F(GaudiCommandToBinaryTest, dynamic_execute)
{
    const unsigned REG_OFFSET = 8;
    const unsigned VALUE      = 3;

    const SyncObjectManager::SyncId WHICH     = 6;
    int16_t                         SYNC_VALUE = 102;
    int                             OPERATION  = SYNC_OP_ADD;
    int                             PREDICATE = 31;

    std::vector<std::shared_ptr<GaudiQueueCommand>> cmdVector;
    cmdVector.push_back(std::make_unique<WriteRegister>(REG_OFFSET, VALUE));
    cmdVector.push_back(std::make_unique<SignalSemaphoreWithPredicate>(WHICH, SYNC_VALUE, PREDICATE, OPERATION));
    DynamicExecute dynamic(cmdVector, ENABLE_BYPASS);

    TSize dma_dims[] = {10, 20};
    const unsigned int dma_dim_num = 2;
    pTensor input = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pTensor output = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pNode node = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::memcpyNodeTypeName, "dma_memcpy");

    BasicFieldsContainerInfo bfci;
    size_t signalCount = DynamicExecuteFieldInfo::SINGLE_SIGNAL;
    bfci.add(std::make_shared<DynamicExecuteFieldInfo>(FieldType::FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL,
                                                                     signalCount, node, nullptr));
    dynamic.SetContainerInfo(bfci);
    dynamic.prepareFieldInfos();
    auto fieldInfo = std::dynamic_pointer_cast<DynamicShapeFieldInfo>
            (dynamic.getBasicFieldsContainerInfo().retrieveBasicFieldInfoSet().begin()->second);
    EXPECT_EQ(fieldInfo->getFieldIndexOffset(), 1);

    auto* metadata = (dynamic_execution_sm_params_t*)fieldInfo->getMetadata().data();
    EXPECT_EQ(metadata->cmd_len, 3);

    uint8_t* vector = (uint8_t*)metadata->commands;
    auto predMask = (1 << 5) - 1;
    EXPECT_EQ(vector[0] & predMask, PREDICATE);
    EXPECT_EQ(vector[8] & predMask, DEFAULT_PREDICATE);
}

TEST_F(GaudiCommandToBinaryTest, patch_execute)
{
    const unsigned REG_OFFSET = 8;
    const unsigned VALUE      = 3;
    const unsigned PREDICATE  = 31;

    std::vector<std::shared_ptr<GaudiQueueCommand>> cmdVector;
    cmdVector.push_back(std::make_unique<WriteRegister>(REG_OFFSET, VALUE));
    DynamicExecute dynamic(cmdVector, ENABLE_BYPASS);

    TSize dma_dims[] = {10, 20};
    const unsigned int dma_dim_num = 2;
    pTensor input = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pTensor output = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pNode node = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::memcpyNodeTypeName, "dma_memcpy");

    BasicFieldsContainerInfo bfci;
    size_t signalCount = DynamicExecuteFieldInfo::NO_SIGNAL;
    bfci.add(std::make_shared<DynamicExecuteFieldInfo>(FieldType::FIELD_DYNAMIC_EXECUTE_NO_SIGNAL,
                                                       signalCount, node, nullptr));
    dynamic.SetContainerInfo(bfci);
    dynamic.prepareFieldInfos();
    auto fieldInfo = std::dynamic_pointer_cast<DynamicShapeFieldInfo>
            (dynamic.getBasicFieldsContainerInfo().retrieveBasicFieldInfoSet().begin()->second);
    EXPECT_EQ(fieldInfo->getFieldIndexOffset(), 1);

    auto* metadata = (dynamic_execution_sm_params_t*)fieldInfo->getMetadata().data();
    EXPECT_EQ(metadata->cmd_len, 1);

    uint8_t* vector = (uint8_t*)metadata->commands;
    auto predMask = (1 << 5) - 1;
    EXPECT_EQ(vector[0] & predMask, PREDICATE);
}

TEST_F(GaudiCommandToBinaryTest, dynamic_execute_mme)
{
    const unsigned REG_OFFSET = 8;
    const unsigned VALUE      = 3;

    const SyncObjectManager::SyncId WHICH     = 6;
    int16_t                         SYNC_VALUE = 102;
    int                             OPERATION  = SYNC_OP_ADD;
    int                             PREDICATE = 31;

    std::vector<std::shared_ptr<GaudiQueueCommand>> cmdVector;
    cmdVector.push_back(std::make_unique<WriteRegister>(REG_OFFSET, VALUE));
    cmdVector.push_back(std::make_unique<SignalSemaphoreWithPredicate>(WHICH, SYNC_VALUE, PREDICATE, OPERATION));
    cmdVector.push_back(std::make_unique<SignalSemaphoreWithPredicate>(WHICH + 1, SYNC_VALUE, PREDICATE, OPERATION));
    DynamicExecute dynamic(cmdVector, DISABLE_BYPASS);

    // This is not an mme node, but thats fine, the node is only there to provide metadata about the tensors.
    TSize dma_dims[] = {10, 20};
    const unsigned int dma_dim_num = 2;
    pTensor input = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pTensor output = TensorPtr(new Tensor(dma_dim_num, dma_dims, syn_type_float, nullptr, nullptr, false, true));
    pNode node = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::memcpyNodeTypeName, "dma_memcpy");

    BasicFieldsContainerInfo bfci;
    size_t signalCount = DynamicExecuteFieldInfo::MME_SIGNAL_COUNT;
    bfci.add(std::make_shared<DynamicExecuteFieldInfo>(FieldType::FIELD_DYNAMIC_EXECUTE_MME,
                                                       signalCount, node, nullptr));
    dynamic.SetContainerInfo(bfci);
    dynamic.prepareFieldInfos();
    auto fieldInfo = std::dynamic_pointer_cast<DynamicShapeFieldInfo>
            (dynamic.getBasicFieldsContainerInfo().retrieveBasicFieldInfoSet().begin()->second);
    EXPECT_EQ(fieldInfo->getFieldIndexOffset(), 1);

    auto* metadata = (dynamic_execution_sm_params_t*)fieldInfo->getMetadata().data();
    EXPECT_EQ(metadata->cmd_len, 7);

    uint8_t* vector = (uint8_t*)metadata->commands;
    auto predMask = (1 << 5) - 1;
    EXPECT_EQ(vector[0] & predMask, PREDICATE);
    EXPECT_EQ(vector[8] & predMask, DEFAULT_PREDICATE);
    EXPECT_EQ(vector[24] & predMask, DEFAULT_PREDICATE);
}
