#include "graph_compiler/compilation_hal_reader.h"
#include "graph_compiler/sync/sync_types.h"
#include "graph_optimizer_test.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "platform/gaudi3/graph_compiler/block_data.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"

#include <gtest/gtest.h>

using namespace gaudi3;

class Gaudi3CommandToBinaryTest : public GraphOptimizerTest
{
public:
    static void validateAddressFieldInfoSetContent(const AddressFieldInfoSet&          AfciSet,
                                                   const std::map<uint64_t, uint32_t>& targetToNewOffset);
};

void Gaudi3CommandToBinaryTest::validateAddressFieldInfoSetContent(
    const AddressFieldInfoSet&          AfciSet,
    const std::map<uint64_t, uint32_t>& targetToNewOffset)
{
    ASSERT_EQ(AfciSet.size(), targetToNewOffset.size());

    for (auto& singleAddressFieldsInfoPair : AfciSet)
    {
        // We assume that the test assigned different target address to each address field info
        auto currentMapping = targetToNewOffset.find(singleAddressFieldsInfoPair.second->getTargetAddress());
        ASSERT_TRUE(targetToNewOffset.end() != currentMapping);

        // Compare updated address field info
        ASSERT_EQ(currentMapping->second, singleAddressFieldsInfoPair.first);
        ASSERT_EQ(currentMapping->second, singleAddressFieldsInfoPair.second->getFieldIndexOffset());
    }
}

TEST_F(Gaudi3CommandToBinaryTest, write_reg)
{
    const unsigned                     REG_OFFSET     = 8;
    const unsigned                     VALUE          = 3;
    const std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}};

    packet_wreg32            result = {0};
    BasicFieldsContainerInfo addrContainer;
    WriteRegister            wRegCommand(REG_OFFSET, VALUE);

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
    ASSERT_EQ(result.reg, 0);
    ASSERT_EQ(result.eng_barrier, 0);
    ASSERT_EQ(result.msg_barrier, 1);
    ASSERT_EQ(result.swtc, 0);

    wRegCommand.prepareFieldInfos();
    AddressFieldInfoSet afci = wRegCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet();
    validateAddressFieldInfoSetContent(afci, targetToOffset);
}

TEST_F(Gaudi3CommandToBinaryTest, eb_padding)
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
        ASSERT_EQ(result[i].reg, 0);
        ASSERT_EQ(result[i].eng_barrier, 0);
        ASSERT_EQ(result[i].msg_barrier, 0);
        ASSERT_EQ(result[i].swtc, 0);
    }
}

TEST_F(Gaudi3CommandToBinaryTest, Execute)
{
    const unsigned                     REG_OFFSET     = gaudi3::getRegForExecute(DEVICE_TPC, 0);
    const std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}};

    packet_wreg32            result = {0};
    BasicFieldsContainerInfo addrContainer;
    Execute                  executeCommand(DEVICE_TPC);

    std::pair<uint64_t, uint32_t> singleTargetToOffset = *targetToOffset.begin();
    addrContainer.addAddressEngineFieldInfo(nullptr,
                                            getMemorySectionNameForMemoryID(0),
                                            0,
                                            singleTargetToOffset.first,
                                            singleTargetToOffset.second,
                                            FIELD_MEMORY_TYPE_DRAM);

    executeCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(executeCommand.GetBinarySize(), sizeof(packet_wreg32));
    ASSERT_EQ(executeCommand.writeInstruction(&result), sizeof(packet_wreg32));

    ASSERT_EQ(result.value, 0x1);
    ASSERT_EQ(result.reg_offset, REG_OFFSET);
    ASSERT_EQ(result.opcode, PACKET_WREG_32);
    ASSERT_EQ(result.pred, 0);
    ASSERT_EQ(result.reg, 0);
    ASSERT_EQ(result.eng_barrier, 0);
    ASSERT_EQ(result.msg_barrier, 1);
    ASSERT_EQ(result.swtc, 0);

    executeCommand.prepareFieldInfos();
    AddressFieldInfoSet afci = executeCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet();
    validateAddressFieldInfoSetContent(afci, targetToOffset);
}

TEST_F(Gaudi3CommandToBinaryTest, nop)
{
    packet_nop result = {0};
    Nop        nopCommand;

    ASSERT_EQ(nopCommand.GetBinarySize(), sizeof(packet_nop));
    ASSERT_EQ(nopCommand.writeInstruction(&result), sizeof(packet_nop));

    ASSERT_EQ(result.opcode, PACKET_NOP);
    ASSERT_EQ(result.eng_barrier, 0x0);
    ASSERT_EQ(result.msg_barrier, 0x1);
}

TEST_F(Gaudi3CommandToBinaryTest, wait)
{
    const WaitID   WAIT_ID         = ID_2;
    const unsigned WAIT_CYCLES     = 42;
    const unsigned INCREMENT_VALUE = 7;

    packet_wait result;
    Wait        waitCommand(WAIT_ID, WAIT_CYCLES, INCREMENT_VALUE);

    ASSERT_EQ(waitCommand.writeInstruction((void*)&result), sizeof(packet_wait));
    ASSERT_EQ(waitCommand.GetBinarySize(), sizeof(packet_wait));

    ASSERT_EQ(result.num_cycles_to_wait, WAIT_CYCLES);
    ASSERT_EQ(result.inc_val, INCREMENT_VALUE);
    ASSERT_EQ(result.id, WAIT_ID);

    ASSERT_EQ(result.opcode, PACKET_WAIT);
    ASSERT_EQ(result.eng_barrier, 0x0);
    ASSERT_EQ(result.msg_barrier, 0x0);

    waitCommand.prepareFieldInfos();
}

TEST_F(Gaudi3CommandToBinaryTest, write_many_registers)
{
    const unsigned FIRST_REG_OFFSET = 0x3;
    const unsigned COUNT_32BIT      = 8;
    uint32_t       VALUES[COUNT_32BIT] =
        {0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777, 0x88888888};
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}, {0x1001, 5}, {0x1002, 7}};

    const unsigned OUTPUT_SIZE =
        sizeof(packet_wreg32) * 2 + sizeof(packet_wreg_bulk) + (COUNT_32BIT - 2) * sizeof(uint32_t);
    char*              pOutput = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));
    WriteManyRegisters writeManyRegistersCommand(FIRST_REG_OFFSET, COUNT_32BIT, VALUES);

    BasicFieldsContainerInfo addrContainer;
    for (auto& singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                                getMemorySectionNameForMemoryID(0),
                                                0,
                                                singleTargetToNewOffset.first,
                                                singleTargetToNewOffset.second,
                                                FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second = (singleTargetToNewOffset.second == 0 ? 0 : singleTargetToNewOffset.second + 3);
    }
    writeManyRegistersCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(writeManyRegistersCommand.GetBinarySize(), OUTPUT_SIZE);
    ASSERT_EQ(writeManyRegistersCommand.writeInstruction(pOutput), OUTPUT_SIZE);

    unsigned currentBufferOffset = 0;
    unsigned currentRegIndex     = 0;

    auto* firstWriteReg = reinterpret_cast<packet_wreg32*>(pOutput + currentBufferOffset);
    ASSERT_EQ(firstWriteReg->value, VALUES[currentRegIndex]);
    ASSERT_EQ(firstWriteReg->reg_offset, FIRST_REG_OFFSET + currentRegIndex);
    ASSERT_EQ(firstWriteReg->opcode, PACKET_WREG_32);
    ASSERT_EQ(firstWriteReg->pred, 0);
    ASSERT_EQ(firstWriteReg->eng_barrier, 0);
    ASSERT_EQ(firstWriteReg->swtc, 0);
    ASSERT_EQ(firstWriteReg->msg_barrier, 1);
    currentRegIndex++;
    currentBufferOffset += sizeof(packet_wreg32);

    auto* writeBulk = reinterpret_cast<packet_wreg_bulk*>(pOutput + currentBufferOffset);
    ASSERT_EQ(writeBulk->size64, (COUNT_32BIT - 2) / 2);
    ASSERT_EQ(writeBulk->pred, 0);
    ASSERT_EQ(writeBulk->reg_offset, FIRST_REG_OFFSET + currentRegIndex * sizeof(uint32_t));
    ASSERT_EQ(writeBulk->opcode, PACKET_WREG_BULK);
    ASSERT_EQ(writeBulk->eng_barrier, 0);
    ASSERT_EQ(writeBulk->msg_barrier, 1);
    ASSERT_EQ(firstWriteReg->swtc, 0);

    for (; currentRegIndex < COUNT_32BIT - 1; currentRegIndex += 2)
    {
        ptrToInt currentVal;
        currentVal.u32[0] = VALUES[currentRegIndex];
        currentVal.u32[1] = VALUES[currentRegIndex + 1];
        ASSERT_EQ(writeBulk->values[currentRegIndex / 2], currentVal.u64);
    }

    currentBufferOffset += sizeof(packet_wreg_bulk) + (COUNT_32BIT - 2) * sizeof(uint32_t);

    auto* secondWriteReg = reinterpret_cast<packet_wreg32*>(pOutput + currentBufferOffset);
    ASSERT_EQ(secondWriteReg->value, VALUES[currentRegIndex]);
    ASSERT_EQ(secondWriteReg->reg_offset, FIRST_REG_OFFSET + currentRegIndex * sizeof(uint32_t));
    ASSERT_EQ(secondWriteReg->opcode, PACKET_WREG_32);
    ASSERT_EQ(secondWriteReg->pred, 0);
    ASSERT_EQ(secondWriteReg->eng_barrier, 0);
    ASSERT_EQ(secondWriteReg->reg, 0);
    ASSERT_EQ(secondWriteReg->msg_barrier, 1);
    ASSERT_EQ(firstWriteReg->swtc, 0);

    writeManyRegistersCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(
        writeManyRegistersCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
        targetToOffset);
}

TEST_F(Gaudi3CommandToBinaryTest, fence)
{
    const WaitID   WAIT_ID = ID_1;
    const uint32_t VALUE   = 0x98;

    std::vector<packet_fence> result;
    Fence                     fenceCommand(WAIT_ID, VALUE);

    unsigned numPkts           = (VALUE / 0xF) + 1;
    unsigned currFenceAggValue = VALUE;
    result.resize(numPkts);

    ASSERT_EQ(fenceCommand.writeInstruction(result.data()), result.size() * sizeof(packet_fence));
    ASSERT_EQ(fenceCommand.GetBinarySize(), result.size() * sizeof(packet_fence));

    for (unsigned i = 0; i < numPkts; ++i)
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
        ASSERT_EQ(result[i].msg_barrier, 0x0);
    }

    fenceCommand.prepareFieldInfos();
}

TEST_F(Gaudi3CommandToBinaryTest, suspend)
{
    const WaitID   WAIT_ID         = ID_3;
    const unsigned WAIT_CYCLES     = 1984;
    const unsigned INCREMENT_VALUE = 5;

    const unsigned OUTPUT_SIZE = sizeof(packet_wait) + sizeof(packet_fence);
    char*          pResult     = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));

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
    ASSERT_EQ(waitPacket->msg_barrier, 0x0);

    currentResultOffset += sizeof(packet_wait);

    /****** Check fence part ******/
    unsigned currFenceAggValue = INCREMENT_VALUE;
    unsigned numFencePkts      = 1;

    auto* fencePackets = reinterpret_cast<packet_fence*>(pResult + currentResultOffset);
    for (unsigned i = 0; i < numFencePkts; ++i)
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
        ASSERT_EQ(fencePackets[i].msg_barrier, 0x0);
    }

    suspendCommand.prepareFieldInfos();
}

TEST_F(Gaudi3CommandToBinaryTest, invalidate_tpc_caches)
{
    tpc::reg_tpc_cmd cmd;
    cmd._raw              = 0;
    cmd.icache_invalidate = 1;
    cmd.dcache_invalidate = 1;
    cmd.lcache_invalidate = 1;
    cmd.tcache_invalidate = 1;

    const std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}};

    packet_wreg32            result = {0};
    BasicFieldsContainerInfo addrContainer;
    InvalidateTPCCaches      invTpcCachesCommand(DEFAULT_PREDICATE);

    std::pair<uint64_t, uint32_t> singleTargetToOffset = *targetToOffset.begin();
    addrContainer.addAddressEngineFieldInfo(nullptr,
                                            getMemorySectionNameForMemoryID(0),
                                            0,
                                            singleTargetToOffset.first,
                                            singleTargetToOffset.second,
                                            FIELD_MEMORY_TYPE_DRAM);

    invTpcCachesCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(invTpcCachesCommand.GetBinarySize(), sizeof(packet_wreg32));
    ASSERT_EQ(invTpcCachesCommand.writeInstruction(&result), sizeof(packet_wreg32));

    ASSERT_EQ(result.value, cmd._raw);
    ASSERT_EQ(result.reg_offset, GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd));
    ASSERT_EQ(result.opcode, PACKET_WREG_32);
    ASSERT_EQ(result.pred, DEFAULT_PREDICATE);
    ASSERT_EQ(result.reg, 0);
    ASSERT_EQ(result.eng_barrier, 1);
    ASSERT_EQ(result.msg_barrier, 1);
    ASSERT_EQ(result.swtc, 0);

    invTpcCachesCommand.prepareFieldInfos();
    AddressFieldInfoSet afci = invTpcCachesCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet();
    validateAddressFieldInfoSetContent(afci, targetToOffset);
}

TEST_F(Gaudi3CommandToBinaryTest, upload_kernels_addr)
{
    CompilationHalReader::setHalReader(Gaudi3HalReader::instance());
    const unsigned NUM_OF_PADDING = Gaudi3HalReader::instance()->getNumUploadKernelEbPad();
    const unsigned NUM_OF_PACKETS = 3;
    ptrToInt       ADDR;
    ADDR.u64                                    = 0x1234567887654321;
    std::map<uint64_t, uint32_t> targetToOffset = {{0x1000, 0}, {0x1001, 1}};

    packet_wreg32 result[NUM_OF_PACKETS + NUM_OF_PADDING];
    memset(&result, 0, sizeof(result));
    UploadKernelsAddr uploadKernelsAddrCommand(ADDR.u32[0], ADDR.u32[1]);

    BasicFieldsContainerInfo addrContainer;
    for (auto& singleTargetToNewOffset : targetToOffset)
    {
        addrContainer.addAddressEngineFieldInfo(nullptr,
                                                getMemorySectionNameForMemoryID(0),
                                                0,
                                                singleTargetToNewOffset.first,
                                                singleTargetToNewOffset.second + NUM_OF_PADDING,
                                                FIELD_MEMORY_TYPE_DRAM);

        singleTargetToNewOffset.second = (singleTargetToNewOffset.second == 1 ? (NUM_OF_PADDING + 1) * 2 :  NUM_OF_PADDING * 2);
    }
    uploadKernelsAddrCommand.SetContainerInfo(addrContainer);

    ASSERT_EQ(uploadKernelsAddrCommand.GetBinarySize(), (NUM_OF_PACKETS + NUM_OF_PADDING) * sizeof(packet_wreg32));
    ASSERT_EQ(uploadKernelsAddrCommand.writeInstruction(&result), (NUM_OF_PACKETS + NUM_OF_PADDING) * sizeof(packet_wreg32));

    for (unsigned i = 0; i < NUM_OF_PACKETS + NUM_OF_PADDING; i++)
    {
        ASSERT_EQ(result[i].pred, DEFAULT_PREDICATE);
        ASSERT_EQ(result[i].opcode, PACKET_WREG_32);
        ASSERT_EQ(result[i].msg_barrier, 0);
    }

    ASSERT_EQ(result[NUM_OF_PADDING].reg_offset, GET_ADDR_OF_TPC_BLOCK_FIELD(icache_base_adderess_low));
    ASSERT_EQ(result[NUM_OF_PADDING].value, ADDR.u32[0] & Gaudi3HalReader::instance()->getPrefetchAlignmentMask());
    ASSERT_EQ(result[NUM_OF_PADDING].eng_barrier, 0);

    ASSERT_EQ(result[NUM_OF_PADDING + 1].reg_offset, GET_ADDR_OF_TPC_BLOCK_FIELD(icache_base_adderess_high));
    ASSERT_EQ(result[NUM_OF_PADDING + 1].value, ADDR.u32[1]);
    ASSERT_EQ(result[NUM_OF_PADDING + 1].eng_barrier, 0);

    tpc::reg_tpc_cmd cmd;
    cmd._raw                 = 0;
    cmd.icache_prefetch_64kb = 1;
    ASSERT_EQ(result[NUM_OF_PADDING + 2].reg_offset, GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd));
    ASSERT_EQ(result[NUM_OF_PADDING + 2].value, cmd._raw);
    ASSERT_EQ(result[NUM_OF_PADDING + 2].eng_barrier, 1);
    uploadKernelsAddrCommand.prepareFieldInfos();
    validateAddressFieldInfoSetContent(
        uploadKernelsAddrCommand.getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet(),
        targetToOffset);
}

TEST_F(Gaudi3CommandToBinaryTest, qman_delay)
{
    const WaitID   WAIT_ID         = ID_2;
    const unsigned INCREMENT_VALUE = 1;

    const unsigned OUTPUT_SIZE = sizeof(packet_wait) + sizeof(packet_wreg32);
    char*          pResult     = reinterpret_cast<char*>(alloca(OUTPUT_SIZE));

    QmanDelay qmanDelayCommand(DEFAULT_PREDICATE);

    ASSERT_EQ(qmanDelayCommand.writeInstruction(pResult), OUTPUT_SIZE);
    ASSERT_EQ(qmanDelayCommand.GetBinarySize(), OUTPUT_SIZE);

    unsigned currentResultOffset = 0;

    /****** Check wreg32 part ******/
    auto* wregPacket = reinterpret_cast<packet_wreg32*>(pResult + currentResultOffset);

    uint16_t reg_offset =
        QMAN_BLOCK_BASE + offsetof(block_qman, cp_fence2_rdata) + sizeof(struct qman::reg_cp_fence2_rdata) * 4;
    ASSERT_EQ(wregPacket->reg_offset, reg_offset);
    ASSERT_EQ(wregPacket->reg, 0x0);
    ASSERT_EQ(wregPacket->value, 0x1);
    ASSERT_EQ(wregPacket->pred, DEFAULT_PREDICATE);
    ASSERT_EQ(wregPacket->swtc, 0x0);

    ASSERT_EQ(wregPacket->opcode, PACKET_WREG_32);
    ASSERT_EQ(wregPacket->eng_barrier, 0x0);
    ASSERT_EQ(wregPacket->msg_barrier, 0x1);

    currentResultOffset += sizeof(packet_wreg32);

    /****** Check fence part ******/
    unsigned currFenceAggValue = INCREMENT_VALUE;
    unsigned numFencePkts      = 1;

    auto* fencePackets = reinterpret_cast<packet_fence*>(pResult + currentResultOffset);
    for (unsigned i = 0; i < numFencePkts; ++i)
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
        ASSERT_EQ(fencePackets[i].pred, DEFAULT_PREDICATE);
        ASSERT_EQ(fencePackets[i].opcode, PACKET_FENCE);
        ASSERT_EQ(fencePackets[i].eng_barrier, 0x0);
        ASSERT_EQ(fencePackets[i].msg_barrier, 0x0);
    }

    qmanDelayCommand.prepareFieldInfos();
}
