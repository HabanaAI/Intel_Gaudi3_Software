#include <string>
#include <gtest/gtest.h>
#include "platform/gaudi3/graph_compiler/command_queue.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "platform/gaudi3/graph_compiler/block_data.h"
#include "graph_optimizer_test.h"

using namespace gaudi3;

static const unsigned REG_OFFSET = 8;

class CommandQueueGaudi3Test : public GraphOptimizerTest
{
};

TEST_F(CommandQueueGaudi3Test, write_reg)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    WriteRegister* wr = new WriteRegister(REG_OFFSET, 0);
    cmdq->PushBack(QueueCommandPtr {wr});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += wr->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    ASSERT_EQ(wr->getRegOffset(), REG_OFFSET);
    delete cmdq;
    // Queue destructor will delete the wr command
}

TEST_F(CommandQueueGaudi3Test, eb_padding)
{
    unsigned      numPadding          = 30;
    uint32_t      regOffset           = getRegForEbPadding();
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    EbPadding* eb = new EbPadding(numPadding);
    cmdq->PushBack(QueueCommandPtr {eb});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += eb->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    ASSERT_EQ(eb->getRegOffset(), regOffset);
    delete cmdq;
    // Queue destructor will delete the wr command
}

static WriteManyRegisters* createWriteManyRegs(unsigned numRegs)
{
    unsigned* regVals = new unsigned[numRegs];

    for (unsigned i = 0; i < numRegs; ++i)
    {
        regVals[i] = i;
    }

    WriteManyRegisters* wmr = new WriteManyRegisters(REG_OFFSET, numRegs, regVals);
    delete[] regVals;
    return wmr;
}

static void testWriteManyRegsBasic(unsigned numRegs)
{
    unsigned            expectedNumElements = 0;
    unsigned            expectedBinSize     = 0;
    WriteManyRegisters* wmr                 = createWriteManyRegs(numRegs);
    CommandQueue*       cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_EQ(wmr->GetFirstReg(), REG_OFFSET);
    ASSERT_EQ(wmr->GetCount(), numRegs);

    cmdq->PushBack(QueueCommandPtr {wmr});
    expectedNumElements++;
    expectedBinSize += wmr->GetBinarySize();

    // Add yet another WriteRegister to the queue
    WriteRegister* wr = new WriteRegister(REG_OFFSET, 0);
    cmdq->PushBack(QueueCommandPtr {wr});
    expectedNumElements++;
    expectedBinSize += wr->GetBinarySize();

    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the commands
}

TEST_F(CommandQueueGaudi3Test, write_many_regs_even_basic)
{
    testWriteManyRegsBasic(6);
}

TEST_F(CommandQueueGaudi3Test, write_many_regs_odd_basic)
{
    testWriteManyRegsBasic(7);
}

TEST_F(CommandQueueGaudi3Test, write_many_regs_only_one_basic)
{
    testWriteManyRegsBasic(1);
}

TEST_F(CommandQueueGaudi3Test, wait)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    const WaitID   WAIT_ID         = ID_2;
    const unsigned WAIT_CYCLES     = 42;
    const unsigned INCREMENT_VALUE = 7;
    Wait*          waitCmd         = new Wait(WAIT_ID, WAIT_CYCLES, INCREMENT_VALUE);

    cmdq->PushBack(QueueCommandPtr {waitCmd});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += waitCmd->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the wait command
}

TEST_F(CommandQueueGaudi3Test, suspend)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    const WaitID   WAIT_ID         = ID_2;
    const unsigned WAIT_CYCLES     = 42;
    const unsigned INCREMENT_VALUE = 7;
    Suspend*       suspendCmd      = new Suspend(WAIT_ID, WAIT_CYCLES, INCREMENT_VALUE);

    cmdq->PushBack(QueueCommandPtr {suspendCmd});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += suspendCmd->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the wait command
}

TEST_F(CommandQueueGaudi3Test, nop)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    Nop* nopCmd = new Nop();

    cmdq->PushBack(QueueCommandPtr {nopCmd});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += nopCmd->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the wait command
}

TEST_F(CommandQueueGaudi3Test, fence)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi3::TpcQueue(0, 0, false);

    ASSERT_TRUE(cmdq->Empty());

    Fence* fenceCmd = new Fence(ID_0, 2);

    cmdq->PushBack(QueueCommandPtr {fenceCmd});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += fenceCmd->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the wait command
}
