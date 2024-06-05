#include <string>
#include <gtest/gtest.h>
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "graph_optimizer_test.h"
#include "graph_compiler/compilation_hal_reader.h"

using namespace gaudi;

static const unsigned REG_OFFSET = 8;

class CommandQueueGaudiTest : public GraphOptimizerTest
{
    virtual void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_F(CommandQueueGaudiTest, write_reg)
{
    unsigned      expectedNumElements = 0;
    unsigned      expectedBinSize     = 0;
    CommandQueue* cmdq                = new gaudi::TpcQueue(0, 0, false, false);

    ASSERT_TRUE(cmdq->Empty());

    WriteRegister *wr = new WriteRegister(REG_OFFSET, 0);
    cmdq->PushBack(QueueCommandPtr{wr});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += wr->GetBinarySize();
    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    ASSERT_EQ(wr->getRegOffset(), REG_OFFSET);
    delete cmdq;
    // Queue destructor will delete the wr command
}

static WriteManyRegisters *createWriteManyRegs(unsigned numRegs)
{
    unsigned *regVals = new unsigned[numRegs];

    for (unsigned i=0; i<numRegs; ++i)
    {
        regVals[i] = i;
    }

    WriteManyRegisters *wmr = new WriteManyRegisters(REG_OFFSET, numRegs, regVals);
    delete[] regVals;
    return wmr;
}

static void testWriteManyRegsBasic(unsigned numRegs)
{
    unsigned            expectedNumElements = 0;
    unsigned            expectedBinSize     = 0;
    WriteManyRegisters* wmr                 = createWriteManyRegs(numRegs);
    CommandQueue*       cmdq                = new gaudi::TpcQueue(0, 0, false, false);

    ASSERT_EQ(wmr->GetFirstReg(), REG_OFFSET);
    ASSERT_EQ(wmr->GetCount(), numRegs);

    cmdq->PushBack(QueueCommandPtr{wmr});
    expectedNumElements++;
    expectedBinSize += wmr->GetBinarySize();

    // Add yet another WriteRegister to the queue
    WriteRegister *wr = new WriteRegister(REG_OFFSET, 0);
    cmdq->PushBack(QueueCommandPtr{wr});
    expectedNumElements++;
    expectedBinSize += wr->GetBinarySize();

    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);
    delete cmdq;
    // Queue destructor will delete the commands
}

TEST_F(CommandQueueGaudiTest, write_many_regs_even_basic)
{
    testWriteManyRegsBasic(6);
}

TEST_F(CommandQueueGaudiTest, write_many_regs_odd_basic)
{
    testWriteManyRegsBasic(7);
}

TEST_F(CommandQueueGaudiTest, write_many_regs_only_one_basic)
{
    testWriteManyRegsBasic(1);
}

TEST_F(CommandQueueGaudiTest, load_predicates)
{
    unsigned                 expectedNumElements = 0;
    unsigned                 expectedBinSize     = 0;
    uint64_t                 srcAddr             = 0x1000;
    BasicFieldsContainerInfo bfci;
    CommandQueue*            cmdq = new gaudi::TpcQueue(0, 0, false, false);

    QueueCommand* loadPred = new LoadPredicates(srcAddr);
    cmdq->PushBack(QueueCommandPtr{loadPred});
    ASSERT_FALSE(cmdq->Empty());
    expectedNumElements++;
    expectedBinSize += loadPred->GetBinarySize();

    ASSERT_EQ(cmdq->Size(true), expectedNumElements);
    ASSERT_EQ(cmdq->GetBinarySize(true), expectedBinSize);

    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   srcAddr,
                                   (uint32_t)0, // offset of address field in the cmd payload
                                   FIELD_MEMORY_TYPE_DRAM);

    loadPred->SetContainerInfo(bfci);
    loadPred->prepareFieldInfos();

    const AddressFieldInfoSet& afis = loadPred->getBasicFieldsContainerInfo().retrieveAddressFieldInfoSet();
    ASSERT_EQ(afis.size(), 1); // only one patch-point is expected
    const AddressFieldInfoPair& afip = *(afis.begin());
    ASSERT_EQ(afip.first, 2); // the offset of the patched field should be 2 to account for the header
    ASSERT_EQ((afip.second)->getTargetAddress(), srcAddr);
    ASSERT_EQ((afip.second)->getFieldIndexOffset(), 2); // offset of the patched field again

    delete cmdq;
    // Queue destructor will delete the wr command
}


class DmaDescQueueForTest : public gaudi::DmaDescQueue
{
public:
    DmaDescQueueForTest() : gaudi::DmaDescQueue(0, 0, 0, false)
    {
        memset(&desc, 0, sizeof(desc));

        setAllPropertiesToData();

        // Fill up the shadow with history of all zeros
        getDescShadow().updateLoadedSegment(0, sizeof(desc)/sizeof(uint32_t), pDesc);
    }

    void setAllPropertiesToData()
    {
        // Fill up the shadow with Data property for the entire descriptor
        DescriptorShadow::AllRegistersProperties writeAll =
                std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(
                        sizeof(gaudi::DmaDesc), DescriptorShadow::RegisterProperties::createFromHandling(
                                DescriptorShadow::RegisterDataHandling::Data));
        getDescShadow().setAllRegProperties(writeAll);
    }

    void optimizePatchpoints(BasicFieldsContainerInfo& bfci)
    {
        gaudi::DmaDescQueue::optimizePatchpoints(nullptr, desc, &bfci);
    }

    DescriptorShadow& shadow() { return getDescShadow(); }

    gaudi::DmaDesc desc;
    uint32_t*      pDesc = (uint32_t*)&desc;

};

class MockDynamicShapeFieldInfo : public DynamicShapeFieldInfo
{
public:
    MockDynamicShapeFieldInfo(uint32_t fieldOffset, size_t size) :
        DynamicShapeFieldInfo(fieldOffset, FIELD_DYNAMIC_TEST, ShapeFuncID::SHAPE_FUNC_MAX_ID, nullptr, nullptr)
    {
        m_size = size;
    }

    void setSize(size_t size) { m_size = size; }

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<MockDynamicShapeFieldInfo>(*this); }
};

TEST_F(CommandQueueGaudiTest, verify_optimizePatchpoints)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;

    //-----------------------------------------------------------------------------------
    // Patchpoint #1, low+high, positions: 2, 4
    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   (uint64_t)(queue.pDesc+2),
                                   (uint64_t)(queue.pDesc+4),
                                   (uint64_t)(queue.pDesc),
                                   FIELD_MEMORY_TYPE_DRAM);

    // Mark positions 2 and 4 in the descriptor as 'ignore'. This should cause patchpoint #1 to be
    // deleted and the shadow will remain 'ignore' in positions 2 and 4
    queue.shadow().setPropertiesAt(2, DescriptorShadow::RegisterProperties::getIgnore());
    queue.shadow().setPropertiesAt(4, DescriptorShadow::RegisterProperties::getIgnore());
    //-----------------------------------------------------------------------------------
    // Patchpoint #2, low+high, positions: 6, 8
    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   (uint64_t)(queue.pDesc+6),
                                   (uint64_t)(queue.pDesc+8),
                                   (uint64_t)(queue.pDesc),
                                   FIELD_MEMORY_TYPE_DRAM);
    // Both the descriptor and the shadow were initialized with 0's. There will be no change in the history, but since
    // this is a transition from none-patching to patching, patchpoint #2 should not be deleted and the shadow will be
    // 'patching' in positions 6 and 8
    //-----------------------------------------------------------------------------------
    // patchpoint #3, full, positions: 10, 11
    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   0x1234,        // address
                                   (uint32_t)10,  // offset of address field in the descriptor
                                   FIELD_MEMORY_TYPE_DRAM);

    // Set the values in the descriptor at positions 11 to something else that 0. This will cause
    // a change in the history b/w the shadow and the descriptor and thus patchpoint #3 shall remain
    // intact and the shadow shall be marked with 'patching' in positions 10 and 11.
    queue.pDesc[11] = 1;
    //-----------------------------------------------------------------------------------
    // patchpoint #4, low+high, positions: 13, 15
    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   (uint64_t)(queue.pDesc+13),
                                   (uint64_t)(queue.pDesc+15),
                                   (uint64_t)(queue.pDesc),
                                   FIELD_MEMORY_TYPE_DRAM);
    // Set the values in the shadow at positions 13 to something else that 0. This will cause
    // a change in the history b/w the shadow and the descriptor and thus patchpoint #4 shall remain
    // intact and the shadow shall be marked with 'patching' in positions 13 and 15.
    uint32_t nonzero = 1;
    queue.shadow().updateLoadedSegment(13, 14, &nonzero);
    //-----------------------------------------------------------------------------------

    // So we set up all the framing for the test, let the job be done
    queue.optimizePatchpoints(bfci);

    // Verify all the expected results stated above
    const AddressFieldInfoSet& afis = bfci.retrieveAddressFieldInfoSet();
    ASSERT_EQ(afis.size(), 5); // patchpoint #3 (full) and patchpoint #2 and #4 which are actually low PP and high PP

    for (auto afip : afis) // afip = address-field-info-pair :-)
    {
        // don't assume the order of elements in the set
        ASSERT_TRUE(afip.first == 6 || afip.first == 8 || afip.first == 10 || afip.first == 13 || afip.first == 15);
    }

    ASSERT_EQ(queue.shadow().propertiesAt(2), DescriptorShadow::RegisterProperties::getIgnore());
    ASSERT_EQ(queue.shadow().propertiesAt(4), DescriptorShadow::RegisterProperties::getIgnore());
    ASSERT_EQ(queue.shadow().propertiesAt(6), DescriptorShadow::RegisterProperties::getPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(8), DescriptorShadow::RegisterProperties::getPatching());

    ASSERT_EQ(queue.shadow().propertiesAt(10), DescriptorShadow::RegisterProperties::getPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(11), DescriptorShadow::RegisterProperties::getPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(13), DescriptorShadow::RegisterProperties::getPatching());
    ASSERT_EQ(queue.shadow().propertiesAt(15), DescriptorShadow::RegisterProperties::getPatching());
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_dynamic_then_static)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset = 10;
    size_t                    ppLen = 4;
    MockDynamicShapeFieldInfo patchPoint(ppOffset, ppLen);

    bfci.add(std::make_pair(ppOffset, patchPoint.clone()));

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);
    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }

    // Now a static descriptor shouldn't optimize the registers in [ppOffset : ppOffset + ppLen]
    BasicFieldsContainerInfo  bfci2;
    queue.setAllPropertiesToData();
    queue.optimizePatchpoints(bfci2);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci2);
    ASSERT_EQ(bfci2.retrieveBasicFieldInfoSet().size(), 0);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getGeneralData());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), true);
    }
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_static_then_dynamic)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset = 10;
    uint32_t                  data[]   = {1, 2, 3, 4};
    size_t                    ppLen    = sizeof(data) / sizeof(uint32_t);

    memcpy(&queue.pDesc[ppOffset], data, ppLen * sizeof(uint32_t));

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);
    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 0);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getGeneralData());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), true);
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).value(), data[i]);
    }

    BasicFieldsContainerInfo  bfci2;
    MockDynamicShapeFieldInfo patchPoint(ppOffset, ppLen);
    bfci2.add(std::make_pair(ppOffset, patchPoint.clone()));

    queue.optimizePatchpoints(bfci2);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci2);

    // Although the values are the same and there is a dynamic pp - it should be written and the valid map should be false
    // because we can't relay on the data there.
    ASSERT_EQ(bfci2.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_regular_and_dynamic_pp)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset = 10;
    uint32_t                  data[]   = {1, 2};
    size_t                    ppLen    = sizeof(data) / sizeof(uint32_t);
    memcpy(&queue.pDesc[ppOffset], data, ppLen * sizeof(uint32_t));
    queue.shadow().updateLoadedSegment(ppOffset, ppOffset + ppLen, data);

    MockDynamicShapeFieldInfo dynamicPatchPoint(ppOffset, ppLen);
    bfci.add(std::make_pair(ppOffset, dynamicPatchPoint.clone()));

    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   (uint64_t)data,
                                   ppOffset,
                                   FIELD_MEMORY_TYPE_DRAM);

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);

    ASSERT_EQ(bfci.retrieveAddressFieldInfoSet().size(), 1);
    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_regular_and_dynamic_pp_partial_overlap)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset        = 10;
    uint32_t                  data[]          = {1, 2, 3, 4, 5, 6};
    size_t                    dataElements    = 6;
    uint64_t*                 targetAddresses = (uint64_t*)(&data[2]);
    size_t                    ppLen           = 4;

    memcpy(&queue.pDesc[ppOffset], data, sizeof(data));
    queue.shadow().updateLoadedSegment(ppOffset, ppOffset + dataElements, data);

    MockDynamicShapeFieldInfo dynamicPatchPoint(ppOffset, ppLen);
    bfci.add(std::make_pair(ppOffset, dynamicPatchPoint.clone()));

    for (int i = 0; i < ppLen; i++)
    {
        queue.shadow().setPropertiesAt(ppOffset + 2 + i,  DescriptorShadow::RegisterProperties::getPatching());
    }

    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   targetAddresses[0],
                                   ppOffset + 2,
                                   FIELD_MEMORY_TYPE_DRAM);

    bfci.addAddressEngineFieldInfo(nullptr,
                                   getMemorySectionNameForMemoryID(0),
                                   0,
                                   targetAddresses[1],
                                   ppOffset + 4,
                                   FIELD_MEMORY_TYPE_DRAM);

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);

    ASSERT_EQ(bfci.retrieveAddressFieldInfoSet().size(), 1);
    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }

    for (int i = ppLen; i < ppLen + 2; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), true);
    }
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_overlaping_dynamic_pps)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset = 10;
    uint32_t                  data[]   = {1, 2, 3, 4};
    size_t                    ppLen    = sizeof(data) / sizeof(uint32_t);

    memcpy(&queue.pDesc[ppOffset], data, ppLen * sizeof(uint32_t));
    queue.shadow().updateLoadedSegment(ppOffset, ppOffset + ppLen, data);

    MockDynamicShapeFieldInfo dynamicPatchPoint(ppOffset, ppLen);
    bfci.add(std::make_pair(ppOffset, dynamicPatchPoint.clone()));

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);

    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }

    BasicFieldsContainerInfo  bfci2;
    MockDynamicShapeFieldInfo dynamicPatchPoint2(ppOffset, ppLen);
    bfci2.add(std::make_pair(ppOffset, dynamicPatchPoint2.clone()));

    queue.setAllPropertiesToData();
    queue.optimizePatchpoints(bfci2);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci2);

    ASSERT_EQ(bfci2.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }
}

TEST_F(CommandQueueGaudiTest, dynamic_shape_shadow_disabled_partial_overlaping_dynamic_pps)
{
    BasicFieldsContainerInfo  bfci;
    DmaDescQueueForTest       queue;
    uint32_t                  ppOffset  = 10;
    uint32_t                  ppOffset2 = 12;
    uint32_t                  data[]    = {1, 2, 3, 4, 5, 6};
    size_t                    elements  = sizeof(data) / sizeof(uint32_t);
    size_t                    ppLen     = 4;

    memcpy(&queue.pDesc[ppOffset], data, sizeof(data));
    queue.shadow().updateLoadedSegment(ppOffset, ppOffset + elements, data);

    MockDynamicShapeFieldInfo dynamicPatchPoint(ppOffset, ppLen);
    bfci.add(std::make_pair(ppOffset, dynamicPatchPoint.clone()));

    queue.optimizePatchpoints(bfci);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci);

    ASSERT_EQ(bfci.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }
    for (int i = ppLen; i < elements; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getGeneralData());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), true);
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).value(), data[i]);
    }

    BasicFieldsContainerInfo  bfci2;
    MockDynamicShapeFieldInfo dynamicPatchPoint2(ppOffset2, ppLen);
    bfci2.add(std::make_pair(ppOffset2, dynamicPatchPoint2.clone()));

    queue.setAllPropertiesToData();
    queue.optimizePatchpoints(bfci2);
    queue.addLoadDesc(nullptr, DescSection(queue.desc), &bfci2);

    ASSERT_EQ(bfci2.retrieveBasicFieldInfoSet().size(), 1);
    for(int i = 0; i < elements - ppLen; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getGeneralData());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), true);
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).value(), data[i]);
    }
    for (int i = elements - ppLen; i < elements; i++)
    {
        ASSERT_EQ(queue.shadow().propertiesAt(ppOffset + i), DescriptorShadow::RegisterProperties::getDynamicPatching());
        ASSERT_EQ(queue.shadow().getDataAt(ppOffset + i).is_set(), false);
    }
}
