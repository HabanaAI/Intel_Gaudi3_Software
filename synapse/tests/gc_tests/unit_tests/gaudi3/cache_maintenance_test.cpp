#include <bitset>
#include "gtest/gtest.h"
#include "graph_optimizer_test.h"
#include "recipe_test_base.h"
#include "scoped_configuration_change.h"
#include "tensor.h"
#include "node.h"
#include "node_roi.h"
#include "node_factory.h"
#include "recipe.h"
#include "recipe_allocator.h"
#include "recipe_blob.h"
#include "recipe_program.h"
#include "recipe_ecb.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "platform/gaudi3/graph_compiler/gaudi3_eng_arc_hooks.h"
#include "platform/gaudi3/graph_compiler/gaudi3_eng_arc_command.h"

using namespace gaudi3;

static constexpr unsigned converterSafetyFactor = 2;

static bool getValOfReg(const QueueCommand* cmd, unsigned regAddr, unsigned& regVal)
{
    const WriteRegister*      wreg32 = nullptr;
    const WriteManyRegisters* wbulk = nullptr;
    bool                      writtenByBulk = false;
    bool                      writtenByWreg32 = false;
    unsigned                  bulkIdx = 0;
    if ((wbulk = dynamic_cast<const WriteManyRegisters*>(cmd)) != nullptr)
    {
        unsigned startBulk = wbulk->GetFirstReg();
        unsigned endBulk   = startBulk + wbulk->GetCount() * sizeof(uint32_t);
        if (startBulk <= regAddr && regAddr < endBulk)
        {
            writtenByBulk = true;
            bulkIdx = (regAddr - startBulk) / sizeof(uint32_t);
        }
    }
    else if ((wreg32 = dynamic_cast<const WriteRegister*>(cmd)) != nullptr && (wreg32->getRegOffset() == regAddr))
    {
        writtenByWreg32 = true;
    }
    if (writtenByWreg32) regVal = wreg32->getValue();
    if (writtenByBulk) regVal = wbulk->getValue(bulkIdx);
    return (writtenByWreg32 || writtenByBulk);
}

class CacheMaintenanceTest : public GraphOptimizerTest
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY, "false");
    }
};

class Gaudi3GraphCacheMaintenanceTest : public Gaudi3Graph
{
public:
    CommandQueuePtr getTpcQueue() { return m_codeGenerator->getTpcDispatcher()->getQueue(0); }
    const std::vector<CommandQueuePtr>& getMmeQueues() { return m_codeGenerator->getMmeDispatcher()->getQueues(); }
};

TEST_F(CacheMaintenanceTest, basic_tpc_flow)
{
    Gaudi3GraphCacheMaintenanceTest g;

    const unsigned tensor_dim = 1;
    const TSize    size       = 1024 * 32 * 10;

    TensorPtr cpyI = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr cpyO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr rluO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    synMemoryDescriptor memDesc(true);
    cpyI->setMemoryDescriptor(memDesc);
    rluO->setMemoryDescriptor(memDesc);
    cpyI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    rluO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    // The relu node depends on the memcpy node and thus will be second in the post-compilation graph
    NodePtr nTpcMemcpy = NodeFactory::createNode({cpyI}, {cpyO}, nullptr, "memcpy", "node_name_memcpy");
    NodePtr nTpcRelu = NodeFactory::createNode({cpyO}, {rluO}, nullptr, "relu_fwd_f32", "node_name_relu");

    GraphEditor::addNode(g, nTpcMemcpy);
    GraphEditor::addNode(g, nTpcRelu);

    NodeAnnotation& ann = nTpcMemcpy->getNodeAnnotation();
    CacheMetaData md;
    md.mcid           = 7;  // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction = DEGRADE;
    md.cacheClass = Top;
    md.cacheDirective = HomeAllocate;
    ann.inputsCacheMetaData.push_back(md);
    md.mcid           = 8;  // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction = DISCARD;
    md.cacheClass = High;
    md.cacheDirective = DcoreAllocate;
    ann.outputsCacheMetaData.push_back(md);
    ASSERT_EQ(g.compile(), true);

    //---- Verify QMAN commands ----
    const CommandQueuePtr& queue = g.getTpcQueue();
    bool foundWreg64ForMcid7 = false;
    bool foundWreg64ForMcid8 = false;
    bool foundWriteAllocAndClassForTensor1 = false;
    for (const auto& cmd : queue->getCommands(false))
    {
        //---- Verify wreg64 for mcid configuration ----
        WriteReg64* wreg64 = nullptr;
        if ((wreg64 = dynamic_cast<WriteReg64*>(cmd.get())) != nullptr)
        {
            if (wreg64->getDregOffset() * sizeof(uint32_t) == 0xa1cc) // 0xa1cc is tensor 0 HBW_AXI_CFG
            {
                uint16_t mcid    = 0xFFFF & wreg64->getValue();
                unsigned arcache = (0x000F0000 & wreg64->getValue()) >> 16;
                unsigned awcache = (0x00F00000 & wreg64->getValue()) >> 20;
                unsigned clas    = (0x03000000 & wreg64->getValue()) >> 24;
                ASSERT_EQ(mcid, 1);
                ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                ASSERT_EQ(arcache, 0x7); // home alloc
                ASSERT_EQ(awcache, 0x7); // home alloc
                ASSERT_EQ(clas, 0x3);    // Top class
                foundWreg64ForMcid7 = true;
            }
            if (wreg64->getDregOffset() * sizeof(uint32_t) == 0xa794) // 0xa794 is MCID_FAST_CONFIG
            {
                // The frequency of the two mcids is identical and the sorting is done from high to low; thus, 8 is
                // expected to be configured via the FAST_CONFIG
                uint16_t mcid = 0xFFFF & wreg64->getValue();
                ASSERT_EQ(mcid, 1);
                uint16_t mask = wreg64->getValue() >> 16;
                ASSERT_EQ(mask, 2); // the second tensor is configured by FAST_CONFIG
                ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                foundWreg64ForMcid8 = true;
            }
        }

        //---- Verify allocation policy & class for tensor 1 via the private tensor HBW_AXI_CFG ----
        if (!foundWriteAllocAndClassForTensor1) // verify first node only, skip over the 2nd node
        {
            unsigned axiCfgAddr = 0xa228; // 0xa228 is tensor 1 HBW_AXI_CFG
            unsigned regVal = 0;
            if (getValOfReg(cmd.get(), axiCfgAddr, regVal))
            {
                foundWriteAllocAndClassForTensor1 = true;
                unsigned mcid    = 0x0000FFFF & regVal;
                unsigned arcache = (0x000F0000 & regVal) >> 16;
                unsigned awcache = (0x00F00000 & regVal) >> 20;
                unsigned clas    = (0x03000000 & regVal) >> 24;
                ASSERT_EQ(mcid, 0);      // mcid is 0 since we programed it via the fast config
                ASSERT_EQ(arcache, 0xb); // dcore alloc
                ASSERT_EQ(awcache, 0xb); // dcore alloc
                ASSERT_EQ(clas, 0x2);    // High class
            }
        }
    }
    ASSERT_TRUE(foundWreg64ForMcid7);
    ASSERT_TRUE(foundWreg64ForMcid8);
    ASSERT_TRUE(foundWriteAllocAndClassForTensor1);

    //---- Verify CME commands ----
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ASSERT_EQ(recipe->max_used_mcid_degrade, 1 + converterSafetyFactor);
    ASSERT_EQ(recipe->max_used_mcid_discard, 1 + converterSafetyFactor);

    ecb_t pDynamicEcb = {0};
    ecb_t pStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pStaticEcb  = recipe->arc_jobs[i].static_ecb;
            break;
        }
    }

    NodePtr firstNode = g.getExeSortedNodes().front();

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDynamicEcb.cmds != nullptr);
    ASSERT_TRUE(pStaticEcb.cmds_size == 0);
    ASSERT_TRUE(pStaticEcb.cmds_eng_offset == 0);
    ASSERT_TRUE(pStaticEcb.cmds == nullptr);

    // DEGRADE
    struct cme_arc_cmd_degrade_cls_t* cmeDegradeCmd = (cme_arc_cmd_degrade_cls_t*)(pDynamicEcb.cmds);
    ASSERT_EQ(cmeDegradeCmd->cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
    ASSERT_EQ(cmeDegradeCmd->mcid_offset, 1);
    ASSERT_EQ(cmeDegradeCmd->use_discard_base, 0);
    // operation shall be executed AFTER the first node
    ASSERT_EQ(cmeDegradeCmd->target_bitmap, 1 << VIRTUAL_SOB_INDEX_TPC);
    ASSERT_EQ(cmeDegradeCmd->tpc.threshold_v2, firstNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value());

    // DISCARD
    struct cme_arc_cmd_discard_cls_t* cmeDiscardCmd = (cme_arc_cmd_discard_cls_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_degrade_cls_t));
    ASSERT_EQ(cmeDiscardCmd->cmd_type, CME_ECBL_CMD_DISCARD_CLS);
    ASSERT_EQ(cmeDiscardCmd->mcid_offset, 1);
    // operation shall be executed AFTER the first node
    ASSERT_EQ(cmeDiscardCmd->target_bitmap, 1 << VIRTUAL_SOB_INDEX_TPC);
    ASSERT_EQ(cmeDiscardCmd->tpc.threshold_v2, firstNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value());
}

TEST_F(CacheMaintenanceTest, mme_mcid_patching)
{
    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME (batch gemm) node
    uint8_t             dim            = 4;
    auto                dataType       = syn_type_single;
    const TSize         sizesIn0[]     = {64, 16, 2, 3};
    const TSize         sizesIn1[]     = {64, 64, 2, 3};
    TSize               bgemmOutSize[] = {64, 16, 2, 3};
    auto                bgemmGuid      = NodeFactory::batchGemmNodeTypeName;
    TensorPtr           bgemmInput0    = TensorPtr(new Tensor(dim, sizesIn0, dataType));
    TensorPtr           bgemmInput1    = TensorPtr(new Tensor(dim, sizesIn1, dataType));
    TensorPtr           bgemmOut       = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

    bgemmInput0->setDramOffset(0x10000);
    bgemmInput1->setDramOffset(0x20000);
    bgemmInput0->setMemoryDescriptor(memDesc);
    bgemmInput1->setMemoryDescriptor(memDesc);
    bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    synGEMMParams params {};
    NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "bgemm_node");
    GraphEditor::addNode(g, bgemm);

    // Set cache metadata for MME node:
    //   first input:   mcid 11 with discard
    //   second input:  mcid 12 with degrade
    //   output:        mcid 13 with discard
    NodeAnnotation& ann1 = bgemm->getNodeAnnotation();
    ann1.splitToLogicalROIs = false;
    md.mcid                 = 11;  // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction = DISCARD;
    md.cacheClass = Top;
    md.cacheDirective = SharedAllocate;
    ann1.inputsCacheMetaData.push_back(md); // first input
    md.mcid           = 12;                 // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction = DEGRADE;
    md.cacheClass = High;
    md.cacheDirective = HomeAllocate;
    ann1.inputsCacheMetaData.push_back(md); // second input
    md.mcid           = 13;  // will be converted to mcid: 1 by cmeTasks pass (13 and 11 has same dependency map)
    md.cmAction = DISCARD;
    md.cacheClass = Top;
    md.cacheDirective = SharedAllocate;
    ann1.outputsCacheMetaData.push_back(md); // output

    // Compile
    ASSERT_EQ(g.compile(), true);

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        bool foundWreg64ForMcid11 = false;
        bool foundWreg64ForMcid12 = false;
        bool foundWreg64ForMcid13 = false;
        bool foundAllocationPolicy = false;
        for (const auto& cmd : q->getCommands(false))
        {
            // Verify the mcids and class
            WriteReg64* wreg64 = nullptr;
            if ((wreg64 = dynamic_cast<WriteReg64*>(cmd.get())) != nullptr)
            {
                if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x250) // 0x250 is A operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x3); // Top class
                    foundWreg64ForMcid11 = true;
                }
                else if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x254) // 0x254 is B operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x2); // High class
                    foundWreg64ForMcid12 = true;
                }
                else if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x258) // 0x258 is C operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x3); // Top class
                    foundWreg64ForMcid13 = true;
                }
            }

            // Verify the allocation policy
            unsigned axiCacheDataAddr = 0x25c;
            unsigned regVal = 0;
            if (getValOfReg(cmd.get(), axiCacheDataAddr, regVal))
            {
                unsigned allocA = 0x0000000F & regVal;
                unsigned allocB = (0x000000F0 & regVal) >> 4;
                unsigned allocC = (0x00000F00 & regVal) >> 8;
                ASSERT_EQ(allocA, 0xf); // shared alloc
                ASSERT_EQ(allocB, 0x7); // home alloc
                ASSERT_EQ(allocC, 0xf); // shared alloc
                foundAllocationPolicy = true;
            }
        }
        ASSERT_TRUE(foundWreg64ForMcid11);
        ASSERT_TRUE(foundWreg64ForMcid12);
        ASSERT_TRUE(foundWreg64ForMcid13);
        ASSERT_TRUE(foundAllocationPolicy);
    }
}

TEST_F(CacheMaintenanceTest, mme_mcid_patching_dedw)
{
    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME (batch gemm) node
    uint8_t     dim         = 2;
    auto        dataType    = syn_type_single;
    const TSize sizesIn0[]  = {16, 16, 1, 1};
    const TSize sizesIn1[]  = {16, 16, 1, 1};
    TSize       sizesOut[]  = {16, 16, 1, 1};
    auto        guid        = NodeFactory::batchGemmDeDwNodeTypeName;
    TensorPtr   input0      = TensorPtr(new Tensor(dim, sizesIn0, dataType));
    TensorPtr   input1      = TensorPtr(new Tensor(dim, sizesIn1, dataType));
    TensorPtr   out         = TensorPtr(new Tensor(dim, sizesOut, dataType));

    input0->setDramOffset(0x10000);
    input1->setDramOffset(0x20000);
    out->setDramOffset(0x30000);
    input0->setMemoryDescriptor(memDesc);
    input1->setMemoryDescriptor(memDesc);
    out->setMemoryDescriptor(memDesc);
    input0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    input1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);

    synGEMMParams params {};
    NodePtr dedw = NodeFactory::createNode({input0, input1}, {out}, &params, guid, "dedw");
    GraphEditor::addNode(g, dedw);

    // Set cache metadata for MME node:
    //   first input:   mcid 11 with discard
    //   second input:  mcid 12 with degrade
    //   output:        mcid 13 with discard
    NodeAnnotation& ann1 = dedw->getNodeAnnotation();
    ann1.splitToLogicalROIs = false;
    md.mcid                 = 11;  // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction             = DISCARD;
    md.cacheClass           = Top;
    md.cacheDirective       = SharedAllocate;
    ann1.inputsCacheMetaData.push_back(md); // input 0
    md.mcid           = 12;                 // will be converted to mcid: 1 by cmeTasks pass
    md.cmAction       = DEGRADE;
    md.cacheClass     = High;
    md.cacheDirective = HomeAllocate;
    ann1.inputsCacheMetaData.push_back(md); // input 1
    md.mcid           = 13;  // will be converted to mcid: 1 by cmeTasks pass (13 and 11 has same dependency map)
    md.cmAction       = DISCARD;
    md.cacheClass     = Top;
    md.cacheDirective = SharedAllocate;
    ann1.outputsCacheMetaData.push_back(md); // output

    // Compile
    ASSERT_EQ(g.compile(), true);

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        bool foundWreg64ForMcid11 = false;
        bool foundWreg64ForMcid12 = false;
        bool foundWreg64ForMcid13 = false;
        bool foundAllocationPolicy = false;
        for (const auto& cmd : q->getCommands(false))
        {
            // Verify the mcids and class
            WriteReg64* wreg64 = nullptr;
            if ((wreg64 = dynamic_cast<WriteReg64*>(cmd.get())) != nullptr)
            {
                // !!! Note: In dedw A and B are swapped !!!

                if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x250) // 0x250 is A operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x2); // High class
                    foundWreg64ForMcid12 = true;
                }
                else if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x254) // 0x254 is B operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x3); // Top class
                    foundWreg64ForMcid11 = true;
                }
                else if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x258) // 0x258 is C operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x3); // Top class
                    foundWreg64ForMcid13 = true;
                }
            }

            // Verify the allocation policy
            unsigned axiCacheDataAddr = 0x25c;
            unsigned regVal = 0;
            if (getValOfReg(cmd.get(), axiCacheDataAddr, regVal))
            {
                unsigned allocA = 0x0000000F & regVal;
                unsigned allocB = (0x000000F0 & regVal) >> 4;
                unsigned allocC = (0x00000F00 & regVal) >> 8;
                // In dedw A and B are swapped
                ASSERT_EQ(allocA, 0x7); // home alloc
                ASSERT_EQ(allocB, 0xf); // shared alloc
                ASSERT_EQ(allocC, 0xf); // shared alloc
                foundAllocationPolicy = true;
            }
        }
        ASSERT_TRUE(foundWreg64ForMcid11);
        ASSERT_TRUE(foundWreg64ForMcid12);
        ASSERT_TRUE(foundWreg64ForMcid13);
        ASSERT_TRUE(foundAllocationPolicy);
    }
}

TEST_F(CacheMaintenanceTest, tpc_and_mme_flow)
{
    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME (batch gemm) node
    uint8_t             dim            = 4;
    auto                dataType       = syn_type_single;
    const TSize         sizesIn0[]     = {64, 16, 2, 3};
    const TSize         sizesIn1[]     = {64, 64, 2, 3};
    TSize               bgemmOutSize[] = {64, 16, 2, 3};
    auto                bgemmGuid      = NodeFactory::batchGemmNodeTypeName;
    TensorPtr           bgemmInput0    = TensorPtr(new Tensor(dim, sizesIn0, dataType));
    TensorPtr           bgemmInput1    = TensorPtr(new Tensor(dim, sizesIn1, dataType));
    TensorPtr           bgemmOut       = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

    bgemmInput0->setDramOffset(0x10000);
    bgemmInput1->setDramOffset(0x20000);
    bgemmInput0->setMemoryDescriptor(memDesc);
    bgemmInput1->setMemoryDescriptor(memDesc);
    bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    synGEMMParams params {};
    NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "bgemm_node");
    GraphEditor::addNode(g, bgemm);

    // Add TPC (nop) node
    TSize     size1  = 1;
    TensorPtr tpcO   = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy0 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy1 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy2 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    tpcO->setMemoryDescriptor(memDesc);
    dummy0->setMemoryDescriptor(memDesc);
    dummy1->setMemoryDescriptor(memDesc);
    dummy2->setMemoryDescriptor(memDesc);
    tpcO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    dummy0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    dummy1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    dummy2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    NodePtr tpcNop = NodeFactory::createNode({bgemmOut, dummy0, dummy1, dummy2}, {tpcO}, nullptr, NOP_KERNEL_NAME, "tpc_node");
    GraphEditor::addNode(g, tpcNop);

    // Set cache metadata for MME node:
    //   first input:   mcid 0 with NOP
    //   second input:  mcid 10 with degrade
    //   output:        mcid 0 with NOP
    NodeAnnotation& ann1 = bgemm->getNodeAnnotation();
    ann1.splitToLogicalROIs = false;
    md.mcid = 0;
    md.cmAction = NOP;
    md.cacheClass = High;
    md.cacheDirective = HomeAllocate;
    ann1.inputsCacheMetaData.push_back(md); // first input
    md.mcid           = 15;
    md.cmAction = DEGRADE;
    md.cacheClass = Top;
    md.cacheDirective = SharedAllocate;
    ann1.inputsCacheMetaData.push_back(md); // second input
    md.mcid = 0;
    md.cmAction = NOP;
    md.cacheClass = High;
    md.cacheDirective = HomeAllocate;
    ann1.outputsCacheMetaData.push_back(md); // output

    // Set cache metadata for TPC node:
    //   first input:   mcid 10 with degrade
    //   second input:  mcid 0 with NOP
    //   third input:   mcid 11 with discard
    //   forth inputs:  mcid 11 with discard
    //   output:        mcid 12 with degrade
    NodeAnnotation& ann2 = tpcNop->getNodeAnnotation();
    ann2.splitToLogicalROIs = false;
    md.mcid                 = 10;  // will be converted to mcid: 2 by cmeTasks pass
    md.cmAction = DEGRADE;
    md.cacheClass = Top;
    md.cacheDirective = SharedAllocate;
    ann2.inputsCacheMetaData.push_back(md); // first input, tensor 0
    md.mcid = 0;
    md.cmAction = NOP;
    md.cacheClass = Normal;
    md.cacheDirective = NoAllocate;
    ann2.inputsCacheMetaData.push_back(md); // second input, tensor 1
    md.mcid           = 11;                 // will be converted to mcid: 1 by cmeTasks pass
    md.cacheClass = High;
    md.cmAction = DISCARD;
    md.cacheDirective = DcoreAllocate;
    ann2.inputsCacheMetaData.push_back(md); // third input, tensor 2
    ann2.inputsCacheMetaData.push_back(md); // forth input, tensor 3
    md.mcid           = 12;                 // will be converted to mcid: 2 by cmeTasks pass
    md.cmAction = DEGRADE;
    md.cacheClass = Top;
    md.cacheDirective = SharedAllocate;
    ann2.outputsCacheMetaData.push_back(md); // output, tensor 4

    // Compile
    ASSERT_EQ(g.compile(), true);

    //---- Verify CME commands ----
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ecb_t pDynamicEcb = {0};
    ecb_t pStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pStaticEcb  = recipe->arc_jobs[i].static_ecb;
            break;
        }
    }

    ASSERT_EQ(g.getExeSortedNodes().size(), 2);
    NodePtr mmeNode = g.getExeSortedNodes()[0];
    NodePtr tpcNode = g.getExeSortedNodes()[1];

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDynamicEcb.cmds != nullptr);
    ASSERT_TRUE(pStaticEcb.cmds_size == 0);
    ASSERT_TRUE(pStaticEcb.cmds_eng_offset == 0);
    ASSERT_TRUE(pStaticEcb.cmds == nullptr);

    // Degrade mcid 15 - will be converted to mcid: 1 by cmeTasks pass
    struct cme_arc_cmd_degrade_cls_t* cmeDegradeCmd = (cme_arc_cmd_degrade_cls_t*)(pDynamicEcb.cmds);
    ASSERT_EQ(cmeDegradeCmd->cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
    ASSERT_EQ(cmeDegradeCmd->mcid_offset, 1);
    ASSERT_EQ(cmeDegradeCmd->use_discard_base, 0);
    // operation shall be executed AFTER MME node
    ASSERT_TRUE(cmeDegradeCmd->target_bitmap & (1 << VIRTUAL_SOB_INDEX_MME));
    const NodeROI& roi = mmeNode->getLogicalRois()->front();
    unsigned inputTensor1SobVal = roi.inputRois[1].m_overlapRoi.subRois->back().relSoIdx + 1;
    ASSERT_EQ(cmeDegradeCmd->mme.threshold_v2, inputTensor1SobVal);

    // Degrade mcid 10 - will be converted to mcid: 2 by cmeTasks pass
    cmeDegradeCmd = (cme_arc_cmd_degrade_cls_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_degrade_cls_t));
    ASSERT_EQ(cmeDegradeCmd->cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
    ASSERT_EQ(cmeDegradeCmd->mcid_offset, 2);
    // operation shall be executed AFTER the TPC node
    ASSERT_EQ(cmeDegradeCmd->target_bitmap, 1 << VIRTUAL_SOB_INDEX_TPC);
    ASSERT_EQ(cmeDegradeCmd->tpc.threshold_v2, tpcNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value());

    // Discard mcid 11 - will be converted to mcid: 1 by cmeTasks pass
    struct cme_arc_cmd_discard_cls_t* cmeDiscardCmd =
        (cme_arc_cmd_discard_cls_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_degrade_cls_t) * 2);
    ASSERT_EQ(cmeDiscardCmd->cmd_type, CME_ECBL_CMD_DISCARD_CLS);
    ASSERT_EQ(cmeDiscardCmd->mcid_offset, 1);
    // operation shall be executed AFTER the TPC node
    ASSERT_EQ(cmeDiscardCmd->target_bitmap, 1 << VIRTUAL_SOB_INDEX_TPC);
    ASSERT_EQ(cmeDiscardCmd->tpc.threshold_v2, tpcNode->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value());

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        bool foundWreg64ForMcid15  = false;
        bool foundAllocationPolicy = false;
        for (const auto& cmd : q->getCommands(false))
        {
            // Verify the mcid and class
            WriteReg64* wreg64 = nullptr;
            if ((wreg64 = dynamic_cast<WriteReg64*>(cmd.get())) != nullptr)
            {
                if (wreg64->getDregOffset() * sizeof(uint32_t) == 0x254) // 0x254 is B operand AXI
                {
                    unsigned mcid = (0x3FFFC000 & wreg64->getValue()) >> 14;
                    unsigned clas = (0xC0000000 & wreg64->getValue()) >> 30;
                    ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                    ASSERT_EQ(mcid, 1);
                    ASSERT_EQ(clas, 0x3); // Top class
                    foundWreg64ForMcid15 = true;
                }
            }

            // Verify the allocation policy
            unsigned axiCacheDataAddr = 0x25c;
            unsigned regVal = 0;
            if (getValOfReg(cmd.get(), axiCacheDataAddr, regVal))
            {
                unsigned allocA = 0x0000000F & regVal;
                unsigned allocB = (0x000000F0 & regVal) >> 4;
                unsigned allocC = (0x00000F00 & regVal) >> 8;
                ASSERT_EQ(allocA, 0x7); // home alloc
                ASSERT_EQ(allocB, 0xf); // shared alloc
                ASSERT_EQ(allocC, 0x7); // home alloc
                foundAllocationPolicy = true;
            }
        }
        ASSERT_TRUE(foundWreg64ForMcid15);
        ASSERT_TRUE(foundAllocationPolicy);
    }

    //---- Verify TPC QMAN commands ----
    const CommandQueuePtr& queue = g.getTpcQueue();
    bool foundWreg64ForMcid10 = false;
    bool foundWreg64ForMcid11 = false;
    bool foundWreg64ForMcid12 = false;
    bool foundWriteAllocAndClassForTensor1 = false;
    bool foundWriteAllocAndClassForTensor2 = false;
    bool foundWriteAllocAndClassForTensor3 = false;
    unsigned axiCfgAddr = 0;
    unsigned regVal = 0;
    for (const auto& cmd : queue->getCommands(false))
    {
        // Verify wreg64 for mcid configuration
        WriteReg64* wreg64 = nullptr;
        if ((wreg64 = dynamic_cast<WriteReg64*>(cmd.get())) != nullptr)
        {
            if (wreg64->getDregOffset() * sizeof(uint32_t) == 0xa794) // 0xa794 is MCID_FAST_CONFIG
            {
                // The highest frequency mcid is 11 for tensors 2 and 3 (third and forth)
                uint16_t mcid = 0xFFFF & wreg64->getValue();
                ASSERT_EQ(mcid, 1);
                uint16_t mask = wreg64->getValue() >> 16;
                ASSERT_EQ(mask, 0xc); // tensors 2 and 3 (third and forth) are configured by FAST_CONFIG
                ASSERT_EQ(wreg64->getBaseIndex(), 30); // 30 is the base of discard
                foundWreg64ForMcid11 = true;
            }
            if (wreg64->getDregOffset() * sizeof(uint32_t) == 0xa1cc) // 0xa1cc is tensor 0 HBW_AXI_CFG
            {
                uint16_t mcid    = 0xFFFF & wreg64->getValue();
                unsigned arcache = (0x000F0000 & wreg64->getValue()) >> 16;
                unsigned awcache = (0x00F00000 & wreg64->getValue()) >> 20;
                unsigned clas    = (0x03000000 & wreg64->getValue()) >> 24;
                ASSERT_EQ(mcid, 2);
                ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                ASSERT_EQ(arcache, 0xf);               // shared alloc
                ASSERT_EQ(awcache, 0xf);               // shared alloc
                ASSERT_EQ(clas, 0x3);                  // Top class
                foundWreg64ForMcid10 = true;
            }
            if (wreg64->getDregOffset() * sizeof(uint32_t) == 0xa33c) // 0xa33c is tensor 4 HBW_AXI_CFG
            {
                uint16_t mcid    = 0xFFFF & wreg64->getValue();
                unsigned arcache = (0x000F0000 & wreg64->getValue()) >> 16;
                unsigned awcache = (0x00F00000 & wreg64->getValue()) >> 20;
                unsigned clas    = (0x03000000 & wreg64->getValue()) >> 24;
                ASSERT_EQ(mcid, 2);
                ASSERT_EQ(wreg64->getBaseIndex(), 28); // 28 is the base of degrade
                ASSERT_EQ(arcache, 0xf);               // shared alloc
                ASSERT_EQ(awcache, 0xf);               // shared alloc
                ASSERT_EQ(clas, 0x3);                  // Top class
                foundWreg64ForMcid12 = true;
            }
            ASSERT_NE(wreg64->getDregOffset() * sizeof(uint32_t), 0xa228); // we shouldn't write mcid 0 with wreg64 (tensor 1)
        }

        // Verify allocation policy & class for tensors 1, 2 and 3 via the private tensors HBW_AXI_CFG
        axiCfgAddr = 0xa228; // 0xa228 is tensor 1 HBW_AXI_CFG
        regVal = 0;
        if (getValOfReg(cmd.get(), axiCfgAddr, regVal))
        {
            foundWriteAllocAndClassForTensor1 = true;
            unsigned mcid    = 0x0000FFFF & regVal;
            unsigned arcache = (0x000F0000 & regVal) >> 16;
            unsigned awcache = (0x00F00000 & regVal) >> 20;
            unsigned clas    = (0x03000000 & regVal) >> 24;
            ASSERT_EQ(mcid, 0);      // mcid is 0
            ASSERT_EQ(arcache, 0x3); // no allocate
            ASSERT_EQ(awcache, 0x3); // no allocate
            ASSERT_EQ(clas, 0x1);    // Normal class
        }
        axiCfgAddr = 0xa284; // 0xa284 is tensor 2 HBW_AXI_CFG
        regVal = 0;
        if (getValOfReg(cmd.get(), axiCfgAddr, regVal))
        {
            foundWriteAllocAndClassForTensor2 = true;
            unsigned mcid    = 0x0000FFFF & regVal;
            unsigned arcache = (0x000F0000 & regVal) >> 16;
            unsigned awcache = (0x00F00000 & regVal) >> 20;
            unsigned clas    = (0x03000000 & regVal) >> 24;
            ASSERT_EQ(mcid, 0);      // mcid is 0 since we programed it via the fast config
            ASSERT_EQ(arcache, 0xb); // dcore alloc
            ASSERT_EQ(awcache, 0xb); // dcore alloc
            ASSERT_EQ(clas, 0x2);    // High class
        }
        axiCfgAddr = 0xa2e0; // 0xa2e0 is tensor 3 HBW_AXI_CFG
        regVal = 0;
        if (getValOfReg(cmd.get(), axiCfgAddr, regVal))
        {
            foundWriteAllocAndClassForTensor3 = true;
            unsigned mcid    = 0x0000FFFF & regVal;
            unsigned arcache = (0x000F0000 & regVal) >> 16;
            unsigned awcache = (0x00F00000 & regVal) >> 20;
            unsigned clas    = (0x03000000 & regVal) >> 24;
            ASSERT_EQ(mcid, 0);      // mcid is 0 since we programed it via the fast config
            ASSERT_EQ(arcache, 0xb); // dcore alloc
            ASSERT_EQ(awcache, 0xb); // dcore alloc
            ASSERT_EQ(clas, 0x2);    // High class
        }
    }
    ASSERT_TRUE(foundWreg64ForMcid10);
    ASSERT_TRUE(foundWreg64ForMcid11);
    ASSERT_TRUE(foundWreg64ForMcid12);
    ASSERT_TRUE(foundWriteAllocAndClassForTensor1);
    ASSERT_TRUE(foundWriteAllocAndClassForTensor2);
    ASSERT_TRUE(foundWriteAllocAndClassForTensor3);
}

TEST_F(CacheMaintenanceTest, tpc_no_maintenance)
{
    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add TPC (nop) node
    TSize     size1  = 1;
    TensorPtr tpcI   = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr tpcO   = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy0 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy1 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    TensorPtr dummy2 = TensorPtr(new Tensor(1, &size1, syn_type_single));
    tpcI->setMemoryDescriptor(memDesc);
    tpcO->setMemoryDescriptor(memDesc);
    dummy0->setMemoryDescriptor(memDesc);
    dummy1->setMemoryDescriptor(memDesc);
    dummy2->setMemoryDescriptor(memDesc);
    tpcI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    tpcO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 2);
    dummy0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 3);
    dummy1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 4);
    dummy2->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    NodePtr tpcNop = NodeFactory::createNode({tpcI, dummy0, dummy1, dummy2}, {tpcO}, nullptr, NOP_KERNEL_NAME, "tpc_node");
    GraphEditor::addNode(g, tpcNop);

    // Set cache metadata for TPC node to NOP and mcid 0
    NodeAnnotation& ann = tpcNop->getNodeAnnotation();
    ann.splitToLogicalROIs = false;
    md.mcid = 0;
    md.cmAction = NOP;
    md.cacheClass = Normal;
    md.cacheDirective = NoAllocate;
    ann.inputsCacheMetaData.push_back(md);
    ann.inputsCacheMetaData.push_back(md);
    ann.inputsCacheMetaData.push_back(md);
    ann.inputsCacheMetaData.push_back(md);
    ann.outputsCacheMetaData.push_back(md);

    // Compile
    ASSERT_EQ(g.compile(), true);
}

TEST_F(CacheMaintenanceTest, rollover_mme_only)
{
    constexpr unsigned numNodes      = 10;
    constexpr unsigned rolloverLimit = 7;

    setGlobalConfForTest(GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING, std::to_string(rolloverLimit));

    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME nodes
    uint8_t       dim            = 2;
    auto          dataType       = syn_type_single;
    const TSize   sizesIn0[]     = {64, 16, 1, 1};
    const TSize   sizesIn1[]     = {64, 64, 1, 1};
    TSize         bgemmOutSize[] = {64, 16, 1, 1};
    auto          bgemmGuid      = NodeFactory::batchGemmNodeTypeName;
    synGEMMParams params {};

    for (unsigned i = 0; i < numNodes; i++)
    {
        TensorPtr bgemmInput0 = TensorPtr(new Tensor(dim, sizesIn0, dataType));
        TensorPtr bgemmInput1 = TensorPtr(new Tensor(dim, sizesIn1, dataType));
        TensorPtr bgemmOut    = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

        bgemmInput0->setDramOffset(i * 3 * 0x100000 + 0x100000);
        bgemmInput1->setDramOffset(i * 3 * 0x100000 + 0x200000);
        bgemmInput0->setMemoryDescriptor(memDesc);
        bgemmInput1->setMemoryDescriptor(memDesc);
        bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 3 + 1);
        bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 3 + 2);
        bgemmOut->setDramOffset(i * 3 * 0x100000 + 0x300000);
        bgemmOut->setMemoryDescriptor(memDesc);
        bgemmOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 3 + 3);

        NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "");
        GraphEditor::addNode(g, bgemm);

        // Set cache metadata for MME node:
        NodeAnnotation& ann = bgemm->getNodeAnnotation();
        ann.splitToLogicalROIs = false;
        md.mcid = i * 3 + 1;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // first input
        md.mcid = i * 3 + 2;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // second input
        md.mcid = i * 3 + 3;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.outputsCacheMetaData.push_back(md); // output
    }

    // Compile
    ASSERT_EQ(g.compile(), true);

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        unsigned exeCmdCounter = 0;
        bool     expectingRollover = false;
        for (const auto& cmd : q->getCommands(false))
        {
            if (expectingRollover)
            {
                McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
                ASSERT_TRUE(rolloverCmd != nullptr);
                ASSERT_EQ(rolloverCmd->getTarget(), rolloverLimit);
                ASSERT_EQ(rolloverCmd->getTargetXps(), 0);
                ASSERT_TRUE(rolloverCmd->isSwitchCQ());
                break;
            }

            if (cmd->isExe()) exeCmdCounter++;

            // If we encountered rolloverLimit exe commands then the following command should be rollover
            if (exeCmdCounter == rolloverLimit) expectingRollover = true;
        }
    }

    //---- Verify CME ECB commands ----
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ecb_t pDynamicEcb = {0};
    ecb_t pStaticEcb  = {0};

    bool foundTpc = false;

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pStaticEcb  = recipe->arc_jobs[i].static_ecb;
        }
        // There are no TPC nodes in the graph but since TPC must participate in rollover flow, we expect to get TPC
        // rollover command via the pre-nodes mechanism
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            foundTpc = true;
        }
    }
    ASSERT_TRUE(foundTpc);

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDynamicEcb.cmds != nullptr);
    ASSERT_TRUE(pStaticEcb.cmds_size == 0);
    ASSERT_TRUE(pStaticEcb.cmds_eng_offset == 0);
    ASSERT_TRUE(pStaticEcb.cmds == nullptr);

    struct cme_arc_cmd_mcid_rollover_t* cmeRolloverCmd =
        (cme_arc_cmd_mcid_rollover_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_discard_cls_t) * rolloverLimit);

    ASSERT_EQ(cmeRolloverCmd->cmd_type, CME_ECBL_CMD_MCID_ROLLOVER);
    ASSERT_EQ(cmeRolloverCmd->signal_mme, 1);
    ASSERT_EQ(cmeRolloverCmd->signal_rot, 0);

    //---- Verify TPC ECB commands ----
    ecb_t pTpcDynamicEcb = {0};
    ecb_t pTpcStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            pTpcDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pTpcStaticEcb  = recipe->arc_jobs[i].static_ecb;
        }
    }
    // For TPC - we expect to get TPC rollover command via the pre-nodes mechanism
    struct eng_arc_cmd_mcid_rollover_t* tpcPreNodeRolloverCmd =
        (eng_arc_cmd_mcid_rollover_t*)(pTpcDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t));

    ASSERT_EQ(tpcPreNodeRolloverCmd->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(tpcPreNodeRolloverCmd->switch_cq, 0);
    ASSERT_EQ(tpcPreNodeRolloverCmd->yield, 0);
    ASSERT_EQ(tpcPreNodeRolloverCmd->target, 0);
    ASSERT_EQ(tpcPreNodeRolloverCmd->target_xpose, 0);

    // and the following command should be NOP-switch-CQ
    struct eng_arc_cmd_nop_t* nopCmd = (eng_arc_cmd_nop_t*)(pTpcDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                                            sizeof(eng_arc_cmd_mcid_rollover_t));

    ASSERT_EQ(nopCmd->cmd_type, ECB_CMD_NOP);
    ASSERT_EQ(nopCmd->switch_cq, 1);
    ASSERT_EQ(nopCmd->yield, 0);
    ASSERT_EQ(nopCmd->padding, 0);

    //---- Verify MME ECB commands ----
    ecb_t pMmeDynamicEcb = {0};
    ecb_t pMmeStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::MME)
        {
            pMmeDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pMmeStaticEcb  = recipe->arc_jobs[i].static_ecb;
        }
    }
    // For MME - we expect to get MME rollover command. Command should appaer after several other commands:
    // 1 ListSizeEngArcCommand, 1 NopEngArcCommand, 10 ScheduleDmaEngArcCommand, 7 DynamicWorkDistEngArcCommand
    struct eng_arc_cmd_mcid_rollover_t* mmeRolloverCmd =
        (eng_arc_cmd_mcid_rollover_t*)(pMmeDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * rolloverLimit);

    ASSERT_EQ(mmeRolloverCmd->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(mmeRolloverCmd->switch_cq, 1);
    ASSERT_EQ(mmeRolloverCmd->yield, 1);
    ASSERT_EQ(mmeRolloverCmd->target, rolloverLimit);
    ASSERT_EQ(mmeRolloverCmd->target_xpose, 0);
}

TEST_F(CacheMaintenanceTest, rollover_mme_and_tpc)
{
    constexpr unsigned numNodes      = 10;
    constexpr unsigned rolloverLimit = 7;

    setGlobalConfForTest(GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING, std::to_string(rolloverLimit));

    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME nodes
    uint8_t       dim            = 2;
    auto          dataType       = syn_type_single;
    const TSize   sizesIn0[]     = {64, 16, 1, 1};
    const TSize   sizesIn1[]     = {64, 64, 1, 1};
    TSize         bgemmOutSize[] = {64, 16, 1, 1};
    auto          bgemmGuid      = NodeFactory::batchGemmNodeTypeName;
    synGEMMParams params {};

    for (unsigned i = 0; i < numNodes; i++)
    {
        TensorPtr bgemmInput0 = TensorPtr(new Tensor(dim, sizesIn0, dataType));
        TensorPtr bgemmInput1 = TensorPtr(new Tensor(dim, sizesIn1, dataType));
        TensorPtr bgemmOut    = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

        bgemmInput0->setDramOffset(i * 3 * 0x100000 + 0x100000);
        bgemmInput1->setDramOffset(i * 3 * 0x100000 + 0x200000);
        bgemmInput0->setMemoryDescriptor(memDesc);
        bgemmInput1->setMemoryDescriptor(memDesc);
        bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 1);
        bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 2);
        bgemmOut->setDramOffset(i * 3 * 0x100000 + 0x300000);
        bgemmOut->setMemoryDescriptor(memDesc);
        bgemmOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 3);

        NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "");
        GraphEditor::addNode(g, bgemm);

        // Set cache metadata for MME node:
        NodeAnnotation& ann = bgemm->getNodeAnnotation();
        ann.splitToLogicalROIs = false;
        md.mcid = i * 5 + 1;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // first input
        md.mcid = i * 5 + 2;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // second input
        md.mcid = i * 5 + 3;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.outputsCacheMetaData.push_back(md); // output

        // Add TPC (nop) node
        TSize     size1  = 1;
        TensorPtr dummy0 = TensorPtr(new Tensor(1, &size1, syn_type_single));
        TensorPtr dummy1 = TensorPtr(new Tensor(1, &size1, syn_type_single));
        dummy0->setMemoryDescriptor(memDesc);
        dummy1->setMemoryDescriptor(memDesc);
        dummy0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 4);
        dummy1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 5);
        NodePtr tpcNop = NodeFactory::createNode({dummy0}, {dummy1}, nullptr, NOP_KERNEL_NAME, "tpc_node");
        GraphEditor::addNode(g, tpcNop);

        // Set cache metadata for TPC node:
        NodeAnnotation& ann2 = tpcNop->getNodeAnnotation();
        ann2.splitToLogicalROIs = false;
        md.mcid = i * 5 + 4;
        md.cmAction = DISCARD;
        md.cacheClass = Top;
        md.cacheDirective = SharedAllocate;
        ann2.inputsCacheMetaData.push_back(md); // input
        md.mcid = i * 5 + 5;
        md.cmAction = DISCARD;
        md.cacheClass = Top;
        md.cacheDirective = SharedAllocate;
        ann2.outputsCacheMetaData.push_back(md); // output
    }

    // Compile
    ASSERT_EQ(g.compile(), true);

    //---- Verify TPC QMAN commands ----
    const CommandQueuePtr& queue = g.getTpcQueue();
    unsigned rolloverCount  = 0;
    unsigned expectedTarget = 3; // the first rollover happens when TPC signaled 3
    for (const auto& cmd : queue->getCommands(false))
    {
        McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
        if (rolloverCmd != nullptr)
        {
            ASSERT_EQ(rolloverCmd->getTarget(), expectedTarget);
            ASSERT_EQ(rolloverCmd->getTargetXps(), 0);
            expectedTarget = rolloverLimit; // the second rollover happens when TPC signaled 7
            rolloverCount++;
        }
    }
    ASSERT_EQ(rolloverCount, 2); // 2 rollovers

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        unsigned rolloverCount  = 0;
        unsigned expectedTarget = 4; // the first rollover happens when MME signaled 4
        for (const auto& cmd : q->getCommands(false))
        {
            McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
            if (rolloverCmd != nullptr)
            {
                ASSERT_EQ(rolloverCmd->getTarget(), expectedTarget);
                ASSERT_EQ(rolloverCmd->getTargetXps(), 0);
                expectedTarget = rolloverLimit; // the second rollover happens when MME signaled 7
                rolloverCount++;
            }
        }
        ASSERT_EQ(rolloverCount, 2); // 2 rollovers
    }

    //---- Verify CME ECB commands ----
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ecb_t pDynamicEcb = {0};
    ecb_t pStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pStaticEcb  = recipe->arc_jobs[i].static_ecb;
            break;
        }
    }

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDynamicEcb.cmds != nullptr);
    ASSERT_TRUE(pStaticEcb.cmds_size == 0);
    ASSERT_TRUE(pStaticEcb.cmds_eng_offset == 0);
    ASSERT_TRUE(pStaticEcb.cmds == nullptr);

    // Verify the first rollover
    struct cme_arc_cmd_mcid_rollover_t* cmeRolloverCmd =
        (cme_arc_cmd_mcid_rollover_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_discard_cls_t) * rolloverLimit);

    ASSERT_EQ(cmeRolloverCmd->cmd_type, CME_ECBL_CMD_MCID_ROLLOVER);
    ASSERT_EQ(cmeRolloverCmd->signal_mme, 1);
    ASSERT_EQ(cmeRolloverCmd->signal_rot, 0);

    // Verify the second rollover, skip over 2 sets of rolloverLimit discard commands and 1 rollover command
    cmeRolloverCmd = (cme_arc_cmd_mcid_rollover_t*)
                        (pDynamicEcb.cmds +
                         sizeof(cme_arc_cmd_discard_cls_t) * 2 * rolloverLimit +
                         sizeof(cme_arc_cmd_mcid_rollover_t));

    ASSERT_EQ(cmeRolloverCmd->cmd_type, CME_ECBL_CMD_MCID_ROLLOVER);
    ASSERT_EQ(cmeRolloverCmd->signal_mme, 1);
    ASSERT_EQ(cmeRolloverCmd->signal_rot, 0);

    //---- Verify TPC ECB commands ----
    ecb_t pTpcDynamicEcb = {0};
    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            pTpcDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            break;
        }
    }
    // For TPC - we expect to get 2 rollover commands

    // 1st rollover
    struct eng_arc_cmd_mcid_rollover_t* tpcRolloverCmd1 =
        (eng_arc_cmd_mcid_rollover_t*)(pTpcDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * 3);

    ASSERT_EQ(tpcRolloverCmd1->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(tpcRolloverCmd1->switch_cq, 1);
    ASSERT_EQ(tpcRolloverCmd1->yield, 0);
    ASSERT_EQ(tpcRolloverCmd1->target, 3);
    ASSERT_EQ(tpcRolloverCmd1->target_xpose, 0);

    // 2nd rollover
    struct eng_arc_cmd_mcid_rollover_t* tpcRolloverCmd2 =
        (eng_arc_cmd_mcid_rollover_t*)(pTpcDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * rolloverLimit +
                                       sizeof(eng_arc_cmd_mcid_rollover_t));

    ASSERT_EQ(tpcRolloverCmd2->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(tpcRolloverCmd2->switch_cq, 1);
    ASSERT_EQ(tpcRolloverCmd2->yield, 1);
    ASSERT_EQ(tpcRolloverCmd2->target, rolloverLimit);
    ASSERT_EQ(tpcRolloverCmd2->target_xpose, 0);

    //---- Verify MME ECB commands ----
    ecb_t pMmeDynamicEcb = {0};
    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::MME)
        {
            pMmeDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            break;
        }
    }
    // For MME - we expect to get 2 rollover commands

    // 1st rollover
    struct eng_arc_cmd_mcid_rollover_t* mmeRolloverCmd1 =
        (eng_arc_cmd_mcid_rollover_t*)(pMmeDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * 4);

    ASSERT_EQ(mmeRolloverCmd1->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(mmeRolloverCmd1->switch_cq, 1);
    ASSERT_EQ(mmeRolloverCmd1->yield, 1);
    ASSERT_EQ(mmeRolloverCmd1->target, 4);
    ASSERT_EQ(mmeRolloverCmd1->target_xpose, 0);

    // 2nd rollover
    struct eng_arc_cmd_mcid_rollover_t* mmeRolloverCmd2 =
        (eng_arc_cmd_mcid_rollover_t*)(pMmeDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * rolloverLimit +
                                       sizeof(eng_arc_cmd_mcid_rollover_t));

    ASSERT_EQ(mmeRolloverCmd2->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(mmeRolloverCmd2->switch_cq, 1);
    ASSERT_EQ(mmeRolloverCmd2->yield, 1);
    ASSERT_EQ(mmeRolloverCmd2->target, rolloverLimit);
    ASSERT_EQ(mmeRolloverCmd2->target_xpose, 0);
}

TEST_F(CacheMaintenanceTest, rollover_mme_and_xpose)
{
    constexpr unsigned numNodes      = 10;
    constexpr unsigned rolloverLimit = 7;

    setGlobalConfForTest(GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING, std::to_string(rolloverLimit));
    setGlobalConfForTest(GCFG_ENABLE_PERSISTENT_OUTPUT_REUSE, "false");

    Gaudi3GraphCacheMaintenanceTest g;
    synMemoryDescriptor             memDesc(true);  // persistent
    CacheMetaData                   md;

    // Add MME nodes
    uint8_t       dim            = 2;
    auto          dataType       = syn_type_single;
    const TSize   sizesIn0[]     = {64, 16, 1, 1};
    const TSize   sizesIn1[]     = {64, 64, 1, 1};
    TSize         bgemmOutSize[] = {64, 16, 1, 1};
    auto          bgemmGuid      = NodeFactory::batchGemmNodeTypeName;
    synGEMMParams params {};

    for (unsigned i = 0; i < numNodes; i++)
    {
        TensorPtr bgemmInput0 = TensorPtr(new Tensor(dim, sizesIn0, dataType));
        TensorPtr bgemmInput1 = TensorPtr(new Tensor(dim, sizesIn1, dataType));
        TensorPtr bgemmOut    = TensorPtr(new Tensor(dim, bgemmOutSize, dataType));

        bgemmInput0->setDramOffset(i * 5 * 0x100000 + 0x100000);
        bgemmInput1->setDramOffset(i * 5 * 0x100000 + 0x200000);
        bgemmInput0->setMemoryDescriptor(memDesc);
        bgemmInput1->setMemoryDescriptor(memDesc);
        bgemmInput0->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 1);
        bgemmInput1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 2);
        bgemmOut->setDramOffset(i * 5 * 0x100000 + 0x300000);
        bgemmOut->setMemoryDescriptor(memDesc);
        bgemmOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 3);

        NodePtr bgemm = NodeFactory::createNode({bgemmInput0, bgemmInput1}, {bgemmOut}, &params, bgemmGuid, "");
        GraphEditor::addNode(g, bgemm);

        // Set cache metadata for MME node:
        NodeAnnotation& ann = bgemm->getNodeAnnotation();
        ann.splitToLogicalROIs = false;
        md.mcid = i * 5 + 1;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // first input
        md.mcid = i * 5 + 2;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.inputsCacheMetaData.push_back(md); // second input
        md.mcid = i * 5 + 3;
        md.cmAction = DISCARD;
        md.cacheClass = High;
        md.cacheDirective = HomeAllocate;
        ann.outputsCacheMetaData.push_back(md); // output

        // Add transpose node
        const TSize inMaxSizes[] = {4, 6, 1, 2};
        const TSize outMaxSizes[] = {2, 6, 1, 4};
        TensorPtr xposeIn = TensorPtr(new Tensor(4, inMaxSizes, syn_type_single));
        TensorPtr xposeOut = TensorPtr(new Tensor(4, outMaxSizes, syn_type_single));
        xposeIn->setDramOffset(i * 5 * 0x100000 + 0x400000);
        xposeIn->setMemoryDescriptor(memDesc);
        xposeIn->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 4);
        xposeOut->setDramOffset(i * 5 * 0x100000 + 0x500000);
        xposeOut->setMemoryDescriptor(memDesc);
        xposeOut->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + i * 5 + 5);
        synTransposeParams xposeParams = {{TPD_4Dim_Batch, TPD_Width, TPD_Height, TPD_Channel}, 4};
        NodePtr xposeNode = NodeFactory::createNode({xposeIn}, {xposeOut}, &xposeParams, NodeFactory::transposeNodeTypeName, "");
        GraphEditor::addNode(g, xposeNode);

        // Set cache metadata for transpose node:
        NodeAnnotation& ann2 = xposeNode->getNodeAnnotation();
        ann2.splitToLogicalROIs = false;
        md.mcid = i * 5 + 4;
        md.cmAction = DISCARD;
        md.cacheClass = Top;
        md.cacheDirective = SharedAllocate;
        ann2.inputsCacheMetaData.push_back(md); // input
        md.mcid = i * 5 + 5;
        md.cmAction = DISCARD;
        md.cacheClass = Top;
        md.cacheDirective = SharedAllocate;
        ann2.outputsCacheMetaData.push_back(md); // output
    }

    // Compile
    ASSERT_EQ(g.compile(), true);

    // TPC nodes are inserted during graph compilation
    //---- Verify TPC QMAN commands ----
    const CommandQueuePtr& queue = g.getTpcQueue();
    for (const auto& cmd : queue->getCommands(false))
    {
        McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
        if (rolloverCmd != nullptr)
        {
            ASSERT_EQ(rolloverCmd->getTarget(), rolloverLimit);
            ASSERT_EQ(rolloverCmd->getTargetXps(), 0);
            ASSERT_TRUE(rolloverCmd->isSwitchCQ());
            break;
        }
    }

    //---- Verify MME QMAN commands ----
    for (CommandQueuePtr q : g.getMmeQueues())
    {
        bool expectingCancledRollover = true;
        bool expectingRollover = false;
        for (const auto& cmd : q->getCommands(false))
        {
            McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
            if (rolloverCmd == nullptr) continue;

            if (expectingCancledRollover)
            {
                McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
                ASSERT_TRUE(rolloverCmd != nullptr);
                ASSERT_EQ(rolloverCmd->getTarget(), 0);
                ASSERT_EQ(rolloverCmd->getTargetXps(), 0);
                ASSERT_TRUE(rolloverCmd->isSwitchCQ());
                expectingCancledRollover = false;
                expectingRollover = true;
                continue;
            }
            if (expectingRollover)
            {
                McidRollover* rolloverCmd = dynamic_cast<McidRollover*>(cmd.get());
                ASSERT_TRUE(rolloverCmd != nullptr);
                ASSERT_EQ(rolloverCmd->getTarget(), rolloverLimit);
                ASSERT_EQ(rolloverCmd->getTargetXps(), rolloverLimit);
                ASSERT_TRUE(rolloverCmd->isSwitchCQ());
                break;
            }
        }
    }

    //---- Verify CME ECB commands ----
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ecb_t pDynamicEcb = {0};
    ecb_t pStaticEcb  = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            pStaticEcb  = recipe->arc_jobs[i].static_ecb;
            break;
        }
    }

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDynamicEcb.cmds != nullptr);
    ASSERT_TRUE(pStaticEcb.cmds_size == 0);
    ASSERT_TRUE(pStaticEcb.cmds_eng_offset == 0);
    ASSERT_TRUE(pStaticEcb.cmds == nullptr);

    struct cme_arc_cmd_mcid_rollover_t* cmeRolloverCmd =
        (cme_arc_cmd_mcid_rollover_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_discard_cls_t) * rolloverLimit);

    ASSERT_EQ(cmeRolloverCmd->cmd_type, CME_ECBL_CMD_MCID_ROLLOVER);
    ASSERT_EQ(cmeRolloverCmd->signal_mme, 1);
    ASSERT_EQ(cmeRolloverCmd->signal_rot, 0);

    //---- Verify TPC ECB commands ----
    ecb_t pTpcDynamicEcb = {0};
    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            pTpcDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            break;
        }
    }
    // For TPC - we expect to get 2 rollover commands

    // 1st rollover
    struct eng_arc_cmd_mcid_rollover_t* tpcRolloverCmd =
        (eng_arc_cmd_mcid_rollover_t*)(pTpcDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) + sizeof(eng_arc_cmd_sched_dma_t) * 10 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * rolloverLimit);

    ASSERT_EQ(tpcRolloverCmd->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(tpcRolloverCmd->switch_cq, 1);
    ASSERT_EQ(tpcRolloverCmd->yield, 1);
    ASSERT_EQ(tpcRolloverCmd->target, rolloverLimit);
    ASSERT_EQ(tpcRolloverCmd->target_xpose, 0);

    //---- Verify MME ECB commands ----
    ecb_t pMmeDynamicEcb = {0};
    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::MME)
        {
            pMmeDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            break;
        }
    }
    // For MME - we expect to get MME rollover command. Command should appaer after several other commands:
    // 1 ListSizeEngArcCommand, 2 NopEngArcCommand, 20 ScheduleDmaEngArcCommand, 14 DynamicWorkDistEngArcCommand
    // Note that although both target and target_xpose are set - one rollover command is canceled
    struct eng_arc_cmd_mcid_rollover_t* mmeRolloverCmd =
        (eng_arc_cmd_mcid_rollover_t*)(pMmeDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                       sizeof(eng_arc_cmd_nop_t) * 2 + sizeof(eng_arc_cmd_sched_dma_t) * 20 +
                                       sizeof(eng_arc_cmd_wd_fence_and_exec_t) * 2 * rolloverLimit);

    ASSERT_EQ(mmeRolloverCmd->cmd_type, ECB_CMD_MCID_ROLLOVER);
    ASSERT_EQ(mmeRolloverCmd->switch_cq, 1);
    ASSERT_EQ(mmeRolloverCmd->yield, 1);
    ASSERT_EQ(mmeRolloverCmd->target, rolloverLimit);
    ASSERT_EQ(mmeRolloverCmd->target_xpose, rolloverLimit);
    // Note the cancled rollover command
}

TEST_F(CacheMaintenanceTest, ensure_non_tensor_axi_config_is_set)
{
    Gaudi3GraphCacheMaintenanceTest g;

    const unsigned tensor_dim = 1;
    const TSize    size       = 1024 * 32 * 10;

    TensorPtr tI = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr tO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    synMemoryDescriptor memDesc(true);
    tI->setMemoryDescriptor(memDesc);
    tO->setMemoryDescriptor(memDesc);
    tI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    tO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    NodePtr nTpc = NodeFactory::createNode({tI}, {tO}, nullptr, "relu_fwd_f32", "node_name_relu");
    GraphEditor::addNode(g, nTpc);
    ASSERT_EQ(g.compile(), true);

    bool foundIcache = false;
    bool foundDcache = false;
    for (const auto& cmd : g.getTpcQueue()->getCommands(false))
    {
        unsigned regVal = 0;
        if (getValOfReg(cmd.get(), 0xa898, regVal)) // 0xa898 is icache_axi_cfg
        {
            foundIcache = true;
            ASSERT_EQ(regVal, 0x3);
        }
        if (getValOfReg(cmd.get(), 0xa8a0, regVal)) // 0xa8a0 is dcache_axi_cfg
        {
            foundDcache = true;
            ASSERT_EQ(regVal, 0x3);
        }
    }
    ASSERT_TRUE(foundIcache);
    ASSERT_TRUE(foundDcache);
}
