#include <bitset>
#include "gtest/gtest.h"
#include "graph_optimizer_test.h"
#include "recipe_test_base.h"
#include "scoped_configuration_change.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "recipe.h"
#include "recipe_allocator.h"
#include "recipe_blob.h"
#include "recipe_program.h"
#include "recipe_ecb.h"
#include "gaudi2_arc_eng_packets.h"
// the type of graph is irrelevant to this test suit, we just need a graph with new recipe
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "platform/gaudi2/graph_compiler/gaudi2_eng_arc_hooks.h"

static const unsigned TEST_BLOB_SIZE = 32;

class RecipeArcTest : public GraphOptimizerTest
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY, "false");
    }
};

TEST_F(RecipeArcTest, empty_ecb)
{
    Gaudi2Graph g;  // the type of graph is irrelevant to this test, we just need a graph with new recipe
    g.compile();    // compile here so a queue cmd factory will exist for NOP addition in addBlob

    // Prepare blob container
    RecipeBlobContainer    blobContainer(&g);
    RecipeProgramContainer progContainer;
    RecipeEcbContainer     ecbContainer;
    uint32_t               numArcJobs = 1;                       // purposeley non-zero value
    arc_job_t*             pArcJobs   = (arc_job_t*)0xffffffff;  // purposeley non-nullptr value
    RecipeAllocator        recipeAlloc;

    ecbContainer.setEngArcHooks(std::make_shared<gaudi2::EngArcHooks>(gaudi2::EngArcHooks::instance()));
    ecbContainer.generateECBs(progContainer, blobContainer);
    ecbContainer.serialize(&numArcJobs, &pArcJobs, &recipeAlloc);

    ASSERT_EQ(numArcJobs, 0);
    ASSERT_EQ(pArcJobs, nullptr);
}

TEST_F(RecipeArcTest, ecb_with_nop_padding)
{
    Gaudi2Graph g;  // the type of graph is irrelevant to this test, we just need a graph with new recipe
    g.compile();    // compile here so a queue cmd factory will exist for NOP addition in addBlob

    // Prepare blob container
    RecipeBlobContainer blobContainer(&g);
    uint64_t            idx;
    bool                isReused;

    for (unsigned i = 0; i < 32; i++)
    {
        auto blob = new RecipeBlob(&g);  // will be deleted by the d'tor of the container
        std::fill_n(blob->reserveBytes(TEST_BLOB_SIZE), TEST_BLOB_SIZE, i);  // avoid compression by unique value init
        blobContainer.addBlob(blob, idx, isReused);
    }

    // Prepare program container
    RecipeProgramContainer progContainer;
    unsigned               progIdx;
    auto&                  program = progContainer.getProgram(0, DEVICE_TPC, progIdx);

    for (unsigned i = 0; i < 32; i++)
    {
        program.insertBlobIndex(i);
    }

    ASSERT_EQ(blobContainer.getBlobCount(), 32);
    ASSERT_EQ(progContainer.getNumPrograms(), 1);

    // Start testing the ECB generation
    RecipeEcbContainer                ecbContainer;
    uint32_t                          numArcJobs = 0;
    arc_job_t*                        pArcJobs   = nullptr;
    char*                             pChar      = nullptr;
    struct eng_arc_cmd_list_size_t*   sizeCmd    = nullptr;
    struct eng_arc_cmd_static_desc_v2_t* staticCmd  = nullptr;
    struct eng_arc_cmd_nop_t*         nopCmd     = nullptr;
    RecipeAllocator                   recipeAlloc;

    ecbContainer.setEngArcHooks(std::make_shared<gaudi2::EngArcHooks>(gaudi2::EngArcHooks::instance()));
    ecbContainer.generateECBs(progContainer, blobContainer);
    ecbContainer.serialize(&numArcJobs, &pArcJobs, &recipeAlloc);

    ASSERT_EQ(numArcJobs, 1);
    ASSERT_NE(pArcJobs, nullptr);

    // We expect to have 1 ListSize command, 31 static_cp_dma, 1 nop with no padding, 1 static_cp_dma, 1 nop with
    // switch_cq and finally 1 nop with padding
    unsigned expEcbSize = sizeof(struct eng_arc_cmd_list_size_t) + 2 * sizeof(struct eng_arc_cmd_nop_t) +
                          32 * sizeof(struct eng_arc_cmd_static_desc_v2_t);
    unsigned finalNopSize = STATIC_COMPUTE_ECB_LIST_BUFF_SIZE - (expEcbSize % STATIC_COMPUTE_ECB_LIST_BUFF_SIZE);
    expEcbSize += finalNopSize;

    ASSERT_EQ(pArcJobs[0].static_ecb.cmds_size, expEcbSize);

    // 1 ListSize
    pChar   = (char*)pArcJobs[0].static_ecb.cmds;
    sizeCmd = (struct eng_arc_cmd_list_size_t*)pChar;
    ASSERT_EQ(sizeCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_LIST_SIZE);
    ASSERT_EQ(sizeCmd->list_size, expEcbSize);
    ASSERT_EQ(sizeCmd->yield, 0);  // no yielding is expected
    pChar += sizeof(struct eng_arc_cmd_list_size_t);

    // 31 static cp dma
    for (unsigned i = 0; i < 31; i++)
    {
        staticCmd = (struct eng_arc_cmd_static_desc_v2_t*)pChar;
        ASSERT_EQ(staticCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_STATIC_DESC_V2);
        ASSERT_EQ(staticCmd->cpu_index, CPU_ID_ALL);
        pChar += sizeof(struct eng_arc_cmd_static_desc_v2_t);
    }

    // 1 nop without padding
    nopCmd = (struct eng_arc_cmd_nop_t*)pChar;
    ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
    ASSERT_EQ(nopCmd->padding, 0);  // no padding in this nop
    ASSERT_EQ(nopCmd->yield, 0);    // no yielding is expected
    pChar += sizeof(struct eng_arc_cmd_nop_t);

    // 1 static cp dma
    staticCmd = (struct eng_arc_cmd_static_desc_v2_t*)pChar;
    ASSERT_EQ(staticCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_STATIC_DESC_V2);
    ASSERT_EQ(staticCmd->cpu_index, CPU_ID_ALL);
    pChar += sizeof(struct eng_arc_cmd_static_desc_v2_t);

    // 1 nop with switch CQ
    nopCmd = (struct eng_arc_cmd_nop_t*)pChar;
    ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
    ASSERT_EQ(nopCmd->padding, 0);
    ASSERT_EQ(nopCmd->switch_cq, 1);
    ASSERT_EQ(nopCmd->yield, 0);  // no yielding is expected
    pChar += sizeof(struct eng_arc_cmd_nop_t);

    // 1 final nop with padding
    nopCmd = (struct eng_arc_cmd_nop_t*)pChar;
    ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
    // the padding field is in DWord units and doesn't include the size of the nop command itself
    ASSERT_EQ(nopCmd->padding, (finalNopSize / DWORD_SIZE) - 1);
    ASSERT_EQ(nopCmd->yield, 0);  // no yielding is expected
}

class Gaudi2GraphArcWorkDistTest : public Gaudi2Graph
{
public:
    CommandQueuePtr getTpcQueue0() { return m_codeGenerator->getTpcDispatcher()->getQueue(0); }

    // queue 0 on first dispather
    CommandQueuePtr getDmaQueue0() { return m_codeGenerator->getDmaDispatchers().begin()->second->getQueue(0); }

    // queue 0 on last dispatcher
    CommandQueuePtr getDmaQueue1() { return m_codeGenerator->getDmaDispatchers().rbegin()->second->getQueue(0); }
};

TEST_F(RecipeArcTest, dynamic_work_dist)
{
    Gaudi2GraphArcWorkDistTest g;

    const unsigned tensor_dim = 1;
    const TSize    size       = 1024 * 32 * 10;

    TensorPtr dmaI = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr dmaO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr tpcO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    synMemoryDescriptor memDesc(true);
    dmaI->setMemoryDescriptor(memDesc);
    tpcO->setMemoryDescriptor(memDesc);
    dmaI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    tpcO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    // The Relu node depends on the DMA node and thus will have pipeline syncs
    NodePtr nDma = NodeFactory::createNode({dmaI}, {dmaO}, nullptr, "memcpy", "node_name_memcpy");
    NodePtr nTpc = NodeFactory::createNode({dmaO}, {tpcO}, nullptr, "relu_fwd_f32", "node_name_relu");

    GraphEditor::addNode(g, nDma);
    GraphEditor::addNode(g, nTpc);
    g.compile();

    unsigned executeCount = 0;

    const CommandQueuePtr& queue = g.getTpcQueue0();

    for (const auto& cmd : queue->getCommands(false))
    {
        if (cmd->isExe()) executeCount++;
    }

    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ASSERT_TRUE(recipe->dynamic_blobs_buffer_size > 0);
    ASSERT_TRUE(recipe->dynamic_blobs_buffer != nullptr);

    ecb_t pDynamicEcb = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            pDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
            break;
        }
    }

    ASSERT_TRUE(pDynamicEcb.cmds_size > 0);

    // We expect to have:
    //   1 listSize command
    //   1 Nop commands that switches the CQ from the dynamic to the static fetcher (no yielding yet)
    //   1 schedule_dma command and 1 work_distribute command for each Execute queue command
    //       We expect to have less than 8 engine activations in this test and so due to the engine ramp-up optimization
    //       the commands will be organized such all schedule_dma first followed by all work_dist commands.
    //       If this assumption changes and we will get more than 8 activations, the test shall be modified accordingly.
    //   1 nop command to pad up-to the end of the chunk

    struct eng_arc_cmd_sched_dma_t*         schedDmaCmd = nullptr;
    struct eng_arc_cmd_wd_fence_and_exec_t* wdCmd       = nullptr;
    struct eng_arc_cmd_nop_t*               nopCmd      = nullptr;

    nopCmd = (eng_arc_cmd_nop_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t));  // skip over the listSize
    ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);
    ASSERT_TRUE(nopCmd->switch_cq);  // switches to the static CQ
    ASSERT_EQ(nopCmd->padding, 0);
    ASSERT_LE(executeCount, WD_CTXT_COUNT);  // this test assumes less than 8 engine executions

    schedDmaCmd =
        (eng_arc_cmd_sched_dma_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) + sizeof(eng_arc_cmd_nop_t));

    for (unsigned i = 0; i < executeCount; i++)
    {
        ASSERT_EQ(schedDmaCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_SCHED_DMA);
        ASSERT_EQ(schedDmaCmd->yield, 1);
        schedDmaCmd++;
    }

    wdCmd = (eng_arc_cmd_wd_fence_and_exec_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) +
                                               sizeof(eng_arc_cmd_nop_t) +
                                               (executeCount * (sizeof(eng_arc_cmd_sched_dma_t))));

    for (unsigned i = 0; i < executeCount; i++)
    {
        ASSERT_EQ(wdCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_WD_FENCE_AND_EXE);
        ASSERT_EQ(wdCmd->wd_ctxt_id, i % WD_CTXT_COUNT);
        wdCmd++;
    }

    nopCmd = (eng_arc_cmd_nop_t*)wdCmd;
    ASSERT_EQ(nopCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_NOP);

    unsigned nopPadSize =
        (STATIC_COMPUTE_ECB_LIST_BUFF_SIZE - sizeof(eng_arc_cmd_list_size_t) - 2 * sizeof(eng_arc_cmd_nop_t) -
         executeCount * sizeof(eng_arc_cmd_sched_dma_t) - executeCount * sizeof(eng_arc_cmd_wd_fence_and_exec_t)) /
        DWORD_SIZE;

    ASSERT_EQ(nopCmd->padding, nopPadSize);
}

TEST_F(RecipeArcTest, sync_scheme_reset_procedure_validate_ecb_limit_is_5)
{
    static const unsigned SIGNAL_LIMIT = 5;
    ScopedConfigurationChange arcSupportMode("ARC_SYNC_SCHEME_SIGNAL_LIMIT", "36"); // 36 is 5 in the TPC domain
    ScopedConfigurationChange removeMemcpy("ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");

    Gaudi2GraphArcWorkDistTest g;

    const unsigned tensor_dim = 1;
    const TSize    size       = 1024 * 32 * 10;

    TensorPtr dmaI = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr tpcO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    synMemoryDescriptor memDesc(true);
    dmaI->setMemoryDescriptor(memDesc);
    tpcO->setMemoryDescriptor(memDesc);
    dmaI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    tpcO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    TensorPtr internal1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal3 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal4 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal5 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal6 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal7 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal8 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal9 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    // The Relu node depends on the DMA node and thus will have pipeline syncs
    NodePtr nDmaI = NodeFactory::createNode({dmaI}, {internal1}, nullptr, "memcpy", "node_name_memcpy1");
    NodePtr nTpc1 = NodeFactory::createNode({internal1}, {internal2}, nullptr, "relu_fwd_f32", "node_name_relu1");

    NodePtr nDma1 = NodeFactory::createNode({internal2}, {internal3}, nullptr, "memcpy", "node_name_memcpy2");
    NodePtr nTpc2 = NodeFactory::createNode({internal3}, {internal4}, nullptr, "relu_fwd_f32", "node_name_relu2");

    NodePtr nDma2 = NodeFactory::createNode({internal4}, {internal5}, nullptr, "memcpy", "node_name_memcpy3");
    NodePtr nTpc3 = NodeFactory::createNode({internal5}, {internal6}, nullptr, "relu_fwd_f32", "node_name_relu3");

    NodePtr nDma3 = NodeFactory::createNode({internal6}, {internal7}, nullptr, "memcpy", "node_name_memcpy4");
    NodePtr nTpc4 = NodeFactory::createNode({internal7}, {internal8}, nullptr, "relu_fwd_f32", "node_name_relu4");

    NodePtr nDma4 = NodeFactory::createNode({internal8}, {internal9}, nullptr, "memcpy", "node_name_memcpy5");
    NodePtr nTpcO = NodeFactory::createNode({internal9}, {tpcO}, nullptr, "relu_fwd_f32", "node_name_relu5");

    GraphEditor::addNode(g, nDmaI);
    GraphEditor::addNode(g, nTpc1);

    GraphEditor::addNode(g, nDma1);
    GraphEditor::addNode(g, nTpc2);

    GraphEditor::addNode(g, nDma2);
    GraphEditor::addNode(g, nTpc3);

    GraphEditor::addNode(g, nDma3);
    GraphEditor::addNode(g, nTpc4);

    GraphEditor::addNode(g, nDma4);
    GraphEditor::addNode(g, nTpcO);

    ASSERT_TRUE(g.compile()) << "graph compilation failed";

    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ecb_t pTpcDynamicEcb = {0};
    ecb_t pDmaDynamicEcb = {0};

    for (unsigned i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::TPC)
        {
            pTpcDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
        }
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::DMA)
        {
            pDmaDynamicEcb = recipe->arc_jobs[i].dynamic_ecb;
        }
    }

    ASSERT_TRUE(pTpcDynamicEcb.cmds_size > 0);
    ASSERT_TRUE(pDmaDynamicEcb.cmds_size > 0);

    struct eng_arc_cmd_reset_soset_t* resetCmd = nullptr;

    // Validate TPC
    resetCmd = (eng_arc_cmd_reset_soset_t*)(pTpcDynamicEcb.cmds
                    + sizeof(eng_arc_cmd_list_size_t)
                    + sizeof(eng_arc_cmd_nop_t)
                    + (sizeof(eng_arc_cmd_sched_dma_t) * 7)
                    + SIGNAL_LIMIT * (sizeof(eng_arc_cmd_sched_dma_t) + sizeof(eng_arc_cmd_wd_fence_and_exec_t)));

    ASSERT_EQ(resetCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_RESET_SOSET);
    ASSERT_TRUE(resetCmd->switch_cq);
    ASSERT_EQ(resetCmd->target, 36);  // TPC first signal is 32, so fifth signal is 36
    ASSERT_EQ(resetCmd->num_cmpt_engines, 29);  // 24 TPCs and 5 DMAs

    resetCmd = (eng_arc_cmd_reset_soset_t*)((char*)resetCmd
                    + sizeof(eng_arc_cmd_reset_soset_t)
                    + 3 * (sizeof(eng_arc_cmd_sched_dma_t) + sizeof(eng_arc_cmd_wd_fence_and_exec_t))
                    + 2 * sizeof(eng_arc_cmd_wd_fence_and_exec_t));

    ASSERT_EQ(resetCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_RESET_SOSET);
    ASSERT_TRUE(resetCmd->switch_cq);
    ASSERT_EQ(resetCmd->target, 36);  // TPC first signal is 32, so fifth signal is 36
    ASSERT_EQ(resetCmd->num_cmpt_engines, 29);  // 24 TPCs and 5 DMAs

    // Validate DMA
    resetCmd = (eng_arc_cmd_reset_soset_t*)(pDmaDynamicEcb.cmds
                    + sizeof(eng_arc_cmd_list_size_t)
                    + sizeof(eng_arc_cmd_nop_t)
                    + (sizeof(eng_arc_cmd_sched_dma_t) * 7)
                    + 3 * (sizeof(eng_arc_cmd_sched_dma_t) + sizeof(eng_arc_cmd_wd_fence_and_exec_t))
                    + sizeof(eng_arc_cmd_wd_fence_and_exec_t));

    ASSERT_EQ(resetCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_RESET_SOSET);
    ASSERT_TRUE(resetCmd->switch_cq);
    ASSERT_EQ(resetCmd->target, 4);  // DMA happened to be at signal 4 when the TPC signals the fifth time
    ASSERT_EQ(resetCmd->num_cmpt_engines, 29);  // 24 TPCs and 5 DMAs

    resetCmd = (eng_arc_cmd_reset_soset_t*)((char*)resetCmd
                    + sizeof(eng_arc_cmd_reset_soset_t)
                    + 4 * sizeof(eng_arc_cmd_wd_fence_and_exec_t));

    ASSERT_EQ(resetCmd->cmd_type, eng_arc_cmd_t::ECB_CMD_RESET_SOSET);
    ASSERT_TRUE(resetCmd->switch_cq);
    ASSERT_EQ(resetCmd->target, 4);  // DMA happened to be at signal 4 when the TPC signals the fifth time
    ASSERT_EQ(resetCmd->num_cmpt_engines, 29);  // 24 TPCs and 5 DMAs
}

TEST_F(RecipeArcTest, sync_scheme_reset_procedure_validate_ecb_limit_is_2)
{
    ScopedConfigurationChange arcSupportMode("ARC_SYNC_SCHEME_SIGNAL_LIMIT", "33"); // 33 is 2 in the TPC domain
    ScopedConfigurationChange removeMemcpy("ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");

    Gaudi2GraphArcWorkDistTest g;

    const unsigned tensor_dim = 1;
    const TSize    size       = 1024 * 32 * 10;

    TensorPtr dmaI = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr tpcO = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    synMemoryDescriptor memDesc(true);
    dmaI->setMemoryDescriptor(memDesc);
    tpcO->setMemoryDescriptor(memDesc);
    dmaI->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 5);
    tpcO->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 6);

    TensorPtr internal1 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal2 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal3 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal4 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal5 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal6 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal7 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal8 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));
    TensorPtr internal9 = TensorPtr(new Tensor(tensor_dim, &size, syn_type_single));

    // The Relu node depends on the DMA node and thus will have pipeline syncs
    NodePtr nDmaI = NodeFactory::createNode({dmaI}, {internal1}, nullptr, "memcpy", "node_name_memcpy1");
    NodePtr nTpc1 = NodeFactory::createNode({internal1}, {internal2}, nullptr, "relu_fwd_f32", "node_name_relu1");

    NodePtr nDma1 = NodeFactory::createNode({internal2}, {internal3}, nullptr, "memcpy", "node_name_memcpy2");
    NodePtr nTpc2 = NodeFactory::createNode({internal3}, {internal4}, nullptr, "relu_fwd_f32", "node_name_relu2");

    NodePtr nDma2 = NodeFactory::createNode({internal4}, {internal5}, nullptr, "memcpy", "node_name_memcpy3");
    NodePtr nTpc3 = NodeFactory::createNode({internal5}, {internal6}, nullptr, "relu_fwd_f32", "node_name_relu3");

    NodePtr nDma3 = NodeFactory::createNode({internal6}, {internal7}, nullptr, "memcpy", "node_name_memcpy4");
    NodePtr nTpc4 = NodeFactory::createNode({internal7}, {internal8}, nullptr, "relu_fwd_f32", "node_name_relu4");

    NodePtr nDma4 = NodeFactory::createNode({internal8}, {internal9}, nullptr, "memcpy", "node_name_memcpy5");
    NodePtr nTpcO = NodeFactory::createNode({internal9}, {tpcO}, nullptr, "relu_fwd_f32", "node_name_relu5");

    GraphEditor::addNode(g, nDmaI);
    GraphEditor::addNode(g, nTpc1);

    GraphEditor::addNode(g, nDma1);
    GraphEditor::addNode(g, nTpc2);

    GraphEditor::addNode(g, nDma2);
    GraphEditor::addNode(g, nTpc3);

    GraphEditor::addNode(g, nDma3);
    GraphEditor::addNode(g, nTpc4);

    GraphEditor::addNode(g, nDma4);
    GraphEditor::addNode(g, nTpcO);

    ASSERT_FALSE(g.compile()) << "graph compilation passed but it should fail due to violation of logical IDs";
}
