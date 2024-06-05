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
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/queue_command.h"
#include "platform/gaudi3/graph_compiler/gaudi3_eng_arc_hooks.h"
#include "platform/gaudi3/graph_compiler/gaudi3_eng_arc_command.h"
#include "compilation_hal_reader.h"

using namespace gaudi3;

class Gaudi3RecipeArcTest : public GraphOptimizerTest
{
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY, "false");
    }
};

class Gaudi3GraphArcWorkDistTest : public Gaudi3Graph
{
public:
    CommandQueuePtr getTpcQueue0() { return m_codeGenerator->getTpcDispatcher()->getQueue(0); }
    std::unordered_map<NodePtr, gaudi3::TpcDescriptorsWrappers>& getTpcDescs() { return m_tpcNodesDescriptorsWrappers; }
    std::list<EngArcCmdPtr>& getCmeCommands() { return m_codeGenerator->getCmeCommands(); }
    McidConverter& getMcidConverter() { return m_codeGenerator->getMcidConverter(); }
    void regenerateProgram()
    {
        m_codeGenerator->getCommandQueueByIdForTesting().clear();
        m_codeGenerator->generate(this);
        m_codeGenerator->initQueues();
        CompilationHalReaderSetter compHalReaderSetter(this);
        m_codeGenerator->fillQueues();
        m_codeGenerator->generateRecipes(*this);
    }
    void regenerateRecipe()
    {
        m_codeGenerator->generateRecipes(*this);
    }
};

TEST_F(Gaudi3RecipeArcTest, program_with_dcore_locality)
{
    Gaudi3GraphArcWorkDistTest g;

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

    // Inflate the number of contexts to 4 by duplicating the one that exists
    for (auto& descWraps : g.getTpcDescs())
    {
        for (auto& descWrap : descWraps.second)
        {
            descWrap.addFwCtx(descWrap.getFwCtx());
            descWrap.addFwCtx(descWrap.getFwCtx());
            descWrap.addFwCtx(descWrap.getFwCtx());
        }
    }

    g.regenerateProgram();

    // verify ArcExeWdTpc has 4 contexts
    const CommandQueuePtr& queue = g.getTpcQueue0();
    ArcExeWdTpc*           exeWdTpcCmd = nullptr;
    for (auto& cmd : queue->getCommands(false))
    {
        if ((exeWdTpcCmd = dynamic_cast<ArcExeWdTpc*>(cmd.get())) != nullptr)
        {
            ASSERT_EQ(exeWdTpcCmd->getNumCtxs(), 4);
        }
    }

    // verify schedule dma ECB command uses dcore index
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

    struct eng_arc_cmd_sched_dma_t* schedDmaCmd =
        (eng_arc_cmd_sched_dma_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) + sizeof(eng_arc_cmd_nop_t));

    ASSERT_EQ(schedDmaCmd->wd_type, TPC_WD_USING_DCORE_INDEX); // check only the first sched_dma command
}

TEST_F(Gaudi3RecipeArcTest, cme_commands_recipe_serialization)
{
    unsigned converterSafetyFactor = 2;
    Gaudi3GraphArcWorkDistTest g;
    ASSERT_EQ(g.compile(), true);

    std::list<EngArcCmdPtr>& cmeCmds = g.getCmeCommands();

    // We start with room for 64 DWORDs
    EngArcCmdPtr cmd1 = std::make_shared<Gaudi3CmeNopEngArcCommand>();
    cmeCmds.push_back(cmd1); // will leave us with 63 DWORDs

    EngArcCmdPtr cmd2 = std::make_shared<Gaudi3CmeNopEngArcCommand>(61);
    cmeCmds.push_back(cmd2); // will leave us with 1 DWORD

    EngArcCmdPtr cmd3 = std::make_shared<Gaudi3ResetSobsArcCommand>(1, 1, 1,true, true);
    cmeCmds.push_back(cmd3); // ResetSob requires 2 DWORDs which means we will insert a nop and open a new chunk

    uint16_t dummy1;
    unsigned dummy2;
    g.getMcidConverter().convertDegrade(77, dummy1);
    g.getMcidConverter().convertDiscard(88, dummy1, dummy2);

    g.regenerateRecipe();

    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);

    ASSERT_EQ(recipe->max_used_mcid_degrade, 77 + converterSafetyFactor);
    ASSERT_EQ(recipe->max_used_mcid_discard, 88 + converterSafetyFactor);

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

    struct cme_arc_cmd_nop_t* cmeNopCmd = (cme_arc_cmd_nop_t*)(pDynamicEcb.cmds);
    ASSERT_EQ(cmeNopCmd->cmd_type, CME_ECBL_CMD_NOP);
    cmeNopCmd = (cme_arc_cmd_nop_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t));
    ASSERT_EQ(cmeNopCmd->cmd_type, CME_ECBL_CMD_NOP);
    ASSERT_EQ(cmeNopCmd->padding, 61);
    cmeNopCmd = (cme_arc_cmd_nop_t*)(pDynamicEcb.cmds + sizeof(eng_arc_cmd_list_size_t) * 2 + 61 * sizeof(uint32_t));
    ASSERT_EQ(cmeNopCmd->cmd_type, CME_ECBL_CMD_NOP);
    ASSERT_EQ(cmeNopCmd->padding, 0);
    struct eng_arc_cmd_reset_soset_t* resetSobCmd = (eng_arc_cmd_reset_soset_t*)(pDynamicEcb.cmds + 256);
    ASSERT_EQ(resetSobCmd->cmd_type, ECB_CMD_RESET_SOSET);
    cmeNopCmd = (cme_arc_cmd_nop_t*)(pDynamicEcb.cmds + 256 + sizeof(eng_arc_cmd_reset_soset_t));
    ASSERT_EQ(cmeNopCmd->cmd_type, CME_ECBL_CMD_NOP);
    ASSERT_EQ(cmeNopCmd->padding, 61);
}

TEST_F(Gaudi3RecipeArcTest, cme_degrade_discard_commands_recipe_serialization)
{
    Gaudi3GraphArcWorkDistTest g;
    ASSERT_EQ(g.compile(), true);

    std::list<EngArcCmdPtr>& cmeCmds = g.getCmeCommands();

    ASSERT_EQ(cmeCmds.size(), 0);

    DependencyMap deps;

    deps[gaudi3::DEVICE_TPC_LOGICAL_QUEUE] = 185;
    deps[gaudi3::DEVICE_MME_LOGICAL_QUEUE] = 57;

    // Degrade
    EngArcCmdPtr cmd1 = std::make_shared<Gaudi3CmeDegradeArcCommand>(deps, 77);
    cmeCmds.push_back(cmd1);

    // Discard
    deps[gaudi3::DEVICE_XPS_LOGICAL_QUEUE] = 278;

    EngArcCmdPtr cmd2 = std::make_shared<Gaudi3CmeDiscardArcCommand>(deps, 85);
    cmeCmds.push_back(cmd2);

    // Discard --> Degrade
    deps[gaudi3::DEVICE_ROT_LOGICAL_QUEUE] = 19;

    EngArcCmdPtr cmd3 = std::make_shared<Gaudi3CmeDegradeArcCommand>(deps, 117, 1);
    cmeCmds.push_back(cmd3);

    g.regenerateRecipe();

    ASSERT_EQ(g.getCmeCommands().size(), 3);

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

    // DEGRADE
    struct cme_arc_cmd_degrade_cls_t* cmeDegradeCmd1 = (cme_arc_cmd_degrade_cls_t*)(pDynamicEcb.cmds);
    ASSERT_EQ(cmeDegradeCmd1->cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
    ASSERT_EQ(cmeDegradeCmd1->mcid_offset, 77);
    ASSERT_EQ(cmeDegradeCmd1->use_discard_base, 0);
    ASSERT_EQ(cmeDegradeCmd1->tpc.threshold_v2, 185);
    ASSERT_EQ(cmeDegradeCmd1->mme.threshold_v2, 57);
    ASSERT_EQ(cmeDegradeCmd1->target_bitmap, 3);

    // DISCARD
    struct cme_arc_cmd_discard_cls_t* cmeDiscardCmd = (cme_arc_cmd_discard_cls_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_degrade_cls_t));
    ASSERT_EQ(cmeDiscardCmd->cmd_type, CME_ECBL_CMD_DISCARD_CLS);
    ASSERT_EQ(cmeDiscardCmd->mcid_offset, 85);
    ASSERT_EQ(cmeDiscardCmd->tpc.threshold_v2, 185);
    ASSERT_EQ(cmeDiscardCmd->mme.threshold_v2, 57);
    ASSERT_EQ(cmeDiscardCmd->mme_xpose.threshold_v2, 278);
    ASSERT_EQ(cmeDiscardCmd->target_bitmap, 7);

    // DISCARD --> DEGRADE
    struct cme_arc_cmd_degrade_cls_t* cmeDegradeCmd2 = (cme_arc_cmd_degrade_cls_t*)(pDynamicEcb.cmds + sizeof(cme_arc_cmd_degrade_cls_t) + sizeof(cme_arc_cmd_discard_cls_t));
    ASSERT_EQ(cmeDegradeCmd2->cmd_type, CME_ECBL_CMD_DEGRADE_CLS);
    ASSERT_EQ(cmeDegradeCmd2->mcid_offset, 117);
    ASSERT_EQ(cmeDegradeCmd2->use_discard_base, 1);
    ASSERT_EQ(cmeDegradeCmd2->tpc.threshold_v2, 185);
    ASSERT_EQ(cmeDegradeCmd2->mme.threshold_v2, 57);
    ASSERT_EQ(cmeDegradeCmd2->mme_xpose.threshold_v2, 278);
    ASSERT_EQ(cmeDegradeCmd2->rot.threshold_v2, 19);
    ASSERT_EQ(cmeDegradeCmd2->target_bitmap, 15);
}