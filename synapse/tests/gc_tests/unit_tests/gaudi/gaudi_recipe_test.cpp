#include <memory>
#include <unordered_set>
#include <thread>
#include <vector>
#include "gaudi_recipe_test.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "platform/gaudi/graph_compiler/command_queue.h"
#include "platform/gaudi/graph_compiler/recipe_generator.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"
#include "recipe_allocator.h"

QueueCommand* GaudiRecipeTest::getWriteReg(unsigned valueCmd) const
{
    return new gaudi::WriteRegister(0x1000, valueCmd);
}

QueueCommand* GaudiRecipeTest::getMonArm() const
{
    return new gaudi::MonitorArm(0, 0, MONITOR_SO_OP_EQ, 0, Settable<uint8_t>());
}

CommandQueue*
GaudiRecipeTest::getTpcQueue(unsigned engineId, unsigned engineIndex, unsigned stream, bool sendSyncEvents) const
{
    return new gaudi::TpcQueue(engineId, engineIndex, sendSyncEvents, false);
}

// generator tests

TEST_F(GaudiRecipeTest, generator_basic)
{
    GaudiGraph g;
    // compile here and in other tests so a queue cmd factory will exist for NOP addition in addBlob
    g.compile();
    generator_basic(&g);
}

TEST_F(GaudiRecipeTest, generator_continuous_blobs)
{
    GaudiGraph g;
    g.compile();
    generator_continuous_blobs(&g);
}

TEST_F(GaudiRecipeTest, generator_single_blob)
{
    GaudiGraph g;
    g.compile();
    generator_single_blob(&g);
}

TEST_F(GaudiRecipeTest, generator_two_queues)
{
    GaudiGraph g;
    g.compile();
    generator_two_queues(&g);
}

TEST_F(GaudiRecipeTest, generator_activate_execute_jobs)
{
    GaudiGraph g;
    g.compile();
    generator_activate_execute_jobs(&g);
}

// blob tests

TEST_F(GaudiRecipeTest, blob_basic)
{
    GaudiGraph g;
    g.compile();
    blob_basic(&g);
}

TEST_F(GaudiRecipeTest, blob_container_basic)
{
    GaudiGraph g;
    g.compile();
    blob_container_basic(&g);
}

TEST_F(GaudiRecipeTest, blob_container_4_different_blobs)
{
    GaudiGraph g;
    g.compile();
    blob_container_4_different_blobs(&g);
}

TEST_F(GaudiRecipeTest, blob_container_blobs_chunks_test)
{
    GaudiGraph g;
    g.compile();
    blob_container_blobs_chunks_test(&g);
}

// patch point containter tests

TEST_F(GaudiRecipeTest, patch_point_container)
{
    patch_point_container();
}

// section type patch point containter tests

TEST_F(GaudiRecipeTest, section_type_patch_point_container)
{
    section_type_patch_point_container();
}

// hash system tests

TEST_F(GaudiRecipeTest, unique_hash_system_base_test)
{
    GaudiGraph g;
    g.compile();
    unique_hash_system_base_test(&g);
}

TEST_F(GaudiRecipeTest, tpc_kernel_dissasmbly_relu_add_relu)
{
    ScopedConfigurationChange enableProfiler("ENABLE_PROFILER", "true");

    GaudiGraph      g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");

    setDummyAddressAndSection(i1);
    setDummyAddressAndSection(o1);

    GraphEditor::addNode(g, n1);

    pTensor         i2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n2 = NodeFactory::createGenericTPCNode({i2, o1}, {o2}, nullptr, "add_fwd_f32", "");

    setDummyAddressAndSection(i2);
    setDummyAddressAndSection(o2);

    GraphEditor::addNode(g, n2);

    pTensor         o3 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n3 = NodeFactory::createGenericTPCNode({o2}, {o3}, nullptr, "relu_fwd_f32", "");

    setDummyAddressAndSection(o3);

    GraphEditor::addNode(g, n3);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);
    ASSERT_NE(recipe, nullptr);
    ASSERT_EQ(recipe->debug_profiler_info.num_nodes, 3);
    ASSERT_NE(recipe->debug_profiler_info.nodes[0].kernel_blob_index, RecipeGenerator::DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT);
    ASSERT_NE(recipe->debug_profiler_info.nodes[1].kernel_blob_index, RecipeGenerator::DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[0].kernel_blob_index,
              recipe->debug_profiler_info.nodes[2].kernel_blob_index);
    ASSERT_NE(recipe->debug_profiler_info.nodes[0].kernel_blob_index,
              recipe->debug_profiler_info.nodes[1].kernel_blob_index);
}

TEST_F(GaudiRecipeTest, tpc_kernel_dissasmbly_relu_add_relu_gemm)
{
    ScopedConfigurationChange enableProfiler("ENABLE_PROFILER", "true");

    GaudiGraph      g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;

    // Relu
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(i1);
    setDummyAddressAndSection(o1);
    GraphEditor::addNode(g, n1);

    // Add
    pTensor         i2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n2 = NodeFactory::createGenericTPCNode({i2, o1}, {o2}, nullptr, "add_fwd_f32", "");
    setDummyAddressAndSection(i2);
    setDummyAddressAndSection(o2);
    GraphEditor::addNode(g, n2);

    // Relu
    pTensor         o3 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n3 = NodeFactory::createGenericTPCNode({o2}, {o3}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(o3);
    GraphEditor::addNode(g, n3);

    // Gemm
    const TSize  sizes[] = {1,1};
    pTensor      o4 = pTensor(new Tensor(2, sizes, syn_type_float));
    pTensor      w4 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));

    w4->setAsWeights();
    w4->setAsStaticParam();

    pNode n4 = NodeFactory::createNode({o3, w4, nullptr, nullptr}, {o4}, nullptr, NodeFactory::gemmNodeTypeName, "");
    setDummyAddressAndSection(w4);
    setDummyAddressAndSection(o4);

    GraphEditor::addNode(g, n4);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);
    ASSERT_NE(recipe, nullptr);

    std::vector<uint32_t> blobIndexes;
    for (int i = 0; i < recipe->debug_profiler_info.num_nodes; i++)
    {
        if (strcmp(recipe->debug_profiler_info.nodes[i].operation, "DmaMemcpy") == 0 ||
            strcmp(recipe->debug_profiler_info.nodes[i].operation, "GEMM") == 0)
        {
            ASSERT_EQ(recipe->debug_profiler_info.nodes[i].kernel_blob_index,
                      RecipeGenerator::DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT);
        }
            // TPC nodes
        else
        {
            ASSERT_NE(recipe->debug_profiler_info.nodes[i].kernel_blob_index,
                      RecipeGenerator::DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT);
            blobIndexes.push_back(recipe->debug_profiler_info.nodes[i].kernel_blob_index);
        }
    }

    // The first and third operation are the same and should have the same blob id,
    // and it has to be different than the second
    ASSERT_EQ(blobIndexes[0], blobIndexes[2]);
    ASSERT_NE(blobIndexes[0], blobIndexes[1]);
}

TEST_F(GaudiRecipeTest, sync_scheme_info)
{
    GaudiGraph     g;
    const unsigned tensor_dim = 1;
    const TSize    size       = 1;

    // Relu
    pTensor i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode   n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(i1);
    setDummyAddressAndSection(o1);
    GraphEditor::addNode(g, n1);

    // Add
    pTensor i2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor o2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode   n2 = NodeFactory::createGenericTPCNode({i2, o1}, {o2}, nullptr, "add_fwd_f32", "");
    setDummyAddressAndSection(i2);
    setDummyAddressAndSection(o2);
    GraphEditor::addNode(g, n2);

    // Relu
    pTensor o3 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode   n3 = NodeFactory::createGenericTPCNode({o2}, {o3}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(o3);
    GraphEditor::addNode(g, n3);

    // Gemm
    const TSize sizes[] = {1, 1};
    pTensor     o4      = pTensor(new Tensor(2, sizes, syn_type_float));
    pTensor     w4      = pTensor(new Tensor(tensor_dim, &size, syn_type_float));

    w4->setAsWeights();
    w4->setAsStaticParam();

    pNode n4 = NodeFactory::createNode({o3, w4, nullptr, nullptr}, {o4}, nullptr, NodeFactory::gemmNodeTypeName, "");
    setDummyAddressAndSection(w4);
    setDummyAddressAndSection(o4);

    GraphEditor::addNode(g, n4);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);
    ASSERT_NE(recipe, nullptr);

    ASSERT_EQ(recipe->debug_sync_scheme_info.node_sync_info_nr, 6);

    unsigned tpcSigVal = 0, mmeSigVal = 0, dmaSigVal = 0;

    for (int i = 0; i < recipe->debug_sync_scheme_info.node_sync_info_nr; i++)
    {
        ASSERT_EQ(recipe->debug_sync_scheme_info.node_sync_info_legacy[i].pipe_level, 1);
        if (recipe->debug_sync_scheme_info.node_sync_info_legacy[i].engine_type == Recipe::TPC)
        {
            ASSERT_EQ(recipe->debug_sync_scheme_info.node_sync_info_legacy[i].sob_id, 40);
            tpcSigVal = recipe->debug_sync_scheme_info.node_sync_info_legacy[i].emitted_signal;
        }
        else if (recipe->debug_sync_scheme_info.node_sync_info_legacy[i].engine_type == Recipe::MME)
        {
            ASSERT_EQ(recipe->debug_sync_scheme_info.node_sync_info_legacy[i].sob_id, 56);
            mmeSigVal = recipe->debug_sync_scheme_info.node_sync_info_legacy[i].emitted_signal;
        }
        else if (recipe->debug_sync_scheme_info.node_sync_info_legacy[i].engine_type == Recipe::DMA)
        {
            ASSERT_EQ(recipe->debug_sync_scheme_info.node_sync_info_legacy[i].sob_id, 48);
            dmaSigVal = recipe->debug_sync_scheme_info.node_sync_info_legacy[i].emitted_signal;
        }
        else
        {
            assert("Got invalid engine type");
        }
    }

    ASSERT_EQ(tpcSigVal, 3);
    ASSERT_EQ(dmaSigVal, 2);
    ASSERT_EQ(mmeSigVal, 1);
}

TEST_F(GaudiRecipeTest, debug_profilier_context_id_test)
{
    ScopedConfigurationChange enableProfiler("ENABLE_PROFILER", "true");

    GaudiGraph      g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;

    // Relu
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(i1);
    setDummyAddressAndSection(o1);
    GraphEditor::addNode(g, n1);

    // Add
    pTensor         i2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n2 = NodeFactory::createGenericTPCNode({i2, o1}, {o2}, nullptr, "add_fwd_f32", "");
    setDummyAddressAndSection(i2);
    setDummyAddressAndSection(o2);
    GraphEditor::addNode(g, n2);

    // Relu
    pTensor         o3 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n3 = NodeFactory::createGenericTPCNode({o2}, {o3}, nullptr, "relu_fwd_f32", "");
    setDummyAddressAndSection(o3);
    GraphEditor::addNode(g, n3);

    // Gemm
    const TSize  sizes[] = {1,1};
    pTensor      o4 = pTensor(new Tensor(2, sizes, syn_type_float));
    pTensor      w4 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));

    w4->setAsWeights();
    w4->setAsStaticParam();

    pNode n4 = NodeFactory::createNode({o3, w4, nullptr, nullptr}, {o4}, nullptr, NodeFactory::gemmNodeTypeName, "");
    setDummyAddressAndSection(w4);
    setDummyAddressAndSection(o4);

    GraphEditor::addNode(g, n4);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);
    ASSERT_NE(recipe, nullptr);
    ASSERT_EQ(recipe->debug_profiler_info.num_nodes, 6);
    uint16_t recipeDebugId = recipe->debug_profiler_info.recipe_id;
    ASSERT_GE(recipeDebugId, 0x8000); // check MSB = 1
    ASSERT_EQ(recipe->debug_profiler_info.nodes[0].device_type, EDeviceType::DEVICE_TPC);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[0].full_context_id, 0x0);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[0].context_id, recipeDebugId);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[1].device_type, EDeviceType::DEVICE_TPC);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[1].full_context_id, 0x1);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[1].context_id, 0x1);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[2].device_type, EDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[2].full_context_id, 0x0);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[2].context_id, recipeDebugId);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[3].device_type, EDeviceType::DEVICE_TPC);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[3].full_context_id, 0x2);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[3].context_id, 0x2);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[4].device_type, EDeviceType::DEVICE_MME);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[4].full_context_id, 0x0);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[4].context_id, recipeDebugId);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[5].device_type, EDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[5].full_context_id, 0x1);
    ASSERT_EQ(recipe->debug_profiler_info.nodes[5].context_id, 0x1);

}

static void compileGraph(GaudiGraph *graph)
{
    graph->compile();
}

TEST_F(GaudiRecipeTest, debug_profilier_context_id_compile_parallel_test)
{
    const unsigned numOfGraphs = 500;

    ScopedConfigurationChange enableProfiler("ENABLE_PROFILER", "true");
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;

    std::vector<std::unique_ptr<GaudiGraph>> graphs;
    graphs.reserve(numOfGraphs);
    for (unsigned i = 0; i < numOfGraphs; i++)
    {
        graphs.push_back(std::make_unique<GaudiGraph>());
        GaudiGraph* g = graphs.back().get();

        // Relu
        pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
        pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
        pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");
        setDummyAddressAndSection(i1);
        setDummyAddressAndSection(o1);
        GraphEditor::addNode(*g, n1);

        // Add
        pTensor         i2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
        pTensor         o2 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
        pNode           n2 = NodeFactory::createGenericTPCNode({i2, o1}, {o2}, nullptr, "add_fwd_f32", "");
        setDummyAddressAndSection(i2);
        setDummyAddressAndSection(o2);
        GraphEditor::addNode(*g, n2);

        // Relu
        pTensor         o3 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
        pNode           n3 = NodeFactory::createGenericTPCNode({o2}, {o3}, nullptr, "relu_fwd_f32", "");
        setDummyAddressAndSection(o3);
        GraphEditor::addNode(*g, n3);

        // Gemm
        const TSize  sizes[] = {1,1};
        pTensor      o4 = pTensor(new Tensor(2, sizes, syn_type_float));
        pTensor      w4 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));

        w4->setAsWeights();
        w4->setAsStaticParam();

        pNode n4 = NodeFactory::createNode({o3, w4, nullptr, nullptr}, {o4}, nullptr, NodeFactory::gemmNodeTypeName, "");
        setDummyAddressAndSection(w4);
        setDummyAddressAndSection(o4);

        GraphEditor::addNode(*g, n4);
    }

    std::vector<std::thread> threadVector;
    threadVector.reserve(numOfGraphs);
    for (const auto& graphPtr : graphs)
    {
        // Create a new thread to trigger compilation
        threadVector.emplace_back(&compileGraph, graphPtr.get());
    }

    for (auto& th : threadVector)
    {
        th.join();
    }

    std::unordered_set<uint16_t> graphRecipeDebugIds;
    for (const auto& graphPtr : graphs)
    {
        RecipeAllocator recipeAlloc;
        recipe_t*       recipe = graphPtr->serializeDataPlane(&recipeAlloc);
        ASSERT_NE(recipe, nullptr);
        ASSERT_EQ(recipe->debug_profiler_info.num_nodes, 6);
        uint16_t recipeDebugId = recipe->debug_profiler_info.recipe_id;
        ASSERT_TRUE(graphRecipeDebugIds.find(recipeDebugId) == graphRecipeDebugIds.end());
        graphRecipeDebugIds.insert(recipeDebugId);
    }
}


void GaudiRecipeTestStagedSubmission::SetUp()
{
    hl_gcfg::setEnableExperimentalFlagsValue(true);
    GCFG_ENABLE_STAGED_SUBMISSION.setValue(true);
    GaudiRecipeTest::SetUp();
}

TEST_F(GaudiRecipeTestStagedSubmission, generator_node_exe)
{
    GaudiGraph g;
    g.compile();
    generator_node_exe(&g);
}