#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include "gaudi_recipe_test.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "gaudi_code_generator.h"
#include "platform/gaudi/graph_compiler/queue_command.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "queue_dispatcher.h"
#include "params_file_manager.h"
#include "graph_optimizer_test.h"
#include "recipe_allocator.h"
#include "scoped_configuration_change.h"
#include "gaudi_code_generator.h"

using namespace std;

class TestGaudiGraph : public GaudiGraph // expose members for testing
{
public:
    const QueueDispatcherPtr& getTpcDispatcher() { return m_codeGenerator->getTpcDispatcher(); }
    const QueueDispatcherMap& getDmaDispatchers() { return m_codeGenerator->getDmaDispatchers(); }
    const QueueDispatcherPtr& getMmeDispatcher() { return m_codeGenerator->getMmeDispatcher(); }
    const CommandQueuePtr&    getCompletionQueue() { return m_codeGenerator->getCompletionQueue(); }

    uint64_t getPredTableDeviceAddr()
    {
        return (static_cast<const GaudiCodeGenerator&>(*m_codeGenerator.get())).getPredicateTables().at(0).deviceAddr;
    }

    char* getPredTableHostAddr()
    {
        return (static_cast<const GaudiCodeGenerator&>(*m_codeGenerator.get()))
            .getPredicateTables()
            .at(0)
            .hostAddr.get();
    }
};

TEST_F(GaudiRecipeTest, verify_patching_points_tpc)
{
    GaudiGraph      g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i1->setDramOffset(0x1000);
    o1->setDramOffset(0x3000);

    synMemoryDescriptor memDescPersist(true);
    i1->setMemoryDescriptor(memDescPersist);
    i1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o1->setMemoryDescriptor(memDescPersist);
    o1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    GraphEditor::addNode(g, n1);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = g.serializeDataPlane(&recipeAlloc);
    ASSERT_NE(recipe, nullptr);

    const auto exeSortedNodes           = g.getExeSortedNodes();
    const auto isProducingTpcDescriptor = [&g](const NodePtr& node) {
        return !downcaster<GaudiCodeGenerator>(g.getCodeGenerator().get())->getTPCNodeDescriptorsWrappers(node).empty();
    };

    const unsigned numTpcNodes = std::count_if(exeSortedNodes.begin(), exeSortedNodes.end(), isProducingTpcDescriptor);
    ASSERT_EQ(numTpcNodes, 1) << "Expected 1 tpc node, but found " << numTpcNodes;

    const auto tpcDescriptorProducerItr =
        std::find_if(exeSortedNodes.begin(), exeSortedNodes.end(), isProducingTpcDescriptor);
    const gaudi::TPCDescriptorsWrappers descWrap = downcaster<GaudiCodeGenerator>(g.getCodeGenerator().get())
                                                       ->getTPCNodeDescriptorsWrappers(*tpcDescriptorProducerItr);
    ASSERT_EQ(descWrap.size(), 1) << "Expected 1 tpc descriptor, but found " << descWrap.size();

    const BasicFieldsContainerInfo& afci = descWrap[0].getBasicFieldsContainerInfo();
    const AddressFieldInfoSet&      afis = afci.retrieveAddressFieldInfoSet();

    // Tpc descriptor has 6 address fields: input & output tensors low and+high, and kernel base address low+high.
    ASSERT_EQ(afis.size(), 6) << "Expected 6 address fields, but found " << afis.size();

    // We expect 3 patch-points for the above 6 afis (the low+high are optimized to a joined FULL patch-point)
    // We also load predicate for each TPC engine, so we expect additional 8 patch-point (of FULL).
    // !!!  If we will do TPC i-cache prefetch, each one of the 8 TPC engines will be initialized with the kenels base address
    // !!!  low+high (not joint to full) so we will have 16 more patch-points, currently, we don't prefetch.
    ASSERT_EQ(recipe->patch_points_nr, 11);
}

TEST_F(GaudiRecipeTest, verify_patching_points_dma)
{
    static const TSize c_parallelLevel = 2;
    static const TSize c_numRois = 4;

    GaudiGraph      g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 16*1024*c_parallelLevel*c_numRois/sizeof(uint32_t); // 16KB X parallelLevel X num rois / sizeof element
    pTensor         i = pTensor(new Tensor(tensor_dim, &size, syn_type_single));
    pTensor         o = pTensor(new Tensor(tensor_dim, &size, syn_type_single));
    pNode           n = NodeFactory::createNode({i}, {o}, nullptr, NodeFactory::dmaMemcpyNodeTypeName, "N1");
    std::dynamic_pointer_cast<DMANode>(n)->setParallelLevel(c_parallelLevel);

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i->setDramOffset(0x1000);
    o->setDramOffset(0x1001 + size * i->getElementSizeInBytes()); // Non-overlapping offset.

    synMemoryDescriptor memDescPersist(true);
    i->setMemoryDescriptor(memDescPersist);
    i->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o->setMemoryDescriptor(memDescPersist);
    o->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    GraphEditor::addNode(g, n);

    ASSERT_TRUE(g.compile()) << "failed to compile graph";
    //recipe_t* recipe = g.serialize();

    gaudi::DMADescriptorsWrappers descWrap = downcaster<GaudiCodeGenerator>(g.getCodeGenerator().get())->getDMANodeDescriptorsWrappers(n);
    ASSERT_EQ(descWrap.size(), c_parallelLevel * c_numRois);

    for (auto dw : descWrap)
    {
        const BasicFieldsContainerInfo& afci = dw.getBasicFieldsContainerInfo();
        const AddressFieldInfoSet&      afis = afci.retrieveAddressFieldInfoSet();

        ASSERT_EQ(afis.size(), 4); // 4 address fields in DMA descriptor: src_lo, src_hi, dst_lo, dst_hi

        for (AddressFieldInfoPair afip : afis)
        {
            ASSERT_EQ(afip.second->getSectionName(), WORKSPACE_MEMORY_SECTION_NAME);
            ASSERT_EQ(afip.second->getMemorySectionId(), MEMORY_ID_RESERVED_FOR_WORKSPACE);
        }
    }
}

TEST_F(GraphOptimizerTest, non_upgradeable_tpc_node)
{
    static const TSize c_parallelLevel = 2;
    static const TSize c_numRois = 4;

    GaudiGraph  g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 16*1024*c_parallelLevel*c_numRois/sizeof(uint32_t); // 16KB X parallelLevel X num rois / sizeof element
    pTensor         i = pTensor(new Tensor(tensor_dim, &size, syn_type_single));
    pTensor         o = pTensor(new Tensor(tensor_dim, &size, syn_type_single));
    pNode           n = NodeFactory::createNode({i}, {o}, nullptr, "add_fwd_i128", "non_upgradeable_node");

    // this test is expected to fail as add_fwd_i128 do not exist in either perf_lib or complexGuid
    ASSERT_EQ(GraphEditor::addNode(g, n), false) << "adding node to graph should fail";
}

TEST_F(GraphOptimizerTest, atomic_nodes_validation)
{
    GaudiGraph g;

    pTensor inputA = std::make_shared<Tensor>(syn_type_bf16);
    pTensor inputB = std::make_shared<Tensor>(syn_type_bf16);

    pNode n1 = NodeFactory::createNode({}, {inputA}, nullptr, NOP_KERNEL_NAME, "test_node1");
    pNode n2 = NodeFactory::createNode({inputA}, {inputB}, nullptr, NOP_KERNEL_NAME, "test_node2");
    pNode n3 = NodeFactory::createNode({inputB}, {}, nullptr, NOP_KERNEL_NAME, "test_node3");

    GraphEditor::addNode(g, n1);
    GraphEditor::addNode(g, n2);
    GraphEditor::addNode(g, n3);

    g.getGraphAnnotation().addAtomicNodesPair(n1, n2);
    ASSERT_TRUE(validateAtomicNodes(g));

    g.getGraphAnnotation().addAtomicNodesPair(n2, n3);
    ASSERT_TRUE(validateAtomicNodes(g));

    g.getGraphAnnotation().addAtomicNodesPair(n1, n3);
    ASSERT_FALSE(validateAtomicNodes(g));
}

TEST_F(GraphOptimizerTest, emplace_input_tensor)
{
    pTensor inputA = std::make_shared<Tensor>(syn_type_bf16);
    inputA->setName("A", true);
    pTensor inputB = std::make_shared<Tensor>(syn_type_bf16);
    inputB->setName("B", true);
    pTensor inputC = std::make_shared<Tensor>(syn_type_bf16);
    inputC->setName("C", true);
    pTensor emplaceInput = std::make_shared<Tensor>(syn_type_bf16);
    emplaceInput->setName("R", true);

    pNode n = NodeFactory::createNode({inputA, inputB, inputC}, {}, nullptr, NOP_KERNEL_NAME, "test_node");
    n->emplaceInput(1, emplaceInput);
    auto inputs =  n->getInputs();

    ASSERT_EQ(inputs.size(), 4);
    ASSERT_EQ(inputs[0]->getName(), "A");
    ASSERT_EQ(inputs[1]->getName(), "R");
    ASSERT_EQ(inputs[2]->getName(), "B");
    ASSERT_EQ(inputs[3]->getName(), "C");
}

TEST_F(GraphOptimizerTest, emplace_output_tensor)
{
    pTensor outputA = std::make_shared<Tensor>(syn_type_bf16);
    outputA->setName("A", true);
    pTensor outputB = std::make_shared<Tensor>(syn_type_bf16);
    outputB->setName("B", true);
    pTensor outputC = std::make_shared<Tensor>(syn_type_bf16);
    outputC->setName("C", true);
    pTensor emplaceOutput = std::make_shared<Tensor>(syn_type_bf16);
    emplaceOutput->setName("R", true);

    pNode n = NodeFactory::createNode({}, {outputA, outputB, outputC}, nullptr, NOP_KERNEL_NAME, "test_node");
    n->emplaceOutput(1, emplaceOutput);
    auto outputs =  n->getOutputs();

    ASSERT_EQ(outputs.size(), 4);
    ASSERT_EQ(outputs[0]->getName(), "A");
    ASSERT_EQ(outputs[1]->getName(), "R");
    ASSERT_EQ(outputs[2]->getName(), "B");
    ASSERT_EQ(outputs[3]->getName(), "C");
}

TEST_F(GraphOptimizerTest, load_tpc_predicates)
{
    TestGaudiGraph  g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 1;
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i1->setDramOffset(0x1000);
    o1->setDramOffset(0x3000);

    synMemoryDescriptor memDescPersist(true);
    i1->setMemoryDescriptor(memDescPersist);
    i1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o1->setMemoryDescriptor(memDescPersist);
    o1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);

    GraphEditor::addNode(g, n1);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";

    QueueDispatcherPtr tpcDispatcher = g.getTpcDispatcher();
    for (unsigned i = 0; i < tpcDispatcher->getNumEngines(); ++i)
    {
        const CommandQueuePtr& queue = tpcDispatcher->getQueue(i);
        const std::vector<QueueCommandPtr>& cmds = queue->getCommands(false); // get the Exe part
        bool found = false;
        gaudi::LoadPredicates* loadPredCmd = nullptr;
        for (auto& cmd : cmds)
        {
            loadPredCmd = dynamic_cast<gaudi::LoadPredicates*>(cmd.get());
            if (loadPredCmd != nullptr)
            {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << "missing LoadPredicates command";
        uint64_t  cmdHbmAddr = loadPredCmd->getSrcAddrForTesting();
        uint64_t  offsetInTable = cmdHbmAddr - g.getPredTableDeviceAddr();
        uint32_t* predLineInHost = (uint32_t*)(g.getPredTableHostAddr() + offsetInTable);
        for (unsigned i = 0; i <= g.getHALReader()->getNumPredicateBits(); ++i)
        {
            // predicate 0 is reserved, predicate 1 is for engine 0, predicate 2 is for engine 1, and so on
            unsigned expectedVal = (i == queue->GetEngineIndex()+1)? 1 : 0;
            ASSERT_EQ(predLineInHost[i], expectedVal);
        }
    }
}

TEST_F(GraphOptimizerTest, duplicate_and_predicate_tpc_descriptor)
{
    static const unsigned NUM_CHECKED_PRED = 9;

    ScopedConfigurationChange CompressBlobsMode("COMPRESS_BLOBS", "true");
    ScopedConfigurationChange MaxAvailableTpcMode("TPC_ENGINES_ENABLED_MASK", "0xf3");  // turn off TPC2 and TPC3

    TestGaudiGraph  g;
    const unsigned  tensor_dim = 1;
    const TSize     size = 512*512;
    pTensor         i1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pTensor         o1 = pTensor(new Tensor(tensor_dim, &size, syn_type_float));
    pNode           n1 = NodeFactory::createGenericTPCNode({i1}, {o1}, nullptr, "relu_fwd_f32", "");

    // set some boguse addresses to the tensors and allocate host memory so we won't assert
    i1->setDramOffset(0x1000);
    o1->setDramOffset(0x3000);

    synMemoryDescriptor memDescPersist(true);
    i1->setMemoryDescriptor(memDescPersist);
    i1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    o1->setMemoryDescriptor(memDescPersist);
    o1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    GraphEditor::addNode(g, n1);
    ASSERT_TRUE(g.compile()) << "failed to compile graph";

    QueueDispatcherPtr tpcDispatcher = g.getTpcDispatcher();
    for (unsigned i = 0; i < tpcDispatcher->getNumEngines(); ++i)
    {
        const CommandQueuePtr& queue = tpcDispatcher->getQueue(i);
        const std::vector<QueueCommandPtr>& cmds = queue->getCommands(false); // get the Exe part
        bool found[NUM_CHECKED_PRED] = {0};
        for (auto& cmd : cmds)
        {
            gaudi::WriteManyRegisters* wmrCmd = dynamic_cast<gaudi::WriteManyRegisters*>(cmd.get());
            if (!wmrCmd) continue; // check only WriteManyRegisters

            unsigned binSize = cmd->GetBinarySize();
            char* buf = new char[binSize];
            cmd->writeInstruction(buf);
            unsigned pred = buf[4] & 0x1F; // extract the predicate value from the command
            ASSERT_LE(pred, 8);
            found[pred] = true;
            delete[] buf;
        }
        for (unsigned i = 0; i < NUM_CHECKED_PRED; ++i)
        {
            // TPC2,3 are disabled so predicate 3,4 should not appear in the program
            ASSERT_EQ(found[i], (i != 3 && i != 4) ? true : false);
        }
    }
}

TEST_F(GraphOptimizerTest, compilation_of_gaudi_graph_with_only_one_node_which_is_logical_operation)
{
    TestGaudiGraph graph;

    TSize    sizes[]          = {4, 4, 1, 1};
    TSize    reshaped_sizes[] = {16, 1, 1, 1};
    unsigned dims             = 4;

    TensorPtr t1          = std::make_shared<Tensor>(dims, sizes, syn_type_int32);
    TensorPtr t1_reshaped = std::make_shared<Tensor>(dims, reshaped_sizes, syn_type_int32);

    NodePtr reshape = NodeFactory::createNode({t1}, {t1_reshaped}, nullptr, 0, NodeFactory::reshapeNodeTypeName, "n1");

    synMemoryDescriptor memDesc(true);  // persistent

    t1->setMemoryDescriptor(memDesc);
    t1->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);

    ASSERT_TRUE(GraphEditor::addNode(graph, reshape)) << "Failed to add reshape node to graph";

    ASSERT_TRUE(graph.compile()) << "Failed to compile graph";

    // ensure graph has only one node and that node is logical operation
    ASSERT_EQ(graph.getExeSortedNodes().size(), 1);
    ASSERT_TRUE(graph.getExeSortedNodes().back()->isLogicalOperation());

    // ensure all queues are filled with at least 1 command
    auto& mmeDispatcher = graph.getMmeDispatcher();
    for (unsigned i = 0; i < mmeDispatcher->getNumEngines(); ++i)
    {
        ASSERT_GE(mmeDispatcher->getQueue(i)->getCommands(false).size(), 1);
    }
    auto& tpcDispatcher = graph.getTpcDispatcher();
    for (unsigned i = 0; i < tpcDispatcher->getNumEngines(); ++i)
    {
        ASSERT_GE(tpcDispatcher->getQueue(i)->getCommands(false).size(), 1);
    }
    for (const std::pair<const QueueDispatcherParams, QueueDispatcherPtr>& dispatcher : graph.getDmaDispatchers())
    {
        for (unsigned i = 0; i < dispatcher.second->getNumEngines(); ++i)
        {
            ASSERT_GE(dispatcher.second->getQueue(i)->getCommands(false).size(), 1);
        }
    }
    ASSERT_GE(graph.getCompletionQueue()->getCommands(false).size(), 1);

    // ensure the recipe contains jobs for all engines
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = graph.serializeDataPlane(&recipeAlloc);
    ASSERT_EQ(recipe->execute_jobs_nr, 16);  // 8 TPC, 2, MME, 5 DMA, 1 Completion
}

TEST_F(GraphOptimizerTest, compilation_of_empty_gaudi_graph)
{
    TestGaudiGraph graph;

    ASSERT_TRUE(graph.compile()) << "Failed to compile graph";

    // ensure graph is empty
    ASSERT_EQ(graph.getExeSortedNodes().size(), 0);

    // ensure all queues are filled with at least 1 command
    auto& mmeDispatcher = graph.getMmeDispatcher();
    for (unsigned i = 0; i < mmeDispatcher->getNumEngines(); ++i)
    {
        ASSERT_GE(mmeDispatcher->getQueue(i)->getCommands(false).size(), 1);
    }
    auto& tpcDispatcher = graph.getTpcDispatcher();
    for (unsigned i = 0; i < tpcDispatcher->getNumEngines(); ++i)
    {
        ASSERT_GE(tpcDispatcher->getQueue(i)->getCommands(false).size(), 1);
    }
    for (const std::pair<const QueueDispatcherParams, QueueDispatcherPtr>& dispatcher : graph.getDmaDispatchers())
    {
        for (unsigned i = 0; i < dispatcher.second->getNumEngines(); ++i)
        {
            ASSERT_GE(dispatcher.second->getQueue(i)->getCommands(false).size(), 1);
        }
    }
    ASSERT_GE(graph.getCompletionQueue()->getCommands(false).size(), 1);

    // ensure the recipe contains jobs for all engines
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = graph.serializeDataPlane(&recipeAlloc);
    ASSERT_EQ(recipe->execute_jobs_nr, 16);  // 8 TPC, 2, MME, 5 DMA, 1 Completion
}
