#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "global_conf_test_setter.h"
#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "graph_visualization.h"
#include "node_factory.h"
#include "synapse_common_types.hpp"

// Currently input reuse suggestions should be applied only in cases when the suggestion does not modify the graph (other than
// setting the aliases).
// To comply with this requirement, the pass runs after GRAPH_MUTATIONS_GROUP.
// This test suite checks that the inplace is applied/not applied when necessary.
class InplaceInuptReuseSuggestionTest : public GraphOptimizerTest
{
public:
    TensorPtr
    createPersistentTensor(unsigned dim, const SizeArray& sizes, synDataType type, const std::string name = "")
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        const auto          t = createTensor(dim, sizes, type, name);
        t->setMemorySectionID(getNextMemSectionId());
        t->setMemoryDescriptor(persistentMemoryDesc);
        return t;
    }
    TensorPtr createTensor(unsigned dim, const SizeArray& sizes, synDataType type, const std::string& name = "")
    {
        TensorPtr t = std::make_shared<Tensor>(dim, sizes.data(), type);
        t->setName(name);
        return t;
    }
    unsigned getNextMemSectionId() { return m_sectionId++; }
protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS, "true");
    }
    unsigned         m_sectionId        = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR;
};

TEST_F(InplaceInuptReuseSuggestionTest, basic_inplace_suggestion)
{
    setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
    setGlobalConfForTest(GCFG_RUN_TPC_FUSER, "false");

    GaudiGraph g;
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto addIn1 = createPersistentTensor(dim, sizes, syn_type_float, "addIn1");
    auto addIn2 = createPersistentTensor(dim, sizes, syn_type_float, "addIn2");
    auto addOut = createTensor(dim, sizes, syn_type_float, "addOut");
    NodePtr add = NodeFactory::createNode({addIn1,addIn2}, {addOut}, nullptr, "add_fwd_f32", "add");
    GraphEditor::addNode(g, add);

    auto reluOut = createTensor(dim, sizes, syn_type_float, "reluOut");
    NodePtr relu = NodeFactory::createNode({addOut}, {reluOut}, nullptr, "relu_fwd_f32", "relu");
    GraphEditor::addNode(g, relu);

    auto           splitOut1   = createPersistentTensor(dim, sizes, syn_type_float, "splitOut1");
    auto           splitOut2   = createPersistentTensor(dim, sizes, syn_type_float, "splitOut2");
    synSplitParams splitParams = {1};
    NodePtr        split = NodeFactory::createNode({reluOut}, {splitOut1, splitOut2}, &splitParams, "split", "split");
    GraphEditor::addNode(g, split);

    graphVisualizationPre(g);
    g.compile();
    graphVisualizationPost(g);

    ASSERT_TRUE(!addOut->isAliasedTensor()) << "reluIn is persistent so inplace reuse not allowed";
    ASSERT_TRUE(reluOut->isAliasedTensor()) << "reluOut2 reuse relu2's input";
    ASSERT_TRUE(reluOut->getAliasTensor() == addOut) << "reluOut2 reuse relu2's input";
}

TEST_F(InplaceInuptReuseSuggestionTest, block_reuse_of_static_tensor)
{
    GaudiGraph g;
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut1 = createTensor(dim, sizes, syn_type_float, "reluOut1");
    NodePtr relu1 = NodeFactory::createNode({reluIn}, {reluOut1}, nullptr, "relu_fwd_f32", "relu1");
    reluIn->setAsStaticParam();
    GraphEditor::addNode(g, relu1);

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    ASSERT_TRUE(!reluOut1->isAliasedTensor()) << "reluOut1 shouldn't reuse reluIn because it is a static tensor";
}

TEST_F(InplaceInuptReuseSuggestionTest, inplace_reuse_changes_location_of_tensor)
{
    GaudiGraph g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut1 = createTensor(dim, sizes, syn_type_float, "reluOut1");
    reluOut1->setTensorInSram();
    NodePtr relu1 = NodeFactory::createNode({reluIn}, {reluOut1}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, relu1);

    auto reluOut2 = createTensor(dim, sizes, syn_type_float, "reluOut2");
    reluOut2->setTensorInDram();
    NodePtr relu2 = NodeFactory::createNode({reluOut1}, {reluOut2}, nullptr, "relu_fwd_f32", "relu2");
    GraphEditor::addNode(g, relu2);

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    ASSERT_TRUE(!reluOut1->isAliasedTensor()) << "reluIn and reluOut1 are in different locations so reuse is not allowed";
    ASSERT_TRUE(!reluOut2->isAliasedTensor()) << "reluOut1 and reluOut2 are in different locations so reuse is not allowed";
}

TEST_F(InplaceInuptReuseSuggestionTest, inplace_reuse_discarded_tensors)
{
    // [reluIn1]->relu1->[reluOut1]->reshape->[reluIn2]->relu2->[reluOut2]
    // relu2 discards reluIn2. reluIn2 is an alias of reluOut1, so it is discarded as well.

    GaudiGraph                 g;
    CompilationHalReaderSetter compHalReaderSetter(&g);
    SizeArray                  sizes = {40, 40, 40, 40, 1};
    unsigned                   dim   = 4;

    auto    reluIn1  = createTensor(dim, sizes, syn_type_float, "reluIn1");
    auto    reluOut1 = createTensor(dim, sizes, syn_type_float, "reluOut1");
    NodePtr relu1    = NodeFactory::createNode({reluIn1}, {reluOut1}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, relu1);

    auto    reluIn2  = createTensor(dim, sizes, syn_type_float, "reluIn2");
    auto    reluOut2 = createTensor(dim, sizes, syn_type_float, "reluOut2");
    NodePtr relu2    = NodeFactory::createNode({reluIn2}, {reluOut2}, nullptr, "relu_fwd_f32", "relu2");
    GraphEditor::addNode(g, relu2);

    NodePtr reshape =
        NodeFactory::createNode({reluOut1}, {reluIn2}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    GraphEditor::addNode(g, reshape);
    reluIn2->setAsAliasSubTensor(reluOut1);

    relu2->getNodeAnnotation().inputsCacheMetaData.resize(1);
    relu2->getNodeAnnotation().inputsCacheMetaData.at(0).cmAction = CacheMaintenanceAction::DISCARD;

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    EXPECT_FALSE(reluOut1->isAliasedTensor()) << "reluOut1 is discarded by relu2, so it can't use reluIn1 memory";
    EXPECT_FALSE(reluOut2->isAliasedTensor()) << "reluIn2 is discarded, so its memory can't be used for reluOut2";
}

TEST_F(InplaceInuptReuseSuggestionTest, inplace_reuse_chain_allowed)
{
    GaudiGraph g;
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut1 = createTensor(dim, sizes, syn_type_float, "reluOut1");
    NodePtr relu1 = NodeFactory::createNode({reluIn}, {reluOut1}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, relu1);

    auto reluOut2 = createTensor(dim, sizes, syn_type_float, "reluOut2");
    NodePtr relu2 = NodeFactory::createNode({reluOut1}, {reluOut2}, nullptr, "relu_fwd_f32", "relu2");
    GraphEditor::addNode(g, relu2);

    auto reluOut3 = createTensor(dim, sizes, syn_type_float, "reluOut3");
    NodePtr relu3 = NodeFactory::createNode({reluOut2}, {reluOut3}, nullptr, "relu_fwd_f32", "relu3");
    GraphEditor::addNode(g, relu3);

    auto reluOut4 = createTensor(dim, sizes, syn_type_float, "reluOut4");
    NodePtr relu4 = NodeFactory::createNode({reluOut3}, {reluOut4}, nullptr, "relu_fwd_f32", "relu4");
    GraphEditor::addNode(g, relu4);

    auto reluOut5 = createTensor(dim, sizes, syn_type_float, "reluOut5");
    NodePtr relu5 = NodeFactory::createNode({reluOut4}, {reluOut5}, nullptr, "relu_fwd_f32", "relu5");
    GraphEditor::addNode(g, relu5);

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    ASSERT_TRUE(reluOut1->isAliasedTensor()) << "reluOut1 should reuse reluIn";
    ASSERT_TRUE(reluOut2->isAliasedTensor()) << "reluOut2 should reuse reluOut1";
    ASSERT_TRUE(reluOut3->isAliasedTensor()) << "reluOut3 should reuse reluOut2";
    ASSERT_TRUE(reluOut4->isAliasedTensor()) << "reluOut4 should reuse reluOut3";
    ASSERT_TRUE(reluOut5->isAliasedTensor()) << "reluOut5 should reuse reluOut4";
}

class InplaceInuptReuseSuggestionTestNoFuser : public InplaceInuptReuseSuggestionTest
{
    virtual void SetUp() override
    {
        setGlobalConfForTest(GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION, "false");
        setGlobalConfForTest(GCFG_COMPLEX_GUID_EXTRACTOR_MODE, "0");
        setGlobalConfForTest(GCFG_RUN_TPC_FUSER, "false");
        setGlobalConfForTest(GCFG_ENABLE_REMOVE_REDUNDANT_MEMCPY, "false");
        InplaceInuptReuseSuggestionTest::SetUp();
    }
};

TEST_F(InplaceInuptReuseSuggestionTestNoFuser, allow_reuse_only_for_last_consumer)
{
    GaudiGraph g;
    SizeArray  sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createPersistentTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createTensor(dim, sizes, syn_type_float, "reluOut1");
    NodePtr relu = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, relu);

    auto add1In1 = createPersistentTensor(dim, sizes, syn_type_float, "add1In1");
    auto add1Out = createTensor(dim, sizes, syn_type_float, "add1Out");
    NodePtr add1 = NodeFactory::createNode({add1In1, reluOut}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    auto add2Out = createTensor(dim, sizes, syn_type_float, "add2Out");
    NodePtr add2 = NodeFactory::createNode({add1Out, reluOut}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    auto memcpy1Out = createPersistentTensor(dim, sizes, syn_type_float, "memcpy1Out");
    NodePtr memcpy1 =
        NodeFactory::createNode({add1Out}, {memcpy1Out}, nullptr, NodeFactory::tpcMemcpyNodeTypeName, "memcpy1");
    GraphEditor::addNode(g, memcpy1);

    auto memcpy2Out = createPersistentTensor(dim, sizes, syn_type_float, "memcpy2Out");
    NodePtr memcpy2 =
        NodeFactory::createNode({add2Out}, {memcpy2Out}, nullptr, NodeFactory::tpcMemcpyNodeTypeName, "memcpy2");
    GraphEditor::addNode(g, memcpy2);

    graphVisualizationPre(g);
    g.compile();
    graphVisualizationPost(g);

    ASSERT_TRUE(!reluOut->isAliasedTensor()) << "reluOut1 should reuse reluIn";
    ASSERT_TRUE(!add1Out->isAliasedTensor())
        << "add1 shouldn't reuse any input - add1In1 is persistent and reluOut has a consumer that runs later in the execution";
    ASSERT_TRUE(add2Out->isAliasedTensor() && add2Out->getAliasTensor() == reluOut) << "addOut should alias reluOut";
}

TEST_F(InplaceInuptReuseSuggestionTest, block_reuse_when_input_has_multiple_consumers)
{
    GaudiGraph g;
    SizeArray sizes = {40, 40, 40, 40, 1};
    unsigned dim = 4;

    auto reluIn = createPersistentTensor(dim, sizes, syn_type_float, "reluIn");
    auto reluOut = createTensor(dim, sizes, syn_type_float, "reluOut1");
    NodePtr relu = NodeFactory::createNode({reluIn}, {reluOut}, nullptr, "relu_fwd_f32", "relu1");
    GraphEditor::addNode(g, relu);

    auto add1In1 = createPersistentTensor(dim, sizes, syn_type_float, "add1In1");
    auto add1Out = createTensor(dim, sizes, syn_type_float, "add1Out");
    NodePtr add1 = NodeFactory::createNode({add1In1, reluOut}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    auto add2In1 = createPersistentTensor(dim, sizes, syn_type_float, "add2In1");
    auto add2Out = createTensor(dim, sizes, syn_type_float, "add2Out");
    NodePtr add2 = NodeFactory::createNode({add2In1, reluOut}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    GraphEditor::addNode(g, add2);

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    ASSERT_TRUE(!reluOut->isAliasedTensor()) << "reluOut1 should not reuse reluIn because it is persistent";
    ASSERT_TRUE(!add1Out->isAliasedTensor()) << "add1Out should not reuse reluOut1 because reluOut1 has multiple consumers";
    ASSERT_TRUE(!add2Out->isAliasedTensor()) << "add2Out should not reuse reluOut1 because reluOut1 has multiple consumers";
}

TEST_F(InplaceInuptReuseSuggestionTest, block_reuse_for_multibuffered_tensors)
{
    GaudiGraph g;
    SizeArray sizes = {1000, 1000};
    unsigned dim = 2;

    auto add1In0 = createTensor(dim, sizes, syn_type_float, "add1In1");
    auto add1In1 = createTensor(dim, sizes, syn_type_float, "add1In1");
    auto add1Out = createTensor(dim, sizes, syn_type_float, "add1Out");
    add1In0->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(getNextMemSectionId());
    NodePtr add1 = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    GraphEditor::addNode(g, add1);

    graphVisualizationPre(g);
    inPlaceInputReuseSuggestion(g);
    graphVisualizationPost(g);

    ASSERT_TRUE(add1Out->getAliasTensor() == add1In1) << "add1Out should not reuse add1In0 because it is multibuffered, so addIn1 should be reused";
}
