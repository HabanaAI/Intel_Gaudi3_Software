#include "compilation_hal_reader.h"
#include "define_synapse_common.hpp"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "graph_optimizer_test.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "node.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "pipeline_management/fusion_candidates_collector.h"
#include "pipeline_management/fusion_candidates_selector.h"
#include "pipeline_management/fusion_db.h"
#include "pipeline_management/fusion_handlers.h"
#include "pipeline_management/fusion_handlers_factory.h"
#include "pipeline_management/spill_fill_classifier.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "gtest/gtest.h"
#include <iostream>
#include "platform/gaudi2/graph_compiler/passes.h"
#include "types.h"
#include "brain_conf.h"

class SpillFillTest : public GraphOptimizerTest
{
public:
    TensorPtr createPersistentTensor(const std::string name = "")
    {
        synMemoryDescriptor persistentMemoryDesc(true);
        const auto          t = createTensorInDram(name);
        t->setMemoryDescriptor(persistentMemoryDesc);
        return t;
    }
    TensorPtr createTensorInSram(const std::string& name = "")
    {
        SizeArray   sizes = {32, 32, 32, 32};
        unsigned    dims  = 4;
        synDataType dtype = syn_type_float;
        TensorPtr   t     = std::make_shared<Tensor>(dims, sizes.data(), dtype);
        t->setName(name);
        t->setTensorInSram();
        return t;
    }
    TensorPtr createTensorInDram(const std::string& name = "")
    {
        SizeArray   sizes = {32, 32, 32, 32};
        unsigned    dims  = 4;
        synDataType dtype = syn_type_float;
        TensorPtr   t     = std::make_shared<Tensor>(dims, sizes.data(), dtype);
        t->setName(name);
        t->setTensorInDram();
        return t;
    }
    virtual void runTest(Gaudi2Graph& m_graph)
    {
        graphVisualizationPre(m_graph);
        ASSERT_TRUE(tpcFuser(m_graph));
        ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));
        ASSERT_TRUE(fuseSpillFillDirectives(m_graph));
        graphVisualizationPost(m_graph);
    }
    NodePtr addNodeToGraph(Gaudi2Graph&       m_graph,
                           const char*        guid,
                           TensorVector       inputTensorIndices,
                           TensorVector       outputTensorIndices,
                           const std::string& nodeName   = "",
                           UserParams         userParams = nullptr,
                           const unsigned     paramsSize = 0)
    {
        auto node =
            NodeFactory::createNode(inputTensorIndices, outputTensorIndices, userParams, paramsSize, guid, nodeName);
        GraphEditor::addNode(m_graph, node);
        return node;
    }

    template<typename ContainerType>
    std::optional<CandidateInfo> getCandidateFromNode(const NodePtr& node, const ContainerType& candidates)
    {
        for (const auto& candidate : candidates)
        {
            if (candidate.getNode() == node) return candidate;
        }
        return {};
    }

    Gaudi2Graph m_graph;

protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_LAYERED_PIPELINE_BRAIN, "true");
        setGlobalConfForTest(GCFG_ENABLE_SPILL_FILL_FUSION, "true");
    }
};

// Input graph: [DRAM] ---> relu ---> [SRAM] ---> memcpy ---> [DRAM]
// Output graph: [DRAM] ---> relu ---> [SRAM]
//                              \---> [DRAM]
TEST_F(SpillFillTest, fuse_relu_memcpy)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);
    TensorPtr                  in   = createPersistentTensor();
    TensorPtr                  out  = createTensorInSram();
    auto                       relu = addNodeToGraph(m_graph, "relu_fwd_f32", {in}, {out});

    TensorPtr memcpyOut = createTensorInDram();
    auto      memcpy    = addNodeToGraph(m_graph, "memcpy", {out}, {memcpyOut});

    runTest(m_graph);

    auto nodes = m_graph.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 1) << "Graph should contain one node (relu)";
    ASSERT_EQ(nodes.front()->getNodeType(), Node::TYPE_USER);
    ASSERT_EQ(nodes.front()->getNumOutputs(), 2);
    ASSERT_EQ(nodes.front()->getOutput(0), out);
    ASSERT_EQ(nodes.front()->getOutput(1), memcpyOut);
}

// Input graph: [DRAM] ---> memcpy1 ---> [SRAM] ---> memcpy2 ---> [DRAM]
TEST_F(SpillFillTest, spill_fill_classifier_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn1  = createTensorInDram();
    TensorPtr memcpyOut1 = createTensorInSram();
    auto      memcpy1    = addNodeToGraph(m_graph, "memcpy", {memcpyIn1}, {memcpyOut1});
    TensorPtr memcpyOut2 = createTensorInDram();
    auto      memcpy2    = addNodeToGraph(m_graph, "memcpy", {memcpyOut1}, {memcpyOut2});

    SpillFillClassifier classifier(m_graph);
    const auto&         nodes = m_graph.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 2) << "Graph should contain two memcpy nodes";
    ASSERT_TRUE(isMemcpy(*nodes.front()));
    ASSERT_TRUE(isMemcpy(*nodes.back()));
    auto fills = classifier.getFillDirectives();
    ASSERT_EQ(fills.size(), 1);
    ASSERT_EQ(fills.front(), memcpy1);
    auto spills = classifier.getSpillDirectives();
    ASSERT_EQ(spills.size(), 1);
    ASSERT_EQ(spills.front(), memcpy2);
}

// Input graph:  batch_gemm ---> [SRAM] ---> memcpy ---> [DRAM] ---> relu2 --> [DRAM]
//                                     \---> relu1 ---> [DRAM]
TEST_F(SpillFillTest, spill_candidates_collector)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr mmeIn0 = createTensorInDram();
    TensorPtr mmeIn1 = createTensorInDram();
    TensorPtr mmeOut = createTensorInSram();
    auto      bgemm  = addNodeToGraph(m_graph, NodeFactory::batchGemmNodeTypeName, {mmeIn0, mmeIn1}, {mmeOut}, "bgemm");

    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {mmeOut}, {memcpyOut}, "memcpy");

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {mmeOut}, {reluOut}, "relu1");

    TensorPtr reluOut2 = createTensorInDram();
    auto      relu2    = addNodeToGraph(m_graph, "relu_fwd_f32", {memcpyOut}, {reluOut2}, "relu2");

    FusionCandidatesCollector collector;
    auto                      candidates = collector.getSpillFusionCandidates(m_graph, spill);
    ASSERT_EQ(candidates.size(), 2);
    ASSERT_TRUE(getCandidateFromNode(bgemm, candidates).has_value());
    ASSERT_TRUE(getCandidateFromNode(relu1, candidates).has_value());
    ASSERT_FALSE(getCandidateFromNode(relu2, candidates).has_value());
}

// Input graph:  batch_gemm ---> [SRAM] ---> memcpy ---> [DRAM]
//                                     \---> relu1 ---> [DRAM]
TEST_F(SpillFillTest, fusion_handler_is_fusion_valid_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr mmeIn0 = createTensorInDram();
    TensorPtr mmeIn1 = createTensorInDram();
    TensorPtr mmeOut = createTensorInSram();
    auto      bgemm  = addNodeToGraph(m_graph, NodeFactory::batchGemmNodeTypeName, {mmeIn0, mmeIn1}, {mmeOut});

    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {mmeOut}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {mmeOut}, {reluOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      bgemmCandidate = getCandidateFromNode(bgemm, candidates);
    ASSERT_TRUE(bgemmCandidate.has_value());
    ASSERT_FALSE(TpcFusionHandler::isValidForFusion(m_graph, spill, bgemmCandidate.value()));

    auto relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    ASSERT_TRUE(TpcFusionHandler::isValidForFusion(m_graph, spill, relu1Candidate.value()));
}

// Input graph: [SRAM] ---> memcpy ---> [DRAM] ---> relu --> [DRAM]
TEST_F(SpillFillTest, DISABLED_fusion_handler_fuse_test)
{
    TpcRecompileDb             tpcFusionDb;
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn  = createTensorInSram();
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {memcpyIn}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {memcpyIn}, {reluOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    TpcFusionHandler handler(m_graph, relu1Candidate.value(), tpcFusionDb);
    ASSERT_TRUE(handler.fuse(spill));
}

// Input graph:  batch_gemm ---> [SRAM] ---> memcpy ---> [DRAM]
//                                     \---> relu1 ---> [DRAM]
TEST_F(SpillFillTest, fusion_db_reject_invalid_fusion)
{
    TpcRecompileDb             tpcFusionDb;
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn  = createTensorInSram();
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {memcpyIn}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {memcpyIn}, {reluOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    // register fusion as unsuccessul:
    void*    resultKernelElf  = nullptr;
    unsigned resultKernelSize = 0;
    tpcFusionDb.registerTpcFusion({relu1->getGUID(), relu1Candidate->getTensorIdToDuplicate()},
                                  {false, resultKernelElf, resultKernelSize});
    auto cachedFusionOpt = tpcFusionDb.getFusionFromDb({relu1->getGUID(), relu1Candidate->getTensorIdToDuplicate()});
    ASSERT_TRUE(cachedFusionOpt.has_value());
    const auto& [fusionSucceeded, elf, elfSize] = cachedFusionOpt.value();
    ASSERT_FALSE(fusionSucceeded);
    ASSERT_EQ(elf, nullptr);
    ASSERT_EQ(elfSize, 0);
    TpcFusionHandler handler(m_graph, relu1Candidate.value(), tpcFusionDb);
    ASSERT_FALSE(handler.fuse(spill));
}

// Input graph:  batch_gemm ---> [SRAM] ---> memcpy ---> [DRAM]
//                                     \---> relu1 ---> [DRAM]
TEST_F(SpillFillTest, fusion_db_fetch_valid_fusion)
{
    TpcRecompileDb             tpcFusionDb;
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn  = createTensorInSram();
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {memcpyIn}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {memcpyIn}, {reluOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    // register fusion as successul:
    void*    resultKernelElf  = nullptr;
    unsigned resultKernelSize = 0;
    tpcFusionDb.registerTpcFusion({relu1->getGUID(), relu1Candidate->getTensorIdToDuplicate()},
                                  {true, resultKernelElf, resultKernelSize});
    auto cachedFusionOpt = tpcFusionDb.getFusionFromDb({relu1->getGUID(), relu1Candidate->getTensorIdToDuplicate()});
    ASSERT_TRUE(cachedFusionOpt.has_value());
    const auto& [fusionSucceeded, elf, elfSize] = cachedFusionOpt.value();
    ASSERT_TRUE(fusionSucceeded);
    ASSERT_EQ(elf, nullptr);
    ASSERT_EQ(elfSize, 0);
}

// Input graph:  batch_gemm ---> [SRAM] ---> memcpy ---> [DRAM]
//                                     \---> relu1 ---> [DRAM]
TEST_F(SpillFillTest, tpc_fusion_factory_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr mmeIn0 = createTensorInDram();
    TensorPtr mmeIn1 = createTensorInDram();
    TensorPtr mmeOut = createTensorInSram();
    auto      bgemm  = addNodeToGraph(m_graph, NodeFactory::batchGemmNodeTypeName, {mmeIn0, mmeIn1}, {mmeOut});

    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {mmeOut}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {mmeOut}, {reluOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));

    FusionCandidatesCollector collector;
    auto                      candidates = collector.getSpillFusionCandidates(m_graph, spill);

    auto relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    TpcFusionHandlersFactory tpcFusionFactory;
    ASSERT_NE(tpcFusionFactory.createForCandidate(m_graph, spill, relu1Candidate.value()), nullptr);

    auto bgemmCandidate = getCandidateFromNode(bgemm, candidates);
    ASSERT_TRUE(bgemmCandidate.has_value());
    ASSERT_EQ(tpcFusionFactory.createForCandidate(m_graph, spill, bgemmCandidate.value()), nullptr);
}

// Input graph:  [SRAM] --------------------------------------------------> add ---> [DRAM]
//                     \---> memcpy ---> [DRAM] ---> relu ---> [DRAM] ----/
TEST_F(SpillFillTest, fusion_rejected_cycle_in_graph)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn  = createTensorInSram();
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {memcpyIn}, {memcpyOut});

    TensorPtr reluOut = createTensorInDram();
    auto      relu1   = addNodeToGraph(m_graph, "relu_fwd_f32", {memcpyOut}, {reluOut});

    TensorPtr addOut = createTensorInDram();
    auto      add    = addNodeToGraph(m_graph, "add_fwd_f32", {memcpyIn, reluOut}, {addOut});

    ASSERT_TRUE(gaudi2::loadTpcKernels(m_graph));
    FusionCandidatesCollector collector;
    auto                      candidates   = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      addCandidate = getCandidateFromNode(add, candidates);
    ASSERT_TRUE(addCandidate.has_value());
    ASSERT_FALSE(TpcFusionHandler::isValidForFusion(m_graph, spill, addCandidate.value()));
}

class SpillCandidateSelectorTest : public SpillFillTest
{
public:
    class TestHandlerFactory : public FusionHandlersFactory
    {
    public:
        unsigned m_numCalls {};
    };
    class AlwaysSucceedHandler : public FusionHandler
    {
    public:
        AlwaysSucceedHandler(const CandidateInfo& candidate) : FusionHandler(candidate) {};
        bool fuse(const NodePtr& directive) override { return true; };
    };
    class AlwaysSucceedHandlerFactory : public TestHandlerFactory
    {
    public:
        AlwaysSucceedHandlerFactory() {};
        FusionHandlerPtr
        createForCandidate(HabanaGraph& m_graph, const NodePtr& directive, const CandidateInfo& candidate)
        {
            this->m_numCalls++;
            return std::make_shared<AlwaysSucceedHandler>(candidate);
        };
    };
    class AlwaysFailHandlerFactory : public TestHandlerFactory
    {
    public:
        AlwaysFailHandlerFactory() {};
        FusionHandlerPtr
        createForCandidate(HabanaGraph& m_graph, const NodePtr& directive, const CandidateInfo& candidate)
        {
            this->m_numCalls++;
            return nullptr;
        };
    };
};

// Input graph: [DRAM] ---> relu ---> [SRAM] ---> memcpy ---> [DRAM]
TEST_F(SpillCandidateSelectorTest, selector_select_handlers_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr reluIn    = createTensorInDram();
    TensorPtr reluOut   = createTensorInSram();
    auto      relu1     = addNodeToGraph(m_graph, "relu_fwd_f32", {reluIn}, {reluOut});
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {reluOut}, {memcpyOut});

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    std::shared_ptr<AlwaysSucceedHandlerFactory> succeedFactoryPtr(new AlwaysSucceedHandlerFactory());
    std::shared_ptr<AlwaysFailHandlerFactory>    failFactoryPtr(new AlwaysFailHandlerFactory());
    auto                                         handlers = FusionCandidatesSelector::selectHandlers(m_graph,
                                                             spill,
                                                                                                     {relu1Candidate.value()},
                                                                                                     {succeedFactoryPtr, failFactoryPtr});
    ASSERT_EQ(handlers.size(), 1);
    ASSERT_EQ(succeedFactoryPtr->m_numCalls, 1);
    ASSERT_EQ(failFactoryPtr->m_numCalls, 1);
    ASSERT_TRUE(handlers.front()->fuse(spill));
}

// Input graph: [SRAM] ---> memcpy ---> [DRAM]
TEST_F(SpillCandidateSelectorTest, selector_no_candidates_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr memcpyIn  = createTensorInSram();
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {memcpyIn}, {memcpyOut});

    std::shared_ptr<AlwaysSucceedHandlerFactory> succeedFactoryPtr(new AlwaysSucceedHandlerFactory());
    std::shared_ptr<AlwaysFailHandlerFactory>    failFactoryPtr(new AlwaysFailHandlerFactory());
    auto handlers = FusionCandidatesSelector::selectHandlers(m_graph, spill, {}, {succeedFactoryPtr, failFactoryPtr});
    ASSERT_EQ(handlers.size(), 0);
}

// Input graph: [DRAM] ---> relu ---> [SRAM] ---> memcpy ---> [DRAM]
TEST_F(SpillCandidateSelectorTest, selector_no_handlers_test)
{
    CompilationHalReaderSetter compHalReaderSetter(&m_graph);

    TensorPtr reluIn    = createTensorInDram();
    TensorPtr reluOut   = createTensorInSram();
    auto      relu1     = addNodeToGraph(m_graph, "relu_fwd_f32", {reluIn}, {reluOut});
    TensorPtr memcpyOut = createTensorInDram();
    auto      spill     = addNodeToGraph(m_graph, "memcpy", {reluOut}, {memcpyOut});

    FusionCandidatesCollector collector;
    auto                      candidates     = collector.getSpillFusionCandidates(m_graph, spill);
    auto                      relu1Candidate = getCandidateFromNode(relu1, candidates);
    ASSERT_TRUE(relu1Candidate.has_value());
    auto handlers = FusionCandidatesSelector::selectHandlers(m_graph, spill, {relu1Candidate.value()}, {});
    ASSERT_EQ(handlers.size(), 0);
}

class FusionCandidatesDbTest : public SpillFillTest
{
public:
    FusionCandidatesDbTest() : m_graph(), m_db()
    {
    }

    void SetUp() override
    {
        SpillFillTest::SetUp();
        run();
    }

    void run()
    {
        buildGraph();
        m_db          = FusionCandidatesDb(m_graph);
        m_directives  = m_db.getDirectives();
        m_candidates1 = m_db.getCandidatesForDirective(m_spill1);
        m_candidates2 = m_db.getCandidatesForDirective(m_spill2);
    }

    // Input graph: [DRAM] ---> relu1 ---> [SRAM] ---> memcpy ---> [DRAM]
    //                                          \---> bgemm ---> [SRAM] ---> memcpy ---> [DRAM]
    //                                                                  \---> relu2 ---> [DRAM]
    void buildGraph()
    {
        TensorPtr reluIn    = createTensorInDram();
        TensorPtr reluOut   = createTensorInSram();
        m_relu1             = addNodeToGraph(m_graph, "relu_fwd_f32", {reluIn}, {reluOut}, "relu1");
        TensorPtr memcpyOut = createTensorInDram();
        m_spill1            = addNodeToGraph(m_graph, "memcpy", {reluOut}, {memcpyOut}, "m_spill1");

        TensorPtr mmeIn1 = createTensorInDram();
        TensorPtr mmeOut = createTensorInSram();
        m_bgemm = addNodeToGraph(m_graph, NodeFactory::batchGemmNodeTypeName, {reluOut, mmeIn1}, {mmeOut}, "bgemm");

        TensorPtr memcpyOut2 = createTensorInDram();
        m_spill2             = addNodeToGraph(m_graph, "memcpy", {mmeOut}, {memcpyOut2}, "m_spill2");

        TensorPtr reluOut2 = createTensorInDram();
        m_relu2            = addNodeToGraph(m_graph, "relu_fwd_f32", {mmeOut}, {reluOut2}, "relu2");
    }

    NodePtr            m_relu1;
    NodePtr            m_relu2;
    NodePtr            m_bgemm;
    NodePtr            m_spill1;
    NodePtr            m_spill2;
    Gaudi2Graph        m_graph;
    NodeVector         m_directives;
    FusionCandidatesDb m_db;
    CandidatesInfoSet  m_candidates1;
    CandidatesInfoSet  m_candidates2;
};

TEST_F(FusionCandidatesDbTest, get_directives_test)
{
    ASSERT_EQ(m_directives.size(), 2);
}

TEST_F(FusionCandidatesDbTest, get_candidates_test)
{
    ASSERT_EQ(m_candidates1.size(), 2);
    ASSERT_EQ(m_candidates2.size(), 2);
}

TEST_F(FusionCandidatesDbTest, check_common_candidates)
{
    CandidatesInfoSet intersection;
    set_intersection(m_candidates1.begin(),
                     m_candidates1.end(),
                     m_candidates2.begin(),
                     m_candidates2.end(),
                     std::inserter(intersection, intersection.begin()),
                     CandidateInfoCmp());
    ASSERT_EQ(intersection.size(), 1);
    ASSERT_TRUE(HabanaGraph::runsOnMME(intersection.begin()->getNode()));
}

TEST_F(FusionCandidatesDbTest, update_fused_test)
{
    // When updating relu1 as chosen candidate for fusion for spill1
    m_db.updateFused(m_spill1, getCandidateFromNode(m_relu1, m_candidates1).value());
    m_candidates1 = m_db.getCandidatesForDirective(m_spill1);
    // Then the db should be updated -
    // 1. spill1 has only 1 candidate in its list, which is relu1
    ASSERT_EQ(m_candidates1.size(), 1);
    ASSERT_TRUE(m_candidates1.find(getCandidateFromNode(m_relu1, m_candidates1).value()) != m_candidates1.end());
    // 2. spill2 candidates are unaffected since relu1 was not one of its candidates
    m_candidates2 = m_db.getCandidatesForDirective(m_spill2);
    ASSERT_EQ(m_candidates2.size(), 2);
    // When updating relu2 as chosen candidate for fusion for spill2
    m_db.updateFused(m_spill2, getCandidateFromNode(m_relu2, m_candidates2).value());
    m_candidates2 = m_db.getCandidatesForDirective(m_spill2);
    // Then the db should be updated - spill2 has only 1 candidate in its list, which is relu2
    ASSERT_EQ(m_candidates2.size(), 1);
    ASSERT_TRUE(m_candidates2.find(getCandidateFromNode(m_relu2, m_candidates2).value()) != m_candidates2.end());
}

TEST_F(FusionCandidatesDbTest, update_fused_with_intersection_chosen_test)
{
    // When updating bgemm as chosen candidate for fusion for spill1
    m_db.updateFused(m_spill1, getCandidateFromNode(m_bgemm, m_candidates1).value());
    m_candidates1 = m_db.getCandidatesForDirective(m_spill1);
    // Then the db should be updated -
    // 1. spill1 has only 1 candidate in its list, which is bgemm
    // 2. spill2 also has only 1 candidate, which is relu1 (since bgemm was one of its candidates)
    ASSERT_EQ(m_candidates1.size(), 1);
    ASSERT_TRUE(m_candidates1.find(getCandidateFromNode(m_bgemm, m_candidates1).value()) != m_candidates1.end());
    m_candidates2 = m_db.getCandidatesForDirective(m_spill2);
    ASSERT_EQ(m_candidates2.size(), 1);
    ASSERT_TRUE(m_candidates2.find(getCandidateFromNode(m_relu2, m_candidates2).value()) != m_candidates2.end());
}