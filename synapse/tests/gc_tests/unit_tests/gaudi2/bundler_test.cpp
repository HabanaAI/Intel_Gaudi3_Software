#include "bundler/bundle_seed_collector.h"
#include "common_type_utils.h"
#include "bundler/layered_brain_bundle.h"
#include "layered_brain_test_common.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "gaudi2_graph.h"
#include "graph_editor.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "bundler/bundlers.h"
#include "types.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <iterator>
#include "layered_brain_test.h"

using namespace gc::layered_brain;
using namespace gc::layered_brain::bundler;

class MmeBundlerTest : public LayeredBrainCommonTest<Gaudi2Graph>
{
};

/**
       +----+
       |add0|------+
       +----+      |
                   |  +------+
       +----+      +->|bgemm |
       |add1|-------->|      |
       +----+         +------+
 */
TEST_F(MmeBundlerTest, mme_bundler_two_add_one_batch_gemm_bundle)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto add0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0Out = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm =
        NodeFactory::createNode({add0Out, add1Out}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();
    const auto nBundles  = bundleMap.size();

    // Expecting 1 bundle with bgemm and 2 tpc add nodes
    ASSERT_EQ(nBundles, 1);
    ASSERT_TRUE(!bundleMap.empty());
    ASSERT_EQ(bundleMap.begin()->second.size(), 3);
}

/**
       +----+
       |add0|------+
       +----+      |
                   |  +------+
       +----+      +->|gemm  |
       |add1|-------->|      |
       +----+         +------+
 */
TEST_F(MmeBundlerTest, mme_bundler_two_add_one_gemm_bundle)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto add0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0Out = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto gemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto gemm =
        NodeFactory::createNode({add0Out, add1Out}, {gemmOut}, nullptr, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(gemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm)) << "Failed adding node: " << gemm->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();
    const auto nBundles  = bundleMap.size();

    ASSERT_TRUE(nBundles == 1);
    ASSERT_TRUE(bundleMap.begin()->second.size() == 3);
}

/**

       +------+   +---------+   +-------+   +----+
       |bgemm0|-->|transpose|-->|reshape|-->|add1|
       +------+   +---------+   +-------+   +----+
 */
TEST_F(MmeBundlerTest, bgemm_with_tpc_and_consumers)
{
    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto bgemm0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto bgemm0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0    = NodeFactory::createNode({bgemm0In0, bgemm0In1},
                                                {bgemm0Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_0");
    ASSERT_TRUE(bgemm0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm0)) << "Failed adding node: " << bgemm0->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch},
                                        .tensorDim   = 4};

    const auto transpose = NodeFactory::createNode({bgemm0Out},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto reshapeOut = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto reshape =
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add1In1 = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({reshapeOut, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();
    ASSERT_EQ(bundleMap.size(), 1);  // Expecting a single bundle
    const auto& firstBundleNodes = bundleMap.begin()->second;
    // Batch gemm without producer: expecting all nodes bundled
    ASSERT_EQ(firstBundleNodes.size(), 4);
}

/*
                                               ┌─────┐         ┌──────┐
                                               │ add ├────────►│      │
                                               └─────┘         │      │
                                                               │ gemm │
                                                               │      │
           ┌─────┐      ┌────────────┐         ┌────────┐      │      │
           │add  ├─────►│transpose   ├────────►│reshape ├─────►│      │
           └─────┘      └────────────┘         └────────┘      └──────┘
*/
TEST_F(MmeBundlerTest, mme_bundler_add_transpose_reshape_chain_and_single_add_with_gemm)
{
    const auto addIn0      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto addIn1      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto transposeIn = createTensor(SizeVector {64, 32}, syn_type_single);

    const auto add = NodeFactory::createNode({addIn0, addIn1}, {transposeIn}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {32, 64}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel}, .tensorDim = 2};

    const auto transpose = NodeFactory::createNode({transposeIn},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto reshapeOut = createTensor(SizeVector {128, 16}, syn_type_single);
    const auto reshape =
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add1In0 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto gemmOut = createTensor(SizeVector {512, 16}, syn_type_single);
    const auto gemm =
        NodeFactory::createNode({reshapeOut, add1Out}, {gemmOut}, nullptr, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(gemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm)) << "Failed adding node: " << gemm->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();
    const auto nBundles  = bundleMap.size();

    ASSERT_TRUE(nBundles == 1) << "Num bundles: " << nBundles;
    const auto nBundleNodes = bundleMap.begin()->second.size();
    ASSERT_TRUE(nBundleNodes == 5) << "Num bundles nodes: " << nBundleNodes;
}

TEST_F(MmeBundlerTest, mme_bundler_two_bundles_first_add_transpose_reshape_chain_and_single_add_and_second_gemm_and_add)
{
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    ScopedConfigurationChange limitGemmBundles("LIMIT_GEMM_BUNDLES_EXPANSION", "true");
    const auto                addIn0      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto                addIn1      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto                transposeIn = createTensor(SizeVector {64, 32}, syn_type_single);

    const auto add = NodeFactory::createNode({addIn0, addIn1}, {transposeIn}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {32, 64}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel}, .tensorDim = 2};

    const auto transpose = NodeFactory::createNode({transposeIn},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto reshapeOut = createTensor(SizeVector {128, 16}, syn_type_single);
    const auto reshape =
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add1In0 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto gemmOut = createTensor(SizeVector {512, 16}, syn_type_single);
    const auto gemm =
        NodeFactory::createNode({reshapeOut, add1Out}, {gemmOut}, nullptr, NodeFactory::gemmNodeTypeName, "gemm0");
    ASSERT_TRUE(gemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm)) << "Failed adding node: " << gemm->getNodeName();

    const auto add2In1 = createTensor(SizeVector {512, 16}, syn_type_single);
    const auto add2Out = createTensor(SizeVector {512, 16}, syn_type_single);
    const auto add2    = NodeFactory::createNode({gemmOut, add2In1}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    ASSERT_TRUE(add2 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add2)) << "Failed adding node: " << add->getNodeName();

    const auto gemm1In1 = createTensor(SizeVector {512, 256}, syn_type_single);
    const auto gemm1Out = createTensor(SizeVector {256, 16}, syn_type_single);
    const auto gemm1 =
        NodeFactory::createNode({add2Out, gemm1In1}, {gemm1Out}, nullptr, NodeFactory::gemmNodeTypeName, "gemm1");
    ASSERT_TRUE(gemm1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm1)) << "Failed adding node: " << gemm1->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto           bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto           bundleMap = bundler->generateBundles();
    const auto           nBundles  = bundleMap.size();
    // (add)->(transpose)->(reshape)->(gemm)->(add2)->(gemm1)
    //                                 /
    //                              (add1)

    // Expecting gemm to bundle all producers and consumers expander not to deploy.
    // Then expecting gemm1 to bundle the remaining producer.
    std::vector<NodePtr> firstBundle {add, transpose, reshape, add1, gemm};
    std::vector<NodePtr> secondBundle {add2, gemm1};

    const auto isSameBundle = [](const auto& nodes) {
        std::optional<unsigned> bundleId;
        return std::all_of(nodes.begin(), nodes.end(), [&bundleId](const NodePtr& n) {
            const auto bi = n->getNodeAnnotation().bundleInfo;
            if (!bi.is_set()) return false;
            if (!bundleId.has_value())
            {
                bundleId = bi->bundleIndex;
                return true;
            }
            else
            {
                return (bundleId.value() == bi->bundleIndex);
            }
        });
    };

    ASSERT_TRUE(nBundles == 2) << "Num bundles: " << nBundles;
    ASSERT_TRUE(isSameBundle(firstBundle));
    ASSERT_TRUE(isSameBundle(secondBundle));
}

/**
       +----+   +------+
       |add +-->|bgemm |
       +----+   +------+
 */
TEST_F(MmeBundlerTest, mme_bundler_isolated_single_batch_gemm_and_tpc_producer)
{
    constexpr unsigned batch0   = 4;
    constexpr unsigned batch1   = 3;
    const auto         bgemmIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto         bgemmIn1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto         bgemmOut = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto         bgemm =
        NodeFactory::createNode({bgemmIn0, bgemmIn1}, {bgemmOut}, nullptr, NodeFactory::batchGemmNodeTypeName, "bgemm");
    ASSERT_TRUE(bgemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm)) << "Failed adding node: " << bgemm->getNodeName();

    const auto addIn0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto addIn1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add    = NodeFactory::createNode({addIn0, addIn1}, {bgemmIn0}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const auto bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();

    const auto nBundles = bundleMap.size();
    ASSERT_TRUE(nBundles == 1);

    const auto& nBundleNodes = bundleMap.begin()->second.size();
    ASSERT_TRUE(nBundleNodes == 2);
}

class IterativeMmeBundlerTest : public LayeredBrainCommonTest<Gaudi2Graph>
{
protected:
    std::vector<BundleAndExpanders> getInitialBundles(const std::unique_ptr<Bundler>& bundler)
    {
        std::vector<BundleAndExpanders> initialBundles {};
        for (auto& seedCollector : bundler->getSeedCollectors())
        {
            for (auto& seedCandidateAndExpanders : seedCollector->collect(true /*iterative mode*/))
            {
                auto& bundleSeedCandidate = seedCandidateAndExpanders.first;
                // candidate seed always passes evaluation
                bundleSeedCandidate->acceptCandidates();
                initialBundles.push_back(std::move(seedCandidateAndExpanders));
            }
        }
        return initialBundles;
    }

    bool expandStep(const BundlerPtr& bundler, const BundlePtr& bundle, BundleExpanders& expanders)
    {
        const bool success = bundler->expansionStep(bundle, expanders);
        if (success)
        {
            // every successful expansion passes evaluation
            bundle->acceptCandidates();
        }
        return success;
    }

    void expandBundle(const BundlerPtr& bundler, BundlePtr& bundle, BundleExpanders& expanders)
    {
        bool success = false;
        do
        {
            success = expandStep(bundler, bundle, expanders);
        } while (success);
    }

    bool expandAndValidate(const BundlerPtr&              bundler,
                           BundleExpanders&               expanders,
                           const BundlePtr&               bundle,
                           const std::vector<NodeVector>& expectedExpansion)
    {
        HB_ASSERT_PTR(bundler);
        HB_ASSERT_PTR(bundle);
        unsigned expectedBundleSize = bundle->getNodes().size();  // start from initial seed size
        for (const auto& expansionStep : expectedExpansion)
        {
            expectedBundleSize += expansionStep.size();
            [[maybe_unused]] const bool success = expandStep(bundler, bundle, expanders);
            if (expectedBundleSize != bundle->getNodes().size()) return false;
            if (!std::all_of(expansionStep.begin(), expansionStep.end(), [&bundle](const auto& n) {
                    return bundle->getNodes().find(n) != bundle->getNodes().end();
                }))
                return false;
        }
        return true;
    }
};

/*
                                               ┌─────┐         ┌──────┐
                                               │ add ├────────►│      │
                                               └─────┘         │      │
                                                               │ gemm │
                                                               │      │
           ┌─────┐      ┌────────────┐         ┌────────┐      │      │
           │add  ├─────►│transpose   ├────────►│reshape ├─────►│      │
           └─────┘      └────────────┘         └────────┘      └──────┘
*/
TEST_F(IterativeMmeBundlerTest, mme_bundler_add_transpose_reshape_chain_and_single_add_with_gemm)
{
    const auto addIn0      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto addIn1      = createTensor(SizeVector {64, 32}, syn_type_single);
    const auto transposeIn = createTensor(SizeVector {64, 32}, syn_type_single);

    const auto add = NodeFactory::createNode({addIn0, addIn1}, {transposeIn}, nullptr, "add_fwd_f32", "add");
    ASSERT_TRUE(add != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add)) << "Failed adding node: " << add->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {32, 64}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel}, .tensorDim = 2};

    const auto transpose = NodeFactory::createNode({transposeIn},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto reshapeOut = createTensor(SizeVector {128, 16}, syn_type_single);
    const auto reshape =
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add1In0 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1In1 = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {512, 128}, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto gemmOut = createTensor(SizeVector {512, 16}, syn_type_single);
    const auto gemm =
        NodeFactory::createNode({reshapeOut, add1Out}, {gemmOut}, nullptr, NodeFactory::gemmNodeTypeName, "gemm");
    ASSERT_TRUE(gemm != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, gemm)) << "Failed adding node: " << gemm->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);

    // Expecting exactly gemm seed bundled
    ASSERT_EQ(initialBundles.size(), 1);
    auto& [bundle, expanders] = initialBundles.front();

    // Expecting first valid expansion step to add
    // (add)->(transpose)->(reshape) to bundle
    // Second expansion step should append the second add (add1)
    ASSERT_TRUE(expandAndValidate(bundler, expanders, bundle, {{add, transpose, reshape}, {add1}}));
    const auto nBundleNodes = bundle->getNodes().size();
    ASSERT_TRUE(nBundleNodes == 5) << "Num bundles nodes: " << nBundleNodes;

    // expecting no other expansions
    ASSERT_FALSE(expandStep(bundler, bundle, expanders));
}

/**

       +----+   +------+   +---------+   +-------+   +----+
       |add0|-->|bgemm0|-->|transpose|-->|reshape|-->|add1|
       +----+   +------+   +---------+   +-------+   +----+
 */
TEST_F(IterativeMmeBundlerTest, bgemm_with_tpc_producer_and_consumer)
{
    ScopedConfigurationChange limitGemmBundles("LIMIT_GEMM_BUNDLES_EXPANSION", "true");

    constexpr unsigned batch0 = 4;
    constexpr unsigned batch1 = 3;

    const auto add0In0 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0In1 = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0Out = createTensor(SizeVector {64, 32, batch0, batch1}, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto bgemm0In0 = add0Out;
    const auto bgemm0In1 = createTensor(SizeVector {128, 64, batch0, batch1}, syn_type_single);
    const auto bgemm0Out = createTensor(SizeVector {128, 32, batch0, batch1}, syn_type_single);
    const auto bgemm0    = NodeFactory::createNode({bgemm0In0, bgemm0In1},
                                                {bgemm0Out},
                                                nullptr,
                                                NodeFactory::batchGemmNodeTypeName,
                                                "bgemm_0");
    ASSERT_TRUE(bgemm0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, bgemm0)) << "Failed adding node: " << bgemm0->getNodeName();

    const auto         transposeOut = createTensor(SizeVector {32, 128, batch0, batch1}, syn_type_single);
    synTransposeParams transposeParams {.permutation = {TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch},
                                        .tensorDim   = 4};

    const auto transpose = NodeFactory::createNode({bgemm0Out},
                                                   {transposeOut},
                                                   &transposeParams,
                                                   NodeFactory::transposeNodeTypeName,
                                                   "transpose");
    ASSERT_TRUE(transpose != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, transpose)) << "Failed adding node: " << transpose->getNodeName();

    const auto reshapeOut = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto reshape =
        NodeFactory::createNode({transposeOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add1In1 = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto add1Out = createTensor(SizeVector {64, 64, batch0, batch1}, syn_type_single);
    const auto add1    = NodeFactory::createNode({reshapeOut, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();
    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler = std::make_unique<MmeBundler>(m_graph);

    auto initialBundles = getInitialBundles(bundler);

    // Expecting exactly bgemm seed bundled
    ASSERT_EQ(initialBundles.size(), 1);
    auto& [bundle, expanders] = initialBundles.front();
    ASSERT_TRUE(bundle->getNodes().find(bgemm0) != bundle->getNodes().end());

    // first expansion should add the producer add0
    ASSERT_TRUE(expandAndValidate(bundler, expanders, bundle, {{add0}}));
    // expecting no other expansions, no consumers should be bundled when bgemm has a producer
    ASSERT_FALSE(expandStep(bundler, bundle, expanders));
    for (const auto& n : {transpose, reshape, add1})
    {
        ASSERT_TRUE(bundle->getNodes().find(n) == bundle->getNodes().end());
    }

    const auto nBundleNodes = bundle->getNodes().size();
    ASSERT_EQ(nBundleNodes, 2);
}

//   (add0) --> (conv) --> (add2) --> (add3)
//   (add1) ------^
TEST_F(IterativeMmeBundlerTest, prefer_consumers_test)
{
    ScopedConfigurationChange enableConvBundling("ENABLE_LB_PREFER_CONSUMERS", "true");
    const SizeVector          conv2dFwdIfmSize {16, 1024, 1024, 64};
    const SizeVector          conv2dFwdWeightSize {64, 16, 4, 4};
    const SizeVector          conv2dFwdOfmSize {16,
                                       conv2dFwdIfmSize.at(1) - conv2dFwdWeightSize.at(2) + 1,
                                       conv2dFwdIfmSize.at(2) - conv2dFwdWeightSize.at(3) + 1,
                                       conv2dFwdIfmSize.at(3)};

    const SizeVector reshapeOutSize {conv2dFwdOfmSize.at(0),
                                     conv2dFwdOfmSize.at(1) * conv2dFwdOfmSize.at(2) * conv2dFwdOfmSize.at(3)};

    const auto add0In0 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0In1 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0Out = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1In1 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1Out = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto             convOut = createTensor(conv2dFwdOfmSize, syn_type_single);
    synConvolutionParamsV2 params;
    const auto             conv =
        NodeFactory::createNode({add0Out, add1Out}, {convOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(conv != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, conv)) << "Failed adding node: " << conv->getNodeName();

    const auto add2In0 = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2Out = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2    = NodeFactory::createNode({add2In0, convOut}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    ASSERT_TRUE(add2 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add2)) << "Failed adding node: " << add2->getNodeName();

    const auto add3In0 = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add3Out = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add3    = NodeFactory::createNode({add3In0, add2Out}, {add3Out}, nullptr, "add_fwd_f32", "add3");
    ASSERT_TRUE(add3 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add3)) << "Failed adding node: " << add3->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto             initialBundles = getInitialBundles(bundler);

    ASSERT_EQ(initialBundles.size(), 1) << fmt::format("Expected a single conv seed, actual: {}",
                                                       initialBundles.size());
    auto& [bundle, expanders] = initialBundles.front();

    // Expander preference is set towards consumers.
    // We expect conv seed's consumer expander to expand bundle with both add consumers.
    // Once consumer expander is exhausted, we expect the producer expanders to expand the bundle with their
    // producers
    ASSERT_TRUE(expandAndValidate(bundler, expanders, bundle, {{add2}, {add3}, {add0}, {add1}}));
    // expecting no other expansions
    ASSERT_FALSE(expandStep(bundler, bundle, expanders));
}

//   (add0) --> (conv) --> (add2) --> (add3)
//   (add1) ------^
TEST_F(IterativeMmeBundlerTest, prefer_producers_test)
{
    ScopedConfigurationChange enableConvBundling("ENABLE_LB_PREFER_CONSUMERS", "false");
    const SizeVector          conv2dFwdIfmSize {16, 1024, 1024, 64};
    const SizeVector          conv2dFwdWeightSize {64, 16, 4, 4};
    const SizeVector          conv2dFwdOfmSize {16,
                                       conv2dFwdIfmSize.at(1) - conv2dFwdWeightSize.at(2) + 1,
                                       conv2dFwdIfmSize.at(2) - conv2dFwdWeightSize.at(3) + 1,
                                       conv2dFwdIfmSize.at(3)};

    const SizeVector reshapeOutSize {conv2dFwdOfmSize.at(0),
                                     conv2dFwdOfmSize.at(1) * conv2dFwdOfmSize.at(2) * conv2dFwdOfmSize.at(3)};

    const auto add0In0 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0In1 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0Out = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1In1 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1Out = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto             convOut = createTensor(conv2dFwdOfmSize, syn_type_single);
    synConvolutionParamsV2 params;
    const auto             conv =
        NodeFactory::createNode({add0Out, add1Out}, {convOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(conv != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, conv)) << "Failed adding node: " << conv->getNodeName();

    const auto add2In0 = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2Out = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2    = NodeFactory::createNode({add2In0, convOut}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    ASSERT_TRUE(add2 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add2)) << "Failed adding node: " << add2->getNodeName();

    const auto add3In0 = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add3Out = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add3    = NodeFactory::createNode({add3In0, add2Out}, {add3Out}, nullptr, "add_fwd_f32", "add3");
    ASSERT_TRUE(add3 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add3)) << "Failed adding node: " << add3->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto             initialBundles = getInitialBundles(bundler);

    ASSERT_EQ(initialBundles.size(), 1) << fmt::format("Expected a single conv seed, actual: {}",
                                                       initialBundles.size());
    auto& [bundle, expanders] = initialBundles.front();

    // Since expander preference is set to producers, we expect
    // conv seed's producer expanders to first expand bundle with
    // bundle producers.
    // Producer expanders are exhausted after expanding bundle with both add producers.
    // Then we expect the consumer expander to expand the bundle with the remaining consumers.
    ASSERT_TRUE(expandAndValidate(bundler, expanders, bundle, {{add0}, {add1}, {add2}, {add3}}));
    // expecting no other expansions
    ASSERT_FALSE(expandStep(bundler, bundle, expanders));
}

//  (add0)-->(conv)-->(reshape)
//  (add1)---^    \-->((add3)
TEST_F(IterativeMmeBundlerTest, two_seed_consumers_first_logical_second_physical)
{
    ScopedConfigurationChange enableConvBundling("ENABLE_LB_PREFER_CONSUMERS", "true");
    const SizeVector          conv2dFwdIfmSize {16, 1024, 1024, 64};
    const SizeVector          conv2dFwdWeightSize {64, 16, 4, 4};
    const SizeVector          conv2dFwdOfmSize {16,
                                       conv2dFwdIfmSize.at(1) - conv2dFwdWeightSize.at(2) + 1,
                                       conv2dFwdIfmSize.at(2) - conv2dFwdWeightSize.at(3) + 1,
                                       conv2dFwdIfmSize.at(3)};

    const SizeVector reshapeOutSize {conv2dFwdOfmSize.at(0),
                                     conv2dFwdOfmSize.at(1) * conv2dFwdOfmSize.at(2) * conv2dFwdOfmSize.at(3)};

    const auto add0In0 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0In1 = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0Out = createTensor(conv2dFwdIfmSize, syn_type_single);
    const auto add0    = NodeFactory::createNode({add0In0, add0In1}, {add0Out}, nullptr, "add_fwd_f32", "add0");
    ASSERT_TRUE(add0 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add0)) << "Failed adding node: " << add0->getNodeName();

    const auto add1In0 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1In1 = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1Out = createTensor(conv2dFwdWeightSize, syn_type_single);
    const auto add1    = NodeFactory::createNode({add1In0, add1In1}, {add1Out}, nullptr, "add_fwd_f32", "add1");
    ASSERT_TRUE(add1 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add1)) << "Failed adding node: " << add1->getNodeName();

    const auto             convOut = createTensor(conv2dFwdOfmSize, syn_type_single);
    synConvolutionParamsV2 params;
    const auto             conv =
        NodeFactory::createNode({add0Out, add1Out}, {convOut}, &params, NodeFactory::convolutionNodeTypeName, "conv");
    ASSERT_TRUE(conv != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, conv)) << "Failed adding node: " << conv->getNodeName();

    const auto reshapeOut = createTensor(reshapeOutSize, syn_type_single);

    const auto reshape =
        NodeFactory::createNode({convOut}, {reshapeOut}, nullptr, NodeFactory::reshapeNodeTypeName, "reshape");
    ASSERT_TRUE(reshape != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, reshape)) << "Failed adding node: " << reshape->getNodeName();

    const auto add2In0 = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2Out = createTensor(conv2dFwdOfmSize, syn_type_single);
    const auto add2    = NodeFactory::createNode({add2In0, convOut}, {add2Out}, nullptr, "add_fwd_f32", "add2");
    ASSERT_TRUE(add2 != nullptr);
    ASSERT_TRUE(GraphEditor::addNode(m_graph, add2)) << "Failed adding node: " << add2->getNodeName();

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto             initialBundles = getInitialBundles(bundler);

    ASSERT_EQ(initialBundles.size(), 1) << fmt::format("Expected a single conv seed, actual: {}",
                                                       initialBundles.size());
    auto& [bundle, expanders] = initialBundles.front();

    // First expansion step should attempt expanding with the first consumer
    // though due to it being a logical node without additional consumers,
    // it should be discarded and the second conv output physical consumer should be taken.
    // Second expansion step should add conv input0 producer add0
    // Last expansion step should add conv input1 producer add1
    ASSERT_TRUE(expandAndValidate(bundler, expanders, bundle, {{add2}, {add0}, {add1}}));
    const auto nBundleNodes = bundle->getNodes().size();
    ASSERT_EQ(nBundleNodes, 4) << "Num bundles nodes: " << nBundleNodes;

    // expecting no other expansions
    ASSERT_FALSE(expandStep(bundler, bundle, expanders));
}

class MultiConvBundlesTest : public IterativeMmeBundlerTest
{
protected:
    SizeVector calcConv2dOutputSize(const SizeVector& ifmSize, const SizeVector& weightSize)
    {
        SizeVector outSize;
        outSize.push_back(weightSize.at(0));
        outSize.push_back(ifmSize.at(1) - weightSize.at(2) + 1);
        outSize.push_back(ifmSize.at(2) - weightSize.at(3) + 1);
        outSize.push_back(ifmSize.at(3));
        return outSize;
    }

    enum class ConvType
    {
        FWD,
        BWD_DEDX,
        BWD_DEDW
    };

    NodePtr createConv2DNode(ConvType type, TensorPtr inA, TensorPtr inB, TensorPtr out)
    {
        static unsigned    nConv2D = 0;
        std::string_view   guid;
        SizeVector         xSize;
        SizeVector         wSize;
        SizeVector         ySize;
        switch (type)
        {
            case ConvType::FWD:
                guid  = NodeFactory::convolutionNodeTypeName;
                xSize = inA ? toSizeVector(inA) : getConv2dFwdIFMSize();
                wSize = inB ? toSizeVector(inB) : getConv2dFwdWeightSize();
                ySize = out ? toSizeVector(out) : calcConv2dOutputSize(xSize, wSize);
                break;

            case ConvType::BWD_DEDX:
                guid  = NodeFactory::deDxNodeTypeName;
                ySize = out ? toSizeVector(out) : getConv2dFwdIFMSize();
                wSize = inB ? toSizeVector(inB) : getConv2dFwdWeightSize();
                xSize = inA ? toSizeVector(inA) : calcConv2dOutputSize(ySize, wSize);
                break;

            case ConvType::BWD_DEDW:
                guid  = NodeFactory::deDwNodeTypeName;
                wSize = inB ? toSizeVector(inB) : getConv2dFwdIFMSize();
                ySize = out ? toSizeVector(out) : getConv2dFwdWeightSize();
                xSize = inA ? toSizeVector(inA) : calcConv2dOutputSize(ySize, wSize);
                break;
        }
        synConvolutionParamsV2 params;

        const auto conv = NodeFactory::createNode(
            {inA ? inA : createTensor(xSize, syn_type_single), inB ? inB : createTensor(wSize, syn_type_single)},
            {out ? out : createTensor(ySize, syn_type_single)},
            &params,
            guid,
            fmt::format("{}_{}", guid, nConv2D++));
        const auto success = GraphEditor::addNode(m_graph, conv);
        HB_ASSERT(success, "Failed adding node {} [{}]", conv->getNodeName(), conv->getNodeTypeStr());
        return conv;
    }

    NodePtr createTpcAddNode(TensorPtr inA, TensorPtr inB, TensorPtr out)
    {
        static unsigned    nTpc = 0;
        SizeVector         sizes;

        // Use sizes of first existing input operand
        // Expecting at least one operand exists and assuming
        // if more than one exist, their sizes are valid
        if (inA)
        {
            sizes = toSizeVector(inA->getAllSizesInElements(), inA->getDim());
        }
        else if (inB)
        {
            sizes = toSizeVector(inB->getAllSizesInElements(), inB->getDim());
        }
        else if (out)
        {
            sizes = toSizeVector(out->getAllSizesInElements(), out->getDim());
        }
        else
        {
            HB_ASSERT(false, "Expecting at least one input operand not null");
        }
        const auto add = NodeFactory::createNode(
            {inA ? inA : createTensor(sizes, syn_type_single), inB ? inB : createTensor(sizes, syn_type_single)},
            {out ? out : createTensor(sizes, syn_type_single)},
            nullptr,
            "add_fwd_f32",
            fmt::format("add_{}", nTpc++));
        const auto success = GraphEditor::addNode(m_graph, add);
        HB_ASSERT(success, "Failed adding node {} [{}]", add->getNodeName(), add->getNodeTypeStr());
        return add;
    }

    const SizeVector& getConv2dFwdIFMSize() const { return m_conv2dFwdIfmSize; }
    const SizeVector& getConv2dFwdWeightSize() const { return m_conv2dFwdWeightSize; }

    const SizeVector m_conv2dFwdIfmSize {16, 1024, 1024, 64};
    const SizeVector m_conv2dFwdWeightSize {64, 16, 4, 4};
};

TEST_F(MultiConvBundlesTest, three_conv_fwd_with_tpc_consumer_sharing_operand_with_tpc_producers)
{
    ScopedConfigurationChange enableConvBundling("ENABLE_LAYERED_BRAIN_CONV_BUNDLING", "true");
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    const auto                sharedOperand = createTensor(getConv2dFwdWeightSize());
    const auto                conv0Ifm      = createTensor(getConv2dFwdIFMSize());
    const auto                conv1Ifm      = createTensor(getConv2dFwdIFMSize());
    const auto                conv2Ifm      = createTensor(getConv2dFwdIFMSize());
    const auto                conv0         = createConv2DNode(ConvType::FWD, conv0Ifm, sharedOperand, nullptr);
    const auto                conv1         = createConv2DNode(ConvType::FWD, conv1Ifm, sharedOperand, nullptr);
    const auto                conv2         = createConv2DNode(ConvType::FWD, conv2Ifm, sharedOperand, nullptr);
    const auto                conv0Consumer = createTpcAddNode(conv0->getOutput(0), nullptr, nullptr);
    const auto                conv1Consumer = createTpcAddNode(conv1->getOutput(0), nullptr, nullptr);
    const auto                conv2Consumer = createTpcAddNode(conv2->getOutput(0), nullptr, nullptr);
    const auto                sharedOperandProducer0 = createTpcAddNode(nullptr, nullptr, sharedOperand);
    const auto sharedOperandProducer1 = createTpcAddNode(nullptr, nullptr, sharedOperandProducer0->getInput(0));

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);
    ASSERT_EQ(initialBundles.size(), 1) << "Expecting exactly one bundle";
    auto& [bundle, expanders] = initialBundles.front();
    expandBundle(bundler, bundle, expanders);
    bundler->logGraphBundlingStatus();

    std::vector<NodePtr> expectedBundledNodes {conv0,
                                               conv1,
                                               conv2,
                                               sharedOperandProducer0,
                                               sharedOperandProducer1,
                                               conv0Consumer,
                                               conv1Consumer,
                                               conv2Consumer};
    for (const auto& n : expectedBundledNodes)
    {
        ASSERT_EQ(n->getNodeAnnotation().bundleInfo.is_set(), true);
    }
}

TEST_F(MultiConvBundlesTest, dedx_and_dedw_with_tpc_consumer_sharing_operand_with_tpc_producers)
{
    ScopedConfigurationChange enableConvBundling("ENABLE_LAYERED_BRAIN_CONV_BUNDLING", "true");
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    const auto sharedGradIn = createTensor(calcConv2dOutputSize(getConv2dFwdIFMSize(), getConv2dFwdWeightSize()));
    const auto dedx         = createConv2DNode(ConvType::BWD_DEDX, sharedGradIn, nullptr, nullptr);
    const auto dedw         = createConv2DNode(ConvType::BWD_DEDW, sharedGradIn, nullptr, nullptr);

    const auto dedxConsumer = createTpcAddNode(dedx->getOutput(0), nullptr, nullptr);
    const auto dedwConsumer = createTpcAddNode(dedw->getOutput(0), nullptr, nullptr);

    const auto sharedOperandProducer0 = createTpcAddNode(nullptr, nullptr, sharedGradIn);
    const auto sharedOperandProducer1 = createTpcAddNode(nullptr, nullptr, sharedOperandProducer0->getInput(0));

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);
    ASSERT_EQ(initialBundles.size(), 1) << "Expecting exactly one bundle";
    auto& [bundle, expanders] = initialBundles.front();
    expandBundle(bundler, bundle, expanders);
    bundler->logGraphBundlingStatus();

    std::vector<NodePtr> expectedBundledNodes {dedx,
                                               dedw,
                                               sharedOperandProducer0,
                                               sharedOperandProducer1,
                                               dedxConsumer,
                                               dedwConsumer};
    for (const auto& n : expectedBundledNodes)
    {
        ASSERT_EQ(n->getNodeAnnotation().bundleInfo.is_set(), true);
    }
}

class MultiGemmBundlesTest
: public IterativeMmeBundlerTest
, public ::testing::WithParamInterface<bool>
{
public:
    MultiGemmBundlesTest() : m_batch(GetParam()) {}

protected:
    NodePtr createTpcAddNode(TensorPtr inA, TensorPtr inB, TensorPtr out)
    {
        static unsigned nTpc    = 0;
        const auto      add     = NodeFactory::createNode({inA ? inA : createTensor(getAddSize(), syn_type_single),
                                                  inB ? inB : createTensor(getAddSize(), syn_type_single)},
                                                 {out ? out : createTensor(getAddSize(), syn_type_single)},
                                                 nullptr,
                                                 "add_fwd_f32",
                                                 fmt::format("add_{}", nTpc++));
        const auto      success = GraphEditor::addNode(m_graph, add);
        HB_ASSERT(success, "Failed adding node {} [{}]", add->getNodeName(), add->getNodeTypeStr());
        return add;
    }

    std::vector<NodePtr> createAddChain(unsigned numAdds)
    {
        std::vector<NodePtr> chain;
        TensorPtr            prevOut;
        for (unsigned i = 0; i < numAdds; ++i)
        {
            chain.push_back(createTpcAddNode(nullptr, nullptr, prevOut ? prevOut : nullptr));
            prevOut = (i & 0x1) == 0 ? chain.back()->getInput(0) : chain.back()->getInput(1);
        }
        return chain;
    }

    NodePtr createGemmNode(TensorPtr inA, TensorPtr inB, TensorPtr out)
    {
        static unsigned nMme = 0;
        const auto      gemm =
            NodeFactory::createNode({inA ? inA : createTensor(getGemmSize(), syn_type_single),
                                     inB ? inB : createTensor(getGemmSize(), syn_type_single)},
                                    {out ? out : createTensor(getGemmSize(), syn_type_single)},
                                    nullptr,
                                    m_batch ? NodeFactory::batchGemmNodeTypeName : NodeFactory::gemmNodeTypeName,
                                    fmt::format("{}gemm_{}", m_batch ? "b" : "", nMme++));
        const auto success = GraphEditor::addNode(m_graph, gemm);
        HB_ASSERT(success, "Failed adding node {} [{}]", gemm->getNodeName(), gemm->getNodeTypeStr());
        return gemm;
    }

    const SizeVector& getAddSize() const { return m_batch ? m_addSizeWithBatch : m_addSize; }
    const SizeVector& getGemmSize() const { return m_batch ? m_batchGemmSize : m_gemmSize; }

    static constexpr TSize   m_batchDim0 = 3;
    static constexpr TSize   m_batchDim1 = 4;
    const bool               m_batch;
    const SizeVector         m_gemmSize {1024, 1024};
    const SizeVector         m_batchGemmSize {1024, 1024, m_batchDim0, m_batchDim1};
    const SizeVector         m_addSize {1024, 1024};
    const SizeVector         m_addSizeWithBatch {1024, 1024, m_batchDim0, m_batchDim1};
};

TEST_P(MultiGemmBundlesTest, gemms_with_consumer_and_producer_sharing_add_chain)
{
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    ScopedConfigurationChange limitGemmBundles("LIMIT_GEMM_BUNDLES_EXPANSION", "true");

    auto sharedAddChain   = createAddChain(6);
    auto addGemm0Producer = createTpcAddNode(nullptr, nullptr, nullptr);
    auto gemm0 = createGemmNode(addGemm0Producer->getOutput(0), sharedAddChain.front()->getOutput(0), nullptr);
    auto addGemm0Consumer = createTpcAddNode(gemm0->getOutput(0), nullptr, nullptr);
    auto addGemm1Producer = createTpcAddNode(nullptr, nullptr, nullptr);
    auto gemm1 = createGemmNode(sharedAddChain.front()->getOutput(0), addGemm1Producer->getOutput(0), nullptr);
    auto addGemm1Consumer = createTpcAddNode(gemm1->getOutput(0), nullptr, nullptr);
    auto addGemm2Producer = createTpcAddNode(nullptr, nullptr, nullptr);
    auto gemm2 = createGemmNode(sharedAddChain.front()->getOutput(0), addGemm2Producer->getOutput(0), nullptr);
    auto addGemm2Consumer = createTpcAddNode(gemm2->getOutput(0), nullptr, nullptr);

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);
    ASSERT_EQ(initialBundles.size(), 1) << "Expecting exactly one bundle";
    auto& [bundle, expanders] = initialBundles.front();
    expandBundle(bundler, bundle, expanders);

    // expecting all nodes bundled
    std::vector<NodePtr> expectedBundleNodes(sharedAddChain.begin(), sharedAddChain.end());
    expectedBundleNodes.push_back(gemm0);
    expectedBundleNodes.push_back(gemm1);
    expectedBundleNodes.push_back(gemm2);
    expectedBundleNodes.push_back(addGemm0Producer);
    expectedBundleNodes.push_back(addGemm1Producer);
    expectedBundleNodes.push_back(addGemm2Producer);

    const auto& bundleNodes = bundle->getNodes();
    const std::vector<NodePtr> expectedUnbundledNodes = {addGemm0Consumer, addGemm1Consumer, addGemm2Consumer};

    ASSERT_EQ(bundleNodes.size(), expectedBundleNodes.size());

    for (const auto& node : expectedBundleNodes)
    {
        ASSERT_TRUE(bundleNodes.find(node) != bundleNodes.end());
    }

    for (const auto& node : expectedUnbundledNodes)
    {
        ASSERT_TRUE(bundleNodes.find(node) == bundleNodes.end());
    }
    bundler->logGraphBundlingStatus();
}

TEST_P(MultiGemmBundlesTest, gemm_with_ancestor_bundled_alone)
{
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    ScopedConfigurationChange limitGemmBundles("LIMIT_GEMM_BUNDLES_EXPANSION", "true");

    auto gemm0ProducerChain = createAddChain(4);
    auto addGemm0Producer   = createTpcAddNode(nullptr, nullptr, nullptr);
    auto gemm0 = createGemmNode(addGemm0Producer->getOutput(0), gemm0ProducerChain.front()->getOutput(0), nullptr);
    auto addGemm1Producer = createTpcAddNode(nullptr, nullptr, nullptr);
    auto gemm1            = createGemmNode(gemm0->getOutput(0), addGemm1Producer->getOutput(0), nullptr);
    auto addGemm1Consumer = createTpcAddNode(gemm1->getOutput(0), nullptr, nullptr);

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);
    ASSERT_EQ(initialBundles.size(), 2) << "Expecting exactly two bundles";

    // expand bundles
    for (auto& [bundle, expanders] : initialBundles)
    {
        expandBundle(bundler, bundle, expanders);
    }
    bundler->logGraphBundlingStatus();

    BundlePtr gemm0Bundle, gemm1Bundle;
    if (initialBundles.at(0).first->getNodes().size() > initialBundles.at(1).first->getNodes().size())
    {
        gemm0Bundle = initialBundles.at(0).first;
        gemm1Bundle = initialBundles.at(1).first;
    }
    else
    {
        gemm0Bundle = initialBundles.at(1).first;
        gemm1Bundle = initialBundles.at(0).first;
    }

    std::vector<NodePtr> expectedBundle0Nodes(gemm0ProducerChain.begin(), gemm0ProducerChain.end());
    expectedBundle0Nodes.push_back(addGemm0Producer);
    expectedBundle0Nodes.push_back(gemm0);
    const auto& gemm0BundleNodes = gemm0Bundle->getNodes();
    ASSERT_EQ(gemm0BundleNodes.size(), expectedBundle0Nodes.size());
    for (const auto& n : expectedBundle0Nodes)
    {
        ASSERT_TRUE(gemm0BundleNodes.find(n) != gemm0BundleNodes.end());
    }

    std::vector<NodePtr> expectedBundle1Nodes({gemm1, addGemm1Producer});
    const auto&          gemm1BundleNodes = gemm1Bundle->getNodes();
    ASSERT_EQ(gemm1BundleNodes.size(), expectedBundle1Nodes.size());
    for (const auto& n : expectedBundle1Nodes)
    {
        ASSERT_TRUE(gemm1BundleNodes.find(n) != gemm1BundleNodes.end());
    }

    // batch gemm with a producer shouldn't bundle a consumer
    ASSERT_TRUE(gemm1Bundle->getNodes().find(addGemm1Consumer) == gemm1Bundle->getNodes().end());
}

TEST_P(MultiGemmBundlesTest, multi_gemms_with_cycles)
{
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    auto sharedInput0 = createTensor(getGemmSize(), syn_type_single);
    auto sharedInput1 = createTensor(getGemmSize(), syn_type_single);
    auto sharedInput2 = createTensor(getGemmSize(), syn_type_single);

    auto gemm0 = createGemmNode(sharedInput0, sharedInput2, nullptr);
    auto gemm1 = createGemmNode(sharedInput2, sharedInput1, nullptr);
    auto gemm2 = createGemmNode(sharedInput1, gemm0->getOutput(0), nullptr);
    auto gemm3 = createGemmNode(sharedInput0, gemm1->getOutput(0), nullptr);

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler        = std::make_unique<MmeBundler>(m_graph);
    auto       initialBundles = getInitialBundles(bundler);
    ASSERT_EQ(initialBundles.size(), 3) << "Expecting exactly 3 bundle";

    // expand bundles, in this case involves only accepting the candidates
    for (auto& [bundle, expanders] : initialBundles)
    {
        expandBundle(bundler, bundle, expanders);
    }
    bundler->logGraphBundlingStatus();
    const NodeSet gemms = {gemm0, gemm1, gemm2, gemm3};
    // expecting all gemms bundled
    ASSERT_TRUE(std::all_of(gemms.begin(), gemms.end(), [](const auto& gemm) {
        return gemm->getNodeAnnotation().bundleInfo.is_set();
    }));

    // expecting one multi MME bundle (gemm0, gemm3) and the rest gemm1 & gemm2 as single MME bundles
    ASSERT_EQ(gemm0->getNodeAnnotation().bundleInfo->bundleIndex, gemm3->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_NE(gemm0->getNodeAnnotation().bundleInfo->bundleIndex, gemm1->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_NE(gemm1->getNodeAnnotation().bundleInfo->bundleIndex, gemm2->getNodeAnnotation().bundleInfo->bundleIndex);
}

TEST_P(MultiGemmBundlesTest, multi_gemms_with_cycles_fwd_progress)
{
    ScopedConfigurationChange enableMultiMmeSeeds("ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS", "true");
    auto sharedInput0 = createTensor(getGemmSize(), syn_type_single);
    auto sharedInput1 = createTensor(getGemmSize(), syn_type_single);
    auto sharedInput2 = createTensor(getGemmSize(), syn_type_single);

    auto gemm0 = createGemmNode(sharedInput0, sharedInput2, nullptr);
    auto gemm1 = createGemmNode(sharedInput2, sharedInput1, nullptr);
    auto gemm2 = createGemmNode(sharedInput1, gemm0->getOutput(0), nullptr);
    auto gemm3 = createGemmNode(sharedInput0, gemm1->getOutput(0), nullptr);

    gaudi2::loadTpcKernels(m_graph);
    const BundlerPtr bundler   = std::make_unique<MmeBundler>(m_graph);
    const auto bundleMap = bundler->generateBundles();
    ASSERT_EQ(bundleMap.size(), 3);

    // expecting all gemms bundled
    for (const auto& gemm : {gemm0, gemm1, gemm2, gemm3})
    {
        ASSERT_TRUE(gemm->getNodeAnnotation().bundleInfo.is_set());
    }

    // expecting one multi MME bundle (gemm0, gemm3) and the rest gemm1 & gemm2 as single MME bundles
    ASSERT_EQ(gemm0->getNodeAnnotation().bundleInfo->bundleIndex, gemm3->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_NE(gemm0->getNodeAnnotation().bundleInfo->bundleIndex, gemm1->getNodeAnnotation().bundleInfo->bundleIndex);
    ASSERT_NE(gemm1->getNodeAnnotation().bundleInfo->bundleIndex, gemm2->getNodeAnnotation().bundleInfo->bundleIndex);
}

INSTANTIATE_TEST_SUITE_P(, MultiGemmBundlesTest, ::testing::Bool() /* BATCH/GEMM*/);

class AttentionBundlesTest : public LayeredBrainTest
{
public:
    AttentionBundlesTest() : m_halSetter(&m_graph) {}

protected:
    void reshape(const TensorPtr& t, const std::vector<TSize>& shape) const { t->reshape(shape.size(), shape.data()); }

    // Replaces the 'idx'th node in the chain with a MME node
    void replaceWithMME(size_t idx, TSize w, TSize h, TSize cd)
    {
        NodePtr n = m_nodeChain.at(idx);
        ASSERT_EQ(n->getNumInputs(), 1);
        ASSERT_EQ(n->getNumOutputs(), 1);
        TensorPtr a   = n->getInput(0);
        TensorPtr b   = newTensor();
        TensorPtr out = n->getOutput(0);
        reshape(a, {cd, h});
        reshape(b, {w, cd});
        reshape(out, {w, h});
        synGEMMParams params;
        NodePtr       mme =
            NodeFactory::createNode({a, b}, {out}, &params, NodeFactory::gemmNodeTypeName, fmt::format("gemm{}", idx));
        m_nodeChain[idx]  = mme;
        ASSERT_TRUE(GraphEditor::replaceNodes(m_graph, {n}, {mme}) == REPLACE_NODE_SUCCESS);
    }

    bool isNodeInBundle(const NodePtr& n, const NodeVector& bundleNodes) const
    {
        return std::find(bundleNodes.begin(), bundleNodes.end(), n) != bundleNodes.end();
    }

    CompilationHalReaderSetter m_halSetter;
};

TEST_F(AttentionBundlesTest, create_attention_bundles)
{
    setGlobalConfForTest(GCFG_ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS, "True");

    // tpc0->mme1->tpc2->mme3->mme4->mme5->tpc6->mme7->mme8->tpc9
    createGraph(10);
    reshape(m_nodeChain.at(0)->getInput(0), {64, 128});
    replaceWithMME(1, 128, 128, 64);  // Larger output to comply with MME_OUTPUTS_LARGER_THAN_INPUTS rule
    replaceWithMME(3, 64, 128, 128);  // Larger input to comply with MME_INPUTS_LARGER_THAN_OUTPUTS rule
    replaceWithMME(4, 64, 128, 64);
    replaceWithMME(5, 128, 128, 64);  // Larger output to comply with MME_OUTPUTS_LARGER_THAN_INPUTS rule
    replaceWithMME(7, 64, 128, 128);  // Larger input to comply with MME_INPUTS_LARGER_THAN_OUTPUTS rule
    replaceWithMME(8, 128, 128, 64);

    BPGraphContext bpgCtx(m_graph);
    MmeBundler     bundler(m_graph);
    const auto&    allBundles = bundler.generateBundles();

    // Expected 4 bundles:
    // mme1->tpc2->mme3
    // mme4
    // mme5->tpc6->mme7
    // mme8->tpc9
    ASSERT_EQ(allBundles.size(), 4);

    for (const auto& bundle : allBundles)
    {
        const auto& bundleNodes = bundle.second;
        if (isNodeInBundle(m_nodeChain.at(1), bundleNodes))  // mme1->tpc2->mme3
        {
            ASSERT_EQ(bundleNodes.size(), 3);
            ASSERT_TRUE(isNodeInBundle(m_nodeChain.at(2), bundleNodes));
            ASSERT_TRUE(isNodeInBundle(m_nodeChain.at(3), bundleNodes));
        }
        else if (isNodeInBundle(m_nodeChain.at(4), bundleNodes))  // mme4
        {
            ASSERT_EQ(bundleNodes.size(), 1);
        }
        else if (isNodeInBundle(m_nodeChain.at(5), bundleNodes))  // mme5->tpc6->mme7
        {
            ASSERT_EQ(bundleNodes.size(), 3);
            ASSERT_TRUE(isNodeInBundle(m_nodeChain.at(6), bundleNodes));
            ASSERT_TRUE(isNodeInBundle(m_nodeChain.at(7), bundleNodes));
        }
        else if (isNodeInBundle(m_nodeChain.at(8), bundleNodes))  // mme8->tpc9
        {
            ASSERT_EQ(bundleNodes.size(), 2);
            ASSERT_TRUE(isNodeInBundle(m_nodeChain.at(9), bundleNodes));
        }
    }
}