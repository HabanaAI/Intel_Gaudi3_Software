#include "brain_data.h"
#include "common_type_utils.h"
#include "layered_brain_test_common.h"
#include "math_utils.h"
#include "node_factory.h"
#include "gaudi3_graph.h"
#include "graph_editor.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include "cacheline_aligner.h"
#include "types.h"
#include "tensor_view_node.h"

class CachelineAlignerTest : public LayeredBrainCommonTest<Gaudi3Graph>
{
protected:
    /**
     * @brief Runs cacheline aligner on BPT producers output tensor
     */
    void test(const NodePtr& bptProducer, const BPTClonePersistenceMap& clonePersistenceMap)
    {
        ScopedConfigurationChange maxMemoryIncreaseRatio("MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO",
                                                         std::to_string(std::numeric_limits<float>::max()));
        const auto                success = GraphEditor::addNode(m_graph, bptProducer);
        HB_ASSERT(success, "Failed adding node {} [{}]", bptProducer->getNodeName(), bptProducer->getNodeTypeStr());
        gaudi3::loadTpcKernels(m_graph);
        CachelineAligner clAligner(m_graph, clonePersistenceMap);
        clAligner.run();
    }

    NodePtr createJoin(const SizeVector& size, synDataType dtype, bool persistentOutput = false) const
    {
        static unsigned nJoin = 0;
        auto            join  = std::make_shared<TensorViewNode>(createTensor(size, dtype, persistentOutput),
                                                     false,
                                                     fmt::format("join_{}", nJoin++));
        for (const TensorPtr& t :
             {createTensor(size, dtype, true /*persistent*/), createTensor(size, dtype, true /*persistent*/)})
        {
            join->addView(t, SizeVector(t->getDim(), 0));
        }
        return join;
    }

    bool isFCDStrideCLAligned(const TensorPtr& t) const
    {
        HB_ASSERT(!t->isZeroSizedDataTensor(), "Zero sized tensors cannot be aligned");
        const auto cachelineSize = getCachelineSize();
        const auto fcdStride     = t->getStrideInBytes(1);
        return (fcdStride % cachelineSize == 0) || (fcdStride == cachelineSize / 2);
    }

    TStride getAlignedFCDStrideInBytes(TStride initialFCDSize, unsigned alignTargetSize) const
    {
        return round_to_multiple(initialFCDSize, alignTargetSize);
    }

    BPTClonePersistenceMap createBPTClonePersistenceMap(const TensorPtr& bptClone, bool persistence)
    {
        BPTClonePersistenceMap cloneMap;
        cloneMap.emplace(bptClone, persistence);
        return cloneMap;
    }

    unsigned getCachelineSize() const { return CompilationHalReader::getHalReader()->getCacheLineSizeInBytes(); }
};

TEST_F(CachelineAlignerTest, unaligned_fcd_stride_handled)
{
    const SizeVector  sizes {30528, 512, 28};
    const synDataType dtype(syn_type_bf16);
    const auto        bptProducer       = createJoin(sizes, dtype, false /*persistentOutput*/);
    auto              bpt         = bptProducer->getOutput(0);
    const auto        initialFCDStride  = bpt->getStrideInBytes(1);
    const auto        expectedFCDStride = getAlignedFCDStrideInBytes(initialFCDStride, getCachelineSize());
    // sizeof(syn_type_bf16)=2B, initial FCD stride in bytes is 2*30528=61056B
    // 61056%256=128 ==> output bpt is unaligned
    // We expect the aligner to handle it.
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
    test(bptProducer, {} /*Empty bptClonePersistence map, simulate non-dry run w/o graph insulation*/);
    EXPECT_TRUE(isFCDStrideCLAligned(bpt));
    EXPECT_EQ(bpt->getStrideInBytes(1), expectedFCDStride);
}

TEST_F(CachelineAlignerTest, unaligned_fcd_stride_of_persistent_clone_orig_non_persistent_handled)
{
    const SizeVector  sizes {30528, 512, 28};
    const synDataType dtype(syn_type_bf16);
    const auto        bptProducer       = createJoin(sizes, dtype, true /*persistentOutput*/);
    auto              bpt               = bptProducer->getOutput(0);
    const auto        initialFCDStride  = bpt->getStrideInBytes(1);
    const auto        expectedFCDStride = getAlignedFCDStrideInBytes(initialFCDStride, getCachelineSize());
    // sizeof(syn_type_bf16)=2B, initial FCD stride in bytes is 2*30528=61056B
    // 61056%256=128 ==> output bpt is unaligned.
    // Simulate that although bpt seems persistent, it's actually a bpt clone and orig tensor isn't persistent.
    // ==> output bpt should be aligned
    const auto bptClonePersistenceMap(createBPTClonePersistenceMap(bpt, false));
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
    test(bptProducer, bptClonePersistenceMap);
    EXPECT_TRUE(isFCDStrideCLAligned(bpt));
    EXPECT_EQ(bpt->getStrideInBytes(1), expectedFCDStride);
}

TEST_F(CachelineAlignerTest, unaligned_fcd_stride_real_in_logical_not_handled)
{
    const SizeVector  sizes {30528, 512, 28};
    const synDataType dtype(syn_type_bf16);
    const auto        bptProducer = createJoin(sizes, dtype, false /*persistentOutput*/);
    auto              bpt         = bptProducer->getOutput(0);
    bpt->setIsRealInLogical(true);
    // sizeof(syn_type_bf16)=2B, initial FCD stride in bytes is 2*30528=61056B
    // 61056%256=128 ==> output bpt is unaligned
    // Output bpt will be unhandled due to it being real in logical
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
    test(bptProducer, {} /*Empty bptClonePersistence map, simulate non-dry run w/o graph insulation*/);
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
}

TEST_F(CachelineAlignerTest, persistent_unaligned_fcd_stride_not_handled)
{
    const SizeVector  sizes {30528, 512, 28};
    const synDataType dtype(syn_type_bf16);
    const auto        bptProducer = createJoin(sizes, dtype, true /*persistentOutput*/);
    auto              bpt         = bptProducer->getOutput(0);
    // sizeof(syn_type_bf16)=2B, initial FCD stride in bytes is 2*30528=61056B
    // 61056%256=128 ==> output bpt is unaligned
    // Output bpt will be unhandled due to it being persistent
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
    test(bptProducer, {} /*Empty bptClonePersistence map, simulate non-dry run w/o graph insulation*/);
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
}

TEST_F(CachelineAlignerTest, unaligned_fcd_stride_real_in_aliasing_not_handled)
{
    const SizeVector  sizes {30528, 512, 28};
    const synDataType dtype(syn_type_bf16);
    const auto        bptProducer = createJoin(sizes, dtype, false /*persistentOutput*/);
    auto              bpt         = bptProducer->getOutput(0);
    bpt->setIsRealInAliasing(true);
    // sizeof(syn_type_bf16)=2B, initial FCD stride in bytes is 2*30528=61056B
    // 61056%256=128 ==> output bpt is unaligned
    // Output bpt will be unhandled due to it being real in aliasing
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
    test(bptProducer, {} /*Empty bptClonePersistence map, simulate non-dry run w/o graph insulation*/);
    EXPECT_FALSE(isFCDStrideCLAligned(bpt));
}