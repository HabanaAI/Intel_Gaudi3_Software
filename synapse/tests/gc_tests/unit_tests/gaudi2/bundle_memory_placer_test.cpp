#include "layered_brain_test.h"
#include "memory_usage_db.h"
#include "bundle_memory_placer.h"

using namespace gc::layered_brain;

class BundleMemPlacerTest : public LayeredBrainTest
{
protected:
    using Placement = MemoryUsageDB::SliceEntry::Directives::Placement;
};

class POCBundleMemPlacerTest : public BundleMemPlacerTest
{
protected:
    static constexpr unsigned BUNDLE_ID = 12;

    // Create bundle with numBundleNodes
    // Add all bundle nodes to the steps of the DB.
    // Set consuming steps and producing steps of every tensor.
    void setup(int numBundleNodes)
    {
        createGraph(numBundleNodes);
        bundleNodes(BUNDLE_ID, 0, numBundleNodes);

        for (int n = 0; n < numBundleNodes; n++)
        {
            const auto& node = m_nodeChain[n];
            m_db.steps.push_back({n, m_nodeChain[n]});
            const auto& input  = node->getInput(0);
            const auto& output = node->getOutput(0);
            m_db.slices[input].properties.consumingSteps.insert(n);
            m_db.slices[output].properties.producingStep = n;
        }
    }

    NodePtr addJoin(TensorVector slices)
    {
        synConcatenateParams params {};
        params.axis     = 0;
        const auto& out = newTensor();
        TSize       concatOutputSize =
            std::accumulate(slices.begin(), slices.end(), 0, [&](uint64_t acc, const TensorPtr& t) {
                return acc + (t ? t->getSizeInElements(params.axis) : 0);
            });
        SizeArray outSizes       = out->getAllSizesInElements();
        outSizes.at(params.axis) = concatOutputSize;
        out->reshape(out->getDim(), outSizes.data());

        NodePtr concat = NodeFactory::createNode(slices,
                                                 {out},
                                                 &params,
                                                 NodeFactory::concatenateNodeLogicalInternalTypeName,
                                                 "UnTile");
        m_nodeChain.push_back(concat);
        bundleNode(BUNDLE_ID, concat, m_nodeChain.size() - 1);
        size_t concatStep = m_db.steps.size();
        m_db.steps.push_back({concatStep, concat});
        m_db.slices[out].properties.producingStep = concatStep;

        for (TensorPtr& slice : slices)
        {
            slice->setAsAliasSubTensor(out);
            m_db.slices[slice].properties.joinedBy = concat;
            // Not adding concat as a consuming step to slice. A join is not
            // considered an internal consumer.
        }

        return concat;
    }

    void executePlacer(BundleSRAMAllocator* allocator)
    {
        POCBundleMemoryPlacer placer {m_db};
        for (auto& step : m_db.steps)
        {
            placer.placeStepSlices(step, allocator);
        }
    }

    struct OptimisticSRAMAllocator : public BundleSRAMAllocator
    {
        bool allocate(const TensorPtr& slice) override { return true; }
        void free(const TensorPtr& slice) override {}
    } m_alwaysFreeAllocator;

    struct PesimisticSRAMAllocator : public BundleSRAMAllocator
    {
        bool allocate(const TensorPtr& slice) override { return false; }
        void free(const TensorPtr& slice) override {}
    } m_alwaysOccupiedAllocator;

    struct LoggingSRAMAllocator : public BundleSRAMAllocator
    {
        struct Op
        {
            enum
            {
                ALLOC,
                FREE
            } type;
            TensorPtr slice;

            bool operator==(const Op& other) const { return type == other.type && slice == other.slice; }

            std::string str() const
            {
                return std::string("<") + (type == ALLOC ? "ALLOC" : "FREE") + ", " + slice->getName() + ">";
            }
        };
        typedef std::vector<Op> Log;
        Log                     log;

        bool allocate(const TensorPtr& slice) override
        {
            log.push_back({Op::ALLOC, slice});
            return true;
        }
        void free(const TensorPtr& slice) override { log.push_back({Op::FREE, slice}); }
    } m_loggingAllocator;

    void validateLog(const LoggingSRAMAllocator::Log& expLog) const
    {
        ASSERT_EQ(m_loggingAllocator.log.size(), expLog.size());
        for (int i = 0; i < expLog.size(); i++)
        {
            EXPECT_TRUE(expLog[i] == m_loggingAllocator.log[i]);
        }
    }

    // Validates the log and prints in case of missmatch
    void validatePrintLog(const LoggingSRAMAllocator::Log& expLog)
    {
        validateLog(expLog);
        if (HasFailure())
        {
            errPrintLog(expLog, "Expected Log");
            errPrintLog(m_loggingAllocator.log, "Actual Log");
        }
    }

    void errPrintLog(const LoggingSRAMAllocator::Log& log, const std::string& headline)
    {
        LOG_ERR(GO_TEST, "{}", headline);
        for (const auto& entry : log)
        {
            LOG_ERR(GO_TEST, "  {}", entry.str());
        }
    }

    MemoryUsageDB m_db;
};

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_set_pure_intermediate_slice_in_sram)
{
    setup(2);

    POCBundleMemoryPlacer placer {m_db};
    placer.placeStepSlices(m_db.steps[0], &m_alwaysFreeAllocator);

    const TensorPtr&           slice      = m_nodeChain[0]->getOutput(0);
    MemoryUsageDB::SliceEntry& sliceEntry = m_db.slices[slice];
    EXPECT_EQ(Placement::SRAM, sliceEntry.directives.placement);
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_set_in_hbm_when_out_of_sram)
{
    setup(2);

    POCBundleMemoryPlacer placer {m_db};
    placer.placeStepSlices(m_db.steps[0], &m_alwaysOccupiedAllocator);

    const TensorPtr&           slice      = m_nodeChain[0]->getOutput(0);
    MemoryUsageDB::SliceEntry& sliceEntry = m_db.slices[slice];
    EXPECT_EQ(Placement::HBM, sliceEntry.directives.placement);
}

// Bundle output placement can be set externally. For example, if the bundle output was placed in SRAM for RMW usage.
TEST_F(POCBundleMemPlacerTest, naive_bmp_should_not_set_bundle_output_placement)
{
    setup(2);

    POCBundleMemoryPlacer placer {m_db};
    placer.placeStepSlices(m_db.steps[1], &m_alwaysFreeAllocator);

    const TensorPtr&           slice      = m_nodeChain[1]->getOutput(0);
    MemoryUsageDB::SliceEntry& sliceEntry = m_db.slices[slice];
    EXPECT_EQ(Placement::UNSET, sliceEntry.directives.placement);
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_allocate_and_free_all_intermediate_step_operands)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    setup(4);
    const auto& t0 = m_nodeChain[0]->getOutput(0);
    const auto& t1 = m_nodeChain[1]->getOutput(0);
    const auto& t2 = m_nodeChain[2]->getOutput(0);

    executePlacer(&m_loggingAllocator);

    validateLog({
        {LoggingSRAMAllocator::Op::ALLOC, t0},  // n0: Alloc t0
        {LoggingSRAMAllocator::Op::ALLOC, t1},  // n1: Alloc t1
        {LoggingSRAMAllocator::Op::FREE, t0},   //     Free t0
        {LoggingSRAMAllocator::Op::ALLOC, t2},  // n2: Alloc t2
        {LoggingSRAMAllocator::Op::FREE, t1},   //     Free t1
        {LoggingSRAMAllocator::Op::FREE, t2},   // n3: Free t2
    });
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_free_at_the_last_consumer)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //             |             ^
    //             |             |
    //             +-------------+
    setup(4);
    const auto& t0 = m_nodeChain[0]->getOutput(0);
    const auto& t1 = m_nodeChain[1]->getOutput(0);
    const auto& t2 = m_nodeChain[2]->getOutput(0);
    m_nodeChain[2]->addInput(1, t0);
    m_db.slices[t0].properties.consumingSteps.insert(2);

    executePlacer(&m_loggingAllocator);

    // The order of free of t1 and t0 is not really enforced in any way, but for the simplicity of the testing infra, I
    // didn't add an option for validating a log with don't-care partial order. If this ever breaks, this will probably
    // have to be developed.
    validatePrintLog({
        {LoggingSRAMAllocator::Op::ALLOC, t0},  // n0: Alloc t0
        {LoggingSRAMAllocator::Op::ALLOC, t1},  // n1: Alloc t1
        {LoggingSRAMAllocator::Op::ALLOC, t2},  // n2: Alloc t2
        {LoggingSRAMAllocator::Op::FREE, t1},   //     Free t1
        {LoggingSRAMAllocator::Op::FREE, t0},   //     Free t0
        {LoggingSRAMAllocator::Op::FREE, t2},   // n3: Free t2
    });
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_skip_allocation_of_internal_aliases)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]->n3->[out]
    //             |         ^         |
    //             |         |         |
    //             +--alias--+--alias--+
    setup(4);
    const auto& t0 = m_nodeChain[0]->getOutput(0);
    const auto& t1 = m_nodeChain[1]->getOutput(0);
    const auto& t2 = m_nodeChain[2]->getOutput(0);
    t0->setAsAliasSubTensor(t1);
    t2->setAsAliasSubTensor(t1);
    // Adding more steps that consume t1, since they consume an alias to it.
    m_db.slices[t1].properties.consumingSteps.insert(3);
    m_db.slices[t1].properties.consumingSteps.insert(1);

    executePlacer(&m_loggingAllocator);

    validatePrintLog({
        // n0: nop
        // n1: Alloc t1
        {LoggingSRAMAllocator::Op::ALLOC, t1},
        // n2: nop
        // n3: Free t1
        {LoggingSRAMAllocator::Op::FREE, t1},
    });
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_allocate_aliases_to_join)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]--+
    //             |        |              |
    //             |        |              v
    //             |        +----------->concat->[out]
    //             |                       ^
    //             |                       |
    //             +-----------------------+
    setup(3);
    const auto& t0     = m_nodeChain[0]->getOutput(0);
    const auto& t1     = m_nodeChain[1]->getOutput(0);
    const auto& t2     = m_nodeChain[2]->getOutput(0);
    NodePtr     concat = addJoin({t0, t1, t2});

    executePlacer(&m_loggingAllocator);

    validatePrintLog({
        {LoggingSRAMAllocator::Op::ALLOC, t0},  // n0: Alloc t0
        {LoggingSRAMAllocator::Op::ALLOC, t1},  // n1: Alloc t1
        {LoggingSRAMAllocator::Op::FREE, t0},   //     Free t0
        {LoggingSRAMAllocator::Op::FREE, t1},   // n2: Free t1
    });
}

TEST_F(POCBundleMemPlacerTest, naive_bmp_should_handle_fwd_alias_chains)
{
    // [in]->n0->[t0]->n1->[t1]->n2->[t2]--+->n3->[t3]
    //             |        |         ^    |
    //             |        +--alias--+    v
    //             |                     concat->[out]->n4->[t4]
    //             |                       ^        |         ^
    //             |                       |        +--alias--+
    //             +-----------------------+
    // out is the bundle output
    // out is an alias of t4

    setup(4);
    const auto& t0 = m_nodeChain[0]->getOutput(0);
    const auto& t1 = m_nodeChain[1]->getOutput(0);
    const auto& t2 = m_nodeChain[2]->getOutput(0);

    t1->setAsAliasSubTensor(t2);

    NodePtr concat = addJoin({t0, t2});

    // consumer of out
    m_nodeChain.push_back(
        NodeFactory::createNode({lastTensor()}, {newTensor()}, nullptr, TPCNode::NOP_KERNEL_NAME, "nop"));
    GraphEditor::addNode(m_graph, m_nodeChain.back());
    const TensorPtr& t4 = lastTensor();

    concat->getOutput(0)->setAsAliasSubTensor(t4);

    executePlacer(&m_loggingAllocator);

    validatePrintLog({
        {LoggingSRAMAllocator::Op::ALLOC, t0},  // n0: Alloc t0
        {LoggingSRAMAllocator::Op::ALLOC, t2},  // n1: Alloc t2
        {LoggingSRAMAllocator::Op::FREE, t0},   //     Free t0
        {LoggingSRAMAllocator::Op::FREE, t2},   // n3: Free t2
    });
}
