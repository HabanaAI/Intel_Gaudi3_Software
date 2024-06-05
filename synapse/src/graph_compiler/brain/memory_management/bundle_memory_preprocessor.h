#pragma once

#include "habana_graph.h"
#include "bundle_memory_manager.h"

namespace gc::layered_brain
{
class POCBundleMemoryPreProcessor : public BundleMemoryPreProcessor
{
public:
    POCBundleMemoryPreProcessor(HabanaGraph& graph, const BundleNodes& bundle);

    MemoryUsageDB buildMemUsageDB() override;

private:
    using Slice                  = TensorPtr;
    using InboundAliases         = std::vector<Slice>;
    using InboundAliasesPerSlice = std::unordered_map<Slice, InboundAliases>;

    HabanaGraph&       m_graph;
    const BundleNodes& m_nodes;

    MemoryUsageDB m_db;

    void addStep(size_t step);
    void addStepEntry(size_t step);
    void updateSliceEntries(size_t step);
    void processInput(size_t step, const Slice& input);
    void processOutput(size_t step, const Slice& output);

    void                   processAliases();
    InboundAliasesPerSlice buildAliasForest();
    void                   aggregateConsumersAndAliases(const Slice& slice, InboundAliasesPerSlice& aliasTrees);
    void                   accumulateConsumers(const Slice& alias, const Slice& targetSlice);
    void                   accumulateAndMoveAliases(const Slice& alias, const Slice& targetSlice);

    MemoryUsageDB::SliceEntry::Properties& properties(const Slice& slice) { return m_db.slices[slice].properties; }

    const MemoryUsageDB::SliceEntry::Properties& properties(const Slice& slice) const
    {
        return m_db.slices.at(slice).properties;
    }

    void processExternalInteractions();
    void processExternalInteractions(const Slice& slice);
    void processExternalProducer(const TensorPtr& slice, const NodePtr& producer);
    void processExternalConsumer(const TensorPtr& slice, const NodePtr& consumer);

    bool isJoin(const NodePtr& node) const;
    bool isFork(const NodePtr& node) const;
    bool isInBundle(const NodePtr& node) const;
    bool isIntermediate(const Slice& slice) const;

    // Printing
    void logDB() const;
    void logSlice(const Slice& s, int index) const;
    static void logAliasForest(const InboundAliasesPerSlice& forest);
};

}  // namespace gc::layered_brain