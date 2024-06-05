#pragma once

#include "bundle_memory_manager.h"
#include "graph_editor.h"

namespace gc::layered_brain
{
class POCBundleMemoryDirectiveExecutor : public BundleMemoryDirectiveExecutor
{
public:
    POCBundleMemoryDirectiveExecutor(HabanaGraph& graph, const MemoryUsageDB& db);
    bool executeDirectivesFor(TensorPtr slice) override;

private:
    using Placement = MemoryUsageDB::SliceEntry::Directives::Placement;

    void    addSpillFor(TensorPtr slice);
    NodePtr insertSpillNode(TensorPtr slice);
    void    scheduleSpill(NodePtr spillNode, TensorPtr slice);

    void    addFillFor(TensorPtr slice);
    NodePtr insertFillNode(TensorPtr slice);
    void    scheduleFill(NodePtr fillNode, TensorPtr slice);

    void scheduleWith(NodePtr n, int step);
    void moveAliasing(TensorPtr from, TensorPtr to);

    HabanaGraph&         m_graph;
    const MemoryUsageDB& m_db;
};
}  // namespace gc::layered_brain