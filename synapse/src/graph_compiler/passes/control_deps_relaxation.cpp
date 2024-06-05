#include "habana_graph.h"
#include "graph_editor.h"

bool relaxCtrlDeps(HabanaGraph& g)
{
    // relax all control dependencies
    for (const NodePtr& n : g.getNodes())
    {
        if (n == nullptr) continue;
        GraphEditor::recalculateNodeControlDependencies(g, n);
    }
    return true;
}
