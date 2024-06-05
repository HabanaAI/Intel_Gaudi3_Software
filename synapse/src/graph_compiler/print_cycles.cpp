#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "graph.h"
#include "log_manager.h"
#include "node.h"
#include "types.h"

#define UNDEFINED (-1)

struct vertexWithIndex
{
    NodePtr node;
    int     index;
    int     lowlink;
    bool    onStack;

    vertexWithIndex(NodePtr n)
    {
        node    = n;
        index   = UNDEFINED;
        lowlink = UNDEFINED;
        onStack = false;
    }
};

typedef std::shared_ptr<vertexWithIndex>                        vertexWithIndexPtr;
typedef std::shared_ptr<std::unordered_set<vertexWithIndexPtr>> stronglyConnectedComponent;

bool strongConnect(const vertexWithIndexPtr&                        v,
                   std::stack<vertexWithIndexPtr>&                  stack,
                   std::unordered_map<NodePtr, vertexWithIndexPtr>& map,
                   const Graph*                                     g,
                   std::unordered_set<stronglyConnectedComponent>&  sccs,
                   unsigned* const                                  globalIndex)
{
    v->index   = *globalIndex;
    v->lowlink = *globalIndex;
    ++(*globalIndex);
    stack.push(v);
    v->onStack     = true;
    auto consumers = g->getNodeConsumers(v->node, Node::TENSOR_TYPE_ALL, false);
    bool ret       = false;
    for (const auto& c : consumers)
    {
        // use the map to fetch the index from the consumer NodePtr.
        auto w = map[c];
        if (w->index == UNDEFINED)
        {
            // recursive DFS
            ret |= strongConnect(w, stack, map, g, sccs, globalIndex);
            // read about bookkeeping in the wikipedia page
            v->lowlink = std::min(v->lowlink, w->lowlink);
        }
        else if (w->onStack)
        {
            v->lowlink = std::min(v->lowlink, w->index);
        }
    }

    // if lowlink == index, than v is a root node of a strongly connected component
    if (v->lowlink == v->index)
    {
        vertexWithIndexPtr w;

        // create a new strongly connected component (scc)
        auto scc = std::make_shared<std::unordered_set<vertexWithIndexPtr>>();
        do
        {
            // pop the stack into the scc until we reach the root node
            w = stack.top();
            stack.pop();
            w->onStack = false;
            scc->insert(w);
        } while (w != v);
        if (scc->size() > 1)
        {
            // a scc with more than one node is a cycle. save that scc.
            sccs.insert(scc);
            return true;
        }
    }

    return ret;
}

// This check is done with Tarjan's strongly connected components algorithm with complexity of O(V+E).
// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
// return true if a cycle was found in the graph.
bool Graph::printGraphCycles() const
{
    LOG_DEBUG(GC, "printGraphCycles");
    const auto&                                     nodes = this->getNodes();
    std::set<vertexWithIndexPtr>                    vertices;
    std::stack<vertexWithIndexPtr>                  stack;
    std::unordered_map<NodePtr, vertexWithIndexPtr> map;
    std::unordered_set<stronglyConnectedComponent>  sccs;

    // wrap the nodes in a struct "vertexWithIndex" that keeps track of its index.
    // map NodePtr to vertexWithIndex for easy access to the index from the node itself.
    for (const auto& n : nodes)
    {
        auto v = std::make_shared<vertexWithIndex>(n);
        vertices.emplace(v);
        map[n] = v;
    }

    unsigned globalIndex  = 0;
    bool     isCycleFound = false;
    for (const auto& v : vertices)
    {
        if (v->index == UNDEFINED)
        {
            // this is basically DFS with bookkeeping to find strongly connected components.
            isCycleFound |= strongConnect(v, stack, map, this, sccs, &globalIndex);
        }
    }

    if (isCycleFound)
    {
        for (const auto& scc : sccs)
        {
            LOG_ERR(GC, "A cycle was found in the graph! The nodes are:");
            for (const auto& v : *scc)
            {
                LOG_ERR(GC, "{}", v->node->getNodeName());
            }
        }
        return true;
    }

    LOG_DEBUG(GC, "No cycles found in the graph.");
    return false;
}