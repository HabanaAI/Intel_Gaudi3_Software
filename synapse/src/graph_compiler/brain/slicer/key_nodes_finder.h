#pragma once

#include "common_tile_size_calculator.h"
#include "synapse_common_types.h"
#include "types.h"
#include "brain_data.h"

namespace gc::layered_brain
{
// The key node finder is responsible to identify the key operations and order them according to cost from high to low.
// This enables the key node solver to first decide on a set of strategies for the costliest operation,
// before considering smaller operations.
class KeyNodesFinder
{
public:
    explicit KeyNodesFinder(const HabanaGraph& graph, const NodeVector& bundleNodes)
    : m_graph(graph), m_bundleNodes(bundleNodes)
    {
    }
    virtual NodeVector getSortedKeyNodes() = 0;
    virtual ~KeyNodesFinder() {}

protected:
    const HabanaGraph& m_graph;
    const NodeVector   m_bundleNodes;
};

// Implementation for multi MME bundles - order the nodes according to existing consumers within the
// bundle and the total work, from high to low.
// 1. Existing consumers: we prefer to allow the operations that have more consumers in the bundle to first decide on
//    strategies. The reason is that, mostly, strategies would prefer not to slice on common dimension if it is not
//    huge and would rather “spatial” slicing on other dimensions.
//    This kind of slicing is preferable for pipelining content to the consumers of the MME operation.
// 2. Work size: to break a tie between MME operations, if the first criteria doesn’t help, total work size should be
//    used to decide on operations order.
class MultiMmeBundleKeyNodesFinder : public KeyNodesFinder
{
public:
    explicit MultiMmeBundleKeyNodesFinder(const HabanaGraph& graph, const NodeVector& bundleNodes)
    : KeyNodesFinder(graph, bundleNodes)
    {
    }
    NodeVector getSortedKeyNodes() override;

private:
    uint64_t getOperandsTotalSizeElements(const NodePtr& node) const;
    unsigned getNumConsumersInBundle(const NodePtr& node) const;
};

}  // namespace gc::layered_brain