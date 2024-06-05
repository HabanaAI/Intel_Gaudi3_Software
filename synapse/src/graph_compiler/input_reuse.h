#pragma once

#include "node_factory.h"

using ReusePairsMap = std::map<TensorPtr, TensorVector, TensorComparator>;
class InputInplaceReuse
{
public:
    bool runInputInplaceReuse(HabanaGraph& g);
    static bool isAlreadyReused(const HabanaGraph& g, const TensorPtr& input, const Node& node);
    static bool isAlreadyReusedPersistentTensors(const HabanaGraph& g, const TensorPtr& input, const TensorPtr& output);

protected:
    virtual bool
    outputViableForInplace(const HabanaGraph& g, const NodePtr& node, const TensorPtr& nodeOutput) const = 0;
    virtual bool          applyReuse(HabanaGraph&        g,
                                     const NodePtr&      node,
                                     const TensorPtr&    nodeOutput,
                                     const TensorVector& reuseCandidates) = 0;
    virtual ReusePairsMap getReusePairs(const NodePtr& node)              = 0;
};