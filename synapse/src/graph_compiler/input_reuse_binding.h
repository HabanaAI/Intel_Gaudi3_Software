#pragma once

#include "input_reuse.h"

class InputInplaceReuseBinding : public InputInplaceReuse
{
protected:
    bool outputViableForInplace(const HabanaGraph& g, const NodePtr& node, const TensorPtr& nodeOutput) const override;
    bool          applyReuse(HabanaGraph&        g,
                             const NodePtr&      node,
                             const TensorPtr&    nodeOutput,
                             const TensorVector& reuseCandidates) override;
    ReusePairsMap getReusePairs(const NodePtr& node) override;

    void validateOtherConsumerOrder(const HabanaGraph& g, const NodePtr& node, const NodeSet& consumers);
};
