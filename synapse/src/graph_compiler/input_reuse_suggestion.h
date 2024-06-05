#pragma once

#include "input_reuse.h"

class InputInplaceReuseSuggestion : public InputInplaceReuse
{
protected:
    bool outputViableForInplace(const HabanaGraph& g, const NodePtr& node, const TensorPtr& nodeOutput) const override;
    bool viableInputCandidate(const HabanaGraph& g,
                              const NodePtr&     node,
                              const TensorPtr&   input,
                              const TensorPtr&   output) const;
    bool          outputMemcopyRequired(const HabanaGraph& g, const NodePtr& node, const TensorPtr& output) const;
    bool          inputMemcopyRequired(const HabanaGraph& g,
                                       const NodePtr&     node,
                                       const TensorPtr&   input,
                                       const TensorPtr&   output) const;
    bool isMemoryConsumedAfterNode(const HabanaGraph& g,
                                   const NodePtr&     node,
                                   const TensorPtr&   input) const;
    bool areTensorsMultibuffered(const TensorPtr& input, const TensorPtr& output) const;
    bool isDiscarded(const HabanaGraph& g, const TensorPtr& t) const;
    bool applyReuse(HabanaGraph&        g,
                    const NodePtr&      node,
                    const TensorPtr&    nodeOutput,
                    const TensorVector& reuseCandidates) override;
    ReusePairsMap getReusePairs(const NodePtr& node) override;
};
