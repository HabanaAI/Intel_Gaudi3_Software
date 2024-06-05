#pragma once
#include "bundle.h"
#include "bundlizer.h"

class HabanaGraph;

// Handles reshape node that separate MME operand from TPC producer/consumer of the same operand
class ReshapeAligner
{
public:
    explicit ReshapeAligner(HabanaGraph& graph) : m_graph {graph} {}

    // In case there is reshape between the producer candidate and the stitched operand,
    // remove it from the bundle if possible (move it earlier in the graph)
    virtual bool alignProducerReshape(pBundleExpansion& expansionCandidate);

    // In case there is reshape between the stitched operand and the consumer candidate,
    // remove it from the bundle if possible (move it later in the graph)
    virtual bool alignConsumerReshape(pBundleExpansion& expansionCandidate);

protected:
    // Validations
    bool canAlignReshapes(const pBundleExpansion& expansionCandidate) const;

    // Insert reshapes for the tpc node
    void insertReshapesBeforeTheBundle(pNode &tpcNode, const pNode &reshapeNode) const;

    // Reversing the reshape if its input has a consumer or the input is a persistent tensor
    void reverseReshapeIfRequired(pNode &reshapeNode, const pTensor &mmeInput) const;

    // Deduces the index of tpc node output that is going to be replaced by mme input
    bool deduceOutputIndex(unsigned int& outputIndexToBeReplaced, pNode tpcNode, pNode reshapeNode);

    // Due to reshape removal from the bundle need to adjust the tpc node outputs shapes
    void adjustTpcNodeOutputsShape(const pNode &tpcNode, const pNode &reshapeNode, pTensor &mmeInput) const;

    // Reshape TPC inputs which are not the candidate stitched input, but have the same shape as the replaced input.
    void alignConsumerInputShapes(const pBundleExpansion& expansionCandidate);

    // Reshape TPC outputs which have the same shape as the replaced input.
    void alignConsumerOutputShapes(const pBundleExpansion& expansionCandidate);

    HabanaGraph& m_graph;
};
