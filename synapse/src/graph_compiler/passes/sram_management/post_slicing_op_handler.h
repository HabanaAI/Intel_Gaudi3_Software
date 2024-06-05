#pragma once

#include "types.h"
#include "bundle.h"

/**
 * Given a sliced node and its sliced tensors -
 * This class updates the node parameters to produce a correct sliced operation.
 */
class PostSlicingOpHandler
{
public:
    PostSlicingOpHandler() {}
    virtual void updateSlicedNode(NodePtr pSlicedNode, const pSliceReference& sliceRef) = 0;
};

class PostSlicingConvHandler : public PostSlicingOpHandler
{
public:
    PostSlicingConvHandler();
    PostSlicingConvHandler(const OffsetArray& padBefore, const OffsetArray& padAfter);
    virtual void updateSlicedNode(NodePtr pSlicedNode, const pSliceReference& sliceRef) override;

    // padding to add to dedx node
    OffsetArray m_paddingBefore;
    OffsetArray m_paddingAfter;
};

