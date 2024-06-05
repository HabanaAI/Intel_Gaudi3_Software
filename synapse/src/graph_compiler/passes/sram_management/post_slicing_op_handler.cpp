#include "post_slicing_op_handler.h"
#include "conv_base_node.h"
#include "log_manager.h"
#include "slicing_utils.h"

PostSlicingConvHandler::PostSlicingConvHandler(): m_paddingBefore({0}), m_paddingAfter({0})
{}

PostSlicingConvHandler::PostSlicingConvHandler(const OffsetArray& padBefore, const OffsetArray& padAfter):
m_paddingBefore(padBefore), m_paddingAfter(padAfter)
{}

void PostSlicingConvHandler::updateSlicedNode(NodePtr pSlicedNode, const pSliceReference& sliceRef)
{
    ConvBaseNode* pConv = dynamic_cast<ConvBaseNode*>(pSlicedNode.get());
    if (pConv == nullptr)
    {
        LOG_DEBUG(BE_SLICER, "PostSlicingConvHandler is relevant only for conv nodes");
        return;
    }

    ROIPosition roiPos(SlicedOperandUtils::isFirstSlice(sliceRef), SlicedOperandUtils::isLastSlice(sliceRef));

    pConv->updateConvForXOperandROI(roiPos, m_paddingBefore, m_paddingAfter);
}

