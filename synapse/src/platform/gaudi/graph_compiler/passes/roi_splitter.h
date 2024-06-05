#pragma once

#include "../../../../graph_compiler/roi_splitter.h"

class GaudiROISplitter : public ROISplitter
{
public:
    virtual std::list<NodeROI> splitDMA(NodePtr node, HabanaGraph& g) const override;

private:
    std::list<NodeROI> splitMemsetDMA(NodePtr node, HabanaGraph& g) const;
    NodeROI createMemsetROI(NodeROI& nodeRoi, unsigned baseOffset, unsigned roiSize) const;
};