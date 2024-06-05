#pragma once

#include "types.h"
#include "node_roi.h"

class HabanaGraph;

void splitToPhysicalROIsForNode(const HabanaGraph&  graph,
                                const pNode&        node,
                                std::list<NodeROI>* rois,
                                std::list<NodeROI>& phisicalRois);