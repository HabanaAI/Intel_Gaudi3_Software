#pragma once

#include "node_roi.h"

class HabanaGraph;

void generateOverlapRois(TensorROIVector& tensorRois, std::list<OverlapRoi>& roiList);
bool isNodeHandlingInternalDependencies(const NodePtr& node);
bool canSignal(const NodeROI& roi);
bool shouldBlockOnControlEdges(const NodePtr& node, const HabanaGraph& g);
