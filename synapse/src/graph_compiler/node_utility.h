#pragma once

#include "habana_device_types.h"
#include "node_roi.h"
#include "types.h"

//  A utility class for managing and orchestrating generic operations on nodes within a data structure (codeGen/graph).
class NodeUtility
{
public:
    HabanaDeviceType getNodeDeviceType(const NodePtr& node) const;
    NodePtr          getNodeSharedPtr(const Node& node) const;

private:
};
