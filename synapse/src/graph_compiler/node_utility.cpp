#include "node_utility.h"
#include "node.h"
#include "types.h"

HabanaDeviceType NodeUtility::getNodeDeviceType(const NodePtr& node) const
{
    HabanaDeviceType deviceType = node->getNodeDeviceType();
    HB_ASSERT(deviceType != LAST_HABANA_DEVICE, "Unsupported node type. {}", node->getNodeName());
    return deviceType;
}

NodePtr NodeUtility::getNodeSharedPtr(const Node& node) const
{
    return const_cast<Node&>(node).shared_from_this();
}
