#pragma once

#include "node.h"

class HabanaGraph;
namespace replace_ops_with_logical_ops
{
bool tryReplaceMemcopyWithIdentity(HabanaGraph& g, const NodePtr& node);
};