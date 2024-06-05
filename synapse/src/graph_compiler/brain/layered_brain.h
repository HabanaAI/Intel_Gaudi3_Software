#pragma once

#include "types.h"
#include "node.h"

//
// Shared definitions for all the components of the layered pipeline brain
//

namespace gc::layered_brain
{
typedef NodeVector BundleNodes;
typedef unsigned   BundleIndex;

typedef std::unordered_map<BundleIndex, BundleNodes> Bundles;

inline std::optional<BundleIndex> getBundleIndex(const NodePtr& node)
{
    const auto& bi = node->getNodeAnnotation().bundleInfo;
    return bi.is_set() ? std::optional(bi->bundleIndex) : std::nullopt;
}

}  // namespace gc::layered_brain