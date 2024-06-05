#pragma once

#include "types.h"
#include "tile.h"
#include <optional>

namespace gc::layered_brain
{
// The node DCORE ROIs setter is responsible to split the work between the available DCOREs
// according to the selected dim, in granularity units.
// The DCORE ROIs are saved in node-annotation and later copied to the real node ROIs.
class NodeDcoreROIsSetter
{
public:
    using NodeTile = gc::access_pattern::NodeTile;

    NodeDcoreROIsSetter(const NodePtr& node, unsigned numDcores) : m_node(node), m_numDcores(numDcores) {}

    void splitToDcoreROIs(unsigned dim, TSize granularity, const std::optional<unsigned>& perforationGroup) const;
    void setDcoreROIs(const std::vector<NodeTile>&   dcoreNodeTiles,
                      const std::optional<unsigned>& perforationDim,
                      const std::optional<unsigned>& perforationGroup) const;

private:
    bool               isPerforationSupported(const std::vector<TSize>& dcoreSizes, unsigned dim) const;
    std::vector<TSize> split(TSize size) const;
    std::vector<TSize> splitByGranularity(TSize size, TSize granularity) const;
    void               setNodePerforationData(const std::optional<unsigned>& perforationDim,
                                              const std::optional<unsigned>& perforationGroup) const;
    void               validatePerforation() const;

    const NodePtr  m_node;
    const unsigned m_numDcores;
};

}  // namespace gc::layered_brain