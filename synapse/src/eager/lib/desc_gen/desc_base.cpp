#include "desc_base.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"
#include "utils/general_utils.h"

namespace eager_mode
{
DescGeneratorBase::DescGeneratorBase(EagerGraph& graph, const EagerNode& node)
: m_graph(graph), m_chipType(synDeviceType2ChipType(graph.getDeviceType())), m_node(node)
{
}
}  // namespace eager_mode
