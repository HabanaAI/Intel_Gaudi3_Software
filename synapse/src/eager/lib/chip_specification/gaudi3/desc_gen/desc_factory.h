#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"

namespace eager_mode
{
class EagerGraph;
class EagerNode;

namespace gaudi3_spec_info
{
class DescFactory
{
public:
    static DescGeneratorBasePtr createDescGenerator(EagerGraph& graph, const EagerNode& node, EngineType engineType);
};

}  // namespace gaudi3_spec_info

}  // namespace eager_mode