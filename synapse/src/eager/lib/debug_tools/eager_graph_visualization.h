#pragma once

#include <string_view>

namespace eager_mode
{
class EagerGraph;

void visualizeGraph(const EagerGraph& graph, std::string_view name);
}  // namespace eager_mode