#pragma once

#include "json.hpp"

class HabanaGraph;

class TpcJsonSerializer
{
public:
    static nlohmann_hcl::json serialize(const HabanaGraph& habanaGraph);
};