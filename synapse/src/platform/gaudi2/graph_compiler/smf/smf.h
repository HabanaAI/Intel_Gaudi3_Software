#pragma once

#include <cstdint>

#include "recipe.h"
#include "shape_func_registry.h"
#include "recipe_metadata.h"
#include "graph_compiler/smf/smf_utils.h"

namespace gaudi2
{
void mmeSyncObjectSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void mmeDynamicExecutionSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
}  // namespace gaudi2
