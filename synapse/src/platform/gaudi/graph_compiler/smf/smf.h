#pragma once

#include <cstdint>

#include "recipe.h"

#include "graph_compiler/smf/smf_utils.h"

#include "shape_func_registry.h"

#include "recipe_metadata.h"

namespace gaudi {

void dynamicExecutionShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void dmaSizeShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void mmeValidElementsSMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void tpcSizeShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void tpcIndexSpaceShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void bulkSizeStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void lastStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void viewSizeStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void viewBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void sliceStrideShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
void sliceBaseAddressShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs);
}
