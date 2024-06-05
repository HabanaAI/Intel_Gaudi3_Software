#include "vtune_stat.h"
#include "graph_compiler/smf/index_space_utils.h"

namespace gaudi
{
void tpcIndexSpaceShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();

    return tpcIndexSpaceSmfHelper(params, outputs, false);
}
}
