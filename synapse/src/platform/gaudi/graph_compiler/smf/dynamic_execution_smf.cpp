#include <defs.h>
#include <algorithm>
#include "vtune_stat.h"
#include "graph_compiler/types.h"
#include "smf.h"

namespace gaudi
{

void dynamicExecutionShapeManipulationFunction(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();
    auto& metadata = *static_cast<dynamic_execution_sm_params_t*>(params->metadata);
    RoiIntersectionType intersection;
    if (metadata.num_projections == 0)
    {
        intersection = getIntersectionTypeFromAllTensors(params);
    }
    else
    {
        intersection = getIntersectionTypeFromProjection(params);
    }

    if (intersection == COMPLETELY_OUTSIDE)
    {
        // The node is disabled
        // Return our data to patch the registers
        // We assume the caller knows how many items to allocate
        HB_ASSERT(params->inPatchValuesNr >= metadata.cmd_len,
               "Output patch values buffer too small");

        outputs->outPatchValuesNr = metadata.cmd_len;
        std::copy(metadata.commands,
                  metadata.commands + metadata.cmd_len,
                  outputs->outputPatchValues);

        // We are using value 0 as always false, 1 as alwas true
        //  and 2 as true on completley outside but false otherwise.
        outputs->outputShouldBypass = (metadata.should_bypass == ENABLE_BYPASS) || (metadata.should_bypass == ENABLE_BYPASS_ONLY_COMPLETLY_OUT);

        LOG_TRACE_DYNAMIC_PATCHING("SMF_DYNAMIC_EXE completely outside");
    }
    else
    {
        // The node is enabled, we don't need to patch anything
        // Return zero as the length of patched data
        outputs->outPatchValuesNr = 0;
        // When getting not ENABLE_BYPASS we alway want to patch.
        outputs->outputShouldBypass = (intersection == COMPLETELY_INSIDE) && (metadata.should_bypass == ENABLE_BYPASS);

        LOG_TRACE_DYNAMIC_PATCHING("SMF_DYNAMIC_EXE completely {}",
                                   intersection == COMPLETELY_INSIDE ? "completly inside" : "intersects");
    }
}

}  // namespace gaudi
