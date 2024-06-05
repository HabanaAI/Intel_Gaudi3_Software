#include "smf/smf.h"

#include "defs.h"
#include "gaudi2_tpc_tid_metadata.h"
#include "graph_compiler/smf/index_space_utils.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "math_utils.h"
#include "vtune_stat.h"

#include <algorithm>


namespace gaudi2
{

void tpcIndexSpaceGaudi2SMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();

    tpc_sm_params_gaudi2_t* metadata = static_cast<tpc_sm_params_gaudi2_t*>(params->metadata);
    auto indexSpaceCopy = metadata->m_indexSpaceCopy;

    bool needPatching = false;

    for (unsigned i = 0; i < DYNAMIC_EXECUTE_MAX_DIMENSIONS; ++i)
    {
        if ((1U<<i) & metadata->m_dimensions_mask)
        {
             LOG_TRACE_DYNAMIC_PATCHING("INDEX SPACE: running for TID dimension {}", i);
             ShapeManipulationParams subSMFParams = *params;
             subSMFParams.metadata = &metadata->m_dimensions[i];
             subSMFParams.inPatchValuesNr = 1;

             ShapeManipulationOutputs subSMFOutputs = *outputs;
             subSMFOutputs.outputPatchValues = &indexSpaceCopy.grid_size[i];

             tpcIndexSpaceSmfHelper(&subSMFParams, &subSMFOutputs, true);

             needPatching = needPatching || (subSMFOutputs.outPatchValuesNr > 0);

             LOG_TRACE_DYNAMIC_PATCHING("INDEX SPACE: end run for TID dimension {}, need patching: {}", i, subSMFOutputs.outPatchValuesNr > 0);
        }
        else
        {
             LOG_TRACE_DYNAMIC_PATCHING("INDEX SPACE: skipping TID dimension {}", i);
        }
    }

    if (needPatching)
    {
        LOG_TRACE_DYNAMIC_PATCHING("INDEX SPACE: patched all");
        memcpy(outputs->outputPatchValues, &indexSpaceCopy, sizeof(indexSpaceCopy));
        outputs->outPatchValuesNr = sizeof(indexSpaceCopy)/sizeof(uint32_t);
    }
    else
    {
        LOG_TRACE_DYNAMIC_PATCHING("INDEX SPACE: patched none");
        outputs->outPatchValuesNr = 0;
    }
}

}  // namespace gaudi2
