#include "smf/smf.h"

#include "defs.h"
#include "gaudi3_tpc_tid_metadata.h"
#include "graph_compiler/smf/index_space_utils.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "math_utils.h"
#include "vtune_stat.h"

#include <algorithm>


namespace gaudi3
{

void tpcIndexSpaceGaudi3SMF(const ShapeManipulationParams* params, ShapeManipulationOutputs* outputs)
{
    STAT_FUNCTION();

    tpc_sm_params_gaudi3_t* metadata = static_cast<tpc_sm_params_gaudi3_t*>(params->metadata);
    auto indexSpaceCopy = metadata->m_indexSpaceCopy;

    bool needPatching = false;

    for (unsigned i = 0; i < DYNAMIC_EXECUTE_MAX_DIMENSIONS; ++i)
    {
        if ((1U<<i) & metadata->m_dimensions_mask)
        {
             ShapeManipulationParams subSMFParams = *params;
             subSMFParams.metadata = &metadata->m_dimensions[i];
             subSMFParams.inPatchValuesNr = 1;

             ShapeManipulationOutputs subSMFOutputs = *outputs;
             subSMFOutputs.outputPatchValues = &indexSpaceCopy.grid_size[i];

             tpcIndexSpaceSmfHelper(&subSMFParams, &subSMFOutputs, true);

             if (subSMFOutputs.outPatchValuesNr > 0)
             {
                 needPatching = true;
                 if (indexSpaceCopy.grid_size[i] == 0)
                 {
                     // We need to thell the hw that this is a null workload.
                     // The correct way to do so is to change box_size[0] to 0.
                     // Changing grid_size and/or dim_slices to 0 doesn't work for Gaudi3.
                     indexSpaceCopy.grid_size[i] = 1;
                     indexSpaceCopy.dim_slices[i] = 1;
                     indexSpaceCopy.box_size[i] = 1;
                     indexSpaceCopy.box_size[0] = 0;
                     break;
                 }
                 else
                 {
                     indexSpaceCopy.dim_slices[i] = div_round_up(indexSpaceCopy.grid_size[i],indexSpaceCopy.box_size[i]);
                 }
             }
        }
    }

    if (needPatching)
    {
        memcpy(outputs->outputPatchValues, &indexSpaceCopy, sizeof(indexSpaceCopy));
        outputs->outPatchValuesNr = sizeof(indexSpaceCopy)/sizeof(uint32_t);
    }
    else
    {
        outputs->outPatchValuesNr = 0;
    }
}

}  // namespace gaudi2
