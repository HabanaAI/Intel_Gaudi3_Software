#include "spatial_dims_mapping.h"
#include "index_space_dimensions.h"
#include "mme_assert.h"
#include <unordered_map>

namespace MmeCommon::AccessPatternDetails::Conv
{
SpatialDimsMapping::SpatialDimsMapping(unsigned spatialDimsNr) : m_spatialDimsNr(spatialDimsNr)
{
    MME_ASSERT(spatialDimsNr > 0 && spatialDimsNr <= 3, "Invalid spatial dimensions number");
}

SpatialDimsMapping::SpatialDimIndices SpatialDimsMapping::getIndices(SpatialDim spDim) const
{
    static const std::unordered_map<SpatialDim, Dim> SPATIAL_IDX_SPC_DIMS = {
        {SpatialDim::WIDTH, Conv::DIM_WIDTH},
        {SpatialDim::HEIGHT, Conv::DIM_HEIGHT},
        {SpatialDim::DEPTH, Conv::DIM_DEPTH},
        {SpatialDim::BATCH, Conv::DIM_BATCH},
    };

    static const std::unordered_map<SpatialDim, Dim> FILTER_IDX_SPC_DIMS = {
        {SpatialDim::WIDTH, Conv::DIM_FILTER_S},
        {SpatialDim::HEIGHT, Conv::DIM_FILTER_R},
        {SpatialDim::DEPTH, Conv::DIM_FILTER_Q},
        {SpatialDim::BATCH, {}},  // dummy - should not be used.
    };

    Dim spd = Dim(spDim);

    SpatialDimIndices ret {};
    ret.idxSpcDim = SPATIAL_IDX_SPC_DIMS.at(spDim);
    ret.filterDim = FILTER_IDX_SPC_DIMS.at(spDim);
    if (spd < m_spatialDimsNr)
    {
        ret.xyDim = spd + 1;  // skip C or K
        ret.wDim = spd + 2;  // skip C, K
        }
        else if (spDim == SpatialDim::BATCH)
        {
            ret.xyDim = m_spatialDimsNr + 1;  // skip C or K
            ret.wDim = {};  // dummy - should not be used.
        }
        else
        {
            MME_ASSERT(false, "Invalid spatial dimension");
        }
        return ret;
}
}  // namespace MmeCommon::AccessPatternDetails::Conv
