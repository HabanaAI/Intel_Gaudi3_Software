#include "include/gaudi/new_descriptor_generator/dedw_unroll.h"
#include "include/mme_assert.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe.h"
#include "include/mme_common/recipe_generator.h"
#include "src/gaudi/gaudi_geo_attr.h"
#include "src/gaudi/gaudi_mme_hal_reader.h"

using namespace MmeCommon;

namespace gaudi
{
DedwUnroll::DedwUnroll(const MmeLayerParams& params, const GeoAttr& geoAttr)
{
    // Make sure initial values are set by default
    m_unrollFactor = 1;
    m_unrollDim = Mme::c_mme_max_tensor_dims;

    bool hasDilation = false;
    for (unsigned dim = 0; dim < params.conv.dilation.size(); dim++)
    {
        if (params.conv.dilation[dim] != 1)
        {
            hasDilation = true;
            break;
        }
    }
    if ((params.opType != e_mme_dedw) || !params.strategy.unrollEn ||
        (params.strategy.geometry != e_mme_geometry_4wx1h) ||
        (params.getOperand(e_mme_op_c).sizes[0] > geoAttr.m_subMatrixWidth) || hasDilation)
    {
        return;  // no dedw unroll
    }

    GaudiGeoAttr commonGeoAttr(params);
    RecipeGenerator recipeGen(e_mme_conv_recipe, params, gaudi::MmeHalReader::getInstance(), commonGeoAttr);

    if (recipeGen.isPartialSBReuse() || !params.strategy.unrollEn)
    {
        return;
    }

    auto& recipe = recipeGen.get();
    // todo AlonG: Review these with Eitan
    MME_ASSERT(0 == geoAttr.m_euElemWidth % geoAttr.m_totalBports, "eu element width should be aligned to cPorts");
    const unsigned outputPortWidth = geoAttr.m_geoTotalElemWidth;  // m_euElemWidth / geoAttr.m_totalBports;
    const unsigned maxUnrollFactor = geoAttr.m_totalBports;

    // todo AlonG: extend to 2w2h geometry (SW-59129: Extend dedw unroll for 2w2h geometry in Gaudi)
    const bool shouldUnroll =
        ((params.strategy.unrollEn) && (params.strategy.geometry == e_mme_geometry_4wx1h) &&
         (recipe.getOperand(e_mme_op_c).sizes[0] <= outputPortWidth) && (maxUnrollFactor > 1) && (params.spBase == 0));

    if (!shouldUnroll)
    {
        return;
    }

    for (unsigned tensorDim = 1; tensorDim < Mme::c_mme_max_conv_dims; tensorDim++)
    {
        const unsigned weightDim = tensorDim + 1;
        const unsigned convDim = tensorDim - 1;
        if ((recipe.getOperand(e_mme_op_a).sizes[tensorDim] !=
             recipe.getOperand(e_mme_op_b).sizes[tensorDim] * params.conv.stride[convDim]) ||
            (recipe.getOperand(e_mme_op_c).sizes[weightDim] == 1))
        {
            continue;  // This conv dim isn't suitable for unroll, try next dim
        }
        unsigned currUnrollFactor =
            div_round_up(recipe.getOperand(e_mme_op_c).sizes[weightDim], params.conv.stride[convDim]);
        currUnrollFactor = std::min(currUnrollFactor, maxUnrollFactor);
        if (currUnrollFactor > 1)
        {
            m_unrollDim = tensorDim;
            m_unrollFactor = currUnrollFactor;
            break;
        }
    }
}

}  // namespace gaudi
