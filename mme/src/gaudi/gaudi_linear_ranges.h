#pragma once
#include "include/gaudi/mme_agu_stat.h"
#include "include/mme_common/recipe.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "mme_common/common_linear_ranges.h"

namespace gaudi
{
class RoiCalculator : public MmeCommon::CommonRoiCalculator<Mme::Desc>
{
public:
    RoiCalculator(const MmeCommon::MmeRecipe& recipe, const MmeCommon::MmeLayerParams& params)
    : CommonRoiCalculator(recipe, params, std::nullopt)
    {
    }
    ~RoiCalculator() = default;

protected:
    virtual void addSimulatedTensor(MmeCommon::EMmeOperand operand, MmeActivation& act, OverlapRoi& roi) const override;
    virtual bool isStoreEn(MmeActivation& act) const override;
};

}  // namespace gaudi
