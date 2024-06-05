#pragma once
#include "mme_agu_simulator.h"
#include "include/mme_common/recipe.h"
#include "include/mme_common/skip_data.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "common_linear_ranges.h"
#include "src/mme_common/common_geo_attr.h"

namespace Gaudi2
{
class RoiCalculator : public MmeCommon::CommonRoiCalculator<Mme::Desc>
{
public:
    RoiCalculator(const MmeCommon::MmeRecipe& recipe, const MmeCommon::MmeLayerParams& params)
    : CommonRoiCalculator(recipe, params, std::nullopt)
    {
    }
    ~RoiCalculator() = default;

    static void countSkipSignals(ActivationVec& activations);

protected:
    virtual void addSimulatedTensor(MmeCommon::EMmeOperand operand, MmeActivation& act, OverlapRoi& roi) const override;
    virtual bool isStoreEn(MmeActivation& act) const override;

private:
    void simulateAgu(const MmeCommon::EMmeOperand operand,
                     const Mme::Desc& desc,
                     std::vector<Gaudi2::AguRanges>* ranges) const;
    unsigned mmeOperand2aguIdx(const MmeCommon::EMmeOpType opType,
                               const MmeCommon::EMmeOperand operand,
                               const Mme::Desc& desc,
                               std::array<EMmeOperandIdx, 4>& aguIdx) const;
};

class ConvRoiCalculator : public RoiCalculator
{
public:
    ConvRoiCalculator(const MmeCommon::MmeRecipe& recipe, const MmeCommon::MmeLayerParams& params)
    : RoiCalculator(recipe, params)
    {
    }
    virtual void setSkipRoiCalc(const MmeCommon::CommonGeoAttr& geoAttr, MmeActivation& act) const override;
};

class BGemmRoiCalculator : public RoiCalculator
{
public:
    BGemmRoiCalculator(const MmeCommon::MmeRecipe& recipe, const MmeCommon::MmeLayerParams& params)
    : RoiCalculator(recipe, params)
    {
    }
    virtual void setSkipRoiCalc(const MmeCommon::CommonGeoAttr& geoAttr, MmeActivation& act) const override;
};

}  // namespace Gaudi2
