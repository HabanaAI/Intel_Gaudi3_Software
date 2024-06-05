#pragma once

#include "include/mme_common/mme_common_enum.h"
#include "include/sync/overlap.h"

namespace MmeCommon
{
struct MmeRecipe;

template<typename Desc>
class CommonRoiCalculator
{
public:
    CommonRoiCalculator(const MmeCommon::MmeRecipe& recipe,
                        const MmeCommon::MmeLayerParams& params,
                        const std::optional<std::shared_ptr<std::vector<MmeCommon::MmeLayerParams>>>& paramsVec)
    : m_recipe(&recipe), m_params(&params)
    {
        if (paramsVec.has_value())
        {
            m_paramsVec = paramsVec.value();
        }
    }
    virtual ~CommonRoiCalculator() = default;
    void createRoi(uint64_t addr,
                   MmeActivation<Desc>& activation,
                   const MmeCommon::EMmeOperand operand,
                   bool isSram,
                   bool squashIORois) const;
    virtual void setSkipRoiCalc(const MmeCommon::CommonGeoAttr& geoAttr, MmeActivation<Desc>& act) const {}
    void resetRecipe(const MmeCommon::MmeRecipe& recipe) { m_recipe = &recipe; }
    void resetParams(const MmeCommon::MmeLayerParams& params) { m_params = &params; }
    void resetParamsVec(const std::shared_ptr<std::vector<MmeLayerParams>>& paramsVec) { m_paramsVec = paramsVec; }

protected:
    const MmeCommon::MmeRecipe* m_recipe = nullptr;
    const MmeCommon::MmeLayerParams* m_params = nullptr;
    std::shared_ptr<std::vector<MmeLayerParams>> m_paramsVec;

    OverlapRoi& operand2Roi(MmeActivation<Desc>& activation, const MmeCommon::EMmeOperand operand) const;
    static const SkipData&
    getSkipDataForOperand(const EMmeOperand operand, const MmeActivation<Desc>& act, const EMmeOpType opType);
    virtual void addSimulatedTensor(MmeCommon::EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const;
    void addSplitTensor(MmeCommon::EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const;
    void addWholeTensor(MmeCommon::EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const;
    void addWholeStridedTensor(MmeCommon::EMmeOperand operand, MmeActivation<Desc>& act, OverlapRoi& roi) const;
    void addTensorSegments(MmeCommon::EMmeOperand operand,
                           MmeActivation<Desc>& act,
                           OverlapRoi& roi,
                           SizeArray sizes,
                           SizeArray bases) const;

    uint64_t
    getOperandSizeInBytes(MmeCommon::EMmeOperand operand, bool primaryTensor, const MmeLayerParams& params) const;
    virtual bool isStoreEn(MmeActivation<Desc>& act) const;
};
}  // namespace MmeCommon
