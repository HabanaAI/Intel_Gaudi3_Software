#include "common_agu_config.h"
#include "gaudi3/mme.h"

namespace gaudi3
{
class Gaudi3AguConfig : public MmeCommon::CommonAguConfig
{
public:
    Gaudi3AguConfig(const MmeCommon::MmeLayerParams& params,
                    const MmeCommon::CommonGeoAttr& geoAttr,
                    const MmeCommon::MmeHalReader& mmeHal,
                    unsigned mmeIdx,
                    const MmeCommon::MmeRecipe& recipe)
    : CommonAguConfig(params, geoAttr, mmeHal, mmeIdx, recipe) {};
    virtual ~Gaudi3AguConfig() = default;

    virtual void configureDescriptor(void* descPtr) override;
    void configTensor(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
    void configTensorParams(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
    void configPorts(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);

    virtual void setAssociatedDimAndSize(MmeCommon::EMmeLoopMask mask,
                                         unsigned size,
                                         unsigned dimA,
                                         unsigned dimB,
                                         unsigned dimOut,
                                         void* descPtr) override;
    virtual void setSpatialLoopSize(unsigned size, void* descPtr) override;
    virtual void setPartialHeightLoopMaskA(unsigned mask, void* descPtr) override;
    virtual void setPartialHeightLoopMaskB(unsigned mask, void* descPtr) override;

    Mme::MmeTensorDesc& getDescTensor(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
};
}  // namespace gaudi3