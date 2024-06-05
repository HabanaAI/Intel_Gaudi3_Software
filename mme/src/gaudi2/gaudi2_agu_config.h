#include "common_agu_config.h"
#include "gaudi2/mme.h"

namespace Gaudi2
{
class Gaudi2AguConfig : public MmeCommon::CommonAguConfig
{
public:
    Gaudi2AguConfig(const MmeCommon::MmeLayerParams& params,
                    const MmeCommon::CommonGeoAttr& geoAttr,
                    const MmeCommon::MmeHalReader& mmeHal,
                    unsigned mmeIdx,
                    const MmeCommon::MmeRecipe& recipe)
    : CommonAguConfig(params, geoAttr, mmeHal, mmeIdx, recipe) {};
    virtual ~Gaudi2AguConfig() = default;

    virtual void configureDescriptor(void* descPtr) override;
    void configTensor(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
    void configTensorParams(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
    void configPorts(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand);
    void applyWorkarounds(Mme::Desc* desc);
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

    using SbIndicesVec = llvm_vecsmall::SmallVector<unsigned, 4>;
    const SbIndicesVec& getSbIndices(MmeCommon::EMmeInternalOperand operand);

private:
    void configureRouting(Mme::Desc* desc);
    void setFakeSpatialLoop(Mme::Desc* desc);
    void addVirtualDim(Mme::Desc* desc);
    void setAguReads(Mme::Desc* desc, MmeCommon::EMmeInternalOperand operand, const SbIndicesVec& sbIndices);
};
}  // namespace Gaudi2