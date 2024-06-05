#ifndef MME__GAUDI2_MME_DESCRIPTOR_GENERATOR_H
#define MME__GAUDI2_MME_DESCRIPTOR_GENERATOR_H

#include "include/gaudi2/gaudi2_mme_signal_info.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "include/utils/gaudi2_desc_dumper.h"

namespace Gaudi2
{
using MmeAddressPair = std::pair<Mme::MmeAddress&, Mme::MmeAddress&>;
using MmeActivation = MmeCommon::MmeActivation<Mme::Desc>;
using ActivationVec = MmeCommon::ActivationVec<Mme::Desc>;

struct ReuseAttr
{
    unsigned denseStepsNr;
    unsigned denseLoopSelector;
    unsigned lastDenseStepSize;
    unsigned aPartialHeightStepsNr;
    unsigned aPartialHeightLoopSelector;
    unsigned lastAPartialHeightStepSize;

    unsigned spatialLoopSelector;
    unsigned spatialStepsNr;

    MmeCommon::EMmeLoopMask accumDimLoopMask;
    MmeCommon::EMmeLoopMask reuse2dLoopMask;  // The loop that wrap the accum-dim loop
};

class RoiCalculator;
class RecipeIterator;
// When using full ROIs and no partials:

//      All the tensor bases should be set to 0.
//      All the tensor sizes should be set to the full tensor sizes.
//      SpSize should be set to the number of the spatial elements in tensor Y (fwd & dedw) or tensor X (in dedx).
//      SpSize should be set to 0.

// In order to do partials and ROIs:

// FWD:
//      ROI - Dense dimension: y.sizes[0] & y.bases[0]. (w.sizes[0] & w.bases[0] should be adjusted)
//            Spatial base: spBase - relative to y.bases[1..n] & y.sizes[1..n]
//            Spatial size: spSize
//      Partial Channel - x.sizes[0] & x.bases[0]. (w.sizes[1] & w.bases[1] should ba adjusted)
//      Partial Filter - w.sizes[2..n] & w.bases[2..n].
//

// DEDX:
//      ROI - Dense dimension: x.sizes[0] & x.bases[0]. (w.sizes[1] & w.bases[1] should be adjusted)
//            Spatial base: SpBase - relative to x.bases[1..n] & x.sizes[1..n]
//            Spatial size: SpSize
//      Partial Channel - y.sizes[0] & y.bases[0]. (w.sizes[0] & w.bases[0] should ba adjusted)
//      Partial Filter - w.sizes[2..n] & w.bases[2..n].
//

// DEDW:
//      ROI - K: w.sizes[0] & w.bases[0]. (y.sizes[0] & y.bases[0] should be adjusted)
//            C: w.sizes[1] & w.bases[1]. (x.sizes[0] & x.bases[0] should be adjusted)
//            Filter: w.sizes[2..n] & w.bases[2..n]. (Conv.padding should be adjusted)
//      Partial spatial - y.sizes[1..n] & y.bases[1..n] & SpBase & SpSize
//
//      Note: currently all the bases must be 0 in DEDW operation
class MmeDescriptorGenerator final : public MmeCommon::MmeDescriptorGenerator<Mme::Desc>
{
public:
    MmeDescriptorGenerator()
    : MmeCommon::MmeDescriptorGenerator<Mme::Desc>() {}

    virtual ~MmeDescriptorGenerator() = default;
    void mmeGenerateActivations() override;
    /* LayerParams getter - original params are the original input from the user (GC) - not changed at all,
     * whereas params is the sub-problem params - same as originalParams but
     * they could be altered in case we have several sub-problems (see dedx with strides or dedw with unroll).
     * original params should be used in general , but when looking for tensor sizes\strides\bases - better to take
     * params.
     * In BGemm - original params and params are the same - no division to sub problems and no alteration of the
     * user layer params.
     */
    static std::unique_ptr<MmeDescriptorGenerator> createMmeDescGenerator();

    //  this function is used to set the same SO to each MME across all activations, used by mme user.
    virtual void mmePatchSyncObjects(const uint32_t mmeIdx,
                                     const uint32_t addr0,
                                     const uint32_t addr1,
                                     const uint32_t slaveAddr0 = 0,
                                     const uint32_t slaveAddr1 = 0) override;

    //  this function is used when each descriptors gets a unique SO.
    static void mmePatchSyncObject(Mme::Desc& desc,
                                   const uint32_t addr0,
                                   const uint32_t addr1,
                                   const uint32_t slaveAddr0 = 0,
                                   const uint32_t slaveAddr1 = 0);

    virtual void patchSignalColoring(MmeActivation& activation, const bool addr0isSram, const bool addr1isSram) override;

    void mmeGetDescValidMask(const Mme::Desc& desc,
                             bool* mask,
                             bool* aguOut1FromAguOut0_DW0,
                             bool* aguOut1FromAguOut0_DW1_4) override;

    virtual std::vector<std::string> getRecipeDebugInfo(bool verbose = true) const override;
    std::string getRecurringMisalignmentDebugInfo() const override;

protected:
    virtual bool validateParams(const MmeCommon::MmeLayerParams& params, std::string& errorMsg) override;

    unsigned getRollAccumsVal(const unsigned accumNrSp,
                              const unsigned accumNrDense,
                              const bool isBGemm8x = false,
                              const bool incInLast = false) const;
    Mme::EMmeShuffleAMode getShuffleAVal(const bool transA, const bool bgemm) const;
    MmeCommon::EMmeDataType getOutputDataType(const MmeCommon::EMmeDataType recipeOutputDataType) const;

    // Descriptor configurators
    void setDescRateLimiters(Mme::Desc* desc);
    void setPmuSaturation(Mme::Desc& desc);
    void setDescFp8Bias(Mme::Desc* desc) const;

    void getReuseAttr(const MmeCommon::EMmePattern pattern, const MmeCommon::EMmeOpType op, ReuseAttr* reuseAttr) const;
    virtual void setSBCacheDisable(const MmeCommon::EMmeOperand operand,
                                   const bool isSram,
                                   const bool addrIsAligned,
                                   Mme::Desc& desc) override;
    virtual void setSinglePerfEvent(MmeCommon::EMmeTraceEngine engine, unsigned startEndMask, Mme::Desc& desc) override;
    // commond descriptor configuration after specific operation is choosen.
    void commonDescriptorConfigPost(Mme::Desc& desc);
    unsigned countTetrises(const Mme::Desc* desc);

    void setFieldsFromParams(const ReuseAttr& reuseAttr, Mme::Desc* desc);
    void setEngineBrains(const ReuseAttr& reuseAttr, Mme::Desc* desc);
    void setSimpleFields(Mme::Desc* desc);
    void buildDescNew(unsigned mmeIDx, Mme::Desc* desc);

    std::string getVerboseRecipeSummaryStr() const;
    virtual std::shared_ptr<MmeDescriptorDumper<Gaudi2::Mme::Desc>> getDescriptorDumper() const override
    {
        return std::make_shared<Gaudi2DescriptorDumper>();
    }

    virtual void createRoiCalculator() override;

private:
    void createHardcodedDesc(Mme::Desc* desc, Mme::EMmeCore coreType);

protected:  // member variables
    Gaudi2SignalingInfo m_signalingInfo;
};

using pMmeDescriptorGenerator = std::unique_ptr<MmeDescriptorGenerator>;
}  // namespace Gaudi2

#endif //MME__GAUDI2_MME_DESCRIPTOR_GENERATOR_H
