#ifndef MME__GAUDI_MME_DESCRIPTOR_GENERATOR_H
#define MME__GAUDI_MME_DESCRIPTOR_GENERATOR_H

#include <list>
#include <memory>
#include <vector>
#include <stdint.h>
#include "gaudi/mme.h"
#include "new_descriptor_generator/dedw_unroll.h"
#include "include/mme_common/conv_sub_problems.h"
#include "include/mme_common/recipe.h"
#include "include/mme_common/recurring_misalignment_opt.h"

namespace gaudi
{
class GeoAttr;
class GaudiGeoAttr;

MmeBatchMode getBatchGemmBatchMode(MmeCommon::EMmeOpType opType,
                                   MmeCommon::EMmeGeometry geometry,
                                   unsigned outputHeight,
                                   unsigned outputWidth,
                                   unsigned portElemWidth);  // Number of elements per port (64 for f16, 32 for float)

// When using full ROIs and no partials:

//      All the tensor bases should be set to 0.
//      All the tensor sizes should be set to the full tensor sizes.
//      SpSize should be set to the number of the spatial elements in tensor Y
//      (fwd & dedw) or tensor X (in dedx). SpSize should be set to 0.

// In order to do partials and ROIs:

// FWD:
//      ROI - Dense dimension: y.sizes[0] & y.bases[0]. (w.sizes[0] & w.bases[0]
//      should be adjusted)
//            Spatial base: spBase - relative to y.bases[1..n] & y.sizes[1..n]
//            Spatial size: spSize
//      Partial Channel - x.sizes[0] & x.bases[0]. (w.sizes[1] & w.bases[1]
//      should ba adjusted) Partial Filter - w.sizes[2..n] & w.bases[2..n].
//

// DEDX:
//      ROI - Dense dimension: x.sizes[0] & x.bases[0]. (w.sizes[1] & w.bases[1]
//      should be adjusted)
//            Spatial base: SpBase - relative to x.bases[1..n] & x.sizes[1..n]
//            Spatial size: SpSize
//      Partial Channel - y.sizes[0] & y.bases[0]. (w.sizes[0] & w.bases[0]
//      should ba adjusted) Partial Filter - w.sizes[2..n] & w.bases[2..n].
//

// DEDW:
//      ROI - K: w.sizes[0] & w.bases[0]. (y.sizes[0] & y.bases[0] should be
//      adjusted)
//            C: w.sizes[1] & w.bases[1]. (x.sizes[0] & x.bases[0] should be
//            adjusted) Filter: w.sizes[2..n] & w.bases[2..n]. (Conv.padding
//            should be adjusted)
//      Partial spatial - y.sizes[1..n] & y.bases[1..n] & SpBase & SpSize
//
//      Note: currently all the bases must be 0 in DEDW operation

void generateDescriptors(const MmeCommon::MmeLayerParams& params, std::list<MmeActivation>& activations);
void getTensorAddressFieldsAndMapping(const MmeCommon::EMmeOperand operand,
                                      Mme::Desc& desc,
                                      uint32_t*& addrHigh,
                                      uint32_t*& addrLow,
                                      const Mme::MmeTensorDesc*& tensorDesc,
                                      bool isBgemm);
void getTensorAddressFields(const MmeCommon::EMmeOperand operand,
                            Mme::Desc& desc,
                            uint32_t** addrHigh,
                            uint32_t** addrLow,
                            const bool isBgemm = false);
void getRoiBaseOffsetFields(const MmeCommon::EMmeOperand operand,
                            const MmeCommon::EMmeOpType opType,
                            Mme::Desc& desc,
                            std::vector<int32_t*>& roiBaseOffsetVec);

void patchPadding(int32_t* roiBaseOffset,
                  uint32_t* tensorAStrides,
                  uint32_t* oldConvPadding,
                  uint32_t* oldConvStrides,
                  uint32_t* oldConvDilation,
                  uint32_t* actualPadding,
                  MmeCommon::EMmeOpType opType);

void patchTensorView(const MmeCommon::EMmeOperand operand,
                     Mme::Desc& desc0,
                     Mme::Desc& desc1,
                     const uint64_t addr,
                     const bool isBgemm = false);
void patchSyncObject(Mme::Desc& desc,
                     const Mme::MmeHalf half,
                     const uint64_t addr,
                     const unsigned value = 1,
                     const bool inc = true,
                     const bool perfEvent = false);

unsigned getUserDataVal(const bool reductionEn, const MmeCommon::EMmeDataType dt, const MmeCommon::RoundingMode rm);
bool getOperandMapping(const MmeCommon::EMmeOperand operand,
                       Mme::Desc& desc,
                       const Mme::MmeTensorDesc*& tensorDesc,
                       bool isBgemm = false);

class MmeDescriptorGenerator
{
public:
    static std::unique_ptr<MmeDescriptorGenerator> createMmeDescGenerator(const MmeCommon::MmeLayerParams& params);
    virtual ~MmeDescriptorGenerator() = default;

    virtual void mmeGenerateActivations() = 0;
    std::list<MmeActivation>& getMmeActivations() { return m_activations; }
    static bool isZeroCD(const MmeCommon::MmeLayerParams& params);
    bool isSignalOverflow() { return m_signalOverflow; }
    const MmeCommon::MmeRecipe& getRecipe() const;
    std::vector<std::string> getRecipeDebugInfo(bool verbose = true) const;
    virtual std::vector<std::vector<std::string>> dumpDescriptors(bool dumpAsBinary) const;

protected:
    MmeDescriptorGenerator(const MmeCommon::MmeLayerParams& params);

    // These operations are implemented in subclasses.
    virtual void setDescHeader(Mme::Desc& desc) = 0;
    virtual void setHeaderLoopSelectors(DescGroup& descGroup) = 0;
    virtual void mmeCalcLayerRecipe() = 0;
    virtual void setDescSBReuse(DescGroup& descGroup) = 0;
    virtual void setAguConfig(DescGroup& descGroup) = 0;
    virtual void setDescLowering(DescGroup& descGroup) = 0;
    virtual void setPaddingValues(Mme::Desc& desc) = 0;

    virtual void setDescUnroll(DescList& descList) {};
    virtual void fixUnrollDesc(DescGroup& descGroup) {};

    const MmeCommon::MmeLayerParams& getParams() const { return m_curParams; }
    MmeCommon::MmeLayerParams processParams(const MmeCommon::MmeLayerParams& params);
    static bool validateParams(const MmeCommon::MmeLayerParams& params, std::string& errorMsg);

    const MmeRoi getRoi();
    const std::shared_ptr<GeoAttr> getGeoAttr() const { return m_geoParamsSPtr; }

    MmeCommon::MmeLayerParams makeParamsForDedwAsBgemm(const MmeCommon::MmeLayerParams& params);
    MmeCommon::MmeLayerParams makeParamsForZeroCD(const MmeCommon::MmeLayerParams& params);

    void setReuseAttr(const GeoAttr& geoParams, const MmeCommon::EMmeOpType op, const bool reuseA, const bool reuseB);
    void initSignalingInfo(Mme::Desc& desc);
    void resetMetaDataFields(Mme::Desc& desc);
    void buildDesc(DescGroup& descGroup);
    void setDescPerfFields();
    void resetDescPerfFields(Mme::Desc& desc);
    void setUserDataFields(Mme::Desc& desc);
    void setDescRateLimiters(Mme::Desc& desc);
    void setSignals(DescGroup& descGroup, bool isLast);
    void setActivation(MmeActivation& activation, DescGroup& descGroup);
    void buildDescTensorPointers(Mme::Desc& desc);
    bool isPatternRaster(const MmeCommon::EMmePattern pattern) const;
    void handlePartials(DescGroup& descGroup);
    void setAddressOffsets(Mme::Desc& desc);
    static bool isDedwAsBgemm(MmeCommon::MmeLayerParams params);
    void getCommonDims(MmeCommon::EMmeOpType opType, unsigned& cdDimInA, unsigned& cdDimInB) const;
    void
    params2OperandsViews(MmeCommon::MmeTensorView* a, MmeCommon::MmeTensorView* b, MmeCommon::MmeTensorView* c) const;
    const MmeCommon::MmeLayerParams& getOriginalParams() const { return m_originalParams; }
    MmeCommon::MmeLayerParams* GetVolatileParams() { return &m_curParams; }
    const void setCurParams(MmeCommon::MmeLayerParams newParams) { m_curParams = newParams; }
    void setExecParams() { m_curParams = m_originalParams; };
    void handleSignalOverflow(DescGroup& descGroup);
    unsigned countTetrises(const Mme::Desc& desc);

    bool isDedwAsBgemm() { return m_dedwAsBgemm; }
    bool isDedwAsBgemmOddCD() { return m_dedwAsBgemmOddCD; }
    void setDedwAsBgemm(bool val) { m_dedwAsBgemm = val; }
    void setDedwAsBgemmOddCD(bool val) { m_dedwAsBgemmOddCD = val; }
    void setZeroCD(bool val) { m_zeroCD = val; }
    bool getZeroCD() { return m_zeroCD; }
    std::string getVerboseRecipeSummaryStr() const;
    const std::shared_ptr<MmeCommon::CommonRoiCalculator<Mme::Desc>>& getRoiCalculator() const;
    virtual std::shared_ptr<MmeCommon::CommonRoiCalculator<Mme::Desc>> createRoiCalculator() const;
    void
    activation2Roi(const ExecuteParams& params);

    bool m_zeroCD;
    bool m_dedwAsBgemm;
    bool m_dedwAsBgemmOddCD;
    bool m_signalOverflow = false;
    const MmeCommon::MmeLayerParams m_originalParams;
    MmeCommon::MmeLayerParams m_curParams;
    std::shared_ptr<GeoAttr> m_geoParamsSPtr;
    std::shared_ptr<GaudiGeoAttr> m_commonGeoAttrSPtr;
    MmeCommon::MmeRecipe m_recipe;
    std::list<MmeActivation> m_activations;
    gaudiReuseAttr m_reuseAttr;
    MmeCommon::ConvSubProblemContainer m_convSubProblems;
    std::shared_ptr<MmeCommon::CommonRoiCalculator<Mme::Desc>> m_roiCalculator;
};

class MmeConvDescriptorGenerator : public MmeDescriptorGenerator
{
public:
    static std::unique_ptr<MmeDescriptorGenerator> createMmeConvDescGenerator(const MmeCommon::MmeLayerParams& params);
    virtual ~MmeConvDescriptorGenerator() = default;
    void mmeGenerateActivations() override;

protected:
    explicit MmeConvDescriptorGenerator(const MmeCommon::MmeLayerParams& params)
    : MmeDescriptorGenerator(params), m_dedwUnroll(params, *m_geoParamsSPtr)
    {
        mmeCalcLayerRecipe();
    }
    void mmeCalcLayerRecipe() override;
    void setAguConfig(DescGroup& descGroup) override;
    void setPaddingValues(Mme::Desc& desc) override;
    void setHeaderLoopSelectors(DescGroup& descGroup) override;

    unsigned calcTotalNumOfSubProblems(const MmeCommon::CommonGeoAttr& geoAttr) const;
    unsigned getTotalDedxNumOfDesc() const;
    unsigned getTotalDedwNumOfDesc() const;
    void makeParamsForDedxSubProblem(unsigned numOfSubProblems, unsigned subProblemIdx);
    void makeParamsForDedwSubProblem(unsigned numOfSubProblems, unsigned subProblemIdx);
    bool extractGcdFromConvParams(std::array<unsigned, MME_MAX_CONV_DIMS - 1>* stride,
                                  std::array<unsigned, MME_MAX_CONV_DIMS - 1>* dilation,
                                  std::array<unsigned, MME_MAX_CONV_DIMS - 1>* commonDivs) const;
    bool shouldAddMemsetDesc(const MmeCommon::MmeLayerParams& newParams) const;
    bool skipRecipeGeneration(const MmeCommon::MmeLayerParams& params);

    DedwUnroll m_dedwUnroll;
};

class MmeFwdDedxDescriptorGenerator final : public MmeConvDescriptorGenerator
{
public:
    MmeFwdDedxDescriptorGenerator(const MmeCommon::MmeLayerParams& params) : MmeConvDescriptorGenerator(params) {}
    virtual ~MmeFwdDedxDescriptorGenerator() = default;

protected:
    void setDescHeader(Mme::Desc& desc) override;
    void setDescSBReuse(DescGroup& descGroup) override;
    void setDescLowering(DescGroup& descGroup) override;
};

class MmeDedwDescriptorGenerator final : public MmeConvDescriptorGenerator
{
public:
    MmeDedwDescriptorGenerator(const MmeCommon::MmeLayerParams& params) : MmeConvDescriptorGenerator(params) {}
    virtual ~MmeDedwDescriptorGenerator() = default;

protected:
    void setDescHeader(Mme::Desc& desc) override;
    void setDescSBReuse(DescGroup& descGroup) override;
    void setDescLowering(DescGroup& descGroup) override;
    void fixUnrollDesc(DescGroup& descGroup) override;
    bool enableUnroll();
};

class MmeBgemmDescriptorGenerator final : public MmeDescriptorGenerator
{
public:
    explicit MmeBgemmDescriptorGenerator(const MmeCommon::MmeLayerParams& params);
    virtual ~MmeBgemmDescriptorGenerator() = default;
    void mmeGenerateActivations() override;

protected:
    void setDescHeader(Mme::Desc& desc) override;
    void mmeCalcLayerRecipe() override;
    void setHeaderLoopSelectors(DescGroup& descGroup) override;
    void setDescSBReuse(DescGroup& descGroup) override;
    void setAguConfig(DescGroup& descGroup) override;
    void setPaddingValues(Mme::Desc& desc) override;
    void setDescLowering(DescGroup& descGroup) override {}
    void handlePartials(DescGroup& descGroup) {}
    void handleTransO();

    void fixDedwAsBgemm(DescGroup& descGroup);

};
unsigned getDataTypeShiftAmount(MmeCommon::EMmeDataType dt);

}  // namespace gaudi

#endif //MME__GAUDI_MME_DESCRIPTOR_GENERATOR_H
