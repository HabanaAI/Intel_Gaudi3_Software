#ifndef MME__DESCRIPTOR_GENERATOR_BASE_H
#define MME__DESCRIPTOR_GENERATOR_BASE_H

#include "include/mme_common/conv_sub_problems.h"
#include "include/mme_common/descriptor_cache.h"
#include "include/mme_common/mme_brain.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/skip_data.h"
#include "include/sync/overlap.h"
#include "include/utils/descriptor_dumper.h"

namespace MmeCommon
{
// cant include common linear ranges do to conflicting includes, so define the templated class here.
template<typename Desc>
class CommonRoiCalculator;

typedef enum
{
    INPUT_TENSOR_A,
    INPUT_TENSOR_B,
    OUTPUT_TENSOR_C,
    OUTPUT_TENSOR_O,
    AUX_ROLE_MASKED_BGEMM_A,
    AUX_ROLE_MASKED_BGEMM_B,
    AUX_TENSOR_0 = AUX_ROLE_MASKED_BGEMM_A,     // For gc compatibility, to be removed
    AUX_TENSOR_1 = AUX_ROLE_MASKED_BGEMM_B,     // For gc compatibility, to be removed
    AUX_ROLE_CD_SCRATCHPAD,
    AUX_ROLE_CD_REDUCTION,
    AUX_TENSOR_SCRATCHPAD = AUX_ROLE_CD_SCRATCHPAD,  // For gc compatibility, to be removed
    AUX_TENSOR_REDUCTION = AUX_ROLE_CD_REDUCTION,    // For gc compatibility, to be removed
    NUM_TENSORS
} TensorRoles;

typedef enum
{
    NO_CD_PARALLEL,
    CD_PARALLEL_COMPUTE,
    CD_PARALLEL_REDUCTION_ADD
} CDParallelOp;

struct TensorMemoryMetaData
{
    uint64_t addr = 0;
    bool     isSram = false;
};

struct MmePatchMetaData
{
    std::array<std::optional<TensorMemoryMetaData>, TensorRoles::NUM_TENSORS> tensorMetaData;
    bool bOperandUsed = true;
    bool oOperandUsed = false;
};

// Operand roles maps for each operand (a, b and c) its role. It can be the standard tensor or
// an aux tensor. In addition, it supports using primary and secondary tensors for each operand,
// which is currently relevant for c only (primary - c, secondary - o).
using OperandRoles = std::array<TensorRoles, e_mme_op_nr>;

//  this is a base class to all chips descriptor generators
//  its goal is to create a unified interface for MME description generation
//  and serving as a factory for the different description generators.
class MmeDescriptorGeneratorBase
{
public:
    MmeDescriptorGeneratorBase() = default;
    virtual ~MmeDescriptorGeneratorBase() = default;
    static std::unique_ptr<MmeDescriptorGeneratorBase>
        createMmeDescGenerator(const ChipType chipType, const bool isDmaOperation, const unsigned numOfTotalMmes);
    virtual const CommonGeoAttr& getGeoAttr() const = 0;
    virtual void mmeGenerateActivations() = 0;
    virtual unsigned getMmeActivationNr() const = 0;

    virtual void patchMmeDescriptors(MmePatchMetaData patchMetaData, bool calcRoi) = 0;
    virtual void patchMcids(uint16_t mcidA, uint16_t mcidB, uint16_t mcidC) {};
    virtual void patchMcids(uint16_t mcidA,
                            uint16_t mcidB,
                            uint16_t mcidC,
                            std::optional<uint16_t> mcidAuxScratchpad,
                            std::optional<uint16_t> mcidAuxReductionAdd) {};

    virtual void mmeIncrementDataTensorViews(const EMmeOperand operand, const uint64_t offset) = 0;
    virtual TensorRoles getAuxTensorRole(MmeAuxTensorIdx idx) const = 0;

    //  patching function for the actual dualGemm sizes
    virtual void patchDualGemm(MmeCommon::MmeTensorView& x,
                               MmeCommon::MmeTensorView& w,
                               MmeCommon::MmeTensorView& y,
                               const uint64_t addrX,
                               const uint64_t addrW,
                               const uint64_t addrY,
                               const uint64_t addrO,
                               const bool YisSram,
                               const bool OisSram,
                               unsigned gemmIdx)
    {
        MME_ASSERT(0, "dualGemm not supported");
    }
    //  this function is used to set the same SO to each MME across all activations, used by mme user.
    virtual void mmePatchSyncObjects(const uint32_t mmeIdx,
                                     const uint32_t addr0,
                                     const uint32_t addr1,
                                     const uint32_t slaveAddr0 = 0,
                                     const uint32_t slaveAddr1 = 0) = 0;

    virtual void patchContextId(uint64_t contextId) = 0;
    virtual bool setParams(const MmeCommon::MmeLayerParams& newParams) { return false; }
    virtual unsigned int addParams(const MmeLayerParams& newParams) = 0;
    virtual ChipType getChipType() const = 0;
    virtual const MmeCommon::MmeRecipe& getRecipe() const = 0;
    virtual std::vector<std::string> getDescCacheDebugInfo() const = 0;
    virtual std::vector<std::string> getRecipeDebugInfo(bool verbose = true) const = 0;
    virtual std::vector<std::string> getBrainDebugInfo(const MmeLayerParams& curParams) const;
    virtual std::vector<std::vector<std::string>> dumpDescriptors(bool dumpAsBinary) const = 0;
    virtual MmeBrain& getMMEBrain() = 0;
    virtual const MmeBrain& getMMEBrain() const = 0;
    virtual std::string getRecurringMisalignmentDebugInfo() const { return ""; }

    unsigned getMmeNr() const;
    MmeCommon::EMmeGeometry getGeometry() const;
    unsigned getGeometryWidth() const;
    unsigned getGeometryHeight() const;
    unsigned getEffectiveBatchConcurrency() const;
    unsigned getGeometryCdConcurrency() const;
    MmeDimsIndex getSpInterleavingDim(EMmeInternalOperand operand) const;
    bool isAsymPortConfigMode() const;
};

template<typename Desc>
struct MmeActivation
{
    MmeActivation (unsigned MmeNr)
    {
        descriptors.resize(MmeNr);
    }

    std::vector<Desc> descriptors;
    Desc& getDesc(unsigned idx) { return descriptors[idx]; }
    const Desc& getDesc(unsigned idx) const { return descriptors[idx]; }

    bool operator==(const MmeActivation<Desc>& other) const
    {
        return (numSignals == other.numSignals &&
        skipDataA == other.skipDataA &&
        skipDataB == other.skipDataB &&
        skipDataC == other.skipDataC &&
        spView == other.spView &&
        fcdView == other.fcdView &&
        nonSpatialView == other.nonSpatialView &&
        roiO == other.roiO &&
        roiW == other.roiW &&
        roiX == other.roiX &&
        roiY == other.roiY &&
        isGemm == other.isGemm &&
        isMask == other.isMask &&
        isCdReduction == other.isCdReduction &&
        operandRoles == other.operandRoles &&
        numTetrises == other.numTetrises &&
        numRollups == other.numRollups &&
        descriptors == other.descriptors);
    }


    MmeActivation(const MmeActivation& other)
    : descriptors(other.descriptors), roiX(other.roiX), roiY(other.roiY), roiW(other.roiW), roiO(other.roiO)
    {
        isGemm = other.isGemm;
        isMask = other.isMask;
        isCdReduction = other.isCdReduction;
        numSignals = other.numSignals;
        spView = other.spView;
        fcdView = other.fcdView;
        nonSpatialView = other.nonSpatialView;
        numTetrises = other.numTetrises;
        numRollups = other.numRollups;
        skipDataA = other.skipDataA;
        skipDataB = other.skipDataB;
        skipDataC = other.skipDataC;
        operandRoles = other.operandRoles;
    }

    unsigned numSignals = 0;

    // For ROI calculation
    SkipData skipDataA;
    SkipData skipDataB;
    SkipData skipDataC;

    SingleDimSubView spView;
    SingleDimSubView fcdView;
    MultiDimSubView nonSpatialView;

    OverlapRoi roiX;
    OverlapRoi roiY;
    OverlapRoi roiW;
    OverlapRoi roiO;

    bool isGemm = false;
    bool isMask = false;
    bool isCdReduction = false;

    unsigned numTetrises = 0;
    unsigned numRollups = 0;
    // By default, primary tensors are used (and not aux tensors)
    OperandRoles operandRoles = {{INPUT_TENSOR_A, INPUT_TENSOR_B, OUTPUT_TENSOR_C}};
    std::optional<unsigned> paramsIdx;
};
template<typename Desc>
using ActivationList = std::list<MmeActivation<Desc>>;

template<typename Desc>
using ActivationVec = std::vector<MmeActivation<Desc>>;

template<typename Desc>
class MmeDescriptorGenerator : public MmeDescriptorGeneratorBase
{
private:
    void patchOutputView(MmeActivation<Desc>& activation,
                         const EMmeOperand operand,
                         const uint64_t addr,
                         const bool isSram,
                         const bool isPrimary,
                         const bool squashRois,
                         const bool calcRoi);
protected:
    void createRecipes(const MmeLayerParams& params);
    const KeyAndHash<MmeLayerParams>& getOriginalParamsAndHash() const
    {
        MME_ASSERT(m_originalParams.has_value(), "LayerParams have not been initialized");
        return *m_originalParams;
    }
    const MmeLayerParams& getOriginalParams() const { return getOriginalParamsAndHash().getKey(); }
    OperandRoles getOperandRoles() { return m_operandRoles; }
    const std::shared_ptr<CommonRoiCalculator<Desc>>& getRoiCalculator() const;
    const std::shared_ptr<CommonRoiCalculator<Desc>>& getRoiCalculator(const MmeCommon::MmeRecipe& recipe) const;
    virtual void createRoiCalculator();
    MmeDescriptorGenerator();
    uint64_t& getInputAddrPtr(Desc& desc, const EMmeOperand operand, bool secondGemm = false) const;
    uint64_t& getOutputAddrPtr(Desc& desc, const EMmeOperand operand, bool isPrimary, bool secondGemm = false) const;
    virtual void setSBCacheDisable(const MmeCommon::EMmeOperand operand,
                                   const bool isSram,
                                   const bool addrIsAligned,
                                   Desc& desc) = 0;
    void configurePerfEvents(ActivationVec<Desc>& activations);
    void setDescPerfEvent(const bool first, const bool last, const bool partialDesc, Desc& desc);
    virtual void setSinglePerfEvent(EMmeTraceEngine engine, unsigned startEndMask, Desc& desc) = 0;
    bool canSquashRois();
    virtual ChipType getChipType() const override;
    virtual std::shared_ptr<MmeDescriptorDumper<Desc>> getDescriptorDumper() const = 0;
    virtual void configureMemoryDirectives(Desc& desc, const MmeCommon::MmeRecipe& recipe) const;
    MmeCommon::ConvSubProblem* getCurrentSubProblem() { return m_convSubProblems.current; }
    const MmeCommon::ConvSubProblem* getCurrentSubProblem() const { return m_convSubProblems.current; }
    virtual bool isLastSubProblem() const { return m_convSubProblems.isLast(); }
    static bool validateParamOfGemmOp(const MmeCommon::MmeLayerParams& params, std::string& errorMsg);
    void calculateLinearRanges(MmeActivation<Desc>& activation,
                               const std::shared_ptr<CommonRoiCalculator<Desc>>& roiCalc,
                               const EMmeOperand operand,
                               const bool isInput,
                               const bool isSram,
                               const bool squashRoi);
    static bool validateParamOfReductionAddOp(const MmeCommon::MmeLayerParams& params, std::string& errorMsg);
    bool getActivationsFromCache();
    bool addParamsActivationsToCache();

    ActivationVec<Desc> m_activations;
    MmeRecipe m_recipe;
    std::shared_ptr<CommonGeoAttr> m_commonGeoAttr;
    std::optional<KeyAndHash<MmeLayerParams>> m_originalParams;
    std::shared_ptr<std::vector<MmeLayerParams>> m_paramsVec;

    OperandRoles m_operandRoles = {INPUT_TENSOR_A, INPUT_TENSOR_B, OUTPUT_TENSOR_C};
    MmeCommon::MmeBrain m_mmeBrain;
    MmeCommon::ConvSubProblemContainer m_convSubProblems;
    std::shared_ptr<CommonRoiCalculator<Desc>> m_roiCalculator;
    bool m_perforated = false;

public:
    virtual ~MmeDescriptorGenerator() = default;
    virtual ActivationVec<Desc>& getMmeActivations() { return m_activations; }
    virtual const ActivationVec<Desc>& getMmeActivations() const { return m_activations; }
    unsigned getMmeActivationNr() const override { return getMmeActivations().size(); };
    virtual const CommonGeoAttr& getGeoAttr() const override { return *m_commonGeoAttr; }
    virtual bool setParams(const MmeCommon::MmeLayerParams& newParams) override;
    virtual unsigned int addParams(const MmeLayerParams& newParams) override;
    virtual bool validateParams(const MmeCommon::MmeLayerParams& params, std::string& errorMsg) = 0;

    uint64_t* mmeGetTensorAddressFields(const EMmeOperand operand, Desc& desc, bool secondGemm = false) const;
    template<typename TensorDesc>
    TensorDesc* mmeGetTensorDescriptor(const EMmeOperand operand, Desc& desc) const;

    // tensor patching
    void patchInputTensor(MmeActivation<Desc>& activation,
                          const EMmeOperand operand,
                          const std::optional<TensorMemoryMetaData>& tensorMemoryData,
                          bool calcRoi);
    void patchOutputTensor(MmeActivation<Desc>& activation,
                           const EMmeOperand operand,
                           const std::optional<TensorMemoryMetaData>& primaryOutputTensorMemoryData,
                           bool oOperandUsed,
                           const std::optional<TensorMemoryMetaData>& secondaryOutputTensorMemoryData,
                           const bool calcRoi);
    virtual void patchMmeDescriptors(MmePatchMetaData patchMetaData, bool calcRoi) override;
    virtual TensorRoles getAuxTensorRole(MmeAuxTensorIdx idx) const override;

    //  second address patching function - increment the base address of "operand" by "addrOffset"
    void mmeIncrementDataTensorViews(const EMmeOperand operand, const uint64_t addrOffset) override;

    virtual void
    mmeGetDescValidMask(const Desc& desc, bool* mask, bool* aguOut1FromAguOut0_DW0, bool* aguOut1FromAguOut0_DW1_4) = 0;
    virtual void patchSignalColoring(MmeActivation<Desc>& activation, const bool addr0isSram, const bool addr1isSram) = 0;
    void patchDebugWkldID(const unsigned wkldID, Desc& desc);
    void patchContextId(uint64_t contextId) override;
    size_t calculateActivationsCount() const;
    virtual MmeBrain& getMMEBrain() override { return m_mmeBrain; }
    virtual const MmeBrain& getMMEBrain() const override { return m_mmeBrain; }
    const MmeCommon::MmeRecipe& getRecipe() const override
    {
        return (getCurrentSubProblem() != nullptr) ? getCurrentSubProblem()->recipe : m_recipe;
    }
    const MmeLayerParams& getParams() const
    {
        return (getCurrentSubProblem() != nullptr) ? getCurrentSubProblem()->params : getOriginalParams();
    }
    virtual std::vector<std::vector<std::string>> dumpDescriptors(bool dumpAsBinary) const override;
    const MmeCommon::ConvSubProblemContainer getSubProblems() { return m_convSubProblems;};
    std::vector<std::string> getDescCacheDebugInfo() const override;
    void setPerforated(bool val) { m_perforated = val; }
    bool isPerforated() const { return m_perforated; }
    static std::shared_ptr<const std::vector<MmeActivation<Desc>>>
    getSharedOwnershipForCachedActivations(const MmeLayerParams& originalParams);
};

template<typename Desc>
using pMmeDescriptorGenerator = std::unique_ptr<MmeDescriptorGenerator<Desc>>;
using pMmeDescriptorGeneratorBase = std::unique_ptr<MmeDescriptorGeneratorBase>;
}  // namespace MmeCommon

#endif //MME__DESCRIPTOR_GENERATOR_BASE_H
