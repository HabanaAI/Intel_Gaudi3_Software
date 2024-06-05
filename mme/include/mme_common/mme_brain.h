#ifndef MME__BRAIN_H
#define MME__BRAIN_H

#include <memory>
#include <sstream>
#include "include/mme_aspects.h"
#include "include/mme_common/recipe_generator.h"

namespace MmeCommon
{
static constexpr uint64_t defaultTileSize = 4 * 1024 * 1024;

class CommonGeoAttr;

struct operandMemoryAttr
{
    float accessBW = 0.0;
    //  number of time the whole tensor will be fetched
    unsigned accessesPerDcore = 0;
    unsigned accessesPerChip = 0;
};
struct PerfAttr
{
    //  mme EU utilization - between 0-1
    float maxUtilization = 0.0f;  // utilization of the whole node
    float mmeUtilization = 0.0f;  // utilization of a single slice
    uint64_t expectedReadInputCycles = 0;
    uint64_t expectedComputeCycles = 0;
    uint64_t expectedRuntimeCycles = 0;
    float expectedRuntime = 0;
    operandMemoryAttr memoryAttrA;
    operandMemoryAttr memoryAttrB;
    operandMemoryAttr memoryAttrC;
    operandMemoryAttr memoryAttrAux;
    float unaligedPenaltyA = 1;
    float unaligedPenaltyB = 1;

    // old fields to be replaced with the fields above
    unsigned fetchNrA = 0;
    unsigned fetchNrB = 0;

    unsigned numOfActivations = 0;
    std::vector<unsigned> rollUpArray;
    std::string print() const;
};

struct MmeSolutionRequirements
{
    // solution graph implications
    std::vector<unsigned> perforationDimVec;
    std::optional<unsigned> bwInflationDim;
    std::vector<unsigned> utilizationInflationDims;
    std::vector<unsigned> walkDims;
    std::vector<unsigned> cdDims;
    bool requiresMemset = false;
    bool requiresCast = false;
    bool performsReduction = false;
    bool cdSliced = false;
    std::string print() const
    {
        std::stringstream ss;
        ss << "Requirements:" << std::endl;
        ss << "PerforationDim : [" << arrayToStr(perforationDimVec.begin(), perforationDimVec.end()) << "] bwInflationDim: " << bwInflationDim.value_or(-1) << std::endl;
        ss << "UtilizationInflationDims: [" << arrayToStr(utilizationInflationDims.begin(), utilizationInflationDims.end()) << "]" << std::endl;
        ss << "WalkDims: [" << arrayToStr(walkDims.begin(), walkDims.end()) << "] cdDims: [" << arrayToStr(cdDims.begin(), cdDims.end()) << "]" << std::endl;
        ss << "requireMemset: " << requiresMemset << " requireCast : " << requiresCast << " performReduction: " << performsReduction << " cdSliced: " << cdSliced << std::endl;
        return ss.str();
    }
};

enum OptimizationSolutions
{
    NO_OPT,
    BATCH_CONCURRENCY,
    CD_CONCURRENCY,
    HYBRID_CONCURRENCY
};

struct MmeBrainSolution
{
    OptimizationSolutions solutionType;  // indication for which optimization this solution implements
    MmeStrategy strategy;  // handle for codeGen to generate the chosen solution
    PerfAttr perfAttr;  // solution perf and memory attributes
    MmeSolutionRequirements requirements;  // indication for GC for the requirements for this solution
    // relaxedTile indicates for each dimension whether solutionDimMultiplier is a multiplier
    // of the nodes access pattern alone or of dimMultipliers and commonGranularity
    std::vector<bool> relaxedTile;
    std::vector<unsigned> solutionDimMultipliers;
    // required to make sure solution inflation is done in multiples of it.
    std::vector<unsigned> previousSolutionMultipliers;
    std::string print(const std::string& nodeName = "") const;
};
using MmeBrainSolutionPtr = std::shared_ptr<MmeBrainSolution>;
using MmeBrainSolutionContainer = std::vector<MmeBrainSolutionPtr>;

struct MmeBrainOperationModes
{
    bool addAlignmentPenaltyCalc = false;
    bool addTieBreakerPreferredReuseOperand = false;
    bool addOptimizationToLBSolutions = false;
};
struct MmeBrainKnobs
{
    float minUtilization = 0.9;
    uint64_t maxTileSize = 4 * 1024 * 1024;  // in Bytes
    uint64_t minInputReuse = 4;
    uint64_t minCd = 2048;  // in Elements
    float utilizationEpsilon = 0.05;
    MmeBrainOperationModes operationModes;
};

class MmeBrain
{
public:
    MmeBrain(const ChipType chipType, MmeBrainOperationModes operationModes = {false, false, false});

    static MmeLayerParams getDefaultParams(const ChipType chipType);
    static std::unique_ptr<CommonGeoAttr> getGeoAttr(const ChipType chipType, const MmeLayerParams& params);
    static unsigned getGeometryWidth(ChipType chipType, const MmeLayerParams& params);
    static unsigned getGeometryHeight(ChipType chipType, const MmeLayerParams& params);
    static unsigned getGeometryConcurrency(ChipType chipType, const MmeLayerParams& params);
    static unsigned getGeometryCdConcurrency(ChipType chipType, const MmeLayerParams& params);
    static unsigned getEffectiveBatchConcurrency(ChipType chipType, const MmeLayerParams& params);
    static bool isAsymPortConfigMode(ChipType chipType, const MmeLayerParams& params);

    std::vector<EMmeGeometry> getGeometries(MmeLayerParams& params);  // available geometries
    std::vector<EMmePattern> getPatterns(MmeLayerParams& params);  // available walking patterns
    std::vector<OptimizationSolutions> getRelevantOptimizations(const MmeLayerParams& params,
                                                                const MultiplierArray& commonGranularity,
                                                                const MultiplierArray& previousMultiplier);
    void getRecommendedStrategy(MmeLayerParams& params, bool isGeoPreferredShort = true);
    void setBrainKnobs(MmeBrainKnobs& knobs) { m_knobs = knobs; };
    void setOperationModes(const MmeBrainOperationModes& operationModes) { m_knobs.operationModes = operationModes; };
    const MmeBrainOperationModes& getOperationModes() const { return m_knobs.operationModes; };
    MmeBrainSolutionContainer getMmeSolutions(const MmeLayerParams& params,
                                              const MultiplierArray& commonGranularity,
                                              const MultiplierArray& previousMultiplier,
                                              const bool cdPerforationEn = false);
    MmeBrainSolutionPtr inflateForUtilization(const MmeLayerParams& params,
                                              const MmeBrainSolutionPtr curSolution,
                                              const MultiplierArray& commonGranularity,
                                              PhysicalAspects::Name aspectToInflate,
                                              const std::optional<float>& utilizationThreshold = std::nullopt);
    void chooseConcurrency(MmeLayerParams& params);
    void applyCdConcurrency(MmeLayerParams& paramsForCdConcurrency, float& cdConcurrencyLevel);
    void applyBatchConcurrency(MmeLayerParams& paramsForBatchConcurrency, float& batchConcurrencyLevel);
    void applyHybridConcurrency(MmeLayerParams& paramsForBatchConcurrency, float& batchConcurrencyLevel);
    //  set perfAttr according to mme params
    void getPerfAttr(const MmeLayerParams& params,
                     PerfAttr& perfAttr,
                     std::optional<MmeLayerParams> slicedParams = std::nullopt,
                     std::optional<bool> cdPerforation = std::nullopt);
    void getCdDim(const MmeLayerParams& params, std::vector<unsigned>& cdDims) const;
    bool gaudiShouldUnroll(MmeLayerParams& params);
    void addNumOfRollUpsForCurActivation(unsigned int numOfRollup)
    {
        m_perfAttr.rollUpArray.push_back(numOfRollup);
    }
    unsigned applyTensorFlattening(MmeCommon::MmeLayerParams& params);
    unsigned flattenBgemm(MmeCommon::MmeLayerParams& params);
    unsigned getFlatteningFactor() { return m_flattening; };
    bool convIsConvertibleToGemm(MmeCommon::MmeLayerParams& params);
    bool bgemmCanBeFlattened(MmeCommon::MmeLayerParams& params);
    static void shiftDimensions(MmeCommon::MmeLayerParams& params, unsigned trivialDim, unsigned distance);
    static void trivialDimsReduction(MmeCommon::MmeLayerParams& params);
    std::string getGeometryDebugInfo(const CommonGeoAttr& geoAttr) const;
    std::string getPerfDebugInfo(PerfAttr& perfAttr) const;

    static bool opSupportsChoosingConcurrency(EMmeOpType opType)
    {
        // Currently only dedw supports cd concurrency
        return isDedwOperation(opType);
    };
    void setParamsToSolutionSize(MmeLayerParams& params, const MultiplierArray& granuleSizes);
    void setParamsToSolutionSize(MmeLayerParams& params,
                                 const MultiplierArray& solutionMultipliers,
                                 const MultiplierArray& commonGranularity);
    bool dataTypeSupportsReduction(EMmeDataType dt);
    unsigned getMinCd(bool cdPerforation) const;
    bool skipCdConcurrencySolution(const MmeBrainSolutionPtr& newSolution,
                                   const MmeLayerParams& params,
                                   const OptimizationSolutions& opt,
                                   const bool cdPerforationEn) const;

private:
    const ChipType m_chipType;
    const MmeHalReader& m_mmeHal;
    std::optional<RecipeGenerator> m_recipeGenerator;
    std::shared_ptr<CommonGeoAttr> m_geoAttr;
    PerfAttr m_perfAttr = {0};
    unsigned m_flattening = 1;
    MmeBrainKnobs m_knobs;

    void calcFetchNr(const MmeLayerParams& params);
    float calcUtilizationImpl(const MmeLayerParams& params);
    float calcSliceUtilization(const MmeLayerParams& params,
                               MmeLayerParams& slicedParams,
                               const std::vector<PhysicalAspects::Name>& aspects);
    void calcUtilization(const MmeLayerParams& params, std::optional<MmeLayerParams> slicedParams = std::nullopt);
    void calcExpectedCycles(const MmeLayerParams& params);
    void calcNumOfActivations();

    //  set geometry and walking pattern
    void getRecommendedGeometryAndPattern(MmeLayerParams& params, bool isGeoPreferredShort = true);

    void chooseGeometry(MmeLayerParams& params, bool isGeoPreferredShort);
    void chooseWalkingPattern(MmeLayerParams& params);
    void chooseConvWalkingPattern(MmeLayerParams& params);
    void chooseBgemmWalkingPattern(MmeLayerParams& params);
    void choosePackingFactorForReductionAdd(MmeLayerParams& params);

    void getRecommendedStrategyIncludingConcurrency(const MmeLayerParams& params);
    const MmeLayerParams& ChooseConcurrency(const MmeLayerParams& paramsForCdConcurrency,
                                            const MmeLayerParams& paramsForBatchConcurrency);

    MmeBrainSolutionPtr generateSolution(const MmeLayerParams& params,
                                         const MultiplierArray& commonGranularity,
                                         const MultiplierArray& previousMultiplier,
                                         const bool cdSplit,
                                         const bool cdPerforation,
                                         const OptimizationSolutions opt);

    MultiplierArray getSolutionMultipliers(const MmeLayerParams& params,
                                           const MultiplierArray& commonGranularity,
                                           const MultiplierArray& previousMultiplier,
                                           const bool cdSplit,
                                           const bool cdPerforation);

    void getBwInflationDim(const MmeLayerParams& params, std::optional<unsigned>& inflationDim);

    void getPerforationDim(const MmeLayerParams& params,
                           std::vector<unsigned>& perforationDimVec,
                           const bool cdPerforation = false);
    std::vector<unsigned> getWalkDims(const MmeLayerParams& params, bool cdSplit) const;
    void getSolutionReq(const MmeLayerParams& params,
                        const bool cdSplit,
                        MmeSolutionRequirements& solutionRequirements,
                        const bool cdPerforation);
    void calcMemoryAttributes(const MmeLayerParams& params, const bool cdPerforation);

    inline unsigned getRollUpTime();
    void getNumStepsPerGeometry(const MmeLayerParams& params,
                                const CommonGeoAttr& curGeoAttr,
                                unsigned& fcdSteps,
                                unsigned& spSteps,
                                unsigned& batchSteps,
                                unsigned& constrainedSteps);
    float calcConstrainedStepCost(const MmeLayerParams& params);
    unsigned getNumSpatialSteps(const MmeLayerParams& params, const CommonGeoAttr& curGeoAttr);
    std::vector<EMmeGeometry> getSortedGeometries(MmeLayerParams& params, bool isGeoPreferredShort);
    void calcExpectedReadInputCycles(const MmeLayerParams& params);
    unsigned calcExpectedReadInputCyclesPerOperand(const MmeLayerParams& params,
                                                   MmeCommon::EMmeInternalOperand operand);
    float getGeometryUnalignedPenalty(const MmeLayerParams& params, MmeCommon::EMmeInternalOperand operand);
};
}  // namespace MmeCommon

#endif //MME__BRAIN_H
