#include "include/mme_common/mme_brain.h"
#include "common_geo_attr.h"
#include "mme_hal_factory.h"
#include "mme_assert.h"
#include "mme_geo_factory.h"
#include "mme_common/mme_params_factory.h"
#include "include/gaudi/new_descriptor_generator/dedw_unroll.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe_generator.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "mme_params_dumper.h"
#include "multipliers_generator.h"
#include <bitset>
#include <sstream>

#include "mme_access_pattern.h"
#include "index_space_dimensions.h"
#include "spatial_dims_mapping.h"
#include "access_pattern_utils.h"

#include "utils/logger.h"

#define Mega 1000000

namespace MmeCommon
{
using namespace AccessPatternDetails;
using MmeAspect = PhysicalAspects::Name;

static inline bool dedwIsConvertibleToGemm(const MmeLayerParams& params)
{
    return ((params.isDedwOperation()) && allValuesEqualTo<unsigned>(params.w.sizes.data() + 2, 3, 1) &&
            allValuesEqualTo<int>(params.conv.padding.data(), 3, 0) &&
            allValuesEqualTo<unsigned>(params.conv.stride.data(), 3, 1) &&
            allValuesEqualTo<unsigned>(params.conv.dilation.data(), 3, 1));
}

MmeBrain::MmeBrain(const ChipType chipType, MmeBrainOperationModes operationModes) :
    m_chipType(chipType), m_mmeHal(getMmeHal(chipType))
{
    setOperationModes(operationModes);
}

void MmeBrain::getPerfAttr(const MmeLayerParams& params,
                           PerfAttr& perfAttr,
                           std::optional<MmeLayerParams> slicedParams,
                           std::optional<bool> cdPerforation)
{
    m_geoAttr = getGeoAttr(m_chipType, params);
    m_recipeGenerator.emplace(params.isGemmOperation() ? e_mme_bgemm_recipe : e_mme_conv_recipe,
                              params,
                              m_mmeHal,
                              *m_geoAttr);
    m_recipeGenerator->generateRecipe();

    calcFetchNr(params);
    calcUtilization(params, slicedParams);
    calcExpectedCycles(params);
    calcNumOfActivations();
    calcMemoryAttributes(params, cdPerforation.value_or(false));
    perfAttr = m_perfAttr;
}

void MmeBrain::getCdDim(const MmeLayerParams& params, std::vector<unsigned>& cdDims) const
{
    PhysicalAspects::Factory factory(&params);
    IndexSpaceAspect commonDimAspect = factory.create(PhysicalAspects::INPUTS_COMMON);
    for (auto idxSpcDim : commonDimAspect)
    {
        cdDims.push_back(idxSpcDim);
    }
}

void getUtilInflationDim(const MmeLayerParams& params, std::vector<unsigned>& inflationDims)
{
    switch (params.opType)
    {
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            if (params.canFlatten())
            {
                inflationDims.push_back(Gemm::DIM_OUT_HEIGHT);
                inflationDims.push_back(Gemm::DIM_BATCH_0);
            }
            break;
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            inflationDims.push_back(Conv::DIM_WIDTH);
            inflationDims.push_back(Conv::DIM_HEIGHT);
            if (params.conv.spatialDimsNr > 2) inflationDims.push_back(Conv::DIM_DEPTH);
            inflationDims.push_back(Conv::DIM_BATCH);
            break;
        case e_mme_dedw:
            break;
        default:
            MME_ASSERT(0, "operation not supported yet");
            break;
    }
}

void MmeBrain::getBwInflationDim(const MmeLayerParams& params, std::optional<unsigned>& inflationDim)
{
    switch (params.strategy.pattern)
    {
        case e_mme_sp_reduction_fck:
            if (params.getFcdSize() > m_geoAttr->getGeometryWidth())
            {
                if (params.isGemmOperation()) inflationDim.emplace(Gemm::DIM_OUT_FCD);
                else
                    inflationDim.emplace(Conv::DIM_OUT_CHANNELS);
            }
            break;
        case e_mme_sp_reduction_kfc:  // dedw
        case e_mme_sp_reduction_fkc:  // bgemm
            if (params.getSpatialSize() > m_geoAttr->getGeometryHeight())
            {
                if (params.isGemmOperation()) inflationDim.emplace(Gemm::DIM_OUT_HEIGHT);
                else
                    inflationDim.emplace(Conv::DIM_IN_CHANNELS);
            }
            break;
        case e_mme_z_reduction_skf:
            if (params.getFcdSize() > m_geoAttr->getGeometryWidth())
            {
                inflationDim.emplace(params.isDedxOperation() ? Conv::DIM_IN_CHANNELS : Conv::DIM_OUT_CHANNELS);
            }
            break;
        case e_mme_z_reduction_ksf:
            if (params.getSpatialSize() > m_geoAttr->getGeometryHeight())
            {
                inflationDim.emplace(Conv::DIM_BATCH);
            }
            break;
        default:
            MME_ASSERT(0, "walk pattern not supported yet");
            break;
    }
}

unsigned MmeBrain::getMinCd(bool cdPerforation) const
{
    unsigned dcoresConcurrency = cdPerforation ? m_mmeHal.getDcoreNr() : 1;
    return m_knobs.minCd * dcoresConcurrency;
}

void MmeBrain::getPerforationDim(const MmeLayerParams& params,
                                 std::vector<unsigned>& perforationDimVec,
                                 const bool cdPerforation)
{
    switch (params.opType)
    {
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
            if (cdPerforation)
            {
                getCdDim(params, perforationDimVec);
            }
            else if (m_geoAttr->getMmeGrid().fcd >= m_mmeHal.getDcoreNr())
            {
                perforationDimVec.push_back(Gemm::DIM_OUT_FCD);
            }
            else if (m_geoAttr->getGeometryConcurrency() >= m_mmeHal.getDcoreNr())
            {
                // Perforation dim must be equal to concurrency dim
                const MmeDimsIndex concurrentDim = m_geoAttr->getConcurrentDim();
                if ((concurrentDim >= MmeCommon::GEMM_DIM_B1) &&
                    (params.getOperand(e_mme_op_c).sizes[concurrentDim] >= 4))
                {
                    perforationDimVec.push_back(Gemm::DIM_BATCH_0 + (concurrentDim - MmeCommon::GEMM_DIM_B1));
                }
            }
            else if (m_geoAttr->getMmeGrid().spatial >= m_mmeHal.getDcoreNr())
            {
                // DCores should cover the height.
                if (params.canFlatten())
                {
                    // this right here is a perf bug, even when we flatten we would like to perforation over a higher
                    // batch dim, once we will support batch concurrency on higher dim we shouldnt get here anymore.
                    perforationDimVec.push_back(Gemm::DIM_BATCH_0);
                }
                else
                {
                    perforationDimVec.push_back(Gemm::DIM_OUT_HEIGHT);
                }
            }
            // if no axis has MMEs to split between the dcores then we do not recommend any perforation
            break;
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            if (cdPerforation)
            {
                getCdDim(params, perforationDimVec);
            }
            else if (m_geoAttr->getGeometryHeight() > m_geoAttr->getGeometryWidth())
            {
                perforationDimVec.push_back(Conv::DIM_BATCH);
            }
            else
            {
                perforationDimVec.push_back(params.isDedxOperation() ? Conv::DIM_IN_CHANNELS : Conv::DIM_OUT_CHANNELS);
            }
            break;
        case e_mme_dedw:
            if (cdPerforation)
            {
                perforationDimVec.push_back(Conv::DIM_BATCH);
            }
            else if (m_geoAttr->getGeometryConcurrency() == 1 && m_geoAttr->getGeometryCdConcurrency() == 1)
            {
                if (m_geoAttr->getGeometryHeight() >= m_geoAttr->getGeometryWidth())
                {
                    perforationDimVec.push_back(Conv::DIM_IN_CHANNELS);
                }
                else
                {
                    perforationDimVec.push_back(Conv::DIM_OUT_CHANNELS);
                }
            }
            else if (m_geoAttr->getGeometryCdConcurrency() >= m_mmeHal.getDcoreNr())
            {
                perforationDimVec.push_back(Conv::DIM_BATCH);
            }
            break;
        default:
            break;
    }
}

std::vector<unsigned> MmeBrain::getWalkDims(const MmeLayerParams& params, bool cdSplit) const
{
    std::vector<unsigned> walkDims {};
    if (cdSplit)
    {
        getCdDim(params, walkDims);
    }
    return walkDims;
}

static bool cdSplitRequiresCast(const MmeLayerParams& params)
{
    return params.getOperand(e_mme_op_c).elementType != e_type_fp32;
}

void MmeBrain::getSolutionReq(const MmeLayerParams& params,
                              const bool cdSplit,
                              MmeSolutionRequirements& solutionRequirements,
                              const bool cdPerforation)
{
    getCdDim(params, solutionRequirements.cdDims);
    getUtilInflationDim(params, solutionRequirements.utilizationInflationDims);
    getBwInflationDim(params, solutionRequirements.bwInflationDim);
    getPerforationDim(params, solutionRequirements.perforationDimVec, cdPerforation);
    bool isCDConcurrency = getGeoAttr(m_chipType, params)->getGeometryCdConcurrency() != 1;
    bool isNonDeterministicCDConcurrency = isCDConcurrency && !params.strategy.isDeterministic &&
                                           !cdPerforation;  // TODO: enable CDConcurrency with CDPerforation
    solutionRequirements.cdSliced = cdSplit;
    solutionRequirements.performsReduction = cdSplit || isNonDeterministicCDConcurrency;
    solutionRequirements.requiresCast = cdSplit && cdSplitRequiresCast(params);
    solutionRequirements.requiresMemset = isNonDeterministicCDConcurrency;
    solutionRequirements.walkDims = getWalkDims(params, cdSplit);
}

float MmeBrain::getGeometryUnalignedPenalty(const MmeLayerParams& params, MmeCommon::EMmeInternalOperand operand)
{
    if (!m_knobs.operationModes.addAlignmentPenaltyCalc)
    {
        return 1;
    }
    MME_ASSERT((operand == e_mme_op_a || operand == e_mme_op_b), "Input operand need to be a or b");

    unsigned numRecurringMisalignmentProblems =
        RecurringMisalignmentOptimization::calcNumSubProblems(params, m_mmeHal, operand);
    MME_ASSERT((numRecurringMisalignmentProblems != 0),
               "The number of recurringMisalignment subProblem must be at least 1");
    // #(numRecurringMisalignmentProblems - 1) is the number of problems with CL alignments != 0
    // Unaligned memory accesses result in 2 cycles per 128B bytes for MME's Suspension Buffer
    // so the unalignedPenalty = (sum of memory accesses / # subproblems depending on the number of CL alignments)
    float unalignedPenalty =
        (float) (1 + (numRecurringMisalignmentProblems - 1) * 2) / numRecurringMisalignmentProblems;
    return unalignedPenalty;
}

void MmeBrain::getNumStepsPerGeometry(const MmeLayerParams& params,
                                      const CommonGeoAttr& curGeoAttr,
                                      unsigned& fcdSteps,
                                      unsigned& spSteps,
                                      unsigned& batchSteps,
                                      unsigned& constrainedSteps)
{
    unsigned concurrentLevel = curGeoAttr.getGeometryConcurrency();
    fcdSteps = div_round_up(params.getFcdSize(), curGeoAttr.getGeometryWidth());
    spSteps = div_round_up(params.getSpatialSize(), curGeoAttr.getGeometryHeight());
    batchSteps = params.getBatchSize(concurrentLevel);
    constrainedSteps = 0;  //  activations in which the MME stalled on missing input ports
    if (curGeoAttr.isGeometryPortConstrained())
    {
        bool isRaster = params.isPatternRaster();
        // incase one step only on first walking direction.
        isRaster = (isRaster && (fcdSteps != 1)) || (!isRaster && spSteps == 1);

        if (isRaster)
        {
            constrainedSteps = spSteps;
        }
        else
        {
            constrainedSteps = fcdSteps;
        }

        bool resetRecipe = false;
        if (!m_recipeGenerator.has_value())
        {
            m_recipeGenerator.emplace(params.isGemmOperation() ? e_mme_bgemm_recipe : e_mme_conv_recipe,
                                      params,
                                      m_mmeHal,
                                      *m_geoAttr);
            m_recipeGenerator->generateRecipe();
            resetRecipe = true;
        }

        const auto& recipe = m_recipeGenerator->get();
        auto& recipeIterator = recipe.getIterator();
        unsigned fcdSplits = recipe.getFcdSubviews().size();
        unsigned spSplits = recipe.getSpSubviews().size();
        unsigned convSplits = recipe.getNonSpatialSubviews().size();
        if (params.isDedwOperation())
        {
            // in dedw operations the output height splits is represented by conv splits instead of spatial splits
            constrainedSteps *= fcdSplits * convSplits;
        }
        else
        {
            constrainedSteps *= fcdSplits * spSplits;
        }

        if (resetRecipe)
        {
            m_recipeGenerator.reset();
        }
    }

    m_perfAttr.unaligedPenaltyA = getGeometryUnalignedPenalty(params, e_mme_op_a);
    m_perfAttr.unaligedPenaltyB = getGeometryUnalignedPenalty(params, e_mme_op_b);

    if (m_perfAttr.unaligedPenaltyA > 1 && m_perfAttr.unaligedPenaltyB > 1)
    {
        constrainedSteps +=
            std::max(spSteps, fcdSteps) * std::max(m_perfAttr.unaligedPenaltyA - 1, m_perfAttr.unaligedPenaltyB - 1);
    }
    else
    {
        if (m_perfAttr.unaligedPenaltyA > 1)
        {
            constrainedSteps += spSteps * (m_perfAttr.unaligedPenaltyA - 1);
        }
        if (m_perfAttr.unaligedPenaltyB > 1)
        {
            constrainedSteps += fcdSteps * (m_perfAttr.unaligedPenaltyB - 1);
        }
    }
}

unsigned MmeBrain::getNumSpatialSteps(const MmeLayerParams& params, const CommonGeoAttr& curGeoAttr)
{
    unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
    getNumStepsPerGeometry(params, curGeoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);
    return fcdSteps * spSteps;
}

float MmeBrain::calcConstrainedStepCost(const MmeLayerParams& params)
{
    bool transA = m_geoAttr->isTransposed(e_mme_op_a);
    bool transB = m_geoAttr->isTransposed(e_mme_op_b);

    // TODO if an operand is non transposed the cost could still be less than 2 due to SB cache, improve in the future
    if (!transA && !transB) return 2;

    float costA = 2, costB = 2;
    if (transA)
    {
        unsigned spSize = params.getSpatialSize();
        unsigned spSizePerPort = div_round_up(spSize, m_geoAttr->getInterleavedSpatialPortsNr(e_mme_op_a));
        costA = 1 + spSizePerPort / (float) m_geoAttr->getTeHeight();
    }

    if (transB)
    {
        unsigned fcdSize = params.getFcdSize();
        MME_ASSERT(m_geoAttr->getInterleavedSpatialPortsNr(e_mme_op_b) == 1,
                   "B ports cant interleave when they are transposed");
        unsigned fcdSizeFirstPort = std::min(fcdSize, m_geoAttr->getTeHeight());
        costB = 1 + fcdSizeFirstPort / (float) m_geoAttr->getTeHeight();
    }

    // once the first port finishes its work the constraint is over, so the cost is the cost of the fastest port
    return std::min(costA, costB);
}

void MmeBrain::calcExpectedCycles(const MmeLayerParams& params)
{
    MME_ASSERT(m_recipeGenerator.has_value(), "RecipeGenerator is not set");
    const auto& recipe = m_recipeGenerator->get();
    unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
    getNumStepsPerGeometry(params, *m_geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);
    unsigned cdConcurrencyLevel = m_geoAttr->getGeometryCdConcurrency();
    // get CD time in cycles - CDSize or rollup min time.
    float constrinedStepCost = calcConstrainedStepCost(params);
    unsigned effectiveCD = div_round_up(params.getCDSize(), cdConcurrencyLevel);
    unsigned minCDToConsider = std::max(effectiveCD, getRollUpTime());
    unsigned minCDToConsiderWithConstrains = std::max((unsigned) (effectiveCD * constrinedStepCost), getRollUpTime());
    if (m_recipeGenerator->isPartialSBReuse())
    {
        // if CD on each partial is lower than RollUpTime -> we are bound on rollup for each descriptor.
        // since we balance the partials size - its good enough to calculate this approximately.
        unsigned approxCDSizePerPartial = div_round_up(params.getCDSize(), recipe.getPartialsNr());
        if (approxCDSizePerPartial < getRollUpTime())
        {
            minCDToConsider = recipe.getPartialsNr() * getRollUpTime();
        }
    }

    uint64_t numOfRegularGeometries = (fcdSteps * spSteps - constrainedSteps) * batchSteps;
    uint64_t numOfConstrainedGeometries = constrainedSteps * batchSteps;
    if (params.isConvOperation())
    {
        // in the past the cost of port constrained was hard coded to be twice the original CD.
        // in time we discovered that if we are rollup bound then the port constraint cost is hidden by it.
        // this triggered a different slicing in some QA test that uncovered a bug.
        // for now keep the legacy behavior for conv operations as this optimization is needed for bgemm
        // ticket - SW-158858
        numOfRegularGeometries += constrainedSteps;
    }
    m_perfAttr.expectedComputeCycles =
        numOfRegularGeometries * minCDToConsider + numOfConstrainedGeometries * minCDToConsiderWithConstrains;

    if (m_knobs.operationModes.addAlignmentPenaltyCalc)
    {
        calcExpectedReadInputCycles(params);
        m_perfAttr.expectedRuntimeCycles =
            std::max(m_perfAttr.expectedComputeCycles, m_perfAttr.expectedReadInputCycles);
    }
    else
    {
        m_perfAttr.expectedRuntimeCycles = m_perfAttr.expectedComputeCycles;
    }
    // time[us] = #cycles / freq[MHz]
    m_perfAttr.expectedRuntime = (double) m_perfAttr.expectedRuntimeCycles / m_mmeHal.getClkFreqMHz();
}

void MmeBrain::calcFetchNr(const MmeLayerParams& params)
{
    // TODO: calculate #fetches in DMA operations.
    if (params.isDmaOperation())
    {
        m_perfAttr.fetchNrB = 0;
        m_perfAttr.fetchNrA = 1;
        return;
    }

    MME_ASSERT(m_recipeGenerator.has_value(), "RecipeGenerator is not set");
    const auto& recipe = m_recipeGenerator->get();
    auto& recipeIterator = recipe.getIterator();

    bool reuseA = true;
    bool reuseB = true;
    bool partialReuse = false;

    m_perfAttr.fetchNrA = 0;
    m_perfAttr.fetchNrB = 0;
    //  consider an operand reused only if it is reused in all activations
    for (auto iters : recipeIterator)
    {
        recipeIterator.setCurIterVals(iters);
        reuseA &= recipe.reuseA();
        reuseB &= recipe.reuseB();
        partialReuse |= m_recipeGenerator->isPartialSBReuse();
    }

    unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
    getNumStepsPerGeometry(params, *m_geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);

    unsigned fcdSplits = recipe.getFcdSubviews().size();
    unsigned spSplits = recipe.getSpSubviews().size();
    unsigned convSplits = recipe.getNonSpatialSubviews().size();

    //  very naive solution, assume 1D reuse of the walking pattern direction operand
    //  thus the other operand will be reread according to the amount of steps for the reused operand
    switch (params.strategy.pattern)
    {
        default:
            MME_ASSERT(0, "pattern not supported");
        //  FWD/DEDX
        case e_mme_z_reduction_skf:
            m_perfAttr.fetchNrA = reuseA ? (partialReuse ? fcdSplits : 1) : fcdSteps;
            m_perfAttr.fetchNrB = reuseB ? 1 : spSteps;
            break;
        case e_mme_z_reduction_ksf:
            m_perfAttr.fetchNrB = reuseB ? (partialReuse ? spSplits : 1) : spSteps;
            m_perfAttr.fetchNrA = reuseA ? 1 : fcdSteps;
            break;
        //  DEDW/BGEMM
        case e_mme_sp_reduction_cfk:
        case e_mme_sp_reduction_fck:
            m_perfAttr.fetchNrA = reuseA ? (partialReuse ? fcdSplits : 1) : fcdSteps;
            m_perfAttr.fetchNrB = reuseB ? 1 : spSteps * (params.isGemmOperation() ? 1 : batchSteps);
            break;
        case e_mme_sp_reduction_ckf:
            if (params.isGemmOperation())
            {
                // currently there is no reused between batches in bgemm, reuse is supposed in broadcast update this
                m_perfAttr.fetchNrA = fcdSteps;
                m_perfAttr.fetchNrB = spSteps;
            }
            else  // dedw
            {
                m_perfAttr.fetchNrA = reuseA ? 1 : fcdSteps;
                m_perfAttr.fetchNrB = reuseB ? spSteps : spSteps * batchSteps;
            }
            break;
        case e_mme_sp_reduction_fkc:
            m_perfAttr.fetchNrA = reuseA ? 1 : fcdSteps;
            if (reuseB)
            {
                if (params.isGemmOperation())
                {
                    m_perfAttr.fetchNrB = partialReuse ? spSplits : 1;
                }
                else
                {
                    m_perfAttr.fetchNrB = partialReuse ? convSplits : 1;
                    m_perfAttr.fetchNrB *= batchSteps;  //  make sure this is not redundant
                }
            }
            else
            {
                m_perfAttr.fetchNrB = spSteps;
            }
            break;
        case e_mme_sp_reduction_kcf:
            if (params.isGemmOperation())
            {
                // currently there is no reused between batches in bgemm, so each movement will reread inputs
                // once broadcasting is supported update this logic
                m_perfAttr.fetchNrA = fcdSteps;
                m_perfAttr.fetchNrB = spSteps;
            }
            // fallthrought for DEDW
        case e_mme_sp_reduction_kfc:
            m_perfAttr.fetchNrA = reuseA ? 1 : fcdSteps;
            if (reuseB)
            {
                if (params.isGemmOperation())
                {
                    m_perfAttr.fetchNrB = partialReuse ? spSplits : 1;
                }
                else
                {
                    m_perfAttr.fetchNrB = partialReuse ? convSplits : 1;
                }
            }
            else
            {
                m_perfAttr.fetchNrB = spSteps * (params.isGemmOperation() ? 1 : batchSteps);
            }

            break;
    }

    // in conv operations, each pixel in operand A will be read multiple times according to the filter properties
    if (params.isConvOperation())
    {
        for (int convDim = 0; convDim < MME_MAX_CONV_DIMS; convDim++)
        {
            unsigned filterReads = div_round_up(params.w.sizes[DIM_S + convDim], params.conv.stride[convDim]);
            m_perfAttr.fetchNrA *= filterReads;
        }
    }
}

float MmeBrain::calcUtilizationImpl(const MmeLayerParams& params)
{
    unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
    getNumStepsPerGeometry(params, *m_geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);

    LOG_DEBUG(MME_BRAIN,
              "Calculating utilization of node with output size: [{}]",
              fmt::join(params.getOperand(e_mme_op_c).sizes.begin(), params.getOperand(e_mme_op_c).sizes.end(), ","));

    float constrainedStepCost = calcConstrainedStepCost(params) - 1;
    unsigned lastSpatialStepSize = params.getSpatialSize() % m_geoAttr->getGeometryHeight();
    lastSpatialStepSize = lastSpatialStepSize == 0 ? m_geoAttr->getGeometryHeight() : lastSpatialStepSize;
    float lastSpatialUtil = (float) lastSpatialStepSize / m_geoAttr->getGeometryHeight();
    LOG_TRACE(MME_BRAIN, "  last spatial util: {:.6}", lastSpatialUtil);

    unsigned lastFcdStepSize = params.getFcdSize() % m_geoAttr->getGeometryWidth();
    lastFcdStepSize = lastFcdStepSize == 0 ? m_geoAttr->getGeometryWidth() : lastFcdStepSize;
    float lastFcdUtil = (float) lastFcdStepSize / m_geoAttr->getGeometryWidth();
    LOG_TRACE(MME_BRAIN, "  last fcd util: {:.6}", lastFcdUtil);

    float lastBatchUtil = 1.0;
    if (m_geoAttr->supportsConcurrency())
    {
        int concurrentDim = m_geoAttr->getConcurrentDim();
        unsigned lastBatchStepSize = params.y.sizes[concurrentDim] % m_geoAttr->getGeometryConcurrency();
        lastBatchStepSize = lastBatchStepSize == 0 ? m_geoAttr->getGeometryConcurrency() : lastBatchStepSize;
        lastBatchUtil = (float) lastBatchStepSize / m_geoAttr->getGeometryConcurrency();
    }
    LOG_TRACE(MME_BRAIN, "  last batch util: {:.6}", lastBatchUtil);

    float fullUtilSteps = 1.0 * (fcdSteps - 1) * (spSteps - 1);  //  all fully utilized gemms
    float totalPartialHeightUtil = lastSpatialUtil * (fcdSteps - 1);  //  all gemms that have only partial height
    float totalPartialWidthUtil = lastFcdUtil * (spSteps - 1);  //  all gemms that have only partial width
    float lastGemmUtilization =
        lastSpatialUtil * lastFcdUtil;  //  the corner gemm that has both partial height and width

    float singleBatchUtil = (fullUtilSteps + totalPartialHeightUtil + totalPartialWidthUtil + lastGemmUtilization) /
                            (fcdSteps * spSteps + constrainedSteps * constrainedStepCost);
    float utilization = (singleBatchUtil * (batchSteps - 1) + singleBatchUtil * lastBatchUtil) / batchSteps;
    LOG_DEBUG(MME_BRAIN, "  overll util: {:.6}", utilization);
    return utilization;
}

float MmeBrain::calcSliceUtilization(const MmeLayerParams& params,
                                     MmeLayerParams& slicedParams,
                                     const std::vector<PhysicalAspects::Name>& aspects)
{
    int outWidthIdx = params.isConvOperation() ? DIM_K : GEMM_DIM_W;
    int inWidthIdx = params.isConvOperation()                  ? DIM_K
                     : isTransposed(params.opType, e_mme_in_b) ? GEMM_DIM_H
                                                               : GEMM_DIM_W;
    unsigned originalWidth = slicedParams.getOperand(e_mme_op_c).sizes[outWidthIdx];

    int convBatchDim = DIM_W + params.conv.spatialDimsNr;
    int outHeightIdx = params.isFwdOrDedx() ? convBatchDim : GEMM_DIM_H;
    int inHeightIdx = params.isFwdOrDedx()                      ? convBatchDim
                      : isTransposed(params.opType, e_mme_in_a) ? GEMM_DIM_H
                                                                : GEMM_DIM_W;
    unsigned originalHeight = slicedParams.getOperand(e_mme_op_c).sizes[outHeightIdx];

    auto dimTailSize = [&](EMmeInternalOperand operand, int dim) {
        const auto& sizes = params.getOperand(operand).sizes;
        const auto& slicedSizes = slicedParams.getOperand(operand).sizes;
        auto tailSize = sizes[dim] % slicedSizes[dim];
        return tailSize == 0 ? sizes[dim] : tailSize;
    };

    auto setOperandsDimsSizes = [&](const std::initializer_list<std::pair<EMmeInternalOperand, int>>& operandsAndDims,
                                    unsigned size) {
        for (const auto& [operand, dim] : operandsAndDims)
        {
            slicedParams.getOperand(operand).sizes[dim] = size;
        }
    };

    unsigned originalBatch = 0;
    std::string axisNames = "";
    for (auto aspect : aspects)
    {
        unsigned tailSize;
        switch (aspect)
        {
            default:
                MME_ASSERT(0, "unexpected aspect");
            case MmeAspect::OUTPUT_WIDTH:
                tailSize = dimTailSize(e_mme_op_c, outWidthIdx);
                setOperandsDimsSizes({{e_mme_op_c, outWidthIdx}, {e_mme_op_b, inWidthIdx}}, tailSize);
                axisNames += "width ";
                break;
            case MmeAspect::OUTPUT_HEIGHT:
                tailSize = dimTailSize(e_mme_op_c, outHeightIdx);
                setOperandsDimsSizes({{e_mme_op_c, outHeightIdx}, {e_mme_op_b, inHeightIdx}}, tailSize);
                axisNames += "height ";
                break;
            case MmeAspect::GROUPS:
                int concurrentDim = m_geoAttr->getConcurrentDim();
                originalBatch = slicedParams.getOperand(e_mme_op_c).sizes[concurrentDim];
                tailSize = dimTailSize(e_mme_op_c, concurrentDim);
                setOperandsDimsSizes(
                    {{e_mme_op_a, concurrentDim}, {e_mme_op_b, concurrentDim}, {e_mme_op_c, concurrentDim}},
                    tailSize);
                axisNames += "batch ";
                break;
        }
    }

    LOG_DEBUG(MME_BRAIN,
              "Calculating utilization of tail {} slices with output size: [{}]",
              axisNames,
              fmt::join(slicedParams.getOperand(e_mme_op_c).sizes.begin(),
                        slicedParams.getOperand(e_mme_op_c).sizes.end(),
                        ","));
    float tailSliceUtil = calcUtilizationImpl(slicedParams);

    for (auto aspect : aspects)
    {
        switch (aspect)
        {
            default:
                MME_ASSERT(0, "unexpected aspect");
            case MmeAspect::OUTPUT_WIDTH:
                setOperandsDimsSizes({{e_mme_op_c, outWidthIdx}, {e_mme_op_b, inWidthIdx}}, originalWidth);
                break;
            case MmeAspect::OUTPUT_HEIGHT:
                setOperandsDimsSizes({{e_mme_op_c, outHeightIdx}, {e_mme_op_b, inHeightIdx}}, originalHeight);
                break;
            case MmeAspect::GROUPS:
                int concurrentDim = m_geoAttr->getConcurrentDim();
                setOperandsDimsSizes(
                    {{e_mme_op_a, concurrentDim}, {e_mme_op_b, concurrentDim}, {e_mme_op_c, concurrentDim}},
                    originalBatch);
                break;
        }
    }

    return tailSliceUtil;
}

void MmeBrain::calcUtilization(const MmeLayerParams& params, std::optional<MmeLayerParams> optionalSlicedParams)
{
    LOG_DEBUG(MME_BRAIN,
              "Calculating utilization of node with output size: [{}]",
              fmt::join(params.getOperand(e_mme_op_c).sizes.begin(), params.getOperand(e_mme_op_c).sizes.end(), ","));
    m_perfAttr.maxUtilization = calcUtilizationImpl(params);

    // if given sliced params we need to calculate the node utilization considering the slicing (utilization of all slices combined)
    if (optionalSlicedParams.has_value())
    {
        auto& slicedParams = optionalSlicedParams.value();
        unsigned heightSlices = div_round_up(params.getSpatialSize(), slicedParams.getSpatialSize());
        unsigned widthSlices = div_round_up(params.getFcdSize(), slicedParams.getFcdSize());
        unsigned batchSlices = 1;
        // since we only support concurrency on the first batch dim, if a bgemm is flattened it cannot support
        // concurrency [SW-110115]
        if (m_geoAttr->supportsConcurrency() && !params.canFlatten())
        {
            int concurrentDim = m_geoAttr->getConcurrentDim();
            auto& nodeOutput = params.getOperand(e_mme_op_c).sizes;
            auto& sliceOutput = slicedParams.getOperand(e_mme_op_c).sizes;
            batchSlices = div_round_up(nodeOutput[concurrentDim], sliceOutput[concurrentDim]);
        }

        LOG_DEBUG(MME_BRAIN,
                  "Calculating utilization of a nominal slice with output size: [{}]",
                  fmt::join(slicedParams.getOperand(e_mme_op_c).sizes.begin(),
                            slicedParams.getOperand(e_mme_op_c).sizes.end(),
                            ","));
        float fullSliceUtil = calcUtilizationImpl(slicedParams);
        float heightTailSliceUtil = calcSliceUtilization(params, slicedParams, {MmeAspect::OUTPUT_HEIGHT});
        float widthTailSliceUtil = calcSliceUtilization(params, slicedParams, {MmeAspect::OUTPUT_WIDTH});
        float widthHeightTailSliceUtil =
            calcSliceUtilization(params, slicedParams, {MmeAspect::OUTPUT_WIDTH, MmeAspect::OUTPUT_HEIGHT});

        float fullSliceUtilWeighted =
            fullSliceUtil * (widthSlices - 1) * (heightSlices - 1);  //  all fully utilized gemms
        float heightTailSliceUtilWeighted =
            heightTailSliceUtil * (widthSlices - 1);  //  all gemms that have only partial height
        float widthTailSliceUtilWeighted =
            widthTailSliceUtil * (heightSlices - 1);  //  all gemms that have only partial width
        float widthHeightTailSliceUtilWeighted = widthHeightTailSliceUtil;  //  all gemms that have only partial width

        if (m_geoAttr->supportsConcurrency() && batchSlices > 1)
        {
            fullSliceUtilWeighted *= (batchSlices - 1);  //  all fully utilized gemms
            heightTailSliceUtilWeighted *= (batchSlices - 1);  //  all gemms that have only partial height
            widthTailSliceUtilWeighted *= (batchSlices - 1);  //  all gemms that have only partial width
            widthHeightTailSliceUtilWeighted *= (batchSlices - 1);

            float batchTailSliceUtil = calcSliceUtilization(params, slicedParams, {MmeAspect::GROUPS});
            float heightBatchTailSliceUtil =
                calcSliceUtilization(params, slicedParams, {MmeAspect::OUTPUT_HEIGHT, MmeAspect::GROUPS});
            float widthBatchTailSliceUtil =
                calcSliceUtilization(params, slicedParams, {MmeAspect::OUTPUT_WIDTH, MmeAspect::GROUPS});
            float lastTailSliceUtil =
                calcSliceUtilization(params,
                                     slicedParams,
                                     {MmeAspect::OUTPUT_WIDTH, MmeAspect::OUTPUT_HEIGHT, MmeAspect::GROUPS});

            float batchTailSliceUtilWeighted =
                batchTailSliceUtil * (widthSlices - 1) * (heightSlices - 1);  //  all fully utilized gemms
            float heightBatchTailSliceUtilWeighted =
                heightBatchTailSliceUtil * (widthSlices - 1);  //  all gemms that have only partial height
            float widthBatchTailSliceUtilWeighted =
                widthBatchTailSliceUtil * (heightSlices - 1);  //  all gemms that have only partial width
            float lastTailSliceUtilWeighted =
                lastTailSliceUtil;  //  the corner gemm that has both partial height and width

            float slicedUtil =
                (fullSliceUtilWeighted + heightTailSliceUtilWeighted + widthTailSliceUtilWeighted +
                 lastTailSliceUtilWeighted + batchTailSliceUtilWeighted + heightBatchTailSliceUtilWeighted +
                 widthBatchTailSliceUtilWeighted + widthHeightTailSliceUtilWeighted) /
                (widthSlices * heightSlices * batchSlices);
            m_perfAttr.mmeUtilization = slicedUtil;
            LOG_DEBUG(MME_BRAIN,
                      "final sliced util: {:.6}, with {} width slices {} height slices and {} batch slices",
                      slicedUtil,
                      widthSlices,
                      heightSlices,
                      batchSlices);
        }
        else
        {
            float slicedUtil = (fullSliceUtilWeighted + heightTailSliceUtilWeighted + widthTailSliceUtilWeighted +
                                widthHeightTailSliceUtil) /
                               (widthSlices * heightSlices);
            m_perfAttr.mmeUtilization = slicedUtil;
            LOG_DEBUG(MME_BRAIN,
                      "final sliced util: {:.6}, with {} width slices and {} height slices",
                      slicedUtil,
                      widthSlices,
                      heightSlices);
        }
    }
    else
    {
        m_perfAttr.mmeUtilization = m_perfAttr.maxUtilization;
    }
}

void MmeBrain::calcMemoryAttributes(const MmeLayerParams& params, const bool cdPerforation)
{
    // the calcualtion below should have been done at fetchNr stage, eventually this should move there.
    unsigned totalAFetches = m_perfAttr.fetchNrA * m_geoAttr->getFcdMmeNr(e_mme_op_c);
    unsigned totalBFetches = m_perfAttr.fetchNrB * m_geoAttr->getSpatialMmeNr(e_mme_op_c);
    m_perfAttr.memoryAttrA.accessesPerChip = totalAFetches;
    m_perfAttr.memoryAttrB.accessesPerChip = totalBFetches;
    m_perfAttr.memoryAttrA.accessesPerDcore = totalAFetches;
    m_perfAttr.memoryAttrB.accessesPerDcore = totalBFetches;

    std::vector<unsigned> perforationDimVec = {};
    getPerforationDim(params, perforationDimVec, cdPerforation);
    // Perf dim are on the same direction - so each one has the same effect on #accesses
    if (!perforationDimVec.empty())
    {
        switch (params.opType)
        {
            case e_mme_ab:
            case e_mme_atb:
            case e_mme_abt:
            case e_mme_atbt:
                switch (perforationDimVec.front())
                {
                    case Gemm::DIM_OUT_FCD:
                        m_perfAttr.memoryAttrA.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    case Gemm::DIM_OUT_HEIGHT:
                        m_perfAttr.memoryAttrB.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    default:
                        if (params.canFlatten() && perforationDimVec.front() == Gemm::DIM_BATCH_0)
                        {
                            m_perfAttr.memoryAttrB.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        }
                        break;
                }
                break;
            case e_mme_fwd:
                switch (perforationDimVec.front())
                {
                    case Conv::DIM_OUT_CHANNELS:
                        m_perfAttr.memoryAttrA.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    case Conv::DIM_BATCH:
                        m_perfAttr.memoryAttrB.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    default:
                        break;
                }
                break;
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                switch (perforationDimVec.front())
                {
                    case Conv::DIM_IN_CHANNELS:
                        m_perfAttr.memoryAttrA.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    case Conv::DIM_BATCH:
                        m_perfAttr.memoryAttrB.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    default:
                        break;
                }
                break;
            case e_mme_dedw:
                switch (perforationDimVec.front())
                {
                    case Conv::DIM_OUT_CHANNELS:
                        m_perfAttr.memoryAttrA.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    case Conv::DIM_IN_CHANNELS:
                        m_perfAttr.memoryAttrB.accessesPerDcore /= m_mmeHal.getDcoreNr();
                        break;
                    default:
                        break;
                }
                break;
            default:
                MME_ASSERT(0, "unexpected opType");
        }
    }

    // initialize access BW to a single port BW - 200 GHz
    m_perfAttr.memoryAttrA.accessBW = m_mmeHal.getSinglePortBw() * m_geoAttr->getChipPortsNr(e_mme_op_a);
    m_perfAttr.memoryAttrB.accessBW = m_mmeHal.getSinglePortBw() * m_geoAttr->getChipPortsNr(e_mme_op_b);

    // Handle SB reuse
    const auto& recipe = m_recipeGenerator->get();
    if (recipe.reuseA() || recipe.reuseB())
    {
        // In partial SB reuse, the reused steps are constrained by number of accums
        unsigned accumsStepsNr = -1U;
        if (m_recipeGenerator->isPartialSBReuse())
        {
            accumsStepsNr = m_mmeHal.getAccumsNr() * (m_geoAttr->getDoubleAccumsBit() ? 2 : 1);
        }

        // The more reuse steps there are, the less access BW (in average) will be
        unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
        getNumStepsPerGeometry(params, *m_geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);
        if (recipe.reuseA())
        {
            m_perfAttr.memoryAttrA.accessBW /= std::min(fcdSteps, accumsStepsNr);
        }
        if (recipe.reuseB())
        {
            m_perfAttr.memoryAttrB.accessBW /= std::min(spSteps, accumsStepsNr);
        }
    }
}

void MmeBrain::getRecommendedGeometryAndPattern(MmeLayerParams& params, bool isGeoPreferredShort)
{
    trivialDimsReduction(params);
    //  geometry is first set according to output size
    if (params.strategy.geometry == e_mme_geometry_nr)
    {
        chooseGeometry(params, isGeoPreferredShort);
    }
    //  walking pattern is set to maximize reuse over the wider operand
    if (params.strategy.pattern == e_mme_patterns_nr)
    {
        chooseWalkingPattern(params);
    }
    // Make sure geoAttr is in sync with the selected geo
    m_geoAttr = getGeoAttr(m_chipType, params);

    m_flattening = applyTensorFlattening(params);
}

static float getBatchAcceleration(unsigned concurrency, unsigned filterSize, unsigned numSpatialSteps)
{
    // Normalize the acceleration by the filter size
    unsigned numStepsForSingleBatch = div_round_up(filterSize, concurrency);
    return ((float) filterSize) / (numStepsForSingleBatch * numSpatialSteps);
}
static float getCdConcurrencyAcceleration(const MmeLayerParams& params,
                                          const upCommonGeoAttr& geoAttr,
                                          const MmeHalReader& mmeHal,
                                          unsigned concurrency,
                                          unsigned numSpatialSteps)
{
    float cdConcurrencyAcceleration = ((float) concurrency) / numSpatialSteps;
    // Reduce the effective cd concurrency in case of recurring unalignment and use of CD dim 0.
    if (RecurringMisalignmentOptimization::isRecurringMisalignment(params, mmeHal, e_mme_op_a) &&
        ((geoAttr->getSpInterleavingDim(MmeCommon::e_mme_op_a) == MmeCommon::DIM_W) ||
         (geoAttr->getSpInterleavingDim(MmeCommon::e_mme_op_b) == MmeCommon::DIM_W)))
    {
        cdConcurrencyAcceleration /= 2;
    }
    return cdConcurrencyAcceleration;
}

void MmeBrain::applyCdConcurrency(MmeLayerParams& paramsForCdConcurrency, float& cdConcurrencyAcceleration)
{
    cdConcurrencyAcceleration = 0;
    if (paramsForCdConcurrency.strategy.cdConcurrencyEn != TurnedOff)
    {
        // Set param to perform cd concurrency
        paramsForCdConcurrency.strategy.cdConcurrencyEn = TurnedOn;
        paramsForCdConcurrency.strategy.batchConcurrencyEn = TurnedOff;
        paramsForCdConcurrency.memoryCfg.reductionOp = e_mme_reduction_add;
        getRecommendedGeometryAndPattern(paramsForCdConcurrency);

        auto geoAttrCdCon = MmeCommon::getGeoAttr(m_chipType, paramsForCdConcurrency);
        paramsForCdConcurrency.strategy.reductionLevel = geoAttrCdCon->getGeometryCdConcurrency();
        choosePackingFactorForReductionAdd(paramsForCdConcurrency);

        unsigned numSpatialSteps = getNumSpatialSteps(paramsForCdConcurrency, *m_geoAttr);
        cdConcurrencyAcceleration = getCdConcurrencyAcceleration(paramsForCdConcurrency,
                                                                 geoAttrCdCon,
                                                                 m_mmeHal,
                                                                 geoAttrCdCon->getGeometryCdConcurrency(),
                                                                 numSpatialSteps);
    }
}

void MmeBrain::applyBatchConcurrency(MmeLayerParams& paramsForBatchConcurrency, float& batchConcurrencyAcceleration)
{
    batchConcurrencyAcceleration = 0;
    // Drop batch concurrency if the node is convertible to gemm
    if ((paramsForBatchConcurrency.strategy.batchConcurrencyEn != TurnedOff) &&
        !dedwIsConvertibleToGemm(paramsForBatchConcurrency))
    {
        // Set param to perform batch concurrency
        paramsForBatchConcurrency.strategy.batchConcurrencyEn = TurnedOn;
        paramsForBatchConcurrency.strategy.cdConcurrencyEn = TurnedOff;
        getRecommendedGeometryAndPattern(paramsForBatchConcurrency);

        auto geoAttrBatchCon = MmeCommon::getGeoAttr(m_chipType, paramsForBatchConcurrency);
        unsigned numSpatialSteps = getNumSpatialSteps(paramsForBatchConcurrency, *m_geoAttr);
        batchConcurrencyAcceleration = getBatchAcceleration(
            geoAttrBatchCon->getGeometryConcurrency(),
            paramsForBatchConcurrency.getOperand(e_mme_op_c).sizes[geoAttrBatchCon->getConcurrentDim()],
            numSpatialSteps);
    }
}

void MmeBrain::applyHybridConcurrency(MmeLayerParams& paramsForHybridConcurrency, float& hybridConcurrencyAcceleration)
{
    hybridConcurrencyAcceleration = 0;
    // Drop batch concurrency if the node is convertible to gemm
    if ((paramsForHybridConcurrency.strategy.cdConcurrencyEn != TurnedOff) &&
        (paramsForHybridConcurrency.strategy.batchConcurrencyEn != TurnedOff))
    {
        // Set param to perform both cd batch concurrency
        paramsForHybridConcurrency.strategy.batchConcurrencyEn = TurnedOn;
        paramsForHybridConcurrency.strategy.cdConcurrencyEn = TurnedOn;
        getRecommendedGeometryAndPattern(paramsForHybridConcurrency);

        auto geoAttr = MmeCommon::getGeoAttr(m_chipType, paramsForHybridConcurrency);
        paramsForHybridConcurrency.strategy.reductionLevel = geoAttr->getGeometryCdConcurrency();
        choosePackingFactorForReductionAdd(paramsForHybridConcurrency);

        unsigned numSpatialSteps = getNumSpatialSteps(paramsForHybridConcurrency, *m_geoAttr);
        float cdConcurrencyAcceleration = getCdConcurrencyAcceleration(paramsForHybridConcurrency,
                                                                       geoAttr,
                                                                       m_mmeHal,
                                                                       geoAttr->getGeometryCdConcurrency(),
                                                                       numSpatialSteps);
        float batchAcceleration =
            getBatchAcceleration(geoAttr->getGeometryConcurrency(),
                                 paramsForHybridConcurrency.getOperand(e_mme_op_c).sizes[geoAttr->getConcurrentDim()],
                                 numSpatialSteps);
        hybridConcurrencyAcceleration = batchAcceleration * cdConcurrencyAcceleration;
    }
}

void MmeBrain::chooseConcurrency(MmeLayerParams& params)
{
    // If any of the concurrencies is turned off, we skip it.
    auto paramsForCdConcurrency = params;
    auto paramsForBatchConcurrency = params;
    auto paramsForHybridConcurrency = params;
    auto paramsForNoConcurrency = params;
    paramsForNoConcurrency.strategy.batchConcurrencyEn = TurnedOff;
    paramsForNoConcurrency.strategy.cdConcurrencyEn = TurnedOff;

    float cdConcurrencyAcceleration = 0;  // initial value
    float batchConcurrencyAccelleration = 0;
    float hybridConcurrencyAccelleration = 0;

    applyCdConcurrency(paramsForCdConcurrency, cdConcurrencyAcceleration);
    applyBatchConcurrency(paramsForBatchConcurrency, batchConcurrencyAccelleration);
    applyHybridConcurrency(paramsForHybridConcurrency, hybridConcurrencyAccelleration);

    // Choose batch concurrency if it achieves higher parallelism, otherwise cd concurrency
    // todo [SW-107825]: Improve the decision between batch concurrency and cd concurrency

    // Choose according to these options:
    // 1) batch concurrency >= others           => Choose batch
    // 2) else, hybrid concurrency >= others    => Choose hybrid
    // 3) else, cd concurrency > 1              => Choose cd
    // 4) else,                                 => Choose no concurrency
    if ((batchConcurrencyAccelleration >= cdConcurrencyAcceleration) &&
        (batchConcurrencyAccelleration >= hybridConcurrencyAccelleration))
    {
        params = paramsForBatchConcurrency;
    }
    else if ((hybridConcurrencyAccelleration >= batchConcurrencyAccelleration) &&
             (hybridConcurrencyAccelleration >= cdConcurrencyAcceleration))
    {
        params = paramsForHybridConcurrency;
    }
    else if ((cdConcurrencyAcceleration >= batchConcurrencyAccelleration) &&
             (cdConcurrencyAcceleration >= hybridConcurrencyAccelleration))
    {
        params = paramsForCdConcurrency;
    }
    else
    {
        params = paramsForNoConcurrency;
    }
    m_geoAttr = getGeoAttr(m_chipType, params);
}

void MmeBrain::getRecommendedStrategy(MmeLayerParams& params, bool isGeoPreferredShort)
{
    // MME Brain chooses strategy and pattern fields for all ops
    // In addition, for some ops it chooses also the concurrency.
    // Concurrency should come first

    // If the op supports choosing concurrency, and at least one of the concurrency fields
    // is in undefined state, choose concurrency
    if (opSupportsChoosingConcurrency(params.opType))
    {
        if (params.strategy.cdConcurrencyEn == Undefined || params.strategy.batchConcurrencyEn == Undefined)
        {
            chooseConcurrency(params);
        }
    }
    else
    {
        // set default values to avoid failing on asserts later on
        params.strategy.cdConcurrencyEn =
            params.strategy.cdConcurrencyEn == Undefined ? TurnedOff : params.strategy.cdConcurrencyEn;
        params.strategy.batchConcurrencyEn =
            params.strategy.batchConcurrencyEn == Undefined ? TurnedOn : params.strategy.batchConcurrencyEn;
    }

    getRecommendedGeometryAndPattern(params, isGeoPreferredShort);
}

std::vector<EMmeGeometry> MmeBrain::getSortedGeometries(MmeLayerParams& params, bool isGeoPreferredShort)
{
    if (m_chipType == e_mme_Gaudi)
    {
        if (isGeoPreferredShort)
        {
            return {e_mme_geometry_4wx1h, e_mme_geometry_2wx2h, e_mme_geometry_1wx4h};
        }
        return {e_mme_geometry_1wx4h, e_mme_geometry_2wx2h, e_mme_geometry_4wx1h};
    }
    else if (m_chipType == e_mme_Gaudi2)
    {
        if (isGeoPreferredShort)
        {
            return {e_mme_geometry_4xw, e_mme_geometry_2xw, e_mme_geometry_2xh, e_mme_geometry_4xh};
        }
        return {e_mme_geometry_4xh, e_mme_geometry_2xh, e_mme_geometry_2xw, e_mme_geometry_4xw};
    }
    else if (m_chipType == e_mme_Gaudi3)
    {
        if (params.isDmaOperation())
        {
            //  TODO support 2xW geometry for memcpy
            return {e_mme_geometry_4xw};
        }
        if (isGeoPreferredShort)
        {
            return {e_mme_geometry_4xw, e_mme_geometry_2xw, e_mme_geometry_2xh, e_mme_geometry_4xh};
        }
        return {e_mme_geometry_4xh, e_mme_geometry_2xh, e_mme_geometry_2xw, e_mme_geometry_4xw};
    }
    else
    {
        MME_ASSERT(0, "chip type not supported by MME Brain");
        return {};
    }
}

std::vector<EMmeGeometry> MmeBrain::getGeometries(MmeLayerParams& params)
{
    if (m_chipType == e_mme_Gaudi)
    {
        return {e_mme_geometry_4wx1h, e_mme_geometry_2wx2h, e_mme_geometry_1wx4h};
    }
    else if (m_chipType == e_mme_Gaudi2)
    {
        return {e_mme_geometry_2xh, e_mme_geometry_2xw, e_mme_geometry_4xh, e_mme_geometry_4xw};
    }
    else if (m_chipType == e_mme_Gaudi3)
    {
        constexpr unsigned mmeSize = 256;
        std::vector<EMmeGeometry> geoemtries;
        const unsigned fcdSize = params.getFcdSize();
        const unsigned spSize = params.getSpatialSize();
        if (params.isNativeDmaOperation())
        {
            geoemtries = {e_mme_geometry_4xw, e_mme_geometry_2xw, e_mme_geometry_2xh, e_mme_geometry_4xh};
        }
        if (params.isGemmOperation() && params.getBatchSize() >= m_mmeHal.getDcoreNr())
        {
            // if we have at least 4 batches it is always preferable to perforate over the batch dim.
            // currently this is also the logic in getPerforationDim. in case that logic changes this one might need to
            // change too. return two geometries that will represent the two possible internal dcore geometries
            geoemtries = {e_mme_geometry_4xh, e_mme_geometry_4xw};
            // in case the gemm is smaller than a single MME than there is only one solution, full batch concurrency
            // remove one of the geoemtries as all geometries will produce the same final solution
            if (fcdSize <= mmeSize && spSize <= mmeSize)
            {
                geoemtries.resize(1);
            }
            return geoemtries;
        }
        else
        {
            // cant perforate over batch dims, check relevant geometries
            if (fcdSize >= 2 * mmeSize)
            {
                if (spSize >= 2 * mmeSize)
                {
                    geoemtries.push_back(e_mme_geometry_2xw);
                }
                if (fcdSize >= 4 * mmeSize)
                {
                    geoemtries.push_back(e_mme_geometry_4xw);
                }
            }
            if (spSize >= 2 * mmeSize)
            {
                if (fcdSize > mmeSize)
                {
                    geoemtries.push_back(e_mme_geometry_2xh);
                }
                if (spSize >= 4 * mmeSize)
                {
                    geoemtries.push_back(e_mme_geometry_4xh);
                }
            }
            if (geoemtries.empty()) return {e_mme_geometry_2xh};
            return geoemtries;
        }
    }
    else
    {
        MME_ASSERT(0, "chip type not supported by MME Brain");
        return {};
    }
}

std::vector<EMmePattern> MmeBrain::getPatterns(MmeLayerParams& params)
{
    std::vector<EMmePattern> patterns;
    auto geoAttr = getGeoAttr(m_chipType, params);
    unsigned fcdSteps, spSteps, batchSteps, constrainedSteps;
    getNumStepsPerGeometry(params, *geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);
    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_transposed_dedx:
            if (fcdSteps > 1) patterns.push_back(e_mme_z_reduction_skf);
            if (spSteps > 1) patterns.push_back(e_mme_z_reduction_ksf);
            if (patterns.empty()) return {e_mme_z_reduction_skf};
            return patterns;
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
        {
            getNumStepsPerGeometry(params, *geoAttr, fcdSteps, spSteps, batchSteps, constrainedSteps);
            if (fcdSteps > 1) patterns.push_back(e_mme_sp_reduction_fck);
            if (spSteps > 1) patterns.push_back(params.isDedwOperation() ? e_mme_sp_reduction_kfc : e_mme_sp_reduction_fkc);
            if (patterns.empty()) return {e_mme_sp_reduction_fck};
            return patterns;
        }
        default:
            MME_ASSERT(0, "unsupported operation");
            return {};
    }
}

void MmeBrain::chooseGeometry(MmeLayerParams& params, bool isGeoPreferredShort)
{
    uint64_t minRuntimeCycles = -1;
    EMmeGeometry bestGeo = e_mme_geometry_nr;
    std::vector<EMmeGeometry> geometries;
    if (m_knobs.operationModes.addTieBreakerPreferredReuseOperand)
    {
        geometries = getSortedGeometries(params, isGeoPreferredShort);
    }
    else
    {
        geometries = getGeometries(params);
    }
    bool walkingPatternChanged = false;
    const MmeLayerParams origParams = params;
    for (EMmeGeometry curGeo : geometries)
    {
        params.strategy.geometry = curGeo;
        m_geoAttr = getGeoAttr(m_chipType, params);
        if (params.strategy.pattern == e_mme_patterns_nr)
        {
            chooseWalkingPattern(params);
            walkingPatternChanged = true;
        }
        // Flatten the tensors according to the geometry, if applicable
        applyTensorFlattening(params);
        PerfAttr perfAttr;
        getPerfAttr(params, perfAttr);

        if (perfAttr.expectedRuntimeCycles < minRuntimeCycles)
        {
            minRuntimeCycles = perfAttr.expectedRuntimeCycles;
            bestGeo = curGeo;
        }
        if (walkingPatternChanged)
        {
            params.strategy.pattern = e_mme_patterns_nr;
        }
        // Restore tensor dims
        params.x = origParams.x;
        params.w = origParams.w;
        params.y = origParams.y;
    }
    MME_ASSERT(bestGeo != e_mme_geometry_nr, "geometry choosing failed");
    params.strategy.geometry = bestGeo;
}

// This function identifies the best packing factor for the gemm operation that implements the reductionAdd
// Todo: SW-152671 - DCDC - Merge packing for dcdc with the packing logic in gc
void MmeBrain::choosePackingFactorForReductionAdd(MmeLayerParams& origParams)
{
    // reductionAdd is currently applicable for deterministic cd concurrency only
    if (origParams.strategy.cdConcurrencyEn != TurnedOn || !origParams.strategy.isDeterministic)
    {
        return;
    }
    // Sanity check
    MME_ASSERT(origParams.opType == e_mme_dedw, "Deterministic cd concurrency is currently supported for dedw only");

    unsigned reductionLevel = origParams.strategy.reductionLevel;

    auto gemmParams = origParams;
    gemmParams.opType = e_mme_ab;
    gemmParams.y.sizes = origParams.w.sizes;    // gemm output is identical to the dedw output
    gemmParams.w.sizes = origParams.w.sizes;    // gemm output is Nx compared to the dedw output
    gemmParams.w.sizes[MME_MAX_TENSOR_DIMS -1] *= reductionLevel;

    unsigned packingFactor = 1;
    unsigned bestPackingFactor = 1;
    uint64_t minRuntimeCycles = -1;

    unsigned numOutputElements = multiplyElements(gemmParams.y.sizes.begin(), gemmParams.y.sizes.end());

    for (int d = MME_MAX_TENSOR_DIMS-1; d >= 0; d--)
    {
        // Flatten all tensors according to the packing factor
        // x tensor
        gemmParams.x.sizes = {packingFactor * reductionLevel, packingFactor, 1, 1, 1 };
        calcContiguousStrides(gemmParams.x.strides, gemmParams.x.sizes);
        // w tensor
        gemmParams.w.sizes = {numOutputElements / packingFactor, packingFactor * reductionLevel, 1, 1, 1};
        calcContiguousStrides(gemmParams.w.strides, gemmParams.w.sizes);
        // y tensor
        gemmParams.y.sizes = {numOutputElements / packingFactor, packingFactor, 1, 1, 1};
        calcContiguousStrides(gemmParams.y.strides, gemmParams.y.sizes);

        PerfAttr perfAttr;
        getPerfAttr(gemmParams, perfAttr);
        if (perfAttr.expectedRuntimeCycles <= minRuntimeCycles)
        {
            minRuntimeCycles = perfAttr.expectedRuntimeCycles;
            bestPackingFactor = packingFactor;
        }
        packingFactor *= origParams.w.sizes[d];
    }

    origParams.strategy.packingFactor = bestPackingFactor;
}

void MmeBrain::chooseWalkingPattern(MmeLayerParams& params)
{
    MME_ASSERT(params.strategy.geometry != e_mme_geometry_nr, "geometry must be set before selecting walking pattern");

    if (params.isDmaOperation())
    {
        // only a single walking pattern is supported for dma operations
        params.strategy.pattern = e_mme_sp_reduction_fck;
    }
    else if (params.isFwdOrDedx())
    {
        chooseConvWalkingPattern(params);
    }
    else
    {
        chooseBgemmWalkingPattern(params);
    }
}

void MmeBrain::chooseConvWalkingPattern(MmeLayerParams& params)
{
    unsigned fcdSteps = div_round_up(params.getFcdSize(), m_geoAttr->getGeometryWidth());
    unsigned spSteps = div_round_up(params.getSpatialSize(), m_geoAttr->getGeometryHeight());
    switch (params.strategy.geometry)
    {
        default:
            MME_ASSERT(0, "geometry not supported");
        case e_mme_geometry_4xw:
            params.strategy.pattern = (spSteps > 1) ? e_mme_z_reduction_ksf : e_mme_z_reduction_skf;
            break;
        case e_mme_geometry_4xh:
            params.strategy.pattern = (fcdSteps > 1) ? e_mme_z_reduction_skf : e_mme_z_reduction_ksf;
            break;
        case e_mme_geometry_2xw:
        case e_mme_geometry_2xh:
            //  naively choose FCD first walk, improve in the future.
            params.strategy.pattern = (fcdSteps > 1) ? e_mme_z_reduction_skf : e_mme_z_reduction_ksf;
            break;
    }
}

void MmeBrain::chooseBgemmWalkingPattern(MmeLayerParams& params)
{
    unsigned fcdSteps = div_round_up(params.getFcdSize(), m_geoAttr->getGeometryWidth());
    unsigned spSteps = div_round_up(params.getSpatialSize(), m_geoAttr->getGeometryHeight());
    switch (params.strategy.geometry)
    {
        default:
            MME_ASSERT(0, "geometry not supported");
        case e_mme_geometry_4xw:
            if (spSteps > 1)  // there are SP steps, SP first
            {
                params.strategy.pattern =
                    (params.isDedwOperation()) ? e_mme_sp_reduction_kfc : e_mme_sp_reduction_fkc;
            }
            else  // no SP steps, FCD first
            {
                params.strategy.pattern = e_mme_sp_reduction_fck;
            }
            break;
        case e_mme_geometry_4xh:
            // the symmetric geometries do not have a native preference. Naively choose FCD first walk, improve in the
            // future.
        case e_mme_geometry_2xw:
        case e_mme_geometry_2xh:
            if (fcdSteps > 1)  // there are FCD steps, FCD first
            {
                params.strategy.pattern = e_mme_sp_reduction_fck;
            }
            else  // No FCD steps, SP first
            {
                params.strategy.pattern =
                    (params.isDedwOperation()) ? e_mme_sp_reduction_kfc : e_mme_sp_reduction_fkc;
            }
            break;
    }
}

bool MmeBrain::gaudiShouldUnroll(MmeLayerParams& params)
{
    gaudi::GeoAttr geoParams(params, gaudi::MmeHalReader::getInstance());
    gaudi::DedwUnroll dedwUnroll(params, geoParams);
    return dedwUnroll.shouldUnroll();
}

unsigned MmeBrain::getRollUpTime()
{
    // TODO: need to consider output BW and dtype in Gaudi.
    switch (m_chipType)
    {
        case e_mme_Gaudi:
            return 128;
            break;
        case e_mme_Gaudi2:
            return 256;
            break;
        case e_mme_Gaudi3:
            return 512;  // to be verified;
            break;
        default:
            MME_ASSERT(0, "chip type not supported by MME Brain");
    }
    return 0;
}

void MmeBrain::calcNumOfActivations()
{
    const auto& recipe = m_recipeGenerator->get();
    unsigned fcdSplits = recipe.getFcdSubviews().size();
    unsigned spSplits = recipe.getSpSubviews().size();
    unsigned convSplits = recipe.getNonSpatialSubviews().size();

    m_perfAttr.numOfActivations = fcdSplits * spSplits * convSplits;
}

// A conv node is convertible to bgemm if paddings are all 0, strides are all 1,
// dilations are all 1, kernel size is 1x1x1 and x and y tensors are dense
bool MmeBrain::convIsConvertibleToGemm(MmeCommon::MmeLayerParams& params)
{
    if (params.x.isStrided() || params.y.isStrided())
    {
        return false;
    }
    for (unsigned d = 0; d < Mme::c_mme_max_conv_dims - 1; d++)
    {
        if ((params.conv.padding[d] != 0) || (params.conv.stride[d] != 1) || (params.conv.dilation[d] != 1) ||
            (params.w.sizes[d + 2] != 1))
        {
            return false;
        }
    }
    return true;
}

bool MmeBrain::bgemmCanBeFlattened(MmeCommon::MmeLayerParams& params)
{
    // Check that there is a broadcast on the first batch dim of B tensor
    if (!((params.w.sizes[2] == 1) && (params.x.sizes[2] != 1) && (params.y.sizes[2] != 1)))
    {
        return false;
    }
    if (params.x.strides[2] != (params.x.strides[1] * params.x.sizes[1]))
    {
        return false;
    }
    if (params.y.strides[2] != (params.y.strides[1] * params.y.sizes[1]))
    {
        return false;
    }

    return true;
}

unsigned MmeBrain::flattenBgemm(MmeCommon::MmeLayerParams& params)
{
    unsigned flatteningFactor = 1;
    // Batch size is taken from y because x might be broadcasted
    unsigned gemmHeight = params.y.sizes[1];
    unsigned batchSize = params.y.sizes[2];
    unsigned batchDcoreBase = params.y.dcoreBases[2];
    m_geoAttr = getGeoAttr(m_chipType, params);
    if (m_geoAttr->getGeometryHeight() % gemmHeight == 0)
    {
        // if we have an ideal flattening factor simply select it.
        // in this case we wont apply the flattening as it might not be a divider of the batch dim.
        // this flattening factor will help GC select the ideal slice size, and the actual flattening will be done by
        // the MME recipe.
        unsigned idealFactor = m_geoAttr->getGeometryHeight() / gemmHeight;
        // since port constrained geometries rely on several steps to ammortize the first activation cost
        // we cant return a flattening factor that will only fill a single geometry.
        // for now request the entire batch dim, optimize later.
        if (m_geoAttr->isGeometryPortConstrained()) return batchSize;
        if (idealFactor < batchSize) return idealFactor;
    }
    // find all divisors of the full batch
    std::vector<unsigned> divisors = getAllDivisors(batchSize);
    // Now that we have all divisors at hand, find the one of the best utilization
    float bestUtilization = 0.0f;
    unsigned bestDivisor = 1;
    for (auto divisor : divisors)
    {
        // Find the utilization of a single gemm
        MmeLayerParams newParams = params;
        // This utilization check needs to be made without recipe flattening otherwise all the results will be the same
        // since the recipe will perform full flattening
        newParams.strategy.flattenEn = false;
        // Update the tensor dims according to the flattening factor (divisor)
        newParams.x.sizes[1] = newParams.y.sizes[1] = gemmHeight * divisor;
        newParams.x.sizes[2] = newParams.y.sizes[2] = batchSize / divisor;
        newParams.x.dcoreBases[2] = batchDcoreBase / divisor;
        newParams.y.dcoreBases[2] = batchDcoreBase / divisor;
        // update the strides accordingly
        newParams.x.strides[2] = newParams.x.strides[1] * newParams.x.sizes[1];
        newParams.y.strides[2] = newParams.y.strides[1] * newParams.y.sizes[1];

        m_geoAttr = getGeoAttr(m_chipType, newParams);
        calcUtilization(newParams);
        float utilization = m_perfAttr.mmeUtilization;

        uint64_t maxFcd = std::max(newParams.x.sizes[0], newParams.y.sizes[0]);
        uint64_t flattenedSp = newParams.x.sizes[1];
        uint64_t gemmSize = maxFcd * flattenedSp;

        if (((utilization > bestUtilization) || ((utilization == bestUtilization) && (divisor < bestDivisor))) &&
            (gemmSize < m_knobs.maxTileSize))
        {
            bestUtilization = utilization;
            bestDivisor = divisor;
            // Keep the params that reflect the best flattening
            params.x = newParams.x;
            params.y = newParams.y;
        }
    }

    return bestDivisor;
}

// Tensor flattening can be applied in a few cases
unsigned MmeBrain::applyTensorFlattening(MmeCommon::MmeLayerParams& params)
{
    if (!params.strategy.flattenEn)
    {
        return 1;
    }

    // Case 1: bgemm, where A is non-transposed and B first dim is broadcast
    //    x shape: [32k, 96, 40]            x shape: [32k, 96*40, 1]
    //    w shape: [1k, 32k, 1]     -->     w shape: [1k, 32k, 1]
    //    y shape: [1k, 96, 40]             y shape: [1k, 96*40, 1]
    if (params.opType == e_mme_ab || params.opType == e_mme_abt)
    {
        // Check that there is a broadcast on the first batch dim of B tensor
        if (bgemmCanBeFlattened(params))
        {
            return flattenBgemm(params);
        }
        else
        {
            return 1;
        }
    }

    // Case 2: conv, which can be mapped to bgemm
    // Meaning: kernel is 1x1x1, padding 0, dilation 1, stride 1.
    //    x = {20, 10, 4, 5, 2}          x = {20, 400, 1, 1, 1}
    //    w = {100, 20, 1, 1, 1}   -->   w = {100, 20, 1, 1, 1}
    //    y = {100, 10, 4, 5, 2}         y = {100, 400, 1, 1, 1}
    if (params.isConvOperation())
    {
        // Temporarily disable flattening until GC disables it in case of dynamic nodes
        return 1;

        // Check that flattening conditions are met
        if (convIsConvertibleToGemm(params))
        {
            unsigned flatteningFactor = (params.x.sizes[2] * params.x.sizes[3] * params.x.sizes[4]);
            params.x.sizes[1] *= flatteningFactor;
            params.y.sizes[1] *= flatteningFactor;
            params.x.sizes[2] = params.x.sizes[3] = params.x.sizes[4] = 1;
            params.y.sizes[2] = params.y.sizes[3] = params.y.sizes[4] = 1;
            // update the strides accordingly
            params.x.strides[2] = params.x.strides[1] * params.x.sizes[1];
            params.y.strides[2] = params.y.strides[1] * params.y.sizes[1];
            return flatteningFactor;
        }
    }

    return 1;
}

/*
 * shift dimension up in case of a trivial dimension
 */
void MmeBrain::shiftDimensions(MmeCommon::MmeLayerParams& params, unsigned trivialDim, unsigned distance)
{
    bool isBgemm = params.isGemmOperation();
    bool isMaskBgemm = params.strategy.maskedBgemm;

    for (int i = 0; i < distance; ++i)
    {
        for (unsigned dim = trivialDim; dim < Mme::c_mme_max_tensor_dims - 1; ++dim)
        {
            // record permutation
            params.permutation[dim + 1] = dim;
            
            unsigned weightDim = isBgemm ? dim : dim + 1;
            params.x.sizes[dim] = params.x.sizes[dim + 1];
            params.x.bases[dim] = params.x.bases[dim + 1];
            params.x.dcoreBases[dim] = params.x.dcoreBases[dim + 1];
            params.x.strides[dim] = params.x.strides[dim + 1];
            params.y.sizes[dim] = params.y.sizes[dim + 1];
            params.y.bases[dim] = params.y.bases[dim + 1];
            params.y.dcoreBases[dim] = params.y.dcoreBases[dim + 1];
            params.y.strides[dim] = params.y.strides[dim + 1];
            if (isMaskBgemm)
            {
                params.xAux.sizes[dim] = params.xAux.sizes[dim + 1];
                params.xAux.bases[dim] = params.xAux.bases[dim + 1];
                params.xAux.dcoreBases[dim] = params.xAux.dcoreBases[dim + 1];
                params.xAux.strides[dim] = params.xAux.strides[dim + 1];
                params.yAux.sizes[dim] = params.yAux.sizes[dim + 1];
                params.yAux.bases[dim] = params.yAux.bases[dim + 1];
                params.yAux.dcoreBases[dim] = params.yAux.dcoreBases[dim + 1];
                params.yAux.strides[dim] = params.yAux.strides[dim + 1];
            }

            if (weightDim < Mme::c_mme_max_tensor_dims - 1)
            {
                params.w.sizes[weightDim] = params.w.sizes[weightDim + 1];
                params.w.bases[weightDim] = params.w.bases[weightDim + 1];
                params.w.dcoreBases[weightDim] = params.w.dcoreBases[weightDim + 1];
                params.w.strides[weightDim] = params.w.strides[weightDim + 1];
                if (isMaskBgemm)
                {
                    params.wAux.sizes[weightDim] = params.wAux.sizes[weightDim + 1];
                    params.wAux.bases[weightDim] = params.wAux.bases[weightDim + 1];
                    params.wAux.dcoreBases[weightDim] = params.wAux.dcoreBases[weightDim + 1];
                    params.wAux.strides[weightDim] = params.wAux.strides[weightDim + 1];
                }
            }

            if (dim < Mme::c_mme_max_conv_dims - 1)
            {
                params.conv.stride[dim - 1] = params.conv.stride[dim];
                params.conv.dilation[dim - 1] = params.conv.dilation[dim];
                params.conv.padding[dim - 1] = params.conv.padding[dim];
            }
        }
        params.permutation[trivialDim] = Mme::c_mme_max_tensor_dims - 1;

        params.x.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
        params.x.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
        params.x.dcoreBases[Mme::c_mme_max_tensor_dims - 1] = 0;
        params.w.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
        params.w.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
        params.y.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
        params.y.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
        params.y.dcoreBases[Mme::c_mme_max_tensor_dims - 1] = 0;

        if (isMaskBgemm)
        {
            params.xAux.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
            params.xAux.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
            params.xAux.dcoreBases[Mme::c_mme_max_tensor_dims - 1] = 0;
            params.wAux.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
            params.wAux.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
            params.yAux.sizes[Mme::c_mme_max_tensor_dims - 1] = 1;
            params.yAux.bases[Mme::c_mme_max_tensor_dims - 1] = 0;
            params.yAux.dcoreBases[Mme::c_mme_max_tensor_dims - 1] = 0;
        }

        params.conv.stride[Mme::c_mme_max_conv_dims - 2] = 1;
        params.conv.dilation[Mme::c_mme_max_conv_dims - 2] = 1;
        params.conv.padding[Mme::c_mme_max_conv_dims - 2] = 0;
    }
}

/*
 * a trivial dimension is of size 1, the convolution on it is of size 1 and without padding.
 * these dimension dont actually do anything, but leaving such them can hurt mme performance.
 * in this function we will shift upper dimension onto trivial dimension.
 * dimX:[64,1,128,1,64] -> dimX:[64,128,64,1,1]
 */
void MmeBrain::trivialDimsReduction(MmeCommon::MmeLayerParams& params)
{
    if (params.isDmaOperation()) return;
    bool isBgemm = params.isGemmOperation();
    std::bitset<Mme::c_mme_max_tensor_dims> validSpDims = {};
    validSpDims.set(0); /* the dense dimension cannot be trivial */
    if (isBgemm) validSpDims.set(1); /* in bgemm the spatial dim also cant be trivial */
    // for convolution we are searching for trivial spatial dims, for bgemm trivial batch dims
    unsigned firstDim = isBgemm ? MmeCommon::GEMM_DIM_B1 : MmeCommon::DIM_W;
    for (unsigned tensorDim = firstDim; tensorDim < Mme::c_mme_max_tensor_dims; ++tensorDim)
    {
        if (isBgemm)
        {
            if (params.x.sizes[tensorDim] != 1 || params.y.sizes[tensorDim] != 1 || params.w.sizes[tensorDim] != 1 ||
                params.x.bases[tensorDim] != 0 || params.y.bases[tensorDim] != 0 || params.w.bases[tensorDim] != 0)
            {
                validSpDims.set(tensorDim);
            }
        }
        else
        {
            unsigned weightDim = tensorDim + 1;
            unsigned convDim = tensorDim - 1;

            unsigned padding = 0;
            unsigned convSize = 1;
            if (convDim < Mme::c_mme_max_conv_dims - 1)
            {
                padding = params.conv.padding[convDim];
                convSize = params.w.sizes[weightDim];
            }

            if (params.x.sizes[tensorDim] != 1 || params.y.sizes[tensorDim] != 1 || params.x.bases[tensorDim] != 0 ||
                params.y.bases[tensorDim] != 0 || padding != 0 || convSize != 1)
            {
                validSpDims.set(tensorDim);
            }
        }
    }

    unsigned shifted = 0;
    std::optional<unsigned> prevDim;
    for (unsigned curDim = 0; curDim < Mme::c_mme_max_tensor_dims; ++curDim)
    {
        if (!validSpDims.test(curDim)) continue;
        if (!prevDim.has_value())
        {
            prevDim = curDim;
            continue;
        }
        unsigned trivialDim = *prevDim + 1;
        unsigned dist = curDim - trivialDim;
        /* if the distance is greater than 1 we have a trivial dimension between them */
        if (dist > 0)
        {
            /* compensate in case we already performed a shift */
            shiftDimensions(params, trivialDim - shifted, dist);
            shifted += dist;
        }
        prevDim = curDim;
    }
}

unsigned MmeBrain::calcExpectedReadInputCyclesPerOperand(const MmeLayerParams& params,
                                                         MmeCommon::EMmeInternalOperand operand)
{
    MME_ASSERT((operand == e_mme_op_a || operand == e_mme_op_b), "Input operand need to be a or b");
    float unalignedPenalty = (operand == e_mme_op_a) ? m_perfAttr.unaligedPenaltyA : m_perfAttr.unaligedPenaltyB;

    unsigned tensorSize = 1;
    for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        tensorSize *= params.getOperand(operand).sizes[dim];
    }
    unsigned totalReadDataSize = (operand == e_mme_op_a ? m_perfAttr.fetchNrA : m_perfAttr.fetchNrB) * tensorSize;
    unsigned numberOfPortsReadDiffData =
        (operand == e_mme_op_a ? m_geoAttr->getGeometryHeight() : m_geoAttr->getGeometryWidth()) /
        m_geoAttr->getPortSize(operand);
    unsigned dataPerPortInElements = ((float) (totalReadDataSize * unalignedPenalty) / numberOfPortsReadDiffData);
    unsigned totalReadTimeAInCycles = (float) (dataPerPortInElements) / m_geoAttr->getPortSize(operand);
    return totalReadTimeAInCycles;
}

void MmeBrain::calcExpectedReadInputCycles(const MmeLayerParams& params)
{
    unsigned totalReadTimeAInCyclesA = calcExpectedReadInputCyclesPerOperand(params, e_mme_op_a);
    unsigned totalReadTimeAInCyclesB = calcExpectedReadInputCyclesPerOperand(params, e_mme_op_b);

    if (m_geoAttr->isGeometryPortConstrained())
    {
        // In 4xh and 4xw you can't read opA and opB in parallel.
        // You read them serially: First the small operand is being read (opB in 4xh, opA in 4xw),
        // then the large operand.
        m_perfAttr.expectedReadInputCycles = totalReadTimeAInCyclesA + totalReadTimeAInCyclesB;
    }
    else
    {
        m_perfAttr.expectedReadInputCycles = std::max(totalReadTimeAInCyclesA, totalReadTimeAInCyclesB);
    }
}

std::string MmeBrain::getGeometryDebugInfo(const CommonGeoAttr& geoAttr) const
{
    std::stringstream ss;
    ss << "MME Geometry ";
    ss << MmeCommon::MmeParamsDumper::getGeometryName(geoAttr.getGeometry());
    ss << ": ";
    ss << "Width=" << std::to_string(geoAttr.getGeometryWidth());
    ss << ", Height=" << std::to_string(geoAttr.getGeometryHeight());
    ss << ", Concurrency=" << std::to_string(geoAttr.getEffectiveBatchConcurrency());
    if (geoAttr.supportsConcurrency())
        ss << ", Concurrency dim=" << std::to_string(geoAttr.getConcurrentDim());
    ss << ", CDConcurrency=" << std::to_string(geoAttr.getGeometryCdConcurrency());
    ss << ", CDConcurrency dim=" << std::to_string(geoAttr.getSpInterleavingDim(e_mme_op_a));
    ss << ", Flattening=" << m_flattening;
    return ss.str();
}

std::string MmeBrain::getPerfDebugInfo(PerfAttr& perfAttr) const
{
    std::stringstream ss;
    ss << "Perf Attributes: ";
    ss << "Utilization=" << std::to_string(perfAttr.mmeUtilization);
    ss << ", expected cycles=" << std::to_string(perfAttr.expectedRuntimeCycles);
    ss << ", expected runtime=" << std::to_string(perfAttr.expectedRuntime) << " us";
    ss << ", #FetchesA=" << std::to_string(perfAttr.fetchNrA);
    ss << ", #FetchesB=" << std::to_string(perfAttr.fetchNrB);
    return ss.str();
}

MmeLayerParams MmeBrain::getDefaultParams(const ChipType chipType)
{
    return getMmeLayerParams(chipType);
}

unsigned MmeBrain::getGeometryWidth(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->getGeometryWidth();
}

unsigned MmeBrain::getGeometryHeight(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->getGeometryHeight();
}

unsigned MmeBrain::getGeometryConcurrency(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->getGeometryConcurrency();
}

unsigned MmeBrain::getGeometryCdConcurrency(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->getGeometryCdConcurrency();
}

unsigned MmeBrain::getEffectiveBatchConcurrency(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->getEffectiveBatchConcurrency();
}

bool MmeBrain::isAsymPortConfigMode(ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params)->isAsymPortConfigMode();
}

std::unique_ptr<CommonGeoAttr> MmeBrain::getGeoAttr(const ChipType chipType, const MmeLayerParams& params)
{
    return MmeCommon::getGeoAttr(chipType, params);
}

MultiplierArray MmeBrain::getSolutionMultipliers(const MmeLayerParams& params,
                                                 const MultiplierArray& commonGranularity,
                                                 const MultiplierArray& previousMultiplier,
                                                 const bool cdSplit,
                                                 const bool cdPerforation)
{
    MME_ASSERT(params.isGemmOperation() || params.isConvOperation(), "unexpected operation");

    auto accessPattern = AccessPatternFactory::createFrom(&params);
    Brain::SolutionMultipliers::MmeAspects::MultipliersGenerator multiplierGenerator(params,
                                                                                     accessPattern,
                                                                                     commonGranularity,
                                                                                     previousMultiplier);

    auto geoAttr = getGeoAttr(m_chipType, params);

    // height, width and batch are all manifest in the output
    auto outputOperand = params.getExternalOperand(e_mme_op_c);
    multiplierGenerator.inflateUpTo(MmeAspect::OUTPUT_WIDTH, geoAttr->getGeometryWidth(), outputOperand);
    multiplierGenerator.inflateUpTo(MmeAspect::OUTPUT_HEIGHT, geoAttr->getGeometryHeight(), outputOperand);
    multiplierGenerator.inflateUpTo(MmeAspect::GROUPS, geoAttr->getGeometryConcurrency(), outputOperand);
    if (cdSplit)
    {
        // Common dimension is manifested in both inputs, but in case the filter dims are counted towards the common dim
        // size, they only manifest in operandB of a convolution.
        auto inputBOperand = params.getExternalOperand(e_mme_op_b);
        multiplierGenerator.inflateAtLeastTo(MmeAspect::INPUTS_COMMON, getMinCd(cdPerforation), inputBOperand);
    }
    else
    {
        multiplierGenerator.setMaxMultiplier(MmeAspect::INPUTS_COMMON);
    }

    auto solutionMultipliers = multiplierGenerator.getSolution();

    // Validate solution
    for (size_t idx = 0; idx < solutionMultipliers.size(); idx++)
    {
        auto dimMaxMultiplier = div_round_up(accessPattern.indexSpace.at(idx), commonGranularity.at(idx));
        MME_ASSERT(solutionMultipliers.at(idx) > 0 && solutionMultipliers.at(idx) <= dimMaxMultiplier,
                   fmt::format("Bad solution multiplier generated for node dim {}: {} (expected range: 1-{})",
                               idx,
                               solutionMultipliers.at(idx),
                               dimMaxMultiplier));
    }

    return solutionMultipliers;
}

static void solutionSizeForOperand(SizeArray& operandSize, /* IN_OUT */
                                   const MultiplierArray& sizes,
                                   const AccessPattern::TensorAccessPattern::DimsAPVector& dimsAP)
{
    for (int dim = 0; dim < dimsAP.size(); dim++)
    {
        const auto& dimAP = dimsAP.at(dim);
        auto projectedSize = dimAP.size * sizes.at(dimAP.indexSpaceDim);
        operandSize.at(dim) = std::min(uint64_t(operandSize.at(dim)), projectedSize);
    }
}

void MmeBrain::setParamsToSolutionSize(MmeLayerParams& params, const MultiplierArray& sizes)
{
    auto accPtrn = AccessPatternFactory::createFrom(&params);
    solutionSizeForOperand(params.getOperand(e_mme_op_a).sizes,
                           sizes,
                           AccessPatternDetails::Utils::accessPatternForOperandA(accPtrn).dimsAccessPattern);
    solutionSizeForOperand(params.getOperand(e_mme_op_b).sizes,
                           sizes,
                           AccessPatternDetails::Utils::accessPatternForOperandB(accPtrn).dimsAccessPattern);
    solutionSizeForOperand(params.getOperand(e_mme_op_c).sizes,
                           sizes,
                           AccessPatternDetails::Utils::accessPatternForOperandC(accPtrn).dimsAccessPattern);
}

void MmeBrain::setParamsToSolutionSize(MmeLayerParams& params,
                                       const MultiplierArray& solutionMultipliers,
                                       const MultiplierArray& commonGranularity)
{
    MultiplierArray sizes(solutionMultipliers.size());
    for (int dim = 0; dim < sizes.size(); dim++)
    {
        sizes.at(dim) = solutionMultipliers.at(dim) * commonGranularity.at(dim);
    }

    setParamsToSolutionSize(params, sizes);
}

void setStrategy(const MmeLayerParams& params, MmeStrategy& strategy)
{
    // for now simply copy the default strategy, in the future here we will check different optimizations and choose the
    // best one. either that or we will also iterate over several strategies for a given geometry-pattern pair and
    // return all of them.
    strategy = params.strategy;
}

static void setOptimizationToParams(MmeLayerParams& params, const OptimizationSolutions opt)
{
    switch (opt)
    {
        case NO_OPT:
            params.strategy.batchConcurrencyEn = TurnedOff;
            params.strategy.cdConcurrencyEn = TurnedOff;
            break;
        case BATCH_CONCURRENCY:
            params.strategy.batchConcurrencyEn = TurnedOn;
            params.strategy.cdConcurrencyEn = TurnedOff;
            break;
        case CD_CONCURRENCY:
            params.strategy.batchConcurrencyEn = TurnedOff;
            params.strategy.cdConcurrencyEn = TurnedOn;
            break;
        case HYBRID_CONCURRENCY:
            params.strategy.batchConcurrencyEn = TurnedOn;
            params.strategy.cdConcurrencyEn = TurnedOn;
            break;
    }
}

void updateParamsForCdPerforation(MmeLayerParams& params, unsigned numDcores)
{
    switch (params.opType)
    {
        case e_mme_fwd:
        case e_mme_transposed_dedx:
        case e_mme_ab:
        case e_mme_reductionAdd:
            params.x.sizes[0] /= numDcores;
            params.w.sizes[1] /= numDcores;
            // TODO: if x[0] < 4 need to divide w.sizes[2] * w.sizes[3] * w.sizes[4];
            break;
        case e_mme_dedx:
            params.y.sizes[0] /= numDcores;
            params.w.sizes[0] /= numDcores;
            // TODO: if x[0] < 4 need to divide * w.sizes[2] * w.sizes[3] * w.sizes[4];
            break;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            // TODO: support also 5 dims tensors (batch dim = 4)
            params.y.sizes[3] /= numDcores;
            params.x.sizes[3] /= numDcores;
            // CD = B * D * H * W
            break;
        case e_mme_abt:
            params.x.sizes[0] /= numDcores;
            params.w.sizes[0] /= numDcores;
            break;
        case e_mme_atb:
            params.x.sizes[1] /= numDcores;
            params.w.sizes[1] /= numDcores;
            break;
        case e_mme_atbt:
            params.x.sizes[1] /= numDcores;
            params.w.sizes[0] /= numDcores;
            break;
            break;
        default:
            break;
    }
}

MmeBrainSolutionPtr MmeBrain::generateSolution(const MmeLayerParams& params,
                                               const MultiplierArray& commonGranularity,
                                               const MultiplierArray& previousMultiplier,
                                               const bool cdSplit,
                                               const bool cdPerforation,
                                               const OptimizationSolutions opt)
{
    MmeLayerParams slicedParams = params;
    MmeBrainSolutionPtr solution = std::make_shared<MmeBrainSolution>();
    solution->solutionType = opt;
    solution->solutionDimMultipliers =
        getSolutionMultipliers(params, commonGranularity, previousMultiplier, cdSplit, cdPerforation);
    setParamsToSolutionSize(slicedParams, solution->solutionDimMultipliers, commonGranularity);
    setStrategy(params, solution->strategy);

    if (cdPerforation)
    {
        unsigned numDcores = m_mmeHal.getDcoreNr();
        MmeLayerParams cdPerfParams = params;
        cdPerfParams.strategy.mmeLimit = params.strategy.mmeLimit / numDcores;
        updateParamsForCdPerforation(cdPerfParams, numDcores);
        getPerfAttr(cdPerfParams, solution->perfAttr, slicedParams, cdPerforation);

        // set aux tensors memoryAttr
        solution->perfAttr.memoryAttrAux.accessesPerChip = solution->perfAttr.memoryAttrC.accessesPerChip;
        solution->perfAttr.memoryAttrAux.accessesPerDcore = solution->perfAttr.memoryAttrC.accessesPerDcore;
        solution->perfAttr.memoryAttrAux.accessBW = m_mmeHal.getSinglePortBw() * m_geoAttr->getChipPortsNr(e_mme_op_c);
    }
    else
    {
        getPerfAttr(params, solution->perfAttr, slicedParams);
    }
    getSolutionReq(params, cdSplit, solution->requirements, cdPerforation);
    solution->previousSolutionMultipliers = previousMultiplier;
    return solution;
}

bool MmeBrain::dataTypeSupportsReduction(EMmeDataType dt)
{
    return !isTypeFp8(dt);
}

bool validateCdSplitSolution(const MmeLayerParams& params,
                             const MultiplierArray& commonGranularity,
                             const MultiplierArray& previousMultiplier,
                             const MmeBrainSolutionPtr solution)
{
    auto accessPattern = AccessPatternFactory::createFrom(&params);
    Brain::SolutionMultipliers::MmeAspects::MultipliersGenerator multiplierGenerator(params,
                                                                                     accessPattern,
                                                                                     commonGranularity,
                                                                                     previousMultiplier);
    multiplierGenerator.setMaxMultiplier(MmeAspect::INPUTS_COMMON);
    auto maxMultipliers = multiplierGenerator.getSolution();

    bool isSplit = false;
    for (unsigned cdDim : solution->requirements.cdDims)
    {
        isSplit |= solution->solutionDimMultipliers.at(cdDim) != maxMultipliers.at(cdDim);
    }

    return isSplit;
}

std::vector<OptimizationSolutions> MmeBrain::getRelevantOptimizations(const MmeLayerParams& params,
                                                                      const MultiplierArray& commonGranularity,
                                                                      const MultiplierArray& previousMultiplier)
{
    std::vector<OptimizationSolutions> solutions = {NO_OPT};
    if (m_knobs.operationModes.addOptimizationToLBSolutions == false) return solutions;
    if (params.isDedwOperation())
    {
        // here we should probably add some logic to verify CD concurrency is a valid option as sometimes we cannot
        // consider it. also there is a chance that we will get a lot of solutions now due to this, so we will need to
        // reduce solutions to avoid exploding the compilation time.
        solutions.push_back(BATCH_CONCURRENCY);
        solutions.push_back(CD_CONCURRENCY);
        solutions.push_back(HYBRID_CONCURRENCY);
    }
    // TODO: [SW-160570]
    // for bgemm we also want to expose the concurrency solutions as a separate solution.
    // currently geoAttr decides itself whether to use concurrency or not.
    // because of that we added extra geometries that in turn will turn in "concurrency geometries"
    // to use the logic below we need to remove the extra geometries from getGeometries(), change geoAttr to always
    // use concurrency when its turned on, and then we can uncomment the code below. either way for now in many cases
    // (but not all) the brain will offer solutions with and without concurrency in bgemm
    //    else if (params.isGemmOperation())
    //    {
    //        auto geoAttr = getGeoAttr(m_chipType, params);
    //        if (params.getFcdSize() <= geoAttr->getMmeWidth() && params.getSpatialSize() <= geoAttr->getMmeHeight())
    //        {
    //            // output is smaller than a single MME, no need for non concurrency solutions
    //            solutions.pop_back();
    //        }
    //        solutions.push_back(BATCH_CONCURRENCY);
    //    }

    return solutions;
}

class SolutionsContainer
{
public:
    SolutionsContainer(ChipType chip) : m_chip(chip) {}

    void addSolution(const MmeBrainSolutionPtr& newSolution, const MmeLayerParams& params)
    {
        auto geoAttr = getGeoAttr(m_chip, params);
        if (duplicated(newSolution, geoAttr))
        {
            LOG_WARN(MME_BRAIN, "Duplicated solution found. Filtering it out.");
            return;
        }
        addUniqueSolution(newSolution, std::move(geoAttr));
    }

    const MmeBrainSolutionContainer& getSolutions() const { return m_solutions; }

private:
    const ChipType m_chip;
    MmeBrainSolutionContainer m_solutions;
    std::unordered_map<MmeBrainSolutionPtr, upCommonGeoAttr> m_solutionGeoAttrs;

    bool duplicated(const MmeBrainSolutionPtr& newStrategy, const upCommonGeoAttr& newGeoAttr)
    {
        return std::any_of(m_solutions.begin(), m_solutions.end(), [&](const MmeBrainSolutionPtr& existingStrategy) {
            return compare(existingStrategy, m_solutionGeoAttrs.at(existingStrategy), newStrategy, newGeoAttr);
        });
    }

    static bool compare(const MmeBrainSolutionPtr& s1,
                        const upCommonGeoAttr& g1,
                        const MmeBrainSolutionPtr& s2,
                        const upCommonGeoAttr& g2)
    {
        return compareGeoAttr(g1, g2) && compareSolutions(s1, s2);
    }

    static bool compareGeoAttr(const upCommonGeoAttr& g1, const upCommonGeoAttr& g2)
    {
        return g1->getGeometryWidth() == g2->getGeometryWidth() && g1->getGeometryHeight() == g2->getGeometryHeight() &&
               g1->getGeometryConcurrency() == g2->getGeometryConcurrency();
    }

    static bool compareSolutions(const MmeBrainSolutionPtr& s1, const MmeBrainSolutionPtr& s2)
    {
        return s1->solutionDimMultipliers == s2->solutionDimMultipliers &&
               s1->strategy.pattern == s2->strategy.pattern &&
               s1->requirements.perforationDimVec == s2->requirements.perforationDimVec;
    }

    void addUniqueSolution(const MmeBrainSolutionPtr& newSolution, upCommonGeoAttr&& newGeoAttr)
    {
        m_solutions.push_back(newSolution);
        m_solutionGeoAttrs[newSolution] = std::move(newGeoAttr);
    }
};

bool MmeBrain::skipCdConcurrencySolution(const MmeBrainSolutionPtr& solution,
                                         const MmeLayerParams& params,
                                         const OptimizationSolutions& opt,
                                         const bool cdPerforationEn) const
{
    // cd concurrency optimization with reductionLevel = 4
    if (opt == CD_CONCURRENCY && solution->strategy.cdConcurrencyEn == TurnedOn &&
        solution->strategy.reductionLevel == 4)
    {
        // cd perforation solution is supported
        if (cdPerforationEn && params.getCDSize() >= getMinCd(true))
        {
            return true;  // cd perforation and cd concurrency with reductionLevel 4
                          // are identical, except cd concurrency solution adds memset
                          // therefore, eliminate cd concurrency solution and prefer cd perforation solution
        }
    }
    return false;
}

MmeBrainSolutionContainer MmeBrain::getMmeSolutions(const MmeLayerParams& params,
                                                    const MultiplierArray& commonGranularity,
                                                    const MultiplierArray& previousMultiplier,
                                                    const bool cdPerforationEn)
{
    LOG_TRACE(MME_BRAIN, "{}: Start", __func__);

    MmeLayerParams tempParams = params;
    SolutionsContainer uniqueSolutions(m_chipType);

    std::vector<EMmeGeometry> geometries;
    if (params.strategy.geometry == e_mme_geometry_nr)
    {
        geometries = getGeometries(tempParams);
    }
    else
    {
        geometries.push_back(params.strategy.geometry);
    }

    for (EMmeGeometry geometry : geometries)
    {
        tempParams.strategy.geometry = geometry;
        std::vector<EMmePattern> walkingPatterns;
        if (params.strategy.pattern == e_mme_patterns_nr)
        {
            walkingPatterns = getPatterns(tempParams);
        }
        else
        {
            walkingPatterns.push_back(params.strategy.pattern);
        }

        for (EMmePattern pattern : walkingPatterns)
        {
            tempParams.strategy.pattern = pattern;
            auto optimization = getRelevantOptimizations(params, commonGranularity, previousMultiplier);

            for (auto opt : optimization)
            {
                LOG_INFO(MME_BRAIN, "Generating a solution with for geometry: {}, pattern: {}, and opt: {}", geometry, pattern, opt);
                setOptimizationToParams(tempParams, opt);
                auto solution = generateSolution(tempParams, commonGranularity, previousMultiplier, false, false, opt);
                if (!skipCdConcurrencySolution(solution, tempParams, opt, cdPerforationEn))
                {
                    uniqueSolutions.addSolution(solution, tempParams);
                }

                // slicing over the CD is only supported above a minimum CD value.
                // we are using here the effective CD size and not the size of any single CD dim.
                // this is because the limitation is on the number of cycles each MME will spend before writing an
                // output. and not the actual size of a CD dimension.
                if (params.getCDSize() > m_knobs.minCd &&
                    dataTypeSupportsReduction(params.getOperand(e_mme_op_c).elementType))
                {
                    LOG_INFO(MME_BRAIN, "Generating a solution with CD-Split for geometry: {}", geometry);
                    auto cdSplitSolution =
                        generateSolution(tempParams, commonGranularity, previousMultiplier, true, false, opt);
                    if (validateCdSplitSolution(tempParams, commonGranularity, previousMultiplier, cdSplitSolution))
                    {
                        if (!skipCdConcurrencySolution(solution, tempParams, opt, cdPerforationEn))
                        {
                            uniqueSolutions.addSolution(cdSplitSolution, tempParams);
                        }
                    }
                }
                else
                {
                    LOG_INFO(MME_BRAIN,
                             "Skip CD-split solution generation for geometry: {}. CD size: {}, min CD: {}",
                             geometry,
                             tempParams.getCDSize(),
                             m_knobs.minCd);
                }
            }

            if (cdPerforationEn && tempParams.getCDSize() >= getMinCd(true))
            {
                setOptimizationToParams(tempParams, NO_OPT);
                // Generate cd split + cd perforation solution
                if (dataTypeSupportsReduction(params.getOperand(e_mme_op_c).elementType))
                {
                    auto cdSplitAndPerfSoltuion =
                        generateSolution(tempParams, commonGranularity, previousMultiplier, true, true, NO_OPT);
                    uniqueSolutions.addSolution(cdSplitAndPerfSoltuion, tempParams);
                }
                // Generate cd perforation solution
                auto cdPerfSoltuion =
                    generateSolution(tempParams, commonGranularity, previousMultiplier, false, true, NO_OPT);
                uniqueSolutions.addSolution(cdPerfSoltuion, tempParams);
            }
        }
    }
    LOG_TRACE(MME_BRAIN, "{}: End", __func__);
    return uniqueSolutions.getSolutions();
}

MmeBrainSolutionPtr MmeBrain::inflateForUtilization(const MmeLayerParams& params,
                                                    const MmeBrainSolutionPtr curSolution,
                                                    const MultiplierArray& commonGranularity,
                                                    PhysicalAspects::Name aspectToInflate,
                                                    const std::optional<float>& utilizationThreshold)
{
    auto accessPattern = AccessPatternFactory::createFrom(&params);
    Brain::SolutionMultipliers::MmeAspects::MultipliersGenerator multiplierGenerator(
        params,
        accessPattern,
        commonGranularity,
        curSolution->previousSolutionMultipliers);
    multiplierGenerator.setMultipliers(curSolution->solutionDimMultipliers);

    auto geoAttr = getGeoAttr(m_chipType, params);
    uint64_t geometrySize;
    switch (aspectToInflate)
    {
        default:
            MME_ASSERT(0, "unexpected aspect");
        case MmeAspect::OUTPUT_WIDTH:
            geometrySize = geoAttr->getGeometryWidth();
            break;
        case MmeAspect::OUTPUT_HEIGHT:
            geometrySize = geoAttr->getGeometryHeight();
            break;
    }
    auto outputOperand = params.getExternalOperand(e_mme_op_c);
    uint64_t curSize = multiplierGenerator.aspectCurrentSize(aspectToInflate, outputOperand);
    int curGeoMultiple = div_round_up(curSize, geometrySize);
    MmeBrainSolutionPtr solution = std::make_shared<MmeBrainSolution>(*curSolution);
    float requiredUtilization = utilizationThreshold.value_or(curSolution->perfAttr.mmeUtilization);
    uint64_t inflatedSize = ++curGeoMultiple * geometrySize;
    while (inflatedSize <= multiplierGenerator.aspectFullSize(aspectToInflate, outputOperand))
    {
        // need to reinitialize to original sizes due to multiplier generation initialization issue
        MmeLayerParams slicedParams = params;
        multiplierGenerator.inflateUpTo(aspectToInflate, inflatedSize, outputOperand);
        solution->solutionDimMultipliers = multiplierGenerator.getSolution();
        setParamsToSolutionSize(slicedParams, solution->solutionDimMultipliers, commonGranularity);
        getPerfAttr(params, solution->perfAttr, slicedParams);

        if (solution->perfAttr.mmeUtilization > requiredUtilization)
        {
            LOG_TRACE(MME_BRAIN,
                      "utilization increased compared to required utilization {}, returning new utilization {}",
                      requiredUtilization,
                      solution->perfAttr.mmeUtilization);
            return solution;
        }
        inflatedSize = ++curGeoMultiple * geometrySize;
    }

    LOG_TRACE(MME_BRAIN, "couldnt increase utilization, solution is not inflated");
    return nullptr;
}

std::string MmeBrainSolution::print(const std::string& nodeName) const
{
    std::stringstream ss;
    ss << "Solution: nodeName: " << nodeName << std::endl;
    ss << strategy.print();
    ss << perfAttr.print();
    ss << requirements.print();
    if (!relaxedTile.empty())
    {
        ss << "RelaxedTile: [" << arrayToStr(relaxedTile.begin(), relaxedTile.end()) << "]" << std::endl;
    }
    ss << "SolutionDimMultipliers: [" << arrayToStr(solutionDimMultipliers.begin(), solutionDimMultipliers.end()) << "]" << std::endl;
    return ss.str();
}

std::string PerfAttr::print() const
{
    std::stringstream ss;
    ss << "PerfAttr:" << std::endl;
    ss << "maxUtil: " << maxUtilization << " mmeUtil: " << mmeUtilization << " numOfActivations: " << numOfActivations << std::endl;
    ss << "expectedCycles: " << expectedRuntimeCycles << " expectedRuntime: " << expectedRuntime << std::endl;
    ss << "opA Memory: accessBW : " << memoryAttrA.accessBW << " accessPerDcore: " << memoryAttrA.accessesPerDcore << " accessPerChip " << memoryAttrA.accessesPerChip << std::endl;
    ss << "opB Memory: accessBW : " << memoryAttrB.accessBW << " accessPerDcore: " << memoryAttrB.accessesPerDcore << " accessPerChip " << memoryAttrB.accessesPerChip << std::endl;
    ss << "opC Memory: accessBW : " << memoryAttrC.accessBW << " accessPerDcore: " << memoryAttrC.accessesPerDcore << " accessPerChip " << memoryAttrC.accessesPerChip << std::endl;
    ss << "Aux Memory: accessBW : " << memoryAttrAux.accessBW << " accessPerDcore: " << memoryAttrAux.accessesPerDcore << " accessPerChip " << memoryAttrAux.accessesPerChip << std::endl;
    ss << "UnalignmentPenaltyA : " << unaligedPenaltyA << " unalignmentPenaltyB: " << unaligedPenaltyB << std::endl;
    ss << "fetchNrA: " << fetchNrA << " fetchNrB: " << fetchNrB << std::endl;
    return ss.str();
}


}  // namespace MmeCommon
