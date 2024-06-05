#include "generate_mme_descriptors.h"

#include "cache_types.h"
#include "gaudi3_graph.h"
#include "mme/mme_logger.h"
#include "node_utils.h"
#include "../descriptor_generator.h"
#include "mme_brain_ifc.h"
#include "mme_desc_gen_utils.h"
#include "synapse_common_types.h"
#include "types.h"
#include "mme/mme_strategy_serializer.h"

#include "eager/eager_interface.h"
#include "eager/lib/eager_brain_base.h"
#include "eager/lib/utils/general_defs.h"
#include "mme_reference/data_types/fp8.h"
#include "include/mme_common/mme_common_enum.h"
#include <algorithm>

using namespace MmeCommon;

namespace gaudi3
{
bool getMmeSramReduction(const MmeNode& node)
{
    bool          isAtomicAdd      = false;
    pTensor       outTensor        = node.getOutput(TENSOR_OFM);
    ReductionInfo reductionInfoMme = outTensor->getRealReductionInfo();

    /// TODO - Add support for SUB, MIN, MAX reduction atomic operations
    if (outTensor->isReductionEnabled())
    {
        switch (reductionInfoMme.reductionOperation)
        {
            case REDUCTION_ADD:
                isAtomicAdd = true;
                break;
            default:
                HB_ASSERT(false, "Unsupported Gaudi3 MME reduction operation {}", reductionInfoMme.reductionOperation);
                break;
        }
    }

    return isAtomicAdd;
}

static EMmeCacheDirective castCacheDirecToMME(CacheDirective cacheDirective)
{
    switch (cacheDirective)
    {
        case CacheDirective::NoAllocate:
            return EMmeCacheDirective::NoAllocate;

        case CacheDirective::HomeAllocate:
            return EMmeCacheDirective::HomeAllocate;

        case CacheDirective::DcoreAllocate:
            return EMmeCacheDirective::DcoreAllocate;

        case CacheDirective::SharedAllocate:
            return EMmeCacheDirective::SharedAllocate;

        case CacheDirective::NotSupported:
            return EMmeCacheDirective::NR;

        case CacheDirective::SkipCache:
            return EMmeCacheDirective::SkipCache;

        default:
            return EMmeCacheDirective::NoAllocate;
    }
}

static EMmeCacheClass castCacheClassToMME(CacheClass cacheclass)
{
    switch (cacheclass)
    {
        case CacheClass::Low:
            return EMmeCacheClass::Low;

        case CacheClass::Normal:
            return EMmeCacheClass::Normal;

        case CacheClass::High:
            return EMmeCacheClass::High;

        case CacheClass::Top:
            return EMmeCacheClass::Reserved;

        default:
            return EMmeCacheClass::Normal;
    }
}

static void setActivationsOperandRoles(ActivationVec& activations,
                                       int            start,
                                       int            end,
                                       TensorRoles    aRole,
                                       TensorRoles    bRole,
                                       TensorRoles    cRole)
{
    HB_ASSERT(start >= 0 && end <= activations.size(), "Indices are out of activations list range");
    for (int i = start; i < end; i++)
    {
        activations[i].operandRoles = {{aRole, bRole, cRole}};
    }
}

static void setActivationsParamsIdx(ActivationVec& activations, int start, int end, unsigned int paramsIdx)
{
    HB_ASSERT(start >= 0 && end <= activations.size(), "Indices are out of activations list range");
    for (int i = start; i < end; i++)
    {
        activations[i].paramsIdx = paramsIdx;
    }
}

static void multiplyParamsOutputSizeByFactor(MmeLayerParams& params, unsigned int dim, unsigned int factor)
{
    MmeTensorView& out = params.getOperand(e_mme_op_c);
    HB_ASSERT(dim <= 4, "Not supporting more than 4 dims yet");  // TODO [SW-169897]: support more than 4 dims tensors
    out.sizes[dim] *= factor;

    for (int i = dim + 1; i < out.strides.size(); i++)
    {
        out.strides[i] = out.strides[dim] * factor;
    }
}

void MmeDescriptorBuilder::setCacheDirectives(const MMENodePtr& mmeNode, MmeLayerParams& params, bool useDefault)
{
    CacheMetaDataArray cacheMetaDataArray;
    CacheMetaData aCacheMetaData;
    CacheMetaData bCacheMetaData;
    CacheMetaData cCacheMetaData;

    TensorRoles tensorRoleA = MmeCommon::INPUT_TENSOR_A;
    TensorRoles tensorRoleB = MmeCommon::INPUT_TENSOR_B;
    TensorRoles tensorRoleC = MmeCommon::OUTPUT_TENSOR_C;

    if (!useDefault)
    {
        DescriptorGenerator::getTensorCacheMetaData(mmeNode, cacheMetaDataArray);
    }

    if (mmeNode->isCdPerforated())
    {
        if (params.opType == MmeCommon::e_mme_reductionAdd)
        {
            tensorRoleA = AUX_TENSOR_REDUCTION;
            tensorRoleB = AUX_TENSOR_SCRATCHPAD;
        }
        else
        {
            tensorRoleC = AUX_TENSOR_SCRATCHPAD;
        }
    }

    aCacheMetaData = cacheMetaDataArray[tensorRoleA];
    bCacheMetaData = cacheMetaDataArray[tensorRoleB];
    cCacheMetaData = cacheMetaDataArray[tensorRoleC];

    if (mmeNode->isCdPerforated() && params.opType == MmeCommon::e_mme_reductionAdd)
    {
        // set scratchpad to NoAlloc when is input of reductionAdd activation
        bCacheMetaData.cacheDirective = CacheDirective::NoAllocate;
    }

    const auto& aCacheDirective     = castCacheDirecToMME(aCacheMetaData.cacheDirective);
    const auto& bCacheDirective     = castCacheDirecToMME(bCacheMetaData.cacheDirective);
    const auto& cCacheDirective     = castCacheDirecToMME(cCacheMetaData.cacheDirective);
    const auto& aCacheClass         = castCacheClassToMME(aCacheMetaData.cacheClass);
    const auto& bCacheClass         = castCacheClassToMME(bCacheMetaData.cacheClass);
    const auto& cCacheClass         = castCacheClassToMME(cCacheMetaData.cacheClass);
    params.memoryCfg.cacheDirective = {aCacheDirective, bCacheDirective, cCacheDirective};
    params.memoryCfg.clss           = {aCacheClass, bCacheClass, cCacheClass};

    LOG_DEBUG(CACHE_MAINT, "MME node {} MCID CM configuration: \n"
                           "OpA: cacheDirective = {}, class = {}, cmAction {}. \n"
                           "OpB: cacheDirective = {}, class = {}, cmAction {}. \n"
                           "OpCout: cacheDirective = {}, class = {}, cmAction {}.\n",
                           mmeNode->getNodeName(),
                           aCacheDirective, aCacheClass, aCacheMetaData.cmAction,
                           bCacheDirective, bCacheClass, bCacheMetaData.cmAction,
                           cCacheDirective, cCacheClass, cCacheMetaData.cmAction);
}

bool generateMmeDescriptors(Gaudi3Graph& g)
{
    const NodeVector& nodes = g.getExeSortedNodes();
    MmeDescriptorBuilder builder(g);
    for (const NodePtr& node : nodes)
    {
        if (HabanaGraph::runsOnMME(node))
        {
            // params initialized here to support eager flow
            MmeCommon::ChipType chipType = MmeBrainIfc::getMmeChipType(g.getTraits().getHalReader()->getDeviceType());
            MmeLayerParams      params   = MmeBrain::getDefaultParams(chipType);
            MmeDescriptorGeneratorPtr descGenerator = builder.createParamsAndActivations(node, params);
            // Save descriptor generator in graph.
            g.setMmeNodeDescriptorGenerator(node, descGenerator);
        }
    }
    return true;
}

void MmeDescriptorBuilder::printDebugInfoAndSetAnnotations(MmeDescriptorGeneratorPtr& descGenerator,
                                                           const MMENodePtr&          mmeNode,
                                                           const MmeLayerParams&      params,
                                                           std::optional<bool>        cdPerforation)
{
    MmeLogger           mmeLogger;
    MmeCommon::PerfAttr perfAttr;
    descGenerator->getMMEBrain().getPerfAttr(params, perfAttr, std::nullopt, cdPerforation);
    mmeNode->getNodeAnnotation().mmeMetaData.mmePerfAttr = std::make_shared<MmeCommon::PerfAttr>(perfAttr);
    mmeLogger.printDebugInfo(perfAttr, &*descGenerator);

    // Fill mmeStrategy info for trace analyzer
    mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategyDebugString =
        mmeLogger.getMmeStrategyInfo(params, *descGenerator, e_mme_Gaudi3);
}

//Dcore offset is given from the base of the big tensor, but mme-stack needs the offset from the small tensor (slice)
// to generate correct descriptors.
void MmeDescriptorBuilder::getDcoreOffsetBasedOnSmallTensor(DcoreRoisVec& dcoreROIs) const
{
    // save a copy of the first dcore offset.
    std::array<TOffset,HABANA_DIM_MAX> firstDcoreOffset;
    std::copy(std::begin(dcoreROIs.front().baseOffset), std::end(dcoreROIs.front().baseOffset), firstDcoreOffset.begin());
    for (auto& dcore : dcoreROIs)
    {
        for (unsigned idx = 0; idx < HABANA_DIM_MAX; idx++)
        {
            dcore.baseOffset[idx] -= firstDcoreOffset[idx];
        }
    }
}

bool MmeDescriptorBuilder::isCDPerforated(const MMENodePtr& mmeNode, const MmeLayerParams& params)
{
    bool ret = false;

    // TODO: remove once litePerforation flow supports CD Parallel
    if (!GCFG_ENABLE_CD_PARALLEL.value())
    {
        return false;
    }

    // TODO [SW-144531]: integrate cd concurrency with cd perforation
    if (params.isDedwCdConcurrency())
    {
        return false;
    }
    if (mmeNode->getNodeAnnotation().perforationDim.has_value())
    {
        ret = mmeNode->getMmeBrainIfc()->isCdDim(mmeNode->getNodeAnnotation().perforationDim.value(), params);
    }
    else if (mmeNode->getNodeAnnotation().perforation.has_value())
    {
        ret = mmeNode->getMmeBrainIfc()->isCdDim(mmeNode->getNodeAnnotation().perforation->indexSpaceDim, params);
    }

    return ret;
}

MmeDescriptorGeneratorPtr
MmeDescriptorBuilder::createParamsAndActivations(const NodePtr& node, MmeLayerParams& params)
{
    MMENodePtr mmeNode = std::static_pointer_cast<MmeNode>(node);
    HB_ASSERT(mmeNode != nullptr, "node runs on mme but can't be cast to mme node");
    std::optional<MmeLayerParams> paramsForReduction;
    unsigned                      paramsIdx = 0;
    if (!m_isEagerMode)
    {
        paramsForReduction = std::make_optional<MmeLayerParams>(params);
        // For Eager flow we call generateLayerParamsFromNode from outside.
        generateLayerParamsFromNode(mmeNode, params);
    }

    const auto& halReader = m_graph.getTraits().getHalReader();
    unsigned numOfMmeMasters = halReader->getNumMmeEngines();
    MmeDescriptorGeneratorPtr descGenerator =
        MmeDescriptorGenerator::createMmeDescGenerator(params.isNativeDmaOperation(), numOfMmeMasters);
    HB_ASSERT(descGenerator != nullptr, "Invalid MME descriptor generator");
    descGenerator->getMMEBrain().setOperationModes(mmeNode->getMmeBrainIfc()->getOperationModes());
    descGenerator->addParams(params);  // add original params to params vector

    if (m_isEagerMode)
    {
#ifndef NDEBUG
        MmeLogger          mmeLogger;
        const std::string& nodeName = node->getNodeName();
        SET_TEMP_LOG_CONTEXT(nodeName);
        params.nodeName = nodeName;
        mmeLogger.printMmeParams(params);
#endif
        descGenerator->setParams(params);
        descGenerator->mmeGenerateActivations();
#ifndef NDEBUG
        printDebugInfoAndSetAnnotations(descGenerator, mmeNode, params);
#endif
        return descGenerator;
    }
    else
    {
        MmeLogger          mmeLogger;
        const std::string& nodeName = mmeNode->getNodeName();
        SET_TEMP_LOG_CONTEXT(nodeName);

        auto roiList = m_graph.GetNodeROIs(mmeNode);
        HB_ASSERT(roiList->size() == 1, "dcore split is only supported for a single logical ROI");
        NodeROI& nodeRoi = roiList->front();
        if (mmeNode->getNodeAnnotation().isPerforated())
        {
            LOG_TRACE(MME_STACK, "Splitting to dcores, original node params - ");
            mmeLogger.printMmeParams(params);
            descGenerator->setPerforated(true);
            getDcoreOffsetBasedOnSmallTensor(nodeRoi.dcoreROIs);

            bool cdPerforated = isCDPerforated(mmeNode, params);
            mmeNode->setCdPerforated(cdPerforated);

            MmeLayerParams computeParams = params;
            // dcore split - generate descriptors per dcore.
            for (unsigned dcoreIdx = 0; dcoreIdx < nodeRoi.dcoreROIs.size(); dcoreIdx++)
            {
                // dcores with empty jobs should have activations created by mmeGenerateNullDescs
                if (doesArrayContainZeros(nodeRoi.dcoreROIs[dcoreIdx].size))
                {
                    LOG_TRACE(MME_STACK,
                              "Dcore Roi {} gets zero work size, sizes are: [{}]",
                              dcoreIdx,
                              fmt::join(nodeRoi.dcoreROIs[dcoreIdx].size, ","));
                    descGenerator->setZeroActivationsForDcore();
                    continue;
                }
                DcoreROI&       dcoreRoi    = nodeRoi.dcoreROIs[dcoreIdx];
                MmeLayerParams  dcoreParams = params;
                generateLayerParamsFromRoi(mmeNode, dcoreParams, dcoreRoi, dcoreIdx);
                dcoreParams.strategy.mmeLimit = numOfMmeMasters / halReader->getNumDcores();
                descGenerator->setParams(dcoreParams);
                mmeLogger.printMmeParams(dcoreParams);
                descGenerator->mmeGenerateActivations();
                if (dcoreIdx == 0)
                {
                    if (cdPerforated)
                    {
                        computeParams = dcoreParams;
                        // When CD Perforated, params output is auxScratchpad tensor (which holds all dcores partial
                        // results). We fix params output size, to scratchpad full size by adding a new outer dim of
                        // size: numDcores
                        multiplyParamsOutputSizeByFactor(computeParams,
                                                         mmeNode->getOutput(0)->getDim(),
                                                         nodeRoi.dcoreROIs.size());
                    }
                    // Keep computeParams in descGenerator params vector
                    paramsIdx = descGenerator->addParams(computeParams);
                }
                printDebugInfoAndSetAnnotations(descGenerator, mmeNode, dcoreParams, cdPerforated);
            }
            // Pad real activations with null activations (empty jobs) so each DCORE
            // will have the same amount of activations and number of signals.
            descGenerator->mmeGenerateNullDescs();

            descGenerator->reorderDcoreActivations(params);
            int end = descGenerator->getMmeActivations().size();
            setActivationsParamsIdx(descGenerator->getMmeActivations(), 0, end, paramsIdx);
            setActivationsOperandRoles(descGenerator->getMmeActivations(),
                                       0,
                                       end,
                                       INPUT_TENSOR_A,
                                       INPUT_TENSOR_B,
                                       cdPerforated ? AUX_TENSOR_SCRATCHPAD : OUTPUT_TENSOR_C);
            if (cdPerforated)
            {
                LOG_DEBUG(GC, "node {} is perforated on CD", node->getNodeName());

                generateLayerParamsForReduction(mmeNode, *paramsForReduction, nodeRoi.dcoreROIs.size());
                descGenerator->setParams(*paramsForReduction);
                mmeLogger.printMmeParams(*paramsForReduction);

                descGenerator->mmeGenerateActivations();
                setActivationsOperandRoles(descGenerator->getMmeActivations(),
                                           end,
                                           descGenerator->getMmeActivations().size(),
                                           AUX_TENSOR_REDUCTION,
                                           AUX_TENSOR_SCRATCHPAD,
                                           OUTPUT_TENSOR_C);
                paramsIdx = descGenerator->addParams(*paramsForReduction);
                setActivationsParamsIdx(descGenerator->getMmeActivations(),
                                        end,
                                        descGenerator->getMmeActivations().size(),
                                        paramsIdx);
                printDebugInfoAndSetAnnotations(descGenerator, mmeNode, *paramsForReduction);
            }
        }
        else
        {
            descGenerator->setParams(params);
            paramsIdx = descGenerator->addParams(params);
            mmeLogger.printMmeParams(params);
            descGenerator->mmeGenerateActivations();
            setActivationsParamsIdx(descGenerator->getMmeActivations(),
                                    0,
                                    descGenerator->getMmeActivations().size(),
                                    paramsIdx);
            setActivationsOperandRoles(descGenerator->getMmeActivations(),
                                       0,
                                       descGenerator->getMmeActivations().size(),
                                       INPUT_TENSOR_A,
                                       INPUT_TENSOR_B,
                                       OUTPUT_TENSOR_C);
        }
        descGenerator->setParams(params);
        printDebugInfoAndSetAnnotations(descGenerator, mmeNode, params);
        return descGenerator;
    }
}

struct ParamsAndPerf
{
    bool                    walkingDown;
    MmeCommon::EMmeGeometry geometry;
    MmeCommon::EMmePattern  pattern;
    MmeCommon::PerfAttr     perf;
};
using ParamsAndPerfsVector = std::vector<ParamsAndPerf>;

struct PerfComp
{
    TensorPtr opA;
    TensorPtr opB;

    static constexpr double UTILIZATION_THRESHOLD = 0.01;

    // returns true if lhs "is less than" rhs.
    bool operator()(const ParamsAndPerf& lhs, const ParamsAndPerf& rhs)
    {
        if (std::abs(lhs.perf.mmeUtilization - rhs.perf.mmeUtilization) > UTILIZATION_THRESHOLD)
        {
            // MME utilization difference is bigger than the threshold. lhs
            return lhs.perf.mmeUtilization < rhs.perf.mmeUtilization;
        }
        else if (opA->inSram() != opB->inSram())  // one operand is in HBM and the other in SRAM
        {
            // If the one of the strategies reuse the operand that is in SRAM and the other reuse the one in HBM,
            // the SRAM reuser "is less than" the other
            if (lhs.walkingDown && !rhs.walkingDown)
            {
                // lhs is reusing opB
                // rhs is reusing opA
                // if opB is in SRAM, then lhs "is less than" rhs, since lhs is the SRAM reuser
                return opB->inSram();
            }
            else if (!lhs.walkingDown && rhs.walkingDown)
            {
                // lhs is reusing opA
                // rhs is reusing opB
                // if opA is in SRAM, then lhs "is less than" rhs, since lhs is the SRAM reuser
                return opA->inSram();
            }
        }
        // else - utilization is similar and both inputs are in the same memory
        return lhs.perf.fetchNrA * opA->getDenseSizeInBytes() + lhs.perf.fetchNrB * opB->getDenseSizeInBytes() >
               rhs.perf.fetchNrA * opA->getDenseSizeInBytes() + rhs.perf.fetchNrB * opB->getDenseSizeInBytes();
    }
};

ParamsAndPerfsVector generatePerfAttrCandidates(MmeLayerParams layerParams, MmeCommon::MmeBrain& brain)
{
    ParamsAndPerfsVector candidates;
    for (bool walkDown : {false, true})
    {
        layerParams.setPattern(walkDown);
        for (auto geometry : brain.getGeometries(layerParams))
        {
            layerParams.strategy.geometry = geometry;
            MmeCommon::PerfAttr perf {};
            brain.getPerfAttr(layerParams, perf);
            LOG_TRACE(MME_STACK,
                      "Params: walking down: {}, geometry: {} => Perf: utilization: {}, #fetchA: {}, #fetchB: {}",
                      walkDown,
                      geometry,
                      perf.mmeUtilization,
                      perf.fetchNrA,
                      perf.fetchNrB);
            candidates.push_back(ParamsAndPerf {walkDown, geometry, layerParams.strategy.pattern, perf});
        }
    }
    return candidates;
}

ParamsAndPerfsVector::const_iterator
chooseBestCandidate(ParamsAndPerfsVector& candidates, TensorPtr xTensor, TensorPtr wTensor)
{
    return std::max_element(candidates.begin(), candidates.end(), PerfComp {xTensor, wTensor});
}

void MmeDescriptorBuilder::chooseStrategy(const MMENodePtr& node,
                                          const TensorPtr&  xTensor,
                                          const TensorPtr&  wTensor,
                                          MmeLayerParams&   layerParams)
{
    std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    mmeNode->getMmeBrainIfc()->getRecommendedStrategy(layerParams, /*ignoreTensorAliasing*/ false, /*isGeoPreferredShort*/ true);

    if (layerParams.opType == MmeCommon::e_mme_dedw || layerParams.opType == MmeCommon::e_mme_gemm_transpose ||
        layerParams.isDmaOperation())
    {
        return;  // TODO - support smart strategy selection for DeDw
    }

    MmeCommon::MmeBrain  brain(MmeCommon::e_mme_Gaudi3, mmeNode->getMmeBrainIfc()->getOperationModes());
    ParamsAndPerfsVector candidates = generatePerfAttrCandidates(layerParams, brain);
    auto                 bestParams = chooseBestCandidate(candidates, xTensor, wTensor);

    layerParams.strategy.geometry = bestParams->geometry;
    layerParams.strategy.pattern  = bestParams->pattern;
}

void MmeDescriptorBuilder::chooseSignalingMode(MmeLayerParams& params)
{
    // TODO: replace with e_mme_signaling_desc_with_store once simulating linear ranges in the MME is supported
    //
    //       Notice that e_mme_signaling_desc_with_store has a restriction for now [SW-11381],
    //       so before the assignment, the following condition is needed:
    //       if (mmeNode->getNodeAnnotation().splitToLogicalROIs && params.opType != MmeCommon::e_mme_dedx)

    params.controls.signalingMode  = e_mme_signaling_once;
    params.controls.squashIORois   = true;  // needed in signaling once.
    params.controls.slaveSignaling = Gaudi3Graph::isMmeSlaveSignalingEnabled();
}

MmeCommon::RoundingMode synapseToMmeRoundingMode(synRoundingMode synapseValue)
{
    switch (synapseValue)
    {
        case synRoundingMode::synRoundToNearest:
            return MmeCommon::RoundingMode::RoundToNearest;
        case synRoundingMode::synRoundToZero:
            return MmeCommon::RoundingMode::RoundToZero;
        case synRoundingMode::synRoundUp:
            return MmeCommon::RoundingMode::RoundUp;
        case synRoundingMode::synRoundDown:
            return MmeCommon::RoundingMode::RoundDown;
        case synRoundingMode::synStochasticRounding:
            return MmeCommon::RoundingMode::StochasticRounding;
        case synRoundingMode::synRoundAwayFromZero:
            return MmeCommon::RoundingMode::RoundAwayFromZero;
        case synRoundingMode::synStochasticRoundingAndNearest:
            return MmeCommon::RoundingMode::StochasticRoundingAndNearest;
        default:
        {
            HB_ASSERT(false, "Not a valid synRoundingMode!");
            return MmeCommon::RoundingMode::RoundToNearest;
        }
    }
}

void MmeDescriptorBuilder::chooseRoundingMode(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    // Input from user - default rounding mode is round to nearest.
    MmeCommon::RoundingMode roundingMode = synapseToMmeRoundingMode(mmeNode->getRoundingMode());
    // stochastic rounding is enabled only for a selected list of operations ("whitelist")
    if (GCFG_MME_ENABLE_STOCHASTIC_ROUNDING.value())
    {
        auto elementTypeOut = mmeNode->getDataOutputUnsafe(0)->getElementType();
        switch (params.opType)
        {
            case MmeCommon::e_mme_dedx:
            case MmeCommon::e_mme_transposed_dedx:
            case MmeCommon::e_mme_dedw:
            case MmeCommon::e_mme_ab:
            case MmeCommon::e_mme_atb:
            case MmeCommon::e_mme_abt:
            case MmeCommon::e_mme_atbt:

                if (elementTypeOut == syn_type_fp8_152 || elementTypeOut == syn_type_fp8_143)
                {
                    // in fp8 we use nearest for denormals.
                    roundingMode = MmeCommon::RoundingMode::StochasticRoundingAndNearest;
                }
                else
                {
                    roundingMode = MmeCommon::RoundingMode::StochasticRounding;
                }
                break;
            default:
                break;
        }
    }
    params.controls.conversionRoundingMode = roundingMode;
}

void MmeDescriptorBuilder::generateLayerParamsFromNode(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    setTensors(mmeNode, params);
    if (!m_isEagerMode)
    {
        // no need to set name for eager mode
        params.nodeName = mmeNode->getNodeName();
    }
    generateLayerParamsCommon(mmeNode, params);
}

void MmeDescriptorBuilder::setTensorsForReduction(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    TensorPtr xTensor = mmeNode->getInput(TENSOR_AUX_CD_REDUCTION);
    TensorPtr wTensor = mmeNode->getInput(TENSOR_AUX_CD_SCRATCHPAD);
    TensorPtr yTensor = mmeNode->getOutput(0);

    const auto& hal = m_graph.getTraits().getHalReader();

    params.x = getTensorViewCommon(m_chipType, *xTensor, *hal, false);
    params.w = getTensorViewCommon(m_chipType, *wTensor, *hal, false);
    params.y = getTensorViewCommon(m_chipType, *yTensor, *hal, false);

    // flatten reductionAdd input
    unsigned scratchpadSize = multiplyElements(params.w.sizes);
    unsigned numDcores      = hal->getNumDcores();
    HB_ASSERT(scratchpadSize % numDcores == 0, "Unexpected AUX_CD_SCRATCHPAD tensor sizes");
    params.w.sizes[0] = scratchpadSize / numDcores;
    params.w.sizes[1] = numDcores;
    for (int i = 2; i < params.w.sizes.size(); i++)
    {
        params.w.sizes[i] = 1;
    }
    params.w.strides[1] = params.w.sizes[0];
    params.w.strides[2] = params.w.sizes[0] * params.w.sizes[1];
    for (int i = 3; i < params.w.strides.size(); i++)
    {
        params.w.strides[i] = params.w.strides[2];
    }

    HB_ASSERT(params.w.sizes[0] == multiplyElements(params.y.sizes),
              "AUX_CD_SCRATCHPAD tensor sizes aren't as expected");
    HB_ASSERT(params.x.sizes[0] == numDcores, "AUX_CD_SCRATCHPAD tensor sizes aren't as expected");

    // flatten reductionAdd output
    params.y.sizes[0]   = params.w.sizes[0];
    params.y.sizes[1]   = params.x.sizes[1];
    params.y.strides[1] = params.y.sizes[0];
    params.y.strides[2] = params.y.sizes[0] * params.y.sizes[1];
    for (int i = 2; i < params.y.sizes.size(); i++)
    {
        params.y.sizes[i] = 1;
    }
    for (int i = 3; i < params.y.strides.size(); i++)
    {
        params.y.strides[i] = params.y.strides[2];
    }
}

void MmeDescriptorBuilder::setTensorsFromRoi(const MMENodePtr& mmeNode,
                                             const DcoreROI&   dcoreRoi,
                                             MmeLayerParams&   params,
                                             TensorPtr         outTensor)
{
    TensorPtr aTensor = mmeNode->getInput(TENSOR_IFM);
    TensorPtr bTensor = mmeNode->getInput(TENSOR_WEIGHT);
    bool      isDmaOp = mmeNode->isDmaOperation() || mmeNode->isTransposeViaGemm();
    params.opType     = getOperationTypeCommon(m_chipType, *mmeNode);
    const auto& hal   = m_graph.getTraits().getHalReader();

    // inputs- generate tile from dcoreRoi
    auto                                                        accessPattern = mmeNode->getNodeAccessPattern();
    llvm_vecsmall::SmallVector<uint64_t, tpc_lib_api::MAX_TENSOR_DIM> geometry(std::begin(dcoreRoi.size),
                                                                               std::end(dcoreRoi.size));
    llvm_vecsmall::SmallVector<uint64_t, tpc_lib_api::MAX_TENSOR_DIM> offset(std::begin(dcoreRoi.baseOffset),
                                                                             std::end(dcoreRoi.baseOffset));
    geometry.resize(accessPattern->getNodeResolution().size());
    offset.resize(accessPattern->getNodeResolution().size());

    auto tile    = gc::access_pattern::NodeTile(geometry, offset);
    auto aTile   = accessPattern->getTensorTile(aTensor, tile);
    auto bTile   = bTensor ? accessPattern->getTensorTile(bTensor, tile) : TensorTile(geometry);
    auto outTile = accessPattern->getTensorTile(outTensor, tile);

    MmeTensorView aView = getTensorViewFromTile(m_chipType, aTensor, aTile, *hal, isDmaOp);
    MmeTensorView bView =
        bTensor ? getTensorViewFromTile(m_chipType, bTensor, bTile, *hal, isDmaOp) : MmeCommon::MmeTensorView();
    MmeTensorView outView = getTensorViewFromTile(m_chipType, outTensor, outTile, *hal, isDmaOp);

    setTensorViewByOp(*mmeNode, params, aView, bView, outView);
}

void MmeDescriptorBuilder::generateLayerParamsForReduction(const MMENodePtr& mmeNode,
                                                           MmeLayerParams&   params,
                                                           unsigned          reductionLevel)
{
    HB_ASSERT(!m_isEagerMode, "not yet implemented");
    params.opType = MmeCommon::e_mme_reductionAdd;
    setTensorsForReduction(mmeNode, params);
    params.nodeName = mmeNode->getNodeName();
    generateLayerParamsCommon(mmeNode, params);
    params.strategy.reductionLevel = reductionLevel;
    params.strategy.pipelineLevel  = 1;
    params.strategy.pattern        = e_mme_sp_reduction_fck;
    params.strategy.geometry       = e_mme_geometry_4xw;
}

void MmeDescriptorBuilder::generateLayerParamsFromRoi(const MMENodePtr& mmeNode,
                                                      MmeLayerParams&   params,
                                                      DcoreROI&         dcoreRoi,
                                                      unsigned          dcoreIdx)
{
    HB_ASSERT(!m_isEagerMode, "not yet implemented");

    TensorPtr outTensor = mmeNode->getOutput(TENSOR_OFM);
    if (mmeNode->isCdPerforated())
    {
        outTensor = mmeNode->getInput(TENSOR_AUX_CD_SCRATCHPAD);
    }

    setTensorsFromRoi(mmeNode, dcoreRoi, params, outTensor);
    normalizeTensorDims(*mmeNode, m_chipType, params);
    params.nodeName = mmeNode->getNodeName();
    generateLayerParamsCommon(mmeNode, params);

    // When cdPerforated, params output is a scratchpad aux tensor (stores partial results),
    // therefore, shouldn't use reduction
    if (mmeNode->isCdPerforated())
    {
        params.controls.atomicAdd = false;
    }
}

void MmeDescriptorBuilder::setSpSizeAndBase(MmeLayerParams& params)
{
    params.spBase       = 0;
    params.spSize       = 1;
    const auto& spSizes = (params.opType == e_mme_dedx || params.opType == MmeCommon::e_mme_transposed_dedx)
                              ? params.x.sizes
                              : params.y.sizes;
    params.spSize       = MmeCommon::multiplyElements(std::begin(spSizes) + 1, std::end(spSizes));
}

void MmeDescriptorBuilder::setConvParams(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    MmeBrainIfc::getMmeConvParams(*mmeNode, params.conv);
}

void MmeDescriptorBuilder::setControls(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    if (isInferenceQuantization(m_graph) && mmeNode->getInput(TENSOR_IFM)->getElementType() == syn_type_fp8_143)
    {
        params.controls.clippingEn = true;
    }

    chooseRoundingMode(mmeNode, params);
    chooseSignalingMode(params);
    params.controls.atomicAdd    = getMmeSramReduction(*mmeNode);
    const TensorPtr& secondInput = mmeNode->getInput(1);
    if (m_graph.getQuantizationEnabled())
    {
        params.controls.fp8BiasIn = mmeNode->getInput(0)->getExpBias();
        if (secondInput)
        {
            params.controls.fp8BiasIn2 = secondInput->getExpBias();
        }
        params.controls.fp8BiasOut = mmeNode->getDataOutputUnsafe(0)->getExpBias();
    }
    else
    {
        const MmeExpBias& mmeExpBias = mmeNode->getMmeExpBias();
        params.controls.fp8BiasIn    = mmeExpBias.fp8BiasIn[TENSOR_IFM];
        if (secondInput && mmeExpBias.fp8BiasIn.size() > TENSOR_WEIGHT)
        {
            params.controls.fp8BiasIn2 = mmeExpBias.fp8BiasIn[TENSOR_WEIGHT];
        }
        params.controls.fp8BiasOut = mmeExpBias.fp8BiasOut;
    }
    if (mmeNode->isTransposeViaGemm() && secondInput)
    {
        // on transpose via gemm we dont use inf\nan so that u\int will not be clipped\inferred as inf\nan.
        params.controls.infNanModeA = InfNanMode::e_mme_no_inf_nan;
        params.controls.infNanModeB = InfNanMode::e_mme_no_inf_nan;
        params.controls.infNanModeOut = InfNanMode::e_mme_no_inf_nan;
        // bias of 15 is good for all dtypes (not only fp8_152)
        params.controls.fp8BiasIn = EXPONENT_BIAS_FP8_152_15;
        params.controls.fp8BiasIn2 = EXPONENT_BIAS_FP8_152_15;
        params.controls.fp8BiasOut = EXPONENT_BIAS_FP8_152_15;
    }
}

void MmeDescriptorBuilder::setStrategy(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    params.strategy.teAccelerationEn = !mmeNode->isDynamicShape();
    auto& bigNode = mmeNode->getNodeAnnotation().origBigNode;
    if (bigNode && bigNode->getNodeAnnotation().mmeMetaData.takeStrategyFromAnnotation)
    {
        params.strategy = bigNode->getNodeAnnotation().mmeMetaData.mmeStrategy;
        params.strategy.mmeLimit  = mmeNode->getGraphTraits()->getHalReader()->getNumMmeEngines();
    }
    else if (!m_isEagerMode)  // Do not choose smart strategy for eager
    {
        TensorPtr xTensor;
        TensorPtr wTensor;
        TensorPtr yTensor;
        TensorPtr oTensor;
        getTensorRolesCommon(*mmeNode, params.opType, xTensor, wTensor, yTensor, oTensor);
        params.strategy.mmeLimit  = mmeNode->getGraphTraits()->getHalReader()->getNumMmeEngines();
        chooseStrategy(mmeNode, xTensor, wTensor, params);
    }
    else
    {
        params.strategy.mmeLimit  = mmeNode->getGraphTraits()->getHalReader()->getNumMmeEngines();
        chooseRecommendedParamsForEager(params, mmeNode->getMmeBrainIfc()->getOperationModes());
        chooseStrategyForEager(mmeNode.get(), params);
    }

    if (!m_isEagerMode)
    {
        graph_serialize::MmeStrategySerializer::processNewStrategy(params.strategy,
                                                                   m_graph.getRecipeName(),
                                                                   mmeNode->getNodeName());
    }
}
MmeDescriptorBuilder::MmeDescriptorBuilder(const HabanaGraph& graph) : m_graph(graph)
{
    m_chipType    = MmeBrainIfc::getMmeChipType(m_graph.getTraits().getHalReader()->getDeviceType());
    m_isEagerMode = graph.getCompilationMode() == CompilationMode::Eager;
}

void MmeDescriptorBuilder::setTensors(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    bool isDmaOp  = mmeNode->isDmaOperation() || mmeNode->isTransposeViaGemm();
    params.opType = getOperationTypeCommon(m_chipType, *mmeNode);
    TensorPtr xTensor, wTensor, yTensor, oTensor;
    getTensorRolesCommon(*mmeNode, params.opType, xTensor, wTensor, yTensor, oTensor);
    const auto& hal = mmeNode->getGraphTraits()->getHalReader();
    params.x        = getTensorViewCommon(m_chipType, *xTensor, *hal, isDmaOp);
    if (wTensor) params.w = getTensorViewCommon(m_chipType, *wTensor, *hal, isDmaOp);
    params.y        = getTensorViewCommon(m_chipType, *yTensor, *hal, isDmaOp);
    normalizeTensorDims(*mmeNode, m_chipType, params);
}

void MmeDescriptorBuilder::chooseRecommendedParamsForEager(MmeLayerParams& layerParams, const MmeCommon::MmeBrainOperationModes& brainOperationModes)
{
    // Optimizations applied on tensor shape
    MmeCommon::MmeBrain::trivialDimsReduction(layerParams);
}

void MmeDescriptorBuilder::chooseGeometryForEager(const MmeNode* mmeNode, MmeLayerParams& layerParams)
{
    // choose geometry based on the simplistic heuristic
    switch (layerParams.opType)
    {
        case e_mme_trans:
            layerParams.strategy.geometry = e_mme_geometry_4xw;
            break;
        case e_mme_memcpy:
            EAGER_ASSERT(false, "memcpy is not supported yet");
            layerParams.strategy.geometry = e_mme_geometry_4xw;
            break;
        default:
        {
            const eager_mode::EagerMmeBrainBase& eagerBrain = eager_mode::getEagerMmeBrain(m_graph);
            layerParams.strategy.geometry =
                eagerBrain.getBestGeometry(layerParams.getFcdSize(), layerParams.getSpatialSize()).geometry->geometry;
        }
        break;
    }
}

void MmeDescriptorBuilder::chooseStrategyForEager(const MmeNode* mmeNode, MmeLayerParams& layerParams)
{
    const auto& mmeMetaData = mmeNode->getNodeAnnotation().mmeMetaData;
    auto&       strategy    = layerParams.strategy;

    strategy.cdConcurrencyEn    = mmeMetaData.mmeStrategy.cdConcurrencyEn;
    strategy.batchConcurrencyEn = mmeMetaData.mmeStrategy.batchConcurrencyEn;
    strategy.packingFactor      = mmeMetaData.packing[PACKING_X];
    strategy.pipelineLevel      = 1;
    strategy.loweringEn         = true;
    strategy.alignedAddresses   = false;

    EAGER_ASSERT(!mmeNode->isDynamicShape() && strategy.flattenEn,
                 "flattenEn should be updated once Eager supports dynamic shapes");

    chooseGeometryForEager(mmeNode, layerParams);

    if (layerParams.isFwdOrDedx())
    {
        strategy.pattern = e_mme_z_reduction_skf;
    }
    else
    {
        strategy.pattern = e_mme_sp_reduction_fck;
    }

    strategy.sbReuse = [&] {
        if (unlikely(!GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value())) return false;

        {
            // 0 for force-off and 1 for force-on, otherwise (likely) continue with our default hueristics
            auto conf = GCFG_ENABLE_EAGER_SB_REUSE_G3.value();
            if (unlikely(!(conf & ~1))) return !!conf;
        }

        // TODO[SW-169983]: Remove these limitations
        if (strategy.cdConcurrencyEn) return false;

        // TODO[SW-169983]: Remove this limitation
        return false;
    }();
}

void MmeDescriptorBuilder::generateLayerParamsCommon(const MMENodePtr& mmeNode, MmeLayerParams& params)
{
    params.useDescCache = GCFG_ENABLE_MME_DESCRIPTOR_CACHE.value();
    setSpSizeAndBase(params);
    setCacheDirectives(mmeNode, params, m_isEagerMode);
    setTracing(params);
    setConvParams(mmeNode, params);
    setControls(mmeNode, params);
    setStrategy(mmeNode, params);
}

}  // namespace gaudi3
