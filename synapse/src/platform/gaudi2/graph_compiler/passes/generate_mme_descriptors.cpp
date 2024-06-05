#include "generate_mme_descriptors.h"

// synapse api (relative to include/)
#include "synapse_common_types.h"

#include "eager/eager_interface.h"
#include "eager/lib/eager_brain_base.h"
#include "eager/lib/utils/general_defs.h"

// src/graph_compiler/
#include "habana_global_conf.h"
#include "mme_brain_ifc.h"
#include "mme_desc_gen_utils.h"
#include "mme/mme_logger.h"
#include "mme/mme_strategy_serializer.h"

// src/infra/
#include "defs.h"
#include "log_manager.h"

// src/platform/gaudi2/graph_compiler/
#include "../descriptor_generator.h"
#include "../gaudi2_graph.h"

// relative to <mme>/
#include "include/gaudi2/mme_descriptor_generator.h"
#include "include/mme_common/mme_common_enum.h"

// relative to <specs>
#include "gaudi2/mme.h"

#include <algorithm>
#include <functional>
#include <include/general_utils.h>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Gaudi2;
using namespace MmeCommon;

namespace gaudi2
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
                HB_ASSERT(false, "Unsupported Gaudi2 MME reduction operation {}", reductionInfoMme.reductionOperation);
                break;
        }
    }

    return isAtomicAdd;
}

bool generateMmeDescriptors(Gaudi2Graph& g)
{
    const NodeVector& nodes = g.getExeSortedNodes();
    Gaudi2CodeGenerator* gaudi2CodeGen = downcaster<Gaudi2CodeGenerator>(g.getCodeGenerator().get()); //TODO SW-78739 when pass will get Gaudi2CodeGen
    for (const NodePtr& node : nodes)
    {
        if (HabanaGraph::runsOnMME(node))
        {
            MmeLayerParams params = MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi2);
            auto           descGenerator = MMEBrain::createDescriptorGenerator(g, node, params);
            NodePtr        nodeShared = g.getNodeSharedPtr(*node);
            HB_ASSERT(nodeShared != nullptr, "Invalid node object");
            // Save descriptor generator in graph.
            gaudi2CodeGen->setMmeNodeDescriptorGenerator(nodeShared, descGenerator);
        }
    }
    return true;
}

/*----------------------------------- MME Brain --------------------------------*/

void MMEBrain::printDebugInfoAndSetAnnotations(MmeDescriptorGeneratorPtr& descGenerator,
                                               MmeNode*                   mmeNode,
                                               const MmeLayerParams&      params)
{
    MmeLogger           mmeLogger;
    MmeCommon::PerfAttr perfAttr;
    descGenerator->getMMEBrain().getPerfAttr(params, perfAttr);
    mmeNode->getNodeAnnotation().mmeMetaData.mmePerfAttr = std::make_shared<MmeCommon::PerfAttr>(perfAttr);
    mmeLogger.printDebugInfo(perfAttr, &*descGenerator);

    // Fill mmeStrategy info for trace analyzer
    mmeNode->getNodeAnnotation().mmeMetaData.mmeStrategyDebugString =
        mmeLogger.getMmeStrategyInfo(params, *descGenerator, e_mme_Gaudi2);
}

void MMEBrain::generateLayerParamsFromNode(HabanaGraph& g, const NodePtr& node, MmeLayerParams& params)
{
    MmeNode* mmeNode = static_cast<MmeNode*>(node.get());
    params.opType   = getOperationTypeCommon(e_mme_Gaudi2, *mmeNode);
    TensorPtr xTensor;
    TensorPtr wTensor;
    TensorPtr yTensor;
    TensorPtr oTensor;
    getTensorRolesCommon(*mmeNode, params.opType, xTensor, wTensor, yTensor, oTensor);
    const auto& hal = g.getHALReader();

    params.x = getTensorViewCommon(e_mme_Gaudi2, *xTensor, *hal, false);
    params.w = getTensorViewCommon(e_mme_Gaudi2, *wTensor, *hal, false);
    params.y = getTensorViewCommon(e_mme_Gaudi2, *yTensor, *hal, false);
    if (node.get()->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        params.strategy.maskedBgemm = true;
        params.xAux = getTensorViewCommon(e_mme_Gaudi2, *(node->getInput(TENSOR_AUX_BGEMM_MASK_A)), *hal, false);
        params.wAux = getTensorViewCommon(e_mme_Gaudi2, *(node->getInput(TENSOR_AUX_BGEMM_MASK_B)), *hal, false);
        params.yAux = params.y;
    }
    normalizeTensorDims(*mmeNode, e_mme_Gaudi2, params);

    params.spBase = 0;
    params.spSize = 1;
    params.useDescCache = GCFG_ENABLE_MME_DESCRIPTOR_CACHE.value();

    const auto& spSizes =
        (params.opType == e_mme_dedx || params.opType == e_mme_transposed_dedx) ? params.x.sizes : params.y.sizes;
    params.spSize       = ::multiplyElements(std::begin(spSizes) + 1, std::end(spSizes));

    setTracing(params);
    MmeBrainIfc::getMmeConvParams(*mmeNode, params.conv);

    params.controls.slaveSignaling = Gaudi2Graph::isMmeSlaveSignalingEnabled();

    const Tensor& ofm = *mmeNode->getDataOutputUnsafe(TENSOR_OFM);
    chooseRoundingMode(mmeNode, params, ofm.getElementType());
    if (g.getQuantizationEnabled())
    {
        params.controls.fp8BiasIn  = mmeNode->getDataInputUnsafe(0)->getExpBias();
        params.controls.fp8BiasIn2 = mmeNode->getDataInputUnsafe(1)->getExpBias();
        params.controls.fp8BiasOut = ofm.getExpBias();
    }
    else
    {
        const MmeExpBias& mmeExpBias = mmeNode->getMmeExpBias();
        HB_ASSERT(mmeExpBias.fp8BiasIn.size() > TENSOR_WEIGHT, "expecting two elements in mmeExpBias.fp8BiasIn");
        params.controls.fp8BiasIn    = mmeExpBias.fp8BiasIn[TENSOR_IFM];
        params.controls.fp8BiasIn2   = mmeExpBias.fp8BiasIn[TENSOR_WEIGHT];
        params.controls.fp8BiasOut   = mmeExpBias.fp8BiasOut;
    }
    // Enable clipping for fp8 in inference
    if (g.getInferenceMode() && node->getInput(TENSOR_IFM)->getElementType() == syn_type_fp8_143)
    {
        params.controls.clippingEn = true;
    }

    if (g.getCompilationMode() == CompilationMode::Eager)
    {
        chooseRecommendedParamsForEager(params, mmeNode->getMmeBrainIfc()->getOperationModes());
        params.controls.atomicAdd = ofm.isReductionEnabled();
        EAGER_ASSERT(!params.controls.atomicAdd || ofm.getRealReductionInfo().reductionOperation == REDUCTION_ADD,
                     "Unsupported Gaudi2 MME reduction operation");
        chooseStrategyForEager(g, mmeNode, params);
        params.controls.signalingMode = e_mme_signaling_once;
        params.controls.squashIORois  = true;
#ifndef NDEBUG
        const std::string& nodeName = node->getNodeName();
        SET_TEMP_LOG_CONTEXT(nodeName);
        params.nodeName = nodeName;
        MmeLogger mmeLogger;
        mmeLogger.printMmeParams(params);
#endif
    }
    else
    {
        const std::string& nodeName = node->getNodeName();
        SET_TEMP_LOG_CONTEXT(nodeName);
        params.nodeName = nodeName;

        params.controls.atomicAdd = getMmeSramReduction(*mmeNode);

        // TODO This is only an initial configuration (need to add round mode, relu etc.)
        chooseStrategy(g, node, xTensor, wTensor, yTensor, node->getNodeAnnotation().splitToLogicalROIs, params);
        graph_serialize::MmeStrategySerializer::processNewStrategy(params.strategy,
                                                                   g.getRecipeName(),
                                                                   mmeNode->getNodeName());
        chooseSignalingMode(mmeNode, params);
        MmeLogger mmeLogger;
        mmeLogger.printMmeParams(params);
    }
}

std::shared_ptr<MmeDescriptorGenerator>
MMEBrain::createDescriptorGenerator(HabanaGraph& g, const NodePtr& node, MmeLayerParams& params)
{
    MmeNode*                  mmeNode = static_cast<MmeNode*>(node.get());
    MmeDescriptorGeneratorPtr descGenerator;

    // For Eager flow we call generateLayerParamsFromNode from outside and skip patching
    // as patching is done in Eager internally to support cache hit hot path without intermidate
    // copy of activations.
    if (g.getCompilationMode() == CompilationMode::Eager)
    {
        // generate descriptor and activations
        descGenerator = MmeDescriptorGenerator::createMmeDescGenerator();
        HB_ASSERT(descGenerator != nullptr, "Invalid MME descriptor generator");
        descGenerator->getMMEBrain().setOperationModes(mmeNode->getMmeBrainIfc()->getOperationModes());
        descGenerator->setParams(params);
        descGenerator->mmeGenerateActivations();

#ifndef NDEBUG
        printDebugInfoAndSetAnnotations(descGenerator, mmeNode, params);
#endif
    }
    else
    {
        generateLayerParamsFromNode(g, node, params);
        // generate descriptor and activations
        descGenerator = MmeDescriptorGenerator::createMmeDescGenerator();
        HB_ASSERT(descGenerator != nullptr, "Invalid MME descriptor generator");
        descGenerator->getMMEBrain().setOperationModes(mmeNode->getMmeBrainIfc()->getOperationModes());
        descGenerator->setParams(params);
        descGenerator->mmeGenerateActivations();
        printDebugInfoAndSetAnnotations(descGenerator, mmeNode, params);
        // Patch mme tensors
        patchActivations(*mmeNode, *descGenerator);
    }
    return descGenerator;
}

void MMEBrain::patchActivations(const MmeNode& mmeNode,  MmeDescriptorGenerator& descGenerator)
{
    TensorPtr aTensor;
    TensorPtr bTensor;
    TensorPtr cTensor;
    TensorPtr oTensor;
    AuxTensorArray auxTensors;
    DescriptorGenerator::getInputOutputTensors(mmeNode, aTensor, bTensor, cTensor, oTensor, auxTensors);
    const EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi2, mmeNode);
    bool calcRoi = mmeNode.getGraphTraits()->getCompilationMode() != CompilationMode::Eager;

    // Record all tensor addresses
    // patchMetaData struct holds for each logical tensor its actual address and location (isSram).
    // It always holds a, c. b, secondary output and aux tensors are optional.
    MmePatchMetaData patchMetaData;
    patchMetaData.bOperandUsed = (opType != e_mme_memcpy && opType != e_mme_trans);
    patchMetaData.oOperandUsed = (oTensor ? true : false);

    patchMetaData.tensorMetaData[INPUT_TENSOR_A] = {aTensor->getTensorOffset(), aTensor->tensorAllocatedInSram()};
    if (patchMetaData.bOperandUsed)
    {
        patchMetaData.tensorMetaData[INPUT_TENSOR_B] = {bTensor->getTensorOffset(), bTensor->tensorAllocatedInSram()};
    }
    patchMetaData.tensorMetaData[OUTPUT_TENSOR_C] = {cTensor->getTensorOffset(), cTensor->tensorAllocatedInSram()};
    if (patchMetaData.oOperandUsed)
    {
        patchMetaData.tensorMetaData[OUTPUT_TENSOR_O] = {oTensor->getTensorOffset(), oTensor->tensorAllocatedInSram()};
    }

    // Aux tensors
    if (auxTensors[MASKED_BGEMM_A])
    {
        patchMetaData.tensorMetaData[AUX_TENSOR_0] = {auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_A]->getTensorOffset(),
                                                      auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_A]->tensorAllocatedInSram()};
    }
    if (auxTensors[MASKED_BGEMM_B])
    {
        patchMetaData.tensorMetaData[AUX_TENSOR_1] = {auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_B]->getTensorOffset(),
                                                      auxTensors[MmeCommon::MmeAuxTensorIdx::MASKED_BGEMM_B]->tensorAllocatedInSram()};
    }

    // Patch the mme tensors
    descGenerator.patchMmeDescriptors(patchMetaData, calcRoi);

    const auto& layerParams = descGenerator.getParams();
    if (layerParams.tracing.traceMode != e_mme_trace_mode_none)
    {
        descGenerator.patchContextId(mmeNode.getContextId());
        LOG_TRACE(GC, "MME node {} got context id {}", mmeNode.getNodeName(), mmeNode.getContextId());
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

    static constexpr double UTILIZATION_THRESHOLD = 1.01;  // 1 percent

    // returns true if lhs "is less than" rhs.
    bool operator()(const ParamsAndPerf& lhs, const ParamsAndPerf& rhs)
    {
        // calculate the ratio between the solutions performance, if its higher than 1% the higher solution will be
        // chosen
        float perfRatio = lhs.perf.mmeUtilization / rhs.perf.mmeUtilization;
        perfRatio       = perfRatio > 1 ? perfRatio : 1 / perfRatio;
        if (perfRatio > UTILIZATION_THRESHOLD)
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
        // utilization is similar and both inputs are in the same memory - check num of fetches
        auto lhsFetch = lhs.perf.fetchNrA * opA->getDenseSizeInBytes() + lhs.perf.fetchNrB * opB->getDenseSizeInBytes();
        auto rhsFetch = rhs.perf.fetchNrA * opA->getDenseSizeInBytes() + rhs.perf.fetchNrB * opB->getDenseSizeInBytes();
        if (lhsFetch != rhsFetch)
        {
            return lhsFetch > rhsFetch;
        }
        // else - the lowest number of activations
        return lhs.perf.numOfActivations > rhs.perf.numOfActivations;
    }
};

ParamsAndPerfsVector generatePerfAttrCandidates(MmeLayerParams layerParams, MmeCommon::MmeBrain& brain)
{
    ParamsAndPerfsVector   candidates;
    const std::string_view geometryNames[] = {"4xw", "2xw", "2xh", "4xh"};
    for (bool walkDown : {false, true})
    {
        layerParams.setPattern(walkDown);
        for (auto geometry : brain.getGeometries(layerParams))
        {
            layerParams.strategy.geometry = geometry;
            MmeCommon::PerfAttr perf {};
            brain.getPerfAttr(layerParams, perf);
            LOG_TRACE(MME_STACK,
                      "Params: walking down: {}, geometry: {} => Perf: utilization: {}, #fetchA: {}, #fetchB: {} "
                      "numOfActivations {}, cycles: {}",
                      walkDown,
                      geometryNames[geometry - 3],
                      perf.mmeUtilization,
                      perf.fetchNrA,
                      perf.fetchNrB,
                      perf.numOfActivations,
                      perf.expectedRuntimeCycles);
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

void MMEBrain::chooseRecommendedParamsForEager(MmeLayerParams& layerParams, const MmeBrainOperationModes& brainOperationModes)
{
    // Optimizations applied on tensor shape
    MmeCommon::MmeBrain::trivialDimsReduction(layerParams);
}

template<typename UnsignedIntegral>
static constexpr auto divRoundUp(UnsignedIntegral value, std::size_t N)
{
    static_assert(std::is_unsigned_v<UnsignedIntegral> && std::is_integral_v<UnsignedIntegral>);
    return (value / N) + !!(value % N);
}

enum class ReuseStrategy
{
    NONE,
    A,
    B
};

namespace
{
struct EagerPrediction
{
    ReuseStrategy resue {ReuseStrategy::NONE};
    uint64_t      fcdGeoPerReuseNr {};
    uint64_t      spGeoPerReuseNr {};
};
}  // anonymous namespace

// get the predicted sb-reuse and if it's B swap the pattern to non-raster(
static EagerPrediction getPredictedReuse(const MmeNode&        mmeNode,
                                         const MmeLayerParams& layerParams,
                                         unsigned              geoTotalElemWidth,
                                         unsigned              geoTotalElemHeight,
                                         bool                  flattened,
                                         /*OUT*/ bool&         preferNonRaster)
{
    EagerPrediction prediction {};

    const MmeStrategy& strategy = layerParams.strategy;
    EAGER_ASSERT(strategy.pattern == e_mme_sp_reduction_fck, "Unsupported MME strategy pattern for Eager");
    EAGER_ASSERT(layerParams.isPatternRaster(), "");

    const auto& aView = mmeNode.getInput(0)->getAllNSizesInElements();
    const auto& bView = mmeNode.getInput(1)->getAllNSizesInElements();
    const auto& cView = mmeNode.getOutput(0)->getAllNSizesInElements();

    // If B is smaller, maybe we'd prefer a non-raster pattern so that we'd
    // possibly end-up with sb-reuse on B
    preferNonRaster = std::accumulate(bView.begin(), bView.begin() + MAX_DIMENSION, uint64_t {1}, std::multiplies<>()) <
                      std::accumulate(aView.begin(), aView.begin() + MAX_DIMENSION, uint64_t {1}, std::multiplies<>());

    // When flattened, the real cView has dim 3 folded into dim 2.
    // To avoid copying the dims vector just to modify this, we just skip dim 3 in such a case here,
    // since it's not a real batch dim.
    const auto BATCH_STARTING_DIM = MAX_DIMENSION - c_batchDimNr + static_cast<int>(flattened);
    const auto batchesNr =
        std::accumulate(cView.data() + BATCH_STARTING_DIM, cView.data() + MAX_DIMENSION, 1, std::multiplies<>());

    static constexpr auto MAX_BATCH_NR_FOR_SB_REUSE = 16;
    if (unlikely(batchesNr > MAX_BATCH_NR_FOR_SB_REUSE))  // TODO: no masked bgemm handling for now
    {
        return prediction;  // No reuse
    }

    prediction.fcdGeoPerReuseNr = divRoundUp(cView[0], geoTotalElemWidth);

    // TODO: Using calcSpGeoContinuousLength, for gemm or batchgemm cases only.
    prediction.spGeoPerReuseNr =
        (batchesNr == 1) ||  // For GEMM: Assumes no filters -> All GEO contribute to same output.
                (strategy.pattern == e_mme_sp_reduction_fkc)  // For BGEMM: If movement in spatial direction
            ? divRoundUp(flattened ? cView[1] * cView[2] : cView[1], geoTotalElemHeight)
            : 1;

    // Raster matters more in graph-mode where there's some coordination between
    // MME and TPC for pipelining. In the eager case when there's no pipelining,
    // in case of reuse A we'll always prefer to raster and in case of B not to raster.

    if (!preferNonRaster)
    {  // Scan fast direction first
        if (prediction.fcdGeoPerReuseNr > 1)
        {
            prediction.resue = ReuseStrategy::A;  // B consumes multiple GEOs
        }
        else if (prediction.spGeoPerReuseNr > 1)
        {
            prediction.resue = ReuseStrategy::B;  // A produces multiple EUs
            preferNonRaster  = true;
        }
    }
    else
    {  // Scan spatial direction first
        if (prediction.spGeoPerReuseNr > 1)
        {
            prediction.resue = ReuseStrategy::B;  // A consumes multiple GEOs
        }
        else if (prediction.fcdGeoPerReuseNr > 1)
        {
            prediction.resue = ReuseStrategy::A;  // B consumes multiple GEOs
            preferNonRaster  = false;
        }
    }

    return prediction;
}

// Basic evaluation if sb-reuse will benefit device-time but not damage compile-time too much.
static bool eagerSbReuseHueristic(const MmeNode&                                                    mmeNode,
                                  const MmeLayerParams&                                             layerParams,
                                  const eager_mode::EagerMmeBrainBase::SupportedGeometryProperties& selectedGeo,
                                  /*OUT*/ bool&                                                     preferNonRaster)
{
    static constexpr auto TEMP_THRESHOLD = 10;

    const auto& strategy = layerParams.strategy;
    EAGER_ASSERT(strategy.pattern == e_mme_sp_reduction_fck, "");
    EAGER_ASSERT(strategy.cdConcurrencyEn == TurnedOff, "");  // TODO: reconsider limitation

    {
        if (unlikely(!GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value())) return false;

        // 0 for force-off and 1 for force-on, otherwise (likely) continue with our default hueristics
        auto conf = GCFG_ENABLE_EAGER_SB_REUSE_G2.value();
        if (unlikely(!(conf & ~1))) return !!conf;
    }

    // TODO[SW-169983]: Remove these limitations
    {
        const Node::eNodeType type = mmeNode.getNodeType();
        if (type != Node::TYPE_GEMM && type != Node::TYPE_BATCH_GEMM) return false;
    }

    const auto& aDims = mmeNode.getInput(0)->getAllNSizesInElements();
    const auto& bDims = mmeNode.getInput(1)->getAllNSizesInElements();
    const auto& cDims = mmeNode.getOutput(0)->getAllNSizesInElements();

    const bool flattened = [&] {
        EAGER_ASSERT(!strategy.dualGemm && !strategy.maskedBgemm, "If supported, return false");

        const auto& aStrides = mmeNode.getInput(0)->getNStridesInBytes();
        const auto& cStrides = mmeNode.getOutput(0)->getNStridesInBytes();

        return strategy.flattenEn &&
               (layerParams.opType == e_mme_ab || layerParams.opType == e_mme_abt) &&  // A transposed bgemm ops
               (bDims[2] == 1 && cDims[2] != 1) &&  // B's first batch dim broadcasted
               aDims[2] == cDims[2] &&              // sanity check that A and C first batch dim matches
               (aStrides[2] == aStrides[1] * aDims[1] &&
                cStrides[2] == cStrides[1] * cDims[1]);  // simple logical reshape (already matching strides)
    }();

    const auto geoTotalElemWidth  = selectedGeo.width;
    const auto geoTotalElemHeight = selectedGeo.height;
    auto       prediciton =
        getPredictedReuse(mmeNode, layerParams, geoTotalElemWidth, geoTotalElemHeight, flattened, preferNonRaster);
    if (prediciton.resue == ReuseStrategy::NONE) return false;

    // Note that A is tranposed by default
    const auto aTransposed = layerParams.opType == e_mme_ab || layerParams.opType == e_mme_abt;
    const auto bTransposed = layerParams.opType == e_mme_abt || layerParams.opType == e_mme_atbt;

    const auto [commonDim, reusedDim, nonReusedDim] = [&] {
        // Note in MME we tend to use "b" and "c" for dims instead of "a",
        // since "a" is a bit more confusing since HW tranposes it,
        // and since both a and c width is potentially affected by flattening.
        const auto cWidth = flattened ? cDims[1] * cDims[2] : cDims[1];
        return prediciton.resue == ReuseStrategy::A ? std::make_tuple(bDims[1 - (int)bTransposed], cWidth, cDims[0])
                                                    : std::make_tuple(bDims[1 - (int)bTransposed], cDims[0], cWidth);
    }();

    const auto& reusedInput          = mmeNode.getInput(prediciton.resue == ReuseStrategy::A ? 0 : 1);
    const auto  elementSize          = reusedInput->getElementSizeInBytes();
    const auto  reusedGeoSideSize    = prediciton.resue == ReuseStrategy::A ? geoTotalElemHeight : geoTotalElemWidth;
    const auto  nonReusedGeoSideSize = prediciton.resue == ReuseStrategy::A ? geoTotalElemWidth : geoTotalElemHeight;

    const auto sbsSizeInBytes = [&] {
        const auto numElementsInCL = Gaudi2::Mme::c_cl_size / elementSize;
        EAGER_ASSERT(isPowerOf2(numElementsInCL), "");

        const auto areStridesAligned = [&](const TStride* strides) {
            return std::all_of(strides + 1, strides + MAX_DIMENSION, [=](TStride s) {
                return !(s & (numElementsInCL - 1));
            });
        };

        // TODO: isTensorAddressCacheLineAligned is done later to fill strategy.alignedAddresses, could be reused
        const bool isAligned = isTensorAddressCacheLineAligned(reusedInput, /*ignoreTensorAliasing*/ false) &&
                               areStridesAligned(reusedInput->getNStridesInBytes());
        const auto sbSizeInCachelinesPerPort = (uint64_t)Gaudi2::Mme::c_mme_sb_size >> !isAligned;

        const auto portsNr = [&] {
            const bool isFp8 = [&] {
                const auto elementType = reusedInput->getElementType();
                return elementType == syn_type_fp8_143 || elementType == syn_type_fp8_152;
            }();
            return (reusedGeoSideSize / Gaudi2::Mme::c_cl_size) << !isFp8;  // portSize is 128 for fp8 and 64 otherwise
        }();

        // TODO: inline portsNr to cancel out cl size? or is it more readable like this?
        return sbSizeInCachelinesPerPort * Gaudi2::Mme::c_cl_size * portsNr;
    }();

    // The common dim split number is also called the number of partials,
    // since these results have to be accumulated as intermediate values,
    // in the output accumulators, before being flushed to the real output.
    const auto partialNr =
        divRoundUp(commonDim * std::min<uint64_t>(reusedGeoSideSize, reusedDim) * elementSize, sbsSizeInBytes);
    if (partialNr > TEMP_THRESHOLD) return false;

    if (partialNr == 1)
    {
        // When there are no partials, the only limit is on the nubmber of sb reuses on the non-reused dim.
        // And on the reused dim we'll require no extra splits.
        const auto totalReuse =
            divRoundUp(prediciton.resue == ReuseStrategy::A ? prediciton.fcdGeoPerReuseNr : prediciton.spGeoPerReuseNr,
                       Gaudi2::Mme::c_mme_max_sb_reuse);
        if (totalReuse > TEMP_THRESHOLD) return false;
    }
    else
    {
        // In case of partials, we are limited not by the 240 max sb reuse but by the much smaller number of
        // 4 accumulators (aka partials).
        const auto reusedDimRepeats =
            prediciton.resue == ReuseStrategy::A ? prediciton.spGeoPerReuseNr : prediciton.fcdGeoPerReuseNr;
        const auto nonReusedDimRepeats =
            divRoundUp(prediciton.resue == ReuseStrategy::A ? prediciton.fcdGeoPerReuseNr : prediciton.spGeoPerReuseNr,
                       Gaudi2::Mme::c_mme_accums_nr);
        if (partialNr * reusedDimRepeats * nonReusedDimRepeats > TEMP_THRESHOLD) return false;
    }

    return true;
}

void MMEBrain::chooseStrategyForEager(HabanaGraph& g, const MmeNode* mmeNode, MmeLayerParams& layerParams)
{
    const auto& mmeMetaData = mmeNode->getNodeAnnotation().mmeMetaData;
    auto&       strategy    = layerParams.strategy;

    strategy.cdConcurrencyEn    = mmeMetaData.mmeStrategy.cdConcurrencyEn;
    strategy.batchConcurrencyEn = mmeMetaData.mmeStrategy.batchConcurrencyEn;
    strategy.packingFactor      = mmeMetaData.packing[PACKING_X];
    strategy.pipelineLevel      = 1;
    strategy.loweringEn         = true;
    strategy.alignedAddresses   = false;
    strategy.sbReuse            = false;  // initial value, possibly overriden

    EAGER_ASSERT(!mmeNode->isDynamicShape() && strategy.flattenEn,
                 "flattenEn should be updated once Eager supports dynamic shapes");

    const eager_mode::EagerMmeBrainBase& eagerMmeBrain = eager_mode::getEagerMmeBrain(g);
    unsigned                             spatialSize   = layerParams.getSpatialSize();

    // CD concurrency determines walk pattern and geometry at early stage
    if (strategy.cdConcurrencyEn == TurnedOff)
    {
        const auto& selectedGeo = *eagerMmeBrain.getBestGeometry(layerParams.getFcdSize(), spatialSize).geometry;
        strategy.geometry       = selectedGeo.geometry;
        if (layerParams.isDmaOperation())
        {
            strategy.pattern = e_mme_sp_reduction_fkc;
        }
        else if (layerParams.isFwdOrDedx())
        {
            strategy.pattern = e_mme_z_reduction_skf;
        }
        else
        {
            strategy.pattern = e_mme_sp_reduction_fck;

            // Note that this must be last since it relies on strategy.pattern etc. being set
            bool preferNonRaster = false;
            strategy.sbReuse     = eagerSbReuseHueristic(*mmeNode, layerParams, selectedGeo, preferNonRaster);
            if (preferNonRaster)
            {
                strategy.pattern = e_mme_sp_reduction_fkc;
            }
        }
    }
    else
    {
        strategy.geometry = mmeMetaData.mmeStrategy.geometry;
        strategy.pattern  = mmeMetaData.mmeStrategy.pattern;
    }
}

void MMEBrain::chooseStrategy(const HabanaGraph& g,
                              const NodePtr      node,
                              TensorPtr          xTensor,
                              TensorPtr          wTensor,
                              TensorPtr          yTensor,
                              bool               setPipeline,
                              MmeLayerParams&    layerParams)
{
    std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    mmeNode->getMmeBrainIfc()->getRecommendedStrategy(layerParams, /*ignoreTensorAliasing*/ false, /*isGeoPreferredShort*/ true);

    if (layerParams.opType == MmeCommon::e_mme_dedw) return;  // TODO - support smart strategy selection for DeDw

    MmeCommon::MmeBrain brain(MmeCommon::e_mme_Gaudi2, mmeNode->getMmeBrainIfc()->getOperationModes());
    ParamsAndPerfsVector candidates = generatePerfAttrCandidates(layerParams, brain);
    auto                 bestParams = chooseBestCandidate(candidates, xTensor, wTensor);

    layerParams.strategy.geometry = bestParams->geometry;
    layerParams.strategy.pattern  = bestParams->pattern;
}

void MMEBrain::chooseSignalingMode(const MmeNode* mmeNode, MmeLayerParams& params)
{
    // TODO [SW-11381] Remove restriction after debugging why dedx fails if split to ROI (synapse dedx tests fail)
    if ((mmeNode->getNodeAnnotation().splitToLogicalROIs && !params.isDedxOperation()) || params.strategy.maskedBgemm)
    {
        params.controls.signalingMode = MmeCommon::e_mme_signaling_desc_with_store;
    }
    else
    {
        params.controls.signalingMode = e_mme_signaling_once;
        params.controls.squashIORois  = true;  // needed in signaling once.
    }
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

void MMEBrain::chooseRoundingMode(const MmeNode*  mmeNode,
                                  MmeLayerParams& params,
                                  synDataType     outputElementType)
{
    // Input from user - default rounding mode is round to nearest.
    RoundingMode roundingMode = synapseToMmeRoundingMode(mmeNode->getRoundingMode());

    // stochastic rounding is enabled only for a selected list of operations ("whitelist")
    if (GCFG_MME_ENABLE_STOCHASTIC_ROUNDING.value())
    {
        switch (params.opType)
        {
            case MmeCommon::e_mme_dedx:
            case MmeCommon::e_mme_transposed_dedx:
            case MmeCommon::e_mme_dedw:
            case MmeCommon::e_mme_ab:
            case MmeCommon::e_mme_atb:
            case MmeCommon::e_mme_abt:
            case MmeCommon::e_mme_atbt:

                if (outputElementType == syn_type_fp8_152 || outputElementType == syn_type_fp8_143)
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

}  // namespace gaudi2
