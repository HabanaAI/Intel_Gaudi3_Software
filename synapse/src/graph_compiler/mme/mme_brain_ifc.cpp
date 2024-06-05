#include "mme_brain_ifc.h"
#include "conv_base_node.h"
#include "defs.h"
#include "log_manager.h"
#include "mme_node.h"
#include "reductions.h"
#include "synapse_common_types.h"
#include "mme_desc_gen_utils.h"
#include "compilation_hal_reader.h"
#include "brain_conf.h"
#include <memory>
#include <optional>

MmeSolution::MmeSolution(const MmeSolution& other) : bvdMultipliers(other.bvdMultipliers)
{
    for (const auto& [mme, brainSol] : other.brainSolution)
    {
        HB_ASSERT_PTR(brainSol);
        HB_ASSERT_PTR(mme);
        brainSolution.insert({mme, std::make_shared<MmeCommon::MmeBrainSolution>(*brainSol)});
    }

    for (const auto& [mme, qor] : other.QORs)
    {
        HB_ASSERT_PTR(qor);
        HB_ASSERT_PTR(mme);
        QORs.insert({mme, std::make_shared<SolutionParams>(*qor)});
    }
}

void MmeSolution::chooseSolution() const
{
    for (auto& [node, solution] : brainSolution)
    {
        node->getNodeAnnotation().mmeMetaData.mmeStrategy = solution->strategy;
        node->getNodeAnnotation().mmeMetaData.takeStrategyFromAnnotation = true;
    }
}

MmeBrainIfc::MmeBrainIfc(const MmeNode& mmeNode, synDeviceType deviceType)
: m_mmeNode(mmeNode),
  m_chipType(getMmeChipType(deviceType)),
  m_mmeBrain(std::make_optional<MmeCommon::MmeBrain>(m_chipType.value(), getOperationModes()))
{
}

static MmeCommon::MmeLayerParams getMmeLayerBaseParamsImpl(MmeCommon::ChipType chipType, const MmeNode& mmeNode)
{
    MmeCommon::MmeLayerParams params  = MmeCommon::MmeBrain::getDefaultParams(chipType);
    params.nodeName                   = mmeNode.getNodeName();
    bool                      isDmaOp = mmeNode.isDmaOperation();
    params.opType                     = getOperationTypeCommon(chipType, mmeNode);
    TensorPtr xTensor, wTensor, yTensor, oTensor;
    getTensorRolesCommon(mmeNode, params.opType, xTensor, wTensor, yTensor, oTensor);

    const auto& hal = CompilationHalReader::getHalReader();
    params.x        = getTensorViewCommon(chipType, *xTensor, *hal, isDmaOp);
    params.w        = wTensor ? getTensorViewCommon(chipType, *wTensor, *hal, isDmaOp) : MmeCommon::MmeTensorView();
    params.y        = getTensorViewCommon(chipType, *yTensor, *hal, isDmaOp);
    normalizeTensorDims(mmeNode, chipType, params);
    MmeBrainIfc::getMmeConvParams(mmeNode, params.conv);

    return params;
}

MmeCommon::MmeLayerParams MmeBrainIfc::getMmeLayerBaseParams() const
{
    HB_ASSERT(isFullyInitialized(), "getMmeLayerBaseParams is called when Mme Brain is not fully initialized");
    return getMmeLayerBaseParamsImpl(*m_chipType, m_mmeNode);
}

// handleLogicalOps pass sets the tensor aliasing. It runs after the sram management. As a
// result, setting of strategy.alignedAddress before handleLogicalOps pass cannot refer to aliasing.
void MmeBrainIfc::setStrategyFields(MmeCommon::MmeLayerParams& params, bool ignoreTensorAliasing)
{
    HB_ASSERT(isFullyInitialized(), "setStrategyFields is called when Mme Brain is not fully initialized");

    bool setPipeline = m_mmeNode.getNodeAnnotation().splitToLogicalROIs;

    TensorPtr xTensor, wTensor, yTensor, oTensor;
    getTensorRolesCommon(m_mmeNode, params.opType, xTensor, wTensor, yTensor, oTensor);

    params.strategy.packingFactor      = m_mmeNode.getNodeAnnotation().mmeMetaData.packing[PACKING_X];
    params.strategy.pipelineLevel = setPipeline ? GCFG_DEFAULT_PIPELINE_DEPTH.value() : 1;
    params.strategy.cdConcurrencyEn    = m_mmeNode.getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn;
    params.strategy.batchConcurrencyEn = m_mmeNode.getNodeAnnotation().mmeMetaData.mmeStrategy.batchConcurrencyEn;

    params.strategy.flattenEn          = !m_mmeNode.isDynamicShape();
    params.strategy.sbReuse            = GCFG_SB_REUSE.value();
    params.strategy.partialsToMemoryEn = GCFG_MME_PARTIALS_TO_MEMORY.value();
    params.strategy.loweringEn         = GCFG_ENABLE_MME_CONV_LOWERING.value();
    params.strategy.alignedAddresses = getAlignedAddresses(&m_mmeNode, params.opType, ignoreTensorAliasing);
    params.strategy.recurringMisalignmentOptEn = GCFG_ENABLE_MME_ALIGN_OPT.value() && params.strategy.alignedAddresses;

    params.strategy.mmeLimit  = m_mmeNode.getGraphTraits()->getHalReader()->getNumMmeEngines();

    params.strategy.geometry = MmeCommon::e_mme_geometry_nr;
    params.strategy.pattern  = MmeCommon::e_mme_patterns_nr;
}

// This is the only function that access mme brain to choose the relevant strategy fields
// The function should be removed when bug TODO [SW-117781] is fixed

void MmeBrainIfc::getRecommendedStrategyFromMmeBrain(MmeCommon::MmeLayerParams& params, bool isGeoPreferredShort)
{
    HB_ASSERT(isFullyInitialized(),
              "getRecommendedStrategyFromMmeBrain is called when Mme Brain is not fully initialized");
    m_mmeBrain->getRecommendedStrategy(params, isGeoPreferredShort);
}

// Choose concurrency first, and then geometry and pattern
void MmeBrainIfc::getRecommendedStrategy(MmeCommon::MmeLayerParams& params,
                                         bool                       ignoreTensorAliasing,
                                         bool                       isGeoPreferredShort)
{
    HB_ASSERT(isFullyInitialized(), "getRecommendedStrategy is called when Mme Brain is not fully initialized");

    setStrategyFields(params, ignoreTensorAliasing);
    getRecommendedStrategyFromMmeBrain(params, isGeoPreferredShort);
}

MmeCommon::MmeLayerParams MmeBrainIfc::getRecommendedMmeLayerParams(bool isGeoPreferredShort)
{
    MmeCommon::MmeLayerParams params = getMmeLayerBaseParams();

    // Temp workaround over a pipeline manager bug
    // TODO [SW-117781]: Pipeline manager / Bundelizer produces better results when its inputs are inaccurate
    if (GCFG_MME_PARTIAL_STRATEGY_SETTING_ENABLED.value())
    {
        LOG_TRACE(GC, "Using the temporary legacy flow that sets only geometry and pattern in MME strategy");
        params.strategy.geometry           = MmeCommon::e_mme_geometry_nr;
        params.strategy.pattern            = MmeCommon::e_mme_patterns_nr;
        params.strategy.flattenEn          = !m_mmeNode.isDynamicShape();
        bool setPipeline                   = m_mmeNode.getNodeAnnotation().splitToLogicalROIs;
        params.strategy.pipelineLevel      = setPipeline ? GCFG_DEFAULT_PIPELINE_DEPTH.value() : 1;
        params.strategy.cdConcurrencyEn    = MmeCommon::TurnedOff;
        params.strategy.batchConcurrencyEn = MmeCommon::TurnedOff;

        getRecommendedStrategyFromMmeBrain(params, isGeoPreferredShort);
    }
    else
    {
        getRecommendedStrategy(params, /*ignoreTensorAliasing*/ true, isGeoPreferredShort);
    }

    return params;
}

MmeCommon::MmeLayerParams MmeBrainIfc::getRecommendedConcurrency()
{
    // Once this bug is fixed, switch to use getRecommendedMmeLayerParams and add the argument to Undef concurrency
    // TODO [SW-117781]: Pippeline manager / Bundlizer produces better results when its inputs are inaccurate
    MmeCommon::MmeLayerParams params = getMmeLayerBaseParams();

    if (!gc::reduction::datatypeValidForAccumulation(m_mmeNode.getOutput(0)->getElementType()))
    {
        // To get proper recommendation for un-reducible output data types, pretend they will be widened
        auto& out       = params.getOperand(MmeCommon::e_mme_op_c);
        out.elementType = MmeCommon::e_type_fp32;
    }

    getRecommendedStrategy(params, /*ignoreTensorAliasing*/ true, /*isGeoPreferredShort*/ true);
    return params;
}

MmeCommon::LayerSemantics MmeBrainIfc::getLayerSemantics(const MmeNode* mmeNode)
{
    using namespace MmeCommon;
    HB_ASSERT_PTR(mmeNode);

    // Layer semantics should be agnostic to HW. chipType is required only for some Gaudi1 related legacy hacks
    // (representing GEMMs using fwd ops). Using Gaudi2, as these hacks don't appear there.
    auto chipType = getMmeChipType(synDeviceGaudi2);
    auto opType   = getOperationTypeCommon(chipType, *mmeNode);

    TensorPtr x, w, y, o;
    getTensorRolesCommon(*mmeNode, opType, x, w, y, o);

    LayerSemantics lsm;
    lsm.op         = opType;
    lsm.convParams = getMmeConvParams(*mmeNode);

    lsm.operandShapes[OperandRole::X] = tensorProperties(x);
    lsm.operandShapes[OperandRole::W] = tensorProperties(w);
    lsm.operandShapes[OperandRole::Y] = tensorProperties(y);
    if (o)
    {
        lsm.operandShapes[OperandRole::OUTPUT_COPY] = tensorProperties(o);
    }
    if (mmeNode->hasBias())
    {
        const auto& bias                     = mmeNode->getInput(TENSOR_BIAS);
        lsm.operandShapes[OperandRole::BIAS] = tensorProperties(bias);
    }
    if (mmeNode->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        const auto& maskA                      = mmeNode->getInput(TENSOR_AUX_BGEMM_MASK_A);
        const auto& maskB                      = mmeNode->getInput(TENSOR_AUX_BGEMM_MASK_B);
        lsm.operandShapes[OperandRole::MASK_A] = tensorProperties(maskA);
        lsm.operandShapes[OperandRole::MASK_B] = tensorProperties(maskB);
    }
    if (auto* convNode = dynamic_cast<const ConvBaseNode*>(mmeNode))
    {
        if (auto shape = convNode->getShapeOperand())
        {
            lsm.operandShapes[OperandRole::SHAPE] = tensorProperties(shape);
        }
    }

    return lsm;
}

MmeCommon::LayerSemantics::TensorProperties MmeBrainIfc::tensorProperties(const TensorPtr& t)
{
    using Shape        = MmeCommon::LayerSemantics::TensorProperties::TensorShape;
    const auto& tShape = t->getAllNSizesInElements();
    return {Shape(tShape.begin(), std::next(tShape.begin(), t->getDim()))};
}

static std::vector<std::pair<Dim, unsigned>> getParallelizedDims(const MmeNode*                  mmeNode,
                                                                 const MmeCommon::AccessPattern& accessPattern)
{
    std::vector<std::pair<Dim, unsigned>> parallelizedDims;

    const auto& dcoreRois = mmeNode->getNodeAnnotation().m_dcoreROIs;
    if (dcoreRois.size() > 1)
    {
        for (Dim idxSpcDim = 0; idxSpcDim < accessPattern.indexSpace.size(); ++idxSpcDim)
        {
            if (dcoreRois.at(0).baseOffset[idxSpcDim] != dcoreRois.at(1).baseOffset[idxSpcDim])
            {
                // perforation on idxSpcDim
                const auto& outputAP = accessPattern.operandAccessPatterns.at(accessPattern.roleC);
                if (std::find_if(outputAP.dimsAccessPattern.begin(),
                                 outputAP.dimsAccessPattern.end(),
                                 [&](const auto& dimAP) { return dimAP.indexSpaceDim == idxSpcDim; }) ==
                    outputAP.dimsAccessPattern.end())
                {
                    // No output dimension is mapped to idxSpcDim ==> idxSpcDim is a CD
                    parallelizedDims.push_back({idxSpcDim, dcoreRois.size()});
                }
            }
        }
    }

    return parallelizedDims;
}

MmeCommon::AccessPattern MmeBrainIfc::generateAccessPattern(const MmeNode* mmeNode)
{
    MmeCommon::AccessPattern accessPattern;
    auto opType = getOperationTypeCommon(MmeCommon::ChipType::e_mme_Gaudi2, *mmeNode);
    switch (opType)
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_dedw:
        case MmeCommon::e_mme_deterministic_dedw:
        case MmeCommon::e_mme_transposed_dedx:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
        {
            auto semanticParams = getLayerSemantics(mmeNode);
            accessPattern       = MmeCommon::AccessPatternFactory::createFrom(&semanticParams);
            if (mmeNode->getNumInputs() > 2)  // Apply for cases with aux-tensors only until SW-166649 is fixed
            {
                for (const auto& [dim, level] : getParallelizedDims(mmeNode, accessPattern))
                {
                    MmeCommon::AccessPatternFactory::applyParallelism(&accessPattern, dim, level);
                }
            }
            break;
        }
        default:
            HB_ASSERT(false, "Unsupported operation type for access pattern: {}", opType);
    }
    return accessPattern;
}

void MmeBrainIfc::setRecommendedConcurrency()
{
    // Todo: extend mme support for selection of concurrencies for all nodes (not only dedw)
    // SW-125186: Extend concurrency 3-value batch concurrency setting for bgemm
    // workaround until this support is added
    auto& mmeMetaData =  m_mmeNode.getNodeAnnotation().mmeMetaData;
    if (m_mmeNode.getNodeType() != Node::TYPE_DEDW)
    {
        mmeMetaData.mmeStrategy.batchConcurrencyEn = MmeCommon::TurnedOn;
        mmeMetaData.mmeStrategy.cdConcurrencyEn    = MmeCommon::TurnedOff;
        return;
    }
    // If none of the concurrency annotations is Undefined, there is nothing to choose
    if (mmeMetaData.mmeStrategy.batchConcurrencyEn != MmeCommon::Undefined &&
        mmeMetaData.mmeStrategy.cdConcurrencyEn    != MmeCommon::Undefined)
    {
        return;
    }

    // Let mme brain choose the concurrencies
    MmeCommon::MmeLayerParams params = getRecommendedConcurrency();
    LOG_DEBUG(MME_STACK,
              "Recommended concurrency for node {}:"
              " CD: {}"
              " Batch: {}",
              m_mmeNode.getNodeName(),
              params.strategy.cdConcurrencyEn,
              params.strategy.batchConcurrencyEn);

    // Update the node annotations based on the provided parameters. The crucial information we require from "params"
    // is whether CDC/batching is enabled or not. Additionally, the "strategy" object already includes other relevant
    // details such as geometry and walking pattern. Instead of redundant calculations, the Eager MME brain can utilize
    // this information directly. Therefore, we assign the entire "strategy" struct for comprehensive data integration.
    mmeMetaData.mmeStrategy = params.strategy;
}

bool MmeBrainIfc::opSupportsChoosingCdConcurrency()
{
    return m_mmeBrain->opSupportsChoosingConcurrency(getOperationTypeCommon(*m_chipType, m_mmeNode));
}

MmeCommon::PerfAttr MmeBrainIfc::getRecommendedConfigMmePerf()
{
    MmeCommon::MmeLayerParams params = getRecommendedMmeLayerParams();
    return getMmePerfFromParams(params);
}

unsigned MmeBrainIfc::getRecommendedGeometryConcurrency()
{
    const MmeCommon::MmeLayerParams mmeParams = getRecommendedMmeLayerParams();
    const MmeCommon::ChipType chipType = getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType());
    return MmeCommon::MmeBrain::getGeometryConcurrency(chipType, mmeParams) * m_mmeBrain->getFlatteningFactor();
}

void MmeBrainIfc::getMmeConvParams(const MmeNode& mmeNode, MmeCommon::MmeConv& conv)
{
    auto convParams = getMmeConvParams(mmeNode);
    if (convParams)
    {
        conv = *convParams;
    }
}

std::optional<MmeCommon::MmeConv> MmeBrainIfc::getMmeConvParams(const MmeNode& mmeNode)
{
    std::optional<MmeCommon::MmeConv> result;
    if (mmeNode.isConvolution())
    {
        MmeCommon::MmeConv& conv = result.emplace();
        const auto&        convNode           = static_cast<const ConvBaseNode&>(mmeNode);
        const auto&        convParams         = convNode.getConvolutionParams();
        conv.stride[CONV_STRIDE_WIDTH]        = convParams.stride[CONV_STRIDE_WIDTH];
        conv.stride[CONV_STRIDE_HEIGHT]       = convParams.stride[CONV_STRIDE_HEIGHT];
        conv.stride[CONV_STRIDE_DEPTH]        = convParams.stride[CONV_STRIDE_DEPTH];
        conv.padding[CONV_PAD_LEFT]           = convParams.padding[CONV_PAD_LEFT];
        conv.padding[CONV_PAD_TOP - 1]        = convParams.padding[CONV_PAD_TOP];
        conv.padding[CONV_PAD_FRONT - 2]      = convParams.padding[CONV_PAD_FRONT];
        conv.dilation[CONV_DIL_WIDTH]         = convParams.dilation[CONV_DIL_WIDTH];
        conv.dilation[CONV_DIL_HEIGHT]        = convParams.dilation[CONV_DIL_HEIGHT];
        conv.dilation[CONV_DIL_DEPTH]         = convParams.dilation[CONV_DIL_DEPTH];
        conv.spatialDimsNr                    = convNode.is3DConvolution() ? 3 : 2;
    }
    return result;
}

bool MmeBrainIfc::isCdDim(unsigned dim, const MmeCommon::MmeLayerParams& params)
{
    std::vector<unsigned int> cdDims;
    m_mmeBrain->getCdDim(params, cdDims);

    return std::find(cdDims.begin(), cdDims.end(), dim) != cdDims.end();
}

MmeCommon::ChipType MmeBrainIfc::getMmeChipType(const synDeviceType deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return MmeCommon::ChipType::e_mme_Gaudi;
            break;
        case synDeviceGaudi2:
            return MmeCommon::ChipType::e_mme_Gaudi2;
            break;
        case synDeviceGaudi3:
            return MmeCommon::ChipType::e_mme_Gaudi3;
            break;
        default:
            HB_ASSERT(0, "invalid device type");
    }
    return MmeCommon::ChipType::e_mme_Gaudi;
}
MmeCommon::MmeBrainOperationModes MmeBrainIfc::getOperationModes()
{
    MmeCommon::MmeBrainOperationModes brainOperationModes;
    brainOperationModes.addAlignmentPenaltyCalc            = GCFG_ADD_ALIGNMENT_PENALTY_MME_BRAIN.value();
    brainOperationModes.addTieBreakerPreferredReuseOperand = GCFG_TIE_BRAEKER_PREFERRED_REUSE_OPERAND_MME_BRAIN.value();
    brainOperationModes.addOptimizationToLBSolutions = GCFG_ENABLE_LB_MME_CONCURRENCY_OPT.value();
    return brainOperationModes;
}

// Calculate MME perfAttr from params directly
MmeCommon::PerfAttr MmeBrainIfc::getMmePerfFromParams(const MmeCommon::MmeLayerParams params)
{
    MmeCommon::MmeBrain brain(getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType()),
                              getOperationModes());
    MmeCommon::PerfAttr perfAttr;
    brain.getPerfAttr(params, perfAttr);
    return perfAttr;
}

// Return dummy MME memcpy params from given sizes and strides
MmeCommon::MmeLayerParams MmeBrainIfc::getMmeMemcpyLayerParams(const SizeArray sizes, const StrideArray strides)
{
    // TODO: complete implementation [SW-116267]
    HB_ASSERT(false, "Not implemented yet - shouldn't get here");
    auto chipType = getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType());
    return MmeCommon::MmeBrain::getDefaultParams(chipType);
}

void convertOptionalToBvds(const NodePtr&                node,
                           const BundleViewContainerPtr& bundleViews,
                           std::optional<unsigned>&      optionalNodeDim,
                           std::optional<unsigned>&      optionalBvd)
{
    // MME stack may list node dimensions that are not mapped from any tensor dimension. These node dims will not be
    // mapped to a BVD either (E.g. BATCH_0 dim for GEMM operation, which the MME stack treats as degenerate bgemm).
    if (optionalNodeDim.has_value() && bundleViews->isNodeDimMappedToBVD(node, *optionalNodeDim))
    {
        optionalBvd.emplace(bundleViews->getBVDForNodeDim(node, optionalNodeDim.value()));
    }
}
void convertVectorToBvds(const NodePtr&                node,
                         const BundleViewContainerPtr& bundleViews,
                         std::vector<unsigned>&        nodeDimVec,
                         std::vector<unsigned>&        bvdVec)
{
    bvdVec.clear();  // In case they were filled with node-dims before
    for (unsigned nodeDim : nodeDimVec)
    {
        // MME stack may list node dimensions that are not mapped from any tensor dimension. These node dims will not be
        // mapped to a BVD either (E.g. DEPTH dim is listed in a convolution spatial dimensions, even if the convolution
        // is 2Dimensional).
        if (!bundleViews->isNodeDimMappedToBVD(node, nodeDim)) continue;
        bvdVec.push_back(bundleViews->getBVDForNodeDim(node, nodeDim));
    }
}

static void convertToBvds(const NodePtr&                              node,
                          const BundleViewContainerPtr&               bundleViews,
                          MmeCommon::MmeBrainSolutionPtr              mmeBrainSolution,
                          SolutionParamsPtr                           solutionParams,
                          std::unordered_map<BundleViewId, uint64_t>& bvdMultipliers)
{
    const auto& accessPattern = node->getNodeAccessPattern();
    int         nodeDimsNr    = accessPattern->getNodeResolution().size();
    for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
    {
        if (bundleViews->isNodeDimMappedToBVD(node, nodeDim))
        {
            int bvd             = bundleViews->getBVDForNodeDim(node, nodeDim);
            bvdMultipliers[bvd] = mmeBrainSolution->solutionDimMultipliers[nodeDim];
        }
    }

    convertVectorToBvds(node,
                        bundleViews,
                        mmeBrainSolution->requirements.perforationDimVec,
                        solutionParams->solutionRequirements.perforationDimVec);
    convertOptionalToBvds(node,
                          bundleViews,
                          mmeBrainSolution->requirements.bwInflationDim,
                          solutionParams->solutionRequirements.bwInflationDim);
    convertVectorToBvds(node,
                        bundleViews,
                        mmeBrainSolution->requirements.utilizationInflationDims,
                        solutionParams->solutionRequirements.utilizationInflationDims);
    convertVectorToBvds(node,
                        bundleViews,
                        mmeBrainSolution->requirements.walkDims,
                        solutionParams->solutionRequirements.walkDims);
    convertVectorToBvds(node,
                        bundleViews,
                        mmeBrainSolution->requirements.cdDims,
                        solutionParams->solutionRequirements.cdDims);
}

static MmeSolutionPtr convertBrainSolution(const NodePtr&                 node,
                                           const BundleViewContainerPtr&  bundleViews,
                                           const MmeSolutionPtr           prevSolution,
                                           MmeCommon::MmeBrainSolutionPtr brainSolution)
{
    MmeSolutionPtr solution =
        prevSolution ? std::make_shared<MmeSolution>(*prevSolution) : std::make_shared<MmeSolution>();
    SolutionParamsPtr solutionParams     = std::make_shared<SolutionParams>();
    solutionParams->perfAttr             = brainSolution->perfAttr;
    solutionParams->solutionRequirements = brainSolution->requirements;
    convertToBvds(node, bundleViews, brainSolution, solutionParams, solution->bvdMultipliers);
    solution->QORs[node]          = solutionParams;
    solution->brainSolution[node] = brainSolution;
    return solution;
}

static void updateQORs(MmeCommon::MmeBrain&          brain,
                       MmeCommon::ChipType           chipType,
                       const BundleViewContainerPtr& bundleViews,
                       const MmeSolutionPtr          prevSolution,
                       MmeSolutionPtr&               mmeSolution)
{
    for (const auto& [mme, qor] : prevSolution->QORs)
    {
        const auto& accessPattern = mme->getNodeAccessPattern();
        int         nodeDimsNr    = accessPattern->getNodeResolution().size();

        MmeCommon::MultiplierArray commonGranularity(nodeDimsNr);
        MmeCommon::MultiplierArray prevMultipliers = prevSolution->brainSolution[mme]->solutionDimMultipliers;
        bool                       recalcQOR       = false;
        for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
        {
            if (bundleViews->isNodeDimMappedToBVD(mme, nodeDim))
            {
                commonGranularity[nodeDim] = bundleViews->getGranularityForNodeDim(mme, nodeDim);

                BundleViewId bvd = bundleViews->getBVDForNodeDim(mme, nodeDim);
                if (prevMultipliers[nodeDim] != mmeSolution->bvdMultipliers[bvd])
                {
                    // a bvd multiplier was updated that is shared with the current node, update it and recalculate QOR
                    recalcQOR                = true;
                    prevMultipliers[nodeDim] = mmeSolution->bvdMultipliers[bvd];
                }
            }
            else
            {
                commonGranularity[nodeDim] = accessPattern->getNodeResolution()[nodeDim];
            }
        }

        if (recalcQOR)
        {
            MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(mme);
            auto       params  = getMmeLayerBaseParamsImpl(chipType, *mmeNode);
            params.strategy = prevSolution->brainSolution[mme]->strategy;
            auto sliceParams   = params;
            brain.setParamsToSolutionSize(sliceParams, prevMultipliers, commonGranularity);

            brain.getPerfAttr(params, mmeSolution->QORs[mme]->perfAttr, sliceParams);
            // perfAttr was updated due to the multipliers change, make sure the relevant brain solution is also updated
            mmeSolution->brainSolution[mme]->perfAttr = mmeSolution->QORs[mme]->perfAttr;
            mmeSolution->brainSolution[mme]->solutionDimMultipliers = prevMultipliers;
        }
    }
}

MmeSolutionContainer MmeBrainIfc::generateLayeredBrainStrategies(const NodePtr&                node,
                                                                 const BundleViewContainerPtr& bundleViews,
                                                                 const MmeSolutionContainer&   previousSolutions)
{
    const auto& accessPattern = node->getNodeAccessPattern();
    int         nodeDimsNr    = accessPattern->getNodeResolution().size();
    MmeNode*    mmeNode       = dynamic_cast<MmeNode*>(node.get());
    HB_ASSERT_PTR(mmeNode);
    auto params = getMmeLayerBaseParams();
    setStrategyFields(params);
    // CD\Batch concurrency are currently not supproted in MME generateSolution.
    // need to remove this once supported - SW-156770
    params.strategy.batchConcurrencyEn = MmeCommon::TurnedOff;
    params.strategy.cdConcurrencyEn = MmeCommon::TurnedOff;


    MmeCommon::MultiplierArray commonGranularity(nodeDimsNr), previousMultipliers(nodeDimsNr);
    for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
    {
        previousMultipliers[nodeDim] = 1;
        if (bundleViews->isNodeDimMappedToBVD(node, nodeDim))
        {
            commonGranularity[nodeDim] = bundleViews->getGranularityForNodeDim(node, nodeDim);
        }
        else
        {
            commonGranularity[nodeDim] = accessPattern->getNodeResolution()[nodeDim];
        }
    }

    MmeCommon::MmeBrain  brain(getMmeChipType(CompilationHalReader::getHalReader()->getDeviceType()),
                               getOperationModes());
    MmeSolutionContainer solutions;

    if (previousSolutions.empty())
    {
        MmeCommon::MmeBrainSolutionContainer brainSolutions =
            brain.getMmeSolutions(params, commonGranularity, previousMultipliers, GCFG_ENABLE_CD_PARALLEL.value());
        for (auto& brainSolution : brainSolutions)
        {
            solutions.push_back(convertBrainSolution(node, bundleViews, MmeSolutionPtr(), brainSolution));
        }
    }
    else
    {
        for (auto& prevSolution : previousSolutions)
        {
            std::fill(previousMultipliers.begin(), previousMultipliers.end(), 1);
            for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
            {
                if (bundleViews->isNodeDimMappedToBVD(node, nodeDim))
                {
                    BundleViewId bvd = bundleViews->getBVDForNodeDim(node, nodeDim);
                    if (prevSolution->bvdMultipliers.find(bvd) != prevSolution->bvdMultipliers.end())
                    {
                        previousMultipliers[nodeDim] = prevSolution->bvdMultipliers.at(bvd);
                    }
                }
            }

            MmeCommon::MmeBrainSolutionContainer brainSolutions =
                brain.getMmeSolutions(params, commonGranularity, previousMultipliers, GCFG_ENABLE_CD_PARALLEL.value());

            for (auto& brainSolution : brainSolutions)
            {
                for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
                {
                    unsigned multiplier = brainSolution->solutionDimMultipliers[nodeDim];
                    if (multiplier % previousMultipliers[nodeDim] != 0)
                    {
                        // if the multiplier is not a full multiple of a previous solution, it will only be valid
                        // if it represents the whole shape.
                        if (bundleViews->isNodeDimMappedToBVD(node, nodeDim))
                        {
                            HB_ASSERT(multiplier * commonGranularity[nodeDim] >=
                                              accessPattern->getNodeResolution()[nodeDim] &&
                                          multiplier * commonGranularity[nodeDim] <
                                              accessPattern->getNodeResolution()[nodeDim] + commonGranularity[nodeDim],
                                      "solution is not a multiple of previous solution, nor was it expanded to be unsliced");
                        }
                    }
                }
                // here we also need to update each node in the previous solutions QOR with the updated multipliers
                MmeSolutionPtr mmeSolution = convertBrainSolution(node, bundleViews, prevSolution, brainSolution);
                updateQORs(brain, *m_chipType, bundleViews, prevSolution, mmeSolution);
                solutions.push_back(mmeSolution);
            }
        }
    }

    return solutions;
}

MmeSolutionPtr MmeBrainIfc::inflateForUtilization(const MmeSolutionPtr&         solutionToInflate,
                                                  const NodePtr&                nodeToInflate,
                                                  const BundleViewContainerPtr& bundleViews,
                                                  const std::optional<float>&   utilizationThreshold)
{
    HB_ASSERT(solutionToInflate->brainSolution.count(nodeToInflate) != 0, "node to inflate is not part of the solution");

    const auto& accessPattern = nodeToInflate->getNodeAccessPattern();
    int         nodeDimsNr    = accessPattern->getNodeResolution().size();
    MmeNode*    mmeNode       = dynamic_cast<MmeNode*>(nodeToInflate.get());
    HB_ASSERT_PTR(mmeNode);
    const auto& brainSolution = solutionToInflate->brainSolution.at(nodeToInflate);
    auto params     = getMmeLayerBaseParams();
    params.strategy = brainSolution->strategy;

    MmeCommon::MultiplierArray commonGranularity(nodeDimsNr);
    for (int nodeDim = 0; nodeDim < nodeDimsNr; nodeDim++)
    {
        if (bundleViews->isNodeDimMappedToBVD(nodeToInflate, nodeDim))
        {
            commonGranularity[nodeDim] = bundleViews->getGranularityForNodeDim(nodeToInflate, nodeDim);
        }
        else
        {
            commonGranularity[nodeDim] = accessPattern->getNodeResolution()[nodeDim];
        }
    }

    // currently we only support inflating over height aspect, in the future logic will be added to determine on which
    // aspect to inflate
    auto newSol = m_mmeBrain->inflateForUtilization(params,
                                                    brainSolution,
                                                    commonGranularity,
                                                    MmeCommon::PhysicalAspects::Name::OUTPUT_HEIGHT,
                                                    utilizationThreshold);
    if (newSol)
    {  // here we also need to update each node in the previous solutions QOR with the updated multipliers
        MmeSolutionPtr mmeSolution = convertBrainSolution(nodeToInflate, bundleViews, solutionToInflate, newSol);
        updateQORs(*m_mmeBrain, *m_chipType, bundleViews, solutionToInflate, mmeSolution);
        return mmeSolution;
    }

    return nullptr;
}

bool MmeBrainIfc::isCdDim(unsigned int dim)
{
    auto                  params = getMmeLayerBaseParams();
    return isCdDim(dim, params);
}

std::vector<unsigned int> MmeBrainIfc::getCDDims()
{
    auto                  params = getMmeLayerBaseParams();
    std::vector<unsigned> cdDims;
    m_mmeBrain->getCdDim(params, cdDims);
    return cdDims;
}
