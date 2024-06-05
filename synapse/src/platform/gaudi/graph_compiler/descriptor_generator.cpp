#include "gaudi/gaudi_tpc_descriptor.h"

#include "convolution_node.h"
#include "include/mme_common/mme_common_enum.h"
#include "node.h"
#include "synapse_common_types.h"
#include "transpose_node.h"
#include "dedx_node.h"
#include "dedw_node.h"
#include "gaudi_graph.h"
#include "node_utils.h"
#include "descriptor_generator.h"
#include "habana_global_conf.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "data_type_utils.h"
#include "include/mme_common/mme_brain.h"
#include "tpc_slice_desc_update.h"
#include "mme_desc_gen_utils.h"
#include "mme_brain_ifc.h"
#include "gaudi_code_generator.h"
#include "mme_logger.h"

#include <bitset>
#include <cstddef>
#include <list>
#include <memory>
#include <numeric>
#include <string_view>
#include <string>

namespace gaudi
{

static MmeCommon::EMmeGeometry getGeometryOfMinActivations(uint64_t width, uint64_t height, unsigned mmeVectorElems)
{
    Settable<uint32_t> minActivations;
    MmeCommon::EMmeGeometry minGeometry = MmeCommon::e_mme_geometry_2wx2h;
    for (auto geometry :
         {MmeCommon::e_mme_geometry_2wx2h, MmeCommon::e_mme_geometry_4wx1h, MmeCommon::e_mme_geometry_1wx4h})
    {
        uint32_t mmeHeight = mmeVectorElems;
        uint32_t mmeWidth  = mmeVectorElems;
        switch (geometry)
        {
            case MmeCommon::e_mme_geometry_1wx4h:
                mmeHeight *= 4;
                break;
            case MmeCommon::e_mme_geometry_4wx1h:
                mmeWidth *= 4;
                break;
            case MmeCommon::e_mme_geometry_2wx2h:
                mmeHeight *= 2;
                mmeWidth *= 2;
                break;
            default:
                HB_ASSERT(false, "Unsupported MME geometry");
        }
        uint32_t currentActivations = std::ceil((double)height / mmeHeight) * std::ceil((double)width / mmeWidth);
        if (!minActivations.is_set() || (currentActivations < minActivations.value()))
        {
            minActivations = currentActivations;
            minGeometry = geometry;
        }
    }
    return minGeometry;
}

static MmeCommon::MmeStrategy getMmeStrategyFwdDedx(const MmeNode&     node,
                                                    synDataType        inputAElementType,
                                                    const SizeArray&   outputSizes,
                                                    const std::string& nodeNameForLog,
                                                    unsigned           packingFactor)
{
    MmeCommon::MmeStrategy strategy = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi).strategy;
    strategy.pattern    = MmeCommon::e_mme_z_reduction_skf;
    strategy.loweringEn = true;
    strategy.dedxDynamicPadding     = node.isDynamicPaddingConvolution();

    unsigned elementSize = dataTypeSizeInBytes(inputAElementType);
    uint64_t width  = outputSizes[DIM_C];
    uint64_t height = outputSizes[DIM_B_FOR_5D_TENSOR] *
                      outputSizes[DIM_B] *
                      outputSizes[DIM_H] *
                      outputSizes[DIM_W];

    unsigned mmeVectorElems = CompilationHalReader::getHalReader()->getMmeVectorSize() / elementSize;

    //Todo: need to update when ROIs are involved
    uint64_t widthInMMEs  = div_round_up(width, mmeVectorElems);
    uint64_t heightInMMes = div_round_up(height, mmeVectorElems); //MME is symmetrical

    if (widthInMMEs * heightInMMes < 4)
    {
        LOG_DEBUG(MME_STACK, "Warning: node {} has small output spatial size and width", nodeNameForLog);
    }

    if (((widthInMMEs == 1) && (heightInMMes > 4)) ||
        ((widthInMMEs == 2) && (heightInMMes > 2)) ||
        ((widthInMMEs == 4) && (heightInMMes > 1)))
    {
        strategy.pattern = MmeCommon::e_mme_z_reduction_ksf;
    }

    strategy.geometry = getGeometryOfMinActivations( width, height, mmeVectorElems);
    MmeCommon::EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    strategy.recurringMisalignmentOptEn =
        (opType == MmeCommon::e_mme_fwd && GCFG_ENABLE_MME_ALIGN_OPT.value() && strategy.alignedAddresses);
    strategy.flattenEn           = !node.isDynamicShape();
    strategy.sbReuse             = GCFG_SB_REUSE.value();
    strategy.packingFactor       = packingFactor;
    return strategy;
}

static MmeCommon::MmeStrategy getMmeStrategyFwdDedx(const MmeNode& node)
{
    HB_ASSERT(node.getNumOutputs() == 1, "Expected single output operand for MME op");
    synDataType inputAElementType = node.getInput(0)->getElementType();
    SizeArray outputSizes = node.getOutput(0)->getAllSizesInElements();
    unsigned    packingFactor     = node.getNodeAnnotation().mmeMetaData.packing[PACKING_X];
    return getMmeStrategyFwdDedx(node, inputAElementType, outputSizes, node.getNodeName(), packingFactor);
}

static MmeCommon::MmeStrategy getMmeStrategyBGemm(synDataType inputAElementType, const SizeArray& outputSizes)
{
    // TODO update when additional bgemm capabilities are introduced
    uint64_t outputWidth    = outputSizes[DIM_C];
    uint64_t outputHeight   = outputSizes[DIM_W];
    unsigned elementSize    = dataTypeSizeInBytes(inputAElementType);
    unsigned mmeVectorElems = CompilationHalReader::getHalReader()->getMmeVectorSize() / elementSize;

    MmeCommon::MmeStrategy strategy = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi).strategy;

    strategy.geometry = getGeometryOfMinActivations( outputWidth, outputHeight, mmeVectorElems);
    strategy.pattern    = MmeCommon::e_mme_sp_reduction_fck;
    strategy.loweringEn = false;
    strategy.sbReuse    = GCFG_SB_REUSE.value();
    strategy.unrollEn = true;
    return strategy;
}

static MmeCommon::MmeStrategy getMmeStrategyBGemm(const MmeNode& node)
{
    HB_ASSERT(node.getNumOutputs() == 1, "Expected single output operand for MME op");
    synDataType inputAElementType = node.getInput(0)->getElementType();
    SizeArray outputSizes = node.getOutput(0)->getAllSizesInElements();

    return getMmeStrategyBGemm(inputAElementType, outputSizes);
}

static MmeCommon::MmeLayerParams setMmeLayerParamsFromNode(const MmeNode& node)
{
    const std::string& nodeName = node.getNodeName();
    SET_TEMP_LOG_CONTEXT(nodeName);

    MmeCommon::MmeLayerParams params = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi);
    params.nodeName                  = nodeName;

    params.controls.signalingMode = MmeCommon::e_mme_signaling_once;
    params.strategy.geometry      = MmeCommon::e_mme_geometry_2wx2h;
    params.strategy.pattern       = MmeCommon::e_mme_sp_reduction_kfc;

    const MmeCommon::EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    pTensor                     xTensor;
    pTensor                     wTensor;
    pTensor                     yTensor;
    pTensor                     oTensor;
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);

    const auto& hal = CompilationHalReader::getHalReader();
    params.x = getTensorViewCommon(MmeCommon::e_mme_Gaudi, *xTensor, *hal, false);
    params.w = getTensorViewCommon(MmeCommon::e_mme_Gaudi, *wTensor, *hal, false);
    params.y = getTensorViewCommon(MmeCommon::e_mme_Gaudi, *yTensor, *hal, false);

    if (opType == MmeCommon::e_mme_dedx)
    {
        params.spSize = multiplyElements(params.x.sizes.data() + 1, params.x.sizes.data() + Mme::c_mme_max_tensor_dims);
    }
    else
    {
        params.spSize = multiplyElements(params.y.sizes.data() + 1, params.y.sizes.data() + Mme::c_mme_max_tensor_dims);
    }

    params.opType = opType;
    MmeBrainIfc::getMmeConvParams(node, params.conv);

    return params;
}

static MmeCommon::MmeStrategy getMmeStrategyDedw(const MmeNode&   node,
                                                 synDataType      inputAElementType,
                                                 const SizeArray& inputASizes,
                                                 const SizeArray& outputSizes)
{
    // check if all the conditions for unroll are met
    MmeCommon::MmeLayerParams params = setMmeLayerParamsFromNode(node);
    params.strategy.pattern          = MmeCommon::e_mme_sp_reduction_fck;
    // 4w1h is the only geo supported by unroll
    params.strategy.geometry   = MmeCommon::e_mme_geometry_4wx1h;
    params.strategy.loweringEn = true;
    params.strategy.sbReuse    = false;
    MmeCommon::MmeBrain mmeBrain(MmeCommon::e_mme_Gaudi, node.getMmeBrainIfc()->getOperationModes());
    if (mmeBrain.gaudiShouldUnroll(params))
    {
        return params.strategy;
    }

    // choose a strategy according to output/input and MME sizes
    params.strategy.pattern    = MmeCommon::e_mme_sp_reduction_fck;
    params.strategy.geometry   = MmeCommon::e_mme_geometry_1wx4h;
    params.strategy.loweringEn = false;
    params.strategy.sbReuse    = GCFG_SB_REUSE.value();
    unsigned elementSize = dataTypeSizeInBytes(inputAElementType);
    uint64_t width  = outputSizes[WEIGHT_DIM_K];
    uint64_t height = outputSizes[WEIGHT_DIM_C];

    unsigned mmeVectorElems = CompilationHalReader::getHalReader()->getMmeVectorSize() / elementSize;

    //Todo: need to update when ROIs are involved
    uint64_t widthInMMEs  = div_round_up(width, mmeVectorElems);
    uint64_t heightInMMes = div_round_up(height, mmeVectorElems); //MME is symmetrical

    if ((widthInMMEs * heightInMMes < 4) && (outputSizes[WEIGHT_DIM_S] > 1))
    {
        params.strategy.loweringEn = true;
        params.strategy.sbReuse    = false;
        height *= outputSizes[WEIGHT_DIM_S];
        heightInMMes = div_round_up(height, mmeVectorElems); //MME is symmetrical
    }

    uint64_t spatialSize = inputASizes[DIM_W] *
                           inputASizes[DIM_H] *
                           inputASizes[DIM_D_FOR_5D_TENSOR] *
                           inputASizes[DIM_B_FOR_5D_TENSOR];

    if (node.getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn == MmeCommon::TurnedOn)
    {
        params.strategy.dedwAsBgemmEn = true;
        params.strategy.geometry      = MmeCommon::e_mme_geometry_2wx2h;
        params.strategy.pattern       = MmeCommon::e_mme_sp_reduction_fck;  // does not really matter
        params.strategy.sbReuse       = false;                              // does not really matter
    }
    else if ((spatialSize <= 512) && (widthInMMEs >= 4))
    {
        params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_kfc;
        params.strategy.geometry = MmeCommon::e_mme_geometry_4wx1h;
    }
    else if ((widthInMMEs == 1) && (heightInMMes == 1) && (outputSizes[WEIGHT_DIM_S] > 1))
    {
        // weight untoll
        params.strategy.geometry = MmeCommon::e_mme_geometry_4wx1h;
        params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_fck;
    }
    else if (widthInMMEs >= heightInMMes)
    {
        if (heightInMMes >= 4)
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_1wx4h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_fck;
        }
        else if (heightInMMes >= 2)
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_2wx2h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_fck;
        }
        else
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_4wx1h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_kfc;
        }
    }
    else
    {
        if (widthInMMEs >= 4)
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_4wx1h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_kfc;
        }
        else if (widthInMMEs >= 2)
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_2wx2h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_kfc;
        }
        else
        {
            params.strategy.geometry = MmeCommon::e_mme_geometry_1wx4h;
            params.strategy.pattern  = MmeCommon::e_mme_sp_reduction_fck;
        }
    }

    return params.strategy;
}

static MmeCommon::MmeStrategy getMmeStrategyDedw(const MmeNode& node)
{
    HB_ASSERT(node.getNumOutputs() == 1, "Expected single output operand for MME op");
    synDataType inputAElementType = node.getInput(0)->getElementType();
    SizeArray inputASizes = node.getInput(TENSOR_DEDY)->getAllSizesInElements();
    SizeArray outputSizes = node.getOutput(0)->getAllSizesInElements();

    return getMmeStrategyDedw(node, inputAElementType, inputASizes, outputSizes);
}

MmeCommon::MmeStrategy DescriptorGenerator::getMmeStrategy(const MmeNode& node)
{
    //Todo: should be a pass eventually
    MmeCommon::EMmeOpType  opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    MmeCommon::MmeStrategy strategy = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi).strategy;

    switch ( opType )
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_dedx:
            strategy = getMmeStrategyFwdDedx(node);
            break;
        case MmeCommon::e_mme_dedw:
            strategy = getMmeStrategyDedw(node);
            break;
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            strategy = getMmeStrategyBGemm(node);
            break;
        default:
            HB_ASSERT( false, "Unsupported operation type");
    }

    return strategy;
}

MmeCommon::MmeStrategy DescriptorGenerator::getMmeStrategy(const MmeNode&        node,
                                                           MmeCommon::EMmeOpType operationType,
                                                           synDataType           inputAElementType,
                                                           const SizeArray&      inputASizes,
                                                           const SizeArray&      outputSizes,
                                                           const std::string&    nodeNameForLog,
                                                           unsigned              packingFactor)
{
    MmeCommon::MmeStrategy strategy = MmeCommon::MmeBrain::getDefaultParams(MmeCommon::e_mme_Gaudi).strategy;

    switch ( operationType )
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_dedx:
            strategy = getMmeStrategyFwdDedx(node, inputAElementType, outputSizes, nodeNameForLog, packingFactor);
            break;
        case MmeCommon::e_mme_dedw:
            strategy = getMmeStrategyDedw(node, inputAElementType, inputASizes, outputSizes);
            break;
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_atbt:
            strategy = getMmeStrategyBGemm(inputAElementType, outputSizes);
            break;
        default:
            HB_ASSERT(false, "Unsupported operation type");
    }

    return strategy;
}


static bool getMmeSramReduction(const MmeNode& node)
{
    bool isAtomicAdd = false;
    pTensor outTensor = node.getOutput(TENSOR_OFM);
    ReductionInfo reductionInfoMme = outTensor->getRealReductionInfo();

    ///TODO - Add support for SUB, MIN, MAX reduction atomic operations
    if (outTensor->isReductionEnabled())
    {
        switch (reductionInfoMme.reductionOperation)
        {
            case REDUCTION_ADD:
                isAtomicAdd = true;
                break;
            default:
                HB_ASSERT(false, "Unsupported Gaudi MME reduction operation {}", reductionInfoMme.reductionOperation);
                break;
        }
    }

    return isAtomicAdd;
}

DescriptorGenerator::DescriptorGenerator(GaudiCodeGenerator* codeGenerator) : m_codeGenerator(codeGenerator) {}

void DescriptorGenerator::visit(MmeNode* node)
{
    HB_ASSERT(false, "Can't generate descriptor for MME node type - not supported");
}

void DescriptorGenerator::visit(MmeTransposeNode* node)
{
    addMmeDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(TPCNode* node)
{
    addTpcDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(TPCSlice* node)
{
    TPCSliceDescUpdate updater(node);
    addTpcDescriptorsToGraph(*static_cast<TPCNode*>(node), &updater);
}

void DescriptorGenerator::visit(ConvolutionNode* node)
{
    addMmeDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(GEMMNode* node)
{
    addMmeDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(DeToDwNode* node)
{
    addMmeDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(DeToDxNode* node)
{
    addMmeDescriptorsToGraph(*node);
}

void DescriptorGenerator::visit(DMANode* node)
{
    addDmaDescriptorsToGraph(*node);
}

MmeCommon::EMmeOpType DescriptorGenerator::getOperationType(const MmeNode& node)
{
    switch (node.getNodeType())
    {
        case Node::TYPE_DEDW:
            return MmeCommon::e_mme_dedw;
        case Node::TYPE_DEDX:
            return MmeCommon::e_mme_dedx;
        case Node::TYPE_CONVOLUTION:
            return MmeCommon::e_mme_fwd;
        case Node::TYPE_BATCH_GEMM:
        case Node::TYPE_BATCH_GEMM_DEDX:
        case Node::TYPE_BATCH_GEMM_DEDW:
        {
            const synGEMMParams& gemmParams = static_cast<const GEMMNode*>(&node)->getGEMMParams();
            if (!gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_ab;
            }
            if (gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_atb;
            }
            if (!gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_abt;
            }
            if (gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_atbt;
            }
        }
        case Node::TYPE_FC:
        case Node::TYPE_GEMM:
        case Node::TYPE_GEMM_DEDX:
        case Node::TYPE_GEMM_DEDW:
        {
            const synGEMMParams& gemmParams = static_cast<const GEMMNode*>(&node)->getGEMMParams();
            if (!gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_fwd;
            }
            else if (gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_dedw;
            }
            else if (!gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_dedx;
            }
            else if (gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_atbt;
            }
            break;
        }
        case Node::TYPE_INTERNAL_TRANSPOSE:
            return MmeCommon::e_mme_dedw;
        default:
            HB_ASSERT(false, "Unsupported Gaudi MME type {}", node.getNodeType());
    }
    return MmeCommon::e_mme_fwd;
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
        default:
        {
            HB_ASSERT(false, "Not a valid synRoundingMode!");
            return MmeCommon::RoundingMode::RoundToNearest;
        }
    }
}

void DescriptorGenerator::generateMmeDescriptor(const MmeNode& node, std::list<MmeActivation>& activations)
{
    MmeCommon::MmeLayerParams params = setMmeLayerParamsFromNode(node);
    params.strategy             =   getMmeStrategy(node);
    params.controls.atomicAdd   =   getMmeSramReduction(node);
    params.controls.roundingMode     = synapseToMmeRoundingMode(node.getRoundingMode());
    // TODO [SW-11381] Remove restriction after debugging why dedx fails if split to ROI (synapse dedx tests fail)
    // TODO [SW-40143] Remove 3D convolution restriction after reducing the number of signals
    // TODO [SW-88249] Remove or revise dynamic padding restriction once spatial slicing is supported
    if (node.getNodeAnnotation().splitToLogicalROIs && params.opType != MmeCommon::e_mme_dedx &&
        !node.is3DConvolution() && !node.isDynamicPaddingConvolution())
    {
        params.controls.signalingMode = MmeCommon::e_mme_signaling_desc_with_store;
    }
    else
    {
        params.controls.squashIORois =  true;
    }
    params.tracing.ctxId = node.getContextId();

    // Flatten the node tensors if applicable
    MmeCommon::MmeBrain mmeBrain(MmeCommon::e_mme_Gaudi, node.getMmeBrainIfc()->getOperationModes());
    bool appliedFlattening = mmeBrain.applyTensorFlattening(params);
    if (appliedFlattening)
    {
        LOG_TRACE(MME_STACK, "Applied flattening to x and y tensors");
    }
    std::shared_ptr<MmeCommon::PerfAttr> perfAttr = std::make_shared<MmeCommon::PerfAttr>();
    mmeBrain.getPerfAttr(params, *perfAttr);
    unsigned numActivations = perfAttr->numOfActivations;
    if (numActivations > GCFG_MAX_MME_ACTIVATIONS.value())
    {
        params.strategy.sbReuse = false;
    }
    MmeLogger mmeLogger;
    mmeLogger.printMmeParams(params);

    if (DES_CACHE.isDesCacheEnabled())
    {
        //DescriptorsCache enabled, try to find descriptor in descriptors' cache
        DES_CACHE.generateDescriptorsCache(params, activations);
    }
    else
    {
        //DescriptorsCache is disabled , generate descriptors
        bool useOldDescriptors = GCFG_MME_ENABLE_USE_OLD_DESCRIPTORS.value();

        if (useOldDescriptors)
        {
            generateDescriptors(params, activations);
        }
        else
        {
            auto descGenerator = gaudi::MmeDescriptorGenerator::createMmeDescGenerator(params);
            descGenerator->mmeGenerateActivations();
            activations = descGenerator->getMmeActivations();
            if (descGenerator->isSignalOverflow())
            {
                LOG_WARN(MME_STACK, "signalingMode changed from output to desc due to signals exceeding max value");
            }
            mmeLogger.printDebugInfoGaudi( &*descGenerator);
        }
    }
    // TODO: add mmeBrain usage to fill other data in PerfAttr.

    for (const auto& act : activations)
    {
        perfAttr->rollUpArray.push_back(act.numRollups);
    }
    std::const_pointer_cast<MmeCommon::PerfAttr>(node.getNodeAnnotation().mmeMetaData.mmePerfAttr) =
        perfAttr;  // ugly hack to remove const
    mmeLogger.printMmePerf(*perfAttr);

    pTensor xTensor;
    pTensor wTensor;
    pTensor yTensor;
    pTensor oTensor;
    const MmeCommon::EMmeOpType opType = getOperationTypeCommon(MmeCommon::e_mme_Gaudi, node);
    getTensorRolesCommon(node, opType, xTensor, wTensor, yTensor, oTensor);
    patchTensorAddresses(activations, xTensor, wTensor, yTensor);
}

void DescriptorGenerator::addMmeDescriptorsToGraph(const MmeNode& node)
{
    // MME descriptors are added at the calculate linear ranges pass
}

void DescriptorGenerator::patchTensorAddresses(std::list<MmeActivation>& activations,
                                               const pTensor& xTensor,
                                               const pTensor& wTensor,
                                               const pTensor& yTensor)
{
    for (auto& act : activations)
    {
        patchTensorView(MmeCommon::e_mme_op_x,
                        act.getDesc(0),
                        act.getDesc(1),
                        xTensor->getTensorOffset(),
                        act.isGemm);
        patchTensorView(MmeCommon::e_mme_op_w,
                        act.getDesc(0),
                        act.getDesc(1),
                        wTensor->getTensorOffset(),
                        act.isGemm);
        patchTensorView(MmeCommon::e_mme_op_y,
                        act.getDesc(0),
                        act.getDesc(1),
                        yTensor->getTensorOffset(),
                        act.isGemm);
    }
}

void DescriptorGenerator::addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater)
{
    std::list<DescAndMask<gaudi::TpcDesc>> descs;
    std::list<NodeROI>& rois = m_codeGenerator->getPhysicalRois(m_codeGenerator->getNodeSharedPtr(node));

    generateTpcDescriptors(node, rois, m_codeGenerator->getKernelAddress(node.getUniqueID()), descs);

    if (updater)
    {
        updater->update(descs);
    }

    auto descIt = descs.begin();
    auto roisIt = rois.begin();
    HB_ASSERT(descs.size() == rois.size(), "Number of ROIs does not match the number of descriptors");
    for (; descIt != descs.end() && roisIt != rois.end() ; ++descIt, ++roisIt)
    {
        m_codeGenerator->updateTPCDescriptorWrapper(node, descIt->first, descIt->second, *roisIt);
    }
}

void DescriptorGenerator::addDmaDescriptorsToGraph(const DMANode& node)
{
    if (node.getDmaType() == DMA_TYPE_INTERNAL)
    {
        std::list<DescAndMask<gaudi::DmaDesc>> descs;
        std::list<NodeROI>& rois = m_codeGenerator->getPhysicalRois(m_codeGenerator->getNodeSharedPtr(node));

        generateDmaDescriptors(node, rois, descs, m_codeGenerator->getSyncObjectManager()->getDummySyncId());

        auto descIt = descs.begin();
        auto roisIt = rois.begin();
        HB_ASSERT(descs.size() == rois.size(), "Number of ROIs does not match the number of descriptors");
        for (; descIt != descs.end() && roisIt != rois.end() ; ++descIt, ++roisIt)
        {
            m_codeGenerator->updateDMADescriptorWrapper(node, descIt->first, descIt->second, *roisIt);
        }
    }
}

}  // end of namespace gaudi
