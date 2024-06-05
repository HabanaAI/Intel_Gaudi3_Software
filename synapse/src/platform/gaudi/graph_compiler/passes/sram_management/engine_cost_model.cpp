#include "engine_cost_model.h"

#include "data_type_utils.h"
#include "defs.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "graph_compiler/passes/sram_management/slicing_utils.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "include/mme_common/mme_brain.h"
#include "mme_desc_gen_utils.h"
#include "pipeline_management/mme_brain_proxy.h"
#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "utils.h"

namespace gaudi
{
static uint64_t trafficBytesToTimeNano(uint64_t hbmTrafficBytes)
{
    // Round up to avoid 0 time
    return std::ceil(hbmTrafficBytes / SlicingBrain::knobs.hbmAvailableBWGBps);
}

uint64_t MMECostModel::getBatchFactor(const pNode&                  node,
                                      const MmeCommon::MmeStrategy& mmeStrategy,
                                      const MmeCommon::EMmeOpType&  operationType,
                                      const pSliceReference&        output,
                                      unsigned                      heightOutputElements,
                                      unsigned                      widthOutputElements) const
{
    uint64_t batchFactor = 1;
    if (node->isBatchGemm())
    {
        // When 2wx2h geometry is used and each GEMM is small enough to fit into half of the MME EUs - 2 GEMMs are
        // executed concurrently.
        MmeBatchMode batchMode = getBatchGemmBatchMode(
            operationType,
            mmeStrategy.geometry,
            heightOutputElements,
            widthOutputElements,
            m_halReader.getMmeVectorSize() / dataTypeSizeInBytes(output->operand->originalTensor->getElementType()));
        bool shouldEnable2XOpt = ((batchMode == mme_batch_2xw) || (batchMode == mme_batch_2xh));

        // TODO [SW-60505]: Support BGEMM optimization on all batch dims,
        // batch factor should come from MME stack.
        for (unsigned dim = DIM_GEMM_BATCH; dim < output->operand->originalTensor->getDim(); ++dim)
        {
            if ((dim == DIM_GEMM_BATCH) && shouldEnable2XOpt)
            {
                batchFactor *= std::ceil((float)output->operand->chunkDimensions[dim] / 2.f);
            }
            else
            {
                batchFactor *= output->operand->chunkDimensions[dim];
            }
        }
    }
    return batchFactor;
}

uint64_t MMECostModel::calcProcessingTime(const pNode& node,
                                         const SliceReferenceList& inputs,
                                         const SliceReferenceList& outputs) const
{
    if (m_dimControlers.count(node) == 0)
    {
        m_dimControlers.emplace(node, node);
    }
    const MmeDimController& dimController = m_dimControlers.at(node);

    // the filter is never split - so its sizes will be relevant for every "sliced" node:
    unsigned filterSizeInElements = multiplyElements(dimController.qrsSizes().begin(), dimController.qrsSizes().end());
    //the common dimension could be split, so we need to reflect the real sizes represented in the input slice ref:
    unsigned cdSizeInElements = SlicedOperandUtils::getAggregatedDimSizeInElements(inputs.front(),
                                                                                   dimController.commonDimOperandA());
    // kernel filter causes more activations:
    unsigned cdTotalSizeInElements = cdSizeInElements * filterSizeInElements;
    // there is a minimum to the common dim size in the mme:
    unsigned minimumCdSizeElements = m_halReader.getMmeMinCDInElements(inputs.front()->operand->finalElementType,
                                                                       outputs.front()->operand->finalElementType);
    unsigned cyclesPerActivation = std::max(cdTotalSizeInElements, minimumCdSizeElements);

    // calculate the number of activations (tetrises)-
    // first - calculate the width and height represented in the output slice reference:
    unsigned heightOutputElements = SlicedOperandUtils::getAggregatedDimSizeInElements(outputs.front(),
                                                                                       dimController.heightOutput());
    unsigned widthOutputElements = SlicedOperandUtils::getAggregatedDimSizeInElements(outputs.front(),
                                                                                      dimController.widthOutput());

    const std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    HB_ASSERT_PTR(mmeNode);
    auto chipType = MmeBrainProxy::getMmeChipType(m_halReader.getDeviceType());
    const MmeCommon::EMmeOpType operationType = getOperationTypeCommon(chipType, *mmeNode);

    const auto& mmeStrategy = getMMEStrategy(node, inputs.front(), outputs.front(), operationType);

    unsigned mmeGeometryHeightElements = 1, mmeGeometryWidthElements = 1;
    // retrieve the actual geometry that will be used by the mme for this activation
    getMMEGeometryInElements(mmeStrategy,
                             inputs.front()->operand->finalElementType,
                             mmeGeometryHeightElements,
                             mmeGeometryWidthElements);
    // calculate activations separately on height and width, in order not to cover up "partly full" activation,
    // i.e. if we have 1x4 activation and the mme geometry for some reason is 2x2 -> it will count as 2 activations
    unsigned heightActivations = std::ceil((double)heightOutputElements / mmeGeometryHeightElements);
    unsigned widthActivations = std::ceil((double)widthOutputElements / mmeGeometryWidthElements);
    uint64_t numOfMMEActivations = heightActivations * widthActivations;

    uint64_t batchFactor =
        getBatchFactor(node, mmeStrategy, operationType, outputs.front(), heightOutputElements, widthOutputElements);

    uint64_t totalCycles = numOfMMEActivations * batchFactor * cyclesPerActivation;

    const auto unalignedFactor = getUnalignedPenaltyFactor(mmeStrategy, inputs, widthActivations, heightActivations);

    return unalignedFactor * totalCycles / SlicingBrain::knobs.freqGHz;
}

float MMECostModel::getUnalignedPenaltyFactor(const MmeCommon::MmeStrategy& mmeStrategy,
                                              const SliceReferenceList&     inputs,
                                              unsigned                      widthActivations,
                                              unsigned                      heightActivations) const
{
    HB_ASSERT(inputs.size() >= 2, "MMECostModel: Invalid number of inputs for MME node");

    const auto& sliceRefA  = inputs.front();
    const auto& sliceRefB  = *std::next(inputs.begin());
    bool        isAAligned = isAlignedToCL(sliceRefA);
    bool        isBAligned = isAlignedToCL(sliceRefB);

    // Unaligned input reads cost 2X BW.
    // 2 operands are unaligned – 2X slower.
    // 1 operand is unaligned and is reused in SB – ((1+SBReuseFactore)/SBReuseFactor)X slower.
    // 1 operand is unaligned and the other operand is reused in SB – 2X slower.

    if (isAAligned && isBAligned)  // Both operands are aligned - no penalty
    {
        return 1;
    }

    // Find which operand is reused in SB according to the walking pattern.
    const auto& reusedOperand = getSbReusedOperand(mmeStrategy, sliceRefA, sliceRefB);

    if (isBAligned && !isAAligned && (sliceRefA == reusedOperand))
    {
        return ((1 + widthActivations) / (float)widthActivations);
    }
    else if (isAAligned && !isBAligned && (sliceRefB == reusedOperand))
    {
        return ((1 + heightActivations) / (float)heightActivations);
    }
    return 2;  // Both operands are unaligned or no reuse on the unaligned operand
}

bool MMECostModel::isAlignedToCL(const pSliceReference& sliceRef) const
{
    if (sliceRef->operand->alignWithCacheLine)
    {
        return true;
    }
    const auto& alignedStrides =
        SlicedOperandUtils::getCacheLineAlignedStrides(SlicedOperandUtils::calcSliceSizesFromSliceRef(sliceRef),
                                                       sliceRef->operand);
    return !alignedStrides.has_value();
}

uint64_t MMECostModel::calcHBMTraffic(const pNode& node,
                                      const SliceReferenceList& inputs,
                                      const SliceReferenceList& outputs) const
{
    uint64_t totalTrafficBytes = 0;
    for (const auto* sliceRefListPtr : {&inputs, &outputs})
    {
        for (const auto& sliceRef : *sliceRefListPtr)
        {
            if (sliceRef->operand->resideInSRAM || sliceRef->operand->originalTensor->inSram()) continue;
            totalTrafficBytes += SlicedOperandUtils::getSliceSizeInBytes(sliceRef);
        }
    }
    return totalTrafficBytes;
}

MmeCommon::MmeStrategy MMECostModel::getMMEStrategy(const pNode&                 node,
                                                    const pSliceReference&       inputASliceRef,
                                                    const pSliceReference&       outputSliceRef,
                                                    const MmeCommon::EMmeOpType& operationType) const
{
    auto chipType = MmeBrainProxy::getMmeChipType(m_halReader.getDeviceType());
    MmeCommon::MmeStrategy         mmeStrategy = MmeCommon::MmeBrain::getDefaultParams(chipType).strategy;
    const std::shared_ptr<MmeNode> mmeNode     = std::dynamic_pointer_cast<MmeNode>(node);
    HB_ASSERT_PTR(mmeNode);
    // use the same function used by the descriptor generator to get the actual geometry that will be used
    mmeStrategy = DescriptorGenerator::getMmeStrategy(
        *mmeNode,
        operationType,
        inputASliceRef->operand->finalElementType,
        SlicedOperandUtils::calcSliceSizesFromSliceRef(inputASliceRef),
        SlicedOperandUtils::calcSliceSizesFromSliceRef(outputSliceRef),
        LOG_LEVEL_AT_LEAST_TRACE(MME_STACK) ? node->getNodeName() + "_for_sram_management" : "",
        node->getNodeAnnotation().mmeMetaData.packing[PACKING_X]);

    return mmeStrategy;
}

pSliceReference MMECostModel::getSbReusedOperand(const MmeCommon::MmeStrategy& mmeStrategy,
                                                 const pSliceReference&        opA,
                                                 const pSliceReference&        opB) const
{
    if (!mmeStrategy.sbReuse)
    {
        return nullptr;  // No reuse
    }
    switch (mmeStrategy.pattern)
    {
        // Walking right (k) and then down (c/s)
        case MmeCommon::e_mme_sp_reduction_fck:
        case MmeCommon::e_mme_sp_reduction_cfk:
        case MmeCommon::e_mme_sp_reduction_ckf:
        case MmeCommon::e_mme_z_reduction_skf:
            return opA;
        // Walking down (c/s) and then right (k)
        case MmeCommon::e_mme_sp_reduction_kfc:
        case MmeCommon::e_mme_sp_reduction_fkc:
        case MmeCommon::e_mme_z_reduction_ksf:
        case MmeCommon::e_mme_sp_reduction_kcf:
            return opB;
        default:
            return nullptr;
    }
}

void MMECostModel::getMMEGeometryInElements(const MmeCommon::MmeStrategy& mmeStrategy,
                                            const synDataType&            finalElementType,
                                            unsigned&                     mmeGeometryHeightElements,
                                            unsigned&                     mmeGeometryWidthElements) const
{
    mmeGeometryHeightElements = 1;
    mmeGeometryWidthElements  = 1;

    switch (mmeStrategy.geometry)
    {
        case MmeCommon::e_mme_geometry_1wx4h:
            mmeGeometryHeightElements = 4;
            break;
        case MmeCommon::e_mme_geometry_4wx1h:
            mmeGeometryWidthElements = 4;
            break;
        case MmeCommon::e_mme_geometry_2wx2h:
            mmeGeometryHeightElements = 2;
            mmeGeometryWidthElements  = 2;
            break;
        default:
            HB_ASSERT(0, "Geometry {} is invalid for gaudi", mmeStrategy.geometry);
    }
    // 2x2 geometry of float -> 64x64 elements, 2x2 geometry of bf16 -> 128x128 and so on
    mmeGeometryHeightElements *= m_halReader.getMmeVectorSize() / dataTypeSizeInBytes(finalElementType);
    mmeGeometryWidthElements *= m_halReader.getMmeVectorSize() / dataTypeSizeInBytes(finalElementType);
}

CostModel::Cost MMECostModel::calcCost(const pNode &node,
                                       const SliceReferenceList &inputs,
                                       const SliceReferenceList &outputs) const
{
    Cost cost(Cost::Engine::MME);
    cost.hbmTrafficBytes = calcHBMTraffic(node, inputs, outputs);
    cost.timeNano = calcProcessingTime(node, inputs, outputs);
    return cost;
}

uint64_t TPCCostModel::calcHBMTraffic(const pNode& node,
                                      const SliceReferenceList& inputs,
                                      const SliceReferenceList& outputs) const
{
    uint64_t totalTrafficBytes = 0;
    for (const auto* sliceRefListPtr : {&inputs, &outputs})
    {
        for (const auto& sliceRef : *sliceRefListPtr)
        {
            if (sliceRef->operand->resideInSRAM || sliceRef->operand->originalTensor->inSram()) continue;
            totalTrafficBytes += SlicedOperandUtils::getSliceSizeInBytes(sliceRef);
        }
    }
    return totalTrafficBytes;
}

CostModel::Cost TPCCostModel::calcCost(const pNode &node,
                                       const SliceReferenceList &inputs,
                                       const SliceReferenceList &outputs) const
{
    Cost cost(Cost::Engine::TPC);
    cost.hbmTrafficBytes = calcHBMTraffic(node ,inputs, outputs);
    // Estimation for TPC processing time in bundling case, which mostly include simple element-wise kernels,
    // is the time it takes to read and write the TPC operands from/to HBM
    cost.timeNano = trafficBytesToTimeNano(cost.hbmTrafficBytes);
    return cost;
}

DmaFetchCostModel::DmaFetchCostModel(const HabanaGraph& graph) : m_graph(graph) {}

CostModel::Cost DmaFetchCostModel::calcCost(const pNode&              node,
                                            const SliceReferenceList& inputs,
                                            const SliceReferenceList& outputs) const
{
    Cost cost(CostEngine::DMA);
    cost.hbmTrafficBytes = calcHBMTraffic(inputs, true);
    cost.timeNano        = trafficBytesToTimeNano(cost.hbmTrafficBytes);

    // Call calcHBMTraffic() for outputs as well (to keep the double buffering mechanism accurate)
    // but ignore the result of the outputs (DmaEvictionCostModel is responsible for them)
    calcHBMTraffic(outputs, false);

    return cost;
}

uint64_t DmaFetchCostModel::calcHBMTraffic(const SliceReferenceList& sliceRefs, bool isInput) const
{
    uint64_t totalTrafficBytes = 0;
    for (const auto& sliceRef : sliceRefs)
    {
        // The following lines just simulate the way the tensor slicer works with the cache
        bool sliceIsCached = false;
        // Get the relevant cache for the given operand (or create if doesn't exist yet)
        const auto& iterFoundCache = m_cacheMap.insert(
            {sliceRef->operand,
             {sliceRef->operand,
              LOG_LEVEL_AT_LEAST_TRACE(BE_SLICER) ? sliceRef->operand->originalTensor->getName() + "_for_dma_cost_model"
                                                  : ""}});
        if (isInput)
        {
            if (!iterFoundCache.second)  // Cache already exists
            {
                sliceIsCached = iterFoundCache.first->second.getCachedSlice(sliceRef->coordinates);
            }
        }
        if (!sliceIsCached)
        {
            iterFoundCache.first->second.addNewSlice(sliceRef->coordinates, true);
            if (isInput &&  (sliceRef->operand->resideInSRAM || sliceRef->operand->originalTensor->inSram()))
            {
                // If the tensor is not in the cache and is in sram, we will need a dma to bring it
                totalTrafficBytes += SlicedOperandUtils::getSliceSizeInBytes(sliceRef);
            }
        }
    }
    return totalTrafficBytes;
}

DmaEvictionCostModel::DmaEvictionCostModel(const HabanaGraph&         graph,
                                           const pBundle&             bundle,
                                           const pMmeSlicingStrategy& strategy)
: m_graph(graph), m_bundleNodeSet(strategy->getMmeSlicingData().getStrategyNodes(bundle))
{
}

CostModel::Cost DmaEvictionCostModel::calcCost(const pNode&              node,
                                               const SliceReferenceList& inputs,
                                               const SliceReferenceList& outputs) const
{
    Cost cost(CostEngine::DMA);
    // Note: the next line can update the engine of the cost to be tpc.
    // TODO: once we have an improved tpc cost model, consider letting the tpc cost model evaluate such an operation.
    std::tie(cost.engine, cost.hbmTrafficBytes) = calcHBMTraffic(outputs);
    cost.timeNano                               = trafficBytesToTimeNano(cost.hbmTrafficBytes);
    return cost;
}

std::pair<CostEngine, uint64_t> DmaEvictionCostModel::calcHBMTraffic(const SliceReferenceList& outputs) const
{
    // The engine is "normally" DMA (except eviction of partials which were originally bf16).
    CostEngine engine            = CostEngine::DMA;
    uint64_t   totalTrafficBytes = 0;
    for (const auto& sliceRef : outputs)
    {
        // An output slice should be evicted if:
        // (1) The original tensor was not in sram but the strategy put it in sram
        // (2) and wasn't evacuated yet
        // (3) and is either persistent or is consumed by another node out of the bundle
        if (sliceRef->operand->resideInSRAM && !sliceRef->operand->originalTensor->inSram() &&
            m_evacuatedSliceRefSet.count(sliceRef) == 0 &&
            BundleSlicer::shouldTensorBeEvicted(sliceRef->operand->originalTensor, m_graph, m_bundleNodeSet))
        {
            m_evacuatedSliceRefSet.insert(sliceRef);
            if (sliceRef->operand->finalElementType == sliceRef->operand->originalTensor->getElementType())
            {
                totalTrafficBytes += SlicedOperandUtils::getSliceSizeInBytes(sliceRef);
            }
            else
            {
                // Support eviction of "partials" output which were originally bf16 - the memcpy will be converted
                // to a  tpc cast node from float to bf16 (so the hbm traffic should calc the bf16 - this is what
                // will written to the hbm)
                totalTrafficBytes +=
                    SlicedOperandUtils::getSliceSizeInBytes(sliceRef, true /* use original tensor element type */);
                engine = CostEngine::TPC;
            }
        }
    }
    return {engine, totalTrafficBytes};
}

} // namespace gaudi
