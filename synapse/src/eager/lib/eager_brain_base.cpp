#include "eager_brain_base.h"

// eager includes (relative to src/eager/lib/)
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/conv_base_node.h"
#include "graph_compiler/habana_nodes/mme_node.h"
#include "graph_compiler/mme/mme_desc_gen_utils.h"
#include "hal_reader/hal_reader.h"

// relative to <mme>/
#include "include/mme_common/mme_brain.h"

// std includes
#include <algorithm>
#include <cstdint>
#include <optional>

namespace eager_mode
{
using namespace MmeCommon;

EagerMmeBrainBase::GeometryApproximatedCost EagerMmeBrainBase::getBestGeometry(unsigned fcdSize,
                                                                               unsigned spatialSize) const
{
    // choose geometry based on the simplistic heuristic
    std::optional<GeometryApproximatedCost> bestGeo;
    for (const auto& currentGeometry : getSupportedGeometries())
    {
        unsigned fcdSteps = MmeCommon::div_round_up(fcdSize, currentGeometry.width);
        unsigned spSteps  = MmeCommon::div_round_up(spatialSize, currentGeometry.height);
        auto     cost     = static_cast<uint64_t>(fcdSteps) * spSteps;
        if (!bestGeo || bestGeo->cost > cost)
        {
            bestGeo.emplace(GeometryApproximatedCost {&currentGeometry, cost});
        }
    }
    EAGER_ASSERT(bestGeo, "");
    return *bestGeo;
}

bool EagerMmeBrainBase::shouldPackConvWeights(const ConvBaseNode& convNode, uint32_t packingFactor) const
{
    const auto& inSizes  = convNode.getInput(0)->getAllNSizesInElements();
    const auto& wSizes   = convNode.getInput(1)->getAllNSizesInElements();
    const auto& outSizes = convNode.getOutput(0)->getAllNSizesInElements();

    // calculate the expected cycles before packing
    uint64_t fcdSize     = outSizes[0];
    uint64_t spatialSize = MmeCommon::multiplyElements(outSizes.begin() + 1, outSizes.begin() + SYN_MAX_TENSOR_DIM);
    GeometryApproximatedCost bestGeo = getBestGeometry(fcdSize, spatialSize);
    // calculate overall CD of all accumulated loops (CD = C * S * R * Q)
    uint64_t nonChangingCdSize       = inSizes[0] * wSizes[3] * wSizes[4];
    uint64_t cdSize                  = wSizes[2] * nonChangingCdSize;
    uint64_t minCDToConsiderInCycles = std::max<uint64_t>(cdSize, m_rollUpTime);
    uint64_t expectedCycles          = minCDToConsiderInCycles * bestGeo.cost;
    // Don't bother packing if expected cycle count is lower than threshold
    if (expectedCycles < m_conv_packing_threshold) return false;

    // calculate the expected cycles after packing
    bestGeo                                    = getBestGeometry(fcdSize * packingFactor, spatialSize / packingFactor);
    const synConvolution3DParamsV2& convParams = convNode.getConvolutionParams();
    uint64_t                        weightPackedDimSSize =
        wSizes[2] + static_cast<uint64_t>(convParams.stride[CONV_STRIDE_WIDTH]) * (packingFactor - 1);
    cdSize                        = weightPackedDimSSize * nonChangingCdSize;
    minCDToConsiderInCycles       = std::max<uint64_t>(cdSize, m_rollUpTime);
    uint64_t packedExpectedCycles = minCDToConsiderInCycles * bestGeo.cost;
    // Avoid packing if it increases the expected cycle count
    if (packedExpectedCycles >= expectedCycles) return false;
    // Avoid packing if it is not improving the expected cycle count significantly
    if (expectedCycles - packedExpectedCycles < m_conv_packing_threshold) return false;
    return true;
}

// Is node candidate for batch and non-deterministic CD concurrency
bool EagerMmeBrainBase::isNodeCandidateForMmeConcurrency(const MmeNode& node, const HalReader& hal) const
{
    if (node.getDeterministic() && !GCFG_FORCE_MME_CD_CONCURRENCY_NON_DETERMINISTIC.value()) return false;

    if (!MmeBrain::opSupportsChoosingConcurrency(getOperationTypeCommon(getChipType(), node)))
    {
        return false;
    }

    const Tensor&     output         = *node.getOutput(0);
    const synDataType outputDataType = output.getElementType();
    // TODO 106087: Apply to nodes of output of fp8 as well. Currently applying to such nodes fail because
    // no cast from fp32 to fp8 is available
    if ((outputDataType == syn_type_fp8_143) || (outputDataType == syn_type_fp8_152) ||
        (outputDataType == syn_type_tf32))
    {
        return false;  // No cast available from fp32 to fp8
    }
    if (!isConcurrencySupportedForOutputDataType(outputDataType)) return false;

    // Verify that output tensor is located in reducible memory
    if (!hal.isReducibleMemory(output.inSram() ? MemoryType::MEMORY_TYPE_SRAM : MemoryType::MEMORY_TYPE_DRAM))
    {
        return false;
    }

    return true;
}

}  // namespace eager_mode
