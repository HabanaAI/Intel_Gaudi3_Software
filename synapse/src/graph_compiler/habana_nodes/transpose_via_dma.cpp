#include "dma_transpose_engine_params.h"
#include "graph_traits.h"
#include "node_factory.h"
#include "kernel_db.h"
#include "data_type_utils.h"
#include "compilation_hal_reader.h"
#include "fcd_ops_utils.h"
#include "cast_nodes_handler.h"

#include "dma_transpose_node.h"
#include "dma_transpose_helper.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "transpose_via_dma.h"
#include "transpose_nodes_creator.h"

TransposePermutationArrayVec
DmaTransposeStrategy::getSplittedPermutationsDma(const TransposePermutationArray& permutation,
                                                 bool                             preferLogicalBeforePhysical)
{
    TransposePermutationArrayVec splittedPermutations;
    if (preferLogicalBeforePhysical)
    {
        splittedPermutations = splitLogicalBeforePhysical(permutation);
    }
    else
    {
        splittedPermutations = splitPermutation(permutation);
    }

    return splittedPermutations;
}

uint32_t DmaTransposeStrategy::getFcdIndex(const TransposePermutationArray& permutation)
{
    auto toRet = std::distance(permutation.begin(),
                               std::find(permutation.begin(), permutation.end(), TransposePermutationDim::TPD_Channel));
    HB_ASSERT(toRet < permutation.size(), "Fcd index not found");
    return toRet;
}

bool DmaTransposeStrategy::isRotation(const TransposePermutationArray& permutation, unsigned int dim)
{
    auto fcdIndex = getFcdIndex(permutation);
    int  j        = 0;
    for (int i = fcdIndex; i < dim; i++, j++)
    {
        if (permutation[i] != j)
        {
            return false;
        }
    }

    for (int i = 0; i < fcdIndex; i++, j++)
    {
        if (permutation[i] != j)
        {
            return false;
        }
    }
    return true;
}

TSize DmaTransposeStrategy::getRotationWriteSizeInElements(const Tensor* input, const TransposePermutationArray& p)
{
    HB_ASSERT(isRotation(p, input->getDim()), "Error - given permutation is not a rotation");
    auto  newFcd = getFcdIndex(p);
    TSize size   = 1;

    for (unsigned dim = 0; dim < newFcd; ++dim)

    {
        size *= input->getSizeInElements(p[dim]);
    }

    return size;
}

TSize DmaTransposeStrategy::getRotationReadSizeInElements(const Tensor* input, const TransposePermutationArray& p)
{
    return input->getDenseSizeInElements() / getRotationWriteSizeInElements(input, p);
}

bool DmaTransposeStrategy::isOptimizedRotation(const Tensor*                    input,
                                               const TransposePermutationArray& permutation,
                                               const DmaTransposeEngineParams&  params)
{
    DmaTransposeHelper helper(input->getElementType(), params);
    auto               newFcdIndex = getFcdIndex(permutation);

    return isRotation(permutation, input->getDim()) && (newFcdIndex != input->getDim() - 1) &&
           (((float)(input->getSizeInElements(0)) / (float)helper.maxSrcDim0()) <= 0.2) &&
           (((float)getRotationReadSizeInElements(input, permutation) / (float)helper.maxSrcDim0()) >= 0.5) &&
           (((float)getRotationWriteSizeInElements(input, permutation) / (float)helper.maximalDestElementsDim0()) >=
            0.5);
}

// For clearer interface. Can be replaced with concerete computation,
// but currently static priority/order is enough.
template<uint64_t Priority>
class TransposeStrategyStaticPriority : public DmaTransposeStrategy
{
public:
    uint64_t priority(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override
    {
        return Priority;
    }
};

/*************************************************************************
 *                 TransposeViaGenericDma Strategy
 *************************************************************************/
class TransposeViaGenericDma : public DmaTransposeStrategy
{
public:
    uint64_t         priority(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    bool             canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector       extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    uint64_t         cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose by generic dma strategy"; }

private:
    NodeVector createTransposeNode(const TensorPtr&                 in,
                                   const TensorPtr&                 out,
                                   const TransposePermutationArray& permutation,
                                   std::string_view                 name) const;
};

uint64_t TransposeViaGenericDma::priority(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    // for static priority
    if (hal == nullptr || transposeNodeParams.isEmpty())
    {
        return GENERIC_DMA_PRIORITY;
    }

    const Tensor*                   input       = transposeNodeParams.input.get();
    auto                            permutation = transposeNodeParams.permutation;
    const DmaTransposeEngineParams& params      = hal->getDmaTransposeEngineParams();

    // In case the permutation is rotation (can be done by one generic dma transpose)
    // and the expected performance is higher than FullyUtilized / DoubleTranspose
    // return high priority
    if (GCFG_ENABLE_PREFER_GENERIC_DMA_OPT.value() && isOptimizedRotation(input, permutation, params))
    {
        LOG_TRACE(GC,
                  "Highest priority for generic dma, node {}: input sizes: {}, permutation: [{}]",
                  transposeNodeParams.nodeName,
                  input->getDimSizesStr(),
                  toString(permutation, ','));
        return OPTIMIZED_ROTATION_PRIORITY;
    }

    // Optimization only for Gaudi or GaudiM.
    if (!GCFG_ENABLE_SINGLE_TRANSPOSE_DYNAMIC_PRIORITY.value())
    {
        return GENERIC_DMA_PRIORITY;
    }
    if (!canBeUsed(transposeNodeParams, hal))
    {
        return GENERIC_DMA_PRIORITY;
    }

    // Will be moved to a cost model later.
    static constexpr float MIN_PHYSICAL_UTILIZATION = 0.98;
    static constexpr float MIN_LOGICAL_UTILIZATION  = 0.75;
    unsigned               clSize                   = hal->getCacheLineSizeInBytes();

    // Don't optimize if we can perform with a single fully utilized transpose.
    auto          newFcdIndex              = getFcdIndex(permutation);
    const Tensor* output                   = transposeNodeParams.output.get();
    auto          inputElementsSizeInBytes = input->getElementSizeInBytes();
    if (newFcdIndex != input->getDim())
    {
        const auto& sizes         = input->getAllNSizesInElements();
        int         basicNumLines = inputElementsSizeInBytes;
        for (int i = 0; i < newFcdIndex; i++)
        {
            basicNumLines *= sizes[permutation[i]];
        }
        if (basicNumLines % params.numLinesDivisor == 0)
        {
            return GENERIC_DMA_PRIORITY;
        }
    }
    // Don't optimize if getPreferLogicalBeforePhysical was set by OptimizeTpcKernels.
    if (transposeNodeParams.preferLogicalBeforePhysical)
    {
        return GENERIC_DMA_PRIORITY;
    }
    // Only FCD is being tranposed
    auto perm = transposeNodeParams.permutation;
    perm.erase(std::remove(perm.begin(), perm.end(), TransposePermutationDim::TPD_Channel));
    for (size_t i = 0; i < perm.size(); i++)
    {
        if (perm[i] != static_cast<TransposePermutationDim>(i + 1))
        {
            return GENERIC_DMA_PRIORITY;
        }
    }

    auto permutations =
        getSplittedPermutationsDma(transposeNodeParams.permutation, transposeNodeParams.preferLogicalBeforePhysical);
    if (permutations.size() < 2)
    {
        return GENERIC_DMA_PRIORITY;
    }
    // Don't optimize if second permutation is just a reshape.
    auto tmp = getTensorAfterTranspose(*input, permutations[0]);
    if (isSameDataMemOrder(*tmp, permutations[1]))
    {
        return GENERIC_DMA_PRIORITY;
    }
    // Calculate sizes for physical transpose.
    unsigned flattenAxis = std::numeric_limits<unsigned>::max();
    auto     SizeArray   = lowerPhysicalTransposeTo2d(*input, permutations[0], flattenAxis);
    if (flattenAxis == std::numeric_limits<unsigned>::max())
    {
        return GENERIC_DMA_PRIORITY;
    }
    unsigned bytesBeforeFcd = SizeArray[0] * inputElementsSizeInBytes;
    unsigned bytesAfterFcd  = SizeArray[1] * inputElementsSizeInBytes;

    float logicalUtilization = output->getDenseSizeInBytes() /
                               (float)(FcdOpsUtils::calculateExpectedCost(*output, clSize, output->getSizeInBytes(0)) *
                                       inputElementsSizeInBytes);

    float PhysicalUtilizationBefore =
        input->getDenseSizeInBytes() /
        (float)(FcdOpsUtils::calculateExpectedCost(*input, clSize, bytesBeforeFcd) * inputElementsSizeInBytes);

    float PhysicalUtilizationAfter =
        input->getDenseSizeInBytes() /
        (float)(FcdOpsUtils::calculateExpectedCost(*input, clSize, bytesAfterFcd) * inputElementsSizeInBytes);
    float PhysicalUtilization = std::min(PhysicalUtilizationBefore, PhysicalUtilizationAfter);

    if (logicalUtilization >= MIN_LOGICAL_UTILIZATION && PhysicalUtilization >= MIN_PHYSICAL_UTILIZATION)
    {
        return TRANSPOSE_UTILIZED_PHYSICAL_THEN_LOGICAL;
    }
    return GENERIC_DMA_PRIORITY;
}

bool TransposeViaGenericDma::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return hal->isDmaTransposeSupported(transposeNodeParams.input->getElementType());
}

NodeVector TransposeViaGenericDma::extract(const TransposeNodeParams& transposeNodeParams,
                                           const HalReaderPtr&        hal) const
{
    TransposePermutationArrayVec splittedPermutations =
        getSplittedPermutationsDma(transposeNodeParams.permutation, transposeNodeParams.preferLogicalBeforePhysical);

    logTransposePermutations(transposeNodeParams.permutation, splittedPermutations);

    const TensorPtr& input   = transposeNodeParams.input;
    const TensorPtr& output  = transposeNodeParams.output;
    TensorPtr        nodeIFM = input;
    NodeVector       ret;
    // avoid using .value()_or to avoid std::string construction
    std::string_view name =
        transposeNodeParams.nodeName.has_value() ? std::string_view {transposeNodeParams.nodeName.value()} : "noName";
    if (splittedPermutations.size() > 1)
    {
        auto firstOfm = getTensorAfterTranspose(*input, splittedPermutations.front());
        ret           = createTransposeNode(input, firstOfm, splittedPermutations.front(), fmt::format("{}_t0", name));
        nodeIFM       = firstOfm;
    }
    auto moreNodes = createTransposeNode(nodeIFM, output, splittedPermutations.back(), name);
    ret.insert(ret.end(), moreNodes.begin(), moreNodes.end());
    return ret;
}

uint64_t TransposeViaGenericDma::cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    const TensorPtr& input = transposeNodeParams.input;

    TransposePermutationArrayVec newPermutations =
        getSplittedPermutationsDma(transposeNodeParams.permutation, transposeNodeParams.preferLogicalBeforePhysical);

    uint64_t cost = m_costModel.getCost(input, newPermutations.front());
    if (newPermutations.size() > 1)
    {
        HB_ASSERT(newPermutations.size() == 2, "physical transpose split into {} permutations", newPermutations.size());
        const auto& t = getTensorAfterTranspose(*input, newPermutations.front());
        cost += m_costModel.getCost(t, newPermutations.back());
    }
    return cost;
}

NodeVector TransposeViaGenericDma::createTransposeNode(const TensorPtr&                 in,
                                                       const TensorPtr&                 out,
                                                       const TransposePermutationArray& permutation,
                                                       std::string_view                 name) const
{
    NodeVector ret;
    if (isSameDataMemOrder(*in, permutation))
    {
        return TransposeViaDMA::createReshapeWithExtractTranspose(in,
                                                                  out,
                                                                  permutation,
                                                                  fmt::format("{}_reshape_transpose_by_dma", name));
    }
    if (LogicalTransposeNode::isSupportedPermutation(*in, *out, permutation))
    {
        LOG_DEBUG(HABANA_NODE, "adding logical transpose node - {}", name);
        ret.emplace_back(
            NodeFactory::createInternalNode({in}, {out}, &permutation, NodeFactory::transposeLogicNodeTypeName, name));
    }
    else
    {
        unsigned flattenAxis = -1;
        lowerPhysicalTransposeTo2d(*in, permutation, flattenAxis);
        auto      flattenNodes = FcdOpsUtils::createFlattenByReshapeNode(in, flattenAxis, name);
        TensorPtr flattenOut   = flattenNodes.back()->getOutput(0);

        if (FlattenNode::isRedundantNode(*in, *flattenOut, flattenAxis))
        {
            // If there is no need to flatten, just create the transpose
            ret.emplace_back(NodeFactory::createInternalNode({in}, {out}, nullptr, NodeFactory::transposeDmaNodeTypeName, name));
        }
        else
        {
            TensorPtr reshapedDmaOutput = getTensorAfterTranspose(*flattenOut,
                                                                  {TPD_Width, TPD_Channel},
                                                                  flattenOut->getName() + "_transpose_dma");
            TensorPtr shapeOut          = out->clone(false, false, false);
            shapeOut->setElementType(syn_type_int32);
            shapeOut->setShapeTensor(SHAPE_TENSOR);
            synTransposeParamsNDims params = permutationToParams(permutation);

            NodePtr exctractTranspose = NodeFactory::createInternalNode({in},
                                                                        {shapeOut},
                                                                        &params,
                                                                        NodeFactory::transposedShapeNodeTypeName,
                                                                        fmt::format("{}_transpose_shape", name));
            NodePtr reshapeAfterDma   = NodeFactory::createInternalNode({reshapedDmaOutput, shapeOut},
                                                                      {out},
                                                                      nullptr,
                                                                      NodeFactory::reshapeNodeTypeName,
                                                                      fmt::format("{}_reshape_out", name));
            ret.insert(ret.end(), flattenNodes.begin(), flattenNodes.end());
            ret.emplace_back(
                NodeFactory::createInternalNode({flattenOut}, {reshapedDmaOutput}, nullptr, NodeFactory::transposeDmaNodeTypeName, name));
            ret.emplace_back(exctractTranspose);
            ret.emplace_back(reshapeAfterDma);
        }
        LOG_DEBUG(HABANA_NODE, "adding physical transpose node - {}", name);
    }

    return ret;
}

/*************************************************************************
 *                 TransposeViaFullyUtilizedDma Strategy
 *************************************************************************/
class TransposeViaFullyUtilizedDma : public TransposeStrategyStaticPriority<FULLY_UTILIZED_PRIORITY>
{
public:
    static constexpr uint32_t MAX_DESCRIPTOR_THRESHOLD = 500;  // each could have 64B/128B on FCD, so 64B * threshold
    static TransposePermutationArray    getFirstPermutation(const TransposePermutationArray& permutation);
    static TransposePermutationArray    getSecondPermutation(const TransposePermutationArray& permutation,
                                                             const NSizeArray&                sizes,
                                                             const DmaTransposeHelper&        helper);
    static TransposePermutationArrayVec splitPermutation(const TransposePermutationArray& permutation,
                                                         const NSizeArray&                input,
                                                         const DmaTransposeHelper&        helper);

    bool             canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector       extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    uint64_t         cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose fully utilizied DMA"; }

private:
    NodeVector createTransposeNode(const pTensor&                   in,
                                   const pTensor&                   out,
                                   const TransposePermutationArray& permutation,
                                   std::string_view                 name) const;
};

bool TransposeViaFullyUtilizedDma::canBeUsed(const TransposeNodeParams& transposeNodeParams,
                                             const HalReaderPtr&        hal) const
{
    TransposeViaGenericDma transposeStrategy;

    if (!transposeStrategy.canBeUsed(transposeNodeParams, hal))
    {
        return false;
    }

    const Tensor* input = transposeNodeParams.input.get();
    if (input->getDim() > TransposeNodesCreator::FULLY_UTILIZED_MAX_SUPPORTED_DIMS)
    {
        return false;
    }
    const auto&        sizes  = input->getAllSizesInElements();
    const auto&        params = hal->getDmaTransposeEngineParams();
    DmaTransposeHelper helper(input->getElementType(), params);
    if (input->getSizeInBytes(0) > helper.maxSrcDim0() * MAX_DESCRIPTOR_THRESHOLD)
    {
        return false;
    }
    auto numLines = helper.numLines(sizes.data());
    return numLines % params.numLinesDivisor == 0;
}

NodeVector TransposeViaFullyUtilizedDma::extract(const TransposeNodeParams& transposeNodeParams,
                                                 const HalReaderPtr&        hal) const
{
    const TensorPtr& input  = transposeNodeParams.input;
    const TensorPtr& output = transposeNodeParams.output;

    auto               params = hal->getDmaTransposeEngineParams();
    DmaTransposeHelper helper(input->getElementType(), params);

    const NSizeArray& sizeArray   = input->getAllNSizesInElements();
    auto              splitted    = splitPermutation(transposeNodeParams.permutation, sizeArray, helper);
    auto              inputTensor = input;
    NodeVector        ret;
    auto              newNodeName = fmt::format("{}_0", transposeNodeParams.nodeName);
    for (size_t i = 0; i < splitted.size(); i++)
    {
        auto transposeNodeOutput = output;
        if (i != splitted.size() - 1)
        {
            transposeNodeOutput = getTensorAfterTranspose(*inputTensor, splitted[i]);
        }
        auto more = createTransposeNode(inputTensor, transposeNodeOutput, splitted[i], newNodeName);
        ret.insert(ret.end(), more.begin(), more.end());
        inputTensor = transposeNodeOutput;
        // reduce temporary string creation for transpose node name
        if (i <= 9)
        {
            newNodeName.back()++;
        }
        else
        {
            auto newNodeName = fmt::format("{}_{}", transposeNodeParams.nodeName, i);
        }
    }

    return ret;
}

uint64_t TransposeViaFullyUtilizedDma::cost(const TransposeNodeParams& transposeNodeParams,
                                            const HalReaderPtr&        hal) const
{
    TensorPtr input = transposeNodeParams.input;

    auto               params = hal->getDmaTransposeEngineParams();
    DmaTransposeHelper helper(input->getElementType(), params);

    const NSizeArray& sizeArray       = input->getAllNSizesInElements();
    auto              newPermutations = splitPermutation(transposeNodeParams.permutation, sizeArray, helper);

    uint64_t cost = 0;
    for (const auto& permutation : newPermutations)
    {
        cost += m_costModel.getCost(input, permutation);
        input = getTensorAfterTranspose(*input, permutation);
    }
    return cost;
}

TransposePermutationArray
TransposeViaFullyUtilizedDma::getFirstPermutation(const TransposePermutationArray& permutation)
{
    TransposePermutationArray first = permutation;
    first.erase(std::remove(first.begin(), first.end(), TransposePermutationDim::TPD_Channel), first.end());
    first.insert(first.begin(), TransposePermutationDim::TPD_Channel);
    return first;
}

TransposePermutationArray
TransposeViaFullyUtilizedDma::getSecondPermutation(const TransposePermutationArray& permutation,
                                                   const NSizeArray&                sizes,
                                                   const DmaTransposeHelper&        helper)
{
    auto newFcdDim = getFcdIndex(permutation);
    while (helper.numLines(sizes.data(), newFcdDim + 1) % helper.params().numLinesDivisor != 0)
    {
        newFcdDim++;
        HB_ASSERT(newFcdDim < permutation.size(),
                  "num lines must be multiple of num lines divisor ({} % {} != 0)",
                  helper.numLines(sizes.data(), newFcdDim),
                  helper.params().numLinesDivisor);
    }
    TransposePermutationArray second = getIdentityPermutation(permutation.size());
    second.erase(second.begin());
    second.insert(second.begin() + newFcdDim, TransposePermutationDim::TPD_Channel);
    return second;
}

/**
 *
 * Split a complex permutation to three simple permutations
 * 1. Changing the order of the input dimensions that aren't the FCD to match the output's order.
 * 2. Moving the input's FCD to a different dimension (would look like NHCW, NCHW, or CNHW) according to rules
 * 3. (If we moved the FCD to a dimension not fit) change the FCD back to correct place.
 *
 * e.g for HNCW:
 * 1. NHWC -> HNWC -> HNCW (transposes:(0, 1, 3, 2), (1, 0, 2, 3))
 */
TransposePermutationArrayVec
TransposeViaFullyUtilizedDma::splitPermutation(const TransposePermutationArray& permutation,
                                               const NSizeArray&                input,
                                               const DmaTransposeHelper&        helper)
{
    TransposePermutationArrayVec ret;
    TransposePermutationArray    first = getFirstPermutation(permutation);
    if (first != getIdentityPermutation(permutation.size()))
    {
        ret.push_back(first);
    }
    auto sizesAfterFirstPermutation = applyPermutationOnSizes(input, first);  // applied with FCD first
    auto second                     = getSecondPermutation(permutation, sizesAfterFirstPermutation, helper);
    ret.push_back(second);
    TransposePermutationArray firstAndSecond = addPermutations(first, second);
    if (firstAndSecond != permutation)
    {
        auto third = subtractPermutations(permutation, firstAndSecond);
        ret.push_back(third);
    }
    return ret;
}

NodeVector TransposeViaFullyUtilizedDma::createTransposeNode(const pTensor&                   in,
                                                             const pTensor&                   out,
                                                             const TransposePermutationArray& permutation,
                                                             std::string_view                 name) const
{
    NodeVector ret;
    if (isSameDataMemOrder(*in, permutation))
    {
        return TransposeViaDMA::createReshapeWithExtractTranspose(
            in,
            out,
            permutation,
            fmt::format("{}_reshape_transpose_via_fully_utilized_dma", name));
    }
    else if (LogicalTransposeNode::isSupportedPermutation(*in, *out, permutation))
    {
        LOG_DEBUG(HABANA_NODE, "adding logical transpose node - {}", name);
        ret.emplace_back(
            NodeFactory::createInternalNode({in}, {out}, &permutation, NodeFactory::transposeLogicNodeTypeName, name));
        // The forward here is for that we won't transpose directly into an aliased tensor.
    }
    else
    {
        ret.emplace_back(NodeFactory::createInternalNode({in}, {out}, &permutation, NodeFactory::transposeDmaNodeTypeName, name));
        static_cast<DMATransposeNode*>(ret.back().get())->setIsFullyUtilized(true);
        LOG_DEBUG(HABANA_NODE, "adding physical transpose node - {}", name);
    }

    return ret;
}

/*************************************************************************
 *               DoubleTransposeViaFullyUtilizedDma Strategy
 *************************************************************************/
class DoubleTransposeViaFullyUtilizedDma : public TransposeStrategyStaticPriority<DOUBLE_TRANSPOSE_PRIORITY>
{
public:
    bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    uint64_t   cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    TransposePermutationArrayVec splitPermutation(TransposePermutationArray permutation) const;
    std::string_view             strategyName() const override { return "Double Transpose"; }
};

bool DoubleTransposeViaFullyUtilizedDma::canBeUsed(const TransposeNodeParams& transposeNodeParams,
                                                   const HalReaderPtr&        hal) const
{
    if (transposeNodeParams.preferTransposeOnlyOnce || !GCFG_ENABLE_DOUBLE_TRANSPOSE.value())
    {
        return false;  // ok
    }
    if (!TransposeViaFullyUtilizedDma {}.canBeUsed(transposeNodeParams, hal))
    {
        return false;
    }

    const TensorPtr&   input     = transposeNodeParams.input;
    const auto&        sizes     = input->getAllNSizesInElements();
    auto               params    = hal->getDmaTransposeEngineParams();
    DmaTransposeHelper helper(input->getElementType(), params);
    const auto&        permutation = transposeNodeParams.permutation;
    auto               newFcdIndex = getFcdIndex(permutation);
    if (newFcdIndex == input->getDim())
    {
        return false;
    }

    auto inputElementSizeInBytes = input->getElementSizeInBytes();
    int  basicNumLines           = inputElementSizeInBytes;
    for (int i = 0; i < newFcdIndex; i++)
    {
        basicNumLines *= sizes[permutation[i]];
    }
    if (basicNumLines % params.numLinesDivisor == 0)
    {
        // TODO: Move this to the cost calculations..
        return false;  // We shouldn't use this approach when we can perform it using only 1 transpose
    }

    // Calculate the numlines for the second transpose
    basicNumLines = inputElementSizeInBytes;
    for (int i = 0; i < input->getDim() - 1; i++)
    {
        basicNumLines *= sizes[permutation[i]];
    }
    if (basicNumLines % params.numLinesDivisor != 0)
    {
        return false;
    }
    // Check if it's more effecient, heuristic
    auto newFcdSize = input->getSizeInBytes(newFcdIndex);
    auto midFcdSize = input->getSizeInBytes(permutation.back());

    // check if the mid FCD size is valid
    if (midFcdSize > helper.maxSrcDim0() * TransposeViaFullyUtilizedDma::MAX_DESCRIPTOR_THRESHOLD)
    {
        return false;
    }

    if (newFcdSize < hal->getCacheLineSizeInBytes() / 4)
    {
        return (midFcdSize * 4 > newFcdSize);
    }
    auto correctedNew = newFcdSize - hal->getCacheLineSizeInBytes() / 4;
    auto correctedMid = midFcdSize - 5;  // batch
    return correctedMid * 8 > correctedNew;
}

uint64_t DoubleTransposeViaFullyUtilizedDma::cost(const TransposeNodeParams& transposeNodeParams,
                                                  const HalReaderPtr&        hal) const
{
    const TensorPtr& input           = transposeNodeParams.input;
    auto             newPermutations = splitPermutation(transposeNodeParams.permutation);
    auto             midTensor       = getTensorAfterTranspose(*input, newPermutations[0]);

    TransposeNodeParams first  = {input, midTensor, newPermutations[0]};
    TransposeNodeParams second = {midTensor, transposeNodeParams.output, newPermutations[1]};

    HB_ASSERT(TransposeViaFullyUtilizedDma().canBeUsed(first, hal), "Invalid node1 params for double transpose");
    HB_ASSERT(TransposeViaFullyUtilizedDma().canBeUsed(second, hal), "Invalid node2 params for double transpose");

    return TransposeViaFullyUtilizedDma().cost(first, hal) + TransposeViaFullyUtilizedDma().cost(second, hal);
}

NodeVector DoubleTransposeViaFullyUtilizedDma::extract(const TransposeNodeParams& transposeNodeParams,
                                                       const HalReaderPtr&        hal) const
{
    auto             permutations = splitPermutation(transposeNodeParams.permutation);
    const TensorPtr& input        = transposeNodeParams.input;
    auto             midTensor =
        getTensorAfterTranspose(*input, permutations[0], fmt::format("{}_mid_transpose", input->getName()));

    auto                newNodeName   = fmt::format("{}_splitted0", transposeNodeParams.nodeName);
    TransposeNodeParams st1NodeParams = {transposeNodeParams.input, midTensor, permutations[0], newNodeName};
    newNodeName.back()                = '1';  // reusing previous string instead of creating a new string
    TransposeNodeParams st2NodeParams = {midTensor, transposeNodeParams.output, permutations[1], newNodeName};

    HB_ASSERT(TransposeViaFullyUtilizedDma().canBeUsed(st1NodeParams, hal),
              "Invalid node1 params for double transpose");
    HB_ASSERT(TransposeViaFullyUtilizedDma().canBeUsed(st2NodeParams, hal),
              "Invalid node2 params for double transpose");

    auto result  = TransposeViaFullyUtilizedDma().extract(st1NodeParams, hal);
    auto result2 = TransposeViaFullyUtilizedDma().extract(st2NodeParams, hal);
    result.insert(result.end(), result2.begin(), result2.end());
    return result;
}

TransposePermutationArrayVec
DoubleTransposeViaFullyUtilizedDma::splitPermutation(TransposePermutationArray permutation) const
{
    TransposePermutationArrayVec ret;
    auto                         first = getIdentityPermutation(permutation.size());
    first.erase(std::remove(first.begin(), first.end(), TPD_Channel));
    first.push_back(TPD_Channel);
    first.erase(std::remove(first.begin(), first.end(), permutation.back()));
    first.insert(first.begin(), permutation.back());
    ret.push_back(first);
    auto second = subtractPermutations(permutation, first);
    ret.push_back(second);
    return ret;
}

/*************************************************************************
 *               TransposeViaDmaAndCasts Strategy
 *************************************************************************/
class TransposeViaDmaAndCasts : public TransposeStrategyStaticPriority<GENERIC_DMA_WITH_CASTS_PRIORITY>
{
public:
    bool             canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    bool             inferCastType(const Tensor* inputTensor, synDataType& castedType) const;
    NodeVector       extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    uint64_t         cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view strategyName() const override { return "Transpose int8 by dma and cast"; }

private:
    TensorPtr getTensorAfterCast(const TensorPtr& input, synDataType newDataType) const;
};

bool TransposeViaDmaAndCasts::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return transposeNodeParams.input->getElementSizeInBytes() == 1 &&
           hal->isDmaTransposeSupported(synDataType::syn_type_int16);  // will be casted
}

bool TransposeViaDmaAndCasts::inferCastType(const Tensor* inputTensor, synDataType& castedType) const
{
    static std::map<synDataType, synDataType> approvedCastTransitions {{syn_type_int8, syn_type_int16},
                                                                       {syn_type_uint8, syn_type_uint16},
                                                                       {syn_type_fp8_152, syn_type_bf16}};
    auto castTransition = approvedCastTransitions.find(inputTensor->getElementType());
    if (castTransition == approvedCastTransitions.end())
    {
        LOG_WARN(HABANA_NODE,
                 "Failed to find the appropriate cast operation for the given node: {}",
                 inputTensor->getElementType());
        return false;
    }
    castedType = castTransition->second;
    return true;
}

uint64_t TransposeViaDmaAndCasts::cost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return m_costModel.getCost(extract(transposeNodeParams, hal));
}

NodeVector TransposeViaDmaAndCasts::extract(const TransposeNodeParams& transposeNodeParams,
                                            const HalReaderPtr&        hal) const
{
    synDataType      castedType    = synDataType::syn_type_na;
    bool             paramsCreated = inferCastType(transposeNodeParams.input.get(), castedType);
    const TensorPtr& input         = transposeNodeParams.input;
    const TensorPtr& output        = transposeNodeParams.output;
    HB_ASSERT(paramsCreated,
              "Failed to infer cast parmeters for the provided Transpose node: {} with input data type {}",
              transposeNodeParams.nodeName,
              input->getElementType());

    auto newInput  = getTensorAfterCast(input, castedType);
    auto newOutput = getTensorAfterCast(output, castedType);

    TransposeNodeParams subtitueTransposeNodeParams = {newInput,
                                                       newOutput,
                                                       transposeNodeParams.permutation,
                                                       transposeNodeParams.nodeName};
    auto                nodeVector = TransposeViaDMA::skipPriorityCreateTransposeNodes(subtitueTransposeNodeParams,
                                                                        hal,
                                                                        GENERIC_DMA_WITH_CASTS_PRIORITY);
    NodePtr             castInputNode =
        CastNodeHandler::createCastNode(input, newInput, fmt::format("{}_cast_input", transposeNodeParams.nodeName));
    nodeVector.insert(nodeVector.begin(), castInputNode);

    NodePtr castOutputNode =
        CastNodeHandler::createCastNode(newOutput, output, fmt::format("{}_cast_output", transposeNodeParams.nodeName));
    nodeVector.emplace_back(castOutputNode);
    return nodeVector;
}

TensorPtr TransposeViaDmaAndCasts::getTensorAfterCast(const TensorPtr& input, synDataType newDataType) const
{
    if (input->getElementType() == newDataType)
    {
        return input;
    }

    TensorPtr casted = input->clone(false, false, false);
    casted->setElementType(newDataType);
    casted->setName(fmt::format("{}_casted_{}", input->getName(), getStringFromSynDataType(newDataType)));
    return casted;
}

/*************************************************************************
 *                          TransposeViaDma
 *************************************************************************/
TransposeStrategyVec TransposeViaDMA::initiateSubclasses(const TransposeNodeParams& transposeNodeParams,
                                                         const HalReaderPtr&        hal,
                                                         DmaTransposePriority       skipStaticPriority)
{
    static constexpr DoubleTransposeViaFullyUtilizedDma DOUBLE_TRANSPOSE_PRIORITY;
    static constexpr TransposeViaFullyUtilizedDma       FULLY_UTILIZED_PRIORITY;
    static constexpr TransposeViaGenericDma             GENERIC_DMA_PRIORITY;
    static constexpr TransposeViaDmaAndCasts            GENERIC_DMA_WITH_CASTS_PRIORITY;

    static const TransposeNodeParams emptyTransposeNodeParams;

    static constexpr auto GET_STRATEGY_PRIORITY_PAIR = [](const DmaTransposeStrategy& strategy) {
        return std::make_pair(&strategy, strategy.priority(emptyTransposeNodeParams, nullptr));
    };

    // strategies are sorted based on static priority to reduce later work in the sort phase
    static const TransposeStrategyVec ALL_STRATEGIES = {GET_STRATEGY_PRIORITY_PAIR(DOUBLE_TRANSPOSE_PRIORITY),
                                                        GET_STRATEGY_PRIORITY_PAIR(FULLY_UTILIZED_PRIORITY),
                                                        GET_STRATEGY_PRIORITY_PAIR(GENERIC_DMA_PRIORITY),
                                                        GET_STRATEGY_PRIORITY_PAIR(GENERIC_DMA_WITH_CASTS_PRIORITY)};

    TransposeStrategyVec filteredDmaStrategies;
    // keep the same order for the reduced strategies
    std::copy_if(ALL_STRATEGIES.begin(),
                 ALL_STRATEGIES.end(),
                 std::back_inserter(filteredDmaStrategies),
                 [skipStaticPriority](const StrategyPriorityPair& strategyPair) {
                     return strategyPair.second != skipStaticPriority;
                 });
    // update the cached prioriteis according to the node in question
    std::for_each(filteredDmaStrategies.begin(), filteredDmaStrategies.end(), [&](StrategyPriorityPair& x) {
        x.second = x.first->priority(transposeNodeParams, hal);
    });
    // sort the strategies based on priority
    std::sort(filteredDmaStrategies.begin(),
              filteredDmaStrategies.end(),
              [&](const StrategyPriorityPair& x, const StrategyPriorityPair& y) { return x.second < y.second; });

    return filteredDmaStrategies;
}

NodeVector TransposeViaDMA::skipPriorityCreateTransposeNodes(const TransposeNodeParams& transposeNodeParams,
                                                             const HalReaderPtr&        hal,
                                                             DmaTransposePriority       skipPriority)
{
    const auto& subClassesAndPriorities = initiateSubclasses(transposeNodeParams, hal, skipPriority);

    for (const auto& [dmaStrategy, priority] : subClassesAndPriorities)
    {
        if (priority != skipPriority && dmaStrategy->canBeUsed(transposeNodeParams, hal))
        {
            LOG_INFO(HABANA_NODE, "TransposeViaDMA: Selected \"{}\", for transposing", dmaStrategy->strategyName());
            return dmaStrategy->extract(transposeNodeParams, hal);
        }
    }
    HB_ASSERT(false, "No available dma strategy to transpose");
    return {};
}

NodeVector TransposeViaDMA::extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    if (transposeNodeParams.output->isAliasedTensor())
    {
        return handleAliasedOutputTranspose(transposeNodeParams, hal);
    }

    return skipPriorityCreateTransposeNodes(transposeNodeParams, hal);
}

uint64_t TransposeViaDMA::calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    uint64_t additionalCost = 0;
    if (transposeNodeParams.output->isAliasedTensor())
    {
        additionalCost = DmaTransposeCostModel().getDmaCost(transposeNodeParams.output);
    }

    const auto& subClassesAndPriorities = initiateSubclasses(transposeNodeParams, hal, UNUSED_PRIORITY);

    for (const auto& subClassAndPriority : subClassesAndPriorities)
    {
        if (subClassAndPriority.first->canBeUsed(transposeNodeParams, hal))
        {
            return subClassAndPriority.first->cost(transposeNodeParams, hal) + additionalCost;
        }
    }
    HB_ASSERT(false, "No available dma strategy to transpose");
    return 0;
}
