#include "dynamic_reshape_shape_node.h"
#include "fcd_ops_utils.h"
#include "hal_reader/hal_reader.h"
#include "log_manager.h"
#include "mme_services.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "transpose_node.h"
#include "transpose_permutation.h"
#include "transpose_utils.h"
#include "types.h"
#include "utils.h"

#include "transpose_via_mme.h"
#include <include/mme_common/mme_common_enum.h>
#include <string_view>

NodeVector TransposeViaMME::createTranspose(const TransposeNodeParams& params, const HalReaderPtr& hal) const
{
    NodeVector nodes;
    if (PhysicalTransposeStrategy {}.canBeUsed(params, hal))
    {
        nodes = PhysicalTransposeStrategy {}.extract(params, hal);
    }
    else
    {
        const auto logicTransposeStrategy = getLogicTransposeStrategy(params, hal);
        HB_ASSERT_PTR(logicTransposeStrategy);
        nodes = logicTransposeStrategy->extract(params, hal);
    }
    return nodes;
}

NodeVector TransposeViaMME::createTransposes(const TransposeNodeParams&          origParams,
                                             const TransposePermutationArrayVec& permutations,
                                             const HalReaderPtr&                 hal) const
{
    NodeVector          nodes;
    TransposeNodeParams firstTranspose(origParams);
    firstTranspose.permutation = permutations.front();
    firstTranspose.input       = origParams.input;
    firstTranspose.output      = getTensorAfterTranspose(*firstTranspose.input,
                                                    firstTranspose.permutation,
                                                    firstTranspose.nodeName.value_or("NoName"));

    TransposeNodeParams secondTranspose(origParams);
    secondTranspose.permutation     = permutations.back();
    secondTranspose.input           = firstTranspose.output;
    secondTranspose.output          = origParams.output;
    const auto firstTransposeNodes  = createTranspose(firstTranspose, hal);
    const auto secondTransposeNodes = createTranspose(secondTranspose, hal);
    nodes.insert(nodes.end(), firstTransposeNodes.begin(), firstTransposeNodes.end());
    nodes.insert(nodes.end(), secondTransposeNodes.begin(), secondTransposeNodes.end());
    return nodes;
}

const TransposeNodeStrategy* TransposeViaMME::getLogicTransposeStrategy(const TransposeNodeParams& logicTransposeParams,
                                                                        const HalReaderPtr&        hal) const
{
    const auto strategyIt = std::find_if(
        m_logicStrategies.begin(),
        m_logicStrategies.end(),
        [&hal, &logicTransposeParams](const auto& strategy) { return strategy->canBeUsed(logicTransposeParams, hal); });
    HB_ASSERT(strategyIt != m_logicStrategies.end(),
              "Expecting at least one logical transpose strategy to be applicable");
    return *strategyIt;
}

bool TransposeViaMME::canBeUsed(const TransposeNodeParams& params, const HalReaderPtr& hal) const
{
    return hal->isMmeTransposeSupported();
}

TransposePermutationArrayVec TransposeViaMME::splitPermutation(const TransposePermutationArray& perm,
                                                               bool                             reversed) const
{
    const auto effPerm(TransposeViaNativeMME::getEffectivePermutation(perm));
    // calculate split permutations for the effective part of the input permutation
    auto permutations(reversed ? splitLogicalBeforePhysical(effPerm) : ::splitPermutation(effPerm));
    HB_ASSERT(permutations.size() == 2, "Expecting physical and logical permutations");
    // align split permutations to original rank
    for (auto idx = effPerm.size(); idx < perm.size(); ++idx)
    {
        const auto dim = static_cast<TransposePermutationDim>(idx);
        permutations.front().push_back(dim);
        permutations.back().push_back(dim);
    }
    return permutations;
}

NodeVector TransposeViaMME::extract(const TransposeNodeParams& params, const HalReaderPtr& hal) const
{
    if (PhysicalTransposeStrategy {}.canBeUsed(params, hal))
    {
        LOG_DEBUG(GC,
                  "TransposeViaMME extract physical: name: {}, input: {}, permutation: [{}], output: {}",
                  params.nodeName.value_or("NoName"),
                  params.input->getDimSizesStr(),
                  toString(params.permutation, ','),
                  params.output->getDimSizesStr());
        // Permutation can be handled by native MME transpose strategy
        return PhysicalTransposeStrategy {}.extract(params, hal);
    }
    else
    {
        // Split the effective permutation into two permutations, a
        // physical transpose and the complementing logical transpose.
        NodeVector nodes;

        const auto preferLogicBeforePhysial = params.input->getSizeInElements(0) > params.output->getSizeInElements(0);
        TransposePermutationArrayVec permutations = splitPermutation(params.permutation, preferLogicBeforePhysial);
        LOG_DEBUG(
            GC,
            "TransposeViaMME extract split: name: {}, input: {}, orig permutation: [{}], preferLogicBeforePhysical: "
            "{}, 1st "
            "permutation: [{}], 2nd permutation: [{}], output: {}",
            params.nodeName.value_or("NoName"),
            params.input->getDimSizesStr(),
            toString(params.permutation, ','),
            preferLogicBeforePhysial,
            toString(permutations.front(), ','),
            toString(permutations.back(), ','),
            params.output->getDimSizesStr());
        nodes = createTransposes(params, permutations, hal);
        return nodes;
    }
}

TransposePermutationArray TransposeViaNativeMME::getEffectivePermutation(const TransposePermutationArray& perm)
{
    return TransposePermutationArray(perm.begin(), perm.begin() + getLastPermutedDimIdx(perm) + 1);
}

bool TransposeViaNativeMME::supported(const TransposePermutationArray& permutation)
{
    if (identityPermutation(permutation)) return false;
    // MME can transpose dims 0 and 1.
    // If the permutation[0,last non identity dim] denoted effective permutation is a
    // cyclic permutation, the operands can be reshaped to achieve a supported transpose.
    const auto effectivePerm(TransposeViaNativeMME::getEffectivePermutation(permutation));
    if (!cyclicPermutation(effectivePerm)) return false;
    const auto loweredRank =
        (permutation.size() - effectivePerm.size()) /*identity dims*/ + 2 /*physical transpose dims*/;
    HB_ASSERT(loweredRank >= 2, "Expecting lowered transpose rank {} >= 2", loweredRank);
    return true;
}

bool TransposeViaNativeMME::canBeUsed(const TransposeNodeParams& params, const HalReaderPtr& hal) const
{
    return hal->isMmeTransposeSupported() && supported(params.permutation);
}

TransposePermutationArray TransposeViaNativeMME::getTransposePermutation(unsigned rank) const
{
    HB_ASSERT(rank >= 2, "Expecting rank {} >= 2", rank);
    TransposePermutationArray permutation(rank);
    // Supported transpose of dims 1 and 0
    permutation.at(0) = TransposePermutationDim::TPD_Width;
    permutation.at(1) = TransposePermutationDim::TPD_Channel;
    // Remaining dims 2,...,rank-1 are identity
    for (auto idx = 2; idx < rank; ++idx)
    {
        permutation.at(idx) = static_cast<TransposePermutationDim>(idx);
    }
    return permutation;
}

TransposeViaNativeMME::MaxMinSizes TransposeViaNativeMME::getLoweredSizes(const TensorPtr&                 in,
                                                                          const TransposePermutationArray& perm) const
{
    HB_ASSERT_PTR(in);
    HB_ASSERT(in->getDim() == perm.size(),
              "Expecting tensor {} dim {} == perm.size() {}",
              in->getName(),
              in->getDim(),
              perm.size());
    const auto effectivePerm(getEffectivePermutation(perm));
    HB_ASSERT(cyclicPermutation(effectivePerm),
              "Expecting a cyclic effective permutation=[{}]",
              toString(effectivePerm, ','));
    TransposeViaNativeMME::MaxMinSizes maxMinSizes;
    const auto                         origMaxSizes = in->getAllNSizesInElements();
    const auto                         origMinSizes = in->getNMinimalSizesInElements();
    const auto                         newFcd       = effectivePerm.front();
    // dim0 = product of dims [0,newFcd)
    maxMinSizes.first.push_back(multiplyElements(origMaxSizes.begin(), origMaxSizes.begin() + newFcd));
    maxMinSizes.second.push_back(multiplyElements(origMinSizes.begin(), origMinSizes.begin() + newFcd));
    // dim1 = product of dims [newFcd, lastPermutedDim]
    maxMinSizes.first.push_back(
        multiplyElements(origMaxSizes.begin() + newFcd, origMaxSizes.begin() + effectivePerm.size()));
    maxMinSizes.second.push_back(
        multiplyElements(origMinSizes.begin() + newFcd, origMinSizes.begin() + effectivePerm.size()));
    // [lastPermutedDim+1,rank-1] are left untouched
    const auto loweredTransposeInputRank = in->getDim() - effectivePerm.size() + 2;
    for (auto dim = effectivePerm.size(); dim < in->getDim(); ++dim)
    {
        maxMinSizes.first.push_back(origMaxSizes.at(dim));
        maxMinSizes.second.push_back(origMinSizes.at(dim));
    }

    if (loweredTransposeInputRank > SYN_MAX_TENSOR_DIM)
    {
        // Aggregate slow dimensions into last MME supported dim
        maxMinSizes.first.at(SYN_MAX_TENSOR_DIM - 1) *=
            multiplyElements(origMaxSizes.begin() + SYN_MAX_TENSOR_DIM, origMaxSizes.end());
        maxMinSizes.second.at(SYN_MAX_TENSOR_DIM - 1) *=
            multiplyElements(origMinSizes.begin() + SYN_MAX_TENSOR_DIM, origMinSizes.end());
        maxMinSizes.first.resize(SYN_MAX_TENSOR_DIM);
        maxMinSizes.second.resize(SYN_MAX_TENSOR_DIM);
    }

    return maxMinSizes;
}

std::optional<std::tuple<NodeVector, TensorPtr, TensorPtr>>
TransposeViaNativeMME::createLoweringSequence(const TensorPtr&                 in,
                                              const TensorPtr&                 out,
                                              const TransposePermutationArray& perm) const
{
    HB_ASSERT_PTR(in);
    HB_ASSERT_PTR(out);
    NodeVector newNodes;
    if (getLastPermutedDimIdx(perm) < 2 && !isHighRank(in))
    {
        // Lowering isn't required
        return std::nullopt;
    }
    const auto maxMinSizes                       = getLoweredSizes(in, perm);
    const auto& [maxLoweredSize, minLoweredSize] = maxMinSizes;
    // create new transpose operands
    const auto loweredInputRank = maxLoweredSize.size();
    const auto newTransposeIn   = in->clone(false, false, false);
    newTransposeIn->reshape(loweredInputRank, maxLoweredSize.data(), nullptr, minLoweredSize.data());

    const auto newPerm = getTransposePermutation(newTransposeIn->getDim());
    const auto newTransposeOut =
        getTensorAfterTranspose(*newTransposeIn, newPerm, fmt::format("{}_transposed", newTransposeIn->getName()));

    TensorVector firstReshapeInputs {in}, secondReshapeInputs {newTransposeOut};
    if (in->isDynamicShape() || out->isDynamicShape())
    {
        const auto [extractShape, reshapeShape, transposeShape] = createTransposeLoweringShapes(in, maxMinSizes, perm);
        firstReshapeInputs.push_back(reshapeShape->getOutput(0));
        secondReshapeInputs.push_back(transposeShape->getOutput(0));
        newNodes.push_back(extractShape);
        newNodes.push_back(reshapeShape);
        newNodes.push_back(transposeShape);
    }
    const auto reshapeBeforeMmeTranspose =
        NodeFactory::createInternalNode(firstReshapeInputs,
                                        {newTransposeIn},
                                        nullptr,
                                        NodeFactory::reshapeNodeTypeName,
                                        fmt::format("{}_1st_reshape", newTransposeIn->getName()));
    HB_ASSERT_PTR(reshapeBeforeMmeTranspose);
    newNodes.push_back(reshapeBeforeMmeTranspose);

    const auto reshapeAfterMmeTranspose =
        NodeFactory::createInternalNode(secondReshapeInputs,
                                        {out},
                                        nullptr,
                                        NodeFactory::reshapeNodeTypeName,
                                        fmt::format("{}_2nd_reshape", newTransposeOut->getName()));
    HB_ASSERT_PTR(reshapeAfterMmeTranspose);
    newNodes.push_back(reshapeAfterMmeTranspose);
    return std::make_tuple(newNodes, newTransposeIn, newTransposeOut);
}

NodePtr
TransposeViaNativeMME::createMmeTranspose(const TensorPtr& in, const TensorPtr& out, const std::string& name) const
{
    const auto perm          = getTransposePermutation(in->getDim());
    MMENodePtr transposeNode = std::static_pointer_cast<MmeTransposeNode>(
        NodeFactory::createInternalNode({in},
                                        {out},
                                        &perm,
                                        NodeFactory::transposeMmeNodeTypeName,
                                        fmt::format("{}_native_mme_transpose", name)));
    HB_ASSERT_PTR(transposeNode);

    // check if transpose via gemm and add unit matrix if needed.
    MmeCommon::MmeServices services;
    services.addAuxTensorToNode(transposeNode, MmeCommon::MmeStrategy {} /*strategy not needed for transpose nodes*/);
    return transposeNode;
}

std::string TransposeViaNativeMME::createTransposeLoweringEinsumEquation(unsigned dim,
                                                                         unsigned newFcd,
                                                                         unsigned lastPermutedDim) const
{
    constexpr char    fcd = 'a';
    std::stringstream loweringEquation;
    for (auto idx = 0; idx < dim; ++idx)
    {
        const char       dimChar   = fcd + idx;
        std::string_view separator = (idx == dim - 1) ? "->" : ",";
        loweringEquation << dimChar << separator;
    }

    for (auto idx = 0; idx < dim - 1; ++idx)
    {
        const char       dimChar = fcd + idx;
        std::string_view separator;
        separator = (idx < newFcd - 1 || (idx >= newFcd && idx < lastPermutedDim) || (idx >= (lastPermutedDim + 3)))
                        ? "*"
                        : ",";
        loweringEquation << dimChar << separator;
    }
    loweringEquation << static_cast<char>(fcd + (dim - 1));
    return loweringEquation.str();
}

bool TransposeViaNativeMME::isHighRank(const TensorPtr& t) const
{
    HB_ASSERT_PTR(t);
    return t->getDim() > SYN_MAX_TENSOR_DIM;
}

std::tuple<NodePtr, NodePtr, NodePtr>
TransposeViaNativeMME::createTransposeLoweringShapes(const TensorPtr&                   transposeIn,
                                                     TransposeViaNativeMME::MaxMinSizes loweredSizes,
                                                     const TransposePermutationArray&   perm) const
{
    const auto extractShape = FcdOpsUtils::createExtractShapeNode(transposeIn);
    HB_ASSERT_PTR(extractShape);
    const auto& extractedShapeOutput = extractShape->getOutput(0);
    const dynamicReshapeParams params {.equation = createTransposeLoweringEinsumEquation(extractedShapeOutput->getDim(),
                                                                                         perm.front(),
                                                                                         getLastPermutedDimIdx(perm))};
    LOG_DEBUG(GC,
              "{}: LoweredMaxSizes: [{}], LoweredMinSizes: [{}], Reshape shape lowering equation: {}",
              HLLOG_FUNC,
              toString(loweredSizes.first, ','),
              toString(loweredSizes.second, ','),
              params.equation);

    const auto reshapeShapeOutput = extractedShapeOutput->clone(false, false, false);
    reshapeShapeOutput->reshape(loweredSizes.first.size(),
                                loweredSizes.first.data(),
                                nullptr,
                                loweredSizes.second.data());

    const auto reshapeShape = NodeFactory::createNode({extractedShapeOutput},
                                                      {reshapeShapeOutput},
                                                      &params,
                                                      NodeFactory::dynamicReshapeNodeTypeName,
                                                      fmt::format("{}_reshape_shape", transposeIn->getName()));
    HB_ASSERT_PTR(reshapeShape);
    const auto transposeShape = FcdOpsUtils::createTransposedShape(extractedShapeOutput, perm);
    HB_ASSERT_PTR(transposeShape);
    return std::make_tuple(extractShape, reshapeShape, transposeShape);
}

NodeVector TransposeViaNativeMME::extract(const TransposeNodeParams& params, const HalReaderPtr& hal) const
{
    NodeVector newNodes;
    TensorPtr  transposeIn  = params.input;
    TensorPtr  transposeOut = params.output;

    if (const auto loweredSequence = createLoweringSequence(transposeIn, transposeOut, params.permutation);
        loweredSequence.has_value())
    {
        // requires lowering, flatten via reshapes
        const auto [reshapes, newTransposeIn, newTransposeOut] = loweredSequence.value();
        transposeIn                                            = newTransposeIn;
        transposeOut                                           = newTransposeOut;
        newNodes.insert(newNodes.end(), reshapes.begin(), reshapes.end());
    }
    const auto mmeTranspose = createMmeTranspose(transposeIn, transposeOut, params.nodeName.value_or("NoName"));

    newNodes.push_back(mmeTranspose);
    return newNodes;
}