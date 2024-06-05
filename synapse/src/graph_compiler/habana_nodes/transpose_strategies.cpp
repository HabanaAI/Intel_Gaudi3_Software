#include "node_factory.h"
#include "kernel_db.h"
#include "data_type_utils.h"
#include "transpose_utils.h"

#include "transpose_strategies.h"
#include "dma_transpose_cost_model.h"

/*************************************************************************
 *                 TransposeViaTransposeShape Strategy
 *************************************************************************/
bool TransposeViaTransposeShape::canBeUsed(const TransposeNodeParams& transposeNodeParams,
                                           const HalReaderPtr&        hal) const
{
    return transposeNodeParams.input->isShapeTensor() && transposeNodeParams.output->isShapeTensor();
}

NodeVector TransposeViaTransposeShape::extract(const TransposeNodeParams& transposeNodeParams,
                                               const HalReaderPtr&        hal) const
{
    synTransposeParamsNDims params = permutationToParams(transposeNodeParams.permutation);
    NodePtr                 transposeShapeNode =
        NodeFactory::createInternalNode({transposeNodeParams.input},
                                        {transposeNodeParams.output},
                                        &params,
                                        NodeFactory::transposedShapeNodeTypeName,
                                        fmt::format("{}_transpose_via_transpose_shape", transposeNodeParams.nodeName));
    return {transposeShapeNode};
}

std::string_view TransposeViaTransposeShape::strategyName() const
{
    return "Transpose by Transpose shape";
}

TransposeStrategyID TransposeViaTransposeShape::getStrategyID() const
{
    return TRANSPOSE_SHAPE;
}

/*************************************************************************
 *                    TransposeViaReshape Strategy
 *************************************************************************/
bool TransposeViaReshape::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return isSameDataMemOrder(*transposeNodeParams.input, transposeNodeParams.permutation);
}

NodeVector TransposeViaReshape::extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return createReshapeWithExtractTranspose(transposeNodeParams.input,
                                             transposeNodeParams.output,
                                             transposeNodeParams.permutation,
                                             fmt::format("{}_transpose_via_reshape", transposeNodeParams.nodeName));
}

std::string_view TransposeViaReshape::strategyName() const
{
    return "Transpose by reshape";
}

TransposeStrategyID TransposeViaReshape::getStrategyID() const
{
    return TRANSPOSE_VIA_RESHAPE;
}

/*************************************************************************
 *                   TransposeViaLogical Strategy
 *************************************************************************/
uint64_t TransposeViaLogical::calculateCost(const TransposeNodeParams& transposeNodeParams,
                                            const HalReaderPtr&        hal) const
{
    return TransposeNodesCreator().getCostModel()->getCost(transposeNodeParams.input, transposeNodeParams.permutation);
}

bool TransposeViaLogical::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    return LogicalTransposeNode::isSupportedPermutation(*transposeNodeParams.input,
                                                        *transposeNodeParams.output,
                                                        transposeNodeParams.permutation);
}

NodeVector TransposeViaLogical::extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    NodeVector ret;
    ret.emplace_back(NodeFactory::createInternalNode({transposeNodeParams.input},
                                                     {transposeNodeParams.output},
                                                     &(transposeNodeParams.permutation),
                                                     "transpose_logic",
                                                     transposeNodeParams.nodeName.value_or("noName")));
    return ret;
}

std::string_view TransposeViaLogical::strategyName() const
{
    return "Transpose by logical";
}

TransposeStrategyID TransposeViaLogical::getStrategyID() const
{
    return LOGICAL_TRANSPOSE;
}

/*************************************************************************
 *                   TransposeWithStaticShape Strategy
 *************************************************************************/
bool TransposeWithStaticShape::canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const
{
    if (!GCFG_ENABLE_STATIC_TRANSPOSE_STRATEGY.value()) return false;
    if (!transposeNodeParams.input->isDynamicShape() && !transposeNodeParams.output->isDynamicShape()) return false;
    NodeVector newNodes = TransposeNodesCreator().getTransposeNodesByParams(transposeNodeParams, STATIC_TRANSPOSE);
    for (const NodePtr& newNode : newNodes)
    {
        // exists physical node that is not transpose node
        if (!newNode->isLogicalOperation() && !newNode->isTranspose()) return true;
    }
    return false;
}

NodeVector TransposeWithStaticShape::extract(const TransposeNodeParams& transposeNodeParams,
                                             const HalReaderPtr&        hal) const
{
    const auto& aux = createAuxilaryNodes(transposeNodeParams);

    NodeVector ret = createTransposeNodes(transposeNodeParams, aux.newTransposeInput, aux.newTransposeOutput);
    ret.push_back(aux.inferMaxNode);
    ret.push_back(aux.transposeShapeNode);
    ret.push_back(aux.identityNode);
    return ret;
}

TransposeWithStaticShape::AuxilaryNodeData
TransposeWithStaticShape::createAuxilaryNodes(const TransposeNodeParams& transposeNodeParams) const
{
    const auto& input  = transposeNodeParams.input;
    const auto& output = transposeNodeParams.output;
    HB_ASSERT_PTR(input);
    HB_ASSERT_PTR(output);

    // Guarantee identity node's output tensor is real to buffer and therefore prevent writing max size data
    // to an alias tensor expecting actual size data
    output->setIsRealInLogical(true);
    auto [newInput, newOutput, inputShape, outputShape] = createNewTensors(input, output);

    AuxilaryNodeData res;
    res.newTransposeInput  = newInput;
    res.newTransposeOutput = newOutput;
    res.inferMaxNode       = createInferMaxShapeNode(input, newInput, inputShape);
    res.transposeShapeNode = createTransposeShapeNode(transposeNodeParams.permutation, inputShape, outputShape);
    res.identityNode       = createIdentityNode(output, newOutput, outputShape);
    return res;
}

std::string_view TransposeWithStaticShape::strategyName() const
{
    return "Transpose with static shape";
}

TransposeStrategyID TransposeWithStaticShape::getStrategyID() const
{
    return STATIC_TRANSPOSE;
}

NodePtr TransposeWithStaticShape::createInferMaxShapeNode(const TensorPtr& input,
                                                          const TensorPtr& newInput,
                                                          const TensorPtr& inputShape) const
{
    HB_ASSERT_PTR(input);
    HB_ASSERT_PTR(newInput);
    HB_ASSERT_PTR(inputShape);
    return NodeFactory::createInternalNode({input},
                                           {newInput, inputShape},
                                           nullptr,
                                           NodeFactory::inferMaxShapeNodeTypeName,
                                           fmt::format("{}/infer_max_shape", input->getName()));
}

NodeVector TransposeWithStaticShape::createTransposeNodes(const TransposeNodeParams& transposeNodeParams,
                                                          const TensorPtr&           newInput,
                                                          const TensorPtr&           newOutput) const
{
    HB_ASSERT_PTR(newInput);
    HB_ASSERT_PTR(newOutput);

    TransposeNodeParams newTransposeNodeParams = {newInput,
                                                  newOutput,
                                                  transposeNodeParams.permutation,
                                                  transposeNodeParams.nodeName,
                                                  transposeNodeParams.preferLogicalBeforePhysical,
                                                  transposeNodeParams.preferTransposeOnlyOnce};
    return TransposeNodesCreator().getTransposeNodesByParams(newTransposeNodeParams, STATIC_TRANSPOSE);
}

NodePtr TransposeWithStaticShape::createTransposeShapeNode(const TransposePermutationArray& permutation,
                                                           const TensorPtr&                 inputShape,
                                                           const TensorPtr&                 outputShape) const
{
    HB_ASSERT_PTR(inputShape);
    HB_ASSERT_PTR(outputShape);

    synTransposeParamsNDims params = permutationToParams(permutation);
    return NodeFactory::createInternalNode({inputShape},
                                           {outputShape},
                                           &params,
                                           NodeFactory::transposedShapeNodeTypeName,
                                           fmt::format("{}/transpose_shape", inputShape->getName()));
}

NodePtr TransposeWithStaticShape::createIdentityNode(const TensorPtr& output,
                                                     const TensorPtr& newOutput,
                                                     const TensorPtr& outputShape) const
{
    HB_ASSERT_PTR(output);
    HB_ASSERT_PTR(newOutput);
    HB_ASSERT_PTR(outputShape);
    return NodeFactory::createInternalNode({newOutput, outputShape},
                                           {output},
                                           nullptr,
                                           NodeFactory::identityNodeTypeName,
                                           fmt::format("{}/insert_shape", output->getName()));
}

// return <newInput, newOutput, inputShape, outputShape>
std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr>
TransposeWithStaticShape::createNewTensors(const TensorPtr& input, const TensorPtr& output) const
{
    HB_ASSERT_PTR(input);
    HB_ASSERT_PTR(output);
    NStrideArray inputStrides  = {1};
    NStrideArray outputStrides = {1};
    input->getNStridesInBytes(inputStrides.data());
    output->getNStridesInBytes(outputStrides.data());

    const auto& inputMaxSizes  = input->getAllNSizesInElements();
    const auto& outputMaxSizes = output->getAllNSizesInElements();

    TensorPtr newInput = input->clone(false, false, false);
    newInput->setName(fmt::format("{}/static_shape", input->getName()));
    newInput->reshape(input->getDim(), inputMaxSizes.data(), inputStrides.data(), inputMaxSizes.data());

    TensorPtr newOutput = output->clone(false, false, false);
    newOutput->setName(fmt::format("{}/static_shape", output->getName()));
    newOutput->reshape(output->getDim(), outputMaxSizes.data(), outputStrides.data(), outputMaxSizes.data());

    TensorPtr inputShape = input->clone(false, false, false);
    inputShape->setName(fmt::format("{}/shape", input->getName()));
    inputShape->setElementType(syn_type_int32);
    inputShape->setShapeTensor(SHAPE_TENSOR);

    TensorPtr outputShape = output->clone(false, false, false);
    outputShape->setName(fmt::format("{}/shape", output->getName()));
    outputShape->setElementType(syn_type_int32);
    outputShape->setShapeTensor(SHAPE_TENSOR);

    return std::make_tuple(newInput, newOutput, inputShape, outputShape);
}

// There is no additional cost by using this strategy.
uint64_t TransposeWithStaticShape::calculateCost(const TransposeNodeParams& transposeNodeParams,
                                                 const HalReaderPtr&        hal) const
{
    return TransposeNodesCreator().getTransposeCostByParams(transposeNodeParams, STATIC_TRANSPOSE);
}