#include "fcd_ops_utils.h"
#include "tensor.h"
#include "transpose_utils.h"
#include "types.h"
#include "utils.h"
#include "node_factory.h"
#include "dynamic_reshape_shape_node.h"
#include <cstdint>
#include <string>

using namespace FcdOpsUtils;

uint64_t FcdOpsUtils::getOptimalCost(const Tensor& tensor)
{
    return tensor.getDenseSizeInBytes();
}

uint64_t FcdOpsUtils::calculateExpectedCost(uint64_t optimalCost, unsigned cacheLineSizeInBytes, uint64_t fcdSize)
{
    // shape tensor not cost anything
    if (optimalCost == 0) return 0;
    if (fcdSize % cacheLineSizeInBytes == 0) return optimalCost;
    uint64_t fullUtilization = fcdSize + cacheLineSizeInBytes - (fcdSize % cacheLineSizeInBytes);
    return (optimalCost * fullUtilization) / fcdSize;
}

uint64_t FcdOpsUtils::calculateExpectedCost(const Tensor& tensor, unsigned cacheLineSizeInBytes, uint64_t fcdSize)
{
    // shape tensor not cost anything
    if (tensor.isShapeTensor()) return 0;
    return calculateExpectedCost(getOptimalCost(tensor), cacheLineSizeInBytes, fcdSize);
}

static NodePtr createFlattenByReshape(const TensorVector& inputs, unsigned axis, std::string_view name)
{
    const TensorPtr& input = inputs[0];
    unsigned         dims  = input->getDim();
    HB_ASSERT(axis < dims, "axis is larger than tensor dim, axis: {}, dim: {}", axis, dims);

    TensorPtr reshapeOutput = createFlattenedTensor(input, axis);
    NodePtr   reshapeNode =
        NodeFactory::createNode(inputs, {reshapeOutput}, nullptr, NodeFactory::reshapeNodeTypeName, name);
    return reshapeNode;
}

// return the shape nodes and shape tensors for reshapes
static std::tuple<NodeList, TensorPtr, TensorPtr>
createShapeTensorsForReshapes(const TensorPtr& input, const unsigned axis, const TransposePermutationArray& perm)
{
    NodeList  shapeNodes;
    NodePtr   extractShape = createExtractShapeNode(input);
    shapeNodes.push_back(extractShape);

    NodePtr   flattenShapeNode = createFlattenShapeNode(extractShape->getOutput(0), axis);
    TensorPtr firstShape       = flattenShapeNode->getOutput(0);
    shapeNodes.push_back(flattenShapeNode);

    NodePtr   transposeShapeNode = createTransposedShape(extractShape->getOutput(0), perm);
    TensorPtr secondShape        = transposeShapeNode->getOutput(0);
    shapeNodes.push_back(transposeShapeNode);

    return std::tie(shapeNodes, firstShape, secondShape);
}

static NodePtr createTwoDimTranspose(const TensorPtr& tensor, bool fromInput, std::string_view name)
{
    TransposePermutationArray twoDimsPermutation(2);
    twoDimsPermutation[0] = (TransposePermutationDim)1;
    twoDimsPermutation[1] = (TransposePermutationDim)0;

    synTransposeParamsNDims twoDimsTransposeParams = permutationToParams(twoDimsPermutation);
    TensorPtr               newTensor              = getTensorAfterTranspose(*tensor, twoDimsPermutation);

    return NodeFactory::createNode({fromInput ? tensor : newTensor},
                                   {fromInput ? newTensor : tensor},
                                   &twoDimsTransposeParams,
                                   NodeFactory::transposeNodeTypeName,
                                   fmt::format("{}/transpose", name));
}

static TransposePermutationArray getShiftTransposePermutation(const unsigned tensorDim, const unsigned axis)
{
    TransposePermutationArray permutation;
    for (unsigned dim = 0; dim < tensorDim; ++dim)
    {
        permutation.push_back((TransposePermutationDim)((dim + axis + 1) % tensorDim));
    }
    return permutation;
}

// create shift transpose with: reshape the n-dim input into 2-dim -> transpose -> reshape back to n-dim
// example [A, B, C, D, E], axis 2:
// [A, B, C, D, E] -> [A * B * C, D * E] -> [D * E, A * B * C] -> [D, E, A, B, C]
static NodeList
createShiftTransposeSequence(const TensorPtr& tensor, unsigned axis, bool fromInput, std::string_view name)
{
    unsigned dims = tensor->getDim();
    if (dims == 2)
    {
        return {createTwoDimTranspose(tensor, fromInput, name)};
    }

    TransposePermutationArray permutation    = getShiftTransposePermutation(dims, axis);
    TransposePermutationArray invPermutation = inversePermutation(permutation);

    if (!fromInput)
    {
        std::swap(permutation, invPermutation);
    }
    TensorPtr input  = (fromInput) ? tensor : getTensorAfterTranspose(*tensor, invPermutation);
    TensorPtr output = (fromInput) ? getTensorAfterTranspose(*tensor, permutation) : tensor;

    HB_ASSERT(input->getDim() >= axis + 2, "can't do utilized transpose on the SCD");
    axis = (fromInput) ? axis : dims - axis - 2;

    // create the first reshape

    TensorVector secondReshapeInputs(1);
    TensorVector firstReshapeInputs = {input};
    NodeList     ret;
    // in case of dynamic shape we need to create shape tensors for the reshape
    // nodes
    if (input->isDynamicShape())
    {
        TensorPtr firstShape;
        TensorPtr secondShape;
        std::tie(ret, firstShape, secondShape) = createShapeTensorsForReshapes(input, axis, permutation);
        firstReshapeInputs.push_back(firstShape);
        secondReshapeInputs.push_back(secondShape);
    }

    ret.push_back(
        createFlattenByReshape(firstReshapeInputs, axis, fmt::format("{}/first_reshape_{}", name, input->getId())));
    const TensorPtr& flattenOutput = ret.back()->getOutput(0);

    NodePtr transposeNode = createTwoDimTranspose(flattenOutput, true, name);
    ret.push_back(transposeNode);

    // create the second reshape
    secondReshapeInputs[0] = transposeNode->getOutput(0);
    NodePtr reshapeNode    = NodeFactory::createNode(secondReshapeInputs,
                                                  {output},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  fmt::format("{}/second_reshape_{}", name, input->getId()));
    ret.push_back(reshapeNode);
    return ret;
}

void FcdOpsUtils::findBestDimForTranspose(const HalReader&                 hal,
                                          const unsigned                   startFromDim,
                                          const TensorVector&              inputs,
                                          const TensorVector&              outputs,
                                          ShiftTransposesForFcdOpsResults& res /*inplace modify */)
{
    unsigned  cacheLineSize = hal.getCacheLineSizeInBytes();
    TStride   elementSize   = inputs[0]->getElementSizeInBytes();

    TensorStridesVector inputSizesBefore(inputs.size(), 1);
    TensorStridesVector inputSizesAfter;
    TensorStridesVector inputOptimalCost;
    unsigned numOfDataInputs  = 0;
    for (unsigned i = 0; i < inputs.size(); ++i)
    {
        if (inputs[i]->isShapeTensor() || inputs[i]->isHost2DeviceTensor()) continue;
        HB_ASSERT(i == numOfDataInputs, "found data tensor after shape tensor");
        ++numOfDataInputs;
        if (inputs[i]->isZeroSizedDataTensor()) return;
        uint64_t denseSizeInElements = inputs[i]->getDenseSizeInElements();
        inputSizesAfter.push_back(denseSizeInElements);
        inputOptimalCost.push_back(denseSizeInElements * inputs[i]->getElementSizeInBytes());
    }

    TensorStridesVector outputSizesBefore(outputs.size(), 1);
    TensorStridesVector outputSizesAfter;
    TensorStridesVector outputOptimalCost;
    unsigned            numOfDataOutputs = 0;
    for (unsigned i = 0; i < outputs.size(); ++i)
    {
        if (outputs[i]->isShapeTensor()) continue;
        HB_ASSERT(i == numOfDataOutputs, "found data tensor after shape tensor");
        ++numOfDataOutputs;
        if (outputs[i]->isZeroSizedDataTensor()) return;
        uint64_t denseSizeInElements = outputs[i]->getDenseSizeInElements();
        outputSizesAfter.push_back(denseSizeInElements);
        outputOptimalCost.push_back(denseSizeInElements * outputs[i]->getElementSizeInBytes());
    }

    // do not support transpose on the SCD [tensor size in elements, 1] -> [1, tensor size in elements]
    for (unsigned dim = 0; dim < inputs[0]->getDim() - 1; ++dim)
    {
        uint64_t cost = 0;
        for (unsigned i = 0; i < numOfDataInputs; ++i)
        {
            inputSizesBefore[i] *= inputs[i]->getSizeInElements(dim);
            inputSizesAfter[i] /= inputs[i]->getSizeInElements(dim);

            // in case that number of elements in one sides is exactly 1, transpose will replace with reshape
            // so we need to add the cost of linear copy
            if (inputSizesBefore[i] == 1 || inputSizesAfter[i] == 1)
            {
                cost += calculateExpectedCost(inputOptimalCost[i], cacheLineSize, inputOptimalCost[i]);
            }
            else
            {
                cost += std::max(
                    calculateExpectedCost(inputOptimalCost[i], cacheLineSize, inputSizesBefore[i] * elementSize),
                    calculateExpectedCost(inputOptimalCost[i], cacheLineSize, inputSizesAfter[i] * elementSize));
            }
        }
        for (unsigned i = 0; i < numOfDataOutputs; ++i)
        {
            outputSizesBefore[i] *= outputs[i]->getSizeInElements(dim);
            outputSizesAfter[i] /= outputs[i]->getSizeInElements(dim);

            // in case that number of elements in one sides is exactly 1, transpose will replace with reshape
            // so we need to add the cost of linear copy
            if (outputSizesBefore[i] == 1 || outputSizesAfter[i] == 1)
            {
                cost += calculateExpectedCost(outputOptimalCost[i], cacheLineSize, outputOptimalCost[i]);
            }
            else
            {
                cost += std::max(
                    calculateExpectedCost(outputOptimalCost[i], cacheLineSize, outputSizesBefore[i] * elementSize),
                    calculateExpectedCost(outputOptimalCost[i], cacheLineSize, outputSizesAfter[i] * elementSize));
            }
        }

        if (dim >= startFromDim && cost < res.expectedCost)
        {
            res.expectedCost = cost;
            res.newFcdDim    = dim + 1;
        }
    }
}

ShiftTransposesForFcdOpsResults FcdOpsUtils::createOppositeShiftTransposes(const HalReader&    hal,
                                                                           std::string_view    name,
                                                                           const TensorVector& inputs,
                                                                           const TensorVector& outputs,
                                                                           const unsigned      startFromDim)
{
    ShiftTransposesForFcdOpsResults res;

    unsigned dims = inputs[0]->getDim();
    HB_ASSERT_DEBUG_ONLY(
        std::all_of(inputs.begin(), inputs.end(), [&](const TensorPtr& t) { return t->getDim() == dims; }) &&
            std::all_of(outputs.begin(), outputs.end(), [&](const TensorPtr& t) { return t->getDim() == dims; }),
        "input and output must be with same dimension");
    HB_ASSERT(dims > startFromDim + 1, "can't start the check on the SCD");

    findBestDimForTranspose(hal, startFromDim, inputs, outputs, res /*inplace modify */);

    NodeList beforeNodes;
    for (const TensorPtr& input : inputs)
    {
        if (input->isShapeTensor())
        {
            const TransposePermutationArray& permutation = getShiftTransposePermutation(dims, res.newFcdDim - 1);
            beforeNodes.push_back(createTransposedShape(input, permutation));
        }
        else
        {
            beforeNodes.splice(beforeNodes.end(),
                               createShiftTransposeSequence(input,
                                                            res.newFcdDim - 1,
                                                            true /* from input */,
                                                            fmt::format("{}_{}", name, input->getName())));
        }
        res.newInputs.push_back(beforeNodes.back()->getOutput(0));
    }
    NodeList afterNodes;
    for (const TensorPtr& output : outputs)
    {
        afterNodes.splice(afterNodes.begin(),
                          createShiftTransposeSequence(output,
                                                       res.newFcdDim - 1,
                                                       false /* from input */,
                                                       fmt::format("{}_{}", name, output->getName())));
        res.newOutputs.push_back(afterNodes.front()->getInput(0));
    }

    res.newNodes.splice(res.newNodes.end(), beforeNodes);
    res.newNodes.splice(res.newNodes.end(), afterNodes);
    return res;
}

NodePtr FcdOpsUtils::createFlattenShapeNode(const TensorPtr& tensor, const unsigned axis)
{
    TensorPtr        flattenOutput = createFlattenedTensor(tensor, axis);
    synFlattenParams flattenParams {.axis = axis};
    return NodeFactory::createNode({tensor},
                                   {flattenOutput},
                                   &flattenParams,
                                   NodeFactory::flattenShapeNodeTypeName,
                                   fmt::format("{}/flatten_shape", tensor->getName()));
}

NodePtr FcdOpsUtils::createExtractShapeNode(const TensorPtr& tensor)
{
    TensorPtr clonedShape = tensor->clone(false, false, false);
    clonedShape->setElementType(syn_type_int32);
    clonedShape->setShapeTensor(SHAPE_TENSOR);
    NodePtr extractShape = NodeFactory::createNode({tensor},
                                                   {clonedShape},
                                                   nullptr,
                                                   NodeFactory::extractShapeNodeTypeName,
                                                   fmt::format("{}/cloned_shape", tensor->getName()));
    return extractShape;
}

NodePtr FcdOpsUtils::createTransposedShape(const TensorPtr& tensor, const TransposePermutationArray& perm)
{
    TensorPtr          transposedShape    = getTensorAfterTranspose(*tensor, perm);
    synTransposeParamsNDims transposeParams    = permutationToParams(perm);
    NodePtr                 transposeShapeNode = NodeFactory::createNode({tensor},
                                                         {transposedShape},
                                                         &transposeParams,
                                                         NodeFactory::transposedShapeNodeTypeName,
                                                         fmt::format("{}/transpose_shape", tensor->getName()));
    return transposeShapeNode;
}

// since flatten node not supported in dynamic shape, we perform flatten by
// reshape, and create shape tensor in case of dynamic shape
llvm_vecsmall::SmallVector<NodePtr, 3>
FcdOpsUtils::createFlattenByReshapeNode(const TensorPtr& tensor, const unsigned axis, std::string_view name)
{
    llvm_vecsmall::SmallVector<NodePtr, 3> ret;
    TensorVector                           reshapeInputs = {tensor};
    if (tensor->isDynamicShape())
    {
        // if is dynamic shape we create two nodes:
        // first, extractShape for the reshape input
        // second, flattenShape that fit to the reshape output
        NodePtr extractShape     = createExtractShapeNode(tensor);
        NodePtr flattenShapeNode = createFlattenShapeNode(extractShape->getOutput(0), axis);
        reshapeInputs.push_back(flattenShapeNode->getOutput(0));
        ret.push_back(std::move(extractShape));
        ret.push_back(std::move(flattenShapeNode));
    }

    NodePtr reshapeNode = createFlattenByReshape(reshapeInputs, axis, fmt::format("{}/flatten_by_reshape", name));

    ret.push_back(std::move(reshapeNode));
    return ret;
}

// Returns a tensor with an aggregated fcd
// For example: if the given tensor dimenstions are N,H,W,C and axis = 1
// The returned tensor is N,H,W*C
static TensorPtr createFcdAggregatedTensor(const TensorPtr& tensor, unsigned axis)
{
    TSize    dim0Size    = 1;
    TSize    dim0MinSize = 1;
    unsigned dim         = 0;

    for (; dim <= axis; ++dim)
    {
        dim0Size *= tensor->getSizeInElements(dim);
        dim0MinSize *= tensor->getMinimalSizeInElements(dim);
    }

    SizeArray newSizes    = {dim0Size};
    SizeArray newMinSizes = {dim0MinSize};

    for (; dim < tensor->getDim(); ++dim)
    {
        newSizes[dim - axis]    = tensor->getSizeInElements(dim);
        newMinSizes[dim - axis] = tensor->getMinimalSizeInElements(dim);
    }

    TensorPtr newTensor = tensor->clone(false, false, false);
    newTensor->setName(fmt::format("{}_fcd_aggregated", tensor->getName()));
    newTensor->reshape(dim - axis, newSizes.data(), nullptr, newMinSizes.data());

    return newTensor;
}

static NodePtr createFcdAggregationByReshape(const TensorVector& inputs, unsigned axis, const std::string& name)
{
    const TensorPtr& input = inputs[0];
    unsigned         dims  = input->getDim();
    HB_ASSERT(axis < dims, "axis is larger than tensor dim, axis: {}, dim: {}", axis, dims);

    TensorPtr reshapeOutput = createFcdAggregatedTensor(input, axis);
    NodePtr   reshapeNode =
        NodeFactory::createNode(inputs, {reshapeOutput}, nullptr, NodeFactory::reshapeNodeTypeName, name);
    return reshapeNode;
}

// Returns an einsum equation which aggregates the fcd up to the given axis
static std::string createAggregateFcdEquation(const unsigned dims, const unsigned axis)
{
    int         outIdx   = 0;
    std::string equation = "";

    // Create the equation's input
    for (int i = 0; i < dims; ++i)
    {
        equation += (char)('a' + i);
        if (i < dims - 1)
        {
            equation += ',';
        }
    }

    equation.append("->");

    // Create the equation's output
    for (; outIdx < axis; ++outIdx)
    {
        equation += (char)('a' + outIdx);
        equation += '*';
    }

    for (; outIdx < dims; ++outIdx)
    {
        equation += (char)('a' + outIdx);

        if (outIdx < dims - 1)
        {
            equation += ',';
        }
    }

    return equation;
}

static NodePtr createAggregateFcdShapeNode(const TensorPtr& tensor, const unsigned axis)
{
    TensorPtr fcdAggregateOutput = createFcdAggregatedTensor(tensor, axis);

    dynamicReshapeParams einsumParams;
    einsumParams.equation = createAggregateFcdEquation(tensor->getDim(), axis);

    return NodeFactory::createNode({tensor},
                                   {fcdAggregateOutput},
                                   &einsumParams,
                                   NodeFactory::dynamicReshapeNodeTypeName,
                                   fmt::format("{}/aggregateFcd_shape", tensor->getName()));
}

// Returns aggregation of all dimensions up to axis into one dimension
NodeList FcdOpsUtils::aggregateFcdByReshapeNode(const TensorPtr& tensor, const unsigned axis, std::string_view name)
{
    NodeList     ret;
    TensorVector reshapeInputs = {tensor};

    if (tensor->isDynamicShape())
    {
        // if is dynamic shape we create two nodes:
        // first, extractShape for the reshape input
        // second, aggregateFcdShape that fit to the reshape output
        NodePtr extractShape = createExtractShapeNode(tensor);
        ret.push_back(extractShape);

        NodePtr aggregateFcdShapeNode = createAggregateFcdShapeNode(extractShape->getOutput(0), axis);
        ret.push_back(aggregateFcdShapeNode);
        reshapeInputs.push_back(aggregateFcdShapeNode->getOutput(0));
    }

    NodePtr reshapeNode =
        createFcdAggregationByReshape(reshapeInputs, axis, fmt::format("{}/aggregate_by_reshape", name));

    ret.push_back(reshapeNode);
    return ret;
}
