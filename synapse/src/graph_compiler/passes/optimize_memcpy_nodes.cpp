#include "optimize_memcpy_nodes.h"

#include "compilation_hal_reader.h"
#include "defs.h"
#include "dma_memcopy_node.h"
#include "dma_transpose_node.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "handle_logical_operations.h"
#include "logical_op_node.h"
#include "node_factory.h"
#include "physical_memory_ops_nodes.h"
#include "synapse_common_types.h"
#include "tpc_node.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>

template<typename T>
static NodeVector getConcreteCopyNodes(const T& nodes)
{
    NodeVector  concreteCopyNodes;
    const auto& isConcreteCopyNode = CompilationHalReader::getHalReader()->isGcDmaSupported()
                                         ? [](const NodePtr& n) { return n->isDma(); }
                                         : isTpcMemcpy;
    std::copy_if(nodes.begin(),
                 nodes.end(),
                 std::back_inserter(concreteCopyNodes),
                 [&isConcreteCopyNode](const NodePtr& n) { return isConcreteCopyNode(n); });
    return concreteCopyNodes;
}

static bool reinstantiateTpcKernel(const NodePtr& concreteCopy, HabanaGraph& g)
{
    bool success = true;
    HB_ASSERT(!CompilationHalReader::getHalReader()->isGcDmaSupported(),
              "Expecting a TPC memcpy to be optimized only when GC cannot use DMA engine");
    HB_ASSERT(isTpcMemcpy(concreteCopy),
              "Expecting {}[{}] to be a TPC memcopy node",
              concreteCopy->getNodeName(),
              concreteCopy->getNodeTypeStr());
    const auto tpc = std::dynamic_pointer_cast<TPCNode>(concreteCopy);
    HB_ASSERT_PTR(tpc);
    tpc->resetInstantiated();
    if (tpc->init(deviceTypeToDeviceID(g.getDeviceType()),
                  &g.getGraphAnnotation().cachedAuxiliaryTensors,
                  g.getNextTPCKernelUniqueId()) != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_WARN(OPT_MEMCPY, "Failed reinstantiating TPC node {}[{}]", tpc->getNodeName(), tpc->getNodeTypeStr());
        success = false;
    }
    return success;
}

// ******************************* StridedMemcpyViaTransposeEngineStrategy ****************************************

// node is fit for transpose engine only if the input is sparse tensor of dimension 2 where the FCD size is 1
// and the output is dense, also if the tensor is dynamic the FCD must be static
bool StridedMemcpyViaTransposeEngineStrategy::isFitForStrategy(const NodePtr& memcpy) const
{
    if (!CompilationHalReader::getHalReader()->isGcDmaSupported() || !memcpy->isDma()) return false;
    const TensorPtr& input  = memcpy->getInput(0);
    const TensorPtr& output = memcpy->getOutput(0);
    HB_ASSERT_PTR(input);
    HB_ASSERT_PTR(output);

    if (!GCFG_ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE.value()) return false;
    // Node must have 1 data input/output
    if (memcpy->getNumInputs() != 1 || memcpy->getNumOutputs() != 1) return false;
    // We're allowed to replace only Dma nodes that created from semantic memcpy
    const auto dmaNode = std::dynamic_pointer_cast<DMAMemcpyNode>(memcpy);
    if (!dmaNode) return false;
    if (!dmaNode->isCreatedFromSemanticMemcpy()) return false;
    // Is supported only for dense output and sparse input
    if (input->isDenseLayout() || !output->isDenseLayout()) return false;
    // Input dim must fit to transpose engine
    TensorShape mergedDenseTensorShape = mergeDenseDimensions(input).first;
    if (mergedDenseTensorShape.getDim() != 2) return false;
    // fcd size in bytes not supported
    TSize inputFcdElementsInBytes = mergedDenseTensorShape.getSize(0) * input->getElementSizeInBytes();
    if (CompilationHalReader::getHalReader()->getDmaTransposeSupportedDataType(inputFcdElementsInBytes) == syn_type_na)
        return false;
    // the FCD must be static
    unsigned int fcd = 0;
    if (mergedDenseTensorShape.isDynamicDim(fcd) || output->isDynamicDim(fcd)) return false;
    // If we able to wrap dma nodewith reshapes- strides on FCD are not supported
    if (input->getDim() != 2 && (dmaNode->getInput(0)->isStridedOnFCD() || dmaNode->getOutput(0)->isStridedOnFCD()))
        return false;
    return true;
}

NodeList StridedMemcpyViaTransposeEngineStrategy::applyStrategy(const NodePtr& dmaNode) const
{
    NodeList beforeDma;
    NodeList afterDma;

    TensorPtr dmaInput  = dmaNode->getInput(0);
    TensorPtr dmaOutput = dmaNode->getOutput(0);
    HB_ASSERT_PTR(dmaInput);
    HB_ASSERT_PTR(dmaOutput);
    // In case current input is not fit to engine (not 2 dims) but the densest form of input will fit,
    // we will wrap the input with reshapes to fit the optimization.
    // The flow: reshape -> StridedMemcpyViaTransposeEngineStrategy -> reshape
    if (dmaInput->getDim() != 2)
    {
        // Createing the new tensors with new shapes to connect the reshapes to the DMA node
        TensorShape inputDenseShape;
        // mergeDenseDimensions returns the densest form of input.
        inputDenseShape = mergeDenseDimensions(dmaInput).first;

        const TensorPtr firstReshapeInput   = dmaInput;
        const TensorPtr secondReshapeOutput = dmaOutput;
        dmaInput  = std::make_shared<Tensor>(inputDenseShape, firstReshapeInput->getElementType());
        dmaOutput = std::make_shared<Tensor>(inputDenseShape, firstReshapeInput->getElementType());

        dmaInput->setName(firstReshapeInput->getName() + "/reshaped");
        dmaOutput->setName(secondReshapeOutput->getName() + "/reshaped");
        // Createing the new reshape nodes
        auto firstReshapeNode =
            createReshape(firstReshapeInput,
                          dmaInput,
                          dmaNode->getNodeName() + "/first_reshape_" + std::to_string(firstReshapeInput->getId()));
        auto secondReshapeNode =
            createReshape(dmaOutput,
                          secondReshapeOutput,
                          dmaNode->getNodeName() + "/second_reshape_" + std::to_string(dmaOutput->getId()));

        beforeDma.push_back(firstReshapeNode);
        afterDma.push_front(secondReshapeNode);
    }

    const auto& halReader(CompilationHalReader::getHalReader());
    if ((dmaInput->getSizeInElements(0) != 1) || !halReader->isDmaTransposeSupported(dmaOutput->getElementType()))
    {
        HB_ASSERT(halReader->getDmaTransposeSupportedDataType(dmaInput->getSizeInBytes(0)) != syn_type_na,
                  "not legal data type");
        HB_ASSERT(dmaInput->getDim() == 2, "num of dimensions is not fit to transpose engine");
        // after the reinterpret cast we refer to the size like it is in form [1,X]
        const TensorPtr firstReinterpretCastInput   = dmaInput;
        const TensorPtr secondReinterpretCastOutput = dmaOutput;
        TSize           newMaxSizes[2]              = {1, dmaInput->getSizeInElements(1)};
        TSize           newMinSizes[2]              = {1, dmaInput->getMinimalSizeInElements(1)};
        // the newDataType get the supported type acoording to dmaInput fcd size in bytes
        synDataType newDataType = halReader->getDmaTransposeSupportedDataType(dmaInput->getSizeInBytes(0));
        dmaInput                = std::make_shared<Tensor>(2, newMaxSizes, newDataType, newMinSizes);
        dmaOutput               = std::make_shared<Tensor>(*dmaInput);

        dmaInput->setName(firstReinterpretCastInput->getName() + "/ReinterpretCasted");
        dmaOutput->setName(secondReinterpretCastOutput->getName() + "/ReinterpretCasted");

        // Createing the new ReinterpretCast nodes
        auto firstReinterpretCastNode  = createReinterpretCast(firstReinterpretCastInput,
                                                              dmaInput,
                                                              dmaNode->getNodeName() + "/first_reinterpret_cast");
        auto secondReinterpretCastNode = createReinterpretCast(dmaOutput,
                                                               secondReinterpretCastOutput,
                                                               dmaNode->getNodeName() + "/second_reinterpret_cast");

        beforeDma.push_back(firstReinterpretCastNode);
        afterDma.push_front(secondReinterpretCastNode);
    }
    if (!beforeDma.empty())
    {
        // The original external tensors of the additional nodes must be the real tensors of the reshapes/casts
        beforeDma.front()->getInput(0)->setIsRealInLogical(true);
        afterDma.back()->getOutput(0)->setIsRealInLogical(true);
    }
    NodeList& ret = beforeDma;
    ret.emplace_back(new StridedDMANodeViaTransposeNode(dmaInput, dmaOutput, dmaNode->getNodeName()));
    ret.splice(ret.end(), afterDma);
    return ret;
}

NodePtr StridedMemcpyViaTransposeEngineStrategy::createReinterpretCast(const TensorPtr&   input,
                                                                       const TensorPtr&   output,
                                                                       const std::string& name) const
{
    NodePtr ret = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::reinterpretCastNodeTypeName, name);
    auto    logical = std::dynamic_pointer_cast<LogicalOpNode>(ret);
    HB_ASSERT_PTR(logical);
    // We don't want any more wasteful memcpy to be added
    logical->setIsPureLogical(true);
    return ret;
}

NodePtr StridedMemcpyViaTransposeEngineStrategy::createReshape(const TensorPtr&   input,
                                                               const TensorPtr&   output,
                                                               const std::string& name) const
{
    NodePtr ret     = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::staticReshapeNodeTypeName, name);
    auto    logical = std::dynamic_pointer_cast<LogicalOpNode>(ret);
    HB_ASSERT_PTR(logical);
    // We don't want any more wasteful memcpy to be added
    logical->setIsPureLogical(true);
    return ret;
}

// ******************************* AggregateFcdWithStaticReshapeBase ****************************************

bool AggregateFcdWithStaticReshapeBase::isFitForStrategy(const NodePtr& memcpy) const
{
    if (!GCFG_ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION.value()) return false;

    const TensorPtr& input  = memcpy->getInput(0);
    const TensorPtr& output = memcpy->getOutput(0);
    // When GC can use DMA engine, the input and the output must be dynamicShape.
    if (CompilationHalReader::getHalReader()->isGcDmaSupported() &&
        (!input->isDynamicShape() || !output->isDynamicShape()))
        return false;

    // when the fcd is dynamic we can't aggregate nothing
    if (input->isDynamicDim(0) || output->isDynamicDim(0)) return false;
    // node must have 1 output
    if (memcpy->getNumOutputs() != 1) return false;
    // Strides on FCD are not supported
    return !memcpy->getInput(0)->isStridedOnFCD();
}

// This function returns the new shape of the memcpy tensor after the optimization,
// if the optimization does not reduce the shape, the function will return an empty shape.
std::optional<TensorShape> AggregateFcdWithStaticReshapeBase::getNewShape(const NodePtr& memcpy) const
{
    const TensorPtr& input  = memcpy->getInput(0);
    const TensorPtr& output = memcpy->getOutput(0);

    TensorShape inputDenseShape;
    TensorShape outputDenseShape;
    // mergeDenseDimensions returns the densest form of input and output.
    std::tie(inputDenseShape, std::ignore)  = mergeDenseDimensions(input);
    std::tie(outputDenseShape, std::ignore) = mergeDenseDimensions(output);

    NSizeArray newMaxSize;
    NSizeArray newMinSize;

    // We want to get the fcd of the densest shape max size, and then to get the corresponding fcd min size.
    // we will choose the common fcd of inputDenseShape and outputDenseShape which is the min fcd,
    // because the minimum is the one with the least aggregations.
    TSize maxFcdSize = std::min(inputDenseShape.getMaxSize(0), outputDenseShape.getMaxSize(0));

    // calculate new fcd size
    newMinSize[0]               = 1;
    newMaxSize[0]               = 1;
    unsigned lastAggregatedDims = 0;
    for (; lastAggregatedDims < input->getDim(); ++lastAggregatedDims)
    {
        newMaxSize[0] *= input->getSizeInElements(lastAggregatedDims);
        newMinSize[0] *= input->getMinimalSizeInElements(lastAggregatedDims);
        if (newMaxSize[0] == maxFcdSize || shouldBreakOnDimWhenCalculateNewFcdSize(memcpy, lastAggregatedDims))
        {
            break;
        }
    }

    // return empty TensorShape in case we can't optimize
    if (newMaxSize[0] == input->getSizeInElements(0) || newMaxSize[0] == 0) return std::nullopt;

    // maxFcdSize is multiplication of dimensions 0, 1, ..., x.
    // The new fcd size (newMaxSize[0]) is also multiplication of dimensions 0, 1, ..., y. With a guarantee that y <= x.
    HB_ASSERT(maxFcdSize >= newMaxSize[0] && maxFcdSize % newMaxSize[0] == 0, "Invalid size for new FCD");

    // Since we are aggregating over all the first dimensions- we want that the fcd of the new shape will be same fcd as
    // densest shape, but we will take the unaggregated dimensions from the original input.
    for (unsigned dim = 1; dim < input->getDim() - lastAggregatedDims; dim++)
    {
        newMaxSize[dim] = input->getSizeInElements(lastAggregatedDims + dim);
        newMinSize[dim] = input->getMinimalSizeInElements(lastAggregatedDims + dim);
    }
    TensorShape newShape = TensorShape(input->getDim() - lastAggregatedDims, newMaxSize, newMinSize);

    return newShape;
}

NodePtr AggregateFcdWithStaticReshapeBase::createStaticReshape(const TensorPtr&   input,
                                                               const TensorPtr&   output,
                                                               const std::string& name) const
{
    NodePtr ret     = NodeFactory::createNode({input}, {output}, nullptr, NodeFactory::staticReshapeNodeTypeName, name);
    auto    logical = std::dynamic_pointer_cast<LogicalOpNode>(ret);
    HB_ASSERT_PTR(logical);
    // We don't want any more wasteful memcpy to be added
    logical->setIsPureLogical(true);
    return ret;
}

std::pair<NodeList, NodePtr>
AggregateFcdWithStaticReshapeBase::createNewMemcpyWrappedByReshapes(const NodePtr&     memcpy,
                                                                    const TensorShape& newShape) const
{
    const TensorPtr& input  = memcpy->getInput(0);
    const TensorPtr& output = memcpy->getOutput(0);
    // The original tensors must be the real tensors of the reshapes
    input->setIsRealInLogical(true);
    output->setIsRealInLogical(true);

    // Creating the new tensors with new shapes to connect the reshapes to the memcopy node
    const TensorPtr& newInput = std::make_shared<Tensor>(newShape, input->getElementType());
    newInput->setName(input->getName() + "/reshaped");
    const TensorPtr& newOutput = std::make_shared<Tensor>(newShape, input->getElementType());
    newOutput->setName(output->getName() + "/reshaped");
    // Creating the new static reshape nodes
    auto firstStaticReshapeNode =
        createStaticReshape(input,
                            newInput,
                            memcpy->getNodeName() + "/first_reshape_" + std::to_string(input->getId()));
    auto secondStaticReshapeNode =
        createStaticReshape(newOutput,
                            output,
                            memcpy->getNodeName() + "/second_reshape_" + std::to_string(newOutput->getId()));
    // Changing the old node so that it is connected to static reshape nodes with the new tensors
    const NodePtr& newMemcpyNode = memcpy->clone();
    newMemcpyNode->replaceInput(0, newInput);
    newMemcpyNode->replaceOutput(0, newOutput);

    return {{firstStaticReshapeNode, newMemcpyNode, secondStaticReshapeNode}, newMemcpyNode};
}

// ******************************* AggregateFcdWithStaticReshape ****************************************

bool AggregateFcdWithStaticReshape::isFitForStrategy(const NodePtr& memcpy) const
{
    if (!AggregateFcdWithStaticReshapeBase::isFitForStrategy(memcpy)) return false;
    // Only memcpies node with 1 input and output are supported
    if (memcpy->getNumInputs() != 1) return false;
    // getNewShape result is set only if there is aggregation in the FCD
    return getNewShape(memcpy).has_value();
}

NodeList AggregateFcdWithStaticReshape::applyStrategy(const NodePtr& memcpy) const
{
    auto newShape = getNewShape(memcpy);
    HB_ASSERT(newShape.has_value(), "Try to apply {} but shape is not set", getStrategyName());
    return createNewMemcpyWrappedByReshapes(memcpy, newShape.value()).first;
}

// ******************************* AggregateDynamicSliceFcdWithStaticReshape ****************************************

bool AggregateDynamicSliceFcdWithStaticReshape::isFitForStrategy(const NodePtr& memcpy) const
{
    // valid only for dynamic slice
    if (!CompilationHalReader::getHalReader()->isGcDmaSupported() || !memcpy->isDma()) return false;
    const auto dmaNode = std::dynamic_pointer_cast<DMAMemcpyNode>(memcpy);
    if (!dmaNode) return false;
    if (dmaNode->getDynamicMemoryOpType() != DMA_OP_DYNAMIC_SLICE) return false;
    if (!AggregateFcdWithStaticReshapeBase::isFitForStrategy(dmaNode)) return false;

    HB_ASSERT(dmaNode->getInput(0)->compareGeometry(*dmaNode->getOutput(0)), "sizes mismatch");
    HB_ASSERT(dmaNode->getNumInputs() == DynamicSliceMemcpyNodeBase::SliceDmaInputs::MAX_NUM_INPUTS,
              "unexpected number of inputs");

    return getNewShape(dmaNode).has_value();
}

bool AggregateDynamicSliceFcdWithStaticReshape::shouldBreakOnDimWhenCalculateNewFcdSize(const NodePtr& dmaNode,
                                                                                        const unsigned dim) const
{
    const TensorPtr& steps  = dmaNode->getInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STEPS_TENSOR);
    const TensorPtr& starts = dmaNode->getInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STARTS_TENSOR);
    HB_ASSERT_PTR(steps);
    HB_ASSERT_PTR(starts);

    HB_ASSERT(steps->getMinimalSizeInElements(dim) == 1,
              "Steps on the FCD are not allowed, and it should get caught earlier");

    static constexpr std::string_view msg =
        "At slice node, larger steps mean smaller tensor, so the min steps must be bigger or equal to the max steps";
    HB_ASSERT(steps->getMinimalSizeInElements(dim) >= steps->getSizeInElements(dim), "{}", msg);
    // Validate that by aggregating dimensions we don't move untrivial steps to the FCD
    if (dim + 1 < steps->getDim())
    {
        HB_ASSERT(steps->getMinimalSizeInElements(dim + 1) >= steps->getSizeInElements(dim + 1), "{}", msg);
        if (steps->getMinimalSizeInElements(dim + 1) != 1) return true;
    }
    // If the start of the dim is not 0, it mean that the next dimension isn't dense.
    return starts->getSizeInElements(dim) != 0;
}

NodePtr AggregateDynamicSliceFcdWithStaticReshape::createMergeShapeNode(const TensorPtr& input,
                                                                        const unsigned   numOfSqueezedDims) const
{
    SifMergeShapesMetadata params;
    params.outputDim = input->getDim() - numOfSqueezedDims;

    NSizeArray maxSizes;
    NSizeArray minSizes;

    for (unsigned dim = 0; dim < params.outputDim; ++dim)
    {
        maxSizes[dim] = input->getSizeInElements(numOfSqueezedDims + dim);
        minSizes[dim] = input->getMinimalSizeInElements(numOfSqueezedDims + dim);

        params.dimMap[dim].inputIdx = 0;
        params.dimMap[dim].dimIdx   = (int)numOfSqueezedDims + dim;
    }

    const auto& output = input->clone(false, false, false);
    output->reshape(params.outputDim, maxSizes.data(), nullptr, minSizes.data());

    return NodeFactory::createNode({input},
                                   {output},
                                   &params,
                                   NodeFactory::mergeShapesNodeTypeName,
                                   input->getName() + "/mergeShape");
}

NodePtr AggregateDynamicSliceFcdWithStaticReshape::createTileShapeNode(const TensorPtr& input,
                                                                       const unsigned   tile) const
{
    ns_TileKernel::ParamsV2 params;
    std::fill(std::begin(params.repeat), std::end(params.repeat), 1);
    params.repeat[0] = tile;

    auto maxSizes = input->getNSizesInElements();
    auto minSizes = input->getNMinimalSizesInElements();

    maxSizes[0] *= tile;
    minSizes[0] *= tile;

    const auto& output = input->clone(false, false, false);
    output->reshape(input->getDim(), maxSizes.data(), nullptr, minSizes.data());

    return NodeFactory::createNode({input},
                                   {output},
                                   &params,
                                   NodeFactory::tileShapeNodeTypeName,
                                   input->getName() + "/tileShape");
}

NodeList AggregateDynamicSliceFcdWithStaticReshape::createDynamicSliceShapeOperations(const NodePtr& originalNode,
                                                                                      const NodePtr& newNode) const
{
    unsigned         numOfSqueezedDims = originalNode->getInput(0)->getDim() - newNode->getInput(0)->getDim();
    const TensorPtr& steps             = originalNode->getInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STEPS_TENSOR);
    const TensorPtr& starts            = originalNode->getInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STARTS_TENSOR);
    HB_ASSERT_PTR(steps);
    HB_ASSERT_PTR(starts);

    auto stepsMergeShape  = createMergeShapeNode(steps, numOfSqueezedDims);
    auto startsMergeShape = createMergeShapeNode(starts, numOfSqueezedDims);

    // Calculate the tile size.
    unsigned tile = 1;
    for (unsigned dim = 0; dim < numOfSqueezedDims; ++dim)
    {
        tile *= originalNode->getInput(0)->getSizeInElements(dim);
    }
    auto startTileShape = createTileShapeNode(startsMergeShape->getOutput(0), tile);

    newNode->replaceInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STEPS_TENSOR, stepsMergeShape->getOutput(0));
    newNode->replaceInput(DynamicSliceMemcpyNodeBase::SliceDmaInputs::STARTS_TENSOR, startTileShape->getOutput(0));

    return {stepsMergeShape, startsMergeShape, startTileShape};
}

NodeList AggregateDynamicSliceFcdWithStaticReshape::applyStrategy(const NodePtr& dmaNode) const
{
    auto newShape = getNewShape(dmaNode);
    HB_ASSERT(newShape.has_value(), "Try to apply {} but shape is not set", getStrategyName());

    HB_ASSERT(dmaNode->getNumInputs() == DynamicSliceMemcpyNodeBase::SliceDmaInputs::MAX_NUM_INPUTS,
              "unexpected number of inputs");

    auto [ret, newMemcpy] = createNewMemcpyWrappedByReshapes(dmaNode, newShape.value());

    ret.splice(ret.end(), createDynamicSliceShapeOperations(dmaNode, newMemcpy));
    return ret;
}

// *************************************************************************************************************

static void updateBundleInfo(const NodePtr& originalNode, const NodeList& newNodes)
{
    for (const auto& node : newNodes)
    {
        node->getNodeAnnotation().bundleInfo = originalNode->getNodeAnnotation().bundleInfo;
    }
}

static void printSizesForLog(const TensorPtr& t, const TensorShape& denseShape, const std::string& str)
{
    const auto& maxSizes = denseShape.getNSizes();
    LOG_TRACE(OPT_MEMCPY,
              "{} max sizes {}, max dense sizes [{}], strides {}",
              str,
              t->getDimSizesStr(),
              toString(maxSizes.begin(), maxSizes.begin() + denseShape.getDim(), ','),
              t->getStridesStr());
    if (t->isDynamicShape())
    {
        const auto& minSizes = denseShape.getNMinSizes();
        LOG_TRACE(OPT_MEMCPY,
                  "{} min sizes {}, min dense sizes [{}]",
                  str,
                  t->getDimSizesStr(false, true),
                  toString(minSizes.begin(), minSizes.begin() + denseShape.getDim(), ','));
    }
}
static constexpr const StridedMemcpyViaTransposeEngineStrategy   stridedMemcpyViaTransposeEngineStrategy;
static constexpr const AggregateFcdWithStaticReshape             aggregateFcdWithStaticReshape;
static constexpr const AggregateDynamicSliceFcdWithStaticReshape aggregateDynamicSliceFcdWithStaticReshape;

// clang-fomat off
// vector of the optimization functions ordered by priority
static constexpr std::array memcpyOptimizationStrategies = {
    (MemcpyOptimizationStrategy*)&stridedMemcpyViaTransposeEngineStrategy,
    (MemcpyOptimizationStrategy*)&aggregateFcdWithStaticReshape,
    (MemcpyOptimizationStrategy*)&aggregateDynamicSliceFcdWithStaticReshape};
// clang-fomat on

static NodeList optimizeMemcpyNode(const HabanaGraph& g, const NodePtr& memcpy)
{
    for (const auto& strategy : memcpyOptimizationStrategies)
    {
        if (strategy->isFitForStrategy(memcpy))
        {
            LOG_TRACE(OPT_MEMCPY, "optimization {} chosen for {}", strategy->getStrategyName(), memcpy->getNodeName());
            return strategy->applyStrategy(memcpy);
        }
    }
    return {};
}

// skip if the node is one of the following:
// (1) not Dma Memcpy Node
// (2) is linear Dma, so it should be with maximal utilization
// (3) is memset node
// (4) the producer is TPC node that may reuse input
// (5) there is producer/consumer which is not handled logical operation and the relevant tensor is not "real in
// logical"
static bool skipOptimization(const HabanaGraph& g, const NodePtr& candidate)
{
    if (CompilationHalReader::getHalReader()->isGcDmaSupported())
    {
        if (!candidate->isDma()) return true;
        const auto dmaNode = std::dynamic_pointer_cast<DMAMemcpyNode>(candidate);
        if (!dmaNode || dmaNode->isLinearDma() || dmaNode->isMemset()) return true;
    }
    else  // gaudi3
    {
        if (!isTpcMemcpy(candidate)) return true;
        // skip TPC memcopy nodes optimized based on glue code suggested manipulation
        const auto& tpcMemcpy = std::dynamic_pointer_cast<TPCNode>(candidate);
        HB_ASSERT_PTR(tpcMemcpy);
        if (tpcMemcpy->isSuggestedOptimizationDone()) return true;
    }

    NodePtr producer = g.getTensorProducer(candidate->getInput(0));

    if (producer && g.runsOnTPC(producer))
    {
        auto tpcNode = std::dynamic_pointer_cast<TPCNode>(producer);
        if (tpcNode && !tpcNode->getReusableInputs().empty()) return true;
    }

    auto nodes = g.getTensorConsumers(candidate->getOutput(0));
    if (GCFG_ENABLE_AGGREGATE_FCD_WITH_RESHAPE_OPTIMIZATION.value())
    {
        if (producer && producer->isLogicalOperation() && !LogicalOpsHandler::isRealInLogical(candidate->getInput(0)))
        {
            auto logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(producer);
            HB_ASSERT_PTR(logicalNode);
            if (!logicalNode->getRunLogicalOperationDone()) return true;
        }
    }
    else
    {
        nodes.push_back(producer);
    }

    if (LogicalOpsHandler::isRealInLogical(candidate->getOutput(0))) return false;

    for (const auto& node : nodes)
    {
        if (!node || !node->isLogicalOperation()) continue;

        auto logicalNode = std::dynamic_pointer_cast<LogicalOpNode>(node);
        HB_ASSERT_PTR(logicalNode);
        if (!logicalNode->getRunLogicalOperationDone()) return true;
    }

    return false;
}

// Atomic nodes are pairs of marked nodes that are required to be adjacent to each other when compilation is finished.
// The following function is needed for cases when the optimized dma memcpy node was marked as part of an atomic pair.
// Because the target node is replaced and removed from the graph as part of the optimiziation, we would like to update
// the atomic node pairs. The traget node will be replaced with the one dma node from the newly extracted nodes. Thus,
// we assume there is ONLY ONE such node.
void updateAtomicNodesAfterOptimization(HabanaGraph& g, const NodePtr& prevNode, const NodeVector& concreteCopyNodes)
{
    HB_ASSERT_PTR(prevNode);
    HB_ASSERT(concreteCopyNodes.size() == 1,
              "Expected to extract just one new copy node, but got {}",
              concreteCopyNodes.size());
    g.getGraphAnnotation().replaceAtomicNode(prevNode.get(), concreteCopyNodes.back());
}

bool optimizeMemcpyNodes(HabanaGraph& g)
{
    if (!GCFG_ENABLE_OPTIMIZE_MEMCPY_NODES.value()) return true;

    std::map<std::string, uint64_t> optimizedNodesCounterByGuid;

    auto nodes = g.getNodes();
    for (const auto& n : nodes)
    {
        if (skipOptimization(g, n)) continue;
        LOG_TRACE(OPT_MEMCPY, "Attempt optimizing {}, GUID: {}", n->getNodeName(), n->getGUID());

        NodeList newNodes = optimizeMemcpyNode(g, n);
        if (newNodes.empty())
        {
            LOG_TRACE(OPT_MEMCPY, "Cannot optimize {}, GUID: {}", n->getNodeName(), n->getGUID());
            continue;
        }

        const auto copyNodes = getConcreteCopyNodes(newNodes);
        HB_ASSERT(copyNodes.size() == 1, "Expecting exactly one concrete copy node, found: {}", copyNodes.size());
        const auto& concreteCopy = copyNodes.front();

        if (isTpcMemcpy(concreteCopy))
        {
            // When GC cannot use DMA engine, TPC memcpy nodes can be optimized and in that case the kernel needs to be
            // reinstantiated to fit the new shape/strides.
            if (!reinstantiateTpcKernel(concreteCopy, g))
            {
                continue;
            }

            // Set suggested manipulation done
            const auto tpcMemcpy = std::dynamic_pointer_cast<TPCNode>(concreteCopy);
            HB_ASSERT_PTR(tpcMemcpy);
            tpcMemcpy->setSuggestedOptimizationDone(true);
        }

        if (LOG_LEVEL_AT_LEAST_TRACE(OPT_MEMCPY))
        {
            const TensorPtr& input  = n->getInput(0);
            const TensorPtr& output = n->getOutput(0);
            TensorShape      inputDenseShape(mergeDenseDimensions(input).first);
            TensorShape      outputDenseShape(mergeDenseDimensions(output).first);

            printSizesForLog(input, inputDenseShape, "Input: ");
            printSizesForLog(output, outputDenseShape, "Output:");
        }

        if (LOG_LEVEL_AT_LEAST_DEBUG(OPT_MEMCPY))
        {
            auto it = optimizedNodesCounterByGuid.find(n->getGUID());
            if (it == optimizedNodesCounterByGuid.end())
            {
                optimizedNodesCounterByGuid[n->getGUID()] = 1;
            }
            else
            {
                ++(it->second);
            }
        }

        updateBundleInfo(n, newNodes);
        updateAtomicNodesAfterOptimization(g, n, copyNodes);
        if (GraphEditor::replaceNodes(g, {n}, newNodes) != ReplaceNodeReturnStatus::REPLACE_NODE_SUCCESS) return false;
    }
    for (auto p : optimizedNodesCounterByGuid)
    {
        LOG_DEBUG(OPT_MEMCPY, "optimized {} nodes with GUID: {}", p.second, p.first);
    }
    return true;
}
