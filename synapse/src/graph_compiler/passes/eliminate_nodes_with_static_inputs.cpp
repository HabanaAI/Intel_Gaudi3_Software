#include "cast_utils.hpp"
#include "const_section_util.h"
#include "data_type_utils.h"
#include "dma_cost_model.h"
#include "log_manager.h"
#include "node.h"
#include <habana_graph.h>
#include <memory>
#include "graph.h"
#include "quantization_utils.h"

#include "passes.h"
#include "graph_editor.h"
#include "tpc_node.h"
#include "types.h"
#include "transpose_node.h"
#include "data_layout_utils.h"
#include "perf_lib_layer_params.h"
#include "common_type_utils.h"

#define CHANNEL_INDEX 0

/*
 * handleCast -
 * Perform the cast on cpu, store it in the output tensor buffer.
 * Make the output tensor static and add the cast node to remove list for later removal,
 * so only the cast output tensor will remain in graph.
 */
static bool handleCast(const NodePtr& castNode)
{
    if (!castNode->isCast()) return false;
    LOG_TRACE(CONST_FOLDING, "Node {} is a cast node with static input", castNode->getNodeName());
    HB_ASSERT(castNode->getNumInputs() == 1 && castNode->getNumOutputs() == 1,
              "Node {} is cast with multiple inputs and\\or outputs",
              castNode->getNodeName());

    TensorPtr castInput  = castNode->getInput(0);
    TensorPtr castOutput = castNode->getOutput(0);
    if (castInput == nullptr || castOutput == nullptr)
    {
        LOG_WARN(CONST_FOLDING, "At least on of cast node {} tensors is null, doing nothing", castNode->getNodeName());
        return false;
    }
    HB_ASSERT(castInput->getTotalElements() == castOutput->getTotalElements(),
              "Cast input and output tensors number of elements isn't equal");
    // cast types according to the cast guid (not cast tensors types since those may be not set yet).
    synDataType castFromType  = getSynDataTypeFromDtypeSuffix(extractDtypeFromCastGUID(castNode->getGUID()));
    synDataType castToType    = getSynDataTypeFromDtypeSuffix(extractDtypeFromGUID(castNode->getGUID()));

    // Unbind old buffer and allocate a new one
    if (castOutput->isBound())
    {
        // TODO consider not to unbind if it is persistent?
        castOutput->unbind();
    }

    LOG_TRACE(CONST_FOLDING, "Performing cast on cpu and storing result in output tensor buffer");
    void* castOutputBuffer = allocateBufferForSynType(castToType, castOutput->getTotalElements());
    castOutput->bind(castOutputBuffer, true);

    // Cast on Cpu - store the casted buffer in output tensor
    CpuCaster cpuCaster(castInput, castOutput);
    cpuCaster.setCastDataTypes(castFromType, castToType);
    if (!GCFG_ENABLE_RUN_ON_CPU_DUMMY_MODE.value())
    {
        if (!cpuCaster.doCast())
        {
            LOG_WARN(CONST_FOLDING, "Casting on cpu for cast node {} failed", castNode->getNodeName());
            return false;
        }
    }
    // Set output as data type match buffer type (so it won't be handled further by quantization passes)
    castOutput->setAsDataTypeMatchData();
    // Set output permutation if needed
    const std::optional<gc::Permutation>& permutedInputPerm = castInput->getPermutation();
    if (permutedInputPerm && !permutedInputPerm.value().isIdentity())
    {
        LOG_TRACE(CONST_FOLDING, "Cast input was permuted, setting permutations in cast output tensor accordingly");
        castOutput->setPermutation(permutedInputPerm.value());
        castOutput->reshape(castOutput->getDim()); // adjust strides according to permutation
    }
    LOG_TRACE(CONST_FOLDING, "Finished handling cast node {} with static input", castNode->getNodeName());
    return true;
}

void packWeights(const TensorPtr& origWeights, const TensorPtr& packeddWeights, unsigned stride, unsigned packingFactor)
{
    const SizeArray& newSizes   = packeddWeights->getAllSizesInElements();
    const SizeArray& orgSizes   = origWeights->getAllSizesInElements();
    StrideArray      orgStrides = origWeights->getAllStridesInBytes();

    uint8_t* newData = static_cast<uint8_t*>(packeddWeights->map());
    /* duplicate data */
    void* data        = origWeights->getAddress();
    TSize newDataSize = packeddWeights->getTotalSizeInBytes();

    memset(newData, 0, newDataSize);

    // generate the weights data according to new dimensions
    TSize orgSizeQ    = orgSizes[WEIGHT_DIM_Q];
    TSize orgSizeR    = orgSizes[WEIGHT_DIM_R];
    TSize orgSizeS    = orgSizes[WEIGHT_DIM_S];
    TSize orgSizeC    = orgSizes[WEIGHT_DIM_C];
    TSize orgSizeK    = orgSizes[WEIGHT_DIM_K];
    TSize newSizeR    = newSizes[WEIGHT_DIM_R];
    TSize newSizeS    = newSizes[WEIGHT_DIM_S];
    TSize newSizeC    = newSizes[WEIGHT_DIM_C];
    TSize newSizeK    = newSizes[WEIGHT_DIM_K];
    TSize orgStridesC = orgStrides[WEIGHT_DIM_C];

    unsigned origWeightsSizeInBytes = origWeights->getElementSizeInBytes();

    for (unsigned p = 0; p < packingFactor; ++p)
    {
        uint64_t sOffset = p * stride;    // how many zeros to have before the weights start
        uint64_t newOffK = p * orgSizeK;  // current instance of the duplicated weights
        uint64_t orgOffK = 0;

        for (TSize q = 0; q < orgSizeQ; ++q)
        {
            // copy the original weights to the correct offset in the new weights, for each packing
            for (TSize r = 0; r < orgSizeR; ++r)
            {
                for (TSize s = 0; s < orgSizeS; ++s)
                {
                    // new weights S dimension size is larger, so the rest remains with zeros
                    for (TSize c = 0; c < orgSizeC; ++c)
                    {
                        uint64_t srcFirstElementIdx = orgOffK + orgSizeK * c + orgSizeK * orgSizeC * s +
                                                      orgSizeK * orgSizeC * orgSizeS * r +
                                                      orgSizeK * orgSizeC * orgSizeS * orgSizeR * q;
                        uint64_t dstFirstElementIdx = newOffK + newSizeK * c + newSizeK * newSizeC * (s + sOffset) +
                                                      newSizeK * newSizeC * newSizeS * r +
                                                      newSizeK * newSizeC * newSizeS * newSizeR * q;

                        memcpy(newData + dstFirstElementIdx * origWeightsSizeInBytes,
                               (uint8_t*)data + srcFirstElementIdx * origWeightsSizeInBytes,
                               orgStridesC);
                    }
                }
            }
        }
    }
    origWeights->getTensorAnnotation().dataInfo.packing[PACKING_X] = packingFactor;
}

static bool isPackingNode(const NodePtr& node)
{
    bool                     isPacking = false;
    if (node && HabanaGraph::runsOnTPC(node))
    {
        std::string_view guidName = static_cast<TPCNode&>(*node).getGUIDWithoutDtype();
        isPacking                 = guidName == "conv_weight_packing_fwd";
    }
    return isPacking;
}

/*
 * handlePacking -
 * Perform weight packing on cpu, store it in the output tensor buffer.
 */
static bool handlePacking(const NodePtr& packNode)
{
    if (!isPackingNode(packNode)) return false;
    LOG_TRACE(CONST_FOLDING, "Node {} is a pack node with static input", packNode->getNodeName());
    HB_ASSERT(packNode->getNumInputs() == 1 && packNode->getNumOutputs() == 1,
              "Node {} is pack with multiple inputs and\\or outputs",
              packNode->getNodeName());

    TensorPtr packInput  = packNode->getInput(0);
    TensorPtr packOutput = packNode->getOutput(0);
    if (packInput == nullptr || packOutput == nullptr)
    {
        LOG_WARN(CONST_FOLDING, "At least one of pack node {} tensors is null, doing nothing", packNode->getNodeName());
        return false;
    }
    HB_ASSERT(HabanaGraph::runsOnTPC(packNode), "expected TPC node");
    const TPCNode& tpcNode = static_cast<TPCNode&>(*packNode);
    HB_ASSERT(tpcNode.getParams() != nullptr, "weight packing params are null");

    if (packInput->getElementType() == packInput->getBufferDataType() &&
        packInput->getElementType() == packOutput->getElementType())
    {
        packOutput->setAsDataTypeMatchData();
    }

    ns_WtPack::Params* wtPackParams = ((ns_WtPack::Params*)tpcNode.getParams());
    if (!GCFG_ENABLE_RUN_ON_CPU_DUMMY_MODE.value())
    {
        packWeights(packInput, packOutput, wtPackParams->stride, wtPackParams->packDegree);
    }

    LOG_TRACE(CONST_FOLDING, "Finished handling pack node {} with static input", packNode->getNodeName());
    return true;
}

static bool handlePermutedTranspose(const NodePtr& transposeNode)
{
    LOG_TRACE(CONST_FOLDING, "Node {} is a transpose node with permuted static input", transposeNode->getNodeName());
    HB_ASSERT(transposeNode->getNumInputs() == 1 && transposeNode->getNumOutputs() == 1,
              "Node {} is transpose with multiple inputs and\\or outputs",
              transposeNode->getNodeName());

    TensorPtr transposeInput  = transposeNode->getInput(0);
    TensorPtr transposeOutput = transposeNode->getOutput(0);
    if (transposeInput == nullptr || transposeOutput == nullptr)
    {
        LOG_WARN(CONST_FOLDING,
                 "At least one of the transpose node {} tensors is null, doing nothing",
                 transposeNode->getNodeName());
        return false;
    }
    HB_ASSERT(transposeInput->getTotalElements() == transposeOutput->getTotalElements(),
              "transpose input and output tensors number of elements isn't equal");

    // Unbind old buffer
    if (transposeOutput->isBound())
    {
        // TODO consider not to unbind if it is persistent?
        transposeOutput->unbind();
    }

    LOG_TRACE(CONST_FOLDING, "Bind the buffer to the transpose output tensor buffer");
    void* transposeOutputBuffer = transposeInput->getData();
    transposeOutput->setTensorBuffer(transposeOutputBuffer,
                                     transposeInput->getBufferSizeInBytes(),
                                     transposeInput->getBufferDataType(),
                                     true);

    LOG_TRACE(CONST_FOLDING, "Finished handling transpose node {} with permuted static input", transposeNode->getNodeName());
    return true;
}

static bool isMultNode(const NodePtr& node)
{
    if (node && HabanaGraph::runsOnTPC(node))
    {
        std::string_view guidName = static_cast<TPCNode&>(*node).getGUIDWithoutDtype();
        if (guidName == "mult_fwd" || guidName == "mult")
        {
            return true;
        }
    }
    return false;
}

template<typename T>
static void multTensorWithNumCVector(T*                                 pFirstIn,
                                     T*                                 pSecondIn,
                                     T*                                 pOut,
                                     QuantizationUtils::channelsIndices channelsIndices,
                                     unsigned                           numChannels)
{
    T mult_val = 1;
    for (unsigned channel = 0; channel < numChannels; ++channel)
    {
        mult_val = pSecondIn[channel];
        for (unsigned idx : channelsIndices[channel])
        {
            pOut[idx] = T(float(pFirstIn[idx]) * float(mult_val));
        }
    }
}

template<typename T>
static bool multTensors(const NodePtr&   multNode,
                        const TensorPtr& firstIn,
                        const TensorPtr& secondIn,
                        const TensorPtr& out,
                        bool             multFullSize,
                        bool             broadcastNumCVector,
                        unsigned         numChannels,
                        unsigned         channelDimIndex)
{
    void* buffer = out->getAddress();
    if (buffer == nullptr)
    {
        LOG_TRACE(CONST_FOLDING, "Allocating new buffer to mult output {} according to input buffer type", out->getName());
        buffer = allocateBufferForSynType(firstIn->getBufferDataType(), firstIn->getTotalElements());
        out->setTensorBuffer(buffer, firstIn->getTotalElements(), firstIn->getBufferDataType(), false);
        out->setShouldFreeBuffer(true);
    }

    T* pOut      = static_cast<T*>(buffer);
    T* pFirstIn  = static_cast<T*>(firstIn->map());
    T* pSecondIn = static_cast<T*>(secondIn->map());
    if (broadcastNumCVector)
    {
        QuantizationUtils::channelsIndices channelsIndices =
            QuantizationUtils::calcChannelsIndices(firstIn, channelDimIndex);
        HB_ASSERT(channelsIndices.size() == numChannels, "mismatch between num of indices group and num of channels");
        multTensorWithNumCVector<T>(pFirstIn, pSecondIn, pOut, channelsIndices, numChannels);
    }
    else
    {
        T mult_val = pSecondIn[0];
        for (unsigned i = 0; i < out->getTotalElements(); ++i)
        {
            if (multFullSize)
            {
                mult_val = pSecondIn[i];
            }
            pOut[i] = T(float(pFirstIn[i]) * float(mult_val));
        }
    }
    return true;
}

static bool handleMult(const NodePtr& multNode)
{
    if (!isMultNode(multNode)) return false;
    LOG_TRACE(CONST_FOLDING, "Node {} is a mult node with static input", multNode->getNodeName());
    TensorPtr firstIn  = multNode->getInput(0);
    TensorPtr secondIn = multNode->getInput(1);
    TensorPtr out      = multNode->getOutput(0);

    unsigned firstInDim   = firstIn->getDim();
    unsigned secondInDim  = secondIn->getDim();
    bool     multFullSize = false;
    if (firstInDim == secondInDim)
    {
        for (unsigned i = 0; i < firstInDim; ++i)
        {
            if (firstIn->getSizeInElements(i) != secondIn->getSizeInElements(i))
            {
                LOG_ERR(CONST_FOLDING, "Mult node folding failed- size mismatch of operands, same dim but different shape");
                return false;
            }
        }
        multFullSize = true;
    }
    else
    {
        // set the tensor that needs to be broadcasted to be the second one
        if (secondInDim != 1 && firstInDim == 1)
        {
            std::swap(firstIn, secondIn);
            std::swap(firstInDim, secondInDim);
        }
    }
    unsigned channelDimIndex     = CHANNEL_INDEX;
    unsigned numChannels         = firstIn->getSizeInElements(channelDimIndex);
    bool     broadcastSingleElem = (secondInDim == 1 && secondIn->getSizeInElements(0) == 1);
    bool     broadcastNumCVector = (secondInDim == 1 && secondIn->getSizeInElements(0) == numChannels);
    if (!multFullSize && !broadcastNumCVector && !broadcastSingleElem)
    {
        LOG_DEBUG(CONST_FOLDING,
                  "Folding mult node {} supports only broadcast of single element or vector at the size of channels",
                  multNode->getNodeName());
        return false;
    }
    if (out->getElementType() != firstIn->getElementType() || out->getElementType() != secondIn->getElementType())
    {
        LOG_DEBUG(CONST_FOLDING,
                  "output and inputs of mult node {} are not of the same data-type, doing nothing",
                  multNode->getNodeName());
        return false;
    }

    unsigned inSize  = firstIn->getTotalElements();
    unsigned outSize = out->getTotalElements();
    HB_ASSERT(inSize == outSize, "size mismatch");
    switch (firstIn->getElementType())
    {
        case syn_type_float:
            return multTensors<AsCppType<syn_type_float>>(multNode,
                                                          firstIn,
                                                          secondIn,
                                                          out,
                                                          multFullSize,
                                                          broadcastNumCVector,
                                                          numChannels,
                                                          channelDimIndex);
        case syn_type_bf16:
            return multTensors<AsCppType<syn_type_bf16>>(multNode,
                                                         firstIn,
                                                         secondIn,
                                                         out,
                                                         multFullSize,
                                                         broadcastNumCVector,
                                                         numChannels,
                                                         channelDimIndex);
        case syn_type_int16:
            return multTensors<AsCppType<syn_type_int16>>(multNode,
                                                          firstIn,
                                                          secondIn,
                                                          out,
                                                          multFullSize,
                                                          broadcastNumCVector,
                                                          numChannels,
                                                          channelDimIndex);
        case syn_type_uint16:
            return multTensors<AsCppType<syn_type_uint16>>(multNode,
                                                           firstIn,
                                                           secondIn,
                                                           out,
                                                           multFullSize,
                                                           broadcastNumCVector,
                                                           numChannels,
                                                           channelDimIndex);
        case syn_type_int32:
            return multTensors<AsCppType<syn_type_int32>>(multNode,
                                                          firstIn,
                                                          secondIn,
                                                          out,
                                                          multFullSize,
                                                          broadcastNumCVector,
                                                          numChannels,
                                                          channelDimIndex);
        case syn_type_uint32:
            return multTensors<AsCppType<syn_type_uint32>>(multNode,
                                                           firstIn,
                                                           secondIn,
                                                           out,
                                                           multFullSize,
                                                           broadcastNumCVector,
                                                           numChannels,
                                                           channelDimIndex);
        default:
            LOG_DEBUG(CONST_FOLDING, "data type is not supported for MultNode folding");
            return false;
    }
    return true;
}

// if mult node can be folded, when this function return- firstIn will point to the input in the const section
static bool canMultBeRemoved(const NodePtr& multNode, TensorPtr& firstIn, TensorPtr& secondIn)
{
    if (firstIn->inConstSection() == secondIn->inConstSection())
    {
        LOG_TRACE(CONST_FOLDING,
                  "Cannot eliminate node: {}, there isn't only one input with const section. tensors: "
                  "{} and {}",
                  multNode->getNodeName(),
                  firstIn->getName(),
                  secondIn->getName());
        return false;
    }
    if (secondIn->inConstSection())
    {
        std::swap(firstIn, secondIn);
    }
    if (!secondIn->isStaticParam())
    {
        LOG_TRACE(CONST_FOLDING,
                  "Cannot eliminate node: {}, the second input is not static. tensors: "
                  "{} and {}",
                  multNode->getNodeName(),
                  firstIn->getName(),
                  secondIn->getName());
        return false;
    }
    return true;
}

static bool shouldSkipElimination(const NodePtr& node)
{
    for (const TensorPtr& t : node->getOutputs())
    {
        // Cannot set persistent as static tensor
        if (t->isPersistent())
        {
            LOG_TRACE(CONST_FOLDING,
                      "Cannot eliminate node: {}, due to persistent output tensor: {}",
                      node->getNodeName(),
                      t->getName());
            return true;
        }
    }

    TensorPtr input = node->getInput(0);
    // Cannot remove persistent tensors if no Const section exist
    if (input->isPersistent() && !input->inConstSection())
    {
        LOG_TRACE(CONST_FOLDING,
                  "Cannot eliminate node: {}, due to persistent (with no Const Section) input tensor: {}",
                  node->getNodeName(),
                  input->getName());
        return true;
    }
    return false;
}

static bool isNodeCandidateForElimination(const NodePtr& n)
{
    // TODO [SW-141316] support elimination of split with static inputs
    // n->getNodeType() == Node::TYPE_INTERNAL_SPLIT ||
    bool isCandidate =
        (n->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE || n->getNodeType() == Node::TYPE_INTERNAL_RESHAPE ||
         n->getNodeType() == Node::TYPE_INTERNAL_EXPAND_DIMS || n->isCast() || isMultNode(n));

    if (GCFG_ENABLE_WEIGHT_PACKING_CONSTANT_FOLDING.value())
    {
        isCandidate = isCandidate || isPackingNode(n);
    }
    else
    {
        //TODO: SW-154419
        LOG_DEBUG(CONST_FOLDING, "{}: constant folding of conv_weight_packing is disabled", __FUNCTION__);
    }

    return isCandidate;
}

bool eliminateNodesWithStaticInputs(HabanaGraph &g)
{
    if (!GCFG_ENABLE_CONSTANT_FOLDING.value())
    {
        LOG_DEBUG(CONST_FOLDING, "{} pass is disabled ", HLLOG_FUNC);
        return true;
    }
    // We are not expecting static tensors in training flow,
    // so there is no use in running this pass
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(CONST_FOLDING, "Not in inference mode, skipping {} pass", HLLOG_FUNC);
        return true;
    }
    //Get all nodes in topological order
    const NodeVector& graphNodes = g.getTopoSortedNodes();
    NodeVector        nodesToRemove;
    for (const NodePtr &n : graphNodes)
    {
        if (isNodeCandidateForElimination(n))
        {
            if (graphNodes.size() - nodesToRemove.size() <= 1)
            {
                // prevent eliminating the last node in the graph
                break;
            }

            // We currently support only nodes with one input
            // or mult with 2 inputs that only one of his inputs is a const tensor (with a const section)
            // due to handling const sections
            if (n->getNumInputs() != 1 && !isMultNode(n)) continue;

            TensorPtr input = n->getInput(0);
            HB_ASSERT_PTR(input);
            if (isMultNode(n))
            {
                TensorPtr secondInput = n->getInput(1);
                HB_ASSERT_PTR(secondInput);
                if (!canMultBeRemoved(n, input, secondInput)) continue;
            }

            // Remove only nodes that all of their input tensors are static and un-permuted
            // Allow permuted inputs to cast since layout is irrelevant to cast op.
            if (!input->isStaticParam())
            {
                LOG_TRACE(CONST_FOLDING,
                          "Cannot eliminate node: {} due to a non-static input tensor: {}",
                          n->getNodeName(),
                          input->getName());
                continue;
            }

            if (input->isDynamicShape())
            {
                LOG_TRACE(CONST_FOLDING,
                          "node: {} with input {} has dynamic shape, skipping elimination",
                          n->getNodeName(),
                          input->getName());
                continue;
            }

            bool is_output_ds = false;
            for (const TensorPtr& t : n->getOutputs())
            {
                if (t->isDynamicShape())
                {
                    LOG_TRACE(CONST_FOLDING,
                              "node: {} with output {} has dynamic shape, skipping elimination",
                              n->getNodeName(),
                              t->getName());
                    is_output_ds = true;
                    break;;
                }
            }
            if (is_output_ds) continue;

            const auto& permutedInputPerm = input->getPermutation();
            bool        permutedTranspose = false;
            if (permutedInputPerm && !permutedInputPerm.value().isIdentity())
            {
                // transpose with permuted input case
                if (n->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE &&
                    isDenseAfterPermute(input, permutedInputPerm.value()))
                {
                    permutedTranspose = true;
                }
                else if (!n->isCast())
                {
                    LOG_TRACE(CONST_FOLDING,
                              "Cannot eliminate node: {} due to a permuted input tensor: {}",
                              n->getNodeName(),
                              input->getName());
                    continue;
                }
            }

            if (shouldSkipElimination(n)) continue;

            if ((permutedTranspose && handlePermutedTranspose(n)) || handleCast(n) || handlePacking(n) ||
                handleMult(n) || n->RunOnCpu())
            {
                LOG_DEBUG(CONST_FOLDING, "Eliminate node: {}", n->getNodeName());
                nodesToRemove.push_back(n);
                const TensorVector outputTensors = n->getOutputs();
                for (const TensorPtr &t : outputTensors)
                {
                    if ((!n->isCast() && !t->isDenseLayout()) || t->isAliasedTensor())
                    {
                        LOG_ERR(CONST_FOLDING, "Eliminated node {} output is not dense", n->getNodeName());
                        return false;
                    }
                    t->setAsStaticParam(true);
                }

                // Save input const section info into output tensors.
                if (input->inConstSection())
                {
                    ConstSectionReplacer::replace(input, outputTensors, g.getTensorConsumers(input).size() == 1);
                }
            }
        }
    }
    LOG_TRACE(CONST_FOLDING, "Number of nodes with static inputs to eliminate - {}", nodesToRemove.size());
    GraphEditor::removeNodes(g, nodesToRemove);

    return true;
}
