#include "node_displacement.h"

// eager includes (relative to src/eager/lib/)
#include "chip_info.h"
#include "desc_gen/tpc_desc_base.h"
#include "eager_graph.h"
#include "node_info/eager_memset_node_output.h"
#include "node_info/eager_node.h"
#include "node_info/suggested_tensor_manipulation.h"
#include "tpc_node_handler.h"
#include "utils/algorithm_utils.h"
#include "utils/general_defs.h"
#include "utils/string_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/data_type_utils.h"
#include "graph_compiler/habana_nodes/dma_memcopy_node.h"
#include "graph_compiler/habana_nodes/logical_op_node.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "graph_compiler/habana_nodes/node_tensor_accessor.h"
#include "graph_compiler/habana_nodes/transpose_nodes_creator.h"
#include "graph_compiler/input_reuse_binding.h"
#include "graph_compiler/layout.h"
#include "graph_compiler/tensor_annotation.h"
#include "graph_compiler/types.h"

// synapse-internal passes includes (relative to src/)
#include "graph_compiler/mme/mme_brain_ifc.h"
#include "graph_compiler/passes/add_mme_bias.h"
#include "graph_compiler/passes/adjust_data_layout.h"
#include "graph_compiler/passes/bn_utils.h"
#include "graph_compiler/passes/cast_nodes_handler.h"
#include "graph_compiler/passes/decode_strided_op.h"
#include "graph_compiler/passes/handle_grouped_convolutions.h"
#include "graph_compiler/passes/handle_huge_tensors.h"
#include "graph_compiler/passes/handle_logical_operations.h"
#include "graph_compiler/passes/handle_tensor_permutations.h"
#include "graph_compiler/passes/pack_convolution_training.h"
#include "graph_compiler/passes/remove_zero_sized_tensors.h"
#include "graph_compiler/passes/transpose_dont_care_nodes.h"
#include "graph_compiler/passes/update_nodes_with_alias_tensors.h"
// synapse api (relative to include/)
#include "synapse_common_types.h"

// std includes
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace eager_mode
{
// Determine if complex GUID lib enabled
static bool isComplexGuidEn()
{
    return GCFG_ENABLE_COMPLEX_GUID_LIB_IN_EAGER.value() != 0;
}

static bool isTpcNodeSupportedAfterInt64PrecisionReduction(const EagerNode& node)
{
    const auto& guid = node->getGUID();
    if (endsWith(guid, "_i64") || endsWith(guid, "_u64"))
    {
        std::string_view guidWithoutDtype = extractGUIDFromFullGUID(node->getGUID());
        if (guidWithoutDtype == "memcpy_nd" || guidWithoutDtype == "broadcast_nd_fwd")
        {
            return true;
        }
    }
    return false;
}

static TensorPtr getExpandedTensor(const Tensor& src)
{
    EAGER_ASSERT(src.getDim() < tpc_lib_api::MAX_TENSOR_DIM, "can't expand ndim tensor beyond maximal size");
    if (unlikely(src.getDim() == tpc_lib_api::MAX_TENSOR_DIM)) return nullptr;
    TensorPtr         reshapeOutput = src.clone(false, false);
    const NSizeArray& inSizes       = src.getAllNSizesInElements();
    NSizeArray        newSizes      = {};
    newSizes[0]                     = 1;
    std::copy_n(inSizes.begin(), src.getDim(), newSizes.begin() + 1);
    reshapeOutput->reshape(src.getDim() + 1, newSizes.data());
    return reshapeOutput;
}

NodeDisplacement::NodeDisplacement(EagerGraph& eagerGraph, EagerNodes& curNodes)
: m_eagerGraph(eagerGraph),
  m_nodes(curNodes),
  m_constantTensorOptimizer(ChipInfo::getRecipeHal(eagerGraph.getChipType())),
  m_enableComplexGuidLib(isComplexGuidEn()),
  m_nodeCollector(*this),
  m_complexGuidExtractor(eagerGraph.getDeviceId(), m_constantTensorOptimizer)
{
}

static bool checkSramUsage(const EagerNode& node, bool checkInputs)
{
    const TensorVector& tensors = checkInputs ? node->getInputs() : node->getOutputs();
    for (const auto& tensor : tensors)
    {
        if ((tensor != nullptr) && tensor->inSram())
        {
            EAGER_REPORT_ERROR("{}: Node {} has {} {} in SRAM. SRAM is not supported at Eager mode",
                               HLLOG_FUNC,
                               node->getGUID(),
                               checkInputs ? "input" : "output",
                               tensor->getName());
            return false;
        }
    }
    return true;
}

static bool hasZeroSizeTensors(const EagerNode& node)
{
    for (const TensorVector* ts : {&node->getInputs(), &node->getOutputs()})
    {
        for (const TensorPtr& t : *ts)
        {
            if (t && t->isZeroSizedDataTensor()) return true;
        }
    }
    return false;
}

static bool isTPCBroadcastSupported(const EagerNode& node)
{
    const TensorPtr& broadcastOutput = node->getOutput(0);
    if (unlikely(!broadcastOutput)) return false;
    switch (broadcastOutput->getElementType())
    {
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_fp8_152:
        case syn_type_fp8_143:
        case syn_type_fp16:
        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_single:
            // int\uint64 are supported after precision reduction
        case syn_type_int64:
        case syn_type_uint64:
            return true;
        default:
            return false;
    }
}

// Check if the given node supported in Eager mode
bool NodeDisplacement::isNodeSupported(const EagerNode& node, bool userNode) const
{
// Compact reporting macro:
// It's a warning for user node as it's possible to fall back to graph mode,
// while internal nodes must report an error as it's too late to fall back.
#define SUPPORTED_NODE_RETURN_WITH_ERR(msg)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (userNode)                                                                                                  \
        {                                                                                                              \
            EAGER_LOG_WARN(msg, HLLOG_FUNC, node->getNodeName(), node->getGUID());                                     \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            EAGER_REPORT_ERROR(msg, HLLOG_FUNC, node->getNodeName(), node->getGUID());                                 \
        }                                                                                                              \
        return false;                                                                                                  \
    } while (0)

    EAGER_ASSERT(!node->isDynamicShape(), "Dynamic shape should be excluded at node addition for Eager");

    // No SRAM support for Eager yet
    if (!checkSramUsage(node, true) || !checkSramUsage(node, false))
    {
        SUPPORTED_NODE_RETURN_WITH_ERR("{}: Node '{}' with guid '{}' uses SRAM. No support for SRAM in Eager mode");
    }

    if (node->getNodeType() == Node::TYPE_BROADCAST && !isTPCBroadcastSupported(node))
    {
        SUPPORTED_NODE_RETURN_WITH_ERR(
            "{}: Node '{}' with guid '{}' has unsupported data type for TPC broadcast kernel");
    }

    // TODO [SW-152705]: Support natively in Eager-Mode
    if (node->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        SUPPORTED_NODE_RETURN_WITH_ERR("{}: Node '{}' has unsupported guid '{}'");
    }

    // TPC constraints
    if (node.getEngineType() == EngineType::TPC)
    {
        if (!KernelDB::instance().isKernelExist(node->getGUIDAndHash(), m_eagerGraph.getDeviceId()))
        {
            if (!isTpcNodeSupportedAfterInt64PrecisionReduction(node))
            {
                SUPPORTED_NODE_RETURN_WITH_ERR("{}: TPC kernel for node '{}' with guid '{}' does not exist");
            }
        }

        // Eager recipe gen is restricted to one patchable blob per node.
        // one entry is reserved for the tpc kernel.
        const size_t maxTensorsPerNode = m_eagerGraph.getGraphTraits()->getHalReader()->getBaseRegistersCacheSize() - 1;
        if (TpcDescGeneratorBase::calcNumberPatchableTensors(node) > maxTensorsPerNode)
        {
            SUPPORTED_NODE_RETURN_WITH_ERR(
                "{}: Number of tensors of TPC node '{}' with guid '{}' is not supported in Eager mode");
        }
    }
    // MME constraints
    else if (node.getEngineType() == EngineType::MME)
    {
        // Eager does not support secondary output
        if (node->getOutputs().size() > 1)
        {
            SUPPORTED_NODE_RETURN_WITH_ERR("{}: There are more than one outputs for MME node '{}' with guid '{}'");
        }
    }
    // Non-DMA constraints
    else if (node.getEngineType() != EngineType::DMA)
    {
        switch (node->getNodeType())
        {
            // A black list of unsupported nodes
            case Node::TYPE_TF_FUSED_BATCH_NORM_GRAD:
            case Node::TYPE_MOMENTS:
                SUPPORTED_NODE_RETURN_WITH_ERR("{}: Node '{}' with guid '{}' is not supported in Eager mode");
            default:
                break;  // Complex GUID or other unsupported node that should to be added to the list above
        }
    }

    // Check layout
    if (userNode && !DataLayoutHandler(m_eagerGraph, node).validate())
    {
        SUPPORTED_NODE_RETURN_WITH_ERR("{}: Data layout for node '{}' with guid '{}' is not supported in Eager mode");
    }

#undef SUPPORTED_NODE_ERR_MSG

    return true;
}

// Add internal node to graph. Internal nodes must be supported
bool NodeDisplacement::addInternalNode(EagerNode& node)
{
    EAGER_ASSERT_PTR(node);
    const bool isInternalNodeSupported = isNodeSupported(node, false);
    m_isInternalNodeCheckDone          = true;
    EAGER_ASSERT(isInternalNodeSupported, "Internal nodes must be supported");
    if (isInternalNodeSupported && !processNewNode(node, false))
    {
        EAGER_REPORT_ERROR("{}: Failed to process node {}", HLLOG_FUNC, node->getGUID());
        return false;
    }
    return isInternalNodeSupported;
}

// Check if the given node must turn into multiple ones or its tensors should be modified
bool NodeDisplacement::processNewNode(EagerNode& node, bool userNode)
{
    EAGER_ASSERT(userNode || m_isInternalNodeCheckDone || isNodeSupported(node, false),
                 "Internal nodes must be supported");
    m_isInternalNodeCheckDone = false;

    node.get()->setGraphTraits(m_eagerGraph.getGraphTraits());

    if (userNode)
    {
        AddNodeResult res = processNewUserNode(node);
        if (res != AddNodeResult::SUCCESS_ADD_REQUIRED) return (res != AddNodeResult::FAIL);
    }

    if (hasZeroSizeTensors(node))
    {
        TensorPtr subTensor;
        if (ZeroSizedTensorRemover(m_eagerGraph).handleZeroSizedOperand(node, subTensor))
        {
            // The node isn't added but the consumers of its output needs adjustment
            return editConsumersForZST(node, subTensor);
        }
        // continue as if not ZST, adding the node as normal
    }

    if (node->isLogicalOperation())
    {
        m_nodeCollector.collectNode(node, true);
        return true;
    }

    // Handling of specific operators
    {
        const AddNodeResult res = handleSpecialOperators(node, false /*handleOnlyConstOptimization*/, userNode);
        if (res != AddNodeResult::SUCCESS_ADD_REQUIRED) return (res != AddNodeResult::FAIL);
    }

    // Before extracting into nodes, handle huge tensors of non-TPC nodes. TPC nodes are handled by complex GUID.
    if (node.getEngineType() != EngineType::TPC)
    {
        // add a cheap check for early exit
        bool anyHugeTensor = false;
        auto isHugeTensor  = [](const TensorPtr& t) { return t && t->getDenseSizeInBytes() > HW_DENSE_TENSOR_LIMIT; };
        for (const TensorVector* operandsPtr : {&node->getInputs(), &node->getOutputs()})
        {
            anyHugeTensor |= std::any_of(operandsPtr->begin(), operandsPtr->end(), isHugeTensor);
            if (anyHugeTensor) break;
        }
        if (anyHugeTensor)
        {
            HugeTensorHandler hugeTensorHandler(m_eagerGraph);
            if (hugeTensorHandler.shouldHandleHugeTensor(node))
            {
                return handleHugeTensor(node, hugeTensorHandler);
            }
        }
    }

    if (node->isMultiNode())
    {
        auto multiNode      = node.get<MultiNode>();
        auto extractedNodes = multiNode->extract(m_eagerGraph);
        if (extractedNodes.empty()) return false;
        for (EagerNode extractedNode : extractedNodes)
        {
            if (!addInternalNode(extractedNode)) return false;
        }
        return true;
    }

    if (userNode && m_enableComplexGuidLib)
    {
        AddNodeResult res = AddNodeResult::SUCCESS_ADD_REQUIRED;
        if (GCFG_ENABLE_COMPLEX_GUID_LIB_IN_EAGER.value() == 1)
        {
            if (EagerComplexGuidExtractor::isNodeNeedsExtract(node,
                                                              FUNCTIONAL_COMPLEX_GUID,
                                                              m_eagerGraph.getDeviceId()))
            {
                res = processNewComplexGuidNode(node);
            }
        }
        else if (node->getGUID() == "non_zero_v2_i8")
        {
            res = processNewComplexGuidNode(node);
        }

        // in case of GLUE_CGUID_GRAPH_UNCHANGED we fallthrough here.
        // calling addInternalNode is forbidden as that would lead to an infinite loop.
        if (res != AddNodeResult::SUCCESS_ADD_REQUIRED) return (res != AddNodeResult::FAIL);
    }

    AddNodeResult res = AddNodeResult::SUCCESS_ADD_REQUIRED;
    if (node.getEngineType() == EngineType::TPC)
    {
        res = processNewTpcNode(node, userNode);
    }
    else if (node.getEngineType() == EngineType::MME)
    {
        res = processNewMmeNode(node);
    }
    if (res != AddNodeResult::SUCCESS_ADD_REQUIRED) return (res != AddNodeResult::FAIL);

    m_nodeCollector.collectNode(node, false);
    return true;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processNewUserNode(const EagerNode& node)
{
    if (AddNodeResult::SUCCESS_NO_ADD_REQUIRED ==
        handleSpecialOperators(node, true /*handleOnlyConstOptimization*/, true /*userNode*/))
    {
        return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
    }

    auto processNewInternalTransposeNodes = [this](const TransposeNodeParamsVector& newTransposeNodesParams) {
        for (const TransposeNodeParams& transposeNodeParams : newTransposeNodesParams)
        {
            EagerModeTransposeTensorPermutationHandler tensorTransposePermutationHandler(transposeNodeParams);
            if (tensorTransposePermutationHandler.canExtract())
            {
                EagerNode newTransposeNode = tensorTransposePermutationHandler.extract();
                if (!addInternalNode(newTransposeNode)) return AddNodeResult::FAIL;
            }
            else
            {
                if (!processTransposeNode(transposeNodeParams)) return AddNodeResult::FAIL;
            }
        }
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    };

    // layout adjustment is not required for nodes without supplied user layouts.
    // we make the cheap check here to aovid class construction overhead and reduce cpu instruction
    // cache misses.
    bool anyLayoutProvided = false;
    auto isLayoutProvided  = [](const Layout& layout) { return !layout.isDontCare(); };
    for (const auto operandsLayoutPtr : {&node->getInputLayouts(), &node->getOutputLayouts()})
    {
        anyLayoutProvided |= std::any_of(operandsLayoutPtr->begin(), operandsLayoutPtr->end(), isLayoutProvided);
        if (anyLayoutProvided) break;
    }
    if (anyLayoutProvided)
    {
        DataLayoutHandler dataLayoutHandler(m_eagerGraph, node);
        if (dataLayoutHandler.canExtract())
        {
            const TransposeNodeParamsVector& newTransposeNodesParams = dataLayoutHandler.extract(m_eagerGraph);
            return processNewInternalTransposeNodes(newTransposeNodesParams);
        }
    }

    // don't care nodes and permuted tensor handling is only required for nodes with permuted operands.
    // so we make the cheap check here for early exit.
    auto isPermutedTensor = [](const TensorPtr& t) {
        if (t == nullptr) return false;
        const std::optional<Permutation>& permutedInputPerm = t->getPermutation();
        return permutedInputPerm && !permutedInputPerm->isIdentity();
    };
    bool anyPermuted = false;
    for (const auto operandsPtr : {&node->getInputs(), &node->getOutputs()})
    {
        anyPermuted |= std::any_of(operandsPtr->begin(), operandsPtr->end(), isPermutedTensor);
        if (anyPermuted) break;
    }
    if (!anyPermuted) return AddNodeResult::SUCCESS_ADD_REQUIRED;

    EagerModeTransposeDontCareNodesHandler dontCareNodeHandler(m_eagerGraph, node);
    if (dontCareNodeHandler.canExtract())
    {
        const TransposeNodeParamsVector& newTransposeNodesParams = dontCareNodeHandler.extract();
        if (processNewInternalTransposeNodes(newTransposeNodesParams) == AddNodeResult::FAIL)
        {
            return AddNodeResult::FAIL;
        }
        EagerNode newReplacementNode = dontCareNodeHandler.fixupNode();
        if (newReplacementNode.get() != nullptr)
        {
            if (!addInternalNode(newReplacementNode)) return AddNodeResult::FAIL;
            return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
        }
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    }
    EagerModeTensorPermutationHandler tensorPermutationHandler(node);
    if (tensorPermutationHandler.canExtract())
    {
        for (EagerNode transposeNode : tensorPermutationHandler.extract())
        {
            if (!addInternalNode(transposeNode)) return AddNodeResult::FAIL;
        }
    }
    return AddNodeResult::SUCCESS_ADD_REQUIRED;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processStridedOp(const EagerNode& node)
{
    if (!StridedOpDecoder::canExtract(node))
    {
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    }
    NodeVector extractedNodes = StridedOpDecoder::extract(node, true /*changeInPlace*/);
    // case where decoding was a nop
    if (extractedNodes.empty())
    {
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    }
    for (EagerNode node : extractedNodes)
    {
        if (!addInternalNode(node)) return AddNodeResult::FAIL;
    }
    return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
}

NodeDisplacement::AddNodeResult
NodeDisplacement::handleSpecialOperators(const EagerNode& node, bool handleOnlyConstOptimization, bool userNode)
{
    if (handleOnlyConstOptimization)
    {
        if (m_constantTensorOptimizer.tryReplaceNodeByConstTensor(node))
        {
            return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
        }
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    }

    if (m_nodeCollector.shouldPostponeNodeProcessing(node))
    {
        m_nodeCollector.collectNode(node, false);
        return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
    }

    AddNodeResult res = AddNodeResult::SUCCESS_ADD_REQUIRED;
    switch (node->getNodeType())
    {
        case Node::TYPE_STRIDED_VIEW:
        {
            if (userNode)
            {
                res = processStridedOp(node);
            }
        }
        break;
        // TODO enable as part of SW-161084 after SW-176025
        // is resolved.
        // case Node::TYPE_STRIDED_INSERT:
        // {
        //     if (userNode)
        //     {
        //         res = processStridedOp(node);
        //     }
        // }
        // break;
        case Node::TYPE_MEMSET:
        {
            EagerNode newNode = getPhysicalMemset(node->getOutput(0)->getElementType(),
                                                  node->getInputs(),
                                                  node->getOutputs(),
                                                  node->getNodeName());
            if (newNode != nullptr)
            {
                newNode->getNodeAnnotation() = node->getNodeAnnotation();
                res = addInternalNode(newNode) ? AddNodeResult::SUCCESS_NO_ADD_REQUIRED : AddNodeResult::FAIL;
            }
        }
        break;
        case Node::TYPE_MEMCOPY:
        {
            EagerNode newNode = createPhysicalMemcpy(node->getInput(0), node->getOutput(0), node->getNodeName());
            if (newNode != nullptr)
            {
                newNode->getNodeAnnotation() = node->getNodeAnnotation();
                res = addInternalNode(newNode) ? AddNodeResult::SUCCESS_NO_ADD_REQUIRED : AddNodeResult::FAIL;
            }
        }
        break;
        case Node::TYPE_BROADCAST:
        {
            res = processNewBroadcastNode(node);
        }
        break;
        case Node::TYPE_USER:
        {
            if (GCFG_ENABLE_EAGER_NODE_DISPLACEMENT_OPTIMIZATIONS.value() &&
                GCFG_ENABLE_BATCH_NORM_SPLIT_IN_EAGER.value())
            {
                res = splitBatchNorm(node);
            }
        }
        break;
        default:
            break;
    };
    return res;
}

NodeDisplacement::AddNodeResult NodeDisplacement::splitBatchNorm(const EagerNode& node)
{
    const auto& tpcNode = *node.get<TPCNode>();
    if (likely(tpcNode.isGuidPrefix("batch_norm_") == false))
    {
        return AddNodeResult::SUCCESS_ADD_REQUIRED;
    }
    NodeList    bnNewNodeList;
    if (unlikely(tpcNode.isGuidPrefix("batch_norm_fwd_bf16")))
    {
        bnNewNodeList = splitBatchNormFwd(tpcNode, syn_type_bf16);
    }
    else if (unlikely(tpcNode.isGuidPrefix("batch_norm_bwd_bf16")))
    {
        bnNewNodeList = splitBatchNormBwd(tpcNode, syn_type_bf16);
    }
    else if (unlikely(tpcNode.isGuidPrefix("batch_norm_fwd_f32")))
    {
        bnNewNodeList = splitBatchNormFwd(tpcNode, syn_type_float);
    }
    else if (unlikely(tpcNode.isGuidPrefix("batch_norm_bwd_f32")))
    {
        bnNewNodeList = splitBatchNormBwd(tpcNode, syn_type_float);
    }

    if (unlikely(!bnNewNodeList.empty()))
    {
        for (EagerNode node : bnNewNodeList)
        {
            if (unlikely(!addInternalNode(node)))
            {
                return AddNodeResult::FAIL;
            }
        }
        return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
    }
    return AddNodeResult::SUCCESS_ADD_REQUIRED;
}

NodeList NodeDisplacement::splitBatchNormFwd(const TPCNode& fullBatchNorm, synDataType dtype)
{
    if (dtype != syn_type_float && dtype != syn_type_bf16)
    {
        EAGER_REPORT_ERROR("splitBatchNorm: Unsupported dtype {}", dtype);
        return {};
    }

    BNUtils::Bn1Bn2FwdInputs  inputs  = {fullBatchNorm.getInput(0),
                                       fullBatchNorm.getInput(1),
                                       fullBatchNorm.getInput(2),
                                       fullBatchNorm.getInput(3),
                                       fullBatchNorm.getInput(4),
                                       nullptr};
    BNUtils::Bn1Bn2FwdOutputs outputs = {fullBatchNorm.getOutput(0),
                                         fullBatchNorm.getOutput(1),
                                         fullBatchNorm.getOutput(2),
                                         fullBatchNorm.getOutput(3),
                                         fullBatchNorm.getOutput(4)};

    // Save the original node's parameters
    ns_BatchNormKernel::Params* fullBnParams = static_cast<ns_BatchNormKernel::Params*>(fullBatchNorm.getParams());
    if (fullBnParams == nullptr)
    {
        EAGER_REPORT_ERROR("splitBatchNorm- node {} parameters is NULL", fullBatchNorm.getNodeName());
        return {};
    }

    bool isTraining = true;
    if (fullBatchNorm.getParamsSize() == sizeof(ns_BatchNormKernel::ParamsV2))
    {
        isTraining = static_cast<ns_BatchNormKernel::ParamsV2*>(fullBnParams)->isTraining;
    }

    NodeList   bn1bn2NodeList;
    const bool retVal = BNUtils::createBn1Bn2NodesFwd(inputs,
                                                      outputs,
                                                      fullBnParams->momentum,
                                                      fullBnParams->epsilon,
                                                      fullBatchNorm.getNodeName(),
                                                      dtype,
                                                      isTraining,
                                                      bn1bn2NodeList,
                                                      /*locateInSram*/ false);
    if (!retVal) return {};
    return bn1bn2NodeList;
}

NodeList NodeDisplacement::splitBatchNormBwd(const TPCNode& fullBatchNorm, synDataType dtype)
{
    if (dtype != syn_type_float && dtype != syn_type_bf16)
    {
        EAGER_REPORT_ERROR("splitBatchNorm: Unsupported dtype {}", dtype);
        return {};
    }

    bool isTraining = true;
    if (fullBatchNorm.getParamsSize() == sizeof(ns_BatchNormKernel::ParamsV2))
    {
        isTraining = static_cast<ns_BatchNormKernel::ParamsV2*>(fullBatchNorm.getParams())->isTraining;
    }

    BNUtils::Bn1Bn2BwdInputs  inputs  = {fullBatchNorm.getInput(0),
                                       fullBatchNorm.getInput(1),
                                       fullBatchNorm.getInput(2),
                                       fullBatchNorm.getInput(3),
                                       fullBatchNorm.getInput(4)};
    BNUtils::Bn1Bn2BwdOutputs outputs = {fullBatchNorm.getOutput(0),
                                         fullBatchNorm.getOutput(2),
                                         fullBatchNorm.getOutput(1)};

    NodeList   bn1bn2NodeList;
    const bool retVal = BNUtils::createBn1Bn2NodesBwd(inputs,
                                                      outputs,
                                                      fullBatchNorm.getNodeName(),
                                                      dtype,
                                                      isTraining,
                                                      bn1bn2NodeList,
                                                      /*locateInSram*/ false);
    if (!retVal) return {};
    return bn1bn2NodeList;
}

bool NodeDisplacement::processTransposeNode(const TransposeNodeParams& nodeParams)
{
    for (EagerNode node : TransposeNodesCreator().getTransposeNodesByParams(nodeParams))
    {
        if (unlikely(!addInternalNode(node))) return false;
    }
    return true;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processNewMmeNode(const EagerNode& node)
{
    // override since the c'tor uses the graph-mode env var
    node.get<MmeNode>()->getNodeAnnotation().mmeMetaData.mmeStrategy.batchConcurrencyEn =
        GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value() && GCFG_ENABLE_EAGER_BATCH_CONCURRENCY.value()
            ? MmeCommon::TurnedOn
            : MmeCommon::TurnedOff;

    Node::eNodeType nodeType = node.get()->getNodeType();
    // Remove NULL tensors from input
    {
        const TensorVector& inputs   = node->getInputs();
        size_t              inputNum = inputs.size();
        // In Gaudi3 we use mme for transpose as well
        if (nodeType == Node::TYPE_INTERNAL_TRANSPOSE)
        {
            if (inputNum != 1 || inputs[0] == nullptr) return AddNodeResult::FAIL;
        }
        else
        {
            // Expect exactly 2 or 3 first input tensors to be non-nullptr and the rest to be nullptr
            if (inputNum < 2 || inputs[0] == nullptr || inputs[1] == nullptr) return AddNodeResult::FAIL;
            const size_t newSize = inputNum > 2 && inputs[2] != nullptr ? 3 : 2;
            if (newSize != inputNum)
            {
                for (size_t i = newSize; i < inputNum; ++i)
                {
                    if (inputs[i] != nullptr) return AddNodeResult::FAIL;
                }
                node.get()->removeDataInputsFromIndex(newSize);
            }
        }
    }

    // bias extraction manipulates the gemm node
    if (MmeBiasNodeHandler::canExtract(node))
    {
        EagerNode newNode(MmeBiasNodeHandler(node).extract());  // This suppose to be the new "add" node
        if ((newNode != nullptr) && !addInternalNode(newNode)) return AddNodeResult::FAIL;
    }

    if (nodeType == Node::TYPE_INTERNAL_TRANSPOSE || nodeType == Node::TYPE_GEMM || nodeType == Node::TYPE_BATCH_GEMM)
        return AddNodeResult::SUCCESS_ADD_REQUIRED;

    GroupedConvolutionManagerTraining groupedConvolutionMgr(node);
    if (groupedConvolutionMgr.canExtract())
    {
        if (!groupedConvolutionMgr.validateGroupedConvolutionNode())
        {
            return AddNodeResult::FAIL;
        }
        auto newNodes = groupedConvolutionMgr.extract(m_eagerGraph);
        if (newNodes.empty())
        {
            return AddNodeResult::FAIL;
        }
        for (EagerNode n : newNodes)
        {
            if (!addInternalNode(n)) return AddNodeResult::FAIL;
        }
        return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
    }

    if (GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value() && GCFG_ENABLE_CONV_PACKING_EAGER.value())
    {
        // packing manipulates the convolution node
        EagerConvolutionPackingManagerTraining packingMgr(m_eagerGraph, node);
        if (packingMgr.canExtract())
        {
            auto [weightsPackingPtr, outputUnpackingPtr] = packingMgr.extract();
            EagerNode weightsPacking {std::move(weightsPackingPtr)};
            EagerNode outputUnpacking {std::move(outputUnpackingPtr)};
            if (!addInternalNode(weightsPacking)) return AddNodeResult::FAIL;
            if (!addInternalNode(outputUnpacking)) return AddNodeResult::FAIL;
        }
    }

    // This implementation produces non-deterministic results because the order of writes in the MME is non
    // deterministic
    if (!handleMmeConcurrency(node))
    {
        return AddNodeResult::FAIL;
    }

    return AddNodeResult::SUCCESS_ADD_REQUIRED;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processNewTpcNode(EagerNode& node, bool userNode)
{
    if (!handle64bitPrecisionNode(node)) return AddNodeResult::FAIL;

    const auto& tpcNode = node.getSafePtr<TPCNode>();
    AliasTensors::updateNodesWithAliasTensors(tpcNode);

    // Handle suggested manipulation - a runtime optimization
    if (!m_disableSuggestedManipulation && node.isSuggestedManipulationRequired() &&
        !EagerModeSuggestedManipulationHandler::shouldSkipSuggestedTensorManipulation(*tpcNode, m_eagerGraph))
    {
        EagerModeSuggestedManipulationHandler handler(m_eagerGraph, *tpcNode);
        if (handler.isSuggestedTensorManipulationAvailable())
        {
            if (handler.applySuggestedTensorManipulation())
            {
                const NodeVector& nodesToAdd = handler.extract();
                for (EagerNode newNode : nodesToAdd)
                {
                    newNode.setSuggestedManipulationNotRequired();
                    if (!addInternalNode(newNode)) return AddNodeResult::FAIL;
                }
                node.setSuggestedManipulationNotRequired();
                if (!addInternalNode(node)) return AddNodeResult::FAIL;
                return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
            }
        }
        node.setSuggestedManipulationNotRequired();
    }

    if (glue::loadKernelAndAllocAuxTensors(m_eagerGraph, *tpcNode) == false) return AddNodeResult::FAIL;

    // This uses the memsetTensors set by the kernel loading function
    TensorVector memsetTensors = tpcNode->getMemsetBeforeExecTensors();
    for (const TensorPtr& tensor : memsetTensors)
    {
        if (tensor->isReductionEnabled(true) || tensor->isPartOfRMWSection()) continue;
        EagerMemsetNodeOutputManager memsetNodeOutputManager(m_eagerGraph, tensor);
        auto [inTensor, memsetNodePtr, reduceNodePtr] = memsetNodeOutputManager.extract(*tpcNode);
        tpcNode->replaceTensor(tensor, inTensor);
        EagerNode memsetNode {std::move(memsetNodePtr)};
        EagerNode reduceNode {std::move(reduceNodePtr)};
        if (!addInternalNode(memsetNode)) return AddNodeResult::FAIL;
        if (!addInternalNode(reduceNode)) return AddNodeResult::FAIL;
    }

    // We do not use node removal + readdition for the tensor replacement to avoid recursing into this function.
    // But that means that the addition here has to come after the tensor replacement to avoid the lemon graph being
    // out of date.
    // This relies on the kernel loading not actually caring on the tpcNode having been already added to the graph.

    // check reusability
    const std::map<TensorPtr, TensorVector, TensorComparator>& reusePairs = tpcNode->getReusableInputBinding();
    for (const auto& candidates : reusePairs)
    {
        TensorPtr           output         = candidates.first;
        const TensorVector& reusableInputs = candidates.second;
        if (!reusableInputs.empty())
        {
            TensorPtr                inputTensor = reusableInputs[0];
            InputInplaceReuseBinding inplaceReuse;
            if (!inplaceReuse.isAlreadyReused(m_eagerGraph, inputTensor, *tpcNode))
            {
                // TODO[SW-127069]: HabanaGraph::getNumberOfTensorConsumers is always empty here!
                // In graph mode logical pass is being run after we have ran the inPlaceInputReuseBinding pass logic.
                // hence setIsRealInLogical will indicate to the handleLogicalOps pass in case the producer of the
                // tpc node input is a logical operation that a memcpy is required.
                // But for Eager we apply the logical pass as we go and handle user nodes based on execution scheduling
                // order, such that the logical operation handling for the tpc node input could have already taken place
                // by that point, making the setIsRealInLogical useless. For that reason we also verify here if the tpc
                // node input tensor is an aliased tensor, meaning the logical operation handling has already taken
                // place and in that case add the memcpy in here. While this is sufficient for the input reuse use case
                // it is not a bullet proof solution as each additional usage of setIsRealInLogical will require a
                // similar corresponding handling. A different approach is to postpone the logical nodes handling
                // to the end, but that would require avoiding the orderLast optimization where we sort each user node
                // extracted nodes separatly.
                if (m_eagerGraph.getNumberOfTensorConsumers(inputTensor) > 1 ||
                    m_eagerGraph.isUserManagedDram(inputTensor) || inputTensor->isStaticParam() ||
                    inputTensor->isAliasedTensor())
                {
                    // add memcopy between input and node
                    auto [memcpyNode, copyTensor] = createMemcpyNode(inputTensor, false);
                    tpcNode->replaceTensor(inputTensor, copyTensor);
                    if (!processNewNode(memcpyNode, false)) return AddNodeResult::FAIL;
                    inputTensor = memcpyNode->getOutput(0);
                }
                else
                {
                    inputTensor->setIsRealInLogical(true);
                }
                if (output->isAliasedTensor() || m_eagerGraph.isUserManagedDram(output))
                {
                    //  add memcopy between node and output
                    auto [memcpyNode, copyTensor] = createMemcpyNode(output, true);
                    tpcNode->replaceTensor(output, copyTensor);
                    if (!processNewNode(memcpyNode, false)) return AddNodeResult::FAIL;
                    output = memcpyNode->getInput(0);
                }
                LOG_INFO(GC,
                         "In {} apply inplace reuse for node = {}, set tensor {} as aliased by tensor {}",
                         HLLOG_FUNC,
                         node->getNodeName(),
                         inputTensor->getName(),
                         output->getName());

                output->setAsAliasSubTensor(inputTensor);
            }
        }
    }

    // Eager recipe generator checks
    {
        const auto&    inputs          = node->getInputs();
        const auto&    outputs         = node->getOutputs();
        const unsigned tensorNr        = inputs.size() + outputs.size();
        const unsigned maxTpcTensorsNr = m_eagerGraph.getGraphTraits()->getHalReader()->getBaseRegistersCacheSize() - 1;
        if ((tensorNr == 0) || (tensorNr > maxTpcTensorsNr))
        {
            // TODO: support TPC nodes with no tensors or > 15
            return AddNodeResult::FAIL;
        }
    }

    return AddNodeResult::SUCCESS_ADD_REQUIRED;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processNewComplexGuidNode(const EagerNode& node)
{
    tpc_lib_api::GlueCodeReturn returnCode = m_complexGuidExtractor.calcExtract(&m_eagerGraph, node);
    m_complexGuidExtractor.clearProtocolGraphData();
    if (returnCode == tpc_lib_api::GLUE_CGUID_GRAPH_UNCHANGED) return AddNodeResult::SUCCESS_ADD_REQUIRED;
    const NodeVector& newNodes = m_complexGuidExtractor.extract();
    if (unlikely(newNodes.empty() || returnCode != tpc_lib_api::GLUE_SUCCESS)) return AddNodeResult::FAIL;
    for (EagerNode n : newNodes)
    {
        if (!addInternalNode(n)) return AddNodeResult::FAIL;
    }
    return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
}

bool NodeDisplacement::editConsumersForZST(const EagerNode& node, const TensorPtr& subTensor)
{
    const TensorVector& outputs = node->getOutputs();
    if (subTensor)
    {
        EAGER_ASSERT(outputs.size() == 1, "Expected a single output in case of subTensor");
        const TensorPtr& t = outputs.front();

        if (t->isAliasedTensor())
        {
            if (Tensor::getRealTensor(t) == Tensor::getRealTensor(subTensor)) return true;

            EagerNode identity = NodeFactory::createInternalNode(
                {subTensor},
                {t},
                nullptr,
                NodeFactory::reshapeNodeTypeName,
                fmt::format("connect__{}__with__{}", subTensor->getName(), t->getName()));
            if (identity.get() == nullptr)
            {
                EAGER_ASSERT_0;
                return false;
            }
            return addInternalNode(identity);
        }
        else
        {
            t->setAsAliasSubTensor(subTensor);
        }
    }
    else
    {
        for (const TensorPtr& t : outputs)
        {
            // TODO [SW-127069]: temporarily skipping the num consumers check.
            if (!t /*|| m_eagerGraph.getNumberOfTensorConsumers(t) == 0*/ || t->isShapeTensor() ||
                t->isZeroSizedDataTensor())
            {
                continue;
            }

            // TODO: can we skip these since we know it's an internal node and just hardcode the memset?
            // Note that it's very likely that t->isDynamicShape(), but we use the max dim size anyway in the memset,
            // potentially doing extra work. This is ok since eager shouldn't be using dynamic shapes in normal flow.
            EagerNode memsetNode = getPhysicalMemset(t->getElementType(), {}, {t}, m_eagerGraph.getNextNodeName());
            if (addInternalNode(memsetNode) == false)
            {
                return false;
            }
        }
    }
    return true;
}

NodePtr NodeDisplacement::createPhysicalMemcpy(const TensorPtr& input, const TensorPtr& output, std::string_view name)
{
    if (input->getElementType() == output->getElementType())
    {
        // Eager priority is to create TPC node over DMA node as TPC faster has over all more ports to transfer
        // data than DMA. Moreover, Eager execute nodes serially which makes TPC way more efficient than DMA.
        // In other hand, we consider the DMA case for cases where TPC kernel is missing. The ideal case is to
        // request it to be added to kernel-db.
        bool              isNdim         = input->getDim() > DMAMemcpyNode::MAX_SUPPORTED_DIM;
        const synDataType inputType      = input->getElementType();
        const HalReader&  halReader      = *m_eagerGraph.getGraphTraits()->getHalReader();
        const bool        isDMASupported = halReader.getNumInternalDmaEngines() > 0;
        if (isNdim && TPCMemcpyNode::isSupportedNdimDataType(inputType))
        {
            const std::string guid =
                fmt::format("{}_{}", NodeFactory::memcpyNdNodeTypeName, getDtypeSuffixFromSynDataType(inputType));
            return NodeFactory::createInternalNode({input},
                                                   {output},
                                                   nullptr,
                                                   guid,
                                                   name,
                                                   NodeFactory::memcpyNdNodeTypeName);
        }
        else if (!isNdim && halReader.isTPCMemcpySupportedDataType(inputType))
        {
            return NodeFactory::createInternalNode({input},
                                                   {output},
                                                   nullptr,
                                                   NodeFactory::tpcMemcpyNodeTypeName,
                                                   name);
        }
        else if (isDMASupported)
        {
            return NodeFactory::createInternalNode({input},
                                                   {output},
                                                   nullptr,
                                                   NodeFactory::dmaMemcpyNodeTypeName,
                                                   name);
        }
        else
        {
            EAGER_ASSERT(false, "DMA is unsupported for memcpy");
            return nullptr;
        }
    }
    else
    {
        return CastNodeHandler::createCastNode(input, output, name, m_eagerGraph.getDeviceId());
    }
}

std::pair<EagerNode, TensorPtr> NodeDisplacement::createMemcpyNode(const TensorPtr& orig, bool copyToOrig)
{
    TensorPtr copyTensor = orig->clone(false, false, false, TensorNameClonePolicy::EMPTY_NAME);

    copyTensor->setName(m_eagerGraph.getNextTensorName(), true);
    ;

    EagerNode memcpyNode(createPhysicalMemcpy(copyToOrig ? copyTensor : orig,
                                              copyToOrig ? orig : copyTensor,
                                              m_eagerGraph.getNextNodeName()));

    if (orig->isEnforcedOutput())
    {
        copyTensor->enforceOutput(false);
    }

    /* Need to reset the strides of the cloned tensor, the will be set by the logical op */
    copyTensor->setDenseStrides();

    return std::make_pair(memcpyNode, copyTensor);
}

template<Node::eParamUsage USAGE>
bool NodeDisplacement::insertMemcpyNodes(const EagerNode& node, const LogicalOpNode::IndicesVec& indices)
{
    auto& tensors = NodeTensorsAccessor<USAGE>::getTensors(node);
    for (uint32_t index : indices)
    {
        auto [memcpy, cloneTensor] = createMemcpyNode(tensors[index], USAGE == Node::USAGE_OUTPUT);
        NodeTensorsAccessor<USAGE>::replace(node, index, cloneTensor);
        if (!processNewNode(memcpy, false)) return false;
    }
    return true;
}

bool NodeDisplacement::processLogicalOp(const EagerNode& node, size_t nodeIdx, bool bwdPass)
{
    const auto&               logicalOp = node.get<LogicalOpNode>();
    LogicalOpNode::IndicesVec requireInputMemcpy, requireOutputMemcpy;

    if (!logicalOp->aliasDirectionValid(logicalOp->getAliasDirection(), requireInputMemcpy, requireOutputMemcpy))
    {
        const auto handleInputs = [&] {
            m_nodeCollector.setInjectionIdx(nodeIdx);
            auto res =
                requireInputMemcpy.empty() || likely(insertMemcpyNodes<Node::USAGE_INPUT>(node, requireInputMemcpy));
            m_nodeCollector.resetInjectionIdx();
            return res;
        };
        const auto handleOutputs = [&] {
            m_nodeCollector.setInjectionIdx(nodeIdx + 1);
            auto res =
                requireOutputMemcpy.empty() || likely(insertMemcpyNodes<Node::USAGE_OUTPUT>(node, requireOutputMemcpy));
            m_nodeCollector.resetInjectionIdx();
            return res;
        };

        // Make sure they are added in reverse order if isBwd so that it's enough to std::reverse to get a increasingly
        // sorted array for the injection.
        if (bwdPass)
        {
            if (unlikely(!handleOutputs() || !handleInputs())) return false;
        }
        else
        {
            if (unlikely(!handleInputs() || !handleOutputs())) return false;
        }
    }
    if (!logicalOp->getRunLogicalOperationDone())
    {
        logicalOp->runAndSetLogicalOp();
    }
    logicalOp->setMustBeDenseIfNeeded();
    return true;
}

NodeDisplacement::AddNodeResult NodeDisplacement::processNewBroadcastNode(const EagerNode& node)
{
    const auto& inputs          = node->getInputs();
    const auto& outputs         = node->getOutputs();
    TensorPtr   broadcastInput  = inputs[0];
    TensorPtr   broadcastOutput = outputs[0];
    EAGER_ASSERT(isTPCBroadcastSupported(node), "TPC Broadcast is not supported for node {}", node->getNodeName());
    // broadcast_nd_fwd validates that both input and output tensors
    // are ndim (> 5) so we need to reshape the tensors in case this
    // is not currently the case.
    EagerNode inputReshapeNode;
    if (broadcastOutput->getDim() <= DMAMemcpyNode::MAX_SUPPORTED_DIM)
    {
        TensorPtr reshapeInput = broadcastOutput->clone(false, false);
        reshapeInput->reshape(DMAMemcpyNode::MAX_SUPPORTED_DIM + 1);
        inputReshapeNode = NodeFactory::createInternalNode({reshapeInput},
                                                           {broadcastOutput},
                                                           nullptr,
                                                           NodeFactory::reshapeNodeTypeName,
                                                           m_eagerGraph.getNextNodeName());
        broadcastOutput  = reshapeInput;
    }
    EagerNode outputReshapeNode;
    if (broadcastInput->getDim() <= DMAMemcpyNode::MAX_SUPPORTED_DIM)
    {
        TensorPtr reshapeOutput = broadcastInput->clone(false, false);
        reshapeOutput->reshape(broadcastOutput->getDim());
        outputReshapeNode = NodeFactory::createInternalNode({broadcastInput},
                                                            {reshapeOutput},
                                                            nullptr,
                                                            NodeFactory::reshapeNodeTypeName,
                                                            m_eagerGraph.getNextNodeName());
        broadcastInput    = reshapeOutput;
    }
    auto newFullGuid =
        fmt::format("broadcast_nd_fwd_{}", getDtypeSuffixFromSynDataType(broadcastInput->getElementType()));
    EagerNode broadcastNode = NodeFactory::createInternalNode({broadcastInput},
                                                              {broadcastOutput},
                                                              nullptr,
                                                              newFullGuid,
                                                              node->getNodeName(),
                                                              "broadcast_nd_fwd");
    if (inputReshapeNode.get() != nullptr && !addInternalNode(inputReshapeNode)) return AddNodeResult::FAIL;
    if (!addInternalNode(broadcastNode)) return AddNodeResult::FAIL;
    if (outputReshapeNode.get() != nullptr && !addInternalNode(outputReshapeNode)) return AddNodeResult::FAIL;
    return AddNodeResult::SUCCESS_NO_ADD_REQUIRED;
}

NodePtr NodeDisplacement::getPhysicalMemset(synDataType         elementType,
                                            const TensorVector& inputs,
                                            const TensorVector& outputs,
                                            std::string_view    nodeName)
{
    const HalReader& halReader = *m_eagerGraph.getGraphTraits()->getHalReader();

    const bool shouldUseTPC = halReader.isTPCMemsetSupportedDataType(elementType);
    if (shouldUseTPC)
    {
        return NodeFactory::createInternalNode(inputs, outputs, nullptr, NodeFactory::tpcMemsetNodeTypeName, nodeName);
    }

    const bool isDMASupported = halReader.getNumInternalDmaEngines() > 0;
    if (isDMASupported)
    {
        return NodeFactory::createInternalNode(inputs, outputs, nullptr, NodeFactory::dmaMemsetNodeTypeName, nodeName);
    }

    EAGER_ASSERT(false, "DMA is unsupported for memset");
    return nullptr;
}

static void trySwapBwd(synDeviceType deviceType, EagerNodes& nodes, size_t idx)
{
    auto& logicalNode = *nodes[idx].get<LogicalOpNode>();

    // Similar to wantBackwardDirectionShouldCallSwapDirection
    if (LogicalOpsHandler::isBackwardNode(logicalNode) ||  //
        !logicalNode.canSwapAliasDirection() ||            //
        LogicalOpsHandler::isRealInLogical(logicalNode.getRealTensor()))
    {
        return;
    }

    // The following conditions are taken from isSwapAliasDirectionProfitable without the 2nd order chain prop

    Tensor* ouputTensorAliasingInput = nullptr;
    for (const auto& t : logicalNode.getOutputs())
    {
        EAGER_ASSERT_PTR(t);
        if (!t->isShapeTensor())
        {
            EAGER_ASSERT(!ouputTensorAliasingInput, "trying to swap directions with multiple inputs");
            ouputTensorAliasingInput = t.get();
        }
    }
    EAGER_ASSERT(ouputTensorAliasingInput, "trying to swap directions without an input");

    auto swapLogicalAfterReal = [&] {
        // If producer is not logical operation, and the logical node is the only consumer,
        // then swap direction is preferred to avoid unnecessary internal Memcpy nodes.

        const Tensor* realTensor = logicalNode.getInput(0).get();

        std::optional<size_t> producerIdx = nodes.getInputProducerIdx(realTensor, idx);
        if (!producerIdx) return false;

        if (const Node& producer = nodes[*producerIdx];
            producer.isLogicalOperation() || !producer.canHandleStridedOutput(deviceType))
        {
            return false;
        }

        return nodes.hasSingleConsumer(realTensor, *producerIdx + 1);
    };

    if (ouputTensorAliasingInput->isAliasedTensor() ||    //
        ouputTensorAliasingInput->isRealInLogical() ||    //
        ouputTensorAliasingInput->isUserManagedDram() ||  //
        swapLogicalAfterReal())
    {
        logicalNode.swapAliasDirection();
    }
}

bool NodeDisplacement::processLogicalNodes(ExecScheduler& execSequencer)
{
    EAGER_ASSERT(m_nodeCollector.hasLogicalNodes(), "Wrong flow? Missed opt...");
    m_disableSuggestedManipulation = true;
    // When the collector reports having seen logical nodes, the logical pass
    // which is a partial implementation of the Graph-Mode one is executed:
    //
    // Logical nodes can be either,
    // - Strictly FWD (Where the real tensor is the sole input eg. Split)
    // - Strictly BWD (Where the real tensor is the sole output eg. Concat)
    // - Or agnostic eg. Reshape. Agnostic nodes are FWD by default.
    //
    // The logical node handling consists of
    // - A backward pass (Reverse topological order), where for each BWD node
    //   (Either strictly BWD or an agnostic node which we can and may swap
    //   based on a local heuristic) we store the required memcpies and the
    //   indices before which they ought to be added.
    // - The new nodes are reversed and injected into the topologically sorted
    //   node array, in a single pass.
    // - A forward pass (In topological order) where for each FWD node the
    //   injected memcpies and the indices before which they are to be injected
    //   are collected.
    // - The new nodes are injected into the topologically sorted node array,
    //   in a single pass.
    // - The serial dependencies are trivially regenerated (Outside of this fn)
    //
    // Diffs vs Graph-Mode to be addressed as needed:
    //
    // - The swap to BWD heuristics are the same.
    // - Graph-Mode has a prioritizing sort of the nodes whereas we use the
    //   existing topological order in both the backward and forward passes.
    // - Graph-Mode has a second order optimization handling forward single
    //   consumer chains.
    // - Other limitations such as memory overlap, strided access etc. are not
    //   handled in this pass in Eager.
    // - Graph-Mode has a pre-pass wrapping TPC with tensor reuse for the
    //   logical pass and restoration for them which isn't addressed here.

    const auto deviceType = m_eagerGraph.getDeviceType();
    for (const bool bwdPass : {true, false})
    {
        const size_t nodeNum = m_nodes.size();  // note that this may differ between for iters if nodes are added
        for (size_t i = 0; i < nodeNum; ++i)
        {
            const auto nodeIdx = bwdPass ? (nodeNum - 1) - i : i;

            auto& node = m_nodes[nodeIdx];
            if (!node->isLogicalOperation()) continue;
            auto& logicalNode = *node.get<LogicalOpNode>();

            if (bwdPass)
            {
                EAGER_ASSERT(!logicalNode.getRunLogicalOperationDone(),
                             "First time handling logical node in bwd pass somehow already handled!");

                // All nodes begin as FWD unless they cannot be (For ex. concat), try first to set them to bwd:
                trySwapBwd(deviceType, m_nodes, nodeIdx);

                if (!LogicalOpsHandler::isBackwardNode(logicalNode)) continue;
            }
            else
            {
                if (logicalNode.getRunLogicalOperationDone()) continue;
                EAGER_ASSERT(LogicalOpsHandler::isForwardNode(logicalNode), "Unhandled backwards node!");
            }

            if (unlikely(!processLogicalOp(node, nodeIdx, bwdPass))) return false;
        }
        m_nodeCollector.injectNodes(execSequencer, bwdPass);
    }
    return true;
}

bool NodeDisplacement::handleMmeConcurrency(const EagerNode& node)
{
    const EagerMmeBrainBase& mmeBrain = m_eagerGraph.getEagerMmeBrain();
    MmeNode&                 mmeNode  = *node.getSafePtr<MmeNode>();
    auto&                    strategy = mmeNode.getNodeAnnotation().mmeMetaData.mmeStrategy;

    // MME concurrency is intentially disabled
    if (unlikely(!GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value() || !GCFG_ENABLE_EAGER_MME_CONCURRENCY.value()))
        return true;

    const HalReader& hal = *m_eagerGraph.getHALReader();
    if (!mmeBrain.isNodeCandidateForMmeConcurrency(mmeNode, hal)) return true;

    // A simple heuristic to select either CD or batch concurrency
    bool               cdcEn       = false;
    const TensorShape& outputShape = mmeNode.getOutput(0)->getShape();
    const unsigned     maxDim      = getMaxBatchDim(outputShape.getSizes(), 2, outputShape.getDim() - 1);
    // Set only one optimization to undefined so that it will be chosen by the mme brain
    const unsigned totalMmeCores = hal.getNumMmeEngines() * hal.getNumMmeCoresPerEngine();
    // For batch concurrency check possibility to split on batch dim and distribute the computation upon cores
    if (maxDim == 1 || maxDim < totalMmeCores)
    {
        // A simple heuristic check to filter out irrelevant CDC cases
        const uint64_t outputSizeOfOneMme = uint64_t(outputShape.getSize(0)) * outputShape.getSize(1);
        cdcEn                             = (outputSizeOfOneMme <= mmeBrain.getTotalMmeSize());
    }

    // Allow CDC or enable batch concurrency as its calculations are lite
    if (!cdcEn)
    {
        strategy.batchConcurrencyEn = MmeCommon::TurnedOn;
        strategy.cdConcurrencyEn    = MmeCommon::TurnedOff;
        return true;
    }

    // The rest of the flow deals with CDC only (no batch concurrency)
    strategy.batchConcurrencyEn = MmeCommon::TurnedOff;
    strategy.cdConcurrencyEn    = MmeCommon::Undefined;

    // Need to prepare the brain since MME code needs it
    mmeNode.initMmeBrainIfc(m_eagerGraph.getDeviceType());
    MmeBrainIfc& brainIfc = *mmeNode.getMmeBrainIfc();

    // TODO[SW-169983]: Remove this limitation
    const bool sbReuseStrat = [&] {
        switch (m_eagerGraph.getChipType())
        {  // clang-format off
            default: EAGER_ASSERT_0; // and fallthrough
            case ChipType::GAUDI2: return GCFG_ENABLE_EAGER_SB_REUSE_G2.value() == 1; // Only if force on
            case ChipType::GAUDI3: return GCFG_ENABLE_EAGER_SB_REUSE_G3.value() == 1; // Only if force on
        }  // clang-format on
    }();

    // TODO[SW-170003]: Avoid GCFG temp changes

    // Save some graph mode related env var to restore them later
    const unsigned defaultPipelineDepth       = GCFG_DEFAULT_PIPELINE_DEPTH.value();
    const bool     sbReuse                    = GCFG_SB_REUSE.value();
    const bool     enableMmeEnableMmeAlignOpt = GCFG_ENABLE_MME_ALIGN_OPT.value();
    // Overwrite current vals to match Eager's config
    GCFG_DEFAULT_PIPELINE_DEPTH.setValue(1);
    GCFG_SB_REUSE.setValue(sbReuseStrat);
    GCFG_ENABLE_MME_ALIGN_OPT.setValue(false);

    // Choose the concurrency
    brainIfc.setRecommendedConcurrency();

    // Restore env vars
    GCFG_DEFAULT_PIPELINE_DEPTH.setValue(defaultPipelineDepth);
    GCFG_SB_REUSE.setValue(sbReuse);
    GCFG_ENABLE_MME_ALIGN_OPT.setValue(enableMmeEnableMmeAlignOpt);

    if (node->getNodeAnnotation().mmeMetaData.mmeStrategy.cdConcurrencyEn == MmeCommon::TurnedOn)
    {
        // If the output data type is not float, add float32 tensor as an output of the node + cast to the
        // original data type. This maintains the accuracy of calculations.
        TensorPtr mmeOutput     = mmeNode.getOutput(0);
        TensorPtr mmeOutputFp32 = mmeOutput->clone();
        bool      isCasted      = false;
        if (mmeOutput->getElementType() != syn_type_float && mmeOutput->getElementType() != syn_type_hb_float)
        {
            // Create the float32 output tensor
            mmeOutputFp32->setElementType(syn_type_float);
            isCasted = true;
        }
        mmeOutputFp32->setName(fmt::format("{}_Fp32", mmeOutput->getName()));
        mmeOutputFp32->getTensorAnnotation().tensorReductionInfo.isReductionEnabled = true;
        // Modify the original MME node
        mmeNode.removeOutput(mmeOutput);
        mmeNode.addOutput(mmeOutputFp32);

        // Create the memset node & tensor
        TensorPtr memsetTensor = mmeOutputFp32->clone();
        memsetTensor->setName(fmt::format("{}_zeros", mmeOutputFp32->getName()));
        // Create the memset node
        EagerNode memsetNode =
            getPhysicalMemset(syn_type_float, {}, {memsetTensor}, fmt::format("{}_memset", mmeOutputFp32->getName()));
        if (!addInternalNode(memsetNode)) return false;

        // Create the Reduction node & tensor
        TensorPtr afterReductionTensor;
        if (isCasted)
        {
            afterReductionTensor = mmeOutputFp32->clone();
            afterReductionTensor->setName(fmt::format("{}_after_reduction", mmeOutputFp32->getName()));
        }
        else
        {
            afterReductionTensor = mmeOutput;
        }
        EagerNode reductionNode = NodeFactory::createNode({memsetTensor, mmeOutputFp32},
                                                          {afterReductionTensor},
                                                          nullptr,
                                                          0,
                                                          NodeFactory::reductionNodeTypeName,
                                                          fmt::format("{}_reduction", mmeOutputFp32->getName()));
        if (!addInternalNode(reductionNode)) return false;

        // Create the cast node
        if (isCasted)
        {
            EagerNode castNode = CastNodeHandler::createCastNode(afterReductionTensor,
                                                                 mmeOutput,
                                                                 fmt::format("{}_cast", mmeOutput->getName()),
                                                                 m_eagerGraph.getDeviceId());
            if (!addInternalNode(castNode)) return false;
        }
    }

    return true;
}

bool NodeDisplacement::handleHugeTensor(const EagerNode& node, HugeTensorHandler& hugeTensorHandler)
{
    NodeVector extractedNodes = hugeTensorHandler.extractNodeWithHugeTensors(node);
    EAGER_ASSERT(!extractedNodes.empty(), "Invalid huge tensor processing result");

#ifndef NDEBUG
    NodesContainer::printNodesWithTensorDetails(extractedNodes,
                                                fmt::format("extracted nodes of \"{}\"", node->getNodeName()));
#endif  // NDEBUG

    for (EagerNode extractedNode : extractedNodes)
    {
        if (extractedNode->isLogicalOperation())
        {
            if (unlikely(extractedNode->getNodeType() == Node::TYPE_INTERNAL_CONCAT))
            {
                // Strides of concat must reflect the real output tensor in order to inject a correct memcpy
                extractedNode.get<LogicalOpNode>()->runAndSetLogicalOp();
            }
            if (unlikely(extractedNode->getNodeType() == Node::TYPE_INTERNAL_REDUCTION))
            {
                // Propagate reduction information to the RMW nodes. Those are the producers of reduction ops' inputs
                // except first one.
                bool isFirstNode = true;
                for (const TensorPtr& reductionIn : extractedNode->getInputs())
                {
                    if (unlikely(isFirstNode))
                    {
                        isFirstNode = false;
                        continue;
                    }
                    TensorAnnotation& ann                      = reductionIn->getTensorAnnotation();
                    ann.tensorReductionInfo.reductionOperation = REDUCTION_ADD;
                    ann.tensorReductionInfo.isReductionEnabled = true;
                }
            }
        }
        if (!addInternalNode(extractedNode)) return false;
    }
    return true;
}

bool NodeDisplacement::handle64bitPrecisionNode(EagerNode& node)
{
    if (node->getOutputs().size() == 0 || !node->getOutput(0)->is64BitElementSize()) return true;
    std::string_view guidWithoutDtype = extractGUIDFromFullGUID(node->getGUID());
    // memcpy\broadcast is only concerned with data type width so no difference between memcpy_nd_i32 or memcpy_nd_u32.
    if (guidWithoutDtype == "memcpy_nd")
    {
        return handle64bitPrecisionNode(node, "memcpy_nd_u32");
    }
    else if (guidWithoutDtype == "broadcast_nd_fwd")
    {
        return handle64bitPrecisionBroadcastNode(node, "broadcast_nd_fwd_u32");
    }
    return true;
}

bool NodeDisplacement::handle64bitPrecisionNode(EagerNode& node, std::string_view newGuid)
{
    const TensorPtr& input = node->getInput(0);
    node->setGUID(newGuid);
    auto [reinterpretInPtr, newInput] = reinterpretTensor(input, true, syn_type_uint32);
    node->replaceTensor(input, newInput);
    EagerNode reinterpretIn(reinterpretInPtr);
    if (!addInternalNode(reinterpretIn)) return false;
    const TensorPtr& output             = node->getOutput(0);
    auto [reinterpretOutPtr, newOutput] = reinterpretTensor(output, false, syn_type_uint32);
    EagerNode reinterpretOut(reinterpretOutPtr);
    node->replaceTensor(output, newOutput);
    if (!addInternalNode(reinterpretOut)) return false;
    return true;
}

bool NodeDisplacement::handle64bitPrecisionBroadcastNode(EagerNode& node, std::string_view newGuid)
{
    const TensorPtr& input  = node->getInput(0);
    const TensorPtr& output = node->getOutput(0);
    // broadcast on FCD requires adding a trivial dimension FCD
    if (input->getSizeInElements(0) != output->getSizeInElements(0))
    {
        TensorPtr reshapeOutput = getExpandedTensor(*input);
        if (unlikely(!reshapeOutput)) return false;
        EagerNode inputReshapeNode = NodeFactory::createInternalNode({input},
                                                                     {reshapeOutput},
                                                                     nullptr,
                                                                     NodeFactory::reshapeNodeTypeName,
                                                                     m_eagerGraph.getNextNodeName());
        node->replaceTensor(input, reshapeOutput);
        if (unlikely(!addInternalNode(inputReshapeNode))) return false;
        TensorPtr reshapeInput = getExpandedTensor(*output);
        if (unlikely(!reshapeInput)) return false;
        EagerNode outputReshapeNode = NodeFactory::createInternalNode({reshapeInput},
                                                                      {output},
                                                                      nullptr,
                                                                      NodeFactory::reshapeNodeTypeName,
                                                                      m_eagerGraph.getNextNodeName());
        node->replaceTensor(output, reshapeInput);
        if (unlikely(!addInternalNode(outputReshapeNode))) return false;
    }
    return handle64bitPrecisionNode(node, newGuid);
}

}  // namespace eager_mode
