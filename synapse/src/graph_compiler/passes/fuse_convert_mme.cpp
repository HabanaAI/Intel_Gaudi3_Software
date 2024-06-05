#include "passes.h"
#include "node_utils.h"
#include "quantization_utils.h"

constexpr unsigned FP8_CONV_BIAS_IDX  = 2;
constexpr unsigned FP8_GEMM_BIAS1_IDX = 4;
constexpr unsigned FP8_GEMM_BIAS2_IDX = 5;
constexpr unsigned FP8_GEMM_INV_SCALE_OUT_IDX = 6;
constexpr unsigned FP8_CONV_INV_SCALE_OUT_IDX = 5;

static unsigned getFP8MmeOutputScaleIndex(NodePtr mmeNode)
{
    if (isFp8GemmGuid(mmeNode))
    {
        return FP8_GEMM_INV_SCALE_OUT_IDX;
    }
    else if (isFp8ConvGuid(mmeNode))
    {
        return FP8_CONV_INV_SCALE_OUT_IDX;
    }
    else
    {
        HB_ASSERT(false, "Can't get output scale index for MME fp8 cguid {}", mmeNode->getGUID());
        return 0;
    }
}

static std::string getNewFP8MmeGuid(NodePtr mmeNode, synDataType toType)
{
    if (isFp8GemmGuid(mmeNode))
    {
        return fmt::format("fp8_gemm_{}", getDtypeSuffixFromSynDataType(toType));
    }
    else if (isFp8ConvGuid(mmeNode))
    {
        return fmt::format("conv2d_fp8_{}", getDtypeSuffixFromSynDataType(toType));
    }
    else
    {
        HB_ASSERT(false, "Can't get output scale index for MME fp8 cguid {}", mmeNode->getGUID());
        return 0;
    }
}

static bool canFuseConvertToMme(HabanaGraph& g, synDataType toType, const NodePtr& mmeNode, const NodePtr& convertNode)
{
    LOG_TRACE(QUANT, "Checking if can fuse convert node {} to mme node {}", convertNode->getNodeName(),
              mmeNode->getNodeName());
    // validate that the MME engine support the cast output data type
    if (!g.getHALReader()->isSupportedMmeDataType(toType)) return false;

    // validate that the tensor between the cast and the MME node is not managed by user (i.e. persistent tensor)
    if (g.isUserManagedDram(convertNode->getInput(0))) return false;

    // remove only if the blocking nodes of cast is subset of the mme blocking nodes
    const auto& convertBlocking = g.getBlockingNodes(convertNode);
    const auto& mmeBlocking     = g.getBlockingNodes(mmeNode);
    if (!std::includes(mmeBlocking.begin(), mmeBlocking.end(), convertBlocking.begin(), convertBlocking.end()))
        return false;

    // validate convert to fp8 can use exp bias instead of scale
    if (!QuantizationUtils::isConvertExpBiasHwAligned(convertNode)) return false;
    // validate fp8 mme can use exp bias instead of scale
    if (!QuantizationUtils::isFp8MmeExpBiasHwAligned(mmeNode)) return false;
    // validate fp8 mme don't have bias tensors inputs
    if ((isFp8GemmGuid(mmeNode) && (mmeNode->getInput(FP8_GEMM_BIAS1_IDX) || mmeNode->getInput(FP8_GEMM_BIAS2_IDX))) ||
        (isFp8ConvGuid(mmeNode) && mmeNode->getInput(FP8_CONV_BIAS_IDX)))
    {
        LOG_TRACE(QUANT, "Can't fuse mme cguid {} since it has bias tensor", mmeNode->getNodeName());
        return false;
    }
    LOG_TRACE(QUANT, "Can fuse convert node {} to mme node {}", convertNode->getNodeName(), mmeNode->getNodeName());
    return true;
}

static void fuseConvertIntoMmeNode(HabanaGraph& g, const NodePtr& convertNode, synDataType toType)
{
    const TensorPtr& output  = convertNode->getOutput(0);
    const TensorPtr& input   = convertNode->getInput(0);
    const NodePtr&   mmeNode = g.getTensorProducer(input);

    // Remove the node to disable the relationship with the graph output
    LOG_DEBUG(QUANT, "fuse Convert Into Mme Node '{}'", convertNode->getNodeName());
    GraphEditor::removeNode(g, convertNode);

    // Switch the mme node output
    GraphEditor::replaceTensor(g, mmeNode, input, output);

    // Need to set new Guid for fp8 mme OP
    std::string guid = getNewFP8MmeGuid(mmeNode, toType);
    LOG_TRACE(QUANT, "New guid after fusion - {}", guid);
    mmeNode->setGUID(guid);

    unsigned outputScaleIndex = getFP8MmeOutputScaleIndex(mmeNode);
    // Pass scale tensor to fp8 mme
    GraphEditor::replaceInput(g, mmeNode, outputScaleIndex, convertNode->getInput(CONVERT_INV_SCALE_IDX));
}

bool fuseConvertMme(HabanaGraph& g)
{
    if (!g.getInferenceMode() || !GCFG_FUSE_CONVERT_TO_MME.value()) return true;

    unsigned int removedNodes = 0;
    // We make a copy of the sortedNodes since during the loop we alter
    // the graph and thus invalidate the graph's sorted nodes cache.
    NodeVector sortedNodes = g.getExeSortedNodes();
    for (const auto& node : sortedNodes)
    {
        if (isFp8MmeCguid(node))
        {
            const pTensor output    = node->getOutput(0);
            const auto    consumers = g.getTensorConsumers(output);
            if (consumers.size() == 1)
            {
                auto convertNode = consumers.front();
                if (isConvertToFp8Node(convertNode))
                {
                    synDataType castToType = convertNode->getOutput(0)->getElementType();
                    if (canFuseConvertToMme(g, castToType, node, convertNode))
                    {
                        // Maintain origin_nodes tracking for debugging purposes
                        node->addOriginNodes(convertNode->getOriginNodes());

                        fuseConvertIntoMmeNode(g, convertNode, castToType);
                        removedNodes++;
                    }
                }
            }
        }
    }
    LOG_DEBUG(QUANT, "fused {} convert nodes into mme", removedNodes);
    return true;
}