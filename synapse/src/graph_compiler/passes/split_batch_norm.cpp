#include "batch_norm_pattern_matcher.h"
#include "bn_utils.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "log_manager.h"
#include "node_factory.h"
#include "node_utils.h"
#include "perf_lib_layer_params.h"
#include "tf_batch_norm_node.h"
#include "tpc_kernel_names.h"

using namespace std;

bool splitBatchNormFwd(HabanaGraph& g, TPCNodePtr fullBatchNorm, synDataType dtype)
{
    if (dtype != syn_type_float && dtype != syn_type_bf16)
    {
        LOG_ERR(GC, "splitBatchNorm: Unsupported dtype {}", dtype);
        return false;
    }

    BNUtils::Bn1Bn2FwdInputs inputs = {fullBatchNorm->getInput(0), fullBatchNorm->getInput(1), fullBatchNorm->getInput(2),
                                       fullBatchNorm->getInput(3), fullBatchNorm->getInput(4), nullptr};

    BNUtils::Bn1Bn2FwdOutputs outputs = {fullBatchNorm->getOutput(0), fullBatchNorm->getOutput(1),
                                         fullBatchNorm->getOutput(2),
                                         fullBatchNorm->getOutput(3), fullBatchNorm->getOutput(4)};

    //Save the original node's parameters
    ns_BatchNormKernel::Params* fullBnParams = static_cast<ns_BatchNormKernel::Params*>(fullBatchNorm->getParams());
    if (fullBnParams == nullptr)
    {
        LOG_ERR(GC, "splitBatchNorm- node {} parameters is NULL", fullBatchNorm->getNodeName());
        return false;
    }

    bool isTraining = true;
    if (fullBatchNorm->getParamsSize() == sizeof(ns_BatchNormKernel::ParamsV2))
    {
        isTraining = static_cast<ns_BatchNormKernel::ParamsV2*>(fullBnParams)->isTraining;
    }
    LOG_TRACE(GC, "Using isTraining {} for node {}", isTraining, fullBatchNorm->getNodeName());

    NodeList bn1bn2NodeList;
    bool     retVal;

    auto packingFactor = BNUtils::shouldOptimizeLowFCD(inputs, outputs, isTraining, fullBatchNorm->getNodeName());
    if (packingFactor.has_value())
    {
        BNUtils::optimizeLowFCD(packingFactor.value(), inputs, outputs, bn1bn2NodeList, fullBatchNorm->getNodeName());
    }

    retVal = BNUtils::createBn1Bn2NodesFwd(inputs,
                                           outputs,
                                           fullBnParams->momentum,
                                           fullBnParams->epsilon,
                                           fullBatchNorm->getNodeName(),
                                           dtype,
                                           isTraining,
                                           bn1bn2NodeList,
                                           /*locateInSram*/ true,
                                           /*fcd optimization */ packingFactor);

    if (retVal != true)
    {
        return false;
    }

    NodeList oldNodes = {fullBatchNorm};
    return (GraphEditor::replaceNodes(g, oldNodes, bn1bn2NodeList) == REPLACE_NODE_SUCCESS);
}

bool splitBatchNormBwd(HabanaGraph& g, TPCNodePtr fullBatchNorm, synDataType dtype)
{
    if (dtype != syn_type_float && dtype != syn_type_bf16)
    {
        LOG_ERR(GC, "splitBatchNorm: Unsupported dtype {}", dtype);
        return false;
    }

    bool isTraining = true;
    if (fullBatchNorm->getParamsSize() == sizeof(ns_BatchNormKernel::ParamsV2))
    {
        LOG_TRACE(GC, "Using ns_BatchNormKernel::ParamsV2 for node {}", fullBatchNorm->getNodeName());
        isTraining = static_cast<ns_BatchNormKernel::ParamsV2*>(fullBatchNorm->getParams())->isTraining;
    }

    BNUtils::Bn1Bn2BwdInputs inputs = {fullBatchNorm->getInput(0), fullBatchNorm->getInput(1), fullBatchNorm->getInput(2),
                                       fullBatchNorm->getInput(3), fullBatchNorm->getInput(4)};

    BNUtils::Bn1Bn2BwdOutputs outputs = {fullBatchNorm->getOutput(0), fullBatchNorm->getOutput(2), fullBatchNorm->getOutput(1)};

    NodeList bnNodeList;
    bool     retVal = BNUtils::createBn1Bn2NodesBwd(inputs,
                                                outputs,
                                                fullBatchNorm->getNodeName(),
                                                dtype,
                                                isTraining,
                                                bnNodeList,
                                                /*locateInSram*/ true);
    if (retVal != true)
    {
        return false;
    }

    NodeList oldNodes = {fullBatchNorm};
    return (GraphEditor::replaceNodes(g, oldNodes, bnNodeList) == REPLACE_NODE_SUCCESS);
}

void splitTpcBn(HabanaGraph& g, TPCNodePtr tpcNode)
{
    bool result = true;
    if (tpcNode->isGuidPrefix("batch_norm_fwd_bf16"))
    {
        result = splitBatchNormFwd(g, tpcNode, syn_type_bf16);
    }
    if (tpcNode->isGuidPrefix("batch_norm_bwd_bf16"))
    {
        result = splitBatchNormBwd(g, tpcNode, syn_type_bf16);
    }
    if (tpcNode->isGuidPrefix("batch_norm_fwd_f32"))
    {
        result = splitBatchNormFwd(g, tpcNode, syn_type_float);
    }
    if (tpcNode->isGuidPrefix("batch_norm_bwd_f32"))
    {
        result = splitBatchNormBwd(g, tpcNode, syn_type_float);
    }
    if (!result)
    {
        LOG_WARN(GC, "splitBatchNorm- failed to split node {}", tpcNode->getNodeName());
    }
}

bool splitTfFusedBnBwd(HabanaGraph& g, NodePtr node)
{
    synDataType dtype = node->getInput(0)->getElementType();
    if (dtype != syn_type_float && dtype != syn_type_bf16)
    {
        LOG_ERR(GC,"splitTfFusedBnBwd: Unsupported dtype {}", dtype);
        return false;
    }

    BNUtils::Bn1Bn2BwdInputs inputs = {node->getInput(1), node->getInput(0), node->getInput(3),
                                       node->getInput(4), node->getInput(2)};

    BNUtils::Bn1Bn2BwdOutputs outputs = {node->getOutput(0), node->getOutput(1),
                                         node->getOutput(2), nullptr};

    NodeList bnNodeList;
    bool     retVal = BNUtils::createBn1Bn2NodesBwd(inputs,
                                                outputs,
                                                node->getNodeName(),
                                                dtype,
                                                true,
                                                bnNodeList,
                                                /*locateInSram*/ true);
    if (retVal != true)
    {
        return false;
    }

    NodeList oldNodes = {node};
    return (GraphEditor::replaceNodes(g, oldNodes, bnNodeList) == REPLACE_NODE_SUCCESS);
}

bool splitBatchNorm(HabanaGraph& g)
{
    InstanceNormToBatchNormPatternMatcher bnPatternMatcher(g);
    NodeSet nodes = g.getNodes();
    for (auto node : nodes)
    {
        if (g.runsOnTPC(node))
        {
            TPCNodePtr tpcNode = std::static_pointer_cast<TPCNode>(node);
            if (GCFG_SKIP_BN_SPLIT.value())
            {
                LOG_TRACE(GC, "Skipping splitBatchNorm for node {}. (SKIP_BN_SPLIT=true)", node->getNodeName());
                continue;
            }
            const auto& [valid, concurrencyLevel] = bnPatternMatcher.matchPattern(tpcNode);
            if (GCFG_SKIP_BN_SPLIT_FOR_IN_REDUCED_TO_BN.value() && valid &&
                (concurrencyLevel >= g.getHALReader()->getNumTpcEngines()))
            {
                // Temp solution until Habana-norm will be fully integrated.
                // In order to create stage1 and stage2 nodes, need to create concatenated tensors of mean+istd and
                // beta+gamma. In addition need to split the output Grad-BetaGamma to Grad-Beta + Grad-Gamma. These
                // internal splits/concats may create additional DMA copies in case we have split/concat on the original
                // BN inputs/outputs. When we have such pattern and it can be parallel on all availabe TPC engines -
                // skip the split (each TPC engine will run a single BN node).
                LOG_TRACE(GC,
                          "Skipping splitBatchNorm for node {} to optimize performance (concurrency level = {})",
                          node->getNodeName(),
                          concurrencyLevel);
                continue;
            }
            splitTpcBn(g, tpcNode);

        }
        else if (node->getNodeType() == Node::TYPE_TF_FUSED_BATCH_NORM_GRAD)
        {
            if (!splitTfFusedBnBwd(g, node))
            {
                return false;
            }
        }
    }
    return true;
}
