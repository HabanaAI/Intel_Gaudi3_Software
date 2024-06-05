#include "habana_graph.h"
#include "habana_pass.h"

#include "tpc_node.h"
#include "tpc_kernel_names.h"
#include <algorithm>
#include "graph_editor.h"
#include "types.h"

static const unsigned BATCH_NORM_OUTPUT_GRAD_AND_FEATURE_MAP_INDEX = 0;
static const unsigned BATCHNORM_OUTPUT_COPY_INDEX                  = 1;

enum eMemcpyLocation
{
    eMemcpyBeforeInput,
    eMemcpyAfterOutput
};

pNode getFirstDRAMMemCpyNode(const NodeList& list)
{
    for (auto node : list)
    {
        if (node->getNodeType() == Node::TYPE_MEMCOPY)
        {
            if (node->getInputs().size() != 1 || node->getOutputs().size() != 1)
            {
                LOG_ERR(GC,
                        "fuseMemcopyInputToBnOutput incorrect number of parameters for node {}",
                        node->getNodeName());
                continue;
            }
            if (node->getOutput(0)->inDram())
            {
                return node;
            }
        }
    }
    return nullptr;
}

void fuseMemcpyOutputToBnOutput(const pNode& batchnormNode, HabanaGraph& g, eMemcpyLocation memCopyLocation)
{
    NodeList consumers;
    if (memCopyLocation == eMemcpyBeforeInput)
    {
        consumers = g.getTensorConsumers(batchnormNode->getInput(TENSOR_IFM));
    }
    else if (memCopyLocation == eMemcpyAfterOutput)
    {
        consumers = g.getTensorConsumers(batchnormNode->getOutput(BATCH_NORM_OUTPUT_GRAD_AND_FEATURE_MAP_INDEX));
    }
    pNode memcpyNode = getFirstDRAMMemCpyNode(consumers);
    if (memcpyNode == nullptr)
    {
        return;
    }
    pTensor memcpyOutput = memcpyNode->getOutput(0);
    pTensor batchNormIFM = batchnormNode->getInput(TENSOR_IFM);
    if (memcpyOutput->getTotalElements() != batchNormIFM->getTotalElements())
    {
        LOG_ERR(GC,
                "fuseMemcopyInputToBnOutput mismatch in number of elements for node {}",
                batchnormNode->getNodeName());
        return;
    }
    if (memcpyOutput->getElementType() != batchNormIFM->getElementType())
    {
        LOG_ERR(GC, "fuseMemcopyInputToBnOutput type mismatch in node {}", batchnormNode->getNodeName());
        return;
    }
    pNode newBnNode = batchnormNode->clone();
    // TODO - SW-23441 - get rid of the special offset.
    //    auto outputOffset = batchnormNode->getGUID().find("stage1_dynamic_fwd") != std::string::npos ? 1 : 0;
    newBnNode->emplaceOutput(BATCHNORM_OUTPUT_COPY_INDEX /*+ outputOffset*/, memcpyOutput);
    // In case the original BN was instantiated already, need to re-instantiate with the new input.
    static_cast<TPCNode*>(newBnNode.get())->resetInstantiated();
    auto status = GraphEditor::replaceNodes(g, {batchnormNode, memcpyNode}, {newBnNode});
    HB_ASSERT(status == REPLACE_NODE_SUCCESS,
              "Failed fusing memcopy node {} into batch norm node {}",
              batchnormNode->getNodeName(),
              memcpyNode->getNodeName());
    LOG_TRACE(GC,
              "Fused memcopy node {} into batch norm node {}",
              batchnormNode->getNodeName(),
              memcpyNode->getNodeName());
}

template<class C, class V>
static bool contains(const C& c, const V& v)
{
    return std::find(begin(c), end(c), v) != end(c);
}

bool fuseBatchNormMemCpy(HabanaGraph& g)
{
    if (!GCFG_ENABLE_BATCH_NORM_MEMCPY_FUSION.value())
    {
        return true;
    }

    auto fwdStage1Guids = getBN1Guids(Direction::FWD);
    auto fwdStage2Guids = getBN2Guids(Direction::FWD);
    auto bwdStage1Guids = getBN1Guids(Direction::BWD);

    // creating a copy since the graph is modified in the loop
    const auto& nodeSet = g.getNodes();
    NodeVector  nodeVec(nodeSet.begin(), nodeSet.end());

    for (const auto& node : nodeVec)
    {
        if (!node || !HabanaGraph::runsOnTPC(node)) continue;

        if (contains(fwdStage1Guids, node->getGUID()))
        {
            fuseMemcpyOutputToBnOutput(node, g, eMemcpyBeforeInput);
        }
        else if (contains(fwdStage2Guids, node->getGUID()) || contains(bwdStage1Guids, node->getGUID()))
        {
            fuseMemcpyOutputToBnOutput(node, g, eMemcpyAfterOutput);
        }
    }
    return true;
}