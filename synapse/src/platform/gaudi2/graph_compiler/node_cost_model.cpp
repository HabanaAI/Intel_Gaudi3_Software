#include "node_cost_model.h"
#include "dma_cost_model.h"
#include "habana_graph.h"
#include "pipeline_management/mme_brain_proxy.h"
#include <optional>

namespace gaudi2
{
std::optional<std::pair<NodeCostModel::EngineType, double>> NodeCostModel::getNodeExpectedDuration(const NodePtr& node)
{
    if (node->isDma())
    {
        DMANode* dmaNode = dynamic_cast<DMANode*>(node.get());
        HB_ASSERT(dmaNode, "node is not dma node");
        DmaCostModel costModel(*m_graph.getHALReader(), &m_graph);
        // for the moment the dma cost model stimulates the split into ROIs from scratch
        // and does not support also computation based on actual generated descriptors.
        // so we cache the computed cost to avoid re-calculation.
        DmaCost result = costModel.getCostModelResult(*dmaNode);
        node->getNodeAnnotation().dmaCost = result;
        return std::make_pair(NodeCostModel::EngineType::DMA, result.durationInUsec);
    }
    else if (HabanaGraph::runsOnMME(node))
    {
        MmeCommon::PerfAttr perfAttr = MmeBrainProxy::getRecommendedConfigMmePerf(node);
        return std::make_pair(NodeCostModel::EngineType::MME, perfAttr.expectedRuntime);
    }
    if (HabanaGraph::runsOnTPC(node))
    {
        TPCNode* tpcNode = dynamic_cast<TPCNode*>(node.get());
        HB_ASSERT(tpcNode, "node is not tpc node");
        std::optional<TPCNode::CostModelResult> result = tpcNode->getCostModelResult();
        if (!result.has_value()) return std::nullopt;
        return std::make_pair(NodeCostModel::EngineType::TPC, result->asicTimeInUsec);
    }
    return std::nullopt;
}

}  // namespace gaudi2