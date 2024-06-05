#include "strided_view_insert_perf_check.h"
#include "node.h"
#include "strided_op_node_utils.h"
#include "strided_view_logical_node.h"
#include "strided_insert_logical_node.h"

GcPerf::GcPerfCheckPtr StridedOpPerfCheck::createPerfCheck(const HabanaGraph& g)
{
    return GcPerf::GcPerfCheckPtr(new StridedOpPerfCheck(g));
}

StridedOpPerfCheck::StridedOpPerfCheck(const HabanaGraph& g) : GcPerf::GcPerfCheck(g) {}

void StridedOpPerfCheck::run(const std::shared_ptr<Node>& node) const
{
    if (node->getNodeType() != Node::TYPE_STRIDED_VIEW && node->getNodeType() != Node::TYPE_STRIDED_INSERT) return;
    const auto* sv = dynamic_cast<LogicalStridedViewNode*>(node.get());
    const auto* si = dynamic_cast<LogicalStridedInsertNode*>(node.get());
    HB_ASSERT(sv || si, "could node cast node {} as StridedView/StridedInsert", node->getNodeName());

    const auto&      params = sv ? sv->getParams() : si->getParams();
    const TensorPtr& view   = sv ? sv->getOutput(0) : si->getInput(LogicalStridedInsertNode::INSERT_TENSOR);

    bool isDense   = StridedOpUtils::isDenseStridedOpParams(params, view);
    bool isDynamic = node->isDynamicShape();

    auto logLevel = GcPerf::LogLevel::LOW;
    if (!isDense || isDynamic)
    {
        logLevel = GcPerf::LogLevel::HIGH;  // strided view insert that can cause redundant copies
    }

    PERF_REPORT(logLevel,
                "{} Node: {} - with view tensor sizes={} minSizes={}, and parameters: {} might cause redundant copies",
                sv ? "StridedView" : "StridedInsert",
                node->getNodeName(),
                view->getDimSizesStr(),
                view->getDimSizesStr(false, true),
                StridedOpUtils::stridedOpParamsString(params, view->getDim()));
}