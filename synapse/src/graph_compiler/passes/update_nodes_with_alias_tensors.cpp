
#include "update_nodes_with_alias_tensors.h"

#include "graph_editor.h"
#include "habana_graph.h"
#include "passes.h"
#include "tpc_node.h"
#include "types.h"
#include "utils.h"

#include <array>
#include <memory>
#include <string_view>

constexpr std::string_view        BATCH_NORM_FWD_GUID            = "batch_norm_fwd";
constexpr std::array<unsigned, 2> BATCH_NORM_ALIAS_TENSORS_INDEX = {3, 4};

static inline bool isAliasingRequired(const NodePtr& node)
{
    // Currently only batchnorm is supported
    return isGuidPrefix(node,BATCH_NORM_FWD_GUID);
}

bool updateNodesWithAliasTensors(HabanaGraph& graph)
{
    auto fn = [](const NodePtr& node){return isAliasingRequired(node);};
    NodeSet sortedNodes = graph.getNodesCond(fn);

    for (auto nodeIter = sortedNodes.rbegin(); nodeIter != sortedNodes.rend(); ++nodeIter)
    {
        NodePtr candidate = *nodeIter;

        if (!HabanaGraph::runsOnTPC(candidate))
        {
            continue;
        }

        auto& tpcNode = static_cast<TPCNode&>(*candidate);
        GraphEditor::editNode(graph, candidate, [&]() {
            AliasTensors::updateNodesWithAliasTensors(std::static_pointer_cast<TPCNode>(candidate));
        });
    }

    return true;
}

void AliasTensors::updateNodesWithAliasTensors(const TPCNodePtr& tpcNode)
{
    if (!isAliasingRequired(tpcNode))
    {
        return;
    }

    for (auto tensorIndexIter : BATCH_NORM_ALIAS_TENSORS_INDEX)
    {
        if (tpcNode->getOutput(tensorIndexIter))
        {
            continue;
        }

        const TensorPtr& inputTensor       = tpcNode->getInput(tensorIndexIter);
        TensorPtr        aliasOutputTensor = inputTensor->cloneGeometry();
        aliasOutputTensor->setName(fmt::format("{}_alias", inputTensor->getName()));
        aliasOutputTensor->setAsAliasSubTensor(inputTensor);
        aliasOutputTensor->maskOutput();

        tpcNode->addOutput(aliasOutputTensor);
    }
}
