#include "habana_graph.h"
#include "node_factory.h"
#include "habana_global_conf.h"

bool insertProbeNanNodes(HabanaGraph& g)
{
    if (GCFG_ENABLE_NAN_INF_PROBE.value() == false)
    {
        return true;
    }
    NodeList probeNodes;
    for (NodePtr node : g.getExeSortedNodes())
    {
        if (g.runsOnMME(node) || g.runsOnTPC(node))
        {
            for (TensorPtr tensor : node->getOutputs())
            {
                const auto consumers = g.getTensorConsumers(tensor);
                // In setReductionMemset there is assumption that reduction is the
                // only consumer of the tensor
                // Do not add probe nan to keep it valid
                const auto internalReduction =
                    std::any_of(std::begin(consumers), std::end(consumers), [](const NodePtr& n) {
                        return n->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
                    });

                if (tensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled || internalReduction)
                {
                    continue;
                }
                std::string guid;
                if (tensor->getElementType() == syn_type_bf16)
                {
                    guid = "probe_nan_bf16";
                }
                else if (tensor->getElementType() == syn_type_float)
                {
                    guid = "probe_nan_f32";
                }
                if (!guid.empty())
                {
                    TSize   tensorSize = tensor->getTotalElements();
                    TSize   sizes[]    = {tensorSize};
                    pTensor dummy      = std::make_shared<Tensor>(1, sizes, tensor->getElementType());
                    pNode   probeNode =
                        NodeFactory::createNode({tensor}, {dummy}, nullptr, guid, tensor->getName() + "_probe");
                    probeNodes.push_back(probeNode);
                }
            }
        }
    }

    for (NodePtr node : probeNodes)
    {
        GraphEditor::addNode(g, node);
    }
    return true;
}
