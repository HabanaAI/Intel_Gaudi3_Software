#include "passes.h"
#include "gaudi_graph.h"
#include "habana_nodes.h"
#include "node_factory.h"
#include "graph_editor.h"
#include <string>
namespace gaudi
{
bool addH2DOp(GaudiGraph& g)
{
    const NodeSet nodes = g.getNodes();
    for (const NodePtr& node : nodes)
    {
        if (g.runsOnTPC(node))
        {
            // Getting inputs by copy because actual node inputs
            // may be modified in the loop
            TensorVector inputs = node->getInputs();
            for (unsigned index = 0; index < inputs.size(); index++)
            {
                if (inputs[index] == nullptr) continue;
                if (inputs[index]->isHost2DeviceTensor() && !inputs[index]->isHostOnly())
                {
                    // add a new device tensor
                    TensorPtr newTensor = inputs[index]->clone(false, false, false);
                    newTensor->setTensorType(DATA_TENSOR);
                    newTensor->setTwinHost2DeviceTensor(inputs[index]);
                    GraphEditor::editNode(g, node, [&](const NodePtr& n) {
                        n->replaceInput(index, newTensor, Node::TENSOR_TYPE_DATA);
                    });

                    // add a DMA node
                    NodePtr dmaNode = NodeFactory::createNode({inputs[index]},
                                                              {newTensor},
                                                              nullptr,
                                                              NodeFactory::memcpyNodeTypeName,
                                                              inputs[index]->getName() + "_copy_internal");

                    // Maintain tracking of origin nodes for debug purposes
                    dmaNode->setOriginNodes(node->getOriginNodes());

                    bool    status  = GraphEditor::addNode(g, dmaNode);
                    HB_ASSERT(status, "Failed adding DMA node for {}", inputs[index]->getName());
                }
            }
        }
    }

    return true;
}
}  // namespace gaudi
