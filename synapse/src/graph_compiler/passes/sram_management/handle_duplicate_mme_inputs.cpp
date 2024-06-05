#include "handle_duplicate_mme_inputs.h"
#include "identity_node.h"
#include "node_factory.h"

void DuplicateMmeInputsHandler::handleDuplicateMmeInputs(HabanaGraph& g)
{
    std::vector<std::pair<NodePtr, unsigned>> nodeInputsDuplicates;
    std::unordered_map<TensorPtr, unsigned>   numDuplications;

    for (const NodePtr& n : g.getNodes())
    {
        if (!n) continue;
        if (!HabanaGraph::runsOnMME(n)) continue;  // handle only mme nodes for now.
        for (unsigned inputIdx = 0; inputIdx < n->getNumInputs(); inputIdx++)
        {
            const TensorPtr& input = n->getInput(inputIdx);
            if (!input) continue;
            if (n->getInputIndexOfTensor(input) != inputIdx)  // duplicate input
            {
                nodeInputsDuplicates.push_back(std::make_pair(n, inputIdx));
                numDuplications[n->getInput(inputIdx)] = 0;
            }
        }
    }

    for (const auto& nodeIndexPair : nodeInputsDuplicates)
    {
        const NodePtr&   n        = nodeIndexPair.first;
        unsigned         inputIdx = nodeIndexPair.second;
        const TensorPtr& input    = n->getInput(inputIdx);
        LOG_DEBUG(SRAM_SLICE, "removing duplicate input for node {}, inputIdx {}", n->getNodeName(), inputIdx);

        TensorPtr dup = input->clone(false, false, false);
        NodePtr   identityNode =
            NodeFactory::createNode({input},
                                    {dup},
                                    nullptr,
                                    NodeFactory::identityNodeTypeName,
                                    fmt::format("{}_duplicate_{}", input->getName(), numDuplications.at(input)));
        // Maintain tracking of origin nodes for debug purposes
        identityNode->setOriginNodes(n->getOriginNodes());

        GraphEditor::addNode(g, identityNode);           // add identity
        GraphEditor::replaceInput(g, n, inputIdx, dup);  // replace duplicate node input
        numDuplications[input]++;                        // make sure we don't create multiple nodes with the same name
    }
}