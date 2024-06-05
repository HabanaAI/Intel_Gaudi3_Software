#include "add_mme_bias.h"

#include <graph_editor.h>
#include "node_factory.h"
#include "data_type_utils.h"
#include "habana_graph.h"
#include "passes.h"

NodePtr MmeBiasNodeHandler::extract()
{
    calcExtract();
    m_node->removeInput(m_bias);
    m_node->replaceOutput(TENSOR_OFM, m_mmeOutput);
    return m_addNode;
}

bool MmeBiasNodeHandler::extract(HabanaGraph& graph)
{
    calcExtract();
    GraphEditor::editNode(graph, m_node, [&](const NodePtr& n) { n->removeInput(m_bias); });
    GraphEditor::replaceOutput(graph, m_node, TENSOR_OFM, m_mmeOutput);
    return (GraphEditor::replaceNodes(graph, {m_node}, {m_node, m_addNode}) == REPLACE_NODE_SUCCESS);
}

void MmeBiasNodeHandler::calcExtract()
{
    HB_ASSERT_PTR(m_bias);
    HB_ASSERT(m_mmeOutput == nullptr, "Output tensor is expected to be null");
    HB_ASSERT(m_addNode == nullptr, "New node is expected to be null");
    const TensorPtr& output = m_node->getOutput(TENSOR_OFM);
    m_mmeOutput             = output->clone();
    m_mmeOutput->setName(output->getName() + "_add_input");
    m_mmeOutput->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);

    m_addNode = NodeFactory::createInternalNode({m_bias, m_mmeOutput},
                                                {output},
                                                nullptr,
                                                getAddFwdGuid(output->getElementType()),
                                                fmt::format("{}_add_bias", m_node->getNodeName()),
                                                "add_fwd");
}

std::string MmeBiasNodeHandler::getAddFwdGuid(synDataType type)
{
    return fmt::format("add_fwd_{}", getDtypeSuffixFromSynDataType(type));
}

bool addMmeBias(HabanaGraph& graph)
{
    const NodeVector& mmeNodes = graph.getSortedMMENodes();
    for (const auto& node : mmeNodes)
    {
        if (MmeBiasNodeHandler::canExtract(node))
        {
            CHECK_RET_FALSE(MmeBiasNodeHandler(node).extract(graph),
                            "Could not extract bias from convolution {}",
                            node->getNodeName());
        }
    }
    return true;
}
