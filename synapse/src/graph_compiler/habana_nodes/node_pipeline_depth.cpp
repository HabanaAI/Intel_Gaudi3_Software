#include "concatenate_node.h"
#include "habana_graph.h"
#include "logical_op_node.h"

#include "node_pipeline_depth.h"

NodePipelineDepth::NodePipelineDepth(const HabanaGraph& graph)
: m_pipelineDepth(0)
, m_graph(graph)
{
}

void NodePipelineDepth::visit(Node* node)
{
    std::list<NodeROI>* rois = m_graph.GetNodeROIs(m_graph.getTensorProducer(node->getOutput(0)));
    if (rois)
    {
        m_pipelineDepth = rois->size();
    }
    else
    {
        m_pipelineDepth = 0;
    }
}

void NodePipelineDepth::visit(LogicalOpNode* node)
{
    NodePtr nodeToCheck = m_graph.getTensorProducer(node->getInput(0));
    if (nodeToCheck)
    {
        nodeToCheck->accept(this);
    }
    else
    {
        m_pipelineDepth = 0;
    }
}

void NodePipelineDepth::visit(ConcatenateNode* node)
{
    m_pipelineDepth = node->getNumInputs();
}

void NodePipelineDepth::visit(SplitNode* node)
{
    m_pipelineDepth = 1;
}

