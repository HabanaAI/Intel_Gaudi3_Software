#include "sliced_node_graph_generator.h"
#include "access_pattern.h"

using namespace gc::layered_brain;

NodeSet SlicedNodeGraphGenerator::createSlicedNode()
{
    SET_TEMP_LOG_CONTEXT("SlicedNodeGraphGenerator");
    HB_ASSERT(m_bundleNodes.size() == 1, "SlicedNodeGraphGenerator expects a single node to slice!");
    LOG_TRACE(LB_SLICER, "Creating sliced graph for node {}", m_bundleNodes.front()->getNodeName());

    BaseClass::createSlicedNodes();
    LOG_TRACE(LB_SLICER, "Sliced node into {} nodes", m_slicedNodes.size());

    return m_slicedNodes;
}
