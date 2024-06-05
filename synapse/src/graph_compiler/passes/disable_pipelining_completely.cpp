#include "habana_graph.h"
#include "passes.h"
#include "habana_global_conf.h"

//avoid any pipelining between different engines.
bool disablePipeliningCompletely(HabanaGraph &g)
{
    SET_TEMP_LOG_CONTEXT(__FUNCTION__);
    LOG_DEBUG(GC, "setting control edges between adjacent nodes in execution order");
    NodeVector nodes    = g.getExeSortedNodes();
    pNode lastNode = nullptr;
    for (pNode currNode : nodes)
    {
        //set pipelineDepth to 1 to ensure adjacent producers-consumers will not pipeline
        currNode->getNodeAnnotation().pipelineDepth = 1;
        if (!lastNode)
        {
            lastNode = currNode;
            continue;
        }
        if (currNode->isLogicalOperation()) continue;
        LOG_TRACE(GC, "Setting node {} as a barrier of node {}", lastNode->getNodeName(),
                currNode->getNodeName());
        g.addControlDependency(lastNode, currNode);

        lastNode = currNode;
    }
    return true;
}

