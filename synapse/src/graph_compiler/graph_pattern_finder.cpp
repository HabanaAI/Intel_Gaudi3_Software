#include "graph.h"
#include "graph_pattern_finder.h"
#include "infra/defs.h"

std::list<NodesList> GraphPatternFinder::matchPattern(const Graph& origGraph, const Graph& pattern)
{
    GraphPatternFinder finder(origGraph, pattern);
    return finder._run();
}

GraphPatternFinder::GraphPatternFinder(const Graph& origGraph, const Graph& pattern)
: m_origGraph(origGraph)
, m_pattern(pattern)
{
}

std::list<NodesList> GraphPatternFinder::_run()
{
    std::list<NodesList> patternsList;
    const auto& rootNodes = m_pattern.getRootNodes();
    HB_ASSERT(rootNodes.size() == 1, "Expected only single root node for pattern matching");
    const auto& firstPatternNode = rootNodes.front();
    unsigned int patternNodesAmount = m_pattern.getNumNodes();

    const auto& allGraphNodes = m_origGraph.getNodes();

    for (const std::shared_ptr<Node>& node : allGraphNodes)
    {
        const std::list<NodesList>& candidatePatterns = _getNodePattern(node, firstPatternNode);
        for (const NodesList& singleCandidate : candidatePatterns)
        {
            if (singleCandidate.size() == patternNodesAmount)
            {
                patternsList.push_back(singleCandidate);
            }
        }
    }

    return patternsList;
}

std::list<NodesList> GraphPatternFinder::_getNodePattern(const std::shared_ptr<Node>& origNode, const std::shared_ptr<Node>& patternNode)
{
    std::list<NodesList> ret;
    if (origNode->getNodeType() == patternNode->getNodeType())
    {
        NodesList patternNodes(1, origNode);
        ret.push_back(patternNodes);
        NodesList patternConsumers;
        for (const std::shared_ptr<Tensor>& patternOutput : patternNode->getOutputs())
        {
            const NodesList consumers = m_pattern.getTensorConsumers(patternOutput);
            patternConsumers.insert(patternConsumers.end(), consumers.begin(), consumers.end());
        }
        if (! patternConsumers.empty())
        {
            NodesList origNodeConsumers;
            for (const std::shared_ptr<Tensor>& origNodeOutput : origNode->getOutputs())
            {
                const NodesList consumers = m_origGraph.getTensorConsumers(origNodeOutput);
                origNodeConsumers.insert(origNodeConsumers.end(), consumers.begin(), consumers.end());
            }
            const auto& manyToManyRes = _getManyNodePatterns(origNodeConsumers, patternConsumers);
            _combineNodeLists(manyToManyRes, ret);
        }
    }
    return ret;
}

std::list<NodesList> GraphPatternFinder::_getManyNodePatterns(const NodesList& origNodes, const NodesList& patternNodes)
{
    std::list<NodesList> ret;
    for (const std::shared_ptr<Node>& patternNode : patternNodes)
    {
        std::list<NodesList> tmp;
        for (const std::shared_ptr<Node>& origNode : origNodes)
        {
            const std::list<NodesList>& SinglPatterns = _getNodePattern(origNode, patternNode);
            tmp.insert(tmp.end(), SinglPatterns.begin(), SinglPatterns.end());
        }
        _combineNodeLists(tmp, ret);
    }
    return ret;
}

void GraphPatternFinder::_combineNodeLists(const std::list<NodesList>& input, std::list<NodesList>& output)
{
    if (output.empty())
    {
        output = input;
    }
    else
    {
        std::list<NodesList> outputCopy = output;
        for (unsigned int duplicateIdx = 1; duplicateIdx < input.size(); ++duplicateIdx)
        {
            output.insert(output.end(), outputCopy.begin(), outputCopy.end());
        }
        for (const auto& inputList : input)
        {
            for (auto& outputList : output)
            {
                outputList.insert(outputList.end(), inputList.begin(), inputList.end());
            }
        }
    }
}
