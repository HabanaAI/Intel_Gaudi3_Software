#pragma once

#include <list>
#include <memory>

class Graph;
class Node;

typedef std::list<std::shared_ptr<Node>> NodesList;

/**
 * Find pattern in a given graph
 * Support only one input for pattern graph
 *
 */
class GraphPatternFinder
{
public:
    /**
     * Return list of (list of) original nodes match the given pattern
     */
    static std::list<NodesList> matchPattern(const Graph& origGraph, const Graph& pattern);

private:
    explicit GraphPatternFinder(const Graph& origGraph, const Graph& pattern);

    std::list<NodesList> _run();

    std::list<NodesList> _getNodePattern(const std::shared_ptr<Node>& origNode,
                                         const std::shared_ptr<Node>& patternNode);

    std::list<NodesList> _getManyNodePatterns(const NodesList& origNodes, const NodesList& patternNodes);

    static void _combineNodeLists(const std::list<NodesList>& input, std::list<NodesList>& output);

    const Graph& m_origGraph;
    const Graph& m_pattern;
};
