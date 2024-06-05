#include <role_pattern.h>
#include <synapse_common_types.h>
#include <node_factory.h>
#include "habana_graph.h"

// forward declaration
bool matchSymmetricRolePattern(HabanaGraph&              g,
                               const pNode&              graphNode,
                               const pNode&              graphProducerNode,
                               RolePattern*              pattern,
                               const pNode&              patternProducerNode,
                               std::map<int, NodeList>&  matches,
                               int                       mapKey,
                               NodeList&                 currNodesList);

// Role pattern mechanism is a virtual and generic mechanism used for pattern matching in a graph.
// Used mainly for but not limited to passes which perform nodes fusion.
RolePattern::RolePattern()
{
}

RolePattern::~RolePattern()
{
}

GraphPtr RolePattern::getPattern()
{
    return m_pattern;
}

// A symmetric Role is a node role that has exactly 2 symmetric inputs (e.g. add_*)
// To-do: enable this feature (it is currently disabled).
bool RolePattern::isSymmetricRole(const pNode n)
{
    return false;
}

template<class T>
bool RolePattern::addNode(const TensorVector& inputs,
                          const TensorVector& outputs,
                          const T*            userParams,
                          const char*         guid,
                          const std::string&  name)
{
    pNode newNode = NodeFactory::createNode(inputs, outputs, userParams, guid, name);
    return m_pattern->addNode(newNode);

}

std::pair<int, int> RolePattern::numInputsRange(const pNode& n) const
{
    return std::make_pair(n->getNumInputs(), n->getNumInputs());
}

// Match a pattern of nodes in a graph. begin at the graph output produces and traverse backwards recursively.
bool matchRolePatternBackward(HabanaGraph&              g,
                              const pNode&              graphNode,
                              RolePattern*              pattern,
                              const pNode&              patternNode,
                              std::map<int, NodeList>&  matches,
                              int                       mapKey)

{
    //match current node
    if ((graphNode == nullptr) || (patternNode == nullptr) || !(pattern->rolesMatch(g, graphNode, patternNode)))
    {
        return false;
    }

    //recursion base case (reached pattern root)
    NodeList patternRoots = pattern->getPattern()->getRootNodes();
    if (std::find(patternRoots.begin(), patternRoots.end(), patternNode) != patternRoots.end())
    {
        matches[mapKey].push_back(graphNode);
        return true;
    }

    //match producers for all inputs
    if ((graphNode->getNumInputs() < pattern->numInputsRange(patternNode).first) ||
        (graphNode->getNumInputs() > pattern->numInputsRange(patternNode).second))
    {
        return false;
    }

    TensorVector graphInputs    = graphNode->getInputs();
    TensorVector patternInputs  = patternNode->getInputs();

    // in case of a symmetric role Node (such as add_*) we need to match both graph node inputs
    // to both inputs of the pattern node and provide all matching patterns
    if (pattern->isSymmetricRole(graphNode))
    {
        NodeList currNodesList  = matches[mapKey];
        pNode graphProducerA    = g.getTensorProducer(graphInputs[0]);
        pNode graphProducerB    = g.getTensorProducer(graphInputs[1]);
        pNode patternProducerA  = pattern->getPattern()->getTensorProducer(patternInputs[0]);
        pNode patternProducerB  = pattern->getPattern()->getTensorProducer(patternInputs[1]);

        bool retSym1 = matchSymmetricRolePattern(g, graphNode, graphProducerA, pattern, patternProducerA,
                                                 matches, mapKey, currNodesList);
        bool retSym2 = matchSymmetricRolePattern(g, graphNode, graphProducerA, pattern, patternProducerB,
                                                 matches, mapKey+1, currNodesList);
        bool retSym3 = matchSymmetricRolePattern(g, graphNode, graphProducerB, pattern, patternProducerA,
                                                 matches, mapKey+2, currNodesList);
        bool retSym4 = matchSymmetricRolePattern(g, graphNode, graphProducerB, pattern, patternProducerB,
                                                 matches, mapKey+3, currNodesList);
        return (retSym1 || retSym2 || retSym3 || retSym4);
    }
    else
    {
        for (unsigned i = 0; i < std::min(patternNode->getNumInputs(), graphNode->getNumInputs()); ++i)
        {
            //input producer
            pNode graphProducer   = g.getTensorProducer(graphInputs[i]);
            pNode patternProducer = pattern->getPattern()->getTensorProducer(patternInputs[i]);
            if (patternProducer == nullptr)
            {
                continue;
            }
            //match pattern for producers
            if (matchRolePatternBackward(g, graphProducer, pattern, patternProducer, matches, mapKey))
            {
                // if pattern producers matched, add graph node to all existing matches
                auto matchesIter = matches.begin();
                while (matchesIter != matches.end())
                {
                    matches[matchesIter->first].push_back(graphNode);
                    matchesIter++;
                }
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

bool matchSymmetricRolePattern(HabanaGraph&              g,
                               const pNode&              graphNode,
                               const pNode&              graphProducerNode,
                               RolePattern*              pattern,
                               const pNode&              patternProducerNode,
                               std::map<int, NodeList>&  matches,
                               int                       mapKey,
                               NodeList&                 currNodesList)
{
    bool foundMatch = false;
    matches[mapKey] = currNodesList;
    if (matchRolePatternBackward(g, graphProducerNode, pattern, patternProducerNode, matches, mapKey))
    {
        foundMatch = true;
        matches[mapKey].push_back(graphNode);
    }
    else
    {
        matches.erase(mapKey);
    }
    return foundMatch;
}

// match multiple Role pattern is a graph, and return all matches for each pattern
NodesPatternVector matchMultiplePatternsWithSingleOutputNode(HabanaGraph& g, const RolePatternVector& patterns)
{
    NodesPatternVector results;
    for (pNode n : g.getNodes())
    {
        for (auto it = std::begin(patterns); it != std::end(patterns); ++it)
        {
            GraphPtr pattern = (*it)->getPattern();
            HB_ASSERT_PTR(pattern);
            pTensor t = pattern->getGraphOutputs().front();
            pNode targetNode = pattern->getTensorProducer(t);
            NodeList newList;
            std::map<int, NodeList> matches;
            matches[0] = newList;

            if (matchRolePatternBackward(g, n, *it, targetNode, matches, 0))
            {
                auto matchesIter = matches.begin();
                while (matchesIter != matches.end())
                {
                    if (matchesIter->second.size() > 0)
                    {
                        results.push_back(std::make_pair(matchesIter->second, *it));
                    }
                    matchesIter++;
                }
            }
        }
    }
    std::reverse(results.begin(), results.end());
    return results;
}
