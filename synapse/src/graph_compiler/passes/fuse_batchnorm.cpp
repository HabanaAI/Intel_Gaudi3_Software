#include "fuse_batchnorm_patterns.h"
#include "habana_graph.h"
#include "graph_editor.h"

using BnStage2PatternFuserPtr = std::shared_ptr<BnStage2PatternFuser>;

static void printPatternMatchResult(const BnStage2PatternFuserPtr& pattern, const PatternMatch& match)
{
    LOG_DEBUG(FUSE_BATCH_NORM, "{}: found valid pattern match {}", HLLOG_FUNC, pattern->getName());
    if (!LOG_LEVEL_AT_LEAST_TRACE(FUSE_BATCH_NORM)) return;
    LOG_TRACE(FUSE_BATCH_NORM, "Printing pattern match for pattern {}. PatternNode -> GraphNode", pattern->getName());
    for (auto it : match)
    {
        LOG_TRACE(FUSE_BATCH_NORM, "{} -> {}", it.first->getNodeName(), it.second->getNodeName());
    }
}

static void findAndReplacePattern(HabanaGraph& g, const BnStage2PatternFuserPtr& pattern)
{
    const auto matches = g.findMatches(pattern->getGraphPattern(), NodeTypeMatchingFunc);
    for (const auto& match : matches)
    {
        if (pattern->isValidPattern(g, match))
        {
            NodeList oldNodes;
            for (auto it : match)
            {
                oldNodes.push_back(it.second);
            }
            auto fusedNodes = pattern->fusePattern(match);
            printPatternMatchResult(pattern, match);
            GraphEditor::replaceNodes(g, oldNodes, fusedNodes);
        }
        else
        {
            LOG_TRACE(FUSE_BATCH_NORM, "{}: invalid pattern match {}", HLLOG_FUNC, pattern->getName());
        }
    }
}

bool fuseBatchNorm(HabanaGraph& g)
{
    if (!GCFG_ENABLE_FUSE_BATCH_NORM.value()) return true;

    for (const auto& patternPtr : getAllBNFuserPatterns())
    {
        findAndReplacePattern(g, patternPtr);
    }
    return true;
}
