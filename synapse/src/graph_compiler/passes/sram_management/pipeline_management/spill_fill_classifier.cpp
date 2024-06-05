#include "spill_fill_classifier.h"
#include "dma_memcopy_node.h"

bool SpillFillClassifier::isSpillDirective(const NodePtr& memcpy)
{
    return memcpy->getInput(0)->inSram() && !memcpy->getOutput(0)->inSram();
}

bool SpillFillClassifier::isFillDirective(const NodePtr& memcpy)
{
    return !memcpy->getInput(0)->inSram() && memcpy->getOutput(0)->inSram();
}

SpillFillClassifier::SpillFillClassifier(const HabanaGraph& g)
{
    for (const auto& node : g.getNodes())
    {
        // Spill/fill directives are described as memcpy nodes. Currently NDims (>5) nodes are not supported.
        if (!isMemcpy(*node) || node->hasHighRankOperand()) continue;

        if (isSpillDirective(node)) m_spillDirectives.push_back(node);
        else if (isFillDirective(node)) m_fillDirectives.push_back(node);
    }
}