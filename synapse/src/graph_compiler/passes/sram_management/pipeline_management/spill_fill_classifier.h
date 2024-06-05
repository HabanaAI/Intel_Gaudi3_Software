#pragma once

#include "habana_graph.h"
#include "types.h"

class SpillFillClassifier
{
public:
    explicit SpillFillClassifier(const HabanaGraph& g);
    const NodeVector& getSpillDirectives() { return m_spillDirectives; }
    const NodeVector& getFillDirectives() { return m_fillDirectives; }

private:
    bool isSpillDirective(const NodePtr& memcpy);
    bool isFillDirective(const NodePtr& memcpy);

    NodeVector m_spillDirectives;
    NodeVector m_fillDirectives;
};
