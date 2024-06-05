#pragma once

#include "habana_graph.h"

class LayerNormFuser {
    private:
        bool         constructLayerNormPattern(Graph* pattern);
        HabanaGraph& m_graph;
    public:

        LayerNormFuser(HabanaGraph& g) : m_graph(g) {};

        bool fuseLayerNorm(Graph* pattern);
};
