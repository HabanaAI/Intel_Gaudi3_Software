#ifndef _GRAPH_FACTORY_H_
#define _GRAPH_FACTORY_H_

#include "habana_graph.h"

class GraphFactory
{
public:
    static HabanaGraphPtr createGraph(synDeviceType deviceType, CompilationMode compilationMode);

private:
    static HabanaGraphPtr createSimGraph();
    static HabanaGraphPtr createGaudiGraph();
    static HabanaGraphPtr createGaudi2Graph();
    static HabanaGraphPtr createGaudi3Graph();
    static HabanaGraphPtr createEagerGraph(synDeviceType deviceType);
};

#endif
