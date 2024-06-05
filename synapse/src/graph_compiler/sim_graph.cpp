#include <memory>
#include "graph_traits.h"
#include "sim_graph.h"
#include "synapse_common_types.h"
#include "code_generation/code_generator_factory.h"

SimGraph::SimGraph()
{
    GlobalConfManager::instance().setDeviceType(getDeviceType());
    m_graphTraits   = std::make_shared<GraphTraits>(synDeviceGaudi);
    m_codeGenerator = CodeGeneratorFactory::createCodeGenerator(synDeviceEmulator, this);
}

SimGraph::SimGraph(const SimGraph& other) : HabanaGraph(other) {}

HabanaGraphPtr SimGraph::clone(bool cloneAllocators, bool keepMappings) const
{
    return HabanaGraphPtr(new SimGraph(*this));
}

SimGraph& SimGraph::operator=(const SimGraph& other)
{
    if (this != &other)
    {
        HabanaGraph::operator=(other);
        SimGraph tmp(other);
        std::swap(m_codeGenerator, tmp.m_codeGenerator);
    }
    return *this;
}

SimGraph::~SimGraph() = default;

bool SimGraph::compile()
{
    if (!validateConnections())
    {
        return false;
    }
    return true;
}

bool SimGraph::execute()
{
    if (!isAcyclicGraph())
    {
        return false;
    }
    for (pNode n : getTopoSortedNodes())
    {
        n->RunOnCpu(*this);
    }
    return true;
}

HabanaGraphPtr SimGraph::createEmptyGraph() const
{
    return std::make_unique<SimGraph>();
}