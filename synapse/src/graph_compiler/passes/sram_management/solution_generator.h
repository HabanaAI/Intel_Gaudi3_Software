#pragma once

#include "strategy_operations_accumulator.h"

class SolutionGenerator
{
public:
    SolutionGenerator(const HabanaGraph& graph, const pBundle& bundle, const SlicingStrategyPtr& strategy)
    : m_graph(graph), m_bundle(bundle), m_strategy(strategy)
    {}

    // Adds sliced operands and execution order to the bundle's solution.
    virtual bool fillSolution();

protected:
    const HabanaGraph&       m_graph;
    pBundle                m_bundle;
    const SlicingStrategyPtr m_strategy;
};

class SolutionGeneratorMantaRay : public SolutionGenerator
{
public:
    SolutionGeneratorMantaRay(const HabanaGraph& graph, const pBundle& bundle, const SlicingStrategyPtr& strategy)
    : SolutionGenerator(graph, bundle, strategy)
    {
    }
    bool fillSolution() override;
};