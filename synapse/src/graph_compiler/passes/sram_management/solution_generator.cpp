#include "solution_generator.h"

class OperationScheduler : public OperationHandler
{
public:
    explicit OperationScheduler(Bundle::Solution& solution)
            : m_solution(solution) {}
    void handleOperation(const pNode& node,
                         const SliceReferenceList& inputs,
                         const SliceReferenceList& outputs) override
    {
        // Create a slice operation, corresponding to the given node
        m_solution.operations.emplace_back(node);
        auto& op = m_solution.operations.back();
        // Attach the relevant slice references as inputs / outputs
        op.inputs.insert(op.inputs.end(), inputs.begin(), inputs.end());
        op.outputs.insert(op.outputs.end(), outputs.begin(), outputs.end());
    }

    Bundle::Solution& m_solution;
};

bool SolutionGenerator::fillSolution()
{
    Bundle::Solution& solution = m_bundle->getSolution();
    solution.operands = m_strategy->getSlicingData().getSlicedOperands();

    HandleEachStrategyOperation handleEachStrategyOperation(m_graph,
                                                            m_bundle->index(),
                                                            m_bundle->getNodes(),
                                                            m_strategy,
                                                            true /* traceLog */);
    OperationScheduler operationScheduler(solution);
    return handleEachStrategyOperation(operationScheduler);
}

bool SolutionGeneratorMantaRay::fillSolution()
{
    Bundle::Solution& solution = m_bundle->getSolution();
    solution.operands          = m_strategy->getSlicingData().getSlicedOperands();

    HandleEachStrategyOperationMantaRay handleEachStrategyOperation(m_graph,
                                                                    m_bundle->index(),
                                                                    m_bundle->getNodes(),
                                                                    m_strategy,
                                                                    true /* traceLog */);
    OperationScheduler                  operationScheduler(solution);
    return handleEachStrategyOperation(operationScheduler);
}
