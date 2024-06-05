#include "suggested_tensor_manipulation.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"

namespace eager_mode
{
std::string EagerModeSuggestedManipulationHandler::getSuggestedTensorManipulationNodeName(
    tpc_lib_api::TensorOperationType /*opType*/,
    unsigned /*tensorIdx*/,
    bool /*isInput*/)
{
    return std::string(static_cast<eager_mode::EagerGraph&>(m_graph).getNextNodeName());
}

void EagerModeSuggestedManipulationHandler::setManipulatedTensorNameName(Tensor& t, std::string_view /*nameSuffix*/)
{
    t.setName(static_cast<eager_mode::EagerGraph&>(m_graph).getNextTensorName(), true);
}

bool EagerModeSuggestedManipulationHandler::isTensorSparse(const TensorPtr& tensor) const
{
    if (!tensor->isDenseLayout())
    {
        return true;
    }
    // TODO SW-165104
    // For Eager logical operation logic runs at the end and for the moment unlike for graph
    // mode we do not take the outcome of logical operation logic under consideration.
    return false;
}

bool EagerModeSuggestedManipulationHandler::applySuggestedTensorManipulation()
{
    LOG_DEBUG(GC, "Applying non-empty tensor manipulation suggestion for node {}", m_node.getNodeName());
    // Preparing for 2nd init call, removing leftovers from previous init run
    TensorVector inputs  = filterAuxAndNullTensors(m_node.getInputs());
    TensorVector outputs = filterAuxAndNullTensors(m_node.getOutputs());
    // These will hold modified Tensors
    TensorVector newInputs;
    newInputs.reserve(inputs.size());
    TensorVector newOutputs;
    newOutputs.reserve(outputs.size());

    m_newInputPermutations = m_node.getNodeAnnotation().inputPermutations;

    bool abortManipulation = false;

    // process inputs
    bool inputsModified =
        applyManipulationCheckAndProcess(m_suggestion.inputTensors, inputs, newInputs, abortManipulation, true);
    // process outputs
    bool outputsModified =
        applyManipulationCheckAndProcess(m_suggestion.outputTensors, outputs, newOutputs, abortManipulation, false);

    // if there is no manipulation to do, or there was an unexpected error while applying the manipulation
    // leave the graph unmodified. there is an assumption here that if outputs are modified, inputs should too
    // for all current suggestion cases.
    if ((!inputsModified && !outputsModified) || abortManipulation) return false;

    m_node.setSuggestedOptimizationDone(true);

    // add nodes for inputs and outputs
    if (!addNodesNeededForSelectedManipulation(&inputs, &newInputs, true) ||
        !addNodesNeededForSelectedManipulation(&outputs, &newOutputs, false))

    {
        // if something went wrong when trying to add needed nodes, abort the suggestion altogether.
        return false;
    }

    // apply the modification to the original tpc node in place
    m_node.resetInstantiated();
    m_node.replaceAllTensors(std::move(newInputs), std::move(newOutputs));
    m_node.getNodeAnnotation().inputPermutations = m_newInputPermutations;

    return true;
}

bool EagerModeSuggestedManipulationHandler::shouldSkipSuggestedTensorManipulation(TPCNode&           node,
                                                                                  const HabanaGraph& graph)
{
    return !GCFG_ENABLE_EAGER_ARCH_OPTIMIZATIONS.value() || !GCFG_ENABLE_SUGGESTED_MANIPULATION_IN_EAGER.value() ||
           SuggestedManipulationHandlerBase::shouldSkipSuggestedTensorManipulation(node, graph);
}

}  // namespace eager_mode