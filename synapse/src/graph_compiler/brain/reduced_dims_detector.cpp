#include "reduced_dims_detector.h"
#include "compilation_hal_reader.h"
#include "access_pattern.h"
#include "dma_cost_model.h"
#include "habana_graph.h"
#include "tpc_node.h"
#include <memory>

using namespace gc::layered_brain;
using namespace gc::access_pattern;

ReducedDimsDetector::ReducedDimsDetector(const NodePtr& node) : m_node(node)
{
    const auto& accessPattern = m_node->getNodeAccessPattern();
    HB_ASSERT_PTR(accessPattern);
    const NodeAccessPattern::Resolution& resolution = accessPattern->getNodeResolution();
    // Collect node dims which are mapped to any input. The access pattern may include node dims which aren't relevant,
    // so selecting only the non degenerated node dims which project to any input
    for (const auto& input: m_node->getInputs())
    {
        if (!input) continue;
        for (Dim tensorDim = 0; tensorDim < input->getDim(); tensorDim++)
        {
            auto nodeDim = accessPattern->getIndexSpaceDim(input, tensorDim);
            // Exclude node dims with resolution 1
            if (resolution.at(nodeDim) > 1)
            {
                m_allSliceableNodeDims.insert(nodeDim);
            }
        }

    }
}

std::unordered_set<Dim> ReducedDimsDetector::getReducedNodeDims() const
{
    std::unordered_set<Dim> reducedNodeDims;
    for (const auto& output : m_node->getOutputs())
    {
        HB_ASSERT_PTR(output);
        std::unordered_set<Dim> reducedNodeDimsForTensor = getReducedNodeDimsForOutput(output);
        reducedNodeDims.insert(reducedNodeDimsForTensor.begin(), reducedNodeDimsForTensor.end());
        if (!reducedNodeDimsForTensor.empty())
        {
            LOG_DEBUG(LAYERED_BRAIN,
                      "Node type {} output idx {} reduced node dims {} - node {}",
                      m_node->getGUID(),
                      m_node->getOutputIndexOfTensor(output),
                      toString(reducedNodeDimsForTensor, ','),
                      m_node->getNodeName());
        }
    }
    return reducedNodeDims;
}

std::unordered_set<Dim> ReducedDimsDetector::getReducedNodeDimsForOutput(const TensorPtr& output) const
{
    if (HabanaGraph::runsOnTPC(m_node) && !isOutputRmw(output)) return {};
    auto missingNodeDims = getMissingNodeDimsInTensor(output);
    return missingNodeDims;
}

// Return node dims which are missing in the output. TPC degenerated node dims are mapped to a dummy node dim, so if
// there is a sliceable input dim, which is mapped to the degenerated output dim, it will be missing in the output.
std::unordered_set<Dim> ReducedDimsDetector::getMissingNodeDimsInTensor(const TensorPtr& tensor) const
{
    const auto&             accessPattern = m_node->getNodeAccessPattern();
    std::unordered_set<Dim> missingDims(m_allSliceableNodeDims.begin(), m_allSliceableNodeDims.end());
    for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
    {
        missingDims.erase(accessPattern->getIndexSpaceDim(tensor, tensorDim));
    }
    return missingDims;
}

// Check if the output is RMW for TPC node
bool ReducedDimsDetector::isOutputRmw(const TensorPtr& output) const
{
    HB_ASSERT(HabanaGraph::runsOnTPC(m_node), "Expected TPC Node");
    const auto& tpcNode   = static_cast<TPCNode&>(*m_node);
    unsigned    outputIdx = m_node->getOutputIndexOfTensor(output);
    return tpcNode.isOutputTensorRmw(outputIdx,
                                     deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType()));
}