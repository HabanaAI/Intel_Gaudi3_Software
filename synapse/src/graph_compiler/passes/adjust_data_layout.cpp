#include "habana_pass.h"
#include "habana_graph.h"
#include "node_io_manager.h"

#include "adjust_data_layout.h"
#include "graph_traits.h"

using namespace gc;

DataLayoutHandler::DataLayoutHandler(const HabanaGraph& g, const NodePtr& node) : m_node(node)
{
    auto& ioManager = m_node->getNodeIOManager();
    m_skipLogic     = ioManager.nodeisAllDontCare();
    if (g.getCompilationMode() == CompilationMode::Eager && !m_skipLogic)
    {
        ioManager.setSupportedIOLayouts(g.getDeviceType());
    }
}

bool DataLayoutHandler::validate() const
{
    const auto& ioManager = m_node->getNodeIOManager();
    return m_skipLogic || ioManager.validateLayouts();
}

bool DataLayoutHandler::canExtract() const
{
    const auto& ioManager = m_node->getNodeIOManager();
    return !m_skipLogic && ioManager.permutationsRequired();
}

const TransposeNodeParamsVector& DataLayoutHandler::extract(HabanaGraph& g)
{
    const auto&              ioManager = m_node->getNodeIOManager();
    PermutationVector        inputPermutations;
    PermutationVector        outputPermutations;
    ioManager.permute(inputPermutations, outputPermutations);
    m_transposeInserter.emplace(m_node, inputPermutations, outputPermutations);
    return m_transposeInserter->extract(g);
}

bool DataLayoutHandler::extractAndReplace(HabanaGraph& g) const
{
    auto&             ioManager = m_node->getNodeIOManager();
    PermutationVector inputPermutations;
    PermutationVector outputPermutations;
    ioManager.permute(inputPermutations, outputPermutations);
    TransposeInserter transposeInserter(m_node, inputPermutations, outputPermutations);
    if (!transposeInserter.InsertTransposesForNodeIO(g))
    {
        return false;
    }
    ioManager.markAdjusted();
    return true;
}

bool adjustDataLayout(HabanaGraph& g)
{
    NodeVector nodes = g.getExeSortedNodes();
    for (pNode node : nodes)
    {
        DataLayoutHandler dataLayoutHandler(g, node);
        if (!dataLayoutHandler.validate())
        {
            LOG_ERR(DATA_LAYOUT, "Can't permute layouts for node {}, validation failed", node->getNodeName());
            return false;
        }

        if (!dataLayoutHandler.canExtract())
        {
            LOG_DEBUG(DATA_LAYOUT, "Can't permute layouts for node {}, cannot extract layout", node->getNodeName());
            continue;
        }

        if (!dataLayoutHandler.extractAndReplace(g))
        {
            LOG_ERR(DATA_LAYOUT, "Can't permute layouts for node {}, transpose insertion failed", node->getNodeName());
            return false;
        }
    }
    return true;
}
