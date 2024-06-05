#include "tpc_desc_base.h"

// eager includes (relative to src/eager/lib/)
#include "utils/numeric_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/tpc_node.h"

void splitTpcDims(HabanaGraph& g, TPCNode& tpcNode);

namespace eager_mode
{
void TpcDescGeneratorBase::splitTpcNodeDims()
{
    NodeAnnotation& ann      = getNode().getNodeAnnotation();
    const auto&     instance = getNode().getInstance();
    unsigned        nDims    = instance.indexSpaceRank;
    ann.tpcSplitDims.reserve(nDims);
    // split from outermost to innermost
    for (int dim = nDims - 1; dim >= 0; --dim)
    {
        ann.tpcSplitDims.push_back(dim);
    }
}

bool TpcDescGeneratorBase::generateDesc()
{
    EAGER_ASSERT(getNode().isInstantiated(), "Node should've been instantiated at displacement etc.");

    // Passes
    m_rois.emplace_back(getNode().generateRoi());
    splitTpcNodeDims();

    if (!generateTpcDesc())
    {
        return false;
    }

    EAGER_ASSERT(m_rois.size() == 1, "Unsupported multiple TPC ROIs");
    EAGER_ASSERT(m_descNr == 1, "Unsupported multiple TPC descriptors");

    // Calculate info required for recipe creation
    m_activationsNr   = 1;
    m_logicalRoisNr   = 1;
    m_requiredWdCtxNr = 1;

    return true;
}

// Calculate expected number of patch points that associated to tensors for a given TPC node
size_t TpcDescGeneratorBase::calcNumberPatchableTensors(const EagerNode& node)
{
    EAGER_ASSERT(node.getEngineType() == EngineType::TPC, "Wrong flow");
    auto ppQualifier = [&](size_t cnt, const TensorPtr& t) -> const size_t {
        if (t == nullptr) return cnt;
        if (auto dim = t->getDim(); dim > MAX_DIMENSIONS_NUM)
        {
            return cnt + divRoundUp(dim, MAX_DIMENSIONS_NUM);
        }
        return cnt + 1;
    };

    const size_t inputsCnt  = std::accumulate(node->getInputs().begin(), node->getInputs().end(), 0, ppQualifier);
    const size_t outputsCnt = std::accumulate(node->getOutputs().begin(), node->getOutputs().end(), 0, ppQualifier);
    return inputsCnt + outputsCnt;
}

}  // namespace eager_mode
