#include "perforator.h"
#include "reduced_dims_detector.h"

using namespace gc::layered_brain;

void Perforator::selectPerforationForStrategy(const StrategyPtr& strategy) const
{
    SET_TEMP_LOG_CONTEXT("Perforator");
    LOG_TRACE(LB_SLICER, "Set perforation for strategy {}, num available DCOREs = {}", strategy->index(), m_numDcores);

    ReducedBVDsPerNode reducedBVDsPerNode;
    for (const auto& n : m_bundleNodes)
    {
        if (HabanaGraph::runsOnTPC(n))
        {
            const auto& reducedNodeDims = ReducedDimsDetector(n).getReducedNodeDims();
            reducedBVDsPerNode[n]       = m_bundleViews->getBvdsForNodeDims(n, reducedNodeDims);
        }
    }

    const auto& candidates = m_bvdCandidatesFinder.findPerforationCandidates(strategy, reducedBVDsPerNode);
    strategy->setPerforationData(m_bvdSelector.selectPerforationPerNode(candidates));
}