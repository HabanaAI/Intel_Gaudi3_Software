#pragma once

#include <memory>
#include <optional>
#include "habana_graph.h"

#include "tpc_node.h"
#include "defs.h"

using SizeArrayVec         = llvm_vecsmall::SmallVector<SizeArray, MAX_NUM_DCORES>;
using OffsetArrayVec       = llvm_vecsmall::SmallVector<OffsetArray, MAX_NUM_DCORES>;
using TpcWdCtxVec          = std::vector<TpcWdCtx>;

class workDistributionManager
{
public:
    workDistributionManager(const HabanaGraph& graph, const NodePtr& node) : m_graph(graph), m_node(node) {}

    void run(std::array<unsigned, MAX_NUM_DCORES>& tpcShuffleIndex, bool& previousTpcNodeLocalityMode);

    static void tpcWorkDistribution(TPCNode&                              tpcNode,
                                    std::list<NodeROI>&                   logicalRois,
                                    uint32_t                              numTpcEngs,
                                    std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex,
                                    bool&                                 previousNodeLocalityMode,
                                    bool                                  fcdFirst  = 0,
                                    bool                                  eagerMode = false,
                                    unsigned                              numDcores = 1);

    static void resetShuffleIndex(std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex);

private:
    enum distributionMethod
    {
        gcdMethod,
        naiveMethod
    };

    using DistributionMethodVec = std::vector<distributionMethod>;

    // returns the utilization and box size for given size
    static UtilizationParams calculateBoxSize(const SizeArray&              gridSize,
                                             SizeArray&                     boxSize,
                                             uint32_t                       numTpcEngs,
                                             const DimVector&               dimPreference,
                                             const std::optional<unsigned>& mandatoryFirstSplitDim,
                                             bool                           eagerMode);

    static UtilizationParams calculateBoxSizeGcdAndNaive(const SizeArray&               gridSize,
                                                         float                          accumGrid,
                                                         SizeArray&                     boxSize,
                                                         uint32_t                       numTpcEngs,
                                                         const DistributionMethodVec&   methods,
                                                         const char*                    methodStr,
                                                         const DimVector&               dimPreference,
                                                         const std::optional<unsigned>& mandatoryFirstSplitDim);

    static UtilizationParams calculateEmptyGrid(const SizeArray& gridSize, SizeArray& boxSize);

    static void fillWdCtx(TpcWdCtxVec&                          tpcWdCtxs,
                          const OffsetArrayVec&                 baseOffset,
                          const SizeArrayVec&                   gridSize,
                          const SizeArrayVec&                   boxSize,
                          uint32_t                              numTpcEngs,
                          const UtilizationParamsVec&           utilization,
                          std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex);

    void tpcWorkDistribution(std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex,
                             bool&                                 previousNodeLocalityMode,
                             bool                                  fcdFirst = false);

    static UtilizationParams calculateUtilization(float            accumGrid,
                                                  const SizeArray& boxSize,
                                                  unsigned         totalNumWorkingEngines,
                                                  uint32_t         numTpcEngs);

    static void validateDcoreRoi(NodeROI& roi, unsigned numDcores);

    const HabanaGraph&     m_graph;
    NodePtr                m_node;
    static constexpr float utilizationThreshold = 1.000;
};