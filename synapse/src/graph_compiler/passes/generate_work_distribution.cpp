#include "generate_work_distribution.h"

#include "defs.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "tpc_node.h"
#include "types.h"

#include <algorithm>
#include <bitset>
#include <iterator>

namespace
{
class Permutations
{
public:
    Permutations(const SizeArray& origRoiSize, const DimVector& dimPreference);

    // Cycle (once) through all permutations of non "1"-sized dims.
    // Returns nullptr if we've completed all permutations.
    // Calling after reaching nullptr is unspecified behavior.
    const DimVector* getNextPermutation();

private:
    // m_currentPerm is split so that dims of size "1" are moved to the end and aren't included as part of the
    // permutations being explored. m_splitEnd is the iterator past which we have "1" dims.
    DimVector      m_currentPerm;
    unsigned char* m_splitEnd;

    // We start mid-permutation sequence, need to track where we are and whether we have another permutation.
    // Note: If we're allowed to start from a sorted position, we could avoid holding this and == checks.
    DimVector m_initialPerm;
    bool      m_started = false;
};
}  // anonymous namespace

Permutations::Permutations(const SizeArray& origRoiSize, const DimVector& dimPreference)
{
    m_currentPerm = dimPreference;
    m_splitEnd    = std::stable_partition(m_currentPerm.begin(), m_currentPerm.end(), [&](uint8_t dim) {
        return origRoiSize[dim] != 1;
    });

    m_initialPerm = m_currentPerm;
}

const DimVector* Permutations::getNextPermutation()
{
    // on first call to function we would like to get currentDimPreference and not next
    if (!m_started)
    {
        m_started = true;
        return &m_currentPerm;
    }
    std::next_permutation(m_currentPerm.begin(), m_splitEnd);
    return m_currentPerm != m_initialPerm ? &m_currentPerm : nullptr;
}

void workDistributionManager::tpcWorkDistribution(std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex,
                                                  bool&                                 previousNodeLocalityMode,
                                                  bool                                  fcdFirst)
{
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(m_node);
    HB_ASSERT(tpcNode != nullptr, "{}: tpcNode is null", __FUNCTION__);
    tpcWorkDistribution(*tpcNode,
                        *m_graph.GetNodeROIs(m_node),
                        m_graph.getNumTpcEng(),
                        shuffleIndex,
                        previousNodeLocalityMode,
                        fcdFirst,
                        false,
                        m_graph.getHALReader()->getNumDcores());
}

void workDistributionManager::tpcWorkDistribution(TPCNode&                              tpcNode,
                                                  std::list<NodeROI>&                   logicalRois,
                                                  uint32_t                              numTpcEngs,
                                                  std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex,
                                                  bool&                                 previousNodeLocalityMode,
                                                  bool                                  fcdFirst,
                                                  bool                                  eagerMode,
                                                  unsigned                              numDcores)
{
    bool localityMode = tpcNode.getNodeAnnotation().isPerforated();
    std::optional<unsigned> mandatoryFirstSplitDim;

    if (tpcNode.hasMandatorySplitDim()) // primary priority
    {
        mandatoryFirstSplitDim = tpcNode.getMandatorySplitDim();
    }
    else if (fcdFirst) // secondary priority
    {
        mandatoryFirstSplitDim = 0;
    }

    for (NodeROI& roi : logicalRois)
    {
        SizeArrayVec         dcoreGridSize;
        SizeArrayVec         boxSize;
        OffsetArrayVec       baseOffset;

        tpcNode.getNodeAnnotation().tpcMetaData.utilizationPerLogicalRoi.emplace_back();
        UtilizationParamsVec& tUtiliz = tpcNode.getNodeAnnotation().tpcMetaData.utilizationPerLogicalRoi.back();

        if (localityMode)
        {
            HB_ASSERT(logicalRois.size() == 1, "expected only a single ROI in locality mode");
            validateDcoreRoi(roi, numDcores);
            numTpcEngs = numTpcEngs / numDcores;
            for (const auto& dRoi : roi.dcoreROIs)
            {
                dcoreGridSize.emplace_back();
                SizeArray& tSize = dcoreGridSize.back();
                std::copy_n(dRoi.size, MAX_DIMENSIONS_NUM, tSize.begin());
                baseOffset.emplace_back();
                OffsetArray& tBaseOffset = baseOffset.back();
                std::copy_n(dRoi.baseOffset, MAX_DIMENSIONS_NUM, tBaseOffset.begin());
                boxSize.emplace_back();
                SizeArray& tBoxSize = boxSize.back();
                tUtiliz.push_back(calculateBoxSize(tSize,
                                                   tBoxSize,
                                                   numTpcEngs,
                                                   tpcNode.getNodeAnnotation().tpcSplitDims,
                                                   mandatoryFirstSplitDim,
                                                   eagerMode));
            }
        }
        else
        {
            dcoreGridSize.emplace_back();
            SizeArray& tSize = dcoreGridSize.back();
            std::copy_n(roi.size, MAX_DIMENSIONS_NUM, tSize.begin());
            baseOffset.emplace_back();
            OffsetArray& tBaseOffset = baseOffset.back();
            std::copy_n(roi.baseOffset, MAX_DIMENSIONS_NUM, tBaseOffset.begin());
            boxSize.emplace_back();
            SizeArray& tBoxSize = boxSize.back();
            tUtiliz.push_back(calculateBoxSize(tSize,
                                               tBoxSize,
                                               numTpcEngs,
                                               tpcNode.getNodeAnnotation().tpcSplitDims,
                                               mandatoryFirstSplitDim,
                                               eagerMode));
        }
        if (previousNodeLocalityMode != localityMode)
        {
            resetShuffleIndex(shuffleIndex);
        }
        fillWdCtx(roi.tpcWdCtx, baseOffset, dcoreGridSize, boxSize, numTpcEngs, tUtiliz, shuffleIndex);
        previousNodeLocalityMode = localityMode;
    }
}

UtilizationParams workDistributionManager::calculateBoxSize(const SizeArray&              gridSize,
                                                           SizeArray&                     boxSize,
                                                           uint32_t                       numTpcEngs,
                                                           const DimVector&               dimPreference,
                                                           const std::optional<unsigned>& mandatoryFirstSplitDim,
                                                           bool                           eagerMode)
{
    static const DistributionMethodVec GCD_AND_NAIVE_METHODS = {gcdMethod, naiveMethod};
    static const DistributionMethodVec NAIVE_METHOD          = {naiveMethod};
    UtilizationParams                  tUtiliz;
    UtilizationParams                  maxUtiliz;
    const char*                        tMethodStr      = nullptr;
    const char*                        chosenMethodStr = nullptr;
    SizeArray                          tBoxSize        = {0};
    DimVector                          chosenDimPreference;

    // If any of the grid dims is 0 - the job should be empty
    const auto accumGrid =
        std::accumulate(std::begin(gridSize), std::end(gridSize), uint64_t {1}, std::multiplies<uint64_t>());
    if (accumGrid == 0)
    {
        return calculateEmptyGrid(gridSize, boxSize);
    }

    auto singleStep = [&](const DimVector& currentDimPreference) {
        // Trial 1: GCD followed by Naive
        tMethodStr = "GCD+Naive";

        tUtiliz = calculateBoxSizeGcdAndNaive(gridSize,
                                              accumGrid,
                                              tBoxSize,
                                              numTpcEngs,
                                              GCD_AND_NAIVE_METHODS,
                                              tMethodStr,
                                              currentDimPreference,
                                              mandatoryFirstSplitDim);

        if (tUtiliz.totalUtilization > maxUtiliz.totalUtilization)
        {
            maxUtiliz           = tUtiliz;
            chosenMethodStr     = tMethodStr;
            chosenDimPreference = currentDimPreference;
            boxSize             = tBoxSize;
        }

        // go to next trial only if there is room for improvement
        if (tUtiliz.totalUtilization < utilizationThreshold)
        {
            // Trial 2: Naive only
            tMethodStr = "NaiveOnly";

            tUtiliz = calculateBoxSizeGcdAndNaive(gridSize,
                                                  accumGrid,
                                                  tBoxSize,
                                                  numTpcEngs,
                                                  NAIVE_METHOD,
                                                  tMethodStr,
                                                  currentDimPreference,
                                                  mandatoryFirstSplitDim);

            if (tUtiliz.totalUtilization > maxUtiliz.totalUtilization)
            {
                maxUtiliz           = tUtiliz;
                chosenMethodStr     = tMethodStr;
                chosenDimPreference = currentDimPreference;
                boxSize             = tBoxSize;
            }
        }
    };

    if (eagerMode)
    {
        singleStep(dimPreference);  // run only once to reduce compile time
    }
    else
    {
        Permutations perm(gridSize, dimPreference);
        while (tUtiliz.totalUtilization < utilizationThreshold)
        {
            auto* currentDimPreference = perm.getNextPermutation();
            if (!currentDimPreference) break;
            singleStep(*currentDimPreference);
        }
    }

    LOG_DEBUG(ROI_SPLITTER,
              "Chosen method for ROI is {} with {} working engines, engine utilization: {}, total utilization: {}, "
              "dimPreference: [{}], Box sizes: [{}],  gridSize: [{}]",
              chosenMethodStr ? chosenMethodStr : "",
              maxUtiliz.totalNumWorkingEngines,
              maxUtiliz.engineUtilization,
              maxUtiliz.totalUtilization,
              fmt::join(chosenDimPreference, ","),
              fmt::join(boxSize, ","),
              fmt::join(gridSize, ","));
    return maxUtiliz;
}

void workDistributionManager::fillWdCtx(TpcWdCtxVec&                          tpcWdCtxs,
                                        const OffsetArrayVec&                 baseOffset,
                                        const SizeArrayVec&                   gridSize,
                                        const SizeArrayVec&                   boxSize,
                                        uint32_t                              numTpcEngs,
                                        const UtilizationParamsVec&           utilization,
                                        std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex)
{
    size_t i = 0;
    for (const auto& tUtiliz : utilization)
    {
        TpcWdCtx& tTpcWdCtx = tpcWdCtxs.emplace_back();
        std::copy_n(baseOffset[i].begin(), MAX_DIMENSIONS_NUM, tTpcWdCtx.baseCord);
        std::copy_n(gridSize[i].begin(), MAX_DIMENSIONS_NUM, tTpcWdCtx.gridSize);
        std::copy_n(boxSize[i].begin(), MAX_DIMENSIONS_NUM, tTpcWdCtx.boxSize);
        for (size_t j = 0; j < MAX_DIMENSIONS_NUM; j++)
        {
            // zero-sized box => a single nop slice. FW expects dimSlices to be non zero
            tTpcWdCtx.dimSlices[j] = boxSize[i][j] > 0 ? div_round_up(gridSize[i][j], boxSize[i][j]) : 1;
        }
        tTpcWdCtx.shuffleIndex = shuffleIndex[i];
        shuffleIndex[i]        = likely(GCFG_ENABLE_TPC_SHUFFLE_INDEX.value())
                              ? (shuffleIndex[i] + tUtiliz.totalNumWorkingEngines) % numTpcEngs
                              : 0;

        LOG_DEBUG(ROI_SPLITTER,
                  "WdCtx filled Dcore {} with shuffle index: {}, Box sizes: [{}],  Grid size: [{}], Base coordinates: "
                  "[{}], Dim slices: [{}]",
                  i,
                  shuffleIndex[i],
                  fmt::join(tTpcWdCtx.boxSize, ","),
                  fmt::join(tTpcWdCtx.gridSize, ","),
                  fmt::join(tTpcWdCtx.baseCord, ","),
                  fmt::join(tTpcWdCtx.dimSlices, ","));
        ++i;
    }
}

static void shiftValueToStart(DimVector& dimVector, uint8_t value)
{
    auto it = std::find(dimVector.begin(), dimVector.end(), value);
    HB_DEBUG_VALIDATE(it != dimVector.end());
    std::copy_backward(dimVector.begin(), it, std::next(it));
    dimVector[0] = value;
}

UtilizationParams workDistributionManager::calculateBoxSizeGcdAndNaive(
    const SizeArray&               gridSize,
    const float                    accumGrid,
    SizeArray&                     boxSize,
    uint32_t                       numTpcEngs,
    const DistributionMethodVec&   methods,
    const char*                    methodStr,
    const DimVector&               dimPreference,
    const std::optional<unsigned>& mandatoryFirstSplitDim)
{
    DimVector newDimPreference(dimPreference);

    if (mandatoryFirstSplitDim.has_value())
    {
        if (newDimPreference.empty())
        {
            newDimPreference.push_back(mandatoryFirstSplitDim.value());
        }
        else
        {
            LOG_DEBUG(ROI_SPLITTER, "Changing dimPreference: moving dim {} to the beginning", mandatoryFirstSplitDim.value());
            shiftValueToStart(newDimPreference, mandatoryFirstSplitDim.value());
        }
    }

    if (newDimPreference.empty())
    {
        LOG_WARN(ROI_SPLITTER, "dim preference empty, work distributed on one engine only");
    }

    uint32_t numEnginesLeft         = numTpcEngs;  // number of available engines left for distribution
    unsigned totalNumWorkingEngines = 1;  // final number of engines that will actually work on this ROI
    unsigned splitDim               = 0;
    unsigned maxDimension           = newDimPreference.size() > 0 ? newDimPreference[0] : 0;
    unsigned numOfWorkingEngines    = 1;

    // Init boxSize
    for (unsigned i = 0; i < MAX_DIMENSIONS_NUM; i++)
    {
        boxSize[i] = gridSize[i];
    }

    std::bitset<HABANA_DIM_MAX> alreadySplit;

    for (distributionMethod method : methods)
    {
        for (unsigned i = 0; i < newDimPreference.size() && (numEnginesLeft > 1); i++)
        {
            splitDim = newDimPreference[i];

            // avoid splitting on the same dim more than once
            if (alreadySplit[splitDim]) continue;

            unsigned boxSizePrev = boxSize[splitDim];
            numOfWorkingEngines  = 1;

            // In GCD method we divide only by gcd value of dimension and remaining TPCs
            // this way we don't harm any future superior splits
            if (method == gcdMethod)
            {
                const unsigned gcdValue = gcd(numEnginesLeft, (uint32_t)boxSizePrev);
                if (gcdValue > 1)
                {
                    boxSize[splitDim]      = boxSizePrev / gcdValue;
                    numOfWorkingEngines    = gcdValue;
                    alreadySplit[splitDim] = true;  // remember this dim was used if the split succeeded
                }
                else if (boxSizePrev > boxSize[maxDimension])
                {
                    maxDimension = splitDim;  // we don't want to further split on a dimension which we already split
                }
            }
            else  // naive method
            {
                // Split dim size to num of TPC engines to get box size of this dim
                // Use ceiling cause we want to work with remaining size at the end
                boxSize[splitDim]      = boxSizePrev / numEnginesLeft + (boxSizePrev % numEnginesLeft != 0);
                numOfWorkingEngines    = boxSizePrev / boxSize[splitDim] + (boxSizePrev % boxSize[splitDim] != 0);
                alreadySplit[splitDim] = true;  // remember this dim was used
            }

            // To how many TPC engines we can split the rest of the ROI
            numEnginesLeft = numEnginesLeft / numOfWorkingEngines;
            totalNumWorkingEngines *= numOfWorkingEngines;
        }

        if (numEnginesLeft <= 1) break;

        // If the naive method will run next, start its splitting on the max dim hoping it will yield better results
        if (numEnginesLeft < numTpcEngs && !mandatoryFirstSplitDim.has_value())
        {
            LOG_DEBUG(ROI_SPLITTER, "Changing dimPreference before Naive method: moving maxDim to the beginning");
            shiftValueToStart(newDimPreference, maxDimension);
        }
    }

    UtilizationParams utiliz = calculateUtilization(accumGrid, boxSize, totalNumWorkingEngines, numTpcEngs);
    LOG_TRACE(ROI_SPLITTER,
              "Method {}: , ROI size: [{}], Dim split order: [{}], "
              "Box sizes: [{}], "
              "Num of working TPCs: {}, Engine utilization: {}, Total Utilization: {}",
              methodStr,
              fmt::join(gridSize, ","),
              fmt::join(newDimPreference, ","),
              fmt::join(boxSize, ","),
              utiliz.totalNumWorkingEngines,
              utiliz.engineUtilization,
              utiliz.totalUtilization);
    return utiliz;
}

UtilizationParams workDistributionManager::calculateEmptyGrid(const SizeArray& gridSize, SizeArray& boxSize)
{
    UtilizationParams utiliz;  // initalized to 0
    for (unsigned i = 0; i < MAX_DIMENSIONS_NUM; i++)
    {
        boxSize[i] = 0;  // FW is expecting box size 0 to detect nop for the DCore
    }

    LOG_TRACE(ROI_SPLITTER,
              "Empty job sent to Dcore ROI size: [{}], Box sizes:  [{}], ",
              fmt::join(gridSize, ","),
              fmt::join(boxSize, ","));
    return utiliz;
}

UtilizationParams workDistributionManager::calculateUtilization(float            accumGrid,
                                                                const SizeArray& boxSize,
                                                                unsigned         totalNumWorkingEngines,
                                                                uint32_t         numTpcEngs)
{
    const auto accumBox = std::accumulate(std::begin(boxSize), std::end(boxSize), 1, std::multiplies<double>());
    if (unlikely(!accumBox))
    {
        LOG_ERR(ROI_SPLITTER, "Box or Grid Size have 0 size dimension");
    }

    UtilizationParams utiliz;
    utiliz.totalNumWorkingEngines = totalNumWorkingEngines;
    utiliz.engineUtilization      = accumGrid / (float)(totalNumWorkingEngines * accumBox);
    utiliz.totalUtilization  = utiliz.engineUtilization * (float)totalNumWorkingEngines / (float)numTpcEngs;
    return utiliz;
}

void workDistributionManager::resetShuffleIndex(std::array<unsigned, MAX_NUM_DCORES>& shuffleIndex)
{
    std::fill(shuffleIndex.begin(), shuffleIndex.end(), 0);
    LOG_TRACE(ROI_SPLITTER, "reset shuffle index vector to 0");
}

void workDistributionManager::validateDcoreRoi(NodeROI& roi, unsigned numDcores)
{
    HB_ASSERT((roi.dcoreROIs.size() == numDcores) || (roi.dcoreROIs.size() == 0),
              "Dcore Rois can be either full or empty");
    TSize totalDcoresSize = 0;
    for (DcoreROI& droi : roi.dcoreROIs)
    {
        totalDcoresSize += multiplyElements(droi.size, droi.size + MAX_DIMENSIONS_NUM);
    }
    TSize fullRoiSize = multiplyElements(roi.size, roi.size + MAX_DIMENSIONS_NUM);
    HB_ASSERT(totalDcoresSize == fullRoiSize,
              "Dcore Rois ({}) don't add up to full Roi ({})",
              totalDcoresSize,
              fullRoiSize);
}

void workDistributionManager::run(std::array<unsigned, MAX_NUM_DCORES>& tpcShuffleIndex,
                                  bool&                                 previousTpcNodeLocalityMode)
{
    HB_ASSERT(m_graph.GetNodeROIs(m_node) != nullptr, "node has no logical rois");

    if (HabanaGraph::runsOnTPC(m_node))
    {
        LOG_DEBUG(ROI_SPLITTER, "Generate work distribution for node {}", m_node->getNodeName());
        tpcWorkDistribution(tpcShuffleIndex, previousTpcNodeLocalityMode);
    }
    else if (HabanaGraph::runsOnMME(m_node))
    {
        HB_ASSERT(false, "{}: Logic is not implemented yet for MME engine", __FUNCTION__);
    }
    else if (m_node->isDma())
    {
        HB_ASSERT(false, "{}: Logic is not implemented yet for DMA engine", __FUNCTION__);
    }
    else if (m_node->isRotate())
    {
        HB_ASSERT(false, "{}: Logic is not implemented yet for Rotator engine", __FUNCTION__);
    }
}

bool generateWorkDistribution(HabanaGraph& g)
{
    const auto& sortedNodes  = g.getExeSortedNodes();
    std::array<unsigned, MAX_NUM_DCORES> tpcShuffleIndex {};
    bool                                 previousTpcNodeLocalityMode = false;
    for (const NodePtr& n : sortedNodes)
    {
        workDistributionManager wdMngr(g, n);
        // First implementation is for TPC only.
        if (HabanaGraph::runsOnTPC(n))
        {
            wdMngr.run(tpcShuffleIndex, previousTpcNodeLocalityMode);
        }
    }
    return true;
}