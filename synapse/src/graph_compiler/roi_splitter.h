#pragma once
#include <list>
#include "types.h"
#include <queue>
#include "node_roi.h"
class HabanaGraph;

struct PipelineSplitInfo
{
    PipelineSplitInfo(TSize t_chunksToSplit, unsigned t_workingDimensionIdx, const NodeROI& t_nodeToSplit):
            chunksToSplit(t_chunksToSplit), workingDimensionIdx(t_workingDimensionIdx), nodeToSplit(t_nodeToSplit) {}

    TSize    chunksToSplit;
    unsigned workingDimensionIdx;
    NodeROI  nodeToSplit;
};

class ROISplitter
{
public:
    bool splitAllNodes(HabanaGraph& g) const;
    bool splitNode(HabanaGraph& graph, pNode node) const;
    void validateDcoreRoi(pNode node) const;

    virtual std::list<NodeROI> splitBatchGemm(pNode node, HabanaGraph& g)              const;
    virtual std::list<NodeROI> splitTranspose(pNode node, HabanaGraph& g)              const;
    virtual std::list<NodeROI> splitRoiSpatialSize(pNode node, HabanaGraph& g)         const;
    virtual std::list<NodeROI> splitTPC(pNode node, HabanaGraph& g)                    const;
    virtual std::list<NodeROI> splitDMA(pNode node, HabanaGraph& g) const;
    virtual std::list<NodeROI> splitDMA(pNode node, HabanaGraph& g, NodeROI& baseRoi) const;
    virtual std::list<NodeROI> splitRotate(pNode node, HabanaGraph& g)                 const;
    virtual std::list<NodeROI> splitSpecialKernels(NodeROI& roi, unsigned nTPCEngines) const;

    std::list<NodeROI> splitRoisBetweenEngines(std::list<NodeROI>& roisToSplit,
                                               const DimVector&    splitDimsOrder,
                                               unsigned            nEngines,
                                               const std::string&  nodeName,
                                               bool preferSplitOnFcd = true /* for legacy reasons */) const;

    virtual std::list<NodeROI> splitSpecialKernelsBetweenEngines(std::list<NodeROI>& roisToSplit, unsigned nTPCEngines);

    virtual std::list<NodeROI> projectDmaRoisToFullSizes(const Node* node, std::list<NodeROI>& roisToExpand);

    virtual std::list<NodeROI> projectTPCRois(const pNode& node, std::list<NodeROI>&  rois);

    static std::list<NodeROI> splitFullRoiToLogicalRoisAlongExternalAxis(const NodeROI&     roi,
                                                                         const DimVector&   splitDimsOrder,
                                                                         unsigned           nChunks,
                                                                         const std::string& nodeName,
                                                                         unsigned           numOfPhysicalEngs = 0);

    static std::list<NodeROI> splitDMABroadcast(const NodePtr& node, const NodeROI& baseRoi);

protected:
    // split helper functions
    std::list<NodeROI>
    splitTPCPipeline(NodeROI& roi, const NodePtr& n, unsigned nEngines, unsigned pipeDepth, HabanaGraph& g) const;

    static std::list<NodeROI> SplitROIBetweenEngines(NodeROI&         roi,
                                                     const DimVector& splitDimsOrder,
                                                     unsigned         nEngines,
                                                     unsigned&        nextSubSplitStartIndex,
                                                     unsigned&        nextMajorSplitStartIndex,
                                                     bool             preferSplitOnFcd = true);

    static std::vector<TSize> splitSamplesIntoPerfectSplit(TSize numSamples, unsigned pipelineDepth);

    static std::vector<TSize> perfectSplitIfPossible(TSize numSamples, unsigned pipelineDepth);

    static void validateSplitInput(const TSize* roiSize, const DimVector& dimPreference, unsigned nChunks);

    static std::vector<TSize> splitSamplesIntoNonzeroPipeline(TSize    numSamples,
                                                              unsigned pipelineDepth,
                                                              unsigned numOfPhysicalEngs = 0);

    static std::list<NodeROI> translateSplitToNodeROI(NodeROI&                                  roi,
                                                      const DimVector&                          dimPreference,
                                                      const std::vector<std::vector<TSize>>&    chunks);

    static void deployPerfectSplitStrategy(const TSize*                        roiSize,
                                           const DimVector&                    dimPreference,
                                           unsigned&                           chunksLeft,
                                           std::vector<std::vector<TSize>>&    chunks);

    static void deployZeroRemainderStrategy(const TSize*                        roiSize,
                                            const DimVector&                    dimPreference,
                                            unsigned&                           chunksLeft,
                                            unsigned                            nChunks,
                                            std::vector<std::vector<TSize>>&    chunks,
                                            unsigned                            maxChunks);

    static void deployPipelineStrategy(const DimVector&               dimPreference,
                                       std::queue<PipelineSplitInfo>& nodesToSplit,
                                       std::list<NodeROI>&            finalSplit,
                                       unsigned&                      nextSubSplitStartIndex,
                                       unsigned&                      nextMajorSplitStartIndex);

    static std::list<NodeROI> splitAllSamples(NodeROI&         roi,
                                              const DimVector& dimPreference,
                                              unsigned         nChunks,
                                              unsigned         maxChunks,
                                              unsigned&        nextSubSplitStartIndex,
                                              unsigned&        nextMajorSplitStartIndex);

    static std::list<NodeROI> splitDenseDMA(pNode node, HabanaGraph& g, NodeROI& roi);

    static std::list<NodeROI> split2dStridedDMA(pNode node, HabanaGraph& g, NodeROI& baseRoi);

    static unsigned calculateTPCPipelineDepth(const NodePtr& node, const HabanaGraph& g);
};
