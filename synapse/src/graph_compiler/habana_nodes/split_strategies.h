#pragma once
#include <vector>
#include "node_roi.h"
#include "dma_transpose_engine_params.h"

class Node;
class HabanaGraph;

using NodeROIContainer = std::list<NodeROI>;

// A way to handle logical splitting and physical splitting
class SplitStrategy
{
public:
    SplitStrategy(const DmaTransposeEngineParams&  teParams,
                  const TransposePermutationArray& p,
                  const DimVector&                 prefferedDimensionsOrder)
    : m_teParams(teParams), m_permutation(p), m_prefferedDimensionsOrder(prefferedDimensionsOrder)
    {
    }

    virtual void splitLogical(const Node&       node,
                              const NodeROI&    inputRoi,
                              uint32_t          logicalEngineCount,
                              uint32_t          futurePhysicalEngineCount,
                              NodeROIContainer& outputContainer) = 0;
    // Splits the logical ROI to the ROIs for the engines, so that
    // each ROI could be executed on a single engine with no issues.
    // Returns the ROI that matches the inputs/outputs layout (same strides)
    virtual void splitRoiToEngines(const Node& node, const NodeROI& inputRoi, uint32_t physicalEngineCount, NodeROIContainer& outputContainer) = 0;

    // Converts a single physical ROI to it's execution format in the engine.
    // Finalize, contrary to splitRoiToEngines, can use different strides than the original tensors.
    virtual void finalize(const Node& node, NodeROI& inputRoi /*IN,OUT*/) {}

    // Splits to physical ROIs
    virtual std::list<NodeROI> splitToPhysical(const Node& node, const std::list<NodeROI>& roisToSplit, uint32_t physicalEngineCount);

    static void updateNumSignals(std::list<NodeROI>::iterator start,
                                 std::list<NodeROI>::iterator last,
                                 uint32_t                     physicalEngineCount);

protected:
    DmaTransposeEngineParams  m_teParams;
    TransposePermutationArray m_permutation;
    DimVector                 m_prefferedDimensionsOrder;
};

// Use special calculation to ensure minimal descriptor count of the transpose engine, but can cost utilization.
class SplitTransposeToLowDescriptorCount : public SplitStrategy
{
public:
    SplitTransposeToLowDescriptorCount(const DmaTransposeEngineParams&  teParams,
                                       const TransposePermutationArray& p,
                                       const DimVector&                 prefferedDimensionsOrder)
    : SplitStrategy(teParams, p, prefferedDimensionsOrder)
    {
    }
    void splitLogical(const Node&       node,
                      const NodeROI&    inputRoi,
                      uint32_t          logicalEngineCount,
                      uint32_t          futurePhysicalEngineCount,
                      NodeROIContainer& outputContainer) override;
    void splitRoiToEngines(const Node& node, const NodeROI& inputRoi, uint32_t physicalEngineCount, NodeROIContainer& outputContainer) override;
    uint32_t optimizeNumOfChunksNum(const Node& node, const NodeROI& inputRoi, uint32_t logicalEngineCount);
    void finalize(const Node& node, NodeROI& inputRoi) override;
};

// Split to keep high utilization
class SplitFullyUtilizedTranspose : public SplitStrategy
{
public:
    SplitFullyUtilizedTranspose(const DmaTransposeEngineParams&  teParams,
                                const TransposePermutationArray& p,
                                const DimVector&                 prefferedDimensionsOrder)
    : SplitStrategy(teParams, p, prefferedDimensionsOrder)
    {
    }
    void splitLogical(const Node&       node,
                      const NodeROI&    inputRoi,
                      uint32_t          logicalEngineCount,
                      uint32_t          futurePhysicalEngineCount,
                      NodeROIContainer& outputContainer) override;
    void splitRoiToEngines(const Node& node, const NodeROI& inputRoi, uint32_t physicalEngineCount, NodeROIContainer& outputContainer) override;

    void finalize(const Node& node, NodeROI& inputRoi) override;
};
