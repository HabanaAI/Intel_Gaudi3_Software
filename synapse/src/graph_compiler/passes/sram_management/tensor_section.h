#pragma once

#include <map>
#include "bundle.h"
#include "tensor_slicer.h"

/**
 * Tensor section contain a real tensor which accessed by multiple sliced nodes
 * Each node can be consumer or producer of part of the real tensor
 * The tensor section gather all slice access and finally add nodes to attach itself
 * to the graph - reduction / concat / split / DMA
 */
class TensorSection
{
public:
    explicit TensorSection(const HalReader&                        halReader,
                           const Bundle::Solution::pSlicedOperand& slicedOperand,
                           uint32_t                                bundleIdx,
                           BundleType                              bundleType,
                           const Settable<uint64_t>&               multiBufferId,
                           uint64_t                                sectionIdx);

    /**
     * Add producer or consumer slice
     *
     * @param coord IN Coordinate of the slice in the real tensor
     * @param opIdx IN The node idx / operation idx the slice came from (produced or consumed) node
     *
     * @return The new created sliced Tensor
     */
    pTensor addProduceSlice(const CoordArray& coord, uint32_t opIdx);
    pTensor addConsumeSlice(const CoordArray& coord, uint32_t opIdx);

    void generateGraphSection(HabanaGraph& graph, bool shouldEvictTensor);

    uint64_t getSectionIdx() const { return m_sectionIdx; }

private:

    class SliceInfo
    {
    public:
        SliceInfo(const pTensor& slice, uint32_t opIdx);

        bool operator<(const SliceInfo& rhs) const;

        pTensor m_slice;
        uint32_t m_opIdx = 0;
    };

    using AxisBaseCoordAndValue = std::pair<CoordArray, std::set<uint32_t>>;
    using AxisBaseCoordToValue = std::map<CoordArray, std::set<uint32_t>>;
    using CoordToSliceInfo = std::map<CoordArray, SliceInfo>;

    void addProducersReduction(HabanaGraph& graph);

    static bool canTensorBeRmwOutput(const HabanaGraph& graph, const TensorPtr& inputTensor);

    NodePtr addReduction(const HabanaGraph& graph, const TensorVector& inputs, const TensorVector& outputs) const;

    void handlePassThroughNodes(HabanaGraph& graph);

    void addProducersMemcpyNodes(HabanaGraph& graph);

    /**
     * Create concat or split nodes
     */
    void addProducersConcat(HabanaGraph& graph);

    void eliminateSingleProducer(HabanaGraph& graph);

    void addConsumersMemcpyNodes(HabanaGraph& graph);

    void addConsumersSplit(HabanaGraph& graph);

    /**
     * Replace consumers of tensor with the original tensor
     */
    void replaceConsumersTensor(HabanaGraph& graph, const pTensor& tensor);

    void addTensorViewNodeForSlices(HabanaGraph& graph, CoordToSliceInfo& coordToSlice, bool realTensorIsInput);

    void addAggregateSlicesNodes(HabanaGraph& graph, CoordToSliceInfo& coordToSlice, bool concat);

    void createAggSlicesNodesForAxis(HabanaGraph& graph,
                                     const AxisBaseCoordToValue& axisConcatCoords,
                                     uint32_t axis,
                                     CoordToSliceInfo& coordToSlice,
                                     bool concat);

    /**
     * Get all axis indices by base coordinate
     * Return true if found more then one axis indices on single base coordinate
     */
    static bool fillAxisByBaseCoord(const CoordToSliceInfo& coordToSlice, uint32_t axis, AxisBaseCoordToValue& baseToAxisIdx);

    /**
     * Gather tensors for all coordinates constructed from base coord and axis indices
     * Remove the slices from coordToSlice
     * Return the minimal operation index used
     */
    static uint32_t gatherSlicesForAxisBaseCoord(uint32_t axis,
                                                 const AxisBaseCoordAndValue& baseAndAxisIdx,
                                                 CoordToSliceInfo& coordToSlice,
                                                 TensorVector& outTensors);

    static pTensor getAggregatedTensor(const TensorVector& slices, uint32_t axis);

    /**
     * Return the aggregated size from all tensors for axis
     */
    static uint32_t getAxisAggSize(const TensorVector& tensors, uint32_t axis);

    /**
     * Add memcpy node to graph
     * If hbm tensor is not provided (nullptr) create clone from slice
     * Return the HBM tensor
     */
    pTensor addMemcpyNode(HabanaGraph& graph, const CoordArray& coord, const SliceInfo& slice, pTensor hbmTensor, bool hbmToSram);

    std::string generateNodeName(const std::string& operation) const;

    std::string getTensorName(const CoordArray& coord, const std::string& operation) const;

    const HalReader& m_halReader;
    pTensor m_origTensor;
    TensorSlicer m_tensorSlicer;
    const uint32_t m_bundleIdx;
    BundleType m_bundleType;
    const uint64_t   m_sectionIdx;

    std::multimap<CoordArray, SliceInfo> m_producers;
    // Producers left for concat after reduction step
    CoordToSliceInfo m_concatProducers;

    // To manage intermediate reduction,
    // create reduction slices when consumer has multiple producers
    // save the reduction output here.
    CoordToSliceInfo m_reductionOutputs;

    std::map<CoordArray, std::set<SliceInfo>> m_consumers;
    CoordToSliceInfo m_splitConsumers;

    // For node name extension
    mutable std::map<std::string, uint32_t> m_nameIdxGenerator;
};
