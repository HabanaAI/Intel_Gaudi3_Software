#pragma once

#include "bundle.h"
#include "bundle_memcpy_scheduler.h"
#include "habana_graph.h"
#include "tensor_section.h"

class BundleSlicer
{
public:
    static void sliceBundle(const Bundle& bundle, HabanaGraph& graph);

    static bool shouldTensorBeEvicted(const pTensor&     tensor,
                                      const HabanaGraph& fullGraph,
                                      const NodeSet&     bundleNodes);

private:
    using Operation = Bundle::Solution::Operation;
    using SliceOperandComp = Bundle::Solution::SlicedOperand::SliceOperandComp;
    using TensorSectionByOperand = std::map<pSlicedOperand, TensorSection, SliceOperandComp>;

    BundleSlicer(const HabanaGraph& graph, uint32_t bundleIdx, BundleType bundleType);

    void addOperation(const Operation& op);

    template<typename Container>
    void replaceOperandsWithSliceTensors(NodePtr& sliceNode, const Container& sliceReferences, bool isInputsContainer);

    pTensor getSliceTensor(const pSliceReference& sliceRef, bool inputSlice, uint32_t tensorIdx);

    std::unordered_set<TensorPtr> getEvictions() const;
    void addGraphNodes(HabanaGraph& slicedBundleGraph);

    pNode getSliceNode(const pNode& origNode) const;

    void createTempGraphVisualization(HabanaGraph& tempGraph, const std::string& suffix);

    void logOperation(const Operation& op) const;

    void logOperands(const std::vector<pSliceReference>& operands) const;

    uint64_t getNextTensorSectionIdx() { return m_nextTensorSectionIdx++; }

    const HabanaGraph&      m_graph;  // Reference to the full graph
    const uint32_t          m_bundleIdx;
    NodeSet m_bundleNodes;
    TensorSectionByOperand m_sections;
    std::list<pNode> m_sliceNodes;
    uint32_t m_opIdx;
    BundleType              m_bundleType;
    BundleMemcpyScheduler m_memcpyScheduler;
    uint64_t                m_nextTensorSectionIdx = 0;
    std::optional<uint64_t> m_sharedChainMultiBufId;  // there is at most 1 such chain in a bundle
};
