#pragma once

#include "node.h"
#include "bundle.h"

class HabanaGraph;

/**
 * The purpose of this class is to change the memcpy nodes
 * operation index so they will be schedule in a way it most likely
 * won't delay the compute nodes
 *
 * Receive a slice bundle graph, and change the operation index
 * to all nodes according to buffer level
 */
class BundleMemcpyScheduler
{
    using Operation = Bundle::Solution::Operation;
public:
    BundleMemcpyScheduler();

    /**
     * Register operation index to use of buffers
     */
    void addOperationBuffer(const pSliceReference& sliceRef,
                            uint64_t               opIdx,
                            bool                   isInput,
                            uint64_t               sectionIdx,
                            uint32_t               tensorIdx);

    void scheduleBundleGraph(HabanaGraph& graph);

private:
    using BufferId = size_t;

    struct BufferPlacement
    {
        uint64_t operationIndex;
        uint32_t tensorIndex;
        bool isInput;
        bool operator<(const BufferPlacement& o) const;
    };

    struct TensorBuffer
    {
        BufferId bufferId;
        uint32_t bufferLevel;
        bool operator<(const TensorBuffer& o) const;
    };

    void setOperationIndices(const NodeVector& nodes);

    static bool isComputeNode(const NodePtr& node);

    static bool areAllSameEngine(const NodeVector& nodes);

    void printBundleExecutionOrder(HabanaGraph& graph) const;

    void printNodeDependencies(const HabanaGraph& graph, const std::map<TensorBuffer, TensorList>& bufferToTensors) const;

    void createBufferDepedencies(const HabanaGraph& graph, uint32_t opIndex, const TensorVector& tensors, bool isInput,
                                 const TensorMap& realTensorMapping,
                                 std::map<TensorBuffer, TensorList>& bufferToTensors) const;

    void createNodeDependencies(const HabanaGraph& graph);

    NodeList scheduleNodes(const HabanaGraph& graph);

    static TensorMap createRealTensorMapping(const HabanaGraph& graph);

    std::map<BufferPlacement, TensorBuffer> m_bufferMapping; // mapping from opIdx,tensorIdx -> bufferId, bufferLevel
    std::map<NodePtr, NodeSet> m_nodeProducers;   // keep new graph dependencies
    std::map<NodePtr, NodeSet> m_nodeConsumers;
    uint32_t m_opIdx;
};
