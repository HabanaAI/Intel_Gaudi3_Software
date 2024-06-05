#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/node_info_defs.h"
#include "node_info/tensor_info.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// std includes
#include <bitset>
#include <cstddef>
#include <optional>
#include <utility>

class Node;

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNode
///////////////////////////////////////////////////////////////////////////////////////////////////

// Representation of Eager node
class EagerNode final
{
public:
    EagerNode() = default;
    EagerNode(const EagerNode& node)
    : m_engineType(node.m_engineType), m_skipSuggestedManipulation(node.m_skipSuggestedManipulation), m_ptr(node.m_ptr)
    {
    }
    EagerNode(EagerNode&& node)
    : m_engineType(node.m_engineType),
      m_skipSuggestedManipulation(node.m_skipSuggestedManipulation),
      m_ptr(std::move(node.m_ptr))
    {
    }
    EagerNode(const NodePtr& node) : m_engineType(calcEngineType(node)), m_ptr(node) {}
    EagerNode(NodePtr&& node) : m_engineType(calcEngineType(node)), m_ptr(std::move(node)) {}
    ~EagerNode() = default;

    template<class T = Node>
    T* get() const
    {
        return static_cast<T*>(m_ptr.get());
    }

    template<class T = Node>
    auto getSafePtr() const
    {
        return std::static_pointer_cast<T>(m_ptr);
    }

    operator const NodePtr&() const { return m_ptr; }
    operator Node&() { return *get(); }
    operator const Node&() const { return *get(); }

    EagerNode& operator=(const EagerNode& node)
    {
        m_engineType = node.m_engineType;
        m_skipSuggestedManipulation = node.m_skipSuggestedManipulation;
        m_ptr        = node.m_ptr;
        return *this;
    }
    EagerNode& operator=(const NodePtr& node)
    {
        m_engineType = calcEngineType(node);
        m_ptr        = node;
        return *this;
    }

    bool operator==(const EagerNode& node) const { return m_ptr == node.m_ptr; }
    bool operator==(const NodePtr& node) const { return m_ptr == node; }
    bool operator==(const Node* node) const { return get() == node; }
    bool operator!=(const EagerNode& node) const { return m_ptr != node.m_ptr; }
    bool operator!=(const NodePtr& node) const { return m_ptr != node; }
    bool operator!=(const Node* node) const { return get() != node; }

    Node*       operator->() { return get(); }
    const Node* operator->() const { return get(); }

    void invalidate() { m_invalidated = true; }
    bool isInvalidated() const { return m_invalidated; }

    void setSuggestedManipulationNotRequired() { m_skipSuggestedManipulation = true; }
    bool isSuggestedManipulationRequired() const { return !m_skipSuggestedManipulation; }

    void swap(EagerNode& other) noexcept
    {
        std::swap(m_engineType, other.m_engineType);
        std::swap(m_skipSuggestedManipulation, other.m_skipSuggestedManipulation);
        std::swap(m_ptr, other.m_ptr);
    }
    EngineType getEngineType() const { return m_engineType; }

private:
    static EngineType calcEngineType(const NodePtr& node);

private:
    EngineType m_engineType = EngineType::INVALID;
    bool       m_invalidated = false;
    bool       m_skipSuggestedManipulation = false;
    NodePtr    m_ptr;
};

using EagerNodesVec = VecNodes<EagerNode>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNodes
///////////////////////////////////////////////////////////////////////////////////////////////////

// Vector of nodes that fits eager static memory allocation policy
class EagerNodes
{
public:
    EagerNodes(bool isOriginalGraph, uint64_t tensorSizeThresholdForParallelExecution);
    EagerNodes(const EagerNodes& other);
    virtual ~EagerNodes() = default;

    bool                   isAddingNewNodesEnabled() const { return m_isAddingNewNodesEnabled; }
    NodesNrType            getPhysicalNodesNr() const { return m_physicalNodesNr; }
    const EagerTensorsSet& getTensors() const { return m_tensors; }
    bool                   areMultipleEnginesUsed() const { return m_usedEnginesScoreBoard.count() >= 2; }

    bool                          empty() const { return m_nodes.empty(); }
    NodesNrType                   size() const { return m_nodes.size(); }
    void                          resize(NodesNrType size) { m_nodes.resize(size); }
    EagerNodesVec::iterator       begin() { return m_nodes.begin(); }
    EagerNodesVec::const_iterator begin() const { return m_nodes.begin(); }
    EagerNodesVec::iterator       end() { return m_nodes.end(); }
    EagerNodesVec::const_iterator end() const { return m_nodes.end(); }
    EagerNode&                    front() { return m_nodes.front(); }
    const EagerNode&              front() const { return m_nodes.front(); }
    EagerNode&                    back() { return m_nodes.back(); }
    const EagerNode&              back() const { return m_nodes.back(); }
    EagerNode*                    data() { return m_nodes.data(); }
    const EagerNode*              data() const { return m_nodes.data(); }
    EagerNode&                    operator[](NodesNrType idx) { return m_nodes[idx]; }
    const EagerNode&              operator[](NodesNrType idx) const { return m_nodes[idx]; }
    void                          push_back(const EagerNode& node);
    void                          push_back(EagerNode&& node);
    template<typename... ArgTypes>
    void             emplace_back(ArgTypes&&... args);
    const EagerNode* findNodeByID(synNodeId nodeID) const;

    // Search bwd from a node at nodeIdx for the producer of one of its input tensors
    std::optional<size_t> getInputProducerIdx(const Tensor* tensor, size_t nodeIdx) const;
    std::optional<size_t> getNextConsumerIdx(const Tensor* tensor, size_t startFrom = 0) const;
    bool                  hasSingleConsumer(const Tensor* tensor, size_t startFrom = 0) const;

    void injectNode(EagerNode& node) { handleNewNode(node); }

private:
    void handleNewNode(const EagerNode& node);
    void fillTensorVectorForDuplicatedNode(const EagerTensorsSetBuilder& srcGraphTensors,
                                           const TensorVector&           src,
                                           TensorVector&                 dst);

protected:
    const bool m_isOriginalGraph;  // Flag to distinguish between original graph (false) or the the one with
                                   // internal nodes (true) - AKA final eager graph
    bool        m_isAddingNewNodesEnabled = true;  // Allow adding new nodes
    NodesNrType m_physicalNodesNr         = 0;     // Number of physical nodes
    // Approximate info of which engines are utilized. It used as a hint to disable parallel execution
    std::bitset<static_cast<unsigned>(EngineType::ENGINES_NR)> m_usedEnginesScoreBoard;
    EagerNodesVec                                              m_nodes;    // Vector of nodes
    EagerTensorsSetBuilder                                     m_tensors;  // Information about all unique tensors
};

template<typename... ArgTypes>
void EagerNodes::emplace_back(ArgTypes&&... args)
{
    EAGER_ASSERT(m_isAddingNewNodesEnabled, "Adding new nodes to eager nodes list is not allowed");
    m_nodes.emplace_back(std::forward<ArgTypes>(args)...);
    handleNewNode(m_nodes.back());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerNodesBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////

class EagerNodesBuilder final : public EagerNodes
{
public:
    EagerNodesBuilder(bool isOriginalGraph, uint64_t tensorSizeThresholdForParallelExecution);
    EagerNodesBuilder(const EagerNodesBuilder& other) : EagerNodes(other) {}
    void                          disableAddingNewNodes();
    EagerTensorsSetBuilder&       getTensors() { return m_tensors; }
    const EagerTensorsSetBuilder& getTensors() const { return m_tensors; }
    bool                          isParallelExecPossible();
};

}  // namespace eager_mode