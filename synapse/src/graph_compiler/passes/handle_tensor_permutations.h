#pragma once

#include "tensor.h"
#include "types.h"
#include "layout.h"

class HabanaGraph;
class TransposeNode;
struct TransposeNodeParams;

class TensorPermutationHandler
{
public:
    virtual ~TensorPermutationHandler() {};

    // disallow copy/move construction/assignment
    TensorPermutationHandler(const TensorPermutationHandler&) = delete;
    TensorPermutationHandler(TensorPermutationHandler&&)      = delete;
    TensorPermutationHandler operator=(const TensorPermutationHandler&) = delete;
    TensorPermutationHandler operator=(TensorPermutationHandler&&) = delete;

protected:
    TensorPermutationHandler() = default;  // disallow construction
    bool    isGradientBucketView(const NodePtr& node) const;
    NodePtr getAsLogicalTranspose(const TensorPtr&                 input,
                                  const TensorPtr&                 output,
                                  const TransposePermutationArray& permutation,
                                  std::string_view                 name,
                                  bool                             stridedOutput) const;
    void    permuteTensor(const TensorPtr& t, gc::Permutation& permutation) const;

    bool    isTensorCandidate(const TensorPtr& t) const;

    /**
     * @brief Insert a physical and logical transpose sequence
     *        such that the consumer/producer of the permuted tensor reads/writes from/to
     *        dense memory, without applying changes to the graph.
     *        changes to the graph are applied by the caller.
     *
     * @param permutedTensor
     * @param adjacentNode Permuted tensor's consumer/producer node
     * @param isAdjacentProducer Whether adjacent node is a producer or a consumer
     * @return A tuple consisting of the newly created logical transpose, physical transpose and permuted tensor.
     */
    std::tuple<NodePtr, NodePtr, TensorPtr>
    getTransposeSequence(const TensorPtr& permutedTensor, const NodePtr& adjacentNode, bool isAdjacentProducer) const;

    /**
     * @param node Input transpose node
     * @return True if input transpose node can be modified otherwise false.
     */
    bool isNodeModifiable(bool isInternalTranspose) const;

    /**
     * @param tensorPermutation
     * @param transposePermutation
     * @param inversePermutation
     * @return True if tensor permutation and transpose permutation (or inverse perm) are equal otherwise false.
     */
    bool isSamePermutation(const gc::Permutation& tensorPermutation,
                           const gc::Permutation& transposePermutation,
                           bool                   inversePermutation) const;
};

class GraphModeTensorPermutationHandler : public TensorPermutationHandler
{
public:
    GraphModeTensorPermutationHandler(HabanaGraph& g) : m_graph(g) {}
    ~GraphModeTensorPermutationHandler() override = default;

    // disallow copy/move construction/assignment
    GraphModeTensorPermutationHandler(const GraphModeTensorPermutationHandler&) = delete;
    GraphModeTensorPermutationHandler(TensorPermutationHandler&&)               = delete;
    GraphModeTensorPermutationHandler operator=(const GraphModeTensorPermutationHandler&) = delete;
    GraphModeTensorPermutationHandler operator=(GraphModeTensorPermutationHandler&&) = delete;

    void handlePermutedTensors();

protected:
    /**
     * @param adjacentNode
     * @param optionalPerm
     * @param inversePerm
     * @return  True if adjacent node (to candidate tensor) can be converted to a logical tranpose to handle
     *          the candidate otherwise false.
     */
    bool canConvertAdjacentNodeToLogicTranspose(const NodePtr&                  adjacentNode,
                                                std::optional<gc::Permutation>& optionalPerm,
                                                bool                            inversePerm) const;

    /**
     * @brief Replace input transpose node in graph with the equivalent logical transpose.
     * @param node
     * @param stridedOutput
     */
    void setAsLogicalTranspose(const NodePtr& node, bool stridedOutput);

    /**
     * @brief Insert a physical and logical transpose sequence
     *        such that the consumer/producer of the permuted tensor reads/writes from/to
     *        dense memory.
     *        Example below shows handling of a consumed permuted tensor.
     *
     *
     *                 +--------+
     *   +-------+     |        |
     *   |Perm   +---->+Consumer|
     *   |tensor |     |        |
     *   +-------+     |        |
     *                 +--------+
     *
     *
     *                +-----+                   +-----+                    +--------+
     *  +-------+     |Logic|     +-------+     |Trans|     +--------+     |        |
     *  |Perm   +---->+Trans+---->+interm +---->+pose +---->+consumer+---->+Consumer|
     *  |tensor |     |pose |     |       |     |     |     |input   |     |        |
     *  +-------+     |     |     +-------+     |     |     +--------+     |        |
     *                +-----+                   +-----+                    +--------+
     *
     *
     * @param permutedTensor
     * @param adjacentNode Permuted tensor's consumer/producer node
     * @param isAdjacentProducer Whether adjacent node is a producer or a consumer
     */
    void insertTransposeSequence(const TensorPtr&       permutedTensor,
                                 const NodePtr&         adjacentNode,
                                 bool                   isAdjacentProducer);

    /**
     * @brief short-cut the sequence of opposite transpose nodes create here:
     *
     *   +--------+  [X]    +-----+    [t]    +-----+ [X']  +--------+
     *   |Producer+-------->+  T  +---------->+ T'  +------>+Consumer|
     *   +--------+         +-----+           +-----+       +--------+
     *
     *   Will turn into:
     *
     *   +--------+  [X]    +-----+    [t]
     *   |Producer+-------->+  T  +------->
     *   +--------+  |      +-----+
     *               |
     *               |   +--------+
     *               +-->+Consumer|
     *                   +--------+
     *   where [X] is the permuted version of [t]
     */
    void shortcutTransposeSequence(const TensorPtr& t);

    void handleTensorPermutation(const TensorPtr& t);

    HabanaGraph& m_graph;
};

class EagerModeTransposeTensorPermutationHandler : public TensorPermutationHandler
{
public:
    EagerModeTransposeTensorPermutationHandler(const TransposeNodeParams& nodeParams) : m_nodeParams(nodeParams) {}
    ~EagerModeTransposeTensorPermutationHandler() override = default;

    // disallow copy/move construction/assignment
    EagerModeTransposeTensorPermutationHandler(const EagerModeTransposeTensorPermutationHandler&) = delete;
    EagerModeTransposeTensorPermutationHandler(EagerModeTransposeTensorPermutationHandler&&)      = delete;
    EagerModeTransposeTensorPermutationHandler operator=(const EagerModeTransposeTensorPermutationHandler&) = delete;
    EagerModeTransposeTensorPermutationHandler operator=(EagerModeTransposeTensorPermutationHandler&&) = delete;

    bool    canExtract() const;
    NodePtr extract() const;

protected:
    const TransposeNodeParams& m_nodeParams;
    mutable bool   m_permuteInputTensor = false;
};

class EagerModeTensorPermutationHandler : public TensorPermutationHandler
{
public:
    EagerModeTensorPermutationHandler(const NodePtr& node) : m_node(node) {}
    ~EagerModeTensorPermutationHandler() override = default;

    // disallow copy/move construction/assignment
    EagerModeTensorPermutationHandler(const EagerModeTensorPermutationHandler&) = delete;
    EagerModeTensorPermutationHandler(EagerModeTensorPermutationHandler&&)      = delete;
    EagerModeTensorPermutationHandler operator=(const EagerModeTensorPermutationHandler&) = delete;
    EagerModeTensorPermutationHandler operator=(EagerModeTensorPermutationHandler&&) = delete;

    bool       canExtract() const;
    NodeVector extract() const;

protected:
    const NodePtr&       m_node;
    mutable TensorVector m_premutedInputTensors;
    mutable TensorVector m_premutedOutputTensors;
};