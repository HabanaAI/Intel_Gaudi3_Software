#pragma once

#include "node_visitor.h"
#include "habana_graph.h"

using Axes = llvm_vecsmall::SmallVector<unsigned, SYN_MAX_TENSOR_DIM>;

enum transposeWrapStatus
{
    transposeWrapSuccess  = 0,
    transposeWrapPostpone = 1,
    transposeWrapFail     = 2,
};

class NodeLayoutsHandler : public NodeVisitor
{
public:
    NodeLayoutsHandler(HabanaGraph&       graph,
                       const NodePtr&     node,
                       gc::Permutation&   permutation,
                       bool               wrapInput,
                       PermutationVector& inputsPermutations,
                       PermutationVector& outputsPermutations,
                       bool               isEagerMode = false);

    virtual void visit(Node* node);

    // TPC Node
    virtual void visit(TPCNode* node);

    // Multi nodes
    virtual void visit(TransposeNode* node);
    virtual void visit(StridedViewNode* node);
    virtual void visit(StridedInsertNode* node);

    // MME nodes
    virtual void visit(GEMMNode* node);

    // Logical nodes
    virtual void visit(LogicalOpNode* node);
    virtual void visit(AggregationNode* node);
    virtual void visit(IdentityNode* node);
    virtual void visit(ReshapeNode* node);
    virtual void visit(SqueezeNode* node);

    transposeWrapStatus shouldWrap();

private:
    // squeeze util functions
    bool deduceSqueezeAxes(Axes& squeezedAxes);
    void fillForSqueezeNode(Axes& squeezedAxes);

    // main functions
    bool shouldSkip();
    void expandDimsForBroadcast();
    void insertTensorsPermutation(bool isInput);
    void fillTensorsPermutations();

    void insertReshape(const TensorPtr& tensorToReshape, unsigned toDims);

    // inputs
    HabanaGraph&         m_graph;
    const NodePtr&       m_node;
    const NodeIOManager& m_io;
    gc::Permutation      m_inputPerm;
    gc::Permutation      m_outputPerm;
    bool                 m_isInput;
    // outputs
    PermutationVector&   m_inputsPermutations;
    PermutationVector&   m_outputsPermutations;
    // internal usage
    bool                 m_postpone = false;
    bool                 m_isEagerMode;
};
