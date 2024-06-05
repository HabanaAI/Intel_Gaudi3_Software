#pragma once

// Base classes
class Node;
class MmeNode;
class LogicalOpNode;
class MultiNode;

// TPC nodes
class TPCNode;
class AddNode;
class ReLUNode;
class ReverseNode;
class TPCMemcpyNode;
class EmbeddingNode;
class MaxPoolNode;
class TPCSlice;

// MME nodes
class ConvolutionNode;
class MmeTransposeNode;
class GEMMNode;
class GEMMDeToDwNode;
class GEMMDeToDxNode;
class BatchGemmNode;
class BatchGemmDeToDwNode;
class BatchGemmDeToDxNode;
class DeToDwNode;
class DeToDxNode;

// Logical nodes
class FlattenNode;
class SplitNode;
class ExpandDimsNode;
class SliceAxisNode;
class ReshapeNode;
class LoweringNode;
class PackingNode;
class LogicalBroadcastNode;
class ConcatenateNode;
class TensorViewNode;
class LogicalTransposeNode;
class ReductionNode;
class IdentityNode;
class SqueezeNode;
class AggregationNode;

// DMA nodes
class DMANode;

// Rotator nodes
class RotateNode;

// Multi nodes
class TransposeNode;
class EinsumNode;
class BroadcastNode;
class StridedViewNode;
class StridedInsertNode;

// Semantic nodes
class MemcpyNode;
class MemsetNode;

// Debug nodes
class DebugNodeBase;

/**
 * Base class to analyze and modify nodes
 *
 * Each derived visitor should implement the visit function
 * only for the nodes it wants to handle
 */
class NodeVisitor
{
public:
    virtual ~NodeVisitor() {}

    // All node needs to be handled by the visitor

    // Base classes
    virtual void visit(Node* node);
    virtual void visit(MmeNode* node);
    virtual void visit(MultiNode* node);
    virtual void visit(LogicalOpNode* node);

    // TPC nodes
    virtual void visit(TPCNode* node);
    virtual void visit(TPCMemcpyNode* node);
    virtual void visit(MaxPoolNode* node);
    virtual void visit(TPCSlice* node);

    // MME nodes
    virtual void visit(ConvolutionNode* node);
    virtual void visit(MmeTransposeNode* node);
    virtual void visit(GEMMNode* node);
    virtual void visit(BatchGemmNode* node);
    virtual void visit(DeToDwNode* node);
    virtual void visit(DeToDxNode* node);

    // Logical Nodes
    virtual void visit(FlattenNode* node);
    virtual void visit(SplitNode* node);
    virtual void visit(ExpandDimsNode* node);
    virtual void visit(SliceAxisNode* node);
    virtual void visit(ReshapeNode* node);
    virtual void visit(LoweringNode* node);
    virtual void visit(PackingNode* node);
    virtual void visit(LogicalBroadcastNode* node);
    virtual void visit(ConcatenateNode* node);
    virtual void visit(TensorViewNode* node);
    virtual void visit(LogicalTransposeNode* node);
    virtual void visit(ReductionNode* node);
    virtual void visit(IdentityNode* node);
    virtual void visit(SqueezeNode* node);
    virtual void visit(AggregationNode* node);

    // DMA Nodes
    virtual void visit(DMANode* node);
    // Rotate node
    virtual void visit(RotateNode* node);

    // Multi nodes
    virtual void visit(TransposeNode* node);
    virtual void visit(EinsumNode* node);
    virtual void visit(BroadcastNode* node);
    virtual void visit(StridedViewNode* node);
    virtual void visit(StridedInsertNode* node);

    // Semantic nodes
    virtual void visit(MemcpyNode* node);
    virtual void visit(MemsetNode* node);

    // Debug nodes
    virtual void visit(DebugNodeBase* node);
};

#define DEFINE_VISITOR_METHOD \
    virtual void accept(NodeVisitor* visitor) override \
    { \
        visitor->visit(this); \
    }
