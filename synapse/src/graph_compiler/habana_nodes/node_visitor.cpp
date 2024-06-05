#include "convolution_node.h"
#include "concatenate_node.h"
#include "reduction_node.h"
#include "dedx_node.h"
#include "dedw_node.h"
#include "habana_nodes.h"
#include "logical_op_node.h"
#include "mme_node.h"
#include "multi_node.h"
#include "node.h"
#include "tensor_view_node.h"
#include "tpc_node.h"
#include "tpc_slice.h"
#include "transpose_node.h"
#include "memcopy_node.h"
#include "memset_node.h"
#include "einsum_node.h"
#include "broadcast_node.h"
#include "identity_node.h"
#include "squeeze_node.h"
#include "strided_view_node.h"
#include "strided_insert_node.h"

#include "node_visitor.h"

#define VISIT_FUNC_IMPL(CLASS_TYPE) \
    void NodeVisitor::visit(CLASS_TYPE* node) \
    { \
        visit(static_cast<CLASS_TYPE::BaseClass*>(node)); \
    }

void NodeVisitor::visit(Node* node)
{
}

VISIT_FUNC_IMPL(MmeNode)
VISIT_FUNC_IMPL(MultiNode)
VISIT_FUNC_IMPL(LogicalOpNode)

VISIT_FUNC_IMPL(TPCNode)
VISIT_FUNC_IMPL(TPCMemcpyNode)
VISIT_FUNC_IMPL(MaxPoolNode)
VISIT_FUNC_IMPL(TPCSlice)

VISIT_FUNC_IMPL(ConvolutionNode)
VISIT_FUNC_IMPL(MmeTransposeNode)
VISIT_FUNC_IMPL(GEMMNode)
VISIT_FUNC_IMPL(BatchGemmNode)
VISIT_FUNC_IMPL(DeToDwNode)
VISIT_FUNC_IMPL(DeToDxNode)

VISIT_FUNC_IMPL(FlattenNode)
VISIT_FUNC_IMPL(SplitNode)
VISIT_FUNC_IMPL(ExpandDimsNode)
VISIT_FUNC_IMPL(SliceAxisNode)
VISIT_FUNC_IMPL(ReshapeNode)
VISIT_FUNC_IMPL(LoweringNode)
VISIT_FUNC_IMPL(PackingNode)
VISIT_FUNC_IMPL(LogicalBroadcastNode)
VISIT_FUNC_IMPL(ConcatenateNode)
VISIT_FUNC_IMPL(TensorViewNode)
VISIT_FUNC_IMPL(LogicalTransposeNode)
VISIT_FUNC_IMPL(ReductionNode)
VISIT_FUNC_IMPL(IdentityNode)
VISIT_FUNC_IMPL(SqueezeNode)
VISIT_FUNC_IMPL(AggregationNode)

VISIT_FUNC_IMPL(DMANode)

VISIT_FUNC_IMPL(RotateNode)
VISIT_FUNC_IMPL(TransposeNode)
VISIT_FUNC_IMPL(EinsumNode)
VISIT_FUNC_IMPL(BroadcastNode)
VISIT_FUNC_IMPL(StridedViewNode)
VISIT_FUNC_IMPL(StridedInsertNode)

VISIT_FUNC_IMPL(MemcpyNode)
VISIT_FUNC_IMPL(MemsetNode)

VISIT_FUNC_IMPL(DebugNodeBase)
