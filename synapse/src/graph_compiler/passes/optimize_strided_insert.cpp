#include "decode_strided_op.h"
#include "habana_graph.h"
#include "strided_insert_node.h"
#include "node_factory.h"
#include "strided_op_node_utils.h"
#include "strided_view_node.h"
#include "slice_insert_node.h"
#include "slice_fwd_node.h"
#include "handle_memory_reuse.h"
#include <algorithm>
#include <iterator>

class ViewIFC
{
};

class StridedOpIFC : public ViewIFC
{
public:
    // Disallow creating an instance of this object
    StridedOpIFC() = delete;
    struct ViewInfo
    {
        const synStridedOpParams& params;
        const TensorPtr&          real;
        const TensorPtr&          view;
    };

    static bool isViewNode(const NodePtr& n) { return n && n->getNodeType() == Node::TYPE_STRIDED_VIEW; }
    static bool isInsertNode(const NodePtr& n) { return n && n->getNodeType() == Node::TYPE_STRIDED_INSERT; }
    static const TensorPtr& getOriginalInput(const NodePtr& n)
    {
        return n->getInput(StridedInsertNode::ORIGINAL_TENSOR);
    }
    static const TensorPtr& getInsertInput(const NodePtr& n) { return n->getInput(StridedInsertNode::INSERT_TENSOR); }

    static const ViewInfo getViewInfo(const NodePtr& n)
    {
        const auto* node = dynamic_cast<const StridedViewNode*>(n.get());
        HB_ASSERT_PTR(node);
        return {node->getParams(), n->getInput(0), n->getOutput(0)};
    }

    static const ViewInfo getInsertInfo(const NodePtr& n)
    {
        const auto* node = dynamic_cast<const StridedInsertNode*>(n.get());
        HB_ASSERT_PTR(node);
        return {node->getParams(), getOriginalInput(n), getInsertInput(n)};
    }

    static bool isDenseView(const ViewInfo& info)
    {
        return StridedOpUtils::isDenseStridedOpParams(info.params, info.view);
    }

    static DataRange<uint64_t> getDenseViewRange(const ViewInfo& info)
    {
        uint64_t offset = info.params.baseOffset;
        return DataRange<uint64_t>(offset, offset + info.view->getDenseSizeInElements());
    }

    static uint64_t getViewOffset(const ViewInfo& info) { return info.params.baseOffset; }

    static bool isOverlap(const ViewInfo& info1, const ViewInfo& info2)
    {
        return StridedOpUtils::isOverlap(info1.view, info2.view, info1.params, info2.params);
    }

    static bool isExactOverlap(const ViewInfo& info1, const ViewInfo& info2)
    {
        unsigned dim1 = info1.view->getDim();
        unsigned dim2 = info2.view->getDim();
        if (dim1 != dim2) return false;
        return info1.view->compareGeometry(*info2.view) &&
               StridedOpUtils::compareParams(info1.params, info2.params, dim1);
    }
};

class SliceOpIFC : public ViewIFC
{
public:
    // Disallow creating an instance of this object
    SliceOpIFC() = delete;

    struct ViewInfo
    {
        const SliceNode::SliceNodeStaticParams& params;
        const TensorPtr&                        real;
        const TensorPtr&                        view;
    };

    static bool             isViewNode(const NodePtr& n) { return n && n->getNodeType() == Node::TYPE_SLICE; }
    static bool             isInsertNode(const NodePtr& n) { return n && n->getNodeType() == Node::TYPE_SLICE_INSERT; }
    static const TensorPtr& getOriginalInput(const NodePtr& n) { return n->getInput(SliceInsertNode::ORIGINAL_TENSOR); }
    static const TensorPtr& getInsertInput(const NodePtr& n) { return n->getInput(SliceInsertNode::INSERT_TENSOR); }

    static const ViewInfo getViewInfo(const NodePtr& n)
    {
        const auto* node = dynamic_cast<const SliceFwdNode*>(n.get());
        HB_ASSERT_PTR(node);
        return {node->getParams(), n->getInput(0), n->getOutput(0)};
    }

    static const ViewInfo getInsertInfo(const NodePtr& n)
    {
        const auto* node = dynamic_cast<const SliceInsertNode*>(n.get());
        HB_ASSERT_PTR(node);
        return {node->getParams(), getOriginalInput(n), getInsertInput(n)};
    }

    static bool isDenseView(const ViewInfo& info)
    {
        if (info.view->getDim() == 0) return false;
        int lastDim = info.view->getDim() - 1;
        for (; lastDim >= 0; lastDim--)
        {
            if (info.view->getSizeInElements(lastDim) != 1) break;
        }

        // corner case - view of a single element, 1d scalar
        if (lastDim < 0) return info.params.steps[0] == 1;

        for (unsigned dim = 0; dim < lastDim; dim++)
        {
            if (info.view->getSizeInElements(dim) == 1) continue;
            if (info.params.steps[dim] != 1) return false;
            if (info.params.ends[dim] != info.real->getSizeInElements(dim)) return false;
            if (info.params.starts[dim] > 0) return false;
        }
        return info.params.steps[lastDim] == 1;
    }

    static uint64_t getViewOffset(const ViewInfo& info)
    {
        uint64_t offset = 0;
        uint64_t stride = 1;
        for (unsigned dim = 0; dim < info.real->getDim(); dim++)
        {
            offset += stride * info.params.starts[dim];
            stride *= info.real->getSizeInElements(dim) * info.params.steps[dim];
        }
        return offset;
    }

    static uint64_t getLastViewIndex(const ViewInfo& info)
    {
        uint64_t offset = 0;
        uint64_t stride = 1;
        for (unsigned dim = 0; dim < info.real->getDim(); dim++)
        {
            offset += info.params.starts[dim] * stride;
            offset += (info.view->getSizeInElements(dim) - 1) * stride * info.params.steps[dim];
            stride *= info.real->getSizeInElements(dim);
        }
        return offset;
    }

    static DataRange<uint64_t> getDenseViewRange(const ViewInfo& info)
    {
        uint64_t offset = getViewOffset(info);
        return DataRange<uint64_t>(offset, offset + info.view->getDenseSizeInElements());
    }

    static std::pair<std::vector<uint64_t>, std::vector<uint64_t>> getShapeAndStrides(const ViewInfo& info)
    {
        std::vector<uint64_t> shape(info.real->getDim()), strides(info.real->getDim());
        uint64_t              stride = 1;
        for (unsigned dim = 0; dim < info.real->getDim(); dim++)
        {
            shape[dim]   = info.view->getSizeInElements(dim);
            strides[dim] = stride *= info.params.steps[dim];
            stride *= info.real->getSizeInElements(dim);
        }
        return std::make_pair(shape, strides);
    }

    static bool isOverlap(const ViewInfo& info1, const ViewInfo& info2)
    {
        DataRange<uint64_t> r1 = getDenseViewRange(info1);
        DataRange<uint64_t> r2 = getDenseViewRange(info2);
        if (!r1.isOverlap(r2)) return false;

        auto [sizes1, strides1] = getShapeAndStrides(info1);
        auto [sizes2, strides2] = getShapeAndStrides(info2);
        return MemoryReuseHandler::isStridedOverlap(sizes1, sizes2, strides1, strides2, r1.start(), r2.start());
    }

    static bool isExactOverlap(const ViewInfo& info1, const ViewInfo& info2)
    {
        unsigned dim1 = info1.real->getDim();
        unsigned dim2 = info2.real->getDim();
        if (dim1 != dim2) return false;
        if (!info1.view->compareGeometry(*info2.view)) return false;
        if (!info1.real->compareGeometry(*info2.real)) return false;
        for (unsigned i = 0; i < dim1; i++)
        {
            if (info1.params.ends[i] != info2.params.ends[i]) return false;
            if (info1.params.starts[i] != info2.params.starts[i]) return false;
            if (info1.params.steps[i] != info2.params.steps[i]) return false;
        }
        return true;
    }
};

/*
    the purpose of this pass it to remove a strided insert chain caused by a gradient bucket. that is:
    [in] -> (SI) -> [.] -> (SI) -> ... (SI) -> [out]
    [d1] ----^     [d2] ----^  [dN] ----^

    will be converted into:

    [in] -----------------------> (MultiInsert) -> [out]
    [d1], [d2], .., [dN] ------------^

    this sequence can be detected by:
    (1) 1-d tensors for [in] and [out]
    (2) [in] and [out] are persistent
    (3) bucket contains a chain of non-dynamic StridedInsert nodes
    (4) all intermediate ORIGINAL_TENSORs have a single consumer, and are not persistent (can be removed)
    (5) all [di] tensors (INSERT_TENSOR) are non-overlapping and dense according to insert params
*/

template<typename IFC>
class GradientBucket
{
public:
    GradientBucket(const HabanaGraph& g, const TensorPtr& bucketOutput)
    : m_output(bucketOutput), m_input(nullptr), m_g(g)
    {
        static_assert(std::is_base_of<ViewIFC, IFC>::value, "IFC must inherit from ViewIFC");
    };

    bool addToBucket(const NodePtr& n);

    unsigned          size() const { return m_nodes.size(); }
    const NodeVector& getNodes() const { return m_nodes; }
    const TensorPtr&  getOutput() const { return m_output; }
    const TensorPtr&  getInput() const { return m_input; }
    void              setInput(const TensorPtr& in) { m_input = in; }

private:
    const TensorPtr m_output;  // strided insert chain output tensor
    TensorPtr       m_input;   // strided insert chain output tensor
    NodeVector      m_nodes;   // strided insert nodes in chain

    std::vector<DataRange<uint64_t>> m_coveredRanges;  // covered bucket ranges
    const HabanaGraph&               m_g;
};

template<typename IFC>
using BucketVector = std::vector<GradientBucket<IFC>>;

// note that the order of addition is important here. nodes in the chain must be added in reverse topological order
template<typename IFC>
bool GradientBucket<IFC>::addToBucket(const NodePtr& n)
{
    if (!IFC::isInsertNode(n)) return false;  // condition (3)
    // cannot handle dynamic offset or strides
    if (n->isDynamicShape()) return false;

    // condition (4)
    if (m_input)
    {
        HB_ASSERT(m_input == n->getOutput(0),
                  "error in handling strided insert chain with input {}",
                  m_input->getName());
        if (m_input->isPersistent() || m_g.getNumberOfTensorConsumers(m_input) > 1)
        {
            return false;  // intermediate tensor that is persistent
        }
    }

    const auto& info = IFC::getInsertInfo(n);
    // check if this is a dense view (condition (5))
    if (!IFC::isDenseView(info)) return false;

    DataRange<uint64_t> range = IFC::getDenseViewRange(info);
    for (const auto& r : m_coveredRanges)  // check all ranges for overlap
    {
        if (range.isOverlap(r))
        {
            LOG_DEBUG(OPTIMIZE_SI, "could node add node {} to gradient bucket due to overlap", n->getNodeName());
            return false;  // condition (5) - check overlap with previous insert tensors
        }
    }

    m_coveredRanges.push_back(range);  // save current ranges for later overlap checks
    m_input = IFC::getOriginalInput(n);
    m_nodes.push_back(n);
    LOG_DEBUG(OPTIMIZE_SI, "added node {} to gradient bucket", n->getNodeName());
    return true;
}

bool isPotentialBucketOutput(const HabanaGraph& g, const TensorPtr& t)
{
    return t && !t->isDynamicShape() && (t->getDim() == 1) && t->isPersistent();  // conditions (1,2)
}

template<typename IFC>
BucketVector<IFC> collectBucketTensors(const HabanaGraph& g)
{
    BucketVector<IFC> buckets;
    for (const TensorPtr& t : g.getGraphOutputs())
    {
        if (isPotentialBucketOutput(g, t))  // check if 't' might be a bucket output
        {
            NodePtr producer = g.getTensorProducer(t);
            if (IFC::isInsertNode(producer))  // condition (3)
            {
                LOG_TRACE(OPTIMIZE_SI, "trying to create bucket for tensor: {}", t->getName());
                GradientBucket<IFC> bucket(g, t);
                bool           res = true;
                while (res && producer)
                {
                    res      = bucket.addToBucket(producer);  // check other conditions
                    producer = g.getTensorProducer(IFC::getOriginalInput(producer));
                }

                if (bucket.size() > 0 && bucket.getInput()->isPersistent() && !producer)
                {
                    // collected bucket, with persistent graph input tensor
                    LOG_DEBUG(OPTIMIZE_SI, "found gradient bucket of size: {}", bucket.size());
                    buckets.push_back(bucket);
                }
                else  // failed some condition (1-5)
                {
                    LOG_DEBUG(OPTIMIZE_SI,
                              "failed to create gradient bucket of size {} for tensor {}",
                              bucket.size(),
                              t->getName());
                }
            }
        }
    }
    return buckets;
}

template<typename IFC>
void combineStridedInsertNodes(HabanaGraph& g)
{
    // get all "bucket" sequences ([in] -> (SI) -> ... -> (SI) -> [out])
    BucketVector<IFC> buckets = collectBucketTensors<IFC>(g);

    for (const GradientBucket<IFC>& bucket : buckets)
    {
        TensorVector          inputs  = {bucket.getInput()};
        const TensorPtr&      out     = bucket.getOutput();
        std::vector<uint64_t> offsets = {0};
        const auto&           nodes   = bucket.getNodes();
        for (const NodePtr& n : nodes)
        {
            inputs.push_back(IFC::getInsertInput(n));
            offsets.push_back(IFC::getViewOffset(IFC::getInsertInfo(n)));
        }

        NodePtr multiInsert = NodeFactory::createNode(inputs,
                                                      {out},
                                                      &offsets,
                                                      NodeFactory::multiInsertNodeTypeName,
                                                      "multi_insert_" + out->getName());
        // replace strided insert chain with multi-insert.
        /* cannot use replaceNode because relaxCtrlDeps hasn't ran yet.
         so replaceNodes might think we will fuse a graph cycle.
         but we are guaranteed that not ctrl edges are required because all intermediate tensors are non-persistent */
        GraphEditor::removeNodes(g, {nodes.begin(), nodes.end()});
        GraphEditor::addNode(g, multiInsert);
    }
}

/**
try to move strided view nodes ahead of non-overlapping writes (written by strided insert)
  @param bypassedNodes - all strided insert nodes that were "touched" during this process
  @return TensorPtr - the earliest input strided view can operate on
  @return bool - wether or not this input is the "complete" view - identical to strided view output
*/
template<typename IFC>
static std::pair<TensorPtr, bool>
findNewStridedViewInput(const HabanaGraph& g, const NodePtr& n, NodeSet& bypassedNodes)
{
    HB_ASSERT(IFC::isViewNode(n), "{}: expecting view node!", __func__);
    const TensorPtr&          output         = n->getOutput(0);
    const auto&               info           = IFC::getViewInfo(n);
    TensorPtr                 input          = n->getInput(0);
    bool                      isExactOverlap = false;

    NodePtr producer = g.getTensorProducer(input);
    while (IFC::isInsertNode(producer))
    {
        bypassedNodes.insert(producer);  // mark this node as "touched"
        const TensorPtr& insert = IFC::getInsertInput(producer);
        const auto&      siInfo = IFC::getInsertInfo(producer);

        if (IFC::isOverlap(info, siInfo))  // overlapping read/write
        {
            if (IFC::isExactOverlap(info, siInfo))  // exact overlap
            {
                // in this case we found a "complete write" of the strided view output.
                // the data in this tensor ('insert') is identical to the view output.
                // return this tensor, as the complete view
                LOG_DEBUG(OPTIMIZE_SI,
                          "found identical data tensor for view output {}, in tensor {}",
                          output->getName(),
                          insert->getName());
                input          = insert;
                isExactOverlap = true;
            }
            else
            {
                // if there is any other overlap between read (view output) and write (insert input), don't bypass.
                LOG_DEBUG(OPTIMIZE_SI, "insert node {} is overlapping - cannot bypass", producer->getNodeName());
            }
            break;
        }
        else  // no overlap between view node and insert node
        {
            LOG_DEBUG(OPTIMIZE_SI,
                      "moving view node {} before non-overlapping insert node {}",
                      n->getNodeName(),
                      producer->getNodeName());
            // move to check the next producer
            input    = IFC::getOriginalInput(producer);
            producer = g.getTensorProducer(input);
        }
    }

    return std::make_pair(input, isExactOverlap);  // return the new input
}

void moveStridedViewToNewInput(HabanaGraph& g, NodePtr viewNode, TensorPtr newInput, bool replaceWithIdentity)
{
    if (replaceWithIdentity)
    {
        NodePtr identity = NodeFactory::createNode({newInput},
                                                   {viewNode->getOutput(0)},
                                                   nullptr,
                                                   NodeFactory::identityNodeTypeName,
                                                   viewNode->getNodeName());
        GraphEditor::replaceNodes(g, {viewNode}, {identity});
    }
    else
    {
        GraphEditor::replaceInput(g, viewNode, 0, newInput);
    }
}

/*
    This optimization tried to move SV nodes to execute on a earlier produced input.
    1. if a strided view node is produced by a strided insert node, with a non-overlapping insert write,
        the strided view can be executed on the input of that SI node. Example:
        [Original] -> (SI) -> [SI_out] -> (SV)  will turn into:   [Original] --------------> (SI) -> [SI_OUT]
        [Insert] ------^                                              ^-(SV)     [Insert] ----^

    2. if a strided view node is produced by a strided insert node with a fully-overlapping insert write,
       (+ Insert tensor has same geometry as SV output), Strided view can be replaced with identity on that insert:
               [Original] -> (SI) -> [SI_out] -> (SV)  will turn into:   [Original]-> (SI) -> [SI_OUT]
        [Insert] ------^                                                                |
                                                                          [Insert] -----+---(Identity) -> [SV_out]

    3. for every such strided insert node that is "bypassed", if it doesn't have any other consumers - it can be removed

*/
template<typename IFC>
static void moveStridedViews(HabanaGraph& g)
{
    NodeVector topoSortedNodes = g.getTopoSortedNodes();
    NodeSet    bypassedNodes;        // all strided insert nodes that were "touched"
    NodeVector bypassedNodesSorted;  // all the bypassed nodes, sorted in reverse topological order
    for (int i = topoSortedNodes.size() - 1; i >= 0; i--)
    {
        const NodePtr& n = topoSortedNodes[i];
        if (IFC::isViewNode(n))
        {
            if (n->isDynamicShape()) continue;  // can't verify overlap when having dynamic shapes, can't optimize
            LOG_TRACE(OPTIMIZE_SI, "attempting to move strided_view node {}", n->getNodeName());

            // (1) + (2) - move ahead strided view node
            auto [newInput, isExactOverlap] = findNewStridedViewInput<IFC>(g, n, bypassedNodes);
            if (newInput != n->getInput(0))  // strided view can be moved ahead
            {
                moveStridedViewToNewInput(g, n, newInput, isExactOverlap);
            }
        }
        else if (bypassedNodes.find(n) != bypassedNodes.end())
        {
            bypassedNodesSorted.push_back(n);  // sort as we go, in reverse topo order
        }
    }

    // (3)
    for (const NodePtr& node : bypassedNodesSorted)
    {
        for (const TensorPtr& out : node->getOutputs())
        {
            // check if this output is needed by anyone else except the original strided view we moved
            if (out->isUserManagedDram() || g.getNumberOfTensorConsumers(out) > 0) continue;

            // otherwise - remove this node
            GraphEditor::removeNode(g, node);
        }
    }
}

// return <can be replaced with reshape, is strided view/insert>
std::pair<bool, bool> isReshapeStridedOpNode(const NodePtr& n)
{
    bool isSV = n->getNodeType() == Node::TYPE_STRIDED_VIEW;
    bool isSI = n->getNodeType() == Node::TYPE_STRIDED_INSERT;
    if (!isSV && !isSI) return std::make_pair(false, false);
    if (n->isDynamicShape()) return std::make_pair(false, false);

    const TensorPtr& viewTensor = isSV ? n->getOutput(0) : n->getInput(StridedInsertNode::INSERT_TENSOR);
    const TensorPtr& realTensor = isSV ? n->getInput(0) : n->getOutput(0);
    if (viewTensor->getDenseSizeInElements() != realTensor->getDenseSizeInElements())
    {
        LOG_TRACE(OPTIMIZE_SI, "{}: different dense size for operands of node {}", __func__, n->getNodeName());
        return std::make_pair(false, false);
    }

    const synStridedOpParams& params = isSV ? dynamic_cast<StridedViewNode*>(n.get())->getParams()
                                            : dynamic_cast<StridedInsertNode*>(n.get())->getParams();
    if (!StridedOpUtils::isDenseStridedOpParams(params, viewTensor))
    {
        LOG_TRACE(OPTIMIZE_SI, "{}: non-dense strided op params for node {}", __func__, n->getNodeName());
        return std::make_pair(false, false);
    }
    return std::make_pair(true, isSV);
}

// replace strided view/insert with reshape if possible
void replaceStridedOpWithReshape(HabanaGraph& g)
{
    std::vector<std::pair<NodePtr, bool>> reshapeStridedOpNodes;
    for (const NodePtr& n : g.getNodes())
    {
        if (!n) continue;
        auto [isReshape, isSV] = isReshapeStridedOpNode(n);
        if (isReshape)
        {
            reshapeStridedOpNodes.push_back(std::make_pair(n, isSV));
        }
    }

    for (const auto& [node, isSV] : reshapeStridedOpNodes)
    {
        LOG_DEBUG(OPTIMIZE_SI, "{}, replacing node {} with reshape", __func__, node->getNodeName());
        const TensorPtr&        input   = isSV ? node->getInput(0) : node->getInput(StridedInsertNode::INSERT_TENSOR);
        NodePtr                 reshape = NodeFactory::createNode({input},
                                                                  {node->getOutput(0)},
                                                  nullptr,
                                                  NodeFactory::reshapeNodeTypeName,
                                                  node->getNodeName());
        ReplaceNodeReturnStatus status  = GraphEditor::replaceNodes(g, {node}, {reshape});
        HB_ASSERT(status == REPLACE_NODE_SUCCESS, "failed to replace node!");
    }
}

static void decodeStridedOperations(HabanaGraph& g)
{
    if (!GCFG_ENABLE_STRIDED_OP_DECODING.value()) return;
    const NodeSet& nodes = g.getNodes();
    NodeVector     stridedNodeCandidates;
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(stridedNodeCandidates), [](const NodePtr& n) {
        return StridedOpIFC::isViewNode(n) || StridedOpIFC::isInsertNode(n);
    });
    for (const NodePtr& stridedOp : stridedNodeCandidates)
    {
        if (!StridedOpDecoder::canExtract(stridedOp)) continue;
        NodeVector extractedNodes = StridedOpDecoder::extract(stridedOp, false /*changeInPlace*/);
        if (extractedNodes.empty()) continue;
        GraphEditor::replaceNodes(g, {stridedOp}, extractedNodes);
    }
}

bool optimizeStridedInsert(HabanaGraph& g)
{
    if (!GCFG_ENABLE_OPTIMIZE_STRIDED_INSERT.value()) return true;

    moveStridedViews<StridedOpIFC>(g);
    combineStridedInsertNodes<StridedOpIFC>(g);

    moveStridedViews<SliceOpIFC>(g);
    combineStridedInsertNodes<SliceOpIFC>(g);

    decodeStridedOperations(g);
    replaceStridedOpWithReshape(g);
    return true;
}