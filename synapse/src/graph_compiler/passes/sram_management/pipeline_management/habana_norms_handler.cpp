#include "habana_norms_handler.h"
#include "node_factory.h"
#include "node_utils.h"
#include "slice_fwd_logical_node.h"
#include "perf_lib_layer_params.h"

void PatternNodesCollector::registerPattern(const SliceTilePattern& pattern)
{
    const auto& sliceNode                     = pattern.slice;
    m_sliceInToNodes[sliceNode->getInput(0)].insert(sliceNode);
    m_sliceOutToNode[sliceNode->getOutput(0)].insert(sliceNode);
    HB_ASSERT(m_sliceOutToNode[sliceNode->getOutput(0)].size() == 1, "Tensor {} can only be produced by a single slice node", sliceNode->getOutput(0)->getName());
    LOG_TRACE(SLICE_NORM, "{}: sliceNode={}, input={}, output={}", __func__, sliceNode->getNodeName(), sliceNode->getInput(0)->getName(), sliceNode->getOutput(0)->getName());
    if (auto tileNode = pattern.tile)
    {
        m_tileOutToNode[tileNode->getOutput(0)].insert(tileNode);
        HB_ASSERT(m_tileOutToNode[tileNode->getOutput(0)].size() == 1, "Tensor {} can only be produced by a single tile node", tileNode->getOutput(0)->getName());
        LOG_TRACE(SLICE_NORM, "{}: tileNode={}, input={}, output={}", __func__, tileNode->getNodeName(), tileNode->getInput(0)->getName(), tileNode->getOutput(0)->getName());
    }
}

NodeSet PatternNodesCollector::getSliceNodesFromInput(const TensorPtr& t)
{
    const TensorPtr& key = t->getTensorAnnotation().origBigTensor ? t->getTensorAnnotation().origBigTensor : t;
    if (const auto& sliceNodes = m_sliceInToNodes.find(key); sliceNodes != m_sliceInToNodes.end())
    {
        return sliceNodes->second;
    }
    return {};
}

NodePtr PatternNodesCollector::getSliceNodeFromOutput(const TensorPtr& t)
{
    const TensorPtr& key = t->getTensorAnnotation().origBigTensor ? t->getTensorAnnotation().origBigTensor : t;
    if (const auto& sliceNode = m_sliceOutToNode.find(key); sliceNode != m_sliceOutToNode.end())
    {
        HB_ASSERT(sliceNode->second.size() == 1,
                  "{}: Tensor {} can only be produced by a single slice node",
                  __FUNCTION__,
                  t->getName());
        return *sliceNode->second.begin();
    }
    return nullptr;
}

NodePtr PatternNodesCollector::getTileNodeFromTensor(const TensorPtr& t)
{
    const TensorPtr& key = t->getTensorAnnotation().origBigTensor ? t->getTensorAnnotation().origBigTensor : t;
    if (const auto& tileNode = m_tileOutToNode.find(key); tileNode != m_tileOutToNode.end())
    {
        HB_ASSERT(tileNode->second.size() == 1,
                  "{}: Tensor {} can only be produced by a single tile node",
                  __FUNCTION__,
                  t->getName());
        return *tileNode->second.begin();
    }
    return nullptr;
}

std::optional<PatternNodesCollector::PipelinedPattern>
PatternNodesCollector::findSliceAndTilePatternFromInput(HabanaGraph& g, const NodePtr& consumer, const TensorPtr& in)
{
    PipelinedPattern pattern;

    if (auto tile = getTileNodeFromTensor(in))
    {
        const TensorPtr& sliceOutput = tile->getInput(0);
        // The new slice input can be reached from the reshape node's input ==> consumer's first input producer's
        // input
        const auto& producers = g.getNodeProducers(consumer);
        auto        reshapeNode =
            std::find_if(producers.begin(), producers.end(), [](const NodePtr& prod) { return isReshapeNode(prod); });
        HB_ASSERT(reshapeNode != producers.end(),
                  "In slice+tile pattern, exactly 1 producer of {} should be a reshape node",
                  consumer->getNodeName());
        const NodePtr&   nodeFromOutput = getSliceNodeFromOutput(sliceOutput);
        const TensorPtr& sliceInput     = (*reshapeNode)->getInput(0);
        pattern.pipelinedSlice          = PipelinedSlice(sliceInput, sliceOutput, nodeFromOutput);
        pattern.pipelinedTile           = tile;
        return pattern;
    }
    return {};
}
// Pattern 1-
// As explained in the drawing below, the original sliceNode's input/output tensors were both consumed by the same node.
// Since this consumer might be sliced at this point, the sliceNode's tensors need to be updated before re-adding it to
// the graph.
// Pattern 2-
// The original tile node can be retrieved from its original consumer. The new slice output is the tile node input,
// which hasn't changed after slicing due to the fact that we removed the slice and tile nodes from the graph, and by
// that, we disconnected the slice output (which is an intermediate tensor) from the graph.
std::optional<PatternNodesCollector::PipelinedPattern> PatternNodesCollector::getPipelinedNodes(HabanaGraph&   g,
                                                                                            const NodePtr& consumer)
{
    TensorPtr        sliceInput;
    TensorPtr        sliceOutput;
    NodeSet          nodesFromInput;
    NodePtr          nodeFromOutput;
    PipelinedPattern pattern1;

    for (const TensorPtr& in : consumer->getInputs())
    {
        if (!in) continue;
        auto pattern2 = findSliceAndTilePatternFromInput(g, consumer, in);
        if (pattern2.has_value())
        {
            return pattern2;
        }
        if (const auto& sliceNodes = getSliceNodesFromInput(in); !sliceNodes.empty())
        {
            sliceInput    = in;
            nodesFromInput = sliceNodes;
            continue;
        }
        if (const auto& slice = getSliceNodeFromOutput(in))
        {
            sliceOutput    = in;
            nodeFromOutput = slice;
            continue;
        }
    }
    if ((std::any_of(nodesFromInput.begin(),
                     nodesFromInput.end(),
                     [&nodeFromOutput](const NodePtr& nodeFromInput) { return nodeFromInput == nodeFromOutput; })) &&
        sliceInput && sliceOutput)
    {
        pattern1.pipelinedSlice = PipelinedSlice(sliceInput, sliceOutput, nodeFromOutput);
        return pattern1;
    }
    return {};
}

std::vector<PatternNodesCollector::PipelinedPattern> PatternNodesCollector::getPatternNodesMetadata(HabanaGraph& g)
{
    std::vector<PatternNodesCollector::PipelinedPattern> ret;
    NodeVector                                           nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        auto pipelinedPattern = getPipelinedNodes(g, node);
        if (!pipelinedPattern.has_value()) continue;
        ret.push_back(pipelinedPattern.value());
    }
    return ret;
}

bool HabanaNormsHandler::validateHandleSliceNodes()
{
    for (const auto& t : m_graph.getTensors())
    {
        if (auto origSliceNode = m_sliceCollector->getSliceNodeFromOutput(t))
        {
            if (!origSliceNode) continue;
            if (!m_graph.getTensorProducer(t) || (m_graph.getTensorProducer(t)->getNodeType() != Node::TYPE_SLICE &&
                                                  m_graph.getTensorProducer(t)->getNodeType() != Node::TYPE_MEMCOPY))
            {
                LOG_ERR(SLICE_NORM,
                        "Tensor {} was produced by slice node {} and wasn't handled!",
                        t->getName(),
                        origSliceNode->getNodeName());
                return false;
            }
        }
        else if (auto origTileNode = m_sliceCollector->getTileNodeFromTensor(t))
        {
            if (!origTileNode) continue;
            const auto& producer = m_graph.getTensorProducer(t);
            if (!producer || (producer != origTileNode))
            {
                LOG_ERR(SLICE_NORM,
                        "Tensor {} was produced by tile node {} and wasn't handled!",
                        t->getName(),
                        origTileNode->getNodeName());
                return false;
            }
            const auto& tileProducer = m_graph.getTensorProducer(origTileNode->getInput(0));
            if (!tileProducer ||
                (tileProducer->getNodeType() != Node::TYPE_SLICE && tileProducer->getNodeType() != Node::TYPE_MEMCOPY))
            {
                LOG_ERR(SLICE_NORM,
                        "Tensor {} was produced by slice node {} and wasn't handled!",
                        origTileNode->getInput(0)->getName(),
                        tileProducer->getNodeName());
                return false;
            }
        }
    }
    return true;
}

SliceNode::SliceNodeStaticParams HabanaNormsHandler::getUpdatedSliceParams(const NodePtr&   node,
                                                                           const TensorPtr& newOut) const
{
    auto sliceNode = std::dynamic_pointer_cast<LogicalSliceFwdNode>(node);
    auto params    = sliceNode->getParams();
    for (unsigned dim = 0; dim < sliceNode->getInput(0)->getDim(); dim++)
    {
        if (params.ends[dim] != 1)
        {
            params.ends[dim] = newOut->getSizeInElements(dim);
        }
    }
    return params;
}

NodePtr HabanaNormsHandler::addUpdatedSliceNode(const TensorPtr& in, const TensorPtr& out, NodePtr origSliceNode)
{
    auto [bundleInfo, newSliceName] = getUpdatedBundleInfoAndNodeName(in, origSliceNode);
    SliceNode::SliceNodeStaticParams newParams = getUpdatedSliceParams(origSliceNode, out);
    NodePtr currSliceNode = NodeFactory::createNode({in}, {out}, &newParams, origSliceNode->getGUID(), newSliceName);
    currSliceNode->getNodeAnnotation().bundleInfo = bundleInfo;
    GraphEditor::addNode(m_graph, currSliceNode);
    LOG_DEBUG(SLICE_NORM,
              "Added {} to the graph with updated tensors: new input- {}, new output- {}",
              currSliceNode->getNodeName(),
              in->getName(),
              out->getName());
    return currSliceNode;
}

std::pair<Settable<BundleInfo>, std::string> HabanaNormsHandler::getUpdatedBundleInfoAndNodeName(const TensorPtr& t, const NodePtr& origNode) const
{
    auto        bundleInfo   = m_graph.getTensorProducer(t)->getNodeAnnotation().bundleInfo;
    std::string newNodeName =
        fmt::format("{}_bundle_{}/op_{}", origNode->getNodeName(), bundleInfo->bundleIndex, bundleInfo->operationIndex);
    return {bundleInfo, newNodeName};
}

NodePtr HabanaNormsHandler::addUpdatedTileNode(const TensorPtr& in, NodePtr origTileNode)
{
    auto [bundleInfo, newTileName]                = getUpdatedBundleInfoAndNodeName(in, origTileNode);
    origTileNode->getNodeAnnotation().bundleInfo = bundleInfo;
    origTileNode->setName(newTileName);
    GraphEditor::addNode(m_graph, origTileNode);
    LOG_DEBUG(SLICE_NORM, "Added {} to the graph", origTileNode->getNodeName());
    return origTileNode;
}

// Returns true if the producer is slice, or if the producer's producer is slice
bool HabanaNormsHandler::isSliceProducer(const NodePtr& producer)
{
    return producer->getNodeType() == Node::TYPE_SLICE ||
           ((producer->getNodeType() == Node::TYPE_MEMCOPY) &&
            (m_graph.getTensorProducer(producer->getInput(0))->getNodeType() == Node::TYPE_SLICE));
}

void HabanaNormsHandler::addSlicedPatternToGraph(const TensorPtr&         newSliceInput,
                                                 const TensorPtr&         newSliceOutput,
                                                 const NodePtr&           origSliceNode,
                                                 const Settable<NodePtr>& pipelinedTileOpt)
{
    // Add the slice node back to the graph
    auto currSliceNode = addUpdatedSliceNode(newSliceInput, newSliceOutput, origSliceNode);

    // Add tile node if exists in the pattern
    if (pipelinedTileOpt.is_set())
    {
        const NodePtr& pipelinedTile = pipelinedTileOpt.value();
        addUpdatedTileNode(pipelinedTile->getInput(0), pipelinedTile);
    }

    // Add memcopy node for sliceNode's output if it's input is in sram (because of double-buffering)
    if (newSliceInput->inSram())
    {
        GraphEditor::insertMemcpyForOutput(m_graph, currSliceNode, 0, newSliceInput->location());
    }
}

void HabanaNormsHandler::addUnslicedPatternToGraph(const TensorPtr&         newSliceInput,
                                                   const NodePtr&           origSliceNode,
                                                   const Settable<NodePtr>& pipelinedTileOpt)
{
    const auto& sliceProducer = m_graph.getTensorProducer(newSliceInput);
    if (sliceProducer)
    {
        // In case the norm node is bundled but not sliced, copy the bundle info from the slice
        // producer to avoid cycle in BP graph.
        origSliceNode->getNodeAnnotation().bundleInfo = sliceProducer->getNodeAnnotation().bundleInfo;
    }
    if (pipelinedTileOpt.is_set())
    {
        const auto& pipelinedTile                     = pipelinedTileOpt.value();
        pipelinedTile->getNodeAnnotation().bundleInfo = origSliceNode->getNodeAnnotation().bundleInfo;
        LOG_DEBUG(SLICE_NORM,
                  "TileNode {}'s tensors didn't change, adding the original tile node back to the graph",
                  pipelinedTile->getNodeName());
        GraphEditor::addNode(m_graph, pipelinedTile);
    }
    LOG_DEBUG(SLICE_NORM,
              "SliceNode {}'s tensors didn't change, adding the original slice node back to the graph",
              origSliceNode->getNodeName());

    GraphEditor::addNode(m_graph, origSliceNode);
}

/* clang-format off
Pattern 1 drawings:
Original graph:
                    +--------+       +-------+---+
                    | Slice  +------>| slice_out |
                    |  Node  |       +---+-------+
                    +--------+           |
                         ^               |
                         |               v
   +----------+    +-----+------+    +--------+
   | Producer +--->|  prod_out  +--->|Consumer|
   |  Node    |    +----+-------+    |  Node  |
   +----------+                      +--------+
After SliceNode removal:
                                     +-------+---+
                                     | slice_out |
                                     +---+-------+
                                         |
                                         |
                                         v
   +----------+    +-----+------+    +--------+
   | Producer +--->|  prod_out  +--->|Consumer|
   |  Node    |    +----+-------+    |  Node  |
   +----------+                      +--------+
Post slicing:
Option 1- no slicing.
Option 2- slice_out doesn't have a producer.
          In case of pattern 2, the consumer node will get tile_out as input instead of slice_out.
                              +----------+
                              |slice_out |
                              +----+-----+
                                   |
                                   |
    +---------+                    |      +----------+
    | Producer|     +----------+   +----->| Consumer |
    | node    +---->| prod_out |--------->| node     |
    +---------+     +----------+   |      +----------+
                                   |
    +---------+                    |      +----------+
    | Producer|     +----------+   +----->| Consumer |
    | node    |---->| prod_out |--------->| node     |
    +---------+     +----------+   |      +----------+
                                   |
    +---------+                    |      +----------+
    | Producer|     +----------+   +----->| Consumer |
    | node    |---->| prod_out |--------->| node     |
    +---------+     +----------+           +----------+
Option 3- slice_out was sliced, new slice output has a producer (split node).
          This case is not valid for pattern 2, since that would mean that the bundle was sliced on an unsupported dim.
    +-------------+        +---------+      +-------------+      +-------------+     +---------+    +-------------+
    | slice_out   |------->| Split   |----->| slice_out/2 |      | slice_out   |---->| Split   |--->| slice_out/2 |
    +-------------+        | node    |      +--+----------+      +-------------+     | node    |    +--+----------+
                           +---------+         |                                     +---------+       |
                                               |                                                       |
                +---------+                    |       +----------+     +---------+                    |       +----------+
                | Producer|     +----------+   +------>| Consumer |     | Producer|     +----------+   +------>| Consumer |
                | node    +---->| prod_out |---------->| node     |     | node    +---->| prod_out |---------->| node     |
                +---------+     +----------+   |       +----------+     +---------+     +----------+   |       +----------+
                                               |                                                       |
                +---------+                    |      +----------+      +---------+                    |       +----------+
                | Producer|     +----------+   +----->| Consumer |      | Producer|     +----------+   +------>| Consumer |
                | node    |---->| prod_out |--------->| node     |      | node    |---->| prod_out |---------->| node     |
                +---------+     +----------+   |      +----------+      +---------+     +----------+   |       +----------+
                                               |                                                       |
                +---------+                    |      +----------+      +---------+                    |       +----------+
                | Producer|     +----------+   +----->| Consumer |      | Producer|     +----------+   +------>| Consumer |
                | node    |---->| prod_out |--------->| node     |      | node    |---->| prod_out |---------->| node     |
                +---------+     +----------+          +----------+      +---------+     +----------+           +----------+
Explanation:
SliceNodes are removed pre-slicing/bundling to allow slicing of the norm, when possible.
This is the post-slicing stage.
Three cases are being handled:
1. SliceNode's original tensors remained unchanged (can happen when the consumer was not bundled) - in this case the
   original SliceNode can be added to the graph without further manipulations.
2. SliceNode's output does not have a producer in the post-slicing graph (option 2 drawing above) - this scenario
   occures when:
   a. In both patterns - Input was sliced, output tensor is unchanged, OR
   b. Only relevant for pattern 1 - Output was sliced, and its producer was removed in
   previous iteration (case 3). Add the slice node as producer to the output, and replace the original input with the
   first input slice that is generated.
3. SliceNode's output has a producer in the post-slicing graph (option 3 drawing above) - If the producer is Slice,
   continue to the next node without doing anything. Else, Remove the producer from the graph, and add SliceNode (or
   clones of it) as producer to each of the original producer's output.
clang-format on */
bool HabanaNormsHandler::handleRemovedSliceNormNodes()
{
    if (!GCFG_ENABLE_SLICE_NORM_BUNDLING.value())
    {
        return true;
    }
    auto patterns = m_sliceCollector->getPatternNodesMetadata(m_graph);
    for (const auto& pattern : patterns)
    {
        const auto&      pipelinedSlice = pattern.pipelinedSlice;
        const TensorPtr& newSliceInput  = pipelinedSlice.newInput;
        const TensorPtr& newSliceOutput = pipelinedSlice.newOutput;
        const NodePtr&   origSliceNode  = pipelinedSlice.originalSliceNode;
        const Settable<NodePtr>& pipelinedTile  = pattern.pipelinedTile;

        // Case 1:
        if (newSliceInput == origSliceNode->getInput(0) && newSliceOutput == origSliceNode->getOutput(0))
        {
            addUnslicedPatternToGraph(newSliceInput, origSliceNode, pipelinedTile);
            continue;
        }

        // Case 3:
        if (auto producer = m_graph.getTensorProducer(newSliceOutput))
        {
            if (isSliceProducer(producer))
            {
                // If the Slice node is already added - newSliceInput is the output of a non first sliced node of the
                // producer, which must run after the first slice of the producer
                addCtrlDepToSliceProducer(producer, newSliceInput);
                continue;
            }
            HB_ASSERT(!pipelinedTile.is_set(),
                      "For slice+tile patterns, if the new slice output has a producer, it must be a slice node (that "
                      "was added in previous iteration). Otherwise, this means that we sliced the "
                      "bundle on an unexpected dim");
            HB_ASSERT(producer->getNodeType() == Node::TYPE_INTERNAL_SPLIT || Node::isForkNode(producer),
                      "sliceNode's output may only be produced by split/fork (for unhandled tensors) or slice (for handled "
                      "tensors), or memcopy (if slice output is in sram)");
            GraphEditor::removeNode(m_graph, producer);
        }

        // Now that the newSliceOutput has no producer, we are at case 2:
        addSlicedPatternToGraph(newSliceInput, newSliceOutput, origSliceNode, pipelinedTile);

    }
    return validateHandleSliceNodes();
}

// The Slice node producer is sliced to multiple slices. The first encountered slice is the producer to Slice, while the
// other slices are expected to run after it. To maintain correct scheduling of the Slice consumers - add control
// dependency to make the first producer's sliced node block the rest of the producer's slices. Thus reflecting that the
// non first consumers are dependent also on the first producer.
void HabanaNormsHandler::addCtrlDepToSliceProducer(const NodePtr& slice, const TensorPtr producerOut)
{
    // Get the Slice node producer, which is the first producer sliced node
    HB_ASSERT(slice->getNumInputs() == 1, "slice node is expected to have a single input");
    const auto& realSliceProducers = m_graph.getRealProducers(slice->getInput(0));
    HB_ASSERT(realSliceProducers.size() == 1, "slice input is expected to have a single real producer");
    const NodePtr& firstProdSlicedNode = *realSliceProducers.begin();
    // Get the non first Slice producer sliced node, which creates producerOut
    const auto& realOutProducers = m_graph.getRealProducers(producerOut);
    HB_ASSERT(realOutProducers.size() == 1, "producer output is expected to have a single real producer");
    const NodePtr& nonFirstProdSlicedNode = *realOutProducers.begin();
    // Schedule the non first sliced node to be executed after the first sliced node
    m_graph.addControlDependency(firstProdSlicedNode, nonFirstProdSlicedNode, Tensor::ControlEdgeType::SCHEDULE);
}

// First pattern that is possible for the norms solution is that the input and output of the
// sliceNode are both inputs to the sliceNode's output's consumer, which will be used later to backtrack the
// sliceNode's tensors. If such common consumer does not exist, then this slice node does not match the pattern.
bool HabanaNormsHandler::existsCommonConsumerForSliceTensors(const NodePtr& node)
{
    const auto& sliceNodeOutConsumers = m_graph.getTensorConsumers(node->getOutput(0));
    const auto& sliceNodeInConsumers  = m_graph.getTensorConsumers(node->getInput(0));
    return sliceNodeOutConsumers.size() == 1 &&
           std::any_of(sliceNodeInConsumers.begin(), sliceNodeInConsumers.end(), [&](const NodePtr& consumer) {
               return consumer == sliceNodeOutConsumers.front();
           });
}

bool HabanaNormsHandler::isReshapeValidForPattern(const NodePtr& reshapeNode,
                                                  const NodePtr& sliceNode,
                                                  const NodePtr& tileNode)
{
    if (!isReshapeNode(reshapeNode)) return false;
    // Reshape output at dim 0 should be equal to the tile node output at dim 0:
    if (reshapeNode->getOutput(0)->getSizeInElements(0) != tileNode->getOutput(0)->getSizeInElements(0)) return false;
    // Slice input should be also the reshape input:
    if (sliceNode->getInput(0) != reshapeNode->getInput(0)) return false;
    // Tile should have a single consumer, and that consumer should also be consumed by the reshape:
    const auto& reshapeConsumers = m_graph.getTensorConsumers(reshapeNode->getOutput(0));
    const auto& tileConsumers    = m_graph.getTensorConsumers(tileNode->getOutput(0));
    if (tileConsumers.size() != 1) return false;
    return std::any_of(reshapeConsumers.begin(), reshapeConsumers.end(), [&](const NodePtr& reshapeConsumer){
        return reshapeConsumer == tileConsumers.front();
    });
}

// Second pattern that is possible for the norms solution is that the output of the
// sliceNode is consumed by a Tile node, and the sliceNode's input is consumed by Reshape. The reshape output and Tile
// output should have a single common consumer.
bool HabanaNormsHandler::sliceNodePatternWithTile(const NodePtr& slice, const NodePtr& tile)
{
    const auto& sliceProducer          = m_graph.getTensorProducer(slice->getInput(0));
    const auto& sliceProducerConsumers = m_graph.getNodeConsumers(sliceProducer);
    auto reshapeConsumer =
        std::find_if(sliceProducerConsumers.begin(), sliceProducerConsumers.end(), [&](const NodePtr& consumer) {
            return isReshapeValidForPattern(consumer, slice, tile);
        });
    return (reshapeConsumer != sliceProducerConsumers.end());
}

// The allowed slice node for pipelining extracts everything except the normalization reduced dimensions from the input.
// Therefore the valid params should have start idx = 0 for all dims, step = 1 (not strided), and end idx should be
// either 1 or dimSize.
bool HabanaNormsHandler::isValidSliceParams(const TensorPtr& input, const SliceNode::SliceNodeStaticParams& params)
{
    for (unsigned dim = 0; dim < input->getDim(); dim++)
    {
        if (params.starts[dim] != 0 || params.steps[dim] != 1 ||
            (params.ends[dim] != 1 && params.ends[dim] != input->getSizeInElements(dim)))
        {
            return false;
        }
    }
    return true;
}

// The allowed tile node for pipelining repeats the FCD (dim 0) only. Therefore the valid params should have tile factor
// != 1 for dim 0, while the rest of the dims should have tile factor 1.
bool HabanaNormsHandler::isValidTileParams(const TensorPtr& input, const ns_TileKernel::ParamsV2* params)
{
    if (!params) return false;
    if (params->repeat[0] == 1) return false;
    for (unsigned dim = 1; dim < input->getDim(); dim++)
    {
        if (params->repeat[dim] != 1) return false;
    }
    return true;
}

bool HabanaNormsHandler::isSlicePatternValidForPipelining(const NodePtr& node)
{
    if (const auto slice = std::dynamic_pointer_cast<LogicalSliceFwdNode>(node); slice)
    {
        if (!node->getNodeAnnotation().originatedFromCguid)
        {
            LOG_DEBUG(SLICE_NORM, "Skipping slice node {} - it's not originated from cguid", node->getNodeName());
            return false;
        }
        if (!existsCommonConsumerForSliceTensors(node))
        {
            LOG_DEBUG(SLICE_NORM,
                      "Skipping slice node {} that originated from cguid - does not match slice pattern",
                      node->getNodeName());
            return false;
        }
        if (!isValidSliceParams(slice->getInput(0), slice->getParams()))
        {
            LOG_DEBUG(SLICE_NORM, "Skipping slice node {} - params are not valid for pipelining", node->getNodeName());
            return false;
        }
        return true;
    }
    return false;
}

bool HabanaNormsHandler::isValidConsumerAccessPattern(const NodeSet& tileConsumers, const TensorPtr& tileOutput)
{
    if (tileConsumers.size() != 1) return false;
    for (const auto& consumer : tileConsumers)
    {
        const auto& consumerInputs = consumer->getInputs();
        bool isConsumingTileOutput = std::any_of(consumerInputs.begin(), consumerInputs.end(), [&](const TensorPtr& in) {
            return in == tileOutput;
        });
        HB_ASSERT(isConsumingTileOutput, "Tile consumer {} is expected to consume {}", consumer->getNodeName(), tileOutput->getName());
        const auto& accessPattern = consumer->getNodeAccessPattern();
        if (!accessPattern) return false;
        auto tileOutputGranularity = accessPattern->getTensorGranularity(tileOutput);
        if (tileOutputGranularity.geometry[0] != tileOutput->getSizeInElements(0))
        {
            LOG_DEBUG(SLICE_NORM,
                      "The tile output has granularity {} != dim size which is {}, so slicing is not allowed",
                      tileOutputGranularity.geometry[0],
                      tileOutput->getSizeInElements(0));
            return false;
        }
    }
    return true;
}

bool HabanaNormsHandler::isSliceAndTilePatternValidForPipelining(const NodePtr& tile)
{
    if (tile->getGUID().find("tile") != std::string::npos)
    {
        TPCNodePtr tileTpc = std::dynamic_pointer_cast<TPCNode>(tile);
        if (!tileTpc) return false;

        const auto& tileProducer = m_graph.getTensorProducer(tile->getInput(0));
        if (!tileProducer || tileProducer->getNodeType() != Node::TYPE_SLICE || !tileProducer->getNodeAnnotation().originatedFromCguid)
        {
            LOG_DEBUG(SLICE_NORM, "Skipping node {} - it's not slice -> tile valid pattern from cguid", tile->getNodeName());
            return false;
        }
        if (!isValidTileParams(tile->getInput(0), static_cast<ns_TileKernel::ParamsV2*>(tileTpc->getParams())))
        {
            LOG_DEBUG(SLICE_NORM, "Skipping tile node {} - params are not valid for pipelining", tile->getNodeName());
            return false;
        }
        if (!sliceNodePatternWithTile(tileProducer, tile))
        {
            LOG_DEBUG(SLICE_NORM,
                      "Skipping tile node {} - pattern with slice {} is not allowed for pipelining",
                      tile->getNodeName(),
                      tileProducer->getNodeName());
            return false;
        }
        if (!isValidConsumerAccessPattern(m_graph.getNodeConsumers(tile), tile->getOutput(0)))
        {
            LOG_DEBUG(SLICE_NORM,
                      "Skipping tile node {} - pattern with slice {} is not allowed for pipelining because of "
                      "consumer's access pattern",
                      tile->getNodeName(),
                      tileProducer->getNodeName());
            return false;
        }
        return true;
    }
    return false;
}
/*
Original graph:
Pattern 1 - Slice Pattern:
                    +--------+       +-----------+
                    | Slice  +------>| slice_out |
                    |  Node  |       +---+-------+
                    +--------+           |
                         ^               |
                         |               v
   +----------+    +-----+------+    +--------+
   | Producer +--->|  prod_out  +--->|Consumer|
   |  Node    |    +----+-------+    |  Node  |
   +----------+                      +--------+
Pattern 2 - Slice and Tile Pattern:
                +-------+     +----------+    +------+      +----------+
                | Slice +---> |slice_out +--->| Tile +----> | tile_out |
                | Node  |     +----------+    | Node |      +-----+----+
                +-------+                     +------+            |
                    ^                                             |
                    |                                             v
+----------+   +----+-----+    +--------+   +-------------+   +--------+
| Producer +-->| prod_out +--->|Reshape +-->| reshape_out +-->|Consumer|
| Node     |   +----------+    |Node    |   +-------------+   |Node    |
+----------+                   +--------+                     +--------+
Explanation of the problem:
In the current slicing scheme, pattern 1 and 2 above prevent the consumer node (which is a norm node in this use case,
but not verified because it doesn't change the behavior) from being bundled because they create an undirected cycle in
the bundle. Removing the SliceNode (and Tile if for pattern 2) breaks that cycle and can result in bundling of the
consumer, which will result in pipelining -> perf gain. To solve it:
1. Pre slicing  - remove the SliceNode (and Tile if for pattern 2)
2. Post slicing - Add the removed nodes back to the graph as part of the consumer's bundle
*/
void HabanaNormsHandler::findAndRemoveSliceNormNodes()
{
    if (!GCFG_ENABLE_SLICE_NORM_BUNDLING.value())
    {
        return;
    }
    NodeSet nodes = m_graph.getNodes();
    for (auto& node : nodes)
    {
        SliceTilePattern pattern;
        if (isSlicePatternValidForPipelining(node))
        {
            LOG_DEBUG(SLICE_NORM, "Found slice node {} that originated from cguid", node->getNodeName());
            pattern.slice = node;
        }
        else if (isSliceAndTilePatternValidForPipelining(node))
        {
            pattern.slice = m_graph.getTensorProducer(node->getInput(0));
            pattern.tile  = node;
            LOG_DEBUG(SLICE_NORM,
                      "Found slice {} and tile {} that originated from cguid",
                      pattern.slice->getNodeName(),
                      pattern.tile->getNodeName());
        }
        // Remove the pattern from the graph, while keeping a mapping of the pattern nodes from their inputs
        // and outputs.
        if (pattern.slice)
        {
            GraphEditor::removeNode(m_graph, pattern.slice);
            if (pattern.tile) GraphEditor::removeNode(m_graph, pattern.tile);
            m_sliceCollector->registerPattern(pattern);
        }
    }
}