#pragma once
#include "habana_graph.h"

struct SliceTilePattern
{
    SliceTilePattern(NodePtr s = nullptr, NodePtr t = nullptr) : slice(s), tile(t) {};
    NodePtr slice;
    NodePtr tile;
};

class PatternNodesCollector
{
public:
    typedef TensorToItemOrderedMap<NodeSet> SliceToNodesMap;
    struct PipelinedSlice
    {
        PipelinedSlice() {};
        PipelinedSlice(const TensorPtr& in, const TensorPtr& out, const NodePtr& slice)
        : newInput(in), newOutput(out), originalSliceNode(slice) {};
        TensorPtr newInput;
        TensorPtr newOutput;
        NodePtr   originalSliceNode;
    };

    struct PipelinedPattern
    {
        PipelinedPattern() {};
        PipelinedPattern(PipelinedSlice s) : pipelinedSlice(s) {};
        PipelinedSlice    pipelinedSlice;
        Settable<NodePtr> pipelinedTile;
    };

    PatternNodesCollector() {};
    std::vector<PipelinedPattern> getPatternNodesMetadata(HabanaGraph& g);
    void                          registerPattern(const SliceTilePattern& node);
    NodePtr                       getTileNodeFromTensor(const TensorPtr& t);
    NodeSet                       getSliceNodesFromInput(const TensorPtr& t);
    NodePtr                       getSliceNodeFromOutput(const TensorPtr& t);

protected:
    std::optional<PatternNodesCollector::PipelinedPattern> getPipelinedNodes(HabanaGraph& g, const NodePtr& consumer);
    std::optional<PatternNodesCollector::PipelinedPattern>
                   findSliceAndTilePatternFromInput(HabanaGraph& g, const NodePtr& consumer, const TensorPtr& in);
    SliceToNodesMap m_tileOutToNode;
    SliceToNodesMap m_sliceOutToNode;
    SliceToNodesMap m_sliceInToNodes;
};

using PatternNodesCollectorPtr = std::shared_ptr<PatternNodesCollector>;

class HabanaNormsHandler
{
public:
    explicit HabanaNormsHandler(HabanaGraph& graph, PatternNodesCollectorPtr collector)
    : m_graph(graph), m_sliceCollector(collector) {};
    void findAndRemoveSliceNormNodes();
    bool handleRemovedSliceNormNodes();

private:
    bool                isSlicePatternValidForPipelining(const NodePtr& node);
    bool                isSliceAndTilePatternValidForPipelining(const NodePtr& tile);
    bool                isValidSliceParams(const TensorPtr& input, const SliceNode::SliceNodeStaticParams& params);
    bool                isValidTileParams(const TensorPtr& input, const ns_TileKernel::ParamsV2* params);
    bool                existsCommonConsumerForSliceTensors(const NodePtr& node);
    bool                sliceNodePatternWithTile(const NodePtr& slice, const NodePtr& tile);
    bool                validateHandleSliceNodes();
    NodePtr             addUpdatedSliceNode(const TensorPtr& in, const TensorPtr& out, NodePtr origSliceNode);
    NodePtr             addUpdatedTileNode(const TensorPtr& in, NodePtr origTileNode);
    SliceNode::SliceNodeStaticParams getUpdatedSliceParams(const NodePtr& node, const TensorPtr& newOut) const;
    bool                isSliceProducer(const NodePtr& producer);
    bool isReshapeValidForPattern(const NodePtr& reshapeNode, const NodePtr& slice, const NodePtr& tileNode);
    void addUnslicedPatternToGraph(const TensorPtr&         newSliceInput,
                                   const NodePtr&           origSliceNode,
                                   const Settable<NodePtr>& pipelinedTileOpt);
    void addSlicedPatternToGraph(const TensorPtr&         newSliceInput,
                                 const TensorPtr&         newSliceOutput,
                                 const NodePtr&           origSliceNode,
                                 const Settable<NodePtr>& pipelinedTileOpt);
    bool isValidConsumerAccessPattern(const NodeSet& tileConsumers, const TensorPtr& tileOutput);
    std::pair<Settable<BundleInfo>, std::string> getUpdatedBundleInfoAndNodeName(const TensorPtr& t,
                                                                       const NodePtr&   origNode) const;
    void addCtrlDepToSliceProducer(const NodePtr& slice, const TensorPtr producerOut);

    HabanaGraph&             m_graph;
    PatternNodesCollectorPtr m_sliceCollector;
};
