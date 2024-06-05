#pragma once

#include "habana_graph.h"
#include "brain_conf.h"
#include "tile.h"
#include "types.h"
#include "tile.h"
// Add cache line aware memset or memget node to warmp cache for a given tensor, to optimize its cache access.
// Cache warmup is used to optimize a HW bug when partial cache line writes might cause multiple engines write to the
// same cache line. The class adds the memset/memget and reduction nodes, and projects the perforation of the partials
// writes producer to the new warmup node.
class AddCacheWarmup
{
public:
    AddCacheWarmup(HabanaGraph& graph, const TensorPtr& tensor, bool allocInSingleDCore);
    void run();

private:
    HabanaGraph&   m_graph;
    TensorPtr      m_origTensor;
    NodePtr        m_producer;
    unsigned       m_producerOutputIdx;
    bool           m_singleDCore;
    const uint64_t m_singleDCoreIndex = GCFG_SINGLE_DCORE_PREFORATION_WORK_DCORE.value();

    using TensorTile = gc::access_pattern::TensorTile;
    using NodeTile   = gc::access_pattern::NodeTile;
    using Dim        = gc::access_pattern::Dim;

    bool    shouldHandle() const;
    NodePtr createReductionNode(const TensorVector& inputs, const TensorPtr& output);
    NodePtr createCacheWarmupNode(const TensorPtr& warmupOut);
    NodePtr createCacheWarmupNodeByGCFG(const TensorPtr& warmupOut);
    NodePtr createClAwareNode(std::string_view guid, const TensorPtr& output);
    void    modifyGraph(const NodePtr& cacheWarmup, const NodePtr& reduction);

    bool outputOverlapsWithInput() const;
    bool shouldReplaceExistingMemset() const;
    bool isFullyWritten() const;
    bool isMemsetBeforeExec() const;
    bool isFCDSmallerThanCL() const;

    // Returns original <memset, reduction> nodes, which reset output before exec for producer
    std::pair<NodePtr, NodePtr> getExistingMemsetReductionNodes() const;
    void                        setPerforation(NodePtr& cacheWarmup);
    void                        setProducerPerforation();
    std::optional<Dim>          findOptimalPerforationDim(std::function<bool(const Dim nodeDim)> pred) const;
    bool                        canPerforateProducer() const;
    void                        setCacheWarmupPerforation(NodePtr& cacheWarmup);
    void                        resetProducerPerforation();
    bool                        shouldAllocTensorInSingleDcore() const;
    bool                        isPerforatedOnSingleDCore() const;
    bool                        isRmwOutput() const;
    bool                        isOutputAllRequired() const;

    template<typename NodeClass>
    static std::shared_ptr<NodeClass> checkedCast(const NodePtr& n)
    {
        auto casted = std::dynamic_pointer_cast<NodeClass>(n);
        HB_ASSERT_PTR(casted);
        return casted;
    }
    static std::optional<unsigned> getProjectedPerforationDim(const NodePtr&   source,
                                                              const NodePtr&   dest,
                                                              const TensorPtr& sourceTensor,
                                                              const TensorPtr& destTensor);
    static std::vector<unsigned>   getTensorPerforationDims(const NodePtr& node, const TensorPtr& tensor);

    static bool isPerforatedOnFCD(const NodePtr& node, const TensorPtr& output);
    static bool tensorOverlapsOnPerforationDim(const NodePtr& node, Dim nodeDim, const TensorPtr& tensor);

    static NodeTile   getProjectedNodeTile(const NodePtr&    node,
                                           const TensorPtr&  tensor,
                                           const TensorTile& baseTensorTile,
                                           const TensorTile& tensorTileQueriedDcore);
    static TensorTile getDcoreRoiTensorTile(const DcoreROI& dcoreROI, const NodePtr& node, const TensorPtr& tensor);

    static void               projectPerforationByCommonTensor(const NodePtr&   source,
                                                               NodePtr&         dest,
                                                               const TensorPtr& sourceTensor,
                                                               const TensorPtr& destTensor);
    static void               setNodeRoiInSingleDcore(const NodePtr& node, uint64_t singleDCoreIndex);
};