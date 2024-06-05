#pragma once

#include <habana_graph.h>
#include <memory>

#include "bfs_scheduler.h"
#include "bundle_plane_node.h"

//
// OG === Original Graph
// BP === Bundle Plane
//

class BundlePlane
{
public:
    // using an efficient method for building when constructed using bundle info annotations
    BundlePlane(HabanaGraph& origGraph, bool useAnnotations = false, std::function<bool(const NodePtr&)> predicate = nullptr);

    BundlePlane();

    // Create a bundle for a previously unbundled node
    // Return true means success.
    bool addNodeToBundle(const NodePtr& ogNode, const BundleInfo& bundleInfo = BundleInfo(0, BundleType::MME));

    // fuse all given nodes (that were not part of any bundle) into a single bundle.
    void createBundleFromNodes(const NodeVector& ogNodes, const BundleInfo& bundleInfo);
    // remove node from a bundle
    void unbundleNode(const NodePtr& node);
    // dismantle some bundle into it's original nodes.
    void removeBundle(const NodePtr& ogNode);

    // Can the candidate node be safely fused into a bundle with the given accepted nodes?
    bool
    validateCandidate(const NodePtr& candidate, const TensorPtr& stitchedTensor, const NodeVector& acceptedNodes) const;

    uint64_t getNumberOfPaths(const NodePtr& ogSource, const NodePtr& ogTarget) const;

    const Graph*     getBundlePlaneGraph() const { return m_bpGraph.get(); }
    const bool       hasBundlePlaneRepresentation(const NodePtr& ogNode) const;
    const NodePtr&   getBundlePlaneRepresentation(const NodePtr& ogNode) const;
    const TensorPtr& getBundlePlaneRepresentation(const TensorPtr& ogTensor) const;

    static BundlePlaneNode* downcastToBundlePlaneNode(const NodePtr& bundlePlaneRepresentation);

    // add regular node to BP graph
    void addNode(const NodePtr& node);

    // remove regular node from BP graph.
    void removeNode(const NodePtr& node, NodePtr newProducer = nullptr);

    // replace oldNode for newNode - does not change tensor connectivity
    void replaceSemanticNodes(const NodePtr& oldNode, const NodePtr& newNode);

    void freezeGraph();
    void unfreezeGraph();
    void addRelationshipInBP(const TensorPtr& origTensor, const NodePtr& origProducer, const NodePtr& origConsumer);

    // remove relationship in BP graph
    void removeRelationshipInBP(const TensorPtr& origTensor, const NodePtr& Node, Node::eParamUsage usage);


    static NodeList createOrigNodesScheduleFromBpgSchedule(const NodeList& bpgSchedule);

private:
    std::shared_ptr<Graph> m_bpGraph;

    std::unordered_map<synNodeId, NodePtr>  m_OGNodeToBPNode;
    std::unordered_map<uint32_t, TensorPtr> m_OGTensorToBPTensor;
    std::unordered_map<unsigned, NodePtr>   m_bundleIdxToBundleNode;
    bool                                    m_freeze = false;
    std::function<bool(const NodePtr&)>     m_predicate;

    // BP graph building methods
    class BundlePlaneAccumulator;
    // efficient method of building a BP graph from a graph that has already been "bundlized"
    void                            buildFromExistingAnnotations(HabanaGraph& origGraph);
    void                            buildInitialBPGraph(const HabanaGraph& graph);
    void                            addBPNodeFromOGNode(const NodePtr& ogNode);
    TensorPtr                       insertBPTensor(const TensorPtr& ogTensor);
    bool                            addNewBundle(const NodePtr& ogNode, const BundleInfo& bundleInfo);
    void                            fuseBundles(const NodeVector& ogNodes, const BundleInfo& bundleInfo);
    std::pair<TensorSet, TensorSet> getExternalTensors(const NodeVector& bpNodes) const;
    static bool                     areSameBundle(const NodePtr& n1, const NodePtr& n2);
    static bool                     areSameBundle(const Settable<BundleInfo>& info1, const Settable<BundleInfo>& info2);
    static bool                     isPartOfBundle(const NodePtr& n);
    bool                            validateBPGraph(const HabanaGraph& origGraph) const;
    TensorSet                       getBundlePlaneRepresentation(const TensorVector& ogTensors) const;
    static NodeList                 getOrigScheduleForBpNode(const std::shared_ptr<BundlePlaneNode>& bpNode);
    void                            tracePrint() const;
    void                            tracePrint(const HabanaGraph& origGraph) const;
};

// RAII object to add a BP graph to a HabanaGraph for the duration of the ctx scope/lifetime.
class BPGraphContext
{
public:
    static constexpr bool CONSTRUCT_BPG_FROM_ANNOTATIONS = true;
    BPGraphContext(HabanaGraph& graph) : m_graph(graph)
    {
        HB_ASSERT(nullptr == m_graph.getBPGraph(),
                  "Unexpected bundle plane context creation for graph with existing BP-graph");
        m_graph.constructBPGraph(CONSTRUCT_BPG_FROM_ANNOTATIONS);
    }
    virtual ~BPGraphContext() { m_graph.discardBPGraph(); }
    HabanaGraph& m_graph;
};
