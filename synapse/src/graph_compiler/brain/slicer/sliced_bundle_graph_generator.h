#pragma once

#include "sliced_graph_generator.h"
#include "habana_graph.h"

namespace gc::layered_brain
{
// The sliced graph generator is responsible to create a sliced representation of the graph according to the
// bundle-views slicing, by traversing the bundle nodes to generate all node slices (slice per ISR)
// and then create and connect all tensors slices (slice per ISMR).
// Bundle inputs need to be connected to their slices via fork (read TensorView) and Bundle outputs
// need to be connected to their slices via join (write TensorView).
// Multiple ISRs with the same output ISMR should go through reduction (doesn't have to be an output BPT).
// In addition to the sliced graph, this component generates the following data to be used by next passes:
// 1. Annotate each sliced node with its original big node (in NodeAnnotation).
// 2. Annotate each sliced tensor with its original big tensor (in TensorAnnotation).
// 3. Sliced tensor by big tensor BVD coord (in LayeredBrainData).
// 4. Output BPTs (in LayeredBrainData).
class SlicedBundleGraphGenerator : public SlicedGraphGenerator
{
public:
    using BaseClass            = SlicedGraphGenerator;
    using CoordPerSlicedTensor = std::map<TensorPtr, BVDCoord>;

    SlicedBundleGraphGenerator(const HabanaGraph&            origGraph,
                               const BundleIdx               bundleIdx,
                               const NodeVector&             bundleNodes,
                               const BundleViewContainerPtr& bundleViews,
                               const StrategyPtr&            slicingStrategy,
                               bool                          dryRun = false)
    : BaseClass(
          bundleNodes,
          bundleViews,
          BVDCoordsGenerator(bundleViews, slicingStrategy, dryRun),
          SlicedTensorGenerator(bundleIdx),
          SlicedNodeGenerator(bundleIdx, bundleViews, slicingStrategy),
          BPTHandler(origGraph, bundleIdx, bundleNodes),
          ReductionHandler(bundleIdx, getRequireCastNodes(slicingStrategy), getRequireMemsetNodes(slicingStrategy))),
      m_graph(origGraph),
      m_bundleIdx(bundleIdx),
      m_strategy(slicingStrategy),
      m_dryRun(dryRun)
    {
    }

    HabanaGraphPtr createSlicedGraph();

private:
    void           validateBundleNodes(const NumSlicesPerBVD& numOfSlicesPerBVD) const;
    HabanaGraphPtr createEmptySlicedGraph() const;
    void           replaceOperandWithSlicedTensor(const NodePtr&   origNode,
                                                  const NodePtr&   slicedNode,
                                                  const BVDCoord&  nodeBVDCoord,
                                                  const TensorPtr& origTensor,
                                                  unsigned         tensorIdx,
                                                  bool             isInput) override;
    void           addSlicedNodesToSlicedGraph() const;
    void           updateBundleData() const;
    BundleEngine   getBundleEngine() const;

    void      insulateSlicedGraph();
    TensorPtr getBPTReplacement(const TensorPtr& origBPT);

    // When MME requires high precision reduction and the original output datatype is not 32b,
    // need to add intermediate f32 output for the MME and cast it back to original type.
    NodeSet getRequireCastNodes(const StrategyPtr& strategy) const;
    NodeSet getRequireMemsetNodes(const StrategyPtr& strategy) const;

    // Reduction nodes are generated after the slices and their coords.
    // Moreover, reduction inputs are slice clones and as such aren't mapped to any BVD coords.
    // For each reduction input, project the node BVD coord of it's producer on it
    // and update the slicer->coord mapping.
    void projectCoordsOnReductionInputs();

    // Cache supported slicer added reductions for which we know the node BVD coords
    // of each input producer.
    // This requirement allows us to later project the reduction input producer bvd coords
    // on the reduction input tensor.
    void cacheSlicerReductions();

    const HabanaGraph& m_graph;
    const BundleIdx    m_bundleIdx;
    const StrategyPtr  m_strategy;
    const bool         m_dryRun;

    HabanaGraphPtr       m_slicedGraph;
    CoordPerSlicedTensor m_slicesCoords;
};

}  // namespace gc::layered_brain