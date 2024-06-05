#pragma once

#include "brain_data.h"
#include "bvd_coords_generator.h"
#include "bpt_handler.h"
#include "reduction_handler.h"
#include "sliced_node_generator.h"
#include "sliced_tensor_generator.h"
#include "types.h"

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
class SlicedGraphGenerator
{
protected:
    explicit SlicedGraphGenerator(const NodeVector&             bundleNodes,
                                  const BundleViewContainerPtr& bundleViews,
                                  const BVDCoordsGenerator&     bvdCoordsGenerator,
                                  const SlicedTensorGenerator&  slicedTensorGenerator,
                                  const SlicedNodeGenerator&    slicedNodeGenerator,
                                  const BPTHandler&             bptHandler,
                                  const ReductionHandler&       reductionHandler)
    : m_bundleNodes {bundleNodes},
      m_bundleViews(bundleViews),
      m_bvdCoordsGenerator(bvdCoordsGenerator),
      m_slicedTensorGenerator(slicedTensorGenerator),
      m_slicedNodeGenerator(slicedNodeGenerator),
      m_bptHandler(bptHandler),
      m_reductionHandler(reductionHandler)
    {
    }

    virtual ~SlicedGraphGenerator() = default;

    void         createSlicedNodes();
    void         validateBundleNodes(const NumSlicesPerBVD& numOfSlicesPerBVD) const;
    virtual void replaceOperandWithSlicedTensor(const NodePtr&   origNode,
                                                const NodePtr&   slicedNode,
                                                const BVDCoord&  nodeBVDCoord,
                                                const TensorPtr& origTensor,
                                                unsigned         tensorIdx,
                                                bool             isInput);

    const NodeVector             m_bundleNodes;
    const BundleViewContainerPtr m_bundleViews;

    BVDCoordsGenerator                    m_bvdCoordsGenerator;
    SlicedTensorGenerator                 m_slicedTensorGenerator;
    SlicedNodeGenerator                   m_slicedNodeGenerator;
    BPTHandler                            m_bptHandler;
    ReductionHandler                      m_reductionHandler;
    std::unordered_map<NodePtr, BVDCoord> m_slicedNodeToCoord;  // sliced node to its node bvd coords
    NodeSet                     m_slicerReductions;  // reduction nodes with inputs that BVD coords can be projected on
    NodeSet                     m_slicedNodes;
};

}  // namespace gc::layered_brain