#include "sliced_graph_generator.h"
#include "access_pattern.h"
#include "brain_data.h"

using namespace gc::layered_brain;

// Make sure there is no overlap/offset on the slicing dim - currently not supported.
void SlicedGraphGenerator::validateBundleNodes(const NumSlicesPerBVD& numOfSlicesPerBVD) const
{
    HB_ASSERT(numOfSlicesPerBVD.size() == m_bundleViews->getNumOfBundleViews(), "Expected num of slices per BVD");
    for (const auto& node : m_bundleNodes)
    {
        const auto& nodeAp = node->getNodeAccessPattern();
        HB_ASSERT_PTR(nodeAp);
        for (const auto& tensor : node->getOperands())
        {
            // Shape tensors are skipped here since currently not all types have access-pattern,
            // assuming overlap/offset on shape tensor will be reflected on non-shape tensors
            // as well.
            if (!tensor || tensor->isShapeTensor()) continue;
            const TensorTile&       granularity = nodeAp->getTensorGranularity(tensor);
            const IntersectionTile& overlap     = nodeAp->getTensorOverlap(tensor);
            for (Dim tensorDim = 0; tensorDim < tensor->getDim(); tensorDim++)
            {
                if (numOfSlicesPerBVD[m_bundleViews->getBVDForTensorDim(tensor, tensorDim)] > 1)  // Dim is sliced
                {
                    HB_ASSERT((overlap.geometry[tensorDim] == 0) && (granularity.offset[tensorDim] == 0),
                              "Non-strict mapping is currently not supported (Node {} has overlap ({}) or offset ({}) "
                              "on tensor {} dim {})",
                              node->getNodeName(),
                              overlap.geometry[tensorDim],
                              granularity.offset[tensorDim],
                              tensor->getName(),
                              tensorDim);
                }
            }
        }
    }
}

// Create a slice for the tensor (or reuse if it was already created), and attach it to the sliced node.
// Reduction/BPT data is collected as well to avoid a second iteration later.
void SlicedGraphGenerator::replaceOperandWithSlicedTensor(const NodePtr&   origNode,
                                                          const NodePtr&   slicedNode,
                                                          const BVDCoord&  nodeBVDCoord,
                                                          const TensorPtr& origTensor,
                                                          unsigned         tensorIdx,
                                                          bool             isInput)
{
    HB_ASSERT_PTR(origTensor);

    const auto& tensorBVDCoord = m_bvdCoordsGenerator.projectBVDCoordOnTensor(origTensor, nodeBVDCoord);

    const auto& slicedNodeTile = slicedNode->getNodeAnnotation().sliceROI;
    HB_ASSERT(slicedNodeTile.has_value(), "Missing sliced ROI for node {}", slicedNode->getNodeName());
    const auto& [slicedTensor, sliceOffset] =
        m_slicedTensorGenerator.getSlicedTensor(origNode, slicedNodeTile.value(), origTensor, tensorBVDCoord);

    if (isInput)
    {
        slicedNode->replaceInput(tensorIdx, slicedTensor);
    }
    else
    {
        slicedNode->replaceOutput(tensorIdx, slicedTensor);
        m_reductionHandler.addProducerForTensorSlice(slicedTensor, slicedNode, origNode, tensorIdx);
    }

    m_slicedNodeGenerator.updateTensorSliceOffset(slicedNode, origTensor, slicedTensor, sliceOffset);

    if (m_bptHandler.isBPT(origTensor))
    {
        m_bptHandler.addTensorSlice(origTensor, slicedTensor, sliceOffset);
    }
}

void SlicedGraphGenerator::createSlicedNodes()
{
    const auto& numOfSlicesPerBVD = m_bvdCoordsGenerator.getNumOfSlicesPerBVD();
    validateBundleNodes(numOfSlicesPerBVD);
    for (const auto& node : m_bundleNodes)
    {
        for (const auto& nodeBVDCoord : m_bvdCoordsGenerator.getBVDCoordsForNode(node))
        {
            auto slicedNode = m_slicedNodeGenerator.getSlicedNode(node, nodeBVDCoord);
            // save node bvd coord to be used later to project producer node bvd coords
            // on reduction inputs
            m_slicedNodeToCoord.emplace(slicedNode, nodeBVDCoord);
            for (auto i = 0; i < node->getNumInputs(); i++)
            {
                if (!node->getInput(i)) continue;
                replaceOperandWithSlicedTensor(node, slicedNode, nodeBVDCoord, node->getInput(i), i, true);
            }

            for (auto i = 0; i < node->getNumOutputs(); i++)
            {
                if (!node->getOutput(i)) continue;
                replaceOperandWithSlicedTensor(node, slicedNode, nodeBVDCoord, node->getOutput(i), i, false);
            }

            m_slicedNodeGenerator.addAuxTensors(node, slicedNode);

            m_slicedNodes.emplace(slicedNode);
        }
    }

    const auto& aggregationNodesForBPTs = m_bptHandler.createForkAndJoinNodes();
    m_slicedNodes.insert(aggregationNodesForBPTs.begin(), aggregationNodesForBPTs.end());

    const auto& reductionNodes = m_reductionHandler.createReductionNodes();
    m_slicedNodes.insert(reductionNodes.begin(), reductionNodes.end());
}
