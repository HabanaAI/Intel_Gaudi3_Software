#include "shape_node.h"
#include "node.h"
#include "tpc_node.h"
#include "dynamic_patch_point_optimizer.h"
#include "recipe_allocator.h"

static void fillNodeDbTensor(tensor_info_t& tensor, const TensorShape& shape);

ShapeNode& ShapeNode::operator=(const ShapeNode& other)
{
    m_patchPoints = other.m_patchPoints;
    return *this;
}

ShapeNode& ShapeNode::operator=(ShapeNode&& other)
{
    m_patchPoints = std::move(other.m_patchPoints);
    return *this;
}

void ShapeNode::addPatchPoint(DynamicPatchPointPtr patchPoint)
{
    if (patchPoint == nullptr || patchPoint->getRoi() == nullptr || m_connectedNode.getPhysicalRois() == nullptr)
    {
        LOG_ERR(GC, "{} Invalid nullptr", HLLOG_FUNC);
        return;
    }

    const NodeROI* ppRoi = patchPoint->getRoi();

    const auto& physicalRoiList = *m_connectedNode.getPhysicalRois();

    if (!m_addPatchPointCache.initialized)
    {
        m_addPatchPointCache.m_lastIterator = physicalRoiList.begin();
        m_addPatchPointCache.m_lastIndex    = 0;
        m_addPatchPointCache.initialized    = true;
    }

    // start from the place we have left at
    auto curRoiIter = m_addPatchPointCache.m_lastIterator;
    auto roiIndex   = m_addPatchPointCache.m_lastIndex;

    for (; curRoiIter != physicalRoiList.end(); ++curRoiIter)
    {
        if (&*curRoiIter == ppRoi)
        {
            break;
        }
        ++roiIndex;
    }

    // if not found, start again from the beginning
    if (curRoiIter == physicalRoiList.end())
    {
        curRoiIter = physicalRoiList.begin();
        roiIndex   = 0;

        for (; curRoiIter != m_addPatchPointCache.m_lastIterator; ++curRoiIter)
        {
            if (&*curRoiIter == ppRoi)
            {
                break;
            }
            roiIndex++;
        }
    }

    if (&*curRoiIter == ppRoi)
    {
        patchPoint->setRoiIndex(roiIndex);
    }

    m_patchPoints.push_back(patchPoint);
    m_addPatchPointCache.m_lastIterator = curRoiIter;
    m_addPatchPointCache.m_lastIndex    = roiIndex;
}

static unsigned make_permutations(RecipeAllocator*                                       pRecipeAlloc,
                                  sif_permutation_t*&                                    permutations,
                                  const std::vector<tpc_lib_api::NodeTensorPermutation>& gcPermutations)
{
    auto size = gcPermutations.size();
    if (size > 0)
    {
        permutations = (sif_permutation_t*)pRecipeAlloc->allocate(size * sizeof(sif_permutation_t));
        std::copy(std::begin(gcPermutations), std::end(gcPermutations), permutations);
    }
    else
    {
        permutations = nullptr;
    }

    return size;
}

// all the memory allocated here is freed in runtime.
void ShapeNode::serialize(const ShapePlaneInfoContainer& shapePlaneInfoContainer,
                          shape_plane_node_t&            serializeNode,
                          RecipeAllocator*               pRecipeAlloc)
{
    const std::list<NodeROI>* rois = m_connectedNode.getPhysicalRois();
    HB_ASSERT_PTR(rois);

    serializeNode.node_match_output_tensors_nr = 0;

    const TensorVector& inTensors  = m_connectedNode.getInputs();
    serializeNode.input_tensors_nr = std::count_if(inTensors.begin(), inTensors.end(), [](const pTensor& t) {
        return t != nullptr && !t->isAuxTensor();
    });
    serializeNode.input_tensors = (uint32_t*)pRecipeAlloc->allocate(serializeNode.input_tensors_nr * sizeof(uint32_t));

    size_t inputSerializeIndex = 0;
    for (const pTensor& tensor : inTensors)
    {
        if (tensor != nullptr && !tensor->isAuxTensor())
        {
            serializeNode.input_tensors[inputSerializeIndex++] =
                shapePlaneInfoContainer.getTensorIndexByID(tensor->getId());
        }
    }

    const TensorVector& outTensors = m_connectedNode.getOutputs();
    serializeNode.output_tensors_nr =
        std::count_if(outTensors.begin(), outTensors.end(), [](const pTensor& t) { return t != nullptr; });
    serializeNode.output_tensors =
        (uint32_t*)pRecipeAlloc->allocate(serializeNode.output_tensors_nr * sizeof(uint32_t));

    size_t outputSerializeIndex = 0;
    for (const pTensor& tensor : outTensors)
    {
        if (tensor != nullptr)
        {
            serializeNode.output_tensors[outputSerializeIndex++] =
                shapePlaneInfoContainer.getTensorIndexByID(tensor->getId());
        }
    }

    // This is a place holder for the runtime.
    memset(serializeNode.nodeData, 0, sizeof(serializeNode.nodeData));

    auto* tpcNode = dynamic_cast<MultiSifNodeInfoHelper*>(&m_connectedNode);

    if (tpcNode != nullptr && tpcNode->getMultiSifInfo() != nullptr)
    {
        auto multiSifInfo = tpcNode->getMultiSifInfo();

        serializeNode.basic_nodes_nr = multiSifInfo->m_nodes.size();
        serializeNode.basic_nodes    = (shape_plane_basic_node_t*)pRecipeAlloc->allocate(serializeNode.basic_nodes_nr *
                                                                                      sizeof(shape_plane_basic_node_t));

        serializeNode.node_db_tensors_nr = multiSifInfo->m_internalTensorsNr;
        serializeNode.node_db_tensors =
            (tensor_info_t*)pRecipeAlloc->allocate(multiSifInfo->m_internalTensorsNr * sizeof(tensor_info_t));

        for (uint32_t i = 0; i < serializeNode.basic_nodes_nr; ++i)
        {
            auto& curMultiSifInfoNode = multiSifInfo->m_nodes[i];

            // serialize a non-multi-sif node to basic node
            size_t        paramsSize = curMultiSifInfoNode.m_nodeParams.size();
            SifNodeParams params     = curMultiSifInfoNode.m_nodeParams.data();

            serializeNode.basic_nodes[i].sif_id      = curMultiSifInfoNode.m_sifID;
            serializeNode.basic_nodes[i].sif_version = curMultiSifInfoNode.m_sifVersion;

            serializeNode.basic_nodes[i].sif_params_nr = paramsSize;
            if (paramsSize > 0)
            {
                serializeNode.basic_nodes[i].sif_params =
                    (uint8_t*)pRecipeAlloc->allocate(paramsSize * sizeof(uint8_t));
                memcpy(serializeNode.basic_nodes[i].sif_params, params, paramsSize);
            }
            else
            {
                serializeNode.basic_nodes[i].sif_params = nullptr;
            }

            serializeNode.basic_nodes[i].input_tensors_db =
                (shape_plane_basic_node_t::EShapePlanceTensorDb*)pRecipeAlloc->allocate(
                    serializeNode.input_tensors_nr * sizeof(shape_plane_basic_node_t::EShapePlanceTensorDb));
            serializeNode.basic_nodes[i].input_tensors_nr = curMultiSifInfoNode.m_inputs.size();
            serializeNode.basic_nodes[i].input_tensors =
                (uint32_t*)pRecipeAlloc->allocate(curMultiSifInfoNode.m_inputs.size() * sizeof(uint32_t));
            for (uint32_t inp = 0; inp < curMultiSifInfoNode.m_inputs.size(); ++inp)
            {
                auto index = curMultiSifInfoNode.m_inputs[inp].m_index;
                if (curMultiSifInfoNode.m_inputs[inp].m_isInternal)
                {
                    serializeNode.basic_nodes[i].input_tensors[inp]    = index;
                    serializeNode.basic_nodes[i].input_tensors_db[inp] = shape_plane_basic_node_t::NODE_TENSOR_DB;

                    fillNodeDbTensor(serializeNode.node_db_tensors[index], curMultiSifInfoNode.m_inputs[inp].m_shape);
                }
                else if (curMultiSifInfoNode.m_inputs[inp].m_takeFromOutput)
                {
                    serializeNode.basic_nodes[i].input_tensors[inp]    = serializeNode.output_tensors[index];
                    serializeNode.basic_nodes[i].input_tensors_db[inp] = shape_plane_basic_node_t::GRAPH_TENSOR_DB;
                }
                else
                {
                    serializeNode.basic_nodes[i].input_tensors[inp]    = serializeNode.input_tensors[index];
                    serializeNode.basic_nodes[i].input_tensors_db[inp] = shape_plane_basic_node_t::GRAPH_TENSOR_DB;
                }
            }

            serializeNode.basic_nodes[i].output_tensors_db =
                (shape_plane_basic_node_t::EShapePlanceTensorDb*)pRecipeAlloc->allocate(
                    serializeNode.output_tensors_nr * sizeof(shape_plane_basic_node_t::EShapePlanceTensorDb));

            serializeNode.basic_nodes[i].output_tensors_nr = curMultiSifInfoNode.m_outputs.size();
            serializeNode.basic_nodes[i].output_tensors =
                (uint32_t*)pRecipeAlloc->allocate(curMultiSifInfoNode.m_outputs.size() * sizeof(uint32_t));
            for (uint32_t outp = 0; outp < curMultiSifInfoNode.m_outputs.size(); ++outp)
            {
                auto index = curMultiSifInfoNode.m_outputs[outp].m_index;
                if (curMultiSifInfoNode.m_outputs[outp].m_isInternal)
                {
                    serializeNode.basic_nodes[i].output_tensors[outp]    = index;
                    serializeNode.basic_nodes[i].output_tensors_db[outp] = shape_plane_basic_node_t::NODE_TENSOR_DB;
                    // no need to fill node_db_tensors[index], it was done when handling inputs
                }
                else
                {
                    serializeNode.basic_nodes[i].output_tensors[outp]    = serializeNode.output_tensors[index];
                    serializeNode.basic_nodes[i].output_tensors_db[outp] = shape_plane_basic_node_t::GRAPH_TENSOR_DB;
                }
            }

            memset(serializeNode.basic_nodes[i].node_name, 0, MAX_SHAPE_PLANE_NODE_NAME_LEN);
            strncpy(serializeNode.basic_nodes[i].node_name,
                    curMultiSifInfoNode.m_nodeName.c_str(),
                    MAX_SHAPE_PLANE_NODE_NAME_LEN - 1);

            serializeNode.basic_nodes[i].input_permutations_nr =
                make_permutations(pRecipeAlloc,
                                  serializeNode.basic_nodes[i].input_permutations,
                                  curMultiSifInfoNode.m_inputPermutations);
        }
    }
    else
    {
        // serialize a non-multi-sif node to basic node
        size_t        paramsSize = m_connectedNode.getShapeInferenceFunctionUserParamsSize();
        SifNodeParams params     = m_connectedNode.getShapeInferenceFunctionUserParams();

        serializeNode.basic_nodes_nr = 1;
        serializeNode.basic_nodes = (shape_plane_basic_node_t*)pRecipeAlloc->allocate(sizeof(shape_plane_basic_node_t));
        serializeNode.basic_nodes[0].sif_id      = m_connectedNode.getShapeInferenceFunctionId();
        serializeNode.basic_nodes[0].sif_version = m_connectedNode.getShapeInferenceFunctionVersion();

        serializeNode.basic_nodes[0].sif_params_nr = paramsSize;
        if (paramsSize > 0)
        {
            serializeNode.basic_nodes[0].sif_params = (uint8_t*)pRecipeAlloc->allocate(paramsSize * sizeof(uint8_t));
            memcpy(serializeNode.basic_nodes[0].sif_params, params, paramsSize);
        }
        else
        {
            serializeNode.basic_nodes[0].sif_params = nullptr;
        }

        serializeNode.basic_nodes[0].input_tensors_nr = serializeNode.input_tensors_nr;
        serializeNode.basic_nodes[0].input_tensors =
            (uint32_t*)pRecipeAlloc->allocate(serializeNode.input_tensors_nr * sizeof(uint32_t));
        std::copy(serializeNode.input_tensors,
                  serializeNode.input_tensors + serializeNode.input_tensors_nr,
                  serializeNode.basic_nodes[0].input_tensors);

        serializeNode.basic_nodes[0].output_tensors_nr = serializeNode.output_tensors_nr;
        serializeNode.basic_nodes[0].output_tensors =
            (uint32_t*)pRecipeAlloc->allocate(serializeNode.output_tensors_nr * sizeof(uint32_t));
        std::copy(serializeNode.output_tensors,
                  serializeNode.output_tensors + serializeNode.output_tensors_nr,
                  serializeNode.basic_nodes[0].output_tensors);

        serializeNode.basic_nodes[0].input_tensors_db =
            (shape_plane_basic_node_t::EShapePlanceTensorDb*)pRecipeAlloc->allocate(
                serializeNode.input_tensors_nr * sizeof(shape_plane_basic_node_t::EShapePlanceTensorDb));

        std::fill(serializeNode.basic_nodes[0].input_tensors_db,
                  serializeNode.basic_nodes[0].input_tensors_db + serializeNode.input_tensors_nr,
                  shape_plane_basic_node_t::GRAPH_TENSOR_DB);

        serializeNode.basic_nodes[0].output_tensors_db =
            (shape_plane_basic_node_t::EShapePlanceTensorDb*)pRecipeAlloc->allocate(
                serializeNode.output_tensors_nr * sizeof(shape_plane_basic_node_t::EShapePlanceTensorDb));
        std::fill(serializeNode.basic_nodes[0].output_tensors_db,
                  serializeNode.basic_nodes[0].output_tensors_db + serializeNode.output_tensors_nr,
                  shape_plane_basic_node_t::GRAPH_TENSOR_DB);

        memset(serializeNode.basic_nodes[0].node_name, 0, MAX_SHAPE_PLANE_NODE_NAME_LEN);
        strncpy(serializeNode.basic_nodes[0].node_name,
                m_connectedNode.getNodeName().c_str(),
                MAX_SHAPE_PLANE_NODE_NAME_LEN - 1);

        serializeNode.basic_nodes[0].input_permutations_nr =
            make_permutations(pRecipeAlloc,
                              serializeNode.basic_nodes[0].input_permutations,
                              m_connectedNode.getInputPermutations());
        serializeNode.node_db_tensors    = nullptr;
        serializeNode.node_db_tensors_nr = 0;
    }

    // Post SIF updates
    serializeNode.node_match_output_tensors_nr = m_postSifUpdate.size();
    serializeNode.output_src_tensors           = nullptr;
    serializeNode.output_dst_tensors           = nullptr;
    if (serializeNode.node_match_output_tensors_nr > 0)
    {
        serializeNode.output_src_tensors =
            (uint32_t*)pRecipeAlloc->allocate(serializeNode.node_match_output_tensors_nr * sizeof(uint32_t));
        serializeNode.output_dst_tensors =
            (uint32_t*)pRecipeAlloc->allocate(serializeNode.node_match_output_tensors_nr * sizeof(uint32_t));
    }

    size_t matchingOutputIndex = 0;
    for (const auto& srcDstTensors : m_postSifUpdate)
    {
        serializeNode.output_src_tensors[matchingOutputIndex] =
            shapePlaneInfoContainer.getTensorIndexByID(srcDstTensors.first->getId());
        serializeNode.output_dst_tensors[matchingOutputIndex] =
            shapePlaneInfoContainer.getTensorIndexByID(srcDstTensors.second->getId());
        matchingOutputIndex++;
    }

    serializeNode.activation_rois_nr = rois->size();
    serializeNode.activation_rois =
        (roi_info_t*)pRecipeAlloc->allocate(serializeNode.activation_rois_nr * sizeof(roi_info_t));

    size_t currRoi = 0;
    for (const NodeROI& roi : *rois)
    {
        serializeRoi(roi, serializeNode.activation_rois[currRoi++], pRecipeAlloc);
    }

    serializeNode.node_patch_points_nr = m_patchPoints.size();
    serializeNode.node_patch_points =
        (sm_patch_point_t*)pRecipeAlloc->allocate(m_patchPoints.size() * sizeof(sm_patch_point_t));

    DynamicPatchPointOptimizer ppOptimizer;
    ppOptimizer.optimizePatchPoints(m_patchPoints);
    for (size_t i = 0; i < m_patchPoints.size(); i++)
    {
        m_patchPoints[i]->serialize(serializeNode.node_patch_points + i, pRecipeAlloc);
    }

    // Serialize debug info
    memset(serializeNode.node_name, 0, MAX_SHAPE_PLANE_NODE_NAME_LEN);
    strncpy(serializeNode.node_name, m_connectedNode.getNodeName().c_str(), MAX_SHAPE_PLANE_NODE_NAME_LEN - 1);
}

void ShapeNode::serializeRoi(const NodeROI& roi, roi_info_t& serializeRoi, RecipeAllocator* pRecipeAlloc)
{
    serializeInOutRoi(roi.inputRois, serializeRoi.roi_in_tensor_nr, serializeRoi.roi_in_tensors, pRecipeAlloc);
    serializeInOutRoi(roi.outputRois, serializeRoi.roi_out_tensor_nr, serializeRoi.roi_out_tensors, pRecipeAlloc);

    memset(serializeRoi.index_space_offset, 0x00, sizeof(serializeRoi.index_space_offset));
    memset(serializeRoi.index_space_size, 0x00, sizeof(serializeRoi.index_space_size));

    if (m_connectedNode.getNodeType() == Node::TYPE_USER)
    {
        for (int d = 0; d < MAX_DIMENSIONS_NUM; d++)
        {
            serializeRoi.index_space_offset[d] = m_connectedNode.getNodeAnnotation().baseOffset[d] + roi.baseOffset[d];
            serializeRoi.index_space_size[d]   = roi.size[d];
        }
    }
}

void ShapeNode::serializeInOutRoi(const TensorROIVector& tensorRois,
                                  uint32_t&              size,
                                  tensor_roi_t*&         convertedRois,
                                  RecipeAllocator*       pRecipeAlloc)
{
    size          = tensorRois.size();
    convertedRois = (tensor_roi_t*)pRecipeAlloc->allocate(size * sizeof(tensor_roi_t));

    for (int i = 0; i < size; i++)
    {
        tensor_roi_t&    serializeTensorRoi = convertedRois[i];
        const TensorROI& tensorRoi          = tensorRois[i];

        castNcopy(serializeTensorRoi.roi_offset_dims, tensorRoi.getLayout().m_baseOffset, MAX_DIMENSIONS_NUM);
        castNcopy(serializeTensorRoi.roi_size_dims, tensorRoi.getLayout().m_size.data(), MAX_DIMENSIONS_NUM);
    }
}

void ShapeNode::addPostSifUpdate(const TensorPtr& src, const TensorPtr& dst)
{
    // The easy way without making m_postSifUpdate into set or map.
    // An associative container would have worse performance for sizes typical for this data.
    using Tpair = std::pair<TensorPtr, TensorPtr>;
    if (std::find(m_postSifUpdate.begin(), m_postSifUpdate.end(), Tpair {src, dst}) == m_postSifUpdate.end())
    {
        m_postSifUpdate.emplace_back(src, dst);
    }
}

static void fillNodeDbTensor(tensor_info_t& tensor, const TensorShape& shape)
{
    tensor.data_type = 0;  // XXX need to fill real data type?

    memset(&tensor, 0, sizeof(tensor));

    tensor.infer_info.geometry.dims = shape.getDim();
    SizeArray maxSize               = shape.getMaxSizes();
    SizeArray minSize               = shape.getMinSizes();
    std::copy(maxSize.begin(), maxSize.end(), tensor.max_dims);
    std::copy(minSize.begin(), minSize.end(), tensor.min_dims);

    tensor.tensor_db_index = static_cast<uint32_t>(-1);  // No global DB index!
    tensor.tensor_type     = tensor_info_t::INTERNAL_TENSOR;
}
