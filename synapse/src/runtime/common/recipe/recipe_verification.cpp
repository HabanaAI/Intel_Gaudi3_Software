#include "recipe_verification.hpp"
#include "defenders.h"
#include "habana_global_conf_runtime.h"
#include "syn_singleton.hpp"
#include "synapse_runtime_logging.h"
#include "utils.h"
#include "recipe_utils.hpp"
#include "runtime/scal/common/recipe_static_info_scal.hpp"
#include "runtime/scal/common/device_scal.hpp"
#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"
#include "runtime/scal/gaudi2/entities/recipe_reader_helper.hpp"
#include "runtime/scal/gaudi3/entities/recipe_reader_helper.hpp"
#include <map>

// Key(offset / address) to Value (size)
typedef std::map<uint64_t, uint64_t>        programDataSectionDB;
typedef eHostAddrToVirtualAddrMappingStatus eMappingStatus;

bool RecipeVerification::verifyProgramCodeBlobs(const recipe_t* pRecipe)
{
    HB_ASSERT(pRecipe != nullptr, "Invalid recipe pointer");

    blob_t*  pCurrentBlob = pRecipe->blobs;
    uint64_t blobsAmount  = pRecipe->blobs_nr;

    // Ranges are: [Base, Last)
    uint64_t patchableBufferBaseAddr = (uint64_t)pRecipe->patching_blobs_buffer;
    uint64_t patchableBufferLastAddr = patchableBufferBaseAddr + pRecipe->patching_blobs_buffer_size;
    uint64_t executionBufferBaseAddr = (uint64_t)pRecipe->execution_blobs_buffer;
    uint64_t executionBufferLastAddr = executionBufferBaseAddr + pRecipe->execution_blobs_buffer_size;
    uint64_t dynamicBufferBaseAddr   = (uint64_t)pRecipe->dynamic_blobs_buffer;
    uint64_t dynamicBufferLastAddr   = dynamicBufferBaseAddr + pRecipe->dynamic_blobs_buffer_size;

    uint64_t blobsBufferBaseAddress = 0;
    uint64_t blobsBufferLastAddress = 0;

    bool status = true;
    for (unsigned i = 0; i < blobsAmount; i++, pCurrentBlob++)
    {
        // Range is: [Base, Last)
        uint64_t blobBaseAddr = (uint64_t)pCurrentBlob->data;
        uint64_t blobLastAddr = blobBaseAddr + pCurrentBlob->size;

        switch (pCurrentBlob->blob_type_all)
        {
            case blob_t::PATCHING:
                blobsBufferBaseAddress = patchableBufferBaseAddr;
                blobsBufferLastAddress = patchableBufferLastAddr;
                break;

            case blob_t::EXE:
                blobsBufferBaseAddress = executionBufferBaseAddr;
                blobsBufferLastAddress = executionBufferLastAddr;
                break;

            case blob_t::DYNAMIC:
                blobsBufferBaseAddress = dynamicBufferBaseAddr;
                blobsBufferLastAddress = dynamicBufferLastAddr;
                break;
        }

        if ((blobBaseAddr < blobsBufferBaseAddress) || (blobLastAddr > blobsBufferLastAddress))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: blob[index {} type {}] range [0x{:x}, 0x{:x}) is outside of buffer range [0x{:x}, 0x{:x})",
                    HLLOG_FUNC,
                    i,
                    pCurrentBlob->blob_type_all,
                    blobBaseAddr,
                    blobLastAddr,
                    blobsBufferBaseAddress,
                    blobsBufferLastAddress);

            status = false;
        }
    }

    return status;
}

bool RecipeVerification::verifyProgramDataBlobs(const recipe_t* pRecipe, bool testMode)
{
    HB_ASSERT(pRecipe != nullptr, "Invalid recipe pointer");
    uint64_t buffStart    = (uint64_t)(pRecipe->program_data_blobs_buffer);
    uint64_t buffSize     = pRecipe->program_data_blobs_size;
    uint64_t numBlobs     = pRecipe->program_data_blobs_nr;
    bool     kernelPrintf = pRecipe->debug_profiler_info.printf_addr_nr > 0;

    if (numBlobs == 0) return true;

    uint64_t blob0data = (uint64_t)pRecipe->program_data_blobs[0].data;
    if (blob0data < buffStart)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: program_data_blobs[0].data < buffStart (0x{:x} < 0x{:x})",
                HLLOG_FUNC,
                blob0data,
                buffStart);
        return false;
    }

    uint64_t prevBlobEnd = buffStart;
    for (int i = 0; i < numBlobs; i++)
    {
        program_data_blob_t& curr = pRecipe->program_data_blobs[i];
        if (curr.section_idx != MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)  // Verify section==1
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Program data section_idx for blob {} is not as expected {}",
                    HLLOG_FUNC,
                    i,
                    curr.section_idx);
            return false;
        }

        if (curr.size == 0)
        {
            LOG_ERR(SYN_RECIPE, "{}: blob size is 0 for blob {}", HLLOG_FUNC, i);
            return false;
        }

        uint64_t blobData = (uint64_t)curr.data;
        if (blobData < prevBlobEnd)  // Verify no overlap with previous
        {
            LOG_ERR(SYN_RECIPE, "{}: Overlap for blob {}", HLLOG_FUNC, i);
            if (!testMode)
            {
                HB_ASSERT(false, "blobs data should be consecutive");
            }
            else
            {
                return false;
            }
        }

        prevBlobEnd = blobData + curr.size;

        // Verify the offset
        if (!kernelPrintf)
        {
            if (blobData != buffStart + curr.offset_in_section)  // Verify offset match the address (data field)
            {
                LOG_ERR(
                    SYN_RECIPE,
                    "{}: Offset is not consistent with data for blob {}. Offset 0x{:x} data 0x{:x} buffStart 0x{:x}",
                    HLLOG_FUNC,
                    i,
                    curr.offset_in_section,
                    blobData,
                    buffStart);
                return false;
            }
        }
    }

    if (pRecipe->workspace_nr < 3)
    {
        LOG_ERR(SYN_RECIPE, "workspace_nr is {}, should be at least 3", pRecipe->workspace_nr);
        return false;
    }

    program_data_blob_t& blobLast     = pRecipe->program_data_blobs[numBlobs - 1];
    uint64_t             blobLastData = (uint64_t)blobLast.data;
    if ((blobLastData + blobLast.size) > (buffStart + buffSize))  // For last blob, verify it is still in buffer
    {
        LOG_ERR(SYN_RECIPE,
                "{}: Last blob is outside the buffer. Data 0x{:x} size 0x{:x} buffStart 0x{:x} buffSize 0x{:x}",
                HLLOG_FUNC,
                blobLastData,
                blobLast.size,
                buffStart,
                buffSize);
        return false;
    }

    if (buffSize != pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA])
    {
        LOG_ERR(SYN_RECIPE,
                "{}: buffSize != workspace_size (0x{:x} != 0x{:x})",
                HLLOG_FUNC,
                buffSize,
                pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA]);
        return false;
    }

    return true;
}  // verifyProgramDataBlobs

bool RecipeVerification::verifyTpc(const recipe_t* pRecipe)
{
    return RecipeUtils::isRecipeTpcValid(pRecipe);
}

bool RecipeVerification::verifyPatching(const recipe_t* pRecipe)
{
    auto deviceType = RecipeUtils::getConfVal(pRecipe, gc_conf_t::DEVICE_TYPE);

    if (deviceType == synDeviceGaudi)
    {
        if (pRecipe->activate_patch_points_nr > pRecipe->patch_points_nr)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: illegal PP num activate {} enqueue {}",
                    HLLOG_FUNC,
                    pRecipe->activate_patch_points_nr,
                    pRecipe->patch_points_nr);
            return false;
        }

        const uint32_t activate_patch_points_nr = pRecipe->activate_patch_points_nr;
        const uint32_t enqueue_patch_points_nr  = pRecipe->patch_points_nr - pRecipe->activate_patch_points_nr;

        if ((activate_patch_points_nr != 0) && (pRecipe->activate_jobs_nr == 0))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: illegal PP num activate {} activate_jobs_nr {}",
                    HLLOG_FUNC,
                    activate_patch_points_nr,
                    pRecipe->activate_jobs_nr);
            return false;
        }

        if ((enqueue_patch_points_nr != 0) && (pRecipe->execute_jobs_nr == 0))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: illegal PP num enqueue {} execute_jobs_nr {}",
                    HLLOG_FUNC,
                    enqueue_patch_points_nr,
                    pRecipe->execute_jobs_nr);
            return false;
        }
    }

    return true;
}

using tensorDbType = shape_plane_basic_node_t::EShapePlanceTensorDb;
bool RecipeVerification::verifyDynamicRecipe(const recipe_t* pRecipe, const shape_plane_graph_t* spg)
{
    if (!spg) return true;  // Not DSD

    uint64_t           recipeTensorNum      = pRecipe->persist_tensors_nr;
    uint64_t           recipeShapeTensorNum = spg->shape_tensors_list_nr;
    ShapeFuncRegistry& sfr                  = ShapeFuncRegistry::instance();
    uint64_t           totalPP              = 0;
    synDeviceType      deviceType = (synDeviceType)RecipeUtils::getConfVal(pRecipe, gc_conf_t::DEVICE_TYPE).value();

    // Verify sp_tensors
    for (int tensor = 0; tensor < spg->sp_tensors_nr; tensor++)
    {
        tensor_info_t& currTensor = spg->sp_tensors[tensor];
        auto           type       = currTensor.tensor_type;
        if (type == tensor_info_t::PERSISTENT_TENSOR)
        {
            if (currTensor.tensor_db_index >= recipeTensorNum)
            {
                LOG_DSD_ERR("For shape_plane_graph data-tensor 0x{:x}: tensor_db_index {} must be less than number of "
                            "persistent tensors in the recipe ({})",
                            tensor,
                            currTensor.tensor_db_index,
                            recipeTensorNum);
                return false;
            }
        }
        else if (type == tensor_info_t::SHAPE_TENSOR)
        {
            if ((currTensor.tensor_db_index >= recipeShapeTensorNum) &&
                (currTensor.tensor_db_index != INVALID_TENSOR_INDEX))
            {
                LOG_DSD_ERR("For shape_plane_graph shape-tensor 0x{:x}: tensor_db_index {} must be less than number of "
                            "shape tensors in the recipe ({})",
                            tensor,
                            currTensor.tensor_db_index,
                            recipeShapeTensorNum);
                return false;
            }
        }
        if (type > tensor_info_t::INTERNAL_TENSOR)
        {
            LOG_DSD_ERR("For shape_plane_graph shape-tensor 0x{:x}: bad tensor type {}", tensor, type);
            return false;
        }
    }

    for (int node = 0; node < spg->sp_node_nr; node++)
    {
        shape_plane_node_t& currNode = spg->sp_nodes[node];

        // check input and output tensor index
        for (int tensor = 0; tensor < currNode.input_tensors_nr; tensor++)
        {
            if (currNode.input_tensors[tensor] >= spg->sp_tensors_nr)
            {
                LOG_DSD_ERR("Node 0x{:x} {} input tensor index 0x{:x}: tensor id {} must be less than number of "
                            "tensors in the shape graph ({})",
                            node,
                            currNode.node_name,
                            tensor,
                            currNode.input_tensors[tensor],
                            spg->sp_tensors_nr);
                return false;
            }
        }
        for (int tensor = 0; tensor < currNode.output_tensors_nr; tensor++)
        {
            if (currNode.output_tensors[tensor] >= spg->sp_tensors_nr)
            {
                LOG_DSD_ERR("Node 0x{:x} {} output tensor index 0x{:x}: tensor id {} must be less than number of "
                            "tensors in the shape graph ({})",
                            node,
                            currNode.node_name,
                            tensor,
                            currNode.output_tensors[tensor],
                            spg->sp_tensors_nr);
                return false;
            }
        }

        // Verify at least one sub-node.
        if (currNode.basic_nodes_nr == 0)  // must be >0
        {
            LOG_DSD_ERR("Node 0x{:x} {}: number of subnodes is 0 (must have at least 1)", node, currNode.node_name);
        }

        // If only one sub-node, no currNode.node_db_tensors_nr is expected
        bool noFuser = (currNode.basic_nodes_nr == 1);
        if (noFuser && (currNode.node_db_tensors_nr != 0))
        {
            LOG_DSD_ERR("Node 0x{:x} {}: number of subnodes is 1, but node_db_tensors_nr is not 0",
                        node,
                        currNode.node_name);
            return false;
        }

        // Check sub nodes
        for (int subNode = 0; subNode < currNode.basic_nodes_nr; subNode++)
        {
            shape_plane_basic_node_t& currSubNode = currNode.basic_nodes[subNode];

            // verify valid sif id
            if (currSubNode.sif_id.sm_func_index != INVALID_SHAPE_FUNC_ID)
            {
                if (sfr.getSIF(currSubNode.sif_id) == nullptr)
                {
                    LOG_DSD_ERR("No SIF function defined Node 0x{:x} {}, SIF 0x{:x}",
                                node,
                                currNode.node_name,
                                currSubNode.sif_id.sm_func_index);
                    return false;
                }
            }
            // check input and output tensor index
            for (int tensor = 0; tensor < currSubNode.input_tensors_nr; tensor++)
            {
                tensorDbType type = currSubNode.input_tensors_db[tensor];
                if (type == tensorDbType::GRAPH_TENSOR_DB)
                {
                    if (currSubNode.input_tensors[tensor] >= spg->sp_tensors_nr)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} input tensor index 0x{:x}: "
                                    "input tensor id {} must be less than number of tensors in the shape graph ({})",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.input_tensors[tensor],
                                    spg->sp_tensors_nr);
                        return false;
                    }
                    if (noFuser && currNode.input_tensors[tensor] != currSubNode.input_tensors[tensor])
                    {
                        LOG_DSD_ERR(
                            "Node 0x{:x} {} subNode 0x{:x} {} input tensor index 0x{:x}: "
                            "only one sub-node, but input tensor with id {} is not the same in both node and sub-node",
                            node,
                            currNode.node_name,
                            subNode,
                            currSubNode.node_name,
                            tensor,
                            currSubNode.input_tensors[tensor]);
                        return false;
                    }
                }
                else
                {
                    if (currSubNode.input_tensors[tensor] >= currNode.node_db_tensors_nr)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} input tensor index 0x{:x}: "
                                    "input tensor id {} must be less than number of tensors in the node ({})",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.input_tensors[tensor],
                                    currNode.node_db_tensors_nr);
                        return false;
                    }
                    if (noFuser)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} input tensor index 0x{:x}: "
                                    "only one sub-node, but input tensor with id {} is of type NODE_TENSOR_DB",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.input_tensors[tensor]);
                        return false;
                    }
                }
            }
            for (int tensor = 0; tensor < currSubNode.output_tensors_nr; tensor++)
            {
                tensorDbType type = currSubNode.output_tensors_db[tensor];
                if (type == tensorDbType::GRAPH_TENSOR_DB)
                {
                    if (currSubNode.output_tensors[tensor] >= spg->sp_tensors_nr)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} output tensor index 0x{:x}: "
                                    "output tensor id {} must be less than number of tensors in the shape graph ({})",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.output_tensors[tensor],
                                    spg->sp_tensors_nr);
                        return false;
                    }
                    if (noFuser && currNode.output_tensors[tensor] != currSubNode.output_tensors[tensor])
                    {
                        LOG_DSD_ERR(
                            "Node 0x{:x} {} subNode 0x{:x} {} output tensor index 0x{:x}: "
                            "only one sub-node, but output tensor with id {} is not the same in both node and sub-node",
                            node,
                            currNode.node_name,
                            subNode,
                            currSubNode.node_name,
                            tensor,
                            currSubNode.output_tensors[tensor]);
                        return false;
                    }
                }
                else
                {
                    if (currSubNode.output_tensors[tensor] >= currNode.node_db_tensors_nr)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} output tensor index 0x{:x}: "
                                    "output tensor id {} must be less than number of tensors in the node ({})",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.output_tensors[tensor],
                                    currNode.node_db_tensors_nr);
                        return false;
                    }
                    if (noFuser)
                    {
                        LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} output tensor index 0x{:x}: "
                                    "only one sub-node, but output tensor with id {} is of type NODE_TENSOR_DB",
                                    node,
                                    currNode.node_name,
                                    subNode,
                                    currSubNode.node_name,
                                    tensor,
                                    currSubNode.output_tensors[tensor]);
                        return false;
                    }
                }
            }
        }

        for (int tensor = 0; tensor < currNode.node_match_output_tensors_nr; tensor++)
        {
            if ((currNode.output_src_tensors[tensor] >= spg->sp_tensors_nr) ||
                (currNode.output_dst_tensors[tensor] >= spg->sp_tensors_nr))
            {
                LOG_DSD_ERR(
                    "Node 0x{:x} {} output tensor index 0x{:x}: "
                    "source tensor id {} and dest tensor id {} must be less than number of tensors in shape graph ({})",
                    node,
                    currNode.node_name,
                    tensor,
                    currNode.output_src_tensors[tensor],
                    currNode.output_dst_tensors[tensor],
                    spg->sp_tensors_nr);
                return false;
            }
        }

        totalPP += currNode.node_patch_points_nr;
        for (int pp = 0; pp < currNode.node_patch_points_nr; pp++)
        {
            sm_patch_point_t& currPp = currNode.node_patch_points[pp];

            if (currPp.smf_id.sm_func_index == INVALID_SHAPE_FUNC_ID)
            {
                LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x}: smf_id is invalid", node, currNode.node_name, pp);
                return false;
            }
            if (sfr.getSMF(currPp.smf_id) == nullptr)
            {
                LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x}: no SMF with id 0x{:x}",
                            node,
                            currNode.node_name,
                            pp,
                            currPp.smf_id.sm_func_index);
                return false;
            }

            if (currPp.patch_point_type != FIELD_DYNAMIC_ADDRESS && currPp.patch_point_type != FIELD_DYNAMIC_TPC_SIZE)
            {
                uint64_t blob_idx = currPp.blob_idx;
                if (blob_idx >= pRecipe->blobs_nr)
                {
                    LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x}: "
                                "blob_idx 0x{:x} must be less than number of blobs in recipe (0x{:x})",
                                node,
                                currNode.node_name,
                                pp,
                                blob_idx,
                                pRecipe->blobs_nr);
                    return false;
                }
                if ((deviceType < synDeviceGaudi2 && !pRecipe->blobs[blob_idx].blob_type.requires_patching) ||
                    (deviceType >= synDeviceGaudi2 && pRecipe->blobs[blob_idx].blob_type.static_exe))
                {
                    LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x} blob_idx 0x{:x}: requires_patching is not set, or "
                                "is static blob if gaudi2",
                                node,
                                currNode.node_name,
                                pp,
                                blob_idx);
                    return false;
                }
                if (((currPp.dw_offset_in_blob + currPp.patch_size_dw) * sizeof(uint32_t)) >
                    pRecipe->blobs[currPp.blob_idx].size)
                {
                    LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x} blob_idx 0x{:x}: "
                                "blob size 0x{:x} is less than patch point offset + patch size (0x{:x} + 0x{:x})",
                                node,
                                currNode.node_name,
                                pp,
                                currPp.blob_idx,
                                pRecipe->blobs[currPp.blob_idx].size,
                                currPp.dw_offset_in_blob * 4,
                                currPp.patch_size_dw * 4);
                    return false;
                }
            }
            else if (currPp.patch_point_type == FIELD_DYNAMIC_ADDRESS)
            {
                if ((currPp.patch_point_idx_low == -1) || (currPp.patch_point_idx_low >= pRecipe->patch_points_nr))
                {
                    LOG_DSD_ERR(
                        "Node 0x{:x} {}, patch point 0x{:x}: "
                        "patch_point_idx_low 0x{:x} must be less than number of patch points in the recipe (0x{:x})",
                        node,
                        currNode.node_name,
                        pp,
                        currPp.patch_point_idx_low,
                        pRecipe->patch_points_nr);
                    return false;
                }
                if ((currPp.patch_point_idx_high != -1) && (currPp.patch_point_idx_high >= pRecipe->patch_points_nr))
                {
                    LOG_DSD_ERR(
                        "Node 0x{:x} {}, patch point 0x{:x}: "
                        "patch_point_idx_high 0x{:x} must be less than number of patch points in the recipe (0x{:x})",
                        node,
                        currNode.node_name,
                        pp,
                        currPp.patch_point_idx_high,
                        pRecipe->patch_points_nr);
                    return false;
                }
            }

            uint64_t roi_idx = currPp.roi_idx;
            if (roi_idx >= currNode.activation_rois_nr)
            {
                LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x}: "
                            "roi_idx 0x{:x} must be less than number of activation ROIs in the node (0x{:x})",
                            node,
                            currNode.node_name,
                            pp,
                            roi_idx,
                            currNode.activation_rois_nr);
                return false;
            }
            roi_info_t& currRoi = currNode.activation_rois[roi_idx];
            if ((currRoi.roi_in_tensor_nr == 0) && (currRoi.roi_out_tensor_nr == 0))
            {
                LOG_DSD_ERR("Node 0x{:x} {}, patch point 0x{:x} roi_idx 0x{:x}: "
                            "both input and output tensor numbers are 0",
                            node,
                            currNode.node_name,
                            pp,
                            roi_idx);
                return false;
            }
        }
    }

    STAT_GLBL_COLLECT(totalPP, DsdPP);

    return true;
}

bool RecipeVerification::verifyStagedInfo(const recipe_t* pRecipe, const shape_plane_graph_t* spg)
{
    if (GCFG_ENABLE_STAGED_SUBMISSION.value() == false)
    {
        return true;
    }

    if (pRecipe->node_nr > 0)
    {
        for (uint64_t program = 0; program < pRecipe->programs_nr; program++)
        {
            if (pRecipe->programs[program].program_length !=
                pRecipe->node_exe_list[pRecipe->node_nr - 1].program_blobs_nr[program])
            {
                LOG_ERR(SYN_RECIPE,
                        "mismatch between number of blobs for program and node info program 0x{:x} 0x{:x} != 0x{:x}",
                        program,
                        pRecipe->programs[program].program_length,
                        pRecipe->node_exe_list[pRecipe->node_nr - 1].program_blobs_nr[program]);
                return false;
            }
        }
    }

    if (GCFG_STAGED_SUBMISSION_NODE_EXE_VALIDATION.value())
    {
        verifyStagesSubmissionNodeExe(pRecipe, spg);
    }

    LOG_DEBUG(SYN_RECIPE, "Verify Staged Info succeeded");
    return true;
}

bool RecipeVerification::verifyRecipeCacheSize(const recipe_t* pRecipe)
{
    auto devType = RecipeUtils::getConfVal(pRecipe, gc_conf_t::DEVICE_TYPE);

    if (devType == synDeviceGaudi)
    {
        const uint64_t blockSize            = GCFG_RECIPE_CACHE_BLOCK_SIZE.value() * 1024;
        uint64_t       numberOfBlocksNeeded = (pRecipe->execution_blobs_buffer_size + blockSize - 1) / blockSize;
        numberOfBlocksNeeded +=
            (pRecipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] + blockSize - 1) / blockSize;

        const uint64_t recipeCacheSize = GCFG_RECIPE_CACHE_SIZE.value() * 1024;
        const uint64_t requiredSize    = numberOfBlocksNeeded * blockSize;
        if (requiredSize > recipeCacheSize)
        {
            LOG_ERR(SYN_RECIPE,
                    "recipe cache size {} is not enough. recipe requires {}. pls change config '{}'",
                    recipeCacheSize,
                    requiredSize,
                    GCFG_RECIPE_CACHE_SIZE.primaryName());
            return false;
        }
    }
    return true;
}

bool RecipeVerification::verifyScalMemorySizes(const RecipeStaticInfoScal& recipeStaticInfoScal)
{
    uint64_t maxHbmGlobalMemorySizeForRecipe = common::DeviceScal::getHbmGlblMaxRecipeSize();
    if (recipeStaticInfoScal.m_glbHbmSizeTotal > maxHbmGlobalMemorySizeForRecipe)
    {
        LOG_ERR(SYN_RECIPE,
                "recipe global hbm memory size {} exceeds QueueComputeScal hbm global memory size for recipe {}",
                recipeStaticInfoScal.m_glbHbmSizeTotal,
                maxHbmGlobalMemorySizeForRecipe);
        return false;
    }

    uint64_t maxHbmSharedMemorySizeForRecipe = common::DeviceScal::getHbmSharedMaxRecipeSize();
    if (recipeStaticInfoScal.m_arcHbmSize > maxHbmSharedMemorySizeForRecipe)
    {
        LOG_ERR(SYN_RECIPE,
                "recipe arc hbm memory size {} exceeds QueueComputeScal arc hbm memory size for recipe {}",
                recipeStaticInfoScal.m_arcHbmSize,
                maxHbmSharedMemorySizeForRecipe);
        return false;
    }

    uint64_t dcSize = MappedMemMgr::getDcSize();
    uint64_t numDc  = MappedMemMgr::getNumDc();

    if ((div_round_up(recipeStaticInfoScal.m_mappedSizeNoPatch, dcSize) +
         div_round_up(recipeStaticInfoScal.m_mappedSizePatch, dcSize)) > numDc)
    {
        return false;
    }

    return true;
}

bool RecipeVerification::verifyStagesSubmissionNodeExe(const recipe_t* pRecipe, const shape_plane_graph_t* spg)
{
    // validate that every patchable blob required for every node is provided by a patch point with node_exe_index <=
    // node
    std::unordered_set<uint64_t> patchPointsPatchableBlobs;
    uint64_t                     currentPatchPointIndex = 0;
    auto                         numNodes               = pRecipe->node_nr;
    bool                         ret                    = true;
    for (uint64_t nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        // collect required blobs for current tested node
        std::vector<std::vector<uint64_t>> patchableBlobsPerProgram;
        patchableBlobsPerProgram.resize(pRecipe->programs_nr);
        node_program_t* nodePrograms = pRecipe->node_exe_list + nodeIndex;
        for (uint64_t program = 0; program < pRecipe->programs_nr; program++)
        {
            uint64_t* blobIndices       = pRecipe->programs[program].blob_indices;
            auto      numOfProgramBlobs = nodePrograms->program_blobs_nr[program];
            for (uint64_t singleProgramBlob = 0; singleProgramBlob < numOfProgramBlobs; singleProgramBlob++)
            {
                // pRecipe->programs[program].program_length already validated above
                blob_t* blob = pRecipe->blobs + (blobIndices[singleProgramBlob]);
                if (blob->blob_type.requires_patching)
                {
                    auto blobIndex = blobIndices[singleProgramBlob];
                    patchableBlobsPerProgram[program].push_back(blobIndex);
                }
            }
        }

        // collect provided blobs for current tested node
        auto patchPointsCount = pRecipe->patch_points_nr;
        for (; currentPatchPointIndex < patchPointsCount; currentPatchPointIndex++)
        {
            patch_point_t* patchPoint = pRecipe->patch_points + currentPatchPointIndex;
            if (patchPoint->node_exe_index > nodeIndex + 1)
            {
                // exit loop when we get to next node
                break;
            }
            patchPointsPatchableBlobs.insert(patchPoint->blob_idx);
        }

        for (uint64_t program = 0; program < pRecipe->programs_nr; program++)
        {
            std::set<uint64_t> missingProgramBlobs;
            for (auto patchableBlobNodeIndex : patchableBlobsPerProgram[program])
            {
                if (patchPointsPatchableBlobs.find(patchableBlobNodeIndex) == patchPointsPatchableBlobs.end())
                {
                    missingProgramBlobs.insert(patchableBlobNodeIndex);
                }
            }
            if (!missingProgramBlobs.empty())
            {
                std::stringstream ssMissingPb;
                ssMissingPb << "Missing " << missingProgramBlobs.size() << " blobs out of "
                            << nodePrograms->program_blobs_nr[program] << " at program " << program << " Node "
                            << nodeIndex;
                for (auto missingProgramBlob : missingProgramBlobs)
                {
                    ssMissingPb << " " << missingProgramBlob;
                }
                LOG_ERR(SYN_RECIPE, "{}", ssMissingPb.str());
                LOG_ERR(SYN_RECIPE, " Patch points number Node {}: {}", nodeIndex, nodePrograms->patch_points_nr);
                {
                    std::stringstream ssPbRequired;
                    ssPbRequired << " Patchable blobs required Node " << nodeIndex << " program " << program << " :";
                    for (auto nodesBlob : patchableBlobsPerProgram[program])
                    {
                        ssPbRequired << " " << nodesBlob;
                    }
                    LOG_ERR(SYN_RECIPE, "{}", ssPbRequired.str());
                }
                {
                    std::stringstream ssPbCovered;
                    ssPbCovered << " Patchable blobs covered by patch points up to node " << nodeIndex << " :";
                    for (auto ppsNodesBlob : patchPointsPatchableBlobs)
                    {
                        ssPbCovered << " " << ppsNodesBlob;
                    }
                    LOG_ERR(SYN_RECIPE, "{}", ssPbCovered.str());
                }
                uint64_t       patchPointIndex = 0;
                patch_point_t* pp              = pRecipe->patch_points;
                for (; patchPointIndex < patchPointsCount; patchPointIndex++, pp++)
                {
                    const auto firstMissingProgramBlobs = *missingProgramBlobs.cbegin();
                    if (pp->blob_idx == firstMissingProgramBlobs)
                    {
                        LOG_ERR(SYN_RECIPE,
                                "The first missing blob {} was found in a patch point with node_exe_index: {}",
                                firstMissingProgramBlobs,
                                pp->node_exe_index);
                        break;
                    }
                }
                if (patchPointIndex == patchPointsCount)
                {
                    LOG_ERR(SYN_RECIPE, "The missing blob was not found in any patch point");
                }
                ret = false;
                break;  // remove break to validate all programs and not stop after first invalid
            }
            if (nodePrograms->patch_points_nr > currentPatchPointIndex)
            {
                LOG_ERR(SYN_RECIPE,
                        "Node {}, Program {}: Has missing patch point. required: {}, provided: {}.",
                        nodeIndex,
                        program,
                        nodePrograms->patch_points_nr,
                        currentPatchPointIndex);
                ret = false;
                break;  // remove break to validate all programs and not stop after first invalid
            }
        }
        if (!ret)
        {
            return ret;  // remove return to validate all nodes and not stop after first invalid
        }
    }
    return ret;
}

bool RecipeVerification::verifyRecipe(const recipe_t* pRecipe, const shape_plane_graph_t* spg)
{
    bool res = verifyProgramCodeBlobs(pRecipe);
    if (!res)
    {
        return res;
    }

    res = verifyProgramDataBlobs(pRecipe);
    if (!res)
    {
        return res;
    }

    res = verifyTpc(pRecipe);
    if (!res)
    {
        return res;
    }

    res = verifyPatching(pRecipe);
    if (!res)
    {
        return res;
    }

    res = verifyDynamicRecipe(pRecipe, spg);
    if (!res)
    {
        return res;
    }

    res = verifyStagedInfo(pRecipe, spg);
    if (!res)
    {
        return res;
    }

    res = verifyRecipeCacheSize(pRecipe);
    if (!res)
    {
        return res;
    }

    res = verifyScalRecipe(pRecipe);
    if (!res)
    {
        return res;
    }

    return true;
}

static inline bool isSizeAlign(uint64_t size, uint32_t sizeAlign, const char* buffer)
{
    if ((size % sizeAlign) != 0)
    {
        LOG_ERR_T(SYN_RECIPE, "Size not aligned for {}. size {:x} align {:x}", buffer, size, sizeAlign);
        return false;
    }
    return true;
}

static inline bool isSizeAlign(uint64_t size, uint32_t sizeAlign, const char* buffer, int index)
{
    if ((size % sizeAlign) != 0)
    {
        LOG_ERR_T(SYN_RECIPE, "Size not aligned for {} {}. size {:x} align {:x}", buffer, index, size, sizeAlign);
        return false;
    }
    return true;
}

bool RecipeVerification::verifyScalRecipe(const recipe_t* recipe)
{
#if 0
    if (!isSizeAlign(recipe->patching_blobs_buffer_size, 128, "patch_buff"))
    {
        return false;
    }

    if (!isSizeAlign(recipe->execution_blobs_buffer_size, 128, "exec_buff"))
    {
        return false;
    }

    if (!isSizeAlign(recipe->dynamic_blobs_buffer_size, 128, "dynamic_buff"))
    {
        return false;
    }
    if (!isSizeAlign(recipe->program_data_blobs_size, 128, "prgData_buff"))
    {
        return false;
    }
#endif

    auto deviceType = RecipeUtils::getConfVal(recipe, gc_conf_t::DEVICE_TYPE);

    const common::RecipeReaderHelper* pRecipeReaderHelper = nullptr;
    if (deviceType == synDeviceGaudi2)
    {
        pRecipeReaderHelper = gaudi2::RecipeReaderHelper::getInstance();
    }
    else if (deviceType == synDeviceGaudi3)
    {
        pRecipeReaderHelper = gaudi3::RecipeReaderHelper::getInstance();
    }
    else
    {
        // Not a scal recipe
        return true;
    }
    uint32_t dynamicEcbListBufferSize = pRecipeReaderHelper->getDynamicEcbListBufferSize();
    uint32_t staticEcbListBufferSize  = pRecipeReaderHelper->getStaticEcbListBufferSize();

    if (recipe->valid_nop_kernel)
    {
        if (recipe->nop_kernel_section != MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            LOG_ERR_T(SYN_RECIPE,
                      "NOP-Kernel section-index {} does not use the PRG-Data default section-index",
                      recipe->nop_kernel_section);

            return false;
        }
    }
    bool isDeviceLessThanGaudi3 = deviceType.is_set() && deviceType.value() < synDeviceGaudi3;
    if (isDeviceLessThanGaudi3)
    {
        if (recipe->max_used_mcid_discard != 0 || recipe->max_used_mcid_degrade != 0)
        {
            LOG_ERR_T(SYN_RECIPE,
                      "mcid is set for device type {}. but it's supported from gaudi3 only."
                      " max_used_mcid_discard: {} max_used_mcid_degrade: {}",
                      deviceType.value(),
                      recipe->max_used_mcid_discard,
                      recipe->max_used_mcid_degrade);
            return false;
        }
    }
    if (recipe->max_used_mcid_discard > SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3 || recipe->max_used_mcid_degrade > SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3)
    {
        LOG_ERR_T(SYN_RECIPE,
                    "max used mcid's above limitation: max_used_mcid_discard: {} MAX is {}. max_used_mcid_degrade: {} MAX is {}",
                    recipe->max_used_mcid_discard,
                    SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3,
                    recipe->max_used_mcid_degrade,
                    SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3);
        return false;
    }

    for (int i = 0; i < recipe->arc_jobs_nr; i++)
    {
        if (!isSizeAlign(recipe->arc_jobs[i].dynamic_ecb.cmds_size, dynamicEcbListBufferSize, "arc_dynamic", i))
        {
            return false;
        }

        if (!isSizeAlign(recipe->arc_jobs[i].static_ecb.cmds_size, staticEcbListBufferSize, "arc_static", i))
        {
            return false;
        }

        if (recipe->arc_jobs[i].dynamic_ecb.cmds_size >= ((1 << 16) * dynamicEcbListBufferSize))
        {
            LOG_ERR_T(SYN_RECIPE,
                      "(offset dynamic to static) / DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE should be < 16 bits -> dynamic "
                      "size must be < 16 bits."
                      "Size for arc_job_dynamic {} is {:x}, DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE {:x}",
                      i,
                      recipe->arc_jobs[i].dynamic_ecb.cmds_size,
                      dynamicEcbListBufferSize);
            return false;
        }
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::CME && isDeviceLessThanGaudi3 == true)
        {
            LOG_ERR_T(SYN_RECIPE,
                      "CME engine is used for device type {}. it can be used starting Gaudi3 only. arc_job_dynamic {}",
                      deviceType.value(),
                      i);
            return false;
        }
        if (recipe->arc_jobs[i].logical_engine_id == Recipe::EngineType::DMA && isDeviceLessThanGaudi3 == false)
        {
            LOG_ERR_T(SYN_RECIPE,
                      "EDMA engine is used for device type {}. it can be used before Gaudi3 only. arc_job_dynamic {}",
                      deviceType.value(),
                      i);
            return false;
        }
    }

    for (uint32_t i = 0; i < recipe->patch_points_nr; i++)
    {
        if (recipe->patch_points[i].memory_patch_point.section_idx == MEMORY_ID_RESERVED_FOR_PROGRAM &&
            recipe->patch_points[i].type != patch_point_t::SOB_PATCH_POINT)
        {
            LOG_ERR_T(SYN_RECIPE, "patch point for section 2 not allowed {:x}", i);
            return false;
        }
    }

    return true;
}
