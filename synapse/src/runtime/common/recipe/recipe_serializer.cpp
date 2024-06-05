#include "recipe_serializer.hpp"
#include "utils.h"
#include "synapse_common_types.h"
#include "recipe.h"
#include "recipe_allocator.h"

#include "graph_compiler/smf/shape_func_registry.h"

#define VERIFY_IS_NULL_POINTER(pointer, name)                                                                          \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(SYN_RECIPE, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                        \
        return synFail;                                                                                                \
    }

static const uint64_t PROGRAM_BLOBS_ALIGNMENT_IN_BYTES = 128;

void serializePersistentTensor(ParamsManagerBase* pParams, const persist_tensor_info_t* pTensor, bool isPermuteTensor)
{
    HB_ASSERT_PTR(pTensor);

    uint32_t tensorNameLength = strlen(pTensor->name) + 1;

    pParams->append((char*)&tensorNameLength, sizeof(uint32_t));
    pParams->append((char*)pTensor->name, tensorNameLength);
    pParams->append((char*)&pTensor->section_idx, sizeof(uint16_t));
    pParams->append((char*)&pTensor->offset_in_section, sizeof(uint64_t));
    pParams->append((char*)&pTensor->size, sizeof(uint64_t));

    pParams->append((char*)&pTensor->elementType, sizeof(uint32_t));
    pParams->append((char*)&pTensor->zp, sizeof(double));
    pParams->append((char*)&pTensor->scale, sizeof(double));
    pParams->append((char*)&pTensor->dimensions, sizeof(uint32_t));
    pParams->append((char*)&pTensor->dimensionsSize[0], sizeof(uint32_t) * HABANA_DIM_MAX);
    pParams->append((char*)&pTensor->permutation[0], sizeof(uint8_t) * HABANA_DIM_MAX);
    pParams->append((char*)&pTensor->batchSize, sizeof(uint32_t));
    pParams->append((char*)&pTensor->isInput, sizeof(bool));
    pParams->append((char*)&pTensor->isExternal, sizeof(bool));
    pParams->append((char*)&pTensor->extTensorExeOrder, sizeof(uint32_t));
    pParams->append((char*)&pTensor->section_type, sizeof(uint8_t));
    pParams->append((char*)&pTensor->tensorType, sizeof(uint8_t));

    if (pTensor->layout != nullptr)
    {
        uint32_t tensorLayoutLength = strlen(pTensor->layout) + 1;
        pParams->append((char*)&tensorLayoutLength, sizeof(uint32_t));
        pParams->append((char*)pTensor->layout, tensorLayoutLength);
    }
    else
    {
        uint32_t tensorLayoutLength = 0;
        pParams->append((char*)&tensorLayoutLength, sizeof(uint32_t));
    }

    if ((isPermuteTensor) && ((pTensor->multi_views_indices_nr != 0) || (pTensor->multi_views_indices != nullptr)))
    {
        LOG_WARN(SYN_RECIPE,
                 "Unexpected (serialize) multi-view field is set for a permute-tensor (amount {})",
                 pTensor->multi_views_indices_nr);
    }
    pParams->append((char*)&pTensor->multi_views_indices_nr, sizeof(uint64_t));
    for (uint64_t multiViewIndex = 0; multiViewIndex < pTensor->multi_views_indices_nr; multiViewIndex++)
    {
        pParams->append((char*)&pTensor->multi_views_indices[multiViewIndex], sizeof(uint32_t));
    }
}

void deserializePersistentTensor(persist_tensor_info_t* pTensor,
                                 ParamsManagerBase*     pParams,
                                 RecipeAllocator*       pRecipeAlloc,
                                 bool                   isPermuteTensor)
{
    uint32_t tensorNameLength   = 0;
    uint32_t tensorLayoutLength = 0;

    pParams->getCurrentData(sizeof(uint32_t), (char*)&tensorNameLength);

    pTensor->name = pRecipeAlloc->allocate(tensorNameLength);
    pParams->getCurrentData(sizeof(char) * tensorNameLength, (char*)pTensor->name);
    pParams->getCurrentData(sizeof(uint16_t), (char*)&pTensor->section_idx);
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pTensor->offset_in_section);
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pTensor->size);

    pParams->getCurrentData(sizeof(uint32_t), (char*)&pTensor->elementType);
    pParams->getCurrentData(sizeof(double), (char*)&pTensor->zp);
    pParams->getCurrentData(sizeof(double), (char*)&pTensor->scale);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pTensor->dimensions);
    pParams->getCurrentData(sizeof(uint32_t) * HABANA_DIM_MAX, (char*)&pTensor->dimensionsSize[0]);
    pParams->getCurrentData(sizeof(uint8_t) * HABANA_DIM_MAX, (char*)&pTensor->permutation[0]);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pTensor->batchSize);
    pParams->getCurrentData(sizeof(bool), (char*)&pTensor->isInput);
    pParams->getCurrentData(sizeof(bool), (char*)&pTensor->isExternal);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pTensor->extTensorExeOrder);
    pParams->getCurrentData(sizeof(uint8_t), (char*)&pTensor->section_type);
    pParams->getCurrentData(sizeof(uint8_t), (char*)&pTensor->tensorType);

    pParams->getCurrentData(sizeof(uint32_t), (char*)&tensorLayoutLength);
    pTensor->layout = nullptr;
    if (tensorLayoutLength > 0)
    {
        pTensor->layout = pRecipeAlloc->allocate(tensorLayoutLength);
        pParams->getCurrentData(sizeof(char) * tensorLayoutLength, (char*)pTensor->layout);
    }

    pTensor->multi_views_indices = nullptr;
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pTensor->multi_views_indices_nr);
    if (pTensor->multi_views_indices_nr > 0)
    {
        if (isPermuteTensor)
        {
            LOG_WARN(SYN_RECIPE,
                     "Unexpected (deserialize) multi-view field is set for a permute-tensor (amount {})",
                     pTensor->multi_views_indices_nr);
        }

        uint64_t multiViewsIndicesSize = sizeof(uint32_t) * pTensor->multi_views_indices_nr;
        pTensor->multi_views_indices   = (uint32_t*)pRecipeAlloc->allocate(multiViewsIndicesSize);
        pParams->getCurrentData(multiViewsIndicesSize, (char*)pTensor->multi_views_indices);
    }
}

synStatus
RecipeSerializer::serialize(const recipe_t* pRecipe, const shape_plane_graph_t* spr, ParamsManagerBase* pParams)
{
    VERIFY_IS_NULL_POINTER(pRecipe, "pRecipe");
    VERIFY_IS_NULL_POINTER(pParams, "pParams");

    // serialize version
    pParams->append((char*)&pRecipe->version_major, sizeof(uint32_t));
    pParams->append((char*)&pRecipe->version_minor, sizeof(uint32_t));

    // serialize execution/patching blobs buffer & size
    pParams->append((char*)&pRecipe->execution_blobs_buffer_size, sizeof(uint64_t));
    if (pRecipe->execution_blobs_buffer_size)
    {
        pParams->append((char*)pRecipe->execution_blobs_buffer, pRecipe->execution_blobs_buffer_size);
    }
    pParams->append((char*)&pRecipe->patching_blobs_buffer_size, sizeof(uint64_t));
    if (pRecipe->patching_blobs_buffer_size)
    {
        pParams->append((char*)pRecipe->patching_blobs_buffer, pRecipe->patching_blobs_buffer_size);
    }

    // serialize dynamic blobs buffer & size
    pParams->append((char*)&pRecipe->dynamic_blobs_buffer_size, sizeof(uint64_t));
    if (pRecipe->dynamic_blobs_buffer_size)
    {
        pParams->append((char*)pRecipe->dynamic_blobs_buffer, pRecipe->dynamic_blobs_buffer_size);
    }

    // serialize blobs
    pParams->append((char*)&pRecipe->blobs_nr, sizeof(uint64_t));
    for (uint64_t blob_index = 0; blob_index < pRecipe->blobs_nr; blob_index++)
    {
        blob_t* pBlob = &pRecipe->blobs[blob_index];
        pParams->append((char*)&pBlob->blob_type_all, sizeof(blob_t::EBlobType));
        pParams->append((char*)&pBlob->size, sizeof(uint32_t));
        // Storing the offset from the relevant blobs' buffer
        uint64_t bufferOffset = 0;
        if (pBlob->blob_type.static_exe)
        {
            bufferOffset = (uint64_t)pBlob->data - (uint64_t)pRecipe->execution_blobs_buffer;
        }
        else if (pBlob->blob_type.dynamic_exe)
        {
            bufferOffset = (uint64_t)pBlob->data - (uint64_t)pRecipe->dynamic_blobs_buffer;
        }
        else if (pBlob->blob_type.requires_patching)
        {
            bufferOffset = (uint64_t)pBlob->data - (uint64_t)pRecipe->patching_blobs_buffer;
        }
        else
        {
            HB_ASSERT(false, "blob_type undefined");
        }

        pParams->append((char*)&bufferOffset, sizeof(uint64_t));
    }

    // serialize programs
    pParams->append((char*)&pRecipe->programs_nr, sizeof(uint32_t));
    for (uint32_t program_index = 0; program_index < pRecipe->programs_nr; program_index++)
    {
        program_t* pProgram = &pRecipe->programs[program_index];
        pParams->append((char*)&pProgram->program_length, sizeof(uint32_t));
        pParams->append((char*)pProgram->blob_indices, sizeof(uint64_t) * pProgram->program_length);
    }

    // serialize activate jobs
    pParams->append((char*)&pRecipe->activate_jobs_nr, sizeof(uint32_t));
    for (uint32_t job_index = 0; job_index < pRecipe->activate_jobs_nr; job_index++)
    {
        job_t* pJob = &pRecipe->activate_jobs[job_index];
        pParams->append((char*)&pJob->engine_id, sizeof(uint32_t));
        pParams->append((char*)&pJob->program_idx, sizeof(uint32_t));
    }

    // serialize execute jobs
    pParams->append((char*)&pRecipe->execute_jobs_nr, sizeof(uint32_t));
    for (uint32_t job_index = 0; job_index < pRecipe->execute_jobs_nr; job_index++)
    {
        job_t* pJob = &pRecipe->execute_jobs[job_index];
        pParams->append((char*)&pJob->engine_id, sizeof(uint32_t));
        pParams->append((char*)&pJob->program_idx, sizeof(uint32_t));
    }

    // serialize arc jobs
    pParams->append((char*)&pRecipe->arc_jobs_nr, sizeof(uint32_t));
    for (uint32_t job_index = 0; job_index < pRecipe->arc_jobs_nr; job_index++)
    {
        arc_job_t* pJob = &pRecipe->arc_jobs[job_index];
        pParams->append((char*)&pJob->logical_engine_id, sizeof(Recipe::EngineType));
        pParams->append((char*)&pJob->engines_filter, sizeof(uint32_t));

        pParams->append((char*)&pJob->static_ecb.cmds_size, sizeof(uint32_t));
        pParams->append((char*)&pJob->static_ecb.cmds_eng_offset, sizeof(uint32_t));
        if (pJob->static_ecb.cmds_size)
        {
            pParams->append((char*)pJob->static_ecb.cmds, pJob->static_ecb.cmds_size);
        }

        pParams->append((char*)&pJob->dynamic_ecb.cmds_size, sizeof(uint32_t));
        pParams->append((char*)&pJob->dynamic_ecb.cmds_eng_offset, sizeof(uint32_t));
        if (pJob->dynamic_ecb.cmds_size)
        {
            pParams->append((char*)pJob->dynamic_ecb.cmds, pJob->dynamic_ecb.cmds_size);
        }
    }

    // serialize persist tensors
    pParams->append((char*)&pRecipe->persist_tensors_nr, sizeof(uint32_t));
    for (uint32_t tensor_index = 0; tensor_index < pRecipe->persist_tensors_nr; tensor_index++)
    {
        persist_tensor_info_t* pTensor = &pRecipe->tensors[tensor_index];
        serializePersistentTensor(pParams, pTensor, false);
    }

    // serialize h2di tensors amount
    pParams->append((char*)&pRecipe->h2di_tensors_nr, sizeof(uint32_t));

    // serialize permute-view tensors
    pParams->append((char*)&pRecipe->permute_tensors_views_nr, sizeof(uint64_t));
    for (uint64_t i = 0; i < pRecipe->permute_tensors_views_nr; i++)
    {
        persist_tensor_info_t* pPermuteViewTensor = &pRecipe->permute_tensors_views[i];
        serializePersistentTensor(pParams, pPermuteViewTensor, true);
    }

    // serialize program_data_blobs buffer & size
    pParams->append((char*)&pRecipe->program_data_blobs_size, sizeof(uint64_t));

    if (pRecipe->program_data_blobs_size)
    {
        pParams->append(pRecipe->program_data_blobs_buffer, pRecipe->program_data_blobs_size);
    }

    // serialize const sections
    pParams->append((char*)&pRecipe->const_sections_nr, sizeof(uint32_t));
    for (uint32_t const_section_idx = 0; const_section_idx < pRecipe->const_sections_nr; const_section_idx++)
    {
        const_section_t* pConstSection = &pRecipe->const_sections[const_section_idx];

        pParams->append((char*)&pConstSection->size, sizeof(uint64_t));
        pParams->append((char*)&pConstSection->section_idx, sizeof(uint16_t));

        bool isConstSectionDataSet = ((uint64_t)pConstSection->data != INVALID_CONST_SECTION_DATA);
        pParams->append((char*)&isConstSectionDataSet, sizeof(uint8_t));
        if (isConstSectionDataSet)
        {
            pParams->append((char*)pConstSection->data, sizeof(char) * pConstSection->size);
        }
    }

    // serialize program data blobs
    pParams->append((char*)&pRecipe->program_data_blobs_nr, sizeof(uint32_t));
    for (uint32_t blob_index = 0; blob_index < pRecipe->program_data_blobs_nr; blob_index++)
    {
        program_data_blob_t* pBlob        = &pRecipe->program_data_blobs[blob_index];
        uint64_t             bufferOffset = (uint64_t)pBlob->data - (uint64_t)pRecipe->program_data_blobs_buffer;

        HB_ASSERT(bufferOffset + pBlob->size <= pRecipe->program_data_blobs_size, "program_data_blobs_size overlow");
        pParams->append((char*)&pBlob->size, sizeof(uint64_t));
        pParams->append((char*)&bufferOffset, sizeof(uint64_t));
        pParams->append((char*)&pBlob->offset_in_section, sizeof(uint64_t));
        pParams->append((char*)&pBlob->section_idx, sizeof(uint16_t));
    }

    // serialize patch points
    pParams->append((char*)&pRecipe->patch_points_nr, sizeof(uint32_t));
    pParams->append((char*)&pRecipe->activate_patch_points_nr, sizeof(uint32_t));
    for (uint32_t patch_index = 0; patch_index < pRecipe->patch_points_nr; patch_index++)
    {
        patch_point_t* pPatch = &pRecipe->patch_points[patch_index];
        pParams->append((char*)&pPatch->type, sizeof(patch_point_t::EPatchPointType));
        pParams->append((char*)&pPatch->blob_idx, sizeof(uint32_t));
        pParams->append((char*)&pPatch->dw_offset_in_blob, sizeof(uint32_t));
        if (pPatch->type == patch_point_t::SOB_PATCH_POINT)
        {
            pParams->append((char*)&pPatch->sob_patch_point.tensor_db_index,
                            sizeof(decltype(pPatch->sob_patch_point.tensor_db_index)));
        }
        else
        {
            pParams->append((char*)&pPatch->memory_patch_point.effective_address, sizeof(uint64_t));
            pParams->append((char*)&pPatch->memory_patch_point.section_idx, sizeof(uint16_t));
        }
        pParams->append((char*)&pPatch->node_exe_index, sizeof(uint32_t));
    }

    // serialize section types patch points
    pParams->append((char*)&pRecipe->section_groups_nr, sizeof(uint32_t));
    for (uint32_t section_index = 0; section_index < pRecipe->section_groups_nr; section_index++)
    {
        section_group_t* pSectionTypePatchPoints = &pRecipe->section_groups_patch_points[section_index];
        pParams->append((char*)&pSectionTypePatchPoints->section_group, sizeof(uint8_t));
        pParams->append((char*)&pSectionTypePatchPoints->patch_points_nr, sizeof(uint32_t));
        pParams->append((char*)pSectionTypePatchPoints->patch_points_index_list,
                        sizeof(uint32_t) * pSectionTypePatchPoints->patch_points_nr);
    }

    const section_group_t* pSectionTypePatchPoints = &pRecipe->sobj_section_group_patch_points;
    if (pRecipe->sobj_section_group_patch_points.patch_points_nr == 0)
    {
        uint32_t dummy32 = 0;
        pParams->append((char*)&dummy32, sizeof(uint8_t));
        pParams->append((char*)&dummy32, sizeof(uint32_t));
    }
    else
    {
        pParams->append((char*)&pSectionTypePatchPoints->section_group, sizeof(uint8_t));
        pParams->append((char*)&pSectionTypePatchPoints->patch_points_nr, sizeof(uint32_t));
        pParams->append((char*)pSectionTypePatchPoints->patch_points_index_list,
                        sizeof(uint32_t) * pSectionTypePatchPoints->patch_points_nr);
    }
    // serialize section id to blob indices
    pParams->append((char*)&pRecipe->section_ids_nr, sizeof(uint32_t));
    for (uint32_t section_id = 0; section_id < pRecipe->section_ids_nr; section_id++)
    {
        section_blobs_t* pSectionIdBlobIndices = &pRecipe->section_blobs_indices[section_id];
        pParams->append((char*)&pSectionIdBlobIndices->section_idx, sizeof(uint16_t));
        pParams->append((char*)&pSectionIdBlobIndices->blobs_nr, sizeof(uint32_t));
        pParams->append((char*)pSectionIdBlobIndices->blob_indices, sizeof(uint32_t) * pSectionIdBlobIndices->blobs_nr);
    }

    // serialize node execution list
    pParams->append((char*)&pRecipe->node_nr, sizeof(uint32_t));
    for (uint64_t node_index = 0; node_index < pRecipe->node_nr; node_index++)
    {
        node_program_t* pNode = &pRecipe->node_exe_list[node_index];
        pParams->append((char*)&pNode->patch_points_nr, sizeof(uint32_t));
        pParams->append((char*)pNode->program_blobs_nr, sizeof(uint32_t) * pRecipe->programs_nr);
    }

    // serialize sections
    pParams->append((char*)&pRecipe->sections_nr, sizeof(uint32_t));

    // serialize workspaces
    pParams->append((char*)&pRecipe->workspace_nr, sizeof(uint32_t));
    pParams->append((char*)pRecipe->workspace_sizes, sizeof(uint64_t) * pRecipe->workspace_nr);

    // serialize profiler info
    pParams->append((char*)&pRecipe->debug_profiler_info.version_major, sizeof(uint32_t));
    pParams->append((char*)&pRecipe->debug_profiler_info.version_minor, sizeof(uint32_t));

    pParams->append((char*)&pRecipe->debug_profiler_info.recipe_id, sizeof(uint16_t));

    pParams->append((char*)&pRecipe->debug_profiler_info.num_nodes, sizeof(uint32_t));
    uint32_t stringsLength = 0;
    for (uint64_t index = 0; index < pRecipe->debug_profiler_info.num_nodes; index++)
    {
        node_symbol_info_t* pNode = &pRecipe->debug_profiler_info.nodes[index];

        pParams->append((char*)&pNode->device_type, sizeof(EDeviceType));
        pParams->append((char*)&pNode->context_id, sizeof(uint16_t));
        pParams->append((char*)&pNode->full_context_id, sizeof(uint32_t));
        pParams->append((char*)&pNode->num_descriptors, sizeof(uint32_t));
        pParams->append((char*)&pNode->kernel_blob_index, sizeof(uint32_t));

        stringsLength = strlen(pNode->node_name) + 1;
        pParams->append((char*)&stringsLength, sizeof(uint32_t));
        pParams->append((char*)pNode->node_name, stringsLength);

        stringsLength = strlen(pNode->operation) + 1;
        pParams->append((char*)&stringsLength, sizeof(uint32_t));
        pParams->append((char*)pNode->operation, stringsLength);

        stringsLength = strlen(pNode->data_type) + 1;
        pParams->append((char*)&stringsLength, sizeof(uint32_t));
        pParams->append((char*)pNode->data_type, stringsLength);

        pParams->append((char*)&pNode->num_rois, sizeof(uint16_t));
        pParams->append((char*)pNode->num_working_engines, sizeof(uint8_t) * pNode->num_rois);
    }

    pParams->append((char*)&pRecipe->debug_profiler_info.printf_addr_nr, sizeof(uint32_t));
    for (uint64_t index = 0; index < pRecipe->debug_profiler_info.printf_addr_nr; index++)
    {
        uint64_t* pPrintf = &pRecipe->debug_profiler_info.printf_addr[index];
        pParams->append((char*)&(*pPrintf), sizeof(uint64_t));
    }

    pParams->append((char*)&pRecipe->debug_profiler_info.printf_section_idx, sizeof(uint64_t));

    // serialize sync scheme debug info
    pParams->append((char*)&pRecipe->debug_sync_scheme_info.node_sync_info_nr, sizeof(uint64_t));
    pParams->append((char*)&pRecipe->debug_sync_scheme_info.sync_scheme_legacy_mode, sizeof(bool));

    if (pRecipe->debug_sync_scheme_info.node_sync_info_nr > 0)
    {
        for (uint32_t info_idx = 0; info_idx < pRecipe->debug_sync_scheme_info.node_sync_info_nr; info_idx++)
        {
            if (pRecipe->debug_sync_scheme_info.sync_scheme_legacy_mode)
            {
                node_sync_info_legacy_t* pNodeSyncInfo =
                    &pRecipe->debug_sync_scheme_info.node_sync_info_legacy[info_idx];

                pParams->append((char*)&pNodeSyncInfo->engine_type, sizeof(Recipe::EngineType));
                pParams->append((char*)&pNodeSyncInfo->node_exe_index, sizeof(uint32_t));
                pParams->append((char*)&pNodeSyncInfo->pipe_level, sizeof(uint16_t));
                pParams->append((char*)&pNodeSyncInfo->emitted_signal, sizeof(uint16_t));
                pParams->append((char*)&pNodeSyncInfo->sob_id, sizeof(uint16_t));
                pParams->append((char*)&pNodeSyncInfo->num_engines, sizeof(uint16_t));
            }
            else
            {
                node_sync_info_arc_t* pNodeSyncInfo = &pRecipe->debug_sync_scheme_info.node_sync_info_arc[info_idx];

                pParams->append((char*)&pNodeSyncInfo->engine_type, sizeof(Recipe::EngineType));
                pParams->append((char*)&pNodeSyncInfo->node_exe_index, sizeof(uint32_t));
                pParams->append((char*)&pNodeSyncInfo->pipe_level, sizeof(uint16_t));
                pParams->append((char*)&pNodeSyncInfo->emitted_signal, sizeof(uint16_t));
            }
        }
    }

    // serialize conf params
    pParams->append((char*)&pRecipe->recipe_conf_nr, sizeof(uint32_t));
    for (uint32_t index = 0; index < pRecipe->recipe_conf_nr; index++)
    {
        gc_conf_t* pConf = &pRecipe->recipe_conf_params[index];

        pParams->append((char*)&pConf->conf_id, sizeof(uint8_t));
        pParams->append((char*)&pConf->conf_value, sizeof(uint64_t));
    }

    // serialize nop kernel info
    pParams->append((char*)&pRecipe->nop_kernel_offset, sizeof(uint64_t));
    pParams->append((char*)&pRecipe->nop_kernel_section, sizeof(uint64_t));
    pParams->append((char*)&pRecipe->valid_nop_kernel, sizeof(bool));

    // serialize mcid range
    pParams->append((char*)&pRecipe->max_used_mcid_discard, sizeof(uint16_t));
    pParams->append((char*)&pRecipe->max_used_mcid_degrade, sizeof(uint16_t));

    pParams->append((char*)&pRecipe->nameSize, sizeof(uint32_t));
    if (pRecipe->nameSize > 0)
    {
        pParams->append(pRecipe->name, pRecipe->nameSize);
    }

    ScanRecipeWrite scanRecipeWrite(spr, pParams);
    scanRecipeWrite.scan();

    if (!pParams->finalize())
    {
        LOG_ERR(SYN_RECIPE, "Recipe file failed to save");
        return synWrongParamsFile;
    }

    return synSuccess;
}

synStatus RecipeSerializer::deserialize(recipe_t*             pRecipe,
                                        shape_plane_graph_t*& spr,
                                        ParamsManager*        pParams,
                                        RecipeAllocator*      pRecipeAlloc)
{
    VERIFY_IS_NULL_POINTER(pRecipe, "pRecipe");
    VERIFY_IS_NULL_POINTER(pParams, "pParams");

    if (!pParams->loadFromDisk())
    {
        LOG_ERR(SYN_RECIPE, "Recipe file failed to load");
        return synWrongParamsFile;
    }

    // deserialize version
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->version_major);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->version_minor);

    // deserialize execution/patching blobs buffer & size
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->execution_blobs_buffer_size);
    if (pRecipe->execution_blobs_buffer_size)
    {
        int nr                          = 1 + pRecipe->execution_blobs_buffer_size / sizeof(uint64_t);
        pRecipe->execution_blobs_buffer = (uint64_t*)pRecipeAlloc->allocate(nr * sizeof(uint64_t), true);
        pParams->getCurrentData(sizeof(uint8_t) * pRecipe->execution_blobs_buffer_size,
                                (char*)pRecipe->execution_blobs_buffer);
    }
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->patching_blobs_buffer_size);
    if (pRecipe->patching_blobs_buffer_size)
    {
        int nr                         = 1 + pRecipe->patching_blobs_buffer_size / sizeof(uint64_t);
        pRecipe->patching_blobs_buffer = (uint64_t*)pRecipeAlloc->allocate(nr * sizeof(uint64_t), true);
        pParams->getCurrentData(sizeof(uint8_t) * pRecipe->patching_blobs_buffer_size,
                                (char*)pRecipe->patching_blobs_buffer);
    }

    // deserialize dynamic blobs buffer & size
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->dynamic_blobs_buffer_size);
    if (pRecipe->dynamic_blobs_buffer_size)
    {
        int nr                        = 1 + pRecipe->dynamic_blobs_buffer_size / sizeof(uint32_t);
        pRecipe->dynamic_blobs_buffer = (uint32_t*)pRecipeAlloc->allocate(nr * sizeof(uint32_t), true);
        pParams->getCurrentData(sizeof(uint8_t) * pRecipe->dynamic_blobs_buffer_size,
                                (char*)pRecipe->dynamic_blobs_buffer);
    }

    // deserialize blobs
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->blobs_nr);

    pRecipe->blobs       = (blob_t*)pRecipeAlloc->allocate(pRecipe->blobs_nr * sizeof(blob_t));
    uint64_t blobsOffset = 0;
    for (uint64_t blob_index = 0; blob_index < pRecipe->blobs_nr; blob_index++)
    {
        blob_t* pBlob = &pRecipe->blobs[blob_index];
        pParams->getCurrentData(sizeof(blob_t::EBlobType), (char*)&pBlob->blob_type_all);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pBlob->size);

        // Retrieving the offset for defining the blob's data pointer
        pParams->getCurrentData(sizeof(uint64_t), (char*)&blobsOffset);
        if (pBlob->blob_type.static_exe)
        {
            pBlob->data = (char*)pRecipe->execution_blobs_buffer + blobsOffset;
        }
        else if (pBlob->blob_type.dynamic_exe)
        {
            pBlob->data = (char*)pRecipe->dynamic_blobs_buffer + blobsOffset;
        }
        else if (pBlob->blob_type.requires_patching)
        {
            pBlob->data = (char*)pRecipe->patching_blobs_buffer + blobsOffset;
        }
        else
        {
            HB_ASSERT(false, "blob_type undefined");
        }
    }

    // deserialize programs
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->programs_nr);

    pRecipe->programs = (program_t*)pRecipeAlloc->allocate(pRecipe->programs_nr * sizeof(program_t));
    for (uint32_t program_index = 0; program_index < pRecipe->programs_nr; program_index++)
    {
        program_t* pProgram = &pRecipe->programs[program_index];
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pProgram->program_length);

        pProgram->blob_indices = (uint64_t*)pRecipeAlloc->allocate(pProgram->program_length * sizeof(uint64_t));
        pParams->getCurrentData(sizeof(uint64_t) * pProgram->program_length, (char*)pProgram->blob_indices);
    }

    // deserialize activate jobs
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->activate_jobs_nr);

    pRecipe->activate_jobs = (job_t*)pRecipeAlloc->allocate(pRecipe->activate_jobs_nr * sizeof(job_t));
    for (uint32_t job_index = 0; job_index < pRecipe->activate_jobs_nr; job_index++)
    {
        job_t* pJob = &pRecipe->activate_jobs[job_index];
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->engine_id);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->program_idx);
    }

    // deserialize execute jobs
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->execute_jobs_nr);

    pRecipe->execute_jobs = (job_t*)pRecipeAlloc->allocate(pRecipe->execute_jobs_nr * sizeof(job_t));
    for (uint32_t job_index = 0; job_index < pRecipe->execute_jobs_nr; job_index++)
    {
        job_t* pJob = &pRecipe->execute_jobs[job_index];
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->engine_id);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->program_idx);
    }

    // deserialize arc jobs
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->arc_jobs_nr);

    pRecipe->arc_jobs = (arc_job_t*)pRecipeAlloc->allocate(pRecipe->arc_jobs_nr * sizeof(arc_job_t));
    for (uint32_t job_index = 0; job_index < pRecipe->arc_jobs_nr; job_index++)
    {
        arc_job_t* pJob = &pRecipe->arc_jobs[job_index];
        pParams->getCurrentData(sizeof(Recipe::EngineType), (char*)&pJob->logical_engine_id);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->engines_filter);

        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->static_ecb.cmds_size);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->static_ecb.cmds_eng_offset);
        if (pJob->static_ecb.cmds_size)
        {
            pJob->static_ecb.cmds = (uint8_t*)pRecipeAlloc->allocate(pJob->static_ecb.cmds_size, true);
            pParams->getCurrentData(sizeof(uint8_t) * pJob->static_ecb.cmds_size, (char*)pJob->static_ecb.cmds);
        }

        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->dynamic_ecb.cmds_size);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pJob->dynamic_ecb.cmds_eng_offset);
        if (pJob->dynamic_ecb.cmds_size)
        {
            pJob->dynamic_ecb.cmds = (uint8_t*)pRecipeAlloc->allocate(pJob->dynamic_ecb.cmds_size, true);
            pParams->getCurrentData(sizeof(uint8_t) * pJob->dynamic_ecb.cmds_size, (char*)pJob->dynamic_ecb.cmds);
        }
    }

    // deserialize persist tensors
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->persist_tensors_nr);
    //
    pRecipe->tensors =
        (persist_tensor_info_t*)pRecipeAlloc->allocate(pRecipe->persist_tensors_nr * sizeof(persist_tensor_info_t));
    for (uint32_t tensor_index = 0; tensor_index < pRecipe->persist_tensors_nr; tensor_index++)
    {
        persist_tensor_info_t* pTensor = &pRecipe->tensors[tensor_index];

        deserializePersistentTensor(pTensor, pParams, pRecipeAlloc, false);
    }

    // deserialize h2di tensors amount
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->h2di_tensors_nr);

    // deserialize permute-view tensors
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->permute_tensors_views_nr);
    //
    pRecipe->permute_tensors_views = (persist_tensor_info_t*)pRecipeAlloc->allocate(pRecipe->permute_tensors_views_nr *
                                                                                    sizeof(persist_tensor_info_t));
    persist_tensor_info_t* pCurrPermuteViewTensor = pRecipe->permute_tensors_views;
    for (uint32_t i = 0; i < pRecipe->permute_tensors_views_nr; i++, pCurrPermuteViewTensor++)
    {
        deserializePersistentTensor(pCurrPermuteViewTensor, pParams, pRecipeAlloc, true);
    }

    // deserialize program_data_blobs buffer & size
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->program_data_blobs_size);

    if (pRecipe->program_data_blobs_size)
    {
        pRecipe->program_data_blobs_buffer = pRecipeAlloc->allocate(pRecipe->program_data_blobs_size, true);
    }
    else
    {
        pRecipe->program_data_blobs_buffer = nullptr;
    }

    pParams->getCurrentData(sizeof(char) * pRecipe->program_data_blobs_size, pRecipe->program_data_blobs_buffer);

    // deserialize const sections
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->const_sections_nr);

    if (pRecipe->const_sections_nr > 0)
    {
        pRecipe->const_sections =
            (const_section_t*)pRecipeAlloc->allocate(pRecipe->const_sections_nr * sizeof(const_section_t));

        for (uint32_t const_section_idx = 0; const_section_idx < pRecipe->const_sections_nr; const_section_idx++)
        {
            const_section_t* pConstSection = &pRecipe->const_sections[const_section_idx];
            pParams->getCurrentData(sizeof(uint64_t), (char*)&pConstSection->size);
            pParams->getCurrentData(sizeof(uint16_t), (char*)&pConstSection->section_idx);

            uint8_t dataSetIndication = std::numeric_limits<uint8_t>::max();
            pParams->getCurrentData(sizeof(uint8_t), (char*)&dataSetIndication);

            pConstSection->data = nullptr;

            bool isConstSectionDataSet = (dataSetIndication == 1);
            if (!isConstSectionDataSet)
            {
                pConstSection->data = (char*)INVALID_CONST_SECTION_DATA;
            }
            else if (pConstSection->size)  // we may get zero size sections
            {
                pConstSection->data = pRecipeAlloc->allocate(pConstSection->size * sizeof(char));
                pParams->getCurrentData(sizeof(char) * pConstSection->size, pConstSection->data);
            }
        }
    }

    // deserialize program data blobs
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->program_data_blobs_nr);

    uint64_t dataOffset = 0;
    pRecipe->program_data_blobs =
        (program_data_blob_t*)pRecipeAlloc->allocate(pRecipe->program_data_blobs_nr * sizeof(program_data_blob_t));
    for (uint32_t blob_index = 0; blob_index < pRecipe->program_data_blobs_nr; blob_index++)
    {
        program_data_blob_t* pBlob = &pRecipe->program_data_blobs[blob_index];
        pParams->getCurrentData(sizeof(uint64_t), (char*)&pBlob->size);

        pParams->getCurrentData(sizeof(uint64_t), (char*)&dataOffset);
        pBlob->data = (char*)pRecipe->program_data_blobs_buffer + dataOffset;
        pParams->getCurrentData(sizeof(uint64_t), (char*)&pBlob->offset_in_section);
        pParams->getCurrentData(sizeof(uint16_t), (char*)&pBlob->section_idx);
        HB_ASSERT(dataOffset + pBlob->size <= pRecipe->program_data_blobs_size, "program_data_blobs_size overlow");
    }

    // deserialize patch points
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->patch_points_nr);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->activate_patch_points_nr);

    pRecipe->patch_points = (patch_point_t*)pRecipeAlloc->allocate(pRecipe->patch_points_nr * sizeof(patch_point_t));
    for (uint32_t patchPoint_index = 0; patchPoint_index < pRecipe->patch_points_nr; patchPoint_index++)
    {
        patch_point_t* pPatchPoint = &pRecipe->patch_points[patchPoint_index];
        pParams->getCurrentData(sizeof(patch_point_t::EPatchPointType), (char*)&pPatchPoint->type);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pPatchPoint->blob_idx);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pPatchPoint->dw_offset_in_blob);
        if (pPatchPoint->type == patch_point_t::SOB_PATCH_POINT)
        {
            pParams->getCurrentData(sizeof(decltype(pPatchPoint->sob_patch_point.tensor_db_index)),
                                    (char*)&pPatchPoint->sob_patch_point.tensor_db_index);
        }
        else
        {
            pParams->getCurrentData(sizeof(uint64_t), (char*)&pPatchPoint->memory_patch_point.effective_address);
            pParams->getCurrentData(sizeof(uint16_t), (char*)&pPatchPoint->memory_patch_point.section_idx);
        }
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pPatchPoint->node_exe_index);
    }

    // deserialize section types patch points
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->section_groups_nr);

    pRecipe->section_groups_patch_points =
        (section_group_t*)pRecipeAlloc->allocate(pRecipe->section_groups_nr * sizeof(section_group_t));
    for (uint32_t section_index = 0; section_index < pRecipe->section_groups_nr; section_index++)
    {
        section_group_t* pSectionTypePatchPoints = &pRecipe->section_groups_patch_points[section_index];
        pParams->getCurrentData(sizeof(uint8_t), (char*)&pSectionTypePatchPoints->section_group);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pSectionTypePatchPoints->patch_points_nr);

        pSectionTypePatchPoints->patch_points_index_list =
            (uint32_t*)pRecipeAlloc->allocate(pSectionTypePatchPoints->patch_points_nr * sizeof(uint32_t));
        for (uint32_t pp_index = 0; pp_index < pSectionTypePatchPoints->patch_points_nr; pp_index++)
        {
            pParams->getCurrentData(sizeof(uint32_t),
                                    (char*)&pSectionTypePatchPoints->patch_points_index_list[pp_index]);
        }
    }

    section_group_t* pSectionTypePatchPoints = &pRecipe->sobj_section_group_patch_points;
    pParams->getCurrentData(sizeof(uint8_t), (char*)&pSectionTypePatchPoints->section_group);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pSectionTypePatchPoints->patch_points_nr);

    pSectionTypePatchPoints->patch_points_index_list =
        (uint32_t*)pRecipeAlloc->allocate(pSectionTypePatchPoints->patch_points_nr * sizeof(uint32_t));
    for (uint32_t pp_index = 0; pp_index < pSectionTypePatchPoints->patch_points_nr; pp_index++)
    {
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pSectionTypePatchPoints->patch_points_index_list[pp_index]);
    }

    // deserialize section id to blob indices
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->section_ids_nr);

    pRecipe->section_blobs_indices =
        (section_blobs_t*)pRecipeAlloc->allocate(pRecipe->section_ids_nr * sizeof(section_blobs_t));
    for (uint32_t section_id = 0; section_id < pRecipe->section_ids_nr; section_id++)
    {
        section_blobs_t* pSectionIdBlobIndices = &pRecipe->section_blobs_indices[section_id];
        pParams->getCurrentData(sizeof(uint16_t), (char*)&pSectionIdBlobIndices->section_idx);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pSectionIdBlobIndices->blobs_nr);

        pSectionIdBlobIndices->blob_indices =
            (uint32_t*)pRecipeAlloc->allocate(pSectionIdBlobIndices->blobs_nr * sizeof(uint32_t));
        for (uint32_t blob_index = 0; blob_index < pSectionIdBlobIndices->blobs_nr; blob_index++)
        {
            pParams->getCurrentData(sizeof(uint32_t), (char*)&pSectionIdBlobIndices->blob_indices[blob_index]);
        }
    }

    // deserialize node execution list
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->node_nr);
    if (pRecipe->node_nr > 0)
    {
        pRecipe->node_exe_list = (node_program_t*)pRecipeAlloc->allocate(pRecipe->node_nr * sizeof(node_program_t));

        for (uint64_t node_index = 0; node_index < pRecipe->node_nr; node_index++)
        {
            node_program_t* pNode = &pRecipe->node_exe_list[node_index];
            pParams->getCurrentData(sizeof(uint32_t), (char*)&pNode->patch_points_nr);

            pNode->program_blobs_nr = (uint32_t*)pRecipeAlloc->allocate(pRecipe->programs_nr * sizeof(uint32_t));
            for (uint32_t program_index = 0; program_index < pRecipe->programs_nr; program_index++)
            {
                pParams->getCurrentData(sizeof(uint32_t), (char*)&pNode->program_blobs_nr[program_index]);
            }
        }
    }

    // deserialize sections
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->sections_nr);

    // deserialize workspaces
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->workspace_nr);

    pRecipe->workspace_sizes = (uint64_t*)pRecipeAlloc->allocate(pRecipe->workspace_nr * sizeof(uint64_t));
    pParams->getCurrentData(sizeof(uint64_t) * pRecipe->workspace_nr, (char*)pRecipe->workspace_sizes);

    // deserialize profiler info
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->debug_profiler_info.version_major);
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->debug_profiler_info.version_minor);
    pParams->getCurrentData(sizeof(uint16_t), (char*)&pRecipe->debug_profiler_info.recipe_id);

    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->debug_profiler_info.num_nodes);

    pRecipe->debug_profiler_info.nodes = (node_symbol_info_t*)pRecipeAlloc->allocate(
        pRecipe->debug_profiler_info.num_nodes * sizeof(node_symbol_info_t));

    uint32_t stringsLegth = 0;

    for (uint64_t index = 0; index < pRecipe->debug_profiler_info.num_nodes; index++)
    {
        node_symbol_info_t* pNode = &pRecipe->debug_profiler_info.nodes[index];

        pParams->getCurrentData(sizeof(EDeviceType), (char*)&pNode->device_type);
        pParams->getCurrentData(sizeof(uint16_t), (char*)&pNode->context_id);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pNode->full_context_id);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pNode->num_descriptors);
        pParams->getCurrentData(sizeof(uint32_t), (char*)&pNode->kernel_blob_index);

        pParams->getCurrentData(sizeof(uint32_t), (char*)&stringsLegth);
        pNode->node_name = pRecipeAlloc->allocate(stringsLegth);
        pParams->getCurrentData(sizeof(char) * stringsLegth, (char*)pNode->node_name);

        pParams->getCurrentData(sizeof(uint32_t), (char*)&stringsLegth);
        pNode->operation = pRecipeAlloc->allocate(stringsLegth);
        pParams->getCurrentData(sizeof(char) * stringsLegth, (char*)pNode->operation);

        pParams->getCurrentData(sizeof(uint32_t), (char*)&stringsLegth);
        pNode->data_type = pRecipeAlloc->allocate(stringsLegth);
        pParams->getCurrentData(sizeof(char) * stringsLegth, (char*)pNode->data_type);

        pParams->getCurrentData(sizeof(uint16_t), (char*)&pNode->num_rois);
        pNode->num_working_engines = (uint8_t*)pRecipeAlloc->allocate(pNode->num_rois);
        pParams->getCurrentData(sizeof(uint8_t) * pNode->num_rois, (char*)pNode->num_working_engines);
    }

    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->debug_profiler_info.printf_addr_nr);

    pRecipe->debug_profiler_info.printf_addr =
        (uint64_t*)pRecipeAlloc->allocate(pRecipe->debug_profiler_info.printf_addr_nr * sizeof(uint64_t));
    for (uint64_t index = 0; index < pRecipe->debug_profiler_info.printf_addr_nr; index++)
    {
        uint64_t* pPrintf = &pRecipe->debug_profiler_info.printf_addr[index];

        pParams->getCurrentData(sizeof(uint64_t), (char*)&(*pPrintf));
    }

    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->debug_profiler_info.printf_section_idx);

    // deserialize sync scheme debug info
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->debug_sync_scheme_info.node_sync_info_nr);
    pParams->getCurrentData(sizeof(bool), (char*)&pRecipe->debug_sync_scheme_info.sync_scheme_legacy_mode);

    if (pRecipe->debug_sync_scheme_info.node_sync_info_nr > 0)
    {
        if (pRecipe->debug_sync_scheme_info.sync_scheme_legacy_mode)
        {
            pRecipe->debug_sync_scheme_info.node_sync_info_legacy = (node_sync_info_legacy_t*)pRecipeAlloc->allocate(
                pRecipe->debug_sync_scheme_info.node_sync_info_nr * sizeof(node_sync_info_legacy_t));

            for (uint32_t info_idx = 0; info_idx < pRecipe->debug_sync_scheme_info.node_sync_info_nr; info_idx++)
            {
                node_sync_info_legacy_t* pNodeSyncInfo =
                    &pRecipe->debug_sync_scheme_info.node_sync_info_legacy[info_idx];

                pParams->getCurrentData(sizeof(Recipe::EngineType), (char*)&pNodeSyncInfo->engine_type);
                pParams->getCurrentData(sizeof(uint32_t), (char*)&pNodeSyncInfo->node_exe_index);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->pipe_level);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->emitted_signal);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->sob_id);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->num_engines);
            }
        }
        else
        {
            pRecipe->debug_sync_scheme_info.node_sync_info_arc = (node_sync_info_arc_t*)pRecipeAlloc->allocate(
                pRecipe->debug_sync_scheme_info.node_sync_info_nr * sizeof(node_sync_info_arc_t));

            for (uint32_t info_idx = 0; info_idx < pRecipe->debug_sync_scheme_info.node_sync_info_nr; info_idx++)
            {
                node_sync_info_arc_t* pNodeSyncInfo = &pRecipe->debug_sync_scheme_info.node_sync_info_arc[info_idx];

                pParams->getCurrentData(sizeof(Recipe::EngineType), (char*)&pNodeSyncInfo->engine_type);
                pParams->getCurrentData(sizeof(uint32_t), (char*)&pNodeSyncInfo->node_exe_index);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->pipe_level);
                pParams->getCurrentData(sizeof(uint16_t), (char*)&pNodeSyncInfo->emitted_signal);
            }
        }
    }
    else
    {
        pRecipe->debug_sync_scheme_info.node_sync_info_legacy = nullptr;
    }

    // deserialize gc conf info
    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->recipe_conf_nr);
    pRecipe->recipe_conf_params = (gc_conf_t*)pRecipeAlloc->allocate(pRecipe->recipe_conf_nr * sizeof(gc_conf_t));

    for (uint32_t index = 0; index < pRecipe->recipe_conf_nr; index++)
    {
        gc_conf_t* pConf = &pRecipe->recipe_conf_params[index];

        pParams->getCurrentData(sizeof(uint8_t), (char*)&pConf->conf_id);
        pParams->getCurrentData(sizeof(uint64_t), (char*)&pConf->conf_value);
    }

    // deserialize nop kernel info
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->nop_kernel_offset);
    pParams->getCurrentData(sizeof(uint64_t), (char*)&pRecipe->nop_kernel_section);
    pParams->getCurrentData(sizeof(bool), (char*)&pRecipe->valid_nop_kernel);

    // deserialize mcid
    pParams->getCurrentData(sizeof(uint16_t), (char*)&pRecipe->max_used_mcid_discard);
    pParams->getCurrentData(sizeof(uint16_t), (char*)&pRecipe->max_used_mcid_degrade);

    pParams->getCurrentData(sizeof(uint32_t), (char*)&pRecipe->nameSize);
    if (pRecipe->nameSize > 0)
    {
        pRecipe->name = (char*)pRecipeAlloc->allocate(pRecipe->nameSize);
        pParams->getCurrentData(pRecipe->nameSize, pRecipe->name);
    }

    ScanRecipeRead scanRecipeRead(spr, pParams, pRecipeAlloc);
    scanRecipeRead.scan();
    bool verifyRes = verifyDsdVersion(spr);

    if (verifyRes == false)
    {
        return synUnsupported;
    }

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief This function checks the dynamic recipe
 *   The function is called after the recipe is loaded from disk to verifies that the sif version
 *   when the recipe was created is the same to the current version in synapse
 *
 *   @output true = OK
 *
 ***************************************************************************************************
 */
bool RecipeSerializer::verifyDsdVersion(shape_plane_graph_t* spr)
{
    if (!spr) return true;

    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();
    for (int node = 0; node < spr->sp_node_nr; node++)
    {
        for (int subnode = 0; subnode < spr->sp_nodes[node].basic_nodes_nr; ++subnode)
        {
            sm_function_id_t sifId = spr->sp_nodes[node].basic_nodes[subnode].sif_id;

            if (sifId.sm_func_index == INVALID_SHAPE_FUNC_ID) continue;

            uint64_t currVer   = sfr.getSifVersion(sifId);
            uint64_t loadedVer = spr->sp_nodes[node].basic_nodes[subnode].sif_version;

            if (currVer != loadedVer)
            {
                LOG_ERR_T(SYN_RECIPE,
                          "Loaded sif version different than current one. "
                          "Node 0x{:x} curr/loaded version 0x{:x}/0x{:x}, sif table id {}, sif func id {}",
                          node,
                          currVer,
                          loadedVer,
                          sifId.sm_tableid,
                          sifId.sm_funcid);

                return false;
            }
        }
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function goes over the recipe and calls different functions to save/load the recipe
 *
 *   @output None
 *
 ***************************************************************************************************
 */
template<class RecipeOp>
void ScanRecipe<RecipeOp>::scan()
{
    // Handle shape_plane_graph_t
    m_recipeOp.head(m_spr);  // write/read the pointer to DSD (for read, it returns the original pointer)

    if (!m_spr) return;

    m_recipeOp.single(m_spr);

    m_recipeOp.array(m_spr->sp_nodes, m_spr->sp_node_nr);  // shape_plane_node_t
    // Handle shape_plane_node_t pointers
    for (uint64_t node = 0; node < m_spr->sp_node_nr; node++)
    {
        shape_plane_node_t& currNode = m_spr->sp_nodes[node];

        m_recipeOp.array(currNode.input_tensors,         currNode.input_tensors_nr);
        m_recipeOp.array(currNode.output_tensors,        currNode.output_tensors_nr);
        m_recipeOp.array(currNode.output_src_tensors, currNode.node_match_output_tensors_nr);
        m_recipeOp.array(currNode.output_dst_tensors,   currNode.node_match_output_tensors_nr);

        m_recipeOp.array(currNode.activation_rois, currNode.activation_rois_nr);
        // Handle roi_info_t
        for (uint64_t actRoi = 0; actRoi < currNode.activation_rois_nr; actRoi++)
        {
            roi_info_t& currRoi = currNode.activation_rois[actRoi];

            m_recipeOp.array(currRoi.roi_in_tensors, currRoi.roi_in_tensor_nr);
            m_recipeOp.array(currRoi.roi_out_tensors, currRoi.roi_out_tensor_nr);
        }

        m_recipeOp.array(currNode.node_patch_points, currNode.node_patch_points_nr);
        for (uint64_t actPp = 0; actPp < currNode.node_patch_points_nr; actPp++)
        {
            m_recipeOp.array(currNode.node_patch_points[actPp].metadata,
                             currNode.node_patch_points[actPp].metadata_size);
        }
        m_recipeOp.array(currNode.node_db_tensors, currNode.node_db_tensors_nr);

        m_recipeOp.array(currNode.basic_nodes, currNode.basic_nodes_nr);

        for (uint64_t subnode = 0; subnode < m_spr->sp_nodes[node].basic_nodes_nr; subnode++)
        {
            shape_plane_basic_node_t& currSubnode = currNode.basic_nodes[subnode];
            m_recipeOp.array(currSubnode.input_tensors, currSubnode.input_tensors_nr);
            m_recipeOp.array(currSubnode.output_tensors, currSubnode.output_tensors_nr);
            m_recipeOp.array(currSubnode.input_tensors_db, currSubnode.input_tensors_nr);
            m_recipeOp.array(currSubnode.output_tensors_db, currSubnode.output_tensors_nr);
            m_recipeOp.array(currSubnode.sif_params, currSubnode.sif_params_nr);
            m_recipeOp.array(currSubnode.input_permutations, currSubnode.input_permutations_nr);
        }
    }

    m_recipeOp.array(m_spr->sp_tensors, m_spr->sp_tensors_nr);  // tensor_info_t
    for (uint64_t tensorT = 0; tensorT < m_spr->sp_tensors_nr; tensorT++)
    {
        tensor_info_t& currTensorT = m_spr->sp_tensors[tensorT];
        if (currTensorT.tensor_info_name != nullptr)
        {
            m_recipeOp.charArray(currTensorT.tensor_info_name);
        }
    }

    m_recipeOp.array(m_spr->shape_tensors, m_spr->shape_tensors_list_nr);  // shape_tensor_info_t
    for (uint64_t shapeT = 0; shapeT < m_spr->shape_tensors_list_nr; shapeT++)
    {
        shape_tensor_info_t& currShapeT = m_spr->shape_tensors[shapeT];
        m_recipeOp.charArray(currShapeT.name);
    }
}

/*************************************************************************/
/*                          class WriteToDisk                            */
/*************************************************************************/
// This function is used to write a pointer to the disk
template<typename T>
void WriteToDisk::head(T* element)
{
    m_pParams->append((char*)(&element), sizeof(T*));
}

// This function is used to write an element to the disk
template<typename T>
void WriteToDisk::single(T* element)
{
    array(element, 1);
}

// This function is used to write an array of elements to the disk
template<typename T>
void WriteToDisk::array(T* element, int n)
{
    if (n > 0)
    {
        m_pParams->append((char*)element, sizeof(T) * n);
    }
}

// This function is used to write a char array to the disk. It writes the size and then the char array
void WriteToDisk::charArray(const char* s)
{
    size_t sz = strlen(s) + 1;
    single(&sz);
    m_pParams->append(s, sz);
}

/*************************************************************************/
/*                           class ReadFromDisk                          */
/*************************************************************************/
// This function is used to read a pointer from disk
template<typename T>
void ReadFromDisk::head(T*& element)
{
    m_pParams->getCurrentData(sizeof(T*), (char*)(&element));
}

// This function is used to read an element. It allocates memory for it set the pointer and then reads
// the data to the allocated memory
template<typename T>
void ReadFromDisk::single(T*& element)
{
    element = (T*)m_recipeAllocator->allocate(sizeof(T));
    m_pParams->getCurrentData(sizeof(T), (char*)element);
}

// This function is used to read an array of elements. It allocates memory for it set the pointer and then reads
// the data to the allocated memory
template<typename T>
void ReadFromDisk::array(T*& element, int n)
{
    if (n > 0)
    {
        element = (T*)m_recipeAllocator->allocate(n * sizeof(T));
        m_pParams->getCurrentData(sizeof(T) * n, (char*)element);
    }
    else
    {
        element = nullptr;  // This is for cases there are no elements but memory was allocated for array of size 0
    }
}

// This function is used to read a char array from the disk. It reads the size, allocate memory and read the char array
void ReadFromDisk::charArray(const char*& s)
{
    size_t sz;
    m_pParams->getCurrentData(sizeof(sz), (char*)(&sz));
    array(s, sz);
}
