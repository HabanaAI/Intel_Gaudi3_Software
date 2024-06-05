#include "dfa_defines.hpp"
#include "recipe_logger.hpp"
#include "recipe_handle_impl.hpp"
#include "recipe.h"
#include "log_manager.h"
#include "habana_global_conf_runtime.h"
#include "recipe_dynamic_info.hpp"
#include "defs.h"

// Todo [SW-52887] Remove dumpRecipeToLogScreen
class dumpRecipeToLogScreen
{
public:
    dumpRecipeToLogScreen(const DeviceAgnosticRecipeInfo& deviceAgnosticInfo,
                          const basicRecipeInfo&          basicRecipeInfo,
                          uint64_t                        id,
                          bool                            log,
                          bool                            screen,
                          synapse::LogManager::LogType    logger = synapse::LogManager::LogType::SYN_API)
    : m_deviceAgnosticInfo(deviceAgnosticInfo),
      m_basicRecipeInfo(basicRecipeInfo),
      m_id(id),
      m_log(log),
      m_screen(screen),
      m_logger(logger),
      m_logLevel(SPDLOG_LEVEL_INFO)
    {
    }
    void dump();

private:
    void dump_dsd();
    void dumpTensorInfo(uint64_t idx, const tensor_info_t& tInfo);
    void dumpDevAgnosticInfo();
    void dumpRecipeT();
    void dumpRecipeConf();

    // Todo rename to dump
    template<typename... Args>
    void out(const char* format, Args... args)
    {
        outCollect(true, format, args...);
    }
    void outCollect(bool done, const char* format)
    {
        outCollect(true, "%s", format);
    }  // Special case - only string is given. Compiler doesn't like it
    template<typename... Args>
    void outCollect(bool done, const char* format, Args... args);

    static const int OUT_SIZE = 1000;

    const DeviceAgnosticRecipeInfo& m_deviceAgnosticInfo;
    const basicRecipeInfo&          m_basicRecipeInfo;
    uint64_t                        m_id;
    bool                            m_log;
    bool                            m_screen;
    synapse::LogManager::LogType    m_logger;
    int                             m_position = 0;
    char                            m_buffer[OUT_SIZE];
    uint32_t                        m_logLevel;
};

template<typename... Args>
void dumpRecipeToLogScreen::outCollect(bool done, const char* format, Args... args)
{
    m_position += snprintf(m_buffer + m_position, OUT_SIZE - m_position, format, args...);
    if (done)
    {
        if (m_screen) printf("%s\n", m_buffer);
        if (m_log) SYN_LOG(m_logger, m_logLevel, "{}", m_buffer);
        m_position = 0;
    }
}

void dumpRecipeToLogScreen::dump()
{
    SYN_LOG(m_logger, m_logLevel, "--- Dump recipe 0x{:x} ---", m_id);

    dumpDevAgnosticInfo();
    dumpRecipeT();
    dump_dsd();

    out("===== Done =====");
}

static std::string confEnum2string(gc_conf_t::recipeCompileParams param)
{
    switch (param)
    {
        case gc_conf_t::DEVICE_TYPE:            return "DEVICE_TYPE";
        case gc_conf_t::TIME_STAMP:             return "TIME_STAMP";
        case gc_conf_t::TPC_ENGINE_MASK:        return "TPC_ENGINE_MASK";
        case gc_conf_t::MME_NUM_OF_ENGINES:     return "MME_NUM_OF_ENGINES";
        case gc_conf_t::DMA_ENGINE_MASK:        return "DMA_ENGINE_MASK";
        case gc_conf_t::RCP_NOT_USED:           return "RCP_NOT_USED";
        case gc_conf_t::ROTATOR_NUM_OF_ENGINES: return "ROTATOR_NUM_OF_ENGINES";
        case gc_conf_t::LAST_COMPILE_PARAM:     return "LAST_COMPILE_PARAM";
    }
    return "INVALID PARAM";
}

void dumpRecipeToLogScreen::dumpRecipeConf()
{
    const recipe_t* pRecipe = m_basicRecipeInfo.recipe;

    out("===== recipe has %d config params =====", pRecipe->recipe_conf_nr);

    for (uint32_t i = 0; i < pRecipe->recipe_conf_nr; i++)
    {
        out("conf %d %-25s %16lX",
            i,
            confEnum2string(pRecipe->recipe_conf_params[i].conf_id).c_str(),
            pRecipe->recipe_conf_params[i].conf_value);
    }
}

void dumpRecipeToLogScreen::dumpRecipeT()
{
    const recipe_t* pRecipe = m_basicRecipeInfo.recipe;
    HB_ASSERT(pRecipe != nullptr, "Invalid recipe pointer");
    out("Name: %s", pRecipe->name ? pRecipe->name : "Recipe has no name");

    dumpRecipeConf();

    out("Version:%X/%X", pRecipe->version_major, pRecipe->version_minor);
    out("blobs_nr:%lX programs_nr:%lX activate_jobs_nr:%lX execute_jobs_nr:%lX persist_tensors_nr:%lX "
        "permute_tensors_views_nr:%lX  program_data_blobs_nr:%lX patch_points_nr:%lX activate_patch_points_nr:%lX h2di_tensors_nr:%lX",
        pRecipe->blobs_nr,
        pRecipe->programs_nr,
        pRecipe->activate_jobs_nr,
        pRecipe->execute_jobs_nr,
        pRecipe->persist_tensors_nr,
        pRecipe->permute_tensors_views_nr,
        pRecipe->program_data_blobs_nr,
        pRecipe->patch_points_nr,
        pRecipe->activate_patch_points_nr,
        pRecipe->h2di_tensors_nr);
    out("sections_nr:%lX workspace_nr:%lX", pRecipe->sections_nr, pRecipe->workspace_nr);

    out("execution_blobs_buffer:%lX execution_blobs_buffer_size:%lX",
        TO64(pRecipe->execution_blobs_buffer),
        pRecipe->execution_blobs_buffer_size);
    out("patching_blobs_buffer: %lX patching_blobs_buffer_size: %lX",
        TO64(pRecipe->patching_blobs_buffer),
        pRecipe->patching_blobs_buffer_size);
    out("dynamic_blobs_buffer:  %lX dynamic_blobs_buffer_size   %lX",
        TO64(pRecipe->dynamic_blobs_buffer),
        pRecipe->dynamic_blobs_buffer_size);

    out("==== arcs %x =====", pRecipe->arc_jobs_nr);
    for (int i = 0; i < pRecipe->arc_jobs_nr; i++)
    {
        out("%X) engine %X dynamic addr/size %lX/%X static addr/size %lX/%X\n",
            i,
            pRecipe->arc_jobs[i].logical_engine_id,
            pRecipe->arc_jobs[i].dynamic_ecb.cmds,
            pRecipe->arc_jobs[i].dynamic_ecb.cmds_size,
            pRecipe->arc_jobs[i].static_ecb.cmds,
            pRecipe->arc_jobs[i].static_ecb.cmds_size);
    }

    out("======= activate_jobs ======");
    for (uint i = 0; i < pRecipe->activate_jobs_nr; i++)
    {
        auto& currJob = pRecipe->activate_jobs[i];
        out("activate job:%4X engine:%3lX program:%3lX", i, currJob.engine_id, currJob.program_idx);
    }
    out("======= execute_jobs ======");
    for (uint i = 0; i < pRecipe->execute_jobs_nr; i++)
    {
        auto& currJob = pRecipe->execute_jobs[i];
        out("execute job:%4X engine:%3lX program:%3lX", i, currJob.engine_id, currJob.program_idx);
    }
    out("====== Tensors =====");
    for (uint i = 0; i < pRecipe->persist_tensors_nr; i++)
    {
        auto& currTens = pRecipe->tensors[i];
        out("%4X) %-40s section:%4lX, offset: %8lX size:%8lX elementType:%8lX section_type %X multi_views_nr:%8lX",
            i,
            currTens.name,
            currTens.section_idx,
            currTens.offset_in_section,
            currTens.size,
            currTens.elementType,
            currTens.section_type,
            currTens.multi_views_indices_nr);
    }

    out("====== workspace_sizes =====");
    for (int i = 0; i < pRecipe->workspace_nr; i++)
    {
        out("workspace %X size: %lX", i, pRecipe->workspace_sizes[i]);
    }

    out("====== Const sections ======");
    const_section_t* constSections       = pRecipe->const_sections;
    uint32_t         constSectionsAmount = pRecipe->const_sections_nr;
    for (int i = 0; i < constSectionsAmount; i++, constSections++)
    {
        out("%4X) const section_idx %4X section_size %16lX section_addr %16lX {}",
            i,
            constSections->section_idx,
            constSections->size,
            (uint64_t)constSections->data,
            ((uint64_t)constSections->data == INVALID_CONST_SECTION_DATA) ? "(data not set)" : "");
    }

    out("===== section groups ======= %X", pRecipe->section_groups_nr);
    for (int i = 0; i < pRecipe->section_groups_nr; i++)
    {
        auto& curr = pRecipe->section_groups_patch_points[i];
        out("%4X) section_group %X patch_points_nr %X", i, curr.section_group, curr.patch_points_nr);
    }

    //reduce log level to trace
    uint32_t origLogLevel = m_logLevel;
    m_logLevel = SPDLOG_LEVEL_TRACE;

    out("====== Stage ======");
    out(" Enable : %s Number nodes: %X. List of PP/blobs",
        GCFG_ENABLE_STAGED_SUBMISSION.value() ? "Yes" : "No",
        pRecipe->node_nr);

    for (uint32_t node = 0; node < pRecipe->node_nr; node++)
    {
        outCollect(false, "Node %3X PP: %5X  blobs:", node, pRecipe->node_exe_list[node].patch_points_nr);
        for (uint32_t program = 0; program < pRecipe->programs_nr; program++)
        {
            outCollect(false, "%4X ", pRecipe->node_exe_list[node].program_blobs_nr[program]);
        }
        out("");
    }

    out("====== Blobs: req-patching/static-exec/dynamic_exec ======");
    for (uint i = 0; i < pRecipe->blobs_nr; i++)
    {
        auto& currBlob = pRecipe->blobs[i];
        out("%4X) %s%s%s data:%16lX size:%8lX End:%16lX",
            i,
            currBlob.blob_type.requires_patching ? "*" : " ",
            currBlob.blob_type.static_exe        ? "*" : " ",
            currBlob.blob_type.dynamic_exe       ? "*" : " ",
            TO64(currBlob.data),
            currBlob.size,
            TO64(currBlob.data) + currBlob.size);
    }

    for (uint i = 0; i < pRecipe->programs_nr; i++)
    {
        auto& currProg = pRecipe->programs[i];
        out("============= Program %X (size %lX) ============", i, currProg.program_length);
        for (uint j = 0; j < currProg.program_length; j++)
        {
            outCollect(false, "%5lX", currProg.blob_indices[j]);
            if ((j % 8) == 7) out("");
        }
        out("");
    }

    out("====== program_data_blobs ======");
    out("program_data_blobs_buffer:%lX program_data_blobs_size:%lX",
        TO64(pRecipe->program_data_blobs_buffer),
        pRecipe->program_data_blobs_size);
    for (uint i = 0; i < pRecipe->program_data_blobs_nr; i++)
    {
        auto& currProgData = pRecipe->program_data_blobs[i];
        out("%4X) section:%8lX data:%16lX size:%8lX offset:%8lX",
            i,
            currProgData.section_idx,
            TO64(currProgData.data),
            currProgData.size,
            currProgData.offset_in_section);
    }
    out("valid_nop_kernel %s nop_kernel_section %d nop_kernel_offset %X",
        pRecipe->valid_nop_kernel ? "Y" : "N",
        pRecipe->nop_kernel_section,
        pRecipe->nop_kernel_offset);

    out("max_used_mcid_discard %d max_used_mcid_degrade %d",
        pRecipe->max_used_mcid_discard,
        pRecipe->max_used_mcid_degrade);

    out("====== Patch points ======");
    for (int i = 0; i < pRecipe->patch_points_nr; i++)
    {
        auto currPP = pRecipe->patch_points[i];
        out("%4X) type:%X blob:%4lX offset:%8lX effective_addr:%16lX section:%8lX node %lX addr %16lX",
            i,
            currPP.type,
            currPP.blob_idx,
            currPP.dw_offset_in_blob,
            currPP.memory_patch_point.effective_address,
            currPP.memory_patch_point.section_idx,
            currPP.node_exe_index,
            (uint64_t)(pRecipe->blobs[currPP.blob_idx].data) + (uint64_t)currPP.dw_offset_in_blob * 4);
    }

    //restore log level
    m_logLevel = origLogLevel;

#if 0
    out("====== Blobs Data ======");
    for (uint i = 0; i < pRecipe->blobs_nr; i++)
    {
        auto& currBlob = pRecipe->blobs[i];
        out("%4X) %s key:%16lX data:%16lX size:%8lX End:%16lX", i, currBlob.blob_type.requires_patching ? "Y":"N", currBlob.unique_key,
            TO64(currBlob.data), currBlob.size, TO64(currBlob.data) + currBlob.size);
        for(int i = 0; i < currBlob.size; i=i+4)
        {
            outCollect(false, "%08X ", ((uint32_t*)(currBlob.data))[i]);
            if (i % 32 == 28) out("");
        }
        out("");
    }
#endif
}

void dumpRecipeToLogScreen::dumpTensorInfo(uint64_t idx, const tensor_info_t& tInfo)
{
    outCollect(false,
               "%5X) dims %ld data_type %d tensor_type %d user type %d Index %8d ",
               idx,
               tInfo.infer_info.geometry.dims,
               tInfo.data_type,
               tInfo.tensor_type,
               tInfo.user_tensor_type,
               tInfo.tensor_db_index);
    if (tInfo.tensor_db_index != INVALID_TENSOR_INDEX)
    {
        outCollect(false, " %s ", DynamicRecipe::staticGetTensorName(idx, &m_basicRecipeInfo));
    }
    for (int j = 0; j < MAX_DIMENSIONS_NUM; j++)
    {
        outCollect(false,
                   "curr/max/min/strides %d/%d/%d/%d  ",
                   tInfo.infer_info.geometry.maxSizes[j],
                   tInfo.max_dims[j],
                   tInfo.min_dims[j],
                   tInfo.strides[j]);
    }
    out("");
}

void dumpRecipeToLogScreen::dump_dsd()
{
    const shape_plane_graph_t* p_dsd = m_basicRecipeInfo.shape_plan_recipe;

    if (p_dsd == nullptr)
    {
        out("Not DSD");
        return;
    }

    out("====== version major/minor %d/%d", p_dsd->version_major, p_dsd->version_minor);
    out("=====  sp tensors (%lX) ======", p_dsd->sp_tensors_nr);
    for (uint64_t i = 0; i < p_dsd->sp_tensors_nr; i++)
    {
        tensor_info_t& tInfo = p_dsd->sp_tensors[i];
        dumpTensorInfo(i, tInfo);
    }

    //reduce log level to trace
    uint32_t origLogLevel = m_logLevel;
    m_logLevel = SPDLOG_LEVEL_TRACE;

    out("=====  sp nodes (%lX) ======", p_dsd->sp_node_nr);
    for (uint64_t i = 0; i < p_dsd->sp_node_nr; i++)
    {
        shape_plane_node_t& node = p_dsd->sp_nodes[i];

        out("--- node %X %s has %X sub nodes ---", i, node.node_name, node.basic_nodes_nr);
        outCollect(false, "inputs %X:", node.input_tensors_nr);
        for (int j = 0; j < node.input_tensors_nr; j++)
        {
            outCollect(false, "%lX ", node.input_tensors[j]);
            if (j % 8 == 7) out("");
        }
        out("");
        outCollect(false, "outputs %X:", node.output_tensors_nr);
        for (int j = 0; j < node.output_tensors_nr; j++)
        {
            outCollect(false, "%lX ", node.output_tensors[j]);
            if (j % 8 == 7) out("");
        }
        out("");
        outCollect(false,
                   "node_match_output_tensors_nr %X (pointers %lX %lX):",
                   node.node_match_output_tensors_nr,
                   TO64(node.output_src_tensors),
                   TO64(node.output_dst_tensors));
        for (int j = 0; j < node.node_match_output_tensors_nr; j++)
        {
            outCollect(false, "%lX->%lX ", node.output_src_tensors[j], node.output_dst_tensors[j]);
            if (j % 8 == 7) out("");
        }
        out("");
        out("rois_nr %lX pp_nr %lX sif %lX",
            node.activation_rois_nr,
            node.node_patch_points_nr,
            node.basic_nodes[0].sif_id);
        outCollect(false, "smf_id/blob/offset/size/type:");
        for (int j = 0; j < node.node_patch_points_nr; j++)
        {
            sm_patch_point_t& smPP = node.node_patch_points[j];
            out("%4X) %8lX %4X %3X %2X %2X", j, smPP.smf_id, smPP.blob_idx, smPP.dw_offset_in_blob, smPP.patch_size_dw, smPP.patch_point_type);
        }
        out("node_db_tensors_nr %X", node.node_db_tensors_nr);
        for (int j = 0; j < node.node_db_tensors_nr; j++)
        {
            tensor_info_t& tInfo = node.node_db_tensors[j];
            dumpTensorInfo(j, tInfo);
        }

        for (int sub = 0; sub < node.basic_nodes_nr; sub++)
        {
            auto subNode = node.basic_nodes[sub];
            out("-- subNode %x %s -- sif %X", sub, subNode.node_name, subNode.sif_id.sm_func_index);
            out("input_tensors_nr %X", subNode.input_tensors_nr);
            for (int in = 0; in < subNode.input_tensors_nr; in++)
            {
                outCollect(false, "idx/type %X/%X ", subNode.input_tensors[in], subNode.input_tensors_db[in]);
                if (in % 8 == 7) out("");
            }
            out("");
            out("output_tensors_nr %X", subNode.output_tensors_nr);
            for (int output = 0; output < subNode.output_tensors_nr; output++)
            {
                outCollect(false, "idx/type %X/%X ", subNode.output_tensors[output], subNode.output_tensors_db[output]);
                if (output % 8 == 7) out("");
            }
            out("");
        }

        out("");
    }

    //restore log level
    m_logLevel = origLogLevel;
}

void dumpRecipeToLogScreen::dumpDevAgnosticInfo()
{
    const DeviceAgnosticRecipeInfo& agnosticRecipeInfo = m_deviceAgnosticInfo;

    out("deviceType %d", agnosticRecipeInfo.m_deviceType);
    out("workspaceSize 0x%lX", agnosticRecipeInfo.m_workspaceSize);
    out("isInitialized %d", agnosticRecipeInfo.m_isInitialized);
    out("getNumberOfExternalTensors %ld", agnosticRecipeInfo.m_signalFromGraphInfo.getNumberOfExternalTensors());

    // Consider logging more information from the following classes
    // RecipeTensorsInfo, RecipeDsdStaticInfo, RecipeStaticInfoScal, SignalFromGraphInfo, RecipeStageSubmisssionInfo,
    // DeviceAgnosticRecipeStaticInfo
}

void RecipeLogger::dfaDumpRecipe(const InternalRecipeHandle* internalRecipeHandle,
                                 bool                        isScalDev,
                                 const std::string&          callerMsg)
{
    return dfaDumpRecipe(internalRecipeHandle->deviceAgnosticRecipeHandle,
                         internalRecipeHandle->basicRecipeHandle,
                         isScalDev,
                         internalRecipeHandle->recipeSeqNum,
                         callerMsg);
}

void RecipeLogger::dfaDumpRecipe(const DeviceAgnosticRecipeInfo& deviceAgnosticInfo,
                                 const basicRecipeInfo&          basicRecipeInfo,
                                 bool                            isScalDev,
                                 uint64_t                        id,
                                 const std::string&              callerMsg)
{
    LOG_INFO(SYN_DEV_FAIL, "Dumping recipe to {}", SUSPECTED_RECIPES);

    RecipeLogger::dumpRecipe(deviceAgnosticInfo,
                             basicRecipeInfo,
                             id,
                             true,
                             false,
                             synapse::LogManager::LogType::SYN_FAIL_RECIPE);
    RecipeLogger::dumpSyncScheme(basicRecipeInfo, isScalDev, id, callerMsg);
}

void RecipeLogger::dumpRecipe(InternalRecipeHandle*        pRecipeHandle,
                              bool                         log,
                              bool                         screen,
                              synapse::LogManager::LogType logger)
{
    return dumpRecipe(pRecipeHandle->deviceAgnosticRecipeHandle,
                      pRecipeHandle->basicRecipeHandle,
                      pRecipeHandle->recipeSeqNum,
                      log,
                      screen,
                      logger);
}

void RecipeLogger::dumpRecipe(const DeviceAgnosticRecipeInfo& deviceAgnosticInfo,
                              const basicRecipeInfo&          basicRecipeInfo,
                              uint64_t                        id,
                              bool                            log,
                              bool                            screen,
                              synapse::LogManager::LogType    logger)
{
    HB_ASSERT(basicRecipeInfo.recipe != nullptr, "basicRecipeHandle.recipe == nullptr");
    dumpRecipeToLogScreen x(deviceAgnosticInfo, basicRecipeInfo, id, log, screen, logger);
    x.dump();
}

static std::string EngType2string(Recipe::EngineType type)
{
    switch (type)
    {
        case Recipe::EngineType::DMA: return "DMA";
        case Recipe::EngineType::MME: return "MME";
        case Recipe::EngineType::ROT: return "ROT";
        case Recipe::EngineType::TPC: return "TPC";
        case Recipe::EngineType::CME: return "CME";
        default: return "Unknown " + std::to_string(type);
    }
}

void RecipeLogger::dumpSyncScheme(const basicRecipeInfo& recipeInfo,
                                  bool                   isScalDev,
                                  uint64_t               id,
                                  const std::string&     callerMsg)
{
    const char* name = recipeInfo.recipe->name;

    LOG_INFO(SYN_FAIL_RECIPE, "#Dump sync-scheme 0x{:x} stream {} {}", id, callerMsg, name ? name : "No recipe name");

    debug_sync_scheme_t* p = &recipeInfo.recipe->debug_sync_scheme_info;

    LOG_TRACE(SYN_FAIL_RECIPE,
              "#number of nodes {} scalDev{}. node) engine_type, node_exe_index, pipe_level, emitted_signal",
              p->node_sync_info_nr,
              isScalDev);

    if (isScalDev)
    {
        node_sync_info_arc_t* nodeSyncInfoArc = p->node_sync_info_arc;

        for (uint64_t node = 0; node < p->node_sync_info_nr; node++)
        {
            const node_sync_info_arc_t& infoArc = nodeSyncInfoArc[node];

            std::string engStr = EngType2string(infoArc.engine_type);

            LOG_TRACE(SYN_FAIL_RECIPE,
                      "{:5}) {:4} {:5} {:5} {:5}",
                      node,
                      engStr,
                      infoArc.node_exe_index,
                      infoArc.pipe_level,
                      infoArc.emitted_signal);
        }
    }
    else
    {
        node_sync_info_legacy_t* nodeSyncInfoLegacy = p->node_sync_info_legacy;

        for (uint64_t node = 0; node < p->node_sync_info_nr; node++)
        {
            const node_sync_info_legacy_t& infoLegacy = nodeSyncInfoLegacy[node];

            LOG_TRACE(SYN_FAIL_RECIPE,
                      "{:5}) {:4} {:5} {:5} {:5} {:6} {:4}",
                      node,
                      infoLegacy.engine_type,
                      infoLegacy.node_exe_index,
                      infoLegacy.pipe_level,
                      infoLegacy.emitted_signal,
                      infoLegacy.sob_id,
                      infoLegacy.num_engines);
        }
    }
    LOG_INFO(SYN_FAIL_RECIPE, "#Done");
}
