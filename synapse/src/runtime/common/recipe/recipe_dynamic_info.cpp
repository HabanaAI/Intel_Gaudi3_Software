#include "recipe_dynamic_info.hpp"
#include "recipe_handle_impl.hpp"
#include "define_synapse_common.hpp"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "log_manager.h"
#include "types.h"
#include "utils.h"
#include "vtune_stat.h"
#include "recipe_utils.hpp"

#include "infra/defs.h"
#include "graph_compiler/smf/shape_func_registry.h"

#include <string.h>

/*
 ***************************************************************************************************
 *   @brief This class is used to patch dynamic recipe
 *          It is assumed that each instance is for a specific recipe/stream
 *          The constructor sets members of the class that are set one-time only but used per run
 *          and are different between recipes and different streams
 *
 *   @param  pRecipeInfo
 *   @return
 *
 *   The functions copy data that is per thread from the recipe to local memory so multiple streams can run in parallel
 *
 ***************************************************************************************************
 */
DynamicRecipe::DynamicRecipe(const basicRecipeInfo&            rRecipeInfo,
                             const DeviceAgnosticRecipeInfo&   rDeviceAgnosticRecipeInfo,
                             const DataChunkSmPatchPointsInfo* pDataChunkSmPatchPointsInfo,
                             const data_chunk_patch_point_t*   originalPatchPoints,
                             const RecipeAddrPatcher*          pRecipeAddrPatcher)
: m_rRecipeInfo(rRecipeInfo),
  m_rDeviceAgnosticRecipeInfo(rDeviceAgnosticRecipeInfo),
  m_pDataChunkSmPatchPointsInfo(pDataChunkSmPatchPointsInfo),
  m_isStaticTensors(rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_isStaticTensors),
  m_sfr(ShapeFuncRegistry::instance()),
  m_patchingStatus({DsdPatchingState::PRE_SIF, nullptr})
{
    // Allocate memory that is needed per recipe/per stream

    STAT_GLBL_START(initDynamicRecipe);

    m_originalPatchPoints       = originalPatchPoints;
    m_originalRecipeAddrPatcher = pRecipeAddrPatcher;

    shape_plane_graph_t* shapePlaneRecipe = rRecipeInfo.shape_plan_recipe;

    // Set vector sizes. The vectors are used to save a copy of fields in the recipe that change per run
    m_sp_tensors_private.resize(
        shapePlaneRecipe->sp_tensors_nr);  // Copy of the tensors, each stream should work on its own copy
    size_t tensorsCpySize = m_sp_tensors_private.size() * sizeof(m_sp_tensors_private[0]);
    memcpy(m_sp_tensors_private.data(),
           shapePlaneRecipe->sp_tensors,
           tensorsCpySize);  // Copy once, so we have the right non-dynamic tensor sizes set

    // Fuser
    auto fuserMaxIn        = rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_fuserMaxIn;
    auto fuserMaxOut       = rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_fuserMaxOut;
    auto fuserMaxDbTensors = rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_fuserMaxDbTensors;

    uint16_t maxFuserInvalidArrSize =
        (fuserMaxDbTensors == 0) ? uint16_t {1} : (uint16_t)(((fuserMaxOut - 1) / BITS_IN_UNSIGNED) + 1);

    m_fuserSifInTensors.resize(fuserMaxIn);
    m_fuserSifOutTensors.resize(fuserMaxOut);
    m_fuserNodeDbTensors.resize(fuserMaxDbTensors);
    m_fuserInvalidMask.resize(maxFuserInvalidArrSize);
    m_fuserSifParams.inputTensors   = m_fuserSifInTensors.data();
    m_fuserSifOutputs.outputTensors = m_fuserSifOutTensors.data();
    m_fuserSifOutputs.invalidMask   = m_fuserInvalidMask.data();
    // Fuser Ends

    unsigned maxTpcEngines =
        m_rDeviceAgnosticRecipeInfo.m_deviceType == synDeviceGaudi3 ? 64 : (m_rDeviceAgnosticRecipeInfo.m_deviceType == synDeviceGaudi2 ? 24 : 8);
    m_availableTpcEngines = countSetBits(RecipeUtils::getConfVal(rRecipeInfo.recipe, gc_conf_t::TPC_ENGINE_MASK).value(), maxTpcEngines);

    auto     maxNodeOutputs = rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_maxNodeOutputs;
    uint16_t maxInvalidArrSize =
        (maxNodeOutputs == 0) ? uint16_t {1} : (uint16_t)(((maxNodeOutputs - 1) / BITS_IN_UNSIGNED) + 1);
    m_invalidMaskArr.resize(maxInvalidArrSize);
    m_nodeData_private.resize(shapePlaneRecipe->sp_node_nr);  // scratch pad, no need to copy for now
    m_NodeParams.resize(
        shapePlaneRecipe->sp_node_nr);  // Prepare sif and smf function params once for quick usage every run

    if (m_originalPatchPoints != nullptr)
    {
        m_patchPoints.resize(rRecipeInfo.recipe->patch_points_nr + 1);  // +1 for end of stages indication
    }
    else if (m_originalRecipeAddrPatcher != nullptr)
    {
        m_recipeAddrPatcher = *m_originalRecipeAddrPatcher;
    }

    // Set values that don't change between runs
    const data_chunk_sm_patch_point_t* pCurrentDcSmPatchPoint =
        (m_pDataChunkSmPatchPointsInfo != nullptr) ? m_pDataChunkSmPatchPointsInfo->m_dataChunkSmPatchPoints : nullptr;
    for (int node = 0; node < m_NodeParams.size(); node++)
    {
        shape_plane_node_t& currNode       = shapePlaneRecipe->sp_nodes[node];
        auto&               currNodeParams = m_NodeParams[node];

        // SMF info
        // Note: the address of the vectors below are used later in the function so they have to be set first (or at
        // least resized)
        currNodeParams.tensorSmfInVec.resize(currNode.input_tensors_nr);
        currNodeParams.hasHostTensor = false;
        for (int t = 0; t < currNode.input_tensors_nr; t++)
        {
            uint64_t tensorIdx               = currNode.input_tensors[t];
            currNodeParams.tensorSmfInVec[t] = &m_sp_tensors_private[tensorIdx];
            if (m_sp_tensors_private[tensorIdx].tensor_flags & tensor_info_t::HAS_HOST_ADDRESS)
            {
                currNodeParams.hasHostTensor = true;
            }
        }

        m_NodeParams[node].tensorSmfOutVec.resize(currNode.output_tensors_nr);
        for (size_t t = 0; t < currNode.output_tensors_nr; t++)
        {
            uint64_t tensorIdx                = currNode.output_tensors[t];
            currNodeParams.tensorSmfOutVec[t] = &m_sp_tensors_private[tensorIdx];
        }

        // SIF info
        if (currNode.basic_nodes_nr == 1)  // if >1 then it is fuser, will be done during each run
        {
            currNodeParams.tensorSifInVec.resize(currNode.input_tensors_nr);
            for (int t = 0; t < currNode.input_tensors_nr; t++)
            {
                uint64_t tensorIdx               = currNode.input_tensors[t];
                currNodeParams.tensorSifInVec[t] = &m_sp_tensors_private[tensorIdx].infer_info;
            }

            m_NodeParams[node].tensorSifOutVec.resize(currNode.output_tensors_nr);
            for (size_t t = 0; t < currNode.output_tensors_nr; t++)
            {
                uint64_t tensorIdx                = currNode.output_tensors[t];
                currNodeParams.tensorSifOutVec[t] = &m_sp_tensors_private[tensorIdx].infer_info;
            }

            currNodeParams.sifParams.inputTensors    = currNodeParams.tensorSifInVec.data();
            currNodeParams.sifParams.inputTensorsNr  = (unsigned)currNode.input_tensors_nr;
            currNodeParams.sifParams.outputTensorsNr = (unsigned)currNode.output_tensors_nr;

            currNodeParams.sifParams.nodeParams.nodeParamsSize = currNode.basic_nodes[0].sif_params_nr;
            currNodeParams.sifParams.nodeParams.nodeParams     = currNode.basic_nodes[0].sif_params;

            currNodeParams.sifParams.inputPermutations = currNode.basic_nodes[0].input_permutations;

            currNodeParams.sifOutputs.outputTensors = currNodeParams.tensorSifOutVec.data();
            currNodeParams.sifOutputs.invalidMask   = m_invalidMaskArr.data();

            currNodeParams.invalidArrSize = (currNode.output_tensors_nr == 0)
                                                ? uint16_t {1}
                                                : (uint16_t)(((currNode.output_tensors_nr - 1) / BITS_IN_UNSIGNED) + 1);

            std::tie(currNodeParams.sifFunc, currNodeParams.sifParams.pGuid) = m_sfr.getSIFandGuidInfo(currNode.basic_nodes[0].sif_id);
            // TODO: remove this when [SW-159125] is done
            // pGuid can be nullptr if not a dynamic node, so guid is not needed in this case
            if (currNodeParams.sifParams.pGuid)
            {
                currNodeParams.sifParams.guid = *currNodeParams.sifParams.pGuid;
            }
            currNodeParams.sifParams.maxAvailableTpc = m_availableTpcEngines;
        }

        currNodeParams.smfParams.inputTensorsNr  = currNode.input_tensors_nr;
        currNodeParams.smfParams.inputTensors    = currNodeParams.tensorSmfInVec.data();
        currNodeParams.smfParams.outputTensorsNr = currNode.output_tensors_nr;
        currNodeParams.smfParams.outputTensors   = currNodeParams.tensorSmfOutVec.data();

        // SMF params per patch point
        currNodeParams.PPparamsVec.resize(currNode.node_patch_points_nr);
        for (uint64_t pp = 0; pp < currNode.node_patch_points_nr; pp++)
        {
            const data_chunk_sm_patch_point_t& currPP = *pCurrentDcSmPatchPoint;
            const sm_function_id_t&            smfId  = *(currPP.p_smf_id);

            pCurrentDcSmPatchPoint++;
            if (currPP.patch_point_type == FIELD_DYNAMIC_ADDRESS)
            {
                currNodeParams.PPparamsVec[pp].shapeManOut.outputPatchValues = m_originalPatchPoints != nullptr ?
                    (uint32_t*)(&m_patchPoints[currPP.patch_point_idx_low].memory_patch_point.effective_address) :
                    (uint32_t*)(m_recipeAddrPatcher.getPatchPointEffectiveAddr(currPP.patch_point_idx_low));
            }
            currNodeParams.PPparamsVec[pp].shapeManOut.nodeData = &m_nodeData_private[node][0];
            currNodeParams.PPparamsVec[pp].smfFunc              = m_sfr.getSMF(smfId);
        }
        currNodeParams.smfParams.nodeIdx = node;
    }

    STAT_GLBL_COLLECT_TIME(initDynamicRecipe, globalStatPointsEnum::initDynamicRecipe);
}

/*
 ***************************************************************************************************
 *   @brief This function does the shape inference and verify the outputs sizes are valid
 *
 *   @param  launchTensorsInfo, launchTensorsAmount: tensors from user
 *   @param  tensorIdx2userIdx: two vectors mapping the tensor lists (persistent, shape) from the recipe to the user
 *tensors
 *   @param  blobs: to be patched
 ***************************************************************************************************
 */
bool DynamicRecipe::runSifOnAllNodes(const synLaunchTensorInfoExt* launchTensorsInfo,
                                     const uint32_t                launchTensorsAmount,
                                     const std::vector<uint32_t>*  tensorIdx2userIdx,
                                     uint64_t                      programDataHostAddress)
{
    bool status = runSifOnNodes(launchTensorsInfo, launchTensorsAmount, tensorIdx2userIdx, programDataHostAddress);
    if (!status)
    {
        return false;
    }

    // Verify output tensors
    {
        STAT_GLBL_START(verifyOutputsTime);
        bool resRunVerify = verifyOutputs();
        if (!resRunVerify)
        {
            STAT_EXIT_NO_COLLECT();
            return resRunVerify;
        }
        STAT_GLBL_COLLECT_TIME(verifyOutputsTime, globalStatPointsEnum::DsdSVerifyOutputs);
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function does the dynamic patching by calling the relevant functions
 *
 *   @param  launchTensorsInfo, launchTensorsAmount: tensors from user
 *   @param  tensorIdx2userIdx: two vectors mapping the tensor lists (persistent, shape) from the recipe to the user
 *tensors
 *   @param  blobs: to be patched
 ***************************************************************************************************
 */
bool DynamicRecipe::patch(const synLaunchTensorInfoExt* launchTensorsInfo,
                          const uint32_t                launchTensorsAmount,
                          const std::vector<uint64_t>&  dataChunksHostAddresses,
                          const std::vector<uint32_t>*  tensorIdx2userIdx)
{
    bool status =
        runSifOnAllNodes(launchTensorsInfo, launchTensorsAmount, tensorIdx2userIdx, 0 /* programDataHostAddress - NA*/);
    if (!status)
    {
        return false;
    }

    return runSmfOnAllNodes(dataChunksHostAddresses);
}

/*
 ***************************************************************************************************
 *   @brief This function does the shape inference.
 *          It runs over the node and runs the SIF function
 *
 *   @param  launchTensorsInfo:   user's tensors
 *   @param  launchTensorsAmount: user's tensors-amount
 *   @param  tensorIdx2userIdx:   two vectors mapping the tensor lists (persistent, shape),
 *                                from the recipe to the user tensors
 *
 *   @return true=OK
 ***************************************************************************************************
 */
bool DynamicRecipe::runSifOnNodes(const synLaunchTensorInfoExt* launchTensorsInfo,
                                  const uint32_t                launchTensorsAmount,
                                  const std::vector<uint32_t>*  tensorIdx2userIdx,
                                  uint64_t                      programDataHostAddress)
{
    LOG_DSD_INFO("shape inference");

    if (m_patchingStatus.patchingState != DsdPatchingState::PRE_SIF)
    {
        LOG_DSD_WARN("SIF overrides previous SIF");
    }

    if (!init(launchTensorsInfo, launchTensorsAmount, tensorIdx2userIdx))
    {
        return false;
    }

    shape_plane_graph_t* shapePlanGraph = m_rRecipeInfo.shape_plan_recipe;

    // NOTE: we have to run SIF first and only then SMF. For broadcast, the smf needs the tensor size
    //       that is given by a different node. That node might be after the broadcast node. So we can't run SMF
    //       until all sizes are calculated
    STAT_GLBL_START(sifTime);
    for (uint64_t nodeIdx = 0; nodeIdx < shapePlanGraph->sp_node_nr; nodeIdx++)
    {
        bool res = runSif(nodeIdx, programDataHostAddress);
        if (!res)
        {
            return false;
        }
    }
    STAT_GLBL_COLLECT_TIME(sifTime, globalStatPointsEnum::DsdSif);

    m_patchingStatus.patchingState = DsdPatchingState::SIF_EXECUTED;
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function does the dynamic patching by calling the relevant functions
 *
 *   @param  dataChunksHostAddresses: to be patched
 *   @param  firstNodeIndex          first node index to run smf on
 *   @param  lastNodeIndex           run smf until last node index (lastIndex is not included)
 ***************************************************************************************************
 */
bool DynamicRecipe::runSmfOnNodes(const std::vector<uint64_t>& dataChunksHostAddresses,
                                  uint32_t                     firstNodeIndex,
                                  uint32_t                     lastNodeIndex)
{
    LOG_DSD_INFO("shape manipulation");
    uint32_t nodesNr = (m_rRecipeInfo.shape_plan_recipe)->sp_node_nr;

    // Do the patching
    {
        STAT_GLBL_START(smfTime);

        bool resRunInference = runSmf(dataChunksHostAddresses, firstNodeIndex, lastNodeIndex);
        if (!resRunInference)
        {
            STAT_EXIT_NO_COLLECT();
            return resRunInference;
        }
        STAT_GLBL_COLLECT_TIME(smfTime, globalStatPointsEnum::DsdSmf);
    }

#ifdef SFR_STATS
    m_sfr.sifStatCollect(nullptr, 1);  // This is an indication to the statistics that this cycle is done
    m_sfr.smfStatCollect(nullptr, 1);
#endif

    if (lastNodeIndex == nodesNr)
    {
        m_patchingStatus.patchingState = DsdPatchingState::SMF_EXECUTED;
    }
    return true;
}

bool DynamicRecipe::runSmfOnAllNodes(const std::vector<uint64_t>& dataChunksHostAddresses)
{
    if (m_patchingStatus.patchingState != DsdPatchingState::SIF_EXECUTED)
    {
        LOG_DSD_ERR("Cannot patch before SIF had been executed");
        return false;
    }
    uint32_t nodesNr = (m_rRecipeInfo.shape_plan_recipe)->sp_node_nr;
    return runSmfOnNodes(dataChunksHostAddresses, 0, nodesNr);
}

void DynamicRecipe::patchAbort()
{
    m_patchingStatus.patchingState = DsdPatchingState::PRE_SIF;
}

bool DynamicRecipe::takeOwnership()
{
    bool status = true;

    try
    {
        m_ownershipMutex.lock();
    }
    catch (const std::system_error& err)
    {
        LOG_DSD_ERR("Failed to take ownership");
        status = false;
    }

    return status;
}

void DynamicRecipe::releaseOwnership()
{
    m_ownershipMutex.unlock();
}

/*
 ***************************************************************************************************
 *   @brief This function dumps the blobs data to screen (for debug)
 *
 *   @return None
 ***************************************************************************************************
 */
#if 0
void DynamicRecipe::dumpBlobData()
{
    auto recipe = m_rRecipeInfo.recipe;
    for(int blob = 0; blob < recipe->blobs_nr; blob++)
    {
        uint8_t* data = (uint8_t*)m_blobs[blob].data;
        int      s    = m_blobs[blob].size;

        printf("\nblob %X, size %X\n", blob, s);
        for(int i = 0; i < s; i++)
        {
            printf("%02X", data[i]);
            if(i %  8 ==  7) printf(" ");
            if(i % 64 == 63) printf("\n");
        }
    }
    fflush(stdout);
}

/*
 ***************************************************************************************************
 *   @brief This function dumps the patch points to screen (for debug)
 *
 *   @return None
 ***************************************************************************************************
*/
void DynamicRecipe::dumpPP()
{
    auto recipe = m_rRecipeInfo.recipe;

    printf("\n\n----- PP %lX  has %lX PP ------\n", TO64(&m_patchPoints[0]), recipe->patch_points_nr);
    for(uint64_t pp = 0; pp < recipe->patch_points_nr; pp++)
    {
        printf("pp %lX blob %lX type %X addr %lX section %lX offset %lX\n", pp, m_patchPoints[pp].blob_idx,
               m_patchPoints[pp].type, m_patchPoints[pp].memory_patch_point.effective_address,
               m_patchPoints[pp].memory_patch_point.section_idx, m_patchPoints[pp].dw_offset_in_blob);
    }
    fflush(stdout);
}
#endif

/*
 ***************************************************************************************************
 *   @brief This function does the dynamic patching by calling the relevant functions
 *
 *   @param  launchTensorsInfo:   user's tensors
 *   @param  launchTensorsAmount: user's tensors-amount
 *   @param  tensorIdx2userIdx:   two vectors mapping the tensor lists (persistent, shape),
 *                                from the recipe to the user tensors
 ***************************************************************************************************
 */
bool DynamicRecipe::init(const synLaunchTensorInfoExt* launchTensorsInfo,
                         const uint32_t                launchTensorsAmount,
                         const std::vector<uint32_t>*  tensorIdx2userIdx)
{
    LOG_DSD_INFO("DSD Init");

    m_launchTensorsInfo   = launchTensorsInfo;
    m_launchTensorsAmount = launchTensorsAmount;
    m_tensorIdx2userIdx   = tensorIdx2userIdx;

#if 0  // no need to init or copy from original recipe for now. Keeping it here in case needed in the future
    size_t nodeDataCpySize = m_nodeData_private.size() * sizeof(m_nodeData_private[0]);
    memset(m_nodeData_private.data(), 0, nodeDataCpySize);
#endif

    // Init blobs, pp, tensors from original recipe
    {
        STAT_GLBL_START(initRecipeTime);
        bool resInitRecipe = initRecipe();
        if (!resInitRecipe)
        {
            STAT_EXIT_NO_COLLECT();
            return resInitRecipe;
        }
        STAT_GLBL_COLLECT_TIME(initRecipeTime, globalStatPointsEnum::DsdInitRecipe);
    }

    // Set all inputs
    {
        STAT_GLBL_START(initTensorsTime);
        bool resSetInputs = initTensors();
        if (!resSetInputs)
        {
            STAT_EXIT_NO_COLLECT();
            return resSetInputs;
        }
        STAT_GLBL_COLLECT_TIME(initTensorsTime, globalStatPointsEnum::DsdInitTensors);
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function resets the recipe to the clean state (m_blobs, addr pp)
 *
 *   @return true=OK
 ***************************************************************************************************
 */
bool DynamicRecipe::initRecipe()
{
    // copy all the patch points (they are changed during each run)
    if (m_originalPatchPoints != nullptr)
    {
        memcpy(m_patchPoints.data(), m_originalPatchPoints, m_patchPoints.size() * sizeof(m_patchPoints[0]));
    }
    else if (m_originalRecipeAddrPatcher != nullptr)
    {
        m_recipeAddrPatcher.copyPatchPointDb(*m_originalRecipeAddrPatcher);
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function sets all the input tensors so we can later run the SIF
 *
 *   @return true=OK
 ***************************************************************************************************
 */
bool DynamicRecipe::initTensors()
{
    LOG_DSD_INFO("initTensors");

#ifdef EXTRA_CHECKING
    m_tensorSizeInferred = m_rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_inputAndStaticTensors;
#endif

    const auto& recipeInputs = m_rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_recipeInputs;

    for (auto tensorIdx : recipeInputs)
    {
        LOG_DSD_EXTRA("init tensor 0x{:x} {} isStatic 0x{:x}",
                      tensorIdx,
                      getTensorName(tensorIdx),
                      m_isStaticTensors[tensorIdx]);
        tensor_info_t& currTensor = m_sp_tensors_private[tensorIdx];

        if (m_isStaticTensors[tensorIdx])  // no need to set its size
        {
            const TSize* sizes = &currTensor.infer_info.geometry.maxSizes[0];
            LOG_DSD_EXTRA("static tensorIdx 0x{:x} dim {} - 0x{:x}/0x{:x}/0x{:x}/0x{:x}/0x{:x}",
                          tensorIdx,
                          currTensor.infer_info.geometry.dims,
                          sizes[0],
                          sizes[1],
                          sizes[2],
                          sizes[3],
                          sizes[4]);
            continue;
        }

        tensor_info_t::ETensorType type      = currTensor.tensor_type;
        uint32_t                   idxRecipe = currTensor.tensor_db_index;
        uint32_t                   idxLaunch = m_tensorIdx2userIdx[type][idxRecipe];

        const synLaunchTensorInfoExt* launchTensor;
        launchTensor = &m_launchTensorsInfo[idxLaunch];
        castNcopy(currTensor.infer_info.geometry.maxSizes,
                  launchTensor->tensorSize,
                  tpc_lib_api::MAX_TENSOR_DIM);  // Copy from recipe tensor to private copy

#ifdef EXTRA_CHECKING
        HB_ASSERT(idxLaunch != INVALID_TENSOR_INDEX,
                  "no Tensor");  // This can never happen, we should fail in analyzeNewTensor
        m_tensorSizeInferred[tensorIdx] = true;
#endif

        bool res = verifyTensorSize(tensorIdx,
                                    TensorDBType::GRAPH_TENSOR_DB,
                                    false /*compareToLaunchInfo*/,
                                    NODE_IDX_FOR_INIT_TENSOR,
                                    0,
                                    0);
        if (!res)
        {
            return res;
        }

        const TSize* sizes = &currTensor.infer_info.geometry.maxSizes[0];
        LOG_DSD_EXTRA(
            "setting sizes for idxRecipe 0x{:x} name {} idxLaunch 0x{:x} dim {} - 0x{:x}/0x{:x}/0x{:x}/0x{:x}/0x{:x}",
            idxRecipe,
            getTensorName(tensorIdx),
            idxLaunch,
            currTensor.infer_info.geometry.dims,
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4]);
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function does the shape manipulation.
 *          It runs over a range of nodes and executes the SMF function
 *   @return true=OK
 ***************************************************************************************************
 */
bool DynamicRecipe::runSmf(const std::vector<uint64_t>& dataChunksHostAddresses,
                           uint32_t                     firstNodeIndex,
                           uint32_t                     lastNodeIndex)
{
    LOG_DSD_INFO("shape inference");

    StatsCol stats {};

    if (firstNodeIndex == 0)
    {  // init
        m_patchingStatus.current_dc_pp_smf = m_pDataChunkSmPatchPointsInfo->m_dataChunkSmPatchPoints;
    }

    const data_chunk_sm_patch_point_t* pCurrentDcSmPatchPoint = m_patchingStatus.current_dc_pp_smf;

    for (uint64_t nodeIdx = firstNodeIndex; nodeIdx < lastNodeIndex; nodeIdx++)
    {
        patchPPs(nodeIdx,
                 pCurrentDcSmPatchPoint,
                 dataChunksHostAddresses,
                 stats);  // is it a function call or does the compiler optimize it?
    }

    m_patchingStatus.current_dc_pp_smf = (data_chunk_sm_patch_point_t*)pCurrentDcSmPatchPoint;

    STAT_GLBL_COLLECT(stats.totalBypass, DsdBypass);
    STAT_GLBL_COLLECT(stats.totalSkipPP, DsdSkipPP);
    STAT_GLBL_COLLECT(stats.totalNoPatch, DsdNoPatch);

    return true;
}

/*
***************************************************************************************************
*   @brief This function verify that the output dynamic persistent tensors (calculated by SIF) comply with the min/max
*expected sizes
*
*   @return true=OK
***************************************************************************************************
*/
bool DynamicRecipe::verifyOutputs()
{
    LOG_DSD_INFO("verifyOutputs");

    const auto& recipeOutputs = m_rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo.m_recipeOutputs;

    for (auto tensorIdx : recipeOutputs)
    {
        bool res = verifyTensorSize(tensorIdx,
                                    TensorDBType::GRAPH_TENSOR_DB,
                                    true /*compareToLaunchInfo*/,
                                    NODE_IDX_FOR_VERIFY_OUTPUTS,
                                    0,
                                    0);
        if (!res)
        {
            return res;
        }
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function actually set the host tensor address on special node on both input
 *          and output tensors.
 *
 *          There are two types of Host-To-Device Tensors we are handling:
 *              1) User's Host-To-Device Tensors
 *              2) Intermediates's Host-To-Device Tensors
 *
 ***************************************************************************************************
 */
void DynamicRecipe::handleHostToDeviceTensors(uint32_t nodeIdx, uint64_t programDataHostAddress)
{
    shape_plane_graph_t* spg      = m_rRecipeInfo.shape_plan_recipe;
    shape_plane_node_t&  currNode = spg->sp_nodes[nodeIdx];

    uint32_t numOfH2DInputTensors  = 0;
    uint32_t numOfH2DOutputTensors = 0;

    // Input tensors
    for (uint32_t i = 0; i < currNode.input_tensors_nr; i++)
    {
        uint32_t       inputIdx     = currNode.input_tensors[i];
        tensor_info_t& input_tensor = m_sp_tensors_private[inputIdx];

        if (input_tensor.user_tensor_type == HOST_TO_DEVICE_TENSOR)
        {
            numOfH2DInputTensors++;

            uint64_t hostAddr = 0;
            if (input_tensor.tensor_type == tensor_info_t::PERSISTENT_TENSOR)
            {
                uint32_t idxRecipe = input_tensor.tensor_db_index;
                uint32_t idxLaunch = m_tensorIdx2userIdx[input_tensor.tensor_type][idxRecipe];

                hostAddr = m_launchTensorsInfo[idxLaunch].pTensorAddress;
            }
            else
            {
                hostAddr = programDataHostAddress + input_tensor.section_offset;
            }

            input_tensor.infer_info.hostAddress = (unsigned*)hostAddr;

            LOG_DSD_DEBUG("found host tensor node {:x} {} (table_id {:x} func_id {:x})"
                          " host-to-device input tensor {:x} tensorIdx {:x} {} tensor-type {} hostAddr {:x}",
                          nodeIdx,
                          currNode.node_name,
                          currNode.sif_id.sm_tableid,
                          currNode.sif_id.sm_funcid,
                          i,
                          inputIdx,
                          getTensorName(inputIdx),
                          input_tensor.user_tensor_type,
                          TO64(hostAddr));
        }
    }
    // Internal tensors
    for (uint32_t i = 0; i < currNode.node_db_tensors_nr; i++)
    {
        tensor_info_t internal_tensor = currNode.node_db_tensors[i];

        if (internal_tensor.user_tensor_type == HOST_TO_DEVICE_TENSOR)
        {
            uint64_t hostAddr                      = programDataHostAddress + internal_tensor.section_offset;
            internal_tensor.infer_info.hostAddress = (unsigned*)hostAddr;

            LOG_DSD_DEBUG("found host tensor node {:x} {} (table_id {:x} func_id {:x})"
                          " host-to-device internal tensor {:x} tensor-type {} hostAddr {:x}",
                          nodeIdx,
                          currNode.node_name,
                          currNode.sif_id.sm_tableid,
                          currNode.sif_id.sm_funcid,
                          i,
                          internal_tensor.user_tensor_type,
                          TO64(hostAddr));
        }
    }
    //
    // Output tensors
    for (uint32_t i = 0; i < currNode.output_tensors_nr; i++)
    {
        uint32_t       outputIdx     = currNode.output_tensors[i];
        tensor_info_t& output_tensor = m_sp_tensors_private[outputIdx];

        if (output_tensor.user_tensor_type == HOST_TO_DEVICE_TENSOR)
        {
            numOfH2DOutputTensors++;

            // why cannot output be persistent?
            HB_ASSERT(output_tensor.tensor_type == tensor_info_t::INTERNAL_TENSOR,
                      "Found a non-internal H2D output tensor: node {:x} {} tensor {:x} {}",
                      nodeIdx,
                      currNode.node_name,
                      outputIdx,
                      getTensorName(outputIdx));

            uint64_t hostAddr                    = programDataHostAddress + output_tensor.section_offset;
            output_tensor.infer_info.hostAddress = (unsigned*)hostAddr;

            LOG_DSD_DEBUG("found host tensor node {:x} {} (table_id {:x} func_id {:x})"
                          " host-to-device output tensor {:x} tensorIdx {:x} {} tensor-type {} hostAddr {:x}",
                          nodeIdx,
                          currNode.node_name,
                          currNode.sif_id.sm_tableid,
                          currNode.sif_id.sm_funcid,
                          i,
                          outputIdx,
                          getTensorName(outputIdx),
                          output_tensor.user_tensor_type,
                          TO64(hostAddr));
        }
    }

    // Second-hand copy handling
    // Requires:
    //      A single input and a single output
    //      No H2D output tensors, while there might be up to a single H2D input tensor
    if ((currNode.input_tensors_nr == 1) && (currNode.output_tensors_nr == 1) &&
        ((numOfH2DInputTensors <= 1) && (numOfH2DOutputTensors == 0)))
    {
        uint32_t       inputIdx      = currNode.input_tensors[0];
        tensor_info_t& input_tensor  = m_sp_tensors_private[currNode.input_tensors[0]];
        tensor_info_t& output_tensor = m_sp_tensors_private[currNode.output_tensors[0]];

        if (numOfH2DInputTensors == 1)
        {
            output_tensor.infer_info.hostAddress = input_tensor.infer_info.hostAddress;

            uint64_t hostAddr = (uint64_t)output_tensor.infer_info.hostAddress;
            LOG_DSD_DEBUG("found host tensor node {:x} {} (table_id {:x} func_id {:x})"
                          " host-to-device non-H2D output tensor {:x} tensorIdx {:x} {} hostAddr {:x}",
                          nodeIdx,
                          currNode.node_name,
                          currNode.sif_id.sm_tableid,
                          currNode.sif_id.sm_funcid,
                          0,
                          0,
                          getTensorName(0),
                          TO64(hostAddr));
        }
        else if ((numOfH2DInputTensors == 0) && (input_tensor.tensor_flags & tensor_info_t::HAS_HOST_ADDRESS) &&
                 (output_tensor.tensor_flags & tensor_info_t::HAS_HOST_ADDRESS))
        {
            // Input tensor looks like a second-hand copy of a host2device tensor
            // It should have its hostAddress already set
            HB_ASSERT_DEBUG_ONLY(input_tensor.infer_info.hostAddress != nullptr,
                                 "Found a tensor with HAST_HOST_ADDRESS attribute but without a host address:"
                                 " node {:x} {} tensor {:x} {}",
                                 nodeIdx,
                                 currNode.node_name,
                                 inputIdx,
                                 getTensorName(inputIdx));

            output_tensor.infer_info.hostAddress = input_tensor.infer_info.hostAddress;
            LOG_DSD_DEBUG("found a copy of a host tensor node {:x} {} (table_id {:x} func_id {:x})"
                          " host-to-device tensor {:x} tensorIdx {:x} {} hostAddr {:x}",
                          nodeIdx,
                          currNode.node_name,
                          currNode.sif_id.sm_tableid,
                          currNode.sif_id.sm_funcid,
                          0,
                          inputIdx,
                          getTensorName(inputIdx),
                          TO64(input_tensor.infer_info.hostAddress));
        }
    }
}

bool isZeroSized(tensor_shape_infer_info_t* pTensor)
{  // TODO: will be removed in [SW-70010]
    int tensorSize = 1;
    for (int dimId = 0; dimId < pTensor->geometry.dims; dimId++)
    {
        tensorSize *= pTensor->geometry.maxSizes[dimId];
    }
    return tensorSize == 0;
}
/*
 ***************************************************************************************************
 *   @brief This function actually runs the SIF function
 *
 *   It builds the SIF input and output structures, create valid array on stack, run the function
 *
 ***************************************************************************************************
 */
bool DynamicRecipe::runSif(uint64_t nodeIdx, uint64_t programDataHostAddress)
{
    static tpc_lib_api::DeviceId deviceId = deviceTypeToDeviceID(m_rDeviceAgnosticRecipeInfo.m_deviceType);

    shape_plane_graph_t* spg        = m_rRecipeInfo.shape_plan_recipe;
    shape_plane_node_t&  currNode   = spg->sp_nodes[nodeIdx];
    NodeParams&          currParams = m_NodeParams[nodeIdx];
    bool                 fuserNode  = currNode.basic_nodes_nr > 1;

    if (fuserNode)
    {
        // copy the node tensors to a private DB (so we can modify them)
        memcpy(m_fuserNodeDbTensors.data(),
               currNode.node_db_tensors,
               sizeof(m_fuserNodeDbTensors[0]) * currNode.node_db_tensors_nr);
#ifdef EXTRA_CHECKING
        m_fuserNodeTensorSizeInferred.clear();
        m_fuserNodeTensorSizeInferred.resize(currNode.node_db_tensors_nr);
#endif
    }

    // A User's HostToDevice Tensor can only occur as a single input
    // of a DMA node which also has a single output
    // Below, set host tensor address for a HOST_TO_DEVICE_TENSOR
    // and also for its corresponding DATA_TENSOR which is the output
    // of the node
    //
    // An Intermediate's HostToDevice Tensor may be part of a node,
    // with several IH2D inputs & outputs

    handleHostToDeviceTensors(nodeIdx, programDataHostAddress);

    for (uint32_t subNode = 0; subNode < currNode.basic_nodes_nr; subNode++)
    {
        shape_plane_basic_node_t& currSubNode = currNode.basic_nodes[subNode];

        if (currSubNode.sif_id.sm_func_index != INVALID_SHAPE_FUNC_ID)
        {
            LOG_DSD_EXTRA("node 0x{:x} {} subNode 0x{:x} {}, sif sm_func_index 0x{:x} {}",
                          nodeIdx,
                          currNode.node_name,
                          subNode,
                          currSubNode.node_name,
                          currSubNode.sif_id.sm_func_index,
                          m_sfr.getSifName(currSubNode.sif_id));

#ifdef EXTRA_CHECKING
            for (size_t i = 0; i < currSubNode.input_tensors_nr; i++)
            {
                uint32_t     tensorIdx = currSubNode.input_tensors[i];
                TensorDBType type      = currSubNode.input_tensors_db[i];
                // Check if tensor's size is inferred. We consider a tensor inferred:
                // local node tensor: 1) if its bit is set
                // Global tensor    : 1) if its bit is set OR 2) Not persistent tensor and static

                bool                       tensorSizeInferred = false;
                tensor_shape_infer_info_t* pTensor;
                if (type == TensorDBType::GRAPH_TENSOR_DB)
                {
                    pTensor = &m_sp_tensors_private[tensorIdx].infer_info;

                    if (isZeroSized(pTensor) || m_tensorSizeInferred[tensorIdx] ||
                        ((spg->sp_tensors[tensorIdx].tensor_type != tensor_info_t::PERSISTENT_TENSOR) &&
                         (m_isStaticTensors[tensorIdx])))
                    {
                        tensorSizeInferred = true;
                    }
                }
                else
                {
                    auto const& tensorInfo = m_fuserNodeDbTensors[tensorIdx];
                    bool        isStaticTensor =
                        std::memcmp(tensorInfo.max_dims, tensorInfo.min_dims, tensorInfo.infer_info.geometry.dims) == 0;
                    pTensor = &(m_fuserNodeDbTensors[tensorIdx].infer_info);
                    if (m_fuserNodeTensorSizeInferred[tensorIdx])
                    {
                        tensorSizeInferred = true;
                    }
                    else if (!m_fuserNodeTensorSizeInferred[tensorIdx] && isStaticTensor)
                    {
                        tensorSizeInferred = true;
                        std::memcpy(pTensor->geometry.maxSizes, tensorInfo.max_dims, SYN_MAX_TENSOR_DIM);
                    }
                }

                if (tensorSizeInferred)
                {
                    LOG_DSD_EXTRA("input tensor# 0x{:x} type {} tensorIndex 0x{:x} {} rank {}: [{}]",
                                  i,
                                  type,
                                  tensorIdx,
                                  getTensorName(tensorIdx),
                                  pTensor->geometry.dims,
                                  fmt::join(pTensor->geometry.maxSizes, pTensor->geometry.maxSizes + SYN_MAX_TENSOR_DIM, ","));
                }
                else
                {
                    LOG_DSD_ERR("Node 0x{:x} {} subNode 0x{:x} {} in number 0x{:x} type {} tensorIdx 0x{:x} {}: tensor size not inferred.",
                                nodeIdx,
                                currNode.node_name,
                                subNode,
                                currSubNode.node_name,
                                i,
                                type,
                                tensorIdx,
                                getTensorName(tensorIdx));
                    return false;
                }
            }  // for(in Tensor)
#endif         // EXTRA_CHECKING

            SifParams*  pSifParams;
            SifOutputs* pSifOutputs;
            sif_t       sif;

            if (!fuserNode)
            {
                pSifParams  = &currParams.sifParams;
                pSifOutputs = &currParams.sifOutputs;
                sif         = currParams.sifFunc;
                memset(pSifOutputs->invalidMask, 0, currParams.invalidArrSize * sizeof(pSifOutputs->invalidMask[0]));
                // Checking that the SIF is not null is done in the recipe processing
            }
            else
            {
                pSifParams  = &m_fuserSifParams;
                pSifOutputs = &m_fuserSifOutputs;
                std::tie(sif, pSifParams->pGuid) = m_sfr.getSIFandGuidInfo(currSubNode.sif_id);
                // TODO: remove this when [SW-159125] is done
                // pGuid can be nullptr if not a dynamic node, so guid is not needed in this case
                if (pSifParams->pGuid)
                {
                    pSifParams->guid = *pSifParams->pGuid;
                }
                pSifParams->maxAvailableTpc = m_availableTpcEngines;
                memset(pSifOutputs->invalidMask, 0, m_fuserInvalidMask.size());

                getFuserNodeSifParams(nodeIdx, subNode, &m_fuserSifParams, &m_fuserSifOutputs);
            }

#ifdef PERF_LOG_LEVEL0
            STAT_GLBL_START(sifTime);
#endif
#ifdef SFR_STATS
            auto start = TimeTools::timeNow();
#endif

            // set host address if needed
            if (currParams.hasHostTensor || fuserNode)
            {
                for (int t = 0; t < pSifParams->inputTensorsNr; t++)
                {
                    uint64_t       tensorIdx  = currSubNode.input_tensors[t];
                    tensor_info_t& currTensor = m_sp_tensors_private[tensorIdx];

                    if (currTensor.user_tensor_type == HOST_SHAPE_TENSOR)
                    {
                        uint32_t idxRecipe = currTensor.tensor_db_index;
                        uint32_t idxLaunch = m_tensorIdx2userIdx[SHAPE_TENSOR][idxRecipe];

                        unsigned* hostAddr = (unsigned*)m_launchTensorsInfo[idxLaunch].pTensorAddress;
                        m_sp_tensors_private[tensorIdx].infer_info.hostAddress = hostAddr;

                        LOG_DSD_DEBUG("found host tensor node {:x} {} sub-node {:x} {} host shape tensor {:x} "
                                      "tensorIdx {:x} {} hostAddr {:x}",
                                      nodeIdx,
                                      currNode.node_name,
                                      subNode,
                                      currSubNode.node_name,
                                      t,
                                      tensorIdx,
                                      getTensorName(tensorIdx),
                                      TO64(hostAddr));
                    }
                }
            }

            tpc_lib_api::GlueCodeReturn sifRes = sif(deviceId, pSifParams, pSifOutputs);
            if (sifRes != tpc_lib_api::GLUE_SUCCESS)
            {
                LOG_DSD_ERR("Node 0x{:x} {} sub-node {} {}: SIF id 0x{:x} {} returned an error ({})",
                            nodeIdx,
                            currNode.node_name,
                            subNode,
                            currSubNode.node_name,
                            currSubNode.sif_id.sm_func_index,
                            m_sfr.getSifName(currSubNode.sif_id),
                            enumToString(sifRes));
                return false;
            }
#ifdef SFR_STATS
            m_sfr.sifStatCollect(sif, TimeTools::timeFromNs(start));
#endif
#ifdef PERF_LOG_LEVEL0
            STAT_GLBL_COLLECT_TIME(sifTime, globalStatPointsEnum::DsdSifOnly);
#endif
            for (int i = 0; i < currParams.invalidArrSize; i++)
            {
                LOG_DSD_EXTRA("output mask{} 0x{:x}", i, pSifOutputs->invalidMask[i]);
            }
#ifdef EXTRA_CHECKING
            bool res = validateSifOutput(nodeIdx, subNode, pSifOutputs->invalidMask);
            if (!res)
            {
                return res;
            }
#endif
        }  // if(sifIdx != Invalid
        else
        {
            LOG_DSD_EXTRA("NodeIdx {:x} {} subnode {:x} {}: sif function not set",
                          nodeIdx,
                          currNode.node_name,
                          subNode,
                          currSubNode.node_name);
#ifdef EXTRA_CHECKING
            for (uint32_t i = 0; i < currSubNode.output_tensors_nr; i++)
            {
                uint64_t tensorIdx = currSubNode.output_tensors[i];
                if (m_tensorSizeInferred[tensorIdx] || !m_isStaticTensors[tensorIdx])
                {
                    continue;
                }

                m_tensorSizeInferred[tensorIdx] = true;
                LOG_DSD_EXTRA(
                    "NodeIdx {:x} {} subnode {:x} {}: setting static output as inferred, tensor# 0x{:x} tensorIndex {:x} {}",
                    nodeIdx,
                    currNode.node_name,
                    subNode,
                    currSubNode.node_name,
                    i,
                    tensorIdx,
                    getTensorName(tensorIdx));
            }  // for(output_tensors)
#endif
        }  // if(valid Sif) else
    }      // for(subNode)

    // run node_match_output_tensors_nr
    for (int i = 0; i < currNode.node_match_output_tensors_nr; i++)
    {
        uint64_t source = currNode.output_src_tensors[i];
        uint64_t dest   = currNode.output_dst_tensors[i];

        m_sp_tensors_private[dest].infer_info = m_sp_tensors_private[source].infer_info;
        LOG_DSD_EXTRA("match_output 0x{:x}->0x{:x}", source, dest);

#ifdef EXTRA_CHECKING
        if (m_tensorSizeInferred[source] ||
            ((spg->sp_tensors[source].tensor_type != tensor_info_t::PERSISTENT_TENSOR) && (m_isStaticTensors[source])))
        {
            ;
        }
        else
        {
            LOG_DSD_ERR("Node 0x{:x} {} input number 0x{:x} tensorIndex 0x{:x} {}: match output source tensor's sizes "
                        "were not inferred.",
                        nodeIdx,
                        currNode.node_name,
                        i,
                        source,
                        getTensorName(source));
            return false;
        }
        bool res = verifyTensorSize(dest,
                                    TensorDBType::GRAPH_TENSOR_DB,
                                    true /*compareToLaunchInfo*/,
                                    nodeIdx,
                                    0x8000 + i,
                                    0);  // no need to check for internal. Check all outputs at the end
        if (!res)
        {
            return res;
        }
        if ((m_tensorSizeInferred[dest] == true) &&  // Check if inferring output again
            (!m_isStaticTensors[dest]))              // But allowed for static tensors
        {
            LOG_DSD_ERR("Node 0x{:x} {} output number 0x{:x} tensorIndex 0x{:x} {}: match output dest tensor's sizes "
                        "were already inferred.",
                        nodeIdx,
                        currNode.node_name,
                        i,
                        dest,
                        getTensorName(dest));
            return false;
        }
        m_tensorSizeInferred[dest] = true;
#endif
    }

    return true;
}

/*
 ***************************************************************************************************
 *   @function getFuserNodeSifParams
 *   @brief This function calculates the sif params for fuser (multi sub-nodes in a node).
 *
 *   The function builds the params into a pre-allocated memory. For fuser we don't pre-calculated
 *   the params but instead do it on the fly. We can consider doing it in advance in the future.
 *
 ***************************************************************************************************
 */
void DynamicRecipe::getFuserNodeSifParams(uint32_t    nodeIdx,
                                          uint32_t    subNode,
                                          SifParams*  pSifParams,
                                          SifOutputs* pSifOutputs)
{
    shape_plane_graph_t*      shapePlanRecipe = m_rRecipeInfo.shape_plan_recipe;
    shape_plane_node_t&       currNode        = shapePlanRecipe->sp_nodes[nodeIdx];
    shape_plane_basic_node_t& currSubNode     = currNode.basic_nodes[subNode];

    pSifParams->inputTensorsNr = currSubNode.input_tensors_nr;
    for (uint32_t in = 0; in < currSubNode.input_tensors_nr; in++)
    {
        uint64_t     tensorIdx = currSubNode.input_tensors[in];
        TensorDBType fuserType = currSubNode.input_tensors_db[in];
        LOG_DEBUG(SYN_STREAM,
                  "{} Node {} subNode {} in {} tensorIdx {} type {}",
                  HLLOG_FUNC,
                  nodeIdx,
                  subNode,
                  in,
                  tensorIdx,
                  fuserType);

        pSifParams->inputTensors[in] = (fuserType == TensorDBType::GRAPH_TENSOR_DB)
                                           ? &(m_sp_tensors_private[tensorIdx].infer_info)
                                           : &(m_fuserNodeDbTensors[tensorIdx].infer_info);
    }
    pSifParams->nodeParams.nodeParamsSize = currSubNode.sif_params_nr;
    pSifParams->nodeParams.nodeParams     = currSubNode.sif_params;
    pSifParams->outputTensorsNr           = currSubNode.output_tensors_nr;
    pSifParams->inputPermutations         = currSubNode.input_permutations;

    for (uint32_t out = 0; out < currSubNode.output_tensors_nr; out++)
    {
        uint64_t     tensorIdx = currSubNode.output_tensors[out];
        TensorDBType fuserType = currSubNode.output_tensors_db[out];
        LOG_DEBUG(SYN_STREAM,
                  "{} Node {} subNode {} out {} tensorIdx {} type {}",
                  HLLOG_FUNC,
                  nodeIdx,
                  subNode,
                  out,
                  tensorIdx,
                  fuserType);

        pSifOutputs->outputTensors[out] = (fuserType == TensorDBType::GRAPH_TENSOR_DB)
                                              ? &(m_sp_tensors_private[tensorIdx].infer_info)
                                              : &(m_fuserNodeDbTensors[tensorIdx].infer_info);
    }
}

/*
 ***************************************************************************************************
 *   @function patchPPs
 *   @brief This function runs the SMF (patching)
 *
 *   The function runs on all nodes, on every patch point in each node and calls runSmf function to do the
 *   patching.
 *   In case bypass=true, it skips until the roi with the next index
 *
 ***************************************************************************************************
 */
bool DynamicRecipe::patchPPs(int                                 nodeIdx,
                             const data_chunk_sm_patch_point_t*& pCurrentDcSmPatchPoint,
                             const std::vector<uint64_t>&        dataChunksHostAddresses,
                             StatsCol&                           stats)
{
    STAT_FUNCTION();
    shape_plane_graph_t* shapePlanRecipe  = m_rRecipeInfo.shape_plan_recipe;
    shape_plane_node_t&  currNode         = shapePlanRecipe->sp_nodes[nodeIdx];
    auto                 currNodePpParams = m_NodeParams[nodeIdx].PPparamsVec.begin();

#ifdef EXTRA_CHECKING
    for (int i = 0; i < currNode.output_tensors_nr; i++)
    {
        uint32_t tensorIdx = currNode.output_tensors[i];

        if (!m_tensorSizeInferred[tensorIdx] && !m_isStaticTensors[tensorIdx])
        {
            LOG_DSD_ERR("output tensor for SMF is not set. Node {:x} {}. Tensor number {:x} idx {:x} {}",
                        nodeIdx,
                        currNode.node_name,
                        i,
                        tensorIdx,
                        getTensorName(tensorIdx));
            return false;
        }
    }
#endif

    if (currNode.node_patch_points_nr == 0)
    {
        return true;  // nothing to patch
    }
    const data_chunk_sm_patch_point_t*& pCurrentPatchPoint = pCurrentDcSmPatchPoint;
    uint64_t basePatchPointRoiIndex = (pCurrentPatchPoint == nullptr) ? 0 : pCurrentPatchPoint->roi_idx;
    bool     shouldBypass           = false;

    for (int ppIdx = 0; ppIdx < currNode.node_patch_points_nr; ppIdx++, pCurrentPatchPoint++, currNodePpParams++)
    {
        // shouldBypass until the next roi index is changed.
        if (pCurrentPatchPoint->roi_idx != basePatchPointRoiIndex)
        {
            shouldBypass           = false;
            basePatchPointRoiIndex = pCurrentPatchPoint->roi_idx;
        }

        bool shouldSkipCurPP = (shouldBypass && !pCurrentPatchPoint->is_unskippable);
        LOG_DSD_EXTRA("shouldSkipCurPP {}, shouldBypass {}, is_unskippable {} ",
                      shouldSkipCurPP,
                      shouldBypass,
                      pCurrentPatchPoint->is_unskippable);

        if (!shouldSkipCurPP)
        {
            _setDataChunkNodeParams(*currNodePpParams, *pCurrentPatchPoint, dataChunksHostAddresses);

            bool res = runSmf(nodeIdx, ppIdx, currNode, *pCurrentPatchPoint, shouldBypass, stats);
            if (!res)
            {
                return res;  // Error is logged in the function
            }
            if (shouldBypass)
            {
                stats.totalBypass++;
            }
        }
        else
        {
            LOG_DSD_EXTRA("skipping Node's patch-point index {:x} roi_idx {:x}", ppIdx, basePatchPointRoiIndex);
            stats.totalSkipPP++;
        }
    }
    return true;
}

/*
 ***************************************************************************************************
 *   @brief This function runs the SMF (patching)
 *
 *   The function actually does the patching
 *
 *   The function prepare the input and output params for the SMF function, runs the SMF and does the patching
 *   Based on the results.
 *
 ***************************************************************************************************
 */
bool DynamicRecipe::runSmf(int                                nodeIdx,
                           int                                ppIdx,
                           shape_plane_node_t&                currNode,
                           const data_chunk_sm_patch_point_t& currentPatchPoint,
                           bool&                              shouldBypass,
                           StatsCol&                          stats)
{
    // Get params
    ShapeManipulationParams& params = m_NodeParams[nodeIdx].smfParams;

    // Set fields that are per PP
    params.metadata        = currentPatchPoint.p_pp_metdata;
    params.activationRoi   = &currNode.activation_rois[currentPatchPoint.roi_idx];
    params.inPatchValuesNr = currentPatchPoint.patch_size_dw;

    ShapeManipulationOutputs& outputs = m_NodeParams[nodeIdx].PPparamsVec[ppIdx].shapeManOut;
    outputs.outputShouldBypass        = 0;

    smf_t& smf = m_NodeParams[nodeIdx].PPparamsVec[ppIdx].smfFunc;

    LOG_TRACE_DYNAMIC_PATCHING("Node {} Roi {} Executing SMF {} {}:",
                               currNode.node_name,
                               currentPatchPoint.roi_idx,
                               currentPatchPoint.p_smf_id->sm_func_index,
                               m_sfr.getSmfName(*currentPatchPoint.p_smf_id));

#ifdef PERF_LOG_LEVEL0
    STAT_GLBL_START(smfTime);
#endif
#ifdef SFR_STATS
    auto start = TimeTools::timeNow();
#endif
    smf(&params, &outputs);
#ifdef SFR_STATS
    m_sfr.smfStatCollect(smf, TimeTools::timeFromNs(start));
#endif
#ifdef PERF_LOG_LEVEL0
    STAT_GLBL_COLLECT_TIME(smfTime, globalStatPointsEnum::DsdSmfOnly);
#endif

    if (currentPatchPoint.patch_point_type == FIELD_DYNAMIC_ADDRESS)
    {
        auto pp_high = currentPatchPoint.patch_point_idx_high;
        if (pp_high != -1)
        {
            std::memcpy(m_patchPoints.size() > 0 ?
                        &m_patchPoints[pp_high].memory_patch_point.effective_address : m_recipeAddrPatcher.getPatchPointEffectiveAddr(pp_high),
                        outputs.outputPatchValues,
                        outputs.outPatchValuesNr * sizeof(uint32_t));
        }
    }

    if (outputs.outPatchValuesNr == 0)
    {
        stats.totalNoPatch++;
    }

    if (currentPatchPoint.patch_point_type != FIELD_DYNAMIC_ADDRESS)
    {
        LOG_DSD_EXTRA("nodeIdx {} {} ppIdx {} smf_id 0x{:x} {} Patching: dc-index 0x{:x} offset_in_dc 0x{:x} "
                      "patch_size_dw 0x{:x}",
                      nodeIdx,
                      currNode.node_name,
                      ppIdx,
                      currentPatchPoint.p_smf_id->sm_func_index,
                      m_sfr.getSmfName(*currentPatchPoint.p_smf_id),
                      currentPatchPoint.data_chunk_index,
                      currentPatchPoint.offset_in_data_chunk,
                      outputs.outPatchValuesNr);
    }
    else
    {
        LOG_DSD_EXTRA("nodeIdx {} {} ppIdx {} smf_id 0x{:x} {} Patching: patch_point_idx_low 0x{:x} "
                      "patch_point_idx_high 0x{:x} patch_size_dw 0x{:x}",
                      nodeIdx,
                      currNode.node_name,
                      ppIdx,
                      currentPatchPoint.p_smf_id->sm_func_index,
                      m_sfr.getSmfName(*currentPatchPoint.p_smf_id),
                      currentPatchPoint.patch_point_idx_low,
                      currentPatchPoint.patch_point_idx_high,
                      outputs.outPatchValuesNr);
    }

    shouldBypass = (bool)outputs.outputShouldBypass;

    return true;
}

/*
 ***************************************************************************************************
 *   Function: validateSifOutput
 *   @brief This function checks if the output of the SIF function is valid
 *   @output true = OK
 *
 *   The function goes over all the outputs, for every output it calls a function to check the size,
 *   verifies the tensor wasn't set already (optional, #ifdef) and mark the tensor as set.
 ***************************************************************************************************
 */
#ifdef EXTRA_CHECKING
bool DynamicRecipe::validateSifOutput(uint32_t nodeIdx, uint32_t subNode, unsigned* invalidMask)
{
    shape_plane_graph_t*      shapePlanGraph = m_rRecipeInfo.shape_plan_recipe;
    shape_plane_node_t&       currNode       = shapePlanGraph->sp_nodes[nodeIdx];
    shape_plane_basic_node_t& currSubNode    = shapePlanGraph->sp_nodes[nodeIdx].basic_nodes[subNode];

    for (uint64_t i = 0; i < currSubNode.output_tensors_nr; i++)
    {
        uint64_t tensorIdx = currSubNode.output_tensors[i];
        uint64_t offset    = i / BITS_IN_UNSIGNED;
        uint64_t bit       = i % BITS_IN_UNSIGNED;

        if ((invalidMask[offset] & (1 << bit)) == 0)
        {
            TensorDBType fuserType = currSubNode.output_tensors_db[i];

            bool res = verifyTensorSize(tensorIdx,
                                        fuserType,
                                        false /*compareToLaunchInfo*/,
                                        nodeIdx,
                                        i,
                                        subNode);  // no need to check for internal. Check all outputs at the end
            if (!res)
            {
                return res;
            }

            bool alreadySet =
                (fuserType == TensorDBType::GRAPH_TENSOR_DB) ? m_tensorSizeInferred[tensorIdx] : m_fuserNodeTensorSizeInferred[tensorIdx];
            if (alreadySet == true)  // Setting output again
            {
                LOG_DSD_ERR("NodeIdx 0x{:x} {} subNode 0x{:x} {} output idx 0x{:x} tensorIndex 0x{:x} {} type {}: "
                            "output tensor's sizes were already inferred.",
                            nodeIdx,
                            currNode.node_name,
                            subNode,
                            currSubNode.node_name,
                            i,
                            tensorIdx,
                            getTensorName(tensorIdx),
                            fuserType);
                return false;
            }

            if (fuserType == TensorDBType::GRAPH_TENSOR_DB)
            {
                m_tensorSizeInferred[tensorIdx] = true;
            }
            else
            {
                m_fuserNodeTensorSizeInferred[tensorIdx] = true;
            }

            tensor_shape_infer_info_t& tensor = m_sp_tensors_private[tensorIdx].infer_info;
            LOG_DSD_EXTRA("output tensor# 0x{:x} tensorIndex 0x{:x} {} dims {}: 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
                          i,
                          tensorIdx,
                          getTensorName(tensorIdx),
                          tensor.geometry.dims,
                          tensor.geometry.maxSizes[0],
                          tensor.geometry.maxSizes[1],
                          tensor.geometry.maxSizes[2],
                          tensor.geometry.maxSizes[3],
                          tensor.geometry.maxSizes[4]);
        }
    }
    return true;
}
#endif

/*
 ***************************************************************************************************
 *   Function: verifyTensorSize
 *   @brief This function checks if a tensor size (given by SIF output) is valid
 *   @output true = OK
 *
 *   The function checks for all dims if the tensor is between min and max.
 *   If the tensor size is given by the user (in synLaunch) it also checks that the user given size
 *   is exactly what the user gave
 *
 ***************************************************************************************************
 */
bool DynamicRecipe::verifyTensorSize(uint64_t     tensor,
                                     TensorDBType fuserType,
                                     bool         compareToLaunchInfo,
                                     uint64_t     nodeIdx,
                                     uint32_t     tensorIdx,
                                     uint32_t     subNode)
{
    tensor_info_t& currTensor =
        (fuserType == TensorDBType::GRAPH_TENSOR_DB) ? m_sp_tensors_private[tensor] : m_fuserNodeDbTensors[tensor];

    for (int dim = 0; dim < currTensor.infer_info.geometry.dims; dim++)
    {
        bool failed;

        if (currTensor.tensor_type != tensor_info_t::SHAPE_TENSOR)
        {
            failed =
                (currTensor.infer_info.geometry.maxSizes[dim] > currTensor.max_dims[dim]) ||
                (!GCFG_ENABLE_WIDE_BUCKET.value() && (currTensor.infer_info.geometry.maxSizes[dim] < currTensor.min_dims[dim]));
        }
        else
        {
            unsigned max = std::max(currTensor.max_dims[dim], currTensor.min_dims[dim]);
            unsigned min = std::min(currTensor.max_dims[dim], currTensor.min_dims[dim]);

            failed = (currTensor.infer_info.geometry.maxSizes[dim] > max) ||
                     (!GCFG_ENABLE_WIDE_BUCKET.value() && (currTensor.infer_info.geometry.maxSizes[dim] < min));
        }

        if (failed)
        {
            std::string nodeName;
            if (nodeIdx == NODE_IDX_FOR_INIT_TENSOR)
            {
                nodeName = "initTensors";
            }
            else if (nodeIdx == NODE_IDX_FOR_VERIFY_OUTPUTS)
            {
                nodeName = "verifyOutputs";
            }
            else
            {
                nodeName = m_rRecipeInfo.shape_plan_recipe->sp_nodes[nodeIdx].node_name;
            }
            const char* tensorName =
                (fuserType == TensorDBType::GRAPH_TENSOR_DB) ? getTensorName(tensor) : "NodeTensor";
            const char* tensorType;
            switch (currTensor.tensor_type)
            {
                case tensor_info_t::PERSISTENT_TENSOR:
                    tensorType = "PERSISTENT_TENSOR";
                    break;
                case tensor_info_t::SHAPE_TENSOR:
                    tensorType = "SHAPE_TENSOR";
                    break;
                case tensor_info_t::INTERNAL_TENSOR:
                    tensorType = "INTERNAL_TENSOR";
                    break;
                default:
                    tensorType = "UNKNOWN";
                    break;
            }
            LOG_DSD_ERR("Node 0x{:x} {}, subNode 0x{:x} tensor 0x{:x} {} of type {} tensorIdx in node 0x{:x}: illegal "
                        "tensor dim {} size, actual {}, should be in the range {}-{}",
                        nodeIdx,
                        nodeName,
                        subNode,
                        tensor,
                        tensorName,
                        tensorType,
                        tensorIdx,
                        dim,
                        currTensor.infer_info.geometry.maxSizes[dim],
                        currTensor.min_dims[dim],
                        currTensor.max_dims[dim]);
            return false;
        }
    }

    if (compareToLaunchInfo && (currTensor.tensor_type != tensor_info_t::INTERNAL_TENSOR) &&
        currTensor.tensor_db_index != INVALID_TENSOR_INDEX)
    {
        tensor_info_t::ETensorType type      = currTensor.tensor_type;
        uint32_t                   idxRecipe = currTensor.tensor_db_index;
        uint32_t                   idxLaunch = m_tensorIdx2userIdx[type][idxRecipe];

        const synLaunchTensorInfoExt* launchTensor = &m_launchTensorsInfo[idxLaunch];
        for (int dim = 0; dim < currTensor.infer_info.geometry.dims; dim++)
        {
            if (currTensor.infer_info.geometry.maxSizes[dim] != launchTensor->tensorSize[dim])
            {
                if (isZeroSized(&currTensor.infer_info)) break;
                const char* tensorName =
                    (fuserType == TensorDBType::GRAPH_TENSOR_DB) ? getTensorName(tensor) : "NodeTensor";
                LOG_DSD_ERR("Node 0x{:x}, subNode 0x{:x}, tensor 0x{:x} {}: tensor dim {} size doesn't match "
                            "synLaunchTensorInfoExt, actual {}, should be {}",
                            nodeIdx,
                            subNode,
                            tensor,
                            tensorName,
                            dim,
                            currTensor.infer_info.geometry.maxSizes[dim],
                            launchTensor->tensorSize[dim]);
                return false;
            }
        }
    }
    return true;
}

/*
 ***************************************************************************************************
 *   Function: getTensorName
 *   @brief This function is used to get the tensor name for tracing
 *   @output true = OK
 *
 * The function gets the tensor_db_idx, the type and based on both goes to either recipe_t
 * or shape_tensors to get the name
 *
 ***************************************************************************************************
 */
const char* DynamicRecipe::staticGetTensorName(uint64_t tensor, const basicRecipeInfo* pRecipeInfo)
{
    shape_plane_graph_t* spg      = pRecipeInfo->shape_plan_recipe;
    const char*          tempName = "No name";

    if (tensor >= spg->sp_tensors_nr)
    {
        return "bad-tensor-number";
    }

    uint32_t tensorDbIdx = spg->sp_tensors[tensor].tensor_db_index;
    if (spg->sp_tensors[tensor].tensor_type == tensor_info_t::PERSISTENT_TENSOR)
    {
        if (tensorDbIdx >= pRecipeInfo->recipe->persist_tensors_nr)
        {
            tempName = "Bad-persistent-number-Idx";
        }
        else
        {
            tempName = pRecipeInfo->recipe->tensors[tensorDbIdx].name;
        }
    }
    else if (spg->sp_tensors[tensor].tensor_type == tensor_info_t::INTERNAL_TENSOR)
    {
        tempName = "Internal";
    }
    else if (spg->sp_tensors[tensor].tensor_type == tensor_info_t::SHAPE_TENSOR)
    {
        if (tensorDbIdx >= spg->shape_tensors_list_nr)
        {
            tempName = "Bad-shape-number-Idx";
        }
        else
        {
            tempName = spg->shape_tensors[tensorDbIdx].name;
        }
    }
    return tempName ? tempName : "No-name";
}

void DynamicRecipe::_setDataChunkNodeParams(PPparams&                          nodePatchPointParams,
                                            const data_chunk_sm_patch_point_t& currDcSmPatchPoint,
                                            const std::vector<uint64_t>&       dataChunksHostAddresses)
{
    if (currDcSmPatchPoint.patch_point_type != FIELD_DYNAMIC_ADDRESS)
    {
        HB_ASSERT_DEBUG_ONLY(currDcSmPatchPoint.data_chunk_index < dataChunksHostAddresses.size(),
                             "Invalid data-chunk index");
        uint8_t*  dataChunkBaseAddress = (uint8_t*)dataChunksHostAddresses[currDcSmPatchPoint.data_chunk_index];
        uint32_t* patchAddr            = (uint32_t*)&(dataChunkBaseAddress[currDcSmPatchPoint.offset_in_data_chunk]);

        nodePatchPointParams.shapeManOut.outputPatchValues = patchAddr;
    }
}
