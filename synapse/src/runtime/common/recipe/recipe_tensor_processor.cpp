#include "recipe_tensor_processor.hpp"
#include "recipe_handle_impl.hpp"
#include "define_synapse_common.hpp"
#include "recipe.h"
#include "patching/host_address_patcher.hpp"
#include "basic_recipe_info.hpp"
#include "log_manager.h"
#include "defs.h"

synStatus RecipeTensorsProcessor::process(const basicRecipeInfo& rBasicRecipeInfo,
                                          RecipeTensorsInfo&     rRecipeTensorsInfo,
                                          RecipeDsdStaticInfo&   rRecipeDsdStaticInfo)
{
    initTensorInfo(rBasicRecipeInfo, rRecipeTensorsInfo);
    setSectionsInfo(rBasicRecipeInfo, rRecipeTensorsInfo);
    setSectionTypesInfo(rBasicRecipeInfo, rRecipeTensorsInfo);
    setSectionDb(rBasicRecipeInfo, rRecipeTensorsInfo);
    bool rtn = processShapePlanRecipe(rBasicRecipeInfo, rRecipeTensorsInfo, rRecipeDsdStaticInfo);
    if (!rtn)
    {
        return synFail;
    }

    return synSuccess;
}

bool RecipeTensorsProcessor::testOnlyProcessShapePlanRecipe(const basicRecipeInfo& rBasicRecipeInfo,
                                                            RecipeTensorsInfo&     rRecipeTensorsInfo,
                                                            RecipeDsdStaticInfo&   rRecipeDsdStaticInfo)
{
    return processShapePlanRecipe(rBasicRecipeInfo, rRecipeTensorsInfo, rRecipeDsdStaticInfo);
}

void RecipeTensorsProcessor::initTensorInfo(const basicRecipeInfo& rBasicRecipeInfo,
                                            RecipeTensorsInfo&     rRecipeTensorsInfo)
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    rRecipeTensorsInfo.m_recipe               = rBasicRecipeInfo.recipe;
    rRecipeTensorsInfo.m_shapePlanRecipe      = rBasicRecipeInfo.shape_plan_recipe;
    rRecipeTensorsInfo.m_isTensorName2idxInit = true;
}

void RecipeTensorsProcessor::setSectionsInfo(const basicRecipeInfo& rBasicRecipeInfo,
                                             RecipeTensorsInfo&     rRecipeTensorsInfo)
{
    uint64_t maxSectionId    = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR - 1;
    uint64_t initialSections = 0;

    const uint64_t numPersistTensors = rBasicRecipeInfo.recipe->persist_tensors_nr;
    // Program-data blobs, might have more than one blob, but will be continuous on the device
    if (rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] != 0)
    {
        initialSections++;
    }
    // Workspace may have size 0 and then we don't want it to be counted as a dynamic section
    if (rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE] != 0)
    {
        initialSections++;
    }

    // another section for MEMORY_ID_RESERVED_FOR_ASSERT_ASYNC
    initialSections++;

    // Go over the rest of the sections (the persistent ones)
    SmallSet<std::unordered_set<uint64_t>, 64> sectionsIdDB;  // SmallSet's an opt for Eager
    for (uint64_t i = 0; i < numPersistTensors; i++)
    {
        uint64_t sectionID = rBasicRecipeInfo.recipe->tensors[i].section_idx;

        sectionsIdDB.insert(sectionID);
        maxSectionId = std::max(maxSectionId, sectionID);
    }

    uint64_t numSectionsToPatch             = initialSections + sectionsIdDB.size();
    rRecipeTensorsInfo.m_maxSectionId       = maxSectionId;
    rRecipeTensorsInfo.m_numSectionsToPatch = numSectionsToPatch;

    LOG_DEBUG(SYN_RECIPE,
              "Sections: recipe 0x{:x} has {} persist tensors, sectionsToPatch {}, maxSectionId {}",
              TO64(rBasicRecipeInfo.recipe),
              numPersistTensors,
              numSectionsToPatch,
              maxSectionId);
}

void RecipeTensorsProcessor::setSectionTypesInfo(const basicRecipeInfo& rBasicRecipeInfo,
                                                 RecipeTensorsInfo&     rRecipeTensorsInfo)
{
    rRecipeTensorsInfo.m_sectionToSectionType.resize(rRecipeTensorsInfo.m_maxSectionId + 1, -1);

    // scratch pad
    rRecipeTensorsInfo.m_sectionToSectionType[MEMORY_ID_RESERVED_FOR_WORKSPACE] = DEFAULT_SECTION_TYPE_ID;

    // program data
    rRecipeTensorsInfo.m_sectionToSectionType[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] = PD_SECTION_TYPE_ID;

    for (uint64_t i = 0; i < rBasicRecipeInfo.recipe->persist_tensors_nr; i++)
    {
        auto sectionId   = rBasicRecipeInfo.recipe->tensors[i].section_idx;
        auto sectionType = rBasicRecipeInfo.recipe->tensors[i].section_type;

        //        sectionToSectionType[sectionId] = sectionType;
        rRecipeTensorsInfo.m_sectionToSectionType[sectionId] = sectionType;
    }
}

/* Computes section size according to static information known on recipe.
 * using this information during analyze new tensors and save some of the computation overhead */
void RecipeTensorsProcessor::setSectionDb(const basicRecipeInfo& rBasicRecipeInfo,
                                          RecipeTensorsInfo&     rRecipeTensorsInfo)
{
    uint64_t            maxSectionId = rRecipeTensorsInfo.m_maxSectionId;
    SectionSizeInfoVec& sectionInfo  = rRecipeTensorsInfo.m_sectionsInfo;

    sectionInfo.clear();
    sectionInfo.resize(maxSectionId + 1, {0, 0, false});

    const_section_t* constSections            = rBasicRecipeInfo.recipe->const_sections;
    uint32_t         constSectionsAmount      = rBasicRecipeInfo.recipe->const_sections_nr;
    auto&            constZeroSizeSectionsSet = rRecipeTensorsInfo.m_constZeroSizeSections;
    auto&            constZeroSizeTensors     = rRecipeTensorsInfo.m_constZeroSizeTensors;

    for (int i = 0; i < constSectionsAmount; i++, constSections++)
    {
        uint64_t sectionId      = constSections->section_idx;
        uint64_t sectionSize    = constSections->size;
        bool     isConstSection = true;
        uint64_t lastTensorRecipeidx = 0;

        HB_ASSERT(sectionId <= maxSectionId,
                  "const sectionId {} is larger than maxSectionId {}",
                  sectionId,
                  maxSectionId);

        if (sectionSize == 0)
        {
            constZeroSizeSectionsSet.insert(sectionId);
        }

        sectionInfo[sectionId] = {sectionSize, 0, isConstSection};

        LOG_DEBUG(SYN_RECIPE, "{}: const sectionId {} sectionSize {} lastTensorRecipeidx {} isConstSection {}",
                               HLLOG_FUNC, sectionId, sectionSize, lastTensorRecipeidx, isConstSection);
    }

    for (size_t i = 0; i < rBasicRecipeInfo.recipe->persist_tensors_nr; i++)
    {
        persist_tensor_info_t* pTensor             = &rBasicRecipeInfo.recipe->tensors[i];
        uint64_t               sectionId           = pTensor->section_idx;
        uint64_t               tensorSizeAndOffset = pTensor->size + pTensor->offset_in_section;
        bool                   isConstSection      = sectionInfo[sectionId].isConstSection;

        HB_ASSERT(sectionId <= maxSectionId, "sectionId {} is larger than maxSectionId {}", sectionId, maxSectionId);

        // we update section size only for non-cost sections
        if (isConstSection)
        {
            bool isZeroSizeTensor = false;
            // section is 0 size
            if (constZeroSizeSectionsSet.find(sectionId) != constZeroSizeSectionsSet.end())
            {
                constZeroSizeTensors.insert(i);
                isZeroSizeTensor = true;
            }
            LOG_DEBUG_T(
                SYN_RECIPE,
                "{}: recipe 0x{:x} ignore updating const section size, sectionId {} sectionSize {} constSectionSize {} "
                "tensorIdxInRecipe {}, isZeroSizeTensor {}",
                HLLOG_FUNC,
                TO64(rBasicRecipeInfo.recipe),
                sectionId,
                tensorSizeAndOffset,
                sectionInfo[sectionId].sectionSize,
                i,
                isZeroSizeTensor);
        }
        else
        {
            LOG_DEBUG(SYN_RECIPE, "{}: persistent sectionId {} lastTensorRecipeidx {} tensorSizeAndOffset {} DBSectionSize {}",
                                   HLLOG_FUNC, sectionId, i, tensorSizeAndOffset, sectionInfo[sectionId].sectionSize);

            if (sectionInfo[sectionId].sectionSize < tensorSizeAndOffset)
            {
                sectionInfo[sectionId] = {tensorSizeAndOffset, i, isConstSection};
            }
        }
    }

    for (uint64_t i = 0; i < rBasicRecipeInfo.recipe->workspace_nr; i++)
    {
        uint64_t workspaceSize = rBasicRecipeInfo.recipe->workspace_sizes[i];
        LOG_DEBUG(SYN_RECIPE, "{}: persistent sectionId {} workspaceSize {}", HLLOG_FUNC, i, workspaceSize);

        sectionInfo[i].sectionSize         = std::max(sectionInfo[i].sectionSize, workspaceSize);
        sectionInfo[i].lastTensorRecipeidx = -1;

        HB_ASSERT(i <= MEMORY_ID_RESERVED_FOR_PROGRAM,
                  "workspace section id  {} is larger than {} ",
                  i,
                  MEMORY_ID_RESERVED_FOR_PROGRAM);
    }
    LOG_DEBUG(SYN_RECIPE, "Set Sections DB");
}

/*
 ***************************************************************************************************
 *   @brief Ths function takes the recipe and builds more efficient containers to be use later
 *
 *   @param  pRecipeInfo
 *
 *   The function creates
 *   1) vector with a list of tensors that should be set before doing the dynamic shaping
 *   2) vector of bools indicating for each tensor if it is static tensor
 *   3) vecotr of bools indicating for each tensor if it is static and an input tensor
 *
 ***************************************************************************************************
 */
bool RecipeTensorsProcessor::processShapePlanRecipe(const basicRecipeInfo& rBasicRecipeInfo,
                                                    RecipeTensorsInfo&     rRecipeTensorsInfo,
                                                    RecipeDsdStaticInfo&   rRecipeDsdStaticInfo)
{
    shape_plane_graph_t* spr = rBasicRecipeInfo.shape_plan_recipe;
    if (!spr) return true;
    // Find all inputs

    std::vector<uint64_t>& recipeInputs         = rRecipeDsdStaticInfo.m_recipeInputs;
    std::vector<uint64_t>& recipeOutputs        = rRecipeDsdStaticInfo.m_recipeOutputs;
    std::vector<bool>&     isStaticTensors      = rRecipeDsdStaticInfo.m_isStaticTensors;
    std::vector<bool>&     inputAndStaticTensor = rRecipeDsdStaticInfo.m_inputAndStaticTensors;
    auto&                  maxNodeOutputs       = rRecipeDsdStaticInfo.m_maxNodeOutputs;
    auto&                  fuserMaxIn           = rRecipeDsdStaticInfo.m_fuserMaxIn;
    auto&                  fuserMaxOut          = rRecipeDsdStaticInfo.m_fuserMaxOut;
    auto&                  fuserMaxDbTensors    = rRecipeDsdStaticInfo.m_fuserMaxDbTensors;

    recipeInputs.clear();
    recipeOutputs.clear();

    isStaticTensors.clear();
    isStaticTensors.resize(spr->sp_tensors_nr, false);

    inputAndStaticTensor.clear();
    inputAndStaticTensor.resize(spr->sp_tensors_nr, false);

    for (uint64_t tensor = 0; tensor < spr->sp_tensors_nr; tensor++)
    {
        tensor_info_t& curr = spr->sp_tensors[tensor];
        if (checkIsStaticTensor(curr))
        {
            isStaticTensors[tensor] = true;
        }
        switch (curr.tensor_type)
        {
            case tensor_info_t::PERSISTENT_TENSOR:
            {
                uint64_t               recipeIdx  = curr.tensor_db_index;
                persist_tensor_info_t& tensorInfo = rBasicRecipeInfo.recipe->tensors[recipeIdx];

                if (tensorInfo.isInput)
                {
                    inputAndStaticTensor[tensor] = isStaticTensors[tensor];
                    recipeInputs.push_back(tensor);
                }
                else
                {
                    if (curr.user_tensor_type == DATA_TENSOR_DYNAMIC)  // Check only Dynamic persistent tensors
                    {
                        recipeOutputs.push_back(tensor);
                    }
                }
                break;
            }

            case tensor_info_t::SHAPE_TENSOR:
            {
                uint64_t recipeIdx = curr.tensor_db_index;

                if (recipeIdx != INVALID_TENSOR_INDEX)  // check that is not internal
                {
                    recipeInputs.push_back(tensor);
                }
                break;
            }

            case tensor_info_t::INTERNAL_TENSOR:
                break;
        }
    }

    maxNodeOutputs    = 0;
    fuserMaxIn        = 0;
    fuserMaxOut       = 0;
    fuserMaxDbTensors = 0;

    for (uint32_t node = 0; node < spr->sp_node_nr; node++)
    {
        maxNodeOutputs = std::max(
            maxNodeOutputs,
            spr->sp_nodes[node].output_tensors_nr);  // find max outputs, used later to create an invalid mask
        fuserMaxDbTensors = std::max(fuserMaxDbTensors, spr->sp_nodes[node].node_db_tensors_nr);

        shape_plane_node_t& currNode = spr->sp_nodes[node];
        for (uint32_t subNode = 0; subNode < currNode.basic_nodes_nr; subNode++)
        {
            shape_plane_basic_node_t& currSubNode = currNode.basic_nodes[subNode];
            fuserMaxIn                            = std::max(fuserMaxIn, currSubNode.input_tensors_nr);
            fuserMaxOut                           = std::max(fuserMaxOut, currSubNode.output_tensors_nr);
        }
    }
    return true;
}

bool RecipeTensorsProcessor::checkIsStaticTensor(const tensor_info_t& tensor)
{
    unsigned dim = tensor.infer_info.geometry.dims;

    for (unsigned d = 0; d < dim; d++)
    {
        if (tensor.min_dims[d] != tensor.max_dims[d])
        {
            return false;
        }
    }
    return true;
}
