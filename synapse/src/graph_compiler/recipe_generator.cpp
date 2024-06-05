#include "recipe_generator.h"

#include "code_generator.h"
#include "command_queue.h"
#include "data_type_utils.h"
#include "debug_define.hpp"
#include "dma_cost_model.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "hal_conventions.h"
#include "log_manager.h"
#include "queue_command.h"
#include "recipe.h"
#include "tpc_node.h"
#include "types.h"

#include "recipe_allocator.h"
#include "section_handle.hpp"
#include "define_synapse_common.hpp"

#include "smf/smf.h"

#include <atomic>
#include <string>
#include <unordered_set>

// TODO - pass device-type as argument, and use proper util

static const uint64_t WORKSPACE_SIZE_GRANULARITY_IN_BYTES = 128;

const uint32_t RecipeGenerator::DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT;

RecipeGenerator::RecipeGenerator(const HabanaGraph* g)
: m_sortedNodes(g->getExeSortedNodes()),
  m_recipeAllocator(nullptr),
  m_blobContainer(g),
  m_graph(g),
  m_programDataBlobsHolder(g->getCodeGenerator()->getProgramDataBlobs()),
  m_persistTensorsPreCompilation(g->getInitialPersistentTensors()),
  m_workspaceSizeInBytes(g->getCodeGenerator()->getWorkspaceAllocator().GetCurrentlyUsed()),
  m_dataBlobsSizeInBytes(alignSizeUp(g->getCodeGenerator()->getAllocatorForProgramData().GetCurrentlyUsed(),
                                     g->getHALReader()->getCacheLineSizeInBytes())),
  m_numOfH2DITensors(0)
{
    // Collect all persistent tensors
    for (const TensorPtr& t : g->getTensors())
    {
        if (t->isPersistent())
        {
            m_persistTensorsPostCompilation.insert(t);
        }
        if (!t->isPersistent() && t->isHost2DeviceTensor())
        {
            m_numOfH2DITensors++;
        }
    }
}

RecipeGenerator::~RecipeGenerator()
{
}

void RecipeGenerator::generateRecipes(bool isDynamicShapeGraph)
{
    initPrograms();

    createPrograms(true);   // Activate program
    createPrograms(false);  // Execute program

    if (shouldCreateECBs())  // relevant only for platforms with ARC architecture
    {
        m_ecbContainer.generateECBs(m_programContainer, m_blobContainer);
        m_ecbContainer.registerCmeCommands(m_graph->getCodeGenerator()->getCmeCommands());
    }

    createJobs();

    if (isDynamicShapeGraph)
    {
        generateShapePlanRecipe();
    }
}

void RecipeGenerator::initPrograms()
{
    for (auto queuePair : m_graph->getCodeGenerator()->getCommandQueueById())
    {
        ConstCommandQueuePtr queue   = queuePair.second;
        HabanaDeviceType     devType = queue->GetDeviceType();

        // Init rolloverIds per engine
        if (m_graph->hasPreNodesRollover(devType))
        {
            unsigned       programIndex;
            RecipeProgram& program = m_programContainer.getProgram(queue->GetQueueID(), devType, programIndex);

            program.setPreNodesRolloverIds(m_graph->getDevicePreNodesRolloverIds(devType));
        }
    }
}

void RecipeGenerator::createPrograms(bool isSetup)
{
    if (isSetup)
    {
        HB_ASSERT(m_patchPointContainer.getPatchPoints().empty(), "active program should be configured first");
    }
    for (auto queuePair : m_graph->getCodeGenerator()->getCommandQueueById())
    {
        ConstCommandQueuePtr queue = queuePair.second;
        validateQueue(queue, isSetup);

        // verify last command in queue is marked for commit
        if (queue->Size(isSetup))
        {
            auto &lastCommand = queue->getCommands(isSetup).back();
            HB_ASSERT(lastCommand->isBlobCommitter(), "Last command in queue should be marked for commit");
        }

        // Process all commands one-by-one
        for (auto& cmd : queue->getCommands(isSetup))
        {
            processCmd(cmd.get(), queue->GetQueueID(), queue->GetDeviceType(), isSetup);
        }
    }

    if (isSetup)
    {
        // we mark all active patch points for runtime efficiency.
        m_patchPointContainer.markExistingPatchPointsAsActivateProgram();
    }
}

// Gradually populate the blobs, programs, patch-points and the persist tensor info table
void RecipeGenerator::processCmd(QueueCommand* cmd, unsigned qid, HabanaDeviceType devType, bool isSetup)
{
    // TODO (SW-149500): incorporate the SFG init flow to initPrograms()

    // Special handling for SFG Init command. Update the program meta data and return. No need to create blob
    if (m_graph->getCodeGenerator()->hasSfgDeviceInitData() && isSFGInitCommand(cmd))
    {
        unsigned       programIndex;
        RecipeProgram& program = m_programContainer.getProgram(qid, devType, programIndex, isSetup);
        program.setInitSfgValue(getInitSfgValue(cmd));
        return;
    }

    RecipeBlob* blob = nullptr;

    HB_ASSERT_PTR(cmd);

    if (cmd->isDynamic())
    {
        blob = m_blobContainer.getWorkDistBlob();
    }
    else if (!cmd->getBasicFieldsContainerInfo().empty() ||     // does cmd have at least one patch point?
             isSetup)                                           // we use only a single blob for the activate part
    {
        blob = m_blobContainer.getPatchingBlob();
    }
    else
    {
        blob = m_blobContainer.getExecutionBlob();
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(QMAN))
    {
        blob->registerQueCmdForDbg(cmd);
    }

    if (cmd->getBasicFieldsContainerInfo().hasDynamicPatchPoints())
    {
        ShapeNode* nodeToUpdate = blob->setContainsDynamicPatchPoint(cmd);
        if (nodeToUpdate != nullptr)
        {
            m_shapeNodesToUpdate.insert(nodeToUpdate);
        }
    }

    uint8_t* writePtr = blob->reserveBytes(cmd->GetBinarySize());

    cmd->writeInstruction(writePtr);

    setBlobFlags(blob, cmd);

    // Generate patch points
    if (!cmd->getBasicFieldsContainerInfo().empty())
    {
        // At this point the AFCI contains field index offsets relative to the start of the command payload.
        // We need to update all the offsets in the AFCI to account for the binary header(s) in the command.
        cmd->prepareFieldInfos();

        uint64_t cmdOffsetWithinBlobInBytes = writePtr - blob->getBasePtr();
        std::list<uint64_t> pps;

        // Convert AFCI to recipe patch-points while adding the cmd offset within the blob (in bytes)
        // to all field index offsets; get back indices to the just-created patch-points.
        pps = m_patchPointContainer.insertPatchPoints(cmd->getBasicFieldsContainerInfo(),
                                                      cmdOffsetWithinBlobInBytes,
                                                      blob->getBlobId());

        m_patchPointsToUpdate.splice(m_patchPointsToUpdate.end(), pps); // save for later update
    }

    // Commit staged blobs at the end of descriptor activation or the end of queue
    if (cmd->isBlobCommitter())
    {
        // See comment in commitBlobs about the guaranteed order of the returned blob list
        std::list<BlobCommitInfo>  commitedBlobs = m_blobContainer.commitBlobs();
        unsigned                   programIndex;
        RecipeProgram&             program = m_programContainer.getProgram(qid, devType, programIndex, isSetup);

        // Staged Submission Support:
        // For each program, in each node - update total blob count. Note that blobs are sequential starting from 0
        // Undefined node exe index (-1) is marked hereafter by 0; hence add 1 to all the valid indices.
        // See next comment explaining the special handling of undefined node exe index.
        uint32_t nodeExeIndex = cmd->getNodeExecutionIndex() < 0 ? 0 : cmd->getNodeExecutionIndex() + 1;
        uint64_t blobCount = 0;

        for (const BlobCommitInfo& info : commitedBlobs)
        {
            program.insertBlobIndex(info.index, info.md);
            blobCount = program.blobIndicies().size();

            // special handling for node execution index 0:
            // cmd with node execution index=0 may appear at the beginning of the queue, at the end, and sometimes
            // in the middle. Those in the beginning should be executed before any node execution. These are the only
            // commands that need to be executed before any other command associated with "real" node can run. All other
            // commands with node exe index=0 will "inherit" the last known node execution index for a given program
            if (nodeExeIndex == 0)
            {
                if (m_programIdxToLastNodeExeIndex.find(programIndex) != m_programIdxToLastNodeExeIndex.end())
                {
                    // update node execution index to the last known index
                    nodeExeIndex = m_programIdxToLastNodeExeIndex[programIndex];
                }
            }
            m_nodeExeToProgramBlobsCount[nodeExeIndex][programIndex] = blobCount;
            m_programIdxToLastNodeExeIndex[programIndex] = nodeExeIndex;

            if (info.isPatching)
            {
                finalizePatchPoints(info.isReused, info.index, info.blobId);
            }
            else if (info.isWDWithPatching)
            {
                HB_ASSERT(!info.isReused, "a Work Distribution patchable blob cannot be reused");
                finalizePatchPoints(info.isReused, info.index, info.blobId);
            }

            if (info.isReused)
            {
                finalizeCompressibleDynamicPatchPoints(info.blobId);
            }
        }

        m_shapeNodesToUpdate.clear();
    }
}

void RecipeGenerator::setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const
{
    // Note that this function may be overriden by subclass and the code here is common for all platforms
    if (cmd->isMonitorArm()) blob->setContainsMonArm();
    if (cmd->isExe()) blob->setContainsExe();
}

void RecipeGenerator::finalizeCompressibleDynamicPatchPoints(unsigned blobId)
{
    for (auto shapeNode : m_shapeNodesToUpdate)
    {
        shapeNode->eraseDynamicPatchPointsByPredicate([=](DynamicPatchPointPtr pp_ptr) {
            return pp_ptr->getFieldType() == FIELD_DYNAMIC_TPC_SIZE && pp_ptr->getBlobId() == blobId;
        });
    }
}

void RecipeGenerator::finalizePatchPoints(bool patchingBlobReused, uint64_t patchingBlobIndex, unsigned blobId)
{
    if (patchingBlobReused)
    {
        // We have a reused patching blob and there is no need to re-patch it, so get rid of redundant patch-point

        if (m_patchPointsToUpdate.size())
        {
            uint64_t newPatchPointIndex = m_patchPointsToUpdate.front();
            uint64_t newPatchPointNodeExe = m_patchPointContainer[newPatchPointIndex].getNodeExecIndex();

            HB_ASSERT(m_blobIndexListOfPatchPoints.find(patchingBlobIndex) != m_blobIndexListOfPatchPoints.end(), "Failed to find patching blob index");

            uint64_t firstIndex = m_blobIndexListOfPatchPoints[patchingBlobIndex][0];
            uint64_t currentPatchPointNodeExe = m_patchPointContainer[firstIndex].getNodeExecIndex();
            // Compare the new/old patch point node execution indices. Update the patch points to the lowest node execution index
            // This is needed for staged submission. Otherwise we may remove patch points that appear in nodes scheduled to
            // be executed early.
            if (newPatchPointNodeExe < currentPatchPointNodeExe)
            {
                LOG_DEBUG(RECIPE_GEN, "Blob index {} patch points have lower node execution index ({}) than current: ({}) - updating {} patch points",
                          patchingBlobIndex, newPatchPointNodeExe, currentPatchPointNodeExe, m_blobIndexListOfPatchPoints[patchingBlobIndex].size());

                // for each patch point - update the node execution to the new lower value
                for (auto itr = m_blobIndexListOfPatchPoints[patchingBlobIndex].begin();
                     itr != m_blobIndexListOfPatchPoints[patchingBlobIndex].end();
                     itr++)
                {
                    m_patchPointContainer[*itr].setNodeExecIndex(newPatchPointNodeExe);
                }
            }

            // get rid of the new patch points list
            LOG_TRACE(RECIPE_GEN,"Blob index: {} - removing {} patch points", patchingBlobIndex, m_patchPointsToUpdate.size());

            for (std::list<uint64_t>::reverse_iterator itr = m_patchPointsToUpdate.rbegin();
                 itr != m_patchPointsToUpdate.rend();
                 itr++) // iterate backwards to ease on the erase
            {
                m_patchPointContainer.erasePatchPointByIndex(*itr);
            }
        }
    }
    else // brand new patching blob
    {
        // Now that we know the blob index we can set it in the patch-points
        for (uint64_t ppIdx : m_patchPointsToUpdate)
        {
            m_patchPointContainer[ppIdx].setBlobIndex(patchingBlobIndex);
            m_blobIndexListOfPatchPoints[patchingBlobIndex].push_back(ppIdx);
        }

        // update the dynamic patch points separately
        m_patchPointContainer.setPatchPointsBlobIndex(patchingBlobIndex, blobId);
    }
    m_patchPointsToUpdate.clear();
}

void RecipeGenerator::createJobs()
{
    // Compressing twice will result in invalid state. Currently, each RecipeProgram is both a program and a job.
    HB_ASSERT(m_activateJobs.empty(), "Already created activate jobs!");
    HB_ASSERT(m_executeJobs.empty(), "Already created execute jobs!");

    std::vector<job_t> allJobs;

    for (unsigned i = 0; i < m_programContainer.getNumPrograms(); i++)
    {
        allJobs.emplace_back(job_t {m_programContainer.getProgramByIndex(i).getEngineId(), i});
    }

    for (auto job : allJobs)
    {
        if (m_programContainer.getProgramByIndex(job.program_idx).isSetup())
        {
            m_activateJobs.push_back(job);
        }
        else
        {
            m_executeJobs.push_back(job);
        }
    }
}

void RecipeGenerator::generateShapePlanRecipe()
{
    // Generate Tensor map
    for (const pNode& node : m_sortedNodes)
    {
        for (const pTensor& tensor : node->getOperands())
        {
            if (tensor == nullptr) continue;
            m_shapePlaneContainer.addTensor(tensor);

            // Internally created shape tensors (tensors with a producer, and those marked
            // synTensorInternalNoProducer) aren't included in the shape tensor list
            if (tensor->isShapeTensor() &&
                !tensor->isPropSet(synTensorProperty::synTensorInternalNoProducer) &&
                m_graph->getTensorProducer(tensor) == nullptr)
            {
                m_shapeTensors.insert(tensor);
            }
        }
    }

    // add all persistent tensors, if not exist in list already
    for (const pTensor& tensor : m_persistTensorsPreCompilation)
    {
        if (tensor == nullptr) continue;
        m_shapePlaneContainer.addTensor(tensor);
    }
}

recipe_t* RecipeGenerator::serializeDataPlaneGraph(RecipeAllocator* pRecipeAlloc) const
{
    uint64_t blobsSizeInBytes = 0;

    m_recipeAllocator = pRecipeAlloc;
    recipe_t* recipe  = (recipe_t*)m_recipeAllocator->allocate(sizeof(recipe_t));

    memset(recipe, 0, sizeof(recipe_t));

    recipe->version_major = RECIPE_VERSION_MAJOR;
    recipe->version_minor = getVersionMinor();

    m_blobContainer.serialize(&recipe->blobs_nr,
                              &recipe->blobs,
                              &blobsSizeInBytes,
                              &recipe->execution_blobs_buffer,
                              &recipe->execution_blobs_buffer_size,
                              &recipe->patching_blobs_buffer,
                              &recipe->patching_blobs_buffer_size,
                              &recipe->dynamic_blobs_buffer,
                              &recipe->dynamic_blobs_buffer_size,
                              pRecipeAlloc);

    m_programContainer.serialize(&recipe->programs_nr, &recipe->programs, m_recipeAllocator);
    m_patchPointContainer.serialize(&recipe->patch_points_nr,
                                    &recipe->activate_patch_points_nr,
                                    &recipe->patch_points,
                                    &recipe->section_groups_nr,
                                    &recipe->section_groups_patch_points,
                                    m_graph->getSectionIdToSectionTypeMap(),
                                    &recipe->section_ids_nr,
                                    &recipe->section_blobs_indices,
                                    &recipe->sobj_section_group_patch_points,
                                    m_recipeAllocator,
                                    m_persistTensorsPreCompilation);
    serializeNodeExecutionList(&recipe->node_nr, &recipe->node_exe_list, recipe->programs_nr);
    serializeJobs(&recipe->activate_jobs_nr, &recipe->activate_jobs, m_activateJobs);
    serializeJobs(&recipe->execute_jobs_nr, &recipe->execute_jobs, m_executeJobs);

    recipe->permute_tensors_views_nr = 0;
    recipe->permute_tensors_views    = nullptr;

    serializePersistTensorInfo(&recipe->persist_tensors_nr,
                               &recipe->tensors,
                               &recipe->sections_nr,
                               m_graph->getSectionIdToSectionTypeMap());
    recipe->h2di_tensors_nr = m_numOfH2DITensors;

    serializeConstSections(&recipe->const_sections_nr, &recipe->const_sections);

    serializeProgramDataBlobs(&recipe->program_data_blobs_nr,
                              &recipe->program_data_blobs,
                              &recipe->program_data_blobs_buffer,
                              &recipe->program_data_blobs_size);

    serializeWorkspaceSizes(&recipe->workspace_nr, &recipe->workspace_sizes, blobsSizeInBytes);
    serializeProfileDebugInfo(&recipe->debug_profiler_info);
    serializeSyncSchemeDebugInfo(&recipe->debug_sync_scheme_info);
    m_ecbContainer.serialize(&recipe->arc_jobs_nr, &recipe->arc_jobs, m_recipeAllocator);
    serializeConfParams(&recipe->recipe_conf_nr, &recipe->recipe_conf_params);
    serializeNOPKernel(&recipe->nop_kernel_offset, &recipe->nop_kernel_section, &recipe->valid_nop_kernel);

    if (!GCFG_DISABLE_ALL_CME_COMMANDS.value())
    {
        recipe->max_used_mcid_degrade = m_graph->getCodeGenerator()->getMcidConverter().getMaxUsedPhysicalDegrade();
        recipe->max_used_mcid_discard = m_graph->getCodeGenerator()->getMcidConverter().getMaxUsedPhysicalDiscard();
    }

    if (g_inspectRecipePackets)
    {
        inspectRecipePackets(recipe->execution_blobs_buffer, recipe->execution_blobs_buffer_size, "execution");
        inspectRecipePackets(recipe->patching_blobs_buffer, recipe->patching_blobs_buffer_size, "patching");
    }

    std::string recipeNameStr = m_graph->getRecipeName();
    if (!recipeNameStr.empty())
    {
        recipe->nameSize = recipeNameStr.size() + 1;
        recipe->name     = pRecipeAlloc->allocate(recipe->nameSize);
        std::strcpy(recipe->name, recipeNameStr.c_str());
    }

    return recipe;
}

shape_plane_graph_t* RecipeGenerator::serializeShapePlane(RecipeAllocator* pRecipeAlloc) const
{
    m_recipeAllocator           = pRecipeAlloc;
    shape_plane_graph_t* recipe = (shape_plane_graph_t*)m_recipeAllocator->allocate(sizeof(shape_plane_graph_t));

    memset(recipe, 0, sizeof(shape_plane_graph_t));

    recipe->version_major = RECIPE_VERSION_MAJOR;
    recipe->version_minor = RECIPE_VERSION_MINOR;

    serializeShapeTensors(&recipe->shape_tensors, &recipe->shape_tensors_list_nr);
    serializeTensorsShapePlane(&recipe->sp_tensors, &recipe->sp_tensors_nr);
    serializeNodesShapePlane(&recipe->sp_nodes, &recipe->sp_node_nr);

    if (LOG_LEVEL_AT_LEAST_TRACE(GC))
    {
        printShapePlane(recipe);
    }
    return recipe;
}

void RecipeGenerator::serializeNodeExecutionList(uint32_t* pNodeNum, node_program_t** pNodeExecList, uint64_t programNum) const
{
    if (!GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        *pNodeNum = 0;
        *pNodeExecList = nullptr;
        return;
    }

    const size_t numNodes = m_graph->getNodes().size();
    *pNodeNum             = numNodes;
    *pNodeExecList = (node_program_t*)m_recipeAllocator->allocate(*pNodeNum * sizeof(node_program_t));

    auto nodeToPatchPointsCount = m_patchPointContainer.getAccumulatedPatchPointsPerNode();
    if (!m_patchPointContainer.getPatchPoints().empty())
    {
        HB_ASSERT(!nodeToPatchPointsCount.empty(), "failed to retrieve node to patch points count map");
    }

    // Initialize the main map which will hold the last "known" total blob count per program.
    // We will use this map to serialize the data after we done processing each node. Since each node
    // never uses all programs, and since we need to serialize the blob count for each program
    // in each node - we will use this map for bookkeeping
    std::unordered_map<uint64_t, uint64_t> programBlobCount;
    for (unsigned i = 0; i < programNum; i++)
    {
        programBlobCount[i] = 0;
    }

    uint64_t        blobCount;
    uint64_t        programIndex;
    node_program_t* pNode = *pNodeExecList;
    uint32_t        patchPointsCount = 0;

    // Note that exectuion "0" represents program/jobs which were not originated by any node and yet the job
    // must be submitted before any node execution. And the remaining numNodes values are the actual nodes.
    // Note that we go over all indices looking them up, instead of iterating over m_nodeExeToProgramBlobsCount,
    // since the later excludes logical op nodes.
    for (uint64_t nodeExeIndex = 0; nodeExeIndex <= numNodes; ++nodeExeIndex)
    {
        auto itr = m_nodeExeToProgramBlobsCount.find(nodeExeIndex);

        // If it's a "real" node - update the node/program/blobCount map
        if (itr != m_nodeExeToProgramBlobsCount.end())
        {
            // iterate all over programs and update map with current blobCount
            for (auto iter = itr->second.begin(); iter != itr->second.end(); iter++)
            {
                programIndex = iter->first;
                blobCount = iter->second;

                // For each program, validate that blob count for current node is greater or equal to previous node. This
                // should identify cases in which commands with node execution index are "out of order". For example, node
                // exe index 0 can be are placed in the middle/end of commands queue resulting in setting the blob count for
                // node exe 0 to a much bigger(and incorrect) value
                HB_ASSERT(blobCount >= programBlobCount[programIndex], "Blob count in current node is smaller than previous one");

                if (blobCount > programBlobCount[programIndex])
                {
                    // need to update the blobCount
                    programBlobCount[programIndex] = blobCount;
                }
            }

            LOG_DEBUG(RECIPE_GEN, "Node execution index: {}, uses: {} programs", itr->first, itr->second.size());
        }

        // since not all nodes have patch points we update the total patch points count up until this node, so nodes
        // with no patch points will use the last known patch points count
        if (nodeToPatchPointsCount.find(nodeExeIndex) != nodeToPatchPointsCount.end())
        {
            patchPointsCount = (uint32_t)nodeToPatchPointsCount[nodeExeIndex];
        }

        // do not serialize node 0. use its data to update the program/blobs/patch point count
        if (nodeExeIndex == 0) continue;

        // now do the serialization
        pNode->patch_points_nr = patchPointsCount;
        pNode->program_blobs_nr = (uint32_t*)m_recipeAllocator->allocate(programNum * sizeof(uint32_t));

        LOG_DEBUG(RECIPE_GEN, "Node execution index: {} accumulated num of patch points: {}",
                nodeExeIndex, pNode->patch_points_nr);

        for (auto it = programBlobCount.begin();
             it != programBlobCount.end();
             it++)
        {
            programIndex = it->first;
            blobCount = it->second;
            pNode->program_blobs_nr[programIndex] = blobCount;
            LOG_TRACE(RECIPE_GEN, "Node execution index: {}, program index: {} blob count: {}, patch points count: {}",
                      nodeExeIndex, programIndex, blobCount, patchPointsCount);
        }
        pNode++;
    }
}

void RecipeGenerator::serializeShapeTensors(shape_tensor_info_t** shapeTensors, uint64_t* serializedTensorAmount) const
{
    *serializedTensorAmount = m_shapeTensors.size();
    bool hasShapeTensors = *serializedTensorAmount > 0;
    *shapeTensors =
        (shape_tensor_info_t*)m_recipeAllocator->allocate(*serializedTensorAmount * sizeof(shape_tensor_info_t));
    if (hasShapeTensors)
    {
        memset(*shapeTensors, 0, sizeof(shape_tensor_info_t) * (*serializedTensorAmount));
    }

    uint32_t i = 0;
    for (pTensor tensor : m_shapeTensors)
    {
        shape_tensor_info_t& currTensor = (*shapeTensors)[i++];
        auto len = tensor->getName().size() + 1;
        char*                allocated  = m_recipeAllocator->allocate(len);
        currTensor.name = allocated;
        strcpy(allocated, tensor->getName().c_str());
    }
}

void RecipeGenerator::serializeTensorPermutation(uint8_t*                              tensorPermutation,
                                                 const std::optional<gc::Permutation>& permutation,
                                                 unsigned                              maxDims) const
{
    // init the permutation field
    for (int i = 0; i < maxDims; i++)
    {
        tensorPermutation[i] = i;
    }
    if (permutation.has_value())
    {
        const auto perm = permutation.value().getValues();

        // override with real permutation
        for (int i = 0; i < perm.size(); i++)
        {
            tensorPermutation[i] = perm[i];
        }
    }
}

void RecipeGenerator::serializeTensorsShapePlane(tensor_info_t** serializedTensors, uint64_t* serializedTensorAmount) const
{
    *serializedTensorAmount = m_shapePlaneContainer.getAmountOfTensors();
    *serializedTensors = (tensor_info_t*)m_recipeAllocator->allocate(*serializedTensorAmount * sizeof(tensor_info_t));
    memset(*serializedTensors, 0, sizeof(tensor_info_t) * (*serializedTensorAmount));
    TStride strides[HABANA_DIM_MAX + 1];
    TSize   sizes[HABANA_DIM_MAX];

    for (uint32_t i = 0; i < *serializedTensorAmount; i++)
    {
        const pTensor& currTensor = m_shapePlaneContainer.getTensorByIndex(i);

        tensor_info_t& currSerializedTensor = (*serializedTensors)[i];
        currSerializedTensor.infer_info.geometry.dims = currTensor->getDim();

        // Work around lack of support for NDims in DSD
        // Revert to the lines above when support is in place
        // We can ignore extra dims here because we assert that no
        // NDims tensor participates in shape inference
        // currTensor->getAllSizesInElements(currSerializedTensor.max_dims);
        // currTensor->getAllMinimalSizesInElements(currSerializedTensor.min_dims);
        //
        unsigned n_dims = currTensor->getDim();

        currTensor->getAllSizesInElements(sizes, n_dims);
        memcpy(currSerializedTensor.max_dims, sizes, sizeof(currSerializedTensor.max_dims));
        currTensor->getAllMinimalSizesInElements(sizes, n_dims);
        memcpy(currSerializedTensor.min_dims, sizes, sizeof(currSerializedTensor.min_dims));

        currTensor->getNStridesInBytes(strides);
        // Legacy -- do not include the stride on FCD
        memcpy(currSerializedTensor.strides, strides + 1, sizeof(currSerializedTensor.strides));

        // fill permutation info
        serializeTensorPermutation(currSerializedTensor.permutation, currTensor->getPermutation(), MAX_DIMENSIONS_NUM);

        currSerializedTensor.data_type = currTensor->getElementType();

        // The recipe is created with the max sizes as the actual sizes, and is re calculated in runtime where necessary.
        currTensor->getAllSizesInElements(sizes, n_dims);
        castNcopy(currSerializedTensor.infer_info.geometry.maxSizes, currSerializedTensor.max_dims, MAX_DIMENSIONS_NUM);

        currSerializedTensor.user_tensor_type = currTensor->getTensorType();
        currSerializedTensor.tensor_flags     = tensor_info_t::NO_FLAGS;
        if (currTensor->hasHostData())
        {
            currSerializedTensor.tensor_flags =
                (tensor_info_t::ETensorFlags)(currSerializedTensor.tensor_flags | tensor_info_t::HAS_HOST_ADDRESS);
        }

        switch (currSerializedTensor.user_tensor_type)
        {
            case (OUTPUT_DESCRIBING_SHAPE_TENSOR):
            case (INPUT_DESCRIBING_SHAPE_TENSOR):
            case (HOST_SHAPE_TENSOR):
            {
                currSerializedTensor.tensor_type = tensor_info_t::SHAPE_TENSOR;
                currSerializedTensor.tensor_db_index = INVALID_TENSOR_INDEX;
                // Internally created shape tensors (tensors with a producer, and those marked
                // synTensorInternalNoProducer) aren't included in the shape tensor list
                if (!currTensor->isPropSet(synTensorProperty::synTensorInternalNoProducer) &&
                     m_graph->getTensorProducer(currTensor) == nullptr)
                {
                    currSerializedTensor.tensor_db_index = getTensorIndex(currTensor, m_shapeTensors);
                }

                break;
            }
            case (DATA_TENSOR):
            case (DATA_TENSOR_DYNAMIC):
            case (DEVICE_SHAPE_TENSOR):
            case (HOST_TO_DEVICE_TENSOR):
            {
                if (currTensor->isPersistent() && !currTensor->isAssignedToConstSection())
                {
                    currSerializedTensor.tensor_type = tensor_info_t::PERSISTENT_TENSOR;
                    currSerializedTensor.tensor_db_index = getTensorIndex(currTensor, m_persistTensorsPreCompilation);
                }
                else if (currTensor->isPersistent() &&
                         currTensor->isAssignedToConstSection() &&
                         getTensorIndex(currTensor, m_persistTensorsPreCompilation) < m_persistTensorsPreCompilation.size())
                {
                    currSerializedTensor.tensor_type = tensor_info_t::PERSISTENT_TENSOR;
                    currSerializedTensor.tensor_db_index = getTensorIndex(currTensor, m_persistTensorsPreCompilation);
                }
                else
                {
                    currSerializedTensor.tensor_type = tensor_info_t::INTERNAL_TENSOR;
                    currSerializedTensor.tensor_db_index = INVALID_TENSOR_INDEX;
                }
                break;
            }
            default:
            {
                LOG_ERR(RECIPE_GEN, "Serialize shape plane invalid tensor type {} : {}",
                        currTensor->getName(), currTensor->getTensorType());
            }
        }

        currSerializedTensor.tensor_info_name = nullptr;
        if (GCFG_ENABLE_PROFILER.value() == true)
        {
            std::string tensorName                = currTensor->getName();
            currSerializedTensor.tensor_info_name = m_recipeAllocator->allocate(tensorName.size() + 1);
            memcpy(const_cast<char*>(currSerializedTensor.tensor_info_name), tensorName.c_str(), tensorName.size() + 1);
            LOG_TRACE(RECIPE_GEN, "Profiler Enabled: serialized tensor name {}", currSerializedTensor.tensor_info_name);
        }

        currSerializedTensor.section_offset = maskOutMemoryID(currTensor->getTensorOffset());
    }
}

uint32_t RecipeGenerator::getTensorIndex(const pTensor tensor, const TensorSet& tensorSet) const
{
    auto res = tensorSet.find(tensor);
    return std::distance(tensorSet.begin(), res);
}

void RecipeGenerator::serializeNodesShapePlane(shape_plane_node_t** shapeNodes, uint32_t* shapeNodesAmount) const
{
    *shapeNodesAmount = m_sortedNodes.size();
    *shapeNodes = (shape_plane_node_t*)m_recipeAllocator->allocate(*shapeNodesAmount * sizeof(shape_plane_node_t));
    memset(*shapeNodes, 0, sizeof(shape_plane_node_t) * (*shapeNodesAmount));

    int i = 0;
    for (const pNode& node : m_sortedNodes)
    {
        node->getShapeNode()->serialize(m_shapePlaneContainer, (*shapeNodes)[i++], m_recipeAllocator);
    }
}

void RecipeGenerator::serializeProfileDebugInfo(debug_info_t* debugInfo) const
{
    debugInfo->version_major = RECIPE_DEBUG_INFO_VERSION_MAJOR;
    debugInfo->version_minor = RECIPE_DEBUG_INFO_VERSION_MINOR;
    debugInfo->recipe_id = m_graph->getRecipeDebugId();
    debugInfo->num_nodes = 0;
    debugInfo->nodes = nullptr;
    debugInfo->printf_addr_nr = 0;
    debugInfo->printf_addr = nullptr;

    bool profilerEnabled = GCFG_ENABLE_PROFILER.value();
    bool usingPrintf     = m_graph->getCodeGenerator()->usingPrintf();

    // if profiling is disabled and no TPC printf data - simply return
    if (!profilerEnabled && !usingPrintf) return;

    std::vector<node_symbol_info_t> debugNodes;
    std::vector<uint64_t>           printf_addr;

    for (const NodePtr& node : m_sortedNodes)
    {
        if (node->isLogicalOperation()) continue;
        const bool isTpc = HabanaGraph::runsOnTPC(node);

        // handle printf tensors if data exist - regardless of the ENABLE_PROFILER flag
        if (isTpc)
        {
            const auto& tpcNode = static_cast<TPCNode&>(*node);
            if (tpcNode.isPrintfUsed())
            {
                const TensorPtr& printfTensor = tpcNode.getPrintfTensor();
                // due to limiting the printf buffer size - we may get nullptr
                if (printfTensor != nullptr)
                {
                    debugInfo->printf_addr_nr++;
                    // assuming printf tensors are allocated in program data
                    printf_addr.push_back(printfTensor->getDramOffset());
                }
            }
        }

        if (!profilerEnabled)
        {
            continue;
        }

        node_symbol_info_t debugNode;

        debugNode.kernel_blob_index = DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT;
        if (isTpc)
        {
            debugNode.kernel_blob_index = m_kidToBlobID[static_cast<TPCNode&>(*node).getUniqueID()];
        }

        debugNode.context_id = node->getContextId();
        debugNode.full_context_id = node->getFullContextId();
        debugNode.device_type = m_graph->getNodeUtility().getNodeDeviceType(node);

        const std::string& nodeName = node->getNodeName();
        debugNode.node_name         = m_recipeAllocator->allocate(nodeName.size() + 1);
        char* nonConstName = const_cast<char*>(debugNode.node_name);
        nodeName.copy(nonConstName, nodeName.size());
        nonConstName[nodeName.size()] = '\0';

        const std::string& operation = node->getNodeTypeStr();
        debugNode.operation          = m_recipeAllocator->allocate(operation.size() + 1);
        char* nonConstOperation = const_cast<char*>(debugNode.operation);
        operation.copy(nonConstOperation, operation.size());
        nonConstOperation[operation.size()] = '\0';

        TensorPtr        t        = node->getNumOutputs() ? node->getOutput(0) : node->getInput(0);
        std::string_view dataType = getStringFromSynDataType(t->getElementType());
        debugNode.data_type       = m_recipeAllocator->allocate(dataType.size() + 1);
        char* nonConstDataType    = const_cast<char*>(debugNode.data_type);
        dataType.copy(nonConstDataType, dataType.size());
        nonConstDataType[dataType.size()] = '\0';

        // @TODO fill with real values
        debugNode.num_descriptors = 0;

        // Fill num working engines for TPC nodes. None TPC nodes will get filled with 0.
        auto& nodeAnnotations = node->getNodeAnnotation();
        unsigned numRois = nodeAnnotations.tpcMetaData.utilizationPerLogicalRoi.size();
        debugNode.num_rois = numRois;
        debugNode.num_working_engines = (uint8_t*)m_recipeAllocator->allocate(numRois); // will get nullptr if numRois is 0
        for (unsigned roiIdx = 0; roiIdx < numRois; roiIdx++)
        {
            auto sumWorkingEngines = 0;
            for (auto& dcore : nodeAnnotations.tpcMetaData.utilizationPerLogicalRoi[roiIdx])
            {
                sumWorkingEngines += dcore.totalNumWorkingEngines;
            }
            debugNode.num_working_engines[roiIdx] = sumWorkingEngines;
        }

        debugNodes.push_back(debugNode);
    }

    if (debugNodes.size() > 0)
    {
        debugInfo->num_nodes = debugNodes.size();
        debugInfo->nodes =
            (node_symbol_info_t*)m_recipeAllocator->allocate(debugInfo->num_nodes * sizeof(node_symbol_info_t));
        std::copy(debugNodes.begin(), debugNodes.end(), debugInfo->nodes);
    }

    if (debugInfo->printf_addr_nr > 0)
    {
        debugInfo->printf_section_idx = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;
        debugInfo->printf_addr = (uint64_t*)m_recipeAllocator->allocate(debugInfo->printf_addr_nr * sizeof(uint64_t));
        for (unsigned i=0; i < printf_addr.size(); i++)
        {
            debugInfo->printf_addr[i] = printf_addr[i];
        }
    }
}

// TODO: get engine enum directly from node instead of switching on string from getEngineTypeStr
Recipe::EngineType RecipeGenerator::engineName2logical(std::string_view engineName) const
{
    if (engineName == "TPC") return Recipe::EngineType::TPC;
    if (engineName == "DMA") return Recipe::EngineType::DMA;
    if (engineName == "MME") return Recipe::EngineType::MME;
    if (engineName == "ROT") return Recipe::EngineType::ROT;
    if (engineName == "CME") return Recipe::EngineType::CME;

    HB_ASSERT(0, "Don't know to convert engine name to recipe logical engine id");
    return Recipe::EngineType::INVALID;
}

void RecipeGenerator::collectNodeSyncInfo(std::vector<NodeSyncInfo>& allNodesSyncInfo) const
{
    for (pNode node : m_sortedNodes)
    {
        if (node->isLogicalOperation()) continue;

        NodeSyncInfo syncInfo;

        syncInfo.node_exe_index = node->getExecutionOrderedIndex();
        syncInfo.node_name      = node->getNodeName();
        syncInfo.engine_type    = engineName2logical(node->getEngineTypeStr());

        if (isMMEDmaNode(node))
        {
            // This is GC way to signal the TD that this is Transpose node and it should
            // use the Transpose SOB rather than MME SOB (relevant to gaudi3 only)
            syncInfo.engine_type = Recipe::EngineType::DMA;
        }
        if (node->getNodeAnnotation().arcSyncScheme.size() > 0)
        {
            uint16_t pipe_level = 0;
            for (const auto& pipelineSync : node->getNodeAnnotation().arcSyncScheme)
            {
                // Consider only pipeline levels which signal out at end of activation
                if (pipelineSync.emittedSigVal.is_set() && pipelineSync.emittedSigVal.value())
                {
                    pipe_level++;
                }
            }
            syncInfo.pipe_level = pipe_level;
            if (node->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.is_set())
            {
                syncInfo.emitted_signal = node->getNodeAnnotation().arcSyncScheme.back().emittedSigVal.value();
                allNodesSyncInfo.push_back(syncInfo);
            }
        }
    }
}

void RecipeGenerator::serializeSyncSchemeDebugInfo(debug_sync_scheme_t* syncSchemeInfo) const
{
    std::vector<NodeSyncInfo> allNodesSyncInfo;

    collectNodeSyncInfo(allNodesSyncInfo);

    syncSchemeInfo->node_sync_info_nr       = allNodesSyncInfo.size();
    syncSchemeInfo->sync_scheme_legacy_mode = false;

    LOG_DEBUG(GC, "Node sync info size: {}", syncSchemeInfo->node_sync_info_nr);

    if (syncSchemeInfo->node_sync_info_nr > 0)
    {
        syncSchemeInfo->node_sync_info_arc = (node_sync_info_arc_t*)m_recipeAllocator->allocate(
            syncSchemeInfo->node_sync_info_nr * sizeof(node_sync_info_arc_t));

        node_sync_info_arc_t* pFiller = syncSchemeInfo->node_sync_info_arc;

        for (auto nodeInfo : allNodesSyncInfo)
        {
            pFiller->node_exe_index = nodeInfo.node_exe_index;
            pFiller->engine_type    = nodeInfo.engine_type;
            pFiller->pipe_level     = nodeInfo.pipe_level;
            pFiller->emitted_signal = nodeInfo.emitted_signal;
            pFiller++;
        }
    }
    else
    {
        syncSchemeInfo->node_sync_info_legacy = nullptr;
    }
}

void RecipeGenerator::serializeJobs(uint32_t* pNumJobs, job_t** ppJobs, const std::vector<job_t>& jobs) const
{
    *pNumJobs = jobs.size();
    *ppJobs        = (job_t*)m_recipeAllocator->allocate(*pNumJobs * sizeof(job_t));
    job_t* pFiller = *ppJobs;

    for (auto job : jobs)
    {
        pFiller->engine_id = job.engine_id;
        pFiller->program_idx = job.program_idx;
        pFiller++;
    }
}

void RecipeGenerator::serializeConfParams(uint32_t* pNumConfParams, gc_conf_t** pConfParams) const
{
    // Serializing GC configuration params to be used by RT. Initially there are 8 params (2 are greco only).
    // More can be added at the bottom. In this case pNumConfParams should be increased appropriately,
    // RecipeSerializer code should be added and the static assert below should be adjusted

    static_assert(gc_conf_t::DEVICE_TYPE == 0, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::TIME_STAMP == 1, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::TPC_ENGINE_MASK == 2, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::MME_NUM_OF_ENGINES == 3, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::DMA_ENGINE_MASK == 4, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::ROTATOR_NUM_OF_ENGINES == 6, "GC configuration param enum mismatch");
    static_assert(gc_conf_t::LAST_COMPILE_PARAM == 7, "GC configuration param enum mismatch");

    *pNumConfParams = 5;

    synDeviceType devType = m_graph->getDeviceType();

    *pConfParams = (gc_conf_t*)m_recipeAllocator->allocate(*pNumConfParams * sizeof(gc_conf_t));

    gc_conf_t* pFiller = *pConfParams;

    pFiller->conf_id    = gc_conf_t::DEVICE_TYPE;
    pFiller->conf_value = devType;
    pFiller++;

    pFiller->conf_id    = gc_conf_t::TIME_STAMP;
    pFiller->conf_value = (uint64_t)std::time(nullptr);
    pFiller++;

    pFiller->conf_id = gc_conf_t::TPC_ENGINE_MASK;
    pFiller->conf_value =
        uint64_t(m_graph->getHALReader()->getTpcEnginesMask() & GCFG_TPC_ENGINES_ENABLED_MASK.value());
    pFiller++;

    pFiller->conf_id    = gc_conf_t::MME_NUM_OF_ENGINES;
    pFiller->conf_value = m_graph->getHALReader()->getNumMmeEngines();
    pFiller++;

    pFiller->conf_id    = gc_conf_t::DMA_ENGINE_MASK;
    pFiller->conf_value = m_graph->getHALReader()->getInternalDmaEnginesMask();

    /*
    pFiller++;

    pFiller->conf_id    = gc_conf_t::USING_SINGLE_BLOB;
    */

    pFiller = *pConfParams;
    for (int i = 0; i < *pNumConfParams; i++)
    {
        LOG_TRACE(RECIPE_GEN, "Configuration param id: {}, value: {}", pFiller->conf_id, pFiller->conf_value);
        pFiller++;
    }
}

void RecipeGenerator::serializeNOPKernel(uint64_t* pNOPKernelOffset,
                                         uint64_t* pNOPKernelSection,
                                         bool*     pValidNOPKernel) const
{
    const auto& nopKernel = m_graph->getCodeGenerator()->getNOPKernel();
    *pValidNOPKernel      = nopKernel.nopKernelOffset.is_set();
    *pNOPKernelOffset     = *pValidNOPKernel ? maskOutMemoryID((nopKernel.nopKernelOffset.value())) : 0;
    *pNOPKernelSection    = nopKernel.nopKernelSection;
}

void RecipeGenerator::serializePersistTensorInfo(uint32_t*                          pNumPersistTensors,
                                                 persist_tensor_info_t**            ppTensors,
                                                 uint32_t*                          pNumSections,
                                                 const std::map<uint64_t, uint8_t>& sectionIdToSectionType) const
{
    struct ExtTensorOrder
    {
        uint32_t tensorId;
        uint32_t tensorOder;
    };
    std::vector<ExtTensorOrder> extTensorOrder;
    for (pTensor t : m_persistTensorsPreCompilation)
    {
        if (t->getTensorIsExternal())
        {
            uint32_t tensorOrder = m_graph->getTensorProducer(t)->getExecutionOrderedIndex();
            extTensorOrder.push_back({t->getId(), tensorOrder});
        }
    }
    std::sort(extTensorOrder.begin(), extTensorOrder.end(), [](ExtTensorOrder const& lhs, ExtTensorOrder const& rhs) {
        return lhs.tensorOder < rhs.tensorOder;
    });

    *pNumPersistTensors = m_persistTensorsPreCompilation.size();
    std::set<uint64_t>              uniqueSections;
    std::map<TensorPtr, gc::Layout> graphInputLayouts = m_graph->getInputInferenceLayouts();

    *ppTensors =
        (persist_tensor_info_t*)m_recipeAllocator->allocate(*pNumPersistTensors * sizeof(persist_tensor_info_t));
    persist_tensor_info_t* pFiller = *ppTensors;

    for (pTensor t : m_persistTensorsPreCompilation)
    {
        const std::string& tensorName = t->getName();
        pFiller->offset_in_section    = t->getMemorySectionOffset();
        pFiller->section_idx          = t->getMemorySectionID();
        pFiller->section_type         = m_patchPointContainer.getSectionTypeForSectionId(sectionIdToSectionType, pFiller->section_idx);
        pFiller->size                 = t->getTotalSizeInBytes();
        pFiller->name                 = m_recipeAllocator->allocate(tensorName.size() + 1);
        char* nonConstName            = const_cast<char*>(pFiller->name);
        tensorName.copy(nonConstName, tensorName.size());
        nonConstName[tensorName.size()] = '\0';
        uniqueSections.insert(t->getMemorySectionID());

        pFiller->elementType          = t->getElementType();
        pFiller->tensorType           = t->getTensorType();
        pFiller->zp                   = t->getZeroPoint();
        pFiller->scale                = t->getScale();
        pFiller->dimensions           = t->getDim();
        t->getAllNSizesInElements(pFiller->dimensionsSize);

        pFiller->isExternal = t->getTensorIsExternal();
        if (t->getTensorIsExternal())
        {
            auto it = std::find_if(extTensorOrder.begin(),
                                   extTensorOrder.end(),
                                   [id = t->getId()](ExtTensorOrder const& item) { return item.tensorId == id; });

            pFiller->extTensorExeOrder = std::distance(extTensorOrder.begin(), it);
        }
        else
        {
            pFiller->extTensorExeOrder = std::numeric_limits<uint32_t>::max();
        }

        if (m_bucketViewsStartIndexAndSize.find(t) != m_bucketViewsStartIndexAndSize.end())
        {
            uint32_t viewsStartIdx          = m_bucketViewsStartIndexAndSize.at(t).first;
            uint32_t viewNum                = m_bucketViewsStartIndexAndSize.at(t).second;
            pFiller->multi_views_indices_nr = viewNum;
            pFiller->multi_views_indices    = (uint32_t*)m_recipeAllocator->allocate(viewNum * sizeof(uint32_t));
            for (unsigned i = 0; i < viewNum; i++)
            {
                pFiller->multi_views_indices[i] = viewsStartIdx + i;
            }
        }
        else
        {
            pFiller->multi_views_indices    = nullptr;
            pFiller->multi_views_indices_nr = 0;
        }

        pFiller->layout = nullptr;
        if (graphInputLayouts.find(t) != graphInputLayouts.end())
        {
            std::string_view layoutStr = graphInputLayouts.at(t).toString();
            HB_ASSERT(layoutStr.size() + 1 < MAX_LAYOUT_SIZE, "layout string size should be less than {}", MAX_LAYOUT_SIZE);
            pFiller->layout                 = m_recipeAllocator->allocate(layoutStr.size() + 1);
            char* nonConstLayout            = const_cast<char*>(pFiller->layout);
            layoutStr.copy(nonConstLayout, layoutStr.size());
            nonConstLayout[layoutStr.size()] = '\0';
        }

        // fill permutation info
        serializeTensorPermutation(pFiller->permutation, t->getPermutation(), HABANA_DIM_MAX);

        // if a persistent tensor is not an input we consider it as output as the user may read it
        pFiller->isInput = !m_graph || m_graph->isInputTensor(t);

        unsigned batchPos = t->getBatchPos();
        if ((batchPos == INVALID_BATCH_POS) ||(batchPos != (t->getDim() - 1)))
        {
            // batch position is invalid or not last
            pFiller->batchSize = 0; // this is marked as unset
        }
        else
        {
            pFiller->batchSize = t->getSizeInElements(batchPos);
        }

        pFiller++;
    }

    *pNumSections = uniqueSections.size() + s_numOfWorkspaces;
}

void RecipeGenerator::findConstSections(std::map<uint32_t, std::vector<TensorPtr>>& constSectionIdToTensors,
                                        std::set<uint32_t>&                         zeroSizeSections) const
{
    // Collect all const section tensors that were actually used in the graph (so traverse the post-compile list)
    for (const TensorPtr& t : m_persistTensorsPostCompilation)
    {
        if (t->inConstSection())
        {
            constSectionIdToTensors[t->getMemorySectionID()].push_back(t);
        }
    }

    // locate sections that may have been removed during graph optimization.
    uint32_t sectionID = 0;
    for (auto t : m_graph->getConstSectionTensors())
    {
        sectionID = t->getMemorySectionID();

        if (constSectionIdToTensors.find(sectionID) == constSectionIdToTensors.end())
        {
            LOG_DEBUG(GC, "Const section: {} has zero size", sectionID);
            zeroSizeSections.insert(sectionID);
        }
    }
}

void RecipeGenerator::serializeConstSections(uint32_t* pNumConstSections, const_section_t** ppConstSections) const
{
    std::map<uint32_t, std::vector<TensorPtr>> constSectionIdToTensors;
    std::set<uint32_t>                         zeroSizeSections;

    findConstSections(constSectionIdToTensors, zeroSizeSections);

    *pNumConstSections = constSectionIdToTensors.size() + zeroSizeSections.size();

    if (*pNumConstSections == 0)
    {
        *ppConstSections = nullptr;
        return;
    }

    LOG_DEBUG(GC,
              "Serializing {} const sections and {} zero size sections",
              constSectionIdToTensors.size(),
              zeroSizeSections.size());

    *ppConstSections = (const_section_t*)m_recipeAllocator->allocate(*pNumConstSections * sizeof(const_section_t));

    const_section_t* pFiller = *ppConstSections;
    for (auto iter = constSectionIdToTensors.begin(); iter != constSectionIdToTensors.end(); iter++)
    {
        pFiller->section_idx = iter->first;

        // find total section size in bytes and sort tensors by offset in section
        uint64_t                      sizeInBytes = 0;
        std::map<uint64_t, TensorPtr> offsetToTensor;

        for (auto itr = iter->second.begin(); itr != iter->second.end(); itr++)
        {
            sizeInBytes += (*itr)->getTotalSizeInBytes();
            offsetToTensor[(*itr)->getMemorySectionOffset()] = *itr;
        }
        pFiller->size = sizeInBytes;

        LOG_DEBUG(GC, "Serializing const section: {} size: {}", pFiller->section_idx, pFiller->size);

        // finally allocate space and copy tensors data
        pFiller->data   = m_recipeAllocator->allocate(sizeInBytes);
        uint64_t offset = 0;

        for (auto itr = offsetToTensor.begin(); itr != offsetToTensor.end(); itr++)
        {
            memcpy(pFiller->data + offset, itr->second->getData(), itr->second->getTotalSizeInBytes());
            offset += itr->second->getTotalSizeInBytes();

            // Unbind const section tensors with copyData=true - not needed once serialized
            if (!itr->second->isAliasedTensor() && !itr->second->isRealInAliasing())
            {
                itr->second->unbind();
            }
        }
        pFiller++;
    }

    // serialize the zero size sections
    for (auto iter = zeroSizeSections.begin(); iter != zeroSizeSections.end(); iter++)
    {
        pFiller->section_idx = *iter;
        pFiller->size        = 0;
        pFiller->data        = nullptr;

        pFiller++;
    }
}

void RecipeGenerator::serializeProgramDataBlobs(uint32_t* pNumDataBlobs, program_data_blob_t** ppBlobs, char** programDataBlobsBuffer, uint64_t* pDataBlobsSize) const
{
    static std::atomic<uint64_t> serialNum{0}; // TODO: hash is not implemented for v0.5.0
    size_t blobIndex = 0;
    *pNumDataBlobs = m_programDataBlobsHolder.size();
    *ppBlobs = (program_data_blob_t*)m_recipeAllocator->allocate(*pNumDataBlobs * sizeof(program_data_blob_t));
    program_data_blob_t* pFiller = *ppBlobs;

    // allocate blobs for program data once - for all blobs
    *programDataBlobsBuffer = m_recipeAllocator->allocate(m_dataBlobsSizeInBytes, true);

    if (m_dataBlobsSizeInBytes)
    {
        // Init program data blobs buffer
        memset(*programDataBlobsBuffer, 0xFF, m_dataBlobsSizeInBytes);
    }

    *pDataBlobsSize = m_dataBlobsSizeInBytes;
    unsigned dataOffset = 0;

    for (const std::shared_ptr<ProgramDataBlob>& blob : m_programDataBlobsHolder)
    {
        char* hostAddr = nullptr;

        if (blob->hostAddrPtr != nullptr)
        {
            HB_ASSERT(blob->hostAddrSharedPtr == nullptr, "hostAddrPtr and hostAddrSharedPtr cannot be set simultanusly");
            hostAddr = blob->hostAddrPtr;
        }
        else if (blob->hostAddrSharedPtr != nullptr)
        {
            HB_ASSERT(blob->hostAddrPtr == nullptr, "hostAddrPtr and hostAddrSharedPtr cannot be set simultanusly");
            hostAddr = blob->hostAddrSharedPtr.get();
        }
        else
        {
            HB_ASSERT(0, "host pointer was not set, this is impossible");
        }

        pFiller->size = blob->binSize;
        pFiller->data = *programDataBlobsBuffer + dataOffset;
        HB_ASSERT(dataOffset + blob->binSize <= m_dataBlobsSizeInBytes, "m_dataBlobsSizeInBytes overflow");
        memcpy((char*)pFiller->data, hostAddr, blob->binSize);
        pFiller->offset_in_section = maskOutMemoryID(blob->deviceAddr);
        pFiller->section_idx = MEMORY_ID_RESERVED_FOR_PROGRAM_DATA;

        // All GC allocations should have been done with cache-line alignment
        dataOffset += alignSizeUp(pFiller->size, m_graph->getHALReader()->getCacheLineSizeInBytes());

        const TPCProgramDataBlob* tpcBlob = dynamic_cast<TPCProgramDataBlob*>(blob.get());
        if (tpcBlob != nullptr)
        {
            m_kidToBlobID[tpcBlob->kid] = blobIndex;
        }

        blobIndex++;
        pFiller++;
    }
}

void RecipeGenerator::serializeWorkspaceSizes(uint32_t*   pNumWorkspaces,
                                              uint64_t**  ppWorkspaceSizes,
                                              uint64_t    blobsSizeInBytes) const
{
    *pNumWorkspaces    = s_numOfWorkspaces;
    *ppWorkspaceSizes  = (uint64_t*)m_recipeAllocator->allocate(*pNumWorkspaces * sizeof(uint64_t));

    (*ppWorkspaceSizes)[MEMORY_ID_RESERVED_FOR_WORKSPACE] =
        m_workspaceSizeInBytes + calcPaddingSize(m_workspaceSizeInBytes, WORKSPACE_SIZE_GRANULARITY_IN_BYTES);

    (*ppWorkspaceSizes)[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA] =
        m_dataBlobsSizeInBytes + calcPaddingSize(m_dataBlobsSizeInBytes, WORKSPACE_SIZE_GRANULARITY_IN_BYTES);

    (*ppWorkspaceSizes)[MEMORY_ID_RESERVED_FOR_PROGRAM] =
        blobsSizeInBytes + calcPaddingSize(blobsSizeInBytes, WORKSPACE_SIZE_GRANULARITY_IN_BYTES);
}

void RecipeGenerator::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN)) return;

    LOG_DEBUG(RECIPE_GEN, "Recipe Generator Dump:");
    LOG_DEBUG(RECIPE_GEN, "  Version major = {}", RECIPE_VERSION_MAJOR);
    LOG_DEBUG(RECIPE_GEN, "  Version minor = {}", RECIPE_VERSION_MINOR);

    uint64_t totalBlobsSize = m_blobContainer.print();
    m_programContainer.print();
    m_patchPointContainer.print();
    printAddrPatchPointBreakdown();

    // jobs
    LOG_DEBUG(RECIPE_GEN, "  Jobs Dump:");
    printJobs(m_activateJobs, "Activate");
    printJobs(m_executeJobs,  "Execute");

    // persistent tensors
    LOG_DEBUG(RECIPE_GEN, "  Persistent Tensor Info Dump:");
    LOG_DEBUG(RECIPE_GEN, "    Number of persistent tensors = {}", m_persistTensorsPreCompilation.size());
    uint64_t i = 0;
    for (const pTensor& t : m_persistTensorsPreCompilation)
    {
        LOG_DEBUG(RECIPE_GEN, "    Persistent tensor info {}:", i++);
        LOG_DEBUG(RECIPE_GEN, "      name = {}", t->getName());
        LOG_DEBUG(RECIPE_GEN, "      section index = {}", t->getMemorySectionID());
        LOG_DEBUG(RECIPE_GEN, "      offset in section = 0x{:x} (hard-code until multi tensor per section is supported)", 0);
    }

    // data blobs
    LOG_DEBUG(RECIPE_GEN, "  Program Data Blob Dump:");
    LOG_DEBUG(RECIPE_GEN, "    Number of data blobs = {}", m_programDataBlobsHolder.size());
    i = 0;
    for (const std::shared_ptr<ProgramDataBlob>& blob : m_programDataBlobsHolder)
    {
        LOG_DEBUG(RECIPE_GEN, "    Data blob {}:", i++);
        LOG_DEBUG(RECIPE_GEN, "      HBM effective (virtual) address = 0x{:x}", maskOutMemoryID(blob->deviceAddr));
        LOG_DEBUG(RECIPE_GEN, "      size = {}", blob->binSize);

        const TPCProgramDataBlob* tpcBlob = dynamic_cast<TPCProgramDataBlob*>(blob.get());
        if (tpcBlob != nullptr)
        {
            LOG_DEBUG(RECIPE_GEN, "      kernel ID = {}", tpcBlob->kid);
        }
    }

    // number of sections
    LOG_DEBUG(RECIPE_GEN,
              "  Number of sections (workspaces and persist tensors) = {}",
              m_persistTensorsPreCompilation.size() + s_numOfWorkspaces);

    // workspaces sizes
    LOG_DEBUG(RECIPE_GEN, "  Workspace sizes including padding to 128 bytes:");
    LOG_DEBUG(RECIPE_GEN, "    Non-persist tensors = {}",
        m_workspaceSizeInBytes + calcPaddingSize(m_workspaceSizeInBytes, WORKSPACE_SIZE_GRANULARITY_IN_BYTES));
    LOG_DEBUG(RECIPE_GEN,
              "    Program data = {}",
              m_dataBlobsSizeInBytes + calcPaddingSize(m_dataBlobsSizeInBytes, WORKSPACE_SIZE_GRANULARITY_IN_BYTES));
    LOG_DEBUG(RECIPE_GEN, "    Program instructions = {}",
        totalBlobsSize + calcPaddingSize(totalBlobsSize, WORKSPACE_SIZE_GRANULARITY_IN_BYTES));

    LOG_DEBUG(RECIPE_GEN, "  Arc Jobs Dump:");
    m_ecbContainer.print();

    LOG_DEBUG(RECIPE_GEN, "  Max Used Mcid Degrade = {}",
        m_graph->getCodeGenerator()->getMcidConverter().getMaxUsedPhysicalDegrade());
    LOG_DEBUG(RECIPE_GEN, "  Max Used Mcid Discard = {}",
        m_graph->getCodeGenerator()->getMcidConverter().getMaxUsedPhysicalDiscard());

    LOG_DEBUG(RECIPE_GEN, "--End of Recipe Generator Dump--");
}

void RecipeGenerator::printJobs(const std::vector<job_t>& jobList, std::string_view jobListName) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN)) return;

    LOG_DEBUG(RECIPE_GEN, "    Number of {} jobs = {}", jobListName, jobList.size());

    std::unordered_set<uint64_t> usedIndicies;
    uint64_t i = 0;

    for (auto job : jobList)
    {
        LOG_DEBUG(RECIPE_GEN, "    Job {}:", i++);
        LOG_DEBUG(RECIPE_GEN, "      engine {} (id:{})", getEngineStr(job.engine_id), job.engine_id);
        LOG_DEBUG(RECIPE_GEN, "      Program Dump (if not visible, open QMAN logger):");

        if (LOG_LEVEL_AT_LEAST_DEBUG(QMAN))
        {
            LOG_DEBUG(QMAN, "      ======== Start of Program for Engine {} ========", getEngineStr(job.engine_id));
        }

        auto& blobIndicies = m_programContainer.getProgramByIndex(job.program_idx).blobIndicies();

        uint64_t size        = 0;
        uint64_t uniqueBlobs = 0;
        for (auto& blobIndex : blobIndicies)
        {
            const RecipeBlob* blob = m_blobContainer.getBlobByIndex(blobIndex);

            if (LOG_LEVEL_AT_LEAST_DEBUG(QMAN))
            {
                std::string blobType = blob->isWorkDistBlob() ? "WD" : blob->isPatchingBlob() ? "patch" : "exe";
                LOG_DEBUG(QMAN,
                          "      -------- start of {} blob, index {}, id {} --------",
                          blobType,
                          blobIndex,
                          blob->getBlobId());

                blob->printQueCmds();
            }
            if (usedIndicies.insert(blobIndex).second)
            {
                size += blob->sizeInBytes();
                uniqueBlobs++;
            }
        }
        if (LOG_LEVEL_AT_LEAST_DEBUG(QMAN))
        {
            LOG_DEBUG(QMAN, "      ======== end of program ========");
        }
        LOG_DEBUG(RECIPE_GEN, "      Total size of unique blobs: {}, unique blob count: {} ", size, uniqueBlobs);
    }
}

void RecipeGenerator::printAddrPatchPointBreakdown() const
{
    return;  // it's expensive function, let it run only if you need it

    LOG_DEBUG(GC, "  Address Patch-points Breakdown:");

    for (auto job : m_executeJobs)
    {
        unsigned total = 0, full = 0, high = 0, low = 0;
        for (auto blobIdx : m_programContainer.getProgramByIndex(job.program_idx).blobIndicies())
        {
            if (m_blobContainer.getBlobByIndex(blobIdx)->isPatchingBlob() == false) continue;
            for (auto pp : m_patchPointContainer.getPatchPoints())
            {
                if (pp->getBlobIndex() == blobIdx)
                {
                    if (pp->getType() == patch_point_t::EPatchPointType::SIMPLE_DDW_MEM_PATCH_POINT)     { full++; total++; }
                    if (pp->getType() == patch_point_t::EPatchPointType::SIMPLE_DW_HIGH_MEM_PATCH_POINT) { high++; total++; }
                    if (pp->getType() == patch_point_t::EPatchPointType::SIMPLE_DW_LOW_MEM_PATCH_POINT)  { low++;  total++; }
                }
            }
        }
        LOG_DEBUG(GC, "    {} has {} Address Patch-points (full={}, high={}, low={})", getEngineStr(job.engine_id), total, full, high, low);
    }
}

void RecipeGenerator::printShapePlane(shape_plane_graph_t* recipe) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    LOG_TRACE(GC, "Dumping Shape plane graph {}.{}", recipe->version_major, recipe->version_minor);
    for (int i = 0; i < recipe->sp_node_nr; i++)
    {
        shape_plane_node_t currNode = recipe->sp_nodes[i];
        LOG_TRACE(GC, "Node {} ({}):", currNode.node_name, i);
        LOG_TRACE(GC, "     Inputs:");
        for (int in_tensor = 0; in_tensor < currNode.input_tensors_nr; in_tensor++)
        {
            printTensor(recipe, currNode.input_tensors[in_tensor]);
        }
        LOG_TRACE(GC, "     Outputs:");
        for (int out_tensor = 0; out_tensor < currNode.output_tensors_nr; out_tensor++)
        {
            printTensor(recipe, currNode.output_tensors[out_tensor]);
        }

        for (int roi = 0; roi < currNode.activation_rois_nr; roi++)
        {
            LOG_TRACE(GC, "     Roi {}:", roi);
            printRoi(currNode.activation_rois[roi]);

            LOG_TRACE(GC, "         Patch Points :");
            int pp_this_roi = 0;
            for(int pp = 0; pp < currNode.node_patch_points_nr; pp++)
            {
                if (currNode.node_patch_points[pp].roi_idx == roi)
                {
                    LOG_TRACE(GC,
                              "             {} : {}",
                              pp_this_roi++,
                              ShapeFuncRegistry::instance().getSmfName(currNode.node_patch_points[pp].smf_id));
                }
            }
        }
    }
    LOG_TRACE(GC, "Dumping Shape plane graph Done");
    LOG_TRACE_DYNAMIC_PATCHING("TRACE_DYNAMIC_PATCHING on");
}

void RecipeGenerator::printTensor(shape_plane_graph_t* recipe, uint64_t index) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    tensor_info_t& currTensor = recipe->sp_tensors[index];
    LOG_TRACE(GC, "         Tensor {} : type = {}, datatype = {}", index, currTensor.tensor_type, currTensor.data_type);
    LOG_TRACE(GC, "             max sizes ({})", getDimStr(currTensor.max_dims, 5));
    LOG_TRACE(GC, "             min sizes ({})", getDimStr(currTensor.min_dims, 5));
    LOG_TRACE(GC, "             strides   ({})", getDimStr(currTensor.strides, 5));
}

void RecipeGenerator::printRoi(roi_info_t& roi) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GC)) return;

    LOG_TRACE(GC, "         Inputs :");
    for(int i = 0; i < roi.roi_in_tensor_nr; i++)
    {
        LOG_TRACE(GC, "             Tensor {} :", i);
        LOG_TRACE(GC, "                 Offset ({}) ", getDimStr(roi.roi_in_tensors[i].roi_offset_dims, 5));
        LOG_TRACE(GC, "                 Sizes  ({}) ", getDimStr(roi.roi_in_tensors[i].roi_size_dims, 5));
    }

    LOG_TRACE(GC, "         Outputs :");
    for(int i = 0; i < roi.roi_out_tensor_nr; i++)
    {
        LOG_TRACE(GC, "             Tensor {} :", i);
        LOG_TRACE(GC, "                 Offset ({}) ", getDimStr(roi.roi_out_tensors[i].roi_offset_dims, 5));
        LOG_TRACE(GC, "                 Sizes  ({}) ", getDimStr(roi.roi_out_tensors[i].roi_size_dims, 5));
    }

    uint32_t zeros[sizeof(roi.index_space_size)] = {0};
    if (memcmp(roi.index_space_size, zeros, sizeof(roi.index_space_size)) != 0)
    {
        LOG_TRACE(GC, "         Index Space :");
        LOG_TRACE(GC, "             Offset ({}) :", getDimStr(roi.index_space_offset, 5));
        LOG_TRACE(GC, "             Sizes  ({}) ", getDimStr(roi.index_space_size, 5));
    }
}
