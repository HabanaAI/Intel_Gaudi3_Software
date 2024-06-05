#pragma once

#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <sstream>
#include "utils.h"
#include "types.h"
#include "tensor.h"
#include "program_data_blob.h"
#include "recipe_blob.h"
#include "recipe_program.h"
#include "recipe_patch_point.h"
#include "recipe_ecb.h"
#include "shape_node.h"
#include "recipe_version.h"

#include "define_synapse_common.hpp"

class  QueueCommand;
class  BasicFieldsContainerInfo;
class  RecipeBlob;
struct job_t;
struct persist_tensor_info_t;
struct recipe_t;
class RecipeAllocator;

typedef std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint64_t> > NodeExeMap;

struct NodeSyncInfo
{
    Recipe::EngineType engine_type;
    uint32_t           node_exe_index;
    std::string        node_name;
    uint16_t           pipe_level     = 0;
    uint16_t           emitted_signal = 0;
    uint16_t           sob_id         = 0;
    uint16_t           num_engines    = 0;
};

class RecipeGenerator
{
public:
    RecipeGenerator(const HabanaGraph* g);

    virtual ~RecipeGenerator();

    void                 generateRecipes(bool isDynamicShapeGraph);

    recipe_t*            serializeDataPlaneGraph(RecipeAllocator* pRecipeAlloc) const;
    shape_plane_graph_t* serializeShapePlane(RecipeAllocator* pRecipeAlloc) const;
    void                 print() const;
    virtual bool         isMMEDmaNode(const NodePtr& n) const { return false; }

    static const uint32_t DEBUG_KERNEL_BLOB_INDEX_IRRELEVANT = 0xffffffff;

protected:

    void generateShapePlanRecipe();

    // Non pure virtual to allow generic testing
    virtual std::string getEngineStr(unsigned id) const { return ""; }
    void                serializeProfileDebugInfo(debug_info_t* debugInfo) const;
    virtual void        serializeSyncSchemeDebugInfo(debug_sync_scheme_t* syncSchemeInfo) const;
    virtual void        validateQueue(ConstCommandQueuePtr queue, bool isSetup) const {}
    virtual bool        shouldCreateECBs() const { return false; }
    virtual uint32_t    getVersionMinor() const { return RECIPE_VERSION_MINOR; }
    virtual void inspectRecipePackets(const void* buffer, unsigned bufferSize, std::string_view bufferName) const {}
    virtual void        setBlobFlags(RecipeBlob* blob, QueueCommand* cmd) const;
    virtual bool        isSFGInitCommand(QueueCommand* cmd) const { return false; }
    virtual unsigned    getInitSfgValue(QueueCommand* cmd) const { return 0; }

    Recipe::EngineType engineName2logical(std::string_view engineName) const;

    RecipeEcbContainer       m_ecbContainer;
    const NodeVector&        m_sortedNodes;
    mutable RecipeAllocator* m_recipeAllocator;

private:
    RecipeGenerator() = delete;
    RecipeGenerator(const RecipeGenerator&) = delete;
    RecipeGenerator& operator=(const RecipeGenerator&) = delete;

    void initPrograms();
    void createPrograms(bool isSetup);
    void processCmd(QueueCommand* cmd, unsigned qid, HabanaDeviceType devType, bool isSetup);
    void finalizePatchPoints(bool patchingBlobReused, uint64_t patchingBlobIndex, unsigned blobId);
    void finalizeCompressibleDynamicPatchPoints(unsigned blobId);
    void createJobs();
    void printJobs(const std::vector<job_t>& jobList, std::string_view jobListName) const;

    void saveProgramDataInfo(const ProgramDataBlobSet& programDataBlobs);

    void serializeJobs(uint32_t* pNumJobs, job_t** ppJobs, const std::vector<job_t>& jobs) const;

    void serializeConfParams(uint32_t* pNumConfParams, gc_conf_t** pConfParams) const;

    void serializeNOPKernel(uint64_t* pNOPKernelOffset, uint64_t* pNOPKernelSection, bool* pValidNOPKernel) const;

    void serializePersistTensorInfo(uint32_t*                          pNumPersistTensors,
                                    persist_tensor_info_t**            ppTensors,
                                    uint32_t*                          pNumSections,
                                    const std::map<uint64_t, uint8_t>& sectionIdToSectionType) const;

    void findConstSections(std::map<uint32_t, std::vector<TensorPtr>>& constSectionIdToTensors,
                           std::set<uint32_t>&                         zeroSizeSections) const;

    virtual void collectNodeSyncInfo(std::vector<NodeSyncInfo>& allNodesSyncInfo) const;

    void serializeConstSections(uint32_t* pNumConstSections, const_section_t** ppConstSections) const;

    void serializeProgramDataBlobs(uint32_t*              pNumDataBlobs,
                                   program_data_blob_t**  ppBlobs,
                                   char**                 programDataBlobsBuffer,
                                   uint64_t*              pDataBlobsSize) const;

    void serializeWorkspaceSizes(uint32_t*   pNumWorkspaces,
                                 uint64_t**  ppWorkspaceSizes,
                                 uint64_t    blobsSizeInBytes) const;

    void serializeNodesShapePlane(shape_plane_node_t** shapeNodes, uint32_t* shapeNodesAmount) const;
    void serializeShapeTensors(shape_tensor_info_t** shapeTensors, uint64_t* serializedTensorAmount) const;
    void serializeTensorsShapePlane(tensor_info_t** serializedTensors, uint64_t* serializedTensorAmount) const;

    void serializeTensorPermutation(uint8_t*                              tensorPermutation,
                                    const std::optional<gc::Permutation>& permutation,
                                    unsigned                              maxDims) const;

    uint32_t getTensorIndex(const pTensor tensor, const TensorSet& tensorSet) const;

    void serializeNodeExecutionList(uint32_t* pNodeNum, node_program_t** pNodeExecList, uint64_t programNum) const;

    void printAddrPatchPointBreakdown() const;
    void printShapePlane(shape_plane_graph_t* recipe) const;
    void printTensor(shape_plane_graph_t* recipe, uint64_t index) const;
    void printRoi(roi_info_t& roi) const;

    RecipeBlobContainer                      m_blobContainer;
    RecipeProgramContainer                   m_programContainer;
    RecipePatchPointContainer                m_patchPointContainer;
    std::list<uint64_t>                      m_patchPointsToUpdate; // hold incomplete patch point indices
    std::vector<job_t>                       m_activateJobs;
    std::vector<job_t>                       m_executeJobs;
    const HabanaGraph*                       m_graph;
    const ProgramDataBlobSet&                m_programDataBlobsHolder;
    TensorSet                                m_persistTensorsPostCompilation;
    const TensorSet&                         m_persistTensorsPreCompilation;
    const uint64_t                           m_workspaceSizeInBytes;
    const uint64_t                           m_dataBlobsSizeInBytes;
    uint32_t                                 m_numOfH2DITensors;
    mutable std::map<kernelID, uint32_t>     m_kidToBlobID;
    ShapePlaneInfoContainer                  m_shapePlaneContainer;
    TensorSet                                m_shapeTensors;
    NodeExeMap                               m_nodeExeToProgramBlobsCount;
    std::unordered_map<uint64_t, uint32_t>   m_programIdxToLastNodeExeIndex;

    mutable std::unordered_map<TensorPtr, std::pair<uint32_t, uint32_t>> m_bucketViewsStartIndexAndSize;

    std::unordered_map<uint64_t, std::vector<uint64_t> > m_blobIndexListOfPatchPoints;

    // We have three workspaces: (1) non-persistent tensor, (2) program data, (3) program blobs
    // this is tightly coupled with the first persistent tensor memory id.
    static const uint64_t s_numOfWorkspaces = MEMORY_ID_RESERVED_FOR_PROGRAM + 1;

    std::unordered_set<ShapeNode*> m_shapeNodesToUpdate;
};
