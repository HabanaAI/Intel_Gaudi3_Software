#pragma once

#include "graph_compiler/smf/shape_func_registry.h"
#include "synapse_api_types.h"
#include "types.h"
#include "vtune_stat.h"

#include "runtime/common/recipe/patching/define.hpp"
#include "runtime/scal/common/patching/recipe_addr_patcher.hpp"

#include <mutex>

#define LOG_DSD_EXTRA(...) do { if (false) LOG_DSD_TRACE(__VA_ARGS__) } while(false);
#ifndef VTUNE_ENABLED
#define EXTRA_CHECKING
#endif
// #define PERF_LOG_LEVEL0 // Collect time of SIF only and SMF only function (the collection impacts total time)
// #define LOG_DSD_EXTRA(...) LOG_DSD_TRACE(__VA_ARGS__)

struct basicRecipeInfo;
struct DeviceAgnosticRecipeInfo;

class DynamicRecipe
{
    using TensorDBType = shape_plane_basic_node_t::EShapePlanceTensorDb;

    struct StatsCol
    {
        uint64_t totalBypass  = 0;
        uint64_t totalSkipPP  = 0;
        uint64_t totalNoPatch = 0;
    };

public:
    // Gaudi Ctor
    DynamicRecipe(const basicRecipeInfo&            rRecipeInfo,
                  const DeviceAgnosticRecipeInfo&   rDeviceAgnosticRecipeInfo,
                  const DataChunkSmPatchPointsInfo* pDataChunkSmPatchPointsInfo,
                  const data_chunk_patch_point_t*   originalPatchPoints)
                  : DynamicRecipe(rRecipeInfo, rDeviceAgnosticRecipeInfo, pDataChunkSmPatchPointsInfo, originalPatchPoints, nullptr) {}

    // Gaudi3 Ctor
    DynamicRecipe(const basicRecipeInfo&            rRecipeInfo,
                  const DeviceAgnosticRecipeInfo&   rDeviceAgnosticRecipeInfo,
                  const DataChunkSmPatchPointsInfo* pDataChunkSmPatchPointsInfo,
                  const RecipeAddrPatcher*          pRecipeAddrPatcher)
                  : DynamicRecipe(rRecipeInfo, rDeviceAgnosticRecipeInfo, pDataChunkSmPatchPointsInfo, nullptr, pRecipeAddrPatcher) {}

    bool runSifOnAllNodes(const synLaunchTensorInfoExt* launchTensorsInfo,
                          const uint32_t                launchTensorsAmount,
                          const std::vector<uint32_t>*  tensorIdx2userIdx,
                          uint64_t                      programDataHostAddress);

    bool patch(const synLaunchTensorInfoExt* launchTensorsInfo,
               const uint32_t                launchTensorsAmount,
               const std::vector<uint64_t>&  dataChunksHostAddresses,
               const std::vector<uint32_t>*  tensorIdx2userIdx);

    bool runSmfOnNodes(const std::vector<uint64_t>& dataChunksHostAddresses,
                       uint32_t                     firstNodeIndex,
                       uint32_t                     lastNodeIndex);

    bool runSmfOnAllNodes(const std::vector<uint64_t>& dataChunksHostAddresses);

    void patchAbort();

    bool takeOwnership();
    void releaseOwnership();

    const static char* staticGetTensorName(uint64_t tensor, const basicRecipeInfo* pRecipeInfo);

    inline RecipeAddrPatcher& getRecipeAddrPatcher() { return m_recipeAddrPatcher; };

    data_chunk_patch_point_t*  getPatchPoints() { return m_patchPoints.data(); }
    std::vector<tensor_info_t> getDynamicShapesTensorInfoArray() const { return m_sp_tensors_private; };

private:
#if 0
    void dumpBlobData(); // For dbug only (prints to screen)
    void dumpPP();       // For dbug only (prints to screen)
#endif
    DynamicRecipe(const basicRecipeInfo&            rRecipeInfo,
                  const DeviceAgnosticRecipeInfo&   rDeviceAgnosticRecipeInfo,
                  const DataChunkSmPatchPointsInfo* pDataChunkSmPatchPointsInfo,
                  const data_chunk_patch_point_t*   originalPatchPoints,
                  const RecipeAddrPatcher*          pRecipeAddrPatcher);

    bool init(const synLaunchTensorInfoExt* launchTensorsInfo,
              const uint32_t                launchTensorsAmount,
              const std::vector<uint32_t>*  tensorIdx2userIdx);
    bool initRecipe();
    bool initTensors();
    bool runSifOnNodes();
    bool runSifOnNodes(const synLaunchTensorInfoExt* launchTensorsInfo,
                       const uint32_t                launchTensorsAmount,
                       const std::vector<uint32_t>*  tensorIdx2userIdx,
                       uint64_t                      programDataHostAddress);
    bool runSmf(const std::vector<uint64_t>& dataChunksHostAddresses, uint32_t firstNodeIndex, uint32_t lastNodeIndex);
    bool verifyOutputs();

    bool runSif(uint64_t nodeIdx, uint64_t programDataHostAddress);
    void handleHostToDeviceTensors(uint32_t nodeIdx, uint64_t programDataHostAddress);
    void getFuserNodeSifParams(uint32_t nodeIdx, uint32_t subNode, SifParams* pSifParams, SifOutputs* pSifOutputs);

    bool patchPPs(int                                 nodeIdx,
                  const data_chunk_sm_patch_point_t*& pCurrentDcSmPatchPoint,
                  const std::vector<uint64_t>&        dataChunksHostAddresses,
                  StatsCol&                           stats);

    bool runSmf(int                                nodeIdx,
                int                                ppIdx,
                shape_plane_node_t&                currNode,
                const data_chunk_sm_patch_point_t& currentPatchPoint,
                bool&                              shouldBypass,
                StatsCol&                          stats);

    bool validateSifOutput(uint32_t nodeIdx, uint32_t subNode, unsigned* invalidMask);

    bool        verifyTensorSize(uint64_t     tensor,
                                 TensorDBType fuserType,
                                 bool         compareToLaunchInfo,
                                 uint64_t     nodeIdx,
                                 uint32_t     tensorIdx,
                                 uint32_t     subNode);
    const char* getTensorName(uint64_t tensor) { return staticGetTensorName(tensor, &m_rRecipeInfo); };

private:
    static const uint64_t NODE_IDX_FOR_INIT_TENSOR    = 0xFFFFFFFFFFFFFFFF;
    static const uint64_t NODE_IDX_FOR_VERIFY_OUTPUTS = 0xFFFFFFFFFFFFFFFE;

    const basicRecipeInfo&          m_rRecipeInfo;                // init in constructor
    const DeviceAgnosticRecipeInfo& m_rDeviceAgnosticRecipeInfo;  // init in constructor
    const synLaunchTensorInfoExt*   m_launchTensorsInfo;          // init every run
    uint32_t                        m_launchTensorsAmount;        // init every run
    const std::vector<uint32_t>*    m_tensorIdx2userIdx;          // init every run

    const static size_t nodeDataSize = sizeof(shape_plane_node_t::nodeData) / sizeof(uint64_t);

    // Below are copies of fields in the recipe that can change during the dynamic patching
    std::vector<tensor_info_t>                      m_sp_tensors_private;  // The array of tensors in the shape graph
    std::vector<std::array<uint64_t, nodeDataSize>> m_nodeData_private;

    std::vector<unsigned> m_invalidMaskArr;

    struct PPparams
    {
        ShapeManipulationOutputs shapeManOut;
        smf_t                    smfFunc;
    };

    struct NodeParams  // TODO: decrease sizes
    {
        SifParams               sifParams;
        SifOutputs              sifOutputs;
        uint16_t                invalidArrSize;
        ShapeManipulationParams smfParams;
        sif_t                   sifFunc;
        bool                    hasHostTensor;

        std::vector<tensor_shape_infer_info_t*>
            tensorSifInVec;  // TODO, make all in one array (so the vector is not allocated all over the place)
        std::vector<tensor_shape_infer_info_t*>
            tensorSifOutVec;  // TODO, make all in one array (so the vector is not allocated all over the place)
        std::vector<tensor_info_t*>
            tensorSmfInVec;  // TODO, make all in one array (so the vector is not allocated all over the place)
        std::vector<tensor_info_t*>
            tensorSmfOutVec;  // TODO, make all in one array (so the vector is not allocated all over the place)
        std::vector<PPparams> PPparamsVec;  // TODO: make all in one array
    };

    enum class DsdPatchingState
    {
        PRE_SIF,
        SIF_EXECUTED,
        SMF_EXECUTED = PRE_SIF
    };

    struct DsdPatchingStatusInfo
    {
        DsdPatchingState             patchingState;
        data_chunk_sm_patch_point_t* current_dc_pp_smf;
    };

    void _setDataChunkNodeParams(PPparams&                          nodePatchPointParams,
                                 const data_chunk_sm_patch_point_t& currDcSmPatchPoint,
                                 const std::vector<uint64_t>&       dataChunksHostAddresses);

    const DataChunkSmPatchPointsInfo* m_pDataChunkSmPatchPointsInfo;  // init in constructor

    // Below are copies of fields in the recipe that can change during the dynamic patching
    // DSD-patching steps with the following DBs;
    // 1) "Start fresh" - Gaudi:  Copy the m_originalPatchPoints into the local m_patchPoints
    //                    Gaudi3: Copy the data chunk DB from m_originalRecipeAddrPatcher to m_recipeAddrPatcher
    //
    // 2) Gaudi:  The patching operation updates that local m_patchPoints
    //    Gaudi3: The patching operation updates that local DC DB of m_recipeAddrPatcher
    //
    // 3) Gaudi:  The user of this class will get the DSD-patched copy (m_patchPoints) and use it for the Recipe patching
    //    Gaudi3: The user of this class will get the DSD-patched copy (m_recipeAddrPatcher) and use it for the Recipe patching

    // Gaudi
    std::vector<data_chunk_patch_point_t> m_patchPoints;
    const data_chunk_patch_point_t*       m_originalPatchPoints;
    // Gaudi3
    RecipeAddrPatcher                     m_recipeAddrPatcher;
    const RecipeAddrPatcher*              m_originalRecipeAddrPatcher;

    std::vector<NodeParams>  m_NodeParams;       // Holds pre-processed params for SIF/SMF
    const std::vector<bool>& m_isStaticTensors;  // bitmap which tensors are static
    ShapeFuncRegistry&       m_sfr;

    // Fuser
    std::vector<TensorShapeInfo*> m_fuserSifInTensors;
    std::vector<TensorShapeInfo*> m_fuserSifOutTensors;
    std::vector<tensor_info_t>    m_fuserNodeDbTensors;
    SifParams                     m_fuserSifParams;
    SifOutputs                    m_fuserSifOutputs;
    std::vector<unsigned>         m_fuserInvalidMask;

    DsdPatchingStatusInfo m_patchingStatus;

    std::mutex m_ownershipMutex;

    unsigned m_availableTpcEngines;

#ifdef EXTRA_CHECKING
    std::vector<bool> m_tensorSizeInferred;  // bitmap indicates which tensors has its size calculated
    std::vector<bool> m_fuserNodeTensorSizeInferred;
#endif
};
