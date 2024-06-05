#pragma once

#include "base_test.h"
#include "synapse_api_types.h"
#include <json.hpp>
#include <vector>
#include <bitset>
#include <string>
#include <unordered_map>
#include <optional>

namespace json_tests
{
constexpr uint32_t INVALID_INTERNAL_TENSOR_INDEX  = -1;
constexpr uint32_t INVALID_INTERNAL_SECTION_INDEX = -1;

enum StatisticsCollection
{
    GRAPH_CREATION,
    NODE_CREATION,
    TENOSR_CREATION,
    SECTION_CREATION,
    GRAPH_COMPILE,
    GRAPH_DUPLICATE,
    INFER_SHAPES,
    TENOSR_SET_GEOMETRY,
    TENOSR_SET_DEVICE_LAYOUT,
    TENOSR_SET_HOST_PTR,
    TENOSR_SET_PERMUTATION,
    TENOSR_SET_EXTERNAL,
    TENOSR_SET_ALLOW_PERMUTATION,
    TENOSR_ASSIGN_TO_SECTION,
    SECTION_SET_PERSISTENT,
    SECTION_SET_RMW,
    NODE_SET_ROUNDING_MODE,
    NODE_DEPENDENCY_SET,
    GRAPH_DESTRUCTION,
    TENOSR_DESTRUCTION,
    SECTION_DESTRUCTION,
    WORKSPACE_GET_SIZE,
    TENSOR_RETRIEVE_LAUNCH_INFO_BY_ID_EXT,
    LAUNCH_WITH_EXTERNAL_EVENTS_EXT,
    // keep last
    STATISTICS_COLLECTION_LAST
};

constexpr enum StatisticsCollection getStatisticsCollectionVal(std::string_view functionName)
{
    if (functionName == "synGraphCreate") return GRAPH_CREATION;
    if (functionName == "synGraphCreateEager") return GRAPH_CREATION;
    if (functionName == "synNodeCreateWithId") return NODE_CREATION;
    if (functionName == "synTensorHandleCreate") return TENOSR_CREATION;
    if (functionName == "synSectionCreate") return SECTION_CREATION;
    if (functionName == "synGraphCompile") return GRAPH_COMPILE;
    if (functionName == "synTensorSetGeometry") return TENOSR_SET_GEOMETRY;
    if (functionName == "synTensorSetDeviceFullLayout") return TENOSR_SET_DEVICE_LAYOUT;
    if (functionName == "synTensorSetHostPtr") return TENOSR_SET_HOST_PTR;
    if (functionName == "synTensorSetPermutation") return TENOSR_SET_PERMUTATION;
    if (functionName == "synTensorSetExternal") return TENOSR_SET_EXTERNAL;
    if (functionName == "synTensorSetAllowPermutation") return TENOSR_SET_ALLOW_PERMUTATION;
    if (functionName == "synTensorAssignToSection") return TENOSR_ASSIGN_TO_SECTION;
    if (functionName == "synSectionSetPersistent") return SECTION_SET_PERSISTENT;
    if (functionName == "synSectionSetRMW") return SECTION_SET_RMW;
    if (functionName == "synNodeSetRoundingMode") return NODE_SET_ROUNDING_MODE;
    if (functionName == "synNodeDependencySet") return NODE_DEPENDENCY_SET;
    if (functionName == "synGraphDestroy") return GRAPH_DESTRUCTION;
    if (functionName == "synTensorDestroy") return TENOSR_DESTRUCTION;
    if (functionName == "synSectionDestroy") return SECTION_DESTRUCTION;
    if (functionName == "synGraphDuplicate") return GRAPH_DUPLICATE;
    if (functionName == "synGraphInferShapes") return INFER_SHAPES;
    if (functionName == "synWorkspaceGetSize") return WORKSPACE_GET_SIZE;
    if (functionName == "synTensorRetrieveLaunchInfoByIdExt") return TENSOR_RETRIEVE_LAUNCH_INFO_BY_ID_EXT;
    if (functionName == "synLaunchWithExternalEventsExt") return LAUNCH_WITH_EXTERNAL_EVENTS_EXT;
    return STATISTICS_COLLECTION_LAST;
}

constexpr std::string_view getSynAPIFunctionName(enum StatisticsCollection type, bool isEager)
{
    switch (type)
    {
        case GRAPH_CREATION:
            return isEager ? "synGraphCreateEager" : "synGraphCreate";
        case NODE_CREATION:
            return "synNodeCreateWithId";
        case TENOSR_CREATION:
            return "synTensorHandleCreate";
        case SECTION_CREATION:
            return "synSectionCreate";
        case GRAPH_COMPILE:
            return "synGraphCompile";
        case TENOSR_SET_GEOMETRY:
            return "synTensorSetGeometry";
        case TENOSR_SET_DEVICE_LAYOUT:
            return "synTensorSetDeviceFullLayout";
        case TENOSR_SET_HOST_PTR:
            return "synTensorSetHostPtr";
        case TENOSR_SET_PERMUTATION:
            return "synTensorSetPermutation";
        case TENOSR_SET_EXTERNAL:
            return "synTensorSetExternal";
        case TENOSR_SET_ALLOW_PERMUTATION:
            return "synTensorSetAllowPermutation";
        case TENOSR_ASSIGN_TO_SECTION:
            return "synTensorAssignToSection";
        case SECTION_SET_PERSISTENT:
            return "synSectionSetPersistent";
        case SECTION_SET_RMW:
            return "synSectionSetRMW";
        case NODE_SET_ROUNDING_MODE:
            return "synNodeSetRoundingMode";
        case NODE_DEPENDENCY_SET:
            return "synNodeDependencySet";
        case GRAPH_DESTRUCTION:
            return "synGraphDestroy";
        case TENOSR_DESTRUCTION:
            return "synTensorDestroy";
        case SECTION_DESTRUCTION:
            return "synSectionDestroy";
        case GRAPH_DUPLICATE:
            return "synGraphDuplicate";
        case INFER_SHAPES:
            return "synGraphInferShapes";
        case WORKSPACE_GET_SIZE:
            return "synWorkspaceGetSize";
        case TENSOR_RETRIEVE_LAUNCH_INFO_BY_ID_EXT:
            return "synTensorRetrieveLaunchInfoByIdExt";
        case LAUNCH_WITH_EXTERNAL_EVENTS_EXT:
            return "synLaunchWithExternalEventsExt";
        default:
            return "";
    }
}
struct SectionCreationParams
{
    bool isPersistent;
    bool rmwSection;
};

struct TensorCreationParams
{
    std::string               name;
    uint64_t                  userMemOffset;
    synTensorType             tensorType;
    bool                      isConst;
    bool                      isExternal;
    bool                      isPersistent;
    bool                      allowPermutation;
    uint32_t                  internalSectionIndex;
    std::vector<uint8_t>      data;
    synTensorGeometry         geometry[synGeometryDims];
    synTensorDeviceFullLayout layout;
    synTensorPermutation      permutation;
};

struct NodeCreationParams
{
    std::string                    name;
    std::string                    guid;
    std::vector<uint8_t>           params;
    std::optional<synRoundingMode> roundingMode;
    std::vector<uint32_t>          inputTensorInternalIndices;
    std::vector<uint32_t>          outputTensorInternalIndices;
    std::vector<std::string>       inputLayouts;
    std::vector<std::string>       outputLayouts;
    std::vector<std::string>       blockingNodeNames;
    std::vector<uint32_t>          blockingNodeInternalIndices;
    std::vector<synTensor>         inputTensors;
    std::vector<synTensor>         outputTensors;
    std::vector<const char*>       cstringInputLayouts;
    std::vector<const char*>       cstringOutputLayouts;
    std::vector<synNodeId>         blockingNodes;
};

struct GraphParams
{
    size_t                                    idx;
    std::string                               name;
    size_t                                    maxSectionIdx;
    std::vector<synTensor>                    tensors;
    std::vector<synSectionHandle>             sections;
    std::vector<synNodeId>                    nodes;
    std::vector<TensorCreationParams>         tensorsParams;
    std::vector<NodeCreationParams>           nodesParams;
    std::vector<SectionCreationParams>        sectionsParams;
    std::unordered_map<synTensor, uint32_t>   tensorHandleToInternalIndex;
    std::unordered_map<std::string, uint32_t> tensorNameToInternalIndex;
    std::unordered_map<std::string, uint32_t> nodeNameToInternalIndex;
    std::unordered_set<std::string>           inputTensors;
    std::vector<synTensorHandleMap>           tensorHandleMappingVec;
    std::vector<synNodeHandleMap>             nodeHandleMappingVec;
};

struct RecipeParams
{
    std::vector<synRetrievedLaunchTensorInfoExt> retrievedLaunchTensorInfo;
    std::vector<synLaunchTensorInfoExt>          launchTensorInfo;
    std::vector<std::optional<uint64_t>>         sectionAddresses;
};

class SimpleAllocator
{
public:
    SimpleAllocator(std::string_view name) : m_name(name) {};
    void  setBuffer(void* buffer, size_t capacity, size_t alignment);
    void  resetPosition() { m_nextAddress = 0; }
    void  resetBuffer();
    void* allocate(uint64_t sizeInBytes);
    void* getBuffer() const { return m_buffer; }

private:
    std::string m_name;
    void*       m_buffer      = nullptr;
    size_t      m_nextAddress = 0;
    size_t      m_capacity    = 0;
    size_t      m_alignment   = 128;
};

class PerformanceBaseTest : public JsonTest
{
public:
    PerformanceBaseTest(const ArgParser& args, bool run, bool enableAPIStats = true, bool eventPerLaunch = false);
    virtual ~PerformanceBaseTest();
    virtual void run() = 0;

protected:
    // json parsing into GraphParams
    static void fillTensorAndSectionCreationParams(const nlohmann_hcl::json& jsonGraph, GraphParams& graphParams);
    static void fillNodeCreationParams(const nlohmann_hcl::json& jsonGraph, GraphParams& graphParams);
    static void fillRecipeParams(const GraphParams& graphParams, RecipeParams& recipeParams);

    // synapse graph creation API
    synGraphHandle createGraph();
    synGraphHandle duplicateGraph(synGraphHandle graph, GraphParams& graphParams);
    void           createSections(synGraphHandle graph, GraphParams& graphParams);
    void           createTensors(synGraphHandle graph, GraphParams& graphParams);
    void           createNodes(synGraphHandle graph, GraphParams& graphParams);
    void           setBlockingNodes(synGraphHandle graph, GraphParams& graphParams);

    // synapse graph compilation API
    synRecipeHandle compileGraph(synGraphHandle graph, const std::string& recipeName);

    // synapse recipe execution API
    void executeRecipe(synRecipeHandle recipe, RecipeParams& recipeParams, bool waitForCompletion = true);
    void waitForLaunchCompletion();

    // synapse graph cleanup API
    void cleanup(synGraphHandle*  graph,
                 synGraphHandle*  duplicateGraph,
                 synRecipeHandle* recipe,
                 GraphParams&     graphParams,
                 RecipeParams&    recipeParams);

    // json stats
    void recordJsonStats(const size_t index);
    void resetStats();
    void dumpStats() const;

private:
    void setup();

    // calculate maximal required workspace size and persistent tensors size
    // across all json graphs.
    void pickAllocationSize();

    // synapse configuration API
    void setConfiguration();

    // synapse singleton API
    void initializeSynapseInstance();
    void cleanupSynapseInstance();

    // setup\release device resources required for execution
    void acquireDeviceResources();
    void releaseDeviceResources();

protected:
    bool                                             m_run                    = false;
    bool                                             m_quietMode              = true;
    bool                                             m_keepGoing              = true;
    bool                                             m_eagerMode              = true;
    bool                                             m_eventPerLaunch         = false;
    int                                              m_testIterations         = 0;
    unsigned                                         m_nextLaunchEventId      = 0;
    unsigned                                         m_nextsynchronizeEventId = 0;
    std::bitset<STATISTICS_COLLECTION_LAST>          m_enabled_statistics     = {};
    std::array<uint64_t, STATISTICS_COLLECTION_LAST> m_statistics             = {};
    uint64_t                                         m_totalHostTime          = 0;
    synDeviceType                                    m_deviceType;
    nlohmann_hcl::json                               m_stats;
    std::string                                      m_statsFilePath;
    synDeviceId                                      m_deviceId;
    std::vector<synEventHandle>                      m_eventHandles;
    synStreamHandle                                  m_uploadStreamHandle;
    synStreamHandle                                  m_downloadStreamHandle;
    synStreamHandle                                  m_computeStreamHandle;
    SimpleAllocator                                  m_deviceBufferAllocator;
    size_t                                           m_deviceBufferCapacity = 0;
    uint64_t                                         m_deviceAlignment      = 0;
    std::vector<uint64_t>                            m_passingGraphsIndices;
};
}  // namespace json_tests