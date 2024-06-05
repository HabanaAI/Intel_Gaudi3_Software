#pragma once

#include <vector>
#include "infra/defs.h"
#include "infra/gc_synapse_test.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "synapse_api.h"
#include "synapse_api_types.h"
#include "infra/syn_data_type_type_conversions.h"
#include "tensor_validator.inl"
#include "global_conf_test_setter.h"
#include "recipe.h"
#include "device_memory_allocator.h"
#include "utils/cpu_calculator.h"

#define DEFAULT_SIZES 4
#define EPSILON 0.1

using TestSizes          = std::array<unsigned, SYN_MAX_TENSOR_DIM>;
using TestNSizes         = std::array<unsigned, HABANA_DIM_MAX>;
using TestSizeVec        = std::vector<unsigned>;
using GraphAttributesVec = std::vector<synGraphAttribute>;

template<typename T>
using MaxDimVector = llvm_vecsmall::SmallVector<T, HABANA_DIM_MAX>;

template<typename T, typename U = TSize>
std::optional<MaxDimVector<U>> getVector(T* src, unsigned dims)
{
    return src ? MaxDimVector<U>(src, src + dims) : std::optional<MaxDimVector<U>>();
};

template<typename T>
static inline T* getVectorRawPtrOrNull(std::optional<MaxDimVector<T>>& vec)
{
    if (vec) return vec->data();
    return nullptr;
}

typedef struct synTensorDeviceFullLayoutExt
{
    TStride     strides[HABANA_DIM_MAX];
    synDataType deviceDataType;

    operator synTensorDeviceFullLayout() const
    {
        synTensorDeviceFullLayout ret;
        std::copy(strides, strides + HABANA_DIM_MAX, ret.strides);
        ret.deviceDataType = deviceDataType;
        return ret;
    }
} synTensorDeviceFullLayoutExt;

typedef struct synTensorDescriptorExt
{
public:
    synTensorDescriptorExt()
    : m_batchPos(INVALID_BATCH_POS),
      m_dataType(syn_type_na),
      m_dims(0),
      m_ptr(nullptr),
      m_isWeights(false),
      m_isQuantized(false),
      m_name(nullptr),
      m_isOutput(false),
      m_enablePerChannelQuant(false),
      m_isInput(false),
      m_isStatic(false),
      m_tensorType(DATA_TENSOR),
      m_isSparsityWeights(false)
    {
        memset(m_sizes, 0, SYN_MAX_TENSOR_DIM * sizeof(TSize));
        memset(m_strides, 0, SYN_MAX_TENSOR_DIM * sizeof(TStride));
        memset(m_minSizes, TENSOR_DEFAULT_MIN_SIZE, SYN_MAX_TENSOR_DIM * sizeof(TSize));
    }

    synTensorDescriptorExt(synDataType dataType,
                           unsigned    dims,
                           unsigned    sizes[SYN_MAX_TENSOR_DIM],
                           void*       ptr,
                           bool        isWeights             = false,
                           const char* name                  = nullptr,
                           unsigned    batchPos              = INVALID_BATCH_POS,
                           bool        isQuantized           = false,
                           bool        enablePerChannelQuant = false,
                           bool        setDynamicRange       = false,
                           double      DynamicRangeMin       = 2,
                           double      DynamicRangeMax       = 1,
                           bool        isSparsityWeights     = false)
    : m_batchPos(batchPos),
      m_dataType(dataType),
      m_dims(dims),
      m_ptr(ptr),
      m_isWeights(isWeights),
      m_isQuantized(isQuantized),
      m_name(name),
      m_isOutput(false),
      m_enablePerChannelQuant(enablePerChannelQuant),
      m_isInput(false),
      m_isStatic(false),
      m_tensorType(DATA_TENSOR),
      m_isSparsityWeights(isSparsityWeights)
    {
        memcpy(m_sizes, sizes, dims * sizeof(TSize));
        memset(m_strides, 0, SYN_MAX_TENSOR_DIM * sizeof(TStride));
        memset(m_minSizes, TENSOR_DEFAULT_MIN_SIZE, SYN_MAX_TENSOR_DIM * sizeof(TSize));
        m_dynamicRange.min   = DynamicRangeMin;
        m_dynamicRange.max   = DynamicRangeMax;
        m_dynamicRange.isSet = setDynamicRange;
    }

    operator synTensorDescriptor() const
    {
        auto sizesVec = getVector<const TSize, unsigned>(m_sizes, m_dims);
        auto ret      = synTensorDescriptor(m_dataType,
                                       m_dims,
                                       getVectorRawPtrOrNull(sizesVec),
                                       m_ptr,
                                       m_isWeights,
                                       m_name,
                                       m_batchPos,
                                       m_isQuantized,
                                       m_enablePerChannelQuant,
                                       m_dynamicRange.isSet,
                                       m_dynamicRange.min,
                                       m_dynamicRange.max,
                                       m_isSparsityWeights);
        std::copy(m_strides, m_strides + m_dims, ret.m_strides);
        std::copy(m_minSizes, m_minSizes + m_dims, ret.m_minSizes);
        return ret;
    }

    unsigned              m_batchPos;
    synDataType           m_dataType;
    unsigned              m_dims;
    TSize                 m_sizes[SYN_MAX_TENSOR_DIM];  // In dynamic tensors this is the maxSize
    synQuantizationParams m_quantizationParams[SYN_NUM_DATA_TYPES];
    void*                 m_ptr;
    bool                  m_isWeights;
    bool                  m_isQuantized;  // TODO 17465 - change name to reflect also fp16/bf16
    const char*           m_name;
    TStride               m_strides[SYN_MAX_TENSOR_DIM];
    bool                  m_isOutput;  // used by the GC to optimize memory usage of a segmented op when
    // flag is true, the tensor is marked as output tensor
    bool                            m_enablePerChannelQuant;  // Enable Per Channel Quantization for tensor
    synPerChannelQuantizationParams m_perChannelQuantizationParams[SYN_NUM_DATA_TYPES];
    bool                            m_isInput;
    bool                            m_isStatic;
    TSize                           m_minSizes[SYN_MAX_TENSOR_DIM];
    synTensorType                   m_tensorType;
    DynamicRange                    m_dynamicRange;
    bool                            m_isSparsityWeights;
} synTensorDescriptorExt;

class SynTrainingTestInfra : public SynTest
{
protected:
    static constexpr unsigned      INVALID_SECTION_IDX = -1;
    typedef  std::vector<unsigned> TensorIndices;

    SynTrainingTestInfra();
    ~SynTrainingTestInfra();

    virtual void SetUpTest() override;
    virtual void TearDownTest() override;

    void     createStreams();     // invoked by SetUp
    void     createEvents();      // invoked by SetUp
    unsigned duplicateGraph(unsigned origGraphIndex = 0);
    unsigned createGraph();
    unsigned createGraph(TestCompilationMode requestedMode);  // invoked by SetUp to create a default graph; returns the
                                                              // index of the graph in m_graphs

    unsigned getDuplicateTensorIndex(unsigned newGraphIndex, unsigned origTensorIndex);

    virtual TensorIndices createHugeTensors(unsigned        numTensors,
                                            TensorUsage     usage,
                                            bool            isPersistent    = false,
                                            const char*     name            = nullptr,
                                            MemInitType     initSelect      = MEM_INIT_ALL_ZERO,
                                            const float*    initializer     = nullptr,
                                            TSize*          sizes           = nullptr,
                                            unsigned        dims            = DEFAULT_SIZES,
                                            synDataType     dataType        = syn_type_single,
                                            TStride*        strides         = nullptr,
                                            unsigned        graphIndex      = 0,
                                            TSize           offsetInSection = 0,
                                            const unsigned* sectionIndex    = nullptr,
                                            bool            isConst         = false,
                                            TSize*          minSizes        = nullptr,
                                            synTensorType   tensorType      = synTensorType::DATA_TENSOR,
                                            synTensor       existingTensor  = nullptr);

    virtual TensorIndices createTensors(unsigned        numTensors,
                                        TensorUsage     usage,
                                        bool            isPersistent    = false,
                                        const char*     name            = nullptr,
                                        MemInitType     initSelect      = MEM_INIT_ALL_ZERO,
                                        const float*    initializer     = nullptr,
                                        unsigned*       sizes           = nullptr,
                                        unsigned        dims            = DEFAULT_SIZES,
                                        synDataType     dataType        = syn_type_single,
                                        unsigned*       strides         = nullptr,
                                        unsigned        graphIndex      = 0,
                                        unsigned        offsetInSection = 0,
                                        const unsigned* sectionIndex    = nullptr,
                                        bool            isConst         = false,
                                        unsigned*       minSizes        = nullptr,
                                        synTensorType   tensorType      = synTensorType::DATA_TENSOR,
                                        synTensor       existingTensor  = nullptr);

    unsigned createTensor(TensorUsage  usage,
                          MemInitType  initSelect  = MEM_INIT_ALL_ZERO,
                          const float* initializer = nullptr,
                          unsigned*    sizes       = nullptr,
                          unsigned     dims        = DEFAULT_SIZES,
                          synDataType  dataType    = syn_type_single,
                          unsigned*    strides     = nullptr,
                          unsigned*    minSizes    = nullptr,
                          unsigned     graphIndex  = 0);

    unsigned createPersistTensor(TensorUsage     usage,
                                 MemInitType     initSelect   = MEM_INIT_ALL_ZERO,
                                 const float*    initializer  = nullptr,
                                 unsigned*       sizes        = nullptr,
                                 unsigned        dims         = DEFAULT_SIZES,
                                 synDataType     dataType     = syn_type_single,
                                 unsigned*       strides      = nullptr,
                                 const char*     name         = nullptr,
                                 unsigned        graphIndex   = 0,
                                 unsigned        offsetInSection = 0,
                                 const unsigned* sectionIndex = nullptr,
                                 unsigned*       minSizes = nullptr);

    // Create static tensor (persistent tensor that is stored in const section).
    // If sectionIndex!=nullptr then it is assumed that this is a const section.
    unsigned createConstPersistTensor(TensorUsage     usage,
                                      MemInitType     initSelect      = MEM_INIT_ALL_ZERO,
                                      const float*    initializer     = nullptr,
                                      unsigned*       sizes           = nullptr,
                                      unsigned        dims            = DEFAULT_SIZES,
                                      synDataType     dataType        = syn_type_single,
                                      unsigned*       strides         = nullptr,
                                      const char*     name            = nullptr,
                                      unsigned        graphIndex      = 0,
                                      unsigned        offsetInSection = 0,
                                      const unsigned* sectionIndex    = nullptr);

    unsigned createConstTensor(MemInitType     initSelect   = MEM_INIT_ALL_ZERO,
                               const float*    initializer  = nullptr,
                               unsigned*       sizes        = nullptr,
                               unsigned        dims         = DEFAULT_SIZES,
                               synDataType     dataType     = syn_type_single,
                               unsigned*       strides      = nullptr,
                               const char*     name         = nullptr,
                               unsigned        graphIndex   = 0);

    unsigned createShapeTensor(TensorUsage  usage,
                               unsigned*    sizes,
                               unsigned*    minSizes,
                               unsigned     dims,
                               synDataType  dataType    = syn_type_uint32,
                               const char*  name        = nullptr,
                               unsigned     graphIndex  = 0);

    unsigned createHostShapeTensor(TensorUsage usage,
                                   unsigned*   sizes,
                                   unsigned*   data,
                                   const char* name       = nullptr,
                                   unsigned    graphIndex = 0);

    unsigned createHost2DeviceTensor(TensorUsage usage,
                                     unsigned*   sizes,
                                     unsigned*   data,
                                     unsigned    dims,
                                     const char* name       = nullptr,
                                     unsigned    graphIndex = 0);

    void changeTensorGeometry(unsigned     graphIndex,
                              size_t       tensorIndex,
                              TSize*       sizes,
                              unsigned     dims,
                              MemInitType  initSelect  = MEM_INIT_ALL_ZERO,
                              const float* initializer = nullptr);

    virtual unsigned createConstSection(uint64_t graphIndex = 0);
    bool             checkIfConstSection(synSectionHandle sectionHandle);
    bool             checkIfTensorInConstSection(synTensor tensor, unsigned graphIndex);
    void             initializeConstSection(const synRecipeHandle recipeHandle,
                                            const synSectionId    tensorSectionId,
                                            uint64_t&             deviceSectionBuffer);
    void             initializeAllConstSections(unsigned graphIndex);
    unsigned         createHost2DeviceIntermediateTensor(TensorUsage usage,
                                                         unsigned*   sizes,
                                                         unsigned    dims,
                                                         const char* name       = nullptr,
                                                         unsigned    graphIndex = 0);

    unsigned connectOutputTensorToInputTensor(unsigned outputTensorIndex);

    virtual void addNodeToGraph(const char*  guid,
                                void*        userParams    = nullptr,
                                unsigned     paramSize     = 0,
                                const char*  nodeName      = nullptr,
                                unsigned     graphIndex    = 0,
                                const char** inputLayouts  = nullptr,
                                const char** outputLayouts = nullptr);

    virtual void addNodeToGraph(const char*          guid,
                                const TensorIndices& inputTensorIndices,   // Indices of m_inTensors
                                const TensorIndices& outputTensorIndices,  // Indices of m_outTensors
                                void*                userParams    = nullptr,
                                unsigned             paramSize     = 0,
                                const char*          nodeName      = nullptr,
                                unsigned             graphIndex    = 0,
                                synNodeId*           nodeId        = nullptr,
                                const char**         inputLayouts  = nullptr,
                                const char**         outputLayouts = nullptr);

    virtual void setNodeDependency(const synNodeId*       pBlockingNodesIdList,
                                   const synNodeId*       pBlockedNodesIdList,
                                   const uint32_t         numberblocking,
                                   const uint32_t         numberblocked,
                                   unsigned               graphIndex = 0);

    virtual void compileAndRun(unsigned graphIndex);
    virtual void compileAndRun();

    void compileTopology(const std::string& topologyName = "", unsigned graphIndex = 0);

    void runTopology(unsigned  graphIndex            = 0,
                     bool      initPersistentOutputs = false,
                     synStatus expectedLaunch        = synSuccess,
                     synStatus expectedStreamSync    = synSuccess);

    void copyOutputTensors(unsigned graphIndex);

    void copyInputTensors(unsigned                             graphIndex,
                          uint64_t&                            programAddress,
                          std::vector<synLaunchTensorInfoExt>& concatTensors,
                          bool                                 initPersistentOutputs);

    void resize(unsigned newSize);

    std::vector<unsigned> getGraphTensorIndices(unsigned graphIndex);

    void randomBufferValues(MemInitType initSelect, synDataType type, uint64_t size, void* output);
    void initBufferValues(MemInitType  initSelect,
                          const float* initializer,
                          synDataType  dataType,
                          uint64_t     numElements,
                          uint64_t     memorySize,
                          void*        output);

    synTensorDescriptorExt getTensorDescriptor(synDataType   dataType,
                                               const TSize*  tensorSizes,
                                               unsigned      dims,
                                               const char*   name,
                                               TStride*      strides,
                                               void*         ptr,
                                               bool          isQuantized,
                                               TSize*        minSizes   = nullptr,
                                               synTensorType tensorType = synTensorType::DATA_TENSOR);

    void setActualSizes(unsigned tensorIndex, const TSize* tensorSizes, unsigned graphIndex = 0);
    void setActualSizes(unsigned tensorIndex, const unsigned* tensorSizes, unsigned graphIndex = 0);
    void setActualSizes(unsigned tensorIndex, const TestSizeVec& tensorSizes, unsigned graphIndex = 0);
    void setRuntimeHostBuffer(unsigned tensorIndex, void* buffer, unsigned graphIndex = 0);
    void setAsInternalShapeTensor(unsigned tensorIndex, unsigned graphIndex = 0);
    void setActualScalarParametersData(unsigned tensorIndex, void* data, unsigned dataSize, unsigned graphIndex = 0);

    bool _replaceConstSectionTensorDramAddr(std::vector<synLaunchTensorInfoExt>& concatTensors);

    const char* getUniqueTensorName(TensorUsage usage);

    synTensor getTensorByIndex(size_t index) {return m_tensors[index];}

    static uint64_t getNumberOfElements(const unsigned* sizes, unsigned dims = DEFAULT_SIZES);
    static uint64_t getNumberOfElements(const TSize* sizes, unsigned dims = DEFAULT_SIZES);
    static uint64_t
    getMemorySize(const unsigned* sizes, const unsigned* strides, synDataType type, unsigned dims = DEFAULT_SIZES);
    static uint64_t getMemorySize(const unsigned* sizes, synDataType type, unsigned dims = DEFAULT_SIZES);
    static uint64_t
    getMemorySize(const TSize* sizes, const TStride* strides, synDataType type, unsigned dims = DEFAULT_SIZES);
    static uint64_t         getMemorySize(const TSize* sizes, synDataType type, unsigned dims = DEFAULT_SIZES);
    static uint64_t         getDefaultNumberOfElements();
    static const TSize*  getDefaultSizes();
    static const unsigned* getDefaultSizes32b();

    synRecipeHandle getRecipeHandle(unsigned graphIndex = 0);

    struct TensorCreationParams
    {
        unsigned    tensorIndex;
        TensorUsage usage;
        TSize    offset;
        unsigned    concreteSectionIndex;
        bool        isPersistent;
        bool        isConst;
        bool        isDynamicShape;
        bool        hasStridesSet;
    };
    struct GraphData
    {
        synGraphHandle                                           graphHandle;
        synRecipeHandle                                          recipeHandle;
        std::string                                              recipeName;
        uint64_t                                                 workspaceSize;
        uint64_t                                                 hbmAddr;
        uint64_t                                                 numNodes;
        std::unordered_set<synNodeId>                            nodesById;
        std::unordered_map<synTensor, TensorCreationParams>      tensorCreationParams;
        std::vector<std::pair<unsigned, synLaunchTensorInfoExt>> m_inputEnqueueTensorsExt;
        std::vector<std::pair<unsigned, synLaunchTensorInfoExt>> m_outputEnqueueTensorsExt;
        std::vector<unsigned>                                    m_tensorsMissingActualSize;
        std::vector<std::pair<unsigned, synLaunchTensorInfoExt>> m_nonDeviceShapeTensorsExt;
        std::unordered_map<unsigned, unsigned>                   m_origToDuplicateTensorIndexMap;
    };

    struct ConstSectionTensorInfo
    {
        std::string tensorName = "";
        uint64_t    deviceAddress;  // in the const section
    };

    // safe reinterpret_cast
    template<typename T>
    T* castHostBuffer(size_t tensorIndex);

    template<typename T>
    T* castHostInBuffer(size_t tensorIndex);

    template<typename T>
    T* castHostOutBuffer(size_t tensorIndex);

    uint64_t getTensorElementCount(size_t tensorIndex)
    {
        return getNumberOfElements(m_tensorDescs[tensorIndex].m_sizes, m_tensorDescs[tensorIndex].m_dims);
    }

    synTensorDescriptorExt getTensorDescriptor(size_t tensorIndex)
    {
        return m_tensorDescs[tensorIndex];
    }

    void setConstSectionTensorInfo(ConstSectionTensorInfo info) { m_constSectionTensorInfoVec.push_back(info); }

    /**
     * @brief Validate graph input tensor usage is set correctly
     */
    bool validateInputTensorUsage(const GraphData& graphData) const;

    static const unsigned               m_maxTensorDims    = DEFAULT_SIZES;
    unsigned                            m_maxNumTensors    = 0;
    unsigned                            m_sequentialNumber = 0;
    unsigned                            m_deviceId;
    std::vector<void*>                  m_hostBuffers;
    std::vector<void*>                  m_runtimeHostBuffers;
    std::vector<uint64_t>               m_deviceBuffers;
    std::vector<synTensor>              m_tensors;
    std::vector<synTensorDescriptorExt>    m_tensorDescs;
    synStreamHandle                     m_streamHandleDownload;
    synStreamHandle                     m_streamHandleCompute;
    synStreamHandle                     m_streamHandleUpload;
    synEventHandle                      m_eventHandle;
    bool                                m_isEventCreated;
    bool                                m_tensorInitiatedInCompilationOnlyMode = false;
    std::vector<GraphData>              m_graphs;
    std::vector<ConstSectionTensorInfo> m_constSectionTensorInfoVec;

    uint64_t m_lastLaunchElapsedTime = std::numeric_limits<uint64_t>::max();

    struct SectionsData
    {
        synSectionHandle handle;
        uint64_t         baseAddress;
        uint64_t         size;
        TrivialDeviceMemoryAllocatorPtr allocator;
    };

    synSectionHandle createSection(synGraphHandle graphIndex);
    unsigned createSection(uint64_t size, uint64_t graphIndex = 0);
    unsigned createSection(uint64_t size, synGraphHandle graphHandle);
    unsigned createNonPersistentSection(uint64_t graphIndex = 0, bool isRmw = true);
    void     updateSectionAllocator(unsigned sectionIdx, uint64_t size);

    virtual void pushGlobalConf(const std::string& name, const std::string& value);

    void downloadDataToDevice(const uint64_t src, const uint64_t size, const uint64_t dst);

    std::vector<SectionsData>                        m_persistentSections;
    std::vector<synSectionHandle>                    m_nonPersistentSections;
    std::list<std::string>                           m_uniqueNamesHolder;
    std::list<std::shared_ptr<GlobalConfTestSetter>> m_globalConfStack;

    std::default_random_engine m_generator;  // NOLINT(cert-msc32-c,cert-msc51-cpp) - deterministic on purpose

    void init();
    void reset();

    GraphData& getGraph(unsigned graphIndex);

    bool compareRecipes(const recipe_t& recipe1, const recipe_t& recipe2, bool compareNames = false);

    void getTensorSectionId(const synRecipeHandle& recipeHandle, const synTensor& tensor, synSectionId& sectionId);
    /*DEPRECATED*/
    void setGraphAttributes(synGraphAttribute* attributes, uint64_t* values, uint32_t size, unsigned graphIndex = 0);
    void setGraphAttributesV2(synGraphAttribute* attributes, synGraphAttributeVal* values, uint32_t size, unsigned graphIndex = 0);
    void setGraphInferenceMode(unsigned graphIndex = 0);
    void setGraphInferenceModeAndQuantizationEnabled(unsigned graphIndex = 0);
    void setTensorQuantizationData(size_t tensorIndex, synQuantizationProperty prop, void* propVal, uint64_t propSize);
    bool compareFP8Results(const float* refOutput,
                           const float* actOutput,
                           unsigned     numElems,
                           synDataType  dType,
                           std::string& errMsg);
};

void validateResults(const synTensorDescriptor& desc, char* firstData, char* secondData);

template<typename T>
T* SynTrainingTestInfra::castHostBuffer(size_t tensorIndex)
{
    HB_ASSERT(asSynType<T>() == m_tensorDescs[tensorIndex].m_dataType, "inconsistency between tensor initialization and buffer's type cast");
    HB_ASSERT_PTR(m_hostBuffers[tensorIndex]);
    return reinterpret_cast<T*>(m_hostBuffers[tensorIndex]);
}

template<typename T>
T* SynTrainingTestInfra::castHostInBuffer(size_t tensorIndex)
{
    return castHostBuffer<T>(tensorIndex);
}

template<typename T>
T* SynTrainingTestInfra::castHostOutBuffer(size_t tensorIndex)
{
    return castHostBuffer<T>(tensorIndex);
}

class SynGaudiTestInfra : public SynTrainingTestInfra
{
};

class SynTrainingTpcTestInfra : public SynTrainingTestInfra
{
};
