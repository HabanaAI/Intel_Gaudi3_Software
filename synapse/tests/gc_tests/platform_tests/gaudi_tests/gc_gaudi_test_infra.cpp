#include "gc_gaudi_test_infra.h"

#include "data_types/fp8.h"
#include "infra/gc_test_configuration.h"
#include "infra/recipe/recipe_compare.hpp"
#include "log_manager.h"
#include "scoped_configuration_change.h"
#include "syn_singleton.hpp"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "test_utils.h"

#include <algorithm>
#include <stdexcept>
#include <stdint.h>

SynTrainingTestInfra::SynTrainingTestInfra()
{
    if (m_deviceType == synDeviceTypeInvalid)
    {
        LOG_WARN(SYN_TEST,
                 "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
        m_deviceType = synDeviceGaudi;
    }
    m_isEventCreated = false;

    m_streamHandleDownload = nullptr;
    m_streamHandleCompute  = nullptr;
    m_streamHandleUpload   = nullptr;
    m_eventHandle          = nullptr;

    setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
}

SynTrainingTestInfra::~SynTrainingTestInfra() {}

void SynTrainingTestInfra::SetUpTest()
{
    SynTest::SetUpTest();

    if (shouldRunTest())
    {
        init();
    }
}

void SynTrainingTestInfra::TearDownTest()
{
    reset();

    SynTest::TearDownTest();
}

void SynTrainingTestInfra::createStreams()
{
    synStatus status = synStreamCreateGeneric(&m_streamHandleDownload, m_deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Download (host->device copy) stream";

    status = synStreamCreateGeneric(&m_streamHandleUpload, m_deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Upload (device->host copy) stream";

    status = synStreamCreateGeneric(&m_streamHandleCompute, m_deviceId, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to create Compute stream";
}

void SynTrainingTestInfra::createEvents()
{
    ASSERT_EQ(synEventCreate(&m_eventHandle, m_deviceId, 0), synSuccess) << "Failed to create event";
    m_isEventCreated = true;
}

unsigned SynTrainingTestInfra::createGraph()
{
    return createGraph(m_testConfig.m_compilationMode);
}

unsigned SynTrainingTestInfra::duplicateGraph(unsigned origGraphIndex)
{
    unsigned                        numTensors = m_graphs[origGraphIndex].tensorCreationParams.size();
    unsigned                        numNodes   = m_graphs[origGraphIndex].numNodes;
    std::vector<synTensorHandleMap> tensorsMap(numTensors);
    std::vector<synNodeHandleMap>   nodesMap(numNodes);
    GraphData                       newGraphData  = {};
    auto                            newGraphIndex = m_graphs.size();

    EXPECT_EQ(synSuccess,
              synGraphDuplicate(m_graphs[origGraphIndex].graphHandle,
                                &newGraphData.graphHandle,
                                tensorsMap.data(),
                                &numTensors,
                                nodesMap.data(),
                                &numNodes))
        << "Failed to create duplicate graph for graph index " << origGraphIndex;
    newGraphData.numNodes = numNodes;
    // add the new graph to the test infra tracked graphs
    m_graphs.push_back(newGraphData);

    bool canVerifyNodes = (m_graphs[origGraphIndex].numNodes == m_graphs[origGraphIndex].nodesById.size());
    for (const auto& nodesPair : nodesMap)
    {
        if (canVerifyNodes)
        {
            EXPECT_NE(m_graphs[origGraphIndex].nodesById.find(nodesPair.origHandle),
                      m_graphs[origGraphIndex].nodesById.end())
                << " original node id is not found in the Graph";
        }
        m_graphs[newGraphIndex].nodesById.emplace(nodesPair.newHandle);
    }

    for (const auto& tensorsPair : tensorsMap)
    {
        const auto origTensorIter = m_graphs[origGraphIndex].tensorCreationParams.find(tensorsPair.origHandle);
        EXPECT_NE(origTensorIter, m_graphs[origGraphIndex].tensorCreationParams.end())
            << " original tensor id is not found in the Graph";
        const auto& originalTensorCreationParams = origTensorIter->second;
        auto        origTensorIndex              = originalTensorCreationParams.tensorIndex;
        char        newTensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
        synTensorGetName(tensorsPair.newHandle, ENQUEUE_TENSOR_NAME_MAX_SIZE, newTensorName);
        // make a copy of the original tensor descriptor as the corresponding vector might grow and get re-allcoated
        auto origTensorDescriptor = m_tensorDescs[origTensorIndex];
        // add the new tensor to the test infra tracked tensors
        auto newlycreatedTensorIndices = createHugeTensors(
            1,
            originalTensorCreationParams.usage,
            originalTensorCreationParams.isPersistent,
            newTensorName,
            m_tensorInitiatedInCompilationOnlyMode ? MEM_INIT_COMPILATION_ONLY : MEM_INIT_FROM_INITIALIZER_NO_CAST,
            static_cast<const float*>(m_hostBuffers[origTensorIndex]),
            origTensorDescriptor.m_sizes,
            origTensorDescriptor.m_dims,
            origTensorDescriptor.m_dataType,
            originalTensorCreationParams.hasStridesSet ? origTensorDescriptor.m_strides : nullptr,
            newGraphIndex,
            originalTensorCreationParams.offset,
            originalTensorCreationParams.concreteSectionIndex != INVALID_SECTION_IDX
                ? &originalTensorCreationParams.concreteSectionIndex
                : nullptr,
            originalTensorCreationParams.isConst,
            originalTensorCreationParams.isDynamicShape ? origTensorDescriptor.m_minSizes : nullptr,
            origTensorDescriptor.m_tensorType,
            tensorsPair.newHandle);
        // add a mapping from original tensor to duplicated one
        m_graphs[newGraphIndex].m_origToDuplicateTensorIndexMap[origTensorIndex] = newlycreatedTensorIndices.front();
    }

    return newGraphIndex;
}

unsigned SynTrainingTestInfra::getDuplicateTensorIndex(unsigned newGraphIndex, unsigned origTensorIndex)
{
    return m_graphs[newGraphIndex].m_origToDuplicateTensorIndexMap[origTensorIndex];
}

unsigned SynTrainingTestInfra::createGraph(TestCompilationMode requestedMode)
{
    GraphData graphData = {};
    if (requestedMode == COMP_EAGER_MODE_TEST)
    {
        EXPECT_EQ(synSuccess, synGraphCreateEager(&graphData.graphHandle, m_deviceType))
            << "Failed to create eager " << toString(m_deviceType) << " graph";
    }
    else
    {
        EXPECT_EQ(synSuccess, synGraphCreate(&graphData.graphHandle, m_deviceType))
            << "Failed to create " << toString(m_deviceType) << " graph";
    }
    m_graphs.push_back(graphData);

    // initialize random seed for tensor random values
    m_generator = std::default_random_engine();  // NOLINT(cert-msc32-c,cert-msc51-cpp) - deterministic on purpose

    return m_graphs.size() - 1;
}

void SynTrainingTestInfra::resize(unsigned newSize)
{
    if (newSize <= m_maxNumTensors) return;  // allow only to grow, never to shrink

    m_hostBuffers.resize(newSize, nullptr);
    m_runtimeHostBuffers.resize(newSize, nullptr);
    m_deviceBuffers.resize(newSize, 0);
    m_tensors.resize(newSize, nullptr);
    m_tensorDescs.resize(newSize);

    m_maxNumTensors = newSize;
}

void SynTrainingTestInfra::randomBufferValues(MemInitType initSelect, synDataType type, uint64_t size, void* output)
{
    static const float                   maxRand     = 2.0f;   // divide by 10 to avoid overflow
    static const float                   minRand     = -2.0f;  // divide by 10 to avoid overflow
    static const std::pair<float, float> randomRange = {minRand, maxRand};
    static const std::pair<float, float> randomRangeNeg = {minRand, minRand/2.0f};
    static const std::pair<float, float> nonNegRange = {0, maxRand};

    static const float                   maxRandInt     = 120.0f;   // divide by 10 to avoid overflow
    static const float                   minRandInt     = -120.0f;  // divide by 10 to avoid overflow
    static const std::pair<float, float> randomRangeInt = {minRandInt, maxRandInt};
    static const std::pair<float, float> randomRangeNegInt = {minRandInt, minRandInt/2.0f};
    static const std::pair<float, float> nonNegRangeInt = {0, maxRandInt};

    assert(initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE || initSelect == MEM_INIT_RANDOM_POSITIVE || initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY);
    std::pair<float, float> range =
        (initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY)? randomRangeNeg :
        (initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE)? randomRange : nonNegRange;
    std::pair<float, float> rangeInt =
        (initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE_ONLY)? randomRangeNegInt :
        (initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE)? randomRangeInt : nonNegRangeInt;
    switch (type)
    {
        case syn_type_float:
            fillWithRandom<float>(m_generator, (float*)output, size, range);
            break;
        case syn_type_bf16:
            fillWithRandom(m_generator, (bfloat16*)output, size, range);
            break;
        case syn_type_fp16:
            fillWithRandom(m_generator, (fp16_t*)output, size, range);
            break;
        case syn_type_int8:
            fillWithRandom(m_generator, (int8_t*)output, size, rangeInt);
            break;
        case syn_type_uint8:
            fillWithRandom(m_generator, (uint8_t*)output, size, rangeInt);
            break;
        case syn_type_int16:
            fillWithRandom(m_generator, (int16_t*)output, size, rangeInt);
            break;
        case syn_type_uint16:
            fillWithRandom(m_generator, (uint16_t*)output, size, rangeInt);
            break;
        case syn_type_int32:
            fillWithRandom(m_generator, (int32_t*)output, size, rangeInt);
            break;
        case syn_type_uint32:
            fillWithRandom(m_generator, (uint32_t*)output, size, rangeInt);
            break;
        case syn_type_int64:
            fillWithRandom(m_generator, (int64_t*)output, size, rangeInt);
            break;
        case syn_type_uint64:
            fillWithRandom(m_generator, (uint64_t*)output, size, rangeInt);
            break;
        case syn_type_fp8_152:
            fillWithRandom(m_generator, (fp8_152_t*)output, size, range);
            break;
        default:
            assert(0 && "Unsupported data type for Gaudi test");
    }
}

void SynTrainingTestInfra::initBufferValues(MemInitType  initSelect,
                                            const float* initializer,
                                            synDataType  dataType,
                                            uint64_t     numElements,
                                            uint64_t     memorySize,
                                            void*        output)
{
    if (initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE || initSelect == MEM_INIT_RANDOM_POSITIVE)
    {
        randomBufferValues(initSelect, dataType, numElements, output);
    }
    else if (initSelect == MEM_INIT_FROM_INITIALIZER || initSelect == MEM_INIT_FROM_INITIALIZER_NO_CAST)
    {
        HB_ASSERT_PTR(initializer);

        if (dataType == syn_type_bf16 && initSelect != MEM_INIT_FROM_INITIALIZER_NO_CAST)
        // Note, only half of the buffer is initialized and used
        {
            uint16_t* bf16Data = static_cast<uint16_t*>(output);
            for (uint64_t j = 0; j < numElements; j++)
            {
                bf16Data[j] = Bfloat16(initializer[j]);
            }
        }
        else if (dataType == syn_type_fp8_152 && initSelect != MEM_INIT_FROM_INITIALIZER_NO_CAST)
        // Note, only quarter of the buffer is initialized and used
        {
            fp8_152_t* fp8Data = static_cast<fp8_152_t*>(output);
            for (uint64_t j = 0; j < numElements; j++)
            {
                fp8Data[j] = fp8_152_t(initializer[j]);
            }
        }
        else if (dataType == syn_type_fp8_143 && initSelect != MEM_INIT_FROM_INITIALIZER_NO_CAST)
        // Note, only quarter of the buffer is initialized and used
        {
            fp8_143_t* fp8Data = static_cast<fp8_143_t*>(output);
            for (uint64_t j = 0; j < numElements; j++)
            {
                fp8Data[j] = fp8_143_t(initializer[j]);
            }
        }
        else
        {
            // In case it's needed, add support for a new data type
            HB_ASSERT(sizeof(dataType) == sizeof(syn_type_single),
                      "Data type not supported for buffer initialization: {} ",
                      dataType);

            memcpy(output, initializer, memorySize);
        }
    }
    else if (initSelect == MEM_INIT_ALL_ONES)
    {
        switch (dataType)
        {
            case syn_type_uint8:
            case syn_type_int8:
                setBuffer<uint8_t>(output, numElements, []() { return 1; });
                break;
            case syn_type_bf16:
                setBuffer<uint16_t>(output, numElements, []() { return bfloat16(1.0f); });
                break;
            case syn_type_fp8_152:
                setBuffer<fp8_152_t>(output, numElements, []() { return fp8_152_t((float)1.0); });
                break;
            case syn_type_int64:
            case syn_type_uint64:
                setBuffer<int64_t>(output, numElements, []() { return 1; });
                break;
            case syn_type_uint32:
            case syn_type_int32:
                setBuffer<int>(output, numElements, []() { return 1; });
                break;
            case syn_type_single:
                setBuffer<float>(output, numElements, []() { return 1; });
                break;
            default:
                HB_ASSERT(false, "unsupported tensor data type: {}", dataType);
        }
    }
    else  // MEM_INIT_ALL_ZERO
    {
        memset(output, 0, memorySize);
    }
}

synTensorDescriptorExt SynTrainingTestInfra::getTensorDescriptor(synDataType     dataType,
                                                              const TSize* tensorSizes,
                                                              unsigned        dims,
                                                              const char*     name,
                                                              TStride*       strides,
                                                              void*           ptr,
                                                              bool            isQuantized,
                                                              TSize*       minSizes,
                                                              synTensorType   tensorType)
{
    synTensorDescriptorExt desc;

    desc.m_dataType    = dataType;
    desc.m_dims        = dims;
    desc.m_name        = name;
    desc.m_ptr         = ptr;
    desc.m_isQuantized = isQuantized;
    desc.m_tensorType  = tensorType;

    memset(desc.m_strides, 0, sizeof(desc.m_strides));
    memset(desc.m_sizes, 1, sizeof(desc.m_sizes));
    memcpy(desc.m_sizes, tensorSizes, dims * sizeof(TSize));

    if (minSizes != nullptr)
    {
        memset(desc.m_minSizes, 1, sizeof(desc.m_sizes));
        memcpy(desc.m_minSizes, minSizes, dims * sizeof(TSize));
    }

    if (strides)
    {
        memcpy(desc.m_strides, strides, (dims + 1) * sizeof(TStride));
    }

    return desc;
}

SynTrainingTestInfra::TensorIndices SynTrainingTestInfra::createHugeTensors(unsigned        numTensors,
                                                                            TensorUsage     usage,
                                                                            bool            isPersistent,
                                                                            const char*     name,
                                                                            MemInitType     initSelect,
                                                                            const float*    initializer,
                                                                            TSize*          sizes,
                                                                            unsigned        dims,
                                                                            synDataType     dataType,
                                                                            TStride*        strides,
                                                                            unsigned        graphIndex,
                                                                            TSize           offset,
                                                                            const unsigned* sectionIndex,
                                                                            bool            isConst,
                                                                            TSize*          minSizes,
                                                                            synTensorType   tensorType,
                                                                            synTensor       existingTensor)
{
    synStatus            status         = synSuccess;
    void**               pHostBuffers   = nullptr;
    uint64_t*            pDeviceBuffers = nullptr;
    synTensor*           pTensors       = nullptr;
    synTensorDescriptorExt* pTensorDescs   = nullptr;

    bool existingTensorAPIBadInvocation = existingTensor && (numTensors > 1 || GCFG_OLD_TENSOR_CREATION_API.value());
    EXPECT_FALSE(existingTensorAPIBadInvocation)
        << "unsupported use case of createTensors for tracking an existing tensor created through the duplicate API";

    TSize* tensorSizes = (TSize*)getDefaultSizes();

    if (sizes != nullptr)
    {
        tensorSizes = sizes;
    }

    bool isDynamicShape = false;
    if (tensorType != DATA_TENSOR && minSizes == nullptr)
    {
        minSizes = tensorSizes;
    }
    if (minSizes != nullptr)
    {
        // If the max and min sizes are different, consider
        isDynamicShape = memcmp(tensorSizes, minSizes, sizeof(TSize) * dims) != 0;
    }

    if ((tensorType == synTensorType::DATA_TENSOR) && isDynamicShape)
    {
        tensorType = synTensorType::DATA_TENSOR_DYNAMIC;
    }

    std::vector<std::pair<unsigned, synLaunchTensorInfoExt>>* pEnqueueTensors = nullptr;
    bool                                                   is2DShape = tensorType == synTensorType::HOST_SHAPE_TENSOR;
    bool isHost2Device    = tensorType == synTensorType::HOST_TO_DEVICE_TENSOR;
    bool isNonDeviceShape = is2DShape || tensorType == synTensorType::SHAPE_TENSOR;
    bool isCompileOnly    = initSelect == MEM_INIT_COMPILATION_ONLY;

    unsigned      startIndex = 0;
    unsigned      endIndex   = 0;
    TSize      memorySize = 0;
    TensorIndices ret;

    EXPECT_TRUE(usage == INPUT_TENSOR || usage == OUTPUT_TENSOR);

    // Determine memory size in bytes
    memorySize = getMemorySize(tensorSizes, strides, dataType, dims);

    if (isHost2Device) memorySize *= 2;  // hack

    startIndex = m_tensors.size();
    endIndex   = startIndex + numTensors;

    if (endIndex > m_maxNumTensors)
    {
        resize(endIndex);
    }
    pHostBuffers   = m_hostBuffers.data();
    pDeviceBuffers = m_deviceBuffers.data();
    pTensors       = m_tensors.data();
    pTensorDescs   = m_tensorDescs.data();
    auto& currentGraph = getGraph(graphIndex);

    if (isNonDeviceShape)
    {
        pEnqueueTensors = &currentGraph.m_nonDeviceShapeTensorsExt;
    }
    else if (usage == INPUT_TENSOR)
    {
        pEnqueueTensors = &currentGraph.m_inputEnqueueTensorsExt;
    }
    else  // OUTPUT_TENSOR
    {
        pEnqueueTensors = &currentGraph.m_outputEnqueueTensorsExt;
    }

    for (unsigned i = startIndex; i < endIndex; ++i)
    {
        synSectionHandle pSectionHandle = nullptr;
        unsigned         concreteSectionIndex = INVALID_SECTION_IDX;
        if (isPersistent || isNonDeviceShape)
        {
            if (isPersistent)
            {
                if (sectionIndex == nullptr)
                {
                    concreteSectionIndex = isCompileOnly? createSection(graphIndex) : createSection(memorySize, graphIndex);
                }
                else
                {
                    concreteSectionIndex = *sectionIndex;
                }

                pSectionHandle = m_persistentSections[concreteSectionIndex].handle;
                // A const persistent tensor lays in const section, this section will be allocated
                // after the compilation stage (the actual size of the section should be taken from the recipe).
                if (isConst && checkIfConstSection(pSectionHandle))
                {
                    pDeviceBuffers[i] = 0;
                }
                else
                {
                    if (!isHost2Device)
                    {
                        pDeviceBuffers[i] = m_persistentSections[concreteSectionIndex].baseAddress + offset;
                    }
                    if (!isCompileOnly)
                    {
                        EXPECT_LE((memorySize + offset), m_persistentSections[concreteSectionIndex].size);
                    }
                }
            }
            else  // non-device shape tensor, do not allocate sections
            {
                pDeviceBuffers[i] = 0;
            }

            // Persistant tensor must have a name otherwise we won't be able to provide its HBM address
            // in the enqueue. So, if the caller didn't provide a name, auto-generate a unique one.
            if (name == nullptr)
            {
                name = getUniqueTensorName(usage);
            }
            else
            {
                m_uniqueNamesHolder.emplace_back(name);
                name = m_uniqueNamesHolder.back().c_str();
            }

            synLaunchTensorInfoExt launchInfo = {name, pDeviceBuffers[i], DATA_TENSOR, {0, 0, 0, 0, 0}, 0};
            memset(launchInfo.tensorSize, 0, sizeof(launchInfo.tensorSize));
            for (int j = 0 ; j < dims ; ++j)
            {
                launchInfo.tensorSize[j] =  tensorSizes[j] ;
            }
            launchInfo.tensorType = tensorType;

            // Record this tensor for the enqueue since we will have to provide its HBM address
            pEnqueueTensors->emplace_back(i, launchInfo);

            if (isDynamicShape)
            {
                currentGraph.m_tensorsMissingActualSize.push_back(i);
            }
        }

        if (!isPersistent && !isConst && (sectionIndex != nullptr))  // tensor belong to non-persistent section
        {
            EXPECT_TRUE(m_nonPersistentSections.size() > *sectionIndex)
                << "Wrong sectionIndex for non-persistent tensor";
            pSectionHandle = m_nonPersistentSections[*sectionIndex];
        }

        if (isPersistent || isConst || is2DShape || isHost2Device)
        {
            if (isCompileOnly)
            {
                m_tensorInitiatedInCompilationOnlyMode = true;
            }
            else
            {
                status = synHostMalloc(m_deviceId, memorySize, 0, &pHostBuffers[i]);
                if (isHost2Device)
                {
                    pDeviceBuffers[i] = (uint64_t)pHostBuffers[i];
                }
                EXPECT_EQ(status, synSuccess) << "Failed to allocate host memory for persistent tensor";

                // Initialize host memory
                initBufferValues(initSelect,
                                 initializer,
                                 dataType,
                                 getNumberOfElements(tensorSizes, dims),
                                 memorySize,
                                 pHostBuffers[i]);
            }
        }
        void* ptr         = isConst || is2DShape || isHost2Device ? static_cast<void*>(pHostBuffers[i]) : nullptr;
        bool  isQuantized = isConst ? true : false;
        if (GCFG_OLD_TENSOR_CREATION_API.value())
        {
            pTensorDescs[i] =
                getTensorDescriptor(dataType, tensorSizes, dims, name, nullptr, ptr, isQuantized, minSizes, tensorType);

            auto tensorDescNonExt = static_cast<synTensorDescriptor>(pTensorDescs[i]);
            if (!isConst)
            {
                status = synTensorCreate(&pTensors[i], &tensorDescNonExt, pSectionHandle, offset);
                EXPECT_EQ(status, synSuccess) << "Failed to create tensor";
                if (strides)
                {
                    synTensorDeviceFullLayoutExt layoutExt;
                    layoutExt.deviceDataType = dataType;
                    memset(layoutExt.strides, 0, sizeof(layoutExt.strides));
                    memcpy(layoutExt.strides, strides, (dims + 1) * sizeof(TStride));
                    auto layout =static_cast<synTensorDeviceFullLayout>(layoutExt);
                    status = synTensorSetDeviceFullLayout(pTensors[i], &layout);
                    EXPECT_EQ(status, synSuccess) << "Failed to set strides to tensor";
                }
            }
            else
            {
                status = synConstTensorCreate(&pTensors[i], &tensorDescNonExt);
                EXPECT_EQ(status, synSuccess) << "Failed to create const tensor";
            }
        }
        else
        {
            pTensorDescs[i] =
                getTensorDescriptor(dataType, tensorSizes, dims, name, strides, ptr, isQuantized, minSizes, tensorType);
            auto graphHandle = m_graphs[graphIndex].graphHandle;

            if (existingTensor)
            {
                pTensors[i] = existingTensor;
            }
            else
            {
                status = synTensorHandleCreate(&pTensors[i], graphHandle, tensorType, name);
                EXPECT_EQ(status, synSuccess) << "Failed to create tensor";
            }
            // There are some legacy tests which create const persistent tensors with non-const sections,
            // those tensors cannot be assigned to their section because of a validation test, carries out as part
            // of this call tensor->isPropsValid(), that fails if the tensor has host pointer and non-const section.
            bool constTensorWithoutConstSection = isConst && !checkIfConstSection(pSectionHandle);
            if (pSectionHandle != nullptr && !constTensorWithoutConstSection)
            {
                if (existingTensor == nullptr)
                {
                    synTensorAssignToSection(pTensors[i], pSectionHandle, offset);
                }
            }
            if (ptr != nullptr)
            {
                synTensorSetHostPtr(pTensors[i], ptr, memorySize, dataType, true);
            }

            synTensorGeometryExt geometry;
            geometry.dims = dims;
            std::copy(tensorSizes, tensorSizes + dims, geometry.sizes);
            synTensorSetGeometryExt(pTensors[i], &geometry, synGeometryMaxSizes);
            if (minSizes != nullptr)
            {
                std::copy(minSizes, minSizes + dims, geometry.sizes);
                synTensorSetGeometryExt(pTensors[i], &geometry, synGeometryMinSizes);
            }
            synTensorDeviceFullLayoutExt deviceLayoutExt;
            deviceLayoutExt.deviceDataType = dataType;
            if (strides != nullptr)
            {
                EXPECT_LE(memorySize, std::numeric_limits<TStride>().max());
                std::fill_n(deviceLayoutExt.strides, ARRAY_SIZE(deviceLayoutExt.strides), memorySize);
                std::copy(strides, strides + dims, deviceLayoutExt.strides);
            }
            else
            {
                std::fill_n(deviceLayoutExt.strides, ARRAY_SIZE(deviceLayoutExt.strides), 0);
            }
            auto deviceLayout = static_cast<synTensorDeviceFullLayout>(deviceLayoutExt);
            synTensorSetDeviceFullLayout(pTensors[i], &deviceLayout);
        }

        ret.push_back(i);
        TensorCreationParams tensorCreationParams =
            {i, usage, offset, concreteSectionIndex, isPersistent, isConst, isDynamicShape, strides != nullptr};
        currentGraph.tensorCreationParams.emplace(pTensors[i], tensorCreationParams);
    }
    return ret;
}

SynTrainingTestInfra::TensorIndices SynTrainingTestInfra::createTensors(unsigned        numTensors,
                                                                        TensorUsage     usage,
                                                                        bool            isPersistent,
                                                                        const char*     name,
                                                                        MemInitType     initSelect,
                                                                        const float*    initializer,
                                                                        unsigned*       sizes,
                                                                        unsigned        dims,
                                                                        synDataType     dataType,
                                                                        unsigned*       strides,
                                                                        unsigned        graphIndex,
                                                                        unsigned        offset,
                                                                        const unsigned* sectionIndex,
                                                                        bool            isConst,
                                                                        unsigned*       minSizes,
                                                                        synTensorType   tensorType,
                                                                        synTensor       existingTensor)
{
    std::optional<TensorStridesVector> stridesExt  = getVector<unsigned, TStride>(strides, dims + 1);
    std::optional<TensorSizesVector>   sizesExt    = getVector(sizes, dims);
    std::optional<TensorSizesVector>   minSizesExt = getVector(minSizes, dims);
    return createHugeTensors(numTensors,
                        usage,
                        isPersistent,
                        name,
                        initSelect,
                        initializer,
                        getVectorRawPtrOrNull(sizesExt),
                        dims,
                        dataType,
                        getVectorRawPtrOrNull(stridesExt),
                        graphIndex,
                        offset,
                        sectionIndex,
                        isConst,
                        getVectorRawPtrOrNull(minSizesExt),
                        tensorType,
                        existingTensor);
}

std::vector<unsigned> SynTrainingTestInfra::getGraphTensorIndices(unsigned graphIndex)
{
    auto&                 currentGraph = getGraph(graphIndex);
    std::vector<unsigned> tensorIndices;
    tensorIndices.reserve(currentGraph.m_inputEnqueueTensorsExt.size() + currentGraph.m_outputEnqueueTensorsExt.size());
    for (auto& enqueueTensorsExt : {currentGraph.m_inputEnqueueTensorsExt, currentGraph.m_outputEnqueueTensorsExt})
    {
        for (const auto& tensorInfo : enqueueTensorsExt)
        {
            tensorIndices.push_back(tensorInfo.first);
        }
    }
    return tensorIndices;
}

void SynTrainingTestInfra::changeTensorGeometry(unsigned     graphIndex,
                                                size_t       tensorIndex,
                                                TSize*    sizes,
                                                unsigned     dims,
                                                MemInitType  initSelect,
                                                const float* initializer)
{
    auto& currentGraph = getGraph(graphIndex);
    // update synLaunchTensorInfo
    synLaunchTensorInfoExt* launchTensorInfo = nullptr;
    auto updateLaunchTensorInfo = [this, tensorIndex, sizes, dims, &launchTensorInfo](auto& enqueueTensorsExt) {
        auto enqueueTensorsExtIter = find_if(enqueueTensorsExt.begin(),
                                             enqueueTensorsExt.end(),
                                             [tensorIndex](const auto& item) { return item.first == tensorIndex; });
        if (enqueueTensorsExtIter != enqueueTensorsExt.end())
        {
            launchTensorInfo = &enqueueTensorsExtIter->second;
            memset(launchTensorInfo->tensorSize, 0, sizeof(launchTensorInfo->tensorSize));
            memcpy(launchTensorInfo->tensorSize, sizes, sizeof(TSize) * dims);
            // update tensorDescs sizes
            memcpy(m_tensorDescs[tensorIndex].m_sizes,
                   launchTensorInfo->tensorSize,
                   sizeof(TSize) * SYN_MAX_TENSOR_DIM);
            return true;
        }
        return false;
    };
    ASSERT_TRUE(updateLaunchTensorInfo(currentGraph.m_inputEnqueueTensorsExt) ||
                updateLaunchTensorInfo(currentGraph.m_outputEnqueueTensorsExt));

    if (m_hostBuffers[tensorIndex])
    {
        bool      isHost2Device = (m_deviceBuffers[tensorIndex] == (uint64_t)m_hostBuffers[tensorIndex]);
        synStatus status        = synHostFree(m_deviceId, m_hostBuffers[tensorIndex], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free tensor's host memory";
        m_hostBuffers[tensorIndex] = nullptr;
        // we multiply by two, mimicking the hack from tensor allocation above for isHost2Device
        uint64_t memorySize = 2 * getMemorySize(sizes, nullptr, m_tensorDescs[tensorIndex].m_dataType, dims);
        status              = synHostMalloc(m_deviceId, memorySize, 0, &m_hostBuffers[tensorIndex]);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate tensor's host memory";
        if (m_tensorDescs[tensorIndex].m_ptr)
        {
            m_tensorDescs[tensorIndex].m_ptr = m_hostBuffers[tensorIndex];
        }
        // Initialize host memory
        initBufferValues(initSelect,
                         initializer,
                         m_tensorDescs[tensorIndex].m_dataType,
                         getNumberOfElements(sizes, dims),
                         memorySize,
                         m_hostBuffers[tensorIndex]);
        if (isHost2Device)
        {
            m_deviceBuffers[tensorIndex] = (uint64_t)m_hostBuffers[tensorIndex];
        }
        // change the device memory allocation for the tensor
        auto concreteSectionIndex = currentGraph.tensorCreationParams[m_tensors[tensorIndex]].concreteSectionIndex;
        if (concreteSectionIndex != INVALID_SECTION_IDX)
        {
            updateSectionAllocator(concreteSectionIndex, memorySize);
            launchTensorInfo->pTensorAddress = m_persistentSections[concreteSectionIndex].baseAddress;
            m_deviceBuffers[tensorIndex]     = m_persistentSections[concreteSectionIndex].baseAddress;
        }
    }

    // update the geometry
    synTensorGeometryExt geometry;
    geometry.dims = dims;
    std::copy(sizes, sizes + dims, geometry.sizes);
    synTensorSetGeometryExt(m_tensors[tensorIndex], &geometry, synGeometryMaxSizes);
}

unsigned SynTrainingTestInfra::createTensor(TensorUsage  usage,
                                            MemInitType  initSelect,
                                            const float* initializer,
                                            unsigned*    sizes,
                                            unsigned     dims,
                                            synDataType  dataType,
                                            unsigned*    strides,
                                            unsigned*    minSizes,
                                            unsigned     graphIndex)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        false,
                                        nullptr,
                                        initSelect,
                                        initializer,
                                        sizes,
                                        dims,
                                        dataType,
                                        strides,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        false,
                                        minSizes);
    return index[0];
}

unsigned SynTrainingTestInfra::createConstTensor(MemInitType  initSelect,
                                                 const float* initializer,
                                                 unsigned*    sizes,
                                                 unsigned     dims,
                                                 synDataType  dataType,
                                                 unsigned*    strides,
                                                 const char*  name,
                                                 unsigned     graphIndex)
{
    TensorIndices index = createTensors(1,
                                        INPUT_TENSOR,
                                        false,
                                        name,
                                        initSelect,
                                        initializer,
                                        sizes,
                                        dims,
                                        dataType,
                                        strides,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        true);
    return index[0];
}

unsigned SynTrainingTestInfra::createConstPersistTensor(TensorUsage     usage,
                                                        MemInitType     initSelect,
                                                        const float*    initializer,
                                                        unsigned*       sizes,
                                                        unsigned        dims,
                                                        synDataType     dataType,
                                                        unsigned*       strides,
                                                        const char*     name,
                                                        unsigned        graphIndex,
                                                        unsigned        offsetInSection,
                                                        const unsigned* sectionIndex)
{
    unsigned concreteSectionIndex = INVALID_SECTION_IDX;
    if (sectionIndex == nullptr)
    {
        concreteSectionIndex = createConstSection(graphIndex);
        HB_ASSERT(concreteSectionIndex != INVALID_SECTION_IDX, "Expecting create section to succeed");
        sectionIndex         = &concreteSectionIndex;
        offsetInSection      = 0;
    }
    else
    {
        EXPECT_TRUE(checkIfConstSection(m_persistentSections[*sectionIndex].handle));
    }
    TensorIndices index = createTensors(1,
                                        usage,
                                        true,
                                        name,
                                        initSelect,
                                        initializer,
                                        sizes,
                                        dims,
                                        dataType,
                                        strides,
                                        graphIndex,
                                        offsetInSection,
                                        sectionIndex,
                                        true);
    return index[0];
}

unsigned SynTrainingTestInfra::createShapeTensor(TensorUsage usage,
                                                 unsigned*   sizes,
                                                 unsigned*   minSizes,
                                                 unsigned    dims,
                                                 synDataType dataType,
                                                 const char* name,
                                                 unsigned    graphIndex)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        false,
                                        name,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        sizes,
                                        dims,
                                        dataType,
                                        nullptr,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        false,
                                        minSizes,
                                        synTensorType::SHAPE_TENSOR);

    return index[0];
}

unsigned SynTrainingTestInfra::createHostShapeTensor(TensorUsage usage,
                                                     unsigned*   sizes,
                                                     unsigned*   data,
                                                     const char* name,
                                                     unsigned    graphIndex)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        false,
                                        name,
                                        MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                        reinterpret_cast<const float*>(data),
                                        sizes,
                                        3,
                                        syn_type_uint32,
                                        nullptr,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        false,
                                        nullptr,
                                        synTensorType::HOST_SHAPE_TENSOR);

    return index[0];
}

unsigned SynTrainingTestInfra::createHost2DeviceTensor(TensorUsage usage,
                                                       unsigned*   sizes,
                                                       unsigned*   data,
                                                       unsigned    dims,
                                                       const char* name,
                                                       unsigned    graphIndex)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        true,
                                        name,
                                        MEM_INIT_FROM_INITIALIZER_NO_CAST,
                                        reinterpret_cast<const float*>(data),
                                        sizes,
                                        dims,
                                        syn_type_uint32,
                                        nullptr,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        false,
                                        nullptr,
                                        synTensorType::HOST_TO_DEVICE_TENSOR);

    return index[0];
}

unsigned SynTrainingTestInfra::createConstSection(uint64_t graphIndex)
{
    const auto handle = createSection(getGraph(graphIndex).graphHandle);

    synSectionSetConst(handle, true);

    m_persistentSections.push_back({handle, 0, 0, nullptr});
    return m_persistentSections.size() - 1;
}

bool SynTrainingTestInfra::checkIfConstSection(synSectionHandle sectionHandle)
{
    if (sectionHandle == nullptr) return false;
    bool sectionIsConst = false;
    synSectionGetConst(sectionHandle, &sectionIsConst);
    return sectionIsConst;
}

unsigned SynTrainingTestInfra::createHost2DeviceIntermediateTensor(TensorUsage usage,
                                                                   unsigned*   sizes,
                                                                   unsigned    dims,
                                                                   const char* name,
                                                                   unsigned    graphIndex)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        false,
                                        name,
                                        MEM_INIT_NONE,
                                        nullptr,
                                        sizes,
                                        dims,
                                        syn_type_uint32,
                                        nullptr,
                                        graphIndex,
                                        0,
                                        nullptr,
                                        false,
                                        nullptr,
                                        synTensorType::HOST_TO_DEVICE_TENSOR);

    return index[0];
}

unsigned SynTrainingTestInfra::createPersistTensor(TensorUsage     usage,
                                                   MemInitType     initSelect,
                                                   const float*    initializer,
                                                   unsigned*       sizes,
                                                   unsigned        dims,
                                                   synDataType     dataType,
                                                   unsigned*       strides,
                                                   const char*     name,
                                                   unsigned        graphIndex,
                                                   unsigned        offsetInSection,
                                                   const unsigned* sectionIndex,
                                                   unsigned*       minSizes)
{
    TensorIndices index = createTensors(1,
                                        usage,
                                        true,
                                        name,
                                        initSelect,
                                        initializer,
                                        sizes,
                                        dims,
                                        dataType,
                                        strides,
                                        graphIndex,
                                        offsetInSection,
                                        sectionIndex,
                                        false,
                                        minSizes);
    return index[0];
}

unsigned SynTrainingTestInfra::connectOutputTensorToInputTensor(unsigned outputTensorIndex)
{
    return outputTensorIndex;
}
void SynTrainingTestInfra::addNodeToGraph(const char*  guid,
                                          void*        userParams,
                                          unsigned     paramsSize,
                                          const char*  nodeName,
                                          unsigned     graphIndex,
                                          const char** inputLayouts,
                                          const char** outputLayouts)
{
    std::vector<unsigned> inputTensorIndices;
    std::vector<unsigned> outputTensorIndices;
    for (unsigned i = 0; i < m_tensors.size() - 1; ++i)
    {
        inputTensorIndices.push_back(i);
    }
    // Last tensor consider as output
    outputTensorIndices.push_back(m_tensors.size() - 1);

    addNodeToGraph(guid,
                   inputTensorIndices,
                   outputTensorIndices,
                   userParams,
                   paramsSize,
                   nodeName,
                   graphIndex,
                   nullptr,
                   inputLayouts,
                   outputLayouts);
}

void SynTrainingTestInfra::addNodeToGraph(const char*          guid,
                                          const TensorIndices& inputTensorIndices,
                                          const TensorIndices& outputTensorIndices,
                                          void*                userParams,
                                          unsigned             paramsSize,
                                          const char*          nodeName,
                                          unsigned             graphIndex,
                                          synNodeId*           nodeId,
                                          const char**         inputLayouts,
                                          const char**         outputLayouts)
{
    std::vector<synTensor> inTensors;
    std::vector<synTensor> outTensors;
    ASSERT_FALSE(((userParams && paramsSize == 0) || (userParams == nullptr && paramsSize != 0)))
        << "Invalid user param";
    // put all tensors in continuous array
    for (auto i : inputTensorIndices)
    {
        if (i == INVALID_TENSOR_INDEX)
        {
            // support nullptr tensors
            inTensors.push_back(nullptr);
        }
        else
        {
            assert(i < m_maxNumTensors);
            inTensors.push_back(m_tensors[i]);
        }
    }
    for (auto i : outputTensorIndices)
    {
        if (i == INVALID_TENSOR_INDEX)
        {
            // support nullptr tensors
            outTensors.push_back(nullptr);
        }
        else
        {
            assert(i < m_maxNumTensors);
            outTensors.push_back(m_tensors[i]);
        }
    }
    auto&          graphData   = getGraph(graphIndex);
    synGraphHandle graphHandle = graphData.graphHandle;

    if (nodeId != nullptr)
    {
        ASSERT_EQ(synSuccess,
                  synNodeCreateWithId(graphHandle,
                                      inTensors.data(),
                                      outTensors.data(),
                                      inTensors.size(),
                                      outTensors.size(),
                                      userParams,
                                      paramsSize,
                                      guid,
                                      nodeName,
                                      nodeId,
                                      inputLayouts,
                                      outputLayouts))
            << "Failed to create node with GUID " << guid;
        graphData.nodesById.emplace(*nodeId);
    }
    else
    {
        ASSERT_EQ(synSuccess,
                  synNodeCreate(graphHandle,
                                inTensors.data(),
                                outTensors.data(),
                                inTensors.size(),
                                outTensors.size(),
                                userParams,
                                paramsSize,
                                guid,
                                nodeName,
                                inputLayouts,
                                outputLayouts))
            << "Failed to create node with GUID " << guid;
    }
    ++graphData.numNodes;
}

void SynTrainingTestInfra::setNodeDependency(const synNodeId* pBlockingNodesIdList,
                                             const synNodeId* pBlockedNodesIdList,
                                             const uint32_t   numberblocking,
                                             const uint32_t   numberblocked,
                                             unsigned int     graphIndex)
{
    EXPECT_EQ(synNodeDependencySet(getGraph(graphIndex).graphHandle,
                                   pBlockingNodesIdList,
                                   pBlockedNodesIdList,
                                   numberblocking,
                                   numberblocked),
              synSuccess);
}

void SynTrainingTestInfra::compileAndRun()
{
    compileAndRun(0);
}

void SynTrainingTestInfra::compileAndRun(unsigned graphIndex)
{
    compileTopology("", graphIndex);
    runTopology(graphIndex);
}

bool SynTrainingTestInfra::validateInputTensorUsage(const GraphData& graphData) const
{
    const HabanaGraph* pGraph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);
    for (const auto& t : pGraph->getGraphInputs())
    {
        if (!t || !t->isDataTensor()) continue;
        const auto synTensor = reinterpret_cast<::synTensor>(t.get());
        const auto it        = graphData.tensorCreationParams.find(synTensor);
        if (it == graphData.tensorCreationParams.end()) continue;
        if (it->second.usage != INPUT_TENSOR)
        {
            return false;
        }
    }
    return true;
}

void SynTrainingTestInfra::compileTopology(const std::string& topologyName, unsigned graphIndex)
{
    GraphData& graphData = getGraph(graphIndex);

    // validate graph input tensors usage
    ASSERT_TRUE(validateInputTensorUsage(graphData)) << "All graph input tensor usage must be set to INPUT";

    if (std::strlen(topologyName.c_str()) == 0)
    {
        graphData.recipeName = GetTestFileName();
    }
    else
    {
        graphData.recipeName = topologyName;
    }

    synStatus status =
        synGraphCompile(&graphData.recipeHandle, graphData.graphHandle, graphData.recipeName.c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile recipe";
    if (GCFG_PRESERVE_TESTS_RECIPE.value())
    {
        status = synRecipeSerialize(graphData.recipeHandle, graphData.recipeName.c_str());
        ASSERT_EQ(status, synSuccess) << "Failed to serialize recipe";
    }

    status = synWorkspaceGetSize(&graphData.workspaceSize, graphData.recipeHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to get workspace size";
}

void SynTrainingTestInfra::copyInputTensors(unsigned                             graphIndex,
                                            uint64_t&                            programAddress,
                                            std::vector<synLaunchTensorInfoExt>& concatTensors,
                                            bool                                 initPersistentOutputs)
{
    synStatus  status    = synSuccess;
    GraphData& graphData = getGraph(graphIndex);

    ASSERT_EQ(graphData.m_tensorsMissingActualSize.size(), 0)
        << graphData.m_tensorsMissingActualSize.size()
        << " Dynamic tensors weren't provided with actual size (The first is "
        << graphData.m_tensorsMissingActualSize[0] << " )";

    if (0 != graphData.workspaceSize)
    {
        // Allocate HBM for workspace and program
        status = synDeviceMalloc(m_deviceId, graphData.workspaceSize, 0, 0, &graphData.hbmAddr);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate HBM memory for workspace and program";
    }

    programAddress = graphData.hbmAddr;

    uint64_t copyMaxAmount = std::max(graphData.m_inputEnqueueTensorsExt.size(),
                                      graphData.m_outputEnqueueTensorsExt.size());
    uint64_t* srcArray  = new uint64_t[copyMaxAmount];
    uint64_t* sizeArray = new uint64_t[copyMaxAmount];
    uint64_t* dstArray  = new uint64_t[copyMaxAmount];
    // Copy graph inputs from host memory to HBM
    uint64_t  numOfCopies = 0;
    for (auto inputEnqueueTensor : graphData.m_inputEnqueueTensorsExt)
    {
        if (inputEnqueueTensor.second.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            // inputEnqueueTensor.second.pTensorAddress = m_deviceBuffers[inputEnqueueTensor.first];
            continue;  // hack! XXX XXX XXX do not commit
        }

        unsigned inputIndex = inputEnqueueTensor.first;
        // check if the tensor is in const section, because in case it is then
        // the data of the const section has already copied to the device (by the call to initializeAllConstSections),
        // and the tensor address in synLaunchTensorInfoExt will be updated accordingly in the call to
        // _replaceConstSectionTensorDramAddr
        if (checkIfTensorInConstSection(m_tensors[inputIndex], graphIndex)) continue;

        uint64_t memorySize = getMemorySize(m_tensorDescs[inputIndex].m_sizes,
                                            m_tensorDescs[inputIndex].m_strides,
                                            m_tensorDescs[inputIndex].m_dataType,
                                            m_tensorDescs[inputIndex].m_dims);

        srcArray[numOfCopies]  = (uint64_t)m_hostBuffers[inputIndex];
        sizeArray[numOfCopies] = memorySize;
        dstArray[numOfCopies]  = (uint64_t)m_deviceBuffers[inputIndex];
        numOfCopies++;
    }
    status = synMemCopyAsyncMultiple(m_streamHandleDownload,
                                     srcArray,
                                     sizeArray,
                                     dstArray,
                                     HOST_TO_DRAM,
                                     numOfCopies);
    ASSERT_EQ(status, synSuccess) << "Failed to copy input tensor to the device";


    if (initPersistentOutputs)
    {
        numOfCopies = 0;
        for (auto outputEnqueueTensor : graphData.m_outputEnqueueTensorsExt)
        {
            unsigned outputIndex = outputEnqueueTensor.first;
            uint64_t memorySize  = getMemorySize(m_tensorDescs[outputIndex].m_sizes,
                                                 m_tensorDescs[outputIndex].m_strides,
                                                 m_tensorDescs[outputIndex].m_dataType,
                                                 m_tensorDescs[outputIndex].m_dims);

            srcArray[numOfCopies]  = (uint64_t)m_hostBuffers[outputIndex];
            sizeArray[numOfCopies] = memorySize;
            dstArray[numOfCopies]  = (uint64_t)m_deviceBuffers[outputIndex];
            numOfCopies++;
        }
        status = synMemCopyAsyncMultiple(m_streamHandleDownload,
                                         srcArray,
                                         sizeArray,
                                         dstArray,
                                         HOST_TO_DRAM,
                                         numOfCopies);
        ASSERT_EQ(status, synSuccess) << "Failed to copy output tensor to the device";
    }

    delete[] srcArray;
    delete[] sizeArray;
    delete[] dstArray;

    // Wait until all downloads are done before proceeding to the compute
    status = synEventRecord(m_eventHandle, m_streamHandleDownload);
    ASSERT_EQ(status, synSuccess) << "Failed to record-event (copy to the device)";

    status = synStreamWaitEvent(m_streamHandleCompute, m_eventHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to wait on stream event: completion of copy to the device";

    for (auto inputEnqueueTensor : graphData.m_inputEnqueueTensorsExt)
    {
        if (inputEnqueueTensor.second.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            inputEnqueueTensor.second.pTensorAddress = m_deviceBuffers[inputEnqueueTensor.first];
        }
        concatTensors.push_back(inputEnqueueTensor.second);
    }

    for (auto outputEnqueueTensor : graphData.m_outputEnqueueTensorsExt)
    {
        concatTensors.push_back(outputEnqueueTensor.second);
    }

    for (auto shapeTensor : graphData.m_nonDeviceShapeTensorsExt)
    {
        if (shapeTensor.second.tensorType == HOST_SHAPE_TENSOR)
        {
            shapeTensor.second.pTensorAddress = (uint64_t)m_runtimeHostBuffers[shapeTensor.first];
        }
        concatTensors.push_back(shapeTensor.second);
    }

    // special case for running inference with const section
    // replace the device address of the allocated tensor with his new address in the const section
    EXPECT_TRUE(_replaceConstSectionTensorDramAddr(concatTensors));

    synLaunchTensorInfoExt* concTensors       = &concatTensors[0];
    uint32_t             totalNumOfTensors = concatTensors.size();
    prepareTensorInfo(graphData.recipeHandle, concTensors, totalNumOfTensors);
}

bool SynTrainingTestInfra::_replaceConstSectionTensorDramAddr(std::vector<synLaunchTensorInfoExt>& concatTensors)
{
    for (auto& tensorInfo : m_constSectionTensorInfoVec)
    {
        if (!tensorInfo.tensorName.empty())
        {
            bool bFound = false;
            for (auto& tInfo : concatTensors)
            {
                if (tensorInfo.tensorName.compare(tInfo.tensorName) == 0)
                {
                    bFound = true;

                    // override the address
                    tInfo.pTensorAddress = tensorInfo.deviceAddress;
                }
            }
            if (!bFound) return false;
        }
    }
    m_constSectionTensorInfoVec.clear();
    return true;
}

void SynTrainingTestInfra::downloadDataToDevice(const uint64_t src, const uint64_t size, const uint64_t dst)
{
    synStatus status = synMemCopyAsync(m_streamHandleDownload, src, size, dst, HOST_TO_DRAM);

    status = synStreamSynchronize(m_streamHandleDownload);

    ASSERT_EQ(status, synSuccess) << "downloadDataToDevice Failed to copy data to the device";
}

void SynTrainingTestInfra::runTopology(unsigned  graphIndex,
                                       bool      initPersistentOutputs,
                                       synStatus expectedLaunch,
                                       synStatus expectedStreamSync)
{
    HB_ASSERT(!m_tensorInitiatedInCompilationOnlyMode, "There is a tensor that initiated in compilation only mode");

    // Initialize const sections if any
    initializeAllConstSections(graphIndex);

    ASSERT_FALSE(HasFailure()) << __func__ << " Graph idx: " << graphIndex
                               << " - Test has failure before launch. Skipping launch";

    GraphData&                       graphData      = getGraph(graphIndex);
    uint64_t                         programAddress = 0;
    std::vector<synLaunchTensorInfoExt> concatTensors;

    // returns the persistent tensors for the launch
    copyInputTensors(graphIndex, programAddress, concatTensors, initPersistentOutputs);

    synEventHandle hBegin {};
    synStatus      status = synEventCreate(&hBegin, m_deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(synSuccess, status);
    synEventHandle hEnd {};
    status = synEventCreate(&hEnd, m_deviceId, EVENT_COLLECT_TIME);
    ASSERT_EQ(synSuccess, status);

    status = synEventRecord(hBegin, m_streamHandleCompute);
    ASSERT_EQ(synSuccess, status);

    status = synLaunchExt(m_streamHandleCompute,
                       concatTensors.data(),
                       concatTensors.size(),
                       programAddress,
                       graphData.recipeHandle,
                       0);
    ASSERT_EQ(status, expectedLaunch) << "Unexpected status for synLaunch";

    if (expectedLaunch != synSuccess) return;

    status = synEventRecord(hEnd, m_streamHandleCompute);
    ASSERT_EQ(synSuccess, status);

    status = synStreamSynchronize(m_streamHandleCompute);
    ASSERT_EQ(status, expectedStreamSync) << "Failed to sync compute stream";

    if (expectedStreamSync != synSuccess) return;

    // wait for the event before checking elapsed time. The synStreamSynchronize might not
    // be enough because of possible re-ordering of completions in lkd
    status = synEventSynchronize(hEnd);
    ASSERT_EQ(status, synSuccess);

    status = synEventElapsedTime(&m_lastLaunchElapsedTime, hBegin, hEnd);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_TEST, "Elapsed time failed");
    }
    copyOutputTensors(graphIndex);
}

void SynTrainingTestInfra::copyOutputTensors(unsigned graphIndex)
{
    // Wait for the completion of the compute
    synStatus status = synEventRecord(m_eventHandle, m_streamHandleCompute);
    ASSERT_EQ(status, synSuccess) << "Failed to record event (enqueue)";

    status = synStreamWaitEvent(m_streamHandleUpload, m_eventHandle, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to wait on stream event: completion of enqueue (compute)";

    GraphData& graphData = getGraph(graphIndex);

    uint64_t copyMaxAmount = graphData.m_outputEnqueueTensorsExt.size();
    uint64_t* srcArray  = new uint64_t[copyMaxAmount];
    uint64_t* sizeArray = new uint64_t[copyMaxAmount];
    uint64_t* dstArray  = new uint64_t[copyMaxAmount];

    // Copy graph outputs from HBM to host memory
    uint64_t  numOfCopies = 0;
    for (auto outputEnqueueTensor : graphData.m_outputEnqueueTensorsExt)
    {
        unsigned outputIndex = outputEnqueueTensor.first;
        uint64_t memorySize  = getMemorySize(m_tensorDescs[outputIndex].m_sizes,
                                            m_tensorDescs[outputIndex].m_strides,
                                            m_tensorDescs[outputIndex].m_dataType,
                                            m_tensorDescs[outputIndex].m_dims);

        srcArray[numOfCopies]  = (uint64_t)m_deviceBuffers[outputIndex];
        sizeArray[numOfCopies] = memorySize;
        dstArray[numOfCopies]  = (uint64_t)m_hostBuffers[outputIndex];
        numOfCopies++;
    }
    status = synMemCopyAsyncMultiple(m_streamHandleUpload,
                                     srcArray,
                                     sizeArray,
                                     dstArray,
                                     DRAM_TO_HOST,
                                     numOfCopies);
    ASSERT_EQ(status, synSuccess) << "Failed to copy output tensor from the device";

    delete[] srcArray;
    delete[] sizeArray;
    delete[] dstArray;

    // Wait for the completion of last operation
    status = synStreamSynchronize(m_streamHandleUpload);
    ASSERT_EQ(status, synSuccess) << "Failed to synchronize stream (copy from the device to host)";
}

void SynTrainingTestInfra::init()
{
    m_deviceId = _getDeviceId();
    createStreams();
    createEvents();
    switch (m_testConfig.m_compilationMode)
    {
        case COMP_EAGER_MODE_TEST:
        case COMP_GRAPH_MODE_TEST:
            createGraph(m_testConfig.m_compilationMode);
            break;
        case COMP_BOTH_MODE_TESTS:
            createGraph(COMP_GRAPH_MODE_TEST);
            createGraph(COMP_EAGER_MODE_TEST);
            break;
    }
}

void SynTrainingTestInfra::reset()
{
    synStatus status = synSuccess;

    for (unsigned i = 0; i < m_maxNumTensors; ++i)
    {
        if (m_tensors[i])
        {
            status = synTensorDestroy(m_tensors[i]);
            ASSERT_EQ(status, synSuccess) << "Failed to destroy tensor";
            m_tensors[i] = nullptr;
        }
        if (m_hostBuffers[i] != nullptr)
        {
            status = synHostFree(m_deviceId, m_hostBuffers[i], 0);
            ASSERT_EQ(status, synSuccess) << "Failed to free tensor's host memory";
            m_hostBuffers[i] = nullptr;
        }
    }

    // Release graph related resources
    for (auto graphData : m_graphs)
    {
        if (graphData.graphHandle)
        {
            status = synGraphDestroy(graphData.graphHandle);
            ASSERT_EQ(status, synSuccess) << "Failed to destroy graph";
        }

        if (graphData.hbmAddr)
        {
            status = synDeviceFree(m_deviceId, graphData.hbmAddr, 0);
            ASSERT_EQ(status, synSuccess) << "Failed to free HBM memory (workspace + program)";
        }

        if (graphData.recipeHandle)
        {
            status = synRecipeDestroy(graphData.recipeHandle);
            ASSERT_EQ(status, synSuccess) << "Failed to destroy recipe";
        }

        if (graphData.recipeName.c_str())
        {
            CleanTestIntermediatesFiles(graphData.recipeName, false);
        }

        graphData.m_inputEnqueueTensorsExt.clear();
        graphData.m_outputEnqueueTensorsExt.clear();
    }
    m_graphs.clear();

    // Release persistent memory sections
    for (auto memSec : m_persistentSections)
    {
        if (memSec.handle)
        {
            status = synSectionDestroy(memSec.handle);
            ASSERT_EQ(status, synSuccess) << "Failed to destroy section";
        }
    }

    // Release non-persistent memory sections
    for (auto memSec : m_nonPersistentSections)
    {
        if (memSec)
        {
            status = synSectionDestroy(memSec);
            ASSERT_EQ(status, synSuccess) << "Failed to destroy section";
        }
    }

    if (m_isEventCreated)
    {
        status = synEventDestroy(m_eventHandle);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy event";
    }

    if (m_streamHandleDownload != nullptr)
    {
        status = synStreamDestroy(m_streamHandleDownload);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy Download stream";
        m_streamHandleDownload = nullptr;
    }

    if (m_streamHandleCompute != nullptr)
    {
        status = synStreamDestroy(m_streamHandleCompute);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy Compute stream";
        m_streamHandleCompute = nullptr;
    }

    if (m_streamHandleUpload != nullptr)
    {
        status = synStreamDestroy(m_streamHandleUpload);
        ASSERT_EQ(status, synSuccess) << "Failed to destroy Upload stream";
        m_streamHandleUpload = nullptr;
    }

    m_uniqueNamesHolder.clear();
    m_sequentialNumber = 0;

    for (unsigned idx = 0; idx < m_globalConfStack.size(); idx++)
    {
        m_globalConfStack.pop_front();  // revert global configs in order
    }

    m_hostBuffers.clear();
    m_deviceBuffers.clear();
    m_tensors.clear();
    m_tensorDescs.clear();
    m_persistentSections.clear();
    m_nonPersistentSections.clear();
    m_uniqueNamesHolder.clear();
    m_globalConfStack.clear();
    m_maxNumTensors = 0;
    m_constSectionTensorInfoVec.clear();
}

uint64_t SynTrainingTestInfra::getNumberOfElements(const unsigned* sizes, unsigned dims)
{
    auto vecSizes = getVector(sizes,dims);
    return getNumberOfElements(getVectorRawPtrOrNull(vecSizes),dims);
}

uint64_t SynTrainingTestInfra::getNumberOfElements(const TSize* sizes, unsigned dims)
{
    TSize elements = dims ? 1 : 0;

    for (int i = 0; i < dims; i++)
    {
        elements *= sizes[i];
    }

    return elements;
}

uint64_t
SynTrainingTestInfra::getMemorySize(const unsigned* sizes, const unsigned* strides, synDataType type, unsigned dims)
{
    auto sizesExtVec   = getVector(sizes, dims);
    auto stridesExtVec = getVector<const unsigned, TStride>(strides, dims + 1);
    return getMemorySize(getVectorRawPtrOrNull(sizesExtVec), getVectorRawPtrOrNull(stridesExtVec), type, dims);
}

uint64_t SynTrainingTestInfra::getMemorySize(const unsigned* sizes, synDataType type, unsigned dims)
{
    return getMemorySize(sizes, nullptr, type, dims);
}

uint64_t
SynTrainingTestInfra::getMemorySize(const TSize* sizes, const TStride* strides, synDataType type, unsigned dims)
{
    if (strides != nullptr)
    {
        if (std::none_of(strides, strides + dims + 1, [](uint32_t v) { return v == 0; }))
        {
            TStride lastElementOffset = 0;
            for (unsigned d = 0; d < dims; d++)
            {
                lastElementOffset += strides[d] * (sizes[d] - 1);
            }
            return lastElementOffset + getElementSizeInBytes(type);
        }
    }
    return getNumberOfElements(sizes, dims) * getElementSizeInBytes(type);
}
uint64_t SynTrainingTestInfra::getMemorySize(const TSize* sizes, synDataType type, unsigned dims)
{
    return getMemorySize(sizes, nullptr, type, dims);
}

uint64_t SynTrainingTestInfra::getDefaultNumberOfElements()
{
    return getNumberOfElements(getDefaultSizes());
}

const TSize* SynTrainingTestInfra::getDefaultSizes()
{
    static const TSize defaultSizes[m_maxTensorDims] = {16, 16, 16, 16};
    return defaultSizes;
}

const unsigned* SynTrainingTestInfra::getDefaultSizes32b()
{
    static const std::vector<unsigned> defaultSizes32b(getDefaultSizes(), getDefaultSizes() + m_maxTensorDims);
    return defaultSizes32b.data();
}

const char* SynTrainingTestInfra::getUniqueTensorName(TensorUsage usage)
{
    static const char* names[] = {"autoGenPersistInputTensorName_", "autoGenPersistOutputTensorName_", "unexpected_"};
    m_uniqueNamesHolder.push_back(std::string(names[usage]) + std::to_string(m_sequentialNumber++));
    return m_uniqueNamesHolder.back().c_str();
}

synSectionHandle SynTrainingTestInfra::createSection(synGraphHandle graphHandle)
{
    synSectionHandle handle;
    synStatus        status = synSectionCreate(&handle, 0, graphHandle);
    EXPECT_EQ(status, synSuccess) << "Failed to create memory section for persistent tensor";
    return handle;
}

unsigned SynTrainingTestInfra::createSection(uint64_t size, uint64_t graphIndex)
{
    return createSection(size, getGraph(graphIndex).graphHandle);
}

unsigned SynTrainingTestInfra::createSection(uint64_t size, synGraphHandle graphHandle)
{
    const auto handle = createSection(graphHandle);
    // create allocator for each section
    TrivialDeviceMemoryAllocatorPtr allocator = std::make_shared<TrivialDeviceMemoryAllocator>(m_deviceId, size);
    m_persistentSections.push_back({handle, allocator->getDeviceMemory(size), size, allocator});
    return m_persistentSections.size() - 1;
}

void SynTrainingTestInfra::updateSectionAllocator(unsigned sectionIdx, uint64_t size)
{
    m_persistentSections[sectionIdx].allocator.reset();
    m_persistentSections[sectionIdx].allocator   = std::make_shared<TrivialDeviceMemoryAllocator>(m_deviceId, size);
    m_persistentSections[sectionIdx].baseAddress = m_persistentSections[sectionIdx].allocator->getDeviceMemory(size);
    m_persistentSections[sectionIdx].size        = size;
}

unsigned SynTrainingTestInfra::createNonPersistentSection(uint64_t graphIndex, bool isRmw)
{
    synSectionHandle handle;
    synStatus        status = synSectionCreate(&handle, 0, getGraph(graphIndex).graphHandle);
    EXPECT_EQ(status, synSuccess) << "Failed to create memory section for non-persistent rmw tensor";

    EXPECT_EQ(synSuccess, synSectionSetPersistent(handle, false));
    EXPECT_EQ(synSuccess, synSectionSetRMW(handle, isRmw));

    m_nonPersistentSections.push_back(handle);
    return m_nonPersistentSections.size() - 1;
}

void SynTrainingTestInfra::setActualSizes(unsigned tensorIndex, const TestSizeVec& tensorSizes, unsigned graphIndex)
{
    setActualSizes(tensorIndex, tensorSizes.data(), graphIndex);
}

void SynTrainingTestInfra::setRuntimeHostBuffer(unsigned tensorIndex, void* bufferData, unsigned graphIndex)
{
    EXPECT_LE(tensorIndex + 1, m_runtimeHostBuffers.size());
    m_runtimeHostBuffers[tensorIndex] = bufferData;
}

void SynTrainingTestInfra::setActualScalarParametersData(unsigned tensorIndex,
                                                         void*    data,
                                                         unsigned size,
                                                         unsigned graphIndex)
{
    EXPECT_LE(tensorIndex + 1, m_hostBuffers.size());
    unsigned elementSize = getElementSizeInBytes(getTensorDescriptor(tensorIndex).m_dataType);
    for (auto& pair : getGraph(graphIndex).m_inputEnqueueTensorsExt)
    {
        if (pair.first == tensorIndex)
        {
            unsigned total = 1;
            for (unsigned i = 0; i < getTensorDescriptor(tensorIndex).m_dims; ++i)
            {
                total *= pair.second.tensorSize[i];
            }
            EXPECT_EQ(total * elementSize, size);
            break;
        }
    }

    memcpy(m_hostBuffers[tensorIndex], data, size);
}

void SynTrainingTestInfra::setActualSizes(unsigned tensorIndex, const unsigned* tensorSizes, unsigned graphIndex)
{
    auto sizesExtVec = getVector(tensorSizes, getTensorDescriptor(tensorIndex).m_dims);
    setActualSizes(tensorIndex, getVectorRawPtrOrNull(sizesExtVec), graphIndex);
}

void SynTrainingTestInfra::setActualSizes(unsigned tensorIndex, const TSize* tensorSizes, unsigned graphIndex)
{
    // The tensor can be input or output.
    for (auto& pair : getGraph(graphIndex).m_inputEnqueueTensorsExt)
    {
        if (pair.first == tensorIndex)
        {
            for (int i = 0; i < getTensorDescriptor(tensorIndex).m_dims; i++)
            {
                pair.second.tensorSize[i] = tensorSizes[i];
            }
            break;
        }
    }
    for (auto& pair : getGraph(graphIndex).m_outputEnqueueTensorsExt)
    {
        if (pair.first == tensorIndex)
        {
            for (int i = 0; i < getTensorDescriptor(tensorIndex).m_dims; i++)
            {
                pair.second.tensorSize[i] = tensorSizes[i];
            }
            break;
        }
    }
    for (auto& pair : getGraph(graphIndex).m_nonDeviceShapeTensorsExt)
    {
        if (pair.first == tensorIndex)
        {
            for (int i = 0; i < getTensorDescriptor(tensorIndex).m_dims; i++)
            {
                pair.second.tensorSize[i] = tensorSizes[i];
            }
            break;
        }
    }

    auto to_rm = std::remove(getGraph(graphIndex).m_tensorsMissingActualSize.begin(),
                             getGraph(graphIndex).m_tensorsMissingActualSize.end(),
                             tensorIndex);

    getGraph(graphIndex).m_tensorsMissingActualSize.erase(to_rm, getGraph(graphIndex).m_tensorsMissingActualSize.end());
}

void SynTrainingTestInfra::setAsInternalShapeTensor(unsigned tensorIndex, unsigned graphIndex)
{
    auto to_rm = std::remove(m_graphs[graphIndex].m_tensorsMissingActualSize.begin(),
                             m_graphs[graphIndex].m_tensorsMissingActualSize.end(),
                             tensorIndex);

    m_graphs[graphIndex].m_tensorsMissingActualSize.erase(to_rm, m_graphs[graphIndex].m_tensorsMissingActualSize.end());
}

void SynTrainingTestInfra::pushGlobalConf(const std::string& name, const std::string& value)
{
    m_globalConfStack.emplace_front(new GlobalConfTestSetter(name, value));
}

synRecipeHandle SynTrainingTestInfra::getRecipeHandle(unsigned graphIndex)
{
    GraphData& graphData = getGraph(graphIndex);

    return graphData.recipeHandle;
}

void validateResults(const synTensorDescriptor& desc, char* firstData, char* secondData)
{
    unsigned length = multiplyElements(desc.m_sizes, desc.m_sizes + desc.m_dims);
    if (desc.m_dataType == syn_type_bf16)
    {
        validateResult((bfloat16*)firstData, (bfloat16*)secondData, length);
    }
    else if (desc.m_dataType == syn_type_float)
    {
        validateResult((float*)firstData, (float*)secondData, length);
    }
    else
    {
        assert(0 && "unsupported type");
    }
}

SynTrainingTestInfra::GraphData& SynTrainingTestInfra::getGraph(unsigned graphIndex)
{
    if (graphIndex >= m_graphs.size())
    {
        throw std::out_of_range("graph index is out of range, index: " + std::to_string(graphIndex));
    }
    return m_graphs[graphIndex];
}

bool SynTrainingTestInfra::compareRecipes(const recipe_t& recipe1, const recipe_t& recipe2, bool compareNames)
{
    return RecipeCompare::compare(recipe1, recipe2, compareNames);
}

void SynTrainingTestInfra::getTensorSectionId(const synRecipeHandle& recipeHandle,
                                              const synTensor& tensor,
                                              synSectionId& sectionId)
{
    uint32_t   numOfTensors = 0;
    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchAmount(recipeHandle, &numOfTensors));

    uint64_t ids[numOfTensors];
    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchIds(recipeHandle, ids, numOfTensors));

    synRetrievedLaunchTensorInfoExt tensorInfos[numOfTensors];
    for (unsigned i = 0; i < numOfTensors; i++)
    {
        tensorInfos[i].tensorId = ids[i];
    }
    ASSERT_EQ(synSuccess, synTensorRetrieveLaunchInfoByIdExt(recipeHandle, numOfTensors, tensorInfos));

    // get tensor name
    char tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
    synTensorGetName(tensor, ENQUEUE_TENSOR_NAME_MAX_SIZE, tensorName);

    // search for tensor according to tensor name and set it's sectionId
    for (unsigned tensorIdx = 0; tensorIdx < numOfTensors; tensorIdx++)
    {
        if (strcmp(tensorInfos[tensorIdx].tensorName, tensorName) == 0)
        {
            sectionId = tensorInfos[tensorIdx].tensorSectionId;
            return;
        }
    }
    sectionId = INVALID_SECTION_ID;
}

bool SynTrainingTestInfra::checkIfTensorInConstSection(synTensor tensor, unsigned graphIndex)
{
    // Const section is also persistent
    const auto& graphData = getGraph(graphIndex);
    const auto it = graphData.tensorCreationParams.find(tensor);
    HB_ASSERT(it != graphData.tensorCreationParams.end(), "Expecting tensor has creation params in graph data");
    const auto tensorCreationParams = it->second;
    if (tensorCreationParams.isConst && tensorCreationParams.isPersistent)
    {
        // now checking whether the tensor is in const section
        const auto sectionHandle = m_persistentSections[tensorCreationParams.concreteSectionIndex].handle;
        return checkIfConstSection(sectionHandle);
    }
    return false;
}

void SynTrainingTestInfra::initializeConstSection(const synRecipeHandle recipeHandle,
                                                  const synSectionId    tensorSectionId,
                                                  uint64_t&       deviceSectionBuffer)
{
    uint64_t sectionSize = 0, hostSectionData = 0;
    // when synRecipeSectionGetProp called with SECTION_SIZE it puts the size of the data in sectionSize.
    ASSERT_EQ(synRecipeSectionGetProp(recipeHandle, tensorSectionId, SECTION_SIZE, &sectionSize), synSuccess)
        << "Failed to get size of the data in the section";

    if (sectionSize == 0) return;

    // when synRecipeSectionGetProp called with SECTION_DATA it puts the address of the data in
    // hostSectionData.
    ASSERT_EQ(synRecipeSectionGetProp(recipeHandle, tensorSectionId, SECTION_DATA, &hostSectionData), synSuccess)
        << "Failed to get virtual address of the data in the section";

    // allocate memory on DRAM for the const section
    ASSERT_EQ(synDeviceMalloc(_getDeviceId(), sectionSize, 0, 0, &deviceSectionBuffer), synSuccess)
        << "Failed to malloc memory on the device";

    ASSERT_EQ(synHostMap(_getDeviceId(), sectionSize, (void*)hostSectionData), synSuccess)
        << "Failed to map between virtual to physical address on the host";

    downloadDataToDevice(hostSectionData, sectionSize, deviceSectionBuffer);

    // it might be that different tests have the same hostSectionData (same virtual
    // address to the section's data), while they also have different sectionSize, this will cause synHostMap to fail.
    // The call to synHostUnmap removes the hostSectionData from the container which stores the virtual addresses.
    ASSERT_EQ(synHostUnmap(_getDeviceId(), (void*)hostSectionData), synSuccess)
        << "Failed to unmap between virtual to physical address on the host";
}

void SynTrainingTestInfra::initializeAllConstSections(unsigned graphIndex)
{
    const auto&                                                     graphData    = getGraph(graphIndex);
    synRecipeHandle                                                 recipeHandle = graphData.recipeHandle;
    std::unordered_map<unsigned, std::pair<synSectionId, uint64_t>> sectionIndexToIdAndDeviceBuffer;
    for (const auto& tensor : m_tensors)
    {
        // check if tensors belongs to graph
        const auto it1 = graphData.tensorCreationParams.find(tensor);
        if (it1 == graphData.tensorCreationParams.end()) continue;
        const auto& tensorCreationParams = it1->second;

        // check if tensor in const section
        if (!checkIfTensorInConstSection(tensor, graphIndex)) continue;

        uint64_t     deviceSectionBuffer = 0;
        synSectionId currTensorSectionId = INVALID_SECTION_ID;
        getTensorSectionId(recipeHandle, tensor, currTensorSectionId);
        ASSERT_NE(currTensorSectionId, INVALID_SECTION_ID) << "Expecting tensor const section id is valid";

        // check if const section already initialized
        const auto it2 = sectionIndexToIdAndDeviceBuffer.find(tensorCreationParams.concreteSectionIndex);
        if (it2 != sectionIndexToIdAndDeviceBuffer.end())
        {
            synSectionId sectionId = it2->second.first;
            // verify tensor section ids of the same const section are equal
            ASSERT_EQ(currTensorSectionId, sectionId)
                << "Expecting tensors in the same const section have the same section id";
            deviceSectionBuffer = it2->second.second;
        }
        else
        {
            // validate that the section is const.
            uint64_t isConst = 0;
            ASSERT_EQ(synRecipeSectionGetProp(recipeHandle, currTensorSectionId, IS_CONST, &isConst), synSuccess)
                << "Failed to get the constness of the section";
            ASSERT_EQ(isConst, 1) << "section is not const!";

            initializeConstSection(recipeHandle, currTensorSectionId, deviceSectionBuffer);
            sectionIndexToIdAndDeviceBuffer.emplace(tensorCreationParams.concreteSectionIndex,
                                                    std::make_pair(currTensorSectionId, deviceSectionBuffer));
        }
        // get tensor name
        char tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
        memset(tensorName, 0, ENQUEUE_TENSOR_NAME_MAX_SIZE);
        synTensorGetName(tensor, ENQUEUE_TENSOR_NAME_MAX_SIZE, tensorName);

        setConstSectionTensorInfo({tensorName, deviceSectionBuffer + tensorCreationParams.offset});
    }
}

/*DEPRECATED*/
void SynTrainingTestInfra::setGraphAttributes(synGraphAttribute* attributes,
                                              uint64_t*          values,
                                              uint32_t           size,
                                              unsigned           graphIndex)
{
    auto& graphData = getGraph(graphIndex);
    ASSERT_EQ(synSuccess, synGraphSetAttribute(graphData.graphHandle, attributes, values, size));
}

void SynTrainingTestInfra::setGraphAttributesV2(synGraphAttribute*    attributes,
                                                synGraphAttributeVal* values,
                                                uint32_t              size,
                                                unsigned              graphIndex)
{
    auto& graphData = getGraph(graphIndex);
    ASSERT_EQ(synSuccess, synGraphSetAttributes(graphData.graphHandle, attributes, values, size));
}

void SynTrainingTestInfra::setGraphInferenceMode(unsigned graphIndex)
{
    GraphAttributesVec attributes = {GRAPH_ATTRIBUTE_INFERENCE};
    synGraphAttributeVal values[] = {1};
    setGraphAttributesV2(attributes.data(), values, 1, graphIndex);
}

void SynTrainingTestInfra::setGraphInferenceModeAndQuantizationEnabled(unsigned graphIndex)
{
    GraphAttributesVec attributes = {GRAPH_ATTRIBUTE_INFERENCE, GRAPH_ATTRIBUTE_QUANTIZATION};
    std::vector<uint64_t> values  = {1, 1};
    setGraphAttributes(attributes.data(), values.data(), values.size(), graphIndex);
}

void SynTrainingTestInfra::setTensorQuantizationData(size_t                  tensorIndex,
                                                     synQuantizationProperty prop,
                                                     void*                   propVal,
                                                     uint64_t                propSize)
{
    ASSERT_EQ(synSuccess, synTensorSetQuantizationData(getTensorByIndex(tensorIndex), prop, propVal, propSize));
}

// RelativeError = norm(Reference-Actual) / norm(Reference)
// norm(array) = sqrt(sum(array[i]^2))
// For fp8_143: Expected RelativeError = 0.125
// For fp8_152: Expected RelativeError = 0.25
// return true if: calculated relative error is less than expected relative error.
bool SynTrainingTestInfra::compareFP8Results(const float* refOutput,
                                             const float* actOutput,
                                             unsigned     numElems,
                                             synDataType  dType,
                                             std::string& errMsg)
{
    float expectedRelativeError = 0.00001;
    if (dType == syn_type_fp8_152)
    {
        expectedRelativeError = 0.25;
    }
    else if (dType == syn_type_fp8_143)
    {
        expectedRelativeError = 0.125;
    }

    float normRef  = 0;
    float normDiff = 0;
    for (uint64_t i = 0; i < numElems; i++)
    {
        float refVal  = refOutput[i];
        float diffVal = refOutput[i] - actOutput[i];
        normRef += refVal * refVal;
        normDiff += diffVal * diffVal;
    }
    normRef  = sqrt(normRef);
    normDiff = sqrt(normDiff);
    if (normDiff / normRef < expectedRelativeError)
    {
        return true;
    }

    errMsg = fmt::format("relativeError = {}, expectedRelativeError={}, relativeError >= expectedRelativeError",
                         normDiff / normRef,
                         expectedRelativeError);
    return false;
}

TEST_F_GC(SynGaudiTestInfra, spill_persistent_tensor_test, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    ScopedConfigurationChange enableSpillPersistentTensorsPass("SPILL_PERSISTENT_TENSORS", "true");

    const unsigned sectionIdx = createSection(4);

    unsigned tensor_0_sizes[] = {1024};
    unsigned tensor_0         = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_0_sizes, 1);

    unsigned tensor_2_sizes[] = {1};
    unsigned tensor_2         = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_2_sizes, 1);

    unsigned tensor_3_sizes[] = {1};
    unsigned tensor_3 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_3_sizes, 1, syn_type_int32);

    unsigned param = 0;
    addNodeToGraph("reduce_max_fwd_f32", {tensor_0}, {tensor_2, tensor_3}, &param, sizeof(param), "reduce_max_fwd_f32");

    unsigned tensor_9_sizes[] = {1};
    unsigned tensor_9         = createPersistTensor(OUTPUT_TENSOR,
                                            MEM_INIT_ALL_ZERO,
                                            nullptr,
                                            tensor_9_sizes,
                                            1,
                                            syn_type_single,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            &sectionIdx);

    addNodeToGraph("identity", {tensor_2}, {tensor_9}, nullptr, 0, "identity");

    unsigned tensor_8_sizes[] = {1};
    unsigned tensor_8 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_8_sizes, 1, syn_type_int8);

    unsigned tensor_11_sizes[] = {1};
    unsigned tensor_11         = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_11_sizes, 1);

    unsigned tensor_10_sizes[] = {1};
    unsigned tensor_10         = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             tensor_10_sizes,
                                             1,
                                             syn_type_single,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             &sectionIdx);

    addNodeToGraph("where_fwd_f32", {tensor_8, tensor_11, tensor_9}, {tensor_10}, nullptr, 0, "where_f32");

    unsigned tensor_12_sizes[] = {1};
    unsigned tensor_12         = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, tensor_12_sizes, 1);

    synStridedOpParams stridedOpParams;
    stridedOpParams.baseOffset = 0;
    stridedOpParams.strides[0] = 1;

    addNodeToGraph("strided_insert",
                   {tensor_2, tensor_10},
                   {tensor_12},
                   &stridedOpParams,
                   sizeof(stridedOpParams),
                   "strided_insert");

    compileTopology();
}
