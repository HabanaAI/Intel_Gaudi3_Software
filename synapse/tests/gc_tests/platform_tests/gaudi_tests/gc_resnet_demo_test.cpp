#include "gc_gaudi_test_infra.h"
#include "log_manager.h"
#include "gc_resnet_demo_test.h"
#include "synapse_api.h"
#include <future>
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "habana_global_conf.h"

SynTrainingResNetTest::TensorInfo::TensorInfo(unsigned               dims,
                                              synDataType            dataType,
                                              uint64_t               tensorSize,
                                              bool                   isPresist,
                                              std::string            tensorName,
                                              synTensor              originalTensor,
                                              SynTrainingResNetTest* test,
                                              bool                   isIntermediate)
:

  m_dims(dims),
  m_dataType(dataType),
  m_tensorSize(tensorSize),
  m_isPresist(isPresist),
  m_isInput(false),
  m_tensorName(tensorName),
  m_pOriginalTensor(originalTensor),
  m_pTest(test),
  m_isIntermediate(isIntermediate)
{}

void SynTrainingResNetTest::TensorInfo::setIsInput(bool isInput)
{
    if (!m_isPresist)
    {
        assert(false);
    }

    m_isInput = isInput;
}

void SynTrainingResNetTest::TensorInfo::print(unsigned int deviceId, FILE* fd) const
{
    if (!m_isPresist)
    {
        return;
    }

    uint64_t address = m_pTest->m_dramMap[m_tensorName];
    fprintf(fd, "\n");
    fprintf(fd,
            "Tensor - Name: %s, Type: %s, Data Type: %u bits, Num Elements: %lu, Address: 0x%016lx\n",
            m_tensorName.c_str(),
            m_isInput          ? "input"
            : m_isIntermediate ? "intermediate"
                               : "output",
            getElementSizeInBytes(m_dataType) * 8,
            m_tensorSize,
            address);
    if (m_tensorSize > c_max_tensor_size_to_dump)
    {
        fprintf(fd, "Tensor is too big. SKIPPED.\n");
    }
    else if (m_dataType == syn_type_bf16)
    {
        void* data = nullptr;
        synHostMalloc(deviceId, m_tensorSize * 2, 0, &data);
        m_pTest->uploadTensorData(address, data, m_tensorSize * 2);
        for (unsigned i = 0; i < m_tensorSize; i++)
        {
            union
            {
                float    f;
                uint16_t u16[2];
            } val;

            val.u16[0] = 0;
            val.u16[1] = ((uint16_t*)data)[i];

            fprintf(fd,
                    "Address 0x%016lx, Element Offset: %05u, Hex: 0x%04x, Val: %f\n",
                    address + (i * 2),
                    i,
                    val.u16[1],
                    val.f);
        }
        synHostFree(deviceId, data, 0);
    }
    else if (m_dataType == syn_type_single)
    {
        void* data = nullptr;
        synHostMalloc(deviceId, m_tensorSize * 4, 0, &data);
        m_pTest->uploadTensorData(address, data, m_tensorSize * 4);
        for (unsigned i = 0; i < m_tensorSize; i++)
        {
            union
            {
                float    f;
                uint32_t u32;
            } val;

            val.f = ((float*)data)[i];

            fprintf(fd,
                    "Address 0x%016lx, Element Offset: %05u, Hex: 0x%08x, Val: %f\n",
                    address + (i * 4),
                    i,
                    val.u32,
                    val.f);
        }
        synHostFree(deviceId, data, 0);
    }
    else
    {
        fprintf(fd, "Element type not supported (%u). SKIPPED.\n", m_dataType);
    }
}

bool SynTrainingResNetTest::TensorInfo::initFromFile(unsigned int deviceId, std::string basePath)
{
    if (!m_isPresist || !m_isInput)
        return true;

    if (m_tensorName.find("_wu") != std::string::npos)
    {
        return true;
    }

    void*     data       = nullptr;
    size_t    bufferSize = alignSizeToCL(m_tensorSize * getElementSizeInBytes(m_dataType));
    synStatus status     = synHostMalloc(deviceId, bufferSize, 0, &data);
    bool      file_res   = true;

    // When running on PLDM, reading uninitialized memory may cause HW interrupts since the mem is not scrubbed.
    // Allocating each tensor in CL alignment and initializing all the buffer to 0 should prevent this.
    std::fill((uint32_t*)data, ((uint32_t*)data) + (bufferSize / sizeof(uint32_t)), uint32_t(0));

    switch (m_dataType)
    {
        case syn_type_bf16:
        {
            bfloat16* typed_data = static_cast<bfloat16*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_single:
        {
            float* typed_data = static_cast<float*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_int8:
        {
            uint8_t* typed_data = static_cast<uint8_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_int16:
        {
            int16_t* typed_data = static_cast<int16_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_int32:
        {
            int32_t* typed_data = static_cast<int32_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_uint8:
        {
            uint8_t* typed_data = static_cast<uint8_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case syn_type_uint16:
        {
            uint16_t * typed_data = static_cast<uint16_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
        case  syn_type_uint32:
        {
            uint32_t* typed_data = static_cast<uint32_t*>(data);
            file_res = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
        }
            break;
            case syn_type_fp8_152:
            {
                fp8_152_t* typed_data = static_cast<fp8_152_t*>(data);
                file_res           = read_file(basePath + m_tensorName, typed_data, m_tensorSize);
            }
            break;
            default:
                return false;
    }
    if(!file_res)
        return false;

    m_pTest->downloadTensorData(data, m_pTest->m_dramMap[m_tensorName], bufferSize);
    status = synHostFree(deviceId, data, 0);
    return status == synSuccess;
}

void SynTrainingResNetTest::TensorInfo::uploadAndCheck(unsigned int deviceId, synStreamHandle streamHandle)
{
    LOG_TRACE(SYN_TEST, "{}: calling memcpyAsync for tensor {}", HLLOG_FUNC, m_tensorName);
    synStatus status = synMemCopyAsync(streamHandle,
                                       m_pTest->m_dramMap[m_tensorName],
                                       m_tensorSize * getElementSizeInBytes(m_dataType),
                                       (uint64_t)m_data,
                                       DRAM_TO_HOST);

    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";
    LOG_TRACE(SYN_TEST, "{}: calling streamSynchronize for tensor {}", HLLOG_FUNC, m_tensorName);

    status = synStreamSynchronize(streamHandle);
    LOG_TRACE(SYN_TEST, "{}: called streamSynchronize for tensor {}", HLLOG_FUNC, m_tensorName);

    ASSERT_EQ(status, synSuccess) << "synStreamSynchronize Failed";
    if (m_ref_arr != nullptr && m_data != nullptr)
    {
        switch (m_dataType)
        {
            case syn_type_bf16:
            {
                bfloat16* typed_data = static_cast<bfloat16*>(m_data);
                ::validateResult((bfloat16*)m_ref_arr, typed_data, m_tensorSize, m_tensorName);
                delete[](bfloat16*) m_ref_arr;
            }
            break;
            case syn_type_float:
            {
                float* typed_data = static_cast<float*>(m_data);
                ::validateResult((float*)m_ref_arr, typed_data, m_tensorSize, m_tensorName);
                delete[](bfloat16*) m_ref_arr;
            }
            break;
            default:
                // currently only float and bfloat are supported
                break;
        }
        ASSERT_EQ(synHostFree(deviceId, m_data, 0), synSuccess);
        m_ref_arr = nullptr;
        m_data    = nullptr;
    }
    else
    {
        LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: {}", m_tensorName);
    }
}
bool SynTrainingResNetTest::TensorInfo::allocateAndPrepareReference(unsigned int    deviceId,
                                                                 std::string     basePath,
                                                                 synStreamHandle streamHandle)
{
    if (!m_isPresist || m_isInput)  // make sure this is external TODO
    {
        return true;
    }
    synStatus status = synSuccess;
    if (m_data == nullptr)
    {
        status = synHostMalloc(deviceId, m_tensorSize * getElementSizeInBytes(m_dataType), 0, &m_data);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_TEST, "unable to allocate check buffer, status {}", status);
            return false;
        }
    }
    std::memset((void*)m_data, 0, m_tensorSize * getElementSizeInBytes(m_dataType));
    status = synMemCopyAsync(streamHandle,
                             (uint64_t)m_data,
                             m_tensorSize * getElementSizeInBytes(m_dataType),
                             (uint64_t)m_pTest->m_dramMap[m_tensorName],
                             HOST_TO_DRAM);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_TEST, "unable to memset check buffer, status {}", status);
        return false;
    }
    status = synStreamSynchronize(streamHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_TEST, "synStreamSynchronize failed while trying to memset buffer to 0");
        return false;
    }
    if (m_ref_arr == nullptr)
    {
        switch (m_dataType)
        {
            case syn_type_bf16:
            {
                m_ref_arr     = new bfloat16[m_tensorSize];
                bool file_res = read_file(basePath + m_tensorName, (bfloat16*)m_ref_arr, m_tensorSize);
                if (!file_res)
                {
                    LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: {}", m_tensorName);
                    return false;
                }
            }
            break;
            case syn_type_float:
            {
                m_ref_arr     = new float[m_tensorSize];
                bool file_res = read_file(basePath + m_tensorName, (float*)m_ref_arr, m_tensorSize);
                if (!file_res)
                {
                    LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: {}", m_tensorName);
                    return false;
                }
            }
            break;
            default:
                // currently only float and bfloat are supported
                break;
        }
    }
    return true;
}

bool SynTrainingResNetTest::TensorInfo::validateResult(unsigned int deviceId, std::string basePath) const
{
    if (!m_isPresist || m_isInput)
    {
        return true;
    }

    synStatus status;
    void* data = nullptr;
    status = synHostMalloc(deviceId, m_tensorSize*getElementSizeInBytes(m_dataType), 0, &data);

    if (status!=synSuccess)
        return false;

    m_pTest->uploadTensorData(m_pTest->m_dramMap[m_tensorName], data, m_tensorSize*getElementSizeInBytes(m_dataType));
    switch (m_dataType)
    {
        case syn_type_bf16:
        {
            bfloat16 *ref_arr = new bfloat16[m_tensorSize];
            bfloat16 *typed_data = static_cast<bfloat16 *>(data);
            bool      file_res   = read_file(basePath + m_tensorName, (bfloat16*)ref_arr, m_tensorSize);
            if (file_res)
            {
                ::validateResult(ref_arr, typed_data, m_tensorSize, m_tensorName);
            }
            else
            {
                LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: {}", m_tensorName);
            }
            delete[] ref_arr;
        }
            break;
        case syn_type_float:
        {
            float *ref_arr = new float[m_tensorSize];
            float *typed_data = static_cast<float *>(data);
            bool   file_res   = read_file(basePath + m_tensorName, (float*)ref_arr, m_tensorSize);
            if (file_res)
            {
                ::validateResult(ref_arr, typed_data, m_tensorSize, m_tensorName);
            }
            else
            {
                LOG_WARN(SYN_TEST, "Result compare skipped due to missing file: {}", m_tensorName);
            }
            delete[] ref_arr;
        }
            break;
        default:
            LOG_WARN(SYN_TEST, "currently only float and bfloat are supported for validation");
            break;
    }
    status = synHostFree(deviceId, data, 0);
    return (status == synSuccess);
}
bool SynTrainingResNetTest::TensorInfo::cleanup()
{
    if (m_ref_arr)
    {
        if (syn_type_float == m_dataType)
        {
            if (m_ref_arr)
            {
                delete[](float*) m_ref_arr;
            }
        }
        else
        {
            delete[](bfloat16*) m_ref_arr;
        }
        m_ref_arr = nullptr;
        m_data    = nullptr;
    }
    int status = 0;
    synSectionHandle section = nullptr;
    uint64_t offset = 0;
    synTensorGetSection(m_pOriginalTensor, &section, &offset);
    if (section)
    {
        status |= synSectionDestroy(section);
    }
    status |= synTensorDestroy(m_pOriginalTensor);

    return status == synSuccess;
}

synTensor SynTrainingResNetTest::createTensor(unsigned        dims,
                                           synDataType     data_type,
                                           const unsigned* tensor_size,
                                           bool            is_presist,
                                           const char*     name,
                                           synGraphHandle  graphHandle,
                                           uint64_t        deviceAddr)
{
    assert(deviceAddr == -1 || is_presist);

    synTensor tensor =
        SynGaudiAutoGenTest::createTensor(dims, data_type, tensor_size, is_presist, name, graphHandle, deviceAddr);

    uint64_t tensorSize = 1;
    for (unsigned i = 0; i < dims; ++i)
    {
        tensorSize *= tensor_size[dims - 1 - i];
    }

    // allowing intermediate tensor dump for debug purposes
    bool inter = !is_presist;
    if (inter && GCFG_DEBUG_MODE.value() == 3)
    {
        uint64_t allocSize = getElementSizeInBytes(data_type) * tensorSize;
        uint64_t addr;
        hbmAlloc(allocSize, &addr, name);
        is_presist = true;
        m_graphIntermediates.push_back({name, addr, DATA_TENSOR, {0, 0, 0, 0, 0}});
    }

    assert(m_tensorInfoMap.count(name) == 0);
    m_tensorInfoMap.emplace(name, TensorInfo(dims, data_type, tensorSize, is_presist, name, tensor, this, inter));
    m_tensorOrder.push_back(name);
    return tensor;
}

void SynTrainingResNetTest::executeTraining(const LaunchInfo&     launchInfo,
                                         const TensorInfoList& inputs,
                                         const TensorInfoList& outputs,
                                         bool                  skipValidation)
{
    // in GCFG_DEBUG_MODE <=3 m_graphIntermediates is empty so nothing happens
    TensorInfoList localOutputs = outputs;
    localOutputs.insert(localOutputs.begin(), m_graphIntermediates.begin(), m_graphIntermediates.end());

    for (auto& tensorLaunchInfo : inputs)
    {
        m_tensorInfoMap.at(tensorLaunchInfo.tensorName).setIsInput(true);
    }

    for (auto& mapElement : m_tensorInfoMap)
    {
        ASSERT_TRUE(mapElement.second.initFromFile(_getDeviceId(), m_pathPrefix));
    }

    SynGaudiAutoGenTest::executeTraining(launchInfo, inputs, localOutputs);

    if (GCFG_DEBUG_MODE.value() >= 2)
    {
        FILE* outputTensorDump = fopen("output_tensor_dump.txt", "w");
        FILE* inputTensorDump  = fopen("input_tensor_dump.txt", "w");
        for (const auto& tensorName : m_tensorOrder)
        {
            const auto& mapElement = m_tensorInfoMap.at(tensorName);
            if (GCFG_DEBUG_MODE.value() == 3 || (GCFG_DEBUG_MODE.value() == 2 && mapElement.isPersist()))
            {
                if (mapElement.isInput())
                {
                    mapElement.print(_getDeviceId(), inputTensorDump);
                }
                else
                {
                    mapElement.print(_getDeviceId(), outputTensorDump);
                }
            }
        }
        fclose(outputTensorDump);
    }

    if (!skipValidation)  // TODO SW-59132 - enable validation for fp8 resnet tests
    {
        for (const auto& mapElement : m_tensorInfoMap)
        {
            ASSERT_TRUE(mapElement.second.validateResult(_getDeviceId(), m_pathPrefix));
            if (GCFG_DEBUG_MODE.value() > 0)
            {
                if (HasFailure()) break;
            }
        }
    }
}

void SynTrainingResNetTest::waitUploadCheck(synEventHandle* eventHandle, TensorInfo* tensorInfo)
{
    ASSERT_EQ(synSuccess, synStreamWaitEvent(m_streamHandleUpload, *eventHandle, 0));
    tensorInfo->uploadAndCheck(_getDeviceId(), m_streamHandleUpload);
}

void SynTrainingResNetTest::executeTrainingRuntime(const LaunchInfo&     launchInfo,
                                                const TensorInfoList& tensors,
                                                synEventHandle*       eventHandles,
                                                size_t                numOfEvents,
                                                size_t                externalTensorsIndices[])
{
    for (auto& mapElement : m_tensorInfoMap)
    {
        ASSERT_TRUE(mapElement.second.initFromFile(_getDeviceId(), m_pathPrefix));
        if (nullptr != eventHandles)
        {
            ASSERT_TRUE(
                mapElement.second.allocateAndPrepareReference(_getDeviceId(), m_pathPrefix, m_streamHandleDownload));
        }
    }
    ASSERT_EQ(synSuccess,
              synLaunchWithExternalEvents(m_streamHandleCompute,
                                          tensors.data(),
                                          tensors.size(),
                                          launchInfo.m_workspaceAddr,
                                          launchInfo.m_recipeHandle,
                                          eventHandles,
                                          numOfEvents,
                                          0));
    if (nullptr != eventHandles)
    {
        std::future<void> futureArr[numOfEvents];
        for (size_t eventIdx = 0; eventIdx < numOfEvents; eventIdx++)
        {
            futureArr[eventIdx] =
                std::async(std::launch::async,
                           &SynTrainingResNetTest::waitUploadCheck,
                           this,
                           &eventHandles[eventIdx],
                           &m_tensorInfoMap.at(tensors.at(externalTensorsIndices[eventIdx]).tensorName));
        }
        for (size_t eventIdx = 0; eventIdx < numOfEvents; eventIdx++)
        {
            futureArr[eventIdx].wait();
        }
    }
    synStatus status = synStreamSynchronize(m_streamHandleCompute);
    ASSERT_EQ(synSuccess, status) << "Synchronization after compute failed!";
    for (const auto& mapElement : m_tensorInfoMap)
    {
        ASSERT_TRUE(mapElement.second.validateResult(_getDeviceId(), m_pathPrefix));
        if (GCFG_DEBUG_MODE.value() > 0)
        {
            if (HasFailure()) break;
        }
    }
}

void SynTrainingResNetTest::cleanup()
{
    for (auto& mapElement : m_tensorInfoMap)
    {
        ASSERT_TRUE(mapElement.second.cleanup());
    }
    for (const auto& mappedAddress : m_dramMap)
    {
        if (mappedAddress.first.find("_wu") == std::string::npos)
        {
            hbmFree(mappedAddress.second, mappedAddress.first.c_str());
        }
    }
    clearDramMap();
}

void SynTrainingResNetTest::SetUpTest()
{
    SynGaudiAutoGenTest::SetUpTest();
    const char* envSoftwareLfsData = std::getenv("SOFTWARE_LFS_DATA");
    ASSERT_TRUE(envSoftwareLfsData) << "SOFTWARE_LFS_DATA is not set!";
    std::string softwareLfsData = envSoftwareLfsData;
    m_pathPrefix                = softwareLfsData.append("/demos/gaudi/functional/");
}

SynTrainingResNetTest::SynTrainingResNetTest()
{
    ReleaseDevice();
    setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
}