#include "gc_gaudi_test_infra.h"
#include "gc_autogen_test.h"
#include "synapse_api.h"
#include "infra/gc_test_configuration.h"

#define SYN_DO(cmd_)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        synStatus status_ = (cmd_);                                                                                    \
        if (unlikely(status_ != synSuccess))                                                                           \
        {                                                                                                              \
            LOG_ERR(SYN_TEST, "{}:{} \"{}\" returned synStatus {}", __FILE__, __LINE__, #cmd_, status_);               \
            assert(0);                                                                                                 \
        }                                                                                                              \
    } while (false)

void SynGaudiAutoGenTest::clearDramMap()
{
    m_dramMap.clear();
}

template<>
bool read_file(const std::string& file_name, bfloat16* output, uint32_t num_of_elements)
{
    // File is in float, needs to convert it to bfloat16
    float* temp_output = new float[num_of_elements];
    bool   file_res    = read_file(file_name, temp_output, num_of_elements);
    if (file_res)
    {
        for (uint32_t idx = 0; idx < num_of_elements; ++idx)
        {
            output[idx] = bfloat16(temp_output[idx]);
        }
    }
    delete[] temp_output;
    return file_res;
}

template<>
bool read_file(const std::string& file_name, fp8_152_t* output, uint32_t num_of_elements)
{
    // File is in float, needs to convert it to fp8_152_t
    float* temp_output = new float[num_of_elements];
    bool   file_res    = read_file(file_name, temp_output, num_of_elements);
    if (file_res)
    {
        for (uint32_t idx = 0; idx < num_of_elements; ++idx)
        {
            output[idx] = fp8_152_t(temp_output[idx]);
        }
    }
    delete[] temp_output;
    return file_res;
}

template<>
bool read_file(const std::string& file_name, int16_t* output, uint32_t num_of_elements)
{
    std::ifstream file(file_name);
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        uint32_t length = file.tellg();
        if (length == num_of_elements * sizeof(int16_t))
        {
            file.seekg(0, std::ios::beg);
            file.read((char*)output, length);
            file.close();
            return true;
        }
        // in case we need to convert from int8
        else if (length == num_of_elements * sizeof(uint8_t))
        {
            uint8_t* tmpOutput = new uint8_t[num_of_elements];
            file.seekg(0, std::ios::beg);
            file.read((char*)tmpOutput, length);
            file.close();
            for (uint32_t i = 0; i < num_of_elements; i++)
            {
                output[i] = (int16_t)tmpOutput[i];
            }
            delete[] tmpOutput;
        }
        else
        {
            file.close();
            LOG_ERR(SYN_TEST,
                    "File '{}' unexpected length. Expected: {}, actual: {}.",
                    file_name,
                    num_of_elements * sizeof(uint16_t),
                    length);
            return false;
        }
    }
    else
    {
        LOG_ERR(SYN_TEST, "File '{}' doesn't exist", file_name);
        return false;
    }
    return true;
}

void SynGaudiAutoGenTest::SetUpTest()
{
    SynGaudiTestInfra::SetUpTest();
}

void SynGaudiAutoGenTest::TearDownTest()
{
    SynGaudiTestInfra::TearDownTest();
}

synTensor SynGaudiAutoGenTest::createTensor(unsigned        dims,
                                            synDataType     data_type,
                                            const unsigned* tensor_size,
                                            bool            is_presist,
                                            const char*     name,
                                            synGraphHandle  graphHandle,
                                            uint64_t        deviceAddr)
{
    assert(deviceAddr == -1 || is_presist);

    synSectionHandle sectionHandle = nullptr;
    if (is_presist)
    {
        // currently this infra is not used. when we enable it the section
        // create API needs the graph handle
        SYN_DO(synSectionCreate(&sectionHandle, 0, graphHandle));
        SYN_DO(synSectionSetPersistent(sectionHandle, true));
    }

    synTensor tensor;
    {
        synTensorDescriptor desc;
        desc.m_dataType = data_type;
        desc.m_dims     = dims;
        desc.m_name     = name;
        memset(desc.m_strides, 0, sizeof(desc.m_strides));
        for (unsigned i = 0; i < dims; ++i)
        {
            desc.m_sizes[i] = tensor_size[dims - 1 - i];
        }

        SYN_DO(synTensorCreate(&tensor, &desc, sectionHandle, 0));
    }

    return tensor;
}

synStatus SynGaudiAutoGenTest::hbmAlloc(uint64_t size, uint64_t* addr, const char* name)
{
    if (m_dramMap.find(std::string(name)) != m_dramMap.end())
    {
        *addr = m_dramMap[std::string(name)];
        return synSuccess;
    }
    // We want to write 0 to any uninitialized suffix that is caused by the HW reading full CLs, in order to avoid HW
    // errors on PLDM. Allocate aligned to CL to make sure we are allowed.
    size             = alignSizeToCL(size);
    synStatus status = synDeviceMalloc(_getDeviceId(), size, 0, 0, addr);

    m_dramMap[name]                          = *addr;
    m_dramMap[std::string(name) + "_wu"]     = *addr;
    m_dramMap[std::string(name) + "_wu_out"] = *addr;

    return status;
}

synStatus SynGaudiAutoGenTest::hbmFree(uint64_t addr, const char* name)
{
    return synDeviceFree(_getDeviceId(), addr, 0);
}

void SynGaudiAutoGenTest::downloadTensorData(void* data, uint64_t tensorAddr, unsigned sizeBytes)
{
    memcpyTensorData((uint64_t)data, tensorAddr, sizeBytes, HOST_TO_DRAM);
}

void SynGaudiAutoGenTest::uploadTensorData(uint64_t tensorAddr, void* data, unsigned sizeBytes)
{
    memcpyTensorData(tensorAddr, (uint64_t)data, sizeBytes, DRAM_TO_HOST);
}

void SynGaudiAutoGenTest::memcpyTensorData(uint64_t src, uint64_t dst, unsigned sizeBytes, synDmaDir direction)
{
    synStatus       status;
    bool            destroyStream = false;
    synStreamHandle streamHandle = 0;

    if (direction == HOST_TO_DRAM)
    {
        streamHandle = m_streamHandleDownload;
    }
    else
    {
        streamHandle = m_streamHandleUpload;
    }

    if (streamHandle == 0)
    {
        status = synStreamCreateGeneric(&streamHandle, _getDeviceId(), 0);
        ASSERT_EQ(status, synSuccess) << "Failed to create download stream";
        destroyStream = true;
    }

    synEventHandle eventHandle;
    status = synEventCreate(&eventHandle, _getDeviceId(), 0);
    ASSERT_EQ(status, synSuccess) << "Failed create event";

    status = synMemCopyAsync(streamHandle, src, sizeBytes, dst, direction);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";

    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "synStreamSynchronize Failed";

    status = synEventDestroy(eventHandle);
    ASSERT_EQ(status, synSuccess) << "synStreamSynchronize Failed";

    if (destroyStream)
    {
        status = synStreamDestroy(streamHandle);
        ASSERT_EQ(status, synSuccess) << "synStreamSynchronize Failed";
    }
}

SynGaudiAutoGenTest::LaunchInfo SynGaudiAutoGenTest::compileAllocateAndLoadGraph(synGraphHandle graphHandle)
{
    LaunchInfo launchInfo;
    synStatus  status;
    UNUSED(status);

    LOG_DEBUG(SYN_TEST, "Compiling {}...", GetTestFileName().c_str());

    status = synGraphCompile(&launchInfo.m_recipeHandle, graphHandle, GetTestFileName().c_str(), nullptr);

    HB_ASSERT(status == synSuccess, "Failed on synGraphCompile");

    uint64_t workspaceSize;
    launchInfo.m_workspaceSize = 0;
    status                     = synWorkspaceGetSize(&workspaceSize, launchInfo.m_recipeHandle);

    HB_ASSERT(status == synSuccess, "Failed to get workspace size");

    launchInfo.m_workspaceSize = workspaceSize;

    if (workspaceSize)
    {
        // TODO: assert(m_testConfig.m_compilationMode != COMP_EAGER_MODE_TEST && "Eager tests should NOT require a
        // workspace");
        status = synDeviceMalloc(_getDeviceId(), workspaceSize, 0, 0, &launchInfo.m_workspaceAddr);

        HB_ASSERT(status == synSuccess, "Memory allocation for workspace failed");
    }
    LOG_DEBUG(SYN_TEST, "Loading recipe {}...", GetTestFileName().c_str());
    return launchInfo;
}

void SynGaudiAutoGenTest::executeTraining(const LaunchInfo&     launchInfo,
                                          const TensorInfoList& inputs,
                                          const TensorInfoList& outputs,
                                          bool                  skipValidation)
{
    synStatus       status;
    synStreamHandle computeStream = m_streamHandleCompute;
    bool            destroyStream = false;

    if (computeStream == 0)
    {
        status = synStreamCreateGeneric(&computeStream, _getDeviceId(), 0);
        ASSERT_EQ(synSuccess, status) << "Failed to create compute stream!";
        destroyStream = true;
    }

    TensorInfoList concatTensors(inputs);
    concatTensors.insert(std::end(concatTensors), std::begin(outputs), std::end(outputs));
    uint32_t totalNumOfTensors = concatTensors.size();

    prepareTensorInfo(launchInfo.m_recipeHandle, &concatTensors[0], totalNumOfTensors);

    status = synLaunch(computeStream,
                       concatTensors.data(),
                       concatTensors.size(),
                       launchInfo.m_workspaceAddr,
                       launchInfo.m_recipeHandle,
                       0);
    ASSERT_EQ(synSuccess, status) << "Enqueue training failed!";

    status = synStreamSynchronize(computeStream);
    ASSERT_EQ(synSuccess, status) << "Synchronization after compute failed!";

    if (destroyStream)
    {
        status = synStreamDestroy(computeStream);
        ASSERT_EQ(synSuccess, status) << "Failed to destroy compute stream!";
    }
}

SynGaudiAutoGenTest::SynGaudiAutoGenTest()
{
    setTestPackage(TEST_PACKAGE_AUTOGEN);
}