#pragma once

#include <vector>
#include "assert.h"
#include "math_utils.h"
#include "string.h"
#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <thread>
#include <csignal>
#include <fstream>
#include <cmath>
#include <map>
#include <time.h>
#include "perf_lib_layer_params.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"


template<class T>
bool read_file(const std::string& file_name, T* output, uint32_t num_of_elements)
{
    std::ifstream file(file_name);
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        uint32_t length = file.tellg();
        if (length != num_of_elements * sizeof(T))
        {
            file.close();
            LOG_ERR(SYN_TEST,
                    "File '{}' unexpected length. Expected: {}, actual: {}.",
                    file_name,
                    num_of_elements * sizeof(T),
                    length);
            return false;
        }
        file.seekg(0, std::ios::beg);
        file.read((char*)output, length);
        file.close();
        return true;
    }
    else
    {
        LOG_ERR(SYN_TEST, "File '{}' doesn't exist", file_name);
        return false;
    }
}

template<>
bool read_file(const std::string& file_name, bfloat16* output, uint32_t num_of_elements);

template<>
bool read_file(const std::string& file_name, fp8_152_t* output, uint32_t num_of_elements);

template<>
bool read_file(const std::string& file_name, int16_t* output, uint32_t num_of_elements);

template<class Tfile, class Tbuff>
bool read_file_with_type(const std::string& file_name, Tbuff* output, uint32_t num_of_elements)
{
    Tfile* temp_output = new Tfile[num_of_elements];
    bool   file_res    = read_file(file_name, temp_output, num_of_elements);
    if (file_res)
    {
        for (uint32_t idx = 0; idx < num_of_elements; ++idx)
        {
            output[idx] = (Tbuff)(temp_output[idx]);
        }
    }
    delete[] temp_output;
    return file_res;
}

class SynGaudiAutoGenTest : public SynGaudiTestInfra
{
public:
    SynGaudiAutoGenTest();
    virtual synTensor createTensor(unsigned        dims,
                                   synDataType     data_type,
                                   const unsigned* tensor_size,
                                   bool            is_presist,
                                   const char*     name,
                                   synGraphHandle  graphHandle = nullptr,
                                   uint64_t        deviceAddr  = -1);

    synStatus hbmAlloc(uint64_t size, uint64_t* addr, const char* name);
    synStatus hbmFree(uint64_t addr, const char* name);

    static const size_t CL_SIZE = 128;
    static size_t       alignSizeToCL(size_t size) { return CL_SIZE * div_round_up(size, CL_SIZE); }

protected:
    static constexpr unsigned batchSize = 64;

    void SetUpTest() override;
    void TearDownTest() override;

    void downloadTensorData(void* data, uint64_t tensorAddr, unsigned sizeBytes);  // Host to Device
    void uploadTensorData(uint64_t tensorAddr, void* data, unsigned sizeBytes);    // Device to Host
    void memcpyTensorData(uint64_t src, uint64_t dst, unsigned sizeBytes, synDmaDir direction);
    void clearDramMap();
    struct LaunchInfo
    {
        synRecipeHandle m_recipeHandle  = nullptr;
        uint64_t        m_workspaceAddr = 0;
        uint64_t        m_workspaceSize;
    };

    using TensorInfoList = std::vector<synLaunchTensorInfo>;

    LaunchInfo compileAllocateAndLoadGraph(synGraphHandle graphHandle);
    virtual void
                                    executeTraining(const LaunchInfo&     launchInfo,
                                                    const TensorInfoList& inputs,
                                                    const TensorInfoList& outputs,
                                                    bool skipValidation = false);  // TODO SW-59132 - enable validation for fp8 resnet tests
    std::map<std::string, uint64_t> m_dramMap;
};
