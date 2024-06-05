#include "test_recipe_relu.hpp"
#include "scoped_configuration_change.h"
#include "synapse_api.h"
#include "../utils/cpu_calculator.h"
#include "tensor_validator.inl"
#include "test_tensors_container.hpp"
#include <iostream>
#include <vector>

TestRecipeRelu::TestRecipeRelu(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeRelu>(),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 1 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    unsigned       numOfDims = 4;
    const unsigned Z         = 64U;
    const unsigned W         = 112U;
    const unsigned H         = 112U;
    const unsigned batch     = 64U;
    //
    unsigned tensorDimSizes[SYN_MAX_TENSOR_DIM] = {Z, W, H, batch};  //{64, 112, 112, 64}
    //
    uint64_t tensorSize = Z * W * H * batch * sizeof(float);

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_fp16;
    m_tensorInfoVecInputs[0].m_tensorName = "input";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);

    // Init m_tensorInfoVecOutputs
    m_tensorInfoVecOutputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecOutputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecOutputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecOutputs[0].m_dataType   = syn_type_fp16;
    m_tensorInfoVecOutputs[0].m_tensorName = "output";
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[0].m_tensorDimsSize);

    const char* envSoftwareLfsData = std::getenv("SOFTWARE_LFS_DATA");
    if (envSoftwareLfsData)
    {
        std::string softwareLfsData = envSoftwareLfsData;
        m_pathPrefix                = softwareLfsData.append("/demos/gaudi/functional/");
    }
}

void TestRecipeRelu::validateResultsWithFile(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    // Init tensors data from files
    float* referenceData = new float[getTensorSizeOutput(0)];

    bool referenceFileReadStatus =
        read_file(m_pathPrefix + "worker_0_relu_output", referenceData, getTensorSizeOutput(0));
    ASSERT_TRUE(referenceFileReadStatus) << "Failed to read reference-file";

    ::validateResult(referenceData,
                     (float*)rLaunchTensorMemory.m_tensorInfoVecOutputs[0].getTestHostBuffer().getBuffer(),
                     getTensorSizeOutput(0) / sizeof(float));
}

void TestRecipeRelu::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    float* output = (float*)rLaunchTensorMemory.m_tensorInfoVecOutputs[0].getTestHostBuffer().getBuffer();
    float* input  = (float*)rLaunchTensorMemory.m_tensorInfoVecInputs[0].getTestHostBuffer().getBuffer();

    int elementsAmt = getTensorSizeOutput(0) / sizeof(float);

    for (int i = 0; i < elementsAmt; i++)
    {
        float expectedResult = std::max((float)0.0, *input);
        ASSERT_EQ(expectedResult, *output) << "Mismatch for at index " << i << " Expected:" << expectedResult
                                           << " Result: " << *output << " operand " << *input;
        input++;
        output++;
    }
}

void TestRecipeRelu::_graphCreation()
{
    synStatus status(synSuccess);

    // Tensors
    //
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         4U,
                         syn_type_single,
                         m_tensorInfoVecInputs[0].m_tensorDimsSize,
                         true,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         false,
                         0,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    createTrainingTensor(m_outputTensorsContainer,
                         0 /* tensorIndex */,
                         4U,
                         syn_type_single,
                         m_tensorInfoVecOutputs[0].m_tensorDimsSize,
                         true,
                         m_tensorInfoVecOutputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr,
                         false,
                         0,
                         nullptr,
                         DATA_TENSOR,
                         nullptr /* minTensorSize */);

    // Create DMA node
    status = synNodeCreate(m_graphHandle,
                           // input/output tensor vectors
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           // input/output tensor vector sizes
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           // user params
                           nullptr,
                           0,
                           // guid and node name
                           "relu_fwd_f32",
                           "worker_0_relu",
                           // input/output layouts
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}

bool TestRecipeRelu::read_file(const std::string& file_name, float* output, uint32_t readLength) const
{
    std::ifstream file(file_name);
    if (file.good())
    {
        file.seekg(0, std::ios::end);
        uint32_t length = file.tellg();
        if (length != readLength)
        {
            file.close();
            LOG_ERR(SYN_RT_TEST,
                    "File '{}' unexpected length. Expected: {}, actual: {}.",
                    file_name,
                    readLength,
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
        LOG_ERR(SYN_RT_TEST, "File '{}' doesn't exist", file_name);
        return false;
    }
    // Shouldn't get here
    return false;
}