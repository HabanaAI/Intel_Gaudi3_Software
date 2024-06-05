#include "test_recipe_assert_async.hpp"
#include "test_tensors_container.hpp"
#include "node_factory.h"
#include "synapse_api.h"
#include "../infra/test_types.hpp"

TestRecipeAssertAsync::TestRecipeAssertAsync(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeAssertAsync>(),
                 deviceType,
                 1 /* inputTensorsAmount */,
                 0 /* innerTensorsAmount */,
                 0 /* outputTensorsAmount */,
                 0 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    unsigned numOfDims                          = 1;
    unsigned tensorDimSizes[SYN_MAX_TENSOR_DIM] = {1, 0, 0, 0};

    uint64_t tensorSize = sizeof(int8_t);

    // Init m_tensorInfoVecInputs
    m_tensorInfoVecInputs[0].m_dimsAmount = numOfDims;
    m_tensorInfoVecInputs[0].m_dataType   = syn_type_int8;
    m_tensorInfoVecInputs[0].m_tensorType = DATA_TENSOR;
    m_tensorInfoVecInputs[0].m_tensorSize = tensorSize;
    m_tensorInfoVecInputs[0].m_tensorName = "input";
    m_tensorInfoVecInputs[0].m_tensorSize = 1 * dataTypeSizeInBytes(syn_type_int8);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[0].m_tensorDimsSize);
}

void TestRecipeAssertAsync::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    LOG_ERR(SYN_TEST, "This method ({}) should be overriden", __FUNCTION__);
    ASSERT_TRUE(false) << "This method (" << __func__ << ") must be overriden";
}

void TestRecipeAssertAsync::_graphCreation()
{
    // Tensors
    createTrainingTensor(m_inputTensorsContainer,
                         0 /* tensorIndex */,
                         m_tensorInfoVecInputs[0],
                         true /* isPersist */,
                         m_tensorInfoVecInputs[0].m_tensorName,
                         m_graphHandle,
                         nullptr /* pSectionHandle */,
                         nullptr /* hostBuffer */);

    synAssertAsyncParams params;
    params.msg_id = 44;

    // Create DMA node
    synStatus status = synNodeCreate(m_graphHandle,
                                     // input/output tensor vectors
                                     m_inputTensorsContainer.tensors(),
                                     nullptr,
                                     // input/output tensor vector sizes
                                     1,
                                     0,
                                     // user params
                                     &params,
                                     sizeof(synAssertAsyncParams),
                                     // guid and node name
                                     "assert_async",
                                     "assert_async",
                                     // input/output layouts
                                     nullptr,
                                     nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}