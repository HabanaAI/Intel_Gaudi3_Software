#include "test_recipe_tpc_const_section.hpp"

#include "node_factory.h"

#include "synapse_api.h"
#include "test_tensor_init.hpp"
#include "../infra/test_types.hpp"

#include "test_tensors_container.hpp"
#include "test_utils.h"

#include <vector>

// isConstSection - in case set, the second tensor will be on a const-section
TestRecipeTpcConstSection::TestRecipeTpcConstSection(synDeviceType deviceType)
: TestRecipeBase(makeUniqueRecipeName<TestRecipeTpcConstSection>(),
                 deviceType,
                 2 /* inputTensorsAmount   */,
                 0 /* innerTensorsAmount   */,
                 1 /* outputTensorsAmount  */,
                 1 /* uniqueSectionsAmount */,
                 false /* eagerMode */)
{
    const unsigned numOfDims                          = 4;
    const unsigned N                                  = 4;
    const unsigned H                                  = 4;
    const unsigned W                                  = 4;
    const unsigned B                                  = 1;
    const TSize    tensorDimSizes[SYN_MAX_TENSOR_DIM] = {N, W, H, B};
    const uint64_t tensorSizeInElements               = N * W * H * B;

    // Init m_tensorInfoVecInputs
    // Tensor-0
    unsigned tensorIndex                             = 0;  // Per type(input / output)
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In1";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Tensor-1
    tensorIndex++;
    m_tensorInfoVecInputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecInputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecInputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecInputs[tensorIndex].m_sectionType = TestSectionType::CONST_SECTION;
    m_tensorInfoVecInputs[tensorIndex].m_tensorName  = "In2";
    m_tensorInfoVecInputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecInputs[tensorIndex].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecInputs[tensorIndex].m_tensorDimsSize);

    // Const-section definition
    {
        m_uniqueSectionsInfo[0].m_sectionSize    = m_tensorInfoVecInputs[tensorIndex].m_tensorSize;
        m_uniqueSectionsInfo[0].m_isConstSection = true;

        m_tensorInfoVecInputs[tensorIndex].m_sectionIndex = 0;
        m_tensorInfoVecInputs[tensorIndex].m_tensorFlags  = (uint32_t)TensorFlag::CONST_SECTION;
    }

    // Init m_tensorInfoVecOutputs
    // Tensor-0
    tensorIndex                                       = 0;  // Per type(input / output)
    m_tensorInfoVecOutputs[tensorIndex].m_dimsAmount  = numOfDims;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorType  = DATA_TENSOR;
    m_tensorInfoVecOutputs[tensorIndex].m_dataType    = syn_type_single;
    m_tensorInfoVecOutputs[tensorIndex].m_sectionType = TestSectionType::NON_CONST_SECTION;
    m_tensorInfoVecOutputs[tensorIndex].m_tensorName  = "Out";
    m_tensorInfoVecOutputs[tensorIndex].m_tensorSize =
        tensorSizeInElements * dataTypeSizeInBytes(m_tensorInfoVecOutputs[tensorIndex].m_dataType);
    std::copy(tensorDimSizes, tensorDimSizes + numOfDims, m_tensorInfoVecOutputs[tensorIndex].m_tensorDimsSize);
}

void TestRecipeTpcConstSection::validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const
{
    static const std::vector<uint64_t> DefaultAllocHostBuffers;
    validateResults(rLaunchTensorMemory, DefaultAllocHostBuffers);
}

void TestRecipeTpcConstSection::validateResults(const LaunchTensorMemory&    rLaunchTensorMemory,
                                                const std::vector<uint64_t>& rConstSectionsHostBuffers) const
{
    // Host-Buffer
    const float* aData = reinterpret_cast<const float*>(getHostBuffer(true /* isInput */,
                                                                      0 /* tensorIndex */,
                                                                      rLaunchTensorMemory.m_tensorInfoVecInputs,
                                                                      rConstSectionsHostBuffers));
    const float* bData = reinterpret_cast<const float*>(getHostBuffer(true /* isInput */,
                                                                      1 /* tensorIndex */,
                                                                      rLaunchTensorMemory.m_tensorInfoVecInputs,
                                                                      rConstSectionsHostBuffers));
    const float* cData = reinterpret_cast<const float*>(getHostBuffer(false /* isInput */,
                                                                      0 /* tensorIndex */,
                                                                      rLaunchTensorMemory.m_tensorInfoVecOutputs,
                                                                      rConstSectionsHostBuffers));

    const size_t refOutputLen =
        m_tensorInfoVecOutputs[0].m_tensorSize / dataTypeSizeInBytes(m_tensorInfoVecOutputs[0].m_dataType);
    for (size_t index = 0; index < refOutputLen; index++)
    {
        ASSERT_EQ((aData[index] + bData[index]), cData[index])
            << "Result validation failed at index " << index << " aData: " << aData[index] << " bData: " << bData[index]
            << " cData: " << cData[index];
    }
}

void TestRecipeTpcConstSection::_graphCreation()
{
    synStatus status(synSuccess);

    // Const section creation
    createUniqueSections(m_uniqueSections, m_graphHandle);

    // Tensors creation
    for (unsigned i = 0; i < 2; i++)
    {
        synSectionHandle* pSectionHandle = nullptr;
        void*             hostBuffer     = nullptr;
        unsigned          sectionIndex   = m_tensorInfoVecInputs[i].m_sectionIndex;
        if (sectionIndex != INVALID_SECTION_ID)
        {
            synSectionHandle& sectionHandle = m_uniqueSections.section(sectionIndex);
            pSectionHandle                  = &sectionHandle;

            ASSERT_LT(sectionIndex, m_uniqueSectionsInfo.size()) << "Invalid sectionIndex (uniqueSectionsInfo)";
            ASSERT_LT(sectionIndex, m_uniqueSectionsMemory.size()) << "Invalid sectionIndex (uniqueSectionsMemory)";
            hostBuffer =
                (void*)((uint8_t*)m_uniqueSectionsMemory[sectionIndex] + m_tensorInfoVecInputs[i].m_sectionOffset);

            initBufferValues(
                {TensorInitOp::CONST, -25},
                m_tensorInfoVecInputs[i].m_dataType,
                getNumberOfElements(m_tensorInfoVecInputs[i].m_tensorDimsSize, m_tensorInfoVecInputs[i].m_dimsAmount),
                hostBuffer);
        }

        createTrainingTensor(m_inputTensorsContainer,
                             i,
                             m_tensorInfoVecInputs[i],
                             true /* isPersist */,
                             m_tensorInfoVecInputs[i].m_tensorName,
                             m_graphHandle,
                             pSectionHandle,
                             hostBuffer);
    }
    //
    // Single output presistent-tensor
    {
        createTrainingTensor(m_outputTensorsContainer,
                             0 /* tensorIndex */,
                             m_tensorInfoVecOutputs[0],
                             true /* isPersist */,
                             m_tensorInfoVecOutputs[0].m_tensorName,
                             m_graphHandle,
                             nullptr /* pSectionHandle */,
                             nullptr /* hostBuffer */);
    }

    // Create add_f32 node
    status = synNodeCreate(m_graphHandle,
                           m_inputTensorsContainer.tensors(),
                           m_outputTensorsContainer.tensors(),
                           m_inputTensorsContainer.size(),
                           m_outputTensorsContainer.size(),
                           nullptr,
                           0,
                           "add_fwd_f32",
                           "addNode",  // guid and node name
                           nullptr,
                           nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synNodeCreate";
}
