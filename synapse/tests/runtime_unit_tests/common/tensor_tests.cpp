#include <gtest/gtest.h>
#include "tensor_tests.hpp"
#include "test_utils.h"
#include "hpp/synapse.hpp"

synTensor UTTensorTest::createInferenceTensor(const std::vector<unsigned>& dims,
                                              TensorUsage                  usage,
                                              synDataType                  dataType,
                                              const unsigned*              tensorSize,
                                              const char*                  name,
                                              const synGraphHandle         graphHandle)
{
    synStatus        status;
    synSectionHandle pSectionHandle = nullptr;
    synSectionCreate(&pSectionHandle, 0, graphHandle);
    synTensorDescriptor desc;
    desc.m_dataType = dataType;
    desc.m_dims     = dims.size();
    memcpy(desc.m_sizes, const_cast<unsigned*>(dims.data()), sizeof(unsigned) * dims.size());
    desc.m_ptr                   = nullptr;
    desc.m_isWeights             = false;
    desc.m_isQuantized           = true;
    desc.m_name                  = name;
    desc.m_isOutput              = usage == OUTPUT_TENSOR;
    desc.m_enablePerChannelQuant = false;
    desc.m_quantizationParams[0] = synQuantizationParams(dataType);
    desc.m_isInput               = usage == INPUT_TENSOR;
    desc.m_isStatic              = false;

    synTensor tensor;
    status = synTensorCreate(&tensor, &desc, pSectionHandle, 0);
    assert(status == synSuccess && "Create tensor failed!");

    if (pSectionHandle)
    {
        m_sections.push_back(pSectionHandle);
    }
    UNUSED(status);

    return tensor;
}

synTensor UTTensorTest::createWeightTensor(const synGraphHandle         graphHandle,
                                           TensorUsage                  usage,
                                           const std::vector<unsigned>& dims,
                                           synDataType                  dataType,
                                           const std::string&           name,
                                           char*                        data,
                                           unsigned                     dataSize)
{
    unsigned                     dimsSize = dims.size();
    const std::vector<unsigned>* pDims    = &dims;
    std::vector<unsigned>        newDims(dimsSize);

    auto nameCopy = name;
    EXPECT_LE(name.size(), ENQUEUE_TENSOR_NAME_MAX_SIZE);

    if (name.empty())
    {
        static uint32_t value = 0;
        nameCopy              = "tensor_" + std::to_string(value++);
    }

    synTensorDescriptor desc;
    desc.m_dataType = dataType;
    desc.m_dims     = (*pDims).size();
    memcpy(desc.m_sizes, const_cast<unsigned*>((*pDims).data()), sizeof(unsigned) * (*pDims).size());
    desc.m_ptr                   = data;
    desc.m_isWeights             = true;
    desc.m_isQuantized           = true;
    desc.m_name                  = nameCopy.c_str();
    desc.m_isOutput              = usage == OUTPUT_TENSOR;
    desc.m_enablePerChannelQuant = false;
    desc.m_batchPos              = INVALID_BATCH_POS;
    desc.m_quantizationParams[0] = synQuantizationParams(dataType);
    desc.m_isInput               = usage == INPUT_TENSOR;
    desc.m_isStatic              = true;

    synTensor        tensor;
    synSectionHandle sectionHandle  = nullptr;
    const uint64_t   SECTION_OFFSET = 0;

    EXPECT_EQ(synTensorCreate(&tensor, &desc, sectionHandle, SECTION_OFFSET), synSuccess);
    return tensor;
}

void UTTensorTest::check_tensors_creation(const std::string& nameTensor1, const std::string& nameTensor2)
{
    syn::Context                context;
    const std::vector<unsigned> dims(SYN_MAX_TENSOR_DIM, 1);
    synTensorDescriptor         desc;
    desc.m_dataType = syn_type_fixed;
    desc.m_dims     = dims.size();
    desc.m_name     = nameTensor1.c_str();
    memcpy(desc.m_sizes, const_cast<unsigned*>(dims.data()), sizeof(unsigned) * dims.size());
    synTensor        tensor1;
    synTensor        tensor2;
    synSectionHandle sectionHandle  = nullptr;
    const uint64_t   SECTION_OFFSET = 0;
    EXPECT_EQ(synTensorCreate(&tensor1, &desc, sectionHandle, SECTION_OFFSET), synSuccess);
    desc.m_name = nameTensor2.c_str();
    EXPECT_EQ(synTensorCreate(&tensor2, &desc, sectionHandle, SECTION_OFFSET), synSuccess);
    if (nameTensor1 == nameTensor2)
    {
        EXPECT_EQ(tensor1, tensor2);
    }
    else
    {
        EXPECT_NE(tensor1, tensor2);
    }
    EXPECT_EQ(synTensorDestroy(tensor2), synSuccess);
    EXPECT_EQ(synTensorDestroy(tensor1), synSuccess);
}

TEST_F(UTTensorTest, check_tensor_creation_name_identical)
{
    const std::string nameTensor("tensor");
    check_tensors_creation(nameTensor, nameTensor);
}

TEST_F(UTTensorTest, check_tensor_creation_name_difference)
{
    const std::string nameTensor1("tensor1");
    const std::string nameTensor2("tensor2");
    check_tensors_creation(nameTensor1, nameTensor2);
}

void UTTensorTest::destroySections()
{
    for (auto section : m_sections)
    {
        auto status = synSectionDestroy(section);
        assert(status == synSuccess && "Destroy section failed!");
        UNUSED(status);
    }
    m_sections.clear();
}
