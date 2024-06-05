#pragma once

#include <gtest/gtest.h>
#include "synapse_api.h"
#include "tensor.h"

class UTTensorTest : public ::testing::Test
{
public:
    static synTensor createInferenceTensor(const std::vector<unsigned>& dims,
                                           TensorUsage                  usage,
                                           synDataType                  dataType,
                                           const unsigned*              tensorSize,
                                           const char*                  name,
                                           const synGraphHandle         graphHandle);

    static synTensor createWeightTensor(const synGraphHandle         graphHandle,
                                        TensorUsage                  usage,
                                        const std::vector<unsigned>& dims,
                                        synDataType                  dataType,
                                        const std::string&           name,
                                        char*                        data,
                                        unsigned                     dataSize);

    static void check_tensors_creation(const std::string& nameTensor1, const std::string& nameTensor2);

    static void destroySections();

    inline static std::vector<synSectionHandle> m_sections;
};
