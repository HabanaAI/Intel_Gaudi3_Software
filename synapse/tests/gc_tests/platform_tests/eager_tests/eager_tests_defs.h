#pragma once

#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "tensor.h"
#include "test_configuration.h"
#include "node_factory.h"
#include "gaudi_dual_execution_test_infra.h"
#include "utils.h"

namespace eager_mode
{
class SynTrainingEagerTests : public SynGaudiTestInfra
{
public:
    SynTrainingEagerTests()
    {
        ReleaseDevice();
        m_testConfig.m_numOfTestDevices = 1;
        m_testConfig.m_compilationMode  = COMP_EAGER_MODE_TEST;
        m_testConfig.m_supportedDeviceTypes.clear();
        setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3});
        setTestPackage(TEST_PACKAGE_EAGER);
    }

    void runNdimMemcpy(synDataType dataType, std::string_view guid);
    void broadcastOnFcdTest(synDataType dataType, std::string_view guid);
    void broadcastNonFcdTest(synDataType dataType, std::string_view guid);
};

template<typename T>
size_t prod(const T& vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), size_t {1}, std::multiplies<size_t>());
}

}  // namespace eager_mode