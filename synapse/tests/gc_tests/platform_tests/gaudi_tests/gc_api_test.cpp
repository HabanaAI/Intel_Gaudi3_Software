#include "synapse_api.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "gtest/gtest.h"

class SynGCAPITest : public SynTest
{
public:
    SynGCAPITest()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
            m_deviceType = synDeviceGaudi;
        }
        setSupportedDevices({synDeviceGaudi});
    };
};

TEST_F_GC(SynGCAPITest, use_deterministic)
{
    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, synDeviceGaudi)) << "Failed to create graph";

    synTensor in, out;
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&in, graphHandle, DATA_TENSOR, "in0")) << "Failed to create tensor";
    ASSERT_EQ(synSuccess, synTensorHandleCreate(&out, graphHandle, DATA_TENSOR, "out0")) << "Failed to create tensor";

    synTensorGeometry geometry {};
    geometry.dims = 1;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(in, &geometry, synGeometryDims)) << "Failed synTensorSetGeometry";
    ASSERT_EQ(synSuccess, synTensorSetGeometry(out, &geometry, synGeometryDims)) << "Failed synTensorSetGeometry";
    ;

    synTensorDeviceLayout deviceLayout {};
    deviceLayout.deviceDataType = syn_type_bf16;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(in, &deviceLayout)) << "Failed synTensorSetDeviceLayout";
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(out, &deviceLayout)) << "Failed synTensorSetDeviceLayout";

    synNodeId memcpyNodeId;
    ASSERT_EQ(synSuccess,
              synNodeCreateWithId(graphHandle,
                                  &in,
                                  &out,
                                  1,
                                  1,
                                  nullptr,
                                  0,
                                  "memcpy",
                                  "my_memcpy",
                                  &memcpyNodeId,
                                  nullptr,
                                  nullptr))
        << "Failed to create node with GUID "
        << "memcpy";

    bool useDeterministic = true;
    ASSERT_EQ(synSuccess, synNodeGetDeterministic(graphHandle, memcpyNodeId, &useDeterministic))
        << "Failed synNodeGetDeterministic";
    ASSERT_EQ(useDeterministic, false) << "Expecting default to be false";
    ASSERT_EQ(synSuccess, synNodeSetDeterministic(graphHandle, memcpyNodeId, true)) << "Failed synNodeSetDeterministic";
    ASSERT_EQ(synSuccess, synNodeGetDeterministic(graphHandle, memcpyNodeId, &useDeterministic))
        << "Failed synNodeGetDeterministic";
    ASSERT_EQ(useDeterministic, true) << "Expecting useDeterministic to be true after setting";
}