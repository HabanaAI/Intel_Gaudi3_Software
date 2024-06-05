#include "syn_singleton.hpp"
#include "synapse_api.h"
#include "infra/gc_synapse_test_common.hpp"

class SynGaudiNDimsTests : public SynTestCommon
{
public:
    SynGaudiNDimsTests()
    {
        if (m_deviceType == synDeviceTypeInvalid)
        {
            LOG_WARN(SYN_TEST,
                     "No device type specified in SYN_DEVICE_TYPE env variable, using default value: synDeviceGaudi");
            m_deviceType = synDeviceGaudi;
        }
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2});
    }
};

TEST_F_GC(SynGaudiNDimsTests, remove_contiguous_reshapes)
{
    synTensor             r1, r2, r3, r4, r5;
    synTensorDeviceLayout l1, l2, l3, l4, l5;
    synTensorGeometry     g1, g2, g3, g4, g5;
    synSectionHandle      s1, s5;
    synGraphHandle        graphHandle;
    NStrideArray          shape2 = {5, 5, 5, 25};

    std::array<unsigned, HABANA_DIM_MAX> shape1 = {5, 5, 5, 5, 5, 1};

    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, m_deviceType));

    synTensorHandleCreate(&r1, graphHandle, DATA_TENSOR, "r1-data");
    ASSERT_EQ(synSuccess, synSectionCreate(&s1, 0, graphHandle));
    ASSERT_EQ(synSuccess, synTensorAssignToSection(r1, s1, 0));
    memset(&g1, 0, sizeof(g1));
    g1.dims     = 5;
    g1.sizes[0] = 5;
    g1.sizes[1] = 5;
    g1.sizes[2] = 5;
    g1.sizes[3] = 5;
    g1.sizes[4] = 5;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r1, &g1, synGeometrySizes));
    memset(&l1, 0, sizeof(l1));
    l1.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r1, &l1));

    synTensorHandleCreate(&r2, graphHandle, SHAPE_TENSOR, "r2-shape");
    ASSERT_EQ(synSuccess, synTensorSetHostPtr(r2, shape1.data(), sizeof(float) * 6, syn_type_float, true));
    memset(&g2, 0, sizeof(g2));
    g2.dims     = 1;
    g2.sizes[0] = 6;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r2, &g2, synGeometrySizes));
    memset(&l2, 0, sizeof(l2));
    l2.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r2, &l2));

    synTensorHandleCreate(&r3, graphHandle, DATA_TENSOR, "r3-output");
    memset(&g3, 0, sizeof(g3));
    g3.dims     = 6;
    g3.sizes[0] = 5;
    g3.sizes[1] = 5;
    g3.sizes[2] = 5;
    g3.sizes[3] = 5;
    g3.sizes[4] = 5, g3.sizes[5] = 1;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r3, &g3, synGeometrySizes));
    memset(&l3, 0, sizeof(l3));
    l3.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r3, &l3));

    synTensorHandleCreate(&r4, graphHandle, SHAPE_TENSOR, "r4-shape");
    ASSERT_EQ(synSuccess, synTensorSetHostPtr(r4, shape2.data(), sizeof(float) * 4, syn_type_float, true));
    memset(&g4, 0, sizeof(g4));
    g4.dims     = 1;
    g4.sizes[0] = 4;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r4, &g4, synGeometrySizes));
    memset(&l4, 0, sizeof(l4));
    l4.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r4, &l4));

    synTensorHandleCreate(&r5, graphHandle, DATA_TENSOR, "r5-output");
    ASSERT_EQ(synSuccess, synSectionCreate(&s5, 0, graphHandle));
    ASSERT_EQ(synSuccess, synTensorAssignToSection(r5, s5, 0));
    memset(&g5, 0, sizeof(g5));
    g5.dims     = 4;
    g5.sizes[0] = 5;
    g5.sizes[1] = 5;
    g5.sizes[2] = 5;
    g5.sizes[3] = 25;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r5, &g5, synGeometrySizes));
    memset(&l5, 0, sizeof(l5));
    l5.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r5, &l5));

    synStatus status;
    synTensor inputs1[]  = {r1, r2};
    synTensor outputs1[] = {r3};
    synTensor inputs2[]  = {r3, r4};
    synTensor outputs2[] = {r5};
    status = synNodeCreate(graphHandle, inputs1, outputs1, 2, 1, nullptr, 4, "reshape", "reshape1", nullptr, nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create reshape Node";
    status = synNodeCreate(graphHandle, inputs2, outputs2, 2, 1, nullptr, 4, "reshape", "reshape2", nullptr, nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create reshape Node";

    synRecipeHandle recipeHandle;
    status = synGraphCompile(&recipeHandle, graphHandle, "ndim_reshapes", nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

    synTensorDestroy(r1);
    synTensorDestroy(r2);
    synTensorDestroy(r3);
    synTensorDestroy(r4);
    synTensorDestroy(r5);
    synSectionDestroy(s1);
    synSectionDestroy(s5);
    synGraphDestroy(graphHandle);
}

TEST_F_GC(SynGaudiNDimsTests, DISABLED_broadcast_7d)
{
    synTensor             r1, r3;
    synTensorDeviceLayout l1, l3;
    synTensorGeometry     g1, g3;
    synSectionHandle      s1, s5;
    synGraphHandle        graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, synDeviceGaudi));

    synTensorHandleCreate(&r1, graphHandle, DATA_TENSOR, "r1-data");
    ASSERT_EQ(synSuccess, synSectionCreate(&s1, 0, graphHandle));
    ASSERT_EQ(synSuccess, synTensorAssignToSection(r1, s1, 0));
    memset(&g1, 0, sizeof(g1));
    g1.dims     = 1;
    g1.sizes[0] = 1;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r1, &g1, synGeometrySizes));
    memset(&l1, 0, sizeof(l1));
    l1.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r1, &l1));

    synTensorHandleCreate(&r3, graphHandle, DATA_TENSOR, "r3-output");
    ASSERT_EQ(synSuccess, synSectionCreate(&s5, 0, graphHandle));
    ASSERT_EQ(synSuccess, synTensorAssignToSection(r3, s5, 0));
    memset(&g3, 0, sizeof(g3));
    g3.dims     = 7;
    g3.sizes[0] = 2;
    g3.sizes[1] = 4;
    g3.sizes[2] = 8;
    g3.sizes[3] = 16;
    g3.sizes[4] = 32;
    g3.sizes[5] = 1;
    g3.sizes[6] = 64;
    ASSERT_EQ(synSuccess, synTensorSetGeometry(r3, &g3, synGeometrySizes));
    memset(&l3, 0, sizeof(l3));
    l3.deviceDataType = syn_type_float;
    ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(r3, &l3));

    synStatus status;
    synTensor inputs1[]  = {r1};
    synTensor outputs1[] = {r3};
    status =
        synNodeCreate(graphHandle, inputs1, outputs1, 1, 1, nullptr, 4, "broadcast", "broadcast", nullptr, nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to create broadcast Node";

    synRecipeHandle recipeHandle;
    status = synGraphCompile(&recipeHandle, graphHandle, "ndim_broadcast", nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to compile graph";

    synTensorDestroy(r1);
    synTensorDestroy(r3);
    synSectionDestroy(s1);
    synSectionDestroy(s5);
    synGraphDestroy(graphHandle);
}