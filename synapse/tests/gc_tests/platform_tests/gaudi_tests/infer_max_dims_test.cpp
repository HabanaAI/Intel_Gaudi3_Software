#include "synapse_api.h"

#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "utils.h"
#include "utils/test_utils.h"

#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <string>
#include <vector>

template<typename T>
static size_t prod(const T& o)
{
    return std::accumulate(std::begin(o), std::end(o), size_t {1}, std::multiplies<size_t>());
}

static synConvolutionParams getNOPConvParams()
{
    synConvolutionParams res;
    res.dH   = 1;
    res.dW   = 1;
    res.kH   = 1;
    res.kW   = 1;
    res.padT = 0;
    res.padB = 0;
    res.padL = 0;
    res.padR = 0;
    res.dilH = 1;
    res.dilW = 1;
    return res;
}

static bool operator==(const synTensorGeometry& a, const synTensorGeometry& b)
{
    return a.dims == b.dims && std::equal(std::begin(a.sizes), std::end(a.sizes), b.sizes);
}

// Conv -> Transpose -> Conv
//
// +--+                                   +--+
// |w0+------+                            |w1+------+
// +--+      |                            +--+      |
//           v                                      v
// +--+   +--+--+   +--+   +----------+   +--+   +--+--+   +--+
// |i0+-->+conv0+-->+t0+-->+transpose1+-->+t1+-->+conv2+-->+o0|
// +--+   +-----+   +--+   +----------+   +--+   +-----+   +--+
//
struct InferMaxDimsParams
{
    bool inferDimsW0;  // Expected: Failure (Cannot infer weight dims)
    bool inferDimsI0;  // Expected: Failure (Cannot infer input dims)
    bool inferDimsT0;  // Expected: Success (As long as tranpose has SIF)
    bool inferDimsW1;  // Expected: Failure (Cannot infer weight dims)
    bool inferDimsT1;  // Expected: Success (As long as tranpose has SIF)
    bool inferDimsO0;  // Expected: Failure + Log no inference for persistent tensors

    // Since we do maxInfer at nodeAddition where possible (A nodes inputs are known)
    // and only leave other cases up to a pass, we want to test when adding nodes in a reverse order as well.
    bool switchConv0Transpose;
};

class InferMaxDimsFixture : public ::testing::TestWithParam<InferMaxDimsParams>
{
    void SetUp() override { ASSERT_EQ(synInitialize(), synSuccess); }

    // Tears down the test fixture.
    void TearDown() override
    {
        ASSERT_EQ(synDestroy(), synSuccess);
        ASSERT_EQ(synDestroy(), synUninitialized);
    }
};

// Test compilation with max-dims inference and verify that the max geometry is behaving as expected
TEST_P(InferMaxDimsFixture, External_CompileOnly)
{
    InferMaxDimsParams params = GetParam();

    constexpr synDataType dataType = syn_type_bf16;
    constexpr unsigned    B        = 2;
    constexpr unsigned    C        = 31;
    constexpr unsigned    H        = 67;
    constexpr unsigned    W        = 67;

    synConvolutionParams conv_params = getNOPConvParams();
    synTransposeParams   transpose_params {{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, 4};

    const std::array<unsigned, 4> i0_dims = {C, W, H, B};
    const std::array<unsigned, 4> t0_dims = {C, W, H, B};
    const std::array<unsigned, 4> t1_dims = {W, C, H, B};
    const std::array<unsigned, 4> o0_dims = {W, C, H, B};

    const std::array<unsigned, 4> w0_dims = {C, C, conv_params.kW, conv_params.kH};
    const std::array<unsigned, 4> w1_dims = {W, W, conv_params.kW, conv_params.kH};

    std::vector<int8_t> i0_data(prod(i0_dims));
    std::vector<int8_t> w0_data(prod(w0_dims));
    std::vector<int8_t> w1_data(prod(w1_dims));

    std::generate(begin(i0_data), end(i0_data), Test_Random_Number_Creator({-2, 2}));
    std::generate(begin(w0_data), end(w0_data), Test_Random_Number_Creator({-2, 2}));
    std::generate(begin(w1_data), end(w1_data), Test_Random_Number_Creator({-2, 2}));

    synGraphHandle graphHandle;
    ASSERT_EQ(synSuccess, synGraphCreate(&graphHandle, synDeviceGaudi));

    // Handle cleanup with RAII since we have expected failure tests (ASSERT) to avoid leaks
    struct S
    {
        synGraphHandle m_graphHandle {};
        ~S() { EXPECT_EQ(synSuccess, synGraphDestroy(m_graphHandle)); }
    } graphHandleCtx {graphHandle};

    synTensor         i0;
    synTensorGeometry i0_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&i0, graphHandle, DATA_TENSOR, "i0"));

        if (!params.inferDimsI0)
        {
            for (size_t i =0; i < i0_dims.size(); i++) i0_geometry.sizes[i] = i0_dims[i];
        }
        i0_geometry.dims = i0_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(i0, &i0_geometry, params.inferDimsI0 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(i0, &deviceLayout));

        synSectionHandle sectionHandle;
        uint64_t         sectionDescriptor = 0;
        ASSERT_EQ(synSuccess, synSectionCreate(&sectionHandle, sectionDescriptor, graphHandle));
        ASSERT_EQ(synSuccess, synSectionSetPersistent(sectionHandle, true));
        ASSERT_EQ(synSuccess, synTensorAssignToSection(i0, sectionHandle, 0));
        ASSERT_EQ(synSuccess, synSectionDestroy(sectionHandle));
    }

    synTensor         w0;
    synTensorGeometry w0_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&w0, graphHandle, DATA_TENSOR, "w0"));

        if (!params.inferDimsW0)
        {
            for (size_t i =0; i < w0_dims.size(); i++) w0_geometry.sizes[i] = w0_dims[i];
        }
        w0_geometry.dims = w0_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(w0, &w0_geometry, params.inferDimsW0 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(w0, &deviceLayout));

        synSectionHandle sectionHandle;
        uint64_t         sectionDescriptor = 0;
        ASSERT_EQ(synSuccess, synSectionCreate(&sectionHandle, sectionDescriptor, graphHandle));
        ASSERT_EQ(synSuccess, synSectionSetPersistent(sectionHandle, true));
        ASSERT_EQ(synSuccess, synTensorAssignToSection(w0, sectionHandle, 0));
        ASSERT_EQ(synSuccess, synSectionDestroy(sectionHandle));
    }

    synTensor         w1;
    synTensorGeometry w1_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&w1, graphHandle, DATA_TENSOR, "w1"));

        if (!params.inferDimsW1)
        {
            for (size_t i =0; i < w1_dims.size(); i++) w1_geometry.sizes[i] = w1_dims[i];
        }
        w1_geometry.dims = w1_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(w1, &w1_geometry, params.inferDimsW1 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(w1, &deviceLayout));

        synSectionHandle sectionHandle;
        uint64_t         sectionDescriptor = 0;
        ASSERT_EQ(synSuccess, synSectionCreate(&sectionHandle, sectionDescriptor, graphHandle));
        ASSERT_EQ(synSuccess, synSectionSetPersistent(sectionHandle, true));
        ASSERT_EQ(synSuccess, synTensorAssignToSection(w1, sectionHandle, 0));
        ASSERT_EQ(synSuccess, synSectionDestroy(sectionHandle));
    }

    synTensor         t0;
    synTensorGeometry t0_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&t0, graphHandle, DATA_TENSOR, "t0"));

        if (!params.inferDimsT0)
        {
            for (size_t i =0; i < t0_dims.size(); i++) t0_geometry.sizes[i] = t0_dims[i];
        }
        t0_geometry.dims = t0_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(t0, &t0_geometry, params.inferDimsT0 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(t0, &deviceLayout));
    }

    synTensor         t1;
    synTensorGeometry t1_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&t1, graphHandle, DATA_TENSOR, "t1"));

        if (!params.inferDimsT1)
        {
            for (size_t i =0; i < t1_dims.size(); i++) t1_geometry.sizes[i] = t1_dims[i];
        }
        t1_geometry.dims = t1_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(t1, &t1_geometry, params.inferDimsT1 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(t1, &deviceLayout));
    }

    synTensor         o0;
    synTensorGeometry o0_geometry {};
    {
        ASSERT_EQ(synSuccess, synTensorHandleCreate(&o0, graphHandle, DATA_TENSOR, "o0"));

        if (!params.inferDimsO0)
        {
            std::copy(o0_dims.begin(), o0_dims.end(), o0_geometry.sizes);
            for (size_t i =0; i < o0_dims.size(); i++) o0_geometry.sizes[i] = o0_dims[i];
        }
        o0_geometry.dims = o0_dims.size();
        ASSERT_EQ(synSuccess,
                  synTensorSetGeometry(o0, &o0_geometry, params.inferDimsO0 ? synGeometryDims : synGeometryMaxSizes));

        synTensorDeviceLayout deviceLayout {};
        deviceLayout.deviceDataType = dataType;
        ASSERT_EQ(synSuccess, synTensorSetDeviceLayout(o0, &deviceLayout));

        synSectionHandle sectionHandle;
        uint64_t         sectionDescriptor = 0;
        ASSERT_EQ(synSuccess, synSectionCreate(&sectionHandle, sectionDescriptor, graphHandle));
        ASSERT_EQ(synSuccess, synSectionSetPersistent(sectionHandle, true));
        ASSERT_EQ(synSuccess, synTensorAssignToSection(o0, sectionHandle, 0));
        ASSERT_EQ(synSuccess, synSectionDestroy(sectionHandle));
    }

    const bool conv0_ok = !params.inferDimsI0 && !params.inferDimsW0;
    for (auto node_idx : {false, true})
    {
        if (node_idx == params.switchConv0Transpose)
        {
            std::array<synTensor, 3> inputs {i0, w0, nullptr};
            std::array<synTensor, 1> outputs {t0};
            ASSERT_EQ(conv0_ok ? synSuccess : synInvalidArgument,
                      synNodeCreate(graphHandle,
                                    inputs.data(),
                                    outputs.data(),
                                    inputs.size(),
                                    outputs.size(),
                                    &conv_params,
                                    sizeof(conv_params),
                                    "spatial_convolution",
                                    "conv0",
                                    nullptr,
                                    nullptr));
        }

        if (node_idx != params.switchConv0Transpose)
        {
            std::array<synTensor, 1> inputs {t0};
            std::array<synTensor, 1> outputs {t1};
            ASSERT_EQ(synSuccess,
                      synNodeCreate(graphHandle,
                                    inputs.data(),
                                    outputs.data(),
                                    inputs.size(),
                                    outputs.size(),
                                    &transpose_params,
                                    sizeof(transpose_params),
                                    "transpose",
                                    "transpose1",
                                    nullptr,
                                    nullptr));
        }
    }

    const bool conv2_ok = !params.inferDimsW1 && !params.inferDimsO0;
    {
        std::array<synTensor, 3> inputs {t1, w1, nullptr};
        std::array<synTensor, 1> outputs {o0};
        ASSERT_EQ(conv2_ok ? synSuccess : synInvalidArgument,
                  synNodeCreate(graphHandle,
                                inputs.data(),
                                outputs.data(),
                                inputs.size(),
                                outputs.size(),
                                &conv_params,
                                sizeof(conv_params),
                                "spatial_convolution",
                                "conv2",
                                nullptr,
                                nullptr));
    }

    synTensorGeometry w0_geometry_ref_set {};
    synTensorGeometry i0_geometry_ref_set {};
    synTensorGeometry t0_geometry_ref_set {};
    synTensorGeometry w1_geometry_ref_set {};
    synTensorGeometry t1_geometry_ref_set {};
    synTensorGeometry o0_geometry_ref_set {};
    {
        std::fill(std::begin(w0_geometry_ref_set.sizes), std::end(w0_geometry_ref_set.sizes), 1);
        std::fill(std::begin(i0_geometry_ref_set.sizes), std::end(i0_geometry_ref_set.sizes), 1);
        std::fill(std::begin(t0_geometry_ref_set.sizes), std::end(t0_geometry_ref_set.sizes), 1);
        std::fill(std::begin(w1_geometry_ref_set.sizes), std::end(w1_geometry_ref_set.sizes), 1);
        std::fill(std::begin(t1_geometry_ref_set.sizes), std::end(t1_geometry_ref_set.sizes), 1);
        std::fill(std::begin(o0_geometry_ref_set.sizes), std::end(o0_geometry_ref_set.sizes), 1);
        std::copy(w0_dims.begin(), w0_dims.end(), w0_geometry_ref_set.sizes);
        std::copy(i0_dims.begin(), i0_dims.end(), i0_geometry_ref_set.sizes);
        std::copy(t0_dims.begin(), t0_dims.end(), t0_geometry_ref_set.sizes);
        std::copy(w1_dims.begin(), w1_dims.end(), w1_geometry_ref_set.sizes);
        std::copy(t1_dims.begin(), t1_dims.end(), t1_geometry_ref_set.sizes);
        std::copy(o0_dims.begin(), o0_dims.end(), o0_geometry_ref_set.sizes);
        w0_geometry_ref_set.dims = w0_dims.size();
        i0_geometry_ref_set.dims = i0_dims.size();
        t0_geometry_ref_set.dims = t0_dims.size();
        w1_geometry_ref_set.dims = w1_dims.size();
        t1_geometry_ref_set.dims = t1_dims.size();
        o0_geometry_ref_set.dims = o0_dims.size();
    }

    {
        synTensorGeometry w0_geometry_res {};
        synTensorGeometry i0_geometry_res {};
        synTensorGeometry t0_geometry_res {};
        synTensorGeometry w1_geometry_res {};
        synTensorGeometry t1_geometry_res {};
        synTensorGeometry o0_geometry_res {};
        std::fill(std::begin(t0_geometry.sizes) + t0_dims.size(), std::end(t0_geometry.sizes), 1);
        std::fill(std::begin(t1_geometry.sizes) + t1_dims.size(), std::end(t1_geometry.sizes), 1);

        EXPECT_EQ(params.inferDimsW0 ? synObjectNotInitialized : synSuccess,
                  synTensorGetGeometry(w0, &w0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(params.inferDimsI0 ? synObjectNotInitialized : synSuccess,
                  synTensorGetGeometry(i0, &i0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(t0, &t0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(params.inferDimsW1 ? synObjectNotInitialized : synSuccess,
                  synTensorGetGeometry(w1, &w1_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(params.inferDimsT0 && params.inferDimsT1 && params.switchConv0Transpose ? synObjectNotInitialized
                                                                                          : synSuccess,
                  synTensorGetGeometry(t1, &t1_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(params.inferDimsO0 ? synObjectNotInitialized : synSuccess,
                  synTensorGetGeometry(o0, &o0_geometry_res, synGeometryMaxSizes));

        EXPECT_EQ(params.inferDimsW0 ? synTensorGeometry {} : w0_geometry_ref_set, w0_geometry_res);
        EXPECT_EQ(params.inferDimsI0 ? synTensorGeometry {} : i0_geometry_ref_set, i0_geometry_res);
        EXPECT_EQ(params.inferDimsT0 ? t0_geometry_ref_set : t0_geometry, t0_geometry_res);
        EXPECT_EQ(params.inferDimsW1 ? synTensorGeometry {} : w1_geometry_ref_set, w1_geometry_res);
        EXPECT_EQ(!params.inferDimsT1                                   ? t1_geometry
                  : !params.inferDimsT0 || !params.switchConv0Transpose ? t1_geometry_ref_set
                                                                        : synTensorGeometry {},
                  t1_geometry_res);
        EXPECT_EQ(params.inferDimsO0 ? synTensorGeometry {} : o0_geometry_ref_set, o0_geometry_res);
    }

    const bool cleanCompilation = conv0_ok && conv2_ok;
    if (!cleanCompilation) return;

    synRecipeHandle recipeHandle;
    ASSERT_EQ(synSuccess, synGraphCompile(&recipeHandle, graphHandle, "recipe", nullptr));

    {
        synTensorGeometry w0_geometry_res {};
        synTensorGeometry i0_geometry_res {};
        synTensorGeometry t0_geometry_res {};
        synTensorGeometry w1_geometry_res {};
        synTensorGeometry t1_geometry_res {};
        synTensorGeometry o0_geometry_res {};
        EXPECT_EQ(synSuccess, synTensorGetGeometry(w0, &w0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(i0, &i0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(t0, &t0_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(w1, &w1_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(t1, &t1_geometry_res, synGeometryMaxSizes));
        EXPECT_EQ(synSuccess, synTensorGetGeometry(o0, &o0_geometry_res, synGeometryMaxSizes));

        EXPECT_EQ(w0_geometry_res, w0_geometry_ref_set);
        EXPECT_EQ(i0_geometry_res, i0_geometry_ref_set);
        EXPECT_EQ(t0_geometry_res, t0_geometry_ref_set);
        EXPECT_EQ(w1_geometry_res, w1_geometry_ref_set);
        EXPECT_EQ(t1_geometry_res, t1_geometry_ref_set);
        EXPECT_EQ(o0_geometry_res, o0_geometry_ref_set);
    }

    ASSERT_EQ(synSuccess, synRecipeDestroy(recipeHandle));
}

INSTANTIATE_TEST_SUITE_P(InferMaxDims,
                         InferMaxDimsFixture,
                         ::testing::Values(InferMaxDimsParams {false, false, false, false, false, false, false},
                                           InferMaxDimsParams {true, false, false, false, false, false, false},
                                           InferMaxDimsParams {false, true, false, false, false, false, false},
                                           InferMaxDimsParams {false, false, true, false, false, false, false},
                                           InferMaxDimsParams {false, false, false, true, false, false, false},
                                           InferMaxDimsParams {false, false, false, false, true, false, false},
                                           InferMaxDimsParams {false, false, false, false, false, true, false},
                                           InferMaxDimsParams {false, false, true, false, true, false, false},
                                           InferMaxDimsParams {false, false, true, false, true, false, true}));
