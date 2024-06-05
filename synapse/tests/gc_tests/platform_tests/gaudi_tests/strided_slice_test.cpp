#include "gc_gaudi_test_infra.h"
#include "gc_dynamic_shapes_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "gtest/gtest.h"

static synSliceParamsV2 createDefaultSliceParams(const unsigned sizes[], int dimCount)
{
    synSliceParamsV2 params;
    memset(params.axes, 0, sizeof(params.axes));
    memset(params.ends, 0, sizeof(params.ends));
    memset(params.starts, 0, sizeof(params.starts));
    memset(params.steps, 0, sizeof(params.steps));

    for (auto i = 0; i < dimCount; i++)
    {
        params.axes[i]   = i;
        params.starts[i] = 0;
        params.ends[i]   = sizes[i];
        params.steps[i]  = 1;
    }

    return params;
}

class SynGaudiSliceGradFcd : public SynGaudiTestInfra
{
public:
    void runSliceGradFcd();
};

TEST_F_GC(SynTrainingTestInfra, strided_slice_bwd_test)
{
    float init[] = {2, 5, 8, 11};
    unsigned  sizesIn[] = {1, 1, 4, 1};
    unsigned  sizesOut[] = {2, 3, 4, 1};
    synSliceParams params;
    memset(&params, 0, sizeof(params));
    params.steps[0] = 1;
    params.steps[1] = 2;
    params.steps[2] = 1;
    params.axes[0] = 0;
    params.axes[1] = 1;
    params.axes[2] = 2;
    params.starts[0] = 1;
    params.starts[1] = 1;
    params.starts[2] = 0;
    params.ends[0] = 2;
    params.ends[1] = 3;
    params.ends[2] = 4;

    unsigned add0 = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, init, sizesIn, 3, syn_type_single,
                                        nullptr, "input");
    unsigned add1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, init, sizesIn, 3, syn_type_single, nullptr, "zeros0");
    unsigned addOut0 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, init, sizesIn, 3, syn_type_single, nullptr, "dy");
    unsigned dy = connectOutputTensorToInputTensor(addOut0);
    unsigned dx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, sizesOut, 3, syn_type_single, nullptr, "dx");
    unsigned add2 = connectOutputTensorToInputTensor(dx);
    unsigned add3 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, init, sizesOut, 3, syn_type_single, nullptr, "zeros1");
    unsigned addOut1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, init, sizesOut, 3, syn_type_single, nullptr, "out");
    addNodeToGraph("add_fwd_f32", {add0, add1}, {addOut0}, nullptr, 0, "add_0");
    addNodeToGraph(NodeFactory::stridedSliceGradNodeTypeName, {dy}, {dx}, &params, sizeof(params), "slice_grad");
    addNodeToGraph("add_fwd_f32", {add2, add3}, {addOut1}, nullptr, 0, "add_1");

    compileTopology();
    runTopology();

    float* pDmaOutput = (float*)m_hostBuffers[addOut1];

    float* expected = init;
    std::list<unsigned> indices;
    for (unsigned i = 1; i < 2; i+=1)
    {
        for (unsigned j = 1; j < 3; j+=2)
        {
            for (unsigned k = 0; k < 4; k+=1)
            {
                indices.push_back(i + j * 2 + k * 2 * 3);
            }
        }
    }

    for (unsigned i = 0; i < 2 * 3 * 4; i++)
    {
        std::cout << *pDmaOutput << " ";
        if (i == indices.front())
        {
            indices.pop_front();
            ASSERT_EQ(*expected, *pDmaOutput) << "Mismatch at index " << i
                                              << " Expected: "        << *expected
                                              << " Result: "          << *pDmaOutput;
            expected++;
        }
        else
        {
            ASSERT_EQ(0, *pDmaOutput) << "Mismatch at index " << i
                                      << " Expected: "        << 0
                                      << " Result: "          << *pDmaOutput;
        }
        pDmaOutput++;
    }
    std::cout << std::endl;
}

void SynGaudiSliceGradFcd::runSliceGradFcd()
{
    constexpr unsigned numOfNodes = 300;

    synSliceParams params;
    memset(&params, 0, sizeof(params));
    params.axes[1]   = 1;
    params.starts[0] = 2;
    params.starts[1] = 1;
    params.ends[0]   = 4;
    params.ends[1]   = 4;
    params.steps[0]  = 2;
    params.steps[1]  = 1;

    unsigned sizesIn[]  = {1, 3};
    unsigned sizesOut[] = {6, 5};

    std::vector<unsigned> outputs;
    std::vector<unsigned> inputs;
    for (auto i = 0; i < numOfNodes; ++i)
    {
        unsigned input  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesIn, 2);
        unsigned output = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesOut, 2);
        inputs.push_back(input);
        outputs.push_back(output);
        addNodeToGraph("strided_slice_grad", {input}, {output}, (void*)&params, sizeof(params));
    }

    compileAndRun();

    for (auto t = 0; t < numOfNodes; ++t)
    {
        float*   in    = (float*)m_hostBuffers[inputs[t]];
        float*   out   = (float*)m_hostBuffers[outputs[t]];
        unsigned zeros = 0;
        for (auto i = 0; i < 5; ++i)
        {
            for (auto j = 0; j < 6; ++j)
            {
                if (*(out + ((i * 6) + j)) == 0.f)
                {
                    ++zeros;
                }
            }
        }
        if (zeros != 27 /* (6 * 5 - 3 * 1) */)
        {
            std::cout << "StridedSliceBwd number: " << (t + 1) << std::endl;
            std::cout << "Input:" << std::endl;
            for (auto i = 0; i < 1 * 3; ++i)
            {
                std::cout << *(in + i) << "\t";
            }
            std::cout << std::endl << "Output:" << std::endl;
            for (auto i = 0; i < 5; ++i)
            {
                for (auto j = 0; j < 6; ++j)
                {
                    std::cout << *(out + ((i * 6) + j)) << "\t";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            ASSERT_TRUE(false);
        }
    }
}

TEST_F_GC(SynGaudiSliceGradFcd, strided_slice_grad_fcd, {synDeviceGaudi})
{
    runSliceGradFcd();
}
// [SW-54371]- Add now passing test to gaudi2 arcmoode > 0 ci.
TEST_F_GC(SynGaudiSliceGradFcd, strided_slice_grad_fcd_non_zero_arc_mode_gaudi2, {synDeviceGaudi2})
{
    runSliceGradFcd();
}

class SynTrainingSliceTest : public SynTrainingTestInfra
{
protected:
    void run(unsigned* sizesIn, unsigned* sizesOut, unsigned numDims, synSliceParamsV2& params)
    {
        static const unsigned MAX_DIM = 6;
        ASSERT_LE(numDims, MAX_DIM);
        unsigned strides[MAX_DIM + 1] = {1, 1, 1, 1, 1, 1, 1};
        for (int i = 0; i < numDims; i++)
        {
            strides[i + 1] = strides[i] * sizesIn[i];
        }
        for (int i = numDims; i < MAX_DIM; i++)
        {
            params.axes[i]   = i;
            params.starts[i] = 0;
            params.ends[i]   = 1;
            params.steps[i]  = 1;
        }

        unsigned in  = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          sizesIn,
                                          numDims,
                                          syn_type_single,
                                          nullptr,
                                          "input");
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           sizesOut,
                                           numDims,
                                           syn_type_single,
                                           nullptr,
                                           "out");
        addNodeToGraph(NodeFactory::sliceNodeTypeName, {in}, {out}, &params, sizeof(params), "slice");

        compileTopology();
        runTopology();

        float* outputData = (float*)m_hostBuffers[out];
        float* inputData  = (float*)m_hostBuffers[in];

        std::list<unsigned> indices;
        unsigned            outIndex = 0;
        for (unsigned n = params.starts[5]; n < params.ends[5]; n += params.steps[5])
            for (unsigned b = params.starts[4]; b < params.ends[4]; b += params.steps[4])
                for (unsigned i = params.starts[3]; i < params.ends[3]; i += params.steps[3])
                    for (unsigned j = params.starts[2]; j < params.ends[2]; j += params.steps[2])
                        for (unsigned k = params.starts[1]; k < params.ends[1]; k += params.steps[1])
                            for (unsigned l = params.starts[0]; l < params.ends[0]; l += params.steps[0])
                            {
                                unsigned inIndex = n * strides[5] + b * strides[4] + i * strides[3] + j * strides[2] +
                                                   k * strides[1] + l * strides[0];
                                ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                                    << "Mismatch at index " << i << "," << j << "," << k << "," << l
                                    << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];
                                outIndex++;
                            }
    }
};

class SynGaudiSliceTest : public SynTrainingSliceTest
{
};

TEST_F_GC(SynTrainingSliceTest, strided_slice_all_dims)
{
    unsigned       sizesIn[]  = {4, 4, 4, 4};
    unsigned       sizesOut[] = {2, 2, 2, 2};
    synSliceParamsV2 params = {0};
    for (int i = 0; i < 4; i++)
    {
        params.axes[i]   = i;
        params.starts[i] = 0;
        params.ends[i]   = 4;
        params.steps[i]  = 2;
    }
    run(sizesIn, sizesOut, 4, params);
}

TEST_F_GC(SynTrainingSliceTest, strided_slice_all_dims_asymmetric)
{
    unsigned sizesIn[]  = {8, 8, 8, 8};
    unsigned sizesOut[] = {4, 2, 2, 3};

    synSliceParamsV2 params = {0};
    params.axes[0]   = 0;
    params.starts[0] = 0;
    params.ends[0]   = 8;
    params.steps[0]  = 2;

    params.axes[1]   = 1;
    params.starts[1] = 0;
    params.ends[1]   = 8;
    params.steps[1]  = 4;

    params.axes[2]   = 2;
    params.starts[2] = 3;
    params.ends[2]   = 6;
    params.steps[2]  = 2;

    params.axes[3]   = 3;
    params.starts[3] = 1;
    params.ends[3]   = 8;
    params.steps[3]  = 3;

    run(sizesIn, sizesOut, 4, params);
}

TEST_F_GC(SynGaudiSliceTest, strided_slice_all_dims_asymmetric_5d)
{
    unsigned sizesIn[]  = {8, 8, 8, 8, 8};
    unsigned sizesOut[] = {4, 2, 2, 3, 2};

    synSliceParamsV2 params = {0};
    params.axes[0]   = 0;
    params.starts[0] = 0;
    params.ends[0]   = 8;
    params.steps[0]  = 2;

    params.axes[1]   = 1;
    params.starts[1] = 0;
    params.ends[1]   = 8;
    params.steps[1]  = 4;

    params.axes[2]   = 2;
    params.starts[2] = 3;
    params.ends[2]   = 6;
    params.steps[2]  = 2;

    params.axes[3]   = 3;
    params.starts[3] = 1;
    params.ends[3]   = 8;
    params.steps[3]  = 3;

    params.axes[4]   = 4;
    params.starts[4] = 5;
    params.ends[4]   = 8;
    params.steps[4]  = 2;

    run(sizesIn, sizesOut, 5, params);
}

TEST_F_GC(SynGaudiSliceTest, strided_slice_all_dims_asymmetric_6d, {synDeviceGaudi})
{
    unsigned sizesIn[]  = {8, 8, 8, 8, 8, 8};
    unsigned sizesOut[] = {4, 2, 2, 3, 2, 2};

    synSliceParamsV2 params = {0};
    params.axes[0]   = 0;
    params.starts[0] = 0;
    params.ends[0]   = 8;
    params.steps[0]  = 2;

    params.axes[1]   = 1;
    params.starts[1] = 0;
    params.ends[1]   = 8;
    params.steps[1]  = 4;

    params.axes[2]   = 2;
    params.starts[2] = 3;
    params.ends[2]   = 6;
    params.steps[2]  = 2;

    params.axes[3]   = 3;
    params.starts[3] = 1;
    params.ends[3]   = 8;
    params.steps[3]  = 3;

    params.axes[4]   = 4;
    params.starts[4] = 5;
    params.ends[4]   = 8;
    params.steps[4]  = 2;

    params.axes[5]   = 5;
    params.starts[5] = 5;
    params.ends[5]   = 8;
    params.steps[5]  = 2;

    run(sizesIn, sizesOut, 6, params);
}

TEST_F_GC(SynGaudiSliceTest, strided_slice_fcd_dim_asymmetric_6d)
{
    unsigned sizesIn[]  = {8, 8, 8, 8, 8, 8};
    unsigned sizesOut[] = {4, 2, 2, 3, 2, 4};

    synSliceParamsV2 params = {0};
    params.axes[0]   = 0;
    params.starts[0] = 0;
    params.ends[0]   = 8;
    params.steps[0]  = 2;

    params.axes[1]   = 1;
    params.starts[1] = 0;
    params.ends[1]   = 8;
    params.steps[1]  = 4;

    params.axes[2]   = 2;
    params.starts[2] = 3;
    params.ends[2]   = 6;
    params.steps[2]  = 2;

    params.axes[3]   = 3;
    params.starts[3] = 1;
    params.ends[3]   = 8;
    params.steps[3]  = 3;

    params.axes[4]   = 4;
    params.starts[4] = 5;
    params.ends[4]   = 8;
    params.steps[4]  = 2;

    params.axes[5]   = 5;
    params.starts[5] = 4;
    params.ends[5]   = 8;
    params.steps[5]  = 1;

    run(sizesIn, sizesOut, 6, params);
}

// check output data after slice optimization (opposite shift transposes)
TEST_F_GC(SynGaudiSliceTest, slice_fcd_from_end, {synDeviceGaudi})
{
    unsigned dimCount   = 5;
    unsigned sizesIn[]  = {12, 256, 2, 10, 1};
    unsigned sizesOut[] = {8, 256, 2, 10, 1};

    synSliceParamsV2 params    = createDefaultSliceParams(sizesIn, dimCount);
    params.ends[0]             = sizesOut[0];

    run(sizesIn, sizesOut, dimCount, params);
}

TEST_F_GC(SynGaudiSliceTest, slice_fcd_from_start, {synDeviceGaudi})
{
    unsigned dimCount   = 5;
    unsigned sizesIn[]  = {12, 256, 2, 10, 1};
    unsigned sizesOut[] = {8, 256, 2, 10, 1};

    synSliceParamsV2 params    = createDefaultSliceParams(sizesIn, dimCount);
    params.starts[0]           = sizesIn[0] - sizesOut[0];

    run(sizesIn, sizesOut, dimCount, params);
}

TEST_F_GC(SynGaudiSliceTest, strided_slice_fcd_2steps, {synDeviceGaudi})
{
    unsigned dimCount   = 5;
    unsigned sizesIn[]  = {12, 128, 16, 8, 1};
    unsigned sizesOut[] = {6, 128, 16, 8, 1};

    synSliceParamsV2 params    = createDefaultSliceParams(sizesIn, dimCount);
    params.steps[0]            = 2;

    run(sizesIn, sizesOut, dimCount, params);
}

TEST_F_GC(SynGaudiSliceTest, strided_slice_fcd_3steps, {synDeviceGaudi})
{
    unsigned dimCount   = 5;
    unsigned sizesIn[]  = {12, 128, 16, 8, 1};
    unsigned sizesOut[] = {4, 128, 16, 8, 1};

    synSliceParamsV2 params    = createDefaultSliceParams(sizesIn, dimCount);
    params.steps[0]            = 3;

    run(sizesIn, sizesOut, dimCount, params);
}

class SynGaudiDynamicSliceGradTest : public SynGaudiDynamicShapesTestsInfra
{
};

TEST_F_GC(SynGaudiDynamicSliceGradTest, dynamic_slice_grad)
{
    static const unsigned NUM_DIMS         = 4;
    unsigned              sizesInMax[]     = {4, 6, 1, 2};
    unsigned              sizesInMin[]     = {3, 6, 1, 1};
    unsigned              sizesInActual[]  = {3, 6, 1, 1};
    unsigned              sizesOutMax[]    = {9, 6, 1, 2};
    unsigned              sizesOutMin[]    = {6, 6, 1, 1};
    unsigned              sizesOutActual[] = {6, 6, 1, 1};

    unsigned strides[] = {1, 1, 1, 1};
    for (int i = 0; i < NUM_DIMS - 1; i++)
    {
        strides[i + 1] = strides[i] * sizesOutActual[i];
    }

    unsigned in    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      sizesInMax,
                                      NUM_DIMS,
                                      syn_type_single,
                                      nullptr,
                                      "in",
                                      0,
                                      0,
                                      nullptr,
                                      sizesInMin);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       sizesOutMax,
                                       NUM_DIMS,
                                       syn_type_single,
                                       nullptr,
                                       "out",
                                       0,
                                       0,
                                       nullptr,
                                       sizesOutMin);
    unsigned shape = createShapeTensor(INPUT_TENSOR, sizesOutMax, sizesOutMin, NUM_DIMS);

    synSliceParamsV2 params    = createDefaultSliceParams(sizesOutMax, 4);
    params.steps[0]            = 2;
    params.ends[0]             = 8;
    params.ends[3]             = 2;

    addNodeToGraph(NodeFactory::stridedSliceGradNodeTypeName,
                   {in, shape},
                   {out},
                   &params,
                   sizeof(params),
                   "slice_grad");

    compileTopology();
    setActualSizes(in, sizesInActual);
    setActualSizes(out, sizesOutActual);
    setActualSizes(shape, sizesOutActual);
    runTopology();

    float* outputData = (float*)m_hostBuffers[out];
    float* inputData  = (float*)m_hostBuffers[in];

    std::list<unsigned> indices;
    unsigned            inIndex = 0;
    for (unsigned i = params.starts[3]; i < std::min((unsigned)params.ends[3], sizesOutActual[3]); i += params.steps[3])
        for (unsigned j = params.starts[2]; j < std::min((unsigned)params.ends[2], sizesOutActual[2]);
             j += params.steps[2])
            for (unsigned k = params.starts[1]; k < std::min((unsigned)params.ends[1], sizesOutActual[1]);
                 k += params.steps[1])
                for (unsigned l = params.starts[0]; l < std::min((unsigned)params.ends[0], sizesOutActual[0]);
                     l += params.steps[0])
                {
                    std::cout << inputData[inIndex] << ", ";

                    unsigned outIndex = i * strides[3] + j * strides[2] + k * strides[1] + l * strides[0];
                    ASSERT_EQ(inputData[inIndex], outputData[outIndex])
                        << "Mismatch at index " << i << "," << j << "," << k << "," << l
                        << " Expected: " << inputData[inIndex] << " Result: " << outputData[outIndex];

                    inIndex++;
                }
}

TEST_F_GC(SynGaudiDynamicSliceGradTest, dynamic_slice_grad2)
{
    unsigned            sizesIn[2]  = {1, 8};
    unsigned            sizesOut[2] = {1, 16};
    synSliceParamsV2    params      = createDefaultSliceParams(sizesOut, 2);
    params.steps[1]                 = 2;

    unsigned in    = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesIn, 2);
    unsigned out   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizesOut, 2);
    unsigned shape = createShapeTensor(INPUT_TENSOR, sizesOut, sizesOut, 2);

    addNodeToGraph(NodeFactory::stridedSliceGradNodeTypeName,
                   {in, shape},
                   {out},
                   &params,
                   sizeof(params),
                   "slice_grad");

    compileTopology();
    runTopology();

    float* outData = (float*)m_hostBuffers[out];
    float* inData  = (float*)m_hostBuffers[in];

    for (unsigned i = 0; i < sizesOut[0] * sizesOut[1]; i++)
    {
        float expected = i % 2 == 0 ? inData[i / 2] : 0.f;
        ASSERT_EQ(outData[i], expected);
    }
}

TEST_F_GC(SynGaudiDynamicSliceGradTest, dynamic_memset)
{
    unsigned sizesOut[1] = {8};

    unsigned out   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizesOut, 1);
    unsigned shape = createShapeTensor(INPUT_TENSOR, sizesOut, sizesOut, 1);

    addNodeToGraph(NodeFactory::memsetNodeTypeName, {shape}, {out}, nullptr, 0, "memset");

    compileTopology();
    runTopology();

    float* outData = (float*)m_hostBuffers[out];

    for (unsigned i = 0; i < sizesOut[0]; i++)
    {
        ASSERT_EQ(outData[i], 0.f);
    }
}
