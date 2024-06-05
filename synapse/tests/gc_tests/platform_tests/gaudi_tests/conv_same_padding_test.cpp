#include "infra/cpu_calculator.h"
#include "utils.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "syn_singleton.hpp"
#include <memory>

// default formatting by clang is unreadable

class SynGaudiConvSamePaddingTest : public SynGaudiTestInfra
{
public:
    SynGaudiConvSamePaddingTest()
    {
        setTestPackage(TEST_PACKAGE_CONVOLUTION);
        setSupportedDevices({synDeviceGaudi});
    }

    void runTest (unsigned input_size, unsigned min_input_size, unsigned act_input_size, unsigned kernel_size, unsigned stride=2,
                  synConvolutionPaddingType postGraphPadding = PADDING_SAME, bool expectManyNodes=false,
                  bool expectSpatialSlice = false)
    {
        const unsigned kernel_dims = 4;
        unsigned kernel_sizes[] = {1, 1, kernel_size, kernel_size};

        const unsigned input_dims = 4;
        unsigned input_sizes[] = {1, input_size, input_size, 1};
        unsigned min_input_sizes[] = {1, min_input_size, min_input_size, 1};

        const unsigned output_size = (input_size + stride - 1) / stride;
        const unsigned min_output_size = (min_input_size + stride - 1) / stride;
        const unsigned output_dims = 4;
        unsigned output_sizes[] = {1, output_size, output_size, 1};
        unsigned min_output_sizes[] = {1, min_output_size, min_output_size, 1};

        // Calculate padding.
        // Even though for padding mode PADDING_SAME the amount of padding
        // is completely determined by input, kernel, and stride sizes,
        // Synapse still requires correct padding to be provided in node
        // parameters, and validates provided values.

        const unsigned pad        = std::max(0U, (output_size - 1) * stride + kernel_size - input_size);
        const unsigned pad_before = pad / 2;
        const unsigned pad_after  = pad - pad_before;

        auto xTensorIndex = createPersistTensor(INPUT_TENSOR,
                MEM_INIT_RANDOM_POSITIVE,
                nullptr,
                input_sizes,
                input_dims,
                syn_type_single,
                nullptr,
                nullptr,
                0,
                0,
                nullptr,
                min_input_sizes);

        auto wTensorIndex = createPersistTensor(INPUT_TENSOR,
                MEM_INIT_RANDOM_POSITIVE,
                nullptr,
                kernel_sizes,
                kernel_dims,
                syn_type_single);

        auto yTensorIndex = createPersistTensor(OUTPUT_TENSOR,
                MEM_INIT_ALL_ZERO,
                nullptr,
                output_sizes,
                output_dims,
                syn_type_single,
                nullptr,
                nullptr,
                0,
                0,
                nullptr,
                min_output_sizes);

        synConvolutionParamsV2 params(/* kernel   */ kernel_size, kernel_size,
                /* stride   */ stride, stride,
                /* padding  */ pad_before, pad_after, pad_before, pad_after,
                /* dilation */ 1, 1,
                /* pad type */ synConvolutionPaddingType::PADDING_SAME);

        addNodeToGraph(NodeFactory::convolutionNodeTypeName,
                {xTensorIndex, wTensorIndex},
                {yTensorIndex},
                (void*)&params,
                sizeof(params));

        unsigned act_input_sizes[] = {1, act_input_size, act_input_size, 1};

        const unsigned act_output_size = (act_input_size + stride - 1) / stride;
        unsigned act_output_sizes[] = {1, act_output_size, act_output_size, 1};

        compileTopology();

        // walk the graph and check the nodes
        //
        //
        GraphData&   graphData = getGraph(0);
        HabanaGraph* g = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

        unsigned convNodeCount = 0;
        for (const auto& node: g->getNodes())
        {
            const auto& convNode = std::dynamic_pointer_cast<ConvBaseNode>(node);
            if (convNode != nullptr)
            {
                EXPECT_EQ(postGraphPadding, convNode->getConvolutionParams().paddingType);
                convNodeCount++;
                auto inputShape = convNode->getInputs()[0]->getAllSizesInElements();
                EXPECT_EQ(expectSpatialSlice,
                          inputShape[1] != input_size || inputShape[2] != input_size);
            }

        }

        std::cerr << "\033[1;34m" << "Found " << convNodeCount << " convolution nodes" << "\033[0m\n";
        EXPECT_EQ(convNodeCount > 1, expectManyNodes);

        setActualSizes(xTensorIndex, act_input_sizes);
        setActualSizes(yTensorIndex, act_output_sizes);
        runTopology();

        // Calculate convolution reference result by hand, instead of calling checkFwdConvolution.
        // If the assert below ever fails, the calculation is right here and can be easily
        // verified in the debugger.

        auto kernel_buffer = static_cast<float*>(m_hostBuffers[wTensorIndex]);
        auto input_buffer = static_cast<float*>(m_hostBuffers[xTensorIndex]);
        auto output_buffer = static_cast<float*>(m_hostBuffers[yTensorIndex]);

        const unsigned act_pad        = std::max(0U, (act_output_size - 1) * stride + kernel_size - act_input_size);
        const unsigned act_pad_before = act_pad / 2;

        for (int i = 0, ii = -act_pad_before; i < act_output_size; ++i, ii += stride)
        {
            std::cerr << "\033[1;34m" << "Checking line " << i << " ...." << "\033[0m\r";
            for (int j = 0, jj = -act_pad_before; j < act_output_size; ++j, jj += stride)
            {
                float sum = 0;
                for (int k = 0; k < kernel_size; ++k)
                {
                    for (int l = 0; l < kernel_size; ++l)
                    {
                        int ik = ii + k;
                        int jl = jj + l;
                        if (ik >= 0 && jl >= 0 && ik < act_input_size && jl < act_input_size)
                        {
                            sum += input_buffer[ik * act_input_size + jl] * kernel_buffer[k * kernel_size + l];
                        }
                    }
                }

                auto idx = i * act_output_size + j;
                ASSERT_NEAR(sum, output_buffer[idx], 0.01) << "Discrepancy at index [" << i << ", " << j << "]";
            }
        }
        std::cerr << "\n\033[1;35m" << "Done                                  " << "\033[0m\n";
    }

};

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_even_odd)
{
    runTest(200, 5, 177, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_odd_even)
{
    runTest(201, 5, 180, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_small_odd_odd)
{
    runTest(7, 4, 5, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_small_odd_even)
{
    runTest(7, 4, 6, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_small_even_even)
{
    runTest(8, 4, 6, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_small_even_odd)
{
    runTest(8, 4, 7, 3);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_large)
{
    runTest(2048, 15, 1577, 11);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_padding_huge)
{
    runTest(4000, 30, 1377, 17);
}

TEST_F_GC(SynGaudiConvSamePaddingTest, same_to_explicit_conv)
{
    runTest(4000, 30, 1377, 17, 1, synConvolutionPaddingType::PADDING_EXPLICIT, true, true);
}
