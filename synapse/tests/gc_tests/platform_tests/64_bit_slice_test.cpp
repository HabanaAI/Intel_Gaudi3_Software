#include "infra/gtest_macros.h"
#include "infra/gc_base_test.h"
#include "data_collector.h"
#include "data_provider.h"
#include "launcher.h"
#include "node_factory.h"

using namespace gc_tests;

// currently disabled due to high host memory requirements (can still be run on multiple card host)
// enable test once [SW-111743] is in
GC_TEST_F_EXC(SynTrainingRunTest, DISABLED_huge_slice_test_ASIC)
{
    syn::Graph graph = m_ctx.createGraph(m_deviceType);

    std::vector<TSize> sizesIn     = {TSize(128) * TSize(45854381)};
    std::vector<TSize> sizesOut    = {TSize(128) * TSize(20243)};
    std::vector<TSize> sizesStarts = {TSize(128) * TSize(45833188)};
    std::vector<TSize> sizesSteps  = {1};

    syn::Tensor inputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "input");
    {
        inputTensor.setGeometry(sizesIn, synGeometryMaxSizes);
        inputTensor.setDeviceLayout({}, syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        inputTensor.assignToSection(section, 0);
    }

    syn::Tensor shapeTensor = graph.createTensor(synTensorType::SHAPE_TENSOR, "shape");
    {
        shapeTensor.setGeometry(sizesOut, synGeometryMaxSizes);
        shapeTensor.setDeviceLayout({}, syn_type_uint64);
    }

    syn::Tensor stepsTensor = graph.createTensor(synTensorType::SHAPE_TENSOR, "steps");
    {
        stepsTensor.setGeometry(sizesSteps, synGeometryMaxSizes);
        stepsTensor.setDeviceLayout({}, syn_type_uint64);
    }

    syn::Tensor startsTensor = graph.createTensor(synTensorType::SHAPE_TENSOR, "starts");
    {
        startsTensor.setGeometry(sizesStarts, synGeometryMaxSizes);
        startsTensor.setDeviceLayout({}, syn_type_uint64);
    }

    syn::Tensor outputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "output");
    {
        outputTensor.setGeometry(sizesOut, synGeometryMaxSizes);
        outputTensor.setDeviceLayout({}, syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        outputTensor.assignToSection(section, 0);
    }

    syn::Node node = graph.createNode({inputTensor, shapeTensor, stepsTensor, startsTensor},
                                      {outputTensor},
                                      {},
                                      NodeFactory::sliceNodeTypeName,
                                      "slice",
                                      {},
                                      {});

    syn::Recipe recipe = graph.compile("huge_slice_recipe");

    syn::Tensors                   tensors       = {inputTensor, shapeTensor, stepsTensor, startsTensor, outputTensor};
    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, tensors);
    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipe);

    Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector);

    const auto inputData  = dataProvider->getElements<float>(inputTensor.getName());
    const auto outputData = dataCollector->getElements<float>(outputTensor.getName());

    for (uint64_t i = 0; i < outputData.size(); ++i)
    {
        float expected = inputData[i + sizesStarts[0]];
        float actual   = outputData[i];
        ASSERT_EQ(expected, actual) << "result missmatch on index: " << i << ", expected: " << expected
                                    << ", actual: " << actual;
    }
}

// Enable test for gaudi [SW-141661] and gaudi3 [SW-141663]
GC_TEST_F_EXC(SynTrainingRunTest, huge_tensor_logical_op_test_ASIC_CI, {synDeviceGaudi, synDeviceGaudi3})
{
    syn::Graph graph = m_ctx.createGraph(m_deviceType);

    std::vector<TSize> sizesIn       = {TSize(1 << 5), TSize(1 << 20), TSize(1 << 7)};
    std::vector<TSize> sizesSliceOut = {TSize(1 << 5), TSize(1 << 7), TSize(1 << 7)};
    std::vector<TSize> sizesOut      = {TSize(1 << 7), TSize(1 << 7), TSize(1 << 5)};

    syn::Tensor inputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "input");
    {
        inputTensor.setGeometry(sizesIn, synGeometryMaxSizes);
        inputTensor.setDeviceLayout({}, syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        inputTensor.assignToSection(section, 0);
    }

    syn::Tensor sliceOutTensor = graph.createTensor(synTensorType::DATA_TENSOR, "slice_out");
    {
        sliceOutTensor.setGeometry(sizesSliceOut, synGeometryMaxSizes);
        sliceOutTensor.setDeviceLayout({}, syn_type_single);
    }

    syn::Tensor outputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "output");
    {
        outputTensor.setGeometry(sizesOut, synGeometryMaxSizes);
        outputTensor.setDeviceLayout({}, syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        outputTensor.assignToSection(section, 0);
    }

    synSliceParamsNDims sliceParams = {{0}};
    sliceParams.axes[0]             = 0;
    sliceParams.starts[0]           = 0;
    sliceParams.ends[0]             = sizesSliceOut[0];
    sliceParams.steps[0]            = 1;

    sliceParams.axes[1]   = 1;
    sliceParams.starts[1] = 1 << 2;
    sliceParams.ends[1]   = sliceParams.starts[1] + sizesSliceOut[1];
    sliceParams.steps[1]  = 1;

    sliceParams.axes[2]   = 2;
    sliceParams.starts[2] = 0;
    sliceParams.ends[2]   = sizesSliceOut[2];
    sliceParams.steps[2]  = 1;

    syn::Node slice = graph.createNode({inputTensor},
                                       {sliceOutTensor},
                                       &sliceParams,
                                       sizeof(sliceParams),
                                       NodeFactory::sliceNodeTypeName,
                                       "slice");

    synTransposeParamsNDims transposeParams;
    transposeParams.tensorDim      = 3;
    transposeParams.permutation[0] = 1;
    transposeParams.permutation[1] = 2;
    transposeParams.permutation[2] = 0;

    syn::Node transpose = graph.createNode({sliceOutTensor},
                                           {outputTensor},
                                           &transposeParams,
                                           sizeof(transposeParams),
                                           NodeFactory::transposeNodeTypeName,
                                           "transpose");

    syn::Recipe recipe = graph.compile("huge_tensor_logical_op_recipe");

    syn::Tensors                   tensors       = {inputTensor, outputTensor};
    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, tensors);
    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipe);

    Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector);

    const auto inputData  = dataProvider->getElements<float>(inputTensor.getName());
    const auto outputData = dataCollector->getElements<float>(outputTensor.getName());

    uint64_t inputStride[3] = {1};
    inputStride[0]          = sliceParams.steps[0];
    inputStride[1]          = sliceParams.steps[1] * sizesIn[0];
    inputStride[2]          = sliceParams.steps[2] * sizesIn[0] * sizesIn[1];

    uint64_t outputStride[3] = {1};
    outputStride[0]          = sizesOut[1] * sizesOut[0];
    outputStride[1]          = 1;
    outputStride[2]          = sizesOut[0];

    for (uint64_t b = 0; b < sizesSliceOut[2]; b++)
    {
        for (uint64_t r = 0; r < sizesSliceOut[1]; r++)
        {
            for (uint64_t c = 0; c < sizesSliceOut[0]; c++)
            {
                uint64_t inputOffset = (b + sliceParams.starts[2]) * inputStride[2] +
                                       (r + sliceParams.starts[1]) * inputStride[1] +
                                       (c + sliceParams.starts[0]) * inputStride[0];
                uint64_t outputOffset = b * outputStride[2] + r * outputStride[1] + c * outputStride[0];
                float    expected     = inputData[inputOffset];
                float    actual       = outputData[outputOffset];
                ASSERT_EQ(expected, actual) << "result mismatch on index: (" << c << ',' << r << "," << b
                                            << "), expected: " << expected << ", actual: " << actual;
            }
        }
    }
}
