#include "infra/gtest_macros.h"
#include "infra/gc_base_test.h"
#include "data_collector.h"
#include "data_provider.h"
#include "launcher.h"
#include "node_factory.h"

using namespace gc_tests;

// TODO enable for gaudi3 [SW-141663]
GC_TEST_F_EXC(SynTrainingRunTest, huge_dma_test_split_ASIC, {synDeviceGaudi3})
{
    syn::Graph graph = m_ctx.createGraph(m_deviceType);

    std::vector<TSize> sizesIn  = {TSize(559104), TSize(6144)};
    std::vector<TSize> sizesOut = {TSize(1024), TSize(6144)};
    syn::Tensors       splitOutputTensors;

    syn::Tensor inputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "input");
    {
        inputTensor.setGeometry(sizesIn, synGeometryMaxSizes);
        inputTensor.setDeviceDataType(syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        inputTensor.assignToSection(section, 0);
    }
    for (unsigned i = 0; i < 546; ++i)
    {
        auto        str           = fmt::format("output_{}", i);
        syn::Tensor outputTensor1 = graph.createTensor(synTensorType::DATA_TENSOR, str);
        {
            outputTensor1.setGeometry(sizesOut, synGeometryMaxSizes);
            outputTensor1.setDeviceDataType(syn_type_single);
            syn::Section section = graph.createSection();
            section.setPersistent(true);
            outputTensor1.assignToSection(section, 0);
        }
        splitOutputTensors.push_back(outputTensor1);
    }
    synSplitParams params;
    params.axis     = 0;
    syn::Node node1 = graph.createNode({inputTensor},
                                       splitOutputTensors,
                                       &params,
                                       sizeof(params),
                                       NodeFactory::splitNodeTypeName,
                                       "split",
                                       {},
                                       {});

    syn::Tensor outputTensor = graph.createTensor(synTensorType::DATA_TENSOR, "output");
    {
        outputTensor.setGeometry(sizesOut, synGeometryMaxSizes);
        outputTensor.setDeviceDataType(syn_type_single);
        syn::Section section = graph.createSection();
        section.setPersistent(true);
        outputTensor.assignToSection(section, 0);
    }

    syn::Node node2 =
        graph
            .createNode({splitOutputTensors[0]}, {outputTensor}, {}, NodeFactory::memcpyNodeTypeName, "memcpy", {}, {});

    syn::Recipe recipe = graph.compile("huge_dma_recipe");

    syn::Tensors tensors = splitOutputTensors;
    tensors.push_back(outputTensor);
    tensors.insert(tensors.begin(), inputTensor);
    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, tensors);
    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipe);

    Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector);

    const auto inputData  = dataProvider->getElements<float>(inputTensor.getName());
    const auto outputData = dataCollector->getElements<float>(outputTensor.getName());

    for (uint64_t i = 0; i < sizesOut[1]; ++i)
    {
        for (uint64_t j = 0; j < sizesOut[0]; ++j)
        {
            float expected = inputData[j + i * sizesIn[0]];
            float actual   = outputData[j + i * sizesOut[0]];
            ASSERT_EQ(expected, actual) << "result missmatch on index: " << i << ", expected: " << expected
                                        << ", actual: " << actual;
        }
    }
}
