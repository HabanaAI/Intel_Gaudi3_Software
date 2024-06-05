#include "data_collector.h"
#include "data_provider.h"
#include "hpp/syn_recipe.hpp"
#include "infra/gc_base_test.h"
#include "infra/gtest_macros.h"
#include "launcher.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include <string>

using namespace gc_tests;

class SynTrainingInfraTests : public SynTrainingRunTest
{
protected:
    SynTrainingInfraTests() : SynTrainingRunTest(), m_graph(m_ctx.createGraph(m_deviceType)) {}

    syn::Tensor createTensor(synTensorType             type,
                             synDataType               dataType,
                             const std::string&        name,
                             const std::vector<TSize>& maxShape,
                             const std::vector<TSize>& minShape = {})
    {
        syn::Tensor tensor = m_graph.createTensor(type, name);
        tensor.setGeometry(maxShape, synGeometryMaxSizes);
        if (!minShape.empty())
        {
            tensor.setGeometry(minShape, synGeometryMinSizes);
        }
        tensor.setDeviceLayout({}, dataType);
        return tensor;
    }

    syn::Tensor createPersistentTensor(synTensorType             type,
                                       synDataType               dataType,
                                       const std::string&        name,
                                       const std::vector<TSize>& maxShape,
                                       const std::vector<TSize>& minShape = {})
    {
        syn::Tensor  tensor  = createTensor(type, dataType, name, maxShape, minShape);
        syn::Section section = m_graph.createSection();
        section.setPersistent(true);
        tensor.assignToSection(section, 0);
        return tensor;
    }

    syn::Graph m_graph;
};

class SynTrainingLauncherTests : public SynTrainingInfraTests
{
protected:
    SynTrainingLauncherTests() : SynTrainingInfraTests()
    {
        m_input1Tensor =
            createPersistentTensor(synTensorType::DATA_TENSOR, synDataType::syn_type_float, "input1", {1, 1000});
        m_input2Tensor =
            createPersistentTensor(synTensorType::DATA_TENSOR, synDataType::syn_type_int32, "input2", {1000});
        m_inputShapeTensor = createTensor(synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR,
                                          synDataType::syn_type_uint32,
                                          "shape",
                                          {1, 500},
                                          {1, 2});
        m_outputTensor     = createPersistentTensor(synTensorType::DATA_TENSOR_DYNAMIC,
                                                synDataType::syn_type_float,
                                                "output",
                                                {1, 500},
                                                {1, 2});

        syn::Node node = m_graph.createNode({m_input1Tensor, m_input2Tensor, m_inputShapeTensor},
                                            {m_outputTensor},
                                            {0, -72, 0, 0},
                                            "unsorted_segment_sum_fwd_f32",
                                            "sum");

        m_recipe = m_graph.compile(m_testName);
    }

    syn::Tensor createTensor(synTensorType             type,
                             synDataType               dataType,
                             const std::string&        name,
                             const std::vector<TSize>& maxShape,
                             const std::vector<TSize>& minShape = {})
    {
        syn::Tensor tensor = m_graph.createTensor(type, name);
        tensor.setGeometry(maxShape, synGeometryMaxSizes);
        if (!minShape.empty())
        {
            tensor.setGeometry(minShape, synGeometryMinSizes);
        }
        tensor.setDeviceLayout({}, dataType);
        return tensor;
    }

    syn::Tensor createPersistentTensor(synTensorType             type,
                                       synDataType               dataType,
                                       const std::string&        name,
                                       const std::vector<TSize>& maxShape,
                                       const std::vector<TSize>& minShape = {})
    {
        syn::Tensor  tensor  = createTensor(type, dataType, name, maxShape, minShape);
        syn::Section section = m_graph.createSection();
        section.setPersistent(true);
        tensor.assignToSection(section, 0);
        return tensor;
    }

    syn::Tensor m_input1Tensor;
    syn::Tensor m_input2Tensor;
    syn::Tensor m_inputShapeTensor;
    syn::Tensor m_outputTensor;
    syn::Recipe m_recipe;
};

GC_TEST_F(SynTrainingInfraTests, basic_compile_and_run)
{
    syn::Tensor input1Tensor =
        createPersistentTensor(synTensorType::DATA_TENSOR, synDataType::syn_type_float, "input1", {1});
    syn::Tensor input2Tensor =
        createPersistentTensor(synTensorType::DATA_TENSOR, synDataType::syn_type_float, "input2", {1});
    syn::Tensor outputTensor =
        createPersistentTensor(synTensorType::DATA_TENSOR, synDataType::syn_type_float, "output", {1});

    syn::Node node = m_graph.createNode({input1Tensor, input2Tensor}, {outputTensor}, {}, "add_f32", "add_f32");

    syn::Recipe recipe = m_graph.compile("basic_compile_and_run");

    syn::Tensors                   tensors       = {input1Tensor, input2Tensor, outputTensor};
    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, tensors);
    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(recipe);

    Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector);

    const auto input1Data = dataProvider->getElements<float>(input1Tensor.getName());
    const auto input2Data = dataProvider->getElements<float>(input2Tensor.getName());
    const auto outputData = dataCollector->getElements<float>(outputTensor.getName());

    for (size_t i = 0; i < outputData.size(); ++i)
    {
        float expected = input1Data[i] + input2Data[i];
        float actual   = outputData[i];
        EXPECT_EQ(expected, actual) << "result missmatch on index: " << i << ", expected: " << expected
                                    << ", actual: " << actual << ", in1: " << input1Data[i]
                                    << ", in2: " << input2Data[i];
    }
}

GC_TEST_F_INC(SynTrainingLauncherTests, launcher_with_data, synDeviceGaudi,synDeviceGaudi2)
{
    syn::Tensors                   tensors       = {m_input1Tensor, m_input2Tensor, m_inputShapeTensor, m_outputTensor};
    std::shared_ptr<DataProvider>  dataProvider  = std::make_shared<RandDataProvider>(0, 100.0, tensors);
    std::shared_ptr<DataCollector> dataCollector = std::make_shared<DataCollector>(m_recipe);

    Launcher::launch(m_device, m_recipe, 1, dataProvider, dataCollector);
}

GC_TEST_F_INC(SynTrainingLauncherTests, launcher_without_data, synDeviceGaudi,synDeviceGaudi2)
{
    Launcher::launch(m_device, m_recipe, 1);
}