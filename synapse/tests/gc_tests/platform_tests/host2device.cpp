#include "data_collector.h"
#include "data_provider.h"
#include "infra/gc_base_test.h"
#include "infra/gtest_macros.h"
#include "launcher.h"
#include "node_factory.h"
#include "synapse_common_types.h"

using namespace gc_tests;

GC_TEST_F_INC(SynTrainingRunTest, DISABLED_host_to_device_tensor_new_api, synDeviceGaudi)
{
    const unsigned        range_size         = 3;
    std::vector<TSize>    range_sizes        = {range_size};
    std::vector<uint32_t> range_compile_data = {0, 100, 1, 0, 2, 1};
    std::vector<uint32_t> range_actual_data  = {0, 8, 2};
    std::vector<TSize>    outputMinSizes     = {2};
    std::vector<TSize>    outputMaxSizes     = {100};
    std::vector<TSize>    outputActualSizes  = {4};

    syn::Graph graph = m_ctx.createGraph(m_deviceType);

    syn::Section hostToDeviceSection = graph.createSection();
    hostToDeviceSection.setPersistent(true);

    std::string hostToDeviceTensorName = "host_to_device";
    syn::Tensor hostToDeviceTensor     = graph.createTensor(HOST_TO_DEVICE_TENSOR, hostToDeviceTensorName);
    hostToDeviceTensor.setGeometry(range_sizes, synGeometryMaxSizes);
    hostToDeviceTensor.setGeometry(range_sizes, synGeometryMinSizes);
    hostToDeviceTensor.setDeviceLayout({}, syn_type_uint32);
    hostToDeviceTensor.setHostPtr(range_compile_data, true);
    hostToDeviceTensor.assignToSection(hostToDeviceSection, 0);

    syn::Section outputSection = graph.createSection();
    outputSection.setPersistent(true);

    std::string outputTensorName = "output";
    syn::Tensor outputTensor     = graph.createTensor(DATA_TENSOR_DYNAMIC, outputTensorName);
    outputTensor.setGeometry(outputMaxSizes, synGeometryMaxSizes);
    outputTensor.setGeometry(outputMinSizes, synGeometryMinSizes);
    outputTensor.setDeviceLayout({}, syn_type_uint32);
    outputTensor.assignToSection(outputSection, 0);

    syn::Node node = graph.createNode({hostToDeviceTensor}, {outputTensor}, {}, "range_i32", "range");

    syn::Recipe recipe = graph.compile("host_to_device_tensor_new_api");

    std::vector<syn::Tensor> tensors = {hostToDeviceTensor, outputTensor};

    auto dataProvider  = std::make_shared<ManualDataProvider>(tensors);
    auto dataCollector = std::make_shared<DataCollector>(recipe);

    dataProvider->setShape(hostToDeviceTensorName, range_sizes);
    dataProvider->setBuffer(hostToDeviceTensor, range_actual_data);
    dataProvider->setShape(outputTensorName, outputActualSizes);
    dataProvider->setBuffer(outputTensor, std::vector<float>(outputActualSizes[0], 0));

    Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector);

    unsigned value   = range_actual_data[0];
    unsigned delta   = range_actual_data[2];
    auto     results = dataCollector->getElements<uint32_t>(outputTensorName);

    for (unsigned i = 0; i < outputActualSizes[0]; ++i, value += delta)
    {
        ASSERT_EQ(results[i], value) << "at index " << i;
    }
}
