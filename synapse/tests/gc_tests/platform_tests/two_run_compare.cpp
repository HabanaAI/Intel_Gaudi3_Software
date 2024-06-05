#include "data_comparator.h"
#include "data_provider.h"
#include "data_collector.h"
#include "hpp/syn_context.hpp"
#include "infra/gc_base_test.h"
#include "infra/gtest_macros.h"
#include "json_tests/graph_loader.h"
#include "json_utils.h"
#include "launcher.h"
#include "graphs.h"
#include "json_tests/file_loader.h"
#include <memory>

using namespace gc_tests;

static std::tuple<std::string, std::string, std::string, std::string, float, float>
instance(const std::string& graph,
         const std::string& confType,
         const std::string& conf1,
         const std::string& conf2  = "",
         float              minVal = -2,
         float              maxVal = 2)
{
    return std::tie(graph, confType, conf1, conf2, minVal, maxVal);
}

auto paramsTwoRun =
    ::testing::Values(instance(json_graphs::spatial_conv, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0", ""),
                      instance(json_graphs::fuse_bcast_cast_tpc, "ENABLE_BROADCAST_TPC_FUSION", "1", "0"),
                      instance(json_graphs::const_scatter_gemms, "SRAM_SLICER_MAX_CAPACITY_BYTES", "0", ""));

class TwoRunCompareJsonTest
: public SynTrainingRunParamTest<std::tuple<std::string, std::string, std::string, std::string, float, float>>
{
public:
    TwoRunCompareJsonTest() { setTestPackage(TEST_PACKAGE_COMPARE_TEST); }
};

GC_TEST_P_INC(TwoRunCompareJsonTest, DISABLED_two_run_compare_json_test, paramsTwoRun, synDeviceGaudi)
{
    auto                           params   = std::get<1>(GetParam());
    const std::string              jsonDataString = std::get<0>(params);
    const std::string              config   = std::get<1>(params);
    const std::vector<std::string> confs    = {std::get<2>(params), std::get<3>(params)};
    const float                    minVal   = std::get<4>(params);
    const float                    maxVal   = std::get<5>(params);

    std::vector<std::shared_ptr<DataCollector>> dataCollectors;

    ScopedConfig expConfigSetter(m_ctx, "ENABLE_EXPERIMENTAL_FLAGS", "1");

    const auto jsonFileLoader = JsonFileLoader::createFromJsonContent(jsonDataString);
    const auto numOfgraphs      = jsonFileLoader->getNumOfGraphs();

    for (int i = 0; i < numOfgraphs; ++i)
    {
        for (size_t i = 0; i < 2; ++i)
        {
            const auto& conf = confs[i];

            std::unique_ptr<ScopedConfig> configSetter;
            if (!conf.empty())
            {
                configSetter = std::make_unique<ScopedConfig>(m_ctx, config, conf);
            }

            auto gl = jsonFileLoader->getGraphLoader(m_ctx, m_deviceType, CompilationMode::Graph, i);

            auto dataProvider = std::static_pointer_cast<DataProvider>(
                std::make_shared<RandDataProvider>(minVal, maxVal, gl.getTensors()));

            auto recipe = gl.getGraph().compile(fmt::format("config-{}", i));

            auto dataCollector = std::make_shared<DataCollector>(recipe);
            dataCollectors.push_back(dataCollector);

            Launcher::launch(m_device, recipe, 1, dataProvider, dataCollector, Launcher::TimeMeasurement::NONE);
        }

        DataComparator dc(dataCollectors[0], dataCollectors[1]);
        auto           res = dc.compare();

        EXPECT_TRUE(res.errors.empty()) << "Data mismatch found: " << res.errors[0];
    }
}
