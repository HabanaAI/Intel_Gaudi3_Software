#pragma once

#include "base_test.h"
#include "playback_tests.h"
#include <memory>

namespace json_tests
{
struct Result
{
    uint64_t                       dataIteration;
    std::shared_ptr<DataCollector> dataCollector;
};

class ConfigCompareTest : public PlaybackTest
{
public:
    ConfigCompareTest(const ArgParser& args);

    void runTest(bool run) override;
    std::vector<Result>
    runRecipe(const size_t index, const syn::Recipe& recipe, const std::shared_ptr<DataProvider>& dataProvider);

    Result runIteration(const size_t                         index,
                        const syn::Recipe&                   recipe,
                        const std::shared_ptr<DataProvider>& dataProvider,
                        uint64_t                             dataIteration,
                        bool                                 metadataOnly);

    virtual void validateData(const size_t                                   index,
                              const uint64_t                                 dataIteration,
                              const std::shared_ptr<DataComparator::Config>& compConfig,
                              const std::shared_ptr<DataContainer>           referenceData,
                              const std::shared_ptr<DataContainer>           actualData);

private:
    void initRunsConfigurations(const ArgParser& args);
    void generateRunsConfigurationsFromConfigFile(const std::string& configFilePath);
    void generateRunsConfigurationsFromConfigsList(const std::vector<std::string>& configs);

    RunsConfigurations m_runsConfs;
};
}  // namespace json_tests