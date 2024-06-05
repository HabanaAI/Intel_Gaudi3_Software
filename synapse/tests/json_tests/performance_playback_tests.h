#pragma once

#include "performance_base_test.h"

namespace json_tests
{
class PerformancePlaybackTest final : private PerformanceBaseTest
{
public:
    PerformancePlaybackTest(const ArgParser& args);

    void run() override;

private:
    void runTest(bool run);
    virtual void executeGraph(size_t                      graphIndex,
                              const std::string&          model,
                              bool                        run,
                              std::map<int, std::string>& failedGraphs,
                              const nlohmann_hcl::json&   jsonGraph);
};
}  // namespace json_tests