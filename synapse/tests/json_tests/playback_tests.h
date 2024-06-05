#pragma once

#include "base_test.h"
#include "graph_loader.h"
#include <optional>
#include <string>
#include <json.hpp>
#include <vector>
#include "gc_tests/platform_tests/infra/gc_tests_utils.h"

namespace json_tests
{
class PlaybackTest
: public RunTypedDeviceTest
, public JsonTest
{
public:
    PlaybackTest(const ArgParser& args);

    void run() override;

protected:
    virtual void runTest(bool run);

    bool                  m_run;
    std::vector<uint64_t> m_iterationsFilter;

    // No value  - according to recording mode
    // Has value - force compilation mode regardless of recording mode
    std::optional<CompilationMode> m_compilationMode;

    int                   m_testIterations;

private:
    void setup();
    void loadGraphConfigs(nlohmann_hcl::json graph, std::vector<std::unique_ptr<ScopedConfig>>& configsSetter);
};
}  // namespace json_tests