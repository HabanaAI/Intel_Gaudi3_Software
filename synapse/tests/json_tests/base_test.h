#pragma once

#include "file_loader.h"
#include "graph_loader.h"
#include "hpp/syn_context.hpp"
#include "hpp/syn_recipe.hpp"
#include "json.hpp"
#include "settable.h"
#include "utils/arg_parse.h"
#include "utils/data_collector.h"
#include "utils/data_comparator.h"
#include "utils/launcher.h"

#include <memory>
#include <set>
// colored console prints, info: green, warning: yellow, error: red

#define JT_LOG_INFO(msg_)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << "\033[1;32m" << msg_ << "\033[0m" << std::endl;                                                   \
    } while (false)
#define JT_LOG_INFO_NON_QUIET(msg_)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (m_quietMode) break;                                                                                        \
        std::cout << "\033[1;32m" << msg_ << "\033[0m" << std::endl;                                                   \
    } while (false)
#define JT_LOG_WARN(msg_)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << "\033[1;33m" << msg_ << "\033[0m" << std::endl;                                                   \
    } while (false)
#define JT_LOG_ERR(msg_)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cerr << "\033[1;31m" << msg_ << "\033[0m" << std::endl;                                                   \
    } while (false)

class DataProvider;

namespace json_tests
{
const std::string an_com_file              = "--com-file";
const std::string an_comp_config_file      = "--comp-config-file";
const std::string an_compile               = "--compile";
const std::string an_config_compare_file   = "--config-compare-file";
const std::string an_config_compare_values = "--config-compare-values";
const std::string an_data_file             = "--data-file";
const std::string an_const_data_only       = "--const";
const std::string an_device_type           = "--device-type";
const std::string an_eager                 = "--eager";
const std::string an_compilation_mode      = "--compilation-mode";
const std::string an_exclude_graphs        = "--exclude-graphs";
const std::string an_graphs_indices        = "--graphs-indices";
const std::string an_graph_name            = "--graph-name";
const std::string an_groups                = "--groups";
const std::string an_group                 = "--group";
const std::string an_json_file             = "--json-file";
const std::string an_keep_going            = "--keep-going";
const std::string an_quiet                 = "--quiet";
const std::string an_recipe_file           = "--recipe-file";
const std::string an_reset_device          = "--reset-device";
const std::string an_run                   = "--run";
const std::string an_run_iter              = "--run-iter";
const std::string an_run_iter_filter       = "--run-iter-filter";
const std::string an_serialize_recipe      = "--serialize-recipe";
const std::string an_stats_file            = "--stats-file";
const std::string an_synapse_api_funcs     = "--synapse-api-funcs";
const std::string an_synthetic_data        = "--synthetic-data";
const std::string an_test_iter             = "--test-iter";
const std::string an_time_measurement      = "--time-measurement";
const std::string an_tensor_name           = "--tensor_name";
const std::string an_data_iter             = "--data-iteration";
const std::string an_output_file           = "--output-file";
const std::string an_element_limit         = "--element_limit";
const std::string an_binary                = "--binary";
const std::string an_split_files           = "--split-files";
const std::string an_find_nans             = "--nans";
const std::string an_find_infs             = "--infs";

synDeviceType deviceTypeFromString(const std::string& name);

template<class T>
std::optional<T> getOptionalArg(const ArgParser& args, const std::string& name)
{
    std::optional<T> ret;
    if (args.getArg<T>(name).hasValue())
    {
        ret = args.getArg<T>(name).getValue();
    }
    return ret;
}

struct HabanaGlobalConfig
{
    std::string configName;
    std::string configValue;
};

using RunsConfigurations = std::vector<std::vector<HabanaGlobalConfig>>;

class BaseTest
{
public:
    BaseTest()          = default;
    virtual ~BaseTest() = default;

    virtual void run() = 0;

protected:
    syn::Context m_ctx;
};

class TypedDeviceTest : public BaseTest
{
public:
    TypedDeviceTest(const ArgParser& args);
    virtual ~TypedDeviceTest() = default;

    virtual syn::Recipe compileGraph(const size_t index, const JsonGraphLoader& gl);

protected:
    synDeviceType           m_deviceType;
    std::set<synDeviceType> m_optionalDeviceTypes;
    nlohmann_hcl::json      m_stats;
    std::string             m_statsFilePath;
    bool                    m_quietMode = false;
    bool                    m_keepGoing = false;
};

std::optional<CompilationMode> stringToCompilationMode(const std::string& s);

class RunTypedDeviceTest : public TypedDeviceTest
{
public:
    RunTypedDeviceTest(const ArgParser& args);
    virtual ~RunTypedDeviceTest() = default;

    virtual void runGraph(const size_t                                   index,
                          const syn::Recipe&                             recipe,
                          const std::shared_ptr<DataProvider>&           dataProvider,
                          const std::shared_ptr<DataComparator::Config>& compConfig);

protected:
    void               dumpStats() const;
    std::set<uint64_t> getDataIterations(const std::shared_ptr<DataProvider>& dataProvider) const;
    std::set<uint64_t> getNonDataIterations(const std::shared_ptr<DataProvider>& dataProvider) const;
    void               runIteration(const size_t                                   index,
                                    const syn::Recipe&                             recipe,
                                    const std::shared_ptr<DataProvider>&           dataProvider,
                                    uint64_t                                       dataIteration,
                                    const std::shared_ptr<DataComparator::Config>& compConfig = nullptr);

    std::shared_ptr<DataProvider>           getDataProvider(const JsonGraphLoader& gl);
    std::shared_ptr<DataComparator::Config> getComparatorConfig(const nlohmann_hcl::json& graph);

    bool m_useSyntheticData;

    // Skip per-iteration prints (only printing approx every 5% of the iters) and skip repeated warnnings
    bool                      m_warnedAboutMeasureFailureAlready = false;
    uint64_t                  m_runIterations;
    std::string               m_dataFilePath;
    bool                      m_constDataOnly;
    std::string               m_comparatorConfigFilePath;
    syn::Device               m_device;
    Launcher::TimeMeasurement m_timeMeasurement;
    double                    m_totalDuration;
};

class JsonTest
{
public:
    JsonTest(const ArgParser& args);
    virtual ~JsonTest() = default;

protected:
    bool                    m_excludeGraphs;
    std::string             m_jsonFilePath;
    std::string             m_recipeFolderPath;
    std::vector<uint64_t>   m_graphsIndices;
    std::vector<uint64_t>   m_groups;
    const JsonFileLoaderPtr m_jsonFileLoader;
};
}  // namespace json_tests
