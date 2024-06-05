#include "base_test.h"
#include "config_compare_tests.h"
#include "consistency_tests.h"
#include "db_parser.h"
#include "model_tests.h"
#include "multi_threaded_playback_tests.h"
#include "performance_playback_tests.h"
#include "playback_tests.h"
#include "recipe_runner.h"
#include "utils/arg_parse.h"
#include "utils/launcher.h"

#include <iostream>

using namespace json_tests;

const std::set<std::string> supportedDevices = {"greco", "gaudi", "gaudiM", "gaudi2", "gaudi3"};

ArgParser createBaseParser(const std::string& name)
{
    ArgParser p(name);

    p.add<std::string>("-j", an_json_file, "", "json file path", {}, true);
    p.add<std::string>("-c", an_device_type, "", "device type", supportedDevices, true);
    p.add("", an_eager, false, "compile in eager mode");
    p.add<std::string>("",
                       an_compilation_mode,
                       "from_recodring",
                       "force compilation mode",
                       {"graph", "eager", "from_recodring"});
    p.add("", an_exclude_graphs, false, fmt::format("{} should be excluded", an_graphs_indices));
    p.addMulti<uint64_t>("", an_graphs_indices, "", "graph indices", {});
    p.add<std::string>("", an_stats_file, "", "statistics output file path", {});
    p.add<std::string>("-d", an_data_file, "", "tensors data file path", {});
    p.add<std::string>("", an_const_data_only, "", "use only const tensors data", {});
    p.add<std::string>("", an_comp_config_file, "", "data comparator config file path", {});
    p.addMulti<uint64_t>("", an_groups, "", "run only graphs from specific groups, if not set, run all groups", {});
    p.add<std::string>("", an_serialize_recipe, "", "serialize the compiled recipe to the provided folder path", {});
    p.addMulti<uint64_t>("",
                         an_run_iter_filter,
                         "",
                         "run only specific iterations, if not set, run all iterations",
                         {});
    p.add("", an_reset_device, false, "release and acquire the device before each graph run");
    p.addMulti<Launcher::TimeMeasurement>("",
                                          an_time_measurement,
                                          Launcher::timeMeasurementToString(Launcher::TimeMeasurement::EVENETS),
                                          "set time measurement mechanism",
                                          {Launcher::timeMeasurementToString(Launcher::TimeMeasurement::NONE),
                                           Launcher::timeMeasurementToString(Launcher::TimeMeasurement::EVENETS),
                                           Launcher::timeMeasurementToString(Launcher::TimeMeasurement::PROFILER)});

    return p;
}

ArgParser createBasePlaybackParser(const std::string& name)
{
    ArgParser p = createBaseParser(name);

    p.add("", an_run, false, "run graphs");
    p.add("", an_synthetic_data, false, "run with synthetic tensors data");
    p.add<uint64_t>("", an_test_iter, "1", "number of test iterations", {});
    p.add<uint64_t>("", an_run_iter, "1", "number of run iterations per test iteration", {});
    p.add("", an_keep_going, false, "continue on compilation failure");
    p.add("",
          an_quiet,
          false,
          "suppress prints and only print progress every approx 0.05 of the iterations. Use together with "
          "high --test_iters counts.");

    return p;
}

int runDbFileParserParser(int argc, char* argv[])
{
    ArgParser p("db_parser");

    p.add<std::string>("-d", an_data_file, "", "tensors data file path", {});
    p.add<std::string>("-g", an_graph_name, "", "graph name (wildcards OK)", {});
    p.add<std::string>("-t", an_tensor_name, "", "tensor name (wildcards OK)", {});
    p.add<std::string>("-o", an_output_file, "", "output file name", {});
    p.add<uint64_t>("", an_group, "", "filter graph group", {});
    p.add<uint64_t>("-i", an_data_iter, "", "filter data iteration", {});
    p.add<uint64_t>("-l", an_element_limit, "", "limit number of elements printed", {});
    p.add("-b", an_binary, false, "dump tensor data in binary");
    p.add("-s", an_split_files, false, "output each tensor to separate file (default in binary)");
    p.add("", an_find_nans, false, "find NaNs (do not save tensor data)");
    p.add("", an_find_infs, false, "find infinities (do not save tensor data)");

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    DbParser test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runConfigCompareTest(int argc, char* argv[])
{
    ArgParser p = createBaseParser("config-compare");

    p.addMulti<std::string>("",
                            an_config_compare_values,
                            "",
                            "one or more configuration pairs of this type: <config_name, config_value>, "
                            "each pair is set on its own compilation and run process. In case only one "
                            "pair is specified then it is compared to the default configurations.",
                            {});
    p.add<std::string>("",
                       an_config_compare_file,
                       "",
                       "json file that contains test runs configurations. The file format is as follows: list of "
                       "runs configurations where each run configurations are written "
                       "as dictionary such that key=<config_name> and value=<config_value>",
                       {});
    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    ConfigCompareTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runPlaybackTest(int argc, char* argv[])
{
    ArgParser p = createBasePlaybackParser("playback");

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    PlaybackTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runPlaybackPerfTest(int argc, char* argv[])
{
    ArgParser p = createBasePlaybackParser("st_perf");

    p.addMulti<std::string>("", an_synapse_api_funcs, "", "SynapseAPI functions we wish to measure", {});

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    PerformancePlaybackTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runMultiThreadedPerfTest(int argc, char* argv[])
{
    ArgParser p = createBasePlaybackParser("mt_perf");

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    MultiThreadedPlaybackTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runModelTest(int argc, char* argv[])
{
    ArgParser p("model");

    p.add<std::string>("", an_com_file, "", "com file path", {}, true);
    p.add("", an_compile, false, "compile graphs");
    p.add("", an_run, false, "run graphs");

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    ModelTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runRecipeTest(int argc, char* argv[])
{
    ArgParser p("recipe");

    p.add<std::string>("", an_recipe_file, "", "recipe file path", {}, true);

    p.add<std::string>("", an_device_type, "", "device type", supportedDevices, true);
    p.add<uint64_t>("", an_run_iter, "1", "number of run iterations", {});
    p.add<std::string>("", an_stats_file, "", "statistics output file path", {});
    p.addMulti<Launcher::TimeMeasurement>("",
                                          an_time_measurement,
                                          Launcher::timeMeasurementToString(Launcher::TimeMeasurement::EVENETS),
                                          "set time measurement mechanism",
                                          {Launcher::timeMeasurementToString(Launcher::TimeMeasurement::NONE),
                                           Launcher::timeMeasurementToString(Launcher::TimeMeasurement::EVENETS),
                                           Launcher::timeMeasurementToString(Launcher::TimeMeasurement::PROFILER)});

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    RecipeRunner test(p);
    test.run();

    return EXIT_SUCCESS;
}

int runConsistencyTest(int argc, char* argv[])
{
    ArgParser p("consistency");

    p.add<std::string>("-j", an_json_file, "", "json file path", {}, true);
    p.add<std::string>("", an_device_type, "", "device type", supportedDevices, true);
    p.add("", an_exclude_graphs, false, fmt::format("{} should be excluded", an_graphs_indices));
    p.addMulti<uint64_t>("", an_graphs_indices, "", "graph indices", {});
    p.add<uint64_t>("", an_test_iter, "1", "number of test iterations", {});
    p.addMulti<uint64_t>("", an_groups, "", "run only graphs from specific groups, if not set, run all groups", {});
    p.add("", an_keep_going, false, "continue on inconsistency");

    p.parse(argc, argv);

    if (p.helpRequested())
    {
        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }

    ConsistencyTest test(p);
    test.run();

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    try
    {
        ArgParser p;
        auto      dbParser     = p.add("", "db_parser", false, "synrec database file parser");
        auto      playbackTest = p.add("", "playback", false, "playback test - compile and run from json file");
        auto      playbackPerfTest =
            p.add("", "st_perf", false, "playback perf test - dedicated mode for more accurate time measurments");
        auto multiThreadedPerfTest =
            p.add("",
                  "mt_perf",
                  false,
                  "multi threaded playback perf test - a mode mimicking the bridge eager pipeline");
        auto modelTest       = p.add("", "model", false, "model test - automated performance test");
        auto recipeTest      = p.add("", "recipe", false, "recipe test - run pre-compiled recipe file");
        auto consistencyTest = p.add("", "consistency", false, "consistency test - compile from json file and compare");
        auto configCompareTest = p.add(
            "",
            "config-compare",
            false,
            "config compare test - an accuracy compare between runs of the same graphs with different configurations");

        p.parse(2, argv);

        if (dbParser.getValue()) return runDbFileParserParser(argc, argv);
        if (playbackTest.getValue()) return runPlaybackTest(argc, argv);
        if (playbackPerfTest.getValue()) return runPlaybackPerfTest(argc, argv);
        if (multiThreadedPerfTest.getValue()) return runMultiThreadedPerfTest(argc, argv);
        if (modelTest.getValue()) return runModelTest(argc, argv);
        if (recipeTest.getValue()) return runRecipeTest(argc, argv);
        if (consistencyTest.getValue()) return runConsistencyTest(argc, argv);
        if (configCompareTest.getValue()) return runConfigCompareTest(argc, argv);

        std::cout << p.getHelp() << std::endl;
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        JT_LOG_ERR("json tests run failed, error: " << e.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        JT_LOG_ERR("json tests run failed with unknown error");
        return EXIT_FAILURE;
    }
}
