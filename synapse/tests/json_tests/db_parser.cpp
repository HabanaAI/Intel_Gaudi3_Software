#include "db_parser.h"

#include "data_serializer/data_serializer.h"
#include "data_serializer/ds_types.h"
#include "data_type_utils.h"
#include "gc_tests/platform_tests/infra/gc_tests_utils.h"
#include "utils/data_provider.h"

#include <fnmatch.h>
#include <fstream>
#include <limits>




namespace json_tests
{
DbParser::DbParser(const ArgParser& args)
: BaseTest(),
  m_dataFilePath(args.getValueOrDefault(an_data_file, std::string())),
  m_graphName(args.getValueOrDefault(an_graph_name, std::string())),
  m_tensorName(args.getValueOrDefault(an_tensor_name, std::string())),
  m_outputFileName(args.getValueOrDefault(an_output_file, std::string())),
  m_iteration(args.getValueOrDefault(an_data_iter, (uint64_t)(-1))),
  m_elementLimit(args.getValueOrDefault(an_element_limit, std::numeric_limits<uint64_t>::max())),
  m_group(args.getValueOrDefault(an_group,  (uint64_t)(-1))),
  m_binary(args.getValueOrDefault(an_binary, false)),
  m_splitFiles(args.getValueOrDefault(an_split_files, false)),
  m_findNans(args.getValueOrDefault(an_find_nans, false)),
  m_findInfs(args.getValueOrDefault(an_find_infs, false))
{
}

static bool globMatch (const std::string& pattern, const std::string& aString)
{
    return fnmatch(pattern.c_str(), aString.c_str(), 0) == 0;
}

void DbParser::run()
{

    if (m_dataFilePath.empty())
    {
        JT_LOG_ERR("Data file is required");
        return;
    }

    runGroups();
}

void DbParser::runGroups()
{
    const auto dataDeserializer = data_serialize::DataDeserializer::create(m_dataFilePath);
    auto groups = dataDeserializer->getGroups();
    bool hasValidGroups = true;

    if (groups.empty())
    {
        JT_LOG_WARN("Querying valid groups not supported on this DB, you have to guess");
        groups.push_back(0);
        hasValidGroups = false;
    }

    bool hasGroup = (m_group != (uint64_t)(-1));
    if (hasGroup && hasValidGroups && std::find(std::begin(groups), std::end(groups), m_group) == std::end(groups))
    {
        JT_LOG_ERR("Group not found, valid groups:");
        for (auto g: groups)
        {
            JT_LOG_INFO(g);
        }
        return;
    }

    const std::vector<uint64_t>& groupsToShow = hasGroup ? std::vector<uint64_t>{m_group} : groups;

    for (auto group: groupsToShow)
    {
        if (groupsToShow.size() > 1)
        {
            JT_LOG_INFO("===================");
            JT_LOG_INFO(fmt::format("Group {}", group));
            JT_LOG_INFO("===================");
        }
        runGraphs(dataDeserializer, group);
        if (groupsToShow.size() > 1)
        {
            JT_LOG_INFO("");
        }
    }
}

void DbParser::runGraphs(const data_serialize::DataDeserializerPtr& dataDeserializer, uint64_t group)
{
    std::vector<data_serialize::GraphInfo> realGraphList;
    const auto               graphs     = dataDeserializer->getGraphsInfos();

    if (m_graphName.empty())
    {
        //  just print graph names and return
        for (const auto& g : graphs)
        {
            JT_LOG_INFO(g.id);
        }
        return;
    }

    for (const auto& g : graphs)
    {
        if (globMatch(m_graphName, g.id))
        {
            realGraphList.push_back(g);
        }
    }

    if (realGraphList.empty())
    {
        JT_LOG_ERR(fmt::format("Graph name {} not found", m_graphName));
    }

    for (const auto& g : realGraphList)
    {
        JT_LOG_INFO(fmt::format("=== Exploring graph {}", g.id));

        runIterations(g);
    }
}

void DbParser::runIterations(const data_serialize::GraphInfo& graphInfo)
{
    const auto dataDeserializer = data_serialize::DataDeserializer::create(m_dataFilePath);
    const auto graphDeserializer = dataDeserializer->getGraph(graphInfo);
    auto iterations = graphDeserializer->getDataIterations();
    auto nonDataIterations = graphDeserializer->getNonDataIterations();
    iterations.insert(nonDataIterations.begin(), nonDataIterations.end());

    bool hasIteration = (m_iteration != (uint64_t)(-1));

    const std::set<uint64_t>& iterationsToShow = hasIteration ? std::set<uint64_t>{m_iteration} : iterations;

    if (hasIteration && std::find(std::begin(iterations), std::end(iterations), m_iteration) == std::end(iterations))
    {
        JT_LOG_ERR(fmt::format("iteration {} not found", m_iteration));
        return;
    }

    for (auto iteration : iterationsToShow)
    {
        if (iterationsToShow.size() > 1)
        {
            JT_LOG_INFO("-----------------");
            JT_LOG_INFO(fmt::format("Iteration {}", iteration));
            JT_LOG_INFO("-----------------");
        }

        runTensors(graphInfo, iteration);
    }
}

void DbParser::runTensors(const data_serialize::GraphInfo& graphInfo, uint64_t iteration)
{
    CapturedDataProvider dataProvider(m_dataFilePath, graphInfo.id, graphInfo.recipeId, graphInfo.group, iteration);
    auto tensors = dataProvider.getTensorsNames();
    if (m_tensorName.empty())
    {
        // just print tensor names and return
        for (const auto& t : tensors)
        {
            auto dataType = dataProvider.getDataType(t);
            auto shape = dataProvider.getShape(t);
            JT_LOG_INFO(fmt::format("{:<80} [{:<20}] {}",
                        t,
                        toString(shape, ','),
                        getStringFromSynDataType(dataType)));
        }
        return;
    }

    std::vector<std::string> realTensorNames;

    for (const auto& t : tensors)
    {
        if (globMatch(m_tensorName, t))
        {
            realTensorNames.push_back(t);
        }
    }

    if (realTensorNames.empty())
    {
        JT_LOG_ERR(fmt::format("Tensor name {} not found", m_tensorName));
    }

    for (const auto& t: realTensorNames)
    {

        auto dataType = dataProvider.getDataType(t);
        auto shape = dataProvider.getShape(t);

        JT_LOG_INFO(fmt::format("Tensor {} : shape [{}] data type {}",
                    t,
                    toString(shape, ','),
                    getStringFromSynDataType(dataType)));

        switch (dataType)
        {
            case syn_type_int8:
                printElements<int8_t>(t, dataProvider, CharCaster());
                break;
            case syn_type_int16:
                printElements<int16_t>(t, dataProvider);
                break;
            case syn_type_int32:
                printElements<int32_t>(t, dataProvider);
                break;
            case syn_type_uint8:
                printElements<uint8_t>(t, dataProvider, CharCaster());
                break;
            case syn_type_uint16:
                printElements<uint16_t>(t, dataProvider);
                break;
            case syn_type_uint32:
                printElements<uint32_t>(t, dataProvider);
                break;
            case syn_type_bf16:
                printElements<uint16_t>(t, dataProvider, BfloatCaster());
                break;
            case syn_type_single:
                printElements<float>(t, dataProvider);
                break;
            default:
                JT_LOG_ERR(fmt::format("Unsupported data type {} ({})", dataType, getStringFromSynDataType(dataType)));
        }
    }
}

std::ostream& DbParser::getStream(const std::string& tensorName)
{
    if (m_splitFiles && m_outputStream.is_open())
    {
        m_outputStream.close();
        m_realStream = nullptr;
    }

    if (m_realStream == nullptr)
    {
        if (!m_outputFileName.empty() || m_splitFiles)
        {
            std::string outputFileName = m_splitFiles ? sanitizeFileName(tensorName) : m_outputFileName;

            m_outputStream.open(outputFileName);
            if (!m_outputStream.is_open())
            {
                JT_LOG_ERR(fmt::format("Could not open output file {}, dumping to standard output", m_outputFileName));
            }
        }
        m_realStream = m_outputStream.is_open() ? &m_outputStream : &std::cout;
    }

    return *m_realStream;

}

}  // namespace json_tests
