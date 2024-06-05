#include "argument_parser.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

void MMETestArgumentParser::printHelp()
{
    std::cout << "Required Arguments:" << std::endl;
    std::cout << " --help                             Print help message" << std::endl;
    std::cout << " --test_type TYPE                   Sets the test comparison type - allowed values - device_sim_sim, "
                 "device_sim_chip, device_sim_ref, device_chip_ref"
              << std::endl;
    std::cout << " --test_config JSON_PATH            Path to json config file" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional Arguments:" << std::endl;
    std::cout << " --out_dir PATH                     Output directory path " << std::endl;
    std::cout << " --dump_unit STRING                 unit to dump" << std::endl;
    std::cout << " --seed NUM (=time(0))              Random seed value, default value - time(0) " << std::endl;
    std::cout << " --repeats NUM (=1)                 Number of repeats on the test - default value = 1 " << std::endl;
    std::cout << " -d [ --dump_memory ] DUMP (=none)  Pick how to dump memory accesses, from all MMEs, a single one or "
                 "none - all|single|none"
              << std::endl;
    std::cout << " -i [ --mme_idx ] NUM (=0)          Pick mme for memory dump (only valid if dump_memory is single)"
              << std::endl;
    std::cout << " --lfsr_path PATH                   Path to lfsr directory " << std::endl;
    std::cout << " --device_idxs IDX1 IDX2... (=0)    whitespace separated list of device indexes " << std::endl;
    std::cout << " --mmeLimit NUM (=0)                limit the amount of MMEs the test can run on " << std::endl;
    std::cout << " --checkRoi BOOL (=0)               run mme linear ranges checker" << std::endl;
    std::cout << " --b2bTestLimit NUM (=1)            run several tests back-to-back" << std::endl;
    std::cout << " --scalFw (=false)                  Run tests in SCAL FW" << std::endl;
    std::cout << " --chipAlt                          Run on GaudiM, Gaudi2B etc.." << std::endl;
}

MMETestArgumentParser::ArgMap MMETestArgumentParser::parseCommandLineToArgMap(int argc, char** argv)
{
    ArgMap argMap;
    for (unsigned i = 1; i < argc; i++)
    {
        std::string flag = std::string(argv[i]);
        std::string value;
        unsigned charsToRemove = 1;
        if (flag[1] == '-')  // indicate --
        {
            charsToRemove++;
        }
        flag.erase(0, charsToRemove);
        if (flag != "help" && flag != "chipAlt")
        {
            i++;
            value = std::string(argv[i]);
        }
        if (flag == "chipAlt")
        {
            value = "true";
        }
        argMap[flag] = value;

        if (flag == "device_idxs")
        {
            m_deviceIdxs.clear();
            std::string list;
            while (value[0] != '-' && i < argc)
            {
                list += value + std::string(" ");
                i++;
                value = std::string(argv[i]);
            }
            argMap[flag] = list;
            i--;  // dont ruin the iterator.
        }
    }
    return argMap;
}

bool MMETestArgumentParser::parse(int argc, char** argv)
{
    bool status = true;
    try
    {
        ArgMap argMap = parseCommandLineToArgMap(argc, argv);
        if (argMap.count("help") > 0)
        {
            printHelp();
            status = false;
        }
        else
        {
            checkRequired(argMap);
            parseArgs(argMap);
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        printHelp();
        return false;
    }
    return status;
}

void MMETestArgumentParser::checkRequired(const ArgMap& argMap)
{
    if (argMap.count("test_type") == 0)
    {
        throw RequiredArgMissing("test_type");
    }
    if (argMap.count("test_config") == 0)
    {
        throw RequiredArgMissing("test_config");
    }
}

void MMETestArgumentParser::parseArgs(ArgMap& argMap)
{
    parseTestType(argMap["test_type"]);
    parsePath("test_config", argMap["test_config"], m_testConfigPath);
    if (argMap.count("out_dir") > 0)
    {
        parsePath("out_dir", argMap["out_dir"], m_outDir, true);
    }
    if (argMap.count("dump_unit") > 0)
    {
        m_dumpUnit = argMap["dump_unit"];
    }
    if (argMap.count("seed") > 0)
    {
        m_seed = std::stoi(argMap["seed"]);
        std::cout << "INFO: Configuration seed - " << m_seed << std::endl;
    }
    if (argMap.count("repeats") > 0)
    {
        m_repeats = std::stoi(argMap["repeats"]);
    }
    if (argMap.count("dump_memory") > 0)
    {
        parseDumpMemory(argMap["dump_memory"]);
    }
    if (argMap.count("d") > 0)
    {
        parseDumpMemory(argMap["d"]);
    }
    if (argMap.count("mme_idx") > 0)
    {
        m_mmeIdx = std::stoi(argMap["mme_idx"]);
    }
    if (argMap.count("i") > 0)
    {
        m_mmeIdx = std::stoi(argMap["i"]);
    }
    if (argMap.count("lfsr_path") > 0)
    {
        parsePath("lfsr_path", argMap["lfsr_path"], m_lfsrPath);
    }
    if (argMap.count("device_idxs") > 0)
    {
        parseNumList(argMap["device_idxs"], " ", m_deviceIdxs);
    }
    if (argMap.count("mmeLimit") > 0)
    {
        m_mmeLimit = std::stoi(argMap["mmeLimit"]);
    }
    if (argMap.count("checkRoi") > 0)
    {
        parseBool(argMap["checkRoi"], m_checkRoi);
    }
    if (argMap.count("scalFw") > 0)
    {
        parseBool(argMap["scalFw"], m_scalFw);
    }
    if (argMap.count("b2bTestLimit") > 0)
    {
        m_b2bTestLimit = std::stoi(argMap["b2bTestLimit"]);
    }
    if (argMap.count("j") > 0)
    {
        m_threadLimit = std::stoi(argMap["j"]);
    }
    if (argMap.count("chipAlt"))
    {
        m_chipAlternative = true;
    }
}

void MMETestArgumentParser::parseTestType(const std::string& type)
{
    if (type == "device_sim_null" || type == "device_sim_ref")
    {
        m_testType = EMMETestType::sim_null;
    }
    else if (type == "device_chip_null" || type == "device_chip_ref")
    {
        m_testType = EMMETestType::chip_null;
    }
    else if (type == "device_sim_chip" || type == "device_chip_sim")
    {
        m_testType = EMMETestType::sim_chip;
    }
    else if (type == "device_null_null" || type == "device_ref_ref")
    {
        m_testType = EMMETestType::null_null;
    }
    else
    {
        throw InvalidArgument("test_type", type.c_str());
    }
    std::cout << "INFO: Test type - " << type << std::endl;
}

void MMETestArgumentParser::parsePath(const std::string& argName,
                                      const std::string& path,
                                      std::string& memberVal,
                                      bool create)
{
    if (!create)
    {
        bool exists = fs::exists(path);
        if (!exists)
        {
            std::cout << "ERROR: " << argName << " file doesnt exists - " << path << std::endl;
            throw InvalidArgument(argName.c_str(), path.c_str());
        }
    }
    std::cout << "INFO: " << argName << " file - " << path << std::endl;
    memberVal = path;
    if (create)
    {
        fs::create_directory(path);
    }
}

void MMETestArgumentParser::parseDumpMemory(const std::string& dumpMode)
{
    if (dumpMode != "none")
    {
        if (dumpMode == "all" || dumpMode == "single")
        {
            std::cout << "Memory dump mode : " << dumpMode << std::endl;
            if (dumpMode == "all")
            {
                m_dumpMem = MmeCommon::e_mme_dump_all;
            }
            else
            {
                m_dumpMem = MmeCommon::e_mme_dump_single;
            }
        }
        else
        {
            throw InvalidArgument("dump_mem", dumpMode.c_str());
        }
    }
}
void MMETestArgumentParser::parseNumList(const std::string& arg,
                                         const std::string& delimiter,
                                         std::vector<unsigned>& val)
{
    std::size_t pos = 0;
    std::size_t lastPos = 0;
    while ((pos = arg.find(delimiter, pos)) != std::string::npos)
    {
        std::string idxStr = arg.substr(lastPos, pos - lastPos);
        pos += delimiter.size();
        lastPos = pos;
        MME_ASSERT(isdigit(idxStr[0]), "deviceIdx is not a number");
        unsigned iVal = std::stoi(idxStr);
        val.push_back(iVal);
    }
}
