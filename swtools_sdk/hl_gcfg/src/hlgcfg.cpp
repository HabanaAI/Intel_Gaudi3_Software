#include <fstream>
#include <string>

#include <hl_gcfg/hlgcfg_item.hpp>
#include <hl_gcfg/hlgcfg.hpp>
#include "logger.hpp"
#include <unordered_map>
#include <set>

namespace hl_gcfg{
HLGCFG_NAMESPACE{

static std::mutex s_mtx;
using GcfgItemVec = std::vector<GcfgItemInterface*>;
static std::unordered_map<std::string, GcfgItemVec> s_globalConfByName;
static std::unordered_map<std::string, std::string> s_globalConfNameByAliases;
// Map for future register global conf
// In case they're register after the file loading
static std::unordered_map<std::string, std::string> s_globalConfStringValue;
static uint32_t s_deviceType = InvalidDeviceType;

static hl_logger::LoggerSPtr s_logger;

static volatile bool s_destroyed = false;

const char COMMENT_CHAR = '#';

static GcfgItemBool GCFG_ENABLE_EXPERIMENTAL_FLAGS(
        "ENABLE_EXPERIMENTAL_FLAGS",
        {"EXP_FLAGS"},
        "Enables experimental flags setting thru file or environment variables.",
        //"Experimental flags are not guaranteed to be fully validated, "
        //"and hence should be used for experiments only and/or with caution",
        false,
        MakePublic);

const std::string & toString(GcfgSource source)
{
    static const std::string sourceStr[] = {"DEFAULT",
                                            "ENV",
                                            "GCFG_FILE",
                                            "OBSERVER",
                                            "RUNTIME"};
    static_assert(sizeof(sourceStr)/sizeof(sourceStr[0]) == (unsigned)GcfgSource::MAX_SIZE, "array size mismatch");
    return sourceStr[(unsigned)source];
}

bool getEnableExperimentalFlagsValue()
{
    return GCFG_ENABLE_EXPERIMENTAL_FLAGS.value();
}
void setEnableExperimentalFlagsValue(bool value)
{
    GCFG_ENABLE_EXPERIMENTAL_FLAGS.setValue(value);
}

std::string getEnableExperimentalFlagsPrimaryName()
{
    return GCFG_ENABLE_EXPERIMENTAL_FLAGS.primaryName();
}

void reset()
{
    s_deviceType = InvalidDeviceType;
    std::lock_guard lock(s_mtx);
    for (auto const & gConfPair : s_globalConfByName)
    {
        for(auto & gcfgItem : gConfPair.second)
        {
            gcfgItem->reset(); //Initialize with default value
        }
    }

    // search specially for ENABLE_EXPERIMENTAL_FLAGS
    const auto& enableExperimentFlagsConf = s_globalConfByName.find(GCFG_ENABLE_EXPERIMENTAL_FLAGS.primaryName());
    if (enableExperimentFlagsConf != s_globalConfByName.end())
    {
        for(auto & gcfgItem : enableExperimentFlagsConf->second)
        {
            gcfgItem->updateFromEnv(true);
        }
    }

    for (auto const & gConfPair : s_globalConfByName)
    {
        for(auto & gcfgItem : gConfPair.second)
        {
            gcfgItem->updateFromEnv(GCFG_ENABLE_EXPERIMENTAL_FLAGS.value()); //Run over with env variables value (if exist)
        }
    }
}

VoidOutcome loadFromFile(const std::string& fileName)
{
    std::ifstream file(fileName);

    if (!file.good())
    {
        HLGCFG_RETURN_WARN(cannotOpenFile, "cannot open file {} for reading", fileName);
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Clear all spaces
        std::string::iterator toRemoveIter = std::remove_if(line.begin(), line.end(), isspace);
        line.erase(toRemoveIter, line.end());

        if (line.empty()) continue;
        if (line[0] == COMMENT_CHAR) continue;

        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos)
        {
            HLGCFG_LOG_ERR("Invalid format for global configuration manager - {}", line);
            continue;
        }

        std::string globalConfName = line.substr(0, equalPos);

        std::lock_guard lock(s_mtx);
        auto globalConfIter = s_globalConfByName.find(globalConfName);
        if (globalConfIter == s_globalConfByName.end())
        {
            HLGCFG_LOG_WARN("Global configuration var with name {} is not registered yet", globalConfName);
            // Save global conf value for future registration
            // a library with its config values can be loaded after a file is loaded.
            // in this case the saved values s_globalConfStringValue will be used
            s_globalConfStringValue[globalConfName] = line.substr(equalPos + 1);
        }
        else
        {
            for(auto const & gcfgItem : globalConfIter->second)
            {
                gcfgItem->setFromString(line.substr(equalPos + 1), GcfgSource::file);
            }
        }
    }
    file.close();

    return {};
}

/* This function creates a descriptive 'used' file, including the currently used global configurations */
VoidOutcome saveToFile(const std::string& fileName)
{
    std::ofstream file(fileName.c_str());
    if (! file.good())
    {
        HLGCFG_RETURN_ERR(cannotOpenFile, "Cannot open file {} to write configuration", fileName);
    }
    else
    {
        file << "\n## This file includes a list of the used Global Configurations:\n" << std::endl;
        std::lock_guard lock(s_mtx);
        for (auto gConfPair : s_globalConfByName)
        {
            const auto & gcfgItem = gConfPair.second[0];
            file << "## " << gcfgItem->description() << "\n";
            file << "## "
                 << "Set source: " << gcfgItem->getSourceStr()
                 << ". Default Value: " << gcfgItem->getDefaultValuesStr() << "\n";
            if (! gcfgItem->getUsedEnvAlias().empty())
            {
                file << COMMENT_CHAR << "# The following was set using the \"" << gcfgItem->getUsedEnvAlias() << "\" env variable.\n";
            }
            file << gcfgItem->primaryName() << "=" << gcfgItem->getValueStr() << "\n\n";
        }
        file.close();
    }
    return {};
}

VoidOutcome registerGcfgItem(GcfgItemInterface & gConf)
{
    std::lock_guard lock(s_mtx);
    auto & gcfgItems = s_globalConfByName[gConf.primaryName()];
    auto existingItemIt = std::find(gcfgItems.begin(), gcfgItems.end(), &gConf);
    if (existingItemIt != gcfgItems.end())
    {
        HLGCFG_RETURN_WARN(configNameAlreadyRegistered, "Global conf {} was already registered. ignore it", gConf.primaryName());
    }
    gcfgItems.push_back(&gConf);

    for (const std::string& alias : gConf.aliases())
    {
        auto retEmplace = s_globalConfNameByAliases.emplace(alias, gConf.primaryName());
        if (!retEmplace.second && gcfgItems.size() == 1)
        {
            HLGCFG_LOG_WARN("Global conf {} has an alias {} that was registered by {}. ignore", gConf.primaryName(), alias, s_globalConfNameByAliases[alias]);
        }
    }
    auto gConfValueIter = s_globalConfStringValue.find(gConf.primaryName());
    if (gConfValueIter != s_globalConfStringValue.end())
    {
        HLGCFG_LOG_INFO("Global conf {} is registered after file loading", gConf.primaryName());
        gConf.setFromString(gConfValueIter->second, GcfgSource::file);
    }
    return {};
}

VoidOutcome unregisterGcfgItem(GcfgItemInterface const & gConf)
{
    if (s_destroyed)
    {
        HLGCFG_RETURN_ERR(configRegistryWasDestroyed, "config registry was already destroyed");
    }
    std::lock_guard lock(s_mtx);
    auto globalConfIter = s_globalConfByName.find(gConf.primaryName());
    if (globalConfIter == s_globalConfByName.end())
    {
        HLGCFG_RETURN_ERR(configNameNotFoundInRegistry, "confName {} not found in conf registry", gConf.primaryName());
    }
    auto & gcfgItems = globalConfIter->second;
    auto gcfgItemIt = std::find(gcfgItems.begin(), gcfgItems.end(), &gConf);
    if (gcfgItemIt == gcfgItems.end())
    {
        HLGCFG_RETURN_ERR(configNameNotFoundInRegistry, "confName {} found in conf registry but gcfgItem not", gConf.primaryName());
    }
    gcfgItems.erase(gcfgItemIt);

    std::set<std::string> existingAliases;
    for (auto const & gcfgItem : gcfgItems)
    {
        for (const std::string& alias : gcfgItem->aliases())
        {
            existingAliases.insert(alias);
        }
    }

    for (auto const & alias : gConf.aliases())
    {
        if (existingAliases.count(alias) == 0)
        {
            s_globalConfNameByAliases.erase(alias);
        }
    }

    if (gcfgItems.empty())
    {
        s_globalConfByName.erase(gConf.primaryName());
    }
    return {};
}
// added for compatibility with the old code
// TODO: remove once all the code will use new approach
VoidOutcome unregisterGcfgItem(const std::string& name)
{
    if (s_destroyed)
    {
        HLGCFG_RETURN_WARN(configRegistryWasDestroyed, "config registry was already destroyed");
    }
    std::lock_guard lock(s_mtx);
    auto globalConfIter = s_globalConfByName.find(name);
    if (globalConfIter == s_globalConfByName.end())
    {
        HLGCFG_RETURN_WARN(configNameNotFoundInRegistry, "confName {} not found in conf registry", name);
    }

    for (auto const & gcfgItem : globalConfIter->second)
    {
        for (const std::string& alias : gcfgItem->aliases())
        {
            s_globalConfNameByAliases.erase(alias);
        }
    }
    s_globalConfByName.erase(name);
    return {};
}

Outcome<std::string> getPrimaryNameFromAlias(const std::string& alias)
{
    // find if an alias of this conf exists
    auto globalConfAliasIter = s_globalConfNameByAliases.find(alias);
    if (globalConfAliasIter == s_globalConfNameByAliases.end())
    {
        HLGCFG_RETURN_ERR(aliasNotFound, "alias {} not found", alias);
    }
    return globalConfAliasIter->second;
}

VoidOutcome setGcfgItemValue(std::string const & gcfgItemName, std::string const & gcfgItemValue, bool enableExperimental)
{
    std::lock_guard lock(s_mtx);
    auto globalConfIter = s_globalConfByName.find(gcfgItemName);
    if (globalConfIter == s_globalConfByName.end())
    {
        auto primary = getPrimaryNameFromAlias(gcfgItemName);
        if (!primary.has_value())
        {
            HLGCFG_RETURN_ERR(configNameNotFoundInRegistry, "gcfgItemName {} not found in registry", gcfgItemName);
        }

        // get the conf object from the primary name
        globalConfIter = s_globalConfByName.find(primary.value());
        if (globalConfIter == s_globalConfByName.end())
        {
            HLGCFG_RETURN_ERR(primaryNameNotFoundForAlias, "alias {} exists but primary name {} isn't registered", gcfgItemName, primary.value());
        }
    }

    auto& gcfgItems = globalConfIter->second;
    unsigned errCount = 0;
    VoidOutcome ret;
    for (auto & gcfgItem : gcfgItems)
    {
        if (gcfgItem->isPublic() || GCFG_ENABLE_EXPERIMENTAL_FLAGS.value() || enableExperimental)
        {
            auto curRet = gcfgItem->setFromString(gcfgItemValue, GcfgSource::runtime);
            if ((curRet.has_error()))
            {
                ret = std::move(curRet);
            }
        }
        else
        {
            HLGCFG_LOG_WARN("Global configuration var with name {} is not supported;"
                            "you tried setting an internal; if this is for internal-use or experiment please "
                            "first set ENABLE_EXPERIMENTAL_FLAGS.", gcfgItemName);
            errCount++;
        }
    }
    if (errCount == gcfgItems.size())
    {
        HLGCFG_RETURN_ERR(privateConfigAccess, "{} is internal config. cannot be set without ENABLE_EXPERIMENTAL_FLAGS", gcfgItemName);
    }
    return ret;
}

Outcome<std::string> getGcfgItemValue(std::string const & gcfgItemName)
{
    std::lock_guard lock(s_mtx);
    auto globalConfIter = s_globalConfByName.find(gcfgItemName);
    if (globalConfIter == s_globalConfByName.end())
    {
        auto primary = getPrimaryNameFromAlias(gcfgItemName);
        if (!primary.has_value()) HLGCFG_RETURN_ERR(primaryKeyNotSet, "primary key for {} is not set", gcfgItemName);

        // get the conf object from the primary name
        globalConfIter = s_globalConfByName.find(primary.value());
        if (globalConfIter == s_globalConfByName.end())
        {
            HLGCFG_RETURN_ERR(primaryNameNotFoundForAlias, "alias {} exists but primary name {} isn't registered", gcfgItemName, primary.value());
        }
    }

    std::string str = globalConfIter->second[0]->getValueStr();
    return {str};
}

uint32_t getDeviceType()
{
    return s_deviceType;
}

void setDeviceType(uint32_t deviceType)
{
    s_deviceType = deviceType;
}

void logRegisteredGcfgItems(hl_logger::LoggerSPtr logger, int logLevel)
{
    if (hl_logger::getLoggingLevel(logger) > logLevel) return;

    HLLOG_UNTYPED(logger, logLevel, "Habana Global Configurations:");
    std::lock_guard lock(s_mtx);
    for (const auto& gConfPair : s_globalConfByName)
    {
        HLLOG_UNTYPED(logger, logLevel, "    {} = {}", gConfPair.second[0]->primaryName(), gConfPair.second[0]->getValueStr());
    }
}

hl_logger::LoggerSPtr getDefaultLogger()
{
    return hl_logger::getLogger(LoggerTypes::HL_GCFG);
}

hl_logger::LoggerSPtr getLogger()
{
    return s_logger ? s_logger : getDefaultLogger();
}

int getLoggingLevel()
{
    return s_logger ? hl_logger::getLoggingLevel(s_logger) : hl_logger::getLoggingLevel(LoggerTypes::HL_GCFG);
}

void setLogger(hl_logger::LoggerSPtr logger)
{
    s_logger = logger;
}

void forEachRegisteredGcfgItem(ProcessGcfgItemFunc processGcfgItemFunc)
{
    std::lock_guard lock(s_mtx);
    for (auto & [k, gcfgItems] : s_globalConfByName)
    {
        for (auto & gcfgItem : gcfgItems)
        {
            processGcfgItemFunc(k, *gcfgItem);
        }
    }
}

}}