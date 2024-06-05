#pragma once

#include <hl_gcfg/hlgcfg.hpp>
#include <hl_gcfg/hlgcfg_item.hpp>
#include "habana_global_conf.h"
#include <string>
#include <synapse_common_types.h>
#include "log_manager.h"

/**
 * for compatibility with old code
 */
class GlobalConfManager
{
public:
    static GlobalConfManager instance()
    {
        return GlobalConfManager();
    }
    /**
     * Load global configuration from file
     * If file doesn't exist, create one with default values
     */
    static void init(const std::string& fileName)
    {
        hl_gcfg::reset();
        if (hl_gcfg::getEnableExperimentalFlagsValue())
        {
            if (!fileName.empty())
            {
                auto fileExist = hl_gcfg::loadFromFile(fileName);
                if (!fileExist)
                {
                    HLGCFG_LOG_INFO("Not using Global Configurations file ({}). {}", fileName, fileExist);
                }
            }
        }

        // Create "used" file:
        if (GCFG_CREATE_USED_CONFIGS_FILE.value())
        {
            std::string usedFileName = hl_logger::getLogsFolderPath() + "/synapse.used";
            hl_gcfg::saveToFile(usedFileName);
        }
    }

    /**
     * Load configuration from file
     * Return true if file exist
     */
    static bool load(const std::string& fileName)
    {
        return !hl_gcfg::loadFromFile(fileName).has_error();
    }

    /**
     * Flush configuration values to file
     * If file exist, it overwrite the file
     */
    static void flush(const std::string& fileName)
    {
        hl_gcfg::saveToFile(fileName);
    }

    static bool setGlobalConf(const char* cfgName, const char* cfgValue)
    {
        auto ret = hl_gcfg::setGcfgItemValue(cfgName, cfgValue);
        if (ret.has_error())
        {
            if (ret.errorCode() == hl_gcfg::ErrorCode::valueWasAlreadySetFromEnv)
            {
                LOG_WARN(GC_CONF, "{}: {}", HLLOG_FUNC, ret.errorDesc());
                return true;
            }
            LOG_ERR(GC_CONF, "{}: {}", HLLOG_FUNC, ret.errorDesc());
        }
        return !ret.has_error();
    }

    static bool getGlobalConf(const char* cfgName, char* cfgValue, uint64_t size)
    {
        auto ret = hl_gcfg::getGcfgItemValue(cfgName);

        if (ret.has_error())
        {
            LOG_ERR(GC_CONF, "{}: {}", HLLOG_FUNC, ret.errorDesc());
        }
        else
        {
            strncpy(cfgValue, ret.value().c_str(), size);
        }

        return !ret.has_error();
    }

    static void setDeviceType(synDeviceType deviceType)
    {
        hl_gcfg::setDeviceType(deviceType);
    }

    static void printGlobalConf(bool dfaFlow)
    {
        synapse::LogManager::LogType logType = dfaFlow ? synapse::LogManager::LogType::SYN_DEV_FAIL
                                                       : synapse::LogManager::LogType::GC_CONF;
        hl_gcfg::logRegisteredGcfgItems(hl_logger::getLogger(logType), SPDLOG_LEVEL_INFO);
    }
};
