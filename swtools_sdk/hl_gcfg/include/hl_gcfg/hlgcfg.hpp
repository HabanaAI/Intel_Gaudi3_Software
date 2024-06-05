#pragma once

#include <string>
#include <ostream>
#include <functional>
#include <hl_logger/hllog_core.hpp>
#include "hlgcfg_defs.hpp"
#include "hlgcfg_item_interface.hpp"

namespace hl_gcfg{
HLGCFG_NAMESPACE{
/**
 * Manager of all global configuration
 * Each global conf should register with the singleton before initialization occur
 *
 * Configuration file has an ini file syntax:
 * <global conf name>=<global conf value>
 * # - if a line starts with '#' it's a comment
 */

/**
 * 1. reset all the values to their defaults
 * 2. read new values from env vars
 */
HLGCFG_API void reset();

/**
 * Load configuration from ini file
 */
HLGCFG_API VoidOutcome loadFromFile(const std::string& fileName);

/**
 * save configuration values into a file
 * If file exists, it overwrites the file
 */
HLGCFG_API VoidOutcome saveToFile(const std::string& fileName);

/**
 * regsiter a newly constructed GcfgItem. internal usage only.
 * usually called from gcfg item ctor
 * @param globalConf a new gcfg item
 * @return status
 */
HLGCFG_API VoidOutcome registerGcfgItem(GcfgItemInterface& gcfgItem);

/**
 * unregister previously registered gcfg item. internal usage only.
 * usually called from gcfg item dtor
 * @param globalConf a registered gcfg item
 * @return
 */
HLGCFG_API VoidOutcome unregisterGcfgItem(GcfgItemInterface const & gcfgItem);

/**
 * unregister previously registered gcfg item. internal usage only.
 * DEPRICATED
 */
HLGCFG_API VoidOutcome unregisterGcfgItem(const std::string& gcfgItemName);

/**
 * set gcfg item value
 * if a config value is private and experimental_flags is off then setGcfgItemValue fails with privateConfigAccess
 * in order to set it regardless of experimental_flags set enableExperimental 'true'
 * @param gcfgItemName   - name of gcfg item
 * @param gcfgItemValue  - new value of gcfg item
 * @param enableExperimental - force setting the value even if experimental_flags is false
 * @return status
 */
HLGCFG_API VoidOutcome setGcfgItemValue(std::string const & gcfgItemName, std::string const & gcfgItemValue, bool enableExperimental = false);

/**
 * get string value of gcfg item
 * @param gcfgItemName - name of gcfg item
 * @return status + string value of gcfg item
 */
HLGCFG_API Outcome<std::string> getGcfgItemValue(std::string const & gcfgItemName);

/**
 * set device type. it defines which defaults are used
 * @param deviceType device type
 */
HLGCFG_API void setDeviceType(uint32_t deviceType);

/**
 * get device type that was set by setDeviceType
 * @return current device type
 */
HLGCFG_API uint32_t getDeviceType();

/**
 * print full configuration into a logger (all the registered gcfg items)
 * @param logger   - logger to output the configuration
 * @param logLevel - log level
 */
HLGCFG_API void logRegisteredGcfgItems(hl_logger::LoggerSPtr logger, int logLevel);

/**
 * get current logger that hl_gcfg is using for diagnostic messages
 * @return current gcfg logger
 */
HLGCFG_API hl_logger::LoggerSPtr getLogger();

/**
 * get logging level of the current logger
 * @return current logging level
 */
HLGCFG_API int getLoggingLevel();

/**
 * set hl_gcfg logger for diagnostic messages
 * by default hl_gcfg has logs into gcfg_log.txt
 * it's using HL_GCFG logger
 * the logger cna be changed to a user-provided
 * @param logger user-provided logger. if it's nullptr hl_gcfg will use its default HL_GCFG logger
 */
HLGCFG_API void setLogger(hl_logger::LoggerSPtr logger);

using ProcessGcfgItemFunc = std::function<void (std::string const & gcfgItemName, GcfgItemInterface& gcfgItemInterface)>;
/**
 * execute a user-provided function for all gcfg items
 * @param processGcfgItemFunc  user-proved function for gcfg items processing
 */
HLGCFG_API void forEachRegisteredGcfgItem(ProcessGcfgItemFunc processGcfgItemFunc);

/**
 * enable_experimental_flags access
 * @return current values of enable_experimental_flags
 */
HLGCFG_API bool getEnableExperimentalFlagsValue();

/**
 * set enable_experimental_flags value
 * @param value new enable_experimental_flags value
 */
HLGCFG_API void setEnableExperimentalFlagsValue(bool value);

/**
 * get enable_experimental_flags primary name
 * @return enable_experimental_flags primary name
 */
HLGCFG_API std::string getEnableExperimentalFlagsPrimaryName();
}}