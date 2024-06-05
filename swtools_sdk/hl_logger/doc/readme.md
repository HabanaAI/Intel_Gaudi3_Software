# hl_logger (habana labs logger library)
fast, reusable logger library that solves all the issues of current approaches (specs_external - very slow, synapse - not reusable).

## hl_logger provides 2 types of api:
1. enum-based api: fast, feature-rich
2. specs_external compatible api: slow, simpler migration to hl_logger

## performance of enum-based api:
1. disabled logging: ~1000x faster
2. buffered logs with level less than the flush level (warning): ~10x faster

# How to use

## connect your project to hl_logger
### 1. build: in CMakeLists.txt of your project:
```cmake
target_include_directories(${YOUR_TARGET} PRIVATE $ENV{HL_LOGGER_INCLUDE_DIRS})
target_link_libraries(${YOUR_TARGET} $ENV{BUILD_ROOT_LATEST}/libhl_logger.so)
```
### 2. CI: in <YOUR_PROJECT_REPO>/.ci/pipelines/build.jenkinsfile add SWTOOLS_SDK as a build dependency:
```python
def buildDependencies = ['SWTOOLS_SDK', ...OTHER_DEPENDENCIES... ]
def requiredArtifact = [
    'SWTOOLS_SDK': ['shared'],# or ['bin', 'shared'], if you have swtools_sdk also as a test dependency
    ...OTHER_ARTIFACTS...
]
```

## Usage of enum-based api:
### 1. make sure you are using c++17 standard in CMakeLists.txt of your project:
```cmake
set(CMAKE_CXX_STANDARD 17)
```

### 2. create a header file that defines your logger names in enum. E.g. my_logger.hpp
```cpp
#include <hl_logger/hllog.hpp>
namespace YourNamespace {
    // define an enum with all the logger types
    // the last item must be LOG_MAX
    enum class LoggerTypes {
        SCAL, HCL_1, HCL_2, A_LOGGER_ON_DEMAND, LOG_MAX
    };
}
// define HLLOG_ENUM_TYPE_NAME that provides full name of your enum with logger
#define HLLOG_ENUM_TYPE_NAME YourNamespace::LoggerTypes
HLLOG_DECLARE_MODULE_LOGGER()
```

### 3. create a source file that implements your loggers. E.g. my_logger.cpp
```cpp
#include "my_logger.hpp"
namespace YourNamespace{
// create loggers (all the log files are created immediately when the module is loaded)
static void createModuleLoggers(LoggerTypes){
    // one logger in one file with non-default parameters
    hl_logger::LoggerCreateParams params;
    params.logFileName         = "scal_log.txt";
    params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;
    // see LoggerCreateParams for other options
    // e.g. params.printProcessID = true; // enable PID printing if needed
    hl_logger::createLogger(LoggerTypes::SCAL, params);
    // several loggers in one file with default parameters
    hl_logger::createLoggers({LoggerTypes::HCL_1, LoggerTypes::HCL_2}, {"log_hcl.txt"});
}

// all the following functions are optional and any/all of them can be omitted

// on-demand loggers
// log files created when the first message is logged into such logger
// this is a recommended way of loggers creation
static void createModuleLoggersOnDemand(LoggerTypes){
    hl_logger::createLoggerOnDemand(LoggerTypes::A_LOGGER_ON_DEMAND, {"some_on_demand_log.txt"});
}
// a callback when a dtor of your module is called (e.g. close an app, dlclose, etc)
// usually is used to log a final message
static void onModuleLoggersBeforeDestroy(LoggerTypes){
    HLLOG_INFO(SCAL, "closing synapse logger. no more log messages will be logged");
}

// a callback when an app got a signal (usually it means a crash)
// can be used to log a stacktrace or any other info
static void onModuleLoggersCrashSignal(LoggerTypes, int signal, const char* signalStr, bool isSevere){
    HLLOG_ERR(SCAL, "Crash. signal : {} {}. Severity: {}", signal, signalStr, isSevere ? "high" : "low");
    hl_logger::logStacktrace(LoggerTypes::SCAL, isSevere ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_INFO);
}
}  // namespace YourNamespace

// define logger internal variables. requires a list of all the logger names (for string representation)
HLLOG_DEFINE_MODULE_LOGGER(SCAL, HCL_1, HCL_2, A_LOGGER_ON_DEMAND, LOG_MAX)
```

### 4. use your header my_logger.hpp for logging in your project
```cpp
#include "my_logger.hpp"
enum class MyEnum{a, b};
void f(){
    MyEnum v = MyEnum::b;
    HLLOG_ERR(SCAL, "{}:{} enum: {}", 1, 2, v); //output: 1:2 enum: 1[b]
    HLLOG_ERR(SCAL, "{}:{}", 1);    // does not compile - parameters mismatch
}
```

## Usage of specs_external api:
```cpp
#include <hl_logger/hllog_se.hpp>
void f(){
    CREATE_LOGGER("LOGGER_SE", "logger_se.log", 1000 * 1024 * 5, 1);
    SET_LOG_LEVEL("LOGGER_SE", 0);
    LOG_ERR("LOGGER_SE", "hello world from old {}", "api");
    DROP_ALL_LOGGERS;
}
```
# Lazy logging
In production, regular logging can be too expensive to be enabled.
Therefore, we have 2 types of logging (they co-exist, but they don't know about each other)
1. regular - writes into a file. Typical log line time 0.3-2us. It's a default mode.
2. lazy - saves data into a cyclic buffer (no formatting, no writing to file).\
   Typical log line time 0.03-0.15us. Only last 2048 log messages per logger type are kept (can be configured with LAZY_LOG_QUEUE_SIZE_...).\
   Writing into a file is triggered in case of a crash.\
   The idea behind lazy logging is to have logs for crash investigation with minimal performance impact in production.

Any logger can be configured to log messages in a regular way or/and in a lazy way using defaultLoggingLevel and defaulLazytLoggingLevel respectively.

Reasons to consider for enabling lazy mode for a specific logger:
1. the logger is critical for crash investigation. e.g. API calls with their parameters.
2. the amount of messages per second is reasonable (e.g. hundreds). if the message rate is too high then only the last 2048 messages will be logged in case of crash.
3. Low frequency messages don't go into the same logger as high frequency ones because the latter can hide low frequency messages. Example: API logger which is used to log an api function name and then 1000 messages per function. In this case such logger should be split into 2 - e.g. API and API_DETAILS.

By default, lazy logging is disabled because all the arguments that are logged must be copyable.
in order to enable lazy logging in your project:
1. use ```#define HLLOG_ENABLE_LAZY_LOGGING```:
2. set lazy logging level for loggers that should support lazy logging ```params.defaultLazyLoggingLevel = HLLOG_LEVEL_...;```

example:
```cpp
// your logger header
// define HLLOG_ENABLE_LAZY_LOGGING BEFORE inclusion of <hl_logger/hllog.hpp>
#define HLLOG_ENABLE_LAZY_LOGGING
#include <hl_logger/hllog.hpp>
namespace YourNamespace {
    enum class LoggerTypes {
        SCAL, LOG_MAX
    };
}

#define HLLOG_ENUM_TYPE_NAME YourNamespace::LoggerTypes
HLLOG_DECLARE_MODULE_LOGGER()

// your logger cpp
#include "my_logger.hpp"
namespace YourNamespace{

static void createModuleLoggers(LoggerTypes){
hl_logger::LoggerCreateParams params;
params.logFileName         = "scal_log.txt";
params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;

// ENABLE LAZY LOGGING FOR A LOGGER
// log lines with INFO level will go to cyclic buffer and will be printed in case of a crash
params.defaulLazytLoggingLevel = HLLOG_LEVEL_INFO;

hl_logger::createLogger(LoggerTypes::SCAL, params);

}
```

# Run-time configuration with environment variables
You can set the following environment variables to adjust the behavior of hl_logger. All the variables are optional.
### Logs folder configuration. Logs folder format: ```<HABANA_LOGS>/[<HLS_ID>/][<ID>/]```
- HABANA_LOGS - set the habana_logs folder. default: "$HOME/.habana_logs" is used
- HLS_ID      - set HLS id (HLS_ID) sub-folder. usually in multi-box configuration. default: no sub-folder
- ID or OMPI_COMM_WORLD_RANK (in this order) - sets device id (ID) sub-folder. default: no sub-folder
### Logging configuration
- ENABLE_CONSOLE (1/true)             - enable printing all the logs to console. default: 0
- ENABLE_LOG_FILE_COLORS (1/true)     - enable colors in log files. default: 0
- LOG_LEVEL_ALL (0-5)                 - set log level for all messages. 0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL. default: defined by createLogger
- LOG_LEVEL_<LOGGER_NAME> (0-5)       - set log level for messages of <LOGGER_NAME> logger. default: defined by createLogger
- LOG_LEVEL_ALL_<LOGGER_PREFIX> (0-5) - set log level for messages of loggers that start with <LOGGER_PREFIX>. default: defined by createLogger
- LOG_FILE_SIZE (size in bytes)       - set max file size of log files. default: defined by createLogger
- ENABLE_SEP_LOG_PER_THREAD (1/true)  - separate log file per thread. default: 0
- DISABLE_SEPARATE_LOG_FILES (1/true) - disable separate log files. they are not rotating and can become too large
- LOG_FLUSH_LEVEL (0-5)               - flush level for all loggers. default: defined by createLogger
- **LAZY_LOG_LEVEL...** (0-5) - set log level for lazy logs. It's ignored if it's higher than the one defined by createLogger. To set any value - env var "ENABLE_EXPERIMENTAL_FLAGS" must be set to 1
- LAZY_LOG_LEVEL_ALL (0-5)             - set log level for all messages for lazy logs. 0-TRACE, 1-DEBUG 2-INFO, 3-WARN, 4-ERR, 5-CRITICAL. default: defined by createLogger
- LAZY_LOG_LEVEL_<LOGGER_NAME> (0-5)   - set log level for messages of <LOGGER_NAME> logger for lazy logs. default: defined by createLogger
- LAZY_LOG_LEVEL_ALL_<LOGGER_PREFIX> (0-5) - set log level for messages of loggers that start with <LOGGER_PREFIX> for lazy logs. default: defined by createLogger
- LAZY_LOG_QUEUE_SIZE_ALL              - set lazy log queue size for all lazy loggers. default 2048
- LAZY_LOG_QUEUE_SIZE_<LOGGER_NAME>    - set lazy log queue size for a specific logger
- LAZY_LOG_QUEUE_SIZE_ALL_<LOGGER_PREFIX> - set lazy log queue size for a loggers that start with <LOGGER_PREFIX>
### Log message format configuration
the following envvars overwrite predefined values. possible values: 1/true/false/0.
- PRINT_FILE_AND_LINE   - print file name and line number of all the log messages. default: 0
- PRINT_DATE            - print date of log messages. default: 0
- PRINT_TIME            - print time of log messages. default: 1
- PRINT_RANK            - print device rank.
  * PRINT_RANK not set  - print rank into console if ENABLE_CONSOLE=1 but not to files
  * PRINT_RANK=1        - print rank into console and into files
  * PRINT_RANK=0        - rank is not printed
- PRINT_TID             - print tid. default: 1
- PRINT_PID             - print pid. default: 1
- PRINT_SPECIAL_CONTEXT - print special context. default: 1

# Troubleshooting
### build error: unresolved external symbol ModuleLoggerData
probably hl_logger binary is outdated on your machine. update swtools_sdk and rebuild hl_logger:
```bash
cd ~/trees/npu-stack/swtools_sdk/
git pull
build_hl_logger -r
```

### build error: HL_LOGGER_INCLUDE_DIRS environment variable is not defined
environment variables are not loaded correctly. update automation and reload bashrc
```bash
cd ~/trees/npu-stack/automation/
git pull
source ~/.bashrc
```

### build fails - libhl_logger.so not found
update swtools_sdk and rebuild hl_logger:
```bash
cd ~/trees/npu-stack/swtools_sdk/
git pull
build_hl_logger -r
```

## [design doc](https://habanalabs-my.sharepoint.com/:w:/g/personal/avaisman_habana_ai/EVCJRyI-tDRMgVtQk6ASGW8B8Ja7ijqFGjqUTHSao0M_Cw?e=T7qIVl)
