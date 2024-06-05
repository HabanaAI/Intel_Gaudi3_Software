#include <logging.h>

// force to expose old spdlog symbols to prevent crashes at destruction
// 1. namespace cannot be anonymous
// 2. function cannot be static
// we should use the majority of the functionality to force the compiler to add the corresponding symbols
namespace synapse
{
void forceSELoggerCompatibilityFunction()
{
    // use a non existent envvar to force the compiler to include spdlog code
    volatile bool falseCheck = false;
    auto loggerEmpty = spdlog::get("");
    if (falseCheck && loggerEmpty)
    {
        CREATE_SINK_LOGGER("SYN_COMPAT_LOG", "test_logger_from_syn.txt", 100, 0);
        CREATE_ERR_LOGGER("SYN_COMPAT_LOG", "test_logger_from_syn.txt", 100, 0);
        auto logger = spdlog::details::registry::instance().get("SYN_COMPAT_LOG");
        spdlog::drop("SYN_COMPAT_LOG");
    }
}
}
