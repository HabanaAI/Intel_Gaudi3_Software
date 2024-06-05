#include "mme_assert.h"
#include <string>
#define FMT_HEADER_ONLY
#include "spdlog/fmt/bundled/format.h"
#include "print_utils.h"
namespace MmeCommon
{
// this function is only for MACROs not use it without them
void hbAssert(bool throwException,
              const char* conditionString,
              const std::string& message,
              const char* file,
              const int line,
              const char* func)
{
    std::string logMessage = fmt::format("{}::{} function: {}, failed condition: ({}), message: {}",
                                         file,
                                         line,
                                         func,
                                         conditionString,
                                         message);
    atomicColoredPrint(COLOR_RED, "%s\n", logMessage.c_str());
    if (throwException)
    {
        throw MMEStackException(logMessage);
    }
}
}