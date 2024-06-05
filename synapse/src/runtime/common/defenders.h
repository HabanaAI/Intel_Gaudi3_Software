#pragma once

#include "syn_logging.h"
#include "defs.h"

#define VERIFY_IS_NULL_POINTER(logger_name, pointer, name)                                                             \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(logger_name, "{}: got null pointer for {}", HLLOG_FUNC, name);                                         \
        return synInvalidArgument;                                                                                     \
    }

#define BREAK_IF_NULL_POINTER(pointer, name)                                                                           \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(SYN_API, "{}: got null pointer for {}", HLLOG_FUNC, name);                                             \
        break;                                                                                                         \
    }

#define BREAK_AND_SET_FAIL_STATUS_IF_NULL_POINTER(log_name, pointer, name, status)                                     \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(log_name, "{}: got null pointer for {}", HLLOG_FUNC, name);                                            \
        status = synFail;                                                                                              \
        break;                                                                                                         \
    }

#define CHECK_POINTER(log_name, pointer, name, retVal)                                                                 \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(log_name, "{}: got null pointer for {}", HLLOG_FUNC, name);                                            \
        HB_ASSERT(0, "got null pointer for name {}", name);                                                            \
        return retVal;                                                                                                 \
    }

#define CHECK_POINTER_RET(log_name, pointer, name, retVal)                                                             \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(log_name, "{}: got null pointer for {}", HLLOG_FUNC, name);                                            \
        return retVal;                                                                                                 \
    }

#define CONTINUE_AND_SET_STATUS_IF_NULL_POINTER(pointer, name, status, statusVal)                                      \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_ERR(SYN_API, "{}: got null pointer for {}", HLLOG_FUNC, name);                                             \
        status = statusVal;                                                                                            \
        continue;                                                                                                      \
    }
