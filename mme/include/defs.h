/*
 * HW env doesnt support hl_logger, this will bypass it.
 */

#ifndef SWTOOLS_DEP
#define LOG_TRACE(log_type, msg, ...)
#define LOG_DEBUG(log_type, msg, ...)
#define LOG_INFO(log_type, msg, ...)
#define LOG_WARN(log_type, msg, ...)
#define LOG_ERR(log_type, msg, ...)
#define LOG_CRITICAL(log_type, msg, ...)

#endif