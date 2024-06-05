#pragma once

#define _NOP_MACRO ((void) 0)

#ifdef VTUNE_ENABLED
#define IS_VTUNE_ENABLED 1
#include "vtune_profiling.h"

#define STAT_CONCAT(x, y)  x##y
#define STAT_CONCAT2(x, y) STAT_CONCAT(x, y)
#define STAT_GLBL_START(name)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        static __itt_string_handle* STAT_CONCAT2(__stat_name, __LINE__) = VTuneProfiler::asHandle(#name);              \
        VTuneProfiler::begin(#name, STAT_CONCAT2(__stat_name, __LINE__));                                              \
    } while (0)

#define STAT_GLBL_COLLECT_TIME(x, point) __itt_task_end(VTuneProfiler::HABANA_DOMAIN)
#define STAT_SET_THREAD_NAME(name) __itt_thread_set_name(name)

#define STAT_FUNCTION()                                                                                                \
    static __itt_string_handle* STAT_CONCAT2(__stat_scope_name, __LINE__) = __itt_string_handle_create(__FUNCTION__);  \
    VTuneProfiler::Task         STAT_CONCAT2(__stat_scope_item, __LINE__)(STAT_CONCAT2(__stat_scope_name, __LINE__));

#define STAT_PAUSE() __itt_pause()
#define STAT_RESUME() __itt_resume()

#define STAT_EXIT_NO_COLLECT() __itt_task_end(VTuneProfiler::HABANA_DOMAIN)

#else

#define STAT_PAUSE() _NOP_MACRO
#define STAT_RESUME() _NOP_MACRO

#define STAT_FUNCTION()            _NOP_MACRO
#define STAT_EXIT_NO_COLLECT()       _NOP_MACRO

#define IS_VTUNE_ENABLED 0

#define STAT_SET_THREAD_NAME(name) _NOP_MACRO
// #error "compile with vtune please"

#endif