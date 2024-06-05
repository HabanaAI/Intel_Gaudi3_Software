#pragma once

#include "ittnotify.h"

// inspired by https://github.com/intel/IntelSEAPI/blob/master/itt_notify.hpp
class VTuneProfiler
{
public:
    static const __itt_domain*  HABANA_DOMAIN;
    static __itt_string_handle* asHandle(__itt_string_handle* v) { return v; }

    static __itt_string_handle* asHandle(const char* name) { return __itt_string_handle_create(name); }
    template<typename T>
    constexpr static __itt_string_handle* asHandle(T name)
    {
        return nullptr;
    }

    static void begin(const char* name, __itt_string_handle* cache)
    {
        __itt_task_begin(HABANA_DOMAIN, __itt_null, __itt_null, cache);
    }

    static void begin(__itt_string_handle* v, __itt_string_handle* cache)
    {
        __itt_task_begin(HABANA_DOMAIN, __itt_null, __itt_null, cache);
    }

    template<typename T>
    static void begin(T* fnPointer, __itt_string_handle* cache)
    {
        void* ptr = *(void**) (&fnPointer);  // Data pointer
        __itt_task_begin_fn(HABANA_DOMAIN, __itt_null, __itt_null, ptr);
    }

    class Task
    {
    public:
        Task(__itt_string_handle* pName)
        {
            if (!HABANA_DOMAIN || !HABANA_DOMAIN->flags) return;
            __itt_task_begin(HABANA_DOMAIN, __itt_null, __itt_null, pName);
        }

        ~Task()
        {
            if (!HABANA_DOMAIN || !HABANA_DOMAIN->flags) return;
            __itt_task_end(HABANA_DOMAIN);
        }
    };
};
