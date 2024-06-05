#include <gtest/gtest.h>
#include "infra/containers/slot_map_alloc.hpp"
#include "infra/containers/slot_map_precache.hpp"
#include <thread>
#include <condition_variable>
#include <mutex>
#include "perf_test.hpp"
struct S
{
    std::atomic<int> a;
    char             b[60];
};

[[maybe_unused]] static OperationFullMeasurementResults testOneAtomic(TestParams params)
{
    static S s[20] {};
    return measure([](unsigned thread_id) { s[0].a += 76; }, "one value atomic", params);
}

OperationFullMeasurementResults testAccessOneItem(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map;
    const unsigned                   maxItemsCount = 2000;
    SMHandle                         handles[maxItemsCount];
    for (unsigned i = 0; i < maxItemsCount; ++i)
    {
        auto item  = map.insert(0);
        handles[i] = item.first;
    }
    return measure(
        [&](unsigned thread_id) {
            auto ptr = map[handles[0]];
            if (!ptr)
            {
                std::cerr << "y";
            }
        },
        "SlotMap one handle RefCount",
        params);
}

OperationFullMeasurementResults testAccessOneItemNoCheck(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map(1024, SlotMapChecks::noRefCounting);
    const unsigned                   maxItemsCount = 2000;
    SMHandle                         handles[maxItemsCount];
    for (unsigned i = 0; i < maxItemsCount; ++i)
    {
        auto item  = map.insert(0);
        handles[i] = item.first;
    }
    return measure(
        [&](unsigned thread_id) {
            auto ptr = map[handles[0]];
            if (!ptr)
            {
                std::cerr << "y";
            }
        },
        "SlotMap one handle NoRefCount",
        params);
}

OperationFullMeasurementResults testAccesssDifferentItemRefCount(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map;
    const unsigned                   maxItemsCount = 2000;
    SMHandle                         handles[maxItemsCount];
    for (unsigned i = 0; i < maxItemsCount; ++i)
    {
        auto item  = map.insert(0);
        handles[i] = item.first;
    }
    return measure(
        [&](unsigned thread_id) {
            auto ptr = map[handles[thread_id]];
            if (!ptr)
            {
                std::cerr << "y";
            }
        },
        "SlotMap diff handles RefCount",
        params);
}

OperationFullMeasurementResults testAccesssDifferentItemNoRefCount(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map(1024, SlotMapChecks::noRefCounting);
    const unsigned                   maxItemsCount = 2000;
    SMHandle                         handles[maxItemsCount];
    for (unsigned i = 0; i < maxItemsCount; ++i)
    {
        auto item  = map.insert(0);
        handles[i] = item.first;
    }
    return measure(
        [&](unsigned thread_id) {
            auto ptr = map[handles[thread_id]];
            if (!ptr)
            {
                std::cerr << "y";
            }
        },
        "SlotMap diff handles NoRefCount",
        params);
}

OperationFullMeasurementResults testInsertDeletePrecacheNoRefCount(TestParams params)
{
    ConcurrentSlotMapPrecache<unsigned> map(1024 * 10, SlotMapChecks::noRefCounting);
    struct HandleAligned
    {
        SMHandle handle {0};
        bool   firtsTime {true};
        char   buf[56];
    };

    HandleAligned    handles[1000] {};
    std::atomic<int> eraseFailed {0};
    std::atomic<int> insertFailed {0};
    auto             res = measure(
        [&](unsigned thread_id) {
            if (!handles[thread_id].firtsTime && !map.erase(handles[thread_id].handle))
            {
                ++eraseFailed;
            }
            handles[thread_id].firtsTime = false;
            auto newItem                 = map.insert(thread_id);
            if (!newItem.second)
            {
                ++insertFailed;
                std::cerr << "failed to insert\n";
            }
            handles[thread_id].handle = newItem.first;
        },
        "SlotMapPrecache NoRefCount insert/delete",
        params,
        [&]() {
            map.eraseAll();
            // map.verify();
            for (auto& h : handles)
            {
                h.handle    = 0;
                h.firtsTime = true;
            }
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

OperationFullMeasurementResults testInsertDeletePrecacheRefCount(TestParams params)
{
    ConcurrentSlotMapPrecache<unsigned> map(1024 * 10, SlotMapChecks::full);
    struct HandleAligned
    {
        SMHandle handle {0};
        bool   firtsTime {true};
        char   buf[56];
    };

    HandleAligned    handles[1000] {};
    std::atomic<int> eraseFailed {0};
    std::atomic<int> insertFailed {0};
    auto             res = measure(
        [&](unsigned thread_id) {
            if (!handles[thread_id].firtsTime && !map.erase(handles[thread_id].handle))
            {
                ++eraseFailed;
            }
            handles[thread_id].firtsTime = false;
            auto newItem                 = map.insert(thread_id);
            if (!newItem.second)
            {
                ++insertFailed;
                std::cerr << "failed to insert\n";
            }
            handles[thread_id].handle = newItem.first;
        },
        "SlotMapPrecache RefCount insert/delete",
        params,
        [&]() {
            map.eraseAll();
            // map.verify();
            for (auto& h : handles)
            {
                h.handle    = 0;
                h.firtsTime = true;
            }
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

OperationFullMeasurementResults testInsertDeleteAllocNoRefCounting(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map(1024 * 10, SlotMapChecks::noRefCounting);
    struct HandleAligned
    {
        SMHandle handle {0};
        bool   firtsTime {true};
        char   buf[56];
    };

    HandleAligned    handles[1000] {};
    std::atomic<int> eraseFailed {0};
    std::atomic<int> insertFailed {0};
    auto             res = measure(
        [&](unsigned thread_id) {
            if (!handles[thread_id].firtsTime && !map.erase(handles[thread_id].handle))
            {
                ++eraseFailed;
            }
            handles[thread_id].firtsTime = false;
            auto newItem                 = map.insert(thread_id);
            if (!newItem.second)
            {
                ++insertFailed;
                std::cerr << "failed to insert\n";
            }
            handles[thread_id].handle = newItem.first;
        },
        "SlotMapAlloc NoRefCount insert/delete",
        params,
        [&]() {
            map.eraseAll();
            for (auto& h : handles)
            {
                h.handle    = 0;
                h.firtsTime = true;
            }
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

OperationFullMeasurementResults testInsertDeleteAllocRefCount(TestParams params)
{
    ConcurrentSlotMapAlloc<unsigned> map(1024 * 10, SlotMapChecks::full);
    struct HandleAligned
    {
        SMHandle handle {0};
        bool   firtsTime {true};
        char   buf[56];
    };

    HandleAligned    handles[1000] {};
    std::atomic<int> eraseFailed {0};
    std::atomic<int> insertFailed {0};
    auto             res = measure(
        [&](unsigned thread_id) {
            if (!handles[thread_id].firtsTime && !map.erase(handles[thread_id].handle))
            {
                ++eraseFailed;
            }
            handles[thread_id].firtsTime = false;
            auto newItem                 = map.insert(thread_id);
            if (!newItem.second)
            {
                ++insertFailed;
                std::cerr << "failed to insert\n";
            }
            handles[thread_id].handle = newItem.first;
        },
        "SlotMapAlloc RefCount insert/delete",
        params,
        [&]() {
            map.eraseAll();
            for (auto& h : handles)
            {
                h.handle    = 0;
                h.firtsTime = true;
            }
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

/// old impl
#include <unordered_map>
#include <deque>
std::recursive_mutex m_mutex;
// so we will not need to deal with (OS) memory management of those items during execution
static const uint16_t s_eventsInPool = 2000;
using QmanEvent                      = int;
std::array<QmanEvent, s_eventsInPool>              m_events;           // pool of all events
std::deque<QmanEvent*>                             m_freeEvents;       // points to free events in the pool
std::unordered_map<uint64_t, QmanEvent*>           m_usedEvents;       // used DB (eventHandle->used event in poo)
uint32_t                                           m_nextEventHandle;  // Event handle for new allocated evetn

void initEventsPool()
{
    m_nextEventHandle = 0;
    m_usedEvents.clear();
    m_freeEvents.clear();
    m_events.fill(QmanEvent());
    for (auto& event : m_events)
    {
        m_freeEvents.push_back(&event);
    }
}

bool getNewEvent(QmanEvent*& pEventHandle, const unsigned int flags)
{
    {
        std::lock_guard<std::recursive_mutex> guard(m_mutex);
        if (m_freeEvents.empty())
        {
            return false;
        }

        pEventHandle = m_freeEvents.front();
        m_freeEvents.pop_front();
        m_usedEvents[m_nextEventHandle] = pEventHandle;
        *pEventHandle                   = m_nextEventHandle;

        m_nextEventHandle++;
    }

    return true;
}

bool destroyEvent(QmanEvent* eventHandle)
{
    {
        std::lock_guard<std::recursive_mutex> guard(m_mutex);

        auto eventHandleItr = m_usedEvents.find(*eventHandle);
        if (eventHandleItr == m_usedEvents.end())
        {
            // LOG_ERR_T(SYN_STREAM, "{}: can't destroy, event not found in used {}", __FUNCTION__, eventDesc);
            return false;
        }

        m_freeEvents.push_back(eventHandleItr->second);
        m_usedEvents.erase(eventHandleItr);
    }
    return true;
}

OperationFullMeasurementResults testInsertDeleteOldWay(TestParams params)
{
    ConcurrentSlotMapPrecache<unsigned> map(1024 * 10, SlotMapChecks::noRefCounting);
    struct HandleAligned
    {
        QmanEvent*           handle {0};
        bool                 firtsTime {true};
        char                 buf[56];
    };

    HandleAligned    handles[1000] {};
    std::atomic<int> eraseFailed {0};
    std::atomic<int> insertFailed {0};
    auto             res = measure(
        [&](unsigned thread_id) {
            if (!handles[thread_id].firtsTime && !destroyEvent(handles[thread_id].handle))
            {
                ++eraseFailed;
            }
            handles[thread_id].firtsTime = false;
            auto newItem                 = map.insert(thread_id);
            if (!getNewEvent(handles[thread_id].handle, 0))
            {
                ++insertFailed;
                std::cerr << "failed to insert\n";
            }
        },
        "Current cache insert/delete",
        params,
        []() { initEventsPool(); });

    return res;
}

OperationFullMeasurementResults testInsertNewDelete(TestParams params)
{
    ConcurrentSlotMapPrecache<unsigned> map(1024 * 10, SlotMapChecks::noRefCounting);
    struct HandleAligned
    {
        QmanEvent*           handle {0};
        char                 buf[56];
    };

    HandleAligned    handles[1000] {};
    int              i   = 0;
    auto             res = measure(
        [&](unsigned thread_id) {
            auto ptr = handles[thread_id].handle;
            if (ptr)
            {
                i = *ptr;
            }
            delete ptr;
            handles[thread_id].handle = new QmanEvent {thread_id + i};
        },
        "simple New/delete",
        params,
        [&]() {
            for (auto& h : handles)
            {
                delete h.handle;
                h.handle = nullptr;
            }
        });

    return res;
}
#if ENABLE_PERFORMANCE_TEST
TEST(SlotMapPerfTest, basic_ops)
{
    TestParams params;
    params.maxNbThreads            = 10;
    params.minNbThreads            = 1;
    params.nbTests                 = 100;
    params.internalLoopInterations = 2000;
    params.interations_per_sleep   = 0;
    initEventsPool();

    PrintTestResults("SlotMap Insert/delete performance check",
                     {testInsertDeleteAllocRefCount(params),
                      testInsertDeleteAllocNoRefCounting(params),
                      testInsertDeletePrecacheRefCount(params),
                      testInsertDeletePrecacheNoRefCount(params),
                      testInsertDeleteOldWay(params),
                      testInsertNewDelete(params)});

    PrintTestResults("SlotMap item access performance check",
                     {testOneAtomic(params),
                      testAccessOneItem(params),
                      testAccessOneItemNoCheck(params),
                      testAccesssDifferentItemRefCount(params),
                      testAccesssDifferentItemNoRefCount(params)});
}
#endif