#pragma once

#include "syn_logging.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

struct ApiCounterPair
{
};

class ApiCounter
{
public:
    ApiCounter() = default;

    ~ApiCounter() = default;

    void incEntry()
    {
        m_entry.fetch_add(1, std::memory_order_relaxed);
    }

    void incExit()
    {
        m_exit.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t entry() const
    {
        return m_entry;
    }

    uint64_t exit() const
    {
        return m_exit;
    }

    void reset()
    {
        m_entry = m_exit = 0;
    }

private:
    std::atomic<uint64_t> m_entry{0};
    std::atomic<uint64_t> m_exit{0};
};

class ApiCounterRegistry
{
public:
    static ApiCounterRegistry& getInstance()
    {
        static ApiCounterRegistry instance;
        return instance;
    }

    ApiCounter& create(std::string_view funcName)
    {
        std::lock_guard lock(m_mutex);
        auto emplaceResult{m_apiCountersDb.emplace(std::make_pair(funcName, std::make_unique<ApiCounter>()))};
        return *emplaceResult.first->second;
    }

    std::string toString()
    {
        std::string countersDump;

        std::lock_guard lock(m_mutex);
        for (const auto &[funcName, counters] : m_apiCountersDb)
        {
            uint64_t enterCounter = counters->entry();
            uint64_t exitCounter  = counters->exit();
            countersDump += fmt::format(FMT_COMPILE("Function: {:50}, Entry counter: {:10}, Exit counter: {:10}{}\n"),
                                                    funcName,
                                                    enterCounter,
                                                    exitCounter,
                                                    enterCounter != exitCounter ? " Counters Inconsistency" : "");
        }

        return countersDump;
    }

    void reset()
    {
        std::lock_guard lock(m_mutex);
        for (auto &[funcName, counters] : m_apiCountersDb)
        {
            counters->reset();
        }
    }

    ApiCounterRegistry(const ApiCounterRegistry&) = delete;
    ApiCounterRegistry(ApiCounterRegistry&&) = delete;
    ~ApiCounterRegistry() = default;

private:
    ApiCounterRegistry() = default;

    std::mutex m_mutex;

    using ApiCountersDB = std::map<std::string_view, std::unique_ptr<ApiCounter>>;
    ApiCountersDB m_apiCountersDb;
};


class CounterIncrementer
{
public:
    CounterIncrementer(ApiCounter& apiCounter) : m_apiCounter(apiCounter)
    {
        m_apiCounter.incEntry();
    }

    ~CounterIncrementer()
    {
        m_apiCounter.incExit();
    }

private:
    ApiCounter& m_apiCounter;
};
