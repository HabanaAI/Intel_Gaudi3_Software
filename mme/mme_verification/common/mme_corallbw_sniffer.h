#pragma once

#include <cstdint>
#include <mutex>
#include <list>
#include "coral_cluster.h"

class CoralMmeLBWSniffer : public CoralLbwSniffer
{
public:

    CoralMmeLBWSniffer() : m_enabled(false) {}
    virtual ~CoralMmeLBWSniffer() {}

    virtual void read(
        const std::string &token,
        const uint64_t address,
        const uint32_t value) override;

    virtual void write(
        const std::string &token,
        const uint64_t address,
        const uint32_t value) override;

    void addSniffingRange(const uint64_t base, const uint64_t size, const std::string & module = "");
    void generateDumpFile(const std::string &fileName);
    void clear();
    void enable() {m_enabled = true; }
    void disable() {m_enabled = false; }
    bool isEnabled() { return m_enabled; }

private:

    struct SniffingRange
    {
        uint64_t start;
        uint64_t end;
        std::string module;
    };

    std::list<SniffingRange> m_ranges;
    std::list<std::pair<uint64_t, uint32_t>> m_configs;
    std::mutex m_mutex;
    bool m_enabled;
};


