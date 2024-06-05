#pragma once

#include <vector>
#include <cstdint>
#include <mutex>
#include "include/sync/segments_space.h"
#include "coral_cluster.h"

class CoralMmeHBWSniffer : public CoralHbwSniffer
{
public:
    CoralMmeHBWSniffer() : m_enabled(false) {}
    virtual ~CoralMmeHBWSniffer() {}

    virtual void read(const std::string& token,
                      const uint64_t     addr,
                      const unsigned int size,
                      const uint32_t     axuser,
                      const uint8_t*     src)  override;
    virtual void write(const std::string& token,
                       const uint64_t     addr,
                       const unsigned int size,
                       const uint32_t     axuser,
                       const uint8_t*     src) override;

    void addReadModule(const std::string &moduleName, const bool isPrefix);
    void addWriteModule(const std::string &moduleName, const bool isPrefix);
    void generateDump(const std::string path);
    void clear();
    void enable() {m_enabled = true; }
    void disable() {m_enabled = false; }
    bool isEnabled() { return m_enabled; }

private:
    struct MemSplit
    {
        static void split(std::vector<uint8_t> & left, std::vector<uint8_t> & right, uint64_t offset);
    };

    struct MemMerge
    {
        static bool merge(std::vector<uint8_t> & left, const std::vector<uint8_t> & right, const uint64_t sizeRight);
    };

    typedef SegmentsSpace<std::vector<uint8_t>, MemSplit, MemMerge> SnifferSegmentSpace;

    void handleTransaction(
        const std::string &token,
        const uint64_t address,
        const uint64_t size,
        const uint8_t *data,
        const bool isWrite);

    static void dumpMemToFile(
        const std::string &fileName,
        SnifferSegmentSpace &segSpace);

    std::vector<std::pair<std::string, bool>> m_rdModules;
    std::vector<std::pair<std::string, bool>> m_wrModules;
    SnifferSegmentSpace m_rdSegSpace;
    SnifferSegmentSpace m_wrSegSpace;
    std::mutex m_rdMutex;
    std::mutex m_wrMutex;
    bool m_enabled;
};


