#undef NDEBUG
#include <cassert>
#include <fstream>
#include <iomanip>
#include "mme_corallbw_sniffer.h"
#include "gaudi2/reg_printer.h"

void CoralMmeLBWSniffer::read(
    const std::string &token,
    const uint64_t address,
    const uint32_t value)
{
    (void)token;
    (void)address;
    (void)value;
}

void CoralMmeLBWSniffer::write(
    const std::string &token,
    const uint64_t address,
    const uint32_t value)
{

    const std::string * moduleName = &token;

    for (const auto & tuple : m_ranges)
    {
        if ((address >= tuple.start) &&
            (address < tuple.end) &&
            (tuple.module.empty() || (moduleName && (*moduleName == tuple.module))))
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_configs.push_back(std::pair<uint64_t, uint32_t>(address, value));
            break;
        }
    }
}

void CoralMmeLBWSniffer::addSniffingRange(
    const uint64_t base,
    const uint64_t size,
    const std::string & module)
{
    SniffingRange range;
    range.start = base;
    range.end = base + size;
    range.module = module;
    m_ranges.push_back(range);
}

void CoralMmeLBWSniffer::generateDumpFile(const std::string &fileName)
{
    std::ofstream outFile;
    outFile.open(fileName, std::ios::out);
    outFile << std::setfill('0');
    for(auto const &reg: m_configs)
    {
        outFile << "0x" << std::hex << std::setw(16) << reg.first << ", 0x" << std::setw(8) << reg.second <<
                    " (" << getRegName(reg.first) << ")" << std::endl;
    }
}

void CoralMmeLBWSniffer::clear()
{
    m_ranges.clear();
    m_configs.clear();
}
