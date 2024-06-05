#undef NDEBUG
#include <cassert>
#include <fstream>
#include <cstring>
#include <iomanip>
#include "mme_coralhbw_sniffer.h"


void CoralMmeHBWSniffer::MemSplit::split(
    std::vector<uint8_t> &left,
    std::vector<uint8_t> &right,
    uint64_t offset)
{
    right.assign(left.begin() + offset, left.end());
    left.resize(offset);
}

bool CoralMmeHBWSniffer::MemMerge::merge(std::vector<uint8_t> & left, const std::vector<uint8_t> & right, const uint64_t sizeRight)
{
    assert(sizeRight == right.size());
    left.insert(left.end(), right.begin(), right.end());
    return true;
}

void CoralMmeHBWSniffer::handleTransaction(
    const std::string& token,
    const uint64_t address,
    const uint64_t size,
    const uint8_t *data,
    const bool isWrite)
{
    if (m_enabled)
    {
        std::vector<std::pair<std::string, bool>> &modules = isWrite ? m_wrModules : m_rdModules;
        std::mutex &mutex = isWrite ? m_wrMutex : m_rdMutex;
        SnifferSegmentSpace &segSpace = isWrite ? m_wrSegSpace : m_rdSegSpace;

        bool found = false;
        for (const auto & module : modules)
        {
            if (module.second) // prefix
            {
                if (!strncmp(token.c_str(), module.first.c_str(), module.first.size()))
                {
                    found = true;
                    break;
                }
            }
            else
            {
                if (token ==  module.first)
                {
                    found = true;
                    break;
                }
            }
        }

        if (found)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!segSpace.isSegmentCovered(address, size))
            {
                std::vector<uint8_t> mem;
                mem.assign(data, data + size);
                segSpace.addSegment(address, size, mem);
            }
        }
    }
}

void  CoralMmeHBWSniffer::dumpMemToFile(
    const std::string &fileName,
    SnifferSegmentSpace &segSpace)
{
    std::ofstream outFile;
    outFile.open(fileName, std::ios::out);
    outFile << std::setfill('0');

    for (auto it = segSpace.cbegin(); it != segSpace.cend(); it++)
    {
        if (it->second.valid)
        {
            assert(it->second.size == it->second.seg.size());
            outFile << "0x" << std::hex << std::setw(16) << it->first << std::endl;
            outFile << std::dec << it->second.size << std::endl;
            unsigned byte = 0;
            for(const uint8_t val : it->second.seg)
            {
                if ((byte!=0) && (byte % 16)==0) outFile << std::endl;  //16B per line
                outFile << std::hex << std::setw(2) << (unsigned)val << ", ";
                byte++;
            }
            outFile << std::endl;
        }
    }
}



void CoralMmeHBWSniffer::read(
    const std::string &token,
    const uint64_t address,
    const unsigned size,
    const uint32_t usr,
    const uint8_t *data)
{
    (void)usr;
    handleTransaction(token, address, size, data, false);
}

void CoralMmeHBWSniffer::write(
    const std::string &token,
    const uint64_t address,
    const unsigned size,
    const uint32_t usr,
    const uint8_t *data)
{
    (void)usr;
    handleTransaction(token, address, size, data, true);
}

void CoralMmeHBWSniffer::addReadModule(const std::string &moduleName, const bool isPrefix)
{
    m_rdModules.push_back(std::pair<std::string, bool>(moduleName, isPrefix));
}

void CoralMmeHBWSniffer::addWriteModule(const std::string &moduleName, const bool isPrefix)
{
    m_wrModules.push_back(std::pair<std::string, bool>(moduleName, isPrefix));
}

void CoralMmeHBWSniffer::generateDump(const std::string path)
{
    m_rdSegSpace.mergeSegments();
    dumpMemToFile(path+"/input.txt", m_rdSegSpace);

    m_wrSegSpace.mergeSegments();
    dumpMemToFile(path+"/output.txt", m_wrSegSpace);
}

void CoralMmeHBWSniffer::clear()
{
    m_rdModules.clear();
    m_wrModules.clear();
    m_rdSegSpace.erase(0, -1);
    m_wrSegSpace.erase(0, -1);
}


