#pragma once

#include <cstdint>
#include <vector>

/*********************************************************************************
 * This class is temporary, used to allocate memory from a pre-given memory area.
 * It splits the area into numChunks chunks. When an allocation is requested, it goes
 * over the chunks and finds an unused one
 *********************************************************************************/
class DumbAllocator
{
public:
    DumbAllocator(uint8_t* startAddr, uint64_t size, uint32_t numChunks);

    uint8_t* allocate(uint64_t size);
    void     free(uint8_t* addr);

private:
    uint8_t*       m_startAddr;
    const uint64_t m_size;
    const uint32_t m_numChunks;
    const uint64_t m_chunkSize;

    std::vector<bool> m_used;

    static const uint64_t ALIGN = 128;
};
