#pragma once

namespace MemProtectUtils
{
    int memWrProtectPages(uint8_t* startAddr, uint64_t size);
    int memWrUnprotectPages(uint8_t* startAddr, uint64_t size);

    static constexpr uint64_t pageSize2M = 2 * 1024 * 1024;
    static constexpr uint64_t pageMask2M = pageSize2M - 1;

    static constexpr uint64_t pageSize4K = 4 * 1024;
    static constexpr uint64_t pageMask4K = pageSize4K - 1;
};