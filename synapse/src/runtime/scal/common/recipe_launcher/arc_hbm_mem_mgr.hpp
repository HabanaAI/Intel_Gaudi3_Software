#pragma once

#include <cstdint>
#include <vector>

#include "nbuff_allocator.hpp"
/*********************************************************************************
 * This class is used to allocate memory of the Arc-HBM for running a recipe.
 * It is currently implemented in a very simple way, assuming no caching. It splits the given
 * memory into chunks (donwloadedRecipes) and each time assigns a new chunk. since
 * we restrict the number of recipes that can be downloaded to the HBM before executed,
 * this is safe and easy.
 * The only assumption is that the needed for a recipe size is smaller than the chunk size.
 * In the future, we can consider adding caching here to avoid the pdma
 *********************************************************************************/
class ArcHbmMemMgr : protected NBuffAllocator
{
public:
    ArcHbmMemMgr() : NBuffAllocator() {}
    virtual ~ArcHbmMemMgr() {}

    void init(uint32_t addrCore, uint64_t addrDev, uint32_t size);

    [[nodiscard]] uint64_t getAddr(uint64_t size, uint64_t& hbmArcAddr, uint32_t& hbmArcCoreAddr);

    void unuseIdOnError();

    using NBuffAllocator::setLongSo;
    using NBuffAllocator::getLastLongSo;

    static uint64_t getMaxRecipeSize(uint64_t size);

private:
    static uint64_t getChunkSize(uint64_t size);

    const static uint64_t ALIGN = 128;

    uint32_t m_addrCore;
    uint64_t m_addrDev;
    uint64_t m_size;
};
