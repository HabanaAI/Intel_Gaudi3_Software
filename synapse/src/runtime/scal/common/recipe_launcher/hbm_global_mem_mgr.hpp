#pragma once

#include "synapse_common_types.h"
#include "nbuff_allocator.hpp"

/*********************************************************************************
 * This class is used to allocate memory from the HBM for running a recipe.
 * It is currently implemented in a very simple way, assuming no caching. It splits the given
 * memory into chunks (donwloadedRecipes) and each time assigns a new chunk. since
 * we restrict the number of recipes that can be downloaded to the HBM before executed,
 * this is safe and easy.
 * The only assumption is that the needed size for a recipe is smaller than the chunk size.
 * In the future, we can consider adding caching here to avoid the pdma
 *********************************************************************************/
class HbmGlblMemMgr : protected NBuffAllocator
{
public:
    HbmGlblMemMgr() {};
    virtual ~HbmGlblMemMgr() {};

    void init(uint64_t addr, uint64_t size);

    [[nodiscard]] uint64_t getAddr(uint64_t size, uint64_t& hbmGlbAddr);

    void            unuseIdOnError();
    static uint64_t getMaxRecipeSize(uint64_t size);

    using NBuffAllocator::setLongSo;
    using NBuffAllocator::getLastLongSo;

private:
    static uint64_t getChunkSize(uint64_t size);

    uint64_t m_addr;
    uint64_t m_size;
};
