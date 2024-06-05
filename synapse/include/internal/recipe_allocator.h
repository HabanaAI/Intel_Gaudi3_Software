#pragma once

#include <cstddef>
#include <cstdint>
#include "llvm/small_vector.h"

class RecipeAllocator
{
public:
    explicit RecipeAllocator(bool ignoreDeviceMapping = false) : m_ignoreDeviceMapping {ignoreDeviceMapping}
    {
    }
    virtual ~RecipeAllocator();

    char* allocate(uint64_t sizeInBytes, bool shouldBeMappedToDevice = false);
    void  freeAll() noexcept;
    bool  freeSingleEntry(char* entryBaseAddress);

    bool isEmpty() const { return m_data.empty(); }

private:
    struct RecipeAllocation
    {
        void*    pData;
        uint64_t size;
        bool     allocatedWithMmap;
    };

    using RecipeAllocationVector = llvm_vecsmall::SmallVector<RecipeAllocation, 2>;
    RecipeAllocationVector m_data;

    // When this class is used for recipe template generation,
    // support ignoring the mapped mem alloc requests.
    bool m_ignoreDeviceMapping {};
};

class RecipeBlockAllocator
{
public:
    // Work with `blockSz` blocks from `recipeAllocator`
    RecipeBlockAllocator(RecipeAllocator& recipeAllocator, size_t blockSz);
    virtual ~RecipeBlockAllocator() = default;

    // Allocate size bytes from the current pool (Or a new pool from recipeAllocator)
    std::byte* alloc(size_t size);

protected:
    RecipeAllocator& m_recipeAllocator;
    std::byte*       m_ptr = nullptr;
    size_t           m_size;
    size_t           m_used;  // number of bytes used from the current block; can be <= m_size
};
