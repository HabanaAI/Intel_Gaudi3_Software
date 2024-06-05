#pragma once

// eager includes (relative to src/eager/lib/)
#include "eager_recipe_memory_allocator.h"
#include "recipe_gen/recipe_defs.h"
#include "utils/memory_utils.h"

// std includes
#include <cstddef>
#include <cstdint>

struct recipe_t;
class RecipeAllocator;

namespace eager_mode
{
class DescGeneratorBase;
class Node2DescContainer;
class ProgramDataBlobManager;
class RecipeHalBase;
struct DataAccumulator;

class EagerRecipeAllocator
{
public:
    EagerRecipeAllocator(const ProgramDataBlobManager& programDataBlobManager,
                         uint64_t                      dataBlobsSizeInBytes,
                         EagerRecipeMemoryAllocator&   recipeAllocator,
                         const RecipeHalBase&          recipeHal,
                         const Node2DescContainer&     descriptors,
                         TensorsNrType                 tensorsNr,
                         bool                          canUseCloneFastPath,
                         bool                          isProgramDataBlobsCopyRequired,
                         bool                          isDebugInfoEnabled);

    // Allocate recipe and all buffers other than the graph and tensor names and initialize the pointers.
    recipe_t*      allocateAndInit(size_t namesSizeOfPersistentTensors);
    const DataBuf& getStringBufAllocator() const { return m_stringBufAlloc; }
    bool           isUsingCloneFastPath() const { return m_isUsingCloneFastPath; }

private:
    void planAllAllocs(DataAccumulator& dataAccum) const;
    recipe_t*
    doAllPlacements(std::byte* const heapBase, std::byte* const mappedBase, const DataAccumulator& dataAccum) const;
    recipe_t*   clone(const DescGeneratorBase& descGenBase, size_t namesSizeOfPersistentTensors);
    static void adjustPointers(recipe_t& actualRecipe, const recipe_t& templateRecipe, std::byte* dataBuf);

private:
    size_t                      m_programDataBlobsNum;
    uint64_t                    m_dataBlobsSizeInBytes;
    EagerRecipeMemoryAllocator& m_recipeAllocator;
    const RecipeHalBase&        m_recipeHal;
    const Node2DescContainer&   m_descriptors;
    const TensorsNrType         m_tensorsNr;
    DataBuf                     m_stringBufAlloc;
    const bool                  m_isUsingCloneFastPath;  // Use special allocation for single-activation graphs
    const bool                  m_isProgramDataBlobsCopyRequired;
    const bool                  m_isDebugInfoEnabled;  // detailed profiler debug info
};

}  // namespace eager_mode
