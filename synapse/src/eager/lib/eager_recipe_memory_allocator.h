#pragma once

// synapse api (relative to include/)
#include "internal/recipe_allocator.h"

// std includes
#include <memory>

namespace eager_mode
{
class EagerRecipeMemoryAllocator : public RecipeAllocator
{
public:
    void addKernelOwnership(const std::shared_ptr<char>& ptr) { m_programPtr = ptr; }

private:
    // A shared pointer to kernel or Elf binary requiring ownership.
    // This is an optimization for recipes with a single program data blob.
    // program_data_blobs_buffer should be a continuous memory and hold all
    // the program data blobs. For the single program data blob use case
    // we can avoid allocating this area and the copy into the allcoated area.
    // But this requires us to make sure the lifetime of the referenced program
    // data blob is extended to amtch that of the kernel.
    // There are 3 cases:
    // case 1 : The kernel binary is owned by the tpc node through a shared pointer
    // (kernel data was copied into it), hence by keeping this shared pointer here
    // we increase the lifetime to match that of the recipe.
    // case 2: The kernel points to the Elf binary which was owned by the tpc node
    // through a shared pointer (we copied the Elf binary into this shared pointer).
    // So by keeping a shared pointer to the Elf we once more match the lifetime of the
    // kernel to that of the recipe.
    // case 3: The Elf binary and kernel are pointing to\into the static Elf binary
    // loaded by tpc_kernels in which case the lifetime is longer than the recipe's
    // as this pointer will remain valid until Singelton destruction
    // (which would also invalidate the recipe).
    // We only need to cache the shared pointer for one of the first two cases.
    // Eager currently seems to only enter the third case, but adding the functionality
    // in here for completness in case this assumption is violated\ does not hold for some
    // cases.
    std::shared_ptr<char> m_programPtr;
};

}  // namespace eager_mode