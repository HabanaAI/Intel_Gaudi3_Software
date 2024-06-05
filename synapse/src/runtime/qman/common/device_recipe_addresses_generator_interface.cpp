#include "device_recipe_addresses_generator_interface.hpp"
#include "defs.h"
#include "runtime/qman/common/qman_types.hpp"
#include "define_synapse_common.hpp"
#include "recipe.h"
#include "math_utils.h"

uint64_t DeviceRecipeAddressesGeneratorInterface::getProgramDataSectionAddress(uint64_t        workspaceAddress,
                                                                               const recipe_t& rRecipe)
{
    uint64_t size = rRecipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA];
    if (size == 0)
    {
        // No Program-Data section
        LOG_DEBUG(SYN_STREAM, "No Program-Data section");
        return INITIAL_WORKSPACE_ADDRESS;
    }

    uint64_t programDataSectionAddress = workspaceAddress + rRecipe.workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE];

    return round_to_multiple(programDataSectionAddress, MANDATORY_KERNEL_ALIGNMENT);
}

uint64_t DeviceRecipeAddressesGeneratorInterface::getProgramCodeSectionAddress(uint64_t        workspaceAddress,
                                                                               const recipe_t& rRecipe)
{
    uint64_t programCodeSectionAddress = 0;

    uint64_t* pCurrWorkspaceSize      = rRecipe.workspace_sizes;
    uint64_t  currentWorkspaceAddress = workspaceAddress;
    for (uint64_t i = 0; i < rRecipe.workspace_nr; i++, pCurrWorkspaceSize++)
    {
        uint64_t currentWorkspaceSize = *pCurrWorkspaceSize;

        if (i == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            currentWorkspaceSize += MANDATORY_KERNEL_ALIGNMENT;
        }

        if (i == MEMORY_ID_RESERVED_FOR_PROGRAM)
        {
            programCodeSectionAddress = currentWorkspaceAddress;
            LOG_TRACE(SYN_STREAM, "{}: programCodeSectionAddress = {:#x} ", HLLOG_FUNC, programCodeSectionAddress);
            break;
        }
        currentWorkspaceAddress += currentWorkspaceSize;
    }
    return programCodeSectionAddress;
}