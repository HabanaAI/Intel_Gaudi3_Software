#pragma once

#include "synapse_common_types.h"

#include "runtime/common/recipe/patching/host_address_patcher.hpp"

#include <vector>
#include <memory>

struct MemoryMappedAddr
{
    uint8_t* hostAddr;
    uint64_t devAddr;
};

typedef std::vector<MemoryMappedAddr> MemoryMappedAddrVec;

enum InMappedStatus
{
    IN,
    OUT,
    BUSY
};

/*********************************************************************************
 * This struct holds information about the different sections of the recipe as it
 * being processed
 *********************************************************************************/
struct MemorySectionsScal
{
    uint64_t m_glbHbmAddr;

    uint64_t m_arcHbmAddr;
    uint32_t m_arcHbmCoreAddr;

    uint64_t            m_nonPatchableDcSize;
    MemoryMappedAddrVec m_nonPatchableMappedAddr;
    InMappedStatus      m_inMappedNoPatch = OUT;

    uint64_t                                  m_patchableDcSize;
    MemoryMappedAddrVec                       m_patchableMappedAddr;
    patching::HostAddressPatchingInformation* m_hostAddrPatchInfo;
    InMappedStatus                            m_inMappedPatch = OUT;

    std::unique_ptr<uint8_t []>               m_ih2dBuffer;

    /*
    ***************************************************************************************************
    *   @brief anyBusyInMapped() - check if any of the m_sections got busy when it tried allocating memory
    *                              from the mapped memory cache (no memory is available for this section)
    *
    *   @param  None
    *   @return true/false
    *
    ***************************************************************************************************
    */
    bool anyBusyInMapped() const { return (m_inMappedNoPatch == BUSY) || (m_inMappedPatch == BUSY); }
};
