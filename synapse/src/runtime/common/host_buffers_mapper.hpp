#pragma once

#include "synapse_common_types.h"

#include <limits>

// Defines an interface class that exposes mappning and unmapping of host-buffers
// By definition

class HostBuffersMapper
{
public:
    static const uint64_t INVALID_ADDRESS = std::numeric_limits<uint64_t>::max();

    HostBuffersMapper()          = default;
    virtual ~HostBuffersMapper() = default;

    // TODO - I fail to see why buffer should not be const, but that requires changes in other places as well

    virtual synStatus mapBuffer(uint64_t&          hostVirtualAddress,
                                void*              buffer,
                                uint64_t           size,
                                bool               isUserRequest,
                                const std::string& mappingDesc,
                                uint64_t           requestedVirtualAddress = INVALID_ADDRESS) const = 0;

    virtual synStatus unmapBuffer(void* buffer, bool isUserRequest) const = 0;
    virtual uint64_t  getMappedSize() const                               = 0;
};