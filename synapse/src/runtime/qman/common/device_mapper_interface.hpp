#pragma once

#include "synapse_common_types.h"

class DeviceMapperInterface
{
public:
    virtual ~DeviceMapperInterface() = default;

    virtual bool
    mapBufferToDevice(uint8_t* buffer, uint64_t size, const std::string& mappingDesc, void** hostVA) const = 0;

    virtual bool unmapBufferFromDevice(void* hostVA) const = 0;
};
