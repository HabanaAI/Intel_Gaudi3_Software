#pragma once

#include "runtime/qman/common/device_mapper_interface.hpp"

class DeviceMapperMock : public DeviceMapperInterface
{
public:
    virtual ~DeviceMapperMock() = default;

    virtual bool
    mapBufferToDevice(uint8_t* buffer, uint64_t size, const std::string& mappingDesc, void** hostVA) const override
    {
        return true;
    }

    virtual bool unmapBufferFromDevice(void* hostVA) const override { return true; }
};
