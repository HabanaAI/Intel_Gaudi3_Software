#pragma once

#include "device_mapper_interface.hpp"

class DevMemoryAllocInterface;

class DeviceMapper : public DeviceMapperInterface
{
public:
    DeviceMapper(DevMemoryAllocInterface& rDevMemAlloc);

    virtual ~DeviceMapper() override = default;

    virtual bool
    mapBufferToDevice(uint8_t* buffer, uint64_t size, const std::string& mappingDesc, void** hostVA) const override;

    virtual bool unmapBufferFromDevice(void* hostVA) const override;

private:
    DevMemoryAllocInterface& m_rDevMemAlloc;
};
