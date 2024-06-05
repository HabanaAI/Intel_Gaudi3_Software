#include "device_mapper.hpp"
#include "runtime/common/device/device_mem_alloc.hpp"
#include "log_manager.h"

DeviceMapper::DeviceMapper(DevMemoryAllocInterface& rDevMemAlloc) : m_rDevMemAlloc(rDevMemAlloc) {}

bool DeviceMapper::mapBufferToDevice(uint8_t*           buffer,
                                     uint64_t           size,
                                     const std::string& mappingDesc,
                                     void**             hostVA) const
{
    if (m_rDevMemAlloc.mapBufferToDevice(size, buffer, false, 0, mappingDesc) != synSuccess)
    {
        LOG_CRITICAL(SYN_RECIPE, "{}: Failed to map buffer 0x{:x}", HLLOG_FUNC, TO64(buffer));
        return false;
    }

    if (HATVA_MAPPING_STATUS_FOUND != m_rDevMemAlloc.getDeviceVirtualAddress(false, buffer, size, (uint64_t*)hostVA))
    {
        LOG_CRITICAL(SYN_RECIPE, "{}: Failed to get buffer 0x{:x}", HLLOG_FUNC, TO64(buffer));
        return false;
    }

    return true;
}

bool DeviceMapper::unmapBufferFromDevice(void* hostVA) const
{
    const synStatus status = m_rDevMemAlloc.unmapBufferFromDevice(hostVA, false, nullptr);

    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_RECIPE, "{}: Failed to unmap buffer {:p}", HLLOG_FUNC, hostVA);
        return false;
    }

    return true;
}
