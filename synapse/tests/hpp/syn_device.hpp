#pragma once

#include "syn_object.hpp"
#include "syn_stream.hpp"
#include "syn_host_buffer.hpp"
#include "syn_device_buffer.hpp"
#include "syn_profiler.hpp"

namespace syn
{
struct DeviceMemoryInfo
{
    uint64_t free;   // Free memory available
    uint64_t total;  // Total memory on device
};

class Device : public SynObject<uint32_t>
{
public:
    Device() = default;

    // TODO: remove when SW-64892 is fixed
    uint64_t getAttribute(synDeviceAttribute attribute)
    {
        uint64_t val = 0;
        SYN_CHECK(synDeviceTypeGetAttribute(&val, &attribute, 1, getInfo().deviceType));
        return val;
    }

    std::vector<uint64_t> getAttributes(const std::vector<synDeviceAttribute>& attributes) const
    {
        std::vector<uint64_t> ret(attributes.size());
        SYN_CHECK(synDeviceGetAttribute(ret.data(), attributes.data(), attributes.size(), handle()));
        return ret;
    }

    std::string getName() const
    {
        char deviceName[maxStringLength];
        SYN_CHECK(synDeviceGetName(deviceName, maxStringLength, handle()));
        return deviceName;
    }

    Stream createStream(const uint32_t flags = 0)
    {
        auto handlePtr = createHandle<synStreamHandle>(synStreamDestroy);
        SYN_CHECK(synStreamCreateGeneric(handlePtr.get(), handle(), flags));
        return Stream(handlePtr);
    }

    Event createEvent(const uint32_t flags = 0)
    {
        auto handlePtr = createHandle<synEventHandle>(synEventDestroy);
        SYN_CHECK(synEventCreate(handlePtr.get(), handle(), flags));
        return Event(handlePtr);
    }

    DeviceBuffer malloc(const uint64_t size, const uint64_t reqAddr = 0, const uint32_t flags = 0)
    {
        uint64_t bufferAddress = 0;
        uint32_t deviceId      = handle();
        SYN_CHECK(synDeviceMalloc(handle(), size, reqAddr, flags, &bufferAddress));

        auto deviceBufferHandlePtr = createHandleWithCustomDeleter<uint64_t>(
            [deviceId, bufferAddress]() { synDeviceFree(deviceId, bufferAddress, 0); },
            bufferAddress);
        return DeviceBuffer(deviceBufferHandlePtr, size);
    }

    HostBuffer hostMalloc(const uint64_t size, const uint32_t flags = 0)
    {
        void*    buffer;
        uint32_t deviceId = handle();
        SYN_CHECK(synHostMalloc(handle(), size, flags, &buffer));
        std::shared_ptr<void> hostBufferHandlePtr(buffer,
                                                  [deviceId, buffer](void* b) { synHostFree(deviceId, buffer, 0); });
        return HostBuffer(hostBufferHandlePtr, size);
    }

    void synchronize() const { SYN_CHECK(synDeviceSynchronize(handle())); }

    void hostMap(const void* buffer, uint64_t size) { SYN_CHECK(synHostMap(handle(), size, buffer)); }

    void hostUnmap(const void* buffer) { SYN_CHECK(synHostUnmap(handle(), buffer)); }

    DeviceMemoryInfo getMemoryInfo() const
    {
        DeviceMemoryInfo memoryInfo = {};
        SYN_CHECK(synDeviceGetMemoryInfo(handle(), &memoryInfo.free, &memoryInfo.total));
        return memoryInfo;
    }

    synDeviceInfo getInfo() const
    {
        synDeviceInfo deviceInfo;
        SYN_CHECK(synDeviceGetInfo(handle(), &deviceInfo));
        return deviceInfo;
    }

    Profiler createProfiler(synTraceType traceType) const { return Profiler(m_handle, traceType); }

private:
    Device(const std::shared_ptr<uint32_t>& handle) : SynObject(handle) {}
    Device(const std::shared_ptr<uint32_t>& handle, synDeviceType type) : SynObject(handle) {}

    friend class Context;  // Context class requires access to Device private constructor
};
}  // namespace syn
