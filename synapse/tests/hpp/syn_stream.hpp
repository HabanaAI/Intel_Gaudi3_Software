#pragma once

#include "syn_object.hpp"
#include "syn_event.hpp"
#include "syn_recipe.hpp"
#include "syn_host_buffer.hpp"
#include "syn_device_buffer.hpp"

namespace syn
{
class Stream : public SynObject<synStreamHandle>
{
public:
    Stream() = default;

    void memCopyAsync(const uint64_t src, const uint64_t dst, const uint64_t size, synDmaDir direction)
    {
        SYN_CHECK(synMemCopyAsync(handle(), src, size, dst, direction));
    }

    void memCopyAsync(const std::vector<uint64_t>& src,
                      const std::vector<uint64_t>& dst,
                      const std::vector<uint64_t>& size,
                      synDmaDir                    direction)
    {
        SYN_CHECK(synMemCopyAsyncMultiple(handle(), src.data(), size.data(), dst.data(), direction, src.size()));
    }

    void memCopyAsync(const HostBuffer& src, const DeviceBuffer& dst)
    {
        SYN_THROW_IF(dst.getSize() < src.getSize(), synInvalidArgument);
        memCopyAsync(src.getAddress(), dst.getAddress(), src.getSize(), synDmaDir::HOST_TO_DRAM);
    }

    void memCopyAsync(const DeviceBuffer& src, const HostBuffer& dst)
    {
        SYN_THROW_IF(dst.getSize() < src.getSize(), synInvalidArgument);
        memCopyAsync(src.getAddress(), dst.getAddress(), src.getSize(), synDmaDir::DRAM_TO_HOST);
    }

    void memCopyAsync(const DeviceBuffer& src, const DeviceBuffer& dst)
    {
        SYN_THROW_IF(dst.getSize() < src.getSize(), synInvalidArgument);
        memCopyAsync(src.getAddress(), dst.getAddress(), src.getSize(), synDmaDir::DRAM_TO_DRAM);
    }

    void record(const Event& eventToRecord) const { SYN_CHECK(synEventRecord(eventToRecord.handle(), handle())); }

    void waitEvent(const Event& eventToWaitOn, const uint32_t flags = 0)
    {
        SYN_CHECK(synStreamWaitEvent(handle(), eventToWaitOn.handle(), flags));
    }

    void synchronize() const { SYN_CHECK(synStreamSynchronize(handle())); }

    void launch(const Recipe&                              recipe,
                const std::vector<synLaunchTensorInfoExt>& tensorsInfo,
                const uint64_t                             workspaceBuffer,
                uint32_t                                   flags = 0)
    {
        SYN_CHECK(
            synLaunchExt(handle(), tensorsInfo.data(), tensorsInfo.size(), workspaceBuffer, recipe.handle(), flags));
    }

    void launch(const Recipe&                              recipe,
                const std::vector<synLaunchTensorInfoExt>& tensorsInfo,
                const uint64_t                             workspaceBuffer,
                const Events&                              events,
                uint32_t                                   flags = 0)
    {
        SYN_CHECK(synLaunchWithExternalEventsExt(handle(),
                                                 tensorsInfo.data(),
                                                 tensorsInfo.size(),
                                                 workspaceBuffer,
                                                 recipe.handle(),
                                                 getHandles<synEventHandle>(events).data(),
                                                 events.size(),
                                                 flags));
    }

    void launch(const Recipe&                           recipe,
                const std::vector<synLaunchTensorInfo>& tensorsInfo,
                const uint64_t                          workspaceBuffer,
                uint32_t                                flags = 0)
    {
        SYN_CHECK(synLaunch(handle(), tensorsInfo.data(), tensorsInfo.size(), workspaceBuffer, recipe.handle(), flags));
    }

    void launch(const Recipe&                           recipe,
                const std::vector<synLaunchTensorInfo>& tensorsInfo,
                const uint64_t                          workspaceBuffer,
                const Events&                           events,
                uint32_t                                flags = 0)
    {
        SYN_CHECK(synLaunchWithExternalEvents(handle(),
                                              tensorsInfo.data(),
                                              tensorsInfo.size(),
                                              workspaceBuffer,
                                              recipe.handle(),
                                              getHandles<synEventHandle>(events).data(),
                                              events.size(),
                                              flags));
    }

    synStatus query() const { return synStreamQuery(handle()); }

    void memsetD8Async(DeviceBuffer deviceBuffer, uint8_t value, uint64_t numOfElements)
    {
        SYN_CHECK(synMemsetD8Async(deviceBuffer.getAddress(), value, numOfElements, handle()));
    }

    void memsetD16Async(DeviceBuffer deviceBuffer, uint16_t value, uint64_t numOfElements)
    {
        SYN_CHECK(synMemsetD16Async(deviceBuffer.getAddress(), value, numOfElements, handle()));
    }

    void memsetD32Async(DeviceBuffer deviceBuffer, uint32_t value, uint64_t numOfElements)
    {
        SYN_CHECK(synMemsetD32Async(deviceBuffer.getAddress(), value, numOfElements, handle()));
    }

private:
    Stream(std::shared_ptr<synStreamHandle> handle) : SynObject(handle) {}

    friend class Device;  // Device class requires access to Stream private constructor
};
}  // namespace syn