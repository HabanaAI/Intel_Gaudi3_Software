#pragma once

#include "syn_device.hpp"
#include "syn_graph.hpp"
#include <mutex>

namespace syn
{
class Context
{
    class SynapseInstance
    {
    public:
        SynapseInstance() { SYN_CHECK(synInitialize()); }
        ~SynapseInstance() { synDestroy(); }
    };

public:
    Context()
    {
        static std::weak_ptr<SynapseInstance> cache;

        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        auto tmp = cache.lock();
        if (tmp == nullptr)
        {
            tmp.reset(new SynapseInstance());
            cache = tmp;
        }
        m_instance = std::move(tmp);
    }

    std::string getDriverVersion() const
    {
        char version[maxStringLength];
        SYN_CHECK(synDriverGetVersion(version, maxStringLength));
        return version;
    }

    void setConfiguration(const std::string& configurationName, const std::string& value)
    {
        SYN_CHECK(synConfigurationSet(configurationName.c_str(), value.c_str()));
    }

    std::string getConfiguration(const std::string& configurationName) const
    {
        char value[maxStringLength];
        SYN_CHECK(synConfigurationGet(configurationName.c_str(), value, maxStringLength));
        return value;
    }

    std::vector<uint64_t> getAttributes(synDeviceType                          deviceType,
                                        const std::vector<synDeviceAttribute>& attributes) const
    {
        std::vector<uint64_t> ret(attributes.size());
        SYN_CHECK(synDeviceTypeGetAttribute(ret.data(), attributes.data(), attributes.size(), deviceType));
        return ret;
    }

    uint32_t getDeviceCount() const
    {
        uint32_t count;
        SYN_CHECK(synDeviceGetCount(&count));
        return count;
    }

    uint32_t getDeviceCount(synDeviceType type) const
    {
        uint32_t count;
        SYN_CHECK(synDeviceGetCountByDeviceType(&count, type));
        return count;
    }

    Device acquire(synDeviceType type)
    {
        uint32_t deviceHandle;
        SYN_CHECK(synDeviceAcquireByDeviceType(&deviceHandle, type));
        auto deviceHandlePtr = createHandle<uint32_t>(synDeviceRelease, deviceHandle);
        return Device(deviceHandlePtr, type);
    }

    Device acquire(synModuleId moduleId)
    {
        uint32_t deviceHandle;
        SYN_CHECK(synDeviceAcquireByModuleId(&deviceHandle, moduleId));
        auto deviceHandlePtr = createHandle<uint32_t>(synDeviceRelease, deviceHandle);
        return Device(deviceHandlePtr);
    }

    Device acquire(const char* pciBus)
    {
        uint32_t deviceHandle;
        SYN_CHECK(synDeviceAcquire(&deviceHandle, pciBus));
        auto deviceHandlePtr = createHandle<uint32_t>(synDeviceRelease, deviceHandle);
        return Device(deviceHandlePtr);
    }

    Graph createGraph(synDeviceType deviceType) { return Graph(deviceType); }

    EagerGraph createEagerGraph(synDeviceType deviceType) { return EagerGraph(deviceType); }

private:
    std::shared_ptr<SynapseInstance> m_instance;
};
}  // namespace syn
