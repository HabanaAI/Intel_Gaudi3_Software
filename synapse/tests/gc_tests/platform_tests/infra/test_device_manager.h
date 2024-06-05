
#pragma once

#include "hpp/synapse.hpp"
#include "synapse_common_types.h"
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>

class TestDeviceManager
{
public:
    class DeviceReleaser
    {
    public:
        DeviceReleaser(syn::Device device, const std::function<void(syn::Device device)>& releaser);
        ~DeviceReleaser();
        syn::Device& dev();

    private:
        syn::Device                                   m_device;
        const std::function<void(syn::Device device)> m_releaser;
    };

    TestDeviceManager(const TestDeviceManager&)  = delete;  // copy constructor
    TestDeviceManager(const TestDeviceManager&&) = delete;  // move constructor
    ~TestDeviceManager();

    TestDeviceManager& operator=(const TestDeviceManager&) = delete;   // assignment operator
    TestDeviceManager& operator=(const TestDeviceManager&&) = delete;  // move operator

    static TestDeviceManager&       instance();
    std::shared_ptr<DeviceReleaser> acquireDevice(synDeviceType deviceType);

    void reset();

    bool synInitialized();

private:
    TestDeviceManager() = default;

    // create device pool
    bool                     shouldAcquireDevices();
    std::vector<syn::Device> acquireDevices(synDeviceType deviceType);

    // destroy device pool
    void releaseDevices();

    // return a device to the device pool
    void returnDevice(syn::Device dev);

    syn::Device tryAcquireDevice(const std::set<synDeviceType>& deviceTypes);

    std::unique_ptr<syn::Context> m_ctx;
    synDeviceType                 m_deviceType;
    std::vector<syn::Device>      m_devices;
};
