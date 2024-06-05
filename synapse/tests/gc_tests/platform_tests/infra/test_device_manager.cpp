#include "test_device_manager.h"
#include "hpp/syn_context.hpp"
#include "infra/gc_tests_utils.h"
#include "log_manager.h"
#include "synapse_common_types.h"
#include <exception>
#include <memory>

TestDeviceManager::DeviceReleaser::DeviceReleaser(syn::Device                                    device,
                                                  const std::function<void(syn::Device device)>& releaser)
: m_device(device), m_releaser(releaser)
{
}

TestDeviceManager::DeviceReleaser::~DeviceReleaser()
{
    m_releaser(m_device);
}

syn::Device& TestDeviceManager::DeviceReleaser::dev()
{
    return m_device;
}

TestDeviceManager& TestDeviceManager::TestDeviceManager::instance()
{
    static TestDeviceManager instance;
    return instance;
}

TestDeviceManager::~TestDeviceManager()
{
    releaseDevices();
};

std::vector<syn::Device> TestDeviceManager::acquireDevices(synDeviceType deviceType)
{
    std::set<synDeviceType> optionalDeviceTypes;

    switch (deviceType)
    {
        case synDeviceGreco:
            optionalDeviceTypes = {synDeviceGreco};
            break;
        case synDeviceGaudi:
            optionalDeviceTypes = {synDeviceGaudi};
            break;
        case synDeviceGaudi2:
            optionalDeviceTypes = {synDeviceGaudi2};
            break;
        case synDeviceGaudi3:
            optionalDeviceTypes = {synDeviceGaudi3};
            break;
        case synDeviceEmulator:
        case synDeviceTypeInvalid:
        case synDeviceTypeSize:
        default:
            break;
    }

    std::vector<syn::Device> devices;
    auto                     dev = tryAcquireDevice(optionalDeviceTypes);
    if (dev)
    {
        devices.push_back(dev);
    }
    return devices;
}

void TestDeviceManager::releaseDevices()
{
    m_devices.clear();
}

void TestDeviceManager::reset()
{
    releaseDevices();
    m_ctx.reset(nullptr);
}

bool TestDeviceManager::synInitialized()
{
    return bool(m_ctx);
}

bool TestDeviceManager::shouldAcquireDevices()
{
    if (m_devices.empty()) return true;
    const auto& dev = m_devices.back();
    try
    {
        dev.getInfo();  // should throw in case the device is in a bad state
        return false;
    }
    catch (const std::exception& e)
    {
        LOG_DEBUG(SYN_TEST, "get device info failed, error: {}, will try to re-acquire", e.what());
        return true;
    }
    catch (...)
    {
        LOG_DEBUG(SYN_TEST, "get device info failed, will try to re-acquire");
        return true;
    }
}

std::shared_ptr<TestDeviceManager::DeviceReleaser> TestDeviceManager::acquireDevice(synDeviceType deviceType)
{
    if (shouldAcquireDevices())
    {
        releaseDevices();
        m_devices = acquireDevices(deviceType);
        if (m_devices.empty())
            throw std::runtime_error("failed to acquire device device type: " + synDeviceTypeToString(deviceType) +
                                     ", did you specified a device? (-c <device>)");
    }
    auto dev = m_devices.back();
    m_devices.pop_back();
    auto releaser = [&](syn::Device device) { returnDevice(device); };
    return std::make_shared<DeviceReleaser>(dev, releaser);
}

syn::Device TestDeviceManager::tryAcquireDevice(const std::set<synDeviceType>& deviceTypes)
{
    m_ctx = std::make_unique<syn::Context>();
    for (const auto& t : deviceTypes)
    {
        if (m_ctx->getDeviceCount(t) == 0) continue;
        try
        {
            LOG_DEBUG(SYN_TEST, "try to acquire device: {}", t);
            return m_ctx->acquire(t);
        }
        catch (const std::exception& e)
        {
            LOG_DEBUG(SYN_TEST, "acquire device attempt failed, error: {}", e.what());
        }
        catch (...)
        {
            LOG_DEBUG(SYN_TEST, "acquire device attempt failed");
        }
    }
    return syn::Device();
}

void TestDeviceManager::returnDevice(syn::Device dev)
{
    if (dev)
    {
        m_devices.push_back(dev);
    }
}