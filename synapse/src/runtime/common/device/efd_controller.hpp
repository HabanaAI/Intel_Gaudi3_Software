#pragma once

#include "synapse_common_types.h"
#include <thread>
#include <atomic>

class DeviceInterface;

class EventFdController
{
public:
    EventFdController(int fd, DeviceInterface* pDeviceInterface);
    virtual ~EventFdController() = default;
    synStatus start();
    synStatus stop();
    void testingOnlySetBgFreq(std::chrono::milliseconds period);

private:
    constexpr static uint32_t NOTIFIER_TIMEOUT = 0;

    void _mainLoop();
    void _queryEvent(int notifierHandle);

    int                          m_fd;
    DeviceInterface*             m_pDeviceInterface;
    std::chrono::milliseconds    m_sleepTime;
    unsigned                     m_interceptHostFailureCounterMax;
    bool                         m_debugEnabled;
    int                          m_notifierHandle;
    std::atomic<bool>            m_finish;
    std::unique_ptr<std::thread> m_thread;
};