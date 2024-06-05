#include "efd_controller.hpp"
#include "synapse_common_types.h"
#include "synapse_runtime_logging.h"
#include "habana_global_conf.h"
#include "drm/habanalabs_accel.h"
#include "device/device_interface.hpp"
#include "hlthunk.h"
#include <unistd.h>
#include "defs.h"

const int         INVALID_NOTIFIER = -1;
const std::string threadName       = "Habana-efd";

EventFdController::EventFdController(int fd, DeviceInterface* pDeviceInterface)
: m_fd(fd),
  m_pDeviceInterface(pDeviceInterface),
  m_notifierHandle(INVALID_NOTIFIER),
  m_finish(false),
  m_thread(nullptr)
{
    HB_ASSERT(m_pDeviceInterface != nullptr, "Invalid device pointer");
    LOG_TRACE(SYN_EVENT_FD, "EventFdController created m_sleepTime {} ms", m_sleepTime.count());

    uint64_t threadSleepInterval = GCFG_EVENT_FD_THREAD_SLEEP_TIME.value();
    uint64_t debugSleepInterval  = GCFG_EVENT_FD_THREAD_DEBUG_INTERVAL.value();
    uint64_t sleepInterval       = 0;
    if (debugSleepInterval == 0)
    {
        m_debugEnabled = false;
        sleepInterval  = threadSleepInterval;
    }
    else
    {
        m_debugEnabled = true;
        sleepInterval  = std::min(threadSleepInterval, debugSleepInterval);
    }
    m_sleepTime = std::chrono::milliseconds(sleepInterval);

    // EventFdController execute bgWork (residual mechanism) once every ~(5 * GCFG_EVENT_FD_THREAD_SLEEP_TIME) ms
    m_interceptHostFailureCounterMax = (5 * (GCFG_EVENT_FD_THREAD_SLEEP_TIME.value() / sleepInterval)) - 1;
}

synStatus EventFdController::start()
{
    LOG_TRACE(SYN_EVENT_FD, "EventFdController start");
    HB_ASSERT(m_notifierHandle == INVALID_NOTIFIER, "Invalid notifier");
    int rc = hlthunk_notifier_create(m_fd);
    if (rc < 0)
    {
        LOG_ERR(SYN_EVENT_FD, "{}: hlthunk_notifier_create failed with status {}", HLLOG_FUNC, rc);
        return synFail;
    }
    m_notifierHandle = rc;
    LOG_TRACE(SYN_EVENT_FD, "EventFdController notifier {} created", m_notifierHandle);
    HB_ASSERT(m_thread == nullptr, "Invalid thread pointer");
    m_thread = std::make_unique<std::thread>(&EventFdController::_mainLoop, this);
    LOG_TRACE(SYN_EVENT_FD, "EventFdController thread created");
    return synSuccess;
}

synStatus EventFdController::stop()
{
    LOG_TRACE(SYN_EVENT_FD, "EventFdController stop");
    if (!m_thread.get())
    {
        return synSuccess;
    }
    m_finish = true;
    m_thread->join();
    m_thread = nullptr;
    LOG_TRACE(SYN_EVENT_FD, "EventFdController thread destroyed");
    HB_ASSERT(m_notifierHandle != INVALID_NOTIFIER, "Invalid notifier");
    int rc = hlthunk_notifier_release(m_fd, m_notifierHandle);
    if (rc)
    {
        LOG_ERR(SYN_EVENT_FD, "{} hlthunk_notifier_release failed with status {}", HLLOG_FUNC, rc);
        return synFail;
    }
    m_notifierHandle = INVALID_NOTIFIER;
    LOG_TRACE(SYN_EVENT_FD, "EventFdController notifier destroyed");
    return synSuccess;
}

void EventFdController::_mainLoop()
{
    LOG_TRACE(SYN_EVENT_FD, "EventFdController thread running");

    pthread_setname_np(pthread_self(), threadName.c_str());

    unsigned interceptHostFailureCounter = 0;

    // Operations done during this thread:
    //      1) In case relevant debug-mode enabled, check stauts
    //      2) Query events' status (calling the Driver)
    //      3) Perform "background-work"
    while (true)
    {
        if (m_finish)
        {
            LOG_TRACE(SYN_EVENT_FD, "EventFdController thread finish indication");
            break;
        }

        if (m_debugEnabled)
        {
            // Devug-Check work-status
            m_pDeviceInterface->debugCheckWorkStatus();
        }

        // Intercept failures from the driver (main mechanism)
        _queryEvent(m_notifierHandle);

        // Intercept failures from the host (residual mechanism in Gaudi2/3)
        if (interceptHostFailureCounter == m_interceptHostFailureCounterMax)
        {
            interceptHostFailureCounter = 0;

            m_pDeviceInterface->bgWork();
        }
        else
        {
            interceptHostFailureCounter++;
        }
    }

    LOG_TRACE(SYN_EVENT_FD, "EventFdController thread termination");
}

void EventFdController::_queryEvent(int notifierHandle)
{
    uint64_t notifier_events = 0;
    uint64_t notifier_cnt    = 0;
    uint32_t flags           = 0;

    int rc = hlthunk_notifier_recv(m_fd, notifierHandle, &notifier_events, &notifier_cnt, flags, m_sleepTime.count());
    if (rc)
    {
        LOG_ERR(SYN_EVENT_FD, "{} hlthunk_notifier_recv failed with status {}", HLLOG_FUNC, rc);
        sleep(1);  // not sure how we can get here, but if we do, make sure not to loop in a tight loop
        return;
    }

    // hlthunk has/had a bug were hlthunk_notifier_recv might return with notifier_events=0 but
    // notifier_cnt!=NOTIFIER_TIMEOUT. hl-thunk will solve it, but adding a check for (notifier_events == 0) just to be
    // on the safe side
    if ((notifier_cnt == NOTIFIER_TIMEOUT) || (notifier_events == 0))
    {
        return;
    }
    else
    {
        if ((notifier_events & HL_NOTIFIER_EVENT_DEVICE_RESET) ||
            (notifier_events & HL_NOTIFIER_EVENT_DEVICE_UNAVAILABLE))
        {
            LOG_ERR(SYN_EVENT_FD, "{} events {:#x} cnt {}", HLLOG_FUNC, notifier_events, notifier_cnt);
        }
        else
        {
            LOG_TRACE(SYN_EVENT_FD, "{} events {:#x} cnt {}", HLLOG_FUNC, notifier_events, notifier_cnt);
        }

        m_pDeviceInterface->notifyEventFd(notifier_events);
    }
}

void EventFdController::testingOnlySetBgFreq(std::chrono::milliseconds period)
{
    auto oldFreq = m_sleepTime;
    m_sleepTime  = period;
    LOG_INFO_T(SYN_EVENT_FD, "{}: m_timeToSleepMs freq set to {}", HLLOG_FUNC, m_sleepTime.count());
    // sleep before returning to caller; It takes around the previous frequency for the
    // request to take effect (as we might be sleeping on the previous value)
    std::this_thread::sleep_for(oldFreq * 2 + std::chrono::milliseconds(100));
}
