#pragma once
#include <condition_variable>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "fs_mme_debug.h"
#include "fs_mme_unit.h"

namespace Gaudi2
{
namespace Mme
{
#ifdef MME_DEADLOCK_DEBUG
class ConditionVariable
{
   public:
    ConditionVariable() : m_name("")
    {
        std::lock_guard<std::mutex> lock(s_registrationMutex);
        s_instances.push_back(this);
    }

    ConditionVariable(const ConditionVariable&) = delete;
    ConditionVariable& operator=(const ConditionVariable&) = delete;

    void wait(std::unique_lock<std::mutex>& lock)
    {
        const auto id = std::this_thread::get_id();
        m_pendingMutex.lock();
        m_pendingThreads.push_back(id);
        m_pendingMutex.unlock();

        m_cond.wait(lock);

        m_pendingMutex.lock();
        m_pendingThreads.remove(id);
        m_pendingMutex.unlock();
    }

    void notify_all() { m_cond.notify_all(); }
    void notify_one() { m_cond.notify_one(); }

    void                              setName(const std::string& name) { m_name = name; }
    const std::string&                getName() const { return m_name; }
    const std::list<std::thread::id>& getPendingThreads() const { return m_pendingThreads; }

    static const std::vector<ConditionVariable*>& getInstances() { return s_instances; }

    // during debug
    void lock() { m_pendingMutex.lock(); }
    void unlock() { m_pendingMutex.unlock(); }

   private:
    static std::mutex                      s_registrationMutex;
    static std::vector<ConditionVariable*> s_instances;

    std::string                m_name;
    std::mutex                 m_pendingMutex;
    std::list<std::thread::id> m_pendingThreads;
    std::condition_variable    m_cond;
};
#else
typedef std::condition_variable ConditionVariable;
#endif
} // namespace Mme
} // namespace Gaudi2
