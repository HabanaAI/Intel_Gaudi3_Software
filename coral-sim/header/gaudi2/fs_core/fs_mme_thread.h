#pragma once
#include <map>
#include <string>
#include <thread>

#include "fs_mme_debug.h"
#include "fs_mme_queue.h"
#include "fs_mme_unit.h"

namespace Gaudi2
{
namespace Mme
{
class Thread : public Gaudi2::Mme::Unit
{
   public:
    Thread(FS_Mme* mme, const std::string& name) : Unit(mme, name), m_terminate(false), m_thread(nullptr)
    {
        m_launchMutex.lock();
        m_thread = new std::thread(&Thread::main, this);
#ifdef MME_DEADLOCK_DEBUG
        std::lock_guard<std::mutex> lock(s_registrationMutex);
        s_instances[m_thread->get_id()] = this;
#endif
    }

    virtual ~Thread()
    {
        if (m_thread)
            delete m_thread;
    }
    void restart()
    {
        m_launchMutex.lock();
        if (m_thread)
            delete m_thread;
        m_thread = new std::thread(&Thread::main, this);
#ifdef MME_DEADLOCK_DEBUG
        std::lock_guard<std::mutex> lock(s_registrationMutex);
        s_instances[m_thread->get_id()] = this;
#endif
    }
    void launch() { m_launchMutex.unlock(); }

    void join()
    {
        if (m_thread)
            m_thread->join();
    }

    void terminate() { m_terminate = true; }

#ifdef MME_DEADLOCK_DEBUG
    // for debug
    static const std::map<std::thread::id, const Thread*>&      getInstances() { return s_instances; };
    static const std::map<std::thread::id, const std::string*>& getExecThreads() { return s_exeThreads; };

    static void registerExeThread(const std::string* name)
    {
        std::unique_lock<std::mutex> lock(s_registrationMutex);
        s_exeThreads[std::this_thread::get_id()] = name;
    }

    static void unregisterExeThread()
    {
        std::unique_lock<std::mutex> lock(s_registrationMutex);
        s_exeThreads.erase(std::this_thread::get_id());
    }

    static void stopRegistration() { s_registrationMutex.lock(); }
    static void resumeRegistration() { s_registrationMutex.unlock(); }
#endif
   protected:
    virtual void execute() = 0;

    bool m_terminate;
    std::thread* m_thread;

   private:
    void main()
    {
        m_launchMutex.lock();
        execute();
        m_launchMutex.unlock();
    }

    std::mutex   m_launchMutex;
#ifdef MME_DEADLOCK_DEBUG
    static std::mutex                                    s_registrationMutex;
    static std::map<std::thread::id, const Thread*>      s_instances;
    static std::map<std::thread::id, const std::string*> s_exeThreads;
#endif
};
} // namespace Mme
} // namespace Gaudi2
