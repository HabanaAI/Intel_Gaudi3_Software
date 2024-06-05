#include <sys/sysinfo.h>

#include "habana_global_conf.h"

#include "thread_work_item.h"
#include "thread_pool.h"

namespace synapse
{

ThreadPool::ThreadPool()
: m_numOfThreads(GCFG_NUM_OF_THREADS_CONF.value())
, m_working(false)
, m_stop(false)
{
}

ThreadPool::ThreadPool(uint32_t numOfThreads)
: m_numOfThreads(numOfThreads)
, m_working(false)
, m_stop(false)
{
}

ThreadPool::~ThreadPool()
{
    finish();
}

void ThreadPool::start()
{
    // If only one thread - using the calling thread
    if (m_working || m_numOfThreads <= 1) return;

    for (uint32_t i = 0; i < m_numOfThreads; ++i)
    {
        m_threads.push_back(std::thread(&ThreadPool::threadWorkFunction, std::ref(*this)));
    }
    m_working = true;
}

void ThreadPool::finish()
{
    if (! m_working) return;

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
        m_cv.notify_all();
    }

    for (std::thread& t : m_threads)
    {
        t.join();
    }

    m_threads.clear();
    m_stop = false;
    m_working = false;
}

bool ThreadPool::addJob(ThreadWorkItem* workItem)
{
    if (workItem == nullptr) return false;

    if (m_numOfThreads <= 1)
    {
        // If only one thread is configured using the calling thread
        workItem->call();
        return true;
    }

    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_stop) return false;

    m_workItems.push_back(workItem);

    // Unlock mutex prior to notify.  Otherwise, the worker thread
    // may get notified before this thread actually releases the lock,
    // so it would immediately block again.
    lock.unlock();
    m_cv.notify_one();

    return true;
}

void ThreadPool::threadWorkFunction()
{
    while (true)
    {
        ThreadWorkItem* wi = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            while (m_workItems.empty())
            {
                if (m_stop) return;
                m_cv.wait(lock);
            }
            wi = m_workItems.front();
            m_workItems.pop_front();
        }
        wi->call();
    }
}

}
