#pragma once

#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

namespace synapse
{

class ThreadWorkItem;

/**
 * Utility to manage number of threads executing work items
 *
 * Note: ThreadPool itself is not thread safe
 *       it protect only its own created threads.
 *       start() and finish() MUST BE CALLED FROM THE SAME THREAD
 */
class ThreadPool
{
public:
    explicit ThreadPool();

    explicit ThreadPool(uint32_t numOfThreads);

    virtual ~ThreadPool();

    /**
     * Start threads work
     */
    void start();

    /**
     * Finish all remaining jobs and wait for all threads to finish
     */
    void finish();

    /**
     * Add new job, return false if the insertion failed
     * -- Insertion of new job can fail if the ThreadPool is not started or finished
     */
    bool addJob(ThreadWorkItem* workItem);

private:
    void threadWorkFunction();

    const uint32_t m_numOfThreads;
    std::list<ThreadWorkItem*> m_workItems;
    std::vector<std::thread> m_threads;

    std::mutex m_mutex;
    std::condition_variable m_cv;

    volatile bool m_working;
    volatile bool m_stop;
};

}
