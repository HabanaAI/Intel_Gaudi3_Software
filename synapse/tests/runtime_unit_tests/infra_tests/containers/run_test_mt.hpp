#pragma once
#include <thread>
#include <condition_variable>
#include <atomic>

inline void runTestsMT(std::function<void(unsigned, unsigned)> testFunc, unsigned nbExecutions, unsigned nbThreads)
{
    std::vector<std::thread> threads;
    std::condition_variable  cv;
    std::mutex               mtx;
    bool                     go = false;
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        threads.push_back(std::thread(
            [&](unsigned thread_id) {
                {
                    std::unique_lock<std::mutex> lck(mtx);
                    cv.wait(lck, [&go]() { return go; });
                }
                for (unsigned i = 0; i < nbExecutions; ++i)
                {
                    testFunc(thread_id, i);
                }
            },
            i));
    };
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    {
        std::unique_lock<std::mutex> lck(mtx);
        go = true;
        cv.notify_all();
    }
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        threads[i].join();
    }
}
