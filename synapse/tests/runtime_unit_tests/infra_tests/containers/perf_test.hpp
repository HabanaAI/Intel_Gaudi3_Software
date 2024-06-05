#pragma once
#include <condition_variable>

struct StartParams
{
    std::mutex              mtx;
    std::condition_variable startCndVar;
    bool                    startTriggered          = false;
    unsigned                iterations_per_sleep    = 0;
    unsigned                internalLoopInterations = 0;
};

template<class F, class... Args>
std::function<void()>
wrapFunction(F f, StartParams& startParams, unsigned threadId, [[maybe_unused]] uint64_t& time, Args... args)
{
    return [f, &startParams, threadId, &time, args...]() {
        (void)time;  // [[maybe_unused]] not supported within lambda capture for now

        unsigned       iteratons_before_sleep = 0;
        const unsigned sleep_iterations       = 10;
        if (startParams.iterations_per_sleep)
        {
            iteratons_before_sleep = threadId * 3;
            // sleep_iterations = unsigned((double(rand()) / RAND_MAX) * sleep_iterations + 0.5);
        }

        {
            std::unique_lock<std::mutex> lck(startParams.mtx);
            startParams.startCndVar.wait(lck, [&startParams]() { return startParams.startTriggered; });
        }

#ifdef MEASURE_IN_THREADS
        auto start = std::chrono::high_resolution_clock::now();
#endif
        for (unsigned i = 0; i < startParams.internalLoopInterations; i++)
        {
            f(threadId, args...);
            if (startParams.iterations_per_sleep)
            {
                if (iteratons_before_sleep == 0)
                {
                    std::atomic<int> v {0};
                    for (unsigned j = 0; j < sleep_iterations; ++j)
                        v = (double(i) * i) / double(i + 78.4);
                    iteratons_before_sleep = startParams.iterations_per_sleep + 1;
                }
                iteratons_before_sleep--;
            }
        }
#ifdef MEASURE_IN_THREADS
        auto end = std::chrono::high_resolution_clock::now();
        time     = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#endif
    };
}

struct TestParams
{
    unsigned maxNbThreads            = 10;
    unsigned minNbThreads            = 1;
    unsigned nbTests                 = 50;
    unsigned internalLoopInterations = 10000;
    unsigned interations_per_sleep   = 0;  // no sleep
    unsigned userparam               = 0;
};

struct ThreadOperationMeasurementResult
{
    unsigned nbThreads;
    double   minTimeUs;
    double   avrgTimeUs;
    double   maxTimeUs;
};
using OperationResults = std::vector<ThreadOperationMeasurementResult>;
struct OperationFullMeasurementResults
{
    std::string      name;
    OperationResults operationResults;
};

// Register the function as a benchmark
template<class F>
OperationFullMeasurementResults
measure(F f, const char* name, TestParams params, std::function<void()> initFunc = nullptr)
{
    OperationFullMeasurementResults results;
    results.name = name;
    for (unsigned cur_thread_count = params.minNbThreads; cur_thread_count <= params.maxNbThreads; ++cur_thread_count)
    {
#ifdef MEASURE_IN_THREADS
        uint64_t min_time  = 999999999;
        uint64_t max_time  = 0;
        uint64_t avrg_time = 0;
#endif

        uint64_t avrg_full_execution_time = 0;
        uint64_t min_full_execution_time  = 9999999;
        uint64_t max_full_execution_time  = 0;
        for (unsigned i = 0; i < params.nbTests; ++i)
        {
            std::vector<std::thread> threads;

            std::vector<uint64_t> times(cur_thread_count, 0);

            StartParams startParams;
            startParams.iterations_per_sleep    = params.interations_per_sleep;
            startParams.internalLoopInterations = params.internalLoopInterations;
            for (unsigned t = 0; t < cur_thread_count; ++t)
            {
                threads.emplace_back(wrapFunction(f, std::ref(startParams), t, std::ref(times[t])));
            }

            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                std::unique_lock<std::mutex> lck(startParams.mtx);
                startParams.startTriggered = true;
                startParams.startCndVar.notify_all();
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            for (auto& thread : threads)
            {
                thread.join();
            }
            auto     end_time = std::chrono::high_resolution_clock::now();
            uint64_t full_execution_time =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
#ifdef MEASURE_IN_THREADS
            auto cur_time = std::accumulate(times.begin(), times.end(), uint64_t(0)) / cur_thread_count;
            min_time      = std::min(cur_time, min_time);
            max_time      = std::max(cur_time, max_time);
            avrg_time += cur_time;
#endif
            avrg_full_execution_time += full_execution_time;
            min_full_execution_time = std::min(min_full_execution_time, full_execution_time);
            max_full_execution_time = std::max(max_full_execution_time, full_execution_time);
            if (initFunc)
            {
                initFunc();
            }
        }
#ifdef MEASURE_IN_THREADS
        avrg_time /= params.nbTests;
#endif

        avrg_full_execution_time /= params.nbTests;

        ThreadOperationMeasurementResult result;
        result.nbThreads  = cur_thread_count;
        result.minTimeUs  = min_full_execution_time / double(params.internalLoopInterations);
        result.avrgTimeUs = avrg_full_execution_time / double(params.internalLoopInterations);
        result.maxTimeUs  = max_full_execution_time / double(params.internalLoopInterations);
        results.operationResults.push_back(result);

        //        std::cout << name << " threads: " << cur_thread_count
        //#ifdef MEASURE_IN_THREADS
        //                  << " : min:\t" << min_time << "us avrg:\t" << avrg_time << "us max:\t" << max_time <<"us"
        //#endif
        //                  << " full execution time: min:\t" << min_full_execution_time << "us op:\t" <<
        //                  min_full_execution_time / float(params.internalLoopInterations) <<  "us avrg:\t"
        //                  << avrg_full_execution_time << "us max:\t" << max_full_execution_time << "us\n";
    }
    return results;
}

inline void PrintTestResults(std::string const& title, std::initializer_list<OperationFullMeasurementResults> results)
{
    std::set<unsigned> threads;
    for (auto const& res : results)
    {
        for (auto const& opRes : res.operationResults)
        {
            threads.insert(opRes.nbThreads);
        }
    }
    // print header
    std::string header        = "|  operation time(us)  max/avrg/min \\ threads  |";
    unsigned    nameFielsSize = header.size();
    unsigned    valFieldSize  = 5;
    for (auto nbThreads : threads)
    {
        char buf[128];
        sprintf(buf, "  %2d  |", nbThreads);
        header += buf;
        valFieldSize = strlen(buf);
    }

    std::string splitter;
    splitter.resize(header.size(), '-');
    splitter += '\n';
    std::cerr << splitter;
    int titleOffset = (int(header.size()) - int(title.size())) / 2;
    if (titleOffset < 0) titleOffset = 0;
    int suffixSize = int(header.size()) - titleOffset - int(title.size()) - 2;
    if (suffixSize < 0) suffixSize = 0;
    char titleBuffer[1024];
    sprintf(titleBuffer, "|%-*s%s%-*s|", titleOffset, " ", title.c_str(), suffixSize, " ");
    std::cerr << titleBuffer << '\n';
    std::cerr << splitter;
    std::cerr << header << '\n';
    std::cerr << splitter;

    for (auto const& res : results)
    {
        std::string maxLine, avrgLine, minLine;
        char        buf[1024];
        sprintf(buf, "|%-*s|", nameFielsSize - 2, " ");
        maxLine = buf;
        minLine = buf;

        sprintf(buf, "|%-*s|", nameFielsSize - 2, res.name.c_str());
        avrgLine = buf;

        for (auto nbThreads : threads)
        {
            char maxBuf[1024];
            char avrgBuf[1024];
            char minBuf[1024];
            sprintf(maxBuf, "%*s|", nameFielsSize - 1, " ");
            sprintf(avrgBuf, "%*s|", nameFielsSize - 1, " ");
            sprintf(minBuf, "%*s|", nameFielsSize - 1, " ");
            for (auto const& opRes : res.operationResults)
            {
                if (nbThreads == opRes.nbThreads)
                {
                    sprintf(maxBuf, "%*.3f|", valFieldSize - 1, opRes.maxTimeUs);
                    sprintf(avrgBuf, "%*.3f|", valFieldSize - 1, opRes.avrgTimeUs);
                    sprintf(minBuf, "%*.3f|", valFieldSize - 1, opRes.minTimeUs);
                    break;
                }
            }
            maxLine += maxBuf;
            avrgLine += avrgBuf;
            minLine += minBuf;
        }

        std::cerr << maxLine << '\n';
        std::cerr << avrgLine << '\n';
        std::cerr << minLine << '\n';
        std::cerr << splitter;
    }
};