#pragma once

#include "performance_base_test.h"
#include "gc_tests/platform_tests/infra/gc_tests_utils.h"
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <condition_variable>
#include <future>

namespace json_tests
{
template<typename T>
class ThreadSafeQueue
{
public:
    void push(const T& item)
    {
        std::unique_lock locker(m_lock);
        m_queue.push(item);
        m_cv.notify_all();
    }

    void push(T&& item)
    {
        std::unique_lock locker(m_lock);
        m_queue.push(std::move(item));
        m_cv.notify_all();
    }

    template<typename... ArgTypes>
    void emplace(ArgTypes&&... args)
    {
        std::unique_lock locker(m_lock);
        m_queue.emplace(std::forward<ArgTypes>(args)...);
        m_cv.notify_all();
    }

    std::optional<T> pop()
    {
        std::unique_lock locker(m_lock);
        m_cv.wait(locker, [this] { return m_finished || !m_queue.empty(); });
        if (m_queue.empty()) return std::nullopt;
        T item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    bool empty()
    {
        std::unique_lock locker(m_lock);
        return m_queue.empty();
    }

    void setFinished(bool value)
    {
        m_finished = value;
        m_cv.notify_all();
    }

    bool finished()
    {
        std::unique_lock locker(m_lock);
        return m_finished && m_queue.empty();
    }

private:
    std::queue<T>           m_queue;
    std::mutex              m_lock;
    std::condition_variable m_cv;
    std::atomic<bool>       m_finished = false;
};

class MultiThreadedPlaybackTest final : private PerformanceBaseTest
{
public:
    MultiThreadedPlaybackTest(const ArgParser& args) : PerformanceBaseTest(args, true, false, true) {}

    void run() override;

    // typical CPU cache line size
    static constexpr size_t CPU_CACHE_LINE_SIZE = 64;

private:
    struct GraphAndRecipeParams
    {
        GraphParams  graph;
        RecipeParams recipe;
    };

    struct CompilationQueueElement
    {
        CompilationQueueElement(GraphAndRecipeParams& _params, synGraphHandle _graph) : params(_params), graph(_graph)
        {
        }

        GraphAndRecipeParams& params;
        synGraphHandle        graph;
    };

    struct ExecutionQueueElement
    {
        ExecutionQueueElement(GraphAndRecipeParams& _params, synGraphHandle _graph, synRecipeHandle _recipe)
        : params(_params), graph(_graph), recipe(_recipe)
        {
        }

        GraphAndRecipeParams& params;
        synGraphHandle        graph;
        synRecipeHandle       recipe;
    };
    // queues connecting the threads for the pipeline phases
    ThreadSafeQueue<CompilationQueueElement> m_compilationQueue;
    ThreadSafeQueue<ExecutionQueueElement>   m_launchQueue;
    ThreadSafeQueue<ExecutionQueueElement>   m_deviceSynchronizeQueue;
    ThreadSafeQueue<ExecutionQueueElement>   m_cleanupQueue;
    // parsed json graphs
    std::vector<GraphAndRecipeParams> m_params;
    // lock protecting logging
    std::mutex m_logLock;
    // statistics
    struct PipelinePhaseStatistics
    {
        void clear()
        {
            waitTime.clear();
            workTime.clear();
        }

        void reserve(size_t capacity)
        {
            waitTime.reserve(capacity);
            workTime.reserve(capacity);
        }

        std::vector<uint64_t> waitTime;
        std::vector<uint64_t> workTime;
        // padding to avoid false cpu cache line sharing between pipeline threads
        std::byte padding[CPU_CACHE_LINE_SIZE - 2 * sizeof(std::vector<uint64_t>)];
    };

    static_assert(sizeof(MultiThreadedPlaybackTest::PipelinePhaseStatistics) %
                      MultiThreadedPlaybackTest::CPU_CACHE_LINE_SIZE ==
                  0);

    PipelinePhaseStatistics                            m_creationStats;
    PipelinePhaseStatistics                            m_compilationStats;
    PipelinePhaseStatistics                            m_launchStats;
    PipelinePhaseStatistics                            m_deviceStats;
    PipelinePhaseStatistics                            m_cleanupStats;
    std::chrono::time_point<std::chrono::steady_clock> m_iterStartTime;

private:
    void logHandlingDuration(std::string_view                         phase,
                             std::chrono::microseconds                duration,
                             std::optional<size_t>                    graphIdx     = std::nullopt,
                             std::optional<std::chrono::microseconds> waitDuration = std::nullopt);
    void recordJsonStats();
    // parse json into graph and recipe params
    // once, before actual test iterations.
    void parseJsonGraphs();
    // pipeline threads routines
    void buildGraphs(std::future<bool>& canStart);
    void compileGraphs(std::promise<bool>& startNotifier, std::future<bool>& canStart);
    void runRecipes(std::promise<bool>& startNotifier, std::future<bool>& canStart);
    void deviceSynchronize(std::promise<bool>& startNotifier, std::future<bool>& canStart);
    void destroyGraphAndRecipes(std::promise<bool>& startNotifier);
};
}  // namespace json_tests