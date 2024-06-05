#include "multi_threaded_playback_tests.h"

#include "base_test.h"
#include "graph_loader.h"
#include "hpp/syn_graph.hpp"
#include "json_utils.h"
#include "utils/data_provider.h"

#include <chrono>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace json_tests
{
void MultiThreadedPlaybackTest::parseJsonGraphs()
{
    m_creationStats.reserve(m_passingGraphsIndices.size());
    m_compilationStats.reserve(m_passingGraphsIndices.size());
    m_launchStats.reserve(m_passingGraphsIndices.size());
    m_cleanupStats.reserve(m_passingGraphsIndices.size());
    m_deviceStats.reserve(m_passingGraphsIndices.size());
    m_params.resize(m_passingGraphsIndices.size());
    uint64_t currentParamsIdx = 0;
    for (uint64_t i : m_passingGraphsIndices)
    {
        const auto&           currentJsonGraph = m_jsonFileLoader->getGraph(i);
        GraphAndRecipeParams& currentParams    = m_params[currentParamsIdx++];
        currentParams.graph.idx                = i;
        currentParams.graph.name               = json_utils::get(currentJsonGraph, "name");
        fillTensorAndSectionCreationParams(currentJsonGraph, currentParams.graph);
        fillNodeCreationParams(currentJsonGraph, currentParams.graph);
        fillRecipeParams(currentParams.graph, currentParams.recipe);
    }
}

void MultiThreadedPlaybackTest::buildGraphs(std::future<bool>& canStart)
{
    canStart.wait();
    m_iterStartTime                             = std::chrono::steady_clock::now();
    std::chrono::microseconds totalCreationTime = std::chrono::microseconds::zero();
    for (auto& currentParams : m_params)
    {
        const auto     startTime    = std::chrono::steady_clock::now();
        synGraphHandle currentGraph = createGraph();
        createSections(currentGraph, currentParams.graph);
        createTensors(currentGraph, currentParams.graph);
        createNodes(currentGraph, currentParams.graph);
        setBlockingNodes(currentGraph, currentParams.graph);
        const auto endTime  = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        totalCreationTime += duration;
        // logHandlingDuration("creation", duration, currentParams.graph.idx);
        m_creationStats.workTime.push_back(duration.count());
        m_compilationQueue.emplace(currentParams, currentGraph);
    }
    logHandlingDuration("creation", totalCreationTime);
    m_compilationQueue.setFinished(true);
}

void MultiThreadedPlaybackTest::compileGraphs(std::promise<bool>& startNotifier, std::future<bool>& canStart)
{
    canStart.wait();
    startNotifier.set_value(true);
    std::chrono::microseconds totalWaitTime        = std::chrono::microseconds::zero();
    std::chrono::microseconds totalCompilationTime = std::chrono::microseconds::zero();
    while (true)
    {
        const auto                             waitStartTime = std::chrono::steady_clock::now();
        std::optional<CompilationQueueElement> ctxt          = m_compilationQueue.pop();
        const auto                             waitEndTime   = std::chrono::steady_clock::now();
        const auto waitDuration = std::chrono::duration_cast<std::chrono::microseconds>(waitEndTime - waitStartTime);
        totalWaitTime += waitDuration;
        if (!ctxt.has_value()) break;
        const auto      compileStartTime = std::chrono::steady_clock::now();
        synRecipeHandle currentRecipe    = compileGraph(ctxt->graph, ctxt->params.graph.name);
        const auto      compileEndTime   = std::chrono::steady_clock::now();
        const auto      compileDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(compileEndTime - compileStartTime);
        totalCompilationTime += compileDuration;
        // logHandlingDuration("compilation", compileDuration, ctxt->params.graph.idx, waitDuration);
        m_compilationStats.waitTime.push_back(waitDuration.count());
        m_compilationStats.workTime.push_back(compileDuration.count());
        m_launchQueue.emplace(ctxt->params, ctxt->graph, currentRecipe);
    }
    logHandlingDuration("compilation", totalCompilationTime, std::nullopt, totalWaitTime);
    m_launchQueue.setFinished(true);
}

void MultiThreadedPlaybackTest::runRecipes(std::promise<bool>& startNotifier, std::future<bool>& canStart)
{
    canStart.wait();
    startNotifier.set_value(true);
    std::chrono::microseconds totalWaitTime      = std::chrono::microseconds::zero();
    std::chrono::microseconds totalLaunchTime    = std::chrono::microseconds::zero();
    while (true)
    {
        const auto                           waitStartTime = std::chrono::steady_clock::now();
        std::optional<ExecutionQueueElement> ctxt          = m_launchQueue.pop();
        const auto                           waitEndTime   = std::chrono::steady_clock::now();
        const auto waitDuration = std::chrono::duration_cast<std::chrono::microseconds>(waitEndTime - waitStartTime);
        totalWaitTime += waitDuration;
        if (!ctxt.has_value()) break;
        const auto launchStartTime = std::chrono::steady_clock::now();
        executeRecipe(ctxt->recipe, ctxt->params.recipe, false);
        const auto launchEndTime = std::chrono::steady_clock::now();
        const auto launchDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(launchEndTime - launchStartTime);
        totalLaunchTime += launchDuration;
        // logHandlingDuration("launch", launchDuration, ctxt->params.graph.idx, waitDuration);
        m_launchStats.waitTime.push_back(waitDuration.count());
        m_launchStats.workTime.push_back(launchDuration.count());
        m_deviceSynchronizeQueue.emplace(ctxt->params, ctxt->graph, ctxt->recipe);
        // reset allocator position
        m_deviceBufferAllocator.resetPosition();
    }
    logHandlingDuration("launch", totalLaunchTime, std::nullopt, totalWaitTime);
    m_deviceSynchronizeQueue.setFinished(true);
}

void MultiThreadedPlaybackTest::deviceSynchronize(std::promise<bool>& startNotifier, std::future<bool>& canStart)
{
    canStart.wait();
    startNotifier.set_value(true);
    std::chrono::microseconds totalWaitTime   = std::chrono::microseconds::zero();
    std::chrono::microseconds totalDeviceTime = std::chrono::microseconds::zero();
    while (true)
    {
        const auto                           waitStartTime = std::chrono::steady_clock::now();
        std::optional<ExecutionQueueElement> ctxt          = m_deviceSynchronizeQueue.pop();
        const auto                           waitEndTime   = std::chrono::steady_clock::now();
        const auto waitDuration = std::chrono::duration_cast<std::chrono::microseconds>(waitEndTime - waitStartTime);
        totalWaitTime += waitDuration;
        if (!ctxt.has_value()) break;
        const auto deviceStartTime = std::chrono::steady_clock::now();
        waitForLaunchCompletion();
        const auto deviceEndTime = std::chrono::steady_clock::now();
        const auto devicDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(deviceEndTime - deviceStartTime);
        totalDeviceTime += devicDuration;
        // logHandlingDuration("device_time", devicDuration, ctxt->params.graph.idx, waitDuration);
        m_deviceStats.waitTime.push_back(waitDuration.count());
        m_deviceStats.workTime.push_back(devicDuration.count());
        m_cleanupQueue.emplace(ctxt->params, ctxt->graph, ctxt->recipe);
    }
    logHandlingDuration("device_time", totalDeviceTime, std::nullopt, totalWaitTime);
    m_cleanupQueue.setFinished(true);
}

void MultiThreadedPlaybackTest::destroyGraphAndRecipes(std::promise<bool>& startNotifier)
{
    startNotifier.set_value(true);
    std::chrono::microseconds totalWaitTime    = std::chrono::microseconds::zero();
    std::chrono::microseconds totalCleanupTime = std::chrono::microseconds::zero();
    while (true)
    {
        const auto                           waitStartTime = std::chrono::steady_clock::now();
        std::optional<ExecutionQueueElement> ctxt          = m_cleanupQueue.pop();
        const auto                           waitEndTime   = std::chrono::steady_clock::now();
        const auto waitDuration = std::chrono::duration_cast<std::chrono::microseconds>(waitEndTime - waitStartTime);
        totalWaitTime += waitDuration;
        if (!ctxt.has_value()) break;
        const auto cleanupStartTime = std::chrono::steady_clock::now();
        cleanup(&ctxt->graph, nullptr, &ctxt->recipe, ctxt->params.graph, ctxt->params.recipe);
        const auto cleanupEndTime = std::chrono::steady_clock::now();
        const auto cleanupDuration =
            std::chrono::duration_cast<std::chrono::microseconds>(cleanupEndTime - cleanupStartTime);
        totalCleanupTime += cleanupDuration;
        // logHandlingDuration("destruction", cleanupDuration, ctxt->params.graph.idx, waitDuration);
        m_cleanupStats.waitTime.push_back(waitDuration.count());
        m_cleanupStats.workTime.push_back(cleanupDuration.count());
    }
    const auto iterEndTime = std::chrono::steady_clock::now();
    const auto totalTime   = std::chrono::duration_cast<std::chrono::microseconds>(iterEndTime - m_iterStartTime);
    m_totalHostTime        = totalTime.count();
    logHandlingDuration("destruction", totalCleanupTime, std::nullopt, totalWaitTime);
    logHandlingDuration("E2E", totalTime);
}

void MultiThreadedPlaybackTest::logHandlingDuration(std::string_view                         phase,
                                                    std::chrono::microseconds                duration,
                                                    std::optional<size_t>                    graphIdx,
                                                    std::optional<std::chrono::microseconds> waitDuration)
{
    if (m_quietMode) return;
    std::unique_lock locker(m_logLock);
    JT_LOG_INFO_NON_QUIET(
        fmt::format("{} {} phase took {} micro{}",
                    graphIdx.has_value() ? fmt::format("graph {}", *graphIdx) : "total",
                    phase,
                    duration.count(),
                    waitDuration.has_value() ? fmt::format(" waited {} micro", waitDuration->count()) : ""));
}

void MultiThreadedPlaybackTest::recordJsonStats()
{
    uint64_t statsIndex = 0;
    if (!m_statsFilePath.empty())
    {
        for (uint64_t graphIndex : m_passingGraphsIndices)
        {
            std::string graphIndexStr(std::to_string(graphIndex));
            m_stats["graphs"][graphIndexStr]["creation"].push_back(m_creationStats.workTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["compilation"].push_back(m_compilationStats.workTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["launch"].push_back(m_launchStats.workTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["device_time"].push_back(m_deviceStats.workTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["cleanup"].push_back(m_cleanupStats.workTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["compilation_wait"].push_back(m_compilationStats.waitTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["launch_wait"].push_back(m_launchStats.waitTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["device_time_wait"].push_back(m_deviceStats.waitTime[statsIndex]);
            m_stats["graphs"][graphIndexStr]["cleanup_wait"].push_back(m_cleanupStats.waitTime[statsIndex]);
            statsIndex++;
        }
        m_stats["iters"].push_back(m_totalHostTime);
    }
}

void MultiThreadedPlaybackTest::run()
{
    parseJsonGraphs();
    for (int i = 0; i < m_testIterations; i++)
    {
        JT_LOG_INFO_NON_QUIET(fmt::format("iteration {}", i));
        // future-promises ensuring threads start working in the
        // desired order, so that later pipeline phases threads
        // start earlier and we do not account spawn delay as
        // latency in handling.
        std::promise<bool> compilePhaseStartNotifier;
        std::promise<bool> launchPhaseStartNotifier;
        std::promise<bool> deviceSynchronizePhaseStartNotifier;
        std::promise<bool> cleanupPhaseStartNotifier;
        std::future<bool>  compilePhaseStarted   = compilePhaseStartNotifier.get_future();
        std::future<bool>  launchPhaseStarted            = launchPhaseStartNotifier.get_future();
        std::future<bool>  deviceSynchronizePhaseStarted = deviceSynchronizePhaseStartNotifier.get_future();
        std::future<bool>  cleanupPhaseStarted   = cleanupPhaseStartNotifier.get_future();

        // spawn the threads
        std::thread destructionThread(&MultiThreadedPlaybackTest::destroyGraphAndRecipes,
                                      this,
                                      std::ref(cleanupPhaseStartNotifier));
        std::thread deviceSynchronizeThread(&MultiThreadedPlaybackTest::deviceSynchronize,
                                            this,
                                            std::ref(deviceSynchronizePhaseStartNotifier),
                                            std::ref(cleanupPhaseStarted));
        std::thread launchThread(&MultiThreadedPlaybackTest::runRecipes,
                                 this,
                                 std::ref(launchPhaseStartNotifier),
                                 std::ref(deviceSynchronizePhaseStarted));
        std::thread compilationThread(&MultiThreadedPlaybackTest::compileGraphs,
                                      this,
                                      std::ref(compilePhaseStartNotifier),
                                      std::ref(launchPhaseStarted));
        std::thread graphCreationThread(&MultiThreadedPlaybackTest::buildGraphs, this, std::ref(compilePhaseStarted));

        // wait for threads completion
        graphCreationThread.join();
        compilationThread.join();
        launchThread.join();
        deviceSynchronizeThread.join();
        destructionThread.join();

        recordJsonStats();

        // clear finished marking on queues
        m_compilationQueue.setFinished(false);
        m_launchQueue.setFinished(false);
        m_deviceSynchronizeQueue.setFinished(false);
        m_cleanupQueue.setFinished(false);

        // clear stats
        m_creationStats.clear();
        m_compilationStats.clear();
        m_launchStats.clear();
        m_deviceStats.clear();
        m_cleanupStats.clear();
    }
    dumpStats();
}

}  // namespace json_tests
