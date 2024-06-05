#include <gtest/gtest.h>
#include "infra/containers/slot_map_alloc.hpp"
#include "infra/containers/slot_map_precache.hpp"
#include <thread>
#include <condition_variable>
#include <mutex>
#include "perf_test.hpp"
#include <atomic>

template<class TDerivedNode>
struct LockFreeIntrusiveStackNodeBase : IntrusiveStackNodeBase<TDerivedNode>
{
    uint16_t abaCounter {0};
};
/**
 * concurrent intrusive stack
 * does not allocate/deallocate any memory
 * it's not lock-free
 * full lock-free implementation requires double CAS that gcc has partial support
 * pop suffers from ABA issue without lock or double cas
 * @tparam TNode node type that must be publicly derived from IntrusiveStackNodeBase
 */
template<class TNode>
class LockFreeIntrusiveStack
{
public:
    LockFreeIntrusiveStack() = default;

    /**
     * push a new node into stack
     * complexity : O(1). lock-free
     * @param newNode
     */
    void push(TNode* newNode)
    {
        newNode->next = m_head.load(std::memory_order_relaxed);
        newNode->abaCounter++;
        TNode* newNodeWithAbaCounter = (TNode*)((uint64_t)newNode | (uint64_t(newNode->abaCounter) << 48));

        while (!m_head.compare_exchange_weak(newNode->next,
                                             newNodeWithAbaCounter,
                                             std::memory_order_release,
                                             std::memory_order_relaxed))
        {
            usleep(250);
        };
    }
    /**
     * pop a node from stack
     * complexity : O(1). lock-free
     * @param newNode pointer to a popped node or nullptr if stack is empty
     */
    TNode* pop()
    {
        // std::lock_guard<std::mutex> lck(m_mtx);

        TNode* nodeToPop = m_head.load(std::memory_order_relaxed);
        while (nodeToPop && !m_head.compare_exchange_strong(nodeToPop,
                                                            ((TNode*)((uint64_t)nodeToPop & 0xFFFFFFFFFFFF))->next,
                                                            std::memory_order_acquire,
                                                            std::memory_order_relaxed))
        {
            usleep(250);
        };
        return (TNode*)((uint64_t)nodeToPop & 0xFFFFFFFFFFFFul);
    }

    LockFreeIntrusiveStack(LockFreeIntrusiveStack const&) = delete;
    LockFreeIntrusiveStack(LockFreeIntrusiveStack&&)      = delete;

private:
    static_assert(std::is_base_of<LockFreeIntrusiveStackNodeBase<TNode>, TNode>::value,
                  "TNode must be publicly derived from IntrusiveStackNodeBase");
    std::atomic<TNode*> m_head {nullptr};
};

struct Node_ : LockFreeIntrusiveStackNodeBase<Node_>
{
    int v;
};

OperationFullMeasurementResults testLockFreeIntrusiveStack(TestParams params)
{
    static std::vector<Node_> nodes(1024);
    for (unsigned i = 0; i < nodes.size(); ++i)
    {
        nodes[i].v = i;
    }
    static LockFreeIntrusiveStack<Node_> stack;

    struct HandleAligned
    {
        Node_* node {nullptr};
        bool   firtsTime {true};
        char   buf[56];
    };

    static HandleAligned    handles[1000] {};
    static std::atomic<int> eraseFailed {0};
    static std::atomic<int> insertFailed {0};
    auto                    res = measure(
        [](unsigned thread_id) {
            if (!handles[thread_id].firtsTime)
            {
                handles[thread_id].node = stack.pop();
                if (!handles[thread_id].node)
                {
                    std::cerr << "bad~\n";
                }
            }

            handles[thread_id].firtsTime = false;
            stack.push(handles[thread_id].node ? handles[thread_id].node : &nodes[thread_id]);
        },
        "LockFreeIntrusiveStack stack insert/delete",
        params,
        []() {
            for (auto& h : handles)
            {
                h.firtsTime = true;
                h.node      = nullptr;
            }
            while (stack.pop())
                ;
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

OperationFullMeasurementResults testIntrusiveStackFullLock(TestParams params)
{
    static std::vector<Node_> nodes(1024);
    for (unsigned i = 0; i < nodes.size(); ++i)
    {
        nodes[i].v = i;
    }
    static ConcurrentIntrusiveStack<Node_> stack;

    struct HandleAligned
    {
        Node_* node {nullptr};
        bool   firtsTime {true};
        char   buf[56];
    };

    static HandleAligned    handles[1000] {};
    static std::atomic<int> eraseFailed {0};
    static std::atomic<int> insertFailed {0};
    auto                    res = measure(
        [](unsigned thread_id) {
            if (!handles[thread_id].firtsTime)
            {
                handles[thread_id].node = stack.pop();
                if (!handles[thread_id].node)
                {
                    std::cerr << "bad~\n";
                }
            }

            handles[thread_id].firtsTime = false;
            stack.push(handles[thread_id].node ? handles[thread_id].node : &nodes[thread_id]);
        },
        "instrusiveFullLock stack insert/delete",
        params,
        []() {
            for (auto& h : handles)
            {
                h.firtsTime = true;
                h.node      = nullptr;
            }
            while (stack.pop())
                ;
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}

#include <deque>
OperationFullMeasurementResults testDequeu(TestParams params)
{
    static std::deque<unsigned> queue;
    struct HandleAligned
    {
        unsigned handle {0};
        bool     firtsTime {true};
        char     buf[56];
    };

    static HandleAligned    handles[1000] {};
    static std::atomic<int> eraseFailed {0};
    static std::atomic<int> insertFailed {0};
    static std::mutex       mtx;
    auto                    res = measure(
        [](unsigned thread_id) {
            if (!handles[thread_id].firtsTime)
            {
                mtx.lock();
                handles[thread_id].handle = queue.back();
                queue.pop_back();
                mtx.unlock();
            }

            handles[thread_id].firtsTime = false;
            mtx.lock();
            queue.push_back(123);
            mtx.unlock();
        },
        "std::dequeu insert/delete",
        params,
        []() {
            for (auto& h : handles)
                h.firtsTime = true;
            queue.clear();
            if (eraseFailed || insertFailed)
            {
                std::cerr << "erase: " << eraseFailed.load() << " insert: " << insertFailed << "\n";
            }
            eraseFailed  = 0;
            insertFailed = 0;
        });

    return res;
}
#if ENABLE_PERFORMANCE_TEST
TEST(IntrusiveStackPerfTest, basic_ops)
{
    TestParams params;
    params.maxNbThreads            = 10;
    params.minNbThreads            = 1;
    params.nbTests                 = 200;
    params.internalLoopInterations = 2000;
    params.interations_per_sleep   = 0;
    PrintTestResults("Intrusive stack performance check",
                     {testLockFreeIntrusiveStack(params),
                      testIntrusiveStackFullLock(params),
                      testDequeu(params),
                      testLockFreeIntrusiveStack(params)});
}
#endif