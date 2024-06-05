#include <gtest/gtest.h>
#include "infra/containers/intrusive_stack.hpp"
#include "run_test_mt.hpp"
TEST(IntrusiveStackTest, checkPushPopSingleThread)
{
    struct Node : IntrusiveStackNodeBase<Node>
    {
        int val;
        Node(int val) : val(val) {}
    };
    std::vector<Node> nodes;
    const unsigned    nbItems = 100;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        nodes.emplace_back(i);
    }

    ConcurrentIntrusiveStack<Node> stack;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        stack.push(&nodes[i]);
    }
    for (unsigned i = 0; i < nbItems; ++i)
    {
        Node* pNode = stack.pop();
        ASSERT_NE(pNode, nullptr);
        ASSERT_EQ(pNode->val, nbItems - i - 1);
    }
    Node* pNode = stack.pop();
    ASSERT_EQ(pNode, nullptr);
}

TEST(IntrusiveStackTest, checkPushMultiThread)
{
    struct Node : IntrusiveStackNodeBase<Node>
    {
        int val;
        Node(int val) : val(val) {}
    };
    std::vector<Node> nodes;

    const unsigned nbItems = 100000;

    std::array<std::atomic<int>, nbItems> poppedItems;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        nodes.emplace_back(i);
        poppedItems[i].store(0);
    }

    ConcurrentIntrusiveStack<Node> stack;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        stack.push(&nodes[i]);
    }

    const unsigned   nbThreads = 6;
    std::atomic<int> totalEmptyItemsCount {0};

    runTestsMT(
        [&](unsigned, unsigned) {
            auto node = stack.pop();
            if (node)
            {
                ++poppedItems.at(node->val);
            }
            else
            {
                ++totalEmptyItemsCount;
            }
        },
        (nbItems + nbThreads - 1) / nbThreads,
        nbThreads);
    ASSERT_EQ(totalEmptyItemsCount, ((nbItems + nbThreads - 1) / nbThreads) * nbThreads - nbItems);

    for (unsigned i = 0; i < nbItems; ++i)
    {
        ASSERT_EQ(poppedItems[i], 1);
    }
}

TEST(IntrusiveStackTest, checkPopMultiThread)
{
    struct Node : IntrusiveStackNodeBase<Node>
    {
        int val;
        Node(int val) : val(val) {}
    };
    std::vector<Node> nodes;

    const unsigned nbItems = 100000;

    static std::array<std::atomic<int>, nbItems> poppedItems;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        nodes.emplace_back(i);
        poppedItems[i].store(0);
    }

    static ConcurrentIntrusiveStack<Node> stack;
    for (unsigned i = 0; i < nbItems; ++i)
    {
        stack.push(&nodes[i]);
    }
    const unsigned   nbThreads = 6;
    std::atomic<int> totalEmptyItemsCount {0};

    runTestsMT(
        [&](unsigned, unsigned) {
            auto node = stack.pop();
            if (node)
            {
                ++poppedItems.at(node->val);
            }
            else
            {
                ++totalEmptyItemsCount;
            }
        },
        (nbItems + nbThreads - 1) / nbThreads,
        nbThreads);

    ASSERT_EQ(totalEmptyItemsCount, ((nbItems + nbThreads - 1) / nbThreads) * nbThreads - nbItems);

    for (unsigned i = 0; i < nbItems; ++i)
    {
        ASSERT_EQ(poppedItems[i], 1);
    }
}

TEST(IntrusiveStackTest, checkSimplePushPopMultiThread)
{
    struct Node : IntrusiveStackNodeBase<Node>
    {
        int val;
        Node(int val) : val(val) {}
    };
    std::vector<Node> nodes;

    const unsigned nbThreads = 10;

    std::set<int> vals;
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        nodes.push_back(i);
        vals.insert(i);
    }
    ConcurrentIntrusiveStack<Node> stack;
    for (unsigned i = 0; i < nbThreads; ++i)
    {
        stack.push(&nodes[i]);
    }
    std::vector<Node*> items(nbThreads, nullptr);
    const unsigned     cnt = 100000;
    runTestsMT(
        [&](unsigned thread_id, unsigned i) {
            if (items[thread_id]) stack.push(items[thread_id]);
            if (i != cnt - 1)
            {
                items[thread_id] = stack.pop();
            }
            ASSERT_EQ(vals.count(items[thread_id]->val), 1);
        },
        cnt,
        nbThreads);

    for (unsigned i = 0; i < nbThreads; ++i)
    {
        auto node = stack.pop();
        ASSERT_NE(node, nullptr);
        ASSERT_EQ(vals.count(node->val), 1);
        vals.erase(node->val);
    }
    ASSERT_EQ(stack.pop(), nullptr);
}
