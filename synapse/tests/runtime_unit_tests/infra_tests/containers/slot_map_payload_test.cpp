#include <gtest/gtest.h>
#include "infra/containers/slot_map.hpp"
#include <thread>
#include <condition_variable>
#include <mutex>

TEST(SlotMapPayloadTest, payloadCheck)
{
    int                    value = 10;
    ConcurrentSlotMap<int> itemsMap(100);
    auto                   handle = itemsMap.insert(&value, 0).first;
    ASSERT_EQ(getSMHandlePayload(handle), 0);
    handle = itemsMap.insert(&value, 1).first;
    ASSERT_EQ(getSMHandlePayload(handle), 1);
    handle = itemsMap.insert(&value, 2).first;
    ASSERT_EQ(getSMHandlePayload(handle), 2);
    handle = itemsMap.insert(&value, 5).first;
    ASSERT_EQ(getSMHandlePayload(handle), 5);
}