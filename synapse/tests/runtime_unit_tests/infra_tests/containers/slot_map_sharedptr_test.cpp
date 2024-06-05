#include <gtest/gtest.h>
#include "infra/containers/slot_map_alloc.hpp"
#include "run_test_mt.hpp"

struct Base
{
    int a;
    Base(int a) : a(a) {}
    virtual ~Base() = default;
};
struct Derived : Base
{
    int b;
    Derived(int a, int b) : Base(a), b(b) {}
};
static void TestSharedPtr(ConcurrentSlotMapAlloc<Derived>& itemsMap)
{
    auto newItem = itemsMap.insert(0, 1, 2);
    ASSERT_NE(newItem.second, nullptr);
    ASSERT_EQ(newItem.second.use_count(), 2);
    SlotMapItemSptr<Derived> derivedItem = newItem.second;
    ASSERT_EQ(derivedItem.use_count(), 3);
    SlotMapItemSptr<Base> baseItem = derivedItem;
    ASSERT_EQ(derivedItem.use_count(), 4);
    baseItem = derivedItem;
    ASSERT_EQ(derivedItem.use_count(), 4);
    SlotMapItemSptr<Base> baseItem2 = itemsMap[newItem.first];
    ASSERT_EQ(derivedItem.use_count(), 5);
    ASSERT_NE(baseItem2, nullptr);
    ASSERT_EQ(&baseItem->a, &baseItem2->a);
    SlotMapItemSptr<Derived> derivedItem2 = SlotMapItemDynamicCast<Derived>(baseItem2);
    ASSERT_EQ(derivedItem.use_count(), 6);
    ASSERT_EQ(derivedItem2.use_count(), 6);
    ASSERT_EQ(baseItem2.use_count(), 6);
    ASSERT_EQ(&baseItem2->a, &derivedItem2->a);
    baseItem.reset();
    derivedItem.reset();
    derivedItem2.reset();
    ASSERT_EQ(newItem.second.use_count(), 3);
    baseItem2.reset();
    newItem.second.reset();
    itemsMap.erase(newItem.first);
}
TEST(SlotMapSharedPtrTest, sharedptrCasts)
{
    ConcurrentSlotMapAlloc<Derived> itemsMap;
    TestSharedPtr(itemsMap);
}
TEST(SlotMapSharedPtrTest, sharedptrCastsMultiThread)
{
    ConcurrentSlotMapAlloc<Derived> itemsMap;
    runTestsMT([&](unsigned, unsigned) { TestSharedPtr(itemsMap); }, 1000, 6);
}
