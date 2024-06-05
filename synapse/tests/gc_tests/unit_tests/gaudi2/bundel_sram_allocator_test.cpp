#include "bundle_sram_allocator.h"
#include "layered_brain_test.h"
#include "synapse_common_types.h"
#include "gtest/gtest.h"
#include <cstdint>

class POCBundleSRAMAllocatorTest : public LayeredBrainTest
{
protected:
    void SetUp() override
    {
        LayeredBrainTest::SetUp();
        // set fragmentation factor to be 1 so the test design and memory size calculation will be easier
        setGlobalConfForTest(GCFG_FRAGMENTATION_COMPENSATION_FACTOR, "1.0");
    }

    std::vector<TensorPtr> setup(std::vector<TSize> sizes)
    {
        unsigned               dim = 1;
        std::vector<TensorPtr> slices;
        for (auto size : sizes)
        {
            TSize inSize[] = {size};  // because data type is syn_type_uint8 (one byte)
            slices.push_back(TensorPtr(new Tensor(dim, inSize, syn_type_uint8)));
        }
        return slices;
    }

    POCBundleSRAMAllocator createAllocator(const uint64_t inputBudgetSize)
    {
        POCBundleSRAMAllocator allocator {inputBudgetSize};
        return allocator;
    }
};

TEST_F(POCBundleSRAMAllocatorTest, sram_allocator_should_allocate_until_havent_enough_budget)
{
    uint64_t               budget      = 100;
    POCBundleSRAMAllocator m_allocator = createAllocator(budget);
    std::vector<TensorPtr> slices      = setup({10, 30, 50, 10, 70});
    for (int i = 0; i < 4; i++)
    {
        EXPECT_TRUE(m_allocator.allocate(slices[i]));
    }
    EXPECT_FALSE(m_allocator.allocate(slices[4]));
}

TEST_F(POCBundleSRAMAllocatorTest, sram_allocator_should_allocate_and_than_free_each_slice)
{
    uint64_t               budget      = 100;
    POCBundleSRAMAllocator m_allocator = createAllocator(budget);
    std::vector<TensorPtr> slices      = setup({10, 30, 50, 10, 70});
    for (int i = 0; i < 5; i++)
    {
        EXPECT_TRUE(m_allocator.allocate(slices[i]));
        m_allocator.free(slices[i]);
    }
}

TEST_F(POCBundleSRAMAllocatorTest, sram_allocator_should_allocate_max_budget_capacity_slices)
{
    uint64_t               budget      = 100;
    POCBundleSRAMAllocator m_allocator = POCBundleSRAMAllocatorTest::createAllocator(budget);
    std::vector<TensorPtr> slices      = setup({100, 30, 120});
    EXPECT_TRUE(m_allocator.allocate(slices[0]));
    EXPECT_FALSE(m_allocator.allocate(slices[1]));
    m_allocator.free(slices[0]);
    EXPECT_TRUE(m_allocator.allocate(slices[1]));
    m_allocator.free(slices[1]);
    EXPECT_FALSE(m_allocator.allocate(slices[2]));
}