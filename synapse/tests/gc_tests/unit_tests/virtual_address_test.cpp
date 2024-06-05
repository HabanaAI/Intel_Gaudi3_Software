#include "graph_optimizer_test.h"
#include "utils.h"

class VirtualAddressUtilsTest : public GraphOptimizerTest
{
};

TEST_F(VirtualAddressUtilsTest, virtual_address_generation_util_should_set_offset_and_mem_id)
{
    constexpr uint64_t offset = 0x5555aaaaf;  // 36b (40b available)
    constexpr uint64_t memId  = 0xaa55;       // 16b

    auto va = getVirtualAddressForMemoryID(memId, offset);
    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va));
    EXPECT_EQ(offset, maskOutMemoryID(va));
}

TEST_F(VirtualAddressUtilsTest, virtual_address_generation_util_should_retain_dma_reserved_field_when_masking)
{
    constexpr uint64_t offset   = 0x5555aaaaf;    // 36b (40b available)
    constexpr uint64_t memId    = 0xaa55;         // 16b
    constexpr uint64_t reserved = 0x33ull << 56;  // shhh.. secret reserved bits used by Gaudi's DMA

    auto va = getVirtualAddressForMemoryID(memId, offset);
    va |= reserved;

    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va));
    EXPECT_EQ(reserved | offset, maskOutMemoryID(va));
}

TEST_F(VirtualAddressUtilsTest, virtual_address_generation_util_should_allow_offset_arithmetic_ops)
{
    constexpr uint64_t offset = 0x1000;
    constexpr uint64_t memId  = 0xaa55;

    auto va = getVirtualAddressForMemoryID(memId, offset);

    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va + 0x10000));
    EXPECT_EQ(offset + 0x777, maskOutMemoryID(va + 0x777));

    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va - 0x2000));
    EXPECT_EQ(offset - 0x2000, maskOutMemoryID(va - 0x2000));
}

TEST_F(VirtualAddressUtilsTest, virtual_address_generation_util_should_allow_offset_arithmetic_ops_null_mem_id)
{
    constexpr uint64_t offset = 0x1000;
    constexpr uint64_t memId  = 0;

    auto va = getVirtualAddressForMemoryID(memId, offset);

    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va + 0x10000));
    EXPECT_EQ(offset + 0x777, maskOutMemoryID(va + 0x777));

    EXPECT_EQ(memId, getMemoryIDFromVirtualAddress(va - 0x2000));
    EXPECT_EQ(offset - 0x2000, maskOutMemoryID(va - 0x2000));
}
