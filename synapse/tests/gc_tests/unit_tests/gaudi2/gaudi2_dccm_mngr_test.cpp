#include <string>
#include <gtest/gtest.h>
#include "eng_arc_dccm_mngr.h"
#include "graph_optimizer_test.h"

class Gaudi2EngArcDccmTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<uint64_t, unsigned, unsigned, unsigned>>
{
};

TEST_P(Gaudi2EngArcDccmTest, allocate_dccm_addr_slot_test)
{
    uint64_t startAddrOffset  = std::get<0>(GetParam());
    unsigned totalNumSlots    = std::get<1>(GetParam());
    unsigned slotSize         = std::get<2>(GetParam());
    unsigned allocationCycles = std::get<3>(GetParam());

    EngArcDccmMngr dccmMngr(startAddrOffset /*startAddrOffset*/,
                            totalNumSlots /*totalNumSlots*/,
                            slotSize /*slotSize*/);

    for (int cycle = 0; cycle < allocationCycles; cycle++)
    {
        for (int slot = 0; slot < totalNumSlots; slot++)
        {
            uint64_t allocatedAddr         = dccmMngr.getSlotAllocation();
            uint64_t expectedAllocatedAddr = startAddrOffset + (slot * slotSize);
            ASSERT_EQ(allocatedAddr, expectedAllocatedAddr)
                << "Allocated addr is: " << allocatedAddr << " Expected addr is:" << expectedAllocatedAddr;
            ASSERT_EQ(slot, dccmMngr.addrToSlot(allocatedAddr))
                << "Slot id is: " << slot << " Expected slot id is:" << dccmMngr.addrToSlot(allocatedAddr);
            ASSERT_EQ(allocatedAddr, dccmMngr.slotToAddr(slot))
                << "Addr for slot id " << slot << " is: " << allocatedAddr
                << " Expected addr is:" << dccmMngr.slotToAddr(slot);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(_,
                         Gaudi2EngArcDccmTest,
                         /*std::make_tuple(startAddrOffset, totalNumSlots, slotSize, allocationCycles)*/
                         ::testing::Values(std::make_tuple(0x1000100, 10, 128, 3),
                                           std::make_tuple(0x12345, 64, 12, 5),
                                           std::make_tuple(0x0, 64, 256, 10),
                                           std::make_tuple(0x10020020, 64, 32, 2),
                                           std::make_tuple(0x0, 64, 256, 100)));