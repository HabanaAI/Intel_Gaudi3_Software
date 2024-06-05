#pragma once

// This class will manage the engine arc DCCM slots allocation
class EngArcDccmMngr
{
public:
    EngArcDccmMngr(uint64_t baseAddr, unsigned totalNumSlots, unsigned slotSize);
    EngArcDccmMngr()  {}
    ~EngArcDccmMngr() {}

    uint64_t getSlotAllocation();
    unsigned addrToSlot(uint64_t slotAddr);
    uint64_t slotToAddr(unsigned slotId);
    void     print() const;

private:
    uint64_t m_baseAddr      = 0;
    unsigned m_totalNumSlots = 0;
    unsigned m_slotSize      = 0;
    unsigned m_nextSlotId    = 0;
};