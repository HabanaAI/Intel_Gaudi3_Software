#include <atomic>
#include "infra/defs.h"
#include "log_manager.h"
#include "eng_arc_dccm_mngr.h"

EngArcDccmMngr::EngArcDccmMngr(uint64_t baseAddr, unsigned totalNumSlots, unsigned slotSize)
: m_baseAddr(baseAddr), m_totalNumSlots(totalNumSlots), m_slotSize(slotSize), m_nextSlotId(0)
{
}

uint64_t EngArcDccmMngr::getSlotAllocation()
{
    uint64_t allocatedAddr = m_baseAddr + m_nextSlotId * m_slotSize;
    m_nextSlotId           = (m_nextSlotId + 1) % m_totalNumSlots;
    return allocatedAddr;
}

unsigned EngArcDccmMngr::addrToSlot(uint64_t slotAddr)
{
    return (slotAddr - m_baseAddr) / m_slotSize;
}

uint64_t EngArcDccmMngr::slotToAddr(unsigned slotId)
{
    return m_baseAddr + (slotId % m_totalNumSlots) * m_slotSize;
}

void EngArcDccmMngr::print() const
{
    LOG_DEBUG(GC_ARC,
              "EngArcDccmMngr startAddrOffset={}, totalNumSlots={}, slotSize={}, nextSlotId={}",
              m_baseAddr,
              m_totalNumSlots,
              m_slotSize,
              m_nextSlotId);
}