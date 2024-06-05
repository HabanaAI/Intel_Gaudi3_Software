#pragma once

#include <cstdint>
#include <array>


/*********************************************************************************
 * This class is used as a base class to allocate memory for recipe launcher in the
 * glbl hbm and arc hbm
 * The class assumes the caller has a lock to allow for only one thread to call it
 * at the same time.
 * The allocator is not the standard one: it returns the memory to be used
 * and what longSo to wait for before this memory may be used
 * The class gets a chunkSize from the caller during (init function). It assumes there are
 * NUM_BUFF of this chunk size
 * Usage:
 * init(chunkSize):   user should set the chunks sizes
 * alloc(size):       the class return an offset to the memory and the longSo to wait for
 *                    before this memory can be used. Returned memory must by consecutive
 * setLongSo(longSO): The user should set the longSo that is being used for the lastAllocation
 *                    Note: this is not done as part of the alloc() as the user doesn't have
 *                    the longSo during the alloc() call
 *
 *********************************************************************************/
class NBuffAllocator
{
public:
    static constexpr uint16_t NUM_BUFF = 16;

protected:
    struct AllocRtn
    {
        uint64_t offset;
        uint64_t longSo;
    };

    NBuffAllocator() {}
    virtual ~NBuffAllocator() {}

    void init(uint64_t chunkSize) { m_chunkSize = chunkSize; }
    [[nodiscard]] AllocRtn alloc(uint64_t size);

    void     setLongSo(uint64_t longSo);
    void     dumpLongSo(); // for debug only
    uint64_t getLastLongSo() { return m_lastLongSo; } // this is used for read after download in RecipeLauncher

private:
    enum class Phase { WAIT_FOR_ALLOC, WAIT_FOR_LONG_SO_SET };

    int16_t  m_firstFree      = 0;
    int16_t  m_prevAllocStart = NUM_BUFF - 1;
    Phase    m_phase          = Phase::WAIT_FOR_ALLOC;
    uint64_t m_chunkSize      = 0;
    uint64_t m_lastLongSo     = 0;

    std::array<uint64_t, NUM_BUFF> m_longSo {};
};
