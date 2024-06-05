#include "global_statistics.hpp"
#include "nbuff_allocator.hpp"
#include "log_manager.h"

/*
 ***************************************************************************************************
 *   @brief alloc() - Alloc consecutive memory (in chunks). It verifies the size (we throw as the size
 *   should be checked as part of the recipe verifier). It then checks the phase (to make sure the user
 *   (didn't forget to set the longSo).
 *   Basically, we keep allocating in a cyclic manner, going to start if we get to the end. Then we go over
 *   all the allocated chunks and return the max longSo
 *   Note, it is tempting to use the longSo of the last allocate chunk instead of finding the max, it doesn't
 *   work because we have the option to allocate more than NUM_BUFF/2
 *
 *   @param  size (the allocation requested size)
 *   @return AllocRtn (offset: in the memory to be used, longSo: the longSo to wait for before using this memory)
 *
 **************************************************************************************************
*/
NBuffAllocator::AllocRtn NBuffAllocator::alloc(uint64_t size)
{
    const uint64_t numBuffs = (size + m_chunkSize - 1) / m_chunkSize;

    if (numBuffs > NUM_BUFF)
    {
        throw std::runtime_error("recipe too big for hbm memory");  // using throw so can run tessts in debug mode
    }

    if (m_phase != Phase::WAIT_FOR_ALLOC)
    {
        throw std::runtime_error("NBuffAllocator phase is not ready");  // using throw so can run tessts in debug mode
    }
    m_phase = Phase::WAIT_FOR_LONG_SO_SET;

    AllocRtn rtn;
    const uint16_t currFirstFree = m_firstFree;
    uint16_t       end           = (currFirstFree + numBuffs - 1) % NUM_BUFF;
    uint16_t       firstBuff;
    if (numBuffs > (NUM_BUFF / 2)) // big allocation
    {
        if (end > currFirstFree) // we didn't have a wrap around
        {
            firstBuff = currFirstFree;
            m_firstFree   = (currFirstFree + numBuffs) % NUM_BUFF;
        }
        else // we had wrap around, start from 0
        {
            firstBuff = 0; // allocate from the start
            m_firstFree   = numBuffs % NUM_BUFF; // next one is after this "big" allocation
        }
    }
    else // if (numBuff > NUM_BUFF/2) else
    {
        // make sure we don't cross double buffer boundry, if so, start on the beginning of the double buffer
        uint8_t startHalf = currFirstFree / (NUM_BUFF / 2);
        uint8_t endHalf   = end           / (NUM_BUFF / 2);

        if (startHalf == endHalf) // all is good
        {
            firstBuff   = currFirstFree;
            m_firstFree = (currFirstFree + numBuffs) % NUM_BUFF;
        }
        else
        {
            firstBuff    = endHalf * NUM_BUFF / 2;
            m_firstFree  = (firstBuff + numBuffs) % NUM_BUFF; // we need the %, we can have a wrap around to 0
        }
    }

    m_prevAllocStart = firstBuff; // keep the first buff allocated for later use in setLongSo

    // find the max longSo of all the allocated buffers
    rtn.longSo = m_longSo[firstBuff];
    rtn.offset = firstBuff * m_chunkSize;

    uint16_t firstChunk = rtn.offset / m_chunkSize;

    for (uint16_t i = numBuffs - 1; i > 0; i--)
    {
        rtn.longSo = std::max(rtn.longSo, m_longSo[firstChunk + i]);
    }

    if ((rtn.longSo == m_lastLongSo) && (m_lastLongSo != 0))
    {
        STAT_GLBL_COLLECT(1, scalHbmSingleBuff);
    }
    return rtn;
}

/*
 ***************************************************************************************************
 *   @brief setLongSo() - save the longSo to the last allocated chunks
 *
 *   @param  longSo: the longSo
 *   @return None
 *
 **************************************************************************************************
*/
void NBuffAllocator::setLongSo(uint64_t longSo)
{
    if (m_phase != Phase::WAIT_FOR_LONG_SO_SET)
    {
        throw std::runtime_error("NBuffAllocator phase is not in_progress");  // using throw so can run tessts in debug mode
    }
    m_phase = Phase::WAIT_FOR_ALLOC;

    uint16_t end = (m_firstFree == 0) ? NUM_BUFF : m_firstFree;
    for (uint16_t i = m_prevAllocStart; i < end; i++)
    {
        m_longSo[i] = longSo;
    }
    m_lastLongSo = longSo;
}

/*
 ***************************************************************************************************
 *   @brief dumpLongSo() - debug only, prints the array of the longSo
 *
 *   @param  None
 *   @return None
 *
 **************************************************************************************************
*/
void NBuffAllocator::dumpLongSo()
{
    printf("--------------------\n");
    for (uint16_t i = 0; i < NUM_BUFF; i++)
    {
        printf("%2d) %ld\n", i, m_longSo[i]);
    }
}
