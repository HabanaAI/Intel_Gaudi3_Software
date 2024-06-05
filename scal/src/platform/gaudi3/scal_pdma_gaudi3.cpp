#include "scal_data_gaudi3.h"
#include "scal_utilities.h"
#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"

Scal_Gaudi3::PDMAStream_Gaudi3::PDMAStream_Gaudi3()
: m_chBlock(nullptr)
{
}

void Scal_Gaudi3::PDMAStream_Gaudi3::init(const unsigned qid, const int fd)
{
    uint64_t chDevAddr;
    if (!pdmaId2baseAddrA(qid, chDevAddr))
    {
        LOG_ERR(SCAL,"{}, pdmaId2baseAddrA() failed for core id {}", __FUNCTION__, qid);
        assert(0);
        return;
    }

    m_chBlock = (gaudi3::block_pdma_ch_a *)mapLbwMemory(fd, chDevAddr, c_pdma_ch_block_size, m_allocatedSize);
    if (!m_chBlock)
    {
        LOG_ERR(SCAL,"{}, pdmaId2baseAddrA() failed for core id {}", __FUNCTION__, qid);
        assert(0);
        return;
    }
    m_qid = qid;
}

Scal_Gaudi3::PDMAStream_Gaudi3::~PDMAStream_Gaudi3()
{
    if (m_chBlock)
    {
        unmapLbwMemory(m_chBlock, m_allocatedSize);
    }
    if (m_asyncThreadActive)
    {
        LOG_INFO(SCAL,"{}, waiting for m_asyncThreadFuture to finish", __FUNCTION__);
        m_asyncThreadFuture.wait();
    }
}

int Scal_Gaudi3::PDMAStream_Gaudi3::setBuffer(const uint64_t deviceAddr, const unsigned size)
{
    if (!m_chBlock)
    {
        LOG_ERR(SCAL,"{}: not initalized", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    if (!deviceAddr)
    {
        LOG_ERR(SCAL,"{}: null buffer", __FUNCTION__);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    if (!size || (size & (size - 1)) || (size > c_max_buffer_size))
    {
        LOG_ERR(SCAL,"{}: illegal buffer size: {}", __FUNCTION__, size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    writeLbwReg(&m_chBlock->pqm_ch.msq_ci._raw, 0);
    writeLbwReg(&m_chBlock->pqm_ch.desc_submit_fifo_cfg._raw, 1);
    writeLbwReg(&m_chBlock->pqm_ch.msq_base_h._raw, upper_32_bits(deviceAddr));
    writeLbwReg(&m_chBlock->pqm_ch.msq_base_l._raw, lower_32_bits(deviceAddr));
    writeLbwReg(&m_chBlock->pqm_ch.msq_size._raw, size - 1);
    m_buffSize = size;

    return SCAL_SUCCESS;
}

void Scal_Gaudi3::PDMAStream_Gaudi3::handlePiOverflowAsync(uint32_t hwPiRemainder)
{
    while (true)
    {
        if (readLbwReg(&m_chBlock->pqm_ch.msq_ci._raw) == m_hwPi)
        {
            std::unique_lock<std::mutex> lock(m_wraparoundMutex);
            uint32_t hwPi = m_localPi + m_hwPiOffset;
            writeLbwReg(&m_chBlock->pqm_ch.msq_ci._raw, hwPiRemainder);
            writeLbwReg(&m_chBlock->pqm_ch.msq_pi._raw, hwPi);
            LOG_WARN(SCAL,"{}, (ASYNC) Handling PI overflow on core id {}. new ci=0x{:x} new pi=0x{:x} m_hwPiOffset=0x{:x}", __FUNCTION__,
                        m_qid, hwPiRemainder, hwPi, m_hwPiOffset);
            m_hwPi = hwPi;
            m_asyncThreadActive = false;
            lock.unlock();
            return;
        }
    }
}

uint32_t Scal_Gaudi3::PDMAStream_Gaudi3::handlePiOverflow(uint32_t hwPi)
{
    m_wraparound = true;
    uint32_t hwPiRemainder = m_hwPi & (m_buffSize - 1);
    if (hwPiRemainder)
    {
        m_hwPiOffset += m_buffSize;
        hwPi += m_buffSize;
    }
    // short busy loop - try to reset CI before using async
    constexpr unsigned c_max_wraparound_retries = 100;
    for (unsigned i = 0; i < c_max_wraparound_retries; i++)
    {
        if (readLbwReg(&m_chBlock->pqm_ch.msq_ci._raw) == m_hwPi)
        {
            writeLbwReg(&m_chBlock->pqm_ch.msq_ci._raw, hwPiRemainder);
            LOG_WARN(SCAL,"{}, (BUSYLOOP) Handling PI overflow on core id {}. new ci=0x{:x} curr pi=0x{:x} i={}  m_hwPiOffset=0x{:x}", __FUNCTION__,
                        m_qid, hwPiRemainder, hwPi, i , m_hwPiOffset);
            m_wraparound = false;
            break;
        }
    }
    if (m_wraparound)
    {
        m_asyncThreadActive = true;
        // launch thread
        LOG_WARN(SCAL,"{}, (ASYNC) launched thread handling pi (0x{:x}) overflow on core id {}.", __FUNCTION__, m_hwPi, m_qid);
        m_asyncThreadFuture = std::async(std::launch::async, &Scal_Gaudi3::PDMAStream_Gaudi3::handlePiOverflowAsync, this, hwPiRemainder);
    }
    return hwPi;
}

int Scal_Gaudi3::PDMAStream_Gaudi3::setPi(const unsigned pi)
{
    if (likely(!m_wraparound))
    {
        m_localPi = pi;
        uint32_t hwPi = pi + m_hwPiOffset;
        if (unlikely(hwPi < m_hwPi)) // PI wraparound
        {
            hwPi = handlePiOverflow(hwPi);
        }
        // write PI in case of no wraparound at all, or in case that wrap around was fixed with busy loop
        if (!m_wraparound)
        {
            writeLbwReg(&m_chBlock->pqm_ch.msq_pi._raw, hwPi);
            m_hwPi = hwPi;
        }
    }
    else
    {
        std::unique_lock<std::mutex> lock(m_wraparoundMutex);
        if (!m_asyncThreadActive)
        {
            // release lock
            m_wraparound = false;
            lock.unlock();
            LOG_WARN(SCAL,"{}, async overflow handler thread done. set new pi 0x{:x} on core id {}.", __FUNCTION__, pi, m_qid);
            return setPi(pi);
        }
        else
        {
            // set local pi with lock
            m_localPi = pi;
            lock.unlock();
        }
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::PDMAStream_Gaudi3::configurePdmaPtrsMode()
{
    writeLbwReg(&m_chBlock->pqm_ch.desc_submit_fifo_cfg._raw, 4);
    return SCAL_SUCCESS;
}