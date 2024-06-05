#include "scal_gaudi3.h"

#include "scal_data_gaudi3.h"
#include "scal_utilities.h"

#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"
#include "gaudi3/asic_reg_structs/pdma_ch_b_regs.h"

Scal_Gaudi3::DirectModePDMAStream::DirectModePDMAStream(std::string&   streamName,
                                                        unsigned       qmanId,
                                                        unsigned       fenceCounterIndex,
                                                        uint64_t       fenceCounterAddress,
                                                        unsigned       priority,
                                                        unsigned       id)
:   PDMAStream_Gaudi3(),
    m_name(streamName),
    m_qmanId(qmanId),
    m_priority(priority),
    m_id(id),
    m_fenceCounterIndex(fenceCounterIndex),
    m_fenceCounterAddr(fenceCounterAddress)
{
}

int Scal_Gaudi3::DirectModePDMAStream::initChannelCQ(uint64_t countersMmuAddress,
                                                     uint64_t isrRegister,
                                                     unsigned isrIndex)
{
    // Reset the Fence-Counter
    writeLbwReg(&m_chBlock->pqm_ch.fence_cnt[m_fenceCounterIndex]._raw, 0);

    if (m_isCqInitialized)
    {
        LOG_ERR(SCAL,"{}: Re-initialization for qmanId {}", __FUNCTION__, m_qmanId);
        assert(0);
        return SCAL_FAILURE;
    }

    // Set CQ's Counter buffer
    writeLbwReg(&m_chBlock->ecmpltn_q_base_lo._raw, lower_32_bits(countersMmuAddress));
    writeLbwReg(&m_chBlock->ecmpltn_q_base_hi._raw, upper_32_bits(countersMmuAddress));
    writeLbwReg(&m_chBlock->ecmpltn_q_base_size._raw, c_cq_size);

    // Set CQ's ISR
    bool isIsrInitRequired = (isrIndex != scal_illegal_index);
    if (isIsrInitRequired)
    {
        writeLbwReg(&m_chBlock->ecmpltn_q_lbw_addr_l._raw, lower_32_bits(isrRegister));
        writeLbwReg(&m_chBlock->ecmpltn_q_lbw_addr_h._raw, upper_32_bits(isrRegister));
        writeLbwReg(&m_chBlock->ecmpltn_q_lbw_payld._raw, isrIndex);
    }

    // Set CQ Mode
    gaudi3::pdma_ch_a::reg_ecmpltn_q_ch_cfg channelConfigMode;
    channelConfigMode._raw         = 0;
    channelConfigMode.enable       = true;
    channelConfigMode.compltn_mode = true; // Scheduler mode (AKA - user, not driver)
    channelConfigMode.lbw_wr_en    = isIsrInitRequired;
    writeLbwReg(&m_chBlock->ecmpltn_q_ch_cfg._raw, channelConfigMode._raw);

    m_isCqInitialized = true;
    
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::DirectModePDMAStream::submit(const unsigned pi, const unsigned submission_alignment)
{
    //  Todo: streamSubmit is on the critical path. if needed, change to assert only
    LOG_TRACE(SCAL, "Submit stream ({}) pi={} {:#x}", m_name, pi, pi);

    // submission_alignment is not relevant for this stream
    return setPi(pi);
}

int Scal_Gaudi3::DirectModePDMAStream::setBuffer(Buffer *buffer)
{
    /*
        check for buffer size restrictions:
            minimal size is 2^16
            max size 2^24
            must be power of 2
            so quant must be ( TODO add assert for that 1/2/4/8/16/32/64/128/256)
            first stage FW supports only 2^16
     */
    if (buffer->size < c_core_counter_max_value)
    {
        LOG_ERR(SCAL,"{}, buffer size too small {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    if (buffer->size > c_core_counter_max_ccb_size)
    {
        LOG_ERR(SCAL,"{}, buffer size too big {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;

    }
    if (!isPowerOf2Unsigned(buffer->size))
    {
        LOG_ERR(SCAL,"{}, buffer size is not power of 2 {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    uint64_t submissionQueueAddress = buffer->base + buffer->pool->deviceBase;
    writeLbwReg(&m_chBlock->pqm_ch.desc_submit_fifo_cfg._raw, 1);
    writeLbwReg(&m_chBlock->pqm_ch.msq_base_h._raw, upper_32_bits(submissionQueueAddress));
    writeLbwReg(&m_chBlock->pqm_ch.msq_base_l._raw, lower_32_bits(submissionQueueAddress));
    writeLbwReg(&m_chBlock->pqm_ch.msq_size._raw, buffer->size - 1);

    m_coreBuffer = buffer;
    m_buffSize = buffer->size;

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::DirectModePDMAStream::getInfo(scal_stream_info_t& info) const
{
    info.name                = m_name.c_str();
    info.scheduler_handle    = nullptr;
    info.index               = m_id;
    info.type                = 0; // TODO
    info.current_pi          = getPi();
    info.control_core_buffer = (scal_buffer_handle_t)m_coreBuffer;

    info.submission_alignment = 0;
    info.command_alignment    = 8;
    info.priority             = m_priority;

    info.isDirectMode         = true;
    info.fenceCounterAddress = m_fenceCounterAddr;

    return SCAL_SUCCESS;
};