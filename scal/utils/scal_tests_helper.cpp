#include "platform/gaudi3/scal_gaudi3.h"
#include "platform/gaudi3/scal_data_gaudi3.h"
#include "common/scal_utilities.h"
#include "scal_tests_helper.h"

PDMA_internals_helper::~PDMA_internals_helper()
{
    if (m_chBlock)
    {
        unmapLbwMemory(m_chBlock, m_allocatedSize);
    }
}

void PDMA_internals_helper::init_baseA_block(int fd, unsigned qid)
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
}

void PDMA_internals_helper::set_CI(unsigned ci)
{
    // NOTE!  this ASSUMES the PDMA is idle and ci==pi

    // HW/SIM writes the same value to pi
    writeLbwReg(&m_chBlock->pqm_ch.msq_ci._raw, ci);
}

unsigned PDMA_internals_helper::getQidFromChannelId(unsigned chid)
{
    for (const PdmaChannelInfo& info : c_pdma_channels_info_arr)
    {
        if (chid == info.channelId)
        {
            return info.engineId;
        }
    }
    return (unsigned)-1;
}