#include <algorithm>
#include "scal_gaudi3.h"
#include "scal_macros.h"
#include "scal_data_gaudi3.h"
#include "gaudi3/asic_reg_structs/qman_regs.h"
#include "gaudi3/gaudi3_pqm_packets.h"
#include "scal_qman_program_gaudi3.h"
#include "gaudi3/asic_reg_structs/pdma_ch_a_regs.h"
#include "infra/monitor.hpp"
#include "infra/sync_mgr.hpp"

class LinPDma : public Qman::Command
{
public:
    LinPDma(uint64_t dst, uint64_t src, uint32_t tsize, uint64_t compAddr = 0, unsigned compVal = 1) :
    Qman::Command(), m_dst(dst), m_src(src), m_tsize(tsize), m_compAddr(compAddr), m_compVal(compVal)
    {
#ifdef LOG_EACH_COMMAND
        LOG_DEBUG(SCAL,"{}: dst={:#x} src={:#x} size={}", __FUNCTION__, dst, src, tsize);
#endif
    }

    LinPDma() : LinPDma(0, 0, 0) {}

    uint64_t m_dst;
    uint64_t m_src;
    uint32_t m_tsize;
    uint64_t m_compAddr;
    uint32_t m_compVal;

    struct packet_lin_pdma_wcomp
    {
        struct pqm_packet_lin_pdma lin_pdma;
        uint32_t src_addr_hi;
        uint32_t dst_addr_hi;
        uint32_t wrcomp_addr_lo;
        uint32_t wrcomp_data;
        uint32_t wrcomp_addr_hi;
        uint32_t fence;
    };

    struct packet_lin_pdma
    {
        struct pqm_packet_lin_pdma lin_pdma;
        uint32_t src_addr_hi;
        uint32_t dst_addr_hi;
    };

    virtual unsigned getSize() const
    {
        return m_compAddr ? sizeof(packet_lin_pdma_wcomp) : sizeof(packet_lin_pdma);
    }

    virtual void serialize(void **buff) const
    {
        packet_lin_pdma_wcomp *packet = (packet_lin_pdma_wcomp *)(*buff);

        memset(packet, 0, getSize());

        packet->lin_pdma.tsize = m_tsize;
        packet->lin_pdma.wrcomp = m_compAddr ? 1 : 0;
        packet->lin_pdma.en_desc_commit = 1;
        packet->lin_pdma.direction = 1;
        packet->lin_pdma.src_addr_lo = lower_32_bits(m_src);
        packet->lin_pdma.dst_addr_lo = lower_32_bits(m_dst);
        packet->lin_pdma.opcode = PQM_PACKET_LIN_PDMA;
        packet->lin_pdma.eng_barrier = m_eb;
        packet->src_addr_hi = upper_32_bits(m_src);
        packet->dst_addr_hi = upper_32_bits(m_dst);
        if (m_compAddr)
        {
            packet->wrcomp_data = m_compVal;
            packet->wrcomp_addr_lo = lower_32_bits(m_compAddr);
            packet->wrcomp_addr_hi = upper_32_bits(m_compAddr);
        }

        (*(uint8_t **)buff) += getSize();
    }

    virtual Qman::Command *clone() const
    {
        return new LinPDma(*this);
    }
};

bool Scal_Gaudi3::submitQmanWkld(const Qman::Workload &wkld)
{
    return m_configChannel.submit(wkld);
}

int Scal_Gaudi3::ConfigPdmaChannel::init()
{
    if ((m_smIdx == -1) ||
        (m_monIdx == -1) ||
        (m_cqIdx == -1) ||
        (m_isrIdx == -1) ||
        (m_qid == -1) ||
        (!m_stream))
    {
        LOG_ERR(SCAL,"{}: config PDMA is not configured", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    if (m_buff_ptr ||
        m_buff_dev_addr ||
        m_ctr_ptr ||
        m_ctr_dev_addr ||
        m_ch_base ||
        !m_used_qids.empty())
    {
        LOG_ERR(SCAL,"{}: config PDMA already initialized", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    // allocate memory for the counter and reset it
    m_ctr_ptr = (volatile uint64_t *)aligned_alloc(c_host_page_size, c_host_page_size);
    if (!m_ctr_ptr)
    {
        LOG_ERR(SCAL,"{}: failed to allocate memory for the host counter", __FUNCTION__);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }
    *m_ctr_ptr = 0;

    // map the counter to the device
    m_ctr_dev_addr = hlthunk_host_memory_map(m_fd, (void*)m_ctr_ptr, 0, c_host_page_size);
    if (!m_ctr_dev_addr)
    {
        LOG_ERR(SCAL,"{}: failed to map the host address of the counter buffer", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    // allocate the command buffer
    m_buff_ptr = aligned_alloc(c_host_page_size, c_command_buffer_size);
    if (!m_buff_ptr)
    {
        LOG_ERR(SCAL,"{}: failed to allocate memory for the command buffer", __FUNCTION__);
        assert(0);
        return SCAL_OUT_OF_MEMORY;
    }

    // map the command buffer to the device
    m_buff_dev_addr = hlthunk_host_memory_map(m_fd, m_buff_ptr, 0, c_command_buffer_size);
    if (!m_buff_dev_addr)
    {
        LOG_ERR(SCAL,"{}: failed to map the host address of the command buffer", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    // set the command buffer
    int ret = m_stream->setBuffer(m_buff_dev_addr, c_command_buffer_size);
    if (ret != SCAL_SUCCESS) return ret;

    // reset pi
    m_pi = 0;

    if (!pdmaId2baseAddrA(m_qid, m_ch_base))
    {
        LOG_ERR(SCAL,"{}, pdmaId2baseAddrA() failed for core id {}", __FUNCTION__, m_qid);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

void Scal_Gaudi3::ConfigPdmaChannel::deinit()
{
    m_ch_base = 0;

    if (m_buff_dev_addr)
    {
        int rc = hlthunk_memory_unmap(m_fd, m_buff_dev_addr);
        if (rc)
        {
            LOG_ERR(SCAL,"{}: fd={} pool.deviceBase hlthunk_memory_unmap() failed rc={} errno={} error={}", __FUNCTION__, m_fd, rc, errno, std::strerror(errno));
            assert(0);
        }
        m_buff_dev_addr = 0;
    }

    if (m_ctr_dev_addr)
    {
        int rc = hlthunk_memory_unmap(m_fd, m_ctr_dev_addr);
        if (rc)
        {
            LOG_ERR(SCAL,"{}: fd={} pool.deviceBase hlthunk_memory_unmap() failed rc={} errno={} error={}", __FUNCTION__, m_fd, rc, errno, std::strerror(errno));
            assert(0);
        }
        m_ctr_dev_addr = 0;
    }

    if (m_buff_ptr)
    {
        free(m_buff_ptr);
        m_buff_ptr = 0;
    }

    if (m_ctr_ptr)
    {
        free((void*)m_ctr_ptr);
        m_ctr_ptr = nullptr;
    }

    m_used_qids.clear();
}

bool Scal_Gaudi3::ConfigPdmaChannel::submit(const Qman::Workload &wkld)
{
    if (!m_ch_base)
    {
        LOG_ERR(SCAL,"{}: Channel not initalized", __FUNCTION__);
        assert(0);
        return false;
    }

    const uint64_t smObjectsBase = SyncMgrG3::getSmBase(m_smIdx);

    const uint64_t monArmAddr     = MonitorG3(smObjectsBase, m_monIdx, 0).getRegsAddr().arm;
    const unsigned fenceIncOffset = offsetof(gaudi3::block_pdma_ch_a, pqm_ch.fence_inc[c_fence_ctr_idx]);

    Qman::Program pdmaProg;

    _configInitialization(pdmaProg);

    if (!wkld.getPDmaTransfers().empty())
    {
        unsigned idx = wkld.getPDmaTransfers().size() - 1;
        for (const auto & tran : wkld.getPDmaTransfers())
        {
            uint64_t compAddr = idx ? 0 : (m_ch_base + fenceIncOffset);
            pdmaProg.addCommand(LinPDma(tran.dst, tran.src, tran.size, compAddr));
            idx--;
        }

        pdmaProg.addCommand(Fence(c_fence_ctr_idx, 1, 1));
    }

    MsgLong compMsg(m_ch_base + fenceIncOffset, 1, true, true);
    // allocate memory for the programs
    unsigned wkldSize = 0;
    for (const auto & sub : wkld.getPrograms())
    {
        for (const auto & prog : sub)
        {
            if (prog.second.isUpperCp)
            {
                LOG_ERR(SCAL,"{}: The wkld contains an upper CP program. qid = {}", __FUNCTION__, prog.first);
                assert(0);
                return false;
            }

            unsigned progSize = prog.second.program.getSize() + compMsg.getSize();
            progSize = (progSize + c_cl_size - 1) & ~(c_cl_size - 1);
            wkldSize += progSize;
        }
    }

    uint8_t * progHostAddr = nullptr;
    uint64_t progDeviceAddr = 0;

    if (wkldSize)
    {
        progHostAddr = (uint8_t*)aligned_alloc(c_cl_size, wkldSize);
        if (!progHostAddr)
        {
            LOG_ERR(SCAL, "aligned_alloc failed errno={}: {}. fd={}, size={}",
                    errno, std::strerror(errno), m_fd, wkldSize);
            assert(0);
            return false;
        }

        progDeviceAddr = hlthunk_host_memory_map(m_fd, progHostAddr, 0, wkldSize);
        if (!progDeviceAddr)
        {
            LOG_ERR(SCAL, "hlthunk_host_memory_map failed errno={}: {}. fd={}, hostAddr={} , size={}",
                    errno, std::strerror(errno), m_fd, progHostAddr, wkldSize);
            assert(0);
            free(progHostAddr);
            return false;
        }

        unsigned progOffset = 0;

        for (const auto & sub : wkld.getPrograms())
        {
            for (const auto & program : sub)
            {
                unsigned qid = program.first;
                const Qman::Program & prog = program.second.program;

                uint64_t qmBase;
                if (!queueId2DccmAddr(qid, qmBase))
                {
                    LOG_ERR(SCAL,"{}, queueId2DccmAddr() failed for queue id {}", __FUNCTION__, qid);
                    assert(0);
                    hlthunk_memory_unmap(m_fd, progDeviceAddr);
                    free(progHostAddr);
                    return false;
                }
                unsigned offsetToQman;
                if (!queueId2OffsetToQman(qid, offsetToQman))
                {
                    LOG_ERR(SCAL,"{}, queueId2OffsetToQman() failed for queue id {}", __FUNCTION__, qid);
                    assert(0);
                    hlthunk_memory_unmap(m_fd, progDeviceAddr);
                    free(progHostAddr);
                    return false;
                }
                qmBase += offsetToQman;

                if (m_used_qids.find(qid) == m_used_qids.end())
                {
                    // - disables the shadow CI/ICI writes
                    pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman, cq_cfg0), 1));

                    // - forces the CP_SWITCH to 0 (twice because of hw bug)
                    pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman,cp_ext_switch), 0));
                    pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman,cp_ext_switch), 0));

                    m_used_qids.insert(qid);
                }

                prog.serialize(progHostAddr + progOffset);
                void * tmpPtr = progHostAddr + progOffset + prog.getSize();
                compMsg.serialize(&tmpPtr);
                unsigned progSize = prog.getSize() + compMsg.getSize();

                pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman, cq_tsize), progSize));
                pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman, cq_ptr_lo), lower_32_bits(progDeviceAddr + progOffset)));
                pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman, cq_ptr_hi), upper_32_bits(progDeviceAddr + progOffset)));
                pdmaProg.addCommand(MsgLong(qmBase + offsetof(gaudi3::block_qman, cq_ctl), 0));

                progOffset += (progSize + c_cl_size - 1) & ~(c_cl_size - 1);
            }

            unsigned rem = sub.size();
            while (rem)
            {
                unsigned dec = std::min(rem, 8U);
                rem -= dec;
                pdmaProg.addCommand(Fence(c_fence_ctr_idx, dec, dec));
            }
        }
    }

    pdmaProg.addCommand(MsgLong(monArmAddr, 0));

    if ((m_pi % c_command_buffer_size) + pdmaProg.getSize() > c_command_buffer_size)
    {
        if (pdmaProg.getSize() > c_command_buffer_size)
        {
            LOG_ERR(SCAL,"{}: PDMA program too big", __FUNCTION__);
            assert(0);
            hlthunk_memory_unmap(m_fd, progDeviceAddr);
            free(progHostAddr);
            return false;
        }

        uint8_t * tmpBuff = (uint8_t*)malloc(pdmaProg.getSize());
        if (!tmpBuff)
        {
            LOG_ERR(SCAL,"{}: Failed to allocate temp command buffer", __FUNCTION__);
            assert(0);
            hlthunk_memory_unmap(m_fd, progDeviceAddr);
            free(progHostAddr);
            return false;
        }

        pdmaProg.serialize(tmpBuff);
        unsigned firstChunkSize = c_command_buffer_size - (m_pi % c_command_buffer_size);
        memcpy(((uint8_t*)m_buff_ptr) + (m_pi % c_command_buffer_size), tmpBuff, firstChunkSize);
        memcpy(m_buff_ptr, tmpBuff + firstChunkSize, pdmaProg.getSize() - firstChunkSize);
        free(tmpBuff);
        LOG_INFO(SCAL,"{}: command buffer overflow, continuing at 0", __FUNCTION__);
    }
    else
    {
        pdmaProg.serialize(((uint8_t*)m_buff_ptr) + (m_pi % c_command_buffer_size));
    }

    m_pi += pdmaProg.getSize();
    uint64_t targetVal = (*m_ctr_ptr) + 1;
    m_stream->setPi(m_pi);

    uint32_t junk;
    int rc = hlthunk_wait_for_interrupt(m_fd, (void*)m_ctr_ptr, targetVal, m_isrIdx, -1, &junk);
    if (rc)
    {
        LOG_ERR(SCAL,"{}: wait for PDMA returned an error. ret = {}", __FUNCTION__, rc);
        assert(0);
        if (progDeviceAddr) hlthunk_memory_unmap(m_fd, progDeviceAddr);
        if (progHostAddr) free(progHostAddr);
        return false;
    }

    if (progDeviceAddr) hlthunk_memory_unmap(m_fd, progDeviceAddr);
    if (progHostAddr) free(progHostAddr);

    return true;
}

int Scal_Gaudi3::ConfigPdmaChannel::configurePdmaPtrsMode()
{
    return m_stream->configurePdmaPtrsMode();
}


bool Scal_Gaudi3::ConfigPdmaChannel::submitPdmaConfiguration(const Qman::Program& configurationProgram)
{
    if (!m_ch_base)
    {
        LOG_ERR(SCAL,"{}: Channel not initalized", __FUNCTION__);
        assert(0);
        return false;
    }

    const uint64_t smObjectsBase = SyncMgrG3::getSmBase(m_smIdx);
    const uint64_t monArmAddr    = MonitorG3(smObjectsBase, m_monIdx).getRegsAddr().arm;

    Qman::Program pdmaProg;

    _configInitialization(pdmaProg);
    MsgLong compMsg(monArmAddr, 0);

    unsigned configurationSize = pdmaProg.getSize() + configurationProgram.getSize() + compMsg.getSize();

    if ((m_pi % c_command_buffer_size) + configurationSize > c_command_buffer_size)
    {
        if (configurationSize > c_command_buffer_size)
        {
            LOG_ERR(SCAL,"{}: PDMA program too big", __FUNCTION__);
            assert(0);
            return false;
        }

        void* tmpBuff = malloc(configurationSize);
        if (!tmpBuff)
        {
            LOG_ERR(SCAL,"{}: Failed to allocate temp command buffer", __FUNCTION__);
            assert(0);
            return false;
        }

        pdmaProg.serialize(tmpBuff);
        configurationProgram.serialize(tmpBuff);
        compMsg.serialize(&tmpBuff);

        unsigned firstChunkSize = (m_pi % c_command_buffer_size) + configurationSize - c_command_buffer_size;

        uint8_t* submissionQueueDst = ((uint8_t*)m_buff_ptr) + (m_pi % c_command_buffer_size);
        memcpy(submissionQueueDst, tmpBuff, firstChunkSize);
        memcpy(m_buff_ptr, (uint8_t*)tmpBuff + firstChunkSize, configurationSize - firstChunkSize);
    }
    else
    {
        uint8_t* submissionQueueDst = ((uint8_t*)m_buff_ptr) + (m_pi % c_command_buffer_size);

        pdmaProg.serialize(submissionQueueDst);
        submissionQueueDst += pdmaProg.getSize();
        configurationProgram.serialize(submissionQueueDst);
        submissionQueueDst += configurationProgram.getSize();
        compMsg.serialize((void**)&submissionQueueDst);
    }

    m_pi += configurationSize;
    uint64_t targetVal = (*m_ctr_ptr) + 1;
    m_stream->setPi(m_pi);

    uint32_t status = 0;
    int rc = hlthunk_wait_for_interrupt(m_fd, (void*)m_ctr_ptr, targetVal, m_isrIdx, -1, &status);
    if (rc)
    {
        LOG_ERR(SCAL,"{}: Wait for PDMA returned an error. ret = {}", __FUNCTION__, rc);
        assert(0);
        return false;
    }

    return true;
}

void Scal_Gaudi3::ConfigPdmaChannel::_configInitialization(Qman::Program& pdmaProg)
{
    // Done once
    // 1) Reset local-channel fence-counter
    // 2) Configure CQ with its submission-queue and ISR-Info
    // 3) Configure MON towards the CQ (later will be arm to fire immediately)
    if (!m_pi)
    {
        const uint64_t smObjectsBase  = SyncMgrG3::getSmBase(m_smIdx);
        const unsigned fenceCtrOffset = offsetof(gaudi3::block_pdma_ch_a, pqm_ch.fence_cnt[c_fence_ctr_idx]);

        // reset the fence counter
        pdmaProg.addCommand(MsgLong(m_ch_base + fenceCtrOffset, 0));

        // configure the CQ
        const uint64_t smGlobalBase = smObjectsBase + (mmHD0_SYNC_MNGR_GLBL_BASE - mmHD0_SYNC_MNGR_OBJS_BASE);
        configureCQ(pdmaProg, smGlobalBase, m_cqIdx, m_ctr_dev_addr, m_isrIdx);

        // configure the monitors
        uint32_t mc = MonitorG3::buildConfVal(0, 0, Monitor::CqEn::on, Monitor::LongSobEn::off, Monitor::LbwEn::on);

        configureOneMonitor(pdmaProg, m_monIdx, smObjectsBase, mc, m_cqIdx, 1);
    }
}

Scal_Gaudi3::DirectModePdmaChannel::DirectModePdmaChannel(std::string&       streamName,
                                                          std::string&       pdmaEngineName,
                                                          const std::string& completionGroupName,
                                                          Scal*              pScal,
                                                          unsigned           qid,
                                                          uint64_t           isrRegister,
                                                          unsigned           isrIndex,
                                                          unsigned           longSoSmIndex,
                                                          unsigned           longSoIndex,
                                                          uint64_t           fenceCounterAddress,
                                                          unsigned           cqIndex,
                                                          unsigned           priority,
                                                          unsigned           id)
: m_stream(streamName, qid, c_fence_ctr_idx, fenceCounterAddress, priority, id),
  m_completionGroup(completionGroupName, pScal, cqIndex, isrRegister, isrIndex,
                    longSoSmIndex, longSoIndex),
  m_pdmaEngineName(pdmaEngineName)
{
}

int Scal_Gaudi3::DirectModePdmaChannel::initChannelCQ()
{
    if (m_isInitialized)
    {
        LOG_ERR(SCAL,"{}: Instance is already initialized", __FUNCTION__);
        assert(0);
        return SCAL_FAILURE;
    }

    int rc = m_stream.initChannelCQ(m_counterMmuAddress, m_completionGroup.m_isrRegister, m_completionGroup.isrIdx);
    if (rc == SCAL_SUCCESS)
    {
        m_isInitialized = true;
    }
    return rc;
}
