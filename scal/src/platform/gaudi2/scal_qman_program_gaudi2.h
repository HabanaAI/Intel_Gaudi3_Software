#pragma once
#include "common/scal_qman_program.h"
#include "gaudi2/gaudi2_packets.h"

// to enable debug of Qman programs (e.g. show the commands we send)
// uncomment the following define
//#define  LOG_EACH_COMMAND

namespace Qman
{
    class LinDmaGaudi2 : public Command
    {
    public:
        uint32_t m_tsize;
        bool m_wrComp;
        bool m_memset;
        uint8_t m_endian;
        bool m_bf16;
        bool m_fp16;
        bool m_contextIdInc;
        bool m_addOffset0;
        uint64_t m_src;
        uint64_t m_dst;

        LinDmaGaudi2(uint64_t dst, uint64_t src, uint32_t tsize, bool wrComp = 0, bool memset = 0) :
            Command(), m_tsize(tsize), m_wrComp(wrComp), m_memset(memset), m_endian(0),
            m_bf16(0), m_fp16(0), m_contextIdInc(), m_addOffset0(0),
            m_src(src), m_dst(dst)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: dst={:#x} src={:#x} size={}", __FUNCTION__, dst, src, tsize);
#endif
        }

        LinDmaGaudi2() :
            LinDmaGaudi2(0, 0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(packet_lin_dma);
        }
        virtual void serialize(void **buff) const
        {
            packet_lin_dma *packet = (packet_lin_dma *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = PACKET_LIN_DMA;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->tsize = m_tsize;
            packet->wrcomp = m_wrComp;
            packet->memset = m_memset;
            packet->endian = m_endian;
            packet->bf16 = m_bf16;
            packet->fp16 = m_fp16;
            packet->context_id_inc = m_contextIdInc;
            packet->add_offset_0 = m_addOffset0;
            packet->src_addr = m_src;
            packet->dst_addr = m_dst;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new LinDmaGaudi2(*this);
        }
    };

    class MsgShortGaudi2 : public Command
    {
    public:
        uint16_t m_offset;
        uint32_t m_value;
        unsigned m_op;
        bool m_noSnoop;
        bool m_weaklyOrdered;
        bool m_dw;
        unsigned m_base;

        MsgShortGaudi2(unsigned base, uint16_t offset, uint32_t value, bool mb = false, bool eb = false) :
            Command(mb, eb), m_offset(offset), m_value(value), m_op(0),
            m_noSnoop(0), m_weaklyOrdered(0), m_dw(0), m_base(base)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: base={} offset={} value={:#x}", __FUNCTION__, base, offset, value);
#endif

        }

        MsgShortGaudi2() :
            MsgShortGaudi2(0, 0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(packet_msg_short);
        }
        virtual void serialize(void **buff) const
        {
            packet_msg_short *packet = (packet_msg_short *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = PACKET_MSG_SHORT;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->base = m_base;
            packet->msg_addr_offset = m_offset;
            packet->weakly_ordered = m_weaklyOrdered;
            packet->no_snoop = m_noSnoop;
            packet->dw = m_dw;
            packet->op = m_op;
            packet->value = m_value;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new MsgShortGaudi2(*this);
        }
    };
};




using Nop = Qman::Nop<packet_nop, PACKET_NOP>;
using MsgShort = Qman::MsgShortGaudi2;
using MsgLong = Qman::MsgLong<packet_msg_long, PACKET_MSG_LONG>;
using WReg32 = Qman::WReg32<packet_wreg32, PACKET_WREG_32>;
using Wait = Qman::Wait<packet_wait, PACKET_WAIT>;
using Fence = Qman::Fence<packet_fence, PACKET_FENCE>;
using LinDma = Qman::LinDmaGaudi2;
using CpDma = Qman::CpDma<packet_cp_dma, PACKET_CP_DMA>;