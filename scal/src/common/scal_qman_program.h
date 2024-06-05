#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <memory>
#include <vector>
#include <map>

#include "logger.h"

// to enable debug of Qman programs (e.g. show the commands we send)
// uncomment the following define
//#define  LOG_EACH_COMMAND

namespace Qman
{
    class Command
    {
    public:
        Command(bool mb = 0, bool eb = 0, bool swtch = 0) :
            m_mb(mb), m_eb(eb), m_swtch(swtch)
        {
        }
        Command(const Command &other) = default;

        virtual ~Command() = default;

        virtual Command &operator=(const Command &) = default;

        virtual unsigned getSize() const = 0;
        virtual void serialize(void **buff) const = 0;

        virtual Command *clone() const = 0;

        bool m_mb;
        bool m_eb;
        bool m_swtch;
    };

    template <typename PACKET, int OPCODE>
    class Nop : public Command
    {
    public:
        Nop(bool mb = 0, bool eb = 0, bool swtc = 0) :
            Command(mb, eb, swtc)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: NOP", __FUNCTION__);
#endif
        }
        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->swtc = m_swtch;

            (*(uint8_t **)buff) += getSize();
        }
        virtual Command *clone() const
        {
            return new Nop(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class Stop : public Command
    {
    public:
        Stop() :
            Command()
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: Stop", __FUNCTION__);
#endif
        }
        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->swtc = m_swtch;

            (*(uint8_t **)buff) += getSize();
        }
        virtual Command *clone() const
        {
            return new Stop(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class WReg32 : public Command
    {
    public:
        uint16_t m_offset;
        uint32_t m_value;
        uint8_t m_pred;
        bool m_reg;
        uint8_t m_regId;

        WReg32(uint16_t offset, uint32_t value, bool mb = 0, bool eb = 0, bool reg = 0, uint8_t regId = 0) :
            Command(mb, eb), m_offset(offset), m_value(value), m_pred(0), m_reg(reg), m_regId(regId)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: offset={} value={:#x}", __FUNCTION__, offset, value);
#endif
        }
        WReg32() :
            WReg32(0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->swtc = m_swtch;
            packet->pred = m_pred;
            packet->reg_offset = m_offset;

            packet->reg = m_reg;

            // Don't set reg_id and value in the same time to avoid collision in the union.
            if (m_reg)
            {
                packet->reg_id = m_regId;
            }
            else
            {
                packet->value = m_value;
            }

            (*(uint8_t **)buff) += getSize();
        }
        virtual Command *clone() const
        {
            return new WReg32(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class WReg64Short : public Command
    {
    public:
        uint32_t m_offset;
        uint16_t m_dregOffset;
        uint8_t m_pred;
        uint8_t m_base;
        uint8_t m_regId;
        bool m_reg;

        WReg64Short() :
            Command(), m_offset(0), m_dregOffset(0), m_pred(0), m_base(0), m_regId(0), m_reg(false)
        {
        }

        WReg64Short(uint16_t dregOffset, uint8_t base, uint32_t offset) :
            Command(), m_offset(offset), m_dregOffset(dregOffset), m_pred(0), m_base(base), m_regId(0), m_reg(false)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: dregOffset={} base={} offset={}", __FUNCTION__, dregOffset, base, offset);
#endif
        }

        WReg64Short(uint16_t dregOffset, uint8_t base, uint8_t regId) :
            Command(), m_offset(0), m_dregOffset(dregOffset), m_pred(0), m_base(base), m_regId(regId), m_reg(true)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }

        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->offset = m_offset;
            packet->dreg_offset = m_dregOffset;
            packet->base = m_base;
            packet->reg_id = m_regId;
            packet->reg = m_reg;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new WReg64Short(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class WReg64Long : public Command
    {
    public:
        uint64_t m_offset;
        uint16_t m_dregOffset;
        uint8_t m_pred;
        uint8_t m_base;
        uint8_t m_dwEnable;
        bool m_rel;

        WReg64Long() :
            Command(), m_offset(0), m_dregOffset(0), m_pred(0), m_base(0), m_dwEnable(0), m_rel(false)
        {
        }

        WReg64Long(uint16_t dregOffset, uint8_t base, uint64_t offset, uint8_t dwEnable, bool rel) :
            Command(), m_offset(offset), m_dregOffset(dregOffset), m_pred(0), m_base(base),
            m_dwEnable(dwEnable), m_rel(rel)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: dregOffset={} base={} offset={:#x}", __FUNCTION__,
                dregOffset, base, offset);
#endif
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }

        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->offset = m_offset;
            packet->dreg_offset = m_dregOffset;
            packet->base = m_base;
            packet->dw_enable = m_dwEnable;
            packet->rel = m_rel;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new WReg64Long(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class WRegBulk : public Command
    {
    public:
        uint16_t m_offset;
        uint8_t m_pred;
        std::vector<uint64_t> m_values;

        WRegBulk() :
            Command(), m_offset(0), m_pred(0), m_values()
        {
        }

        WRegBulk(uint16_t offset, const uint64_t *values, const unsigned numValues) :
            Command(), m_offset(offset), m_pred(0), m_values(numValues)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: offset={:#x}", __FUNCTION__,offset);
#endif
            memcpy(m_values.data(), values, numValues * sizeof(uint64_t));
        }

        WRegBulk(uint16_t offset, const uint32_t *values, unsigned numValues) :
            WRegBulk(offset, (const uint64_t *)values, numValues / 2)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: offset={:#x}", __FUNCTION__,offset);
#endif
            assert((numValues & 1) == 0);
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET) + (m_values.size() * sizeof(uint64_t));
        }

        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->size64 = m_values.size();
            packet->pred = m_pred;
            packet->reg_offset = m_offset;
            memcpy(packet->values, m_values.data(), m_values.size() * sizeof(uint64_t));

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new WRegBulk(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class MsgLong : public Command
    {
    public:
        uint64_t m_addr;
        uint32_t m_value;
        uint8_t m_pred;
        unsigned m_op;
        bool m_noSnoop;
        bool m_weaklyOrdered;

        MsgLong(uint64_t addr, uint32_t value, bool mb = false, bool eb = false) :
            Command(mb, eb), m_addr(addr), m_value(value), m_pred(0),
            m_op(0), m_noSnoop(0), m_weaklyOrdered(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: addr={:#x} value={:#x}", __FUNCTION__, addr, value);
#endif
        }

        MsgLong() :
            MsgLong(0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->weakly_ordered = m_weaklyOrdered;
            packet->no_snoop = m_noSnoop;
            packet->op = m_op;
            packet->value = m_value;
            packet->addr = m_addr;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new MsgLong(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class MsgProt : public Command
    {
    public:
        uint64_t m_addr;
        uint32_t m_value;
        uint8_t m_pred;
        unsigned m_op;
        bool m_noSnoop;
        bool m_weaklyOrdered;

        MsgProt(uint64_t addr, uint32_t value, bool mb = false, bool eb = false) :
            Command(mb, eb), m_addr(addr), m_value(value), m_pred(0),
            m_op(0), m_noSnoop(0), m_weaklyOrdered(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: addr={:#x} value={:#x}", __FUNCTION__, addr, value);
#endif
        }

        MsgProt() :
            MsgProt(0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->weakly_ordered = m_weaklyOrdered;
            packet->no_snoop = m_noSnoop;
            packet->op = m_op;
            packet->value = m_value;
            packet->addr = m_addr;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new MsgProt(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class Fence : public Command
    {
    public:
        unsigned m_decVal;
        uint8_t m_targetVal;
        unsigned m_id;
        uint8_t m_pred;

        Fence(unsigned id, uint8_t targetVal, unsigned decVal, bool mb = false, bool eb = false) :
            Command(mb, eb), m_decVal(decVal), m_targetVal(targetVal), m_id(id), m_pred(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: id={} targetVal={} decVal={}",__FUNCTION__, id, (unsigned)targetVal, decVal);
#endif
        }

        Fence() :
            Fence(0, 0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->dec_val = m_decVal;
            packet->target_val = m_targetVal;
            packet->id = m_id;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new Fence(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class Repeat : public Command
    {
    public:
        bool m_sore;
        bool m_outer;
        uint16_t m_jumpPtr;
        uint8_t m_pred;

        Repeat(bool sore, bool outer, uint16_t jumpPtr = 0) :
            Command(), m_sore(sore), m_outer(outer), m_jumpPtr(jumpPtr), m_pred(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: Repeat", __FUNCTION__);
#endif
        }
        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->sore = m_sore;
            packet->outer = m_outer;
            packet->pred = m_pred;
            packet->jmp_ptr = m_jumpPtr;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new Repeat(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class Wait : public Command
    {
    public:
        uint32_t m_cycles;
        unsigned m_incVal;
        unsigned m_id;

        Wait(uint32_t cycles, unsigned incVal, unsigned id) :
            Command(), m_cycles(cycles), m_incVal(incVal), m_id(id)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: cycles={} incVal={} id={}",__FUNCTION__, cycles, incVal, id);
#endif
        }

        Wait() :
            Wait(0, 0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->num_cycles_to_wait = m_cycles;
            packet->inc_val = m_incVal;
            packet->id = m_id;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new Wait(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class CbList : public Command
    {
    public:
        uint64_t m_tableAddr;
        uint64_t m_indexAddr;
        uint8_t m_pred;
        uint8_t m_sizeDesc;

        CbList(uint64_t tableAddr, uint64_t indexAddr) :
            Command(), m_tableAddr(tableAddr), m_indexAddr(indexAddr), m_pred(0), m_sizeDesc(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: tableAddr={:#x} indexAddr={:#x}", __FUNCTION__, tableAddr, indexAddr);
#endif
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->pred = m_pred;
            packet->size_desc = m_sizeDesc;
            packet->index_addr = m_indexAddr;
            packet->table_addr = m_tableAddr;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new CbList(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class LoadAndExec : public Command
    {
    public:
        uint64_t m_srcAddr;
        bool m_dest;
        bool m_load;
        bool m_exec;
        bool m_eType;
        bool m_pmap;
        uint8_t m_pred;

        LoadAndExec(uint64_t srcAddr, bool load, bool dest, bool exec, bool eType, bool pmap) :
            Command(), m_srcAddr(srcAddr), m_dest(dest), m_load(load), m_exec(exec), m_eType(eType), m_pmap(pmap), m_pred(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: srcAddr={:#x}", __FUNCTION__, srcAddr);
#endif
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->dst = m_dest;
            packet->load = m_load;
            packet->exe = m_exec;
            packet->etype = m_eType;
            packet->pred = m_pred;
            packet->src_addr = m_srcAddr;
            packet->pmap = m_pmap;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new LoadAndExec(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class CpDma : public Command
    {
    public:
        uint32_t m_tSize;
        uint64_t m_src;
        uint8_t m_pred;
        bool m_upperCp;

        CpDma(uint64_t src, uint32_t size) :
            Command(), m_tSize(size), m_src(src), m_pred(0), m_upperCp(false)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: src={:#x} size={}", __FUNCTION__, src, size);
#endif
        }

        CpDma() :
            CpDma(0, 0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->msg_barrier = m_mb;
            packet->tsize = m_tSize;
            packet->src_addr = m_src;
            packet->upper_cp = m_upperCp;
            packet->pred = m_pred;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new CpDma(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class ArbPoint : public Command
    {
    public:
        uint8_t m_prio;
        bool m_release;
        uint8_t m_pred;

        ArbPoint(uint8_t prio) :
            Command(), m_prio(prio), m_release(0), m_pred(0)
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: prio={}", __FUNCTION__, prio);
#endif
        }

        ArbPoint() :
            Command(), m_prio(0), m_release(1), m_pred(0)
        {
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET);
        }
        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = 0;
            packet->msg_barrier = 0;
            packet->priority = m_prio;
            packet->rls = m_release;
            packet->pred = m_pred;

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new ArbPoint(*this);
        }
    };

    template <typename PACKET, int OPCODE>
    class WriteArcStream : public Command
    {
    public:
        std::vector<uint64_t> m_values;

        WriteArcStream() :
            Command(), m_values()
        {
        }

        WriteArcStream(const uint64_t *values, unsigned numValues) :
            Command()
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"{}: numValues={}", __FUNCTION__, numValues);
#endif
            memcpy(m_values.data(), values, numValues * sizeof(uint64_t));
        }

        WriteArcStream(const uint32_t *values, unsigned numValues) :
            WriteArcStream((const uint64_t *)values, numValues / 2)
        {
            assert((numValues & 1) == 0);
        }

        virtual unsigned getSize() const
        {
            return sizeof(PACKET) + (m_values.size() * sizeof(uint64_t));
        }

        virtual void serialize(void **buff) const
        {
            PACKET *packet = (PACKET *)(*buff);

            memset(packet, 0, getSize());

            packet->opcode = OPCODE;
            packet->eng_barrier = m_eb;
            packet->swtc = m_swtch;
            packet->msg_barrier = m_mb;
            packet->size64 = m_values.size();
            memcpy(packet->values, m_values.data(), m_values.size() * sizeof(uint64_t));

            (*(uint8_t **)buff) += getSize();
        }

        virtual Command *clone() const
        {
            return new WriteArcStream(*this);
        }
    };

    class Program
    {
    public:
        void addCommand(const Command & cmd)
        {
            m_cmds.emplace_back(std::shared_ptr<Command>(cmd.clone()));
            m_size += cmd.getSize();
        }

        unsigned getSize() const { return m_size; }

        void serialize(void *buff) const
        {
#ifdef LOG_EACH_COMMAND
            LOG_DEBUG(SCAL,"Program::{}", __FUNCTION__, isUpperCPProgram, deviceAddr);
#endif
            for (const auto & cmd : m_cmds)
            {
                cmd->serialize(&buff);
            }
        }

    private:
        std::vector<std::shared_ptr<Command>> m_cmds;
        unsigned m_size = 0;

    };

    class Workload
    {
    public:
        struct ProgramInfo
        {
            Program program;
            bool isUpperCp;
        };

        typedef std::map<unsigned, ProgramInfo> Qid2ProgramInfoMap;

        const std::vector<Qid2ProgramInfoMap> & getPrograms() const {return m_programs;}
        struct PDmaTransfer
        {
            uint64_t src;
            uint64_t dst;
            uint32_t size;
        };

        const std::vector<PDmaTransfer> & getPDmaTransfers() const {return m_pdmaTransfers;}

        bool addPDmaTransfer(const uint64_t src, const uint64_t dst, uint32_t size)
        {
            m_pdmaTransfers.push_back({src, dst, size});
            return true;
        }

        bool addProgram(const Program & program, const unsigned qid, const bool upperCP = false)
        {
            for (auto &sub : m_programs)
            {
                if (sub.find(qid) == sub.end())
                {
                    sub[qid] = {program, upperCP};
                    return true;
                }
            }

            m_programs.push_back({});
            m_programs.back()[qid] = {program, upperCP};

            return true;
        }

    private:
        std::vector<Qid2ProgramInfoMap> m_programs;
        std::vector<PDmaTransfer> m_pdmaTransfers;

    };
};



