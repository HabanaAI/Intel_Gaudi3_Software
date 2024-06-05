#pragma once
#include <cstddef>

#include "common/scal_macros.h"
#include "common/scal_qman_program.h"

// each monitor:
// MON_PAY_ADDRL, MON_PAY_ADDRH, MON_PAY_DATA,
// MON_ARM, MON_CONFIG,
// MON_STATUS

// addr: smBase + varoffsetof(gaudi2::block_sob_objs, mon_config[monIdx])



class Monitor
{
public:
    struct ConfInfo
    {
        uint64_t payloadAddr;
        uint32_t payloadData;
        uint32_t config;
    };

    enum class CqEn      : bool {on, off};
    enum class LongSobEn : bool {on, off};
    enum class LbwEn     : bool {on, off};
    enum class CompType  : bool {EQUAL, BIG_EQUAL};

    struct RegsAddr
    {
        uint64_t payAddrL;
        uint64_t payAddrH;
        uint64_t payData;
        uint64_t arm;
        uint64_t config;

        std::string getDescription();
    };

    RegsAddr getRegsAddr() const { return m_regsAddr; }

protected:
    Monitor(uint64_t smBase) : m_smBase(smBase) {}

    template <typename PKT> void configure(Qman::Program& prog, ConfInfo confInfo);
    std::string getDescription();

    RegsAddr m_regsAddr {};
    uint64_t m_smBase   = 0;
    uint32_t m_monIdx   = -1;
};

class MonitorG2 : public Monitor
{
public:
    MonitorG2(uint64_t smBase, uint32_t monIdx);
    void configure(Qman::Program& prog, ConfInfo confInfo);

    static uint32_t buildConfVal(unsigned soIdx, unsigned numWrtM1, CqEn cqEn, LongSobEn longSobEn, LbwEn lbwEn);
    static uint32_t setLbwEn(uint32_t org, bool lbwVal);
    static uint32_t buildArmVal(uint32_t soIdx, uint16_t sod, CompType compType = CompType::BIG_EQUAL);
    static uint32_t buildArmVal(uint32_t soIdx, uint16_t sod, uint8_t mask, CompType compType = CompType::BIG_EQUAL);
};

class MonitorG3 : public Monitor
{
public:
    MonitorG3(uint64_t smBase, uint32_t monIdx, uint8_t smIdx = 0);
    void configure(Qman::Program& prog, ConfInfo confInfo);

    static uint32_t buildConfVal(unsigned soIdx, unsigned numWrtM1, CqEn cqEn, LongSobEn longSobEn, LbwEn lbwEn, uint8_t smIdx = 0);
    static uint32_t setLbwEn(uint32_t org, bool lbwVal);
    static uint32_t buildArmVal(uint32_t soIdx, uint16_t sod, CompType compType = CompType::BIG_EQUAL);
    static uint32_t buildArmVal(uint32_t soIdx, uint16_t sod, uint8_t mask, CompType compType = CompType::BIG_EQUAL);
};