#include "linux/types.h"
#include <common/scal_qman_program.h>

struct G2pkts {
#include <gaudi2/gaudi2_packets.h>
};
struct G3pkts {
#include <gaudi3/gaudi3_packets.h>
};

#include "monitor.hpp"
#include "common/scal_macros.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/scal_gaudi3.h"
#include "scal_base.h"

template <typename PKT>
void Monitor::configure(Qman::Program& prog, ConfInfo confInfo)
{
    using MsgLongGx = Qman::MsgLong<typename PKT::packet_msg_long, PKT::PACKET_MSG_LONG>;

    prog.addCommand(MsgLongGx(m_regsAddr.config, confInfo.config));
    // set payload addr Hi & Low
    prog.addCommand(MsgLongGx(m_regsAddr.payAddrH, upper_32_bits(confInfo.payloadAddr)));
    prog.addCommand(MsgLongGx(m_regsAddr.payAddrL, lower_32_bits(confInfo.payloadAddr)));
    // set payload data
    prog.addCommand(MsgLongGx(m_regsAddr.payData, confInfo.payloadData));

    LOG_DEBUG(SCAL, "configure {} addr {:#x} data {:#x} config {:#x}", getDescription(), confInfo.payloadAddr, confInfo.payloadData, confInfo.config);
}

std::string Monitor::getDescription()
{
    return fmt::format("smBase {:#x} monIdx {:#x} {}",
                       m_smBase, m_monIdx, m_regsAddr.getDescription());
}

std::string Monitor::RegsAddr::getDescription()
{
    return fmt::format("regsAddr payAddrL/H data arm config {:#x}/{:#x}/{:#x}/{:#x}",
                       payAddrL, payAddrH, payData, arm, config);
}

/********************************************************************************************/
/*                                         G2                                               */
/********************************************************************************************/
MonitorG2::MonitorG2(uint64_t smBase, uint32_t monIdx) : Monitor(smBase)
{
    m_regsAddr.payAddrL = smBase + varoffsetof(gaudi2::block_sob_objs, mon_pay_addrl[monIdx]);
    m_regsAddr.payAddrH = smBase + varoffsetof(gaudi2::block_sob_objs, mon_pay_addrh[monIdx]);
    m_regsAddr.payData  = smBase + varoffsetof(gaudi2::block_sob_objs, mon_pay_data[ monIdx]);
    m_regsAddr.arm      = smBase + varoffsetof(gaudi2::block_sob_objs, mon_arm[      monIdx]);
    m_regsAddr.config   = smBase + varoffsetof(gaudi2::block_sob_objs, mon_config[   monIdx]);
}

void MonitorG2::configure(Qman::Program& prog, ConfInfo confInfo)
{
    return Monitor::configure<G2pkts>(prog, confInfo);
}

uint32_t MonitorG2::buildConfVal(unsigned soIdx, unsigned numWrtM1, CqEn cqEn, LongSobEn longSobEn, LbwEn lbwEn)
{
    //    Field Name                 |  Bits |   Comments
    //------------------------------------------------------------------------------------------
    //    LONG_SOB MON_CONFIG        |    [0]|Indicates that the monitor monitors 60bit SOBs
    //    CQ EN MON_CONFIG           |    [4]|Indicates the monitor is associated with a completion queue
    //    NUM_WRITES MON_CONFIG      | [5..6]|“0”: single write, “1”: 2 writes, “2”: 3 writes, “3”: 4 writes.
    //    LBW_EN MON_CONFIG          |    [8]|Indicates that a LBW message should be sent post a write to the CQ. Relevant only if CQ_EN is set
    //    LONG_HIGH_GROUP MON_CONFIG |   [31]|Defined which SOB’s would be used for 60xbit count “0”: Lower 4xSOB’s “1”: Upper 4xSOB’s
    //    SID_MSB MON_CONFIG         |[19:16]|Extended SID to monitor groups 256-1023

    gaudi2::sob_objs::reg_mon_config mc {};

    mc.long_sob = (longSobEn == LongSobEn::on);
    mc.cq_en    = (cqEn      == CqEn::on);
    mc.lbw_en   = (lbwEn     == LbwEn::on);
    mc.wr_num   = numWrtM1;
    mc.msb_sid  = (soIdx >> 3) >> 8;
    //    long_high_group : 1;

    return mc._raw;
}
uint32_t MonitorG2::buildArmVal(uint32_t soIdx, uint16_t sod, CompType compType)
{
    return buildArmVal(soIdx, sod, ~(1 << (soIdx % Scal::c_sync_object_group_size)), compType);
}

uint32_t MonitorG2::buildArmVal(uint32_t soIdx, uint16_t sod, uint8_t mask, CompType compType)
{
    //    Field Name                 |  Bits  |   Comments
    //------------------------------------------------------------------------------------------
    //    SID                        |[0..7]  | SOB group id
    //    MASK                       |[8..15] | 0: monitor
    //    SOP                        |   [16] | 1: ==  0: >=
    //    SOD                        |[17..31]| value


    gaudi2::sob_objs::reg_mon_arm ma {};

    ma.sid  = (soIdx >> 3) & 0xFF;
    ma.mask = mask;
    ma.sop  = (compType == CompType::BIG_EQUAL) ? 0 : 1;
    ma.sod  = sod;

    LOG_DEBUG(SCAL, "arm val {:#x}", ma._raw);
    return ma._raw;
}

uint32_t MonitorG2::setLbwEn(uint32_t org, bool lbwVal)
{
    gaudi2::sob_objs::reg_mon_config mc;

    mc._raw   = org;
    mc.lbw_en = lbwVal;

    return mc._raw;
}

/********************************************************************************************/
/*                                         G3                                               */
/********************************************************************************************/
MonitorG3::MonitorG3(uint64_t smBase, uint32_t monIdx, uint8_t smIdx) : Monitor(smBase)
{
    if ((smIdx % Scal_Gaudi3::c_sync_managers_per_hdcores) == 0)
    {
        m_regsAddr.payAddrL = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_addrl_0[monIdx]);
        m_regsAddr.payAddrH = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_addrh_0[monIdx]);
        m_regsAddr.payData  = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_data_0[ monIdx]);
        m_regsAddr.arm      = smBase + varoffsetof(gaudi3::block_sob_objs, mon_arm_0[      monIdx]);
        m_regsAddr.config   = smBase + varoffsetof(gaudi3::block_sob_objs, mon_config_0[   monIdx]);
    }
    else
    {
        m_regsAddr.payAddrL = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_addrl_1[monIdx]);
        m_regsAddr.payAddrH = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_addrh_1[monIdx]);
        m_regsAddr.payData  = smBase + varoffsetof(gaudi3::block_sob_objs, mon_pay_data_1[ monIdx]);
        m_regsAddr.arm      = smBase + varoffsetof(gaudi3::block_sob_objs, mon_arm_1[      monIdx]);
        m_regsAddr.config   = smBase + varoffsetof(gaudi3::block_sob_objs, mon_config_1[   monIdx]);
    }
}

void MonitorG3::configure(Qman::Program& prog, ConfInfo confInfo)
{
    return Monitor::configure<G3pkts>(prog, confInfo);
}

uint32_t MonitorG3::buildConfVal(unsigned soIdx, unsigned numWrtM1, CqEn cqEn, LongSobEn longSobEn, LbwEn lbwEn, uint8_t smIdx)
{
    if ((smIdx % Scal_Gaudi3::c_sync_managers_per_hdcores) == 0)
    {
        gaudi3::sob_objs::reg_mon_config_0 mc {};
        mc.long_sob = (longSobEn == LongSobEn::on);
        mc.cq_en    = (cqEn      == CqEn::on);
        mc.lbw_en   = (lbwEn     == LbwEn::on);
        mc.wr_num   = numWrtM1;
        mc.msb_sid  = (soIdx >> 3) >> 8;
        //    long_high_group : 1;

        return mc._raw;
    }
    else
    {
        gaudi3::sob_objs::reg_mon_config_1 mc {};
        mc.long_sob = (longSobEn == LongSobEn::on);
        mc.cq_en    = (cqEn      == CqEn::on);
        mc.lbw_en   = (lbwEn     == LbwEn::on);
        mc.wr_num   = numWrtM1;
        mc.msb_sid  = (soIdx >> 3) >> 8;
        //    long_high_group : 1;

        return mc._raw;
    }
}

uint32_t MonitorG3::buildArmVal(uint32_t soIdx, uint16_t sod, CompType compType)
{
    return buildArmVal(soIdx, sod, ~(1 << (soIdx % Scal::c_sync_object_group_size)), compType);
}

uint32_t MonitorG3::buildArmVal(uint32_t soIdx, uint16_t sod, uint8_t mask, CompType compType)
{
    //    Field Name                 |  Bits  |   Comments
    //------------------------------------------------------------------------------------------
    //    SID                        |[0..7]  | SOB group id
    //    MASK                       |[8..15] | 0: monitor
    //    SOP                        |   [16] | 1: ==  0: >=
    //    SOD                        |[17..31]| value

    gaudi3::sob_objs::reg_mon_arm_0 ma {};

    ma.sid  = (soIdx >> 3) & 0xFF;
    ma.mask = mask;
    ma.sop  = (compType == CompType::BIG_EQUAL) ? 0 : 1;
    ma.sod  = sod;

    LOG_DEBUG(SCAL, "g3 arm val {:#x}", ma._raw);
    return ma._raw;
}

uint32_t MonitorG3::setLbwEn(uint32_t org, bool lbwVal)
{
    gaudi3::sob_objs::reg_mon_config_0 mc;

    mc._raw   = org;
    mc.lbw_en = lbwVal;

    return mc._raw;
}
