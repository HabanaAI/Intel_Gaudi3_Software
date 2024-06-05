#include "gaudi2_mme_user.h"

// HW registers
#include "gaudi2/mme.h"
#include "gaudi2/asic_reg/gaudi2_blocks.h"
#include "gaudi2/asic_reg_structs/acc_regs.h"
#include "gaudi2/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"
#include "include/gaudi2/gaudi2_utils.h"
#include "memory_config.h"

// coral includes
#include "coral_user_program.h"
#include "coral_user_utils.h"

using namespace MmeCommon;

namespace gaudi2
{
unsigned Gaudi2MmeUser::getAguInAMask(Gaudi2::Mme::Desc& desc)
{
    return desc.header.aguReadsA;
}

unsigned Gaudi2MmeUser::getAguInBMask(Gaudi2::Mme::Desc& desc)
{
    return desc.header.aguReadsB;
}

bool Gaudi2MmeUser::doesCmdExecutesOperand(const Gaudi2::Mme::Desc* desc,
                                           const Gaudi2::Mme::MmeCmd& cmd,
                                           const bool isA)
{
    const std::vector<std::pair<uint32_t, uint32_t>> sbRouting = {
        {desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb0En, desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb0Sel},
        {desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb1En, desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb1Sel},
        {desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb2En, desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb2Sel},
        {desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb3En, desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb3Sel},
        {desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb4En, desc->ctrl.eus[Gaudi2::Mme::MME_CORE_MASTER].sb4Sel}};

    for (unsigned i = 0; i < sbRouting.size(); ++i)
    {
        auto& pair = sbRouting[i];
        // For each agu we check:
        // 1. If it's enabled in the desc
        // 2. If it's A or B
        // 3. if the cmd executes it.
        // To check if for example the cmd and desc activate operand A, we check that the cmd executes an enabled
        // agu that reads A
        if (pair.first && ((pair.second >> 2) == isA ? 0 : 1) && (cmd.aguIn & (1 << i)))
        {
            return true;
        }
    }

    return false;
}

uint64_t Gaudi2MmeUser::getMmeCtrlBase()
{
    return mmDCORE0_MME_CTRL_LO_BASE;
}

unsigned Gaudi2MmeUser::getMmeQueueId(unsigned mmeIdx, unsigned stream)
{
    switch (stream)
    {
        case 0:
            return mmeIdx == 0 ? GAUDI2_QUEUE_ID_DCORE0_MME_0_0 : GAUDI2_QUEUE_ID_DCORE2_MME_0_0;
        case 1:
            return mmeIdx == 0 ? GAUDI2_QUEUE_ID_DCORE0_MME_0_1 : GAUDI2_QUEUE_ID_DCORE2_MME_0_1;
        case 2:
            return mmeIdx == 0 ? GAUDI2_QUEUE_ID_DCORE0_MME_0_2 : GAUDI2_QUEUE_ID_DCORE2_MME_0_2;
        case 3:
            return mmeIdx == 0 ? GAUDI2_QUEUE_ID_DCORE0_MME_0_3 : GAUDI2_QUEUE_ID_DCORE2_MME_0_3;
        default:
            MME_ASSERT(0, "should not get here");
            return 0;
    }
}

void Gaudi2MmeUser::pushFenceCmd(CPProgram& prog, unsigned fenceIdx, unsigned incWaitVal)
{
    prog.addCommandsBack(CPCommand::Fence(fenceIdx, incWaitVal, incWaitVal));
}

void Gaudi2MmeUser::pushWaitCmd(CPProgram& prog,  unsigned waitCycles, unsigned waitValue, unsigned waitIdx)
{
    prog.addCommandsBack(CPCommand::Wait(waitCycles, waitValue, waitIdx));
}

void Gaudi2MmeUser::addMmeLfsrInitSequence(CPProgram& prog, unsigned seed, const LfsrData& lfsrData, bool configLfsr)
{
    uint64_t masterAPBase;
    uint64_t slaveAPBase;
    Gaudi2::Mme::EMmeCore masterCore;
    Gaudi2::Mme::EMmeCore slaveCore;

    switch (prog.getQId())
    {
        case GAUDI2_QUEUE_ID_DCORE0_MME_0_0:
        case GAUDI2_QUEUE_ID_DCORE0_MME_0_1:
        case GAUDI2_QUEUE_ID_DCORE0_MME_0_2:
        case GAUDI2_QUEUE_ID_DCORE0_MME_0_3:
            // south master
            masterAPBase = mmDCORE0_MME_ACC_BASE;
            slaveAPBase = mmDCORE1_MME_ACC_BASE;
            masterCore = Gaudi2::Mme::MME_CORE_MASTER0;
            slaveCore = Gaudi2::Mme::MME_CORE_SLAVE0;
            break;
        case GAUDI2_QUEUE_ID_DCORE2_MME_0_0:
        case GAUDI2_QUEUE_ID_DCORE2_MME_0_1:
        case GAUDI2_QUEUE_ID_DCORE2_MME_0_2:
        case GAUDI2_QUEUE_ID_DCORE2_MME_0_3:
            // north master
            masterAPBase = mmDCORE2_MME_ACC_BASE;
            slaveAPBase = mmDCORE3_MME_ACC_BASE;
            masterCore = Gaudi2::Mme::MME_CORE_MASTER1;
            slaveCore = Gaudi2::Mme::MME_CORE_SLAVE1;
            break;
        default:
            MME_ASSERT(0, "should not get here");
            return;
    }

    prog.addCommandsBack(CPCommand::MsgLong(masterAPBase + offsetof(block_acc, ap_lfsr_poly),  // addr
                                            lfsrData.lfsrPolynomial[masterCore]));  // value,

    prog.addCommandsBack(CPCommand::MsgLong(slaveAPBase + offsetof(block_acc, ap_lfsr_poly),  // addr
                                            lfsrData.lfsrPolynomial[slaveCore]));  // value,

    if (configLfsr)
    {
        for (int i = 0; i < Gaudi2::Mme::c_mme_lfsr_seeds_nr; i++)
        {
            prog.addCommandsBack(CPCommand::MsgLong(masterAPBase + offsetof(block_acc, ap_lfsr_seed_sel),  // addr
                                                    i));  // value,

            prog.addCommandsBack(CPCommand::MsgLong(slaveAPBase + offsetof(block_acc, ap_lfsr_seed_sel),  // addr
                                                    i));  // value,

            prog.addCommandsBack(CPCommand::MsgLong(masterAPBase + offsetof(block_acc, ap_lfsr_seed_wdata),  // addr
                                                    lfsrData.lfsrRegs[masterCore][i]));  // value,

            prog.addCommandsBack(CPCommand::MsgLong(slaveAPBase + offsetof(block_acc, ap_lfsr_seed_wdata),  // addr
                                                    lfsrData.lfsrRegs[slaveCore][i]));  // value,
        }
    }
}

void Gaudi2MmeUser::addClipInfInputConfig(CPProgram& prog, const bool clipInfIn)

{
    MME_ASSERT(!clipInfIn, "clipInfIn not support in Gaudi2");
}

void Gaudi2MmeUser::addMessageBarrier(CPProgram& prog)
{
    prog.addCommandsBack(CPCommand::Nop(true));  // message barrier command
}

//  define global configs to the machine that are configured only once.
//  these values are configured by the driver, since simulator runs without driver they need to be configured manually.
std::list<CPProgram> Gaudi2MmeUser::createSimulatorProgram(const std::list<CPProgram>& progs, unsigned seed)
{
    std::list<CPProgram> newProgList = progs;
    auto& firstProg = newProgList.front();

    std::array<std::pair<uint32_t, uint32_t>, 4> redun_val = getRandomRedundancyFmaWithBitMask(seed);
    constexpr unsigned cmdNr = 20;
    std::array<std::pair<uint64_t, uint32_t>, cmdNr> cmds = {{
        // south pole
        {mmDCORE0_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en32), redun_val[0].second},
        {mmDCORE1_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en32), redun_val[1].second},
        {mmDCORE0_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en33),
         (redun_val[0].first == 32) ? 0x0 : 0x1},
        {mmDCORE1_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en33),
         (redun_val[1].first == 32) ? 0x0 : 0x1},
        {mmDCORE0_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, eu_isolation_dis), 0x1},
        {mmDCORE1_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, eu_isolation_dis), 0x1},
        {mmDCORE0_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun), redun_val[0].first},
        {mmDCORE1_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun), redun_val[1].first},
        {mmDCORE0_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun_psoc_sel_sec), 0x0},
        {mmDCORE1_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun_psoc_sel_sec), 0x0},
        // north pole
        {mmDCORE2_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en32), redun_val[2].second},
        {mmDCORE3_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en32), redun_val[3].second},
        {mmDCORE2_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en33),
         (redun_val[2].first == 32) ? 0x0 : 0x1},
        {mmDCORE3_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, fma_func_redun_clk_en33),
         (redun_val[3].first == 32) ? 0x0 : 0x1},
        {mmDCORE2_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, eu_isolation_dis), 0x1},
        {mmDCORE3_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, eu_isolation_dis), 0x1},
        {mmDCORE2_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun), redun_val[2].first},
        {mmDCORE3_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun), redun_val[3].first},
        {mmDCORE2_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun_psoc_sel_sec), 0x0},
        {mmDCORE3_MME_CTRL_LO_BASE + offsetof(block_mme_ctrl_lo, redun_psoc_sel_sec), 0x0},
    }};

    unsigned cmdNrToProcess = m_mmeNr == 2 ? cmdNr : (cmdNr / 2);
    // program already exists - so adding the config commands in front (reverse order).
    firstProg.addCommandsFront(CPCommand::Nop(true));
    for (unsigned cmdIdx = 0; cmdIdx < cmdNrToProcess; cmdIdx++)
    {
        firstProg.addCommandsFront(CPCommand::MsgLong(cmds[cmdIdx].first,
                                                      cmds[cmdIdx].second,
                                                      false, /*mb*/
                                                      true /*eb*/));
    }
    return newProgList;
}

void Gaudi2MmeUser::addReductionToPowerDesc(Gaudi2::Mme::Desc& powerDesc, MmeCommon::EMmeReductionOp op)
{
    auto dtype = Gaudi2::ConvertDataTypeFromGaudi2((Gaudi2::Mme::EMmeDataType) powerDesc.header.dataTypeOut);
    auto reduction = MmeMemoryConfigMgr::getMmeMemoryConfigMgr(e_mme_Gaudi2);
    reduction->setReductionParams(op, e_mme_reduction_round_nr, dtype, false);
    reduction->setReductionUserBits(powerDesc);
}

void Gaudi2MmeUser::removeReductionToPowerDesc(Gaudi2::Mme::Desc& powerDesc)
{
    auto dtype = Gaudi2::ConvertDataTypeFromGaudi2((Gaudi2::Mme::EMmeDataType) powerDesc.header.dataTypeOut);
    auto reduction = MmeMemoryConfigMgr::getMmeMemoryConfigMgr(e_mme_Gaudi2);
    reduction->setReductionParams(EMmeReductionOp::e_mme_reduction_none, e_mme_reduction_round_nr, dtype, false);
    reduction->setReductionUserBits(powerDesc);
}

void Gaudi2MmeUser::setSoForPowerTest(CPProgram& prog, bool isPowerProg)
{
    static constexpr unsigned QMAN_REG_BASE = 0xA000;
    const SyncInfo& groupSI = m_syncObjectManager->getCurrentGroupSyncObj();
    uint64_t SOAddr = m_syncObjectManager->getGroupSoAddress();
    Gaudi2::Mme::MmeSyncObjectVal soVal = {0};
    soVal.soValue = 1;
    soVal.soOp = 1;  // Increment
    if (!isPowerProg)
    {
        prog.addCommandsBack(CPCommand::MsgLong(SOAddr, soVal.dw));
    }
    else
    {
        prog.addCommandsBack(CPCommand::WReg32(QMAN_REG_BASE + varoffsetof(block_qman, cp_fence3_rdata[0]), 1));
        prog.addPostExecWrite(std::pair<uint64_t, uint32_t>(SOAddr, soVal.dw));
        prog.setPredModeInfo(1);
    }
}

void Gaudi2MmeUser::zeroSignalForPowerTest(Gaudi2::ActivationVec& activations)
{
    for (auto& act : activations)
    {
        for (auto& desc : act.descriptors)
        {
            desc.syncObject.signalEn0 = 0;
            desc.syncObject.signalEn1 = 0;
        }
    }
}

}  // namespace gaudi2
