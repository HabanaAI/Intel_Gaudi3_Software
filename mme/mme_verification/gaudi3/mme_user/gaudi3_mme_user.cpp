#include "gaudi3_mme_user.h"
#include "gaudi3/gaudi3_utils.h"

// HW registers
#include "gaudi3/mme.h"
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "gaudi3/asic_reg_structs/acc_regs.h"
#include "gaudi3/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"
#include "gaudi3/asic_reg_structs/qman_regs.h"
#include "memory_config.h"

// coral includes
#include "coral_user_program.h"
#include "coral_user_utils.h"

using namespace MmeCommon;

namespace gaudi3
{
typedef enum
{
    e_mme_operand_0 = (1 << 0),
    e_mme_operand_1 = (1 << 1),
    e_mme_operand_2 = (1 << 2),
    e_mme_operand_3 = (1 << 3),
} EMmeInputOperand;

unsigned Gaudi3MmeUser::getAguInAMask(Mme::Desc& desc)
{
    return e_mme_operand_0 | e_mme_operand_1;
}

unsigned Gaudi3MmeUser::getAguInBMask(Mme::Desc& desc)
{
    return e_mme_operand_2 | e_mme_operand_3;
}

bool Gaudi3MmeUser::doesCmdExecutesOperand(const Mme::Desc* desc, const Mme::MmeCmd& cmd, const bool isA)
{
    MME_ASSERT(0, "not implemented yet");
    return false;
}

uint64_t Gaudi3MmeUser::getMmeCtrlBase()
{
    return mmHD0_MME_CTRL_LO_BASE;
}

const std::string getDualDieMode()
{
    const char* dual_die_pldm_env = getenv("DUAL_DIE_PLDM");
    std::string dualDieMode;
    if (dual_die_pldm_env != nullptr)
    {
        dualDieMode = std::string(dual_die_pldm_env);
        std::transform(dualDieMode.cbegin(), dualDieMode.cend(), dualDieMode.begin(), ::tolower);
        if (dualDieMode == "true" || dualDieMode == "1") dualDieMode = "wide"; // keep backward compatibility
        MME_ASSERT(dualDieMode == "wide" || dualDieMode == "narrow", "wrong DUAL_DIE_PLDM value, should be narrow or wide");
    }
    return dualDieMode;
}

unsigned Gaudi3MmeUser::getMmeQueueId(unsigned mmeIdx, unsigned stream)
{
    // quick hack to fix dual-die pldm image with half compute.
    // narrow flavor : active MMEs: MME0, MME1, MME7, MME8
    // wide flavor   : active MMEs: MME0, MME2, MME4, MME6
    const std::string dualDieMode = getDualDieMode();
    switch (stream)
    {
        case 0:
            switch (mmeIdx)
            {
                case 0:
                    return GAUDI3_SIM_QUEUE_ID_DCORE0_MME_0;
                case 1:
                    if (dualDieMode == "wide")
                    {
                        return GAUDI3_SIM_QUEUE_ID_DCORE1_MME_0;
                    }
                    else
                        return GAUDI3_SIM_QUEUE_ID_DCORE0_MME_1;
                case 2:
                    if (dualDieMode == "wide")
                    {
                        return GAUDI3_SIM_QUEUE_ID_DCORE2_MME_0;
                    }
                    else if (dualDieMode == "narrow")
                    {
                        return GAUDI3_SIM_QUEUE_ID_DCORE3_MME_0;
                    }
                    else
                        return GAUDI3_SIM_QUEUE_ID_DCORE1_MME_0;
                case 3:
                    if (dualDieMode == "wide")
                    {
                        return GAUDI3_SIM_QUEUE_ID_DCORE3_MME_0;
                    }
                    else if (dualDieMode == "narrow")
                    {
                        return GAUDI3_SIM_QUEUE_ID_DCORE3_MME_1;
                    }
                    else
                    return GAUDI3_SIM_QUEUE_ID_DCORE1_MME_1;
                case 4:
                    return GAUDI3_SIM_QUEUE_ID_DCORE2_MME_0;
                case 5:
                    return GAUDI3_SIM_QUEUE_ID_DCORE2_MME_1;
                case 6:
                    return GAUDI3_SIM_QUEUE_ID_DCORE3_MME_0;
                case 7:
                    return GAUDI3_SIM_QUEUE_ID_DCORE3_MME_1;
                default:
                    MME_ASSERT(0, "should not get here");
                    return 0;
            }
        default:
            MME_ASSERT(0, "should not get here");
            return 0;
    }
}

void Gaudi3MmeUser::pushFenceCmd(CPProgram& prog, unsigned fenceIdx, unsigned incWaitVal)
{
    prog.addCommandsBack(CPCommand::Fence(fenceIdx, incWaitVal, incWaitVal));
}

void Gaudi3MmeUser::pushWaitCmd(CPProgram& prog, unsigned waitCycles, unsigned waitValue, unsigned waitIdx)
{
    prog.addCommandsBack(CPCommand::Wait(waitCycles, waitValue, waitIdx));
}

void Gaudi3MmeUser::addMmeLfsrInitSequence(CPProgram& prog, unsigned seed, const LfsrData& lfsrData, bool configLfsr)
{
    uint64_t masterAPBase = 0;
    uint64_t slaveAPBase = 0;
    gaudi3::Mme::EMmeCore Hdcore, Dcore;

    switch (prog.getQId())
    {
            // need an update from corals, missing queue idx.
        case GAUDI3_SIM_QUEUE_ID_DCORE0_MME_0:
            masterAPBase = mmHD0_MME0_ACC_BASE;
            slaveAPBase = mmHD0_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE0_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME0;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE0_MME_1:
            masterAPBase = mmHD1_MME0_ACC_BASE;
            slaveAPBase = mmHD1_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE0_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME1;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE1_MME_0:
            masterAPBase = mmHD2_MME0_ACC_BASE;
            slaveAPBase = mmHD2_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE1_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME0;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE1_MME_1:
            masterAPBase = mmHD3_MME0_ACC_BASE;
            slaveAPBase = mmHD3_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE1_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME1;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE2_MME_0:
            masterAPBase = mmHD4_MME0_ACC_BASE;
            slaveAPBase = mmHD4_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE2_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME0;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE2_MME_1:
            masterAPBase = mmHD5_MME0_ACC_BASE;
            slaveAPBase = mmHD5_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE2_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME1;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE3_MME_0:
            masterAPBase = mmHD6_MME0_ACC_BASE;
            slaveAPBase = mmHD6_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE3_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME0;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE3_MME_1:
            masterAPBase = mmHD7_MME0_ACC_BASE;
            slaveAPBase = mmHD7_MME1_ACC_BASE;
            Dcore = gaudi3::Mme::DCORE3_MME;
            Hdcore = gaudi3::Mme::HDCORE_MME1;
            break;
        default:
            MME_ASSERT(0, "should not get here");
            return;
    }

    unsigned masterCore = gaudi3::Mme::MME_PAIR_SIZE * Hdcore + gaudi3::Mme::HDCORE_SIZE * Dcore;
    unsigned slaveCore = masterCore + 1;

    prog.addCommandsBack(CPCommand::MsgLong(masterAPBase + offsetof(block_acc, ap_lfsr_poly),  // addr
                                            lfsrData.lfsrPolynomial[masterCore]));  // value,

    prog.addCommandsBack(CPCommand::MsgLong(slaveAPBase + offsetof(block_acc, ap_lfsr_poly),  // addr
                                            lfsrData.lfsrPolynomial[slaveCore]));  // value,
    if (configLfsr)
    {
        for (int i = 0; i < gaudi3::Mme::c_mme_lfsr_seeds_nr; i++)
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

void Gaudi3MmeUser::addClipInfInputConfig(CPProgram& prog, const bool clipInfIn)
{
    uint64_t masterAPBase = 0;
    uint64_t slaveAPBase = 0;
    uint64_t ctrlBase = 0;
    switch (prog.getQId())
    {
            // need an update from corals, missing queue idx.
        case GAUDI3_SIM_QUEUE_ID_DCORE0_MME_0:
            masterAPBase = mmHD0_MME0_ACC_BASE;
            slaveAPBase = mmHD0_MME1_ACC_BASE;
            ctrlBase = mmHD0_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE0_MME_1:
            masterAPBase = mmHD1_MME0_ACC_BASE;
            slaveAPBase = mmHD1_MME1_ACC_BASE;
            ctrlBase = mmHD1_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE1_MME_0:
            masterAPBase = mmHD2_MME0_ACC_BASE;
            slaveAPBase = mmHD2_MME1_ACC_BASE;
            ctrlBase = mmHD2_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE1_MME_1:
            masterAPBase = mmHD3_MME0_ACC_BASE;
            slaveAPBase = mmHD3_MME1_ACC_BASE;
            ctrlBase = mmHD3_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE2_MME_0:
            masterAPBase = mmHD4_MME0_ACC_BASE;
            slaveAPBase = mmHD4_MME1_ACC_BASE;
            ctrlBase = mmHD4_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE2_MME_1:
            masterAPBase = mmHD5_MME0_ACC_BASE;
            slaveAPBase = mmHD5_MME1_ACC_BASE;
            ctrlBase = mmHD5_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE3_MME_0:
            masterAPBase = mmHD6_MME0_ACC_BASE;
            slaveAPBase = mmHD6_MME1_ACC_BASE;
            ctrlBase = mmHD6_MME_CTRL_LO_BASE;
            break;
        case GAUDI3_SIM_QUEUE_ID_DCORE3_MME_1:
            masterAPBase = mmHD7_MME0_ACC_BASE;
            slaveAPBase = mmHD7_MME1_ACC_BASE;
            ctrlBase = mmHD7_MME_CTRL_LO_BASE;
            break;
        default:
            MME_ASSERT(0, "should not get here");
            return;
    }

    acc::reg_misc acc_misc;
    acc_misc._raw = 0;
    acc_misc.fp_clip_inf_input = clipInfIn;
    prog.addCommandsBack(CPCommand::MsgLong(masterAPBase + offsetof(block_acc, misc),  // addr
                                            acc_misc._raw));  // value,

    prog.addCommandsBack(CPCommand::MsgLong(slaveAPBase + offsetof(block_acc, misc),  // addr
                                            acc_misc._raw));  // value,

    mme_ctrl_lo::reg_misc ctrl_misc;
    ctrl_misc._raw = 0;
    ctrl_misc.eu_a_clip_inf_input = clipInfIn;
    ctrl_misc.eu_b_clip_inf_input = clipInfIn;
    ctrl_misc.sb_tr_mode_h_min_a = 0x1f;  //  default value for field
    ctrl_misc.sb_tr_mode_h_min_b = 0x1f;  //  default value for field
    prog.addCommandsBack(CPCommand::MsgLong(ctrlBase + offsetof(block_mme_ctrl_lo, misc),  // addr
                                            ctrl_misc._raw));  // value,
}

void Gaudi3MmeUser::addMessageBarrier(CPProgram& prog)
{
    prog.addCommandsBack(CPCommand::Nop(true));  // message barrier command
}

//  define global configs to the machine that are configured only once.
//  these values are configured by the driver, since simulator runs without driver they need to be configured manually.
std::list<CPProgram> Gaudi3MmeUser::createSimulatorProgram(const std::list<CPProgram>& progs, unsigned seed)
{
    // redundant lines currently not implemented in Gaudi3.
    // once it is introduced to the simulator this code will be enabled.
    return progs;
}

void Gaudi3MmeUser::setSoForPowerTest(CPProgram& prog, bool isPowerProg)
{
    const SyncInfo& groupSI = m_syncObjectManager->getCurrentGroupSyncObj();
    uint64_t SOAddr = m_syncObjectManager->getGroupSoAddress();
    if (!isPowerProg)
    {
        prog.addCommandsBack(CPCommand::MsgLong(SOAddr, groupSI.outputSOTarget));
    }
    else
    {
        prog.addCommandsBack(CPCommand::WReg32(varoffsetof(block_qman, cp_fence3_rdata), 1));
        prog.addPostExecWrite(std::pair<uint64_t, uint32_t>(SOAddr, groupSI.outputSOTarget));
        prog.setPredModeInfo(1);
    }
}

void Gaudi3MmeUser::addReductionToPowerDesc(Mme::Desc& powerDesc, MmeCommon::EMmeReductionOp op)
{
    auto dtype = ConvertDataTypeFromGaudi3((gaudi3::Mme::EMmeDataType) powerDesc.header.dataTypeOut);
    auto reduction = MmeMemoryConfigMgr::getMmeMemoryConfigMgr(e_mme_Gaudi3);
    reduction->setReductionParams(op, e_mme_reduction_round_nr, dtype, false);
    reduction->setReductionUserBits(powerDesc);
}

void Gaudi3MmeUser::removeReductionToPowerDesc(Mme::Desc& powerDesc)
{
    auto reduction = MmeMemoryConfigMgr::getMmeMemoryConfigMgr(e_mme_Gaudi3);
    reduction->setEmptyReductionUserBits(powerDesc);
}

void Gaudi3MmeUser::zeroSignalForPowerTest(ActivationVec& activation)
{
    for (auto& act : activation)
    {
        for (unsigned descIdx = 0; descIdx < act.descriptors.size(); descIdx++)
        {
            act.getDesc(descIdx).syncObject.signalEn0=0;
            act.getDesc(descIdx).syncObject.signalEn1=0;
        }
    }
}

}  // namespace gaudi3
