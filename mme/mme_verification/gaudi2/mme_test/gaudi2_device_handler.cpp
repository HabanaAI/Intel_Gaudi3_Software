#include "gaudi2_device_handler.h"

// Coral includes
#include "coral_user_driver_device.h"
#include "coral_user_sim_device.h"

// Spec includes
#include "gaudi2/asic_reg_structs/acc_regs.h"
#include "gaudi2/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"
#include "gaudi2/mme.h"

#include "src/gaudi2/gaudi2_mme_hal_reader.h"
#include "print_utils.h"

namespace gaudi2
{
Gaudi2DeviceHandler::Gaudi2DeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) :
    DeviceHandler(devA, devB, deviceIdxs, Gaudi2::MmeHalReader::getInstance())
{}

SimDeviceBase* Gaudi2DeviceHandler::getChipSimDevice(const uint64_t pqBaseAddr)
{
    return pqBaseAddr? new SimDevice(pqBaseAddr) : new SimDevice();
}

coral::DriverDeviceBase* Gaudi2DeviceHandler::getChipDriverDevice()
{
    if (m_chipAlternative)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Creating chipDevice with type: Gaudi2B\n");
        return new DriverDevice(HLV_SIM_GAUDI2B);
    }
    else
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Creating chipDevice with type: Gaudi2\n");
        return new DriverDevice(HLV_SIM_GAUDI2);
    }
}

void Gaudi2DeviceHandler::configureMmeSniffer(const unsigned mmeIdx,
                                              CoralMmeHBWSniffer& hbwSniffer,
                                              CoralMmeLBWSniffer& lbwSniffer)
{
    for (int eu = 0; eu < Gaudi2::Mme::MME_CORE_PAIR_SIZE; eu++)
    {
        unsigned euIdx = mmeIdx * Gaudi2::Mme::MME_CORE_PAIR_SIZE + eu;
        const std::string token = "DCORE" + std::to_string(euIdx) + "-MME0";
        hbwSniffer.addReadModule(token, true);
        hbwSniffer.addWriteModule(token, true);
    }

    uint64_t qmanBase;
    if (mmeIdx == Gaudi2::Mme::MME_CORE_MASTER0)
    {
        qmanBase = mmDCORE0_MME_QM_BASE;
    }
    else
    {
        qmanBase = mmDCORE2_MME_QM_BASE;
    }

    const uint64_t lbwRangesStartEnd[][2] = {
        {offsetof(block_qman, glbl_cfg0), offsetof(block_qman, cp_msg_base0_addr_lo)},
        {offsetof(block_qman, cp_fence0_rdata), offsetof(block_qman, cp_barrier_cfg)}};

    for (auto& startEnd : lbwRangesStartEnd)
    {
        lbwSniffer.addSniffingRange(qmanBase + startEnd[0], startEnd[1] - startEnd[0]);
    }
}
}
