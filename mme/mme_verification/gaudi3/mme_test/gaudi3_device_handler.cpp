#include "gaudi3_device_handler.h"

// Coral includes
#include "coral_user_driver_device.h"
#include "coral_user_sim_device.h"

// Spec includes
#include "gaudi3/asic_reg_structs/acc_regs.h"
#include "gaudi3/asic_reg_structs/mme_ctrl_lo_regs.h"
#include "gaudi3/asic_reg_structs/qman_regs.h"
#include "gaudi3/mme.h"

#include "src/gaudi3/gaudi3_mme_hal_reader.h"
#include "print_utils.h"

namespace gaudi3
{
Gaudi3DeviceHandler::Gaudi3DeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) :
    DeviceHandler(devA, devB, deviceIdxs, gaudi3::MmeHalReader::getInstance())
{}

SimDeviceBase* Gaudi3DeviceHandler::getChipSimDevice(const uint64_t pqBaseAddr)
{
    return pqBaseAddr ? new SimDevice(pqBaseAddr, false, m_dieNr) : new SimDevice(DRAM_PHYS_BASE, false, m_dieNr);
}

coral::DriverDeviceBase* Gaudi3DeviceHandler::getChipDriverDevice()
{
    MME_ASSERT(m_chipAlternative == false, "There is no chip alternative for Gaudi3");
    atomicColoredPrint(COLOR_YELLOW, "INFO: Creating chipDevice with type: Gaudi3\n");
    return new DriverDevice(HLV_SIM_GAUDI3);
}

void Gaudi3DeviceHandler::configureMmeSniffer(const unsigned mmeIdx,
                                              CoralMmeHBWSniffer& hbwSniffer,
                                              CoralMmeLBWSniffer& lbwSniffer)
{
    const std::string token = "MME" + std::to_string(mmeIdx);
    hbwSniffer.addReadModule(token, true);
    hbwSniffer.addWriteModule(token, true);

    uint64_t qmanBase;
    switch (mmeIdx)
    {
        //  needs an update from spec, missing defines
        case 0:
            qmanBase = mmHD0_MME_QM_BASE;
            break;
        case 1:
            qmanBase = mmHD1_MME_QM_BASE;
            break;
        case 2:
            qmanBase = mmHD2_MME_QM_BASE;
            break;
        case 3:
            qmanBase = mmHD3_MME_QM_BASE;
            break;
        case 4:
            qmanBase = mmHD4_MME_QM_BASE;
            break;
        case 5:
            qmanBase = mmHD5_MME_QM_BASE;
            break;
        case 6:
            qmanBase = mmHD6_MME_QM_BASE;
            break;
        case 7:
            qmanBase = mmHD7_MME_QM_BASE;
            break;
        default:
            qmanBase = 0;
            MME_ASSERT(0, "invalid mme idx");
    }

    const uint64_t lbwRangesStartEnd[][2] = {
        {offsetof(block_qman, glbl_cfg0), offsetof(block_qman, cp_msg_base_addr[0])},
        {offsetof(block_qman, cp_fence0_rdata), offsetof(block_qman, cp_barrier_cfg)}};

    for (auto& startEnd : lbwRangesStartEnd)
    {
        lbwSniffer.addSniffingRange(qmanBase + startEnd[0], startEnd[1] - startEnd[0]);
    }
}
}  // namespace gaudi3
