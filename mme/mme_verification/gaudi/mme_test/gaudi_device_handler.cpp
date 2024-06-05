#include "gaudi_device_handler.h"

// Coral includes
#include "coral_user_driver_device.h"
#include "coral_user_sim_device.h"

// Spec includes
#include "gaudi/asic_reg_structs/qman_regs.h"
#include "gaudi/mme.h"

#include "gaudi/gaudi_mme_hal_reader.h"
#include "print_utils.h"

namespace gaudi
{
GaudiDeviceHandler::GaudiDeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) :
    DeviceHandler(devA, devB, deviceIdxs, gaudi::MmeHalReader::getInstance())
{}

SimDeviceBase* GaudiDeviceHandler::getChipSimDevice(const uint64_t pqBaseAddr)
{
    return pqBaseAddr ? new SimDevice(pqBaseAddr) : new SimDevice();
}

coral::DriverDeviceBase* GaudiDeviceHandler::getChipDriverDevice()
{
    if (m_chipAlternative)
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Creating chipDevice with type: gaudiM\n");
        return new DriverDevice(HLV_SIM_GAUDI_HL2000M);
    }
    else
    {
        atomicColoredPrint(COLOR_YELLOW, "INFO: Creating chipDevice with type: gaudi\n");
        return new DriverDevice(HLV_SIM_GAUDI);
    }
}

void GaudiDeviceHandler::configureMmeSniffer(const unsigned mmeIdx,
                                             CoralMmeHBWSniffer& hbwSniffer,
                                             CoralMmeLBWSniffer& lbwSniffer)
{
    for (int eu = 0; eu < Mme::MME_MASTERS_NR; eu++)
    {
        unsigned euIdx = mmeIdx * Mme::MME_MASTERS_NR + eu;
        const std::string token = "DCORE" + std::to_string(euIdx) + "-MME0";
        hbwSniffer.addReadModule(token, true);
        hbwSniffer.addWriteModule(token, true);
    }

    uint64_t qmanBase;
    if (mmeIdx == Mme::MME_CORE_SOUTH_MASTER)
    {
        qmanBase = mmMME0_QM_BASE;
    }
    else
    {
        qmanBase = mmMME2_QM_BASE;
    }

    const uint64_t lbwRangesStartEnd[][2] = {
        {offsetof(block_qman, glbl_cfg0), offsetof(block_qman, cp_msg_base0_addr_lo)},
        {offsetof(block_qman, cp_fence0_rdata), offsetof(block_qman, cp_barrier_cfg)}};

    for (const auto& startEnd : lbwRangesStartEnd)
    {
        lbwSniffer.addSniffingRange(qmanBase + startEnd[0], startEnd[1] - startEnd[0]);
    }
}
}  // namespace gaudi
