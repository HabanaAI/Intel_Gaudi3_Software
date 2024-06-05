#include "gaudi3_sync_object_manager.h"

//  gaudi3 spec includes
#include "gaudi3/mme.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"

//  coral includes
#include "cluster_config.h"
#include "coral_user_program.h"

#include "src/gaudi3/gaudi3_mme_hal_reader.h"

namespace gaudi3
{
Gaudi3SyncObjectManager::Gaudi3SyncObjectManager(uint64_t smBase, unsigned mmeLimit) :
    SyncObjectManager(smBase, mmeLimit, gaudi3::MmeHalReader::getInstance())
{}

void Gaudi3SyncObjectManager::addMonitor(CPProgram& prog,
                                         const SyncInfo& syncInfo,
                                         uint32_t monitorIdx,
                                         uint64_t agentAddr,
                                         unsigned int agentPayload)
{
    // mon arm - addr low
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_addrl_0[monitorIdx]),  // address
                           (uint32_t) agentAddr));  // value

    // mon arm - addr high
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_addrh_0[monitorIdx]),  // address
                           (uint32_t)(agentAddr >> 32)));  // value

    // mon arm - value
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_data_0[monitorIdx]),  // address
                           agentPayload));  // value

    // mon config - config extra monitor attributes
    sob_objs::reg_mon_config_0 mon_config = {0};

    mon_config.long_sob = 0;
    mon_config.cq_en = 0;
    mon_config.lbw_en = 0;
    mon_config.msb_sid = syncInfo.outputSOIdx >> 8;

    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_config_0[monitorIdx]),  // address
                           mon_config._raw));  // value

    // mon arm - arm
    sob_objs::reg_mon_arm_0 mon_arm = {0};
    mon_arm.mask = ~(syncInfo.outputSOSel);
    mon_arm.sid = syncInfo.outputSOIdx;
    mon_arm.sod = syncInfo.outputSOTarget;
    mon_arm.sop = ClusterCfg::e_equal;
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_arm_0[monitorIdx]),  // address
                           mon_arm._raw));  // value
}

uint64_t Gaudi3SyncObjectManager::getSoAddress(unsigned soIdx) const
{
    return m_smBase + varoffsetof(block_sob_objs, sob_obj_0[soIdx]);
}

unsigned Gaudi3SyncObjectManager::uploadDmaQidFromStream(unsigned int stream)
{
    // we do not need to offset first available monitor with upload dma qid
    return 0;
}
}  // namespace gaudi3
