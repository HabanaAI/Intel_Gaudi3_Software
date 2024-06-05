#include "gaudi2_sync_object_manager.h"
#include "mme_assert.h"

//  gaudi2 spec includes
#include "gaudi2/mme.h"
#include "gaudi2/asic_reg_structs/sob_objs_regs.h"

//  coral includes
#include "cluster_config.h"
#include "coral_user_program.h"

#include "src/gaudi2/gaudi2_mme_hal_reader.h"

namespace gaudi2
{
Gaudi2SyncObjectManager::Gaudi2SyncObjectManager(uint64_t smBase, unsigned mmeLimit) :
    SyncObjectManager(smBase, mmeLimit, Gaudi2::MmeHalReader::getInstance())
{}

void Gaudi2SyncObjectManager::addMonitor(CPProgram& prog,
                                   const SyncInfo& syncInfo,
                                   uint32_t monitorIdx,
                                   uint64_t agentAddr,
                                   unsigned int agentPayload)
{
    // mon arm - addr low
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_addrl[monitorIdx]),  // address
                           (uint32_t) agentAddr));  // value

    // mon arm - addr high
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_addrh[monitorIdx]),  // address
                           (uint32_t)(agentAddr >> 32)));  // value

    // mon arm - value
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_pay_data[monitorIdx]),  // address
                           agentPayload));  // value

    // mon config - config extra monitor attributes
    sob_objs::reg_mon_config mon_config = {0};

    mon_config.long_sob = 0;
    mon_config.cq_en = 0;
    mon_config.lbw_en = 0;
    mon_config.msb_sid = syncInfo.outputSOIdx >> 8;

    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_config[monitorIdx]),  // address
                           mon_config._raw));  // value

    // mon arm - arm
    sob_objs::reg_mon_arm mon_arm = {0};
    mon_arm.mask = ~(syncInfo.outputSOSel);
    mon_arm.sid = syncInfo.outputSOIdx;
    mon_arm.sod = syncInfo.outputSOTarget;
    mon_arm.sop = ClusterCfg::e_equal;
    prog.addCommandsBack(
        CPCommand::MsgLong(syncInfo.smBase + varoffsetof(block_sob_objs, mon_arm[monitorIdx]),  // address
                           mon_arm._raw));  // value
}

uint64_t Gaudi2SyncObjectManager::getSoAddress(unsigned soIdx) const
{
    return m_smBase + varoffsetof(block_sob_objs, sob_obj[soIdx]);
}

unsigned Gaudi2SyncObjectManager::uploadDmaQidFromStream(unsigned int stream)
{
    return GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0 + stream;
}
}  // namespace gaudi2