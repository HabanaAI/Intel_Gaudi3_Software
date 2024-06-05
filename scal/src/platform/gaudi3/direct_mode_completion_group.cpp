#include "scal_gaudi3.h"
#include "infra/sync_mgr.hpp"

using namespace gaudi3;

Scal_Gaudi3::DirectModeCompletionGroup::DirectModeCompletionGroup(const std::string& name,
                                                                  Scal*              pScal,
                                                                  unsigned           cqIndex,
                                                                  uint64_t           isrRegister,
                                                                  unsigned           isrIndex,
                                                                  unsigned           longSoSmIndex,
                                                                  unsigned           longSoIndex)
: CompletionGroupInterface(true, name, pScal, cqIndex, isrIndex, longSoSmIndex, longSoIndex),
  m_isrRegister(isrRegister)
{
    LOG_TRACE(SCAL, "Created Direct-Mode CG for {} isrRegister {:#x} isrIndex {}", name, isrRegister, isrIndex);
}

bool Scal_Gaudi3::DirectModeCompletionGroup::getInfo(scal_completion_group_info_t& info) const
{
    info.long_so_index = longSoIndex;
    info.long_so_dcore = longSoSmIndex;

    return true;
}

bool Scal_Gaudi3::DirectModeCompletionGroup::getInfo(scal_completion_group_infoV2_t& info) const
{
    memset(&info, 0, sizeof(scal_completion_group_infoV2_t));
    info.isDirectMode  = true;

    info.long_so_index        = longSoIndex;
    info.long_so_sm           = longSoSmIndex;
    info.long_so_sm_base_addr = SyncMgrG3::getSmBase(longSoSmIndex);
    info.tdr_enabled          = compQTdr.enabled;
    info.tdr_value            = compQTdr.enabled ? *(compQTdr.enginesCtr) : -1;
    info.tdr_sos              = compQTdr.sos;
    info.index_in_scheduler   = compQTdr.cqIdx;
    info.sm_base_addr         = compQTdr.enabled ? SyncMgrG3::getSmBase(compQTdr.monSmIdx) : -1;
    info.sos_base             = 0;
    info.sos_num              = 0;
    info.current_value        = *pCounter;
    info.timeoutUs            = scal->getTimeout();
    info.timeoutDisabled      = scal->getTimeoutDisabled();

    return true;
}

std::string Scal_Gaudi3::DirectModeCompletionGroup::getLogInfo(uint64_t target) const
{
    std::string logInfo =
        fmt::format("{}: CQ ({}) #{} of (main) interrupt {:#x} target={:#x} longSoIdx={} (sm {})",
                    __FUNCTION__, name, globalCqIndex, isrIdx, target, longSoIndex, longSoSmIndex);

    return logInfo;
}
