#include "scal_base.h"

void Scal::CompletionGroupInterface::getTimestampInfo(int&      fd,
                                                      unsigned& isrIndex,
                                                      uint64_t& cqCountersHandle,
                                                      unsigned& cqIndex) const
{
    assert(scal != nullptr);

    fd               = scal->getFD();
    isrIndex         = isrIdx;
    cqCountersHandle = scal->m_completionQueuesHandle;
    cqIndex          = globalCqIndex;
}

bool Scal::CompletionGroup::getInfo(scal_completion_group_info_t& info) const
{
    info.scheduler_handle          = Scal::toCoreHandle(scheduler);
    info.index_in_scheduler        = idxInScheduler;
    info.dcore                     = syncManager->smIndex;
    info.sos_base                  = sosBase;
    info.sos_num                   = sosNum;
    info.long_so_dcore             = longSoSmIndex;
    info.long_so_index             = longSoIndex;
    info.current_value             = *(pCounter);
    info.force_order               = force_order;
    info.num_slave_schedulers      = 0;
    for (auto sched : slaveSchedulers)
    {
        info.slave_schedulers[info.num_slave_schedulers]          = Scal::toCoreHandle(sched.scheduler);
        info.index_in_slave_schedulers[info.num_slave_schedulers] = sched.idxInScheduler;
        info.num_slave_schedulers++;
    }

    return true;
}

bool Scal::CompletionGroup::getInfo(scal_completion_group_infoV2_t& info) const
{
    info.isDirectMode              = false;
    info.scheduler_handle          = toCoreHandle(scheduler);
    info.index_in_scheduler        = idxInScheduler;
    info.dcore                     = syncManager->smIndex; // TODO: tdr remove
    info.sm                        = syncManager->smIndex;
    info.sm_base_addr              = monitorsPool->smBaseAddr;
    info.sos_base                  = sosBase;
    info.sos_num                   = sosNum;
    info.long_so_dcore             = longSoSmIndex; // TODO: tdr remove
    info.long_so_sm                = longSoSmIndex;
    info.long_so_sm_base_addr      = longSosPool->smBaseAddr;
    info.long_so_index             = longSoIndex;
    info.current_value             = *pCounter;
    info.tdr_value                 = compQTdr.enabled ? *(compQTdr.enginesCtr) : -1;
    info.tdr_enabled               = compQTdr.enabled;
    info.tdr_sos                   = compQTdr.sos;
    info.timeoutUs                 = scal->getTimeout();
    info.timeoutDisabled           = scal->getTimeoutDisabled();
    info.force_order               = force_order;
    info.num_slave_schedulers      = 0;
    for (auto sched : slaveSchedulers)
    {
        info.slave_schedulers[info.num_slave_schedulers]          = toCoreHandle(sched.scheduler);
        info.index_in_slave_schedulers[info.num_slave_schedulers] = sched.idxInScheduler;
        info.num_slave_schedulers++;
    }

    return true;
}

std::string Scal::CompletionGroup::getLogInfo(uint64_t target) const
{
    std::string schedulerLogInfo("NA (-1)");

    if (scheduler != nullptr)
    {
        schedulerLogInfo = fmt::format("{} ({}) ", scheduler->name, scheduler->cpuId);
    }

    std::string logInfo =
        fmt::format(FMT_COMPILE("{}: CQ ({}) #{} of (main) scheduler {} sm {} interrupt {:#x} target={:#x} "
                    "mon [{}-{}] longSoIdx={} ({} at sm {}) so range [{}-{}]"),
                    __FUNCTION__, name, cqIdx, schedulerLogInfo.c_str(),
                    longSoSmIndex, isrIdx, target,
                    monBase, monBase + monNum / actualNumberOfMonitors - 1,
                    longSoIndex, longSosPool->baseIdx + longSoIndex, longSoSmIndex, sosBase, sosBase + sosNum - 1);

    return logInfo;
}