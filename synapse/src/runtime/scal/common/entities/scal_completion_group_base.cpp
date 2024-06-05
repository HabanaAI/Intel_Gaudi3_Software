#include "scal_completion_group_base.hpp"

#include "log_manager.h"

#include "runtime/scal/common/scal_event.hpp"

#include "timer.h"


ScalCompletionGroupBase::ScalCompletionGroupBase(scal_handle_t                      devHndl,
                                                 const std::string&                 name,
                                                 const common::DeviceInfoInterface* pDeviceInfoInterface)
: m_devHndl(devHndl),
  m_name(name),
  m_cgHndl(nullptr),
  m_cgInfo {},
  m_completionTarget(0),
  m_lastUserCompletionTarget(0),
  m_tdrInfo {},
  m_pDeviceInfoInterface(pDeviceInfoInterface)
{
}

/*
 ***************************************************************************************************
 *   @brief init() - init the completion group
 *                   get handle from scal, get completion group info
 *
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalCompletionGroupBase::init()
{
    ScalRtn rc = scal_get_completion_group_handle_by_name(m_devHndl, m_name.c_str(), &m_cgHndl);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                     "devHndl 0x{:x} scal_get_completion_group_handle_by_name {} failed with rc {}",
                     TO64(m_devHndl),
                     m_name,
                     rc);
        return synFail;
    }

    synStatus status = getCurrentCgInfo(m_cgInfo);
    if (status != synSuccess)
    {
        return status;
    }

    return synSuccess;
}

uint64_t ScalCompletionGroupBase::getLongSoAddress()
{
    if (m_pDeviceInfoInterface == nullptr)
    {
        LOG_ERR(SYN_STREAM, "Operation not supported");
        return synFail;
    }

    return m_pDeviceInfoInterface->getSosAddr(m_cgInfo.long_so_sm, m_cgInfo.long_so_index);
}

/*
 ***************************************************************************************************
 *   @brief increaseBarrierCount() - increment the m_completionGroupBarriers
 *          called everytime barrier is sent by the scal stream
 *
 *   @return new LongSo
 *
 ***************************************************************************************************
 */
ScalLongSyncObject ScalCompletionGroupBase::getIncrementedLongSo(bool isUserReq, uint64_t targetOffset /* = 1 */)
{
    if (targetOffset != 0)
    {
        m_completionTarget += targetOffset;
        _setExpectedCounter();

        if (isUserReq)
        {
            m_lastUserCompletionTarget = m_completionTarget.load();
        }

        LOG_TRACE(SYN_STREAM,
                  "devHndl 0x{:x} name {} longSo index {} incremented by {} "
                  "target 0x{:x} (userTarget 0x{:x}) isUserReq {}",
                  TO64(m_devHndl),
                  m_name,
                  m_cgInfo.long_so_index,
                  targetOffset,
                  m_completionTarget.load(),
                  m_lastUserCompletionTarget.load(),
                  isUserReq);
    }

    return {m_cgInfo.long_so_index, m_completionTarget};
}

/*
 ***************************************************************************************************
 *   @brief getTargetLongSo() - return LonSo advanced by targetOffset
 *                              does not change internal data structures
 *   @param targetOffset - an offset to created LongSo advanced by targetOffset
 *
 *   @return new advances LongSo
 *
 ***************************************************************************************************
 */
ScalLongSyncObject ScalCompletionGroupBase::getTargetLongSo(uint64_t targetOffset) const
{
    return {m_cgInfo.long_so_index, m_completionTarget + targetOffset};
}

/*
 ***************************************************************************************************
 *   @brief wait() - wait for a given completionTarget value with timeout of timeoutMicroSec time
 *
 *   @param - completionTarget, timeout
 *   @return new value
 *   Note: this function doesn't require to be called under a lock
 ***************************************************************************************************
 */
synStatus ScalCompletionGroupBase::longSoWait(const ScalLongSyncObject& rLongSo,
                                              uint64_t                  timeoutMicroSec,
                                              bool                      alwaysWaitForInterrupt) const
{
    if (rLongSo.m_index != m_cgInfo.long_so_index)
    {
        LOG_ERR(
            SYN_STREAM,
            "Unrecognized longSo index detected m_cgHndl 0x{:x} longSo.m_index {} m_cgInfo.long_so_index {:#x}",
            TO64(m_cgHndl),
            rLongSo.m_index,
            m_cgInfo.long_so_index);

        return synInvalidEventHandle;
    }

    // The target is expected to be the number of barriers sent for a specific completion group
    if (rLongSo.m_targetValue > m_completionTarget)
    {
        LOG_ERR(SYN_STREAM,
                     "ScalCompletionGroup::wait completionTarget 0x{:x} is bigger than {}",
                     rLongSo.m_targetValue,
                     m_completionTarget.load());

        return synInvalidEventHandle;
    }

    LOG_TRACE(SYN_STREAM,
                   "{}: name {} m_cgHndl 0x{:x} long-SO [index {} target 0x{:x}] timeout {} {}",
                   HLLOG_FUNC,
                   m_name.c_str(),
                   TO64(m_cgHndl),
                   rLongSo.m_index,
                   rLongSo.m_targetValue,
                   timeoutMicroSec,
                   _getAdditionalPrintInfo());

    ScalRtn rc = 0;
    if (alwaysWaitForInterrupt)
    {
        rc = scal_completion_group_wait_always_interupt(m_cgHndl,
                                                        rLongSo.m_targetValue,
                                                        std::max((uint64_t)1, timeoutMicroSec));
    }
    else
    {
        rc = scal_completion_group_wait(m_cgHndl, rLongSo.m_targetValue, timeoutMicroSec);
    }
    LOG_DEBUG(SYN_STREAM,
                   "{}: name {} m_cgHndl 0x{:x} index {} target 0x{:x} timeout {} done with scal status {:x}",
                   HLLOG_FUNC,
                   m_name,
                   TO64(m_cgHndl),
                   rLongSo.m_index,
                   rLongSo.m_targetValue,
                   timeoutMicroSec,
                   rc);
    if (rc == SCAL_SUCCESS)
    {
        return synSuccess;
    }

    if ((rc == SCAL_TIMED_OUT) && (timeoutMicroSec != SCAL_FOREVER))
    {
        LOG_DEBUG(SYN_STREAM,
                       "{} SCAL_TIMED_OUT: name {} m_cgHndl 0x{:x} index {} target 0x{:x} timeout {} done "
                       "with scal status {:x}",
                       HLLOG_FUNC,
                       m_name.c_str(),
                       TO64(m_cgHndl),
                       rLongSo.m_index,
                       rLongSo.m_targetValue,
                       timeoutMicroSec,
                       rc);

        return synBusy;
    }

    LOG_ERR(SYN_STREAM,
                 "{}: devHndl 0x{:x} m_cgHndl 0x{:x} {} scal_completion_group_wait failed with rc {}. Most "
                 "likely lkd killed the device, please check dmesg",
                 HLLOG_FUNC,
                 TO64(m_devHndl),
                 TO64(m_cgHndl),
                 m_name,
                 rc);

    return synDeviceReset;
}

synStatus ScalCompletionGroupBase::longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec) const
{
    return longSoWait({ m_cgInfo.long_so_index, isUserReq ? m_lastUserCompletionTarget : m_completionTarget },
                      timeoutMicroSec);
};

ScalLongSyncObject ScalCompletionGroupBase::getLastTarget(bool isUserReq) const
{
    return { m_cgInfo.long_so_index, isUserReq ? m_lastUserCompletionTarget : m_completionTarget };
}

synStatus ScalCompletionGroupBase::getCurrentCgInfo(scal_completion_group_infoV2_t& cgInfo)
{
    int rc = scal_completion_group_get_infoV2(m_cgHndl, &cgInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                     "devHndl 0x{:x} m_cgHndl {} scal_completion_group_get_infoV2 failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_cgHndl),
                     rc);

        return synFail;
    }

    return synSuccess;
}

bool ScalCompletionGroupBase::isForceOrdered()
{
    scal_completion_group_infoV2_t cgInfo;

    if (getCurrentCgInfo(cgInfo) != synSuccess)
    {
        return false;
    }

    return cgInfo.force_order;
}

void ScalCompletionGroupBase::longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const
{
    rLongSo = {m_cgInfo.long_so_index, isUserReq ? m_lastUserCompletionTarget : m_completionTarget };
    LOG_TRACE(SYN_STREAM,
                   "devHndl 0x{:x} name {} Recording longSo index {} value {} (isUserReq {})",
                   TO64(m_devHndl),
                   m_name,
                   rLongSo.m_index,
                   rLongSo.m_targetValue,
                   isUserReq);
}

synStatus ScalCompletionGroupBase::eventRecord(bool isUserReq, ScalEvent& scalEvent) const
{
    longSoRecord(isUserReq, scalEvent.longSo);

    LOG_INFO(SYN_STREAM,
                  "devHndl 0x{:x} name {} Recording event longso index {} value {} (isUserReq {} collectTime {})",
                  TO64(m_devHndl),
                  m_name,
                  scalEvent.longSo.m_index,
                  scalEvent.longSo.m_targetValue,
                  isUserReq,
                  scalEvent.collectTime);

    if (scalEvent.collectTime)
    {
        const ScalRtn rc = scal_completion_group_register_timestamp(m_cgHndl,
                                                                    scalEvent.longSo.m_targetValue,
                                                                    scalEvent.timestampBuff->cbOffsetInFd,
                                                                    scalEvent.timestampBuff->indexInCbOffsetInFd);
        if (rc != SCAL_SUCCESS)
        {
            LOG_ERR(SYN_STREAM,
                         "scal_completion_group_register_timestamp failed with rc {}. devHndl 0x{:x} "
                         "CompletionGroup {} handle 0x{:x} Timestamps handle 0x{:x} offset {}",
                         rc,
                         TO64(m_devHndl),
                         m_name,
                         TO64(m_cgHndl),
                         scalEvent.timestampBuff->cbOffsetInFd,
                         scalEvent.timestampBuff->indexInCbOffsetInFd);
            return synFailedToCollectTime;
        }
    }
    return synSuccess;
}

void ScalCompletionGroupBase::_setExpectedCounter() const
{
    scal_completion_group_set_expected_ctr(m_cgHndl, m_completionTarget);
}

/*
 ***************************************************************************************************
 *   @brief tdr() - Get updated completion group info and call a function to check for timeout
 *
 *   @param - tdrType: CHECK: check, return an error if timeout (true), STATUS: just check (returns false)
 *   @return new value
 *   Note: this function doesn't require to be called under a lock
 ***************************************************************************************************
 */
TdrRtn ScalCompletionGroupBase::tdr(TdrType tdrType)
{
    TdrRtn tdrRtn {};
    scal_completion_group_infoV2_t cgInfo;

    synStatus synStatus = getCurrentCgInfo(cgInfo);
    if (synStatus != synSuccess)
    {
        return tdrRtn;  // do not return an error (don't drop the device)
    }

    if (!cgInfo.tdr_enabled)
    {
        return tdrRtn;
    }

    uint64_t sinceArm = TimeTools::timeFromUs(m_tdrInfo.armTime);

    if (tdrType == TdrType::DFA)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: on {}: tdr_value=0x{:x}, current_value=0x{:x}, armed={}, m_prevCompleted={:x}, "
                  "time since armed: {}us, timeoutDisabled={} timeoutUs={}",
                  HLLOG_FUNC,
                  m_name,
                  cgInfo.tdr_value,
                  cgInfo.current_value,
                  m_tdrInfo.armed,
                  m_tdrInfo.prevCompleted,
                  sinceArm,
                  cgInfo.timeoutDisabled,
                  cgInfo.timeoutUs);
    }

    // Nothing in flight
    if (cgInfo.tdr_value == cgInfo.current_value)
    {
        if (m_tdrInfo.armed)
        {
            LOG_TRACE(SYN_STREAM, "tdr cleared arm - {} tdr_value 0x{:x}", m_name, cgInfo.tdr_value);
        }
        m_tdrInfo.armed = false;
        return tdrRtn;
    }
    // if we see something in flight for the first time
    if (!m_tdrInfo.armed)
    {
        m_tdrInfo.armed         = true;                  // start checking
        m_tdrInfo.prevCompleted = cgInfo.current_value;  // keep the current value, so it can be checked next time
        m_tdrInfo.armTime       = TimeTools::timeNow();
        LOG_TRACE(SYN_STREAM,
                  "tdr armed - {} tdr_value 0x{:x} current_value 0x{:x} timeoutDisabled {} timeoutUs {}",
                  m_name,
                  cgInfo.tdr_value,
                  cgInfo.current_value,
                  cgInfo.timeoutDisabled,
                  cgInfo.timeoutUs);
        return tdrRtn;
    }

    // we get here if something is inflight from previous time
    if (m_tdrInfo.prevCompleted < cgInfo.current_value)  // Something was finished, update the time
    {
        m_tdrInfo.prevCompleted = cgInfo.current_value;  // keep the current value, so it can be checked next time
        m_tdrInfo.armTime       = TimeTools::timeNow();  // restart the time
        LOG_TRACE(SYN_STREAM,
                  "tdr restart timer - {} tdr_value 0x{:x} current_value 0x{:x} timeoutDisabled {} timeoutUs {}",
                  m_name,
                  cgInfo.tdr_value,
                  cgInfo.current_value,
                  cgInfo.timeoutDisabled,
                  cgInfo.timeoutUs);
        return tdrRtn;
    }

    // If disabled or "timeout" hasn't passed -> all is good
    if (TimeTools::timeFromUs(m_tdrInfo.armTime) <= cgInfo.timeoutUs)
    {
        return tdrRtn;
    }

    int logLevel = cgInfo.timeoutDisabled ? SPDLOG_LEVEL_INFO : SPDLOG_LEVEL_ERROR;

    auto       logger   = (tdrType == TdrType::DFA) ? synapse::LogManager::LogType::SYN_DEV_FAIL : synapse::LogManager::LogType::SYN_STREAM;
    const auto spLogger = hl_logger::getLogger(logger);

    HLLOG_UNTYPED(spLogger, logLevel, SEPARATOR_STR);
    HLLOG_UNTYPED(spLogger, logLevel, "| Engines timeout reached");
    HLLOG_UNTYPED(spLogger, logLevel, "| on stream:\t\t{}", m_name);
    HLLOG_UNTYPED(spLogger, logLevel,
                 "| completed:\t\t0x{:x} out of 0x{:x} commands",
                 cgInfo.current_value,
                 cgInfo.tdr_value);
    HLLOG_UNTYPED(spLogger, logLevel, "| time waited:\t{}us", sinceArm);
    HLLOG_UNTYPED(spLogger, logLevel,
                 "| timeout:\t\t{}us, timeout is {}.",
                 cgInfo.timeoutUs,
                 (cgInfo.timeoutDisabled ? "disabled" : "enabled"));
    HLLOG_UNTYPED(spLogger, logLevel, SEPARATOR_STR);
    if (cgInfo.timeoutDisabled) return tdrRtn;

    tdrRtn.failed = true;
    tdrRtn.msg    = m_name;

    return tdrRtn;
}

bool ScalCompletionGroupBase::getCgTdrInfo(const CgTdrInfo*& pCgTdrInfo) const
{
    pCgTdrInfo = &m_tdrInfo;
    return true;
};
