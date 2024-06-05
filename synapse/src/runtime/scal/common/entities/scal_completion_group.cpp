#include "scal_completion_group.hpp"

#include "defs.h"

#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"

#include "runtime/scal/common/scal_event.hpp"

#include "timer.h"

ScalCompletionGroup::ScalCompletionGroup(scal_handle_t                      devHndl,
                                         const std::string&                 name)
: ScalCompletionGroupBase(devHndl, name, nullptr)
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
synStatus ScalCompletionGroup::init()
{
    synStatus status = ScalCompletionGroupBase::init();
    if (status != synSuccess)
    {
        return status;
    }

    if (m_cgInfo.isDirectMode)
    {
        LOG_ERR(SYN_STREAM,
                     "devHndl 0x{:x} name {} m_cgHndl 0x{:x} unexpected direct-mode CG",
                     TO64(m_devHndl),
                     m_name,
                     TO64(m_cgHndl));

        return synFail;
    }

    LOG_DEBUG(SYN_STREAM,
                   "devHndl 0x{:x} name {} m_cgHndl 0x{:x} index_in_scheduler {} created",
                   TO64(m_devHndl),
                   m_name,
                   TO64(m_cgHndl),
                   m_cgInfo.index_in_scheduler);

    int rc = scal_control_core_get_info(m_cgInfo.scheduler_handle, &m_schedulerInfo);
    if (rc != SCAL_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                     "devHndl 0x{:x} m_cgHndl {} scal_control_core_get_info failed with rc {}",
                     TO64(m_devHndl),
                     TO64(m_cgHndl),
                     rc);
        return synFail;
    }

    LOG_DEBUG(SYN_STREAM,
                   "devHndl 0x{:x} name {} scheduler_name {} scheduler_index {} created",
                   TO64(m_devHndl),
                   m_name,
                   m_schedulerInfo.name,
                   m_schedulerInfo.idx);

    return synSuccess;
}

std::string ScalCompletionGroup::_getAdditionalPrintInfo() const
{
    return std::string("index_in_schedule ") + std::to_string(m_cgInfo.index_in_scheduler);
}
