#pragma once

#include "scal_completion_group_base.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include "synapse_common_types.h"

/*********************************************************/
/**************    ScalCompletionGroup    ****************/
/*********************************************************/
class ScalCompletionGroup : public ScalCompletionGroupBase
{
public:
    ScalCompletionGroup(scal_handle_t      devHndl,
                        const std::string& name);

    virtual ~ScalCompletionGroup() = default;

    virtual synStatus init() override;

    virtual uint32_t getIndexInScheduler() const override { return m_cgInfo.index_in_scheduler; }

protected:
    virtual std::string _getAdditionalPrintInfo() const override;

private:
    scal_control_core_info_t       m_schedulerInfo;
};
