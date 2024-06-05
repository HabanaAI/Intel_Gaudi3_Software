#pragma once

#include "scal_completion_group_base.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include "synapse_common_types.h"

/*******************************************************************/
/**************    ScalCompletionGroupDirectMode    ****************/
/*******************************************************************/
class ScalCompletionGroupDirectMode : public ScalCompletionGroupBase
{
public:
    ScalCompletionGroupDirectMode(scal_handle_t                      devHndl,
                                  const std::string&                 name,
                                  const common::DeviceInfoInterface* pDeviceInfoInterface);

    virtual ~ScalCompletionGroupDirectMode() = default;

protected:
    virtual std::string _getAdditionalPrintInfo() const override { return "direct-mode"; };
};
