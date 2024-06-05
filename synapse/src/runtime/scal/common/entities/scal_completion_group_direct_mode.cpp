#include "scal_completion_group_direct_mode.hpp"

#include "defs.h"

#include <vector>

ScalCompletionGroupDirectMode::ScalCompletionGroupDirectMode(scal_handle_t                      devHndl,
                                                             const std::string&                 name,
                                                             const common::DeviceInfoInterface* pDeviceInfoInterface)
: ScalCompletionGroupBase(devHndl, name, pDeviceInfoInterface)
{
}