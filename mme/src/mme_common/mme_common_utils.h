#pragma once

#include "include/mme_common/mme_common_enum.h"
#include <optional>

namespace MmeCommon
{
bool isInputAligned(const MmeLayerParams& params, unsigned alignVal, std::optional<EMmeInputOperand> inputOp);

}  // namespace MmeCommon
