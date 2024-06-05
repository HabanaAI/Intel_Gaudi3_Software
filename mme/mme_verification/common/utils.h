#pragma once
#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{

EMmeOperand getInputFromOperation(EMmeOpType operation, bool isA);

EMmeOperand getOutputFromOperation(EMmeOpType operation);

} // namespace MmeCommon
