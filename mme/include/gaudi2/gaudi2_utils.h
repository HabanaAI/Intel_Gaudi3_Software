#ifndef MME__GAUDI2_UTILS_H
#define MME__GAUDI2_UTILS_H

#include "gaudi2/mme.h"
#include "include/mme_common/mme_common_enum.h"

namespace Gaudi2
{
Mme::EMmeDataType ConvertDataTypeToGaudi2(MmeCommon::EMmeDataType dt);
MmeCommon::EMmeDataType ConvertDataTypeFromGaudi2(Mme::EMmeDataType dt);

}

#endif //MME__GAUDI2_UTILS_H
