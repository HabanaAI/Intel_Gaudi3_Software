/* SPDX-License-Identifier: MIT
 *
 * Copyright 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef SHIM_TYPEDEFS_H
#define SHIM_TYPEDEFS_H

#include "shim_types.h"

typedef enum shim_api_type ShimApiType;
typedef void *ShimFunctions;
typedef ShimFunctions(*PFN_ShimGetFunctions)(
        ShimApiType apiType, ShimFunctions pOrigFunctions);
typedef void(*PFN_ShimFinish)(ShimApiType apiType);
typedef void(*PFN_ShimSetApiVersion)(ShimApiType apiType, const char *version);


#endif // SHIM_TYPEDEFS_H
