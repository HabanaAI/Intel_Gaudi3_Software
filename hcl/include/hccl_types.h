/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************/

#ifndef HCCL_TYPES_H_
#define HCCL_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <synapse_api_types.h>

/* Opaque handle to communicator */
#define HCCL_UNIQUE_ID_MAX_BYTES 1024

#define hcclComm_t void*

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTNEXTLINE(modern1ize-use-using)
typedef struct
{
    uint8_t internal[HCCL_UNIQUE_ID_MAX_BYTES];
    size_t  length;
} hcclUniqueId;

/* Reduction operation selector */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum /* aligned to ncclRedOp_t */
{
    hcclSum    = 0,
    hcclProd   = 1,
    hcclMin    = 2,
    hcclMax    = 3,
    hcclAvg    = 4,
    hcclOpNone = 5
} hcclRedOp_t;

/* Error type */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum
{
    hcclSuccess = 0,
    hcclNoDeviceFound,
    hcclUnsupported,
    hcclOutOfMemory,
    hcclUnhandledSynapseError,
    hcclSystemError,
    hcclInternalError,
    hcclInvalidArgument,
    hcclInvalidUsage,
    hcclSocketError,
    hcclLibfabricError,
    hcclTryAgainError,
    hcclBusy,
    hcclNumResults,
    hcclStreamError,
    hcclSynapseTerminated
} hcclResult_t;

/* Data types */
// NOLINTNEXTLINE(modernize-use-using)
typedef enum
{
    hcclInt8     = 0,
    hcclChar     = 0,
    hcclUint8    = 1,
    hcclInt32    = 2,
    hcclInt      = 2,
    hcclUint32   = 3,
    hcclInt64    = 4,
    hcclUint64   = 5,
    hcclFloat16  = 6,
    hcclHalf     = 6,
    hcclFloat32  = 7,
    hcclFloat    = 7,
    hcclFloat64  = 8,
    hcclDouble   = 8,
    hcclBfloat16 = 9,
    hcclNumTypes
} hcclDataType_t;

typedef void (*hcclStreamCallback_t)(synStreamHandle stream, hcclResult_t result, void* userData);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard HCCL_TYPES_H_
