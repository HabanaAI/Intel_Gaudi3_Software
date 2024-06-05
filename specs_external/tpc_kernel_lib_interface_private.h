/*****************************************************************************
* Copyright (C) 2023 HabanaLabs, Ltd.
* All Rights Reserved.
*
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
*
* Authors:
* Keren Luzon <kluzon@habana.ai>
******************************************************************************
*/


#ifndef TPC_KERNEL_LIB_INTERFACE_PRIVATE_H
#define TPC_KERNEL_LIB_INTERFACE_PRIVATE_H

#include "tpc_kernel_lib_interface.h"

namespace tpc_lib_api
{


/* name of function entry points */
#define GET_SUGGESTED_MANIPULATION_ENTRY_POINT_NAME             "GetSuggestedManipulation"
#define FUNCTIONAL_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME          "GetFunctionalComplexGuids"
#define PERFORMANCE_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME         "GetPerformanceComplexGuids"


typedef struct _DebugFlags {
  union {
    struct {
      // By default, the database will return the ISA according to the tensor sizes.
      // This flag forces the database to return the irf44 version kernel even if the
      // tensor sizes are smaller than 31b.
      unsigned  forceIrf44Mode :1;
      unsigned  :31; //reserved .
    };
    unsigned   Value;
  };
} DebugFlags;


typedef enum _TensorOperationType
{
    TENSOR_OP_NONE          = 0,
    TENSOR_OP_TRANSPOSE     = 1,
    TENSOR_OP_RESHAPE       = 2,
    TENSOR_OP_TILE          = 3
} TensorOperationType;


typedef struct _TensorOperation
{
    TensorOperationType opType;
    uint32_t    dims;                        /* Rank of the tensor */
    uint64_t    maxNewShape[MAX_TENSOR_DIM];
    uint64_t    minNewShape[MAX_TENSOR_DIM]; /* minNewShape is required in order to resolve ambiguous DS cases, for example:
                                                Original max sizes: (2,3,1), Original min sizes:   (2,3,0)
                                                New max sizes:      (6,1,1), New min sizes can be: (6,1,0)/(6,0,1)/(0,1,1) */
    uint32_t    permutation[MAX_TENSOR_DIM]; /* Permutation semantics should match numpy.transpose() */
    uint32_t    reserved[25];
} TensorOperation;


typedef struct _TensorManipulationSuggestion
{
    // Note: the following TensorOperation lists are the same length as
    //       HabanaKernelParams's inputTensorNr and outputTensorNr, accordingly.
    _INOUT_  TensorOperation*   inputTensors;
    _INOUT_  TensorOperation*   outputTensors;
} TensorManipulationSuggestion;


/*
 ***************************************************************************************************
 *   @brief Informs graph compiler about suggest manipulation to input/output tensors to improve
 *          performance
 *
 *   @param params            [in]        Input structure identical to the one supplied for kernel
 *                                        instantiation entry point.
 *   @param suggestion        [out]       Instructions on how to transpose/reshape the tensor
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn(*pfnGetSuggestedManipulation)
(
    _IN_     const HabanaKernelParams*      params,
    _OUT_    TensorManipulationSuggestion*  suggestion);



/*
 ***************************************************************************************************
 *   @brief Through this entry point the complex guid library report functional supported GUIDs
 *
 *   @param deviceId          [in]    Queries device ID
 *   @param guidCount     [in/out]    Pointer to size of 'names' array.
 *                                    Size of 'names'is a 2d array of size [guidCount][MAX_NODE_NAME]
 *                                    If the pointed value is zero, the callee is expected to fill requested size
 *   @param guids           [out]     list of supported guids to be filled by the callee.
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef GlueCodeReturn (*pfnGetFunctionalComplexGuids)
(
    _IN_    DeviceId        deviceId,
    _INOUT_ uint32_t*       guidCount,
    _OUT_   GuidInfo*       guids);


/*
 ***************************************************************************************************
 *   @brief Through this entry point the complex guid library report performance supported GUIDs
 *
 *   @param deviceId          [in]    Queries device ID
 *   @param guidCount     [in/out]    Pointer to size of 'names' array.
 *                                    Size of 'names'is a 2d array of size [guidCount][MAX_NODE_NAME]
 *                                    If the pointed value is zero, the callee is expected to fill requested size
 *   @param guids           [out]     list of supported guids to be filled by the callee.
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef GlueCodeReturn (*pfnGetPerformanceComplexGuids)
(
    _IN_    DeviceId        deviceId,
    _INOUT_ uint32_t*       guidCount,
    _OUT_   GuidInfo*       guids);


} /* name space tpc_lib_api */

#endif /* TPC_KERNEL_LIB_INTERFACE_PRIVATE_H */

