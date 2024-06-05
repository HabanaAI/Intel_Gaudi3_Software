/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* Tzachi Cohen <tcohen@habana.ai>
********************************************************************/


#ifndef TPC_KERNEL_LIB_INTERFACE_H
#define TPC_KERNEL_LIB_INTERFACE_H

#include <stdint.h>

#define _IN_
#define _INOUT_
#define _OUT_

namespace tpc_lib_api
{

/* name of function entry points */
#define KERNEL_GUIDS_ENTRY_POINT_NAME               "GetKernelGuids"
#define KERNEL_INSTANTIATION_ENTRY_POINT_NAME       "InstantiateTpcKernel"
#define GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME "GetSupportedDataLayouts"
#define GET_SHAPE_INFERENCE_ENTRY_POINT_NAME        "GetShapeInference"
#define GET_LIB_VERSION_ENTRY_POINT_NAME            "GetLibVersion"

static const uint32_t MAX_SCALAR_PARAMS         = 32; /* number of scalar parameters passed to TPC program */
static const uint32_t MAX_NODE_NAME             = 64;
static const uint32_t MAX_INDEX_SPACE_DIM_SIZE  = 5;
static const uint32_t MAX_TENSOR_DIM            = 25;

typedef enum _DeviceId
{
    DEVICE_ID_GAUDI             = 1 ,
    DEVICE_ID_GRECO             = 2 ,
    DEVICE_ID_GAUDI2            = 3 ,
    DEVICE_ID_GAUDI3            = 4 ,
    // MUST BE LAST
    DEVICE_ID_MAX               = 5
} DeviceId;

typedef enum _GlueCodeReturn
{
    GLUE_SUCCESS                            = 0 ,
    GLUE_NODE_NOT_FOUND                     = 1 ,
    GLUE_INSUFFICIENT_ISA_BUFFER            = 2 ,
    GLUE_INCOMPATIBLE_INPUT_COUNT           = 3 ,
    GLUE_INCOMPATIBLE_INPUT_SIZE            = 4 ,
    GLUE_INCOMPATIBLE_OUTPUT_COUNT          = 5 ,
    GLUE_INCOMPATIBLE_OUTPUT_SIZE           = 6 ,
    GLUE_INCOMPATIBLE_DATA_TYPE             = 7 ,
    GLUE_UNSUPPORTED_LAYER_CONFIGURATION    = 8 ,
    GLUE_INSUFFICIENT_AUX_BUFFER_SIZE       = 9 ,
    GLUE_UNSUPPORTED_QUANT_PARAMS           = 10,
    GLUE_UNSUPPORTED_BROADCAST_MODE         = 11,
    GLUE_UNSUPPORTED_API_VERSION            = 12,
    GLUE_NON_STATIC_INPUT_TENSOR            = 13,
    GLUE_KERNEL_REQUIRE_REDUCIBLE_TENSOR    = 14,
    GLUE_INVALID_SHAPE_INFERENCE_ID         = 15,
    GLUE_KERNEL_INVALID_SCALAR_ARGUMENT     = 16,
    GLUE_UNSUPPORTED_LOW_FCD_INPUT          = 17,
    GLUE_INSUFFICIENT_ELF_BUFFER            = 18,
    GLUE_MISSING_PRIVATE_STRUCTURE          = 19,
    GLUE_UNSUPPORTED_5D_TENSORS             = 20,
    GLUE_CGUID_GRAPH_UNCHANGED              = 21,
    GLUE_SIF_NULL_PTR                       = 22,
    GLUE_UNSUPPORTED_DYNAMIC_SHAPE          = 23,
    GLUE_UNSUPPORTED_HUGE_TENSORS           = 24,
    GLUE_FAILED                             = 400,
} GlueCodeReturn;

typedef enum _TensorDataType
{
    DATA_I4     = 1 << 0,
    DATA_U4     = 1 << 1,
    DATA_I8     = 1 << 2,
    DATA_U8     = 1 << 3,
    DATA_F8_152 = 1 << 4,
    DATA_F8_143 = 1 << 5,
    DATA_I16    = 1 << 6,
    DATA_U16    = 1 << 7,
    DATA_BF16   = 1 << 8,
    DATA_F16    = 1 << 9,
    DATA_I32    = 1 << 10,
    DATA_U32    = 1 << 11,
    DATA_F32    = 1 << 12,
    DATA_I64    = 1 << 13,
    DATA_U64    = 1 << 14,
    NUM_DATATYPES = 15
} TensorDataType;

typedef struct _TensorDataLayout
{
    char layout[MAX_TENSOR_DIM];
} TensorDataLayout;

typedef struct _NodeDataLayouts
{
    TensorDataLayout* inputs;
    uint32_t          inputTensorNr;
    TensorDataLayout* outputs;
    uint32_t          outputTensorNr;
    TensorDataLayout* shapeTensors;
    uint32_t          shapeTensorNr;
} NodeDataLayouts;

typedef struct _UniqueShapeInferenceHash {
    union {
        struct {
            uint64_t  hashValue :56;     // should be filled by glueCode
            uint64_t  sharedObjectId :8; // reserved - used by graphCompiler

        };
        uint64_t   Value;
    };
} UniqueShapeInferenceHash;

typedef struct _GuidInfo
{
    char                      name[MAX_NODE_NAME];
    UniqueShapeInferenceHash  nameHash;
                            /* This hash can be optionally generated by the perf library, it is a utility
                             to accelerate lookup operations within the operator library.
                             The hash is provided by perf-lib on "GetKernelGuids" and returned to perf-lib,
                             by graph compiler on "InstantiateTpcKernel" or other entry points.*/
    union {
        struct {
                uint32_t  supportsDynamicShapes:1; /* kernel supports dynamic shapes*/
                uint32_t  supports64bit:1;         /* kernels supports 64b sizes */
                uint32_t  :30; //reserved.
        };
        uint32_t kernelProperties;
    };
} GuidInfo;

typedef struct   _UserParams
{
    void*       nodeParams;
    uint32_t    nodeParamsSize;
} UserParams;

typedef union _ElementValue
{
    float           fValue;
    uint32_t        u32Value;
    int32_t         i32Value;
    uint16_t        u16Value;
    int16_t         i16Value;
    uint8_t         u8Value;
    int8_t          i8Value;
    uint64_t        u64Value;
    int64_t         i64Value;
} ElementValue;

typedef struct _TensorGeometry
{
    /* Rank of the tensor */
    uint32_t        dims;
    /* Array holding the size of the tensor in each dimension, fastest changing
     * dimension first.
     * if maxSizes == minSizes, shape is static.
     * if minSizes < maxSize the size of the tensor in TPC execution time may be any value between
     * minsize and maxSize. Index space geometry should be returned according to maxSize.
     * The dynamic shapes logic in GC uses only the maxSizes, which then represent the final
     * shape (regardless of whether gc infer max shapes, min shapes or actual shapes)
     */
    uint64_t        maxSizes[MAX_TENSOR_DIM];
    uint64_t        minSizes[MAX_TENSOR_DIM];
    TensorDataType  dataType;
}  TensorGeometry;

typedef struct _TensorQuantizationParam
{
    union {
        int8_t  zeroPoint;
        uint8_t zeroPointU8;
        int8_t  fp8bias;
    };
    double scale;
} TensorQuantizationParam;

typedef struct _Tensor
{
    TensorGeometry          geometry;
    TensorQuantizationParam quantizationParam; /* relevant only for fp8/int8/uint8/int16 datatypes*/
    TensorDataLayout        layout;
    union {
        struct {
            uint32_t  Reducible:1;  /* The tensor is nested in reduction capable memory*/
            uint32_t  Null:1;       /* The tensor is not provided - in case of optional input/output*/
            uint32_t  :30; //reserved.
        };
        uint32_t   flags;
    };
    /* Point to tensor content, if it is known statically on graph compiler time.*/
    const void*             pData;
    /*  This array holds description of any transpose operation that may have been applied to the
     *  tensor between the creation of the private structure and the call into glue code. If the
     *  permutation array does not hold identity mapping, the glue code may need to re-interpret
     *  the private structure in a different way. For example, in a reduce operation the private
     *  structure holds a parameter defining an axis to be reduced. If the axis in question has been
     *  transposed, glue-code should re-interpret the axis to its transpose destination. */
    uint32_t permutation[MAX_TENSOR_DIM];
    uint32_t reserved[6];
} Tensor;

typedef struct _AuxTensor
{
    _OUT_   TensorGeometry    geometry;
    _IN_    void*             pData;
    _INOUT_ uint64_t          bufferSize; /* the buffer size is defined in bytes. */
    union {
        struct {
            uint32_t  sramAlloc:1;  /* This flag instruct GC to allocate the aux tensor in SRAM memory*/
            uint32_t  noInit:1;     /* This flag indicate the aux tensor will be used as scratch pad
                                       hence no init is necessary. 'pData' field will be ignored*/
            uint32_t  :30; //reserved.
        };
        uint32_t   flags;
    };
    uint32_t  reserved[6];
} AuxTensor;


typedef struct _DimIndexSpaceMapping
{
    uint32_t        indexSpaceDim; /* The dimension in the index-space this
                                    * tensor dimension transform corresponds to. */
    float           a;
    float           start_b;
    float           end_b;
    bool            allRequired;
} DimIndexSpaceMapping;

typedef struct _TensorAccessPattern
{
    union {
        struct {
            uint32_t  allRequired :1;           /* Declares the kernel may access any part of the tensor
                                                 * from any index space invocation  */

            uint32_t  memsetBeforeExecution :1; /* Kernel requests GC to memset the kernel before
                                                   execution */
            uint32_t  inputsReusability :16;    /* Relevant to output tensors only.
                                                 * A bit field respective to 16 input tensors,
                                                 * For memory efficiency purposes, it specifies if the
                                                 * output tensor can be nested on the same memory as the input*/
            uint32_t  inputReusabilityBinding:1;/* Relevant to output tensors only.
                                                 * Marks the input reusability mask as binding
                                                 * instead of an optimization suggestion */
            uint32_t  sparseAccess:1;           /* This hints Graph compiler that the tensor is sparsely
                                                 * accessed to help it optimize memory allocation decisions*/
            uint32_t  noRmwAccess:1;            /* Overrides RMW store indication in ELF header */
            uint32_t  fullyAccessedOnce:1;      /* Declares the kernel accessing the entire tensor once
                                                 * When fullyAccessed=1 it means that the tensor is fully accessed by
                                                 * the kernel for *all* test conditions/data
                                                 * When fullyAccessed=0 it means that the kernel may sparsely access
                                                 * the tensor or read it multiple times.*/
            uint32_t fullyWritten : 1;          /* This hints Graph compiler that the tensor is sparsely
                                                 * accessed to help it optimize memory allocation decisions*/
            uint32_t : 8;                       // reserved.
        };
        uint32_t   Value;
    };
    // an array of index space mapping data per dimension.
    DimIndexSpaceMapping   mapping[MAX_TENSOR_DIM];
    ElementValue           memsetValue;/* determines the value to memset if requested 'memsetBeforeExecution'*/
    uint32_t  reserved[6];
} TensorAccessPattern;

typedef struct _HabanaKernelParams
{
_IN_    int        apiVersion;
_IN_    DeviceId   deviceId;                /* asic ID */
_IN_    GuidInfo   guid;                    /* GUID of node in the graph */
_IN_    UserParams nodeParams;              /* Kernel specific parameters */
_IN_    Tensor*    inputTensors;            /* array of the input tensor */
_IN_    uint32_t   inputTensorNr;           /* the number of input tensors */
_IN_    Tensor*    outputTensors;           /* array of the output tensor  */
_IN_    uint32_t   outputTensorNr;          /* the number of output tensors. */
_IN_    uint32_t   maxAvailableTpc;         /* The maximum amount of TPC engines the kernel will execute on.*/
_IN_    uint32_t   useDeterministic;        /* directive to return deterministic version of program */
_IN_    uint64_t   uniqueNodeId;            /* provided to be able to easily trace in the logs */
_IN_    uint32_t   debugFlags;              /* for internal use.- used to debug/profile */
_IN_    uint16_t   validInputTensors;       /* Valid input tensors bit mask */
_IN_    uint16_t   validOutputTensors;      /* Valid output tensors bit mask */
        uint32_t   reserved[24];
} HabanaKernelParams;


typedef struct _DeviceKernel
{
    _IN_    void*    kernelElf  ;   /* A buffer in the host address space to which the kernel binary elf
                                     *  should be written. Allocated by GC */
    _INOUT_ uint32_t elfSize;       /* The size of the buffer supplied by GC. Updated by glue code
                                     * to reflect the size needed for the buffer */
    _OUT_   uint32_t scalarParams[MAX_SCALAR_PARAMS];
    _OUT_   uint32_t paramsNr;
} DeviceKernel;


typedef struct _HabanaKernelInstantiation
{
    _OUT_   uint32_t             indexSpaceRank;
    _OUT_   uint64_t             indexSpaceGeometry[MAX_INDEX_SPACE_DIM_SIZE];
    _OUT_   TensorAccessPattern* inputTensorAccessPattern;
    _OUT_   TensorAccessPattern* outputTensorAccessPattern;
    _INOUT_ AuxTensor*           auxiliaryTensors; // see comment below
    _INOUT_ uint32_t             auxiliaryTensorNr;
    _INOUT_ DeviceKernel         kernel;
    _OUT_   uint32_t             preferredSplitDim;  /* Set a dimension that GC code-gen pass must
                                                    partiotion with each partition size of 1.
                                                    This dimension must be set as allRequired within
                                                    DimIndexSpaceMapping for all tensors.
                                                    At the GC code_gen level (after slice pass),
                                                    GC must begin with this dimension with each
                                                    partition size of 1 and may continue to other
                                                    dimensions in case preferredSplitDim is smaller
                                                    than maxAvailableTpc
                                                    API:
                                                    0 - Disabled feature (default),
                                                    Otherwise: (preferredSplitDim-1) is dim to split
                                                    E.g.: 1 means dim0 (FCD), etc.
                                                */
    uint32_t                     reserved[15];
} HabanaKernelInstantiation;
// In order to reduce hand shakes between GC and PERF lib for aux tensors,
// auxiliaryTensors* list is statically allocated by GC as AuxTensor[MAX_AUX_NUM],
// where MAX_AUX_NUM is defined to be 16. auxiliaryTensorNr will be set
// by GC to MAX_AUX_NUM.
//
// PERF lib will then set auxiliaryTensorNr to the actual number of aux tensors
// required by the kernel (0-15, usually not more than 2), and set the required
// aux parameters accordingly.

typedef struct _NodeTensorPermutation
{
    uint32_t permutation[MAX_TENSOR_DIM];
} NodeTensorPermutation;


typedef struct _TensorShapeInfo
{
    TensorGeometry geometry;
    uint32_t *hostAddress; // can be null
    uint32_t reserved[8];
} TensorShapeInfo;


typedef struct _ShapeInferenceParams
{
    int                     apiVersion;
    uint64_t                uniqueNodeId;       /* Provided to be able to easily trace in the logs */
    GuidInfo                guid;               // transition to using a pointer to guidInfo. will be removed
    TensorShapeInfo**       inputTensors;       /* Pointer to array of the input tensor shapes */
    uint32_t                inputTensorsNr;     /* The number of input tensors shapes */
    UserParams              nodeParams;         /* User specific params */
    uint32_t                outputTensorsNr;    /* The number of output tensors shapes that was set during graph */
                                                /* creation according to max bucket size. */
    NodeTensorPermutation*  inputPermutations;  /* Either null, meaning all permutations are trivial */
                                                /* or inputTensorsNr permutations for all input tensors */
    NodeTensorPermutation*  outputPermutations; /* Ditto for output tensors. */
    GuidInfo*               pGuid;              // transition to using only the pointer. guid (above) will be removed
    uint32_t                maxAvailableTpc;    //
    uint32_t                reserved[3];
} ShapeInferenceParams;


typedef struct _ShapeInferenceOutput
{
    TensorShapeInfo**       outputTensors;  // Pointer to array of the output tensor shapes
    uint32_t*               invalidMask;    // Invalid bit mask per output shape tensor
                                            // only bits 0...('outputTensorsNr'-1) are meaningful.
                                            // Assumed that the invalidMask is initialized to 0 by the caller.
                                            // E.g. if the bit0 (LSB) = 1,then output shape
                                            // tensor0 is invalid.
    uint32_t                reserved[6];
} ShapeInferenceOutput;


/*
 ***************************************************************************************************
 *   @brief through this entry point the kernel library report supported GUIDs
 *
 *   @param deviceId          [in]    Queries device ID
 *   @param kernelCount   [in/out]    Pointer to size of 'names' array.
 *                                    Size of 'names'is a 2d array of size [kernelCount][MAX_NODE_NAME]
 *                                    If the pointed value is zero, the callee is expected to fill requested size
 *   @param guids           [out]     list of supported guids to be filled by the callee.
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef GlueCodeReturn(*pfnGetKernelGuids)
(
    _IN_    DeviceId        deviceId,
    _INOUT_ uint32_t*       kernelCount,
    _OUT_   GuidInfo*       guids);

/*
 ***************************************************************************************************
 *   @brief Kernel instantiation enty point
 *
 *   @param HabanaKernelParams          [in]        Input parameters describing the node
 *   @param HabanaKernelInstantiation   [in/out]    All properties required to integrate the node
 *                                                  into a recipe.
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef GlueCodeReturn(*pfnInstantiateTpcKernel)
(
    _IN_    const HabanaKernelParams*       params,
    _INOUT_ HabanaKernelInstantiation*      instance
);

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler about supported data layouts of specific kernel
 *
 *   @param params              [in]        Input parameters describing the node
 *   @param supportedLayouts    [out]       Array holding supported layouts. The array is allocated
 *                                          by the caller. 'xxxxx' stands for data layout agnostic
 *                                          kernel.
 *   @param layoutCount         [in/out]    in  - Size of 'supportedLayouts' array.
 *                                          out - Number of supported layouts for the kernel in
 *                                                question.
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn(*pfnGetSupportedDataLayout)
(
    _IN_     const HabanaKernelParams*  params,
    _OUT_    NodeDataLayouts*           supportedLayouts,
    _INOUT_  uint32_t*                  layoutCount
);

/*
 ***************************************************************************************************
 *   @brief Given the input sizes of a specific node, the function returns the size of the output tensors.
 *
 *   @param inputParams     [in]        shape inference input params
 *   @param outputData      [out]       shape inference calculated outputs
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef GlueCodeReturn(*pfnGetShapeInference)
(
    _IN_           DeviceId              deviceId,
    _IN_     const ShapeInferenceParams* inputParams,
    _OUT_          ShapeInferenceOutput* outputData
);

/*
 ***************************************************************************************************
 *   @brief supplies graph compiler library version for coherency checks
 *   between graph compiler and runtime
 *
 *
 *   @return               Unique library Version
 ***************************************************************************************************
 */

typedef uint64_t (*pfnGetLibVersion)();


} /* name space tpc_lib_api */

#endif /* TPC_KERNEL_LIB_INTERFACE_H */

