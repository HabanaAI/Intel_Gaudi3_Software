/*****************************************************************************
* Copyright (C) 2018 HabanaLabs, Ltd.
* All Rights Reserved.
*
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
*
* Authors:
* Tzachi Cohen <tcohen@habana.ai>
* Doron Singer <dsinger@habana.ai>
******************************************************************************
*/

#ifndef GRAPH_COMPILER_INTERFACE_H
#define GRAPH_COMPILER_INTERFACE_H

#include <stdint.h>

#define _IN_
#define _INOUT_
#define _OUT_

/* name of function entry points */
#define KERNEL_NAMES_ENTRY_POINT_NAME "GetKernelNames"
#define KERNEL_NAMES_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME "GetKernelSupportingDynamicShapeNames"
#define KERNEL_QUERY_ENTRY_POINT_NAME "HabanaKernel"
#define KERNEL_QUERY_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME "HabanaKernelV2"

namespace gcapi
{

static const unsigned MAX_TENSORS_GOYA       = 8;
static const unsigned MAX_TENSORS_GAUDI      = 16;

static const unsigned MAX_TENSOR_NR            = MAX_TENSORS_GAUDI;
static const unsigned MAX_TENSOR_DIM           = 5;
static const unsigned MAX_HABANA_DIM           = 25; // same as synapse HABANA_DIM_MAX
static const unsigned MAX_TENSORS_PARAMS_NR    = 32; /* number of scalar parameters passed to TPC program */
static const unsigned MAX_NODE_NAME            = 64;
static const unsigned MAX_HABANA_NODE_NAME     = 512;
static const unsigned MAX_USER_PARAMS_SIZE     = 128;
typedef void*       UserParams_t;

typedef enum _DeviceId_t
{
    DEVICE_ID_DALI              = 0 ,
    DEVICE_ID_GOYA              = 0 ,
    DEVICE_ID_GAUDI             = 1 ,
    DEVICE_ID_GOYA2             = 2 ,
    DEVICE_ID_GRECO             = 2 ,
    DEVICE_ID_GAUDI2            = 3 ,
    DEVICE_ID_GAUDI_B           = 4 ,
    DEVICE_ID_GAUDI3            = 5 ,
    // MUST BE LAST
    DEVICE_ID_MAX               = 6
} DeviceId_t;

typedef enum _GlueCodeReturn_t
{
    GLUE_SUCCESS                            = 0 ,
    GLUE_NODE_NOT_FOUND                     = 1 ,
    GLUE_INSUFICIENT_ISA_BUFFER             = 2 ,
    GLUE_INCOMPATIBLE_INPUT_COUNT           = 3 ,
    GLUE_INCOMPATIBLE_INPUT_SIZE            = 4 ,
    GLUE_INCOMPATIBLE_OUTPUT_COUNT          = 5 ,
    GLUE_INCOMPATIBLE_OUTPUT_SIZE           = 6 ,
    GLUE_INCOMPATIBLE_DATA_TYPE             = 7 ,
    GLUE_UNSUPPORTED_LAYER_CONFIGURATION    = 8 ,
    GLUE_INSUFICIENT_AUX_BUFFER_SIZE        = 9 ,
    GLUE_UNSUPPORTED_QUANT_PARAMS           = 10,
    GLUE_UNSUPPORTED_BROADCAST_MODE         = 11,
    GLUE_UNSUPPORTED_API_VERSION            = 12,
    GLUE_NON_STATIC_INPUT_TENSOR            = 13,
    GLUE_KERNEL_REQUIRE_REDUCIBLE_TENSOR    = 14,
    GLUE_KERNEL_REQUIRE_CONTIGUOUS_TENSOR   = 15,
    GLUE_KERNEL_INVALID_SCALAR_ARGUMENT     = 16,
    GLUE_UNSUPPORTED_LOW_FCD_INPUT          = 17,
    GLUE_INSUFICIENT_ELF_BUFFER             = 18,
    GLUE_MISSING_PRIVATE_STRUCTURE          = 19,
    GLUE_UNSUPPORTED_5D_TENSORS             = 20,
    GLUE_CGUID_GRAPH_UNCHANGED              = 21,
    GLUE_FAILED                             = 400,
} GlueCodeReturn_t;


/***************************************************************************************
 * The functions below can be optionally implemented by the kernel plug-in
 ***************************************************************************************
 */

#define GET_SUPPORTED_LAYOUTS_ENTRY_POINT_NAME "GetSupportedLayouts"

typedef struct _TensorDataLayout
{
  char layout[MAX_TENSOR_DIM];
} TensorDataLayout;

typedef struct _TensorDataLayoutV3
{
  char layout[MAX_HABANA_DIM];
} TensorDataLayoutV3;

typedef struct _NodeDataLayouts
{
  TensorDataLayout inputs[MAX_TENSOR_NR];
  TensorDataLayout outputs[MAX_TENSOR_NR];
} NodeDataLayouts;

typedef struct _NodeDataLayoutsV2
{
  TensorDataLayout inputs[MAX_TENSOR_NR];
  TensorDataLayout outputs[MAX_TENSOR_NR];
  TensorDataLayout shapeTensors[MAX_TENSOR_NR];
} NodeDataLayoutsV2;

typedef struct _NodeDataLayoutsV3
{
  TensorDataLayoutV3 inputs[MAX_TENSOR_NR];
  TensorDataLayoutV3 outputs[MAX_TENSOR_NR];
  TensorDataLayoutV3 shapeTensors[MAX_TENSOR_NR];
} NodeDataLayoutsV3;

typedef struct _NodeName
{
  char name[MAX_NODE_NAME];
} NodeName;

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler about supported data layouts of specific kernel
 *
 *   @param deviceId            [in]        Device ID
 *   @param nodeName            [in]        kernel GUID.
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

typedef GlueCodeReturn_t(*pfnGetSupportedDataLayout)
(
    _IN_     DeviceId_t         deviceId,
    _IN_     NodeName*          nodeName,
    _OUT_    NodeDataLayouts*   supportedLayouts,
    _INOUT_  unsigned*          layoutCount
);

typedef struct _TensorGeometry_t
{
    unsigned dims;
    /* Array holding the size of the tensor in each dimension, fastest changing
     * dimension first. */
    unsigned sizes[MAX_TENSOR_DIM];
    /*  This array holds description of any transpose operation that may have been applied to the
     *  tensor between the creation of the private structure and the call into glue code. If the
     *  permutation array does not hold identity mapping, the glue code may need to re-interpret
     *  the private structure in a different way. For example, in a reduce operation the private
     *  structure holds a parameter defining an axis to be reduced. If the axis in question has been
     *  transposed, glue-code should re-interpret the axis to its transpose destination. */
    unsigned permutation[MAX_TENSOR_DIM];
}  TensorGeometry_t;

typedef struct _TensorGeometryV2_t
{
    unsigned dims;
    /* Array holding the max size of the tensor in each dimension, fastest changing
     * dimension first. */
    union {
        unsigned maxSizes[MAX_TENSOR_DIM];
        /* Support of non-dynamic shapes kernels */
        unsigned sizes[MAX_TENSOR_DIM];
    };
    /* Array holding the min size of the tensor in each dimension, fastest changing
 * dimension first. */
    unsigned minSizes[MAX_TENSOR_DIM];
    /*  This array holds description of any transpose operation that may have been applied to the
     *  tensor between the creation of the private structure and the call into glue code. If the
     *  permutation array does not hold identity mapping, the glue code may need to re-interpret
     *  the private structure in a different way. For example, in a reduce operation the private
     *  structure holds a parameter defining an axis to be reduced. If the axis in question has been
     *  transposed, glue-code should re-interpret the axis to its transpose destination. */
    unsigned permutation[MAX_TENSOR_DIM];
}  TensorGeometryV2_t;

/*
 * _TensorGeometryV3_t allows creating high rank tensors (N dims as < N <= MAX_HABANA_DIM).
 */
typedef struct _TensorGeometryV3_t
{
    unsigned dims;
    /* Array holding the max size of the tensor in each dimension, fastest changing
     * dimension first. */
    union {
        unsigned maxSizes[MAX_HABANA_DIM];
        /* Support of non-dynamic shapes kernels */
        unsigned sizes[MAX_HABANA_DIM];
    };
    /* Array holding the min size of the tensor in each dimension, fastest changing
 * dimension first. */
    unsigned minSizes[MAX_HABANA_DIM];
    /*  This array holds description of any transpose operation that may have been applied to the
     *  tensor between the creation of the private structure and the call into glue code. If the
     *  permutation array does not hold identity mapping, the glue code may need to re-interpret
     *  the private structure in a different way. For example, in a reduce operation the private
     *  structure holds a parameter defining an axis to be reduced. If the axis in question has been
     *  transposed, glue-code should re-interpret the axis to its transpose destination. */
    unsigned permutation[MAX_HABANA_DIM];
}  TensorGeometryV3_t;

/*
 * _TensorGeometryV4_t support uint64_t tensor sizes.
 */
typedef struct _TensorGeometryV4_t
{
    unsigned dims;
    union {
        uint64_t maxSizes[MAX_HABANA_DIM];
        /* Support of non-dynamic shapes kernels */
        uint64_t sizes[MAX_HABANA_DIM];
    };
    uint64_t minSizes[MAX_HABANA_DIM];
    unsigned permutation[MAX_HABANA_DIM];
}  TensorGeometryV4_t;

typedef enum _TensorDataType_t
{
    DATA_F32    = 0,
    DATA_F16    = 1,
    DATA_I32    = 2,
    DATA_I16    = 3,
    DATA_I8     = 4,
    DATA_U8     = 5,
    DATA_BF16   = 6,
    DATA_U16    = 7,
    DATA_U32    = 8,
    DATA_I4     = 9,
    DATA_U4     = 10,
    DATA_F8_152 = 11,
    DATA_F8_143 = 12,
    DATA_I64    = 13,
    DATA_U64    = 14,
    NUM_DATATYPES = 15
} TensorDataType_t;

typedef enum _KernelType_t
{
    KERNEL_TYPE_INFERENCE        = 0,
    KERNEL_TYPE_TRAINING_FWD     = 1,
    KERNEL_TYPE_TRAINING_BWD     = 2
} KernelType_t;

typedef struct _TensorQuantizationParam
{
    double scale;
    union
    {
        int8_t zeroPoint;
        uint8_t zeroPointU8;
        int8_t  fp8bias;
    };
} TensorQuantizationParam;

typedef struct _Tensor_t
{
    TensorGeometry_t        geometry;
    TensorDataType_t        dataType;
    TensorQuantizationParam quantizationParam;
    union {
        struct {
            unsigned  :1;  //reserved
            unsigned  Reducible:1;   /* The tensor is nested in reduction capable memory*/
            unsigned  :30; //reserved.
        };
      unsigned   Value;
    };
    unsigned                reserved_1; /* explicit alignment of the struct */
    const void*             pData;      /* Graph Compiler should pass a pointer to the tensor data,
                                         * for any tensor that is defined as a model parameter. */
    unsigned                reserved[6];
} Tensor_t;

typedef struct _TensorV2_t
{
    TensorGeometryV2_t        geometry;
    TensorDataType_t          dataType;
    TensorQuantizationParam   quantizationParam;
    union {
        struct {
            unsigned  :1;  //reserved
            unsigned  Reducible:1;   /* The tensor is nested in reduction capable memory*/
            unsigned  :30; //reserved.
        };
        unsigned   Value;
    };
    unsigned                  reserved_1; /* explicit alignment of the struct */
    const void*               pData;      /* Graph Compiler should pass a pointer to the tensor data,
                                           * for any tensor that is defined as a model parameter. */
    /* data layout of the tensor */
    TensorDataLayout layout;
    unsigned                  reserved[10];
} TensorV2_t;

/*
 * _TensorV3_t supports high rank tensors (N dims as < N <= HIGH_RANK_MAX_TENSOR_DIM).
 */
typedef struct _TensorV3_t
{
    TensorGeometryV3_t        geometry;
    TensorDataType_t          dataType;
    TensorQuantizationParam   quantizationParam;
    union {
        struct {
            unsigned  :1;  //reserved
            unsigned  Reducible:1;   /* The tensor is nested in reduction capable memory*/
            unsigned  :30; //reserved.
        };
        unsigned   Value;
    };
    unsigned                  reserved_1; /* explicit alignment of the struct */
    const void*               pData;      /* Graph Compiler should pass a pointer to the tensor data,
                                       * for any tensor that is defined as a model parameter. */
    /* data layout of the tensor */
    TensorDataLayout layout;
    unsigned                  reserved[10];
} TensorV3_t;

/*
 * _TensorV4_t support uint64_t tensor sizes.
 */
typedef struct _TensorV4_t
{
    TensorGeometryV4_t        geometry;
    TensorDataType_t          dataType;
    TensorQuantizationParam   quantizationParam;
    union {
        struct {
            unsigned  :1; //reserved
            unsigned  Reducible:1;
            unsigned  :30;//reserved.
        };
        unsigned   Value;
    };
    unsigned                  reserved_1;
    const void*               pData;
    TensorDataLayout layout;
    unsigned                  reserved[10];
} TensorV4_t;

/* The caller is responsible to allocate the buffer pointed by 'pData'
 * and the callee is responsible to fill it. If the buffer is not large enough the callee will
 * return 'GLUE_INSUFICIENT_AUX_BUFFER_SIZE' return value and specify required buffer size
 * in 'bufferSize' parameter. */
typedef struct _AuxTensor_t
{
    _OUT_   TensorGeometry_t  geometry;
    _OUT_   TensorDataType_t  dataType;
    _IN_    void*             pData;
    _INOUT_ unsigned          bufferSize; /* the buffer size is defined in bytes. */
} AuxTensor_t;

typedef struct _AuxTensorV2_t
{
    _OUT_   TensorGeometryV2_t  geometry;
    _OUT_   TensorDataType_t    dataType;
    _IN_    void*               pData;
    _INOUT_ unsigned            bufferSize; /* the buffer size is defined in bytes. */
} AuxTensorV2_t;

typedef struct _DimTransform_t
{
    union {
        unsigned dim;           /* The dimension in the index-space this
                                 * tensor dimension transform corresponds to. */
        unsigned indexSpaceDim;
    };
    float           start_a;
    float           start_b;
    float           end_a;
    float           end_b;
} DimTransform_t;

typedef struct _TensorAccessPattern_t
{
    union {
        struct {
            unsigned  allRequired :1;           /* Declares the kernel may access any part of the tensor
                                                 * from any index space invocation  */
            unsigned  inputReusabilityBinding:1;/* Relevant to output tensors only.
                                                 * Marks the input reusability mask as binding
                                                 * instead of an optimization suggestion */
            unsigned  dropTensor :1;            /* The tensor is no longer relevant and should be dropped */
            unsigned  memsetBeforeExecution :1; /* Kernel requests GC to memset the kernel before
                                                   execution for Read-Modify-Write operations*/
            unsigned  inputsReusability :16;    /* Relevant to output tensors only.
                                                 * A bit field respective to 16 input tensors,
                                                 * so that any single input could be aliased to any
                                                 * single output*/
            unsigned  sparseAccess:1;           /* This hints Graph compiler that the tensor is sparsely
                                                 * accessed to help it optmize memory allocation decisions*/
            unsigned  noRmwAccess:1;            /* Overrides RMW store indication in ELF header */
            unsigned  fullyAccessed:1;          /* Declares the kernel accessing the entire tensor
                                                 * When fullyAccessed=1 it means that the tensor is fully accessed by
                                                 * the kernel for *all* test conditions/data
                                                 * When fullyAccessed=0 it means that the kernel may sparsely access
                                                 * the tensor.*/
            unsigned  :9; //reserved.
        };
        unsigned   Value;
    };
    union {
        DimTransform_t dim[MAX_TENSOR_DIM];        /* For each tensor dimension the glue-code define DimTransform_t,
                                                    * dim corresponds to a tensor dimension */
        DimTransform_t tensorDim[MAX_TENSOR_DIM];
    };
} TensorAccessPattern_t;


typedef struct _ProgramFlags {
  union {
    struct {
      unsigned  printfUsed :1;
      unsigned  specialFunctionsUsed :1;
      unsigned  :30; //reserved .
    };
    unsigned   Value;
  };
} ProgramFlags;

/* on error the kernel database will return the expected */
typedef struct _HabanaKernelParams_t
{
    _IN_     int          apiVersion;
    union{
        _IN_     char         nodeName[MAX_NODE_NAME];       /* shall be cleaned up                   */
        _IN_     char         guidName[MAX_NODE_NAME];       /* aligned to synapse api, i.e. pGuid    */
    };
    _IN_     UserParams_t NodeParams;                    /* user specific. */
    _IN_     DeviceId_t   deviceId;                      /* asic ID */
    _IN_     KernelType_t kernelType;                    /* deprecated, don't use */
    _INOUT_  Tensor_t     inputTensors[MAX_TENSOR_NR];   /* array of the input tensor handles.*/
    _INOUT_  unsigned     inputTensorNr;                 /* the number of input tensors */
    _INOUT_  Tensor_t     outputTensors[MAX_TENSOR_NR];  /* array of the output tensor handles. */
    _INOUT_  unsigned     outputTensorNr;                /* the number of output tensors. */
    _IN_     unsigned     debugFlags;                    /* for internal use.- used to debug/profile
                                                          *  programs. */
    _IN_     unsigned     NodeParamsSize;                /* Size of struct pointed by NodeParams */
    _IN_     unsigned     maxAvailableTpc;               /* Kernel writer should know that it will get any number between 1 and maxAvailableTpc
                                                          * Kernels that rely on number of TPC in the index space should expose index space size with maxAvailableTpc
                                                          * Examples: Sparse segment sum, Embedding bag kernels, etc .. */
    _IN_     unsigned     useDeterministic;              /* When useDeterministic=1, perfLib shall choose determistic speciailization */
    _IN_     uint64_t     uniqueNodeId;                  /* unique node id for the nodeName provided by the bridge to be able to easily trace in the logs */
             unsigned     reserved[25];
} HabanaKernelParams_t;

typedef struct _HabanaKernelParamsV2_t
{
    _IN_     int            apiVersion;
    union{
        _IN_     char         nodeName[MAX_NODE_NAME];     /* shall be cleaned up                   */
        _IN_     char         guidName[MAX_NODE_NAME];     /* aligned to synapse api, i.e. pGuid    */
    };
    _IN_     UserParams_t   NodeParams;                    /* user specific. */
    _IN_     DeviceId_t     deviceId;                      /* asic ID */
    _IN_     KernelType_t   kernelType;                    /* deprecated, don't use */
    _INOUT_  TensorV2_t     inputTensors[MAX_TENSOR_NR];   /* array of the input tensor handles.*/
    _INOUT_  unsigned       inputTensorNr;                 /* the number of input tensors */
    _INOUT_  TensorV2_t     outputTensors[MAX_TENSOR_NR];  /* array of the output tensor handles. */
    _INOUT_  unsigned       outputTensorNr;                /* the number of output tensors. */
    _IN_     unsigned       debugFlags;                    /* for internal use.- used to debug/profile
                                                            *  programs. */
    _IN_     unsigned       NodeParamsSize;                /* Size of struct pointed by NodeParams */
    _IN_     unsigned       maxAvailableTpc;               /* Kernel writer should know that it will get any number between 1 and maxAvailableTpc
                                                            * Kernels that rely on number of TPC in the index space should expose index space size with maxAvailableTpc
                                                            * Examples: Sparse segment sum, Embedding bag kernels, etc .. */
    _IN_     unsigned       useDeterministic;              /* When useDeterministic=1, perfLib shall choose determistic speciailization */
    _IN_     uint64_t       uniqueNodeId;                  /* unique node id for the nodeName provided by the bridge to be able to easily trace in the logs */
             unsigned       reserved[25];
} HabanaKernelParamsV2_t;

typedef struct _DeviceKernel_t
{
    _IN_    void*    kernelBinary;  /* A buffer in the host address space to which the kernel binary
                                     *  should be written. Allocated by GC */
    _INOUT_ unsigned binarySize;    /* The size of the buffer supplied by GC. Updated by glue code
                                     * to reflect the size needed for the buffer */
    _OUT_   uint32_t scalarParams[MAX_TENSORS_PARAMS_NR];
    _OUT_   unsigned paramsNr;
} DeviceKernel_t;

typedef union _PadValue
{
    float           fValue;
    unsigned        u32Value;
    int             i32Value;
    unsigned short  u16Value;
    short           i16Value;
    unsigned char   u8Value;
    char            i8Value;
} PadValue;

typedef struct _HabanaKernelInstantiation_t
{
    _OUT_   TensorGeometry_t      indexSpaceGeometry;
    _OUT_   TensorAccessPattern_t inputTensorAccessPattern[MAX_TENSOR_NR];
    _OUT_   PadValue              inputPadValues[MAX_TENSOR_NR];
    _OUT_   TensorAccessPattern_t outputTensorAccessPattern[MAX_TENSOR_NR];
    _INOUT_ AuxTensor_t           auxiliaryTensors[MAX_TENSOR_NR];
    _OUT_   unsigned              auxiliaryTensorCount;
    _INOUT_ DeviceKernel_t        kernel;
    _OUT_   ProgramFlags          flags;
    _INOUT_ void*                 kernelElf;
    _INOUT_ unsigned              elfSize;
    _OUT_   PadValue              outputMemsetValues[MAX_TENSOR_NR];
    _OUT_   unsigned              auxNotRequiringInit; /* This is a bit mask is defining which aux
                                                        *  tensor should be regarded as SRAM
                                                        * scratch pad aux tensor*/
            unsigned              reserved[16];
} HabanaKernelInstantiation_t;


typedef struct _HabanaKernelInstantiationV2_t
{
    _OUT_   TensorGeometryV2_t      indexSpaceGeometry;
    _OUT_   TensorAccessPattern_t   inputTensorAccessPattern[MAX_TENSOR_NR];
    _OUT_   PadValue                inputPadValues[MAX_TENSOR_NR];
    _OUT_   TensorAccessPattern_t   outputTensorAccessPattern[MAX_TENSOR_NR];
    _INOUT_ AuxTensorV2_t           auxiliaryTensors[MAX_TENSOR_NR];
    _OUT_   unsigned                auxiliaryTensorCount;
    _INOUT_ DeviceKernel_t          kernel;
    _OUT_   ProgramFlags            flags;
    _INOUT_ void*                   kernelElf;
    _INOUT_ unsigned                elfSize;
    _OUT_   PadValue                outputMemsetValues[MAX_TENSOR_NR];
    _OUT_   unsigned                auxNotRequiringInit; /* This is a bit mask is defining which aux
                                                          * tensor should be regarded as SRAM
                                                          * scratch pad aux tensor*/
            unsigned                reserved[16];
} HabanaKernelInstantiationV2_t;

typedef GlueCodeReturn_t(*pfnGetKernelNames)
(
    _OUT_ char**    names,
    unsigned*       kernelCount,
    DeviceId_t      deviceId
);

typedef GlueCodeReturn_t(*pfnGetKernelSupportingDynamicShapeNames)
(
        _OUT_ char**    names,
        unsigned*       kernelCount,
        DeviceId_t      deviceId
);

typedef GlueCodeReturn_t(*pfnHabanaKernel)
(
    _INOUT_ HabanaKernelParams_t *          params,
    _INOUT_ HabanaKernelInstantiation_t*    instance
);

typedef GlueCodeReturn_t(*pfnHabanaKernelV2)
(
        _INOUT_ HabanaKernelParamsV2_t *          params,
        _INOUT_ HabanaKernelInstantiation_t*    instance
);

#define GET_SUPPORTED_LAYOUTS_V2_ENTRY_POINT_NAME "GetSupportedLayoutsV2"

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler about supported data layoutsV2 of specific kernel
 *
 *   @param paramsV2            [in]        Device ID, GUID, nodeParams, tensors geometry
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
typedef GlueCodeReturn_t(*pfnGetSupportedDataLayoutV2)
(
    _IN_     HabanaKernelParamsV2_t* paramsV2,
    _OUT_    NodeDataLayoutsV2*      supportedLayoutsV2,
    _INOUT_  unsigned*               layoutCount
);



} /* name space gcapi */

#endif /* GRAPH_COMPILER_INTERFACE_H */
