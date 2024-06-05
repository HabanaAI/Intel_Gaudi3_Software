#ifndef SYNAPSE_API_TYPES_H
#define SYNAPSE_API_TYPES_H

#include "synapse_common_types.h"

static const uint32_t MAX_TENSOR_NR        = 16;
static const unsigned MAX_USER_PARAMS_SIZE = 128;
typedef uint64_t tensor_size_t;

typedef enum
{
    DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS,
    DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS,
    DEVICE_ATTRIBUTE_SRAM_SIZE,
    DEVICE_ATTRIBUTE_DRAM_SIZE,
    DEVICE_ATTRIBUTE_DRAM_FREE_SIZE,
    DEVICE_ATTRIBUTE_TPC_ENABLED_MASK,
    DEVICE_ATTRIBUTE_DRAM_ENABLED,
    DEVICE_ATTRIBUTE_DEVICE_TYPE,
    DEVICE_ATTRIBUTE_CLK_RATE,
    DEVICE_ATTRIBUTE_MAX_RMW_SIZE,
    DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE,
    DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE,
    DEVICE_ATTRIBUTE_MAX_DIMS
} synDeviceAttribute;

typedef enum
{
    MEMORY_ATTRIBUTE_DEVICE     = 1, // hbm or dram
    MEMORY_ATTRIBUTE_HOST       = 2, // currently not supported in Gaudi
    MEMORY_ATTRIBUTE_SRAM       = 4, // currently not supported
    MEMORY_ATTRIBUTE_PERSISTENT = 8, // tensor will stay in memory beyond the lifetime of the graph
} synMemoryAttribute;

typedef enum
{
    RECIPE_ATTRIBUTE_WORKSPACE_SIZE,
    RECIPE_ATTRIBUTE_NUM_PERSISTENT_TENSORS,
    RECIPE_ATTRIBUTE_HOST_MEM_SIZE,
    RECIPE_ATTRIBUTE_NUM_EXTERNAL_TENSORS,
    RECIPE_ATTRIBUTE_PERSISTENT_TENSORS_SIZE,
    RECIPE_ATTRIBUTE_CONST_SECTIONS_SIZE,
    RECIPE_ATTRIBUTE_DEVICE_MEMORY_SIZE,
    RECIPE_ATTRIBUTE_MAX
} synRecipeAttribute;

typedef enum
{
    SECTION_SIZE,
    SECTION_DATA,
    IS_CONST
} synSectionProp;

typedef enum
{
    GRAPH_ATTRIBUTE_INFERENCE,
    GRAPH_ATTRIBUTE_QUANTIZATION,
    GRAPH_ATTRIBUTE_BACKOFF_FACTOR,
    GRAPH_ATTRIBUTE_MAX
} synGraphAttribute;

typedef union
{
    uint64_t iAttrVal;
    double   dAttrVal;
} synGraphAttributeVal;

struct InternalRecipeHandle;
struct SectionHandleExternal; // this is a type that is not defined, used to create synSectionHandle
struct InternalStreamHandle;
struct InternalGraphHandle;
struct EventInterfaceExternal;

typedef struct EventInterfaceExternal*  synEventHandle;
typedef struct InternalRecipeHandle*    synRecipeHandle;

typedef struct SectionHandleExternal*   synSectionHandle;
typedef struct InternalGraphHandle*     synGraphHandle;
typedef uint32_t                        synModuleId;
typedef uint32_t                        synDeviceId;
typedef struct InternalStreamHandle*    synStreamHandle;
typedef uint64_t                        synNodeId;
typedef uint32_t                        synSectionId;


typedef struct
{
    const char*     tensorName;
    uint64_t        pTensorAddress;
    synTensorType   tensorType;
    TSize           tensorSize[HABANA_DIM_MAX];
    uint64_t        tensorId;
} synLaunchTensorInfo;

typedef struct
{
    const char*     tensorName;
    uint64_t        pTensorAddress;
    synTensorType   tensorType;
    TSize           tensorSize[HABANA_DIM_MAX];
    uint64_t        tensorId;
} synLaunchTensorInfoExt;

typedef struct
{
    char            tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
    uint64_t        tensorId;
    synTensorType   tensorType;
    synDataType     tensorDataType;
    uint32_t        tensorDims;
    TSize           tensorMaxSize[HABANA_DIM_MAX];
    TSize           tensorMinSize[HABANA_DIM_MAX];
    uint8_t         tensorPermutation[HABANA_DIM_MAX];
    synSectionId    tensorSectionId;
    uint64_t        tensorOffsetInSection;
    uint8_t         isInput;
    uint32_t        reserved[10];
} synRetrievedLaunchTensorInfo;

typedef struct
{
    char            tensorName[ENQUEUE_TENSOR_NAME_MAX_SIZE];
    uint64_t        tensorId;
    synTensorType   tensorType;
    synDataType     tensorDataType;
    uint32_t        tensorDims;
    TSize           tensorMaxSize[HABANA_DIM_MAX];
    TSize           tensorMinSize[HABANA_DIM_MAX];
    uint8_t         tensorPermutation[HABANA_DIM_MAX];
    synSectionId    tensorSectionId;
    uint64_t        tensorOffsetInSection;
    uint8_t         isInput;
    uint32_t        reserved[10];
} synRetrievedLaunchTensorInfoExt;

typedef struct
{
    uint64_t*   strides;
} TensorExUserContext;

typedef struct
{
    double min;
    double max;
} synQuantDynamicRange;

typedef struct
{
    synQuantDynamicRange* ranges;
    unsigned              numChannels;
} synPerChannelDynamicRange;

typedef enum
{
    SYN_QUANT_DYNAMIC_RANGE,
    SYN_QUANT_METADATA,
    SYN_FP_QUANT_METADATA,
    SYN_QUANT_FLAGS,
    SYN_QUANT_PC_DYNAMIC_RANGE
} synQuantizationProperty;

typedef struct
{
    double zp;
    double scale;
} synQuantZPScale;

typedef struct
{
    synDataType      dataType;
    synQuantZPScale* zpScales;     // An array of zero points and scales
    unsigned         numZPScales;  // The size of the array, value greater than 1 triggers per channel quantization
} synQuantMetadata;

typedef struct
{
    double   scale;
    unsigned expBias;
} synFpQuantParam;

typedef struct
{
    synDataType      dataType;
    synFpQuantParam* fpQuantParams;     // An array of scales and expBiases
    unsigned         numFpQuantParams;  // The size of the array, value greater than 1 triggers per channel quantization
} synFpQuantMetadata;

typedef struct
{
    uint8_t enablePerChannelQuant; //may automatic quant use per-channel
    uint8_t isSparsifiedWeights;   //mark tensor as sparsified weights for automatic quant (force zp=0)
    uint8_t isWeights;             //DEPRECATED mark the tensor as weight tensor
} synQuantFlags;

typedef struct
{
    TSize    sizes[HABANA_DIM_MAX];
    uint32_t dims;
} synTensorGeometry;

typedef struct
{
    TSize       sizes[HABANA_DIM_MAX];
    uint32_t    dims;
} synTensorGeometryExt;

typedef enum
{
    synGeometryMinSizes,
    synGeometryMaxSizes,
    synGeometrySizes = synGeometryMaxSizes,
    synGeometryDims
} synGeometryType;

// deprecated - no support for strided FCD
typedef struct
{
    uint32_t    strides[HABANA_DIM_MAX-1];
    synDataType deviceDataType;
} synTensorDeviceLayout;

typedef struct
{
    uint32_t    strides[HABANA_DIM_MAX];
    synDataType deviceDataType;
} synTensorDeviceFullLayout;


typedef struct
{
    TSize num_strides;
    TSize offset;
    TSize strides[SYN_MAX_TENSOR_DIM + 1];
} synDynamicStridedDmaH2dTensor;

 typedef struct
{
    TSize dims;
    TSize steps[SYN_MAX_TENSOR_DIM];
    TSize starts[SYN_MAX_TENSOR_DIM];
} synDynamicSliceDmaH2dTensor;

typedef struct
{
    uint8_t permutation[HABANA_DIM_MAX];
    uint8_t dims;
} synTensorPermutation;

enum eventCreateFlags {EVENT_COLLECT_TIME = 1};

typedef struct
{
    synTensor origHandle;
    synTensor newHandle;
} synTensorHandleMap;

typedef struct
{
    synNodeId origHandle;
    synNodeId newHandle;
} synNodeHandleMap;

/* Deprecated type */
typedef void CompilationAttribute;

static const uint32_t    INVALID_MODULE_ID     = 0xFFFFFFFF;
static const synDeviceId SYN_INVALID_DEVICE_ID = 0xFFFFFFFF;

#define GC20_CHANGE_FOR_SYNAPSE_PROFILER // For cross-promotion of synapse profiler. TODO: remove when finished

#endif /*SYNAPSE_API_TYPES_H*/