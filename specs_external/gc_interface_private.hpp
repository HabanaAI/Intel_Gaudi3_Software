/*****************************************************************************
* Copyright (C) 2017 HabanaLabs, Ltd.
* All Rights Reserved.
*
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
*
* Authors:
* Tzachi Cohen <tcohen@gaudilabs.com>
******************************************************************************
*/

/*
 This file holds private interface to the kernel database on top of graph compiler.
 It is used to debug/profile kernels in  the database.
*/

#ifndef GRAPH_COMPILER_INTERFACE_PRIVATE_H
#define GRAPH_COMPILER_INTERFACE_PRIVATE_H

#include <memory>
#include <vector>
#include <set>

#include "gc_interface.h"

// TODO: insert this struct under name space gcapi
typedef struct _DebugFlags {
  union {
    struct {
      // By default the database will return the best perfoming version of a kernel.
      // This flag forces thedata base to return the asm version of kernel if exists.
      unsigned  forceAsmVersion :1;
      // This flag forces the database to return the C version of kernel if exists.
      // This flag is mutually exclusive with 'forceAsmVersion'.
      unsigned  forceCVersion :1;
      // This flag forces the database to return the default C implementation.
      // This flag is mutually exclusive with 'forceAsmVersion'.
      unsigned  noSpecializations:1;
      unsigned  :29; //reserved .
    };
    unsigned   Value;
  };
} DebugFlags;


namespace gcapi
{

//Directly mirrors the enum defined in synapse_common_types.h
typedef enum {
  DATA_TENSOR = 0,
  SHAPE_TENSOR,
  OUTPUT_DESCRIBING_SHAPE_TENSOR = SHAPE_TENSOR,
  INPUT_DESCRIBING_SHAPE_TENSOR,
  DATA_TENSOR_DYNAMIC,
  DEVICE_SHAPE_TENSOR,
  HOST_SHAPE_TENSOR,
  HOST_TO_DEVICE_TENSOR,
  TENSOR_TYPE_MAX
} CommonTensorType;

struct CommonTensorAttributes
{
    CommonTensorType type;
    bool             isNotNeeded;   //true -> this tensor is an output of the op and isn't consumed again
    bool             isInitialized; //true -> this tensor contains data in its m_ptr field
};

struct CommonTensor : public Tensor_t
{
    /*
    * this is to distinguish several graph inputs/outputs
    * of identical parameters
    */
    unsigned    uniqueIdentifier;
    bool        isPersistent;
};

typedef enum
{
    SECTION_PERSISTENT,
    SECTION_RMW,
    SECTION_WORKSPACE
} CommonSectionType;

struct CommonSection
{
    CommonSectionType type;
    uint32_t          id;     //must be zero for workspace sections
    uint64_t          offset; //must be zero for workspace sections
};

typedef struct _CommonTensorV2_t : public TensorV2_t
{
    /*
    * this is to distinguish several graph inputs/outputs
    * of identical parameters
    */
    unsigned               uniqueIdentifier;
    CommonSection          section;
    CommonTensorAttributes attributes;
    unsigned               strides[MAX_TENSOR_DIM];
}CommonTensorV2_t;

typedef struct _CommonTensorV3_t : public TensorV3_t // high rank Common tensor
{
    /*
    * this is to distinguish several graph inputs/outputs
    * of identical parameters
    */
    unsigned               uniqueIdentifier;
    CommonSection          section;
    CommonTensorAttributes attributes;
    unsigned               strides[MAX_HABANA_DIM];
}CommonTensorV3_t;

typedef struct _CommonTensorV4_t : public TensorV4_t {
  unsigned uniqueIdentifier;
  CommonSection section;
  CommonTensorAttributes attributes;
  uint64_t strides[MAX_HABANA_DIM];
} CommonTensorV4_t;

struct CommonEdge;
typedef struct _CommonEdgeV2_t CommonEdgeV2_t;


struct CommonNode
{
    char                            guid[MAX_NODE_NAME];
    char                            nodeName[MAX_NODE_NAME];
    UserParams_t                    nodeParams;
    std::vector<CommonEdge>         inputEdges;
    std::vector<CommonEdge>         outputEdges;
    unsigned                        uniqueIdentifier;
    unsigned                        paramsSize;
    bool                            isShapeManipulationOp; // indication if an op is shape manipulation operation, when isShapeManipulationOp=1 it means that the op is shapeManipulation
};

typedef struct _CommonNodeV2_t
{
    char                            guid[MAX_NODE_NAME];
    char                            nodeName[MAX_NODE_NAME];
    UserParams_t                    nodeParams;
    std::vector<CommonEdgeV2_t>     inputEdges;
    std::vector<CommonEdgeV2_t>     outputEdges;
    unsigned                        uniqueIdentifier;
    unsigned                        paramsSize;
    std::set<unsigned>              controlEdgesToNode; //Contains "unique Identifiers" of other nodes
    /*
     * Only set below flag if node has complex guid but no need to further extract it.
     * Flag set will indicate GC not to invoke another extraction on this node, thus avoiding endless recursion.
     */
    bool                            handleAsSimpleNode;
    std::vector<unsigned>           fusedIdentifiers; // Use this field in fused kernels to track what unique IDs it originated from.
    std::vector<unsigned>           newIdentifiers; // Use this field in fused kernels to track new nodes IDs created.
    bool                            isShapeManipulationOp; // indication if an op is shape manipulation operation, when isShapeManipulationOp=1 it means that the op is shapeManipulation
}CommonNodeV2_t;

typedef struct _CommonEdgeV4_t CommonEdgeV4_t;
typedef struct _CommonEdgeV3_t CommonEdgeV3_t;
typedef struct _CommonEdgeV2_t CommonEdgeV2_t;

typedef struct _CommonNodeV3_t
{
    char                            guid[MAX_NODE_NAME];
    char                            nodeName[MAX_HABANA_NODE_NAME];
    UserParams_t                    nodeParams;
    std::vector<CommonEdgeV3_t>     inputEdges;
    std::vector<CommonEdgeV3_t>     outputEdges;
    unsigned                        uniqueIdentifier;
    unsigned                        paramsSize;
    std::set<unsigned>              controlEdgesToNode; //Contains "unique Identifiers" of other nodes
    /*
     * Only set below flag if node has complex guid but no need to further extract it.
     * Flag set will indicate GC not to invoke another extraction on this node, thus avoiding endless recursion.
     */
    bool                            handleAsSimpleNode;
    std::vector<unsigned>           fusedIdentifiers; // Use this field in fused kernels to track what unique IDs it originated from.
    std::vector<unsigned>           newIdentifiers;   // Use this field in fused kernels to track new nodes IDs created.
    bool                            useDeterministic; // when useDeterministic=1, cguid shall extract determistic subgraph
    unsigned                        originalComplexGuidId; // The ComplexGUID node ID that this node was extracted from. -1 means none.
    bool                            isShapeManipulationOp; // indication if an op is shape manipulation operation, when isShapeManipulationOp=1 it means that the op is shapeManipulation
}CommonNodeV3_t;

typedef struct _CommonNodeV4_t
{
    char                            guid[MAX_NODE_NAME];
    char                            nodeName[MAX_HABANA_NODE_NAME];
    UserParams_t                    nodeParams = nullptr;
    std::vector<CommonEdgeV4_t>     inputEdges;
    std::vector<CommonEdgeV4_t>     outputEdges;
    unsigned                        uniqueIdentifier = 0;
    unsigned                        paramsSize = 0;
    std::set<unsigned>              controlEdgesToNode;
    bool                            handleAsSimpleNode = false;
    std::vector<unsigned>           fusedIdentifiers;
    std::vector<unsigned>           newIdentifiers;
    bool                            useDeterministic = false;
    unsigned                        originalComplexGuidId = 0;
    bool                            isShapeManipulationOp = false; // indication if an op is shape manipulation operation, when isShapeManipulationOp=1 it means that the op is shapeManipulation
    char                            originalComplexGuid[MAX_NODE_NAME];
}CommonNodeV4_t;

struct CommonEdge
{
   std::weak_ptr<CommonNode>         targetNode; // in inputEdge - producer Node. in outputEdge - consumer Node.
   std::shared_ptr<CommonTensor>     tensor; // tensor associated with the edge.
};

typedef struct _CommonEdgeV2_t
{
    std::weak_ptr<CommonNodeV2_t>         targetNode; // in inputEdge - producer Node. in outputEdge - consumer Node.
    std::shared_ptr<CommonTensorV2_t>     tensor; // tensor associated with the edge.
}CommonEdgeV2_t;

typedef struct _CommonEdgeV3_t
{
    std::weak_ptr<CommonNodeV3_t>         targetNode; // in inputEdge - producer Node. in outputEdge - consumer Node.
    std::shared_ptr<CommonTensorV3_t>     tensor; // tensor associated with the edge.
}CommonEdgeV3_t;

typedef struct _CommonEdgeV4_t {
  std::weak_ptr<CommonNodeV4_t> targetNode;
  std::shared_ptr<CommonTensorV4_t> tensor;
} CommonEdgeV4_t;

struct CommonGraph
{
    DeviceId_t                                deviceId;
    KernelType_t                              kernelType;
    std::vector<std::shared_ptr<CommonNode>>  nodes;
};

typedef struct _CommonGraphV2_t
{
    DeviceId_t                                     deviceId;
    KernelType_t                                   kernelType;
    std::vector<std::shared_ptr<CommonNodeV2_t>>   nodes;
    unsigned                                       maxAvailableTpc;     // cguid writer should know that it will get any number between 1 and maxAvailableTpc
}CommonGraphV2_t;

typedef struct _CommonGraphV3_t
{
    DeviceId_t                                     deviceId;
    KernelType_t                                   kernelType;
    std::vector<std::shared_ptr<CommonNodeV3_t>>   nodes;
    unsigned                                       maxAvailableTpc;     // cguid writer should know that it will get any number between 1 and maxAvailableTpc
    unsigned                                       eagerMode;           // indication whether we are in eager mode,
                                                                        // when eagerMode=0 means graph mode, eagerMode=1 means eager mode
}CommonGraphV3_t;

typedef struct _CommonGraphV4_t {
    DeviceId_t                                     deviceId;
    KernelType_t                                   kernelType;
    std::vector<std::shared_ptr<CommonNodeV4_t>>   nodes;
    unsigned                                       maxAvailableTpc;     // cguid writer should know that it will get any number between 1 and maxAvailableTpc
    unsigned                                       eagerMode;           // indication whether we are in eager mode,
                                                                        // when eagerMode=0 means graph mode, eagerMode=1 means eager mode
} CommonGraphV4_t;

// Define an alias for types of the form CommonXXX to FuserXXX.
#define DEF_FuserAlias(type)        \
  typedef Common##type Fuser##type;

DEF_FuserAlias(TensorType)
DEF_FuserAlias(TensorAttributes)
DEF_FuserAlias(Tensor)
DEF_FuserAlias(SectionType)
DEF_FuserAlias(Section)
DEF_FuserAlias(Edge)
DEF_FuserAlias(Node)
DEF_FuserAlias(Graph)

DEF_FuserAlias(TensorV2_t)
DEF_FuserAlias(NodeV2_t)
DEF_FuserAlias(EdgeV2_t)
DEF_FuserAlias(GraphV2_t)

DEF_FuserAlias(TensorV3_t)
DEF_FuserAlias(NodeV3_t)
DEF_FuserAlias(EdgeV3_t)
DEF_FuserAlias(GraphV3_t)

DEF_FuserAlias(TensorV4_t)
DEF_FuserAlias(NodeV4_t)
DEF_FuserAlias(EdgeV4_t)
DEF_FuserAlias(GraphV4_t)

typedef enum _FuserRetVal_t
{
    FUSER_SUCCESS = 0,
    FUSER_FAILED
}FuserRetVal_t;

typedef _CommonNodeV2_t  _FuserNodeV2_t;
typedef _CommonGraphV2_t _FuserGraphV2_t;
typedef _CommonNodeV3_t  _FuserNodeV3_t;
typedef _CommonGraphV3_t _FuserGraphV3_t;
typedef _CommonNodeV4_t  _FuserNodeV4_t;
typedef _CommonGraphV4_t _FuserGraphV4_t;

#define FUSER_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME          "FuseGraphV2"
#define RELEASE_GRAPH_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME  "ReleaseFusedGraphV2"
#define FUSER_GET_FUSED_NODE_PREGRAPH_ENTRY_POINT_NAME            "GetFusedNodePreGraph"

#define FUSER_SUPPORTING_DYNAMIC_SHAPES_V3_ENTRY_POINT_NAME          "FuseGraphV3"
#define RELEASE_GRAPH_SUPPORTING_DYNAMIC_SHAPES_V3_ENTRY_POINT_NAME  "ReleaseFusedGraphV3"
#define FUSER_GET_FUSED_NODE_PREGRAPH_V3_ENTRY_POINT_NAME            "GetFusedNodePreGraphV3"

#define FUSER_SUPPORTING_DYNAMIC_SHAPES_V4_ENTRY_POINT_NAME          "FuseGraphV4"
#define RELEASE_GRAPH_SUPPORTING_DYNAMIC_SHAPES_V4_ENTRY_POINT_NAME  "ReleaseFusedGraphV4"
#define FUSER_GET_FUSED_NODE_PREGRAPH_V4_ENTRY_POINT_NAME            "GetFusedNodePreGraphV4"

typedef FuserRetVal_t (*pfnFuseGraphV2)(const CommonGraphV2_t* graphIn, CommonGraphV2_t* graphOut, bool debug);
typedef FuserRetVal_t (*pfnReleaseFuseGraphV2)(CommonGraphV2_t* graphOut);

typedef FuserRetVal_t (*pfnFuseGraphV3)(const CommonGraphV3_t* graphIn, CommonGraphV3_t* graphOut, bool debug);
typedef FuserRetVal_t (*pfnReleaseFuseGraphV3)(CommonGraphV3_t* graphOut);

typedef FuserRetVal_t (*pfnFuseGraphV4)(const CommonGraphV4_t* graphIn, CommonGraphV4_t* graphOut, bool debug);
typedef FuserRetVal_t (*pfnReleaseFuseGraphV4)(CommonGraphV4_t* graphOut);

// Extraction of non-supported guids and tensor geometry (such as NMS guid or high-rank geometry) using graph input
#define COMPLEX_GUID_ENTRY_POINT_NAME                                   "ExtractComplexGUID"
// Delete resources of CommonGraph
#define COMPLEX_GUID_CLEAR_GRAPH_ENTRY_POINT_NAME                       "ClearComplexGUID"
// Identical to the ones above (should be replaced when change is stable).
#define COMPLEX_GUID_V4_ENTRY_POINT_NAME                                "ExtractComplexGUIDV4"
#define COMPLEX_GUID_V4_FUNCTIONAL_ENTRY_POINT_NAME                     "ExtractFunctionalComplexGUIDV4"
#define COMPLEX_GUID_V4_PERFORMANCE_ENTRY_POINT_NAME                    "ExtractPerformanceComplexGUIDV4"
#define COMPLEX_GUID_V4_CLEAR_GRAPH_ENTRY_POINT_NAME                    "ClearComplexGUIDV4"
// Get supported complex guid names
#define COMPLEX_GUID_NAMES_ENTRY_POINT_NAME                             "GetSupportedComplexGUIDs"
// Get supported functional complex guid names
#define COMPLEX_GUID_NAMES_FUNCTIONAL_ENTRY_POINT_NAME                  "GetSupportedFunctionalComplexGUIDs"
// Get supported performance complex guid names
#define COMPLEX_GUID_NAMES_PERFORMANCE_ENTRY_POINT_NAME                 "GetSupportedPerformanceComplexGUIDs"
// Get Dynamic Shape supported complex guid names
#define COMPLEX_GUID_NAMES_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME   "GetDynamicShapeSupportedComplexGUIDs"
// Get supported data layout for complex guids
#define COMPLEX_GUID_GET_SUPPORTED_LAYOUTS_ENTRY_POINT_NAME             "GetSupportedComplexGUIDLayouts"

typedef GlueCodeReturn_t (*pfnExtractComplexGUID)(const CommonGraphV3_t* graphIn, CommonGraphV3_t** graphOut, bool debug);
typedef GlueCodeReturn_t (*pfnReleaseExtractedSubgraph)(CommonGraphV2_t* graphOut);
typedef GlueCodeReturn_t (*pfnClearExtractedSubgraph)(CommonGraphV3_t* graphOut);

typedef GlueCodeReturn_t (*pfnExtractComplexGUIDV4)(const CommonGraphV4_t* graphIn, CommonGraphV4_t** graphOut, bool debug);
typedef GlueCodeReturn_t (*pfnExtractFunctionalComplexGUIDV4)(const CommonGraphV4_t* graphIn, CommonGraphV4_t** graphOut, bool debug);
typedef GlueCodeReturn_t (*pfnExtractPerformanceComplexGUIDV4)(const CommonGraphV4_t* graphIn, CommonGraphV4_t** graphOut, bool debug);
typedef GlueCodeReturn_t (*pfnClearExtractedSubgraphV4)(CommonGraphV4_t* graphOut);

typedef GlueCodeReturn_t (*pfnGetSupportedComplexGUIDs)(char** GUIDs, unsigned* GUIDCount, DeviceId_t deviceId);
typedef GlueCodeReturn_t (*pfnGetSupportedFunctionalComplexGUIDs)(char** GUIDs, unsigned* GUIDCount, DeviceId_t deviceId);
typedef GlueCodeReturn_t (*pfnGetSupportedPerformanceComplexGUIDs)(char** GUIDs, unsigned* GUIDCount, DeviceId_t deviceId);
typedef GlueCodeReturn_t (*pfnGetDynamicShapeSupportedComplexGUIDs)(
    char **GUIDs, unsigned *GUIDCount, DeviceId_t deviceId);
typedef GlueCodeReturn_t (*pfnGetSupportedComplexGUIDLayouts)(
    DeviceId_t deviceId, NodeName *nodeName, NodeDataLayouts *supportedLayouts,
    unsigned *layoutCount);

#define COMPLEX_GUID_GET_SUPPORTED_LAYOUTS_V2_ENTRY_POINT_NAME             "GetSupportedComplexGUIDLayoutsV2"

typedef GlueCodeReturn_t (*pfnGetSupportedComplexGUIDLayoutsV2)(
    DeviceId_t              deviceId,
    NodeName *              nodeName,
    NodeDataLayoutsV3 *     supportedLayouts,
    unsigned *              layoutCount
);

#define COMPLEX_GUID_GET_SUPPORTED_LAYOUTS_V3_ENTRY_POINT_NAME             "GetSupportedComplexGUIDLayoutsV3"
typedef GlueCodeReturn_t (*pfnGetSupportedComplexGUIDLayoutsV3)(
    CommonGraphV3_t *       graphIn,
    NodeDataLayoutsV3 *     supportedLayouts,
    unsigned *              layoutCount
);

#define COMPLEX_GUID_GET_SUPPORTED_LAYOUTS_V4_ENTRY_POINT_NAME             "GetSupportedComplexGUIDLayoutsV4"

typedef GlueCodeReturn_t (*pfnGetSupportedComplexGUIDLayoutsV4)(
    CommonGraphV4_t *       graphIn,
    NodeDataLayoutsV3 *     supportedLayouts,
    unsigned *              layoutCount
);

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler which pre-graph was replaced with a given fused node
 *
 *   @param fusedNode         [in]        The fused node from the fuser post-graph
 *   @param preGraphOut       [out]       The original pre-graph which was replaced by the fused node
 *   @return               The status of the operation
 **************************************************************************************************
 */
typedef GlueCodeReturn_t (*pfnGetFusedNodePreGraph)
(
    _IN_  const CommonNodeV2_t*  fusedNode,
    _OUT_ CommonGraphV2_t**      preGraphOut
);

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler which pre-graph was replaced with a given fused node,
 *   with V3 objects
 *
 *   @param fusedNode         [in]        The fused node from the fuser post-graph
 *   @param preGraphOut       [out]       The original pre-graph which was replaced by the fused node
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn_t (*pfnGetFusedNodePreGraphV3)
(
    _IN_  const CommonNodeV3_t*  fusedNode,
    _OUT_ CommonGraphV3_t**      preGraphOut
);

/*
 ***************************************************************************************************
 *   @brief Informs graph compiler which pre-graph was replaced with a given fused node,
 *   with V4 objects
 *
 *   @param fusedNode         [in]        The fused node from the fuser post-graph
 *   @param preGraphOut       [out]       The original pre-graph which was replaced by the fused node
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn_t (*pfnGetFusedNodePreGraphV4)
(
    _IN_  const CommonNodeV4_t*  fusedNode,
    _OUT_ CommonGraphV4_t**      preGraphOut
);

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
    union
    {
        unsigned permutation [MAX_TENSOR_DIM]; /* permutation semantics should match
                                                *  numpy.transpose() input*/
        unsigned newShape    [MAX_TENSOR_DIM];
    };
} TensorOperation;

/* TensorManipulationSuggestion example:
 *
 * Say we had an elementwise operation with one input and one output that
 * preseves the shape. Say that we wanted to make the fastest changing
 * dimension of the input bigger. For example we want to change the shape
 * [10, 20, 30, 40] into [200, 1, 30, 40].
 *
 * Then the TensorManipulationSuggestion struct will look like this:
 *
 *  TensorManipulationSuggestion
 *  {
 *    .inputTensors =
 *    {
 *      TensorOperation
 *      {
 *        .opType = TENSOR_OP_RESHAPE,
 *        .newShape = {200, 1, 30, 40}
 *      }
 *    }
 *    .outputTensors =
 *    {
 *      TensorOperation
 *      {
 *        .opType = TENSOR_OP_RESHAPE,
 *        .newShape = {200, 1, 30, 40}
 *      }
 *    }
 *  }
 *
 * Note that outputTensors is required, otherwise in the general case
 * the consumer of the API can't guess what the shapes of output are,
 * given the inputs.
 */

typedef struct _TensorManipulationSuggestion
{
    _INOUT_  TensorOperation     inputTensors[MAX_TENSOR_NR];
    _INOUT_  TensorOperation     outputTensors[MAX_TENSOR_NR];
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

#define GET_SUGGESTED_TENSOR_MANIPULATION_ENTRY_POINT_NAME "GetSuggestedTensorManipulation"

typedef GlueCodeReturn_t(*pfnGetSuggestedTensorManipulation)
(
    _IN_     const HabanaKernelParams_t*      params,
    _OUT_    TensorManipulationSuggestion*    suggestion);


#define GET_SUGGESTED_TENSOR_MANIPULATION_SUPPORTING_DYNAMIC_SHAPES_ENTRY_POINT_NAME "GetSuggestedTensorManipulationV2"

typedef GlueCodeReturn_t(*pfnGetSuggestedTensorManipulationV2)
(
    _IN_     const HabanaKernelParamsV2_t*    params,
    _OUT_    TensorManipulationSuggestion*    suggestion);

///////////////////////////////////////////////////////////////////////////////////////////////////

#define GET_COMPLEX_NAMES_ENTRY_POINT_NAME          "GetComplexOperatorNames"
#define GET_COMPLEX_OPERATOR_GRAPH_ENTRY_POINT_NAME "GetComplexOperatorGraph"
#define RELEASE_COMPLEX_OPERATOR_GRAPH_ENTRY_POINT_NAME  "ReleaseComplexOperatorGraph"
/*
 ***************************************************************************************************
 *   @brief Returns to graph compiler a list of complex guids supported by perf-lib.
 *
 *   @param names            [out]      Array of available complex kernel guids. The array should be
 *                                      allocated by the caller.
 *   @param kernelCount      [in/out]   [in] - Size of 'names' arrray. [out] - number of entries populated.
 *                                      If 'names' is null, the required size of 'names' is returned.
 *   @param deviceId         [in]       Relevant device ID
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn_t(*pfnGetComplexOperatorNames)
(
    _OUT_   char*         names[MAX_NODE_NAME],
    _INOUT_ unsigned*     kernelCount,
    _IN_    DeviceId_t    deviceId
);

/*
 ***************************************************************************************************
 *   @brief Given a complex operator instantiation parameters, the function returns a sub-graph
 *          implementing the complex operator out of simple operators.
 *
 *   @param params          [in]   Input structure identical to the one supplied for kernel
 *                                 instantiation entry point.
 *   @param graphOut        [out]  A graph of simple operators implementing the complex operation.
 *                                 The graph should be later released using "ReleaseComplexOperatorGraph" call.
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn_t(*pfnGetComplexOperatorGraph)
(
    _IN_     const HabanaKernelParams_t*    params,
    _OUT_    CommonGraph**                   graphOut
);

/*
 ***************************************************************************************************
 *   @brief Release a simple operators graph which implemented by the complex operation.
 *
 *   @param graphOut        [in]  A graph of simple operators implementing the complex operation,
 *                                to be release.
 *   @return               The status of the operation
 ***************************************************************************************************
 */


typedef GlueCodeReturn_t(*pfnReleaseComplexOperatorGraph)
(
     _IN_    CommonGraph* graphOut
);

#define GET_SHAPE_INFERENCE_ENTRY_POINT_NAME "GetShapeInference"
#define GET_SHAPE_INFERENCE_V2_ENTRY_POINT_NAME "GetShapeInferenceV2" //64bit support
#define GET_SHAPE_INFERENCE_UNIQUE_ID_ENTRY_POINT_NAME "GetShapeInferenceUniqueId"
#define GET_SHAPE_INFERENCE_FUNCTION_ENTRY_POINT_NAME "GetShapeInferenceFunction"
#define GET_SHAPE_INFERENCE_LIB_VERSION_ENTRY_POINT_NAME "GetShapeInferenceLibVersion"

typedef enum _ShapeInferenceReturn_t
{
    SIF_SUCCESS                            = 0 ,
    SIF_NODE_NOT_FOUND                     = 1 ,
    SIF_INCOMPATIBLE_INPUT_COUNT           = 2 ,
    SIF_INCOMPATIBLE_INPUT_SIZE            = 3 ,
    SIF_INCOMPATIBLE_OUTPUT_COUNT          = 4 ,
    SIF_INCOMPATIBLE_OUTPUT_SIZE           = 5 ,
    SIF_MISSING_PRIVATE_STRUCTURE          = 6 ,
    SIF_INVALID_SHAPE_INFERENCE_ID         = 7 ,
    SIF_NULL_PTR                           = 8 ,
    SIF_FAILED                             = 400,
} ShapeInferenceReturn_t;



typedef struct _TensorShapeInfo
{
  unsigned sizes[MAX_TENSOR_DIM]; // in elements
  unsigned dims;
  unsigned *hostAddress; // can be null
  CommonTensorType type;
  unsigned reserved[7];
} TensorShapeInfo;

// Adding V2 to allow 64bit support
typedef struct _TensorShapeInfoV2
{
  uint64_t sizes[MAX_TENSOR_DIM]; // in elements
  unsigned dims;
  unsigned *hostAddress; // can be null
  CommonTensorType type;
  unsigned reserved[7];
} TensorShapeInfoV2;

typedef struct _UniqueShapeInferenceId {
    union {
        struct {
            unsigned  shapeInferenceUniqueId :16; // should be filled by glueCode
            unsigned  :8; //reserved
            unsigned  sharedObjectId :8; // reserved - used by graphCompiler

        };
        unsigned   Value;
    };
} UniqueShapeInferenceId;

typedef struct _NodeTensorPermutation {
    unsigned permutation[MAX_TENSOR_DIM];
} NodeTensorPermutation;


typedef struct _ShapeInferenceParams {
    unsigned              inputTensorsNr;         // The number of input tensors shapes
    TensorShapeInfo       **inputTensors;         // Pointer to array of the input tensor shapes
    unsigned              nodeParamsSize;         // User specific params size
    UserParams_t          nodeParams;             // User specific params
    unsigned              outputTensorsNr;        // The number of output tensors shapes that was set during graph
                                                  // creation according to max bucket size.
    NodeTensorPermutation *inputPermutations;     // Either null, meaning all permutations are trivial,
                                                  // or inputTensorsNr permutations for all input tensors.
    NodeTensorPermutation *outputPermutations;    // Ditto for output tensors.
    unsigned              reserved[2];
} ShapeInferenceParams;

// Adding V2 to allow 64bit support
typedef struct _ShapeInferenceParamsV2 {
    unsigned              inputTensorsNr;         // The number of input tensors shapes
    TensorShapeInfoV2     **inputTensors;         // Pointer to array of the input tensor shapes
    unsigned              nodeParamsSize;         // User specific params size
    UserParams_t          nodeParams;             // User specific params
    unsigned              outputTensorsNr;        // The number of output tensors shapes that was set during graph
                                                  // creation according to max bucket size.
    NodeTensorPermutation *inputPermutations;     // Either null, meaning all permutations are trivial,
                                                  // or inputTensorsNr permutations for all input tensors.
    NodeTensorPermutation *outputPermutations;    // Ditto for output tensors.
    unsigned              reserved[2];
} ShapeInferenceParamsV2;


typedef struct _ShapeInferenceOutput {
    TensorShapeInfo         **outputTensors; // Pointer to array of the output tensor shapes
    unsigned                *invalidMask;    // invalid bit mask per output shape tensor
                                             // only bits 0...('outputTensorsNr'-1) are meaningful.
                                             // Assumed that the invalidMask is initialized to 0 by the caller.
                                             // E.g. if the bit0 (LSB) = 1,then output shape
                                             // tensor0 is invalid.
    unsigned                reserved[2];
} ShapeInferenceOutput;

// Adding V2 to allow 64bit support
typedef struct _ShapeInferenceOutputV2 {
    TensorShapeInfoV2       **outputTensors; // Pointer to array of the output tensor shapes
    unsigned                *invalidMask;    // invalid bit mask per output shape tensor
                                             // only bits 0...('outputTensorsNr'-1) are meaningful.
                                             // Assumed that the invalidMask is initialized to 0 by the caller.
                                             // E.g. if the bit0 (LSB) = 1,then output shape
                                             // tensor0 is invalid.
    unsigned                reserved[2];
} ShapeInferenceOutputV2;

/*
 ***************************************************************************************************
 *   @brief shape inference method which will be invoked in synapse runtime
 *
 *   @param inputParams     [in]        shape inference input params
 *   @param outputData      [out]       shape inference calculated outputs
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef ShapeInferenceReturn_t(*pfnGetShapeInference)
(
        _IN_     const ShapeInferenceParams *inputParams,
        _OUT_          ShapeInferenceOutput *outputData
);


// Adding V2 to allow 64bit support
typedef ShapeInferenceReturn_t(*pfnGetShapeInferenceV2)
(
        _IN_     const ShapeInferenceParamsV2 *inputParams,
        _OUT_          ShapeInferenceOutputV2 *outputData
);


/*
 ***************************************************************************************************
 *   @brief supplies graph compiler a shape inference unique identifier of specific kernel which
 *   will be used in runtime init to build the global shapeInference data base.
 *
 *   @param deviceId                      [in]     Asic ID
 *   @param nodeName                      [in]     Kernel GUID.
 *   @param uniqueShapeInferenceId        [out]    shape inference unique identifier which will be converted to
 *                                                         pointer function to be called in runtime
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */


typedef ShapeInferenceReturn_t(*pfnGetShapeInferenceUniqueId)
(
        _IN_     const DeviceId_t         deviceId,
        _IN_     const NodeName           *nodeName,
        _OUT_    UniqueShapeInferenceId   *uniqueShapeInferenceId
);


/*
 ***************************************************************************************************
 *   @brief supplies graph compiler a shape inference function of specific kernel which will be
 *   invoked in runtime.
 *
 *   @param uniqueShapeInferenceId        [in]     shape inference unique identifier which will be converted to
 *                                                         pointer function to be called in runtime
 *   @param shapeInferenceFunction        [out]    shape inference pointer function to be called in runtime
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef ShapeInferenceReturn_t(*pfnGetShapeInferenceFunction)
(
        _IN_     const UniqueShapeInferenceId  uniqueShapeInferenceId,
        _OUT_    pfnGetShapeInference          *shapeInferenceFunction
);

/*
 ***************************************************************************************************
 *   @brief supplies graph compiler a shape inference library version for coherency checks
 *   between graph compiler and runtime
 *
 *   @param shapeInferenceLibVersion        [out]    shape inference library version
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef ShapeInferenceReturn_t(*pfnGetShapeInferenceLibVersion)
(
        _OUT_    unsigned   *shapeInferenceLibVersion
);

#define GET_KERNEL_BINARIES_ENTRY_POINT_NAME "GetKernelISAs"

typedef struct _DeviceKernelIsa {
    const char *name;
    const void *isa;
    uint64_t    isaSize;
} DeviceKernelIsa_t;

/*
 ***************************************************************************************************
 *   @brief Retrieve a list of all kernels ISA binaries
 *
 *   @param deviceId            [in]        Device ID
 *   @param isaArray            [out]       Array to store returned ISA pointers info.
 *   @param layoutCount         [in/out]    in  - Size of 'isaArray' array.
 *                                          out - Total of avaialble ISA kernels.
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */

typedef GlueCodeReturn_t(*pfnGetKernelISAs)
(
    _IN_     DeviceId_t         deviceId,
    _OUT_    DeviceKernelIsa_t* isaArray,
    _INOUT_  unsigned*          isaArrayCount
);

typedef struct _FusedKernelParams_
{
    char sif[gcapi::MAX_USER_PARAMS_SIZE];
}FusedKernelParams;

} /* name space gcapi */

#endif //GRAPH_COMPILER_INTERFACE_PRIVATE_H
