/*****************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Ehud Katz <ekatz@habana.ai>
 ******************************************************************************
 */

#ifndef GRAPH_COMPILER_PROTOCOL_HPP
#define GRAPH_COMPILER_PROTOCOL_HPP

#include "gc_interface_private.hpp"
#include "tpc_kernel_lib_interface.h"
#include <functional>

namespace gc_protocol
{
namespace utils
{
template<typename T>
struct ArrayView
{
    const T* data;
    size_t   size;

    template<typename U>
    ArrayView<T>& operator=(const U& obj)
    {
        data = obj.data();
        size = obj.size();
        return *this;
    }

    const T* begin() const { return data; }
    const T* end() const { return data + size; }
};
} /* namespace utils */

using ProtocolTensorType        = gcapi::CommonTensorType;  // eradiano: shall be removed once moved to tpc_lib_api
using ProtocolTensorSectionType = gcapi::CommonSectionType; // eradiano: shall be removed once moved to tpc_lib_api
using ProtocolTensorSection     = gcapi::CommonSection;     // eradiano: shall be removed once moved to tpc_lib_api

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
    TENSOR_TYPE_MAX,
    TENSOR_TYPE_INVALID = TENSOR_TYPE_MAX
} ProtocolTensorType_t;

typedef enum
{
    SECTION_PERSISTENT,
    SECTION_RMW,
    SECTION_WORKSPACE
} ProtocolTensorSectionType_t;

struct ProtocolTensorSection_t
{
    ProtocolTensorSectionType_t type;
    uint32_t                    id;     //must be zero for workspace sections
    uint64_t                    offset; //must be zero for workspace sections
};


struct ProtocolTensorQuantizationParams
{
  union
  {
      int    zeroPoint;
      int    fp8bias;
  };
  double scale;
};

// ProtocolTensorAttributes contains tensor properties that are feature-specific.
struct ProtocolTensorAttributes
{
  ProtocolTensorType                type; // eradiano: shall be removed once moved to tpc_lib_api
  // true -> this tensor is an output of the op and isn't consumed again
  bool                              isNotNeeded;
  // true -> this tensor has data in its ProtocolTensor::pData field
  bool                              isInitialized;
  // true -> this tensor is an output to the graph
  bool                              isGraphOutput;
  // Current type of the buffer pointed by ProtocolTensor::pData, as set by user / habana SW.
  // May differ from ProtocolTensor::elementType field.
  gcapi::TensorDataType_t           bufferDataType; // eradiano: shall be removed once moved to tpc_lib_api

  ProtocolTensorQuantizationParams* quantizationParams;
  ProtocolTensorType_t              tensorType;
  tpc_lib_api::TensorDataType       tensorDataType; // bufferDataType without GC1.0 dependency
};

constexpr uint64_t InvalidId = ~uint64_t(0);
constexpr uint64_t NewId     = ~uint64_t(1);

struct ProtocolTensor
{
    uint64_t                  id;
    utils::ArrayView<char>    name;
    // Expected type of the tensor data to be used as Node operand.
    // May differ from ProtocolTensorAttributes::bufferDataType.
    gcapi::TensorDataType_t   elementType; // eradiano: shall be removed once moved to tpc_lib_api

    // Equivalent to dimension
    unsigned                  rank;
    const uint64_t*           maxSizes;
    const uint64_t*           minSizes;
    const unsigned*           reserved;
    const uint64_t*           strides;
    // Points to buffer of data known at compilation time
    const void*               pData;
    ProtocolTensorSection*    section; // eradiano: shall be removed once moved to tpc_lib_api
    ProtocolTensorAttributes* attributes;
    // An array that holds description of any transpose operation applied to the tensor.
    const uint8_t*            perm;
    ProtocolTensorSection_t*  tensorSection;
    tpc_lib_api::TensorDataType elementDataType;// elementType without GC1.0 dependency
};

struct ProtocolNode
{
    uint64_t                    id;
    gcapi::TensorDataType_t     precisionType; // eradiano: shall be removed once moved to tpc_lib_api

    utils::ArrayView<char>      guid;
    utils::ArrayView<char>      name;
    gcapi::UserParams_t         params; // eradiano: shall be removed once moved to tpc_lib_api

    unsigned                    paramsSize;
    // When useDeterministic=true, cguid shall extract deterministic subgraph
    bool                        useDeterministic;
    // When isShapeManipulationOp=true if it is a shape manipulation operation
    bool                        isShapeManipulationOp;
    // Ids of original node(s) that this node was created from.
    utils::ArrayView<uint64_t>  replacedNodeIds;
    // Ids of nodes that are blocking this node execution with control dependency.
    utils::ArrayView<uint64_t>  blockingNodeIds;
    // Ids of new nodes that should be replaced toghther with this node.
    utils::ArrayView<uint64_t>  newAdjacentNodeIds;
    tpc_lib_api::TensorDataType precision; // precisionType without GC1.0 dependency
    tpc_lib_api::UserParams     userParams;// params without GC1.0 dependency

};

struct ProtocolNodeHandler
{
    virtual ~ProtocolNodeHandler() = default;
    virtual bool acceptNode(uint64_t nodeId) { return true; }
    virtual bool handleNode(const ProtocolNode& node) { return true; }
};

struct ProtocolInputTensorHandler
{
    virtual ~ProtocolInputTensorHandler() = default;
    virtual bool acceptInputTensor(uint64_t tensorId) { return true; }
    virtual bool handleInputTensor(const ProtocolTensor& tensor) { return true; }

    bool acceptTensor(uint64_t tensorId) { return acceptInputTensor(tensorId); }
    bool handleTensor(const ProtocolTensor& tensor) { return handleInputTensor(tensor); }
};

struct ProtocolOutputTensorHandler
{
    virtual ~ProtocolOutputTensorHandler() = default;
    virtual bool acceptOutputTensor(uint64_t tensorId) { return true; }
    virtual bool handleOutputTensor(const ProtocolTensor& tensor) { return true; }

    bool acceptTensor(uint64_t tensorId) { return acceptOutputTensor(tensorId); }
    bool handleTensor(const ProtocolTensor& tensor) { return handleOutputTensor(tensor); }
};

struct ProtocolGraph
{
    virtual ~ProtocolGraph() = default;

    virtual unsigned getVersion() const { return 1; }

    virtual gcapi::DeviceId_t getDeviceId() const = 0; // eradiano: shall be removed once moved to tpc_lib_api
    /*
     * Get the number of available TPC engines in current graph configuration
     */
    virtual unsigned getMaxAvailableTpc() const = 0;

    virtual unsigned getEagerMode() const = 0; // indication whether we are in eager mode,
                                               // when eagerMode=0 means graph mode, eagerMode=1 means eager mode

    /*
     * Iterate over the graph nodes in topological order.
     */
    virtual bool     foreachNode(ProtocolNodeHandler& handler) const = 0;
    virtual unsigned getNumNodes() const                             = 0;

    /*
     * Iterate over the node's input tensors in order.
     */
    virtual bool     foreachInputTensor(uint64_t nodeId, ProtocolInputTensorHandler& handler) const = 0;
    virtual unsigned getNumInputTensors(uint64_t nodeId) const                                      = 0;

    /*
     * Iterate over the node's output tensors in order.
     */
    virtual bool     foreachOutputTensor(uint64_t nodeId, ProtocolOutputTensorHandler& handler) const = 0;
    virtual unsigned getNumOutputTensors(uint64_t nodeId) const                                       = 0;

    /*
     * Overload the `foreachNode` with callback functions instead of the `ProtocolNodeHandler` interface. This is
     * implemented on top of the `ProtocolNodeHandler`, by implementing it such that the virtual functions are
     * delegating the work to the provided callbacks.
     */
    bool foreachNode(
        const std::function<bool(const ProtocolNode&)>& handle,
        const std::function<bool(uint64_t)>&            accept = [](uint64_t) { return true; }) const
    {
        struct Impl : public ProtocolNodeHandler
        {
            const std::function<bool(const ProtocolNode&)>& handle;
            const std::function<bool(uint64_t)>&            accept;

            Impl(const std::function<bool(const ProtocolNode&)>& handle, const std::function<bool(uint64_t)>& accept)
            : handle(handle), accept(accept)
            {
            }
            bool acceptNode(uint64_t nodeId) override { return accept(nodeId); }
            bool handleNode(const ProtocolNode& node) override { return handle(node); }
        } impl(handle, accept);
        return foreachNode(impl);
    }
    template<typename TPred>
    bool foreachNodeId(TPred predicate) const
    {
        return foreachNode([](const ProtocolNode&) { return true; },
                           [&](uint64_t nodeId) {
                               predicate(nodeId);
                               return false;
                           });
    }

    /*
     * Overload the `foreachInputTensor` with callback functions instead of the `ProtocolInputTensorHandler` interface.
     * This is implemented on top of the `ProtocolInputTensorHandler`, by implementing it such that the virtual
     * functions are delegating the work to the provided callbacks.
     */
    bool foreachInputTensor(
        uint64_t                                          nodeId,
        const std::function<bool(const ProtocolTensor&)>& handle,
        const std::function<bool(uint64_t)>&              accept = [](uint64_t) { return true; }) const
    {
        struct Impl : public ProtocolInputTensorHandler
        {
            const std::function<bool(const ProtocolTensor&)>& handle;
            const std::function<bool(uint64_t)>&              accept;

            Impl(const std::function<bool(const ProtocolTensor&)>& handle, const std::function<bool(uint64_t)>& accept)
            : handle(handle), accept(accept)
            {
            }
            bool acceptInputTensor(uint64_t tensorId) override { return accept(tensorId); }
            bool handleInputTensor(const ProtocolTensor& tensor) override { return handle(tensor); }
        } impl(handle, accept);
        return foreachInputTensor(nodeId, impl);
    }
    template<typename TPred>
    bool foreachInputTensorId(uint64_t nodeId, TPred predicate) const
    {
        return foreachInputTensor(
            nodeId,
            [](const ProtocolTensor&) { return true; },
            [&](uint64_t tensorId) {
                predicate(tensorId);
                return false;
            });
    }

    /*
     * Overload the `foreachOutputTensor` with callback functions instead of the `ProtocolOutputTensorHandler`
     * interface. This is implemented on top of the `ProtocolOutputTensorHandler`, by implementing it such that the
     * virtual functions are delegating the work to the provided callbacks.
     */
    bool foreachOutputTensor(
        uint64_t                                          nodeId,
        const std::function<bool(const ProtocolTensor&)>& handle,
        const std::function<bool(uint64_t)>&              accept = [](uint64_t) { return true; }) const
    {
        struct Impl : public ProtocolOutputTensorHandler
        {
            const std::function<bool(const ProtocolTensor&)>& handle;
            const std::function<bool(uint64_t)>&              accept;

            Impl(const std::function<bool(const ProtocolTensor&)>& handle, const std::function<bool(uint64_t)>& accept)
            : handle(handle), accept(accept)
            {
            }
            bool acceptOutputTensor(uint64_t tensorId) override { return accept(tensorId); }
            bool handleOutputTensor(const ProtocolTensor& tensor) override { return handle(tensor); }
        } impl(handle, accept);
        return foreachOutputTensor(nodeId, impl);
    }
    template<typename TPred>
    bool foreachOutputTensorId(uint64_t nodeId, TPred predicate) const
    {
        return foreachOutputTensor(
            nodeId,
            [](const ProtocolTensor&) { return true; },
            [&](uint64_t tensorId) {
                predicate(tensorId);
                return false;
            });
    }
    virtual tpc_lib_api::DeviceId getDeviceIdentifier() const { return tpc_lib_api::DEVICE_ID_GAUDI; };
};

// Return codes to be used by synapse_mlir library
typedef enum _SynMLIRReturnCode_t
{
  SYN_MLIR_SUCCESS = 0,
  SYN_MLIR_FAILED  = 400,
} SynMLIRReturnCode_t;

} // namespace gc_protocol


/**************************** Entry points names ******************************/
#define COMPLEX_GUID_FUNCTIONAL_ENTRY_POINT_NAME_P  "ExtractFunctionalComplexGUIDP"  // eradiano: shall be removed once moved to tpc_lib_api
#define COMPLEX_GUID_PERFORMANCE_ENTRY_POINT_NAME_P "ExtractPerformanceComplexGUIDP" // eradiano: shall be removed once moved to tpc_lib_api

#define COMPLEX_GUID_FUNCTIONAL_ENTRY_POINT_NAME  "ExtractFunctionalComplexGUID"
#define COMPLEX_GUID_PERFORMANCE_ENTRY_POINT_NAME "ExtractPerformanceComplexGUID"

#define RUN_SYNAPSE_MLIR_OPTIMIZER_ENTRY_POINT_NAME "RunSynapseMLIROptimizer"




/************************* Entry points declarations **************************/

/*
 ***************************************************************************************************
 *   @brief Functional extract of a complex guid operation.
 *
 *   @param inputGraph     [in]   An interface to the graph containing the complex guid
 *                                that should be extracted.
 *   @param outputGraph    [out]  An interface to the graph contating the nodes
 *                                that were extracted from the containing guid.
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef gcapi::GlueCodeReturn_t (*pfnExtractFunctionalComplexGUIDP) // eradiano: shall be removed once moved to tpc_lib_api
(
  _IN_  const gc_protocol::ProtocolGraph* inputGraph,
  _OUT_ gc_protocol::ProtocolGraph**      outputGraph
);

typedef tpc_lib_api::GlueCodeReturn (*pfnExtractFunctionalComplexGUID)
(
  _IN_  const gc_protocol::ProtocolGraph* inputGraph,
  _OUT_ gc_protocol::ProtocolGraph**      outputGraph
);

/*
 ***************************************************************************************************
 *   @brief Performance extract of a complex guid operation.
 *
 *   @param inputGraph     [in]   An interface to the graph containing the complex guid
 *                                that should be extracted.
 *   @param outputGraph    [out]  An interface to the graph containing the nodes
 *                                that were extracted from the complex guid.
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef gcapi::GlueCodeReturn_t (*pfnExtractPerformanceComplexGUIDP) // eradiano: shall be removed once moved to tpc_lib_api
(
    _IN_  const gc_protocol::ProtocolGraph* inputGraph,
    _OUT_ gc_protocol::ProtocolGraph**      outputGraph
);

typedef tpc_lib_api::GlueCodeReturn (*pfnExtractPerformanceComplexGUID)
(
    _IN_  const gc_protocol::ProtocolGraph* inputGraph,
    _OUT_ gc_protocol::ProtocolGraph**      outputGraph
);

/*
 ***************************************************************************************************
 *   @brief Run synapse optimizations in MLIR.
 *
 *   @param inputGraph     [in]   An interface to the graph that should be optimized.
 *   @param outputGraph    [out]  An interface to the graph that was possibly optimized
 *
 *   @return               The status of the operation
 ***************************************************************************************************
 */
typedef gc_protocol::SynMLIRReturnCode_t (*pfnRunSynapseMLIROptimizer)
(
    _IN_  const gc_protocol::ProtocolGraph* inputGraph,
    _OUT_ gc_protocol::ProtocolGraph**      outputGraph
);


#endif  // GRAPH_COMPILER_PROTOCOL_HPP
