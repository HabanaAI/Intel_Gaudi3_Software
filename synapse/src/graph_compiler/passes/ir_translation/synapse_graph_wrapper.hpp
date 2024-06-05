#pragma once
#include "gc_protocol.hpp"
#include "compilation_hal_reader.h"
#include "types.h"
// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include "gc_interface.h"

class HabanaGraph;

class SynapseWrapperBase : public gc_protocol::ProtocolGraph

{
public:
    SynapseWrapperBase(const tpc_lib_api::DeviceId& deviceID, bool isEagerMode)
    : m_isEagerMode(isEagerMode),
      m_isInference(GraphTraits(deviceIDToDeviceType(deviceID)).inferenceGraph()),
      m_deviceId(deviceID)
    {
    }
    ~SynapseWrapperBase() { m_currentNode = nullptr; }

    gcapi::DeviceId_t getDeviceId() const override { return newGlueCodeToOldDeviceId(m_deviceId); };

    tpc_lib_api::DeviceId getDeviceIdentifier() const override { return m_deviceId; };

    unsigned getMaxAvailableTpc() const override { return TPCNode::getMaxAvailableTpc(m_deviceId); }

    unsigned getEagerMode() const override { return m_isEagerMode; }

    // Iterate on synapse Node's inputs, node is given by node id, activate the InputTensorHandler on each tensor
    bool foreachInputTensor(uint64_t nodeId, gc_protocol::ProtocolInputTensorHandler& handler) const override;

    // Iterate on synapse Node's outputs, node is given by node id, activate the OutputTensorHandler on each tensor
    bool foreachOutputTensor(uint64_t nodeId, gc_protocol::ProtocolOutputTensorHandler& handler) const override;


protected:
    // An auxiliary function that allows differentiation between input/output tensors according to the template typename
    template<typename THandler>
    bool foreachTensor(uint64_t nodeId, THandler& handler, bool isInput) const;
    void createProtocolNodeFromGcNode(const NodePtr& synNode, gc_protocol::ProtocolNode& irNode) const;
    void saveTensorPermutations(const PermutationVector&     inputPermutations,
                                gc_protocol::ProtocolTensor& irTensor,
                                unsigned                     tensorIdx,
                                bool                         isInput) const;

    mutable NodePtr m_currentNode = nullptr;
    const bool      m_isEagerMode;
    const bool      m_isInference;
    const tpc_lib_api::DeviceId m_deviceId;
    // temp variables used to avoid repetitive allocating.
    mutable unsigned                                      m_tmpPerm[Tensor::c_tensorMaxNDim];
    mutable gc_protocol::ProtocolTensor                   m_tmpTensor;
    mutable gc_protocol::ProtocolNode                     m_tmpNode;
    mutable gc_protocol::ProtocolTensorAttributes         m_tmpAttributes;
    mutable gc_protocol::ProtocolTensorSection_t          m_tmpSection;
    mutable gc_protocol::ProtocolTensorQuantizationParams m_tmpProtocolQuantParams;
};

/*
 * SynapseGraphWrapper - contains the synapse graph, implements GraphInterface
 */
class SynapseGraphWrapper : public SynapseWrapperBase
{
public:
    SynapseGraphWrapper(HabanaGraph& graph, bool isEagerMode)
    : SynapseWrapperBase(graph.getDeviceId(), isEagerMode), m_graph(graph)
    {
    }
    // iterate on synapse nodes, and activate the NodeHandler on each node
    bool         foreachNode(gc_protocol::ProtocolNodeHandler& handler) const override;
    unsigned     getNumNodes() const override;
    unsigned     getNumInputTensors(uint64_t nodeId) const override;
    unsigned     getNumOutputTensors(uint64_t nodeId) const override;
    HabanaGraph& getSynapseGraph() { return m_graph; }

private:
    HabanaGraph& m_graph;
};

/*
 * SynapseNodeWrapper - contains the synapse node, implements GraphInterface
 */
class SynapseNodeWrapper : public SynapseWrapperBase
{
public:
    SynapseNodeWrapper(const tpc_lib_api::DeviceId& deviceID, bool isEagerMode)
    : SynapseWrapperBase(deviceID, isEagerMode)
    {
    }
    // activates the NodeHandler on the wrapped node
    bool     foreachNode(gc_protocol::ProtocolNodeHandler& handler) const override;
    unsigned getNumNodes() const override { return 1; }
    void     setNode(const NodePtr& node) { m_currentNode = node; }
    unsigned getNumInputTensors(uint64_t nodeId) const override;
    unsigned getNumOutputTensors(uint64_t nodeId) const override;
};
