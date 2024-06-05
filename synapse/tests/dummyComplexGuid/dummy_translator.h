#pragma once

#include <gc_protocol.hpp>
#include <list>
#include <map>
#include <cstring>

using namespace gc_protocol;
// below are auxiliary alias and struct for relating ir tensors to ir nodes
using IRNodePtr       = std::shared_ptr<ProtocolNode>;
using IRTensorPtr     = std::shared_ptr<ProtocolTensor>;
using IRNodeList      = std::list<IRNodePtr>;
using IRTensorsMap    = std::map<uint64_t, IRTensorPtr>;  // mapping tensors ids to IR tensors
using IRTensorsVector = std::vector<IRTensorPtr>;
static const unsigned int c_tensorMaxDim = tpc_lib_api::MAX_TENSOR_DIM;
static const unsigned int c_numOfNStrides = c_tensorMaxDim + 1;

struct NodeTensors
{
    IRTensorsVector inputs;
    IRTensorsVector outputs;
};
using IRNodesIdsToTensorsMap = std::map<uint64_t, NodeTensors>;

struct DefaultGraphWrapper : public ProtocolGraph
{
    DefaultGraphWrapper(IRNodeList& nodes, IRNodesIdsToTensorsMap& nodesIdsAndTensors)
    : m_wrapperIRNodes(nodes), m_wrapperIRNodesAndIRTensors(nodesIdsAndTensors) {};
    // Destructor
    ~DefaultGraphWrapper()
    {
        m_wrapperIRNodes.clear();
        m_wrapperIRNodesAndIRTensors.clear();
    }
    gcapi::DeviceId_t getDeviceId() const override { return m_deviceId; }
    unsigned getMaxAvailableTpc() const override { return 0; }
    unsigned getEagerMode() const override { return 0;}
    unsigned getNumNodes() const override { return m_wrapperIRNodes.size(); };
    bool     foreachNode(ProtocolNodeHandler& handler) const override
    {
        for (IRNodePtr irNode : m_wrapperIRNodes)
        {
            handler.handleNode(*irNode);
        }
        return true;
    }

    bool foreachInputTensor(uint64_t nodeId, ProtocolInputTensorHandler& handler) const override
    {
        NodeTensors nodeTensors = m_wrapperIRNodesAndIRTensors.find(nodeId)->second;
        for (auto tensor : nodeTensors.inputs)
        {
            handler.handleTensor(*tensor);
        }
        return true;
    }

    unsigned getNumInputTensors(uint64_t nodeId) const override { return 0; };

    bool foreachOutputTensor(uint64_t nodeId, ProtocolOutputTensorHandler& handler) const override
    {
        NodeTensors nodeTensors = m_wrapperIRNodesAndIRTensors.find(nodeId)->second;
        for (auto tensor : nodeTensors.outputs)
        {
            handler.handleTensor(*tensor);
        }
        return true;
    }

    unsigned getNumOutputTensors(uint64_t nodeId) const override { return 0; };

    // below data structs provide a graph representation
    IRNodeList             m_wrapperIRNodes;              // execution order sorted IR nodes
    IRNodesIdsToTensorsMap m_wrapperIRNodesAndIRTensors;  // maps between IR nodes to their IR Tensors
    gcapi::DeviceId_t m_deviceId{gcapi::DeviceId_t::DEVICE_ID_GAUDI};
};

/*
 * A Translator for testing needs.
 * Just translates from gc_protocol IR to gc_protocol IR (the target IR is also gc_protocol IR)
 */
struct DefaultGraphTranslator
: public ProtocolNodeHandler
, public ProtocolInputTensorHandler
, public ProtocolOutputTensorHandler
{
    DefaultGraphTranslator(const ProtocolGraph& graphProvider) : m_graphWrapper(graphProvider) {}

    ~DefaultGraphTranslator()
    {
        for (auto irTensor : m_translatorIRTensors)
        {
            if (irTensor.second == nullptr) continue;
            // deallocate shape related arrays that were allocated at ConvertInput/OutputTensors
            clearIRTensorShapeArrays(*irTensor.second);
        }
        m_translatorIRNodes.clear();
        m_translatorIRNodesIdsAndIRTensors.clear();
        m_currentInputs.clear();
        m_currentOutputs.clear();
    }

    bool convertToDefault()
    {
        m_graphWrapper.foreachNode(*this);
        return true;
    };

    bool handleNode(const ProtocolNode& node) override
    {
        m_currentOutputs.clear();
        m_currentInputs.clear();

        IRNodePtr nodePtr = std::make_shared<ProtocolNode>(node);
        m_graphWrapper.foreachInputTensor(node.id, *this);
        m_graphWrapper.foreachOutputTensor(node.id, *this);

        NodeTensors nodeTensors {m_currentInputs, m_currentOutputs};
        m_translatorIRNodesIdsAndIRTensors.insert({node.id, nodeTensors});
        m_translatorIRNodes.push_back(nodePtr);

        return true;
    }
    bool handleInputTensor(const ProtocolTensor& tensor) override
    {
        if (auto tensorItr = m_translatorIRTensors.find(tensor.id); tensorItr != m_translatorIRTensors.end())
        {
            m_currentInputs.push_back(tensorItr->second);
        }
        else
        {
            IRTensorPtr irTensor = createIRTensorCopy(tensor);
            m_translatorIRTensors.insert({irTensor->id, irTensor});
            m_currentInputs.push_back(irTensor);
        }
        return true;
    }
    bool handleOutputTensor(const ProtocolTensor& tensor) override
    {
        if (auto tensorItr = m_translatorIRTensors.find(tensor.id); tensorItr != m_translatorIRTensors.end())
        {
            m_currentOutputs.push_back(tensorItr->second);
        }
        else
        {
            IRTensorPtr tensorPtr = createIRTensorCopy(tensor);
            m_translatorIRTensors.insert({tensorPtr->id, tensorPtr});
            m_currentOutputs.push_back(tensorPtr);
        }
        return true;
    }

    IRTensorPtr createIRTensorCopy(const ProtocolTensor& origIRTensor)
    {
        IRTensorPtr copyIRTensor     = std::make_shared<ProtocolTensor>();
        copyIRTensor->id             = origIRTensor.id;
        copyIRTensor->name           = origIRTensor.name;
        copyIRTensor->elementType    = origIRTensor.elementType;
        copyIRTensor->rank           = origIRTensor.rank;
        copyIRTensor->pData          = origIRTensor.pData;

        // allocate and copy shape related arrays
        // can't use the orig tensor array pointers, since they were copied by value and may be dangling at some point
        auto copySizes    = new uint64_t[c_tensorMaxDim];
        auto copyMinSizes = new uint64_t[c_tensorMaxDim];
        auto copyStrides  = new uint64_t[c_numOfNStrides];
        for (unsigned i = 0; i < c_tensorMaxDim; i++)
        {
            copySizes[i]    = origIRTensor.maxSizes[i];
            copyMinSizes[i] = origIRTensor.minSizes[i];
            copyStrides[i]  = origIRTensor.strides[i];
        }
        copyStrides[c_numOfNStrides - 1] = origIRTensor.strides[c_numOfNStrides - 1];

        copyIRTensor->maxSizes = copySizes;
        copyIRTensor->minSizes = copyMinSizes;
        copyIRTensor->strides  = copyStrides;

        if (origIRTensor.attributes != nullptr)
        {
            copyIRTensor->attributes = new(ProtocolTensorAttributes);
            copyIRTensor->attributes->isGraphOutput  = origIRTensor.attributes->isGraphOutput;
            copyIRTensor->attributes->isInitialized  = origIRTensor.attributes->isInitialized;
            copyIRTensor->attributes->isNotNeeded    = origIRTensor.attributes->isNotNeeded;
            copyIRTensor->attributes->type           = origIRTensor.attributes->type;
            copyIRTensor->attributes->tensorDataType = origIRTensor.attributes->tensorDataType;

            if (origIRTensor.attributes->quantizationParams != nullptr)
            {
                copyIRTensor->attributes->quantizationParams            = new(ProtocolTensorQuantizationParams);
                copyIRTensor->attributes->quantizationParams->zeroPoint =
                    origIRTensor.attributes->quantizationParams->zeroPoint;
                copyIRTensor->attributes->quantizationParams->scale     =
                    origIRTensor.attributes->quantizationParams->scale;
            }
        }
        if (origIRTensor.section != nullptr)
        {
            copyIRTensor->section = new(ProtocolTensorSection);
            copyIRTensor->section->type   = origIRTensor.section->type;
            copyIRTensor->section->offset = origIRTensor.section->offset;
            copyIRTensor->section->id     = origIRTensor.section->id;
        }

        return copyIRTensor;
    }

    void clearIRTensorShapeArrays(ProtocolTensor& IRtensor)
    {
        delete[] IRtensor.maxSizes;
        delete[] IRtensor.minSizes;
        delete[] IRtensor.strides;
        if (IRtensor.attributes != nullptr)
        {
            delete[] IRtensor.attributes->quantizationParams;
            delete[] IRtensor.attributes;
        }
        delete[] IRtensor.section;
    }

    const ProtocolGraph& m_graphWrapper;

    IRNodeList             m_translatorIRNodes;                 // execution order sorted IR nodes that were converted
    IRTensorsMap           m_translatorIRTensors;               // already created IR tensors
    IRNodesIdsToTensorsMap m_translatorIRNodesIdsAndIRTensors;  // maps between IR nodes to their IR Tensors
    // temporary vectors to store tensors that are related to the current node being converted
    IRTensorsVector m_currentInputs;
    IRTensorsVector m_currentOutputs;
};