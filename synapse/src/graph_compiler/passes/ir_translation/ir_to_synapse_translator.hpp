#pragma once
#include "ir_to_synapse_translator_defs.hpp"

/*
 * AbstractIRToSynapseTranslator -
 * Abstract base class for translating from protocolGraph to synapse.
 * Inherits from protocol handlers thus providing a single base class to inherit from.
 * Derived classes should implement handlers.
 * Contain protocolGraph representation which implements GraphInterface.
 * Contain storage for created GC objects.
 * Derived classes may add additional members.
 */
class AbstractIRToSynapseTranslator
: public gc_protocol::ProtocolNodeHandler
, public gc_protocol::ProtocolInputTensorHandler
, public gc_protocol::ProtocolOutputTensorHandler
{
public:
    AbstractIRToSynapseTranslator(const gc_protocol::ProtocolGraph& graphProvider) : m_protocolGraph(graphProvider) {};
    virtual ~AbstractIRToSynapseTranslator() = default;

    // Convert IR Node to Synapse node
    virtual bool handleNode(const gc_protocol::ProtocolNode& node) override = 0;

    // Convert IR input IR tensor to input Synapse tensor
    virtual bool handleInputTensor(const gc_protocol::ProtocolTensor& tensor) override
    {
        return handleTensor(tensor, m_inputs);
    }
    // Convert IR output IR tensor to output Synapse tensor
    virtual bool handleOutputTensor(const gc_protocol::ProtocolTensor& tensor) override
    {
        return handleTensor(tensor, m_outputs);
    }

    // start translation flow by iterating protocolGraph Ops
    virtual bool startTranslationToSynapse(HabanaGraph* graph = nullptr) = 0;

    const NodeVector& getExecutionSortedNodes() const { return m_createdNodes; };

    const ir_translation_defs::NewTensorMap& getCreatedTensors() const { return m_createdTensors; };

    const ir_translation_defs::IrNodeToGCBlockingNodesIdMap& getIrNodeIdsToBlockingNodesMap() const
    {
        return m_irIdToGCNodeBlockingNodes;
    }

    const NodePtr& getCreatedNodeFromIrId(uint64_t irNodeId) const
    {
        size_t nodeIdx = m_irIdToGCNodeIdx.at(irNodeId);
        return m_createdNodes[nodeIdx];
    }

    virtual bool acceptNode(uint64_t nodeId) override { return nodeId != gc_protocol::InvalidId; };

    virtual bool acceptInputTensor(uint64_t tensorId) override { return tensorId != gc_protocol::InvalidId; }

    virtual bool acceptOutputTensor(uint64_t tensorId) override { return tensorId != gc_protocol::InvalidId; }

protected:
    // Auxiliary function for converting IR tensor to Synapse tensor
    virtual bool handleTensor(const gc_protocol::ProtocolTensor& tensor, TensorVector& tensors) = 0;

    // Auxiliary function for getting appropriate GC section id for IR section with id given in protocolGraph
    virtual unsigned getGCSectionId(const gc_protocol::ProtocolTensorSection_t& irSection) = 0;
    // Create GC node from irNode
    virtual bool createGCNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode);
    // Call to createGCTensor for creating node tensors.
    // Return false in case of error.
    bool createGCNodeAndTensors(const gc_protocol::ProtocolNode& node,
                                NodePtr&                         createdNode);
    // Create GC tensor from tensor.
    // Store tensor in tensor vector.
    // Return false in case of error.
    bool createGCTensor(const gc_protocol::ProtocolTensor& tensor, TensorVector& tensors);
    virtual void setGCTensorName(const gc_protocol::ProtocolTensor& irTensor, Tensor& gcTensor);

    void updateExpBiasForMmeNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode);

    const gc_protocol::ProtocolGraph&       m_protocolGraph;      // contains the protocol IR objects to be converted
    TensorVector                            m_inputs;             // inputs of the current converted node
    TensorVector                            m_outputs;            // outputs of the current converted node
    PermutationVector                       m_inputPermutations;  // handle current node input permutations
    NodeVector                              m_createdNodes;       // created synapse nodes
    HabanaGraph*                            m_originalGraph = nullptr;

    ir_translation_defs::NewTensorMap m_createdTensors;  // map IR tensor ids to the created synapse tensors
    // Maps sections ids to their types, to validate no id have different types.
    ir_translation_defs::SectionTypeMap m_sectionTypeMapping;
    // Maps IR node id to its gcNode paired with IR Node.
    ir_translation_defs::IrNodeToGCNodeIdxMap         m_irIdToGCNodeIdx;
    ir_translation_defs::IrNodeToGCBlockingNodesIdMap m_irIdToGCNodeBlockingNodes;
};

class SynapseNodeReplacer;

/*
 * IRToSynapseTranslator -
 * Translates nodes and their tensors on demand - only if they were modified by Protocol IR.
 * Replaces the translated nodes and tensors into the original HabanaGraph.
 */
class IRToSynapseTranslatorBase : public AbstractIRToSynapseTranslator
{
public:
    IRToSynapseTranslatorBase(const gc_protocol::ProtocolGraph& graphProvider);

    virtual bool handleNode(const gc_protocol::ProtocolNode& node) override = 0;
    // Iterates on graph tensors to store them before.
    virtual bool startTranslationToSynapse(HabanaGraph* graph = nullptr) override = 0;
    // Iterates only on node's inputs
    virtual bool startNodeTranslationToSynapse(HabanaGraph* graph, const NodePtr& origNode) = 0;
    virtual bool handleOutputTensor(const gc_protocol::ProtocolTensor& tensor) override;
    virtual bool handleInputTensor(const gc_protocol::ProtocolTensor& tensor) override;

protected:
    bool handleTensor(const gc_protocol::ProtocolTensor& tensor, TensorVector& tensors) override;

    unsigned getGCSectionId(const gc_protocol::ProtocolTensorSection_t& irSection) override;

    void storeTensorData(const TensorPtr& tensor);
    // if section is new - validate, map section id from protocolGraph to GC section id and turn on predicate.
    // return false for error.
    bool createProtocolSection(const gc_protocol::ProtocolTensorSection_t& section);

    ir_translation_defs::OrigTensorMap  m_originalTensors;   // original tensors in graph
    ir_translation_defs::OrigSectionSet m_originalSections;  // original sections in graph

    // Map sections ids generated by protocolGraph to sections ids generated by GC
    ir_translation_defs::SectionIdMap      m_sectionIdMapping;
    NodePtr                                m_originalNode = nullptr;
};

class IRToSynapseTranslator : public IRToSynapseTranslatorBase
{
public:
    IRToSynapseTranslator(const gc_protocol::ProtocolGraph& graphProvider)
    : IRToSynapseTranslatorBase(graphProvider) {};

    bool handleNode(const gc_protocol::ProtocolNode& node) override;
    // Iterates on graph tensors to store them before.
    bool startTranslationToSynapse(HabanaGraph* graph = nullptr) override;
    // Iterates only on node's inputs
    bool startNodeTranslationToSynapse(HabanaGraph* graph, const NodePtr& origNode) override;

private:
    SynapseNodeReplacer* m_synapseNodeReplacer = nullptr;
};

/*********************************************************************************************************************
 * IRToSynapseDummyGraphTranslator is currently used only in some sanity tests.
 * It has missing implementation of node replacement logic.
 * It shouldn't be used in production code.
 *********************************************************************************************************************/

/*
 * IRToSynapseDummyGraphTranslator -
 * Translates the entire protocolGraph.
 * Doesn't replace translated nodes in HabanaGraph, but stores them internally and provides a getter for them.
 * Node replacement logic not defined yet, and may be implemented outside this class.
 * Currently, used only in translation sanity tests.
 */
class IRToSynapseDummyGraphTranslator : public AbstractIRToSynapseTranslator
{
public:
    IRToSynapseDummyGraphTranslator(const gc_protocol::ProtocolGraph& graphProvider)
    : AbstractIRToSynapseTranslator(graphProvider) {};

    bool handleNode(const gc_protocol::ProtocolNode& node) override;
    bool startTranslationToSynapse(HabanaGraph* graph = nullptr) override;

private:
    bool handleTensor(const gc_protocol::ProtocolTensor& tensor, TensorVector& tensors) override;

    // naive behaviour
    unsigned getGCSectionId(const gc_protocol::ProtocolTensorSection_t& irSection) override { return irSection.id; };
};
