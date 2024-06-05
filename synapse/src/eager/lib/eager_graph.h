#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/node2desc.h"
#include "eager_brain_base.h"
#include "node_info/node_container.h"
#include "program_data_blob_manager.h"
#include "recipe_gen/eager_recipe_generator.h"
#include "utils/general_defs.h"
#include "utils/general_utils.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_graph.h"
#include "graph_compiler/habana_nodes/node.h"
#include "graph_compiler/memory_management/heap_allocator.h"
#include "graph_compiler/memory_management/slab_allocator.h"

// std includes
#include <memory>

namespace eager_mode
{
enum class GraphState : uint8_t
{
    NEW_GRAPH,
    // DUPLICATED state is used for both the original graph and target graph.
    // For original graph it means it already acted as a source for a previous
    // call to duplicate API.
    DUPLICATED,
    SHAPE_INFERENCE_STARTED,
    COMPILATION_STARTED,
    FALLBACK_STARTED
};

class NameBuilder
{
public:
    NameBuilder(std::string_view prefix) : m_prefix(prefix), m_name(prefix) { m_name.insert(m_name.size(), 2, 'a'); }
    std::string_view getNextName()
    {
        // we use hex based counting for cheaper cost and optimize
        // for up to 256 newly created intermidate tensors\nodes.
        if (likely(m_nextId < 256))
        {
            m_name[m_prefix.size()] = 'a' + (m_nextId >> 4);
            m_name.back()           = 'a' + (m_nextId & 0xF);
        }
        else
        {
            m_name = fmt::format("{}{:x}", m_prefix, m_nextId);
        }
        ++m_nextId;
        return m_name;
    }

private:
    uint32_t         m_nextId = 0;
    std::string_view m_prefix;
    std::string      m_name;
};

class EagerGraph final : public HabanaGraph
{
public:
    explicit EagerGraph(synDeviceType deviceType);
    EagerGraph(const EagerGraph& other);
    static bool isValidForEager(synDeviceType deviceType);
    bool        compile() override;

    HabanaGraphPtr  duplicate(TensorPtrMappingVec& tensorsMap, NodeIdMappingVec& nodesMap) override;
    HabanaGraphPtr  clone(bool cloneAllocators = false, bool keepMappings = false) const override;
    bool            addNode(pNode node) override;
    void            removeNode(pNode node, pNode newProducer = nullptr) override;
    synDeviceType   getDeviceType() const override { return chipType2SynDeviceType(m_chipType); }
    ChipType        getChipType() const { return m_chipType; }
    CompilationMode calcTypeForCompilation() const override;
    CompilationMode getCompilationMode() const override { return CompilationMode::Eager; }
    unsigned        getDefaultPipelineDepth() const override { return 1; }

    const EagerMmeBrainBase& getEagerMmeBrain() const;
    HabanaGraphPtr createEmptyGraph() const override { return std::make_unique<EagerGraph>(getDeviceType()); }

    recipe_t*        serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    bool             graphSupports64BitDataTypes() const override { return true; }
    bool             generateExecutionSchedule() const override;
    virtual bool     moveNodesToGraph(HabanaGraph& outputGraph) override;
    RecipeAllocator* consumeEagerCompositeTemplateRecipeAllocator() override;

    ProgramDataBlobManager& getProgramDataBlobManager() { return m_programDataBlobManager; }
    SlabAllocator&          getProgramDataAllocator() { return m_programAllocator; }

    virtual std::optional<uint32_t> getNextTPCKernelUniqueId() override;

    const std::shared_ptr<GraphTraits>& getGraphTraits() const { return m_graphTraits; }
    std::string_view                    getNextTensorName() { return m_tensorNameBuilder.getNextName(); }
    std::string_view                    getNextNodeName() { return m_nodeNameBuilder.getNextName(); }

    QueueDispatcherParams getEagerDMADispatcherParams() const;

    unsigned getNumNodesPreCompilation() override;
    // HabanaGraph API
    const NodeSet&   getNodes() const override;
    const TensorSet& getTensors() const override;
    unsigned         getNumNodes() const override { return m_nodesContainer.getNodes().size(); }
    // add dependency between two sets of nodes in a graph
    virtual void addControlDependency(const NodeSet&          blockingSet,
                                      const NodeSet&          blockedSet,
                                      Tensor::ControlEdgeType controlType) override;
    virtual void addControlDependency(const NodePtr&          blockingNode,
                                      const NodePtr&          blockedNode,
                                      Tensor::ControlEdgeType controlType) override;
    virtual void removeNodeControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType) override;
    virtual void
    removeNodeControlDependency(const NodePtr& Node, const TensorPtr& ctrlEdge, Node::eParamUsage usage) override;
    // Graph API
    virtual NodePtr getNodeByID(synNodeId nodeID) const override;

    virtual uint64_t getNextMemorySectionID(SectionIDGenerator::AllocationManagementType allocType) override
    {
        return m_MemSectionIDGenerator.nextSectionId(allocType);
    }

    bool performMaxShapeInference() override;

    std::pair<unsigned, unsigned> getBreakpointsAndNodeROINr(const NodePtr& n) const override;

private:
    bool compileGraph();
    bool allocateTensors();
    bool loadNOPKernelToProgramDataBlobManager(const KernelInfo& NOPKernelInfo);
    bool transitionGraphState(GraphState newState);

private:
    ChipType               m_chipType;
    GraphState             m_graphState = GraphState::NEW_GRAPH;
    NameBuilder            m_tensorNameBuilder;
    NameBuilder            m_nodeNameBuilder;
    NodesContainer         m_nodesContainer;
    Node2DescContainer     m_node2Desc;
    EagerRecipeGenerator   m_recipeGenerator;
    SlabAllocator          m_programAllocator;
    HeapAllocator          m_workspaceAllocator;
    ProgramDataBlobManager m_programDataBlobManager;
    SectionIDGenerator     m_MemSectionIDGenerator;
    // IDs 1, 2 are reserved for the NOP kernels we instantiate in RecipeTemplates::createAllTemplates()
    uint32_t m_nextTPCKernelUniqueId = 1 + static_cast<uint32_t>(ChipType::CHIPS_NR);

    // Global settings
    bool m_isDebugInfoEnabled;
};

};  // namespace eager_mode
