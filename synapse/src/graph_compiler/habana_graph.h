#pragma once

#include "code_generator.h"
#include "graph_editor.h"
#include "habana_pass.h"
#include "kernel_db.h"
#include "graph.h"
#include "graph_annotation.h"
#include "node_roi.h"
#include "habana_global_conf.h"
#include "node_utility.h"
#include "program_data_blob.h"
#include "section_id_generator.h"
#include "infra/global_conf_manager.h"
#include "graph_serializers/serializer.h"
#include "node_cost_model.h"

#include "define_synapse_common.hpp"
#include "timer.h"

#include <cstdint>
#include <memory>
#include <map>
#include <list>
#include <vector>

class Scheduler;
class GraphTraits;
class BundlePlane;
namespace gc_recipe
{
class Recipe;
}

// HabanaGraph has compilation, scheduling and execution capabilities on
// Habana's accelerators and Host CPU (for debugging and development).
//
//                               +-----------------------+
//                               |         Graph         |
//                               +----------+------------+
//                                          ^
//                                          |
//                               +----------+------------+
//                               |      HabanaGraph      |
//                               +----------+------------+
//                                          ^
//                                          |
//              +-------------+-------------+---------------+-------------+---------------+
//              |             |             |               |             |               |
//   +----------+---------+   |   +---------+-----------+   |   +---------+-----------+   |
//   |     DaliGraph      |   |   |      GaudiGraph     |   |   |       SimGraph      |   |
//   +--------------------+   |   +---------------------+   |   +---------------------+   |
//                            |                             |                             |
//                 +----------+---------+            +------+-------------+     +---------+-----------+
//                 |     GrecoGraph     |            |     Gaudi2Graph    |     |     EagerGraph      |
//                 +--------------------+            +--------------------+     +---------------------+
//
//
// (to maintain this chart use http://asciiflow.com, you can import the chart there, edit
// and then paste it back here.)

class Pass;
class PassManager;
class HabanaGraph;

namespace gc::layered_brain
{
class LayeredBrainData;
}

using HabanaGraphPtr = std::unique_ptr<HabanaGraph>;

class HabanaGraph : public Graph
{
public:

    struct TensorPtrMapping
    {
        TensorPtrMapping(Tensor* orig, const TensorPtr& dup) : origTensor(orig), newTensor(dup) {}
        TensorPtrMapping(Tensor*&& orig, TensorPtr&& dup) : origTensor(orig), newTensor(dup) {}
        TensorPtrMapping() = default;

        Tensor* origTensor;
        TensorPtr newTensor;
    };

    struct NodeIdMapping
    {
        NodeIdMapping(const synNodeId& orig, const synNodeId& dup) : origHandle(orig), newHandle(dup) {}
        NodeIdMapping(synNodeId&& orig, synNodeId&& dup) : origHandle(orig), newHandle(dup) {}
        NodeIdMapping() = default;

        synNodeId origHandle;
        synNodeId newHandle;
    };

    // TensorPtrMappingVec is used mostly for Eager graph use case and Eager graphs mostly consist of
    // a single user node, so MAX_TENSOR_NR should suffice for most cases, avoiding memory allocations.
    using TensorPtrMappingVec = llvm_vecsmall::SmallVector<HabanaGraph::TensorPtrMapping, MAX_TENSOR_NR>;
    // a local storage of 4 elements should be sufficient to avoid allocations for all Eager use cases
    using NodeIdMappingVec = llvm_vecsmall::SmallVector<NodeIdMapping, 4>;


    HabanaGraph(bool PassManagerNeeded = true);
    virtual ~HabanaGraph();
    virtual void clear() override;

    virtual bool compile();
    virtual bool execute();

    virtual HabanaGraphPtr clone(bool cloneAllocators = false, bool keepMappings = false) const = 0;
    virtual HabanaGraphPtr duplicate(TensorPtrMappingVec& tensorsMap, NodeIdMappingVec& nodesMap);

    virtual std::optional<uint32_t> getNextTPCKernelUniqueId();

    virtual void addSetupNode(pNode node);
    virtual void attachNodes(pNode from, pNode to, unsigned outputIndex, unsigned inputIndex) override;

    virtual NodeList      getRootNodes() const override;
    bool                  isInputTensor(const pTensor& t) const override;
    bool                  isOutputTensor(const pTensor& t) const override;
    std::list<pTensor>    getGraphInputs() const;

    std::list<pTensor>    getGraphIntermediates()           const;

    const std::unique_ptr<CodeGenerator>& getCodeGenerator()const {return m_codeGenerator;};
    std::unique_ptr<CodeGenerator>& getCodeGenerator() {return m_codeGenerator;};
    const NodeUtility& getNodeUtility()const {return m_nodeUtility;}
    NodeUtility& getNodeUtility() {return m_nodeUtility;}
    const NodeSet& getSetupNodes() const { return m_setupNodes;}

    void incNumOfIntermediatesDmaNodes();

    static inline bool runsOnMME(const NodePtr& n)
    {
        HB_ASSERT_PTR(n);
        return n->getNodeDeviceType() == DEVICE_MME;
    }

    static inline bool runsOnTPC(const NodePtr& n)
    {
        HB_ASSERT_PTR(n);
        return n->getNodeDeviceType() == DEVICE_TPC;
    }

    static inline bool runsOnRotator(const NodePtr& n)
    {
        HB_ASSERT_PTR(n);
        return n->getNodeDeviceType() == DEVICE_ROTATOR;
    }

    static inline bool isActivationDMA(const NodePtr& n)
    {
        // Activation DMA nodes move data generated in the graph between SRAM and DRAM
        // So they're characterized by having both an input and an output

        HB_ASSERT_PTR(n);

        if (!n->isDma()) return false;
        return n->isMemset() || (n->getNumInputs() + n->getNumOutputs() > 1);
    }

    static inline bool isNonActivationDMA(const NodePtr& n)
    {
        HB_ASSERT_PTR(n);
        return n->isDma() && !isActivationDMA(n);
    }

    static tpc_lib_api::DeviceId deviceTypeToDeviceID(synDeviceType deviceType);

    virtual bool moveNodesToGraph(HabanaGraph& outputGraph);

    // If this query returns a true value the caller may deduce that this tensor is:
    //     Gaudi: persistent
    //     Goya : input/output of the graph.
    virtual bool     isPersistentTensor(const pTensor& tensor) const;

    // If this query returns a true value the caller may deduce that this tensor is:
    //     Gaudi/Gaudi2: persistent or in user managed dram section
    //     Goya : input/output of the graph.

    virtual bool     isUserManagedDram(const pTensor& tensor) const;

    // based on this query, we would have different visualizations
    //      by default:  Show all Nodes (=true)
    //      in Goya:   Hide DMA nodes with either no inputs or outputs (for backward compatability)
    virtual bool     shouldVisualizeDmaNodes() const { return true; };

    virtual bool graphSupports64BitDataTypes() const { return false; }

    bool                                  nodeHasROIs(const NodePtr& n) const;
    std::list<NodeROI>*                   GetNodeROIs(const NodePtr& n) const;
    virtual std::pair<unsigned, unsigned> getBreakpointsAndNodeROINr(const NodePtr& n) const;
    virtual NodeVector                    getSortedMMENodes();
    const GraphAnnotation&    getGraphAnnotation() const;
    GraphAnnotation&          getGraphAnnotation();
    NodeSet&                  getSetupNodes();
    const std::string&        getRecipeName() const;
    void                      setRecipeName(std::string_view recipeName);
    uint16_t                  getRecipeDebugId() const;
    void                      setRecipeDebugId(uint16_t recipeDebugId);

    const NodeVector&         getExeSortedNodes() const;
    uint32_t                  getMaxExecutionOrderedIndex() const { return this->getExeSortedNodes().size() - 1; };
    uint32_t                  getMinExecutionOrderedIndex() const { return 0; };
    void                      invalidateExecutionSchedule() const;
    bool                      areNeighborsIgnoreLogicals(const pNode& producer, const pNode& consumer) const;
    uint32_t                  getNumTensorConsumersIgnoreLogicals(const pTensor& tensor) const;
    void                      PrintNodesAndOperands() const;
    bool                      doesNodeHaveDmaUpConsumer(pNode node) const;

    bool getBreakpointEnable() const { return (GCFG_ENABLE_BREAKPOINT_MODE.value() || getGraphBreakpointMode()); }
    bool disableBreakpointsForNonSignaling() const { return GCFG_DISABLE_NON_SIGNALING_ROI_BREAKPOINT.value(); }

    // Compilation attribute accessors
    virtual unsigned    getNumTpcEng() const;
    bool        isEngineDisabled(HabanaDeviceType deviceType, unsigned engineId) const;
    uint64_t        getAvailableEnginesMask(HabanaDeviceType deviceType) const;
    virtual bool        getMemoryOrientedCompilationEnabled() const;
    bool                getVisualizationStatus() const;

    HabanaDeviceType                  getNodeDebugDeviceType(const pNode& node) const;
    pNode                             getNodeSharedPtr(const Node& node)      const;
    const std::shared_ptr<HalReader>& getHALReader()                          const;
    virtual synDeviceType             getDeviceType()                         const = 0;
    tpc_lib_api::DeviceId             getDeviceId() const { return ::deviceTypeToDeviceID(getDeviceType()); }

    virtual unsigned                  getDefaultPipelineDepth()               const;
    virtual unsigned getRotateStripeWidth(std::shared_ptr<RotateNode> & rotateNode) const;

    virtual CompilationMode getCompilationMode() const { return CompilationMode::Graph; }
    virtual CompilationMode calcTypeForCompilation() const { return getCompilationMode(); }

    /**
     * Get pipeline depth dynamically by node producers
     */
    unsigned                          getPipelineDepth(const pNode& node)     const;
    unsigned                          getPipelineDepth(const Node& node)      const;
    void                              setDefaultPipelineDepth(unsigned depth);
    bool                              rerunPass(PassId id);
    bool                              turnOnPredicate(PredicateId id); // trigger passManager's multiple execution
    virtual bool                      validateMemorySection(const InternalSectionHandle* section) const;

    bool                              pinningBufferSizeIsSet()                 const;
    uint32_t                          getPinningBufferSize()                   const;
    bool                              tensorsPinningDisabled()                 const;
    bool                              prefetchingBufferSizeIsSet()             const;
    uint32_t                          getPrefetchingBufferSize()               const;
    bool                              allocateAllInDramEnabled()               const;

    //Expected to be used only in testing
    void replaceCompilationPass(pPass newPass);

    virtual const GraphTraits&  getTraits() const;

    gc::Timer m_timer;

    // add dependency between two sets of nodes in a graph
    virtual void    addControlDependency(const NodeSet& blockingSet, const NodeSet& blockedSet, Tensor::ControlEdgeType controlType = Tensor::ControlEdgeType::MEM);
    virtual void    addControlDependency(const NodePtr& blockingNode,const NodePtr& blockedNode, Tensor::ControlEdgeType controlType = Tensor::ControlEdgeType::MEM);
    virtual void    removeNodeControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType = Tensor::ControlEdgeType::MEM);
    virtual void    removeNodeControlDependency(const NodePtr& Node, const TensorPtr& ctrlEdge, Node::eParamUsage usage);
    bool            isControlDependencyBetweenNodes(const NodePtr& blocking, const NodePtr& blocked) const;

    NodeSet getBlockedNodes(const NodePtr& blockingNode) const;
    NodeSet getBlockingNodes(const NodePtr& blockedNode) const;
    NodeSet getBlockingNodes(const NodePtr& blockedNode, Tensor::ControlEdgeType controlType) const;
    bool    isControlDependencyConfigured();

    /* please use store and restore carefully. make sure the restored list is valid one restoring */
    void    storeExecutionSchedule();
    void    restoreExecutionSchedule();
    void    clearStoredExecutionSchedule();

    bool    isDynamicShape() const { return m_dynamicNodeCount > 0; }

    virtual HabanaGraphPtr createEmptyGraph() const = 0;

    std::map<uint64_t, uint8_t> & getSectionIdToSectionTypeMap() { return m_sectionIdToSectionType; }
    const std::map<uint64_t, uint8_t> & getSectionIdToSectionTypeMap() const { return m_sectionIdToSectionType; }

    void setSectionIdSectionType(uint64_t sectionId, uint8_t sectionType) { m_sectionIdToSectionType[sectionId] = sectionType; }

    // Following API is used for Eager use case by synGraphInferShapes to infer max shape.
    // For non eager graphs we'll just return failure.
    virtual bool performMaxShapeInference() { return false; };

    void setInputInferenceLayouts(std::map<TensorPtr, gc::Layout> inputInferenceLayouts);
    std::map<TensorPtr, gc::Layout>& getInputInferenceLayouts() { return m_inputInferenceLayouts; }
    const std::map<TensorPtr, gc::Layout>& getInputInferenceLayouts() const  { return m_inputInferenceLayouts; }

    void        setUserNodeTypePrecision(const std::string& guid, synDataType precision);
    bool        getUserNodeTypePrecision(const std::string& guid, synDataType& precision) const;
    synDataType getNodeTypeMinPrecision(const std::string& guid);
    synDataType getNodeTypeMinPrecision(const NodePtr& node);

    void finishDataTypeSelection() { m_preDataTypeSelection = false; };

    void setLayeredBrainData(std::unique_ptr<gc::layered_brain::LayeredBrainData>&& lbd);
    gc::layered_brain::LayeredBrainData* getLayeredBrainData() const;

    void         constructBPGraph(bool useAnnotations = false, std::function<bool(const NodePtr&)> predicate = nullptr);
    void         discardBPGraph();
    BundlePlane* getBPGraph() const;

    void setCompiled() { m_compiled = true; }

    bool isCompiled() { return m_compiled; }

    void setInferenceMode(bool mode);

    bool getInferenceMode() const { return m_graphTraits->inferenceGraph(); }

    void setQuantizationEnabled(bool enabled) { m_graphTraits->setQuantizationEnabled(enabled); }

    bool getQuantizationEnabled() const { return m_graphTraits->isQuantizationEnabled(); }

    void setBackoffFactor(double boFactor);

    double getBackoffFactor() const { return m_graphTraits->backoffFactor(); }

    virtual std::vector<uint32_t> getRollupsArray(NodePtr mmeNode) const { return std::vector<uint32_t>(); }

    void getSignalOutInfo(unsigned &numSigOutTensors, unsigned &numSigOutEngineTypes);

    void dumpGraphToJson(graph_serializer::GraphState state, const std::string& name = "") const;

    /**
     * If enabled, dump data of all TPC nodes to json files
     * Data contains instance and glue params objects
     */
    void dumpTpcNodesDataToJson();
    void dumpTpcNodesDataToJson(uint32_t idx) const;

    // Take away the generated allocator hanging on an Eager graph compiled based on templates
    virtual RecipeAllocator* consumeEagerCompositeTemplateRecipeAllocator() { return nullptr; }

    unsigned getNumSigOutTensors() const { return m_numSigOutTensors; }

    void         setTensorsAlignment();

    virtual bool hasPreNodesRollover(HabanaDeviceType devType) const
    {
        return m_annotation.devicePreNodesRolloverIds.find(devType) != m_annotation.devicePreNodesRolloverIds.end();
    }

    const std::set<unsigned>& getDevicePreNodesRolloverIds(HabanaDeviceType devType) const
    {
        return m_annotation.devicePreNodesRolloverIds[devType];
    }

    const std::vector<pTensor> & getConstSectionTensors() const { return m_constSectionTensors; }

    const TensorSet & getInitialPersistentTensors() const { return m_initialPersistentTensors; }

    pPass addPass(pPass newPass);

    virtual unsigned getNumNodesPreCompilation();

    void setFP32LimitedDevice();
    bool isFP32LimitedDevice() const { return m_deviceLimitationInfo.fp32Limited; }

    const std::map<unsigned, uint32_t>& getLogicalQueueToMaxExecutionIndex();
    virtual void setLogicalQueueToMaxExecutionIndex() {}

    virtual uint64_t getNextMemorySectionID(SectionIDGenerator::AllocationManagementType allocType)
    {
        return m_codeGenerator->getNextMemorySectionID(allocType);
    }
    bool                         runPartialPasses(PassId stopBefore);
    void                         setPassManager(std::unique_ptr<PassManager>& pm);
    std::unique_ptr<PassManager> clonePassManager() const;

    std::optional<std::pair<NodeCostModel::EngineType, double>> getNodeExpectedDuration(const NodePtr& node) const;

    bool wasCreatedUsingDuplicateAPI() const { return m_duplicatedTarget; }

protected:

    HabanaGraph(const HabanaGraph& other, bool copyAddresses = false, bool keepMappings = false);
    HabanaGraph& operator=(const HabanaGraph& other);

    virtual bool addNode(pNode node) override;
    virtual void removeNode(pNode node, pNode newProducer = nullptr) override;
    virtual void printNodeAdditionalInfo(const pNode& node) const;
    virtual bool validateNode(const NodePtr& node) const;
    virtual bool addValidatedNode(pNode node);
    virtual void replaceSemanticNodes(NodePtr oldNode, NodePtr newNode) override;
    virtual bool preProcessAddedNode(const NodePtr& node) const;
    virtual void postProcessAddedNode(const NodePtr& node);
    virtual void postProcessRemovedNode(const NodePtr& node);
    virtual void triggerNewNodeTensorPredicates(const NodePtr& node);

    virtual bool generateExecutionSchedule() const;  // return true if succeeded
    virtual bool generateExecutionSchedule(Scheduler* scheduler) const;

    bool runPassManager();

    void registerPassGroups();

    void    removeNodeInputControlDependencies(const NodePtr& Node, Tensor::ControlEdgeType controlType = Tensor::ControlEdgeType::MEM);
    void    removeNodeOutputControlDependencies(const NodePtr& Node, Tensor::ControlEdgeType controlType = Tensor::ControlEdgeType::MEM);
    void    resetMultibufferInfo(const TensorPtr& tensor);


    uint32_t          getNumOfIntermediates() { return m_numOfDmaIntermediates; }
    void              printGlobalConfigurations() const;
    void              saveUsedGcfgFile();
    bool              validateGraphTensorsAreAllocated() const;
    bool              validateGraphTensorsAreReset() const;
    bool              isTensorDataTypeValid(const TensorPtr& tensor) const;
    bool              isSupported64BitDataType(synDataType elementType) const;
    bool              validateGCOp(const NodePtr& node) const;


    std::map<uint64_t, uint8_t> m_sectionIdToSectionType;

    void copyNodesAndTensors(const HabanaGraph& other, bool copyAddresses = false, bool keepPersistent = false, bool keepMappings = false);
    void copyNodeROIs(const std::list<NodeROI>& origNodeRois, std::list<NodeROI>& destNodeRois, const TensorMap& tensorsMapping);

    template <typename T> NodesMap cloneNodes(const T& nodes, const TensorMap& clonedTensors);

    bool recalculateSampleSize(Graph::GraphTensorsData& inputTensorsData, Graph::GraphTensorsData& outputTensorsData);

    bool getTensorData(const std::vector<std::pair<pTensor, uint32_t>>&  tensorsInfo,
                       TensorUsage                                       tensorLocation,
                       Graph::GraphTensorsData&                          graphTensorsData,
                       bool&                                             shouldBreakToEnqueue);


    virtual void initNodeTypeMinPrecision() {};

    void collectConstSectionAndPersistentTensors();

    // Cache for execution-order sorted nodes
    class SortedNodes : private NodeVector
    {
    public:
        using NodeVector::append;
        using NodeVector::clear;
        using NodeVector::empty;
        using NodeVector::reserve;
        using NodeVector::size;

        SortedNodes&      operator=(const SortedNodes& other) = default;
        const NodeVector& get() const { return *this; };
        void              push_back(const NodePtr& val);
        template<typename InputIt>
        void append(InputIt startIt, InputIt endIt)
        {
            auto nextIdx = size();
            NodeVector::append(startIt, endIt);
            setExecutionOrderedIndexForNodesStartingFrom(nextIdx);
        }

    private:
        void setExecutionOrderedIndexForNodesStartingFrom(size_t startIdx)
        {
            const auto currentSize = size();
            const auto ptrToBuff   = data();
            for (size_t i = startIdx; i < currentSize; ++i)
            {
                ptrToBuff[i]->setExecutionOrderedIndex(i);
            }
        };
    };

    friend class GraphEditor;

    NodeUtility                                m_nodeUtility;
    std::map<NodePtr, std::list<NodeROI>*>     m_nodeROIs;  // Maps from a node to all its ROIs
    NodeSet                                    m_setupNodes;
    std::string                                m_recipeName;
    uint16_t                                   m_recipeDebugID;
    std::shared_ptr<GraphTraits>               m_graphTraits;
    mutable SortedNodes                        m_cacheExeSortedNodes;
    SortedNodes                                m_storedCacheExeSortedNodes;
    mutable GraphAnnotation                    m_annotation;
    uint32_t                                   m_numOfDmaIntermediates = 0;
    std::unique_ptr<PassManager>               m_passManager;
    bool                                       m_ctrlDepWasConfigured = false;
    bool                                       m_duplicatedTarget = false; // created through a call to synGraphDuplicate
    size_t                                     m_dynamicNodeCount = 0;
    std::map<TensorPtr, gc::Layout>            m_inputInferenceLayouts;
    std::map<std::string, synDataType>         m_userNodeTypePrecision;
    std::map<std::string, synDataType>         m_nodeTypeMinPrecision;
    std::unique_ptr<BundlePlane>               m_bundlePlane;
    bool                                       m_compiled = false;
    bool                                       m_preDataTypeSelection = false;
    unsigned                                   m_numSigOutTensors = 0;
    std::unique_ptr<CodeGenerator>             m_codeGenerator;
    std::vector<pTensor>                       m_constSectionTensors;
    TensorMap                                  m_clonedTensors;
    NodesMap                                   m_clonedNodes;
    TensorSet                                  m_initialPersistentTensors;
    std::map<TensorPtr, uint64_t>              m_tensorToSectionId;
    synDeviceLimitationInfo                    m_deviceLimitationInfo {};

    std::unique_ptr<gc::layered_brain::LayeredBrainData> m_layeredBrainData;
    std::shared_ptr<NodeCostModel>                       m_nodeCostModel;
    std::map<unsigned, uint32_t> m_logicalQueueToMaxExecutionIndex;
};


//--------------------------------------------------------------------------------
// Two general purpose down-caster to get a specific graph out of the base pointer
//--------------------------------------------------------------------------------
template<typename T>
T* downcaster(HabanaGraph* g)
{
    T* ptr = dynamic_cast<T*>(g);
    HB_ASSERT_PTR(ptr);
    return ptr;
}
template<typename T>
const T* downcaster(const HabanaGraph* g)
{
    const T* ptr = dynamic_cast<const T*>(g);
    HB_ASSERT_PTR(ptr);
    return ptr;
}
