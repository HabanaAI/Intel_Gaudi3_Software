#ifndef _NODE_BASE_H_
#define _NODE_BASE_H_

#include "access_pattern.h"
#include "graph_traits.h"
#include "node_io_manager.h"
#include "shape_node.h"
#include "smf/shape_func_registry.h"
#include "tensor.h"

#include <atomic>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

enum InputTensorIndex
{
    TENSOR_IFM               = 0,
    TENSOR_DEDY              = 0,
    TENSOR_WEIGHT            = 1,
    TENSOR_X_BWD             = 1,
    TENSOR_SHAPE_BROADCAST   = 1,
    TENSOR_UNIT_MATRIX       = 1,
    TENSOR_BIAS              = 2,
    TENSOR_MD                = 2,
    TENSOR_SHAPE_DEDX        = 2,
    TENSOR_AUX_BGEMM_MASK_A  = 2,
    TENSOR_CIN               = 3,
    TENSOR_AUX_BGEMM_MASK_B  = 3,
    TENSOR_LUT               = 4,
    TENSOR_AUX_CD_SCRATCHPAD = 4,
    TENSOR_ZP_B              = 5,
    TENSOR_AUX_CD_REDUCTION  = 5,
    TENSOR_INPUT_MAX
};

enum OutputTensorIndex
{
    TENSOR_OFM = 0,
    TENSOR_DEDW = 0,
    TENSOR_DEDX = 0,
    TENSOR_SECONDARY_OFM,
    TENSOR_OUTPUT_MAX
};

//A basic implementation of node

// Opaque member type used by the Graph
typedef std::shared_ptr<struct NodeGraphToken> NodeGraphToken_t;

class HabanaGraph;
class NodeVisitor;
class TensorROI;
struct NodeROI;

using PaddingValues = SmallMap<std::map<TensorPtr, uint32_t>, 6>;
class Node : public std::enable_shared_from_this<Node>
{
public:
    enum eParamUsage
    {
        USAGE_INPUT,
        USAGE_OUTPUT,
        UNUSED //More of an error than anything
    };

    // NOTE: Update getNodeTypeStr when modifying this
    enum eNodeType
    {
        TYPE_USER,  // keep first since it's a special case
        TYPE_TENSOR_ADD,
        TYPE_CONVOLUTION,
        TYPE_GEMM,
        TYPE_POOL,
        TYPE_RELU,
        TYPE_FC,
        TYPE_DEDX,
        TYPE_DEDW,
        TYPE_BATCHNORM,
        TYPE_DMA,
        TYPE_INTERNAL_ADDBIAS,
        TYPE_INTERNAL_CONCAT,
        TYPE_INTERNAL_FLATTEN,
        TYPE_INTERNAL_SPLIT,
        TYPE_INTERNAL_EXPAND_DIMS,
        TYPE_INTERNAL_RESHAPE,
        TYPE_INTERNAL_TRANSPOSE,
        TYPE_LOGICAL_TRANSPOSE,
        TYPE_INTERNAL_BROADCAST,
        TYPE_CUSTOM,
        TYPE_DEBUG,
        TYPE_DEBUG2,
        TYPE_INTERNAL_PACKING,
        TYPE_INTERNAL_LOWERING,
        TYPE_BATCH_GEMM,
        TYPE_MASKED_BATCH_GEMM,
        TYPE_SLICE_AXIS,
        TYPE_TENSOR_VIEW,
        TYPE_INTERNAL_REDUCTION,
        TYPE_STRIDED_VIEW,
        TYPE_STRIDED_INSERT,
        TYPE_MULTI_INSERT,
        TYPE_SLICE,
        TYPE_MEMCOPY,
        TYPE_MEMSET,
        TYPE_NMS,
        TYPE_BATCH_GEMM_DEDX,
        TYPE_BATCH_GEMM_DEDW,
        TYPE_GEMM_DEDX,
        TYPE_GEMM_DEDW,
        TYPE_IDENTITY,
        TYPE_MOMENTS,
        TYPE_TF_BATCH_NORM,
        TYPE_TF_FUSED_BATCH_NORM_GRAD,
        TYPE_FCD_BROADCAST,
        TYPE_BROADCAST,
        TYPE_SLICE_GRAD,
        TYPE_SLICE_INSERT,
        TYPE_SLICE_BWD,
        TYPE_WAIT,
        TYPE_LOGICAL_REQUANT,
        TYPE_ROTATE,
        TYPE_PHYSICAL_CONCAT,
        TYPE_EXTRACT_SHAPE,
        TYPE_MERGE_SHAPES,
        TYPE_SPLIT_SHAPE,
        TYPE_FLATTEN_SHAPE,
        TYPE_EXPAND_DIMS_SHAPE,
        TYPE_SQUEEZE_SHAPE,
        TYPE_PHYSICAL_RESHAPE,
        TYPE_TRANSPOSED_SHAPE_NODE,
        TYPE_STATIC_RESHAPE,
        TYPE_TENSOR_VIEW_SHAPE_NODE,
        TYPE_SQUEEZE_NODE,
        TYPE_FROBENIUS_NORM_NODE,
        TYPE_DYNAMIC_SPLIT_NODE,
        TYPE_PHYSICAL_SPLIT,
        TYPE_EINSUM,
        TYPE_PHYSICAL_FLATTEN,
        TYPE_DYNAMIC_RESHAPE_SHAPE,
        TYPE_EINSUM_EXPAND_SHAPE,
        TYPE_DYNAMIC_RANGE,
        TYPE_INFER_SHAPE,
        TYPE_REINTERPRET_CAST,
        TYPE_INFER_MAX_SHAPE,
        TYPE_H2D_OP,
        TYPE_TRANSPOSED_DEDX,
        TYPE_REVERSE,
        TYPE_TILE_SHAPE,
        TYPE_OPERAND_REUSE_INTERNAL,
        TYPE_MAX  // MUST_BE_LAST
    };

    enum eTensorType
    {
        TENSOR_TYPE_DATA,
        TENSOR_TYPE_CONTROL,
        TENSOR_TYPE_ALL
    };

    struct NodeProperties
    {
        LayoutVector inputLayouts;
        LayoutVector outputLayouts;

        NodeProperties() : inputLayouts({}), outputLayouts({}) {}
        NodeProperties(unsigned numInputs, unsigned numOutputs) :
            inputLayouts(LayoutVector(numInputs)), outputLayouts(LayoutVector(numOutputs)) {}
    };

    struct NodeDynamicShapeProjection
    {
        uint32_t tensorIdx;
        uint32_t tensorDim;
        uint32_t indexSpaceDim;
        bool isOutput;
    };

    Node(const TensorVector& inputs,
         const TensorVector& outputs,
         std::string_view    name,
         eNodeType           type                = TYPE_DEBUG,
         ShapeFuncID         sifId               = SHAPE_FUNC_MAX_ID,
         bool                createNodeIoManager = true);
    Node(const Node& other);
    Node& operator=(const Node& other);
    virtual ~Node() {}

    virtual bool operator==(const Node& o) const { return this->equalTo(o); }
    virtual bool operator!=(const Node& o) const { return !(*this == o);    }

    virtual bool equalTo(const Node& o)                                                             const;

    virtual void replaceFirstTensor(const TensorPtr& o, const TensorPtr& n)                              ;
    virtual void replaceTensor(const TensorPtr& o, const TensorPtr& n)                                   ;
    virtual void replaceAllTensors(TensorVector&& inputs, TensorVector&& outputs)                        ;

    void cloneConnectivityFromNode(const Node& o);

    const TensorVector& getInputs()                                                                 const { return m_inputs; }
    const TensorVector& getControlInputs()                                                          const { return m_controlInputs; }
    const TensorPtr& getInput(unsigned i)                                                           const { return i >= m_inputs.size() ? NULL_TENSOR : m_inputs[i]; }
    const TensorPtr& getControlInput(unsigned i)                                                    const { return i >= m_controlInputs.size() ? NULL_TENSOR : m_controlInputs[i]; }
    const TensorPtr& getDataInputUnsafe(unsigned i)                                                 const { return m_inputs[i]; }
    virtual void replaceInput(unsigned index, const TensorPtr& newTensor,
                              eTensorType tensorType = TENSOR_TYPE_DATA)                                 ;
    virtual void        addInput(unsigned         index,
                                 const TensorPtr& newTensor,
                                 eTensorType      tensorType  = TENSOR_TYPE_DATA,
                                 bool             padWithNull = false,
                                 gc::Layout       layout      = gc::Layout());

    void removeDataInputsFromIndex(size_t newEndIdx);

    virtual void  removeInput(const TensorPtr& toRemove, eTensorType tensorType = TENSOR_TYPE_DATA)      ;
    const TensorVector& getOutputs()                                                                const { return m_outputs; }
    const TensorVector& getControlOutputs()                                                         const { return m_controlOutputs; }
    const TensorPtr& getOutput(unsigned i)                                                          const { return i >= m_outputs.size() ? NULL_TENSOR : m_outputs[i]; }
    const TensorPtr& getControlOutput(unsigned i)                                                   const { return i >= m_controlOutputs.size() ? NULL_TENSOR : m_controlOutputs[i]; }
    const TensorPtr& getDataOutputUnsafe(unsigned i)                                                const { return m_outputs[i]; }
    virtual void  addOutput(const TensorPtr& newTensor, eTensorType tensorType = TENSOR_TYPE_DATA)       ;
    virtual void  removeOutput(const TensorPtr& toRemove, eTensorType tensorType = TENSOR_TYPE_DATA)     ;

    virtual void replaceOutput(unsigned index, const TensorPtr& newTensor)                               ;

    virtual void emplaceInput(unsigned index, const TensorPtr& newTensor)                                ;
    virtual void emplaceOutput(unsigned index, const TensorPtr& newTensor)                               ;

    const LayoutVector& getInputLayouts()                                                           const { return m_inputLayouts; }
    const LayoutVector& getOutputLayouts()                                                          const { return m_outputLayouts; }

    virtual void setInputLayouts(const LayoutVector& layouts)                                            ;
    virtual void setOutputLayouts(const LayoutVector& layouts)                                           ;

    const LayoutVector& getInputSupportedLayouts()                                                  const { return m_io->getInputSupportedLayouts(); }
    const LayoutVector& getOutputSupportedLayouts()                                                 const { return m_io->getOutputSupportedLayouts(); }

    virtual TensorVector getOperands()                                                              const;
    virtual TensorVector getInputODSTs()                                                            const;

    virtual unsigned getNumInputs(eTensorType tensorType = TENSOR_TYPE_DATA)                        const;
    virtual unsigned getNumOutputs(eTensorType tensorType = TENSOR_TYPE_DATA)                       const;
    virtual unsigned getNumInputsShapeTensors()                                                     const;
    virtual unsigned getNumOutputsShapeTensors()                                                    const;
    virtual unsigned getNumInputsH2DTensors()                                                       const;
    virtual unsigned getNumOutputsH2DTensors()                                                      const;
    virtual unsigned getNumInputsDataTensors()                                                      const;
    virtual unsigned getNumOutputsDataTensors()                                                     const;
    virtual unsigned getInputIndexOfTensor(const TensorPtr& tensor)                                 const;
    virtual unsigned getOutputIndexOfTensor(const TensorPtr& tensor)                                const;

    virtual eParamUsage getParamUsage(const TensorPtr& param)                                       const;
    virtual TensorSemanticType getParamSemanticType(const TensorPtr& param)                         const;

    virtual unsigned getKDimIndex()                                                                  ;

    eNodeType   getNodeType()                                                                       const { return m_type; }
    virtual std::string getNodeTypeStr()                                                            const;
    virtual std::string_view getEngineTypeStr() const;
    virtual std::string getNodeParametersStr()                                                      const;

    const NodeAnnotation& getNodeAnnotation()                                                       const { return m_annotation; }
    NodeAnnotation& getNodeAnnotation()                                                                   { return m_annotation; }

    NodeIOManager& getNodeIOManager()                                                               const { return *m_io; }

    virtual uint32_t  getPaddingValue(const TensorPtr& t)                                           const;
    virtual void setPaddingValue(const TensorPtr& t, float padVal)                                       ;
    virtual void replaceTensorPadding(const TensorPtr& oldTensor, const TensorPtr& newTensor)            ;

    virtual NodePtr clone()                                                                         const = 0;
    NodePtr cloneWithTensors()                                                                      const;
    virtual NodePtr getSlice()                                                                      const;

    virtual bool isBroadcastableOperation()                                                         const;

    // Logical operation is an operation that doesn't need to be executed on HW,
    // but instead we execute it by changing the tensor's strides.
    //
    // For example: reshape, transpose, broadcast...
    virtual bool isLogicalOperation()                                                               const;
    virtual bool isShapeOperation()                                                                 const;
    bool isWait()                                                                                   const { return m_type == TYPE_WAIT; }
    bool isDebug()                                                                                  const { return (m_type == TYPE_DEBUG) || (m_type == TYPE_DEBUG2); }
    bool isDma()                                                                                    const { return m_type == Node::TYPE_DMA; }
    virtual bool isTranspose()                                                                      const;
    virtual bool isDramSpill()                                                                      const;
    virtual bool isDramFill()                                                                       const;
    bool isRotate()                                                                                 const { return m_type == Node::TYPE_ROTATE; }
    virtual bool isMemset()                                                                         const;
    virtual bool isCast()                                                                           const;
    virtual bool isBatchGemm()                                                                      const;
    virtual bool isSplit() const;
    virtual bool isPartOfRMWSection()                                                               const;
    virtual uint64_t getRMWSectionId()                                                              const;
    virtual bool validateRMWSection(uint64_t maxRMWSectionSize)                                     const;
    virtual bool isDynamicShape()                                                                   const;
    virtual bool isROIDynamic(const NodeROI* roi)                                                   const;
    virtual HabanaDeviceType getNodeDeviceType()                                                    const;

    virtual bool isMultiNode() const { return false; };
    virtual bool canHandleStridedOutput(synDeviceType device = synDeviceTypeInvalid) const { return true; }
    virtual bool canHandleStridedInput(synDeviceType device = synDeviceTypeInvalid) const { return true; }

    virtual bool validateDynamicShapes()                                                            const;

    ShapeNode*                  getShapeNode()                                                            { return &m_shapeNode; }
    sm_function_id_t            getShapeInferenceFunctionId(bool skipStatic = true)                 const;
    virtual uint64_t            getShapeInferenceFunctionVersion()                                  const;
    virtual SifNodeParams       getShapeInferenceFunctionUserParams()                                    ;
    virtual size_t              getShapeInferenceFunctionUserParamsSize()                           const;
    virtual std::vector<NodeDynamicShapeProjection> getDynamicShapeProjectionsTensors()             const;

    virtual void runLogicalOperation()                                                              const;

    // This method performs caching and should not be overridden. Sub classes should override generateNodeAccessPattern
    // instead.
    gc::access_pattern::NodeAccessPatternPtr getNodeAccessPattern()                                 const;

    virtual NodeROI generateRoi()                                                                   const;
    virtual Settable<NodeROI> getInputROI(const NodeROI& roi, uint32_t tensorIdx)                   const;
    virtual Settable<NodeROI> getOutputROI(const NodeROI& roi, uint32_t tensorIdx)                  const;

    void                      setPhysicalRois(std::list<NodeROI>& physicalRois) { m_physicalRois = &physicalRois; }
    const std::list<NodeROI>* getPhysicalRois() const { return m_physicalRois; }

    void                      setLogicalRois(std::list<NodeROI>& logicalRois) { m_logicalRois = &logicalRois; }
    const std::list<NodeROI>* getLogicalRois() const { return m_logicalRois; }

    virtual TensorShape getInputShape(const TensorShape& outputShape,
                                      uint32_t outputIdx,
                                      uint32_t inputIdx)                                            const;

    // infer output min-dims (unless using inferMax)
    bool         inferOutputsShape(synDeviceType deviceType, bool inferMax);
    bool         inferOutputsSizes(synDeviceType deviceType, bool inferMax, bool forbidInvalid = false, bool skipStatic = true);
    virtual void prepareInputTensorForSif(const TensorVector&            inputTensors,
                                          bool                           inferMax,
                                          std::vector<TensorShapeInfo>&  inTensorShapes,
                                          std::vector<TensorShapeInfo*>& inTensorsPointers);

    virtual bool runShapeInferenceFunction(synDeviceType deviceType,
                                           SifParams*    params,
                                           SifOutputs*   outputs,
                                           bool          inferMax,
                                           bool          skipStatic);

    virtual synDataType getRequiredInputType(uint32_t tensorIdx)                                    const;
    virtual synDataType getRequiredOutputType(uint32_t tensorIdx)                                   const;
    virtual bool              validateNode()                                                        const;
    virtual bool              validateNodeLayout()                                                  const;
    virtual bool              validateNodeForGraph(const HabanaGraph& g)                            const = 0;
    virtual bool              validateNode64BitOperands()                                           const;
    virtual bool              is64BitOperands()                                                     const;
    static bool               is64BitOperands(const TensorVector& inputs, const TensorVector& outputs);

    virtual bool              isNode64BitCompatible()                                               const;

    const std::string& getNodeName() const { return m_name; }
    virtual void              print()                                                               const;
    virtual void              PrintOperand(const TensorPtr& t)                                      const;
    // Print node params (given by user) in GRAPH_DATA trace
    virtual void              printParamsRawData()                                                  const;
    virtual bool              RunOnCpu()                                                                 ;
    virtual bool              RunOnCpu(const HabanaGraph& g)                                             ;
    synNodeId                 getId() const { return m_id; }
    void                      setExecutionOrderedIndex(uint32_t executionOrderedIndex)              const { m_executionOrderedIndex = executionOrderedIndex; }
    uint32_t                  getExecutionOrderedIndex()                                            const { return m_executionOrderedIndex; }
    void                      setFullContextId(uint32_t fullContextId)                              const { m_fullContextId = fullContextId; }
    uint32_t                  getFullContextId()                                                    const { return m_fullContextId; }
    void                      setContextId(uint16_t contextId)                                      const { m_contextId = contextId; }
    uint16_t                  getContextId()                                                        const { return m_contextId; }
    void                      setName(const std::string& name)                                            { m_name = name; }
    virtual unsigned          inputDimNameToSize(unsigned inputId,
                                                 char dimensionName)                                const;
    virtual unsigned          outputDimNameToSize(unsigned outputId,
                                                  char dimensionName)                               const;
    virtual unsigned          inputDimNameToIndex(unsigned inputId,
                                                  char dimensionName)                               const;
    virtual unsigned          outputDimNameToIndex(unsigned outputId,
                                                   char dimensionName)                              const;
    virtual void              accept(NodeVisitor* visitor)                                            = 0;
    virtual bool              setGraphTraits(const std::shared_ptr<GraphTraits>& traits)                 ;

    const std::shared_ptr<GraphTraits>&     getGraphTraits()                                        const { return m_graphTraits; }

    virtual std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputs()                 const;
    virtual std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputBinding()           const;
    bool                                                        hasBindingInputReuse()              const;

    uint64_t getReadBytes(TensorLocation location, const std::list<NodeROI>& rois, unsigned clSize) const;
    uint64_t getWriteBytes(TensorLocation location, const std::list<NodeROI>& rois, unsigned clSize) const;

    uint64_t getUsedBytesByROI(TensorLocation location, unsigned clSize, const TensorROIVector& tensorRois) const;

    const StringWithHash& getGUIDAndHash() const { return m_GUID; }
    const std::string&    getGUID() const { return m_GUID.getKey(); }
    virtual void          setGUID(const StringViewWithHash& guidAndHash) { m_GUID = StringWithHash(guidAndHash); }

    void      setParentId(const synNodeId id) { m_parentId = id; }
    synNodeId getParentId() const { return m_parentId; }

    const QuantizerPtr&                       getQuantizer() const { return m_quantizer; }
    void                                      setQuantizer(const QuantizerPtr& quantizer) { m_quantizer = quantizer; }
    synDataType                               getNodePrecision() const { return m_precision; }
    virtual void                              setNodePrecision(synDataType precision);
    virtual void                              setNodePrecisionFromGUID(std::string_view guid);

    void                                      setDeterministic(bool value) { m_deterministic = value; }
    bool                                      getDeterministic() const { return m_deterministic; }

    void                                      setRoundingMode(synRoundingMode value) { m_roundingMode = value; }
    synRoundingMode                           getRoundingMode() const { return m_roundingMode; }
    bool                                      hasHighRankOperand() const;

    virtual void updateCache();

    void            setOriginNodes(const NodesIDs& nodes) { m_originNodes = nodes; }
    const NodesIDs& getOriginNodes() const { return m_originNodes; }
    void            addOriginNodes(const NodesIDs& nodes) { m_originNodes.insert(nodes.begin(), nodes.end()); }

    static synNodeId getUniqueId() { return ++NODE_ID; } // get unique id not related to a specific node

    virtual void setParamsRawData(void* params, size_t size);
    const RawParamsData& getParamsRawData() const { return m_paramsRawData; }
    // make pure virtual (?) after SW-93826 is done
    virtual void                        setParams(UserParams userParams, unsigned userParamsSize);
    virtual std::vector<SifPermutation> getInputPermutations() const;

    virtual bool requiresOutputMaxDimInfer() const;
    virtual void permuteParams(const PermutationVector& inputPermutations);

    virtual bool isH2DManipulationNode() const { return false; }

    // Node utility functions
    static bool isGemmNode(NodePtr n);
    static bool isBatchGemmNode(NodePtr n);
    static bool isDedxNode(NodePtr n);
    static bool isForkNode(const NodePtr& n);
    static bool isJoinNode(const NodePtr& n);

    NodeGraphToken_t m_graphToken;  // Opaque member solely known to, and used by, the Graph

protected:
    void printParamsRawData(void* params, uint32_t size) const;

    virtual gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const { return nullptr; }
    virtual bool hasNodeIOManagerSpecialization() const {return false;}


    TensorVector                   m_inputs;
    TensorVector                   m_outputs;
    TensorVector                   m_controlInputs;
    TensorVector                   m_controlOutputs;
    LayoutVector                   m_inputLayouts;
    LayoutVector                   m_outputLayouts;
    std::unique_ptr<NodeIOManager> m_io;

    eNodeType                      m_type;
    const synNodeId                m_id; //A unique ID for a node in a graph
    synNodeId                      m_parentId;  // A id linked to the original user id in the pre graph
    mutable uint32_t               m_executionOrderedIndex = 0;
    mutable uint32_t               m_fullContextId = 0;
    mutable uint16_t               m_contextId = 0;
    NodeAnnotation                 m_annotation;
    PaddingValues                  m_paddingValues;
    std::string                    m_name;
    std::shared_ptr<GraphTraits>   m_graphTraits;
    ShapeNode                      m_shapeNode;
    std::list<NodeROI>*            m_physicalRois;
    std::list<NodeROI>*            m_logicalRois;

    StringWithHash                 m_GUID;
    uint64_t                       m_shapeInferenceFunctionID;
    QuantizerPtr                   m_quantizer;
    synDataType                    m_precision;
    RawParamsData                  m_paramsRawData;
    bool                           m_deterministic;
    synRoundingMode                m_roundingMode;

    mutable gc::access_pattern::NodeAccessPatternPtr m_nodeAccessPatternCache;

    NodesIDs m_originNodes;

private:
    void     replaceTensor(unsigned index, const TensorPtr& newTensor, TensorVector& dest);
    void     emplaceTensor(unsigned index, const TensorPtr& newTensor, bool isInput);
    unsigned getDimensionNameToSize(unsigned tensorId, char dimensionName, bool isInput) const;
    unsigned getDimensionNameToIndex(unsigned layoutId, char dimensionName, bool isInput) const;

    bool isTensorRoiDynamic(const TensorPtr& tensor, const TensorROI& tensorRoi) const;

    std::unordered_set<uint64_t> getRMWSectionIds(TensorVector& tensors) const;
    TensorVector getRMWTensors() const;
    std::unordered_set<uint64_t> getRMWSectionIds() const;

    static std::atomic<synNodeId> NODE_ID;
    static const TensorPtr NULL_TENSOR;
};

struct NodeComparator
{
    bool operator()(NodePtr n1, NodePtr n2) const
    {
        if (n1 == nullptr) return false;
        if (n2 == nullptr) return true;
        return n1->getId() < n2->getId();
    }
};
typedef std::set<NodePtr, NodeComparator>    NodeSet;
typedef std::set<MMENodePtr, NodeComparator> MMENodeSet;

constexpr bool isInsertReduction(Node::eNodeType t)
{
    return t == Node::TYPE_STRIDED_INSERT || t == Node::TYPE_SLICE_INSERT || t == Node::TYPE_MULTI_INSERT;
}

constexpr bool isReductionOp(Node::eNodeType t)
{
    return t == Node::TYPE_INTERNAL_REDUCTION || isInsertReduction(t);
}

#endif
