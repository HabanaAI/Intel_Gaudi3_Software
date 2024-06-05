#ifndef _HABANA_NODES_H_
#define _HABANA_NODES_H_

#include "access_pattern.h"
#include "concatenate_node.h"
#include "convolution_node.h"
#include "logical_op_node.h"
#include "mme_node.h"
#include "multi_node.h"
#include "node_visitor.h"
#include "node.h"
#include "synapse_common_types.h"
#include "synapse_types.h"
#include "tensor.h"
#include "tpc_node.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

struct SifSplitMetadata;
struct SifDynamicSplitMetadata;

using pSifSplitMetadata        = std::shared_ptr<SifSplitMetadata>;
using pSifDynamicSplitMetadata = std::shared_ptr<SifDynamicSplitMetadata>;

enum DMA_TYPE
{
    DMA_TYPE_INVALID = -1,
    DMA_TYPE_UPSTREAM,
    DMA_TYPE_DOWNSTREAM,
    DMA_TYPE_INTERMEDIATES,
    DMA_TYPE_PREFETCH_STATIC_TENSORS,
    DMA_TYPE_PREFETCH_ACTIVATIONS,
    DMA_TYPE_SPILL,
    DMA_TYPE_INTERNAL
};

enum DMA_OP_TYPE
{
    DMA_OP_COPY,
    DMA_OP_MEMSET,
    DMA_OP_TRANSPOSE,
    DMA_OP_BROADCAST
};

enum DYNAMIC_MEM_OP_TYPE
{
    DMA_OP_NONE = -1,
    DMA_OP_SERIALIZE,
    DMA_OP_DESERIALIZE,
    DMA_OP_DYNAMIC_STRIDE,
    DMA_OP_DYNAMIC_BASE,
    DMA_OP_DYNAMIC_SLICE
};

typedef enum
{
    POOL_MAX,
} PoolOperation;

struct PoolParams
{
    unsigned      W;
    unsigned      H;
    unsigned      dW;
    unsigned      dH;
    PoolOperation op;
};

int64_t scaleCIn(int64_t val, double scaleX, double scaleW, double scaleCIn);

int64_t scaleOutput(int64_t val, double scaleX, double scaleW, double scaleOutput);

class GEMMNode : public MmeNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
public:
    typedef MmeNode BaseClass;

    virtual NodePtr clone() const override;

    const synGEMMParams& getGEMMParams() const { return m_params; }
    void setGEMMParams(const synGEMMParams& params) { m_params = params; }

    unsigned getMMEOperandAIndex() const;
    unsigned getMMEOperandBIndex() const;

    virtual Settable<NodeROI> getInputROI(const NodeROI& roi, uint32_t tensorIdx) const override;

    virtual bool equalTo(const Node& other) const override;
    virtual void print() const override;
    virtual std::string getNodeParametersStr() const override;

    void printParamsRawData() const override;

    bool RunOnCpu() override;

    virtual bool validateNodeLayout() const override;
    virtual TensorSemanticType getParamSemanticType(const TensorPtr& param) const override;

    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;
    virtual unsigned    getKDimIndex() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;
    bool canBeConvertedToGEMM() const override { return true; }

    virtual bool isOperandTransposed(const TensorPtr& t) const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    synGEMMParams m_originalParams; // original parameters as given by the creator of node (=user)
    synGEMMParams m_params; // params matching to the way the MME works

    GEMMNode(const TensorVector& inputs,
             const TensorVector& outputs,
             std::string_view    name,
             Node::eNodeType     type  = Node::TYPE_GEMM,
             ShapeFuncID         sifId = SIF_GEMM);

    void DoGEMM_polymorphic(pTensor A, pTensor B, pTensor bias, pTensor Cin, pTensor C);

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class GEMMDeToDwNode : public GEMMNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef GEMMNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    GEMMDeToDwNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   UserParams          params,
                   std::string_view    name,
                   Node::eNodeType     type = Node::TYPE_GEMM_DEDW);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class GEMMDeToDxNode : public GEMMNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef GEMMNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    GEMMDeToDxNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   UserParams          params,
                   std::string_view    name,
                   Node::eNodeType     type = Node::TYPE_GEMM_DEDX);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class BatchGemmNode : public GEMMNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    static bool      isFullBroadcastLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes);
    static bool      isPartialBroadcastLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes);
    static bool      isSymmetricLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes);
    static bool      allBatchDimsDegenerated(const SizeVector& opSizes);
    typedef GEMMNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNodeLayout() const override;

    bool isBatchGemm() const override;

    bool isFullBroadcastLayout() const;
    bool isPartialBroadcastLayout() const;
    bool isSymmetricLayout() const;
    bool isImplicitFullBroadcast() const;

    bool canBeConvertedToGEMM() const override;

protected:
    BatchGemmNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  UserParams          params,
                  std::string_view    name,
                  Node::eNodeType     type  = TYPE_BATCH_GEMM,
                  ShapeFuncID         sifId = SIF_BATCH_GEMM);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
    bool           isValidOutBatchSizes() const;
};

class BatchGemmDeToDwNode : public BatchGemmNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef BatchGemmNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNodeForGraph(const HabanaGraph& g) const override;
    virtual void    setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    BatchGemmDeToDwNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          params,
                        std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool canBeConvertedToGEMM() const override { return false; }
};

class BatchGemmDeToDxNode : public BatchGemmNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef BatchGemmNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNodeForGraph(const HabanaGraph& g) const override;
    virtual void    setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    BatchGemmDeToDxNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          params,
                        std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class MaskedBatchGemmNode : public BatchGemmNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef BatchGemmNode BaseClass;

    NodePtr clone() const override;
    bool    validateNode() const override;
    bool    validateNodeForGraph(const HabanaGraph& g) const override;
    bool    validateNodeLayout() const override;

    TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;

private:
    MaskedBatchGemmNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          params,
                        std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class MaxPoolNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef Node BaseClass;

    virtual bool validateNode() const override;

    virtual NodePtr clone() const override;

    bool RunOnCpu() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

private:
    PoolParams m_params;

    MaxPoolNode(const TensorVector& inputs,
                const TensorVector& outputs,
                const PoolParams&   params,
                std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              const char*         guid,
                              const std::string&  name);
};

class TPCMemcpyNode : public TPCNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef TPCNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool validateNode() const override;

    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const { return DMA_OP_NONE; }
    virtual bool isDynamicMemoryOp() const { return false; }

    void setGUID(const StringViewWithHash& guidAndHash) override;

    static bool isSupportedNdimDataType(synDataType type);

    bool RunOnCpu() override;

protected:
    TPCMemcpyNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  std::string_view    name,
                  UserParams          params     = nullptr,
                  unsigned            paramsSize = 0);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    virtual bool canHaveAdditionalInputs() const { return false; }
};

class FlattenNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;

    bool RunOnCpu() override;
    void runLogicalOperation() const override;
    unsigned axis() const;
    virtual bool validateNode() const override;
    virtual bool isRedundantNode() const override;
    static bool  isRedundantNode(const Tensor& in, const Tensor& out, uint32_t axis);
    virtual bool canSwapAliasDirection() const override { return true; }

    void printParamsRawData() const override;

    static bool getForceLogicalFlag(unsigned axis);
    static void setForceLogicalFlag(unsigned& axis);
    static void clearForceLogicalFlag(unsigned& axis);

    void         permuteParams(const PermutationVector& inputPermutations) override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool          isAliasStrided() const override { return false; }
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    FlattenNode(const TensorVector& inputs,
                const TensorVector& outputs,
                UserParams          userParams,
                std::string_view    name,
                eNodeType           type = Node::TYPE_INTERNAL_FLATTEN);

private:
    synFlattenParams m_flattenParams;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool m_logicalWasForced = false;
};

class SplitNode : public AggregationNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef AggregationNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    bool RunOnCpu() override;
    unsigned getSplitDim() const { return m_aggDim; }

    static SifNodeParams getShapeInferenceFunctionUserParams(std::vector<uint8_t>& metadataBuffer,
                                                             const size_t          bufferSize,
                                                             const unsigned        aggregationDim,
                                                             const TensorVector&   outputs);
    static size_t        getShapeInferenceFunctionUserParamsSize(const unsigned numOutputs);

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    SplitNode(const TensorVector& inputs,
              const TensorVector& outputs,
              UserParams          userParams,
              std::string_view    name,
              eNodeType           type = Node::TYPE_INTERNAL_SPLIT);

private:
    std::vector<uint8_t> m_sifMetadataBuffer;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    static NodePtr createNodeInternal(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name);

    static NodePtr createSplitNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name,
                                   bool                isInternalNode);

    static bool checkIfPhysicalSplit(const TensorVector& inputs, const TensorVector& outputs, unsigned dim);
};

class ExpandDimsNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool validateNode() const override;
    virtual bool    isNode64BitCompatible() const override;
    bool RunOnCpu() override;
    void runLogicalOperation() const override;
    virtual bool    canSwapAliasDirection() const override { return true; };
    virtual bool    canHandleStridedRealTensor() const override;
    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;
    virtual void    printParamsRawData() const override;

    void permuteParams(const PermutationVector& inputPermutations) override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool          isAliasStrided() const override;
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    ExpandDimsNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   UserParams          userParams,
                   std::string_view    name,
                   eNodeType           type = Node::TYPE_INTERNAL_EXPAND_DIMS);

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    unsigned m_expandDim;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class SliceAxisNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual bool validateNode() const override;
    virtual NodePtr clone() const override;

    bool RunOnCpu() override;
    void runLogicalOperation() const override;
    virtual bool isRedundantNode() const override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    void printParamsRawData() const override;

    void permuteParams(const PermutationVector& inputPermutations) override;

    virtual synDataType getRequiredInputType(uint32_t tensorIdx) const override;
    virtual void        setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool          isAliasStrided() const override { return true; }
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

private:
    synSliceAxisParamsV2 m_params;

    SliceAxisNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  UserParams          params,
                  unsigned            userParamsSize,
                  std::string_view    name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);
};

class ReshapeNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool validateNode() const override;
    virtual bool validateDynamicShapes() const override;
    virtual bool    RunOnCpu() override;
    virtual void    runLogicalOperation() const override;
    virtual void    setMustBeDenseIfNeeded() const override;
    virtual bool canSwapAliasDirection() const override { return true; }
    virtual bool isRedundantNode() const override;
    virtual bool canHandleStridedRealTensor() const override;
    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;
    virtual bool    isNode64BitCompatible() const override;
    virtual bool    isRehsapeOnFcd() const;

    virtual void permuteParams(const PermutationVector& inputPermutations) override;

protected:
    virtual bool isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }

    ReshapeNode(const TensorVector& inputs,
                const TensorVector& outputs,
                std::string_view    name,
                eNodeType           type  = TYPE_INTERNAL_RESHAPE,
                ShapeFuncID         sifId = SIF_RESHAPE);

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    std::pair<bool, NStrideArray> tryGetNewStrides(const TensorPtr& real, const TensorPtr& alias) const;
    void                          correctStrides(const TensorPtr& real, const TensorPtr& alias) const;
    static NodePtr                createNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             UserParams          userParams,
                                             std::string_view    guid,
                                             std::string_view    name);
};

class StaticReshapeNode : public ReshapeNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
public:
    StaticReshapeNode(const TensorVector&       inputs,
                      const TensorVector&       outputs,
                      synStaticReshapeSifParams params,
                      std::string_view          name,
                      eNodeType                 type = TYPE_STATIC_RESHAPE);

    StaticReshapeNode(const StaticReshapeNode& other);
    StaticReshapeNode& operator=(const StaticReshapeNode& other);

    virtual ~StaticReshapeNode() = default;

    virtual bool validateNode() const override;

    virtual bool validateDynamicShapes() const override;
    virtual NodePtr clone() const override;

    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t getShapeInferenceFunctionUserParamsSize() const override;

    static synStaticReshapeSifParams createParamsFromTensors(const TensorPtr& input, const TensorPtr& output);

protected:
    static constexpr unsigned SIF_NUM_DIMS = (unsigned)ARRAY_SIZE(synStaticReshapeSifParams::inputMaxSizes);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    static void detectDynamicDims(const TensorPtr& operand, char* staticDims);
    void        updateSifMaxSizes();

    synStaticReshapeSifParams m_sifParams;
};

class LoweringNode : public ReshapeNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef ReshapeNode BaseClass;

    LoweringNode(const TensorPtr& in, const TensorPtr& out, const std::string& name, unsigned int loweringFactor);
    virtual NodePtr clone() const override;

    virtual Settable<NodeROI> getInputROI(const NodeROI& roi, uint32_t tensorIdx) const override;

    const unsigned int m_loweringFactor;
};

class PackingNode : public StaticReshapeNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef StaticReshapeNode BaseClass;

    PackingNode(const TensorPtr& in,
                const TensorPtr& out,
                const std::string& name,
                unsigned int packingFactor);
    virtual NodePtr clone() const override;

    virtual Settable<NodeROI> getInputROI(const NodeROI& roi, uint32_t tensorIdx) const override;
    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;

    const unsigned int m_packingFactor;
};

// LogicalRequantNode is used to change Qunatization data from input to output tensor
class LogicalRequantNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef LogicalRequantNode BaseClass;

    LogicalRequantNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
    NodePtr        clone() const override;
    bool RunOnCpu() override;
    void runLogicalOperation() const override;
    virtual bool canSwapAliasDirection() const override { return true; }
    virtual bool isRedundantNode() const override;

protected:
    virtual bool isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }
};

class RotateNode : public Node
{
    DEFINE_VISITOR_METHOD
public:
    typedef Node BaseClass;

    RotateNode(const TensorVector& inputs, const TensorVector& outputs, UserParams userParams, std::string_view name);
    static pNode       createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name);
    pNode              clone() const override;
    bool               validateNodeForGraph(const HabanaGraph& g) const override;
    bool               validateNode() const override;
    void               printParamsRawData() const override;

    float       getRotationAngle() const { return m_angle; }
    void        setRotationAngle(  float rotation_angle ) { m_angle = rotation_angle; }

    uint32_t getCoordinateMode() const { return m_coordinate_mode; }
    void     setCoordinateMode(uint32_t coordinate_mode) { m_coordinate_mode = coordinate_mode; }

    uint32_t getRotationMode() const { return m_rotation_mode; }
    void     setRotationMode(uint32_t rotation_mode) { m_rotation_mode = rotation_mode; }

    uint32_t getInterpolationMode() const { return m_interpolation_mode; }
    void     setInterpolationMode(uint32_t interpolation_mode) { m_interpolation_mode = interpolation_mode; }

    uint32_t getInputPixelWidth() const { return m_input_pixel_width; }
    void     setInputPixelWidth(uint32_t input_pixel_width) { m_input_pixel_width = input_pixel_width; }

    uint32_t getOutputPixelWidth() const { return m_output_pixel_width; }
    void     setOutputPixelWidth(uint32_t output_pixel_width) { m_output_pixel_width = output_pixel_width; }

    uint32_t    getParallelLevel() const { return m_parallelLevel; }
    void        setParallelLevel( uint32_t parallelLevel ) {m_parallelLevel = parallelLevel; }
    uint8_t     getBackgroundPixel() const { return m_backgroundPixel; }
    void        setBackgroundPixel( uint8_t backgroundPixel ) { m_backgroundPixel = backgroundPixel; }
    uint32_t    getInputCenterX () const        { return m_inputCenterX ; }
    uint32_t    getInputCenterY () const        { return m_inputCenterY ; }
    uint32_t    getOutputCenterX() const        { return m_outputCenterX; }
    uint32_t    getOutputCenterY() const        { return m_outputCenterY; }
    void        setInputCenterX (uint32_t c)    { m_inputCenterX  = c;    }
    void        setInputCenterY (uint32_t c)    { m_inputCenterY  = c;    }
    void        setOutputCenterX(uint32_t c)    { m_outputCenterX = c;    }
    void        setOutputCenterY(uint32_t c)    { m_outputCenterY = c;    }

    // for debug purposes
    bool        isDumpDescriptors() const { return m_isDumpDescriptors; }
    std::string getDescFilePrefix() const { return m_descFilePrefix; }

    virtual std::string_view getEngineTypeStr() const override;
    virtual void             setParams(UserParams userParams, unsigned userParamsSize) override;
    HabanaDeviceType         getNodeDeviceType() const override { return DEVICE_ROTATOR; }

protected:
    float       m_angle;
    uint32_t    m_inputCenterX;
    uint32_t    m_inputCenterY;
    uint32_t    m_outputCenterX;
    uint32_t    m_outputCenterY;

    uint8_t     m_backgroundPixel;
    uint32_t    m_parallelLevel;

    uint32_t m_coordinate_mode;
    uint32_t m_rotation_mode;
    uint32_t m_interpolation_mode;

    uint8_t m_input_pixel_width;
    uint8_t m_output_pixel_width;

    // For debug
    bool        m_isDumpDescriptors;
    std::string m_descFilePrefix;
};


class WaitNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef Node BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    unsigned getWaitCycles() { return m_waitCycles; }

    void printParamsRawData() const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    unsigned m_waitCycles;

private:
    WaitNode(std::string_view name, UserParams userParams);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class DebugNodeBase : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef Node BaseClass;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool    isNode64BitCompatible() const override;
    virtual NodePtr clone() const override;

protected:
    DebugNodeBase(TensorVector inputs, TensorVector outputs, std::string_view name, Node::eNodeType type = TYPE_DEBUG);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class DebugNode : public DebugNodeBase
{
    DEFINE_VISITOR_METHOD
public:
    typedef Node BaseClass;

    DebugNode(const TensorPtr& opA, const TensorPtr& opB, const std::string& name);
};

class Debug2Node : public DebugNodeBase
{
public:
    typedef DebugNodeBase BaseClass;
    Debug2Node(const TensorPtr& opA, const TensorPtr& opB, const std::string& name);
};

class DebugForkNode : public DebugNodeBase
{
public:
    typedef DebugNodeBase BaseClass;
    DebugForkNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name);
};

class DebugJoinNode : public DebugNodeBase
{
public:
    typedef DebugNodeBase BaseClass;
    DebugJoinNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name);
};

#endif
