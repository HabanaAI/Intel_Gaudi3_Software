#ifndef _TRANSPOSENODE_H_
#define _TRANSPOSENODE_H_

#include "logical_op_node.h"
#include "mme_node.h"
#include "synapse_common_types.h"
#include "transpose_permutation.h"
#include "sif/shape_inference_metadata.h"
#include "multi_node.h"
#include "node_visitor.h"
#include <cstdint>
#include <string_view>
#include <vector>

class HabanaGraph;

struct TransposeNodeParams;
using TransposeNodeParamsVector = llvm_vecsmall::SmallVector<TransposeNodeParams, eager_mode::defaultMaxNodesPerGraph>;

/**
 * Put a transpose of the input IFM to the output OFM
 */
class TransposeNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef MultiNode BaseClass;

    virtual bool validateNode() const override;
    virtual NodePtr                  clone() const override;
    const TransposePermutationArray& permutation() const { return m_permutation; }
    void                             setPermutation(TransposePermutationArray array);
    virtual bool RunOnCpu() override;
    static std::string getPermutationString(const TransposePermutationArray& permutation);

    static void transposeOnCpu(const TensorPtr& in, const TensorPtr& out, const TransposePermutationArray& permutation);

    virtual NodeList extract() override
    {
        HB_ASSERT(false, "Habana graph required");
        return {};
    }
    virtual NodeList extract(const HabanaGraph&) override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual void printParamsRawData() const override;

    bool isNode64BitCompatible() const override { return true; };

    void setPreferLogicalBeforePhysical(bool preferLogicalBeforePhysical) { m_preferLogicalBeforePhysicalHint = preferLogicalBeforePhysical; }
    bool getPreferLogicalBeforePhysical() const { return m_preferLogicalBeforePhysicalHint; }

    void setPreferTransposeOnlyOnce(bool preferTransposeOnlyOnce)
    {
        m_preferTransposeOnlyOnce = preferTransposeOnlyOnce;
    }
    bool getPreferTransposeOnlyOnce() const { return m_preferTransposeOnlyOnce; }

    virtual std::string getNodeParametersStr() const override;
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    void permuteParams(const PermutationVector& inputPermutations) override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    static bool isPhysicalTranspose(const TensorPtr& input, const TransposePermutationArray& permutation);
    static bool isPhysicalTranspose(const TransposeNodeParams& transpose);
    virtual bool isDataMovementMultiNode() const override { return true; };

protected:
    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    TransposePermutationArray m_permutation;
    SifTransposeMetadata      m_sifMetadata;

    // When this flag is set and TransposeViaDma is chosen it will create the logical transpose
    // before the physical one (dma), otherwise the physical transpose will be first.
    bool m_preferLogicalBeforePhysicalHint = false;

    bool m_preferTransposeOnlyOnce = false;

    TransposeNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  const UserParams    params,
                  unsigned            paramsSize,
                  std::string_view    name);

    TransposeNode(const TensorPtr& input,
                  const TensorPtr& output,
                  const UserParams params,
                  unsigned         paramsSize,
                  std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    NodeList handle64BitTranspose() const;

    std::optional<NodeVector> tryToReshapeTwoDimTranspose(const uint64_t twoDimTransposeCost) const;
};

// cheaper representation of transpose node for cases where we wish to
// postpone actual transpose node creation.
struct TransposeNodeParams
{
    TensorPtr                  input;
    TensorPtr                  output;
    TransposePermutationArray  permutation;
    std::optional<std::string> nodeName                    = std::nullopt;
    bool                       preferLogicalBeforePhysical = false;
    bool                       preferTransposeOnlyOnce     = false;

    bool operator==(const TransposeNodeParams& o) const { return input == o.input && output == o.output; }
    bool isEmpty() const { return !input && !output; }

    static TransposeNodeParams fromNode(const TransposeNode& transposeNode)
    {
        TransposeNodeParams ret = {transposeNode.getInput(0),
                                   transposeNode.getOutput(0),
                                   transposeNode.permutation(),
                                   transposeNode.getNodeName(),
                                   transposeNode.getPreferLogicalBeforePhysical(),
                                   transposeNode.getPreferTransposeOnlyOnce()};
        return ret;
    }
};

/**
 * In case of not changing the FCD, the OFM has the same data as the IFM,
 * simply the strides should be reorganize
 * run as a logical operation
 */
class LogicalTransposeNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNode() const override;

    virtual bool RunOnCpu() override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    void runLogicalOperation() const override;

    virtual bool canSwapAliasDirection() const override { return !m_isUserPermutationTranspose; }

    virtual bool isRedundantNode() const override;

    const TransposePermutationArray& permutation() const { return m_permutation; }

    static bool
    isSupportedPermutation(const Tensor& in, const Tensor& out, const TransposePermutationArray& permutation);

    virtual std::string getNodeParametersStr() const override;

    virtual bool canHandleStridedRealTensor() const override { return true; }

    void setAsUserPermutationTranspose() { m_isUserPermutationTranspose = true; }

    bool isUserPermutationTranspose() const { return m_isUserPermutationTranspose; }

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool          isAliasStrided() const override;
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;
    LogicalTransposeNode(const TensorPtr& IFM,
                         const TensorPtr& OFM,
                         std::string_view name,
                         Node::eNodeType  type = Node::TYPE_LOGICAL_TRANSPOSE);

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;
    TransposePermutationArray                m_permutation;
    SifTransposeMetadata            m_sifMetadata;

private:
    bool m_isUserPermutationTranspose;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

/**
 * Physical transpose node - Gaudi3.
 * runs on mme
 */
class MmeTransposeNode : public MmeNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef MmeNode BaseClass;

    virtual bool validateNode() const override;
    virtual NodePtr clone() const override;

    const TransposePermutationArray& permutation() const;

    virtual bool RunOnCpu() override;

    // Unlike other MME nodes, transpose can handle 32-bit data types
    // So we override the default MME implementation to run the Node implementation
    virtual synDataType getRequiredInputType(uint32_t tensorIdx) const override;
    virtual synDataType getRequiredOutputType(uint32_t tensorIdx) const override;
    virtual NodeROI     generateRoi() const override;

    virtual bool        validateNodeForGraph(const HabanaGraph& g) const override;
    bool                canBeConvertedToGEMM() const override { return false; }
    virtual bool        isOperandTransposed(const TensorPtr& t) const override;
    virtual std::string getNodeParametersStr() const override;
    virtual bool        isTransposeViaGemm() const override { return m_isTransposeViaGemm; }
    void                setTransposeViaGemm(bool val) { m_isTransposeViaGemm = val; }

protected:
    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;

    MmeTransposeNode(const TensorPtr&                 IFM,
                     const TensorPtr&                 OFM,
                     const TransposePermutationArray& permutation,
                     std::string_view                 name);

private:
    const TransposePermutationArray m_permutation;
    SifTransposeMetadata            m_sifMetadata; // a duplicate for SIF
    bool                            m_isTransposeViaGemm = false;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
#endif // _TRANSPOSENODE_H_
