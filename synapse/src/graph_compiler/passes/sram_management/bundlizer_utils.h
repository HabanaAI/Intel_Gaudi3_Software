#pragma once

#include "bundle.h"
#include "node.h"
#include "access_pattern.h"

using namespace gc::access_pattern;

// This is an interface for bundle expansion checks. Classes that inherit from this interface, have to implement the
// shouldBreak function.
class BundleExpansionBreakerInterface
{
public:
    virtual ~BundleExpansionBreakerInterface() = 0;
    virtual bool shouldBreak() const           = 0;
};

class BaseBundleExpansionBreaker : public BundleExpansionBreakerInterface
{
public:
    BaseBundleExpansionBreaker(const NodePtr& node, const TensorPtr& connectingTensor, const std::vector<Dim>& dims)
    : m_candidateNode(node),
      m_connectingTensor(connectingTensor),
      m_slicingDims(dims),
      m_accessPattern(m_candidateNode->getNodeAccessPattern()) {};
    BaseBundleExpansionBreaker(const NodePtr& node, const pSlicedOperand& operand, const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand->originalTensor, dims) {};
    virtual ~BaseBundleExpansionBreaker() = default;

protected:
    const NodePtr&              m_candidateNode;
    const TensorPtr&            m_connectingTensor;
    const std::vector<Dim>      m_slicingDims;
    const NodeAccessPatternPtr  m_accessPattern;
};

class GranularityCoverBundleExpansionBreaker : public BaseBundleExpansionBreaker
{
public:
    GranularityCoverBundleExpansionBreaker(const NodePtr&          node,
                                           const TensorPtr&        connectingTensor,
                                           const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, connectingTensor, dims) {};
    GranularityCoverBundleExpansionBreaker(const NodePtr&          node,
                                           const pSlicedOperand&   operand,
                                           const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand, dims) {};
    bool shouldBreak() const override;
    virtual ~GranularityCoverBundleExpansionBreaker() = default;
};

class GranularityMultipleExpansionBreaker : public BaseBundleExpansionBreaker
{
public:
    GranularityMultipleExpansionBreaker(const NodePtr&          node,
                                        const TensorPtr&        connectingTensor,
                                        const std::vector<Dim>& dims) = delete;
    GranularityMultipleExpansionBreaker(const NodePtr&          node,
                                        const pSlicedOperand&   operand,
                                        const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand, dims), m_chunkDimensions(operand->chunkDimensions) {};
    virtual ~GranularityMultipleExpansionBreaker() = default;
    bool shouldBreak() const override;

private:
    const SizeArray m_chunkDimensions;
};

class OverlapExpansionBreaker : public BaseBundleExpansionBreaker
{
public:
    OverlapExpansionBreaker(const NodePtr& node, const TensorPtr& connectingTensor, const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, connectingTensor, dims) {};
    OverlapExpansionBreaker(const NodePtr& node, const pSlicedOperand& operand, const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand, dims) {};
    virtual ~OverlapExpansionBreaker() = default;
    bool shouldBreak() const override;
};

class OffsetExpansionBreaker : public BaseBundleExpansionBreaker
{
public:
    OffsetExpansionBreaker(const NodePtr& node, const TensorPtr& connectingTensor, const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, connectingTensor, dims) {};
    OffsetExpansionBreaker(const NodePtr& node, const pSlicedOperand& operand, const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand, dims) {};
    virtual ~OffsetExpansionBreaker() = default;
    bool shouldBreak() const override;
};

// Break chain when TPC nodes with offset/overlap on the non-stitched operands until it will be supported
// (TODO: SW-99608)
class NonStichedOffsetOverlapExpansionBreaker : public BaseBundleExpansionBreaker
{
public:
    NonStichedOffsetOverlapExpansionBreaker(const NodePtr&          node,
                                            const TensorPtr&        connectingTensor,
                                            const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, connectingTensor, dims) {};
    NonStichedOffsetOverlapExpansionBreaker(const NodePtr&          node,
                                            const pSlicedOperand&   operand,
                                            const std::vector<Dim>& dims)
    : BaseBundleExpansionBreaker(node, operand, dims) {};
    virtual ~NonStichedOffsetOverlapExpansionBreaker() = default;
    bool shouldBreak() const override;
};

using ExpansionBreakerContainer = std::vector<std::unique_ptr<BaseBundleExpansionBreaker>>;

// This is an interface for checking set of condition on bundle expansion. Classes that inherit from this interface,
// have to implement the getExpansionBreakers function.
class BundleChainExpansionCheckerInterface
{
public:
    virtual ~BundleChainExpansionCheckerInterface() = 0;

protected:
    virtual const ExpansionBreakerContainer getExpansionBreakers() const = 0;
};

class BaseBundleChainExpansionChecker : public BundleChainExpansionCheckerInterface
{
public:
    BaseBundleChainExpansionChecker(const NodePtr&          node,
                                    const TensorPtr&        connectingTensor,
                                    const std::vector<Dim>& dims)
    : m_candidateNode(node),
      m_connectingTensor(connectingTensor),
      m_slicingDims(dims),
      m_accessPattern(m_candidateNode->getNodeAccessPattern()) {};
    BaseBundleChainExpansionChecker(const NodePtr& node, const pSlicedOperand& operand, const std::vector<Dim>& dims)
    : BaseBundleChainExpansionChecker(node, operand->originalTensor, dims) {};
    virtual ~BaseBundleChainExpansionChecker() = default;
    bool isChainBreaker() const;

protected:
    const NodePtr&              m_candidateNode;
    const TensorPtr&            m_connectingTensor;
    const std::vector<Dim>      m_slicingDims;
    const NodeAccessPatternPtr  m_accessPattern;
};
