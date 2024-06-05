#pragma once

#include "node_visitor.h"
#include "multi_node.h"
#include "shape_inference_metadata.h"
#include "types.h"

// Optimized version for FCD aggregation.
class AggregateFcdNode : public MultiNode
{
public:
    AggregateFcdNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     std::string_view    name,
                     eNodeType           type,
                     ShapeFuncID         sifId);

    virtual void     printParamsRawData() const override;
    bool             isDataMovementMultiNode() const override;
    virtual NodeList extract() override;

protected:
    uint64_t getExpectedCost() const;
    unsigned getNewDimForAggregate(unsigned newFcdDim) const;
    bool     shouldUseLogicalAggregate() const;
};

// Optimized version for FCD split.
// Add high performance transpose sequence before and after the operation
// to avoid low utilization DMA nodes.
class SplitFcdNode : public AggregateFcdNode
{
    DEFINE_VISITOR_METHOD
public:
    SplitFcdNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    virtual NodePtr clone() const override;

    virtual bool validateNode() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override;
    std::vector<uint8_t>  m_sifMetadataBuffer;
};

// Optimized version for FCD concat.
// Add high performance transpose sequence before and after the operation
// to avoid low utilization DMA nodes.
class ConcatFcdNode : public AggregateFcdNode
{
    DEFINE_VISITOR_METHOD
public:
    ConcatFcdNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    virtual NodePtr clone() const override;

    virtual bool validateNode() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

protected:
    SifNodeParams                    getShapeInferenceFunctionUserParams() override;
    size_t                           getShapeInferenceFunctionUserParamsSize() const override;
    Settable<SifConcatenateMetadata> m_metadata;
};