#pragma once

#include "conv_base_node.h"
#include "node_visitor.h"

enum TpcLoweringType
{
    TPC_LOWERING_NONE,
    TPC_LOWERING_PACK_2_W77_S22_8,
};

class ConvolutionNode : public ConvBaseNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
public:
    typedef ConvBaseNode BaseClass;

    virtual NodePtr clone() const override;

    virtual TensorSemanticType getParamSemanticType(const TensorPtr& param) const override;
    bool RunOnCpu() override;

    virtual bool validateNode() const override;
    virtual bool validateNodeLayout() const override;
    virtual unsigned getKDimIndex() override;

    void            setTPCLowering(TpcLoweringType type);
    TpcLoweringType getTPCLowering() const;
    bool            loweredByTPC() const;

    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;

    virtual std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputs() const override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual bool is3DConvolutionGuid() const override;

    virtual TensorPtr getXOperand() const override;
    virtual TensorPtr getYOperand() const override;
    virtual TensorPtr getWOperand() const override;

    bool         isSpatialSlicingSupported(unsigned dim) const override;
    TSize        getMinSpatialDimOutputROI(unsigned dim) const override;

private:
    ConvolutionNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    template<typename InputType,
            typename WeightType,
            typename OutputType,
            typename StorageFormat = int64_t,
            typename IntermediateClamp = int32_t>
    bool calculateConvolution();

    bool                   m_cin;
    TpcLoweringType        m_tpcLowering;
};
