#pragma once

#include "multi_node.h"
#include "graph_editor.h"
#include "node_visitor.h"

class SliceNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef MultiNode BaseClass;

    struct SliceNodeStaticParams
    {
        NSizeArray starts;
        NSizeArray ends;
        NSizeArray steps;
    };

    SliceNode(const TensorVector& inputs,
              const TensorVector& outputs,
              std::string_view    name,
              eNodeType           type        = Node::TYPE_SLICE,
              ShapeFuncID         sifFunction = SHAPE_FUNC_MAX_ID);

    bool validateNode() const override = 0;

    bool RunOnCpu() override;

    static bool isRedundantSlice(const TensorPtr& unsliced, const TensorPtr& sliced, const SliceNodeStaticParams& p);

    bool validateNodeForGraph(const HabanaGraph& g) const override;

    bool isNode64BitCompatible() const override { return true; }

    bool isDataMovementMultiNode() const override;

    virtual TensorVector getDataInputs() const { return {m_inputs[INPUT_TENSOR]}; }

    virtual bool                 hasShapeTensor() const { return true; }
    virtual uint32_t             getFirstShapeIndex() const { return SHAPE_TENSOR; }
    static SliceNodeStaticParams getDefaultSliceParams(const TensorPtr& unsliced);
    void                         printParamsRawData() const override;

    enum SliceInputs
    {
        INPUT_TENSOR = 0,
        SHAPE_TENSOR,
        STEPS_TENSOR,
        H2D_TENSOR = STEPS_TENSOR,
        STARTS_TENSOR,
        MAX_NUM_INPUTS
    };

    void permuteParams(const PermutationVector& inputPermutations) override;

    void setParams(UserParams userParams, unsigned userParamsSize) override;

    const SliceNodeStaticParams& getParams() const { return m_params; }

protected:
    SifNodeParams         getShapeInferenceFunctionUserParams() override;
    size_t                getShapeInferenceFunctionUserParamsSize() const override;
    virtual TensorPtr     getUnslicedTensor() const = 0;
    virtual TensorPtr     getSlicedTensor() const   = 0;
    virtual bool          isFwd() const             = 0;
    virtual NodePtr
    getSliceNode(const TensorVector& inputs, const TensorPtr& output, const SliceNodeStaticParams& params) = 0;
    virtual NodePtr
    getLogicalNode(const TensorPtr& unsliced, const TensorPtr& sliced, const SliceNodeStaticParams& params) const = 0;
    virtual bool canTranspose() const                                                                      = 0;
    virtual bool shouldExtractToLogicalSlice() const;
    bool         isBeneficialToTranspose() const;

    bool                         validateSlice(const TensorPtr& real, const TensorPtr& aliased) const;
    bool                         validateSliceInExtraction(const TensorPtr& real, const TensorPtr& aliased) const;
    static SliceNodeStaticParams getSliceParams(UserParams          userParams,
                                                unsigned            userParamsSize,
                                                const TensorVector& inputs,
                                                const TensorPtr&    sliced,
                                                const TensorPtr&    unsliced,
                                                const std::string&  name);
    static void                  disableFcdExpansion(SliceNode& n) { n.m_enableFcdExpansion = false; }
    bool                         isFcdSliceNode(const TensorPtr& tensor) const;
    NodeList                     extractSliceFcdNodes();
    NodeList                     extractCast64To32BitNodes();
    NodeList                     extractToIdentity();

    bool     isDynamicSlice() const;
    bool     isRedundant() const;
    NodeList extractDynamicSlice();

    NodeList extractNodes();

    bool                  m_enableFcdExpansion = true;
    SliceNodeStaticParams m_params;

private:
    NodePtr addTransposeNode(const TensorPtr& tensor, unsigned dimToReplace, bool TransposeBefore);

    void extractIntoExpandSequence(NodeList& allNodes);

    bool                  findDim(unsigned& dim);
    SliceNodeStaticParams swapSteps(unsigned axis1, unsigned axis2);
    SliceNodeStaticParams expandDims() const;
    SliceNodeStaticParams expandParamsFor64bitOperands() const;

    unsigned                     countDimsWithSlice() const;
    void                         addShiftTransposes(NodeList& allNodes);
    void                         addFcdTransposes(unsigned axisToReplaceWith, NodeList& allNodes);
    uint64_t                     getSliceExpectedCost() const;
    static synSliceParamsV2      userParams2NDimParams(UserParams userParams, unsigned userParamsSize);
    static SliceNodeStaticParams
    getStaticParamsFromDynamicTensors(const TensorVector& inputs, const TensorPtr& sliced, const TensorPtr& unsliced);

    bool isSliceOnDim(unsigned dim, TSize dimSizeInElements) const;
    bool isStepOnDim(unsigned dim) const;

    NodePtr convertH2DToShape();
};
