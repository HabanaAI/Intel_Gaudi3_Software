#pragma once

#include "transpose_nodes_creator.h"

class TransposeViaPhysical : public TransposeNodeStrategy
{
public:
    virtual ~TransposeViaPhysical() = default;
    TransposeStrategyID getStrategyID() const override { return PHYSICAL_TRANSPOSE; }
};

class TransposeViaTransposeShape : public TransposeNodeStrategy
{
public:
    bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view    strategyName() const override;
    TransposeStrategyID getStrategyID() const override;
    uint64_t calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override
    {
        return 0;
    }
};

class TransposeViaReshape : public TransposeNodeStrategy
{
public:
    bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view    strategyName() const override;
    TransposeStrategyID getStrategyID() const override;
    uint64_t calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override
    {
        return 0;
    }
};

class TransposeViaLogical : public TransposeNodeStrategy
{
public:
    bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    std::string_view    strategyName() const override;
    TransposeStrategyID getStrategyID() const override;
    uint64_t calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
};

// this strategy is needed when due to dynamic shape there are unexpected physical nodes (probably tpc reshapes)
// it convert [MAX-min] -> Transpose -> [MAX-min]
// to [MAX-min] -> InferMaxShape -> [MAX] -> Transpose -> [MAX] -> Identity -> [MAX-min]
//                       |                                            ^
//                       V                                            |
//             shapeTensor [MAX-min] ---> TransposeShape ---> shapeTensor [MAX-min]
class TransposeWithStaticShape : public TransposeNodeStrategy
{
public:
    struct AuxilaryNodeData
    {
        NodePtr inferMaxNode;
        NodePtr transposeShapeNode;
        NodePtr identityNode;

        TensorPtr newTransposeInput;
        TensorPtr newTransposeOutput;
    };

    bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;
    AuxilaryNodeData    createAuxilaryNodes(const TransposeNodeParams& transposeNodeParams) const;
    std::string_view    strategyName() const override;
    TransposeStrategyID getStrategyID() const override;
    uint64_t calculateCost(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const override;

private:
    NodePtr
    createInferMaxShapeNode(const TensorPtr& input, const TensorPtr& newInput, const TensorPtr& inputShape) const;
    NodeVector createTransposeNodes(const TransposeNodeParams& transposeNodeParams,
                                    const TensorPtr&           newInput,
                                    const TensorPtr&           newOutput) const;
    NodePtr    createTransposeShapeNode(const TransposePermutationArray& permutation,
                                        const TensorPtr&                 inputShape,
                                        const TensorPtr&                 outputShape) const;
    NodePtr createIdentityNode(const TensorPtr& output, const TensorPtr& newOutput, const TensorPtr& outputShape) const;
    std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr> createNewTensors(const TensorPtr& input,
                                                                            const TensorPtr& output) const;
};