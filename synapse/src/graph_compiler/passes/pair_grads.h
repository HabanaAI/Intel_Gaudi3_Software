#pragma once

#include "habana_graph.h"

// Upon finding the following pattern:
//                 [wT]
//                  |
//                  v
// [in (3/4D)]->gradA(bgemm)->[dA]
//    |
//    +-->reshape->[flatIn (2D)]
//                     |
//                     v
//        [flatAT]->gradB(gemm)->[dB]
//
// Pair the 2 grads around the flattened input:
//                                  [wT]
// [in (3/4D)]                       |
//    |                              v
//    +-->reshape->[flatIn (2D)]->gradA(gemm)->[flatDA]->reshape->[dA]
//                     |
//                     v
//        [flatAT]->gradB(gemm)->[dB]

class GradAReshapedGradBPairTransformer
{
public:
    GradAReshapedGradBPairTransformer(HabanaGraph& graph) : m_graph(graph) {}

    bool optimizeGradPairs();

private:
    HabanaGraph& m_graph;

    // Preliminary elimination of invalid operands (nulls, shape tensors, ...)
    bool filterTensor(const TensorPtr& tensor) const;

    struct PairingPattern
    {
        NodePtr bgemm;
        NodePtr reshape;
    };

    // Find the handled pattern:
    // [t]->bgemm_gradA->[dA]
    //  +-->reshape->[flat_t]->gemm_gradB->[dB]
    // t is referred to as the seed of the pattern
    std::optional<PairingPattern> patternMatch(const TensorPtr& seed) const;
    // Check if a reshape has the sub-pattern reshape->gemm
    bool matchReshapeSubPattern(const NodePtr& reshape) const;
    // Check if the bgemm can be reshaped to be paired with the gemm
    bool matchBGemmSubPattern(const TensorPtr& seed, const NodePtr& bgemm) const;

    // Creates the paired gemm_gradA and reshape pattern to replace the original bgemm, in order to get:
    // [t]->reshape->[flat_t]->gemm_gradA->[flat_da]->reshape->[dA]
    //                   +---->gemm_gradB->[dB]
    NodeVector createPairedGradA(const PairingPattern& matchPattern) const;
    NodePtr    createGemmFromBGemm(const NodePtr& bgemm, const TensorPtr& flatInputA) const;
    TensorPtr  createFlatOperand(const TensorPtr& orig) const;
    NodePtr    createInferShape(const NodePtr& bgemm, const TensorPtr& shape) const;
    NodePtr    createUnflatten(const TensorPtr& flat, const TensorPtr& shape, const TensorPtr& orig) const;
};