#pragma once

#include "habana_graph.h"
#include "layout.h"
#include "transpose_node.h"

class TransposeInserter
{
public:
    TransposeInserter(const NodePtr&           node,
                      const PermutationVector& inputPermutations,
                      const PermutationVector& outputPermutations,
                      bool userTranspose = false);  // transpose requested by the user (such as in GEMM)

    bool InsertTransposesForNodeIO(HabanaGraph& g);

    const TransposeNodeParamsVector& extract(HabanaGraph& g);

private:
    void calcExtract(HabanaGraph& g);

    NodePtr createTransposeNodeFromParams(const TransposeNodeParams& params);

    void createTransposesForNodeIO(HabanaGraph&             g,
                                   const PermutationVector& permutations,
                                   bool                     isOutput);

    void createTranspose(HabanaGraph&           g,
                         const TensorPtr&       origTensor,
                         const gc::Permutation& permutation,
                         bool                   isOutput,
                         const std::string&     name);

    using TensorMappingVec = llvm_vecsmall::SmallVector<std::pair<TensorPtr, TensorPtr>, MAX_TENSOR_NR>;

    const NodePtr&                      m_node;
    const PermutationVector&            m_inputPermutations;
    const PermutationVector&            m_outputPermutations;
    bool                                m_updateAnnotations;
    TensorMappingVec                    m_origTensorToNewTensor;
    TransposeNodeParamsVector           m_transposesToInsert;
};




