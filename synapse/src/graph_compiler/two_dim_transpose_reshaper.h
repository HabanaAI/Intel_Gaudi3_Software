#pragma once

#include "transpose_permutation.h"
#include "types.h"
#include "transpose_node.h"

class TwoDimTransposeReshaper
{
public:
    TwoDimTransposeReshaper(const TransposeNode& originalTranspose);
    bool                 isValid() const { return m_3DimTranspose.has_value(); }
    const NodeVector&    getWrappingNodes() const;
    const TransposeNode& get3DimTranspose() const;

private:
    enum Dim : unsigned
    {
        FCD = 0,
        SCD = 1
    };
    static inline Dim oppositeDim(const Dim d) { return d == FCD ? SCD : FCD; }

    void tryToReshape2DimTranspose();
    void create3DimTranspose(const Dim dimToSplit, const TSize divisor);
    void createWrappingNodes();

    const TransposeNode&         m_2DimTranspose;
    unsigned                     m_clSizeInElements;
    std::optional<TransposeNode> m_3DimTranspose;
    NodeVector                   m_wrappingNodes;

    static const TransposePermutationArray splitFcdPermutation;
    static const TransposePermutationArray splitScdPermutation;
};