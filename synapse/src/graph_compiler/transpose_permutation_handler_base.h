#ifndef _TRANSPOSEPERMUTATIONHANDLERBASE_H_
#define _TRANSPOSEPERMUTATIONHANDLERBASE_H_

#include <list>
#include <memory>
#include "types.h"
#include "transpose_permutation.h"

class Node;
class Tensor;
struct NodeROI;

class TransposePermutationHandlerBase
{
public:
    static std::list<NodeROI> splitRois(NodeROI& roi, std::shared_ptr<Node> n);

    explicit TransposePermutationHandlerBase(const TransposePermutationArray& permutation);

    uint32_t OfmSpatialSize(const Tensor& OFM) const;

    uint32_t OfmFcdSize(const Tensor& OFM) const;

    uint32_t IfmSpatialSize(const Tensor& IFM) const;

    uint32_t IfmSpatialStrides(const Tensor& IFM) const;

    uint32_t IfmFcdSize(const Tensor& IFM) const;

    uint32_t IfmFcdStrides(const Tensor& IFM) const;

    uint32_t OfmSpatialStrides(const Tensor& OFM) const;

    uint32_t OfmFcdStrides(const Tensor& OFM) const;
protected:
    static const int maxConvIterations = 256;

    void calculateHeightAndWidthDims();

    const TransposePermutationArray m_permutation;
    TransposePermutationArray       m_originalPermDictionary;

    std::list<TransposePermutationDim> m_matrixHeightDims;
    std::list<TransposePermutationDim> m_matrixWidthDims;

    static const unsigned int s_fcdPos      = 0;
    static const unsigned int s_dummyDimPos = 1;
    static const unsigned int s_spatialPos  = 2;
};

#endif // _TRANSPOSEPERMUTATIONHANDLERBASE_H_

