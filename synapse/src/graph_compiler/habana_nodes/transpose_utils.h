#pragma once

#include "hal_reader/hal_reader.h"
#include "synapse_common_types.hpp"
#include "tensor.h"
#include "transpose_node.h"
#include "transpose_permutation.h"
#include "types.h"
#include "utils.h"


// If all non-one dimensions are keeping their order in the permutation
// For example:
//      1) shape=(3, 1, 1, 2), permutation=(0, 3, 1, 2) will output the shape=(3, 2, 1, 1)
//      2) shape=(3, 1, 1, 1, 2), permutation=(1, 0, 2, 4, 3) will output the shape=(1, 3, 1, 2, 1)
//      3) shape=(1, 1, 3, 2), permutation=(0, 2, 3, 1) will output the shape=(1, 3, 2, 1)
// in the examples above the transpose operation does not change the memory layout
// return true if transposed tensor will not change memory layout, otherwise return false
inline bool isSameDataMemOrder(const Tensor& in, const TransposePermutationArray& permutation)
{
    if (!in.isDenseLayout())
    {
        return false;
    }

    const NSizeArray& inSizes        = in.getAllNSizesInElements();
    const size_t      dims           = in.getDim();
    size_t            sizesIdx       = 0;
    size_t            permutationIdx = 0;

    for (; sizesIdx < dims; ++sizesIdx)
    {
        if (inSizes[sizesIdx] != 1)
        {
            for (; permutationIdx < dims; ++permutationIdx)
            {
                if (sizesIdx == permutation[permutationIdx]) break;
            }
            if (permutationIdx++ == dims) return false;
        }
    }
    return true;
}

inline TransposePermutationArray getChangeFcdPermutation(unsigned newFcd, unsigned dimCount)
{
    TransposePermutationArray changeFcdDimPerm;

    for (unsigned int dim = newFcd; dim < dimCount; ++dim)
    {
        changeFcdDimPerm.push_back((TransposePermutationDim)dim);
    }

    for (unsigned int dim = 0 ; dim < newFcd; ++dim)
    {
        changeFcdDimPerm.push_back((TransposePermutationDim)dim);
    }

    return changeFcdDimPerm;
}
/**
 * Subtract one permutation from another, such that if we combine the rhs with result we'll get lhs.
 */
inline TransposePermutationArray subtractPermutations(const TransposePermutationArray& lhs, const TransposePermutationArray& rhs)
{
    TransposePermutationArray dimDictionary(lhs.size(), TPD_Channel);
    for (unsigned int i = 0; i< lhs.size(); ++i)
    {
        dimDictionary[rhs[i]] = static_cast<TransposePermutationDim>(i);
    }

    TransposePermutationArray different;
    for (auto dim : lhs)
    {
        different.push_back(dimDictionary[dim]);
    }
    return different;
}

inline TransposePermutationArray getIdentityPermutation(uint32_t dims)
{
    TransposePermutationArray ret;
    for (size_t i = 0; i < dims; i++)
    {
        ret.push_back(static_cast<TransposePermutationDim>(i));
    }
    return ret;
}

template<typename T>
inline void applyPermutation(const T* input, const TransposePermutationArray& permutation, T* output)
{
    for (unsigned int dim = 0; dim < permutation.size(); ++dim)
    {
        output[dim] = input[permutation[dim]];
    }
}

inline NSizeArray applyPermutationOnSizes(const NSizeArray&                input,
                                          const TransposePermutationArray& permutation,
                                          bool                             fillRemaining = true)
{
    NSizeArray transposedSizes;
    applyPermutation(input.data(), permutation, transposedSizes.data());
    if (fillRemaining)
    {
        // fill other dims with 1
        std::fill(transposedSizes.begin() + permutation.size(), transposedSizes.end(), 1);
    }
    return transposedSizes;
}


/**
 * Adding two permutations to a single permutation.
 */
inline TransposePermutationArray addPermutations(const TransposePermutationArray& lhs, const TransposePermutationArray& rhs)
{
    HB_ASSERT(lhs.size() == rhs.size(), "Added permutations must be with the same sizes, received {}, {}", lhs.size(), rhs.size());
    TransposePermutationArray ret(rhs.size());
    applyPermutation(lhs.data(), rhs, ret.data());
    return ret;
}

/**
 * Calculate the inverse permutation:
 * For each element -
 * Insert its position at the position equals to the value of the element in the array.
 * Example:
 * Permutation = {1, 3, 0, 2, 4}
 * Inverse permutation = {2, 0, 3, 1, 4}
 */
inline TransposePermutationArray inversePermutation(const TransposePermutationArray& perm)
{
    const size_t n = perm.size();

    TransposePermutationArray res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        res[perm[i]] = static_cast<TransposePermutationDim>(i);
    }
    return res;
}


/**
 * Split a complex permutation to two simple permutations
 * First split permutation is for changing the fast changing dimension position
 * The second permutation is for changing inner dimensions position - use with strides changing
 *
 * Ex. original permutation is CWHB
 *   WCHB -> split to WHBC and CBWH
 *   HBWC -> split to HBCW and CWBH
 */
inline TransposePermutationArrayVec splitPermutation(const TransposePermutationArray& permutation)
{
    HB_ASSERT(permutation[0] != TPD_Channel, "There's no need to split permutation, the FCD stays the same");
    TransposePermutationArrayVec splittedPermutations;

    auto newFcd = permutation.front();
    auto dimCount = permutation.size();
    auto changeFcdDimPerm = getChangeFcdPermutation(newFcd, dimCount);

    splittedPermutations.push_back(changeFcdDimPerm);

    // if changing FCD is enough.
    if (changeFcdDimPerm == permutation)
    {
        return splittedPermutations;
    }
    auto secondPermutation = subtractPermutations(permutation, changeFcdDimPerm);

    splittedPermutations.push_back(secondPermutation);
    return splittedPermutations;
}

inline TransposePermutationArrayVec splitLogicalBeforePhysical(const TransposePermutationArray& permutation)
{
    // splitPermutation returns 2 permutations: the first is for physically changing the fast changing dimension
    // position and the second is for changing inner dimensions position (done with logical transpose). When
    // When we want to change the order of the permutaions such that the logical transpose is created
    // before the physical one, we do it by inversing the initial permutation and then inverse again the 2
    // returned permutations, according to the following rules:
    // 1) inverse(inverse(f)) = f
    // 2) inverse(g * h) = inverse(h) * inverse(g)
    TransposePermutationArrayVec permutations;
    TransposePermutationArrayVec splitPermutationsInversed = splitPermutation(inversePermutation(permutation));
    // Iterate in reverse order
    for (auto it = splitPermutationsInversed.rbegin(); it != splitPermutationsInversed.rend(); ++it)
    {
        permutations.push_back(inversePermutation(*it));
    }
    return permutations;
}

inline void logTransposePermutations(const TransposePermutationArray&    originalPermutation,
                                     const TransposePermutationArrayVec& splittedPermutations)
{
    LOG_DEBUG(GC, "original transpose permutation: {}", TransposeNode::getPermutationString(originalPermutation));
    LOG_DEBUG(GC, "first split permutation: {}", TransposeNode::getPermutationString(splittedPermutations[0]));

    if (splittedPermutations.size() > 1)
    {
        LOG_DEBUG(GC, "second split permutation: {}", TransposeNode::getPermutationString(splittedPermutations[1]));
    }

    LOG_DEBUG(GC, "added transpose node - {}", splittedPermutations.size());
}

inline TensorPtr getTensorAfterTranspose(const Tensor&                    tensor,
                                         const TransposePermutationArray& permutation,
                                         const std::string&               name = "")
{
    return std::make_shared<Tensor>(tensor, permutation, name);
}

inline NSizeArray lowerPhysicalTransposeTo2d(const Tensor&                    input,
                                             const TransposePermutationArray& permutation,
                                             /* output */ unsigned&           axis)
{
    NSizeArray output     = {0};
    unsigned   nodeIfmDim = input.getDim();

    // init all dimensions to unused
    std::fill(output.begin(), output.end(), 1);

    // 1b. calculate newFCD by the product of all sizes of new dimension up to the oldFCD
    NSizeArray sizes = input.getNSizesInElements();
    int index = index_of(permutation, TPD_Channel);
    if (index == -1)
    {
        LOG_ERR(GC, "Cannot flatten PhysicalTranspose. Given permutation is invalid");
    }
    else
    {
        axis = nodeIfmDim - index - 1;
    }
    output[DIM_C] = multiplyElements(sizes.begin(), sizes.begin() + nodeIfmDim - index);

    // 1c. calculate the newSpatial by the product of new dimensions up to the oldFCD
    output[DIM_W] = multiplyElements(sizes.begin() + nodeIfmDim - index, sizes.begin() + nodeIfmDim);

    LOG_DEBUG(GC, "Lowered physical transpose dims are: {},{}", output[DIM_C], output[DIM_W]);
    return output;
}

/**
 * @param perm Permutation array
 * @param size Size of input permutation array
 * @return Transpose Permutation Array object given input permutation
 *         array and it's size
 */
template<typename T>
TransposePermutationArray getTransposePermutationArray(const T* perm, unsigned size)
{
    TransposePermutationArray p;
    p.reserve(size);
    for (auto i = 0; i < size; ++i)
    {
        p.emplace_back(static_cast<TransposePermutationDim>(perm[i]));
    }
    return p;
}

inline synTransposeParamsNDims permutationToParams(const TransposePermutationArray& permutation)
{
    synTransposeParamsNDims retVal;
    HB_ASSERT(ARRAY_SIZE(retVal.permutation) >= permutation.size(), "Incompatible sizes");
    for (unsigned i = 0; i < permutation.size(); i++)
    {
        retVal.permutation[i] = permutation[i];
    }
    retVal.tensorDim = permutation.size();
    return retVal;
}

inline synTransposeParamsNDims permutationToParams(const gc::Permutation& gcPerm)
{
    synTransposeParamsNDims params {};
    params.tensorDim = gcPerm.size();
    for (unsigned i = 0; i < params.tensorDim; i++)
    {
        params.permutation[i] = gcPerm.getValues()[i];
    }
    return params;
}

class DmaTransposeHelper;

/**
 * @brief Get the Size After Utilization Protection object
 * @return std::tuple<uint32_t, uint32_t, uint32_t> Index, powerOf2, remainder
 * @return Index    The last dimension which will be flattened (it's a range, starting from 1)
                    For example, if the dimension is 2, so we will flatten NHWC to NKC, if the index is 1 we will do nothing.
   @return powerOf2     For splits, the value of the divisor of any division.
 * @return remainder    The value which we can split however we want
 */
std::tuple<uint32_t, TSize, TSize> getSizeAfterUtilizationProtection(const TSize* inputSizes, uint32_t dim, const DmaTransposeHelper& helper);

bool isSuitableForTransposeViaGemm(const NodePtr& transposeNode);

synDataType getDataTypeForUnitTensor(const NodePtr& transposeNode);

TensorPtr createSparseUnitTensor(synDataType dataType);

/**
 * @brief Returns true if input permutation is cyclic, otherwise false.
 */
inline bool cyclicPermutation(const TransposePermutationArray& perm)
{
    for (unsigned dim = 0; dim < perm.size(); ++dim)
    {
        if ((perm[dim] + 1) % perm.size() != perm[(dim + 1) % perm.size()]) return false;
    }
    return true;
}

/**
 * @brief Returns true if inpuyt permutation is an identity permutation, otherwise false.
 */
inline bool identityPermutation(const TransposePermutationArray& perm)
{
    for (auto dim = 0; dim < perm.size(); ++dim)
    {
        if (perm.at(dim) != dim) return false;
    }
    return true;
}

/**
 * @brief Return the index of the last (highest) permuted dimension
 */
inline unsigned getLastPermutedDimIdx(const TransposePermutationArray& perm)
{
    unsigned idx = perm.size() - 1;
    while (idx > 1 && perm.at(idx) == idx)
    {
        idx--;
    }
    return idx;
}