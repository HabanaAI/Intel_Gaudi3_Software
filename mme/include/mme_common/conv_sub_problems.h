#ifndef MME__CONV_SUB_PROBLEMS_H
#define MME__CONV_SUB_PROBLEMS_H

#include "llvm/small_vector.h"
#include "include/mme_common/recipe.h"

namespace MmeCommon
{
// Representation of a sub-problems
struct ConvSubProblem
{
    ConvSubProblem(const MmeCommon::MmeLayerParams& curParams, unsigned newKey) : params(curParams), key(newKey) {}
    MmeRecipe recipe;
    MmeCommon::MmeLayerParams params;
    OffsetArray addressOffset;
    unsigned key = -1U;
    bool isMemsetDesc() { return m_memsetDesc; }
    void setMemsetDesc(bool isMemsetDesc) { m_memsetDesc = isMemsetDesc; }

private:
    bool m_memsetDesc = false;
};

// An ordered sub-problems container and a manager for current sub-problem
using SubProblemVec = llvm_vecsmall::SmallVector<ConvSubProblem, 1>;
class ConvSubProblemContainer : private SubProblemVec
{

public:
    // use vector methods in this class.
    using SubProblemVec::emplace_back;
    using SubProblemVec::push_back;
    using SubProblemVec::operator[];
    using SubProblemVec::back;
    using SubProblemVec::begin;
    using SubProblemVec::empty;
    using SubProblemVec::end;
    using SubProblemVec::front;
    using SubProblemVec::size;

    ConvSubProblemContainer(const ChipType chipType, const MmeLayerParams& params) : m_chipType(chipType)
    {
        createConvSubProblem(params);
    }
    ConvSubProblemContainer(const ChipType chipType) : m_chipType(chipType) {}
    static bool isMemsetDesc(const MmeLayerParams& newParams);
    static bool isComputeDesc(const MmeLayerParams& newParams);
    static bool isOutOfBounds(const MmeLayerParams& newParams);
    // Add a new element to the container
    void createConvSubProblem(const MmeLayerParams& params);
    void push(MmeCommon::MmeLayerParams params)
    {
        emplace_back(params, size());
        current = &back();
    }
    // remove last element that had been added to the container
    void pop()
    {
        pop_back();
        current = empty() ? nullptr : &back();
    }
    //  is this the last subProblem
    bool isLast() const { return current == &back(); }
    void reset(const MmeLayerParams& newParams);
    ConvSubProblem* current = nullptr;

private:
    using SubProblemVec::clear;
    static bool shouldAddMemsetDesc(const MmeLayerParams& newParams);
    unsigned calcTotalNumOfSubProblems(const MmeLayerParams& params) const;

    void makeParamsForDedxSubProblem(const MmeLayerParams& originalParams,
                                     unsigned numOfSubProblems,
                                     unsigned subProblemIdx);
    void makeParamsForRecurringMisalignmentSubProblem(const MmeLayerParams& originalParams,
                                                      unsigned numOfSubProblems,
                                                      unsigned subProblemIdx);
    bool extractGcdFromConvParams(std::array<unsigned, MME_MAX_CONV_DIMS - 1>* stride,
                                  std::array<unsigned, MME_MAX_CONV_DIMS - 1>* dilation,
                                  std::array<unsigned, MME_MAX_CONV_DIMS - 1>* commonDivs) const;
    void handleMemsetDescRecipe();
    bool skipRecipeGeneration(const MmeLayerParams& params) const;

    unsigned getTotalDedxNumOfDesc(const MmeLayerParams& params) const;
    const ChipType m_chipType;
};

}  // namespace MmeCommon

#endif //MME__CONV_SUB_PROBLEMS_H
