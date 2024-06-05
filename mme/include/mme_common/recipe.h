#ifndef MME__RECIPE_H
#define MME__RECIPE_H

#include "llvm/small_vector.h"
#include "include/settable.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/workarounds.h"
#include "include/utils/iterators.h"

namespace MmeCommon
{
////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeIterator
////////////////////////////////////////////////////////////////////////////////////////////////////

enum RecipeSubviewType : unsigned  // Keys of the iterators. Will be used to retrieve current iterations
{
    e_mme_fcd_subview = 0,
    e_mme_non_spatial_subview = 1,  // conv or batch
    e_mme_sp_subview = 2
};

class RecipeIterator
{
    using SingleIt = MultiLoopIterator::Iterator;
    using Iterators = std::vector<SingleIt>;

public:
    RecipeIterator(EMmeOpType opType,
                   bool isRaster,
                   const SingleDimSubViews& fcdSubviews,
                   const SingleDimSubViews& spSubviews,
                   const MultiDimSubViews& nonSpatialSubviews,
                   bool enableReversedOrder = true);

    // Forwarders for range-based loop
    SingleIt begin() { return m_multiIt.begin(); }
    SingleIt end() { return m_multiIt.end(); }

    // Direct access to current iterators values
    unsigned fcdIdx() const { return m_curIterVals.at(e_mme_fcd_subview); }
    unsigned spIdx() const { return m_curIterVals.at(e_mme_sp_subview); }
    unsigned nonSpatialIdx() const { return m_curIterVals.at(e_mme_non_spatial_subview); }
    unsigned getIdx(RecipeSubviewType subviewType) const { return m_curIterVals.at(subviewType); }

    // Other operations
    void setCurIterVals(const MultiLoopIterator::ItersType& curIterVals) { m_curIterVals = curIterVals; }
    bool isLast() const;
    bool isFirst(RecipeSubviewType subviewType) const;
    bool isLast(RecipeSubviewType subviewType) const;
    size_t size() const { return m_splitNr; }

private:
    Iterators createIterVec(EMmeOpType opType, bool isRaster) const;

private:
    MultiLoopIterator::ItersType m_curIterVals;  // Current values of all iterators
    SingleIt m_fcdIt;
    SingleIt m_nonSpatialIt;
    SingleIt m_spIt;
    MultiLoopIterator m_multiIt;
    const size_t m_splitNr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// ReuseInfo
////////////////////////////////////////////////////////////////////////////////////////////////////

// Type of operand reuse
enum EMmeReuseType
{
    e_mme_no_reuse = 0x0,  // No reuse
    e_mme_1d_reuse_a = 0x1,  // Reuse operand A only
    e_mme_1d_reuse_b = 0x2,  // Reuse operand B only
    e_mme_2d_reuse_ab = 0x4,  // A reused first then B
    e_mme_2d_reuse_ba = 0x8,  // B reused first then A
    // Combined masks
    e_mme_1d_reuse = e_mme_1d_reuse_a | e_mme_1d_reuse_b,
    e_mme_2d_reuse = e_mme_2d_reuse_ab | e_mme_2d_reuse_ba
};

// 2nd operand reuse info
using SecondOperandReuseInfo = std::vector<bool>;
class SecondOperandReuse
{
public:
    SecondOperandReuse() = default;
    SecondOperandReuse(unsigned spatialSubviewsNr, RecipeSubviewType spatialSubviewType);
    void clear() { m_reuseInfo.clear(); }
    bool empty() const { return m_reuseInfo.empty(); }
    RecipeSubviewType getSpatialSubviewType() const { return m_spatialSubviewType; }
    void set(unsigned spatialSubviewIdx);
    bool get(unsigned spatialSubviewIdx) const;
    unsigned getSpatialSubviewsNr() const;
    bool is2ndOperandReused() const;

private:
    SecondOperandReuseInfo m_reuseInfo;  // Reuse decisions
    RecipeSubviewType m_spatialSubviewType = e_mme_non_spatial_subview;  // Relative to output
};

// Class that represent operand reuse
class ReuseInfo
{
public:
    // Construction
    void clear();
    void set2ndOperandReuseInfo(const SecondOperandReuse& secondOperandReuse);
    void setReuse(EMmeInputOperand op);
    void setSbUtilization(float utilization) { m_sbUtilization = utilization; }
    // Queries
    bool isReused(EMmeInputOperand op, unsigned spatialSubviewIdx) const;
    EMmeReuseType getReuseType() const { return m_reuseType; }
    RecipeSubviewType getSpatialSubviewType() const { return m_secondOperandReuse.getSpatialSubviewType(); }
    float getSbUtilization() const { return m_sbUtilization; }

private:
    EMmeReuseType m_reuseType = e_mme_no_reuse;
    SecondOperandReuse m_secondOperandReuse;  // Second operand reuse decision
    float m_sbUtilization = 1.f;  // In case of misalignment SB will not be fully utilized
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// MmeRecipe
////////////////////////////////////////////////////////////////////////////////////////////////////

enum RecipeType
{
    e_mme_conv_recipe,
    e_mme_bgemm_recipe
};

struct MmeRecipe
{
    MmeRecipe() = default;
    MmeRecipe(RecipeType recipeType, EMmeOpType opType) : m_recipeType(recipeType), m_opType(opType) {}

    // TODO: turn struct into class and move those variables to private section and expose only needed operations
    bool raster = false;
    unsigned signalAmount = 1;
    unsigned m_gemmNr = 1;  //  number of gemms that will be accumulated together
    bool maskedBgemm = false;
    int concurrentDim = GEMM_DIM_B1;
    unsigned concurrency = 1;
    EMmeGeometry geometry = e_mme_geometry_2xh;

    // Indices of views and RoiSizes vectors are determined by MmeRecipe::tensorIdx() which returns either 0 or 1
    static constexpr unsigned defaultViewsSize = 2;  // 0 or 1
    llvm_vecsmall::SmallVector<MmeTensorView, defaultViewsSize> aViews;
    llvm_vecsmall::SmallVector<MmeTensorView, defaultViewsSize> bViews;
    llvm_vecsmall::SmallVector<MmeTensorView, defaultViewsSize> cViews;
    llvm_vecsmall::SmallVector<SizeArray, defaultViewsSize> aRoiSizes;
    llvm_vecsmall::SmallVector<SizeArray, defaultViewsSize> bRoiSizes;
    llvm_vecsmall::SmallVector<SizeArray, defaultViewsSize> cRoiSizes;

    ReuseInfo reuseInfo;
    bool lowering = false;
    unsigned teAcceleration = 0;
    EMmeInternalOperand acceleratedOperand = e_mme_op_b;

    const MmeTensorView& getOperand(EMmeInternalOperand operand) const;
    MmeTensorView& getOperand(EMmeInternalOperand operand);
    const SizeArray& getRoiSizes(EMmeInternalOperand operand) const;

    // Access to reuse info
    EMmeReuseType reuseType() const { return reuseInfo.getReuseType(); }
    bool reuseA() const { return isReused(e_mme_in_a); }
    bool reuseB() const { return isReused(e_mme_in_b); }
    float sbUtilization() const { return reuseInfo.getSbUtilization(); }

    // Access current subview
    const SingleDimSubView& curFcd() const { return m_fcdSubviews[getIterator().fcdIdx()]; }
    const SingleDimSubView& curSp() const { return m_spSubviews[getIterator().spIdx()]; }
    const MultiDimSubView& curNonSpatial() const { return m_nonSpatialSubviews[getIterator().nonSpatialIdx()]; }
    const int tensorIdx() const;
    // Direct access to full subviews
    SingleDimSubViews& getFcdSubviews() { return m_fcdSubviews; }
    const SingleDimSubViews& getFcdSubviews() const { return m_fcdSubviews; }
    SingleDimSubViews& getSpSubviews() { return m_spSubviews; }
    const SingleDimSubViews& getSpSubviews() const { return m_spSubviews; }
    MultiDimSubViews& getNonSpatialSubviews() { return m_nonSpatialSubviews; }
    const MultiDimSubViews& getNonSpatialSubviews() const { return m_nonSpatialSubviews; }

    // Recipe iterator operations
    void createIterator(EMmeOpType opType
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
                        ,
                        bool enableReveredOrder = true
#endif
    );
    RecipeIterator& getIterator() const;

    // Special queries for partials
    void setPartialsNr(unsigned val);
    void setPartialsNrPerGemm(std::vector<unsigned> val);
    unsigned getPartialsNr() const;
    unsigned getPartialsNrPerGemmNr(unsigned num) const;
    bool isFirstPartial() const;
    bool isLastPartial() const;
    bool isMaskActivation() const;
    bool isStoreEn() const;
    bool isAccumEn() const;
    bool isReductionEn() const;

    // Other operations
    void setSplitOnBatchDims(bool val);
    bool isSplitOnBatchDims() const;
    void setPartialToMemory(bool val);
    bool isPartialToMemory() const;
    unsigned getElemSize() const
    {
        //  all tensors must be of the same DT.
        return getElementSize(aViews[0].elementType);
    }
    EMmeOpType getOpType() const { return m_opType; }
    SizeArray calcSpPos(unsigned offset) const;
    std::vector<std::string> getRecipeDebugInfo(bool verbose) const;
    RecipeType getRecipeType() { return m_recipeType; }

private:
    bool isReused(EMmeInputOperand op) const;
    static void fillSubviewsDebugInfo(std::vector<std::string>& debugInfo,
                                      const SingleDimSubViews& subviews,
                                      const std::string& subviewName,
                                      const std::string& indent);
    static void fillSubviewsDebugInfo(std::vector<std::string>& debugInfo,
                                      const MultiDimSubViews& subviews,
                                      const std::string& subviewName,
                                      const std::string& indent);

protected:
    SingleDimSubViews m_fcdSubviews;
    SingleDimSubViews m_spSubviews;
    MultiDimSubViews m_nonSpatialSubviews;

private:
    mutable std::shared_ptr<RecipeIterator> m_recipeIterator = nullptr;
    RecipeType m_recipeType = e_mme_conv_recipe;
    EMmeOpType m_opType = e_mme_fwd;
    Settable<unsigned> m_partialsNr;
    Settable<std::vector<unsigned>> m_partialsNrPerGemm;
    // BGEMM specific flag to determine if non-spatial subviews were split on common dim (false) or batch dims (true)
    Settable<bool> m_splitOnBatches;
    Settable<bool> m_partialToMemory;

    // A variable to allow doing calcSpPos(...) calculations effeciently
    mutable SizeArray m_internalSpStrides = {0, 0, 0, 0, 0};
};
}  // namespace MmeCommon

#endif //MME__RECIPE_H
