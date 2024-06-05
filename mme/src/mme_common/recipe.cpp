#include "include/mme_common/recipe.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe_generator.h"
#include "mme_assert.h"
#include <algorithm>
#include <string>
namespace MmeCommon
{
////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeIterator
////////////////////////////////////////////////////////////////////////////////////////////////////

RecipeIterator::RecipeIterator(EMmeOpType opType,
                               bool isRaster,
                               const SingleDimSubViews& fcdSubviews,
                               const SingleDimSubViews& spSubviews,
                               const MultiDimSubViews& nonSpatialSubviews,
                               bool enableReveredOrder)
: m_fcdIt(e_mme_fcd_subview, fcdSubviews.size()),
  m_nonSpatialIt(e_mme_non_spatial_subview,
                 nonSpatialSubviews.size(),
                 enableReveredOrder && (opType == e_mme_dedx || opType == e_mme_transposed_dedx)),
  m_spIt(e_mme_sp_subview, spSubviews.size()),
  m_multiIt(createIterVec(opType, isRaster)),
  m_splitNr(fcdSubviews.size() * nonSpatialSubviews.size() * spSubviews.size())
{
}

RecipeIterator::Iterators RecipeIterator::createIterVec(EMmeOpType opType, bool isRaster) const
{
    std::vector<SingleIt> iteratorVec;
    if (isDedwOperation(opType))
    {
        // In DEDW, SP is the CD so make it the very inner loop
        return isRaster ? Iterators {m_nonSpatialIt, m_fcdIt, m_spIt} : Iterators {m_fcdIt, m_nonSpatialIt, m_spIt};
    }
    // For all other operations, CONV\BATCH is the CD -> make it the very inner loop
    return isRaster ? Iterators {m_spIt, m_fcdIt, m_nonSpatialIt} : Iterators {m_fcdIt, m_spIt, m_nonSpatialIt};
}

// Check if current iteration is the last one
bool RecipeIterator::isLast() const
{
    const bool isLast =
        (m_fcdIt.isLast(m_curIterVals.at(e_mme_fcd_subview)) && m_spIt.isLast(m_curIterVals.at(e_mme_sp_subview)) &&
         m_nonSpatialIt.isLast(m_curIterVals.at(e_mme_non_spatial_subview)));
    return isLast;
}

// Check if current iteration of the given subview is the first one
bool RecipeIterator::isFirst(RecipeSubviewType subviewType) const
{
    bool isFirst = false;
    const unsigned curIdx = getIdx(subviewType);
    switch (subviewType)
    {
        case e_mme_fcd_subview:
            isFirst = m_fcdIt.isFirst(curIdx);
            break;
        case e_mme_non_spatial_subview:
            isFirst = m_nonSpatialIt.isFirst(curIdx);
            break;
        case e_mme_sp_subview:
            isFirst = m_spIt.isFirst(curIdx);
            break;
        default:
            MME_ASSERT(0, "Unsupported subview type");
    }
    return isFirst;
}

// Check if current iteration of the given subview is the last one
bool RecipeIterator::isLast(RecipeSubviewType subviewType) const
{
    bool isLast = false;
    const unsigned curIdx = getIdx(subviewType);
    switch (subviewType)
    {
        case e_mme_fcd_subview:
            isLast = m_fcdIt.isLast(curIdx);
            break;
        case e_mme_non_spatial_subview:
            isLast = m_nonSpatialIt.isLast(curIdx);
            break;
        case e_mme_sp_subview:
            isLast = m_spIt.isLast(curIdx);
            break;
        default:
            MME_ASSERT(0, "Unsupported subview type");
    }
    return isLast;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// ReuseInfo
////////////////////////////////////////////////////////////////////////////////////////////////////

SecondOperandReuse::SecondOperandReuse(unsigned spatialSubviewsNr, RecipeSubviewType spatialSubviewType)
: m_reuseInfo(spatialSubviewsNr, false), m_spatialSubviewType(spatialSubviewType)
{
}

// Set reuse info for a given spatial subview. Note that clear operation is not supported.
void SecondOperandReuse::set(unsigned spatialSubviewIdx)
{
    MME_ASSERT(spatialSubviewIdx < m_reuseInfo.size(), "Index is out of bounds");
    m_reuseInfo[spatialSubviewIdx] = true;
}

// Get reuse info for a given spatial subview
bool SecondOperandReuse::get(unsigned spatialSubviewIdx) const
{
    MME_ASSERT(spatialSubviewIdx < m_reuseInfo.size(), "Index is out of bounds");
    return m_reuseInfo[spatialSubviewIdx];
}

unsigned SecondOperandReuse::getSpatialSubviewsNr() const
{
    const unsigned size = m_reuseInfo.size();
    MME_ASSERT(size != 0, "Invalid reuse info");
    return size;
}

// Check input validity by having all spatial vectors with same size.
// If no second reuse decision was made for any subview return false. Otherwise return true.
bool SecondOperandReuse::is2ndOperandReused() const
{
    MME_ASSERT(!m_reuseInfo.empty(), "Invalid reuse info");
    const bool res = std::any_of(m_reuseInfo.cbegin(), m_reuseInfo.cend(), [](bool b) { return b; });
    return res;
}

// Reset all variables to their default values
void ReuseInfo::clear()
{
    m_reuseType = e_mme_no_reuse;
    m_secondOperandReuse.clear();
}

// Enable 2D reuse if possible
void ReuseInfo::set2ndOperandReuseInfo(const SecondOperandReuse& secondOperandReuse)
{
    MME_ASSERT(m_secondOperandReuse.empty(), "Trying to set 2nd operand reuse info more than one time");
    if (!secondOperandReuse.is2ndOperandReused())
    {
        return;  // No second reuse decision was made for any subview
    }
    // Update second reuse info
    m_secondOperandReuse = secondOperandReuse;

    // Update reuse type
    switch (m_reuseType)
    {
        case e_mme_1d_reuse_a:
            m_reuseType = e_mme_2d_reuse_ab;
            break;
        case e_mme_1d_reuse_b:
            m_reuseType = e_mme_2d_reuse_ba;
            break;
        default:
            MME_ASSERT(0, "Invalid reuse type");
    }
}

// Set one of the operands to be reused.
// This method should be invoked at most one time for each operand.
void ReuseInfo::setReuse(EMmeInputOperand op)
{
    MME_ASSERT(m_reuseType == e_mme_no_reuse, "Trying to set reuse type more than one time");
    m_reuseType = (op == e_mme_in_a) ? e_mme_1d_reuse_a : e_mme_1d_reuse_b;
}

//  Check if the given operand is reused
bool ReuseInfo::isReused(EMmeInputOperand op, unsigned spatialSubviewIdx) const
{
    switch (m_reuseType)
    {
        case e_mme_no_reuse:
            return false;
        case e_mme_1d_reuse_a:
            return (op == e_mme_in_a);
        case e_mme_1d_reuse_b:
            return (op == e_mme_in_b);
        case e_mme_2d_reuse_ab:
            if (op == e_mme_in_a) return true;
            break;
        case e_mme_2d_reuse_ba:
            if (op == e_mme_in_b) return true;
            break;
        default:
            MME_ASSERT(0, "Unsupported reuse type");
    }

    // It's 2D reuse
    assert((m_reuseType == e_mme_2d_reuse_ab) || (m_reuseType == e_mme_2d_reuse_ba));
    return m_secondOperandReuse.get(spatialSubviewIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MmeRecipe
////////////////////////////////////////////////////////////////////////////////////////////////////

// Set number of partials
void MmeRecipe::setPartialsNr(unsigned val)
{
    MME_ASSERT(!m_partialsNr.is_set(), "It's not possible to initialize number of partials more than one time");
    m_partialsNr = val;
}

void MmeRecipe::setPartialsNrPerGemm(std::vector<unsigned> val)
{
    MME_ASSERT(!m_partialsNrPerGemm.is_set(),
               "It's not possible to initialize number of partials per gemm more than one time");
    m_partialsNrPerGemm = val;
}

// Return total number of partials
unsigned MmeRecipe::getPartialsNr() const
{
    MME_ASSERT(m_partialsNr.is_set(), "Partials number was not specified");
    return m_partialsNr.value();
}

unsigned MmeRecipe::getPartialsNrPerGemmNr(unsigned num) const
{
    MME_ASSERT(m_partialsNrPerGemm.is_set(), "Partials number per gemm was not specified");
    return m_partialsNrPerGemm.value().at(num);
}

// Return 'true' if current iteration is on the first partial
bool MmeRecipe::isFirstPartial() const
{
    // partials is not yet enabled on batch splitting - so whenever there is batch split its always the first partial
    if (isSplitOnBatchDims()) return true;

    if (m_recipeType == e_mme_bgemm_recipe)
    {
        const unsigned curSubviewIdx = getIterator().nonSpatialIdx();
        const unsigned partialsNr = getPartialsNr();
        return ((curSubviewIdx % partialsNr) == 0);
    }
    else if (isDedwOperation(m_opType))
    {
        return getIterator().isFirst(e_mme_sp_subview);
    }
    return getIterator().isFirst(e_mme_non_spatial_subview);
}

// Return 'true' if current iteration is on the last partial
bool MmeRecipe::isLastPartial() const
{
    // partials is not yet enabled on batch splitting - so whenever there is batch split its always the last partial
    if (isSplitOnBatchDims()) return true;

    if (m_recipeType == e_mme_bgemm_recipe)
    {
        const unsigned curSubviewIdx = getIterator().nonSpatialIdx();
        const unsigned partialsNr = getPartialsNr();
        const unsigned lastPartialIdx = partialsNr - 1;
        return ((curSubviewIdx % partialsNr) == lastPartialIdx);
    }
    else if (isDedwOperation(m_opType))
    {
        return getIterator().isLast(e_mme_sp_subview);
    }
    return getIterator().isLast(e_mme_non_spatial_subview);
}

bool MmeRecipe::isMaskActivation() const
{
    if (m_gemmNr == 2)
    {
        const unsigned curSubviewIdx = getIterator().nonSpatialIdx();
        const unsigned partialsNr = getPartialsNr();
        unsigned partialsNrFirstGemm = getPartialsNrPerGemmNr(0);
        return !((curSubviewIdx % partialsNr) < partialsNrFirstGemm);
    }
    return false;
}
// Determines the value of storeEn of MME descriptor
bool MmeRecipe::isStoreEn() const
{
    return isPartialToMemory() || isLastPartial();
}

// Determines the value of reduction mode of MME descriptor
bool MmeRecipe::isReductionEn() const
{
    // we will perform reduction in partials to memory mode from the second partial onwards
    return isPartialToMemory() && !isFirstPartial();
}

// Determines the value of accumEn of MME descriptor
bool MmeRecipe::isAccumEn() const
{
    // accumulating in the ACC only from the second partial onwards and as long as we are not in partials to memory mode
    return !isPartialToMemory() && !isFirstPartial();
}

// Recipe generator made a split on batch dims
void MmeRecipe::setSplitOnBatchDims(bool val)
{
    MME_ASSERT(!m_splitOnBatches.is_set(),
               "It's not possible to set split dim of non-spatial subviews more than one time");
    m_splitOnBatches = val;
}

// Recipe generator decided to perform partials as reduction to memory
void MmeRecipe::setPartialToMemory(bool val)
{
    MME_ASSERT(!m_partialToMemory.is_set(),
               "It's not possible to set split dim of non-spatial subviews more than one time");
    m_partialToMemory = val;
}

// Query making split on batch dim
bool MmeRecipe::isSplitOnBatchDims() const
{
    if (!m_splitOnBatches.is_set())
    {
        return false;
    }
    return m_splitOnBatches.value();
}

// Query partials mode
bool MmeRecipe::isPartialToMemory() const
{
    if (!m_partialToMemory.is_set())
    {
        return false;
    }
    return m_partialToMemory.value();
}

// Spatial dimensions BHW are squashed to one dimension. This function maps 1D offset to multi-dimensional.
SizeArray MmeRecipe::calcSpPos(unsigned offset) const
{
    // Onetime calculation for internal spatial strides:
    // (1) Calculate contiguous strides explicitly as view strides may be related to original tensor
    //     that was splitted by Synapse slicer.
    // (2) Exclude dim[1] - by dividing all elements from previous result by that dim
    // This can be done similar to calcContiguousStrides(...):

    // in DEDW the spatial movement is on the common dimension so we will take it from B (Y operand in DEDW)
    MmeCommon::SizeArray roiSize = isDedwOperation(m_opType) ? bRoiSizes[tensorIdx()] : cRoiSizes[tensorIdx()];
    if (m_internalSpStrides[0] == 0)
    {
        m_internalSpStrides[0] = m_internalSpStrides[1] = 1;
        for (unsigned dim = 1; dim < MAX_DIMENSION - 1; dim++)
        {
            m_internalSpStrides[dim + 1] = m_internalSpStrides[dim] * roiSize[dim];
            MME_ASSERT(m_internalSpStrides[dim + 1] != 0, "Invalid sizes");
        }
    }

    // Calculate the return result.
    // The idea is to determine the contribution of each dimension starting from last one.
    SizeArray multiDimPos = {0};
    unsigned pos = offset;
    for (int d = MAX_DIMENSION - 2; d >= 0; d--)
    {
        multiDimPos[d] = pos / m_internalSpStrides[d + 1];
        pos %= m_internalSpStrides[d + 1];
    }
    return multiDimPos;
}

// Check if given operand is either 1D or 2D reused
bool MmeRecipe::isReused(EMmeInputOperand op) const
{
    unsigned spatialSubviewIdx = -1U;
    if (((reuseType() == e_mme_2d_reuse_ab) && (op == e_mme_in_b)) ||
        ((reuseType() == e_mme_2d_reuse_ba) && (op == e_mme_in_a)))
    {
        spatialSubviewIdx = getIterator().getIdx(reuseInfo.getSpatialSubviewType());
    }
    return reuseInfo.isReused(op, spatialSubviewIdx);
}

void MmeRecipe::createIterator(EMmeOpType opType
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
                               ,
                               bool enableReversedOrder
#endif
)
{
    MME_ASSERT(m_recipeIterator == nullptr, "Trying to initialize recipe iterator more than one time");
    m_recipeIterator = std::make_shared<RecipeIterator>(opType,
                                                        raster,
                                                        m_fcdSubviews,
                                                        m_spSubviews,
                                                        getNonSpatialSubviews()
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
                                                            ,
                                                        enableReversedOrder
#endif
    );
}

RecipeIterator& MmeRecipe::getIterator() const
{
    MME_ASSERT(m_recipeIterator != nullptr, "Invalid recipe iterator");
    return *m_recipeIterator;
}

void MmeRecipe::fillSubviewsDebugInfo(std::vector<std::string>& debugInfo,
                                      const SingleDimSubViews& subviews,
                                      const std::string& subviewName,
                                      const std::string& indent)
{
    if (subviews.size() <= 1)
    {
        return;
    }

    std::string subviewStr;
    unsigned repeats = 1;
    for (unsigned i = 0; i < subviews.size(); i++)
    {
        if ((i != 0) && (subviews[i - 1].viewSize == subviews[i].viewSize))
        {
            repeats++;
            continue;
        }
        else if (repeats != 1)
        {
            subviewStr += 'x' + std::to_string(repeats);
            repeats = 1;
        }

        if ((i != 0) && (i < subviews.size()))
        {
            subviewStr += ", ";
        }

        subviewStr += std::to_string(subviews[i].viewSize);
    }

    if (repeats != 1)
    {
        subviewStr += 'x' + std::to_string(repeats);
    }
    subviewStr = indent + subviewName + " splits: { " + subviewStr + " }";
    debugInfo.push_back(subviewStr);
}

void MmeRecipe::fillSubviewsDebugInfo(std::vector<std::string>& debugInfo,
                                      const MultiDimSubViews& subviews,
                                      const std::string& subviewName,
                                      const std::string& indent)
{
    if (subviews.size() <= 1)
    {
        return;
    }

    std::string subviewStr;
    unsigned repeats = 1;
    for (unsigned i = 0; i < subviews.size(); i++)
    {
        bool skip = (i != 0);
        if (skip)
        {
            for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
            {
                if (subviews[i - 1].sizes[dim] != subviews[i].sizes[dim])
                {
                    skip = false;
                    break;
                }
            }
        }
        if (skip)
        {
            repeats++;
            continue;
        }
        else if (repeats != 1)
        {
            subviewStr += 'x' + std::to_string(repeats);
            repeats = 1;
        }

        if ((i != 0) && (i < subviews.size()))
        {
            subviewStr += ", ";
        }

        subviewStr += '[';
        for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS - 1; dim++)
        {
            subviewStr += std::to_string(subviews[i].sizes[dim]) + ", ";
        }
        subviewStr += std::to_string(subviews[i].sizes[MME_MAX_TENSOR_DIMS - 1]) + "]";
    }

    if (repeats != 1)
    {
        subviewStr += 'x' + std::to_string(repeats);
    }
    subviewStr = indent + subviewName + " splits: { " + subviewStr + " }";
    debugInfo.push_back(subviewStr);
}

std::vector<std::string> MmeRecipe::getRecipeDebugInfo(bool verbose) const
{
    std::vector<std::string> debugInfo;

    // Convert operand use info into compact string
    std::string reuseOp;
    switch (reuseType())
    {
        case e_mme_no_reuse:
            reuseOp = "N/A";
            break;
        case e_mme_1d_reuse_a:
            reuseOp = "A";
            break;
        case e_mme_1d_reuse_b:
            reuseOp = "B";
            break;
        case e_mme_2d_reuse_ab:
            reuseOp = "AB";
            break;
        case e_mme_2d_reuse_ba:
            reuseOp = "BA";
            break;
        default:
            MME_ASSERT(0, "Unsupported reuse type");
    }

    // SB utilization
    std::string sbUtilStr;
    if (reuseType() != e_mme_no_reuse)
    {
        const float sbUtilization = reuseInfo.getSbUtilization();
        sbUtilStr = ", SBUtilization=" + std::to_string(unsigned(sbUtilization * 100)) + "%";
    }

    const std::string isRaster = raster ? "True" : "False";
    MME_ASSERT((m_recipeType == e_mme_bgemm_recipe) || (m_recipeType == e_mme_conv_recipe), "Unsupported recipe type");
    const bool isConv = (m_recipeType == e_mme_conv_recipe);
    const std::string nonSpatialSubviewsStr = isConv ? "CONV" : "BATCH";

    std::string summary;
    if (isConv)
    {
        const std::string isLowered = lowering ? "True" : "False";
        summary = "lower=" + isLowered + ", ";
    }
    summary += "raster=" + isRaster + ", reuse=" + reuseOp + ", FCD=" + std::to_string(m_fcdSubviews.size()) +
               ", SP=" + std::to_string(m_spSubviews.size()) + ", " + nonSpatialSubviewsStr + "=" +
               std::to_string(m_nonSpatialSubviews.size()) + sbUtilStr;
    debugInfo.push_back(summary);

    if (verbose)
    {
        static const std::string indent = "    ";
        fillSubviewsDebugInfo(debugInfo, m_fcdSubviews, "FCD", indent);
        fillSubviewsDebugInfo(debugInfo, m_spSubviews, "SP", indent);
        fillSubviewsDebugInfo(debugInfo, m_nonSpatialSubviews, nonSpatialSubviewsStr, indent);
    }

    return debugInfo;
}

MmeTensorView& MmeRecipe::getOperand(EMmeInternalOperand operand)
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
            return aViews[tensorIdx()];
        case e_mme_op_b:
            return bViews[tensorIdx()];
        case e_mme_op_c:
            return cViews[tensorIdx()];
    }
}

const MmeTensorView& MmeRecipe::getOperand(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
            return aViews[tensorIdx()];
        case e_mme_op_b:
            return bViews[tensorIdx()];
        case e_mme_op_c:
            return cViews[tensorIdx()];
    }
}

const SizeArray& MmeRecipe::getRoiSizes(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
            return aRoiSizes[tensorIdx()];
        case e_mme_op_b:
            return bRoiSizes[tensorIdx()];
        case e_mme_op_c:
            return cRoiSizes[tensorIdx()];
    }
}

const int MmeRecipe::tensorIdx() const
{
    if (m_recipeIterator)
    {
        //  is the current tensor mask or bgemm
        return isMaskActivation() ? 1 : 0;
    }
    else
    {
        //  havent set interator yet, return  main tensor idx
        return 0;
    }
}

}  // namespace MmeCommon
