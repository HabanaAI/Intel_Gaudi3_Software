#include "include/mme_common/recipe_generator.h"
#include "include/mme_common/conv_sub_problems.h"
#include "common_geo_attr.h"
#include "mme_hal_reader.h"
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_common_utils.h"
#include "include/mme_common/workarounds.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "mme_assert.h"
#include "src/utils/logger.h"
#include <algorithm>

// A threshold used to avoid having high number of descriptors when applying SB reuse for BGEMM
#define MAX_BATCH_NR_FOR_SB_REUSE 16

// If '0', don't allow splitting. Work as if user mode is no-reuse. If '1' work normally
#define ENABLE_SB_REUSE 1
// Force no-SB-reuse while keeping sub-view split - for debug purposes
#define FORCE_NO_SB_REUSE_BUT_KEEP_SPLIT_RESULT 0
// Enable balancing on number of atomic units between steps
#define ENABLE_SUB_VIEW_BALANCING 1
// Should consider the pipeline level hint from graph compiler?
#define ENABLE_PIPELINE_LEVEL_HINT 1
// Enable 2D SB reuse - TODO:[SW-74393] re-enable 2d Reuse
#define ENABLE_2D_SB_REUSE 0

// All temporary or workaround code must be activated using this macro definition:
#ifndef ENABLE_RECIPE_WORKAROUNDS
static_assert(false, "ENABLE_RECIPE_WORKAROUNDS must be defined");
#endif

namespace MmeCommon
{
// Valid alignment for reduction tree.
// "roundDown": determines how to round the misalignment.
unsigned calcCommonDimAlignment(const MmeRecipe& recipe,
                                const MmeLayerParams& params,
                                const CommonGeoAttr& geoAttr,
                                const MmeHalReader& mmeHalReader,
                                unsigned val,
                                bool alignToCL,
                                bool roundDown)
{
    const unsigned numElementsInCacheLine = mmeHalReader.getClSize() / recipe.getElemSize();
    unsigned baseCommonDim = alignToCL ? numElementsInCacheLine : 1;
    unsigned cdDtAlignment =
        mmeHalReader.getNumElementsForCommonDimAlignment(recipe.bViews[0].elementType, params.opType);

    // In case ports are interleaved on the common dim, each port sees only part of the spatial dims.
    // So we need to make sure that after the division by the interleaving factor, cdDtAlignment is still aligned
    unsigned cdSpInterleavingAlignment = 1;
    if (!geoAttr.isTransposed(e_mme_op_a))
    {
        cdSpInterleavingAlignment =
            std::max(cdSpInterleavingAlignment, geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a));
    }
    if (!geoAttr.isTransposed(e_mme_op_b))
    {
        cdSpInterleavingAlignment =
            std::max(cdSpInterleavingAlignment, geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b));
    }
    // Align also according to all CommonDim dimensions below the dim on which we interleave
    unsigned spInterleavingDim = geoAttr.getSpInterleavingDim(e_mme_op_a);
    for (int d = DIM_W; d < spInterleavingDim; d++)
    {
        cdSpInterleavingAlignment *= recipe.bViews[0].sizes[d];
    }
    unsigned cDAlignment = cdDtAlignment * cdSpInterleavingAlignment;
    baseCommonDim = std::max(baseCommonDim, cDAlignment);

    MME_ASSERT(baseCommonDim != 0, "Wrong base common dim calculation");
    if (val < baseCommonDim)
    {
        return baseCommonDim;
    }
    // Align to baseCommonDim
    unsigned retVal =
        (roundDown ? div_round_down(val, baseCommonDim) : div_round_up(val, baseCommonDim)) * baseCommonDim;
    return retVal;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// GridBase
////////////////////////////////////////////////////////////////////////////////////////////////////

GridBase::GridBase(GridType gridType,
                   const SizeArray& bases,
                   const SizeArray& sizes,
                   const SizeArray& origSizes,
                   const RecipeConstants& constants,
                   unsigned splitDim)
: m_gridType(gridType),
  m_bases(bases),
  m_sizes(sizes),
  m_origSizes(origSizes),
  m_splitDim(splitDim),
  m_constants(constants)
{
}

// Return number of elements in a given step
unsigned GridBase::calcStepSize(unsigned stepIdx) const
{
    MME_ASSERT(stepIdx < m_gridSize, "Trying to find size of a step outside the grid");
    // Map the given index to sub-index in the template (the repeated pattern)
    const unsigned stepsNrPerTemplate = calcStepsNrPerTemplate();
    const unsigned stepIdxInTemplate = stepIdx % stepsNrPerTemplate;
    // Find out if given index match first or last step. First steps came first.
    const unsigned unitsPerStepNr = (stepIdxInTemplate < m_firstStepsNr) ? m_unitsNrPerFirstStep : m_unitsNrPerLastStep;
    // Calc actual step size:
    unsigned stepSize = unitsPerStepNr * m_atomicUnitLength;
    // Last sub-view may have less elements. In this case involve original size of the tensor
    if ((m_partialUnitLength != 0) && isPartialStep(stepIdxInTemplate))
    {
        stepSize = stepSize - m_atomicUnitLength + m_partialUnitLength;  // Replace one atomic unit with partial one
    }
    MME_ASSERT(stepSize != 0, "Step size at given index is zero");
    return stepSize;
}

unsigned GridBase::getSplitDim() const
{
    MME_ASSERT(m_splitDim != MME_MAX_TENSOR_DIMS, "Invalid split dim");
    return m_splitDim;
}

// Return how many steps in the grid
unsigned GridBase::getGridSize() const
{
    MME_ASSERT(m_gridSize != 0, "Grid is not initialized yet");
    return m_gridSize;
}

// Return 'true' is there should be a split into multiple sub-views.
// Return 'false' for a single sub-view case.
bool GridBase::isMultiStep() const
{
    if (!m_isCreated)
    {
        return false;
    }
    MME_ASSERT(m_gridSize != 0, "Grid is not initialized yet");
    return (m_gridSize >= 2);
}

// Calculate the length (in element units) of dimensions that participate in the view.
// Dimensions are from 'dimStart' to 'dimEnd' (all included).
unsigned GridBase::calcViewSize(bool origSize, unsigned dimStart, unsigned dimEnd) const
{
    MME_ASSERT((dimStart <= dimEnd) && (dimEnd < m_sizes.size()), "Given range is invalid");
    unsigned viewLength = 1;
    for (unsigned dim = dimStart; dim <= dimEnd; dim++)
    {
        if (m_sizes[dim] != 0)
        {
            viewLength *= origSize ? m_origSizes[dim] : m_sizes[dim];
        }
    }
    return viewLength;
}

// Return view size from split dim till the end excluding zero dims.
// 'inAtomicUnits' determines in which units to return the size.
unsigned GridBase::getViewSize(bool inAtomicUnits, bool origSize) const
{
    unsigned dimStart = 0;
    if (inAtomicUnits && (m_splitDim != MME_MAX_TENSOR_DIMS))
    {
        dimStart = m_splitDim;
    }
    return calcViewSize(origSize, dimStart);
}

// Return number of steps in the template
unsigned GridBase::calcStepsNrPerTemplate() const
{
    return m_firstStepsNr + m_lastStepsNr;
}

// Return total number of atomic units in the template
unsigned GridBase::calcUnitsNrPerTemplate() const
{
    return (m_firstStepsNr * m_unitsNrPerFirstStep) + (m_lastStepsNr * m_unitsNrPerLastStep);
}

// Check if the grid is balanced or not
bool GridBase::isBalanced() const
{
    if (m_firstStepsNr == 0)
    {
        return false;  // First steps must be different than zero
    }
    if (m_lastStepsNr == 0)
    {
        return true;  // No last steps, the grid has uniform steps
    }
    // Check if steps are already balanced
    if ((m_unitsNrPerFirstStep >= m_unitsNrPerLastStep) && ((m_unitsNrPerFirstStep - m_unitsNrPerLastStep) <= 1))
    {
        return true;  // (first - last <= 1)  means the grid is balanced
    }
    return false;
}

// In order ro accomplish optimal performance we need to follow a heuristic that balance steps
// such that any couple should differ in one atomic unit at most.
void GridBase::applyBalancing()
{
#if not ENABLE_SUB_VIEW_BALANCING
    return;
#endif

    if (isBalanced())
    {
        return;  // The grid is already balanced
    }

    // Do the balance:
    const unsigned totalStepsNr = calcStepsNrPerTemplate();
    const unsigned totalUnitsNr = calcUnitsNrPerTemplate();
    const unsigned optimalUnitsPerStepNr = totalUnitsNr / totalStepsNr;
    MME_ASSERT(optimalUnitsPerStepNr > 0, "Invalid subviews split");
    // Calc outlier number and then distribute it among steps
    const unsigned outlierUnitsNr = totalUnitsNr % totalStepsNr;
    MME_ASSERT(outlierUnitsNr < totalStepsNr, "We receive a totalStepsNr that must hold this inequality true");

    if (outlierUnitsNr == 0)  // A case of all steps are uniform, no need for a last step
    {
        m_unitsNrPerLastStep = 0;
        m_lastStepsNr = 0;
        m_unitsNrPerFirstStep = optimalUnitsPerStepNr;
        m_firstStepsNr = totalStepsNr;
    }
    else  // A case where first step is larger than the second by one
    {
        m_unitsNrPerLastStep = optimalUnitsPerStepNr;
        m_lastStepsNr = totalStepsNr - outlierUnitsNr;
        m_unitsNrPerFirstStep = optimalUnitsPerStepNr + 1;  // +1 for each outlier unit
        m_firstStepsNr = outlierUnitsNr;
    }

    // Verify correctness
    MME_ASSERT(isBalanced(), "Applying balancing should leave the grid balanced");
    // Total number of units must be the same before and after balancing
    const unsigned totalUnitsAfterBalancing = calcUnitsNrPerTemplate();
    MME_ASSERT(totalUnitsAfterBalancing == totalUnitsNr, "Balancing changed number of units");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid
////////////////////////////////////////////////////////////////////////////////////////////////////

Grid::Grid(GridType gridType,
           const SizeArray& bases,
           const SizeArray& sizes,
           const SizeArray& origSizes,
           const RecipeConstants& constants,
           unsigned splitDim)
: GridBase(gridType,
           bases,
           sizes,
           origSizes,
           constants,
           splitDim)
{
    MME_ASSERT(splitDim < MME_MAX_TENSOR_DIMS, "Invalid given split dim");
}

// Primary method to create a grid for FCD/SP/CONV - to be used as guidance to create the sub-views.
// Step sizes of the grid must balanced so that difference between any couple is at most one.
// Input parameters:
// - maxStepSize: geometry size in elements unit of the given dimension
// - stepCapacity: maximum number of geometries that can fit into one step
void Grid::create(unsigned maxStepSize, unsigned stepCapacity)
{
    MME_ASSERT((maxStepSize != 0) == (stepCapacity != 0), "Invalid given max/capacity of setps");
    MME_ASSERT(!m_isCreated, "For each grid only onetime creation is allowed");
    m_isCreated = true;
    if (m_gridType == CONV_GRID)
    {
        createForConv(maxStepSize, stepCapacity);
    }
    else
    {
        createForDefault(maxStepSize, stepCapacity);
    }
}

// Optional extension of the grid up to given size.
// 'm_partialUnitLength' and 'm_atomicUnitLength" are not affected by the extension.
void Grid::extend(unsigned newSize)
{
    // If the following assert hit it means there is no meaning for the extension request
    MME_ASSERT(newSize > m_gridSize, "Invalid grid extension request");
    const unsigned stepsNrPerTemplate = calcStepsNrPerTemplate();
    MME_ASSERT((stepsNrPerTemplate >= 1) && ((m_gridSize % stepsNrPerTemplate) == 0), "A basic grid assumption failed");
    const unsigned repeatsNr = m_gridSize / stepsNrPerTemplate;
    MME_ASSERT(repeatsNr == 1, "A basic grid assumption failed");

    // Try to extend the template by distributing total elements over the new steps
    const unsigned totalUnitsNrPerTemplate = calcUnitsNrPerTemplate();
    const unsigned newStepsPerTemplate =
        std::min(newSize / repeatsNr,  // Required number of steps per template
                 totalUnitsNrPerTemplate);  // It's impossible to extend beyond total number of elements
    if (newStepsPerTemplate <= stepsNrPerTemplate)
    {
        return;
    }

    // Distribute total units (per template) over the new steps
    const unsigned viewSize = getViewSize();  // In atomic units
    const unsigned requiredGridSize = repeatsNr * newStepsPerTemplate;
    distributeUniformly(requiredGridSize, viewSize);

    // Verify that total number of units (per template) had not been changed
    const unsigned newTotalUnitsNr = calcUnitsNrPerTemplate();
    MME_ASSERT(newTotalUnitsNr == totalUnitsNrPerTemplate, "Total units in a template was changed after extension");

    // Verify and modify m_gridSize to the new size
    const unsigned newGridSize = repeatsNr * calcStepsNrPerTemplate();
    MME_ASSERT((newGridSize >= m_gridSize) && (newGridSize <= newSize), "Invalid calculation of grid size");
    m_gridSize = newGridSize;
}

// Create a grid for FCD/SP - to be used as guidance to create the sub-views
void Grid::createForDefault(unsigned maxStepSize, unsigned stepCapacity)
{
    const unsigned viewSize = calcViewSize();
    // Decide if grid will have one or multiple steps
    m_gridSize = (maxStepSize == 0) ? 1 : div_round_up(viewSize, maxStepSize * stepCapacity);
    // Each step in the grid contains number of atomic units. Size of this unit is geometry's length
    m_atomicUnitLength = (maxStepSize == 0) ? 1 : maxStepSize;
    distributeUniformly(m_gridSize, viewSize);
    // Initialize size of last step
    m_partialUnitLength = viewSize % m_atomicUnitLength;
}

// Given a grid size and a view size, format the grid so that both first and last steps
// receive a uniform distribution of atomic units.
void Grid::distributeUniformly(unsigned gridSize, unsigned viewSize)
{
    MME_ASSERT(gridSize != 0, "Invalid grid size input");
    // Calculated total number of geometries
    const unsigned totalGeoNr = div_round_up(viewSize, m_atomicUnitLength);
    MME_ASSERT(totalGeoNr >= gridSize, "Empty splits detected");

    // Find optimal step size and the last one. It reflects a uniform distribution for all steps but the last.
    const unsigned firstStepLength = totalGeoNr / gridSize;
    MME_ASSERT(firstStepLength >= 1, "Length of 1st step must contain at least one unit");
    const unsigned remainingGeoNr = totalGeoNr % gridSize;  // If the result is 0, it means we have equal distribution

    // First steps are larger by one from last ones. This way we preserve difference between any couple to be at most 1
    m_unitsNrPerFirstStep = (remainingGeoNr == 0) ? firstStepLength : (firstStepLength + 1);
    m_firstStepsNr = (gridSize == 1) ? 1 :  // We need to keep at least one first step
                         ((remainingGeoNr == 0) ? (gridSize / 2) : remainingGeoNr);
    m_unitsNrPerLastStep = firstStepLength;
    m_lastStepsNr = gridSize - m_firstStepsNr;

    // Final verification
    const unsigned actualGeoNr = calcUnitsNrPerTemplate();
    MME_ASSERT(actualGeoNr == totalGeoNr, "Number of geometries has changed");
}

// Create a grid for CONV - to be used as guidance to create the sub-views
void Grid::createForConv(unsigned geoSize, unsigned stepCapacity)
{
    // Don't allow filter crossing. In this case grid size consists of two parts:
    // Part 1: calc number of filters
    const unsigned viewSize = calcViewSize();
    const unsigned firstConvDimLength = m_sizes[m_splitDim];
    const unsigned filtersNr = viewSize / firstConvDimLength;
    MME_ASSERT(filtersNr != 0, "Expected at least one filter");
    // Part 2: calc number of geometries that cover the first CONV dim
    m_geoAtFirstConvDimNr = div_round_up(firstConvDimLength, geoSize);
    m_gridSize = m_geoAtFirstConvDimNr * filtersNr;  // Part1 * Part2
    if (m_gridSize == 1)
    {
        return;
    }

    // Calc how many geo can fit into first step considering capacity
    m_unitsNrPerFirstStep = std::min(m_geoAtFirstConvDimNr, stepCapacity);
    unsigned curTotalUnits = m_unitsNrPerFirstStep;  // Total units included by 1st step so far
    // Check if we can add more. Note in reuse A, stepCapacity=1 and so no further units will be added.
    for (unsigned dim = 2; dim < m_sizes.size(); dim++)
    {
        // As we proceed into higher dimensions we should take m_unitsNrPerFirstStep as a whole.
        // Otherwise will take partial filter which is illegal.
        const unsigned dimLimit = stepCapacity / curTotalUnits;  // How many dim units we can add to 1st step
        if (dimLimit <= 1)
        {
            break;  // Can't accept more units
        }
        // Add the new units
        m_unitsNrPerFirstStep = std::min(m_sizes[dim], dimLimit);
        m_splitDim = dim;
        curTotalUnits *= m_unitsNrPerFirstStep;
        MME_ASSERT(curTotalUnits <= stepCapacity, "Units calculation result exceeds step capacity");
    }
    MME_ASSERT((m_unitsNrPerFirstStep != 0) && (curTotalUnits <= stepCapacity), "Wrong units distribution");
    // From this point on m_unitsNrPerFirstStep has its final value

    // We handle two cases: all template fits into first common dim or not fit into
    const bool templateFitsIntoFirstDim = (m_splitDim == 1);
    if (templateFitsIntoFirstDim)  // Split is on first common dim
    {
        m_atomicUnitLength = geoSize;
        m_firstStepsNr = m_geoAtFirstConvDimNr / m_unitsNrPerFirstStep;
        m_unitsNrPerLastStep = m_geoAtFirstConvDimNr % m_unitsNrPerFirstStep;
        m_lastStepsNr = (m_unitsNrPerLastStep == 0) ? 0 : 1;
        // Fix grid size
        m_gridSize = filtersNr * (m_firstStepsNr + m_lastStepsNr);
        // Calc partial geo
        m_partialUnitLength = firstConvDimLength % m_atomicUnitLength;
    }
    else  // Split is on dim higher than first common dim
    {
        m_atomicUnitLength = 1;  // Higher dims goes in dims units - one
        m_firstStepsNr = m_sizes[m_splitDim] / m_unitsNrPerFirstStep;
        m_unitsNrPerLastStep = m_sizes[m_splitDim] % m_unitsNrPerFirstStep;
        m_lastStepsNr = (m_unitsNrPerLastStep == 0) ? 0 : 1;
        // Fix grid size. Calc repeats which are determined by dimensions beyond split-dim
        const unsigned repeats =
            (m_sizes.size() == m_splitDim + 1) ? 1 : multiplyElements(m_sizes.begin() + m_splitDim + 1, m_sizes.end());
        m_gridSize = repeats * (m_firstStepsNr + m_lastStepsNr);
    }
    MME_ASSERT((m_firstStepsNr != 0) && (m_gridSize != 0), "Wrong grid calculation");
    applyBalancing();
}

// Return 'true' if current step (stepIdx) should be partial. 'false' otherwise.
bool Grid::isPartialStep(unsigned stepIdxInTemplate) const
{
    if
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
        (
#endif
            (m_gridType == CONV_GRID)
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
            && (m_constants.mmeHalReader.getChipType() == ChipType::e_mme_Gaudi) &&
            (m_constants.params.isDedwOperation()))
#endif
    {
        if ((stepIdxInTemplate % m_geoAtFirstConvDimNr) == 0)
        {
            return true;  // Last step at split dim (case 3)
        }
    }
    else
    {
        if (stepIdxInTemplate == (calcStepsNrPerTemplate() - 1))
        {
            return true;  // Last sub-view (case 1)
        }
    }
    return false;  // Not a last step
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CommonDimGrid
////////////////////////////////////////////////////////////////////////////////////////////////////

CommonDimGrid::CommonDimGrid(GridType gridType,
                             const SizeArray& bases,
                             const SizeArray& sizes,
                             const SizeArray& origSizes,
                             const RecipeConstants& constants,
                             unsigned fcdDim,
                             bool isMultipleCdCuts)
: GridBase(gridType, bases, sizes, origSizes, constants), m_fcdDim(fcdDim), m_multipleCdCuts(isMultipleCdCuts)
{
    MME_ASSERT(fcdDim < MME_MAX_TENSOR_DIMS, "Invalid FCD dim input");
    MME_ASSERT(sizes[fcdDim] == 0, "FCD size should be zero by definition");
}

// Create CommonDim grid given those parameters:
//  - lastIncludedDim: dimensions index where data associated with all sub-dimensions is fit into SB.
//  - atomicUnitLength: length of data that defines the granularity of the grid (it can be one for higher dims or any
//  number in elements unit).
//  - maxFitNr: maximum atomic units we can consume in a single step.
void CommonDimGrid::create(const CommonDimParams& params)
{
    MME_ASSERT(!m_isCreated, "Common dim grid is already created");
    if (!params.forcedPartial)
    {
        MME_ASSERT(params.reuseType != UNKNOWN_SB_REUSE, "Invalid reuse type");
        MME_ASSERT((params.maxFitNr != 0) && (params.atomicUnitLength != 0), "Invalid common dim creation params");
    }
    m_isCreated = true;
    m_atomicUnitLength = params.atomicUnitLength;
    m_reuseType = params.reuseType;
    m_splitDim = calcSplitDim(params.lastIncludedDim);
    // For partial SB reuse we need to calculate some other additional parameters.
    // that are used to define the partial SB reuse
    if (isPartialReuse())
    {
        // Check validity of lastIncludedDim
        MME_ASSERT(
            (params.lastIncludedDim <= MME_MAX_TENSOR_DIMS) &&
                ((params.lastIncludedDim == MME_MAX_TENSOR_DIMS) == (m_reuseType == SB_PARTIAL_REUSE_NO_DIM_INCLUDED)),
            "Invalid common dim creation params");
        // Calculate parameters for two cases: at least one (or no) dim is fully included in SB
        // Once the following if-else is done, those member variables will be initialized:
        // m_atomicUnitLength, m_unitsNrPerFirstStep, m_firstStepsNr, m_unitsNrPerLastStep, m_lastStepsNr
        if (isAtLeastOneDimIncluded())
        {
            splitAtLeastOneDimIncluded(params.maxFitNr);
        }
        else
        {
            splitNoDimIncluded(params.maxFitNr);
        }
        applyBalancing();
    }
    else  // no SB reuse and non-partial SB reuse goes here
    {
        splitAtLeastOneDimIncluded(params.maxFitNr);
    }

    // Final checks, all parameters must be valid
    MME_ASSERT(m_atomicUnitLength != 0, "Invalid calculation");
    MME_ASSERT(m_unitsNrPerFirstStep != 0, "Invalid calculation");
    MME_ASSERT(m_firstStepsNr != 0, "Invalid calculation");
    // First and last steps must fulfill maxFitNr constraint
    MME_ASSERT((m_unitsNrPerFirstStep <= params.maxFitNr) && (m_unitsNrPerLastStep <= params.maxFitNr),
               "Invalid calculation");

    // Calculate how many steps in this grid
    calcGridSize();
}

// Define multi-step grid when at least one dim is fully included in SB.
// At the end we guarantee (m_unitsNrPerLastStep < m_unitsNrPerFirstStep) and (m_lastStepsNr = 0/1).
void CommonDimGrid::splitAtLeastOneDimIncluded(unsigned maxFitNr)
{
    // This assert is not really a bug. But to obligate having consistent representation of no-split case.
    // As we can represented in two ways:
    //   1. atomicUnitLength=1, maxFitNr="size of last dim"
    //   2. atomicUnitLength="size of last dim", maxFitNr=1
    MME_ASSERT(m_atomicUnitLength == 1, "Flow violated usage agreement");  // Make sure that option 1 is adapted

    const unsigned incDimSize = m_sizes[getSplitDim()];
    MME_ASSERT((incDimSize % m_atomicUnitLength) == 0, "Included dim size must represent atomic units");
    const unsigned incDimSizeAtomic = incDimSize / m_atomicUnitLength;  // Size of included dim in atomic units

    // Define first steps
    m_unitsNrPerFirstStep = maxFitNr;
    MME_ASSERT(m_unitsNrPerFirstStep != 0, "Wrong first step calculation");
    m_firstStepsNr = incDimSizeAtomic / m_unitsNrPerFirstStep;
    // Define last step
    const unsigned firstStepsLength = m_firstStepsNr * m_unitsNrPerFirstStep;
    MME_ASSERT(incDimSizeAtomic >= firstStepsLength, "Wrong units destribution for 1st step");
    m_unitsNrPerLastStep = incDimSizeAtomic - firstStepsLength;
    MME_ASSERT(m_unitsNrPerLastStep < m_unitsNrPerFirstStep, "First steps must be bigger than last - by definition");
    // First steps can fully cover the first weight in this case there will not be any last step
    m_lastStepsNr = (m_unitsNrPerLastStep == 0) ? 0 : 1;
}

// Define multi-step grid when no dim is fully included in SB.
// At the end we guarantee (m_unitsNrPerLastStep <= m_unitsNrPerFirstStep) and (m_lastStepsNr = 1).
// Note: <= not < because we force having only one last step.
void CommonDimGrid::splitNoDimIncluded(unsigned maxFitNr)
{
    // Calc total steps in first common dim.
    // For the SP case we need to use full size since we squash common dim into BWH.
    const unsigned firstCommonDimLength = (m_gridType == SP_GRID) ? calcViewSize() : m_sizes[getFirstCommonDim()];
    const unsigned maxUnitsPerStepLength = m_atomicUnitLength * maxFitNr;
    MME_ASSERT(maxUnitsPerStepLength != 0, "Wrong capacity calculation");
    const unsigned totalStepsNr = div_round_up(firstCommonDimLength, maxUnitsPerStepLength);
    // If this assert hits, it's is an indication of visiting the wrong flow (go to splitAtLeastOneDimIncluded(...))
    MME_ASSERT(totalStepsNr >= 2, "Using no-dim-included flow instead of at-least-one-dim-included");
    // Define first step
    m_unitsNrPerFirstStep = maxFitNr;
    m_firstStepsNr = totalStepsNr - 1;  // The 1 belongs to last step
    // Define last step
    const unsigned firstStepsLength = m_firstStepsNr * m_unitsNrPerFirstStep * m_atomicUnitLength;
    MME_ASSERT(firstStepsLength < firstCommonDimLength, "Wrong 1st step calculation");
    const unsigned lastStepLength = firstCommonDimLength - firstStepsLength;
    m_unitsNrPerLastStep = div_round_up(lastStepLength, m_atomicUnitLength);
    // In this flow we expect at least one atomic unit in last step (hence the assert) and exactly one last step,
    // because otherwise all steps are first, a case which is handled in other flow (splitAtLeastOneDimIncluded(...))
    MME_ASSERT(m_unitsNrPerLastStep > 0 && m_unitsNrPerLastStep <= m_unitsNrPerFirstStep,
               "Wrong units destribution calculation");
    m_lastStepsNr = 1;

    // Initialize partial unit size. Misalignment is possible
    m_partialUnitLength = firstCommonDimLength % m_atomicUnitLength;
}

// Calculate the size of CommonDim grid (m_gridSize), that's equal to total number of sub-views.
void CommonDimGrid::calcGridSize()
{
    MME_ASSERT(m_gridSize == 0, "grid size is already calculated");  // We expect to invoke this method only onetime

    // Calc number of repetitions
    unsigned repetitionsNr = 1;
    // For SP the is only one repetition since we squash common dim into BWH
    if (m_gridType != SP_GRID)
    {
        const unsigned startDim = std::max(getSplitDim() + 1, 2u);
        for (unsigned i = startDim; i < MME_MAX_TENSOR_DIMS; i++)
        {
            if (m_sizes[i] != 0)  // There are DEDX cases where dim is zero
            {
                repetitionsNr *= m_sizes[i];
            }
        }
    }

    const unsigned templateSize = m_firstStepsNr + m_lastStepsNr;
    m_gridSize = templateSize * repetitionsNr;
    MME_ASSERT(m_gridSize != 0, "Wrong grid size calculation");
}

// Returns true is any type of SB-reuse is adapted
bool CommonDimGrid::isPartialReuse() const
{
    return ((m_reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED) ||
            (m_reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY) ||
            (m_reuseType == SB_PARTIAL_REUSE_NO_DIM_INCLUDED));
}

bool CommonDimGrid::isPartialToMemory() const
{
    return m_reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY;
}

// Return 'true' is there should be a split into multiple sub-views.
// Return 'false' on a single sub-view case.
bool CommonDimGrid::isMultiStep() const
{
    if (!GridBase::isMultiStep())
    {
        return false;
    }
    if (m_multipleCdCuts)
    {
        return true;
    }
    if (!isPartialReuse())
    {
        if (m_gridType == BATCH_GRID)
        {
            return (m_gridSize != 1);
        }
        MME_ASSERT(m_gridSize == 1, "Didn't expect multiple steps");
        return false;
    }
    return true;
}

// Return number of partials, for BGEMM it considers first batch only
unsigned CommonDimGrid::getPartialsNr() const
{
    if (m_gridType == BATCH_GRID)
    {
        return calcStepsNrPerTemplate();
    }
    return m_gridSize;
}

// detect if at least one dim is fully included in SB
bool CommonDimGrid::isAtLeastOneDimIncluded() const
{
    return (m_reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED ||
            m_reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY);
}

// detect if all dims are fully included in SB, i.e. no split
bool CommonDimGrid::isAllDimsIncluded() const
{
    return (m_reuseType == SB_PARTIAL_REUSE_ALL_DIMS_INCLUDED);
}

// Common dim is one of the first two that is not FCD. It's different than 'm_splitDim' of 'GridBase' class
unsigned CommonDimGrid::getFirstCommonDim() const
{
    return (1 - m_fcdDim);
}

// If at least one dim is fully included in SB then return the index of outer dim that contains it.
// Otherwise return the index of common dim.
unsigned CommonDimGrid::calcSplitDim(unsigned lastIncludedDim) const
{
    unsigned splitDim = 0;

    if (isAtLeastOneDimIncluded())
    {
        splitDim = lastIncludedDim + 1;
        if (splitDim == m_fcdDim)
        {
            splitDim++;
        }
    }
    else if (lastIncludedDim == (MME_MAX_TENSOR_DIMS - 1))  // All tensor is included
    {
        splitDim = lastIncludedDim;
    }
    else if (lastIncludedDim == MME_MAX_TENSOR_DIMS)
    {
        splitDim = getFirstCommonDim();
    }
    else if (m_constants.params.isGemmOperation() || m_constants.params.isDmaOperation())
    {
        // in Bgemm and DMA we can slice on the batche without triggering partials
        splitDim = lastIncludedDim;
    }
    else
    {
        // There are DEDX cases where dim is zero. Currently it's the only case
        MME_ASSERT(lastIncludedDim <= (MME_MAX_TENSOR_DIMS - 2), "Invalid last dim input");
        MME_ASSERT(m_sizes[lastIncludedDim + 1] == 0, "Expected dim size zero");
        splitDim = lastIncludedDim;
    }
    MME_ASSERT(splitDim < MME_MAX_TENSOR_DIMS, "Wrong dim split calculation");
    MME_ASSERT(lastIncludedDim == MME_MAX_TENSOR_DIMS || m_sizes[lastIncludedDim] != 0, "Invalid dims");
    return splitDim;
}

// Last step (of the template) is special because it can have unaligned size which is divided by atomic unit size with
// reminder. Return 'true' if current step is partial. 'false' otherwise.
bool CommonDimGrid::isPartialStep(unsigned stepIdxInTemplate) const
{
    if ((m_reuseType == SB_PARTIAL_REUSE_NO_DIM_INCLUDED) && (stepIdxInTemplate == (calcStepsNrPerTemplate() - 1)))
    {
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeGrids
////////////////////////////////////////////////////////////////////////////////////////////////////

// Return FCD grid
Grid& RecipeGrids::getFcdGrid()
{
    return getGrid(FCD_GRID, false);
}

const Grid& RecipeGrids::getFcdGrid() const
{
    return const_cast<RecipeGrids*>(this)->getGrid(FCD_GRID, false);
}

// Return SP grid (SP is relative to output)
Grid& RecipeGrids::getSpGrid()
{
    Grid& spGrid = getGrid(FCD_GRID, true);
    MME_ASSERT((spGrid.getType() == SP_GRID) || (spGrid.getType() == CONV_GRID), "Invalid grid type");
    return spGrid;
}

const Grid& RecipeGrids::getSpGrid() const
{
    return const_cast<RecipeGrids*>(this)->getSpGrid();
}

// If 'otherGrid' is 'true': return given grid type.
// If 'otherGrid' is 'false': return the other grid its type not equal to given grid type.
Grid& RecipeGrids::getGrid(GridType gridType, bool otherGrid)
{
    MME_ASSERT(m_grids.size() == 2, "Invalid grids");
    MME_ASSERT(m_grids[0].getType() != m_grids[1].getType(), "Only different grid types are allowed");
    if ((m_grids[0].getType() == gridType) && !otherGrid)
    {
        return m_grids[0];
    }
    MME_ASSERT((m_grids[1].getType() == gridType) || otherGrid, "Invalid grids types");
    return m_grids[1];
}

CommonDimGrid& RecipeGrids::getCommonDimGrid(unsigned idx)
{
    MME_ASSERT(m_commonDimGridVec.size() > idx, "Invalid common dim grid index");
    return m_commonDimGridVec[idx];
}

const CommonDimGrid& RecipeGrids::getCommonDimGrid(unsigned idx) const
{
    return const_cast<RecipeGrids*>(this)->getCommonDimGrid(idx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeGridsCreator
////////////////////////////////////////////////////////////////////////////////////////////////////

RecipeGridsCreator::GridsParams::GridsParams(unsigned fcdCapacity, unsigned spCapacity, unsigned numCdCuts)
: fcdStepCapacity(fcdCapacity), spStepCapacity(spCapacity)
{
    commonDimParamsVec.resize(numCdCuts);
}

RecipeGridsCreator::RecipeGridsCreator(const RecipeConstants& constants, const MmeRecipe& recipe)
: RecipeGrids(constants), m_recipe(recipe)
{
}

int RecipeGridsCreator::getCommonDimCuts(std::array<MmeTensorView, 2>& splitView, int tensorIdx)
{
    const auto& params = m_constants.params;
    unsigned cdCutPoint = RecurringMisalignmentOptimization::getCutPointPerSubProblem(params,
                                                                                      m_constants.geoAttr,
                                                                                      m_constants.mmeHalReader);

    // The cut point must meet the data type alignment requirement
    unsigned alignedCdCutPoint = calcCommonDimAlignment(m_recipe,
                                                        params,
                                                        m_constants.geoAttr,
                                                        m_constants.mmeHalReader,
                                                        cdCutPoint,
                                                        false,
                                                        false);
    if (alignedCdCutPoint != cdCutPoint)
    {
        cdCutPoint = 0;  // Cannot create a partial at this point
    }

    if (cdCutPoint == 0)  // no cut, recipeVec has one element
    {
        splitView[0] =
            (params.opType == e_mme_gemm_transpose) ? m_recipe.aViews[tensorIdx] : m_recipe.bViews[tensorIdx];
        return 1;
    }

    // CD is split into two sections: prefix and remainder
    unsigned bCD = isTransposed(params.opType, e_mme_in_b) ? 0 : 1;
    const MmeTensorView& bView = m_recipe.bViews[tensorIdx];
    MME_ASSERT(cdCutPoint < bView.sizes[bCD], "Wrong cut point calculation");
    splitView[0] = bView;
    splitView[0].bases[bCD] = 0;
    splitView[0].sizes[bCD] = cdCutPoint;
    splitView[1] = bView;
    splitView[1].bases[bCD] = cdCutPoint;
    splitView[1].sizes[bCD] -= cdCutPoint;
    return 2;
}
// Initialize grid classes by determining base and size of sub-views.
// Important: Use recipe tensors A/B/C only, don't take any tensor info from params,
//            because lowering updates recipe tensors but not params.
void RecipeGridsCreator::init()
{
    // FCD - use only output tensor. It manifests in C[0]
    // unaffected by CD splits as both tensors must have the same sp and FCD sizes, simply use one of them.
    initGrid(FCD_GRID,
             m_recipe.cViews.front().bases[0],
             m_recipe.cViews.front().sizes[0],
             m_recipe.cViews.front().sizes[0]);

    if (m_constants.params.isDedwOperation())
    {
        m_numCommonDimCuts.push_back(1);  // no cd cuts
        // Spatial - SP in DEDW is the common dim of B tensor
        auto bBases = m_recipe.bViews.front().bases;
        auto bOrigSizes = m_recipe.bViews.front().sizes;
        auto bSizes = m_recipe.bRoiSizes.front();
        // 0 is equivalent dim to FCD (of output). Clear it as it's not participate in common dim
        static const unsigned fcdDim = 0;
        bBases[fcdDim] = bSizes[fcdDim] = 0;  // Clear FCD
        initCommonDimGrid(SP_GRID, bBases, bSizes, bOrigSizes, fcdDim, false);

        // Convolution - dim 1 of C tensor
        static const unsigned convDim = 1;
        auto cBases = m_recipe.cViews.front().bases;
        auto cSizes = m_recipe.cViews.front().sizes;
        cBases[0] = cSizes[0] = 0;  // Clear FCD
        unsigned concurrentDim = m_constants.geoAttr.getConcurrentDim();
        cSizes[concurrentDim] = div_round_up(cSizes[concurrentDim], m_constants.geoAttr.getGeometryConcurrency());
        initGrid(CONV_GRID, cBases, cSizes, cSizes, convDim);
    }
    else if (m_constants.params.isNativeDmaOperation())
    {
        m_numCommonDimCuts.push_back(1);  // no cd cuts

        initGrid(SP_GRID,
                 m_recipe.cViews.front().bases[1],
                 m_recipe.cViews.front().sizes[1],
                 m_recipe.cViews.front().sizes[1]);

        // initialize the dma "batch" dimensions
        SizeArray sizes = {0}, bases = {0};
        sizes[1] = 1;
        sizes[2] = m_recipe.cViews.front().sizes[2];
        sizes[3] = m_recipe.cViews.front().sizes[3];
        sizes[4] = m_recipe.cViews.front().sizes[4];

        bases[2] = m_recipe.cViews.front().bases[2];
        bases[3] = m_recipe.cViews.front().bases[3];
        bases[4] = m_recipe.cViews.front().bases[4];
        initCommonDimGrid(BATCH_GRID, bases, sizes, sizes, 0, false);
    }
    else  // All other operations:
    {
        const auto gridType = m_constants.params.isGemmOperation() ? BATCH_GRID : CONV_GRID;
        // Spatial
        unsigned spBase, spSize, spSizeOrig;
        if (gridType == BATCH_GRID)
        {
            spBase = m_recipe.cViews.front().bases[1];
            // Size is equal to C[1]
            spSize = m_recipe.cRoiSizes.front()[1];
            spSizeOrig = m_recipe.cViews.front().sizes[1];
        }
        else
        {
            spBase = m_constants.params.spBase;
            // Size is equal to BHW of C tensor
            spSize = multiplyElements(m_recipe.cRoiSizes.front().begin() + 1, m_recipe.cRoiSizes.front().end());
            spSizeOrig =
                multiplyElements(m_recipe.cViews.front().sizes.begin() + 1, m_recipe.cViews.front().sizes.end());
        }
        initGrid(SP_GRID, spBase, spSize, spSizeOrig);

        // Create the commonDimGrid for each CD cut
        for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
        {
            std::array<MmeTensorView, 2> splitView;
            m_numCommonDimCuts.push_back(getCommonDimCuts(splitView, gemm));
            for (int cdCut = 0; cdCut < getNumCdCuts(gemm); cdCut++)
            {
                // Convolution/Batch - CONV/BATCH is mapped to common dim of B tensor
                auto bases = splitView[cdCut].bases;
                auto sizes = splitView[cdCut].sizes;

                // In order to support broadcast(on A/B) the batch sizes is taken from cView(fixed size)
                if (gridType == BATCH_GRID)
                {
                    sizes[2] = m_recipe.cViews[gemm].sizes[2];
                    sizes[3] = m_recipe.cViews[gemm].sizes[3];
                    sizes[4] = m_recipe.cViews[gemm].sizes[4];

                    bases[2] = m_recipe.cViews[gemm].bases[2];
                    bases[3] = m_recipe.cViews[gemm].bases[3];
                    bases[4] = m_recipe.cViews[gemm].bases[4];
                }
                // Determine which dim is equivalent to FCD (of output)
                const unsigned fcdDim = isTransposed(m_constants.params.opType, e_mme_in_b) ? 1 : 0;
                // Clear FCD as it does not participate in common dim
                bases[fcdDim] = 0;
                sizes[fcdDim] = 0;
                initCommonDimGrid(gridType, bases, sizes, sizes, fcdDim, getNumCdCuts(gemm) > 1);
            }
        }
    }
}

void RecipeGridsCreator::create(unsigned fcdStepCapacity, unsigned spStepCapacity)
{
    unsigned cdGrids = 0;
    for (unsigned gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
    {
        cdGrids += getNumCdCuts(gemm);
    }
    GridsParams recipeGridsParams(fcdStepCapacity, spStepCapacity, cdGrids);
    defineGrids(recipeGridsParams);
    createGrids(recipeGridsParams);
}

void RecipeGridsCreator::initGrid(GridType gridType,
                                  const SizeArray& bases,
                                  const SizeArray& sizes,
                                  const SizeArray& origSizes,
                                  unsigned splitDim)
{
    if (!m_grids.empty())
    {
        MME_ASSERT(m_grids.size() == 1, "Invalid grids");
        MME_ASSERT(m_grids[0].getType() != gridType, "Invalid grid type");
    }
    m_grids.emplace_back(gridType, bases, sizes, origSizes, m_constants, splitDim);
}

// A forward for the one above
void RecipeGridsCreator::initGrid(GridType gridType, unsigned base, unsigned size, unsigned origSize)
{
    MME_ASSERT(size != 0, "Invalid size input");
    initGrid(gridType, uint2SizeArr(base, 0), uint2SizeArr(size, 1), uint2SizeArr(origSize, 1), 0);
}

void RecipeGridsCreator::initCommonDimGrid(GridType gridType,
                                           const SizeArray& bases,
                                           const SizeArray& sizes,
                                           const SizeArray& origSizes,
                                           unsigned fcdDim,
                                           bool isMultipleCdCuts)
{
    m_commonDimGridVec.emplace_back(gridType, bases, sizes, origSizes, m_constants, fcdDim, isMultipleCdCuts);
}

// Define all grids to be prepared for creation.
// 'fcdStepCapacity' and 'm_spGeoPerReuseNr' are the spatial results. Assume they are initialized properly.
void RecipeGridsCreator::defineGrids(GridsParams& recipeGridsParams) const
{
    if (m_recipe.reuseType() == e_mme_no_reuse)
    {
        // Define FCD/spatial grids. Start with no limits.
        recipeGridsParams.fcdStepCapacity = recipeGridsParams.spStepCapacity = -1U;
        defineSpatialGrids(recipeGridsParams.fcdStepCapacity, recipeGridsParams.spStepCapacity);

        // Define common dim grid
        for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
        {
            for (int cdCut = 0; cdCut < getNumCdCuts(gemm); cdCut++)
            {
                int cdIdx = gemm * getNumCdCuts(gemm) + cdCut;
                defineCommonDimGrid(recipeGridsParams.commonDimParamsVec[cdIdx], SB_NO_REUSE);
            }
        }
    }
    else  // SB reuse - partial and non-partial
    {
        // Check FCD/spatial limits validity and make initial definition
        MME_ASSERT((recipeGridsParams.fcdStepCapacity != 0) && (recipeGridsParams.spStepCapacity != 0),
                   "Invalid capacities input");
        defineSpatialGrids(recipeGridsParams.fcdStepCapacity, recipeGridsParams.spStepCapacity);

        for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
        {
            for (int cdCut = 0; cdCut < getNumCdCuts(gemm); cdCut++)
            {
                int cdIdx = gemm * getNumCdCuts(gemm) + cdCut;
                // Define common dim grid
                SBReuse sbReuse(m_constants, m_recipe, *this, cdIdx);
                // Multiple CD cuts imply Partials
                if (sbReuse.isPartialSBReuse())
                {
                    sbReuse.defineCommonDimGridForPartialReuse(recipeGridsParams.commonDimParamsVec[cdIdx],
                                                               recipeGridsParams.spStepCapacity);
                    sbReuse.defineSpatialGridsForPartialSBReuse(
                        recipeGridsParams.fcdStepCapacity,
                        recipeGridsParams.spStepCapacity,
                        recipeGridsParams.commonDimParamsVec[cdIdx].partialToMemory);
                }
                else
                {
                    MME_ASSERT(getNumCdCuts(gemm) == 1, "Invalid number of CD cuts");
                    defineCommonDimGrid(recipeGridsParams.commonDimParamsVec[cdIdx], SB_NON_PARTIAL_REUSE);
                    sbReuse.defineSpatialGridsForNonPartialSBReuse(recipeGridsParams.fcdStepCapacity,
                                                                   recipeGridsParams.spStepCapacity);
                }
            }
        }
    }
}

// Constrain FCD and SP grids
void RecipeGridsCreator::defineSpatialGrids(unsigned& fcdStepCapacity, unsigned& spStepCapacity) const
{
    // Outer and conv loops are one-byte length
    MME_ASSERT(m_constants.mmeHalReader.getSizeOfOuterLoopSizeMinus1() == 1,
               "Outer loop counter size is expected to be one byte");
    MME_ASSERT(m_constants.mmeHalReader.getSizeOfKernelSizeDim0() == 1,
               "Conv loop counter size is expected to be one byte");
    // Tetris loop is 4 bytes reg. There is no constraint for it
    MME_ASSERT((m_constants.mmeHalReader.getSizeOfDescNumIterMinus1() == sizeof(unsigned)),
               "Unsupported numIterationsMinus1");

    static const unsigned one_byte_max_loops_nr = (1 << 8);
    const EMmePattern pattern = m_constants.params.strategy.pattern;
    // Constrain FCD
    const EMmeLoopMask fcdLoopMask = pattern2LoopMask(pattern, EMmeLoopDim::dim_k);
    MME_ASSERT(fcdLoopMask != e_mme_tetris_loop, "Wrong FCD loop calculation");
    fcdStepCapacity = std::min(fcdStepCapacity, one_byte_max_loops_nr);
    // Constrain SP - exclude operations that use tetris loop for spatial
    const bool usesTetrisLoop = m_constants.params.isFwdOrDedx();
    if (!usesTetrisLoop)
    {
        const EMmeLoopMask spLoopMask = pattern2LoopMask(pattern, EMmeLoopDim::dim_c);
        MME_ASSERT(spLoopMask != e_mme_tetris_loop, "Wrong spatial loop calculation");
        spStepCapacity = std::min(spStepCapacity, one_byte_max_loops_nr);
    }
}

// Define common dim grid for no SB reuse and non-partial SB reuse
void RecipeGridsCreator::defineCommonDimGrid(CommonDimParams& commonDimParams, SBReuseType reuseType) const
{
    MME_ASSERT((reuseType == SB_NO_REUSE) || (reuseType == SB_NON_PARTIAL_REUSE), "Invalid reuse type input");
    commonDimParams.reuseType = reuseType;
    commonDimParams.atomicUnitLength = 1;  // Unit of high dims

    // Calc last included dim and max fit

    commonDimParams.maxFitNr = 0;
    commonDimParams.forcedPartial = m_recipe.m_gemmNr > 1;
    static const unsigned c_outer_max_loops_nr = (1 << 8);
    if (m_constants.params.isGemmOperation() || m_constants.params.isDmaOperation())
    {
        unsigned batch;
        MME_ASSERT((m_constants.mmeHalReader.getSizeOfOuterLoopSizeMinus1() == 1),
                   "Outer loop counter size is expected to be one bytes");

        for (batch = GEMM_DIM_B1; batch < GEMM_DIM_B3; batch++)
        {
            unsigned loopSize = c_outer_max_loops_nr;
            unsigned batchSize = m_recipe.cViews.front().sizes[batch];
            if (batch == m_constants.geoAttr.getConcurrentDim())
                loopSize *= m_constants.geoAttr.getGeometryConcurrency();
            if (batchSize > loopSize)
            {
                //  found a dimension that doesnt fit, break into descriptor on this dim.
                break;
            }
            if (commonDimParams.forcedPartial)
            {
                //  in partial mode we currently dont allow batch movement.
                //  once batch movement is accounted for remove this logic
                break;
            }
        }
        commonDimParams.lastIncludedDim = batch;
        if (commonDimParams.forcedPartial)
        {
            //  forcing partial even though there is no reuse - make sure the recipe will split on the first batch dim
            commonDimParams.maxFitNr = 1;
        }
        else
        {
            // In order to support broadcast(on A/B) the batch sizes is taken from cView(fixed size)
            commonDimParams.maxFitNr = m_recipe.cViews.front().sizes[commonDimParams.lastIncludedDim];
        }
        // BGEMM is restricted by last batch dim which uses outer loop
        unsigned loopSize = c_outer_max_loops_nr;
        if (commonDimParams.lastIncludedDim == m_constants.geoAttr.getConcurrentDim())
            loopSize *= m_constants.geoAttr.getGeometryConcurrency();
        commonDimParams.maxFitNr = std::min(commonDimParams.maxFitNr, loopSize);
    }
    else
    {
        const SizeArray& bSizes = m_recipe.bViews.front().sizes;
        const int fcd = getCommonDimGrid(0).getFcdDim();
        // Start by including all tensor in a single sub-view
        for (int dim = bSizes.size() - 1; dim >= 0; dim--)
        {
            // There are DEDX cases where dim is zero. Currently it's the only case
            if ((dim != fcd) && (bSizes[dim] != 0))
            {
                commonDimParams.lastIncludedDim = (unsigned) dim;
                commonDimParams.maxFitNr = bSizes[dim];
                break;
            }
        }
        MME_ASSERT(commonDimParams.maxFitNr != 0, "Invalid common dim creation params");
    }
}

// Create all grids - FCD, SP, common
void RecipeGridsCreator::createGrids(const GridsParams& recipeGridsParams)
{
    // Flag to check if grid don't have multiple GEMMs to be executed in same EU.
    // sbReuse is not possible in H6 8x batch concurrency.
    // In multi steps is possible.
    const bool isMultiStepPossible =
        (m_constants.geoAttr.getCoreConcurrency() == 1 || m_constants.params.isDedwOperation());

    // Create FCD grid
    auto& fcdGrid = getFcdGrid();
    unsigned fcdStepSize;  // how many FCD elements can be read by a single geometry
    if (m_constants.params.opType == e_mme_memcpy)
    {
        fcdStepSize = -1;  // 32 bit field
    }
    else
    {
        fcdStepSize = m_constants.geoAttr.getGeometryWidth();
    }
    fcdGrid.create(fcdStepSize, recipeGridsParams.fcdStepCapacity);
    MME_ASSERT(isMultiStepPossible || !fcdGrid.isMultiStep(), "Invalid FCD multi-step grid");

    // Create SP grid
    auto& spGrid = getSpGrid();
    // Generally the mme descriptor can generate the amount of output pixels described by geoTotalElemHeight.
    // but, when A is transposed the input is interleaved on the first spatial dimension.
    // If this dimension is smaller than the amount of ports, the extra ports will not be used.
    // So even though geoTotalElemHeight expresses the geometry of the output it fails to take into account our
    // ability to utilize it.
    // The output is limited to the amount of A ports we can utilize.
    unsigned availableSpSize = m_constants.geoAttr.getGeometryHeight();
    if (m_recipe.acceleratedOperand == e_mme_op_c) availableSpSize >>= m_recipe.teAcceleration;
    unsigned interleavingSpDim = m_recipe.cViews.front().sizes[1]; // need to change back to getSpInterleavingDim [SW-169571]
    if (interleavingSpDim < m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a))
    {
        availableSpSize /= m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a);
        availableSpSize *= interleavingSpDim;

        // Make sure no mistakes were made
        MME_ASSERT(availableSpSize <= m_constants.geoAttr.getGeometryHeight(), "Wrong spatial size calcualtion");
    }
    spGrid.create(availableSpSize, recipeGridsParams.spStepCapacity);
    MME_ASSERT(isMultiStepPossible || !spGrid.isMultiStep(), "Invalid spatial multi-step grid");

    for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
    {
        for (int cdCut = 0; cdCut < getNumCdCuts(gemm); cdCut++)
        {
            // Create common dim grid
            //  TODO common dim also needs to be aligned to the number of interleaved ports that reads it
            int cdIdx = gemm * getNumCdCuts(gemm) + cdCut;
            auto& commonGrid = getCommonDimGrid(cdIdx);
            commonGrid.create(recipeGridsParams.commonDimParamsVec[cdIdx]);

            // If any of the input tensors is transposedand we have partial splits, and the aplit is on the
            // channel dimension (dim0) then we want to ensure that each partial starts at aligned address.
            // Or otherwise partials may start in unaligned addresses with significant impact on run time
            bool aIsTransposed = isTransposed(m_constants.params.opType, e_mme_in_a);
            bool bIsTransposed = isTransposed(m_constants.params.opType, e_mme_in_b);
            if ((aIsTransposed || bIsTransposed) &&  // At least one input tensor is transposed
                (commonGrid.getGridSize() > 1) &&  // Common grid is split to partials
                (commonGrid.getSizes()[commonGrid.getFirstCdDim()] !=
                 m_recipe.bViews[gemm].sizes[bIsTransposed ? 0 : 1]) &&  // make sure we actually have a partial
                (commonGrid.getSplitDim() == commonGrid.getFirstCdDim()))  // Split is done over the channel dim
            {
                MME_ASSERT(
                    commonGrid.calcStepSize(0) % (m_constants.mmeHalReader.getClSize() / m_recipe.getElemSize()) == 0,
                    "Partial step must be aligned to CL");
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeGenerator
////////////////////////////////////////////////////////////////////////////////////////////////////

RecipeGenerator::RecipeGenerator(RecipeType recipeType,
                                 const MmeLayerParams& params,
                                 const MmeHalReader& mmeHalReader,
                                 const CommonGeoAttr& geoAttr
#if GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE
                                 ,
                                 const bool ignoreSbReuse
#endif
                                 )
: m_constants(params, geoAttr, mmeHalReader),
  m_recipe(recipeType, params.opType),
  m_grids(m_constants, m_recipe)
#if GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE
  ,
  m_ignoreSbReuse(ignoreSbReuse)
#endif
{
    init();
}

// Main public method to create MME recipe
const MmeRecipe& RecipeGenerator::generateRecipe()
{
    LOG_TRACE(MME_RECIPE, "generating recipe for op {}", get().getOpType());
    // Define and create all grids
    m_grids.create(m_fcdGeoPerReuseNr, m_spGeoPerReuseNr);

    // Force no-reuse - for debug purpose. Must be done after defining the grids.
#if FORCE_NO_SB_REUSE_BUT_KEEP_SPLIT_RESULT
    m_recipe.reuseInfo.clear();
#endif

    // Possible extension of SP grids according to GC pipeline level. The logic can affect spatial or FCD grids
    // but not common dim grid.
#if ENABLE_PIPELINE_LEVEL_HINT
    handlePipelineLevelHint();
#endif

    // Create recipe sub-views.
    // After this point it's not possible to update grids and expect it to be reflected on subviews.
    SubViewSplitter(m_recipe, m_grids).split();

    // Create RecipeIterator. Must be done after determining subviews.
    m_recipe.createIterator(
        m_constants.params.opType
#if GAUDI_DEBUG_DISABLE_REVERSED_PARTIALS
        // If gaudi, don't allow revered order when walk over splits
        ,
        ((m_constants.mmeHalReader.getChipType() == ChipType::e_mme_Gaudi) && (!m_constants.params.isDedwOperation()))
            ? false
            : true
#endif
    );

    // 2D SB reuse, try to reuse the second operand. It must be done after grid creation and split.
    // Pipeline modifications are done on SP and FCD, thus 2D reuse is not dependant on it.
#if ENABLE_2D_SB_REUSE
    handle2dSBReuse();
#endif

    // set amount of signals required by each recipe
    setSignalAmount();
    LOG_TRACE(MME_RECIPE, "finished generating recipe for op {}", get().getOpType());
    LOG_TRACE(MME_RECIPE, "Recipe info:");
    auto debugInfo = get().getRecipeDebugInfo(true);
    for (auto& info : debugInfo)
    {
        LOG_TRACE(MME_RECIPE, "{}", info);
    }
    return m_recipe;
}

// Public method to return 'true' for partial-SB-reuse, 'false' sor non-partial-SB-reuse or no-SB-reuse.
bool RecipeGenerator::isPartialSBReuse()
{
    //  DMA operations dont support sb reuse yet so they fail to initialize cdCuts array, skip for now.
    if (!m_constants.params.isNativeDmaOperation())
    {
        for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
        {
            if (m_grids.getNumCdCuts(gemm) > 1)
            {
                return true;
            }
        }
    }
    bool isPartialSBReuse = false;
    if (m_recipe.reuseType() != e_mme_no_reuse)
    {
        SBReuse sbReuse(m_constants, m_recipe, m_grids, 0);
        isPartialSBReuse = sbReuse.isPartialSBReuse();
    }
    return isPartialSBReuse;
}

// Main init method
void RecipeGenerator::init()
{
    LOG_TRACE(MME_RECIPE, "Initializing new MME recipe for op: {}", get().getOpType());
    m_recipe.raster = m_constants.params.isPatternRaster();
    m_recipe.geometry = m_constants.params.getGeometry();
    //  initially set the recipe views to be the full tensors - they might be split later
    m_recipe.aViews.push_back(m_constants.params.getOperand(e_mme_op_a));
    m_recipe.bViews.push_back(m_constants.params.getOperand(e_mme_op_b));
    m_recipe.cViews.push_back(m_constants.params.getOperand(e_mme_op_c));

    // Handle lowering
    if (m_constants.params.canLower())
    {
        m_recipe.lowering = true;
        applyLowering();
    }
    // Handle flattening
    if (m_constants.params.canFlatten())
    {
        applyFlattening();
    }
    // Handle TE acceleration
    if (calcTeAcceleration())
    {
        applyTeAcceleration();
    }

    setGemmNr();
    for (int splitIdx = 0; splitIdx < m_recipe.m_gemmNr; splitIdx++)
    {
        recipeSetRoiSize(splitIdx);
    }

    calcFcdGeoContinuousLength();
    calcSpGeoContinuousLength();
    calcReuseOperand();
    calcSbUtilization();
    m_grids.init();
}

void RecipeGenerator::setGemmNr()
{
    auto& params = m_constants.params;
    if (params.strategy.maskedBgemm)
    {
        // push add the auxiliary tensors
        m_recipe.aViews.push_back(m_constants.params.xAux);
        m_recipe.bViews.push_back(m_constants.params.wAux);
        m_recipe.cViews.push_back(m_constants.params.yAux);

        m_recipe.m_gemmNr = 2;
        m_recipe.maskedBgemm = true;
        m_recipe.concurrentDim = m_constants.geoAttr.getConcurrentDim();
        m_recipe.concurrency = m_constants.geoAttr.getGeometryConcurrency();
    }
    else
    {
        m_recipe.m_gemmNr = 1;
    }
}

EMmeInternalOperand RecipeGenerator::getTeAcceleratedOperand() const
{
    unsigned fcd = m_recipe.getOperand(e_mme_op_a).sizes[0];
    unsigned sp = m_recipe.getOperand(e_mme_op_a).sizes[1];
    if (fcd < sp) return e_mme_op_a;
    return e_mme_op_c;
}

unsigned RecipeGenerator::getTeAcceleration(EMmeInternalOperand operand) const
{
    unsigned fcd = m_recipe.getOperand(operand).sizes[0];
    unsigned TEx =
        m_constants.mmeHalReader.getClSize() / (fcd * getElementSize(m_recipe.getOperand(operand).elementType));
    //  round to a power of 2;
    if (TEx < 2) return 0;
    if (TEx < 4) return 1;
    if (TEx < 8) return 2;
    else
        return 3;
}
bool RecipeGenerator::calcTeAcceleration() const
{
    if (m_constants.params.opType != e_mme_trans || !m_constants.params.strategy.teAccelerationEn) return false;

    EMmeInternalOperand acceleratedOperand = getTeAcceleratedOperand();
    const MmeTensorView& tensor = m_recipe.getOperand(acceleratedOperand);

    //  cant accelerate output fp8 type
    if (acceleratedOperand == e_mme_op_c && isTypeFp8(m_recipe.getOperand(acceleratedOperand).elementType)) return 0;
    //  since 1 byte inputs have to be padded to 2 accelerating in that scenario is not possible
    if (acceleratedOperand == e_mme_op_a && isTypeFp8(tensor.elementType) && tensor.sizes[DIM_C] % 2) return false;
    //  tensor has to be contiguous to be accelerated
    if (tensor.sizes[0] != tensor.strides[1]) return false;

    // make sure we actually have any acceleration to do
    unsigned TEx = getTeAcceleration(acceleratedOperand);
    if (TEx == 0) return false;

    //  accelerating unaligned sizes is currently not supported
    //  in case of an unaliged size the recipe should split to a main accelerated desc and a reminder desc.
    if (tensor.sizes[DIM_W] % (1 << TEx) != 0) return false;

    //  no issue, accelerate away
    return true;
}

void RecipeGenerator::applyTeAcceleration()
{
    EMmeInternalOperand acceleratedOperand = getTeAcceleratedOperand();
    unsigned TEx = getTeAcceleration(acceleratedOperand);
    MmeTensorView& tensor = m_recipe.getOperand(acceleratedOperand);

    m_recipe.acceleratedOperand = acceleratedOperand;
    m_recipe.teAcceleration = TEx;
    tensor.sizes[0] <<= TEx;
    tensor.strides[1] <<= TEx;
    tensor.sizes[1] >>= TEx;

    LOG_TRACE(MME_RECIPE, "Applying TE acceleration, accelerated operand : {}, TEx : {}, new accelerated operand shape: sizes - [{}], strides - [{}] ",
              m_recipe.acceleratedOperand, m_recipe.teAcceleration,
              arrayToStr(tensor.sizes.begin(), tensor.sizes.end()),
              arrayToStr(tensor.strides.begin(), tensor.strides.end()) );
}

void RecipeGenerator::applyLowering()
{
    MmeTensorView* wTensorView = nullptr;
    switch (m_constants.params.opType)
    {
        case e_mme_fwd:
        case e_mme_transposed_dedx:
            wTensorView = &m_recipe.bViews.front();
            break;
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            wTensorView = &m_recipe.cViews.front();
            break;
        default:
            MME_ASSERT(0, "Operation is not supported");
    }

    m_recipe.aViews.front().sizes[0] *= wTensorView->sizes[2];

    // Lowering must be applied to tensor A.
    // We may optionally align the Weights tensor to reflect the change in A.
    // The legacy code of gaudi does not do the alignment, while gaudi2 does the alignhment
#if GAUDI_DEBUG_APPLY_LOWERING_IN_FWD_AND_DEDX_TO_A_ONLY
    // If gaudi, don't update w tensor
    ChipType chipType = m_constants.mmeHalReader.getChipType();
    if ((chipType == ChipType::e_mme_Gaudi) &&
        ((m_constants.params.opType == e_mme_fwd) || (m_constants.params.opType == e_mme_dedx)))
    {
        return;
    }
#endif
    wTensorView->sizes[1] *= wTensorView->sizes[2];
    wTensorView->strides[2] *= wTensorView->sizes[2];
    wTensorView->sizes[2] = 1;
    LOG_TRACE(MME_RECIPE, "Applying lowering, new opA view shape: sizes - [{}], strides: [{}], new opB shape : sizes [{}], strides [{}], new opC shape : sizes [{}], strides [{}]",
              arrayToStr(m_recipe.aViews.front().sizes.begin(), m_recipe.aViews.front().sizes.end()),
              arrayToStr(m_recipe.aViews.front().strides.begin(), m_recipe.aViews.front().strides.end()),
              arrayToStr(m_recipe.bViews.front().sizes.begin(), m_recipe.bViews.front().sizes.end()),
              arrayToStr(m_recipe.bViews.front().strides.begin(), m_recipe.bViews.front().strides.end()),
              arrayToStr(m_recipe.cViews.front().sizes.begin(), m_recipe.cViews.front().sizes.end()),
              arrayToStr(m_recipe.cViews.front().strides.begin(), m_recipe.cViews.front().strides.end()) );

}

void RecipeGenerator::applyFlattening()
{
    MmeTensorView& aTensorView = m_recipe.aViews.front();
    MmeTensorView& cTensorView = m_recipe.cViews.front();

    MME_ASSERT(aTensorView.sizes[1] == cTensorView.sizes[1], "expected same Height for A and C");
    MME_ASSERT(aTensorView.sizes[2] == cTensorView.sizes[2], "expected same Batch for A and C");

    unsigned originalSize = cTensorView.sizes[1];
    aTensorView.dcoreBases[1] = aTensorView.dcoreBases[2] * originalSize;
    cTensorView.dcoreBases[1] = cTensorView.dcoreBases[2] * originalSize;
    aTensorView.dcoreBases[2] = cTensorView.dcoreBases[2] = 0;
    unsigned flattenedSize = cTensorView.sizes[1] * cTensorView.sizes[2];
    aTensorView.sizes[1] = cTensorView.sizes[1] = flattenedSize;
    aTensorView.sizes[2] = cTensorView.sizes[2] = 1;
    aTensorView.strides[2] = aTensorView.strides[1] * aTensorView.sizes[1];
    cTensorView.strides[2] = cTensorView.strides[1] * cTensorView.sizes[1];

    LOG_TRACE(MME_RECIPE, "Applying flattening, new opA view shape: sizes - [{}], strides [{}], new opC view shape: sizes- [{}] strides - [{}]",
              arrayToStr(aTensorView.sizes.begin(), aTensorView.sizes.end()),
              arrayToStr(aTensorView.strides.begin(), aTensorView.strides.end()),
              arrayToStr(cTensorView.sizes.begin(), cTensorView.sizes.end()),
              arrayToStr(cTensorView.strides.begin(), cTensorView.strides.end()));
}
// Decide whether A, B or none should be SB-reused.
void RecipeGenerator::calcReuseOperand()
{
#if ENABLE_SB_REUSE
    if (
#if GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE
        ((m_constants.mmeHalReader.getChipType() == ChipType::e_mme_Gaudi) && m_ignoreSbReuse) ||
#endif
        !m_constants.params.isSbReuse())
#endif
    {
        return;  // No reuse
    }

    //  an indication whether we can reuse on a higher dimension in case there is no moemvent in the first dimension of
    //  the walk.
    bool canTransferRaster = true;

    //  Prevent SB reuse if number of batches exceeds a threshold
    //  also prevent reuse in case batch movement is not the last part of the walk.
    if (m_constants.params.isGemmOperation())
    {
        constexpr unsigned batchStartingDim = MAX_DIMENSION - c_batchDimNr;
        const auto& sizes = m_recipe.cViews.front().sizes;
        const unsigned batchesNr = multiplyElements(sizes.begin() + batchStartingDim, sizes.end());
        // maskedBgemm flow required sb reuse flow.
        if (batchesNr > MAX_BATCH_NR_FOR_SB_REUSE && !m_constants.params.strategy.maskedBgemm)
        {
            return;  // No reuse
        }

        switch (m_constants.params.strategy.pattern)
        {
            case e_mme_sp_reduction_kcf:
            case e_mme_sp_reduction_ckf:
                //  batch movement replaces both inputs, cant reuse.
                return;  //  No reuse
            case e_mme_sp_reduction_kfc:
            case e_mme_sp_reduction_cfk:
                //  batch movement is in the middle of the walk, cant reuse over it.
                canTransferRaster = false;
            default:
                break;
        }
    }

    // Choose either A or B
    if (m_recipe.raster)
    // Scan fast direction first
    {
        if (m_fcdGeoPerReuseNr > 1)
        {
            m_recipe.reuseInfo.setReuse(e_mme_in_a);  // B consumes multiple GEOs
        }
        else if (canTransferRaster && m_spGeoPerReuseNr > 1)
        {
            m_recipe.reuseInfo.setReuse(e_mme_in_b);  // A produces multiple EUs
            m_recipe.raster = false;  // Practically it's not a raster walk
        }
    }
    else
    // Scan spatial direction first
    {
        if (m_spGeoPerReuseNr > 1)
        {
            m_recipe.reuseInfo.setReuse(e_mme_in_b);  // A consumes multiple GEOs
        }
        else if (canTransferRaster && m_fcdGeoPerReuseNr > 1)
        {
            m_recipe.reuseInfo.setReuse(e_mme_in_a);  // B consumes multiple GEOs
            m_recipe.raster = true;  // Practically it's a raster walk
        }
    }
}

// Update SB utilization according to misalignment of reused operand
void RecipeGenerator::calcSbUtilization()
{
    if (m_recipe.reuseType() == EMmeReuseType::e_mme_no_reuse) return;
    const EMmeInputOperand reuseOp = m_recipe.reuseA() ? EMmeInputOperand::e_mme_in_a : EMmeInputOperand::e_mme_in_b;
    const float sbUtilization =
        SBReuse::calcSbUtilization(m_constants.mmeHalReader,
                                   m_constants.params,
                                   reuseOp,
                                   m_constants.geoAttr.getPortSize(static_cast<EMmeInternalOperand>(reuseOp)));
    m_recipe.reuseInfo.setSbUtilization(sbUtilization);
}

// zeroCD (is a feature only for BGEMM ) - RoiSizes of size 0 should be padded to 1
void RecipeGenerator::padZeroCD()
{
    unsigned aCD = isTransposed(m_constants.params.opType, e_mme_in_a) ? 0 : 1;
    unsigned bCD = isTransposed(m_constants.params.opType, e_mme_in_b) ? 0 : 1;

    if (m_recipe.aRoiSizes.back()[aCD] == 0)
    {
        m_recipe.aRoiSizes.back()[aCD] = 1;
    }
    if (m_recipe.bRoiSizes.back()[bCD] == 0)
    {
        m_recipe.bRoiSizes.back()[bCD] = 1;
    }
}

void RecipeGenerator::padMemsetDesc()
{
    // in memset desc - one of w dims is 0, need to ammend roiSize to 1 to avoid propegating this.
    // aguConfig is setting validElements to 0 in any case.
    if (ConvSubProblemContainer::isMemsetDesc(m_constants.params) && (m_constants.params.isDedxOperation()))
    {
        for (auto& sizeArrays : m_recipe.bRoiSizes)
        {
            for (auto& size : sizeArrays)
            {
                if (size == 0)
                {
                    size = 1;
                }
            }
        }
    }
}

// pad CD to reduction tree requirements. ** This functions is currently not in use **
void RecipeGenerator::padCommonDim()
{
    unsigned aCD = isTransposed(m_constants.params.opType, e_mme_in_a) ? 0 : 1;
    unsigned bCD = isTransposed(m_constants.params.opType, e_mme_in_b) ? 0 : 1;

    SBReuse sbReuse(m_constants, m_recipe, m_grids, 0);
    m_recipe.aRoiSizes.back()[aCD] = sbReuse.calcValidCommonDimAlignment(m_recipe.aRoiSizes.back()[aCD], /*alignToCL*/false, /*roundDown*/false);
    m_recipe.bRoiSizes.back()[bCD] = sbReuse.calcValidCommonDimAlignment(m_recipe.bRoiSizes.back()[bCD], /*alignToCL*/false, /*roundDown*/false);
}

//  pad the first spatial dimension according to bgemm requirements and number of interleaved ports
void RecipeGenerator::padSpatialDim()
{
    if (m_constants.params.isGemmOperation())
    {
        //  since in bgemm the spatial dimension is in the middle of the tensor we need to pad to the amount of readers.
        //  that is to prevent wrap arounds and make sure we read and write only from the same batch.
        //  in convolution nodes we dont need it since the spatial size reaches the end of the tensor and will leave the
        //  ROI bounds.
        m_recipe.aRoiSizes.back()[1] =
            alignToVal(m_recipe.aRoiSizes.back()[1], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a));
        m_recipe.bRoiSizes.back()[1] =
            alignToVal(m_recipe.bRoiSizes.back()[1], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b));
        m_recipe.cRoiSizes.back()[1] =
            alignToVal(m_recipe.cRoiSizes.back()[1],
                       m_constants.geoAttr.getInterleavedSpatialPortsNr(
                           isTransposed(m_constants.params.opType, e_mme_in_a) ? e_mme_op_a : e_mme_op_c));
    }
    else if (m_constants.params.opType == e_mme_fwd)
    {
        //  in fwd B behaves like a a batched gemm operand, since we have multiple readers we have to make sure
        //  the they dont exceed the first spatial dim as they will wrap around and start reading frmo the next weight.
        m_recipe.bRoiSizes.back()[1] =
            alignToVal(m_recipe.bRoiSizes.back()[1], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b));
    }

    // pad the first spatial dimension to the number of interleaved readers
    unsigned spDimA = m_constants.geoAttr.getSpInterleavingDim(e_mme_op_a);
    unsigned spDimB = m_constants.geoAttr.getSpInterleavingDim(e_mme_op_b);
    m_recipe.aRoiSizes.back()[spDimA] =
        std::max(m_recipe.aRoiSizes.back()[spDimA], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a));
    m_recipe.bRoiSizes.back()[spDimB] =
        std::max(m_recipe.bRoiSizes.back()[spDimB], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b));

    unsigned spDimC = m_constants.geoAttr.getSpInterleavingDim(e_mme_op_c);
    if (isTransposed(m_constants.params.opType, e_mme_in_a))
    {
        // When A is transposed every row in C is a row in A. Since usually A has more ports than C,
        // C might get padded rows also due to A port limitation
        m_recipe.cRoiSizes.back()[spDimC] =
            std::max(m_recipe.cRoiSizes.back()[spDimC],
                     std::max(m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_c),
                              m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a)));
    }
    else
    {
        m_recipe.cRoiSizes.back()[spDimC] =
            std::max(m_recipe.cRoiSizes.back()[spDimC], m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_c));
    }
}

//  pad the first batch dimension according number of interleaved ports
void RecipeGenerator::padBatchDim()
{
    if (!m_constants.geoAttr.supportsConcurrency()) return;
    unsigned dim = m_constants.geoAttr.getConcurrentDim();
    unsigned concurrency = m_constants.geoAttr.getGeometryConcurrency();

    if (!m_constants.params.isDedwOperation())
    {
        // TODO make sure this doesnt interfere with broadcast
        // pad A
        m_recipe.aRoiSizes.back()[dim] = alignToVal(m_recipe.aRoiSizes.back()[dim], concurrency);
        // pad B
        m_recipe.bRoiSizes.back()[dim] = alignToVal(m_recipe.bRoiSizes.back()[dim], concurrency);
    }
    // pad C
    m_recipe.cRoiSizes.back()[dim] = alignToVal(m_recipe.cRoiSizes.back()[dim], concurrency);
}

// pad RoiSizes for all tensors in recipe
void RecipeGenerator::recipeSetRoiSize(int splitIdx)
{
    m_recipe.aRoiSizes.push_back(m_recipe.aViews[splitIdx].sizes);
    m_recipe.bRoiSizes.push_back(m_recipe.bViews[splitIdx].sizes);
    m_recipe.cRoiSizes.push_back(m_recipe.cViews[splitIdx].sizes);

    if (m_constants.params.opType == e_mme_gemm_transpose) m_recipe.bRoiSizes.back()[1] = m_recipe.aRoiSizes.back()[0];

    padZeroCD();
    //    padCommonDim();  common dim padding is currently handle by agu config
    padSpatialDim();
    padBatchDim();
    padMemsetDesc();
}

// Find how many geo are possible to cover single reuse at FCD. The result will be used to make sure that
// we wont exceed c_mme_max_sb_reuse.
void RecipeGenerator::calcFcdGeoContinuousLength()
{
    const auto geoTotalElemWidth = m_constants.geoAttr.getGeometryWidth();
    m_fcdGeoPerReuseNr = div_round_up(m_recipe.cViews.front().sizes[0], geoTotalElemWidth);
}

// Calculate maximum adjacent number of geo to be produced per reuse. The result will be used to make sure that
// we wont exceed c_mme_max_sb_reuse.
// Note: this is not equal the total number of geo to be used in spatial direction. See the example below:
//
// Example: say you have 2x2 filter and DEDW op. AT consists of 4 filters as input.
// Each filter contribute to one output (total 4 outputs). Let's present two cases of SB reuse on B:
// Case 1: Now assume walking pattern is CKF. In first iteration we visit each filter and calculate sub-views at (0, 0)
// 2nd iter we calculate sub-views at (1, 0) and so on.
// Case 2: Walking pattern is KCF. After visiting (0, 0) we move to (0, 1).
// The second case is far more efficient than first because One reuse produces two sub-view while only one in the first.
// Observation: in case 2 we included all geo at SP direction but not case 1.
void RecipeGenerator::calcSpGeoContinuousLength()
{
    // When no SB reuse there is no constrains, also DMA operation dont have a spatial dim.
    if (!m_constants.params.isSbReuse() || m_constants.params.isDmaOperation())
    {
        m_spGeoPerReuseNr = -1U;
        return;
    }

    const EMmeOpType op = m_constants.params.opType;
    const EMmePattern pattern = m_constants.params.getPattern();
    const MmeTensorView& cView = m_recipe.cViews.front();
    m_spGeoPerReuseNr = 1;
    const auto geoTotalElemHeight = m_constants.geoAttr.getGeometryHeight();
    if (m_constants.params.isFwdOrDedx())
    {
        // All filters contribute to single output, thus we consider the whole spatial dimension
        unsigned spSize = 1;
        for (unsigned i = 1; i < MME_MAX_TENSOR_DIMS; ++i)
        {
            spSize *= m_recipe.cRoiSizes.front()[i];
        }
        m_spGeoPerReuseNr = div_round_up(spSize, geoTotalElemHeight);
    }
    else if (isDedwOperation(op))
    {
        const unsigned fcdStepsNr = div_round_up(cView.sizes[0], m_constants.geoAttr.getGeometryWidth());
        const unsigned cStepsNr = div_round_up(cView.sizes[1], geoTotalElemHeight);
        std::array<unsigned, c_batchDimNr> filterSizes;
        std::copy(std::begin(cView.sizes) + 2, std::end(cView.sizes), std::begin(filterSizes));
        unsigned concurrentDim = m_constants.geoAttr.getConcurrentDim() - GEMM_DIM_B1;
        filterSizes[concurrentDim] = div_round_up(filterSizes[concurrentDim], m_constants.geoAttr.getGeometryConcurrency());
        const unsigned filterSteps = multiplyElements(std::begin(filterSizes), std::end(filterSizes));

        // case where k=1: k is naive and can be over looked
        if (fcdStepsNr == 1)
        {
            // C and F are/become adjacent, can reuse on both of them
            m_spGeoPerReuseNr = cStepsNr * filterSteps;
        }
        else
        {
            switch (pattern)
            {
                case e_mme_sp_reduction_fkc:
                    m_spGeoPerReuseNr = cStepsNr;  // A complete filter one after the other
                    break;
                case e_mme_sp_reduction_ckf:
                    m_spGeoPerReuseNr = filterSteps;  // One row at a time, loop on filters
                    break;
                case e_mme_sp_reduction_kcf:
                case e_mme_sp_reduction_kfc:
                    // C and F are adjacent, can reuse on both of them
                    m_spGeoPerReuseNr = cStepsNr * filterSteps;
                    break;
                case e_mme_sp_reduction_cfk:  // Those are the only raster patterns
                case e_mme_sp_reduction_fck:
                    m_spGeoPerReuseNr = 1;  // It's a classic raster scan, complete rows one by one
                    break;
                default:
                    MME_ASSERT(0, "Walk pattern is not supported");
            }
        }
    }
    else if (op == e_mme_atbt || op == e_mme_ab || op == e_mme_atb || op == e_mme_abt || op == e_mme_reductionAdd)
    {
        // Gemm op assumes no filters. Thus all GEO contribute to same output
        const unsigned cStepsNr = div_round_up(m_recipe.cRoiSizes.front()[1], geoTotalElemHeight);
        const unsigned batches = multiplyElements(cView.sizes.begin() + 2, cView.sizes.end());
        if (batches == 1)
        {
            // Gemm op assumes no filters. Hence we support only two patterns.
            m_spGeoPerReuseNr = cStepsNr;
        }
        else  // BGEMM
        {
            // Consider only single batch
            switch (pattern)
            {
                case e_mme_sp_reduction_fck:
                case e_mme_sp_reduction_cfk:
                case e_mme_sp_reduction_ckf:
                    m_spGeoPerReuseNr = 1;  // Classic raster
                    break;
                case e_mme_sp_reduction_kfc:
                case e_mme_sp_reduction_fkc:
                    m_spGeoPerReuseNr = cStepsNr;  // Movement at spatial direction
                    break;
                case e_mme_sp_reduction_kcf:
                    m_spGeoPerReuseNr = 1;  // Non-raster, complete one GEO of one batch then move to the next
                    break;
                default:
                    MME_ASSERT(0, "Walk pattern is not supported");
            }
        }
    }
    else
    {
        MME_ASSERT(0, "Operation is not supported");
    }
}

// Try to extend spatial grids according to pipeline level hint.
// Definition of "pipeline level": it's equal to number of activations at:
//    * spatial direction when waking pattern is raster.
//    * fast direction when waking pattern is not raster.
// This method tries to extend number of activations at the suitable direction as long as this doesn't hurt utilization.
void RecipeGenerator::handlePipelineLevelHint()
{
    const unsigned pipelineLevel = m_constants.params.getPipelineLevel();
    if (pipelineLevel <= 1 || m_constants.params.isNativeDmaOperation())
    {
        return;  // No hint
    }

    Grid& fcdGrid = m_grids.getFcdGrid();
    const unsigned fcdGeoNr = div_round_up(fcdGrid.getViewSize(false), m_constants.geoAttr.getGeometryWidth());
    Grid& spGrid = m_grids.getSpGrid();
    const unsigned spGeoNr = div_round_up(spGrid.getViewSize(false), m_constants.geoAttr.getGeometryHeight());

    if (m_recipe.raster)
    {
        // For raster try to extend number of activations at spatial direction
        const unsigned newSpSteps = std::min(spGeoNr, pipelineLevel);  // This 'min' protects the utilization
        if (newSpSteps > spGrid.getGridSize())
        {
            spGrid.extend(newSpSteps);
        }
    }
    else
    {
        // For non-raster try to extend number of activations at fast direction
        const unsigned newFcdSteps = std::min(fcdGeoNr, pipelineLevel);
        if (newFcdSteps > fcdGrid.getGridSize())
        {
            fcdGrid.extend(newFcdSteps);
        }
    }

    if ((spGrid.getGridSize() * fcdGrid.getGridSize()) >= pipelineLevel)
    {
        return;  // Pipeline requirements are fulfilled
    }

    // Try to extend at the other direction just to have more throughput
    if (m_recipe.raster)
    {
        const bool reuseA = (m_recipe.reuseA());
        const unsigned maxFcdSteps = reuseA ? fcdGeoNr / m_constants.mmeHalReader.getAccumsNr() : fcdGeoNr;
        const unsigned fastDimPipelineLevel = (pipelineLevel + spGrid.getGridSize() - 1) / spGrid.getGridSize();
        const unsigned newFcdSteps = std::min(maxFcdSteps, fastDimPipelineLevel);
        if (newFcdSteps > fcdGrid.getGridSize())
        {
            fcdGrid.extend(newFcdSteps);
        }
    }
    else
    {
        const bool reuseB = (m_recipe.reuseB());
        const unsigned maxSpSteps = reuseB ? spGeoNr / m_constants.mmeHalReader.getAccumsNr() : spGeoNr;
        const unsigned fastDimPipelineLevel = (pipelineLevel + fcdGrid.getGridSize() - 1) / fcdGrid.getGridSize();
        const unsigned newSpSteps = std::min(maxSpSteps, fastDimPipelineLevel);
        if (newSpSteps > spGrid.getGridSize())
        {
            spGrid.extend(newSpSteps);
        }
    }
}

// This method determines whether to use additional operand or not. By invoking it we assume either A or B is reused.
void RecipeGenerator::handle2dSBReuse()
{
    // Gaudi does not support 2d reuse
    ChipType chipType = m_constants.mmeHalReader.getChipType();
    if (chipType == ChipType::e_mme_Gaudi)
    {
        return;
    }

    const bool reuseA = m_recipe.reuseA();
    const bool reuseB = m_recipe.reuseB();
    // Check if 2D is possible to be supported
    if (
        // No 1D reuse, 2D reuse becomes meaningless:
        (!reuseA && !reuseB) ||
        // 2D reuse is possible when there are multiple geo steps for reuse operand:
        (reuseA && (m_spGeoPerReuseNr == 1)) || (reuseB && (m_fcdGeoPerReuseNr == 1)) ||
        // Partial reuse uses only one spatial geometry:
        m_grids.getCommonDimGrid(0).isPartialReuse())
    {
        return;
    }

    // Calc number of spatial (relative to output) subviews
    const EMmeInputOperand reuse2dOperand = reuseA ? e_mme_in_b : e_mme_in_a;
    const unsigned spatialSubviewsNr = calcSpatialSubviewsNr(reuse2dOperand);
    // The actual decision for second operand reuse
    const RecipeSubviewType spatialSubviewType = calcSpatialSubviewType(reuse2dOperand);
    SecondOperandReuse reuse2dInfo(spatialSubviewsNr, spatialSubviewType);
    // Cache to save repeated inputs
    using CacheKey = unsigned;
    std::map<CacheKey, bool> reuse2dCache;

    // For each spatial subview determine if it second-reused
    for (unsigned spatialSubviewIdx = 0; spatialSubviewIdx < spatialSubviewsNr; spatialSubviewIdx++)
    {
        unsigned secondOperandSpatialLength = calcSpatialLength(spatialSubviewIdx, reuse2dOperand);
        // Handle cache miss
        const CacheKey key(secondOperandSpatialLength);
        if (reuse2dCache.find(key) == reuse2dCache.end())
        {
            // Check if second operand fits into SB
            SBReuse sbReuse(m_constants, m_recipe, m_grids, 0, secondOperandSpatialLength);
            reuse2dCache[key] = !sbReuse.isPartialSBReuse();
        }
        // If second operand fits into SB then it's 2D-reusable
        if (reuse2dCache[key])
        {
            reuse2dInfo.set(spatialSubviewIdx);
        }
    }

    // Enable 2D reuse (or ignore it if no second reuse decision was made)
    m_recipe.reuseInfo.set2ndOperandReuseInfo(reuse2dInfo);
}

// Return type of spatial (relative to output) subview
RecipeSubviewType RecipeGenerator::calcSpatialSubviewType(EMmeInputOperand operand) const
{
    RecipeSubviewType spatialSubviewType;
    if (operand == e_mme_in_a)
    {
        const bool isSp = (m_grids.getSpGrid().getType() == SP_GRID);
        spatialSubviewType = isSp ? e_mme_sp_subview : e_mme_non_spatial_subview;
    }
    else
    {
        spatialSubviewType = e_mme_fcd_subview;
    }
    return spatialSubviewType;
}

// Calc number of spatial subviews of an operand relative to output
unsigned RecipeGenerator::calcSpatialSubviewsNr(EMmeInputOperand operand) const
{
    unsigned spatialSubviewsNr = 0;
    switch (calcSpatialSubviewType(operand))
    {
        case e_mme_fcd_subview:
            spatialSubviewsNr = m_recipe.getFcdSubviews().size();
            break;
        case e_mme_sp_subview:
            spatialSubviewsNr = m_recipe.getSpSubviews().size();
            break;
        case e_mme_non_spatial_subview:
            spatialSubviewsNr = m_recipe.getNonSpatialSubviews().size();
            break;
        default:
            MME_ASSERT(0, "Unsupported subview type");
    }
    MME_ASSERT(spatialSubviewsNr != 0, "Invalid spatial subview");
    return spatialSubviewsNr;
}

// Calculate spatial length of an operand relative to output
unsigned RecipeGenerator::calcSpatialLength(unsigned subviewIdx, EMmeInputOperand operand) const
{
    unsigned spatialLength = 0;
    switch (calcSpatialSubviewType(operand))
    {
        case e_mme_fcd_subview:
            spatialLength = m_recipe.getFcdSubviews()[subviewIdx].viewSize;
            break;
        case e_mme_sp_subview:
            spatialLength = m_recipe.getSpSubviews()[subviewIdx].viewSize;
            break;
        case e_mme_non_spatial_subview:  // This is DEDW case
        {
            SizeArray sizes = m_recipe.getNonSpatialSubviews()[subviewIdx].sizes;
            // Before calculating overall length replace dim 1 with full geometry's spatial length to
            // match SB actual data occupation
            if (sizes[1] < m_constants.geoAttr.getGeometryHeight())
            {
                sizes[1] = m_constants.geoAttr.getGeometryHeight();
            }
            spatialLength = std::accumulate(sizes.begin(), sizes.end(), 1, [](unsigned int acc, unsigned int val) {
                return (val == 0) ? 1 : (acc * val);
            });
        }
        break;
        default:
            MME_ASSERT(0, "Invalid subview type");
    }
    MME_ASSERT(spatialLength != 0, "Invalid spatial length");
    return spatialLength;
}

// set the amount of times each descriptor should signal for signalAmount mode.
// to do this we shall divide the original amount by the number of non-partial descriptors.
// TODO: this should also take into account the amount of MME units so we wont signal to much
void RecipeGenerator::setSignalAmount()
{
    if (m_constants.params.getSignalingMode() == EMmeSignalingMode::e_mme_signaling_amount)
    {
        const bool isSp = (m_grids.getSpGrid().getType() == SP_GRID);
        const unsigned spGridSize = isSp ? m_recipe.getSpSubviews().size() : m_recipe.getNonSpatialSubviews().size();
        // the amount of non partial descriptors
        const unsigned storingRecipeNr = m_recipe.getFcdSubviews().size() * spGridSize;
        m_recipe.signalAmount = div_round_up(m_constants.params.getSignalAmount(), storingRecipeNr);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SBReuse
////////////////////////////////////////////////////////////////////////////////////////////////////

// Main constructor
SBReuse::SBReuse(const RecipeConstants& constants,
                 const MmeRecipe& recipe,
                 const RecipeGrids& grids,
                 unsigned commonDimGridIdx,
                 unsigned secondOperandSpatialLength)
: m_constants(constants),
  m_recipe(recipe),
  m_commonDimGridIdx(commonDimGridIdx),
  m_grids(grids),
  m_2ndOperandSpatialLength(secondOperandSpatialLength),
  m_reuseOpType(calcReuseOperand()),
  m_sbSize(calcSingleSBSize(m_constants.mmeHalReader,
                            m_constants.params,
                            m_constants.geoAttr.getPortSize(static_cast<EMmeInternalOperand>(m_reuseOpType)),
                            m_reuseOpType)),
  m_sbSpan(calcSBSpan()),
  m_sbCommonDimSize(calcSBCommonDimSize())
{
}

// Return which input operand A or B to be reused
EMmeInputOperand SBReuse::calcReuseOperand() const
{
    if (m_recipe.reuseA())
    {
        return isFirstReuse() ? e_mme_in_a : e_mme_in_b;
    }
    MME_ASSERT(m_recipe.reuseB(), "Expected SB reuse on operand B");
    return isFirstReuse() ? e_mme_in_b : e_mme_in_a;
}

// Return "true" if the tensor cannot fully fit into SB (partial SB reuse).
// Or "false" otherwise (non-partial SB reuse).
bool SBReuse::isPartialSBReuse() const
{
    // Calc SB reuse length along common dim
    return (m_sbCommonDimSize > m_sbSpan) || (m_recipe.m_gemmNr > 1) || (m_grids.getNumCdCuts(0) > 1);
}

// Private function to calc SB utilization according to misalignment of reused operand
float SBReuse::calcSbUtilization(const MmeHalReader& mmeHal, const MmeLayerParams& params, EMmeInputOperand reusedInput, const MmeTensorView& operand, unsigned memClElemSize, unsigned elementsPerPortNr)
{
    // For Gaudi2 and below it's 128B extra cache line for every 128B misaligned access: 128/(128*2)=50% penalty.
    if (!isInputAligned(params, memClElemSize, reusedInput))
    {
        return 0.5f;
    }
    return 1.f;
}

// Public function to calc SB utilization according to misalignment of reused operand
float SBReuse::calcSbUtilization(const MmeHalReader& mmeHal, const MmeLayerParams& params, EMmeInputOperand reusedInput, unsigned elementsPerPortNr)
{
    const MmeTensorView& operand = params.getOperand(static_cast<EMmeInternalOperand>(reusedInput));
    const unsigned elementSize = getElementSize(operand.elementType);
    const unsigned memClElemSize = mmeHal.getMemoryClSize() / elementSize;
    return calcSbUtilization(mmeHal, params, reusedInput, operand, memClElemSize, elementsPerPortNr);
}

// Public function to calculate a single-port-SB size that can be used. For unaligned cases we take the half of it.
// After calculating the reusable size, we guarantee all calculations to be done without considering misalignment.
unsigned SBReuse::calcSingleSBSize(const MmeHalReader& mmeHal, const MmeLayerParams& params, unsigned elementsPerPortNr, std::optional<EMmeInputOperand> reusedInput)
{
    unsigned sbSizeInCLs = (params.controls.sbSizeInCLs == 0) ? mmeHal.getSBSize() : params.controls.sbSizeInCLs;
    const EMmeInputOperand operandType = reusedInput.has_value() ? reusedInput.value() : e_mme_in_a;
    const MmeTensorView& operand = params.getOperand(static_cast<EMmeInternalOperand>(operandType));

    // Shrink SB capacity if the reused input is unaligned. In this case multiple cache lines may enter SB
    const unsigned elementSize = getElementSize(operand.elementType);
    const unsigned memClElemSize = mmeHal.getMemoryClSize() / elementSize;
    const float sbUtilization = calcSbUtilization(mmeHal, params, operandType, operand, memClElemSize, elementsPerPortNr);
    sbSizeInCLs = static_cast<unsigned>(sbUtilization * sbSizeInCLs);

    // Return SB size in elements units
    const unsigned clElemSize = mmeHal.getClSize() / elementSize;
    MME_ASSERT(memClElemSize >= clElemSize, "Invalid MME configuration");
    const unsigned sbSize = sbSizeInCLs * clElemSize;  // SB size in elements units
    MME_ASSERT(sbSize != 0, "Invalid SB size");
    return sbSize;
}

// SB span is the projection of SB virtual area on common dimension
unsigned SBReuse::calcSBSpan() const
{
    MME_ASSERT(m_sbSize != 0, "Invalid SB size");
    // Calculate port's height/width of reused operand for a single EU
    const unsigned portSize = m_constants.geoAttr.getPortSize((EMmeInternalOperand) m_reuseOpType);

    // For 2D reuse the whole subview must reside in SB, thus we consider
    // spatial subview (relative to output) to find out total port size
    unsigned portRepeats = 1;
    if (!isFirstReuse())
    {
        MME_ASSERT(m_2ndOperandSpatialLength != 0, "Invalid spatial length of 2nd operand");
        const unsigned totalPorts = div_round_up(m_2ndOperandSpatialLength, portSize);
        const unsigned totalPortsSizePerGeo = (m_reuseOpType == e_mme_in_a) ? m_constants.geoAttr.getGeometryHeight()
                                                                            : m_constants.geoAttr.getGeometryWidth();
        MME_ASSERT((totalPortsSizePerGeo % portSize) == 0, "Wrong port size calculation");
        const unsigned totalPortsNrPerGeo = totalPortsSizePerGeo / portSize;
        portRepeats = div_round_up(totalPorts, totalPortsNrPerGeo);
    }
    MME_ASSERT(portRepeats != 0, "Wrong port repeats calculation");

    unsigned portWidth = m_constants.geoAttr.getEuFacingPortSize((EMmeInternalOperand)m_reuseOpType);
    unsigned sbSpan = div_round_down(m_sbSize, (portWidth * portRepeats));
    // When reused operand is not transposed, in Gaudi2, up to two ports can collaborate to read data from
    // common dim (it could happen with fp8 or with CDC cases). In Gaudi 3 all ports will collaborate.
    // This has same effect if we have larger SB.
    if (!isTransposed(m_constants.params.opType, m_reuseOpType))
    {
        const unsigned portsNrOnCommonDim =
            // Amount of ports interleaving on the CD within a core:
            m_constants.geoAttr.getCoreSpatialEuPort(static_cast<EMmeInternalOperand>(m_reuseOpType)) *
            // Amount of ports interleaving on the CD between the cores:
            m_constants.geoAttr.getGeometryCdConcurrency();
        MME_ASSERT(portsNrOnCommonDim != 0, "Invalid number of ports");

        sbSpan *= portsNrOnCommonDim;
    }

    const unsigned sbSpanAligned = calcValidCommonDimAlignment(sbSpan, /*alignToCL*/ false, /*roundDown*/ true);
    MME_ASSERT(sbSpanAligned != 0 && sbSpanAligned <= sbSpan, "Wrong SB span calculation");
    return sbSpanAligned;
}

// Calculate the SB-occupation size in elements unit of common dimension
unsigned SBReuse::calcSBCommonDimSize() const
{
    // First, calculate common dimension size of one GEMM according to B operand
    unsigned sbCommonDimSize = m_constants.params.getSingleGemmCD();
    unsigned spInterleavingDim = m_constants.geoAttr.getSpInterleavingDim(e_mme_op_a);
    // Align data at common dim to number of interleaved readers
    unsigned interleavedReadersA = isTransposed(m_constants.params.opType, e_mme_in_a)
                                       ? 1
                                       : m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_a);
    unsigned interleavedReadersB = isTransposed(m_constants.params.opType, e_mme_in_b)
                                       ? 1
                                       : m_constants.geoAttr.getInterleavedSpatialPortsNr(e_mme_op_b);
    unsigned interleavedReaders = std::max(interleavedReadersA, interleavedReadersB);

    // account for spatial interleaving padding
    if (m_constants.params.isDedwOperation() &&
        m_constants.params.getOperand(e_mme_op_b).sizes[spInterleavingDim] < interleavedReaders)
    {
        sbCommonDimSize /= m_constants.params.getOperand(e_mme_op_b).sizes[spInterleavingDim];
        sbCommonDimSize *= interleavedReaders;
    }

    // Align data at common dim to cache line if necessary
    if (isTransposed(m_constants.params.opType, m_reuseOpType))
    {
        const unsigned elementSize = m_recipe.getElemSize();
        sbCommonDimSize = alignToVal(sbCommonDimSize * elementSize, m_constants.mmeHalReader.getClSize()) / elementSize;
    }

    unsigned cdDtAlignment =
        m_constants.mmeHalReader.getNumElementsForCommonDimAlignment(m_recipe.bViews[0].elementType,
                                                                     m_constants.params.opType);

    sbCommonDimSize = alignToVal(sbCommonDimSize, std::max(interleavedReaders, cdDtAlignment));

    // Second, complete multiplications of the filter dims, relevant only for fwd, dedx and transposed dedx
    if (m_constants.params.isFwdOrDedx())
    {
        auto bViewLocalSizes = m_grids.getCommonDimGrid(m_commonDimGridIdx).getSizes();
        for (unsigned i = 2; i < MME_MAX_TENSOR_DIMS; ++i)
        {
            sbCommonDimSize *= bViewLocalSizes[i];
        }
    }

    MME_ASSERT(sbCommonDimSize != 0, "Wrong SB common dim size calculation");
    return sbCommonDimSize;  // Common dimension size in elements unit
}

// Define common dim grid creation params for partial SB reuse
void SBReuse::defineCommonDimGridForPartialReuse(CommonDimParams& commonDimParams, unsigned spStepCapacity)
{
    switch (m_grids.getCommonDimGrid(m_commonDimGridIdx).getType())
    {
        case CONV_GRID:
        case BATCH_GRID:
            defineCommonDimGridOnConvOrBatchForPartialReuse(commonDimParams, spStepCapacity);
            break;
        case SP_GRID:
            defineCommonDimGridOnSpForPartialReuse(commonDimParams);
            break;
        default:
            MME_ASSERT(0, "Unsupported grid");
    }
}

// Find total GEO limits of FCD and SP grids
// We assume both "fcdStepCapacity" and "spStepCapacity" have valid initial values.
void SBReuse::defineSpatialGridsForPartialSBReuse(unsigned& fcdStepCapacity,
                                                  unsigned& spStepCapacity,
                                                  bool partialToMemory)
{
    unsigned accConstraintCapacity;
    if (partialToMemory)
    {
        // if we are performing partials to memory the amount of steps taken is not limited
        // by the number of ACCs because we are not using them for accumulation.
        accConstraintCapacity = m_constants.mmeHalReader.getMaxSBReuse();
    }
    else
    {
        // Acc has a constraint for the partial reuse case and it's constrained to c_mme_accums_nr.
        // When reuse operand is A and pattern is not a raster then FCD geometry must be one.
        // When reuse operand is B and pattern is not a raster then SP geometry must be one.
        unsigned accNr = m_constants.mmeHalReader.getAccumsNr();
        if (m_constants.geoAttr.getDoubleAccumsBit()) accNr *= 2;
        accConstraintCapacity =
            std::min(accNr, m_constants.mmeHalReader.getMaxSBReuse());
    }
    // If either one of the raster asserts hits then raster flag supposed to be fixed,
    // otherwise capacity is restricted to one which doesn't make sense.
    if (m_reuseOpType == e_mme_in_a)
    {
        MME_ASSERT(m_recipe.raster, "Expected raster walk on reuse A");
        fcdStepCapacity = std::min(fcdStepCapacity, accConstraintCapacity);
    }
    else
    {
        MME_ASSERT(!m_recipe.raster, "Expected non-raster walk on reuse B");
        spStepCapacity = std::min(spStepCapacity, accConstraintCapacity);
    }

    // Constraining FCD and SP
    // ~~~~~~~~~~~~~~~~~~~~~~~
    // Consider an example where you have FWD operation to be executed on Gaudi2 with decision to reuse operand B and
    // split common dim into two parts.
    // Assume also that operand C is produced by 4 GEMMs so that m_fcdGeoPerReuseNr = m_spGeoPerReuseNr = 2.
    // This example fits a non-raster walking pattern, for FWD it must be KSF (see isPatternRaster function).
    // KSF and the partial reuse obligates the subviews to be visited by this order:
    //
    //             +---+---+
    //             | 2 | 4 |
    //           B +---+---+
    //             | 1 | 3 |
    //             +---+---+
    //
    //  +---+---+  +---+---+
    //  | 3 | 1 |  | 1 | 3 |
    //  +---+---+  +---+---+
    //  | 4 | 2 |  | 2 | 4 |
    //  +---+---+  +---+---+
    //      A          C
    //
    // In Gaudi2 there are 4 accumulators (accumsNr=4, acc0, acc1, acc2, acc3).
    // For this partial reuse example we need MIN{accumsNr, m_spGeoPerReuseNr}=2 accumulators to perform the vertical
    // GEMM calculations. Thus there will be 8 GEMM calculations as following:
    //
    // (1) acc0 = A1 * B1
    // (2) acc1 = A2 * B1
    // (3) C1   = A3 * B2 + acc0
    // (4) C2   = A4 * B2 + acc1
    // (5) acc0 = A1 * B3
    // (6) acc1 = A2 * B3
    // (7) C3   = A3 * B4 + acc0
    // (8) C4   = A4 * B4 + acc1
    //
    // Following the non-raster walking pattern, suspension buffer builds B1 at (1) and replay it at (2).
    // Then builds B2 at (3) and replay it at (4), etc..
    // SB capacity prevents it to store More than one B subview and it’s impossible (by HW design) to alternate
    // build-reply for two subview or more.
    // Thus each B subviews must be visited in a different descriptor. In particular, in this example FCD must be split
    // into 2 parts so that B1 and B3 cannot be processed at same descriptor.
    // Note that overall descriptors are 4 caused by FCD and common dim splits. No split on spatial subview because we
    // consume 2 accumulators out of 4 (spatial would be split into two when numbers of consumed accums are at least 5).
    //
    // Conclusion: It's not possible to reuse A on two or more spatial geometries because SB cannot repeat A for first
    // FCD walk then repeat the second FCD row. Similar claim for reuse B.
    // In such cases we need to restrict the capacities to 1.
    if (m_reuseOpType == e_mme_in_a)
    {
        spStepCapacity = 1;
    }
    else
    {
        fcdStepCapacity = 1;
    }
}

// Find total GEO limits of FCD and SP grids
// We assume both "fcdStepCapacity" and "spStepCapacity" have valid initial values.
void SBReuse::defineSpatialGridsForNonPartialSBReuse(unsigned& fcdStepCapacity, unsigned& spStepCapacity)
{
    // number of sb reuse steps is limited to c_mme_max_sb_reuse
    const unsigned maxReuseSteps = m_constants.mmeHalReader.getMaxSBReuse();
    if (m_reuseOpType == e_mme_in_a)
    {
        MME_ASSERT(m_recipe.raster, "Expected raster walk on reuse A");
        fcdStepCapacity = std::min(fcdStepCapacity, maxReuseSteps);
    }
    else
    {
        MME_ASSERT(!m_recipe.raster, "Expected non-raster walk on reuse B");
        spStepCapacity = std::min(spStepCapacity, maxReuseSteps);
    }
}

// Valid alignment for reduction tree.
// "roundDown": determines how to round the misalignment.
unsigned SBReuse::calcValidCommonDimAlignment(unsigned val, bool alignToCL, bool roundDown) const
{
    return calcCommonDimAlignment(m_recipe,
                                  m_constants.params,
                                  m_constants.geoAttr,
                                  m_constants.mmeHalReader,
                                  val,
                                  alignToCL,
                                  roundDown);
}

// Define common dim grid on SP (for DEDW)
void SBReuse::defineCommonDimGridOnSpForPartialReuse(CommonDimParams& commonDimParams)
{
    MME_ASSERT(m_commonDimGridIdx == 0, "Expect dedw to never have CD cuts");
    // This is a case where no dim is fully included in SB
    commonDimParams.reuseType = SB_PARTIAL_REUSE_NO_DIM_INCLUDED;
    commonDimParams.atomicUnitLength = calcValidCommonDimAlignment(1, /*alignToCL*/false, /*roundDown*/true);  // Atomic unit is 1
    commonDimParams.lastIncludedDim = MME_MAX_TENSOR_DIMS;  // "don't care" - the highest dim to fit into SB
    const unsigned spLength = m_grids.getCommonDimGrid(m_commonDimGridIdx).getViewSize();
    MME_ASSERT((m_sbSpan % commonDimParams.atomicUnitLength) == 0, "Invalid common dim creation params");
    unsigned splitsNr = div_round_up(spLength, m_sbSpan);
    MME_ASSERT(splitsNr >= 2, "Wrong splits number calculation");
    commonDimParams.maxFitNr = m_sbSpan / commonDimParams.atomicUnitLength;
}

// Define a grid that supports SB partial reuse for CONV and BATCH grids.
void SBReuse::defineCommonDimGridOnConvOrBatchForPartialReuse(CommonDimParams& commonDimParams, unsigned spStepCapacity)
{
    const auto& grid = m_grids.getCommonDimGrid(m_commonDimGridIdx);
    MME_ASSERT((grid.getType() == CONV_GRID) || (grid.getType() == BATCH_GRID), "Unsupported common dim grid");

    bool shouldAlignToCl = isTransposed(m_constants.params.opType, m_reuseOpType);
    const auto& sizes = grid.getSizes();
    MME_ASSERT(!sizes.empty(), "Invalid common dim grid");
    const unsigned dimsNr = sizes.size() - ((grid.getType() == BATCH_GRID) ? c_batchDimNr : 0);

    // Calculate the following variables:
    unsigned lastIncludedDim = MME_MAX_TENSOR_DIMS;  // The highest dim to fit into SB
    std::optional<unsigned> splitDim;  // The dim to split on. It's the outer dim of last-included-dim
    unsigned maxFitLength = 1;  // Maximum length of atomic units we can consume in a single step

    for (unsigned i = 0; i < dimsNr; i++)
    {
        if (i == grid.getFcdDim())
        {
            continue;  // FCD does not participate in this calculation, as it's not part of the sub-view
        }
        MME_ASSERT(sizes[i] != 0, "Invalid common dim grid");

        unsigned curFitLength = maxFitLength * sizes[i];
        if (i <= 1)  // It's enough to align to cache line once. All the rest are guaranteed to be aligned
        {
            curFitLength = calcValidCommonDimAlignment(curFitLength, /*alignToCL*/false, /*roundDown*/false);
            if (shouldAlignToCl)
            {
                const unsigned elementSize = m_recipe.getElemSize();
                curFitLength =
                    alignToVal(curFitLength * elementSize, m_constants.mmeHalReader.getClSize()) / elementSize;
            }
        }
        if (curFitLength > m_sbSpan)
        {
            splitDim = i;  // Exceeding SB span or reaching last dim
            break;
        }
        lastIncludedDim = i;
        maxFitLength = curFitLength;
        if (curFitLength == m_sbSpan)
        {
            splitDim = i;  // Coalescing SB span
            break;
        }
    }
    MME_ASSERT(!splitDim.has_value() || splitDim.value() != grid.getFcdDim(), "Should not split FCD");

    // Calculate the following variables:
    unsigned atomicUnitLength = 1;  // Length of atomic unit
    unsigned maxFitNr = 0;  // Maximum atomic units we can consume in a single step
    SBReuseType reuseType = SB_NO_REUSE;
    if (lastIncludedDim != MME_MAX_TENSOR_DIMS)
    {
        if (!splitDim.has_value())  // No dim is split
        {
            MME_ASSERT((m_grids.getNumCdCuts(0) > 1) || (m_recipe.m_gemmNr > 1),
                       "No dim is split is allowed only on CD cuts");
            reuseType = SB_PARTIAL_REUSE_ALL_DIMS_INCLUDED;
            if (grid.getType() == BATCH_GRID)
            {  //  we have no split, set max fir to be the entire CD.
                maxFitNr = sizes[grid.getFirstCdDim()];
            }
            else
            {
                maxFitNr = 1;
            }
        }
        else
        {
            // This is a case where at least one dim is fully included in SB
            reuseType = SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED;
            MME_ASSERT(grid.getType() != BATCH_GRID, "We don't allow multiple batches to coexist in the same SB");
            MME_ASSERT((lastIncludedDim < (dimsNr - 1)), "SB non-partial usage flow must be adapted");
            MME_ASSERT(maxFitLength != 0, "Wrong calculations");
            maxFitNr = std::min(m_sbSpan / maxFitLength, sizes[splitDim.value()]);
            MME_ASSERT(maxFitNr != 0, "Invalid fits number");

            //  check if we would prefer to perform reduction to memory instead of accumulating in the ACCs
            if (m_constants.params.strategy.partialsToMemoryEn &&
                !isTypeFp8(m_constants.params.getOperand(e_mme_op_c).elementType))
            {
                // calc amount of partials
                unsigned partialsNr = 1;
                for (unsigned dim = lastIncludedDim + 1; dim < dimsNr; dim++)
                {
                    partialsNr *= sizes[dim];
                }

                //  in port constrained geoemtries reuse if very important for performance as the lack of reuse
                //  causes a 2x perf degradation. in case we are constraint and have a few partials and enough spatial
                //  steps it is preferable to avoid the ACC limitation and perform reduction directly to memory.
                if (partialsNr <= 3 && spStepCapacity >= 8 && m_constants.geoAttr.isGeometryPortConstrained())
                {
                    reuseType = SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY;
                }
            }
        }
    }
    else
    {
        // This is a case where no dim is fully included in SB
        reuseType = SB_PARTIAL_REUSE_NO_DIM_INCLUDED;
        lastIncludedDim = MME_MAX_TENSOR_DIMS;  // Suitable for no-dim-included case
        atomicUnitLength = calcValidCommonDimAlignment(
            shouldAlignToCl ? (m_constants.mmeHalReader.getClSize() / m_recipe.getElemSize()) : 1,
            /*alignToCL*/true, /*roundDown*/true);  // It's intentional that 'alignToCL' is hardcoded 'true' and not 'shouldAlignToCl'
        MME_ASSERT(atomicUnitLength != 0 && atomicUnitLength <= m_sbSpan, "Wrong atomic unit calculation");
        const unsigned firstConvDimSize = sizes[1 - grid.getFcdDim()];
        const unsigned firstConvDimSizeAligned = div_round_up(firstConvDimSize, atomicUnitLength) * atomicUnitLength;
        const unsigned splitsNr = div_round_up(firstConvDimSizeAligned, m_sbSpan);
        MME_ASSERT(splitsNr >= 2, "Wrong splits calculation");
        maxFitNr = div_round_up(firstConvDimSizeAligned, (splitsNr * atomicUnitLength));
        MME_ASSERT(maxFitNr != 0, "Invalid fits number");

        // Theoretical corner case (concrate case at test with name: "dedx failure from resnet50 [SW-54937]"):
        //   atomicUnitLength = u
        //   firstConvDimSizeAligned = 5u
        //   m_sbSpan = 1.9u
        // That gives:
        //   splitsNr = div_round_up(5u, 1.9u) = 3
        //   maxFitNr = div_round_up(5u, 3 * u) = 2
        // That means we are trying to store two atomic units in SB.
        // But that is wrong since m_sbSpan=1.9u i.e. SB can store less than two atomic units.
        if ((maxFitNr * atomicUnitLength) > m_sbSpan)
        {
            MME_ASSERT(maxFitNr != 1, "Invalid fits number");
            maxFitNr--;  // It has the effect of using round down (instead of up) on division result above
            MME_ASSERT(((maxFitNr * atomicUnitLength) <= m_sbSpan), "SB span smaller then maxFit units");
        }
    }

    // Define the grid then perform a basic sanity check
    commonDimParams.reuseType = reuseType;
    commonDimParams.lastIncludedDim = lastIncludedDim;
    commonDimParams.atomicUnitLength = atomicUnitLength;
    commonDimParams.maxFitNr = maxFitNr;
    commonDimParams.partialToMemory = (reuseType == SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SubViewSplitter
////////////////////////////////////////////////////////////////////////////////////////////////////

// Split into sub-views using grid. Or don't split but leave a single sub-view
void SubViewSplitter::split()
{
    commonDimReset();

    // Create the two sub-views at non-common dimensions
    for (const auto& grid : m_grids.getGrids())
    {
        if (grid.isMultiStep())
        {
            split(grid);
        }
        else
        {
            noSplit(grid);
        }
    }

    // Split each of the CommonDim grids
    unsigned numPartials = 0;
    MME_ASSERT(m_recipe.m_gemmNr != 0, "invalid number of gemms that will be accumulated together");
    std::vector<unsigned> numPartialsPerGemm(m_recipe.m_gemmNr, 0);
    for (int gemm = 0; gemm < m_recipe.m_gemmNr; gemm++)
    {
        for (int cdCut = 0; cdCut < m_grids.getNumCdCuts(gemm); cdCut++)
        {
            int cdIdx = gemm * m_grids.getNumCdCuts(gemm) + cdCut;
            const auto& grid = m_grids.getCommonDimGrid(cdIdx);
            auto splitOnBatchDim = false;
            // Create a sub-views at common dimension
            if (grid.isMultiStep())
            {
                if (grid.getType() == BATCH_GRID)
                {
                    constexpr unsigned batchStartingDim = MAX_DIMENSION - c_batchDimNr;
                    splitOnBatchDim = grid.getSplitDim() >= batchStartingDim;
                    //  if there is no gemm acumulation and the split was only on batch dim mark this as
                    //  a non partial CD split, otherwise there are partials.
                    if (m_recipe.m_gemmNr == 1)
                    {
                        m_recipe.setSplitOnBatchDims(splitOnBatchDim);
                    }
                    commonDimSplitForConvAndBatch(grid);
                }
                else if (grid.getType() == CONV_GRID)
                {
                    commonDimSplitForConvAndBatch(grid);
                    if (grid.isPartialToMemory()) m_recipe.setPartialToMemory(true);
                }
                else
                {
                    MME_ASSERT(cdIdx == 0, "invalid number of CD cuts");
                    commonDimSplitForSP();
                }
            }
            else
            {
                commonDimNoSplit(grid);
            }

            unsigned partialNr = splitOnBatchDim ? 1 : grid.getPartialsNr();
            numPartials += partialNr;
            numPartialsPerGemm.at(gemm) += partialNr;
        }
    }

    if (m_recipe.maskedBgemm)
    {
        //  now we have two bgemm operations that each has been split on its CD.
        //  to allow accumulating them in the ACCs we have to interleave their commonDim views.
        //  this will allow us to perform several bgemm, store them in the ACC and then perform several mask gemms.
        auto& convSubViews = m_recipe.getNonSpatialSubviews();
        int batch1 = m_recipe.cViews.front().sizes[GEMM_DIM_B1];
        int batch2 = m_recipe.cViews.front().sizes[GEMM_DIM_B2];
        int batch3 = m_recipe.cViews.front().sizes[GEMM_DIM_B3];
        int actNrPerGemm = batch1 * batch2 * batch3;
        std::vector<unsigned> gemmSubViewIdx(m_recipe.m_gemmNr, 0);
        unsigned startIndexPerGemm = 0;
        for (unsigned gemmNr = 1; gemmNr < gemmSubViewIdx.size(); gemmNr++)
        {
            startIndexPerGemm += actNrPerGemm * numPartialsPerGemm.at(gemmNr - 1);
            gemmSubViewIdx.at(gemmNr) = startIndexPerGemm;
        }
        MultiDimSubViews interleavedConvSubViews(convSubViews.size());
        unsigned interleavedSubViewIdx = 0;
        for (unsigned actNr = 0; actNr < actNrPerGemm; actNr++)
        {
            for (unsigned gemmNr = 0; gemmNr < numPartialsPerGemm.size(); gemmNr++)
            {
                for (unsigned partial = 0; partial < numPartialsPerGemm.at(gemmNr); partial++)
                {
                    interleavedConvSubViews[interleavedSubViewIdx] = convSubViews[gemmSubViewIdx.at(gemmNr)];
                    gemmSubViewIdx.at(gemmNr)++;
                    interleavedSubViewIdx++;
                }
            }
        }

        convSubViews = interleavedConvSubViews;
    }

    // Set number of partials at recipe struct
    m_recipe.setPartialsNrPerGemm(numPartialsPerGemm);
    m_recipe.setPartialsNr(numPartials);
}

// Map grid type to spatial subViews (relative to output)
SingleDimSubViews& SubViewSplitter::getSpatialSubViews(GridType gridType)
{
    if (gridType == FCD_GRID)
    {
        return m_recipe.getFcdSubviews();
    }
    MME_ASSERT(gridType == SP_GRID, "Unsupported grid");
    return m_recipe.getSpSubviews();
}

// Create a single sub-view at non-common dimensions
void SubViewSplitter::noSplit(const Grid& grid)
{
    if (grid.getType() == CONV_GRID)
    {
        auto& covSubViews = m_recipe.getNonSpatialSubviews();
        covSubViews.resize(1);
        covSubViews[0].bases = grid.getBases();
        covSubViews[0].sizes = grid.getSizes();
    }
    else
    {
        auto& subViews = getSpatialSubViews(grid.getType());
        subViews.resize(1);
        subViews[0].viewBase = grid.getViewBase();
        subViews[0].viewSize = grid.getViewSize();
        subViews[0].viewOrigSize = grid.getViewSize(true, true);
    }
}

// Split non-common dimension sub-views
void SubViewSplitter::split(const GridBase& grid)
{
    if (grid.getType() == CONV_GRID)
    {
        commonDimSplitForConvAndBatch(grid);
    }
    else
    {
        auto& subViews = getSpatialSubViews(grid.getType());
        const unsigned gridSize = grid.getGridSize();
        subViews.resize(gridSize);
        unsigned base = grid.getViewBase();
        for (unsigned i = 0; i < gridSize; i++)
        {
            subViews[i].viewBase = base;
            subViews[i].viewSize = grid.calcStepSize(i);
            subViews[i].viewOrigSize = subViews[i].viewSize;
            base += subViews[i].viewSize;
        }
        // Final verification
        MME_ASSERT(base == grid.getViewBase() + grid.getViewSize(), "Wrong split calculation");
    }
}

void SubViewSplitter::commonDimReset()
{
    const auto& grid = m_grids.getCommonDimGrid(0);
    if ((grid.getType() == CONV_GRID) || (grid.getType() == BATCH_GRID))
    {
        auto& covSubViews = m_recipe.getNonSpatialSubviews();
        covSubViews.resize(0);
    }
    else
    {
        auto& subViews = getSpatialSubViews(grid.getType());
        subViews.resize(0);
    }
}

// Create a single sub-view at common dimension
void SubViewSplitter::commonDimNoSplit(const GridBase& grid)
{
    if ((grid.getType() == CONV_GRID) || (grid.getType() == BATCH_GRID))
    {
        auto& convSubViews = m_recipe.getNonSpatialSubviews();
        MultiDimSubView convSubView;
        convSubView.sizes = grid.getSizes();
        convSubView.bases = grid.getBases();
        convSubViews.push_back(convSubView);
    }
    else
    {
        auto& subViews = getSpatialSubViews(grid.getType());
        SingleDimSubView subView;
        subView.viewBase = grid.getViewBase();
        subView.viewSize = grid.getViewSize(false);
        subView.viewOrigSize = grid.getViewSize(false, true);
        subViews.push_back(subView);
    }
}

// Create multiple sub-views at common dimension which is SP
void SubViewSplitter::commonDimSplitForSP()
{
    const auto& grid = m_grids.getCommonDimGrid(0);
    MME_ASSERT(grid.getType() == SP_GRID, "Invalid grid type");
    split(grid);
}

// Create multiple sub-views at CONV
// 'unitSize': size of atomic unit
void SubViewSplitter::commonDimSplitForConvAndBatch(const GridBase& grid)
{
    MME_ASSERT((grid.getType() == CONV_GRID) || (grid.getType() == BATCH_GRID), "Invalid grid type");
    const unsigned gridSize = grid.getGridSize();
    auto& convSubViews = m_recipe.getNonSpatialSubviews();
    const unsigned splitDim = grid.getSplitDim();
    const SizeArray& orgSizes = grid.getSizes();

    // Sizes to be added to each step, it preserves original sizes below splitDim and set the rest to 1
    SizeArray orgSizesPerStep = orgSizes;
    std::fill(orgSizesPerStep.begin() + splitDim + 1, orgSizesPerStep.end(), 1u);

    const unsigned viewSize = grid.getViewSize();
    MultiDimCounter basesCnt(orgSizes, splitDim);  // A multi-dim counter for bases

    unsigned convSizeAcc = 0;
    for (unsigned i = 0; i < gridSize; i++)
    {
        MultiDimSubView subView;
        // Update sizes
        const unsigned stepSize = grid.calcStepSize(i);
        subView.sizes = orgSizesPerStep;
        subView.sizes[splitDim] = stepSize;
        // Update bases
        subView.bases = grid.getBases();
        for (unsigned j = splitDim; j < subView.bases.size(); j++)
        {
            subView.bases[j] += basesCnt[j];
        }
        convSubViews.push_back(subView);
        // Update bases counter
        basesCnt += stepSize;
        // Update size accumulator
        convSizeAcc += stepSize;
        // Assert to catch problem on specific iteration - for faster debug
        MME_ASSERT(convSizeAcc <= viewSize, "Wrong split calculations");
    }
    MME_ASSERT(convSizeAcc == viewSize, "Wrong split calculations");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SubViewSplitter::MultiDimCounter
////////////////////////////////////////////////////////////////////////////////////////////////////

SubViewSplitter::MultiDimCounter::MultiDimCounter(const SizeArray& orgSizes, unsigned splitDim)
: m_firstCountingDim(splitDim)
{
    // Init counting limits
    m_limits[splitDim] = orgSizes[splitDim];
    for (unsigned i = splitDim + 1; i < orgSizes.size(); i++)
    {
        if (orgSizes[i] != 0)
        {
            m_limits[i] = orgSizes[i];
        }
    }
}

// Increment counters at all dims.
// Start by incrementing the first counting dim, then propagate overflows to counters at higher dims.
void SubViewSplitter::MultiDimCounter::operator+=(unsigned stepSize)
{
    MME_ASSERT(stepSize <= m_limits[m_firstCountingDim], "Invalid step size input");
    m_counters[m_firstCountingDim] += stepSize;
    for (unsigned i = m_firstCountingDim; i < m_counters.size(); i++)
    {
        if (m_counters[i] < m_limits[i])
        {
            break;
        }
        if (m_limits[i] != 0)  // 0 means the dim is excluded from incrementation
        {
            MME_ASSERT(m_counters[i] == m_limits[i], "Wrong saturation");
            m_counters[i] = 0;  // Counter is saturated, reset it
            // Propagate the carry to next counter
            unsigned nextCtr = i + 1;
            if (nextCtr < m_counters.size())
            {
                if (m_limits[nextCtr] == 0)  // Don't increment the excluded dim (in DEDX this dim is 1)
                {
                    nextCtr++;
                }
                if (nextCtr < m_counters.size())
                {
                    m_counters[nextCtr]++;
                    MME_ASSERT((m_limits[nextCtr] == 0) || (m_counters[nextCtr] <= m_limits[nextCtr]),
                               "Wrong counter calculation");
                }
            }
        }
    }
}
}  // namespace MmeCommon
