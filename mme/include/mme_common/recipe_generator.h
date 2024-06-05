#ifndef MME__RECIPE_GENERATOR_H
#define MME__RECIPE_GENERATOR_H

#include "include/mme_common/recipe.h"

namespace MmeCommon
{
class CommonGeoAttr;
class MmeHalReader;

// Readme
//
// Terminology used in documentation and naming:
//   SB: Suspension Buffer.
//   EU: single execution unit (one pole contains two EUs)
//   GEO: Two poles and their logical geometry (2xh, 2xw, 4xh, 4xw)
//   FCD: this term is associated to fast common dimension of the output.
//   SP: spatial dimension which is equal to BHW (dims).
//   CONV: the convolution dimension.
//   Template: is the repeated pattern which consists of 'first' and 'lasts' steps.
//   Atomic Unit: is data that defines the granularity of the grid. It's the smallest size
//                of data that we can aggregate when defining grids. The smaller atomic unit is
//                the more flex and optimal balancing we get.
//
// Data units:
//   All data sizes are in elements unit unless explicitly stated in bytes.
//
// How to split a workload?
//   We represent a workload by x1,x2,x3 as appears in the left figure. Operations are defined in the right one:
//
//                       +---------+
//                       |         |              Operations are defined as:
//                       |    B    | x3                  OP     |  A * B  = C
//                       |         |                   ---------+-------------
//                       +---------+                    FWD/A*B |  X * W  = Y         Dimensions:
//          +---------+  +---------+                    A*BT    |  X * WT = Y           W dims = K C S R (Q)
//          |         |  |         |                    DEDX    | dy * WT = X           X dims = C W H (D) B
//          |    A    |  |    C    | x2                 AT*B    | XT * W  = Y           Y dims = K W H (D) B
//          |         |  |         |                    DEDW    | XT * dY = W
//          +---------+  +---------+                    AT*BT   | XT * WT = Y
//                           x1
//
//   For all operations: FCD is determined by x1.
//   For FWD and DEDX: SP is determined by x2 and CONV by x3.
//   For DEDW: SP is determined by x3 and CONV by x2.
//
//   When either A or B are reused and SB cannot fully fit the data, we split only on common dimension.
//   For all operations the common dimension is x3. For this we use CommonDimGrid class.
//   For a non-common-dimension split we use Grid class.
//
//   With that being said, for FWD and DEDX: FCD and SP are represented by Grid and CONV by CommonDimGrid.
//   For DEDW: FCD and CONV are represented by Grid and SP by CommonDimGrid.

////////////////////////////////////////////////////////////////////////////////////////////////////
// External Parameters
////////////////////////////////////////////////////////////////////////////////////////////////////

// A container struct of frequently used external and const data.
// It will be passed between classes to shrink prototypes of their consumers.
struct RecipeConstants
{
    const MmeLayerParams& params;
    const CommonGeoAttr& geoAttr;
    const MmeHalReader& mmeHalReader;
    // Construction and other operations
    RecipeConstants(const MmeLayerParams& paramsIn, const CommonGeoAttr& geoAttrIn, const MmeHalReader& mmeHalReaderIn)
    : params(paramsIn), geoAttr(geoAttrIn), mmeHalReader(mmeHalReaderIn)
    {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Grid Hierarchy
// The purpose of (1D) grid classes is to represent and manage the way we are going
// to split sub-view (via class SubViewSplitter).
// This hierarchy is logically isolated, it's neither aware of operation nor topology. In addition
// it doesn't use any other class not part of the hierarchy.
////////////////////////////////////////////////////////////////////////////////////////////////////

// SB-reuse strategies
enum SBReuseType
{
    UNKNOWN_SB_REUSE,  // To be given as default then to check against uninitialized value
    SB_NO_REUSE,  // SB will not be used fore tensor reuse
    // Part of the tensor wil be reused:
    SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED,  // At least one dim at common dim will be fully included in SB
    SB_PARTIAL_REUSE_AT_LEAST_ONE_DIM_INCLUDED_TO_MEMORY,  // same as the mode above, but perform reduction to memory
                                                           // instead of to ACCs
    SB_PARTIAL_REUSE_NO_DIM_INCLUDED,  // No dim at common dim will be fully included in SB
    SB_PARTIAL_REUSE_ALL_DIMS_INCLUDED,  // All dims are included, i.e. no partials. Can occur when CD is split
    SB_NON_PARTIAL_REUSE  // Every port of whole tensor will be included in SB
};

// Possible types of grids that can be represented by 'GridBase' derivatives:
enum GridType
{
    SP_GRID,
    FCD_GRID,
    CONV_GRID,
    BATCH_GRID
};

// GridBase class contains common variables and methods between different grids
class GridBase
{
public:
    GridBase(GridType gridType,
             const SizeArray& bases,
             const SizeArray& sizes,
             const SizeArray& origSizes,
             const RecipeConstants& constants,
             unsigned splitDim = MME_MAX_TENSOR_DIMS);
    virtual ~GridBase() = default;

    GridType getType() const { return m_gridType; }
    const SizeArray& getBases() const { return m_bases; }
    const SizeArray& getSizes() const { return m_sizes; }
    unsigned getViewBase() const { return m_bases[m_splitDim]; }
    unsigned getViewSize(bool inAtomicUnits = true, bool origSize = false) const;
    unsigned calcStepSize(unsigned stepIdx) const;
    unsigned getSplitDim() const;
    unsigned getGridSize() const;
    virtual bool isMultiStep() const;

protected:
    unsigned
    calcViewSize(bool origSize = false, unsigned dimStart = 0, unsigned dimEnd = MME_MAX_TENSOR_DIMS - 1) const;
    virtual bool isPartialStep(unsigned stepIdxInTemplate) const = 0;
    unsigned calcStepsNrPerTemplate() const;
    unsigned calcUnitsNrPerTemplate() const;
    bool isBalanced() const;
    void applyBalancing();

protected:
    // Variables to be initializes via constructor:

    const GridType m_gridType;
    const SizeArray m_bases = {0};  // Bases of overall sub-view (can be non-zero if GC used slicing)
    const SizeArray m_sizes = {0};  // Sizes in elements unit of overall sub-views
    const SizeArray m_origSizes = {0};  // Sizes in elements unit without padding
    // Optional initialization via constructor:
    unsigned m_splitDim;  // Which dimension to split

    // Variables to be initializes at create(...):

    unsigned m_isCreated = false;  // A flag to tell if grid create method was invoked
    unsigned m_gridSize = 0;  // Size of grid, that's equal to total number of sub-views
    unsigned m_atomicUnitLength = 0;  // Size of data that represents one unit to be tiled at sub-views
    // Representation of first and last steps in atomic units:
    unsigned m_unitsNrPerFirstStep = 0;  // Represents how many atomic units in a 1st step
    unsigned m_firstStepsNr = 0;  // Number of steps relative to sub-views start that have same size
    unsigned m_unitsNrPerLastStep = 0;  // Represents how many atomic units in a last step
    unsigned m_lastStepsNr = 0;  // Number of steps relative to sub-views end that have same size

    // Partial unit length is a partial atomic unit for Grid but not for CommonDimGrid.
    // It can be used at most one time per template to correct the misalignment of a dimension as it split
    // by default according to atomic unit length.
    // Use cases:
    //  - Last sub-view: restricted to FCD and SP (SP of non-DEDW operations) because original tensor may not come
    //                   aligned to GEO.
    //  - Non-partial reuse no-dim-included case: as size may not be aligned so we need actual size en every template
    //                                            termination.
    //  - CONV in DEDW: we may use this value at the end of each every geometry if CONV>GEO.
    unsigned m_partialUnitLength = 0;  // Value is in elements unit
    const RecipeConstants& m_constants;
};

// This Grid class represents divisions that are not made on common dimension.
// At most two possible lengths for the steps. Last step can have different length
// as it contains remaining elements after division.
class Grid : public GridBase
{
public:
    Grid(GridType gridType,
         const SizeArray& bases,
         const SizeArray& sizes,
         const SizeArray& origSizes,
         const RecipeConstants& constants,
         unsigned splitDim);
    virtual ~Grid() = default;

    void create(unsigned maxStepSize, unsigned stepCapacity);
    void extend(unsigned newSize);

private:
    void createForDefault(unsigned maxStepSize, unsigned stepCapacity);
    void distributeUniformly(unsigned gridSize, unsigned viewSize);
    void createForConv(unsigned geoSize, unsigned stepCapacity);
    bool isPartialStep(unsigned stepIdxInTemplate) const final;

private:
    unsigned m_geoAtFirstConvDimNr = 0;  // Number of geometries at first CONV dim
};

// A struct used to determine all required params for common dim grid creation
struct CommonDimParams
{
    SBReuseType reuseType = UNKNOWN_SB_REUSE;
    unsigned lastIncludedDim = MME_MAX_TENSOR_DIMS;
    unsigned atomicUnitLength = 0;
    unsigned maxFitNr = 0;
    bool forcedPartial = false;
    bool partialToMemory = false;
};
using CommonDimParamsVec = llvm_vecsmall::SmallVector<CommonDimParams, 1>;

// This class defines how to split common dimension
class CommonDimGrid : public GridBase
{
public:
    // Construction
    CommonDimGrid(GridType gridType,
                  const SizeArray& bases,
                  const SizeArray& sizes,
                  const SizeArray& origSizes,
                  const RecipeConstants& constants,
                  unsigned fcdDim,
                  bool isMultipleCdCuts);
    virtual ~CommonDimGrid() = default;
    void create(const CommonDimParams& params);
    // Queries
    unsigned getFcdDim() const { return m_fcdDim; }
    unsigned getFirstCdDim() const { return 1 - m_fcdDim; }
    bool isPartialToMemory() const;
    bool isPartialReuse() const;
    bool isMultiStep() const final;
    unsigned getPartialsNr() const;

private:
    unsigned getFirstCommonDim() const;
    unsigned calcSplitDim(unsigned lastIncludedDim) const;
    bool isAtLeastOneDimIncluded() const;
    bool isAllDimsIncluded() const;
    void splitAtLeastOneDimIncluded(unsigned maxFitNr);
    void splitNoDimIncluded(unsigned maxFitNr);
    void calcGridSize();
    bool isPartialStep(unsigned stepIdxInTemplate) const final;

private:
    // Variables that are initialized via constructor
    const unsigned m_fcdDim;  // FCD (of Cout)
    // Variables that are calculated via CommonDimGrid
    SBReuseType m_reuseType = SB_NO_REUSE;
    bool m_multipleCdCuts = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Recipe Grids
////////////////////////////////////////////////////////////////////////////////////////////////////

// Contains all types of grids and some operations to access them
class RecipeGrids
{
public:
    RecipeGrids(const RecipeConstants& extRecipeParams) : m_constants(extRecipeParams) {}

    using GridVec = llvm_vecsmall::SmallVector<Grid, 2>;
    using NumCommonDimCutsVec = llvm_vecsmall::SmallVector<unsigned, 1>;
    using CommonDimGridVec = llvm_vecsmall::SmallVector<CommonDimGrid, 1>;

    // Direct accessors to the grids:
    const GridVec& getGrids() const { return m_grids; }
    Grid& getFcdGrid();
    const Grid& getFcdGrid() const;
    Grid& getSpGrid();  // SP is relative to output
    const Grid& getSpGrid() const;
    CommonDimGrid& getCommonDimGrid(unsigned commonDimGridIdx);
    const CommonDimGrid& getCommonDimGrid(unsigned commonDimGridIdx) const;
    const unsigned getNumCdCuts(int gemm) const { return m_numCommonDimCuts[gemm]; }

private:
    Grid& getGrid(GridType gridType, bool otherGrid);

protected:
    const RecipeConstants& m_constants;
    // All grids
    GridVec m_grids;  // Restricted to two grids
    CommonDimGridVec m_commonDimGridVec;
    NumCommonDimCutsVec m_numCommonDimCuts;  //  number of cut on the CD for a single operation (for recurring
                                             //  misalignment optimization)
};

// Initialize and create grids
class RecipeGridsCreator : public RecipeGrids
{
    // Private params used to create the grids
    struct GridsParams
    {
        GridsParams(unsigned fcdCapacity, unsigned spCapacity, unsigned numCdCuts);
        CommonDimParamsVec commonDimParamsVec;
        unsigned fcdStepCapacity;
        unsigned spStepCapacity;  // Spatial is relative to output
    };

public:
    RecipeGridsCreator(const RecipeConstants& extRecipeParams, const MmeRecipe& recipe);
    void init();
    void create(unsigned fcdStepCapacity, unsigned spStepCapacity);

private:
    // Init grids
    void initGrid(GridType gridType,
                  const SizeArray& bases,
                  const SizeArray& sizes,
                  const SizeArray& origSizes,
                  unsigned splitDim);
    void initGrid(GridType gridType, unsigned base, unsigned size, unsigned origSize);
    void initCommonDimGrid(GridType gridType,
                           const SizeArray& bases,
                           const SizeArray& sizes,
                           const SizeArray& origSizes,
                           unsigned fcdDim,
                           bool isMultipleCdCuts);
    // Define grids
    void defineGrids(GridsParams& recipeGridsParams) const;
    void defineSpatialGrids(unsigned& fcdStepCapacity, unsigned& spStepCapacity) const;
    void defineCommonDimGrid(CommonDimParams& commonDimParams, SBReuseType reuseType) const;
    // Create grids
    void createGrids(const GridsParams& recipeGridsParams);
    int getCommonDimCuts(std::array<MmeTensorView, 2>& splitView, int tensorIdx);

private:
    const MmeRecipe& m_recipe;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// RecipeGenerator
////////////////////////////////////////////////////////////////////////////////////////////////////

// Main class to create MmeRecipe
class RecipeGenerator
{
public:
    // Construction
    RecipeGenerator(RecipeType recipeType,
                    const MmeLayerParams& params,
                    const MmeHalReader& mmeHalReader,
                    const CommonGeoAttr& geoAttr
#if GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE
                    ,
                    const bool ignoreSbReuse = false
#endif
    );
    const MmeRecipe& generateRecipe();
    // Queries
    const MmeRecipe& get() const { return m_recipe; }
    bool isPartialSBReuse();

private:
    void init();
    void applyLowering();
    void applyFlattening();
    void calcReuseOperand();
    void calcSbUtilization();
    void padZeroCD();
    void padCommonDim();
    void padSpatialDim();
    void padBatchDim();
    void padMemsetDesc();
    void recipeSetRoiSize(int splitIdx);
    void calcFcdGeoContinuousLength();
    void calcSpGeoContinuousLength();
    void handlePipelineLevelHint();
    void handle2dSBReuse();
    void setSignalAmount();
    void setGemmNr();
    //  MME transpose TE acceleration
    EMmeInternalOperand getTeAcceleratedOperand() const;
    unsigned getTeAcceleration(EMmeInternalOperand operand) const;
    bool calcTeAcceleration() const;
    void applyTeAcceleration();

    // Spatial subviews operations
    RecipeSubviewType calcSpatialSubviewType(EMmeInputOperand operand) const;
    unsigned calcSpatialSubviewsNr(EMmeInputOperand operand) const;
    unsigned calcSpatialLength(unsigned subviewIdx, EMmeInputOperand operand) const;

private:
    const RecipeConstants m_constants;
    MmeRecipe m_recipe;  // sub-view will be modified by the splitter
    RecipeGridsCreator m_grids;
    // Maximum number of reuse steps according to selected geometry
    unsigned m_fcdGeoPerReuseNr = 0;
    unsigned m_spGeoPerReuseNr = 0;
#if GAUDI_DEBUG_IGNORE_SB_REUSE_IN_RECIPE
    bool m_ignoreSbReuse = false;
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SBReuse
////////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate steps info for reuse A/B case
class SBReuse
{
public:
    SBReuse(const RecipeConstants& constants,
            const MmeRecipe& recipe,
            const RecipeGrids& grids,
            unsigned commonDimGridIdx,
            unsigned secondOperandSpatialLength = 0);

    void defineCommonDimGridForPartialReuse(CommonDimParams& commonDimParams, unsigned spStepCapacity);
    void defineSpatialGridsForPartialSBReuse(unsigned& fcdStepCapacity, unsigned& spStepCapacity, bool partialToMemory);
    void defineSpatialGridsForNonPartialSBReuse(unsigned& fcdStepCapacity, unsigned& spStepCapacity);
    bool isPartialSBReuse() const;
    static float calcSbUtilization(const MmeHalReader& mmeHal, const MmeLayerParams& params, EMmeInputOperand reusedInput, unsigned elementsPerPortNr);
    static unsigned calcSingleSBSize(const MmeHalReader& mmeHal, const MmeLayerParams& params, unsigned elementsPerPortNr, std::optional<EMmeInputOperand> reusedInput = std::nullopt);
    unsigned calcValidCommonDimAlignment(unsigned val, bool alignToCL, bool roundDown) const;

private:  // This section is only for private methods
    bool isFirstReuse() const { return m_2ndOperandSpatialLength == 0; }
    EMmeInputOperand calcReuseOperand() const;
    static float calcSbUtilization(const MmeHalReader& mmeHal,
                                   const MmeLayerParams& params,
                                   EMmeInputOperand reusedInput,
                                   const MmeTensorView& operand,
                                   unsigned memClElemSize,
                                   unsigned elementsPerPortNr);
    unsigned calcSBSpan() const;
    unsigned calcSBCommonDimSize() const;
    void defineCommonDimGridOnSpForPartialReuse(CommonDimParams& commonDimParams);
    void defineCommonDimGridOnConvOrBatchForPartialReuse(CommonDimParams& commonDimParams, unsigned spStepCapacity);

private:  // This section is only for member variables
    const RecipeConstants& m_constants;
    const unsigned m_commonDimGridIdx;
    const MmeRecipe& m_recipe;
    const RecipeGrids& m_grids;

    const unsigned m_2ndOperandSpatialLength;  // Length of spatial subview of 2D reuse operand
    const EMmeInputOperand m_reuseOpType;  // Reuse A or B
    const unsigned m_sbSize;  // Size of a single SB that can be reused
    const unsigned m_sbSpan;  // Span of reusable SB above common dimension
    const unsigned m_sbCommonDimSize;  // Size of common dimension aligned to cache line to match its occupation in SB
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SubViewSplitter
////////////////////////////////////////////////////////////////////////////////////////////////////

// Create actual subviews of MmeRecipe.
// No member variables in this class. All methods are static.
class SubViewSplitter
{
public:
    SubViewSplitter(MmeRecipe& recipe, const RecipeGrids& grids) : m_recipe(recipe), m_grids(grids) {}
    void split();

private:
    // Utility-class to store maintain a multi-dimensional counter
    class MultiDimCounter
    {
    public:
        MultiDimCounter(const SizeArray& orgSizes, unsigned splitDim);
        ~MultiDimCounter() = default;
        void operator+=(unsigned stepSize);  // Increment
        unsigned operator[](unsigned dim) const { return m_counters[dim]; }  // Get current counter value at a given dim
    private:
        const unsigned m_firstCountingDim;  // The dim where the counting starts
        SizeArray m_limits = {0};  // Limits of each dim
        SizeArray m_counters = {0};  // Counters for each dim
    };

    SingleDimSubViews& getSpatialSubViews(GridType gridType);
    void noSplit(const Grid& grid);
    void split(const GridBase& grid);
    void commonDimNoSplit(const GridBase& grid);
    void commonDimSplitForSP();
    void commonDimSplitForConvAndBatch(const GridBase& grid);
    void commonDimReset();

private:
    MmeRecipe& m_recipe;
    const RecipeGrids& m_grids;
};

}  // namespace MmeCommon

#endif //MME__RECIPE_GENERATOR_H
