#ifndef MME__MME_COMMON_ENUM_H
#define MME__MME_COMMON_ENUM_H

#include "llvm/small_vector.h"
#include "include/general_utils.h"

//============ Common Definitions =======================
#define MME_MAX_TENSOR_DIMS 5
#define MME_MAX_CONV_DIMS 4

#define EXPONENT_BIAS_UFP16_31          31  //  valid bias for ufp16
#define EXPONENT_BIAS_FP16_15           15  //  default bias for fp16
#define EXPONENT_BIAS_FP8_152_15        15
#define EXPONENT_BIAS_FP8_152_MIN_VALUE 1
#define EXPONENT_BIAS_FP8_152_MAX_VALUE 30
#define EXPONENT_BIAS_FP8_143_TYPES     4
#define EXPONENT_BIAS_FP8_143_3         3
#define EXPONENT_BIAS_FP8_143_7         7
#define EXPONENT_BIAS_FP8_143_11        11
#define EXPONENT_BIAS_FP8_143_15        15
//============ Common Enums =======================
namespace MmeCommon
{

enum ChipType
{
    e_mme_Gaudi,
    e_mme_Gaudi2,
    e_mme_Gaudi3,
};

enum EMmeDataType
{
    e_type_first_int_type = 0,
    e_type_int4 = 0,
    e_type_uint4,
    e_type_int8,
    e_type_uint8,
    e_type_int16,
    e_type_uint16,
    e_type_int32,
    e_type_int32_26,
    e_type_int32_16,
    // float types
    e_type_fp8_143,
    e_type_fp8_152,
    e_type_fp16,
    e_type_bf16,
    e_type_fp32,
    e_type_fp32_ieee,
    e_type_tf32,
    e_types_nr,  // must be last

    //  Gaudi3 types, will be moved at a later stage
    e_type_ufp16,  //  unsigned fp16

    // enums values for config parser
    e_type_last_int_type = e_type_fp8_143,
    e_type_first_float_type = e_type_fp8_143,
    e_type_last_output_float_type = e_type_fp32_ieee,
    e_type_last_float_type = e_types_nr,

};

// this enum is align to HW.
// use with caution. change with even more caution.
enum RoundingMode
{
    // common rounding modes
    RoundToNearest = 0,
    RoundToZero = 1,
    RoundUp = 2,
    RoundDown = 3,
    StochasticRounding = 4,

    // only in gaudi2/3
    RoundAwayFromZero = 6,
    StochasticRoundingAndNearest = 7,

    RoundingMode_nr = 8,
};

enum EMmeOpType
{
    //  Conv operations
    e_mme_fwd = 0x0,
    e_mme_dedx,
    e_mme_dedw,
    e_mme_deterministic_dedw,
    e_mme_transposed_dedx,
    //  Bgemm operations
    e_mme_ab,
    e_mme_abt,
    e_mme_atb,
    e_mme_atbt,
    e_mme_gemm_transpose,
    // DMA operations
    e_mme_memcpy,
    e_mme_trans,
    // Reduction add
    e_mme_reductionAdd
};
inline bool isDedwOperation(const EMmeOpType opType)
{
    return (opType == e_mme_dedw || opType == e_mme_deterministic_dedw);
}

enum EMmeOperand
{
    e_mme_op_x = 0x0,  // operand x
    e_mme_op_w,  // operand w
    e_mme_op_y,  // operand y
    e_mme_op_o,  // copy of the output (y in fwd, w in dedw, x in dedx)
};

enum EMmeInternalOperand
{
    e_mme_op_a = 0x0,
    e_mme_op_b = 0x1,
    e_mme_op_c = 0x2,
    e_mme_op_nr = 0x3,
};

enum EMmeGaudiInternalAgu
{
    e_mme_agu_shared,
    e_mme_agu_local,
    e_mme_agu_out,
};

enum EMmeInputOperand
{
    e_mme_in_a = EMmeInternalOperand::e_mme_op_a,
    e_mme_in_b = EMmeInternalOperand::e_mme_op_b
};

enum EMmePrefetch
{
    e_mme_prefetch_none = 0x0,
    e_mme_prefetch_A,
    e_mme_prefetch_B,
    e_mme_prefetch_nr
};

enum EMmeLoopDim
{
    dim_k,
    dim_c,
    dim_s,
    dim_f,
    dim_b = dim_f
};

enum EMmeLoopMask
{
    e_mme_gemm_loop = (1 << 0) - 1,  // Single gemmm
    e_mme_conv_loop_0 = (1 << 1) - 1,  // Single loop of gemms
    e_mme_conv_loop_1 = (1 << 2) - 1,  // Two loops of gemms
    e_mme_conv_loop_2 = (1 << 3) - 1,  // Three loops of gemms.
    e_mme_conv_loop_3 = (1 << 4) - 1,  // four loops of gemms - Every Tetris.
    e_mme_tetris_loop = (1 << 5) - 1,  // Loop of Tetrises
    e_mme_outer_loop = (1 << 6) - 1,  // Two loops of Tetrises
};

inline uint8_t getLoopFromLoopMask(EMmeLoopMask mask)
{
    return (mask + 1) >> 1;
}

inline EMmeLoopMask getLoopMaskfromLoop(uint8_t loopIdx)
{
    MME_ASSERT(loopIdx < 7, "loop overflow");
    return (EMmeLoopMask)((1 << loopIdx) - 1);
}

enum EMmeSignalingMode
{
    e_mme_signaling_none = 0x0,
    e_mme_signaling_once,
    e_mme_signaling_desc,
    e_mme_signaling_desc_with_store,
    e_mme_signaling_chunk,  // signal when the slowest (real) loop increments, row or col according to walking pattern
    e_mme_signaling_output,
    e_mme_signaling_partial,
    e_mme_signaling_nr,

    //  disabling this feature for now, and prone ot bugs.
    e_mme_signaling_amount,  //  signal at least the amount specified in the recipe (including pipeline split)
};

enum InfNanMode
{
    e_mme_full_inf_nan = 0x0,
    e_mme_no_inf_nan,
    e_mme_minimal_inf_nan,
    e_mme_infNan_nr,  // number of modes
};

struct MmeConv
{
#ifdef VERIF_COMP
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> stride = {{1, 1, 1}};
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> dilation = {{1, 1, 1}};
    std::array<int, MME_MAX_CONV_DIMS - 1> padding = {{0, 0, 0}};
#else
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> stride = {1, 1, 1};
    std::array<unsigned, MME_MAX_CONV_DIMS - 1> dilation = {1, 1, 1};
    std::array<int, MME_MAX_CONV_DIMS - 1> padding = {0, 0, 0};
#endif
    float paddingValue = 0;
    unsigned spatialDimsNr = 3;  // How many spatial dimensions are present (1D/2D/3D convolution)
};

struct MmeTensorView
{
    EMmeDataType elementType = e_type_fp32;
#ifdef VERIF_COMP
    SizeArray bases = {{0, 0, 0, 0, 0}};  // the view base coord in the tensor. The pointer in
                                          // the descriptor points this coord.
    SizeArray sizes = {{1, 1, 1, 1, 1}};  // the actual size of the view in elements.
    SizeArray strides = {{1, 1, 1, 1, 1}};  // the view's stride in elements. strides[0] must
                                            // always be set to 1.
#else
    SizeArray bases = {0, 0, 0, 0, 0};  // the view base coord in the tensor. The pointer in
                                        // the descriptor points this coord.
    SizeArray dcoreBases = {0, 0, 0, 0, 0};  // offset for each dcore
    SizeArray sizes = {1, 1, 1, 1, 1};  // the actual size of the view in elements.
    SizeArray strides = {1, 1, 1, 1, 1};  // the view's stride in elements. strides[0] must
                                          // always be set to 1.
#endif

    // Return 'true' if this tensor view is a subview of original tensor (as a result of SRAM slicer).
    // Otherwise it's equal so return 'false'.
    bool isStrided() const
    {
        unsigned curStrides = 1;
        for (unsigned dim = 0; dim < MME_MAX_TENSOR_DIMS; dim++)
        {
            if (curStrides != strides[dim])
            {
                return true;
            }
            curStrides *= sizes[dim];
        }
        return false;
    }

    bool areStridesFullyAligned(unsigned alignVal) const
    {
        for (unsigned i = 1; i < strides.size(); i++)
        {
            if ((strides[i] % alignVal) != 0)
            {
                return false;
            }
        }
        return true;
    }
};

enum PmuConfig
{
    PMUCFGNONE,
    PMUCFGMODE1,
    PMUCFGMODE2,
    PMUCFGMODE3,
    PMUCFGMODE4
};

struct MmeControls
{
    RoundingMode roundingMode = RoundToNearest;  // EU rounding mode
    RoundingMode conversionRoundingMode = RoundToNearest;  // AP (output conversion) rounding mode
    RoundingMode accRoundingMode = RoundToZero;  // ACC accumulation rounding mode
    EMmeSignalingMode signalingMode = e_mme_signaling_output;  // the signaling mode
    unsigned signalAmount = 1;  // amount of signals required when using signal_amount mode
    bool slaveSignaling = false;  // slave will signal independently.
    bool useSameColorSet = true; // use same or different color-sets for duplicate outputs - relevant for gaudi2 onwards
    bool atomicAdd = false;  // accum the output in the DRAM
    bool squashIORois = false;  // squash all the IO ROIs to the first activation
    bool reluEn = false;
    bool flushDenormals = false;
    bool stochasticFlush = false;
    bool sbCacheEn = true;
    //  TODO change name, keeping to not break synapse
    unsigned fp8BiasIn = EXPONENT_BIAS_FP8_152_15;  // user defined bias for fp8 input.
    unsigned fp8BiasIn2 = EXPONENT_BIAS_FP8_152_15;  // user defined bias for fp8 input.
    unsigned fp8BiasOut = EXPONENT_BIAS_FP8_143_7;  // user defined bias for fp8 output.
    InfNanMode infNanModeA = InfNanMode::e_mme_full_inf_nan;
    InfNanMode infNanModeB = InfNanMode::e_mme_full_inf_nan;
    InfNanMode infNanModeOut = InfNanMode::e_mme_full_inf_nan;
    unsigned pmuSaturationVal = 0;  // PMU rate val. Zero means disabled
    int sbSizeInCLs = 0;  // debug feature - limit sb size to trigger partials on smaller tests
    bool clippingEn = false;
    bool clipInfIn = false;
    uint16_t structPadding = 0;
};

// The values are used in gaudi1 to set the associated dims
enum EMmePattern
{
    // dedw/bgemm patterns
    e_mme_sp_reduction_kfc = 0x0,
    e_mme_sp_reduction_fkc,
    e_mme_sp_reduction_fck,
    e_mme_sp_reduction_cfk,
    e_mme_sp_reduction_kcf,
    e_mme_sp_reduction_ckf,
    // fwd/dedx/dma patterns
    e_mme_z_reduction_ksf,
    e_mme_z_reduction_skf,

    // enums values for config parser
    e_mme_patterns_nr,
    e_patterns_first_dedw = e_mme_sp_reduction_kfc,
    e_patterns_last_dedw = e_mme_z_reduction_ksf,
    e_patterns_first_fwd = e_mme_z_reduction_ksf,
    e_patterns_last_fwd = e_mme_patterns_nr,
};

typedef enum
: uint8_t
{
    e_mme_reduction_add = 0x0,
    e_mme_reduction_sub = 0x1,
    e_mme_reduction_min = 0x2,
    e_mme_reduction_max = 0x3,
    e_mme_reduction_max_0 = 0x4,
    e_mme_reduction_none = 0x5,  // default
    e_mme_reduction_nr,

} EMmeReductionOp;

typedef enum
: uint8_t
{
    e_mme_reduction_round_half_to_nearest_even = 0x0,
    e_mme_reduction_round_to_zero = 0x1,
    e_mme_reduction_round_up = 0x2,
    e_mme_reduction_round_down = 0x3,
    e_mme_reduction_round_nr,
} EMmeReductionRm;

typedef enum
: uint8_t
{
    SkipCache,
    NoAllocate,
    HomeAllocate,
    DcoreAllocate,
    SharedAllocate,
    NR,
} EMmeCacheDirective;

typedef enum
: uint8_t
{
    Low = 0x0,
    Normal = 0x1,
    High = 0x2,
    Reserved = 0x3,
} EMmeCacheClass;

typedef enum
: uint8_t
{
    Bucket0 = 0x0,
    Bucket1 = 0x1,
    Bucket2 = 0x2,
    Bucket3 = 0x3,
    BucketBP = 0xF,
} EMmeCacheQOS;
// Geometry is still not generalized, so currently it is still Gaudi2 specific
// todo AlonG: generalize the geometry enum for all platforms
typedef enum
{
    //  Gaudi - TODO combine with Gaudi2 geometries
    e_mme_geometry_4wx1h,
    e_mme_geometry_2wx2h,
    e_mme_geometry_1wx4h,
    //  Gaudi2
    e_mme_geometry_4xw,
    e_mme_geometry_2xw,
    e_mme_geometry_2xh,
    e_mme_geometry_4xh,

    // enums values for config parser
    e_mme_geometry_nr,
    e_first_geometry_gaudi2 = e_mme_geometry_4xw,
    e_last_geometry_gaudi2 = e_mme_geometry_nr,
} EMmeGeometry;

typedef struct _MmeGeometryGrid
{
    unsigned width = 0;
    unsigned height = 0;
    unsigned batch = 0;
} MmeGeometryGrid;

typedef enum
{
    RL_NONE,
    RL_PARTIAL,
    RL_ALL
} EMmeRateLimiter;

typedef enum
{
    e_mme_dump_none,
    e_mme_dump_single,
    e_mme_dump_all,
} EMmeDump;

typedef enum
{
    TurnedOff,
    TurnedOn,
    Undefined
} BoolWithUndef;

struct MmeStrategy
{
    EMmeGeometry geometry = e_mme_geometry_2xh;
    unsigned mmeLimit = 0;  // generate descriptors for fewer MMEs
    EMmePattern pattern = e_mme_z_reduction_skf;
    unsigned pipelineLevel = 1;  // number of minimal requested descriptors
    unsigned packingFactor = 1;
    unsigned reductionLevel = 1;    // Number of slices for reduction add operation

    bool loweringEn = true;  // allow mme lowering. (fwd, dedw and transposed dedx)
    bool teAccelerationEn = false;  // allow TE acceleration
    bool sbReuse = false;  // allow SB reuse.
    bool partialsToMemoryEn = false;  // allow performing partials to memory instead of accumulating in ACCs
    bool alignedAddresses = false;
    bool unrollEn = false;  // allow weight unrolling (dedw)
    bool dedwAsBgemmEn = false;  // allow mapping dedw to bgemm
    bool recurringMisalignmentOptEn = false;  // apply optimization that addresses recurring misalignments
    BoolWithUndef batchConcurrencyEn = TurnedOff;
    BoolWithUndef cdConcurrencyEn = TurnedOff;
    bool isDeterministic = false;
    bool flattenEn = true;  // flatten the tensors in case of Gemm-convertible node
    bool dualGemm = false;  // build dualGemm descriptor for this node, valid only for Gaudi3
    bool partial = false;  // allow partials.
    bool signalPartial = false;  // signal after every partial chunk
    bool memsetDedxVoidPixels = true;  // memset pixels in x that do not participate in the FWD conv (dedx)
    bool dedxDynamicPadding = false;  //  align dedx sub-problems to ease padding patching
    bool maskedBgemm = false;  //  add a mask using the two auxiliary tensors
    std::string print() const;
};

enum EMmeTraceMode
{
    e_mme_trace_mode_none = 0x0,
    e_mme_trace_mode_layer_act,  // start event: A first desc. End event: Cout last desc
    e_mme_trace_mode_desc,  // start event: A. End event Cout - on each descriptor.
    e_mme_trace_mode_advanced,  // for all engine send events on start and finish
    e_mme_trace_mode_nr  // must be last.
};

struct MmeTracing
{
    EMmeTraceMode traceMode = e_mme_trace_mode_none;
    EMmeTraceMode traceModeX = e_mme_trace_mode_layer_act;
    EMmeTraceMode traceModeY = e_mme_trace_mode_layer_act;
    EMmeTraceMode traceModeW = e_mme_trace_mode_layer_act;
    uint16_t ctxId = 0;
    uint16_t structPadding = 0;
};

enum EMmeTraceEngine
{
    e_mme_trace_input,
    e_mme_trace_output,
    e_mme_trace_eu,
};

enum EMmeTraceOperandMask
{
    e_mme_operand_0 = (1 << 0),
    e_mme_operand_1 = (1 << 1),
    e_mme_operand_2 = (1 << 2),
    e_mme_operand_3 = (1 << 3),
    e_mme_operand_4 = (1 << 4),
};

struct MmeMemoryConfig
{
    // reduction config
    EMmeReductionOp reductionOp = EMmeReductionOp::e_mme_reduction_none;
    EMmeReductionRm reductionRm = EMmeReductionRm::e_mme_reduction_round_down;
    bool reductionEn() const { return reductionOp != EMmeReductionOp::e_mme_reduction_none; }
    // cache config - applicable for gaudi3 and up
#ifdef VERIF_COMP
    std::array<EMmeCacheDirective, e_mme_op_nr> cacheDirective = {{EMmeCacheDirective::DcoreAllocate}};
    std::array<EMmeCacheClass, e_mme_op_nr> clss = {{EMmeCacheClass::Normal}};
    std::array<EMmeCacheQOS, e_mme_op_nr> qos = {{EMmeCacheQOS::Bucket1}};  // TODO: change according to new specs.
    std::array<uint16_t, e_mme_op_nr> mcId = {{0}};
#else
    std::array<EMmeCacheDirective, e_mme_op_nr> cacheDirective = {EMmeCacheDirective::DcoreAllocate};
    std::array<EMmeCacheClass, e_mme_op_nr> clss = {EMmeCacheClass::Normal};
    std::array<uint16_t, e_mme_op_nr> mcId = {0};
    std::array<EMmeCacheQOS, e_mme_op_nr> qos = {EMmeCacheQOS::Bucket1};  // TODO: change according to new specs.
    uint8_t structPadding = 0;
#endif
};

// Number of batch dims
static constexpr unsigned c_batchDimNr = 3;

struct MultiDimSubView
{
#ifdef VERIF_COMP
    SizeArray bases = {{0}};  // the sub-view base coord in the tensor relative to tensor view B.
    SizeArray sizes = {{0}};  // the actual size of the sub-view in elements.
#else
    SizeArray bases = {0};  // the sub-view base coord in the tensor relative to tensor view B.
    SizeArray sizes = {0};  // the actual size of the sub-view in elements.
#endif
    bool operator==(const MultiDimSubView& other) const { return bases == other.bases && sizes == other.sizes; }
};
using MultiDimSubViews = llvm_vecsmall::SmallVector<MultiDimSubView, 1>;

struct SingleDimSubView
{
    unsigned viewBase = 0;
    unsigned viewSize = 0;
    unsigned viewOrigSize = 0;

    bool operator==(const SingleDimSubView& other) const
    {
        return viewBase == other.viewBase && viewSize == other.viewSize && viewOrigSize == other.viewOrigSize;
    }
};
using SingleDimSubViews = llvm_vecsmall::SmallVector<SingleDimSubView, 1>;

struct OffsetArray
{
    std::array<unsigned, MME_MAX_TENSOR_DIMS> xOffset = {0};
    std::array<unsigned, MME_MAX_TENSOR_DIMS> wOffset = {0};
    std::array<unsigned, MME_MAX_TENSOR_DIMS> yOffset = {0};
};

using MultiplierArray = std::vector<unsigned>;

inline EMmeLoopMask pattern2LoopMask(EMmePattern pattern, EMmeLoopDim loopDim)
{
    switch (loopDim)
    {
        case EMmeLoopDim::dim_k:
            switch (pattern)
            {
                case e_mme_sp_reduction_cfk:
                case e_mme_sp_reduction_fck:
                    return e_mme_conv_loop_0;
                case e_mme_sp_reduction_fkc:
                    return e_mme_conv_loop_1;
                case e_mme_sp_reduction_ckf:
                case e_mme_z_reduction_skf:
                    return e_mme_conv_loop_3;
                case e_mme_sp_reduction_kfc:
                case e_mme_sp_reduction_kcf:
                case e_mme_z_reduction_ksf:
                    return e_mme_outer_loop;
                default:
                    MME_ASSERT(0, "invalid reduction type");
            }
        case EMmeLoopDim::dim_c:
            switch (pattern)
            {
                case e_mme_sp_reduction_kfc:
                case e_mme_sp_reduction_fkc:
                    return e_mme_conv_loop_0;
                case e_mme_sp_reduction_fck:
                    return e_mme_conv_loop_1;
                case e_mme_sp_reduction_kcf:
                    return e_mme_conv_loop_3;
                case e_mme_sp_reduction_ckf:
                case e_mme_sp_reduction_cfk:
                    return e_mme_outer_loop;
                case e_mme_z_reduction_ksf:
                case e_mme_z_reduction_skf:
                default:
                    MME_ASSERT(0, "invalid reduction type");
            }
        case EMmeLoopDim::dim_s:
            return e_mme_tetris_loop;
        case EMmeLoopDim::dim_f:
            switch (pattern)
            {
                case e_mme_sp_reduction_kcf:
                case e_mme_sp_reduction_ckf:
                case e_mme_z_reduction_ksf:
                case e_mme_z_reduction_skf:
                    return e_mme_conv_loop_0;
                case e_mme_sp_reduction_kfc:
                case e_mme_sp_reduction_cfk:
                    return e_mme_conv_loop_1;
                case e_mme_sp_reduction_fkc:
                case e_mme_sp_reduction_fck:
                    return e_mme_conv_loop_2;
                default:
                    MME_ASSERT(0, "invalid reduction type");
            }
    }

    MME_ASSERT(0, "should not get here");
    return (EMmeLoopMask) -1;
}

//=============== Small util functions =======================
inline bool isTypeInteger(EMmeDataType dt)
{
    return (dt >= e_type_first_int_type && dt < e_type_last_int_type);
}
inline bool isTypeFp8(EMmeDataType dt)
{
    return ((dt == e_type_fp8_143) || (dt == e_type_fp8_152));
}
inline bool isTypeFp16(EMmeDataType dt)
{
    return ((dt == e_type_fp16) || (dt == e_type_ufp16));
}

// Return log2 of the size of given data type in bytes
inline uint8_t getLogElementSize(EMmeDataType dataType)
{
    switch (dataType)
    {
        case e_type_fp8_143:
        case e_type_fp8_152:
            return 0;
        case e_type_ufp16:
        case e_type_fp16:
        case e_type_bf16:
            return 1;
        case e_type_fp32:
        case e_type_tf32:
        case e_type_fp32_ieee:
            return 2;
        default:
            MME_ASSERT(0, "invalid data type");
    }
    return 0;
}

// Return size of given data type in bytes
inline uint8_t getElementSize(EMmeDataType dataType)
{
    return 1 << getLogElementSize(dataType);
}

// Convert EMmeOperand to EMmeInternalOperand
inline EMmeInternalOperand mmeOpToInternalOp(EMmeOperand operand, EMmeOpType opType)
{
    switch (operand)
    {
        case e_mme_op_x:
            return (opType == e_mme_dedx || opType == e_mme_transposed_dedx) ? e_mme_op_c : e_mme_op_a;
        case e_mme_op_w:
            return isDedwOperation(opType) ? e_mme_op_c : e_mme_op_b;
        case e_mme_op_y:
            if (opType == e_mme_dedx || opType == e_mme_transposed_dedx) return e_mme_op_a;
            else if (isDedwOperation(opType))
                return e_mme_op_b;
            else
                return e_mme_op_c;
        case e_mme_op_o:
            return e_mme_op_c;
        default:
            MME_ASSERT(0, "invalid operand");
            return e_mme_op_c;
    }
}

inline EMmeGaudiInternalAgu mmeOperandToGaudiAgu(EMmeOperand operand, EMmeOpType opType, bool transO)
{
    // agu_shared is effectively operand A and agu_local is operand B,
    // only when transpose the output we swap so operand B is agu_shared and operand A is agu_local.
    // output transpose happens on pair geometry 2xh -> 1wx4h or 2x2xh (bgemm_2x)
    MME_ASSERT(opType != e_mme_trans && opType != e_mme_memcpy, "only Gaudi operations are supported by this method");
    bool isFwdOrBgemm = (!isDedwOperation(opType)) && (opType != e_mme_dedx);
    switch (operand)
    {
        case e_mme_op_x:
            if (opType == e_mme_dedx)
            {
                return e_mme_agu_out;
            }
            return transO ? e_mme_agu_local : e_mme_agu_shared;
            break;
        case e_mme_op_w:
            if (isDedwOperation(opType))
            {
                return e_mme_agu_out;
            }
            return transO ? e_mme_agu_shared : e_mme_agu_local;
            break;
        case e_mme_op_y:
            if (isFwdOrBgemm)
            {
                return e_mme_agu_out;
            }
            else if (opType == e_mme_dedx)
            {
                return transO ? e_mme_agu_local : e_mme_agu_shared;
            }
            else if (isDedwOperation(opType))
            {
                return transO ? e_mme_agu_shared : e_mme_agu_local;
            }
            break;
        default:
        {
            MME_ASSERT(0, "invalid operand");
        }
    }
    return e_mme_agu_out;
}

// Convert EMmeInternalOperand to EMmeOperand. Note: there is no mapping to secondary output
inline EMmeOperand internalOpToMmeOp(EMmeInternalOperand operand, EMmeOpType opType)
{
    if (isDedwOperation(opType))
    {
        static const EMmeOperand op[3] = { e_mme_op_x, e_mme_op_y, e_mme_op_w };
        return op[operand];
    }
    if (opType == e_mme_dedx || opType == e_mme_transposed_dedx)
    {
        static const EMmeOperand op[3] = { e_mme_op_y, e_mme_op_w, e_mme_op_x };
        return op[operand];
    }
    static const EMmeOperand op[3] = { e_mme_op_x, e_mme_op_w, e_mme_op_y };
    return op[operand];
}

// Check if the given operand is transposed by HW
inline bool isTransposed(EMmeOpType opType, EMmeInputOperand operand)
{
    switch (opType)
    {
        case e_mme_trans:
        case e_mme_fwd:
        case e_mme_ab:
        case e_mme_reductionAdd:
        case e_mme_transposed_dedx:
            return (operand == e_mme_in_a);
        case e_mme_dedx:
        case e_mme_abt:
            return true;
        case e_mme_gemm_transpose:
        case e_mme_memcpy:
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
        case e_mme_atb:
            return false;
        case e_mme_atbt:
            return (operand == e_mme_in_b);
        default:
            MME_ASSERT(0, "invalid operation type");
    }
    return false;
}

// Raster walk refers to GEMM calculations order (of the output).
// Assume FCD (relative to output) has multiple views {fcdV1, fcdV2, ..., fcdVn}.
// We call the pattern raster when GEMM1 consumes fcdV1, GEMM2 consumes fcdV2,.... GEMMn consumes fcdVn.
// Otherwise we call it non-raster.
// (K) associated to FCD (relative to output).
// (C)(S) associated to SP (spatial, relative to output). (C) Used for DEDW. (S) used for FWD and DEDX.
// (F) associated to the three filters.
//
// Example 1 - CFK:
// (K) First scan FCD subview of filter 1 then move to second filter (F). Before moving to (C)
//
// Example 2 - CKF:
// (F) First scan first GEMM of filter 1, then move to first GEMM of filter 2...
// After scanning all filters, go back to first one and move to second GEMM at FCD direction (K)
// When complete scanning all GEMMS at first rows of all filters, move to second raw (C) and perform previous process.
//
// Example 3 - KCF:
// (F) First scan first GEMM of filter 1, then move to first GEMM of filter 2... (same as previous example).
// After scanning all filters, go back to first one and move to second GEMM at SP direction (C)
// When complete scanning all GEMMS at first columns of all filters, move to second column (K) and repeat as before.
inline bool isPatternRaster(EMmePattern pattern)
{
    switch (pattern)
    {
        case e_mme_z_reduction_skf:
        case e_mme_sp_reduction_fck:
        case e_mme_sp_reduction_cfk:
            return true;
        case e_mme_sp_reduction_ckf:
        case e_mme_sp_reduction_kfc:
        case e_mme_sp_reduction_fkc:
        case e_mme_sp_reduction_kcf:
        case e_mme_z_reduction_ksf:
            return false;
        default:
            MME_ASSERT(0, "invalid walking pattern");
            return false;
    }
}

//==============================================
struct MmeLayerParams
{
    std::string nodeName;
    EMmeOpType opType = e_mme_fwd;
    MmeTensorView x;
    MmeTensorView y;
    MmeTensorView w;
    //  auxiliary tensors
    MmeTensorView xAux;
    MmeTensorView yAux;
    MmeTensorView wAux;

    unsigned spBase = 0;
    unsigned spSize = 0;

    MmeConv conv;
    MmeControls controls;
    MmeStrategy strategy;
    MmeTracing tracing;
    MmeMemoryConfig memoryCfg;
    bool useDescCache = true;
    uint8_t permutation[MME_MAX_TENSOR_DIMS] = {0, 1, 2, 3, 4};

    bool inline isUsingDescCache() const { return useDescCache; }
    bool inline isConvOperation() const { return (isFwdOrDedx() || isDedwOperation()); }
    bool inline isGemmOperation() const
    {
        static_assert(MmeCommon::c_batchDimNr == 3, "Need to add support for a new number of batches.");
        return (opType == e_mme_ab || opType == e_mme_atb || opType == e_mme_abt || opType == e_mme_atbt ||
                opType == e_mme_reductionAdd || opType == e_mme_gemm_transpose);
    }
    bool inline isDedwCdConcurrency() const
    {
        return ((isDedwOperation()) && (strategy.cdConcurrencyEn == TurnedOn));
    }
    bool inline isDedxOperation() const { return (opType == e_mme_dedx || opType == e_mme_transposed_dedx); }
    bool inline isFwdOrDedx() const { return (opType == e_mme_fwd || isDedxOperation()); }
    bool inline isDedwOperation() const { return (opType == e_mme_dedw || opType == e_mme_deterministic_dedw); }
    bool inline isDedwOrGemm() const { return (isDedwOperation() || isGemmOperation()); }
    bool inline isDmaOperation() const { return (isNativeDmaOperation() || isGemmDmaOperation()); }
    bool inline isNativeDmaOperation() const { return (opType == e_mme_memcpy || opType == e_mme_trans); }
    bool inline isGemmDmaOperation() const { return opType == e_mme_gemm_transpose; }
    bool inline isReductionAdd() const { return (opType == e_mme_reductionAdd);}
    // Map input/output operands X/Y/W to A/B/C view form - depends on operation.
    // For more details see readme section in recipe_generator.h
    const inline EMmeOperand getExternalOperand(EMmeInternalOperand operand) const
    {
        switch (opType)
        {
            case e_mme_memcpy:
            case e_mme_trans:
            case e_mme_gemm_transpose:
            case e_mme_fwd:
            case e_mme_atbt:
            case e_mme_ab:
            case e_mme_atb:
            case e_mme_abt:
            case e_mme_reductionAdd:
                if (operand == e_mme_op_a) return e_mme_op_x;
                if (operand == e_mme_op_b) return e_mme_op_w;
                if (operand == e_mme_op_c) return e_mme_op_y;
                break;
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                if (operand == e_mme_op_a) return e_mme_op_y;
                if (operand == e_mme_op_b) return e_mme_op_w;
                if (operand == e_mme_op_c) return e_mme_op_x;
                break;
            case e_mme_dedw:
            case e_mme_deterministic_dedw:
                if (operand == e_mme_op_a) return e_mme_op_x;
                if (operand == e_mme_op_b) return e_mme_op_y;
                if (operand == e_mme_op_c) return e_mme_op_w;
                break;
            default:
                MME_ASSERT(0, "should not get here");
        }
        return e_mme_op_x;
    }

    inline MmeTensorView& getOperand(EMmeInternalOperand operand, bool primaryTensor = true)
    {
        return getOperand(getExternalOperand(operand), primaryTensor);
    }

    inline MmeTensorView& getOperand(EMmeOperand operand, bool primaryTensor = true)
    {
        switch (operand)
        {
            case e_mme_op_x:
                return primaryTensor ? x : xAux;
            case e_mme_op_w:
                return primaryTensor ? w : wAux;
            case e_mme_op_y:
            case e_mme_op_o:
                break;
            default:
                MME_ASSERT(0, "invalid operand");
        }
        return y;
    }

    const inline MmeTensorView& getOperand(EMmeInternalOperand operand, bool primaryTensor = true) const
    {
        return getOperand(getExternalOperand(operand), primaryTensor);
    }

    const inline MmeTensorView& getOperand(EMmeOperand operand, bool primaryTensor = true) const
    {
        switch (operand)
        {
            case e_mme_op_x:
                return primaryTensor ? x : xAux;
            case e_mme_op_w:
                return primaryTensor ? w : wAux;
            case e_mme_op_y:
            case e_mme_op_o:
                break;
            default:
                MME_ASSERT(0, "invalid operand");
        }
        return y;
    }

    inline bool canLower() const
    {
        if (!isLoweringEnabled() ||
            // Lowering is relevant only to conv operations
            !(isDedwOperation() || opType == e_mme_fwd || opType == e_mme_transposed_dedx))
        {
            return false;
        }
        // Data must be continuous in memory
        auto& aView = getOperand(e_mme_op_a);

        return (conv.dilation[0] == 1) && (aView.strides[1] == w.sizes[1]) &&
               (w.strides[2] == w.strides[1] * w.sizes[1]);
    }

    inline bool canFlatten() const
    {
        if (!isFlatteningEnabled() || strategy.dualGemm || strategy.maskedBgemm ||
            // Flattening is relevant only to A transposed bgemm ops
            !(opType == e_mme_ab || opType == e_mme_abt))
        {
            return false;
        }
        // Check that first batch dim of B is broadcasted, and that we have batches in C
        // as an extra precaution check that C and A batches are equal
        if (w.sizes[2] != 1 || y.sizes[2] == 1 || y.sizes[2] != x.sizes[2])
        {
            return false;
        }
        // Data must be continuous in memory
        if (x.strides[2] != (x.strides[1] * x.sizes[1]))
        {
            return false;
        }
        if (y.strides[2] != (y.strides[1] * y.sizes[1]))
        {
            return false;
        }
        return true;
    }

    inline unsigned getFcdSize() const
    {
        auto& cView = getOperand(e_mme_op_c);
        return cView.sizes[0];
    }

    inline unsigned getSpatialSize() const
    {
        auto& cView = getOperand(e_mme_op_c);
        if (isGemmOperation())
        {
            if (isGemmDmaOperation())
            {
                return getOperand(e_mme_op_a).sizes[0];
            }
            if (canFlatten())
            {
                return cView.sizes[1] * cView.sizes[2];
            }
            else
            {
                return cView.sizes[1];
            }
        }
        else if (isDedwOperation())
        {
            if (canLower())
            {
                return cView.sizes[1] * cView.sizes[2];
            }
            else
            {
                return cView.sizes[1];
            }
        }
        else
        {
            return multiplyElements(cView.sizes.begin() + 1, cView.sizes.end());
        }
    }

    inline unsigned getBatchSize(unsigned concurrency = 1) const
    {
        if (isFwdOrDedx() || isReductionAdd()) return 1;

        //  TODO take bgemm 2x and unroll into account
        auto& cView = getOperand(e_mme_op_c);
        if (canFlatten())
        {
            return cView.sizes[3] * cView.sizes[4];
        }
        else if (canLower())
        {
            unsigned firstBatch = div_round_up(cView.sizes[3], concurrency);
            return firstBatch * cView.sizes[4];
        }
        else
        {
            unsigned firstBatch = div_round_up(cView.sizes[2], concurrency);
            return firstBatch * cView.sizes[3] * cView.sizes[4];
        }
    }

    //  returns the CD of a single Gemm loop
    const inline unsigned getSingleGemmCD() const
    {
        switch (opType)
        {
            case e_mme_fwd:
                return canLower() ? (x.sizes[0] * std::min(w.sizes[2], x.sizes[1])) : x.sizes[0];
                break;
            case e_mme_transposed_dedx:
                return canLower() ? (y.sizes[0] * std::min(w.sizes[2], y.sizes[1])) : y.sizes[0];
            case e_mme_dedx:
                return y.sizes[0];
            default:
                return getCDSize();
        }
    }

    //  returns the overall CD of all accumulated loops
    const inline unsigned getCDSize() const
    {
        switch (opType)
        {
            case e_mme_fwd:
                // CD = C * S * R * Q
                return x.sizes[0] * w.sizes[2] * w.sizes[3] * w.sizes[4];
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                // CD = K * S * R * Q
                return y.sizes[0] * w.sizes[2] * w.sizes[3] * w.sizes[4];
            case e_mme_dedw:
            case e_mme_deterministic_dedw:
                // CD = B * D * H * W
                return y.sizes[1] * y.sizes[2] * y.sizes[3] * y.sizes[4];
            case e_mme_ab:
            case e_mme_abt:
            case e_mme_reductionAdd:
                return x.sizes[0];
            case e_mme_atb:
            case e_mme_atbt:
                return x.sizes[1];
            case e_mme_gemm_transpose:
                return x.sizes[0];
            default:
                break;
        }
        return 0;  // DMA ops.
    }
    bool isSbReuse() const { return strategy.sbReuse && !isDmaOperation(); }
    bool isLoweringEnabled() const { return strategy.loweringEn; }
    bool isFlatteningEnabled() const { return strategy.flattenEn; }
    bool isPatternRaster() const { return MmeCommon::isPatternRaster(strategy.pattern); }
    const bool isDeterministicCdConcurrency() const
    {
        return (opType == e_mme_deterministic_dedw) ||
               (opType == e_mme_dedw && strategy.isDeterministic && strategy.cdConcurrencyEn == TurnedOn);
    }
    EMmeGeometry getGeometry() const { return strategy.geometry; }
    EMmePattern getPattern() const { return strategy.pattern; }
    unsigned getPipelineLevel() const { return strategy.pipelineLevel; }
    unsigned getPackingFactor() const { return strategy.packingFactor; }
    unsigned getReductionLevel() const { return strategy.reductionLevel; }
    unsigned getSignalAmount() const { return controls.signalAmount; }
    EMmeSignalingMode getSignalingMode() const { return controls.signalingMode; }
    bool getAlignedAddress() const { return strategy.alignedAddresses; }
    bool operator<(const MmeLayerParams& rhs) const;
    bool operator==(const MmeLayerParams& rhs) const;

    inline void setPattern(bool downFirst)
    {
        EMmePattern pattern;
        switch (opType)
        {
            default:
                MME_ASSERT(0, "invalid operation");
            case e_mme_ab:
            case e_mme_atb:
            case e_mme_abt:
            case e_mme_atbt:
            case e_mme_reductionAdd:
                pattern = downFirst ? e_mme_sp_reduction_fkc : e_mme_sp_reduction_fck;
                break;
            case e_mme_dedw:
            case e_mme_deterministic_dedw:
                pattern = downFirst ? e_mme_sp_reduction_kfc : e_mme_sp_reduction_fck;
                break;
            case e_mme_fwd:
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                pattern = downFirst ? e_mme_z_reduction_ksf : e_mme_z_reduction_skf;
                break;
        }
        strategy.pattern = pattern;
    }

protected:
    MmeLayerParams() {}
};

}  // namespace MmeCommon

#endif //MME__MME_COMMON_ENUM_H
