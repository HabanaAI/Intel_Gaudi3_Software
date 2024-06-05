/*****************************************************************************
 * Copyright (C) 2018 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Keren Luzon <kluzon@gaudilabs.com>
 ******************************************************************************
 */

/*
 * NOTE: This file is parsed automatically. Please use the following structure:
 *
 *  Kernel name: kernel1, kernel2, kernel3, etc..
 *
 * IMPORTANT:Do not write anything after "Kernel name: " that is not a kernel guid.
 *
 * namespace ns_someNamespace
 * {
 *      struct Params
 *      {
 *          ...
 *      };
 * }
 *
 *  Only use 1 struct per namespace. Inheritance is ok, but not multiple inheritance.
 */

#ifndef PERF_LIB_LAYER_PARAMS_H
#define PERF_LIB_LAYER_PARAMS_H

#define ASSERT_SIZE_IS_GREATER(P1, P2) \
static_assert(sizeof(P1) > sizeof(P2), "sizeof(" #P1 ") > sizeof(" #P2 ")")

typedef union
{
    float f;
    int i;
} fint_t;

enum EPoolingConvention
{
    POOLING_CONVENTION_VALID = 0,
    POOLING_CONVENTION_FULL = 1,
    POOLING_CONVENTION_SAME = 2,
    POOLING_CONVENTION_FULL_PYTORCH = 3
};
typedef enum _UpsampleType_t
{
    UPSAMPLE_TYPE_BILINEAR = 0,
    UPSAMPLE_TYPE_NEAREST_NEIGHBOR = 1,
    UPSAMPLE_TYPE_TRILINEAR = 2
} EUpsampleType_t;

typedef enum _SparseLengthsSumType_t
{
    EMBEDDED_SC_ZP = 0,
    SEPARATE_SC_ZP
} ESparseLengthsSumType_t;

typedef enum _GridGeneratorType_t
{
    GRID_GENERATOR_TYPE_AFFINE = 0,
    GRID_GENERATOR_TYPE_WARP
} EGridGeneratorType_t;

typedef enum _CrossEntropyMode_t
{
    CROSS_ENTROPY_MODE_SUM = 0,
    CROSS_ENTROPY_MODE_MEAN = 1,
    // CROSS_ENTROPY_MODE_NO_REDUCTION mode is only applicable in fwd non-sparse
    // (dense) version
    CROSS_ENTROPY_MODE_NO_REDUCTION = 2
} ECrossEntropyMode_t;

typedef enum _CastF32RoundMode_t
{
    CAST_ROUND_HALF_NE = 0,
    CAST_ROUND_DOWN = 1,
    CAST_ROUND_UP = 2,
    CAST_ROUND_SR = 3,
    CAST_ROUND_ZERO = 4,
    CAST_ROUND_DEFAULT = 5,
    CAST_ROUND_HALF_AZ = 6,
    // Place holder for SR_RNE since it is supported in HW.
    //CAST_ROUND_SR_RNE = 7,
    CAST_ROUND_SFTZ = 8
} CastF32RoundMode_t;

typedef enum _CastSatMode_t
{
    CAST_NO_CLIP_NO_SAT = 0,
    CAST_CLIP = 1,
    CAST_SAT = 2
} CastSatMode_t;

typedef enum _ResizeInterpolationMode_t
{
    RESIZE_INTER_NEAREST = 0,
    RESIZE_INTER_LINEAR = 1,
    RESIZE_INTER_CUBIC = 2,
    RESIZE_INTER_AREA = 3,
    RESIZE_INTER_GAUSSIAN = 4
} ResizeInterpolationMode_t;

typedef enum _GridSampleInterpolation_t
{
    SAMPLE_NEAREST = 0,
    SAMPLE_BILINEAR = 1,
    SAMPLE_CUBIC = 2
} GridSampleInterpolation_t;

typedef enum _GridSamplePad_t
{
    PAD_ZEROS = 0,
    PAD_BORDER = 1,
    PAD_REFLECTION = 2
} GridSamplePad_t;

typedef enum _ResizeNearestMode_t
{
    // versions older than 11 did not have nearest_mode attr.
    // ROUND_DEFAULT is used to maintain backward compatibility
    ROUND_DEFAULT = 0,
    ROUND_PREFER_FLOOR = 1,
    ROUND_PREFER_CEIL = 2,
    FLOOR = 3,
    CEIL = 4
} ResizeNearestMode_t;

typedef enum _ResizeCoordinateTransformationMode_t
{
    HALF_PIXEL_MODE = 0,
    PYTORCH_HALF_PIXEL_MODE = 1,
    ALIGN_CORNERS_MODE = 2,
    ASYMMETRIC_MODE = 3,
    TF_HALF_PIXEL_FOR_NN_MODE = 4
} ResizeCoordinateTransformationMode_t;

typedef enum _RoiAlignMode_t
{
    ROI_ALIGN_AVG = 0,
    ROI_ALIGN_MAX = 1
} RoiAlignMode_t;

typedef enum _SortDirection_t
{
    SORT_ASCENDING = 0,
    SORT_DESCENDING = 1,
} SortDirection_t;

typedef enum _CropAndResizeMode_t
{
    CROP_AND_RESIZE_MODE_BILINEAR = 0,
    CROP_AND_RESIZE_MODE_NEAREST = 1
} CropAndResizeMode_t;

typedef enum
{
    EMBEDDING_BAG_MODE_SUM = 0,
    EMBEDDING_BAG_MODE_MEAN
} EmbeddingBagMode_t;

typedef enum
{
    BINARY_WITH_ALPHA_MODE_ADD = 0,
    BINARY_WITH_ALPHA_MODE_SUB = 1,
    BINARY_WITH_ALPHA_MODE_RSUB = 2,
    BINARY_WITH_ALPHA_MODE_CMUL = 3,
    BINARY_WITH_ALPHA_MODE_CDIV = 4
} BinaryWithAlphaMode_t;

typedef enum
{
    LEFT = 0,
    RIGHT = 1
} ShiftDir_t;

typedef enum
{
    ARITHMETIC = 0,
    LOGICAL
} ShiftMode_t;

typedef enum
{
    NLL_LOSS_MODE_MEAN = 0,
    NLL_LOSS_MODE_SUM,
    NLL_LOSS_MODE_NONE
} NLLLossMode_t;

typedef enum
{
    MSE_LOSS_REDUCTION_MODE_MEAN = 0,
    MSE_LOSS_REDUCTION_MODE_SUM,
    MSE_LOSS_REDUCTION_MODE_NONE
} MSELossMode_t;

typedef enum
{
    LOSS_REDUCTION_MODE_MEAN = 0,
    LOSS_REDUCTION_MODE_SUM,
    LOSS_REDUCTION_MODE_NONE
} LossMode_t;

typedef enum _RoundMode_t
{
    ROUND_HALF_AWAY_FROM_ZERO = 0,
    ROUND_HALF_NEAREST_EVEN = 1,
} RoundMode_t;

typedef enum {
  DIV_ROUND_NONE = 0,
  DIV_ROUND_TRUNC = 1,
  DIV_ROUND_FLOOR = 2
} DivRoundMode_t;

typedef enum _PadMode_t
{
    PAD_MODE_CONSTANT = 0,
    PAD_MODE_REFLECT,
    PAD_MODE_EDGE,
    PAD_MODE_SYMMETRIC
} PadMode_t;

typedef enum _ColorSpaceMode_t
{
    RGB_TO_YCBCR = 0,
    RGB_TO_BGR,
    YCBCR_TO_RGB,
    YCBCR_TO_BGR,
    BGR_TO_RGB,
    BGR_TO_YCBCR,
    GRAY_TO_RGB,
    GRAY_TO_BGR,
    GRAY_TO_YCBCR,
    RGB_TO_GRAY,
    YCBCR_TO_GRAY,
    BGR_TO_GRAY,
} ColorSpaceMode_t;

typedef enum _SpatialCorrelation_t
{
    SPATIAL_CORRELATION_TYPE_NEAREST_NEIGHBOR = 0,
    SPATIAL_CORRELATION_TYPE_BILINEAR,
} SpatialCorrelation_t;

typedef enum _ExpandIntoJaggedIndices_t
{
    EXPAND_JAGGED_INDICES_WITH_NO_INCREMENT = 0,
    EXPAND_JAGGED_INDICES_WITH_INCREMENT,
} ExpandIntoJaggedIndices_t;

typedef enum
{
    RIGHT_RIGHT = 0,
    RIGHT_LEFT = 1,
    LEFT_RIGHT = 2,
    LEFT_LEFT = 3,
} DiagAlign_t;

typedef enum _RadixSortKeyType_t
{
    RADIX_SORT_KEY_UNSIGNED_INT_ASCENDING = 0,
    RADIX_SORT_KEY_UNSIGNED_INT_DESCENDING,
    RADIX_SORT_KEY_SIGNED_INT_ASCENDING,
    RADIX_SORT_KEY_SIGNED_INT_DESCENDING,
    RADIX_SORT_KEY_FLOAT_ASCENDING,
    RADIX_SORT_KEY_FLOAT_DESCENDING,
    RADIX_SORT_KEY_BF16_ASCENDING,
    RADIX_SORT_KEY_BF16_DESCENDING,
    RADIX_SORT_KEY_F16_ASCENDING,
    RADIX_SORT_KEY_F16_DESCENDING,
    RADIX_SORT_KEY_I16_ASCENDING,
    RADIX_SORT_KEY_I16_DESCENDING,
    RADIX_SORT_KEY_U16_ASCENDING,
    RADIX_SORT_KEY_U16_DESCENDING,
} RadixSortKeyType_t;

typedef enum _RadixSortFlavor_t
{
    KEY_METADATA_IN_KEY_METADATA_OUT = 0,
    KEY_IN_KEY_METADATA_OUT,
    KEY_IN_KEY_OUT,
    KEY_METADATA_IN_KEY_METADATA_OUT_VALID_COUNT,
    KEY_IN_KEY_METADATA_OUT_VALID_COUNT,
    KEY_IN_KEY_OUT_VALID_COUNT,
    KEY_METADATA_IN_KEY_METADATA_OUT_TOPK,
    KEY_IN_KEY_METADATA_OUT_TOPK,
    KEY_IN_KEY_OUT_TOPK,
    KEY_METADATA_IN_KEY_METADATA_OUT_VALID_COUNT_TOPK,
    KEY_IN_KEY_METADATA_OUT_VALID_COUNT_TOPK,
    KEY_IN_KEY_OUT_VALID_COUNT_TOPK,
} RadixSortFlavor_t;

typedef enum _KTensorType_t
{
    K_TENSOR_NONE = 0,
    K_TENSOR_SHAPE
} KTensorType_t;
typedef enum _ScatterNdUpdateMode_t
{
    // behaviour is undefined in case of duplicate indices
    SCATTER_ND_UPDATE_MODE_NON_DETERMINISTIC = 0,
    // behaviour is defined in case of duplicate indices (last update is applied)
    SCATTER_ND_UPDATE_MODE_DETERMINISTIC,
} ScatterNdUpdateMode_t;

typedef enum _PosWeightMode_t
{
    POS_WEIGHT_DISABLE = 0,
    POS_WEIGHT_ENABLE,
} PosWeightMode_t;

typedef enum _InterpolationMethod_t
{
    NEAREST_NEIGHBOR = 0,
    BILINEAR
} InterpolationMethod_t;

typedef enum _FillMode_t
{
    CONSTANT = 0,
    REFLECT,
    WRAP,
    NEAREST
} FillMode_t;

typedef enum _BatchNormMode_t
{
    BN_MODE_NON_DETERMINISTIC = 0,
    BN_MODE_DETERMINISTIC = 1,
    BN_MODE_DETERMINISTIC_TPC_ID = 2
} BatchNormMode_t;

typedef enum _ShrinkMode_t
{
    HARD_SHRINK = 0,
    SOFT_SHRINK = 1
} ShrinkMode_t;

typedef enum _BinCountMode_t
{
    //bit0 -> "size" tensor, bit1 -> "weights" tensor, bit2 ->"shape tensor"
    NO_WEIGHT = 1,
    USE_WEIGHT = 3,
    NO_WEIGHT_DSHAPE = 4,
    USE_WEIGHT_DSHAPE = 6
} BinCountMode_t;

typedef enum _KLDivMode_t
{
    KLDIV_LOSS_MODE_SUM = 0,
    KLDIV_LOSS_MODE_MEAN =  1,
    KLDIV_LOSS_MODE_NONE = 2
} KLDivMode_t;

typedef enum _RandomPoissonFlavor_t
{
    WITHOUT_DIST    = 0,
    WITH_DIST       = 1
} RandomPoissonFlavor_t;

typedef enum _BatchNormRoundingMode_t
{
    BN_ROUND_MODE_RHNE = 0,
    BN_ROUND_MODE_RZ   = 1,
    BN_ROUND_MODE_RU   = 2,
    BN_ROUND_MODE_RD   = 3,
    BN_ROUND_MODE_SR   = 4,
    BN_ROUND_MODE_RHAZ = 6
} BatchNormRoundingMode_t;

typedef enum _DataTypeSize_t
{
    DATA_8BIT  = 0,
    DATA_16BIT = 1,
    DATA_32BIT = 2
} DataTypeSize_t;

typedef enum _TypeOfP_t
{
    TYPE_P_IS_INT  = 0,
    TYPE_P_IS_FLOAT = 1
} TypeOfP_t;

typedef enum _ScatterIndicesSizeControl_t
{
    CONTROL_IND_SIZE = 0,
    NO_CONTROL_IND_SIZE = 1,
} ScatterIndicesSizeControl_t;

typedef enum _ScatterReduceMode_t
{
    SCATTER_REDUCE_SUM,
    SCATTER_REDUCE_PROD,
    SCATTER_REDUCE_MEAN,
    SCATTER_REDUCE_AMAX,
    SCATTER_REDUCE_AMIN
} ScatterReduceMode_t;

typedef enum _BoundsCheckMode_t
{
    // Raise an exception (CPU) or device-side assert (HPU)
    BCM_FATAL = 0,
    // Log the first out-of-bounds instance per kernel, and set to zero.
    BCM_WARNING = 1,
    // Set to zero.
    BCM_IGNORE = 2,
    // No bounds checks.
    BCM_NONE = 3
} BoundsCheckMode_t;

typedef enum _RotaryPosEmbeddingMode_t
{
    // LLaMA (First version)
    ROTARY_POS_EMBEDDING_MODE_BLOCKWISE = 0,
    // GPT - J
    ROTARY_POS_EMBEDDING_MODE_PAIRWISE = 1
} RotaryPosEmbeddingMode_t;

typedef enum _ScaledMaskedSoftmaxExpMode
{
    USE_LUT = 0,
    NO_LUT = 1,
    EXPERIMENTAL = 15
} ScaledMaskedSoftmaxExpMode_t;

typedef enum _ScaledMaskedTriangularSoftmaxMode
{
    DEFAULT_MODE                =   0,
    SLICED_MODE                 =   0x1 << 0,
    HF8_1B_MODE                 =   0x1 << 1,
    HF8_1C_MODE                 =   0x1 << 2,
    FUSED_MULT_INV_ATTN         =   0x1 << 3,
    SQUEEZED_FCD_RETAINED       =   0x1 << 4,
    CL_ALIGNED_PADDED_RETAINED  =   0x1 << 5
} ScaledMaskedTriangularSoftmaxMode_t;

typedef enum _CdistComputeMode
{
    USE_MM_FOR_EUCLID_DIST_IF_NECESSARY = 0,
    USE_MM_FOR_EUCLID_DIST = 1,
    DONOT_USE_MM_FOR_EUCLID_DIST = 2
} CdistComputeMode_t;

typedef enum _RmsNormBwdMode
{
    DEFAULT = 0,
    DEFAULT_RMS_NORM_BWD_MODE = 0,
    STATIC_CASE_GC_SLICE_ENABLED  = 1,
    STATIC_CASE_WIDTH_PARTITIONING  = 1
} RmsNormBwdMode_t;

typedef enum _RepeatScalarAPIMode
{
    VPU_LD_FCD_REPEAT = 0,
    SCALAR_LD_FCD_REPEAT_NO_PARTIAL_WR = 1
} RepeatMode_t;

typedef enum _SoftmaxMode_t
{
    DEFAULT_SOFTMAX,
    SOFTMAX_HF8_1B,
    SOFTMAX_HF8_1C,
    SOFTMAX_HF8_2,
} SoftmaxMode_t;

/*
 * Separate softmax mode definition for SDPA to keep it independent of
 * _SoftmaxMode_t so that any other mode can be added.
 */
typedef enum _SdpaSoftmaxMode_t
{
    SDPA_DEFAULT_SOFTMAX = 0,
    SDPA_SOFTMAX_HF8_1B = 1,
    SDPA_SOFTMAX_HF8_1C = 2,
} SdpaSoftmaxMode_t;

typedef enum _SdpaFlags_t
{
    SDPA_FLAGS_AMAX_S = 1<<0,
    SDPA_FLAGS_D_SCALE_Q = 1<<8,
    SDPA_FLAGS_D_SCALE_K = 1<<9,
    SDPA_FLAGS_D_SCALE_V = 1<<10,
    SDPA_FLAGS_Q_SCALE_S = 1<<11,
    SDPA_FLAGS_Q_SCALE_O = 1<<12,
} SdpaFlags_t;

typedef enum _UpperTriangularMaskArea_t
{
    DEFAULT_LOWER_TRIANGLE_MASK_OUT = 0,
    UPPER_TRIANGLE_MASK_OUT = 1,
} UpperTriangularMaskArea_t;

typedef enum _MemcpyMode
{
    MEMCPY_DEFAULT = 0,
    MEMCPY_1D = 1,
    MEMCPY_IRF44 = 2,
} MemcpyMode_t;

typedef enum _SigmoidFlavor
{
    SIGMOID_DEFAULT = 0,
    NO_SATURATION_SIGMOID = 1,
} SigmoidFlavor_t;

/////////////////////////////// GENERAL_KERNELS ///////////////////////////////

/*
 * Kernel name: long_loop
 */
namespace ns_LongLoop
{
    struct Params
    {
        unsigned int count;
    };
} // namespace ns_LongLoop

/*
 * Kernel name: l2norm_i8, l2norm_i16
 */
namespace ns_L2normKernel
{
    struct Params
    {
        float epsilon;
    };
} // namespace ns_L2normKernel

/*
 * Kernel name: l2norm_stage1_f32, l2norm_stage1_bf16
 */
namespace ns_L2NormStage1Kernel
{
    struct Params
    {
        int dim1ChunksSize;
    };
} // namespace ns_L2NormStage1Kernel

/*
 * Kernel name: l2norm_stage2_f32, l2norm_stage2_bf16
 */
namespace ns_L2NormStage2Kernel
{
    struct Params
    {
        float epsilon;
        int dim1ChunksSize;
    };
} // namespace ns_L2NormStage2Kernel

/*
 * Kernel name: printf_test
 */
namespace ns_PrintfTestKernel
{
    struct Params
    {
        int int_val;
        unsigned int uint_val;
        float float_val;
        short short_val __attribute__((aligned(4)));
        unsigned short ushort_val __attribute__((aligned(4)));
        char char_val __attribute__((aligned(4)));
        unsigned char uchar_val __attribute__((aligned(4)));
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_PrintfTestKernel

/*
 * Kernel name:  printf_vpu_demo_i8, printf_vpu_demo_u8, printf_vpu_demo_i16,
 *               printf_vpu_demo_i32, printf_vpu_demo_f32, printf_vpu_demo_bf16
 */
namespace ns_PrintfVpuDemoKernel
{
    struct Params
    {
        fint_t val_0;
        int pos_0;
        fint_t val_1;
        int pos_1;
        fint_t val_2;
        int pos_2;
    };
} // namespace ns_PrintfVpuDemoKernel

/*
 * Kernel name: bincount_f32, bincount_i32
 */
namespace ns_BinCountKernel
{
    struct Params
    {
        BinCountMode_t bincount_mode;
    };
} // namespace ns_BinCountKernel

/*
 * Kernel name: bounds_check_indices_fwd_i32
 */
namespace ns_BoundsCheckIndicesKernel
{
    struct Params
    {
        BoundsCheckMode_t mode;
    };
} // namespace ns_BoundsCheckIndicesKernel

///////////////////////////// ELEMENTWISE_KERNELS /////////////////////////////

/*
 * Kernel name: binary_with_alpha_fwd_f32, binary_with_alpha_f32
 */
namespace ns_BinaryWithAlphaKernel
{
    struct Params
    {
        fint_t alpha;
        BinaryWithAlphaMode_t mode;
    };
} // namespace ns_BinaryWithAlphaKernel


/*
 * Kernel name: linear_bwd_f32
 */
namespace ns_LinearBwdKernel
{
    struct Params
    {
        bool gradBias; // compute gradient for bias or not

    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LinearBwdKernel


/*
 * Kernel name: embedding_dense_pt_bwd_f32
 */
namespace ns_EmbeddingDensePtBwdKernel
{
    struct Params
    {
        int num_weights;
        int padding_idx;
        bool scaleGradByFreq;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_EmbeddingDensePtBwdKernel


/*
 * Kernel name: embedding_renorm_fwd_f32
 */
namespace ns_EmbeddingRenormFwdKernel
{
    struct Params
    {
        double max_norm;
        double norm_type;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_EmbeddingRenormFwdKernel


/*
 * Kernel name: rsqrt_fwd_f32
 */
namespace ns_RsqrtKernel
{
    struct Params
    {
        bool useNonLut; // there is NonLut based flavor for rsqrt_fwd_f32
                        // pros: VLM will be free
                        // cons: accuaracy and performance degraded
                        //default is false
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RsqrtKernel

/*
 * Kernel name: reciprocal_fwd_bf16
 */
namespace ns_ReciprocalKernel
{
    struct Params
    {
        bool useNonLut; // there is NonLut based flavor for reciprocal_fwd_bf16
                        // pros: when the user doesn't want to use LUT because of memory issues
                        // cons: performance degrades incomparison to LUT
                        //default is false
    };
} // namespace ns_ReciprocalKernel

/*
 * Kernel name: div_mod_fwd_i16, div_mod_fwd_i8, div_mod_fwd_i32,
 *              div_mod_i8, div_mod_i16, div_mod_i32
 */
namespace ns_DivModKernel
{
    struct Params
    {
        bool isPyCompatible; // python div_mod requires remainder with the same sign of
                             // divisor except of zero remainder, unlike C++
                             // for backward compatibility, default is false
    };
    struct ParamsV2 : public Params
    {
        bool isTruncRoundingMode;  // Boolean to check whether the rounding_mode is 'trunc' or 'floor'
                                   // default is false, using floor mode
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_DivModKernel

/*
 * Kernel name: clamp_i8, clamp_i16, clamp_f32
 */
namespace ns_ClampKernel
{
    struct Params
    {
        fint_t upperBound;
        fint_t lowerBound;
    };
} // namespace ns_ClampKernel

/*
 * Kernel name: hardtanh_f32
 */
namespace ns_HardTanhKernel
{
    struct Params
    {
        fint_t upperBound;
        fint_t lowerBound;
    };
} // namespace ns_HardTanhKernel

/*
 * Kernel name: hard_sigmoid_f32
 */
namespace ns_HardSigmoidKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_HardSigmoidKernel

/*
 * Kernel name: relu_i8, relu_u8, relu_i16, relu_f32, relu_bf16
 */
namespace ns_ReluKernel
{
    struct Params
    {
        fint_t threshold;
    };
    struct ParamsV2 : public Params
    {
        fint_t replacementValue;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ReluKernel

/*
 * Kernel name: selu_f32, selu_i16
 */
namespace ns_SeluKernel
{
    struct Params
    {
        float gamma;
        float alpha;
    };
    struct ParamsV2 : public Params
    {
        bool isInputFeaturemap;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_SeluKernel

/*
 * Kernel name: elu_i8, elu_i16, elu_f32
 */
namespace ns_EluKernel
{
    struct Params
    {
        float alpha;
    };
    struct ParamsV2 : public Params
    {
        bool isInputFeaturemap;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        float scale;
        float input_scale;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_EluKernel

/*
 * Kernel name: leakyrelu_i8, leakyrelu_i16, leakyrelu_f32
 */
namespace ns_LeakyReluKernel
{
    struct Params
    {
        double alpha;
    };
} // namespace ns_LeakyReluKernel

/*
 * Kernel name: batch_norm_i8,  batch_norm_relu_i8,
 *              batch_norm_i16, batch_norm_relu_i16,
 *              batch_norm_f32, batch_norm_relu_f32,
 */
namespace ns_BatchNormKernel
{
    struct Params
    {
        fint_t threshold; // threshold.i to use with .._i8 and .._i16 kernels
                          // threshold.f to use with .._f32 kernel
        float momentum;
        float epsilon;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BatchNormKernel

/*
 * Kernel name: batch_norm_variance_f32,  batch_norm_variance_bf16
 */
namespace ns_BatchNormVarienceKernel
{
    struct Params
    {
        fint_t threshold; // threshold.i to use with .._i8 and .._i16 kernels
                          // threshold.f to use with .._f32 kernel
        float momentum;
        float epsilon;
        float N;
    };
} // namespace ns_BatchNormVarienceKernel

/*
 * Kernel name: batch_norm_stage1_bwd_bf16
 */
namespace ns_BatchNormStage1Kernel
{
    struct Params
    {
        int disable_beta_gamma_update;
        int N;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        BatchNormMode_t deterministic_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        BatchNormRoundingMode_t rounding_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_BatchNormStage1Kernel

/*
 * Kernel name: batch_norm_stage2_relu_bwd_bf16,        batch_norm_stage2_relu_bwd_f32
 *              batch_norm_stage2_relu_fwd_bf16,        batch_norm_stage2_relu_fwd_f32
 *              batch_norm_stage2_add_relu_bwd_bf16,    batch_norm_stage2_add_relu_bwd_f32
 *              batch_norm_stage2_add_relu_fwd_bf16,    batch_norm_stage2_add_relu_fwd_f32
 */
namespace ns_BatchNormStage2Kernel
{
    struct Params
    {
        float momentum;
        int disable_runnings_update;
        int N;
        float epsilon;
    };
    struct ParamsV2 : public Params
    {
        bool isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        BatchNormMode_t deterministic_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        BatchNormRoundingMode_t rounding_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_BatchNormStage2Kernel

namespace ns_BatchnormStage15Kernel
{
    struct Params
    {
        float momentum;
        float eps;
        int   disable_runnings_update;
        int   kernelType;
        int   useSigmaXk;
        int   enConcat;
    };
} // namespace ns_BatchnormStage15Kernel

/*
 * Kernel name: instance_norm_f32, instance_norm_relu_f32,
 */
namespace ns_InstanceNormKernel
{
    struct Params
    {
        fint_t threshold; // threshold.f to use with .._f32 kernel
    };
} // namespace ns_InstanceNormKernel

/*
 * Kernel name: instance_norm_fwd_f32
 */
namespace ns_InstanceNormTrainingKernel
{
    struct Params
    {
        float momentum;
        float eps;
    };
    struct ParamsRelu : Params
    {
        int enRelu;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRelu, Params);
} // namespace ns_InstanceNormTrainingKernel

/*
 * Kernel name: div_round_fwd_f32, div_round_fwd_bf16, div_round_fwd_i32
 */
namespace ns_DivRoundKernel
{
    struct Params
    {
        DivRoundMode_t mode;
    };
} // namespace ns_DivRoundKernel

/*
 * Kernel name: layer_norm_relu_f32,
 */
namespace ns_LayerNormKernelRelu
{
    struct Params
    {
        fint_t threshold; // threshold.f to use with .._f32 kernel
        bool epsValid;
        fint_t eps; // threshold.f to use with .._f32 kernel
    };
    struct ParamsNorm : Params
    {
        int NormAxisBmp;  // A bit-map for CWHN. Set res bit for the dim to be normalized
        int ParamAxisBmp; // A bit-map for CWHN. Set res bit for the dim to be normalized
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
} // namespace ns_LayerNormKernelRelu

/*
 * Kernel name: layer_norm_f32,
 *              layer_norm_fwd_f32, layer_norm_fwd_bf16
 */
namespace ns_LayerNormKernel
{
    struct Params
    {
        bool epsValid;
        float eps;
    };
    struct ParamsNorm : Params
    {
        int NormAxisBmp;  // A bit-map for CWHN. Set res bit for the dim to be normalized
        int ParamAxisBmp; // A bit-map for CWHN. Set res bit for the dim to be normalized
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
    // It should derive after Params as ParamsNorm contents are meaningless in ParamsPt case
    // but only linear params hierarchy is currently supported
    struct ParamsPt : ParamsNorm
    {
        unsigned normalizedShapeDims;
    };
    ASSERT_SIZE_IS_GREATER(ParamsPt, ParamsNorm);
    struct ParamsRmsNorm : public ParamsPt
    {
        bool fastMath;
    };
    ASSERT_SIZE_IS_GREATER(ParamsRmsNorm, ParamsPt);
} // namespace ns_LayerNormKernel

/*
 * Kernel name: layer_norm_bwd_stage1_f32,
 *              layer_norm_bwd_stage1_bf16
 */
namespace ns_LayerNormStage1Kernel
{
    struct Params
    {
        int chunkSize; // size of chunk to be accumulated for dBeta and dGamma along width
    };
} // namespace ns_LayerNormStage1Kernel

/*
 * Kernel name: lrn_f32
 */
namespace ns_LrnKernel
{
    struct Params
    {
        float alpha;
        float beta;
        float knorm;
        int nsize;
    };
} // namespace ns_LrnKernel

/*
 * Kernel name: rrelu_with_noise_fwd_f32, rrelu_with_noise_fwd_bf16
 */
namespace ns_RreluWithNoiseKernel
{
    struct Params
    {
        float lower;
        float upper;
        bool training;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RreluWithNoiseKernel


/*
 * Kernel name: lpnorm_f32
 */
namespace ns_LpNormKernel
{
    struct Params
    {
        float p;
        float eps;
        int dim;
    };
    struct ParamsNorm : Params
    {
        bool epsMax;  // Performs 'max(denominator,eps)' instead of 'denominator += eps'
    };
    ASSERT_SIZE_IS_GREATER(ParamsNorm, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LpNormKernel

/*
 * Kernel name: lpnorm_frobenius_stage1_f32,
 *              lpnorm_frobenius_stage1_bf16
 */
namespace ns_LpNormFroStage1Kernel
{
    struct Params
    {
        int chunkSize; // size of chunk to be accumulated for norm along width
    };
}

/*
 * Kernel name: norm_moments_f32,
 *              norm_moments_bf16
 */
namespace ns_NormMomentsKernel
{
    struct Params
    {
        unsigned int NormAxisBmp; // bit-map (bit0: FCD). Set bit for the dim to be normalized
    };
    struct ParamsV2 : public Params
    {
        bool useEstMean; // use MeanEstimation tensor (K) in calculation.
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_NormMomentsKernel

/*
 * Kernel name: norm_stage2_f32,
 *              norm_stage2_bf16
 */
namespace ns_NormStage2
{
    struct Params
    {
        float momentum;
        int disable_runnings_update;
        float epsilon;
    };
} // namespace ns_NormStage2

/*
 * Kernel name: softmax_f32, logsoftmax_f32,
 *              softmax_i16, logsoftmax_i16
 */
namespace ns_Softmax
{
    struct Params
    {
        int dim;
    };
    struct ParamsV3 : public Params
    {
        DataTypeSize_t dataTypeSize;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);
    struct ParamsV4 : public ParamsV3
    {
        int validCountAxis;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        int is_approximation;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    struct ParamsV6 : public ParamsV5
    {
        int triangularMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV6, ParamsV5);
    struct ParamsV7 : public ParamsV6
    {
        SoftmaxMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV7, ParamsV6);
} // namespace ns_Softmax

/*
 * Kernel name: mult_bf16
 */
namespace ns_MultSwizzled
{
    struct Params
    {
        DataTypeSize_t dataTypeSize;
    };
} // namespace ns_MultSwizzled

namespace ns_Mult = ns_MultSwizzled;

/*
* Kernel name: constant_i8, constant_i16, constant_f32
*/
namespace ns_ConstantKernel
{
    struct Params
    {
        fint_t constant;
    };

    struct Params_v2
    {
        int const_low;
        int const_high;
    };
} // namespace ns_ConstantKernel

/*
 * Kernel name: random_uniform_f32
 */
namespace ns_RandomUniform
{
    struct Params
    {
        float low;
        float high;
        unsigned int seed;
    };

    struct ParamsV2
    {
        fint_t low;
        fint_t high;
        unsigned int seed;
        int dummy; // to avoid Params size ambiguity
    };
} // namespace ns_RandomUniform

namespace ns_PhiloxRandomUniform
{
    struct Params
    {
        float low;
        float high;

        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        unsigned int counterInitial;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int low_i;
        int high_i;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_PhiloxRandomUniform


/*
 * Kernel name: random_normal_f32
 */
namespace ns_RandomNormal
{
    struct Params
    {
        float mean;
        float stddev;
        unsigned int seed;
    };
} // namespace ns_RandomNormal

/*
 * Kernel name: random_multinomial_f32,
 *              random_multinomial_log_f32
 */
namespace ns_RandomMultinomial
{
    struct Params
    {
        int outcomes;
        int shape;
        int get_prob;
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        int num_samples;
        bool replacement;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int isTfMode;
        bool useRandomSeed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RandomMultinomial

/*
 * Kernel name: random_exponential_f32
 */
namespace ns_RandomExponential
{
    struct Params
    {
        float beta;
        unsigned int seed;
    };
} // namespace ns_RandomExponential

/*
* Kernel name : random_gamma_f32
*/
namespace ns_RandomGamma
{
    struct Params
    {
        float alpha;
        float beta;
        unsigned int seed;
    };
} // namespace ns_RandomGamma

/*
* Kernel name: random_negative_binomial_f32
*/
namespace ns_RandomNegativeBinomial
{
    struct Params
    {
        float k;
        float p;
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        bool isAdditionEnable;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RandomNegativeBinomial

/*
* Kernel name: random_bernoulli_fwd_f32, random_bernoulli_fwd_bf16
*/
namespace ns_RandomBernoulli
{
    struct Params
    {
        unsigned int seed;
    };
    struct ParamsV2 : public Params
    {
        float probability;
    };
} // namespace ns_RandomBernoulli

/*
 * Kernel name: random_uniform_fwd_f32, random_uniform_fwd_bf16
 */
namespace ns_XavierFill
{
    struct Params
    {
        unsigned int seed;
    };
} // namespace ns_XavierFill

/*
* Kernel name : pick_f32
*/
namespace ns_Pick
{
    struct Params
    {
        int axis;
    };
} // namespace ns_Pick

/*
 * Kernel name : random_poisson_f32
 */
namespace ns_RandomPoisson
{
    struct Params
    {
        float lambda;
        unsigned int seed;
        RandomPoissonFlavor_t poissonFlavor;
    };
} // namespace ns_RandomPoisson

/*
 * Kernel name : pow_f32
 */
namespace ns_Power
{
    struct Params
    {
        float exp_val;
    };
} // namespace ns_Power

/*
 * Kernel name: bitonic_sort_f32
 */
namespace ns_BitonicSort
{
    struct Params
    {
        int chunkSize;        // Size of each chunk to be sorted (Same for input/output)
        int axis;             // axis to be sorted
        SortDirection_t dir;  // ascending/descending
        bool isInputSqueezed; // Is the input squeezed
    };
    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicSort

/*
 * Kernel name: search_sorted_fwd_f32 search_sorted_fwd_i32 search_sorted_fwd_bf16
*/
namespace ns_SearchSorted
{
    struct Params
    {
        int right;
    };
} // namespace ns_SearchSorted

/*
 * Kernel name: sort_step_f32
 */

/*
 * Multiple kernel invocations lead to sorted output
 *
 * ( i) totalIterations - number of times the kernel has to be invoked
 *
 *                 - (n * ( n + 1 ) ) / 2
 *
 *             where n = log2 of size of dimension to be sorted
 *
 *       Example : Input : 4D Tensor of size {4,16,5,5}
 *                         axis = 1
 *                         is_ascend = 1
 *
 *           Size of dimension 1 = 16
 *           log2(16) = 4
 *
 *           totalIterations = ( 4 * ( 4 + 1 ) ) / 2 = 10
 *
 *           totalIterations = 10
 *
 *           for(int i = 0; i < totalIterations; i++)
 *           {
 *               iterationNumber = i;
 *
 *               // Kernel Invocations
 *
 *           }
 */
namespace ns_SortStep
{
    struct Params
    {
        // python layer params
        int totalIterations;
        int iterationNumber; // current iteration number

        // user params
        int axis;       // axis to be sorted
        bool is_ascend; // ascending/descending
    };
    struct ParamsV2 : public Params
    {
        int isStable; // 1: stable 0: not stable
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_SortStep

/*
 * Kernel name : filter_and_squeeze_f32
 */
namespace ns_FilterAndSqueeze
{
    struct Params
    {
        fint_t threshold;
    };
} // namespace ns_FilterAndSqueeze

/*
 * Kernel name: scalar_merge_f32
 */

namespace ns_ScalarMerge
{
    struct Params
    {
        int inputChunkSize;
        int topK; // 0 < topK <= IFM dim[axis] size anything beyond that is an error.
        SortDirection_t sortDirection;
        int axis;
        bool isInputSqueezed;
    };

    struct ParamsV2 : public Params
    {
        int currIterNum; // Static iteration number to enable early exit in DS case
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_ScalarMerge

/*
 * Kernel name : nantonum_fwd_f32, nantonum_fwd_bf16
 */
namespace ns_NanToNumKernel
{
    struct Params
    {
        double nan;
        double posInf;
        double negInf;
    };
} // namespace ns_NanToNumKernel

/*
 * Kernel name: sort_bwd_f32
 */
namespace ns_SortBwd
{
    struct Params
    {
        int topK;
        int axis;
    };
} // namespace ns_SortBwd

/*
 * Kernel name: sort_pre_process_f32, sort_pre_process_i32
 */
namespace ns_SortPreProcess
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool hasActualValidCount;   // Flag to generate actual VC for better performance
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    /// disable size check
    struct ParamsV3 : public ParamsV2
    {
        bool vcShapeDataTensor;   // Flag to indicate if VC received is a shape data tensor
    };
    // ******************************************
    // Oops: sizeof(ParamsV3) == sizeof(ParamsV2)
    // ******************************************

    struct ParamsV4 : public ParamsV3
    {
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_SortPreProcess

/*
 * Kernel name: generate_bitonic_chunks_f32, generate_bitonic_chunks_i32
 */
namespace ns_GenerateBitonicChunks
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GenerateBitonicChunks
/*
* Kernel name: large_bitonic_merge_f32, large_bitonic_merge_i32
*/
namespace ns_LargeBitonicMerge
{
    struct Params
    {
        int staticChunkSize;     // Chunk size at the GBC stage
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
        int blockId;
        int stageId;
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LargeBitonicMerge

/*
 * Kernel name: small_bitonic_merge_f32, small_bitonic_merge_i32
 */
namespace ns_SmallBitonicMerge
{
    struct Params
    {
        int staticChunkSize;     // Chunk size at the GBC stage
        int blockId;             // Block number corresponding to required SBM stage
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
    };

    struct ParamsV2 : public Params
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_SmallBitonicMerge

//////////////////////////////// GNMT_KERNELS /////////////////////////////////

/*
 * Kernel name: top_k_st1_i8,   top_k_st2_i8,
 *              top_k_st1_i16,  top_k_st2_i16,
 *              top_k_st1_i32,  top_k_st2_i32,
 *              top_k_st1_f32,  top_k_st2_f32,
 */
namespace ns_TopK
{
    struct Params
    {
        unsigned int kSize;
        unsigned int axis;
        unsigned int chunkSize;
    };

    struct ParamsV2 : public Params
    {
        unsigned int bottomK;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_TopK

/*
 * Kernel name: gru_st1_i8, gru_st2_i8,
 */
namespace ns_GruKernel
{
    struct Params
    {
        int expX;
        int expCt;
    };
} // namespace ns_GruKernel
/*
 * Kernel name: lstm_i16, lstm_f32
 */
namespace ns_LstmKernel
{
    struct Params
    {
        int timeStamp;            // begins with 0, must be LT B vector length
        int expGatesIntermediate; // must be GE to expGatesIn extracted from Tensors
    };
} // namespace ns_LstmKernel

/*
 * Kernel name: bs_st0_f32,
 *              bs_st1_i16,
 *              bs_st2_i16,
 *              bs_st3_i16,
 *              bs_st4_i16,
 *              bs_st5_i16,
 *              bs_st6_i16,
 *              bs_st7_i16,
 *              bs_st8_i16,
 *              bs_st8_i8,
 *              bs_st9_i16
 */
namespace ns_BsKernel
{
    struct Params
    {
        unsigned int k;       //  bsw;
        unsigned int b;       //  bdec;
        unsigned int vocSize; // vocabularySize;
        unsigned int maxOutputLen;
        unsigned int decoderOutputSize;
        float scaleFix;
        float scaleGem;
        int zp;
    };
} // namespace ns_BsKernel

////////////////////////////// REDUCTION_KERNELS //////////////////////////////

/*
 * Kernel name: reduce_sum_f32,
 *              reduce_sum_i8,
 *              reduce_sum_i16,
 *              reduce_prod_f32,
 *              reduce_L1_f32,
 *              reduce_L2_f32,
 *              reduce_log_sum_f32,
 *              reduce_log_sum_exp_f32,
 *              reduce_sum_square_f32,
 *              reduce_max_f32,
 *              reduce_min_f32,
 *              reduce_mean_f32,
 *              argmin_i8,
 *              argmin_i16,
 *              argmin_f32,
 *              argmax_i8,
 *              argmax_i16,
 *              argmax_f32,
 *              hardmax_f32
 */
namespace ns_Reduction
{
    struct Params
    {
        unsigned int reductionDimension;
    };

    struct ParamsV2
    {
        unsigned int reductionDimensionMask; // Bit mask representing dimensions(axes) to be reduced.
                                             // A '1' at Bit_n means dimension 'n' should be reduced
                                             // (with Bit_0 being LSB).
                                             // Eg: mask with binary value 1011 means
                                             // dimensions 3, 1 and 0 are to be reduced(TPC order).
        bool keepDim;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_Reduction

/*
 * Kernel name: reduce_Lp_f32
 */
namespace ns_ReduceLp
{
    struct Params : public ns_Reduction::Params
    {
        unsigned int p;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Reduction::Params);
} // namespace ns_ReduceLp

namespace ns_ReduceLpV2
{
    struct Params : public ns_Reduction::Params
    {
        fint_t p;
        TypeOfP_t typeOfP;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Reduction::Params);
} // namespace ns_ReduceLpV2

/*
 * Kernel name : kthvalue_fwd_f32, kthvalue_fwd_f16,
 *               kthvalue_fwd_bf16, kthvalue_fwd_i32
 */
namespace ns_Kthvalue
{
    struct Params
    {
        unsigned int k_value;
        unsigned int axis;
        bool keep_dims;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Kthvalue

/////////////////////////// SPATIAL_POOLING_KERNELS ///////////////////////////

/*
 * Kernel name: filter_2d_f32,
 *              filter_2d_i8,       filter_2d_i16,
 *              filter_2d_relu_i8,  filter_2d_relu_i16,
 *              maxpool_2d_i8,      maxpool_2d_i16
 */
namespace ns_SpatialReduction
{
    struct Params
    {
        int pad_w_begin;
        int pad_w_end;
        int pad_h_begin;
        int pad_h_end;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int dilation_w;
        int dilation_h;
        EPoolingConvention pooling_convention;
    };
} // namespace ns_SpatialReduction

/*
 * Kernel name: filter_3d_f32,
 *              maxpool_3d_f32
 */
namespace ns_SpatialReduction3D
{
    struct Params : public ns_SpatialReduction::Params
    {
        int pad_d_begin;
        int pad_d_end;
        int kernel_d;
        int stride_d;
        int dilation_d;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction::Params);
} // namespace ns_SpatialReduction3D

/*
 * Kernel name: avg_pool_2d_i8,         avg_pool_2d_i16,
 *              avg_pool_2d_relu_i8,    avg_pool_2d_relu_i16
 */
namespace ns_AveragePooling
{
    struct Params : public ns_SpatialReduction::Params
    {
        int includePadding;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction::Params);
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
} // namespace ns_AveragePooling

/* Kernel name: avg_pool_2d_f32
                avg_pool_2d_bf16 */
namespace ns_AveragePoolingWithDivisorOverride
{
    struct Params : public ns_AveragePooling::Params
    {
        int divisorOverride;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_AveragePooling::Params);
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
} // namespace ns_AveragePoolingWithDivisorOverride

/*
 * Kernel name : avg_pool_3d_f32
 */
namespace ns_AveragePooling3D
{
    struct Params : public ns_SpatialReduction3D::Params
    {
        int includePadding;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction3D::Params);
} // namespace ns_AveragePooling3D

/* Kernel name: avg_pool_3d_f32
*               avg_pool_3d_bf16
*               avg_pool_3d_f16
*/
namespace ns_AveragePooling3DWithDivisorOverride
{
    struct Params : public ns_AveragePooling3D::Params
    {
        int divisorOverride;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_AveragePooling3D::Params);
} // namespace ns_AveragePooling3DWithDivisorOverride

/*
 * Kernel name: maxpool_roi_i8,
                maxpool_roi_i16,
                maxpool_roi_bwd_f32,
                maxpool_roi_bwd_f16,
                maxpool_roi_bwd_bf16,
 */
namespace ns_MaxPoolRoiKernel
{
    struct Params
    {
        unsigned int num_channels;
        unsigned int pooled_shapeX;
        unsigned int pooled_shapeY;
        unsigned int num_roi;
    };
    struct ParamsSegmentsPerAxis : public Params
    {
        int segmentsX;
        int segmentsY;
    };
} // namespace ns_MaxPoolRoiKernel

/*
 * The struct is deprecated, ns_PadKernelEx should be used instead
 */
namespace ns_PadKernel
{
    struct Params
    {
        fint_t value;
        unsigned int pads[10];
    };
} // namespace ns_PadKernel

/*
 * Kernel name: pad_i8, pad_i16, pad_f32, pad_i32
 */
namespace ns_PadKernelEx
{
    struct Params : public ns_PadKernel::Params
    {
        PadMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_PadKernel::Params);
    struct ParamsWithAlign : public Params
    {
        unsigned int output_size_alignment[5];
    };
    ASSERT_SIZE_IS_GREATER(ParamsWithAlign, Params);
} // namespace ns_PadKernelEx

/*
 * Kernel name: memset_from_vc_i8, memset_from_vc_i16, memset_from_vc_f32, memset_from_vc_i32
 */
namespace ns_MemsetFromVC
{
    struct Params
    {
        int axis;
    };
} // namespace ns_MemsetFromVC


/*
 * Kernel name: broadcast_non_fcd_u8,
 *              broadcast_non_fcd_i8,
 *              broadcast_non_fcd_i16,
 *              broadcast_non_fcd_u16,
 *              broadcast_non_fcd_i32,
 *              broadcast_non_fcd_f32,
 *              broadcast_non_fcd_f16,
 *              broadcast_non_fcd_bf16
 */
namespace ns_BroadcastNonFcd
{
    struct Params
    {
        int axis;
    };

    struct ParamsV2 : public Params
    {
        //this param is used in BroadcastNonFcd cguid as an indicator for a second pass
        int cguidAfterTile;//default 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BroadcastNonFcd

/*
 * Kernel name : lpPooling_3d_f32, lpPooling_2d_f32
 */
namespace ns_LpPooling3D
{
    struct Params : public ns_SpatialReduction3D::Params
    {
        int p;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_SpatialReduction3D::Params);
} // namespace ns_LpPooling3D

/*
 * Kernel name: sequence_reverse_i32,
 *              sequence_reverse_i16,
 *              sequence_reverse_i8,
 *              sequence_reverse_f32
 */
namespace ns_SequenceLength
{
    struct Params
    {
        int use_sequence_length; // To use variable-length sequences, set use_sequence_length to
                                 // True, otherwise each example in the batch is assumed
                                 //  to have the max sequence length.
        int batch_axis;
        int time_axis;
    };
} // namespace ns_SequenceLength

/*
 * Kernel name : tile_f32
 */
namespace ns_TileKernel
{
    struct Params
    {
        int repeat[4];
    };
    struct ParamsV2
    {
        int repeat[5];
    };
} // namespace ns_TileKernel

/*
 * Kernel name :sequence_mask_var_len_f32,
 *              sequence_mask_var_len_i8,
 *              sequence_mask_var_len_i16
 */
namespace ns_SequenceMask
{
    struct Params
    {
        bool use_sequence_length;
        float mask_value;
    };
} // namespace ns_SequenceMask

/*
 * Kernel name : dropout_fwd_f32, dropout_bwd_f32
 */
namespace ns_DropoutKernel
{
    struct Params
    {
        float ratio;
        unsigned int seed;
    };
    struct ParamsOptionalMaskOut : public Params
    {
        bool disableMaskOut;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalMaskOut, Params);
    struct ParamsOptionalTrain : public ParamsOptionalMaskOut
    {
        int isTraining;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalTrain, ParamsOptionalMaskOut);
} // namespace ns_DropoutKernel

/*
 * Kernel name : scaled_triangular_dropout_mask_fwd_f32,
 *               scaled_triangular_dropout_mask_fwd_f16,
 *               scaled_triangular_dropout_mask_fwd_bf16
 */
namespace ns_ScaledTriangularDropoutMaskKernel
{
    struct Params
    {
        float ratio;
        unsigned int seed;
        unsigned int allSlicesBase;
        ScaledMaskedTriangularSoftmaxMode_t mode;
        bool isSeedH2D;
        bool isAllSlicesBaseH2D;
        bool isResetUpperTriangle;
    };
} // namespace ns_ScaledTriangularDropoutMaskKernel

/*
 * Kernel name : repeat_f32
 */
namespace ns_RepeatKernel
{
    struct Params
    {
        int repeats;
        int axis;
    };
    struct ParamsV2 : public Params
    {
        RepeatMode_t scalarAPIMode;
    };
} // namespace ns_RepeatKernel

/*
 * Kernel name : repeat_fwd_f32, repeat_bwd_f32
 */
namespace ns_RepeatKernelGaudiTF
{
    struct Params
    {
        int axis;
    };
} // namespace ns_RepeatKernelGaudiTF

/*
 * Kernel name : roll_fwd_f32, roll_fwd_bf16, roll_fwd_i32
 */
namespace ns_RollKernel
{
    struct Params
    {
        int shifts[8];
        int dims[8];
    };
} // namespace ns_RollKernel

/*
 * Kernel name : take_f32
 */
namespace ns_TakeKernel
{
    struct Params
    {
        int axis;
        int mode;
    };
} // namespace ns_TakeKernel

/*
 * Kernel name : gather_f32
 */
namespace ns_GatherKernel
{
    struct Params
    {
        int axis;
    };

    struct ParamsV2 : public Params
    {
        int batchDims;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_GatherKernel

/*
 * Kernel name : flip_f32
 */
namespace ns_FlipKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2
    {
        int dims[5];
        unsigned int num_dims;
    };
} // namespace ns_FlipKernel

/*
 * Kernel name : compress_f32
 */
namespace ns_CompressKernel
{
    struct Params
    {
        int axis;
    };
} // namespace ns_CompressKernel

/*
 * Kernel name : ravel_multi_index_i32
 */
namespace ns_RavelKernel
{
    struct Params
    {
        int shape[5];
    };
} // namespace ns_RavelKernel

/*
 * Kernel name : one_hot_f32
 */
namespace ns_OneHotKernel
{
    struct Params
    {
        int axis;
        int depth;
        float on_value;
        float off_value;
    };
    struct ParamsV2 : public Params
    {
        int is_out_reshaped; //defualt 0
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_OneHotKernel

/*
 * Kernel name : correlation_f32
 */
namespace ns_CorrelationKernel
{
    struct Params
    {
        int kernel_size;
        int max_displacement;
        int stride1;
        int stride2;
        int pad_size;
        int is_multiply;
    };
} // namespace ns_CorrelationKernel

/*
 * Kernel name : spatial_correlation_f32
 */
namespace ns_SpatialCorrelationKernel
{
    struct Params
    {
        SpatialCorrelation_t interpolation;
    };
} // namespace ns_SpatialCorrelationKernel

/*
 * Kernel name : expand_jagged_indices_fwd_dim0_i32
 */
namespace ns_ExpandIntoJaggedIndicesKernel
{
    struct Params
    {
        ExpandIntoJaggedIndices_t increment;
    };
} // namespace ns_ExpandIntoJaggedIndicesKernel

/*
 * Kernel name : upsample_f32, upsample_i8, upsample_u8
 */
namespace ns_UpsampleKernel
{
    struct Params
    {
        EUpsampleType_t mode;
        int scale;
    };
} // namespace ns_UpsampleKernel

/*
 * Kernel name : sparse_lengths_sum_i8, sparse_lengths_sum_i16
 */
namespace ns_SparseLengthsSum
{
    struct Params
    {
        ESparseLengthsSumType_t mode;
    };
} // namespace ns_SparseLengthsSum

/*
 * Kernel name : deformation_transformation_f32, deformation_transformation_i16
 */
namespace ns_DeformationTransformation
{
    struct Params
    {
        int groupCount;
        int kernelWidth;
        int kernelHeight;
        int strideW;
        int strideH;
        int dilationW;
        int dilationH;
        int padW;
        int padH;
    };
} // namespace ns_DeformationTransformation

/*
 * Kernel name : fully_connected_f32
 */
namespace ns_FullyConnected
{
    struct Params
    {
        int is_relu;
    };
} // namespace ns_FullyConnected

/*
 * Kernel name : grid_generator
 */
namespace ns_GridGeneratorKernel
{
    struct Params
    {
        EGridGeneratorType_t mode;
    };
} // namespace ns_GridGeneratorKernel

/*
 * Kernel name : smooth_l1_f32
 */
namespace ns_SmoothL1Kernel
{
    struct Params
    {
        float sigma;
    };
} // namespace ns_SmoothL1Kernel

/*
 * Kernel name : shrink_f32, shrink_fwd_f32, shrink_fwd_bf16,
 *               shrink_bwd_f32, shrink_bwd_bf16
 */
namespace ns_ShrinkKernel
{
    struct Params
    {
        float lambda;
        float bias;
    };
    struct ParamsV2 : public Params
    {
        float lowerBound;
        float upperBound;
        ShrinkMode_t mode;
    };
} // namespace ns_ShrinkKernel

/*
 * Kernel name : optimizer_sgd_bwd_bf16, optimizer_sgd_bwd_f32
 */
namespace ns_OptimizerSGD
{
    struct Params
    {
        float wd;
        float mom;
        float damp;
        bool nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_OptimizerSGD

/*
 * Kernel name : optimizer_sparse_sgd_bf16
 */
namespace ns_OptimizerSparseSGD
{
    struct Params
    {
        float mom;
        bool nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_OptimizerSparseSGD

/*
 * Kernel name : optimizer_adagrad_bwd_bf16, optimizer_adagrad_bwd_f32,
 *               optimizer_adagrad_bf16, optimizer_adagrad_f32
 */
namespace ns_OptimizerAdagrad
{
    struct Params
    {
        float wd;
        float lrd;
        float eps;
    };
} // namespace ns_OptimizerAdagrad

/*
 * Kernel name : optimizer_sparse_adagrad_bf16,
 *               optimizer_sparse_adagrad_with_valid_count_2d_f32,
 *               optimizer_sparse_adagrad_with_valid_count_2d_bf16
 */
namespace ns_OptimizerSparseAdagrad
{
    struct Params
    {
        float decay;
        float eps;
    };
} // namespace ns_OptimizerSparseAdagrad

/*
 * Kernel name : optimizer_sparse_rowwise_adagrad_with_valid_count_2d_bf16,
 *               optimizer_sparse_rowwise_adagrad_with_valid_count_2d_f32
 */
namespace ns_OptimizerRowWiseAdagrad
{
    struct Params
    {
        float eps;
    };
    struct ParamsV2 : public Params
    {
        float decay;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_OptimizerRowWiseAdagrad

/*
 * Kernel name : range_f32,
 *               range_i16,
 *               range_i8
 */
namespace ns_RangeKernel
{
    struct Params
    {
        fint_t start;
        fint_t limit;
        fint_t delta;
    };
} // namespace ns_RangeKernel

/*
 * Kernel name : nms_f32
 */
namespace ns_Nms
{
    struct Params
    {
        float nms_threshold;
    };
} // namespace ns_Nms

/*
 * Kernel name : post_nms_f32
 */
namespace ns_PostNms
{
    struct Params
    {
        int max_output_size;
    };
    struct ParamsV2 : public Params
    {
        int max_size_per_batch;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    enum PostNms_type_t
    {
        ONNX_FRMT = 0,
        TF_COMBINED_NON_MAX_SUPPRESSION_FRMT,
        TF_NON_MAX_SUPPRESSION_FRMT
    };
    struct ParamsV3 : public ParamsV2
    {
        int pad_to_max_output_size; // meset
        PostNms_type_t  frmt;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_PostNms

/*
 * Kernel name : complex_nms_f32
 */
namespace ns_ComplexNMS
{
    struct Params
    {
        ns_FilterAndSqueeze::Params filterAndSqueeze;
        ns_BitonicSort::Params bitonicSort;
        ns_ScalarMerge::Params mergeSort;
        ns_GatherKernel::Params gather;
        ns_Nms::Params nms;
        ns_PostNms::Params postNms;
    };
} // namespace ns_ComplexNMS

/*
 * Kernel name : softmax_cross_entropy_f32, softmax_cross_entropy_bwd_bf16,
 *               softmax_cross_entropy_bwd_f32, softmax_cross_entropy_fwd_bf16,
 *               softmax_cross_entropy_fwd_f32
 */
namespace ns_SoftmaxCrossEntropy
{
    struct Params
    {
        ECrossEntropyMode_t mode;
        int batchSize;
    };

    struct ParamsV2 : public Params
    {
        int axis;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public Params
    {
        bool isTfMode; //To imitate TF at BWD
        int reserved;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);

    struct ParamsV4 : public Params
    {
        bool sparse;
        int reserved[2];
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, Params);
    struct ParamsV5 : public Params
    {
        int is_approximation;
        int padding[3];
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, Params);
} // namespace ns_SoftmaxCrossEntropy

/*
 * Kernel name : binary_cross_entropy_fwd_f32, binary_cross_entropy_bwd_f32
 */
namespace ns_BinaryCrossEntropy
{
    struct Params
    {
        bool isWeightsUsed;
        ECrossEntropyMode_t mode;
    };

    struct ParamsOptionalSigmoid : public Params
    {
        bool binaryCrossEntropyWithoutSigmoid;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalSigmoid, Params);
    struct ParamsOptionalPosWeight : public ParamsOptionalSigmoid
    {
        PosWeightMode_t posMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalPosWeight, ParamsOptionalSigmoid);
    struct ParamsOptionalNormalize : public ParamsOptionalPosWeight
    {
        int isNormalizeWeights;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalNormalize, ParamsOptionalPosWeight);
} // namespace ns_BinaryCrossEntropy

/*
 * Kernel name : eye_like_f32
 */
namespace ns_EyeLikeKernel
{
    struct Params
    {
        int k;
    };
} // namespace ns_EyeLikeKernel

/*
 * Kernel name: max_unpool_2d_f32
 */
namespace ns_MaxUnpoolKernel
{
    struct Params
    {
        int pad_w_begin;
        int pad_w_end;
        int pad_h_begin;
        int pad_h_end;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int out_width;
        int out_height;
    };
} // namespace ns_MaxUnpoolKernel

/*
 * Kernel name : logit_f32
 */
namespace ns_LogitKernel
{
    struct Params
    {
        float epsilon;
    };
} // namespace ns_LogitKernel

/*
 * Kernel name : cast_f32_to_i16,
 *               cast_f32_to_i32,
 *               cast_f32_to_i8,
 *               cast_f32_to_u8,
 *               cast_i16_to_f32,
 *               cast_i32_to_f32,
 *               cast_i8_to_f32,
 *               cast_u8_to_f32
 */
namespace ns_CastKernel
{
    struct Params
    {
        CastF32RoundMode_t round_mode;
    };
    // ParamV2 'seed' argument is relevant only to stochastic round
    struct ParamsV2 : public Params
    {
        unsigned int seed;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // ParamsV3 is relevant only to fp8
    struct ParamsV3 : public ParamsV2
    {
        CastSatMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool safeCastCheck;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_CastKernel

/*
 * Kernel name : scatter_f32
 */
namespace ns_ScatterKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsIndices : public Params
    {
        ScatterIndicesSizeControl_t isIndicesLarger; //if "0" control indices size, "1" -no control
    };
    ASSERT_SIZE_IS_GREATER(ParamsIndices, Params);

    struct ParamsV2 : public ParamsIndices
    {
        int dim;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, ParamsIndices);
    struct ParamsReduce : public ParamsV2
    {
        int include_self;
    };
    ASSERT_SIZE_IS_GREATER(ParamsReduce, ParamsV2);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_ScatterKernel

/*
 * Kernel name : scatter_nd_fwd_f32, scatter_nd_fwd_bf16
 */
namespace ns_ScatterNDKernel
{
    struct Params
    {
        int origIndicesDims;
        int origIndicesShape[5];
    };
    struct ParamsV2 : public Params
    {
        bool isUnsorted; // sort = 1
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ScatterNDKernel

/*
 * Kernel name : scatter_nd_update_fwd_f32, scatter_nd_update_fwd_bf16
 */
namespace ns_ScatterNdUpdateKernel
{
    struct Params
    {
        ScatterNdUpdateMode_t mode;
    };
} // namespace ns_ScatterNdUpdateKernel

/*
 * Kernel name : scatter_reduce_fwd_f32, scatter_reduce_fwd_f16, scatter_reduce_fwd_bf16
 */
namespace ns_ScatterReduceKernel
{
    struct Params
    {
        int dim;
        bool include_self;
        ScatterReduceMode_t mode;
    };
} // namespace ns_ScatterReduceKernel

/*
 * Kernel name : gather_elements_f32, gather_elements_i32
 */
namespace ns_GatherElementsKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        bool isSort; // sort = 1
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GatherElementsKernel

/*
 * Kernel name : resize_u8, resize_i8, resize_f32
 */
namespace ns_ResizeKernel
{
    struct Params
    {
        ResizeInterpolationMode_t mode;
        float scaleDim1;
        float scaleDim2;
        float scaleDim3;
        ResizeCoordinateTransformationMode_t coordTransMode;
        ResizeNearestMode_t nearestMode;
        bool excludeOutside;
        bool useScales; // bool value to select between scales and output sizes
        float cubicCoeffA;
        int size1;
        int size2;
        int size3;
    };

    struct ParamsGauss : Params
    {
        int filterWidth;
        float sigma;
    };
    ASSERT_SIZE_IS_GREATER(ParamsGauss, Params);
} // namespace ns_ResizeKernel

/*
 * Kernel name : grid_sample_bf16,
 *               grid_sample_f32
 */
namespace ns_GridSample
{
    struct Params
    {
        GridSampleInterpolation_t interp;
        GridSamplePad_t pad;
        bool alignCorners;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GridSample

namespace ns_TopkNode
{
    struct Params
    {
        unsigned int bsw; // K value for the topk algorithm.
        unsigned int axis; // Axis to search on for the top K values.
        unsigned int bottomK; // If false, K largest elements will be selected, else, K smallest
    };
} // namespace ns_TopkNode

namespace ns_TopkNodeV2
{
    struct Params
    {
        unsigned int bsw;     // K value for the topk algorithm
    };

    struct ParamsV2 : public Params
    {
    unsigned int axis;    // Axis to search on for the top K values.
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);

    struct ParamsV3 : public ParamsV2
    {
        bool bottomK; // If false, K largest elements will be selected, else K smallest
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool isStable; // stable sort = 1 / unstable sort = 0
        bool isVcData; // was rank - 1 data tensor provided for VC
        KTensorType_t kType; // was K tensor provided, at which format, consumed when we have
                           // 2 input tensors
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
    struct ParamsV5 : public ParamsV4
    {
        bool avoidHandleNans; // for floating point, default 0, backward compatible
    };
    ASSERT_SIZE_IS_GREATER(ParamsV5, ParamsV4);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************

} // namespace ns_TopkNode

namespace ns_FullBitonicSortNode
{
    struct Params
    {
        int staticChunkSize;     // Size of each chunk to be sorted according to the graph.
        int axis;                // axis to be sorted
        SortDirection_t sortDir; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_FullBitonicSortNode

namespace ns_BitonicScalarMergeNode
{
    struct Params
    {
        unsigned int bsw;     // TopK/BottomK sorted size
        unsigned int axis;                // axis to be sorted
        bool bottomK; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicScalarMergeNode

namespace ns_BitonicSortStepNode
{
    struct Params
    {
        int axis;     // axis to be sorted
        bool is_ascend; // ascending/descending
        bool isStable; // stable sort = 1 / unstable sort = 0
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_BitonicSortStepNode

namespace ns_TwoStageTopKNode
{
    struct Params
    {
        unsigned int kSize;     // Size of each chunk to be sorted according to the graph.
        unsigned int axis;      // axis to be sorted
    };

    struct ParamsV2 : public Params
    {
        unsigned int chunkSize;
        bool bottomK;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_TwoStageTopKNode

namespace ns_LogicalSifOpShapeToH2d
{
    struct Params
    {
        int axis;
    };
} // namespace ns_LogicalSifOpShapeToH2d

namespace ns_Sort
{
    struct Params
    {
        int axis;
        bool isDescending;
        RadixSortFlavor_t sortFlavor;
    };
} // namespace ns_Sort

namespace ns_FullRadixSort
{
    struct Params
    {
        int axis;
        bool isDescending;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_FullRadixSort

/*
 * Kernel name : radix_sort_stg1_i32,
 *               radix_sort_stg2_i32,
 */
namespace ns_RadixSort
{
    struct Params
    {
        int stg; // current iteration
        bool sortDescending; // 0: ascending 1: descending
        RadixSortKeyType_t keyType;
        RadixSortFlavor_t sortFlavor; // sort api
        int numTpc; // number of TPCs to work on
        bool isNansHandlingBsCompatible; // handle NaNs same as bitonic sort (replace with +/-INF)
    };

    struct ParamsV2 : public Params
    {
        int maxNumTpc; // Maximum number of available TPCs on device (Required for SIF)
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RadixSort

namespace ns_StoreTnsrSqueeze
{
    struct Params
    {
        int stg;
        int maxBktLen;
        int sSignBit;
    };
} // namespace ns_StoreTnsrSqueeze

/*
 * Kernel name : roialign_f32, roialign_fwd_f32, roialign_fwd_bf16
 */
namespace ns_RoiAlignKernel
{
    struct Params
    {
        RoiAlignMode_t mode;
        int sampling_ratio;
        float spatial_scale;
    };
    struct ParamsAlignment : public Params
    {
        bool aligned;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAlignment, Params);
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_RoiAlignKernel

/*
 * Kernel name : roialign_bwd_f32, roialign_bwd_bf16
 */
namespace ns_RoiAlignBwdKernel
{
    struct Params
    {
        RoiAlignMode_t mode;
        int sampling_ratio;
        float spatial_scale;
        int aligned;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsSegmentsPerAxis : public ParamsIsValidCount
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsIsValidCount);
} // namespace ns_RoiAlignBwdKernel

/*
 * Kernel name : isinf_f32
 */
namespace ns_IsInfKernel
{
    struct Params
    {
        int detect_negative;
        int detect_positive;
    };
} // namespace ns_IsInfKernel

/*
 * Kernel name : quad_tree_fwd_f32,
 *               quad_tree_fwd_bf16,
 *               quad_tree_torch_fwd_f32
 */
namespace ns_QuadTree
{
    struct Params
    {
        int segments; // should be pow(4, x)
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsAbsoluteCoords: ParamsIsValidCount
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsIsValidCount);
    struct ParamsTorchVersion: public ParamsAbsoluteCoords
    {
        int enableTorchVersion;
    };
    ASSERT_SIZE_IS_GREATER(ParamsTorchVersion, ParamsAbsoluteCoords);
    struct ParamsAligned: public ParamsTorchVersion
    {
        int enableAlignment;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAligned, ParamsTorchVersion);
    struct ParamsSegmentsPerAxis : public ParamsAligned
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsAligned);
} // namespace ns_QuadTree

/*
* Kernel name: addr_fwd_f32, addr_fwd_bf16
*/
namespace ns_AddrKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddrKernel

/*
* Kernel name: addmm_fwd_f32, addmm_fwd_bf16
*/
namespace ns_AddmmKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddmmKernel

/*
* Kernel name: addmv_fwd_f32, addmv_fwd_bf16
*/
namespace ns_AddmvKernel
{
    struct Params
    {
        float alpha;
        float beta;
    };
} // namespace ns_AddmvKernel

/*
* Kernel name: combined_nms_fwd_f32
*/
namespace ns_CombinedNmsKernel
{
    struct Params
    {
       float score_threshold;
       float nms_threshold;
       int max_total_size;
       int max_output_size_per_class;
    };
} // namespace ns_CombinedNmsKernel
/*
 * Kernel name : crop_and_resize_f32, crop_and_resize_fwd_f32,
 *               crop_and_resize_fwd_bf16
 */
namespace ns_CropAndResizeKernel
{
    struct Params
    {
        CropAndResizeMode_t mode;
        float extrapolationValue;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsIsOptionalCropsSize : public ParamsIsValidCount
    {
        int isOptionalCropsSize;
        int dummy;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsOptionalCropsSize, ParamsIsValidCount);
    struct ParamsBorderReplicate : public ParamsIsOptionalCropsSize
    {
        bool borderReplicate;
    };
    ASSERT_SIZE_IS_GREATER(ParamsBorderReplicate, ParamsIsOptionalCropsSize);
    struct ParamsAbsoluteCoords : ParamsBorderReplicate
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsBorderReplicate);
} // namespace ns_CropAndResizeKernel

/*
 * Kernel name : crop_and_resize_bwd_f32, crop_and_resize_bwd_bf16
 */
namespace ns_CropAndResizeBwdKernel
{
    struct Params
    {
        CropAndResizeMode_t mode;
    };
    struct ParamsIsValidCount : public Params
    {
        bool isValidCount;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsValidCount, Params);
    struct ParamsIsOptionalCropsSize : public ParamsIsValidCount
    {
        int isOptionalCropsSize;
        int dummy;
    };
    ASSERT_SIZE_IS_GREATER(ParamsIsOptionalCropsSize, ParamsIsValidCount);
    // relevant for Reducible version only. For quad-tree based offsets value will be ignored
    struct ParamsSplitedOutput : public ParamsIsOptionalCropsSize
    {
        int outputWidthOffset;
        int outputHeightOffset;
        int outputBatchOffset;
        int outputChannelOffset;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSplitedOutput, ParamsIsOptionalCropsSize);
    struct ParamsBorderReplicate : public ParamsSplitedOutput
    {
        bool borderReplicate;
    };
    ASSERT_SIZE_IS_GREATER(ParamsBorderReplicate, ParamsSplitedOutput);
    struct ParamsAbsoluteCoords : ParamsBorderReplicate
    {
        bool enableAbsoluteCoords;
        float levelScalarFactor;
    };
    ASSERT_SIZE_IS_GREATER(ParamsAbsoluteCoords, ParamsBorderReplicate);
    struct ParamsSegmentsPerAxis : public ParamsAbsoluteCoords
    {
        int segmentsX;
        int segmentsY;
    };
    ASSERT_SIZE_IS_GREATER(ParamsSegmentsPerAxis, ParamsAbsoluteCoords);
} // namespace ns_CropAndResizeBwdKernel

/*
 * Kernel name : complex_pyramid_roi_align_fwd_f32
 */
namespace ns_PyramidRoiAlignKernel
{
    struct Params
    {
        int num_levels;
        int down_scale_factor;
        ns_CropAndResizeKernel::ParamsIsValidCount cropAndResize_params;
        ns_AveragePoolingWithDivisorOverride::ParamsIsValidCount avgPool_params;
    };
} // namespace ns_PyramidRoiAlignKernel

/*
 * Kernel name : complex_pyramid_roi_align_bwd_f32
 */
namespace ns_PyramidRoiAlignBwdKernel
{
    struct Params
    {
        int num_levels;
        int down_scale_factor;
        ns_QuadTree::ParamsIsValidCount quadTree_params;
        ns_CropAndResizeBwdKernel::ParamsIsValidCount cropAndResize_params;
        ns_AveragePoolingWithDivisorOverride::ParamsIsValidCount avgPool_params;
    };
} // namespace ns_PyramidRoiAlignBwdKernel

/*
 * Kernel name : embedding_sgd_fwd_f32, embedding_sgd_bwd_f32
 */
namespace ns_EmbeddingWithSgdKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerSGD::Params sgd;
    };
} // namespace ns_EmbeddingWithSgdKernel

namespace ns_EmbeddingWithAdagradKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerAdagrad::Params adagrad;
    };
} // namespace ns_EmbeddingWithAdagradKernel

// TODO: remove structures with 'bag' after renaming
namespace ns_EmbeddingBagWithSgdKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerSGD::Params sgd;
    };
} // namespace ns_EmbeddingBagWithSgdKernel

namespace ns_EmbeddingBagWithAdagradKernel
{
    struct Params
    {
        EmbeddingBagMode_t mode;
        ns_OptimizerAdagrad::Params adagrad;
    };
} // namespace ns_EmbeddingBagWithAdagradKernel

namespace ns_EmbeddingBagSumStage2
{
    struct Params
    {
        float learningRate;
        float eps;
    };
}// namespace ns_EmbeddingBagSumStage2

/*
 * Kernel name : cummax_f32/bf16/i32
 */
namespace ns_CumMaxKernel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        int numTpc;
    };
} // namespace ns_CumMaxKernel

/*
 * Kernel name : cumsum_f32
 */
namespace ns_CumSumKernel
{
    struct Params
    {
        int axis;
        int exclusive;
        int reverse;
    };
} // namespace ns_CumSumKernel

/*
 * Kernel name : cumprod_f32
 */
namespace ns_CumProdKernel
{
    struct Params
    {
        int axis;
        int exclusive;
        int reverse;
    };
} // namespace ns_CumProdKernel

/*
 * Kernel name : bitshift_f32
 */
namespace ns_BitShiftKernel
{
    struct Params
    {
        ShiftDir_t direction;
    };
    struct ParamsV2 : public Params
    {
        ShiftMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_BitShiftKernel

/*
 * Kernel name : matmul_bwd_f32, matmul_bwd_bf16
 */
namespace ns_MatmulBwdKernel
{
    struct Params
    {
        bool skip_other_transpose;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_MatmulBwdKernel

/*
 * Kernel name : matrix_band_part_fwd_f32, matrix_band_part_fwd_bf16,
 *               matrix_band_part_bwd_f32, matrix_band_part_bwd_bf16
 */
namespace ns_MatrixBandPartKernel
{
    struct Params
    {
        int numLower;
        int numUpper;
    };
    struct triParams: public Params
    {
        int excludeDiag; // if it is 1 numLower is  first diagonal of the band and numUpper is upper diagonal
    };
    ASSERT_SIZE_IS_GREATER(triParams, Params);
} // namespace ns_MatrixBandPartKernel

/*
 * Kernel name : transpose_f32
 */
namespace ns_TransposeKernel
{
    struct Params
    {
        int axes[4];
    };
} // namespace ns_TransposeKernel

/*
 * Kernel name : bn_get_moments_stage2_f32/bf16
 */
namespace ns_BatchnormGetMomentsKernel
{
    struct Params
    {
        float N;
    };
} // namespace ns_BatchnormGetMomentsKernel

/*
 * Kernel name : unique_fwd_f32
 */

namespace ns_UniqueKernel
{
    struct Params
    {
        int returnInverse;
        int returnCounts;
        int dim;
    };
    struct ParamsV2 : public Params
    {
        int sorted;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_UniqueKernel

/*
 * Kernel name : nll_loss_f32/bf16
 */
namespace ns_NLLLossKernel
{
    struct Params
    {
        NLLLossMode_t mode;
    };
    struct ParamsOptionalIgnoreIndex : public Params
    {
        int ignoreIndexValue;
    };
    ASSERT_SIZE_IS_GREATER(ParamsOptionalIgnoreIndex, Params);
} // namespace ns_NLLLossKernel

/*
 * Kernel name : matrix_diag_f32/bf16
 */
namespace ns_MatrixDiag
{
    struct Params
    {
        int kMin;
        int kMax;
        int rows;
        int cols;
        float pad;
        DiagAlign_t align;
    };
} // namespace ns_MatrixDiag

/*
 * Kernel name : matrix_set_diag_f32/bf16
 */
namespace ns_MatrixSetDiag
{
    struct Params
    {
        int kMin;
        int kMax;
        DiagAlign_t align;
    };
} // namespace ns_MatrixSetDiag

/*
 * Kernel name : smooth_l1loss_fwd_f32, smooth_l1loss_fwd_bf16
 */
namespace ns_SmoothL1LossKernel
{
    struct Params
    {
        LossMode_t mode;
        double beta;
    };
} // namespace ns_SmoothL1LossKernel

/*
 * Kernel name : mse_loss_f32, mse_loss_bf16
 */
namespace ns_MSELossKernel
{
    struct Params
    {
        MSELossMode_t mode;
    };
} // namespace ns_MSELossKernel

/*
 * Kernel name: eye_fwd_f32, eye_fwd_bf16
 */
namespace ns_Eye
{
    struct Params
    {
        int rows;
        int cols;
    };
} // namespace ns_Eye

/*
 * Kernel name : l1_loss_fwd_f32, l1_loss_fwd_bf16
 */
namespace ns_L1LossKernel
{
    struct Params
    {
        LossMode_t mode;
    };
} // namespace ns_L1LossKernel

/*
 * Kernel name : round_f32
 */

namespace ns_RoundKernel
{
    struct Params
    {
        RoundMode_t roundMode;
    };
    struct ParamsV2 : Params
    {
        int num_decimal_round;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RoundKernel

/*
 * Kernel name : huber_loss_fwd_f32, huber_loss_fwd_bf16
 */
namespace ns_HuberLossKernel
{
    struct Params
    {
        LossMode_t mode;
        double delta;
    };
} // namespace ns_HuberLossKernel

/*
 * Kernel name : adaptive_avg_pool_3d_fwd_f32,
 *               adaptive_avg_pool_3d_fwd_bf16
 */
namespace ns_AdaptiveAvgPool3D
{
    struct Params
    {
        int outputWidth;
        int outputHeight;
        int outputBatch;
    };
} // namespace ns_AdaptiveAvgPool3D

/*
 * Kernel name : adaptive_avg_pool_2d_fwd_f32,
 *               adaptive_avg_pool_2d_fwd_bf16
 */
namespace ns_AdaptiveAvgPool
{
    struct Params
    {
        int outputWidth;
        int outputHeight;
    };
} // namespace ns_AdaptiveAvgPool

/*
 * Kernel name : addcmul_fwd_f32,  addcmul_fwd_bf16
 */
namespace ns_AddcmulKernel
{
    struct Params
    {
        float value;
    };
} // namespace ns_AddcmulKernel

/*
 * Kernel name : brightness_u8,
 *               brightness_u16
 */
namespace ns_Brightness
{
    struct Params
    {
        float brightness_scale;
    };
} // namespace ns_Brightness

/*
 * Kernel name : affine_transform_u8
 */
namespace ns_AffineTransform
{
    struct Params
    {
        int resample;
        int fill;
    };
    struct ShearParams : Params
    {
        float shearX;
        float shearY;
        int isAngle;
        float centerX;
        float centerY;
    };
    ASSERT_SIZE_IS_GREATER(ShearParams, Params);
    struct TranslateParams : Params
    {
        float offsetX;
        float offsetY;
    };
    ASSERT_SIZE_IS_GREATER(TranslateParams, Params);
} // namespace ns_AffineTransform

/*
 * Kernel name : hue_u8,
 *               hue_u16
 */
namespace ns_Hue
{
    struct Params
    {
        float degree;
    };
}

/*
 * Kernel name : random_hue_f32, random_hue_f16, random_hue_bf16
 *               random_hue_i32, random_hue_i16, random_hue_i8
 *               random_hue_ui32, random_hue_ui16, random_hue_ui8
 *
 */
namespace ns_RandomHue
{
    struct Params
    {
        float max_delta;
        int seed;
    };
} // namespace ns_Hue

/*
 * Kernel name : random_saturation_f32, random_saturation_f16, random_saturation_bf16
 *               random_saturation_i32, random_saturation_i16, random_saturation_i8
 *               random_saturation_ui32, random_saturation_ui16, random_saturation_ui8
 *
 */
namespace ns_RandomSaturation
{
    struct Params
    {
        float low;
        float high;
        int seed;
    };
} // namespace ns_RandomSaturation

/*
 * Kernel name : random_contrast_f32, random_contrast_f16, random_contrast_bf16
 *               random_contrast_i32, random_contrast_i16, random_contrast_i8
 *               random_contrast_ui32, random_contrast_ui16, random_contrast_ui8
 *
 */
namespace ns_RandomContrast
{
    struct Params
    {
        float low;
        float high;
        int seed;
    };
} // namespace ns_RandomContrast

/*
 * Kernel name : crop_u8,
 *               crop_u16
 */
namespace ns_Crop
{
    struct Params
    {
        int crop_w;
        int crop_h;
        int crop_d;
        float crop_pos_x;
        float crop_pos_y;
        float crop_pos_z;
        int pad_val;
    };
} // namespace ns_Crop

/*
 * Kernel name : crop_mirror_i8,
 *               crop_mirror_u8,
 *               crop_mirror_u16
 */
namespace ns_CropMirror
{
    struct Params : public ns_Crop::Params
    {
        int mirror;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Crop::Params);
} // namespace ns_CropMirror

/*
 * Kernel name : crop_mirror_norm_u8,
 *               crop_mirror_norm_u16
 */
namespace ns_CropMirrorNorm
{
    struct Params : public ns_Crop::Params
    {
        int mirror;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_Crop::Params);
} // namespace ns_CropMirrorNorm

/*
 * Kernel name : saturation_u8,
 *               saturation_u16
 */
namespace ns_Saturation
{
    struct Params
    {
        float saturation_level;
    };
} // namespace ns_Saturation

/*
 * Kernel name : contrast_u8,
 *               contrast_u16
 */
namespace ns_Contrast
{
    struct Params
    {
        float contrast_scale;
    };
} // namespace ns_Contrast

/*
 * Kernel name : color_space_conversion_u8,
 *               color_space_conversion_u16
 */
namespace ns_ColorSpaceConversion
{
    struct Params
    {
        ColorSpaceMode_t colorSpaceMode;
    };
} // namespace ns_ColorSpaceConversion

/*
 * Kernel name : bbflip_f32
 */
namespace ns_BbFlip
{
    struct Params
    {
        int horizontal;
        int vertical;
        bool ltrb;
        bool isHorizontalTensor;
        bool isVerticalTensor;
    };
} // namespace ns_BbFlip

/*
 * Kernel name : flip_3d_u8,
 *               flip_3d_u16
 */
namespace ns_Flip3D
{
    struct Params
    {
        int horizontal;
        int vertical;
        int depthwise;
    };
} // namespace ns_Flip3D

namespace ns_ImageProjectiveTransform
{
    struct Params
    {
        InterpolationMethod_t interpolation;
    };
    struct ParamsV3 : Params
    {
        float      fill_value;
        FillMode_t fill_mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, Params);
} // namespace ns_ImageProjectiveTransform

/*
 * Kernel name: where_stage1_i8,  where_stage2_i32
 */
namespace ns_TfWhere
{
    struct Params
    {
        unsigned int tpcCount;
    };
} // namespace ns_TfWhere

/*
 * Kernel name : normalize_u8,
 *               normalize_u16
 */
namespace ns_Normalize
{
    struct Params
    {
        float scale;
        float shift;
        int axis;
        bool batch;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Normalize

/*
 * Kernel name : gconv_fwd_bf16, gconv_fwd_f32
 */
namespace ns_GConv
{
    struct Params
    {
        int k;
    };
} // namespace ns_GConv

/*
 * Kernel name : conv_weight_packing_fwd_f32, conv_weight_packing_fwd_bf16
 */
namespace ns_WtPack
{
    struct Params
    {
        int packDegree;
        int stride;
    };
} // namespace ns_WtPack

/*
 * Kernel name : unsorted_f32,
 *               unsorted_bf16
 */
namespace ns_UnsortedSegmentSum
{
    struct Params
    {
        int numSegments;
    };
} // namespace ns_UnsortedSegmentSum

/*
 * Kernel name : color_fwd_u8,
 */
namespace ns_Color
{
    struct Params
    {
        float colorFactor;
    };
}

/*
 * Kernel name : solarize_fwd_u8,
 */
namespace ns_Solarize
{
    struct Params
    {
        float threshold;
    };
}

/*
 * Kernel name: reverse_f32
 */
namespace ns_ReverseKernel
{
    struct Params
    {
        int axis;
    };
} // namespace ns_PadKernelEx



/*
 * Kernel name : sharpness_fwd_u8,
 */
namespace ns_Sharpness
{
    struct Params
    {
        float factor;
    };
}

/*
 * Kernel name : histogram_f32
 */
namespace ns_Histogram
{
    struct Params
    {
        int bins;
        int density;
        float min;
        float max;
    };
}

/*
 * Kernel name : autocontrast_lut_u8
 */
namespace ns_HistogramCutOff
{
    struct Params
    {
        float cutOffLowPercent;
        float cutOffHighPercent;
    };
}

/*
 * Kernel name : ctc_loss_fwd_f32, ctc_loss_bwd_f32
 */
namespace ns_CTCLoss
{
    struct Params
    {
        int  blankIndex;
        LossMode_t reductionMode;
        int zeroInfinity;        // Whether to zero infinite losses and the associated gradients.
    };
}

/*
 * Kernel name : kl_div_fwd_f32, kl_div_fwd_bf16, kl_div_bwd_f32, kl_div_bwd_bf16
 */
namespace ns_KLDivParams
{
    struct Params
    {
        KLDivMode_t mode;
        int log_target;
    };
} // namespace ns_KLDivParams

/*
 * Kernel name: gelu_fwd_f32, gelu_fwd_bf16
 */
namespace ns_GeluKernel
{
    struct Params
    {
        bool approximation;
    };

    typedef enum _GeluAccuracyMode_t
    {
        DefaultAccuracyMode = 0,
        FastFP8AccuracyLUT  = 1,
        FastFP8AccuracyCalc = 2
    } GeluAccuracyMode_t;

    struct ParamsV2 : public Params
    {
        GeluAccuracyMode_t accuracy_mode;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_GeluKernel


/*
 * Kernel name: grouped_non_zero_stage1_fwd_i8,
 *              grouped_non_zero_stage2_fwd_i32
 */
namespace ns_GroupedNonZero
{
    struct Params
    {
        unsigned int tpcCount;
    };
} // namespace ns_GroupedNonZero

namespace ns_SoftmaxDropout
{
    struct Params
    {
        ns_Softmax::Params softmax;
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
    };
    struct ParamsV2
    {
        ns_Softmax::ParamsV4 softmax;
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
    };
} // namespace ns_SoftmaxDropout

/*
 * Kernel name: batched_nms_fwd_f32,
 */
namespace ns_BatchedNmsKernel
{
    struct Params
    {
        float nms_threshold;
        int max_num_classes;
    };
} // namespace ns_BatchedNmsKernel
/*
* Kernel name: probe_nan_f32
*/
namespace ns_ProbAny
{
    struct Params
    {
        float probValue;
        int cmp; // 0 - eq, 1 - neq, 2 - geq, 3 - grt, 4 - leq, 5 - less
        unsigned int id;
    };
} // namespace ns_ProbAny

/*
 * Kernel name: non_zero_v2_fwd_i8, non_zero_v2_fwd_i32,
 *              non_zero_v2_fwd_f32, non_zero_v2_fwd_bf16
 */
namespace ns_NonzeroV2
{
    struct Params
    {
        unsigned int pads[10];
        unsigned int max_chunks;
        unsigned int group_size;
    };
} // namespace ns_NonZeroV2Params

/*
 * Kernel name: assert_async_u8
 */
namespace ns_AssertAsync
{
    struct Params
    {
        unsigned int node_id;
        unsigned int msg_id;
    };
    struct ParamsV2 : public Params
    {
        unsigned int msi_interrupt;
    };
    struct ParamsV3 : public ParamsV2
    {
        unsigned int test_mode; // 1 - enabled, 0 - disabled
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_AssertAsync

/*
 * Kernel name : erfinv_fwd_f32
 */
namespace ns_erfinv
{
    struct Params
    {
        bool isTfMode; //To imitate TF
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_erfinv

/*
 * Kernel name: softplus_fwd_f32, softplus_fwd_bf16,
 *              softplus_fwd_f32, softplus_bwd_bf16
 */
namespace ns_Softplus
{
    struct Params
    {
        float beta;
        float threshold;
    };

    typedef enum _SoftplusAccuracyMode_t
    {
        DEFAULT_EXP_LOG1P = 0,
        APPROX_EXP_LOG1P = 1,
    } SoftplusAccuracyMode_t;

    struct ParamsV2 : public Params
    {
        SoftplusAccuracyMode_t accuracyMode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_Softplus

/////////////////////////// RESOURCE_APPLY_KERNELS ////////////////////////////

/*
 * Kernel name: resource_apply_adam_f32, resource_apply_keras_momentum_f32,
 *              resource_sparse_apply_keras_momentum_f32,
 *              resource_apply_adam_bf16, resource_apply_keras_momentum_bf16,
 *              resource_sparse_apply_keras_momentum_bf16
 */
namespace ns_ResourceApplyNesterov
{
    struct Params
    {
        bool use_nesterov;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ResourceApplyNesterov

/*
 * Kernel name: resource_apply_adagrad_f32, resource_apply_adagrad_v2_f32,
 *              resource_sparse_apply_adagrad_f32, resource_sparse_apply_adagrad_v2_f32,
 *              resource_apply_adagrad_bf16, resource_apply_adagrad_v2_bf16,
 *              resource_sparse_apply_adagrad_bf16, resource_sparse_apply_adagrad_v2_bf16
 */
namespace ns_ResourceApplyUpdateSlots
{
    struct Params
    {
        bool update_slots;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_ResourceApplyUpdateSlots

///////////////////////////// LOGICAL_SIF_KERNELS //////////////////////////////
/*
 * Kernel name: logical_sif_op_flatten_shape_f32
 */
namespace ns_LogicalSifOpFlattenShape
{
    struct Params
    {
        int start;
        int end;
        bool keepDims;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LogicalSifOpFlattenShape

/*
 * Kernel name: logical_sif_op_unflatten_shape_f32
 */
namespace ns_LogicalSifOpUnflattenShape
{
    struct Params
    {
        int start;
        int end;
        bool recoverDims; // useful if flatten was done with keepDim = false
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_LogicalSifOpUnflattenShape

/*
 * Kernel name: logical_sif_op_calc_tile_f32
 */
namespace ns_LogicalSifOpCalcTile
{
    struct Params
    {
        bool isDimTiled[5];
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace logical_sif_op_calc_tile

/*
 * Kernel name: logical_sif_op_calc_broadcast_non_fcd_f32
 */
namespace ns_LogicalSifOpCalcBroadcastNonFcd
{
    struct Params
    {
        int vecLen;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace logical_sif_op_calc_broadcast_non_fcd

/////////////////////////// LINEAR_ALGEBRA_KERNELS ////////////////////////////

/*
 * Kernel name: qr_fwd_f32, qr_fwd_bf16
 */
namespace ns_Qr
{
    struct Params
    {
        bool full_matrices;
    };
    // **********************************************************************************
    // when extend this struct, add an int dummy member if needed to avoid size ambiguity
    // use ASSERT_SIZE_IS_GREATER(ParamsV<x>, ParamsV<x-1>) macro for validation
    // **********************************************************************************
} // namespace ns_Qr

/*
 * Kernel name : rsub_f32, rsub_fwd_f32
 */
namespace ns_Rsub
{
    struct Params
    {
        float alpha;
    };
} // namespace ns_Rsub

/*
 * Kernel name : layer_norm_fp8_f32, layer_norm_fp8_bf16
 */
namespace ns_LayerNormFp8
{
    struct Params : public ns_CastKernel::Params
    {
        float eps;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_CastKernel::Params);
} // namespace ns_LayerNormFp8

/*
 * Kernel name : count_non_zero_fwd_f32, count_non_zero_fwd_bf16,
                 count_non_zero_fwd_f16, count_non_zero_fwd_i32
 */
namespace ns_CountNonZero
{
    struct Params
    {
        int dims;
    };
} // namespace ns_CountNonZero

/*
 * Kernel name : split_permute_cat_fwd_f32, split_permute_cat_fwd_bf16
 */
namespace ns_SplitPermuteCat
{
    struct Params
    {
        int batchSize;
        int dims;
        int numFeatures;
    };
} // namespace ns_SplitPermuteCat

/*
 * Kernel name : dropout_fp8_f32, dropout_fp8_bf16
 */
namespace ns_DropoutFp8
{
    struct Params : public ns_CastKernel::Params
    {
        float ratio;
        unsigned int seed;
    };
    ASSERT_SIZE_IS_GREATER(Params, ns_CastKernel::Params);
} // namespace ns_DropoutFp8

/*
 * Kernel name : custom_softmax_fwd_bf16
 */
namespace ns_CustomSoftmax
{
    struct Params
    {
        int flavor;
    };
} // namespace ns_CustomSoftmax

//  For RoPE CGUID
namespace ns_RoPE
{
    struct Params
    {
        unsigned int base;
        unsigned int offset;
    };
} // namespace ns_RoPE

/*
 * Kernel name : rope_st2_fwd_f32, rope_st2_fwd_f16, rope_st2_fwd_bf16
 */
namespace ns_RoPESt2
{
    struct Params
    {
        unsigned int offset;
    };
    struct ParamsV2 : public Params
    {
        RotaryPosEmbeddingMode_t mode;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_RoPESt2

////////////// Scaled Dot Product Attention(SDPA) Kernels /////////////////////
namespace ns_Sdpa
{
    struct Params
    {
        float scale;    // Softmax scale, typ. 1.0/sqrt(head dim)
        bool is_causal; // is attention mask a lower triangular matrix of 1s
        ns_DropoutKernel::ParamsOptionalMaskOut dropout;
        bool is_inference;
    };
    struct ParamsV2 : public Params
    {
        SdpaSoftmaxMode_t softmax_mode;
    };
    struct ParamsV3 : public ParamsV2
    {
        unsigned int flags; // Flags to convey different operating
                            // modes like fp8 measurement
    };
} // namespace ns_Sdpa

/*
 * Kernel name : index_copy_fwd_f32, index_copy_fwd_f16, index_copy_fwd_bf16,
                 index_copy_fwd_i32, index_copy_fwd_u16,
                 index_copy_fwd_i16, index_copy_fwd_u8, index_copy_fwd_i8
 */
namespace ns_IndexCopy
{
    struct Params
    {
        int axis;   //  Axis is in python order
    };
} // namespace ns_IndexCopy

/*
 * Kernel name : rms_norm_bwd_f32, rms_norm_bwd_f16, rms_norm_bwd_bf16
 */
namespace ns_RmsNorm
{
    struct Params
    {
        unsigned int bwdStage;
        int numTpc; // number of TPCs to work on
    };

    struct ParamsV2 : public Params
    {
        RmsNormBwdMode_t bwdMode;
    };

    struct ParamsV3 : public ParamsV2
    {
        bool useStages;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
    struct ParamsV4 : public ParamsV3
    {
        bool fastMath;
        int dummy; // to avoid Params size ambiguity
    };
    ASSERT_SIZE_IS_GREATER(ParamsV4, ParamsV3);
} // namespace ns_RmsNorm

namespace ns_ScaledMaskedSoftmax
{
    struct Params
    {
        float invScaleAttn;
        unsigned int groupedBatchSize;
        unsigned int isUseMax;
        ScaledMaskedSoftmaxExpMode_t expMode;
    };
    struct ParamsV2 : public Params
    {
        ScaledMaskedTriangularSoftmaxMode_t mode;
        int allSlicesBase;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
    struct ParamsV3 : public ParamsV2
    {
        int isToZeroUpperTriangle;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV3, ParamsV2);
} // namespace ns_ScaledMaskedSoftmax

/*
 * Kernel name : pnorm_dist_fwd_f32, pnorm_dist_fwd_f16, pnorm_dist_fwd_bf16
 */
namespace ns_PnormDist
{
    struct Params
    {
        float p;
        unsigned int axis;
    };
} // namespace ns_PnormDist

/*
 * Kernel name : pdist_fwd_f32, pdist_fwd_f16, pdist_fwd_bf16
 */
namespace ns_Pdist
{
    struct Params
    {
        float p;
    };
} // namespace ns_Pdist

namespace ns_Cdist
{
    struct Params
    {
        float p;
        CdistComputeMode_t compute_mode;
    };
} // namespace ns_Cdist

/*
 * Kernel  name : logical_sif_op_calc_pdist
 */
namespace ns_IterativeCguid
{
    struct Params
    {
        unsigned int iteration;
    };
} // namespace ns_IterativeCguid

/*
 * Kernel name : scale_f32, scale_bf16
 */
namespace ns_Scale
{
    struct Params
    {
        float scale;
    };
} // namespace ns_Scale

/*
* Kernel name : native_group_norm_fwd_f32, native_group_norm_fwd_bf16,
*               native_group_norm_fwd_f16
*/
namespace ns_NativeGroupNorm
{
    struct Params
    {
        unsigned int N;
        unsigned int G;
        float epsilon;
    };
} // namespace ns_NativeGroupNorm

/*
* Kernel name : quantize_per_tensor_f32, quantize_per_tensor_bf16,
*               dequantize_per_tensor_f32, dequantize_per_tensor_bf16
*/
namespace ns_QuantizationPerTensor
{
    struct Params
    {
        float scale;
        int zero_point;
    };
    struct ParamsV2 : public Params
    {
        int quant_min;
        int quant_max;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_QuantizationPerTensor

/*
* Kernel name : quantize_per_channel_f32, quantize_per_channel_bf16
*               dequantize_per_channel_f32, dequantize_per_channel_bf16
*/
namespace ns_QuantizationPerChannel
{
    struct Params
    {
        int axis;
    };
    struct ParamsV2 : public Params
    {
        int quant_min;
        int quant_max;
    };
    ASSERT_SIZE_IS_GREATER(ParamsV2, Params);
} // namespace ns_QuantizationPerChannel

/*
* Kernel name : upper_triangular_fwd_f32, upper_triangular_fwd_bf16, upper_triangular_fwd_f16
*/
namespace ns_UpperTriangularKernel
{
    struct Params
    {
        UpperTriangularMaskArea_t maskArea;
        unsigned int allSlicesBase;
        ScaledMaskedTriangularSoftmaxMode_t slicedMode;
    };
} // namespace ns_UpperTriangularKernel

/*
 * Kernel name : pixel_shuffle_fwd_f32, pixel_shuffle_fwd_f16, pixel_shuffle_fwd_bf16,
 *               pixel_shuffle_fwd_i32, pixel_shuffle_fwd_u16, pixel_shuffle_fwd_i16,
 *               pixel_shuffle_fwd_u8, pixel_shuffle_fwd_i8
 */
namespace ns_PixelShuffleKernel
{
    struct Params
    {
        unsigned int upscale_factor;
    };
} // namespace ns_PixelShuffleKernel

/*
* Kernel name : memcpy_f32
*/
namespace ns_Memcpy
{
    struct Params
    {
        MemcpyMode_t mode;
    };
} // namespace ns_Memcpy

/*
 * Kernel name: sigmoid_fwd_f32, sigmoid_fwd_i16, sigmoid_fwd_bf16, sigmoid_fwd_f16
 */
namespace ns_SigmoidKernel
{
    struct Params
    {
        SigmoidFlavor_t flavor;
    };
} // namespace ns_SigmoidKernel

namespace ns_Cholesky
{
    struct Params
    {
        int start_w;
        int start_h;
        int end_w;
        int end_h;
    };
}

/*
 * Kernel name: put_fwd_f32, put_fwd_bf16, put_fwd_f16
 */
namespace ns_PutKernel
{
    struct Params
    {
        bool accumulate;
    };
} // namespace ns_PutKernel

#endif /* PERF_LIB_LAYER_PARAMS_H */
