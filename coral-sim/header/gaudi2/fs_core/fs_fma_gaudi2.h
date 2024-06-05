#pragma once

/*
 *  This File holds all FMA functions
 *  It is used for ASIC (RTL) verification
 *  DO NOT MODIFY IT without discussing with Hilla Ben-Yaacov
 *
 */
#include <assert.h>
#include <fenv.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include <cmath>
#include <string>

namespace Gaudi2
{
static constexpr uint32_t RND_TO_NE   = 0;
static constexpr uint32_t RND_TO_0    = 1;
static constexpr uint32_t RND_TO_PINF = 2;
static constexpr uint32_t RND_TO_NINF = 3;
static constexpr uint32_t RND_SR      = 4;
static constexpr uint32_t RND_HALF_AZ = 6;

static constexpr uint32_t VPE_RM_NEAREST_EVEN           = RND_TO_NE;
static constexpr uint32_t VPE_RM_TO_0                   = RND_TO_0;
static constexpr uint32_t VPE_RM_INF                    = RND_TO_PINF;
static constexpr uint32_t VPE_RM_NINF                   = RND_TO_NINF;
static constexpr uint32_t VPE_RM_STOCHASTIC             = RND_SR;
static constexpr uint32_t VPE_RM_DEFAULT                = 5;
static constexpr uint32_t VPE_RM_RHAZ                   = RND_HALF_AZ;
static constexpr uint32_t VPE_RM_STOCHASTIC_W_RNE_DNORM = 7;

static constexpr uint32_t DEFAULT_NAN_BFP16 = 0x7FFF;
static constexpr uint32_t DEFAULT_NAN_FP32  = 0x7FFFFFFF;
static constexpr uint32_t DEFAULT_NAN_FP16  = 0x7FFF;
static constexpr uint32_t QNAN_BIT          = 0x0040;
static constexpr uint32_t QNAN_BIT_FP32     = 0x00400000;
static constexpr uint32_t DNORM_FTZ         = 1;

static constexpr uint32_t DEFAULT_NAN_FP19     = 0x3FFFF;
static constexpr uint32_t EXPONENT_OFFSET_FP19 = 10;
static constexpr uint32_t SIGN_OFFSET_FP19     = 18;

static constexpr uint32_t UNIT_VAL_FP32 = 0x3F800000;
static constexpr uint32_t UNIT_VAL_BF16 = 0x3F80;
static constexpr uint32_t UNIT_VAL_FP16 = 0x3C00;

#ifdef RUN_WITH_SV
// typedef int 	bool;
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef signed short int   int16_t;
typedef unsigned short int uint16_t;
typedef signed int         int32_t;
typedef unsigned int       uint32_t;
typedef int8_t             int8;
typedef uint8_t            uint8;
typedef int16_t            int16;
typedef uint16_t           uint16;
typedef int32_t            int32;
typedef uint32_t           uint32;
#endif

float fs_bf16_to_f32(uint16_t input);

static inline __attribute__((always_inline)) std::string roundingModeToString(uint32_t roundingMode)
{
    std::string roundingModeStr = "Unknown rounding mode";

    switch (roundingMode) {
        case RND_TO_NE: roundingModeStr = "RND_TO_NE"; break;
        case RND_TO_0: roundingModeStr = "RND_TO_0"; break;
        case RND_TO_PINF: roundingModeStr = "RND_TO_PINF"; break;
        case RND_TO_NINF: roundingModeStr = "RND_TO_NINF"; break;
        case RND_SR: roundingModeStr = "RND_SR"; break;
        case RND_HALF_AZ: roundingModeStr = "RND_HALF_AZ"; break;
    }

    return roundingModeStr;
}

// sbs implements select bits x[high:low]
static inline __attribute__((always_inline)) uint32_t sbs(uint32_t x, uint8_t high, uint8_t low)
{
    return (high == 31) ? (x >> low) : ((x & ((uint32_t)(1U << (high + 1U)) - 1U)) >> low);
}
// cbs implements concatenate bits {x[31-pos:0],y[pos-1,0]}
static inline __attribute__((always_inline)) uint32_t cbs(uint32_t x, uint32_t y, uint8_t pos)
{
    return ((x << pos) | (y & ((1U << pos) - 1)));
}
// ibs implements insert bits x[high:low] = y[high-low-1:0]
static inline __attribute__((always_inline)) uint32_t ibs(uint32_t x, uint32_t high, uint32_t low, uint32_t y)
{
    return (high == 31) ? ((x & ((1U << low) - 1U)) | (y << low))
                        : ((x & (~((1U << (high + 1U)) - (1U << low)))) | ((y << low) & (((1U << (high + 1U)) - 1U))));
}

// lsbs implements select bits x[high:low] for uint64_t
static inline __attribute__((always_inline)) uint64_t lsbs(uint64_t x, uint64_t high, uint64_t low)
{
    uint64_t one_ll         = 1ll;
    uint64_t sixty_three_ll = 63ll;
    return (high == sixty_three_ll) ? (x >> low) : ((x & ((one_ll << (high + one_ll)) - one_ll)) >> low);
}
// lcbs implements concatenate bits {x[63-pos:0],y[pos-1,0]} for uint64_t
static inline __attribute__((always_inline)) uint64_t lcbs(uint64_t x, uint64_t y, uint64_t pos)
{
    uint64_t one_ll = 1ll;
    return ((x << pos) | (y & ((one_ll << pos) - one_ll)));
}
// libs implements insert bits x[high:low] = y[high-low-1:0] for uint64_t
static inline __attribute__((always_inline)) uint64_t libs(uint64_t x, uint64_t high, uint64_t low, uint64_t y)
{
    uint64_t one_ll         = 1ll;
    uint64_t sixty_three_ll = 63ll;
    return (high == sixty_three_ll) ? ((x & ((one_ll << low) - one_ll)) | (y << low))
                                    : ((x & (~((one_ll << (high + one_ll)) - (one_ll << low)))) |
                                       ((y << low) & (((one_ll << (high + one_ll)) - one_ll))));
}

static inline __attribute__((always_inline)) bool is_nan_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_inf_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) == 0));
}
static inline __attribute__((always_inline)) bool is_denorm_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0x00) && (sbs(x, 22, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_zero_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0x00) && (sbs(x, 22, 0) == 0));
}

static inline __attribute__((always_inline)) bool is_nan_bfp16(uint16_t x)
{
    return ((sbs(x, 14, 7) == 0xFF) && (sbs(x, 6, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_inf_bfp16(uint16_t x)
{
    return ((sbs(x, 14, 7) == 0xFF) && (sbs(x, 6, 0) == 0));
}
static inline __attribute__((always_inline)) bool is_denorm_bfp16(uint16_t x)
{
    return ((sbs(x, 14, 7) == 0x00) && (sbs(x, 6, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_zero_bfp16(uint16_t x)
{
    return ((sbs(x, 14, 7) == 0x00) && (sbs(x, 6, 0) == 0));
}

static inline __attribute__((always_inline)) bool is_nan_fp16(uint16_t x)
{
    return ((sbs(x, 14, 10) == 0x1F) && (sbs(x, 9, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_inf_fp16(uint16_t x)
{
    return ((sbs(x, 14, 10) == 0x1F) && (sbs(x, 9, 0) == 0));
}
static inline __attribute__((always_inline)) bool is_denorm_fp16(uint16_t x)
{
    return ((sbs(x, 14, 10) == 0x00) && (sbs(x, 9, 0) != 0));
}
static inline __attribute__((always_inline)) bool is_zero_fp16(uint16_t x)
{
    return ((sbs(x, 14, 10) == 0x00) && (sbs(x, 9, 0) == 0));
}

// lead zero detect - find leading 1
static inline __attribute__((always_inline)) uint8_t lzd(uint32_t x)
{
    if (x == 0)
        return 32;
    // builtin_clz - Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is
    // 0, the result is undefined.
    return (31 - __builtin_clz(x));

    //  int leading_1_ind;
    //  for (leading_1_ind=31; leading_1_ind>=0 ; leading_1_ind--)
    //    {
    //      if (sbs(x,leading_1_ind,leading_1_ind)==1)
    //        break;
    //      if (leading_1_ind==0)
    //        {
    //          leading_1_ind=32;
    //          break;
    //        }
    //    }
    //  return leading_1_ind;
}

// lead zero detect - find leading 1
static inline __attribute__((always_inline)) uint8_t lzd_ll(uint64_t x)
{
    if (x == 0)
        return 64;
    // builtin_clz - Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is
    // 0, the result is undefined.
    return (63 - __builtin_clzll(x));

    //  int leading_1_ind;
    //  for (leading_1_ind=31; leading_1_ind>=0 ; leading_1_ind--)
    //    {
    //      if (sbs(x,leading_1_ind,leading_1_ind)==1)
    //        break;
    //      if (leading_1_ind==0)
    //        {
    //          leading_1_ind=32;
    //          break;
    //        }
    //    }
    //  return leading_1_ind;
}

///////// BF16
/////////////////////////////////////////////////////
// BF16 data type - accumulation into BF16
uint16_t fma_bfp16(uint16_t a, uint16_t b, uint16_t c, uint8_t round_mode);
uint16_t fp32_to_bf16(float input, int roundingMode, uint32_t sr_register, bool clip_fp);
uint32_t fp32_to_fp19(float input, int roundingMode, uint32_t sr_register, bool clip_fp);
uint32_t fp32_to_tf32(float input, int roundingMode, uint32_t sr_register, bool clip_fp);
uint16_t add_bf16(uint16_t a, uint16_t b, uint8_t round_mode);
/////////////////////////////////////////////////////

/////////FP32 - pure FP32 FMA + FP32 adders
/////////////////////////////////////////////////////
// fp32 - no flush of result
uint32_t fma_fp32_no_flush(uint32_t a, uint32_t b, uint32_t c, uint8_t round_mode);
// fp32 = fp32 + fp32*fp32
uint32_t fma_fp32(uint32_t a, uint32_t b, uint32_t c, uint8_t round_mode);
// BF16 data type - accumulation into FP32
uint32_t fma_bfp16_fp32(uint16_t a, uint16_t b, uint32_t c, uint8_t round_mode);
uint32_t add_fp32(uint32_t a, uint32_t b, uint8_t round_mode);
uint32_t add_fp32_4args(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint8_t round_mode);
/////////////////////////////////////////////////////

/////////FP16 - Conversion is from Sergei's code
/////////////////////////////////////////////////////
static constexpr uint32_t SIGN_OFFSET_FP32      = 31;
static constexpr uint32_t SIGN_MASK_FP32        = 0x80000000;
static constexpr uint32_t EXPONENT_OFFSET_FP32  = 23;
static constexpr uint32_t EXPONENT_MASK_FP32    = 0x7F800000;
static constexpr uint32_t EXPONENT_BIAS_FP32    = 127;
static constexpr uint32_t SIGNIFICAND_MASK_FP32 = 0x007FFFFF;

static constexpr uint32_t SIGN_OFFSET_FP16      = 15;
static constexpr uint32_t SIGN_MASK_FP16        = 0x8000;
static constexpr uint32_t EXPONENT_OFFSET_FP16  = 10;
static constexpr uint32_t EXPONENT_MASK_FP16    = 0x7C00;
static constexpr uint32_t EXPONENT_BIAS_FP16    = 15;
static constexpr uint32_t SIGNIFICAND_MASK_FP16 = 0x03FF;

bool fp16_is_zero(uint16_t val);
bool fp16_is_infinity(uint16_t val);
bool fp16_is_nan(uint16_t val);
bool fp16_is_negative(uint16_t val);
bool fp16_is_denormal(uint16_t val);
int  lzcnt(uint32_t bits, uint32_t int_num);

static inline __attribute__((always_inline)) uint32_t bf16_to_fp32(uint16_t inputUint, bool clip_fp)
{
    uint32_t output = (uint32_t)inputUint;
    output          = (output << 16);

    if (clip_fp & is_inf_fp32(output)) {
        output = output - 1; // will return +/-max_norm_value
    }

    return output;
}

static inline __attribute__((always_inline)) uint32_t fp16_to_fp32(uint16_t inputUint, bool clip_fp)
{

    int32_t inputMantissa = (inputUint & SIGNIFICAND_MASK_FP16);
    int32_t inputExponent = (inputUint & EXPONENT_MASK_FP16) >> EXPONENT_OFFSET_FP16;
    int32_t inputSign     = (inputUint & SIGN_MASK_FP16) >> SIGN_OFFSET_FP16;

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;

    if (fp16_is_zero(inputUint)) {
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (fp16_is_nan(inputUint)) {
        outputExponent = 0xFF;
        outputMantissa = 0x007FFFFF;
        outputSign     = 0;
    } else if (fp16_is_infinity(inputUint)) {
        outputExponent = 0xFF;
        outputMantissa = 0x0;
    } else {
        outputExponent                = inputExponent - EXPONENT_BIAS_FP16 + EXPONENT_BIAS_FP32;
        int32_t mantissaForAdjustment = inputMantissa;
        if (fp16_is_denormal(inputUint)) {
            int shift = lzcnt(EXPONENT_OFFSET_FP16, inputMantissa);
            // Shift leading 1 to bit 10 (normalize) and fixup the exponent accordingly
            mantissaForAdjustment = (inputMantissa << (shift + 1)) & SIGNIFICAND_MASK_FP16;
            outputExponent -= shift;
        }
        // Normal case
        outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16);
    }

    uint32_t outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

    if (clip_fp & is_inf_fp32(outputUint)) {
        outputUint = outputUint - 1; // will return +/-max_norm_value
    }

    return outputUint;
}

uint32_t fma_fp16_fp32(uint16_t a,
                       uint16_t b,
                       uint32_t c,
                       uint8_t  round_mode,
                       bool     fp16_ftz_in  = false,
                       bool     fp16_ftz_out = true);
int      fp_accommodate_rounding(uint32_t     intValuePreRounding,
                                 bool         roundedMSB,
                                 bool         roundedLSBs,
                                 unsigned int sign,
                                 int          roundingMode,
                                 uint32_t     lfsrVal,
                                 uint32_t     discardedAlignedLeft);

uint16_t fp32_to_fp16(float input, int roundingMode, uint32_t lfsrVal, bool clip_fp16);
uint16_t fma_fp16_fp16(uint16_t a, uint16_t b, uint16_t c, uint8_t round_mode, bool fp16_ftz_in, bool fp16_ftz_out);
///////////////////////////////////////

/////////LFSR
/////////////////////////////////////////////////////
uint32_t mme_lfsr32(const uint32_t prev_val, const uint32_t polynomial);
uint16_t mme_lfsr16(const uint16_t prev_val, const uint16_t polynomial);

// RMW_OPCODE ST_TNSR
static constexpr uint32_t ST_TNSR_RMW_ADD       = 0;
static constexpr uint32_t ST_TNSR_RMW_SUB       = 1;
static constexpr uint32_t ST_TNSR_RMW_MIN       = 2;
static constexpr uint32_t ST_TNSR_RMW_MAX       = 3;
static constexpr uint32_t ST_TNSR_RMW_MAX_0_ADD = 4;

// RMW_DATA_TYPE ST_TNSR
static constexpr uint32_t ST_TNSR_RMW_INT8   = 0;
static constexpr uint32_t ST_TNSR_RMW_INT16  = 1;
static constexpr uint32_t ST_TNSR_RMW_INT32  = 2;
static constexpr uint32_t ST_TNSR_RMW_UINT8  = 3;
static constexpr uint32_t ST_TNSR_RMW_UINT16 = 4;
static constexpr uint32_t ST_TNSR_RMW_UINT32 = 5;
static constexpr uint32_t ST_TNSR_RMW_BF16   = 6;
static constexpr uint32_t ST_TNSR_RMW_FP32   = 7;
static constexpr uint32_t ST_TNSR_RMW_FP16   = 8;
static constexpr uint32_t ST_TNSR_RMW_FP8    = 9;

// Mul-Add tree

static inline __attribute__((always_inline)) void fp_mult(uint32_t a,
                                                          uint32_t b,
                                                          int32_t* ab_exp,
                                                          int64_t* ab_sig,
                                                          uint8_t* is_zero,
                                                          uint8_t* is_inf,
                                                          uint8_t* is_nan,
                                                          uint8_t* is_neg,
                                                          bool     fp32_emul,
                                                          int      fp32_emul_part,
                                                          bool     is_fp8)
{
    int64_t  a_sig;
    int64_t  b_sig;
    uint64_t a_leading_1;
    uint64_t b_leading_1;
    uint32_t a_exp          = 0;
    uint32_t b_exp          = 0;
    uint32_t a_man          = 0;
    uint32_t b_man          = 0;
    uint32_t denorm_shift_a = 0;
    uint32_t denorm_shift_b = 0;

    if (is_nan_fp32(a) || is_nan_fp32(b) || (is_inf_fp32(a) && is_zero_fp32(b)) ||
        (is_inf_fp32(b) && is_zero_fp32(a))) {
        *is_nan  = 1;
        *is_zero = 0;
        *is_inf  = 0;
        *is_neg  = 0;
        *ab_exp  = 0xff;
        *ab_sig  = -1;
    } else if (is_inf_fp32(a) || is_inf_fp32(b)) {
        *is_nan  = 0;
        *is_zero = 0;
        *is_inf  = 1;
        *is_neg  = sbs(a, 31, 31) ^ sbs(b, 31, 31);
        *ab_exp  = 0xff;
        *ab_sig  = 0;
    } else if (is_zero_fp32(a) || is_zero_fp32(b) || is_denorm_fp32(a) || is_denorm_fp32(b)) {
        *is_nan  = 0;
        *is_zero = 1;
        *is_inf  = 0;
        *is_neg  = sbs(a, 31, 31) ^ sbs(b, 31, 31);
        *ab_exp  = -127; // 0; //force smallest exponent possible
        *ab_sig  = 0;
    } else {
        *is_nan  = 0;
        *is_zero = 0;
        *is_inf  = 0;
        *is_neg  = sbs(a, 31, 31) ^ sbs(b, 31, 31);
        a_exp    = sbs(a, 30, 23);
        b_exp    = sbs(b, 30, 23);
        if (is_fp8) {
            // denormalize denormals in order to keep precision same as RTL
            if (a_exp < 113) {
                denorm_shift_a = 113 - a_exp;
                a_exp          = 113;
            }
            if (b_exp < 113) {
                denorm_shift_b = 113 - b_exp;
                b_exp          = 113;
            }
        }
        *ab_exp = a_exp + b_exp;
        *ab_exp -= 127;
        a_leading_1 = ((fp32_emul == 0) || (fp32_emul_part == 2) || (fp32_emul_part == 3)) ? 1 : 0;
        b_leading_1 = ((fp32_emul == 0) || (fp32_emul_part == 1) || (fp32_emul_part == 3)) ? 1 : 0;
        a_man       = sbs(a, 22, 0);
        b_man       = sbs(b, 22, 0);
        a_sig       = libs(a_man, 23, 23, a_leading_1);
        b_sig       = libs(b_man, 23, 23, b_leading_1);
        if (is_fp8) {
            a_sig = a_sig >> denorm_shift_a;
            b_sig = b_sig >> denorm_shift_b;
        }
        *ab_sig = a_sig * b_sig;
        // printf("mult normals %08x %08x %016llx\n", libs(sbs(a,22,0),23,23,1), libs(sbs(b,22,0),23,23,1),
        // *(uint64_t*)ab_sig);
    }
    // printf("old: a %08x b %08x exp %08x sig %lx\n", a, b, *ab_exp, *ab_sig);
}

static inline __attribute__((always_inline)) void fp_mult_nonf32f8(uint32_t a,
                                                                   uint32_t b,
                                                                   int32_t* ab_exp,
                                                                   int64_t* ab_sig,
                                                                   uint8_t* is_zero,
                                                                   uint8_t* is_inf,
                                                                   uint8_t* is_nan,
                                                                   uint8_t* is_neg,
                                                                   bool     fp32_emul,
                                                                   int      fp32_emul_part,
                                                                   bool     is_fp8)
{
    float    a_fp, b_fp;
    double   c_fp;
    uint64_t c_int;
    memcpy(&a_fp, &a, sizeof(float));
    memcpy(&b_fp, &b, sizeof(float));
    c_fp = (double)a_fp * (double)b_fp;
    memcpy(&c_int, &c_fp, sizeof(double));
    int32_t exp = lsbs(c_int, 62, 52);
    *ab_exp     = exp - 1023 + 127;

    *ab_sig = lsbs(c_int, 51, 0);
    *ab_sig = *ab_sig | (1ull << 52);
    *ab_sig >>= 6;
    *is_neg  = lsbs(c_int, 63, 63);
    *is_nan  = std::isnan(c_fp);
    *is_zero = (c_int & 0x7fffffffffffffffull) == 0ull;
    *is_inf  = std::isinf(c_fp);
    // printf("new: a %08x b %08x exp %08x sig %lx pro %lx\n", a, b, *ab_exp, *ab_sig, c_int);
}

static inline __attribute__((always_inline)) void fp_add(int32_t  a_exp,
                                                         int64_t  a_sig,
                                                         int32_t  b_exp,
                                                         int64_t  b_sig,
                                                         int32_t* res_exp,
                                                         int64_t* res_sig,
                                                         uint8_t  a_is_zero,
                                                         uint8_t  a_is_inf,
                                                         uint8_t  a_is_nan,
                                                         uint8_t  a_is_neg,
                                                         uint8_t  a_sticky,
                                                         uint8_t  b_is_zero,
                                                         uint8_t  b_is_inf,
                                                         uint8_t  b_is_nan,
                                                         uint8_t  b_is_neg,
                                                         uint8_t  b_sticky,
                                                         uint8_t* res_is_zero,
                                                         uint8_t* res_is_inf,
                                                         uint8_t* res_is_nan,
                                                         uint8_t* res_is_neg,
                                                         uint8_t* res_sticky,
                                                         uint8_t* res_pad_1,
                                                         uint8_t  round_mode)
{
    *res_sticky = 0;
    *res_pad_1  = 0;
    if (a_is_nan || b_is_nan || (a_is_inf && b_is_inf && (a_is_neg != b_is_neg))) {
        *res_is_nan  = 1;
        *res_is_zero = 0;
        *res_is_inf  = 0;
        *res_is_neg  = 0;
        *res_exp     = 0xff;
        *res_sig     = -1;
    } else if (a_is_inf || b_is_inf) {
        *res_is_nan  = 0;
        *res_is_zero = 0;
        *res_is_inf  = 1;
        *res_is_neg  = a_is_inf ? a_is_neg : b_is_neg;
        *res_exp     = 0xff;
        *res_sig     = 0;
    } else if (a_is_zero && b_is_zero) {
        *res_is_nan  = 0;
        *res_is_zero = 1;
        *res_is_inf  = 0;
        *res_is_neg  = (round_mode == 3) ? (a_is_neg | b_is_neg) : (a_is_neg & b_is_neg);
        *res_exp     = 0;
        *res_sig     = 0;
    } else if (a_is_zero) {
        *res_is_nan  = 0;
        *res_is_zero = 0;
        *res_is_inf  = 0;
        *res_is_neg  = b_is_neg;
        *res_exp     = b_exp;
        *res_sig     = b_sig;
    } else if (b_is_zero) {
        *res_is_nan  = 0;
        *res_is_zero = 0;
        *res_is_inf  = 0;
        *res_is_neg  = a_is_neg;
        *res_exp     = a_exp;
        *res_sig     = a_sig;
    } else {
        *res_is_nan  = 0;
        *res_is_zero = 0;
        *res_is_inf  = 0;

        int32_t max_exp    = (a_exp >= b_exp) ? a_exp : b_exp;
        int32_t a_exp_diff = max_exp - a_exp;
        if (a_exp_diff > 62)
            a_exp_diff = 62;
        int64_t a_sig_shifted = a_sig >> a_exp_diff;
        uint8_t sticky_a      = a_sticky | (a_sig != (int64_t)((uint64_t)a_sig_shifted << a_exp_diff));
        int32_t b_exp_diff    = max_exp - b_exp;
        if (b_exp_diff > 62)
            b_exp_diff = 62;
        int64_t b_sig_shifted = b_sig >> b_exp_diff;
        uint8_t sticky_b      = b_sticky | (b_sig != (int64_t)((uint64_t)b_sig_shifted << b_exp_diff));
        *res_exp              = max_exp;
        *res_sig              = a_sig_shifted + b_sig_shifted;
        *res_pad_1            = (a_is_neg | b_is_neg) & (sticky_a | sticky_b) &
                     ((*res_sig & 0x1) != 0); // TODO: is this correct? (align to DW)
        // printf("add normals inputs %016llx %016llx\n", *(uint64_t*)&a_sig, *(uint64_t*)&b_sig);
        // printf("add normals %016llx %016llx %016llx\n", *(uint64_t*)&a_sig_shifted, *(uint64_t*)&b_sig_shifted,
        // *(uint64_t*)res_sig);

        *res_is_neg = (*res_sig < 0);
        if (*res_sig == 0) {
            *res_is_zero = 1;
            *res_is_neg  = (round_mode == 3) ? (a_is_neg | b_is_neg) : (a_is_neg & b_is_neg);
        }
        *res_sticky = sticky_a | sticky_b;
        // printf("res_sticky %d\n",*res_sticky);
    }
}

static inline __attribute__((always_inline)) void fp_convert(int32_t   sum_exp,
                                                             int64_t   sum_sig,
                                                             uint8_t   sum_is_zero,
                                                             uint8_t   sum_is_inf,
                                                             uint8_t   sum_is_nan,
                                                             uint8_t   sum_is_neg,
                                                             uint8_t   sum_sticky,
                                                             uint32_t* result,
                                                             uint8_t   round_mode,
                                                             uint8_t   K)
{
    // printf("sum_sig %016llx\n",*(uint64_t*)&sum_sig);
    sum_sig = ((uint64_t)sum_sig << 1ll) | (uint64_t)sum_sticky;
    if (sum_sig < 0)
        sum_sig = -sum_sig;
    sum_sig = sum_sig >> 1ll;
    // printf("sum_sig %016llx\n",*(uint64_t*)&sum_sig);

    if (sum_is_nan)
        *result = DEFAULT_NAN_FP32;
    else if (sum_is_inf)
        *result = sum_is_neg ? 0xff800000 : 0x7f800000;
    else if (sum_is_zero || (sum_sig == 0)) {
        // depends on sticky and round_mode
        if (sum_sticky && sum_is_neg && round_mode == 3)
            *result = 0xbf800000; //-min normal
        else if (sum_sticky && !sum_is_neg && round_mode == 2)
            *result = 0x3f800000; //+min normal
        else if (sum_is_neg)
            *result = 0x80000000; //-0
        else
            *result = 0;
    } else {
        // normalize and round with sticky
        uint8_t leading_1_ind = lzd_ll(*(uint64_t*)&sum_sig);
        // printf("leading 1 ind %d\n",leading_1_ind);
        int64_t  shift_val;
        uint64_t highest_bit = 21 + K; // 52 in case K=31
        sum_exp += 1; // plus 1 because exp was referring to position 49 and now I am aligning to position 50
        // printf("sum_sig %016llx\n",*(uint64_t*)&sum_sig);
        shift_val = leading_1_ind - highest_bit;
        sum_exp   = sum_exp + shift_val;
        // printf("sum_sig %016llx\n",*(uint64_t*)&sum_sig);
        bool res_g;
        bool res_rs;

        if (sum_exp == 0)
            leading_1_ind += 1;

        if (leading_1_ind >= 24)
            res_g = lsbs(*(uint64_t*)&sum_sig, leading_1_ind - 24, leading_1_ind - 24) != 0;
        else
            res_g = 0;
        if (leading_1_ind >= 25)
            res_rs = lsbs(*(uint64_t*)&sum_sig, leading_1_ind - 25, 0) != 0 || sum_sticky != 0;
        else
            res_rs = 0;
        if (leading_1_ind >= 23)
            sum_sig = lsbs(*(uint64_t*)&sum_sig, leading_1_ind, leading_1_ind - 23);
        else
            sum_sig = lsbs(*(uint64_t*)&sum_sig, leading_1_ind, 0) << (23 - leading_1_ind);

        uint32_t res_man = sum_sig;
        uint8_t  res_sgn = sum_is_neg;

        bool need_rnd =
            (((round_mode == RND_TO_PINF) & (res_rs | res_g) & (res_sgn == 0)) |
             ((round_mode == RND_TO_NINF) & (res_rs | res_g) & (res_sgn == 1)) |
             ((round_mode == RND_TO_NE) & ((res_g & res_rs) | (res_g & !res_rs & (sbs(res_man, 0, 0) == 1)))));
        // printf("sum_sig %016llx sum_exp %02x g %d rs %d need_rnd %d\n", *(uint64_t*)&sum_sig, sum_exp, res_g, res_rs,
        // need_rnd);

        res_man = res_man + need_rnd;

        if ((sum_exp > 0 && sbs(res_man, 24, 24) == 1) || (sum_exp == 0 && sbs(res_man, 23, 23) == 1))
            sum_exp = sum_exp + 1;

        if (sum_exp > 254) {
            if (sum_is_neg)
                *result = 0xff800000;
            else
                *result = 0x7f800000;
        } else if (sum_exp < 1) {
            if (sum_is_neg) {
                if (round_mode == 3)
                    *result = 0x80800000;
                else
                    *result = 0x80000000;
            } else {
                if (round_mode == 2)
                    *result = 0x00800000;
                else
                    *result = 0;
            }
        } else {
            *result = 0;
            *result = ibs(*result, 31, 31, res_sgn);
            *result = ibs(*result, 30, 23, sum_exp);
            *result = ibs(*result, 22, 0, res_man);
        }
    }
}

static inline __attribute__((always_inline)) uint32_t fma_mul_add_tree_n(uint32_t* a,
                                                                         uint32_t* b,
                                                                         uint32_t  c,
                                                                         uint8_t   round_mode,
                                                                         uint8_t   N,
                                                                         uint8_t   K,
                                                                         bool      fp32_emul,
                                                                         bool      c_after_norm,
                                                                         bool      is_fp8,
                                                                         bool      no_opt = false)
{
    assert(N < 32);

    uint8_t anbn_is_zero[32];
    uint8_t anbn_is_inf[32];
    uint8_t anbn_is_nan[32];
    uint8_t anbn_is_neg[32];
    int32_t anbn_exp[32];
    int64_t anbn_sig[32];

    uint8_t anbn_sum_is_zero = 0, anbn_sum_is_inf = 0, anbn_sum_is_nan = 0, anbn_sum_is_neg = 0;
    uint8_t sum_plus_inf = 0, sum_minus_inf = 0;
    uint8_t sum_all_neg = 1; //, anbn_sum_not_zero = 0;
    uint8_t c_is_zero   = 0, c_is_inf, c_is_nan, c_is_neg;
    uint8_t res_is_zero = 0, res_is_inf, res_is_nan, res_is_neg;

    int32_t c_exp;
    int64_t c_sig;

    int32_t res_exp;
    int64_t res_sig;

    uint32_t result;

    // zero denormals
    int i;
    if (DNORM_FTZ) {
#ifndef __PPC__
#pragma omp simd
#endif
        for (i = 0; i < N; i++) {
            if (is_denorm_fp32(a[i]))
                a[i] = ibs(a[i], 30, 0, 0);
            if (is_denorm_fp32(b[i]))
                b[i] = ibs(b[i], 30, 0, 0);
        }

        if (is_denorm_fp32(c))
            c = ibs(c, 30, 0, 0);
    }

    int32_t max_exp = 0x80000000;
    int32_t exp_diff;

    int64_t anbn_sum = 0;
    if (fp32_emul or is_fp8 or no_opt) {
#ifndef __PPC__
#pragma omp simd
#endif
        for (i = 0; i < N; i++) {
            fp_mult(a[i],
                    b[i],
                    &anbn_exp[i],
                    &anbn_sig[i],
                    &anbn_is_zero[i],
                    &anbn_is_inf[i],
                    &anbn_is_nan[i],
                    &anbn_is_neg[i],
                    fp32_emul,
                    i % 4,
                    is_fp8);
            if (anbn_exp[i] > max_exp)
                max_exp = anbn_exp[i];
            // printf("a %08x b %08x ab_exp %02x ab_sig %016llx flags %d %d %d %d\n",a[i], b[i], anbn_exp[i],
            // anbn_sig[i], anbn_is_zero[i], anbn_is_inf[i], anbn_is_nan[i], anbn_is_neg[i]);
            //        if(anbn_is_zero[i]==0)
            //            anbn_sum_not_zero = 1;
        }
    } else {
#ifndef __PPC__
#pragma omp simd
#endif
        for (i = 0; i < N; i++) {
            fp_mult_nonf32f8(a[i],
                             b[i],
                             &anbn_exp[i],
                             &anbn_sig[i],
                             &anbn_is_zero[i],
                             &anbn_is_inf[i],
                             &anbn_is_nan[i],
                             &anbn_is_neg[i],
                             fp32_emul,
                             i % 4,
                             is_fp8);
            if (anbn_exp[i] > max_exp)
                max_exp = anbn_exp[i];
            // printf("a %08x b %08x ab_exp %02x ab_sig %016llx flags %d %d %d %d\n",a[i], b[i], anbn_exp[i],
            // anbn_sig[i], anbn_is_zero[i], anbn_is_inf[i], anbn_is_nan[i], anbn_is_neg[i]);
            //        if(anbn_is_zero[i]==0)
            //            anbn_sum_not_zero = 1;
        }
    }

    // printf("max_exp %02x\n",max_exp);
    anbn_sum_is_inf = 0;
    anbn_sum_is_nan = 0;
#ifndef __PPC__
// TODO [SW-157138]: w/a compilation issue with clang:
//  error: loop not vectorized: the optimizer was unable to perform the requested transformation; the transformation
//  might be disabled or specified as part of an unsupported transformation ordering
//  [-Werror,-Wpass-failed=transform-warning]
#ifndef __clang__
#pragma omp simd
#endif // __clang__
#endif
    for (i = 0; i < N; i++) {
        exp_diff = max_exp - anbn_exp[i];
        if (exp_diff > (K + 22)) // K-26+48
            exp_diff = (K + 22);
        // mult result is 2.46 (and not 2.20), so it is already shifted left by 26
        if (K > 26)
            anbn_sig[i] = (anbn_sig[i] << (K - 26)) >> exp_diff;
        else
            anbn_sig[i] = (anbn_sig[i] >> (26 - K)) >> exp_diff;
        //		if(anbn_sig[i] < 0 && exp_diff==(K+22)) //if negative number was all lost except its sign - zero it
        //			anbn_sig[i] = 0;
        // printf("ab_sig_shifted %016llx exp_diff %d\n", anbn_sig[i], exp_diff);
        if (anbn_is_neg[i])
            anbn_sig[i] = -(anbn_sig[i]);

        sum_all_neg &= anbn_is_neg[i];
        anbn_sum += anbn_sig[i];
        anbn_sum_is_inf = (anbn_sum_is_inf != 0) | (anbn_is_inf[i] != 0);
        anbn_sum_is_nan = (anbn_sum_is_nan != 0) | (anbn_is_nan[i] != 0);
        anbn_sum_is_neg = anbn_sum_is_nan ? 0 : (anbn_is_inf[i] != 0) ? anbn_is_neg[i] : anbn_sum_is_neg;
        sum_plus_inf    = (sum_plus_inf != 0) | (anbn_is_inf[i] != 0 && anbn_is_neg[i] == 0);
        sum_minus_inf   = (sum_minus_inf != 0) | (anbn_is_inf[i] != 0 && anbn_is_neg[i] != 0);
    }
    anbn_sum_is_nan  = (anbn_sum_is_nan != 0) | (sum_plus_inf != 0 && sum_minus_inf != 0);
    anbn_sum_is_inf  = (anbn_sum_is_inf != 0) & (anbn_sum_is_nan == 0);
    anbn_sum_is_zero = (anbn_sum == 0) && (anbn_sum_is_inf == 0) && (anbn_sum_is_nan == 0);
    anbn_sum_is_neg  = (anbn_sum_is_nan == 0 && anbn_sum_is_inf == 0)
                          ? (((anbn_sum_is_zero != 0) && (sum_all_neg != 0)) || (anbn_sum < 0))
                          : anbn_sum_is_neg;

    if (c_after_norm) {
        fp_convert(max_exp,
                   anbn_sum,
                   anbn_sum_is_zero,
                   anbn_sum_is_inf,
                   anbn_sum_is_nan,
                   anbn_sum_is_neg,
                   0,
                   &result,
                   round_mode,
                   K);
        // printf("res_tree %08x\n", result);
        result = fma_fp32(result, 0x3f800000, c, round_mode);
    } else {
        // printf("anbn_sum %016llx flags %d %d %d %d \n", anbn_sum, anbn_sum_is_zero, anbn_sum_is_inf, anbn_sum_is_nan,
        // anbn_sum_is_neg);
        anbn_sum = (uint64_t)anbn_sum << 2ll; // 2 more bits for guard and round bits
        max_exp  = max_exp - 2;

        fp_mult(c, 0x3f800000, &c_exp, &c_sig, &c_is_zero, &c_is_inf, &c_is_nan, &c_is_neg, 0, 0, 0);
        // mult result is 2.46 (and not 2.20), so it is already shifted left by 26
        if (K > 26)
            c_sig = (c_sig << (K - 26));
        else
            c_sig = (c_sig >> (26 - K));

        if (c_is_neg)
            c_sig = -c_sig;

        c_sig = (uint64_t)c_sig << 2ll; // 2 more bits for guard and round bits
        c_exp = c_exp - 2;

        // bug fix - align C 5 more bits to the left, to be in the same location as the upper-most possible leading 1 of
        // anbn_sum
        c_sig = (uint64_t)c_sig << 5ll;
        c_exp = c_exp - 5;

        anbn_sum = ((uint64_t)anbn_sum) << (26ll - K);
        c_sig    = (uint64_t)c_sig << (26ll - K);
        // printf("c_sig %016llx c_exp %02x flags %d %d %d %d \n", c_sig, c_exp, c_is_zero, c_is_inf, c_is_nan,
        // c_is_neg);

        uint32_t num_leading_digits =
            28; // 28: 24 are discarded, 1 more discarded when adding sticky, 1 for sign, 1 to prevent overflow when
                // doing 2's complement, +1 just to be on the safe side :-)

        uint32_t leading_ones = (1 << (num_leading_digits + 1)) - 1;
        if (lsbs(*(uint64_t*)&anbn_sum, 63, 63 - num_leading_digits) == 0 ||
            lsbs(*(uint64_t*)&anbn_sum, 63, 63 - num_leading_digits) == leading_ones) // 28 sign bits
        {
            // discard 24 sign bits
            anbn_sum = (uint64_t)anbn_sum << ((int64_t)(24ll)); // + K - 26ll));
            max_exp  = max_exp - (24); // + K - 26);
        }

        uint8_t res_sticky = 0;
        uint8_t res_pad_1  = 0; // will not be used

        fp_add(max_exp,
               anbn_sum,
               c_exp,
               c_sig,
               &res_exp,
               &res_sig,
               anbn_sum_is_zero,
               anbn_sum_is_inf,
               anbn_sum_is_nan,
               anbn_sum_is_neg,
               0,
               c_is_zero,
               c_is_inf,
               c_is_nan,
               c_is_neg,
               0,
               &res_is_zero,
               &res_is_inf,
               &res_is_nan,
               &res_is_neg,
               &res_sticky,
               &res_pad_1,
               round_mode);

        // printf("res_exp %08x res_sig %016llx flags %d %d %d %d %d\n",res_exp, res_sig, res_is_zero, res_is_inf,
        // res_is_nan, res_is_neg, res_sticky); here need to convert back, normalize and round_mode
        fp_convert(res_exp,
                   res_sig,
                   res_is_zero,
                   res_is_inf,
                   res_is_nan,
                   res_is_neg,
                   res_sticky,
                   &result,
                   round_mode,
                   26); // K);
        // fp_convert(max_exp, anbn_sum, anbn_sum_is_zero, anbn_sum_is_inf, anbn_sum_is_nan, anbn_sum_is_neg, 0,
        // &result, round_mode, K); printf("result %08x\n",result);
    }

    bool final_result_is_zero = is_zero_fp32(result);
    // zero sign exception
    // if(anbn_sum_not_zero!=0 && sum_all_neg!=0 && c_is_zero!=0 && res_is_zero!=0)
    if ((anbn_sum < 0) && c_is_zero != 0 && final_result_is_zero != 0) {
        result =
            ibs(result,
                31,
                31,
                1); // force negative sign for zero result when anbn sum is negative and not zero but c is exactly zero)
    }

    return result;
}

void select_rounding_mode_dp(double sum, float c, uint32_t c_exp);

uint32_t fma_mul_add_tree_double_prec(uint32_t* a, uint32_t* b, uint32_t c, uint8_t N, uint8_t K, bool is_fp8);

uint32_t fma_mul_add_tree_4(uint32_t a0,
                            uint32_t b0,
                            uint32_t a1,
                            uint32_t b1,
                            uint32_t a2,
                            uint32_t b2,
                            uint32_t a3,
                            uint32_t b3,
                            uint32_t c,
                            uint8_t  round_mode,
                            uint8_t  K,
                            bool     fp32_emul,
                            bool     c_after_norm);

uint32_t fma_mul_add_tree_8(uint32_t a0,
                            uint32_t b0,
                            uint32_t a1,
                            uint32_t b1,
                            uint32_t a2,
                            uint32_t b2,
                            uint32_t a3,
                            uint32_t b3,
                            uint32_t a4,
                            uint32_t b4,
                            uint32_t a5,
                            uint32_t b5,
                            uint32_t a6,
                            uint32_t b6,
                            uint32_t a7,
                            uint32_t b7,
                            uint32_t c,
                            uint8_t  round_mode,
                            uint8_t  K,
                            bool     fp32_emul,
                            bool     c_after_norm);

uint32_t fma_mul_add_tree_N8_K4_add_C_after_norm(uint32_t a0,
                                                 uint32_t b0,
                                                 uint32_t a1,
                                                 uint32_t b1,
                                                 uint32_t a2,
                                                 uint32_t b2,
                                                 uint32_t a3,
                                                 uint32_t b3,
                                                 uint32_t a4,
                                                 uint32_t b4,
                                                 uint32_t a5,
                                                 uint32_t b5,
                                                 uint32_t a6,
                                                 uint32_t b6,
                                                 uint32_t a7,
                                                 uint32_t b7,
                                                 uint32_t c,
                                                 uint8_t  round_mode);

uint32_t fma_mul_add_tree_N8_K12_add_C_after_norm(uint32_t a0,
                                                  uint32_t b0,
                                                  uint32_t a1,
                                                  uint32_t b1,
                                                  uint32_t a2,
                                                  uint32_t b2,
                                                  uint32_t a3,
                                                  uint32_t b3,
                                                  uint32_t a4,
                                                  uint32_t b4,
                                                  uint32_t a5,
                                                  uint32_t b5,
                                                  uint32_t a6,
                                                  uint32_t b6,
                                                  uint32_t a7,
                                                  uint32_t b7,
                                                  uint32_t c,
                                                  uint8_t  round_mode);

uint32_t fma_mul_add_tree_N8_K26_add_C_after_norm(uint32_t a0,
                                                  uint32_t b0,
                                                  uint32_t a1,
                                                  uint32_t b1,
                                                  uint32_t a2,
                                                  uint32_t b2,
                                                  uint32_t a3,
                                                  uint32_t b3,
                                                  uint32_t a4,
                                                  uint32_t b4,
                                                  uint32_t a5,
                                                  uint32_t b5,
                                                  uint32_t a6,
                                                  uint32_t b6,
                                                  uint32_t a7,
                                                  uint32_t b7,
                                                  uint32_t c,
                                                  uint8_t  round_mode);

uint32_t fma_mul_add_tree_tf32_N8_K4_add_C_before_norm(uint32_t a0,
                                                       uint32_t b0,
                                                       uint32_t a1,
                                                       uint32_t b1,
                                                       uint32_t a2,
                                                       uint32_t b2,
                                                       uint32_t a3,
                                                       uint32_t b3,
                                                       uint32_t a4,
                                                       uint32_t b4,
                                                       uint32_t a5,
                                                       uint32_t b5,
                                                       uint32_t a6,
                                                       uint32_t b6,
                                                       uint32_t a7,
                                                       uint32_t b7,
                                                       uint32_t c,
                                                       uint8_t  round_mode);

uint32_t fma_mul_add_tree_tf32_N8_K12_add_C_before_norm(uint32_t a0,
                                                        uint32_t b0,
                                                        uint32_t a1,
                                                        uint32_t b1,
                                                        uint32_t a2,
                                                        uint32_t b2,
                                                        uint32_t a3,
                                                        uint32_t b3,
                                                        uint32_t a4,
                                                        uint32_t b4,
                                                        uint32_t a5,
                                                        uint32_t b5,
                                                        uint32_t a6,
                                                        uint32_t b6,
                                                        uint32_t a7,
                                                        uint32_t b7,
                                                        uint32_t c,
                                                        uint8_t  round_mode);

static inline __attribute__((always_inline)) uint32_t fma_mul_add_tree_tf32_N8_K26_add_C_before_norm(uint32_t a0,
                                                                                                     uint32_t b0,
                                                                                                     uint32_t a1,
                                                                                                     uint32_t b1,
                                                                                                     uint32_t a2,
                                                                                                     uint32_t b2,
                                                                                                     uint32_t a3,
                                                                                                     uint32_t b3,
                                                                                                     uint32_t a4,
                                                                                                     uint32_t b4,
                                                                                                     uint32_t a5,
                                                                                                     uint32_t b5,
                                                                                                     uint32_t a6,
                                                                                                     uint32_t b6,
                                                                                                     uint32_t a7,
                                                                                                     uint32_t b7,
                                                                                                     uint32_t c,
                                                                                                     uint8_t round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 0);
}

static inline __attribute__((always_inline)) void
fma_mul_add_tree_tf32_N8_K26_add_C_before_norm_vec_bf16(const uint16_t vecA[][8],
                                                        const uint16_t vecB[][8],
                                                        uint32_t       vecC[][32],
                                                        uint32_t       height,
                                                        uint32_t       width)
{
    uint8_t  round_mode = RND_TO_NE;
    uint32_t a[8];
    uint32_t b[8];
    for (unsigned hIdx = 0; hIdx < height; hIdx++) {
#ifndef __PPC__
#pragma omp simd
#endif
        for (unsigned wIdx = 0; wIdx < width; wIdx++) {
            for (int cd = 0; cd < 8; cd++) {
                a[cd] = bf16_to_fp32(vecA[hIdx][cd], 0);
                b[cd] = bf16_to_fp32(vecB[wIdx][cd], 0);
            }

            vecC[hIdx][wIdx] = fma_mul_add_tree_n(a, b, vecC[hIdx][wIdx], round_mode, 8, 26, 0, 0, 0);
        }
    }
}

uint32_t fma_mul_add_tree_tf32_N8_K26_add_C_before_norm_dp(uint32_t a0,
                                                           uint32_t b0,
                                                           uint32_t a1,
                                                           uint32_t b1,
                                                           uint32_t a2,
                                                           uint32_t b2,
                                                           uint32_t a3,
                                                           uint32_t b3,
                                                           uint32_t a4,
                                                           uint32_t b4,
                                                           uint32_t a5,
                                                           uint32_t b5,
                                                           uint32_t a6,
                                                           uint32_t b6,
                                                           uint32_t a7,
                                                           uint32_t b7,
                                                           uint32_t c,
                                                           uint8_t  round_mode);

uint32_t fma_mul_add_tree_tf32_N4_K26_add_C_before_norm(uint32_t a0,
                                                        uint32_t b0,
                                                        uint32_t a1,
                                                        uint32_t b1,
                                                        uint32_t a2,
                                                        uint32_t b2,
                                                        uint32_t a3,
                                                        uint32_t b3,
                                                        uint32_t c,
                                                        uint8_t  round_mode,
                                                        bool     clip_fp);

uint32_t fma_mul_add_tree_bf16_N8_K26_add_C_before_norm(uint16_t a0,
                                                        uint16_t b0,
                                                        uint16_t a1,
                                                        uint16_t b1,
                                                        uint16_t a2,
                                                        uint16_t b2,
                                                        uint16_t a3,
                                                        uint16_t b3,
                                                        uint16_t a4,
                                                        uint16_t b4,
                                                        uint16_t a5,
                                                        uint16_t b5,
                                                        uint16_t a6,
                                                        uint16_t b6,
                                                        uint16_t a7,
                                                        uint16_t b7,
                                                        uint32_t c,
                                                        uint8_t  round_mode);

uint32_t fma_mul_add_tree_bf16_N8_K26_add_C_before_norm_dp(uint16_t a0,
                                                           uint16_t b0,
                                                           uint16_t a1,
                                                           uint16_t b1,
                                                           uint16_t a2,
                                                           uint16_t b2,
                                                           uint16_t a3,
                                                           uint16_t b3,
                                                           uint16_t a4,
                                                           uint16_t b4,
                                                           uint16_t a5,
                                                           uint16_t b5,
                                                           uint16_t a6,
                                                           uint16_t b6,
                                                           uint16_t a7,
                                                           uint16_t b7,
                                                           uint32_t c,
                                                           uint8_t  round_mode);

uint32_t fma_mul_add_tree_fp16_N8_K26_add_C_before_norm(uint16_t a0,
                                                        uint16_t b0,
                                                        uint16_t a1,
                                                        uint16_t b1,
                                                        uint16_t a2,
                                                        uint16_t b2,
                                                        uint16_t a3,
                                                        uint16_t b3,
                                                        uint16_t a4,
                                                        uint16_t b4,
                                                        uint16_t a5,
                                                        uint16_t b5,
                                                        uint16_t a6,
                                                        uint16_t b6,
                                                        uint16_t a7,
                                                        uint16_t b7,
                                                        uint32_t c,
                                                        uint8_t  round_mode);

uint32_t fma_mul_add_tree_N8_K26_add_C_before_norm_fp32_emul(uint32_t a0,
                                                             uint32_t b0,
                                                             uint32_t a1,
                                                             uint32_t b1,
                                                             uint32_t a2,
                                                             uint32_t b2,
                                                             uint32_t a3,
                                                             uint32_t b3,
                                                             uint32_t a4,
                                                             uint32_t b4,
                                                             uint32_t a5,
                                                             uint32_t b5,
                                                             uint32_t a6,
                                                             uint32_t b6,
                                                             uint32_t a7,
                                                             uint32_t b7,
                                                             uint32_t c,
                                                             uint8_t  round_mode);

inline __attribute__((always_inline)) uint32_t
fma_mul_add_tree_fp32_N2_K26_add_C_before_norm_fp32_emul(uint32_t a0,
                                                         uint32_t b0,
                                                         uint32_t a1,
                                                         uint32_t b1,
                                                         uint32_t c,
                                                         uint8_t  round_mode)
{
    uint32_t a[8] = {a0 & 0xff800fff,
                     a0 & 0xff800fff,
                     a0 & 0xfffff000,
                     a0 & 0xfffff000,
                     a1 & 0xff800fff,
                     a1 & 0xff800fff,
                     a1 & 0xfffff000,
                     a1 & 0xfffff000};
    uint32_t b[8] = {b0 & 0xff800fff,
                     b0 & 0xfffff000,
                     b0 & 0xff800fff,
                     b0 & 0xfffff000,
                     b1 & 0xff800fff,
                     b1 & 0xfffff000,
                     b1 & 0xff800fff,
                     b1 & 0xfffff000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0);
}

uint32_t fma_mul_add_tree_fp32_N2_K26_add_C_before_norm_fp32_emul_dp(uint32_t a0,
                                                                     uint32_t b0,
                                                                     uint32_t a1,
                                                                     uint32_t b1,
                                                                     uint32_t c,
                                                                     uint8_t  round_mode);

uint32_t fma_mul_add_tree_N16_K26_add_C_before_norm(uint16_t a0,
                                                    uint16_t b0,
                                                    uint16_t a1,
                                                    uint16_t b1,
                                                    uint16_t a2,
                                                    uint16_t b2,
                                                    uint16_t a3,
                                                    uint16_t b3,
                                                    uint16_t a4,
                                                    uint16_t b4,
                                                    uint16_t a5,
                                                    uint16_t b5,
                                                    uint16_t a6,
                                                    uint16_t b6,
                                                    uint16_t a7,
                                                    uint16_t b7,
                                                    uint16_t a8,
                                                    uint16_t b8,
                                                    uint16_t a9,
                                                    uint16_t b9,
                                                    uint16_t a10,
                                                    uint16_t b10,
                                                    uint16_t a11,
                                                    uint16_t b11,
                                                    uint16_t a12,
                                                    uint16_t b12,
                                                    uint16_t a13,
                                                    uint16_t b13,
                                                    uint16_t a14,
                                                    uint16_t b14,
                                                    uint16_t a15,
                                                    uint16_t b15,
                                                    uint32_t c,
                                                    uint8_t  round_mode);

uint32_t fma_mul_add_tree_N16_K26_add_C_before_norm_dp(uint16_t a0,
                                                       uint16_t b0,
                                                       uint16_t a1,
                                                       uint16_t b1,
                                                       uint16_t a2,
                                                       uint16_t b2,
                                                       uint16_t a3,
                                                       uint16_t b3,
                                                       uint16_t a4,
                                                       uint16_t b4,
                                                       uint16_t a5,
                                                       uint16_t b5,
                                                       uint16_t a6,
                                                       uint16_t b6,
                                                       uint16_t a7,
                                                       uint16_t b7,
                                                       uint16_t a8,
                                                       uint16_t b8,
                                                       uint16_t a9,
                                                       uint16_t b9,
                                                       uint16_t a10,
                                                       uint16_t b10,
                                                       uint16_t a11,
                                                       uint16_t b11,
                                                       uint16_t a12,
                                                       uint16_t b12,
                                                       uint16_t a13,
                                                       uint16_t b13,
                                                       uint16_t a14,
                                                       uint16_t b14,
                                                       uint16_t a15,
                                                       uint16_t b15,
                                                       uint32_t c,
                                                       uint8_t  round_mode);

uint32_t fma_mul_add_tree_N16_K4_add_C_before_norm(uint16_t a0,
                                                   uint16_t b0,
                                                   uint16_t a1,
                                                   uint16_t b1,
                                                   uint16_t a2,
                                                   uint16_t b2,
                                                   uint16_t a3,
                                                   uint16_t b3,
                                                   uint16_t a4,
                                                   uint16_t b4,
                                                   uint16_t a5,
                                                   uint16_t b5,
                                                   uint16_t a6,
                                                   uint16_t b6,
                                                   uint16_t a7,
                                                   uint16_t b7,
                                                   uint16_t a8,
                                                   uint16_t b8,
                                                   uint16_t a9,
                                                   uint16_t b9,
                                                   uint16_t a10,
                                                   uint16_t b10,
                                                   uint16_t a11,
                                                   uint16_t b11,
                                                   uint16_t a12,
                                                   uint16_t b12,
                                                   uint16_t a13,
                                                   uint16_t b13,
                                                   uint16_t a14,
                                                   uint16_t b14,
                                                   uint16_t a15,
                                                   uint16_t b15,
                                                   uint32_t c,
                                                   uint8_t  round_mode);

uint32_t fma_mul_add_tree_N16_K4_add_C_before_norm_dp(uint16_t a0,
                                                      uint16_t b0,
                                                      uint16_t a1,
                                                      uint16_t b1,
                                                      uint16_t a2,
                                                      uint16_t b2,
                                                      uint16_t a3,
                                                      uint16_t b3,
                                                      uint16_t a4,
                                                      uint16_t b4,
                                                      uint16_t a5,
                                                      uint16_t b5,
                                                      uint16_t a6,
                                                      uint16_t b6,
                                                      uint16_t a7,
                                                      uint16_t b7,
                                                      uint16_t a8,
                                                      uint16_t b8,
                                                      uint16_t a9,
                                                      uint16_t b9,
                                                      uint16_t a10,
                                                      uint16_t b10,
                                                      uint16_t a11,
                                                      uint16_t b11,
                                                      uint16_t a12,
                                                      uint16_t b12,
                                                      uint16_t a13,
                                                      uint16_t b13,
                                                      uint16_t a14,
                                                      uint16_t b14,
                                                      uint16_t a15,
                                                      uint16_t b15,
                                                      uint32_t c,
                                                      uint8_t  round_mode);

uint32_t fma_mul_add_tree_fp8_N16_K26_add_C_before_norm(uint8_t  a0,
                                                        uint8_t  b0,
                                                        uint8_t  a1,
                                                        uint8_t  b1,
                                                        uint8_t  a2,
                                                        uint8_t  b2,
                                                        uint8_t  a3,
                                                        uint8_t  b3,
                                                        uint8_t  a4,
                                                        uint8_t  b4,
                                                        uint8_t  a5,
                                                        uint8_t  b5,
                                                        uint8_t  a6,
                                                        uint8_t  b6,
                                                        uint8_t  a7,
                                                        uint8_t  b7,
                                                        uint8_t  a8,
                                                        uint8_t  b8,
                                                        uint8_t  a9,
                                                        uint8_t  b9,
                                                        uint8_t  a10,
                                                        uint8_t  b10,
                                                        uint8_t  a11,
                                                        uint8_t  b11,
                                                        uint8_t  a12,
                                                        uint8_t  b12,
                                                        uint8_t  a13,
                                                        uint8_t  b13,
                                                        uint8_t  a14,
                                                        uint8_t  b14,
                                                        uint8_t  a15,
                                                        uint8_t  b15,
                                                        uint32_t c,
                                                        uint8_t  round_mode,
                                                        uint8_t  exp_width,
                                                        uint8_t  man_width,
                                                        uint8_t  exp_bias_a,
                                                        uint8_t  exp_bias_b);

uint32_t fma_mul_add_tree_fp8_N16_K4_add_C_before_norm(uint8_t  a0,
                                                       uint8_t  b0,
                                                       uint8_t  a1,
                                                       uint8_t  b1,
                                                       uint8_t  a2,
                                                       uint8_t  b2,
                                                       uint8_t  a3,
                                                       uint8_t  b3,
                                                       uint8_t  a4,
                                                       uint8_t  b4,
                                                       uint8_t  a5,
                                                       uint8_t  b5,
                                                       uint8_t  a6,
                                                       uint8_t  b6,
                                                       uint8_t  a7,
                                                       uint8_t  b7,
                                                       uint8_t  a8,
                                                       uint8_t  b8,
                                                       uint8_t  a9,
                                                       uint8_t  b9,
                                                       uint8_t  a10,
                                                       uint8_t  b10,
                                                       uint8_t  a11,
                                                       uint8_t  b11,
                                                       uint8_t  a12,
                                                       uint8_t  b12,
                                                       uint8_t  a13,
                                                       uint8_t  b13,
                                                       uint8_t  a14,
                                                       uint8_t  b14,
                                                       uint8_t  a15,
                                                       uint8_t  b15,
                                                       uint32_t c,
                                                       uint8_t  round_mode,
                                                       uint8_t  exp_width,
                                                       uint8_t  man_width,
                                                       uint8_t  exp_bias_a,
                                                       uint8_t  exp_bias_b);

uint32_t dp4_ref_double(uint32_t a0,
                        uint32_t b0,
                        uint32_t a1,
                        uint32_t b1,
                        uint32_t a2,
                        uint32_t b2,
                        uint32_t a3,
                        uint32_t b3,
                        uint32_t c,
                        uint8_t  round_mode,
                        bool     c_after_norm);

uint32_t dp8_ref_double(uint32_t a0,
                        uint32_t b0,
                        uint32_t a1,
                        uint32_t b1,
                        uint32_t a2,
                        uint32_t b2,
                        uint32_t a3,
                        uint32_t b3,
                        uint32_t a4,
                        uint32_t b4,
                        uint32_t a5,
                        uint32_t b5,
                        uint32_t a6,
                        uint32_t b6,
                        uint32_t a7,
                        uint32_t b7,
                        uint32_t c,
                        uint8_t  round_mode,
                        bool     c_after_norm);

uint32_t dp4_fma_float_ref(uint32_t a0,
                           uint32_t b0,
                           uint32_t a1,
                           uint32_t b1,
                           uint32_t a2,
                           uint32_t b2,
                           uint32_t a3,
                           uint32_t b3,
                           uint32_t acc,
                           uint8_t  round_mode);

uint32_t dp8_fma_float_ref(uint32_t a0,
                           uint32_t b0,
                           uint32_t a1,
                           uint32_t b1,
                           uint32_t a2,
                           uint32_t b2,
                           uint32_t a3,
                           uint32_t b3,
                           uint32_t a4,
                           uint32_t b4,
                           uint32_t a5,
                           uint32_t b5,
                           uint32_t a6,
                           uint32_t b6,
                           uint32_t a7,
                           uint32_t b7,
                           uint32_t acc,
                           uint8_t  round_mode);

uint32_t
dp2_ref_double(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1, uint32_t c, uint8_t round_mode, bool c_after_norm);

// FP8

static constexpr uint32_t SIGN_MASK_FP8           = 0x80;
static constexpr uint32_t EXPONENT_MASK_FP8_152   = 0x7C;
static constexpr uint32_t EXPONENT_OFFSET_FP8_152 = 2;

static inline __attribute__((always_inline)) bool fp8_is_zero(uint8_t val)
{
    return (val & (~SIGN_MASK_FP8)) ? 0 : 1;
}

static inline __attribute__((always_inline)) bool fp8_is_infinity(uint8_t val, uint8_t exponent_offset_fp8)
{
    bool isAllExponentBitsSet  = sbs(val, 6, exponent_offset_fp8) == sbs(0xff, 6, exponent_offset_fp8);
    bool isAllMantissaBitsZero = (sbs(val, exponent_offset_fp8 - 1, 0) == 0);
    return (isAllExponentBitsSet & isAllMantissaBitsZero);
}

static inline __attribute__((always_inline)) bool fp8_is_nan(uint8_t val, uint8_t exponent_offset_fp8)
{
    bool isAllExponentBitsSet = sbs(val, 6, exponent_offset_fp8) == sbs(0xff, 6, exponent_offset_fp8);
    bool isAnyMantissaBitSet  = (sbs(val, exponent_offset_fp8 - 1, 0) != 0);
    return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

static inline __attribute__((always_inline)) bool fp8_is_negative(uint8_t val)
{
    return ((val & SIGN_MASK_FP8) == SIGN_MASK_FP8);
}

static inline __attribute__((always_inline)) bool fp8_is_denormal(uint8_t val, uint8_t exponent_offset_fp8)
{ // Do not consider zero as denormal
    bool isAllExponentBitsZero = sbs(val, 6, exponent_offset_fp8) == 0;
    bool isAnyMantissaBitSet   = (sbs(val, exponent_offset_fp8 - 1, 0) != 0);
    return (isAllExponentBitsZero & isAnyMantissaBitSet);
}

uint32_t fp8_to_fp32(uint8_t input, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias, bool clip_fp, bool is_fp19);
uint16_t fp8_to_fp16(uint8_t input, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias, bool clip_fp);
uint32_t fma_fp8_fp32(uint8_t  a,
                      uint8_t  b,
                      uint32_t c,
                      uint8_t  round_mode,
                      uint8_t  exp_width,
                      uint8_t  man_width,
                      uint8_t  exp_bias);
uint32_t fma_2xfp8_fp32(uint8_t  a0,
                        uint8_t  b0,
                        uint8_t  a1,
                        uint8_t  b1,
                        uint32_t c,
                        uint8_t  round_mode,
                        uint8_t  exp_width,
                        uint8_t  man_width,
                        uint8_t  exp_bias);
uint8_t  fp32_to_fp8(float   input,
                     uint8_t exp_width,
                     uint8_t man_width,
                     uint8_t exp_bias,
                     int     roundingMode,
                     int32_t lfsrVal,
                     bool    ftz_fp8,
                     bool    clip_fp);
void     set_rounding_mode(uint8_t round_mode);
int      get_rounding_mode();
void     fesetround_local(int mode);

uint32_t
executeOp(uint32_t src1_32bit, uint32_t src2_32bit, uint8_t OpcodeRMW, uint8_t DataTypeRMW, uint8_t round_mode);
} // namespace Gaudi2
