/*
 *  This File holds all FMA functions
 *  It is used for ASIC (RTL) verification
 *  DO NOT MODIFY IT without discussing with Hilla Ben-Yaacov
 *
 */

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cstring>
#include "fs_fma_gaudi3.h"

namespace gaudi3
{

float fs_bf16_to_f32(uint16_t input)
{
    uint32_t val_32b = (uint32_t)input << 16;
    float    f32;
    memcpy(&f32, &val_32b, sizeof(float));
    return f32;
}

// BF16 data type - accumulation into BF16
uint16_t fma_bfp16(uint16_t a, uint16_t b, uint16_t c, uint8_t round_mode, bool dnorm_ftz)
{

    // compare reuslt with FP32 FMA
    // uint32_t a_32bit = a << 16;
    // uint32_t b_32bit = b << 16;
    // uint32_t c_32bit = c << 16;
    // uint32_t result_32bit;
    // // result_32bit = fma_bfp16_fp32(a, b, c_32bit, round_mode);
    // // result of x86 FP32, with flush at input, no flush at output
    // result_32bit = fma_fp32_no_flush(a_32bit, b_32bit, c_32bit, round_mode);
    // uint16_t result_16bit;
    // result_16bit = fp32_to_bf16(*(float*)&result_32bit, round_mode, 0, 0, 0, 1);
    // if (dnorm_ftz)
    // {
    // // flush result_16bit if denormal
    // if (is_denorm_bfp16(result_16bit)) {
    // // flush result to 0
    // if ((round_mode == RND_TO_PINF) && (sbs(result_16bit, 15, 15) == 0))
    // result_16bit = 0x0080; // res=+min_normal
    // else if ((round_mode == RND_TO_NINF) && (sbs(result_16bit, 15, 15) == 1))
    // result_16bit = 0x8080; // res=-min_normal
    // else
    // result_16bit = ibs(result_16bit, 14, 0, 0); // res=+-0
    // }
    // }

    // our reference code
    uint16_t result       = 0;
    bool     result_ready = 0;

    uint8_t a_is_dnorm = is_denorm_bfp16(a);
    uint8_t b_is_dnorm = is_denorm_bfp16(b);
    uint8_t c_is_dnorm = is_denorm_bfp16(c);

    if (dnorm_ftz)
    {
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (is_denorm_bfp16(a))
            a = ibs(a, 14, 0, 0); // a=+-0

        if (is_denorm_bfp16(b))
            b = ibs(b, 14, 0, 0); // b=+-0

        if (is_denorm_bfp16(c))
            c = ibs(c, 14, 0, 0); // c=+-0
            //------------------------
    }

    uint16_t a_exp = sbs(a, 14, 7);
    uint16_t b_exp = sbs(b, 14, 7);
    uint16_t c_exp = sbs(c, 14, 7);
    uint32_t a_man = sbs(a, 6, 0);
    uint32_t b_man = sbs(b, 6, 0);
    uint32_t c_man = sbs(c, 6, 0);
    uint8_t  a_sgn = sbs(a, 15, 15);
    uint8_t  b_sgn = sbs(b, 15, 15);
    uint8_t  c_sgn = sbs(c, 15, 15);

    int16_t a_exp_unbiased;
    int16_t b_exp_unbiased;
    int16_t c_exp_unbiased;

    uint8_t ab_sgn    = a_sgn ^ b_sgn;
    uint8_t sub_ind   = ab_sgn ^ c_sgn;
    uint8_t a_is_zero = is_zero_bfp16(a);
    uint8_t b_is_zero = is_zero_bfp16(b);
    uint8_t c_is_zero = is_zero_bfp16(c);
    uint8_t leading_1_ind;

    if (!a_is_dnorm && !a_is_zero) {
        if (!a_is_zero)
            a_man = cbs(1, a_man, 7);
        a_exp_unbiased = a_exp - 127;
    } else {
        a_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(a_man);
        if (leading_1_ind <= 6) {
            a_man          = a_man << (7 - leading_1_ind);
            a_exp_unbiased = a_exp_unbiased - (7 - leading_1_ind);
        }
    }

    if (!b_is_dnorm && !b_is_zero) {
        if (!b_is_zero)
            b_man = cbs(1, b_man, 7);
        b_exp_unbiased = b_exp - 127;
    } else {
        b_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(b_man);
        if (leading_1_ind <= 6) {
            b_man          = b_man << (7 - leading_1_ind);
            b_exp_unbiased = b_exp_unbiased - (7 - leading_1_ind);
        }
    }

    if (!c_is_dnorm && !c_is_zero) {
        if (!c_is_zero)
            c_man = cbs(1, c_man, 7);
        c_exp_unbiased = c_exp - 127;
    } else {
        c_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(c_man);
        if (leading_1_ind <= 6) {
            c_man          = c_man << (7 - leading_1_ind);
            c_exp_unbiased = c_exp_unbiased - (7 - leading_1_ind);
        }
    }
    uint32_t ab_man          = a_man * b_man;
    int16_t  ab_exp_unbiased = a_exp_unbiased + b_exp_unbiased;
    uint8_t  res_sgn;

    //  uint8_t a_exp_is_zero = (sbs(a,14,7)==0);
    //  uint8_t b_exp_is_zero = (sbs(b,14,7)==0);
    // uint8_t c_exp_is_zero = (sbs(c,14,7)==0);
    uint8_t a_is_inf = is_inf_bfp16(a);
    uint8_t b_is_inf = is_inf_bfp16(b);
    uint8_t c_is_inf = is_inf_bfp16(c);
    //  uint8_t ab_res_is_inf = is_inf_bfp16(cbs((ab_exp_unbiased+127),ab_man,7));
    uint8_t mul_res_is_zero = (a_is_zero | b_is_zero);
    uint8_t a_is_nan        = is_nan_bfp16(a);
    uint8_t b_is_nan        = is_nan_bfp16(b);
    uint8_t c_is_nan        = is_nan_bfp16(c);

    // exceptions
    uint8_t mul_res_is_inf = (!a_is_zero && b_is_inf) | (!b_is_zero && a_is_inf);

    uint8_t mul_res_is_def_nan = (a_is_zero & b_is_inf) | (b_is_zero & a_is_inf);

    uint8_t mul_res_is_nan = mul_res_is_def_nan | a_is_nan | b_is_nan;

    // Both sources are inf with (different signs and add instruction) or (same signs and sub instruction). i.e. inf-inf
    uint8_t res_is_def_nan = (mul_res_is_inf & c_is_inf & sub_ind);

    // nan handling - if any input is nan or result is nan - return nan
    uint8_t res_is_nan = res_is_def_nan | c_is_nan | mul_res_is_nan;

    // add res is inf - if one of the sources is infinity and it's add instruction OR one of the sources is infinity -
    // select the exception result
    uint8_t res_is_inf = !res_is_nan && ((mul_res_is_inf && c_is_inf && !sub_ind) || (mul_res_is_inf ^ c_is_inf));

    // special results - bypass calculated result
    if (res_is_inf) {
        if (mul_res_is_inf & !c_is_inf)
            res_sgn = ab_sgn;
        else // if (!mul_res_is_inf & c_is_inf)
            res_sgn = c_sgn;

        result       = cbs(res_sgn, 0x7F80, 15); //+-inf
        result_ready = 1;
    }
    if (res_is_nan && (result_ready == 0)) {
        //      if (b_is_nan)
        //        return (b | QNAN_BIT);
        //      if (a_is_nan)
        //        return (a | QNAN_BIT);
        //      if (c_is_nan)
        //        return (c | QNAN_BIT);
        result       = DEFAULT_NAN_BFP16; // indefinite nan value
        result_ready = 1;
    }

    uint8_t sign_zero_exception;
    if (!c_is_zero && mul_res_is_zero)
        sign_zero_exception = c_sgn;
    else if (c_is_zero && !mul_res_is_zero)
        sign_zero_exception = ab_sgn;
    else
        sign_zero_exception = (round_mode == RND_TO_NINF) && (c_sgn ^ ab_sgn);

    sign_zero_exception = sign_zero_exception || (c_sgn && ab_sgn);

    uint8_t c_sgn_zero = (c_is_zero && (a_is_zero || b_is_zero)) ? sign_zero_exception : c_sgn;

    c = c_is_zero ? cbs(c_sgn_zero, 0x0000, 15) : c;

    if ((a_is_zero | b_is_zero) && (result_ready == 0)) {
        result       = c;
        result_ready = 1;
    }

    // align c_man to ab_man (regardless of exponents)
    // compensate for 7 additional fraction bits after multiplication
    c_man = c_man << 7;
    // add 2 lsbs for G and R bits, for both operands (Guard and Round - for rounding at the end)
    ab_man = ab_man << 2;
    c_man  = c_man << 2;

    // Sticky bits
    uint16_t c_s  = 0;
    uint16_t ab_s = 0;

    int16_t res_exp_unbiased;

    int16_t exp_diff = c_exp_unbiased - ab_exp_unbiased;

    if (exp_diff >= 0) {
        c_s = 0;
        if (exp_diff > 18) // larger exp_diff means that we are discarding the entire ab_man
            exp_diff = 18;
        if (exp_diff > 2)
            ab_s = (sbs(ab_man, exp_diff - 3 + 2, 0 + 2) != 0);
        else
            ab_s = 0;
        ab_man           = ab_man >> exp_diff;
        res_exp_unbiased = c_exp_unbiased;
    } else {
        ab_s = 0;
        if (exp_diff < -18) // smaller exp_diff means that we are discarding the entire c_man
            exp_diff = -18;
        if (-exp_diff > 2)
            c_s = (sbs(c_man, -exp_diff - 3 + 2, 0 + 2) != 0);
        else
            c_s = 0;
        c_man            = c_man >> (-exp_diff);
        res_exp_unbiased = ab_exp_unbiased;
    }
    uint16_t res_s = ab_s | c_s;

    uint32_t res_man;
    int32_t  man_diff;

    if (sub_ind) {
        man_diff = cbs(ab_man, ab_s, 1) - cbs(c_man, c_s, 1);
        if (man_diff < 0) {
            res_man = (-man_diff) >> 1;
            res_sgn = c_sgn;
        } else if (man_diff > 0) {
            res_man = man_diff >> 1;
            res_sgn = ab_sgn;
        } else {
            res_man = 0;
            // res_sgn = 0;

            if (ab_s && !c_s)
                res_sgn = ab_sgn;
            else if (!ab_s && c_s)
                res_sgn = c_sgn;
            else
                res_sgn =
                    (round_mode ==
                     RND_TO_NINF); // x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        }
    } else {
        res_man = ab_man + c_man;
        res_sgn = c_sgn;
    }

    // find leading 1
    leading_1_ind = lzd(res_man);

    // normalize
    if (leading_1_ind == 18) {
        res_s            = res_s | (sbs(res_man, 1, 0) != 0);
        res_man          = res_man >> 2;
        res_exp_unbiased = res_exp_unbiased + 2;
    } else if (leading_1_ind == 17) {
        res_s            = res_s | sbs(res_man, 0, 0);
        res_man          = res_man >> 1;
        res_exp_unbiased = res_exp_unbiased + 1;
    } else if (leading_1_ind <= 16) {
        // res_s = res_s;
        res_man          = res_man << (16 - leading_1_ind);
        res_exp_unbiased = res_exp_unbiased - (16 - leading_1_ind);
    } else {
        // res_man is 0
        if (res_s == 0) // result is 0, and grs are 0, no need for rounding
            res_exp_unbiased = -127;
        else
            res_exp_unbiased = res_exp_unbiased - 19;
    }

    // separate GR bits from mantissa
    uint8_t res_g = sbs(res_man, 8, 8);
    uint8_t res_r = sbs(res_man, 7, 7);
    // fix Sticky bit
    res_s = res_s | (sbs(res_man, 6, 0) != 0);
    // align mantissa to the right (dispose of 7 extra fraction bits from mult + 1 G bit + 1 R bit)
    res_man = res_man >> 9;

    // round
    uint8_t need_rnd;

    // detect underflow
    int16_t shift_val;
    if ((res_exp_unbiased < -127) || (res_exp_unbiased == -127 && res_man != 0)) {
        // denormalize small values
        shift_val        = -126 - res_exp_unbiased;
        res_exp_unbiased = -127;

        if (shift_val > 30) // 19
            shift_val = 30;
        if (shift_val == 1) {
            res_s = res_s | res_r;
            res_r = res_g;
            res_g = sbs(res_man, 0, 0);
        } else if (shift_val == 2) {
            // fix Sticky bit
            res_s = res_s | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        } else // shift_val>=3
        {
            // fix Sticky bit
            res_s = res_s | (sbs(res_man, shift_val - 3, 0) != 0) | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        }

        res_man  = res_man >> shift_val;
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        res_man = res_man + need_rnd;

        if (sbs(res_man, 7, 7) == 1)
            res_exp_unbiased = res_exp_unbiased + 1;
    } else {
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        res_man = res_man + need_rnd;
        if (sbs(res_man, 8, 8) == 1) {
            res_s            = res_s | res_r;
            res_r            = res_g;
            res_g            = sbs(res_man, 0, 0);
            res_man          = res_man >> 1;
            res_exp_unbiased = res_exp_unbiased + 1;
        }

        // detect overflow
        if (res_exp_unbiased >= 128) {
            // result is +inf or -inf
            if ((round_mode == RND_TO_0) || (round_mode == RND_TO_PINF && res_sgn == 1) ||
                (round_mode == RND_TO_NINF && res_sgn == 0)) {
                // largest value which is not inf
                res_exp_unbiased = 127;
                res_man          = 0x7F;
            } else {
                // inf
                res_exp_unbiased = 128;
                res_man          = 0;
            }
        }
    }
    //  else

    if (result_ready == 0) {
        // construct result
        result = ibs(result, 15, 15, res_sgn);
        result = ibs(result, 14, 7, (res_exp_unbiased + 127));
        result = ibs(result, 6, 0, res_man);

        if (dnorm_ftz) {
            // flush result_16bit if denormal
            if (is_denorm_bfp16(result)) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(result, 15, 15) == 0))
                    result = 0x0080; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(result, 15, 15) == 1))
                    result = 0x8080; // res=-min_normal
                else
                    result = ibs(result, 14, 0, 0); // res=+-0
            }
        }
    }

    // uint8_t special_non_equal_case = 0;

    // // Round to Nearest Even
    // //  C is much smaller than A*B
    // //  @rounding when having exactly half as fraction
    // //  our BF16 and x86-FP32+convert might differ in their rounding up / down
    // if (round_mode == RND_TO_NE && sbs(result_32bit, 15, 0) == 0x8000 && sbs(result_16bit, 0, 0) == 0 &&
    // sbs(result, 0, 0) == 1)
    // special_non_equal_case = 1;
    // // Round to Nearest Even
    // //  C = min_normal
    // //  A*B is a very small denormal
    // //  A*B mantissa is subtracted from C
    // //  BF16 FMA will give +min_normal/-min_normal (because A*B is shifted outside the adder, leaving only sticky_bit
    // =
    // //  1, and after rounding it is added towards min_normal) FP32+convert will give +0/-0 (because A*B is not small
    // //  enough, it is not fully shifted outside the adder, and after rounding it is rounded to denormal and flushed
    // to
    // //  0)
    // //   if (round_mode == RND_TO_NE && sbs(result_32bit, 30, 0) == 0x00000000 && sbs(c, 14, 0) == 0x0080 &&
    // sbs(result,
    // //   14, 0) == 0x0080)
    // //    special_non_equal_case = 1;

    // // Round to Nearest Even
    // //  C = very small normal
    // //  A*B is a very small normal
    // //  A*B mantissa is subtracted from C
    // //  C-A*B=almost largest denormal
    // //  BF16 FMA will give +min_normal/-min_normal (rounding of 1.111111111100000 up)
    // //  FP32+convert will give +0/-0 (because 1.111111111100000 will be flushed to zero, and only then converted to
    // //  BF16)
    // //   if (round_mode == RND_TO_NE && sbs(result_32bit, 30, 0) == 0x00000000 && sbs(result, 14, 0) == 0x0080)
    // //    special_non_equal_case = 1;

    // if (((result != result_16bit) && special_non_equal_case == 0))
    // assert(0 && "Warning: BF16 mismatch (x86+convert vs. our reference code");

    // if (((result != result_16bit) && special_non_equal_case == 1))
    return result;

    // return result_16bit;
}

#if 0
//BF16 data type - accumulation into FP32
uint32_t my_fma_bfp16_fp32_old (uint16_t a, uint16_t b, uint32_t c, uint8_t round_mode)
{

  uint8_t a_is_dnorm = is_denorm_bfp16(a);
  uint8_t b_is_dnorm = is_denorm_bfp16(b);
  uint8_t c_is_dnorm = is_denorm_fp32(c);

#if DNORM_FTZ
  //flush denormals to zero
  //-----------------------

  //flush denormal inputs to zero
  if(is_denorm_bfp16(a))
    a = ibs(a,14,0,0); //a=+-0

  if(is_denorm_bfp16(b))
    b = ibs(b,14,0,0); //b=+-0

  if(is_denorm_fp32(c))
    c = ibs(c,30,0,0); //c=+-0
  //------------------------
#endif

  uint16_t a_exp = sbs(a,14,7);
  uint16_t b_exp = sbs(b,14,7);
  uint16_t c_exp = sbs(c,30,23);
  uint32_t a_man = sbs(a,6,0);
  uint32_t b_man = sbs(b,6,0);
  uint32_t c_man = sbs(c,22,0);
  uint16_t a_sgn = sbs(a,15,15);
  uint16_t b_sgn = sbs(b,15,15);
  uint16_t c_sgn = sbs(c,31,31);

  int16_t a_exp_unbiased;
  int16_t b_exp_unbiased;
  int16_t c_exp_unbiased;

  uint16_t ab_sgn = a_sgn^b_sgn;
  uint8_t sub_ind = ab_sgn^c_sgn;
  uint8_t a_is_zero = is_zero_bfp16(a);
  uint8_t b_is_zero = is_zero_bfp16(b);
  uint8_t c_is_zero = is_zero_fp32(c);
  uint8_t leading_1_ind;
      
  if(!a_is_dnorm && !a_is_zero)
    {
      if(!a_is_zero)
        a_man = cbs(1,a_man,7);
      a_exp_unbiased = a_exp - 127;
    }
  else
    {
      a_exp_unbiased = -126;
      //normalize
      leading_1_ind = lzd(a_man);
      if(leading_1_ind<=6)
        {
          a_man = a_man<<(7-leading_1_ind);
          a_exp_unbiased = a_exp_unbiased - (7-leading_1_ind);
        }
    }
    
  if(!b_is_dnorm && !b_is_zero)
    {
      if(!b_is_zero)
        b_man = cbs(1,b_man,7);
      b_exp_unbiased = b_exp - 127;
    }
  else
    {
      b_exp_unbiased = -126;
      //normalize
      leading_1_ind = lzd(b_man);
      if(leading_1_ind<=6)
        {
          b_man = b_man<<(7-leading_1_ind);
          b_exp_unbiased = b_exp_unbiased - (7-leading_1_ind);
        }
    }
  
  if(!c_is_dnorm && !c_is_zero)
    {
      if(!c_is_zero)
        c_man = cbs(1,c_man,23);
      c_exp_unbiased = c_exp - 127;
    }
  else
    {
      c_exp_unbiased = -126;
      //normalize
      leading_1_ind = lzd(c_man);
      if(leading_1_ind<=22)
        {
          c_man = c_man<<(23-leading_1_ind);
          c_exp_unbiased = c_exp_unbiased - (23-leading_1_ind);
        }
    }
  uint32_t ab_man = a_man*b_man;
  int16_t ab_exp_unbiased = a_exp_unbiased + b_exp_unbiased;       
  uint8_t res_sgn;

  //  uint8_t a_exp_is_zero = (sbs(a,14,7)==0);
  //  uint8_t b_exp_is_zero = (sbs(b,14,7)==0);
  //uint8_t c_exp_is_zero = (sbs(c,30,23)==0);
  uint8_t a_is_inf = is_inf_bfp16(a);
  uint8_t b_is_inf = is_inf_bfp16(b);
  uint8_t c_is_inf = is_inf_fp32(c);
  //  uint8_t ab_res_is_inf = is_inf_bfp16(cbs((ab_exp_unbiased+127),ab_man,7));
  uint8_t mul_res_is_zero = (a_is_zero | b_is_zero);
  uint8_t a_is_nan = is_nan_bfp16(a);
  uint8_t b_is_nan = is_nan_bfp16(b);
  uint8_t c_is_nan = is_nan_fp32(c);

  //exceptions
  uint8_t mul_res_is_inf = (!a_is_zero & b_is_inf) | (!b_is_zero & a_is_inf); 

  uint8_t mul_res_is_def_nan = (a_is_zero & b_is_inf) | (b_is_zero & a_is_inf);

  uint8_t mul_res_is_nan = mul_res_is_def_nan | a_is_nan | b_is_nan ;

  // Both sources are inf with (different signs and add instruction) or (same signs and sub instruction). i.e. inf-inf
  uint8_t res_is_def_nan = (mul_res_is_inf & c_is_inf & sub_ind) ;

  // nan handling - if any input is nan or result is nan - return nan
  uint8_t res_is_nan = res_is_def_nan | c_is_nan | mul_res_is_nan;

  // add res is inf - if one of the sources is infinity and it's add instruction OR one of the sources is infinity - select the exception result
  uint8_t res_is_inf = !res_is_nan &
    ((mul_res_is_inf & c_is_inf & !sub_ind) |
     (mul_res_is_inf^c_is_inf));

  //special results - bypass calculated result
  if (res_is_inf)
    {
      if (mul_res_is_inf & !c_is_inf)
        res_sgn = ab_sgn;
      else //if (!mul_res_is_inf & c_is_inf)
        res_sgn = c_sgn;

      return cbs(res_sgn,0x7F800000,31); //+-inf
    }
  if (res_is_nan)
    {
//      if (b_is_nan)
//        return (cbs(b,0x0000,16) | QNAN_BIT_FP32);
//      if (a_is_nan)
//        return (cbs(a,0x0000,16) | QNAN_BIT_FP32);
//      if (c_is_nan)
//        return (c | QNAN_BIT_FP32);
      return DEFAULT_NAN_FP32; //indefinite nan value
    }
    
    uint8_t sign_zero_exception;
    if(!c_is_zero & mul_res_is_zero)
        sign_zero_exception = c_sgn;
    else if(c_is_zero & !mul_res_is_zero)
        sign_zero_exception = ab_sgn;
    else 
        sign_zero_exception = (round_mode==RND_TO_NINF) && (c_sgn^ab_sgn);

   sign_zero_exception = sign_zero_exception || (c_sgn && ab_sgn);

   uint8_t c_sgn_zero = (c_is_zero && (a_is_zero || b_is_zero)) ? sign_zero_exception : c_sgn;

   c = c_is_zero ? cbs(c_sgn_zero,0x00000000, 31) : c;
      
  if(a_is_zero | b_is_zero)
    return c;

  //align c_man to ab_man (regardless of exponents)
  //compensate for 9 additional fraction bits of mantissa (after multiplication we have 14bits fraction, but c has 23bits fraction)
  ab_man = ab_man<<9;
  //add 2 lsbs for G and R bits, for both operands (Guard and Round - for rounding at the end)
  ab_man = ab_man<<2;
  c_man = c_man<<2;  

  //Sticky bits
  uint16_t c_s = 0;
  uint16_t ab_s = 0;

  int16_t res_exp_unbiased;

  int16_t exp_diff = c_exp_unbiased - ab_exp_unbiased;
  
  if (exp_diff >= 0)
    {
      c_s = 0;
      if(exp_diff>27) //larger exp_diff means that we are discarding the entire ab_man
        exp_diff = 27;
      if (exp_diff > 2)
        ab_s = (sbs(ab_man,exp_diff-3+2,0+2)!=0);
      else
        ab_s = 0;
      ab_man = ab_man>>exp_diff;
      res_exp_unbiased = c_exp_unbiased;
    }
  else if (exp_diff < 0)
    {
      ab_s = 0;
      if(exp_diff<-27) //smaller exp_diff means that we are discarding the entire c_man
        exp_diff = -27;
      if (-exp_diff > 2)
        c_s = (sbs(c_man,-exp_diff-3+2,0+2)!=0);
      else
        c_s = 0;
      c_man = c_man>>(-exp_diff); 
      res_exp_unbiased = ab_exp_unbiased;
    }
  uint16_t res_s = ab_s | c_s;
  
  uint32_t res_man;  
  int32_t man_diff;

  if(sub_ind)
    {
      man_diff = cbs(ab_man,ab_s,1) - cbs(c_man,c_s,1);  
      if (man_diff < 0)
        {
          res_man = (-man_diff)>>1;
          res_sgn = c_sgn;
        }
      else if (man_diff > 0)
        {
          res_man =  man_diff>>1;
          res_sgn = ab_sgn;
        }
      else
        {
          res_man = 0;
          //res_sgn = 0;
            
          if(ab_s && !c_s)
            res_sgn = ab_sgn;
          else if(!ab_s && c_s)
            res_sgn = c_sgn;
          else
            res_sgn = (round_mode==RND_TO_NINF); //x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        }
    }
  else
    {
      res_man = ab_man + c_man;
      res_sgn = c_sgn;
    }
  
  //find leading 1
  leading_1_ind = lzd(res_man);
    
  //normalize
  if(leading_1_ind==27)
    {
      res_s = res_s | (sbs(res_man,1,0)!=0);
      res_man = res_man >> 2;
      res_exp_unbiased = res_exp_unbiased + 2;
    }
  else if(leading_1_ind==26)
    {
      res_s = res_s | sbs(res_man,0,0);
      res_man = res_man >> 1;
      res_exp_unbiased = res_exp_unbiased + 1;
    }
  else if(leading_1_ind<=25)
    {      
      //res_s = res_s;
      res_man = res_man << (25-leading_1_ind);
      res_exp_unbiased = res_exp_unbiased - (25-leading_1_ind);
    }
  else
    {
      //res_man is 0
      if(res_s==0) //result is 0, and grs are 0, no need for rounding
        res_exp_unbiased = -127;
      else
        res_exp_unbiased = res_exp_unbiased - 28;
    }
    
  //separate GR bits from mantissa
  uint8_t res_g = sbs(res_man,1,1);
  uint8_t res_r = sbs(res_man,0,0);

  //align mantissa to the right (dispose of 7 extra fraction bits from mult + 1 G bit + 1 R bit)
  res_man = res_man>>2;  

  //round
  uint8_t need_rnd;
  
    //detect underflow
  int16_t shift_val;

  if((res_exp_unbiased<-127) || (res_exp_unbiased==-127 && res_man!=0))
    {
      //denormalize small values
      shift_val = -126 - res_exp_unbiased;
      res_exp_unbiased = -127;

      if(shift_val>30) //24
        shift_val=30;
      if(shift_val==1)
        {
          res_s = res_s | res_r;
          res_r = res_g;
          res_g = sbs(res_man,0,0);
        }
      else if(shift_val==2)
        {
          //fix Sticky bit
          res_s = res_s | res_g | res_r;
          
          res_g = sbs(res_man,shift_val-1,shift_val-1);
          res_r = sbs(res_man,shift_val-2,shift_val-2);
        }
      else //shift_val>=3
        {
          //fix Sticky bit
          res_s = res_s | (sbs(res_man,shift_val-3,0)!=0) | res_g | res_r;
          
          res_g = sbs(res_man,shift_val-1,shift_val-1);
          res_r = sbs(res_man,shift_val-2,shift_val-2);
        }
  
      res_man = res_man>>shift_val;
      need_rnd = (((round_mode==RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn==0)) | 
                  ((round_mode==RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn==1)) | 
                  ((round_mode==RND_TO_NE) & 
                   ((res_g & (res_r | res_s)) | 
                    (res_g & !res_r & !res_s & (sbs(res_man,0,0)==1)))) | 
                   ((round_mode==RND_HALF_AZ) & res_g));
    
      res_man = res_man + need_rnd;
        
      if (sbs(res_man,23,23)==1)
        res_exp_unbiased = res_exp_unbiased + 1;
      
    }
  else
  {
  need_rnd = (((round_mode==RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn==0)) | 
                    ((round_mode==RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn==1)) | 
                    ((round_mode==RND_TO_NE) & 
                     ((res_g & (res_r | res_s)) | 
                      (res_g & !res_r & !res_s & (sbs(res_man,0,0)==1)))) | 
                      ((round_mode==RND_HALF_AZ) & res_g));
  res_man = res_man + need_rnd;

  if (sbs(res_man,24,24)==1)
    {
      res_s = res_s | res_r;
      res_r = res_g;
      res_g = sbs(res_man,0,0);
      res_man = res_man>>1;
      res_exp_unbiased = res_exp_unbiased + 1;
    }
  
  //detect overflow
  if(res_exp_unbiased>=128)
    {
      //result is +inf or -inf
      if((round_mode==RND_TO_0) || 
         (round_mode==RND_TO_PINF && res_sgn==1) || 
         (round_mode==RND_TO_NINF && res_sgn==0))
        {
          //largest value which is not inf
          res_exp_unbiased = 127;
          res_man = 0x7FFFFF;   
        }
      else
        {
          //inf
          res_exp_unbiased = 128;
          res_man = 0;   
        }
    }
  }

  //construct result
  uint32_t result = 0;
  result = ibs(result,31,31,res_sgn);
  result = ibs(result,30,23,(res_exp_unbiased+127));
  result = ibs(result,22,0,res_man);

#if DNORM_FTZ         
  //flush result_16bit if denormal
  if(is_denorm_fp32(result))
    {
      //flush result to 0    
      if((round_mode==RND_TO_PINF) && (sbs(result,31,31)==0)) 
        result = 0x00800000; //res=+min_normal
      else if ((round_mode==RND_TO_NINF) && (sbs(result,31,31)==1))
        result = 0x80800000; //res=-min_normal
      else
        result = ibs(result,30,0,0); //res=+-0
    }
#endif
  return result;
}
#endif

uint16_t fp32_to_bf16(float input, int roundingMode, uint32_t sr_register, bool clip_fp, bool dnorm_ftz_out, bool clip_fp_inf_input)
{
    const uint32_t inputUint = *(const uint32_t*)&input;

    uint16_t res = 0;

    if (is_nan_fp32(inputUint)) {
        res = DEFAULT_NAN_BFP16;
    } else if (is_zero_fp32(inputUint) || is_inf_fp32(inputUint)) {
        // zero and inf are checked separately to prevent stochastic round up
        res = sbs(inputUint, 31, 16);
    } else {

        uint16_t out_sgn = sbs(inputUint, 31, 31);
        uint16_t out_exp = sbs(inputUint, 30, 23);
        uint16_t out_man = sbs(inputUint, 22, 16);
        bool     out_g   = (sbs(inputUint, 15, 15) == 1) ? 1 : 0;
        bool     out_rs  = (sbs(inputUint, 14, 0) != 0) ? 1 : 0;
        uint32_t out_grs = sbs(inputUint, 15, 0) << 16; // aligned left for comparing with LFSR[31:0]

        if (!is_denorm_fp32(inputUint) && !is_zero_fp32(inputUint))
            out_man = cbs(1, out_man, 7);

        bool need_rnd =
            (((roundingMode == RND_TO_PINF) && (out_rs || out_g) && (out_sgn == 0)) ||
             ((roundingMode == RND_TO_NINF) && (out_rs || out_g) && (out_sgn == 1)) ||
             ((roundingMode == RND_TO_NE) && ((out_g && out_rs) || (out_g && !out_rs && (sbs(out_man, 0, 0) == 1)))) ||
             ((roundingMode == RND_HALF_AZ) && out_g) || ((roundingMode == RND_SR) && (out_grs >= sr_register)));

        if (need_rnd) {
            out_man = out_man + 1;

            if (out_exp == 0) // denormal
            {
                if (sbs(out_man, 7, 7) == 1)
                    out_exp = out_exp + 1;
            } else // normal
            {
                if (sbs(out_man, 8, 8) == 1)
                    out_exp = out_exp + 1;
            }
        }
        // construct result
        res = ibs(res, 15, 15, out_sgn);
        res = ibs(res, 14, 7, out_exp);
        res = ibs(res, 6, 0, out_man);
    }

    if (clip_fp && is_inf_bfp16(res) && (clip_fp_inf_input || !is_inf_fp32(inputUint))) {
        // uint16_t out_sgn = sbs(inputUint, 31, 31);
        // uint16_t out_exp = 0xfe;
        // uint16_t out_man = 0x7f;

        // res = 0;

        // res = ibs(res, 15, 15, out_sgn);
        // res = ibs(res, 14, 7, out_exp);
        // res = ibs(res, 6, 0, out_man);

        res = res - 1; // will return +/-max_norm_value
    }

    if (dnorm_ftz_out && is_denorm_bfp16(res)) {
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(res, 15, 15) == 0))
            res = 0x0080; // res=+min_normal
        else if ((roundingMode == RND_TO_NINF) && (sbs(res, 15, 15) == 1))
            res = 0x8080; // res=-min_normal
        else
            res = ibs(res, 14, 0, 0); // res=+-0
    }

    return res;
}

uint32_t fp32_to_fp19(float input, int roundingMode, uint32_t sr_register, bool clip_fp, bool clip_fp_inf_input)
{
    const uint32_t inputUint = *(const uint32_t*)&input;

    uint32_t res = 0;

    if (is_nan_fp32(inputUint)) {
        res = DEFAULT_NAN_FP19;
    } else if (is_zero_fp32(inputUint) || is_inf_fp32(inputUint)) {
        // zero and inf are checked separately to prevent stochastic round up
        res = sbs(inputUint, 31, 13);
    } else {

        uint16_t out_sgn = sbs(inputUint, 31, 31);
        uint16_t out_exp = sbs(inputUint, 30, 23);
        uint16_t out_man = sbs(inputUint, 22, 13);
        bool     out_g   = (sbs(inputUint, 12, 12) == 1) ? 1 : 0;
        bool     out_rs  = (sbs(inputUint, 11, 0) != 0) ? 1 : 0;
        uint32_t out_grs = sbs(inputUint, 12, 0) << 19; // aligned left for comparing with LFSR[31:0]

        if (!is_denorm_fp32(inputUint) && !is_zero_fp32(inputUint))
            out_man = cbs(1, out_man, 10);

        bool need_rnd =
            (((roundingMode == RND_TO_PINF) && (out_rs || out_g) && (out_sgn == 0)) ||
             ((roundingMode == RND_TO_NINF) && (out_rs || out_g) && (out_sgn == 1)) ||
             ((roundingMode == RND_TO_NE) && ((out_g && out_rs) || (out_g && !out_rs && (sbs(out_man, 0, 0) == 1)))) ||
             ((roundingMode == RND_HALF_AZ) && out_g) || ((roundingMode == RND_SR) && (out_grs >= sr_register)));

        if (need_rnd) {
            out_man = out_man + 1;

            if (out_exp == 0) // denormal
            {
                if (sbs(out_man, 10, 10) == 1)
                    out_exp = out_exp + 1;
            } else // normal
            {
                if (sbs(out_man, 11, 11) == 1)
                    out_exp = out_exp + 1;
            }
        }
        // construct result
        res = ibs(res, 18, 18, out_sgn);
        res = ibs(res, 17, 10, out_exp);
        res = ibs(res, 9, 0, out_man);
    }

    if (clip_fp && is_inf_fp32(res << 13) && (clip_fp_inf_input || !is_inf_fp32(inputUint))) {
        // uint16_t out_sgn = sbs(inputUint, 31, 31);
        // uint16_t out_exp = 0xfe; //max_exp-1
        // uint16_t out_man = 0x3ff; //all ones

        // res = 0;

        // res = ibs(res, 18, 18, out_sgn);
        // res = ibs(res, 17, 10, out_exp);
        // res = ibs(res, 9, 0, out_man);
        res = res - 1; // will return +/-max_norm_value
    }

    return res;
}

//conversion from fp32 to format 1-8-x, where x is man_width. e.g. fp19 (1-8-10) or fp24 (1-8-15)
uint32_t fp32_to_fp18x(float input, int roundingMode, uint32_t sr_register, bool clip_fp, bool clip_fp_inf_input, uint8_t man_width)
{
    const uint32_t inputUint = *(const uint32_t*)&input;

    uint32_t res = 0;

    uint32_t discarded_bits_num = EXPONENT_OFFSET_FP32 - man_width;
    if (is_nan_fp32(inputUint)) {
        res = DEFAULT_NAN_FP32 >> discarded_bits_num;
    } else if (is_zero_fp32(inputUint) || is_inf_fp32(inputUint)) {
        // zero and inf are checked separately to prevent stochastic round up
        res = sbs(inputUint, 31, discarded_bits_num);
    } else {

        uint16_t out_sgn = sbs(inputUint, 31, 31);
        uint16_t out_exp = sbs(inputUint, 30, 23);
        uint32_t out_man = sbs(inputUint, 22, discarded_bits_num);
        bool     out_g   = (sbs(inputUint, discarded_bits_num-1, discarded_bits_num-1) == 1) ? 1 : 0;
        bool     out_rs  = (sbs(inputUint, discarded_bits_num-2, 0) != 0) ? 1 : 0;
        uint32_t out_grs = sbs(inputUint, discarded_bits_num-1, 0) << (man_width+1+8); // aligned left for comparing with LFSR[31:0]
        
        if (!is_denorm_fp32(inputUint) && !is_zero_fp32(inputUint))
            out_man = cbs(1, out_man, man_width);

        bool need_rnd =
            (((roundingMode == RND_TO_PINF) && (out_rs || out_g) && (out_sgn == 0)) ||
             ((roundingMode == RND_TO_NINF) && (out_rs || out_g) && (out_sgn == 1)) ||
             ((roundingMode == RND_TO_NE) && ((out_g && out_rs) || (out_g && !out_rs && (sbs(out_man, 0, 0) == 1)))) ||
             ((roundingMode == RND_HALF_AZ) && out_g) || ((roundingMode == RND_SR) && (out_grs >= sr_register)));

        if (need_rnd) {
            out_man = out_man + 1;

            if (out_exp == 0) // denormal
            {
                if (sbs(out_man, man_width, man_width) == 1)
                    out_exp = out_exp + 1;
            } else // normal
            {
                if (sbs(out_man, man_width+1, man_width+1) == 1)
                    out_exp = out_exp + 1;
            }
        }
        // construct result
        res = ibs(res, man_width+8, man_width+8, out_sgn);
        res = ibs(res, man_width+8-1, man_width, out_exp);
        res = ibs(res, man_width-1, 0, out_man);
    }

    if (clip_fp && is_inf_fp32(res << discarded_bits_num) && (clip_fp_inf_input || !is_inf_fp32(inputUint))) {
        // uint16_t out_sgn = sbs(inputUint, 31, 31);
        // uint16_t out_exp = 0xfe; //max_exp-1
        // uint16_t out_man = 0x3ff; //all ones

        // res = 0;

        // res = ibs(res, 18, 18, out_sgn);
        // res = ibs(res, 17, 10, out_exp);
        // res = ibs(res, 9, 0, out_man);
        res = res - 1; // will return +/-max_norm_value
    }

    return res;
}


uint32_t fp32_to_tf32(float input, int roundingMode, uint32_t sr_register, bool clip_fp, bool clip_fp_inf_input)
{
    uint32_t output = fp32_to_fp19(input, roundingMode, sr_register, clip_fp, clip_fp_inf_input);

    return (output << 13);
}

uint32_t fp32_to_fp32(float input, int roundingMode, bool clip_fp, bool dnorm_ftz_out, bool clip_fp_inf_input)
{
    const uint32_t inputUint = *(const uint32_t*)&input;

    uint32_t res = inputUint;

    if (clip_fp && is_inf_fp32(res)) {
        res = res - 1; // will return +/-max_norm_value
    }

    if (dnorm_ftz_out) {
        // flush denormal output to zero
        if (is_denorm_fp32(res)) {
            // flush result to 0
            if ((roundingMode == RND_TO_PINF) && (sbs(res, 31, 31) == 0))
                res = 0x00800000; // res=+min_normal
            else if ((roundingMode == RND_TO_NINF) && (sbs(res, 31, 31) == 1))
                res = 0x80800000; // res=-min_normal
            else
                res = ibs(res, 30, 0, 0); // res=+-0
        }
    }

    return res;
}

static inline float bf16_to_float(uint16_t a)
{
    uint32_t val_32b;
    float    val_fp32;
    val_32b  = ((uint32_t)a) << 16;
    memcpy(&val_fp32, &val_32b, sizeof(float));
    return val_fp32;
}

// initialize round_mode
void set_rounding_mode(uint8_t round_mode)
{
    int current_round_mode = fegetround();
    int next_round_mode;

    switch (round_mode) {
        case RND_TO_0: next_round_mode = FE_TOWARDZERO; break;
        case RND_TO_PINF: next_round_mode = FE_UPWARD; break;
        case RND_TO_NINF: next_round_mode = FE_DOWNWARD; break;
        case RND_TO_NE: next_round_mode = FE_TONEAREST; break;
        default: next_round_mode = FE_TONEAREST; break;
    }

    if (current_round_mode != next_round_mode) {
        fesetround(next_round_mode);
    }
}
/////////FP32 - pure FP32 FMA + FP32 adders
/////////////////////////////////////////////////////
// fp32 - no flush of result
uint32_t fma_fp32_no_flush(uint32_t a, uint32_t b, uint32_t c, uint8_t round_mode)
{
    uint32_t* res;

    float* a_f;
    float* b_f;
    float* c_f;
    float  res_f;

    set_rounding_mode(round_mode);

    //  uint8_t a_is_dnorm = is_denorm_fp32(a);
    //  uint8_t b_is_dnorm = is_denorm_fp32(b);
    //  uint8_t c_is_dnorm = is_denorm_fp32(c);

    if ((round_mode == RND_TO_NE) || (round_mode == RND_TO_0)) {

        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        a_f = (float*)(&a);
        b_f = (float*)(&b);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    } else {

        //#if DNORM_FTZ
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (is_denorm_fp32(a))
            a = ibs(a, 30, 0, 0); // a=+-0

        if (is_denorm_fp32(b))
            b = ibs(b, 30, 0, 0); // b=+-0

        if (is_denorm_fp32(c))
            c = ibs(c, 30, 0, 0); // c=+-0
            //------------------------
        //#endif

        a_f = (float*)(&a);
        b_f = (float*)(&b);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);
    }

    if (is_nan_fp32(*res))
        *res = DEFAULT_NAN_FP32; // indefinite nan value

    return *res;
}

// fp32 = fp32 + fp32*fp32
uint32_t fma_fp32(uint32_t a, uint32_t b, uint32_t c, uint8_t round_mode, bool dnorm_ftz)
{
    uint32_t* res;

    float* a_f;
    float* b_f;
    float* c_f;
    float  res_f;

    set_rounding_mode(round_mode);

    //  uint8_t a_is_dnorm = is_denorm_fp32(a);
    //  uint8_t b_is_dnorm = is_denorm_fp32(b);
    //  uint8_t c_is_dnorm = is_denorm_fp32(c);

    if ((round_mode == RND_TO_NE) || (round_mode == RND_TO_0)) {

        if (dnorm_ftz) {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        }
        a_f = (float*)(&a);
        b_f = (float*)(&b);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        if (dnorm_ftz) {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        }
    } else {
        if (dnorm_ftz) {
            // flush denormals to zero
            //-----------------------

            // flush denormal inputs to zero
            if (is_denorm_fp32(a))
                a = ibs(a, 30, 0, 0); // a=+-0

            if (is_denorm_fp32(b))
                b = ibs(b, 30, 0, 0); // b=+-0

            if (is_denorm_fp32(c))
                c = ibs(c, 30, 0, 0); // c=+-0
            //------------------------
        }

        a_f = (float*)(&a);
        b_f = (float*)(&b);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        if (dnorm_ftz) {
            // flush denormal output to zero
            if (is_denorm_fp32(*res)) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(*res, 31, 31) == 0))
                    *res = 0x00800000; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(*res, 31, 31) == 1))
                    *res = 0x80800000; // res=-min_normal
                else
                    *res = ibs(*res, 30, 0, 0); // res=+-0
            }
        }
    }

    if (is_nan_fp32(*res))
        *res = DEFAULT_NAN_FP32; // indefinite nan value

    return *res;
}

// BF16 data type - accumulation into FP32
uint32_t fma_bfp16_fp32(uint16_t a, uint16_t b, uint32_t c, uint8_t round_mode, bool dnorm_ftz)
{
    uint32_t a_32bit = a << 16;
    uint32_t b_32bit = b << 16;
    uint32_t result_fp32_x86;
    result_fp32_x86 = fma_fp32(a_32bit, b_32bit, c, round_mode, dnorm_ftz);

    return result_fp32_x86;
}

uint32_t add_fp32(uint32_t a, uint32_t b, uint8_t round_mode, bool dnorm_ftz)
{
    uint32_t* res;

    float* a_f;
    float* b_f;
    float  res_f;

    set_rounding_mode(round_mode);

    //  uint8_t a_is_dnorm = is_denorm_fp32(a);
    //  uint8_t b_is_dnorm = is_denorm_fp32(b);

    if (dnorm_ftz) {
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (is_denorm_fp32(a))
            a = ibs(a, 30, 0, 0); // a=+-0

        if (is_denorm_fp32(b))
            b = ibs(b, 30, 0, 0); // b=+-0
        //------------------------
    }

    a_f = (float*)(&a);
    b_f = (float*)(&b);

    res_f = (*a_f) + (*b_f);

    res = (uint32_t*)(&res_f);

    if (dnorm_ftz) {
        // flush denormal output to zero
        if (is_denorm_fp32(*res)) {
            // flush result to 0
            if ((round_mode == RND_TO_PINF) && (sbs(*res, 31, 31) == 0))
                *res = 0x00800000; // res=+min_normal
            else if ((round_mode == RND_TO_NINF) && (sbs(*res, 31, 31) == 1))
                *res = 0x80800000; // res=-min_normal
            else
                *res = ibs(*res, 30, 0, 0); // res=+-0
        }
    }

    if (is_nan_fp32(*res))
        *res = DEFAULT_NAN_FP32; // indefinite nan value

    return *res;
}

uint32_t add_fp32_4args(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint8_t round_mode)
{
    uint32_t res_ab = add_fp32(a, b, round_mode);
    uint32_t res_cd = add_fp32(c, d, round_mode);
    uint32_t res    = add_fp32(res_ab, res_cd, round_mode);
    return res;
}

/////////until here - FP32
/////////////////////////////////////////////////////

uint16_t add_bf16(uint16_t a, uint16_t b, uint8_t round_mode)
{
    uint32_t a_32bit = cbs(a, 0, 16);
    uint32_t b_32bit = cbs(b, 0, 16);
    uint32_t res_32bit;
    uint16_t res_16bit;
    float*   a_f;

    res_32bit = add_fp32(a_32bit, b_32bit, round_mode);
    a_f       = (float*)&res_32bit;
    res_16bit = fp32_to_bf16(*a_f, round_mode, 0, 0, 0, 1);

    return res_16bit;
}

/////////FP16 - Conversion is from Sergei's code
/////////////////////////////////////////////////////

#define SIGN_OFFSET_FP32 31
#define SIGN_MASK_FP32 0x80000000
#define EXPONENT_OFFSET_FP32 23
#define EXPONENT_MASK_FP32 0x7F800000
#define EXPONENT_BIAS_FP32 127
#define SIGNIFICAND_MASK_FP32 0x007FFFFF

#define SIGN_OFFSET_FP16 15
#define SIGN_MASK_FP16 0x8000
#define EXPONENT_OFFSET_FP16 10
#define EXPONENT_MASK_FP16 0x7C00
#define EXPONENT_BIAS_FP16 15
#define SIGNIFICAND_MASK_FP16 0x03FF

bool fp16_is_zero(uint16_t val)
{
    return (val & (~SIGN_MASK_FP16)) ? 0 : 1;
}

bool fp16_is_infinity(uint16_t val)
{
    return (val & 0x7FFF) == EXPONENT_MASK_FP16 ? 1 : 0;
}

bool fp16_is_nan(uint16_t val)
{
    bool isAllExponentBitsSet = ((val & EXPONENT_MASK_FP16) == EXPONENT_MASK_FP16);
    bool isAnyMantissaBitSet  = ((val & SIGNIFICAND_MASK_FP16) != 0);
    return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

bool fp16_is_negative(uint16_t val)
{
    return ((val & SIGN_MASK_FP16) == SIGN_MASK_FP16);
}

bool fp16_is_denormal(uint16_t val)
{ // Do not consider zero as denormal
    return (((val & EXPONENT_MASK_FP16) == 0) && ((val & SIGNIFICAND_MASK_FP16) != 0));
}

int lzcnt(uint32_t bits, uint32_t int_num)
{
    int i;
    int msb = bits - 1;
    int lsb = 0;
    for (i = msb; i >= lsb; i--) {
        if ((int_num & (1 << i)) != 0) {
            break;
        }
    }
    return bits - i - 1;
}

uint32_t fma_fp16_fp32(uint16_t a, uint16_t b, uint32_t c, uint8_t round_mode, bool fp16_ftz_in, bool fp16_ftz_out, bool fp32_dnorm_ftz)
{
    uint32_t* res;

    if (fp16_ftz_in) {
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (is_denorm_fp16(a))
            a = ibs(a, 14, 0, 0); // a=+-0

        if (is_denorm_fp16(b))
            b = ibs(b, 14, 0, 0); // b=+-0
        //------------------------
    }

    uint32_t a_32bit = fp16_to_fp32(a, 0);
    uint32_t b_32bit = fp16_to_fp32(b, 0);

    float* a_f;
    float* b_f;
    float* c_f;
    float  res_f;

    set_rounding_mode(round_mode);

    if ((round_mode == RND_TO_NE) || (round_mode == RND_TO_0)) {

        if (fp32_dnorm_ftz)
        {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        }

        a_f = (float*)(&a_32bit);
        b_f = (float*)(&b_32bit);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        if (fp32_dnorm_ftz)
        {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        }
    } else {

        if (fp32_dnorm_ftz)
        {
            // flush denormals to zero
            //-----------------------

            if (is_denorm_fp32(c))
                c = ibs(c, 30, 0, 0); // c=+-0
            //------------------------
        }

        a_f = (float*)(&a_32bit);
        b_f = (float*)(&b_32bit);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        if (fp16_ftz_out == 1) {
            // flush denormal output to zero
            if (is_denorm_fp32(*res)) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(*res, 31, 31) == 0))
                    *res = 0x00800000; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(*res, 31, 31) == 1))
                    *res = 0x80800000; // res=-min_normal
                else
                    *res = ibs(*res, 30, 0, 0); // res=+-0
            }
        }
    }

    if (is_nan_fp32(*res))
        *res = DEFAULT_NAN_FP32; // indefinite nan value

    return *res;
}

static inline float fp16_to_float(uint16_t a)
{
    uint32_t val_32b;
    float    val_fp32;
    val_32b  = fp16_to_fp32(a, 0);
    val_fp32 = *(float*)&val_32b;
    return val_fp32;
}

int fp_accommodate_rounding(uint32_t     intValuePreRounding,
                            bool         roundedMSB,
                            bool         roundedLSBs,
                            unsigned int sign,
                            int          roundingMode,
                            uint32_t     lfsrVal,
                            uint32_t     discardedAlignedLeft)
{
    uint32_t result = 0;
    result          = intValuePreRounding;
    switch (roundingMode) {
        case RND_TO_0: result = intValuePreRounding; break;
        case RND_TO_PINF:
            if ((sign == 0) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_TO_NINF:
            if ((sign == 1) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_HALF_AZ:
            if (roundedMSB == 1) // half or above half will be rounded away from zero
            {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_SR:
            if (discardedAlignedLeft >= lfsrVal) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_TO_NE:
        default:
            if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
                (((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
    }
    return result;
}

uint16_t fp32_to_fp16(float input, int roundingMode, int32_t lfsrVal, bool clip_fp16, bool dnorm_ftz_out, bool clip_fp_inf_input)
{
    int      inputExponent, inputSign, unbiasedExp = 0;
    uint32_t inputMantissa;
    bool     roundedMSB = 0, roundedLSBs = 0;
    int      minExp     = -25;
    int      minNormExp = -14;
    int      maxExp     = 15;
    uint16_t output;

    uint32_t inputUint = *(uint32_t*)&input;

    inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
    inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
    inputSign     = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;
    if (is_nan_fp32(inputUint)) {
        // return the same NAN always (0x7FFF), as NVDA does
        outputSign     = 0x0;
        outputExponent = 0x1F;
        outputMantissa = 0x3FF;
    } else if (is_zero_fp32(inputUint)) {
        // return +-0
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (is_inf_fp32(inputUint)) {
        // return +-infinity
        outputExponent = 0x1F;
        outputMantissa = 0x0;
    } else {
        // Valid number
        unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
        inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

        if (unbiasedExp > maxExp) {

            if ((roundingMode == RND_TO_0) || (inputSign && (roundingMode == RND_TO_PINF)) ||
                (!inputSign && (roundingMode == RND_TO_NINF)))

            { // +- 65504.0 - that's what NVDA does
                outputMantissa = 0x3FF;
                outputExponent = maxExp + EXPONENT_BIAS_FP16;
            } else { // +-infinity
                outputExponent = 0x1F;
                outputMantissa = 0x0;
            }
        } else if (unbiasedExp < minExp) {
            // The result will be either 0 or 0x1
            roundedMSB     = 0;
            roundedLSBs    = 1;
            outputMantissa = fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, 0);
            outputExponent = 0x0;
        } else { // minExp <= unbiasedExp <= maxExp
            outputExponent = unbiasedExp;
            int rc_bit_idx =
                (unbiasedExp < minNormExp) ? -(unbiasedExp + 2) : (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16 - 1);
            roundedMSB                    = (inputMantissa >> rc_bit_idx) & 0x1;
            roundedLSBs                   = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
            uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
            outputMantissa                = inputMantissa >> (rc_bit_idx + 1);
            outputMantissa                = fp_accommodate_rounding(
                outputMantissa, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal, discardedAlignedLeft);
            if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << EXPONENT_OFFSET_FP16))) ||
                (outputMantissa & (1 << (EXPONENT_OFFSET_FP16 + 1)))) { // Should handle two cases:
                // 1. The number was denormal, and after rounding became normal
                // 2. The number was rounded to the 1.0 * 2^(next exponent)
                outputExponent = outputExponent + 1;
            }
            if (outputExponent > maxExp) {
                // return infinity
                outputExponent = 0x1F;
                outputMantissa = 0x0;
            } else {
                if (outputExponent < minNormExp) {
                    outputExponent = 0x0;
                } else {
                    outputExponent += EXPONENT_BIAS_FP16;
                }
                // normalize - leave 10 bits
                outputMantissa &= SIGNIFICAND_MASK_FP16;
            }
        }
    }
    output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) | (outputSign << SIGN_OFFSET_FP16);

    if (clip_fp16 && is_inf_fp16(output) && (clip_fp_inf_input || !is_inf_fp32(inputUint))) {
        // outputMantissa = 0x3FF;
        // outputExponent = 0x1E;
        // output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) | (inputSign << SIGN_OFFSET_FP16);
        output = output - 1; // will return +/-max_norm_value
    }

    if (dnorm_ftz_out && is_denorm_fp16(output)) {
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(output, 15, 15) == 0))
            output = 0x0400; // output=+min_normal
        else if ((roundingMode == RND_TO_NINF) && (sbs(output, 15, 15) == 1))
            output = 0x8400; // output=-min_normal
        else
            output = ibs(output, 14, 0, 0); // output=+-0
    }

    return output;
}

// FP16 data type - accumulation into FP16
uint16_t fma_fp16_fp16(uint16_t a, uint16_t b, uint16_t c, uint8_t round_mode, bool fp16_ftz_in, bool fp16_ftz_out)
{

    if (fp16_ftz_in) {
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (is_denorm_fp16(a))
            a = ibs(a, 14, 0, 0); // a=+-0

        if (is_denorm_fp16(b))
            b = ibs(b, 14, 0, 0); // b=+-0

        if (is_denorm_fp16(c))
            c = ibs(c, 14, 0, 0); // c=+-0
        //------------------------
    }

    // compare reuslt with FP32 FMA
    // uint32_t a_32bit = fp16_to_fp32(a, 0);
    // uint32_t b_32bit = fp16_to_fp32(b, 0);
    // uint32_t c_32bit = fp16_to_fp32(c, 0);
    // uint32_t result_32bit;
    // // result_32bit = fma_bfp16_fp32(a, b, c_32bit, round_mode);
    // // result of x86 FP32, with flush at input, no flush at output
    // result_32bit = fma_fp32_no_flush(a_32bit, b_32bit, c_32bit, round_mode);
    // uint16_t result_16bit;
    // result_16bit = fp32_to_fp16(*(float*)&result_32bit, (int)round_mode, 0, 0, 0, 1);

    // if (fp16_ftz_out) {
    // // flush result_16bit if denormal
    // if (is_denorm_fp16(result_16bit)) {
    // // flush result to 0
    // if ((round_mode == RND_TO_PINF) && (sbs(result_16bit, 15, 15) == 0))
    // result_16bit = 0x0400; // res=+min_normal
    // else if ((round_mode == RND_TO_NINF) && (sbs(result_16bit, 15, 15) == 1))
    // result_16bit = 0x8400; // res=-min_normal
    // else
    // result_16bit = ibs(result_16bit, 14, 0, 0); // res=+-0
    // }
    // }
    // our reference code
    uint16_t result       = 0;
    bool     result_ready = 0;

    uint8_t a_is_dnorm = is_denorm_fp16(a);
    uint8_t b_is_dnorm = is_denorm_fp16(b);
    uint8_t c_is_dnorm = is_denorm_fp16(c);

    uint16_t a_exp = sbs(a, 14, 10);
    uint16_t b_exp = sbs(b, 14, 10);
    uint16_t c_exp = sbs(c, 14, 10);
    uint32_t a_man = sbs(a, 9, 0);
    uint32_t b_man = sbs(b, 9, 0);
    uint32_t c_man = sbs(c, 9, 0);
    uint8_t  a_sgn = sbs(a, 15, 15);
    uint8_t  b_sgn = sbs(b, 15, 15);
    uint8_t  c_sgn = sbs(c, 15, 15);

    int16_t a_exp_unbiased;
    int16_t b_exp_unbiased;
    int16_t c_exp_unbiased;

    uint8_t ab_sgn    = a_sgn ^ b_sgn;
    uint8_t sub_ind   = ab_sgn ^ c_sgn;
    uint8_t a_is_zero = is_zero_fp16(a);
    uint8_t b_is_zero = is_zero_fp16(b);
    uint8_t c_is_zero = is_zero_fp16(c);
    uint8_t leading_1_ind;

    if (!a_is_dnorm && !a_is_zero) {
        if (!a_is_zero)
            a_man = cbs(1, a_man, 10);
        a_exp_unbiased = a_exp - 15;
    } else {
        a_exp_unbiased = -14;
        // normalize
        leading_1_ind = lzd(a_man);
        if (leading_1_ind <= 9) {
            a_man          = a_man << (10 - leading_1_ind);
            a_exp_unbiased = a_exp_unbiased - (10 - leading_1_ind);
        }
    }

    if (!b_is_dnorm && !b_is_zero) {
        if (!b_is_zero)
            b_man = cbs(1, b_man, 10);
        b_exp_unbiased = b_exp - 15;
    } else {
        b_exp_unbiased = -14;
        // normalize
        leading_1_ind = lzd(b_man);
        if (leading_1_ind <= 9) {
            b_man          = b_man << (10 - leading_1_ind);
            b_exp_unbiased = b_exp_unbiased - (10 - leading_1_ind);
        }
    }

    if (!c_is_dnorm && !c_is_zero) {
        if (!c_is_zero)
            c_man = cbs(1, c_man, 10);
        c_exp_unbiased = c_exp - 15;
    } else {
        c_exp_unbiased = -14;
        // normalize
        leading_1_ind = lzd(c_man);
        if (leading_1_ind <= 9) {
            c_man          = c_man << (10 - leading_1_ind);
            c_exp_unbiased = c_exp_unbiased - (10 - leading_1_ind);
        }
    }
    uint32_t ab_man          = a_man * b_man;
    int16_t  ab_exp_unbiased = a_exp_unbiased + b_exp_unbiased;
    uint8_t  res_sgn;

    uint8_t a_is_inf = is_inf_fp16(a);
    uint8_t b_is_inf = is_inf_fp16(b);
    uint8_t c_is_inf = is_inf_fp16(c);

    uint8_t mul_res_is_zero = (a_is_zero | b_is_zero);
    uint8_t a_is_nan        = is_nan_fp16(a);
    uint8_t b_is_nan        = is_nan_fp16(b);
    uint8_t c_is_nan        = is_nan_fp16(c);

    // exceptions
    uint8_t mul_res_is_inf = (!a_is_zero && b_is_inf) | (!b_is_zero && a_is_inf);

    uint8_t mul_res_is_def_nan = (a_is_zero & b_is_inf) | (b_is_zero & a_is_inf);

    uint8_t mul_res_is_nan = mul_res_is_def_nan | a_is_nan | b_is_nan;

    // Both sources are inf with (different signs and add instruction) or (same signs and sub instruction). i.e. inf-inf
    uint8_t res_is_def_nan = (mul_res_is_inf & c_is_inf & sub_ind);

    // nan handling - if any input is nan or result is nan - return nan
    uint8_t res_is_nan = res_is_def_nan | c_is_nan | mul_res_is_nan;

    // add res is inf - if one of the sources is infinity and it's add instruction OR one of the sources is infinity -
    // select the exception result
    uint8_t res_is_inf = !res_is_nan && ((mul_res_is_inf && c_is_inf && !sub_ind) || (mul_res_is_inf ^ c_is_inf));

    // special results - bypass calculated result
    if (res_is_inf) {
        if (mul_res_is_inf & !c_is_inf)
            res_sgn = ab_sgn;
        else // if (!mul_res_is_inf & c_is_inf)
            res_sgn = c_sgn;

        result       = cbs(res_sgn, 0x7C00, 15); //+-inf
        result_ready = 1;
    }
    if (res_is_nan && (result_ready == 0)) {
        //      if (b_is_nan)
        //        return (b | QNAN_BIT);
        //      if (a_is_nan)
        //        return (a | QNAN_BIT);
        //      if (c_is_nan)
        //        return (c | QNAN_BIT);
        result       = DEFAULT_NAN_FP16; // indefinite nan value
        result_ready = 1;
    }

    uint8_t sign_zero_exception;
    if (!c_is_zero && mul_res_is_zero)
        sign_zero_exception = c_sgn;
    else if (c_is_zero && !mul_res_is_zero)
        sign_zero_exception = ab_sgn;
    else
        sign_zero_exception = (round_mode == RND_TO_NINF) && (c_sgn ^ ab_sgn);

    sign_zero_exception = sign_zero_exception || (c_sgn && ab_sgn);

    uint8_t c_sgn_zero = (c_is_zero && (a_is_zero || b_is_zero)) ? sign_zero_exception : c_sgn;

    c = c_is_zero ? cbs(c_sgn_zero, 0x0000, 15) : c;

    if ((a_is_zero | b_is_zero) & (result_ready == 0)) {
        if (fp16_ftz_out) {
            if (c_is_dnorm) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(c, 15, 15) == 0))
                    result = 0x0400; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(c, 15, 15) == 1))
                    result = 0x8400; // res=-min_normal
                else
                    result = cbs(c_sgn_zero, 0x0000, 15); // res=+-0

                result_ready = 1;
            } else {
                result       = c;
                result_ready = 1;
            }
        } else {
            result       = c;
            result_ready = 1;
        }
    }

    // align c_man to ab_man (regardless of exponents)
    // compensate for 10 additional fraction bits after multiplication
    c_man = c_man << 10;
    // add 2 lsbs for G and R bits, for both operands (Guard and Round - for rounding at the end)
    ab_man = ab_man << 2;
    c_man  = c_man << 2;

    // Sticky bits
    uint16_t c_s  = 0;
    uint16_t ab_s = 0;

    int16_t res_exp_unbiased;

    int16_t exp_diff = c_exp_unbiased - ab_exp_unbiased;

    if (exp_diff >= 0) {
        c_s = 0;
        if (exp_diff > 24) // larger exp_diff means that we are discarding the entire ab_man
            exp_diff = 24;
        if (exp_diff > 2)
            ab_s = (sbs(ab_man, exp_diff - 3 + 2, 0 + 2) != 0);
        else
            ab_s = 0;
        ab_man           = ab_man >> exp_diff;
        res_exp_unbiased = c_exp_unbiased;
    } else {
        ab_s = 0;
        if (exp_diff < -24) // smaller exp_diff means that we are discarding the entire c_man
            exp_diff = -24;
        if (-exp_diff > 2)
            c_s = (sbs(c_man, -exp_diff - 3 + 2, 0 + 2) != 0);
        else
            c_s = 0;
        c_man            = c_man >> (-exp_diff);
        res_exp_unbiased = ab_exp_unbiased;
    }
    uint16_t res_s = ab_s | c_s;

    uint32_t res_man;
    int32_t  man_diff;

    if (sub_ind) {
        man_diff = cbs(ab_man, ab_s, 1) - cbs(c_man, c_s, 1);
        if (man_diff < 0) {
            res_man = (-man_diff) >> 1;
            res_sgn = c_sgn;
        } else if (man_diff > 0) {
            res_man = man_diff >> 1;
            res_sgn = ab_sgn;
        } else {
            res_man = 0;
            // res_sgn = 0;

            if (ab_s && !c_s)
                res_sgn = ab_sgn;
            else if (!ab_s && c_s)
                res_sgn = c_sgn;
            else
                res_sgn =
                    (round_mode ==
                     RND_TO_NINF); // x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        }
    } else {
        res_man = ab_man + c_man;
        res_sgn = c_sgn;
    }

    // find leading 1
    leading_1_ind = lzd(res_man);

    // normalize
    if (leading_1_ind == 24) {
        res_s            = res_s | (sbs(res_man, 1, 0) != 0);
        res_man          = res_man >> 2;
        res_exp_unbiased = res_exp_unbiased + 2;
    } else if (leading_1_ind == 23) {
        res_s            = res_s | sbs(res_man, 0, 0);
        res_man          = res_man >> 1;
        res_exp_unbiased = res_exp_unbiased + 1;
    } else if (leading_1_ind <= 22) {
        // res_s = res_s;
        res_man          = res_man << (22 - leading_1_ind);
        res_exp_unbiased = res_exp_unbiased - (22 - leading_1_ind);
    } else {
        // res_man is 0
        if (res_s == 0) // result is 0, and grs are 0, no need for rounding
            res_exp_unbiased = -15;
        else
            res_exp_unbiased = res_exp_unbiased - 25;
    }

    // separate GR bits from mantissa
    uint8_t res_g = sbs(res_man, 11, 11);
    uint8_t res_r = sbs(res_man, 10, 10);
    // fix Sticky bit
    res_s = res_s | (sbs(res_man, 9, 0) != 0);
    // align mantissa to the right (dispose of 10 extra fraction bits from mult + 1 G bit + 1 R bit)
    res_man = res_man >> 12;

    // round
    uint8_t need_rnd;

    // detect underflow
    int16_t shift_val;
    if ((res_exp_unbiased < -15) || (res_exp_unbiased == -15 && res_man != 0)) {
        // denormalize small values
        shift_val        = -14 - res_exp_unbiased;
        res_exp_unbiased = -15;

        if (shift_val > 30) // 19
            shift_val = 30;
        if (shift_val == 1) {
            res_s = res_s | res_r;
            res_r = res_g;
            res_g = sbs(res_man, 0, 0);
        } else if (shift_val == 2) {
            // fix Sticky bit
            res_s = res_s | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        } else // shift_val>=3
        {
            // fix Sticky bit
            res_s = res_s | (sbs(res_man, shift_val - 3, 0) != 0) | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        }

        res_man  = res_man >> shift_val;
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        res_man = res_man + need_rnd;

        if (sbs(res_man, 10, 10) == 1)
            res_exp_unbiased = res_exp_unbiased + 1;
    } else {
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        res_man = res_man + need_rnd;
        if (sbs(res_man, 11, 11) == 1) {
            res_s            = res_s | res_r;
            res_r            = res_g;
            res_g            = sbs(res_man, 0, 0);
            res_man          = res_man >> 1;
            res_exp_unbiased = res_exp_unbiased + 1;
        }

        // detect overflow
        if (res_exp_unbiased >= 16) {
            // result is +inf or -inf
            if ((round_mode == RND_TO_0) || (round_mode == RND_TO_PINF && res_sgn == 1) ||
                (round_mode == RND_TO_NINF && res_sgn == 0)) {
                // largest value which is not inf
                res_exp_unbiased = 15;
                res_man          = 0x3FF;
            } else {
                // inf
                res_exp_unbiased = 16;
                res_man          = 0;
            }
        }
    }
    //  else

    if (result_ready == 0) {
        // construct result
        result = ibs(result, 15, 15, res_sgn);
        result = ibs(result, 14, 10, (res_exp_unbiased + 15));
        result = ibs(result, 9, 0, res_man);

        if (fp16_ftz_out) {
            // flush result_16bit if denormal
            if (is_denorm_fp16(result)) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(result, 15, 15) == 0))
                    result = 0x0400; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(result, 15, 15) == 1))
                    result = 0x8400; // res=-min_normal
                else
                    result = ibs(result, 14, 0, 0); // res=+-0
            }
        }
    }
    if (is_nan_fp16(result))
        result = DEFAULT_NAN_FP16; // indefinite nan value

    // uint8_t special_non_equal_case = 0;

    // // Round to Nearest Even
    // //  C is much smaller than A*B
    // //  @rounding when having exactly half as fraction
    // //  our BF16 and x86-FP32+convert might differ in their rounding up / down
    // if (round_mode == RND_TO_NE && sbs(result_32bit, 12, 0) == 0x1000 && sbs(result_16bit, 0, 0) == 0 &&
    // sbs(result, 0, 0) == 1)
    // special_non_equal_case = 1;

    // if (((result != result_16bit) && special_non_equal_case == 0)) {
    // printf(
    // "a=%04x, b=%04x, c=%04x, res_x86=%04x res=%04x round_mode %d\n", a, b, c, result_16bit, result, round_mode);
    // assert(0 && "Warning: FP16 mismatch (x86+convert vs. our reference code");
    // }
    // if (((result != result_16bit) && special_non_equal_case == 1))
    return result;

    // return result_16bit;
}

////////////////////////until here FP16
///////////////////////////////////////

uint32_t mme_lfsr32(const uint32_t prev_val, const uint32_t polynomial)
{
    uint32_t lfsr = prev_val;
    uint32_t lsb  = lfsr & 1;
    lfsr >>= 1;
    if (lsb) {
        lfsr ^= polynomial;
    }

    return lfsr;
}

uint16_t mme_lfsr16(const uint16_t prev_val, const uint16_t polynomial)
{
    uint16_t lfsr = prev_val;
    uint16_t lsb  = lfsr & 1;
    lfsr >>= 1;
    if (lsb) {
        lfsr ^= polynomial;
    }
    return lfsr;
}

static inline uint32_t SaturatedAddUnsigned(uint32_t op1, uint32_t op2, uint32_t max_val, uint32_t min_val)
{
    uint64_t intermediateVal = (uint64_t)(op1) + (uint64_t)(op2);
    uint32_t res1            = (intermediateVal > (uint64_t)max_val)
                        ? max_val
                        : (intermediateVal < (uint64_t)min_val) ? min_val : (uint32_t)intermediateVal;
    return res1;
}
static inline int32_t SaturatedAddSigned(int32_t op1, int32_t op2, int32_t max_val, int32_t min_val)
{
    int64_t intermediateVal = (int64_t)(op1) + (int64_t)(op2);
    int32_t res1            = (intermediateVal > (int64_t)max_val)
                       ? max_val
                       : (intermediateVal < (int64_t)min_val) ? min_val : (int32_t)intermediateVal;
    return res1;
}

static inline uint32_t SaturatedSubUnsigned(uint32_t op1, uint32_t op2, uint32_t max_val, uint32_t min_val)
{
    int64_t  intermediateVal = (int64_t)(op1) - (int64_t)(op2);
    uint32_t res1            = (intermediateVal > (int64_t)max_val)
                        ? max_val
                        : (intermediateVal < (int64_t)min_val) ? min_val : (uint32_t)intermediateVal;
    return res1;
}
static inline int32_t SaturatedSubSigned(int32_t op1, int32_t op2, int32_t max_val, int32_t min_val)
{
    int64_t intermediateVal = (int64_t)(op1) - (int64_t)(op2);
    int32_t res1            = (intermediateVal > (int64_t)max_val)
                       ? max_val
                       : (intermediateVal < (int64_t)min_val) ? min_val : (int32_t)intermediateVal;
    return res1;
}

static inline uint32_t MaxUnsigned(uint32_t op1, uint32_t op2)
{
    uint32_t res1 = (op1 >= op2) ? op1 : op2;
    return res1;
}

static inline int32_t MaxSigned(int32_t op1, int32_t op2)
{
    int32_t res1 = (op1 >= op2) ? op1 : op2;
    return res1;
}

static inline uint32_t MaxFp32(uint32_t op1, uint32_t op2, bool suppress_nans)
{
    uint32_t res1;
    // float    res;
    if (suppress_nans) {
        if (is_nan_fp32(op1) && is_nan_fp32(op2))
            res1 = op2;
        else if (is_nan_fp32(op1) && !is_nan_fp32(op2))
            res1 = op2;
        else if (!is_nan_fp32(op1) && is_nan_fp32(op2))
            res1 = op1;
        else {
            res1 = (*(float*)&op1) >= (*(float*)&op2) ? op1 : op2;
            // res  = fmaxf(*(float*)&op2, *(float*)&op1);
            //  res1 = *(uint32_t*)&res;
        }
    } else
        res1 = (*(float*)&op1) >= (*(float*)&op2) ? op1 : op2;
    return res1;
}

static inline uint16_t MaxBf16(uint16_t op1, uint16_t op2, bool suppress_nans)
{
    uint16_t res1;
    // float    res;
    if (suppress_nans) {
        if (is_nan_bfp16(op1) && is_nan_bfp16(op2))
            res1 = op2;
        else if (is_nan_bfp16(op1) && !is_nan_bfp16(op2))
            res1 = op2;
        else if (!is_nan_bfp16(op1) && is_nan_bfp16(op2))
            res1 = op1;
        else {
            res1 = (bf16_to_float(op1)) >= (bf16_to_float(op2)) ? op1 : op2;
            // res  = fmaxf(bf16_to_float(op2), bf16_to_float(op1));
            // res1 = fp32_to_bf16(res, 0, 0, 0, 0, 0);
        }
    } else
        res1 = (bf16_to_float(op1)) >= (bf16_to_float(op2)) ? op1 : op2;
    return res1;
}

static inline uint16_t MaxFp16(uint16_t op1, uint16_t op2, bool suppress_nans)
{
    uint16_t res1;
    // uint32_t op1_32bit = fp16_to_fp32(op1, 0);
    // uint32_t op2_32bit = fp16_to_fp32(op2, 0);
    // float    res;
    if (suppress_nans) {
        if (is_nan_fp16(op1) && is_nan_fp16(op2))
            res1 = op2;
        else if (is_nan_fp16(op1) && !is_nan_fp16(op2))
            res1 = op2;
        else if (!is_nan_fp16(op1) && is_nan_fp16(op2))
            res1 = op1;
        else {
            res1 = (fp16_to_float(op1)) >= (fp16_to_float(op2)) ? op1 : op2;
            // res  = fmaxf(*(float*)&op2_32bit, *(float*)&op1_32bit);
            // res1 = fp32_to_fp16(res, 0, 0, 0, 0, 0);
        }
    } else
        res1 = (fp16_to_float(op1)) >= (fp16_to_float(op2)) ? op1 : op2;
    return res1;
}

static inline uint32_t MinUnsigned(uint32_t op1, uint32_t op2)
{
    uint32_t res1 = (op1 <= op2) ? op1 : op2;
    return res1;
}

static inline int32_t MinSigned(int32_t op1, int32_t op2)
{
    int32_t res1 = (op1 <= op2) ? op1 : op2;
    return res1;
}

static inline uint32_t MinFp32(uint32_t op1, uint32_t op2, bool suppress_nans)
{
    uint32_t res1;
    // float    res;
    if (suppress_nans) {
        if (is_nan_fp32(op1) && is_nan_fp32(op2))
            res1 = op2;
        else if (is_nan_fp32(op1) && !is_nan_fp32(op2))
            res1 = op2;
        else if (!is_nan_fp32(op1) && is_nan_fp32(op2))
            res1 = op1;
        else {
            res1 = (*(float*)&op1) <= (*(float*)&op2) ? op1 : op2;
            // res  = fminf(*(float*)&op2_32bit, *(float*)&op1_32bit);
            // res1 = fp32_to_fp16(res, 0, 0, 0, 0, 0);
        }
    } else
        res1 = (*(float*)&op1) <= (*(float*)&op2) ? op1 : op2;
    return res1;
}

static inline uint16_t MinBf16(uint16_t op1, uint16_t op2, bool suppress_nans)
{
    uint16_t res1;
    // float    res;
    if (suppress_nans) {
        if (is_nan_bfp16(op1) && is_nan_bfp16(op2))
            res1 = op2;
        else if (is_nan_bfp16(op1) && !is_nan_bfp16(op2))
            res1 = op2;
        else if (!is_nan_bfp16(op1) && is_nan_bfp16(op2))
            res1 = op1;
        else {
            res1 = (bf16_to_float(op1)) <= (bf16_to_float(op2)) ? op1 : op2;
            // res  = fminf(bf16_to_float(op2), bf16_to_float(op1));
            // res1 = fp32_to_bf16(res, 0, 0, 0, 0, 0);
        }
    } else
        res1 = (bf16_to_float(op1)) <= (bf16_to_float(op2)) ? op1 : op2;
    return res1;
}

static inline uint16_t MinFp16(uint16_t op1, uint16_t op2, bool suppress_nans)
{
    uint16_t res1;
    // uint32_t op1_32bit = fp16_to_fp32(op1, 0);
    // uint32_t op2_32bit = fp16_to_fp32(op2, 0);
    // float    res;
    if (suppress_nans) {
        if (is_nan_fp16(op1) && is_nan_fp16(op2))
            res1 = op2;
        else if (is_nan_fp16(op1) && !is_nan_fp16(op2))
            res1 = op2;
        else if (!is_nan_fp16(op1) && is_nan_fp16(op2))
            res1 = op1;
        else {
            res1 = (fp16_to_float(op1)) <= (fp16_to_float(op2)) ? op1 : op2;
            // res  = fminf(*(float*)&op2_32bit, *(float*)&op1_32bit);
            // res1 = fp32_to_fp16(res, 0, 0, 0, 0, 0);
        }
    } else
        res1 = (fp16_to_float(op1)) <= (fp16_to_float(op2)) ? op1 : op2;
    return res1;
}

#define VECTOR_SIZE 4

static inline uint32_t ST_TNSR_RMW(uint32_t src1_32bit,
                                   uint32_t src2_32bit,
                                   uint8_t  OpcodeRMW,
                                   uint8_t  DataTypeRMW,
                                   uint8_t  round_mode,
                                   bool     clip_fp,
                                   bool     clip_fp_inf_input,
                                   bool     suppress_nans)
{
    uint8_t* src1      = (uint8_t*)&src1_32bit;
    uint8_t* src2      = (uint8_t*)&src2_32bit;
    uint32_t dst_32bit = 0;
    uint8_t* dst       = (uint8_t*)&dst_32bit;

    set_rounding_mode(round_mode);

    uint8_t elementSize;
    switch (OpcodeRMW) {
        case ST_TNSR_RMW_ADD:
            switch (DataTypeRMW) {
                case ST_TNSR_RMW_UINT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0xFFFFFFFF, 0x0);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int32_t op1, op2;
                        op1                  = *(int32_t*)&src1[itr];
                        op2                  = *(int32_t*)&src2[itr];
                        int32_t res1         = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x7FFFFFFF, 0x80000000);
                        *(int32_t*)&dst[itr] = (int32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint32_t res1         = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0x0000FFFF, 0x0);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int16_t op1, op2;
                        op1                  = *(int16_t*)&src1[itr];
                        op2                  = *(int16_t*)&src2[itr];
                        int32_t res1         = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x00007FFF, 0xFFFF8000);
                        *(int16_t*)&dst[itr] = (int16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint32_t res1        = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0x000000FF, 0x0);
                        *(uint8_t*)&dst[itr] = (uint8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int8_t op1, op2;
                        op1                 = *(int8_t*)&src1[itr];
                        op2                 = *(int8_t*)&src2[itr];
                        int32_t res1        = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x0000007F, 0xFFFFFF80);
                        *(int8_t*)&dst[itr] = (int8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_BF16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = fma_bfp16(op1, (uint16_t)UNIT_VAL_BF16, op2, round_mode, 1);
                        if (clip_fp && is_inf_bfp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_bfp16(op1) && !is_inf_bfp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = fma_fp16_fp16(op1, (uint16_t)UNIT_VAL_FP16, op2, round_mode, 1, 1);
                        if (clip_fp && is_inf_fp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp16(op1) && !is_inf_fp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = fma_fp32(op1, (uint32_t)UNIT_VAL_FP32, op2, round_mode, 1);
                        if (clip_fp && is_inf_fp32(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp32(op1) && !is_inf_fp32(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1           = *(uint8_t*)&src1[itr];
                        op2           = *(uint8_t*)&src2[itr];
                        uint16_t res1 = fma_fp16_fp16(op1 << 8, (uint16_t)UNIT_VAL_FP16, op2 << 8, round_mode, 1, 1);
                        uint32_t res_fp32 = fp16_to_fp32(res1, 0);
                        uint8_t  res_fp8;
                        res_fp8 = fp32_to_fp8(*(float*)&res_fp32,
                                              5 /*exp_width*/,
                                              2 /*man_width*/,
                                              15 /*exp_bias*/,
                                              round_mode,
                                              0,
                                              1,
                                              0,
                                              0,
                                              0,
                                              0);
                        if (clip_fp && fp8_is_infinity(res_fp8, 2) &&
                            (clip_fp_inf_input || (!fp8_is_infinity(op1, 2) && !fp8_is_infinity(op2, 2)))) {
                            res_fp8 = res_fp8 - 1; // will return +/-max_norm_value
                        }
                        *(uint8_t*)&dst[itr] = (uint8_t)res_fp8;
                    }
                    break;
                default: break;
            }
            break;
        case ST_TNSR_RMW_SUB:
            switch (DataTypeRMW) {
                case ST_TNSR_RMW_UINT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = SaturatedSubUnsigned((uint32_t)op1, (uint32_t)op2, 0xFFFFFFFF, 0x0);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int32_t op1, op2;
                        op1                  = *(int32_t*)&src1[itr];
                        op2                  = *(int32_t*)&src2[itr];
                        int32_t res1         = SaturatedSubSigned((int32_t)op1, (int32_t)op2, 0x7FFFFFFF, 0x80000000);
                        *(int32_t*)&dst[itr] = (int32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint32_t res1         = SaturatedSubUnsigned((uint32_t)op1, (uint32_t)op2, 0x0000FFFF, 0x0);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int16_t op1, op2;
                        op1                  = *(int16_t*)&src1[itr];
                        op2                  = *(int16_t*)&src2[itr];
                        int32_t res1         = SaturatedSubSigned((int32_t)op1, (int32_t)op2, 0x00007FFF, 0xFFFF8000);
                        *(int16_t*)&dst[itr] = (int16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint32_t res1        = SaturatedSubUnsigned((uint32_t)op1, (uint32_t)op2, 0x000000FF, 0x0);
                        *(uint8_t*)&dst[itr] = (uint8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int8_t op1, op2;
                        op1                 = *(int8_t*)&src1[itr];
                        op2                 = *(int8_t*)&src2[itr];
                        int32_t res1        = SaturatedSubSigned((int32_t)op1, (int32_t)op2, 0x0000007F, 0xFFFFFF80);
                        *(int8_t*)&dst[itr] = (int8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_BF16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = fma_bfp16(op1, (uint16_t)UNIT_VAL_BF16, op2 ^ 0x8000, round_mode, 1);
                        if (clip_fp && is_inf_bfp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_bfp16(op1) && !is_inf_bfp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1           = *(uint16_t*)&src1[itr];
                        op2           = *(uint16_t*)&src2[itr];
                        uint16_t res1 = fma_fp16_fp16(op1, (uint16_t)UNIT_VAL_FP16, op2 ^ 0x8000, round_mode, 1, 1);
                        if (clip_fp && is_inf_fp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp16(op1) && !is_inf_fp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = fma_fp32(op1, (uint32_t)UNIT_VAL_FP32, op2 ^ 0x80000000, round_mode, 1);
                        if (clip_fp && is_inf_fp32(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp32(op1) && !is_inf_fp32(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1 = *(uint8_t*)&src1[itr];
                        op2 = *(uint8_t*)&src2[itr];
                        uint16_t res1 =
                            fma_fp16_fp16(op1 << 8, (uint16_t)UNIT_VAL_FP16, (op2 << 8) ^ 0x8000, round_mode, 1, 1);
                        uint32_t res_fp32 = fp16_to_fp32(res1, 0);
                        uint8_t  res_fp8;
                        res_fp8 = fp32_to_fp8(*(float*)&res_fp32,
                                              5 /*exp_width*/,
                                              2 /*man_width*/,
                                              15 /*exp_bias*/,
                                              round_mode,
                                              0,
                                              1,
                                              0,
                                              0,
                                              0,
                                              0);
                        if (clip_fp && fp8_is_infinity(res_fp8, 2) &&
                            (clip_fp_inf_input || (!fp8_is_infinity(op1, 2) && !fp8_is_infinity(op2, 2)))) {
                            res_fp8 = res_fp8 - 1; // will return +/-max_norm_value
                        }
                        *(uint8_t*)&dst[itr] = (uint8_t)res_fp8;
                    }
                    break;
                default: break;
            }
            break;
        case ST_TNSR_RMW_MIN:
            switch (DataTypeRMW) {
                case ST_TNSR_RMW_UINT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = MinUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int32_t op1, op2;
                        op1                  = *(int32_t*)&src1[itr];
                        op2                  = *(int32_t*)&src2[itr];
                        int32_t res1         = MinSigned((int32_t)op1, (int32_t)op2);
                        *(int32_t*)&dst[itr] = (int32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint32_t res1         = MinUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int16_t op1, op2;
                        op1                  = *(int16_t*)&src1[itr];
                        op2                  = *(int16_t*)&src2[itr];
                        int32_t res1         = MinSigned((int32_t)op1, (int32_t)op2);
                        *(int16_t*)&dst[itr] = (int16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint32_t res1        = MinUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint8_t*)&dst[itr] = (uint8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int8_t op1, op2;
                        op1                 = *(int8_t*)&src1[itr];
                        op2                 = *(int8_t*)&src2[itr];
                        int32_t res1        = MinSigned((int32_t)op1, (int32_t)op2);
                        *(int8_t*)&dst[itr] = (int8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_BF16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = MinBf16(op1, op2, suppress_nans);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = MinFp16(op1, op2, suppress_nans);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = MinFp32(op1, op2, suppress_nans);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint16_t res1        = MinFp16(op1 << 8, op2 << 8, suppress_nans);
                        uint8_t  res_fp8     = res1 >> 8;
                        *(uint8_t*)&dst[itr] = (uint8_t)res_fp8;
                    }
                    break;
                default: break;
            }
            break;
        case ST_TNSR_RMW_MAX:
            switch (DataTypeRMW) {
                case ST_TNSR_RMW_UINT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = MaxUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int32_t op1, op2;
                        op1                  = *(int32_t*)&src1[itr];
                        op2                  = *(int32_t*)&src2[itr];
                        int32_t res1         = MaxSigned((int32_t)op1, (int32_t)op2);
                        *(int32_t*)&dst[itr] = (int32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint32_t res1         = MaxUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int16_t op1, op2;
                        op1                  = *(int16_t*)&src1[itr];
                        op2                  = *(int16_t*)&src2[itr];
                        int32_t res1         = MaxSigned((int32_t)op1, (int32_t)op2);
                        *(int16_t*)&dst[itr] = (int16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint32_t res1        = MaxUnsigned((uint32_t)op1, (uint32_t)op2);
                        *(uint8_t*)&dst[itr] = (uint8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int8_t op1, op2;
                        op1                 = *(int8_t*)&src1[itr];
                        op2                 = *(int8_t*)&src2[itr];
                        int32_t res1        = MaxSigned((int32_t)op1, (int32_t)op2);
                        *(int8_t*)&dst[itr] = (int8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_BF16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = MaxBf16(op1, op2, suppress_nans);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1                   = *(uint16_t*)&src1[itr];
                        op2                   = *(uint16_t*)&src2[itr];
                        uint16_t res1         = MaxFp16(op1, op2, suppress_nans);
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1                   = *(uint32_t*)&src1[itr];
                        op2                   = *(uint32_t*)&src2[itr];
                        uint32_t res1         = MaxFp32(op1, op2, suppress_nans);
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1                  = *(uint8_t*)&src1[itr];
                        op2                  = *(uint8_t*)&src2[itr];
                        uint16_t res1        = MaxFp16(op1 << 8, op2 << 8, suppress_nans);
                        uint8_t  res_fp8     = res1 >> 8;
                        *(uint8_t*)&dst[itr] = (uint8_t)res_fp8;
                    }
                    break;
                default: break;
            }
            break;
        case ST_TNSR_RMW_MAX_0_ADD:
            switch (DataTypeRMW) {
                case ST_TNSR_RMW_UINT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1           = *(uint32_t*)&src1[itr];
                        op2           = *(uint32_t*)&src2[itr];
                        uint32_t res1 = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0xFFFFFFFF, 0x0);
                        // res1 is always positive, no need to do MAX(res1,0)
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int32_t op1, op2;
                        op1                  = *(int32_t*)&src1[itr];
                        op2                  = *(int32_t*)&src2[itr];
                        int32_t res1         = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x7FFFFFFF, 0x80000000);
                        res1                 = MaxSigned(res1, 0);
                        *(int32_t*)&dst[itr] = (int32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1           = *(uint16_t*)&src1[itr];
                        op2           = *(uint16_t*)&src2[itr];
                        uint32_t res1 = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0x0000FFFF, 0x0);
                        // res1 is always positive, no need to do MAX(res1,0)
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int16_t op1, op2;
                        op1                  = *(int16_t*)&src1[itr];
                        op2                  = *(int16_t*)&src2[itr];
                        int32_t res1         = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x00007FFF, 0xFFFF8000);
                        res1                 = MaxSigned(res1, 0);
                        *(int16_t*)&dst[itr] = (int16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_UINT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1           = *(uint8_t*)&src1[itr];
                        op2           = *(uint8_t*)&src2[itr];
                        uint32_t res1 = SaturatedAddUnsigned((uint32_t)op1, (uint32_t)op2, 0x000000FF, 0x0);
                        // res1 is always positive, no need to do MAX(res1,0)
                        *(uint8_t*)&dst[itr] = (uint8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_INT8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        int8_t op1, op2;
                        op1                 = *(int8_t*)&src1[itr];
                        op2                 = *(int8_t*)&src2[itr];
                        int32_t res1        = SaturatedAddSigned((int32_t)op1, (int32_t)op2, 0x0000007F, 0xFFFFFF80);
                        res1                = MaxSigned(res1, 0);
                        *(int8_t*)&dst[itr] = (int8_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_BF16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1           = *(uint16_t*)&src1[itr];
                        op2           = *(uint16_t*)&src2[itr];
                        uint16_t res1 = fma_bfp16(op1, (uint16_t)UNIT_VAL_BF16, op2, round_mode, 1);
                        res1          = MaxBf16(res1, 0, suppress_nans);
                        if (res1 == 0x8000) //-0 --> +0, align to HW implementation
                            res1 = 0;
                        if (clip_fp && is_inf_bfp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_bfp16(op1) && !is_inf_bfp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP16:
                    elementSize = 2;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint16_t op1, op2;
                        op1           = *(uint16_t*)&src1[itr];
                        op2           = *(uint16_t*)&src2[itr];
                        uint16_t res1 = fma_fp16_fp16(op1, (uint16_t)UNIT_VAL_FP16, op2, round_mode, 1, 1);
                        res1          = MaxFp16(res1, 0, suppress_nans);
                        if (res1 == 0x8000) //-0 --> +0, align to HW implementation
                            res1 = 0;
                        if (clip_fp && is_inf_fp16(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp16(op1) && !is_inf_fp16(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint16_t*)&dst[itr] = (uint16_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP32:
                    elementSize = 4;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint32_t op1, op2;
                        op1           = *(uint32_t*)&src1[itr];
                        op2           = *(uint32_t*)&src2[itr];
                        uint32_t res1 = fma_fp32(op1, (uint32_t)UNIT_VAL_FP32, op2, round_mode, 1);
                        res1          = MaxFp32(res1, 0, suppress_nans);
                        if (res1 == 0x80000000) //-0 --> +0, align to HW implementation
                            res1 = 0;
                        if (clip_fp && is_inf_fp32(res1) &&
                            (clip_fp_inf_input || (!is_inf_fp32(op1) && !is_inf_fp32(op2)))) {
                            res1 = res1 - 1; // will return +/-max_norm_value
                        }
                        *(uint32_t*)&dst[itr] = (uint32_t)res1;
                    }
                    break;
                case ST_TNSR_RMW_FP8:
                    elementSize = 1;
                    for (uint32_t itr = 0; itr < VECTOR_SIZE; itr += elementSize) {
                        uint8_t op1, op2;
                        op1           = *(uint8_t*)&src1[itr];
                        op2           = *(uint8_t*)&src2[itr];
                        uint16_t res1 = fma_fp16_fp16(op1 << 8, (uint16_t)UNIT_VAL_FP16, op2 << 8, round_mode, 1, 1);
                        res1          = MaxFp16(res1, 0, suppress_nans);
                        if (res1 == 0x8000) //-0 --> +0, align to HW implementation
                            res1 = 0;
                        uint32_t res_fp32 = fp16_to_fp32(res1, 0);
                        uint8_t  res_fp8;
                        res_fp8              = fp32_to_fp8(*(float*)&res_fp32,
                                              5 /*exp_width*/,
                                              2 /*man_width*/,
                                              15 /*exp_bias*/,
                                              round_mode,
                                              0,
                                              1,
                                              0,
                                              0,
                                              0,
                                              0);
                        if (clip_fp && fp8_is_infinity(res_fp8, 2) &&
                            (clip_fp_inf_input || (!fp8_is_infinity(op1, 2) && !fp8_is_infinity(op2, 2)))) {
                            res_fp8 = res_fp8 - 1; // will return +/-max_norm_value
                        }
                        *(uint8_t*)&dst[itr] = (uint8_t)res_fp8;
                    }
                    break;
                default: break;
            }
            break;
        default: break;
    }
    return dst_32bit;
}

uint32_t executeOp(uint32_t src1_32bit,
                   uint32_t src2_32bit,
                   uint8_t  OpcodeRMW,
                   uint8_t  DataTypeRMW,
                   uint8_t  round_mode,
                   bool     clip_fp,
                   bool     clip_fp_inf_input,
                   bool     suppress_nans)
{
    return ST_TNSR_RMW(
        src1_32bit, src2_32bit, OpcodeRMW, DataTypeRMW, round_mode, clip_fp, clip_fp_inf_input, suppress_nans);
}

void select_rounding_mode_dp(double sum, float c, uint32_t c_exp)
{

    uint32_t ab_exp = lsbs(*(uint64_t*)&sum, 62, 52);
    uint64_t sum_c_ab;
    if (ab_exp != 0) {
        ab_exp = ab_exp - 1023 + 127;
    }

    uint64_t ab_man = libs(lsbs(*(uint64_t*)&sum, 51, 0), 52, 52, 1);

    uint32_t c_ab_exp_diff = (c_exp > ab_exp) ? (c_exp - ab_exp) : 0;

    uint8_t leading_one;

    bool     l_bit;
    bool     g_bit;
    bool     rs_bits;
    bool     need_rnd;
    uint64_t sticky;
    bool     same_sign;

    // if ab_exp is much larger than c_exp,
    // c becomes sticky bit and will affect rounding only if it makes the fraction > 0.5
    // therefore we check the special case:
    // if ab_man discarded bits are exactly 0.5, and c is all sticky
    // need to force roudning/no-rounding, depends on matching-signs/not-matching-signs
    if (((ab_exp - 52) > c_exp) && ((ab_man & 0x000001fffffffll) == 0x0000010000000ll)) {
        // same sign
        if ((sum > 0 && c > 0) || (sum < 0 && c < 0)) {
            // discarded_bits + stikcy > 0.5 --> round away from 0 (upward for positive, downward for negative)
            if (sum > 0)
                fesetround(FE_UPWARD);
            else
                fesetround(FE_DOWNWARD);
        } else if (c != 0) {
            // opppsite sign
            // discarded_bits + stikcy < 0.5 --> round towards 0 (downward for positive, upward for negative)
            fesetround(FE_TOWARDZERO);
            // if (sum > 0)
            // fesetround(FE_DOWNWARD);
            // else
            // fesetround(FE_UPWARD);
        }
    } else if (c_exp > ab_exp && c != 0) {
        // if c_exp is larger than ab_exp
        // discarded_bits can be partially inside double-precision sum result and partially outside (=sticky)
        // need to calculate the exact sum and force single rounding decision, with sticky bit
        sticky = lsbs(ab_man, c_ab_exp_diff - 1, 0);
        if (c_ab_exp_diff > 48)
            c_ab_exp_diff = 48;
        same_sign = (sbs(c, 31, 31) == lsbs(*(uint64_t*)&sum, 63, 63));
        if (same_sign)
            sum_c_ab = (((uint64_t)ibs(sbs(c, 22, 0), 23, 23, 1)) << 29ll) + (ab_man >> c_ab_exp_diff);
        else {
            sum_c_ab = (((uint64_t)ibs(sbs(c, 22, 0), 23, 23, 1)) << 29ll) - (ab_man >> c_ab_exp_diff);
            if (sticky != 0)
                sum_c_ab--;
        }

        leading_one = lzd_ll(sum_c_ab);

        l_bit    = lsbs(sum_c_ab, leading_one, leading_one) != 0;
        g_bit    = (leading_one >= 24) ? (lsbs(sum_c_ab, leading_one - 24, leading_one - 24) != 0) : 0;
        rs_bits  = (leading_one >= 25) ? (lsbs(sum_c_ab, leading_one - 25, 0) != 0 || sticky != 0) : (sticky != 0);
        need_rnd = (l_bit | rs_bits) & g_bit;

        // if we need rounding - force round away from zero  (upward for positive, downward for negative)
        // else - force round towards zero (downward for positive, upward for negative)
        if (need_rnd) {
            if (c > 0)
                fesetround(FE_UPWARD);
            else
                fesetround(FE_DOWNWARD);
        } else
            fesetround(FE_TOWARDZERO);
    } else // set rounding mode to RNE
        fesetround(FE_TONEAREST);
}

uint32_t fma_mul_add_tree_double_prec(uint32_t* a,
                                      uint32_t* b,
                                      uint32_t  c,
                                      uint8_t   N,
                                      uint8_t   K,
                                      bool      is_bias15,
                                      bool      dnorm_ftz,
                                      bool      c_in_tree)
{
    int       i;
    uint32_t* result;
    // bool mult_neg = 0;
    bool mult_pos = 0;

    if (dnorm_ftz) {
        // zero denormals
        for (i = 0; i < N; i++) {
            if (is_denorm_fp32(a[i]))
                a[i] = ibs(a[i], 30, 0, 0);
            if (is_denorm_fp32(b[i]))
                b[i] = ibs(b[i], 30, 0, 0);
        }

        if (is_denorm_fp32(c))
            c = ibs(c, 30, 0, 0);
    }

    int32_t max_exp = 0;
    // int64_t anbn_sum = 0;
    int32_t a_exp[16], b_exp[16], a_max_exp = 0, b_max_exp = 0, anbn_exp[16], c_exp = 0;
    c_exp = sbs(c, 30, 23);

    int32_t  anbn_exp_fp8;
    uint32_t anchor_factor_dnorm = 1;
    int32_t  min_norm_exp;

    double a_d, b_d;
    // find max_exp for calculating double precision anchor value
    for (i = 0; i < N; i++) {
        a_exp[i] = sbs(a[i], 30, 23);
        b_exp[i] = sbs(b[i], 30, 23);

        if (a_exp[i] == 0xff || b_exp[i] == 0xff)
            anbn_exp[i] = 2 * 0xff; // result is inf/nan
        else if (a_exp[i] != 0 && b_exp[i] != 0)
            anbn_exp[i] = a_exp[i] + b_exp[i];
        else {
            if (dnorm_ftz)
                anbn_exp[i] = 0; // result is 0
            else {
                a_d = (double)(*(float*)&a[i]);
                b_d = (double)(*(float*)&b[i]);
                if (is_denorm_fp32(a[i]))
                    a_exp[i] = (lsbs(*(uint64_t*)&a_d, 62, 52) - 1023 + 127);
                if (is_denorm_fp32(b[i]))
                    b_exp[i] = (lsbs(*(uint64_t*)&b_d, 62, 52) - 1023 + 127);
                if (is_zero_fp32(a[i]) || is_zero_fp32(b[i]))
                    anbn_exp[i] = 0;
                else
                    anbn_exp[i] = a_exp[i] + b_exp[i];
                if (VERBOSE)
                    printf("exponents %x %x %x\n", (unsigned)anbn_exp[i], (unsigned)a_exp[i], (unsigned)b_exp[i]);
            }
        }

        // special handling for fp8, since fp8 denormals are not normalized before FMA
        // FP8 always enter as FP9 with bias=15, min_norm_exp=-14, which is equivalent to FP32 biased exponent of 113
        // the normalization can affect the shift value when doing alignment, because we can get a slightly different
        // max_exp
        // same goes for FP32/FP19/BF16 denormals, but min_norm_exp = 1
        if (is_bias15 || (dnorm_ftz == 0)) {
            min_norm_exp = is_bias15 ? 113 : 1;
            anbn_exp_fp8 = anbn_exp[i];
            if (anbn_exp[i] != 2 * 0xff && anbn_exp[i] != 0) {
                if (a_exp[i] < min_norm_exp)
                    anbn_exp_fp8 += (min_norm_exp - a_exp[i]);
                if (b_exp[i] < min_norm_exp)
                    anbn_exp_fp8 += (min_norm_exp - b_exp[i]);
            }

            if (anbn_exp_fp8 >= max_exp) {
                max_exp           = anbn_exp_fp8;
                a_max_exp         = a[i];
                b_max_exp         = b[i];
                anchor_factor_dnorm = (1 << (anbn_exp_fp8 - anbn_exp[i]));
            }
        } else {
            if (anbn_exp[i] >= max_exp) {
                max_exp   = anbn_exp[i];
                a_max_exp = a[i];
                b_max_exp = b[i];
            }
        }
    }

    // subtract one bias from exponents sum
    if (max_exp != 0)
        max_exp -= 127;

    if (VERBOSE)
        printf("max_exp %d \n", max_exp);
    if (c_in_tree == 1) {
        // accumulator is sign+4+2.24 = sign+6.24 --> if msb is exp=0 then last integer bit is exp=-5
        // therefore no point in having max_exp smaller than -5
        if (max_exp < -5)
            max_exp = -5;
    }

    if (VERBOSE)
        printf("max_exp %d \n", max_exp);

    if (VERBOSE)
        printf("a %08x b %08x\n", (unsigned)a_max_exp, (unsigned)b_max_exp);

    if (dnorm_ftz == 1) {
        a_max_exp = a_max_exp & 0x7f800000;
        b_max_exp = b_max_exp & 0x7f800000;
    }

    double a_max_exp_d = (double)(*(float*)&a_max_exp);
    double b_max_exp_d = (double)(*(float*)&b_max_exp);

    uint64_t a_max_exp_64bit = libs(0, 62, 52, lsbs(*(uint64_t*)&a_max_exp_d, 62, 52));
    uint64_t b_max_exp_64bit = libs(0, 62, 52, lsbs(*(uint64_t*)&b_max_exp_d, 62, 52));

    if (VERBOSE)
        printf("a %08x b %08x\n", (unsigned)a_max_exp, (unsigned)b_max_exp);
    float anchor_factor = 1.0 * 64.0;
    if (VERBOSE)
        printf("factor %f anchor_factor_dnorm %x\n", anchor_factor, anchor_factor_dnorm);
    if (is_bias15 || (dnorm_ftz == 0))
        anchor_factor = anchor_factor * anchor_factor_dnorm;
    if (K == 4)
        anchor_factor *= (1 << 22);
    if (VERBOSE)
        printf("factor %f \n", anchor_factor);
    double anchor_pos;
    if (dnorm_ftz == 1)
        anchor_pos = (double)(*(float*)&a_max_exp) * (double)(*(float*)&b_max_exp);
    else
        anchor_pos = (*(double*)&a_max_exp_64bit) * (*(double*)&b_max_exp_64bit);

    if (VERBOSE)
        printf("anchor_pos %016lx \n", *(uint64_t*)&anchor_pos);
    anchor_pos        = anchor_pos * (double)anchor_factor;
    double anchor_neg = -anchor_pos;
    if (VERBOSE)
        printf("anchor_pos %016lx \n", *(uint64_t*)&anchor_pos);

    // if result is inf/nan, no need for anchor - otherwise we might get nan instead of inf
    if (max_exp == (2 * 0xff - 127)) {
        anchor_pos = 0.0;
        anchor_neg = 0.0;
    }

    // for sumAB we need rounding mode RZ
    fesetround(FE_TOWARDZERO);

    double sum_pos = anchor_pos;
    double sum_neg = anchor_neg;
    double mult_res;

    uint32_t exp_diff;
    uint32_t a_man;
    uint32_t b_man;
    uint64_t ab_man_exp_diff_47;
    uint32_t limit = 21 + K;

    // sum separately the positive/negative products
    for (i = 0; i < N; i++) {
        if (anbn_exp[i] != 0)
            anbn_exp[i] -= 127;
        exp_diff = max_exp - anbn_exp[i];

        if (VERBOSE) {
            printf("anbn_exp %d, max_exp %d, exp_diff %d\n", anbn_exp[i], max_exp, (int)exp_diff);
            printf("sum_pos %016lx \n", *(uint64_t*)&sum_pos);
            printf("sum_neg %016lx \n", *(uint64_t*)&sum_neg);
        }
        if (exp_diff < limit || c_in_tree == 1) {
            mult_res = (double)(*(float*)&a[i]) * (double)(*(float*)&b[i]);
            if (lsbs(*(uint64_t*)&mult_res, 63, 63) == 0) {
                sum_pos += mult_res;
                mult_pos = 1;
            } else {
                sum_neg += mult_res;
                // mult_neg = 1;

                if (VERBOSE) {
                    printf("mult_res %016lx \n", *(uint64_t*)&mult_res);
                    printf("sum_pos %016lx \n", *(uint64_t*)&sum_pos);
                    printf("sum_neg %016lx \n", *(uint64_t*)&sum_neg);
                }
            }
        } else if (exp_diff == limit) {
            a_man              = ibs(sbs(a[i], 22, 0), 23, 23, 1);
            b_man              = ibs(sbs(b[i], 22, 0), 23, 23, 1);
            ab_man_exp_diff_47 = (uint64_t)a_man * (uint64_t)b_man;
            if (lsbs(ab_man_exp_diff_47, 47, 47) != 0) {
                mult_res = (double)(*(float*)&a[i]) * (double)(*(float*)&b[i]);
                if (lsbs(*(uint64_t*)&mult_res, 63, 63) == 0) {
                    sum_pos += mult_res;
                    mult_pos = 1;
                } else {
                    sum_neg += mult_res;
                    // mult_neg = 1;
                }
            }
            if (VERBOSE) {
                printf("limit mult_res %016lx \n", *(uint64_t*)&mult_res);
                printf("limit sum_pos %016lx \n", *(uint64_t*)&sum_pos);
                printf("limit sum_neg %016lx \n", *(uint64_t*)&sum_neg);
            }
        }
    }
    uint64_t ab_man_for_print = lsbs(*(uint64_t*)&sum_pos, 51, 0);
    ab_man_for_print          = libs(ab_man_for_print, 52, 52, 1);
    // ab_man_for_print = ab_man_for_print;// >> 1ll;

    if (VERBOSE)
        printf("ab_man %016lx\n", ab_man_for_print);

    ab_man_for_print = lsbs(*(uint64_t*)&sum_neg, 51, 0);
    ab_man_for_print = libs(ab_man_for_print, 52, 52, 1);
    // ab_man_for_print = ab_man_for_print;// >> 1ll;

    if (VERBOSE)
        printf("ab_man %016lx\n", ab_man_for_print);

    double sum = sum_pos + sum_neg;

    // if sum is zero and all the multiplications were negative, force -0
    if (sum == 0 && mult_pos == 0) {
        sum = -0.0;
    }
    uint64_t ab_man = libs(lsbs(*(uint64_t*)&sum, 51, 0), 52, 52, 1);

    if (VERBOSE)
        printf("ab_man from sum %016lx\n", ab_man);

    // naive implemntation has issues of double rounding:
    // 1. rounding during double precision addition
    // 2. rounding when converting double precision to single precision
    // therefore, we check for special rounding cases which will result in mismatch,
    // and we use different rounding mode for forcing rounding/no-rounding in certain cases.
    // this imitates proper single rounding
    if (c_in_tree == 0)
        select_rounding_mode_dp(sum, c, c_exp);

    sum += (double)(*(float*)&c);

    if (c_in_tree == 0) {
        // for rounding during conversion we must have RNE
        fesetround(FE_TONEAREST);
    }

    bool override_inf = 0;
    if (c_in_tree == 1) {
        // check overflow and force +inf/-inf
        int64_t final_exp = lsbs(*(uint64_t*)&sum, 62, 52);
        // uint64_t final_man = lsbs(*(uint64_t*)&sum, 51, 0);
        if ((final_exp - 1023 + 127 > 254) && (final_exp != 2047)) // FP32 overflow and not NaN
            override_inf = 1;
    }

    if (VERBOSE)
        printf("sum %016lx \n", *(uint64_t*)&sum);
    float sum_fp = (float)sum;

    result = (uint32_t*)&sum_fp;

    if (VERBOSE)
        printf("result %08x \n", *result);
    if (override_inf) {
        if (sbs(*result, 31, 31))
            *result = 0xff800000;
        else
            *result = 0x7f800000;
    }

    if (dnorm_ftz) {
        // flush denormals to 0
        if (is_denorm_fp32(*result))
            *result = ibs(*result, 30, 0, 0);
    }
    // force default nan
    if (is_nan_fp32(*result))
        *result = DEFAULT_NAN_FP32;

    // disable +0/-0 logic
    if (c_in_tree == 1) {
        if (*result == 0x80000000)
            *result = 0;
    }
    return *result;
}

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
                            bool     c_after_norm)
{
    uint32_t a[4] = {a0, a1, a2, a3};
    uint32_t b[4] = {b0, b1, b2, b3};
    return fma_mul_add_tree_n(a, b, c, round_mode, 4, K, fp32_emul, c_after_norm, 0, 0, 0, 0, 1, 0);
}

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
                            bool     c_after_norm)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, K, fp32_emul, c_after_norm, 0, 0, 0, 0, 1, 0);
}

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
                                                 uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 1, 0, 0, 0, 0, 1, 0);
}
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
                                                  uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 12, 0, 1, 0, 0, 0, 0, 1, 0);
}

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
                                                  uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 1, 0, 0, 0, 0, 1, 0);
}

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
                                                       uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 0, 0, 0, 0, 0, 1, 0);
}
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
                                                        uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 12, 0, 0, 0, 0, 0, 0, 1, 0);
}

uint32_t fma_mul_add_tree_tf32_N8_K26_add_C_before_norm(uint32_t a0,
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
                                                        uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 0, 0, 0, 0, 1, 0);
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
                                                           uint8_t  round_mode)
{
    uint32_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint32_t b[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    return fma_mul_add_tree_double_prec(a, b, c, 8, 26, 0, 1, 0);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K4_add_C_in_tree_no_ftz(uint32_t a0,
                                                               uint32_t b0,
                                                               uint32_t a1,
                                                               uint32_t b1,
                                                               uint32_t c,
                                                               bool     clip_fp,
                                                               bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[9] = {a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 4, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K4_add_C_in_tree_no_ftz_dp(uint32_t a0,
                                                                  uint32_t b0,
                                                                  uint32_t a1,
                                                                  uint32_t b1,
                                                                  uint32_t c,
                                                                  bool     clip_fp,
                                                                  bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K26_add_C_in_tree_no_ftz(uint32_t a0,
                                                                uint32_t b0,
                                                                uint32_t a1,
                                                                uint32_t b1,
                                                                uint32_t c,
                                                                bool     clip_fp,
                                                                bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[9] = {a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 26, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K26_add_C_in_tree_no_ftz_dp(uint32_t a0,
                                                                   uint32_t b0,
                                                                   uint32_t a1,
                                                                   uint32_t b1,
                                                                   uint32_t c,
                                                                   bool     clip_fp,
                                                                   bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 26, 0, 0, 1);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K4_add_C_before_norm(uint32_t a0,
                                                            uint32_t b0,
                                                            uint32_t a1,
                                                            uint32_t b1,
                                                            uint32_t c,
                                                            uint8_t  round_mode,
                                                            bool     clip_fp,
                                                            bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[8] = {a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000};
    uint32_t b[8] = {b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000};

    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_tf32_emul_N2_K26_add_C_before_norm(uint32_t a0,
                                                             uint32_t b0,
                                                             uint32_t a1,
                                                             uint32_t b1,
                                                             uint32_t c,
                                                             uint8_t  round_mode,
                                                             bool     clip_fp,
                                                             bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[8] = {a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xff9fe000,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xff9fe000,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000};
    uint32_t b[8] = {b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b0_tf32 & 0xff9fe000,
                     b0_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000,
                     b1_tf32 & 0xff9fe000,
                     b1_tf32 & 0xffe00000};

    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_tf32_emul_N4_K4_add_C_before_norm(uint32_t a0,
                                                            uint32_t b0,
                                                            uint32_t a1,
                                                            uint32_t b1,
                                                            uint32_t a2,
                                                            uint32_t b2,
                                                            uint32_t a3,
                                                            uint32_t b3,
                                                            uint32_t c,
                                                            uint8_t  round_mode,
                                                            bool     clip_fp,
                                                            bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a2_tf32 = fp32_to_tf32(*(float*)&a2, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a3_tf32 = fp32_to_tf32(*(float*)&a3, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b2_tf32 = fp32_to_tf32(*(float*)&b2, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b3_tf32 = fp32_to_tf32(*(float*)&b3, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[16] = {a0_tf32 & 0xff9fe000,
                      a0_tf32 & 0xff9fe000,
                      a0_tf32 & 0xffe00000,
                      a0_tf32 & 0xffe00000,
                      a1_tf32 & 0xff9fe000,
                      a1_tf32 & 0xff9fe000,
                      a1_tf32 & 0xffe00000,
                      a1_tf32 & 0xffe00000,
                      a2_tf32 & 0xff9fe000,
                      a2_tf32 & 0xff9fe000,
                      a2_tf32 & 0xffe00000,
                      a2_tf32 & 0xffe00000,
                      a3_tf32 & 0xff9fe000,
                      a3_tf32 & 0xff9fe000,
                      a3_tf32 & 0xffe00000,
                      a3_tf32 & 0xffe00000};
    uint32_t b[16] = {b0_tf32 & 0xff9fe000,
                      b0_tf32 & 0xffe00000,
                      b0_tf32 & 0xff9fe000,
                      b0_tf32 & 0xffe00000,
                      b1_tf32 & 0xff9fe000,
                      b1_tf32 & 0xffe00000,
                      b1_tf32 & 0xff9fe000,
                      b1_tf32 & 0xffe00000,
                      b2_tf32 & 0xff9fe000,
                      b2_tf32 & 0xffe00000,
                      b2_tf32 & 0xff9fe000,
                      b2_tf32 & 0xffe00000,
                      b3_tf32 & 0xff9fe000,
                      b3_tf32 & 0xffe00000,
                      b3_tf32 & 0xff9fe000,
                      b3_tf32 & 0xffe00000};

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 4, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_tf32_emul_N4_K26_add_C_before_norm(uint32_t a0,
                                                             uint32_t b0,
                                                             uint32_t a1,
                                                             uint32_t b1,
                                                             uint32_t a2,
                                                             uint32_t b2,
                                                             uint32_t a3,
                                                             uint32_t b3,
                                                             uint32_t c,
                                                             uint8_t  round_mode,
                                                             bool     clip_fp,
                                                             bool     clip_fp_inf_input)
{
    uint32_t a0_tf32 = fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a1_tf32 = fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a2_tf32 = fp32_to_tf32(*(float*)&a2, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t a3_tf32 = fp32_to_tf32(*(float*)&a3, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b0_tf32 = fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b1_tf32 = fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b2_tf32 = fp32_to_tf32(*(float*)&b2, 0, 0, clip_fp, clip_fp_inf_input);
    uint32_t b3_tf32 = fp32_to_tf32(*(float*)&b3, 0, 0, clip_fp, clip_fp_inf_input);

    uint32_t a[16] = {a0_tf32 & 0xff9fe000,
                      a0_tf32 & 0xff9fe000,
                      a0_tf32 & 0xffe00000,
                      a0_tf32 & 0xffe00000,
                      a1_tf32 & 0xff9fe000,
                      a1_tf32 & 0xff9fe000,
                      a1_tf32 & 0xffe00000,
                      a1_tf32 & 0xffe00000,
                      a2_tf32 & 0xff9fe000,
                      a2_tf32 & 0xff9fe000,
                      a2_tf32 & 0xffe00000,
                      a2_tf32 & 0xffe00000,
                      a3_tf32 & 0xff9fe000,
                      a3_tf32 & 0xff9fe000,
                      a3_tf32 & 0xffe00000,
                      a3_tf32 & 0xffe00000};
    uint32_t b[16] = {b0_tf32 & 0xff9fe000,
                      b0_tf32 & 0xffe00000,
                      b0_tf32 & 0xff9fe000,
                      b0_tf32 & 0xffe00000,
                      b1_tf32 & 0xff9fe000,
                      b1_tf32 & 0xffe00000,
                      b1_tf32 & 0xff9fe000,
                      b1_tf32 & 0xffe00000,
                      b2_tf32 & 0xff9fe000,
                      b2_tf32 & 0xffe00000,
                      b2_tf32 & 0xff9fe000,
                      b2_tf32 & 0xffe00000,
                      b3_tf32 & 0xff9fe000,
                      b3_tf32 & 0xffe00000,
                      b3_tf32 & 0xff9fe000,
                      b3_tf32 & 0xffe00000};

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 26, 1, 0, 0, 1, 1, 0, 1, 0);
}

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
                                                        bool     clip_fp,
                                                        bool     clip_fp_inf_input)
{
    uint32_t a[8] = {fp32_to_tf32(*(float*)&a0, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&a1, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&a2, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&a3, 0, 0, clip_fp, clip_fp_inf_input),
                     0,
                     0,
                     0,
                     0};
    uint32_t b[8] = {fp32_to_tf32(*(float*)&b0, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&b1, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&b2, 0, 0, clip_fp, clip_fp_inf_input),
                     fp32_to_tf32(*(float*)&b3, 0, 0, clip_fp, clip_fp_inf_input),
                     0,
                     0,
                     0,
                     0};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 0, 0, 0, 0, 1, 0);
}

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
                                                        uint8_t  round_mode)
{
    uint32_t a[8] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0)};
    uint32_t b[8] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0)};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 0, 0, 1, 0, 1, 0);
}

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
                                                           uint8_t  round_mode)
{
    uint32_t a[8] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0)};
    uint32_t b[8] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0)};
    return fma_mul_add_tree_double_prec(a, b, c, 8, 26, 0, 1, 0);
}

uint32_t fma_mul_add_tree_bf16_N8_K4_add_C_in_tree_no_ftz_dp(uint16_t a0,
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
                                                             uint32_t c)
{
    uint32_t a[9] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0),
                     c};
    uint32_t b[9] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0),
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t fma_mul_add_tree_bf16_N8_K26_add_C_in_tree_no_ftz(uint16_t a0,
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
                                                           uint32_t c)
{
    uint32_t a[9] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0),
                     c};
    uint32_t b[9] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0),
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 26, 0, 1, 0, 0, 0, 0, 0, 1);
}

uint32_t fma_mul_add_tree_bf16_N8_K26_add_C_in_tree_no_ftz_dp(uint16_t a0,
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
                                                              uint32_t c)
{
    uint32_t a[9] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0),
                     c};
    uint32_t b[9] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0),
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 26, 0, 0, 1);
}

uint32_t fma_mul_add_tree_bf16_N8_K4_add_C_before_norm(uint16_t a0,
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
                                                       uint8_t  round_mode)
{
    uint32_t a[8] = {bf16_to_fp32(a0, 0),
                     bf16_to_fp32(a1, 0),
                     bf16_to_fp32(a2, 0),
                     bf16_to_fp32(a3, 0),
                     bf16_to_fp32(a4, 0),
                     bf16_to_fp32(a5, 0),
                     bf16_to_fp32(a6, 0),
                     bf16_to_fp32(a7, 0)};
    uint32_t b[8] = {bf16_to_fp32(b0, 0),
                     bf16_to_fp32(b1, 0),
                     bf16_to_fp32(b2, 0),
                     bf16_to_fp32(b3, 0),
                     bf16_to_fp32(b4, 0),
                     bf16_to_fp32(b5, 0),
                     bf16_to_fp32(b6, 0),
                     bf16_to_fp32(b7, 0)};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 0, 0, 0, 1, 0, 1, 0);
}

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
                                                        uint8_t  round_mode)
{
    uint32_t a[8] = {fp16_to_fp32(a0, 0),
                     fp16_to_fp32(a1, 0),
                     fp16_to_fp32(a2, 0),
                     fp16_to_fp32(a3, 0),
                     fp16_to_fp32(a4, 0),
                     fp16_to_fp32(a5, 0),
                     fp16_to_fp32(a6, 0),
                     fp16_to_fp32(a7, 0)};
    uint32_t b[8] = {fp16_to_fp32(b0, 0),
                     fp16_to_fp32(b1, 0),
                     fp16_to_fp32(b2, 0),
                     fp16_to_fp32(b3, 0),
                     fp16_to_fp32(b4, 0),
                     fp16_to_fp32(b5, 0),
                     fp16_to_fp32(b6, 0),
                     fp16_to_fp32(b7, 0)};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 0, 0, 0, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp16_N8_K4_add_C_before_norm(uint16_t a0,
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
                                                       uint8_t  round_mode)
{
    uint32_t a[8] = {fp16_to_fp32(a0, 0),
                     fp16_to_fp32(a1, 0),
                     fp16_to_fp32(a2, 0),
                     fp16_to_fp32(a3, 0),
                     fp16_to_fp32(a4, 0),
                     fp16_to_fp32(a5, 0),
                     fp16_to_fp32(a6, 0),
                     fp16_to_fp32(a7, 0)};
    uint32_t b[8] = {fp16_to_fp32(b0, 0),
                     fp16_to_fp32(b1, 0),
                     fp16_to_fp32(b2, 0),
                     fp16_to_fp32(b3, 0),
                     fp16_to_fp32(b4, 0),
                     fp16_to_fp32(b5, 0),
                     fp16_to_fp32(b6, 0),
                     fp16_to_fp32(b7, 0)};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 0, 0, 0, 0, 0, 1, 0);
}

uint32_t
fma_mul_add_tree_fp16_emul_N2_K4_add_C_in_tree_no_ftz(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t c)
{
    uint32_t a[9] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     c};
    uint32_t b[9] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 4, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t
fma_mul_add_tree_fp16_emul_N2_K4_add_C_in_tree_no_ftz_dp(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t c)
{

    uint32_t a0_tf32 = fp16_to_fp32(a0, 0);
    uint32_t a1_tf32 = fp16_to_fp32(a1, 0);
    uint32_t b0_tf32 = fp16_to_fp32(b0, 0);
    uint32_t b1_tf32 = fp16_to_fp32(b1, 0);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t
fma_mul_add_tree_fp16_emul_N2_K26_add_C_in_tree_no_ftz(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t c)
{
    uint32_t a[9] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     c};
    uint32_t b[9] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 26, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp16_emul_N2_K26_add_C_in_tree_no_ftz_dp(uint16_t a0,
                                                                   uint16_t b0,
                                                                   uint16_t a1,
                                                                   uint16_t b1,
                                                                   uint32_t c)
{
    uint32_t a0_tf32 = fp16_to_fp32(a0, 0);
    uint32_t a1_tf32 = fp16_to_fp32(a1, 0);
    uint32_t b0_tf32 = fp16_to_fp32(b0, 0);
    uint32_t b1_tf32 = fp16_to_fp32(b1, 0);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 26, 0, 0, 1);
}

uint32_t fma_mul_add_tree_cfp16_emul_N2_K4_add_C_in_tree_no_ftz(uint16_t a0,
                                                                uint16_t b0,
                                                                uint16_t a1,
                                                                uint16_t b1,
                                                                uint32_t c,
                                                                uint8_t  bias_a,
                                                                uint8_t  bias_b,
                                                                bool     is_unsigned_a,
                                                                bool     is_unsigned_b,
                                                                bool     inf_nan_mode_a,
                                                                bool     inf_nan_mode_b)
{
    uint32_t a[9] = {cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     c};
    uint32_t b[9] = {cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 4, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_cfp16_emul_N2_K4_add_C_in_tree_no_ftz_dp(uint16_t a0,
                                                                   uint16_t b0,
                                                                   uint16_t a1,
                                                                   uint16_t b1,
                                                                   uint32_t c,
                                                                   uint8_t  bias_a,
                                                                   uint8_t  bias_b,
                                                                   bool     is_unsigned_a,
                                                                   bool     is_unsigned_b,
                                                                   bool     inf_nan_mode_a,
                                                                   bool     inf_nan_mode_b)
{

    uint32_t a0_tf32 = cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a);
    uint32_t a1_tf32 = cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a);
    uint32_t b0_tf32 = cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b);
    uint32_t b1_tf32 = cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t fma_mul_add_tree_cfp16_emul_N2_K26_add_C_in_tree_no_ftz(uint16_t a0,
                                                                 uint16_t b0,
                                                                 uint16_t a1,
                                                                 uint16_t b1,
                                                                 uint32_t c,
                                                                 uint8_t  bias_a,
                                                                 uint8_t  bias_b,
                                                                 bool     is_unsigned_a,
                                                                 bool     is_unsigned_b,
                                                                 bool     inf_nan_mode_a,
                                                                 bool     inf_nan_mode_b)
{
    uint32_t a[9] = {cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xff9fe000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a) & 0xffe00000,
                     c};
    uint32_t b[9] = {cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xff9fe000,
                     cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b) & 0xffe00000,
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 26, 1, 1, 0, 1, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_cfp16_emul_N2_K26_add_C_in_tree_no_ftz_dp(uint16_t a0,
                                                                    uint16_t b0,
                                                                    uint16_t a1,
                                                                    uint16_t b1,
                                                                    uint32_t c,
                                                                    uint8_t  bias_a,
                                                                    uint8_t  bias_b,
                                                                    bool     is_unsigned_a,
                                                                    bool     is_unsigned_b,
                                                                    bool     inf_nan_mode_a,
                                                                    bool     inf_nan_mode_b)
{
    uint32_t a0_tf32 = cfp16_to_fp32(a0, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a);
    uint32_t a1_tf32 = cfp16_to_fp32(a1, 5 + is_unsigned_a, 10, bias_a, 0, 0, inf_nan_mode_a);
    uint32_t b0_tf32 = cfp16_to_fp32(b0, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b);
    uint32_t b1_tf32 = cfp16_to_fp32(b1, 5 + is_unsigned_b, 10, bias_b, 0, 0, inf_nan_mode_b);

    uint32_t a0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : a0_tf32 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0_tf32) || is_inf_fp32(b0_tf32) ? 0 : b0_tf32 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0_tf32) ? a0_tf32 : a0_tf32 & 0xff9fe000;
    uint32_t b0_L         = is_inf_fp32(a0_tf32) ? b0_tf32 : b0_tf32 & 0xff9fe000;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));

    uint32_t a1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : a1_tf32 & 0xff800000;
    uint32_t b1_leading_1 = is_inf_fp32(a1_tf32) || is_inf_fp32(b1_tf32) ? 0 : b1_tf32 & 0xff800000;
    uint32_t a1_L         = is_inf_fp32(b1_tf32) ? a1_tf32 : a1_tf32 & 0xff9fe000;
    uint32_t b1_L         = is_inf_fp32(a1_tf32) ? b1_tf32 : b1_tf32 & 0xff9fe000;
    float    a1_L_f       = *(float*)&(a1_L) - (*(float*)&(a1_leading_1));
    float    b1_L_f       = *(float*)&(b1_L) - (*(float*)&(b1_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     a0_tf32 & 0xffe00000,
                     a0_tf32 & 0xffe00000,
                     *(uint32_t*)&a1_L_f,
                     *(uint32_t*)&a1_L_f,
                     a1_tf32 & 0xffe00000,
                     a1_tf32 & 0xffe00000,
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b0_L_f,
                     b0_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     *(uint32_t*)&b1_L_f,
                     b1_tf32 & 0xffe00000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 26, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp16_emul_N2_K4_add_C_before_norm(uint16_t a0,
                                                            uint16_t b0,
                                                            uint16_t a1,
                                                            uint16_t b1,
                                                            uint32_t c,
                                                            uint8_t  round_mode)
{
    uint32_t a[8] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xffe00000};
    uint32_t b[8] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp16_emul_N2_K26_add_C_before_norm(uint16_t a0,
                                                             uint16_t b0,
                                                             uint16_t a1,
                                                             uint16_t b1,
                                                             uint32_t c,
                                                             uint8_t  round_mode)
{
    uint32_t a[8] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xff9fe000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a0, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xff9fe000,
                     fp16_to_fp32(a1, 0) & 0xffe00000,
                     fp16_to_fp32(a1, 0) & 0xffe00000};
    uint32_t b[8] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b0, 0) & 0xff9fe000,
                     fp16_to_fp32(b0, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000,
                     fp16_to_fp32(b1, 0) & 0xff9fe000,
                     fp16_to_fp32(b1, 0) & 0xffe00000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp16_emul_N4_K4_add_C_before_norm(uint16_t a0,
                                                            uint16_t b0,
                                                            uint16_t a1,
                                                            uint16_t b1,
                                                            uint16_t a2,
                                                            uint16_t b2,
                                                            uint16_t a3,
                                                            uint16_t b3,
                                                            uint32_t c,
                                                            uint8_t  round_mode)
{
    uint32_t a[16] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                      fp16_to_fp32(a0, 0) & 0xff9fe000,
                      fp16_to_fp32(a0, 0) & 0xffe00000,
                      fp16_to_fp32(a0, 0) & 0xffe00000,
                      fp16_to_fp32(a1, 0) & 0xff9fe000,
                      fp16_to_fp32(a1, 0) & 0xff9fe000,
                      fp16_to_fp32(a1, 0) & 0xffe00000,
                      fp16_to_fp32(a1, 0) & 0xffe00000,
                      fp16_to_fp32(a2, 0) & 0xff9fe000,
                      fp16_to_fp32(a2, 0) & 0xff9fe000,
                      fp16_to_fp32(a2, 0) & 0xffe00000,
                      fp16_to_fp32(a2, 0) & 0xffe00000,
                      fp16_to_fp32(a3, 0) & 0xff9fe000,
                      fp16_to_fp32(a3, 0) & 0xff9fe000,
                      fp16_to_fp32(a3, 0) & 0xffe00000,
                      fp16_to_fp32(a3, 0) & 0xffe00000};
    uint32_t b[16] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                      fp16_to_fp32(b0, 0) & 0xffe00000,
                      fp16_to_fp32(b0, 0) & 0xff9fe000,
                      fp16_to_fp32(b0, 0) & 0xffe00000,
                      fp16_to_fp32(b1, 0) & 0xff9fe000,
                      fp16_to_fp32(b1, 0) & 0xffe00000,
                      fp16_to_fp32(b1, 0) & 0xff9fe000,
                      fp16_to_fp32(b1, 0) & 0xffe00000,
                      fp16_to_fp32(b2, 0) & 0xff9fe000,
                      fp16_to_fp32(b2, 0) & 0xffe00000,
                      fp16_to_fp32(b2, 0) & 0xff9fe000,
                      fp16_to_fp32(b2, 0) & 0xffe00000,
                      fp16_to_fp32(b3, 0) & 0xff9fe000,
                      fp16_to_fp32(b3, 0) & 0xffe00000,
                      fp16_to_fp32(b3, 0) & 0xff9fe000,
                      fp16_to_fp32(b3, 0) & 0xffe00000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 4, 1, 0, 0, 1, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp16_emul_N4_K26_add_C_before_norm(uint16_t a0,
                                                             uint16_t b0,
                                                             uint16_t a1,
                                                             uint16_t b1,
                                                             uint16_t a2,
                                                             uint16_t b2,
                                                             uint16_t a3,
                                                             uint16_t b3,
                                                             uint32_t c,
                                                             uint8_t  round_mode)
{
    uint32_t a[16] = {fp16_to_fp32(a0, 0) & 0xff9fe000,
                      fp16_to_fp32(a0, 0) & 0xff9fe000,
                      fp16_to_fp32(a0, 0) & 0xffe00000,
                      fp16_to_fp32(a0, 0) & 0xffe00000,
                      fp16_to_fp32(a1, 0) & 0xff9fe000,
                      fp16_to_fp32(a1, 0) & 0xff9fe000,
                      fp16_to_fp32(a1, 0) & 0xffe00000,
                      fp16_to_fp32(a1, 0) & 0xffe00000,
                      fp16_to_fp32(a2, 0) & 0xff9fe000,
                      fp16_to_fp32(a2, 0) & 0xff9fe000,
                      fp16_to_fp32(a2, 0) & 0xffe00000,
                      fp16_to_fp32(a2, 0) & 0xffe00000,
                      fp16_to_fp32(a3, 0) & 0xff9fe000,
                      fp16_to_fp32(a3, 0) & 0xff9fe000,
                      fp16_to_fp32(a3, 0) & 0xffe00000,
                      fp16_to_fp32(a3, 0) & 0xffe00000};
    uint32_t b[16] = {fp16_to_fp32(b0, 0) & 0xff9fe000,
                      fp16_to_fp32(b0, 0) & 0xffe00000,
                      fp16_to_fp32(b0, 0) & 0xff9fe000,
                      fp16_to_fp32(b0, 0) & 0xffe00000,
                      fp16_to_fp32(b1, 0) & 0xff9fe000,
                      fp16_to_fp32(b1, 0) & 0xffe00000,
                      fp16_to_fp32(b1, 0) & 0xff9fe000,
                      fp16_to_fp32(b1, 0) & 0xffe00000,
                      fp16_to_fp32(b2, 0) & 0xff9fe000,
                      fp16_to_fp32(b2, 0) & 0xffe00000,
                      fp16_to_fp32(b2, 0) & 0xff9fe000,
                      fp16_to_fp32(b2, 0) & 0xffe00000,
                      fp16_to_fp32(b3, 0) & 0xff9fe000,
                      fp16_to_fp32(b3, 0) & 0xffe00000,
                      fp16_to_fp32(b3, 0) & 0xff9fe000,
                      fp16_to_fp32(b3, 0) & 0xffe00000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 26, 1, 0, 0, 1, 1, 0, 1, 0);
}

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
                                                             uint8_t  round_mode)
{
    uint32_t a[8] = {a0 & 0xff800fff,
                     a1 & 0xff800fff,
                     a2 & 0xfffff000,
                     a3 & 0xfffff000,
                     a4 & 0xff800fff,
                     a5 & 0xff800fff,
                     a6 & 0xfffff000,
                     a7 & 0xfffff000};
    uint32_t b[8] = {b0 & 0xff800fff,
                     b1 & 0xfffff000,
                     b2 & 0xff800fff,
                     b3 & 0xfffff000,
                     b4 & 0xff800fff,
                     b5 & 0xfffff000,
                     b6 & 0xff800fff,
                     b7 & 0xfffff000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0, 0, 0, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp32_N2_K26_add_C_before_norm_fp32_emul(uint32_t a0,
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
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0, 0, 0, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp32_N2_K26_add_C_before_norm_fp32_emul_dp(uint32_t a0,
                                                                     uint32_t b0,
                                                                     uint32_t a1,
                                                                     uint32_t b1,
                                                                     uint32_t c,
                                                                     uint8_t  round_mode)
{
    uint32_t a[2] = {a0, a1};
    uint32_t b[2] = {b0, b1};
    return fma_mul_add_tree_double_prec(a, b, c, 2, 26, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp32_N1_K4_add_C_in_tree_no_ftz_fp32_emul_dp(uint32_t a0, uint32_t b0, uint32_t c)
{
    uint32_t a0_leading_1 = is_inf_fp32(a0) || is_inf_fp32(b0) ? 0 : a0 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0) || is_inf_fp32(b0) ? 0 : b0 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0) ? a0 : a0 & 0xff8000ff;
    uint32_t a0_M         = is_inf_fp32(b0) ? a0 : a0 & 0xff80ff00;
    uint32_t b0_L         = is_inf_fp32(a0) ? b0 : b0 & 0xff8000ff;
    uint32_t b0_M         = is_inf_fp32(a0) ? b0 : b0 & 0xff80ff00;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    a0_M_f       = *(float*)&(a0_M) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));
    float    b0_M_f       = *(float*)&(b0_M) - (*(float*)&(b0_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_M_f,
                     *(uint32_t*)&a0_M_f,
                     *(uint32_t*)&a0_M_f,
                     (a0 & 0xffff0000),
                     (a0 & 0xffff0000),
                     (a0 & 0xffff0000),
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     *(uint32_t*)&b0_L_f,
                     *(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     *(uint32_t*)&b0_L_f,
                     *(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp32_N1_K26_add_C_in_tree_no_ftz_fp32_emul(uint32_t a0, uint32_t b0, uint32_t c)
{
    uint32_t a[9] = {a0 & 0xff8000ff,
                     a0 & 0xff8000ff,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000,
                     c};
    uint32_t b[9] = {b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     0x3f800000};
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 26, 1, 1, 0, 0, 1, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp32_N1_K26_add_C_in_tree_no_ftz_fp32_emul_dp(uint32_t a0, uint32_t b0, uint32_t c)
{
    uint32_t a0_leading_1 = is_inf_fp32(a0) || is_inf_fp32(b0) ? 0 : a0 & 0xff800000;
    uint32_t b0_leading_1 = is_inf_fp32(a0) || is_inf_fp32(b0) ? 0 : b0 & 0xff800000;
    uint32_t a0_L         = is_inf_fp32(b0) ? a0 : a0 & 0xff8000ff;
    uint32_t a0_M         = is_inf_fp32(b0) ? a0 : a0 & 0xff80ff00;
    uint32_t b0_L         = is_inf_fp32(a0) ? b0 : b0 & 0xff8000ff;
    uint32_t b0_M         = is_inf_fp32(a0) ? b0 : b0 & 0xff80ff00;
    float    a0_L_f       = *(float*)&(a0_L) - (*(float*)&(a0_leading_1));
    float    a0_M_f       = *(float*)&(a0_M) - (*(float*)&(a0_leading_1));
    float    b0_L_f       = *(float*)&(b0_L) - (*(float*)&(b0_leading_1));
    float    b0_M_f       = *(float*)&(b0_M) - (*(float*)&(b0_leading_1));

    uint32_t a[9] = {*(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_L_f,
                     *(uint32_t*)&a0_M_f,
                     *(uint32_t*)&a0_M_f,
                     *(uint32_t*)&a0_M_f,
                     (a0 & 0xffff0000),
                     (a0 & 0xffff0000),
                     (a0 & 0xffff0000),
                     c};
    uint32_t b[9] = {*(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     *(uint32_t*)&b0_L_f,
                     *(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     *(uint32_t*)&b0_L_f,
                     *(uint32_t*)&b0_M_f,
                     b0 & 0xffff0000,
                     0x3f800000};

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 26, 0, 0, 1);
}

uint32_t
fma_mul_add_tree_fp32_N1_K4_add_C_before_norm_fp32_emul(uint32_t a0, uint32_t b0, uint32_t c, uint8_t round_mode)
{
    uint32_t a[8] = {a0 & 0xff8000ff,
                     a0 & 0xff8000ff,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000};
    uint32_t b[8] = {b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 1, 0, 0, 0, 1, 0, 1, 0);
}

uint32_t
fma_mul_add_tree_fp32_N1_K26_add_C_before_norm_fp32_emul(uint32_t a0, uint32_t b0, uint32_t c, uint8_t round_mode)
{
    uint32_t a[8] = {a0 & 0xff8000ff,
                     a0 & 0xff8000ff,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xff80ff00,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000,
                     a0 & 0xffff0000};
    uint32_t b[8] = {b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000,
                     b0 & 0xff8000ff,
                     b0 & 0xff80ff00,
                     b0 & 0xffff0000};
    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 1, 0, 0, 0, 1, 0, 1, 0);
}

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
                                                    uint8_t  round_mode)
{

    uint16_t a_16[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint16_t b_16[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 26, 0, 0, 1, 0, 0, 0, 1, 0);
}

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
                                                       uint8_t  round_mode)
{

    uint16_t a_16[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint16_t b_16[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_double_prec(a, b, c, 16, 26, 1, 1, 0);
}

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
                                                   uint8_t  round_mode)
{

    uint16_t a_16[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint16_t b_16[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 4, 0, 0, 1, 0, 0, 0, 1, 0);
}

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
                                                      uint8_t  round_mode)
{

    uint16_t a_16[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint16_t b_16[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_double_prec(a, b, c, 16, 4, 1, 1, 0);
}

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
                                                        uint8_t  exp_bias_b)
{

    uint8_t a_8[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint8_t b_8[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp8_to_fp32(a_8[i], exp_width, man_width, exp_bias_a, 0, 0, 0);
        b[i] = fp8_to_fp32(b_8[i], exp_width, man_width, exp_bias_b, 0, 0, 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 26, 0, 0, 1, 0, 0, 0, 1, 0);
}

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
                                                       uint8_t  exp_bias_b)
{

    uint8_t a_8[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    uint8_t b_8[16] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15};

    uint32_t a[16];
    uint32_t b[16];

    int i;
    for (i = 0; i < 16; i++) {
        a[i] = fp8_to_fp32(a_8[i], exp_width, man_width, exp_bias_a, 0, 0, 0);
        b[i] = fp8_to_fp32(b_8[i], exp_width, man_width, exp_bias_b, 0, 0, 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 16, 4, 0, 0, 1, 0, 0, 0, 1, 0);
}

uint32_t fma_mul_add_tree_fp8_N8_K4_add_C_in_tree_no_ftz(uint8_t  a0,
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
                                                         uint32_t c,
                                                         uint8_t  exp_width_a,
                                                         uint8_t  exp_width_b,
                                                         uint8_t  man_width_a,
                                                         uint8_t  man_width_b,
                                                         uint8_t  exp_bias_a,
                                                         uint8_t  exp_bias_b,
                                                         uint8_t  inf_nan_mode_a,
                                                         uint8_t  inf_nan_mode_b)
{

    uint8_t a_8[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint8_t b_8[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    uint32_t a[9];
    uint32_t b[9];

    int i;
    for (i = 0; i < 8; i++) {
        a[i] = fp8_to_fp32(a_8[i], exp_width_a, man_width_a, exp_bias_a, 0, 0, inf_nan_mode_a);
        b[i] = fp8_to_fp32(b_8[i], exp_width_b, man_width_b, exp_bias_b, 0, 0, inf_nan_mode_b);
    }
    a[8] = c;
    b[8] = 0x3f800000;
    // fma_mul_add_tree_n(a, b, c, round_mode, N, K, fp32_emul, c_after_norm, denormalize_bias15, fp16_emul, is_gaudi3,
    // is_gaudi2_bug_fix, dnorm_ftz, c_in_tree);
    return fma_mul_add_tree_n(a, b, 0, RND_TO_0, 9, 4, 0, 1, 0, 0, 0, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp8_N8_K4_add_C_in_tree_no_ftz_dp(uint8_t  a0,
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
                                                            uint32_t c,
                                                            uint8_t  exp_width_a,
                                                            uint8_t  exp_width_b,
                                                            uint8_t  man_width_a,
                                                            uint8_t  man_width_b,
                                                            uint8_t  exp_bias_a,
                                                            uint8_t  exp_bias_b,
                                                            uint8_t  inf_nan_mode_a,
                                                            uint8_t  inf_nan_mode_b)
{

    uint8_t a_8[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint8_t b_8[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    uint32_t a[9];
    uint32_t b[9];

    int i;
    for (i = 0; i < 8; i++) {
        a[i] = fp8_to_fp32(a_8[i], exp_width_a, man_width_a, exp_bias_a, 0, 0, inf_nan_mode_a);
        b[i] = fp8_to_fp32(b_8[i], exp_width_b, man_width_b, exp_bias_b, 0, 0, inf_nan_mode_b);
    }
    a[8] = c;
    b[8] = 0x3f800000;

    // fma_mul_add_tree_double_prec(a, b, c, N, K, is_bias15, dnorm_ftz, c_in_tree)
    return fma_mul_add_tree_double_prec(a, b, 0, 9, 4, 0, 0, 1);
}

uint32_t fma_mul_add_tree_fp8_N8_K4_add_C_before_norm(uint8_t  a0,
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
                                                      uint32_t c,
                                                      uint8_t  round_mode,
                                                      uint8_t  exp_width_a,
                                                      uint8_t  exp_width_b,
                                                      uint8_t  man_width_a,
                                                      uint8_t  man_width_b,
                                                      uint8_t  exp_bias_a,
                                                      uint8_t  exp_bias_b,
                                                      uint8_t  inf_nan_mode_a,
                                                      uint8_t  inf_nan_mode_b)
{

    uint8_t a_8[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint8_t b_8[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    uint32_t a[8];
    uint32_t b[8];

    int i;
    for (i = 0; i < 8; i++) {
        a[i] = fp8_to_fp32(a_8[i], exp_width_a, man_width_a, exp_bias_a, 0, 0, inf_nan_mode_a);
        b[i] = fp8_to_fp32(b_8[i], exp_width_b, man_width_b, exp_bias_b, 0, 0, inf_nan_mode_b);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 0, 1, 0, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_N8_K4_add_C_before_norm(uint16_t a0,
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
                                                  uint8_t  round_mode)
{

    uint16_t a_16[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint16_t b_16[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    uint32_t a[8];
    uint32_t b[8];

    int i;
    for (i = 0; i < 8; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 4, 0, 0, 1, 0, 1, 0, 1, 0);
}

uint32_t fma_mul_add_tree_N8_K26_add_C_before_norm(uint16_t a0,
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
                                                   uint8_t  round_mode)
{

    uint16_t a_16[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    uint16_t b_16[8] = {b0, b1, b2, b3, b4, b5, b6, b7};

    uint32_t a[8];
    uint32_t b[8];

    int i;
    for (i = 0; i < 8; i++) {
        a[i] = fp16_to_fp32(a_16[i], 0);
        b[i] = fp16_to_fp32(b_16[i], 0);
    }

    return fma_mul_add_tree_n(a, b, c, round_mode, 8, 26, 0, 0, 1, 0, 1, 0, 1, 0);
}

uint32_t dp4_ref_double_c_in_tree_no_ftz(uint32_t a0,
                                         uint32_t b0,
                                         uint32_t a1,
                                         uint32_t b1,
                                         uint32_t a2,
                                         uint32_t b2,
                                         uint32_t a3,
                                         uint32_t b3,
                                         uint32_t c)
{
    set_rounding_mode(RND_TO_0);
    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double a2_d = (double)(*(float*)&a2);
    double b2_d = (double)(*(float*)&b2);
    double a3_d = (double)(*(float*)&a3);
    double b3_d = (double)(*(float*)&b3);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    double a2b2_d = a2_d * b2_d;
    double a3b3_d = a3_d * b3_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    double a2b2a3b3_d = a2b2_d + a3b3_d;

    // add sticky in last bit and perform last addition with proper rounding
    double result_d = a0b0a1b1_d + a2b2a3b3_d;
    result_d        = result_d + c_d;
    // printf("result %e \n", result_d);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    result = (float)result_d;
    // printf("result float %08x \n", *(uint32_t*)&result);

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

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
                        bool     c_after_norm)
{
    // initialize round_mode
    /*
        set_rounding_mode(round_mode);
    */
    set_rounding_mode(RND_TO_0);

    // zero denormals
    //#if DNORM_FTZ
    if (is_denorm_fp32(a0))
        a0 = ibs(a0, 30, 0, 0);
    if (is_denorm_fp32(b0))
        b0 = ibs(b0, 30, 0, 0);
    if (is_denorm_fp32(a1))
        a1 = ibs(a1, 30, 0, 0);
    if (is_denorm_fp32(b1))
        b1 = ibs(b1, 30, 0, 0);
    if (is_denorm_fp32(a2))
        a2 = ibs(a2, 30, 0, 0);
    if (is_denorm_fp32(b2))
        b2 = ibs(b2, 30, 0, 0);
    if (is_denorm_fp32(a3))
        a3 = ibs(a3, 30, 0, 0);
    if (is_denorm_fp32(b3))
        b3 = ibs(b3, 30, 0, 0);
    if (is_denorm_fp32(c))
        c = ibs(c, 30, 0, 0);
    //#endif
    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double a2_d = (double)(*(float*)&a2);
    double b2_d = (double)(*(float*)&b2);
    double a3_d = (double)(*(float*)&a3);
    double b3_d = (double)(*(float*)&b3);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    double a2b2_d = a2_d * b2_d;
    double a3b3_d = a3_d * b3_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    double a2b2a3b3_d = a2b2_d + a3b3_d;

    // add sticky in last bit and perform last addition with proper rounding
    double result_d = a0b0a1b1_d + a2b2a3b3_d;
    // printf("result %e \n", result_d);
    // initialize round_mode
    set_rounding_mode(round_mode);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    if (c_after_norm) {
        result = (float)result_d;

        //#if DNORM_FTZ
        if (is_denorm_fp32(*res_32bit))
            *res_32bit = ibs(*res_32bit, 30, 0, 0);
        //#endif
        // printf("res_tree %08x\n", *(uint32_t*)&result);
        result = result + *(float*)&c;
    } else {
        result_d = result_d + c_d;
        result   = (float)result_d;
    }
    // printf("result float %08x \n", *(uint32_t*)&result);
    // zero denormals

    //#if DNORM_FTZ
    if (is_denorm_fp32(*res_32bit))
        *res_32bit = ibs(*res_32bit, 30, 0, 0);
    //#endif

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}
uint32_t dp8_ref_double_c_in_tree_no_ftz(uint32_t a0,
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
                                         uint32_t c)
{
    set_rounding_mode(RND_TO_0);

    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double a2_d = (double)(*(float*)&a2);
    double b2_d = (double)(*(float*)&b2);
    double a3_d = (double)(*(float*)&a3);
    double b3_d = (double)(*(float*)&b3);
    double a4_d = (double)(*(float*)&a4);
    double b4_d = (double)(*(float*)&b4);
    double a5_d = (double)(*(float*)&a5);
    double b5_d = (double)(*(float*)&b5);
    double a6_d = (double)(*(float*)&a6);
    double b6_d = (double)(*(float*)&b6);
    double a7_d = (double)(*(float*)&a7);
    double b7_d = (double)(*(float*)&b7);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    double a2b2_d = a2_d * b2_d;
    double a3b3_d = a3_d * b3_d;
    double a4b4_d = a4_d * b4_d;
    double a5b5_d = a5_d * b5_d;
    double a6b6_d = a6_d * b6_d;
    double a7b7_d = a7_d * b7_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    double a2b2a3b3_d = a2b2_d + a3b3_d;
    double a4b4a5b5_d = a4b4_d + a5b5_d;
    double a6b6a7b7_d = a6b6_d + a7b7_d;

    double ab0123_d = a0b0a1b1_d + a2b2a3b3_d;
    double ab4567_d = a4b4a5b5_d + a6b6a7b7_d;
    // add sticky in last bit and perform last addition with proper rounding
    double result_d = ab0123_d + ab4567_d;
    result_d        = result_d + c_d;

    // printf("result %e \n", result_d);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    result = (float)result_d;

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

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
                        bool     c_after_norm)
{
    // initialize round_mode
    /*
        set_rounding_mode(round_mode);
    */
    set_rounding_mode(RND_TO_0);

    // zero denormals
    //#if DNORM_FTZ
    if (is_denorm_fp32(a0))
        a0 = ibs(a0, 30, 0, 0);
    if (is_denorm_fp32(b0))
        b0 = ibs(b0, 30, 0, 0);
    if (is_denorm_fp32(a1))
        a1 = ibs(a1, 30, 0, 0);
    if (is_denorm_fp32(b1))
        b1 = ibs(b1, 30, 0, 0);
    if (is_denorm_fp32(a2))
        a2 = ibs(a2, 30, 0, 0);
    if (is_denorm_fp32(b2))
        b2 = ibs(b2, 30, 0, 0);
    if (is_denorm_fp32(a3))
        a3 = ibs(a3, 30, 0, 0);
    if (is_denorm_fp32(b3))
        b3 = ibs(b3, 30, 0, 0);
    if (is_denorm_fp32(a4))
        a4 = ibs(a4, 30, 0, 0);
    if (is_denorm_fp32(b4))
        b4 = ibs(b4, 30, 0, 0);
    if (is_denorm_fp32(a5))
        a5 = ibs(a5, 30, 0, 0);
    if (is_denorm_fp32(b5))
        b5 = ibs(b5, 30, 0, 0);
    if (is_denorm_fp32(a6))
        a6 = ibs(a6, 30, 0, 0);
    if (is_denorm_fp32(b6))
        b6 = ibs(b6, 30, 0, 0);
    if (is_denorm_fp32(a7))
        a7 = ibs(a7, 30, 0, 0);
    if (is_denorm_fp32(b7))
        b7 = ibs(b7, 30, 0, 0);
    if (is_denorm_fp32(c))
        c = ibs(c, 30, 0, 0);
    //#endif
    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double a2_d = (double)(*(float*)&a2);
    double b2_d = (double)(*(float*)&b2);
    double a3_d = (double)(*(float*)&a3);
    double b3_d = (double)(*(float*)&b3);
    double a4_d = (double)(*(float*)&a4);
    double b4_d = (double)(*(float*)&b4);
    double a5_d = (double)(*(float*)&a5);
    double b5_d = (double)(*(float*)&b5);
    double a6_d = (double)(*(float*)&a6);
    double b6_d = (double)(*(float*)&b6);
    double a7_d = (double)(*(float*)&a7);
    double b7_d = (double)(*(float*)&b7);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    double a2b2_d = a2_d * b2_d;
    double a3b3_d = a3_d * b3_d;
    double a4b4_d = a4_d * b4_d;
    double a5b5_d = a5_d * b5_d;
    double a6b6_d = a6_d * b6_d;
    double a7b7_d = a7_d * b7_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    double a2b2a3b3_d = a2b2_d + a3b3_d;
    double a4b4a5b5_d = a4b4_d + a5b5_d;
    double a6b6a7b7_d = a6b6_d + a7b7_d;

    double ab0123_d = a0b0a1b1_d + a2b2a3b3_d;
    double ab4567_d = a4b4a5b5_d + a6b6a7b7_d;
    // add sticky in last bit and perform last addition with proper rounding
    double result_d = ab0123_d + ab4567_d;

    // printf("result %e \n", result_d);
    // initialize round_mode

    set_rounding_mode(round_mode);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    if (c_after_norm) {
        result = (float)result_d;

        //#if DNORM_FTZ
        if (is_denorm_fp32(*res_32bit))
            *res_32bit = ibs(*res_32bit, 30, 0, 0);
        //#endif

        result = result + *(float*)&c;
    } else {
        result_d = result_d + c_d;
        result   = (float)result_d;
    }
    // printf("result float %08x \n", *(uint32_t*)&result);
    // zero denormals

    //#if DNORM_FTZ
    if (is_denorm_fp32(*res_32bit))
        *res_32bit = ibs(*res_32bit, 30, 0, 0);
    //#endif

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

uint32_t dp4_fma_float_ref(uint32_t a0,
                           uint32_t b0,
                           uint32_t a1,
                           uint32_t b1,
                           uint32_t a2,
                           uint32_t b2,
                           uint32_t a3,
                           uint32_t b3,
                           uint32_t acc,
                           uint8_t  round_mode)
{
    // initialize round_mode
    set_rounding_mode(round_mode);

    // zero denormals
    //#if DNORM_FTZ
    if (is_denorm_fp32(a0))
        a0 = ibs(a0, 30, 0, 0);
    if (is_denorm_fp32(b0))
        b0 = ibs(b0, 30, 0, 0);
    if (is_denorm_fp32(a1))
        a1 = ibs(a1, 30, 0, 0);
    if (is_denorm_fp32(b1))
        b1 = ibs(b1, 30, 0, 0);
    if (is_denorm_fp32(a2))
        a2 = ibs(a2, 30, 0, 0);
    if (is_denorm_fp32(b2))
        b2 = ibs(b2, 30, 0, 0);
    if (is_denorm_fp32(a3))
        a3 = ibs(a3, 30, 0, 0);
    if (is_denorm_fp32(b3))
        b3 = ibs(b3, 30, 0, 0);
    if (is_denorm_fp32(acc))
        acc = ibs(acc, 30, 0, 0);
    //#endif
    float res_f;
    res_f = fmaf(*(float*)&a0, *(float*)&b0, *(float*)&acc);
    res_f = fmaf(*(float*)&a1, *(float*)&b1, res_f);
    res_f = fmaf(*(float*)&a2, *(float*)&b2, res_f);
    res_f = fmaf(*(float*)&a3, *(float*)&b3, res_f);

    uint32_t* res_32bit;

    res_32bit = (uint32_t*)&res_f;
    // zero denormals

    //#if DNORM_FTZ
    if (is_denorm_fp32(*res_32bit))
        *res_32bit = ibs(*res_32bit, 30, 0, 0);
    //#endif

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

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
                           uint8_t  round_mode)
{
    // initialize round_mode
    set_rounding_mode(round_mode);

    // zero denormals
    //#if DNORM_FTZ
    if (is_denorm_fp32(a0))
        a0 = ibs(a0, 30, 0, 0);
    if (is_denorm_fp32(b0))
        b0 = ibs(b0, 30, 0, 0);
    if (is_denorm_fp32(a1))
        a1 = ibs(a1, 30, 0, 0);
    if (is_denorm_fp32(b1))
        b1 = ibs(b1, 30, 0, 0);
    if (is_denorm_fp32(a2))
        a2 = ibs(a2, 30, 0, 0);
    if (is_denorm_fp32(b2))
        b2 = ibs(b2, 30, 0, 0);
    if (is_denorm_fp32(a3))
        a3 = ibs(a3, 30, 0, 0);
    if (is_denorm_fp32(b3))
        b3 = ibs(b3, 30, 0, 0);
    if (is_denorm_fp32(a4))
        a4 = ibs(a4, 30, 0, 0);
    if (is_denorm_fp32(b4))
        b4 = ibs(b4, 30, 0, 0);
    if (is_denorm_fp32(a5))
        a5 = ibs(a5, 30, 0, 0);
    if (is_denorm_fp32(b5))
        b5 = ibs(b5, 30, 0, 0);
    if (is_denorm_fp32(a6))
        a6 = ibs(a6, 30, 0, 0);
    if (is_denorm_fp32(b6))
        b6 = ibs(b6, 30, 0, 0);
    if (is_denorm_fp32(a7))
        a7 = ibs(a7, 30, 0, 0);
    if (is_denorm_fp32(b7))
        b7 = ibs(b7, 30, 0, 0);
    if (is_denorm_fp32(acc))
        acc = ibs(acc, 30, 0, 0);
    //#endif
    float res_f;
    res_f = fmaf(*(float*)&a0, *(float*)&b0, *(float*)&acc);
    res_f = fmaf(*(float*)&a1, *(float*)&b1, res_f);
    res_f = fmaf(*(float*)&a2, *(float*)&b2, res_f);
    res_f = fmaf(*(float*)&a3, *(float*)&b3, res_f);
    res_f = fmaf(*(float*)&a4, *(float*)&b4, res_f);
    res_f = fmaf(*(float*)&a5, *(float*)&b5, res_f);
    res_f = fmaf(*(float*)&a6, *(float*)&b6, res_f);
    res_f = fmaf(*(float*)&a7, *(float*)&b7, res_f);

    uint32_t* res_32bit;

    res_32bit = (uint32_t*)&res_f;
    // zero denormals

    //#if DNORM_FTZ
    if (is_denorm_fp32(*res_32bit))
        *res_32bit = ibs(*res_32bit, 30, 0, 0);
    //#endif

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

uint32_t dp2_ref_double_c_in_tree_no_ftz(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1, uint32_t c)
{
    // initialize round_mode
    set_rounding_mode(RND_TO_0);

    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    double result_d   = a0b0a1b1_d + c_d;
    // printf("result %e \n", result_d);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    result = (float)result_d;

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

uint32_t
dp2_ref_double(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1, uint32_t c, uint8_t round_mode, bool c_after_norm)
{
    // initialize round_mode
    set_rounding_mode(RND_TO_0);

    // zero denormals
    //#if DNORM_FTZ
    if (is_denorm_fp32(a0))
        a0 = ibs(a0, 30, 0, 0);
    if (is_denorm_fp32(b0))
        b0 = ibs(b0, 30, 0, 0);
    if (is_denorm_fp32(a1))
        a1 = ibs(a1, 30, 0, 0);
    if (is_denorm_fp32(b1))
        b1 = ibs(b1, 30, 0, 0);
    if (is_denorm_fp32(c))
        c = ibs(c, 30, 0, 0);
    //#endif
    // printf("inputs: %08x %08x %08x %08x %08x %08x %08x %08x \n",a0, a1, b0, b1, a2, b2, a3, b3);

    double a0_d = (double)(*(float*)&a0);
    double b0_d = (double)(*(float*)&b0);
    double a1_d = (double)(*(float*)&a1);
    double b1_d = (double)(*(float*)&b1);
    double c_d  = (double)(*(float*)&c);
    // printf("inputs: %e %e %e %e %e %e %e %e \n",a0_d, b0_d, a1_d, b1_d, a2_d, b2_d, a3_d, b3_d);
    double a0b0_d = a0_d * b0_d;
    double a1b1_d = a1_d * b1_d;
    // printf("mult products: %e %e %e %e \n",a0b0_d, a1b1_d, a2b2_d, a3b3_d);

    double a0b0a1b1_d = a0b0_d + a1b1_d;
    // add sticky in last bit and perform last addition with proper rounding
    double result_d = a0b0a1b1_d;
    // printf("result %e \n", result_d);
    // initialize round_mode
    set_rounding_mode(round_mode);

    float     result;
    uint32_t* res_32bit;
    res_32bit = (uint32_t*)&result;

    if (c_after_norm) {
        result = (float)result_d;

        //#if DNORM_FTZ
        if (is_denorm_fp32(*res_32bit))
            *res_32bit = ibs(*res_32bit, 30, 0, 0);
        //#endif
        // printf("res_tree %08x\n", *(uint32_t*)&result);
        result = result + *(float*)&c;
    } else {
        result_d = result_d + c_d;
        result   = (float)result_d;
    }
    // printf("result float %08x \n", *(uint32_t*)&result);
    // zero denormals

    //#if DNORM_FTZ
    if (is_denorm_fp32(*res_32bit))
        *res_32bit = ibs(*res_32bit, 30, 0, 0);
    //#endif

    // default nan
    if (is_nan_fp32(*res_32bit))
        *res_32bit = DEFAULT_NAN_FP32;

    return (*res_32bit);
}

// FP8

uint32_t fp8_to_fp32(uint8_t input,
                     uint8_t exp_width,
                     uint8_t man_width,
                     uint8_t exp_bias,
                     bool    clip_fp,
                     bool    is_fp19,
                     uint8_t inf_nan_mode)
{
    const uint8_t inputUint = input;
    uint32_t      outputUint; // = (uint32_t *)output;
    int32_t       exponent_offset_fp8 = man_width;
    int32_t       sign_offset_fp8     = 7;

    int32_t inputMantissa = sbs(inputUint, man_width - 1, 0);
    int32_t inputExponent = sbs(inputUint, 6, exponent_offset_fp8);
    int32_t inputSign     = sbs(inputUint, sign_offset_fp8, sign_offset_fp8);

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;

    if (fp8_is_zero(inputUint)) {
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if ((fp8_is_nan(inputUint, exponent_offset_fp8) && (inf_nan_mode == 0)) ||
               ((inputUint == 0x7f || inputUint == 0xff) && (inf_nan_mode == 2))) {
        outputExponent = 0xFF;
        if (is_fp19)
            outputMantissa = 0x000003FF;
        else
            outputMantissa = 0x007FFFFF;
        outputSign = 0;
    } else if (fp8_is_infinity(inputUint, exponent_offset_fp8) && (inf_nan_mode == 0)) {
        outputExponent = 0xFF;
        outputMantissa = 0x0;
    } else {
        outputExponent                = inputExponent - exp_bias + EXPONENT_BIAS_FP32;
        int32_t mantissaForAdjustment = inputMantissa;
        if (fp8_is_denormal(inputUint, exponent_offset_fp8)) {
            int shift = lzcnt(exponent_offset_fp8, inputMantissa);
            // Shift leading 1 (normalize) and fixup the exponent accordingly
            mantissaForAdjustment = sbs((inputMantissa << (shift + 1)), man_width - 1, 0);
            outputExponent -= shift;
        }
        // Normal case
        if (is_fp19)
            outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP19 - exponent_offset_fp8);
        else
            outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - exponent_offset_fp8);
    }

    if (is_fp19)
        outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP19 | outputSign << SIGN_OFFSET_FP19;
    else
        outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

    if (clip_fp) {
        if (is_fp19) // convert to FP32
            outputUint = outputUint << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP19);

        if (is_inf_fp32(outputUint)) {
            //	outputMantissa = 0x7FFFFF; //all ones
            //	outputExponent = 0xFE; //max_exp-1
            //	outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

            outputUint = outputUint - 1; // will return +/-max_norm_value
        }

        if (is_fp19) // convert back to FP19
            outputUint = outputUint >> (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP19);
    }

    return outputUint;
}

uint16_t
fp8_to_fp16(uint8_t input, uint8_t exp_width, uint8_t man_width, uint8_t exp_bias, bool clip_fp, uint8_t inf_nan_mode)
{
    uint32_t fp32_res = fp8_to_fp32(input, exp_width, man_width, exp_bias, clip_fp, 0, inf_nan_mode);
    uint16_t fp16_res = fp32_to_fp16(*(float*)&fp32_res, 0, 0, clip_fp, 0, 1);
    return fp16_res;
}

uint32_t fma_fp8_fp32(uint8_t  a,
                      uint8_t  b,
                      uint32_t c,
                      uint8_t  round_mode,
                      uint8_t  exp_width,
                      uint8_t  man_width,
                      uint8_t  exp_bias,
                      bool     fp8_ftz_in,
                      bool     fp32_dnorm_ftz,
                      uint8_t  inf_nan_mode)
{
    uint32_t* res;

    uint32_t a_32bit;
    uint32_t b_32bit;

    if (fp8_ftz_in) {
        // flush denormals to zero
        //-----------------------

        // flush denormal inputs to zero
        if (fp8_is_denormal(a, man_width))
            a = ibs(a, 6, 0, 0); // a=+-0

        if (fp8_is_denormal(b, man_width))
            b = ibs(b, 6, 0, 0); // b=+-0
        //------------------------
    }

    a_32bit = fp8_to_fp32(a, exp_width, man_width, exp_bias, 0, 0, inf_nan_mode);
    b_32bit = fp8_to_fp32(b, exp_width, man_width, exp_bias, 0, 0, inf_nan_mode);

    float* a_f;
    float* b_f;
    float* c_f;
    float  res_f;

    set_rounding_mode(round_mode);

    if ((round_mode == RND_TO_NE) || (round_mode == RND_TO_0)) {

        if (fp32_dnorm_ftz)
        {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        }

        a_f = (float*)(&a_32bit);
        b_f = (float*)(&b_32bit);

        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);
        if (fp32_dnorm_ftz)
        {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        }
    } else {

        if (fp32_dnorm_ftz)
        {
            // flush denormals to zero
            //-----------------------

            if (is_denorm_fp32(c))
                c = ibs(c, 30, 0, 0); // c=+-0
            //------------------------
        }

        a_f = (float*)(&a_32bit);
        b_f = (float*)(&b_32bit);
        c_f = (float*)(&c);

        res_f = fmaf(*a_f, *b_f, *c_f);

        res = (uint32_t*)(&res_f);

        if (fp32_dnorm_ftz)
        {
            // flush denormal output to zero
            if (is_denorm_fp32(*res)) {
                // flush result to 0
                if ((round_mode == RND_TO_PINF) && (sbs(*res, 31, 31) == 0))
                    *res = 0x00800000; // res=+min_normal
                else if ((round_mode == RND_TO_NINF) && (sbs(*res, 31, 31) == 1))
                    *res = 0x80800000; // res=-min_normal
                else
                    *res = ibs(*res, 30, 0, 0); // res=+-0
            }
        }
    }

    if (is_nan_fp32(*res))
        *res = DEFAULT_NAN_FP32; // indefinite nan value

    return *res;
}

uint32_t fma_2xfp8_fp32(uint8_t  a0,
                        uint8_t  b0,
                        uint8_t  a1,
                        uint8_t  b1,
                        uint32_t c,
                        uint8_t  round_mode,
                        uint8_t  exp_width,
                        uint8_t  man_width,
                        uint8_t  exp_bias)
{

    set_rounding_mode(round_mode);

    float a0_f;
    float b0_f;
    float a1_f;
    float b1_f;

    uint32_t a0_32bit = fp8_to_fp32(a0, exp_width, man_width, exp_bias, 0, 0, 0);
    uint32_t b0_32bit = fp8_to_fp32(b0, exp_width, man_width, exp_bias, 0, 0, 0);
    uint32_t a1_32bit = fp8_to_fp32(a1, exp_width, man_width, exp_bias, 0, 0, 0);
    uint32_t b1_32bit = fp8_to_fp32(b1, exp_width, man_width, exp_bias, 0, 0, 0);

    a0_f = *(float*)&a0_32bit;
    b0_f = *(float*)&b0_32bit;
    a1_f = *(float*)&a1_32bit;
    b1_f = *(float*)&b1_32bit;

    float c_f;
    float a0b0_f;
    float a1b1_f;

    c_f = *(float*)(&c);
    // printf("a0 %02x b0 %02x a1 %02x b1 %02x\n", a0, b0, a1, b1);

    a0b0_f = fmaf(a0_f, b0_f, 0);
    a1b1_f = fmaf(a1_f, b1_f, 0);

    uint32_t a0b0 = *(uint32_t*)&a0b0_f;
    uint32_t a1b1 = *(uint32_t*)&a1b1_f;
    // printf("a0b0 %08x a1b1 %08x\n", a0b0, a1b1);
    // printf("a0b0_f %f a1b1_f %f c_f %f\n", a0b0_f, a1b1_f, c_f);
    uint32_t result        = 0;
    bool     result_ready  = 0;
    uint8_t  a0b0_is_dnorm = is_denorm_fp32(a0b0); // not supposed to happen
    uint8_t  a1b1_is_dnorm = is_denorm_fp32(a1b1); // not supposed to happen
    uint8_t  c_is_dnorm    = is_denorm_fp32(c);

    //#if DNORM_FTZ
    // flush denormals to zero
    //-----------------------

    // flush denormal inputs to zero
    if (is_denorm_fp32(a0b0))
        a0b0 = ibs(a0b0, 30, 0, 0); // a0b0=+-0

    if (is_denorm_fp32(a1b1))
        a1b1 = ibs(a1b1, 30, 0, 0); // a1b1=+-0

    if (is_denorm_fp32(c))
        c = ibs(c, 30, 0, 0); // c=+-0
                              //------------------------
    //#endif

    uint16_t a0b0_exp = sbs(a0b0, 30, 23);
    uint16_t a1b1_exp = sbs(a1b1, 30, 23);
    uint16_t c_exp    = sbs(c, 30, 23);
    uint32_t a0b0_man = sbs(a0b0, 22, 0);
    uint32_t a1b1_man = sbs(a1b1, 22, 0);
    uint32_t c_man    = sbs(c, 22, 0);
    uint16_t a0b0_sgn = sbs(a0b0, 31, 31);
    uint16_t a1b1_sgn = sbs(a1b1, 31, 31);
    uint16_t c_sgn    = sbs(c, 31, 31);

    int16_t a0b0_exp_unbiased;
    int16_t a1b1_exp_unbiased;
    int16_t c_exp_unbiased;

    // uint8_t sub_ind = (fabsf(a0b0_f) >= fabsf(a1b1_f)) ? a0b0_sgn^c_sgn : a1b1_sgn^c_sgn; //??
    uint8_t a0b0_is_zero = is_zero_fp32(a0b0);
    uint8_t a1b1_is_zero = is_zero_fp32(a1b1);
    uint8_t c_is_zero    = is_zero_fp32(c);
    uint8_t leading_1_ind;

    if (!a0b0_is_dnorm && !a0b0_is_zero) {
        if (!a0b0_is_zero)
            a0b0_man = cbs(1, a0b0_man, 23);
        a0b0_exp_unbiased = a0b0_exp - 127;
    } else {
        a0b0_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(a0b0_man);
        if (leading_1_ind <= 22) {
            a0b0_man          = a0b0_man << (23 - leading_1_ind);
            a0b0_exp_unbiased = a0b0_exp_unbiased - (23 - leading_1_ind);
        }
    }

    if (!a1b1_is_dnorm && !a1b1_is_zero) {
        if (!a1b1_is_zero)
            a1b1_man = cbs(1, a1b1_man, 23);
        a1b1_exp_unbiased = a1b1_exp - 127;
    } else {
        a1b1_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(a1b1_man);
        if (leading_1_ind <= 22) {
            a1b1_man          = a1b1_man << (23 - leading_1_ind);
            a1b1_exp_unbiased = a1b1_exp_unbiased - (23 - leading_1_ind);
        }
    }

    if (!c_is_dnorm && !c_is_zero) {
        if (!c_is_zero)
            c_man = cbs(1, c_man, 23);
        c_exp_unbiased = c_exp - 127;
    } else {
        c_exp_unbiased = -126;
        // normalize
        leading_1_ind = lzd(c_man);
        if (leading_1_ind <= 22) {
            c_man          = c_man << (23 - leading_1_ind);
            c_exp_unbiased = c_exp_unbiased - (23 - leading_1_ind);
        }
    }
    // printf("exp unbiased %d %d %d\n", a0b0_exp_unbiased, a1b1_exp_unbiased, c_exp_unbiased);
    uint8_t res_sgn;

    uint8_t a0b0_is_inf = is_inf_fp32(a0b0);
    uint8_t a1b1_is_inf = is_inf_fp32(a1b1);
    uint8_t c_is_inf    = is_inf_fp32(c);
    uint8_t a0b0_is_nan = is_nan_fp32(a0b0);
    uint8_t a1b1_is_nan = is_nan_fp32(a1b1);
    uint8_t c_is_nan    = is_nan_fp32(c);
    // printf("nan %d %d %d\n",a0b0_is_nan, a1b1_is_nan, c_is_nan);
    // exceptions
    //  Both sources are inf with (different signs and add instruction) or (same signs and sub instruction). i.e.
    //  inf-inf
    uint8_t res_is_def_nan = (a0b0_is_inf & a1b1_is_inf & (a0b0_sgn ^ a1b1_sgn)) |
                             (a0b0_is_inf & c_is_inf & (a0b0_sgn ^ c_sgn)) |
                             (a1b1_is_inf & c_is_inf & (a1b1_sgn ^ c_sgn));

    // nan handling - if any input is nan or result is nan - return nan
    uint8_t res_is_nan = res_is_def_nan | c_is_nan | a0b0_is_nan | a1b1_is_nan;

    // add res is inf - if one of the sources is infinity and it's add instruction OR one of the sources is infinity -
    // select the exception result
    uint8_t res_is_inf = !res_is_nan && (a0b0_is_inf | a1b1_is_inf | c_is_inf);

    // special results - bypass calculated result
    if (res_is_inf) {
        if (a0b0_is_inf)
            res_sgn = a0b0_sgn;
        else if (a1b1_is_inf)
            res_sgn = a1b1_sgn;
        else // c_is_inf
            res_sgn = c_sgn;

        result       = cbs(res_sgn, 0x7F800000, 31); //+-inf
        result_ready = 1;
    }
    if (res_is_nan && (result_ready == 0)) {
        result       = DEFAULT_NAN_FP32; // indefinite nan value
        result_ready = 1;
    }

    uint8_t sign_zero_exception;
    if (!c_is_zero && a0b0_is_zero && a1b1_is_zero)
        sign_zero_exception = c_sgn;
    else if (c_is_zero && !a0b0_is_zero && a1b1_is_zero)
        sign_zero_exception = a0b0_sgn;
    else if (c_is_zero && a0b0_is_zero && !a1b1_is_zero)
        sign_zero_exception = a1b1_sgn;
    else
        sign_zero_exception = (round_mode == RND_TO_NINF) && (c_sgn | a0b0_sgn | a1b1_sgn);

    sign_zero_exception = sign_zero_exception || (c_sgn && a0b0_sgn && a1b1_sgn);

    uint8_t c_sgn_zero = (c_is_zero && a0b0_is_zero && a1b1_is_zero) ? sign_zero_exception : c_sgn;

    c = c_is_zero ? cbs(c_sgn_zero, 0x00000000, 31) : c;

    // if 2 operands are 0, then the result is the third operand
    if (((a0b0_is_zero && a1b1_is_zero) || (fabsf(a0b0_f) == fabsf(a1b1_f) && a0b0_sgn != a1b1_sgn)) &&
        (result_ready == 0)) {
        result = c;
        // x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        if (result == 0 && (round_mode == RND_TO_NINF) && (fabsf(a0b0_f) == fabsf(a1b1_f) && a0b0_sgn != a1b1_sgn))
            result = 0x80000000;
        result_ready = 1;
    }
    if (((a0b0_is_zero && c_is_zero) || (fabsf(a0b0_f) == fabsf(c_f) && a0b0_sgn != c_sgn)) && (result_ready == 0)) {
        result = a1b1;
        // x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        if (result == 0 && (round_mode == RND_TO_NINF) && (fabsf(a0b0_f) == fabsf(c_f) && a0b0_sgn != c_sgn))
            result = 0x80000000;
        result_ready = 1;
    }
    if (((a1b1_is_zero && c_is_zero) || (fabsf(a1b1_f) == fabsf(c_f) && a1b1_sgn != c_sgn)) && (result_ready == 0)) {
        result = a0b0;
        // x+(-x)=+0 for all rounding modes except RND_TO_NINF, x+(-x)=-0 for RND_TO_NINF
        if (result == 0 && (round_mode == RND_TO_NINF) && (fabsf(a1b1_f) == fabsf(c_f) && a1b1_sgn != c_sgn))
            result = 0x80000000;
        result_ready = 1;
    }
    // printf("result_ready %d\n",result_ready);

    int64_t a0b0_man_ll = a0b0_man;
    int64_t a1b1_man_ll = a1b1_man;
    int64_t c_man_ll    = c_man;
    int64_t res_man_ll;
    // align c_man_ll to ab_man (regardless of exponents)
    a0b0_man_ll = a0b0_man_ll << 37ll;
    a1b1_man_ll = a1b1_man_ll << 37ll;
    c_man_ll    = c_man_ll << 37ll;
    // printf("a0b0 = %016llx, a1b1 = %016llx, c = %016llx\n",a0b0_man_ll, a1b1_man_ll, c_man_ll);
    // Sticky bits
    // uint32_t c_s = 0;
    // uint32_t a0b0_s = 0;
    // uint32_t a1b1_s = 0;

    int16_t res_exp_unbiased;

    // find max exp
    uint8_t c_is_largest    = (fabsf(c_f) >= fabsf(a0b0_f) && fabsf(c_f) >= fabsf(a1b1_f));
    uint8_t a0b0_is_largest = (fabsf(a0b0_f) >= fabsf(c_f) && fabsf(a0b0_f) >= fabsf(a1b1_f));
    uint8_t a1b1_is_largest = (fabsf(a1b1_f) >= fabsf(c_f) && fabsf(a1b1_f) >= fabsf(a0b0_f));
    uint8_t c_second_largest =
        (a0b0_is_largest && (fabsf(c_f) >= fabsf(a1b1_f))) || (a1b1_is_largest && (fabsf(c_f) >= fabsf(a0b0_f)));
    uint8_t a0b0_second_largest =
        (c_is_largest && (fabsf(a0b0_f) >= fabsf(a1b1_f))) || (a1b1_is_largest && (fabsf(a0b0_f) >= fabsf(c_f)));
    // uint8_t a1b1_second_largest = (c_is_largest && (fabsf(a1b1_f) >= fabsf(a0b0_f))) ||
    //							  (a0b0_is_largest && (fabsf(a1b1_f) >= fabsf(c_f)));
    int16_t max_exp_unbiased = c_is_largest ? c_exp_unbiased : a0b0_is_largest ? a0b0_exp_unbiased : a1b1_exp_unbiased;
    // printf("max exp unbiased %d\n", max_exp_unbiased);
    float max_num_f = c_is_largest ? c_f : a0b0_is_largest ? a0b0_f : a1b1_f;

    int64_t exp_diff_a0b0         = max_exp_unbiased - a0b0_exp_unbiased;
    int64_t exp_diff_a1b1         = max_exp_unbiased - a1b1_exp_unbiased;
    int64_t exp_diff_c            = max_exp_unbiased - c_exp_unbiased;
    int64_t exp_diff_a0b0_clipped = exp_diff_a0b0;
    int64_t exp_diff_a1b1_clipped = exp_diff_a1b1;
    int64_t exp_diff_c_clipped    = exp_diff_c;
    res_exp_unbiased              = max_exp_unbiased;
    // printf("largest %d %d %d second %d %d %d\n",a0b0_is_largest, a1b1_is_largest, c_is_largest, a0b0_second_largest,
    // a1b1_second_largest, c_second_largest); printf("exp_diff_a0b0 %d, exp_diff_a1b1 %d, exp_diff_c
    // %d\n",exp_diff_a0b0,exp_diff_a1b1,exp_diff_c);
    uint32_t max_num_sgn = sbs(*(uint32_t*)&max_num_f, 31, 31);

    if (a0b0_f != max_num_f && (a0b0_sgn != max_num_sgn)) // if a0b0 smaller than max and has a different sign
        a0b0_man_ll = -a0b0_man_ll;
    if (a1b1_f != max_num_f && (a1b1_sgn != max_num_sgn)) // if a1b1 smaller than max and has a different sign
        a1b1_man_ll = -a1b1_man_ll;
    if (c_f != max_num_f && (c_sgn != max_num_sgn)) // if c smaller than max and has a different sign
        c_man_ll = -c_man_ll;
    // printf("a0b0 = %016llx, a1b1 = %016llx, c = %016llx (after sign adjustement)\n",a0b0_man_ll, a1b1_man_ll,
    // c_man_ll);

    int64_t max_shift = 37; // sign, overflow bit, 2, 8, 2, 24, 2, 24
    if (exp_diff_a0b0 > 0) {
        if ((exp_diff_a0b0 > max_shift) &&
            ((exp_diff_a1b1 > max_shift) ||
             (exp_diff_c > max_shift))) // larger exp_diff means that we are discarding the entire ab_man
            exp_diff_a0b0_clipped = max_shift;
        if (exp_diff_a0b0 > 63)
            exp_diff_a0b0_clipped = 63;
        // a0b0_s = sbs(a0b0_man_ll>>39, exp_diff_a0b0 - 1, 0);
        a0b0_man_ll = a0b0_man_ll >> exp_diff_a0b0_clipped;
    }
    if (exp_diff_a1b1 > 0) {
        if ((exp_diff_a1b1 > max_shift) &&
            ((exp_diff_a0b0 > max_shift) ||
             (exp_diff_c > max_shift))) // larger exp_diff means that we are discarding the entire ab_man
            exp_diff_a1b1_clipped = max_shift;
        if (exp_diff_a1b1 > 63)
            exp_diff_a1b1_clipped = 63;
        // a1b1_s = (sbs(a1b1_man_ll>>39, exp_diff_a1b1 - 1, 0) != 0);
        a1b1_man_ll = a1b1_man_ll >> exp_diff_a1b1_clipped;
    }
    if (exp_diff_c > 0) {
        if ((exp_diff_c > max_shift) &&
            ((exp_diff_a0b0 > max_shift) ||
             (exp_diff_a1b1 > max_shift))) // larger exp_diff means that we are discarding the entire ab_man
            exp_diff_c_clipped = max_shift;
        if (exp_diff_c > 63)
            exp_diff_c_clipped = 63;
        // c_s = sbs(c_man_ll>>39, exp_diff_c - 1, 0);
        c_man_ll = c_man_ll >> exp_diff_c_clipped;
    }
    // uint16_t res_s = a0b0_s | a1b1_s | c_s;
    // printf("a0b0 = %016llx, a1b1 = %016llx, c = %016llx (after shift)\n",a0b0_man_ll, a1b1_man_ll, c_man_ll);

    uint32_t res_man = 0;
    // int32_t man_diff;
    uint32_t res_s = 0;

    // calculate sticky_bit
    if (c_is_largest) {
        if ((exp_diff_a0b0 > max_shift) || (exp_diff_a1b1 > max_shift)) {
            if (a0b0_second_largest)
                a1b1_man_ll = 0; // a1b1_man_ll >> (exp_diff_a1b1 - exp_diff_a0b0);
            else
                a0b0_man_ll = 0; // a0b0_man_ll >> (exp_diff_a0b0 - exp_diff_a1b1);
        }
    } else if (a0b0_is_largest) {
        if ((exp_diff_a1b1 > max_shift) || (exp_diff_c > max_shift)) {
            if (c_second_largest)
                a1b1_man_ll = 0; // a1b1_man_ll >> (exp_diff_a1b1 - exp_diff_c);
            else
                c_man_ll = 0; // c_man_ll >> (exp_diff_c - exp_diff_a1b1);
        }
    } else if (a1b1_is_largest) {
        if ((exp_diff_a0b0 > max_shift) || (exp_diff_c > max_shift)) {
            if (c_second_largest)
                a0b0_man_ll = 0; // a0b0_man_ll >> (exp_diff_a0b0 - exp_diff_c);
            else
                c_man_ll = 0; // c_man_ll >> (exp_diff_c - exp_diff_a0b0);
        }
    }
    // printf("a0b0 = %016llx, a1b1 = %016llx, c = %016llx (after second shift)\n",a0b0_man_ll, a1b1_man_ll, c_man_ll);

    res_man_ll = a0b0_man_ll + a1b1_man_ll + c_man_ll;
    res_sgn    = max_num_sgn; // TODO: fix if we have x+(-x)
    // printf("res_man_ll = %016llx, res_man = %08x res_exp_unbiased %x\n",res_man_ll, res_man, res_exp_unbiased);
    if (res_man_ll < 0) {
        res_man_ll = -res_man_ll;
        res_sgn    = 1 - res_sgn;
    }
    // printf("res_man_ll = %016llx, res_man = %08x res_exp_unbiased %x\n",res_man_ll, res_man, res_exp_unbiased);
    // find leading 1
    leading_1_ind = lzd_ll(res_man_ll);
    // printf("leading_1_ind %d\n", leading_1_ind);
    // normalize
    if (leading_1_ind == 62) {
        res_s            = (res_man_ll & ((1ll << (62 - 24 - 2 + 1)) - 1)) != 0;
        res_man_ll       = res_man_ll >> (62 - 24 - 2 + 1);
        res_man          = res_man_ll;
        res_exp_unbiased = res_exp_unbiased + 2;

        // res_s = res_s | (sbs(res_man, 1, 0) != 0);
        // res_man = res_man >> 2;
        // res_exp_unbiased = res_exp_unbiased + 2;
    } else if (leading_1_ind == 61) {
        res_s            = (res_man_ll & ((1ll << (61 - 24 - 2 + 1)) - 1)) != 0;
        res_man_ll       = res_man_ll >> (61 - 24 - 2 + 1);
        res_man          = res_man_ll;
        res_exp_unbiased = res_exp_unbiased + 1;

        // res_s = res_s | sbs(res_man, 0, 0);
        // res_man = res_man >> 1;
        // res_exp_unbiased = res_exp_unbiased + 1;
    } else if (leading_1_ind <= 60) {
        res_s            = (res_man_ll & ((1ll << (leading_1_ind - 24 - 2 + 1)) - 1)) != 0;
        res_man_ll       = res_man_ll >> (leading_1_ind - 24 - 2 + 1);
        res_man          = res_man_ll;
        res_exp_unbiased = res_exp_unbiased - (60 - leading_1_ind);

        // res_s = res_s;
        // res_man = res_man << (25 - leading_1_ind);
        // res_exp_unbiased = res_exp_unbiased - (25 - leading_1_ind);
    } else {
        // res_man is 0
        if (res_s == 0) // result is 0, and grs are 0, no need for rounding
            res_exp_unbiased = -127;
        else
            res_exp_unbiased = res_exp_unbiased - 28;
    }

    // printf("res_man_ll = %016llx, res_man = %08x res_exp_unbiased %x\n",res_man_ll, res_man, res_exp_unbiased);
    // printf("res_s %d \n",res_s);

    // separate GR bits from mantissa
    uint8_t res_g = sbs(res_man, 1, 1);
    uint8_t res_r = sbs(res_man, 0, 0);
    // printf("res_man = %08x, res_g %d, res_r %d, res_s %d \n",res_man, res_g, res_r, res_s);

    // align mantissa to the right (dispose of 7 extra fraction bits from mult + 1 G bit + 1 R bit)
    res_man = res_man >> 2;

    // round
    uint8_t need_rnd;

    // detect underflow
    int16_t shift_val;

    if ((res_exp_unbiased < -127) || (res_exp_unbiased == -127 && res_man != 0)) {
        // denormalize small values
        shift_val        = -126 - res_exp_unbiased;
        res_exp_unbiased = -127;

        if (shift_val > 30) // 24
            shift_val = 30;
        if (shift_val == 1) {
            res_s = res_s | res_r;
            res_r = res_g;
            res_g = sbs(res_man, 0, 0);
        } else if (shift_val == 2) {
            // fix Sticky bit
            res_s = res_s | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        } else // shift_val>=3
        {
            // fix Sticky bit
            res_s = res_s | (sbs(res_man, shift_val - 3, 0) != 0) | res_g | res_r;

            res_g = sbs(res_man, shift_val - 1, shift_val - 1);
            res_r = sbs(res_man, shift_val - 2, shift_val - 2);
        }

        res_man  = res_man >> shift_val;
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        res_man = res_man + need_rnd;

        if (sbs(res_man, 23, 23) == 1)
            res_exp_unbiased = res_exp_unbiased + 1;
    } else {
        need_rnd = (((round_mode == RND_TO_PINF) & (res_s | res_r | res_g) & (res_sgn == 0)) |
                    ((round_mode == RND_TO_NINF) & (res_s | res_r | res_g) & (res_sgn == 1)) |
                    ((round_mode == RND_TO_NE) &
                     ((res_g & (res_r | res_s)) | (res_g & !res_r & !res_s & (sbs(res_man, 0, 0) == 1)))) |
                    ((round_mode == RND_HALF_AZ) & res_g));

        // printf("need_rnd %d res_man %08x mode %d g %d r %d s %d lsb %d\n", need_rnd, res_man, round_mode, res_g,
        // res_r, res_s, sbs(res_man,0,0));
        res_man = res_man + need_rnd;
        // printf("need_rnd %d res_man %08x mode %d\n", need_rnd, res_man, round_mode);

        if (sbs(res_man, 24, 24) == 1) {
            res_s            = res_s | res_r;
            res_r            = res_g;
            res_g            = sbs(res_man, 0, 0);
            res_man          = res_man >> 1;
            res_exp_unbiased = res_exp_unbiased + 1;
        }

        // detect overflow
        if (res_exp_unbiased >= 128) {
            // result is +inf or -inf
            if ((round_mode == RND_TO_0) || (round_mode == RND_TO_PINF && res_sgn == 1) ||
                (round_mode == RND_TO_NINF && res_sgn == 0)) {
                // largest value which is not inf
                res_exp_unbiased = 127;
                res_man          = 0x7FFFFF;
            } else {
                // inf
                res_exp_unbiased = 128;
                res_man          = 0;
            }
        }
    }

    if (result_ready == 0) {
        // construct result

        result = ibs(result, 31, 31, res_sgn);
        result = ibs(result, 30, 23, (res_exp_unbiased + 127));
        result = ibs(result, 22, 0, res_man);

        //#if DNORM_FTZ
        // flush result_16bit if denormal
        if (is_denorm_fp32(result)) {
            // flush result to 0
            if ((round_mode == RND_TO_PINF) && (sbs(result, 31, 31) == 0))
                result = 0x00800000; // res=+min_normal
            else if ((round_mode == RND_TO_NINF) && (sbs(result, 31, 31) == 1))
                result = 0x80800000; // res=-min_normal
            else
                result = ibs(result, 30, 0, 0); // res=+-0
        }
        //#endif
    }
    return result;
}

uint8_t fp32_to_fp8(float    input,
                    uint8_t  exp_width,
                    uint8_t  man_width,
                    uint8_t  exp_bias,
                    int      roundingMode,
                    uint32_t lfsrVal,
                    bool     ftz_fp8,
                    bool     clip_fp,
                    bool     clip_fp_inf_input,
                    bool     stochastic_ftz,
                    uint8_t  inf_nan_mode,
                    uint8_t* inf_exception,
                    uint8_t* overflow_exception,
                    uint8_t* nan_exception)
{
    // inf_nan_mode:
    // 0 - legacy_fp8
    // 1 - fp8_no_inf_nan (Tesla)
    // 2 - fp8_nvidia
    // 3 - reserved

    uint8_t  output;
    uint8_t  tmp_output;
    int      inputExponent, inputSign, unbiasedExp = 0;
    uint32_t inputMantissa;
    bool     roundedMSB = 0, roundedLSBs = 0;
    // int minExp = -25;
    int     minNormExp          = 1 - exp_bias; //-14
    int     maxExp = ((1 << exp_width) - 1) - exp_bias - 1 + (inf_nan_mode != 0); // 15, if inf_nan_mode!=0 - inc by 1
    int     minExp              = minNormExp - man_width - 1; //-25 //min denormal value can come from rounding of 0.5
    int32_t exponent_offset_fp8 = man_width;
    int32_t sign_offset_fp8     = 7;
    int     roundingModeDnorm   = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_NEAREST_EVEN : roundingMode;
    int     roundingModeNorm    = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_STOCHASTIC : roundingMode;

    uint8_t is_inf      = 0;
    uint8_t is_overflow = 0;
    uint8_t is_nan      = 0;

    const uint32_t inputUint = *(const uint32_t*)&input;

    inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
    inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
    inputSign     = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

    int32_t  outputExponent = 0;
    int32_t  outputMantissa = 0;
    int32_t outputSign = inputSign;
    int      rc_bit_idx;
    int32_t  shift_val;
    bool     stochastic_ftz_decision = 0;
    uint32_t discardedAlignedLeft    = 0;
    if (is_nan_fp32(inputUint)) {
        is_nan = 1;
        if (inf_nan_mode == 0) // regular FP8*
        {
            // return the same NAN always (0x7F)
            outputSign     = 0x0;
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        }
        if (inf_nan_mode == 1) // no_inf_nan
        {
            // return max_norm_value - equal to our default NaN
            outputSign     = 0x0;
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        }
        if (inf_nan_mode == 2) // Nvidia
        {
            // nan is 0x7f/0xff
            // outputSign - same as input sign - FIXME - always negative NaN?
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        }
    } else if (is_zero_fp32(inputUint)) {
        // return +-0
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (is_inf_fp32(inputUint)) {
        is_inf = 1;
        if (inf_nan_mode == 0) // regular FP8*
        {
            // return +-infinity
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = 0x0;
        }
        if (inf_nan_mode == 1) { // no_inf_nan
            // +-max_norm_value
            // outputSign - same as input sign
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        }
        if (inf_nan_mode == 2) // Nvidia
        {
            //+/-inf are mapped to nan (0x7f/0xff)
            // outputSign - same as input sign
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        }
    } else {
        // Valid number
        unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
        inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

        if (unbiasedExp > maxExp) {
            is_overflow = 1;
            if (inf_nan_mode == 0) {
                if ((roundingMode == (VPE_RM_TO_0)) || (inputSign && (roundingMode == (VPE_RM_INF))) ||
                    (!inputSign && (roundingMode == (VPE_RM_NINF))))

                { // +- max_normal
                    outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
                    outputExponent = maxExp + exp_bias;
                } else { // +-infinity
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = 0x0;
                }
            }
            if (inf_nan_mode == 1) // no_inf_nan
            {
                // +-max_norm_value
                // outputSign - same as input sign
                outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
            }
            if (inf_nan_mode == 2) { // Nvidia
                if ((roundingMode == (VPE_RM_TO_0)) || (inputSign && (roundingMode == (VPE_RM_INF))) ||
                    (!inputSign && (roundingMode == (VPE_RM_NINF))))

                { // +- max_normal - mantissa has 0 in LSB!
                    outputMantissa = sbs(0xfe, man_width - 1, 0); // 0x2;
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                } else { // +-inf (= +/-nan)
                    outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                }
            }
        } else if (unbiasedExp < minExp) {
            // The result will be either 0 or 0x1
            roundedMSB  = 0;
            roundedLSBs = 1;
            discardedAlignedLeft = 0;
            if (stochastic_ftz) {
                rc_bit_idx = (EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1) + (minNormExp - unbiasedExp);
                if (ftz_fp8)
                    rc_bit_idx = rc_bit_idx + exponent_offset_fp8;
                shift_val = 31 - rc_bit_idx;
                if (shift_val >= 0)
                    discardedAlignedLeft = inputMantissa << shift_val;
                else if (shift_val >= -24)
                    discardedAlignedLeft = inputMantissa >> (-shift_val);
                else
                    discardedAlignedLeft = 0;

                if (ftz_fp8)
                    stochastic_ftz_decision =
                        (discardedAlignedLeft >= lfsrVal) && (roundingMode == RND_TO_NE || roundingMode == RND_SR);
            }
            outputMantissa = fp_accommodate_rounding(
                0,
                roundedMSB,
                roundedLSBs,
                inputSign,
                (stochastic_ftz && (roundingMode == RND_TO_NE || roundingMode == RND_SR)) ? RND_SR : roundingModeDnorm,
                lfsrVal,
                discardedAlignedLeft);

            outputExponent = 0x0;

        } else { // minExp <= unbiasedExp <= maxExp
            outputExponent = unbiasedExp;
            rc_bit_idx     = (unbiasedExp < minNormExp)
                             ? ((EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1) + (minNormExp - unbiasedExp))
                             : (EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1);
            if ((unbiasedExp < minNormExp) && ftz_fp8 && stochastic_ftz)
                rc_bit_idx = rc_bit_idx + exponent_offset_fp8;
            shift_val                     = 31 - rc_bit_idx;
            roundedMSB                    = (((inputMantissa >> rc_bit_idx)) & 0x1) != 0;
            roundedLSBs                   = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
            discardedAlignedLeft          = inputMantissa << shift_val;
            outputMantissa                = inputMantissa >> (rc_bit_idx + 1);
            if ((unbiasedExp < minNormExp) && ftz_fp8 && stochastic_ftz)
                stochastic_ftz_decision =
                    (discardedAlignedLeft >= lfsrVal) && (roundingMode == RND_TO_NE || roundingMode == RND_SR);
            if (unbiasedExp < minNormExp) {
                outputMantissa = fp_accommodate_rounding(outputMantissa,
                                                         roundedMSB,
                                                         roundedLSBs,
                                                         inputSign,
                                                         roundingModeDnorm,
                                                         lfsrVal,
                                                         discardedAlignedLeft);
            } else
                outputMantissa = fp_accommodate_rounding(outputMantissa,
                                                         roundedMSB,
                                                         roundedLSBs,
                                                         inputSign,
                                                         roundingModeNorm,
                                                         lfsrVal,
                                                         discardedAlignedLeft);
            if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << exponent_offset_fp8))) ||
                (outputMantissa & (1 << (exponent_offset_fp8 + 1)))) { // Should handle two cases:
                // 1. The number was denormal, and after rounding became normal
                // 2. The number was rounded to the 1.0 * 2^(next exponent)
                outputExponent = outputExponent + 1;
            }
            if (outputExponent > maxExp) {
                is_overflow = 1;
                if (inf_nan_mode == 0) {
                    // return infinity
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = 0x0;
                }
                if (inf_nan_mode == 1) { // no_inf_nan
                    // +-max_norm_value
                    // outputSign - same as input sign
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
                }
                if (inf_nan_mode == 2) { // Nvidia
                    // +/-inf = +/-nan = 0x7f/0xff
                    // outputSign - same as input sign
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
                }
            } else {
                if (outputExponent < minNormExp) {
                    outputExponent = 0x0;
                } else {
                    outputExponent += exp_bias;
                }
                // normalize - leave man_width bits
                outputMantissa = sbs(outputMantissa, man_width - 1, 0);
            }

            tmp_output = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
            if (inf_nan_mode == 2) {
                if (tmp_output == 0x7f || tmp_output == 0xff)
                    is_overflow = 1;
                // can't reach +/-inf if rounding in the opposite direction (similar to clip_fp)
                if ((roundingMode == (VPE_RM_TO_0)) || (inputSign && (roundingMode == (VPE_RM_INF))) ||
                    (!inputSign && (roundingMode == (VPE_RM_NINF)))) {
                    if (tmp_output == 0x7f || tmp_output == 0xff)
                        outputMantissa = outputMantissa - 1;
                }
            }
        }
    }
    output = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);

    if (clip_fp && fp8_is_infinity(output, exponent_offset_fp8) && (clip_fp_inf_input || !is_inf_fp32(inputUint)) &&
        (inf_nan_mode == 0)) {
        // outputExponent = sbs(0xff, exp_width - 1, 0) - 1;//exponent is max_exp-1 (all ones minus 1)
        // outputMantissa = sbs(0xff, man_width - 1, 0); //mantissa is all ones
        // output = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
        output = output - 1; // will return +/-max_norm_value
    }

    if (clip_fp && (output == 0x7f || output == 0xff) && (clip_fp_inf_input || !is_inf_fp32(inputUint)) &&
        !is_nan_fp32(inputUint) && (inf_nan_mode == 2)) {
        // outputExponent = sbs(0xff, exp_width - 1, 0) - 1;//exponent is max_exp-1 (all ones minus 1)
        // outputMantissa = sbs(0xff, man_width - 1, 0); //mantissa is all ones
        // output = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
        output = output - 1; // will return +/-max_norm_value
    }

    if (ftz_fp8 && (fp8_is_denormal(output, exponent_offset_fp8) || (stochastic_ftz && stochastic_ftz_decision))) {
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(output, 7, 7) == 0)) {
            // output=+min_normal
            outputSign     = 0;
            outputExponent = 1;
            outputMantissa = 0;
            output         = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
        } else if ((roundingMode == RND_TO_NINF) && (sbs(output, 7, 7) == 1)) {
            // output=+min_normal
            outputSign     = 1;
            outputExponent = 1;
            outputMantissa = 0;
            output         = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
        } else {
            outputExponent = (stochastic_ftz && stochastic_ftz_decision);
            outputMantissa = 0;
            output         = outputMantissa | (outputExponent << exponent_offset_fp8) | (outputSign << sign_offset_fp8);
            // output = ibs(output, 6, 0, 0); // output=+-0
        }
    }
    if (inf_exception != NULL)
        *inf_exception = is_inf;
    if (nan_exception != NULL)
        *nan_exception = is_nan;
    if (overflow_exception != NULL)
        *overflow_exception = is_overflow;

    return output;
}

//////////////////////////until here FP8

// CFP16

uint16_t fp32_to_cfp16(float    input,
                       uint8_t  exp_width,
                       uint8_t  man_width,
                       uint8_t  exp_bias,
                       int      roundingMode,
                       uint32_t lfsrVal,
                       bool     ftz_fp16,
                       bool     clip_fp,
                       bool     clip_fp_inf_input,
                       bool     no_inf_nan,
                       uint8_t* inf_exception,
                       uint8_t* overflow_exception,
                       uint8_t* nan_exception)
{
    uint16_t output;
    int      inputExponent, inputSign, unbiasedExp = 0;
    uint32_t inputMantissa;
    bool     roundedMSB = 0, roundedLSBs = 0;
    // int minExp = -25;
    int     minNormExp = 1 - exp_bias; //-14
    int     maxExp     = ((1 << exp_width) - 1) - exp_bias - 1 + (no_inf_nan == 1); // 15, if no_inf_nan - inc by 1
    int     minExp     = minNormExp - man_width - 1; //-25 //min denormal value can come from rounding of 0.5
    int32_t exponent_offset_fp = man_width;
    int32_t sign_offset_fp     = man_width + exp_width;
    bool    is_unsigned        = (sign_offset_fp == 16);
    int     roundingModeDnorm  = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_NEAREST_EVEN : roundingMode;
    int     roundingModeNorm   = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_STOCHASTIC : roundingMode;

    uint8_t is_inf      = 0;
    uint8_t is_overflow = 0;
    uint8_t is_nan      = 0;

    const uint32_t inputUint = *(const uint32_t*)&input;

    inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
    inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
    inputSign     = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;
    int     rc_bit_idx;
    int32_t shift_val;

    uint32_t discardedAlignedLeft = 0;
    if (is_nan_fp32(inputUint)) {
        is_nan = 1;
        if (no_inf_nan == 0) // regular FP16
        {
            // return the same NAN always (0x7FFF)
            outputSign     = 0x0;
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
        } else {
            // return max_norm_value - equal to our default NaN
            outputSign     = 0x0;
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
        }
    } else if (is_zero_fp32(inputUint) || (is_unsigned && sbs(inputUint, 31, 31))) {
        if (is_inf_fp32(inputUint)) // in case of unsigned and -inf
            is_inf = 1;
        if ((is_unsigned && sbs(inputUint, 31, 31)) &&
            !is_zero_fp32(inputUint)) // negative number which is not nan or zero - overflow for unsigned
            is_overflow = 1;
        // return +-0
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (is_inf_fp32(inputUint)) {
        is_inf = 1;
        if (no_inf_nan == 0) // regular FP16*
        {
            // return +-infinity
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = 0x0;
        } else {
            // +-max_norm_value
            // outputSign - same as input sign
            outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
            outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
        }
    } else {
        // Valid number
        unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
        inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

        if (unbiasedExp > maxExp) {
            is_overflow = 1;
            if (no_inf_nan == 0) {
                if ((roundingMode == (VPE_RM_TO_0)) || (inputSign && (roundingMode == (VPE_RM_INF))) ||
                    (!inputSign && (roundingMode == (VPE_RM_NINF))))

                { // +- max_normal
                    outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
                    outputExponent = maxExp + exp_bias;
                } else { // +-infinity
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = 0x0;
                }
            } else {
                // +-max_norm_value
                // outputSign - same as input sign
                outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
            }
        } else if (unbiasedExp < minExp) {
            // The result will be either 0 or 0x1
            roundedMSB  = 0;
            roundedLSBs = 1;
            outputMantissa =
                fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, roundingModeDnorm, lfsrVal, 0);

            outputExponent = 0x0;
        } else { // minExp <= unbiasedExp <= maxExp
            outputExponent = unbiasedExp;
            rc_bit_idx     = (unbiasedExp < minNormExp)
                             ? ((EXPONENT_OFFSET_FP32 - exponent_offset_fp - 1) + (minNormExp - unbiasedExp))
                             : (EXPONENT_OFFSET_FP32 - exponent_offset_fp - 1);
            // if ((unbiasedExp < minNormExp) && ftz_fp16)
            // rc_bit_idx = rc_bit_idx + exponent_offset_fp;
            shift_val            = 31 - rc_bit_idx;
            roundedMSB           = (((inputMantissa >> rc_bit_idx)) & 0x1) != 0;
            roundedLSBs          = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
            discardedAlignedLeft = inputMantissa << shift_val;
            outputMantissa       = inputMantissa >> (rc_bit_idx + 1);
            if (unbiasedExp < minNormExp) {
                outputMantissa = fp_accommodate_rounding(outputMantissa,
                                                         roundedMSB,
                                                         roundedLSBs,
                                                         inputSign,
                                                         roundingModeDnorm,
                                                         lfsrVal,
                                                         discardedAlignedLeft);
            } else
                outputMantissa = fp_accommodate_rounding(outputMantissa,
                                                         roundedMSB,
                                                         roundedLSBs,
                                                         inputSign,
                                                         roundingModeNorm,
                                                         lfsrVal,
                                                         discardedAlignedLeft);
            if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << exponent_offset_fp))) ||
                (outputMantissa & (1 << (exponent_offset_fp + 1)))) { // Should handle two cases:
                // 1. The number was denormal, and after rounding became normal
                // 2. The number was rounded to the 1.0 * 2^(next exponent)
                outputExponent = outputExponent + 1;
            }
            if (outputExponent > maxExp) {
                is_overflow = 1;
                if (no_inf_nan == 0) {
                    // return infinity
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = 0x0;
                } else {
                    // +-max_norm_value
                    // outputSign - same as input sign
                    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
                    outputMantissa = sbs(0xffff, man_width - 1, 0); // 0x3FF;
                }
            } else {
                if (outputExponent < minNormExp) {
                    outputExponent = 0x0;
                } else {
                    outputExponent += exp_bias;
                }
                // normalize - leave man_width bits
                outputMantissa = sbs(outputMantissa, man_width - 1, 0);
            }
        }
    }
    output = outputMantissa | (outputExponent << exponent_offset_fp) | (outputSign << sign_offset_fp);

    if (clip_fp && cfp16_is_infinity(output, exponent_offset_fp, is_unsigned) &&
        (clip_fp_inf_input || !is_inf_fp32(inputUint)) && (no_inf_nan == 0)) {
        output = output - 1; // will return +/-max_norm_value
    }

    if (ftz_fp16 && (cfp16_is_denormal(output, exponent_offset_fp, is_unsigned))) { // TODO: test for unsigned
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(output, man_width + exp_width, man_width + exp_width) == 0)) {
            // output=+min_normal
            outputSign     = 0;
            outputExponent = 1;
            outputMantissa = 0;
            output         = outputMantissa | (outputExponent << exponent_offset_fp) | (outputSign << sign_offset_fp);
        } else if ((roundingMode == RND_TO_NINF) && (sbs(output, man_width + exp_width, man_width + exp_width) == 1)) {
            // output=+min_normal
            outputSign     = 1;
            outputExponent = 1;
            outputMantissa = 0;
            output         = outputMantissa | (outputExponent << exponent_offset_fp) | (outputSign << sign_offset_fp);
        } else {
            output = ibs(output, man_width + exp_width - 1, 0, 0); // output=+-0
        }
    }

    if (inf_exception != NULL)
        *inf_exception = is_inf;
    if (nan_exception != NULL)
        *nan_exception = is_nan;
    if (overflow_exception != NULL)
        *overflow_exception = is_overflow;

    return output;
}

uint32_t cfp16_to_fp32(uint16_t input,
                       uint8_t  exp_width,
                       uint8_t  man_width,
                       uint8_t  exp_bias,
                       bool     clip_fp,
                       bool     is_fp19,
                       bool     no_inf_nan)
{
    const uint16_t inputUint = input;
    uint32_t       outputUint; // = (uint32_t *)output;
    int32_t        exponent_offset_fp = man_width;
    int32_t        sign_offset_fp     = man_width + exp_width;
    bool           is_unsigned        = (sign_offset_fp == 16);

    int32_t inputMantissa = sbs(inputUint, man_width - 1, 0);
    int32_t inputExponent = sbs(inputUint, exponent_offset_fp + exp_width - 1, exponent_offset_fp);
    int32_t inputSign     = (sign_offset_fp < 16) ? sbs(inputUint, sign_offset_fp, sign_offset_fp) : 0;

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;

    if (cfp16_is_zero(inputUint, is_unsigned)) {
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (cfp16_is_nan(inputUint, exponent_offset_fp, is_unsigned) && (no_inf_nan == 0)) {
        outputExponent = 0xFF;
        if (is_fp19)
            outputMantissa = 0x000003FF;
        else
            outputMantissa = 0x007FFFFF;
        outputSign = 0;
    } else if (cfp16_is_infinity(inputUint, exponent_offset_fp, is_unsigned) && (no_inf_nan == 0)) {
        outputExponent = 0xFF;
        outputMantissa = 0x0;
    } else {
        outputExponent                = inputExponent - exp_bias + EXPONENT_BIAS_FP32;
        int32_t mantissaForAdjustment = inputMantissa;
        if (cfp16_is_denormal(inputUint, exponent_offset_fp, is_unsigned)) {
            int shift = lzcnt(exponent_offset_fp, inputMantissa);
            // Shift leading 1 (normalize) and fixup the exponent accordingly
            mantissaForAdjustment = sbs((inputMantissa << (shift + 1)), man_width - 1, 0);
            outputExponent -= shift;
        }
        // Normal case
        if (is_fp19)
            outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP19 - exponent_offset_fp);
        else
            outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - exponent_offset_fp);
    }

    if (is_fp19)
        outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP19 | outputSign << SIGN_OFFSET_FP19;
    else
        outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

    if (clip_fp) {
        if (is_fp19) // convert to FP32
            outputUint = outputUint << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP19);

        if (is_inf_fp32(outputUint)) {
            //	outputMantissa = 0x7FFFFF; //all ones
            //	outputExponent = 0xFE; //max_exp-1
            //	outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

            outputUint = outputUint - 1; // will return +/-max_norm_value
        }

        if (is_fp19) // convert back to FP19
            outputUint = outputUint >> (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP19);
    }

    return outputUint;
}

////////////////////////until here CFP16
uint8_t fp16_to_fp8_143_bias15(uint16_t input,
                               int      roundingMode,
                               uint32_t sr_register,
                               bool     clip_fp,
                               bool     ftz_fp8,
                               bool     clip_fp_inf_input,
                               uint8_t* inf_exception,
                               uint8_t* overflow_exception,
                               uint8_t* nan_exception)
{

    int roundingModeDnorm    = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_NEAREST_EVEN : roundingMode;
    int roundingModeNorm     = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_STOCHASTIC : roundingMode;
    int roundingModeSelected = is_denorm_fp16(input) ? roundingModeDnorm : roundingModeNorm;

    uint16_t res         = 0;
    uint8_t  is_inf      = 0;
    uint8_t  is_overflow = 0;
    uint8_t  is_nan      = 0;

    if (is_nan_fp16(input)) {
        is_nan = 1;
        res    = DEFAULT_NAN_FP8;
    } else if (is_zero_fp16(input) || is_inf_fp16(input)) {
        if (is_inf_fp16(input))
            is_inf = 1;

        // zero and inf are checked separately to prevent stochastic round up
        res = ibs(res, 7, 7, sbs(input, 15, 15)); // sign
        res = ibs(res, 6, 0, sbs(input, 13, 7)); // exp+mantissa
    } else {

        uint16_t out_sgn = sbs(input, 15, 15);
        uint16_t out_exp = sbs(input, 14, 10);
        uint16_t out_man = sbs(input, 9, 7);

        if (out_exp >= 15) {
            is_overflow = 1;
            if ((roundingModeSelected == (RND_TO_0)) || (out_sgn && (roundingModeSelected == (RND_TO_PINF))) ||
                (!out_sgn && (roundingModeSelected == (RND_TO_NINF))))

            { // +- max_normal
                out_man = 7;
                out_exp = 14;
            } else { // +-infinity
                out_exp = 15;
                out_man = 0;
            }
        } else {
            bool     out_g   = (sbs(input, 6, 6) == 1) ? 1 : 0;
            bool     out_rs  = (sbs(input, 5, 0) != 0) ? 1 : 0;
            uint32_t out_grs = sbs(input, 6, 0) << 25; // aligned left for comparing with LFSR[31:0]

            if (!is_denorm_fp16(input) && !is_zero_fp16(input))
                out_man = cbs(1, out_man, 3);

            bool need_rnd = (((roundingModeSelected == RND_TO_PINF) && (out_rs || out_g) && (out_sgn == 0)) ||
                             ((roundingModeSelected == RND_TO_NINF) && (out_rs || out_g) && (out_sgn == 1)) ||
                             ((roundingModeSelected == RND_TO_NE) &&
                              ((out_g && out_rs) || (out_g && !out_rs && (sbs(out_man, 0, 0) == 1)))) ||
                             ((roundingModeSelected == RND_HALF_AZ) && out_g) ||
                             ((roundingModeSelected == RND_SR) && (out_grs >= sr_register)));

            if (need_rnd) {
                out_man = out_man + 1;

                if (out_exp == 0) // denormal
                {
                    if (sbs(out_man, 3, 3) == 1)
                        out_exp = out_exp + 1;

                } else // normal
                {
                    if (sbs(out_man, 4, 4) == 1)
                        out_exp = out_exp + 1;
                }

                if (out_exp >= 15)
                    is_overflow = 1;
            }
        }
        // construct result
        res = ibs(res, 7, 7, out_sgn);
        res = ibs(res, 6, 3, out_exp);
        res = ibs(res, 2, 0, out_man);
    }

    if (clip_fp & fp8_is_infinity(res, 3) & (clip_fp_inf_input || !is_inf_fp16(input))) {
        // uint16_t out_sgn = sbs(input, 15, 15);
        // uint16_t out_exp = 0x1e;
        // uint16_t out_man = 0x3;

        // res = 0;

        // res = ibs(res, 7, 7, out_sgn);
        // res = ibs(res, 6, 2, out_exp);
        // res = ibs(res, 1, 0, out_man);

        res = res - 1; // will return +/-max_norm_value
    }

    if (ftz_fp8 && (fp8_is_denormal(res, 3))) {
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(res, 7, 7) == 0)) {
            // output=+min_normal
            res = 0 | (1 << 3) | (0 << 7);
        } else if ((roundingMode == RND_TO_NINF) && (sbs(res, 7, 7) == 1)) {
            // res=+min_normal
            res = 0 | (1 << 3) | (1 << 7);
        } else {
            res = ibs(res, 6, 0, 0); // output=+-0
        }
    }

    if (inf_exception != NULL)
        *inf_exception = is_inf;
    if (nan_exception != NULL)
        *nan_exception = is_nan;
    if (overflow_exception != NULL)
        *overflow_exception = is_overflow;

    return res;
}
uint8_t fp16_to_fp8_152(uint16_t input,
                        int      roundingMode,
                        uint32_t sr_register,
                        bool     clip_fp,
                        bool     ftz_fp8,
                        bool     clip_fp_inf_input,
                        uint8_t* inf_exception,
                        uint8_t* overflow_exception,
                        uint8_t* nan_exception)
{

    uint16_t res         = 0;
    uint8_t  is_inf      = 0;
    uint8_t  is_overflow = 0;
    uint8_t  is_nan      = 0;

    int roundingModeDnorm    = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_NEAREST_EVEN : roundingMode;
    int roundingModeNorm     = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM) ? VPE_RM_STOCHASTIC : roundingMode;
    int roundingModeSelected = is_denorm_fp16(input) ? roundingModeDnorm : roundingModeNorm;

    if (is_nan_fp16(input)) {
        is_nan = 1;
        res    = DEFAULT_NAN_FP8;
    } else if (is_zero_fp16(input) || is_inf_fp16(input)) {
        if (is_inf_fp16(input)) {
            is_inf = 1;
        }
        // zero and inf are checked separately to prevent stochastic round up
        res = sbs(input, 15, 8);
    } else {

        uint16_t out_sgn = sbs(input, 15, 15);
        uint16_t out_exp = sbs(input, 14, 10);
        uint16_t out_man = sbs(input, 9, 8);
        bool     out_g   = (sbs(input, 7, 7) == 1) ? 1 : 0;
        bool     out_rs  = (sbs(input, 6, 0) != 0) ? 1 : 0;
        uint32_t out_grs = sbs(input, 7, 0) << 24; // aligned left for comparing with LFSR[31:0]

        if (!is_denorm_fp16(input) && !is_zero_fp16(input))
            out_man = cbs(1, out_man, 2);

        bool need_rnd = (((roundingModeSelected == RND_TO_PINF) && (out_rs || out_g) && (out_sgn == 0)) ||
                         ((roundingModeSelected == RND_TO_NINF) && (out_rs || out_g) && (out_sgn == 1)) ||
                         ((roundingModeSelected == RND_TO_NE) &&
                          ((out_g && out_rs) || (out_g && !out_rs && (sbs(out_man, 0, 0) == 1)))) ||
                         ((roundingModeSelected == RND_HALF_AZ) && out_g) ||
                         ((roundingModeSelected == RND_SR) && (out_grs >= sr_register)));

        if (need_rnd) {
            out_man = out_man + 1;

            if (out_exp == 0) // denormal
            {
                if (sbs(out_man, 2, 2) == 1)
                    out_exp = out_exp + 1;

            } else // normal
            {
                if (sbs(out_man, 3, 3) == 1)
                    out_exp = out_exp + 1;
            }

            if (out_exp >= 31)
                is_overflow = 1;
        }
        // construct result
        res = ibs(res, 7, 7, out_sgn);
        res = ibs(res, 6, 2, out_exp);
        res = ibs(res, 1, 0, out_man);
    }

    if (clip_fp & fp8_is_infinity(res, 2) & (clip_fp_inf_input || !is_inf_fp16(input))) {
        // uint16_t out_sgn = sbs(input, 15, 15);
        // uint16_t out_exp = 0x1e;
        // uint16_t out_man = 0x3;

        // res = 0;

        // res = ibs(res, 7, 7, out_sgn);
        // res = ibs(res, 6, 2, out_exp);
        // res = ibs(res, 1, 0, out_man);

        res = res - 1; // will return +/-max_norm_value
    }

    if (ftz_fp8 && (fp8_is_denormal(res, 2))) {
        // flush denormal output to zero
        if ((roundingMode == RND_TO_PINF) && (sbs(res, 7, 7) == 0)) {
            // output=+min_normal
            res = 0 | (1 << 2) | (0 << 7);
        } else if ((roundingMode == RND_TO_NINF) && (sbs(res, 7, 7) == 1)) {
            // res=+min_normal
            res = 0 | (1 << 2) | (1 << 7);
        } else {
            res = ibs(res, 6, 0, 0); // output=+-0
        }
    }

    if (inf_exception != NULL)
        *inf_exception = is_inf;
    if (nan_exception != NULL)
        *nan_exception = is_nan;
    if (overflow_exception != NULL)
        *overflow_exception = is_overflow;

    return res;
}
} // namespace gaudi3
