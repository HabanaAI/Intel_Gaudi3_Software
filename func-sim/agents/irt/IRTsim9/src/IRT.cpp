/*
 * IRT.cpp
 *
 *  Created on: Jun 4, 2019
 *      Author: espektor
 */
/*
Todo

1.	Output image stripe can be any value from 1 to maximum allowed output stripe width - added with granularity of 8
pixels

2.	Output image location in the memory is not restricted to be aligned to 128 bytes boundary.
    a.	Output image write manager write data not from beginning of 128 bytes boundaries and will use byte enable
    b.	Output image write manager write data that not necessary multiple of 128 bytes and will use byte enable
IRT is not required to align address to 128 bytes and split transactions to addresses aligned to 128 bytes. It will be
done outside of IRT. But if transaction is not 128 bytes, it must use byte enable to specify invalid bytes.

3.	Output image location in the memory may be not continues when advancing from line to line. That is to allow write
rotator outputs as patches that are combined in bigger image in output memory To support that we will use stride
parameter that will define address jump when going from line to line in output image

4.	Rotator will be able to support larger output image stripe than 128 if rotation angle allow that (for small rotation
angles) and we still fit into rotation memory height of 128 lines x 256 pixels Rotator will be able to support larger
rotation angle that 60 degree (for small output stripes) and we still fit into rotation memory height of 128 lines x 256
pixels

5.	Simultaneous multiplane support from packed input image – pending decision till we will have area estimation
    To support rotation of packed image (stored as RGBRGB… and not as 3 separate planes), next need to be added to
rotator a.	3 instances of rotation memory to store R/G/B planes separately b.	Input image read manager will read
packed image and rotation memory write manager will separate it toward rotation memories as 3 planes c.	3 instances of
2x2 selector and of 8x bilinear interpolator to work on 3 planes x 8 elements
    d.	Interpolated pixels will be stored in packed format to output FIFO and then will be written into output memory
by output image write manager as packed image Alternatively we can have 3 FIFOs and 3 output image write managers and
store output image as 3 separate planes

*/

#define _USE_MATH_DEFINES

#include "IRT.h"
#include <algorithm>
#include <cmath>
#include "IRTutils.h"
#include "lanczos2.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Woverflow"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

namespace irt_utils_gaudi3
{
int      cycle;
int      cycle_per_task;
FILE*    log_file;
int      warp_gen_mode       = 2;
int      max_log_trace_level = 0;
int      cseed               = 1234;
uint16_t h9taskid;
bool     first_vld_warp_line;
// bool super_irt = 0;
bool           irt_h5_mode      = 0;
bool           read_iimage_BufW = 0;
bool           read_mimage_BufW = 0;
uint8_t        hl_only          = 0;
uint8_t        descr_gen_only   = 0;
bool           print_log_file   = 0;
bool           print_out_files  = 0;
bool           rand_input_image = 0;
extern FILE*   test_res;
Eirt_bmp_order irt_bmp_rd_order = e_irt_bmp_02H, irt_bmp_wr_order = e_irt_bmp_02H;
bool           irt_multi_image = 0;
bool           test_file_flag  = 0;
bool           test_failed     = 0;
int            range_bound     = 30;
// int temp_calc_cnt,temp_owm_cnt;
#if defined(RUN_WITH_SV) || defined(HABANA_SIMULATION)
#else
extern uint8_t*     ext_mem;
extern uint64_t***  input_image;
extern uint64_t**** output_image;
// bool test_failed = 0;
#endif

char file_warp_x[100];
char file_warp_y[100];
char file_descr_out[100];

int                         icc_cnt = 0;
queue<resamp_rmwm_inf>      rmwm_infQ[NUM_BANK_ROW];
queue<resamp_rmrm_inf>      rmrm_infQ;
queue<resamp_2x2_inf>       p2x2_infQ;
queue<cache_tag>            victim_infQ;
queue<rescale_h_interp_inf> rescale_h_intp_infQ;
queue<rescale_v_interp_inf> rescale_v_intp_infQ;
queue<resamp_shfl_inf>      gshfl_infQ;
queue<resamp_shfl_inf>      wshfl_infQ;
/*******************/
/* Strings section */
/*******************/
const char* irt_int_mode_s[2] = {"Bi-linear", "Nearest"};
const char* irt_irt_mode_s[9] =
    {"Rotation", "Affine", "Projection", "Mesh", "Resamp_FWD", "Resamp_BWD1", "Resamp_BWD2", "Rescale", "RescaleGrad"};
const char* irt_affn_mode_s[4]     = {"Rotation", "Scaling", "Rot_Scale", "Scale_Rot"};
const char* irt_proj_mode_s[4]     = {"Rotation", "Affine", "Projection", "Unknown"};
const char* irt_refl_mode_s[4]     = {"None", "Y axis", "X axis", "Origin"};
const char* irt_shr_mode_s[4]      = {"None shear", "Horizontal", "Vertical", "Hor & Vert"};
const char* irt_proj_order_s[6]    = {"YRP", "RYP", "YPR", "PYR", "PRY", "RPY"};
const char* irt_mesh_mode_s[4]     = {"Rotation", "Affine", "Projection", "Distortion"};
const char* irt_mesh_order_s[2]    = {"Predistortion", "Postdistortion"};
const char* irt_mesh_rel_mode_s[2] = {"Absolute", "Relative"};
const char* irt_mesh_format_s[5]   = {"Reserved0", "Fixed point", "Reserved2", "Reserved3", "FP32"};
const char* irt_mesh_fp_loc_s[16]  = {"S15",
                                     "S14.1",
                                     "S13.2",
                                     "S12.3",
                                     "S11.4",
                                     "S10.5",
                                     "S9.6",
                                     "S8.7",
                                     "S7.8",
                                     "S6.9",
                                     "S5.10",
                                     "S4.11",
                                     "S3.12",
                                     "S2.13",
                                     "S1.14",
                                     "S.15"};
const char* irt_flow_mode_s[4]     = {"wCFIFO_fixed_adaptive_2x2",
                                  "wCFIFO_fixed_adaptive_wxh",
                                  "nCFIFO_fixed_adaptive_wxh",
                                  "Reserved"};
const char* irt_rate_mode_s[4]     = {"Fixed", "Adaptive_2x2", "Adaptive_wxh", "Reserved"};
const char* irt_pixel_width_s[5]   = {"8-bit", "10-bit", "12-bit", "14-bit", "16-bit"};
const char* irt_bg_mode_s[3]       = {"Programmable value", "Frame boundary extension", "Input Pad Mode"};
const char* irt_crd_mode_s[2]      = {"Fixed point", "FP32"};
const char* irt_hl_format_s[5]     = {"arch ", "fpt64", "fpt32", "fixed", "fix16"};
const char* irt_mem_type_s[2]      = {"Rotation", "Mesh"};
const char* irt_prj_matrix_s[7]    = {"Roll", "Pitch", " Yaw ", "Shear", "Rotate", "Shear X", "Shear Y"};
const char* irt_flip_mode_s[4]     = {"Hor & Vert", "Vertical", "Horizontal", "None"};
const char* irt_buf_format_s[2]    = {"Static", "Dynamic"};
const char* irt_xi_start_call_s[7] = {"Desc gen", "IRM", "OICC", "RMRM_HL", "RMRM_OICC", "RMRM_TOP", "RMRM_MMRM"};
const char* irt_resamp_dtype_s[5]  = {"INT8", "INT16", "FP16", "BFP16", "FP32"};

/***********************/
/* IRT_UTILS functions */
/***********************/
int64_t IRT_top::IRT_UTILS::irt_min_int64(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

int64_t IRT_top::IRT_UTILS::irt_max_int64(int64_t a, int64_t b)
{
    return (a > b) ? a : b;
}

int16_t IRT_top::IRT_UTILS::irt_min_int16(int16_t a, int16_t b)
{
    return (a < b) ? a : b;
}

int16_t IRT_top::IRT_UTILS::irt_max_int16(int16_t a, int16_t b)
{
    return (a > b) ? a : b;
}

int16_t IRT_top::IRT_UTILS::irt_sat_int16(int16_t in, int16_t a, int16_t b)
{

    int16_t out;
    out = IRT_top::IRT_UTILS::irt_max_int16(in, a);
    out = IRT_top::IRT_UTILS::irt_min_int16(out, b);

    return out;
}

float IRT_top::IRT_UTILS::irt_fp32_to_float(uint32_t in)
{

    float32_struct in_fp32;
    float          out;

    in_fp32.sign     = (in >> FP32_SIGN_LOC) & 1;
    in_fp32.exp      = (in >> FP32_EXP_LOC) & FP32_EXP_MSK;
    in_fp32.mantissa = in & FP32_MANTISSA_MSK;
    out = (float)((1.0 + (double)in_fp32.mantissa / ((double)FP32_MANTISSA_MSK + 1)) * pow(2.0, in_fp32.exp - 127) *
                  (in_fp32.sign == 0 ? 1.0 : -1.0));

    memcpy(&out, &in, sizeof(in));

    return out;
}

int16_t IRT_top::IRT_UTILS::irt_ui16_to_i16(uint16_t in)
{

    int16_t out;

    memcpy(&out, &in, sizeof(in));

    return out;
}

uint32_t IRT_top::IRT_UTILS::irt_float_to_fp32(float in)
{

    float32_struct dout_fp32;
    float          mantissa;
    int            dout_int;

    dout_fp32.sign     = in > 0 ? 0 : 1;
    mantissa           = frexp(fabs(in), &dout_fp32.exp);
    dout_fp32.mantissa = (int)rint(((2 * mantissa - 1) * (FP32_MANTISSA_MSK + 1)));
    dout_fp32.exp--;
    dout_int = (dout_fp32.sign << FP32_SIGN_LOC) | (((dout_fp32.exp + 127) & FP32_EXP_MSK) << FP32_EXP_LOC) |
               (dout_fp32.mantissa & FP32_MANTISSA_MSK);

    memcpy(&dout_int, &in, sizeof(in));

    return dout_int;
}

int64_t IRT_top::IRT_UTILS::irt_min_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize)
{
    int64_t out;
    out = in[0];
    for (uint8_t k = 1; k < psize; k++) {
        out = IRT_top::IRT_UTILS::irt_min_int64(out, in[k]);
    }
    return out;
}

int64_t IRT_top::IRT_UTILS::irt_max_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize)
{
    int64_t out;
    out = in[0];
    for (uint8_t k = 1; k < psize; k++) {
        out = IRT_top::IRT_UTILS::irt_max_int64(out, in[k]);
    }
    return out;
}

float IRT_top::IRT_UTILS::conversion_bit_float(uint32_t in, Eirt_resamp_dtype_enum DataType, bool clip_fp)
{
    // bool clip_fp=0; //TODO later replace with descr
    float    out_f;
    uint32_t out_int;
    uint16_t in16 = (uint16_t)(in & 0xFFFF);
    uint16_t in8  = (uint8_t)(in & 0xFF);

    switch (DataType) {
        case e_irt_fp16:
            // out_f = half2float(in16);
            out_int = gaudi3::fp16_to_fp32(in16, clip_fp);
            memcpy(&out_f, &out_int, sizeof(uint32_t));
            // IRT_TRACE_UTILS("fp16_to_fp32:in %x in16 %x clip_fp %d DataType %d out_int %x out_f %f
            // \n",in,in16,clip_fp,DataType,out_int,out_f);
            break;
        case e_irt_bfp16:
            // out_f = bf16_to_float(in16);
            out_int = gaudi3::bf16_to_fp32(in16, clip_fp);
            memcpy(&out_f, &out_int, sizeof(uint32_t));
            break;
        case e_irt_fp32: memcpy(&out_f, &in, sizeof(uint32_t)); break;
        case e_irt_int8: // TODO
            out_f = (float)in8;
            break;
        case e_irt_int16: // TODO
            out_f = (float)in16;
            break;
    };
    return out_f;
}

uint32_t IRT_top::IRT_UTILS::conversion_float_bit(float                  in,
                                                  Eirt_resamp_dtype_enum DataType,
                                                  int16_t                bli_shift,
                                                  uint16_t               MAX_VALo,
                                                  uint16_t               Ppo,
                                                  bool                   clip_fp,
                                                  bool                   clip_fp_inf_input,
                                                  bool                   ftz_en,
                                                  bool                   print)
{
    uint32_t out_bit, out_tmp = 0;
    int64_t  res;
#ifdef USE_OLD_FP_CONV
    IRThalf hlf_union;
    half    hlf; // = half(in);
#endif
    double inf           = in;
    int   rounding_mode = 0; // fow now fixed to RHNE

    switch (DataType) {
        case e_irt_fp16:
            // hlf = half(in);
            // hlf_union.val_hf = hlf;
            // out_bit = (uint32_t)hlf_union.val_uint;
            out_bit = (uint32_t)gaudi3::fp32_to_fp16(in, rounding_mode, 0, clip_fp, ftz_en, clip_fp_inf_input);
            if (print)
                IRT_TRACE_UTILS("fp32_to_fp16:in %f \n", in);
            break;
        case e_irt_bfp16:
            // out_bit = (uint32_t)float_to_bf16(in) & 0xFFFF;
            out_bit = (uint32_t)gaudi3::fp32_to_bf16(in, rounding_mode, 0, clip_fp);
            break;
        case e_irt_fp32:
            out_tmp = (uint32_t)gaudi3::fp32_to_fp32(in, rounding_mode, clip_fp, ftz_en, clip_fp_inf_input);
            if (print)
                IRT_TRACE_UTILS("fp32_to_fp32:in %f out_bit %x clip_fp %d clip_fp_inf_input %d\n",
                                in,
                                out_tmp,
                                clip_fp,
                                clip_fp_inf_input);
            memcpy(&out_bit, &out_tmp, sizeof(float));
            break;
        case e_irt_int8:
        case e_irt_int16:
            if (print)
                IRT_TRACE_UTILS("inf %f bli_shift %d\n", inf, bli_shift);
            inf *= (float)pow(2.0, bli_shift);
            // TODO - (int32_t)rint((float)inf) -> inf > RAND_MAX, casting converts final value from +ve to -ve, hence
            // using below patch to manually check range and convert
            if (inf > (float)(RAND_MAX)) {
                res = (int64_t)(RAND_MAX); // TODO Check with Sagar
            } else if (inf < ((((double)RAND_MAX + 1) * -1))) {
                res = (((int64_t)RAND_MAX + 1) * -1);
            } else {
                if (print)
                    IRT_TRACE_UTILS("New inf %f rint %f %f rintf %f \n",
                                    inf,
                                    rint((float)inf),
                                    (float)rint((float)inf),
                                    rintf((float)inf));
                res = (int64_t)rint((double)inf);
            }
            if (print)
                IRT_TRACE_UTILS("rint res %lx \n", res);
            res = std::min((int64_t)res, (int64_t)MAX_VALo); // if (res > 255) res = 255;
            if (print)
                IRT_TRACE_UTILS("min res %lx \n", res);
            res = std::max(res, (int64_t)0); // if (res < 0) res = 0;
            if (print)
                IRT_TRACE_UTILS("max res %lx \n", res);
            res = (res << Ppo);
            if (print)
                IRT_TRACE_UTILS("ppo res %lx \n", res);
            out_bit = (uint16_t)res;
            if (print)
                IRT_TRACE_UTILS("out res %lx \n", res);
    };
    return out_bit;
}
int16_t IRT_top::IRT_UTILS::conversion_float_fxd16(float inf, uint8_t int16_point)
{
    return ((int16_t)rint(((double)inf) * pow(2.0, int16_point)));
}
float IRT_top::IRT_UTILS::conversion_fxd16_float(int32_t infxd, uint8_t int16_point, bool input_32bit)
{
    if (input_32bit) {
        return ((float)(((double)infxd) / ((float)pow(2.0, int16_point))));
    } else {
        int16_t infxd16 = (int16_t)infxd;
        return ((float)(((double)infxd16) / ((float)pow(2.0, int16_point))));
    }
}
//---------------------------------------
// pi - phase index
// ti - tap index
float IRT_top::IRT_UTILS::get_rescale_coeff(filter_type_t flt_type, Coeff_LUT_t* myLUT, int pi, int ti)
{
    float coeff = 0;
    switch (flt_type) {
        case e_irt_lanczos2:
        case e_irt_lanczos3: coeff = myLUT->Coeff_table[(pi * myLUT->num_taps) + ti]; break;
        case e_irt_bicubic: coeff = myLUT->Coeff_table[(pi * myLUT->num_taps) + ti]; break;
    }
    return coeff;
}

/******************/
/* IICC functions */
/******************/
int64_t IRT_top::irt_iicc_fixed_0(int32_t coef0, int32_t coef1, int16_t x, int16_t y, uint8_t prec_align)
{
    return ((int64_t)coef0 * x + (int64_t)coef1 * y) * (int64_t)pow(2, prec_align);
}

int64_t IRT_top::irt_iicc_fixed_k(int64_t coord_0, int32_t coef, uint8_t k, uint8_t prec_align)
{
    return coord_0 + ((int64_t)coef * (int64_t)pow(2, prec_align)) * k;
}

float IRT_top::irt_iicc_float_0(float coef0, float coef1, float x, float y)
{
    return coef0 * x + coef1 * y;
}

float IRT_top::irt_iicc_float_k(float coord_0, float coef, uint8_t k)
{
    return coord_0 + coef * k;
}

void IRT_top::irt_iicc_fixed(int64_t out[IRT_ROT_MAX_PROC_SIZE],
                             int32_t coef[2],
                             int16_t x,
                             int16_t y,
                             uint8_t prec_align)
{

    out[0] = irt_iicc_fixed_0(
        coef[0], coef[1], x, y, prec_align); // have precision of 32b, need to round to 31 by nearest even

    out[0] = IRT_top::irt_fix32_to_fix31(out[0]);

    for (uint8_t k = 1; k < IRT_ROT_MAX_PROC_SIZE; k++) {
        out[k] = irt_iicc_fixed_k(out[0], coef[0], k, prec_align);
    }
}

void IRT_top::irt_iicc_float_inc(float out[2], float coef[3], float x, float y, uint8_t k)
{

    float res_tmp0 = coef[0] * x + coef[1] * y;
    float res_tmp1 = res_tmp0 + coef[0] * k;
    out[0]         = res_tmp0 + coef[2];
    out[1]         = res_tmp1 + coef[2];
}

void IRT_top::irt_iicc_float(float   out[IRT_ROT_MAX_PROC_SIZE],
                             float   N[3],
                             float   D[3],
                             float   x,
                             float   y,
                             float   rot_slope,
                             float   affine_slope,
                             uint8_t psize,
                             float   psize_inv,
                             uint8_t mode)
{

    float Nom[2], Den[2];
    irt_iicc_float_inc(Nom, N, x, y, psize - 1);
    irt_iicc_float_inc(Den, D, x, y, psize - 1);

    float Ci0 = Nom[0] / Den[0];
    float Cip = Nom[1] / Den[1];

    float proj_slop = (Cip - Ci0) * psize_inv;
    float slope     = 0;
    switch (mode) {
        case e_irt_rotation:
            out[0] = Nom[0];
            slope  = rot_slope;
            break;
        case e_irt_affine:
            out[0] = Nom[0];
            slope  = affine_slope;
            break;
        case e_irt_projection:
            out[0] = Ci0;
            slope  = proj_slop;
            break;
        case e_irt_mesh:
            out[0] = 0;
            slope  = 0;
            break;
    }

    for (uint8_t k = 1; k < psize; k++) {
        out[k] = irt_iicc_float_k(out[0], slope, k);
    }
}

void IRT_top::irt_iicc(const irt_desc_par&   irt_desc,
                       int16_t               Xo0,
                       int16_t               Yo,
                       int64_t               Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                       int64_t               Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                       bus8XiYi_float_struct mbli_out,
                       Eirt_block_type       block_type)
{

    float   Xi_float[IRT_ROT_MAX_PROC_SIZE] = {0}, Yi_float[IRT_ROT_MAX_PROC_SIZE] = {0};
    float   N0[3] = {0}, N1[3] = {0}, D0[3] = {0}, D1[3] = {0};
    int32_t M0[2] = {0}, M1[2] = {0};

    if (block_type == e_irt_block_rot) {
        switch (irt_desc.irt_mode) {
            case e_irt_rotation:
                M0[0] = irt_desc.cosi;
                M0[1] = irt_desc.sini;
                M1[0] = -irt_desc.sini;
                M1[1] = irt_desc.cosi;
                N0[0] = irt_desc.cosf;
                N0[1] = irt_desc.sinf;
                N0[2] = 0;
                N1[0] = -irt_desc.sinf;
                N1[1] = irt_desc.cosf;
                N1[2] = 0;
                D0[2] = 1;
                D1[2] = 1;
                break;
            case e_irt_affine:
                M0[0] = irt_desc.M11i;
                M0[1] = irt_desc.M12i;
                M1[0] = irt_desc.M21i;
                M1[1] = irt_desc.M22i;
                N0[0] = irt_desc.M11f;
                N0[1] = irt_desc.M12f;
                N0[2] = 0;
                N1[0] = irt_desc.M21f;
                N1[1] = irt_desc.M22f;
                N1[2] = 0;
                D0[2] = 1;
                D1[2] = 1;
                break;
            case e_irt_projection:
                N0[0] = irt_desc.prj_Af[0];
                N0[1] = irt_desc.prj_Af[1];
                N0[2] = irt_desc.prj_Af[2];
                N1[0] = irt_desc.prj_Bf[0];
                N1[1] = irt_desc.prj_Bf[1];
                N1[2] = irt_desc.prj_Bf[2];
                D0[0] = irt_desc.prj_Cf[0];
                D0[1] = irt_desc.prj_Cf[1];
                D0[2] = irt_desc.prj_Cf[2];
                D1[0] = irt_desc.prj_Df[0];
                D1[1] = irt_desc.prj_Df[1];
                D1[2] = irt_desc.prj_Df[2];
                break;
            default: break;
        }
    } else {
        M0[0] = irt_desc.mesh_Gh;
        M0[1] = 0;
        M1[0] = 0;
        M1[1] = irt_desc.mesh_Gv;
        N0[0] = 0;
        N0[1] = 0;
        N0[2] = 0;
        N1[0] = 0;
        N1[1] = 0;
        N1[2] = 0;
        D0[2] = 1;
        D1[2] = 1;
    }

    if (block_type == e_irt_block_rot) {
        IRT_top::irt_iicc_fixed(Xi_fixed, M0, Xo0, Yo, irt_desc.prec_align);
        IRT_top::irt_iicc_fixed(Yi_fixed, M1, Xo0, Yo, irt_desc.prec_align);
        IRT_top::irt_iicc_float(Xi_float,
                                N0,
                                D0,
                                (float)Xo0 / 2,
                                (float)Yo / 2,
                                irt_desc.cosf,
                                irt_desc.M11f,
                                irt_desc.proc_size,
                                (float)(1.0 / (irt_desc.proc_size - 1)),
                                irt_desc.irt_mode);
        IRT_top::irt_iicc_float(Yi_float,
                                N1,
                                D1,
                                (float)Xo0 / 2,
                                (float)Yo / 2,
                                -irt_desc.sinf,
                                irt_desc.M21f,
                                irt_desc.proc_size,
                                (float)(1.0 / (irt_desc.proc_size - 1)),
                                irt_desc.irt_mode);
        //      IRT_TRACE("Xo0 = %x Yo = %x N0 [0,1,2] [%llx:%llx:%llx] D0 [0,1,2]
        //      [%llx:%llx:%llx]\n",Xo0,Yo,N0[0],N0[1],N0[2],D0[0],D0[1],D0[2]);
    } else {
        IRT_top::irt_iicc_fixed(Xi_fixed, M0, Xo0, Yo, 0);
        IRT_top::irt_iicc_fixed(Yi_fixed, M1, Xo0, Yo, 0);
    }

    for (uint8_t k = 0; k < IRT_ROT_MAX_PROC_SIZE; k++) {
        if (block_type == e_irt_block_rot) {
            if (irt_desc.irt_mode == e_irt_projection || irt_desc.irt_mode == e_irt_mesh ||
                ((irt_desc.irt_mode == e_irt_rotation || irt_desc.irt_mode == e_irt_affine) &&
                 irt_desc.crd_mode == e_irt_crd_mode_fp32)) {
                if (irt_desc.irt_mode == e_irt_mesh) {
                    Xi_fixed[k] = (int64_t)rint(mbli_out.pix[k].X * (float)pow(2, 31));
                    Yi_fixed[k] = (int64_t)rint(mbli_out.pix[k].Y * (float)pow(2, 31));
                } else {
                    Xi_fixed[k] = (int64_t)rint(Xi_float[k] * (float)pow(2, 31));
                    Yi_fixed[k] = (int64_t)rint(Yi_float[k] * (float)pow(2, 31));
                    //               IRT_TRACE("index=%d Xi_fixed = %llx Xi_float = %f Yi_fixed=%llx
                    //               Yi_float=%f\n",k,Xi_fixed[k],Xi_float[k],Yi_fixed[k],Yi_float[k]);
                }
            }
            if (irt_desc.irt_mode == e_irt_rotation || irt_desc.irt_mode == e_irt_affine) {
                Xi_fixed[k] += ((int64_t)irt_desc.image_par[IIMAGE].Xc) << 30;
                Yi_fixed[k] += ((int64_t)irt_desc.image_par[IIMAGE].Yc) << 30;
            } else if (irt_desc.irt_mode == e_irt_mesh && irt_desc.mesh_rel_mode == e_irt_mesh_relative) {
                Xi_fixed[k] += ((int64_t)Xo0 + (int64_t)k * 2 + irt_desc.image_par[OIMAGE].Xc) << 30;
                Yi_fixed[k] += ((int64_t)Yo + irt_desc.image_par[OIMAGE].Yc) << 30;
            }
        } else {
            if (irt_desc.mesh_sparse_h == 0)
                Xi_fixed[k] = ((int64_t)Xo0 + (int64_t)k * 2) << 30;
            if (irt_desc.mesh_sparse_v == 0)
                Yi_fixed[k] = ((int64_t)Yo) << 30;
        }
    }
}

int64_t IRT_top::irt_fix32_to_fix31(int64_t in)
{

    int64_t out;

    if ((in >> 1) & 1) { // odd after round, .1 will be round up
        out = (in + 1) >> 1;
    } else { // even after round, .1 will be down
        out = in >> 1;
    }

    return out;
}

void IRT_top::xi_start_calc_err_print(const char error_str[50],
                                      int64_t    Xi_start_tmp,
                                      int16_t    line,
                                      uint8_t    caller,
                                      uint8_t    desc)
{
    IRT_TRACE_UTILS("Task %d %s error: input coordinates range exceeds [-32767:32767]: %s = %" PRId64
                    " at input line %d cycle %d\n",
                    desc,
                    irt_xi_start_call_s[caller],
                    error_str,
                    Xi_start_tmp,
                    line,
                    cycle);
    IRT_TRACE_TO_RES_UTILS(
        test_res,
        "was not run, task %d %s error: input coordinates range exceeds [-32767:32767]: %s = %" PRId64
        " at input line %d cycle %d\n",
        desc,
        irt_xi_start_call_s[caller],
        error_str,
        Xi_start_tmp,
        line,
        cycle);
    IRT_CLOSE_FAILED_TEST(0);
}

int16_t IRT_top::xi_start_calc(const irt_desc_par& irt_desc, int16_t line, uint8_t caller, uint8_t desc)
{

    int64_t line_diff, line_diff_x_tan, /*line_diff_x_tan_msb, line_diff_x_tan_lsb,*/ Xi_first_delta, Xi_start_fixed,
        Xi_start_tmp;
    int16_t Xi_start; //, im_read_slope_msb, im_read_slope_lsb;

    // fixed point calculation for Xi_start
    // int line_diff = (line<<irt_top->irt_cfg.pIRT_TOTAL_PREC) -
    // irt_top->irt_desc[irt_top->task[e_irt_irm]].Yi_first_fixed;
    line_diff = ((int64_t)line * ((uint64_t)1 << IRT_SLOPE_PREC)) - irt_desc.Yi_first_fixed;
    // im_read_slope_msb = irt_desc.im_read_slope >> 16;
    // im_read_slope_lsb = irt_desc.im_read_slope & 0xffff;
    // line_diff_x_tan_msb = (int64_t)line_diff * im_read_slope_msb;
    // line_diff_x_tan_lsb = (int64_t)line_diff * im_read_slope_lsb;
    // line_diff_x_tan = (int64_t)line_diff_x_tan_msb * (1 << 16) + line_diff_x_tan_lsb;
    line_diff_x_tan = (int64_t)line_diff * irt_desc.im_read_slope;
    Xi_first_delta  = (line_diff_x_tan + IRT_SLOPE_ROUND) >> IRT_SLOPE_PREC;
    Xi_start_fixed  = irt_desc.Xi_first_fixed + Xi_first_delta;

    Xi_start_tmp = ((Xi_start_fixed + 0 * IRT_SLOPE_ROUND) >> IRT_SLOPE_PREC);
    if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
        IRT_top::xi_start_calc_err_print("initial Xi_start", Xi_start_tmp, line, caller, desc);
    }

    // TBD
    if (irt_desc.rot_dir == IRT_ROT_DIR_NEG) { // negative rotation angle
        Xi_start_tmp -= (int64_t)irt_desc.Xi_start_offset; // 50; //2
        if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
            IRT_top::xi_start_calc_err_print("Xi_start - Xi_start_offset", Xi_start_tmp, line, caller, desc);
        }
        Xi_start_tmp -= (int64_t)irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
        if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
            IRT_top::xi_start_calc_err_print(
                "Xi_start - Xi_start_offset_flip * read_hflip", Xi_start_tmp, line, caller, desc);
        }
    }
    if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle
        Xi_start_tmp += (int64_t)irt_desc.Xi_start_offset;
        if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
            IRT_top::xi_start_calc_err_print("Xi_start + Xi_start_offset", Xi_start_tmp, line, caller, desc);
        }
        Xi_start_tmp += (int64_t)irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
        if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
            IRT_top::xi_start_calc_err_print(
                "Xi_start + Xi_start_offset_flip * read_hflip", Xi_start_tmp, line, caller, desc);
        }
    }

    if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle
        Xi_start_tmp -= (int64_t)irt_desc.image_par[IIMAGE].S;
        if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
            IRT_top::xi_start_calc_err_print("Xi_start - Si", Xi_start_tmp, line, caller, desc);
        }
    }

    if (irt_desc.rot90) {
        if (irt_desc.rot90_inth == 0) { // interpolation is not required, working regulary
            if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle
                Xi_start_tmp += (int64_t)irt_desc.image_par[IIMAGE].S;
                if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
                    IRT_top::xi_start_calc_err_print("for rot90 Xi_start + Si", Xi_start_tmp, line, caller, desc);
                }
            } else {
                Xi_start_tmp -= ((int64_t)irt_desc.image_par[IIMAGE].S - 3);
                if (abs(Xi_start_tmp) > IRT_Xi_start_max) {
                    IRT_top::xi_start_calc_err_print("for rot90 Xi_start - (Si - 3)", Xi_start_tmp, line, caller, desc);
                }
            }
        } else {
            if (irt_desc.rot_dir == IRT_ROT_DIR_NEG) {
                // Xi_start -= 2;
            }
        }
    }

    Xi_start = (int16_t)Xi_start_tmp;
    if (irt_desc.bg_mode == e_irt_bg_frame_repeat) {
        Xi_start = IRT_top::IRT_UTILS::irt_sat_int16(
            Xi_start, -(irt_desc.image_par[IIMAGE].S - 1), irt_desc.image_par[IIMAGE].W - 1);
    }
    // printf("line %d xi_start %d\n", line, Xi_start);
    return Xi_start;
}

int16_t IRT_top::yi_start_calc(const irt_desc_par& irt_desc, Eirt_block_type block_type)
{

    int16_t Yi_start;
    if (block_type == e_irt_block_rot) {
#if 0
		Yi_start = (int16_t)(irt_desc.Yi_first_fixed >> IRT_SLOPE_PREC) - irt_desc.read_vflip;
		if (irt_desc.bg_mode == e_irt_bg_frame_repeat) {
			Yi_start = IRT_top::IRT_UTILS::irt_max_int16(Yi_start, 0);
		}
#endif
        Yi_start = irt_desc.Yi_start;
        // Yi_start = (int16_t)(irt_desc.Yi_first_fixed >> IRT_SLOPE_PREC) - irt_desc.read_vflip;

    } else {
        Yi_start = 0;
    }

    return Yi_start;
}

int16_t IRT_top::YB_adjustment(const irt_desc_par& irt_desc, Eirt_block_type block_type, int16_t YB)
{

    if (block_type == e_irt_block_rot) {
#ifdef IRT_USE_FLIP_FOR_MINUS1
        if (irt_desc.read_vflip || (irt_desc.rot90 && irt_desc.rot90_intv == 0))
#else
        if (irt_desc.rot90 && irt_desc.rot90_intv == 0)
#endif
            YB--; // interpolation is not required so we dont need extra line
    } else {
        if (irt_desc.mesh_sparse_v == 0)
            YB--;
        YB = IRT_top::IRT_UTILS::irt_min_int16(YB, (int16_t)irt_desc.image_par[MIMAGE].H - 1);
    }

    return YB;
}
/**************************/
/* Rate control functions */
/**************************/
uint8_t IRT_top::irt_rate_ctrl1(IRT_top* irt_top,
                                int64_t  Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                int64_t  Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                uint8_t  proc_size)
{

    int16_t XL_fixed[IRT_ROT_MAX_PROC_SIZE], XR_fixed[IRT_ROT_MAX_PROC_SIZE], YT_fixed[IRT_ROT_MAX_PROC_SIZE],
        YB_fixed[IRT_ROT_MAX_PROC_SIZE];
    int16_t int_win_w[IRT_ROT_MAX_PROC_SIZE], int_win_h[IRT_ROT_MAX_PROC_SIZE];
    uint8_t adj_proc_size_l = 1;

    // calculating interpolation region corners for 8 proc sizes
    for (uint8_t k = 0; k < proc_size; k++) { // calculating corners for 8 proc sizes
        XL_fixed[k]  = (int16_t)(IRT_top::IRT_UTILS::irt_min_int64(Xi_fixed[0], Xi_fixed[k]) >> 31);
        XR_fixed[k]  = (int16_t)(IRT_top::IRT_UTILS::irt_max_int64(Xi_fixed[0], Xi_fixed[k]) >> 31) + 1;
        YT_fixed[k]  = (int16_t)(IRT_top::IRT_UTILS::irt_min_int64(Yi_fixed[0], Yi_fixed[k]) >> 31);
        YB_fixed[k]  = (int16_t)(IRT_top::IRT_UTILS::irt_max_int64(Yi_fixed[0], Yi_fixed[k]) >> 31) + 1;
        int_win_w[k] = XR_fixed[k] - XL_fixed[k] + 1;
        int_win_h[k] = YB_fixed[k] - YT_fixed[k] + 1;
        if ((int_win_h[k] <= IRT_INT_WIN_H) && (int_win_w[k] <= IRT_INT_WIN_W)) // fit interpolation window
            adj_proc_size_l = k + 1; // set for number of already processed pixels in group
    }

#if 0
	if (irt_desc.rate_mode == rate_adaptive) {
		IRT_TRACE("Task %d IRT model %s: for output range[%d][%d:%d] processing rate is limited to %d pixels/cycle\n", irt_task,
			irt_irt_mode_s[irt_desc.irt_mode], row, col, col + 7, adj_proc_size);
	}
	else {
		if (adj_proc_size < irt_desc.proc_size) {
			IRT_TRACE("Task %d IRT model %s: output range[%d][%d:%d] does not fit input interpolation window 8x9 for %d processing rate, require %dx%d window from input range[%d:%d][%d:%d]\n",
				irt_task, irt_irt_mode_s[irt_desc.irt_mode], row, col, col + 7, irt_desc.proc_size, int_win_h[adj_proc_size - 1], int_win_w[adj_proc_size - 1],
				YT_fixed[adj_proc_size - 1], YB_fixed[adj_proc_size - 1], XL_fixed[adj_proc_size - 1], XR_fixed[adj_proc_size - 1]);
			IRT_TRACE_TO_RES(test_res, " failed, Task %d IRT model %s: output range[%d][%d:%d] does not fit input interpolation window 8x9 for %d processing rate, require %dx%d window from input range[%d:%d][%d:%d]\n",
				irt_task, irt_irt_mode_s[irt_desc.irt_mode], row, col, col + 7, irt_desc.proc_size, int_win_h[adj_proc_size - 1], int_win_w[adj_proc_size - 1],
				YT_fixed[adj_proc_size - 1], YB_fixed[adj_proc_size - 1], XL_fixed[adj_proc_size - 1], XR_fixed[adj_proc_size - 1]);
			IRT_CLOSE_FAILED_TEST(0);
		}
	}
#endif

    return adj_proc_size_l;
}

uint8_t IRT_top::irt_rate_ctrl2(IRT_top*              irt_top,
                                irt_cfifo_data_struct cfifo_data_o[IRT_ROT_MAX_PROC_SIZE],
                                bool                  rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
                                int16_t               rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
                                uint8_t               max_proc_size,
                                bool                  rate_mode)
{

    uint8_t adj_proc_size_l;
    bool    max_rate_found = 0;

    irt_rot_mem_rd_stat rd_stat;

    for (uint8_t i = 0; i < IRT_ROT_MEM_ROW_BANKS; i++) {
        for (uint8_t j = 0; j < IRT_ROT_MEM_COL_BANKS; j++) {
            rd_sel[i][j]                = 0;
            rd_addr[i][j]               = 0;
            rd_stat.bank_num_read[i][j] = 0;
            rd_stat.bank_min_addr[i][j] = 10000;
            rd_stat.bank_max_addr[i][j] = -10000;
        }
    }

    adj_proc_size_l = 0;

    for (uint8_t pixel = 0; pixel < max_proc_size && max_rate_found == 0; pixel++) {

        // calculation banks contention
        irt_top->irt_rot_mem_rd_stat_calc(irt_top, cfifo_data_o[pixel], rd_stat);

        // checking results accumulated till pixel
        for (uint8_t row_bank = 0; row_bank < IRT_ROT_MEM_ROW_BANKS; row_bank++) {
            for (uint8_t col_bank = 0; col_bank < IRT_ROT_MEM_COL_BANKS; col_bank++) {
                if (rd_stat.bank_num_read[row_bank][col_bank] > 1 /*bank is accessed more than once, can be conflict*/
                    && rate_mode == 1 /*check conflicts*/ &&
                    rd_stat.bank_max_addr[row_bank][col_bank] !=
                        rd_stat.bank_min_addr[row_bank][col_bank]) { // there is conflict on the bank because more than
                                                                     // 1 different addresses are accessed
                    max_rate_found = 1;
                } else {
                    // bank_num_read[row_bank][col_bank] <= 1 //bank is accessed at most once - no conflict
                    // || bank_max_addr[row_bank][col_bank] == bank_min_addr[row_bank][col_bank] // 1 addresses is
                    // accessed
                    // setting rotation memory controls
                    if (rd_stat.bank_num_read[row_bank][col_bank] > 0) {
                        rd_sel[row_bank][col_bank] = 1;
                    } else {
                        rd_sel[row_bank][col_bank] = 0;
                    }
                    rd_addr[row_bank][col_bank] = rd_stat.bank_min_addr[row_bank][col_bank];
                    // IRT_TRACE("row_bank %d, col_bank %d, reads %d, rd_sel %d, rd_addr %d at cycle %d\n", row_bank,
                    // col_bank, bank_num_read[row_bank][col_bank], rd_sel[row_bank][col_bank],
                    // rd_addr[row_bank][col_bank], cycle);
                }
            }
        }
        if (max_rate_found == 0) {
            adj_proc_size_l = pixel + 1;
        }
    }

    return adj_proc_size_l;
}

void IRT_top::irt_rot_mem_rd_stat_calc(IRT_top* irt_top, irt_cfifo_data_struct rd_ctrl, irt_rot_mem_rd_stat& rd_stat)
{

    uint8_t bank_row;

    for (uint8_t idx = 0; idx < 2; idx++) {

        bank_row = rd_ctrl.bank_row[idx];
        if (bank_row >= IRT_ROT_MEM_ROW_BANKS) {
            IRT_TRACE("irt_rot_mem_rd_stat_calc: non valid bank_row %u\n", bank_row);
            IRT_TRACE_TO_RES(test_res, " failed, irt_rot_mem_rd_stat_calc: non valid bank_row %u\n", bank_row);
            IRT_CLOSE_FAILED_TEST(0);
        }

        for (uint8_t bank_col = 0; bank_col < IRT_ROT_MEM_COL_BANKS; bank_col++) {

            rd_stat.bank_num_read[bank_row][bank_col] += rd_ctrl.rd_sel[idx][bank_col];

            if (rd_ctrl.rd_sel[idx][bank_col]) {
                rd_stat.bank_min_addr[bank_row][bank_col] = (uint16_t)std::min(
                    (int)rd_stat.bank_min_addr[bank_row][bank_col], (int)rd_ctrl.rd_addr[idx][bank_col]);
                rd_stat.bank_max_addr[bank_row][bank_col] = (uint16_t)std::max(
                    (int)rd_stat.bank_max_addr[bank_row][bank_col], (int)rd_ctrl.rd_addr[idx][bank_col]);
            }
        }
    }
}

void IRT_top::irt_rot_desc_print(IRT_top* irt_top, uint8_t desc)
{
    IRT_TRACE("***************************\n");
    if (irt_top->irt_desc[desc].crd_mode == e_irt_crd_mode_fixed) {
        IRT_TRACE("* Rotation angle   %6.2f[%6.2f] *\n",
                  ((double)asin((double)irt_top->irt_desc[desc].sini / pow(2, irt_top->rot_pars[desc].TOTAL_PREC)) *
                   180.0 / M_PI),
                  ((double)asin((double)irt_top->irt_desc[desc].sini / pow(2, irt_top->rot_pars[desc].TOTAL_PREC)) *
                       180.0 / M_PI +
                   180.0 * irt_top->irt_desc[desc].read_vflip));
        IRT_TRACE("* sin %d cos %d *\n", irt_top->irt_desc[desc].sini, irt_top->irt_desc[desc].cosi);
    } else {
        IRT_TRACE(
            "* Rotation angle   %6.2f[%6.2f] *\n",
            ((double)asin(irt_top->irt_desc[desc].sinf) * 180.0 / M_PI),
            ((double)asin(irt_top->irt_desc[desc].sinf) * 180.0 / M_PI + 180.0 * irt_top->irt_desc[desc].read_vflip));
        IRT_TRACE("* sin %f cos %f *\n", (double)irt_top->irt_desc[desc].sinf, (double)irt_top->irt_desc[desc].cosf);
    }
    IRT_TRACE("***************************\n");
}

void IRT_top::irt_aff_desc_print(IRT_top* irt_top, uint8_t desc)
{
    IRT_TRACE("***************************\n");
    IRT_TRACE("* Affine mode = %s *\n", irt_top->rot_pars[desc].affine_mode);
    IRT_TRACE("* Reflection mode = %s *\n", irt_refl_mode_s[irt_top->rot_pars[desc].reflection_mode]);
    IRT_TRACE("* Shear mode = %s *\n", irt_shr_mode_s[irt_top->rot_pars[desc].shear_mode]);
    IRT_TRACE("* Shear angles: x = %5.2f, y = %5.2f *\n",
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_x],
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_y]);
    IRT_TRACE("* Scaling factors: Sx = %5.2f, Sy = %5.2f *\n", irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy);
    IRT_TRACE("* Affine matrix:[%10.6f  %10.6f]\n",
              (double)irt_top->irt_desc[desc].M11f,
              (double)irt_top->irt_desc[desc].M12f);
    IRT_TRACE("                [%10.6f  %10.6f]\n",
              (double)irt_top->irt_desc[desc].M21f,
              (double)irt_top->irt_desc[desc].M22f);
    IRT_TRACE("* Affine matrix:[%08x  %08x]\n",
              (uint32_t)irt_top->irt_desc[desc].M11i,
              (uint32_t)irt_top->irt_desc[desc].M12i);
    IRT_TRACE("                [%08x  %08x]\n",
              (uint32_t)irt_top->irt_desc[desc].M21i,
              (uint32_t)irt_top->irt_desc[desc].M22i);
    IRT_TRACE("***************************\n");
}

void IRT_top::irt_prj_desc_print(IRT_top* irt_top, uint8_t desc)
{
    IRT_TRACE("***************************\n");
    IRT_TRACE("* Projection mode = %s *\n", irt_proj_mode_s[irt_top->rot_pars[desc].proj_mode]);
    IRT_TRACE("* Projection order = %s *\n", irt_top->rot_pars[desc].proj_order);
    IRT_TRACE("* Projection angles original roll %3.2f pitch %3.2f yaw %3.2f\n",
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_roll],
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_pitch],
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_yaw]);
    IRT_TRACE("* Projection angles adjusted roll %3.2f pitch %3.2f yaw %3.2f\n",
              irt_top->rot_pars[desc].irt_angles_adj[e_irt_angle_roll],
              irt_top->rot_pars[desc].irt_angles_adj[e_irt_angle_pitch],
              irt_top->rot_pars[desc].irt_angles_adj[e_irt_angle_yaw]);
    IRT_TRACE(
        "* Projection Zd = %4.2f, Wd = %4.2f *\n", irt_top->rot_pars[desc].proj_Zd, irt_top->rot_pars[desc].proj_Wd);
    IRT_TRACE("* Scaling factors: Sx = %5.2f, Sy = %5.2f *\n", irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy);
    IRT_TRACE("* Shear mode = %s *\n", irt_shr_mode_s[irt_top->rot_pars[desc].shear_mode]);
    IRT_TRACE("* Shear angles: x = %3.2f, y = %3.2f *\n",
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_x],
              irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_y]);
    IRT_TRACE("* Projection matrix\n");
    for (int i = 0; i < 3; i++)
        IRT_TRACE("* %10.4f %10.4f %10.4f\n",
                  irt_top->rot_pars[desc].proj_R[i][0],
                  irt_top->rot_pars[desc].proj_R[i][1],
                  irt_top->rot_pars[desc].proj_R[i][2]);
    IRT_TRACE("* Projection fp64 coefficients for Xi:[%10.4f %10.4f %10.4f]  Yi:[%10.4f %10.4f %10.4f]\n",
              (double)irt_top->rot_pars[desc].prj_Ad[0],
              (double)irt_top->rot_pars[desc].prj_Ad[1],
              (double)irt_top->rot_pars[desc].prj_Ad[2],
              (double)irt_top->rot_pars[desc].prj_Bd[0],
              (double)irt_top->rot_pars[desc].prj_Bd[1],
              (double)irt_top->rot_pars[desc].prj_Bd[2]);
    IRT_TRACE("*                                     [%10.4f %10.4f %10.4f]     [%10.4f %10.4f %10.4f]\n",
              (double)irt_top->rot_pars[desc].prj_Cd[0],
              (double)irt_top->rot_pars[desc].prj_Cd[1],
              (double)irt_top->rot_pars[desc].prj_Cd[2],
              (double)irt_top->rot_pars[desc].prj_Dd[0],
              (double)irt_top->rot_pars[desc].prj_Dd[1],
              (double)irt_top->rot_pars[desc].prj_Dd[2]);
    IRT_TRACE("* Projection fp32 coefficients for Xi:[%10.4f %10.4f %10.4f]  Yi:[%10.4f %10.4f %10.4f]\n",
              (double)irt_top->irt_desc[desc].prj_Af[0],
              (double)irt_top->irt_desc[desc].prj_Af[1],
              (double)irt_top->irt_desc[desc].prj_Af[2],
              (double)irt_top->irt_desc[desc].prj_Bf[0],
              (double)irt_top->irt_desc[desc].prj_Bf[1],
              (double)irt_top->irt_desc[desc].prj_Bf[2]);
    IRT_TRACE("*                                     [%10.4f %10.4f %10.4f]     [%10.4f %10.4f %10.4f]\n",
              (double)irt_top->irt_desc[desc].prj_Cf[0],
              (double)irt_top->irt_desc[desc].prj_Cf[1],
              (double)irt_top->irt_desc[desc].prj_Cf[2],
              (double)irt_top->irt_desc[desc].prj_Df[0],
              (double)irt_top->irt_desc[desc].prj_Df[1],
              (double)irt_top->irt_desc[desc].prj_Df[2]);
    IRT_TRACE("***************************\n");
}

void IRT_top::irt_mesh_desc_print(IRT_top* irt_top, uint8_t desc)
{
    IRT_TRACE("**************************\n");
    IRT_TRACE("* Mesh image parameters *\n");
    IRT_TRACE("**************************\n");
    IRT_TRACE("* ADDR  %" PRIx64 " *\n", irt_top->irt_desc[desc].image_par[MIMAGE].addr_start);
    IRT_TRACE("* W = %u, H = %u *\n",
              irt_top->irt_desc[desc].image_par[MIMAGE].W,
              irt_top->irt_desc[desc].image_par[MIMAGE].H);
    IRT_TRACE("* S = %u, Hs = %u *\n",
              irt_top->irt_desc[desc].image_par[MIMAGE].S,
              irt_top->irt_desc[desc].image_par[MIMAGE].Hs);
    IRT_TRACE("* Pw = %d, size = %dB *\n",
              8 * (1 << irt_top->irt_desc[desc].image_par[MIMAGE].Ps),
              1 << irt_top->irt_desc[desc].image_par[MIMAGE].Ps);
    IRT_TRACE("* DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_top->irt_desc[desc].image_par[MIMAGE].DataType]);
    IRT_TRACE("* Mesh point %s mesh_point_location %d\n",
              irt_mesh_fp_loc_s[irt_top->irt_desc[desc].mesh_point_location],
              irt_top->irt_desc[desc].mesh_point_location);
    if (irt_top->irt_desc[desc].irt_mode >= 4)
        IRT_TRACE("* Mesh Stripe Stride %d\n", irt_top->irt_desc[desc].mesh_stripe_stride);
    if (irt_top->irt_desc[desc].irt_mode < 4) {
        IRT_TRACE("* Mesh format %s\n", irt_mesh_format_s[irt_top->irt_desc[desc].mesh_format]);
        IRT_TRACE("* Mesh point %s\n", irt_mesh_fp_loc_s[irt_top->irt_desc[desc].mesh_point_location]);
        IRT_TRACE("* Mesh relative mode %s\n", irt_mesh_rel_mode_s[irt_top->irt_desc[desc].mesh_rel_mode]);
        IRT_TRACE(
            "* Mesh sparse h/v %d/%d\n", irt_top->irt_desc[desc].mesh_sparse_h, irt_top->irt_desc[desc].mesh_sparse_v);
        IRT_TRACE("* Mesh Gh/Gv %u/%u\n", irt_top->irt_desc[desc].mesh_Gh, irt_top->irt_desc[desc].mesh_Gv);
        IRT_TRACE("* Mesh mode = %s *\n", irt_mesh_mode_s[irt_top->rot_pars[desc].mesh_mode]);
        IRT_TRACE("* Mesh order = %s *\n", irt_mesh_order_s[irt_top->rot_pars[desc].mesh_order]);
        IRT_TRACE("* Mesh Sh/Sv %f/%f\n", irt_top->rot_pars[desc].mesh_Sh, irt_top->rot_pars[desc].mesh_Sv);
        IRT_TRACE("* Mesh distortion x/y/r %3.2f/%3.2f/%3.2f\n",
                  irt_top->rot_pars[desc].dist_x,
                  irt_top->rot_pars[desc].dist_y,
                  irt_top->rot_pars[desc].dist_r);
    }
    IRT_TRACE("***************************\n");
}

void IRT_top::irt_resamp_desc_print(IRT_top* irt_top, uint8_t desc)
{
    IRT_TRACE("**************************\n");
    IRT_TRACE("* WARP image parameters *\n");
    IRT_TRACE("**************************\n");
    IRT_TRACE("* ADDR  %" PRIx64 " *\n", irt_top->irt_desc[desc].image_par[MIMAGE].addr_start);
    IRT_TRACE("* W = %u, H = %u *\n",
              irt_top->irt_desc[desc].image_par[MIMAGE].W,
              irt_top->irt_desc[desc].image_par[MIMAGE].H);
    IRT_TRACE("* S = %u, Hs = %u *\n",
              irt_top->irt_desc[desc].image_par[MIMAGE].S,
              irt_top->irt_desc[desc].image_par[MIMAGE].Hs);
    IRT_TRACE("* Ps %d Pw = %d, size = %dB *\n",
              irt_top->irt_desc[desc].image_par[MIMAGE].Ps,
              8 * (1 << irt_top->irt_desc[desc].image_par[MIMAGE].Ps),
              1 << irt_top->irt_desc[desc].image_par[MIMAGE].Ps);
    IRT_TRACE("* DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_top->irt_desc[desc].image_par[MIMAGE].DataType]);
    IRT_TRACE("* Mesh point %s mesh_point_location %d\n",
              irt_mesh_fp_loc_s[irt_top->irt_desc[desc].mesh_point_location],
              irt_top->irt_desc[desc].mesh_point_location);
    IRT_TRACE("**************************\n");
    IRT_TRACE("* GRAD image parameters *\n");
    IRT_TRACE("**************************\n");
    IRT_TRACE("* ADDR  %" PRIx64 " *\n", irt_top->irt_desc[desc].image_par[GIMAGE].addr_start);
    IRT_TRACE("* W = %u, H = %u *\n",
              irt_top->irt_desc[desc].image_par[GIMAGE].W,
              irt_top->irt_desc[desc].image_par[GIMAGE].H);
    IRT_TRACE("* S = %u, Hs = %u *\n",
              irt_top->irt_desc[desc].image_par[GIMAGE].S,
              irt_top->irt_desc[desc].image_par[GIMAGE].Hs);
    IRT_TRACE("* Ps %d Pw = %d, size = %dB *\n",
              irt_top->irt_desc[desc].image_par[GIMAGE].Ps,
              8 * (1 << irt_top->irt_desc[desc].image_par[GIMAGE].Ps),
              1 << irt_top->irt_desc[desc].image_par[GIMAGE].Ps);
    IRT_TRACE("* DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_top->irt_desc[desc].image_par[GIMAGE].DataType]);
    IRT_TRACE("***************************\n");
}

void IRT_top::irt_desc_print(IRT_top* irt_top, uint8_t desc)
{

    IRT_TRACE("***************************\n");
    IRT_TRACE("* IRT descriptor %d *\n", desc);
    IRT_TRACE("***************************\n");
    IRT_TRACE("* Precision parameters    *\n");
    IRT_TRACE("***************************\n");
    IRT_TRACE("* COORD_WIDTH = %u, COORD_PREC  = %u, SLOPE_PREC   = %d\n",
              irt_top->rot_pars[desc].MAX_COORD_WIDTH,
              irt_top->rot_pars[desc].COORD_PREC,
              IRT_SLOPE_PREC);
    IRT_TRACE("* PIXEL_WIDTH = %u, WEIGHT_PREC = %u, WEIGHT_SHIFT = %u\n",
              irt_top->rot_pars[desc].MAX_PIXEL_WIDTH,
              irt_top->rot_pars[desc].WEIGHT_PREC,
              irt_top->rot_pars[desc].WEIGHT_SHIFT);
    IRT_TRACE("* TOTAL_PREC  = %u\n", irt_top->rot_pars[desc].TOTAL_PREC);
    IRT_TRACE("* PRJ_NOM_PREC  = %u, PRJ_DEN_PREC  = %u\n",
              irt_top->rot_pars[desc].PROJ_NOM_PREC,
              irt_top->rot_pars[desc].PROJ_DEN_PREC);
    IRT_TRACE("* Precision alignment = %u *\n", irt_top->irt_desc[desc].prec_align);

    for (uint8_t i = 0; i < 2; i++) {
        IRT_TRACE("******************************\n");
        IRT_TRACE("* %s memory parameters *\n", i == 0 ? "Rotation" : "Mesh    ");
        IRT_TRACE("******************************\n");
        IRT_TRACE("* Buf_format = %s, Buf_1B_mode = %d *\n",
                  irt_buf_format_s[irt_top->irt_cfg.buf_format[i]],
                  irt_top->irt_cfg.buf_1b_mode[i]);
        IRT_TRACE(
            "* Buf_mode = %u, BufW = %u, BufW_entries = %u \n",
            irt_top->irt_desc[desc].image_par[i == 0 ? IIMAGE : MIMAGE].buf_mode,
            irt_top->irt_cfg.rm_cfg[i][irt_top->irt_desc[desc].image_par[i == 0 ? IIMAGE : MIMAGE].buf_mode].BufW,
            irt_top->irt_cfg.rm_cfg[i][irt_top->irt_desc[desc].image_par[i == 0 ? IIMAGE : MIMAGE].buf_mode].Buf_EpL);
        IRT_TRACE(
            "* BufH = %u, BufH_mod = %u \n",
            irt_top->irt_cfg.rm_cfg[i][irt_top->irt_desc[desc].image_par[i == 0 ? IIMAGE : MIMAGE].buf_mode].BufH,
            irt_top->irt_cfg.rm_cfg[i][irt_top->irt_desc[desc].image_par[i == 0 ? IIMAGE : MIMAGE].buf_mode].BufH_mod);
        IRT_TRACE("* BufW_req = %u, BufH_req = %u, BufH_delta = %u\n",
                  i == 0 ? irt_top->rot_pars[desc].IBufW_req : irt_top->rot_pars[desc].MBufW_req,
                  i == 0 ? irt_top->rot_pars[desc].IBufH_req : irt_top->rot_pars[desc].MBufH_req,
                  i == 0 ? irt_top->rot_pars[desc].IBufH_delta : 0);
        IRT_TRACE("********************\n");
    }

    IRT_TRACE("***************************\n");
    IRT_TRACE("* Output image parameters *\n");
    IRT_TRACE("***************************\n");
    IRT_TRACE("* ADDR  %" PRIx64 " *\n", irt_top->irt_desc[desc].image_par[OIMAGE].addr_start);
    IRT_TRACE("* W = %u, H = %u *\n", irt_top->irt_desc[desc].image_par[OIMAGE].W, irt_top->irt_desc[desc].Ho);
    IRT_TRACE("* Xc = %.1f, Yc = %.1f *\n",
              (double)irt_top->irt_desc[desc].image_par[OIMAGE].Xc / 2,
              (double)irt_top->irt_desc[desc].image_par[OIMAGE].Yc / 2);
    IRT_TRACE("* S = %u, H = %u, Hs = %u *\n",
              irt_top->irt_desc[desc].image_par[OIMAGE].S,
              irt_top->irt_desc[desc].image_par[OIMAGE].H,
              irt_top->irt_desc[desc].image_par[OIMAGE].Hs);
    IRT_TRACE("* Size = %u *\n", irt_top->irt_desc[desc].image_par[OIMAGE].Size);
    IRT_TRACE("* Pw = %u, size = %dB, padding = %u, max value = %u *\n",
              irt_top->rot_pars[desc].Pwo,
              1 << irt_top->irt_desc[desc].image_par[OIMAGE].Ps,
              irt_top->irt_desc[desc].Ppo,
              irt_top->irt_desc[desc].MAX_VALo);
    if (irt_top->irt_desc[desc].irt_mode > 3)
        IRT_TRACE("* DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_top->irt_desc[desc].image_par[OIMAGE].DataType]);

    IRT_TRACE("**************************\n");
    IRT_TRACE("* Input image parameters *\n");
    IRT_TRACE("**************************\n");
    IRT_TRACE("* ADDR  %" PRIx64 " *\n", irt_top->irt_desc[desc].image_par[IIMAGE].addr_start);
    IRT_TRACE("* W = %u, H = %u *\n",
              irt_top->irt_desc[desc].image_par[IIMAGE].W,
              irt_top->irt_desc[desc].image_par[IIMAGE].H);
    IRT_TRACE("* Xc = %.1f, Yc = %.1f *\n",
              (double)irt_top->irt_desc[desc].image_par[IIMAGE].Xc / 2,
              (double)irt_top->irt_desc[desc].image_par[IIMAGE].Yc / 2);
    IRT_TRACE("* S = %u, Hs = %u *\n",
              irt_top->irt_desc[desc].image_par[IIMAGE].S,
              irt_top->irt_desc[desc].image_par[IIMAGE].Hs);
    IRT_TRACE("* Si_delta = %f, Si_factor = %f *\n",
              irt_top->rot_pars[desc].Si_delta,
              irt_top->rot_pars[desc].affine_Si_factor);
    IRT_TRACE("* Size = %u *\n", irt_top->irt_desc[desc].image_par[IIMAGE].Size);
    IRT_TRACE("* Pw = %d, size = %dB, padding = %u, mask = %x *\n",
              irt_top->rot_pars[desc].Pwi,
              1 << irt_top->irt_desc[desc].image_par[IIMAGE].Ps,
              irt_top->rot_pars[desc].Ppi,
              irt_top->irt_desc[desc].Msi);
    if (irt_top->irt_desc[desc].irt_mode > 3)
        IRT_TRACE("* DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_top->irt_desc[desc].image_par[IIMAGE].DataType]);
    IRT_TRACE("********************\n");

    IRT_TRACE("***********************\n");
    IRT_TRACE("* General  parameters *\n");
    IRT_TRACE("***********************\n");
    IRT_TRACE("* Transformation mode = %s *\n", irt_irt_mode_s[irt_top->irt_desc[desc].irt_mode]);
    IRT_TRACE("* Interpolation  mode = %s, bli_shift = %d, bli_shift_fix = %d *\n",
              irt_int_mode_s[irt_top->irt_desc[desc].int_mode],
              irt_top->irt_desc[desc].bli_shift,
              irt_top->rot_pars[desc].bli_shift_fix);
    IRT_TRACE("* Coordinate     mode = %s *\n", irt_crd_mode_s[irt_top->irt_desc[desc].crd_mode]);
    IRT_TRACE("* Flow           mode = %s *\n", irt_flow_mode_s[irt_top->irt_cfg.flow_mode]);
    IRT_TRACE("* Rate control   mode = %s, proc_size = %d, auto gen proc size = %d *\n",
              irt_rate_mode_s[irt_top->irt_desc[desc].rate_mode],
              irt_top->irt_desc[desc].proc_size,
              irt_top->rot_pars[desc].proc_auto);
    IRT_TRACE("* output wr format = %d *\n", irt_top->irt_desc[desc].oimage_line_wr_format);
    IRT_TRACE("* background mode = %d %s *\n",
              irt_top->irt_desc[desc].bg_mode,
              irt_bg_mode_s[irt_top->irt_desc[desc].bg_mode]);
    IRT_TRACE("* background = %u *\n", irt_top->irt_desc[desc].bg);
    IRT_TRACE("* resize grad en = %d *\n", irt_top->irt_desc[desc].resize_bli_grad_en);
    IRT_TRACE("* relative Mode en = %d *\n", irt_top->irt_desc[desc].mesh_rel_mode);
    IRT_TRACE("* Warp Shuffle = %d *\n", irt_top->irt_desc[desc].warp_stride);
    IRT_TRACE("* clip fp %d *\n", irt_top->irt_desc[desc].clip_fp);
    IRT_TRACE("* clip fp in inf %d *\n", irt_top->irt_desc[desc].clip_fp_inf_input);
    IRT_TRACE("* ftz en %d *\n", irt_top->irt_desc[desc].ftz_en);

    if  (irt_top->irt_desc[desc].irt_mode == e_irt_rotation ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_affine     /*&& irt_top->rot_pars[desc].affine_flags[e_irt_aff_rotation]*/) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_projection /*&& irt_top->rot_pars[desc].proj_mode == e_irt_rotation*/) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_projection /*&& irt_top->rot_pars[desc].proj_mode == e_irt_affine && irt_top->rot_pars[desc].affine_flags[e_irt_aff_rotation]*/) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_mesh       && irt_top->rot_pars[desc].mesh_mode == e_irt_rotation) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_mesh       && irt_top->rot_pars[desc].mesh_mode == e_irt_affine /*&& irt_top->rot_pars[desc].affine_flags[e_irt_aff_rotation]*/) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_mesh       && irt_top->rot_pars[desc].mesh_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_rotation) ||
		(irt_top->irt_desc[desc].irt_mode == e_irt_mesh       && irt_top->rot_pars[desc].mesh_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_affine && irt_top->rot_pars[desc].affine_flags[e_irt_aff_rotation]))
        irt_top->irt_rot_desc_print(irt_top, desc);

    if (irt_top->irt_desc[desc].irt_mode == e_irt_affine ||
        (irt_top->irt_desc[desc].irt_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_affine) ||
        (irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_affine) ||
        (irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_projection &&
         irt_top->rot_pars[desc].proj_mode == e_irt_affine))
        irt_top->irt_aff_desc_print(irt_top, desc);

    if (irt_top->irt_desc[desc].irt_mode == e_irt_projection ||
        (irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_projection))
        irt_top->irt_prj_desc_print(irt_top, desc);

    if (irt_top->irt_desc[desc].irt_mode == e_irt_mesh || irt_top->irt_desc[desc].irt_mode == e_irt_rescale)
        irt_top->irt_mesh_desc_print(irt_top, desc);

    if (irt_top->irt_desc[desc].irt_mode == e_irt_resamp_fwd || irt_top->irt_desc[desc].irt_mode == e_irt_resamp_bwd1 ||
        irt_top->irt_desc[desc].irt_mode == e_irt_resamp_bwd2)
        irt_top->irt_resamp_desc_print(irt_top, desc);

    if (irt_top->irt_desc[desc].irt_mode <= e_irt_mesh || irt_top->irt_desc[desc].irt_mode == e_irt_rescale) {
        IRT_TRACE("********************\n");
        IRT_TRACE("* Rotation dir     %s *\n", irt_top->irt_desc[desc].rot_dir == IRT_ROT_DIR_POS ? "POS" : "NEG");
        IRT_TRACE(
            "* Read flip[h/v]   [%d/%d] *\n", irt_top->irt_desc[desc].read_hflip, irt_top->irt_desc[desc].read_vflip);
        IRT_TRACE("* Rot90 %d inth %d intv %d *\n",
                  irt_top->irt_desc[desc].rot90,
                  irt_top->irt_desc[desc].rot90_inth,
                  irt_top->irt_desc[desc].rot90_intv);
        IRT_TRACE("* Input read slope %6.2f %d *\n",
                  irt_top->rot_pars[desc].im_read_slope,
                  irt_top->irt_desc[desc].im_read_slope);
        IRT_TRACE("* Xo_first/last  %15.8f / %15.8f *\n",
                  (double)irt_top->rot_pars[desc].Xo_first,
                  (double)irt_top->rot_pars[desc].Xo_last);
        IRT_TRACE("* Yo_first/last  %15.8f / %15.8f *\n",
                  (double)irt_top->rot_pars[desc].Yo_first,
                  (double)irt_top->rot_pars[desc].Yo_last);
        IRT_TRACE("* Xi_first/last  %15.8f / %15.8f *\n",
                  (double)irt_top->rot_pars[desc].Xi_first,
                  (double)irt_top->rot_pars[desc].Xi_last);
        IRT_TRACE("* Yi_first/last  %15.8f / %15.8f *\n",
                  (double)irt_top->rot_pars[desc].Yi_first,
                  (double)irt_top->rot_pars[desc].Yi_last);
        IRT_TRACE("* Xi_first_fixed %15.8f (%" PRId64 ") *\n",
                  irt_top->irt_desc[desc].Xi_first_fixed / pow(2.0, IRT_SLOPE_PREC),
                  irt_top->irt_desc[desc].Xi_first_fixed);
        IRT_TRACE("* Yi_first_fixed %15.8f (%" PRId64 ") *\n",
                  irt_top->irt_desc[desc].Yi_first_fixed / pow(2.0, IRT_SLOPE_PREC),
                  irt_top->irt_desc[desc].Yi_first_fixed);
        IRT_TRACE("* Xi_start		%d *\n", irt_top->rot_pars[desc].Xi_start);
        IRT_TRACE("* Yi_start/end   %d / %d *\n", irt_top->irt_desc[desc].Yi_start, irt_top->irt_desc[desc].Yi_end);
        IRT_TRACE("* Xi_start_offset/Xi_start_offset_flip %d / %d *\n",
                  irt_top->irt_desc[desc].Xi_start_offset,
                  irt_top->irt_desc[desc].Xi_start_offset_flip);
        IRT_TRACE("********************\n");
    }
}

/******************/
/* Resets section */
/******************/
void IRT_top::pars_reset()
{
    // irt_cfg.BufW = 0; irt_cfg.BufW_entries = 0; irt_cfg.BufH_mod = 0; irt_cfg.BufH = 0;
    // memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    memset(&rot_pars, 0, MAX_TASKS * sizeof(rotation_par));

    for (uint8_t i = 0; i < MAX_TASKS; i++) {
        irt_par_init(rot_pars[i]);
    }
}

void IRT_top::desc_reset()
{
    // irt_cfg.BufW = 0; irt_cfg.BufW_entries = 0; irt_cfg.BufH_mod = 0; irt_cfg.BufH = 0;
    // memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    memset(irt_desc, 0, MAX_TASKS * sizeof(irt_desc_par));
    num_of_tasks = 0;
}
//---------------------------------------
// ROTSIM_DV_INTGR -- ADDED TO ALLOW SINGLE DESCRIPTOR SUPPORT
void IRT_top::reset_done_status()
{
    memset(&irt_done, 0, sizeof(irt_done));
}
//---------------------------------------

void IRT_top::reset(bool reset_subblocks)
{

    // descriptor reset
    // desc_reset();

    // global variables reset
    memset(&mem_ctrl, 0, sizeof(mem_ctrl));
    memset(&task, 0, sizeof(task));

    // local variables reset
    pixel_valid = 0;
    pixel_valid = 0;
    memset(&irt_done, 0, sizeof(irt_done));
    ofifo_pop          = 0;
    task_end           = 0;
    task_end           = 0;
    ofifo_empty        = 1;
    ofifo_full         = 0;
    cfifo_fullness     = 0;
    cfifo_emptyness    = 0;
    ififo_pop          = 0;
    ififo_read         = 0;
    ififo_full         = 0;
    ififo_empty        = 0;
    mfifo_pop          = 0;
    mfifo_read         = 0;
    mfifo_full         = 0;
    mfifo_empty        = 0;
    mesh_pixel_valid   = 0;
    mesh_task_end      = 0;
    oicc_task          = 0;
    micc_task          = 0;
    iirc_task          = 0;
    adj_proc_size      = 0;
    mesh_adj_proc_size = 0;
    memset(&irt_top_sig, 0, sizeof(irt_top_sig));
    memset(irt_done, 0, sizeof(irt_done));
    memset(rm_rd_mode, 0, sizeof(rm_rd_mode));
    memset(mm_rd_mode, 0, sizeof(mm_rd_mode));
    memset(ird_shift, 0, sizeof(ird_shift));
    memset(mrd_shift, 0, sizeof(mrd_shift));
    memset(rm_bg_flags, 0, sizeof(rm_bg_flags));
    memset(cfifo_data_in, 0, sizeof(cfifo_data_in));
    memset(cfifo_data_out, 0, sizeof(cfifo_data_out));

    // memset(mesh_ofifo, 0, sizeof(mesh_ofifo));
    mesh_ofifo_valid = 0;

    // functions reset
    if (reset_subblocks) {
        IRT_IIRM_block->reset();
        IRT_IFIFO_block->reset();
        IRT_RMWM_block->reset();
        IRT_RMEM_block->reset();
        IRT_OICC_block->reset();
        IRT_IIRC_block->reset();
        IRT_OFIFO_block->reset();
        IRT_OIWM_block->reset();
        IRT_MIRM_block->reset();
        IRT_MFIFO_block->reset();
        IRT_MMWM_block->reset();
        IRT_MMEM_block->reset();
        IRT_MICC_block->reset();
        IRT_MIRC_block->reset();
        // IRT_MEXP_block->reset();

        IRT_OICC2_block->reset();
        IRT_IIRC2_block->reset();
        IRT_MICC2_block->reset();
        IRT_RESAMP_block->reset();
    }
}

void IRT_top::IRT_RESAMP::reset()
{
    task_start = 0;
}

template <Eirt_block_type block_type>
void IRT_top::IRT_IRM<block_type>::reset()
{

    Eirt_blocks_enum task_type = block_type == e_irt_block_rot ? e_irt_irm : e_irt_mrm;

    tasks_start = 0;
    task_start  = 0;
    line_start  = 1;
    mem_rd_int  = 0;

    Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);

    // line = irt_top->irt_desc[irt_top->task[e_irt_irm]].Yi_start;
    line           = Yi_start;
    Xi_start       = 0;
    Xi_start_fixed = 0;
    pixel          = 0;
}

template <class BUS_OUT, Eirt_block_type block_type>
void IRT_top::IRT_IFIFO<BUS_OUT, block_type>::reset()
{
    irt_ififo_wp      = 0;
    irt_ififo_rp      = 0;
    irt_ififo_r_entry = 0;
    fifo_fullness     = 0;
    memset(&irt_ififo, 0, sizeof(irt_ififo));
    memset(&irt_ififo_meta, 0, sizeof(irt_ififo_meta));
}

template <class BUS_IN,
          class BUS_OUT,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_HEIGHT,
          Eirt_block_type block_type>
void IRT_top::IRT_RMWM<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_HEIGHT, block_type>::reset()
{

    Eirt_blocks_enum task_type = block_type == e_irt_block_rot ? e_irt_rmwm : e_irt_mmwm;

    tasks_start = 0;
    pixel       = 0;
    line_start  = 0;
    task_start  = 0;
    done        = 0;
    // line = irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Yi_start;

    Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);

    line = Yi_start;
}

template <class BUS,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_WIDTH,
          uint16_t        BANK_HEIGHT,
          Eirt_block_type block_type>
void IRT_top::IRT_ROT_MEM<BUS, ROW_BANKS, COL_BANKS, BANK_WIDTH, BANK_HEIGHT, block_type>::reset()
{
    task_int = 0;
    memset(&rot_mem, 0, sizeof(rot_mem));
    memset(&meta_mem, 0, sizeof(meta_mem));
    for (uint8_t row_banks = 0; row_banks < ROW_BANKS; row_banks++) {
        for (uint8_t col_banks = 0; col_banks < COL_BANKS; col_banks++) {
            for (uint16_t bank_height = 0; bank_height < BANK_HEIGHT; bank_height++) {
                for (uint16_t bank_width = 0; bank_width < BANK_WIDTH; bank_width++) {
#if defined(STANDALONE_ROTATOR) //|| defined (RUN_WITH_SV) ROTSIM_DV_INTGR
                    rot_mem[row_banks][col_banks][bank_height].pix[bank_width] = rand() % 256;
#else
                    rot_mem[row_banks][col_banks][bank_height].pix[bank_width] = 0;
#endif
                }
            }
        }
    }
}

template <uint16_t ROW_BANKS, Eirt_block_type block_type>
void IRT_top::IRT_OICC<ROW_BANKS, block_type>::reset()
{
    line       = 0;
    pixel      = 0;
    YT_min     = 0;
    line_start = 1;
    task_done  = 0;

    Eirt_blocks_enum task_type = block_type == e_irt_block_rot ? e_irt_oicc : e_irt_micc;

    Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
}

template <Eirt_block_type block_type>
void IRT_top::IRT_OICC2<block_type>::reset()
{
    line       = 0;
    pixel      = 0;
    line_start = 1;
    task_done  = 0;
}

template <Eirt_block_type block_type>
void IRT_top::IRT_CFIFO<block_type>::reset()
{
    cfifo_wp       = 0;
    cfifo_rp       = 0;
    cfifo_fullness = 0;
    memset(&cfifo, 0, sizeof(cfifo));
}

template <Eirt_block_type block_type>
void IRT_top::IRT_IIIRC<block_type>::reset()
{
    error_flag = 0;
}

template <Eirt_block_type block_type>
void IRT_top::IRT_IIIRC2<block_type>::reset()
{
    line       = 0;
    pixel      = 0;
    YT_min     = 0;
    line_start = 1;
    task_done  = 0;

    Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[e_irt_iiirc]], block_type);

    error_flag   = 0;
    YT_min       = 0;
    line_start   = 1;
    task_done    = 0;
    timeout_cntr = 0;
}

template <uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, Eirt_block_type block_type>
void IRT_top::IRT_RMRM<ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>::reset()
{
}

template <class BUS_IN,
          class BUS_OUT,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_WIDTH,
          Eirt_block_type block_type>
void IRT_top::IRT_8x16_IWC<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>::reset()
{
}

template <class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, Eirt_block_type block_type>
void IRT_top::IRT_2x2_sel<BUS_IN, BUS_OUT, ROW_BANKS, block_type>::reset()
{
}

template <class BUS_IN, class BUS_OUT, Eirt_block_type block_type>
void IRT_top::IRT_BLI<BUS_IN, BUS_OUT, block_type>::reset()
{
}

void IRT_top::IRT_OFIFO::reset()
{
    irt_ofifo_wp    = 0;
    irt_ofifo_rp    = 0;
    fullness        = 0;
    irt_ofifo_w_cnt = 0;
    pixel_cnt       = 0;
    memset(irt_ofifo, 0, sizeof(irt_ofifo));
}

void IRT_top::IRT_OWM::reset()
{
    tasks_start = 0;
    task_start  = 0;
    done        = 0; //, line_start=1;
    line        = 0;
    pixel       = 0;
}

template <Eirt_block_type block_type>
IRT_top::IRT_IRM<block_type>::IRT_IRM(IRT_top* irt_top1)
{

    irt_top = irt_top1;
    reset();

#ifdef CREATE_IMAGE_DUMPS_IRM
    irm_rd_image = new uint16_t**[PLANES];
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        irm_rd_image[plain] = new uint16_t*[IMAGE_H];
        for (uint16_t row = 0; row < IMAGE_H; row++) {
            irm_rd_image[plain][row] = new uint16_t[IMAGE_W];
        }
    }
#endif
}

template <Eirt_block_type block_type>
IRT_top::IRT_IRM<block_type>::~IRT_IRM()
{

#ifdef CREATE_IMAGE_DUMPS_IRM
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        for (uint16_t row = 0; row < IMAGE_H; row++) {
            delete[] irm_rd_image[plain][row];
            irm_rd_image[plain][row] = nullptr;
        }
        delete[] irm_rd_image[plain];
        irm_rd_image[plain] = nullptr;
    }
    delete[] irm_rd_image;
    irm_rd_image = nullptr;
#endif
}

IRT_top::IRT_OWM::IRT_OWM(IRT_top* irt_top1)
{

    irt_top = irt_top1;
    reset();

#ifdef CREATE_IMAGE_DUMPS_OWM
    owm_wr_image = new uint16_t**[PLANES];
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        owm_wr_image[plain] = new uint16_t*[IMAGE_H];
        for (uint16_t row = 0; row < IMAGE_H; row++) {
            owm_wr_image[plain][row] = new uint16_t[IMAGE_W];
        }
    }
#endif
}

IRT_top::IRT_OWM::~IRT_OWM()
{

#ifdef CREATE_IMAGE_DUMPS_OWM
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        for (uint16_t row = 0; row < IMAGE_H; row++) {
            delete[] owm_wr_image[plain][row];
            owm_wr_image[plain][row] = nullptr;
        }
        delete[] owm_wr_image[plain];
        owm_wr_image[plain] = nullptr;
    }
    delete[] owm_wr_image;
    owm_wr_image = nullptr;
#endif
}

IRT_top::IRT_top(bool enable_mesh_mode) : m_enable_mesh_mode(enable_mesh_mode)
{

    memset(&irt_cfg, 0, sizeof(irt_cfg_pars));
    pars_reset();
    desc_reset();
    num_of_tasks = 0;
    memset(&task, 0, sizeof(task));
    memset(&irt_top_sig, 0, sizeof(irt_ext_sig));
    memset(&mem_ctrl, 0, sizeof(mem_ctrl));
    memset(&mesh_ofifo, 0, sizeof(bus8XiYi_float_struct));
    mesh_ofifo_valid = 0;

    pixel_valid      = 0;
    mesh_pixel_valid = 0;
    memset(&irt_done, 0, sizeof(irt_done));
    memset(&rm_rd_mode, 0, sizeof(rm_rd_mode));
    memset(&mm_rd_mode, 0, sizeof(mm_rd_mode));
    memset(&rm_bg_flags, 0, sizeof(rm_bg_flags));
    memset(&mm_bg_flags, 0, sizeof(mm_bg_flags));
    memset(&ird_shift, 0, sizeof(ird_shift));
    memset(&mrd_shift, 0, sizeof(mrd_shift));
    /*ofifo_push=0,*/ ofifo_pop                                      = 0;
    task_end                                                         = 0;
    mesh_task_end                                                    = 0;
    ofifo_empty                                                      = 0;
    ofifo_full                                                       = 0;
    ififo_full                                                       = 0;
    ififo_empty                                                      = 0;
    ififo_pop                                                        = 0;
    ififo_read                                                       = 0;
    mfifo_full                                                       = 0;
    mfifo_empty                                                      = 0;
    mfifo_pop                                                        = 0;
    mfifo_read                                                       = 0;
    cfifo_fullness                                                   = 0;
    cfifo_emptyness                                                  = 0;
    Xo0                                                              = 0;
    Yo                                                               = 0;
    YT_bank_row                                                      = 0;
    YmT_bank_row                                                     = 0; // XL, XR, YT, YB, Xi0_fixed, Yi0_fixed;
    Xm0                                                              = 0;
    Ym                                                               = 0;
    /*rm_Xi_start_addr, moved to external struct signals*/ oicc_task = 0;
    micc_task                                                        = 0;
    iirc_task                                                        = 0;
    adj_proc_size                                                    = 0;
    mesh_adj_proc_size                                               = 0;
    memset(&cfifo_data_in, 0, sizeof(irt_cfifo_data_struct));
    memset(&cfifo_data_out, 0, sizeof(irt_cfifo_data_struct));
    first_call = 0;

    reset(false);

    IRT_IIRM_block  = new IRT_IRM<e_irt_block_rot>(this);
    IRT_IFIFO_block = new IRT_IFIFO<bus32B_struct, e_irt_block_rot>(this);
    IRT_RMWM_block  = new IRT_RMWM<bus32B_struct,
                                  bus16B_struct,
                                  IRT_ROT_MEM_ROW_BANKS,
                                  IRT_ROT_MEM_COL_BANKS,
                                  IRT_ROT_MEM_BANK_HEIGHT,
                                  e_irt_block_rot>(this);
    IRT_RMEM_block  = new IRT_ROT_MEM<bus16B_struct,
                                     IRT_ROT_MEM_ROW_BANKS,
                                     IRT_ROT_MEM_COL_BANKS,
                                     IRT_ROT_MEM_BANK_WIDTH,
                                     IRT_ROT_MEM_BANK_HEIGHT,
                                     e_irt_block_rot>(this);
    IRT_OICC_block  = new IRT_OICC<IRT_ROT_MEM_ROW_BANKS, e_irt_block_rot>(this);
    IRT_CFIFO_block = new IRT_CFIFO<e_irt_block_rot>(this);
    IRT_IIRC_block  = new IRT_IIIRC<e_irt_block_rot>(this);
    IRT_RMRM_block =
        new IRT_RMRM<IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH, e_irt_block_rot>(this);
    IRT_IIWC_block  = new IRT_8x16_IWC<bus16B_struct,
                                      bus16ui16_struct,
                                      IRT_ROT_MEM_ROW_BANKS,
                                      IRT_ROT_MEM_COL_BANKS,
                                      IRT_ROT_MEM_BANK_WIDTH,
                                      e_irt_block_rot>(this);
    IRT_I2x2_block  = new IRT_2x2_sel<bus16ui16_struct, p2x2_ui16_struct, IRT_ROT_MEM_ROW_BANKS, e_irt_block_rot>(this);
    IRT_IBLI_block  = new IRT_BLI<p2x2_ui16_struct, uint16_t, e_irt_block_rot>(this);
    IRT_OFIFO_block = new IRT_OFIFO(this);
    IRT_OIWM_block  = new IRT_OWM(this);

    IRT_MIRM_block  = new IRT_IRM<e_irt_block_mesh>(this);
    IRT_MFIFO_block = new IRT_IFIFO<bus128B_struct, e_irt_block_mesh>(this);
    IRT_MMWM_block  = new IRT_RMWM<bus128B_struct,
                                  bus64B_struct,
                                  IRT_MESH_MEM_ROW_BANKS,
                                  IRT_MESH_MEM_COL_BANKS,
                                  IRT_MESH_MEM_BANK_HEIGHT,
                                  e_irt_block_mesh>(this);
    IRT_MMEM_block  = new IRT_ROT_MEM<bus64B_struct,
                                     IRT_MESH_MEM_ROW_BANKS,
                                     IRT_MESH_MEM_COL_BANKS,
                                     IRT_MESH_MEM_BANK_WIDTH,
                                     IRT_MESH_MEM_BANK_HEIGHT,
                                     e_irt_block_mesh>(this);
    IRT_MICC_block  = new IRT_OICC<IRT_MESH_MEM_ROW_BANKS, e_irt_block_mesh>(this);
    IRT_MIRC_block  = new IRT_IIIRC<e_irt_block_mesh>(this);
    IRT_MMRM_block =
        new IRT_RMRM<IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_WIDTH, e_irt_block_mesh>(this);
    IRT_MIWC_block = new IRT_8x16_IWC<bus64B_struct,
                                      bus16ui64_struct,
                                      IRT_MESH_MEM_ROW_BANKS,
                                      IRT_MESH_MEM_COL_BANKS,
                                      IRT_MESH_MEM_BANK_WIDTH,
                                      e_irt_block_mesh>(this);
    IRT_M2x2_block =
        new IRT_2x2_sel<bus16ui64_struct, p2x2_ui64_struct, IRT_MESH_MEM_ROW_BANKS, e_irt_block_mesh>(this);
    IRT_MBLI_block = new IRT_BLI<p2x2_fp32_struct, float, e_irt_block_mesh>(this);
    // IRT_MEXP_block = new IRT_MEXP(this);

    IRT_OICC2_block = new IRT_OICC2<e_irt_block_rot>(this);
    IRT_IIRC2_block = new IRT_IIIRC2<e_irt_block_rot>(this);
    IRT_IIWC2_block = new IRT_8x16_IWC<bus16B_struct,
                                       bus16ui16_struct,
                                       IRT_MESH_MEM_ROW_BANKS,
                                       IRT_MESH_MEM_COL_BANKS,
                                       IRT_ROT_MEM_BANK_WIDTH,
                                       e_irt_block_rot>(this);
    IRT_I2x22_block =
        new IRT_2x2_sel<bus16ui16_struct, p2x2_ui16_struct, IRT_MESH_MEM_ROW_BANKS, e_irt_block_rot>(this);

    IRT_MICC2_block = new IRT_OICC2<e_irt_block_mesh>(this);

    IRT_RESAMP_block = new IRT_RESAMP(this);

    irt_cfg_init(irt_cfg);
    if (m_enable_mesh_mode) {
        irt_mesh_mems_create(irt_cfg.mesh_images);
    }
}

IRT_top::~IRT_top()
{

    delete IRT_IIRM_block;
    delete IRT_IFIFO_block;
    delete IRT_RMWM_block;
    delete IRT_RMEM_block;
    delete IRT_OICC_block;
    delete IRT_CFIFO_block;
    delete IRT_IIRC_block;
    delete IRT_RMRM_block;
    delete IRT_IIWC_block;
    delete IRT_I2x2_block;
    delete IRT_IBLI_block;
    delete IRT_OFIFO_block;
    delete IRT_OIWM_block;

    delete IRT_MIRM_block;
    delete IRT_MFIFO_block;
    delete IRT_MMWM_block;
    delete IRT_MMEM_block;
    delete IRT_MICC_block;
    delete IRT_MIRC_block;
    delete IRT_MMRM_block;
    delete IRT_MIWC_block;
    delete IRT_M2x2_block;
    delete IRT_MBLI_block;

    delete IRT_OICC2_block;
    delete IRT_IIRC2_block;
    delete IRT_IIWC2_block;
    delete IRT_I2x22_block;
    delete IRT_MICC2_block;

    delete IRT_RESAMP_block;

    if (m_enable_mesh_mode) {
        irt_mesh_mems_delete(irt_cfg.mesh_images);
    }
}

void IRT_top::setInstanceName(const std::string& str)
{
    m_name = str;

    IRT_IIRM_block->setInstanceName(str + ".IRT_IIRM");
    IRT_IFIFO_block->setInstanceName(str + ".IRT_IFIFO");
    IRT_RMWM_block->setInstanceName(str + ".IRT_RMWM");
    IRT_RMEM_block->setInstanceName(str + ".IRT_RMEM");
    IRT_OICC_block->setInstanceName(str + ".IRT_OICC");
    IRT_CFIFO_block->setInstanceName(str + ".IRT_CFIFO");
    IRT_IIRC_block->setInstanceName(str + ".IRT_IIIRC");
    IRT_RMRM_block->setInstanceName(str + ".IRT_RMRM");
    IRT_IIWC_block->setInstanceName(str + ".IRT_IIWC");
    IRT_I2x2_block->setInstanceName(str + ".IRT_I2x2");
    IRT_IBLI_block->setInstanceName(str + ".IRT_IBLI");
    IRT_OFIFO_block->setInstanceName(str + ".IRT_OFIFO");
    IRT_OIWM_block->setInstanceName(str + ".IRT_OIWM");

    IRT_MIRM_block->setInstanceName(str + ".IRT_MIRM");
    IRT_MFIFO_block->setInstanceName(str + ".IRT_MFIFO");
    IRT_MMWM_block->setInstanceName(str + ".IRT_MMWM");
    IRT_MMEM_block->setInstanceName(str + ".IRT_MMEM");
    IRT_MICC_block->setInstanceName(str + ".IRT_MICC");
    IRT_MIRC_block->setInstanceName(str + ".IRT_MIRC");
    IRT_MMRM_block->setInstanceName(str + ".IRT_MMRM");
    IRT_MIWC_block->setInstanceName(str + ".IRT_MIWC");
    IRT_M2x2_block->setInstanceName(str + ".IRT_M2x2");
    IRT_MBLI_block->setInstanceName(str + ".IRT_MBLI");
    // IRT_MEXP_block->setInstanceName(str + ".IRT_MEXP");

    IRT_OICC2_block->setInstanceName(str + ".IRT_OICC2");
    IRT_IIRC2_block->setInstanceName(str + ".IRT_IIRC2");
    IRT_IIWC2_block->setInstanceName(str + ".IRT_IIWC2");
    IRT_I2x22_block->setInstanceName(str + ".IRT_I2x22");
    IRT_MICC2_block->setInstanceName(str + ".IRT_MICC2");

    IRT_RESAMP_block->setInstanceName(str + ".IRT_RESAMP");
}

template <Eirt_block_type block_type>
bool IRT_top::IRT_IRM<block_type>::run(uint64_t&             addr,
                                       meta_data_struct&     meta_out,
                                       bool&                 mem_rd,
                                       const bus128B_struct& rd_data,
                                       meta_data_struct      meta_in,
                                       bus128B_struct&       fifo_wr_data,
                                       bool&                 fifo_push,
                                       meta_data_struct&     fifo_meta_in,
                                       bool                  fifo_full,
                                       uint16_t&             lsb_pad,
                                       uint16_t&             msb_pad)
{

    Eirt_blocks_enum task_type  = block_type == e_irt_block_rot ? e_irt_irm : e_irt_mrm;
    uint8_t          image_type = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    std::string      my_string  = getInstanceName();
    uint16_t         pad1;
    int32_t          byte_start;

    irt_top->irt_top_sig.irm_rd_first[block_type] = 0;
    irt_top->irt_top_sig.irm_rd_last[block_type]  = 0;
    if (irt_top->task[task_type] >= irt_top->num_of_tasks) {
        mem_rd_int = 0;
        mem_rd     = 0;
        fifo_push  = 0;
        return 1; // TBD - removed after adding multiple tasks support
    }
    //---------------------------------------

    //	IRT_TRACE_TO_LOG(3,IRTLOG,"%s num_of_tasks %d cycle %d\n", m_name.c_str(), irt_top->num_of_tasks, cycle);
    if (block_type == e_irt_block_mesh) {
        // irt_top->task[task_type]++;
        // return 0;
        if (irt_top->irt_desc[irt_top->task[task_type]].irt_mode !=
            e_irt_mesh) { // IRM works as MRM and task is not mesh, droppped
            //			IRT_TRACE_TO_LOG(3,IRTLOG,"%s task %d is not mesh task and dropped at cycle %d\n",
            // m_name.c_str(), irt_top->task[task_type], cycle);
            irt_top->task[task_type]++;
            mem_rd_int = 0;
            mem_rd     = 0;
            fifo_push  = 0;
            return 0;
        }
    }

#ifdef CREATE_IMAGE_DUMPS_IRM
    int width, height;
    if (block_type == e_irt_block_rot) {
        width  = irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].S;
        height = irt_top->irt_desc[irt_top->task[task_type]].Yi_end - Yi_start + 1;
    } else {
        width  = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].W;
        height = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H;
    }
    int row_padded = (width * 3 + 3) & (~3);
#endif

    // ififo write
    // if (block_type == e_irt_block_rot) {
    //   IRT_TRACE("%s task %d rd_data\n", m_name.c_str(), irt_top->task[task_type]);
    //}
    for (uint8_t byte = 0; byte < IRT_IFIFO_WIDTH; byte++) {
        // if (block_type == e_irt_block_rot) {
        //   IRT_TRACE("%x ",rd_data.pix[byte]);
        //}
        if (irt_top->irt_desc[irt_top->task[task_type]].read_hflip == 0 || block_type == e_irt_block_mesh) {
            fifo_wr_data.pix[byte] = rd_data.pix[byte];
        } else {
            if (irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].Ps == 0) { // 1B/pixel
                fifo_wr_data.pix[byte] = rd_data.pix[IRT_IFIFO_WIDTH - 1 - byte];
            } else { // 2B pixel
                fifo_wr_data.pix[byte] = rd_data.pix[IRT_IFIFO_WIDTH - 2 - byte + 2 * (byte & 1)];
            }
        }
    }
    // if (block_type == e_irt_block_rot) {
    //      IRT_TRACE("\n");
    //}
    fifo_meta_in.line     = meta_in.line;
    fifo_meta_in.Xi_start = meta_in.Xi_start;
    fifo_meta_in.task     = meta_in.task;
    fifo_push             = mem_rd_int;
    if (mem_rd_int == 1) {
        if (block_type == e_irt_block_rot)
            IRT_TRACE_TO_LOG(2,
                             IRTLOG,
                             "%s read: task %d, line %d, pixel %d from Xi_start %d at cycle %d\n",
                             my_string.c_str(),
                             irt_top->task[task_type],
                             line,
                             pixel,
                             meta_in.Xi_start,
                             cycle);

#ifdef CREATE_IMAGE_DUMPS_IRM
        uint16_t pix_val, pix_offset;
        if (block_type == e_irt_block_rot) {
            for (int pix = 0; pix < (128 >> irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].Ps); pix++) {
                if (irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].Ps == 0)
                    pix_val = (uint16_t)rd_data.pix[pix];
                else
                    pix_val = (rd_data.pix[2 * pix + 1] << 8) + rd_data.pix[2 * pix];
                irm_rd_image[irt_top->task[e_irt_irm]][line - Yi_start]
                            [(pixel >> irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].Ps) + pix] =
                                (unsigned char)(pix_val >> (irt_top->rot_pars[irt_top->task[e_irt_irm]].Pwi - 8));
            }
        } else {
            for (int pix = 0; pix < (128 >> irt_top->irt_desc[irt_top->task[e_irt_mrm]].image_par[MIMAGE].Ps); pix++) {
                pix_offset = pix << irt_top->irt_desc[irt_top->task[e_irt_mrm]].image_par[MIMAGE].Ps;
                if (irt_top->irt_desc[irt_top->task[e_irt_mrm]].mesh_format == e_irt_mesh_flex) {
                    fprintf(irm_out_file,
                            "%02x%02x%02x%02x\n",
                            (rd_data.pix[pix_offset + 3] & 0xff),
                            (rd_data.pix[pix_offset + 2] & 0xff),
                            (rd_data.pix[pix_offset + 1] & 0xff),
                            (rd_data.pix[pix_offset] & 0xff));
                } else {
                    fprintf(irm_out_file,
                            "%02x%02x%02x%02x%02x%02x%02x%02x\n",
                            rd_data.pix[pix_offset + 7],
                            rd_data.pix[pix_offset + 6],
                            rd_data.pix[pix_offset + 5],
                            rd_data.pix[pix_offset + 4],
                            rd_data.pix[pix_offset + 3],
                            rd_data.pix[pix_offset + 2],
                            rd_data.pix[pix_offset + 1],
                            rd_data.pix[pix_offset]);
                }
            }
        }
#endif
        pixel += IRT_IFIFO_WIDTH;
    }

    if (tasks_start == 0) {
        if (block_type == e_irt_block_rot) {
            IRT_TRACE("********************\n");
            IRT_TRACE("* Running %s IRT model version %d from %s \n", irt_h5_mode ? "H5" : "H6", IRT_VERSION, IRT_DATE);
            IRT_TRACE("********************\n");
        }
#ifdef CREATE_IMAGE_DUMPS_IRM
        if (block_type == e_irt_block_rot) {
            // irm_out_file = fopen("irt_irm_out.bmp", "wb");
            // IRT_TRACE("IRT IRM output file: Width[%d], Height[%d], Size[%d]\n", width, height, 54 + 3 * width *
            // height); generate_bmp_header(irm_out_file, width, height);
        } else {
            irm_out_file = fopen("irt_mrm_out.txt", "w");
            // IRT_TRACE("IRT MRM output file: Width[%d], Height[%d], Size[%d]\n", width, height, width * height);
        }
#endif
        tasks_start = 1;
    }

    if (task_start == 0) {
#if 0
		IRT_TRACE("********************\n");
		if (super_irt == 1) {
			IRT_TRACE("* Running SUPER IRT model *\n");
		} else {
			IRT_TRACE("* Running REG IRT model *\n");
		}
		IRT_TRACE("* %s task %d *\n", my_string.c_str(), irt_top->task[task_type]);
		IRT_TRACE("********************\n");
#endif
        if (block_type == e_irt_block_rot/* && ((irt_multi_image == 0 && irt_top->task[task_type] == 0) || (irt_multi_image == 1 && ((irt_top->task[task_type] % PLANES) == 0)))*/) {//ROTSIM_DV_INTGR
            irt_top->irt_desc_print(irt_top, irt_top->task[task_type]);
        }

        task_start = 1;
    }

    uint16_t pixel_BufW, pixel_max;
    if (block_type == e_irt_block_rot) {
        pixel_BufW =
            irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].buf_mode]
                .BufW >>
            irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].Ps; // in pixels
        // pixel_max = (read_iimage_BufW) ? pixel_BufW :
        // irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].S;
        pixel_max = (read_iimage_BufW) ? pixel_BufW
                                       : (irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].S + 0 * 1 +
                                          irt_top->irt_desc[irt_top->task[task_type]].Xi_start_offset +
                                          irt_top->irt_desc[irt_top->task[task_type]].Xi_start_offset_flip *
                                              irt_top->irt_desc[irt_top->task[task_type]].read_hflip);
        if (irt_h5_mode && pixel_max > 256) {
            pixel_max = 256;
        }
    } else {
        pixel_BufW =
            irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].buf_mode]
                .BufW >>
            irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].Ps; // in pixels
        pixel_max = (read_mimage_BufW) ? pixel_BufW : irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].S;
    }

    // ext memory read
    mem_rd_int = 0;
    if (fifo_full == 0) {
        mem_rd_int = 1;
        IRT_TRACE_TO_LOG(2,
                         IRTLOG,
                         "%s: task %d, line %d, Xi_start %d, pixel %d, addr %llx, image width %d\n",
                         my_string.c_str(),
                         irt_top->task[task_type],
                         line,
                         Xi_start,
                         pixel,
                         addr,
                         irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].W);
        if (pixel >= /*Xi_start +*/ (pixel_max << irt_top->irt_desc[irt_top->task[task_type]]
                                                      .image_par[image_type]
                                                      .Ps)) { // image_pars[IIMAGE].S) {//line read is finished
            line++;
            line_start = 1;
            if (((block_type == e_irt_block_rot) && (line == irt_top->irt_desc[irt_top->task[e_irt_irm]].Yi_end + 1)) ||
                ((block_type == e_irt_block_mesh) &&
                 (line == irt_top->irt_desc[irt_top->task[e_irt_mrm]].image_par[MIMAGE].H))) { // task is finished
                IRT_TRACE("%s task %d is finished at cycle %d\n", m_name.c_str(), irt_top->task[task_type], cycle);
                irt_top->task[task_type]++;
                task_start = 0;
                Yi_start   = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
                line       = Yi_start;

                if (irt_top->task[task_type] == irt_top->num_of_tasks ||
                    (block_type == e_irt_block_mesh &&
                     irt_top->irt_desc[irt_top->task[task_type]].irt_mode != e_irt_mesh)) {
                    //		         IRT_TRACE("%s: task %d, num_of_tasks %d\n",
                    // my_string.c_str(),irt_top->task[task_type],irt_top->num_of_tasks);
                    mem_rd_int = 0;
                    mem_rd     = 0;
                }

                if (irt_top->task[task_type] == irt_top->num_of_tasks) {
#ifdef CREATE_IMAGE_DUMPS_IRM
                    if (block_type == e_irt_block_rot) {
                        char irm_out_file_name[150];
                        sprintf(irm_out_file_name, "irt_irm_out.bmp");
                        WriteBMP(irt_top, 0, PLANES, irm_out_file_name, width, height, irm_rd_image);
                    } else
                        fclose(irm_out_file);
#endif
                    return 1;
                }
            }
        }
    }

    mem_rd = mem_rd_int;
    // if (mem_rd_int && block_type == e_irt_block_mesh && irt_top->task[e_irt_mrm] == 1 && line == 29 && pixel == 384)
    //	IRT_TRACE("AAA\n");

    if (block_type == e_irt_block_rot) {
        // fixed point calculation for Xi_start
        Xi_start = IRT_top::xi_start_calc(irt_top->irt_desc[irt_top->task[e_irt_irm]],
                                          line,
                                          e_irt_xi_start_calc_caller_irm,
                                          irt_top->task[e_irt_irm]);
    } else {
        Xi_start = 0;
    }
#if 0
	if (mem_rd_int) {// && fabs(Xi_start_float - Xi_start1)>8) {
		IRT_TRACE("IRT_IRM: Y=%d, Xi_start diff: Xi_start float = %d, Xi_start fixed = %d\n", line, Xi_start_float, Xi_start1);
		IRT_TRACE("Yi_first_fixed = %lld, Xi_first_fixed=%lld\n", irt_top->irt_desc[irt_top->task[irt_task]].Yi_first_fixed, irt_top->irt_desc[irt_top->task[irt_task]].Xi_first_fixed);
		IRT_TRACE("tand_fixed=%d, tan=%f\n", irt_top->irt_desc[irt_top->task[irt_task]].im_read_slope, irt_top->rot_pars[irt_top->task[irt_task]].im_read_slope);
		IRT_TRACE("line_diff = %lld, line_diff_x_tan=%lld, round=%d, Xi_first_delta=%lld, Xi_start1=%d\n", line_diff, line_diff_x_tan, irt_top->rot_pars[irt_top->task[e_irt_irm]].COORD_ROUND, Xi_first_delta, Xi_start1);
		exit (0);
	}
#endif

    if (line_start == 1) {
        pixel      = 0; // Xi_start;
        line_start = 0;
    }

    int32_t line_adj, byte_adj;
    if (irt_top->irt_desc[irt_top->task[e_irt_irm]].read_vflip == 0 || block_type == e_irt_block_mesh) {
        line_adj = line;
    } else {
        line_adj = irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].H - 1 - line;
    }

    if (block_type == e_irt_block_rot && irt_top->irt_desc[irt_top->task[e_irt_irm]].bg_mode == e_irt_bg_frame_repeat) {
        //	   IRT_TRACE_TO_LOG(2,IRTLOG,"%s read: BG_MODE BEFORE:: line_adj %d H %d
        //\n",my_string.c_str(),line_adj,irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].H - 1);
        line_adj = IRT_top::IRT_UTILS::irt_sat_int16(
            line_adj, 0, irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].H - 1);
        //	   IRT_TRACE_TO_LOG(2,IRTLOG,"%s read: BG_MODE AFTER:: line_adj %d \n",my_string.c_str(),line_adj);
    }

    if (irt_top->irt_desc[irt_top->task[e_irt_irm]].read_hflip == 0 || block_type == e_irt_block_mesh) {
        byte_adj = (Xi_start * (1 << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps)) + pixel;
    } else {
        byte_adj = ((irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].W /*- 1 - 127 */ - Xi_start) *
                    (1 << irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].Ps)) -
                   pixel - IRT_IFIFO_WIDTH;
    }

    addr = irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].addr_start +
           (int64_t)line_adj * (uint64_t)irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Hs +
           (int64_t)byte_adj;
    if (mem_rd_int)
        IRT_TRACE_TO_LOG(2,
                         IRTLOG,
                         "%s read: addr_start=0x%llx, line=%d, pixel=%d, addr = 0x%llx Hs = %lu byte_adj = %x Xi_start "
                         "%x at cycle %d\n",
                         my_string.c_str(),
                         irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].addr_start,
                         line_adj,
                         pixel,
                         addr,
                         (uint64_t)irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Hs,
                         byte_adj,
                         Xi_start,
                         cycle);

    /*
    if (pixel == Xi_start)
        IRT_TRACE("IRM read: addr_start=0x%llx, line=%d, pixel=%d, addr = 0x%llx\n",
    irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].addr_start, line, pixel, addr);
    */

    pixel_BufW =
        irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
            .BufW >>
        irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps;
    if (block_type == e_irt_block_rot) {
        pixel_max = (read_iimage_BufW) ? pixel_BufW
                                       : (irt_top->irt_desc[irt_top->task[e_irt_irm]].image_par[IIMAGE].S + 0 * 1 +
                                          irt_top->irt_desc[irt_top->task[e_irt_irm]].Xi_start_offset +
                                          irt_top->irt_desc[irt_top->task[e_irt_irm]].Xi_start_offset_flip *
                                              irt_top->irt_desc[irt_top->task[e_irt_irm]].read_hflip);
        if (irt_h5_mode && pixel_max > 256) {
            pixel_max = 256;
        }
    } else {
        pixel_max = (read_mimage_BufW) ? pixel_BufW : irt_top->irt_desc[irt_top->task[e_irt_mrm]].image_par[MIMAGE].S;
    }

    irt_top->irt_top_sig.irm_rd_first[block_type] = mem_rd_int & (line == Yi_start) & (pixel == 0); // Xi_start);

    bool    read_image_BufW = block_type == e_irt_block_rot ? read_iimage_BufW : read_mimage_BufW;
    int16_t last_line       = block_type == e_irt_block_rot
                            ? irt_top->irt_desc[irt_top->task[task_type]].Yi_end
                            : (int16_t)irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
    uint16_t BufW =
        irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
            .BufW;
    uint16_t last_pixel =
        read_image_BufW ? BufW : (pixel_max << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps);

    if (block_type == e_irt_block_rot) {
        if (read_iimage_BufW) {
            irt_top->irt_top_sig.irm_rd_last[block_type] =
                mem_rd_int & (line == irt_top->irt_desc[irt_top->task[task_type]].Yi_end) &
                (pixel + IRT_IFIFO_WIDTH >=
                 /*Xi_start +*/ (
                     irt_top->irt_cfg
                         .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                         .BufW)); // << irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].Ps));
        } else {
            irt_top->irt_top_sig.irm_rd_last[block_type] =
                mem_rd_int & (line == irt_top->irt_desc[irt_top->task[task_type]].Yi_end) &
                (pixel + IRT_IFIFO_WIDTH >=
                 /*Xi_start +*/ (pixel_max << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps));
        }
    } else {
        if (read_mimage_BufW) {
            irt_top->irt_top_sig.irm_rd_last[block_type] =
                mem_rd_int & (line == irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1) &
                (pixel + IRT_IFIFO_WIDTH >=
                 (irt_top->irt_cfg
                      .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                      .BufW)); // << irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].Ps));
        } else {
            irt_top->irt_top_sig.irm_rd_last[block_type] =
                mem_rd_int & (line == irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1) &
                (pixel + IRT_IFIFO_WIDTH >=
                 (pixel_max << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps));
        }
    }

    irt_top->irt_top_sig.irm_rd_last[block_type] =
        mem_rd_int & (line == last_line) & (pixel + IRT_IFIFO_WIDTH >= (int16_t)last_pixel);

#if 0
	if (mem_rd_int && pixel==Xi_start)
		//IRT_TRACE_TO_LOG(log_file, "IRT_IRM: task %d, line %d, Xi_start %d/%d/%d, pixel %d, addr %x, image width %d\n", task, line, Xi_start_float, Xi_start_fixed, Xi_start, pixel, addr, image_pars[IIMAGE].W);
		IRT_TRACE("IRT_IRM: task %d, line %d, Xi_start %d/%f/%d, pixel %d, addr %x, image width %d\n", task[irt_task], line, Xi_start_float, Xi_start_fixed/pow(2.0, irt_top->irt_cfg.pIRT_TOTAL_PREC), Xi_start, pixel, addr, irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].W);
#endif

    lsb_pad = 0;
    msb_pad = 0;

    if ((line < 0 || line >= (int16_t)irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].H) &&
        irt_top->irt_desc[irt_top->task[task_type]].bg_mode ==
            e_irt_bg_prog_value) { // Lines outside of image vertical boundaries
        lsb_pad = IRT_IFIFO_WIDTH;
        msb_pad = IRT_IFIFO_WIDTH;
        // IRT_TRACE("%s task %d, OoO lpad:mpad = %x : %x\n", my_string.c_str(),irt_top->task[task_type], lsb_pad,
        // msb_pad);
    } else { // inside of image vertical boundaries

#if 0
		//occurs in 1st input image line
		if (addr < irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].addr_start) //read address is less than input image start address
			lsb_pad = irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].addr_start - addr;

		//occurs in last input image line
		if (addr + 127 > irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].addr_end) //read address + 127 exceeds input image end address
			msb_pad = addr + 127 - irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].addr_end;
#endif
        // int16_t pixel_start = (irt_top->irt_desc[irt_top->task[task_type]].read_flip == 0) ? pixel :
        // irt_top->irt_desc[irt_top->task[task_type]].image_par[IIMAGE].W - 1 - 127 - pixel;
        byte_start = byte_adj;
#if 1
        // Read starts before image 1st pixel in line
        if (byte_start < 0) {
            lsb_pad = (-byte_start);
        }
        // Read ends after image last pixel in line
        uint16_t pixel_end = block_type == e_irt_block_rot
                                 ? irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].W
                                 : irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].S;
        uint32_t byte_end =
            ((uint32_t)pixel_end << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) - 1;
        if ((byte_start + IRT_IFIFO_WIDTH - 1) > (int32_t)byte_end) {
            msb_pad = byte_start + IRT_IFIFO_WIDTH - 1 - byte_end;
        }
        IRT_TRACE_TO_LOG(2,
                         IRTLOG,
                         "%s task %d, Valid lpad:mpad = %d : %d byte_start %d byte_end %d pixel_end %d IRT_IFIFO_WIDTH "
                         "%d block_type %d W %d S %d\n",
                         my_string.c_str(),
                         irt_top->task[task_type],
                         lsb_pad,
                         msb_pad,
                         byte_start,
                         byte_end,
                         pixel_end,
                         IRT_IFIFO_WIDTH,
                         (int)block_type,
                         irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].W,
                         irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].S);

#endif

#if 1
        // Read stripe is not multiple of IRT_IFIFO_WIDTH

        // Xi_start - 1st read pixel
        // irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S - read stripe width
        // irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S + Xi_start - last pixel to read
        // irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S + Xi_start - pixel = remain pixels to read

        /*
                if (irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S + 3 - pixel + Xi_start >=
           IRT_IFIFO_WIDTH) msb_pad = 0; else //number of remain pixels to read from input stripe < IRT_IFIFO_WIDTH
                    msb_pad = IRT_IFIFO_WIDTH - (irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S + 3 -
           pixel + Xi_start);
        */

#if 1
        // pixel_start = pixel;//(irt_top->irt_desc[irt_top->task[irt_task]].read_flip == 0) ? pixel :
        // irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].W - 1 - 127 - pixel; read_pixels = pixel_start -
        // Xi_start remained pixels = Si - read_pixels = Si – (pixel - Xi_start)
        //		if (irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S - ((pixel - 7) - Xi_start) <
        // IRT_IFIFO_WIDTH) {//number of remain pixels to read from input stripe < IRT_IFIFO_WIDTH //was -3
        if ((pixel_max << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) -
                ((pixel - 7 - irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) /*- Xi_start*/) <
            128) { // number of remain pixels to read from input stripe < 128 //was -3
            // pad1 = IRT_IFIFO_WIDTH - (irt_top->irt_desc[irt_top->task[irt_task]].image_par[IIMAGE].S - ((pixel - 7) -
            // Xi_start)); //was -3
            pad1 = IRT_IFIFO_WIDTH -
                   ((pixel_max << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) -
                    ((pixel - 7 -
                      irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) /*- Xi_start*/)); // was -3
            if (irt_top->irt_desc[irt_top->task[task_type]].read_hflip == 0 || block_type == e_irt_block_mesh) {
                msb_pad = pad1 > msb_pad ? pad1 : msb_pad;
            } else {
                lsb_pad = pad1 > lsb_pad ? pad1 : lsb_pad;
            }
            // IRT_TRACE("%s task %d, Inside Check lpad:mpad = %d : %d\n", my_string.c_str(),irt_top->task[task_type],
            // lsb_pad, msb_pad);
        }
        // IRT_TRACE("%s task %d, pixel_max %d, Ps %d, pixel %d, pad1 %d\n", my_string.c_str(),irt_top->task[task_type],
        // pixel_max,irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps,pixel,pad1);

        if (lsb_pad > IRT_IFIFO_WIDTH) {
            lsb_pad = IRT_IFIFO_WIDTH;
        }

        if (msb_pad > IRT_IFIFO_WIDTH) {
            msb_pad = IRT_IFIFO_WIDTH;
        }
        // IRT_TRACE("%s task %d, Final Check lpad:mpad = %d : %d\n", my_string.c_str(),irt_top->task[task_type],
        // lsb_pad, msb_pad);
#endif

#endif
    }

    // if (mem_rd_int && block_type == e_irt_block_rot)
    //	IRT_TRACE("%s task %d, reading line %d, Xi_start %d, pixel %d, lsb_pad %d, msb_pad %d, addr %llx\n",
    // my_string.c_str(), irt_top->task[task_type], line, Xi_start, pixel, lsb_pad, msb_pad, addr);

    meta_out.line     = line;
    meta_out.Xi_start = Xi_start;
    meta_out.task     = irt_top->task[task_type];

    return 0;
}

template <class BUS_OUT, Eirt_block_type block_type>
void IRT_top::IRT_IFIFO<BUS_OUT, block_type>::run(bool                  push,
                                                  bool                  pop,
                                                  bool                  read,
                                                  const bus128B_struct& data_in,
                                                  meta_data_struct      metain,
                                                  BUS_OUT&              data_out,
                                                  meta_data_struct&     metaout,
                                                  bool&                 empty,
                                                  bool&                 full)
{

    int         IRT_FIFO_R_ENTRIES = block_type == e_irt_block_rot ? IRT_IFIFO_R_ENTRIES : IRT_MFIFO_R_ENTRIES;
    int         IRT_FIFO_OUT_WIDTH = block_type == e_irt_block_rot ? IRT_IFIFO_OUT_WIDTH : IRT_MFIFO_OUT_WIDTH;
    std::string my_string          = getInstanceName();

    if (fifo_fullness > IRT_IFIFO_DEPTH) {
        IRT_TRACE("Error: %s: wp=%d, rp=%d, rd_slot=%d, fullness=%d\n",
                  my_string.c_str(),
                  irt_ififo_wp,
                  irt_ififo_rp,
                  irt_ififo_r_entry,
                  fifo_fullness);
        IRT_TRACE_TO_RES(test_res, " failed, %s overflow\n", my_string.c_str());
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (push) {
        // if (block_type == e_irt_block_rot) IRT_TRACE("%s push: wp = %d, rp = %d, slot=%d, fullness=%d, task %d, line
        // %d, Xi_start %d\n", my_string.c_str(), irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry,fifo_fullness,
        // metain.task, metain.line, metain.Xi_start);
        if (fifo_fullness >= IRT_IFIFO_DEPTH) {
            IRT_TRACE("Push to full %s: wp=%d, rp=%d, rd_slot=%d, fullness=%d\n",
                      my_string.c_str(),
                      irt_ififo_wp,
                      irt_ififo_rp,
                      irt_ififo_r_entry,
                      fifo_fullness);
            IRT_TRACE_TO_RES(test_res, " failed, %s overflow\n", my_string.c_str());
            IRT_CLOSE_FAILED_TEST(0);
        }
        for (uint8_t i = 0; i < IRT_IFIFO_WIDTH; i++) {
            irt_ififo[irt_ififo_wp].pix[i] = data_in.pix[i];
        }
        irt_ififo_meta[irt_ififo_wp].task     = metain.task;
        irt_ififo_meta[irt_ififo_wp].line     = metain.line;
        irt_ififo_meta[irt_ififo_wp].Xi_start = metain.Xi_start;
        irt_ififo_wp                          = (irt_ififo_wp + 1) % IRT_IFIFO_DEPTH;
        fifo_fullness++;
    }

    if (pop || read) {
        if (fifo_fullness == 0) {
            IRT_TRACE("Pop from empty %s: wp=%d, rp=%d, rd_slot=%d, fullness=%d\n",
                      my_string.c_str(),
                      irt_ififo_wp,
                      irt_ififo_rp,
                      irt_ififo_r_entry,
                      fifo_fullness);
            IRT_TRACE_TO_RES(test_res, " failed, %s underflow\n", my_string.c_str());
            IRT_CLOSE_FAILED_TEST(0);
        }

        // if (block_type == e_irt_block_rot) IRT_TRACE("IFIFO pop: wp = %d, rp = %d, slot=%d, fullness=%d, task %d,
        // line %d, Xi_start %d\n", irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry, fifo_fullness,
        // irt_ififo_meta[irt_ififo_rp].task, irt_ififo_meta[irt_ififo_rp].line, irt_ififo_meta[irt_ififo_rp].Xi_start);
        if (pop) {
            irt_ififo_r_entry = 0;
            irt_ififo_rp      = (irt_ififo_rp + 1) % IRT_IFIFO_DEPTH;
            fifo_fullness--;
        } else {
            if (read) {
                if (irt_ififo_r_entry == IRT_FIFO_R_ENTRIES - 1) {
                    irt_ififo_rp = (irt_ififo_rp + 1) % IRT_IFIFO_DEPTH;
                    fifo_fullness--;
                }
                irt_ififo_r_entry = (irt_ififo_r_entry + 1) % IRT_FIFO_R_ENTRIES;
            }
        }
        IRT_TRACE_TO_LOG(3,
                         IRTLOG,
                         "IFIFO pop: wp = %d, rp = %d, slot=%d, fullness=%d, line %d, Xi_start %d full %d empty %d\n",
                         irt_ififo_wp,
                         irt_ififo_rp,
                         irt_ififo_r_entry,
                         fifo_fullness,
                         metaout.line,
                         metaout.Xi_start,
                         (fifo_fullness >= IRT_IFIFO_DEPTH - 2) ? 1 : 0,
                         (fifo_fullness == 0) ? 1 : 0);
    }

    for (uint8_t i = 0; i < IRT_FIFO_OUT_WIDTH; i++) {
        data_out.pix[i] = irt_ififo[irt_ififo_rp].pix[i + IRT_FIFO_OUT_WIDTH * irt_ififo_r_entry];
    }
    metaout.task     = irt_ififo_meta[irt_ififo_rp].task;
    metaout.line     = irt_ififo_meta[irt_ififo_rp].line;
    metaout.Xi_start = irt_ififo_meta[irt_ififo_rp].Xi_start;

    //	if (block_type == e_irt_block_mesh && pop)
    //		IRT_TRACE("%s pop: wp = %d, rp = %d, slot=%d, fullness=%d, line %d, Xi_start %d\n", my_string.c_str(),
    // irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry,fifo_fullness, metaout.line, metaout.Xi_start); 	if (block_type
    // == e_irt_block_mesh && read) 		IRT_TRACE("%s read: wp = %d, rp = %d, slot=%d, fullness=%d, line %d,
    // Xi_start %d\n", my_string.c_str(), irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry, fifo_fullness, metaout.line,
    // metaout.Xi_start);

    full  = (fifo_fullness >= IRT_IFIFO_DEPTH - 2) ? 1 : 0;
    empty = (fifo_fullness == 0) ? 1 : 0;
}

template <class BUS_IN,
          class BUS_OUT,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_HEIGHT,
          Eirt_block_type block_type>
bool IRT_top::IRT_RMWM<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_HEIGHT, block_type>::run(
    bool              ififo_empty,
    bool&             ififo_pop,
    bool&             ififo_read,
    const BUS_IN&     ififo_data,
    meta_data_struct  ififo_meta,
    uint16_t&         rm_wr_addr,
    bool              rm_wr_sel[ROW_BANKS],
    BUS_OUT           rm_din[COL_BANKS],
    uint16_t&         rm_meta_wr_addr,
    meta_data_struct& rm_meta_data)
{

    Eirt_blocks_enum task_type      = block_type == e_irt_block_rot ? e_irt_rmwm : e_irt_mmwm;
    uint8_t          image_type     = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    uint8_t          mem_bank_width = block_type == e_irt_block_rot ? IRT_ROT_MEM_BANK_WIDTH : IRT_MESH_MEM_BANK_WIDTH;
    uint8_t          mem_pxl_shift  = block_type == e_irt_block_rot ? IRT_ROT_MEM_PXL_SHIFT : IRT_MESH_MEM_PXL_SHIFT;
    std::string      my_string      = getInstanceName();
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];

    int16_t line_tmp, line1;

    //   //---------------------------------------
    //   //	if (irt_top->wait_resamp_task(task_type) == 1) {
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[task_type].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //---------------------------------------

#ifdef CREATE_IMAGE_DUMPS_RMWM
    int width, height, row_padded;
    if (block_type == e_irt_block_rot) {
        width  = irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].S;
        height = irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Yi_end - Yi_start + 1;
    } else {
        width  = irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].W;
        height = irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].H;
    }
    row_padded = (width * 3 + 3) & (~3);
#endif

    ififo_pop                    = 0;
    ififo_read                   = 0;
    irt_mem_ctrl->rmwr_task_done = 0;
    for (uint8_t bank_row = 0; bank_row < ROW_BANKS; bank_row++) {
        rm_wr_sel[bank_row] = 0;
    }

    if (block_type == e_irt_block_mesh) {
        if (irt_top->task[task_type] >= irt_top->num_of_tasks) {
            return 1; // TBD - removed after adding multiple tasks support
        }
        if (irt_top->irt_desc[irt_top->task[task_type]].irt_mode !=
            e_irt_mesh) { // IRM works as MRM and task is not mesh, droppped
            IRT_TRACE("%s task %d is not mesh task and dropped at cycle %d\n",
                      m_name.c_str(),
                      irt_top->task[task_type],
                      cycle);
            irt_top->task[task_type]++;
            irt_mem_ctrl->start_line[irt_top->task[task_type]] = irt_mem_ctrl->bot_ptr;
            Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
            irt_mem_ctrl->last_line[irt_top->task[task_type]] = Yi_start - 1;
            line                                              = Yi_start;
            return 0;
        }
    }

    if (tasks_start == 0) {

#ifdef CREATE_IMAGE_DUMPS_RMWM
        if (block_type == e_irt_block_rot) {
            rmwm_out_file = fopen("irt_rmwm_out.bmp", "wb");
            generate_bmp_header(rmwm_out_file, width, height);
        } else {
            rmwm_out_file = fopen("irt_mmwm_out.txt", "w");
        }
        // if(irm_out_file == NULL)
        //	throw "Argument Exception";
#endif

        tasks_start = 1;
        // IRT_TRACE("IRT_RMWM: setting rm_last_line[%d]=%d\n", irt_top->task[e_irt_rmwm],
        // irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Yi_start - 1);
        if (block_type == e_irt_block_rot) {
            irt_mem_ctrl->last_line[irt_top->task[e_irt_rmwm]] = Yi_start - 1;
        } else {
            irt_mem_ctrl->last_line[irt_top->task[e_irt_mmwm]] = -1;
        }
        irt_mem_ctrl->last_line[irt_top->task[task_type]] =
            irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type) - 1;
        IRT_TRACE_TO_LOG(4,
                         IRTLOG,
                         "%s:1756 setting last_line[%d]=%d\n",
                         my_string.c_str(),
                         irt_mem_ctrl->last_line[irt_top->task[task_type]],
                         irt_top->task[task_type]);
    }

    // if (block_type == e_irt_block_rot)
    //	IRT_TRACE_TO_LOG(3,IRTLOG,"%s: ififo_empty %d, done %d, buf_format %d, buf_mode %d, fullness %d[%d:%d], BufH
    //%d\n", my_string.c_str(), ififo_empty, done, irt_top->irt_cfg.buf_format[block_type],
    //		irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode,
    //	   irt_mem_ctrl->fullness, irt_mem_ctrl->first_line[irt_top->task[task_type]],
    // irt_mem_ctrl->last_line[irt_top->task[task_type]],
    //		irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode].BufH);

    if (ififo_empty == 0 && done == 0 &&
        ((irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static &&
          irt_mem_ctrl->fullness <
              irt_top->irt_cfg
                  .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                  .BufH) ||
         //		 (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_dynamic && irt_mem_ctrl->fullness <
         // irt_top->irt_cfg.Hb[block_type]))) {
         (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_dynamic &&
          ((irt_top->irt_cfg.Hb[block_type] + 1 - irt_mem_ctrl->fullness) >
           irt_top->irt_cfg
               .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
               .Buf_EpL)))) { // ROTSIM_DV_INTGR -- DYNAMIC MODE TASK SWITCH DATA CORRUPTION FIX
        ififo_read = 1;
        // wr = 1;
        // addr = irt_desc[task].oimage_addr + line * image_pars[OIMAGE].W /*+ image_pars[OIMAGE].S *
        // image_pars[OIMAGE].Ns*/ + pixel;
        if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
            line_tmp = irt_mem_ctrl->start_line[irt_top->task[task_type]] + ififo_meta.line - Yi_start;
            line1    = line_tmp &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                        .BufH_mod;
        } else {
            line_tmp = ififo_meta.line - Yi_start;
            line1    = line_tmp;
        }
        //		if (line1 < 0) line1 = 0;
        uint16_t line_in_bank_row = (line1 >> (int)log2(ROW_BANKS)); //((line1>>3)&0x1f);
        // rm_wr_addr = (line_sel_in_bank_row << 4) + (pixel >> 4);
        uint16_t BufW_entries =
            irt_top->irt_cfg
                .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                .Buf_EpL; // << irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].Ps;
        if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
            rm_wr_addr      = (line_in_bank_row * BufW_entries) + (pixel >> mem_pxl_shift);
            rm_meta_wr_addr = line_in_bank_row;
        } else {
            rm_wr_addr = (irt_mem_ctrl->start_line[irt_top->task[task_type]] + line_in_bank_row * BufW_entries +
                          (pixel >> mem_pxl_shift)) &
                         irt_top->irt_cfg.Hb_mod[block_type];
            rm_meta_wr_addr = (irt_mem_ctrl->start_line[irt_top->task[task_type]] + line_in_bank_row * BufW_entries) %
                              IRT_META_MEM_BANK_HEIGHT;
        }

        if (rm_wr_addr < 0 || rm_wr_addr >= BANK_HEIGHT) {
            IRT_TRACE("%s write address %d invalid, task %d, line %d, pixel %d\n",
                      my_string.c_str(),
                      rm_wr_addr,
                      irt_top->task[task_type],
                      line,
                      pixel);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s write address %d invalid, task %d, line %d, pixel %d\n",
                             my_string.c_str(),
                             rm_wr_addr,
                             irt_top->task[task_type],
                             line,
                             pixel);
            IRT_CLOSE_FAILED_TEST(0);
        }
        //		if (block_type == e_irt_block_mesh)
        //			IRT_TRACE("%s write: task %d, line %d, pixel %d ififo_meta[%d][%d].task %d ififo_meta[%d][%d].line
        //%d at cycle %d\n", my_string.c_str(), irt_top->task[task_type], line, pixel, line1 % ROW_BANKS,
        // rm_meta_wr_addr,
        // ififo_meta.task, line1 % ROW_BANKS, rm_meta_wr_addr, ififo_meta.line, cycle); 	IRT_TRACE("%s write: task
        // %d, line %d, pixel %d ififo_meta[%d][%d].task %d ififo_meta[%d][%d].line %d\n", my_string.c_str(),
        // irt_top->task[task_type], line, pixel, line1 % 8, rm_meta_wr_addr, ififo_meta.task, line1 % 8,
        // rm_meta_wr_addr, ififo_meta.line);

        if (irt_top->task[task_type] != ififo_meta.task) {
            IRT_TRACE("%s: task %d different from ififo_meta.task %d at line %d\n",
                      my_string.c_str(),
                      irt_top->task[task_type],
                      ififo_meta.task,
                      line);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s: task %d different from ififo_meta.task %d\n",
                             my_string.c_str(),
                             irt_top->task[task_type],
                             ififo_meta.task);
            IRT_CLOSE_FAILED_TEST(0);
        }
        if (line != ififo_meta.line) {
            IRT_TRACE("%s at task %d line %d different from ififo_meta.line %d\n",
                      my_string.c_str(),
                      irt_top->task[task_type],
                      line,
                      ififo_meta.line);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s: at task %d line %d different from ififo_meta.line %d\n",
                             my_string.c_str(),
                             irt_top->task[task_type],
                             line,
                             ififo_meta.line);
            IRT_CLOSE_FAILED_TEST(0);
        }
        rm_meta_data.Xi_start = ififo_meta.Xi_start;
        rm_meta_data.line     = ififo_meta.line;
        rm_meta_data.task     = ififo_meta.task;

        // if (block_type == e_irt_block_mesh) {
        // IRT_TRACE("%s: task %d writing line %d into line %d pixel %d, Xi_start addr %d, Y %d, Xi_start %d, rm_wr_addr
        // %d at cycle %d\n", my_string.c_str(), irt_top->task[task_type], line, line1, pixel, rm_meta_wr_addr,
        // rm_meta_data.line, rm_meta_data.Xi_start, rm_wr_addr, cycle); IRT_TRACE("start_line %d, ififo_meta.line %d,
        // Yi_start %d\n", irt_mem_ctrl->start_line[irt_top->task[task_type]], ififo_meta.line, Yi_start);
        //}

//		if (pixel==0)
//#ifdef STANDALONE_ROTATOR
#if 0
			//IRT_TRACE_TO_LOG(log_file, "IRT_RMWM: task %d writing line %d into line %d pixel %d, Xi_start addr %d, Y %d, Xi_start %d, rm_wr_addr %d\n", irt_top->task[e_irt_rmwm], line_tmp, line1, pixel, rm_Xi_start_wr_addr, rm_Xi_start.line, rm_Xi_start.Xi_start, rm_wr_addr);
			IRT_TRACE("IRT_RMWM: ififo_data\n");
			for (int i=0; i<16; i++)
				IRT_TRACE( "%x ", ififo_data.pix[i]);
			IRT_TRACE( "\n");
#endif
        //#endif

        for (uint8_t bank_row = 0; bank_row < ROW_BANKS; bank_row++) {
            //			if (ififo_meta.task < task[e_irt_oicc]) { // data from old task, will be drain
            //				rm_wr_sel[bank_row]=0;
            //			} else {
            if (bank_row == (line1 % ROW_BANKS)) {
                rm_wr_sel[bank_row] = 1;
            } else {
                rm_wr_sel[bank_row] = 0;
            }
            //			}
        }

        //		if (rm_wr_sel[line1%8]=1)
        //			IRT_TRACE("RMWM write: irt_top->meta_mem[%d][%d], task %d, line %d, Xi_start %d\n", (line1%8),
        // rm_Xi_start_wr_addr, ififo_meta.task, ififo_meta.line, ififo_meta.Xi_start);
        for (uint8_t i = 0; i < mem_bank_width; i++) {
            if ((block_type == e_irt_block_rot && (irt_top->irt_cfg.buf_1b_mode[e_irt_block_rot] == 0 ||
                                                   irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].Ps ==
                                                       1)) || // pixel is 2B or pixel 1B w/o reshuffle storage
                (block_type == e_irt_block_mesh && (irt_top->irt_cfg.buf_1b_mode[e_irt_block_mesh] == 0 ||
                                                    irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].Ps ==
                                                        3))) { // pixel is 8B or pixel 4B w/o reshuffle storage
                rm_din[0].pix[i] = ififo_data.pix[i]; // ext_mem[image_pars[IIMAGE].ADDR+task*IMAGE_W*IMAGE_H + line *
                                                      // image_pars[IIMAGE].W + pixel + i]; //data_in.pix[i];
                rm_din[1].pix[i] = ififo_data.pix[i + mem_bank_width];
            } else { // pixel is 1B (input image) or 4B (mesh image) w/ reshuffle storage
                // data is stored: [p0 p16 p1 p17 p2 p18 p3 p19 p4 p20 p5 p21 p6 p22 p7 p23] [p8 p24 p9 p25 p10 p26 p11
                // p27 p12 p28 p13 p29 p14 p30 p15 p31] p8 start location in IFIFO is 8 for rot and 32 for mesh, means
                // mem_bank_width >> 1, used as offset for rm_din[1] even "pixel" locations in rm_din (p0, p1, p2,...,
                // p8, p9) come from left side of IFIFO, odd pixels from right side: for rotation pixel is i&1, for mesh
                // pixel is (i>>2)&1 - that is * mem_bank_width that is 1/2 of fifo width
                //
                if (block_type == e_irt_block_rot) {
                    // each pixel is 1B. Each mem bank holds 16 bytes (16 pixels), 2 banks stores 32 pixels
                    rm_din[0].pix[i] = ififo_data.pix[(i & 1) * mem_bank_width + (i >> 1)];
                    rm_din[1].pix[i] = ififo_data.pix[(i & 1) * mem_bank_width + (i >> 1) + (mem_bank_width >> 1)];
                } else {
                    // each "pixel" is 4B (2B for X and 2B for Y). Each mem bank holds 64 bytes (16 pixels), 2 banks
                    // stores 32 pixels i>>2 is pixel slot, i & 3 is byte inside of pixel
                    rm_din[0].pix[i] = ififo_data.pix[((i >> 2) & 1) * mem_bank_width + (i >> 3) * 4 + (i & 3)];
                    rm_din[1].pix[i] =
                        ififo_data
                            .pix[((i >> 2) & 1) * mem_bank_width + (i >> 3) * 4 + (i & 3) + (mem_bank_width >> 1)];
                }
            }
        }
#if 0
		//if (block_type == e_irt_block_mesh) {
		if (block_type == e_irt_block_rot) {
			IRT_TRACE("IRT_RMWM: rm_din\n");
			for (uint8_t bank = 0; bank < 2; bank++){
				for (int8_t i = mem_bank_width - 1; i >= 0; i--) {
					IRT_TRACE("%02x", rm_din[bank].pix[i]);
				}
				IRT_TRACE("\n");
			}
			IRT_TRACE("\n");
		}
#endif
#ifdef CREATE_IMAGE_DUMPS_RMWM
        uint64_t pix_val = 0;
        for (int pix = 0;
             pix < ((2 * mem_bank_width) >> irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps);
             pix++) {
            pix_val = 0;
            for (int idx = 0; idx < (1 << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps);
                 idx++) {
                pix_val =
                    pix_val +
                    ((uint64_t)ififo_data
                         .pix[(1 << irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) * pix + idx]
                     << (8 * idx));
            }
            irt_rmwm_image[irt_top->task[task_type]][line - Yi_start]
                          [(pixel >> irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].Ps) + pix] =
                              pix_val >>
                              (block_type == e_irt_block_rot ? irt_top->rot_pars[irt_top->task[task_type]].Pwi - 8 : 0);
        }

#endif
        // IRT_TRACE("ORM source addr %d\n", image_pars[IIMAGE].ADDR+task*IMAGE_W*IMAGE_H + line * image_pars[IIMAGE].W
        // + pixel);

        pixel += 2 * mem_bank_width;

        uint16_t pixel_BufW, pixel_max;
        if (block_type == e_irt_block_rot) {
            pixel_BufW =
                irt_top->irt_cfg
                    .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].buf_mode]
                    .BufW >>
                irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].Ps;
            //		int pixel_max = (super_irt == 1) ? BufW :
            //((irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].S <= 128) ? 128 : 256);
            pixel_max = (read_iimage_BufW) ? pixel_BufW
                                           : (irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].S + 0 * 1 +
                                              irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Xi_start_offset +
                                              irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Xi_start_offset_flip *
                                                  irt_top->irt_desc[irt_top->task[e_irt_rmwm]].read_hflip);
            if (irt_h5_mode && pixel_max > 256) {
                pixel_max = 256;
            }
        } else {
            pixel_BufW =
                irt_top->irt_cfg
                    .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].buf_mode]
                    .BufW >>
                irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].Ps;
            pixel_max =
                (read_mimage_BufW) ? pixel_BufW : irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].S;
        }
        //		printf("IRT_RMWM Si=%d, offset = %d, pixel_max = %d\n",
        // irt_top->irt_desc[irt_top->task[e_irt_rmwm]].image_par[IIMAGE].S, pixel_max);

        if (pixel >= (pixel_max << irt_top->irt_desc[irt_top->task[task_type]]
                                       .image_par[image_type]
                                       .Ps)) { // irt_desc[task].Si) {//((image_pars[IIMAGE].S<=128) ? 128 : 256)) {
                                               // //line write is finished
            pixel     = 0;
            ififo_pop = 1;
            line++;
            //			if (ififo_meta.task >= task[e_irt_oicc]) { // data from old task will be drain because "if" will
            // not be executed
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                if (irt_mem_ctrl->lines_valid[irt_top->task[task_type]] == 0) {
                    irt_mem_ctrl->lines_valid[irt_top->task[task_type]] = 1;
                }

                if (irt_mem_ctrl->rd_done[irt_top->task[task_type]] == 0) { // avoid writing is read is completed
                    irt_mem_ctrl->last_line[irt_top->task[task_type]]++;
                    IRT_TRACE_TO_LOG(4,
                                     IRTLOG,
                                     "%s:1920 setting last_line[%d]=%d\n",
                                     my_string.c_str(),
                                     irt_mem_ctrl->last_line[irt_top->task[task_type]],
                                     irt_top->task[task_type]);
                    // if (block_type == e_irt_block_mesh)
                    //	IRT_TRACE("%s: incrementing rm_last_line[%d] to %d\n", my_string.c_str(),
                    // irt_top->task[task_type], irt_mem_ctrl->last_line[irt_top->task[task_type]]);
                    // rm_last_line[ififo_meta.task]++;
                    irt_mem_ctrl->bot_ptr =
                        (irt_mem_ctrl->bot_ptr + 1) &
                        irt_top->irt_cfg
                            .rm_cfg[block_type]
                                   [irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                            .BufH_mod; // IRT_ROT_MEM_HEIGHT;
                    irt_mem_ctrl->fullness = (irt_mem_ctrl->fullness + 1); //%IRT_ROT_MEM_HEIGHT;
                }
            } else {
                if (((line - Yi_start) >> (int)log2(ROW_BANKS)) > 0 &&
                    ((line - Yi_start) % ROW_BANKS) == 0) { // 8 lines are written
                    if (irt_mem_ctrl->lines_valid[irt_top->task[task_type]] == 0) {
                        irt_mem_ctrl->lines_valid[irt_top->task[task_type]] = 1;
                    }

                    if (irt_mem_ctrl->rd_done[irt_top->task[task_type]] == 0) { // avoid writing is read is completed
                        irt_mem_ctrl->last_line[irt_top->task[task_type]] += ROW_BANKS;
                        // IRT_TRACE("IRT_RMWM: incrementing rm_last_line[%d] to %d\n", irt_top->task[e_irt_rmwm],
                        // irt_top->rm_last_line[irt_top->task[e_irt_rmwm]]);
                        irt_mem_ctrl->bot_ptr =
                            (irt_mem_ctrl->bot_ptr +
                             irt_top->irt_cfg
                                 .rm_cfg[block_type]
                                        [irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                                 .Buf_EpL) %
                            irt_top->irt_cfg.Hb[block_type]; // IRT_ROT_MEM_HEIGHT;
                        irt_mem_ctrl->fullness =
                            (irt_mem_ctrl->fullness +
                             irt_top->irt_cfg
                                 .rm_cfg[block_type]
                                        [irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                                 .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                        IRT_TRACE_TO_LOG(
                            3,
                            IRTLOG,
                            "%s : task %d incrementing last_line %d fullness %d Buf_EpL %d\n",
                            my_string.c_str(),
                            irt_top->task[task_type],
                            irt_mem_ctrl->last_line[irt_top->task[task_type]],
                            irt_mem_ctrl->fullness,
                            irt_top->irt_cfg
                                .rm_cfg[block_type]
                                       [irt_top->irt_desc[irt_top->task[task_type]].image_par[image_type].buf_mode]
                                .Buf_EpL);
                    }
                }
            }
//			}
#if 1
#ifdef STANDALONE_ROTATOR
            // IRT_TRACE_TO_LOG(log_file, "IRT_RMWM status cur: task %d, first/last lines [%d/%d:%d], top/bot ptr
            // [%d:%d], fullness %d\n", 		irt_top->task[e_irt_rmwm],
            // rm_first_line[irt_top->task[e_irt_rmwm]],irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Yi_start,
            // irt_top->rm_last_line[irt_top->task[e_irt_rmwm]], rm_top_ptr , irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
            // IRT_TRACE_TO_LOG(log_file, "IRT_RMWM status prv: task %d, first/last lines [%d:%d], top/bot ptr [%d:%d],
            // fullness %d\n", 		(irt_top->task[e_irt_rmwm]-1)%MAX_TASKS,
            // rm_first_line[(irt_top->task[e_irt_rmwm]-1)%MAX_TASKS],
            // rm_last_line[(irt_top->task[e_irt_rmwm]-1)%MAX_TASKS], rm_top_ptr , irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
#endif
#endif
            line_start = 1;

            if (((block_type == e_irt_block_rot) &&
                 (line == irt_top->irt_desc[irt_top->task[e_irt_rmwm]].Yi_end + 1)) ||
                ((block_type == e_irt_block_mesh) &&
                 (line == irt_top->irt_desc[irt_top->task[e_irt_mmwm]].image_par[MIMAGE].H))) { // task is finished
                IRT_TRACE_TO_LOG(4,
                                 IRTLOG,
                                 "%s task %d is finished at cycle %d\n",
                                 my_string.c_str(),
                                 irt_top->task[task_type],
                                 cycle);
                irt_mem_ctrl->wr_done[irt_top->task[task_type]] = 1;
                irt_top->task[task_type]++;
                irt_mem_ctrl->start_line[irt_top->task[task_type]] = irt_mem_ctrl->bot_ptr;
                IRT_TRACE_TO_LOG(4,
                                 IRTLOG,
                                 "%s task %d UPDATED start_line %d%d\n",
                                 my_string.c_str(),
                                 irt_top->task[task_type],
                                 irt_mem_ctrl->start_line[irt_top->task[task_type]],
                                 cycle);
#ifdef STANDALONE_ROTATOR
                // IRT_TRACE_TO_LOG(log_file, "IRT_RMWM setting rm_start_line[%d]=%d\n", irt_top->task[e_irt_rmwm],
                // irt_top->rm_bot_ptr);
#endif
                Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
                irt_mem_ctrl->last_line[irt_top->task[task_type]] = Yi_start - 1;
                irt_mem_ctrl->rmwr_task_done                      = 1;
                line                                              = Yi_start;
                task_start                                        = 0;
                if (irt_top->task[task_type] == irt_top->num_of_tasks) {
                    done = 1;

#ifdef CREATE_IMAGE_DUMPS_RMWM
                    unsigned char* data = new unsigned char[row_padded];
                    if (block_type == e_irt_block_rot) {
                        for (int i = 0; i < height; i++) {
                            for (int j = 0; j < width * 3; j += 3) {
                                data[j]     = (unsigned char)irt_rmwm_image[0][height - 1 - i][j / 3];
                                data[j + 1] = (unsigned char)irt_rmwm_image[1][height - 1 - i][j / 3];
                                data[j + 2] = (unsigned char)irt_rmwm_image[2][height - 1 - i][j / 3];
                            }
                            fwrite(data, sizeof(unsigned char), row_padded, rmwm_out_file);
                        }
                    } else {
                        for (int i = 0; i < height; i++) {
                            for (int j = 0; j < width; j++) {
                                fprintf(rmwm_out_file, "%llx\n", irt_rmwm_image[0][i][j]);
                            }
                        }
                    }
                    fclose(rmwm_out_file);
#endif
                }
            }
        }
    }
    return done;
}

template <class BUS,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_WIDTH,
          uint16_t        BANK_HEIGHT,
          Eirt_block_type block_type>
void IRT_top::IRT_ROT_MEM<BUS, ROW_BANKS, COL_BANKS, BANK_WIDTH, BANK_HEIGHT, block_type>::run(
    bool             wr_port,
    uint16_t         wr_addr,
    bool             wr_sel[ROW_BANKS],
    BUS              data_in[COL_BANKS],
    uint16_t         meta_wr_addr,
    meta_data_struct meta_data,
    int16_t          rd_addr[ROW_BANKS][COL_BANKS],
    bool             rd_sel[ROW_BANKS][COL_BANKS],
    BUS              data_out[ROW_BANKS][COL_BANKS],
    bool             rd_en,
    uint8_t          irt_task)
{

    // uint8_t image_type = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    std::string          my_string    = getInstanceName();
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];

    for (uint8_t bank_row = 0; bank_row < ROW_BANKS; bank_row++) {
        if (wr_sel[bank_row] == 1 && wr_port == 1) {
            if (meta_wr_addr >= IRT_META_MEM_BANK_HEIGHT) {
                IRT_TRACE("%s meta mem write address %d invalid, task %d\n", my_string.c_str(), meta_wr_addr, irt_task);
                IRT_TRACE_TO_RES(test_res,
                                 " failed, %s meta mem write address %d invalid, task %d\n",
                                 my_string.c_str(),
                                 meta_wr_addr,
                                 irt_task);
                IRT_CLOSE_FAILED_TEST(0);
            }
            meta_mem[bank_row][meta_wr_addr].Xi_start = meta_data.Xi_start;
            meta_mem[bank_row][meta_wr_addr].line     = meta_data.line;
            meta_mem[bank_row][meta_wr_addr].task     = meta_data.task;

            // IRT_TRACE("irt_top->meta_mem[%d][%d] write: task %d, line %d, Xi_start %d\n" , bank_row, meta_wr_addr,
            // meta_data.task, meta_data.line, meta_data.Xi_start);
            for (uint8_t i = 0; i < BANK_WIDTH; i++) {
                if (wr_addr < 0 || wr_addr >= BANK_HEIGHT) {
                    IRT_TRACE("%s write address %d invalid, task %d\n", my_string.c_str(), wr_addr, irt_task);
                    IRT_TRACE_TO_RES(test_res,
                                     " failed, %s write address %d invalid, task %d\n",
                                     my_string.c_str(),
                                     wr_addr,
                                     irt_task);
                    IRT_CLOSE_FAILED_TEST(0);
                }
                rot_mem[bank_row][0][wr_addr].pix[i] = data_in[0].pix[i];
                rot_mem[bank_row][1][wr_addr].pix[i] = data_in[1].pix[i];
            }
            //---------------------------------------
            // ROTSIM_DV_INTGR
            IRT_TRACE_TO_LOG(
                4, IRTLOG, "%s:WRITE : task %d ROW:WADDR [%x,%x] ", my_string.c_str(), irt_task, bank_row, wr_addr);
            for (uint8_t c = 0; c < 2; c++) {
                IRT_TRACE_TO_LOG(4, IRTLOG, " COL[%d] DATA [", c);
                for (uint8_t i = 0; i < BANK_WIDTH; i++) {
                    IRT_TRACE_TO_LOG(4, IRTLOG, "%d,", rot_mem[bank_row][c][wr_addr].pix[i]);
                }
                IRT_TRACE_TO_LOG(4, IRTLOG, " ] ");
            }
            IRT_TRACE_TO_LOG(4, IRTLOG, " \n ");
            //---------------------------------------
        }

        for (uint8_t bank_col = 0; bank_col < COL_BANKS; bank_col++) {
            if (rd_sel[bank_row][bank_col] == 1 && wr_port == 0 && rd_en) {
                if (rd_addr[bank_row][bank_col] < 0 || rd_addr[bank_row][bank_col] >= BANK_HEIGHT) {
                    IRT_TRACE("%s read address %d invalid, task %d, bank_row %d, bank_col %d\n",
                              my_string.c_str(),
                              rd_addr[bank_row][bank_col],
                              irt_task,
                              bank_row,
                              bank_col);
                    IRT_TRACE_TO_RES(test_res,
                                     " failed, %s read address %d invalid, task %d, bank_row %d, bank_col %d\n",
                                     my_string.c_str(),
                                     rd_addr[bank_row][bank_col],
                                     irt_task,
                                     bank_row,
                                     bank_col);
                    IRT_CLOSE_FAILED_TEST(0);
                }
                for (uint8_t i = 0; i < BANK_WIDTH; i++) {
                    data_out[bank_row][bank_col].pix[i] =
                        rot_mem[bank_row][bank_col][rd_addr[bank_row][bank_col]].pix[i];
                }
                //---------------------------------------
                // ROTSIM_DV_INTGR
                IRT_TRACE_TO_LOG(4,
                                 IRTLOG,
                                 "%s:READ : task %d ROW:RADDR [%x,%x] ",
                                 my_string.c_str(),
                                 irt_task,
                                 bank_row,
                                 rd_addr[bank_row][bank_col]);
                IRT_TRACE_TO_LOG(4, IRTLOG, " COL[%d] DATA [", bank_col);
                for (uint8_t i = 0; i < BANK_WIDTH; i++) {
                    IRT_TRACE_TO_LOG(4, IRTLOG, "%d,", rot_mem[bank_row][bank_col][rd_addr[bank_row][bank_col]].pix[i]);
                }
                IRT_TRACE_TO_LOG(4, IRTLOG, " ] \n");
                //---------------------------------------
            } else {
                for (uint8_t i = 0; i < BANK_WIDTH; i++) {
                    data_out[bank_row][bank_col].pix[i] =
                        ((i & 0xf) << 4) | (bank_row << 1) | (bank_col); // not valid data
                }
            }
        }
    }
#ifdef CREATE_IMAGE_DUMPS_RM
    int width = task_int <= irt_top->num_of_tasks - 1
                    ? irt_top->irt_cfg
                              .rm_cfg[block_type][irt_top->irt_desc[task_int]
                                                      .image_par[block_type == e_irt_block_rot ? IIMAGE : MIMAGE]
                                                      .buf_mode]
                              .BufW >>
                          irt_top->irt_desc[task_int].image_par[block_type == e_irt_block_rot ? IIMAGE : MIMAGE].Ps
                    : 0; // image_pars[IIMAGE].S;
    //	int height = irt_desc[task_int].Yi_end - irt_desc[task_int].Yi_start + 1;
    int row_padded = (width * 3 + 3) & (~3);
#endif

    if (irt_mem_ctrl->rmwr_task_done && wr_port == 1) {

#ifdef CREATE_IMAGE_DUMPS_RM
        if (task_int == 0) {
            if (block_type == e_irt_block_rot) {
                rm_out_file = fopen("irt_rm_out.bmp", "wb");
                generate_bmp_header(
                    rm_out_file,
                    width,
                    irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode].BufH);
            } else {
                rm_out_file = fopen("irt_mm_out.txt", "w");
            }
            // IRT_TRACE("Openning RM output file\n");
        }

        uint64_t pix_val;
        if (task_int <= irt_top->num_of_tasks - 1) {
            if (block_type == e_irt_block_rot) {
                for (int i = 0;
                     i <
                     (irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode].BufH %
                      IMAGE_H);
                     i++) {
                    int i1 = (i + irt_mem_ctrl->start_line[task_int]) &
                             irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                 .BufH_mod;
                    for (int j = 0; j < width; j++) {
                        if (irt_top->irt_desc[task_int].image_par[IIMAGE].Ps == 0)
                            pix_val =
                                (uint64_t)rot_mem[i1 % ROW_BANKS][(j >> (IRT_ROT_MEM_PXL_SHIFT - 1) /*3*/) & 1]
                                                 [((i1 >> (int)log2(ROW_BANKS)) *
                                                   irt_top->irt_cfg
                                                       .rm_cfg[block_type]
                                                              [irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                                       .Buf_EpL) +
                                                  (j >> IRT_ROT_MEM_PXL_SHIFT)]
                                                     .pix[j & (IRT_ROT_MEM_BANK_WIDTH - 1)];
                        else {
                            pix_val =
                                (uint64_t)rot_mem[i1 % ROW_BANKS][(j >> (IRT_ROT_MEM_PXL_SHIFT - 1 - 1) /*3*/) & 1]
                                                 [((i1 >> (int)log2(ROW_BANKS)) *
                                                   irt_top->irt_cfg
                                                       .rm_cfg[block_type]
                                                              [irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                                       .Buf_EpL) +
                                                  (j >> (IRT_ROT_MEM_PXL_SHIFT - 1))]
                                                     .pix[2 * (j & (IRT_ROT_MEM_BANK_WIDTH / 2 - 1))];
                            pix_val +=
                                (uint64_t)rot_mem[i1 % ROW_BANKS][(j >> (IRT_ROT_MEM_PXL_SHIFT - 1 - 1) /*3*/) & 1]
                                                 [((i1 >> (int)log2(ROW_BANKS)) *
                                                   irt_top->irt_cfg
                                                       .rm_cfg[block_type]
                                                              [irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                                       .Buf_EpL) +
                                                  (j >> (IRT_ROT_MEM_PXL_SHIFT - 1))]
                                                     .pix[2 * (j & (IRT_ROT_MEM_BANK_WIDTH / 2 - 1)) + 1]
                                << 8;
                        }
                        irt_rm_image[task_int][i][j] =
                            (pix_val >> (irt_top->rot_pars[irt_top->task[task_int]].Pwi - 8));
                    }
                }
            } else {
                uint16_t log2Ps = irt_top->irt_desc[task_int].image_par[MIMAGE].Ps;
                uint16_t Ps     = (1 << irt_top->irt_desc[task_int].image_par[MIMAGE].Ps);
                for (int i = 0;
                     i <
                     irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[MIMAGE].buf_mode].BufH;
                     i++) {
                    int i1 = (i + irt_mem_ctrl->start_line[task_int]) &
                             irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[MIMAGE].buf_mode]
                                 .BufH_mod;
                    for (int j = 0; j < width; j++) {
                        int bank_col = (j >> (IRT_MESH_MEM_PXL_SHIFT - 1 - log2Ps)) & 1;
                        int bank_addr =
                            ((i1 >> (int)log2(ROW_BANKS)) *
                             irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[MIMAGE].buf_mode]
                                 .Buf_EpL) +
                            (j >> (IRT_MESH_MEM_PXL_SHIFT - log2Ps));
                        int pixel_start = Ps * (j & (IRT_MESH_MEM_BANK_WIDTH / Ps - 1));
                        pix_val         = 0;
                        for (int idx = 0; idx < Ps; idx++) {
                            pix_val +=
                                (uint64_t)rot_mem[i1 % ROW_BANKS][(j >> (IRT_MESH_MEM_PXL_SHIFT - 1 - log2Ps)) & 1]
                                                 [((i1 >> (int)log2(ROW_BANKS)) *
                                                   irt_top->irt_cfg
                                                       .rm_cfg[block_type]
                                                              [irt_top->irt_desc[task_int].image_par[MIMAGE].buf_mode]
                                                       .Buf_EpL) +
                                                  (j >> (IRT_MESH_MEM_PXL_SHIFT - log2Ps))]
                                                     .pix[Ps * (j & (IRT_MESH_MEM_BANK_WIDTH / Ps - 1)) + idx]
                                << (8 * idx);
                        }
                        irt_rm_image[task_int][i][j] = pix_val;
                    }
                }
            }
        }
#endif
        if (task_int == irt_top->num_of_tasks - 1) {
            // IRT_TRACE("Closing RM output file %d %d\n", height, width);
#ifdef CREATE_IMAGE_DUMPS_RM
            unsigned char* data = new unsigned char[row_padded];
            if (block_type == e_irt_block_rot) {
                for (int i = 0;
                     i <
                     irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode].BufH;
                     i++) {
                    for (int j = 0; j < width * 3; j += 3) {
                        data[j] = (unsigned char)
                            irt_rm_image[0]
                                        [irt_top->irt_cfg
                                             .rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                             .BufH -
                                         1 - i][j / 3];
                        data[j + 1] = (unsigned char)
                            irt_rm_image[1]
                                        [irt_top->irt_cfg
                                             .rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                             .BufH -
                                         1 - i][j / 3];
                        data[j + 2] = (unsigned char)
                            irt_rm_image[2]
                                        [irt_top->irt_cfg
                                             .rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[IIMAGE].buf_mode]
                                             .BufH -
                                         1 - i][j / 3];
                    }
                    fwrite(data, sizeof(unsigned char), row_padded, rm_out_file);
                }
            } else {
                for (int i = 0;
                     i <
                     irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[task_int].image_par[MIMAGE].buf_mode].BufH;
                     i++) {
                    for (int j = 0; j < irt_top->irt_desc[task_int].image_par[MIMAGE].W; j++)
                        fprintf(rm_out_file, "%llx\n", irt_rm_image[0][i][j]);
                }
            }
            fclose(rm_out_file);
#endif
        }
        task_int++;
    }
}

template <uint16_t ROW_BANKS, Eirt_block_type block_type>
bool IRT_top::IRT_OICC<ROW_BANKS, block_type>::run(int16_t& Xo0,
                                                   int16_t& Yo,
                                                   uint8_t& plane,
                                                   int16_t  YT,
                                                   int16_t  YB,
                                                   uint8_t& pixel_valid,
                                                   bool&    ofifo_push,
                                                   bool     ofifo_full,
                                                   bool&    task_end,
                                                   uint8_t& adj_proc_size)
{

    Eirt_blocks_enum     task_type    = block_type == e_irt_block_rot ? e_irt_oicc : e_irt_micc;
    uint8_t              iimage_type  = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    uint8_t              oimage_type  = OIMAGE; // block_type == e_irt_block_rot ? OIMAGE : OIMAGE;
    std::string          my_string    = getInstanceName();
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];

    uint16_t lines_to_release = 0, entries_to_release = 0;
    ofifo_push                          = 0;
    pixel_valid                         = 0;
    task_end                            = 0;
    irt_top->irt_top_sig.micc_task_done = 0;

    //   //---------------------------------------
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[task_type].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //   //---------------------------------------

    if (irt_top->task[task_type] >= irt_top->num_of_tasks) {
        return 1; // TBD - remove after multitask support
    }

    if (block_type == e_irt_block_mesh) {
        // irt_top->task[task_type]++;
        // return 0;
        if (irt_top->irt_desc[irt_top->task[task_type]].irt_mode !=
            e_irt_mesh) { // OICC works as MICC and task is not mesh, droppped
            IRT_TRACE("%s task %d is not mesh task and dropped at cycle %d\n",
                      m_name.c_str(),
                      irt_top->task[task_type],
                      cycle);
            irt_top->task[task_type]++;
            return 0;
        }
    }

    if (line < 0) {
        IRT_TRACE("Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_TRACE_TO_RES(test_res, "Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (task_done == 1) {
        if (irt_mem_ctrl->wr_done[irt_top->task[task_type]] == 1) { // all lines in rotation memory, can release the
                                                                    // task
            IRT_TRACE("%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[task_type], cycle);
            if (block_type == e_irt_block_mesh)
                irt_top->irt_top_sig.micc_task_done = 1;
            // release all lines at end of task
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                              irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1)) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness = (irt_mem_ctrl->fullness - (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                                                    irt_mem_ctrl->first_line[irt_top->task[task_type]] +
                                                                    1)); //%IRT_ROT_MEM_HEIGHT;
            } else {
                // irt_mem_ctrl->top_ptr = (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                // irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1) *
                // irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode].Buf_EpL)
                // & irt_top->irt_cfg.Hb_mod[block_type]; irt_mem_ctrl->fullness = (irt_mem_ctrl->fullness -
                // (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                // irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1) *
                // irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode].Buf_EpL);//%IRT_ROT_MEM_HEIGHT;

                lines_to_release = irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                   irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1;
                entries_to_release = (lines_to_release >> (int)log2(ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
            }
#ifdef STANDALONE_ROTATOR
            //			if (print_log_file)
            //				IRT_TRACE_TO_LOG(log_file, "Rotation memory update at end of task: cycle %d, task %d, line
            //%d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 					cycle,
            // irt_top->task[e_irt_oicc], line, irt_top->rm_first_line[irt_top->task[e_irt_oicc]],
            // irt_top->rm_last_line[irt_top->task[e_irt_oicc]], irt_top->rm_top_ptr, irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
#endif
            line                                                = 0;
            irt_mem_ctrl->lines_valid[irt_top->task[task_type]] = 0;
            irt_mem_ctrl->wr_done[irt_top->task[task_type]]     = 0;
            irt_mem_ctrl->rd_done[irt_top->task[task_type]]     = 0;
            task_done                                           = 0;
            irt_top->task[task_type]++;
            //			task_end = 1;
            if (irt_top->task[task_type] == irt_top->num_of_tasks) {
                return 1;
            } else {
                return 0; // may be removed TBD
            }
        } else { // not all lines in rotation memory, waiting
            return 0;
        }
    }

    if (block_type == e_irt_block_rot) {
        Xo0 = /*image_pars[OIMAGE].Ns*image_pars[OIMAGE].S +*/ pixel * 2 -
              irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].Xc;
        Yo = line * 2 - irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].Yc;
    } else {
        Xo0 = pixel * 2;
        Yo  = line * 2;
    }
    plane = irt_top->task[task_type];

    int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], Yi_fixed[IRT_ROT_MAX_PROC_SIZE];
    IRT_top::irt_iicc(irt_top->irt_desc[irt_top->task[task_type]],
                      Xo0,
                      Yo,
                      Xi_fixed,
                      Yi_fixed,
                      irt_top->irt_top_sig.mbli_out,
                      block_type); // calculating 8 coordinates

    if (block_type == e_irt_block_rot) {
        if (irt_top->irt_desc[irt_top->task[task_type]].rate_mode == e_irt_rate_adaptive_wxh) {
            adj_proc_size = irt_top->irt_rate_ctrl1(irt_top, Xi_fixed, Yi_fixed, IRT_ROT_MAX_PROC_SIZE);
        } else {
            adj_proc_size = irt_top->irt_desc[irt_top->task[task_type]].proc_size;
        }
    } else {
        // adj_proc_size = IRT_MESH_MAX_PROC_SIZE;
        adj_proc_size = irt_top->irt_desc[irt_top->task[task_type]].proc_size;
    }

    uint16_t remain_pixels = irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S - pixel;
    adj_proc_size          = (uint8_t)std::min(remain_pixels, (uint16_t)adj_proc_size);

    int16_t YT_int = (int16_t)((IRT_top::IRT_UTILS::irt_min_int64(Yi_fixed[0], Yi_fixed[adj_proc_size - 1])) >> 31);
    int16_t YB_int = (int16_t)((IRT_top::IRT_UTILS::irt_max_int64(Yi_fixed[0], Yi_fixed[adj_proc_size - 1])) >> 31) + 1;

    YB_int = irt_top->YB_adjustment(irt_top->irt_desc[irt_top->task[task_type]], block_type, YB_int);

    //	if (block_type == e_irt_block_mesh)
    // IRT_TRACE("%s: task %d, line %d, pixel %d, YB_int %d, YB %d, mem_first_line %d mem_last_line %d\n",
    // my_string.c_str(), irt_top->task[task_type], line, pixel, YB_int, YB,
    // irt_mem_ctrl->first_line[irt_top->task[task_type]], irt_mem_ctrl->last_line[irt_top->task[task_type]]);

    if (irt_top->task[task_type] == 0 && line == 0 && line_start == 1) {
        Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
        irt_mem_ctrl->first_line[irt_top->task[task_type]] = Yi_start; // first task init
        if (block_type == e_irt_block_rot)
            YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // rot_mem_first_line[task%2];
        else
            YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
    }

    if (!(line == 0 && line_start == 1)) {
        // in first pixel of first line YT is not valid because not calculated yet by IRT_IIICC
        //		if (YT < YT_min) IRT_TRACE_TO_LOG(log_file, "IRT_OICC: updating YT_min to %d, task %d, line %d, pixel
        //%d\n", YT, task, line, pixel);
        YT_min = YT < YT_min
                     ? YT
                     : YT_min; // tracking minimum Y value of input image required for output image line calculation
    }

    if (line_start == 1) {
        line_start = 0;
        if (line == 0) { // updating rotation memory registers in beginning of task
            Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
            irt_mem_ctrl->first_line[irt_top->task[task_type]] = Yi_start;
            irt_mem_ctrl->top_ptr                              = irt_mem_ctrl->start_line[irt_top->task[task_type]];
#ifdef STANDALONE_ROTATOR
            // IRT_TRACE_TO_LOG(log_file, "Rotation memory update in beginning of task: cycle %d, task %d, line %d,
            // YT_min %d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 		cycle, task[e_irt_oicc],
            // line,
            // YT_min, irt_top->rm_first_line[irt_top->task[e_irt_oicc]],
            // irt_top->rm_last_line[irt_top->task[e_irt_oicc]], rm_top_ptr , irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
#endif
            if (block_type == e_irt_block_rot)
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // rot_mem_first_line[task%2];
            else
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
        } else {
            // updating rotation memory registers in beginning of line
#ifdef STANDALONE_ROTATOR
            if (print_log_file)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "%s: %s memory release status (before): cycle %d, task %d, line %d, YT_min %d, "
                                 "first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
                                 my_string.c_str(),
                                 irt_mem_type_s[block_type],
                                 cycle,
                                 irt_top->task[task_type],
                                 line,
                                 YT_min,
                                 irt_mem_ctrl->first_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->last_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->top_ptr,
                                 irt_mem_ctrl->bot_ptr,
                                 irt_mem_ctrl->fullness);
            if (irt_top->task[task_type] != 0)
                if (print_log_file)
                    IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                     log_file,
                                     "task %d, first/last lines [%d:%d], top/bot ptr [%d:%d]\n",
                                     irt_top->task[task_type] - 1,
                                     irt_mem_ctrl->first_line[irt_top->task[task_type] - 1],
                                     irt_mem_ctrl->last_line[irt_top->task[task_type] - 1],
                                     irt_mem_ctrl->top_ptr,
                                     irt_mem_ctrl->bot_ptr);
#endif
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]])) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     (YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]])); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = YT_min;
            } else { // releasing only in group of 8 lines
                lines_to_release   = YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]];
                entries_to_release = (lines_to_release >> (uint8_t)log2(ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] += entries_to_release * ROW_BANKS;
            }
#ifdef STANDALONE_ROTATOR
            if (print_log_file)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "Rotation memory release status (after   ): cycle %d, task %d, line %d, YT_min %d, "
                                 "first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
                                 cycle,
                                 irt_top->task[task_type],
                                 line,
                                 YT_min,
                                 irt_mem_ctrl->first_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->last_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->top_ptr,
                                 irt_mem_ctrl->bot_ptr,
                                 irt_mem_ctrl->fullness);
                /*
                            if (task>0)
                                IRT_TRACE_TO_LOG(log_file, "task %d, first/last lines [%d:%d], top/bot ptr [%d:%d]\n",
                                        task[e_irt_oicc]-1, irt_top->rm_first_line[irt_top->task[e_irt_oicc]-1],
                   irt_top->rm_last_line[irt_top->task[e_irt_oicc]-1], rm_top_ptr, irt_top->rm_bot_ptr);
                */
#endif
            // rot_mem_first_line[task%2] = YT_min;
            if (block_type == e_irt_block_rot)
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // 0; //reseting every line
            else
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
        }
    }

    if (irt_mem_ctrl->last_line[irt_top->task[task_type]] < YB_int ||
        irt_mem_ctrl->lines_valid[irt_top->task[task_type]] == 0) { // && task==0 && line==0 && pixel==0) {
        // if (block_type == e_irt_block_rot)
        //	IRT_TRACE("Warning - %s: cycle %d not all lines from[%d:%d] of task %d are in memory [%d:%d], top/bot ptr
        //[%d:%d], fullness %d\n", my_string.c_str(), 		cycle, YT, YB_int, irt_top->task[task_type],
        // irt_mem_ctrl->first_line[irt_top->task[task_type]], irt_mem_ctrl->last_line[irt_top->task[task_type]],
        // irt_mem_ctrl->top_ptr, irt_mem_ctrl->bot_ptr, irt_mem_ctrl->fullness); IRT_TRACE_TO_LOG(log_file, "Warning -
        // IRT_OICC: cycle %d not all lines from[%d:%d] of task %d are in rotation memory [%d:%d]\n", cycle, YT, YB_int,
        // task, rot_mem_first_line[task%2], rot_mem_last_line[task%2]); exit(0);
#ifdef STANDALONE_ROTATOR
        // IRT_TRACE_TO_LOG(log_file, "Warning - IRT_OICC: cycle %d not all lines from[%d:%d] of task %d are in rotation
        // memory [%d:%d], top/bot ptr [%d:%d], fullness %d\n", cycle, YT, YB_int, task[e_irt_oicc],
        // irt_top->rm_first_line[irt_top->task[e_irt_oicc]], irt_top->rm_last_line[irt_top->task[e_irt_oicc]],
        // rm_top_ptr, irt_top->rm_bot_ptr, irt_top->rm_fullness);
#endif
        return 0;
    }

    if (irt_mem_ctrl->last_line[irt_top->task[task_type]] < YB && pixel != 0) {
        IRT_TRACE(
            "Error - %s: for pixel[%d,%d] task %d not all lines from[%d:%d] are in %s memory [%d:%d], YB_int=%d\n",
            my_string.c_str(),
            line,
            pixel,
            irt_top->task[task_type],
            YT,
            YB,
            irt_mem_type_s[block_type],
            irt_mem_ctrl->first_line[irt_top->task[task_type]],
            irt_mem_ctrl->last_line[irt_top->task[task_type]],
            YB_int);
        IRT_TRACE_TO_RES(
            test_res,
            " failed, %s pixel[%d,%d] task %d not all lines from[%d:%d] are in %s memory [%d:%d], YB_int=%d\n",
            my_string.c_str(),
            line,
            pixel,
            irt_top->task[task_type],
            YT,
            YB,
            irt_mem_type_s[block_type],
            irt_mem_ctrl->first_line[irt_top->task[task_type]],
            irt_mem_ctrl->last_line[irt_top->task[task_type]],
            YB_int);
        IRT_CLOSE_FAILED_TEST(0);
        return 0;
    }
    if ((YT_int < irt_mem_ctrl->first_line[irt_top->task[task_type]]) &&
        !(irt_top->task[task_type] == 0 &&
          line == 0)) { // this line already out of rotation memory, not checked in first entrance, YT is not valid
        IRT_TRACE("Error - %s: line %d out of %s memory [%d:%d] at task %d line %d, pixel %d, cycle %d\n",
                  my_string.c_str(),
                  YT_int,
                  irt_mem_type_s[block_type],
                  irt_mem_ctrl->first_line[irt_top->task[task_type]],
                  irt_mem_ctrl->last_line[irt_top->task[task_type]],
                  irt_top->task[task_type],
                  line,
                  pixel,
                  cycle);
        IRT_TRACE_TO_RES(test_res,
                         " failed, %s line %d out of %s memory [%d:%d] at task %d line %d, pixel %d, cycle %d\n",
                         my_string.c_str(),
                         YT_int,
                         irt_mem_type_s[block_type],
                         irt_mem_ctrl->first_line[irt_top->task[task_type]],
                         irt_mem_ctrl->last_line[irt_top->task[task_type]],
                         irt_top->task[task_type],
                         line,
                         pixel,
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    // if (block_type == e_irt_block_rot) {
    if (ofifo_full) { // no space in output FIFO
        return 0;
    }
    //}
    pixel_valid = adj_proc_size;
    ofifo_push  = 1;
    // if (task[e_irt_oicc]==8)
    // if (block_type == e_irt_block_mesh)
    // IRT_TRACE("%s: task %d, line %d, pixel %d valid %x at cycle %d\n", my_string.c_str(), irt_top->task[task_type],
    // line, pixel, pixel_valid, cycle);

//	IRT_TRACE_TO_LOG(log_file, "Plane %d, pixel[%3d,%3d], [yo,xo]=[%3d,%3d], YB_int %d, rot_mem_last_line %d\n", task,
// line, pixel, Yo, Xo0, YB_int, rot_mem_last_line[task%2]);
#ifdef STANDALONE_ROTATOR
    if (print_log_file)
        IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                         log_file,
                         "Plane %d, pixel[%3d,%3d], [yo,xo]=[%3d,%3d]\n",
                         irt_top->task[task_type],
                         line,
                         pixel,
                         Yo,
                         Xo0);
#endif

    if (block_type == e_irt_block_rot) {
        pixel += adj_proc_size;
    } else {
        if (ofifo_full == 0) {
            pixel += adj_proc_size;
        }
    }

    // IRT_TRACE("IRT_OICC line %d pixel %d of task %d is finished\n", line, pixel, task);
    // processing rate statistic calculation
    if (block_type == e_irt_block_rot) {
        if (adj_proc_size < irt_top->rot_pars[irt_top->task[e_irt_oicc]].min_proc_rate) {
            irt_top->rot_pars[irt_top->task[e_irt_oicc]].min_proc_rate = adj_proc_size;
        }
        if (adj_proc_size > irt_top->rot_pars[irt_top->task[e_irt_oicc]].max_proc_rate) {
            irt_top->rot_pars[irt_top->task[e_irt_oicc]].max_proc_rate = adj_proc_size;
        }

        irt_top->rot_pars[irt_top->task[e_irt_oicc]].acc_proc_rate += adj_proc_size;
        irt_top->rot_pars[irt_top->task[e_irt_oicc]].rate_hist[adj_proc_size - 1]++;
        irt_top->rot_pars[irt_top->task[e_irt_oicc]].cycles++;
    }

    if (pixel >= irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S) {
#ifdef STANDALONE_ROTATOR
        // IRT_TRACE("IRT_OICC line %d of task %d is finished, YT_min=%d, rotation memory range is [%d:%d]\n", line,
        // task, YT_min, irt_top->rm_first_line[irt_top->task%2], rm_last_line[task%2]);
#endif
        if ((irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1)) == 0) {
            pixel_valid = IRT_ROT_MAX_PROC_SIZE;
        } else {
            pixel_valid =
                irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1);
        }
        pixel_valid = adj_proc_size;

        // IRT_TRACE("%s: task %d, line %d, pixel %d valid %x is finished at cycle %d\n", my_string.c_str(),
        // irt_top->task[task_type], line, pixel, pixel_valid, cycle);

        pixel      = 0;
        line_start = 1;
        line++;
        if (line == irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].H) {
            task_done                                       = 1;
            task_end                                        = 1;
            irt_mem_ctrl->rd_done[irt_top->task[task_type]] = 1;
            if (irt_mem_ctrl->wr_done[irt_top->task[task_type]] == 0) {
                IRT_TRACE("%s task %d is done at cycle %d, waiting %s to complete writing\n",
                          my_string.c_str(),
                          irt_top->task[task_type],
                          cycle,
                          block_type == e_irt_block_rot ? "RMWM" : "MMWM");
            }

            //			rm_top_ptr = (rm_top_ptr + (YT_min - irt_top->rm_first_line[irt_top->task[e_irt_oicc]])) &
            // irt_top->irt_cfg.BufH_mod; 			irt_top->rm_fullness = (irt_top->rm_fullness - (YT_min -
            // irt_top->rm_first_line[irt_top->task[e_irt_oicc]]));//%IRT_ROT_MEM_HEIGHT;
            //			irt_top->rm_first_line[irt_top->task[e_irt_oicc]] = YT_min;
            // release all lines
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                              irt_mem_ctrl->first_line[irt_top->task[task_type]])) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness                             = (irt_mem_ctrl->fullness -
                                          (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                           irt_mem_ctrl->first_line[irt_top->task[task_type]])); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
            } else {
                lines_to_release = irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                   irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1;
                entries_to_release = (lines_to_release >> (int)log2(ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
            }
#if 0
			IRT_TRACE("IRT_OICC task %d is finished at cycle %d\n", task[e_irt_oicc], cycle);
			//release all lines at end of task
			rm_top_ptr  = (rm_top_ptr  + (irt_top->rm_last_line[irt_top->task[e_irt_oicc]]-irt_top->rm_first_line[irt_top->task[e_irt_oicc]]+1))&irt_top->irt_cfg.BufH_mod;
			irt_top->rm_fullness = (irt_top->rm_fullness - (irt_top->rm_last_line[irt_top->task[e_irt_oicc]]-irt_top->rm_first_line[irt_top->task[e_irt_oicc]]+1));//%IRT_ROT_MEM_HEIGHT;
#ifdef STANDALONE_ROTATOR
			IRT_TRACE_TO_LOG(log_file, "Rotation memory update at end of task: cycle %d, task %d, line %d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
						cycle, task[e_irt_oicc], line, irt_top->rm_first_line[irt_top->task[e_irt_oicc]], irt_top->rm_last_line[irt_top->task[e_irt_oicc]], rm_top_ptr , irt_top->rm_bot_ptr, irt_top->rm_fullness);
#endif
			line = 0;
			irt_top->rm_lines_valid[irt_top->task[e_irt_oicc]] = 0;
			task[e_irt_oicc]++;
			task_end = 1;
			if (task[e_irt_oicc]==num_of_tasks)
				return 1;
#endif
        }
    }

    // IRT_TRACE_TO_LOG(log_file, "IRT_OICC: task %d, line %d, pixel %d, pixel_valid %x\n", task, line, pixel,
    // pixel_valid);
    return 0;
}
template <Eirt_block_type block_type>
bool IRT_top::IRT_OICC2<block_type>::run(int16_t&              Xo0,
                                         int16_t&              Yo,
                                         uint8_t&              plane,
                                         uint8_t&              pixel_valid,
                                         bool&                 cfifo_push,
                                         uint8_t               cfifo_emptyness,
                                         bool                  mesh_ofifo_valid,
                                         bool&                 task_end,
                                         uint8_t&              adj_proc_size,
                                         irt_cfifo_data_struct cfifo_data_in[IRT_ROT_MAX_PROC_SIZE])
{

    Eirt_blocks_enum task_type   = block_type == e_irt_block_rot ? e_irt_oicc2 : e_irt_micc2;
    uint8_t          iimage_type = IIMAGE; // block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    uint8_t          oimage_type = OIMAGE; // block_type == e_irt_block_rot ? OIMAGE : OIMAGE;
    // uint8_t mem_pxl_shift = block_type == e_irt_block_rot ? IRT_ROT_MEM_PXL_SHIFT : IRT_ROT_MEM_PXL_SHIFT;
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];
    std::string          my_string    = getInstanceName();

    cfifo_push  = 0;
    pixel_valid = 0;
    task_end    = 0;
    if (irt_top->task[task_type] >= irt_top->num_of_tasks) {
        return 1; // TBD - remove after multitask support
    }

    //   //---------------------------------------
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[task_type].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //   //---------------------------------------
    if (block_type == e_irt_block_mesh) {
        // irt_top->task[task_type]++;
        // return 0;
        if (irt_top->irt_desc[irt_top->task[task_type]].irt_mode !=
            e_irt_mesh) { // OICC works as MICC and task is not mesh, droppped
            IRT_TRACE("%s task %d is not mesh task and dropped at cycle %d\n",
                      m_name.c_str(),
                      irt_top->task[task_type],
                      cycle);
            irt_top->task[task_type]++;
            return 0;
        }
    }

    if (line < 0) {
        IRT_TRACE("Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_TRACE_TO_RES(test_res, "Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (task_done == 1) {
        IRT_TRACE("%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[task_type], cycle);
        // release all lines at end of task
        line      = 0;
        task_done = 0;
        irt_top->task[task_type]++;
        //			task_end = 1;
        if (irt_top->task[task_type] == irt_top->num_of_tasks) {
            return 1;
        } else {
            return 0; // may be removed TBD
        }
    }

    if (block_type == e_irt_block_rot) {
        Xo0 = /*image_pars[OIMAGE].Ns*image_pars[OIMAGE].S +*/ pixel * 2 -
              irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].Xc;
        Yo = line * 2 - irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].Yc;
    } else {
        Xo0 = pixel * 2;
        Yo  = line * 2;
    }
    plane = irt_top->task[task_type];

    // adj_proc_size = (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_mesh) ? IRT_MESH_MAX_PROC_SIZE :
    // irt_top->irt_desc[irt_top->task[task_type]].proc_size;
    adj_proc_size = irt_top->irt_desc[irt_top->task[task_type]].proc_size;

    uint16_t remain_pixels = irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S - pixel;
    adj_proc_size          = (uint8_t)std::min(remain_pixels, (uint16_t)adj_proc_size);

    uint8_t irt_task = irt_top->task[task_type];
    int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], Yi_fixed[IRT_ROT_MAX_PROC_SIZE];

    //	if (block_type == e_irt_block_mesh)
    // IRT_TRACE("%s: task %d, line %d, pixel %d at cycle %d\n", my_string.c_str(), irt_top->task[task_type], line,
    // pixel, cycle);
    if (block_type == e_irt_block_rot &&
        (cfifo_emptyness < adj_proc_size ||
         (mesh_ofifo_valid == 0 && irt_top->irt_desc[irt_task].irt_mode == e_irt_mesh) ||
         irt_mem_ctrl->lines_valid[irt_top->task[task_type]] ==
             0)) { // no space in coordinate FIFO, not mesh output, start_line not set by RMWM
        // IRT_TRACE("%s return: task %d, line %d, pixel %d at cycle %d\n", my_string.c_str(), irt_top->task[task_type],
        // line, pixel, cycle); IRT_TRACE("cfifo_emptyness %d, adj_proc_size %d, mesh_ofifo_valid %d, lines_valid %d\n",
        // cfifo_emptyness, adj_proc_size, mesh_ofifo_valid, irt_mem_ctrl->lines_valid[irt_top->task[task_type]]);
        return 0;
    }

    IRT_top::irt_iicc(
        irt_top->irt_desc[irt_task], Xo0, Yo, Xi_fixed, Yi_fixed, irt_top->irt_top_sig.mbli_out, e_irt_block_rot);
    icc_cnt++; // ROTSIM_DV_INTGR
    int16_t YB_int = (int16_t)((IRT_top::IRT_UTILS::irt_max_int64(Yi_fixed[0], Yi_fixed[adj_proc_size - 1])) >> 31) + 1;

    if (block_type == e_irt_block_rot && YB_int > irt_mem_ctrl->last_line[irt_top->task[task_type]]) {
        // return 0; TBD, lead to deadlock with bg mode = 1 in case when line 0 cannot come to memory because of BuH
        // limitation
    }
    // IRT_TRACE("%s: task %d, line %d, pixel %d YB int %d last_line %d at cycle %d\n", my_string.c_str(),
    // irt_top->task[task_type], line, pixel, YB_int, irt_mem_ctrl->last_line[irt_top->task[task_type]], cycle);

    pixel_valid = adj_proc_size;
    cfifo_push  = 1;
    // if (task[e_irt_oicc]==8)
    // if (block_type == e_irt_block_rot)
    // IRT_TRACE("%s: task %d, line %d, pixel %d valid %x at cycle %d\n", my_string.c_str(), irt_top->task[task_type],
    // line, pixel, pixel_valid, cycle);

    pixel += adj_proc_size;

    // IRT_TRACE("IRT_OICC line %d pixel %d of task %d is finished\n", line, pixel, task);
    if (pixel >= irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S) {
#ifdef STANDALONE_ROTATOR
        // IRT_TRACE("IRT_OICC line %d of task %d is finished, YT_min=%d, rotation memory range is [%d:%d]\n", line,
        // task, YT_min, irt_top->rm_first_line[irt_top->task%2], rm_last_line[task%2]);
#endif
        if ((irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1)) == 0) {
            pixel_valid = IRT_ROT_MAX_PROC_SIZE;
        } else {
            pixel_valid =
                irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1);
        }
        pixel_valid = adj_proc_size;

        // IRT_TRACE("IRT_OICC: task %d line %d, pixel %d valid %x is finished at cycle %d\n", task[e_irt_oicc], line,
        // pixel, pixel_valid, cycle);

        pixel      = 0;
        line_start = 1;
        line++;
        if (line == irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].H) {
            task_done = 1;
            task_end  = 1;
        }
    }

    // cfifo push side
    int16_t /*Yi_start,*/ YT, YB; //, XL, XR;

    /*
                Yi_start = irt_top->yi_start_calc(block_type, irt_top->irt_desc[irt_top->task[task_type]]);
    */

    for (uint8_t i = 0; i < adj_proc_size; i++) {

        cfifo_data_in[i].Xi_fixed = Xi_fixed[i];
        cfifo_data_in[i].Yi_fixed = Yi_fixed[i];
        cfifo_data_in[i].task     = irt_task;
        cfifo_data_in[i].line     = (Yo + irt_top->irt_desc[irt_task].image_par[oimage_type].Yc) >> 1;
        cfifo_data_in[i].pixel    = ((Xo0 + irt_top->irt_desc[irt_task].image_par[oimage_type].Xc) >> 1) + i;
        cfifo_data_in[i].YT       = (int16_t)(Yi_fixed[i] >> 31);
        cfifo_data_in[i].YB       = cfifo_data_in[i].YT + 1;
        cfifo_data_in[i].XL       = (int16_t)(Xi_fixed[i] >> 31);
        cfifo_data_in[i].XR       = cfifo_data_in[i].XL + 1;

#ifdef IRT_USE_INTRP_REQ_CHECK // check if interpolation is required
        if ((Xi_fixed[i] & 0x7FFFFFFF) == 0) { // interpolation is not required
#ifdef IRT_USE_FLIP_FOR_MINUS1
            if (irt_top->irt_desc[irt_task].read_hflip == 0)
#endif
                cfifo_data_in[i].XR = cfifo_data_in[i].XL;
#ifdef IRT_USE_FLIP_FOR_MINUS1
            else
                cfifo_data_in[i].XL = cfifo_data_in[i].XR;
#endif
        }
        if ((Yi_fixed[i] & 0x7FFFFFFF) == 0) { // interpolation is not required
#ifdef IRT_USE_FLIP_FOR_MINUS1
            if (irt_top->irt_desc[irt_task].read_vflip == 0)
#endif
                cfifo_data_in[i].YB = cfifo_data_in[i].YT;
#ifdef IRT_USE_FLIP_FOR_MINUS1
            else
                cfifo_data_in[i].YT = cfifo_data_in[i].YB;
#endif
        }
#else
        cfifo_data_in[i].YB = irt_top->YB_adjustment(irt_top->irt_desc[irt_task], block_type, cfifo_data_in[i].YB);
#endif

#ifdef IRT_USE_FLIP_FOR_MINUS1
        if (irt_top->irt_desc[irt_task].read_vflip) {
            cfifo_data_in[i].YT--;
            cfifo_data_in[i].YB--;
        }

        if (irt_top->irt_desc[irt_task].read_hflip) {
            cfifo_data_in[i].XL--;
            cfifo_data_in[i].XR--;
        }
#endif
        if (irt_top->irt_desc[irt_task].bg_mode == e_irt_bg_frame_repeat) {
            // saturation to horizontal image boundaries
            if (Xi_fixed[i] < 0 || Xi_fixed[i] > (((int64_t)irt_top->irt_desc[irt_task].image_par[iimage_type].W - 1)
                                                  << 31)) { // outside of horizontal boundary
                cfifo_data_in[i].Xi_fixed = IRT_top::IRT_UTILS::irt_min_int64(
                    IRT_top::IRT_UTILS::irt_max_int64(0, Xi_fixed[i]),
                    ((int64_t)irt_top->irt_desc[irt_task].image_par[iimage_type].W - 1) << 31);
                cfifo_data_in[i].XL = (int16_t)(cfifo_data_in[i].Xi_fixed >> 31);
                cfifo_data_in[i].XR = cfifo_data_in[i].XL;
            }

            // saturation to vertical image  boundaries
            if (Yi_fixed[i] < 0 || Yi_fixed[i] > (((int64_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H - 1)
                                                  << 31)) { // outside of vertical boundary
                cfifo_data_in[i].Yi_fixed =
                    cfifo_data_in[i].Yi_fixed &
                    0xFFFFFFFF80000000; // IRT_top::IRT_UTILS::irt_min_int64(IRT_top::IRT_UTILS::irt_max_int64(0,
                                        // Yi_fixed[i]), ((int64_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H
                                        // - 1) << 31);
                cfifo_data_in[i].YT = (int16_t)(cfifo_data_in[i].Yi_fixed >> 31);
                cfifo_data_in[i].YB = cfifo_data_in[i].YT;
                IRT_TRACE_TO_LOG(
                    4,
                    IRTLOG,
                    "FRAME_REP YT:YB %d:%d Yi_fixed %llx H %d H<< %llx cFIFO-YI %llx \n",
                    cfifo_data_in[i].YB,
                    cfifo_data_in[i].YT,
                    (long long unsigned int)Yi_fixed[i],
                    irt_top->irt_desc[irt_task].image_par[iimage_type].H,
                    (long long unsigned int)(((int64_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H - 1) << 31),
                    (long long unsigned int)cfifo_data_in[i].Yi_fixed);
            }
#if 1
            if (irt_top->irt_desc[irt_task].bg_mode == e_irt_bg_frame_repeat) {
                // cfifo_data_in[i].YT = IRT_top::IRT_UTILS::irt_sat_int16(cfifo_data_in[i].YT, 0,
                // (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H - 1); cfifo_data_in[i].YB =
                // IRT_top::IRT_UTILS::irt_sat_int16(cfifo_data_in[i].YB, 0,
                // (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H - 1);
                cfifo_data_in[i].XL = IRT_top::IRT_UTILS::irt_sat_int16(
                    cfifo_data_in[i].XL, 0, (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].W - 1);
                cfifo_data_in[i].XR = IRT_top::IRT_UTILS::irt_sat_int16(
                    cfifo_data_in[i].XR, 0, (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].W - 1);
            }
#endif
        }

        YT = cfifo_data_in[i].YT;
        YB = cfifo_data_in[i].YB; // XL = cfifo_data_in[i].XL; XR = cfifo_data_in[i].XR;
#if 0
		if (irt_top->irt_desc[irt_task].rot90 && irt_top->irt_desc[irt_task].rot90_intv == 0) {
			YB--;
			cfifo_data_in[i].YB--;
		}
#endif

        bool rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS], rd_mode[IRT_ROT_MEM_ROW_BANKS],
            msb_lsb_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
            bg_flag[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH];
        int16_t rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
        uint8_t rd_shift[IRT_ROT_MEM_ROW_BANKS], bank_row[IRT_ROT_MEM_ROW_BANKS];
        memset(&rd_sel, 0, sizeof(rd_sel));
        memset(&rd_mode, 0, sizeof(rd_mode));
        memset(&msb_lsb_sel, 0, sizeof(msb_lsb_sel));
        memset(&bg_flag, 0, sizeof(bg_flag));
        memset(&rd_sel, 0, sizeof(rd_sel));
        memset(&rd_addr, 0, sizeof(rd_addr));
        memset(&rd_shift, 0, sizeof(rd_shift));
        memset(&bank_row, 0, sizeof(bank_row));

        irt_top->IRT_RMRM_block->run(cfifo_data_in[i].XL,
                                     cfifo_data_in[i].XR,
                                     cfifo_data_in[i].YT,
                                     cfifo_data_in[i].YB,
                                     bank_row,
                                     rd_sel,
                                     rd_addr,
                                     rd_mode,
                                     rd_shift,
                                     msb_lsb_sel,
                                     bank_row[0],
                                     bg_flag,
                                     irt_task,
                                     cfifo_push & 0,
                                     Xo0 + i,
                                     Yo,
                                     e_irt_rmrm_caller_irt_oicc);

#if 0
		IRT_TRACE("Task %d, [%d, %d] = [%llx, %llx][%.2f, %.2f] [YT:YB][%d:%d] [XL:XR][%d:%d] bank_row[0] %d, rd_sel[%d][%d][%d][%d] \n", irt_task,
			cfifo_data_in[i].line, cfifo_data_in[i].pixel, cfifo_data_in[i].Xi_fixed, cfifo_data_in[i].Yi_fixed, (double)cfifo_data_in[i].Xi_fixed / pow(2, 31), (double)cfifo_data_in[i].Yi_fixed / pow(2, 31),
			cfifo_data_in[i].YT, cfifo_data_in[i].YB, cfifo_data_in[i].XL, cfifo_data_in[i].XR,
			bank_row[0], rd_sel[bank_row[0]][0], rd_sel[bank_row[0]][1], rd_sel[(bank_row[0] + 1) % IRT_ROT_MEM_ROW_BANKS][0], rd_sel[(bank_row[0] + 1) % IRT_ROT_MEM_ROW_BANKS][1]);
#endif
        // IRT_TRACE_DBG("[%u, %u]: %f %f\n", cfifo_data_in[i].line, cfifo_data_in[i].pixel,
        // (double)cfifo_data_in[i].Xi_fixed / pow(2, 31), (double)cfifo_data_in[i].Yi_fixed / pow(2, 31));

        cfifo_data_in[i].Xi_start[0] =
            IRT_top::xi_start_calc(irt_top->irt_desc[irt_task], YT, e_irt_xi_start_calc_caller_oicc, irt_task);
        cfifo_data_in[i].Xi_start[1] =
            IRT_top::xi_start_calc(irt_top->irt_desc[irt_task], YB, e_irt_xi_start_calc_caller_oicc, irt_task);

        for (uint8_t bank_row1 = 0; bank_row1 < IRT_ROT_MEM_ROW_BANKS; bank_row1++) {
            for (uint8_t bank_col = 0; bank_col < 2; bank_col++) {
                if (rd_sel[bank_row1][bank_col] == 1) {
                    if (rd_addr[bank_row1][bank_col] < 0 || rd_addr[bank_row1][bank_col] >= IRT_ROT_MEM_BANK_HEIGHT) {
                        IRT_TRACE("%s read address %d invalid, task %d, line %d, pixel %d, bank_row %d, bank_col %d\n",
                                  my_string.c_str(),
                                  rd_addr[bank_row1][bank_col],
                                  irt_task,
                                  cfifo_data_in[i].line,
                                  cfifo_data_in[i].pixel,
                                  bank_row1,
                                  bank_col);
                        IRT_TRACE("[YT:YB]=[%d:%d], [XL:XR]=[%d:%d], Xi_starts=[%d,%d], banks[0,1]=[%d,%d], "
                                  "rd_sel[%d,%d][%d,%d]\n",
                                  cfifo_data_in[i].YT,
                                  cfifo_data_in[i].YB,
                                  cfifo_data_in[i].XL,
                                  cfifo_data_in[i].XR,
                                  cfifo_data_in[i].Xi_start[0],
                                  cfifo_data_in[i].Xi_start[1],
                                  bank_row[0],
                                  bank_row[1],
                                  rd_sel[bank_row[0]][0],
                                  rd_sel[bank_row[0]][1],
                                  rd_sel[bank_row[1]][0],
                                  rd_sel[bank_row[1]][1]);
                        IRT_TRACE_TO_RES(test_res,
                                         " failed, %s read address %d invalid, task %d, line %d, pixel %d, bank_row "
                                         "%d, bank_col %d\n",
                                         my_string.c_str(),
                                         rd_addr[bank_row1][bank_col],
                                         irt_task,
                                         cfifo_data_in[i].line,
                                         cfifo_data_in[i].pixel,
                                         bank_row1,
                                         bank_col);
                        IRT_CLOSE_FAILED_TEST(0);
                    }
                }
            }
        }

        for (uint8_t idx = 0; idx < 2; idx++) {
            cfifo_data_in[i].bank_row[idx] = bank_row[idx];
            cfifo_data_in[i].rd_shift[idx] = rd_shift[bank_row[idx]];
            cfifo_data_in[i].rd_mode[idx]  = rd_mode[bank_row[idx]];
            for (uint8_t bank_col = 0; bank_col < 2; bank_col++) {
                cfifo_data_in[i].rd_addr[idx][bank_col]     = rd_addr[bank_row[idx]][bank_col];
                cfifo_data_in[i].rd_sel[idx][bank_col]      = rd_sel[bank_row[idx]][bank_col];
                cfifo_data_in[i].msb_lsb_sel[idx][bank_col] = msb_lsb_sel[bank_row[idx]][bank_col];
                for (uint8_t j = 0; j < IRT_ROT_MEM_BANK_WIDTH; j++) {
                    cfifo_data_in[i].bg_flag[idx][bank_col][j] = bg_flag[bank_row[idx]][bank_col][j];
                }
            }
        }

#if 0
		IRT_TRACE("Task %d, line %d, pixel %d, bank_row[0] %d, bank_row[1] %d, rd_sel [%d][%d][%d][%d], addr[%x][%x][%x][%x]\n",
			irt_task, cfifo_data_in[i].line, cfifo_data_in[i].pixel,
			cfifo_data_in[i].bank_row[0], cfifo_data_in[i].bank_row[1],
			cfifo_data_in[i].rd_sel[0][0], cfifo_data_in[i].rd_sel[0][1], cfifo_data_in[i].rd_sel[1][0], cfifo_data_in[i].rd_sel[1][1],
			cfifo_data_in[i].rd_addr[0][0], cfifo_data_in[i].rd_addr[0][1], cfifo_data_in[i].rd_addr[1][0], cfifo_data_in[i].rd_addr[1][1]);
#endif
    }
    return 0;
}

template <Eirt_block_type block_type>
void IRT_top::IRT_CFIFO<block_type>::run(bool                  push,
                                         bool                  pop,
                                         uint8_t               push_size,
                                         uint8_t               pop_size,
                                         irt_cfifo_data_struct data_in[IRT_ROT_MAX_PROC_SIZE],
                                         irt_cfifo_data_struct data_out[IRT_ROT_MAX_PROC_SIZE],
                                         uint8_t&              emptyness,
                                         uint8_t&              fullness)
{

    std::string my_string = getInstanceName();

    if (cfifo_fullness > IRT_CFIFO_SIZE) {
        IRT_TRACE("Error: %s: wp=%d, rp=%d, fullness=%d at cycle %d\n",
                  my_string.c_str(),
                  cfifo_wp,
                  cfifo_rp,
                  cfifo_fullness,
                  cycle);
        IRT_TRACE_TO_RES(test_res, " failed, %s overflow at cycle %d\n", my_string.c_str(), cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (push) {
        if (cfifo_fullness + push_size > IRT_CFIFO_SIZE) {
            IRT_TRACE("Push to full %s: wp=%d, rp=%d, push_size=%d, fullness=%d at cycle %d\n",
                      my_string.c_str(),
                      cfifo_wp,
                      cfifo_rp,
                      push_size,
                      cfifo_fullness,
                      cycle);
            IRT_TRACE_TO_RES(test_res, " failed, %s overflow at cycle %d\n", my_string.c_str(), cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }
        for (uint8_t i = 0; i < push_size; i++) {
            cfifo[(cfifo_wp + i) % IRT_CFIFO_SIZE] = data_in[i];
        }
        cfifo_wp = (cfifo_wp + push_size) % IRT_CFIFO_SIZE;
        cfifo_fullness += push_size;
        IRT_TRACE_TO_LOG(
            4,
            IRTLOG,
            "%s push: wp = %d, rp = %d, push size = %d, fullness = %d, task = %d, line = %d, pixel = %d at cycle %d\n",
            my_string.c_str(),
            cfifo_wp,
            cfifo_rp,
            push_size,
            cfifo_fullness,
            data_in[0].task,
            data_in[0].line,
            data_in[0].pixel,
            cycle);
    }

    if (pop) {

        if (cfifo_fullness - pop_size < 0) {
            IRT_TRACE("Pop from empty %s: wp=%d, rp=%d, pop_size=%d, fullness=%d at cycle %d\n",
                      my_string.c_str(),
                      cfifo_wp,
                      cfifo_rp,
                      pop_size,
                      cfifo_fullness,
                      cycle);
            IRT_TRACE_TO_RES(test_res, " failed, %s underflow at cycle %d\n", my_string.c_str(), cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }

        cfifo_rp = (cfifo_rp + pop_size) % IRT_CFIFO_SIZE;
        cfifo_fullness -= pop_size;

        IRT_TRACE_TO_LOG(
            4,
            IRTLOG,
            "%s pop: wp = %d, rp = %d, pop size = %d, fullness = %d, task = %d, line = %d, pixel = %d at cycle %d\n",
            my_string.c_str(),
            cfifo_wp,
            cfifo_rp,
            pop_size,
            cfifo_fullness,
            cfifo[cfifo_rp].task,
            cfifo[cfifo_rp].line,
            cfifo[cfifo_rp].pixel,
            cycle);
    }

    for (uint8_t i = 0; i < IRT_ROT_MAX_PROC_SIZE; i++) {
        data_out[i] = cfifo[(cfifo_rp + i) % IRT_CFIFO_SIZE];
    }

    //	if (pop)
    // IRT_TRACE("IFIFO pop: wp = %d, rp = %d, slot=%d, fullness=%d, line %d, Xi_start %d\n", irt_ififo_wp,
    // irt_ififo_rp, irt_ififo_r_entry,fifo_fullness, metaout.line, metaout.Xi_start);

    fullness  = cfifo_fullness;
    emptyness = IRT_CFIFO_SIZE - cfifo_fullness;
}

template <Eirt_block_type block_type>
void IRT_top::IRT_IIIRC<block_type>::run(int16_t  Xo0,
                                         int16_t  Yo,
                                         int64_t  Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                         int64_t  Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                         int16_t& XL,
                                         int16_t& XR,
                                         int16_t& YT,
                                         int16_t& YB,
                                         uint8_t  irt_task,
                                         bool     ofifo_push,
                                         uint8_t& adj_proc_size)
{

    // Eirt_blocks_enum task_type = block_type == e_irt_block_rot ? irt_oicc : irt_micc;
    // uint8_t iimage_type = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    // uint8_t oimage_type = block_type == e_irt_block_rot ? OIMAGE : MIMAGE;
    uint8_t     row_banks = block_type == e_irt_block_rot ? IRT_ROT_MEM_ROW_BANKS : IRT_MESH_MEM_ROW_BANKS;
    std::string my_string = getInstanceName();
    int16_t     line, pixel;

    uint16_t remain_pixels;
    if (block_type == e_irt_block_rot) {
        line          = (Yo + irt_top->irt_desc[irt_task].image_par[OIMAGE].Yc) >> 1;
        pixel         = (Xo0 + irt_top->irt_desc[irt_task].image_par[OIMAGE].Xc) >> 1;
        remain_pixels = irt_top->irt_desc[irt_task].image_par[OIMAGE].S - pixel;
    } else {
        line  = Yo >> 1;
        pixel = Xo0 >> 1;
        if (irt_top->irt_desc[irt_task].irt_mode != e_irt_mesh) { // IIIRC works as MICC and task is not mesh, dropped
            return;
        }
        remain_pixels = irt_top->irt_desc[irt_task].image_par[OIMAGE].S - pixel;
    }
    adj_proc_size = (uint8_t)std::min(remain_pixels, (uint16_t)adj_proc_size);

    // fixed and float calculation
    IRT_top::irt_iicc(
        irt_top->irt_desc[irt_task], Xo0, Yo, Xi_fixed, Yi_fixed, irt_top->irt_top_sig.mbli_out, block_type);
    uint8_t idx_last = adj_proc_size - 1;

    XL = (int16_t)((IRT_top::IRT_UTILS::irt_min_int64(Xi_fixed[0], Xi_fixed[idx_last])) >> 31);
    XR = (int16_t)((IRT_top::IRT_UTILS::irt_max_int64(Xi_fixed[0], Xi_fixed[idx_last])) >> 31) + 1;
    YT = (int16_t)((IRT_top::IRT_UTILS::irt_min_int64(Yi_fixed[0], Yi_fixed[idx_last])) >> 31);
    YB = (int16_t)((IRT_top::IRT_UTILS::irt_max_int64(Yi_fixed[0], Yi_fixed[idx_last])) >> 31) + 1;

    if (ofifo_push) {
        // IRT_TRACE("%s: task %d, pixel[%d, %d:%d] [XL:XR] [%d:%d]\n", my_string.c_str(), irt_task, line, pixel, pixel
        // + remain_pixels - 1, XL, XR); IRT_TRACE("%s: pixel[%d, %d:%d] requires [%d,%d] lines of input image\n",
        // my_string.c_str(), line, pixel, pixel + remain_pixels - 1, YT, YB); IRT_TRACE("%s: pixel[%d, %d:%d] requires
        // [%f,%f] lines of input image\n", my_string.c_str(), line, pixel, pixel + remain_pixels - 1,
        // Yi_fixed[0]/pow(2.0, 31), Yi_fixed[idx_last] /pow(2.0, 31)); exit (0);
    }

    YB = irt_top->YB_adjustment(irt_top->irt_desc[irt_task], block_type, YB);

    if (block_type == e_irt_block_rot) {
#ifdef IRT_USE_FLIP_FOR_MINUS1
        if (irt_top->irt_desc[irt_task].read_vflip == 1)
            YT--;
        if (irt_top->irt_desc[irt_task].read_hflip == 1)
            XL--;
        if (irt_top->irt_desc[irt_task].read_hflip == 1)
            XR--;
#endif
    } else {
        if (irt_top->irt_desc[irt_task].mesh_sparse_h == 0)
            XR--;
    }

    if ((YB - YT > row_banks - 1) && error_flag == 0) {
        IRT_TRACE("%s error: pixel[%d, %d:%d] requires more than %u lines of %s image [%d,%d]\n",
                  my_string.c_str(),
                  line,
                  pixel,
                  pixel + adj_proc_size - 1,
                  row_banks,
                  block_type == e_irt_block_rot ? "input" : "mesh",
                  YT,
                  YB);
        error_flag = 1;
        IRT_TRACE_TO_RES(test_res,
                         " failed, %s pixel[%d, %d:%d] requires more than 8 lines of input image [%d,%d]\n",
                         my_string.c_str(),
                         line,
                         pixel,
                         pixel + 7,
                         YT,
                         YB);
        IRT_CLOSE_FAILED_TEST(0);
    }

    YB = IRT_UTILS::irt_min_int16(YB, YT + row_banks - 1);

    if (ofifo_push) {
#ifdef STANDALONE_ROTATOR
        if (print_log_file)
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             "IRT_IIIRC: task=%d, Xo0=%d, Yo=%d, Xi0=%lld, Yi0=%lld, Xi7[%d]=%lld, Yi7=%lld\n",
                             irt_task,
                             Xo0,
                             Yo,
                             Xi_fixed[0],
                             Yi_fixed[0],
                             idx_last,
                             Xi_fixed[idx_last],
                             Yi_fixed[idx_last]);
#endif
        // IRT_TRACE_TO_LOG(log_file, "%d, %d\n", YT, YB);
        // IRT_TRACE_TO_LOG(log_file, "%f, %f\n",
        // Yi0_fixed/pow(2.0,irt_top->irt_desc[irt_task].TOTAL_PREC),Yi7_fixed/pow(2.0,irt_top->irt_desc[irt_task].TOTAL_PREC));
        // //IRT_TRACE_TO_LOG(log_file, "IRT_IIIRC: XL=%d, XR=%d, YT=%d, YB=%d\n", XL, XR, YT, YB);
    }
}

// based on original OICC & IRT_IIIRC
template <Eirt_block_type block_type>
bool IRT_top::IRT_IIIRC2<block_type>::run(int16_t&              Xo0,
                                          int16_t&              Yo,
                                          uint8_t&              pixel_valid,
                                          bool&                 ofifo_push,
                                          bool                  ofifo_full,
                                          bool&                 task_end,
                                          uint8_t&              adj_proc_size,
                                          int64_t               Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                          int64_t               Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                                          int16_t&              XL,
                                          int16_t&              XR,
                                          int16_t&              YT,
                                          int16_t&              YB,
                                          uint8_t&              irt_task,
                                          uint8_t               cfifo_fullness,
                                          bool&                 cfifo_pop,
                                          uint8_t&              cfifo_pop_size,
                                          irt_cfifo_data_struct cfifo_data_out[IRT_ROT_MAX_PROC_SIZE],
                                          bool                  rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
                                          int16_t               rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS])
{

    Eirt_blocks_enum     task_type    = block_type == e_irt_block_rot ? e_irt_iiirc : e_irt_micc;
    uint8_t              iimage_type  = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    uint8_t              oimage_type  = OIMAGE; // block_type == e_irt_block_rot ? OIMAGE : OIMAGE;
    uint8_t              row_banks    = block_type == e_irt_block_rot ? IRT_ROT_MEM_ROW_BANKS : IRT_MESH_MEM_ROW_BANKS;
    std::string          my_string    = getInstanceName();
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];

    //   //---------------------------------------
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[task_type].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //   //---------------------------------------
    for (uint8_t i = 0; i < IRT_ROT_MAX_PROC_SIZE; i++) {
        Xi_fixed[i] = cfifo_data_out[i].Xi_fixed;
        Yi_fixed[i] = cfifo_data_out[i].Yi_fixed;
    }

    if (ofifo_push && cfifo_data_out[0].task != irt_top->task[task_type]) {
        IRT_TRACE("%s: task %d is not equal to task %d from CFIFO at cycle %d\n",
                  m_name.c_str(),
                  irt_top->task[task_type],
                  cfifo_data_out[0].task,
                  cycle);
        IRT_TRACE_TO_RES(test_res,
                         "failed, %s task %d is not equal to task %d from CFIFO at cycle %d\n",
                         m_name.c_str(),
                         irt_top->task[task_type],
                         cfifo_data_out[0].task,
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (ofifo_push && cfifo_data_out[0].line != line) {
        IRT_TRACE("%s: line %d is not equal to line %d from CFIFO at task %d at cycle %d\n",
                  m_name.c_str(),
                  Yo,
                  cfifo_data_out[0].line,
                  irt_top->task[task_type],
                  cycle);
        IRT_TRACE_TO_RES(test_res,
                         "failed, %s line %d is not equal to line %d from CFIFO at task %d at cycle %d\n",
                         m_name.c_str(),
                         line,
                         cfifo_data_out[0].line,
                         irt_top->task[task_type],
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (ofifo_push && cfifo_data_out[0].pixel != pixel) {
        IRT_TRACE("%s: pixel %d is not equal to pixel %d from CFIFO at task %d at cycle %d\n",
                  m_name.c_str(),
                  pixel,
                  cfifo_data_out[0].pixel,
                  irt_top->task[task_type],
                  cycle);
        IRT_TRACE_TO_RES(test_res,
                         "failed, %s pixel %d is not equal to pixel %d from CFIFO at task %d at cycle %d\n",
                         m_name.c_str(),
                         Xo0,
                         cfifo_data_out[0].pixel,
                         irt_top->task[task_type],
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    /****************/
    /* OICC section	*/
    /****************/
    uint16_t lines_to_release = 0, entries_to_release = 0;
    ofifo_push  = 0;
    pixel_valid = 0;
    task_end    = 0;
    if (irt_top->task[task_type] >= irt_top->num_of_tasks) {
        return 1; // TBD - remove after multitask support
    }

    if (block_type == e_irt_block_mesh) {
        // irt_top->task[task_type]++;
        // return 0;
        if (irt_top->irt_desc[irt_top->task[task_type]].irt_mode !=
            e_irt_mesh) { // OICC works as MICC and task is not mesh, droppped
            IRT_TRACE("%s task %d is not mesh task and dropped at cycle %d\n",
                      m_name.c_str(),
                      irt_top->task[task_type],
                      cycle);
            irt_top->task[task_type]++;
            return 0;
        }
    }

    if (line < 0) {
        IRT_TRACE("Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_TRACE_TO_RES(test_res, "Error - %s: line < 0 at cycle %d\n", my_string.c_str(), cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (task_done == 1) {
        if (irt_mem_ctrl->wr_done[irt_top->task[task_type]] == 1) { // all lines in rotation memory, can release the
                                                                    // task
            IRT_TRACE("%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[task_type], cycle);
            // release all lines at end of task
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                              irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1)) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness = (irt_mem_ctrl->fullness - (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                                                    irt_mem_ctrl->first_line[irt_top->task[task_type]] +
                                                                    1)); //%IRT_ROT_MEM_HEIGHT;
            } else {
                // irt_mem_ctrl->top_ptr = (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                // irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1) *
                // irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode].Buf_EpL)
                // & irt_top->irt_cfg.Hb_mod[block_type]; irt_mem_ctrl->fullness = (irt_mem_ctrl->fullness -
                // (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                // irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1) *
                // irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode].Buf_EpL);//%IRT_ROT_MEM_HEIGHT;

                lines_to_release = irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                   irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1;
                entries_to_release = (lines_to_release >> (uint8_t)log2(IRT_ROT_MEM_ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
            }
#ifdef STANDALONE_ROTATOR
            //			if (print_log_file)
            //				IRT_TRACE_TO_LOG(log_file, "Rotation memory update at end of task: cycle %d, task %d, line
            //%d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 					cycle,
            // irt_top->task[e_irt_oicc], line, irt_top->rm_first_line[irt_top->task[e_irt_oicc]],
            // irt_top->rm_last_line[irt_top->task[e_irt_oicc]], irt_top->rm_top_ptr, irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
#endif
            line                                                = 0;
            irt_mem_ctrl->lines_valid[irt_top->task[task_type]] = 0;
            irt_mem_ctrl->wr_done[irt_top->task[task_type]]     = 0;
            irt_mem_ctrl->rd_done[irt_top->task[task_type]]     = 0;
            task_done                                           = 0;
            irt_top->task[task_type]++;
            //			task_end = 1;
            if (irt_top->task[task_type] == irt_top->num_of_tasks) {
                return 1;
            } else {
                return 0; // may be removed TBD
            }
        } else { // not all lines in rotation memory, waiting
            return 0;
        }
    }

    Xo0      = pixel;
    Yo       = line;
    irt_task = irt_top->task[task_type];

    uint16_t remain_pixels = irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S - pixel;
    uint8_t  max_proc_size =
        (uint8_t)std::min((int16_t)remain_pixels, (int16_t)irt_top->irt_desc[irt_top->task[task_type]].proc_size);

    if (block_type == e_irt_block_rot && cfifo_fullness < max_proc_size) { // not enough pixels in CFIFO
        // IRT_TRACE("Warning - %s: task %d pixel[%d, %d:%d]: cycle %d not enough pixels in CFIFO, fullness %d, max proc
        // size %d\n", my_string.c_str(), irt_top->task[task_type], line, pixel, pixel + adj_proc_size - 1, cycle,
        // cfifo_fullness, max_proc_size);
        return 0;
    }

    switch (irt_top->irt_desc[irt_top->task[task_type]].rate_mode) {
        case e_irt_rate_fixed:
            adj_proc_size = irt_top->irt_rate_ctrl2(
                irt_top, cfifo_data_out, rd_sel, rd_addr, max_proc_size, e_irt_rate_adaptive_2x2);
            break;
        case e_irt_rate_adaptive_wxh:
            adj_proc_size = irt_top->irt_rate_ctrl1(
                irt_top, Xi_fixed, Yi_fixed, irt_top->irt_desc[irt_top->task[task_type]].proc_size);
            break;
        case e_irt_rate_adaptive_2x2:
            adj_proc_size = irt_top->irt_rate_ctrl2(
                irt_top, cfifo_data_out, rd_sel, rd_addr, max_proc_size, e_irt_rate_adaptive_2x2);
            break;
        default: adj_proc_size = irt_top->irt_desc[irt_top->task[task_type]].proc_size; break;
    }

    // checking that proc_size is e_irt_rate_fixed can be meet w/o memory contention
    if (irt_top->irt_desc[irt_top->task[task_type]].rate_mode == e_irt_rate_fixed) {
        if (adj_proc_size < max_proc_size) { // cannot reach proc size
            IRT_TRACE("%s error: pixel[%d, %d:%d] in fixed rate mode does not reach programmble proc_size %d, reaches "
                      "%d size\n",
                      my_string.c_str(),
                      cfifo_data_out[0].line,
                      cfifo_data_out[0].pixel,
                      pixel + max_proc_size - 1,
                      irt_top->irt_desc[irt_top->task[task_type]].proc_size,
                      adj_proc_size);
            error_flag = 1;
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s pixel[%d, %d:%d] in fixed rate mode does not reach programmble proc_size %d, "
                             "reaches %d size\n",
                             my_string.c_str(),
                             cfifo_data_out[0].line,
                             cfifo_data_out[0].pixel,
                             pixel + max_proc_size - 1,
                             irt_top->irt_desc[irt_top->task[task_type]].proc_size,
                             adj_proc_size);
            IRT_CLOSE_FAILED_TEST(0);
        }
    }

    adj_proc_size = (uint8_t)std::min((int16_t)adj_proc_size, (int16_t)max_proc_size);

    // if (block_type == e_irt_block_rot && irt_top->task[task_type] == 1 && line == 33 && pixel == 96)
    //	IRT_TRACE("AAA");

    int16_t YT_int = (int16_t)((IRT_top::IRT_UTILS::irt_min_int64(Yi_fixed[0], Yi_fixed[adj_proc_size - 1])) >> 31);
    int16_t YB_int = (int16_t)((IRT_top::IRT_UTILS::irt_max_int64(Yi_fixed[0], Yi_fixed[adj_proc_size - 1])) >> 31) + 1;
    // YT = (int16_t)((IRT_top::IRT_UTILS::irt_min_int64(Yi_fixed[0], Yi_fixed[remain_pixels - 1])) >> 31);

    YT_int = (int16_t)(IRT_top::IRT_UTILS::irt_min_coord(Yi_fixed, adj_proc_size) >> 31);
    YB_int = (int16_t)(IRT_top::IRT_UTILS::irt_max_coord(Yi_fixed, adj_proc_size) >> 31) + 1;

    YB_int = irt_top->YB_adjustment(irt_top->irt_desc[irt_top->task[task_type]], block_type, YB_int);

    // IRT_TRACE("%s: task %d, line %d, pixel %d, YB_int %d, YB %d, mem_first_line %d mem_last_line %d at cycle %d\n",
    // my_string.c_str(), irt_top->task[task_type], line, pixel, YB_int, YB,
    // irt_mem_ctrl->first_line[irt_top->task[task_type]], irt_mem_ctrl->last_line[irt_top->task[task_type]], cycle);

    if (irt_top->task[task_type] == 0 && line == 0 && line_start == 1) {
        Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
        irt_mem_ctrl->first_line[irt_top->task[task_type]] = Yi_start; // first task init
        if (block_type == e_irt_block_rot)
            YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // rot_mem_first_line[task%2];
        else
            YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
    }

    // YT = YT_next;
    if (!(line == 0 && line_start == 1)) {
        // in first pixel of first line YT is not valid because not calculated yet by IRT_IIICC
        // if (YT < YT_min) IRT_TRACE("%s: updating YT_min to %d, task %d, line %d, pixel %d at cycle %d\n",
        // my_string.c_str(), YT, irt_top->task[task_type], line, pixel, cycle); YT_min = YT < YT_min ? YT : YT_min; //
        // tracking minimum Y value of input image required for output image line calculation
    }

    if (line_start == 1) {
        line_start = 0;
        if (line == 0) { // updating rotation memory registers in beginning of task
            Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_top->task[task_type]], block_type);
            irt_mem_ctrl->first_line[irt_top->task[task_type]] = Yi_start;
            irt_mem_ctrl->top_ptr                              = irt_mem_ctrl->start_line[irt_top->task[task_type]];
#ifdef STANDALONE_ROTATOR
            // IRT_TRACE_TO_LOG(log_file, "Rotation memory update in beginning of task: cycle %d, task %d, line %d,
            // YT_min %d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 		cycle, task[e_irt_oicc],
            // line,
            // YT_min, irt_top->rm_first_line[irt_top->task[e_irt_oicc]],
            // irt_top->rm_last_line[irt_top->task[e_irt_oicc]], rm_top_ptr , irt_top->rm_bot_ptr,
            // irt_top->rm_fullness);
#endif
            if (block_type == e_irt_block_rot)
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // rot_mem_first_line[task%2];
            else
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
        } else {
            // updating rotation memory registers in beginning of line
#ifdef STANDALONE_ROTATOR
            if (print_log_file)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "%s: %s memory release status (before): cycle %d, task %d, line %d, YT_min %d, "
                                 "first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
                                 my_string.c_str(),
                                 irt_mem_type_s[block_type],
                                 cycle,
                                 irt_top->task[task_type],
                                 line,
                                 YT_min,
                                 irt_mem_ctrl->first_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->last_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->top_ptr,
                                 irt_mem_ctrl->bot_ptr,
                                 irt_mem_ctrl->fullness);
            if (irt_top->task[task_type] != 0)
                if (print_log_file)
                    IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                     log_file,
                                     "task %d, first/last lines [%d:%d], top/bot ptr [%d:%d]\n",
                                     irt_top->task[task_type] - 1,
                                     irt_mem_ctrl->first_line[irt_top->task[task_type] - 1],
                                     irt_mem_ctrl->last_line[irt_top->task[task_type] - 1],
                                     irt_mem_ctrl->top_ptr,
                                     irt_mem_ctrl->bot_ptr);

                    // IRT_TRACE("%s: %s memory release status (before): cycle %d, task %d, line %d, YT_min %d,
                    // first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 	my_string.c_str(),
                    // irt_mem_type_s[block_type], cycle, irt_top->task[task_type], line, YT_min,
                    // irt_mem_ctrl->first_line[irt_top->task[task_type]],
                    // irt_mem_ctrl->last_line[irt_top->task[task_type]], irt_mem_ctrl->top_ptr, irt_mem_ctrl->bot_ptr,
                    // irt_mem_ctrl->fullness);

#endif
            IRT_TRACE_TO_LOG(4,
                             IRTLOG,
                             "%s: %s memory release status (before): cycle %d, task %d, line %d, YT_min %d, first/last "
                             "lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
                             my_string.c_str(),
                             irt_mem_type_s[block_type],
                             cycle,
                             irt_top->task[task_type],
                             line,
                             YT_min,
                             irt_mem_ctrl->first_line[irt_top->task[task_type]],
                             irt_mem_ctrl->last_line[irt_top->task[task_type]],
                             irt_mem_ctrl->top_ptr,
                             irt_mem_ctrl->bot_ptr,
                             irt_mem_ctrl->fullness);
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]])) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     (YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]])); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = YT_min;
            } else { // releasing only in group of 8 lines
                lines_to_release   = YT_min - irt_mem_ctrl->first_line[irt_top->task[task_type]];
                entries_to_release = (lines_to_release >> (uint8_t)log2(IRT_ROT_MEM_ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] += entries_to_release * IRT_ROT_MEM_ROW_BANKS;
            }
            IRT_TRACE_TO_LOG(
                4,
                IRTLOG,
                "%s: Rotation memory release status (after): cycle %d, task %d, line %d, YT_min %d, first/last lines "
                "[%d:%d], top/bot ptr [%d:%d], fullness %d lines_to_release %d entries_to_release %d Buf_EpL %d\n",
                my_string.c_str(),
                cycle,
                irt_top->task[task_type],
                line,
                YT_min,
                irt_mem_ctrl->first_line[irt_top->task[task_type]],
                irt_mem_ctrl->last_line[irt_top->task[task_type]],
                irt_mem_ctrl->top_ptr,
                irt_mem_ctrl->bot_ptr,
                irt_mem_ctrl->fullness,
                lines_to_release,
                entries_to_release,
                irt_top->irt_cfg
                    .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                    .Buf_EpL);
#ifdef STANDALONE_ROTATOR
            if (print_log_file)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "Rotation memory release status (after   ): cycle %d, task %d, line %d, YT_min %d, "
                                 "first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n",
                                 cycle,
                                 irt_top->task[task_type],
                                 line,
                                 YT_min,
                                 irt_mem_ctrl->first_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->last_line[irt_top->task[task_type]],
                                 irt_mem_ctrl->top_ptr,
                                 irt_mem_ctrl->bot_ptr,
                                 irt_mem_ctrl->fullness);

                // IRT_TRACE("Rotation memory release status (after   ): cycle %d, task %d, line %d, YT_min %d,
                // first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 	cycle, irt_top->task[task_type],
                // line,
                // YT_min, irt_mem_ctrl->first_line[irt_top->task[task_type]],
                // irt_mem_ctrl->last_line[irt_top->task[task_type]], irt_mem_ctrl->top_ptr, irt_mem_ctrl->bot_ptr,
                // irt_mem_ctrl->fullness);
                /*
                            if (task>0)
                                IRT_TRACE_TO_LOG(log_file, "task %d, first/last lines [%d:%d], top/bot ptr [%d:%d]\n",
                                        task[e_irt_oicc]-1, irt_top->rm_first_line[irt_top->task[e_irt_oicc]-1],
                   irt_top->rm_last_line[irt_top->task[e_irt_oicc]-1], rm_top_ptr, irt_top->rm_bot_ptr);
                */
#endif
            // rot_mem_first_line[task%2] = YT_min;
            if (block_type == e_irt_block_rot)
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].Yi_end; // 0; //reseting every line
            else
                YT_min = irt_top->irt_desc[irt_top->task[task_type]].image_par[MIMAGE].H - 1;
        }
    }

    if (irt_mem_ctrl->last_line[irt_top->task[task_type]] < YB_int ||
        irt_mem_ctrl->lines_valid[irt_top->task[task_type]] == 0) { // && task==0 && line==0 && pixel==0) {
        // if (block_type == e_irt_block_rot)
        // IRT_TRACE("Warning - %s: task %d pixel[%d, %d:%d]: cycle %d not all lines from[%d:%d] are in rotation memory
        // [%d:%d], top/bot ptr [%d:%d], fullness %d\n", 		my_string.c_str(), irt_top->task[task_type], line,
        // pixel, pixel
        //+ adj_proc_size - 1, 		cycle, YT, YB_int, irt_mem_ctrl->first_line[irt_top->task[task_type]],
        // irt_mem_ctrl->last_line[irt_top->task[task_type]], irt_mem_ctrl->top_ptr, irt_mem_ctrl->bot_ptr,
        // irt_mem_ctrl->fullness); IRT_TRACE_TO_LOG(log_file, "Warning - IRT_OICC: cycle %d not all lines from[%d:%d]
        // of
        // task %d are in rotation memory [%d:%d]\n", cycle, YT, YB_int, task, rot_mem_first_line[task%2],
        // rot_mem_last_line[task%2]); exit(0);
#ifdef STANDALONE_ROTATOR
        // IRT_TRACE_TO_LOG(log_file, "Warning - IRT_OICC: cycle %d not all lines from[%d:%d] of task %d are in rotation
        // memory [%d:%d], top/bot ptr [%d:%d], fullness %d\n", cycle, YT, YB_int, task[e_irt_oicc],
        // irt_top->rm_first_line[irt_top->task[e_irt_oicc]], irt_top->rm_last_line[irt_top->task[e_irt_oicc]],
        // rm_top_ptr, irt_top->rm_bot_ptr, irt_top->rm_fullness);
#endif
        timeout_cntr++;
        if (timeout_cntr >
            3 * (MAX_SI_WIDTH * MIN_SI_HEIGHT) / IRT_IFIFO_OUT_WIDTH) { // H9-FIX DUE TO ROT_MEM SIZE INCREASE
            IRT_TRACE("%s: stopped on timeout at cycle %d at task %d pixel[%d, %d:%d]\n",
                      my_string.c_str(),
                      cycle,
                      irt_top->task[task_type],
                      line,
                      pixel,
                      pixel + adj_proc_size - 1);
            IRT_TRACE("pixel[%d, %d:%d]: cycle %d not all lines from[%d:%d] of task %d are in rotation memory [%d:%d], "
                      "top/bot ptr [%d:%d], fullness %d\n",
                      line,
                      pixel,
                      pixel + adj_proc_size - 1,
                      cycle,
                      YT,
                      YB_int,
                      irt_top->task[task_type],
                      irt_mem_ctrl->first_line[irt_top->task[task_type]],
                      irt_mem_ctrl->last_line[irt_top->task[task_type]],
                      irt_mem_ctrl->top_ptr,
                      irt_mem_ctrl->bot_ptr,
                      irt_mem_ctrl->fullness);
            IRT_TRACE_TO_RES(test_res,
                             "%s: stopped on timeout at cycle %d at task %d pixel[%d, %d:%d]]\n",
                             my_string.c_str(),
                             cycle,
                             irt_top->task[task_type],
                             line,
                             pixel,
                             pixel + adj_proc_size - 1);
            IRT_CLOSE_FAILED_TEST(0);
        }
        return 0;
    }

    timeout_cntr = 0;

    if (irt_mem_ctrl->last_line[irt_top->task[task_type]] < YB && pixel != 0) {
        IRT_TRACE(
            "Error - %s: for pixel[%d,%d] task %d not all lines from[%d:%d] are in %s memory [%d:%d], YB_int=%d\n",
            my_string.c_str(),
            line,
            pixel,
            irt_top->task[task_type],
            YT,
            YB,
            irt_mem_type_s[block_type],
            irt_mem_ctrl->first_line[irt_top->task[task_type]],
            irt_mem_ctrl->last_line[irt_top->task[task_type]],
            YB_int);
        IRT_TRACE_TO_RES(
            test_res,
            " failed, %s pixel[%d,%d] task %d not all lines from[%d:%d] are in %s memory [%d:%d], YB_int=%d\n",
            my_string.c_str(),
            line,
            pixel,
            irt_top->task[task_type],
            YT,
            YB,
            irt_mem_type_s[block_type],
            irt_mem_ctrl->first_line[irt_top->task[task_type]],
            irt_mem_ctrl->last_line[irt_top->task[task_type]],
            YB_int);
        IRT_CLOSE_FAILED_TEST(0);
        return 0;
    }

    if ((YT_int < irt_mem_ctrl->first_line[irt_top->task[task_type]]) &&
        irt_mem_ctrl->lines_valid[irt_top->task[task_type]]) { //!(irt_top->task[task_type] == 0 && line == 0)) {// this
                                                               //! line already out of rotation memory, not checked in
                                                               //! first entrance, YT is not valid
        IRT_TRACE("Error - %s: line Yi %d for ", my_string.c_str(), YT_int);
        for (uint8_t index = 0; index < adj_proc_size; index++) {
            IRT_TRACE("Yi_fixed[%d] = %.8f, ", index, (double)Yi_fixed[index] / pow(2, 31));
        }
        IRT_TRACE(" out of %s memory [%d:%d] at task %d line %d, pixel %d, cycle %d\n",
                  irt_mem_type_s[block_type],
                  irt_mem_ctrl->first_line[irt_top->task[task_type]],
                  irt_mem_ctrl->last_line[irt_top->task[task_type]],
                  irt_top->task[task_type],
                  line,
                  pixel,
                  cycle);

        IRT_TRACE_TO_RES(test_res,
                         " failed, %s line Yi %d out of %s memory [%d:%d] at task %d line %d, pixel %d, cycle %d\n",
                         my_string.c_str(),
                         YT_int,
                         irt_mem_type_s[block_type],
                         irt_mem_ctrl->first_line[irt_top->task[task_type]],
                         irt_mem_ctrl->last_line[irt_top->task[task_type]],
                         irt_top->task[task_type],
                         line,
                         pixel,
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (block_type == e_irt_block_rot && ofifo_full) { // no space in output FIFO
        return 0;
    }
    pixel_valid    = adj_proc_size;
    ofifo_push     = 1;
    cfifo_pop_size = adj_proc_size;
    cfifo_pop      = 1;
    // IRT_TRACE("[%d,%d] proc_size = %d\n", line, pixel, adj_proc_size);

#if 0
	IRT_TRACE("proc size %d\n", adj_proc_size);
	for (uint8_t bank_row = 0; bank_row < 8; bank_row++) {
		IRT_TRACE("bank row %d, rd_sel[0] = %d, rd_addr[0]=%d	", bank_row, rd_sel[bank_row][0], rd_addr[bank_row][0]);
		IRT_TRACE("bank row %d, rd_sel[1] = %d, rd_addr[1]=%d\n", bank_row, rd_sel[bank_row][1], rd_addr[bank_row][1]);
	}
#endif

    // if (task[e_irt_oicc]==8)
    // if (block_type == e_irt_block_mesh)
    // IRT_TRACE("%s: task %d, line %d, pixel %d valid %x at cycle %d\n", my_string.c_str(), irt_top->task[task_type],
    // line, pixel, pixel_valid, cycle);

//	IRT_TRACE_TO_LOG(log_file, "Plane %d, pixel[%3d,%3d], [yo,xo]=[%3d,%3d], YB_int %d, rot_mem_last_line %d\n", task,
// line, pixel, Yo, Xo0, YB_int, rot_mem_last_line[task%2]);
#ifdef STANDALONE_ROTATOR
    if (print_log_file)
        IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "Plane %d, pixel[%3d,%3d]\n", irt_top->task[task_type], line, pixel);
#endif

    if ((block_type == e_irt_block_rot) || ((block_type == e_irt_block_mesh) && (ofifo_full == 0))) {
        pixel += adj_proc_size;
    }

    // IRT_TRACE("IRT_OICC line %d pixel %d of task %d is finished\n", line, pixel, task);
    // processing rate statistic calculation
    if (block_type == e_irt_block_rot) {
        if (adj_proc_size < irt_top->rot_pars[irt_top->task[task_type]].min_proc_rate) {
            irt_top->rot_pars[irt_top->task[task_type]].min_proc_rate = adj_proc_size;
        }
        if (adj_proc_size > irt_top->rot_pars[irt_top->task[task_type]].max_proc_rate) {
            irt_top->rot_pars[irt_top->task[task_type]].max_proc_rate = adj_proc_size;
        }

        irt_top->rot_pars[irt_top->task[task_type]].acc_proc_rate += adj_proc_size;
        irt_top->rot_pars[irt_top->task[task_type]].rate_hist[adj_proc_size - 1]++;
        irt_top->rot_pars[irt_top->task[task_type]].cycles++;
    }

    if (pixel >= irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S) {
#ifdef STANDALONE_ROTATOR
        // IRT_TRACE("IRT_OICC line %d of task %d is finished, YT_min=%d, rotation memory range is [%d:%d]\n", line,
        // task, YT_min, irt_top->rm_first_line[irt_top->task%2], rm_last_line[task%2]);
#endif
        if ((irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1)) == 0) {
            pixel_valid = IRT_ROT_MAX_PROC_SIZE;
        } else {
            pixel_valid =
                irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].S & (IRT_ROT_MAX_PROC_SIZE - 1);
        }
        pixel_valid = adj_proc_size;

        IRT_TRACE_TO_LOG(3,
                         IRTLOG,
                         "%s : task %d line %d, pixel %d valid %x is finished at cycle %d\n",
                         my_string.c_str(),
                         irt_top->task[task_type],
                         line,
                         pixel,
                         pixel_valid,
                         cycle);

        pixel      = 0;
        line_start = 1;
        line++;
        if (line == irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].H) {
            task_done                                       = 1;
            task_end                                        = 1;
            irt_mem_ctrl->rd_done[irt_top->task[task_type]] = 1;
            if (irt_mem_ctrl->wr_done[irt_top->task[task_type]] == 0) {
                IRT_TRACE("%s task %d is done at cycle %d, waiting RMWM to complete writing\n",
                          my_string.c_str(),
                          irt_top->task[task_type],
                          cycle);
            }

            //			rm_top_ptr = (rm_top_ptr + (YT_min - irt_top->rm_first_line[irt_top->task[e_irt_oicc]])) &
            // irt_top->irt_cfg.BufH_mod; 			irt_top->rm_fullness = (irt_top->rm_fullness - (YT_min -
            // irt_top->rm_first_line[irt_top->task[e_irt_oicc]]));//%IRT_ROT_MEM_HEIGHT;
            //			irt_top->rm_first_line[irt_top->task[e_irt_oicc]] = YT_min;
            // release all lines
            if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr + (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                              irt_mem_ctrl->first_line[irt_top->task[task_type]])) &
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .BufH_mod;
                irt_mem_ctrl->fullness                             = (irt_mem_ctrl->fullness -
                                          (irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                           irt_mem_ctrl->first_line[irt_top->task[task_type]])); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
            } else {
                lines_to_release = irt_mem_ctrl->last_line[irt_top->task[task_type]] -
                                   irt_mem_ctrl->first_line[irt_top->task[task_type]] + 1;
                entries_to_release = (lines_to_release >> (uint8_t)log2(IRT_ROT_MEM_ROW_BANKS));
                irt_mem_ctrl->top_ptr =
                    (irt_mem_ctrl->top_ptr +
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL) &
                    irt_top->irt_cfg.Hb_mod[block_type];
                irt_mem_ctrl->fullness =
                    (irt_mem_ctrl->fullness -
                     entries_to_release *
                         irt_top->irt_cfg
                             .rm_cfg[block_type]
                                    [irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                             .Buf_EpL); //%IRT_ROT_MEM_HEIGHT;
                irt_mem_ctrl->first_line[irt_top->task[task_type]] = irt_mem_ctrl->last_line[irt_top->task[task_type]];
                IRT_TRACE(
                    "%s task %d line %d  H %d fullness %d entries_to_release %d Buf_EpL %d cycle %d\n",
                    my_string.c_str(),
                    irt_top->task[task_type],
                    line,
                    irt_top->irt_desc[irt_top->task[task_type]].image_par[oimage_type].H,
                    irt_mem_ctrl->fullness,
                    entries_to_release,
                    irt_top->irt_cfg
                        .rm_cfg[block_type][irt_top->irt_desc[irt_top->task[task_type]].image_par[iimage_type].buf_mode]
                        .Buf_EpL,
                    cycle);
            }
        }
    }

    // IRT_TRACE_TO_LOG(log_file, "IRT_OICC: task %d, line %d, pixel %d, pixel_valid %x\n", task, line, pixel,
    // pixel_valid);

    /*****************/
    /* IIIRC section */
    /*****************/
    // remain_pixels = irt_top->irt_desc[irt_task].image_par[OIMAGE].S - pixel;
    // if (remain_pixels > adj_proc_size) remain_pixels = adj_proc_size;

    XL = (int16_t)(IRT_top::IRT_UTILS::irt_min_coord(Xi_fixed, adj_proc_size) >> 31);
    XR = (int16_t)(IRT_top::IRT_UTILS::irt_max_coord(Xi_fixed, adj_proc_size) >> 31) + 1;
    YT = (int16_t)(IRT_top::IRT_UTILS::irt_min_coord(Yi_fixed, adj_proc_size) >> 31);
    YB = (int16_t)(IRT_top::IRT_UTILS::irt_max_coord(Yi_fixed, adj_proc_size) >> 31) + 1;

#if 0
	for (uint8_t k = 0; k < adj_proc_size; k++)
		IRT_TRACE("Coord %d = %f, ", k, (double)Yi_fixed[k] / pow(2, 31));
	IRT_TRACE("\n");
#endif

    // YT_next = YT;

    if (ofifo_push) {
        // IRT_TRACE("%s: task %d, pixel[%d, %d:%d] [XL:XR] [%d:%d]\n", my_string.c_str(), irt_task, Yo , Xo0, Xo0 +
        // remain_pixels - 1, XL, XR); IRT_TRACE("%s: pixel[%d, %d:%d] requires [%d,%d] lines of input image\n",
        // my_string.c_str(), Yo, Xo0, Xo0 + remain_pixels - 1, YT, YB); IRT_TRACE("%s: pixel[%d, %d:%d] requires
        // [%f,%f] lines of input image\n", my_string.c_str(), Yo, Xo0, Xo0 + remain_pixels - 1, Yi_fixed[0]/pow(2.0,
        // 31), (double)Yi_fixed[remain_pixels - 1] /pow(2.0, 31)); exit (0);
    }

    YB = irt_top->YB_adjustment(irt_top->irt_desc[irt_task], block_type, YB);

    if (block_type == e_irt_block_rot) {
#ifdef IRT_USE_FLIP_FOR_MINUS1
        if (irt_top->irt_desc[irt_task].read_vflip == 1)
            YT--;
        if (irt_top->irt_desc[irt_task].read_hflip == 1)
            XL--;
        if (irt_top->irt_desc[irt_task].read_hflip == 1)
            XR--;
#endif
    } else {
        if (irt_top->irt_desc[irt_task].mesh_sparse_h == 0)
            XR--;
    }

    // if (YT < YT_min) IRT_TRACE("%s: updating YT_min to %d, task %d, line %d, pixel %d\n", my_string.c_str(), YT,
    // irt_top->task[task_type], line, pixel);
    YT_min =
        YT < YT_min ? YT : YT_min; // tracking minimum Y value of input image required for output image line calculation

#if 0
	if ((YB - YT > row_banks - 1) && error_flag == 0 && irt_top->irt_desc[irt_task].rate_mode != e_irt_rate_adaptive_2x2) {
		IRT_TRACE("%s error: pixel[%d, %d:%d] requires more than %d lines of input image [%d,%d]\n", my_string.c_str(),
			cfifo_data_out[0].line, cfifo_data_out[0].pixel, pixel + adj_proc_size - 1, row_banks, YT, YB);
		error_flag = 1;
		IRT_TRACE_TO_RES(test_res, " failed, %s pixel[%d, %d:%d] requires more than 8 lines of input image [%d,%d]\n", my_string.c_str(),
			cfifo_data_out[0].line, cfifo_data_out[0].pixel, pixel + adj_proc_size - 1, YT, YB);
		IRT_CLOSE_FAILED_TEST(0);
	}
#endif

    if (irt_top->irt_desc[irt_task].rate_mode != e_irt_rate_adaptive_2x2)
        YB = IRT_UTILS::irt_min_int16(YB, YT + row_banks - 1);

    if (ofifo_push) {
#ifdef STANDALONE_ROTATOR
        if (print_log_file)
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             "IRT_IIIRC: task=%d, pixel=%d, line=%d, Xi0=%lld, Yi0=%lld, Xi7[%d]=%lld, Yi7=%lld\n",
                             irt_task,
                             pixel,
                             line,
                             Xi_fixed[0],
                             Yi_fixed[0],
                             adj_proc_size - 1,
                             Xi_fixed[adj_proc_size - 1],
                             Yi_fixed[adj_proc_size - 1]);
#endif
        // IRT_TRACE_TO_LOG(log_file, "%d, %d\n", YT, YB);
        // IRT_TRACE_TO_LOG(log_file, "%f, %f\n",
        // Yi0_fixed/pow(2.0,irt_top->irt_desc[irt_task].TOTAL_PREC),Yi7_fixed/pow(2.0,irt_top->irt_desc[irt_task].TOTAL_PREC));
        // //IRT_TRACE_TO_LOG(log_file, "IRT_IIIRC: XL=%d, XR=%d, YT=%d, YB=%d\n", XL, XR, YT, YB);
    }

    return 0;
}

template <uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, Eirt_block_type block_type>
void IRT_top::IRT_RMRM<ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>::run(
    int16_t  XL,
    int16_t  XR,
    int16_t  YT,
    int16_t  YB,
    uint8_t  bank_row[ROW_BANKS],
    bool     rd_sel[ROW_BANKS][COL_BANKS],
    int16_t  rd_addr[ROW_BANKS][COL_BANKS],
    bool     rd_mode[ROW_BANKS],
    uint8_t  rd_shift[ROW_BANKS],
    bool     msb_lsb_sel[ROW_BANKS][COL_BANKS],
    uint8_t& YT_bank_row,
    bool     bg_flag[ROW_BANKS][COL_BANKS][BANK_WIDTH],
    uint8_t  irt_task,
    bool     rd_en,
    int16_t  Xo,
    int16_t  Yo,
    uint8_t  caller)
{

    uint8_t iimage_type   = block_type == e_irt_block_rot ? IIMAGE : MIMAGE;
    uint8_t oimage_type   = block_type == e_irt_block_rot ? OIMAGE : MIMAGE;
    uint8_t mem_pxl_shift = IRT_ROT_MEM_PXL_SHIFT; // block_type == e_irt_block_rot ? IRT_ROT_MEM_PXL_SHIFT :
                                                   // IRT_ROT_MEM_PXL_SHIFT; //IRT_MESH_MEM_PXL_SHIFT;

    // meta_data_struct *rmeta_mem = irt_top->IRT_RMEM_block->meta_mem;
    // meta_data_struct mmeta_mem[IRT_MESH_MEM_ROW_BANKS][IRT_META_MEM_BANK_HEIGHT] =
    // &irt_top->IRT_MMEM_block->meta_mem;
    std::string my_string = getInstanceName();

    if (block_type == e_irt_block_mesh) {
        if (irt_top->irt_desc[irt_task].irt_mode != e_irt_mesh) { // RMRM works as MMRM and task is not mesh, droppped
            return;
        }
    }
    //   //---------------------------------------
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[task_type]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[task_type].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //   //---------------------------------------

    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[block_type];

    int16_t line =
        (Yo + block_type == e_irt_block_rot ? irt_top->irt_desc[irt_task].image_par[oimage_type].Yc : 0) >> 1;
    int16_t pixel =
        (Xo + block_type == e_irt_block_rot ? irt_top->irt_desc[irt_task].image_par[oimage_type].Xc : 0) >> 1;

    int16_t addr;
    int16_t Yi_start;
    Yi_start = irt_top->yi_start_calc(irt_top->irt_desc[irt_task], block_type);

    int16_t YT1;
    if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
        YT1 = (irt_mem_ctrl->start_line[irt_task] + YT - Yi_start) &
              irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_task].image_par[iimage_type].buf_mode].BufH_mod;
    } else {
        YT1 = (YT - Yi_start);
    }
    YT_bank_row = YT1 % ROW_BANKS;
    //	int YT_line_sel_in_bank_row = YT1>>3;
    //	int YT_Xi_start = Xi_start_mem[YT_bank_row][YT_line_sel_in_bank_row];
    //	int YT_XL8 = ((XL - YT_Xi_start)>>3)<<3;

    //	int XL8 = (XL>>3)<<3;
    for (uint8_t row_bank = 0; row_bank < ROW_BANKS; row_bank++) {
        rd_sel[row_bank][0]      = 0;
        rd_sel[row_bank][1]      = 0;
        msb_lsb_sel[row_bank][0] = 0;
        msb_lsb_sel[row_bank][1] = 0;
        for (uint16_t i = 0; i < BANK_WIDTH; i++) {
            bg_flag[row_bank][0][i] = 0;
            bg_flag[row_bank][1][i] = 0;
        }
    }

    //	static flag=0;
    // if (rd_en) IRT_TRACE("IRT_RMRM: [YT:YB][XL:XR]=[%d:%d][%d:%d]\n", YT, YB, XL, XR);

    for (int16_t Y = YT, idx = 0; Y <= YB; Y++, idx++) {

        int16_t Y1;
        if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
            Y1 = (irt_mem_ctrl->start_line[irt_task] + Y - Yi_start) &
                 irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_task].image_par[iimage_type].buf_mode]
                     .BufH_mod;
        } else {
            Y1 = (Y - Yi_start);
        }

        if (Y1 < 0) {
            IRT_TRACE("%s: Y1 %d is negative at MEM_START_LINE %d Y %d Yi_start %d [YT:YB]=[%d:%d]\n",
                      my_string.c_str(),
                      Y1,
                      irt_mem_ctrl->start_line[irt_task],
                      Y,
                      Yi_start,
                      YT,
                      YB);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s Y1 %d is negative at RM_START_LINE %d Y %d Yi_start %d [YT:YB] = [%d:%d]\n",
                             my_string.c_str(),
                             Y1,
                             irt_mem_ctrl->start_line[irt_task],
                             Y,
                             Yi_start,
                             YT,
                             YB);
            IRT_CLOSE_FAILED_TEST(0);
        }

        bank_row[idx]             = Y1 % ROW_BANKS;
        uint16_t line_in_bank_row = Y1 >> (uint8_t)log2(ROW_BANKS);
        int16_t  rm_meta_rd_addr;
        if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
            rm_meta_rd_addr = line_in_bank_row;
        } else {
            rm_meta_rd_addr =
                (irt_mem_ctrl->start_line[irt_task] +
                 line_in_bank_row *
                     irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_task].image_par[iimage_type].buf_mode]
                         .Buf_EpL) %
                IRT_META_MEM_BANK_HEIGHT;
        }

        if (bank_row[idx] < 0 || bank_row[idx] > ROW_BANKS - 1) {
            IRT_TRACE("%s: bank row %d at RM_START_LINE %d Y1 %d Yi_start %d\n",
                      my_string.c_str(),
                      bank_row[idx],
                      irt_mem_ctrl->start_line[irt_task],
                      Y,
                      Yi_start);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s bank row %d at RM_START_LINE %d Y1 %d Yi_start %d\n",
                             my_string.c_str(),
                             bank_row[idx],
                             irt_mem_ctrl->start_line[irt_task],
                             Y,
                             Yi_start);
            IRT_CLOSE_FAILED_TEST(0);
        }

        if ((irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) &&
            (line_in_bank_row > irt_top->irt_cfg.Hb[block_type] - 1)) { // IRT_ROT_MEM_BANK_HEIGHT/8*2-1) {
            IRT_TRACE("%s: line_in_bank_row %d is wrong\n", my_string.c_str(), line_in_bank_row);
            IRT_TRACE_TO_RES(
                test_res, " failed, %s line_in_bank_row %d is wrong\n", my_string.c_str(), line_in_bank_row);
            IRT_CLOSE_FAILED_TEST(0);
        }

        int16_t Xi_start = block_type == e_irt_block_rot
                               ? irt_top->IRT_RMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].Xi_start
                               : irt_top->IRT_MMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].Xi_start;
        int16_t Xi_start_calc =
            block_type == e_irt_block_rot
                ? IRT_top::xi_start_calc(
                      irt_top->irt_desc[irt_task], Y, e_irt_xi_start_calc_caller_rmrm + caller, irt_task)
                : 0;

        if (rd_en && Xi_start != Xi_start_calc && block_type == e_irt_block_rot) {
            IRT_TRACE("%s: calculated Xi_start %d is not equal to Xi_start %d from meta_mem[%d][%d] at task %d Yo=%d "
                      "Xo=%d [YT/YB]=[%d/%d] at cycle %d\n",
                      my_string.c_str(),
                      Xi_start_calc,
                      Xi_start,
                      bank_row[idx],
                      rm_meta_rd_addr,
                      irt_task,
                      line,
                      pixel,
                      YT,
                      YB,
                      cycle);
            IRT_TRACE_TO_RES(test_res,
                             "%s calculated Xi_start %d is not equal to Xi_start %d from meta_mem[%d][%d] at task %d "
                             "Yo=%d Xo=%d [YT/YB]=[%d/%d] at cycle %d\n",
                             my_string.c_str(),
                             Xi_start_calc,
                             Xi_start,
                             bank_row[idx],
                             rm_meta_rd_addr,
                             irt_task,
                             line,
                             pixel,
                             YT,
                             YB,
                             cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }
        if (rd_en & 0)
            IRT_TRACE("%s: calculated Xi_start %d is equal to Xi_start %d from meta_mem[%d][%d] at task %d Yo=%d Xo=%d "
                      "[YT/YB]=[%d/%d] at cycle %d\n",
                      my_string.c_str(),
                      Xi_start_calc,
                      Xi_start,
                      bank_row[idx],
                      rm_meta_rd_addr,
                      irt_task,
                      line,
                      pixel,
                      YT,
                      YB,
                      cycle);

        if (block_type == e_irt_block_rot)
            Xi_start = Xi_start_calc;
        int16_t X1 = XL - Xi_start;

        int16_t meta_line = block_type == e_irt_block_rot
                                ? irt_top->IRT_RMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].line
                                : irt_top->IRT_MMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].line;
        if (rd_en && Y != meta_line) {
            IRT_TRACE("%s: Y %d is not equal to line %d from meta_mem[%d][%d] at task %d Yo=%d Xo=%d [YT/YB]=[%d/%d] "
                      "at cycle %d\n",
                      my_string.c_str(),
                      Y,
                      meta_line,
                      bank_row[idx],
                      rm_meta_rd_addr,
                      irt_task,
                      line,
                      pixel,
                      YT,
                      YB,
                      cycle);
            IRT_TRACE_TO_RES(test_res,
                             "failed, %s Y %d is not equal to line %d from meta_mem[%d][%d] at task %d Yo=%d "
                             "Xo=%d[YT/YB]=[%d/%d] at cycle %d\n",
                             my_string.c_str(),
                             Y,
                             meta_line,
                             bank_row[idx],
                             rm_meta_rd_addr,
                             irt_task,
                             line,
                             pixel,
                             YT,
                             YB,
                             cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }

        uint8_t meta_task = block_type == e_irt_block_rot
                                ? irt_top->IRT_RMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].task
                                : irt_top->IRT_MMEM_block->meta_mem[bank_row[idx]][rm_meta_rd_addr].task;
        if (rd_en && (irt_task != meta_task)) {
            IRT_TRACE("%s: task %d is not equal to task %d from meta_mem[%d][%d] at line [Yo=%d, Yi=%d] at cycle %d\n",
                      my_string.c_str(),
                      irt_task,
                      meta_task,
                      bank_row[idx],
                      rm_meta_rd_addr,
                      line,
                      Y,
                      cycle);
            for (uint8_t i = 0; i < MAX_TASKS; i++)
                IRT_TRACE("%d ", irt_mem_ctrl->last_line[i]);
            IRT_TRACE("\n");
            for (uint8_t b = 0; b < ROW_BANKS; b++)
                for (uint8_t a = 0; a < 16; a++)
                    IRT_TRACE("%s: bank %d, addr %d, task %d, meta_mem.line %d, meta_mem.Xi_start %d at cycle %d\n",
                              my_string.c_str(),
                              b,
                              a,
                              block_type == e_irt_block_rot ? irt_top->IRT_RMEM_block->meta_mem[b][a].task
                                                            : irt_top->IRT_MMEM_block->meta_mem[b][a].task,
                              block_type == e_irt_block_rot ? irt_top->IRT_RMEM_block->meta_mem[b][a].line
                                                            : irt_top->IRT_MMEM_block->meta_mem[b][a].line,
                              block_type == e_irt_block_rot ? irt_top->IRT_RMEM_block->meta_mem[b][a].Xi_start
                                                            : irt_top->IRT_MMEM_block->meta_mem[b][a].Xi_start,
                              cycle);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s task %d is not equal to task %d from meta_mem[%d][%d] at line [Yo=%d, Yi=%d] "
                             "at cycle %d\n",
                             my_string.c_str(),
                             irt_task,
                             meta_task,
                             bank_row[idx],
                             rm_meta_rd_addr,
                             line,
                             Y,
                             cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }
#if 0
        if (rd_en)
            IRT_TRACE("pixel[%d,%d]: X1 %d is negative at Y %d Xi_start %d [XL:XR]=[%d:%d] at task %d at "
                      "irt_top->meta_mem[%d][%d]\n",
                      line,
                      pixel,
                      X1,
                      Y,
                      Xi_start,
                      XL,
                      XR,
                      irt_task,
                      bank_row[idx],
                      line_in_bank_row);
#endif

        if (rd_en && X1 < 0) {
            IRT_TRACE("%s: pixel[%d,%d]: X1 %d is negative at Y %d Xi_start %d [YT:YB]=[%d:%d] [XL:XR]=[%d:%d] at task "
                      "%d at irt_top->meta_mem[%d][%d] at cycle %d\n",
                      my_string.c_str(),
                      line,
                      pixel,
                      X1,
                      Y,
                      Xi_start,
                      YT,
                      YB,
                      XL,
                      XR,
                      irt_task,
                      bank_row[idx],
                      line_in_bank_row,
                      cycle);
#if 0
			for (int b=0; b<8; b++)
				for (int a=0; a<16; a++)
					IRT_TRACE("bank %d, addr %d, task %d, line %d, Xi_start %d\n", b, a, irt_top->meta_mem[b][a].task, irt_top->meta_mem[b][a].line, irt_top->meta_mem[b][a].Xi_start);
#endif
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s pixel[%d,%d]: X1 %d is negative at Y %d Xi_start %d [YT:YB]=[%d:%d] "
                             "[XL:XR]=[%d:%d] at task %d at irt_top->meta_mem[%d][%d] at cycle %d\n",
                             my_string.c_str(),
                             line,
                             pixel,
                             X1,
                             Y,
                             Xi_start,
                             YT,
                             YB,
                             XL,
                             XR,
                             irt_task,
                             bank_row[idx],
                             line_in_bank_row,
                             cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }

        uint16_t addr_offset =
            X1 >> (mem_pxl_shift -
                   (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps &
                    1)); // 16/32 is number of 16/32 pixels entries in bank rows (from to bank columns) for stripe 256
        uint16_t BufW_entries =
            irt_top->irt_cfg.rm_cfg[block_type][irt_top->irt_desc[irt_task].image_par[iimage_type].buf_mode]
                .Buf_EpL; // << irt_top->irt_desc[irt_task].image_par[IIMAGE].Ps;
        if (irt_top->irt_cfg.buf_format[block_type] == e_irt_buf_format_static) {
            addr = (line_in_bank_row * BufW_entries) + addr_offset;
        } else {
            addr = (irt_mem_ctrl->start_line[irt_task] + line_in_bank_row * BufW_entries + addr_offset) &
                   irt_top->irt_cfg.Hb_mod[block_type];
        }

#if 0
		IRT_TRACE("pixel[%d][%d]: Y1 %d, line_in_bank_row %d, XL %d, Xi_start %d, X1 %d, BufW_entries %d, addr_offset %d addr %d\n", 
			(Yo + irt_top->irt_desc[irt_task].image_par[OIMAGE].Yc) / 2, (Xo + irt_top->irt_desc[irt_task].image_par[OIMAGE].Xc) / 2,
			Y1, line_in_bank_row, XL, Xi_start, X1, BufW_entries, addr_offset, addr);
#endif

        if (rd_en && (addr < 0 || addr >= irt_top->irt_cfg.Hb[block_type])) { // || (Yo+image_pars[OIMAGE].Yc)==265)) {
            IRT_TRACE("%s: task %d, pixel[%d,%d], Y=%d\n", my_string.c_str(), irt_task, line, pixel, Y);
            IRT_TRACE("%s: addr %d invalid: bank_row=%d, line_in_bank_row=%d BufW_entries=%d, X1=%d, XL=%d, "
                      "Xi_start=%d at cycle %d\n",
                      my_string.c_str(),
                      addr,
                      bank_row[idx],
                      line_in_bank_row,
                      BufW_entries,
                      X1,
                      XL,
                      Xi_start,
                      cycle);
#if 0
			for (int b=0; b<8; b++)
				for (int a=0; a<32; a++)
					IRT_TRACE("bank %d, addr %d, task %d, line %d, Xi_start %d\n", b, a, irt_top->meta_mem[b][a].task, irt_top->meta_mem[b][a].line, irt_top->meta_mem[b][a].Xi_start);
#endif
            if (addr < 0 || addr >= irt_top->irt_cfg.Hb[block_type]) {
                IRT_TRACE("%s: failed, address is out of rotation memory at cycle %d\n", my_string.c_str(), cycle);
                IRT_TRACE_TO_RES(
                    test_res, " failed, %s address is out of rotation memory at cycle %d\n", my_string.c_str(), cycle);
                IRT_CLOSE_FAILED_TEST(0);
            }
        }

        if (rd_en &&
            (X1 >> (mem_pxl_shift - (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps & 1))) >= BufW_entries) {
            IRT_TRACE("%s: (X1>>%d)=%d >= BufW_entries=%d and cause to read next line entry of bank row at cycle %d\n",
                      my_string.c_str(),
                      mem_pxl_shift - (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps & 1),
                      X1 >> (mem_pxl_shift - (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps & 1)),
                      BufW_entries,
                      cycle);
            IRT_TRACE("%s: task %d, pixel[%d,%d], Y=%d\n", my_string.c_str(), irt_task, line, pixel, Y);
            IRT_TRACE(
                "%s: addr %d invalid: bank_row=%d, line_in_bank_row=%d BufW_entries=%d, X1=%d, XL=%d, Xi_start=%d\n",
                my_string.c_str(),
                addr,
                bank_row[idx],
                line_in_bank_row,
                BufW_entries,
                X1,
                XL,
                Xi_start);
            IRT_TRACE_TO_RES(
                test_res, " failed, %s rotation memory address is invalid at cycle %d\n", my_string.c_str(), cycle);
            IRT_CLOSE_FAILED_TEST(0);
        }
#if 0
		if (rd_en)
			IRT_TRACE_TO_LOG(log_file, "IRT_RMRM: Y%d: Xi_start=%d, X1=%d, shift %d, rd_mode %d\n", Y, Xi_start, X1, rd_shift[bank_row[idx]], rd_mode[bank_row]);
#endif

        rd_shift[bank_row[idx]] =
            X1 & ((BANK_WIDTH >> (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps & 1)) - 1);
        rd_mode[bank_row[idx]] =
            (X1 >> (mem_pxl_shift - 1 - (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps & 1))) & 1;

        uint8_t X1_43 = (X1 >> (mem_pxl_shift - 2)) & 3;

        uint8_t Ps1;
        if (block_type == e_irt_block_rot) {
            Ps1 = irt_top->irt_cfg.buf_1b_mode[block_type] == 1 ||
                  irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 1;
        } else {
            Ps1 = irt_top->irt_cfg.buf_1b_mode[block_type] == 1 ||
                  irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 3;
        }
        rd_shift[bank_row[idx]] = X1 & ((IRT_ROT_MEM_BANK_WIDTH >> Ps1) - 1);
        rd_mode[bank_row[idx]]  = (X1 >> (mem_pxl_shift - 1 - Ps1)) & 1;

        switch (X1_43) { //[4:3]
            case 0:
                msb_lsb_sel[bank_row[idx]][0] = 0;
                msb_lsb_sel[bank_row[idx]][1] = 0;
                break; // LSB LSB
            case 1:
                msb_lsb_sel[bank_row[idx]][0] = 1;
                msb_lsb_sel[bank_row[idx]][1] = 0;
                break; // LSB MSB
            case 2:
                msb_lsb_sel[bank_row[idx]][0] = 1;
                msb_lsb_sel[bank_row[idx]][1] = 1;
                break; // MSB MSB
            case 3:
                msb_lsb_sel[bank_row[idx]][0] = 0;
                msb_lsb_sel[bank_row[idx]][1] = 1;
                break; // MSB LSB
        }

        if (((block_type == e_irt_block_rot) && ((irt_top->irt_cfg.buf_1b_mode[block_type] == 0) ||
                                                 (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 1))) ||
            ((block_type == e_irt_block_mesh) && ((irt_top->irt_cfg.buf_1b_mode[block_type] == 0) ||
                                                  (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 3)))) {
            if (rd_mode[bank_row[idx]] == 0) {
                rd_addr[bank_row[idx]][0] = addr;
                rd_addr[bank_row[idx]][1] = addr;
            } else {
                rd_addr[bank_row[idx]][1] = addr;
                rd_addr[bank_row[idx]][0] = (addr + 1) % irt_top->irt_cfg.Hb[block_type];
            }
        } else {
            if (X1_43 != 3) {
                rd_addr[bank_row[idx]][0] = addr;
                rd_addr[bank_row[idx]][1] = addr;
            } else {
                rd_addr[bank_row[idx]][1] = addr;
                rd_addr[bank_row[idx]][0] = (addr + 1) % irt_top->irt_cfg.Hb[block_type];
            }
        }

        bool XL_col = ((XL - Xi_start) >> (mem_pxl_shift - 1 - Ps1)) & 1;
        bool XR_col = ((XR - Xi_start) >> (mem_pxl_shift - 1 - Ps1)) & 1;
        if (XL_col != XR_col) { // both columns will be read
            // rd_sel[bank_row[idx]][0] = 1;
            // rd_sel[bank_row[idx]][1] = 1;
        } else {
            // rd_sel[bank_row[idx]][XL_col] = 1;
        }

        // rd_sel[bank_row[idx]][0]=1;
        // rd_sel[bank_row[idx]][1]=1;

#if 1
        if (rd_en) {
            if (block_type == e_irt_block_mesh) {
                // IRT_TRACE("%s: Y %d, line %d, task %d, Y1 %d, X1 %d\n", my_string.c_str(), Y, meta_line, meta_task,
                // Y1, X1); IRT_TRACE("%s: task %d, [YT:YB]=[%d:%d], [XL:XR]=[%d:%d], bank_row = %d line_sel = %d\n",
                // my_string.c_str(), irt_task, YT, YB, XL, XR, bank_row, line_in_bank_row); IRT_TRACE("%s: rd_mode %d,
                // rd_shift %d, rd_addr[0/1] = [%d/%d]\n", my_string.c_str(), rd_mode[bank_row], rd_shift[bank_row],
                // rd_addr[bank_row][0], rd_addr[bank_row][1]);
            }

#ifdef STANDALONE_ROTATOR
            if (print_log_file) {
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "%s: Y %d, Xi_start %d(%d)(%d), Y1 %d, X1 %d\n",
                                 my_string.c_str(),
                                 Y,
                                 Xi_start,
                                 meta_line,
                                 meta_task,
                                 Y1,
                                 X1);
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "%s: task %d, [YT:YB]=[%d:%d], Y=%d, Y1=%d, bank_row = %d line_sel = %d\n",
                                 my_string.c_str(),
                                 irt_task,
                                 YT,
                                 YB,
                                 Y,
                                 Y1,
                                 bank_row[idx],
                                 line_in_bank_row);
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "%s: rd_mode %d, rd_shift %d, rd_addr[0/1] = [%d/%d]\n",
                                 my_string.c_str(),
                                 rd_mode[bank_row[idx]],
                                 rd_shift[bank_row[idx]],
                                 rd_addr[bank_row[idx]][0],
                                 rd_addr[bank_row[idx]][1]);
            }
#endif
        }
#endif

        // pixels are out image boundaries
        for (uint8_t j = 0; j < BANK_WIDTH; j++) {
            if ((Y < 0 || Y >= (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].H) &&
                irt_top->irt_desc[irt_task].bg_mode == e_irt_bg_prog_value) { // vertical boundaries ???
#if 1
#ifdef STANDALONE_ROTATOR
                if (rd_en)
                    if (print_log_file)
                        IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                         log_file,
                                         "%s: BG GEN: Y %d, j %d out of vert boundaries\n",
                                         my_string.c_str(),
                                         Y,
                                         j);
#endif
#endif
                bg_flag[bank_row[idx]][0][j] = 1;
                bg_flag[bank_row[idx]][1][j] = 1;
            } else { // line is inside of the image
                int16_t Xj0 = Xi_start + (X1 & 0x3F8) +
                              j; // 0x3F8 = (8-pixels group number)*8, Xi_start + (X1 & 0x3F8) is 1st pixel at bank
                int16_t Xj1 = Xi_start + (X1 & 0x3F8) + j + BANK_WIDTH;
                if (((block_type == e_irt_block_rot) &&
                     (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 0)) || // pixel is 1B (input)
                    ((block_type == e_irt_block_mesh) &&
                     (irt_top->irt_desc[irt_task].image_par[iimage_type].Ps == 2))) { // pixel is 4B (mesh)
                    if (irt_top->irt_cfg.buf_1b_mode[block_type] == 0) { // w/o reshuffle storage
                        Xj0 =
                            Xi_start + (X1 & ~(BANK_WIDTH - 1)) +
                            j; // X1&0xFFF0 = (16-pixels group number)*16, Xi_start + (X1 & 0xFFF0) is 1st pixel at bank
                        Xj1 = Xi_start + (X1 & ~(BANK_WIDTH - 1)) + j + BANK_WIDTH; // 16 pixels difference between
                                                                                    // banks
                    } else { // w/ reshuffle storage
                        // X1&0xFFF8 = (8-pixels group number)*8, Xi_start + (X1 & 0xFFF7) is 1st pixel at bank
                        // j&1 is are pixels from upper bytes that are +16 relative to lower bytes
                        // j>>1 - pixel increase only once per 2 bytes
                        // Xj0 = Xi_start + (X1 & ~(mem_bank_width / 2 - 1)) + 16 * (j & 1) + (j >> 1);
                        // Xj0 = Xi_start + (X1 & ~(mem_bank_width / 2 - 1)) + 16 * (j & 1) + (j >> 1) + mem_bank_width
                        // / 2; //8 pixels difference between banks
                        if (block_type == e_irt_block_rot) { // 1B granularity
                            Xj0 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) +
                                  (j >> 1); // X1&0xFFF0 = (16-pixels group number)*16, Xi_start + (X1 & 0xFFF0) is 1st
                                            // pixel at bank
                            Xj1 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) + (j >> 1) +
                                  BANK_WIDTH / 2; // 16 pixels difference between banks
                        } else { // 4B granularity
                            Xj0 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) + (j >> 3) * 4 + (j & 3);
                            Xj1 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) + (j >> 3) * 4 + (j & 3) + BANK_WIDTH / 2;
                        }
                    }
                } else { // pixel is 2B (input) or 8B (mesh) w/o reshuffle storage
                    Xj0 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) +
                          j; // X1&0xFFF8 = (8-pixels group number)*8, Xi_start + (X1 & 0xFFF7) is 1st pixel at bank
                    Xj1 = Xi_start + (X1 & ~(BANK_WIDTH / 2 - 1)) + j +
                          BANK_WIDTH / 2; // 8 pixels difference between banks
                }
                if (block_type == e_irt_block_mesh) { // Xj for mesh pixel is x4 of Xj for input pixel: 1B input/4B mesh
                                                      // or 2B input/4B mesh
                    Xj0 >>= 2;
                    Xj1 >>= 2;
                }
#if 1
#ifdef STANDALONE_ROTATOR
                if (rd_en && (Xj0 < 0 || Xj0 >= (int16_t)irt_top->irt_desc[irt_task].image_par[iimage_type].W))
                    if (print_log_file)
                        IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                         log_file,
                                         "%s: BG GEN: Y %d, j %d out of horizontal boundaries, Xi_start %d, X1 %d\n",
                                         my_string.c_str(),
                                         Y,
                                         j,
                                         Xi_start,
                                         X1);
#endif
#endif
                if (rd_mode[bank_row[idx]] == 0) {
                    bg_flag[bank_row[idx]][0][j] = (Xj0 < (0 + irt_top->irt_desc[irt_task].read_hflip * 0) ||
                                                    Xj0 >= ((int)irt_top->irt_desc[irt_task].image_par[iimage_type].W -
                                                            irt_top->irt_desc[irt_task].read_hflip * 0))
                                                       ? 1
                                                       : 0;
                    bg_flag[bank_row[idx]][1][j] = (Xj1 < (0 + irt_top->irt_desc[irt_task].read_hflip * 0) ||
                                                    Xj1 >= ((int)irt_top->irt_desc[irt_task].image_par[iimage_type].W -
                                                            irt_top->irt_desc[irt_task].read_hflip * 0))
                                                       ? 1
                                                       : 0;
                } else {
                    bg_flag[bank_row[idx]][1][j] = (Xj0 < (0 + irt_top->irt_desc[irt_task].read_hflip * 0) ||
                                                    Xj0 >= ((int)irt_top->irt_desc[irt_task].image_par[iimage_type].W -
                                                            irt_top->irt_desc[irt_task].read_hflip * 0))
                                                       ? 1
                                                       : 0;
                    bg_flag[bank_row[idx]][0][j] = (Xj1 < (0 + irt_top->irt_desc[irt_task].read_hflip * 0) ||
                                                    Xj1 >= ((int)irt_top->irt_desc[irt_task].image_par[iimage_type].W -
                                                            irt_top->irt_desc[irt_task].read_hflip * 0))
                                                       ? 1
                                                       : 0;
                }
                // read rotation memory only when pixels are not out of bound and read only relevant column banks
                if (bg_flag[bank_row[idx]][0][j] == 0 && (XL_col == 0 || XR_col == 0)) {
                    rd_sel[bank_row[idx]][0] = 1;
                }
                if (bg_flag[bank_row[idx]][1][j] == 0 && (XL_col == 1 || XR_col == 1)) {
                    rd_sel[bank_row[idx]][1] = 1;
                }
            }
        }
    }
    // if (rd_en) exit(0);
}

template <class BUS_IN,
          class BUS_OUT,
          uint16_t        ROW_BANKS,
          uint16_t        COL_BANKS,
          uint16_t        BANK_WIDTH,
          Eirt_block_type block_type>
void IRT_top::IRT_8x16_IWC<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>::run(
    BUS_IN        in_pix[ROW_BANKS][COL_BANKS],
    bool          in_pix_flag[ROW_BANKS][COL_BANKS][BANK_WIDTH],
    bool          rd_mode[ROW_BANKS],
    uint8_t       rd_shift[ROW_BANKS],
    bool          msb_lsb_sel[ROW_BANKS][COL_BANKS],
    int16_t       XL,
    int16_t       YT,
    uint8_t       YT_bank_row,
    BUS_OUT       out_pix[ROW_BANKS],
    bus16f_struct out_pix_flag[ROW_BANKS],
    bool          ofifo_push,
    int16_t       Yo,
    int16_t       Xo0,
    uint8_t       oicc_task)
{

    std::string my_string = getInstanceName();

    uint8_t concat_line[ROW_BANKS][COL_BANKS * BANK_WIDTH], reorder_line[ROW_BANKS][COL_BANKS * BANK_WIDTH];
    bool    concat_flag[ROW_BANKS][COL_BANKS * BANK_WIDTH], reorder_flag[ROW_BANKS][COL_BANKS * BANK_WIDTH];
    // used for debug - input comes from ext memory

#ifdef STANDALONE_ROTATOR
    if ((print_log_file && block_type == e_irt_block_mesh && ofifo_push)) {
        IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%s Y=%d, XL=%d\n", my_string.c_str(), YT, XL);
        for (uint8_t i = 0; i < ROW_BANKS; i++) {
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "in_pix: YT_bank_row %d bank_row %d [", YT_bank_row, i);
            for (uint8_t j = 0; j < BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)((unsigned char)in_pix[i][0].pix[j]));
            for (uint8_t j = 0; j < BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)((unsigned char)in_pix[i][1].pix[j]));
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "][");
            for (uint8_t j = 0; j < BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)(in_pix_flag[i][0][j]));
            for (uint8_t j = 0; j < BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)(in_pix_flag[i][1][j]));
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "]\n");
        }
    }
#endif

    // 16 pixels interpolation lines creation
    for (uint8_t i = 0; i < ROW_BANKS; i++) {
        for (uint8_t j = 0; j < BANK_WIDTH; j++) {
            if ((irt_top->irt_cfg.buf_1b_mode[block_type] == 0) || // w/o reshuffle storage
                ((block_type == e_irt_block_rot) &&
                 (irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps == 1)) || // pixel is 2B
                ((block_type == e_irt_block_mesh) &&
                 (irt_top->irt_desc[oicc_task].image_par[MIMAGE].Ps == 3))) { // pixel is 8B
                if (rd_mode[i] == 0) { // we read both bank-columns from  same address
                    concat_line[i][j]              = in_pix[i][0].pix[j];
                    concat_line[i][BANK_WIDTH + j] = in_pix[i][1].pix[j];
                    concat_flag[i][j]              = in_pix_flag[i][0][j];
                    concat_flag[i][BANK_WIDTH + j] = in_pix_flag[i][1][j];
                } else { // we read bank-column 1 from bank address A and bank-column 0 from addr A+1
                    concat_line[i][j]              = in_pix[i][1].pix[j];
                    concat_line[i][BANK_WIDTH + j] = in_pix[i][0].pix[j];
                    concat_flag[i][j]              = in_pix_flag[i][1][j];
                    concat_flag[i][BANK_WIDTH + j] = in_pix_flag[i][0][j];
                }
            } else { // pixel is 1B (rot) or 4B (mesh) w/ reshuffle storage
                if (block_type == e_irt_block_rot) { // pixel is 1B (rot) w/ reshuffle storage
                    if (rd_mode[i] == 0) { // we read both bank-columns from  same address
                        if (j < BANK_WIDTH / 2) { // 1st 8 locations filled from bank 0, LSB - using even bytes, MSB -
                                                  // using odd bytes
                            concat_line[i][j] = in_pix[i][0].pix[j * 2 + msb_lsb_sel[i][0]];
                            concat_flag[i][j] = in_pix_flag[i][0][j * 2 + msb_lsb_sel[i][0]];
                        } else { //{ //next 8 locations filled from bank 1,
                            concat_line[i][j] = in_pix[i][1].pix[(j - BANK_WIDTH / 2) * 2 + msb_lsb_sel[i][1]];
                            concat_flag[i][j] = in_pix_flag[i][1][(j - BANK_WIDTH / 2) * 2 + msb_lsb_sel[i][1]];
                        }
                    } else { // we read bank-column 1 from bank address A and bank-column 0 from addr A+1
                        if (j < BANK_WIDTH / 2) { // 1st 8 locations filled from bank 1, LSB - using even bytes, MSB -
                                                  // using odd bytes
                            concat_line[i][j] = in_pix[i][1].pix[j * 2 + msb_lsb_sel[i][1]];
                            concat_flag[i][j] = in_pix_flag[i][1][j * 2 + msb_lsb_sel[i][1]];
                        } else { //{ //next 8 locations filled from bank 0,
                            concat_line[i][j] = in_pix[i][0].pix[(j - BANK_WIDTH / 2) * 2 + msb_lsb_sel[i][0]];
                            concat_flag[i][j] = in_pix_flag[i][0][(j - BANK_WIDTH / 2) * 2 + msb_lsb_sel[i][0]];
                        }
                    }
                } else { // pixel is 4B (mesh) w/ reshuffle storage
                    if (rd_mode[i] == 0) { // we read both bank-columns from  same address
                        if (j < BANK_WIDTH / 2) { // 1st 8 locations filled from bank 0, LSB - using even pixels, MSB -
                                                  // using odd pixels
                            concat_line[i][j] = in_pix[i][0].pix[(j >> 2) * 8 + (j & 3) + 4 * msb_lsb_sel[i][0]];
                            concat_flag[i][j] = in_pix_flag[i][0][(j >> 2) * 8 + (j & 3) + 4 * msb_lsb_sel[i][0]];
                        } else { //{ //next 8 locations filled from bank 1,
                            concat_line[i][j] =
                                in_pix[i][1].pix[((j >> 2) - BANK_WIDTH / 8) * 8 + (j & 3) + 4 * msb_lsb_sel[i][1]];
                            concat_flag[i][j] =
                                in_pix_flag[i][1][((j >> 2) - BANK_WIDTH / 8) * 8 + (j & 3) + 4 * msb_lsb_sel[i][1]];
                        }
                    } else { // we read bank-column 1 from bank address A and bank-column 0 from addr A+1
                        if (j < BANK_WIDTH / 2) { // 1st 8 locations filled from bank 1, LSB - using even pixels, MSB -
                                                  // using odd pixels
                            concat_line[i][j] = in_pix[i][1].pix[(j >> 2) * 8 + (j & 3) + 4 * msb_lsb_sel[i][1]];
                            concat_flag[i][j] = in_pix_flag[i][1][(j >> 2) * 8 + (j & 3) + 4 * msb_lsb_sel[i][1]];
                        } else { //{ //next 8 locations filled from bank 0,
                            concat_line[i][j] =
                                in_pix[i][0].pix[((j >> 2) - BANK_WIDTH / 8) * 8 + (j & 3) + 4 * msb_lsb_sel[i][0]];
                            concat_flag[i][j] =
                                in_pix_flag[i][0][((j >> 2) - BANK_WIDTH / 8) * 8 + (j & 3) + 4 * msb_lsb_sel[i][0]];
                        }
                    }
                }
            }
        }
    }

#ifdef STANDALONE_ROTATOR
    if (print_log_file && block_type == e_irt_block_mesh && ofifo_push) {
        for (uint8_t i = 0; i < ROW_BANKS; i++) {
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             "%s concatenated line: YT_bank_row %d bank_row %d [",
                             my_string.c_str(),
                             YT_bank_row,
                             i);
            for (uint8_t j = 0; j < COL_BANKS * BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)((unsigned char)concat_line[i][j]));
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "][");
            for (uint8_t j = 0; j < COL_BANKS * BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%2x ", (unsigned int)((unsigned char)concat_flag[i][j]));
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "]\n");
        }
    }
#endif

    // lines reorder
    for (uint8_t i = 0; i < ROW_BANKS; i++) {
        for (uint8_t j = 0; j < COL_BANKS * BANK_WIDTH; j++) {
            reorder_line[i][j] = concat_line[(YT_bank_row + i) % ROW_BANKS][j];
            reorder_flag[i][j] = concat_flag[(YT_bank_row + i) % ROW_BANKS][j];
        }
    }

    // lanes shift
    uint8_t Ps1, mesh_ps;
    if (block_type == e_irt_block_rot) {
        Ps1 = irt_top->irt_cfg.buf_1b_mode[e_irt_block_rot] == 1 ||
              irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps == 1;
        mesh_ps = 0;
    } else {
        Ps1 = irt_top->irt_cfg.buf_1b_mode[e_irt_block_mesh] == 1 ||
              irt_top->irt_desc[oicc_task].image_par[MIMAGE].Ps == 3;
        mesh_ps = (irt_top->irt_desc[oicc_task].image_par[MIMAGE].Ps == 2 ? 4 : 8);
    }

    for (uint8_t i = 0; i < ROW_BANKS; i++) {
        for (uint8_t j = 0; j < 16; j++) {

            if (block_type == e_irt_block_rot) {
                if (rd_shift[(YT_bank_row + i) % ROW_BANKS] + j >= 0 &&
                    rd_shift[(YT_bank_row + i) % ROW_BANKS] + j < ((COL_BANKS * BANK_WIDTH) >> Ps1)) {
                    out_pix[i].pix[j] = reorder_line[i][(j + rd_shift[(YT_bank_row + i) % ROW_BANKS])
                                                        << irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps];
                    if (irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps == 1)
                        out_pix[i].pix[j] |=
                            ((uint16_t)reorder_line[i][((j + rd_shift[(YT_bank_row + i) % ROW_BANKS])
                                                        << irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps) +
                                                       1]
                             << 8);
                    out_pix_flag[i].pix[j] = reorder_flag[i][(
                        j + rd_shift[(YT_bank_row + i) %
                                     ROW_BANKS])]; // << irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps];
                } else {
                    out_pix[i].pix[j]      = 0;
                    out_pix_flag[i].pix[j] = 1;
                }
            } else {
                if (rd_shift[(YT_bank_row + i) % ROW_BANKS] + j >= 0 &&
                    rd_shift[(YT_bank_row + i) % ROW_BANKS] + j < ((COL_BANKS * BANK_WIDTH) >> Ps1)) {
                    out_pix[i].pix[j] = 0;
                    for (int8_t byte = mesh_ps - 1; byte >= 0; byte--) {
                        out_pix[i].pix[j] = (out_pix[i].pix[j] << 8) |
                                            (reorder_line[i][((j + rd_shift[(YT_bank_row + i) % ROW_BANKS])
                                                              << irt_top->irt_desc[oicc_task].image_par[MIMAGE].Ps) +
                                                             byte]);
                    }
                    out_pix_flag[i].pix[j] = reorder_flag[i][(
                        j + rd_shift[(YT_bank_row + i) %
                                     ROW_BANKS])]; // << irt_top->irt_desc[oicc_task].image_par[IIMAGE].Ps];
                } else {
                    out_pix[i].pix[j]      = 0;
                    out_pix_flag[i].pix[j] = 1;
                }
            }
        }
    }

#ifdef STANDALONE_ROTATOR
    if (print_log_file && ofifo_push) {
        for (uint8_t i = 0; i < ROW_BANKS; i++) {
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             "BG gen: i %d rd_shift %d rd_shift+j[",
                             i,
                             rd_shift[(YT_bank_row + i) % ROW_BANKS]);
            for (uint8_t j = 0; j < COL_BANKS * BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%d ", rd_shift[(YT_bank_row + i) % ROW_BANKS] + j);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "] flag[");
            for (uint8_t j = 0; j < COL_BANKS * BANK_WIDTH; j++)
                IRT_TRACE_TO_LOG(
                    LOG_TRACE_L1, log_file, "%d ", reorder_flag[i][j + rd_shift[(YT_bank_row + i) % ROW_BANKS]]);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "]\n");
        }
    }
#endif

#ifdef STANDALONE_ROTATOR
    if (print_log_file && block_type == e_irt_block_mesh && ofifo_push) {
        for (uint8_t i = 0; i < ROW_BANKS; i++) {
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%s out_pix: [", my_string.c_str());
            for (uint8_t j = 0; j < 16; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%llx ", (uint64_t)out_pix[i].pix[j]);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "][");
            for (uint8_t j = 0; j < 16; j++)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "%x ", out_pix_flag[i].pix[j]);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "]\n");
        }
    }
#endif
    // used for debug - 2x2 sel gets pixels directly from ext memory
#if 0
	for (int i=0; i<8; i++) {
		for (int j=0; j<16; j++) {
			out_pix[i].pix[j] = ext_mem[image_pars[IIMAGE].ADDR+ oicc_task*IMAGE_W*IMAGE_H + (YT+i) * image_pars[IIMAGE].W + XL8+j];
			out_pix_flag[i].pix[j] = ((YT+i)<0 || (YT+i)>=image_pars[IIMAGE].H || (XL8+i)<0 || (XL8+i)>=image_pars[IIMAGE].W) ? 1 : 0;
		}
	}
#endif
#ifdef STANDALONE_ROTATOR
#ifdef CREATE_IMAGE_DUMPS_IWC
    uint16_t row_padded = (16 * 3 + 3) & (~3);
    uint8_t* data       = new uint8_t[row_padded];
#endif

    if (ofifo_push && ((Yo + irt_top->irt_desc[irt_top->task[e_irt_oicc]].image_par[OIMAGE].Yc) >> 1) == 0 &&
        ((Xo0 + irt_top->irt_desc[irt_top->task[e_irt_oicc]].image_par[OIMAGE].Xc) >> 1) == 0) {
#ifdef CREATE_IMAGE_DUMPS_IWC
        if (oicc_task == 0) {
            f1 = fopen("IRT_8x16_IWC_in.bmp", "wb");
            f2 = fopen("IRT_8x16_IWC_concat.bmp", "wb");
            f3 = fopen("IRT_8x16_IWC_reorder.bmp", "wb");
            f4 = fopen("IRT_8x16_IWC_shift.bmp", "wb");
            generate_bmp_header(f1, 16, 8);
            generate_bmp_header(f2, 16, 8);
            generate_bmp_header(f3, 16, 8);
            generate_bmp_header(f4, 16, 8);
        }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                // input
                win16x8[0][oicc_task][i].pix[j]     = in_pix[i][0].pix[j];
                win16x8[0][oicc_task][i].pix[j + 8] = in_pix[i][1].pix[j];
                // after concat
                win16x8[1][oicc_task][i].pix[j]     = concat_line[i][j];
                win16x8[1][oicc_task][i].pix[j + 8] = concat_line[i][j + 8];
                // after reorder
                win16x8[2][oicc_task][i].pix[j]     = reorder_line[i][j];
                win16x8[2][oicc_task][i].pix[j + 8] = reorder_line[i][j + 8];
                // after shift
                win16x8[3][oicc_task][i].pix[j]     = (uint8_t)out_pix[i].pix[j];
                win16x8[3][oicc_task][i].pix[j + 8] = (uint8_t)out_pix[i].pix[j + 8];
            }
        }

        if (oicc_task == 2) {
            for (int i = 0; i < 8; i++) {
                // input
                for (int j = 0; j < 16 * 3; j += 3) {
                    data[j]     = win16x8[0][0][i].pix[j / 3];
                    data[j + 1] = win16x8[0][1][i].pix[j / 3];
                    data[j + 2] = win16x8[0][2][i].pix[j / 3];
                    ;
                }
                fwrite(data, sizeof(unsigned char), row_padded, f1);
                // after concat
                for (int j = 0; j < 16 * 3; j += 3) {
                    data[j]     = win16x8[1][0][i].pix[j / 3];
                    data[j + 1] = win16x8[1][1][i].pix[j / 3];
                    data[j + 2] = win16x8[1][2][i].pix[j / 3];
                    ;
                }
                fwrite(data, sizeof(unsigned char), row_padded, f2);
                // after reorder
                for (int j = 0; j < 16 * 3; j += 3) {
                    data[j]     = win16x8[2][0][i].pix[j / 3];
                    data[j + 1] = win16x8[2][1][i].pix[j / 3];
                    data[j + 2] = win16x8[2][2][i].pix[j / 3];
                    ;
                }
                fwrite(data, sizeof(unsigned char), row_padded, f3);
                // after shift
                for (int j = 0; j < 16 * 3; j += 3) {
                    data[j]     = win16x8[3][0][i].pix[j / 3];
                    data[j + 1] = win16x8[3][1][i].pix[j / 3];
                    data[j + 2] = win16x8[3][2][i].pix[j / 3];
                    ;
                }
                fwrite(data, sizeof(unsigned char), row_padded, f4);
            }
            fclose(f1);
            fclose(f2);
            fclose(f3);
            fclose(f4);
        }
#endif
    }
#endif
}

template <class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, Eirt_block_type block_type>
void IRT_top::IRT_2x2_sel<BUS_IN, BUS_OUT, ROW_BANKS, block_type>::run(uint8_t       index,
                                                                       int16_t       XL,
                                                                       int16_t       XR,
                                                                       int16_t       YT,
                                                                       int16_t       YB,
                                                                       float         Xi0_float,
                                                                       float         Yi0_float,
                                                                       int64_t       Xi_fixed,
                                                                       int64_t       Yi_fixed,
                                                                       BUS_IN        in_pix[ROW_BANKS],
                                                                       bus16f_struct in_flags[ROW_BANKS],
                                                                       BUS_OUT&      out_pix,
                                                                       uint8_t       irt_task,
                                                                       bool          ofifo_push)
{

    std::string my_string = getInstanceName();

    if (irt_top->irt_desc[irt_task].crd_mode == e_irt_crd_mode_fp32 ||
        irt_top->irt_desc[irt_task].irt_mode == e_irt_projection) {
        // Xi_fixed[index] = (int64_t)rint(Xi_float * pow(2, 31));
        // Yi_fixed[index] = (int64_t)rint(Yi_float * pow(2, 31));
    }

    // int64_t Xi_fixed[ROW_BANKS], Yi_fixed[ROW_BANKS];
    // IRT_top::irt_iicc(irt_top, irt_task, Xo0, Yo, 8, Xi_fixed, Yi_fixed);

    int      XI_fixed = (int)(Xi_fixed >> 31);
    int      YI_fixed = (int)(Yi_fixed >> 31);
    uint32_t Xf_fixed = (uint32_t)(Xi_fixed - ((int64_t)XI_fixed * ((int64_t)1 << 31)));
    uint32_t Yf_fixed = (uint32_t)(Yi_fixed - ((int64_t)YI_fixed * ((int64_t)1 << 31)));
    //---------------------------------------
    // ROTSIM_DV_INTGR - rmrm probe poinM
    irt_top->irt_top_sig.dv_Xf_fixed[index] = Xf_fixed;
    irt_top->irt_top_sig.dv_Yf_fixed[index] = Yf_fixed;
    //---------------------------------------

    //---------------------------------------
    // ROTSIM_DV_INTGR - rmrm probe poinM
    irt_top->irt_top_sig.dv_Xf_fixed[index] = Xf_fixed;
    irt_top->irt_top_sig.dv_Yf_fixed[index] = Yf_fixed;
    //---------------------------------------

#if 1
    if (block_type == e_irt_block_rot) {
#ifdef IRT_USE_FLIP_FOR_MINUS1
        if (irt_top->irt_desc[irt_task].read_vflip == 1) {
            Yf_fixed = (uint32_t)(((int64_t)1 << 31) - Yi_fixed + ((int64_t)YI_fixed * ((int64_t)1 << 31)));
        }
        if (irt_top->irt_desc[irt_task].read_hflip == 1) {
            Xf_fixed = (uint32_t)(((int64_t)1 << 31) - Xi_fixed + ((int64_t)XI_fixed * ((int64_t)1 << 31)));
        }
#endif
    }
#endif

    int Y0, Y1, X0, X1;
#ifdef IRT_USE_FLIP_FOR_MINUS1
    if (irt_top->irt_desc[irt_task].read_vflip == 0 || block_type == e_irt_block_mesh) {
#endif
        Y0 = YI_fixed - YT;
        Y1 = YI_fixed - YT + (YB - YT);
#ifdef IRT_USE_FLIP_FOR_MINUS1
    } else {
        Y0 = YI_fixed - YT;
        Y1 = YI_fixed - YT - (YB - YT);
    }
#endif
#ifdef IRT_USE_FLIP_FOR_MINUS1
    if (irt_top->irt_desc[irt_task].read_hflip == 0 || block_type == e_irt_block_mesh) {
#endif
        X0 = XI_fixed - XL;
        X1 = XI_fixed - XL + (XR - XL);
#ifdef IRT_USE_FLIP_FOR_MINUS1
    } else {
        X0 = XI_fixed - XL;
        X1 = XI_fixed - XL - (XR - XL);
    }
#endif

    if (block_type == e_irt_block_rot) { // relevant in rot core only
        switch (irt_top->irt_desc[irt_task].irt_mode) {
            case e_irt_rotation:
                if (irt_top->irt_desc[irt_task].rot90) {
                    Y1 = std::min(Y1, ROW_BANKS - 1); // Y1 < 8 ? Y1 : 7;
                    if (irt_top->irt_desc[irt_task].rot90_intv == 0)
                        Y1 = Y0;
                    if (irt_top->irt_desc[irt_task].rot90_inth == 0)
                        X1 = X0;
                    IRT_TRACE_TO_LOG(5, IRTLOG, "inside rot90_intv Y1 %d Y0 %d\n", Y1, Y0);
                }
                break;
            default:
                if (irt_top->irt_desc[irt_task].rot90) {
                    Y1 = std::min(Y1, ROW_BANKS - 1); // Y1 < 8 ? Y1 : 7;
                    if (irt_top->irt_desc[irt_task].rot90_intv == 0)
                        Y1 = Y0;
                    if (irt_top->irt_desc[irt_task].rot90_inth == 0)
                        X1 = X0;
                }
                break;
        }
    }

    if (ofifo_push) {
        // IRT_TRACE("index = %d, Ys = %d, Yi0_fixed = %f, YI_fixed = %d\n", index, Ys, Yi0_fixed/pow(2.0,
        // irt_top->irt_desc[irt_task].TOTAL_PREC), YI_fixed);
        if (Y0 < 0 || Y0 > ROW_BANKS - 1) {
            IRT_TRACE("%s: YI_fixed - Ys = %d invalid, index=%d, YI_fixed=%d, Yi0_fixed=%f, Ys=%d, task %d\n",
                      my_string.c_str(),
                      Y0,
                      index,
                      YI_fixed,
                      Yi_fixed / pow(2.0, irt_top->rot_pars[irt_task].TOTAL_PREC),
                      YT,
                      irt_task);
            IRT_TRACE_TO_RES(
                test_res, " failed, %s YI_fixed - Ys = %d invalid, task %d\n", my_string.c_str(), Y0, irt_task);
            IRT_CLOSE_FAILED_TEST(0);
        }
        if (X0 < 0 || X0 > 15) {
            IRT_TRACE("%s: XI_fixed - Xs = %d invalid, task %d\n", my_string.c_str(), X0, irt_task);
            IRT_TRACE_TO_RES(
                test_res, " failed, %s XI_fixed - Xs = %d invalid, task %d\n", my_string.c_str(), X0, irt_task);
            IRT_CLOSE_FAILED_TEST(0);
        }

        if (Y1 < 0 || Y1 > ROW_BANKS - 1) {
            IRT_TRACE("%s: Y1 = YI_fixed[%d] - Ys[%d] %s 1 = %d invalid, index %d, task %d\n",
                      my_string.c_str(),
                      YI_fixed,
                      YT,
                      irt_top->irt_desc[irt_task].read_vflip == 0 ? "+" : "-",
                      Y1,
                      index,
                      irt_task);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s Y1 = YI_fixed[%d] - Ys[%d] %s 1 = %d invalid, index %d, task %d",
                             my_string.c_str(),
                             YI_fixed,
                             YT,
                             irt_top->irt_desc[irt_task].read_vflip == 0 ? "+" : "-",
                             Y1,
                             index,
                             irt_task);
            IRT_CLOSE_FAILED_TEST(0);
        }
        if (X1 < 0 || X1 > 15) {
            IRT_TRACE("%s: X1 = XI_fixed[%d] - Xs[%d] %s 1 = %d invalid, index %d, task %d\n",
                      my_string.c_str(),
                      XI_fixed,
                      XL,
                      irt_top->irt_desc[irt_task].read_hflip == 0 ? "+" : "-",
                      X1,
                      index,
                      irt_task);
            IRT_TRACE_TO_RES(test_res,
                             " failed, %s X1 = XI_fixed[%d] - Xs[%d] %s 1 = %d invalid, index %d, task %d",
                             my_string.c_str(),
                             XI_fixed,
                             XL,
                             irt_top->irt_desc[irt_task].read_hflip == 0 ? "+" : "-",
                             X1,
                             index,
                             irt_task);
            IRT_CLOSE_FAILED_TEST(0);
        }
    }

    out_pix.pix[0] = in_pix[Y0].pix[X0];
    out_pix.pix[1] = in_pix[Y0].pix[X1];
    out_pix.pix[2] = in_pix[Y1].pix[X0];
    out_pix.pix[3] = in_pix[Y1].pix[X1];

    out_pix.pix_bg[0] = in_flags[Y0].pix[X0];
    out_pix.pix_bg[1] = in_flags[Y0].pix[X1];
    out_pix.pix_bg[2] = in_flags[Y1].pix[X0];
    out_pix.pix_bg[3] = in_flags[Y1].pix[X1];
    IRT_TRACE_TO_LOG(
        5,
        IRTLOG,
        "2x2 task%d : Y1:Y0 %d:%d X1:X0 %d:%d rot90:90V %d:%d pix[0:3][%lx:%lx:%lx:%lx] bg[0:3][%x:%x:%x:%x]\n",
        irt_task,
        Y1,
        Y0,
        X1,
        X0,
        irt_top->irt_desc[irt_task].rot90,
        irt_top->irt_desc[irt_task].rot90_intv,
        in_pix[Y0].pix[X0],
        in_pix[Y0].pix[X1],
        in_pix[Y1].pix[X0],
        in_pix[Y1].pix[X1],
        (unsigned int)in_flags[Y0].pix[X0],
        (unsigned int)in_flags[Y0].pix[X1],
        (unsigned int)in_flags[Y1].pix[X0],
        (unsigned int)in_flags[Y1].pix[X1]);

#if 0
	if (irt_top->irt_desc[irt_task].read_flip==1) {
		out_pix.pix[0] = in_pix[Y0].pix[X0];
		out_pix.pix[1] = in_pix[Y0].pix[X1];
		out_pix.pix[2] = in_pix[Y1].pix[X0];
		out_pix.pix[3] = in_pix[Y1].pix[X1];

		out_pix.pix_bg[0]= in_flags[Y0].pix[X0];
		out_pix.pix_bg[1]= in_flags[Y0].pix[X1];
		out_pix.pix_bg[2]= in_flags[Y1].pix[X0];
		out_pix.pix_bg[3]= in_flags[Y1].pix[X1];
	}
#endif
    IRT_TRACE_TO_LOG(5,
                     IRTLOG,
                     "%s [%d]: Y[%d,%d] X[%d,%d] YT,YB[%d,%d] [%llx,%llx,%llx,%llx] \n",
                     my_string.c_str(),
                     index,
                     Y0,
                     Y1,
                     X0,
                     X1,
                     YT,
                     YB,
                     (long long unsigned int)out_pix.pix[0],
                     (long long unsigned int)out_pix.pix[1],
                     (long long unsigned int)out_pix.pix[2],
                     (long long unsigned int)out_pix.pix[3]);

#if 1
    out_pix.weights_fixed[0] = Xf_fixed;
    out_pix.weights_fixed[1] = Yf_fixed;
#endif

#if 1
#ifdef STANDALONE_ROTATOR
    if (ofifo_push) {
        if (print_log_file && block_type == e_irt_block_mesh && ofifo_push) {
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             "%s %d: [y00=%2d,x00=%2d], [%llx,%llx,%llx,%llx] [%4x,%4x]",
                             my_string.c_str(),
                             index,
                             YI_fixed,
                             XI_fixed,
                             (uint64_t)out_pix.pix[0],
                             (uint64_t)out_pix.pix[1],
                             (uint64_t)out_pix.pix[2],
                             (uint64_t)out_pix.pix[3],
                             out_pix.weights_fixed[0],
                             out_pix.weights_fixed[1]);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                             log_file,
                             " [%d,%d,%d,%d]",
                             out_pix.pix_bg[0],
                             out_pix.pix_bg[1],
                             out_pix.pix_bg[2],
                             out_pix.pix_bg[3]);
            IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "\n");
        }
    }
#endif
#endif
}

template <class BUS_IN, class BUS_OUT, Eirt_block_type block_type>
BUS_OUT IRT_top::IRT_BLI<BUS_IN, BUS_OUT, block_type>::run(BUS_IN p2x2, uint16_t bg, bool ofifo_push, uint8_t irt_task)
{

    std::string my_string = getInstanceName();

#ifdef STANDALONE_ROTATOR
    double result, result1, result2;
#endif
    int res;

    float weights[2][2];
    weights[0][0] = (float)((double)p2x2.weights_fixed[0] / pow(2.0, 31)); // xw0
    weights[0][1] = (float)((pow(2.0, 31) - (double)p2x2.weights_fixed[0]) / pow(2.0, 31)); // xw1
    weights[1][0] = (float)((double)p2x2.weights_fixed[1] / pow(2.0, 31)); // yw0
    weights[1][1] = (float)((pow(2.0, 31) - (double)p2x2.weights_fixed[1]) / pow(2.0, 31)); // yw1

    // force weight to 1 and 0 for nearest neigbour interpolation for rotation core
    if (block_type == e_irt_block_rot && irt_top->irt_desc[irt_task].int_mode) {

        for (uint8_t i = 0; i < 2; i++) {
            for (uint8_t j = 0; j < 2; j++) {
                if (weights[i][j] > (float)0.5) {
                    // IRT_TRACE_TO_LOG(2,IRTLOG,"NNI-1[%d:%d-%f] ",i,j, weights[i][j]);
                    weights[i][j] = (float)1.0;
                } else if (weights[i][j] < (float)0.5) {
                    // IRT_TRACE_TO_LOG(2,IRTLOG,"NNI-0[%d:%d-%f] ",i,j,weights[i][j]);
                    weights[i][j] = (float)0.0;
                } else {
                    // IRT_TRACE_TO_LOG(2,IRTLOG,"NNI-0.5[%d:%d-%f] ",i,j,weights[i][j]);
                }
            }
        }
        // IRT_TRACE_TO_LOG(2,IRTLOG,"\n");

        //---------------------------------------
        // ROTSIM_DV_INTGR
        if (p2x2.weights_fixed[0] > 0x40000000) {
            weights[0][0] = (float)1.0;
            weights[0][1] = (float)0.0;
        } else if (p2x2.weights_fixed[0] < 0x40000000) {
            weights[0][0] = (float)0.0;
            weights[0][1] = (float)1.0;
        }
        if (p2x2.weights_fixed[1] > 0x40000000) {
            weights[1][0] = (float)1.0;
            weights[1][1] = (float)0.0;
        } else if (p2x2.weights_fixed[1] < 0x40000000) {
            weights[1][0] = (float)0.0;
            weights[1][1] = (float)1.0;
        }
        //---------------------------------------

        for (uint8_t i = 0; i < 2; i++) {
#ifdef STANDALONE_ROTATOR
            if (p2x2.weights[i] > pow(2.0, irt_top->rot_pars[irt_task].TOTAL_PREC - 1))
                p2x2.weights[i] = pow(2.0, irt_top->rot_pars[irt_task].TOTAL_PREC);
            else if (p2x2.weights[i] < pow(2.0, irt_top->rot_pars[irt_task].TOTAL_PREC))
                p2x2.weights[i] = 0;
#endif
            if (p2x2.weights_fixed[i] > ((uint32_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - 1)))
                p2x2.weights_fixed[i] = ((uint32_t)1 << irt_top->rot_pars[irt_task].TOTAL_PREC);
            else if (p2x2.weights_fixed[i] < ((uint32_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - 1)))
                p2x2.weights_fixed[i] = 0;
        }
    }

#ifdef STANDALONE_ROTATOR

    // horizontal followed by vertical
    result1 = (uint64_t)((uint16_t)p2x2.pix[0] & irt_top->irt_desc[irt_task].Msi) * (1 - p2x2.weights[0]);
    result1 += (uint64_t)((uint16_t)p2x2.pix[1] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights[0];

    result2 = (uint64_t)((uint16_t)p2x2.pix[2] & irt_top->irt_desc[irt_task].Msi) * (1 - p2x2.weights[0]);
    result2 += (uint64_t)((uint16_t)p2x2.pix[3] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights[0];

    result = result1 * (1 - p2x2.weights[1]);
    result += result2 * p2x2.weights[1];

    // vertical vertical by horizontal
    result1 = (uint64_t)((uint16_t)p2x2.pix[0] & irt_top->irt_desc[irt_task].Msi) * (1 - p2x2.weights[1]);
    result1 += (uint64_t)((uint16_t)p2x2.pix[2] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights[1];

    result2 = (uint64_t)((uint16_t)p2x2.pix[1] & irt_top->irt_desc[irt_task].Msi) * (1 - p2x2.weights[1]);
    result2 += (uint64_t)((uint16_t)p2x2.pix[3] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights[1];

    result = result1 * (1 - p2x2.weights[0]);
    result += result2 * p2x2.weights[0];

#endif

    // result = result >> (2*IRT_BILINEAR_INT_WEIGHTS_PREC-1);
    // result += 1;
    // result  = result >> 1;

    // fixed point calculation
    uint64_t result_fixed, result1_fixed, result2_fixed;
    if (irt_top->rot_pars[irt_task].WEIGHT_SHIFT != 0) {
        p2x2.weights_fixed[0] = ((p2x2.weights_fixed[0] >> (irt_top->rot_pars[irt_task].WEIGHT_SHIFT - 1)) + 1) >> 1;
        p2x2.weights_fixed[1] = ((p2x2.weights_fixed[1] >> (irt_top->rot_pars[irt_task].WEIGHT_SHIFT - 1)) + 1) >> 1;
    }

    // horizontal followed by vertical
    result1_fixed =
        (uint64_t)((uint16_t)p2x2.pix[0] & irt_top->irt_desc[irt_task].Msi) *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[0]);
    result1_fixed += (uint64_t)((uint16_t)p2x2.pix[1] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights_fixed[0];

    result2_fixed =
        (uint64_t)((uint16_t)p2x2.pix[2] & irt_top->irt_desc[irt_task].Msi) *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[0]);
    result2_fixed += (uint64_t)((uint16_t)p2x2.pix[3] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights_fixed[0];

    result_fixed =
        result1_fixed *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[1]);
    result_fixed += result2_fixed * p2x2.weights_fixed[1];

    // vertical vertical by horizontal
    result1_fixed =
        (uint64_t)((uint16_t)p2x2.pix[0] & irt_top->irt_desc[irt_task].Msi) *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[1]);
    result1_fixed += (uint64_t)((uint16_t)p2x2.pix[2] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights_fixed[1];

    result2_fixed =
        (uint64_t)((uint16_t)p2x2.pix[1] & irt_top->irt_desc[irt_task].Msi) *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[1]);
    result2_fixed += (uint64_t)((uint16_t)p2x2.pix[3] & irt_top->irt_desc[irt_task].Msi) * p2x2.weights_fixed[1];

    result_fixed =
        result1_fixed *
        (((uint64_t)1 << (irt_top->rot_pars[irt_task].TOTAL_PREC - irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) -
         p2x2.weights_fixed[0]);
    result_fixed += result2_fixed * p2x2.weights_fixed[0];

    // BLI works in FP32, code above will be removed later
    float result1_fp32, result2_fp32, result_fp32;
    float pix[4];
    for (uint8_t i = 0; i < 4; i++) {
        if (block_type == e_irt_block_rot) {
            pix[i] = (float)((uint16_t)p2x2.pix[i] & irt_top->irt_desc[irt_task].Msi);
        } else {
            pix[i] = p2x2.pix[i];
        }
    }

    // horizontal followed by vertical
    result1_fp32 = pix[0] * weights[0][1] + pix[1] * weights[0][0];
    result2_fp32 = pix[2] * weights[0][1] + pix[3] * weights[0][0];
    result_fp32  = result1_fp32 * weights[1][1] + result2_fp32 * weights[1][0];

    // vertical vertical by horizontal
    result1_fp32 = pix[0] * weights[1][1] + pix[2] * weights[1][0];
    result2_fp32 = pix[1] * weights[1][1] + pix[3] * weights[1][0];
    result_fp32  = result1_fp32 * weights[0][1] + result2_fp32 * weights[0][0];

#if 0
	if (block_type == e_irt_block_rot && ofifo_push) {
		IRT_TRACE("%s: Task %d [%d, %d]: pixels[0x%x][0x%x][0x%x][0x%x] w[%f][%f][%f][%f]\n", my_string.c_str(), irt_task, p2x2.line, p2x2.pixel,
			(uint16_t)p2x2.pix[0] & irt_top->irt_desc[irt_task].Msi, (uint16_t)p2x2.pix[1] & irt_top->irt_desc[irt_task].Msi, (uint16_t)p2x2.pix[2] & irt_top->irt_desc[irt_task].Msi, (uint16_t)p2x2.pix[3] & irt_top->irt_desc[irt_task].Msi,
			weights[0][0], weights[0][1], weights[1][0], weights[1][1]);
	}
#endif

#if 1
    if (block_type == e_irt_block_rot) {
        if (irt_top->irt_desc[irt_task].bg_mode ==
            e_irt_bg_frame_repeat) { // && irt_top->irt_desc[irt_task].rot90 == 0) {
            IRT_TRACE("e_irt_bg_frame_repeat --  ");
            if (p2x2.pix_bg[0] & p2x2.pix_bg[1] & p2x2.pix_bg[2] & !p2x2.pix_bg[3]) { // top left corner
                result_fp32 = pix[3];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    1; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 1\n");
            } else if (p2x2.pix_bg[0] & p2x2.pix_bg[1] & !p2x2.pix_bg[2] & p2x2.pix_bg[3]) { // top right corner
                result_fp32 = pix[2];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    2; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 2\n");
            } else if (p2x2.pix_bg[0] & p2x2.pix_bg[1] & !p2x2.pix_bg[2] & !p2x2.pix_bg[3]) { // top side
                result_fp32 = pix[2];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    3; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 3\n");
            } else if (!p2x2.pix_bg[0] & p2x2.pix_bg[1] & p2x2.pix_bg[2] & p2x2.pix_bg[3]) { // bottom right corner
                result_fp32 = pix[0];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    4; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 4\n");
            } else if (!p2x2.pix_bg[0] & p2x2.pix_bg[1] & !p2x2.pix_bg[2] & p2x2.pix_bg[3]) { // right side
                result_fp32 = pix[0];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    5; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 5\n");
            } else if (!p2x2.pix_bg[0] & !p2x2.pix_bg[1] & p2x2.pix_bg[2] & p2x2.pix_bg[3]) { // bottom side
                result_fp32 = pix[0];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    6; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 6\n");
            } else if (p2x2.pix_bg[0] & !p2x2.pix_bg[1] & p2x2.pix_bg[2] & p2x2.pix_bg[3]) { // bottom left corner
                result_fp32 = pix[1];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    7; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 7\n");
            } else if (p2x2.pix_bg[0] & !p2x2.pix_bg[1] & p2x2.pix_bg[2] & !p2x2.pix_bg[3]) { // left side
                result_fp32 = pix[1];
                irt_top->irt_top_sig.bg_mode_oob_cp[irt_top->irt_top_sig.bg_mode_oob_cp_pxl_index++] =
                    8; //   ROTSIM_DV_INTGR -- for OOB coverage
                IRT_TRACE("bg_mode_oob_cp 8\n");
            }
        }
    }
#endif

    // int res1 = (int)floor(result_fp32 * pow(2.0, 9) + 0.5);
    // res1 = res1 >> (irt_top->irt_desc[irt_task].bli_shift);
    // res1 = (res1 + 1) >> 1;
    if (block_type == e_irt_block_rot) {
        result_fp32 *= (float)pow(2.0, irt_top->irt_desc[irt_task].bli_shift);
    }

    if (ofifo_push && print_log_file)
        IRT_TRACE_TO_LOG(LOG_TRACE_L1, log_file, "result_fixed %" PRIu64 "\n", result_fixed);

    // result_fixed =
    // ((((result_fixed+IRT_COORD_CAL_ROUND)>>irt_top->irt_desc[irt_task].COORD_CAL_PREC)+IRT_COORD_CAL_ROUND)>>irt_top->irt_desc[irt_task].COORD_CAL_PREC);
    // result_fixed =
    // ((result_fixed>>irt_top->irt_desc[irt_task].COORD_CAL_PREC)+IRT_COORD_CAL_ROUND)>>irt_top->irt_desc[irt_task].COORD_CAL_PREC;

    result_fixed =
        ((result_fixed >> (irt_top->rot_pars[irt_task].bli_shift_fix - 2 * irt_top->rot_pars[irt_task].WEIGHT_SHIFT)) +
         1) >>
        1;

    // res = (int) res_fixed + 100;
#ifdef STANDALONE_ROTATOR
    res = (int)floor((double)result + 0.5);
#endif

    res = (int)result_fixed;
    // BLI works in FP32, code above will be removed later
    res = (int)rint(result_fp32);

#if 0
	if (block_type == e_irt_block_rot && ofifo_push)
		IRT_TRACE_TO_LOG(log_file, "BLI: [%2x,%2x,%2x,%2x] res %d\n", p2x2.pix[0], p2x2.pix[1], p2x2.pix[2], p2x2.pix[3], res);

	if (block_type == e_irt_block_mesh && ofifo_push)
		IRT_TRACE_TO_LOG(log_file, "BLI: [%f,%fx,%fx,%fx] res %f\n", p2x2.pix[0], p2x2.pix[1], p2x2.pix[2], p2x2.pix[3], result_fp32);
#endif
    bool rot90_int = irt_top->irt_desc[irt_task].rot90_inth | irt_top->irt_desc[irt_task].rot90_intv;
    if (block_type == e_irt_block_rot) {
        if (irt_top->irt_desc[irt_task].bg_mode ==
            e_irt_bg_frame_repeat) { // && irt_top->irt_desc[irt_task].rot90 == 0) {

        } else {
            if ((p2x2.pix_bg[0] | p2x2.pix_bg[1] | p2x2.pix_bg[2] | p2x2.pix_bg[3]) == 1 &&
                (irt_top->irt_desc[irt_task].rot90 == 0 || rot90_int || p2x2.pix_bg[0])) { //?
                res = (int)bg;
            }
        }
    } else {
        if ((p2x2.pix_bg[0] | p2x2.pix_bg[1] | p2x2.pix_bg[2] | p2x2.pix_bg[3]) == 1) {
            res = (int)bg;
        }
    }

    res = std::min(res, (int)irt_top->irt_desc[irt_task].MAX_VALo); // if (res > 255) res = 255;
    res = std::max(res, 0); // if (res < 0) res = 0;
    switch (irt_top->irt_cfg.debug_mode) {
        case e_irt_debug_bg_out: res = 0; break;
        default: break;
    }
    //   IRT_TRACE_TO_LOG(2,IRTLOG,"BLI task %d: PIX[%2x,%2x,%2x,%2x] res %x res<<pso %x Ppo %d weights [%x,%x]
    //   [%f,%f,%f,%f] [%f,%f,%f,%f] MAX_VALo %x\n", irt_task, p2x2.pix[0], p2x2.pix[1], p2x2.pix[2], p2x2.pix[3], res,
    //   (res << irt_top->irt_desc[irt_task].Ppo), irt_top->irt_desc[irt_task].Ppo,
    //         p2x2.weights_fixed[0], p2x2.weights_fixed[1],
    //         (float)((double)p2x2.weights_fixed[0] / pow(2.0, 31)),(float)((pow(2.0, 31) -
    //         (double)p2x2.weights_fixed[0]) / pow(2.0, 31)),(float)((double)p2x2.weights_fixed[1] / pow(2.0,
    //         31)),(float)((pow(2.0, 31) - (double)p2x2.weights_fixed[1]) / pow(2.0, 31)),
    //         weights[0][0],weights[0][1],weights[1][0],weights[1][1], (int)irt_top->irt_desc[irt_task].MAX_VALo
    //         );

    // return p2x2.pix[3];
    return (block_type == e_irt_block_rot ? (BUS_OUT)(res << irt_top->irt_desc[irt_task].Ppo) : (BUS_OUT)result_fp32);
}

void IRT_top::IRT_OFIFO::run(bool            push,
                             bool            pop,
                             uint8_t         pixel_valid,
                             bus8ui16_struct data_in,
                             bus128B_struct& data_out,
                             bool&           empty,
                             bool&           full,
                             bool            task_end)
{

    Eirt_blocks_enum ofifo_task =
        irt_top->irt_cfg.flow_mode == e_irt_flow_nCFIFO_fixed_adaptive_wxh ? e_irt_oicc : e_irt_iiirc;
    if (pop) {
        if (fullness == 0) {
            IRT_TRACE("Pop from empty output FIFO\n");
            IRT_TRACE_TO_RES(test_res, " failed, Pop from empty output FIFO\n");
            IRT_CLOSE_FAILED_TEST(0);
        }
        IRT_TRACE_TO_LOG(5,
                         IRTLOG,
                         "OFIFO pop: slot = %d, wp = %d, rp = %d, fullness=%d\n",
                         irt_ofifo_w_cnt,
                         irt_ofifo_wp,
                         irt_ofifo_rp,
                         fullness);
        for (uint8_t k = 0; k < IRT_OFIFO_WIDTH; k++) {
            irt_ofifo[irt_ofifo_rp].en[k] = 0;
        }
        irt_ofifo_rp = (irt_ofifo_rp + 1) % IRT_OFIFO_DEPTH;
        fullness--;
    }

    if (push) {
        if (fullness >= IRT_OFIFO_DEPTH) {
            IRT_TRACE("Push to full output FIFO\n");
            IRT_TRACE_TO_RES(test_res, " failed, Push to full output FIFO\n");
            IRT_CLOSE_FAILED_TEST(0);
        }
        // IRT_TRACE("OFIFO push: task %d, pixel_cnt = %d slot = %d, wp = %d, rp = %d, fullness=%d, pixel valid %x cycle
        // %d\n", irt_top->task[ofifo_task], pixel_cnt, irt_ofifo_w_cnt, irt_ofifo_wp, irt_ofifo_rp, fullness,
        // pixel_valid, cycle);

        // IRT_TRACE("OFIFO: push %d pixels to entry %d cnt %d at cycle %d\n", pixel_valid, irt_ofifo_wp,
        // irt_ofifo_w_cnt, cycle);
        for (int i = 0; i < IRT_ROT_MAX_PROC_SIZE; i++) {
            if (i < pixel_valid) {
                irt_ofifo[irt_ofifo_wp].pix[irt_ofifo_w_cnt] = (unsigned char)(data_in.pix[i] & 0xff);
                irt_ofifo[irt_ofifo_wp].en[irt_ofifo_w_cnt]  = 1;
                if (irt_top->irt_desc[irt_top->task[ofifo_task]].image_par[OIMAGE].Ps == 0) { // 1 byte/pixel
                    irt_ofifo_w_cnt++;
                } else { // 2 bytes/pixel
                    irt_ofifo[irt_ofifo_wp].pix[irt_ofifo_w_cnt + 1] = (unsigned char)((data_in.pix[i] >> 8) & 0xff);
                    irt_ofifo[irt_ofifo_wp].en[irt_ofifo_w_cnt + 1]  = 1;
                    irt_ofifo_w_cnt += 2;
                }
                pixel_cnt++;
                if (irt_ofifo_w_cnt == IRT_OFIFO_WIDTH) { // slot is finished
                    irt_ofifo_w_cnt = 0;
                    irt_ofifo_wp    = (irt_ofifo_wp + 1) % IRT_OFIFO_DEPTH;
                    fullness++;
                } else if (irt_top->irt_desc[irt_top->task[ofifo_task]].oimage_line_wr_format == 0 &&
                           pixel_cnt == irt_top->irt_desc[irt_top->task[ofifo_task]].image_par[OIMAGE].S) { // line end
                    irt_ofifo_w_cnt = 0;
                    irt_ofifo_wp    = (irt_ofifo_wp + 1) % IRT_OFIFO_DEPTH;
                    fullness++;
                }
                if (pixel_cnt == irt_top->irt_desc[irt_top->task[ofifo_task]].image_par[OIMAGE].S) { // line end
                    pixel_cnt = 0;
                }
                IRT_TRACE_TO_LOG(5,
                                 IRTLOG,
                                 "OFIFO: push %d pixels to entry %d cnt %d at cycle %d\n",
                                 pixel_valid,
                                 irt_ofifo_wp,
                                 irt_ofifo_w_cnt,
                                 cycle);
            }
        }

        if (task_end && irt_ofifo_w_cnt != 0) { // complete OFIFO slot at end of task
            for (uint8_t i = irt_ofifo_w_cnt; i < IRT_OFIFO_WIDTH; i++) {
                irt_ofifo[irt_ofifo_wp].en[irt_ofifo_w_cnt] = 0;
            }
            irt_ofifo_w_cnt = 0;
            irt_ofifo_wp    = (irt_ofifo_wp + 1) % IRT_OFIFO_DEPTH;
            fullness++;
            // IRT_TRACE("OFIFO: complete current slot at end of task at cycle %d\n");
            // exit(0);
        }
    }

    for (uint8_t i = 0; i < IRT_OFIFO_WIDTH; i++) {
        data_out.pix[i] = irt_ofifo[irt_ofifo_rp].pix[i];
        data_out.en[i]  = irt_ofifo[irt_ofifo_rp].en[i];
        // if (fullness!=0)
        // IRT_TRACE("IRT_OFIFO: irt_ofifo_rp %d, i %d, en %d %d\n", irt_ofifo_rp, i, irt_ofifo[irt_ofifo_rp].en[i],
        // data_out.en[i]);
    }

    full  = fullness >= IRT_OFIFO_DEPTH ? 1 : 0;
    empty = fullness == 0 ? 1 : 0;
}

bool IRT_top::IRT_OWM::run(bool                  ofifo_empty,
                           bool&                 ofifo_pop,
                           const bus128B_struct& data_in,
                           uint64_t&             addr,
                           bus128B_struct&       data_out,
                           uint8_t&              wr)
{

    std::string my_string = getInstanceName();
    uint8_t     done_owm;

    ofifo_pop = 0;
    wr        = 0;
    done_owm  = 0;

    irt_top->irt_top_sig.owm_wr_first = 0;
    irt_top->irt_top_sig.owm_wr_last  = 0;

    //   //---------------------------------------
    //   //wait for resamp task to complete in the given queue
    //	if (((irt_top->irt_desc[irt_top->task[e_irt_owm]].irt_mode == e_irt_resamp_fwd) ||
    //       (irt_top->irt_desc[irt_top->task[e_irt_owm]].irt_mode == e_irt_resamp_bwd2) ||
    //       (irt_top->irt_desc[irt_top->task[e_irt_owm]].irt_mode == e_irt_resamp_bwd2)) &&
    //       (irt_top->task[e_irt_owm].irt_desc_done == 0)) {
    //		return 1;
    //	}
    //   //---------------------------------------
    if (irt_top->task[e_irt_owm] >= irt_top->num_of_tasks) {
        return 1;
    }

    if (tasks_start == 0) {
        tasks_start = 1;
    }

    if (task_start == 0) {
#if 0
		IRT_TRACE("/**************************************************************************************\n");
		IRT_TRACE("IRT_OWM task %d\n", irt_top->task[e_irt_owm]);
		IRT_TRACE("/**************************************************************************************\n");

		IRT_TRACE("ADDR %llx\n", irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].addr_start);
		IRT_TRACE("W/H %d/%d\n", irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].W, irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H);
		IRT_TRACE("Xc/Yc %d/%d\n",  irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Xc, irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Yc);
		IRT_TRACE("S%d \n",   irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].S);
		IRT_TRACE("line wr format %d\n\n", irt_top->irt_desc[irt_top->task[e_irt_owm]].oimage_line_wr_format);
#endif
        task_start = 1;
    }

    // ext memory write
    if (ofifo_empty == 0) {
        ofifo_pop = 1;
        wr        = 1;
        if (irt_top->irt_desc[irt_top->task[e_irt_owm]].oimage_line_wr_format == 0) {
            addr = irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].addr_start +
                   (uint64_t)line * (uint64_t)irt_top->irt_desc[irt_top->task[e_irt_owm]]
                                        .image_par[OIMAGE]
                                        .Hs /*+ image_pars[OIMAGE].S * image_pars[OIMAGE].Ns*/
                   + (uint64_t)pixel;
        } else {
            addr = irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].addr_start + (uint64_t)pixel;
        }

        irt_top->irt_top_sig.owm_wr_first = (line == 0) && (pixel == 0);
        if (irt_top->irt_desc[irt_top->task[e_irt_owm]].oimage_line_wr_format == 0) {
            irt_top->irt_top_sig.owm_wr_last =
                (line == irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H - 1) &&
                (pixel + IRT_OFIFO_WIDTH >= ((uint32_t)irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].S
                                             << irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Ps));
        } else {
            irt_top->irt_top_sig.owm_wr_last =
                (pixel + IRT_OFIFO_WIDTH) >=
                (((uint32_t)irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].S
                  << irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Ps) *
                 (uint32_t)irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H);
        }

        // IRT_TRACE_TO_LOG(log_file, "IRT_OWM write: line %d, pixel %d, addr %x ", line, pixel, addr);
        // if (task[e_irt_owm] == 8)
        // IRT_TRACE("IRT_OWM write: pixel %d, addr %x at cycle %d\n", pixel, addr, cycle);
        uint32_t en0 = 0, en1 = 0, en2 = 0, en3 = 0;
        for (uint8_t i = 0; i < IRT_OFIFO_WIDTH; i++) {
            data_out.pix[i] = data_in.pix[i]; // ext_mem[image_pars[IIMAGE].ADDR+task*IMAGE_W*IMAGE_H + line *
                                              // image_pars[IIMAGE].W + pixel + i]; //data_in.pix[i];
            data_out.en[i] = data_in.en[i];
            if (i < 32) {
                en0 = (en0 << 1) + data_out.en[i];
            } else if (i < 64) {
                en1 = (en1 << 1) + data_out.en[i];
            } else if (i < 96) {
                en2 = (en2 << 1) + data_out.en[i];
            } else {
                en3 = (en3 << 1) + data_out.en[i];
            }
        }
        // IRT_TRACE_TO_LOG(log_file, "en: 0x_%x%x%x%x\n", en3, en2, en1, en0);
        // IRT_TRACE("ORM source addr %d\n", image_pars[IIMAGE].ADDR+task*IMAGE_W*IMAGE_H + line * image_pars[IIMAGE].W
        // + pixel);

#ifdef CREATE_IMAGE_DUMPS_OWM
        // IRT_TRACE("OWM write: task %d, line %d, pixel %d\n", irt_top->task[e_irt_owm], line, pixel);
        for (uint8_t pix = 0; pix < IRT_OFIFO_WIDTH; pix++) {
            if (irt_top->irt_desc[irt_top->task[e_irt_owm]].oimage_line_wr_format == 0) {
                owm_wr_image[irt_top->task[e_irt_owm]][line][pixel + pix] = data_out.pix[pix];
            } else {
                owm_wr_image[irt_top->task[e_irt_owm]]
                            [pixel / irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].W]
                            [(pixel % irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].W) + pix] =
                                data_out.pix[pix];
            }
        }
#endif

        pixel += IRT_OFIFO_WIDTH;

        if (irt_top->irt_desc[irt_top->task[e_irt_owm]].oimage_line_wr_format == 0) {
            if (pixel >=
                ((uint32_t)irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].S
                 << irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Ps)) { // line write is finished
                pixel = 0;
                // if (task[e_irt_owm] == 8)
                // IRT_TRACE("IRT_OWM line %d is finished at cycle %d\n", line, cycle);
                line++;
                // line_start = 1;
                if (line == irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H) { // task is finished
                    IRT_TRACE(
                        "%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[e_irt_owm], cycle);
                    irt_top->task[e_irt_owm]++;
                    //               irt_top->task[e_irt_owm].irt_desc_done == 1
                    irt_top->irt_top_sig.dv_owm_task_done = 1;
                    line                                  = 0;
                    task_start                            = 0;

                    if (irt_top->task[e_irt_owm] == irt_top->num_of_tasks) {
                        done_owm = 1;
                    }
                }
            }
        } else {
            if (pixel >= (((uint32_t)irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].S *
                           irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H)
                          << irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].Ps)) { // task is finished
                IRT_TRACE("%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[e_irt_owm], cycle);
                irt_top->task[e_irt_owm]++;
                //            irt_top->task[e_irt_owm].irt_desc_done == 1
                irt_top->irt_top_sig.dv_owm_task_done = 1;
                pixel                                 = 0;
                task_start                            = 0;

                if (irt_top->task[e_irt_owm] == irt_top->num_of_tasks) {
                    done_owm = 1;
                }
            }
        }
    }

#ifdef CREATE_IMAGE_DUMPS_OWM
    if (irt_top->task[e_irt_owm] == irt_top->num_of_tasks) {

        char owm_out_file_name[150];

        // if(irm_out_file == NULL)
        //	throw "Argument Exception";
        sprintf(owm_out_file_name, "irt_owm_out.bmp");

        IRT_TRACE("IRT OWM output file: Width[%d], Height[%d], Size[%d]\n",
                  irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].W,
                  irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H,
                  54 + 3 * irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].W *
                           irt_top->irt_desc[irt_top->task[e_irt_owm]].image_par[OIMAGE].H);
        // IRT_TRACE("Generating OWM out file width %d, height %d\n", width, height);

        WriteBMP(irt_top,
                 0,
                 3,
                 owm_out_file_name,
                 irt_top->irt_desc[irt_top->task[e_irt_owm] - 1].image_par[OIMAGE].W,
                 irt_top->irt_desc[irt_top->task[e_irt_owm] - 1].image_par[OIMAGE].H,
                 owm_wr_image);
    }
#endif

    return done_owm;
}
//---------------------------------------
// one dimensional memory allocation.
float* IRT_top::IRT_RESAMP::mycalloc1(int num_img)
{
    float* m1_out;
    m1_out = (float*)calloc(num_img, sizeof(float));
    return (m1_out);
}
//---------------------------------------
// three dimensional memory allocation.
int*** IRT_top::IRT_RESAMP::mycalloc3_int(int num_img, int num_ifms, int ifm_rows)
{
    int    k, l;
    int*** m3_out;

    m3_out = (int***)calloc(num_img, sizeof(int**));

    for (k = 0; k < num_img; k++) {
        m3_out[k] = (int**)calloc(num_ifms, sizeof(int*));
        for (l = 0; l < num_ifms; l++) {
            m3_out[k][l] = (int*)calloc(ifm_rows, sizeof(int));
        }
    }
    return (m3_out);
}

void IRT_top::IRT_RESAMP::myfree3_int(int*** m3_out, int num_img, int num_ifms, int ifm_rows)
{
    int k, l;
    for (k = 0; k < num_img; k++) {
        for (l = 0; l < num_ifms; l++) {
            free(m3_out[k][l]);
        }
        free(m3_out[k]);
    }
    free(m3_out);
}

//---------------------------------------
// match the queue
int32_t IRT_top::IRT_RESAMP::matchq(queue<queue<uint32_t>> mq, int32_t*** bli_win2, int32_t id, int32_t& pos)
{
    int   match = 0;
    int   sc_psize;

    int*** bli_win1;
    pos      = 0;
    bli_win1 = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);

    queue<queue<uint32_t>> m = mq;
    while (!m.empty()) {
        // while (m.size()>1) {
        queue<uint32_t> tmp = m.front();
        sc_psize            = (int)floor((tmp.size() / 8));
        for (int ti = 0; ti < sc_psize; ti++) {
            for (int si = 0; si < 4; si++) {
                bli_win1[ti][si][0] = tmp.front();
                tmp.pop();
                bli_win1[ti][si][1] = tmp.front();
                tmp.pop();
            }
        }
        for (int i = 0; i < sc_psize; i++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++) {
                    // if((bli_win1[i][k][0]==bli_win2[//id][l][0])&(bli_win1[i][k][1]==bli_win2[id][l][1]))
                    if ((bli_win1[i][k][0] == bli_win2[id][l][0]) &
                        ((bli_win1[i][k][1] >> 1) == (bli_win2[id][l][1] >> 1))) {
                        match = 1;
                        myfree3_int(bli_win1, IRT_ROT_MAX_PROC_SIZE, 4, 2);
                        return match;
                    }
                }
            }
        }
        m.pop();
        pos = pos + 1;
    }
    myfree3_int(bli_win1, IRT_ROT_MAX_PROC_SIZE, 4, 2);
    // printf("strange : c_o : %d tag: %d match: %d pos : %d size : %d \n " , c_o,tag,match,pos1,m.size());
    return match;
}

//---------------------------------------
// match the 2 bli windows
int32_t IRT_top::IRT_RESAMP::match_bli_win(int32_t*** bli_win1,
                                           int32_t*** bli_win2,
                                           int32_t    bli1_size,
                                           int32_t    bli2_size,
                                           int32_t    data_update_en)
{
    int match    = 0;
    int most_neg = -32768;
    int no_n_m_1 = 0;
    for (int i = 0; i < bli1_size; i++) {
        for (int j = 0; j < bli2_size; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++) {
                    // if((bli_win1[i][k][0]==bli_win2[j][l][0])&(bli_win1[i][k][1]==bli_win2[j][l][1]))
                    if ((bli_win1[i][k][0] == bli_win2[j][l][0]) &
                        ((bli_win1[i][k][1] >> 1) == (bli_win2[j][l][1] >> 1))) {
                        match = match + 1;
                        if (data_update_en == 1) {
                            bli_win2[j][l][0] = most_neg;
                            bli_win2[j][l][1] = most_neg;
                        }
                        if ((bli_win1[i][k][1] & 0x1) != (bli_win2[j][l][1] & 0x1)) {
                            no_n_m_1 = 1;
                        }
                    }
                }
            }
        }
    }
    if (match > 0)
        match = (no_n_m_1 == 1) ? 2 : 1;
    return match;
}
//---------------------------------------
void IRT_top::IRT_RESAMP::copy_bli_win(int*** bli_win1, int*** bli_win2, int bli1_size)
{
    for (int qi3 = 0; qi3 < bli1_size; qi3++) {
        for (int qi4 = 0; qi4 < 4; qi4++) {
            bli_win2[qi3][qi4][0] = bli_win1[qi3][qi4][0];
            bli_win2[qi3][qi4][1] = bli_win1[qi3][qi4][1];
        }
    }
}
//---------------------------------------
int32_t IRT_top::IRT_RESAMP::bank_row_count(int32_t*** bli_win, int32_t bli_size)
{
    int bank_row_dist[IRT_ROT_MEM_ROW_BANKS];
    int bank_row_cnt = 0;

    for (int h = 0; h < IRT_ROT_MEM_ROW_BANKS; h++) {
        bank_row_dist[h] = 0;
    }

    for (int i = 0; i < bli_size; i++) {
        for (int j = 0; j < 4; j++) {
            bank_row_dist[bli_win[i][j][0] % 8] = 1;
        }
    }

    for (int k = 0; k < IRT_ROT_MEM_ROW_BANKS; k++) {
        bank_row_cnt = bank_row_cnt + bank_row_dist[k];
        // printf(" %d   ",bank_row_dist[k] );
    }
    // printf("\n");
    // printf("DUMP data2 %d \n ",bank_row_cnt);

    return bank_row_cnt;
}
//---------------------------------------
void IRT_top::IRT_RESAMP::coord_to_2x2_window(float data[2], int32_t*** bli_win, int32_t pid)
{
    bli_win[pid][0][0] = (int)floor(data[0]);
    bli_win[pid][0][1] = (int)floor(data[1]);
    bli_win[pid][1][0] = (int)floor(data[0]);
    bli_win[pid][1][1] = (int)ceil(data[1]);
    bli_win[pid][2][0] = (int)ceil(data[0]);
    bli_win[pid][2][1] = (int)floor(data[1]);
    bli_win[pid][3][0] = (int)ceil(data[0]);
    bli_win[pid][3][1] = (int)ceil(data[1]);
}
//---------------------------------------
/*
void IRT_top::IRT_RESAMP::warp_shake(int stride)
{
  int no_x_seg1 = (int)ceil((float)m_warp.w/(float)STRIPE_SIZE);
  int last_x_seg1 = m_warp.w - ((no_x_seg1-1)*(int)STRIPE_SIZE);
  int stripe_w1=(int)STRIPE_SIZE;
  float tmp_row[STRIPE_SIZE][2];
  int num_elm;
  int base;

  for(int i=0;i<BATCH_SIZE;i++)
  {
    for(int segment=0; segment<no_x_seg1; segment++)
    {
     stripe_w1 = (segment==(no_x_seg1-1))? last_x_seg1 : (int)STRIPE_SIZE;
     for(int j=0;j<m_warp.h;j++)
     {
       base=0;
       for(int k=0;k<stride; k++)
       {
         num_elm= (int)floor(stripe_w1/stride)+ (((stripe_w1%stride)>k)?1:0);
         for(int l=0;l< num_elm;l++)
         {
           tmp_row[base+l][0]=m_warp.data[i][j][(segment*STRIPE_SIZE)+(k+(l*stride))][0];
           tmp_row[base+l][1]=m_warp.data[i][j][(segment*STRIPE_SIZE)+(k+(l*stride))][1];
           //if(j==0)
           //  printf("(%0.2f %0.2f), (%d, %d , %d stripew %d , stride
%d)",m_warp.data[i][j][(segment*STRIPE_SIZE)+(k+(l*stride))][0],m_warp.data[i][j][(segment*STRIPE_SIZE)+(k+(l*stride))][1],((segment*STRIPE_SIZE)+(k+(l*stride))),base,num_elm,stripe_w1,stride);
         }
         base=base+num_elm;
       }
       //copy back shuffled warp data.
       for(int m=0; m<stripe_w1; m++)
       {
         m_warp.data[i][j][(segment*STRIPE_SIZE)+m][0] = tmp_row[m][0];
         m_warp.data[i][j][(segment*STRIPE_SIZE)+m][1] = tmp_row[m][1];
         //if(j==0)
         //  printf("(%0.2f %0.2f) ",tmp_row[m][0],tmp_row[m][1]);
       }
       //if(j==0)
       //  printf("\n");
     }
    }
  }
}
  */
//---------------------------------------
// Check resampler descriptor for correct range of all fields
void IRT_top::IRT_RESAMP::resamp_descr_checks(irt_desc_par& descr)
{
    // Error if called for non-resamp tasks
    if ((descr.irt_mode != e_irt_resamp_fwd) && (descr.irt_mode != e_irt_resamp_bwd1) &&
        (descr.irt_mode != e_irt_resamp_bwd2) && (descr.irt_mode != e_irt_rescale)) {
        IRT_TRACE("Error-: IRT_RESAMP::run() called for non-resamp tasks irt_mode %d\n", uint8_t(descr.irt_mode));
        IRT_TRACE_TO_RES(test_res,
                         "Error-: IRT_RESAMP::run() called for non-resamp tasks : %d cycles=%d\n",
                         uint8_t(descr.irt_mode),
                         cycle);
        IRT_CLOSE_FAILED_TEST(0);
    }
    //---------------------------------------
    if (((descr.image_par[MIMAGE].W != descr.image_par[OIMAGE].W) ||
         (descr.image_par[MIMAGE].H != descr.image_par[OIMAGE].H) || (descr.image_par[MIMAGE].W > IRT_MAX_IMAGE_W) ||
         (descr.image_par[MIMAGE].H > 8192 /*IRT_MAX_IMAGE_H*/)) &&
        (descr.irt_mode != e_irt_rescale)) {
        IRT_TRACE("Error - %s: RESAMP TASKS WARP_IMAGE & OUT IMAGE DIMENSION NOT SAME OR WARP image out of range\n",
                  my_string.c_str());
        IRT_TRACE_TO_RES(
            test_res, "Error - %s: RESAMP TASKS WARP_IMAGE & OUT IMAGE DIMENSION NOT SAME\n", my_string.c_str());
        IRT_CLOSE_FAILED_TEST(0);
    }
    //---------------------------------------
    // warp_stride range check
    if (descr.warp_stride != 0 && descr.warp_stride != 1 && descr.warp_stride != 3) {
        IRT_TRACE("Error - %s: RESAMP WARP_STRIDE > 4 NOT RECOMMENDED \n", my_string.c_str());
        IRT_TRACE_TO_RES(test_res, "Error - %s: RESAMP WARP_STRIDE > 8 NOT RECOMMENDED\n", my_string.c_str());
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (descr.irt_mode <= 3) {
        descr.image_par[OIMAGE].Ps = descr.image_par[OIMAGE].DataType <= 8 ? 0 : 1;
    } else {
        switch (descr.image_par[OIMAGE].DataType) {
            case e_irt_int8: descr.image_par[OIMAGE].Ps = 0; break;
            case e_irt_int16: descr.image_par[OIMAGE].Ps = 1; break;
            case e_irt_fp16: descr.image_par[OIMAGE].Ps = 1; break;
            case e_irt_bfp16: descr.image_par[OIMAGE].Ps = 1; break;
            case e_irt_fp32: descr.image_par[OIMAGE].Ps = 2; break;
        }
    }
    if (descr.irt_mode == e_irt_resamp_bwd1) { // BWD1 has 2 elements per output pix
        descr.image_par[OIMAGE].Ps++;
    }

    if (descr.irt_mode <= 3) {
        descr.image_par[IIMAGE].Ps = descr.image_par[IIMAGE].DataType <= 8 ? 0 : 1;
    } else {
        switch (descr.image_par[IIMAGE].DataType) {
            case e_irt_int8: descr.image_par[IIMAGE].Ps = 0; break;
            case e_irt_int16: descr.image_par[IIMAGE].Ps = 1; break;
            case e_irt_fp16: descr.image_par[IIMAGE].Ps = 1; break;
            case e_irt_bfp16: descr.image_par[IIMAGE].Ps = 1; break;
            case e_irt_fp32: descr.image_par[IIMAGE].Ps = 2; break;
        }
    }

    if (descr.irt_mode <= 3) {
        descr.image_par[MIMAGE].Ps = descr.mesh_format == e_irt_mesh_flex ? 2 : 3;
    } else {
        switch (descr.image_par[MIMAGE].DataType) {
            case e_irt_int8: descr.image_par[MIMAGE].Ps = 1; break;
            case e_irt_int16: descr.image_par[MIMAGE].Ps = 2; break;
            case e_irt_fp16: descr.image_par[MIMAGE].Ps = 2; break;
            case e_irt_bfp16: descr.image_par[MIMAGE].Ps = 2; break;
            case e_irt_fp32: descr.image_par[MIMAGE].Ps = 3; break;
        }
        if (descr.irt_mode == e_irt_rescale) {
            descr.image_par[MIMAGE].Ps--;
            // IRT_TRACE_UTILS("->>>>inside rescale Ps %d\n",descr.image_par[MIMAGE].Ps);
        }
    }
    if (descr.irt_mode > 3) {
        switch (descr.image_par[GIMAGE].DataType) {
            case e_irt_int8: descr.image_par[GIMAGE].Ps = 0; break;
            case e_irt_int16: descr.image_par[GIMAGE].Ps = 1; break;
            case e_irt_fp16: descr.image_par[GIMAGE].Ps = 1; break;
            case e_irt_bfp16: descr.image_par[GIMAGE].Ps = 1; break;
            case e_irt_fp32: descr.image_par[GIMAGE].Ps = 2; break;
        }
    }
    //---------------------------------------
    descr.image_par[IIMAGE].PsBytes = (1 << descr.image_par[IIMAGE].Ps);
    descr.image_par[OIMAGE].PsBytes = (1 << descr.image_par[OIMAGE].Ps);
    descr.image_par[MIMAGE].PsBytes = (1 << descr.image_par[MIMAGE].Ps);
    descr.image_par[GIMAGE].PsBytes = (1 << descr.image_par[GIMAGE].Ps);
    //----
}

uint8_t IRT_top::IRT_RESAMP::no_conflict_check(int16_t  x1,
                                               int16_t  y1,
                                               int16_t  x1_p1,
                                               int16_t  y1_p1,
                                               int16_t  x2,
                                               int16_t  y2,
                                               int16_t  x2_p1,
                                               int16_t  y2_p1,
                                               uint8_t  sc_en,
                                               uint16_t in_data_type,
                                               bool     bwd2_en)
{
    // uint16_t in_data_type = descr.image_par[IIMAGE].Ps;
    // bool bwd2_en = (descr.irt_mode==6) ? 1 :0 ;
    int16_t  num_bank_row_m1         = (!bwd2_en) ? int16_t(NUM_BANK_ROW - 1) : int16_t((NUM_BANK_ROW / 2) - 1);
    uint8_t  no_conflict             = 0;
    int16_t  y2_m_y1_olp             = (y2 - y1 == 0) | (y2 - y1_p1 == 0) | (y2_p1 - y1 == 0);
    int16_t  x2_m_x1                 = (x2 >= x1) ? abs(x2_p1 - x1) : abs(x1_p1 - x2);
    int16_t  x1_p_1                  = x1_p1;
    int16_t  x2_p_1                  = x2_p1;
    int16_t  x1_frac_not_0           = ((x1_p1 - x1) == 0) ? 0 : 1;
    int16_t  x2_frac_not_0           = ((x2_p1 - x2) == 0) ? 0 : 1;
    int16_t  diff_for_space_conflict = (x1_frac_not_0 & x2_frac_not_0) ? 2 : ((x1_frac_not_0 | x2_frac_not_0) ? 1 : 0);
    uint16_t rshft_val               = ((in_data_type == 2) & (!bwd2_en)) ? 2 : 3;
    uint16_t bit_pos                 = ((in_data_type == 2) & (!bwd2_en)) ? 0x0004 : 0x0008;
    uint8_t  bc_x1                   = (((x1 & bit_pos) >> rshft_val) << 1) | ((x1_p_1 & bit_pos) >> rshft_val);
    uint8_t  bc_x2                   = (((x2 & bit_pos) >> rshft_val) << 1) | ((x2_p_1 & bit_pos) >> rshft_val);

    uint16_t bit_pos2      = ((in_data_type == 2) & (!bwd2_en)) ? 0x0020 : 0x0040;
    uint32_t same_cl_x1    = (x1 & bit_pos2);
    uint32_t same_cl_x1_p1 = (x1_p1 & bit_pos2);
    uint32_t same_cl_x2    = (x2 & bit_pos2);
    uint32_t same_cl_x2_p1 = (x2_p1 & bit_pos2);
    uint8_t  same_cl =
        ((same_cl_x1 == same_cl_x1_p1) & (same_cl_x2 == same_cl_x2_p1)) ? ((same_cl_x1 == same_cl_x2) ? 1 : 0) : 1;
    uint8_t bc = (bc_x1 << 2) | bc_x2;
    bool    c1 = ((bc == 0) | (bc == 15)) ? 1 : 0;
    bool    c2 =
        ((bc == 1) | (bc == 2) | (bc == 4) | (bc == 7) | (bc == 8) | (bc == 11) | (bc == 13) | (bc == 14)) ? 1 : 0;
    bool     c3      = ((bc == 3) | (bc == 12)) ? 1 : 0;
    bool     c4      = ((bc == 5) | (bc == 10)) ? 1 : 0;
    bool     c5      = ((bc == 6) | (bc == 9)) ? 1 : 0;
    bool     c6      = (sc_en == 0) ? 1 : ((x2_m_x1 > diff_for_space_conflict) ? 1 : 0);
    bool     rc1     = ((abs(y2 - y1) & num_bank_row_m1) == 0);
    bool     rc2     = (((abs(y2 - y1_p1)) & num_bank_row_m1) == 0);
    bool     rc3     = (((abs(y2_p1 - y1)) & num_bank_row_m1) == 0);
    uint16_t c1_dist = ((in_data_type == 2) & (!bwd2_en)) ? 4 : 8; // fp32 or others.
    uint16_t c2_dist = ((in_data_type == 2) & (!bwd2_en)) ? 5 : 9;
    uint16_t c3_dist = ((in_data_type == 2) & (!bwd2_en)) ? 8 : 16;
    uint16_t c4_dist = 2;
    if (y2_m_y1_olp) {
        if (c1 & (x2_m_x1 < c1_dist) & c6)
            no_conflict = same_cl;
        else if (c2 & (x2_m_x1 < c2_dist) & c6)
            no_conflict = same_cl;
        else if (c3 & (x2_m_x1 < c3_dist) & c6)
            no_conflict = same_cl;
        else if (c4 & (x2_m_x1 < c4_dist) & c6)
            no_conflict = same_cl;
        else if (c5)
            no_conflict = 0;
        else
            no_conflict = 0;
    } else if (rc1 | rc2 | rc3) {
        no_conflict = 0;
    } else {
        no_conflict = 1;
    }
    //  IRT_TRACE_TO_LOG(2,parseFile,"conflict_check::[Y:X]=[%0d:%0d][%0d:%0d] no_conflict=%0d bc=%0d
    //  C[1:6]=[%0d:%0d:%0d:%0d:%0d:%0d] RC[%0d:%0d:%0d]\n",y1,x1,y2,x2,no_conflict,bc,c1,c2,c3,c4,c5,c6,rc1,rc2,rc3);

    return no_conflict;
}

/*
#ifdef RUN_WITH_SV
//---------------------------------------
uint8_t IRT_top::IRT_RESAMP::fv_proc_size_calc(uint8_t tag_vld, float Xf[IRT_ROT_MAX_PROC_SIZE], float
Xc[IRT_ROT_MAX_PROC_SIZE], float Yf[IRT_ROT_MAX_PROC_SIZE], float Yc[IRT_ROT_MAX_PROC_SIZE], bool bwd_pass2_en,uint16_t
in_data_type) { uint8_t proc_size=1;
   //resolve bank and space conflicts
   uint8_t num_elements=0;
   for(uint8_t i=0; i<IRT_ROT_MAX_PROC_SIZE; i++) {
      num_elements += ((tag_vld>>i)&1);
   }
   IRT_TRACE("fv_proc_size_calc tag_vld %d num_elements = %d\n",tag_vld,num_elements);
   for(uint32_t i=0 ; i<num_elements-1 ; i++) {
      for(uint32_t j=0; j< (i+1) ; j++) {
         int16_t x1 = Xf[j]; //bli_win[j][0][1];
         int16_t y1 = Yf[j]; //bli_win[j][0][0];
         int16_t x1_p1 = Xc[j];//bli_win[j][3][1];
         int16_t y1_p1 = Yc[j];//bli_win[j][3][0];
         int16_t x2 = Xf[i+1];//bli_win[i+1][0][1];
         int16_t y2 = Yf[i+1];//bli_win[i+1][0][0];
         int16_t x2_p1 = Xc[i+1];//bli_win[i+1][3][1];
         int16_t y2_p1 = Yc[i+1];//bli_win[i+1][3][0];
         uint8_t no_conflict =
no_conflict_check(x1,y1,x1_p1,y1_p1,x2,y2,x2_p1,y2_p1,bwd_pass2_en,in_data_type,bwd_pass2_en); if(no_conflict == 0)
            return proc_size;
      }
      proc_size = proc_size+1;
   }
   return proc_size;
}
#endif
*/

//---------------------------------------
uint8_t IRT_top::IRT_RESAMP::proc_size_calc(float    Xi[IRT_ROT_MAX_PROC_SIZE],
                                            float    Yi[IRT_ROT_MAX_PROC_SIZE],
                                            uint8_t  num_elements,
                                            uint8_t  sc_en,
                                            uint16_t in_data_type,
                                            bool     bwd2_en)
{
    float   sc_coord[2];
    int***  bli_win;
    uint8_t proc_size = 1;
#ifdef RUN_WITH_SV
    uint8_t fv_tag_vld = 0;
    float   fv_Xf[IRT_ROT_MAX_PROC_SIZE], fv_Xc[IRT_ROT_MAX_PROC_SIZE];
    float   fv_Yf[IRT_ROT_MAX_PROC_SIZE], fv_Yc[IRT_ROT_MAX_PROC_SIZE];
    uint8_t fv_psize;
#endif
    bli_win = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);
    for (int si = 0; si < num_elements; si++) {
        sc_coord[0] = Yi[si];
        sc_coord[1] = Xi[si];
        coord_to_2x2_window(sc_coord, bli_win, si);
#ifdef RUN_WITH_SV
        //---------------------------------------
        // FV array preparation
        fv_tag_vld = (fv_tag_vld << 1) | 1;
        fv_Xf[si]  = bli_win[si][0][1];
        fv_Yf[si]  = bli_win[si][0][0];
        fv_Xc[si]  = bli_win[si][3][1];
        fv_Yc[si]  = bli_win[si][3][0];
#endif
    }
    //#ifdef RUN_WITH_SV
    // fv_psize = fv_proc_size_calc(fv_tag_vld,fv_Xf,fv_Xc,fv_Yf,fv_Yc,sc_en,in_data_type);
    // IRT_TRACE("fv_proc_size_calc num_elements %d fv_tag_vld %x fv_psize = %d\n",num_elements,fv_tag_vld,fv_psize);
    //#endif
    // resolve bank and space conflicts
    for (uint8_t i = 0; i < num_elements - 1; i++) {
        for (uint8_t j = 0; j < (i + 1); j++) {
            int16_t x1    = bli_win[j][0][1];
            int16_t y1    = bli_win[j][0][0];
            int16_t x1_p1 = bli_win[j][3][1];
            int16_t y1_p1 = bli_win[j][3][0];
            int16_t x2    = bli_win[i + 1][0][1];
            int16_t y2    = bli_win[i + 1][0][0];
            int16_t x2_p1 = bli_win[i + 1][3][1];
            int16_t y2_p1 = bli_win[i + 1][3][0];
            uint8_t no_conflict =
                no_conflict_check(x1, y1, x1_p1, y1_p1, x2, y2, x2_p1, y2_p1, sc_en, in_data_type, bwd2_en);
            if (no_conflict == 0) {
                myfree3_int(bli_win, IRT_ROT_MAX_PROC_SIZE, 4, 2);
                return proc_size;
            }
        }
        proc_size = proc_size + 1;
    }
    myfree3_int(bli_win, IRT_ROT_MAX_PROC_SIZE, 4, 2);
    return proc_size;
}

// two dimensional memory allocation.
int** IRT_top::IRT_RESAMP::mycalloc2_int(int num_img, int num_ifms)
{
    int   j;
    int** m2_out;
    m2_out = (int**)calloc(num_img, sizeof(int*));
    for (j = 0; j < num_img; j++) {
        m2_out[j] = (int*)calloc(num_ifms, sizeof(int));
    }
    return (m2_out);
}

void IRT_top::IRT_RESAMP::myfree2_int(int** m2_out, int num_img, int num_ifms)
{
    int k;
    for (k = 0; k < num_img; k++) {
        free(m2_out[k]);
    }
    free(m2_out);
}

//---------------------------------------
uint16_t IRT_top::IRT_RESAMP::get_num_segments(uint32_t width)
{
    uint16_t num_segments = (int)ceil((float)width / (float)RESAMP_SEGMENT_SIZE);
    return num_segments;
}
//---------------------------------------
uint32_t IRT_top::IRT_RESAMP::check_nan(uint32_t pix, uint8_t dtype)
{
    uint32_t opix = pix;
    if ((dtype == 2) && (gaudi3::is_nan_fp16(pix & 0xFFFF))) {
        opix = gaudi3::DEFAULT_NAN_FP16;
    }
    if ((dtype == 3) && (gaudi3::is_nan_bfp16(pix & 0xFFFF))) {
        opix = gaudi3::DEFAULT_NAN_BFP16;
    }
    if ((dtype == 4) && (gaudi3::is_nan_fp32(pix))) {
        opix = gaudi3::DEFAULT_NAN_FP32;
    }
    return opix;
}
//---------------------------------------
//---------------------------------------
// this function is hardcide for to align with RTL, in RTL warp coord will be conveted
// to Fixed point for further processing S.0.31 format. as part of that conversion +/-Inf
// will be saturated and NAN will be converted to zero. in model to avoid multiple changes
// this function just checks for NAN or Inf and converts sat or zero to align with Inf.
// if there is no NAN/Inf model already aligns with RTL.
float IRT_top::IRT_RESAMP::fp32_nan_inf_conv(float pix)
{
    uint32_t pix_bit;
    uint32_t opix_bit;
    float    opix = pix;
    memcpy(&pix_bit, &pix, sizeof(float));
    if (gaudi3::is_nan_fp32(pix_bit)) { // nan check
        opix_bit = 0x0;
        memcpy(&opix, &opix_bit, sizeof(uint32_t));
        irt_top->irt_top_sig.num_err.coord_nan_err |= 1;
        // IRT_TRACE("is_nan_fp32 ");
    }
    if (gaudi3::is_inf_fp32(pix_bit)) {
        if ((pix_bit & 0x80000000) == 0) { //+inf
            opix_bit = 0x7f7fffff;
            irt_top->irt_top_sig.num_err.coord_pinf_err |= 1;
        } else { //-inf
            opix_bit = 0xff7fffff;
            irt_top->irt_top_sig.num_err.coord_ninf_err |= 1;
        }
        memcpy(&opix, &opix_bit, sizeof(uint32_t));
        // IRT_TRACE("is_inf_fp32 ");
    }
    // IRT_TRACE("fp32_nan_inf_conv: pix %f pix_bit %x opix_bit %x opix %f \n",pix,pix_bit,opix_bit,opix);
    return opix;
}
// caller = 1 [compute]
void IRT_top::IRT_RESAMP::check_num_err(uint32_t pix, uint8_t dtype, uint8_t caller)
{
    if (caller == 1) {
        if (dtype == 2) {
            if (gaudi3::is_nan_fp16(pix & 0xFFFF)) {
                irt_top->irt_top_sig.num_err.rinterp_nan_err |= 1;
            }
            if (gaudi3::is_inf_fp16(pix & 0xFFFF)) {
                if ((pix & 0x8000) == 0)
                    irt_top->irt_top_sig.num_err.rinterp_pinf_err |= 1;
                else
                    irt_top->irt_top_sig.num_err.rinterp_ninf_err |= 1;
            }
        }
        if (dtype == 3) {
            if (gaudi3::is_nan_bfp16(pix & 0xFFFF)) {
                irt_top->irt_top_sig.num_err.rinterp_nan_err |= 1;
            }
            if (gaudi3::is_inf_bfp16(pix & 0xFFFF)) {
                if ((pix & 0x8000) == 0)
                    irt_top->irt_top_sig.num_err.rinterp_pinf_err |= 1;
                else
                    irt_top->irt_top_sig.num_err.rinterp_ninf_err |= 1;
            }
        }
        if (dtype == 4) {
            if (gaudi3::is_nan_fp32(pix)) {
                irt_top->irt_top_sig.num_err.rinterp_nan_err |= 1;
            }
            if (gaudi3::is_inf_fp32(pix)) {
                if ((pix & 0x80000000) == 0)
                    irt_top->irt_top_sig.num_err.rinterp_pinf_err |= 1;
                else
                    irt_top->irt_top_sig.num_err.rinterp_ninf_err |= 1;
            }
        }
    }
    // IRT_TRACE("check_num_err: dtype %d pix %x rint_err[NAN=%d pinf=%d ninf=%d] @cycle %d \n",pix,dtype,
    //      irt_top->irt_top_sig.num_err.rinterp_nan_err,irt_top->irt_top_sig.num_err.rinterp_pinf_err,irt_top->irt_top_sig.num_err.rinterp_ninf_err,cycle);
}
////---------------------------------------
// uint32_t IRT_top::IRT_RESAMP::check_nan(uint32_t pix, uint8_t dtype){
//   uint32_t opix = pix;
//   switch(dtype){
//      case 2: opix = (gaudi3::is_nan_fp16(pix & 0xFFFF)  ? gaudi3::DEFAULT_NAN_FP16  : opix); break;
//      case 3: opix = (gaudi3::is_nan_bfp16(pix & 0xFFFF) ? gaudi3::DEFAULT_NAN_BFP16 : opix); break;
//      case 4: opix = (gaudi3::is_nan_fp32(pix)           ? gaudi3::DEFAULT_NAN_FP32  : opix); break;
//   }
//   return opix;
//}

//---------------------------------------
// call malloc/calloc for all structs used in resamp call
void IRT_top::IRT_RESAMP::resamp_mem_alloc_and_init(irt_desc_par descr)
{
    //---------------------------------------
    // total stripes for complete output image
    no_x_seg = (int)ceil((float)descr.image_par[OIMAGE].W / (float)descr.image_par[OIMAGE].S);
    //---------------------------------------
    c_bli_win = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);
    p_bli_win = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);
    tag_a     = mycalloc2_int(4, 2);
    //---------------------------------------
    // grab local params from descriptor struct for code concise
    // m_warp.w = descr.image_par[MIMAGE].W;
    // m_warp.h = descr.image_par[MIMAGE].H;

    // input_im.w = descr.image_par[IIMAGE].W;
    // input_im.h = descr.image_par[IIMAGE].H;
    //---------------------------------------
    time_conflict_en  = (descr.irt_mode == e_irt_resamp_bwd1) || (descr.irt_mode == e_irt_resamp_bwd2);
    space_conflict_en = (descr.irt_mode == e_irt_resamp_bwd2);
    //---------------------------------------
    // resampler parser global variables
    memset(&w_parser_cnt, 0, sizeof(w_parser_cnt));
    memset(&wr_atrc, 0, sizeof(wr_atrc));
    memset(&wr_dtrc, 0, sizeof(wr_dtrc));
    memset(&gr_atrc, 0, sizeof(gr_atrc));
    memset(&gr_dtrc, 0, sizeof(gr_dtrc));
    memset(&comp_parser_cnt, 0, sizeof(comp_parser_cnt));
    memset(&comp_rescale_cnt, 0, sizeof(comp_rescale_cnt));
    memset(&rescale_v_accum, 0, sizeof(rescale_v_accum));
    memset(&rescale_h_accum, 0, sizeof(rescale_h_accum));
    memset(&irt_top->mem_ctrl[0], 0, sizeof(irt_mem_ctrl_struct)); // used for warp read
    // irt_top->mem_ctrl[0].last_line[0] = -1;
    //---------------------------------------
    avg_proc_size          = 0;
    way_cnt                = 0;
    set_cnt                = 0;
    proc_comp_cnt          = 0;
    curr_compute_cnt       = 0;
    parser_only_done       = 0;
    parser_and_imread_done = 0;
    num_req                = 0;
    p_bli_v                = 0;
    prev_proc_size         = 0;
    num_victims            = 0;
    tc1_stall              = 0;
    num_tc_stall_cycle     = 0;
    num_nm1_conflicts      = 0;
    num_nm2_conflicts      = 0;
    num_stall_misq         = 0;
    num_same_cl            = 0;

    two_lookup_cycles   = 0;
    first_vld_warp_line = 0;
    // end of resampler global variables.

    memset(&owm_tracker, 0, sizeof(owm_tracker));

    if (descr.irt_mode != e_irt_rescale) {
        // TODO - Should we reset whole CACHE ??
        for (uint8_t s = 0; s < NSETS; s++) {
            for (uint8_t w = 0; w < NWAYS; w++) {
                cache_tag_array.valid[s][w]       = 0;
                cache_tag_array.pagenumbers[s][w] = 0;
                if (w < NWAYS - 1)
                    cache_tag_array.tree_bits[s][w] = 0;
            }
        }
        for (uint8_t s = 0; s < NSETS; s++) {
            for (uint8_t w = 0; w < NWAYS; w++) {
                curr_tag_array.valid[s][w]       = 0;
                curr_tag_array.pagenumbers[s][w] = 0;
                if (w < NWAYS - 1)
                    curr_tag_array.tree_bits[s][w] = 0;
            }
        }
        //---------------------------------------
        resamp_rot_mem = (bus128B_struct**)calloc(NSETS, sizeof(bus128B_struct*));
        for (uint8_t i = 0; i < NSETS; i++) {
            resamp_rot_mem[i] = (bus128B_struct*)calloc(NWAYS, sizeof(bus128B_struct));
        }
        //---------------------------------------
        // ALLOCATE WARP IMAGE ARRAY
        // no_x_seg - in terms of elements
        resamp_gimage = new warp_line*[no_x_seg];
        for (int i = 0; i < no_x_seg; i++) {
            (resamp_gimage)[i] = new warp_line[descr.image_par[GIMAGE].H];
            for (int j = 0; j < descr.image_par[GIMAGE].H; j++) {
                resamp_gimage[i][j].x = (float*)calloc(descr.image_par[GIMAGE].S, sizeof(float));
                resamp_gimage[i][j].y = (float*)calloc(descr.image_par[GIMAGE].S, sizeof(float));
            }
        }
        //---------------------------------------
        // ALLOCATE WARP IMAGE ARRAY
        // no_x_seg - in terms of elements
        resamp_wimage = new warp_line*[no_x_seg];
        for (int i = 0; i < no_x_seg; i++) {
            (resamp_wimage)[i] = new warp_line[descr.image_par[MIMAGE].H];
            for (int j = 0; j < descr.image_par[MIMAGE].H; j++) {
                resamp_wimage[i][j].x = (float*)calloc(descr.image_par[MIMAGE].S, sizeof(float));
                resamp_wimage[i][j].y = (float*)calloc(descr.image_par[MIMAGE].S, sizeof(float));
            }
        }
    }
    if (descr.irt_mode == e_irt_rescale) {
        //---------------------------------------
        // ALLOCATE WARP IMAGE ARRAY
        // no_x_seg - in terms of elements
        int16_t Hi    = descr.Yi_end - descr.Yi_start + 1;
        resamp_wimage = new warp_line*[no_x_seg];
        for (int i = 0; i < no_x_seg; i++) {
            (resamp_wimage)[i] = new warp_line[Hi];
            for (int j = 0; j < Hi; j++) {
                resamp_wimage[i][j].x = (float*)calloc(descr.image_par[MIMAGE].S, sizeof(float));
                resamp_wimage[i][j].y = (float*)calloc(descr.image_par[MIMAGE].S, sizeof(float));
            }
        }
    }
    //---------------------------------------
    // owm_write state variable reset
    for (int ii = 0; ii < NUM_OCACHE_RESCALE; ii++) {
        olsc[ii] = 0;
        lbc[ii]  = 0;
    }
    resamp_irm_first_rd = 1;
    bank_cnt            = 0;
    set_cnt             = 0;
    way_cnt             = 0;
    owm_first_trans     = 1;
    //   IRT_TRACE("resamp_mem_alloc_and_init :: curr_compute_cnt %d called at cycle %d\n", curr_compute_cnt, cycle);
}
//---------------------------------------
// de-alloc all assigned mem/struct for resamp
void IRT_top::IRT_RESAMP::resamp_mem_dealloc(irt_desc_par descr)
{
    no_x_seg = (int)ceil((float)descr.image_par[OIMAGE].W / (float)descr.image_par[OIMAGE].S);
    // free(resamp_rot_mem);
    // free(resamp_wimage);
    //---------------------------------------
    if (descr.irt_mode != e_irt_rescale) {
        for (int row = NSETS - 1; row >= 0; row--) {
            delete[] resamp_rot_mem[row];
            resamp_rot_mem[row] = nullptr;
        }
        delete[] resamp_rot_mem;
        resamp_rot_mem = nullptr;
    }
    //---------------------------------------
    for (int i = 0; i < no_x_seg; i++) {
        // for (int j = 0; j < Hi; j++) {
        //   free(resamp_wimage[i][j]);
        //}
        delete[] resamp_wimage[i];
        resamp_wimage[i] = nullptr;
    }
    delete[] resamp_wimage;
    resamp_wimage = nullptr;
    //---------------------------------------

    // free(cache_tag_array);
}
//---------------------------------------
// shuffle a warp/grad line based on stride
void IRT_top::IRT_RESAMP::rshuffle(warp_line* wl, uint8_t wstride)
{
    uint16_t  idx = 0;
    warp_line wli;
    wli.x = (float*)calloc(wl->size, sizeof(float));
    wli.y = (float*)calloc(wl->size, sizeof(float));
    //---------------------------------------
    // IRT_TRACE("before::wstride=%d wl.x = ",wstride);
    // for(int k=0; k<wl->size; k++) {
    //   IRT_TRACE("%f,",wl->x[k]);
    //}
    // IRT_TRACE("\n");
    // IRT_TRACE("before::wl.y = ");
    // for(int k=0; k<wl->size; k++) {
    //   IRT_TRACE("%f,",wl->y[k]);
    //}
    // IRT_TRACE("\n");
    //---------------------------------------
    for (int k = 0; k < wstride; k++) {
        uint8_t num_elm = (int)floor(wl->size / wstride) + (((wl->size % wstride) > k) ? 1 : 0);
        for (int l = 0; l < num_elm; l++) {
            wli.x[idx] = wl->x[k + (l * wstride)];
            wli.y[idx] = wl->y[k + (l * wstride)];
            idx++;
        }
    }
    //---------------------------------------
    // copy shuffled array back
    wl->x = wli.x;
    wl->y = wli.y;
    //---------------------------------------
    // IRT_TRACE("after::wl.x = ");
    // for(int k=0; k<wl->size; k++) {
    //   IRT_TRACE("%f,",wl->x[k]);
    //}
    // IRT_TRACE("\n");
    // IRT_TRACE("after::wl.y = ");
    // for(int k=0; k<wl->size; k++) {
    //   IRT_TRACE("%f,",wl->y[k]);
    //}
    // IRT_TRACE("\n===============================\n");
    ////---------------------------------------
    // for(int k=0; k<wl->size; k++) {
    //   wli.x[base+l] = wl->x[(k+(l*stride))];
    //   wli.y[base+l] = wl->y[(k+(l*stride))];
    //}
}
void IRT_top::IRT_RESAMP::rshuffle_scbd(warp_line* wl, Eirt_blocks_enum block_type, Eirt_resamp_dtype_enum DataType)
{
    resamp_shfl_inf sv_shfl_s;
    uint32_t        svdt;
    memset(&sv_shfl_s, 0, sizeof(resamp_shfl_inf));
    // IRT_TRACE("size %d\n",wl->size);
    sv_shfl_s.size = wl->size;
    for (int k = 0; k < wl->size; k++) {
        svdt           = IRT_top::IRT_UTILS::conversion_float_bit(wl->x[k], DataType, 0, 0, 0, 0, 0, 0, 0);
        sv_shfl_s.x[k] = svdt; // wl->x[k];
        if (block_type == e_irt_wrm) {
            svdt           = IRT_top::IRT_UTILS::conversion_float_bit(wl->y[k], DataType, 0, 0, 0, 0, 0, 0, 0);
            sv_shfl_s.y[k] = svdt; // wl->y[k];
        }
        // IRT_TRACE("sv_shfl_s.x,y=%x,%x\t",sv_shfl_s.x[k],sv_shfl_s.y[k]);
    }
    // IRT_TRACE("\n");
    if (block_type == e_irt_wrm) {
        wshfl_infQ.push(sv_shfl_s); /*IRT_TRACE("pushed to wshfl_infQ\n");*/
    }
    if (block_type == e_irt_grm) {
        gshfl_infQ.push(sv_shfl_s); /*IRT_TRACE("pushed to gshfl_infQ\n");*/
    }
}
//---------------------------------------
// suspension buffer addr/pad/meta generation for warp fetch thread
// block_type -> warp or Grad
template <Eirt_blocks_enum block_type>
bool IRT_top::IRT_RESAMP::warp_read(uint64_t&             addr,
                                    bool&                 rd_en,
                                    uint16_t&             lpad,
                                    uint16_t&             mpad,
                                    meta_data_struct&     meta_out,
                                    const bus128B_struct& rd_data,
                                    meta_data_struct      meta_in,
                                    bool                  rd_data_valid)
{
    //---------------------------------------
    std::string my_string =
        (block_type == e_irt_wrm) ? getInstanceName() + ":WARP_READ:" : getInstanceName() + ":GRAD_READ:";
    resamp_tracker*    atrc        = (block_type == e_irt_wrm) ? &wr_atrc : &gr_atrc;
    resamp_tracker*    dtrc        = (block_type == e_irt_wrm) ? &wr_dtrc : &gr_dtrc;
    FILE*              rFile       = (block_type == e_irt_wrm) ? wrFile : grFile;
    FILE*              shflFileptr = (block_type == e_irt_wrm) ? wshflFile : gshflFile;
    warp_line**        rimage      = (block_type == e_irt_wrm) ? resamp_wimage : resamp_gimage;
    Eirt_tranform_type irt_mode    = descr.irt_mode; // 0 - rotation, 1 - affine transform, 2 - projection, 3 - mesh
    uint16_t           resize_bli_grad_en = (block_type == e_irt_wrm) ? descr.resize_bli_grad_en : 0;
    uint8_t            image_type         = (block_type == e_irt_wrm) ? MIMAGE : GIMAGE;
    uint8_t            nelem              = (block_type == e_irt_wrm && descr.irt_mode != e_irt_rescale) ? 2 : 1;
    uint8_t            bc                 = (block_type == e_irt_wrm && descr.irt_mode != e_irt_rescale)
                     ? ((resize_bli_grad_en == 1) ? 4 : (1 << descr.image_par[image_type].Ps) / 2)
                     : (1 << descr.image_par[image_type].Ps);
    uint8_t Strp = (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) ? descr.mesh_stripe_stride
                                                                                : descr.image_par[image_type].S;
    //---------------------------------------
    int16_t X_start = (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) ? descr.Xi_start_offset : 0;
    int16_t X_end   = (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) ? descr.Xi_last_fixed
                                                                                 : descr.image_par[image_type].W;
    int16_t Y_start = (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) ? descr.Yi_start : 0;
    int16_t Y_end =
        (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) ? descr.Yi_end + 1 : descr.image_par[image_type].H;
    int16_t Y_end_data = (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale)
                             ? (descr.Yi_end - descr.Yi_start + 1)
                             : descr.image_par[image_type].H;
    bool is_rescale = (descr.irt_mode == e_irt_rescale);
#ifdef RUN_WITH_SV
    //---------------------------------------
    // Stripe number for RSB metadata checks
    if (block_type == e_irt_wrm) {
        irt_top->irt_top_sig.mstripe = atrc->pstripe;
    } else {
        irt_top->irt_top_sig.gstripe = atrc->pstripe;
    }
#endif
    //---------------------------------------
    uint16_t stripePlus, last_x_seg, swb, stripe_max_a, stripe_max_d;
    uint32_t addr_line;
    if (block_type == e_irt_wrm && descr.irt_mode == e_irt_rescale) {
        stripePlus   = ((descr.image_par[image_type].S - descr.mesh_stripe_stride) * 2);
        last_x_seg   = (int)descr.image_par[image_type].S;
        swb          = (descr.image_par[image_type].W << descr.image_par[image_type].Ps); // stripe width in bytes;
        stripe_max_a = last_x_seg;
        stripe_max_d = last_x_seg;
    } else {
        stripePlus = 0;
        last_x_seg = ((int)descr.image_par[image_type].W - ((no_x_seg - 1) * (int)descr.image_par[image_type].S));
        swb        = (atrc->pstripe == (no_x_seg - 1))
                  ? (last_x_seg << descr.image_par[image_type].Ps)
                  : (descr.image_par[image_type].S << descr.image_par[image_type].Ps); // stripe width in bytes;
        stripe_max_a = (atrc->pstripe == (no_x_seg - 1)) ? last_x_seg : (uint32_t)descr.image_par[image_type].S;
        stripe_max_d = (dtrc->pstripe == (no_x_seg - 1)) ? last_x_seg : (uint32_t)descr.image_par[image_type].S;
    }
    //---------------------------------------
    bool    rd_addr_done = atrc->pstripe >= no_x_seg;
    bool    rd_data_done = dtrc->pstripe >= no_x_seg;
    // uint8_t wstride      = (descr.irt_mode == e_irt_resamp_bwd2) ? (1 << descr.warp_stride) : 0;
    uint8_t wstride =
        (descr.irt_mode == e_irt_resamp_bwd2) ? ((descr.warp_stride == 1) ? 2 : ((descr.warp_stride == 3) ? 4 : 0)) : 0;
    // stripe_max_a += stripePlus;
    // stripe_max_d += stripePlus;
    /*
    uint16_t stripePlus = (block_type == e_irt_wrm && descr.irt_mode==e_irt_rescale) ? ((descr.image_par[image_type].S -
    descr.mesh_stripe_stride)*2)  : 0; uint16_t last_x_seg = (block_type == e_irt_wrm && descr.irt_mode==e_irt_rescale)
    ? (int)descr.image_par[image_type].S : ((int)descr.image_par[image_type].W - ((no_x_seg-1) *
    (int)descr.image_par[image_type].S));
    //uint16_t last_x_seg = (int)descr.image_par[image_type].W - ((no_x_seg-1) * (int)descr.image_par[image_type].S);
    uint16_t stripe_max_a = (atrc->pstripe == (no_x_seg-1) && descr.irt_mode!=e_irt_rescale) ? last_x_seg :
    (uint32_t)descr.image_par[image_type].S; uint16_t stripe_max_d = (dtrc->pstripe == (no_x_seg-1) &&
    descr.irt_mode!=e_irt_rescale) ? last_x_seg : (uint32_t)descr.image_par[image_type].S;
    //stripe_max_a += stripePlus;
    //stripe_max_d += stripePlus;
    bool rd_addr_done = atrc->pstripe >= no_x_seg;
    bool rd_data_done = dtrc->pstripe >= no_x_seg;
    uint16_t swb = (atrc->pstripe == (no_x_seg-1)) ? (last_x_seg << descr.image_par[image_type].Ps) :
    (descr.image_par[image_type].S << descr.image_par[image_type].Ps); //stripe width in bytes; uint8_t wstride =
    descr.warp_stride;
    */
    //======================================================
    // resize bli grad cfg
    uint16_t       relative_mode_en = (block_type == e_irt_wrm) ? descr.mesh_rel_mode : 0;
    float          gx               = descr.M11f;
    float          gy               = descr.M22f;
    float          Xc_o             = ((float)descr.image_par[OIMAGE].Xc) / 2;
    float          Yc_o             = ((float)descr.image_par[OIMAGE].Yc) / 2;
    float          Xc_i             = ((float)descr.image_par[IIMAGE].Xc) / 2;
    float          Yc_i             = ((float)descr.image_par[IIMAGE].Yc) / 2;
    int16_t        sw               = descr.image_par[image_type].S;
    bus128B_struct warp_data;
    bool           warp_data_valid   = 0;
    bool           clip_fp           = descr.clip_fp;
    bool           clip_fp_inf_input = descr.clip_fp_inf_input;
    uint32_t       warp_el;
    float          Xi[16] = {};
    float          Yi[16] = {};
    // end of resize bli grad cfg
    //======================================================
    // uint16_t Ps = descr.image_par[image_type].Ps;
    // uint16_t Hs = descr.image_par[image_type].Hs;
    // uint16_t S = descr.image_par[image_type].S;
    // uint16_t H = descr.image_par[image_type].H;
    irt_mem_ctrl_struct* irt_mem_ctrl = &irt_top->mem_ctrl[0];
    // IRT_TRACE_TO_LOG(3,rFile,"[stripe:line:coord]=[%d:%d:%d] nelem %d bc %d at cycle %d\n",
    // atrc->pstripe,atrc->pline,atrc->pcoord,nelem,bc,cycle);
    //---------------------------------------
    // DONE GENERATION
    if ((rd_addr_done == 1 && rd_data_done == 1) ||
        (block_type == e_irt_grm && (descr.irt_mode != e_irt_resamp_bwd1 && descr.irt_mode != e_irt_resamp_bwd2))) {
        return 1;
    }
    // warp first read reset
    uint8_t irm_client_index                            = (block_type == e_irt_wrm) ? 1 : 2;
    irt_top->irt_top_sig.irm_rd_first[irm_client_index] = 0;
    irt_top->irt_top_sig.irm_rd_last[irm_client_index]  = 0;
    //---------------------------------------
    if (!rd_addr_done) {
        if (atrc->pstripe == no_x_seg) { // task stripe
            IRT_TRACE("%s task %d is finished at cycle %d\n", my_string.c_str(), irt_top->task[e_resamp_wread], cycle);
            irt_top->task[e_resamp_wread]++;
            return 1;
        }
        // IRT_TRACE_TO_LOG(3,rFile,"%s: [task:stripe:line:coord]=[%d:%d:%d:%d] at cycle %d\n", my_string.c_str(),
        // irt_top->task[e_resamp_wread],atrc->pstripe,atrc->pline,atrc->pcoord,cycle);
        //---------------------------------------
        uint64_t byte = atrc->pcoord << descr.image_par[image_type].Ps;
        // for resize bli grad warp coord will be generated internally
        rd_en = (resize_bli_grad_en) ? 0 : 1;

        //---------------------------------------
        if (atrc->pstripe == 0 && atrc->pline == Y_start && atrc->pcoord == 0) {
            irt_top->irt_top_sig.irm_rd_first[irm_client_index] = 1;
        }
        //---------------------------------------
        if ((descr.bg_mode == 1) && (atrc->pline >= descr.image_par[image_type].H - 1)) {
            addr_line = descr.image_par[image_type].H - 1;
        } else {
            addr_line = atrc->pline;
        }
        IRT_TRACE_TO_LOG(
            3,
            rFile,
            "IRM : start add %lu pline %d Hs %d add stripe %d Stripe %d byte %lu X_start %d and pixel size %d\n",
            descr.image_par[image_type].addr_start,
            atrc->pline,
            descr.image_par[image_type].Hs,
            atrc->pstripe,
            Strp,
            byte,
            X_start,
            descr.image_par[image_type].Ps);
        addr = descr.image_par[image_type].addr_start + (addr_line * descr.image_par[image_type].Hs) +
               (atrc->pstripe * (uint64_t)Strp * (1 << descr.image_par[image_type].Ps)) +
               //(atrc->pstripe * (uint64_t)descr.image_par[image_type].S * (1<< descr.image_par[image_type].Ps)) +
               (uint64_t)byte - ((X_start * -1) * (1 << descr.image_par[image_type].Ps));
        IRT_TRACE_TO_LOG(3,
                         rFile,
                         "IRM rd add %lu and pline %d Hs %d Stripe %d\n",
                         addr,
                         atrc->pline,
                         descr.image_par[image_type].Hs,
                         Strp);
        // pad compute
        if ((descr.irt_mode == e_irt_rescale) && (descr.bg_mode == 2) &&
            (((int32_t)(atrc->pline) < 0) || ((int32_t)(atrc->pline) > descr.image_par[image_type].H - 1))) {
            lpad = 128;
            mpad = 128;
            swb  = 0;
        } else {
            lpad = (byte == 0) ? ((int16_t)(abs(X_start)) << descr.image_par[image_type].Ps) : 0;
            mpad = ((byte + IRT_IFIFO_WIDTH - ((X_start * -1) << descr.image_par[image_type].Ps)) <= swb)
                       ? 0
                       : ((byte + IRT_IFIFO_WIDTH) - swb) - ((X_start * -1) << descr.image_par[image_type].Ps);
        }

        // lpad = 0; // WARP WILL ALWAYS HAVE LPAD == 0
        // mpad = ((byte + IRT_IFIFO_WIDTH) < swb) ? 0 : ((byte + IRT_IFIFO_WIDTH) - swb);
        if (mpad > IRT_IFIFO_WIDTH) {
            mpad = IRT_IFIFO_WIDTH;
        }
        // mpad = ((byte + IRT_IFIFO_WIDTH) < (descr.image_par[image_type].S << descr.image_par[image_type].Ps)) ? 0 :
        //   ((byte + IRT_IFIFO_WIDTH) - (descr.image_par[image_type].S << descr.image_par[image_type].Ps));
        IRT_TRACE_TO_LOG(3,
                         rFile,
                         "lpad = %d mpad = %d byte %lu IRT_IFIFO_WIDTH %d swb %d Strp %d no_x_seg %d\n",
                         lpad,
                         mpad,
                         byte,
                         IRT_IFIFO_WIDTH,
                         swb,
                         Strp,
                         no_x_seg);
        // IRT_TRACE_TO_LOG(3,rFile,"mpad = %d byte %d IRT_IFIFO_WIDTH %d swb %d \n", mpad, byte, IRT_IFIFO_WIDTH, swb);
        // metadata
        meta_out.task   = irt_top->task[e_resamp_wread];
        meta_out.stripe = atrc->pstripe;
        meta_out.line   = atrc->pline;
        meta_out.coord  = atrc->pcoord;
        //---------------------------------------
        //===========================================
        // Warp coord gen block : This is applicable
        // only when resize bli grad enable is there
        // or relative mode enabled
        //===========================================
        if (resize_bli_grad_en) {
            for (int16_t i = 0; i < 4; i++) {
                float Xo        = (float)((atrc->pstripe * sw) + (atrc->pcoord + i * 4)) - Xc_o;
                float Yo        = ((float)atrc->pline) - Yc_o;
                Xi[i * 4]       = (gx * Xo) + Xc_i;
                Yi[i * 4]       = (gy * Yo) + Yc_i;
                Xi[(i * 4) + 1] = Xi[i * 4] + gx;
                Xi[(i * 4) + 2] = Xi[i * 4] + (2 * gx);
                Xi[(i * 4) + 3] = Xi[i * 4] + (3 * gx);
                Yi[(i * 4) + 1] = Yi[i * 4];
                Yi[(i * 4) + 2] = Yi[i * 4];
                Yi[(i * 4) + 3] = Yi[i * 4];

                /*for(int ii=0;ii<4;ii++){
                  printf("warp aaaaa (%f,%f) ",Xi[(i*4)+ii],Yi[(i*4)+ii]);
                }
                printf("\n");*/
            }

            warp_data_valid = 1;
            rd_data_valid   = 1;
        }

        //===========================================
        // end of warp coord gen block
        //===========================================
        int16_t warp_ps = resize_bli_grad_en ? 8 : (1 << descr.image_par[image_type].Ps);
        atrc->pcoord += (IRT_IFIFO_WIDTH / warp_ps);

        if (atrc->pstripe == (no_x_seg - 1) &&
            // atrc->pline   == descr.image_par[image_type].H-1 &&
            atrc->pline == Y_end - 1 && atrc->pcoord >= stripe_max_a) {
            irt_top->irt_top_sig.irm_rd_last[irm_client_index] = 1;
        }
        //      IRT_TRACE("%s: irm_rd_last=%d at cycle %d\n", my_string.c_str(),
        //      irt_top->irt_top_sig.irm_rd_last[1],cycle);
        //---------------------------------------
        if ((atrc->pcoord >= stripe_max_a)) { // line end
            atrc->pline++;
            atrc->pcoord = 0;
        }
        // if(atrc->pline == descr.image_par[image_type].H){ //stripe end
        if ((atrc->pline == Y_end)) { // stripe end
            atrc->pline = 0;
            atrc->pstripe++;
            stripe_max_a = (atrc->pstripe == (no_x_seg - 1)) ? last_x_seg : (int)descr.image_par[image_type].S;
        }
        //---------------------------------------
        IRT_TRACE_TO_LOG(22,
                         rFile,
                         "ADDR_GEN: addr = %lu [lpad:mpad]=[%d:%d] "
                         "[stripe:line:coord:stripe_max_a:Y_start:Y_end]=[%d:%d:%d:%d:%d:%d] "
                         "meta_out[task:stripe:line:coord]=[%d:%d:%d:%d] at cycle %d\n",
                         addr,
                         lpad,
                         mpad,
                         atrc->pstripe,
                         atrc->pline,
                         atrc->pcoord,
                         stripe_max_a,
                         Y_start,
                         Y_end,
                         meta_out.task,
                         meta_out.stripe,
                         meta_out.line,
                         meta_out.coord,
                         cycle);
    }
    //---------------------------------------
    if (!rd_data_done) {
        //---------------------------------------
        ////TODO -- FIX ME - BELOW CHECK MUST BE ENABLED
        // if(pstripe != meta_in.stripe || pline != meta_in.line || pcoord!=meta_in.coord) {
        //   IRT_TRACE("%s:warp_read metadata mismatch::[task:stripe:line:coord]::EXP[%d:%d:%d:%d] REC[%d:%d:%d:%d]\n",
        //   my_string.c_str(),irt_top->task[e_resamp_wread],pstripe,pline,pcoord,meta_in.task,meta_in.stripe,meta_in.line,meta_in.coord);
        //   IRT_TRACE_TO_RES(test_res, "%s: failed, warp_read metadata mismatch\n", my_string.c_str());
        //   IRT_CLOSE_FAILED_TEST(0);
        //}
        //---------------------------------------
        uint32_t pix[2] = {0};
        uint8_t  bi, j = 0;
        if (rd_data_valid &&
            dtrc->pstripe <
                no_x_seg) { // TODO -- pstripe < no_x_seg to be removed once rd_data_valid de-assertion is fixed
            //---------------------------------------
            IRT_TRACE_TO_LOG(22,
                             rFile,
                             "WARP READ_DATA:meta_in[task:stripe:line:coord]=[%d:%d:%d:%d:%d]",
                             meta_in.task,
                             meta_in.stripe,
                             meta_in.line,
                             meta_in.coord,
                             cycle);
            for (int i = 127; i >= 0; i--)
                IRT_TRACE_TO_LOG(12, rFile, "%x:", rd_data.pix[i]);
            IRT_TRACE_TO_LOG(22, rFile, " ]\n");
            IRT_TRACE_TO_LOG(23,
                             compFile,
                             "WARP READ_DATA:meta_in[task:stripe:line:coord]=[%d:%d:%d:%d:%d]",
                             meta_in.task,
                             meta_in.stripe,
                             meta_in.line,
                             meta_in.coord,
                             cycle);
            for (int i = 127; i >= 0; i--)
                IRT_TRACE_TO_LOG(23, compFile, "%x:", rd_data.pix[i]);
            IRT_TRACE_TO_LOG(23, compFile, " ]\n");
            //---------------------------------------
            // printf("bc and nelem %d , %d \n ", bc,nelem);
            for (uint8_t i = 0; i < IRT_IFIFO_WIDTH; i += (bc * nelem)) {
                float Xo = relative_mode_en ? (float)((int16_t)((dtrc->pstripe * sw) + (dtrc->pcoord))) : 0;
                float Yo = relative_mode_en ? (float)((int16_t)dtrc->pline) : 0;
                for (uint8_t c = 0; c < nelem; c++) {
                    pix[c] = 0;
                    for (uint8_t j = 0; j < bc; j++) {
                        pix[c] |= (uint32_t)(rd_data.pix[i + c * bc + j] << (8 * j));
                    }
                    if (descr.irt_mode == e_irt_rescale) {
                        if (descr.image_par[image_type].DataType == e_irt_int16)
                            pix[c] &= descr.Msi;
                    } else {
                        if (descr.image_par[image_type].DataType == e_irt_int16)
                            pix[c] &= 0xFFFF;
                    }
                    //---------------------------------------
                    /// nan  - inf check
                    // pix[c] = check_nan_inf(pix[c],descr.image_par[image_type].DataType);
                    //---------------------------------------
                    if (!resize_bli_grad_en) {
                        if (descr.image_par[image_type].DataType != e_irt_int16) {
                            if (c == 0)
                                rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord] =
                                    IRT_top::IRT_UTILS::conversion_bit_float(
                                        pix[c], descr.image_par[image_type].DataType, clip_fp) /* + Xo*/;
                            else
                                rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord] =
                                    IRT_top::IRT_UTILS::conversion_bit_float(
                                        pix[c], descr.image_par[image_type].DataType, clip_fp) /* + Yo*/;
                        } else {
                            if (c == 0)
                                rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord] =
                                    IRT_top::IRT_UTILS::conversion_fxd16_float(
                                        pix[c],
                                        descr.mesh_point_location,
                                        is_rescale) /* + Xo*/; // TODO - pix[c] downcaste from uint32 to
                                                               // int16 ??
                            else
                                rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord] =
                                    IRT_top::IRT_UTILS::conversion_fxd16_float(
                                        pix[c], descr.mesh_point_location, is_rescale) /* + Yo*/;
                        }
                        if (relative_mode_en) {
                            if (c == 0) {
                                rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord] += Xo;
                            } else {
                                rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord] += Yo;
                            }
                        }
                        IRT_TRACE_TO_LOG(15,
                                         rFile,
                                         "[READ_DATA:  PIXi[%x:%x] PIXf[%f:%f] \n",
                                         pix[0],
                                         pix[1],
                                         rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord],
                                         rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord]);
                    } else {
                        if (c == 0)
                            rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord] = Xi[i / 8];
                        else
                            rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord] = Yi[i / 8];
                    }
                }
                IRT_TRACE_TO_LOG(
                    15,
                    rFile,
                    "[READ_DATA: [stripe:line:coord:stripe_max_d] [%d:%d:%d:%d] Dtype %d PIXi[%x:%x] "
                    "PIXf[%f:%f]PIXf.FP32[%x:%x]\n",
                    dtrc->pstripe,
                    dtrc->pline,
                    dtrc->pcoord,
                    stripe_max_d,
                    descr.image_par[image_type].DataType,
                    pix[0],
                    pix[1],
                    rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord],
                    rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord],
                    IRT_top::IRT_UTILS::conversion_float_bit(
                        rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord], e_irt_fp32, 0, 0, 0, 0, 0, 0, 0),
                    IRT_top::IRT_UTILS::conversion_float_bit(
                        rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord], e_irt_fp32, 0, 0, 0, 0, 0, 0, 0));
                // IRT_TRACE("%s:[READ_DATA: [stripe:line:coord:stripe_max_d:image_type:dtype] [%d:%d:%d:%d:%d:%d]
                // PIXi[%x:%x] PIXf[%f:%f]\n",
                //      my_string.c_str(),dtrc->pstripe,dtrc->pline,dtrc->pcoord,stripe_max_d,image_type,descr.image_par[image_type].DataType,
                //      pix[0],pix[1],rimage[dtrc->pstripe][dtrc->pline].x[dtrc->pcoord],rimage[dtrc->pstripe][dtrc->pline].y[dtrc->pcoord]);

                //---------------------------------------
                if ((dtrc->pcoord + 1) == stripe_max_d) {
                    rimage[dtrc->pstripe][dtrc->pline].valid = 1;
                    rimage[dtrc->pstripe][dtrc->pline].size =
                        (dtrc->pstripe == (no_x_seg - 1)) ? last_x_seg : descr.image_par[image_type].S;
                    irt_mem_ctrl->last_line[0] = dtrc->pline; // TODO - bg_mode=0 needs change here
                    first_vld_warp_line        = 1;
                    //---------------------------------------
                    IRT_TRACE_TO_LOG(12, rFile, " wstride :: %d ", wstride);
                    if (wstride > 0) {
                        rshuffle(&rimage[dtrc->pstripe][dtrc->pline], wstride);
#if defined(RUN_WITH_SV)
                        //******************************
                        uint16_t pps = dtrc->pstripe;
                        uint16_t ppl = dtrc->pline;
                        uint16_t ppc = dtrc->pcoord;
                        uint16_t ppe = rimage[pps][ppl].size;
                        for (int k = 0; k < ppe; k++) {
                            IRT_TRACE_TO_LOG(
                                10,
                                shflFileptr,
                                "[%d:%d:%d] \t\t\t::\t\t\t [%f:%f] \t\t\t::\t\t\t  [%x:%f]::\t\t\t  [%x:%f] \n",
                                pps,
                                ppl,
                                k,
                                rimage[pps][ppl].y[k],
                                rimage[pps][ppl].x[k],
                                IRT_top::IRT_UTILS::conversion_float_bit(
                                    rimage[pps][ppl].x[k], descr.image_par[image_type].DataType, 0, 0, 0, 0, 0, 0, 0),
                                rimage[pps][ppl].x[k],
                                IRT_top::IRT_UTILS::conversion_float_bit(
                                    rimage[pps][ppl].y[k], descr.image_par[image_type].DataType, 0, 0, 0, 0, 0, 0, 0),
                                rimage[pps][ppl].y[k]);
                        }
                        rshuffle_scbd(
                            &rimage[dtrc->pstripe][dtrc->pline], block_type, descr.image_par[image_type].DataType);
#endif
                    }
                    //******************************
                    //---------------------------------------
                    // IRT_TRACE_TO_LOG(3,rFile,"line %d write to rimage complete \n",irt_mem_ctrl->last_line[0]);
                    // break;
                }
                dtrc->pcoord++;
                //---------------------------------------
                if (dtrc->pcoord >= stripe_max_d) { // line end
                    dtrc->pcoord = 0;
                    dtrc->pline++;
                }
                // if(dtrc->pline == descr.image_par[image_type].H){ //stripe end
                if (dtrc->pline == Y_end_data) { // stripe end
                    dtrc->pline = 0;
                    dtrc->pstripe++;
                    stripe_max_d = (dtrc->pstripe == (no_x_seg - 1)) ? last_x_seg : (int)descr.image_par[image_type].S;
                }
                if (dtrc->pcoord == 0) { // line ended, so ignore further elements
                    break;
                }
            }
        }
    }
    //---------------------------------------
    return 0;
}
//---------------------------------------
// four dimensional memory allocation.
float**** IRT_top::IRT_RESAMP::mycalloc4(int num_img, int num_ifms, int ifm_rows, int ifm_cols)
{
    int       j, k, l;
    float**** m4_out;

    m4_out = (float****)calloc(num_img, sizeof(float***));

    for (j = 0; j < num_img; j++) {
        m4_out[j] = (float***)calloc(num_ifms, sizeof(float**));
        for (k = 0; k < num_ifms; k++) {
            m4_out[j][k] = (float**)calloc(ifm_rows, sizeof(float*));
            for (l = 0; l < ifm_rows; l++) {
                m4_out[j][k][l] = (float*)calloc(ifm_cols, sizeof(float));
            }
        }
    }
    return (m4_out);
}
//---------------------------------------
// return a normally distributed random number
double IRT_top::IRT_RESAMP::normalRandom()
{
    // double y1 = RandomGenerator();
    // double y2 = RandomGenerator();
    double y1 = ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
    double y2 = ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
    return cos(2 * 3.14 * y2) * sqrt(-2. * log(y1));
}
//---------------------------------------
void IRT_top::IRT_RESAMP::coord_to_tag(float data[2], int32_t** tag, uint16_t cl_size)
{
    // std::string my_string = getInstanceName() + ":COORD_TO_TAG:";
    //---------------------------------------
    tag[0][0] = (int)floor(data[0]);
    tag[0][1] = (int)floor(floor(data[1]) / (float)cl_size);
    tag[1][0] = (int)floor(data[0]);
    tag[1][1] = (int)floor(ceil(data[1]) / (float)cl_size);
    tag[2][0] = (int)ceil(data[0]);
    tag[2][1] = (int)floor(floor(data[1]) / (float)cl_size);
    tag[3][0] = (int)ceil(data[0]);
    tag[3][1] = (int)floor(ceil(data[1]) / (float)cl_size);
    //---------------------------------------
    // IRT_TRACE("%s DATA [%f %f] ", my_string.c_str(),data[0], data[1]);
    // for(int i=0;i<4;i++){
    //   IRT_TRACE(("coord_to_tag [%d %d] ",tag[i][0], tag[i][1]);
    //}
    // IRT_TRACE("\n");

}

// merging tags
void IRT_top::IRT_RESAMP::tag_merger(uint8_t    p_size,
                                     int32_t*** tag,
                                     int32_t*** m_tag,
                                     int32_t**  m_tag_valid,
                                     uint8_t&   two_lookup_en)
{
    uint8_t no_conf = 1;
    for (uint8_t i = 0; i < p_size; i++) {
        for (uint8_t j = 0; j < 4; j++) {
            uint8_t bank_row             = (tag[i][j][0]) & (int32_t)(NUM_BANK_ROW - 1);
            uint8_t tag_c                = (tag[i][j][1]) & 0x1;
            m_tag[tag_c][bank_row][0]    = tag[i][j][0];
            m_tag[tag_c][bank_row][1]    = tag[i][j][1];
            m_tag_valid[tag_c][bank_row] = 1;
        }
    }

    for (uint8_t i = 0; i < IRT_ROT_MEM_ROW_BANKS; i++) {
        if ((m_tag_valid[1][i] == 1) & (m_tag_valid[0][i] == 0)) {
            m_tag[0][i][0]    = m_tag[1][i][0];
            m_tag[0][i][1]    = m_tag[1][i][1];
            m_tag_valid[0][i] = m_tag_valid[1][i];
            m_tag_valid[1][i] = 0;
        }
    }

    for (uint8_t i = 0; i < IRT_ROT_MEM_ROW_BANKS; i++) {
        no_conf = ((m_tag_valid[0][i] == 1) & (m_tag_valid[1][i] == 1)) ? 0 : 1;
        if (no_conf == 0)
            break;
    }
    two_lookup_en = (no_conf == 0) ? 1 : 0;

    // if(no_conf==0) {
    //  two_lookup_cycles = two_lookup_cycles+1;
    //  printf("======================== %d \n",two_lookup_cycles);
    //}
}

// set and ref generation
void IRT_top::IRT_RESAMP::ref_gen(int32_t tag[2], uint8_t& bank_row, uint8_t& set, uint32_t& ref)
{
    bank_row = (uint8_t)(tag[0] & (int32_t)(NUM_BANK_ROW - 1));
    // uint8_t bank_row_set =(uint8_t) (((int)floor((float)abs(tag[0])/(float)NUM_BANK_ROW))%(NSETS/(NUM_BANK_ROW*2)));
    uint8_t  bank_row_set = (uint8_t)((tag[0] >> (uint8_t)log2(NUM_BANK_ROW)) & ((NSETS / (NUM_BANK_ROW * 2)) - 1));
    uint8_t  bank_col_set = (uint8_t)(tag[1] & 0x1);
    uint32_t row_key      = (((tag[0] >> 5) & 0x000001FF) << 8);
    uint32_t col_key      = ((tag[1] >> 1) & 0x000000FF);
    set                   = bank_row * (NSETS / NUM_BANK_ROW) + bank_row_set * 2 + bank_col_set;
    ref                   = (row_key | col_key);
}

// float IRT_top::IRT_RESAMP::conversion_bit_float(uint32_t Y){
//    float out_f;
//    //assert(sizeof(out_f)==sizeof(Y));
//    memcpy(&out_f, &Y, sizeof(uint32_t));
//    return out_f;
//}
//
// uint32_t IRT_top::IRT_RESAMP::conversion_float_bit(float X){
//    uint32_t out_bit;
//    //assert(sizeof(out_bit)==sizeof(X));
//    memcpy(&out_bit, &X, sizeof(float));
//    return out_bit;
//}
//---------------------------------------
void IRT_top::IRT_RESAMP::set_plru_4miss(uint8_t set, uint8_t hit_way)
{ // set MRU
    //---------------------------------------
    if (hit_way > NWAYS) {
        IRT_TRACE("Error - set_plru setting out-of-range cache-WAY [hitway:NWAYS] = [%d:%d]  \n", hit_way, NWAYS);
        IRT_TRACE_TO_RES(
            test_res, "Error - set_plru setting out-of-range WEAY [hitway:NWAYS] = [%d:%d]  \n", hit_way, NWAYS);
        IRT_CLOSE_FAILED_TEST(0);
    }
    //---------------------------------------
    uint16_t my_tree_bits = 0;
    for (int ss = 0; ss < NWAYS - 1; ss++) {
        my_tree_bits |= ((cache_tag_array.tree_bits[set][ss] & 0x1) << ss);
    }
    // IRT_TRACE_TO_LOG(2,parseFile,"set_plru_4miss - BEFORE: set=%d;tree_bits:%x\n",set,my_tree_bits);
    //---------------------------------------
    switch (hit_way) {
        case 0:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 1;
            cache_tag_array.tree_bits[set][3] = 1;
            cache_tag_array.tree_bits[set][7] = 1;
            break;
        case 1:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 1;
            cache_tag_array.tree_bits[set][3] = 1;
            cache_tag_array.tree_bits[set][7] = 0;
            break;
        case 2:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 1;
            cache_tag_array.tree_bits[set][3] = 0;
            cache_tag_array.tree_bits[set][8] = 1;
            break;
        case 3:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 1;
            cache_tag_array.tree_bits[set][3] = 0;
            cache_tag_array.tree_bits[set][8] = 0;
            break;
        case 4:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 0;
            cache_tag_array.tree_bits[set][4] = 1;
            cache_tag_array.tree_bits[set][9] = 1;
            break;
        case 5:
            cache_tag_array.tree_bits[set][0] = 1;
            cache_tag_array.tree_bits[set][1] = 0;
            cache_tag_array.tree_bits[set][4] = 1;
            cache_tag_array.tree_bits[set][9] = 0;
            break;
        case 6:
            cache_tag_array.tree_bits[set][0]  = 1;
            cache_tag_array.tree_bits[set][1]  = 0;
            cache_tag_array.tree_bits[set][4]  = 0;
            cache_tag_array.tree_bits[set][10] = 1;
            break;
        case 7:
            cache_tag_array.tree_bits[set][0]  = 1;
            cache_tag_array.tree_bits[set][1]  = 0;
            cache_tag_array.tree_bits[set][4]  = 0;
            cache_tag_array.tree_bits[set][10] = 0;
            break;
        case 8:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 1;
            cache_tag_array.tree_bits[set][5]  = 1;
            cache_tag_array.tree_bits[set][11] = 1;
            break;
        case 9:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 1;
            cache_tag_array.tree_bits[set][5]  = 1;
            cache_tag_array.tree_bits[set][11] = 0;
            break;
        case 10:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 1;
            cache_tag_array.tree_bits[set][5]  = 0;
            cache_tag_array.tree_bits[set][12] = 1;
            break;
        case 11:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 1;
            cache_tag_array.tree_bits[set][5]  = 0;
            cache_tag_array.tree_bits[set][12] = 0;
            break;
        case 12:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 0;
            cache_tag_array.tree_bits[set][6]  = 1;
            cache_tag_array.tree_bits[set][13] = 1;
            break;
        case 13:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 0;
            cache_tag_array.tree_bits[set][6]  = 1;
            cache_tag_array.tree_bits[set][13] = 0;
            break;
        case 14:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 0;
            cache_tag_array.tree_bits[set][6]  = 0;
            cache_tag_array.tree_bits[set][14] = 1;
            break;
        case 15:
            cache_tag_array.tree_bits[set][0]  = 0;
            cache_tag_array.tree_bits[set][2]  = 0;
            cache_tag_array.tree_bits[set][6]  = 0;
            cache_tag_array.tree_bits[set][14] = 0;
            break;
    }
    //---------------------------------------
    for (int ss = 0; ss < NWAYS - 1; ss++) {
        my_tree_bits |= ((cache_tag_array.tree_bits[set][ss] & 0x1) << ss);
    }
    // IRT_TRACE_TO_LOG(2,parseFile,"set_plru_4miss - after: set=%d;tree_bits:%x\n",set,my_tree_bits);
}
//---------------------------------------
void IRT_top::IRT_RESAMP::set_plru_4hit(uint8_t set, uint8_t hit_way)
{ // set MRU
    //---------------------------------------
    if (hit_way > NWAYS) {
        IRT_TRACE("Error - set_plru setting out-of-range cache-WAY [hitway:NWAYS] = [%d:%d]  \n", hit_way, NWAYS);
        IRT_TRACE_TO_RES(
            test_res, "Error - set_plru setting out-of-range WEAY [hitway:NWAYS] = [%d:%d]  \n", hit_way, NWAYS);
        IRT_CLOSE_FAILED_TEST(0);
    }
    //---------------------------------------
    uint16_t my_tree_bits = 0;
    for (int ss = 0; ss < NWAYS - 1; ss++) {
        my_tree_bits |= ((cache_tag_array.tree_bits[set][ss] & 0x1) << ss);
    }
    // IRT_TRACE_TO_LOG(2,parseFile,"set_plru_4hit - BEFORE: set=%d;tree_bits:%x\n",set,my_tree_bits);
    //---------------------------------------
    switch (hit_way) {
        case 0:
            if (cache_tag_array.tree_bits[set][7] == 0)
                cache_tag_array.tree_bits[set][7] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][3] == 0)
                cache_tag_array.tree_bits[set][3] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 0)
                cache_tag_array.tree_bits[set][1] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 1:
            if (cache_tag_array.tree_bits[set][7] == 1)
                cache_tag_array.tree_bits[set][7] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][3] == 0)
                cache_tag_array.tree_bits[set][3] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 0)
                cache_tag_array.tree_bits[set][1] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 2:
            if (cache_tag_array.tree_bits[set][8] == 0)
                cache_tag_array.tree_bits[set][8] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][3] == 1)
                cache_tag_array.tree_bits[set][3] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 0)
                cache_tag_array.tree_bits[set][1] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 3:
            if (cache_tag_array.tree_bits[set][8] == 1)
                cache_tag_array.tree_bits[set][8] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][3] == 1)
                cache_tag_array.tree_bits[set][3] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 0)
                cache_tag_array.tree_bits[set][1] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 4:
            if (cache_tag_array.tree_bits[set][9] == 0)
                cache_tag_array.tree_bits[set][9] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][4] == 0)
                cache_tag_array.tree_bits[set][4] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 1)
                cache_tag_array.tree_bits[set][1] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 5:
            if (cache_tag_array.tree_bits[set][9] == 1)
                cache_tag_array.tree_bits[set][9] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][4] == 0)
                cache_tag_array.tree_bits[set][4] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 1)
                cache_tag_array.tree_bits[set][1] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 6:
            if (cache_tag_array.tree_bits[set][10] == 0)
                cache_tag_array.tree_bits[set][10] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][4] == 1)
                cache_tag_array.tree_bits[set][4] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 1)
                cache_tag_array.tree_bits[set][1] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 7:
            if (cache_tag_array.tree_bits[set][10] == 1)
                cache_tag_array.tree_bits[set][10] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][4] == 1)
                cache_tag_array.tree_bits[set][4] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][1] == 1)
                cache_tag_array.tree_bits[set][1] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 0)
                cache_tag_array.tree_bits[set][0] = 1;
            else
                break;
            break;

        case 8:
            if (cache_tag_array.tree_bits[set][11] == 0)
                cache_tag_array.tree_bits[set][11] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][5] == 0)
                cache_tag_array.tree_bits[set][5] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 0)
                cache_tag_array.tree_bits[set][2] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 9  : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 1;
        // cache_tag_array.tree_bits[set][5] = 1; cache_tag_array.tree_bits[set][11] = 0; break;
        case 9:
            if (cache_tag_array.tree_bits[set][11] == 1)
                cache_tag_array.tree_bits[set][11] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][5] == 0)
                cache_tag_array.tree_bits[set][5] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 0)
                cache_tag_array.tree_bits[set][2] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 10 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 1;
        // cache_tag_array.tree_bits[set][5] = 0; cache_tag_array.tree_bits[set][12] = 1; break;
        case 10:
            if (cache_tag_array.tree_bits[set][12] == 0)
                cache_tag_array.tree_bits[set][12] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][5] == 1)
                cache_tag_array.tree_bits[set][5] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 0)
                cache_tag_array.tree_bits[set][2] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 11 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 1;
        // cache_tag_array.tree_bits[set][5] = 0; cache_tag_array.tree_bits[set][12] = 0; break;
        case 11:
            if (cache_tag_array.tree_bits[set][12] == 1)
                cache_tag_array.tree_bits[set][12] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][5] == 1)
                cache_tag_array.tree_bits[set][5] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 0)
                cache_tag_array.tree_bits[set][2] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 12 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 0;
        // cache_tag_array.tree_bits[set][6] = 1; cache_tag_array.tree_bits[set][13] = 1; break;
        case 12:
            if (cache_tag_array.tree_bits[set][13] == 0)
                cache_tag_array.tree_bits[set][13] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][6] == 0)
                cache_tag_array.tree_bits[set][6] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 1)
                cache_tag_array.tree_bits[set][2] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 13 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 0;
        // cache_tag_array.tree_bits[set][6] = 1; cache_tag_array.tree_bits[set][13] = 0; break;
        case 13:
            if (cache_tag_array.tree_bits[set][13] == 1)
                cache_tag_array.tree_bits[set][13] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][6] == 0)
                cache_tag_array.tree_bits[set][6] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 1)
                cache_tag_array.tree_bits[set][2] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 14 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 0;
        // cache_tag_array.tree_bits[set][6] = 0; cache_tag_array.tree_bits[set][14] = 1; break;
        case 14:
            if (cache_tag_array.tree_bits[set][14] == 0)
                cache_tag_array.tree_bits[set][14] = 1;
            else
                break;
            if (cache_tag_array.tree_bits[set][6] == 1)
                cache_tag_array.tree_bits[set][6] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 1)
                cache_tag_array.tree_bits[set][2] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;

        // case 15 : cache_tag_array.tree_bits[set][0] = 0;  cache_tag_array.tree_bits[set][2] = 0;
        // cache_tag_array.tree_bits[set][6] = 0; cache_tag_array.tree_bits[set][14] = 0; break;
        case 15:
            if (cache_tag_array.tree_bits[set][14] == 1)
                cache_tag_array.tree_bits[set][14] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][6] == 1)
                cache_tag_array.tree_bits[set][6] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][2] == 1)
                cache_tag_array.tree_bits[set][2] = 0;
            else
                break;
            if (cache_tag_array.tree_bits[set][0] == 1)
                cache_tag_array.tree_bits[set][0] = 0;
            else
                break;
            break;
    }
    for (int ss = 0; ss < NWAYS - 1; ss++) {
        my_tree_bits |= ((cache_tag_array.tree_bits[set][ss] & 0x1) << ss);
    }
    // IRT_TRACE_TO_LOG(2,parseFile,"set_plru_4hit - after: set=%d;tree_bits:%x\n",set,my_tree_bits);
}
//---------------------------------------
void IRT_top::IRT_RESAMP::get_plru(uint8_t set, uint8_t& miss_way, bool lru_peek)
{ // set MRU
    if (cache_tag_array.tree_bits[set][0] == 0) {
        if (cache_tag_array.tree_bits[set][1] == 0) {
            if (cache_tag_array.tree_bits[set][3] == 0) {
                miss_way = (cache_tag_array.tree_bits[set][7] == 0) ? 0 : 1;
            } else {
                miss_way = (cache_tag_array.tree_bits[set][8] == 0) ? 2 : 3;
            }
        } else { // tree_bits[set][1] ==  1
            if (cache_tag_array.tree_bits[set][4] == 0) {
                miss_way = (cache_tag_array.tree_bits[set][9] == 0) ? 4 : 5;
            } else {
                miss_way = (cache_tag_array.tree_bits[set][10] == 0) ? 6 : 7;
            }
        }
    } else { // tree_bits[set][0] == 1)
        if (cache_tag_array.tree_bits[set][2] == 0) {
            if (cache_tag_array.tree_bits[set][5] == 0) {
                miss_way = (cache_tag_array.tree_bits[set][11] == 0) ? 8 : 9;
            } else {
                miss_way = (cache_tag_array.tree_bits[set][12] == 0) ? 10 : 11;
            }
        } else { // tree_bits[set][2] ==  1
            if (cache_tag_array.tree_bits[set][6] == 0) {
                miss_way = (cache_tag_array.tree_bits[set][13] == 0) ? 12 : 13;
            } else {
                miss_way = (cache_tag_array.tree_bits[set][14] == 0) ? 14 : 15;
            }
        }
    }
    //---------------------------------------
    if (!lru_peek) {
        set_plru_4miss(set, miss_way);
    }
    // IRT_TRACE_TO_LOG(2,parseFile,"get_plru - AFTER: lru_peek=%d set=%d\n",lru_peek,set);
}
//---------------------------------------
// way - For miss -> indicate victim-way [PLRU]
//      For Hit  -> indicates matching-way [MRU]
void IRT_top::IRT_RESAMP::ReferencePage(uint8_t set, int ref, uint8_t& way, uint8_t& pvalid, uint8_t& miss_hit)
{
    uint8_t miss_or_hit = 1; // 1 -> MISS  0 -> HIT
    uint8_t hit_way     = NWAYS + 1;
    uint8_t miss_way    = NWAYS + 1;
    //---------------------------------------
    for (int i = 0; i < NWAYS; i++) {
        if ((cache_tag_array.pagenumbers[set][i] == ref) && (cache_tag_array.valid[set][i] == 1)) { // HIT
            miss_or_hit = 0; // HIT
            hit_way     = i;
            break;
        }
    }
    //---------------------------------------
    if (miss_or_hit == 0) { // HIT CASE
        if (irt_top->irt_cfg.plru_mode == 1) {
            set_plru_4miss(set, hit_way); // set MRU
            // IRT_TRACE_TO_LOG(2,parseFile,"calling set_plru_4miss for plru_mode=1\n");
        } else {
            set_plru_4hit(set, hit_way); // set MRU
            // IRT_TRACE_TO_LOG(2,parseFile,"calling set_plru_4hit for plru_mode=0\n");
        }
        way    = hit_way;
        pvalid = cache_tag_array.valid[set][hit_way];
    } else { // miss case
        get_plru(set, miss_way, 0); // get LRU & set it as MRU
        pvalid                                     = cache_tag_array.valid[set][miss_way];
        cache_tag_array.pagenumbers[set][miss_way] = ref;
        cache_tag_array.valid[set][miss_way]       = 1;
        way                                        = miss_way;
    }
    miss_hit = miss_or_hit;
    //---------------------------------------
    // get next LRU for debug purpose
    get_plru(set, irt_top->irt_top_sig.next_lru_way, 1); // get LRU & set it as MRU
}
//---------------------------------------
// way - For miss -> indicate victim-way [PLRU]
//      For Hit  -> indicates matching-way [MRU]
void IRT_top::IRT_RESAMP::ReferencePage_comp(uint8_t set, int ref, uint8_t& way, uint8_t& pvalid, uint8_t& miss_hit)
{
    uint8_t miss_or_hit = 1; // 1 -> MISS  0 -> HIT
    // uint8_t hit_way     = NWAYS + 1;
    // uint8_t miss_way    = NWAYS + 1;
    //---------------------------------------
    for (int i = 0; i < NWAYS; i++) {
        if ((curr_tag_array.pagenumbers[set][i] == ref) && (curr_tag_array.valid[set][i] == 1)) { // HIT
            miss_or_hit = 0; // HIT
            way         = i;
            break;
        }
    }
    // if(miss_or_hit == 0){
    //   //set_plru(set, hit_way);   //set MRU
    //   way = hit_way;
    //   //pvalid = cache_tag_array.valid[set][hit_way];
    //} else {
    //   //get_plru(set, miss_way);   //get LRU & set it as MRU
    //   //pvalid = cache_tag_array.valid[set][hit_way];
    //   //cache_tag_array.pagenumbers[set][miss_way] = ref;
    //   //cache_tag_array.valid[set][miss_way] = 1;
    //   way = 0;
    //}
    pvalid   = 0;
    miss_hit = miss_or_hit;
}
//---------------------------------------
void IRT_top::IRT_RESAMP::gen_irm_reads(uint8_t proc_size, irt_desc_par descr)
{
    //   uint16_t rd_cnt = missQ.size();
    //   cache_tag tag;
    //   uint32_t segment;
    //   sb_struct rsb;
    //   //---------------------------------------
    //   for(int i = 0; i < missQ.size(); i++) {
    //      tag = missQ[i];
    //      //---------------------------------------
    //	   memset(&rsb, 0, sizeof(rsb));
    //      segment = (int)ceil((float)tag.Yi/(float)RESAMP_SEGMENT_SIZE);
    //      //address calclation
    //      rsb.addr = descr.image_par[IIMAGE].addr_start + ((uint64_t)tag.Xi * (uint64_t)descr.image_par[IIMAGE].Hs) +
    //      (segment * descr.image_par[IIMAGE].Ps * RESAMP_SEGMENT_SIZE);
    //      //pad calculation
    //      if(((tag.Yi * descr.image_par[IIMAGE].Ps) + RESAMP_SEGMENT_SIZE) < (descr.image_par[IIMAGE].W *
    //      descr.image_par[IIMAGE].Ps)){ //byte wise calculation
    //         rsb.lpad = 0;
    //         rsb.mpad = 0;
    //      } else {
    //         rsb.lpad = 0;
    //         rsb.mpad = ((tag.Yi * descr.image_par[IIMAGE].Ps) + RESAMP_SEGMENT_SIZE) - (descr.image_par[IIMAGE].W *
    //         descr.image_par[IIMAGE].Ps);
    //      }
    //      //metadata  TODO - IRM METADATA TO BE REVIEWED
    //      rsb.metadata.set     = tag.set;
    //      rsb.metadata.way     = tag.way;
    //      rsb.metadata.key_row = tag.key_row;
    //      rsb.metadata.key_col = tag.key_col;
    //      //---------------------------------------
    //   }
}

//------------------------------------------------------------------------------
// Resampler parser:
// take 8 coordinates from warp global array and calculate proc size
// if no valid on warp coordinate array, then don’t parse for a given call.
// do tag lookup with proc size push victims to victimQ and misses to MissQ
// calculate input addr w.r.t MissQ and send input_addr_out
//-------------------------------------------------------------------------------

void IRT_top::IRT_RESAMP::warp_parser(uint64_t&         iimage_addr,
                                      meta_data_struct& meta_out,
                                      bool&             imem_rd,
                                      uint16_t&         lsb_pad,
                                      uint16_t&         msb_pad)
{
    // IRT_TRACE_TO_LOG(2,parseFile,"======parser at cycle: %d====== \n",cycle);
    // desc access
    // uint16_t Ho = 128;//irt_top->irt_desc[desc].image_par[OIMAGE].H;
    // uint16_t Wo = 128;//irt_top->irt_desc[desc].image_par[OIMAGE].W;
    uint16_t Hi = /*128;*/ descr.image_par[IIMAGE].H;
    uint16_t Wi = /*128;*/ descr.image_par[IIMAGE].W;
    uint16_t Hm = /*128;*/ descr.image_par[MIMAGE].H;
    uint16_t Wm = /*128;*/ descr.image_par[MIMAGE].W;
    uint16_t Hs = /*512;*/ descr.image_par[IIMAGE].Hs;

    uint16_t num_warp_stripes = no_x_seg; // TODO
    uint64_t addr_start       = descr.image_par[IIMAGE].addr_start; // TODO
    uint8_t  sc_en            = space_conflict_en; // TODO replace with desc
    // uint16_t Sw = 128; //TODO replace with desc
    Eirt_bg_mode_type  bg_mode = descr.bg_mode; // TODO replace with desc
    uint16_t           cl_size = IRT_IFIFO_WIDTH / (1 << descr.image_par[IIMAGE].Ps); // 32; //TODO replace with desc
    uint16_t           in_data_type = (1 << descr.image_par[IIMAGE].Ps); // TODO replace with desc
    uint16_t           irt_mode     = descr.irt_mode;
    uint16_t           sw_proc_size = descr.proc_size;
    Eirt_int_mode_type int_mode     = descr.int_mode;

    // local varibles
    float      ri[IRT_ROT_MAX_PROC_SIZE];
    float      ci[IRT_ROT_MAX_PROC_SIZE];
    float      coord[2];
    uint32_t   i, j, k, l;
    uint8_t    num_elm = 0;
    uint8_t    p_size  = 0;
    int32_t*** tag_a;
    int32_t*** m_tag;
    int32_t**  m_tag_valid;
    uint32_t   ref;
    uint32_t   warp_el;
    uint8_t    bank_row;
    uint8_t    bank_row_set;
    uint8_t    bank_col_set;
    uint8_t    set;
    uint8_t    way;
    uint8_t    pvalid;
    uint8_t    miss_or_hit;
    bool       clip_fp           = descr.clip_fp;
    bool       clip_fp_inf_input = descr.clip_fp_inf_input;

    // local structs
    cache_tag cache_tag_int;
    cache_tag req_tag;

    float p_sat_x      = (bg_mode == e_irt_bg_in_pad) ? (float)Wi : (float)Wi - 1;
    float p_sat_y      = (bg_mode == e_irt_bg_in_pad) ? (float)Hi : (float)Hi - 1;
    float n_sat_x      = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;
    float n_sat_y      = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;
    bool  bwd_pass2_en = (irt_mode == 6) ? 1 : 0;

    // local arrays mem allocations
    tag_a       = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);
    m_tag       = mycalloc3_int(2, NUM_BANK_ROW, 2);
    m_tag_valid = mycalloc2_int(2, NUM_BANK_ROW);

    // global counters
    i = w_parser_cnt.pstripe;
    j = w_parser_cnt.pline;
    k = w_parser_cnt.pcoord;

    //---------------------------------------
    irt_top->irt_top_sig.resamp_psize     = 0;
    irt_top->irt_top_sig.resamp_psize_vld = 0;
    //---------------------------------------
    uint16_t stripe_w = resamp_wimage[i][j].size;
    if ((resamp_wimage[i][j].valid == 1) & (!parser_only_done) & ((missQ.size() < MISSQ_MAX_DEPTH))) {
        IRT_TRACE_TO_LOG(2,
                         parseFile,
                         "====================\n[stripe:row:col:comp_cnt] [%u:%u:%u:%u] at cycle %d\n",
                         w_parser_cnt.pstripe,
                         w_parser_cnt.pline,
                         w_parser_cnt.pcoord,
                         w_parser_cnt.comp_cnt,
                         cycle);
        // step1: collecting 8 coordinates
        for (l = k; l < (std::min((int)(k + sw_proc_size), (int)stripe_w)); l++) {
            num_elm   = num_elm + 1; // end of line we may not have 8 elements.
            float Tx  = fp32_nan_inf_conv(resamp_wimage[i][j].x[l]);
            float Ty  = fp32_nan_inf_conv(resamp_wimage[i][j].y[l]);
            ci[l - k] = (Tx < n_sat_x) ? n_sat_x : (Tx > p_sat_x) ? p_sat_x : Tx;
            ri[l - k] = (Ty < n_sat_y) ? n_sat_y : (Ty > p_sat_y) ? p_sat_y : Ty;
            // IRT_TRACE_TO_LOG(2,parseFile,"Tx,Ty:[%f:%f]
            // wimage[%f,%f]\n",Tx,Ty,resamp_wimage[i][j].x[l],resamp_wimage[i][j].y[l]);
        }

        // step2: calling proc size function
        p_size = proc_size_calc(ci, ri, num_elm, sc_en, descr.image_par[IIMAGE].Ps, ((descr.irt_mode == 6) ? 1 : 0));
        //      IRT_TRACE("fv_proc_size_calc p_size = %d\n",p_size);

        proc_comp_cnt         = proc_comp_cnt + 1;
        w_parser_cnt.comp_cnt = proc_comp_cnt;
        //---------------------------------------
        // accumulate statistic
        avg_proc_size += p_size;
        //---------------------------------------
        irt_top->irt_top_sig.resamp_psize     = p_size;
        irt_top->irt_top_sig.resamp_psize_vld = 1;
        //---------------------------------------
        IRT_TRACE_TO_LOG(2, parseFile, "proce size : %d\n", p_size);
        // for(int k=0; k < p_size; k++ )
        //   IRT_TRACE_TO_LOG(2,parseFile,"[(row,col):(%f , %f) \n",ri[k], ci[k] );

        // step3: converting coorinate to tags
        for (int ii = 0; ii < p_size; ii++) {
            coord[0] = ri[ii];
            coord[1] = ci[ii];
            coord_to_tag(coord, tag_a[ii], cl_size);
        }

        // step4: merging tags befoe lookup. generate tags for each bank row.
        uint8_t two_lookup_count_en;
        tag_merger(p_size, tag_a, m_tag, m_tag_valid, two_lookup_count_en);
        two_lookup_cycles = two_lookup_cycles + two_lookup_count_en;

        for (uint8_t i = 0; i < p_size; i++) {
            IRT_TRACE_TO_LOG(2, parseFile, "row,col:[%f:%f] tag_a[r,c]=", ri[i], ci[i]);
            IRT_TRACE_TO_LOG(2,
                             parseFile,
                             "[%d:%d] [%d:%d] [%d:%d] [%d:%d] \n",
                             tag_a[i][0][0],
                             tag_a[i][0][1],
                             tag_a[i][1][0],
                             tag_a[i][1][1],
                             tag_a[i][2][0],
                             tag_a[i][2][1],
                             tag_a[i][3][0],
                             tag_a[i][3][1]);
        }
        IRT_TRACE_TO_LOG(2, parseFile, "Merged-Tags : ");
        for (uint8_t j = 0; j < 8; j++) {
            if (m_tag_valid[0][j] == 1)
                IRT_TRACE_TO_LOG(2, parseFile, "[%d:%d] ", m_tag[0][j][0], m_tag[0][j][1]);
            if (m_tag_valid[1][j] == 1)
                IRT_TRACE_TO_LOG(2, parseFile, "[%d:%d] ", m_tag[1][j][0], m_tag[1][j][1]);
            if (j == 7)
                IRT_TRACE_TO_LOG(2, parseFile, "\n");
            // if(m_tag_valid[0][j]==1) IRT_TRACE_TO_LOG(2,parseFile,"m_tag[r,c]=(%d, %d) m_tag_valid=%d
            // ",m_tag[0][j][0],m_tag[0][j][1],m_tag_valid[0][j]); if(m_tag_valid[1][j]==1)
            // IRT_TRACE_TO_LOG(2,parseFile,"m_tag[r,c]=(%d, %d)
            // m_tag_valid=%d",m_tag[1][j][0],m_tag[1][j][1],m_tag_valid[1][j]);
        }

        // step5: tag lookup and pushing, total 16 tag lookups possible for a given proc size.
        for (uint8_t jj = 0; jj < 2; jj++) {
            for (uint8_t kk = 0; kk < NUM_BANK_ROW; kk++) {
                if (m_tag_valid[jj][kk] == 1) {
                    // step5a : set and key generation from tag
                    bank_row = (uint8_t)(m_tag[jj][kk][0] & (int32_t)(NUM_BANK_ROW - 1));
                    // bank_row_set =(uint8_t)
                    // (((int)floor(abs(m_tag[jj][kk][0])/NUM_BANK_ROW))%(NSETS/(NUM_BANK_ROW*2)));
                    bank_row_set     = (uint8_t)((m_tag[jj][kk][0] >> (uint8_t)log2(NUM_BANK_ROW)) &
                                             ((NSETS / (NUM_BANK_ROW * 2)) - 1));
                    bank_col_set     = (uint8_t)(m_tag[jj][kk][1] & 0x1);
                    uint32_t row_key = (((m_tag[jj][kk][0] >> 5) & 0x000001FF) << 8);
                    uint32_t col_key = ((m_tag[jj][kk][1] >> 1) & 0x000000FF);
                    set              = bank_row * (NSETS / NUM_BANK_ROW) + bank_row_set * 2 + bank_col_set;
                    ref              = (row_key | col_key);
                    // step5b : tag lookup
                    ReferencePage(set, ref, way, pvalid, miss_or_hit);
                    //---------------------------------------
                    uint16_t my_tree_bits = 0;
                    for (int ss = 0; ss < NWAYS - 1; ss++) {
                        my_tree_bits |= ((cache_tag_array.tree_bits[set][ss] & 0x1) << ss);
                    }
                    //---------------------------------------
                    lookup_cnt++;
                    if (miss_or_hit == 0) {
                        hit_cnt++;
                    }
                    // steb 5c: prepare cache tag array
                    cache_tag_int.pch          = proc_comp_cnt;
                    cache_tag_int.ref          = ref;
                    cache_tag_int.set          = set;
                    cache_tag_int.way          = way;
                    cache_tag_int.row          = m_tag[jj][kk][0];
                    cache_tag_int.col          = m_tag[jj][kk][1];
                    cache_tag_int.bank_row     = bank_row;
                    cache_tag_int.bank_row_set = bank_row_set;
                    cache_tag_int.bank_col_set = bank_col_set;
                    cache_tag_int.miss_or_hit  = miss_or_hit;
                    cache_tag_int.parser_cnt   = w_parser_cnt;

                    if (miss_or_hit == 0) {
                        IRT_TRACE_TO_LOG(2,
                                         parseFile,
                                         "PARSER:HIT::row=%u;col=%u;set=%u;ref=%d;way=%d;next_lru=%u;tree_bits=%x;pch=%"
                                         "x;pvalid=%x at cycle %d \n",
                                         cache_tag_int.row,
                                         cache_tag_int.col,
                                         set,
                                         cache_tag_int.ref,
                                         way,
                                         irt_top->irt_top_sig.next_lru_way,
                                         my_tree_bits,
                                         cache_tag_int.pch,
                                         pvalid,
                                         cycle);
                    }
                    // step5d: push previous cache tag to VictimQ if pvalid is there
                    if (miss_or_hit == 1 && pvalid == 1) { // miss for a given tag
                        victimQ.push(cache_tag_array.tags[set][way]);
                        victim_infQ.push(cache_tag_array.tags[set][way]);
                        IRT_TRACE_TO_LOG(2,
                                         parseFile,
                                         "PARSER:VICTIMQ-PUSH::row=%u;col=%u;set=%u;ref=%d;way=%d;next_lru=%d;tree_"
                                         "bits=%x;pch=%x;pvalid=%d; at cycle %d \n",
                                         cache_tag_array.tags[set][way].row,
                                         cache_tag_array.tags[set][way].col,
                                         set,
                                         cache_tag_array.tags[set][way].ref,
                                         way,
                                         irt_top->irt_top_sig.next_lru_way,
                                         my_tree_bits,
                                         cache_tag_array.tags[set][way].pch,
                                         pvalid,
                                         cycle);
                    }
                    // step5e: push current cache tag int to MissQ
                    if (miss_or_hit == 1) {
                        missQ.push(cache_tag_int);
                        IRT_TRACE_TO_LOG(2,
                                         parseFile,
                                         "PARSER:MISSQ-PUSH::row=%d;col=%d;set=%d;ref=%d;way=%d;next_lru=%d;tree_bits=%"
                                         "x;pch=%x;pvalid=%d at cycle %d \n",
                                         cache_tag_int.row,
                                         cache_tag_int.col,
                                         set,
                                         cache_tag_int.ref,
                                         way,
                                         irt_top->irt_top_sig.next_lru_way,
                                         my_tree_bits,
                                         cache_tag_int.pch,
                                         pvalid,
                                         cycle);
                    }
                    // step5f: push cache tag to cache tag array
                    cache_tag_array.tags[set][way] = cache_tag_int;
                }
            }
        }

        // end of 16 tag lookups.
        // update counters
        if (k + p_size < stripe_w) {
            w_parser_cnt.pcoord = w_parser_cnt.pcoord + p_size;
        } else {
            w_parser_cnt.pcoord = 0;
            if (j + 1 < Hm) {
                IRT_TRACE_TO_LOG(2, parseFile, "Line done for line :%d Hm: %d \n", j, Hm);
                w_parser_cnt.pline = w_parser_cnt.pline + 1;
            } else {
                w_parser_cnt.pline = 0;
                if (i + 1 < num_warp_stripes) {
                    w_parser_cnt.pstripe = w_parser_cnt.pstripe + 1;
                } else {
                    parser_only_done = 1; // all coordinates done for parsing.
                    if ((irt_mode == 4) || (irt_mode == 5)) {
                        w_parser_cnt.done = 1; // all coordinates done for parsing.
                    }
                    //---------------------------------------
                    // PUSH PADDED IRM READ FOR LAST INDICATION
                    cache_tag_int.row = -2;
                    missQ.push(cache_tag_int);
                }
                IRT_TRACE_TO_LOG(2,
                                 parseFile,
                                 "WARP_PARSER:RUNNING irt_mode %d w_parser_cnt.done set to %d :: i:j %d:%d :: "
                                 "num_warp_stripes = %d i+1 %d parser_only_done %d\n",
                                 irt_mode,
                                 w_parser_cnt.done,
                                 i,
                                 j,
                                 num_warp_stripes,
                                 i + 1,
                                 parser_only_done);
            }
        }

    } else {
        if (!(missQ.size() < MISSQ_MAX_DEPTH)) {
            num_stall_misq = num_stall_misq + 1;
        }
        // add waiting for warp data under debug trace
        // bwd pass2 flush out logic, push all tags to victimQ.
        // w_parser_cnt.done will be generated in bwd pass2 case only after pushing all tags.
        if ((parser_only_done == 1) & bwd_pass2_en & (w_parser_cnt.done == 0)) {
            if (cache_tag_array.valid[(bank_cnt * 8) + set_cnt][way_cnt] == 1) {
                victimQ.push(cache_tag_array.tags[(bank_cnt * 8) + set_cnt][way_cnt]);
                victim_infQ.push(cache_tag_array.tags[(bank_cnt * 8) + set_cnt][way_cnt]);
                //---------------------------------------
                IRT_TRACE_TO_LOG(
                    2,
                    parseFile,
                    "PARSER:CACHE-FLUSH:VICTIMQ-PUSH::pch:set:way:tag:bankId:valid = %x:%d:%d:%d:%d:%d at cycle %d \n",
                    cache_tag_array.tags[(bank_cnt * 8) + set_cnt][way_cnt].pch,
                    (bank_cnt * 8) + set_cnt,
                    way_cnt,
                    cache_tag_array.tags[(bank_cnt * 8) + set_cnt][way_cnt].ref,
                    cache_tag_array.tags[(bank_cnt * 8) + set_cnt][way_cnt].bank_row,
                    cache_tag_array.valid[(bank_cnt * 8) + set_cnt][way_cnt],
                    cycle);
            }
            //---------------------------------------
            // Flush sequence -> B0S0W0 - B7S0W0 - B0S0W1 - B7S0W1 .. B0S0W15 - B7S0W15 - B0S1W0 - B7S1W0 - B0S1W1 -
            // B7S1W1 ..
            if ((bank_cnt + 1) < NUM_BANK_ROW) {
                bank_cnt++;
            } else {
                if ((way_cnt + 1) < NWAYS) {
                    way_cnt++;
                    bank_cnt = 0;
                } else {
                    if ((set_cnt + 1) < NSETS_PER_BANK) {
                        set_cnt++;
                        bank_cnt = 0;
                        way_cnt  = 0;
                    } else {
                        set_cnt           = 0;
                        bank_cnt          = 0;
                        way_cnt           = 0;
                        w_parser_cnt.done = 1;
                    }
                }
            }
            IRT_TRACE_TO_LOG(2,
                             parseFile,
                             "CACHE-FLUSH:VICTIMQ-PUSH::set:way:bankId:done = %d:%d:%d:%d @cycle %d\n",
                             set_cnt,
                             way_cnt,
                             bank_cnt,
                             w_parser_cnt.done,
                             cycle);
        }
        // end of victq flush logic.
    }
    // step6: pop missQ and generate IM read addr with padding.
    irt_top->irt_top_sig.irm_rd_first[0] = 0;
    irt_top->irt_top_sig.irm_rd_last[0]  = 0;
    if (!missQ.empty()) {
        num_req               = num_req + 1;
        req_tag               = missQ.front();
        int32_t num_input_seg = (int32_t)ceil((float)Wi / (float)cl_size);
        // if address is -ve generate pad data req
        // if(req_tag.row < 0 | req_tag.col <0 | req_tag.row > (Hi-1) | req_tag.col > (Wi-1) | bwd_pass2_en)
        if ((req_tag.row < 0) | (req_tag.col < 0) | (req_tag.row > (Hi - 1)) | ((req_tag.col * cl_size) > (Wi - 1)) |
            bwd_pass2_en) {
            lsb_pad = 128;
            msb_pad = 128;
            imem_rd = 1;
            // iimage_addr = bwd_pass2_en ? 0 : addr_start + (req_tag.row * Hs) + req_tag.col*128;
            iimage_addr = addr_start + (abs(req_tag.row) * Hs) + abs(req_tag.col) * 128;
        } else {
            // actual address
            iimage_addr = addr_start + (req_tag.row * Hs) + req_tag.col * 128;
            imem_rd     = 1;
            lsb_pad     = 0;
            msb_pad     = 0;
            // msb padding incase towards the end of each input line.
            if (req_tag.col == num_input_seg - 1)
                msb_pad = (num_input_seg * 128) - Wi * in_data_type;
        }
        //---------------------------------------
        irt_top->irt_top_sig.irm_rd_first[0] = resamp_irm_first_rd;
        resamp_irm_first_rd                  = 0;
        //---------------------------------------
        IRT_TRACE_TO_LOG(2,
                         parseFile,
                         "IRM_ADDR - req_tag[row:col]=[%d:%d] :iimage_addr = %llx lpad:mpad = %d:%d\n",
                         req_tag.row,
                         req_tag.col,
                         iimage_addr,
                         lsb_pad,
                         msb_pad);
        // forming meta data to send out.
        meta_out.bank_row = req_tag.bank_row;
        meta_out.set      = req_tag.set;
        meta_out.way      = req_tag.way;
        meta_out.tag      = req_tag.ref;
        // below elements only for debug
        meta_out.task       = irt_top->task[e_resamp_wread];
        meta_out.parser_cnt = req_tag.parser_cnt;
        //---------------------------------------
        missQ.pop();
        if (missQ.empty() & parser_only_done) {
            parser_and_imread_done              = 1;
            irt_top->irt_top_sig.irm_rd_last[0] = 1;
        }
    } else {
        imem_rd           = 0;
        iimage_addr       = 0;
        meta_out.bank_row = 0;
        meta_out.set      = 0;
        meta_out.way      = 0;
        meta_out.tag      = 0;
        lsb_pad           = 0;
        msb_pad           = 0;
        // below elements only for debug
        meta_out.stripe = 0;
        meta_out.line   = 0;
        meta_out.coord  = 0;
    }
    // end addr generation
    myfree3_int(tag_a, IRT_ROT_MAX_PROC_SIZE, 4, 2);
    myfree3_int(m_tag, 2, NUM_BANK_ROW, 2);
    myfree2_int(m_tag_valid, 2, NUM_BANK_ROW);
}
//----------------------end of parser--------------------------------------------

//---------------------------------------
// load
void IRT_top::IRT_RESAMP::fill_ctrl(const bus128B_struct& iimage_rd_data,
                                    meta_data_struct      iimage_meta_in,
                                    bool                  imem_rd_data_valid)
{

    sb_struct       line;
    resamp_rmwm_inf rmwm_inf;
    uint16_t        in_data_type = (1 << descr.image_par[IIMAGE].Ps);
    uint16_t        irt_mode     = descr.irt_mode;
    uint8_t         inp_line_data[128];
    std::string     my_string = getInstanceName() + ":FILL_CTRL:";
    uint8_t         bwd2_en   = (irt_mode == 6) ? 1 : 0;

    //---------------------------------------
    if (imem_rd_data_valid) {
        line.data     = iimage_rd_data;
        line.metadata = iimage_meta_in;
        inputQ.push(line);
        IRT_TRACE_TO_LOG(2,
                         fillFile,
                         "inputQ.push -> [task:stripe:line:coord:set:tag:way:last]=[%d:%d:%d:%d:%d:%d:%d:%d] :: "
                         "proc_comp_cnt %d :: cycle %d  data = ",
                         iimage_meta_in.parser_cnt.task,
                         iimage_meta_in.parser_cnt.pstripe,
                         iimage_meta_in.parser_cnt.pline,
                         iimage_meta_in.parser_cnt.pcoord,
                         line.metadata.set,
                         line.metadata.tag,
                         line.metadata.way,
                         line.metadata.last,
                         line.metadata.parser_cnt.comp_cnt,
                         cycle);
        for (int i = 128; i > 0; i--) {
            IRT_TRACE_TO_LOG(2, fillFile, "%x,", line.data.pix[i - 1]);
        }
        IRT_TRACE_TO_LOG(2, fillFile, "cycle %d \n", cycle);
    }
    // take the input data and put it in Rotation memory based on Valid bit //update tag in current tag array.
    // IRT_TRACE_TO_LOG(2,fillFile,"inputQ.size %d empy %d \n",inputQ.size(),inputQ.empty());
    if (!inputQ.empty()) {
        sb_struct inp_line = inputQ.front();
        // IRT_TRACE_TO_LOG(2,fillFile,"inputQ.front ->
        // [task:stripe:line:coord:set:tag:way:valid]=[%d:%d:%d:%d:%d:%d:%d:%d] :: proc_comp_cnt %d :: cycle %d \n",
        // inp_line.metadata.parser_cnt.task, inp_line.metadata.parser_cnt.pstripe, inp_line.metadata.parser_cnt.pline,
        // inp_line.metadata.parser_cnt.pcoord,inp_line.metadata.set,inp_line.metadata.tag,inp_line.metadata.way,curr_tag_array.valid[inp_line.metadata.set][inp_line.metadata.way],
        // inp_line.metadata.parser_cnt.comp_cnt,cycle);
        // TODO Ambili make tag array 3d [bnkrows][sets][ways]
        if (curr_tag_array.valid[inp_line.metadata.set][inp_line.metadata.way] == 0) {
            resamp_rot_mem[inp_line.metadata.set][inp_line.metadata.way]             = inp_line.data;
            curr_tag_array.pagenumbers[inp_line.metadata.set][inp_line.metadata.way] = inp_line.metadata.tag;
            curr_tag_array.valid[inp_line.metadata.set][inp_line.metadata.way]       = 1;
            // XXXXXXXXXXXXXXXXXXROT mem write dump XXXXXXXXXXXXXXXXXXXXXXXXX
            // interleave line if the input data type is int8.
            uint16_t tmpi = 0;
            for (uint16_t i = 0; i < (128 / 2); i++) {
                if (in_data_type == 1) {
                    inp_line_data[tmpi]     = inp_line.data.pix[i];
                    inp_line_data[tmpi + 1] = inp_line.data.pix[64 + i];
                } else {
                    inp_line_data[tmpi]     = inp_line.data.pix[tmpi];
                    inp_line_data[tmpi + 1] = inp_line.data.pix[tmpi + 1];
                }
                tmpi = tmpi + 2;
            }
            //**********************************************
            // XXXXXXXXXXXXXXXXX Start of ROT mem write dump XXXXXXXXXXXXXXXXXXXXX
            // generating data,addr, valid.
            uint8_t num_lines = bwd2_en ? 2 : 4; // 1 $line occupies how many sram rows
            uint8_t c_set     = (inp_line.metadata.set & (uint8_t)(NUM_BANK_ROW - 1));
            uint8_t m_set     = bwd2_en ? (c_set << 1) + (uint8_t)(inp_line.metadata.bank_row >> 2) : c_set;
            for (uint8_t ii = 0; ii < num_lines; ii++) {
                for (uint8_t jj = 0; jj < 32; jj++) {
                    rmwm_inf.data[jj] = inp_line_data[ii * 32 + jj];
                }
                for (uint8_t kk = 0; kk < 4; kk++) {
                    rmwm_inf.addr[kk] = ((uint32_t)(num_lines * NWAYS) * (uint32_t)m_set) +
                                        ((uint32_t)inp_line.metadata.way * num_lines) + ii;
                    rmwm_inf.valid[kk] = 1;
                }
                rmwm_inf.write_type = 1; // rmwm-write
                // irt_top->irt_top_sig.rmwm_infQ[inp_line.metadata.bank_row].push(rmwm_inf);
                if (inp_line.metadata.last == 0) {
                    if (bwd2_en)
                        rmwm_infQ[inp_line.metadata.bank_row].push(rmwm_inf);
                    rmwm_infQ[inp_line.metadata.bank_row ^ (bwd2_en << 2)].push(rmwm_inf);
                }
            }
            // XXXXXXXXXXXXXXXXX end of ROT mem write dump XXXXXXXXXXXXXXXXXXXXX
            //**********************************************
            inputQ.pop();
            IRT_TRACE_TO_LOG(2,
                             fillFile,
                             "inputQ.pop -> [set:tag:way]=[%d:%d:%d] :: cycle %d \n",
                             inp_line.metadata.set,
                             inp_line.metadata.tag,
                             inp_line.metadata.way,
                             cycle);
        }
    }
}

void IRT_top::IRT_RESAMP::victim_ctrl()
{
    resamp_rmrm_inf rmrm_inf;
    memset(&rmrm_inf, 0, sizeof(resamp_rmrm_inf));
    int Psi = (1 << descr.image_par[IIMAGE].Ps);
    //---------------------------------------
    // Pop VICTIMQ and compare with compute loop and invalidate data
    if (!victimQ.empty()) {
        cache_tag curr_victim = victimQ.front();
        // IRT_TRACE_TO_LOG(2,fillFile,"VICTIM_CTRL :: curr_victim : [set:way:ref:pch] = [%d:%d:%d:%d] curr_compute_cnt
        // %d cycle %d\n",curr_victim.set,curr_victim.way,curr_victim.ref,curr_victim.pch,curr_compute_cnt,cycle);
        // if(victim_ctrl_ToQ_print) IRT_TRACE_TO_LOG(2,fillFile,"PARSER-VICTIMQ-ToQ::row:col:set:ref:way:pch=
        // %d:%d:%d:%d:%d:%x curr_compute_cnt %d at cycle %d
        // \n",curr_victim.row,curr_victim.col,curr_victim.set,curr_victim.ref,curr_victim.way,curr_victim.pch,curr_compute_cnt,cycle);
        victim_ctrl_ToQ_print = 0;
        if ((curr_victim.pch) <= curr_compute_cnt) { // TODO - REVIEW WITH PRADEEP.. changed it to <= due to BWD2
            sb_struct line;
            curr_tag_array.valid[curr_victim.set][curr_victim.way] = 0;
            // IRT_TRACE_TO_LOG(2,fillFile,"VICTIM_CTRL :: curr_victim - POP : [set:way:pch] = [%d:%d:%d]
            // curr_compute_cnt %d cycle %d\n",curr_victim.set,curr_victim.way, curr_victim.pch,curr_compute_cnt,cycle);
            // IRT_TRACE_TO_LOG(2,fillFile,"VICTIM_CTRL-VICTIMQ-POP::set:way:ref:pch: = %d:%d:%d:%d curr_compute_cnt %d
            // at cycle %d \n",curr_victim.set,curr_victim.way,curr_victim.ref,curr_victim.pch,curr_compute_cnt,cycle);
            IRT_TRACE_TO_LOG(
                2,
                fillFile,
                "PARSER-VICTIMQ-POP::row:col:set:ref:way:pch= %d:%d:%d:%d:%d:%x curr_compute_cnt %d at cycle %d \n",
                curr_victim.row,
                curr_victim.col,
                curr_victim.set,
                curr_victim.ref,
                curr_victim.way,
                curr_victim.pch,
                curr_compute_cnt,
                cycle);
            victimQ.pop();
            num_victims           = num_victims + 1; // used for print of perf data only.
            victim_ctrl_ToQ_print = 1;
            if (descr.irt_mode == e_irt_resamp_bwd2) {
                uint16_t cl_size = IRT_IFIFO_WIDTH / Psi;
                // if(!(curr_victim.row < 0 | curr_victim.row > (descr.image_par[IIMAGE].H-1) | curr_victim.col < 0 |
                // (curr_victim.col*cl_size) > (descr.image_par[IIMAGE].W-1))){//drop VictimQ for OoB segments
                bool is_segment_oob = (curr_victim.row < 0) | (curr_victim.row > (descr.image_par[IIMAGE].H - 1)) |
                                      (curr_victim.col < 0) |
                                      ((curr_victim.col * cl_size) > (descr.image_par[IIMAGE].W - 1));
                line.data = resamp_rot_mem[curr_victim.set][curr_victim.way];
                // IRT_TRACE_TO_LOG(2,fillFile,"VICTIM_CTRL: EN");
                for (int i = 0; i < 128; i++) {
                    if (is_segment_oob |
                        (((curr_victim.col * cl_size * Psi) + i) >=
                         (descr.image_par[IIMAGE].W *
                          Psi))) { // segment is out-of-image OR last segment can have PAD bytes if (Wi % 128) != 0
                        line.data.en[i] = 0;
                    } else {
                        line.data.en[i] = 1;
                    }
                    // IRT_TRACE_TO_LOG(2,fillFile," %d, ",line.data.en[i]);
                }
                // IRT_TRACE_TO_LOG(2,fillFile,"\n");
                // IRT_TRACE_TO_LOG(2,fillFile,"VICTIM_CTRL: = is_segment_oob %d col %d cl_Size %d col*cl_size %d Wi %d
                // at cycle %d
                // \n",is_segment_oob,curr_victim.col,cl_size,(curr_victim.col*cl_size),descr.image_par[IIMAGE].W,cycle);
                line.metadata.bank_row = curr_victim.bank_row;
                line.metadata.set      = curr_victim.set;
                line.metadata.way      = curr_victim.way;
                line.metadata.tag      = curr_victim.ref;
                line.metadata.row      = curr_victim.row;
                line.metadata.col      = curr_victim.col;
                outputQ.push(line);
                //            }
            }
        }
    }
}

//---------------------------------------
// temp function to keep pushing random number of compute bytes
// bool IRT_top::IRT_RESAMP::compute(){
//	std::string my_string = getInstanceName() + ":COMPUTE:";
//   uint8_t pix_val, proc_size;
//   //---------------------------------------
//   proc_size = (rand() % 8) + 1;
//	IRT_TRACE("%s proc_size = %d pix = ",my_string.c_str(),proc_size);
//   //---------------------------------------
//   for(int i=0; i < proc_size; i++){
//      for(int j=0; j < (1<<descr.image_par[OIMAGE].Ps); j++){
//         pix_val = rand() & 0xff;
//         owmDataQ.push(pix_val);
//		   IRT_TRACE("%x,",pix_val);
//      }
//   }
//	IRT_TRACE("\n");
//}

void IRT_top::IRT_RESAMP::compute()
{

    // desc access
    // uint16_t Ho = 128;//irt_top->irt_desc[desc].image_par[OIMAGE].H;
    // uint16_t Wo = 128;//irt_top->irt_desc[desc].image_par[OIMAGE].W;
    uint16_t Hi = /*128;*/ descr.image_par[IIMAGE].H;
    uint16_t Wi = /*128;*/ descr.image_par[IIMAGE].W;
    uint16_t Hm = /*128;*/ descr.image_par[MIMAGE].H;
    uint16_t Wm = /*128;*/ descr.image_par[MIMAGE].W;

    uint16_t num_warp_stripes = no_x_seg; // TODO
    uint8_t  sc_en            = space_conflict_en; // TODO replace with desc
    // uint16_t Sw = 128; //TODO replace with desc
    uint16_t bg_mode       = descr.bg_mode; // TODO replace with desc
    uint16_t cl_size       = IRT_IFIFO_WIDTH / (1 << descr.image_par[IIMAGE].Ps); // 32; //TODO replace with desc
    uint16_t rot_mem_w_elm = 32 / (1 << descr.image_par[IIMAGE].Ps); // 32; //TODO replace with desc
    uint16_t in_data_type  = (1 << descr.image_par[IIMAGE].Ps); // TODO replace with desc
    uint16_t MSI           = descr.Msi;
    uint16_t irt_mode      = descr.irt_mode;
    uint16_t out_data_type = (irt_mode != 5) ? (1 << descr.image_par[OIMAGE].Ps)
                                             : (1 << (descr.image_par[OIMAGE].Ps - 1)); // TODO replace with desc
    Eirt_resamp_dtype_enum in_dtype          = descr.image_par[IIMAGE].DataType;
    Eirt_resamp_dtype_enum out_dtype         = descr.image_par[OIMAGE].DataType;
    Eirt_resamp_dtype_enum grad_dtype        = descr.image_par[GIMAGE].DataType;
    Eirt_int_mode_type     int_mode          = descr.int_mode;
    int16_t                bli_shift         = descr.bli_shift;
    uint16_t               MAX_VALo          = descr.MAX_VALo;
    uint16_t               Ppo               = descr.Ppo;
    bool                   clip_fp           = descr.clip_fp;
    bool                   clip_fp_inf_input = descr.clip_fp_inf_input;
    bool                   ftz_en            = descr.ftz_en;
    uint16_t               sw_proc_size      = descr.proc_size;
    std::string            my_string         = getInstanceName() + ":COMPUTE:";

    // local varibles
    float      ri[IRT_ROT_MAX_PROC_SIZE];
    float      ci[IRT_ROT_MAX_PROC_SIZE];
    float      grad_out[IRT_ROT_MAX_PROC_SIZE];
    float      coord[2];
    uint32_t   i, j, k, l;
    uint8_t    num_elm = 0;
    uint8_t    p_size  = 0;
    int32_t*** tag_a;
    int32_t*** m_tag;
    int32_t**  m_tag_valid;
    uint32_t   ref;
    uint32_t   cmp_ref;
    uint8_t    bank_row;
    uint8_t    cmp_bank_row;
    uint8_t    bank_row_set;
    uint8_t    bank_col_set;
    uint8_t    set;
    uint8_t    cmp_set;
    uint8_t    way;
    uint8_t    cmp_way;
    uint8_t    pvalid;
    uint8_t    miss_or_hit;
    uint8_t    cmp_miss_or_hit;
    uint32_t   l_miss = 0;
    int32_t    f_tag[2];
    int32_t    cmp_tag[2];
    float      im_P[4];
    float      g_out[4];
    uint32_t   im_Phex[4];
    float      p_out[IRT_ROT_MAX_PROC_SIZE];
    float      w_out_r[IRT_ROT_MAX_PROC_SIZE];
    float      w_out_c[IRT_ROT_MAX_PROC_SIZE];
    uint8_t    way_offset_row;
    uint8_t    way_offset_col;
    uint8_t    way_offset_col2;
    // local structs
    // cache_tag cache_tag_int;
    // cache_tag req_tag;
    bus128B_struct rot_mem_data;

    float p_sat_x     = (bg_mode == e_irt_bg_in_pad) ? (float)Wi : (float)Wi - 1;
    float p_sat_y     = (bg_mode == e_irt_bg_in_pad) ? (float)Hi : (float)Hi - 1;
    float n_sat_x     = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;
    float n_sat_y     = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;
    bool  bwd_pass_en = ((irt_mode == 5) | (irt_mode == 6)) ? 1 : 0;
    bool  bwd2_en     = (irt_mode == 6) ? 1 : 0;
    // local arrays mem allocations
    tag_a       = mycalloc3_int(IRT_ROT_MAX_PROC_SIZE, 4, 2);
    m_tag       = mycalloc3_int(2, NUM_BANK_ROW, 2);
    m_tag_valid = mycalloc2_int(2, NUM_BANK_ROW);

    // global counters
    i                 = comp_parser_cnt.pstripe;
    j                 = comp_parser_cnt.pline;
    k                 = comp_parser_cnt.pcoord;
    uint16_t stripe_w = resamp_wimage[i][j].size;
    //---------------------------------------
    // IRT_TRACE_TO_LOG(5,IRTLOG,"SAGAR : compute thread compute_timeout = %0d
    // %0d\n",compute_timeout,MAX_COMPUTE_STALL);
    /*
    if (compute_timeout > MAX_COMPUTE_STALL) {
       //IRT_TRACE_TO_LOG(5,IRTLOG,"SAGAR : inside condition - compute thread compute_timeout = %0d
    %0d\n",compute_timeout,MAX_COMPUTE_STALL); IRT_TRACE_TO_LOG(5,IRTLOG,"Error - %s: compute timed-out :: Either
    warp-data not received or All-Cache-Hit not hit  %d\n", my_string.c_str(), cycle); IRT_TRACE_TO_RES(test_res, "Error
    - %s: compute timed-out :: Either warp-data not received or All-Cache-Hit not hit \n", my_string.c_str());
        IRT_CLOSE_FAILED_TEST(0);
    }
    */
    //---------------------------------------
    if ((resamp_wimage[i][j].valid == 1) & (!comp_parser_cnt.done) &
        ((!bwd_pass_en) || ((resamp_gimage[i][j].valid == 1) & bwd_pass_en))) {
        ////---------------------------------------
        // step1: collecting 8 coordinates
        for (l = k; l < (std::min((int)(k + sw_proc_size), (int)stripe_w)); l++) {
            num_elm         = num_elm + 1; // end of line we may not have 8 elements.
            float Tx        = fp32_nan_inf_conv(resamp_wimage[i][j].x[l]);
            float Ty        = fp32_nan_inf_conv(resamp_wimage[i][j].y[l]);
            ci[l - k]       = (Tx < n_sat_x) ? n_sat_x : (Tx > p_sat_x) ? p_sat_x : Tx;
            ri[l - k]       = (Ty < n_sat_y) ? n_sat_y : (Ty > p_sat_y) ? p_sat_y : Ty;
            grad_out[l - k] = resamp_gimage[i][j].x[l];
            // IRT_TRACE_TO_LOG(6,compFile," gimage[%d:%d:%d] = grad_out[%d] = %f ",i,j,l,l-k,grad_out[l-k]);
            // IRT_TRACE_TO_LOG(6,compFile,"T[%f:%f] = C[%f:%f] ",Tx,Ty,ci[l-k],ri[l-k]);
        }
        // IRT_TRACE_TO_LOG(2,compFile," \n");

        // step2: calling proc size function
        // uint16_t inps = descr.image_par[IIMAGE].Ps;
        // p_size= proc_size_calc(ci,ri,num_elm,sc_en);
        p_size = proc_size_calc(ci, ri, num_elm, sc_en, descr.image_par[IIMAGE].Ps, ((descr.irt_mode == 6) ? 1 : 0));
        //      IRT_TRACE("fv_proc_size_calc p_size = %d\n",p_size);
        //---------------------------------------
        // step3: converting coorinate to tags
        for (int ii = 0; ii < p_size; ii++) {
            coord[0] = ri[ii];
            coord[1] = ci[ii];
            coord_to_tag(coord, tag_a[ii], cl_size);
        }

        // step4: merging tags befoe lookup. generate tags for each bank row.
        uint8_t two_lookup_count_drop;
        tag_merger(p_size, tag_a, m_tag, m_tag_valid, two_lookup_count_drop);
        // step5: tag lookup and total 16 tag lookups possible for a given proc size.
        l_miss = 0;

        // for(uint8_t jj=0; jj<2; jj++)
        //  for(uint8_t kk=0; kk<NUM_BANK_ROW; kk++)
        //     if(m_tag_valid[jj][kk]==1)IRT_TRACE_TO_LOG(2,compFile,"m_tag[%0d][%0d][1,0]: %0d:%0d
        //     \n",jj,kk,m_tag[jj][kk][0],m_tag[jj][kk][1]);
        for (uint8_t jj = 0; jj < 2; jj++) {
            for (uint8_t kk = 0; kk < NUM_BANK_ROW; kk++) {
                if (m_tag_valid[jj][kk] == 1) {
                    // step5a : set and key generation from tag
                    f_tag[0] = m_tag[jj][kk][0];
                    f_tag[1] = m_tag[jj][kk][1];
                    ref_gen(f_tag, bank_row, set, ref);
                    // tag lookup
                    ReferencePage_comp(set, ref, way, pvalid, miss_or_hit);
                    if (miss_or_hit == 1) { // miss
                        l_miss = l_miss + 1;
                        // IRT_TRACE_TO_LOG(2,compFile,"Miss for row,col: %d,%d tag :%d at cycle
                        // %d\n",f_tag[0],f_tag[1],ref,cycle);
                    }
                }
            }
        }
        // end of 16 tag lookups.

        // Time conflict Stall check for BWD2 case
        if ((TC_EN == 1) & bwd2_en) {
            queue<uint32_t> temp;
            float           c_coord[2];
            int32_t         pos;
            uint32_t        tc_bli_match = 0;
            uint32_t        tc_match     = 0;
            for (int tqi = 0; tqi < p_size; tqi++) {
                c_coord[0] = ri[tqi];
                c_coord[1] = ci[tqi];
                coord_to_2x2_window(c_coord, c_bli_win, tqi);
            }
            if (p_bli_v == 0) {
                copy_bli_win(c_bli_win, p_bli_win, p_size);
                prev_proc_size = p_size;
                p_bli_v        = 1;
            } else if (tc1_stall == 0) {
                // compare with prev proc size starting with n-1
                tc_bli_match = match_bli_win(c_bli_win, p_bli_win, p_size, prev_proc_size, (int32_t)TC_1_EN);
                uint32_t stall1 =
                    (tc_bli_match > 0) ? ((tc_bli_match == 1) ? (uint32_t)TC_1_PIPE : (uint32_t)TC_1_PIPE2) : 0;
                // compare with prev proc size starting from n-2
                for (int sci = 0; sci < p_size; sci++) {
                    tc_match = matchq(proc_q, c_bli_win, sci, pos);
                    if (tc_match == 1)
                        break;
                }
                uint32_t stall2    = (tc_match == 1) ? ((uint32_t)P_LATENCY - pos) : 0;
                tc1_stall          = max(stall1, stall2);
                num_tc_stall_cycle = num_tc_stall_cycle + tc1_stall;
                num_nm1_conflicts  = ((stall2 == 0) & (stall1 > 0)) ? num_nm1_conflicts + 1 : num_nm1_conflicts;
                num_nm2_conflicts  = num_nm2_conflicts + ((stall2 > 0) ? 1 : 0);

                // collect prev proc size data in queue
                for (uint8_t qi1 = 0; qi1 < prev_proc_size; qi1++) {
                    for (uint8_t qi2 = 0; qi2 < 4; qi2++) {
                        temp.push(p_bli_win[qi1][qi2][0]);
                        temp.push(p_bli_win[qi1][qi2][1]);
                    }
                }
                if (proc_q.size() == P_LATENCY)
                    proc_q.pop();
                proc_q.push(temp);
                copy_bli_win(c_bli_win, p_bli_win, p_size);
                prev_proc_size = p_size;
            } else {
                tc1_stall = tc1_stall - 1;
                if (proc_q.size() == P_LATENCY)
                    proc_q.pop();
                proc_q.push(dummy_q);
            }
        }

        // End of Time conflict stall check.

        if ((l_miss == 0) & (tc1_stall == 0)) // all hit or no stall condition for BWD2
        {
            //---------------------------------------
            IRT_TRACE_TO_LOG(20, compFile, "==================================\n");
            IRT_TRACE_TO_LOG(20,
                             compFile,
                             "comp_parser_cnt[stripe:h:w:proc_size:curr_compute_cnt] [%d:%d:%d:%d:%d] Wimage.valid=%d "
                             "Gimage.valid=%d  @cycle %d\n",
                             comp_parser_cnt.pstripe,
                             comp_parser_cnt.pline,
                             comp_parser_cnt.pcoord,
                             p_size,
                             curr_compute_cnt,
                             resamp_wimage[i][j].valid,
                             resamp_gimage[i][j].valid,
                             cycle);
            //---------------------------------------
            compute_timeout          = 0;
            resamp_rmrm_inf rmrm_inf;
            memset(&rmrm_inf, 0, sizeof(resamp_rmrm_inf));
            resamp_2x2_inf p2x2_inf; // = {0};
            memset(&p2x2_inf, 0, sizeof(resamp_2x2_inf));

            curr_compute_cnt = curr_compute_cnt + 1;

            IRT_TRACE_TO_LOG(15, compFile, "ALL-HITS :: PROC_SIZE = %d at cycle %d\n", p_size, cycle);
            // doing compute after all hit comes, one pixel at a time.
            p2x2_inf.proc_size = p_size;
            //---------------------------------------
            // for(uint8_t p1=0; p1<p_size; p1++){
            //   for (uint8_t p2=0; p2<4; p2++){
            //      IRT_TRACE_TO_LOG(5,compFile,"tag_a[%d][%d]\n ",index,fx,p2,cl_size);
            //   }
            //}
            //---------------------------------------
            for (uint8_t p1 = 0; p1 < p_size; p1++) {
                float    Wx           = ci[p1];
                uint64_t Wx_int       = (uint64_t)rint(pow(2.0, 31) * Wx);
                uint32_t xf_int       = Wx_int & 0x7FFFFFFF;
                uint32_t one_m_xf_int = (uint32_t)((uint64_t)pow(2.0, 31) - xf_int);
                float    Wy           = ri[p1];
                uint64_t Wy_int       = (uint64_t)rint(pow(2.0, 31) * Wy);
                uint32_t yf_int       = Wy_int & 0x7FFFFFFF;
                uint32_t one_m_yf_int = (uint32_t)((uint64_t)pow(2.0, 31) - yf_int);
                int      fx           = (int)floor(Wx);
                int      fy           = (int)floor(Wy);
                int      cx           = (int)ceil(Wx); // fx+1;//UNUSED FOR NOW
                int      cy           = (int)ceil(Wy); // fy+1;//UNUSED FOR NOW
                float    xf           = (float)((double)xf_int / (double)pow(2.0, 31));
                float    yf           = (float)((double)yf_int / (double)pow(2.0, 31));
                float    one_m_xf     = (float)((double)one_m_xf_int / (double)pow(2.0, 31));
                float    one_m_yf     = (float)((double)one_m_yf_int / (double)pow(2.0, 31));
                //---------------------------------------
                // Nearest neighbour weight adjustment
                // TODO - Weight for 0.5
                if (int_mode == e_irt_int_nearest) {
                    if (xf > 0.5) {
                        xf       = (float)1.0;
                        xf_int   = 0x80000000;
                        one_m_xf = (float)0.0;
                    } // xf_int updated to align PSEL internal checker
                    if (xf < 0.5) {
                        xf       = (float)0.0;
                        xf_int   = 0x0;
                        one_m_xf = (float)1.0;
                    }
                    if (yf > 0.5) {
                        yf       = (float)1.0;
                        yf_int   = 0x80000000;
                        one_m_yf = (float)0.0;
                    } // xf_int updated to align PSEL internal checker
                    if (yf < 0.5) {
                        yf       = (float)0.0;
                        yf_int   = 0x0;
                        one_m_yf = (float)1.0;
                    }
                    // if(xf_int > 0x40000000){ xf = (float)1.0; xf_int = 0x80000000;} //xf_int updated to align PSEL
                    // internal checker if(xf_int < 0x40000000){ xf = (float)0.0; xf_int = 0x0;} if(yf_int >
                    // 0x40000000){ yf = (float)1.0; yf_int = 0x80000000;} if(yf_int < 0x40000000){ yf = (float)0.0;
                    // yf_int = 0x0;}
                }
                rmrm_inf.xf[p1] = IRT_top::IRT_UTILS::conversion_float_bit(
                    xf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0);
                rmrm_inf.yf[p1] = IRT_top::IRT_UTILS::conversion_float_bit(
                    yf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0);
                p2x2_inf.weights[p1][0] = xf_int;
                p2x2_inf.weights[p1][1] = yf_int;
                p2x2_inf.grad[p1]       = IRT_top::IRT_UTILS::conversion_float_bit(
                    grad_out[p1], grad_dtype, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0);
                // collecting 2x2 window for BLI compute
#ifdef RUN_WITH_SV
                //---------------------------------------
                // DEBUG INFO FOR SV
                // IF BWD2 - each pixel is written back to memory individually, instead of whole 2x2 windows together.
                // this causes issue for p2x2 checker for input pixel compare
                if (irt_mode == 6) {
                    for (uint8_t p2 = 0; p2 < 4; p2++) {
                        cmp_tag[0] = tag_a[p1][p2][0];
                        cmp_tag[1] = tag_a[p1][p2][1];
                        ref_gen(cmp_tag, cmp_bank_row, cmp_set, cmp_ref);
                        ReferencePage_comp(cmp_set, cmp_ref, cmp_way, pvalid, cmp_miss_or_hit);
                        rot_mem_data           = resamp_rot_mem[cmp_set][cmp_way];
                        int      fx_sat        = (xf == 0) /*||((fx+(p2%2))<n_sat_x))*/ ? fx : (fx + (p2 % 2));
                        uint32_t index         = fx_sat & (cl_size - 1);
                        uint32_t in_pix_in_mem = 0;
                        //---------------------------------------
                        for (uint8_t p3 = 0; p3 < in_data_type; p3++) {
                            uint32_t tmp_in = rot_mem_data.pix[(index * in_data_type) + p3];
                            in_pix_in_mem   = in_pix_in_mem | (tmp_in << 8 * p3);
                        }
                        if (in_data_type == 2) {
                            in_pix_in_mem &= MSI;
                        }
                        p2x2_inf.pix_vld[p1] = 1;
                        p2x2_inf.pix[p1][p2] = in_pix_in_mem;
                    }
                    IRT_TRACE_TO_LOG(15, compFile, "\n");
                }
#endif
                for (uint8_t p2 = 0; p2 < 4; p2++) {
                    cmp_tag[0] = tag_a[p1][p2][0];
                    cmp_tag[1] = tag_a[p1][p2][1];
                    ref_gen(cmp_tag, cmp_bank_row, cmp_set, cmp_ref);
                    ReferencePage_comp(cmp_set, cmp_ref, cmp_way, pvalid, cmp_miss_or_hit);
                    rot_mem_data           = resamp_rot_mem[cmp_set][cmp_way];
                    int      fx_sat        = (xf == 0) /*||((fx+(p2%2))<n_sat_x))*/ ? fx : (fx + (p2 % 2));
                    uint32_t index         = fx_sat & (cl_size - 1);
                    uint32_t in_pix_in_mem = 0;
                    // IRT_TRACE_TO_LOG(15,compFile,"index %d fx = %d p2 %d cl_size %d\n ",index,fx,p2,cl_size);
                    for (uint8_t p3 = 0; p3 < in_data_type; p3++) {
                        // uint32_t tmp_in = rot_mem_data.pix[index*4 + p3];
                        uint32_t tmp_in = rot_mem_data.pix[(index * in_data_type) + p3];
                        in_pix_in_mem   = in_pix_in_mem | (tmp_in << 8 * p3);
                        // IRT_TRACE_TO_LOG(15,compFile,"rot_mem_data.pix[%0d] = %x : in_pix_in_mem = %x
                        // ",(index*in_data_type) + p3,tmp_in,in_pix_in_mem);
                    }
                    if (in_data_type == 2) { // 2 byte mode
                        in_pix_in_mem &= MSI;
                    }
#ifdef RUN_WITH_SV
                    //---------------------------------------
                    // debug info for SV
                    if (irt_mode != 6) {
                        p2x2_inf.pix_vld[p1] = 1;
                        p2x2_inf.pix[p1][p2] = in_pix_in_mem;
                    }
                    //---------------------------------------
                    uint32_t index_f = bwd2_en ? ((((uint32_t)floor(index / rot_mem_w_elm)) >> 0x1) * rot_mem_w_elm) +
                                                     (index & (rot_mem_w_elm - 1))
                                               : index;
                    way_offset_row    = (uint8_t)floor(index_f / rot_mem_w_elm);
                    way_offset_col    = (uint8_t)floor(index_f / (rot_mem_w_elm / 2)) & 0x1;
                    way_offset_col2   = (uint8_t)floor(index_f / (rot_mem_w_elm / 4)) & 0x1;
                    uint8_t num_lines = bwd2_en ? 2 : 4; // 1 $line occupies how many sram rows
                    uint8_t bank_row_f =
                        bwd2_en ? ((((uint32_t)floor(index / rot_mem_w_elm) & 0x1) == 0) ? (cmp_bank_row & 0x3)
                                                                                         : ((cmp_bank_row & 0x3) + 0x4))
                                : cmp_bank_row;
                    uint8_t c_set   = (cmp_set & (uint8_t)(NUM_BANK_ROW - 1));
                    uint8_t act_set = bwd2_en ? ((c_set << 1) + (uint8_t)(cmp_bank_row >> 2)) : c_set;
                    //               way_offset_col = (uint8_t)floor(index/(rot_mem_w_elm/2)) & 0x1;
                    rmrm_inf.addr[bank_row_f][2 * way_offset_col] =
                        ((uint32_t)(num_lines * NWAYS) * (uint32_t)act_set) + ((uint32_t)cmp_way * num_lines) +
                        way_offset_row;
                    rmrm_inf.addr[bank_row_f][2 * way_offset_col + 1] =
                        ((uint32_t)(num_lines * NWAYS) * (uint32_t)act_set) + ((uint32_t)cmp_way * num_lines) +
                        way_offset_row;
                    rmrm_inf.valid[bank_row_f][2 * way_offset_col] =
                        bwd2_en ? ((way_offset_col2 ? 0 : 1) | rmrm_inf.valid[bank_row_f][2 * way_offset_col]) : 1;
                    rmrm_inf.valid[bank_row_f][2 * way_offset_col + 1] =
                        bwd2_en ? (way_offset_col2 | rmrm_inf.valid[bank_row_f][2 * way_offset_col + 1]) : 1;
                    //               rmrm_inf.valid[cmp_bank_row][2*way_offset_col]=1;
                    //               rmrm_inf.valid[cmp_bank_row][2*way_offset_col+1]=1;
                    rmrm_inf.proc_size = p_size;
                    // IRT_TRACE_TO_LOG(5,compFile,"ccc %d index %d bwd2_en %d rot_mem_w_elm %d way_offset_row %d
                    // way_offset_col/col2 %d %d addr %d %d valid %d
                    // %d\n",curr_compute_cnt,index,bwd2_en,rot_mem_w_elm,way_offset_row,way_offset_col,way_offset_col2,rmrm_inf.addr[cmp_bank_row][2*way_offset_col],rmrm_inf.addr[cmp_bank_row][2*way_offset_col+1],rmrm_inf.valid[cmp_bank_row][2*way_offset_col],rmrm_inf.valid[cmp_bank_row][2*way_offset_col+1]);
                    // End of Rot mem addr gen debug purpose pnly.
                    //#############################################
#endif
                    // im_P[p2] = IRT_top::IRT_UTILS::conversion_bit_float(in_pix_in_mem, in_dtype, clip_fp);
                    im_P[p2]    = IRT_top::IRT_UTILS::conversion_bit_float(in_pix_in_mem, in_dtype, 0);
                    im_Phex[p2] = in_pix_in_mem;
                    IRT_TRACE_TO_LOG(15,
                                     compFile,
                                     "Wx %f W %f im_P[%d] = %6.32f in_pix_in_mem = %x in_dtype %d \n",
                                     Wx,
                                     Wy,
                                     p2,
                                     im_P[p2],
                                     in_pix_in_mem,
                                     in_dtype);
                    //---------------------------------------
                    if (irt_mode == 6) {
                        IRT_TRACE_TO_LOG(15,
                                         compFile,
                                         "Wx %f W %f im_P[%d] = %6.32f in_pix_in_mem = %x in_dtype %d \n",
                                         Wx,
                                         Wy,
                                         p2,
                                         im_P[p2],
                                         in_pix_in_mem,
                                         in_dtype);
                        float tmp = 0;
                        switch (p2) { // compute done only when weight's non-zero. It causes issue when
                                      // grad_out=Nan/inf.
                            case 0:
                                if ((one_m_xf != 0) && (one_m_yf != 0))
                                    tmp = (grad_out[p1] * (one_m_xf * one_m_yf));
                                break;
                            case 1:
                                if ((xf != 0) && (one_m_yf != 0))
                                    tmp = (grad_out[p1] * (xf * one_m_yf));
                                break;
                            case 2:
                                if ((one_m_xf != 0) && (yf != 0))
                                    tmp = (grad_out[p1] * (one_m_xf * yf));
                                break;
                            case 3:
                                if ((xf != 0) && (yf != 0))
                                    tmp = (grad_out[p1] * (xf * yf));
                                break;
                        }
                        g_out[p2]      = im_P[p2] + tmp;
                        uint32_t tmp_o = IRT_top::IRT_UTILS::conversion_float_bit(
                            g_out[p2], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                        //---------------------------------------
                        check_num_err(tmp_o, out_dtype, 1);
                        tmp_o = check_nan(tmp_o, out_dtype);
                        //---------------------------------------
                        switch (p2) {
                            case 0:
                                IRT_TRACE_TO_LOG(
                                    20,
                                    compFile,
                                    "-->grad_out=%f x:y=[%d][%d] 1mxf:1myf[%f,%f] grad_data[%f + %f]=[%f][%x] \n",
                                    grad_out[p1],
                                    fx,
                                    fy,
                                    one_m_xf,
                                    one_m_yf,
                                    im_P[p2],
                                    tmp,
                                    g_out[p2],
                                    tmp_o);
                                break;
                            case 1:
                                IRT_TRACE_TO_LOG(
                                    20,
                                    compFile,
                                    "-->grad_out=%f x:y=[%d][%d] xf:1myf[%f,%f]  grad_data[%f + %f]=[%f][%x] \n",
                                    grad_out[p1],
                                    cx,
                                    fy,
                                    xf,
                                    one_m_yf,
                                    im_P[p2],
                                    tmp,
                                    g_out[p2],
                                    tmp_o);
                                break;
                            case 2:
                                IRT_TRACE_TO_LOG(
                                    20,
                                    compFile,
                                    "-->grad_out=%f x:y=[%d][%d] 1mxf:yf[%f,%f]  grad_data[%f + %f]=[%f][%x] \n",
                                    grad_out[p1],
                                    fx,
                                    cy,
                                    one_m_xf,
                                    yf,
                                    im_P[p2],
                                    tmp,
                                    g_out[p2],
                                    tmp_o);
                                break;
                            case 3:
                                IRT_TRACE_TO_LOG(
                                    20,
                                    compFile,
                                    "-->grad_out=%f x:y=[%d][%d] xf:yf[%f,%f]  grad_data[%f + %f]=[%f][%x] \n",
                                    grad_out[p1],
                                    cx,
                                    cy,
                                    xf,
                                    yf,
                                    im_P[p2],
                                    tmp,
                                    g_out[p2],
                                    tmp_o);
                                break;
                        }
                        IRT_TRACE_TO_LOG(
                            15,
                            compFile,
                            "-->set:way = [%d:%d] row:col=[%d:%d] index=%d:fx=%d:p2=%d:cl_size=%d:out_dtype=%d \n",
                            cmp_set,
                            cmp_way,
                            cmp_tag[0],
                            cmp_tag[1],
                            index,
                            fx,
                            p2,
                            cl_size,
                            out_dtype);
                        //---------------------------------------
                        // uint8_t b_index = (fx + p2%2)%(cl_size);
                        for (uint8_t bp3 = 0; bp3 < in_data_type; bp3++) {
                            uint8_t tmp_o_byte = (uint8_t)((tmp_o >> (8 * bp3)) & 0x000000FF);
                            rot_mem_data.pix[(index * ((int)out_dtype)) + bp3] = tmp_o_byte;
                            rot_mem_data.en[(index * ((int)out_dtype)) + bp3] =
                                1; // TODO -- SHOULD BE IN OWM_WRITE FUNC ??
                        }
                        resamp_rot_mem[cmp_set][cmp_way] = rot_mem_data;
                    }
                    //---------------------------------------
                }
                if (irt_mode == 4) {
                    // BLI compute
                    p_out[p1] =
                        ((one_m_yf)*im_P[0] + yf * im_P[2]) * (one_m_xf) + ((one_m_yf)*im_P[1] + yf * im_P[3]) * (xf);
                    IRT_TRACE_TO_LOG(
                        20,
                        compFile,
                        "\tOutPix[%d] [Float:FP32][%f:%x] :: IN_PIX[Float:Hex] = [%f:%f:%f:%f] [%x,%x,%x,%x] ::",
                        p1,
                        p_out[p1],
                        IRT_top::IRT_UTILS::conversion_float_bit(
                            p_out[p1], e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0),
                        im_P[0],
                        im_P[1],
                        im_P[2],
                        im_P[3],
                        im_Phex[0],
                        im_Phex[1],
                        im_Phex[2],
                        im_Phex[3]);
                    IRT_TRACE_TO_LOG(20,
                                     compFile,
                                     "\tWeights[xf:yf] : [%f:%f] [%x:%x] ",
                                     xf,
                                     yf,
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         xf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0),
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         yf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0));
                    IRT_TRACE_TO_LOG(20,
                                     compFile,
                                     "\t[1Mxf:1Myf] : [%f:%f] [%x:%x] \n",
                                     one_m_xf,
                                     one_m_yf,
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         one_m_xf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0),
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         one_m_yf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0));
#ifdef RUN_WITH_SV
                    p2x2_inf.oPix[p1][0] = IRT_top::IRT_UTILS::conversion_float_bit(
                        p_out[p1], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                    p2x2_inf.oPix[p1][0] = check_nan(p2x2_inf.oPix[p1][0], out_dtype);
                    // IRT_TRACE("oPix[%d] = %x \n",p1,p2x2_inf.oPix[p1]);
#endif
                } else if (irt_mode == 5) {
                    w_out_c[p1] = grad_out[p1] * ((one_m_yf) * (im_P[1] - im_P[0]) + (yf) * (im_P[3] - im_P[2]));
                    w_out_r[p1] = grad_out[p1] * ((one_m_xf) * (im_P[2] - im_P[0]) + (xf) * (im_P[3] - im_P[1]));
                    IRT_TRACE_TO_LOG(20,
                                     compFile,
                                     "\tgrad_out[%d]=%f w_out[%d]=[%f:%f] IN_PIX[Float] = [%f:%f:%f:%f] \n",
                                     p1,
                                     grad_out[p1],
                                     p1,
                                     w_out_c[p1],
                                     w_out_r[p1],
                                     im_P[0],
                                     im_P[1],
                                     im_P[2],
                                     im_P[3]);
                    IRT_TRACE_TO_LOG(20,
                                     compFile,
                                     "\tWeights[xf:yf] : [%f:%f] [%x,%x] \n",
                                     xf,
                                     yf,
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         xf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0),
                                     IRT_top::IRT_UTILS::conversion_float_bit(
                                         yf, e_irt_fp32, 0, 0, 0, clip_fp, clip_fp_inf_input, ftz_en, 0));
#ifdef RUN_WITH_SV
                    p2x2_inf.oPix[p1][0] = IRT_top::IRT_UTILS::conversion_float_bit(
                        w_out_c[p1], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                    p2x2_inf.oPix[p1][0] = check_nan(p2x2_inf.oPix[p1][0], out_dtype);
                    p2x2_inf.oPix[p1][1] = IRT_top::IRT_UTILS::conversion_float_bit(
                        w_out_r[p1], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                    p2x2_inf.oPix[p1][1] = check_nan(p2x2_inf.oPix[p1][1], out_dtype);
#endif
                } else {
                    // printf("ERROR: IRT Mode not supported\n");
                }
            }
            //---------------------------------------
            p2x2_infQ.push(p2x2_inf);
            rmrm_infQ.push(rmrm_inf);
            /*
            IRT_TRACE_TO_LOG(20,compFile,"Rot Mem Addr : Format (col[0to3],valid) proc size - %d\n",rmrm_inf.proc_size);
            for(uint8_t di=0;di<8;di++){
              IRT_TRACE_TO_LOG(20,compFile,"\tBank[%d] : [%d,%d],  [%d,%d],  [%d,%d],  [%d,%d] \n", di,
            rmrm_inf.addr[di][0],rmrm_inf.valid[di][0],rmrm_inf.addr[di][1],rmrm_inf.valid[di][1],rmrm_inf.addr[di][2],rmrm_inf.valid[di][2],rmrm_inf.addr[di][3],rmrm_inf.valid[di][3]);
            }
            */

            // end of proc size compute.
            // push the output to OWM Q.
            IRT_TRACE_TO_LOG(20, compFile, "Float-to-Hex:[float:hex] for Output Image Format - \n");
            for (uint8_t p4 = 0; p4 < p_size; p4++) {
                uint32_t tmp_out, tmp_out1 = 0;
                if (irt_mode == 4) {
                    tmp_out = IRT_top::IRT_UTILS::conversion_float_bit(
                        p_out[p4], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                } else if (irt_mode == 5) {
                    tmp_out = IRT_top::IRT_UTILS::conversion_float_bit(
                        w_out_c[p4], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                    tmp_out1 = IRT_top::IRT_UTILS::conversion_float_bit(
                        w_out_r[p4], out_dtype, bli_shift, MAX_VALo, Ppo, clip_fp, clip_fp_inf_input, ftz_en, 0);
                }
                //---------------------------------------
                check_num_err(tmp_out, out_dtype, 1);
                check_num_err(tmp_out1, out_dtype, 1);
                tmp_out  = check_nan(tmp_out, out_dtype);
                tmp_out1 = check_nan(tmp_out1, out_dtype);
                //---------------------------------------
                if (irt_mode == 4) {
                    IRT_TRACE_TO_LOG(20, compFile, "\tOutPix[%d] : [%f] [%x] \n", p4, p_out[p4], tmp_out);
                } else if (irt_mode == 5) {
                    IRT_TRACE_TO_LOG(20,
                                     compFile,
                                     "[w_out_c:w_out_r][%f:%f][%x:%x] \n",
                                     w_out_c[p4],
                                     w_out_r[p4],
                                     tmp_out,
                                     tmp_out1);
                }

                // IRT_TRACE_TO_LOG(2,compFile,"tmp_out=%x owmDataQ[ = ",tmp_out);
                if ((irt_mode == 4) | (irt_mode == 5)) {
                    for (uint8_t p5 = 0; p5 < out_data_type; p5++) {
                        uint8_t tmp_out_byte = (uint8_t)((tmp_out >> (8 * p5)) & 0x000000FF);
                        owmDataQ.push(tmp_out_byte);
                        // push extra data only in bwd pass
                        // IRT_TRACE_TO_LOG(2,compFile,"%x,",tmp_out_byte);
                    }
                    if (irt_mode == 5) {
                        for (uint8_t p6 = 0; p6 < out_data_type; p6++) {
                            uint8_t tmp_out_byte1 = (uint8_t)((tmp_out1 >> (8 * p6)) & 0x000000FF);
                            owmDataQ.push(tmp_out_byte1);
                        }
                    }
                }
            }
            // IRT_TRACE_TO_LOG(2,compFile,"]\n");
            // IRT_TRACE_TO_LOG(2,compFile," cbc = %d\n",cbc);
            // IRT_TRACE_TO_LOG(2,compFile,"==================================");
            // update counters
            if (k + p_size < stripe_w) {
                comp_parser_cnt.pcoord = comp_parser_cnt.pcoord + p_size;
            } else {
                comp_parser_cnt.pcoord = 0;
                if (j + 1 < Hm) {
                    comp_parser_cnt.pline = comp_parser_cnt.pline + 1;
                } else {
                    comp_parser_cnt.pline = 0;
                    if (i + 1 < num_warp_stripes)
                        comp_parser_cnt.pstripe = comp_parser_cnt.pstripe + 1;
                    else
                        comp_parser_cnt.done = 1; // all coordinates done for parsing.
                }
            }
            // IRT_TRACE_TO_LOG(2,compFile,"==================================\n");
        } else {
            // waiting for data, compute idle cycle
            p_size = 0;
            // IRT_TRACE_TO_LOG(5,IRTLOG,"SAGAR : waiting for data: : PROC_SIZE = %d at cycle %d L MISS = %d
            // compute_timeout = %0d\n",p_size,cycle,l_miss,compute_timeout);
            compute_timeout++;
        }

    } else {
        if (!comp_parser_cnt.done) {
            // IRT_TRACE_TO_LOG(2,compFile,"==================================\nWaiting for Warp Data @ cycle %d
            // ,num_stripes %d height %d width %d,\n",cycle,i,j,k); compute_timeout++; add waiting for warp data under
            // debug trace parser data not available
        } else {
            // IRT_TRACE_TO_LOG(2,compFile,"Compute is done @ cycle %d ,num_stripes %d height %d width
            // %d,\n\n",cycle,i,j,k);
            compute_timeout = 0;
        }
    }
    myfree3_int(tag_a, IRT_ROT_MAX_PROC_SIZE, 4, 2);
    myfree3_int(m_tag, 2, NUM_BANK_ROW, 2);
    myfree2_int(m_tag_valid, 2, NUM_BANK_ROW);
}

float IRT_top::IRT_RESAMP::get_resamp_data(int      plane,
                                           int      line,
                                           int      pixel,
                                           uint16_t Hi,
                                           uint16_t Wi,
                                           int      bg_mode,
                                           float    pad_val)
{
    float out;
    int   sat_x;
    int   sat_y;

    if (bg_mode != 2) {
        sat_x = min(max(0, pixel), Wi - 1);
        sat_y = min(max(0, line), Hi - 1);
    } else {
        sat_x = pixel;
        sat_y = line;
    }

    IRT_TRACE_TO_LOG(22,
                     compFile,
                     "plane %d,line %d, pixel %d, Hi %d, Wi %d, bg_mode %d, pad_val %f\n",
                     plane,
                     line,
                     pixel,
                     Hi,
                     Wi,
                     bg_mode,
                     pad_val);
    // if((line<0 || line>(Hi-1) || pixel<0  || pixel>(Wi-1)) && (bg_mode==2)) {
    //  IRT_TRACE_TO_LOG(2,compFile,"returning pad val \n");
    //  return pad_val;
    //}
    // else if((line<0 || line>(Hi-1) || pixel<0  || pixel>(Wi-1)) && (bg_mode==0)) {
    //  IRT_TRACE_TO_LOG(2,compFile,"returning 0\n");
    //  return 0;
    //}
    // else {
    IRT_TRACE_TO_LOG(2, compFile, "returning pixel from plane %d line %d and pixel %d\n", plane, line, pixel);
    return out = resamp_wimage[plane][line].x[pixel];
    //}
}

//----------------------------------------------
// rescale compute function
void IRT_top::IRT_RESAMP::compute_rescale()
{
    // desc access
    uint16_t Hi                = descr.image_par[IIMAGE].H;
    uint16_t Wi                = descr.image_par[IIMAGE].W;
    uint16_t Hm                = descr.image_par[MIMAGE].H;
    uint16_t Wm                = descr.image_par[MIMAGE].W;
    uint16_t Sm                = descr.image_par[MIMAGE].S;
    uint16_t Ho                = descr.image_par[OIMAGE].H;
    uint16_t Wo                = descr.image_par[OIMAGE].W;
    uint16_t So                = descr.image_par[OIMAGE].S;
    bool     clip_fp           = descr.clip_fp;
    bool     clip_fp_inf_input = descr.clip_fp_inf_input;
    bool     ftz_en            = descr.ftz_en;

    uint16_t          num_warp_stripes = no_x_seg; // TODO
    Eirt_bg_mode_type bg_mode          = descr.bg_mode; // TODO replace with desc
    uint16_t          in_data_type     = (1 << descr.image_par[MIMAGE].Ps); // TODO replace with desc
    uint16_t          irt_mode         = descr.irt_mode;
    uint16_t          out_data_type    = (irt_mode != 5) ? (1 << descr.image_par[OIMAGE].Ps)
                                             : (1 << (descr.image_par[OIMAGE].Ps - 1)); // TODO replace with desc
    Eirt_resamp_dtype_enum in_dtype     = descr.image_par[MIMAGE].DataType;
    Eirt_resamp_dtype_enum out_dtype    = descr.image_par[OIMAGE].DataType;
    int16_t                bli_shift    = descr.bli_shift;
    uint16_t               MAX_VALo     = descr.MAX_VALo;
    uint16_t               Ppo          = descr.Ppo;
    uint16_t               sw_proc_size = descr.proc_size;
    std::string            my_string    = getInstanceName() + ":COMPUTE:";
    irt_mem_ctrl_struct*   irt_mem_ctrl = &irt_top->mem_ctrl[0];
    filter_type_t          flt_type     = HLpars.filter_type;
    // filter_type_t flt_type = irt_top->rot_pars[0].filter_type;
    uint8_t num_Htaps = descr.rescale_LUT_x->num_taps;
    uint8_t num_Vtaps = descr.rescale_LUT_y->num_taps;
    int     Hoff      = (num_Htaps % 2 == 0) ? (num_Htaps / 2) - 1 : (num_Htaps / 2);
    int     Voff      = (num_Vtaps % 2 == 0) ? (num_Vtaps / 2) - 1 : (num_Vtaps / 2);
    // local varibles
    uint32_t i, j, k;
    uint32_t vtap_cnt, htap_cnt;
    uint32_t vrowMod2, hrowMod2;
    uint32_t num_vld_pxl;
    uint8_t  num_elm = 0;
    float    im_P;
    // saturation values for OOB
    // TODO = bg_mode==0 should
    float p_sat_x = (bg_mode == e_irt_bg_in_pad) ? (float)Wm : (float)Wm - 1;
    float p_sat_y = (bg_mode == e_irt_bg_in_pad) ? (float)Hm : (float)Hm - 1;
    float n_sat_x = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;
    float n_sat_y = (bg_mode == e_irt_bg_in_pad) ? (float)-1 : (float)0;

    // global counters
    i        = comp_rescale_cnt.pstripe;
    j        = comp_rescale_cnt.pline;
    k        = comp_rescale_cnt.pcoord;
    vtap_cnt = comp_rescale_cnt.vtap_cnt;
    htap_cnt = comp_rescale_cnt.htap_cnt;
    vrowMod2 = comp_rescale_cnt.vrowMod2;
    hrowMod2 = comp_rescale_cnt.hrowMod2;
    //---------------------------------------
    // calc Yt & Yb and check if resamp_wimage has all required lines
    // float    Yi0  = (descr.rescale_LUT_y->Gf*(j+irt_top->rot_pars[0].nudge)) - irt_top->rot_pars[0].nudge;//get Yi
    // for current line
    float Yi0; // get Yi for current line
    if ((j * 2 + vrowMod2) <= (Ho - 1))
        Yi0 = (descr.rescale_LUT_y->Gf * (j * 2 + vrowMod2 - ((float)descr.image_par[OIMAGE].Yc / 2))) +
              ((float)descr.image_par[IIMAGE].Yc / 2);
    else {
        Yi0 = (descr.rescale_LUT_y->Gf * (j * 2 - ((float)descr.image_par[OIMAGE].Yc / 2))) +
              ((float)descr.image_par[IIMAGE].Yc / 2);
        IRT_TRACE_TO_LOG(2, compFile, "Compute for last pad line\n");
    }
    int16_t Ybot = floor(Yi0) + ceil((float)descr.rescale_LUT_y->num_taps / 2); // Yi+(num_taps/2) is last line needed
    IRT_TRACE_TO_LOG(23,
                     compFile,
                     "Yi0 %f Ybot %d OP Yc %f IP Yc %f\n",
                     Yi0,
                     Ybot,
                     ((float)descr.image_par[OIMAGE].Yc / 2),
                     ((float)descr.image_par[IIMAGE].Yc / 2));

    // Rescale V INTP data type Q
    rescale_v_interp_inf rescale_v_intp_data;
    memset(&rescale_v_intp_data, 0, sizeof(rescale_v_interp_inf));
    // Rescale H INTP data type Q
    rescale_h_interp_inf rescale_h_intp_data;
    memset(&rescale_h_intp_data, 0, sizeof(rescale_h_interp_inf));

    // Ybot = (Ybot < n_sat_y) ? n_sat_y  : (Ybot > p_sat_y) ?  p_sat_y : Ybot;
    if (bg_mode == e_irt_bg_frame_repeat) {
        Ybot = (Ybot < n_sat_y) ? n_sat_y : (Ybot > p_sat_y) ? p_sat_y : Ybot;
        // if(Ybot < n_sat_y) Ybot = n_sat_y;
        // else if(Ybot > p_sat_y) Ybot = p_sat_y;
    }
    //---------------------------------------
    if (compute_timeout > MAX_COMPUTE_STALL) {
        IRT_TRACE(
            "Rescale Error - %s: compute timed-out :: Either warp-data not received or All-Cache-Hit not hit %d\n",
            my_string.c_str(),
            cycle);
        IRT_TRACE_TO_RES(test_res,
                         "Error - %s: compute timed-out :: Either warp-data not received or All-Cache-Hit not hit \n",
                         my_string.c_str());
        IRT_CLOSE_FAILED_TEST(0);
    }
    //---------------------------------------
    IRT_TRACE_TO_LOG(2, compFile, "==================================\n");
    IRT_TRACE_TO_LOG(2,
                     compFile,
                     "[stripe:h:w:vtap_cnt:htap_cnt:vrowMod2] [%d:%d:%d:%d:%d:%d] Ybot = %d last_line %d\n",
                     comp_rescale_cnt.pstripe,
                     comp_rescale_cnt.pline,
                     comp_rescale_cnt.pcoord,
                     vtap_cnt,
                     htap_cnt,
                     comp_rescale_cnt.vrowMod2,
                     Ybot,
                     irt_mem_ctrl->last_line[0]);
    //---------------------------------------

    //---------------------------------------
    // IRT_TRACE_TO_LOG(2,compFile,"[stripe:h:w] [%d:%d:%d]
    // \n",comp_rescale_cnt.pstripe,comp_rescale_cnt.pline,comp_rescale_cnt.pcoord); float xi[8],xf[8];
    int64_t  Xi_fixed[8];
    int64_t  Yi_fixed[8];
    uint32_t XI_fixed[8];
    uint32_t YI_fixed;
    uint32_t Xf_fixed[8];
    uint32_t Yf_fixed;
    float    pxl0;
    float    pxl1;
    int32_t  M0[2], M1[2];
    M0[0] = descr.mesh_Gh;
    M0[1] = 0;
    M1[0] = 0;
    M1[1] = descr.mesh_Gv;

    IRT_top::irt_iicc_fixed(Xi_fixed,
                            M0,
                            2 * k - descr.image_par[OIMAGE].Xc,
                            2 * (j * 2 + vrowMod2) - descr.image_par[OIMAGE].Yc,
                            descr.prec_align); // todo nudge
    if ((j * 2 + vrowMod2) <= (Ho - 1))
        IRT_top::irt_iicc_fixed(Yi_fixed,
                                M1,
                                2 * k - descr.image_par[OIMAGE].Xc,
                                2 * (j * 2 + vrowMod2) - descr.image_par[OIMAGE].Yc,
                                descr.prec_align);
    else
        IRT_top::irt_iicc_fixed(Yi_fixed,
                                M1,
                                2 * k - descr.image_par[OIMAGE].Xc,
                                2 * (j * 2) - descr.image_par[OIMAGE].Yc,
                                descr.prec_align);
    // IRT_TRACE_TO_LOG(2,compFile,"[stripe:h:w] [%d:%d:%d]: Xi:%d,Yi:%d
    // \n",comp_rescale_cnt.pstripe,comp_rescale_cnt.pline,comp_rescale_cnt.pcoord,Xi_fixed[0],Yi_fixed[0]);
    int hpi[8];
    int hpi_abs[8];
    for (uint8_t l = 0; l < (std::min((int)(sw_proc_size), (int)(So - k))); l++) {
        //---------------------------------------
        Xi_fixed[l] = Xi_fixed[l] + (int64_t)((int64_t)(descr.image_par[IIMAGE].Xc) << 30);
        XI_fixed[l] = (int)(Xi_fixed[l] >> 31); // floor(xi)
        Xf_fixed[l] = (uint32_t)(Xi_fixed[l] - ((int64_t)XI_fixed[l] * ((int64_t)1 << 31)));
        hpi_abs[l]  = (int)rint((double)Xf_fixed[l] / (double)(pow(2, descr.rescale_phases_prec_H))); // phase index
        XI_fixed[l] = XI_fixed[l] + hpi_abs[l] / descr.rescale_LUT_x->num_phases;
        hpi[l]      = hpi_abs[l] % descr.rescale_LUT_x->num_phases; // phase index
        // hpi[l]           = Xf_fixed[l] >> descr.rescale_phases_prec_H ;   //phase index
    }
    // float yi = (descr.rescale_LUT_y->Gf*(((float)j)+HLpars.nudge))-HLpars.nudge;
    Yi_fixed[0] = Yi_fixed[0] + (int64_t)((int64_t)(descr.image_par[IIMAGE].Yc) << 30);
    YI_fixed    = (int)(Yi_fixed[0] >> 31); // floor(xi)
    Yf_fixed    = (uint32_t)(Yi_fixed[0] - ((int64_t)YI_fixed * ((int64_t)1 << 31)));
    int vpi_abs = (int)rint((double)Yf_fixed / (double)(pow(2, descr.rescale_phases_prec_V))); // phase index
    YI_fixed    = YI_fixed + vpi_abs / descr.rescale_LUT_y->num_phases;
    int vpi     = vpi_abs % descr.rescale_LUT_y->num_phases; // phase index
    // int vpi               = Yf_fixed >> descr.rescale_phases_prec_V;
    int Xflr_pre = XI_fixed[0] - Hoff;
    int Xflr     = Xflr_pre;
    if (bg_mode != 2) {
        Xflr = (Xflr_pre < n_sat_x) ? n_sat_x : (Xflr_pre > p_sat_x) ? p_sat_x : Xflr_pre;
    }
    int Yflr = YI_fixed;
    IRT_TRACE_TO_LOG(
        2,
        compFile,
        "[precision CFG] rescale_phases_prec_V : %d , rescale_phases_prec_H:%d,prec_align:%d,mesh_Gh:%u, \n",
        descr.rescale_phases_prec_V,
        descr.rescale_phases_prec_H,
        descr.prec_align,
        descr.mesh_Gh);
    IRT_TRACE_TO_LOG(
        2,
        compFile,
        "[stripe:h:w] [%d:%d:%d]: Xi:%llx,Yi:%llx,XI:%d,YI:%d Xf:%d,Yf:%d Xflr:%d, Yflr:%d,hpi:%d,vpi:%d\n",
        comp_rescale_cnt.pstripe,
        comp_rescale_cnt.pline,
        comp_rescale_cnt.pcoord,
        Xi_fixed[3],
        Yi_fixed[0],
        XI_fixed[3],
        YI_fixed,
        Xf_fixed[3],
        Yf_fixed,
        Xflr,
        Yflr,
        hpi[3],
        vpi);
    // IRT_TRACE_TO_LOG(4,compFile,"beforeSaturation - hti=%d\n",hti);
    // vertical interpolation - accumulate verticle pixels and interpolate
    int vti0 = (Yflr - Voff + vtap_cnt); // tap index
    int vti1 = (Yflr - Voff + vtap_cnt + 1); // tap index
    if (bg_mode != 2) { //
        vti0 = (vti0 < n_sat_y) ? n_sat_y : (vti0 > p_sat_y) ? p_sat_y : vti0;
        vti1 = (vti1 < n_sat_y) ? n_sat_y : (vti1 > p_sat_y) ? p_sat_y : vti1;
    }
    float rescale_bg_pxl = ((HLpars.Pwi == 8) ? (descr.mrsb_bg & 0xFF) : (descr.mrsb_bg & descr.Msi));
    // IRT_TRACE_TO_LOG(2,compFile,"vpi=%d  \n",vpi);
    float vcoeff0 = IRT_top::IRT_UTILS::get_rescale_coeff(
        flt_type,
        descr.rescale_LUT_y,
        vpi,
        vtap_cnt); // TODO REvtap_cntEW -- FILTER TYPE SHOULD NOT BE USED IN ARCH MODEL
    float vcoeff1 = IRT_top::IRT_UTILS::get_rescale_coeff(
        flt_type,
        descr.rescale_LUT_y,
        vpi,
        vtap_cnt + 1); // TODO REVIEW -- FILTER TYPE SHOULD NOT BE USED IN ARCH MODEL
    if (((irt_mem_ctrl->last_line[0] + descr.Yi_start) >= Ybot) && !comp_rescale_cnt.done &&
        (vtap_cnt <= (num_Vtaps - 2)) && first_vld_warp_line) {
        for (int px = 0; px < 16; px++) {
            int Xflr0 = (Xflr + px); // < n_sat_x) ? n_sat_x  : ((Xflr+px) > p_sat_x) ?  p_sat_x : (Xflr+px);
            IRT_TRACE_TO_LOG(2,
                             compFile,
                             "Line %d:%d Xflr0 %d Hm %d Sm %d bg_mode %d and rescale bg vale %f\n",
                             vti0,
                             vti1,
                             Xflr0,
                             Hm,
                             Sm,
                             bg_mode,
                             rescale_bg_pxl);
            pxl0 = get_resamp_data(
                i, vti0 - descr.Yi_start, Xflr0 - descr.Xi_start_offset, Hm, Sm, bg_mode, rescale_bg_pxl);
            pxl1 = get_resamp_data(
                i, vti1 - descr.Yi_start, Xflr0 - descr.Xi_start_offset, Hm, Sm, bg_mode, rescale_bg_pxl);
            IRT_TRACE_TO_LOG(23, compFile, "pxl0 %f and pxl1 %f\n", pxl0, pxl1);
            if (vtap_cnt == 0)
                rescale_v_accum.pix[vrowMod2][px] = 0;
            if (vtap_cnt == num_Vtaps - 1) {
                rescale_v_accum.pix[vrowMod2][px] += vcoeff0 * pxl0;
                IRT_TRACE_TO_LOG(22,
                                 compFile,
                                 "Vaccum:pix=%d coeff=%f Inx=%f at cycle %d with xflr0 %0d\n",
                                 px,
                                 vcoeff0,
                                 pxl0,
                                 cycle,
                                 Xflr0);
            } else {
                rescale_v_accum.pix[vrowMod2][px] +=
                    vcoeff0 * pxl0 + vcoeff1 * pxl1; // For rescale mode - INput IMage is always in FP32 as warp_read
                                                     // accumulated in FP32 format
                IRT_TRACE_TO_LOG(22,
                                 compFile,
                                 "Vaccum:pix=%d coeff=%f:%f Inx=%f:%f at cycle %d with xflr0 %0d\n",
                                 px,
                                 vcoeff0,
                                 vcoeff1,
                                 pxl0,
                                 pxl1,
                                 cycle,
                                 Xflr0);
            }

            rescale_v_intp_data.in_a[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(vcoeff0);
            rescale_v_intp_data.in_c[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(vcoeff1);
            rescale_v_intp_data.in_b[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(pxl0);
            rescale_v_intp_data.in_d[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(pxl1);
        }
        IRT_TRACE_TO_LOG(2,
                         compFile,
                         "Check - H taps %0d, proc_size %0d, Y scale factor %0f, OP image W %0d, IP image W %0d OP "
                         "stripe %0d and co-ordinate %0d\n",
                         num_Htaps,
                         sw_proc_size,
                         descr.rescale_LUT_y->Gf,
                         Wo,
                         Wi,
                         So,
                         k);
        num_vld_pxl =
            num_Htaps + (int)floor((std::min((int)(sw_proc_size - 1), (int)(So - k - 1))) * descr.rescale_LUT_x->Gf);
        if (num_vld_pxl + Xflr > Sm - 1) {
            num_vld_pxl = Sm - Xflr;
            IRT_TRACE_TO_LOG(2, compFile, "Inside saturation logic, num_vld_pxl %d\n", num_vld_pxl);
        }
        rescale_v_intp_data.vld_pxl  = num_vld_pxl;
        rescale_v_intp_data.task_num = h9taskid; // irt_top->task[e_irt_resamp];
        rescale_v_intp_infQ.push(rescale_v_intp_data);

        // if(vtap_cnt >= (num_Vtaps-2) && ((vrowMod2==1)||(j==Ho-1))) {
        if (vtap_cnt >= (num_Vtaps - 2) && ((vrowMod2 == 1))) {
            for (int px = 0; px < 8; px++) {
                rescale_v_accum.xi[px] = (float)XI_fixed[px]; // For rescale mode - INput IMage is always in FP32 as
                                                              // warp_read accumulated in FP32 format
                rescale_v_accum.hpi[px] =
                    hpi[px]; // For rescale mode - INput IMage is always in FP32 as warp_read accumulated in FP32 format
                rescale_v_accum.XL =
                    Xflr; // For rescale mode - INput IMage is always in FP32 as warp_read accumulated in FP32 format
                rescale_v_accum.curr_line = j;
            }
            rescale_v_accum.num_vld_pxl = std::min((int)(sw_proc_size), (int)(So - k));
            IRT_TRACE_TO_LOG(2, compFile, "Rescale: Num vld pxl :%d ", rescale_v_accum.num_vld_pxl);
            lineBuffer.push(rescale_v_accum);
            //---------------------------------------
            IRT_TRACE_TO_LOG(2, compFile, "Rescale: Vinterp Output ");
            for (int px = 0; px < 16; px++) {
                IRT_TRACE_TO_LOG(2,
                                 compFile,
                                 "%x",
                                 IRT_top::IRT_UTILS::conversion_float_bit(
                                     rescale_v_accum.pix[vrowMod2][15 - px], e_irt_fp32, 0, 0, 0, 0, 0, 0, 0));
            }
            IRT_TRACE_TO_LOG(2, compFile, "\n");
        }
        // else if(comp_rescale_cnt.vrowMod2 || ((j==Ho-1))) { comp_rescale_cnt.vrowMod2 =0 ; comp_rescale_cnt.vtap_cnt
        // = vtap_cnt+2; }
        else if (comp_rescale_cnt.vrowMod2) {
            comp_rescale_cnt.vrowMod2 = 0;
            comp_rescale_cnt.vtap_cnt = vtap_cnt + 2;
        } else {
            IRT_TRACE_TO_LOG(2, compFile, "Incrementing vrowMod2 for line:%d", j);
            comp_rescale_cnt.vrowMod2++;
        }
        compute_timeout = 0;
    } else {
        if (!comp_rescale_cnt.done) {
            IRT_TRACE_TO_LOG(2,
                             compFile,
                             "==================================\n Vertical Interpolation Waiting for INPUT Data @ "
                             "cycle %d ,num_stripes %d height %d width %d done %d vtap cnt %d and num_Vtaps %d\n",
                             cycle,
                             i,
                             j,
                             k,
                             comp_rescale_cnt.done,
                             vtap_cnt,
                             num_Vtaps);
            compute_timeout++;
        } else {
            IRT_TRACE_TO_LOG(
                2, compFile, "Compute is done @ cycle %d ,num_stripes %d height %d width %d,\n", cycle, i, j, k);
            compute_timeout = 0;
        }
    }
    // horizontal interpolation - verticle pixels accumulation into final output pixel
    // IRT_TRACE_TO_LOG(2,compFile,"linebufEmpty=%d htap_cnt = %d: num_Htaps=%d
    // \n",lineBuffer.empty(),htap_cnt,num_Htaps-2);
    if ((!lineBuffer.empty()) && (htap_cnt <= (num_Htaps - 2))) {
        // IRT_TRACE_TO_LOG(2,compFile,"inside HOR Interp \n");
        bus16px_struct line  = lineBuffer.front();
        int            XL    = line.XL;
        int            hline = line.curr_line;
        // IRT_TRACE_TO_LOG(2,compFile,"psize %d So %d k %d\n",(std::min((int)(sw_proc_size),(int)(So-k))),So,k);
        // IRT_TRACE_TO_LOG(2,compFile,"->num_Htaps %d Xflr %d\n",num_Htaps,Xflr);
        // for(int px=0; px < (std::min((int)(sw_proc_size),(int)(So-k))); px++){
        for (int px = 0; px < (line.num_vld_pxl); px++) {
            // IRT_TRACE_TO_LOG(2,compFile,"px %d \n",px);
            int Xflr = (int)floor(line.xi[px]) - Hoff;
            int hti0 = (Xflr + htap_cnt);
            int hti1 = (Xflr + htap_cnt + 1);
            hti0     = (hti0 < n_sat_x) ? n_sat_x : (hti0 > p_sat_x) ? p_sat_x : hti0; // saturate X coordinate
            hti1     = (hti1 < n_sat_x) ? n_sat_x : (hti1 > p_sat_x) ? p_sat_x : hti1; // saturate X coordinate
            IRT_TRACE_TO_LOG(2,
                             compFile,
                             "--> px %d htap_cnt %d num_Htaps %d Xflr %d line.hpi=%f htap_cnt %d \n",
                             px,
                             htap_cnt,
                             num_Htaps,
                             Xflr,
                             line.hpi[px],
                             htap_cnt);
            float hcoeff0 = IRT_top::IRT_UTILS::get_rescale_coeff(
                flt_type,
                descr.rescale_LUT_x,
                line.hpi[px],
                htap_cnt); // TODO REVIEW -- FILTER TYPE SHOULD NOT BE USED IN ARCH MODEL
            float hcoeff1 = IRT_top::IRT_UTILS::get_rescale_coeff(
                flt_type,
                descr.rescale_LUT_x,
                line.hpi[px],
                htap_cnt + 1); // TODO REVIEW -- FILTER TYPE SHOULD NOT BE USED IN ARCH MODEL
            IRT_TRACE_TO_LOG(2,
                             compFile,
                             "---> px %d coeff [%f : %f] htap_cnt %d num_Htaps %d Xflr %d\n",
                             px,
                             hcoeff0,
                             hcoeff1,
                             htap_cnt,
                             num_Htaps,
                             Xflr);
            if (htap_cnt == 0) {
                rescale_h_accum[hrowMod2][px] = 0;
            }
            if (htap_cnt == num_Htaps - 1)
                rescale_h_accum[hrowMod2][px] += hcoeff0 * line.pix[hrowMod2][hti0 - XL];
            else
                rescale_h_accum[hrowMod2][px] +=
                    hcoeff0 * line.pix[hrowMod2][hti0 - XL] + hcoeff1 * line.pix[hrowMod2][hti1 - XL];
            IRT_TRACE_TO_LOG(2, compFile, "rescale_h_accum[%d] = %p \n", px, rescale_h_accum[px]);
            rescale_h_intp_data.in_a[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(hcoeff0);
            rescale_h_intp_data.in_c[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(hcoeff1);
            rescale_h_intp_data.in_b[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(line.pix[hrowMod2][hti0 - XL]);
            rescale_h_intp_data.in_d[px] = IRT_top::IRT_UTILS::irt_float_to_fp32(line.pix[hrowMod2][hti1 - XL]);
        }
        rescale_h_intp_data.vld_pxl  = line.num_vld_pxl; //(std::min((int)(sw_proc_size),(int)(So-k)) );
        rescale_h_intp_data.task_num = h9taskid; //  irt_top->task[e_irt_resamp];
        rescale_h_intp_infQ.push(rescale_h_intp_data);
        if (htap_cnt < num_Htaps - 2) {
            // if((hrowMod2==0) && (hline < Ho-1))  comp_rescale_cnt.hrowMod2++;
            if ((hrowMod2 == 0))
                comp_rescale_cnt.hrowMod2++;
            else {
                comp_rescale_cnt.hrowMod2 = 0;
                comp_rescale_cnt.htap_cnt = htap_cnt + 2;
            }
        } else if (htap_cnt >= (num_Htaps - 2)) {
            // IRT_TRACE_TO_LOG(2,compFile,"htap_cnt = %d num_Htaps %d\n",htap_cnt,num_Htaps);
            for (int px = 0; px < line.num_vld_pxl /*(std::min((int)(sw_proc_size),(int)(So-k))) */; px++) {
                // IRT_TRACE_TO_LOG(2,compFile,"px %d proc_size %d so-k %d\n",px,sw_proc_size,(int)(So-k));
                // FINAL OUTPUT PIXEL
                uint32_t tmp_out = IRT_top::IRT_UTILS::conversion_float_bit(rescale_h_accum[hrowMod2][px],
                                                                            out_dtype,
                                                                            bli_shift,
                                                                            MAX_VALo,
                                                                            Ppo,
                                                                            clip_fp,
                                                                            clip_fp_inf_input,
                                                                            ftz_en,
                                                                            0);
                // PUSH BYTES TO OUTPUTQ FOR OWM_WRITE FUNCTION
                IRT_TRACE_TO_LOG(2,
                                 compFile,
                                 "===>>OutputPix::[stripe:line:coord:Mod2] [%d:%d:%d:%d] Pix[%f : %x]  \n",
                                 comp_rescale_cnt.pstripe,
                                 comp_rescale_cnt.pline,
                                 comp_rescale_cnt.pcoord,
                                 comp_rescale_cnt.hrowMod2,
                                 rescale_h_accum[hrowMod2][px],
                                 tmp_out);
                for (uint8_t p5 = 0; p5 < out_data_type; p5++) {
                    uint8_t tmp_out_byte = (uint8_t)((tmp_out >> (8 * p5)) & 0x000000FF);
                    owmDataQ.push(tmp_out_byte);
                    // temp_calc_cnt++;
                }
            }
            // if((comp_rescale_cnt.hrowMod2==0) && (hline < Ho-1)) { comp_rescale_cnt.hrowMod2++; }
            if ((comp_rescale_cnt.hrowMod2 == 0)) {
                comp_rescale_cnt.hrowMod2++;
            } else {
                lineBuffer.pop();
                comp_rescale_cnt.hrowMod2 = 0;
                comp_rescale_cnt.htap_cnt = 0;
            }
        }
    }
    // IRT_TRACE_TO_LOG(2,compFile," \n");
    // update counters
    // if(vtap_cnt >= (num_Vtaps-2) && ((vrowMod2==1)||(j==Ho-1))) { //TODO change to max(vtaps, htaps) cycles
    if (vtap_cnt >= (num_Vtaps - 2) && ((vrowMod2 == 1))) { // TODO change to max(vtaps, htaps) cycles
        comp_rescale_cnt.vtap_cnt = 0;
        comp_rescale_cnt.vrowMod2 = 0;
        if ((k + sw_proc_size) < So) {
            comp_rescale_cnt.pcoord = comp_rescale_cnt.pcoord + sw_proc_size;
        } else {
            comp_rescale_cnt.pcoord = 0;
            // if(j*2+1 < Ho-1) {
            if (j < ceil((float)Ho / 2) - 1) {
                comp_rescale_cnt.pline = comp_rescale_cnt.pline + 1;
                IRT_TRACE_TO_LOG(2, compFile, "Incrementing line to %d", comp_rescale_cnt.pline);
            } else {
                IRT_TRACE_TO_LOG(2,
                                 compFile,
                                 "Inside else with line %d co-ordinate %d and j %d Ho %d\n",
                                 comp_rescale_cnt.pline,
                                 comp_rescale_cnt.pcoord,
                                 j,
                                 Ho);
                comp_rescale_cnt.pline = 0;
                if (i + 1 < num_warp_stripes) {
                    comp_rescale_cnt.pstripe = comp_rescale_cnt.pstripe + 1;
                } else {
                    comp_rescale_cnt.done = 1; // all coordinates done for parsing.
                    IRT_TRACE_TO_LOG(2,
                                     compFile,
                                     "Compute done with line %d co-ordinate %d and j %d\n",
                                     comp_rescale_cnt.pline,
                                     comp_rescale_cnt.pcoord,
                                     j);
                }
            }
        }
    }
    // IRT_TRACE_TO_LOG(2,compFile,"Rescale: compute_rescale exit \n");
}

//---------------------------------------
bool IRT_top::IRT_RESAMP::owm_writes(uint64_t& oimage_addr, uint8_t& omem_wr, bus128B_struct& oimage_wr_data)
{
    std::string my_string = getInstanceName() + ":OWM_WRITES:";
    //   uint8_t done_owm;
    uint8_t  oline_byte_cnt;
    uint16_t swb; // stripe width in bytes;
    bool     oline_ready  = 0;
    bool     bwd_pass2_en = (descr.irt_mode == 6) ? 1 : 0;
    uint32_t stripe_width_o; // = last_stripe_width
    irt_top->irt_top_sig.owm_wr_first = 0; // first line of first stripe
    irt_top->irt_top_sig.owm_wr_last  = 0; // last line of last stripe
    omem_wr                           = 0;
    //---------------------------------------
    // IRT_TRACE("OWM_WRITE:DONE:[stripe:line:coord]=[%d:%d:%d] [task:num_of_tasks] [%d:%d]  at cycle %d\n",
    // owm_tracker.pstripe,owm_tracker.pline,owm_tracker.pcoord,irt_top->task[e_irt_resamp],irt_top->num_of_tasks,cycle);
    if ((irt_top->task[e_irt_resamp] >= irt_top->num_of_tasks) ||
        (owm_tracker.pcoord == 0 && owm_tracker.pline == 0 && owm_tracker.pstripe == no_x_seg) ||
        (bwd_pass2_en & w_parser_cnt.done & outputQ.empty() & victimQ.empty())) {
        IRT_TRACE("OWM_WRITE:DONE:[stripe:line:coord]=[%d:%d:%d][%d:%d:%d] [task:num_of_tasks] [%d:%d]  at cycle %d\n",
                  owm_tracker.pstripe,
                  owm_tracker.pline,
                  owm_tracker.pcoord,
                  no_x_seg,
                  descr.image_par[OIMAGE].H,
                  descr.image_par[OIMAGE].S,
                  irt_top->task[e_irt_resamp],
                  irt_top->num_of_tasks,
                  cycle);
        return 1;
    }
    uint16_t last_x_seg = (int)descr.image_par[OIMAGE].W - ((no_x_seg - 1) * (int)descr.image_par[OIMAGE].S);
    //---------------------------------------
    swb = (owm_tracker.pstripe == (no_x_seg - 1)) ? (last_x_seg << descr.image_par[OIMAGE].Ps)
                                                  : (descr.image_par[OIMAGE].S << descr.image_par[OIMAGE].Ps);
    // stripe_width_o = (owm_tracker.pstripe == (no_x_seg-1)) ? descr.last_stripe_width :
    // ((uint32_t)descr.image_par[OIMAGE].S);
    stripe_width_o = (owm_tracker.pstripe == (no_x_seg - 1))
                         ? (descr.image_par[OIMAGE].W - (descr.image_par[OIMAGE].S * (no_x_seg - 1)))
                         : ((uint32_t)descr.image_par[OIMAGE].S);
    //---------------------------------------
    if (!owmDataQ.empty() ||
        ((owmDataQ0.size() != 0 && ofifo_dptr == 0) || (owmDataQ1.size() != 0 && ofifo_dptr == 1))) {
        if (!owmDataQ.empty()) {
            if (owmDataQ.size() % (1 << descr.image_par[OIMAGE].Ps)) {
                IRT_TRACE("owmDataQ.size() had incomplete elements owmDataQ.size = %lu PsBytes = %d\n",
                          owmDataQ.size(),
                          (1 << descr.image_par[OIMAGE].Ps));
                IRT_TRACE_TO_RES(test_res,
                                 "%s:owmDataQ.size() had incomplete elements owmDataQ.size = %lu PsBytes = %d\n",
                                 my_string.c_str(),
                                 owmDataQ.size(),
                                 (1 << descr.image_par[OIMAGE].Ps));
                IRT_CLOSE_FAILED_TEST(0);
            }
        }
        //---------------------------------------
        while (owmDataQ.size() != 0) {
            if (ofifo_dptr == 0)
                owmDataQ0.push(owmDataQ.front());
            else
                owmDataQ1.push(owmDataQ.front());

            owmDataQ.pop();
        }

        oline_ready = 0;
        while (((owmDataQ0.size() != 0 && ofifo_dptr == 0) || (owmDataQ1.size() != 0 && ofifo_dptr == 1)) &&
               (oline_ready != 1)) {
            // copy data to local array
            if (ofifo_dptr == 0) {
                owr_data[ofifo_dptr].pix[lbc[ofifo_dptr] % IRT_OFIFO_WIDTH] = owmDataQ0.front();
                owr_data[ofifo_dptr].en[lbc[ofifo_dptr] % IRT_OFIFO_WIDTH]  = 1;
                IRT_TRACE_TO_LOG(23,
                                 owFile,
                                 "ofifo_dptr %d owmQ.size=%lu lbc=%d swb %d data[%d]=%x at cycle %d\n",
                                 ofifo_dptr,
                                 owmDataQ0.size(),
                                 lbc[ofifo_dptr],
                                 swb,
                                 lbc[ofifo_dptr] % IRT_OFIFO_WIDTH,
                                 owmDataQ0.front(),
                                 cycle);
                owmDataQ0.pop();
            } else {
                owr_data[ofifo_dptr].pix[lbc[ofifo_dptr] % IRT_OFIFO_WIDTH] = owmDataQ1.front();
                owr_data[ofifo_dptr].en[lbc[ofifo_dptr] % IRT_OFIFO_WIDTH]  = 1;
                IRT_TRACE_TO_LOG(20,
                                 owFile,
                                 "ofifo_dptr %d owmQ.size=%lu lbc=%d swb %d data[%d]=%x at cycle %d\n",
                                 ofifo_dptr,
                                 owmDataQ1.size(),
                                 lbc[ofifo_dptr],
                                 swb,
                                 lbc[ofifo_dptr] % IRT_OFIFO_WIDTH,
                                 owmDataQ1.front(),
                                 cycle);
                owmDataQ1.pop();
            }

            // temp_owm_cnt++;
            //         IRT_TRACE_TO_LOG(20,compFile,"temp_owm_cnt %d @cycle %d\n",temp_owm_cnt,cycle);

            // obc++;
            if ((lbc[ofifo_dptr] > 0) && (lbc[ofifo_dptr] % (1 << descr.image_par[OIMAGE].Ps) == 0)) {
                ofifo_coord[ofifo_dptr]++; // owm_tracker.pcoord++;
                owm_tracker.pcoord = ofifo_coord[ofifo_dptr];
            }
            // lbc++;
            //---------------------------------------
            if ((((lbc[ofifo_dptr] + 1) % IRT_OFIFO_WIDTH) == 0) ||
                ((lbc[ofifo_dptr] + 1 == swb) &&
                 (descr.oimage_line_wr_format ==
                  0))) { // 128Byte accumulated OR lineByteCount reaches current StripeWidthInBytes
                oline_ready = 1;
                // irt_top->irt_top_sig.owm_wr_first = (owm_tracker.pstripe == 0) && (owm_tracker.pline == 0) &&
                // ((owm_tracker.pcoord << descr.image_par[OIMAGE].Ps) <= IRT_OFIFO_WIDTH) && (ofifo_dptr==0);
                irt_top->irt_top_sig.owm_wr_first = owm_first_trans;
                owm_first_trans                   = 0;
                //---------------------------------------
                IRT_TRACE_TO_LOG(2, owFile, "========================================\n");
                // IRT_TRACE_TO_LOG(15,owFile,"lbc=%d swb %d lbc-swa=%d owm_wr_first %d [stripe:line:coord]=[%d:%d:%d]
                // ofifo_dptr %d at cycle %d\n", lbc,swb,((lbc+1) %
                // swb),irt_top->irt_top_sig.owm_wr_first,owm_tracker.pstripe,owm_tracker.pline,owm_tracker.pcoord,ofifo_dptr,cycle);
                //---------------------------------------

                // Need to hanle for rescale, for OWM line> OP H, need to disable all bytes
                if (descr.irt_mode == 7 && (ofifo_dptr + owm_tracker.pline) > (descr.image_par[OIMAGE].H - 1)) {
                    IRT_TRACE_TO_LOG(22, owFile, "Rescale case- ofifo ptr + pline increased more then H\n");
                    for (int eni = 0; eni < 128; eni++)
                        owr_data[ofifo_dptr].en[eni] = 0;
                }
                omem_wr        = 1;
                oimage_wr_data = owr_data[ofifo_dptr];

                //---------------------------------------
                // memset(&owr_data, 0, sizeof(bus128B_struct));//MEMSET cant be used as rescale runs on two deep struct
                // array
                for (int eni = 0; eni < 128; eni++) {
                    owr_data[ofifo_dptr].en[eni] = 0;
                }
                //---------------------------------------

                if (descr.oimage_line_wr_format == 0) {
                    // uint64_t byte = owm_tracker.pcoord << descr.image_par[OIMAGE].Ps;
                    uint64_t byte = ((uint64_t)olsc[ofifo_dptr]) << descr.image_par[OIMAGE].Ps;

                    oimage_addr = descr.image_par[OIMAGE].addr_start +
                                  (uint64_t)(owm_tracker.pline + ofifo_dptr) * (uint64_t)descr.image_par[OIMAGE].Hs +
                                  (owm_tracker.pstripe * (uint64_t)descr.image_par[OIMAGE].S *
                                   (1 << descr.image_par[OIMAGE].Ps)) +
                                  (uint64_t)byte;

                    irt_top->irt_top_sig.owm_wr_last =
                        (owm_tracker.pstripe == (no_x_seg - 1)) &&
                        (owm_tracker.pline == (descr.image_par[OIMAGE].H - 1 - ofifo_dptr)) &&
                        (owm_tracker.pcoord == (uint32_t)(stripe_width_o - 1)) &&
                        ((descr.irt_mode != 7) || ((descr.irt_mode == 7) && (ofifo_dptr == 1)));
                    // Need to handle owm__wr_last for ODD lines
                    if ((descr.irt_mode == 7) && (ofifo_dptr + owm_tracker.pline) > (descr.image_par[OIMAGE].H - 1) &&
                        (owm_tracker.pcoord == (uint32_t)(stripe_width_o - 1))) {
                        irt_top->irt_top_sig.owm_wr_last = 1;
                    }
                    // irt_top->irt_top_sig.owm_wr_last = (owm_tracker.pstripe == no_x_seg);
                    IRT_TRACE_TO_LOG(
                        3,
                        owFile,
                        "addr = %lx addr_start %lx line*Hs %lu pstripe %d S %lu Ps %d pstripe*s<<Ps %lu byte %lu "
                        "pcoord %d "
                        "owm_wr_last %d [task:stripe:line:coord]=[%d:%d:%d:%d] omem_wr %d \n",
                        oimage_addr,
                        descr.image_par[OIMAGE].addr_start,
                        (uint64_t)owm_tracker.pline * (uint64_t)descr.image_par[OIMAGE].Hs,
                        owm_tracker.pstripe,
                        (uint64_t)descr.image_par[OIMAGE].S,
                        descr.image_par[OIMAGE].Ps,
                        (owm_tracker.pstripe * (uint64_t)descr.image_par[OIMAGE].S * (1 << descr.image_par[OIMAGE].Ps)),
                        byte,
                        owm_tracker.pcoord,
                        irt_top->irt_top_sig.owm_wr_last,
                        irt_top->task[e_irt_resamp],
                        owm_tracker.pstripe,
                        owm_tracker.pline,
                        owm_tracker.pcoord,
                        omem_wr);
                    // IRT_TRACE("%s: addr = %x addr_start %x line*Hs %x pstripe*S<<Ps %x byte %x owm_wr_last %d \n",
                    // my_string.c_str(),oimage_addr,(uint64_t)pline * (uint64_t) descr.image_par[OIMAGE].Hs,(pstripe *
                    // (uint64_t)descr.image_par[OIMAGE].S * (1<< descr.image_par[OIMAGE].Ps)),byte,
                    // irt_top->irt_top_sig.owm_wr_last);
                } else {
                    oimage_addr = descr.image_par[OIMAGE].addr_start + (uint64_t)opixel;
                    irt_top->irt_top_sig.owm_wr_last =
                        (opixel + IRT_OFIFO_WIDTH) >=
                        (((uint64_t)descr.image_par[OIMAGE].W << descr.image_par[OIMAGE].Ps) *
                         (uint32_t)descr.image_par[OIMAGE].H);
                    opixel += IRT_OFIFO_WIDTH;
                    IRT_TRACE_TO_LOG(2,
                                     owFile,
                                     "addr = %lx addr_start %lx opixel %lx owm_wr_last %d \n",
                                     oimage_addr,
                                     descr.image_par[OIMAGE].addr_start,
                                     opixel,
                                     irt_top->irt_top_sig.owm_wr_last);
                }
                //---------------------------------------
                IRT_TRACE_TO_LOG(5, owFile, "Data_bytes[127:0]: ");
                for (int j = 0; j < 128; j++) {
                    IRT_TRACE_TO_LOG(5, owFile, "%x,", (oimage_wr_data.en[j] == 1) ? oimage_wr_data.pix[j] : 0);
                }
                IRT_TRACE_TO_LOG(5, owFile, "\n");
                //---------------------------------------
                if((((lbc[ofifo_dptr]+1)%swb)==0)/*&& //all bytes for a given line are done 
                  ((descr.irt_mode!=7)||(descr.irt_mode==7 && ofifo_dptr==NUM_OCACHE_RESCALE-1))*/){ // rescale mode - 2nd cache is done
                    lbc[ofifo_dptr] =
                        (descr.oimage_line_wr_format == 0) ? 0 : lbc[ofifo_dptr]; // reset LBC only for oline_fmt == 0
                    // ofifo_coord[0] = 0;
                    // ofifo_coord[1] = 0;
                    ofifo_coord[ofifo_dptr] = 0;
                    // owm_tracker.pcoord = 0;
                    if ((descr.irt_mode == 7 && ofifo_dptr == NUM_OCACHE_RESCALE - 1) || (descr.irt_mode != 7)) {
                        owm_tracker.pline =
                            owm_tracker.pline + 1 + ofifo_dptr; // rescale mode -> INCREASE by 2 cache lines
                        owm_tracker.pcoord = 0;
                    }
                    olsc[ofifo_dptr] = 0;
                    if (owm_tracker.pline >= descr.image_par[OIMAGE].H) {
                        owm_tracker.pline = 0;
                        owm_tracker.pstripe++;
                    }
                } else {
                    lbc[ofifo_dptr]++;
                    olsc[ofifo_dptr] = ofifo_coord[ofifo_dptr] /*owm_tracker.pcoord*/ + 1;
                }
                IRT_TRACE_TO_LOG(
                    2,
                    owFile,
                    "oline_ready==1 owm_wr_first:last=[%d:%d]  [task:stripe:line:coord]=[%d:%d:%d:%d] at cycle %d\n",
                    irt_top->irt_top_sig.owm_wr_first,
                    irt_top->irt_top_sig.owm_wr_last,
                    irt_top->task[e_irt_resamp],
                    owm_tracker.pstripe,
                    owm_tracker.pline,
                    owm_tracker.pcoord,
                    cycle);
                if (descr.irt_mode != 7)
                    break; // break to allow given write trans to be consumed by ext function, else multple cace wites
                           // may happen in same cycle and we loose the data
            } else {
                lbc[ofifo_dptr]++;
            }
        }
        if (descr.irt_mode == 7) {
            ofifo_dptr++;
            if (ofifo_dptr == NUM_OCACHE_RESCALE)
                ofifo_dptr = 0;
        }
    }
    //---------------------------------------
    // BWD-2 - last owm_write is sent only when parser_cnt.done == 1 -> required for LAST indication
    if ((w_parser_cnt.done == 0 && outputQ.size() > 1) || (w_parser_cnt.done == 1 && !outputQ.empty())) {
        sb_struct line;
        line = outputQ.front();

        omem_wr        = 2; // REDUCTION WRITES for BWD2
        oimage_wr_data = line.data;
        oimage_addr = descr.image_par[OIMAGE].addr_start + (line.metadata.row * (uint64_t)descr.image_par[OIMAGE].Hs) +
                      line.metadata.col * 128;
        irt_top->irt_top_sig.bnk_row_owm = line.metadata.bank_row;
        //---------------------------------------
        // OWM FIRST INDICATION
        irt_top->irt_top_sig.owm_wr_first = (owm_first_int == 0) ? 1 : 0;
        owm_first_int                     = 1;
        //---------------------------------------
        outputQ.pop();
        //---------------------------------------
        // int m_pad_cnt=0;
        // for (int i = 0; i < 128; i++) {
        //   if (oimage_wr_data.en[i] == 1) m_pad_cnt++;
        //}
        // IRT_TRACE_TO_LOG(3,owFile,"row:col=%d:%d addr = %x m_pad %d Data_bytes[127:0]:
        // ",line.metadata.row,line.metadata.col, oimage_addr,(128-m_pad_cnt));
        IRT_TRACE_TO_LOG(3,
                         owFile,
                         "row:col=%d:%d addr = %lu Data_bytes[127:0]: ",
                         line.metadata.row,
                         line.metadata.col,
                         oimage_addr);
        for (int j = 0; j < 128; j++) {
            IRT_TRACE_TO_LOG(3, owFile, "%x,", (oimage_wr_data.en[j] == 1) ? oimage_wr_data.pix[j] : 0);
        }
        IRT_TRACE_TO_LOG(3, owFile, "\n");
        //---------------------------------------
        // BWD2 last indication
        if (bwd_pass2_en & w_parser_cnt.done & outputQ.empty() & victimQ.empty()) {
            irt_top->irt_top_sig.owm_wr_last = 1;
        } else {
            irt_top->irt_top_sig.owm_wr_last = 0;
        }
        IRT_TRACE_TO_LOG(3,
                         owFile,
                         "owm_wr_last: %d bwd_pass2_en %d done %d empty  %d:%d @cycle %d\n",
                         irt_top->irt_top_sig.owm_wr_last,
                         bwd_pass2_en,
                         w_parser_cnt.done,
                         outputQ.empty(),
                         victimQ.empty(),
                         cycle);
    }

    return 0;
}
//---------------------------------------
bool IRT_top::IRT_RESAMP::run(uint64_t&             iimage_addr,
                              bool&                 imem_rd,
                              uint16_t&             ilsb_pad_rd,
                              uint16_t&             imsb_pad_rd,
                              meta_data_struct&     iimage_meta_out,
                              const bus128B_struct& iimage_rd_data,
                              meta_data_struct      iimage_meta_in,
                              bool                  imem_rd_data_valid,
                              uint64_t&             mimage_addr,
                              bool&                 mmem_rd,
                              uint16_t&             mlsb_pad_rd,
                              uint16_t&             mmsb_pad_rd,
                              meta_data_struct&     mimage_meta_out,
                              const bus128B_struct& mimage_rd_data,
                              meta_data_struct      mimage_meta_in,
                              bool                  mmem_rd_data_valid,
                              uint64_t&             gimage_addr,
                              bool&                 gmem_rd,
                              uint16_t&             glsb_pad_rd,
                              uint16_t&             gmsb_pad_rd,
                              meta_data_struct&     gimage_meta_out,
                              const bus128B_struct& gimage_rd_data,
                              meta_data_struct      gimage_meta_in,
                              bool                  gmem_rd_data_valid,
                              uint64_t&             oimage_addr,
                              uint8_t&              omem_wr,
                              bus128B_struct&       oimage_wr_data)
{
    int i, warp_line, k, l, m, n;

    //---------------------------------------
    //  float *proc_bin;
    bool resamp_done[7];
    char wrFileptr[50], owFileptr[50], compFileptr[50], ParserFileptr[50], fillFileptr[50], victimFileptr[50],
        grFileptr[50], VinterpFileptr[50], wshflFileptr[50], gshflFileptr[50];
    //---------------------------------------
    // Check resampler descriptor for correct range of all fields
    if (task_start == 0) {
        descr  = irt_top->irt_desc[irt_top->task[e_irt_resamp]];
        HLpars = irt_top->rot_pars[irt_top->task[e_irt_resamp]];
        resamp_descr_checks(descr);

        resamp_mem_alloc_and_init(descr);

        if (descr.irt_mode == e_irt_rescale) {
            wr_atrc.pline = descr.Yi_start;
        }
        task_start = 1;
        h9taskid   = 0;
        // IRT_TRACE_TO_LOG(5,IRTLOG,"SAGAR : h9taskid [%d] addr_pline %u and compute_timeout %d\n", h9taskid,
        // wr_atrc.pline, compute_timeout);
        if (max_log_trace_level > 0) {
            sprintf(wrFileptr, "warp_read_task%d.txt", h9taskid);
            sprintf(grFileptr, "grad_read_task%d.txt", h9taskid);
            sprintf(owFileptr, "owm_write_task%d.txt", h9taskid);
            sprintf(compFileptr, "compute_task%d.txt", h9taskid);
            sprintf(ParserFileptr, "parser_task%d.txt", h9taskid);
            sprintf(fillFileptr, "fillCtrl_task%d.txt", h9taskid);
            sprintf(victimFileptr, "victimCtrl_task%d.txt", h9taskid);
            if (descr.irt_mode == 7) {
                sprintf(VinterpFileptr, "rescale_vinterp_task%d.txt", h9taskid);
            }
            wrFile     = fopen(wrFileptr, "w");
            grFile     = fopen(grFileptr, "w");
            owFile     = fopen(owFileptr, "w");
            compFile   = fopen(compFileptr, "w");
            parseFile  = fopen(ParserFileptr, "w");
            fillFile   = fopen(fillFileptr, "w");
            victimFile = fopen(victimFileptr, "w");
            if (descr.warp_stride != 0) {
                sprintf(wshflFileptr, "warp_shfl_task%d.txt", h9taskid);
                sprintf(gshflFileptr, "grad_shfl_task%d.txt", h9taskid);
                wshflFile = fopen(wshflFileptr, "w");
                gshflFile = fopen(gshflFileptr, "w");
            }
        }
        // VinterpFile = fopen(VinterpFileptr, "w");
        irt_top->irt_desc_print(irt_top, irt_top->task[e_irt_resamp]);
    }
    /*
    //---------------------------------------
    // reshuffle warp-image per warp_stride value
    warp_shake(irt_top->irt_desc[irt_top->task[e_irt_resamp]].warp_stride);
    */
    //---------------------------------------
    // compute warp fetch sequence for suspension buffer
    //  IRT_TRACE("warp_read %d \n",cycle);
    resamp_done[e_resamp_wread] = warp_read<e_irt_wrm>(mimage_addr,
                                                       mmem_rd,
                                                       mlsb_pad_rd,
                                                       mmsb_pad_rd,
                                                       mimage_meta_out,
                                                       mimage_rd_data,
                                                       mimage_meta_in,
                                                       mmem_rd_data_valid);
    //---------------------------------------
    if (descr.irt_mode != 7) {
        //   IRT_TRACE("garp_read %d \n",cycle);
        resamp_done[e_resamp_gread] = warp_read<e_irt_grm>(gimage_addr,
                                                           gmem_rd,
                                                           glsb_pad_rd,
                                                           gmsb_pad_rd,
                                                           gimage_meta_out,
                                                           gimage_rd_data,
                                                           gimage_meta_in,
                                                           gmem_rd_data_valid);
        warp_parser(iimage_addr, iimage_meta_out, imem_rd, ilsb_pad_rd, imsb_pad_rd);
        fill_ctrl(iimage_rd_data, iimage_meta_in, imem_rd_data_valid);
        victim_ctrl();
    }
    //---------------------------------------
    if (descr.irt_mode == e_irt_rescale) {
        //      IRT_TRACE("compute_rescale %d \n",cycle);
        compute_rescale();
    } else {
        compute();
    }
    //---------------------------------------
    //      IRT_TRACE("owm_writes %d \n",cycle);
    resamp_done[e_resamp_owrite] = owm_writes(oimage_addr, omem_wr, oimage_wr_data);
    //---------------------------------------
    // TODO - fixed done for rescale
    // bool done = (descr.irt_mode != 7) ? resamp_done[e_resamp_owrite] : resamp_done[e_resamp_wread];
    bool done = resamp_done[e_resamp_owrite];
    if ((cycle % 1000) == 0) {
        IRT_TRACE("resamp_done[wread:parser:compute:owrite] [%d:%d:%d:%d] @cycle %d\n",
                  resamp_done[e_resamp_wread],
                  w_parser_cnt.done,
                  comp_parser_cnt.done,
                  resamp_done[e_resamp_owrite],
                  cycle);
    }
    if (done) {
        resamp_mem_dealloc(descr);
        if (max_log_trace_level > 0) {
            fclose(wrFile);
            fclose(owFile);
            fclose(compFile);
            fclose(parseFile);
            fclose(fillFile);
            fclose(victimFile);
            if (descr.warp_stride != 0) {
                fclose(wshflFile);
                fclose(gshflFile);
            }
        }
        // irt_top->task[e_irt_resamp]++;
        task_start = 0;
        h9taskid++;
        //===========Generating Resampler Perf Data for FWD/BWD1/BWD2==========.
        uint8_t  irt_mode    = descr.irt_mode;
        uint16_t warp_ps     = (1 << descr.image_par[MIMAGE].Ps);
        uint16_t grad_ps     = (1 << descr.image_par[GIMAGE].Ps);
        uint16_t out_ps      = (1 << descr.image_par[OIMAGE].Ps);
        uint16_t Hm          = descr.image_par[MIMAGE].H;
        uint16_t Wm          = descr.image_par[MIMAGE].W;
        uint32_t warp_bytes  = ceil(Wm / (IRT_IFIFO_WIDTH / warp_ps)) * IRT_IFIFO_WIDTH * Hm;
        uint32_t input_bytes = (irt_mode == 6) ? 0 : num_req * IRT_IFIFO_WIDTH;
        uint32_t grad_bytes  = (irt_mode == 4) ? 0 : ceil(Wm / (IRT_IFIFO_WIDTH / grad_ps)) * IRT_IFIFO_WIDTH * Hm;
        uint32_t out_bytes   = (irt_mode == 6) ? (num_victims * IRT_IFIFO_WIDTH)
                                             : ceil(Wm / (IRT_IFIFO_WIDTH / out_ps)) * IRT_IFIFO_WIDTH * Hm;
        uint32_t comp_pixels  = Wm * Hm;
        uint32_t rd_bw_cycles = (warp_bytes + input_bytes + grad_bytes) / BYTES_PER_CYCLE;
        uint32_t wr_bw_cycles = (out_bytes / BYTES_PER_CYCLE);
        uint32_t comp_bubbles = (two_lookup_cycles > num_tc_stall_cycle) ? two_lookup_cycles : 0;
        // uint32_t compute_cycles = cycle_per_task; //compue cycles with all bubbles.
        uint32_t compute_cycles = cycle_per_task + two_lookup_cycles; // compue cycles with all bubbles.
        uint32_t bound_cycles   = max(max(rd_bw_cycles, compute_cycles), wr_bw_cycles);
        float    comp_proc_size = (float)avg_proc_size / (float)proc_comp_cnt;
        float    avg_proc_size  = (float)comp_pixels / (float)bound_cycles;
        float    rd_bw =
            min((float)BW,
                (float)((float)(warp_bytes + input_bytes + grad_bytes) / (float)bound_cycles) * (float)FREQ); // GBPS
        float wr_bw = min((float)BW, (float)((float)out_bytes / (float)bound_cycles) * (float)FREQ); // GBPS

        //===============End of Resampler Perf Data=====================.
        IRT_TRACE("resamp_done[wread:parser:compute:owrite:h9taskid] [%d:%d:%d:%d:%d]\n",
                  resamp_done[e_resamp_wread],
                  w_parser_cnt.done,
                  comp_parser_cnt.done,
                  resamp_done[e_resamp_owrite],
                  h9taskid);
        IRT_TRACE("Done compute_timeout %d at task end \n", compute_timeout);
        //---------------------------------------
        if (descr.irt_mode != 7) {
            IRT_TRACE("========================================================\n");
            IRT_TRACE("|                   Cache Perf Summary                 |\n");
            IRT_TRACE("========================================================\n");
            IRT_TRACE(" Total Lookups: %d HITS: %d  Hit-: %d \n", lookup_cnt, hit_cnt, (hit_cnt * 100) / lookup_cnt);
            IRT_TRACE("========================================================\n");
            IRT_TRACE("|                Resamp Perf Summary                   |\n");
            IRT_TRACE("========================================================\n");
            IRT_TRACE("| Comp_Proc_Size : %f  Avg_Proc_Size: %f   \n", comp_proc_size, avg_proc_size);
            IRT_TRACE("| Bound_Cycles   : %d                             \n", bound_cycles);
            IRT_TRACE("| Read_BW_Cycles : %d \t Read_BW      : %f  \n", rd_bw_cycles, rd_bw);
            IRT_TRACE("| Write_BW_Cycles: %d \t Write_BW     : %f   \n", wr_bw_cycles, wr_bw);
            IRT_TRACE("| Num miss req : %d \t num victims req : %d   \n", num_req, num_victims);
            IRT_TRACE("| TC_Stall_cycles: %d \t N-1 Conf:%d \t N-2 Conf:%d \n",
                      num_tc_stall_cycle,
                      num_nm1_conflicts,
                      num_nm2_conflicts);
            IRT_TRACE("=================================================================\n");
            IRT_TRACE("| num_stall_misq : %d  two_lookup_cycles:%d                          \n",
                      num_stall_misq,
                      two_lookup_cycles);
            IRT_TRACE("=================================================================\n");

            IRT_TRACE("========================================================\n");
            IRT_TRACE("|                   Other Debug Calc                   |\n");
            IRT_TRACE("========================================================\n");
            IRT_TRACE("| warp_bytes : %d  input_bytes: %d  \n", warp_bytes, input_bytes);
            IRT_TRACE("| grad_bytes : %d  out_bytes  : %d  \n", grad_bytes, out_bytes);
            IRT_TRACE("| compute_cycles: %d  comp_bubbles  : %d  \n", compute_cycles, comp_bubbles);
            IRT_TRACE("========================================================\n");

            irt_top->irt_top_sig.perf.rd_bw          = rd_bw;
            irt_top->irt_top_sig.perf.wr_bw          = wr_bw;
            irt_top->irt_top_sig.perf.comp_proc_size = comp_proc_size;
            irt_top->irt_top_sig.perf.avg_proc_size  = avg_proc_size;
            irt_top->irt_top_sig.perf.bound_cycles   = bound_cycles;
            irt_top->irt_top_sig.perf.compute_cycles = compute_cycles;
            irt_top->irt_top_sig.perf.rd_bw_cycles   = rd_bw_cycles;
            irt_top->irt_top_sig.perf.wr_bw_cycles   = wr_bw_cycles;
        }
        //---------------------------------------
        // EOT CHECKS
        inputQ.pop();
        if (!victimQ.empty() || !outputQ.empty() || !inputQ.empty() || !owmDataQ.empty()) {
            if (!inputQ.empty()) {
                inputQ.pop();
            } else {
                IRT_TRACE("FAILED - TEST FINISHED UNEXPECTED victimQ:outputQ:inputQ:owmDataQ = [%lu:%lu:%lu:%lu]\n",
                          victimQ.size(),
                          outputQ.size(),
                          inputQ.size(),
                          owmDataQ.size());
                IRT_TRACE_TO_RES(test_res, "FAILED - TEST FINISHED UNEXPECTED \n");
                IRT_CLOSE_FAILED_TEST(0);
            }
        }
#ifdef RUN_WITH_SV
        irt_top->irt_top_sig.dv_owm_task_done = 1;
#endif
    }
    return (done);
}

// IRT_top is called every cycle
// iimage_addr - address to the input image to read data. Part of the large image.
// mem_rd - indication for memory read.
// lsb_pad_rd - lsb padding. Number of irrelevant bytes in the begging of the 128 bytes buffer.
// msb_pad_rd - msb padding. Number of irrelevant bytes in the end of the 128 bytes buffer.
// iimage_meta_out - context to be received when the data returns from the read operation.
// iimage_rd_data - input data
// iimage_meta_in - returned meta data from the read.
// oimage_addr - address to write rotated output image.
// mem_write - enable write to the output address.
// oimage_wr_data - rotated 128 bytes output data and enable per pixel to decide whether to write to memory.

// bool IRT_top::run (uint64_t &iimage_addr, bool& imem_rd, uint16_t& ilsb_pad_rd, uint16_t& imsb_pad_rd,
// meta_data_struct &iimage_meta_out, 				  const bus128B_struct& iimage_rd_data, meta_data_struct
// iimage_meta_in, uint64_t& oimage_addr, bool& mem_write, bus128B_struct &oimage_wr_data, 				  uint64_t&
// mimage_addr, bool& mmem_rd, uint16_t& mlsb_pad_rd, uint16_t& mmsb_pad_rd, meta_data_struct& mimage_meta_out,
// const bus128B_struct& mimage_rd_data, meta_data_struct mimage_meta_in) {
bool IRT_top::run(uint64_t&             iimage_addr,
                  bool&                 imem_rd,
                  uint16_t&             ilsb_pad_rd,
                  uint16_t&             imsb_pad_rd,
                  meta_data_struct&     iimage_meta_out,
                  const bus128B_struct& iimage_rd_data,
                  meta_data_struct      iimage_meta_in,
                  bool                  imem_rd_data_valid,
                  uint64_t&             mimage_addr,
                  bool&                 mmem_rd,
                  uint16_t&             mlsb_pad_rd,
                  uint16_t&             mmsb_pad_rd,
                  meta_data_struct&     mimage_meta_out,
                  const bus128B_struct& mimage_rd_data,
                  meta_data_struct      mimage_meta_in,
                  bool                  mmem_rd_data_valid,
                  uint64_t&             gimage_addr,
                  bool&                 gmem_rd,
                  uint16_t&             glsb_pad_rd,
                  uint16_t&             gmsb_pad_rd,
                  meta_data_struct&     gimage_meta_out,
                  const bus128B_struct& gimage_rd_data,
                  meta_data_struct      gimage_meta_in,
                  bool                  gmem_rd_data_valid,
                  uint64_t&             oimage_addr,
                  uint8_t&              mem_write,
                  bus128B_struct&       oimage_wr_data)
{

    bus128B_struct  ofifo_out, ififo_din, mfifo_din;
    bus8ui16_struct irt_ibli_out;
    // bus8XiYi_float_struct irt_mbli_out;
    bus16B_struct rm_data_out[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
    bus64B_struct mm_data_out[IRT_MESH_MEM_ROW_BANKS]
                             [IRT_MESH_MEM_COL_BANKS]; /*rm_dout[2], moved to external struct signals*/
    ;
    bus128B_struct   mfifo_dout;
    bus32B_struct    ififo_dout;
    bus16ui16_struct ii_8x16_int_region[IRT_ROT_MEM_ROW_BANKS];
    bus16ui64_struct mi_8x16_int_region[IRT_MESH_MEM_ROW_BANKS];
    bus16f_struct    ii_8x16_int_region_flags[IRT_ROT_MEM_ROW_BANKS], mi_8x16_int_region_flags[IRT_MESH_MEM_ROW_BANKS];
    bus8XiYi_struct  mexp_dout;
    meta_data_struct ififo_meta_in, ififo_meta, mfifo_meta_in, mfifo_meta;
    // p2x2_struct p2x2[8];
    // irt_top_sig.ofifo_push = 0; irt_top_sig.mofifo_push = 0;
    float Xi0 = 0, Yi0 = 0;
    float Xmi0 = 0, Ymi0 = 0;
    // meta_data_struct rm_Xi_start; moved to external struct signals
    bool ififo_push = 0, mfifo_push = 0, imsb_lsb_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
         mmsb_lsb_sel[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS];

    memset(&ofifo_out, 0, sizeof(ofifo_out));
    memset(&ififo_din, 0, sizeof(ififo_din));
    memset(&mfifo_din, 0, sizeof(mfifo_din));
    memset(&irt_ibli_out, 0, sizeof(irt_ibli_out));
    // memset(&irt_mbli_out, 0, sizeof(irt_mbli_out));
    memset(&rm_data_out, 0, sizeof(rm_data_out));
    memset(&mfifo_dout, 0, sizeof(mfifo_dout));
    memset(&ififo_dout, 0, sizeof(ififo_dout));
    memset(&ii_8x16_int_region, 0, sizeof(ii_8x16_int_region));
    memset(&ii_8x16_int_region_flags, 0, sizeof(ii_8x16_int_region_flags));
    memset(&mexp_dout, 0, sizeof(mexp_dout));
    memset(&ififo_meta_in, 0, sizeof(ififo_meta_in));
    memset(&ififo_meta, 0, sizeof(ififo_meta));
    memset(&mfifo_meta_in, 0, sizeof(mfifo_meta_in));
    memset(&mfifo_meta, 0, sizeof(mfifo_meta));
    memset(&imsb_lsb_sel, 0, sizeof(imsb_lsb_sel));
    memset(&mmsb_lsb_sel, 0, sizeof(mmsb_lsb_sel));

    //=========================================================
    // H6 ARCH MODEL FOR ROTATION/AFFINE/PROJ/MESH TASKS
    //=========================================================
    // TODO - use pointer irt_desc[0].irt_mode instead of '0' index
    if (irt_desc[0].irt_mode <= 3) {
        /*********************/
        /* Mesh core section */
        /*********************/
        // Mesh read manager. Reads the mesh image from the memory. Write the image to the mesh FIFO
        irt_done[e_irt_mrm] = IRT_MIRM_block->run(mimage_addr,
                                                  mimage_meta_out,
                                                  mmem_rd,
                                                  mimage_rd_data,
                                                  mimage_meta_in,
                                                  mfifo_din,
                                                  mfifo_push,
                                                  mfifo_meta_in,
                                                  mfifo_full,
                                                  mlsb_pad_rd,
                                                  mmsb_pad_rd);

        // keeps the data from MRM until it can be writen to mesh write manager to mesh memory
        IRT_MFIFO_block->run(mfifo_push,
                             mfifo_pop,
                             mfifo_read,
                             mfifo_din,
                             mfifo_meta_in,
                             mfifo_dout,
                             mfifo_meta,
                             mfifo_empty,
                             mfifo_full);

        // Mesh memory write manager. Copies the data from mesh FIFO to mesh buffer.
        irt_done[e_irt_mmwm] = IRT_MMWM_block->run(mfifo_empty,
                                                   mfifo_pop,
                                                   mfifo_read,
                                                   mfifo_dout,
                                                   mfifo_meta,
                                                   irt_top_sig.mm_wr_addr,
                                                   irt_top_sig.mm_wr_sel,
                                                   irt_top_sig.mm_din,
                                                   irt_top_sig.mm_meta_addr,
                                                   irt_top_sig.mm_meta_data);

        // Save the content of mesh data in mesh memory
        IRT_MMEM_block->run(1,
                            irt_top_sig.mm_wr_addr,
                            irt_top_sig.mm_wr_sel,
                            irt_top_sig.mm_din,
                            irt_top_sig.mm_meta_addr,
                            irt_top_sig.mm_meta_data,
                            irt_top_sig.mm_rd_addr,
                            irt_top_sig.mm_rd_sel,
                            mm_data_out,
                            0,
                            micc_task);

        irt_top_sig.mofifo_push = 0;
        // Mesh image coordinate control. Main state machine that control the flow of the mesh pixels generation. 8
        // pixels to calculate. irt_done[irt_micc2] = IRT_MICC2_block->run(Xm0, Ym, micc_task, mesh_pixel_valid,
        // irt_top_sig.mofifo_push, mesh_ofifo_valid, mesh_task_end, adj_proc_size);

        irt_done[e_irt_micc] = IRT_MICC_block->run(Xm0,
                                                   Ym,
                                                   micc_task,
                                                   irt_top_sig.YmT,
                                                   irt_top_sig.YmB,
                                                   mesh_pixel_valid,
                                                   irt_top_sig.mofifo_push,
                                                   mesh_ofifo_valid,
                                                   mesh_task_end,
                                                   mesh_adj_proc_size);

        // Mesh image interpolation region calculation. Which input pixel are needed for the calculation.
        IRT_MIRC_block->run(Xm0,
                            Ym,
                            irt_top_sig.Xm_fixed,
                            irt_top_sig.Ym_fixed,
                            irt_top_sig.XmL,
                            irt_top_sig.XmR,
                            irt_top_sig.YmT,
                            irt_top_sig.YmB,
                            micc_task,
                            irt_top_sig.mofifo_push,
                            mesh_adj_proc_size);

        // Mesh memory read manager. Using the mesh image coordinates, calculates the address of the mesh pixels.
        IRT_MMRM_block->run(irt_top_sig.XmL,
                            irt_top_sig.XmR,
                            irt_top_sig.YmT,
                            irt_top_sig.YmB,
                            irt_top_sig.mm_bank_row,
                            irt_top_sig.mm_rd_sel,
                            irt_top_sig.mm_rd_addr,
                            mm_rd_mode,
                            mrd_shift,
                            mmsb_lsb_sel,
                            YmT_bank_row,
                            mm_bg_flags,
                            micc_task,
                            irt_top_sig.mofifo_push,
                            Xm0,
                            Ym,
                            e_irt_rmrm_caller_irt_mmrm);

        // Use the mesh memory to read the mesh pixels.
        IRT_MMEM_block->run(0,
                            irt_top_sig.mm_wr_addr,
                            irt_top_sig.mm_wr_sel,
                            irt_top_sig.mm_din,
                            irt_top_sig.mm_meta_addr,
                            irt_top_sig.mm_meta_data,
                            irt_top_sig.mm_rd_addr,
                            irt_top_sig.mm_rd_sel,
                            mm_data_out,
                            irt_top_sig.mofifo_push,
                            micc_task);

        // Mesh window calculation. Adjust the returned pixels to the rectangle memory.
        IRT_MIWC_block->run(mm_data_out,
                            mm_bg_flags,
                            mm_rd_mode,
                            mrd_shift,
                            mmsb_lsb_sel,
                            irt_top_sig.XmL,
                            irt_top_sig.YmT,
                            YmT_bank_row,
                            mi_8x16_int_region,
                            mi_8x16_int_region_flags,
                            irt_top_sig.mofifo_push,
                            Ym,
                            Xm0,
                            micc_task);

        // int XL8 = (irt_top_sig.XL>>3)<<3;
        // int16_t XmL8 = irt_top_sig.XmL;

        // Fetch for each mesh pixel the relevant 2x2 input pixels for the interpolation.
        int16_t XmL = 0, XmR = 0, YmT = 0, YmB = 0;
        for (uint8_t index = 0; index < mesh_adj_proc_size; index++) {
            // saturation to horizontal image boundaries
            if ((irt_top_sig.Xm_fixed[index] >> IRT_MESH_G_PREC) >=
                (int64_t)irt_desc[micc_task].image_par[MIMAGE].S - 1) { // outside of horizontal boundary
                irt_top_sig.Xm_fixed[index] = ((int64_t)irt_desc[micc_task].image_par[MIMAGE].S - 1) << IRT_MESH_G_PREC;
            }

            // saturation to vertical image  boundaries
            if ((irt_top_sig.Ym_fixed[index] >> IRT_MESH_G_PREC) >=
                (int64_t)irt_desc[micc_task].image_par[MIMAGE].H - 1) { // outside of vertical boundary
                irt_top_sig.Ym_fixed[index] = ((int64_t)irt_desc[micc_task].image_par[MIMAGE].H - 1) << IRT_MESH_G_PREC;
            }

            XmL = (int16_t)(irt_top_sig.Xm_fixed[index] >> IRT_MESH_G_PREC);
            YmT = (int16_t)(irt_top_sig.Ym_fixed[index] >> IRT_MESH_G_PREC);
            XmL = IRT_top::IRT_UTILS::irt_sat_int16(XmL, 0, (int16_t)irt_desc[micc_task].image_par[MIMAGE].S - 1);
            XmR = IRT_top::IRT_UTILS::irt_sat_int16(XmL + 1, 0, (int16_t)irt_desc[micc_task].image_par[MIMAGE].S - 1);
            YmT = IRT_top::IRT_UTILS::irt_sat_int16(YmT, 0, (int16_t)irt_desc[micc_task].image_par[MIMAGE].H - 1);
            YmB = IRT_top::IRT_UTILS::irt_sat_int16(YmT + 1, 0, (int16_t)irt_desc[micc_task].image_par[MIMAGE].H - 1);
            // if (irt_top_sig.mofifo_push)
            //	IRT_TRACE("[index %d] Xmi0 %f, irt_top_sig.XmL %d, XmL %d, XmR %d, YmT %d, YmB %d, Xm %f\n", index,
            //(double)Xmi0, irt_top_sig.XmL, XmL, XmR, YmT, YmB, (double)irt_top_sig.Xm_fixed[index]/pow(2, 31));
            IRT_M2x2_block->run(index,
                                irt_top_sig.XmL,
                                irt_top_sig.XmL + (XmR - XmL),
                                irt_top_sig.YmT,
                                irt_top_sig.YmT + (YmB - YmT),
                                Xmi0,
                                Ymi0,
                                irt_top_sig.Xm_fixed[index],
                                irt_top_sig.Ym_fixed[index],
                                mi_8x16_int_region,
                                mi_8x16_int_region_flags,
                                irt_top_sig.mp2x2x64[index],
                                micc_task,
                                irt_top_sig.mofifo_push);
            // IRT_M2x2_block->run(index, XmL, XmR, YmT, YmB, Xmi0, Ymi0, irt_top_sig.Xm_fixed[index],
            // irt_top_sig.Ym_fixed[index], mi_8x16_int_region, mi_8x16_int_region_flags, irt_top_sig.mp2x2x64[index],
            // micc_task, irt_top_sig.mofifo_push);
        }

        uint64_t mp2x2x64[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES];
        uint16_t x_ui16[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES], y_ui16[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES];
        float    x_fp32[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES], y_fp32[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES];
        uint32_t x_ui32[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES], y_ui32[IRT_ROT_MAX_PROC_SIZE][IRT_MESH_FP32_BYTES];

        if (irt_top_sig.mofifo_push) {
            for (int index = 0; index < mesh_adj_proc_size; index++) {
                for (int pixel = 0; pixel < IRT_MESH_FP32_BYTES; pixel++) {
                    mp2x2x64[index][pixel] = irt_top_sig.mp2x2x64[index].pix[pixel];
                    x_ui32[index][pixel]   = (uint32_t)(irt_top_sig.mp2x2x64[index].pix[pixel] & 0xffffffff);
                    y_ui32[index][pixel] =
                        (uint32_t)((irt_top_sig.mp2x2x64[index].pix[pixel] >> (8 * IRT_MESH_FP32_BYTES)) & 0xffffffff);
                    x_ui16[index][pixel] = mp2x2x64[index][pixel] & 0xffff;
                    y_ui16[index][pixel] = (mp2x2x64[index][pixel] >> (8 * IRT_MESH_FP32_BYTES / 2)) & 0xffff;
                    x_fp32[index][pixel] = (float)IRT_top::IRT_UTILS::irt_ui16_to_i16(x_ui16[index][pixel]) /
                                           (float)pow(2.0, irt_desc[micc_task].mesh_point_location);
                    y_fp32[index][pixel] = (float)IRT_top::IRT_UTILS::irt_ui16_to_i16(y_ui16[index][pixel]) /
                                           (float)pow(2.0, irt_desc[micc_task].mesh_point_location);

                    irt_top_sig.mp2x2_fp32_x[index].pix[pixel] =
                        irt_desc[micc_task].mesh_format == e_irt_mesh_fp32
                            ? IRT_top::IRT_UTILS::irt_fp32_to_float(x_ui32[index][pixel])
                            : x_fp32[index][pixel];
                    irt_top_sig.mp2x2_fp32_y[index].pix[pixel] =
                        irt_desc[micc_task].mesh_format == e_irt_mesh_fp32
                            ? IRT_top::IRT_UTILS::irt_fp32_to_float(y_ui32[index][pixel])
                            : y_fp32[index][pixel];

                    irt_top_sig.mp2x2_fp32_x[index].pix_bg[pixel] = irt_top_sig.mp2x2x64[index].pix_bg[pixel];
                    irt_top_sig.mp2x2_fp32_y[index].pix_bg[pixel] = irt_top_sig.mp2x2x64[index].pix_bg[pixel];
                }
                irt_top_sig.mp2x2_fp32_x[index].weights[0]       = irt_top_sig.mp2x2x64[index].weights[0];
                irt_top_sig.mp2x2_fp32_x[index].weights[1]       = irt_top_sig.mp2x2x64[index].weights[1];
                irt_top_sig.mp2x2_fp32_y[index].weights[0]       = irt_top_sig.mp2x2x64[index].weights[0];
                irt_top_sig.mp2x2_fp32_y[index].weights[1]       = irt_top_sig.mp2x2x64[index].weights[1];
                irt_top_sig.mp2x2_fp32_x[index].weights_fixed[0] = irt_top_sig.mp2x2x64[index].weights_fixed[0];
                irt_top_sig.mp2x2_fp32_x[index].weights_fixed[1] = irt_top_sig.mp2x2x64[index].weights_fixed[1];
                irt_top_sig.mp2x2_fp32_y[index].weights_fixed[0] = irt_top_sig.mp2x2x64[index].weights_fixed[0];
                irt_top_sig.mp2x2_fp32_y[index].weights_fixed[1] = irt_top_sig.mp2x2x64[index].weights_fixed[1];
            }
        }

        int                   mbli_bg = 0;
        bus8XiYi_float_struct mbli_out;

        for (int index = 0; index < mesh_adj_proc_size; index++) {
            // Bilinear interpolation
            mbli_out.pix[index].X = IRT_MBLI_block->run(
                irt_top_sig.mp2x2_fp32_x[index], irt_desc[micc_task].bg, irt_top_sig.mofifo_push, micc_task);
            mbli_out.pix[index].Y = IRT_MBLI_block->run(
                irt_top_sig.mp2x2_fp32_y[index], irt_desc[micc_task].bg, irt_top_sig.mofifo_push, micc_task);
            mbli_bg = (mbli_bg << 1) +
                      (irt_top_sig.mp2x2_fp32_x[index].pix_bg[0] | irt_top_sig.mp2x2_fp32_x[index].pix_bg[1] |
                       irt_top_sig.mp2x2_fp32_x[index].pix_bg[2] | irt_top_sig.mp2x2_fp32_x[index].pix_bg[3]);
            mbli_bg += (irt_top_sig.mp2x2_fp32_y[index].pix_bg[0] | irt_top_sig.mp2x2_fp32_y[index].pix_bg[1] |
                        irt_top_sig.mp2x2_fp32_y[index].pix_bg[2] | irt_top_sig.mp2x2_fp32_y[index].pix_bg[3]);
        }

#ifdef RUN_WITH_SV
        irt_top_sig.mesh_ofifo_valid = mesh_ofifo_valid;
#endif
        if (irt_top_sig.mofifo_push && mesh_ofifo_valid == 0) {
#if 0
         for (int index = 0; index < mesh_adj_proc_size; index++) {
            IRT_TRACE("MESH Results for pixel [%d, %d] at task %d at cycle %d:\n", Ym >> 1, (Xm0 >> 1) + index, micc_task, cycle); //Ym = line * 2, Xm0 = pixel * 2
            IRT_TRACE("weights[%8x,%8x] bli input: y: pix[%.8f,%.8f][%.8f,%.8f], x: pix[%.2f,%.2f][%.2f,%.2f], output[X, Y]:[%.8f,%.8f]\n",
                  irt_top_sig.mp2x2x64[index].weights_fixed[0], irt_top_sig.mp2x2x64[index].weights_fixed[1],
                  irt_top_sig.mp2x2_fp32_y[index].pix[0], irt_top_sig.mp2x2_fp32_y[index].pix[1], irt_top_sig.mp2x2_fp32_y[index].pix[2], irt_top_sig.mp2x2_fp32_y[index].pix[3],
                  irt_top_sig.mp2x2_fp32_x[index].pix[0], irt_top_sig.mp2x2_fp32_x[index].pix[1], irt_top_sig.mp2x2_fp32_x[index].pix[2], irt_top_sig.mp2x2_fp32_x[index].pix[3],
                  mbli_out.pix[index].X, mbli_out.pix[index].Y);
         }
#endif

            // for (int index = 0; index < mesh_adj_proc_size; index++)
            //	IRT_TRACE("[%d, %d] = [%.2f, %.2f], task %d\n", Ym >> 1, (Xm0 >> 1) + index, mbli_out.pix[index].X,
            // mbli_out.pix[index].Y, micc_task);
            for (int index = 0; index < mesh_adj_proc_size; index++) {
                mesh_ofifo.pix[index].X           = mbli_out.pix[index].X;
                mesh_ofifo.pix[index].Y           = mbli_out.pix[index].Y;
                irt_top_sig.mbli_out.pix[index].X = mbli_out.pix[index].X;
                irt_top_sig.mbli_out.pix[index].Y = mbli_out.pix[index].Y;
            }
            mesh_ofifo_valid = 1;
#ifdef RUN_WITH_SV
            irt_top_sig.mesh_mmrm_pxl_oob  = mbli_bg;
            irt_top_sig.mesh_adj_proc_size = mesh_adj_proc_size;
            irt_top_sig.mesh_intp_task_end = mesh_task_end;
#endif
        }
        /************************/
        /* Rotator core section */
        /************************/

        // Input read manager. Reads the input image from the memory. Write the image to the input FIFO
        irt_done[e_irt_irm] = IRT_IIRM_block->run(iimage_addr,
                                                  iimage_meta_out,
                                                  imem_rd,
                                                  iimage_rd_data,
                                                  iimage_meta_in,
                                                  ififo_din,
                                                  ififo_push,
                                                  ififo_meta_in,
                                                  ififo_full,
                                                  ilsb_pad_rd,
                                                  imsb_pad_rd);

        // keeps the data from the IRM until it can be writen to rotation memory
        IRT_IFIFO_block->run(ififo_push,
                             ififo_pop,
                             ififo_read,
                             ififo_din,
                             ififo_meta_in,
                             ififo_dout,
                             ififo_meta,
                             ififo_empty,
                             ififo_full);

        // Rotation memory write manager. Copies the data from the input FIFO to the rotator's buffer.
        irt_done[e_irt_rmwm] = IRT_RMWM_block->run(ififo_empty,
                                                   ififo_pop,
                                                   ififo_read,
                                                   ififo_dout,
                                                   ififo_meta,
                                                   irt_top_sig.rm_wr_addr,
                                                   irt_top_sig.rm_wr_sel,
                                                   irt_top_sig.rm_din,
                                                   irt_top_sig.rm_meta_addr,
                                                   irt_top_sig.rm_meta_data);

        // Save the content of the stripe to the rotation memory
        IRT_RMEM_block->run(1,
                            irt_top_sig.rm_wr_addr,
                            irt_top_sig.rm_wr_sel,
                            irt_top_sig.rm_din,
                            irt_top_sig.rm_meta_addr,
                            irt_top_sig.rm_meta_data,
                            irt_top_sig.rm_rd_addr,
                            irt_top_sig.rm_rd_sel,
                            rm_data_out,
                            0,
                            oicc_task);
        /*
           if (irt_done[irt_oicc_done]==0)
           ofifo_push = 1;
           else
           ofifo_push = 0;
           */
        irt_top_sig.ofifo_push = 0;
        irt_top_sig.cfifo_push = 0;
        // oicc_valid = 0;
        int16_t XiL8;
        int     ibli_bg = 0;

        switch (irt_cfg.flow_mode) {
            case e_irt_flow_nCFIFO_fixed_adaptive_wxh:
                // old flow
                irt_done[e_irt_oicc] = IRT_OICC_block->run(Xo0,
                                                           Yo,
                                                           oicc_task,
                                                           irt_top_sig.YiT,
                                                           irt_top_sig.YiB,
                                                           pixel_valid,
                                                           irt_top_sig.ofifo_push,
                                                           ofifo_full || mesh_ofifo_valid == 0,
                                                           task_end,
                                                           adj_proc_size);

                if (irt_top_sig.ofifo_push)
                    mesh_ofifo_valid = 0;
                // Input image interpolation region calculation. Which input pixel are needed for the calculation.
                IRT_IIRC_block->run(Xo0,
                                    Yo,
                                    irt_top_sig.Xi_fixed,
                                    irt_top_sig.Yi_fixed,
                                    irt_top_sig.XiL,
                                    irt_top_sig.XiR,
                                    irt_top_sig.YiT,
                                    irt_top_sig.YiB,
                                    oicc_task,
                                    irt_top_sig.ofifo_push,
                                    adj_proc_size);

                // Rotation memory read manager. Using the input image coordinates, calculates the address of the input
                // pixels.
                IRT_RMRM_block->run(irt_top_sig.XiL,
                                    irt_top_sig.XiR,
                                    irt_top_sig.YiT,
                                    irt_top_sig.YiB,
                                    irt_top_sig.rm_bank_row,
                                    irt_top_sig.rm_rd_sel,
                                    irt_top_sig.rm_rd_addr,
                                    rm_rd_mode,
                                    ird_shift,
                                    imsb_lsb_sel,
                                    YT_bank_row,
                                    rm_bg_flags,
                                    oicc_task,
                                    irt_top_sig.ofifo_push,
                                    Xo0,
                                    Yo,
                                    e_irt_rmrm_caller_irt_top);

                irt_done[e_irt_iiirc] = 1;
                // Then go to IRT_RMEM_block
                break;

            case e_irt_flow_wCFIFO_fixed_adaptive_2x2:
            case e_irt_flow_wCFIFO_fixed_adaptive_wxh:
                // new flow with coordinate FIFO and new rate control

                // output image coordinate control. Main state machine that control the flow of the output generation. 8
                // pixels to calculate.
                irt_done[e_irt_oicc] = IRT_OICC2_block->run(Xo0,
                                                            Yo,
                                                            oicc_task,
                                                            pixel_valid,
                                                            irt_top_sig.cfifo_push,
                                                            cfifo_emptyness,
                                                            mesh_ofifo_valid,
                                                            task_end,
                                                            irt_top_sig.cfifo_push_size,
                                                            cfifo_data_in);

                if (irt_top_sig.cfifo_push && irt_desc[oicc_task].irt_mode == e_irt_mesh) {
                    mesh_ofifo_valid = 0;
                    // IRT_TRACE("MESH OFIFO pop at cycle %d\n", cycle);
                    for (int index = 0; index < irt_desc[micc_task].proc_size; index++) {
                        // IRT_TRACE("[%.2f,%.2f] ", (double)cfifo_data_in[index].Xi_fixed/pow(2.0,31),
                        // (double)cfifo_data_in[index].Yi_fixed / pow(2.0, 31));
                    }
                    // IRT_TRACE("\n");
                }

                // push to CFIFO
                IRT_CFIFO_block->run(irt_top_sig.cfifo_push,
                                     0,
                                     irt_top_sig.cfifo_push_size,
                                     0,
                                     cfifo_data_in,
                                     cfifo_data_out,
                                     cfifo_emptyness,
                                     cfifo_fullness);
                //---------------------------------------
                if (irt_top_sig.cfifo_push == 1) {
                    for (uint8_t s = 0; s < irt_top_sig.cfifo_push_size; s++) { // ROTSIM_DV_INTGR
                        irt_top_sig.dv_cfifo_data[s] = cfifo_data_in[s]; // ROTSIM_DV_INTGR
                    } // ROTSIM_DV_INTGR
                }
                //---------------------------------------
                // parse CFIFO data for rate calculation )behaves as OICC + old IIRC + RMRM + rate control)
                irt_done[e_irt_iiirc] = IRT_IIRC2_block->run(Xo0,
                                                             Yo,
                                                             pixel_valid,
                                                             irt_top_sig.ofifo_push,
                                                             ofifo_full,
                                                             task_end,
                                                             adj_proc_size,
                                                             irt_top_sig.Xi_fixed,
                                                             irt_top_sig.Yi_fixed,
                                                             irt_top_sig.XiL,
                                                             irt_top_sig.XiR,
                                                             irt_top_sig.YiT,
                                                             irt_top_sig.YiB,
                                                             iirc_task,
                                                             cfifo_fullness,
                                                             irt_top_sig.cfifo_pop,
                                                             irt_top_sig.cfifo_pop_size,
                                                             cfifo_data_out,
                                                             irt_top_sig.rm_rd_sel,
                                                             irt_top_sig.rm_rd_addr);

                switch (irt_cfg.flow_mode) {
                    case e_irt_flow_wCFIFO_fixed_adaptive_wxh:

                        // pop from CFIFO
                        IRT_CFIFO_block->run(0,
                                             irt_top_sig.ofifo_push,
                                             0,
                                             irt_top_sig.cfifo_pop_size,
                                             cfifo_data_in,
                                             cfifo_data_out,
                                             cfifo_emptyness,
                                             cfifo_fullness);

                        // Rotation memory read manager. Using the input image coordinates, calculates the address of
                        // the input pixels.
                        IRT_RMRM_block->run(irt_top_sig.XiL,
                                            irt_top_sig.XiR,
                                            irt_top_sig.YiT,
                                            irt_top_sig.YiB,
                                            irt_top_sig.rm_bank_row,
                                            irt_top_sig.rm_rd_sel,
                                            irt_top_sig.rm_rd_addr,
                                            rm_rd_mode,
                                            ird_shift,
                                            imsb_lsb_sel,
                                            YT_bank_row,
                                            rm_bg_flags,
                                            oicc_task,
                                            irt_top_sig.ofifo_push,
                                            Xo0,
                                            Yo,
                                            e_irt_rmrm_caller_irt_top);

                        // Then go to IRT_RMEM_block
                        break;

                    case e_irt_flow_wCFIFO_fixed_adaptive_2x2:
                        // Use the rotation memory to read the input pixels.
                        //			if (irt_top_sig.ofifo_push) {
                        IRT_RMEM_block->run(0,
                                            irt_top_sig.rm_wr_addr,
                                            irt_top_sig.rm_wr_sel,
                                            irt_top_sig.rm_din,
                                            irt_top_sig.rm_meta_addr,
                                            irt_top_sig.rm_meta_data,
                                            irt_top_sig.rm_rd_addr,
                                            irt_top_sig.rm_rd_sel,
                                            rm_data_out,
                                            irt_top_sig.ofifo_push,
                                            iirc_task);
#ifdef RUN_WITH_SV
                        irt_top_sig.interp_window_size = adj_proc_size;
#endif
                        //---------------------------------------
                        for (uint8_t index = 0; index < adj_proc_size; index++) {
                            for (uint8_t bank_col = 0; bank_col < IRT_ROT_MEM_COL_BANKS; bank_col++) {
                                rm_rd_mode[cfifo_data_out[index].bank_row[bank_col]] =
                                    cfifo_data_out[index].rd_mode[bank_col];
                                ird_shift[cfifo_data_out[index].bank_row[bank_col]] =
                                    cfifo_data_out[index].rd_shift[bank_col];
                                imsb_lsb_sel[cfifo_data_out[index].bank_row[0]][bank_col] =
                                    cfifo_data_out[index].msb_lsb_sel[0][bank_col];
                                imsb_lsb_sel[cfifo_data_out[index].bank_row[1]][bank_col] =
                                    cfifo_data_out[index].msb_lsb_sel[1][bank_col];
                                for (uint8_t j = 0; j < IRT_ROT_MEM_BANK_WIDTH; j++) {
                                    rm_bg_flags[cfifo_data_out[index].bank_row[0]][bank_col][j] =
                                        cfifo_data_out[index].bg_flag[0][bank_col][j];
                                    rm_bg_flags[cfifo_data_out[index].bank_row[1]][bank_col][j] =
                                        cfifo_data_out[index].bg_flag[1][bank_col][j];
                                }
                            }
                            IRT_IIWC_block->run(rm_data_out,
                                                rm_bg_flags,
                                                rm_rd_mode,
                                                ird_shift,
                                                imsb_lsb_sel,
                                                cfifo_data_out[index].XL,
                                                cfifo_data_out[index].YT,
                                                cfifo_data_out[index].bank_row[0],
                                                ii_8x16_int_region,
                                                ii_8x16_int_region_flags,
                                                irt_top_sig.ofifo_push,
                                                Yo,
                                                Xo0,
                                                iirc_task);
                            XiL8 = cfifo_data_out[index].XL;
                            for (uint8_t sindex = 0; sindex < 8; sindex++) {
                                IRT_TRACE_TO_LOG(
                                    15,
                                    IRTLOG,
                                    "ii_8x16_int_region[%d][15:0] %x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x \n",
                                    sindex,
                                    ii_8x16_int_region[sindex].pix[15],
                                    ii_8x16_int_region[sindex].pix[14],
                                    ii_8x16_int_region[sindex].pix[13],
                                    ii_8x16_int_region[sindex].pix[12],
                                    ii_8x16_int_region[sindex].pix[11],
                                    ii_8x16_int_region[sindex].pix[10],
                                    ii_8x16_int_region[sindex].pix[9],
                                    ii_8x16_int_region[sindex].pix[8],
                                    ii_8x16_int_region[sindex].pix[7],
                                    ii_8x16_int_region[sindex].pix[6],
                                    ii_8x16_int_region[sindex].pix[5],
                                    ii_8x16_int_region[sindex].pix[4],
                                    ii_8x16_int_region[sindex].pix[3],
                                    ii_8x16_int_region[sindex].pix[2],
                                    ii_8x16_int_region[sindex].pix[1],
                                    ii_8x16_int_region[sindex].pix[0]);
                            }
                            IRT_TRACE_TO_LOG(15,
                                             IRTLOG,
                                             "XL:XR %d:%d YT:YB = %d:%d Yi_fixed=%li Xi_fixed=%li\n",
                                             cfifo_data_out[index].XL,
                                             cfifo_data_out[index].XR,
                                             cfifo_data_out[index].YT,
                                             cfifo_data_out[index].YB,
                                             cfifo_data_out[index].Xi_fixed,
                                             cfifo_data_out[index].Yi_fixed);
                            IRT_I2x2_block->run(index,
                                                cfifo_data_out[index].XL,
                                                cfifo_data_out[index].XR,
                                                cfifo_data_out[index].YT,
                                                cfifo_data_out[index].YB,
                                                Xi0,
                                                Yi0,
                                                cfifo_data_out[index].Xi_fixed,
                                                cfifo_data_out[index].Yi_fixed,
                                                ii_8x16_int_region,
                                                ii_8x16_int_region_flags,
                                                irt_top_sig.ip2x2[index],
                                                iirc_task,
                                                irt_top_sig.ofifo_push);
                            irt_top_sig.ip2x2[index].line  = cfifo_data_out[index].line;
                            irt_top_sig.ip2x2[index].pixel = cfifo_data_out[index].pixel;
                            irt_ibli_out.pix[index]        = IRT_IBLI_block->run(
                                irt_top_sig.ip2x2[index], irt_desc[iirc_task].bg, irt_top_sig.ofifo_push, iirc_task);
                            // irt_top_sig.bg_mode_oob_cp_pxl_index++;//   ROTSIM_DV_INTGR -- for OOB coverage
                        }
                        //---------------------------------------
                        // IRT_TRACE_TO_LOG(1,IRTLOG,"BLI OUTPUT :");
                        // for (uint8_t index = 0; index < adj_proc_size; index++) {
                        //   IRT_TRACE_TO_LOG(1,IRTLOG," %x ,", irt_ibli_out.pix[index]);
                        //}
                        // IRT_TRACE_TO_LOG(1,IRTLOG,"\n");
                        //---------------------------------------
                        IRT_CFIFO_block->run(0,
                                             irt_top_sig.ofifo_push,
                                             0,
                                             irt_top_sig.cfifo_pop_size,
                                             cfifo_data_in,
                                             cfifo_data_out,
                                             cfifo_emptyness,
                                             cfifo_fullness);
                        //			}
                        // IRT_TRACE_TO_LOG(1,IRTLOG,"ofifo_push %d \n", irt_top_sig.ofifo_push);
                        // if(irt_top_sig.ofifo_push==1){
                        //    for (uint8_t ii=0;ii<8;ii++) {
                        //       for (uint8_t jj=0;jj<2;jj++) {
                        //          IRT_TRACE_TO_LOG(1,IRTLOG,"rd_en[%d,%d]=%d ::
                        //          ",jj,ii,irt_top_sig.rm_rd_sel[jj][ii]);
                        //       }
                        //       IRT_TRACE_TO_LOG(1,IRTLOG,"\n");
                        //    }
                        // }
                        goto label_OFIFO;

                    default: break;
                }
                break;
            default: break;
        }

#ifdef STANDALONE_ROTATOR
#if 0
      if (irt_top_sig.ofifo_push) {
         IRT_TRACE_TO_LOG(log_file, "[YT,XL,YB,XR]=[%3d,%3d,%3d,%3d]\n", YT, XL, YB, XR);
         //IRT_TRACE_TO_LOG(log_file, "IRT_RMWM status: task %d, first/last lines [%d:%d], top/bot ptr [%d:%d], fullness %d\n", oicc_task, rm_first_line[oicc_task], rm_last_line[oicc_task], rm_top_ptr , irt_top->rm_bot_ptr, irt_top->rm_fullness);
      }
#endif
#endif

        // Use the rotation memory to read the input pixels.
        IRT_RMEM_block->run(0,
                            irt_top_sig.rm_wr_addr,
                            irt_top_sig.rm_wr_sel,
                            irt_top_sig.rm_din,
                            irt_top_sig.rm_meta_addr,
                            irt_top_sig.rm_meta_data,
                            irt_top_sig.rm_rd_addr,
                            irt_top_sig.rm_rd_sel,
                            rm_data_out,
                            irt_top_sig.ofifo_push,
                            oicc_task);
        // Input window calculation. Adjust the returned pixels to the rectangle memory.
        IRT_IIWC_block->run(rm_data_out,
                            rm_bg_flags,
                            rm_rd_mode,
                            ird_shift,
                            imsb_lsb_sel,
                            irt_top_sig.XiL,
                            irt_top_sig.YiT,
                            YT_bank_row,
                            ii_8x16_int_region,
                            ii_8x16_int_region_flags,
                            irt_top_sig.ofifo_push,
                            Yo,
                            Xo0,
                            oicc_task);

        // int XL8 = (irt_top_sig.XL>>3)<<3;
        XiL8 = irt_top_sig.XiL;

        // used for debug - 2x2 sel gets pixels directly from ext memory
#if 0
      XL8 = (XL>>3)<<3;
      for (int i=0; i<8; i++) {
         for (int j=0; j<16; j++) {
            ii_8x16_int_region[i].pix[j] = ext_mem[image_pars[IIMAGE].ADDR+ oicc_task *IMAGE_W*IMAGE_H + (YT+i) * image_pars[IIMAGE].W + XL8+j];
            ii_8x16_int_region_flags[i].pix[j] = ((YT+i)<0 || (YT+i)>=image_pars[IIMAGE].H || (XL8+i)<0 || (XL8+i)>=image_pars[IIMAGE].W) ? 1 : 0;
         }
      }
#endif

        int irt_2x2_sel_max_index;
        if (irt_h5_mode)
            irt_2x2_sel_max_index = irt_desc[oicc_task].proc_size;
        else
            irt_2x2_sel_max_index = adj_proc_size;

        // Fetch for each output pixel the relevant 2x2 input pixels for the interpolation.
        for (uint8_t index = 0; index < irt_2x2_sel_max_index; index++)
            IRT_I2x2_block->run(index,
                                XiL8,
                                XiL8 + 1,
                                irt_top_sig.YiT,
                                irt_top_sig.YiT + 1,
                                Xi0,
                                Yi0,
                                irt_top_sig.Xi_fixed[index],
                                irt_top_sig.Yi_fixed[index],
                                ii_8x16_int_region,
                                ii_8x16_int_region_flags,
                                irt_top_sig.ip2x2[index],
                                oicc_task,
                                irt_top_sig.ofifo_push);

#ifdef STANDALONE_ROTATOR
#if 1
        if (irt_top_sig.ofifo_push) {
            if (print_log_file)
                IRT_TRACE_TO_LOG(LOG_TRACE_L1,
                                 log_file,
                                 "[YT,XL,YB,XR]=[%3d,%3d,%3d,%3d]\n",
                                 irt_top_sig.YiT,
                                 irt_top_sig.XiL,
                                 irt_top_sig.YiB,
                                 irt_top_sig.XiR);
            // IRT_TRACE_TO_LOG(log_file, "IRT_RMWM status: task %d, first/last lines [%d:%d], top/bot ptr [%d:%d],
            // fullness %d\n", oicc_task, rm_first_line[oicc_task], rm_last_line[oicc_task], rm_top_ptr ,
            // irt_top->rm_bot_ptr, irt_top->rm_fullness);
        }
#endif
#endif

        for (uint8_t index = 0; index < irt_2x2_sel_max_index; index++) {
            // Bilinear interpolation
            irt_ibli_out.pix[index] = IRT_IBLI_block->run(
                irt_top_sig.ip2x2[index], irt_desc[oicc_task].bg, irt_top_sig.ofifo_push, oicc_task);
            ibli_bg = (ibli_bg << 1) + (irt_top_sig.ip2x2[index].pix_bg[0] | irt_top_sig.ip2x2[index].pix_bg[1] |
                                        irt_top_sig.ip2x2[index].pix_bg[2] | irt_top_sig.ip2x2[index].pix_bg[3]);
        }
        irt_top_sig.bli_bg_pxl = irt_desc[oicc_task].bg;
        irt_top_sig.psel_rot90 = irt_desc[oicc_task].rot90;

        //	if (ofifo_push) IRT_TRACE_TO_LOG(log_file, "BLI_INPUT: bli_bg %x\n", bli_bg);

        // used for debug - IRT_BLI gets pixels directly from ext memory
#if 0
      for (int i=0; i<8; i++) {

         int xo =  Xo0+i;
         int yo =  Yo;
         float xi_f =  irt_desc[0].cosd[1]*xo + irt_desc[0].sind[1]*yo + image_pars[IIMAGE].Xc;
         float yi_f = -irt_desc[0].sind[1]*xo + irt_desc[0].cosd[1]*yo + image_pars[IIMAGE].Yc;
         int xi = floor (xi_f);
         if (xi<0 || xi>=image_pars[IIMAGE].W-1) {
            if (xi<0) xi = 0;
            else xi = image_pars[IIMAGE].W-2;
            p2x2[i].pix_bg[0]=1;
            p2x2[i].pix_bg[2]=1;
         } else {
            p2x2[i].pix_bg[0]=0;
            p2x2[i].pix_bg[2]=0;
         }

         int yi = floor (yi_f);
         if (yi<0 || yi>=image_pars[IIMAGE].H-1) {
            if (yi<0) yi = 0;
            else yi = image_pars[IIMAGE].H-2;
            p2x2[i].pix_bg[1]=1;
            p2x2[i].pix_bg[3]=1;
         } else {
            p2x2[i].pix_bg[1]=0;
            p2x2[i].pix_bg[3]=0;
         }

         float xf = xi_f - (float) xi;
         float yf = yi_f - (float) yi;

         p2x2[i].pix[0] = ext_mem[image_pars[IIMAGE].ADDR + oicc_task*IMAGE_W*IMAGE_H + yi * image_pars[IIMAGE].W + xi];
         p2x2[i].pix[1] = ext_mem[image_pars[IIMAGE].ADDR + oicc_task*IMAGE_W*IMAGE_H + yi * image_pars[IIMAGE].W + xi+1];
         p2x2[i].pix[2] = ext_mem[image_pars[IIMAGE].ADDR + oicc_task*IMAGE_W*IMAGE_H + (yi+1) * image_pars[IIMAGE].W + xi];
         p2x2[i].pix[3] = ext_mem[image_pars[IIMAGE].ADDR + oicc_task*IMAGE_W*IMAGE_H + yi * image_pars[IIMAGE].W + xi+1];
         p2x2[i].weights[0] = xf;
         p2x2[i].weights[1] = yf;

         irt_bilinear_int_out.pix[i] = IRT_BLI (p2x2[i], 0);
         //irt_bilinear_int_out.pix[i] = ext_mem[image_pars[IIMAGE].ADDR+oicc_task*IMAGE_W*IMAGE_H + yi * image_pars[IIMAGE].W + xi];

      }
#endif

#if 0
      static int pixel = 0, line = 0, task = 0;
      for (int i=0; i<8; i++) {
         int	Xi0 = floor( irt_desc[task].cosd[1]*(pixel+i-image_pars[OIMAGE].Xc) + irt_desc[task].sind[1]*(line-image_pars[OIMAGE].Yc) + image_pars[IIMAGE].Xc);
         int Yi0 = floor(-irt_desc[task].sind[1]*(pixel+i-image_pars[OIMAGE].Xc) + irt_desc[task].cosd[1]*(line-image_pars[OIMAGE].Yc) + image_pars[IIMAGE].Yc);
         irt_bilinear_int_out.pix[i] = ext_mem[image_pars[IIMAGE].ADDR+task*IMAGE_W*IMAGE_H + Yi0 * image_pars[IIMAGE].W + Xi0];
      }
      if (ofifo_push)
         pixel += 8;
      pixel_valid = 0xff;
      if (task < 3) {
         if (pixel >= image_pars[OIMAGE].W) {
            IRT_TRACE("OFIFO write: task %d, line %d H %d\n", task, line, image_pars[OIMAGE].H);
            pixel = 0;
            line++;
            if (line == image_pars[OIMAGE].H) {
               IRT_TRACE("OFIFO write: task %d finished\n", task);
               line = 0;
               task++;
            }
         }
      }
#endif

    label_OFIFO:
        // Write the 8 output pixels to the output FIFO. Collect 16 groups of 8 to the cache line.
        IRT_OFIFO_block->run(
            irt_top_sig.ofifo_push, ofifo_pop, pixel_valid, irt_ibli_out, ofifo_out, ofifo_empty, ofifo_full, task_end);

        // Output write manger. Writes the 128 output pixels to the memory.
        irt_done[e_irt_owm] =
            IRT_OIWM_block->run(ofifo_empty, ofifo_pop, ofifo_out, oimage_addr, oimage_wr_data, mem_write);

#if 0
      if (irt_top_sig.mofifo_push) {
         printf("[%d, %d] mp2x2x64: ", Ym, Xm0);
         for (int index = 0; index < 8; index++) {
            printf("[%llx %llx %llx %llx]", (uint64_t)irt_top_sig.mp2x2x64[index].pix[0], (uint64_t)irt_top_sig.mp2x2x64[index].pix[1], (uint64_t)irt_top_sig.mp2x2x64[index].pix[2], (uint64_t)irt_top_sig.mp2x2x64[index].pix[3]);
         }
         printf("\n");
      }
#endif

#if 0
      if (irt_top_sig.mofifo_push) {
         printf("[%d, %d]: mp2x2_fp32_x", Ym, Xm0);
         for (int index = 0; index < 8; index++) {
            printf("[%f %f %f %f]", irt_top_sig.mp2x2_fp32_x[index].pix[0], irt_top_sig.mp2x2_fp32_x[index].pix[1], irt_top_sig.mp2x2_fp32_x[index].pix[2], irt_top_sig.mp2x2_fp32_x[index].pix[3]);
         }
         printf("\n");6407
         printf("[%d, %d]: mp2x2_fp32_y", Ym, Xm0);
         for (int index = 0; index < 8; index++) {
            printf("[%f %f %f %f]", irt_top_sig.mp2x2_fp32_y[index].pix[0], irt_top_sig.mp2x2_fp32_y[index].pix[1], irt_top_sig.mp2x2_fp32_y[index].pix[2], irt_top_sig.mp2x2_fp32_y[index].pix[3]);
         }
         printf("\n");
      }
#endif

        /******************************************/

        // Done all descriptors
        // printf("Done status: %d %d %d %d %d\n",irt_done[e_irt_irm] ,irt_done[e_irt_rmwm] , irt_done[e_irt_oicc]
        // ,irt_done[e_irt_iiirc], irt_done[e_irt_owm]); bool irt_mesh_done = irt_done[e_irt_mrm] & irt_done[e_irt_mmwm]
        // & irt_done[e_irt_micc];

        return irt_done[e_irt_irm] & irt_done[e_irt_rmwm] & irt_done[e_irt_oicc] & irt_done[e_irt_iiirc] &
               irt_done[e_irt_owm];

    } else { // RESAMPLER TASKS
        bool done = IRT_RESAMP_block->run(iimage_addr,
                                          imem_rd,
                                          ilsb_pad_rd,
                                          imsb_pad_rd,
                                          iimage_meta_out,
                                          iimage_rd_data,
                                          iimage_meta_in,
                                          imem_rd_data_valid,
                                          mimage_addr,
                                          mmem_rd,
                                          mlsb_pad_rd,
                                          mmsb_pad_rd,
                                          mimage_meta_out,
                                          mimage_rd_data,
                                          mimage_meta_in,
                                          mmem_rd_data_valid,
                                          gimage_addr,
                                          gmem_rd,
                                          glsb_pad_rd,
                                          gmsb_pad_rd,
                                          gimage_meta_out,
                                          gimage_rd_data,
                                          gimage_meta_in,
                                          gmem_rd_data_valid,
                                          oimage_addr,
                                          mem_write,
                                          oimage_wr_data);
        return done;
    }
}
} // namespace irt_utils_gaudi3
#pragma GCC diagnostic pop
