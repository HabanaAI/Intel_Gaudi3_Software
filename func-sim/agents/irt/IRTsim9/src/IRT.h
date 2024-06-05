/*
 * IRT.h
 *
 *  Created on: Jun 4, 2019
 *      Author: espektor
 */

// non static function         requires class object creation (    new()) and used with -> and can    access members of
// classes.
//    static function doesn't requires class object creation (w/o new()) and used with :: and cannot access members of
//    classes
//	  static function cannot call to non-static function (because static does not require object creation, while
// non-static require it public function can be used in any class private function can be used only it its class

#ifndef IRT_H_
#define IRT_H_

#include <queue>
#include <string>
#include <vector>
#include "fs_fma_gaudi3.h"
//---------------------------------------
// Removed after adding fs_fma based conversion function
#ifdef USE_OLD_FP_CONV
#include "floatx.hpp"
#include "half/half.hpp"
using flx::detail::bf16_to_float;
using flx::detail::float_to_bf16;
using half_float::half;
using half_float::detail::float2half;
using half_float::detail::half2float;
using bf16 = flx::floatx<8, 7>;
#endif
////---------------------------------------
using std::queue;
using std::vector;
using namespace std;
namespace irt_utils_gaudi3
{
#define _CRT_SECURE_NO_WARNINGS 1
#define DEFAULT_INF_FP32 0x7F800000
extern uint16_t h9taskid;

/*******************/
/* Defines section */
/*******************/
#define IRT_VERSION 91
#define IRT_DATE "2020-11-16"

#if !defined(RUN_WITH_SV) && !defined(HABANA_SIMULATION)
//#define CREATE_IMAGE_DUMPS
//#define CREATE_IMAGE_DUMPS_IRM
//#define CREATE_IMAGE_DUMPS_RMWM
//#define CREATE_IMAGE_DUMPS_RM
//#define CREATE_IMAGE_DUMPS_IWC
//#define CREATE_IMAGE_DUMPS_OWM
//#define STANDALONE_ROTATOR
#ifndef STANDALONE_ROTATOR
    #define STANDALONE_ROTATOR
#endif
//#define IRT_USE_FLIP_FOR_MINUS1
#else
#if defined(HABANA_SIMULATION)
//#define IRT_USE_FLIP_FOR_MINUS1
//#define IRT_USE_INTRP_REQ_CHECK
//#define CREATE_IMAGE_DUMPS
//#define CREATE_IMAGE_DUMPS_OWM
#endif
#endif
// Enabled for H9 as fract0 optimization is supported
//#define IRT_USE_INTRP_REQ_CHECK - TODO - affine tests failed when this is enabled. Need to confirm

extern int max_log_trace_level;
extern int cseed;
#define LOG_TRACE_L1 1
//#define IRT_PROJ_COEFS_NORMALIZATION

#define IRT_BILINEAR_INT_WEIGHTS_PREC 12
#define IRT_OFIFO_DEPTH 8
#define IRT_OFIFO_WIDTH 128
#define IRT_IFIFO_DEPTH 8
#define IRT_IFIFO_WIDTH 128

//#define IRT_ROT_MEM_HEIGHT1 128
#define IRT_META_MEM_BANK_HEIGHT 512 // 1024 lines over 8 banks  sjagadale  -- H9-RESAMPLER MEMORY INCREASE CHANGE
#define IRT_ROT_MEM_ROW_BANKS 8
#define IRT_ROT_MEM_COL_BANKS 2
#define IRT_ROT_MEM_BANK_WIDTH 16
#define IRT_ROT_MEM_ROW_WIDTH (2 * IRT_ROT_MEM_BANK_WIDTH)
#define IRT_ROT_MEM_PXL_SHIFT 5 // LOG2(IRT_ROT_MEM_ROW_WIDTH)
#define IRT_ROT_MEM_BANK_HEIGHT 512 // sjagadale  -- H9-RESAMPLER MEMORY INCREASE CHANGE
#define IRT_IFIFO_OUT_WIDTH (2 * IRT_ROT_MEM_BANK_WIDTH)
#define IRT_IFIFO_R_ENTRIES (128 / IRT_IFIFO_OUT_WIDTH)

#define IRT_MESH_MEM_ROW_BANKS 2
#define IRT_MESH_MEM_COL_BANKS 2
#define IRT_MESH_MEM_BANK_WIDTH 64
#define IRT_MESH_MEM_ROW_WIDTH (2 * IRT_MESH_MEM_BANK_WIDTH)
#define IRT_MESH_MEM_PXL_SHIFT 7 // LOG2(IRT_MESH_MEM_BANK_WIDTH)
#define IRT_MESH_MEM_BANK_HEIGHT 256 // 192      //sjagadale  -- H9-RESAMPLER MEMORY INCREASE CHANGE
#define IRT_MESH_G_PREC 31
#define IRT_MFIFO_OUT_WIDTH (2 * IRT_MESH_MEM_BANK_WIDTH) // 8 pixels * 2 components * 4 bytes (FP32)
#define IRT_MFIFO_R_ENTRIES (128 / IRT_MFIFO_OUT_WIDTH)

#define IRT_RM_CLIENTS 2 // Rsb / Mrsb / Grsb
#define IRT_RM_MODES 8
#define IRT_RM_MAX_STAT_MODE 7
#define IRT_RM_MAX_DYN_MODE 7
#define IRT_MM_MAX_STAT_MODE 7
#define IRT_MM_MAX_DYN_MODE 7
#define IRT_RM_DYN_MODE_LINE_RELEASE IRT_ROT_MEM_ROW_BANKS
#define IRT_MM_DYN_MODE_LINE_RELEASE IRT_MESH_MEM_ROW_BANKS

#define IRT_CFIFO_SIZE 16
#define FP32_SIGN_LOC 31
#define FP32_EXP_LOC 23
#define FP32_EXP_MSK 0xff
#define FP32_MANTISSA_MSK 0x7fffff

#define IIMAGE_W (8 * 1024)
#define IIMAGE_H (8 * 1024)
#define OIMAGE_W (8 * 1024)
#define OIMAGE_H (8 * 1024)
#define PLANES 3
#define BYTEs4PIXEL 2 // bytes per pixel
#define BYTEs4MESH 8
#define CALC_FORMATS_ROT 5

#define IIMAGE_SIZE IIMAGE_W* IIMAGE_H* PLANES* BYTEs4PIXEL
#define OIMAGE_SIZE OIMAGE_W* OIMAGE_H* PLANES* BYTEs4PIXEL
#define MIMAGE_SIZE OIMAGE_W* OIMAGE_H* BYTEs4MESH
#define GIMAGE_SIZE OIMAGE_W* OIMAGE_H* BYTEs4PIXEL

#define IMEM_NUM_IMAGES 1

#define OUT_NUM_IMAGES 1 // 10 //store up to 10 processed output images
#define OMEM_NUM_IMAGES                                                                                                \
    (CALC_FORMATS_ROT * OUT_NUM_IMAGES) // for all calculation formats of hl model x OUT_NUM_IMAGES images
#define EMEM_SIZE IIMAGE_SIZE + OIMAGE_SIZE* OUT_NUM_IMAGES + MIMAGE_SIZE* OUT_NUM_IMAGES // input, output, mesh

// used for multitask support
#define OUT_NUM_IMAGES_MT 1 // store up to 10 processed output images
#define MAX_TASKS_MT (PLANES * OUT_NUM_IMAGES_MT) + 1
#define OMEM_NUM_IMAGES_MT                                                                                             \
    (CALC_FORMATS_ROT * OUT_NUM_IMAGES_MT) // for all calculation formats of hl model x OUT_NUM_IMAGES images
#define EMEM_SIZE_MT                                                                                                   \
    IIMAGE_SIZE + OIMAGE_SIZE* OUT_NUM_IMAGES_MT + MIMAGE_SIZE* OUT_NUM_IMAGES_MT + GIMAGE_SIZE // input, output, mesh

#define IIMAGE_REGION_START 0
#define OIMAGE_REGION_START IIMAGE_REGION_START + IIMAGE_SIZE
#define MIMAGE_REGION_START OIMAGE_REGION_START + OIMAGE_SIZE* OUT_NUM_IMAGES
#define MIMAGE_REGION_START_MT OIMAGE_REGION_START + OIMAGE_SIZE* OUT_NUM_IMAGES_MT
#define GIMAGE_REGION_START MIMAGE_REGION_START + GIMAGE_SIZE* OUT_NUM_IMAGES
#define GIMAGE_REGION_START_MT MIMAGE_REGION_START + GIMAGE_SIZE* OUT_NUM_IMAGES_MT

#define IIMAGE 0
#define OIMAGE 1
#define MIMAGE 2
#define GIMAGE 3

#define IRT_PIXEL_WIDTH 0 // 12 //8
#define IRT_COORD_WIDTH 16
#define IRT_XiYi_first_max (1 << 14) - 1
#define IRT_Xi_start_max (1 << 15) - 1
#define IRT_WEIGHT_PREC_EXT                                                                                            \
    1 + 2 + 1 // +1 because of 0.5 error, +2 because of interpolation, +1 because of weigths multiplication
#define IRT_COORD_PREC_EXT 1
#define IRT_SLOPE_PREC 17
#define IRT_SLOPE_ROUND (1 << (IRT_SLOPE_PREC - 1))

#define MAX_ROT_ANGLE 89 // 65//58.99
#define MAX_SI_WIDTH 4096
#define MIN_SI_WIDTH 32
#define MAX_SI_HEIGHT 1024
#define MIN_SI_HEIGHT 32 // 8  sjagadale  -- H9-RESAMPLER MEMORY INCREASE CHANGE
#define MAX_SM_WIDTH (4096 * 4)
#define MIN_SM_WIDTH 128
#define RESCALE_MIN_SM_WIDTH 64
#define IRT_ROT_DIR_POS 1
#define IRT_ROT_DIR_NEG 0

#define IRT_INT_WIN_W 9
#define IRT_INT_WIN_H 8

#define IRT_ROT_MAX_PROC_SIZE 8
#define IRT_MESH_MAX_PROC_SIZE 8
#define IRT_MESH_FP32_BYTES 4

#if defined(RUN_WITH_SV)
#define MAX_TASKS 200 // ROTSIM_DV_INTGR
#else
#define MAX_TASKS (PLANES * OUT_NUM_IMAGES + 1)
#endif
//---------------------------------------
#define CL_SIZE 64
#define RESAMP_STRIPE_SIZE 128
#define RESAMP_SEGMENT_SIZE 128
#define BATCH_SIZE 1
#define IRT_MAX_IMAGE_W 7680
#define IRT_MAX_IMAGE_H 4320
#define CSIZE 16
#define NUM_BANK_ROW 8
#define NSETS 64
#define NWAYS 16
#define TOTALSETS (NSETS * NUM_BANK_ROW)
#define NSETS_PER_BANK (NSETS / NUM_BANK_ROW)
#define NSETS_C 8
#define HSIZE 134560
#define MAX_SEG 128
//#define RANGE_BOUND 80
#define RANGE_BOUND 30
#define STRIPE_SIZE 128
#define ROUNDTRIP 700
//#define MAX_PROC_SIZE 8
#define MAX_STRIDE 8
#define P_LATENCY 5
#define TC_EN 1
#define SC_EN 1
#define TC_1_EN 1
#define TC_1_PIPE 2
#define TC_1_PIPE2 6
#define REQ_Q_DEPTH 2
#define EXE 2
#define WARP_ELM_SIZE 8
#define IN_ELM_SIZE 2
#define OUT_ELM_SIZE 2
#define NCOORD 2
#define MAX_COMPUTE_STALL 1200
#define MISSQ_MAX_DEPTH 8
#define LANCZOS_MAX_PHASES 128
#define LANCZOS_MAX_PREC 8
#define RESCALE_MAX_TAPS 16
#define RESCALE_MAX_GX_PREC 24
#define NUM_OCACHE_RESCALE 2
#define BYTES_PER_CYCLE 48 // 95%        46   //with 90% Util.
#define FREQ 1.05 // GHz. TODO
#define BW 51.2 // GBPS
//---------------------------------------
#define VERIF_IMAGE_OFFSET (OIMAGE_H * OIMAGE_W * 4)
#define VERIF_OIMAGE_OFFSET (0)
#define VERIF_MIMAGE_OFFSET (VERIF_OIMAGE_OFFSET + VERIF_IMAGE_OFFSET)
#define VERIF_IIMAGE_OFFSET (VERIF_MIMAGE_OFFSET + VERIF_IMAGE_OFFSET)
#define VERIF_GIMAGE_OFFSET (VERIF_IIMAGE_OFFSET + VERIF_IMAGE_OFFSET)
//---------------------------------------
#define ROT_FP32_EXP_MASK 0x7F800000
#define ROT_FP16_EXP_MASK 0x7C00
#define ROT_BFP16_EXP_MASK 0x7F80
//---------------------------------------
/*******************/
/* Strings section */
/*******************/
extern const char* irt_int_mode_s[2];
extern const char* irt_irt_mode_s[9];
extern const char* irt_affn_mode_s[4];
extern const char* irt_proj_mode_s[4];
extern const char* irt_refl_mode_s[4];
extern const char* irt_shr_mode_s[4];
extern const char* irt_proj_order_s[6];
extern const char* irt_mesh_mode_s[4];
extern const char* irt_mesh_order_s[2];
extern const char* irt_mesh_rel_mode_s[2];
extern const char* irt_mesh_format_s[5];
extern const char* irt_mesh_fp_loc_s[16];
extern const char* irt_flow_mode_s[4];
extern const char* irt_rate_mode_s[4];
extern const char* irt_pixel_width_s[5];
extern const char* irt_bg_mode_s[3];
extern const char* irt_crd_mode_s[2];
extern const char* irt_hl_format_s[5];
extern const char* irt_mem_type_s[2];
extern const char* irt_prj_matrix_s[7];
extern const char* irt_flip_mode_s[4];
extern const char* irt_buf_format_s[2];
extern const char* irt_resamp_dtype_s[5];

/*******************/
/* Enums section */
/*******************/
#define TASK_BLOCKS 13
typedef enum
{
    e_irt_irm    = 0,
    e_irt_rmwm   = 1,
    e_irt_oicc   = 2,
    e_irt_owm    = 3,
    e_irt_mrm    = 4,
    e_irt_mmwm   = 5,
    e_irt_micc   = 6,
    e_irt_oicc2  = 7,
    e_irt_iiirc  = 8,
    e_irt_micc2  = 9,
    e_irt_resamp = 10,
    e_irt_wrm    = 11,
    e_irt_grm    = 12,
} Eirt_blocks_enum;

typedef enum
{
    e_irt_angle_roll  = 0,
    e_irt_angle_pitch = 1,
    e_irt_angle_yaw   = 2,
    e_irt_angle_shear = 3,
    e_irt_angle_rot   = 4,
    e_irt_angle_shr_x = 5,
    e_irt_angle_shr_y = 6,
} Eirt_angle_enum;

typedef enum
{
    e_irt_rotation     = 0,
    e_irt_affine       = 1,
    e_irt_projection   = 2,
    e_irt_mesh         = 3,
    e_irt_resamp_fwd   = 4,
    e_irt_resamp_bwd1  = 5,
    e_irt_resamp_bwd2  = 6,
    e_irt_rescale      = 7,
    e_irt_BiLinearGrad = 8,
} Eirt_tranform_type;

typedef enum
{
    e_irt_rate_fixed = 0, // fixed according to proc_size
    e_irt_rate_adaptive_2x2 =
        1, // self-adjusted according to rotation memory reading contention using multiple 2x2 reads, up to proc_size
    e_irt_rate_adaptive_wxh = 2, // self-adjusted according to single interpolation window size, up to proc_size
} Eirt_rate_type;

typedef enum
{
    e_irt_mesh_reserv0 = 0,
    e_irt_mesh_flex    = 1,
    e_irt_mesh_reserv2 = 2,
    e_irt_mesh_reserv3 = 3,
    e_irt_mesh_fp32    = 4,
} Eirt_mesh_format_type;

typedef enum
{
    e_irt_mesh_absolute = 0,
    e_irt_mesh_relative = 1,
} Eirt_mesh_rel_mode_type;

typedef enum
{
    e_irt_bg_prog_value   = 0,
    e_irt_bg_frame_repeat = 1,
    e_irt_bg_in_pad       = 2,
} Eirt_bg_mode_type;

typedef enum
{
    e_irt_int_bilinear = 0,
    e_irt_int_nearest  = 1,
} Eirt_int_mode_type;

typedef enum
{
    e_irt_crd_mode_fixed = 0,
    e_irt_crd_mode_fp32  = 1,
} Eirt_coord_mode_type;

typedef enum
{
    e_irt_debug_bypass = 0,
    e_irt_debug_bg_out = 1,
} Eirt_debug_type;

typedef enum
{
    e_irt_block_rot  = 0,
    e_irt_block_mesh = 1,
} Eirt_block_type;

typedef enum
{
    e_irt_flow_wCFIFO_fixed_adaptive_2x2 = 0,
    e_irt_flow_wCFIFO_fixed_adaptive_wxh = 1,
    e_irt_flow_nCFIFO_fixed_adaptive_wxh = 2,
} Eirt_flow_type;

typedef enum
{
    e_irt_buf_format_static  = 0,
    e_irt_buf_format_dynamic = 1,
} Eirt_buf_format;

typedef enum
{
    e_irt_buf_select_man  = 0,
    e_irt_buf_select_auto = 1,
} Eirt_buf_select;

typedef enum
{
    e_irt_bmp_H20 = 0,
    e_irt_bmp_02H = 1,
} Eirt_bmp_order;

typedef enum
{
    e_irt_aff_rotation   = 0, // R
    e_irt_aff_scaling    = 1, // S
    e_irt_aff_reflection = 2, // M
    e_irt_aff_shearing   = 3, // T
} Eirt_aff_ops;

typedef enum
{
    e_irt_xi_start_calc_caller_desc_gen      = 0,
    e_irt_xi_start_calc_caller_irm           = 1,
    e_irt_xi_start_calc_caller_oicc          = 2,
    e_irt_xi_start_calc_caller_rmrm          = 3,
    e_irt_xi_start_calc_caller_rmrm_hl_model = 3,
    e_irt_xi_start_calc_caller_rmrm_oicc     = 4,
    e_irt_xi_start_calc_caller_rmrm_top      = 5,
    e_irt_xi_start_calc_caller_rmrm_mmrm     = 6,
} Eirt_xi_start_calc_caller;

typedef enum
{
    e_irt_rmrm_caller_hl_model = 0,
    e_irt_rmrm_caller_irt_oicc = 1,
    e_irt_rmrm_caller_irt_top  = 2,
    e_irt_rmrm_caller_irt_mmrm = 3,
} Eirt_irt_rmrm_caller;

#define RESAMP_TASK_BLOCKS 7
typedef enum
{
    e_resamp_wread   = 0,
    e_resamp_parser  = 1,
    e_resamp_fill    = 2,
    e_resamp_victimQ = 3,
    e_resamp_compute = 4,
    e_resamp_owrite  = 5,
    e_resamp_gread   = 6,
} Eirt_resamp_blocks_enum;

typedef enum
{
    e_irt_int8  = 0,
    e_irt_int16 = 1,
    e_irt_fp16  = 2,
    e_irt_bfp16 = 3,
    e_irt_fp32  = 4,
} Eirt_resamp_dtype_enum;
/*******************/
// rescale
/*******************/
typedef enum
{
    e_irt_lanczos2 = 0,
    e_irt_lanczos3 = 1,
    e_irt_bicubic  = 2,
} filter_type_t;

typedef struct Coeff_LUT
{
    float   Gf; // inverse scale factor
    int     num_phases;
    uint8_t num_taps;
    float*  Coeff_table;
} Coeff_LUT_t;

struct num_err_status_s
{
    uint32_t rinterp_pinf_err;
    uint32_t rinterp_ninf_err;
    uint32_t rinterp_nan_err;
    uint32_t minterp_pinf_err;
    uint32_t minterp_ninf_err;
    uint32_t minterp_nan_err;
    uint32_t coord_pinf_err;
    uint32_t coord_ninf_err;
    uint32_t coord_nan_err;
};
/*******************/
/* Struct section */
/*******************/
// mesh structure
typedef struct _mesh_xy_fp32
{
    float x, y;
} mesh_xy_fp32;

typedef struct _mesh_xy_fp32_meta
{
    float    x, y;
    uint32_t Si, IBufH_req, MBufH_req, IBufW_req;
    float    Xi_first, Yi_first, Xi_last, Yi_last, Ymin;
    bool     Ymin_dir, Ymin_dir_swap, XiYi_inf;
} mesh_xy_fp32_meta;

typedef struct _mesh_xy_fp64_meta
{
    double   x, y;
    uint32_t Si, IBufH_req, MBufH_req, IBufW_req;
    double   Xi_first, Yi_first, Xi_last, Yi_last, Ymin;
    bool     Ymin_dir, Ymin_dir_swap, XiYi_inf;
} mesh_xy_fp64_meta;

typedef struct _mesh_xy_fi16
{
    int16_t x, y;
} mesh_xy_fi16;

// Internal memory params. Constant values.
struct irt_rm_cfg
{
    uint16_t BufW; // buffer width in byte
    uint16_t Buf_EpL; // entries per input line
    uint16_t BufH;
    uint16_t BufH_mod;
};

struct irt_mesh_images
{
    mesh_xy_fp32_meta** mesh_image_full; // full mesh matrix
    mesh_xy_fp32**      mesh_image_rel; // relative mesh matrix
    mesh_xy_fp32**      mesh_image_fp32; // sparce mesh matrix in fp32 format
    mesh_xy_fi16**      mesh_image_fi16; // sparce mesh_matrix in fi16 format
    mesh_xy_fp64_meta** mesh_image_intr; // interpolated sparce matrix
    mesh_xy_fp64_meta** proj_image_full; // projection mesh matrix
};

struct irt_cfg_pars
{
    // configurated as high level parameters:
    bool buf_1b_mode[IRT_RM_CLIENTS]; // memory buffer 1byte storage format: 0 - 0...31, 1 - {16,0}, ...{31, 15}; [0] -
                                      // rot_mem, [1] - mesh_mem
    Eirt_buf_format buf_format[IRT_RM_CLIENTS]; //[0] - rot_mem, [1] - mesh_mem
    Eirt_buf_select buf_select[IRT_RM_CLIENTS]; // memory buffer mode select: 0 - manual by parameter, 1 - auto by
                                                // descriptor gen;  [0] - rot_mem, [1] - mesh_mem
    uint8_t        buf_mode[IRT_RM_CLIENTS]; // memory buffer manual mode; [0] - rot_mem, [1] - mesh_mem
    Eirt_flow_type flow_mode; // 0 - new with fixed or full adaptive rate ctrl, 1 - new with fixed or rectangle adaptive
                              // rate ctrl, 2 - old with fixed with rectangle adaptive rate ctrl (default 0)

    // below are autogenerated and set at reset
    uint16_t        Hb[IRT_RM_CLIENTS], Hb_mod[IRT_RM_CLIENTS];
    irt_rm_cfg      rm_cfg[IRT_RM_CLIENTS][IRT_RM_MODES];
    Eirt_debug_type debug_mode;

    irt_mesh_images mesh_images;
    uint16_t        lanczos_max_phases_h;
    uint16_t        lanczos_max_phases_v;
    uint8_t         bwd2_err_margin;
    uint16_t        ovrld_rescale_coeff;
    bool            plru_mode = 0; // flag to enable TPC like PLRU
};

struct image_par
{
    // main parameters:
    uint64_t               addr_start; // 1st byte of the image
    uint64_t               addr_end; // last byte of the image
    uint16_t               W; // image width
    uint16_t               H; // image height
    int16_t                Xc; // image horizontal rotation center .5 precision
    int16_t                Yc; // image vertical rotation center .5 precision
    uint16_t               S; // image stripe size to rotate
    uint32_t               Hs; // image stride size
    Eirt_resamp_dtype_enum DataType;
    // calculated parameters:
    uint16_t Ps; // pixel size: 0 - 1 byte, 1 - 2 byte, 2 - 4 bytes, 3 - 8 bytes
    uint16_t PsBytes; // pixel size: 0 - 1 byte, 1 - 2 byte, 2 - 4 bytes, 3 - 8 bytes
    uint32_t Size; // Width * Height
    uint16_t last_stripe_width;
    uint16_t num_stripes;
    // memory parameters, set by descriptor generator from high level parameters
    uint8_t buf_mode;
};

struct irt_desc_par
{
    // main parameters, provided as high level parameters:
    bool                 oimage_line_wr_format; // output image write mode
    uint16_t             bg; // output image background value
    uint32_t             mrsb_bg;
    Eirt_int_mode_type   int_mode; // 0 - bilinear, 1 - nearest neighbor
    Eirt_bg_mode_type    bg_mode; // 0 - programmable background, 1 - frame boundary repeatition
    Eirt_tranform_type   irt_mode; // 0 - rotation, 1 - affine transform, 2 - projection, 3 - mesh
    Eirt_rate_type       rate_mode; // 0 - fixed, equal to processing size, 1 - adaptive
    Eirt_coord_mode_type crd_mode; // 0 - fixed point, 1 - fp32
    uint8_t              proc_size;

    // see image_par struct above for details
    struct image_par image_par[4]; //[0] - input image, [1] - output image, [2] - mesh /warp [3] grad image
    uint16_t         Ho, Wo;

    // pixel width parameters - derived from high level parameters
    uint16_t Msi; // input pixel mask
    int16_t  bli_shift; // bi-linear interpolation shift
    uint16_t MAX_VALo; // output pixel max value, equal to 2^pixel_width - 1
    uint16_t Ppo; // output pixel padding

    // rotation calculated parameters - derived from high level parameters
    bool rot_dir;
    bool read_hflip, read_vflip;
    bool rot90;
    bool rot90_intv; // rotation 90 interpolation over input image lines is required. Reduce Wo to allow IBufH fit rot
                     // memory and reduce proc_size to 7 to fit 8 lines
    bool    rot90_inth; // rotation 90 interpolation over input image pixels is required. Increase Si.
    int32_t im_read_slope;
    int32_t cosi, sini;
    float   cosf, sinf;
    int16_t Yi_start; // index of first row
    int16_t Yi_end; // index of last row
    int64_t Xi_first_fixed, Yi_first_fixed, Xi_last_fixed;
    int16_t Xi_start_offset, Xi_start_offset_flip;
    uint8_t prec_align;

    // affine calculated parameters - derived from high level parameters
    int32_t M11i, M12i, M21i, M22i; // affine matrix parameters
    float   M11f, M12f, M21f, M22f; // affine matrix parameters

    // projection calculated parameters - derived from high level parameters
    float prj_Af[3], prj_Bf[3], prj_Cf[3], prj_Df[3];

    // mesh parameters, provided as high level parameters:
    Eirt_mesh_format_type   mesh_format; // 1 - flex point, 4 - fp32
    uint8_t                 mesh_point_location;
    bool                    mesh_sparse_h, mesh_sparse_v; // 0 - not sparse, 1 - sparse
    uint32_t                mesh_Gh, mesh_Gv; // mesh granularity U.20
    Eirt_mesh_rel_mode_type mesh_rel_mode; // 0 - abs, 1 - relative
    bool                    resize_bli_grad_en;

    //---------------------------------------
    // BWD2
    uint8_t warp_stride;
    //---------------------------------------
    // rescale descriptors
    Coeff_LUT_t* rescale_LUT_x;
    Coeff_LUT_t* rescale_LUT_y;
    uint8_t      rescale_prec; // TODO - SHOULDNT BE IN DESCR FILD..resanem as lancsoz_prec
    uint16_t     mesh_stripe_stride; // ignore
    uint8_t      rescale_phases_prec_H;
    uint8_t      rescale_phases_prec_V;
    //---------------------------------------
    bool clip_fp;
    bool clip_fp_inf_input;
    bool ftz_en;
};

struct rotation_par
{
    // auto calculated at reset:
    uint8_t MAX_PIXEL_WIDTH, MAX_COORD_WIDTH;
    // precisions parameters
    uint8_t  COORD_PREC, WEIGHT_PREC, TOTAL_PREC, WEIGHT_SHIFT;
    uint8_t  PROJ_NOM_PREC, PROJ_DEN_PREC;
    uint32_t COORD_ROUND, TOTAL_ROUND, PROJ_NOM_ROUND, PROJ_DEN_ROUND;
    bool     rot_prec_auto_adj, mesh_prec_auto_adj, oimg_auto_adj, use_rectangular_input_stripe, use_Si_delta_margin;
    float    oimg_auto_adj_rate;

    // temporal variables for internal usage
    uint32_t IBufW_req, IBufH_req, IBufW_entries_req;
    uint32_t MBufW_req, MBufH_req, MBufW_entries_req;

    // pixel width configuration - provided as high level parameters :
    uint8_t Pwi; // input image pixel width
    uint8_t Ppi; // image pixel alignment to specify where valid pixel bits are located
    uint8_t Pwo; // output image pixel width
    // derived from pixel width configuration high level parameters
    uint8_t bli_shift_fix;

    // rotation parameters - provided as high level parameters
    double irt_angles[e_irt_angle_shr_y + 1];
    // double rot_angle;
    // rotation parameters - derived from rot_angle and image size
    double                     irt_angles_adj[e_irt_angle_shr_y + 1];
    double /*rot_angle_adj*,*/ im_read_slope;
    double                     cosd, sind;
    int                        cos16, sin16;
    // used for HL model w/o flip (with rot_angle and not rot_angle_adj)
    double  cosd_hl, sind_hl;
    float   cosf_hl, sinf_hl;
    int32_t cosi_hl, sini_hl;
    double  Xi_first, Xi_last, Yi_first, Yi_last;
    double  Xo_first, Xo_last, Yo_first, Yo_last;
    // int16_t Yi_start;
    int16_t  Xi_start;
    uint16_t So8;
    double   Si_delta;
    uint32_t IBufH_delta;
    // bool rot90_intv; //rotation 90 interpolation over input image lines is required. Reduce Wo to allow IBufH fit rot
    // memory and reduce proc_size to 7 to fit 8 lines bool rot90_inth; //rotation 90 interpolation over input image
    // pixels is required. Increase Si.

    // affine parameters - provided as high level parameters
    char    affine_mode[5];
    uint8_t reflection_mode, shear_mode; // shear modes
    // double shr_x_angle, shr_y_angle; //shear angles
    double Sx, Sy; // scaling factors
    // affine parameters - derived from affine high level parameters
    uint8_t affine_flags[4];
    double  M11d, M12d, M21d, M22d;
    double  affine_Si_factor;

    // projection parameters - provided as high level parameters
    uint8_t proj_mode; // projection mode (0 - rotation, 1 - affine, 2 - projection)
    char    proj_order[5]; // projection order (any composition and order of YRPS (Y-yaw, R-roll, P-pitch, S-shearing)
    double /*proj_angle[3],*/ proj_Zd,
        proj_Wd; // proj_Zi; projection angles (roll, pitch, yaw), projection focal distance, projection plane distance
    // projection parameters - derived from projection high level parameters
    double  proj_R_orig[3][3], proj_R[3][3], proj_T[3];
    double  prj_Ad[3], prj_Bd[3], prj_Cd[3], prj_Dd[3];
    int64_t prj_Ai[3], prj_Bi[3], prj_Ci[3], prj_Di[3];

    // mesh distortion parameters - provided as high level parameters
    double dist_x, dist_y, dist_r;
    // mesh parameters - provided as high level parameters
    uint8_t mesh_mode; //(0 - rotation, 1 - affine, 2 - projection, 3 - distortion)
    bool    mesh_order; //(0 - pre-distortio, 1 - post-distortion)
    double  mesh_Sh, mesh_Sv; // mesh image horizontal/vertical scaling

    bool mesh_dist_r0_Sh1_Sv1; // detection of mesh mode with distortion 0 and no sparse matrix
    bool mesh_matrix_error; // detection of mesh mode with distortion 0 and no sparse matrix and non flex mode

    // processing statistics - internal usage
    uint8_t  min_proc_rate, max_proc_rate;
    bool     proc_auto;
    uint32_t rate_hist[IRT_ROT_MAX_PROC_SIZE], acc_proc_rate, cycles;

    // memory parameters
    Eirt_buf_format buf_format[IRT_RM_CLIENTS]; //[0] - rot_mem, [1] - mesh_mem
    Eirt_buf_select buf_select[IRT_RM_CLIENTS]; // memory buffer mode select: 0 - manual by parameter, 1 - auto by
                                                // descriptor gen;  [0] - rot_mem, [1] - mesh_mem
    uint8_t buf_mode[IRT_RM_CLIENTS]; // memory buffer manual mode; [0] - rot_mem, [1] - mesh_mem
    //---------------------------------------
    // RESAMP parameters
    filter_type_t filter_type;
    uint8_t       rescale_Gx_prec;
    // float nudge;
    //   Eirt_resamp_dtype_enum resamp_dtype;
};

// buses structures
struct p2x2_ui16_struct
{
    uint16_t pix[4];
    bool     pix_bg[4];
    uint32_t weights_fixed[2];
    double   weights[2];
    uint16_t line, pixel;
};

struct p2x2_ui64_struct
{
    uint64_t pix[4];
    bool     pix_bg[4];
    uint32_t weights_fixed[2];
    double   weights[2];
};

struct p2x2_ui32_struct
{
    uint32_t pix[4];
    bool     pix_bg[4];
    uint32_t weights_fixed[2];
#ifdef STANDALONE_ROTATOR
    double weights[2];
#endif
};

struct p2x2_fp32_struct
{

    float    pix[4];
    bool     pix_bg[4];
    uint32_t weights_fixed[2];
    double   weights[2];
    uint16_t line, pixel;
};

struct bus8B_struct
{
    uint8_t pix[8];
};

struct bus8ui16_struct
{
    uint16_t pix[8];
};

struct bus8ui64_struct
{
    uint64_t pix[8];
};

struct bus16B_struct
{
    uint8_t pix[16];
};

struct bus16ui16_struct
{
    uint16_t pix[16];
};

struct bus16f_struct
{
    bool pix[16];
};

struct bus32f_struct
{
    bool pix[32];
};

struct bus32B_struct
{
    uint8_t pix[32];
};

struct bus64B_struct
{
    uint8_t pix[64];
};

struct bus128B_struct
{
    uint8_t  pix[128];
    uint16_t en[128];
};

struct bus16ui64_struct
{
    uint64_t pix[16];
};

struct XiYi_struct
{
    int64_t X, Y;
};

struct XiYi_float_struct
{
    float X, Y;
};

struct bus8XiYi_struct
{
    XiYi_struct pix[8];
};

struct bus16px_struct
{
    float pix[2][16];
    float xi[8];
    float hpi[8];
    int   XL;
    int   num_vld_pxl;
    int   curr_line;
};
struct bus8XiYi_float_struct
{
    XiYi_float_struct pix[8];
};

struct float32_struct
{
    bool sign;
    int  exp;
    int  mantissa;
};

struct resamp_tracker
{
    uint16_t task     = 0;
    uint32_t pstripe  = 1; // no of stripes
    uint32_t pline    = 0;
    uint32_t pcoord   = 0;
    uint32_t comp_cnt = 0;
    bool     done     = false;
};

struct rescale_tracker
{
    uint16_t task;
    uint32_t pstripe; // no of stripes
    uint32_t pline;
    uint32_t pcoord;
    uint32_t vtap_cnt;
    uint32_t htap_cnt;
    uint32_t vrowMod2;
    uint32_t hrowMod2;
    bool     done;
};

// struct num_err_s {
//   bool rinterp_pinf_error  ;
//   bool rinterp_ninf_error  ;
//   bool rinterp_nan_error	 ;
//   bool minterp_pinf_error  ;
//   bool minterp_ninf_error  ;
//   bool minterp_nan_error	 ;
//   bool coord_pinf_error	 ;
//   bool coord_ninf_error	 ;
//   bool coord_nan_error	  ;
//};

struct meta_data_struct
{
    int16_t line     = 0;
    int16_t Xi_start = 0;
    uint8_t task     = 0;
    //---------------------------------------
    // warp_read metadata
    int16_t stripe = 0;
    int16_t coord  = 0;
    //---------------------------------------
    // resamp meta data
    uint8_t  bank_row = 0;
    uint8_t  set      = 0;
    uint8_t  way      = 0;
    uint32_t tag      = 0;
    // for owm address compute for bwd-2
    int32_t row  = 0;
    int32_t col  = 0;
    int8_t  last = 0; // debug purpose only
    //---------------------------------------
    // only for debug purpose
    resamp_tracker parser_cnt;
};

struct irt_cfifo_data_struct
{
    int64_t  Xi_fixed, Yi_fixed;
    uint16_t line, pixel;
    int16_t  rd_addr[2][IRT_ROT_MEM_COL_BANKS];
    uint8_t  task;
    int16_t  Xi_start[2], XL, XR, YT, YB;
    uint8_t  bank_row[2], rd_shift[2];
    bool     rd_sel[2][IRT_ROT_MEM_COL_BANKS], msb_lsb_sel[2][IRT_ROT_MEM_COL_BANKS],
        bg_flag[2][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH], rd_mode[2];
};

struct irt_rot_mem_rd_stat
{
    uint8_t bank_num_read[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS]; // number of accesses to bank
    int16_t bank_max_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
        bank_min_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
};
struct perf_info_s
{
    float    rd_bw;
    float    wr_bw;
    float    comp_proc_size;
    float    avg_proc_size;
    uint32_t bound_cycles;
    uint32_t compute_cycles;
    uint32_t rd_bw_cycles;
    uint32_t wr_bw_cycles;
};

struct irt_ext_sig
{

    uint16_t         rm_wr_addr;
    bool             rm_wr_sel[IRT_ROT_MEM_ROW_BANKS];
    bus16B_struct    rm_din[IRT_ROT_MEM_COL_BANKS];
    uint16_t         rm_meta_addr;
    meta_data_struct rm_meta_data;
    uint8_t          rm_bank_row[IRT_ROT_MEM_ROW_BANKS];
    bool             rm_rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
    int16_t          rm_rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
    bool             ofifo_push;
    int64_t          Xi_fixed[IRT_ROT_MAX_PROC_SIZE], Yi_fixed[IRT_ROT_MAX_PROC_SIZE];
    p2x2_ui16_struct ip2x2[IRT_ROT_MAX_PROC_SIZE];
    bool             irm_rd_first[IRT_RM_CLIENTS], irm_rd_last[IRT_RM_CLIENTS], owm_wr_first, owm_wr_last;
    int16_t          XiL, XiR, YiT, YiB;
    uint16_t         bli_bg_pxl;
    bool             psel_rot90;

    bool    cfifo_push, cfifo_pop;
    uint8_t cfifo_push_size, cfifo_pop_size;

    uint16_t              mm_wr_addr;
    bool                  mm_wr_sel[IRT_MESH_MEM_ROW_BANKS];
    bus64B_struct         mm_din[IRT_MESH_MEM_COL_BANKS];
    uint16_t              mm_meta_addr;
    meta_data_struct      mm_meta_data;
    uint8_t               mm_bank_row[IRT_MESH_MEM_ROW_BANKS];
    bool                  mm_rd_sel[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS];
    int16_t               mm_rd_addr[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS];
    int16_t               XmL, XmR, YmT, YmB;
    bool                  mofifo_push;
    int64_t               Xm_fixed[IRT_ROT_MAX_PROC_SIZE], Ym_fixed[IRT_ROT_MAX_PROC_SIZE];
    p2x2_ui64_struct      mp2x2x64[IRT_ROT_MAX_PROC_SIZE];
    p2x2_fp32_struct      mp2x2_fp32_x[IRT_ROT_MAX_PROC_SIZE], mp2x2_fp32_y[IRT_ROT_MAX_PROC_SIZE];
    bus8XiYi_float_struct mbli_out;
    bool                  micc_task_done;
    //---------------------------------------
    // ROTSIM_DV_INTGR
    int          adj_proc_size, interp_window_size;
    int          mesh_adj_proc_size;
    bool         mesh_ofifo_valid;
    int          mesh_intp_task_end, mesh_mmrm_pxl_oob;
    unsigned int dv_weights_fixed[8][2];
    int64_t      dv_xi[8], dv_yi[8];
    int64_t      dv_xi_xc[8], dv_yi_yc[8];
    int64_t      dv_xi_xc_xl[8], dv_yi_yc_yt[8];
    bool         dv_irt_done[TASK_BLOCKS];
    uint32_t     dv_Xf_fixed[8], dv_Yf_fixed[8];
    ;
    int                   dv_XI_fixed[8][2], dv_YI_fixed[8][2];
    irt_cfifo_data_struct dv_cfifo_data[8];
    int                   dv_Xint[8][2], dv_Yint[8][2];
    int                   dv_owm_task_done;
    int                   bg_mode_oob_cp[8]; // frame extension OOB cover point data from model
    int                   bg_mode_oob_cp_pxl_index;
    //---------------------------------------
    uint8_t resamp_psize;
    bool    resamp_psize_vld;
    uint8_t next_lru_way;
    uint8_t bnk_row_owm;
    //   queue<resamp_rmwm_inf>  rmwm_infQ[NUM_BANK_ROW];
    int mstripe, gstripe;
    //---------------------------------------
    // Perf Info
    perf_info_s      perf;
    num_err_status_s num_err;
    //---------------------------------------
};

struct im
{
    int       h;
    int       w;
    int       c;
    float**** data;
};

typedef struct _warp_line
{
    float*   x; // array of x coordinates. size == descr.[WIMAGE].S
    float*   y; // array of x coordinates. size == descr.[WIMAGE].S
    bool     valid; // indicates if line fill by warp_read is done
    uint16_t size; // line size == stripe size  i.e. number of X/Y //to simply consumption by PARSER
} warp_line;

// struct warp_parser_cnt {
//   uint32_t num_stripes; // no of stripes
//   uint32_t height;
//   uint32_t s_width;
//   bool done;
//};

struct cache_tag
{
    // actual tag data
    uint32_t pch = 0; // only use in victimQ
    uint8_t  set = 0;
    uint8_t  way = 0;
    uint32_t ref = 0;
    // debug data
    int32_t  row          = 0;
    int32_t  col          = 0;
    uint32_t bank_row     = 0;
    uint32_t bank_row_set = 0;
    uint32_t bank_col_set = 0;
    uint32_t miss_or_hit  = 0;
    //---------------------------------------
    // only for debug purpose
    resamp_tracker parser_cnt;
};

struct irt_mem_ctrl_struct
{

    int16_t  first_line[MAX_TASKS], last_line[MAX_TASKS], start_line[MAX_TASKS];
    uint16_t top_ptr, bot_ptr, fullness;
    bool     wr_done[MAX_TASKS], lines_valid[MAX_TASKS], rmwr_task_done, rd_done[MAX_TASKS];
};
// A Queue Node (Queue is implemented using Doubly Linked List)
typedef struct QNode
{
    struct QNode *prev, *next;
    unsigned      pageNumber; // the page number stored in this QNode
} QNode;

// A Queue (A FIFO collection of Queue Nodes)
typedef struct Queue
{
    unsigned count; // Number of filled frames
    unsigned numberOfFrames; // total number of frames
    QNode *  front, *rear;
} Queue;

// A hash (Collection of pointers to Queue Nodes)
typedef struct Hash
{
    int     capacity; // how many pages can be there
    QNode** array; // an array of queue nodes
} Hash;

struct sb_struct
{
    uint64_t         addr = 0;
    uint16_t         lpad = 0;
    uint16_t         mpad = 0;
    meta_data_struct metadata; // TODO - CHECK IF ROT metadata can be used here
    bus128B_struct   data; // used for OWM transaction
};

struct tag_array
{
    uint64_t  pagenumbers[NSETS][NWAYS];
    uint8_t   valid[NSETS][NWAYS];
    uint8_t   tree_bits[NSETS][NWAYS - 1]; // PLRU need n-1 bits for n-way set
    cache_tag tags[NSETS][NWAYS]; // PLRU need n-1 bits for n-way set
};
//---------------------------------------
// Below prints only for SV debug
struct resamp_2x2_inf
{
    uint32_t proc_size;
    uint32_t pix_vld[8];
    uint32_t pix[8][4];
    uint32_t oPix[8][2];
    uint32_t weights[8][2];
    uint32_t grad[8];
};

struct resamp_rmwm_inf
{
    uint32_t data[32];
    uint32_t addr[4];
    uint32_t valid[4];
    uint32_t write_type; // 1 - rmwm write Vs 2 - interp Writes
};
struct resamp_rmrm_inf
{
    uint32_t proc_size;
    uint32_t addr[NUM_BANK_ROW][4];
    uint32_t valid[NUM_BANK_ROW][4];
    uint32_t xf[NUM_BANK_ROW]; // fract
    uint32_t yf[NUM_BANK_ROW]; // fract
};
struct resamp_victim_inf
{
    uint32_t pch;
    uint32_t set;
    uint32_t way;
    uint32_t ref;
    uint32_t bnkid;
};

struct rescale_v_interp_inf
{
    uint32_t in_a[16];
    uint32_t in_b[16];
    uint32_t in_c[16];
    uint32_t in_d[16];
    uint32_t vld_pxl;
    uint32_t task_num;
};

struct rescale_h_interp_inf
{
    uint32_t in_a[8];
    uint32_t in_b[8];
    uint32_t in_c[8];
    uint32_t in_d[8];
    uint32_t vld_pxl;
    uint32_t task_num;
};
struct resamp_shfl_inf
{
    uint32_t x[128];
    uint32_t y[128];
    uint32_t size;
};
extern queue<resamp_rmwm_inf> rmwm_infQ[NUM_BANK_ROW];
extern queue<resamp_rmrm_inf> rmrm_infQ;
extern queue<resamp_2x2_inf>  p2x2_infQ;
extern queue<cache_tag>       victim_infQ;
extern queue<resamp_shfl_inf> gshfl_infQ;
extern queue<resamp_shfl_inf> wshfl_infQ;

extern queue<rescale_v_interp_inf> rescale_v_intp_infQ;
extern queue<rescale_h_interp_inf> rescale_h_intp_infQ;
//---------------------------------------
#ifdef USE_OLD_FP_CONV
union IRThalf
{
    IRThalf(){};
    half     val_hf;
    uint16_t val_uint;
};
#endif
// union IRTbf16 {
//      IRTbf16() {};
//      flx::floatx<8,7> val_hf;
//      uint64_t val_uint;
//    };
class IRT_top
{
   public:
    IRT_top(bool enable_mesh_mode);
    ~IRT_top();
    void pars_reset();
    void desc_reset();
    void reset(bool reset_subblocks = true);
    void reset_done_status(); // ROTSIM_DV_INTGR -- ADDED TO ALLOW SINGLE DESCRIPTOR SUPPORT
    //---------------------------------------
    // IRT_ RUN
    bool run(uint64_t&             iimage_addr,
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
             bus128B_struct&       oimage_wr_data);

    //---------------------------------------
    const std::string& getInstanceName() const { return m_name; }
    void               setInstanceName(const std::string& str);
    void               irt_desc_print(IRT_top* irt_top, uint8_t desc);
    void               irt_rot_desc_print(IRT_top* irt_top, uint8_t desc);
    void               irt_aff_desc_print(IRT_top* irt_top, uint8_t desc);
    void               irt_prj_desc_print(IRT_top* irt_top, uint8_t desc);
    void               irt_mesh_desc_print(IRT_top* irt_top, uint8_t desc);
    void               irt_resamp_desc_print(IRT_top* irt_top, uint8_t desc);

    // configuration and descriptors
    irt_cfg_pars irt_cfg;
    rotation_par rot_pars[MAX_TASKS];
    irt_desc_par irt_desc[MAX_TASKS];
    //---------------------------------------
    // irt_desc_done -> For non-resamp - updated by OWM module for completion.
    //              -> For resamp - update by RESAMP
    //                            - monitor for each pipe before proceeding for next task
    bool irt_desc_done[MAX_TASKS]; // task level completion status
    ////---------------------------------------
    uint8_t num_of_tasks;

    // global variables
    uint8_t             task[TASK_BLOCKS];
    bool                irt_done[TASK_BLOCKS];
    irt_ext_sig         irt_top_sig;
    irt_mem_ctrl_struct mem_ctrl[IRT_RM_CLIENTS];

    bus8XiYi_float_struct mesh_ofifo;

    // IICC functions
    static int64_t irt_iicc_fixed_0(int32_t coef0, int32_t coef1, int16_t x, int16_t y, uint8_t prec_align);
    static int64_t irt_iicc_fixed_k(int64_t coord_0, int32_t coef, uint8_t k, uint8_t prec_align);
    static float   irt_iicc_float_0(float coef0, float coef1, float x, float y);
    static float   irt_iicc_float_k(float coord_0, float coef, uint8_t k);
    static void
                   irt_iicc_fixed(int64_t out[IRT_ROT_MAX_PROC_SIZE], int32_t M[2], int16_t x, int16_t y, uint8_t prec_align);
    static void    irt_iicc_float(float   out[IRT_ROT_MAX_PROC_SIZE],
                                  float   N[3],
                                  float   D[3],
                                  float   x,
                                  float   y,
                                  float   rot_slope,
                                  float   affine_slope,
                                  uint8_t psize,
                                  float   psize_inv,
                                  uint8_t mode);
    static void    irt_iicc_float_inc(float out[2], float coef[3], float x, float y, uint8_t k);
    static void    irt_iicc(const irt_desc_par&   irt_desc,
                            int16_t               Xo0,
                            int16_t               Yo,
                            int64_t               Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                            int64_t               Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                            bus8XiYi_float_struct mbli_out,
                            Eirt_block_type       block_type);
    static int64_t irt_fix32_to_fix31(int64_t in);

    static int16_t xi_start_calc(const irt_desc_par& irt_desc, int16_t line, uint8_t caller, uint8_t desc);
    static void
                   xi_start_calc_err_print(const char error_str[50], int64_t Xi_start_tmp, int16_t line, uint8_t caller, uint8_t desc);
    static int16_t yi_start_calc(const irt_desc_par& irt_desc, Eirt_block_type block_type);
    static int16_t YB_adjustment(const irt_desc_par& irt_desc, Eirt_block_type block_type, int16_t YB);

    // static void		irt_XiYi_to_rot_mem_read(IRT_top* irt_top, irt_desc_par irt_desc, uint8_t irt_task, std::string
    // my_string, irt_cfifo_data_struct& rot_mem_rd_ctrl);

    // rate control functions
    uint8_t irt_rate_ctrl1(IRT_top* irt_top,
                           int64_t  Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                           int64_t  Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                           uint8_t  proc_size);
    uint8_t irt_rate_ctrl2(IRT_top*              irt_top,
                           irt_cfifo_data_struct cfifo_data_o[IRT_ROT_MAX_PROC_SIZE],
                           bool                  rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
                           int16_t               rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS],
                           uint8_t               max_proc_size,
                           bool                  rate_mode);
    void    irt_rot_mem_rd_stat_calc(IRT_top* irt_top, irt_cfifo_data_struct rd_ctrl, irt_rot_mem_rd_stat& rd_stat);
    //---------------------------------------

    class IRT_UTILS
    {
       public:
        IRT_UTILS() {}
        static int64_t  irt_min_int64(int64_t a, int64_t b);
        static int64_t  irt_max_int64(int64_t a, int64_t b);
        static int16_t  irt_min_int16(int16_t a, int16_t b);
        static int16_t  irt_max_int16(int16_t a, int16_t b);
        static int16_t  irt_sat_int16(int16_t in, int16_t a, int16_t b);
        static float    irt_fp32_to_float(uint32_t in);
        static uint32_t irt_float_to_fp32(float in);
        static int16_t  irt_ui16_to_i16(uint16_t in);
        static int64_t  irt_min_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize);
        static int64_t  irt_max_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize);
        //---------------------------------------
        // resamp utility functions
        // TODO - sjagadale - do we need these to be static functions
        static float    conversion_bit_float(uint32_t in, Eirt_resamp_dtype_enum DataType, bool clip_fp);
        static uint32_t conversion_float_bit(float                  in,
                                             Eirt_resamp_dtype_enum DataType,
                                             int16_t                bli_shift,
                                             uint16_t               MAX_VALo,
                                             uint16_t               Ppo,
                                             bool                   clip_fp,
                                             bool                   clip_fp_inf_input,
                                             bool                   ftz_en,
                                             bool                   print);
        static int16_t  conversion_float_fxd16(float inf, uint8_t int16_point);
        static float    conversion_fxd16_float(int32_t infxd, uint8_t int16_point, bool input_32bit);
        static float    get_rescale_coeff(filter_type_t flt_type, Coeff_LUT_t* myLUT, int pi, int ti);
    };

    template <Eirt_block_type block_type>
    class IRT_IRM
    {
       public:
        IRT_IRM<block_type>(IRT_top* irt_top1);
        ~IRT_IRM<block_type>();
        void               reset();
        bool               run(uint64_t&             addr,
                               meta_data_struct&     meta_out,
                               bool&                 mem_rd,
                               const bus128B_struct& rd_data,
                               meta_data_struct      meta_in,
                               bus128B_struct&       fifo_wr_data,
                               bool&                 fifo_push,
                               meta_data_struct&     fifo_meta_in,
                               bool                  fifo_full,
                               uint16_t&             lsb_pad,
                               uint16_t&             msb_pad);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        bool        tasks_start, task_start, line_start, mem_rd_int;
        int16_t     Yi_start, Xi_start; /*Xi_start_float,*/
        int64_t     Xi_start_fixed;
        int16_t     line, pixel;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_IRM
        FILE*       irm_out_file;
        uint16_t*** irm_rd_image;
#endif
    };

    template <class BUS_OUT, Eirt_block_type block_type>
    class IRT_IFIFO
    {
       public:
        IRT_IFIFO<BUS_OUT, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(bool                  push,
                               bool                  pop,
                               bool                  read,
                               const bus128B_struct& data_in,
                               meta_data_struct      metain,
                               BUS_OUT&              data_out,
                               meta_data_struct&     metaout,
                               bool&                 empty,
                               bool&                 full);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*         irt_top;
        uint8_t          irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry, fifo_fullness;
        bus128B_struct   irt_ififo[IRT_IFIFO_DEPTH];
        meta_data_struct irt_ififo_meta[IRT_IFIFO_DEPTH];
        std::string      m_name;
    };

    template <class BUS_IN,
              class BUS_OUT,
              uint16_t        ROW_BANKS,
              uint16_t        COL_BANKS,
              uint16_t        BANK_HEIGHT,
              Eirt_block_type block_type>
    class IRT_RMWM
    {
       public:
        IRT_RMWM<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_HEIGHT, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void reset();
        bool run(bool              ififo_empty,
                 bool&             ififo_pop,
                 bool&             ififo_read,
                 const BUS_IN&     ififo_data,
                 meta_data_struct  ififo_meta,
                 uint16_t&         rm_wr_addr,
                 bool              rm_wr_sel[ROW_BANKS],
                 BUS_OUT           rm_din[COL_BANKS],
                 uint16_t&         rm_meta_wr_addr,
                 meta_data_struct& rm_meta_data);

        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        bool        tasks_start, line_start, task_start, done;
        int16_t     pixel, line, Yi_start;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_RMWM
        uint64_t irt_rmwm_image[PLANES][500][500];
        FILE*    rmwm_out_file;
#endif
    };

    template <class BUS,
              uint16_t        ROW_BANKS,
              uint16_t        COL_BANKS,
              uint16_t        BANK_WIDTH,
              uint16_t        BANK_HEIGHT,
              Eirt_block_type block_type>
    class IRT_ROT_MEM
    {
       public:
        IRT_ROT_MEM<BUS, ROW_BANKS, COL_BANKS, BANK_WIDTH, BANK_HEIGHT, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(bool             wr_port,
                               uint16_t         wr_addr,
                               bool             wr_sel[ROW_BANKS],
                               BUS              data_in[COL_BANKS],
                               uint16_t         meta_wr_addr,
                               meta_data_struct meta_data,
                               int16_t          rd_addr[ROW_BANKS][COL_BANKS],
                               bool             rd_sel[ROW_BANKS][COL_BANKS],
                               BUS              data_out[ROW_BANKS][COL_BANKS],
                               bool             rd_en,
                               uint8_t          irt_task);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }
        meta_data_struct   meta_mem[ROW_BANKS][IRT_META_MEM_BANK_HEIGHT];

       private:
        IRT_top* irt_top;
        BUS      rot_mem[ROW_BANKS][COL_BANKS][BANK_HEIGHT];

        uint8_t     task_int;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_RM
        FILE*    rm_out_file;
        uint64_t irt_rm_image[PLANES][500][500];
#endif
    };

    template <uint16_t ROW_BANKS, Eirt_block_type block_type>
    class IRT_OICC
    {
       public:
        IRT_OICC<ROW_BANKS, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        bool               run(int16_t& Xo0,
                               int16_t& Yo,
                               uint8_t& plane,
                               int16_t  YT,
                               int16_t  YB,
                               uint8_t& pixel_valid,
                               bool&    ofifo_push,
                               bool     ofifo_full,
                               bool&    task_end,
                               uint8_t& adj_proc_size);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        int16_t     line, YT_min, Yi_start;
        uint16_t    pixel;
        bool        line_start, task_done;
        std::string m_name;
    };

    template <Eirt_block_type block_type>
    class IRT_OICC2
    {
       public:
        IRT_OICC2<block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        bool               run(int16_t&              Xo0,
                               int16_t&              Yo,
                               uint8_t&              plane,
                               uint8_t&              pixel_valid,
                               bool&                 cfifo_push,
                               uint8_t               cfifo_emptyness,
                               bool                  mesh_ofifo_valid,
                               bool&                 task_end,
                               uint8_t&              adj_proc_size,
                               irt_cfifo_data_struct cfifo_data_in[IRT_ROT_MAX_PROC_SIZE]);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        uint16_t    line, pixel;
        bool        line_start, task_done;
        std::string m_name;
    };

    template <Eirt_block_type block_type>
    class IRT_CFIFO
    {
       public:
        IRT_CFIFO<block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(bool                  push,
                               bool                  pop,
                               uint8_t               push_size,
                               uint8_t               pop_size,
                               irt_cfifo_data_struct data_in[IRT_ROT_MAX_PROC_SIZE],
                               irt_cfifo_data_struct data_out[IRT_ROT_MAX_PROC_SIZE],
                               uint8_t&              emptyness,
                               uint8_t&              fullness);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*              irt_top;
        irt_cfifo_data_struct cfifo[IRT_CFIFO_SIZE];
        uint8_t               cfifo_wp, cfifo_rp, cfifo_fullness;
        std::string           m_name;
    };

    template <Eirt_block_type block_type>
    class IRT_IIIRC
    {
       public:
        IRT_IIIRC<block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(int16_t  Xo0,
                               int16_t  Yo,
                               int64_t  Xi_fixed[IRT_ROT_MAX_PROC_SIZE],
                               int64_t  Yi_fixed[IRT_ROT_MAX_PROC_SIZE],
                               int16_t& XL,
                               int16_t& XR,
                               int16_t& YT,
                               int16_t& YB,
                               uint8_t  irt_task,
                               bool     ofifo_push,
                               uint8_t& adj_proc_size);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        bool        error_flag;
        std::string m_name;
    };

    // based on original OICC & IRT_IIIRC
    template <Eirt_block_type block_type>
    class IRT_IIIRC2
    {
       public:
        IRT_IIIRC2<block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        bool               run(int16_t&              Xo0,
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
                               int16_t               rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS]);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        int16_t     line, YT_min, Yi_start;
        uint16_t    pixel;
        uint16_t    timeout_cntr;
        bool        line_start, task_done, error_flag;
        std::string m_name;
    };

    template <uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, Eirt_block_type block_type>
    class IRT_RMRM
    {
       public:
        IRT_RMRM<ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(int16_t  XL,
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
                               uint8_t  caller);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        std::string m_name;
    };

    template <class BUS_IN,
              class BUS_OUT,
              uint16_t        ROW_BANKS,
              uint16_t        COL_BANKS,
              uint16_t        BANK_WIDTH,
              Eirt_block_type block_type>
    class IRT_8x16_IWC
    {
       public:
        IRT_8x16_IWC<BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(BUS_IN        in_pix[ROW_BANKS][COL_BANKS],
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
                               uint8_t       oicc_task);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        std::string m_name;
#ifdef STANDALONE_ROTATOR
        bus16B_struct win16x8[4][MAX_TASKS][ROW_BANKS];
        FILE *        f1, *f2, *f3, *f4;
#endif
    };

    template <class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, Eirt_block_type block_type>
    class IRT_2x2_sel
    {
       public:
        IRT_2x2_sel<BUS_IN, BUS_OUT, ROW_BANKS, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(uint8_t       index,
                               int16_t       XL,
                               int16_t       XR,
                               int16_t       YT,
                               int16_t       YB,
                               float         Xi0,
                               float         Yi0,
                               int64_t       Xi_fixed,
                               int64_t       Yi_fixed,
                               BUS_IN        in_pix[ROW_BANKS],
                               bus16f_struct in_flags[ROW_BANKS],
                               BUS_OUT&      out_pix,
                               uint8_t       irt_task,
                               bool          ofifo_push);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        std::string m_name;
    };

    template <class BUS_IN, class BUS_OUT, Eirt_block_type block_type>
    class IRT_BLI
    {
       public:
        IRT_BLI<BUS_IN, BUS_OUT, block_type>(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        BUS_OUT            run(BUS_IN p2x2, uint16_t, bool ofifo_push, uint8_t irt_task);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        std::string m_name;
    };

    class IRT_OFIFO
    {
       public:
        IRT_OFIFO(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        }
        void               reset();
        void               run(bool            push,
                               bool            pop,
                               uint8_t         pixel_valid,
                               bus8ui16_struct data_in,
                               bus128B_struct& data_out,
                               bool&           empty,
                               bool&           full,
                               bool            task_end);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*       irt_top;
        uint8_t        irt_ofifo_wp, irt_ofifo_rp, fullness, irt_ofifo_w_cnt;
        uint16_t       pixel_cnt;
        bus128B_struct irt_ofifo[IRT_OFIFO_DEPTH];
        std::string    m_name;
    };

    class IRT_OWM
    {
       public:
        IRT_OWM(IRT_top* irt_top1);
        ~IRT_OWM();
        void               reset();
        bool               run(bool                  ofifo_empty,
                               bool&                 ofifo_pop,
                               const bus128B_struct& data_in,
                               uint64_t&             addr,
                               bus128B_struct&       data_out,
                               uint8_t&              wr);
        const std::string& getInstanceName() const { return m_name; }
        void               setInstanceName(const std::string& str) { m_name = str; }

       private:
        IRT_top*    irt_top;
        bool        tasks_start, task_start, done; //, line_start=1;
        uint16_t    line;
        uint32_t    pixel;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_OWM
        uint16_t*** owm_wr_image;
#endif
    };

    class IRT_RESAMP
    {
       public:
        IRT_RESAMP(IRT_top* irt_top1)
        {
            irt_top = irt_top1;
            reset();
        };
        //		~IRT_RESAMP  ();
        //---------------------------------------
        // struct im m_warp;
        // struct im input_im;
        // struct im output_im;
        int32_t***        c_bli_win;
        uint16_t          no_x_seg;
        uint16_t          last_x_seg;
        int               stripe_w;
        uint8_t           time_conflict_en, space_conflict_en;
        int***            p_bli_win;
        struct cache_tag* pTag;
        int**             tag_a;
        int**             tag_valid;
        tag_array         cache_tag_array;
        tag_array         curr_tag_array; // FROM AMBILI
        int               l_miss;
        uint32_t          curr_compute_cnt; // TODO Ambili : This should be driven by compute thread
        queue<cache_tag>  missQ;
        queue<cache_tag>  victimQ;
        queue<sb_struct>  inputQ;
        queue<sb_struct>  outputQ;
        //      queue<resamp_rmwm_inf>  rmwm_infQ[NUM_BANK_ROW];
        //---------------------------------------
        // STATs counters
        uint32_t lookup_cnt, hit_cnt;
        //---------------------------------------
        //===========Resamp Perf============
        uint32_t num_req;
        uint32_t num_victims;
        uint32_t p_bli_v;
        uint32_t prev_proc_size;
        uint32_t tc1_stall;
        uint32_t num_tc_stall_cycle;
        uint32_t num_nm1_conflicts;
        uint32_t num_nm2_conflicts;
        uint32_t num_stall_misq;
        uint32_t num_same_cl;
        uint32_t two_lookup_cycles;

        queue<queue<uint32_t>> proc_q;
        queue<uint32_t>        dummy_q;
        //========end of Resamp Perf========
        void reset();
        // bool run(uint8_t* iimage, float* wimage, irt_resamp_ctrl_struct ctrl_out);
        void setInstanceName(const std::string& str) { m_name = str; }
        void resamp_descr_checks(irt_desc_par& descr);
        bool run(uint64_t&             iimage_addr,
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
                 bus128B_struct&       oimage_wr_data);
        //---------------------------------------
       private:
        //---------------------------------------
        bool         task_start;
        irt_desc_par descr;
        rotation_par HLpars;
        bool         resamp_irm_first_rd;
        //---------------------------------------
        // MEMORY STRUCTURES FOR ALL IMAGES
        bus128B_struct**      resamp_rot_mem; // resamp_rot_mem[NSETS][NWAYS] Bytes  [8][8][16] Bytes
        warp_line**           resamp_wimage; //[wimage.num_stripes][wimage.height]  //2 for X & Y coordinate
        warp_line**           resamp_gimage; //[IIMAGEH][num_segments][128]Bytes
        resamp_tracker        w_parser_cnt;
        resamp_tracker        comp_parser_cnt;
        rescale_tracker       comp_rescale_cnt;
        uint32_t              proc_comp_cnt;
        uint8_t               bank_cnt;
        uint8_t               way_cnt;
        uint8_t               set_cnt;
        bool                  parser_only_done; // used only for internl to parser function.
        bool                  parser_and_imread_done;
        uint16_t              compute_timeout;
        uint8_t               victim_ctrl_ToQ_print = 1;
        int                   avg_proc_size;
        bus16px_struct        rescale_v_accum;
        queue<bus16px_struct> lineBuffer;
        float                 rescale_h_accum[2][8];
        bool                  owm_first_int;
        //---------------------------------------
        // RESAMPLER COMMON FUNCTIONS
        float*    mycalloc1(int num_img);
        float**   mycalloc2(int num_img, int num_ifms);
        int***    mycalloc3_int(int num_img, int num_ifms, int ifm_rows);
        int**     mycalloc2_int(int num_img, int num_ifms);
        float**** mycalloc4(int num_img, int num_ifms, int ifm_rows, int ifm_cols);
        void      myfree3_int(int*** m3_out, int num_img, int num_ifms, int ifm_rows);
        void      myfree2_int(int** m2_out, int num_img, int num_ifms);
        double    normalRandom();
        //---------------------------------------
        //      void cache_data_fetch(uint8_t proc_size);
        void    coord_to_2x2_window(float data[2], int32_t*** bli_win, int32_t pid);
        int32_t bank_row_count(int32_t*** bli_win, int32_t bli_size);
        void    copy_bli_win(int32_t*** bli_win1, int32_t*** bli_win2, int32_t bli1_size);
        int32_t match_bli_win(int32_t*** bli_win1,
                              int32_t*** bli_win2,
                              int32_t    bli1_size,
                              int32_t    bli2_size,
                              int32_t    data_update_en);
        void    coord_to_tag(float data[2], int32_t** tag, uint16_t cl_size);
        void
             tag_merger(uint8_t p_size, int32_t*** tag, int32_t*** m_tag, int32_t** m_tag_valid, uint8_t& two_lookup_en);
        void ref_gen(int32_t tag[2], uint8_t& bank_row, uint8_t& set, uint32_t& ref);
        void reset_local_tags(uint8_t index);
        //---------------------------------------
        int32_t  matchq(queue<queue<uint32_t>> mq, int32_t*** bli_win2, int32_t id, int32_t& pos);
        void     set_mru(uint8_t set);
        uint8_t  get_lru(uint8_t set);
        void     set_plru_4miss(uint8_t set, uint8_t hit_way); // set MRU
        void     set_plru_4hit(uint8_t set, uint8_t hit_way); // set MRU
        void     get_plru(uint8_t set, uint8_t& miss_way, bool lru_peek); // set MRU
        void     set_tags_valid(uint8_t index, uint8_t value);
        void     ReferencePage(uint8_t set, int ref, uint8_t& way, uint8_t& pvalid, uint8_t& miss_hit);
        void     ReferencePage_comp(uint8_t set, int ref, uint8_t& way, uint8_t& pvalid, uint8_t& miss_hit);
        uint16_t get_num_segments(uint32_t width);
        // uint32_t check_nan_inf(uint32_t pix, uint8_t dtype);
        uint32_t check_nan(uint32_t pix, uint8_t dtype);
        float    fp32_nan_inf_conv(float pix);
        void     check_num_err(uint32_t pix, uint8_t dtype, uint8_t caller);
        //---------------------------------------
        // iimage_ptr->byte wise data for H * W * Ps
        // wimage_ptr-> byte wise data for H * W * Ps -> Ps is for X + Y co-ordinates ie. FP32 -> Ps=8, BINT16 -> PS=4
        void resamp_preload_mems(irt_desc_par descr, uint8_t* iimage_ptr, uint8_t* wimage_ptr, uint8_t* gimage_ptr);
        void gen_irm_reads(uint8_t proc_size, irt_desc_par descr);
        // uint8_t no_conflict_check(int16_t x1,int16_t y1,int16_t x1_p1,int16_t y1_p1,int16_t x2,int16_t y2,int16_t
        // x2_p1,int16_t y2_p1,uint8_t sc_en);
        uint8_t no_conflict_check(int16_t  x1,
                                  int16_t  y1,
                                  int16_t  x1_p1,
                                  int16_t  y1_p1,
                                  int16_t  x2,
                                  int16_t  y2,
                                  int16_t  x2_p1,
                                  int16_t  y2_p1,
                                  uint8_t  sc_en,
                                  uint16_t in_data_type,
                                  bool     bwd2_en);
        uint8_t proc_size_calc(float    Xi[IRT_ROT_MAX_PROC_SIZE],
                               float    Yi[IRT_ROT_MAX_PROC_SIZE],
                               uint8_t  num_elm,
                               uint8_t  sc_en,
                               uint16_t in_data_type,
                               bool     bwd2_en);
#ifdef RUN_WITH_SV
        uint8_t fv_proc_size_calc(uint8_t  tag_vld,
                                  float    Xf[IRT_ROT_MAX_PROC_SIZE],
                                  float    Xc[IRT_ROT_MAX_PROC_SIZE],
                                  float    Yf[IRT_ROT_MAX_PROC_SIZE],
                                  float    Yc[IRT_ROT_MAX_PROC_SIZE],
                                  bool     bwd_pass2_en,
                                  uint16_t in_data_type);
#endif
        void  resamp_mem_alloc_and_init(irt_desc_par descr);
        void  resamp_mem_dealloc(irt_desc_par descr);
        void  rshuffle(warp_line* wl, uint8_t wstride);
        void  rshuffle_scbd(warp_line* wl, Eirt_blocks_enum block_type, Eirt_resamp_dtype_enum DataType);
        float get_resamp_data(int plane, int line, int pixel, uint16_t Hi, uint16_t Wi, int bg_mode, float pad_val);
        // warp read state variables
        resamp_tracker wr_atrc, wr_dtrc, owm_tracker;
        resamp_tracker gr_atrc, gr_dtrc;
        uint32_t       lbc[NUM_OCACHE_RESCALE]; // line byte count
        uint32_t       olsc[NUM_OCACHE_RESCALE]; // output line start coordinate
        uint32_t       ofifo_coord[NUM_OCACHE_RESCALE];
        // owm_write state variables
        queue<uint8_t> owmDataQ;
        queue<uint8_t> owmDataQ0, owmDataQ1;
        bus128B_struct owr_data[2];
        uint8_t        ofifo_dptr; // data cache line pointer for ping-pong
        uint64_t       opixel;
        //      uint32_t cbc,obc;
        uint8_t owm_first_trans;
        //---------------------------------------
        template <Eirt_blocks_enum block_type>
        bool warp_read(uint64_t&             wimage_addr,
                       bool&                 wmem_rd,
                       uint16_t&             wlsb_pad_rd,
                       uint16_t&             wmsb_pad_rd,
                       meta_data_struct&     wimage_meta_out,
                       const bus128B_struct& wimage_rd_data,
                       meta_data_struct      wimage_meta_in,
                       bool                  wmem_rd_data_valid);
        //---------------------------------------
        void warp_parser(uint64_t&         iimage_addr,
                         meta_data_struct& meta_out,
                         bool&             imem_rd,
                         uint16_t&         lsb_pad,
                         uint16_t&         msb_pad);
        void fill_ctrl(const bus128B_struct& iimage_rd_data, meta_data_struct iimage_meta_in, bool imem_rd_data_valid);
        void victim_ctrl();

        void compute();
        void compute_rescale();
        bool owm_writes(uint64_t& oimage_addr, uint8_t& omem_wr, bus128B_struct& oimage_wr_data);
        //---------------------------------------
        const std::string& getInstanceName() const { return m_name; }

       private:
        IRT_top*    irt_top;
        std::string m_name;
        std::string my_string; // = getInstanceName();
        // uint16_t mytaskid;
        FILE* wrFile;
        FILE* owFile;
        FILE* compFile;
        FILE* parseFile;
        FILE* VinterpFile;
        FILE* fillFile;
        FILE* victimFile;
        FILE* grFile;
        FILE* wshflFile;
        FILE* gshflFile;
    };
    IRT_RMRM<IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH, e_irt_block_rot>* IRT_RMRM_block;

   private:
    uint8_t pixel_valid, mesh_pixel_valid;
    bool    rm_rd_mode[IRT_ROT_MEM_ROW_BANKS], mm_rd_mode[IRT_MESH_MEM_ROW_BANKS],
        rm_bg_flags[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH],
        mm_bg_flags[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS][IRT_MESH_MEM_BANK_WIDTH];
    uint8_t /*rm_rd_addr[8][2], rm_rd_sel[8][2],*/ ird_shift[IRT_ROT_MEM_ROW_BANKS], mrd_shift[IRT_MESH_MEM_ROW_BANKS];
    bool /*ofifo_push=0,*/                         ofifo_pop, task_end, mesh_task_end;
    bool ofifo_empty, ofifo_full, ififo_full, ififo_empty, ififo_pop, ififo_read, mfifo_full, mfifo_empty, mfifo_pop,
        mfifo_read;
    bool    m_enable_mesh_mode;
    uint8_t cfifo_fullness, cfifo_emptyness;
    int16_t Xo0, Yo;
    uint8_t YT_bank_row, YmT_bank_row; // XL, XR, YT, YB, Xi0_fixed, Yi0_fixed;
    int16_t Xm0, Ym;
    uint8_t /*rm_Xi_start_addr, moved to external struct signals*/ oicc_task, micc_task, iirc_task;
    uint8_t                                                        adj_proc_size, mesh_adj_proc_size;
    irt_cfifo_data_struct cfifo_data_in[IRT_ROT_MAX_PROC_SIZE], cfifo_data_out[IRT_ROT_MAX_PROC_SIZE];
    bool                  mesh_ofifo_valid;
    bool                  first_call;
    std::string           m_name;

    IRT_IRM<e_irt_block_rot>*                         IRT_IIRM_block;
    IRT_IFIFO<bus32B_struct, e_irt_block_rot>*        IRT_IFIFO_block;
    IRT_RMWM<bus32B_struct,
             bus16B_struct,
             IRT_ROT_MEM_ROW_BANKS,
             IRT_ROT_MEM_COL_BANKS,
             IRT_ROT_MEM_BANK_HEIGHT,
             e_irt_block_rot>*                        IRT_RMWM_block;
    IRT_ROT_MEM<bus16B_struct,
                IRT_ROT_MEM_ROW_BANKS,
                IRT_ROT_MEM_COL_BANKS,
                IRT_ROT_MEM_BANK_WIDTH,
                IRT_ROT_MEM_BANK_HEIGHT,
                e_irt_block_rot>*                     IRT_RMEM_block;
    IRT_OICC<IRT_ROT_MEM_ROW_BANKS, e_irt_block_rot>* IRT_OICC_block;
    IRT_CFIFO<e_irt_block_rot>*                       IRT_CFIFO_block;
    IRT_IIIRC<e_irt_block_rot>*                       IRT_IIRC_block;
    // IRT_RMRM	 <									IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_BANK_WIDTH,
    // e_irt_block_rot > *IRT_RMRM_block;
    IRT_8x16_IWC<bus16B_struct,
                 bus16ui16_struct,
                 IRT_ROT_MEM_ROW_BANKS,
                 IRT_ROT_MEM_COL_BANKS,
                 IRT_ROT_MEM_BANK_WIDTH,
                 e_irt_block_rot>*                                                           IRT_IIWC_block;
    IRT_2x2_sel<bus16ui16_struct, p2x2_ui16_struct, IRT_ROT_MEM_ROW_BANKS, e_irt_block_rot>* IRT_I2x2_block;
    IRT_BLI<p2x2_ui16_struct, uint16_t, e_irt_block_rot>*                                    IRT_IBLI_block;
    IRT_OFIFO*                                                                               IRT_OFIFO_block;
    IRT_OWM*                                                                                 IRT_OIWM_block;

    IRT_IRM<e_irt_block_mesh>*                          IRT_MIRM_block;
    IRT_IFIFO<bus128B_struct, e_irt_block_mesh>*        IRT_MFIFO_block;
    IRT_RMWM<bus128B_struct,
             bus64B_struct,
             IRT_MESH_MEM_ROW_BANKS,
             IRT_MESH_MEM_COL_BANKS,
             IRT_MESH_MEM_BANK_HEIGHT,
             e_irt_block_mesh>*                         IRT_MMWM_block;
    IRT_ROT_MEM<bus64B_struct,
                IRT_MESH_MEM_ROW_BANKS,
                IRT_MESH_MEM_COL_BANKS,
                IRT_MESH_MEM_BANK_WIDTH,
                IRT_MESH_MEM_BANK_HEIGHT,
                e_irt_block_mesh>*                      IRT_MMEM_block;
    IRT_OICC<IRT_MESH_MEM_ROW_BANKS, e_irt_block_mesh>* IRT_MICC_block;
    IRT_IIIRC<e_irt_block_mesh>*                        IRT_MIRC_block;
    IRT_RMRM<IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_WIDTH, e_irt_block_mesh>* IRT_MMRM_block;
    IRT_8x16_IWC<bus64B_struct,
                 bus16ui64_struct,
                 IRT_MESH_MEM_ROW_BANKS,
                 IRT_MESH_MEM_COL_BANKS,
                 IRT_MESH_MEM_BANK_WIDTH,
                 e_irt_block_mesh>*                                                                      IRT_MIWC_block;
    IRT_2x2_sel<bus16ui64_struct, p2x2_ui64_struct, IRT_MESH_MEM_ROW_BANKS, e_irt_block_mesh>*           IRT_M2x2_block;
    IRT_BLI<p2x2_fp32_struct, float, e_irt_block_mesh>*                                                  IRT_MBLI_block;

    IRT_OICC2<e_irt_block_rot>*                                                               IRT_OICC2_block;
    IRT_IIIRC2<e_irt_block_rot>*                                                              IRT_IIRC2_block;
    IRT_8x16_IWC<bus16B_struct,
                 bus16ui16_struct,
                 IRT_MESH_MEM_ROW_BANKS,
                 IRT_MESH_MEM_COL_BANKS,
                 IRT_ROT_MEM_BANK_WIDTH,
                 e_irt_block_rot>*                                                            IRT_IIWC2_block;
    IRT_2x2_sel<bus16ui16_struct, p2x2_ui16_struct, IRT_MESH_MEM_ROW_BANKS, e_irt_block_rot>* IRT_I2x22_block;

    IRT_OICC2<e_irt_block_mesh>* IRT_MICC2_block;
    IRT_RESAMP*                  IRT_RESAMP_block;
};
} // namespace irt_utils_gaudi3

#endif /* IRT_H_ */
