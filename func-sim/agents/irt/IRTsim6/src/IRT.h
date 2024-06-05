/*
 * IRT.h
 *
 *  Created on: Jun 4, 2019
 *      Author: espektor
 */


//non static function         requires class object creation (    new()) and used with -> and can    access members of classes.
//    static function doesn't requires class object creation (w/o new()) and used with :: and cannot access members of classes
//	  static function cannot call to non-static function (because static does not require object creation, while non-static require it
//public function can be used in any class
//private function can be used only it its class

#ifndef IRT_H_
#define IRT_H_

#include <string>
#define _CRT_SECURE_NO_WARNINGS 1


/*******************/
/* Defines section */
/*******************/
#define IRT_VERSION 92
#define IRT_DATE "2021-08-23"

#if !defined(RUN_WITH_SV) && !defined (HABANA_SIMULATION)
//#define CREATE_IMAGE_DUMPS_IRM
//#define CREATE_IMAGE_DUMPS_RMWM
//#define CREATE_IMAGE_DUMPS_RMEM
//#define CREATE_IMAGE_DUMPS_IWC - does not work
//#define CREATE_IMAGE_DUMPS_OWM
//#define IRT_USE_FLIP_FOR_MINUS1
//#define IRT_USE_INTRP_REQ_CHECK
#else
#if defined(HABANA_SIMULATION)
//#define IRT_USE_FLIP_FOR_MINUS1
//#define IRT_USE_INTRP_REQ_CHECK
//#define CREATE_IMAGE_DUMPS
//#define CREATE_IMAGE_DUMPS_OWM
#endif
#endif

#define IRT_MAX_TRACE_LEVEL 3
#define IRT_TRACE_LEVEL_HELP  0
#define IRT_TRACE_LEVEL_ERROR 1
#define IRT_TRACE_LEVEL_WARN  2
#define IRT_TRACE_LEVEL_INFO  3

#define MAX_LOG_TRACE_LEVEL 1
#define LOG_TRACE_L1 1
//#define IRT_PROJ_COEFS_NORMALIZATION

#define IRT_BILINEAR_INT_WEIGHTS_PREC 12
#define IRT_OFIFO_DEPTH 8
#define IRT_OFIFO_WIDTH 128
#define IRT_IFIFO_DEPTH 8
#define IRT_IFIFO_WIDTH 128

//#define IRT_ROT_MEM_HEIGHT1 128
#define IRT_META_MEM_BANK_HEIGHT 128 //1024 lines over 8 banks
#define IRT_ROT_MEM_ROW_BANKS 8
#define IRT_ROT_MEM_COL_BANKS 2
#define IRT_ROT_MEM_BANK_WIDTH 16
#define IRT_ROT_MEM_ROW_WIDTH (2 * IRT_ROT_MEM_BANK_WIDTH)
#define IRT_ROT_MEM_PXL_SHIFT 5 //LOG2(IRT_ROT_MEM_ROW_WIDTH)
#define IRT_ROT_MEM_BANK_HEIGHT 128
#define IRT_IFIFO_OUT_WIDTH (2 * IRT_ROT_MEM_BANK_WIDTH)
#define IRT_IFIFO_R_ENTRIES (128 / IRT_IFIFO_OUT_WIDTH)

#define IRT_MESH_MEM_ROW_BANKS 2
#define IRT_MESH_MEM_COL_BANKS 2
#define IRT_MESH_MEM_BANK_WIDTH 64
#define IRT_MESH_MEM_ROW_WIDTH (2 * IRT_MESH_MEM_BANK_WIDTH)
#define IRT_MESH_MEM_PXL_SHIFT 7 //LOG2(IRT_MESH_MEM_BANK_WIDTH)
#define IRT_MESH_MEM_BANK_HEIGHT 64
#define IRT_MESH_G_PREC 31
#define IRT_MFIFO_OUT_WIDTH (2 * IRT_MESH_MEM_BANK_WIDTH) //8 pixels * 2 components * 4 bytes (FP32)
#define IRT_MFIFO_R_ENTRIES (128 / IRT_MFIFO_OUT_WIDTH)

#define IRT_RM_CLIENTS 2
#define IRT_RM_MODES 8
#define IRT_RM_MAX_STAT_MODE 7
#define IRT_RM_MAX_DYN_MODE 6
#define IRT_MM_MAX_STAT_MODE 6
#define IRT_MM_MAX_DYN_MODE 5
#define IRT_RM_DYN_MODE_LINE_RELEASE IRT_ROT_MEM_ROW_BANKS
#define IRT_MM_DYN_MODE_LINE_RELEASE IRT_MESH_MEM_ROW_BANKS

#define IRT_CFIFO_SIZE 16
#define FP32_SIGN_LOC 31
#define FP32_EXP_LOC 23
#define FP32_EXP_MSK 0xff
#define FP32_MANTISSA_MSK 0x7fffff

#define IIMAGE_W (4*1024)
#define IIMAGE_H (4*1024)
#define OIMAGE_W (4*1024)
#define OIMAGE_H (4*1024)
#define PLANES 3
#define BYTEs4PIXEL 2 //bytes per pixel
#define BYTEs4MESH 8
#define CALC_FORMATS_ROT 5

#define IIMAGE_SIZE IIMAGE_W * IIMAGE_H * PLANES * BYTEs4PIXEL
#define OIMAGE_SIZE OIMAGE_W * OIMAGE_H * PLANES * BYTEs4PIXEL
#define MIMAGE_SIZE OIMAGE_W * OIMAGE_H * BYTEs4MESH

#define IMEM_NUM_IMAGES 1

#define OUT_NUM_IMAGES 1 //10 //store up to 10 processed output images
#define MAX_TASKS (PLANES * OUT_NUM_IMAGES + 1)
#define OMEM_NUM_IMAGES (CALC_FORMATS_ROT * OUT_NUM_IMAGES) //for all calculation formats of hl model x OUT_NUM_IMAGES images
#define EMEM_SIZE  IIMAGE_SIZE + OIMAGE_SIZE * OUT_NUM_IMAGES + MIMAGE_SIZE * OUT_NUM_IMAGES // input, output, mesh

//used for multitask support
#define OUT_NUM_IMAGES_MT 1 //store up to 10 processed output images
#define MAX_TASKS_MT (PLANES * OUT_NUM_IMAGES_MT) + 1
#define OMEM_NUM_IMAGES_MT (CALC_FORMATS_ROT * OUT_NUM_IMAGES_MT) //for all calculation formats of hl model x OUT_NUM_IMAGES images
#define EMEM_SIZE_MT IIMAGE_SIZE + OIMAGE_SIZE * OUT_NUM_IMAGES_MT + MIMAGE_SIZE * OUT_NUM_IMAGES_MT // input, output, mesh

#define IIMAGE_REGION_START 0
#define OIMAGE_REGION_START    IIMAGE_REGION_START + IIMAGE_SIZE
#define MIMAGE_REGION_START    OIMAGE_REGION_START + OIMAGE_SIZE * OUT_NUM_IMAGES
#define MIMAGE_REGION_START_MT OIMAGE_REGION_START + OIMAGE_SIZE * OUT_NUM_IMAGES_MT

#define IIMAGE 0
#define OIMAGE 1
#define MIMAGE 2

#define IRT_PIXEL_WIDTH		0//12 //8
#define IRT_COORD_WIDTH		16
#define IRT_XiYi_first_max  (1 << 14) - 1
#define IRT_Xi_start_max    (1 << 15) - 1
#define IRT_WEIGHT_PREC_EXT 1 + 2 + 1 // +1 because of 0.5 error, +2 because of interpolation, +1 because of weigths multiplication
#define IRT_COORD_PREC_EXT  1
#define IRT_SLOPE_PREC		17
#define IRT_SLOPE_ROUND		(1 << (IRT_SLOPE_PREC - 1))

#define MAX_ROT_ANGLE 89//65//58.99
#define MAX_SI_WIDTH 4096
#define MIN_SI_WIDTH 32
#define MAX_SI_HEIGHT 1024
#define MIN_SI_HEIGHT 8
#define MAX_SM_WIDTH (4096 * 4)
#define MIN_SM_WIDTH 128
#define IRT_ROT_DIR_POS 1
#define IRT_ROT_DIR_NEG 0

#define IRT_INT_WIN_W 9
#define IRT_INT_WIN_H 8

#define IRT_ROT_MAX_PROC_SIZE 8
#define IRT_MESH_MAX_PROC_SIZE 6
#define IRT_MESH_FP32_BYTES 4

/*******************/
/* Strings section */
/*******************/
extern const char* irt_int_mode_s[2];
extern const char* irt_irt_mode_s[4];
extern const char* irt_affn_mode_s[4];
extern const char* irt_proj_mode_s[4];
extern const char* irt_refl_mode_s[4];
extern const char* irt_shr_mode_s[4];
extern const char* irt_proj_order_s[6];
extern const char* irt_mesh_mode_s[4];
extern const char* irt_mesh_order_s[2];
extern const char* irt_mesh_rel_mode_s[2];
extern const char* irt_mesh_format_s[2];
extern const char* irt_mesh_fp_loc_s[16];
extern const char* irt_flow_mode_s[4];
extern const char* irt_rate_mode_s[4];
extern const char* irt_pixel_width_s[5];
extern const char* irt_bg_mode_s[2];
extern const char* irt_crd_mode_s[2];
extern const char* irt_hl_format_s[5];
extern const char* irt_mem_type_s[2];
extern const char* irt_prj_matrix_s[7];
extern const char* irt_flip_mode_s[4];
extern const char* irt_buf_format_s[2];

/*******************/
/* Enums section */
/*******************/
#define TASK_BLOCKS 10
typedef enum {
	e_irt_irm = 0,
	e_irt_rmwm = 1,
	e_irt_oicc = 2,
	e_irt_owm = 3,
	e_irt_mrm = 4,
	e_irt_mmwm = 5,
	e_irt_micc = 6,
	e_irt_oicc2 = 7,
	e_irt_iiirc = 8,
	e_irt_micc2 = 9,
} Eirt_blocks_enum;

typedef enum {
	e_irt_angle_roll = 0,
	e_irt_angle_pitch = 1,
	e_irt_angle_yaw = 2,
	e_irt_angle_shear = 3,
	e_irt_angle_rot = 4,
	e_irt_angle_shr_x = 5,
	e_irt_angle_shr_y = 6,
} Eirt_angle_enum;

typedef enum {
	e_irt_rotation = 0,
	e_irt_affine = 1,
	e_irt_projection = 2,
	e_irt_mesh = 3,
} Eirt_tranform_type;

typedef enum {
	e_irt_rate_fixed = 0, //fixed according to proc_size
	e_irt_rate_adaptive_2x2 = 1, //self-adjusted according to rotation memory reading contention using multiple 2x2 reads, up to proc_size
	e_irt_rate_adaptive_wxh = 2, //self-adjusted according to single interpolation window size, up to proc_size
} Eirt_rate_type;

typedef enum {
	e_irt_mesh_flex = 0,
	e_irt_mesh_fp32 = 1,
} Eirt_mesh_format_type;

typedef enum {
	e_irt_mesh_absolute = 0,
	e_irt_mesh_relative = 1,
} Eirt_mesh_rel_mode_type;

typedef enum {
	e_irt_bg_prog_value = 0,
	e_irt_bg_frame_repeat = 1,
} Eirt_bg_mode_type;

typedef enum {
	e_irt_int_bilinear = 0,
	e_irt_int_nearest = 1,
} Eirt_int_mode_type;

typedef enum {
	e_irt_crd_mode_fixed = 0,
	e_irt_crd_mode_fp32 = 1,
} Eirt_coord_mode_type;

typedef enum {
	e_irt_debug_bypass = 0,
	e_irt_debug_bg_out = 1,
} Eirt_debug_type;

typedef enum {
	e_irt_block_rot = 0,
	e_irt_block_mesh = 1,
} Eirt_block_type;

typedef enum {
	e_irt_flow_wCFIFO_fixed_adaptive_2x2 = 0,
	e_irt_flow_wCFIFO_fixed_adaptive_wxh = 1,
	e_irt_flow_nCFIFO_fixed_adaptive_wxh = 2,
} Eirt_flow_type;

typedef enum {
	e_irt_buf_format_static = 0,
	e_irt_buf_format_dynamic = 1,
} Eirt_buf_format;

typedef enum {
	e_irt_buf_select_man = 0,
	e_irt_buf_select_auto = 1,
} Eirt_buf_select;

typedef enum {
	e_irt_bmp_H20 = 0,
	e_irt_bmp_02H = 1,
} Eirt_bmp_order;

typedef enum {
	e_irt_aff_rotation = 0, //R
	e_irt_aff_scaling = 1,  //S
	e_irt_aff_reflection = 2, //M
	e_irt_aff_shearing = 3, //T
} Eirt_aff_ops;

typedef enum {
	e_irt_xi_start_calc_caller_desc_gen = 0,
	e_irt_xi_start_calc_caller_irm = 1,
	e_irt_xi_start_calc_caller_oicc = 2,
	e_irt_xi_start_calc_caller_rmrm = 3,
	e_irt_xi_start_calc_caller_rmrm_oicc = 4,
	e_irt_xi_start_calc_caller_rmrm_top = 5,
	e_irt_xi_start_calc_caller_rmrm_mmrm = 6,
} Eirt_xi_start_calc_caller;

typedef enum {
	e_irt_rmrm_caller_hl_model = 0,
	e_irt_rmrm_caller_irt_oicc = 1,
	e_irt_rmrm_caller_irt_top = 2,
	e_irt_rmrm_caller_irt_mmrm = 3,
} Eirt_irt_rmrm_caller;

/*******************/
/* Struct section */
/*******************/
//mesh structure
typedef struct _mesh_xy_fp32 {
	float x, y;
} mesh_xy_fp32;

typedef struct _mesh_xy_fp32_meta {
	float x, y;
	uint32_t Si, IBufH_req, MBufH_req, IBufW_req;
	float Xi_first, Yi_first, Xi_last, Yi_last, Ymin;
	bool Ymin_dir, Ymin_dir_swap, XiYi_inf;
} mesh_xy_fp32_meta;

typedef struct _mesh_xy_fp64_meta {
	double x, y;
	uint32_t Si, IBufH_req, MBufH_req, IBufW_req;
	double Xi_first, Yi_first, Xi_last, Yi_last, Ymin;
	bool Ymin_dir, Ymin_dir_swap, XiYi_inf;
} mesh_xy_fp64_meta;

typedef struct _mesh_xy_fi16 {
	int16_t x, y;
} mesh_xy_fi16;

// Internal memory params. Constant values.
struct irt_rm_cfg {
	uint16_t BufW; //buffer width in byte
	uint16_t Buf_EpL; //entries per input line
	uint16_t BufH;
	uint16_t BufH_mod;
};

struct irt_mesh_images {
	mesh_xy_fp32_meta** mesh_image_full; //full mesh matrix
	mesh_xy_fp32** mesh_image_rel;  //relative mesh matrix
	mesh_xy_fp32** mesh_image_fp32; //sparce mesh matrix in fp32 format
	mesh_xy_fi16** mesh_image_fi16; //sparce mesh_matrix in fi16 format
	mesh_xy_fp64_meta** mesh_image_intr; //interpolated sparce matrix
	mesh_xy_fp64_meta** proj_image_full; //projection mesh matrix
};

struct irt_cfg_pars {
	//configurated as high level parameters:
	bool buf_1b_mode[IRT_RM_CLIENTS]; //memory buffer 1byte storage format: 0 - 0...31, 1 - {16,0}, ...{31, 15}; [0] - rot_mem, [1] - mesh_mem
	Eirt_buf_format buf_format[IRT_RM_CLIENTS];  //[0] - rot_mem, [1] - mesh_mem
	Eirt_buf_select buf_select[IRT_RM_CLIENTS];  //memory buffer mode select: 0 - manual by parameter, 1 - auto by descriptor gen;  [0] - rot_mem, [1] - mesh_mem
	uint8_t buf_mode[IRT_RM_CLIENTS]; //memory buffer manual mode; [0] - rot_mem, [1] - mesh_mem
	Eirt_flow_type flow_mode; //0 - new with fixed or full adaptive rate ctrl, 1 - new with fixed or rectangle adaptive rate ctrl, 2 - old with fixed with rectangle adaptive rate ctrl (default 0)

	//below are autogenerated and set at reset
	uint16_t Hb[IRT_RM_CLIENTS], Hb_mod[IRT_RM_CLIENTS];
	irt_rm_cfg rm_cfg[IRT_RM_CLIENTS][IRT_RM_MODES];
	Eirt_debug_type debug_mode;

	irt_mesh_images mesh_images;
};

struct image_par {
	//main parameters:
	uint64_t addr_start;	// 1st byte of the image
	uint64_t addr_end;		// last byte of the image
	uint16_t W;				// image width
	uint16_t H;				// image height
	int16_t Xc;				// image horizontal rotation center .5 precision
	int16_t Yc;				// image vertical rotation center .5 precision
	uint16_t S;				// image stripe size to rotate
	uint16_t Ps;			// pixel size: 0 - 1 byte, 1 - 2 byte, 2 - 4 bytes, 3 - 8 bytes
	uint32_t Hs;			// image stride size
	//calculated parameters:
	uint32_t Size;			// Width * Height

	//memory parameters, set by descriptor generator from high level parameters
	uint8_t buf_mode;
};

struct irt_desc_par {
	//main parameters, provided as high level parameters:
	bool oimage_line_wr_format; //output image write mode
	uint16_t bg;				//output image background value
	Eirt_int_mode_type int_mode; //0 - bilinear, 1 - nearest neighbor
	Eirt_bg_mode_type bg_mode; //0 - programmable background, 1 - frame boundary repeatition
	Eirt_tranform_type irt_mode; // 0 - rotation, 1 - affine transform, 2 - projection, 3 - mesh
	Eirt_rate_type rate_mode; //0 - fixed, equal to processing size, 1 - adaptive
	Eirt_coord_mode_type crd_mode; // 0 - fixed point, 1 - fp32
	uint8_t proc_size;

	//see image_par struct above for details
	struct image_par image_par[3]; //[0] - input image, [1] - output image, [2] - mesh matrix/grid
	uint16_t Ho, Wo;

	//pixel width parameters - derived from high level parameters
	uint16_t Msi;     //input pixel mask
	int16_t bli_shift; //bi-linear interpolation shift
	uint16_t MAX_VALo; //output pixel max value, equal to 2^pixel_width - 1
	uint16_t Ppo;		//output pixel padding

	//rotation calculated parameters - derived from high level parameters
	bool rot_dir;
	bool read_hflip, read_vflip;
	bool rot90;
	bool rot90_intv; //rotation 90 interpolation over input image lines is required. Reduce Wo to allow IBufH fit rot memory and reduce proc_size to 7 to fit 8 lines
	bool rot90_inth; //rotation 90 interpolation over input image pixels is required. Increase Si.
	int32_t im_read_slope;
	int32_t cosi, sini;
	float cosf, sinf;
	int16_t Yi_start; // index of first row
	int16_t Yi_end; // index of last row
	int64_t Xi_first_fixed, Yi_first_fixed;
	int16_t Xi_start_offset, Xi_start_offset_flip;
	uint8_t prec_align;

	//affine calculated parameters - derived from high level parameters
	int32_t M11i, M12i, M21i, M22i; //affine matrix parameters
	float M11f, M12f, M21f, M22f; //affine matrix parameters

	//projection calculated parameters - derived from high level parameters
	float prj_Af[3], prj_Bf[3], prj_Cf[3], prj_Df[3];

	//mesh parameters, provided as high level parameters:
	Eirt_mesh_format_type mesh_format; //0 - flex point, 1 - fp32
	uint8_t mesh_point_location;
	bool mesh_sparse_h, mesh_sparse_v; //0 - not sparse, 1 - sparse
	uint32_t mesh_Gh, mesh_Gv; //mesh granularity U.20
	Eirt_mesh_rel_mode_type mesh_rel_mode; // 0 - abs, 1 - relative
};

struct rotation_par {
	//auto calculated at reset:
	uint8_t MAX_PIXEL_WIDTH, MAX_COORD_WIDTH;
	//precisions parameters
	uint8_t COORD_PREC, WEIGHT_PREC, TOTAL_PREC, WEIGHT_SHIFT;
	uint8_t PROJ_NOM_PREC, PROJ_DEN_PREC;
	uint32_t COORD_ROUND, TOTAL_ROUND, PROJ_NOM_ROUND, PROJ_DEN_ROUND;
	bool rot_prec_auto_adj, mesh_prec_auto_adj, oimg_auto_adj, use_rectangular_input_stripe, use_Si_delta_margin;
	float oimg_auto_adj_rate;

	//temporal variables for internal usage
	uint32_t IBufW_req, IBufH_req, IBufW_entries_req;
	uint32_t MBufW_req, MBufH_req, MBufW_entries_req;

	//pixel width configuration - provided as high level parameters :
	uint8_t Pwi;					// input image pixel width
	uint8_t Ppi;					// image pixel alignment to specify where valid pixel bits are located
	uint8_t Pwo;					// output image pixel width
	//derived from pixel width configuration high level parameters
	uint8_t bli_shift_fix;

	//rotation parameters - provided as high level parameters
	double irt_angles[e_irt_angle_shr_y + 1];
	//double rot_angle;
	//rotation parameters - derived from rot_angle and image size
	double irt_angles_adj[e_irt_angle_shr_y + 1];
	double /*rot_angle_adj*,*/ im_read_slope;
	double cosd, sind;
	int cos16, sin16;
	//used for HL model w/o flip (with rot_angle and not rot_angle_adj)
	double cosd_hl, sind_hl;
	float  cosf_hl, sinf_hl;
	int32_t cosi_hl, sini_hl;
	double Xi_first, Xi_last, Yi_first, Yi_last;
	double Xo_first, Xo_last, Yo_first, Yo_last;
	//int16_t Yi_start;
	int16_t Xi_start;
	uint16_t So8;
	double Si_delta;
	uint32_t IBufH_delta;
	//bool rot90_intv; //rotation 90 interpolation over input image lines is required. Reduce Wo to allow IBufH fit rot memory and reduce proc_size to 7 to fit 8 lines
	//bool rot90_inth; //rotation 90 interpolation over input image pixels is required. Increase Si.

	//affine parameters - provided as high level parameters
	char affine_mode[5];
	uint8_t reflection_mode, shear_mode; //shear modes
	//double shr_x_angle, shr_y_angle; //shear angles
	double Sx, Sy; //scaling factors
	//affine parameters - derived from affine high level parameters
	uint8_t affine_flags[4];
	double M11d, M12d, M21d, M22d;
	double affine_Si_factor;

	//projection parameters - provided as high level parameters
	uint8_t proj_mode; // projection mode (0 - rotation, 1 - affine, 2 - projection)
	char proj_order[5]; //projection order (any composition and order of YRPS (Y-yaw, R-roll, P-pitch, S-shearing)
	double /*proj_angle[3],*/ proj_Zd, proj_Wd;// proj_Zi; projection angles (roll, pitch, yaw), projection focal distance, projection plane distance
	//projection parameters - derived from projection high level parameters
	double proj_R_orig[3][3], proj_R[3][3], proj_T[3];
	double prj_Ad[3], prj_Bd[3], prj_Cd[3], prj_Dd[3];
	int64_t prj_Ai[3], prj_Bi[3], prj_Ci[3], prj_Di[3];

	//mesh distortion parameters - provided as high level parameters
	double dist_x, dist_y, dist_r;
	//mesh parameters - provided as high level parameters
	uint8_t mesh_mode; //(0 - rotation, 1 - affine, 2 - projection, 3 - distortion)
	bool mesh_order; //(0 - pre-distortio, 1 - post-distortion)
	double mesh_Sh, mesh_Sv; //mesh image horizontal/vertical scaling

	bool mesh_dist_r0_Sh1_Sv1; //detection of mesh mode with distortion 0 and no sparse matrix
	bool mesh_matrix_error; //detection of mesh mode with distortion 0 and no sparse matrix and non flex mode

	//processing statistics - internal usage
	uint8_t min_proc_rate, max_proc_rate;
	bool proc_auto;
	uint32_t rate_hist[IRT_ROT_MAX_PROC_SIZE], acc_proc_rate, cycles;

	// memory parameters
	Eirt_buf_format buf_format[IRT_RM_CLIENTS];  //[0] - rot_mem, [1] - mesh_mem
	Eirt_buf_select buf_select[IRT_RM_CLIENTS];  //memory buffer mode select: 0 - manual by parameter, 1 - auto by descriptor gen;  [0] - rot_mem, [1] - mesh_mem
	uint8_t buf_mode[IRT_RM_CLIENTS]; //memory buffer manual mode; [0] - rot_mem, [1] - mesh_mem

	//analize parameters
	bool affine_rotation;
	bool projection_direct_rotation, projection_affine_rotation, projection_rotation, projection_affine;
	bool mesh_direct_rotation, mesh_rotation, mesh_affine_rotation, mesh_projection_rotation, mesh_projection_affine, mesh_direct_affine, mesh_affine;
	bool irt_rotate, irt_affine, irt_affine_st_inth, irt_affine_st_intv, irt_affine_hscaling, irt_affine_vscaling, irt_affine_shearing;

};

//buses structures
struct p2x2_ui16_struct {
	uint16_t pix[4];
	bool pix_bg[4];
	uint32_t weights_fixed[2];
	double weights[2];
	uint16_t line, pixel;
};

struct p2x2_ui64_struct {
	uint64_t pix[4];
	bool pix_bg[4];
	uint32_t weights_fixed[2];
	double weights[2];
};

struct p2x2_ui32_struct {
	uint32_t pix[4];
	bool pix_bg[4];
	uint32_t weights_fixed[2];
#ifdef STANDALONE_ROTATOR
	double weights[2];
#endif
};

struct p2x2_fp32_struct {

	float pix[4];
	bool pix_bg[4];
	uint32_t weights_fixed[2];
	double weights[2];
	uint16_t line, pixel;
};

struct bus8B_struct {
	uint8_t pix[8];
};

struct bus8ui16_struct {
	uint16_t pix[8];
};

struct bus8ui64_struct {
	uint64_t pix[8];
};

struct bus16B_struct {
	uint8_t pix[16];
};

struct bus16ui16_struct {
	uint16_t pix[16];
};

struct bus16f_struct {
	bool pix[16];
};

struct bus32f_struct {
	bool pix[32];
};

struct bus32B_struct {
	uint8_t pix[32];
};

struct bus64B_struct {
	uint8_t pix[64];
};

struct bus128B_struct {
	uint8_t pix[128];
	uint16_t en[128];
};

struct bus16ui64_struct {
	uint64_t pix[16];
};

struct XiYi_struct {
	int64_t X, Y;
};

struct XiYi_float_struct {
	float X, Y;
};

struct bus8XiYi_struct {
	XiYi_struct pix[8];
};

struct bus8XiYi_float_struct {
	XiYi_float_struct pix[8];
};

struct float32_struct {
	bool sign;
	int exp;
	int mantissa;
};

struct meta_data_struct {
	int16_t line;
	int16_t Xi_start;
	uint8_t task;
};

struct irt_cfifo_data_struct {
	int64_t Xi_fixed, Yi_fixed;
	uint16_t line, pixel;
	int16_t rd_addr[2][IRT_ROT_MEM_COL_BANKS];
	uint8_t task;
	int16_t Xi_start[2], XL, XR, YT, YB;
	uint8_t bank_row[2], rd_shift[2];
	bool rd_sel[2][IRT_ROT_MEM_COL_BANKS], msb_lsb_sel[2][IRT_ROT_MEM_COL_BANKS], bg_flag[2][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH], rd_mode[2];
};

struct irt_rot_mem_rd_stat {
	uint8_t bank_num_read[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS]; //number of accesses to bank
	int16_t bank_max_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS], bank_min_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
};

struct irt_ext_sig {

	uint16_t rm_wr_addr;
	bool rm_wr_sel[IRT_ROT_MEM_ROW_BANKS];
	bus16B_struct rm_din[IRT_ROT_MEM_COL_BANKS];
	uint16_t rm_meta_addr;
	meta_data_struct rm_meta_data;
	uint8_t rm_bank_row[IRT_ROT_MEM_ROW_BANKS];
	bool rm_rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
	int16_t rm_rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS];
	bool ofifo_push;
	int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], Yi_fixed[IRT_ROT_MAX_PROC_SIZE];
	p2x2_ui16_struct ip2x2[IRT_ROT_MAX_PROC_SIZE];
	bool irm_rd_first[IRT_RM_CLIENTS], irm_rd_last[IRT_RM_CLIENTS], owm_wr_first, owm_wr_last;
	int16_t XiL, XiR, YiT, YiB;
	uint16_t bli_bg_pxl;
	bool psel_rot90;

	bool cfifo_push, cfifo_pop;
	uint8_t cfifo_push_size, cfifo_pop_size;

	uint16_t mm_wr_addr;
	bool mm_wr_sel[IRT_MESH_MEM_ROW_BANKS];
	bus64B_struct mm_din[IRT_MESH_MEM_COL_BANKS];
	uint16_t mm_meta_addr;
	meta_data_struct mm_meta_data;
	uint8_t mm_bank_row[IRT_MESH_MEM_ROW_BANKS];
	bool mm_rd_sel[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS];
	int16_t mm_rd_addr[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS];
	int16_t XmL, XmR, YmT, YmB;
	bool mofifo_push;
	int64_t Xm_fixed[IRT_ROT_MAX_PROC_SIZE], Ym_fixed[IRT_ROT_MAX_PROC_SIZE];
	p2x2_ui64_struct mp2x2x64[IRT_ROT_MAX_PROC_SIZE];
	p2x2_fp32_struct mp2x2_fp32_x[IRT_ROT_MAX_PROC_SIZE], mp2x2_fp32_y[IRT_ROT_MAX_PROC_SIZE];
	bus8XiYi_float_struct mbli_out;
	bool micc_task_done;
};


struct irt_mem_ctrl_struct {

	int16_t first_line[MAX_TASKS], last_line[MAX_TASKS], start_line[MAX_TASKS];
	uint16_t top_ptr, bot_ptr, fullness;
	bool wr_done[MAX_TASKS], lines_valid[MAX_TASKS], rmwr_task_done, rd_done[MAX_TASKS];
};

class IRT_top {
public:
	IRT_top();
	~IRT_top();
	void pars_reset();
	void desc_reset();
	void reset(bool reset_subblocks = true);
	bool run(uint64_t& iimage_addr, bool& imem_rd, uint16_t& ilsb_pad_rd, uint16_t& imsb_pad_rd, meta_data_struct& iimage_meta_out,	const bus128B_struct&  iimage_rd_data, meta_data_struct iimage_meta_in,
			 uint64_t& oimage_addr, bool& mem_wr, bus128B_struct& oimage_wr_data,
			 uint64_t& mimage_addr, bool& mmem_rd, uint16_t& mlsb_pad_rd, uint16_t& mmsb_pad_rd, meta_data_struct& mimage_meta_out,	const bus128B_struct&  mimage_rd_data, meta_data_struct mimage_meta_in);
    const std::string &getInstanceName() const { return m_name; }
    void setInstanceName(const std::string &str);
	void irt_desc_print(IRT_top* irt_top, uint8_t desc);
	void irt_rot_desc_print(IRT_top* irt_top, uint8_t desc);
	void irt_aff_desc_print(IRT_top* irt_top, uint8_t desc);
	void irt_prj_desc_print(IRT_top* irt_top, uint8_t desc);
	void irt_mesh_desc_print(IRT_top* irt_top, uint8_t desc);

	//configuration and descriptors
	irt_cfg_pars irt_cfg;
	rotation_par rot_pars[MAX_TASKS];
	irt_desc_par irt_desc[MAX_TASKS];
	uint8_t num_of_tasks;

	//global variables
	uint8_t task[TASK_BLOCKS];
	bool irt_done[TASK_BLOCKS];
	irt_ext_sig irt_top_sig;
	irt_mem_ctrl_struct mem_ctrl[IRT_RM_CLIENTS];

	bus8XiYi_float_struct mesh_ofifo;

	// IICC functions
	static int64_t	irt_iicc_fixed_0(int32_t coef0, int32_t coef1, int16_t x, int16_t y, uint8_t prec_align);
	static int64_t	irt_iicc_fixed_k(int64_t coord_0, int32_t coef, uint8_t k, uint8_t prec_align);
	static float	irt_iicc_float_0(float coef0, float coef1, float x, float y);
	static float	irt_iicc_float_k(float coord_0, float coef, uint8_t k);
	static void		irt_iicc_fixed(int64_t out[IRT_ROT_MAX_PROC_SIZE], int32_t M[2], int16_t x, int16_t y, uint8_t prec_align);
	static void		irt_iicc_float(float out[IRT_ROT_MAX_PROC_SIZE], float N[3], float D[3], float x, float y, float rot_slope, float affine_slope, uint8_t psize, float psize_inv, uint8_t mode);
	static void		irt_iicc_float_inc(float out[2], float coef[3], float x, float y, uint8_t k);
	static void		irt_iicc(const irt_desc_par &irt_desc, int16_t Xo0, int16_t Yo, int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], int64_t Yi_fixed[IRT_ROT_MAX_PROC_SIZE], bus8XiYi_float_struct mbli_out, Eirt_block_type block_type);
	static int64_t	irt_fix32_to_fix31(int64_t in);

	static int16_t	xi_start_calc(const irt_desc_par& irt_desc, int16_t line, uint8_t caller, uint8_t desc);
	static void     xi_start_calc_err_print(const char error_str[50], int64_t Xi_start_tmp, int16_t line, uint8_t caller, uint8_t desc);
	static int16_t  yi_start_calc(const irt_desc_par& irt_desc, Eirt_block_type block_type);
	static int16_t  YB_adjustment(const irt_desc_par& irt_desc, Eirt_block_type block_type, int16_t YB);

	//static void		irt_XiYi_to_rot_mem_read(IRT_top* irt_top, irt_desc_par irt_desc, uint8_t irt_task, std::string my_string, irt_cfifo_data_struct& rot_mem_rd_ctrl);

	//rate control functions
	uint8_t	irt_rate_ctrl1(IRT_top* irt_top, int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], int64_t Yi_fixed[IRT_ROT_MAX_PROC_SIZE], uint8_t proc_size);
	uint8_t	irt_rate_ctrl2(IRT_top* irt_top, irt_cfifo_data_struct cfifo_data_o[IRT_ROT_MAX_PROC_SIZE], bool rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS], int16_t rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS], uint8_t max_proc_size, bool rate_mode);
	void	irt_rot_mem_rd_stat_calc(IRT_top* irt_top, irt_cfifo_data_struct rd_ctrl, irt_rot_mem_rd_stat& rd_stat);


	class IRT_UTILS {
	public:
		IRT_UTILS() { }
		static int64_t irt_min_int64(int64_t a, int64_t b);
		static int64_t irt_max_int64(int64_t a, int64_t b);
		static int16_t irt_min_int16(int16_t a, int16_t b);
		static int16_t irt_max_int16(int16_t a, int16_t b);
		static int16_t irt_sat_int16(int16_t in, int16_t a, int16_t b);
		static float	irt_fp32_to_float(uint32_t in);
		static uint32_t irt_float_to_fp32(float in);
		static int16_t	irt_ui16_to_i16(uint16_t in);
		static int64_t irt_min_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize);
		static int64_t irt_max_coord(int64_t in[IRT_ROT_MAX_PROC_SIZE], uint8_t psize);
	};

	template < Eirt_block_type block_type >
	class IRT_IRM {
	public:
		IRT_IRM < block_type > (IRT_top* irt_top1);
		~IRT_IRM < block_type > ();
		void reset();
		bool run(uint64_t& addr, meta_data_struct& meta_out, bool& mem_rd, const bus128B_struct& rd_data, meta_data_struct meta_in,
			bus128B_struct& fifo_wr_data, bool& fifo_push, meta_data_struct& fifo_meta_in, bool fifo_full, uint16_t& lsb_pad, uint16_t& msb_pad);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top *irt_top;
		bool tasks_start, task_start, line_start, mem_rd_int;
		int16_t Yi_start, Xi_start; /*Xi_start_float,*/
		int64_t Xi_start_fixed;
		int16_t line, pixel;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_IRM
		FILE* irm_out_file;
		uint16_t*** irm_rd_image;
#endif
	};

	template < class BUS_OUT, Eirt_block_type block_type >
	class IRT_IFIFO {
	public:
		IRT_IFIFO < BUS_OUT, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset(); }
		void reset();
		void run(bool push, bool pop, bool read, const bus128B_struct& data_in, meta_data_struct metain, BUS_OUT& data_out, meta_data_struct& metaout, bool& empty, bool& full);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
		uint8_t irt_ififo_wp, irt_ififo_rp, irt_ififo_r_entry, fifo_fullness;
		bus128B_struct irt_ififo[IRT_IFIFO_DEPTH];
		meta_data_struct irt_ififo_meta[IRT_IFIFO_DEPTH];
        std::string m_name;
	};

	template < class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_HEIGHT, Eirt_block_type block_type >
	class IRT_RMWM {
	public:
		IRT_RMWM < BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_HEIGHT, block_type >(IRT_top* irt_top1); //{ irt_top = irt_top1; reset();}
		~IRT_RMWM < BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_HEIGHT, block_type > ();
		void reset();
		bool run(bool ififo_empty, bool& ififo_pop, bool& ififo_read, const BUS_IN& ififo_data, meta_data_struct ififo_meta,
			uint16_t& rm_wr_addr, bool rm_wr_sel[ROW_BANKS], BUS_OUT rm_din[COL_BANKS], uint16_t& rm_meta_wr_addr, meta_data_struct& rm_meta_data);

        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
		bool tasks_start, line_start, task_start, done;
		int16_t pixel, line, Yi_start;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_RMWM
		//uint64_t irt_rmwm_image[PLANES][500][500];
		FILE* rmwm_out_file;
		uint16_t*** irt_rmwm_image;
#endif
	};

	template < class BUS, uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, uint16_t BANK_HEIGHT, Eirt_block_type block_type >
	class IRT_ROT_MEM {
	public:
		IRT_ROT_MEM < BUS, ROW_BANKS, COL_BANKS, BANK_WIDTH, BANK_HEIGHT, block_type >(IRT_top* irt_top1);// { irt_top = irt_top1; reset();}
		~IRT_ROT_MEM < BUS, ROW_BANKS, COL_BANKS, BANK_WIDTH, BANK_HEIGHT, block_type >();
		void reset();
		void run(bool wr_port, uint16_t wr_addr, bool wr_sel[ROW_BANKS], BUS data_in[COL_BANKS], uint16_t meta_wr_addr, meta_data_struct meta_data,
			int16_t rd_addr[ROW_BANKS][COL_BANKS], bool rd_sel[ROW_BANKS][COL_BANKS], BUS data_out[ROW_BANKS][COL_BANKS], bool rd_en, uint8_t irt_task);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }
		meta_data_struct meta_mem[ROW_BANKS][IRT_META_MEM_BANK_HEIGHT];

	private:
		IRT_top* irt_top;
		BUS rot_mem[ROW_BANKS][COL_BANKS][BANK_HEIGHT];

		uint8_t task_int;
        std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_RMEM
		FILE* rmem_out_file;
		//uint64_t irt_rm_image[PLANES][500][500];
		uint16_t*** irt_rmem_image;
#endif

	};

	template < uint16_t ROW_BANKS, Eirt_block_type block_type >
	class IRT_OICC {
	public:
		IRT_OICC < ROW_BANKS, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		bool run(int16_t& Xo0, int16_t& Yo, uint8_t& plane, int16_t YT, int16_t YB, uint8_t& pixel_valid, bool& ofifo_push, bool ofifo_full, bool& task_end, uint8_t& adj_proc_size);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
		int16_t line, YT_min, Yi_start;
		uint16_t pixel;
		bool line_start, task_done;
        std::string m_name;
	};

	template < Eirt_block_type block_type >
	class IRT_OICC2 {
	public:
		IRT_OICC2 < block_type >(IRT_top* irt_top1) { irt_top = irt_top1; reset(); }
		void reset();
		bool run(int16_t& Xo0, int16_t& Yo, uint8_t& plane, uint8_t& pixel_valid, bool& cfifo_push, uint8_t cfifo_emptyness, bool mesh_ofifo_valid, bool& task_end, uint8_t& adj_proc_size, irt_cfifo_data_struct cfifo_data_in[IRT_ROT_MAX_PROC_SIZE]);
		const std::string& getInstanceName() const { return m_name; }
		void setInstanceName(const std::string& str) { m_name = str; }

	private:
		IRT_top* irt_top;
		uint16_t line, pixel;
		bool line_start, task_done;
		std::string m_name;
	};

	template < Eirt_block_type block_type >
	class IRT_CFIFO {
	public:
		IRT_CFIFO < block_type >(IRT_top* irt_top1) { irt_top = irt_top1; reset(); }
		void reset();
		void run(bool push, bool pop, uint8_t push_size, uint8_t pop_size, irt_cfifo_data_struct data_in[IRT_ROT_MAX_PROC_SIZE], irt_cfifo_data_struct data_out[IRT_ROT_MAX_PROC_SIZE], uint8_t& emptyness, uint8_t& fullness);
		const std::string& getInstanceName() const { return m_name; }
		void setInstanceName(const std::string& str) { m_name = str; }

	private:
		IRT_top* irt_top;
		irt_cfifo_data_struct cfifo[IRT_CFIFO_SIZE];
		uint8_t cfifo_wp, cfifo_rp, cfifo_fullness;
		std::string m_name;
	};

	template < Eirt_block_type block_type >
	class IRT_IIIRC {
	public:
		IRT_IIIRC < block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		void run(int16_t Xo0, int16_t Yo, int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], int64_t Yi_fixed[IRT_ROT_MAX_PROC_SIZE], int16_t& XL, int16_t& XR, int16_t& YT, int16_t& YB, uint8_t irt_task, bool ofifo_push, uint8_t& adj_proc_size);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
		bool error_flag;
        std::string m_name;
	};

	//based on original OICC & IRT_IIIRC
	template < Eirt_block_type block_type >
	class IRT_IIIRC2 {
	public:
		IRT_IIIRC2 < block_type >(IRT_top* irt_top1) { irt_top = irt_top1; reset(); }
		void reset();
		bool run(int16_t& Xo0, int16_t& Yo, uint8_t& pixel_valid, bool& ofifo_push, bool ofifo_full, bool& task_end, uint8_t& adj_proc_size,
			int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE], int64_t Yi_fixed[IRT_ROT_MAX_PROC_SIZE], int16_t& XL, int16_t& XR, int16_t& YT, int16_t& YB, uint8_t& irt_task,
			uint8_t cfifo_fullness, bool& cfifo_pop, uint8_t& cfifo_pop_size, irt_cfifo_data_struct cfifo_data_out[IRT_ROT_MAX_PROC_SIZE],
			bool rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS], int16_t rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS]);
		const std::string& getInstanceName() const { return m_name; }
		void setInstanceName(const std::string& str) { m_name = str; }

	private:
		IRT_top* irt_top;
		int16_t line, YT_min, Yi_start;
		uint16_t pixel;
		uint16_t timeout_cntr;
		bool line_start, task_done, error_flag;
		std::string m_name;
	};

	template < uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, Eirt_block_type block_type >
	class IRT_RMRM {
	public:
		IRT_RMRM < ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		void run(int16_t XL, int16_t XR, int16_t YT, int16_t YB, uint8_t bank_row[ROW_BANKS],
			bool rd_sel[ROW_BANKS][COL_BANKS], int16_t rd_addr[ROW_BANKS][COL_BANKS], bool rd_mode[ROW_BANKS], uint8_t rd_shift[ROW_BANKS], bool msb_lsb_sel[ROW_BANKS][COL_BANKS],
			uint8_t& YT_bank_row, bool bg_flag[ROW_BANKS][COL_BANKS][BANK_WIDTH], uint8_t irt_task, bool rd_en, int16_t Xo, int16_t Yo, uint8_t caller);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
        std::string m_name;
	};

	template < class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, uint16_t COL_BANKS, uint16_t BANK_WIDTH, Eirt_block_type block_type >
	class IRT_8x16_IWC {
	public:
		IRT_8x16_IWC < BUS_IN, BUS_OUT, ROW_BANKS, COL_BANKS, BANK_WIDTH, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		void run(BUS_IN in_pix[ROW_BANKS][COL_BANKS], bool in_pix_flag[ROW_BANKS][COL_BANKS][BANK_WIDTH], bool rd_mode[ROW_BANKS], uint8_t rd_shift[ROW_BANKS], bool msb_lsb_sel[ROW_BANKS][COL_BANKS], int16_t XL, int16_t YT, uint8_t YT_bank_row, BUS_OUT out_pix[ROW_BANKS], bus16f_struct out_pix_flag[ROW_BANKS], bool ofifo_push, int16_t Yo, int16_t Xo0, uint8_t oicc_task);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
        std::string m_name;
#ifdef STANDALONE_ROTATOR
		bus16B_struct win16x8[4][MAX_TASKS][ROW_BANKS];
		FILE* f1, * f2, * f3, * f4;
#endif
	};

	template < class BUS_IN, class BUS_OUT, uint16_t ROW_BANKS, Eirt_block_type block_type >
	class IRT_2x2_sel {
	public:
		IRT_2x2_sel < BUS_IN, BUS_OUT, ROW_BANKS, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		void run(uint8_t index, int16_t XL, int16_t XR, int16_t YT, int16_t YB, float Xi0, float Yi0, int64_t Xi_fixed, int64_t Yi_fixed, BUS_IN in_pix[ROW_BANKS], bus16f_struct in_flags[ROW_BANKS], BUS_OUT& out_pix, uint8_t irt_task, bool ofifo_push);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
        std::string m_name;
	};

	template < class BUS_IN, class BUS_OUT, Eirt_block_type block_type >
	class IRT_BLI {
	public:
		IRT_BLI < BUS_IN, BUS_OUT, block_type > (IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		BUS_OUT run(BUS_IN p2x2, uint16_t, bool ofifo_push, uint8_t irt_task);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
        std::string m_name;
	};

	class IRT_OFIFO {
	public:
		IRT_OFIFO(IRT_top* irt_top1) { irt_top = irt_top1; reset();}
		void reset();
		void run(bool push, bool pop, uint8_t pixel_valid, bus8ui16_struct data_in, bus128B_struct& data_out, bool& empty, bool& full, bool task_end);
        const std::string &getInstanceName() const { return m_name; }
        void setInstanceName(const std::string &str) { m_name = str; }

	private:
		IRT_top* irt_top;
		uint8_t irt_ofifo_wp, irt_ofifo_rp, fullness, irt_ofifo_w_cnt;
		uint16_t pixel_cnt;
		bus128B_struct irt_ofifo[IRT_OFIFO_DEPTH];
        std::string m_name;

	};

	class IRT_OWM {
	public:
		IRT_OWM(IRT_top* irt_top1);
		~IRT_OWM();
		void reset();
		bool run(bool ofifo_empty, bool& ofifo_pop, const bus128B_struct& data_in, uint64_t& addr, bus128B_struct& data_out, bool& wr);
		const std::string& getInstanceName() const { return m_name; }
		void setInstanceName(const std::string& str) { m_name = str; }

	private:
		IRT_top* irt_top;
		bool tasks_start, task_start, done;//, line_start=1;
		uint16_t line;
		uint32_t pixel;
		std::string m_name;

#ifdef CREATE_IMAGE_DUMPS_OWM
		uint16_t*** owm_wr_image;
#endif
	};

	IRT_RMRM	 <									IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH,			e_irt_block_rot >* IRT_RMRM_block;

private:
	uint8_t pixel_valid, mesh_pixel_valid;
	bool rm_rd_mode[IRT_ROT_MEM_ROW_BANKS], mm_rd_mode[IRT_MESH_MEM_ROW_BANKS], rm_bg_flags[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH], mm_bg_flags[IRT_MESH_MEM_ROW_BANKS][IRT_MESH_MEM_COL_BANKS][IRT_MESH_MEM_BANK_WIDTH];
	uint8_t /*rm_rd_addr[8][2], rm_rd_sel[8][2],*/ ird_shift[IRT_ROT_MEM_ROW_BANKS], mrd_shift[IRT_MESH_MEM_ROW_BANKS];
	bool /*ofifo_push=0,*/ ofifo_pop, task_end, mesh_task_end;
	bool ofifo_empty, ofifo_full, ififo_full, ififo_empty, ififo_pop, ififo_read, mfifo_full, mfifo_empty, mfifo_pop, mfifo_read;
	uint8_t cfifo_fullness, cfifo_emptyness;
	int16_t Xo0, Yo;
	uint8_t YT_bank_row, YmT_bank_row; //XL, XR, YT, YB, Xi0_fixed, Yi0_fixed;
	int16_t Xm0, Ym;
	uint8_t /*rm_Xi_start_addr, moved to external struct signals*/ oicc_task, micc_task, iirc_task;
	uint8_t adj_proc_size, mesh_adj_proc_size;
	irt_cfifo_data_struct cfifo_data_in[IRT_ROT_MAX_PROC_SIZE], cfifo_data_out[IRT_ROT_MAX_PROC_SIZE];
	bool mesh_ofifo_valid;
	bool first_call;
    std::string m_name;

	IRT_IRM		 <																													e_irt_block_rot > *IRT_IIRM_block;
	IRT_IFIFO	 < bus32B_struct,																									e_irt_block_rot > *IRT_IFIFO_block;
	IRT_RMWM	 < bus32B_struct, bus16B_struct, IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_HEIGHT,				e_irt_block_rot > *IRT_RMWM_block;
	IRT_ROT_MEM  < bus16B_struct, IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH, IRT_ROT_MEM_BANK_HEIGHT,	e_irt_block_rot > *IRT_RMEM_block;
	IRT_OICC	 <				  IRT_ROT_MEM_ROW_BANKS,																			e_irt_block_rot > *IRT_OICC_block;
	IRT_CFIFO	 <																													e_irt_block_rot > *IRT_CFIFO_block;
	IRT_IIIRC	 <																													e_irt_block_rot > *IRT_IIRC_block;
	//IRT_RMRM	 <									IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_BANK_WIDTH,									e_irt_block_rot > *IRT_RMRM_block;
	IRT_8x16_IWC < bus16B_struct, bus16ui16_struct, IRT_ROT_MEM_ROW_BANKS, IRT_ROT_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH,			e_irt_block_rot > *IRT_IIWC_block;
	IRT_2x2_sel  < bus16ui16_struct, p2x2_ui16_struct, IRT_ROT_MEM_ROW_BANKS,														e_irt_block_rot > *IRT_I2x2_block;
	IRT_BLI		 < p2x2_ui16_struct, uint16_t,																						e_irt_block_rot > *IRT_IBLI_block;
	IRT_OFIFO	 *IRT_OFIFO_block;
	IRT_OWM		 *IRT_OIWM_block;

	IRT_IRM		 <																													e_irt_block_mesh > *IRT_MIRM_block;
	IRT_IFIFO	 < bus128B_struct,																									e_irt_block_mesh > *IRT_MFIFO_block;
	IRT_RMWM	 < bus128B_struct, bus64B_struct, IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_HEIGHT,			e_irt_block_mesh > *IRT_MMWM_block;
	IRT_ROT_MEM  < bus64B_struct, IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_WIDTH, IRT_MESH_MEM_BANK_HEIGHT,e_irt_block_mesh > *IRT_MMEM_block;
	IRT_OICC	 <				  IRT_MESH_MEM_ROW_BANKS,																			e_irt_block_mesh > *IRT_MICC_block;
	IRT_IIIRC	 <																													e_irt_block_mesh > *IRT_MIRC_block;
	IRT_RMRM	 <									IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_WIDTH,		e_irt_block_mesh > *IRT_MMRM_block;
	IRT_8x16_IWC < bus64B_struct, bus16ui64_struct, IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_MESH_MEM_BANK_WIDTH,		e_irt_block_mesh > *IRT_MIWC_block;
	IRT_2x2_sel	 < bus16ui64_struct, p2x2_ui64_struct, IRT_MESH_MEM_ROW_BANKS,														e_irt_block_mesh > *IRT_M2x2_block;
	IRT_BLI		 < p2x2_fp32_struct, float,																							e_irt_block_mesh > *IRT_MBLI_block;

	IRT_OICC2	 <																													e_irt_block_rot > *IRT_OICC2_block;
	IRT_IIIRC2	 <																													e_irt_block_rot > *IRT_IIRC2_block;
	IRT_8x16_IWC < bus16B_struct, bus16ui16_struct, IRT_MESH_MEM_ROW_BANKS, IRT_MESH_MEM_COL_BANKS, IRT_ROT_MEM_BANK_WIDTH,			e_irt_block_rot > *IRT_IIWC2_block;
	IRT_2x2_sel  < bus16ui16_struct, p2x2_ui16_struct, IRT_MESH_MEM_ROW_BANKS,														e_irt_block_rot > *IRT_I2x22_block;

	IRT_OICC2	 <																													e_irt_block_mesh > *IRT_MICC2_block;

};

#endif /* IRT_H_ */
