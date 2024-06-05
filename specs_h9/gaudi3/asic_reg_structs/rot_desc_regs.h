/***********************************
** This is an auto-generated file **
**       DO NOT EDIT BELOW        **
************************************/

#ifndef ASIC_REG_STRUCTS_GAUDI3_ROT_DESC_H_
#define ASIC_REG_STRUCTS_GAUDI3_ROT_DESC_H_

#include <stdint.h>
#include "gaudi3_types.h"

#pragma pack(push, 1)

#ifdef __cplusplus
namespace gaudi3 {
namespace rot_desc {
#else
#	ifndef static_assert
#		if defined( __STDC__ ) && defined( __STDC_VERSION__ ) && __STDC_VERSION__ >= 201112L
#			define static_assert(...) _Static_assert(__VA_ARGS__)
#		else
#			define static_assert(...)
#		endif
#	endif
#endif

/*
 CONTEXT_ID 
*/
typedef struct reg_context_id {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_context_id;
static_assert((sizeof(struct reg_context_id) == 4), "reg_context_id size is not 32-bit");
/*
 IN_IMG_START_ADDR_L 
*/
typedef struct reg_in_img_start_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_in_img_start_addr_l;
static_assert((sizeof(struct reg_in_img_start_addr_l) == 4), "reg_in_img_start_addr_l size is not 32-bit");
/*
 IN_IMG_START_ADDR_H 
*/
typedef struct reg_in_img_start_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_in_img_start_addr_h;
static_assert((sizeof(struct reg_in_img_start_addr_h) == 4), "reg_in_img_start_addr_h size is not 32-bit");
/*
 OUT_IMG_START_ADDR_L 
*/
typedef struct reg_out_img_start_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_out_img_start_addr_l;
static_assert((sizeof(struct reg_out_img_start_addr_l) == 4), "reg_out_img_start_addr_l size is not 32-bit");
/*
 OUT_IMG_START_ADDR_H 
*/
typedef struct reg_out_img_start_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_out_img_start_addr_h;
static_assert((sizeof(struct reg_out_img_start_addr_h) == 4), "reg_out_img_start_addr_h size is not 32-bit");
/*
 CFG 
*/
typedef struct reg_cfg {
	union {
		struct {
			uint32_t rot_dir : 1,
				read_flip : 1,
				rot_90 : 1,
				out_image_line_wr_format : 1,
				v_read_flip : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_cfg;
static_assert((sizeof(struct reg_cfg) == 4), "reg_cfg size is not 32-bit");
/*
 IM_READ_SLOPE 
 b'input image read slope'
*/
typedef struct reg_im_read_slope {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_im_read_slope;
static_assert((sizeof(struct reg_im_read_slope) == 4), "reg_im_read_slope size is not 32-bit");
/*
 SIN_D 
 b'sin of rotation degree D'
*/
typedef struct reg_sin_d {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_sin_d;
static_assert((sizeof(struct reg_sin_d) == 4), "reg_sin_d size is not 32-bit");
/*
 COS_D 
 b'cos of rotation degree D'
*/
typedef struct reg_cos_d {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cos_d;
static_assert((sizeof(struct reg_cos_d) == 4), "reg_cos_d size is not 32-bit");
/*
 IN_IMG 
 b'Input image dimensions'
*/
typedef struct reg_in_img {
	union {
		struct {
			uint32_t width : 16,
				height : 16;
		};
		uint32_t _raw;
	};
} reg_in_img;
static_assert((sizeof(struct reg_in_img) == 4), "reg_in_img size is not 32-bit");
/*
 IN_STRIDE 
*/
typedef struct reg_in_stride {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_in_stride;
static_assert((sizeof(struct reg_in_stride) == 4), "reg_in_stride size is not 32-bit");
/*
 IN_STRIPE 
*/
typedef struct reg_in_stripe {
	union {
		struct {
			uint32_t width : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_in_stripe;
static_assert((sizeof(struct reg_in_stripe) == 4), "reg_in_stripe size is not 32-bit");
/*
 IN_CENTER 
 b'Input image center coordinate'
*/
typedef struct reg_in_center {
	union {
		struct {
			uint32_t x : 16,
				y : 16;
		};
		uint32_t _raw;
	};
} reg_in_center;
static_assert((sizeof(struct reg_in_center) == 4), "reg_in_center size is not 32-bit");
/*
 OUT_IMG 
 b'Output image dimensions'
*/
typedef struct reg_out_img {
	union {
		struct {
			uint32_t height : 16,
				width : 16;
		};
		uint32_t _raw;
	};
} reg_out_img;
static_assert((sizeof(struct reg_out_img) == 4), "reg_out_img size is not 32-bit");
/*
 OUT_STRIDE 
*/
typedef struct reg_out_stride {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_out_stride;
static_assert((sizeof(struct reg_out_stride) == 4), "reg_out_stride size is not 32-bit");
/*
 OUT_STRIPE 
*/
typedef struct reg_out_stripe {
	union {
		struct {
			uint32_t width : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_out_stripe;
static_assert((sizeof(struct reg_out_stripe) == 4), "reg_out_stripe size is not 32-bit");
/*
 OUT_CENTER 
 b'Output image center coordinate'
*/
typedef struct reg_out_center {
	union {
		struct {
			uint32_t x : 16,
				y : 16;
		};
		uint32_t _raw;
	};
} reg_out_center;
static_assert((sizeof(struct reg_out_center) == 4), "reg_out_center size is not 32-bit");
/*
 BACKGROUND 
*/
typedef struct reg_background {
	union {
		struct {
			uint32_t pxl_val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_background;
static_assert((sizeof(struct reg_background) == 4), "reg_background size is not 32-bit");
/*
 CPL_MSG_EN 
*/
typedef struct reg_cpl_msg_en {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_en;
static_assert((sizeof(struct reg_cpl_msg_en) == 4), "reg_cpl_msg_en size is not 32-bit");
/*
 IDLE_STATE 
*/
typedef struct reg_idle_state {
	union {
		struct {
			uint32_t start_en : 1,
				end_en : 1,
				payload : 8,
				_reserved10 : 22;
		};
		uint32_t _raw;
	};
} reg_idle_state;
static_assert((sizeof(struct reg_idle_state) == 4), "reg_idle_state size is not 32-bit");
/*
 CPL_MSG_ADDR 
*/
typedef struct reg_cpl_msg_addr {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_addr;
static_assert((sizeof(struct reg_cpl_msg_addr) == 4), "reg_cpl_msg_addr size is not 32-bit");
/*
 CPL_MSG_DATA 
*/
typedef struct reg_cpl_msg_data {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_cpl_msg_data;
static_assert((sizeof(struct reg_cpl_msg_data) == 4), "reg_cpl_msg_data size is not 32-bit");
/*
 X_I_START_OFFSET 
 b'Offset used during Xi_start calculation for input'
*/
typedef struct reg_x_i_start_offset {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_x_i_start_offset;
static_assert((sizeof(struct reg_x_i_start_offset) == 4), "reg_x_i_start_offset size is not 32-bit");
/*
 X_I_START_OFFSET_FLIP 
 b'Additional offset used during Xi_start calculation'
*/
typedef struct reg_x_i_start_offset_flip {
	union {
		struct {
			uint32_t val : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_x_i_start_offset_flip;
static_assert((sizeof(struct reg_x_i_start_offset_flip) == 4), "reg_x_i_start_offset_flip size is not 32-bit");
/*
 X_I_FIRST 
*/
typedef struct reg_x_i_first {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_x_i_first;
static_assert((sizeof(struct reg_x_i_first) == 4), "reg_x_i_first size is not 32-bit");
/*
 Y_I_FIRST 
*/
typedef struct reg_y_i_first {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_y_i_first;
static_assert((sizeof(struct reg_y_i_first) == 4), "reg_y_i_first size is not 32-bit");
/*
 Y_I 
*/
typedef struct reg_y_i {
	union {
		struct {
			uint32_t end_val : 16,
				start_val : 16;
		};
		uint32_t _raw;
	};
} reg_y_i;
static_assert((sizeof(struct reg_y_i) == 4), "reg_y_i size is not 32-bit");
/*
 OUT_STRIPE_SIZE 
 b'Output stripe size (Stripe width * H)'
*/
typedef struct reg_out_stripe_size {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_out_stripe_size;
static_assert((sizeof(struct reg_out_stripe_size) == 4), "reg_out_stripe_size size is not 32-bit");
/*
 RSB_CFG_0 
*/
typedef struct reg_rsb_cfg_0 {
	union {
		struct {
			uint32_t cache_inv : 1,
				uncacheable : 1,
				perf_evt_start : 1,
				perf_evt_end : 1,
				pad_duplicate : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_rsb_cfg_0;
static_assert((sizeof(struct reg_rsb_cfg_0) == 4), "reg_rsb_cfg_0 size is not 32-bit");
/*
 RSB_PAD_VAL 
*/
typedef struct reg_rsb_pad_val {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_rsb_pad_val;
static_assert((sizeof(struct reg_rsb_pad_val) == 4), "reg_rsb_pad_val size is not 32-bit");
/*
 OWM_CFG 
*/
typedef struct reg_owm_cfg {
	union {
		struct {
			uint32_t perf_evt_start : 1,
				perf_evt_end : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_owm_cfg;
static_assert((sizeof(struct reg_owm_cfg) == 4), "reg_owm_cfg size is not 32-bit");
/*
 CTRL_CFG 
 b'Control cfg for int_md, irt_md, pwi, pwo...'
*/
typedef struct reg_ctrl_cfg {
	union {
		struct {
			uint32_t int_mode : 1,
				bg_mode : 2,
				_reserved4 : 1,
				irt_mode : 3,
				_reserved8 : 1,
				img_pwi : 3,
				_reserved12 : 1,
				img_pwo : 3,
				_reserved16 : 1,
				buf_cfg_mode : 3,
				buf_fmt_mode : 1,
				rate_mode : 1,
				_reserved24 : 3,
				proc_size : 4,
				coord_calc_data_type : 3,
				_reserved31 : 1;
		};
		uint32_t _raw;
	};
} reg_ctrl_cfg;
static_assert((sizeof(struct reg_ctrl_cfg) == 4), "reg_ctrl_cfg size is not 32-bit");
/*
 PIXEL_PAD 
*/
typedef struct reg_pixel_pad {
	union {
		struct {
			uint32_t mask_pwi : 16,
				lsb_o : 3,
				_reserved19 : 13;
		};
		uint32_t _raw;
	};
} reg_pixel_pad;
static_assert((sizeof(struct reg_pixel_pad) == 4), "reg_pixel_pad size is not 32-bit");
/*
 PREC_SHIFT 
*/
typedef struct reg_prec_shift {
	union {
		struct {
			uint32_t bli : 8,
				coord_align : 4,
				_reserved12 : 20;
		};
		uint32_t _raw;
	};
} reg_prec_shift;
static_assert((sizeof(struct reg_prec_shift) == 4), "reg_prec_shift size is not 32-bit");
/*
 MAX_VAL 
*/
typedef struct reg_max_val {
	union {
		struct {
			uint32_t sat : 16,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_max_val;
static_assert((sizeof(struct reg_max_val) == 4), "reg_max_val size is not 32-bit");
/*
 A0_M11 
 b'projection coefficients FP32/Fixed32'
*/
typedef struct reg_a0_m11 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_a0_m11;
static_assert((sizeof(struct reg_a0_m11) == 4), "reg_a0_m11 size is not 32-bit");
/*
 A1_M12 
 b'projection coefficients FP32/Fixed32'
*/
typedef struct reg_a1_m12 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_a1_m12;
static_assert((sizeof(struct reg_a1_m12) == 4), "reg_a1_m12 size is not 32-bit");
/*
 A2 
 b'projection coefficients FP32'
*/
typedef struct reg_a2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_a2;
static_assert((sizeof(struct reg_a2) == 4), "reg_a2 size is not 32-bit");
/*
 B0_M21 
 b'projection coefficients FP32/Fixed32'
*/
typedef struct reg_b0_m21 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_b0_m21;
static_assert((sizeof(struct reg_b0_m21) == 4), "reg_b0_m21 size is not 32-bit");
/*
 B1_M22 
 b'projection coefficients FP32/Fixed32'
*/
typedef struct reg_b1_m22 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_b1_m22;
static_assert((sizeof(struct reg_b1_m22) == 4), "reg_b1_m22 size is not 32-bit");
/*
 B2 
 b'projection coefficients FP32'
*/
typedef struct reg_b2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_b2;
static_assert((sizeof(struct reg_b2) == 4), "reg_b2 size is not 32-bit");
/*
 C0 
 b'projection coefficients FP32'
*/
typedef struct reg_c0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_c0;
static_assert((sizeof(struct reg_c0) == 4), "reg_c0 size is not 32-bit");
/*
 C1 
 b'projection coefficients FP32'
*/
typedef struct reg_c1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_c1;
static_assert((sizeof(struct reg_c1) == 4), "reg_c1 size is not 32-bit");
/*
 C2 
 b'projection coefficients FP32'
*/
typedef struct reg_c2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_c2;
static_assert((sizeof(struct reg_c2) == 4), "reg_c2 size is not 32-bit");
/*
 D0 
 b'projection coefficients FP32'
*/
typedef struct reg_d0 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d0;
static_assert((sizeof(struct reg_d0) == 4), "reg_d0 size is not 32-bit");
/*
 D1 
 b'projection coefficients FP32'
*/
typedef struct reg_d1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d1;
static_assert((sizeof(struct reg_d1) == 4), "reg_d1 size is not 32-bit");
/*
 D2 
 b'projection coefficients FP32'
*/
typedef struct reg_d2 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_d2;
static_assert((sizeof(struct reg_d2) == 4), "reg_d2 size is not 32-bit");
/*
 INV_PROC_SIZE_M_1 
 b'1/(pro_size-1) in fp32 fmt'
*/
typedef struct reg_inv_proc_size_m_1 {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_inv_proc_size_m_1;
static_assert((sizeof(struct reg_inv_proc_size_m_1) == 4), "reg_inv_proc_size_m_1 size is not 32-bit");
/*
 MESH_IMG_START_ADDR_L 
 b'Mesh start address low'
*/
typedef struct reg_mesh_img_start_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mesh_img_start_addr_l;
static_assert((sizeof(struct reg_mesh_img_start_addr_l) == 4), "reg_mesh_img_start_addr_l size is not 32-bit");
/*
 MESH_IMG_START_ADDR_H 
 b'Mesh start address high'
*/
typedef struct reg_mesh_img_start_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mesh_img_start_addr_h;
static_assert((sizeof(struct reg_mesh_img_start_addr_h) == 4), "reg_mesh_img_start_addr_h size is not 32-bit");
/*
 MESH_IMG 
 b'Mesh image dimensions'
*/
typedef struct reg_mesh_img {
	union {
		struct {
			uint32_t height : 16,
				width : 16;
		};
		uint32_t _raw;
	};
} reg_mesh_img;
static_assert((sizeof(struct reg_mesh_img) == 4), "reg_mesh_img size is not 32-bit");
/*
 MESH_STRIDE 
 b'Mesh Horizontal Stride HSm'
*/
typedef struct reg_mesh_stride {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mesh_stride;
static_assert((sizeof(struct reg_mesh_stride) == 4), "reg_mesh_stride size is not 32-bit");
/*
 MESH_STRIPE 
 b'Mesh Horizontal Stripe widthHo'
*/
typedef struct reg_mesh_stripe {
	union {
		struct {
			uint32_t width : 16,
				stripe_pitch : 16;
		};
		uint32_t _raw;
	};
} reg_mesh_stripe;
static_assert((sizeof(struct reg_mesh_stripe) == 4), "reg_mesh_stripe size is not 32-bit");
/*
 MESH_CTRL 
 b'Mesh control reg: sparse, relative, frac width...'
*/
typedef struct reg_mesh_ctrl {
	union {
		struct {
			uint32_t data_type : 3,
				_reserved4 : 1,
				rel_mode : 1,
				_reserved8 : 3,
				sparse_mode : 2,
				_reserved12 : 2,
				fxd_pt_frac_w : 4,
				_reserved16 : 16;
		};
		uint32_t _raw;
	};
} reg_mesh_ctrl;
static_assert((sizeof(struct reg_mesh_ctrl) == 4), "reg_mesh_ctrl size is not 32-bit");
/*
 MESH_GH 
 b'Mesh Horizontal Granularity'
*/
typedef struct reg_mesh_gh {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mesh_gh;
static_assert((sizeof(struct reg_mesh_gh) == 4), "reg_mesh_gh size is not 32-bit");
/*
 MESH_GV 
 b'Mesh Vertical Granularity'
*/
typedef struct reg_mesh_gv {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mesh_gv;
static_assert((sizeof(struct reg_mesh_gv) == 4), "reg_mesh_gv size is not 32-bit");
/*
 MRSB_CFG_0 
 b'Mesh SB Config'
*/
typedef struct reg_mrsb_cfg_0 {
	union {
		struct {
			uint32_t cache_inv : 1,
				uncacheable : 1,
				perf_evt_start : 1,
				perf_evt_end : 1,
				pad_duplicate : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_mrsb_cfg_0;
static_assert((sizeof(struct reg_mrsb_cfg_0) == 4), "reg_mrsb_cfg_0 size is not 32-bit");
/*
 MRSB_PAD_VAL 
 b'Mesh PAD Value'
*/
typedef struct reg_mrsb_pad_val {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_mrsb_pad_val;
static_assert((sizeof(struct reg_mrsb_pad_val) == 4), "reg_mrsb_pad_val size is not 32-bit");
/*
 BUF_CFG 
*/
typedef struct reg_buf_cfg {
	union {
		struct {
			uint32_t rot_epl : 8,
				mesh_epl : 8,
				mesh_mode : 3,
				_reserved20 : 1,
				mesh_fmt : 1,
				_reserved21 : 11;
		};
		uint32_t _raw;
	};
} reg_buf_cfg;
static_assert((sizeof(struct reg_buf_cfg) == 4), "reg_buf_cfg size is not 32-bit");
/*
 CID_OFFSET 
 b'Context ID offset to increment/decrement: signed'
*/
typedef struct reg_cid_offset {
	union {
		struct {
			uint32_t val : 8,
				_reserved8 : 24;
		};
		uint32_t _raw;
	};
} reg_cid_offset;
static_assert((sizeof(struct reg_cid_offset) == 4), "reg_cid_offset size is not 32-bit");
/*
 GRSB_CFG_0 
 b'Grad output SB Config'
*/
typedef struct reg_grsb_cfg_0 {
	union {
		struct {
			uint32_t cache_inv : 1,
				uncacheable : 1,
				perf_evt_start : 1,
				perf_evt_end : 1,
				pad_duplicate : 1,
				_reserved5 : 27;
		};
		uint32_t _raw;
	};
} reg_grsb_cfg_0;
static_assert((sizeof(struct reg_grsb_cfg_0) == 4), "reg_grsb_cfg_0 size is not 32-bit");
/*
 GRSB_PAD_VAL 
 b'Grad output PAD Value'
*/
typedef struct reg_grsb_pad_val {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_grsb_pad_val;
static_assert((sizeof(struct reg_grsb_pad_val) == 4), "reg_grsb_pad_val size is not 32-bit");
/*
 RESCALE_CFG 
 b'Rescale configuration fields'
*/
typedef struct reg_rescale_cfg {
	union {
		struct {
			uint32_t sym_filter_en : 1,
				retain_asp_ratio : 1,
				filter_type : 2,
				num_taps_h : 5,
				num_taps_v : 5,
				phase_prec_shift_v : 5,
				phase_prec_shift_h : 5,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_rescale_cfg;
static_assert((sizeof(struct reg_rescale_cfg) == 4), "reg_rescale_cfg size is not 32-bit");
/*
 GRAD_IMG_START_ADDR_L 
 b'Grad start address low'
*/
typedef struct reg_grad_img_start_addr_l {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_grad_img_start_addr_l;
static_assert((sizeof(struct reg_grad_img_start_addr_l) == 4), "reg_grad_img_start_addr_l size is not 32-bit");
/*
 GRAD_IMG_START_ADDR_H 
 b'Grad start address high'
*/
typedef struct reg_grad_img_start_addr_h {
	union {
		struct {
			uint32_t val : 32;
		};
		uint32_t _raw;
	};
} reg_grad_img_start_addr_h;
static_assert((sizeof(struct reg_grad_img_start_addr_h) == 4), "reg_grad_img_start_addr_h size is not 32-bit");
/*
 GRAD_STRIDE 
 b'Grad Horizontal stride'
*/
typedef struct reg_grad_stride {
	union {
		struct {
			uint32_t grad_image_stride : 32;
		};
		uint32_t _raw;
	};
} reg_grad_stride;
static_assert((sizeof(struct reg_grad_stride) == 4), "reg_grad_stride size is not 32-bit");
/*
 GRAD_CTRL 
 b'Grad control register'
*/
typedef struct reg_grad_ctrl {
	union {
		struct {
			uint32_t datatype : 3,
				_reserved4 : 1,
				shfl_en : 1,
				_reserved8 : 3,
				shfl_stride : 1,
				_reserved12 : 3,
				bligrad_en : 1,
				_reserved13 : 19;
		};
		uint32_t _raw;
	};
} reg_grad_ctrl;
static_assert((sizeof(struct reg_grad_ctrl) == 4), "reg_grad_ctrl size is not 32-bit");
/*
 ROT_IRM_RL 
 b'ROT IRM rate limiter en, satuarion and timeout'
*/
typedef struct reg_rot_irm_rl {
	union {
		struct {
			uint32_t timeout : 8,
				saturation : 8,
				en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_rot_irm_rl;
static_assert((sizeof(struct reg_rot_irm_rl) == 4), "reg_rot_irm_rl size is not 32-bit");
/*
 MESH_IRM_RL 
 b'ROT IRM rate limiter en, satuarion and timeout'
*/
typedef struct reg_mesh_irm_rl {
	union {
		struct {
			uint32_t timeout : 8,
				saturation : 8,
				en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_mesh_irm_rl;
static_assert((sizeof(struct reg_mesh_irm_rl) == 4), "reg_mesh_irm_rl size is not 32-bit");
/*
 GRAD_IRM_RL 
 b'ROT IRM rate limiter en, satuarion and timeout'
*/
typedef struct reg_grad_irm_rl {
	union {
		struct {
			uint32_t timeout : 8,
				saturation : 8,
				en : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_grad_irm_rl;
static_assert((sizeof(struct reg_grad_irm_rl) == 4), "reg_grad_irm_rl size is not 32-bit");
/*
 WCH_CTRL 
 b'WCH Control'
*/
typedef struct reg_wch_ctrl {
	union {
		struct {
			uint32_t wsb_src_flush : 1,
				wsb_src_uncachable : 1,
				_reserved2 : 30;
		};
		uint32_t _raw;
	};
} reg_wch_ctrl;
static_assert((sizeof(struct reg_wch_ctrl) == 4), "reg_wch_ctrl size is not 32-bit");
/*
 ROT_NULL_CMD 
 b'Null command for rotator'
*/
typedef struct reg_rot_null_cmd {
	union {
		struct {
			uint32_t val : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_rot_null_cmd;
static_assert((sizeof(struct reg_rot_null_cmd) == 4), "reg_rot_null_cmd size is not 32-bit");
/*
 IN_IMG_AXI_CFG 
 b'In Image AXI CFG'
*/
typedef struct reg_in_img_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				arcache : 4,
				src_mode : 2,
				class_rd : 2,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_in_img_axi_cfg;
static_assert((sizeof(struct reg_in_img_axi_cfg) == 4), "reg_in_img_axi_cfg size is not 32-bit");
/*
 OUT_IMG_AXI_CFG 
 b'Out Image AXI CFG'
*/
typedef struct reg_out_img_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				awcache : 4,
				class_wr : 2,
				_reserved22 : 10;
		};
		uint32_t _raw;
	};
} reg_out_img_axi_cfg;
static_assert((sizeof(struct reg_out_img_axi_cfg) == 4), "reg_out_img_axi_cfg size is not 32-bit");
/*
 OUT_IMG_AXI_CFG_REDUCTION 
 b'Out Image AXI Reduction CFG'
*/
typedef struct reg_out_img_axi_cfg_reduction {
	union {
		struct {
			uint32_t reduction_ind_wr : 1,
				_reserved4 : 3,
				operation_wr : 3,
				_reserved8 : 1,
				round_mode_wr : 2,
				_reserved12 : 2,
				data_type_wr : 4,
				clip_wr : 1,
				_reserved17 : 15;
		};
		uint32_t _raw;
	};
} reg_out_img_axi_cfg_reduction;
static_assert((sizeof(struct reg_out_img_axi_cfg_reduction) == 4), "reg_out_img_axi_cfg_reduction size is not 32-bit");
/*
 MESH_AXI_CFG 
 b'Mesh AXI CFG'
*/
typedef struct reg_mesh_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				arcache : 4,
				src_mode : 2,
				class_rd : 2,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_mesh_axi_cfg;
static_assert((sizeof(struct reg_mesh_axi_cfg) == 4), "reg_mesh_axi_cfg size is not 32-bit");
/*
 GRAD_AXI_CFG 
 b'Grad AXI CFG'
*/
typedef struct reg_grad_axi_cfg {
	union {
		struct {
			uint32_t mcid : 16,
				arcache : 4,
				src_mode : 2,
				class_rd : 2,
				_reserved24 : 8;
		};
		uint32_t _raw;
	};
} reg_grad_axi_cfg;
static_assert((sizeof(struct reg_grad_axi_cfg) == 4), "reg_grad_axi_cfg size is not 32-bit");
/*
 PUSH_DESC 
 b'push desc'
*/
typedef struct reg_push_desc {
	union {
		struct {
			uint32_t ind : 1,
				_reserved1 : 31;
		};
		uint32_t _raw;
	};
} reg_push_desc;
static_assert((sizeof(struct reg_push_desc) == 4), "reg_push_desc size is not 32-bit");

#ifdef __cplusplus
} /* rot_desc namespace */
#endif

/*
 ROT_DESC block
*/

#ifdef __cplusplus

struct block_rot_desc {
	struct rot_desc::reg_context_id context_id;
	struct rot_desc::reg_in_img_start_addr_l in_img_start_addr_l;
	struct rot_desc::reg_in_img_start_addr_h in_img_start_addr_h;
	struct rot_desc::reg_out_img_start_addr_l out_img_start_addr_l;
	struct rot_desc::reg_out_img_start_addr_h out_img_start_addr_h;
	struct rot_desc::reg_cfg cfg;
	struct rot_desc::reg_im_read_slope im_read_slope;
	struct rot_desc::reg_sin_d sin_d;
	struct rot_desc::reg_cos_d cos_d;
	struct rot_desc::reg_in_img in_img;
	struct rot_desc::reg_in_stride in_stride;
	struct rot_desc::reg_in_stripe in_stripe;
	struct rot_desc::reg_in_center in_center;
	struct rot_desc::reg_out_img out_img;
	struct rot_desc::reg_out_stride out_stride;
	struct rot_desc::reg_out_stripe out_stripe;
	struct rot_desc::reg_out_center out_center;
	struct rot_desc::reg_background background;
	struct rot_desc::reg_cpl_msg_en cpl_msg_en;
	struct rot_desc::reg_idle_state idle_state;
	struct rot_desc::reg_cpl_msg_addr cpl_msg_addr;
	struct rot_desc::reg_cpl_msg_data cpl_msg_data;
	uint32_t _pad88[1];
	struct rot_desc::reg_x_i_start_offset x_i_start_offset;
	struct rot_desc::reg_x_i_start_offset_flip x_i_start_offset_flip;
	struct rot_desc::reg_x_i_first x_i_first;
	struct rot_desc::reg_y_i_first y_i_first;
	struct rot_desc::reg_y_i y_i;
	struct rot_desc::reg_out_stripe_size out_stripe_size;
	struct rot_desc::reg_rsb_cfg_0 rsb_cfg_0;
	struct rot_desc::reg_rsb_pad_val rsb_pad_val;
	uint32_t _pad124[4];
	struct rot_desc::reg_owm_cfg owm_cfg;
	struct rot_desc::reg_ctrl_cfg ctrl_cfg;
	struct rot_desc::reg_pixel_pad pixel_pad;
	struct rot_desc::reg_prec_shift prec_shift;
	struct rot_desc::reg_max_val max_val;
	struct rot_desc::reg_a0_m11 a0_m11;
	struct rot_desc::reg_a1_m12 a1_m12;
	struct rot_desc::reg_a2 a2;
	struct rot_desc::reg_b0_m21 b0_m21;
	struct rot_desc::reg_b1_m22 b1_m22;
	struct rot_desc::reg_b2 b2;
	struct rot_desc::reg_c0 c0;
	struct rot_desc::reg_c1 c1;
	struct rot_desc::reg_c2 c2;
	struct rot_desc::reg_d0 d0;
	struct rot_desc::reg_d1 d1;
	struct rot_desc::reg_d2 d2;
	struct rot_desc::reg_inv_proc_size_m_1 inv_proc_size_m_1;
	struct rot_desc::reg_mesh_img_start_addr_l mesh_img_start_addr_l;
	struct rot_desc::reg_mesh_img_start_addr_h mesh_img_start_addr_h;
	struct rot_desc::reg_mesh_img mesh_img;
	struct rot_desc::reg_mesh_stride mesh_stride;
	struct rot_desc::reg_mesh_stripe mesh_stripe;
	struct rot_desc::reg_mesh_ctrl mesh_ctrl;
	struct rot_desc::reg_mesh_gh mesh_gh;
	struct rot_desc::reg_mesh_gv mesh_gv;
	struct rot_desc::reg_mrsb_cfg_0 mrsb_cfg_0;
	struct rot_desc::reg_mrsb_pad_val mrsb_pad_val;
	struct rot_desc::reg_buf_cfg buf_cfg;
	struct rot_desc::reg_cid_offset cid_offset;
	struct rot_desc::reg_grsb_cfg_0 grsb_cfg_0;
	struct rot_desc::reg_grsb_pad_val grsb_pad_val;
	uint32_t _pad268[1];
	struct rot_desc::reg_rescale_cfg rescale_cfg;
	struct rot_desc::reg_grad_img_start_addr_l grad_img_start_addr_l;
	struct rot_desc::reg_grad_img_start_addr_h grad_img_start_addr_h;
	uint32_t _pad284[1];
	struct rot_desc::reg_grad_stride grad_stride;
	uint32_t _pad292[1];
	struct rot_desc::reg_grad_ctrl grad_ctrl;
	uint32_t _pad300[1];
	struct rot_desc::reg_rot_irm_rl rot_irm_rl;
	struct rot_desc::reg_mesh_irm_rl mesh_irm_rl;
	uint32_t _pad312[1];
	struct rot_desc::reg_grad_irm_rl grad_irm_rl;
	struct rot_desc::reg_wch_ctrl wch_ctrl;
	struct rot_desc::reg_rot_null_cmd rot_null_cmd;
	struct rot_desc::reg_in_img_axi_cfg in_img_axi_cfg;
	struct rot_desc::reg_out_img_axi_cfg out_img_axi_cfg;
	struct rot_desc::reg_out_img_axi_cfg_reduction out_img_axi_cfg_reduction;
	struct rot_desc::reg_mesh_axi_cfg mesh_axi_cfg;
	struct rot_desc::reg_grad_axi_cfg grad_axi_cfg;
	uint32_t _pad348[16];
	struct rot_desc::reg_push_desc push_desc;
};
#else

typedef struct block_rot_desc {
	reg_context_id context_id;
	reg_in_img_start_addr_l in_img_start_addr_l;
	reg_in_img_start_addr_h in_img_start_addr_h;
	reg_out_img_start_addr_l out_img_start_addr_l;
	reg_out_img_start_addr_h out_img_start_addr_h;
	reg_cfg cfg;
	reg_im_read_slope im_read_slope;
	reg_sin_d sin_d;
	reg_cos_d cos_d;
	reg_in_img in_img;
	reg_in_stride in_stride;
	reg_in_stripe in_stripe;
	reg_in_center in_center;
	reg_out_img out_img;
	reg_out_stride out_stride;
	reg_out_stripe out_stripe;
	reg_out_center out_center;
	reg_background background;
	reg_cpl_msg_en cpl_msg_en;
	reg_idle_state idle_state;
	reg_cpl_msg_addr cpl_msg_addr;
	reg_cpl_msg_data cpl_msg_data;
	uint32_t _pad88[1];
	reg_x_i_start_offset x_i_start_offset;
	reg_x_i_start_offset_flip x_i_start_offset_flip;
	reg_x_i_first x_i_first;
	reg_y_i_first y_i_first;
	reg_y_i y_i;
	reg_out_stripe_size out_stripe_size;
	reg_rsb_cfg_0 rsb_cfg_0;
	reg_rsb_pad_val rsb_pad_val;
	uint32_t _pad124[4];
	reg_owm_cfg owm_cfg;
	reg_ctrl_cfg ctrl_cfg;
	reg_pixel_pad pixel_pad;
	reg_prec_shift prec_shift;
	reg_max_val max_val;
	reg_a0_m11 a0_m11;
	reg_a1_m12 a1_m12;
	reg_a2 a2;
	reg_b0_m21 b0_m21;
	reg_b1_m22 b1_m22;
	reg_b2 b2;
	reg_c0 c0;
	reg_c1 c1;
	reg_c2 c2;
	reg_d0 d0;
	reg_d1 d1;
	reg_d2 d2;
	reg_inv_proc_size_m_1 inv_proc_size_m_1;
	reg_mesh_img_start_addr_l mesh_img_start_addr_l;
	reg_mesh_img_start_addr_h mesh_img_start_addr_h;
	reg_mesh_img mesh_img;
	reg_mesh_stride mesh_stride;
	reg_mesh_stripe mesh_stripe;
	reg_mesh_ctrl mesh_ctrl;
	reg_mesh_gh mesh_gh;
	reg_mesh_gv mesh_gv;
	reg_mrsb_cfg_0 mrsb_cfg_0;
	reg_mrsb_pad_val mrsb_pad_val;
	reg_buf_cfg buf_cfg;
	reg_cid_offset cid_offset;
	reg_grsb_cfg_0 grsb_cfg_0;
	reg_grsb_pad_val grsb_pad_val;
	uint32_t _pad268[1];
	reg_rescale_cfg rescale_cfg;
	reg_grad_img_start_addr_l grad_img_start_addr_l;
	reg_grad_img_start_addr_h grad_img_start_addr_h;
	uint32_t _pad284[1];
	reg_grad_stride grad_stride;
	uint32_t _pad292[1];
	reg_grad_ctrl grad_ctrl;
	uint32_t _pad300[1];
	reg_rot_irm_rl rot_irm_rl;
	reg_mesh_irm_rl mesh_irm_rl;
	uint32_t _pad312[1];
	reg_grad_irm_rl grad_irm_rl;
	reg_wch_ctrl wch_ctrl;
	reg_rot_null_cmd rot_null_cmd;
	reg_in_img_axi_cfg in_img_axi_cfg;
	reg_out_img_axi_cfg out_img_axi_cfg;
	reg_out_img_axi_cfg_reduction out_img_axi_cfg_reduction;
	reg_mesh_axi_cfg mesh_axi_cfg;
	reg_grad_axi_cfg grad_axi_cfg;
	uint32_t _pad348[16];
	reg_push_desc push_desc;
} block_rot_desc;
#endif

#ifndef DONT_INCLUDE_OFFSET_VAL_CONST
const offsetVal block_rot_desc_defaults[] =
{
	// offset	// value
	{ 0x90  , 0x8030000           , 1 }, // ctrl_cfg
	{ 0x94  , 0xffff              , 1 }, // pixel_pad
	{ 0x98  , 0xf00               , 1 }, // prec_shift
	{ 0x9c  , 0xffff              , 1 }, // max_val
	{ 0xfc  , 0x30808             , 1 }, // buf_cfg
};
#endif

#ifdef __cplusplus
} /* gaudi3 namespace */
#endif

#pragma pack(pop)
#endif /* ASIC_REG_STRUCTS_GAUDI3_ROT_DESC_H_ */
