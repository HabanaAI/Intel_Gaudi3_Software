/*
 * IRThl.cpp
 *
 *  Created on: Jun 11, 2019
 *      Author: espektor
 */

#define _USE_MATH_DEFINES
//#define IRT_HL_NO_FLIP

#include "stdio.h"
#include "stdlib.h"
#include <algorithm>
#include <math.h>
#include <cmath>
#include "IRT.h"
#include "IRTutils.h"
//#include "IRTsim.h"

extern FILE* log_file;
extern uint8_t hl_only;
extern bool print_out_files;
extern FILE* test_res;
extern bool test_file_flag;
extern Eirt_bmp_order irt_bmp_wr_order;
extern uint8_t num_of_images;

extern uint8_t* ext_mem;
extern uint16_t*** input_image;
extern uint16_t**** output_image;

enum proj_err_type {
	proj_inc_Xo = 0,
	proj_inc_Xo3 = 1,
	proj_slope_O2I = 2,
	proj_slope_I2I = 3,
	proj_slope_S2I = 4,
};

void IRThl_advanced_transform(IRT_top* irt_top, uint8_t image, uint8_t planes, char input_file[150], char* outfilename, char* outfiledir, uint8_t transform_type) {

	uint8_t desc = image * planes;

	uint16_t Ho = irt_top->irt_desc[desc].image_par[OIMAGE].H;
	uint16_t So = irt_top->irt_desc[desc].image_par[OIMAGE].S;
	int16_t Xoc = irt_top->irt_desc[desc].image_par[OIMAGE].Xc;
	int16_t Yoc = irt_top->irt_desc[desc].image_par[OIMAGE].Yc;
	uint16_t Hi = irt_top->irt_desc[desc].image_par[IIMAGE].H;
	uint16_t Wi = irt_top->irt_desc[desc].image_par[IIMAGE].W;
	int16_t Xic = irt_top->irt_desc[desc].image_par[IIMAGE].Xc;
	int16_t Yic = irt_top->irt_desc[desc].image_par[IIMAGE].Yc;

	uint16_t Hm = irt_top->irt_desc[desc].image_par[MIMAGE].H;
	uint16_t Wm = irt_top->irt_desc[desc].image_par[MIMAGE].W;
	bool mesh_format = irt_top->irt_desc[desc].mesh_format;

	char out_file_name[150];
	int16_t xo_i = 0, yo_i = 0;
	uint32_t cycles = 0;
	uint8_t min_proc_rate = IRT_ROT_MAX_PROC_SIZE, max_proc_rate = 0;
	uint32_t acc_proc_rate = 0, rate_hist[IRT_ROT_MAX_PROC_SIZE + 1] = { 0 };
	uint8_t adj_proc_size = 0;
	double xo_d = 0, yo_d = 0, xi_fpt64 = 0, yi_fpt64 = 0, xf_fpt64 = 0, yf_fpt64 = 0, out_fpt64 = 0, xim_fpt64 = 0, yim_fpt64 = 0;
	double xis_fpt64 = 0, xie_fpt64 = 0, yis_fpt64 = 0, yie_fpt64 = 0, xi_slope_d = 0, yi_slope_d = 0;
	float xis_fpt32 = 0, xie_fpt32 = 0, yis_fpt32 = 0, yie_fpt32 = 0, xi_slope_f = 0, yi_slope_f = 0;
	float xi_fpt32 = 0, yi_fpt32 = 0, xo_f = 0, xo_arch = 0, yo_f = 0, xim_fpt32 = 0, yim_fpt32 = 0, xf_fpt32 = 0, yf_fpt32 = 0, xf_arch = 0, yf_arch = 0, out_fpt32 = 0, xi_farch = 0, yi_farch = 0;
	float xw0, xw1, yw0, yw1;
	int64_t xi_fixed = 0, yi_fixed = 0, xim_fixed = 0, yim_fixed = 0, xi_fix31 = 0, yi_fix31 = 0, xim_arch = 0, yim_arch = 0;
	uint64_t xf_fixed = 0, yf_fixed = 0;
	int16_t xi0[CALC_FORMATS] = { 0 }, yi0[CALC_FORMATS] = { 0 };
	int16_t i0[CALC_FORMATS] = { 0 }, i1[CALC_FORMATS] = { 0 }, j0[CALC_FORMATS] = { 0 }, j1[CALC_FORMATS] = { 0 };
	int32_t plane, out1[CALC_FORMATS] = { 0 }, remain_pixels = 0;
	int16_t XL_fixed = 0, XR_fixed = 0, YT_fixed = 0, YB_fixed = 0;
	int32_t xi_fix16 = 0, yi_fix16 = 0;
	uint32_t xf_fix16 = 0, yf_fix16 = 0, xf_fix31 = 0, yf_fix31 = 0;
	uint64_t out_fix16 = 0;
	bool oob_flag_h[CALC_FORMATS] = { 0 }, oob_flag_v[CALC_FORMATS] = { 0 }, noint_flag_h[CALC_FORMATS] = { 0 }, noint_flag_v[CALC_FORMATS] = { 0 };

	int16_t row_bank_access[IRT_ROT_MEM_ROW_BANKS][5] = { 0 };
	irt_rot_mem_rd_stat rot_mem_rd_stat;

	//int16_t Yi_start = (int16_t)(irt_top->irt_desc[desc].Yi_first_fixed >> IRT_SLOPE_PREC) - irt_top->irt_desc[desc].read_vflip;
	//uint16_t BufW_entries = irt_top->irt_cfg.rm_cfg[irt_rot][irt_top->irt_desc[desc].image_par[IIMAGE].buf_mode].Buf_EpL;// << irt_top->irt_desc[irt_task].image_par[IIMAGE].Ps;
	//uint8_t Ps1 = irt_top->irt_cfg.buf_1b_mode[irt_rot] == 1 || irt_top->irt_desc[desc].image_par[IIMAGE].Ps == 1;

	bool rate_reach_flag = 0;

	uint64_t out_fixed;
	//int64_t N, D;

	double p_fpt64[4];
	float p_fpt32[4];
	uint64_t p_fixed[4], w[4];
	uint64_t p_fix16[4], w16[4];
	bool p_valid[4], i_valid[4];
	uint16_t p_value[4];
	double mp_fpt64[2][4], mp_fixed[2][4];
	float mp_fpt32[2][4], mp_value[2][4], mp_arch[2][4];
	//double /*proj_coord_err[2] = { 0, 0 }, */ yi_fpt32_inc, xi_fpt32_inc;
	int proj_coord_err_hist[CALC_FORMATS][2][PROJ_ERR_BINS];

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < PROJ_ERR_BINS; j++) {
			proj_coord_err_hist[i][0][j] = 0;
			proj_coord_err_hist[i][1][j] = 0;
		}
	}

	for (int format = 0; format <= e_irt_crdf_fix16; format++) {
		coord_err[format] = 0;
	}

	if (irt_top->irt_desc[desc].rate_mode == e_irt_rate_fixed)
		adj_proc_size = irt_top->irt_desc[desc].proc_size; //constant will be used
	else
		adj_proc_size = IRT_ROT_MAX_PROC_SIZE; //set for 8 for initial try;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\nRunning %s high level model on image %d %s %dx%d with ", irt_irt_mode_s[irt_top->irt_desc[desc].irt_mode], image, input_file, So, Ho);

	if (irt_top->irt_desc[desc].irt_mode == e_irt_rotation ||
		irt_top->irt_desc[desc].irt_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_rotation ||
		irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_rotation)
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f rotation angle\n", irt_top->rot_pars[desc].irt_angles[e_irt_angle_rot]);

	if (irt_top->irt_desc[desc].irt_mode == e_irt_affine ||
		irt_top->irt_desc[desc].irt_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_affine ||
		irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_affine) {
		if (irt_top->rot_pars[desc].affine_flags[0]) IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f rotation angle\n", irt_top->rot_pars[desc].irt_angles[e_irt_angle_rot]);
		if (irt_top->rot_pars[desc].affine_flags[1]) IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f/%3.2f scaling factors\n", irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy);
		if (irt_top->rot_pars[desc].affine_flags[2]) IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s reflection mode\n", irt_refl_mode_s[irt_top->rot_pars[desc].reflection_mode]);
		if (irt_top->rot_pars[desc].affine_flags[3]) IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s shearing mode\n", irt_shr_mode_s[irt_top->rot_pars[desc].shear_mode]);
	}

	if (irt_top->irt_desc[desc].irt_mode == e_irt_projection && irt_top->rot_pars[desc].proj_mode == e_irt_projection ||
		irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_projection)
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s order, roll %3.2f, pitch %3.2f, yaw %3.2f, Sx %3.2f, Sy %3.2f, Zd %3.2f, Wd %3.2f %s shear mode, X shear %3.2f, Y shear %3.2f\n",
			irt_top->rot_pars[desc].proj_order, irt_top->rot_pars[desc].irt_angles[e_irt_angle_roll], irt_top->rot_pars[desc].irt_angles[e_irt_angle_pitch], irt_top->rot_pars[desc].irt_angles[e_irt_angle_yaw], irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy, irt_top->rot_pars[desc].proj_Zd, irt_top->rot_pars[desc].proj_Wd,
			irt_shr_mode_s[irt_top->rot_pars[desc].shear_mode], irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_x], irt_top->rot_pars[desc].irt_angles[e_irt_angle_shr_y]);

	if (irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->rot_pars[desc].mesh_mode == e_irt_mesh)
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f distortion\n", irt_top->rot_pars[desc].dist_r);

	if ((desc % PLANES)==0)
		irt_top->irt_desc_print(irt_top, desc);

	for (uint16_t row = 0; row < Ho; row++) {
		for (uint16_t col = 0; col < So; col += adj_proc_size) {

			//IRT_TRACE_UTILS("%d, %d\n", row, col);
			for (int i = 0; i < IRT_ROT_MEM_ROW_BANKS; i++) {
				row_bank_access[i][0] = 0; row_bank_access[i][1] = 10000; row_bank_access[i][2] = 0;
				row_bank_access[i][3] = 10000; row_bank_access[i][4] = 0;
			}

			for (uint8_t row_bank = 0; row_bank < IRT_ROT_MEM_ROW_BANKS; row_bank++) {
				for (uint8_t col_bank = 0; col_bank < 2; col_bank++) {
					rot_mem_rd_stat.bank_num_read[row_bank][col_bank] = 0;
					rot_mem_rd_stat.bank_min_addr[row_bank][col_bank] = 10000;
					rot_mem_rd_stat.bank_max_addr[row_bank][col_bank] = -1;

				}
			}

			if (irt_top->irt_desc[desc].rate_mode == e_irt_rate_fixed)
				adj_proc_size = irt_top->irt_desc[desc].proc_size; //constant will be used
			else
				adj_proc_size = IRT_ROT_MAX_PROC_SIZE; //set for 8 for initial try;

			cycles++;
			remain_pixels = So - col;
			//if (remain_pixels > adj_proc_size) remain_pixels = adj_proc_size;
			adj_proc_size = (uint8_t)std::min((int)adj_proc_size, int(remain_pixels));//to not exceed line width

			for (uint8_t pixel = 0; pixel < adj_proc_size; pixel++) {

				xo_i = 2 * (col + pixel) - Xoc; yo_i = 2 * row - Yoc;
				xo_d = (double)xo_i / 2; yo_d = (double)yo_i / 2;
				xo_f = (float)xo_i / 2; yo_f = (float)yo_i / 2;

				xo_arch = (float)(col + pixel - ((col + pixel) % irt_top->irt_desc[desc].proc_size)) - (float)Xoc / 2;

				//				printf("%d %d %d\n", row, Wo, pixel);
				xi_fix16 = 0; yi_fix16 = 0;
				switch (transform_type) {
				case e_irt_rotation:
					xi_fpt64 =  irt_top->rot_pars[desc].cosd * xo_d  + irt_top->rot_pars[desc].sind * yo_d + (double)Xic / 2;
					yi_fpt64 = -irt_top->rot_pars[desc].sind * xo_d  + irt_top->rot_pars[desc].cosd * yo_d + (double)Yic / 2;
					xi_fpt32 =  irt_top->irt_desc[desc].cosf * xo_f  + irt_top->irt_desc[desc].sinf * yo_f + (float)Xic / 2;
					yi_fpt32 = -irt_top->irt_desc[desc].sinf * xo_f  + irt_top->irt_desc[desc].cosf * yo_f + (float)Yic / 2;
					xi_fixed =  (int64_t)irt_top->irt_desc[desc].cosi * xo_i + (int64_t)irt_top->irt_desc[desc].sini * yo_i + ((int64_t)Xic << irt_top->rot_pars[desc].TOTAL_PREC);
					yi_fixed = -(int64_t)irt_top->irt_desc[desc].sini * xo_i + (int64_t)irt_top->irt_desc[desc].cosi * yo_i + ((int64_t)Yic << irt_top->rot_pars[desc].TOTAL_PREC);
					xi_fix16 =  irt_top->rot_pars[desc].cos16 * xo_i + irt_top->rot_pars[desc].sin16 * yo_i + ((int32_t)Xic << 16);
					yi_fix16 = -irt_top->rot_pars[desc].sin16 * xo_i + irt_top->rot_pars[desc].cos16 * yo_i + ((int32_t)Yic << 16);
					//xi_farch = ( irt_top->irt_desc[desc].cosf * (xo_f - pixel) + irt_top->irt_desc[desc].sinf * yo_f) + pixel * irt_top->irt_desc[desc].cosf;
					//yi_farch = (-irt_top->irt_desc[desc].sinf * (xo_f - pixel) + irt_top->irt_desc[desc].cosf * yo_f) - pixel * irt_top->irt_desc[desc].sinf;
					xi_farch =  (irt_top->irt_desc[desc].cosf * xo_arch + irt_top->irt_desc[desc].sinf * yo_f) + (xo_f - xo_arch) * irt_top->irt_desc[desc].cosf;
					yi_farch = (-irt_top->irt_desc[desc].sinf * xo_arch + irt_top->irt_desc[desc].cosf * yo_f) - (xo_f - xo_arch) * irt_top->irt_desc[desc].sinf;
					xi_fix31 =  (((int64_t)irt_top->irt_desc[desc].cosi * xo_i + (int64_t)irt_top->irt_desc[desc].sini * yo_i) << irt_top->irt_desc[desc].prec_align);
					yi_fix31 = ((-(int64_t)irt_top->irt_desc[desc].sini * xo_i + (int64_t)irt_top->irt_desc[desc].cosi * yo_i) << irt_top->irt_desc[desc].prec_align);
#ifdef IRT_HL_NO_FLIP
					xi_fpt64 =			   irt_top->rot_pars[desc].cosd_hl * xo_d +			 irt_top->rot_pars[desc].sind_hl * yo_d  + (double)Xic / 2;
					yi_fpt64 =		      -irt_top->rot_pars[desc].sind_hl * xo_d +			 irt_top->rot_pars[desc].cosd_hl * yo_d  + (double)Yic / 2;
					xi_fpt32 =		 	   irt_top->rot_pars[desc].cosf_hl * xo_f +			 irt_top->rot_pars[desc].sinf_hl * yo_f  + (float)Xic / 2;
					yi_fpt32 =			  -irt_top->rot_pars[desc].sinf_hl * xo_f +			 irt_top->rot_pars[desc].cosf_hl * yo_f  + (float)Yic / 2;
					xi_fixed =    (int64_t)irt_top->rot_pars[desc].cosi_hl * xo_i + (int64_t)irt_top->rot_pars[desc].sini_hl * yo_i  + ((int64_t)Xic << irt_top->rot_pars[desc].TOTAL_PREC);
					yi_fixed =   -(int64_t)irt_top->rot_pars[desc].sini_hl * xo_i + (int64_t)irt_top->rot_pars[desc].cosi_hl * yo_i  + ((int64_t)Yic << irt_top->rot_pars[desc].TOTAL_PREC);
					xi_farch =			  (irt_top->rot_pars[desc].cosf_hl * xo_arch +		 irt_top->rot_pars[desc].sinf_hl * yo_f) + (xo_f - xo_arch) * irt_top->rot_pars[desc].cosf_hl;
					yi_farch =			 (-irt_top->rot_pars[desc].sinf_hl * xo_arch +		 irt_top->rot_pars[desc].cosf_hl * yo_f) - (xo_f - xo_arch) * irt_top->rot_pars[desc].sinf_hl;
					xi_fix31 = ((( int64_t)irt_top->rot_pars[desc].cosi_hl * xo_i + (int64_t)irt_top->rot_pars[desc].sini_hl * yo_i) << irt_top->irt_desc[desc].prec_align);
					yi_fix31 = ((-(int64_t)irt_top->rot_pars[desc].sini_hl * xo_i + (int64_t)irt_top->rot_pars[desc].cosi_hl * yo_i) << irt_top->irt_desc[desc].prec_align);
#endif
					break;
				case e_irt_affine: //affine transform
					xi_fpt64 =  irt_top->rot_pars[desc].M11d * xo_d + irt_top->rot_pars[desc].M12d * yo_d + (double)Xic / 2;
					yi_fpt64 =  irt_top->rot_pars[desc].M21d * xo_d + irt_top->rot_pars[desc].M22d * yo_d + (double)Yic / 2;
					xi_fpt32 =  irt_top->irt_desc[desc].M11f * xo_f + irt_top->irt_desc[desc].M12f * yo_f + (float)Xic / 2;
					yi_fpt32 =  irt_top->irt_desc[desc].M21f * xo_f + irt_top->irt_desc[desc].M22f * yo_f + (float)Yic / 2;
					xi_fixed = (int64_t)irt_top->irt_desc[desc].M11i * xo_i + (int64_t)irt_top->irt_desc[desc].M12i * yo_i + ((int64_t)Xic << irt_top->rot_pars[desc].TOTAL_PREC);
					yi_fixed = (int64_t)irt_top->irt_desc[desc].M21i * xo_i + (int64_t)irt_top->irt_desc[desc].M22i * yo_i + ((int64_t)Yic << irt_top->rot_pars[desc].TOTAL_PREC);
					//xi_farch = (irt_top->irt_desc[desc].M11f * (xo_f - pixel) + irt_top->irt_desc[desc].M12f * yo_f) + pixel * irt_top->irt_desc[desc].M11f;
					//yi_farch = (irt_top->irt_desc[desc].M21f * (xo_f - pixel) + irt_top->irt_desc[desc].M22f * yo_f) + pixel * irt_top->irt_desc[desc].M21f;
					xi_farch = (irt_top->irt_desc[desc].M11f * xo_arch + irt_top->irt_desc[desc].M12f * yo_f) + (xo_f - xo_arch) * irt_top->irt_desc[desc].M11f;
					yi_farch = (irt_top->irt_desc[desc].M21f * xo_arch + irt_top->irt_desc[desc].M22f * yo_f) + (xo_f - xo_arch) * irt_top->irt_desc[desc].M21f;
					xi_fix31 = (((int64_t)irt_top->irt_desc[desc].M11i * xo_i + (int64_t)irt_top->irt_desc[desc].M12i * yo_i) << irt_top->irt_desc[desc].prec_align);
					yi_fix31 = (((int64_t)irt_top->irt_desc[desc].M21i * xo_i + (int64_t)irt_top->irt_desc[desc].M22i * yo_i) << irt_top->irt_desc[desc].prec_align);
					break;
				case e_irt_projection: //projection transform
					xi_fpt64 = (irt_top->rot_pars[desc].prj_Ad[0] * xo_d + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * xo_d + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yi_fpt64 = (irt_top->rot_pars[desc].prj_Bd[0] * xo_d + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * xo_d + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					xi_fpt32 = (irt_top->irt_desc[desc].prj_Af[0] * xo_f + irt_top->irt_desc[desc].prj_Af[1] * yo_f + irt_top->irt_desc[desc].prj_Af[2]) / (irt_top->irt_desc[desc].prj_Cf[0] * xo_f + irt_top->irt_desc[desc].prj_Cf[1] * yo_f + irt_top->irt_desc[desc].prj_Cf[2]);
					yi_fpt32 = (irt_top->irt_desc[desc].prj_Bf[0] * xo_f + irt_top->irt_desc[desc].prj_Bf[1] * yo_f + irt_top->irt_desc[desc].prj_Bf[2]) / (irt_top->irt_desc[desc].prj_Df[0] * xo_f + irt_top->irt_desc[desc].prj_Df[1] * yo_f + irt_top->irt_desc[desc].prj_Df[2]);
#if 0
					N = (irt_top->rot_pars[desc].prj_Ai[0] * xo_i + irt_top->rot_pars[desc].prj_Ai[1] * yo_i + irt_top->rot_pars[desc].prj_Ai[2]);
					D = (irt_top->rot_pars[desc].prj_Ci[0] * xo_i + irt_top->rot_pars[desc].prj_Ci[1] * yo_i + irt_top->rot_pars[desc].prj_Ci[2]);
					xi_fixed = (int64_t)floor(((double)N * pow(2.0, irt_top->rot_pars[desc].PROJ_DEN_PREC - irt_top->rot_pars[desc].PROJ_NOM_PREC + irt_top->rot_pars[desc].TOTAL_PREC)) / (double)D);
					N = (irt_top->rot_pars[desc].prj_Bi[0] * xo_i + irt_top->rot_pars[desc].prj_Bi[1] * yo_i + irt_top->rot_pars[desc].prj_Bi[2]);
					D = (irt_top->rot_pars[desc].prj_Di[0] * xo_i + irt_top->rot_pars[desc].prj_Di[1] * yo_i + irt_top->rot_pars[desc].prj_Di[2]);
					yi_fixed = (int64_t)floor(((double)N * pow(2.0, irt_top->rot_pars[desc].PROJ_DEN_PREC - irt_top->rot_pars[desc].PROJ_NOM_PREC + irt_top->rot_pars[desc].TOTAL_PREC)) / (double)D);
					xi_farch = (irt_top->irt_desc[desc].prj_Af[0] * xo_f + irt_top->irt_desc[desc].prj_Af[1] * yo_f + irt_top->irt_desc[desc].prj_Af[2]) / (irt_top->irt_desc[desc].prj_Cf[0] * xo_f + irt_top->irt_desc[desc].prj_Cf[1] * yo_f + irt_top->irt_desc[desc].prj_Cf[2]);
					yi_farch = (irt_top->irt_desc[desc].prj_Bf[0] * xo_f + irt_top->irt_desc[desc].prj_Bf[1] * yo_f + irt_top->irt_desc[desc].prj_Bf[2]) / (irt_top->irt_desc[desc].prj_Df[0] * xo_f + irt_top->irt_desc[desc].prj_Df[1] * yo_f + irt_top->irt_desc[desc].prj_Df[2]);

					//using Xo as reference
					xi_fpt32_inc = (irt_top->rot_pars[desc].prj_Ad[0] * xo_d + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yi_fpt32_inc = (irt_top->rot_pars[desc].prj_Bd[0] * xo_d + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					proj_coord_err_hist[proj_inc_Xo][0][std::min((int)floor(fabs(xi_fpt32_inc - xi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					proj_coord_err_hist[proj_inc_Xo][1][std::min((int)floor(fabs(yi_fpt32_inc - yi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					//using Xo+3 as reference
					xi_fpt32_inc = (irt_top->rot_pars[desc].prj_Ad[0] * xo_d + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * (xo_d - pixel + 4) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yi_fpt32_inc = (irt_top->rot_pars[desc].prj_Bd[0] * xo_d + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * (xo_d - pixel + 4) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					proj_coord_err_hist[proj_inc_Xo3][0][std::min((int)floor(fabs(xi_fpt32_inc - xi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					proj_coord_err_hist[proj_inc_Xo3][1][std::min((int)floor(fabs(yi_fpt32_inc - yi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					//using output to input slope
					xis_fpt64 = (irt_top->rot_pars[desc].prj_Ad[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yis_fpt64 = (irt_top->rot_pars[desc].prj_Bd[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * (xo_d - pixel) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					xie_fpt64 = (irt_top->rot_pars[desc].prj_Ad[0] * (xo_d - pixel + 7) + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * (xo_d - pixel + 7) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yie_fpt64 = (irt_top->rot_pars[desc].prj_Bd[0] * (xo_d - pixel + 7) + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * (xo_d - pixel + 7) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					xi_slope = (xie_fpt64 - xis_fpt64) / 7.0;
					yi_slope = (yie_fpt64 - yis_fpt64) / 7.0;
					xi_fpt32_inc = xis_fpt64 + xi_slope * (double)pixel;
					yi_fpt32_inc = yis_fpt64 + yi_slope * (double)pixel;
					proj_coord_err_hist[proj_slope_O2I][0][std::min((int)floor(fabs(xi_fpt32_inc - xi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					proj_coord_err_hist[proj_slope_O2I][1][std::min((int)floor(fabs(yi_fpt32_inc - yi_fpt64) * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					//using input to input slope
					yi_slope = (yie_fpt64 - yis_fpt64) / (xie_fpt64 - xis_fpt64);
					xi_fpt32_inc = xi_fpt64;
					yi_fpt32_inc = yis_fpt64 + yi_slope * (xi_fpt64 - xis_fpt64);
					proj_coord_err_hist[proj_slope_I2I][0][std::min((int)floor(fabs(xi_fpt32_inc - xi_fpt64) * 100 * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					proj_coord_err_hist[proj_slope_I2I][1][std::min((int)floor(fabs(yi_fpt32_inc - yi_fpt64) * 100 * PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					//using whole stripe as slope
					xis_fpt64 = (irt_top->rot_pars[desc].prj_Ad[0] * (-(double)Xoc / 2) + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * (-(double)Xoc / 2) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yis_fpt64 = (irt_top->rot_pars[desc].prj_Bd[0] * (-(double)Xoc / 2) + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * (-(double)Xoc / 2) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					xie_fpt64 = (irt_top->rot_pars[desc].prj_Ad[0] * ((double)Wo - 1 - (double)Xoc / 2) + irt_top->rot_pars[desc].prj_Ad[1] * yo_d + irt_top->rot_pars[desc].prj_Ad[2]) / (irt_top->rot_pars[desc].prj_Cd[0] * ((double)Wo - 1 - (double)Xoc / 2) + irt_top->rot_pars[desc].prj_Cd[1] * yo_d + irt_top->rot_pars[desc].prj_Cd[2]);
					yie_fpt64 = (irt_top->rot_pars[desc].prj_Bd[0] * ((double)Wo - 1 - (double)Xoc / 2) + irt_top->rot_pars[desc].prj_Bd[1] * yo_d + irt_top->rot_pars[desc].prj_Bd[2]) / (irt_top->rot_pars[desc].prj_Dd[0] * ((double)Wo - 1 - (double)Xoc / 2) + irt_top->rot_pars[desc].prj_Dd[1] * yo_d + irt_top->rot_pars[desc].prj_Dd[2]);
					yi_slope = (yie_fpt64 - yis_fpt64) / (xie_fpt64 - xis_fpt64);
					xi_fpt32_inc = xi_fpt64;
					yi_fpt32_inc = yis_fpt64 + yi_slope * (xi_fpt64 - xis_fpt64);
					proj_coord_err_hist[proj_slope_S2I][0][std::min((int)floor(fabs(xi_fpt32_inc - xi_fpt64)* PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
					proj_coord_err_hist[proj_slope_S2I][1][std::min((int)floor(fabs(yi_fpt32_inc - yi_fpt64)* PROJ_ERR_BINS + 0.5), PROJ_ERR_BINS - 1)] ++;
#endif
					//arch implementation
					xis_fpt64 =	((irt_top->rot_pars[desc].prj_Ad[0] * xo_arch + irt_top->rot_pars[desc].prj_Ad[1] * yo_d) + irt_top->rot_pars[desc].prj_Ad[2]) /
								((irt_top->rot_pars[desc].prj_Cd[0] * xo_arch + irt_top->rot_pars[desc].prj_Cd[1] * yo_d) + irt_top->rot_pars[desc].prj_Cd[2]);
					xie_fpt64 =	((irt_top->rot_pars[desc].prj_Ad[0] * xo_arch + irt_top->rot_pars[desc].prj_Ad[1] * yo_d) + irt_top->rot_pars[desc].prj_Ad[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->rot_pars[desc].prj_Ad[2]) /
								((irt_top->rot_pars[desc].prj_Cd[0] * xo_arch + irt_top->rot_pars[desc].prj_Cd[1] * yo_d) + irt_top->rot_pars[desc].prj_Cd[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->rot_pars[desc].prj_Cd[2]);

					yis_fpt64 =	((irt_top->rot_pars[desc].prj_Bd[0] * xo_arch + irt_top->rot_pars[desc].prj_Bd[1] * yo_d) + irt_top->rot_pars[desc].prj_Bd[2]) /
								((irt_top->rot_pars[desc].prj_Dd[0] * xo_arch + irt_top->rot_pars[desc].prj_Dd[1] * yo_d) + irt_top->rot_pars[desc].prj_Dd[2]);
					yie_fpt64 =	((irt_top->rot_pars[desc].prj_Bd[0] * xo_arch + irt_top->rot_pars[desc].prj_Bd[1] * yo_d) + irt_top->rot_pars[desc].prj_Bd[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->rot_pars[desc].prj_Bd[2]) /
								((irt_top->rot_pars[desc].prj_Dd[0] * xo_arch + irt_top->rot_pars[desc].prj_Dd[1] * yo_d) + irt_top->rot_pars[desc].prj_Dd[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->rot_pars[desc].prj_Dd[2]);

					xi_slope_d = (xie_fpt64 - xis_fpt64) * (double)(1.0 / (irt_top->irt_desc[desc].proc_size - 1));
					yi_slope_d = (yie_fpt64 - yis_fpt64) * (double)(1.0 / (irt_top->irt_desc[desc].proc_size - 1));

					if (irt_top->irt_desc[desc].proc_size == 1) {
						xi_fpt64 = xis_fpt64;
						yi_fpt64 = yis_fpt64;
					} else {
						xi_fpt64 = xis_fpt64 + xi_slope_d * (xo_d - xo_arch);
						yi_fpt64 = yis_fpt64 + yi_slope_d * (xo_d - xo_arch);
					}

					xis_fpt32 = ((irt_top->irt_desc[desc].prj_Af[0] * xo_arch + irt_top->irt_desc[desc].prj_Af[1] * yo_f) + irt_top->irt_desc[desc].prj_Af[2]) /
								((irt_top->irt_desc[desc].prj_Cf[0] * xo_arch + irt_top->irt_desc[desc].prj_Cf[1] * yo_f) + irt_top->irt_desc[desc].prj_Cf[2]);
					xie_fpt32 = ((irt_top->irt_desc[desc].prj_Af[0] * xo_arch + irt_top->irt_desc[desc].prj_Af[1] * yo_f) + irt_top->irt_desc[desc].prj_Af[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->irt_desc[desc].prj_Af[2]) /
								((irt_top->irt_desc[desc].prj_Cf[0] * xo_arch + irt_top->irt_desc[desc].prj_Cf[1] * yo_f) + irt_top->irt_desc[desc].prj_Cf[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->irt_desc[desc].prj_Cf[2]);

					yis_fpt32 = ((irt_top->irt_desc[desc].prj_Bf[0] * xo_arch + irt_top->irt_desc[desc].prj_Bf[1] * yo_f) + irt_top->irt_desc[desc].prj_Bf[2]) /
								((irt_top->irt_desc[desc].prj_Df[0] * xo_arch + irt_top->irt_desc[desc].prj_Df[1] * yo_f) + irt_top->irt_desc[desc].prj_Df[2]);
					yie_fpt32 = ((irt_top->irt_desc[desc].prj_Bf[0] * xo_arch + irt_top->irt_desc[desc].prj_Bf[1] * yo_f) + irt_top->irt_desc[desc].prj_Bf[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->irt_desc[desc].prj_Bf[2]) /
								((irt_top->irt_desc[desc].prj_Df[0] * xo_arch + irt_top->irt_desc[desc].prj_Df[1] * yo_f) + irt_top->irt_desc[desc].prj_Df[0] * (irt_top->irt_desc[desc].proc_size - 1) + irt_top->irt_desc[desc].prj_Df[2]);

					xi_slope_f = (xie_fpt32 - xis_fpt32) * (float)(1.0 / (irt_top->irt_desc[desc].proc_size - 1));
					yi_slope_f = (yie_fpt32 - yis_fpt32) * (float)(1.0 / (irt_top->irt_desc[desc].proc_size - 1));

					if (irt_top->irt_desc[desc].proc_size == 1) {
						xi_farch = xis_fpt32;
						yi_farch = yis_fpt32;
					} else {
						xi_farch = xis_fpt32 + xi_slope_f * (xo_f - xo_arch);
						yi_farch = yis_fpt32 + yi_slope_f * (xo_f - xo_arch);
					}
					xi_fixed = 0;
					yi_fixed = 0;
					xi_fix31 = 0;
					yi_fix31 = 0;
					xi_fix16 = 0;
					yi_fix16 = 0;

					break;
				case e_irt_mesh: //distortion
					//mapping output image indexes to mesh image coordinate
					yim_fpt64 = (double)row;
					yim_fpt32 = (float)row;
					yim_fixed = (int64_t)row;
					yim_arch  = (int64_t)row;
					if (irt_top->irt_desc[desc].mesh_sparse_v == 1) { //sparse in v direction
						yim_fpt64 = yim_fpt64 * (double)irt_top->irt_desc[desc].mesh_Gv / pow(2.0, IRT_MESH_G_PREC);
						yim_fpt32 = yim_fpt32 * (float)irt_top->irt_desc[desc].mesh_Gv / pow((float)2, IRT_MESH_G_PREC);
						yim_fixed = (yim_fixed * (int64_t)irt_top->irt_desc[desc].mesh_Gv + (1 << (IRT_MESH_G_PREC - 1))) >> IRT_MESH_G_PREC;
						yim_arch = yim_arch * (int64_t)irt_top->irt_desc[desc].mesh_Gv;
					} else {
						yim_fixed = yim_fixed << irt_top->rot_pars[desc].TOTAL_PREC;
						yim_arch  = yim_arch << IRT_MESH_G_PREC;
					}
					//yi_fixed <<= IRT_TOTAL_PREC;

					xim_fpt64 = (double)col + pixel;
					xim_fpt32 = (float)col + pixel;
					xim_fixed = (int64_t)col + pixel;
					xim_arch  = (int64_t)col + pixel;
					if (irt_top->irt_desc[desc].mesh_sparse_h == 1) {//sparse in h direction
						xim_fpt64 = xim_fpt64 * (double) irt_top->irt_desc[desc].mesh_Gh / pow(2.0, IRT_MESH_G_PREC);
						xim_fpt32 = xim_fpt32 * (float)irt_top->irt_desc[desc].mesh_Gh / pow((float)2, IRT_MESH_G_PREC);
						xim_fixed = (xim_fixed * (int64_t)irt_top->irt_desc[desc].mesh_Gh + (1 << (IRT_MESH_G_PREC - 1))) >> IRT_MESH_G_PREC;
						xim_arch  = xim_arch  * (int64_t)irt_top->irt_desc[desc].mesh_Gh;
					} else {
						xim_fixed = xim_fixed << irt_top->rot_pars[desc].TOTAL_PREC;
						xim_arch = xim_arch << IRT_MESH_G_PREC;
					}
					//xi_fixed <<= IRT_TOTAL_PREC;

					//calculating interpolation weights and mesh indexes
					j0[e_irt_mesh_fpt64] = (int16_t)floor(xim_fpt64);
					j1[e_irt_mesh_fpt64] = j0[e_irt_mesh_fpt64] + 1;
					xf_fpt64  = xim_fpt64 - (double)j0[e_irt_mesh_fpt64];

					i0[e_irt_mesh_fpt64] = (int16_t)floor(yim_fpt64);
					i1[e_irt_mesh_fpt64] = i0[e_irt_mesh_fpt64] + 1;
					yf_fpt64  = yim_fpt64 - (double)i0[e_irt_mesh_fpt64];

					j0[e_irt_mesh_fpt32] = (int16_t)floor(xim_fpt32);
					j1[e_irt_mesh_fpt32] = j0[e_irt_mesh_fpt32] + 1;
					xf_fpt32  = xim_fpt32 - (float)j0[e_irt_mesh_fpt32];

					i0[e_irt_mesh_fpt32] = (int16_t)floor(yim_fpt32);
					i1[e_irt_mesh_fpt32] = i0[e_irt_mesh_fpt32] + 1;
					yf_fpt32  = yim_fpt32 - (float)i0[e_irt_mesh_fpt32];

					j0[e_irt_mesh_fixed] = (int16_t)(xim_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
					j1[e_irt_mesh_fixed] = j0[e_irt_mesh_fixed] + 1;
					xf_fixed  = (xim_fixed - ((int64_t)j0[e_irt_mesh_fixed] << irt_top->rot_pars[desc].TOTAL_PREC));

					i0[e_irt_mesh_fixed] = (int16_t)(yim_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
					i1[e_irt_mesh_fixed] = i0[e_irt_mesh_fixed] + 1;
					yf_fixed = (yim_fixed - ((int64_t)i0[e_irt_mesh_fixed] << irt_top->rot_pars[desc].TOTAL_PREC));

					j0[e_irt_mesh_arch]  = (int16_t)(xim_arch >> IRT_MESH_G_PREC);
					j1[e_irt_mesh_arch] = j0[e_irt_mesh_arch] + 1;
					xf_fix31 = (uint32_t)(xim_arch - ((int64_t)j0[e_irt_mesh_arch] << IRT_MESH_G_PREC));
					xf_arch = (float)((double)((xim_arch - ((int64_t)j0[e_irt_mesh_arch] << IRT_MESH_G_PREC)) / pow(2.0, IRT_MESH_G_PREC)));

					i0[e_irt_mesh_arch] = (int16_t)(yim_arch >> IRT_MESH_G_PREC);
					i1[e_irt_mesh_arch] = i0[e_irt_mesh_arch] + 1;
					yf_fix31 = (uint32_t)(yim_arch - ((int64_t)i0[e_irt_mesh_arch] << IRT_MESH_G_PREC));
					yf_arch = (float)((double)((yim_arch - ((int64_t)i0[e_irt_mesh_arch] << IRT_MESH_G_PREC)) / pow(2.0, IRT_MESH_G_PREC)));

					if (j1[e_irt_mesh_arch] >= Wm) { //reset weight for oob
						xf_fix31 = 0;
						xf_arch = 0;
					}
					if (i1[e_irt_mesh_arch] >= Hm) {//reset weight for oob
						yf_fix31 = 0;
						yf_arch = 0;
					}

					for (uint8_t format = e_irt_mesh_fpt64; format <= e_irt_mesh_arch; format++) {
						i1[format] = IRT_top::IRT_UTILS::irt_min_int16(i1[format], Hm - 1);
						j1[format] = IRT_top::IRT_UTILS::irt_min_int16(j1[format], Wm - 1);
					}

					for (uint8_t format = e_irt_mesh_fpt64; format <= e_irt_mesh_arch; format++) {
						//check pixel inside of image boundary
						p_valid[0] = i0[format] >= 0 && i0[format] < Hm && j0[format] >= 0 && j0[format] < Wm;
						p_valid[1] = i0[format] >= 0 && i0[format] < Hm && j1[format] >= 0 && j1[format] < Wm;
						p_valid[2] = i1[format] >= 0 && i1[format] < Hm && j0[format] >= 0 && j0[format] < Wm;
						p_valid[3] = i1[format] >= 0 && i1[format] < Hm && j1[format] >= 0 && j1[format] < Wm;

						////selecting X for interpolation assign pixel
						mp_value[0][0] = (float)(p_valid[0] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i0[format]][j0[format]].x : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i0[format]][j0[format]].x / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[0][1] = (float)(p_valid[1] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i0[format]][j1[format]].x : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i0[format]][j1[format]].x / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[0][2] = (float)(p_valid[2] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i1[format]][j0[format]].x : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i1[format]][j0[format]].x / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[0][3] = (float)(p_valid[3] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i1[format]][j1[format]].x : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i1[format]][j1[format]].x / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);

						////selecting Y for interpolation assign pixel
						mp_value[1][0] = (float)(p_valid[0] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i0[format]][j0[format]].y : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i0[format]][j0[format]].y / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[1][1] = (float)(p_valid[1] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i0[format]][j1[format]].y : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i0[format]][j1[format]].y / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[1][2] = (float)(p_valid[2] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i1[format]][j0[format]].y : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i1[format]][j0[format]].y / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
						mp_value[1][3] = (float)(p_valid[3] ? (mesh_format ? irt_top->irt_cfg.mesh_images.mesh_image_fp32[i1[format]][j1[format]].y : (float)irt_top->irt_cfg.mesh_images.mesh_image_fi16[i1[format]][j1[format]].y / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);

						//recover mesh values from ext memory
						uint32_t mp_value_fp32[2][4];
						int16_t mp_value_int16[2][4];
						uint64_t mimage_addr_start = irt_top->irt_desc[desc].image_par[MIMAGE].addr_start;
						uint64_t mimage_row_start, mimage_col_start, mimage_pxl_start, mimage_crd_start;
						uint16_t mesh_BPC = irt_top->irt_desc[desc].image_par[MIMAGE].Ps == 2 ? 2 : 4;
						uint8_t idx = 0;
						if (test_file_flag) {
							for (int16_t i = i0[format]; i <= i1[format]; i++) {
								for (int16_t j = j0[format]; j <= j1[format]; j++) {

									mimage_row_start = (uint64_t)i * irt_top->irt_desc[desc].image_par[MIMAGE].Hs;
									mimage_col_start = (uint64_t)j * mesh_BPC * 2;
									mimage_pxl_start = mimage_addr_start + mimage_row_start + mimage_col_start;

									for (uint8_t crd = 0; crd <= 1; crd++) {
										mp_value_fp32[crd][idx] = 0;
										mp_value_int16[crd][idx] = 0;
										mimage_crd_start = mimage_pxl_start + ((uint64_t)crd * mesh_BPC);

										for (uint16_t byte = 0; byte < mesh_BPC; byte++) {
											mp_value_fp32[crd][idx]  |= ((uint32_t)ext_mem[mimage_crd_start + (uint64_t)byte] << (8 * byte));
											mp_value_int16[crd][idx] |= ((uint16_t)ext_mem[mimage_crd_start + (uint64_t)byte] << (8 * byte));
										}
										mp_value[crd][idx] = (float)(p_valid[idx] ? (mesh_format ? IRT_top::IRT_UTILS::irt_fp32_to_float(mp_value_fp32[crd][idx]) : 
																			  (float)mp_value_int16[crd][idx] / pow(2.0, irt_top->irt_desc[desc].mesh_point_location)) : 0);
									}
									idx++;
								}
							}
						}
						if (format == e_irt_mesh_fpt64) {
							//printf("[%d, %d] = [%f, %f, %f, %f]\n", row, col + pixel, mp_value[0][0], mp_value[0][1], mp_value[0][2], mp_value[0][3]);
							//printf("[%d, %d] = [%f, %f, %f, %f]\n", row, col + pixel, mp_value[1][0], mp_value[1][1], mp_value[1][2], mp_value[1][3]);
						}
						//cast pixel according to format
						for (uint8_t i = 0; i < 4; i++) {
							for (uint8_t j = 0; j < 2; j++) {
								switch (format) {
								case e_irt_mesh_fpt64: mp_fpt64[j][i] = (double)mp_value[j][i]; break;
								case e_irt_mesh_fpt32: mp_fpt32[j][i] = mp_value[j][i]; break;
								case e_irt_mesh_fixed: mp_fixed[j][i] = (double)mp_value[j][i]; break;
								case e_irt_mesh_arch:  mp_arch[j][i]  = mp_value[j][i]; break;
								}
							}
						}
					}

					//interpolation
					double mi_fpt64[2], mi_fixed[2];
					float mi_fpt32[2], mi_arch[2];

					xw0 = (float)((double)xf_fix31 / pow(2.0, 31));
					xw1 = (float)((pow(2.0, 31) - (double)xf_fix31) / pow(2.0, 31));
					yw0 = (float)((double)yf_fix31 / pow(2.0, 31));
					yw1 = (float)((pow(2.0, 31) - (double)yf_fix31) / pow(2.0, 31));

					for (uint8_t j = 0; j < 2; j++) {
						//horizontal followed by vertical
						mi_fpt64[j] = (mp_fpt64[j][0] * (1 - xf_fpt64) + mp_fpt64[j][1] * xf_fpt64) * (1 - yf_fpt64) + (mp_fpt64[j][2] * (1 - xf_fpt64) + mp_fpt64[j][3] * xf_fpt64) * yf_fpt64;
						mi_fpt32[j] = (mp_fpt32[j][0] * (1 - xf_fpt32) + mp_fpt32[j][1] * xf_fpt32) * (1 - yf_fpt32) + (mp_fpt32[j][2] * (1 - xf_fpt32) + mp_fpt32[j][3] * xf_fpt32) * yf_fpt32;
						//xi_fixed = mp0_fixed * (1 - xf_fixed) * (1 - yf_fixed) + mp1_fixed * (xf_fixed) * (1 - yf_fixed) + mp2_fixed * (1 - xf_fixed) * (yf_fixed) + mp3_fixed * (xf_fixed) * (yf_fixed);
						mi_fixed[j] = (mp_fixed[j][0] * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)xf_fixed) + mp_fixed[j][1] * (double)xf_fixed) * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)yf_fixed) +
							(mp_fixed[j][2] * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)xf_fixed) + mp_fixed[j][3] * (double)xf_fixed) * (double)yf_fixed;
						mi_fixed[j] *= pow(4.0, -irt_top->rot_pars[desc].TOTAL_PREC);

						mi_arch[j] = (mp_arch[j][0] * (1 - xf_arch) + mp_arch[j][1] * xf_arch) * (1 - yf_arch) +
									 (mp_arch[j][2] * (1 - xf_arch) + mp_arch[j][3] * xf_arch) * yf_arch;

						//vertical vertical by horizontal
						mi_fpt64[j] = (mp_fpt64[j][0] * (1 - yf_fpt64) + mp_fpt64[j][2] * yf_fpt64) * (1 - xf_fpt64) + (mp_fpt64[j][1] * (1 - yf_fpt64) + mp_fpt64[j][3] * yf_fpt64) * xf_fpt64;
						mi_fpt32[j] = (mp_fpt32[j][0] * (1 - yf_fpt32) + mp_fpt32[j][2] * yf_fpt32) * (1 - xf_fpt32) + (mp_fpt32[j][1] * (1 - yf_fpt32) + mp_fpt32[j][3] * yf_fpt32) * xf_fpt32;
						//xi_fixed = mp0_fixed * (1 - xf_fixed) * (1 - yf_fixed) + mp1_fixed * (xf_fixed) * (1 - yf_fixed) + mp2_fixed * (1 - xf_fixed) * (yf_fixed) + mp3_fixed * (xf_fixed) * (yf_fixed);
						mi_fixed[j] = (mp_fixed[j][0] * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)yf_fixed) + mp_fixed[j][2] * (double)yf_fixed) * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)xf_fixed) +
									  (mp_fixed[j][1] * (pow(2, irt_top->rot_pars[desc].TOTAL_PREC) - (double)yf_fixed) + mp_fixed[j][3] * (double)yf_fixed) * (double)xf_fixed;
						mi_fixed[j] *= pow(4.0, -irt_top->rot_pars[desc].TOTAL_PREC);

						mi_arch[j] = (mp_arch[j][0] * (1 - yf_arch) + mp_arch[j][2] * yf_arch) * (1 - xf_arch) +
									 (mp_arch[j][1] * (1 - yf_arch) + mp_arch[j][3] * yf_arch) * xf_arch;

						//as in arch model
						mi_arch[j] = (mp_arch[j][0] * yw1 + mp_arch[j][2] * yw0) * xw1 + (mp_arch[j][1] * yw1 + mp_arch[j][3] * yw0) * xw0;

					}

					xi_fpt64 = mi_fpt64[0]; yi_fpt64 = mi_fpt64[1];
					xi_fpt32 = mi_fpt32[0]; yi_fpt32 = mi_fpt32[1];
					xi_farch = mi_arch[0];  yi_farch = mi_arch[1];

#if 0
					IRT_TRACE("MESH Results for pixel [%d, %d] at task %d at cycle %d:\n", row, col + pixel, desc, 0);
					IRT_TRACE("weights[%8x,%8x] bli input: y: pix[%.8f,%.8f][%.8f,%.8f], x: pix[%.2f,%.2f][%.2f,%.2f], output[X, Y]:[%.8f,%.8f]\n",
						xf_fix31, yf_fix31,
						mp_arch[1][0], mp_arch[1][1], mp_arch[1][2], mp_arch[1][3], mp_arch[0][0], mp_arch[0][1], mp_arch[0][2], mp_arch[0][3],
						xi_farch, yi_farch);
#endif

					xi_fixed = (int64_t)rint(mi_fixed[0] * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)) * 2;
					yi_fixed = (int64_t)rint(mi_fixed[1] * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)) * 2;
					xi_fix16 = (int32_t)rint(mi_fixed[0] * pow(2.0, 16)) * 2;
					yi_fix16 = (int32_t)rint(mi_fixed[1] * pow(2.0, 16)) * 2;

					if (irt_top->irt_desc[desc].mesh_rel_mode) {
						xi_fpt64 += (double)col + (double)pixel;
						yi_fpt64 += (double)row;
						xi_fpt32 += (float)col + (float)pixel;
						yi_fpt32 += (float)row;
						xi_fixed += ((int64_t)col * 2 + (int64_t)pixel * 2) << irt_top->rot_pars[desc].TOTAL_PREC;
						yi_fixed += ((int64_t)row * 2) << irt_top->rot_pars[desc].TOTAL_PREC;
						xi_fix16 += ((int32_t)col * 2 + (int32_t)pixel * 2) << 16;
						yi_fix16 += ((int32_t)row * 2) << 16;
						//xi_farch += ((float)col + (float)pixel);
						//yi_farch += (float)row;
					}
					//printf("Mesh HL: [%d][%d] = [%f][%f]\n", row, col, xi_farch, yi_farch);

					//printf("Expended mesh values for Y[%d][%d] are [%f] from [%f][%f][%f][%f]\n", row, col + pixel, yi_fpt32, mp_fpt32_y[0], mp_fpt32_y[1], mp_fpt32_y[2], mp_fpt32_y[3]);
					//printf("Expended mesh values for X[%d][%d] are [%f] from [%f][%f][%f][%f]\n", row, col + pixel, xi_fpt32, mp_fpt32_x[0], mp_fpt32_x[1], mp_fpt32_x[2], mp_fpt32_x[3]);
					break;
				}

				//IRT_TRACE_DBG("[%u, %u]: %f %f\n", row, col + pixel, xi_farch, yi_farch);

				bool use_fixed = irt_top->irt_desc[desc].crd_mode == e_irt_crd_mode_fixed && (irt_top->irt_desc[desc].irt_mode == e_irt_rotation || irt_top->irt_desc[desc].irt_mode == e_irt_affine);
				bool add_center = irt_top->irt_desc[desc].irt_mode == e_irt_rotation || irt_top->irt_desc[desc].irt_mode == e_irt_affine;
				bool add_rel = irt_top->irt_desc[desc].irt_mode == e_irt_mesh && irt_top->irt_desc[desc].mesh_rel_mode;

				xi_fixed = IRT_top::irt_fix32_to_fix31(xi_fixed);
				yi_fixed = IRT_top::irt_fix32_to_fix31(yi_fixed);

				xi_fix31 = IRT_top::irt_fix32_to_fix31(xi_fix31);
				yi_fix31 = IRT_top::irt_fix32_to_fix31(yi_fix31);

				xi_fix16 >>= 1;
				yi_fix16 >>= 1;

				xi_fix31 = (int64_t)(use_fixed ? xi_fix31 : rint(xi_farch * pow(2, 31))) + (add_center ? ((int64_t)Xic << 30) : 0) + (add_rel ? (((int64_t)col + pixel) << 31) : 0);
				yi_fix31 = (int64_t)(use_fixed ? yi_fix31 : rint(yi_farch * pow(2, 31))) + (add_center ? ((int64_t)Yic << 30) : 0) + (add_rel ? ( (int64_t)row << 31) : 0);

				//IRT_TRACE_DBG("[%u, %u]: %f %f\n", row, col + pixel, (double)xi_fix31 / pow(2, 31), (double)yi_fix31 / pow(2, 31));
				//IRT_TRACE_DBG("[%u, %u]: %f %f\n", row, col + pixel, xi_farch, yi_farch);

				if (irt_top->irt_desc[desc].bg_mode == e_irt_bg_frame_repeat) {
					for (uint8_t format = 0; format <= e_irt_crdf_fix16; format++) {
						oob_flag_v[format] = 0; oob_flag_h[format] = 0;
						switch (format) {
						case e_irt_crdf_arch:
							oob_flag_v[format] = yi_fix31 < 0 || yi_fix31 > (((int64_t)Hi - 1) << 31);
							oob_flag_h[format] = xi_fix31 < 0 || xi_fix31 > (((int64_t)Wi - 1) << 31);
							break;
						case e_irt_crdf_fpt64:
							oob_flag_v[format] = yi_fpt64 < 0 || yi_fpt64 > (double)Hi - 1;
							oob_flag_h[format] = xi_fpt64 < 0 || xi_fpt64 > (double)Wi - 1;
							break;
						case e_irt_crdf_fpt32:
							oob_flag_v[format] = yi_fpt32 < 0 || yi_fpt32 > (float)Hi - 1;
							oob_flag_h[format] = xi_fpt32 < 0 || xi_fpt32 > (float)Wi - 1;
							break;
						case e_irt_crdf_fixed:
							oob_flag_v[format] = yi_fixed < 0 || yi_fixed > (((int64_t)Hi - 1) << irt_top->rot_pars[desc].TOTAL_PREC);
							oob_flag_h[format] = xi_fixed < 0 || xi_fixed > (((int64_t)Wi - 1) << irt_top->rot_pars[desc].TOTAL_PREC);
							break;
						case e_irt_crdf_fix16:
							oob_flag_v[format] = yi_fix16 < 0 || yi_fix16 > (((int64_t)Hi - 1) << 16);
							oob_flag_h[format] = xi_fix16 < 0 || xi_fix16 > (((int64_t)Wi - 1) << 16);
							break;
						}
					}
#if 1
					xi_fix31 = IRT_top::IRT_UTILS::irt_min_int64(IRT_top::IRT_UTILS::irt_max_int64(0, xi_fix31), ((int64_t)Wi - 1) << 31);
					//yi_fix31 = IRT_top::IRT_UTILS::irt_min_int64(IRT_top::IRT_UTILS::irt_max_int64(0, yi_fix31), ((int64_t)Hi - 1) << 31);
					xi_fpt64 = std::fmin(std::fmax(0, xi_fpt64), ((double)Wi - 1));
					//yi_fpt64 = std::fmin(std::fmax(0, yi_fpt64), ((double)Hi - 1));
					xi_fpt32 = std::fmin(std::fmax((float)0, xi_fpt32), ((float)Wi - 1));
					//yi_fpt32 = std::fmin(std::fmax((float)0, yi_fpt32), ((float)Hi - 1));
					xi_fixed = IRT_top::IRT_UTILS::irt_min_int64(IRT_top::IRT_UTILS::irt_max_int64(0, xi_fixed), ((int64_t)Wi - 1) << irt_top->rot_pars[desc].TOTAL_PREC);
					//yi_fixed = IRT_top::IRT_UTILS::irt_min_int64(IRT_top::IRT_UTILS::irt_max_int64(0, yi_fixed), ((int64_t)Hi - 1) << irt_top->rot_pars[desc].TOTAL_PREC);
					xi_fix16 = std::min(std::max(0, xi_fix16), (Wi - 1) << 16);
					//yi_fix16 = std::min(std::max(0, yi_fix16), (Hi - 1) << 16);
#endif
				}
				uint8_t mask_notint_flag = 1;
#ifdef	IRT_USE_INTRP_REQ_CHECK
				mask_notint_flag = 0;
#endif

				j0[e_irt_crdf_arch] = (int16_t)(xi_fix31 >> 31);
				xf_arch  = (float)((double)((xi_fix31 - ((int64_t)j0[e_irt_crdf_arch] << 31)) / pow(2.0, 31)));
				xf_fix31 = (int)   (xi_fix31 - ((int64_t)j0[e_irt_crdf_arch] << 31));
				noint_flag_h[e_irt_crdf_arch] = mask_notint_flag ? 0 : xf_fix31 == 0;
				j1[e_irt_crdf_arch] = j0[e_irt_crdf_arch] + 1 - (oob_flag_h[e_irt_crdf_arch] || noint_flag_h[e_irt_crdf_arch]);

				i0[e_irt_crdf_arch] = (int16_t)(yi_fix31 >> 31);
				yf_arch = (float)((double)((yi_fix31 - ((int64_t)i0[e_irt_crdf_arch] << 31)) / pow(2.0, 31)));
				yf_fix31 = (int)  (yi_fix31 - ((int64_t)i0[e_irt_crdf_arch] << 31));
				noint_flag_v[e_irt_crdf_arch] = mask_notint_flag ? 0 : yf_fix31 == 0;
				i1[e_irt_crdf_arch] = i0[e_irt_crdf_arch] + 1 - (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0 || oob_flag_v[e_irt_crdf_arch] || noint_flag_v[e_irt_crdf_arch]);

				j0[e_irt_crdf_fpt64] = (int16_t)floor(xi_fpt64);
				xf_fpt64 = xi_fpt64 - (double)j0[e_irt_crdf_fpt64];
				noint_flag_h[e_irt_crdf_fpt64] = mask_notint_flag ? 0 : xf_fpt64 == 0;
				j1[e_irt_crdf_fpt64] = j0[e_irt_crdf_fpt64] + 1 - (oob_flag_h[e_irt_crdf_fpt64] || noint_flag_h[e_irt_crdf_fpt64]);

				i0[e_irt_crdf_fpt64] = (int16_t)floor(yi_fpt64);
				yf_fpt64 = yi_fpt64 - (double)i0[e_irt_crdf_fpt64];
				noint_flag_v[e_irt_crdf_fpt64] = mask_notint_flag ? 0 : yf_fpt64 == 0;
				i1[e_irt_crdf_fpt64] = i0[e_irt_crdf_fpt64] + 1 - (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0 || oob_flag_v[e_irt_crdf_fpt64] || noint_flag_v[e_irt_crdf_fpt64]);

				j0[e_irt_crdf_fpt32] = (int16_t)floor(xi_fpt32);
				xf_fpt32 = xi_fpt32 - (float)j0[e_irt_crdf_fpt32];
				noint_flag_h[e_irt_crdf_fpt32] = mask_notint_flag ? 0 : xf_fpt32 == 0;
				j1[e_irt_crdf_fpt32] = j0[e_irt_crdf_fpt32] + 1 - (oob_flag_h[e_irt_crdf_fpt32] || noint_flag_h[e_irt_crdf_fpt32]);

				i0[e_irt_crdf_fpt32] = (int16_t)floor(yi_fpt32);
				yf_fpt32 = yi_fpt32 - (float)i0[e_irt_crdf_fpt32];
				noint_flag_v[e_irt_crdf_fpt32] = mask_notint_flag ? 0 : yf_fpt32 == 0;
				i1[e_irt_crdf_fpt32] = i0[e_irt_crdf_fpt32] + 1 - (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0 || oob_flag_v[e_irt_crdf_fpt32] || noint_flag_v[e_irt_crdf_fpt32]);

				j0[e_irt_crdf_fixed] = (int16_t)(xi_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
				xf_fixed = (int)(xi_fixed - ((int64_t)j0[e_irt_crdf_fixed] << irt_top->rot_pars[desc].TOTAL_PREC));
				noint_flag_h[e_irt_crdf_fixed] = mask_notint_flag ? 0 : xf_fixed == 0;
				j1[e_irt_crdf_fixed] = j0[e_irt_crdf_fixed] + 1 - (oob_flag_h[e_irt_crdf_fixed] || noint_flag_h[e_irt_crdf_fixed]);

				i0[e_irt_crdf_fixed] = (int16_t)(yi_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
				yf_fixed = (int)(yi_fixed - ((int64_t)i0[e_irt_crdf_fixed] << irt_top->rot_pars[desc].TOTAL_PREC));
				noint_flag_v[e_irt_crdf_fixed] = mask_notint_flag ? 0 : yf_fixed == 0;
				i1[e_irt_crdf_fixed] = i0[e_irt_crdf_fixed] + 1 - (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0 || oob_flag_v[e_irt_crdf_fixed] || noint_flag_v[e_irt_crdf_fixed]);

				j0[e_irt_crdf_fix16] = xi_fix16 >> 16;
				xf_fix16 = (int)(xi_fix16 - ((uint32_t)j0[e_irt_crdf_fix16] << 16));
				noint_flag_h[e_irt_crdf_fix16] = mask_notint_flag ? 0 : xf_fix16 == 0;
				j1[e_irt_crdf_fix16] = j0[e_irt_crdf_fix16] + 1 - (oob_flag_h[e_irt_crdf_fix16] || noint_flag_h[e_irt_crdf_fix16]);

				i0[e_irt_crdf_fix16] = yi_fix16 >> 16;
				yf_fix16 = (int)(yi_fix16 - ((uint32_t)i0[e_irt_crdf_fix16] << 16));
				noint_flag_v[e_irt_crdf_fix16] = mask_notint_flag ? 0 : yf_fix16 == 0;
				i1[e_irt_crdf_fix16] = i0[e_irt_crdf_fix16] + 1 - (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0 || oob_flag_v[e_irt_crdf_fix16] || noint_flag_v[e_irt_crdf_fix16]);

				for (uint8_t format = 0; format <= e_irt_crdf_fix16; format++) {
					if (j0[e_irt_crdf_fpt64] != j0[format] || i0[e_irt_crdf_fpt64] != i0[format])
						coord_err[format]++;
				}

				//IRT_TRACE_DBG("yf_fix31 %d, yf_fpt64 %f, yf_fpt32 %f\n", yf_fix31, yf_fpt64, yf_fpt32);

				uint8_t i0_bank = ((i0[e_irt_crdf_arch] % 8) + 8) % 8; //i0_fixed can be negative, additional +8 and %8 required to get %8 value to be positive
				uint8_t i1_bank = ((i1[e_irt_crdf_arch] % 8) + 8) % 8;
				row_bank_access[i0_bank][0]++;
				row_bank_access[i0_bank][1] = IRT_top::IRT_UTILS::irt_min_int16(row_bank_access[i0_bank][1], j0[e_irt_crdf_fixed]);
				row_bank_access[i0_bank][2] = IRT_top::IRT_UTILS::irt_max_int16(row_bank_access[i0_bank][2], j1[e_irt_crdf_fixed]);
				row_bank_access[i0_bank][3] = IRT_top::IRT_UTILS::irt_min_int16(row_bank_access[i0_bank][3], j0[e_irt_crdf_fixed] >> 3);
				row_bank_access[i0_bank][4] = IRT_top::IRT_UTILS::irt_max_int16(row_bank_access[i0_bank][4], j1[e_irt_crdf_fixed] >> 3);

				row_bank_access[i1_bank][0]++;
				row_bank_access[i1_bank][1] = IRT_top::IRT_UTILS::irt_min_int16(row_bank_access[i1_bank][1], j0[e_irt_crdf_fixed]);
				row_bank_access[i1_bank][2] = IRT_top::IRT_UTILS::irt_max_int16(row_bank_access[i1_bank][2], j1[e_irt_crdf_fixed]);
				row_bank_access[i1_bank][3] = IRT_top::IRT_UTILS::irt_min_int16(row_bank_access[i1_bank][3], j0[e_irt_crdf_fixed] >> 3);
				row_bank_access[i1_bank][4] = IRT_top::IRT_UTILS::irt_max_int16(row_bank_access[i1_bank][4], j1[e_irt_crdf_fixed] >> 3);

				irt_cfifo_data_struct rot_mem_rd_ctrl, rot_mem_rd_ctrl1;

				rot_mem_rd_ctrl.YT = i0[e_irt_crdf_arch]; rot_mem_rd_ctrl.YB = i1[e_irt_crdf_arch]; rot_mem_rd_ctrl.XL = j0[e_irt_crdf_arch]; rot_mem_rd_ctrl.XR = j1[e_irt_crdf_arch];
				rot_mem_rd_ctrl.pixel = col; rot_mem_rd_ctrl.line = row;
				irt_top->mem_ctrl[e_irt_block_rot].start_line[desc] = 0;

				rot_mem_rd_ctrl1.YT = i0[e_irt_crdf_arch]; rot_mem_rd_ctrl1.YB = i1[e_irt_crdf_arch];
				rot_mem_rd_ctrl1.XL = j0[e_irt_crdf_arch]; rot_mem_rd_ctrl1.XR = j1[e_irt_crdf_arch];
				rot_mem_rd_ctrl1.pixel = col; rot_mem_rd_ctrl1.line = row;

				bool rd_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS] = { 0 }, rd_mode[IRT_ROT_MEM_ROW_BANKS] = { 0 }, msb_lsb_sel[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS] = { 0 }, bg_flag[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS][IRT_ROT_MEM_BANK_WIDTH] = { 0 };
				int16_t rd_addr[IRT_ROT_MEM_ROW_BANKS][IRT_ROT_MEM_COL_BANKS] = { 0 };
				uint8_t rd_shift[IRT_ROT_MEM_ROW_BANKS] = { 0 }, bank_row[IRT_ROT_MEM_ROW_BANKS] = { 0 };

				//IRT_TRACE("Pixel [%d][%d]: YT %d, YB %d, XL %d, XR %d\n", row, col + pixel, i0[e_irt_crdf_arch], i1[e_irt_crdf_arch], j0[e_irt_crdf_arch], j1[e_irt_crdf_arch]);

				irt_top->IRT_RMRM_block->run(j0[e_irt_crdf_arch], j1[e_irt_crdf_arch], i0[e_irt_crdf_arch], i1[e_irt_crdf_arch],
					bank_row, rd_sel, rd_addr, rd_mode,	rd_shift, msb_lsb_sel, bank_row[0], bg_flag, desc, 0, xo_i, yo_i, e_irt_rmrm_caller_hl_model);

				for (uint8_t idx = 0; idx < 2; idx++) {
					rot_mem_rd_ctrl.bank_row[idx] = bank_row[idx];
					for (uint8_t bank_col = 0; bank_col < IRT_ROT_MEM_COL_BANKS; bank_col++) {
						rot_mem_rd_ctrl.rd_addr[idx][bank_col] = rd_addr[bank_row[idx]][bank_col];
						rot_mem_rd_ctrl.rd_sel[idx][bank_col] = rd_sel[bank_row[idx]][bank_col];
					}
#if 0
					IRT_TRACE("Pixel [%d][%d]: idx %u, YT %d, YB %d, XL %d, XR %d, bank_row %u, rd_sel[%u][%u], rd_addr[%d][%d]\n", row, col + pixel, idx,
						i0[e_irt_crdf_arch], i1[e_irt_crdf_arch], j0[e_irt_crdf_arch], j1[e_irt_crdf_arch],
						bank_row[idx], rd_sel[bank_row[idx]][0], rd_sel[bank_row[idx]][1],
						rd_addr[bank_row[idx]][0], rd_addr[bank_row[idx]][0]);
#endif
				}

				irt_top->irt_rot_mem_rd_stat_calc(irt_top, rot_mem_rd_ctrl, rot_mem_rd_stat);

#if 0
				IRT_TRACE("HL model %s: [%d, %d]:\n", irt_irt_mode_s[transform_type], row, pixel);
				for (uint8_t row_bank = 0; row_bank < IRT_ROT_MEM_ROW_BANKS; row_bank++) {
					for (uint8_t col_bank = 0; col_bank < IRT_ROT_MEM_COL_BANKS; col_bank++) {
						IRT_TRACE("bank_num_read[%u][%u](%u) > 1 && bank_max_addr[%u][%u](%d) != bank_min_addr[%u][%u] (%d)\n",
							row_bank, col_bank, rot_mem_rd_stat.bank_num_read[row_bank][col_bank],
							row_bank, col_bank, rot_mem_rd_stat.bank_max_addr[row_bank][col_bank],
							row_bank, col_bank, rot_mem_rd_stat.bank_min_addr[row_bank][col_bank]);
					}
				}
#endif

				if (pixel == 0) { //1st pixel in group storing its corner, used in fixed point only
					xi0[e_irt_crdf_arch] = j0[e_irt_crdf_arch]; yi0[e_irt_crdf_arch] = i0[e_irt_crdf_arch];
				} else { //non 1st pixel, checking valid interpolation window
					XL_fixed = IRT_top::IRT_UTILS::irt_min_int16(xi0[e_irt_crdf_arch], j0[e_irt_crdf_arch]);
					XR_fixed = IRT_top::IRT_UTILS::irt_max_int16(xi0[e_irt_crdf_arch] + 1, j1[e_irt_crdf_arch]);
					YT_fixed = IRT_top::IRT_UTILS::irt_min_int16(yi0[e_irt_crdf_arch], i0[e_irt_crdf_arch]);
					YB_fixed = IRT_top::IRT_UTILS::irt_max_int16(yi0[e_irt_crdf_arch] + 1, i1[e_irt_crdf_arch]);

					if (irt_top->irt_desc[desc].rot90 && irt_top->irt_desc[desc].rot90_intv == 0)
						YB_fixed--;
#if 0
					if (pixel == adj_proc_size - 1)
						IRT_TRACE("HL model %s: output range[%d][%d:%d] and processing rate %d pixels/cycle require %dx%d window from input range [%d:%d][%d:%d]\n", irt_irt_mode_s[transform_type], row, col, col + pixel, adj_proc_size, YB_fixed - YT_fixed + 1, XR_fixed - XL_fixed + 1, YT_fixed, YB_fixed, XL_fixed, XR_fixed);
#endif
					if (hl_only != 1) {
						switch (irt_top->irt_cfg.flow_mode) {
						case e_irt_flow_nCFIFO_fixed_adaptive_wxh:
						case e_irt_flow_wCFIFO_fixed_adaptive_wxh:
							if (((YB_fixed - YT_fixed + 1) > IRT_INT_WIN_H) || ((XR_fixed - XL_fixed + 1) > IRT_INT_WIN_W)) { //dont fit interpolation window
								switch (irt_top->irt_desc[desc].rate_mode) {
								case e_irt_rate_adaptive_wxh:
									adj_proc_size = pixel; //set for number of already processed pixels in group
									rate_reach_flag = 1;
									//IRT_TRACE("HL model %s: for output range[%d][%d:%d] processing rate is limited to %d pixels/cycle\n", irt_irt_mode_s[transform_type], row, col, col + pixel, adj_proc_size);
									break;
								case e_irt_rate_fixed: //fixed, error if fixed does not feet window
									IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "HL model %s: output range[%d][%d:%d] does not fit input interpolation window 8x9 for %d processing rate, require %dx%d window from input range[%d:%d][%d:%d]\n", irt_irt_mode_s[transform_type], row, col, col + pixel, irt_top->irt_desc[desc].proc_size, YB_fixed - YT_fixed + 1, XR_fixed - XL_fixed + 1, YT_fixed, YB_fixed, XL_fixed, XR_fixed);
									IRT_TRACE_TO_RES(test_res, " failed, HL model %s: output range[%d][%d:%d] does not fit input interpolation window 8x9 for %d processing rate, require %dx%d window from input range[%d:%d][%d:%d]\n", irt_irt_mode_s[transform_type], row, col, col + pixel, irt_top->irt_desc[desc].proc_size, YB_fixed - YT_fixed + 1, XR_fixed - XL_fixed + 1, YT_fixed, YB_fixed, XL_fixed, XR_fixed);
									IRT_CLOSE_FAILED_TEST(0);
									break;
								case e_irt_rate_adaptive_2x2:
									IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "HL model %s: %s rate mode is not supported in %s flow\n", irt_irt_mode_s[transform_type], irt_rate_mode_s[irt_top->irt_desc[desc].rate_mode], irt_flow_mode_s[irt_top->irt_cfg.flow_mode]);
									IRT_TRACE_TO_RES(test_res, " failed, HL model %s: %s rate mode is not supported in %s flow\n", irt_irt_mode_s[transform_type], irt_rate_mode_s[irt_top->irt_desc[desc].rate_mode], irt_flow_mode_s[irt_top->irt_cfg.flow_mode]);
									IRT_CLOSE_FAILED_TEST(0);
									break;
								}
							}
							break;
						case e_irt_flow_wCFIFO_fixed_adaptive_2x2:
							switch (irt_top->irt_desc[desc].rate_mode) {
							case e_irt_rate_adaptive_wxh:
								IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "HL model %s: %s rate mode is not supported in %s flow\n", irt_irt_mode_s[transform_type], irt_rate_mode_s[irt_top->irt_desc[desc].rate_mode], irt_flow_mode_s[irt_top->irt_cfg.flow_mode]);
								IRT_TRACE_TO_RES(test_res, " failed, HL model %s: %s rate mode is not supported in %s flow\n", irt_irt_mode_s[transform_type], irt_rate_mode_s[irt_top->irt_desc[desc].rate_mode], irt_flow_mode_s[irt_top->irt_cfg.flow_mode]);
								IRT_CLOSE_FAILED_TEST(0);
								break;
							case e_irt_rate_fixed:
							case e_irt_rate_adaptive_2x2:
								for (uint8_t row_bank = 0; row_bank < IRT_ROT_MEM_ROW_BANKS; row_bank++) {
									for (uint8_t col_bank = 0; col_bank < IRT_ROT_MEM_COL_BANKS; col_bank++) {
										if (rot_mem_rd_stat.bank_num_read[row_bank][col_bank] > 1 /*bank is accessed more than once, can be conflict*/ &&
											rot_mem_rd_stat.bank_max_addr[row_bank][col_bank] != rot_mem_rd_stat.bank_min_addr[row_bank][col_bank] && // there is conflict on the bank because more than 1 different addresses are accessed
											rot_mem_rd_stat.bank_max_addr[row_bank][col_bank] != -1 && rot_mem_rd_stat.bank_min_addr[row_bank][col_bank] != 10000) {
											//adj_proc_size = pixel;//set for number of already processed pixels in group
											rate_reach_flag = 1;
#if 0
											IRT_TRACE("HL model %s: rate is reached at pixel %d, bank_num_read[%u][%u](%u) > 1 && bank_max_addr[%u][%u](%d) != bank_min_addr[%u][%u] (%d)\n",
												irt_irt_mode_s[transform_type], pixel, 
												row_bank, col_bank, rot_mem_rd_stat.bank_num_read[row_bank][col_bank],
												row_bank, col_bank, rot_mem_rd_stat.bank_max_addr[row_bank][col_bank],
												row_bank, col_bank, rot_mem_rd_stat.bank_min_addr[row_bank][col_bank]);
#endif
										}
									}
								}
							}
							break;
						}

						if (rate_reach_flag) {
							adj_proc_size = pixel;

							if (irt_top->irt_desc[desc].rate_mode == e_irt_rate_fixed && adj_proc_size < irt_top->irt_desc[desc].proc_size) {
								IRT_TRACE(IRT_TRACE_LEVEL_INFO, "HL model %s: output range[%d][%d:%d] does not fit %d processing rate, require %dx%d window from input range[%d:%d][%d:%d], maximum rate is %d\n",
									irt_irt_mode_s[transform_type], row, col, col + pixel, irt_top->irt_desc[desc].proc_size,
									YB_fixed - YT_fixed + 1, XR_fixed - XL_fixed + 1, YT_fixed, YB_fixed, XL_fixed, XR_fixed, adj_proc_size);
								IRT_TRACE_TO_RES(test_res, " failed, HL model %s: output range[%d][%d:%d] does not fit %d processing rate, require %dx%d window from input range[%d:%d][%d:%d], maximum rate is %d\n",
									irt_irt_mode_s[transform_type], row, col, col + pixel, irt_top->irt_desc[desc].proc_size,
									YB_fixed - YT_fixed + 1, XR_fixed - XL_fixed + 1, YT_fixed, YB_fixed, XL_fixed, XR_fixed, adj_proc_size);
								//IRT_CLOSE_FAILED_TEST(0);
							}

							adj_proc_size = (uint8_t)std::min((int)adj_proc_size, (int)irt_top->irt_desc[desc].proc_size);
#if 0
							IRT_TRACE("[row:col]=[%d:%d] , [YT:YB]=[%d:%d] , [XL:XR]=[%d:%d], Yi_start %d\n", row, col, YT_fixed, YB_fixed, XL_fixed, XR_fixed, Yi_start);
							IRT_TRACE("           |Xi_start| Y1 |  X1 |row_bank|X1_43|line_in_row|addr_off| addr|rd_addr[0]/[1]|rd_mode|XL_col|XR_col|rd_sel[0]/[1]|\n");
							for (uint8_t p = 0; p <= pixel; p++) {
								for (uint8_t idx = 0; idx <= 1; idx++) {
									IRT_TRACE("pix %d, Y %d |  %4d  | %2d | %d |   %d    |  %d  |     %d     |   %d    | %3d |   %d  /  %d  |   %d   |  %d   |  %d   |   %d  /  %d   |\n", p, idx,
										Xi_start[p][idx], Y1[p][idx], X1[p][idx], Y_row_bank[p][idx], X1_43[p][idx], line_in_bank_row[p][idx],
										addr_offset[p][idx], addr[p][idx], rd_addr[p][idx][0], rd_addr[p][idx][1], rd_mode[p][idx], XL_col[p][idx], XR_col[p][idx],
										rd_sel[p][idx][0], rd_sel[p][idx][1]);
								}
							}
							for (uint8_t row_bank = 0; row_bank < 8; row_bank++) {
								for (uint8_t col_bank = 0; col_bank < 2; col_bank++) {
									IRT_TRACE("row_bank = %d, col_bank %d |	%d	|	%d	|	%d	|\n", row_bank, col_bank,
										bank_num_read[1][row_bank][col_bank], bank_max_addr[1][row_bank][col_bank], bank_min_addr[1][row_bank][col_bank]);
								}
							}
#endif
							rate_reach_flag = 0;
						}
					}

					if (pixel >= adj_proc_size - 1) {
						//min_proc_rate = std::min(adj_proc_size, min_proc_rate); //if (adj_proc_size < min_proc_rate) min_proc_rate = adj_proc_size;
						//max_proc_rate = std::max(adj_proc_size, max_proc_rate); //if (adj_proc_size > max_proc_rate) max_proc_rate = adj_proc_size;
						rate_reach_flag = 0;
					}
					if (adj_proc_size < 1 || adj_proc_size > IRT_ROT_MAX_PROC_SIZE) {
						IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "HL model %s: output range[%d][%d] adj_proc_size %d is error\n", irt_irt_mode_s[transform_type], row, col, adj_proc_size);
						IRT_TRACE_TO_RES(test_res, " failed, HL model %s: output range[%d][%d] adj_proc_size %d is error\n", irt_irt_mode_s[transform_type], row, col, adj_proc_size);
						IRT_CLOSE_FAILED_TEST(0);
					}
				}

				//force weight to 1 and 0 for nearest neigbour interpolation
				if (irt_top->irt_desc[desc].int_mode) {

					if (xf_fix31 > (1 << 30))
						xf_fix31 = 0x80000000;
					else if (xf_fix31 < (1 << 30))
						xf_fix31 = 0;
					if (yf_fix31 > (1 << 30))
						yf_fix31 = 0x80000000;
					else if (yf_fix31 < (1 << 30))
						yf_fix31 = 0;

					if (xf_arch > 0.5)
						xf_arch = 1;
					else if (xf_arch < 0.5)
						xf_arch = 0;
					if (yf_arch > 0.5)
						yf_arch = 1;
					else if (yf_arch < 0.5)
						yf_arch = 0;

					if (xf_fpt64 > 0.5)
						xf_fpt64 = 1;
					else if (xf_fpt64 < 0.5)
						xf_fpt64 = 0;
					if (yf_fpt64 > 0.5)
						yf_fpt64 = 1;
					else if (yf_fpt64 < 0.5)
						yf_fpt64 = 0;

					if (xf_fpt32 > 0.5)
						xf_fpt32 = 1;
					else if (xf_fpt32 < 0.5)
						xf_fpt32 = 0;
					if (yf_fpt32 > 0.5)
						yf_fpt32 = 1;
					else if (yf_fpt32 < 0.5)
						yf_fpt32 = 0;

					if (xf_fixed > irt_top->rot_pars[desc].TOTAL_ROUND)
						xf_fixed = ((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC);
					else if (xf_fixed < irt_top->rot_pars[desc].TOTAL_ROUND)
						xf_fixed = 0;
					if (yf_fixed > irt_top->rot_pars[desc].TOTAL_ROUND)
						yf_fixed = ((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC);
					else if (yf_fixed < irt_top->rot_pars[desc].TOTAL_ROUND)
						yf_fixed = 0;

					if (xf_fix16 > (1 << 15))
						xf_fix16 = 1 << 16;
					else if (xf_fix16 < (1 << 15))
						xf_fix16 = 0;
					if (yf_fix16 > (1 << 15))
						yf_fix16 = 1 << 16;
					else if (yf_fix16 < (1 << 15))
						yf_fix16 = 0;

				}

				//IRT_TRACE_DBG("Before flip: [%u, %u]: [%d, %d] [%d, %d]\n", row, col + pixel, i0[e_irt_crdf_arch], i1[e_irt_crdf_arch], j0[e_irt_crdf_arch], j1[e_irt_crdf_arch]);
				//IRT_TRACE_DBG("                       [%d, %d] [%d, %d]\n", oob_flag_v[e_irt_crdf_arch], oob_flag_h[e_irt_crdf_arch], noint_flag_v[e_irt_crdf_arch], noint_flag_h[e_irt_crdf_arch]);

				//for (int format = 0; format <= e_irt_crdf_fix16; format++)
					//IRT_TRACE_DBG("Before vflip: format %d, [%u, %u]: [%d, %d] [%d, %d], oob_flag_v %d, noint_flag_v %d, oob_flag_h %d, noint_flag_h %d\n", 
					//	format, row, col + pixel, i0[format], i1[format], j0[format], j1[format], oob_flag_v[format], noint_flag_v[format], oob_flag_h[format], noint_flag_h[format]);
#ifndef IRT_HL_NO_FLIP
				for (int format = 0; format <= e_irt_crdf_fix16; format++) {
					if (irt_top->irt_desc[desc].read_vflip == 1) {
						i0[format] = Hi - 1 - i0[format];// +1;
#ifndef IRT_USE_FLIP_FOR_MINUS1
						i0[format] -= (oob_flag_v[format] == 0 && noint_flag_v[format] == 0);
#endif
						i1[format] = i0[format] + 1 - (oob_flag_v[format] || noint_flag_v[format]); //-1
						//i1[format] -= noint_flag_v[format];
					}
					if (irt_top->irt_desc[desc].read_hflip == 1) {
						j0[format] = Wi - 1 - j0[format];// +1;
#ifndef IRT_USE_FLIP_FOR_MINUS1
						j0[format] -= (oob_flag_h[format] == 0 && noint_flag_h[format] == 0);
#endif
						j1[format] = j0[format] + 1 - (oob_flag_h[format] || noint_flag_h[format]); //-1
						//j1[format] -= noint_flag_h[format];
					}

					//if (format == e_irt_crdf_arch)
					//IRT_TRACE_DBG("After flip: [%u, %u]: [%d, %d] [%d, %d]\n", row, col + pixel, i0[e_irt_crdf_arch], i1[e_irt_crdf_arch], j0[e_irt_crdf_arch], j1[e_irt_crdf_arch]);

					if (irt_top->irt_desc[desc].bg_mode == e_irt_bg_frame_repeat) {
						i0[format] = IRT_top::IRT_UTILS::irt_sat_int16(i0[format], 0, (int16_t)Hi - 1);
						i1[format] = IRT_top::IRT_UTILS::irt_sat_int16(i1[format], 0, (int16_t)Hi - 1);
						j0[format] = IRT_top::IRT_UTILS::irt_sat_int16(j0[format], 0, (int16_t)Wi - 1);
						j1[format] = IRT_top::IRT_UTILS::irt_sat_int16(j1[format], 0, (int16_t)Wi - 1);
					}
				}
				if (irt_top->irt_desc[desc].read_vflip == 1) {
#if 1//def IRT_USE_FLIP_FOR_MINUS1
					yf_fpt64 = 1 - yf_fpt64;
					yf_fpt32 = 1 - yf_fpt32;
					yf_fix31 = ((uint32_t)1 << 31) - yf_fix31;
					yf_fixed = ((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed;
					yf_fix16 = ((uint32_t)1 << 16) - yf_fix16;
#endif
				}
				if (irt_top->irt_desc[desc].read_hflip == 1) {
#if 1//def IRT_USE_FLIP_FOR_MINUS1
					xf_fpt64 = 1 - xf_fpt64;
					xf_fpt32 = 1 - xf_fpt32;
					xf_fix31 = ((uint32_t)1 << 31) - xf_fix31;
					xf_fixed = ((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed;
					xf_fix16 = ((uint32_t)1 << 16) - xf_fix16;
#endif
				}
#endif
				//for (int format = 0; format <= e_irt_crdf_fix16; format++)
				//IRT_TRACE_DBG("After vflip: format %d, [%u, %u]: [%d, %d] [%d, %d]\n", format, row, col + pixel, i0[format], i1[format], j0[format], j1[format]);
				//IRT_TRACE_DBG("After sat: [%u, %u]: [%d, %d] [%d, %d]\n", row, col + pixel, i0[e_irt_crdf_arch], i1[e_irt_crdf_arch], j0[e_irt_crdf_arch], j1[e_irt_crdf_arch]);
				
				for (plane = 0; plane < PLANES; plane++) {

					for (int format = 0; format <= e_irt_crdf_fix16; format++) {

						//check pixel inside of image boundary
						p_valid[0] = i0[format] >= 0 && i0[format] < Hi && j0[format] >= 0 && j0[format] < Wi;
						p_valid[1] = i0[format] >= 0 && i0[format] < Hi && j1[format] >= 0 && j1[format] < Wi;
						p_valid[2] = i1[format] >= 0 && i1[format] < Hi && j0[format] >= 0 && j0[format] < Wi;
						p_valid[3] = i1[format] >= 0 && i1[format] < Hi && j1[format] >= 0 && j1[format] < Wi;

						//assign pixel
						p_value[0] = (p_valid[0] ? input_image[plane][i0[format]][j0[format]] : 0) & irt_top->irt_desc[desc].Msi;
						p_value[1] = (p_valid[1] ? input_image[plane][i0[format]][j1[format]] : 0) & irt_top->irt_desc[desc].Msi;
						p_value[2] = (p_valid[2] ? input_image[plane][i1[format]][j0[format]] : 0) & irt_top->irt_desc[desc].Msi;
						p_value[3] = (p_valid[3] ? input_image[plane][i1[format]][j1[format]] : 0) & irt_top->irt_desc[desc].Msi;

						//cast pixel according to format
						for (int i = 0; i < 4; i++) {
							switch (format) {
							case e_irt_crdf_arch:  p_fpt32[i] = (float)p_value[i]; break;
							case e_irt_crdf_fpt64: p_fpt64[i] = (double)p_value[i]; break;
							case e_irt_crdf_fpt32: p_fpt32[i] = (float)p_value[i]; break;
							case e_irt_crdf_fixed: p_fixed[i] = (uint64_t)p_value[i]; break;
							case e_irt_crdf_fix16: p_fix16[i] = (uint64_t)p_value[i]; break;
							}
						}

						//calculate validity for interpolation
						i_valid[0] = i0[format] >= 0 && i1[format] < Hi && j0[format] >= 0 && j1[format] < Wi && irt_top->irt_desc[desc].read_hflip == 0 && irt_top->irt_desc[desc].read_vflip == 0;
						i_valid[1] = i0[format] >= 0 && i1[format] < Hi && j1[format] >= 0 && j0[format] < Wi && irt_top->irt_desc[desc].read_hflip == 1 && irt_top->irt_desc[desc].read_vflip == 0;
						i_valid[2] = i1[format] >= 0 && i0[format] < Hi && j0[format] >= 0 && j1[format] < Wi && irt_top->irt_desc[desc].read_hflip == 0 && irt_top->irt_desc[desc].read_vflip == 1;
						i_valid[3] = i1[format] >= 0 && i0[format] < Hi && j1[format] >= 0 && j0[format] < Wi && irt_top->irt_desc[desc].read_hflip == 1 && irt_top->irt_desc[desc].read_vflip == 1;

#ifndef IRT_HL_NO_FLIP
						i_valid[0] = p_valid[0] & p_valid[1] & p_valid[2] & p_valid[3];
						i_valid[1] = 0;
						i_valid[2] = 0;
						i_valid[3] = 0;
#endif
						//interpolation
						bool rot90_int = irt_top->irt_desc[desc].rot90_inth | irt_top->irt_desc[desc].rot90_intv;
						uint64_t out_fixed_intp1, out_fixed_other;
						if (i_valid[0] || i_valid[1] || i_valid[2] || i_valid[3] || p_valid[0] && irt_top->irt_desc[desc].rot90 && rot90_int == 0) {
							switch (format) {
							case e_irt_crdf_arch:
								out_fpt32 = p_fpt32[0] * (1 - xf_arch) * (1 - yf_arch) + p_fpt32[1] * xf_arch * (1 - yf_arch) + p_fpt32[2] * (1 - xf_arch) * yf_arch + p_fpt32[3] * xf_arch * yf_arch;

								xw0 = (float)(				  (double)xf_fix31  / pow(2.0, 31));
								xw1 = (float)((pow(2.0, 31) - (double)xf_fix31) / pow(2.0, 31));
								yw0 = (float)(				  (double)yf_fix31  / pow(2.0, 31));
								yw1 = (float)((pow(2.0, 31) - (double)yf_fix31) / pow(2.0, 31));
								out_fpt32 = p_fpt32[0] * xw1 * yw1 + p_fpt32[1] * xw0 * yw1 + p_fpt32[2] * xw1 * yw0 + p_fpt32[3] * xw0 * yw0;
								//horizontal followed by vertical
								out_fpt32 = (p_fpt32[0] * xw1 + p_fpt32[1] * xw0) * yw1 + (p_fpt32[2] * xw1 + p_fpt32[3] * xw0) * yw0;
								//vertical vertical by horizontal
								out_fpt32 = (p_fpt32[0] * yw1 + p_fpt32[2] * yw0) * xw1 + (p_fpt32[1] * yw1 + p_fpt32[3] * yw0) * xw0;

								out1[format] = (int)rint(out_fpt32 * pow(2, irt_top->irt_desc[desc].bli_shift));

								//IRT_TRACE_DBG(" [%f,%f,%f,%f]\n", p_fpt32[0], p_fpt32[1], p_fpt32[2], p_fpt32[3]);
								//out1[format] = (int)floor(out_fpt32 * pow(2.0, 9) + 0.5);
								//out1[format] = out1[format] >> (8 - (irt_top->rot_pars[desc].Pwo - irt_top->rot_pars[desc].Pwi - irt_top->rot_pars[desc].Ppi));
								//out1[format] = (out1[format] + 1) >> 1;
								//IRT_TRACE("BLI: Task %d [%d, %d]: pixels[0x%x][0x%x][0x%x][0x%x] w[%f][%f][%f][%f]\n", plane, row, col + pixel,
								//	p_value[0], p_value[1], p_value[2], p_value[3], xw0, xw1, yw0, yw1);

								break;
							case e_irt_crdf_fpt64:
								//horizontal followed by vertical
								out_fpt64 = p_fpt64[0] * (1 - xf_fpt64) * (1 - yf_fpt64) + p_fpt64[1] * xf_fpt64 * (1 - yf_fpt64) +
											p_fpt64[2] * (1 - xf_fpt64) * yf_fpt64 + p_fpt64[3] * xf_fpt64 * yf_fpt64;

								//vertical vertical by horizontal
								out_fpt64 = (p_fpt64[0] * (1 - yf_fpt64) + p_fpt64[2] * yf_fpt64) * (1 - xf_fpt64) +
											(p_fpt64[1] * (1 - yf_fpt64) + p_fpt64[3] * yf_fpt64) * xf_fpt64;

								out1[format] = (int)floor(out_fpt64 * pow(2.0, irt_top->rot_pars[desc].Pwo - irt_top->rot_pars[desc].Pwi) + 0.5);
								out1[format] = (int)rint(out_fpt64 * pow(2, irt_top->irt_desc[desc].bli_shift));
								break;
							case e_irt_crdf_fpt32:
								//horizontal followed by vertical
								out_fpt32 = p_fpt32[0] * (1 - xf_fpt32) * (1 - yf_fpt32) + p_fpt32[1] * xf_fpt32 * (1 - yf_fpt32) +
											p_fpt32[2] * (1 - xf_fpt32) * yf_fpt32 + p_fpt32[3] * xf_fpt32 * yf_fpt32;
								//vertical vertical by horizontal
								out_fpt32 = (p_fpt32[0] * (1 - yf_fpt32) + p_fpt32[2] * yf_fpt32) * (1 - xf_fpt32) +
											(p_fpt32[1] * (1 - yf_fpt32) + p_fpt32[3] * yf_fpt32) * xf_fpt32;

								out1[format] = (int)floor(out_fpt32 * pow(2.0, irt_top->rot_pars[desc].Pwo - irt_top->rot_pars[desc].Pwi) + 0.5);
								out1[format] = (int)rint(out_fpt32 * pow(2, irt_top->irt_desc[desc].bli_shift));
								break;
							case e_irt_crdf_fixed:
								//calculating weights with 2*IRT_COORD_CAL_PREC precision
								w[0] = (((int64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed) * (((int64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed);
								w[1] = xf_fixed * (((int64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed);
								w[2] = (((int64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed) * yf_fixed;
								w[3] = xf_fixed * yf_fixed;
								out_fixed = p_fixed[0] * w[0] + p_fixed[1] * w[1] + p_fixed[2] * w[2] + p_fixed[3] * w[3];
								out1[e_irt_crdf_fixed] = (int)(((out_fixed >> irt_top->rot_pars[desc].TOTAL_PREC) + irt_top->rot_pars[desc].TOTAL_ROUND) >> irt_top->rot_pars[desc].TOTAL_PREC);
								out1[e_irt_crdf_fixed] = (int)(((out_fixed >> (2 * irt_top->rot_pars[desc].TOTAL_PREC - (irt_top->rot_pars[desc].Pwo - irt_top->rot_pars[desc].Pwi) - 1)) + 1) >> 1);
								//printf("[%lld, %lld, %lld, %lld]\n",
								//	(irt_2_pow_calc_prec - (uint64_t)xf_fixed)*(irt_2_pow_calc_prec - (uint64_t)yf_fixed), ((uint64_t)xf_fixed)* (irt_2_pow_calc_prec - (uint64_t)yf_fixed),
								//	(irt_2_pow_calc_prec - (uint64_t)xf_fixed)* ((uint64_t)yf_fixed), ((uint64_t)xf_fixed)* ((uint64_t)yf_fixed));
								//printf("[%lld, %lld, %lld, %lld]\n", w0 << IRT_COORD_CAL_PREC, w1 << IRT_COORD_CAL_PREC, w2 << IRT_COORD_CAL_PREC, w3 << IRT_COORD_CAL_PREC);
								//printf("[%lld, ", out_fixed >> IRT_COORD_CAL_PREC);

								//calculating weights with 2*IRT_COORD_CAL_PREC - IRT_WEIGHT_PREC precision to save bits after that
								w[0] = ((((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed) * (((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed)) >> irt_top->rot_pars[desc].WEIGHT_PREC;
								w[1] = (xf_fixed * (((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed)) >> irt_top->rot_pars[desc].WEIGHT_PREC;
								w[2] = ((((uint64_t)1 << irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed) * yf_fixed) >> irt_top->rot_pars[desc].WEIGHT_PREC;
								w[3] = (xf_fixed * yf_fixed) >> irt_top->rot_pars[desc].WEIGHT_PREC;
								out_fixed = p_fixed[0] * w[0] + p_fixed[1] * w[1] + p_fixed[2] * w[2] + p_fixed[3] * w[3];
								out1[format] = (int)(((out_fixed >> (irt_top->rot_pars[desc].TOTAL_PREC - irt_top->rot_pars[desc].WEIGHT_PREC)) + irt_top->rot_pars[desc].TOTAL_ROUND) >> irt_top->rot_pars[desc].TOTAL_PREC);
								out1[format] = (int)(((out_fixed >> (irt_top->rot_pars[desc].bli_shift_fix - irt_top->rot_pars[desc].WEIGHT_PREC)) + 1) >> 1);
								//round to half nearest even, e.g. 2.5 -> 2, 3.5 -> 4. We check 2LSB after shift. b0.0 -> b0; 0.1 -> 0. ; 1.0 -> 1. ; 1.1 -> 1. +  1;
								out_fixed_intp1 = out_fixed >> (irt_top->rot_pars[desc].bli_shift_fix - irt_top->rot_pars[desc].WEIGHT_PREC);
								out_fixed_other = out_fixed - (out_fixed_intp1 << (irt_top->rot_pars[desc].bli_shift_fix - irt_top->rot_pars[desc].WEIGHT_PREC));
								if (out_fixed_other == 0) {// value is .5 or 0.0
									switch (out_fixed_intp1 & 0x3) {
									case 0: out1[format] = (int)(out_fixed_intp1 >> 1); break; //0.0 , truncate
									case 1: out1[format] = (int)(out_fixed_intp1 >> 1); break; //round half down to even
									case 2: out1[format] = (int)(out_fixed_intp1 >> 1); break; //0.0 , truncate
									case 3: out1[format] = (int)((out_fixed_intp1 >> 1) + 1); break;//round half up to even
									}
								} else {
									out1[format] = (int)((out_fixed_intp1 + 1) >> 1);
								}
#if 0
								switch (out_fixed & 0x3) {
								case 0: out1[format] = (int)(out_fixed >> 1); break;
								case 1: out1[format] = (int)(out_fixed >> 1); break; //round half down to even
								case 2: out1[format] = (int)(out_fixed >> 1); break;
								case 3: out1[format] = (int)((out_fixed >> 1) + 1); break;//round half up to even
								}
#endif
								//printf("%f -> %d\n", (float) out_fixed / 2.0, out1[format]);
								break;
							case e_irt_crdf_fix16:
								//calculating weights with 2*IRT_COORD_CAL_PREC precision
								w16[0] = (((uint64_t)1 << 16) - xf_fix16) * (((uint64_t)1 << 16) - yf_fix16);
								w16[1] = (uint64_t)xf_fix16 * (((uint64_t)1 << 16) - yf_fix16);
								w16[2] = (((uint64_t)1 << 16) - xf_fix16) * yf_fix16;
								w16[3] = (uint64_t)xf_fix16 * yf_fix16;
								out_fix16 = p_fix16[0] * w16[0] + p_fix16[1] * w16[1] + p_fix16[2] * w16[2] + p_fix16[3] * w16[3];
								out1[format] = (int)(((out_fix16 >> 16) + (1 << 15)) >> 16);
								out1[format] = (int)(((out_fix16 >> (2 * 16 - irt_top->irt_desc[desc].bli_shift - 1)) + 1) >> 1);
								out_fixed_intp1 = out_fix16 >> (2 * 16 - irt_top->irt_desc[desc].bli_shift - 1);
								out_fixed_other = out_fix16 - (out_fixed_intp1 << (2 * 16 - irt_top->irt_desc[desc].bli_shift - 1));
								if (out_fixed_other == 0) {// value is .5 or 0.0
									switch (out_fixed_intp1 & 0x3) {
									case 0: out1[format] = (int)(out_fixed_intp1 >> 1); break; //0.0 , truncate
									case 1: out1[format] = (int)(out_fixed_intp1 >> 1); break; //round half down to even
									case 2: out1[format] = (int)(out_fixed_intp1 >> 1); break; //0.0 , truncate
									case 3: out1[format] = (int)((out_fixed_intp1 >> 1) + 1); break;//round half up to even
									}
								} else {
									out1[format] = (int)((out_fixed_intp1 + 1) >> 1);
								}
								break;
							}

							switch (irt_top->irt_cfg.debug_mode) {
							case e_irt_debug_bg_out: out1[format] = 0; break;
							}
						} else {
							if (irt_top->irt_desc[desc].bg_mode == e_irt_bg_prog_value)
								out1[format] = (int)irt_top->irt_desc[desc].bg;
							else {
								int16_t i0_sat = IRT_top::IRT_UTILS::irt_min_int16(IRT_top::IRT_UTILS::irt_max_int16(0, i0[format]), Hi - 1);
								int16_t j0_sat = IRT_top::IRT_UTILS::irt_min_int16(IRT_top::IRT_UTILS::irt_max_int16(0, j0[format]), Wi - 1);
								out1[format] = (int)(input_image[plane][i0_sat][j0_sat] & irt_top->irt_desc[desc].Msi);
								out1[format] = (int)rint((float)out1[format] * pow(2, irt_top->irt_desc[desc].bli_shift));
							}
						}

						out1[format] = std::max(out1[format], 0); //if (out1_fpt32 < 0) out1_fpt32 = 0;
						out1[format] = std::min(out1[format], (int)irt_top->irt_desc[desc].MAX_VALo); //if (out1_fpt32 > 255) out1_fpt32 = 255;
						output_image[image * CALC_FORMATS_ROT + format][plane][row][col + pixel] = (uint16_t)(out1[format] << irt_top->irt_desc[desc].Ppo);
					}
				}
			}
			acc_proc_rate += adj_proc_size;
#if 0
			if (col == 0)
				IRT_TRACE("\nAcc avg rate: line %2d:", row);
			IRT_TRACE("%.1f ", (double)acc_proc_rate / cycles);


			if (col==0)
				IRT_TRACE("\nLocal rate: line %2d:", row);
			IRT_TRACE("%d ", adj_proc_size);
#endif
			min_proc_rate = (uint8_t)std::min((int)adj_proc_size, (int)min_proc_rate); //if (adj_proc_size < min_proc_rate) min_proc_rate = adj_proc_size;
			max_proc_rate = (uint8_t)std::max((int)adj_proc_size, (int)max_proc_rate); //if (adj_proc_size > max_proc_rate) max_proc_rate = adj_proc_size;
			rate_hist[adj_proc_size]++;
		}
	}

#if 0
	if (transform_type == projection) {
		IRT_TRACE("Projection erros\n");
		//IRT_TRACE("		%f	%f\n", proj_coord_err[0], proj_coord_err[1]);
		for (int i = 0; i < 5; i++) {
			IRT_TRACE("Projection erros for approximation type %d\n", i);
			IRT_TRACE("Error		X	Y\n");
			for (int j = 0; j < PROJ_ERR_BINS; j++) {
				IRT_TRACE("%f	%5d(%2.0f%%)	%5d(%2.0f%%)\n", (double)j / PROJ_ERR_BINS, proj_coord_err_hist[i][0][j], 100.0 * proj_coord_err_hist[i][0][j] / (Wo * Ho), proj_coord_err_hist[i][1][j], 100.0 * proj_coord_err_hist[i][1][j] / (Wo * Ho));
			}
		}
	}
#endif

	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "HL model %s processing rate statistics\n", irt_irt_mode_s[transform_type]);
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Min/Max/Avg: %d/%d/%3.2f\n", min_proc_rate, max_proc_rate, (double)acc_proc_rate/cycles);
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Min/Max/Avg: ");
	for (int bin = 1; bin < IRT_ROT_MAX_PROC_SIZE + 1; bin++) {
		if (rate_hist[bin] != 0) {
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%d/", bin);
			break;
		}
	}
	for (int bin = 8; bin > 0; bin--) {
		if (rate_hist[bin] != 0) {
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%d/", bin);
			break;
		}
	}
	acc_proc_rate = 0;
	int bin_sum = 0;
	for (int bin = 1; bin < 9; bin++) {
		acc_proc_rate += (bin * rate_hist[bin]);
		bin_sum+=rate_hist[bin];
	}
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%3.2f\n", (double)acc_proc_rate/bin_sum);
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Bin:    1    2    3    4    5    6    7    8\n");
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Val:");
	for (int bin = 1; bin < 9; bin++) {
		IRT_TRACE(IRT_TRACE_LEVEL_INFO, " %4d", rate_hist[bin]);
	}
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "\n");

	int n;
	if (print_out_files) {
		switch (transform_type) {
		case e_irt_rotation:
			for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fix16; format_type++) {
				if (num_of_images > 1) {
					n = sprintf(out_file_name, "%s%s_%d_rot_%s.bmp", outfiledir, outfilename, image, irt_hl_format_s[format_type]);
				} else {
					n = sprintf(out_file_name, "%s%s_rot_%s.bmp", outfiledir, outfilename, irt_hl_format_s[format_type]);
				}
				WriteBMP(irt_top, image, PLANES, out_file_name, So, Ho, output_image[image * CALC_FORMATS_ROT + format_type]);
			}
			break;
		case e_irt_affine:
			n = sprintf(out_file_name, "%s%s_%d_aff_%s_angle_%3.2f_Sx_%3.2f_Sy_%3.2f.bmp",
				outfiledir, outfilename, image, irt_top->rot_pars[desc].affine_mode, irt_top->rot_pars[desc].irt_angles[e_irt_angle_rot], irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy);
			//		WriteBMP(image, out_file_name, Wo, Ho);
			for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fixed; format_type++) {
				if (num_of_images > 1) {
					n = sprintf(out_file_name, "%s%s_%d_aff_%s.bmp", outfiledir, outfilename, image, irt_hl_format_s[format_type]);
				} else {
					n = sprintf(out_file_name, "%s%s_aff_%s.bmp", outfiledir, outfilename, irt_hl_format_s[format_type]);
				}
				WriteBMP(irt_top, image, PLANES, out_file_name, So, Ho, output_image[image * CALC_FORMATS_ROT + format_type]);
			}
			break;
		case e_irt_projection:
			n = sprintf(out_file_name, "%s%s_%d_prj_%s_roll_%3.2f_pitch_%3.2f_yaw_%3.2f_Sx_%3.2f_Sy_%3.2f_Zd_%3.2f.bmp",
				outfiledir, outfilename, image, irt_proj_mode_s[irt_top->rot_pars[desc].proj_mode], irt_top->rot_pars[desc].irt_angles[e_irt_angle_roll], irt_top->rot_pars[desc].irt_angles[e_irt_angle_pitch], irt_top->rot_pars[desc].irt_angles[e_irt_angle_yaw], irt_top->rot_pars[desc].Sx, irt_top->rot_pars[desc].Sy, irt_top->rot_pars[desc].proj_Zd);
			//		WriteBMP(image, out_file_name, Wo, Ho);
			for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fixed; format_type++) {
				if (num_of_images > 1) {
					n = sprintf(out_file_name, "%s%s_%d_prj_%s.bmp", outfiledir, outfilename, image, irt_hl_format_s[format_type]);
				} else {
					n = sprintf(out_file_name, "%s%s_prj_%s.bmp", outfiledir, outfilename, irt_hl_format_s[format_type]);
				}
				WriteBMP(irt_top, image, PLANES, out_file_name, So, Ho, output_image[image * CALC_FORMATS_ROT + format_type]);
			}
			break;
		case e_irt_mesh:
			for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fixed; format_type++) {
				if (num_of_images > 1) {
					n = sprintf(out_file_name, "%s%s_%d_mesh_%s.bmp", outfiledir, outfilename, image, irt_hl_format_s[format_type]);
				} else {
					n = sprintf(out_file_name, "%s%s_mesh_%s.bmp", outfiledir, outfilename, irt_hl_format_s[format_type]);
				}
				WriteBMP(irt_top, image, PLANES, out_file_name, So, Ho, output_image[image * CALC_FORMATS_ROT + format_type]);
			}
			break;
		}

		if (num_of_images > 1) {
			generate_image_dump("IRT_dump_arch_image%d_plane%d.txt", outfiledir, So, Ho, irt_bmp_wr_order, output_image[image * CALC_FORMATS_ROT + e_irt_crdf_arch], image); //output_image stored as processed
		} else {
			generate_image_dump("IRT_dump_arch_plane%d.txt", outfiledir, So, Ho, irt_bmp_wr_order, output_image[image * CALC_FORMATS_ROT + e_irt_crdf_arch], image); //output_image stored as processed
		}

	}

#if 0
	if (transform_type == projection) {
		//rotation from input to output
		for (int row = 0; row < Ho; row++)
			for (int col = 0; col < Wo; col++)
				for (plane = 0; plane < 3; plane++)
					output_image[image][plane][row][col] = 0;

		for (int row = 0; row < Hi; row++) {
			for (int col = 0; col < Wi; col++) {
				xi_fpt32 = (double)(col - Xic);
				yi_fpt32 = (double)(row - Yic);

				double xyzi[3], xyzo[3];
				xyzi[0] = xi_fpt32 * irt_top->rot_pars[desc].Sx;
				xyzi[1] = yi_fpt32 * irt_top->rot_pars[desc].Sy;
				xyzi[2] = irt_top->rot_pars[desc].proj_Wd;
				for (int r = 0; r < 3; r++) {
					xyzo[r] = irt_top->rot_pars[desc].proj_T[r];
					for (int idx = 0; idx < 3; idx++)
						xyzo[r] += irt_top->rot_pars[desc].proj_R[r][idx] * xyzi[idx];
				}
				if (xyzo[0] == 0)
					xo = Xoc;
				else
					xo = (int) (irt_top->rot_pars[desc].proj_Zd * xyzo[0] / xyzo[2]) + Xoc;
				if (xyzo[1] == 0)
					yo = Yoc;
				else
					yo = (int) (irt_top->rot_pars[desc].proj_Zd * xyzo[1] / xyzo[2]) + Yoc;


				xi0[e_irt_crdf_fpt32] = (int)floor(xo);
				yi0[e_irt_crdf_fpt32] = (int)floor(yo);

				for (plane = 0; plane < 3; plane++) {
					if ((xi0[e_irt_crdf_fpt32] >= 0) && (xi0[e_irt_crdf_fpt32] < Wo) && (yi0[e_irt_crdf_fpt32] >= 0) && (yi0[e_irt_crdf_fpt32] < Ho))
						output_image[image][plane][yi0[e_irt_crdf_fpt32]][xi0[e_irt_crdf_fpt32]] = input_image[plane][row][col];
				}
			}
		}

		n = sprintf(out_file_name, "%s_%d_prj2.bmp", file_name, image);
		WriteBMP(image, out_file_name, Wo, Ho);

		//direct calculation to sanety check
		for (int row = 0; row < Ho; row++) {
			for (int col = 0; col < Wo; col++) {
				xo = col - Xoc;
				yo = row - Yoc;

				double A0 = irt_top->rot_pars[desc].proj_R[0][0];
				double A1 = irt_top->rot_pars[desc].proj_R[0][1];
				double A2 = 0;//irt_top->rot_pars[desc].proj_R[0][2] * irt_top->rot_pars[desc].proj_Wd;
				double C0 = irt_top->rot_pars[desc].proj_R[2][0];
				double C1 = irt_top->rot_pars[desc].proj_R[2][1];
				double C2 = irt_top->rot_pars[desc].proj_R[2][2] * irt_top->rot_pars[desc].proj_Wd;
				double B0 = irt_top->rot_pars[desc].proj_R[1][0];
				double B1 = irt_top->rot_pars[desc].proj_R[1][1];
				double B2 = 0;//irt_top->rot_pars[desc].proj_R[1][2] * irt_top->rot_pars[desc].proj_Wd;

				double A = (xo + A2) * C0 - irt_top->rot_pars[desc].proj_Zd * A0; //(Xo-Xoc)*R31 - ZdR11
				double B = (xo + A2) * C1 - irt_top->rot_pars[desc].proj_Zd * A1; //(Xo-Xoc)*R32 - ZdR12
				double C = (xo + A2) * (C0 * Xic + C1 * Yic - C2) - irt_top->rot_pars[desc].proj_Zd * (A0 * Xic + A1 * Yic - A2); //(Xo-Xoc)*K - Zd*L1

				double D = (yo + B2) * C0 - irt_top->rot_pars[desc].proj_Zd * B0; //(Yo-Yoc)*R31 - ZdR21
				double E = (yo + B2) * C1 - irt_top->rot_pars[desc].proj_Zd * B1; //(Yo-Yoc)*R32 - ZdR22
				double F = (yo + B2) * (C0 * Xic + C1 * Yic - C2) - irt_top->rot_pars[desc].proj_Zd * (B0 * Xic + B1 * Yic - B2); //(Yo-Yoc)*K - Zd*L2

				//| A B | C
				//| D E | F
				xi_fpt32 = (C * E - B * F) / (A * E - D * B);

				xi_fpt32 = (xi_fpt32 - (double)Xic) / (double)irt_top->rot_pars[desc].Sx + (double)Xic;
				// (C * E - B * F)
				//((Xo-Xoc)*K - Zd*L1)((Yo-Yoc)*R32 - ZdR22)-((Xo-Xoc)*R32 - ZdR12)((Yo-Yoc)*K - Zd*L2) =
				// (Xo-Xoc)*K*(Yo-Yoc)*R32-Zd*L1(Yo-Yoc)*R32-(Xo-Xoc)*KZdR22+Zd^2*L1*R22 -
				// (Xo-Xoc)*R32(Yo-Yoc)*K- ZdR12(Yo-Yoc)*K-(Xo-Xoc)*R32Zd*L2+Zd^2*R12*L2 =
				// Zd(L2*R32-K*R22)(Xo-Xoc)+Zd(K*R12-L1*R32)(Yo-Yoc)+Zd^2*(L1*R22-L2*R12)

				//
				//(A * E - D * B)
				//(((Xo-Xoc)*R31 - ZdR11) * ((Yo-Yoc)*R32 - ZdR22) - ((Yo-Yoc)*R31 - ZdR21) * ((Xo-Xoc)*R32 - ZdR12))
				//(Xo-Xoc)*R31*(Yo-Yoc)*R32 - ZdR11(Yo-Yoc)*R32-(Xo-Xoc)*R31ZdR22+ZdR11ZdR22 -
				//(Yo-Yoc)*R31(Xo-Xoc)*R32 - (Yo-Yoc)*R31ZdR12 - ZdR21(Xo-Xoc)*R32 + ZdR21ZdR12
				// Zd(Xo-Xoc)(R21*R32-R22*R31)+Zd(Yo-Yoc)*(R31*R12-R32*R11)+Zd^2(R11*R22-R12*R21)
				// xi = (Zd(L2*R32-K*R22)(Xo-Xoc)+Zd(K*R12-L1*R32)(Yo-Yoc)+Zd^2*L*(L1*R22-L2*R12))/(Zd(R21*R32-R22*R31)(Xo-Xoc)+Zd(R31*R12-R32*R11)(Yo-Yoc)+Zd^2(R11*R22-R12*R21))
				// A0 = Zd((R21*Xic+R22*Yic)*R32-(R31*Xic+R32*Yic-R33*Wd)*R22), A1 = Zd((R31*Xic+R32*Yic-R33*Wd)*R12-(R11*Xic+R12*Yic)*R32), A2 = Zd^2*((R11*Xic+R12*Yic)*R22-(R21*Xic+R22*Yic)*R12)
				// C0 = Zd(R21*R32-R22*R31),								    C1 = Zd(R31*R12-R32*R11),								     C2 = Zd^2(R11*R22-R12*R21)


				yi_fpt32 = (A * F - C * D) / (A * E - D * B);
				yi_fpt32 = (yi_fpt32 - (double)Yic) / (double)irt_top->rot_pars[desc].Sy + (double)Yic;
				//(A * F - C * D)
				//(((Xo-Xoc)*R31 - ZdR11))((Yo-Yoc)*K - Zd*L2)-((Xo-Xoc)*K - Zd*L1)((Yo-Yoc)*R31 - ZdR21)
				//(Xo-Xoc)*R31(Yo-Yoc)*K-ZdR11(Yo-Yoc)*K-(Xo-Xoc)*R31Zd*L2+ZdR11Zd*L2 -
				//(Xo-Xoc)*K(Yo-Yoc)*R31-Zd*L1*(Yo-Yoc)*R31-(Xo-Xoc)*KZdR21+Zd*L1*ZdR21
				//yi=(Zd(K*R21-R31*L2)(Xo-Xoc)+Zd(R31*L1-K*R11)(Yo-Yoc)+Zd^2(R11*L2-L1*R21))/(Zd(R21*R32-R22*R31)(Xo-Xoc)+Zd(R31*R12-R32*R11)(Yo-Yoc)+Zd^2(R11*R22-R12*R21))
				//B0 = Zd((R31*Xic+R32*Yic-R33*Wd)*R21-(R21*Xic+R22*Yic)*R31), B1 = Zd((R11*Xic+R12*Yic)*R31-(R31*Xic+R32*Yic)*R11), B2 = Zd^2((R21*Xic+R22*Yic)*R11-(R11*Xic+R12*Yic)*R21)



				//xo = 0, yo = 0
				//A = - irt_top->rot_pars[desc].proj_Zd * irt_top->rot_pars[desc].proj_R[0][0]
				//B = - irt_top->rot_pars[desc].proj_Zd * irt_top->rot_pars[desc].proj_R[0][1];
				//C = - irt_top->rot_pars[desc].proj_Zd * (irt_top->rot_pars[desc].proj_R[0][0] * Xic + irt_top->rot_pars[desc].proj_R[0][1] * Yic + irt_top->rot_pars[desc].proj_R[0][2]*irt_top->rot_pars[desc].proj_Wd)
				//D = - irt_top->rot_pars[desc].proj_Zd * irt_top->rot_pars[desc].proj_R[1][0]
				//E = - irt_top->rot_pars[desc].proj_Zd * irt_top->rot_pars[desc].proj_R[1][1]
				//F = - irt_top->rot_pars[desc].proj_Zd * (irt_top->rot_pars[desc].proj_R[1][0] * Xic + irt_top->rot_pars[desc].proj_R[1][1] * Yic + irt_top->rot_pars[desc].proj_R[1][2]*irt_top->rot_pars[desc].proj_Wd)
				//xi * R00 + yi * R01 = R00*Xic + R01*Yic - R02*Wd
				//xi * R10 + yi * R11 = R10*Xic + R11*Yic - R12*Wd

				xi0[e_irt_crdf_fpt32] = (int)floor(xi_fpt32);
				yi0[e_irt_crdf_fpt32] = (int)floor(yi_fpt32);
				xf_fpt32 = xi_fpt32 - (double)xi0[e_irt_crdf_fpt32];
				yf_fpt32 = yi_fpt32 - (double)yi0[e_irt_crdf_fpt32];

				//force weight to 1 and 0 for nearest neigbour interpolation
				if (irt_top->irt_desc[desc].int_mode) {
					if (xf_fpt32 > 0.5)
						xf_fpt32 = 1;
					else if (xf_fpt32 < 0.5)
						xf_fpt32 = 0;
					if (yf_fpt32 > 0.5)
						yf_fpt32 = 1;
					else if (yf_fpt32 < 0.5)
						yf_fpt32 = 0;
				}

				for (plane = 0; plane < 3; plane++) {
					i0[e_irt_crdf_fpt32] = yi0[e_irt_crdf_fpt32];
					i1[e_irt_crdf_fpt32] = yi0[e_irt_crdf_fpt32] + 1;
					j0[e_irt_crdf_fpt32] = xi0[e_irt_crdf_fpt32];
					j1[e_irt_crdf_fpt32] = xi0[e_irt_crdf_fpt32] + 1;
					char p0 = (i0[e_irt_crdf_fpt32] >= 0 && i0[e_irt_crdf_fpt32] < IMAGE_H && j0[e_irt_crdf_fpt32] >= 0 && j0[e_irt_crdf_fpt32] < IMAGE_W) ? input_image[plane][i0[e_irt_crdf_fpt32]][j0[e_irt_crdf_fpt32]] : 0;
					char p1 = (i0[e_irt_crdf_fpt32] >= 0 && i0[e_irt_crdf_fpt32] < IMAGE_H && j1[e_irt_crdf_fpt32] >= 0 && j1[e_irt_crdf_fpt32] < IMAGE_W) ? input_image[plane][i0[e_irt_crdf_fpt32]][j1[e_irt_crdf_fpt32]] : 0;
					char p2 = (i1[e_irt_crdf_fpt32] >= 0 && i1[e_irt_crdf_fpt32] < IMAGE_H && j0[e_irt_crdf_fpt32] >= 0 && j0[e_irt_crdf_fpt32] < IMAGE_W) ? input_image[plane][i1[e_irt_crdf_fpt32]][j0[e_irt_crdf_fpt32]] : 0;
					char p3 = (i1[e_irt_crdf_fpt32] >= 0 && i1[e_irt_crdf_fpt32] < IMAGE_H && j1[e_irt_crdf_fpt32] >= 0 && j1[e_irt_crdf_fpt32] < IMAGE_W) ? input_image[plane][i1[e_irt_crdf_fpt32]][j1[e_irt_crdf_fpt32]] : 0;

					if ((j0[e_irt_crdf_fpt32] >= 0) && (j1[e_irt_crdf_fpt32] < Wi) && (i0[e_irt_crdf_fpt32] >= 0) && (i1[e_irt_crdf_fpt32] < Hi)) {
						out_fpt32 = (uint32_t)((uint8_t)p0) * (1 - xf_fpt32) * (1 - yf_fpt32) + (uint32_t)((uint8_t)p1) * (xf_fpt32) * (1 - yf_fpt32) +
							(uint32_t)((uint8_t)p2) * (1 - xf_fpt32) * (yf_fpt32)+(uint32_t)((uint8_t)p3) * (xf_fpt32) * (yf_fpt32);
					}
					else
						out_fpt32 = 0;

					out1_fpt32 = (int)floor(out_fpt32 + 0.5);
					if (out1_fpt32 < 0) out1_fpt32 = 0;
					if (out1_fpt32 > 255) out1_fpt32 = 255;
					output_image[image][plane][row][col] = (uint8_t)out1_fpt32;
				}
			}
		}

		n = sprintf(out_file_name, "%s_%d_prj3.bmp", file_name, image);
		WriteBMP(image, out_file_name, Wo, Ho);
	}
#endif
}

void IRThl_fpt32 (IRT_top* irt_top, uint8_t image, uint8_t planes, char file_name[50], double D, bool input_hflip, bool input_vflip, int suffix) {

	uint8_t desc = image * planes;

	uint16_t Ho = irt_top->irt_desc[desc].image_par[OIMAGE].H;
	uint16_t Wo = irt_top->irt_desc[desc].image_par[OIMAGE].W;
	double Xoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Xc / 2;
	double Yoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Yc / 2;
	uint16_t Hi = irt_top->irt_desc[desc].image_par[IIMAGE].H;
	uint16_t Wi = irt_top->irt_desc[desc].image_par[IIMAGE].W;
	double Xic = (double)irt_top->irt_desc[desc].image_par[IIMAGE].Xc / 2;
	double Yic = (double)irt_top->irt_desc[desc].image_par[IIMAGE].Yc / 2;

	char out_file_name[50];
	double cosD = cos(D * M_PI / 180), sinD = sin(D * M_PI / 180);
	double out_val = 0, xi = 0, yi = 0, xf = 0, yf = 0;
	double xo = 0, yo = 0;
	int xi0 = 0, yi0 = 0, out1 = 0;

	FILE* hl_out_file;
	if (num_of_images > 1) {
		sprintf(out_file_name, "%s_%d_rot%s.txt", file_name, image, suffix == 0 ? "" : "_flip");
	} else {
		sprintf(out_file_name, "%s_rot%s.txt", file_name, suffix == 0 ? "" : "_flip");
	}
	hl_out_file = fopen(out_file_name, "w");

	printf("Running IRThl_fpt32 at %s mode\n", irt_irt_mode_s[irt_top->irt_desc[desc].irt_mode]);

	for (uint16_t row = 0; row < Ho; row++)
		for (uint16_t col = 0; col < Wo; col++) {
			xo = (double)col - Xoc;
			yo = (double)row - Yoc;

			switch (irt_top->irt_desc[desc].irt_mode) {
			case e_irt_rotation:
				xi = cosD * xo + sinD * yo + Xic;
				yi = -sinD * xo + cosD * yo + Yic;
				break;
			case e_irt_affine:
				xi = irt_top->rot_pars[desc].M11d * xo + irt_top->rot_pars[desc].M12d * yo + Xic;
				yi = irt_top->rot_pars[desc].M21d * xo + irt_top->rot_pars[desc].M22d * yo + Yic;
				break;
			}

			xi0 = (int) floor (xi);
			yi0 = (int) floor (yi);
			xf  = xi - (double) xi0;
			yf  = yi - (double) yi0;

			//force weight to 1 and 0 for nearest neigbour interpolation
			if (irt_top->irt_desc[desc].int_mode) {
				if (xf > 0.5)
					xf = 1;
				else if (xf < 0.5)
					xf = 0;
				if (yf > 0.5)
					yf = 1;
				else if (yf < 0.5)
					yf = 0;
			}

			int i0 = 0, i1 = 0, j0 = 0, j1 = 0, xoff = 0, yoff = 0;
			double wx = 0, wy = 0;

			if (input_vflip == 0) {
				i0 = yi0;
				i1 = yi0 + 1;
				wy = yf;
			} else {
				yoff = Hi - 1;
				i0 = yoff - yi0;
				i1 = yoff - yi0 + 1;
				wy = 1 - yf;
			}

			if (input_hflip == 0) {
				j0 = xi0;
				j1 = xi0 + 1;
				wx = xf;
			} else {
				xoff = Wi - 1;
				j0 = xoff - xi0;
				j1 = xoff - xi0 + 1;
				wx = 1 - xf;
			}

			for (uint8_t plane = 0; plane < 3; plane++) {
				uint16_t p0 = (i0 >= 0 && i0 < Hi && j0 >= 0 && j0 < Wi) ? input_image[plane][i0][j0] : 0;
				uint16_t p1 = (i0 >= 0 && i0 < Hi && j1 >= 0 && j1 < Wi) ? input_image[plane][i0][j1] : 0;
				uint16_t p2 = (i1 >= 0 && i1 < Hi && j0 >= 0 && j0 < Wi) ? input_image[plane][i1][j0] : 0;
				uint16_t p3 = (i1 >= 0 && i1 < Hi && j1 >= 0 && j1 < Wi) ? input_image[plane][i1][j1] : 0;

				if (input_hflip == 0 &&        xi0 >= 0 &&        xi0 + 1 < Wi && input_vflip == 0 &&		 yi0 >= 0 &&		yi0 + 1 < Hi ||
					input_hflip == 1 && xoff - xi0 >= 0 && xoff - xi0 + 1 < Wi && input_vflip == 0 &&		 yi0 >= 0 &&		yi0 + 1 < Hi ||
					input_hflip == 0 &&		   xi0 >= 0 &&		  xi0 + 1 < Wi && input_vflip == 1 && yoff - yi0 >= 0 && yoff - yi0 + 1 < Hi ||
					input_hflip == 1 && xoff - xi0 >= 0 && xoff - xi0 + 1 < Wi && input_vflip == 1 && yoff - yi0 >= 0 && yoff - yi0 + 1 < Hi) {
					out_val = (uint32_t)((uint8_t)p0) * (1 - wx) * (1 - wy) + (uint32_t)((uint8_t)p1) * wx * (1 - wy) +
						(uint32_t)((uint8_t)p2) * (1 - wx) * wy + (uint32_t)((uint8_t)p3) * wx * wy;

					if (plane == 0)
						fprintf(hl_out_file, "[yo][xo]=[%d][%d]: [yi0][xi0]=[%d][%d], [yf][xf]=[%f][%f], out %f\n", row, col,
							input_vflip ? yoff - yi0 : yi0, input_hflip ? xoff - xi0 : xi0, input_vflip ? 1 - yf : yf, input_hflip ? 1 - xf : xf, out_val);
				} else
					out_val = 0;

				out1 = (int) floor(out_val+0.5);
				if (out1 < 0) out1 = 0;
				if (out1 > 255) out1 = 255;
				output_image[image][plane][row][col] = (uint8_t) out1;
			}
		}

	if (num_of_images > 1) {
		sprintf(out_file_name, "%s_%d_rot%s.bmp", file_name, image, suffix == 0 ? "" : "_flip");
	} else {
		sprintf(out_file_name, "%s_rot%s.bmp", file_name, suffix == 0 ? "" : "_flip");
	}
		
	WriteBMP(irt_top, image, desc, out_file_name, Wo, Ho, output_image[image]);
	fclose (hl_out_file);
}

void IRThl_fixed (IRT_top* irt_top, uint8_t image, uint8_t planes, char file_name[50], double D, bool input_hflip, bool input_vflip, int suffix) {

	uint8_t desc = image * planes;

	uint16_t Ho =  irt_top->irt_desc[desc].image_par[OIMAGE].H;
	uint16_t Wo =  irt_top->irt_desc[desc].image_par[OIMAGE].W;
	int16_t  Xoc = irt_top->irt_desc[desc].image_par[OIMAGE].Xc >> 1;
	int16_t  Yoc = irt_top->irt_desc[desc].image_par[OIMAGE].Yc >> 1;
	uint16_t Hi =  irt_top->irt_desc[desc].image_par[IIMAGE].H;
	uint16_t Wi =  irt_top->irt_desc[desc].image_par[IIMAGE].W;
	int16_t  Xic = irt_top->irt_desc[desc].image_par[IIMAGE].Xc >> 1;
	int16_t  Yic = irt_top->irt_desc[desc].image_par[IIMAGE].Yc >> 1;

	int warning_print[4] = { 0,0,0,0 };
	char out_file_name[50];

	double cosD = cos(D*M_PI/180), sinD = sin(D*M_PI/180);
	double out_val_fpt32, xi_fpt32, yi_fpt32, xf_fpt32, yf_fpt32;
	int xi0, yi0;

	int cosD_int = (int) floor(cosD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1);
	int sinD_int = (int) floor(sinD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1);

	double cosDr = (float) (floor(cosD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
	double sinDr = (float) (floor(sinD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

	int M11_int = (int)floor(irt_top->rot_pars[desc].M11d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);
	int M21_int = (int)floor(irt_top->rot_pars[desc].M21d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);
	int M12_int = (int)floor(irt_top->rot_pars[desc].M12d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);
	int M22_int = (int)floor(irt_top->rot_pars[desc].M22d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);

	double M11Dr = (float)(floor(irt_top->rot_pars[desc].M11d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
	double M21Dr = (float)(floor(irt_top->rot_pars[desc].M21d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
	double M12Dr = (float)(floor(irt_top->rot_pars[desc].M12d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
	double M22Dr = (float)(floor(irt_top->rot_pars[desc].M22d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

	int64_t xi_fixed, yi_fixed, xf_fixed, yf_fixed;
	int xi0_fixed, yi0_fixed;

	int xo, yo, XL, YT, XR, YB, out1;
	uint64_t out_val_fixed;
	uint8_t proc_size = fabs(D) < 59.0 ? 8 : 6;
	int16_t remain_pixels;

	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "\nRunning fixed point high level model on image %d %s %dx%d with %3.2f degree rotation with [h/v]flip [%d/%d]\n", image, file_name, Wi, Hi, D, input_hflip, input_vflip);

	FILE* hl_out_file;
	if (num_of_images) {
		(void)sprintf(out_file_name, "%s_%d_rot_fixed%s.txt", file_name, image, suffix == 0 ? "" : "_flip");
	} else {
		(void)sprintf(out_file_name, "%s_rot_fixed%s.txt", file_name, suffix == 0 ? "" : "_flip");
	}
	hl_out_file = fopen(out_file_name, "w");

	for (uint8_t plane = 0; plane < 3; plane++)
		for (uint16_t row = 0; row < Ho; row++)
			for (uint16_t col = 0; col < Wo; col += proc_size) {
				remain_pixels = Wo - col;
				if (remain_pixels > proc_size) remain_pixels = proc_size;
				XL = 1000; YT = 1000, YB=-1000, XR=-1000;
				for (int i=0; i<remain_pixels; i++) {
					xo =  col + i - Xoc;
					yo =  row - Yoc;

					int cosDi_int = (int) floor(i*cosD*pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1);
					int sinDi_int = (int) floor(i*sinD*pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1);
					cosDi_int = i * cosD_int;
					sinDi_int = i * sinD_int;
					double cosDi_fpt32 = floor(i*cosD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);
					double sinDi_fpt32 = floor(i*sinD * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)+0.5*1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);

					int M11i_int = (int)floor(i * irt_top->rot_pars[desc].M11d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);
					int M21i_int = (int)floor(i * irt_top->rot_pars[desc].M21d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1);
					M11i_int = i * M11_int;
					M21i_int = i * M21_int;
					double M11i_fpt32 = floor(i * irt_top->rot_pars[desc].M11d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);
					double M21i_fpt32 = floor(i * irt_top->rot_pars[desc].M21d * pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) + 0.5 * 1) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);

					if (irt_top->irt_desc[desc].irt_mode == e_irt_rotation) {
						xi_fpt32 = cosDr * xo + sinDr * yo + Xic;
						yi_fpt32 = -sinDr * xo + cosDr * yo + Yic;
						xi_fpt32 = cosDr * ((double)xo - i) + sinDr * yo + cosDi_fpt32 + Xic;
						yi_fpt32 = -sinDr * ((double)xo - i) + cosDr * yo - sinDi_fpt32 + Yic;
					} else {
						xi_fpt32 = M11Dr * ((double)xo - i) + M12Dr * yo + M11i_fpt32 + Xic;
						yi_fpt32 = M21Dr * ((double)xo - i) + M22Dr * yo + M21i_fpt32 + Yic;
					}

					xi0 = (int) floor(xi_fpt32);
					yi0 = (int) floor(yi_fpt32);
					xf_fpt32  = xi_fpt32 - (double) xi0;
					yf_fpt32  = yi_fpt32 - (double) yi0;

#if 0
					xi_fpt32_nearest_int = floor (xi_fpt32+0.5);
					yi_fpt32_nearest_int = floor (yi_fpt32+0.5);

					if (fabs(xi_fpt32 - xi_fpt32_nearest_int) < pow(2.0,-8))
						xi_fpt322 = xi_fpt32_nearest_int;
					else
						xi_fpt322 = xi_fpt32;

					if (fabs(yi_fpt32 - yi_fpt32_nearest_int) < pow(2.0,-4))
						yi_fpt322 = yi_fpt32_nearest_int;
					else
						yi_fpt322 = yi_fpt32;

					xi_fpt322 = xi_fpt32; yi_fpt322 = yi_fpt32;
					xi0 = floor(xi_fpt322);
					yi0 = floor(yi_fpt322);
#endif

					if (irt_top->irt_desc[desc].irt_mode == e_irt_rotation) {

						xi_fixed =  (int64_t)cosD_int * xo + (int64_t)sinD_int * yo + Xic * ((uint64_t) pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
						yi_fixed = -(int64_t)sinD_int * xo + (int64_t)cosD_int * yo + Yic * ((uint64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

						xi_fixed =  (int64_t)cosD_int * ((int64_t)xo - i) + (int64_t)sinD_int * yo + cosDi_int + Xic * ((uint64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
						yi_fixed = -(int64_t)sinD_int * ((int64_t)xo - i) + (int64_t)cosD_int * yo - sinDi_int + Yic * ((uint64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

					} else {

						xi_fixed = (int64_t)M11_int * ((int64_t)xo - i) + (int64_t)M12_int * yo + M11i_int + Xic * ((uint64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
						yi_fixed = (int64_t)M21_int * ((int64_t)xo - i) + (int64_t)M22_int * yo + M21i_int + Yic * ((uint64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

					}
					xi0_fixed = (int)(xi_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
					yi0_fixed = (int)(yi_fixed >> irt_top->rot_pars[desc].TOTAL_PREC);
					xf_fixed = (int)(xi_fixed - ((int64_t)xi0_fixed << irt_top->rot_pars[desc].TOTAL_PREC));
					yf_fixed = (int)(yi_fixed - ((int64_t)yi0_fixed << irt_top->rot_pars[desc].TOTAL_PREC));

//					xi_fixed_nearest_int = ((xi_fixed >> (IRT_TOTAL_PREC-1))+1)>>1;
//					yi_fixed_nearest_int = ((yi_fixed >> (IRT_TOTAL_PREC-1))+1)>>1;

					//if (xi0 != xi0_fixed) {
					if (abs(xi0 - xi0_fixed)>1) {
						IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Error: at pixel[%d,%d] xi0 != xi0_fixed (%d != %ld) at image %d\n", row, col, xi0, xi0_fixed, image);
						IRT_CLOSE_FAILED_TEST(0);
					}
					//if (yi0 != yi0_fixed) {
					if (abs(yi0 - yi0_fixed)>1) {
						IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Error: at pixel[%d,%d] yi0 != yi0_fixed (%d != %ld) at image %d\n", row, col + i, yi0, yi0_fixed, image);
						IRT_CLOSE_FAILED_TEST(0);
					}

					float thr = (float)(4 * pow(2.0, -irt_top->rot_pars[desc].TOTAL_PREC));
					if (fabs(xi_fpt32) >= 2048) thr *= 2;
					if (fabs(xi_fpt32) >= 1024) thr *= 2;
					if (fabs(xi_fpt32) >= 512) thr *= 2;
					if (fabs(xi_fpt32 - ((float)xi_fixed)/(float)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC))>thr) {
						IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Error: at pixel[%d,%d] xi_fpt32 != xi_fixed (%f != %f) at image %d\n", row, col + i, xi_fpt32, xi_fixed/pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC), image);
						IRT_CLOSE_FAILED_TEST(0);
					}
					thr = (float) (4*pow(2.0,-irt_top->rot_pars[desc].TOTAL_PREC));
					if (fabs(yi_fpt32) >= 2048) thr *= 2;
					if (fabs(yi_fpt32) >= 1024) thr *= 4;
					if (fabs(yi_fpt32) >= 512) thr *= 2;
					if (fabs(yi_fpt32 - ((float)yi_fixed)/(float)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC))>thr) {
						IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Error: at pixel[%d,%d] yi_fpt32 != yi_fixed (%f != %f) at image %d\n", row, col+i, yi_fpt32, yi_fixed/pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC), image);
						IRT_CLOSE_FAILED_TEST(0);
					}

					if (fabs(xf_fpt32 - ((float)xf_fixed)/(float)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC))>2*(float)pow(2.0,-irt_top->rot_pars[desc].TOTAL_PREC) && warning_print[0] == 0) {
						IRT_TRACE(IRT_TRACE_LEVEL_WARN, "Warning: at pixel[%4d,%4d] xf_fpt32 != xf_fixed (%f != %f) at image %d\n", row, col + i, xf_fpt32, xf_fixed/pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC), image);
						warning_print[0] = 1;
						//exit(0);
					}
					if (fabs(yf_fpt32 - ((float)yf_fixed)/(float)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC))>2*(float)pow(2.0,-irt_top->rot_pars[desc].TOTAL_PREC) && warning_print[1] == 0) {
					//if (yf_fpt32 != ((float)yf_fixed)/pow(2.0,IRT_TOTAL_PREC)) {
						IRT_TRACE(IRT_TRACE_LEVEL_WARN, "Warning: at pixel[%4d,%4d] yf_fpt32 != yf_fixed (%f != %f) at image %d\n", row, col + i, yf_fpt32, yf_fixed/pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC), image);
						warning_print[1] = 1;
						//exit(0);
					}

					if (input_vflip==0) {
						if (yi0_fixed < YT) YT = yi0_fixed;
						if (yi0_fixed + 1 > YB) YB = yi0_fixed + 1;
					} else {
						if (yi0_fixed-1 < YT) YT = yi0_fixed-1;
						if (yi0_fixed > YB) YB = yi0_fixed;
					}

					if (input_hflip == 0) {
						if (xi0_fixed < XL) XL = xi0_fixed;
						if (xi0_fixed + 1 > XR) XR = xi0_fixed + 1;
					} else {
						if (xi0_fixed - 1 < XL) XL = xi0_fixed - 1;
						if (xi0_fixed > XR) XR = xi0_fixed;
					}

					if (i==0) {
						fprintf (hl_out_file, "Plane %d, pixel[%3d,%3d], [yo,xo]=[%3d,%3d]\n", plane, row, col, yo, xo-i);
						fprintf (hl_out_file, "IRT_IIIRC: task=%d, Xo0=%d, Yo=%d, Xi0=%lld, Yi0=%lld, Xi7[%d]=%lld, Yi7=%lld\n",
							plane, xo, yo, xi_fixed, yi_fixed, remain_pixels - 1,
							(int64_t)cosD_int*((int64_t)xo+remain_pixels-1) + (int64_t)sinD_int*yo + Xic*(int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC),
							-(int64_t)sinD_int*((int64_t)xo+remain_pixels-1) + (int64_t)cosD_int*yo + Yic*(int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));
						//fprintf(hl_out_file, "[YT,XL,YB,XR]=[%3d,%3d,%3d,%3d]\n", YT, XL, YB, XR);
					}

					int i0, i1, j0, j1;
					if (input_vflip == 0) {
						i0 = yi0_fixed;
						i1 = yi0_fixed + 1;
					} else {
						i0 = Hi - 1 - yi0_fixed;
						i1 = Hi - 1 - yi0_fixed + 1;
					}
					if (input_hflip == 0) {
						j0 = xi0_fixed;
						j1 = xi0_fixed + 1;
					}
					else {
						j0 = Wi - 1 - xi0_fixed;
						j1 = Wi - 1 - xi0_fixed + 1;
					}

					uint16_t p0 = (i0 >= 0 && i0 < Hi && j0 >= 0 && j0 < Wi) ? input_image[plane][i0][j0] : 0;
					uint16_t p1 = (i0 >= 0 && i0 < Hi && j1 >= 0 && j1 < Wi) ? input_image[plane][i0][j1] : 0;
					uint16_t p2 = (i1 >= 0 && i1 < Hi && j0 >= 0 && j0 < Wi) ? input_image[plane][i1][j0] : 0;
					uint16_t p3 = (i1 >= 0 && i1 < Hi && j1 >= 0 && j1 < Wi) ? input_image[plane][i1][j1] : 0;
					fprintf(hl_out_file, "%d: [y00=%2d,x00=%2d], [%2x,%2x,%2x,%2x] [%4llx,%4llx] [%d,%d,%d,%d]\n", i, yi0_fixed, xi0_fixed,
						(uint32_t)((uint8_t)p0), (uint32_t)((uint8_t)p1), (uint32_t)((uint8_t)p2), (uint32_t)((uint8_t)p3),
						input_hflip == 0 ? xf_fixed : (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed,
						input_vflip == 0 ? yf_fixed : (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed,
						yi0_fixed < 0 || yi0_fixed >(Hi - 1) || xi0_fixed < 0 || xi0_fixed >(Wi - 1),
						yi0_fixed < 0 || yi0_fixed >(Hi - 1) || (xi0_fixed + 1) < 0 || (xi0_fixed + 1) > (Wi - 1),
						(yi0_fixed + 1) < 0 || (yi0_fixed + 1) > (Hi - 1) || xi0_fixed < 0 || xi0_fixed >(Wi - 1),
						(yi0_fixed + 1) < 0 || (yi0_fixed + 1) > (Hi - 1) || (xi0_fixed + 1) < 0 || (xi0_fixed + 1) > (Wi - 1));

#if 1
					if (i==remain_pixels-1)
						fprintf (hl_out_file, "[YT,XL,YB,XR]=[%3d,%3d,%3d,%3d]\n",YT, XL, YB, XR);
#endif

#if 0
					if (abs(xi_fixed - (xi_fixed_nearest_int<<IRT_TOTAL_PREC)) < (1<<(IRT_TOTAL_PREC-8))) {
						xi_fixed2 = (xi_fixed_nearest_int<<IRT_TOTAL_PREC);
						//printf("Error is small, set to nearest integer\n");
					}	else
						xi_fixed2 = xi_fixed;

					if (abs(yi_fixed - (yi_fixed_nearest_int<<IRT_TOTAL_PREC)) < (1<<(IRT_TOTAL_PREC-3)))
						yi_fixed = (yi_fixed_nearest_int<<IRT_TOTAL_PREC);

					//xi0_fixed = xi_fixed2 >> IRT_TOTAL_PREC;
					//yi0_fixed = yi_fixed >> IRT_TOTAL_PREC;
					//			xi_frac = xi_f_int
#endif
#if 0
					if (xi0!=xi0_fixed) {
					//				printf("IRT HL: row %d, col %d, xi_fpt32 %f, yi_fpt32 %f, xi0 %d, yi0 %d\n", row, col, xi_fpt32, yi_fpt32, xi0, yi0);
					//				printf("IRT HL: xi_fixed_nearest_int %d, err %d, thr %d, xi_fixed %d, xi_fixed2 %d\n", xi_fixed_nearest_int, abs(xi_fixed - (xi_fixed_nearest_int<<IRT_TOTAL_PREC)),
					//					(1<<(IRT_TOTAL_PREC-8)), xi_fixed, xi_fixed2);
					//				printf("IRT HL: xi fixed point calculation error xi0 = %d, xi0_fixed = %d\n", xi0, xi0_fixed);
					//				printf(" xi_fpt32 = %f, xi_fixed = %f, diff %f, error 2^%d\n", xi_fpt32, ((float)xi_fixed)/pow(2.0,IRT_TOTAL_PREC),xi_fpt32-((float)xi_fixed)/pow(2.0,IRT_TOTAL_PREC),
					//					(int)(log(fabs(xi_fpt32-((float)xi_fixed)/pow(2.0,IRT_TOTAL_PREC)))/log(2.0)));
					//				exit(0);
					}
					if (yi!=(int)yi_int) {
					//				printf("IRT HL: row %d, col %d, xi_f %f, yi_f %f, xi_f_int %x, yi_f_int %x\n", row, col, xi_f, yi_f, xi_f_int, yi_f_int);
					//				printf("IRT HL: yi fixed point calculation error yi float = %f, yi fixed = %d\n", yi_f, yi_int);
					//				printf(" yi_f = %f, yi_f_int = %f\n", yi_f, yi_f_int/pow(2.0,IRT_TOTAL_PREC));
					//				exit(0);
					}
#endif

					//force weight to 1 and 0 for nearest neigbour interpolation
					if (irt_top->irt_desc[desc].int_mode) {
						if (xf_fpt32 > 0.5)
							xf_fpt32 = 1;
						else if (xf_fpt32 < 0.5)
							xf_fpt32 = 0;

						if (yf_fpt32 > 0.5)
							yf_fpt32 = 1;
						else if (yf_fpt32 < 0.5)
							yf_fpt32 = 0;

						if (xf_fixed > (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC-1))
							xf_fixed = (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);
						else if (xf_fixed < (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC-1))
							xf_fixed = 0;

						if (yf_fixed > (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC-1))
							yf_fixed = (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC);
						else if (yf_fixed < (int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC-1))
							yf_fixed = 0;
					}

//					if (xi0>=0 && xi0+1<Wi && yi0>=0 && yi0+1<Hi) {
					if ((xi0_fixed >= 0 && xi0_fixed + 1 < Wi && yi0_fixed >= 0 && yi0_fixed + 1 < Hi) && (input_hflip == 0) && (input_vflip == 0) ||
						(xi0_fixed >= 1 && xi0_fixed	 < Wi && yi0_fixed >= 0 && yi0_fixed + 1 < Hi) && (input_hflip == 1) && (input_vflip == 0) ||
						(xi0_fixed >= 0 && xi0_fixed + 1 < Wi && yi0_fixed >= 1 && yi0_fixed	 < Hi) && (input_hflip == 0) && (input_vflip == 1) ||
						(xi0_fixed >= 1 && xi0_fixed	 < Wi && yi0_fixed >= 1 && yi0_fixed	 < Hi) && (input_hflip == 1) && (input_vflip == 1)) {

						//cout << height << " " << width << " " << plane << " " << (int) (input_image[plane][yi_int][xi_int]) << endl;
						if (input_hflip==0 && input_vflip == 0) {
							out_val_fpt32 =	(uint32_t)((uint8_t)input_image[plane][yi0  ][xi0  ])*(1-xf_fpt32)*(1-yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][yi0  ][xi0+1])*(xf_fpt32)*(1-yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][yi0+1][xi0  ])*(1-xf_fpt32)*(yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][yi0+1][xi0+1])*(xf_fpt32)*(yf_fpt32);

							out_val_fpt32 =	(uint32_t)((uint8_t)input_image[plane][yi0_fixed  ][xi0_fixed  ])*(pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*(pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed) +
											(uint32_t)((uint8_t)input_image[plane][yi0_fixed  ][xi0_fixed+1])*xf_fixed*(pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed) +
											(uint32_t)((uint8_t)input_image[plane][yi0_fixed+1][xi0_fixed  ])*(pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*yf_fixed +
											(uint32_t)((uint8_t)input_image[plane][yi0_fixed+1][xi0_fixed+1])*((double)xf_fixed)*((double)yf_fixed);

							out_val_fixed =	(uint64_t)((uint8_t)input_image[plane][yi0_fixed  ][xi0_fixed  ])*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][yi0_fixed  ][xi0_fixed+1])*(xf_fixed)*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][yi0_fixed+1][xi0_fixed  ])*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*(yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][yi0_fixed+1][xi0_fixed+1])*(xf_fixed)*(yf_fixed);

						} else {
							out_val_fpt32 =	(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0  ][Wi-1-xi0  ])*(xf_fpt32)*(yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0  ][Wi-1-xi0+1])*(1-xf_fpt32)*(yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0+1][Wi-1-xi0  ])*(xf_fpt32)*(1-yf_fpt32) +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0+1][Wi-1-xi0+1])*(1-xf_fpt32)*(1-yf_fpt32);

							out_val_fpt32 =	(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed  ][Wi-1-xi0_fixed  ])*((double)xf_fixed)*((double)yf_fixed) +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed  ][Wi-1-xi0_fixed+1])*((double)(pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed))*yf_fixed +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed+1][Wi-1-xi0_fixed  ])*xf_fixed*((pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed)) +
											(uint32_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed+1][Wi-1-xi0_fixed+1])*((pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed))*((pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed));

							out_val_fixed =	(uint64_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed  ][Wi-1-xi0_fixed  ])*(xf_fixed)*(yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed  ][Wi-1-xi0_fixed+1])*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*(yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed+1][Wi-1-xi0_fixed  ])*(xf_fixed)*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed) +
											(uint64_t)((uint8_t)input_image[plane][Hi-1-yi0_fixed+1][Wi-1-xi0_fixed+1])*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - xf_fixed)*((int64_t)pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC) - yf_fixed);
						}

						out_val_fpt32 = (float) (out_val_fpt32 / pow(4.0, irt_top->rot_pars[desc].TOTAL_PREC));

						out_val_fixed = (int)floor((floor((double)out_val_fixed / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC)) + pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC - 1)) / pow(2.0, irt_top->rot_pars[desc].TOTAL_PREC));

						out1 = (int) floor (out_val_fpt32+0.5);//(int) out_val;
						out1 = (int) out_val_fixed;
//						fprintf (hl_out_file, "BLI: res %d\n", out1);
						if (out1 < 0) out1 = 0;
						if (out1 > 255) out1 = 255;
						output_image[image][plane][row][col+i] = (uint8_t) out1;
					} else {
						output_image[image][plane][row][col+i] = (uint8_t) irt_top->irt_desc[desc].bg;
					}
				}
			}

	fclose (hl_out_file);
	if (num_of_images > 1) {
		(void)sprintf(out_file_name, "%s_%d_rot_fixed%s.bmp", file_name, image, suffix == 0 ? "" : "_flip");
	} else {
		(void)sprintf(out_file_name, "%s_rot_fixed%s.bmp", file_name, suffix == 0 ? "" : "_flip");
	}
	WriteBMP(irt_top, image, 3, out_file_name, Wo, Ho, output_image[image]);
}

void IRThl(IRT_top* irt_top, uint8_t image, uint8_t planes, uint16_t i_width, uint16_t i_height, char* input_file, char* outfilename, char* outfiledir) {

	uint8_t desc = image * planes;
	char out_file_name[50];
	uint16_t o_height = i_width;
	uint16_t o_width = i_height;
	o_height = irt_top->irt_desc[desc].image_par[OIMAGE].H;
	o_width = irt_top->irt_desc[desc].image_par[OIMAGE].W;

	//printf("\nRunning %s high level model on image %d %s %dx%d\n", irt_irt_mode_s[irt_top->irt_desc[desc].irt_mode], image, file_name, i_width, i_height);
	IRThl_advanced_transform(irt_top, image, planes, input_file, outfilename, outfiledir, irt_top->irt_desc[desc].irt_mode);

	if (print_out_files) {
		for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fix16; format_type++) {
			if (num_of_images > 1) {
				(void)sprintf(out_file_name, "%s%s_hlsim_out_image_%d_%s.bmp", outfiledir, outfilename, image, irt_hl_format_s[format_type]);
			} else {
				(void)sprintf(out_file_name, "%s%s_hlsim_out_image_%s.bmp", outfiledir, outfilename, irt_hl_format_s[format_type]);
			}
			WriteBMP(irt_top, image, PLANES, out_file_name, irt_top->irt_desc[desc].image_par[OIMAGE].S, irt_top->irt_desc[desc].image_par[OIMAGE].H, output_image[image * CALC_FORMATS_ROT + format_type]);
		}
		for (uint8_t format_type = e_irt_crdf_arch; format_type <= e_irt_crdf_fix16; format_type++) {
			if (num_of_images > 1) {
				(void)sprintf(out_file_name, "%s%s_hlsim_out_image_%d_%s.bmp", outfiledir, "IRT", image, irt_hl_format_s[format_type]);
			} else {
				(void)sprintf(out_file_name, "%s%s_hlsim_out_image_%s.bmp", outfiledir, "IRT", irt_hl_format_s[format_type]);
			}
			WriteBMP(irt_top, image, PLANES, out_file_name, irt_top->irt_desc[desc].image_par[OIMAGE].S, irt_top->irt_desc[desc].image_par[OIMAGE].H, output_image[image * CALC_FORMATS_ROT + format_type]);
		}
	}

}

