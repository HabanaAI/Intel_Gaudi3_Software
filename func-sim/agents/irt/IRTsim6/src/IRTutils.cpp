#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <stdio.h>

#include "stdlib.h"
#include "string.h"
#include "math.h"
#include <algorithm>
#include "IRTutils.h"
//#include "IRTsim.h"

//#if defined(STANDALONE_ROTATOR) || defined (HABANA_SIMULATION)

extern bool irt_h5_mode;
extern bool read_iimage_BufW;
extern bool read_mimage_BufW;
extern uint8_t hl_only;
extern Eirt_bmp_order irt_bmp_rd_order, irt_bmp_wr_order;
extern FILE* test_res;

extern uint8_t* ext_mem;
extern uint16_t*** input_image;
extern uint16_t**** output_image;

int coord_err[CALC_FORMATS];
extern bool print_log_file;
extern bool print_out_files;
extern bool rand_input_image;
extern bool irt_multi_image;
extern uint8_t num_of_images;
extern bool test_file_flag;

#if defined(STANDALONE_ROTATOR)
const uint8_t irt_colors[NUM_OF_COLORS][3] = {
	{0, 0, 0		},
	{255, 255, 255	},
	{255, 0, 0		},
	{0, 255, 0		},
	{0, 0, 255		},
	{255, 255, 0	},
	{0, 255, 255	},
	{255, 0, 255	},
	{192, 192, 192	},
	{128, 128, 128	},
	{128, 0, 0		},
	{128, 128, 0	},
	{0, 128, 0		},
	{128, 0, 128	},
	{0, 128, 128	},
	{0, 0, 128		},
};
#endif

void create_mems(uint8_t** ext_mem_l, uint16_t**** input_image_l, uint16_t***** output_image_l) {

	uint32_t emem_size, omem_num_images;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
	emem_size = irt_multi_image ? EMEM_SIZE_MT : EMEM_SIZE;
	omem_num_images = irt_multi_image ? OMEM_NUM_IMAGES_MT : OMEM_NUM_IMAGES;
#else
	emem_size = EMEM_SIZE;
	omem_num_images = OMEM_NUM_IMAGES;
#endif

	//memories definition
	*ext_mem_l = new uint8_t[emem_size]; //input, output, mesh

	*input_image_l = new uint16_t** [PLANES];
	for (uint8_t plain = 0; plain < PLANES; plain++) {
		(*input_image_l)[plain] = new uint16_t* [IIMAGE_H];
		for (uint16_t row = 0; row < IIMAGE_H; row++) {
			(*input_image_l)[plain][row] = new uint16_t[IIMAGE_W];
		}
	}
	for (uint8_t plain = 0; plain < PLANES; plain++) {
		for (uint16_t row = 0; row < IIMAGE_H; row++) {
			for (uint16_t col = 0; col < IIMAGE_W; col++) {
				(*input_image_l)[plain][row][col] = 0;
			}
		}
	}

	*output_image_l = new uint16_t*** [omem_num_images];
	for (uint8_t image = 0; image < omem_num_images; image++) {
		(*output_image_l)[image] = new uint16_t** [PLANES];
		for (uint8_t plain = 0; plain < PLANES; plain++) {
			(*output_image_l)[image][plain] = new uint16_t* [OIMAGE_H];
			for (uint16_t row = 0; row < OIMAGE_H; row++) {
				(*output_image_l)[image][plain][row] = new uint16_t[OIMAGE_W];
			}
		}
	}

}

void delete_mems(uint8_t* ext_mem_l, uint16_t*** input_image_l, uint16_t**** output_image_l) {

	uint32_t omem_num_images;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
	omem_num_images = irt_multi_image ? OMEM_NUM_IMAGES_MT : OMEM_NUM_IMAGES;
#else
	omem_num_images = OMEM_NUM_IMAGES;
#endif

	delete[] ext_mem_l;
	ext_mem_l = nullptr;

	for (uint8_t plain = 0; plain < PLANES; plain++) {
		for (uint16_t row = 0; row < IIMAGE_H; row++) {
			delete[] input_image_l[plain][row];
			input_image_l[plain][row] = nullptr;
		}
		delete[] input_image_l[plain];
		input_image_l[plain] = nullptr;
	}
	delete[] input_image_l;
	input_image_l = nullptr;

	for (uint8_t image = 0; image < omem_num_images; image++) {
		for (uint8_t plain = 0; plain < PLANES; plain++) {
			for (uint16_t row = 0; row < OIMAGE_H; row++) {
				delete[] output_image_l[image][plain][row];
				output_image_l[image][plain][row] = nullptr;
			}
			delete[] output_image_l[image][plain];
			output_image_l[image][plain] = nullptr;
		}
		delete[] output_image_l[image];
	}
	delete[] output_image_l;
	output_image_l = nullptr;

}

void irt_cfg_init(irt_cfg_pars& irt_cfg) {

	//init buffer modes parameters
	//for rotation memory
	irt_cfg.buf_format[e_irt_block_rot] = e_irt_buf_format_static;
	irt_cfg.buf_1b_mode[e_irt_block_rot] = 0;
	irt_cfg.buf_select[e_irt_block_rot] = e_irt_buf_select_auto; //auto
	irt_cfg.buf_mode[e_irt_block_rot] = 3;
	for (uint8_t mode = 0; mode < IRT_RM_MODES; mode++) {
		irt_cfg.rm_cfg[e_irt_block_rot][mode].BufW = (2 * IRT_ROT_MEM_BANK_WIDTH) << mode;
		irt_cfg.rm_cfg[e_irt_block_rot][mode].Buf_EpL = (1 << mode);
		irt_cfg.rm_cfg[e_irt_block_rot][mode].BufH = (8 * IRT_ROT_MEM_BANK_HEIGHT) >> mode;
		irt_cfg.rm_cfg[e_irt_block_rot][mode].BufH_mod = irt_cfg.rm_cfg[0][mode].BufH - 1;
		/*
			Buf mode 0:   32x1024
			Buf mode 1:   64x 512
			Buf mode 2:  128x 256
			Buf mode 3:  256x 128
			Buf mode 4:  512x  64
			Buf mode 5: 1024x  32
			Buf mode 6: 2048x  16
			Buf mode 7: 4096x   8
		*/
	}
	irt_cfg.Hb[e_irt_block_rot] = IRT_ROT_MEM_BANK_HEIGHT;
	irt_cfg.Hb_mod[e_irt_block_rot] = irt_cfg.Hb[e_irt_block_rot] - 1;

	//for mesh memory
	irt_cfg.buf_format[e_irt_block_mesh] = e_irt_buf_format_static;
	irt_cfg.buf_1b_mode[e_irt_block_mesh] = 0;
	irt_cfg.buf_select[e_irt_block_mesh] = e_irt_buf_select_auto; //auto
	irt_cfg.buf_mode[e_irt_block_mesh] = 3;
	for (uint8_t mode = 0; mode < IRT_RM_MODES; mode++) {
		irt_cfg.rm_cfg[e_irt_block_mesh][mode].BufW = (2 * IRT_MESH_MEM_BANK_WIDTH) << mode;
		irt_cfg.rm_cfg[e_irt_block_mesh][mode].Buf_EpL = (1 << mode);
		irt_cfg.rm_cfg[e_irt_block_mesh][mode].BufH = (2 * IRT_MESH_MEM_BANK_HEIGHT) >> mode;
		irt_cfg.rm_cfg[e_irt_block_mesh][mode].BufH_mod = irt_cfg.rm_cfg[1][mode].BufH - 1;
		/*
			Buf mode 0:   128x128
			Buf mode 1:   256x 64
			Buf mode 2:   512x 32
			Buf mode 3:  1024x 16
			Buf mode 4:  2048x  8
			Buf mode 5:  4096x  4
			Buf mode 6:  8192x  2
			Buf mode 7: 16384x  1
		*/
	}
	irt_cfg.Hb[e_irt_block_mesh] = IRT_MESH_MEM_BANK_HEIGHT;
	irt_cfg.Hb_mod[e_irt_block_mesh] = irt_cfg.Hb[e_irt_block_mesh] - 1;

}

void irt_par_init(rotation_par& rot_pars) {

	rot_pars.MAX_PIXEL_WIDTH = IRT_PIXEL_WIDTH;
	rot_pars.MAX_COORD_WIDTH = IRT_COORD_WIDTH;
	sprintf(rot_pars.proj_order, "YRP");
	rot_pars.min_proc_rate = IRT_ROT_MAX_PROC_SIZE;
	rot_pars.WEIGHT_PREC = rot_pars.MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;
	rot_pars.WEIGHT_SHIFT = 0;//rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
	rot_pars.COORD_PREC = rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
	rot_pars.TOTAL_PREC = rot_pars.WEIGHT_PREC + rot_pars.COORD_PREC;
	rot_pars.TOTAL_ROUND = (1 << (rot_pars.TOTAL_PREC - 1));
	rot_pars.PROJ_NOM_PREC = rot_pars.TOTAL_PREC;
	rot_pars.PROJ_DEN_PREC = rot_pars.TOTAL_PREC + 10;
	rot_pars.PROJ_NOM_ROUND = (1 << (rot_pars.PROJ_NOM_PREC - 1));
	rot_pars.PROJ_DEN_ROUND = (1 << (rot_pars.PROJ_DEN_PREC - 1));

}

void irt_mesh_mems_create(irt_mesh_images& mesh_images) {

	//init mesh arrays pointer
	mesh_images.mesh_image_full = new mesh_xy_fp32_meta * [OIMAGE_H];
	mesh_images.mesh_image_rel  = new mesh_xy_fp32 * [OIMAGE_H];
	mesh_images.mesh_image_fp32 = new mesh_xy_fp32 * [OIMAGE_H];
	mesh_images.mesh_image_fi16 = new mesh_xy_fi16 * [OIMAGE_H];
	mesh_images.mesh_image_intr = new mesh_xy_fp64_meta * [OIMAGE_H];
	mesh_images.proj_image_full = new mesh_xy_fp64_meta * [OIMAGE_H];
	for (int row = 0; row < OIMAGE_H; row++) {
		(mesh_images.mesh_image_full)[row] = new mesh_xy_fp32_meta[OIMAGE_W];
		(mesh_images.mesh_image_rel)[row]  = new mesh_xy_fp32[OIMAGE_W];
		(mesh_images.mesh_image_fp32)[row] = new mesh_xy_fp32[OIMAGE_W];
		(mesh_images.mesh_image_fi16)[row] = new mesh_xy_fi16[OIMAGE_W];
		(mesh_images.mesh_image_intr)[row] = new mesh_xy_fp64_meta[OIMAGE_W];
		(mesh_images.proj_image_full)[row] = new mesh_xy_fp64_meta[OIMAGE_W];
	}
}

void irt_mesh_mems_delete(irt_mesh_images& mesh_images) {

	for (int row = OIMAGE_H - 1; row >= 0; row--) {
		delete[] mesh_images.mesh_image_full[row];
		delete[] mesh_images.mesh_image_rel[row];
		delete[] mesh_images.mesh_image_fp32[row];
		delete[] mesh_images.mesh_image_fi16[row];
		delete[] mesh_images.mesh_image_intr[row];
		delete[] mesh_images.proj_image_full[row];

		mesh_images.mesh_image_full[row] = nullptr;
		mesh_images.mesh_image_rel[row] = nullptr;
		mesh_images.mesh_image_fp32[row] = nullptr;
		mesh_images.mesh_image_fi16[row] = nullptr;
		mesh_images.mesh_image_intr[row] = nullptr;
		mesh_images.proj_image_full[row] = nullptr;
	}

	delete[] mesh_images.mesh_image_full;
	delete[] mesh_images.mesh_image_rel;
	delete[] mesh_images.mesh_image_fp32;
	delete[] mesh_images.mesh_image_fi16;
	delete[] mesh_images.mesh_image_intr;
	delete[] mesh_images.proj_image_full;

	mesh_images.mesh_image_full = nullptr;
	mesh_images.mesh_image_rel  = nullptr;
	mesh_images.mesh_image_fp32 = nullptr;
	mesh_images.mesh_image_fi16 = nullptr;
	mesh_images.mesh_image_intr = nullptr;
	mesh_images.proj_image_full = nullptr;
}

//#endif

#ifdef HABANA_SIMULATION

#include "fs_log.h"

FILE* test_res;
#define MAX_LOG_LINE 300

void IRT_TRACE_HABANA_SIM(bool useTestLogger, const std::string& name, const char* msg, ...)
{
	char buff[MAX_LOG_LINE + 1];

	va_list args;
	va_start(args, msg);

	vsnprintf(buff, MAX_LOG_LINE, msg, args);

	FS_LOG_TRACE((useTestLogger) ? TEST_ROT_LOG_NAME : ROT_LOG_NAME, "{}: {}", name.c_str(), buff);

    va_end(args);
}
#endif
#ifdef STANDALONE_ROTATOR

extern FILE* log_file;

// addr - address to the image
// read - do read
// write - do write
// wr_data - data to write to the memory
// rd_data - data read from the memory. Use 0 for padding.
// meta_in - should be equal to meta_out
void ext_memory_wrapper (uint64_t addr, bool read, bool write, bus128B_struct wr_data, bus128B_struct &rd_data, meta_data_struct meta_in, meta_data_struct &meta_out, uint16_t lsb_pad, uint16_t msb_pad) {

	int byte;

	uint32_t emem_size;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
	emem_size = irt_multi_image ? EMEM_SIZE_MT : EMEM_SIZE;
#else
	emem_size = EMEM_SIZE;
#endif

	if (read) { //read
		//IRT_TRACE("Reading address %x\n", addr);
		for (byte = 0; byte < 128; byte++) {
			if ((addr + byte) >= 0 && (addr + byte) <= emem_size - 1) {//(int) sizeof(ext_mem)) {
				rd_data.pix[byte] = ext_mem[addr + byte];
#if 1
				if (byte < lsb_pad) {
					rd_data.pix[byte] = 0xed;
				}
				if (byte > (127 - msb_pad)) { // 128-byte <  msb_pad    byte >  128 - msb_pad
					rd_data.pix[byte] = 0xef;
				}
#endif
			} else {
				rd_data.pix[byte] = 0xee;//(char)0xaa;
				//IRT_TRACE ("External memory read address error: %d\n", addr+byte);
			}
			//IRT_TRACE("Read ext mem byte %d addr %lld, data %d pads %d %d\n", byte, addr + byte, rd_data.pix[byte], lsb_pad, msb_pad);
		}
		//IRT_TRACE("\n");
		meta_out.line = meta_in.line;
		meta_out.Xi_start = meta_in.Xi_start;
		meta_out.task = meta_in.task;
		//IRT_TRACE("%d %d %d\n", meta_out.line, meta_out.Xi_start, meta_out.task);
	}

	if (write) { //write
		//IRT_TRACE ("Ext mem: writting to addr %x\n", addr);
		for (byte = 0; byte < 128; byte++) {
			if ((addr + byte) >= 0 && (addr + byte) <= emem_size - 1) {
				if (wr_data.en[byte] == 1) {
					ext_mem[addr + byte] = wr_data.pix[byte];
				}
			} else {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "External memory of %d size write address error: %llx\n", emem_size, addr + byte);
			}
		}
#if 0
		IRT_TRACE_TO_FILE (log_file, "Addr %x\n",addr);
		for (byte = 0; byte < 128; byte++)
			IRT_TRACE_TO_FILE (log_file, "%02x", (unsigned int) ((unsigned char) wr_data.pix[byte]));
		IRT_TRACE_TO_FILE (log_file, "\n");
		for (byte = 0; byte < 128; byte++)
			IRT_TRACE_TO_FILE (log_file, "%x", wr_data.en[byte]);
		IRT_TRACE_TO_FILE (log_file, "\n");
#endif
	}
}

void irt_params_parsing(IRT_top *irt_top, uint8_t image, int argc, char* argv[], uint16_t i_width, uint16_t i_height) {

	uint8_t task;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Parsing %d input parameters\n", argc);
	//num_of_images = 1;
	//super_irt = 0;

	for (int argcnt = 1; argcnt < argc; argcnt++) {
		//if (!strcmp(argv[argcnt], "-num"))			num_of_images = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-irt_h5_mode"))	irt_h5_mode = 1;
		if (!strcmp(argv[argcnt], "-read_iimage_BufW"))	read_iimage_BufW = 1;
		if (!strcmp(argv[argcnt], "-read_mimage_BufW"))	read_mimage_BufW = 1;
		if (!strcmp(argv[argcnt], "-hl_only"))		hl_only = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-bmp_wr_order"))	irt_bmp_wr_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-bmp_rd_order"))	irt_bmp_rd_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-flow_mode"))	irt_top->irt_cfg.flow_mode = (Eirt_flow_type)atoi(argv[argcnt + 1]);

		if (!strcmp(argv[argcnt], "-rot_buf_format"))  irt_top->irt_cfg.buf_format[e_irt_block_rot]  = (Eirt_buf_format)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-rot_buf_select"))  irt_top->irt_cfg.buf_select[e_irt_block_rot]  = (Eirt_buf_select)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-rot_buf_mode"))    irt_top->irt_cfg.buf_mode[e_irt_block_rot]    = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-rot_buf_1b_mode")) irt_top->irt_cfg.buf_1b_mode[e_irt_block_rot] = (bool)atoi(argv[argcnt + 1]);


		if (!strcmp(argv[argcnt], "-mesh_buf_format"))  irt_top->irt_cfg.buf_format[e_irt_block_mesh]  = (Eirt_buf_format)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-mesh_buf_select"))  irt_top->irt_cfg.buf_select[e_irt_block_mesh]  = (Eirt_buf_select)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-mesh_buf_mode"))    irt_top->irt_cfg.buf_mode[e_irt_block_mesh]    = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-mesh_buf_1b_mode")) irt_top->irt_cfg.buf_1b_mode[e_irt_block_mesh] = (bool)atoi(argv[argcnt + 1]);

		if (!strcmp(argv[argcnt], "-dbg_mode"))			irt_top->irt_cfg.debug_mode = (Eirt_debug_type)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-print_log"))		print_log_file = 1;
		if (!strcmp(argv[argcnt], "-print_images"))		print_out_files = 1;

	}

	//for (uint8_t image = 0; image < num_of_images; image++) {
		for (uint8_t plane = 0; plane < PLANES; plane++) {
			task = image * PLANES + plane;

			//setting defaults
			irt_top->rot_pars[task].MAX_PIXEL_WIDTH = IRT_PIXEL_WIDTH;
			irt_top->rot_pars[task].MAX_COORD_WIDTH = IRT_COORD_WIDTH;
			irt_top->rot_pars[task].COORD_PREC		= irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
			irt_top->rot_pars[task].COORD_ROUND		= (1 << (irt_top->rot_pars[task].COORD_PREC - 1));
			irt_top->rot_pars[task].WEIGHT_PREC		= irt_top->rot_pars[task].MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;
			irt_top->rot_pars[task].WEIGHT_SHIFT    = 0;//irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
			irt_top->rot_pars[task].TOTAL_PREC		= irt_top->rot_pars[task].WEIGHT_PREC + irt_top->rot_pars[task].COORD_PREC;
			irt_top->rot_pars[task].TOTAL_ROUND		= (1 << (irt_top->rot_pars[task].TOTAL_PREC - 1));
			irt_top->rot_pars[task].rot_prec_auto_adj   = 0;
			irt_top->rot_pars[task].mesh_prec_auto_adj = 0;
			irt_top->rot_pars[task].oimg_auto_adj = 0;
			irt_top->rot_pars[task].oimg_auto_adj_rate = 0;
			irt_top->rot_pars[task].use_Si_delta_margin = 0;

			irt_top->rot_pars[task].PROJ_NOM_PREC = irt_top->rot_pars[task].TOTAL_PREC;
			irt_top->rot_pars[task].PROJ_DEN_PREC = irt_top->rot_pars[task].TOTAL_PREC + 10;
			irt_top->rot_pars[task].PROJ_NOM_ROUND = (1 << (irt_top->rot_pars[task].PROJ_NOM_PREC - 1));
			irt_top->rot_pars[task].PROJ_DEN_ROUND = (1 << (irt_top->rot_pars[task].PROJ_DEN_PREC - 1));

			irt_top->irt_desc[task].image_par[IIMAGE].addr_start = (uint64_t)IIMAGE_REGION_START + (uint64_t)plane * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL;
			irt_top->irt_desc[task].image_par[OIMAGE].addr_start = (uint64_t)OIMAGE_REGION_START + (uint64_t)task * OIMAGE_W * OIMAGE_H * BYTEs4PIXEL;
			irt_top->irt_desc[task].image_par[MIMAGE].addr_start = (irt_multi_image ? (uint64_t)MIMAGE_REGION_START_MT : (uint64_t)MIMAGE_REGION_START) + (uint64_t)image * OIMAGE_W * OIMAGE_H * BYTEs4MESH;

			irt_top->irt_desc[task].image_par[IIMAGE].H = i_height;
			irt_top->irt_desc[task].image_par[IIMAGE].W = i_width;
			irt_top->irt_desc[task].image_par[OIMAGE].H = 64;
			irt_top->irt_desc[task].image_par[OIMAGE].W = 128;
			irt_top->irt_desc[task].image_par[MIMAGE].H = 64;
			irt_top->irt_desc[task].image_par[MIMAGE].W = 128;
			irt_top->rot_pars[task].Pwi = 8;
			irt_top->rot_pars[task].Pwo = 8;
			irt_top->rot_pars[task].Ppi = 0;
			irt_top->irt_desc[task].Ppo = 0;

			irt_top->irt_desc[task].int_mode = e_irt_int_bilinear; //bilinear
			irt_top->irt_desc[task].irt_mode = e_irt_rotation; //rotation
			irt_top->irt_desc[task].crd_mode = e_irt_crd_mode_fixed; //fixed point

			irt_top->rot_pars[task].irt_angles[e_irt_angle_rot] = 30;

			sprintf(irt_top->rot_pars[task].affine_mode, "R");
			irt_top->rot_pars[task].reflection_mode = 0;
			irt_top->rot_pars[task].shear_mode = 0;
			irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_x] = 90;
			irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_y] = 90;
			irt_top->rot_pars[task].Sx = 1;
			irt_top->rot_pars[task].Sy = 1;

			irt_top->rot_pars[task].proj_mode = 0;
			sprintf(irt_top->rot_pars[task].proj_order, "YRP");
			irt_top->rot_pars[task].irt_angles[e_irt_angle_roll] = 50.0;
			irt_top->rot_pars[task].irt_angles[e_irt_angle_pitch] = -60.0;
			irt_top->rot_pars[task].irt_angles[e_irt_angle_yaw] = 60.0;
			irt_top->rot_pars[task].proj_Zd = 400.0;
			irt_top->rot_pars[task].proj_Wd = 1000.0;

			irt_top->irt_desc[task].oimage_line_wr_format = 0;
			irt_top->irt_desc[task].bg = 0x0;
			irt_top->irt_desc[task].bg_mode = e_irt_bg_prog_value;

			irt_top->rot_pars[task].dist_x = 1;
			irt_top->rot_pars[task].dist_y = 1;
			irt_top->rot_pars[task].dist_r = 0;

			irt_top->rot_pars[task].mesh_mode = 0;
			irt_top->rot_pars[task].mesh_order = 0;
			irt_top->rot_pars[task].mesh_Sh = 1.0;
			irt_top->rot_pars[task].mesh_Sv = 1.0;
			irt_top->irt_desc[task].mesh_format = e_irt_mesh_flex;
			irt_top->irt_desc[task].mesh_point_location = 0;
			irt_top->irt_desc[task].mesh_rel_mode = e_irt_mesh_absolute;

			irt_top->irt_desc[task].rate_mode = e_irt_rate_fixed;
			irt_top->irt_desc[task].proc_size = IRT_ROT_MAX_PROC_SIZE;
			irt_top->rot_pars[task].proc_auto = 0;

			//parsing main parameters
			for (int argcnt = 1; argcnt < argc; argcnt++) {

				if (!strcmp(argv[argcnt], "-pw"))					irt_top->rot_pars[task].MAX_PIXEL_WIDTH = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-cw"))					irt_top->rot_pars[task].MAX_COORD_WIDTH = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-ws"))					irt_top->rot_pars[task].WEIGHT_SHIFT = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-rot_prec_auto_adj"))	irt_top->rot_pars[task].rot_prec_auto_adj = 1;
				if (!strcmp(argv[argcnt], "-mesh_prec_auto_adj"))	irt_top->rot_pars[task].mesh_prec_auto_adj = 1;
				if (!strcmp(argv[argcnt], "-oimg_auto_adj"))		irt_top->rot_pars[task].oimg_auto_adj = 1;
				if (!strcmp(argv[argcnt], "-oimg_auto_adj_rate"))	irt_top->rot_pars[task].oimg_auto_adj_rate = (float)atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-use_Si_delta_margin"))		irt_top->rot_pars[task].use_Si_delta_margin = 1;
				if (!strcmp(argv[argcnt], "-Ho"))					irt_top->irt_desc[task].image_par[OIMAGE].H = (uint16_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Wo"))					irt_top->irt_desc[task].image_par[OIMAGE].W = (uint16_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Pwo"))					irt_top->rot_pars[task].Pwo = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Ppo"))					irt_top->irt_desc[task].Ppo = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Hi"))					irt_top->irt_desc[task].image_par[IIMAGE].H = (uint16_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Wi"))					irt_top->irt_desc[task].image_par[IIMAGE].W = (uint16_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Pwi"))					irt_top->rot_pars[task].Pwi = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Ppi"))					irt_top->rot_pars[task].Ppi = (uint8_t)atoi(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-int_mode"))				irt_top->irt_desc[task].int_mode = (Eirt_int_mode_type)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-irt_mode"))				irt_top->irt_desc[task].irt_mode = (Eirt_tranform_type)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-crd_mode"))				irt_top->irt_desc[task].crd_mode = (Eirt_coord_mode_type)atoi(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-rot_angle"))			irt_top->rot_pars[task].irt_angles[e_irt_angle_rot] = (float)atof(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-aff_mode"))				sprintf(irt_top->rot_pars[task].affine_mode, "%s", argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-rfl_mode"))				irt_top->rot_pars[task].reflection_mode = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-shr_mode"))				irt_top->rot_pars[task].shear_mode = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-shr_xang"))				irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_x] = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-shr_yang"))				irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_y] = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Sx"))					irt_top->rot_pars[task].Sx = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Sy"))					irt_top->rot_pars[task].Sy = atof(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-prj_mode"))				irt_top->rot_pars[task].proj_mode = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-prj_order"))			sprintf(irt_top->rot_pars[task].proj_order, "%s", argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-roll"))					irt_top->rot_pars[task].irt_angles[e_irt_angle_roll] = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-pitch"))				irt_top->rot_pars[task].irt_angles[e_irt_angle_pitch] = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-yaw"))					irt_top->rot_pars[task].irt_angles[e_irt_angle_yaw] = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Zd"))					irt_top->rot_pars[task].proj_Zd = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-Wd"))					irt_top->rot_pars[task].proj_Wd = atof(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-mesh_mode"))			irt_top->rot_pars[task].mesh_mode = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_order"))			irt_top->rot_pars[task].mesh_order = (bool)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_Sh"))				irt_top->rot_pars[task].mesh_Sh = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_Sv"))				irt_top->rot_pars[task].mesh_Sv = atof(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-mesh_format"))			irt_top->irt_desc[task].mesh_format = (Eirt_mesh_format_type)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_point"))			irt_top->irt_desc[task].mesh_point_location = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_rel"))				irt_top->irt_desc[task].mesh_rel_mode = (Eirt_mesh_rel_mode_type)atoi(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-dist_x"))				irt_top->rot_pars[task].dist_x = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-dist_y"))				irt_top->rot_pars[task].dist_y = atof(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-dist_r"))				irt_top->rot_pars[task].dist_r = atof(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-rate_mode"))			irt_top->irt_desc[task].rate_mode = (Eirt_rate_type)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-proc_size"))			irt_top->irt_desc[task].proc_size = (uint8_t)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-proc_auto"))			irt_top->rot_pars[task].proc_auto = (bool)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-lwf"))					irt_top->irt_desc[task].oimage_line_wr_format = (bool)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-bg_mode"))				irt_top->irt_desc[task].bg_mode = (Eirt_bg_mode_type)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-bg"))					irt_top->irt_desc[task].bg = (uint16_t)atoi(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-rot_buf_format"))		irt_top->rot_pars[task].buf_format[e_irt_block_rot] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-rot_buf_select"))		irt_top->rot_pars[task].buf_select[e_irt_block_rot] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-rot_buf_mode"))			irt_top->rot_pars[task].buf_mode[e_irt_block_rot]   = (uint8_t)atoi(argv[argcnt + 1]);

				if (!strcmp(argv[argcnt], "-mesh_buf_format"))		irt_top->rot_pars[task].buf_format[e_irt_block_mesh] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_buf_select"))		irt_top->rot_pars[task].buf_select[e_irt_block_mesh] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
				if (!strcmp(argv[argcnt], "-mesh_buf_mode"))		irt_top->rot_pars[task].buf_mode[e_irt_block_mesh]   = (uint8_t)atoi(argv[argcnt + 1]);

				if (task == 0 && argv[argcnt][0] == '-') {
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Switch %10s, value %s\n", argv[argcnt], argv[argcnt + 1]);
				}
			}

			irt_top->irt_desc[task].image_par[MIMAGE].H = irt_top->irt_desc[task].image_par[OIMAGE].H;
			irt_top->irt_desc[task].image_par[MIMAGE].W = irt_top->irt_desc[task].image_par[OIMAGE].W;

			irt_top->irt_desc[task].image_par[MIMAGE].S = irt_top->irt_desc[task].image_par[MIMAGE].W;

			//calculating image centers as default
			irt_top->irt_desc[task].image_par[OIMAGE].Yc = irt_top->irt_desc[task].image_par[OIMAGE].H - 1;
			irt_top->irt_desc[task].image_par[OIMAGE].Xc = irt_top->irt_desc[task].image_par[OIMAGE].W - 1;
			irt_top->irt_desc[task].image_par[IIMAGE].Yc = irt_top->irt_desc[task].image_par[IIMAGE].H - 1;
			irt_top->irt_desc[task].image_par[IIMAGE].Xc = irt_top->irt_desc[task].image_par[IIMAGE].W - 1;

			//parsing image centers
			for (int argcnt = 1; argcnt < argc; argcnt++) {
				if (!strcmp(argv[argcnt], "-Yco"))	irt_top->irt_desc[task].image_par[OIMAGE].Yc = (int16_t)(atof(argv[argcnt + 1]) * 2);
				if (!strcmp(argv[argcnt], "-Xco"))	irt_top->irt_desc[task].image_par[OIMAGE].Xc = (int16_t)(atof(argv[argcnt + 1]) * 2);
				if (!strcmp(argv[argcnt], "-Yci"))	irt_top->irt_desc[task].image_par[IIMAGE].Yc = (int16_t)(atof(argv[argcnt + 1]) * 2);
				if (!strcmp(argv[argcnt], "-Xci"))	irt_top->irt_desc[task].image_par[IIMAGE].Xc = (int16_t)(atof(argv[argcnt + 1]) * 2);
			}

			irt_top->rot_pars[task].WEIGHT_PREC		= irt_top->rot_pars[task].MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;// +1 because of 0.5 error, +2 because of interpolation, -1 because of weigths multiplication
			irt_top->rot_pars[task].COORD_PREC		= irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT; // +1 because of polynom in Xi/Yi calculation
			irt_top->rot_pars[task].COORD_ROUND		= (1 << (irt_top->rot_pars[task].COORD_PREC - 1));
			irt_top->rot_pars[task].TOTAL_PREC		= irt_top->rot_pars[task].COORD_PREC + irt_top->rot_pars[task].WEIGHT_PREC;
			irt_top->rot_pars[task].TOTAL_ROUND		= (1 << (irt_top->rot_pars[task].TOTAL_PREC - 1));
			irt_top->rot_pars[task].PROJ_NOM_PREC	= irt_top->rot_pars[task].TOTAL_PREC;
			irt_top->rot_pars[task].PROJ_DEN_PREC	= (irt_top->rot_pars[task].TOTAL_PREC + 10);
			irt_top->rot_pars[task].PROJ_NOM_ROUND	= (1 << (irt_top->rot_pars[task].PROJ_NOM_PREC - 1));
			irt_top->rot_pars[task].PROJ_DEN_ROUND	= (1 << (irt_top->rot_pars[task].PROJ_DEN_PREC - 1));

			irt_top->irt_desc[task].image_par[OIMAGE].Ps = irt_top->rot_pars[task].Pwo <= 8 ? 0 : 1;
			irt_top->irt_desc[task].image_par[IIMAGE].Ps = irt_top->rot_pars[task].Pwi <= 8 ? 0 : 1;
			irt_top->irt_desc[task].image_par[MIMAGE].Ps = irt_top->irt_desc[task].mesh_format == 0 ? 2 : 3;
			irt_top->irt_desc[task].Msi       = ((1 << irt_top->rot_pars[task].Pwi) - 1) << irt_top->rot_pars[task].Ppi;
			irt_top->rot_pars[task].bli_shift_fix = (2 * irt_top->rot_pars[task].TOTAL_PREC) - (irt_top->rot_pars[task].Pwo - (irt_top->rot_pars[task].Pwi + irt_top->rot_pars[task].Ppi)) - 1; //bi-linear interpolation shift
			irt_top->irt_desc[task].bli_shift     = irt_top->rot_pars[task].Pwo - (irt_top->rot_pars[task].Pwi + irt_top->rot_pars[task].Ppi);
			irt_top->irt_desc[task].MAX_VALo  = (1 << irt_top->rot_pars[task].Pwo) - 1; //output pixel max value, equal to 2^pixel_width - 1

			if (task == 0) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "-----------------------------------\n");
				for (int argcnt = 0; argcnt < argc; argcnt++) {
					IRT_TRACE_TO_RES_UTILS(test_res, "%s ", argv[argcnt]);
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s ", argv[argcnt]);
				}
				IRT_TRACE_TO_RES_UTILS(test_res, " - ");
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\n");
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "-----------------------------------\n");
			}

			//irt_rotation_memory_fit_check(irt_top->irt_cfg, irt_top->rot_pars[task], irt_top->irt_desc[task], task);

			if (irt_top->irt_cfg.flow_mode == e_irt_flow_wCFIFO_fixed_adaptive_2x2 && irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_wxh ||
				irt_top->irt_cfg.flow_mode == e_irt_flow_wCFIFO_fixed_adaptive_wxh && irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_2x2 ||
				irt_top->irt_cfg.flow_mode == e_irt_flow_nCFIFO_fixed_adaptive_wxh && irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_2x2) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "%s flow and %s rate control are not supported together\n", irt_flow_mode_s[irt_top->irt_cfg.flow_mode], irt_rate_mode_s[irt_top->irt_desc[task].rate_mode]);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, %s flow and %s rate control are not supported together\n", irt_flow_mode_s[irt_top->irt_cfg.flow_mode], irt_rate_mode_s[irt_top->irt_desc[task].rate_mode]);
				IRT_CLOSE_FAILED_TEST(0);
			}
		}
	//}
		if (test_file_flag) {
			if (irt_top->rot_pars[image].Pwi != 8 || irt_top->rot_pars[image].Ppi != 0) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Pwi = %d and Ppi = %d for image %d are not supported together, must be 8 and 0 in multi-descriptor test\n",
					irt_top->rot_pars[image].Pwi, irt_top->rot_pars[image].Ppi, image);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, Pwi = %d and Ppi = %d for image %d are not supported together, must be 8 and 0 in multi-descriptor test", 
					irt_top->rot_pars[image].Pwi, irt_top->rot_pars[image].Ppi, image);
				IRT_CLOSE_FAILED_TEST(0);
			}
		}
}

#endif

#if defined(STANDALONE_ROTATOR) || defined(CREATE_IMAGE_DUMPS)
void WriteBMP(IRT_top* irt_top, uint8_t image, uint8_t planes, char* filename, uint32_t width, uint32_t height, uint16_t*** out_image)
{
	uint8_t desc = image * planes;
	FILE* f = fopen(filename, "wb");

	//if(f == nullptr)
	  //  throw "Argument Exception";

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Writing file %s %dx%d\n", filename, width, height);

	uint32_t row_padded = (width * 3 + 3) & (~3);
	uint8_t* data = new uint8_t[row_padded];

	generate_bmp_header(f, width, height, row_padded, 3);

	uint32_t row = 0;

	//for(int32_t i = height-1; i >= 0; i--) //image array is swapped in BMP file
	for (uint32_t i = 0; i < height; i++) {//image array is swapped in BMP file
		if (irt_bmp_wr_order == e_irt_bmp_H20)
			row = height - 1 - i;
		else
			row = i;

		for (uint32_t j = 0; j < width; j++) {
			for (uint8_t idx = 0; idx < planes; idx++) {
				data[j * 3 + idx] = (uint8_t)(out_image[idx][row][j] >> (irt_top->rot_pars[desc].Pwo - 8 + irt_top->irt_desc[desc].Ppo));
			}
#if 1
			if (planes < 3) {
				for (uint8_t idx = planes; idx < 3; idx++) {
					data[j * 3 + idx] = (uint8_t)(out_image[0][row][j] >> (irt_top->rot_pars[desc].Pwo - 8 + irt_top->irt_desc[desc].Ppo));
				}
			}
#endif
		}
		fwrite(data, sizeof(uint8_t), row_padded, f);

	}

	fclose(f);
}

void generate_bmp_header(FILE* file, uint32_t width, uint32_t height, uint32_t row_padded, uint8_t planes) {

	uint32_t filesize = 54 + row_padded * height;

	uint8_t bmpfilehdr[14] = { 'B','M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	uint8_t bmpinfohdr[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	bmpfilehdr[2] = (uint8_t)(filesize);
	bmpfilehdr[3] = (uint8_t)(filesize >> 8);
	bmpfilehdr[4] = (uint8_t)(filesize >> 16);
	bmpfilehdr[5] = (uint8_t)(filesize >> 24);

	bmpinfohdr[irt_dib_header_image_width]		= (uint8_t)(width);
	bmpinfohdr[irt_dib_header_image_width + 1]  = (uint8_t)(width >> 8);
	bmpinfohdr[irt_dib_header_image_width + 2]  = (uint8_t)(width >> 16);
	bmpinfohdr[irt_dib_header_image_width + 3]  = (uint8_t)(width >> 24);
	bmpinfohdr[irt_dib_header_image_height]		= (uint8_t)(height);
	bmpinfohdr[irt_dib_header_image_height + 1] = (uint8_t)(height >> 8);
	bmpinfohdr[irt_dib_header_image_height + 2] = (uint8_t)(height >> 16);
	bmpinfohdr[irt_dib_header_image_height + 3] = (uint8_t)(height >> 24);
	bmpinfohdr[irt_dib_header_image_bpp]		= 8 * planes;
	bmpinfohdr[irt_dib_header_bitmap_size]		= (uint8_t)(row_padded * height);
	bmpinfohdr[irt_dib_header_bitmap_size + 1]	= (uint8_t)((row_padded * height) >> 8);

	fwrite(bmpfilehdr, 1, 14, file);
	fwrite(bmpinfohdr, 1, 40, file);
}
#endif

#if defined(STANDALONE_ROTATOR)
void ReadBMP(IRT_top *irt_top, char* filename, char* outdirname, uint32_t& width, uint32_t& height, int argc, char* argv[])
{
    uint8_t pixel_width = 8, pixel_padding = 0;
    FILE* f = fopen(filename, "rb");

	//printf ("ReadBMP: %s, %x\n", filename, f);
    //if(f == nullptr)
      //  throw "Argument Exception";

    uint8_t bmpfilehdr[14], bmpinfohdr[40];
    fread(bmpfilehdr, sizeof(uint8_t), 14, f); // read the 54-byte header
	fread(bmpinfohdr, sizeof(uint8_t), 40, f); // read the 54-byte header

    // extract image height and width from header
    width  = *(uint32_t*)&bmpinfohdr[irt_dib_header_image_width];
    height = *(uint32_t*)&bmpinfohdr[irt_dib_header_image_height];
	uint32_t planes = (*(uint16_t*)&bmpinfohdr[irt_dib_header_image_bpp]) >> 3;

	uint32_t Wi = width;
	uint32_t Hi = height;

	for (int argcnt = 1; argcnt < argc; argcnt++) {
		if (!strcmp(argv[argcnt], "-Pwi"))	pixel_width   = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-Ppi"))	pixel_padding = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-rand_input_image"))	rand_input_image = 1;
		if (!strcmp(argv[argcnt], "-Hi"))	Hi = (uint16_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-Wi"))	Wi = (uint16_t)atoi(argv[argcnt + 1]);
	}

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Reading file %s %dx%d\n", filename, width, height);

	if (width > IIMAGE_W || height > IIMAGE_H) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen error: read image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
			width, height, IIMAGE_W, IIMAGE_H);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, read image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
			width, height, IIMAGE_W, IIMAGE_H);
		IRT_CLOSE_FAILED_TEST(0);
	}
	if (Wi > IIMAGE_W || Hi > IIMAGE_H) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen error: input image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
			Wi, Hi, IIMAGE_W, IIMAGE_H);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, input image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
			Wi, Hi, IIMAGE_W, IIMAGE_H);
		IRT_CLOSE_FAILED_TEST(0);
	}

	uint32_t row_padded = (width * planes + 3) & (~3);
	uint8_t* data = new uint8_t[row_padded];

	uint8_t Psi = pixel_width <= 8 ? 1 : 2;
	uint64_t pixel_addr;

	uint8_t bits_to_add = pixel_width - 8;
	uint32_t row = 0;

	//filling input memory array and ext mem array with initial data
	uint32_t Himax = (uint32_t)std::max((int32_t)Hi, (int32_t)height);
	uint32_t Wimax = (uint32_t)std::max((int32_t)Wi, (int32_t)width);
	for (uint32_t i = 0; i < Himax; i++) {
		row = i;
		for (uint32_t j = 0; j < Wimax; j++) {
			for (uint8_t idx = 0; idx < PLANES; idx++) {
				pixel_addr = irt_top->irt_desc[0].image_par[IIMAGE].addr_start + (uint64_t)idx * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL; //component base address
				pixel_addr += (uint64_t)row * Wimax * Psi; //line base address
				pixel_addr += (uint64_t)j * Psi; //pixel address
				data[idx] = rand() % 255; //irt_colors[i % NUM_OF_COLORS][idx];
				//data[j + idx] = idx * 50 + 50 + rand() % 10;
				input_image[idx][row][j] = (((uint16_t)data[idx] << bits_to_add) + (rand() % (1 << bits_to_add))) << pixel_padding;
				input_image[idx][row][j] |= rand() % (1 << pixel_padding); //garbage on LSB
				input_image[idx][row][j] |= (rand() % (1 << (16 - pixel_width - pixel_padding))) << (pixel_width + pixel_padding);

				//input_image[idx][i][j / 3] = j / 3;

				ext_mem[pixel_addr] = (uint8_t)(input_image[idx][row][j] & 0xff);
				//ext_mem[pixel_addr] = (unsigned char)(((i - (height - 1)) * 100 + ((j/3)%100)));
				if (Psi == 2) {
					ext_mem[pixel_addr + 1] = (uint8_t)((input_image[idx][row][j] >> 8) & 0xff); //store 2nd byte at next address
				}
			}
		}
	}
    //for(int32_t i = height-1; i >= 0; i--) //image array is swapped in BMP file
	for (uint32_t i = 0; i < height; i++) //image array is swapped in BMP file
    {
		if (irt_bmp_rd_order == e_irt_bmp_H20)
			row = height - 1 - i;
		else
			row = i;

        fread(data, sizeof(uint8_t), row_padded, f);

		for (uint32_t j = 0; j < width; j++) {

			for (uint8_t idx = 0; idx < PLANES; idx++) {
				pixel_addr = irt_top->irt_desc[0].image_par[IIMAGE].addr_start + (uint64_t)idx * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL; //component base address
				pixel_addr += (uint64_t)row * Wimax * Psi; //line base address
				pixel_addr += (uint64_t)j * Psi; //pixel address
				if (rand_input_image) {
					data[j * planes + idx] = rand() % 255; //irt_colors[i % NUM_OF_COLORS][idx];
				}
				//data[j + idx] = idx * 50 + 50 + rand() % 10;
				input_image[idx][row][j] = (((uint16_t)data[j * planes + idx] << bits_to_add) + (rand() % (1 << bits_to_add))) << pixel_padding;
				input_image[idx][row][j] |= rand() % (1 << pixel_padding); //garbage on LSB
				input_image[idx][row][j] |= (rand() % (1 << (16 - pixel_width - pixel_padding))) << (pixel_width + pixel_padding);

				//input_image[idx][i][j / 3] = j / 3;

				ext_mem[pixel_addr] = (uint8_t)(input_image[idx][row][j] & 0xff);
				//ext_mem[pixel_addr] = (unsigned char)(((i - (height - 1)) * 100 + ((j/3)%100)));
				if (Psi == 2) {
					ext_mem[pixel_addr + 1] = (uint8_t)((input_image[idx][row][j] >> 8) & 0xff); //store 2nd byte at next address
				}
			}

			//input_image16[i * width*3 + j]   = (unsigned short)(data[j]);
			//input_image16[i * width*3 + j+1] = (unsigned short)(data[j+1]);
			//input_image16[i * width*3 + j+2] = (unsigned short)(data[j+2]);

	       	//input_image[0][i][j/3] = 0xaa; 	ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = 0xaa;
        	//input_image[1][i][j/3] = 0xbb;   ext_mem[image_pars[IIMAGE].ADDR+1*IMAGE_H*IMAGE_W+i*width+j/3] = 0xbb;
        	//input_image[2][i][j/3] = 0xcc;   ext_mem[image_pars[IIMAGE].ADDR+2*IMAGE_H*IMAGE_W+i*width+j/3] = 0xcc;

//			input_image[0][i][j/3] = (i)&0xff;//((i&0xf)<<4)|((j/3)&0xf);
//			ext_mem[irt_top->irt_desc[0].image_par[IIMAGE].addr_start+0*IMAGE_H*IMAGE_W+i*width+j/3] = (i)&0xff;//((i&0xf)<<4)|((j/3)&0xf);
//			input_image[2][i][j/3] = (j/3)&0xff;//((i&0xf)<<4)|((j/3)&0xf);
//			ext_mem[image_pars[IIMAGE].ADDR+2*IMAGE_H*IMAGE_W+i*width+j/3] = (j/3)&0xff;//((i&0xf)<<4)|((j/3)&0xf);

/*
			if (j%2 == 0)
				ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = (unsigned char)i;
			else
				ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = (unsigned char) (j/3);
*/
			}
    }

    fclose(f);
	width = Wimax;
	height = Himax;

	if (print_out_files) {
		if (num_of_images > 1) {
			generate_image_dump("IRT_dump_in_image%d_plane%d.txt", outdirname, width, height, (Eirt_bmp_order)0, input_image, 0); //input_image stored according to irt_bmp_rd_order, dumped as is
		} else {
			generate_image_dump("IRT_dump_in_plane%d.txt", outdirname, width, height, (Eirt_bmp_order)0, input_image, 0); //input_image stored according to irt_bmp_rd_order, dumped as is
		}
	}
}

void irt_quality_comp(IRT_top* irt_top, uint8_t image) {

	uint8_t desc = image * PLANES;
	uint16_t Ho = irt_top->irt_desc[desc].image_par[OIMAGE].H;
	uint16_t So = irt_top->irt_desc[desc].image_par[OIMAGE].S;

	//0 - fixed2double, 1 - fixed2float, 2 - double2float, 3 - float2fix16
	int non_bg_pxl_cnt[CALC_FORMATS] = { 0 }, pixel_err_cnt[CALC_FORMATS] = { 0 }, pixel_err_max[CALC_FORMATS] = { 0 }, pixel_err_hist[CALC_FORMATS][PIXEL_ERR_BINS] = { 0 }, pixel_err_max_idx[CALC_FORMATS][2] = { 0 };
	int pxl_error;
	bool non_bg_flag;

	double mse[CALC_FORMATS] = { 0 };
	for (uint8_t format = e_irt_crdf_arch; format <= e_irt_crdf_fix16; format++) {
		if (format != e_irt_crdf_fpt64) {
			for (uint16_t row = 0; row < Ho; row++) {
				for (uint16_t col = 0; col < So; col++) {
					for (uint8_t idx = 0; idx < PLANES; idx++) {
						non_bg_flag = output_image[image * CALC_FORMATS_ROT + e_irt_crdf_fpt64][idx][row][col] != irt_top->irt_desc[image].bg;
						pxl_error = abs(output_image[image * CALC_FORMATS_ROT + format][idx][row][col] - output_image[image * CALC_FORMATS_ROT + e_irt_crdf_fpt64][idx][row][col]);
						if (non_bg_flag) {//not count bg pixels
							non_bg_pxl_cnt[format]++;
							mse[format] += pow((double)pxl_error, 2);
							if (pxl_error > 0) {
								pixel_err_cnt[format]++;
								pixel_err_hist[format][std::min(pxl_error, PIXEL_ERR_BINS - 1)] ++;
								if (pxl_error > pixel_err_max[format]) {
									pixel_err_max_idx[format][0] = col;
									pixel_err_max_idx[format][1] = row;
								}
								pixel_err_max[format] = std::max(pixel_err_max[format], pxl_error);
							}
						}
					}
				}
			}
		}
	}

	if (desc == 0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "| Image | Format | Coordinates selection |  Pixels errors  |  PSNR  |     Max Error    | Non bg pixels |\n");
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
	}

	double psnr[CALC_FORMATS];

	uint8_t last_format = e_irt_crdf_fixed;
	switch (irt_top->irt_desc[desc].irt_mode) {
	case e_irt_rotation:	last_format = e_irt_crdf_fix16; break;
	case e_irt_affine:		last_format = e_irt_crdf_fixed; break;
	case e_irt_projection:
		switch (irt_top->rot_pars[desc].proj_mode) {
		case e_irt_rotation:	last_format = e_irt_crdf_fpt32; break;
		case e_irt_affine:		last_format = e_irt_crdf_fpt32; break;
		case e_irt_projection:  last_format = e_irt_crdf_fpt32; break;
		}
		break;
	case e_irt_mesh:
		switch (irt_top->rot_pars[desc].mesh_mode) {
		case e_irt_rotation:	last_format = e_irt_crdf_fix16; break;
		case e_irt_affine:		last_format = e_irt_crdf_fixed; break;
		case e_irt_projection:  last_format = e_irt_crdf_fpt32; break;
		case e_irt_mesh:		last_format = e_irt_crdf_fpt32; break;
		}
		if (irt_top->rot_pars[desc].dist_r == 1)
			last_format = e_irt_crdf_fpt32;
		break;
	}

	for (uint8_t format = e_irt_crdf_arch; format <= last_format; format++) {
		mse[format] = mse[format] / ((double)non_bg_pxl_cnt[format]);
		psnr[format] = 10.0 * log10(pow((double)irt_top->irt_desc[desc].MAX_VALo, 2) / mse[format]);
		if (format != e_irt_crdf_fpt64) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "|   %d   | %s  |", image, irt_hl_format_s[format]);
			if (coord_err[format] != 0) {
				//IRT_TRACE_UTILS("Image %d double and %s coordinate difference %d (%2.2f%%)\n", image, irt_hl_format_s[format], coord_err[format], 100.0 * (float)coord_err[format] / (Ho * Wo));
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, " Error %5d (%6.2f%%) |", coord_err[format], 100.0 * (float)coord_err[format] / (Ho * So));
			} else {
				//IRT_TRACE_UTILS("Image %d double and %s coordinate selection match\n", image, irt_hl_format_s[format]);
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "         Match         |");
			}

			if (pixel_err_cnt[format] == 0) {
				//IRT_TRACE_UTILS("Image %d double and %s results are bit exact\n", image, irt_hl_format_s[format]);
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "    Bit exact    |  High  |         0        |");
			} else {
#if 0
				IRT_TRACE_UTILS("Image %d double and %s difference: ", image, irt_hl_format_s[format]);
				IRT_TRACE_UTILS("error count = %d(%2.2f%%), psnr = %.2f , max_error = %d at [%d,%d] from %d non bg pixel components\n",
					pixel_err_cnt[format], 100.0 * (double)pixel_err_cnt[format] / ((double)non_bg_pxl_cnt[format]),
					psnr[format], pixel_err_max[format], pixel_err_max_idx[format][0], pixel_err_max_idx[format][1], non_bg_pxl_cnt[format]);
				for (int bin = 0; bin < 16; bin++)
					IRT_TRACE_UTILS("%d ", pixel_err_hist[format][bin]);
				IRT_TRACE_UTILS("\n");
#endif
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, " %7d(%5.2f%%) | %6.2f | %5d[%4d,%4d] |",
					pixel_err_cnt[format], 100.0 * (double)pixel_err_cnt[format] / ((double)non_bg_pxl_cnt[format]),
					psnr[format], pixel_err_max[format], pixel_err_max_idx[format][0], pixel_err_max_idx[format][1]);
			}

			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, " %8d(%3.0f%%)|\n", non_bg_pxl_cnt[format], (float)100.0 * non_bg_pxl_cnt[format]/(3 * Ho * So));
		}
	}
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
}

void output_image_dump (IRT_top *irt_top, uint8_t image, uint8_t planes, char* output_file, char* dirname) {

	char out_file_name[150] = { '\0' };

	strncpy(out_file_name, dirname, strlen(dirname));
	strncat(out_file_name, output_file, strlen(output_file));

	if (num_of_images > 1) {
		(void)sprintf(out_file_name, "%s_%d.bmp", out_file_name, image);
	} else {
		(void)sprintf(out_file_name, "%s.bmp", out_file_name);
	}

	uint8_t image_first_desc = image * planes;
	FILE* f = nullptr;

	uint16_t*** out_image = nullptr;
	out_image = new uint16_t** [PLANES];
	for (uint8_t plain = 0; plain < PLANES; plain++) {
		(out_image)[plain] = new uint16_t* [irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H];
		for (uint16_t row = 0; row < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H; row++) {
			(out_image)[plain][row] = new uint16_t[irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S];
		}
	}
	for (uint8_t plain = 0; plain < PLANES; plain++) {
		for (uint16_t row = 0; row < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H; row++) {
			for (uint16_t col = 0; col < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S; col++) {
				(out_image)[plain][row][col] = 0;
			}
		}
	}

	uint16_t row_padded = (irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S * 3 + 3) & (~3);

	if (print_out_files) {
		f = fopen(out_file_name, "wb");
		generate_bmp_header(f, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H, row_padded, 3);
	}

	uint8_t* data = new uint8_t[row_padded];

	uint32_t error_count = 0;
	uint16_t pixel_val[3] = { 0 };
	uint64_t pixel_addr = 0;
	bool pixel_error = false;

	uint16_t row = 0;
    //for(int16_t i = irt_top->irt_desc[desc].image_par[OIMAGE].H-1; i >= 0 ; i--) {//image array is swapped in BMP file
	for (uint16_t i = 0; i < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H; i++) {//image array is swapped in BMP file
		if (irt_bmp_wr_order == e_irt_bmp_H20)
			row = irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H - 1 - i;
		else
			row = i;

        for(uint16_t j = 0; j < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S; j++) {
			for (uint8_t idx = 0; idx < planes; idx++) {
				pixel_addr = irt_top->irt_desc[image_first_desc + idx].image_par[OIMAGE].addr_start;// +(uint64_t)idx * IMAGE_W * IMAGE_H; //component base address
				pixel_addr += ((uint64_t)row * irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S) << irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps; //line base address
				pixel_addr += ((uint64_t)j) << irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps; //pixel address
				pixel_val[idx] = (uint16_t)ext_mem[pixel_addr];
				if (irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps) {
					pixel_val[idx] += (ext_mem[pixel_addr + 1] << 8); //concat 2nd byte at MSB
				}
				(out_image)[idx][row][j] = pixel_val[idx];
			}
			uint16_t k = j;
			//data[k]     = ext_mem[irt_top->irt_desc[desc].image_par[OIMAGE].addr_start + (uint64_t)0 * IMAGE_W * IMAGE_H + (uint64_t)row * irt_top->irt_desc[desc].image_par[OIMAGE].W + (uint64_t)j / 3];
			//data[k + 1] = ext_mem[irt_top->irt_desc[desc].image_par[OIMAGE].addr_start + (uint64_t)1 * IMAGE_W * IMAGE_H + (uint64_t)row * irt_top->irt_desc[desc].image_par[OIMAGE].W + (uint64_t)j / 3];
			//data[k + 2] = ext_mem[irt_top->irt_desc[desc].image_par[OIMAGE].addr_start + (uint64_t)2 * IMAGE_W * IMAGE_H + (uint64_t)row * irt_top->irt_desc[desc].image_par[OIMAGE].W + (uint64_t)j / 3];

			//data[k]     = (uint8_t)(pixel_val[0] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 + irt_top->irt_desc[image_first_desc].Ppo));
			//data[k + 1] = (uint8_t)(pixel_val[1] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 + irt_top->irt_desc[image_first_desc].Ppo));
			//data[k + 2] = (uint8_t)(pixel_val[2] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 + irt_top->irt_desc[image_first_desc].Ppo));

			//compare output image with reference image
			pixel_error = 0;
			for (uint8_t idx = 0; idx < planes; idx++) {
				data[k * 3 + idx] = (uint8_t)(pixel_val[idx] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 + irt_top->irt_desc[image_first_desc].Ppo));
				if (pixel_val[idx] != output_image[image * CALC_FORMATS_ROT][idx][row][j]) {
					pixel_error = 1;					
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Pixel error at [%d, %d][%d]: HL [0x%x] LL [0x%x]\n", row, j, idx, output_image[image * CALC_FORMATS_ROT][idx][row][j], pixel_val[idx]);
				}
			}

			if (planes < 3) {
				for (uint8_t idx = planes; idx < 3; idx++) {
					data[j * 3 + idx] = (uint8_t)(pixel_val[0] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 + irt_top->irt_desc[image_first_desc].Ppo));
				}
			}
			error_count += pixel_error;
        }
		if (print_out_files)
			fwrite(data, sizeof(uint8_t), row_padded, f);

    }
	if (print_out_files)
		fclose(f);

	if (print_out_files) {
		if (num_of_images > 1) {
			generate_image_dump("IRT_dump_out_image%d_plane%d.txt", dirname, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H, (Eirt_bmp_order)0, out_image, image); //out_image stored according to irt_bmp_wr_order, dumped as is
		} else {
			generate_image_dump("IRT_dump_out_plane%d.txt", dirname, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H, (Eirt_bmp_order)0, out_image, image); //out_image stored according to irt_bmp_wr_order, dumped as is
		}
	}
	for (uint8_t plain = 0; plain < PLANES; plain++) {
		for (uint16_t row = 0; row < irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H; row++) {
			delete[] out_image[plain][row];
			out_image[plain][row] = nullptr;
		}
		delete[] out_image[plain];
		out_image[plain] = nullptr;
	}
	delete[] out_image;
	out_image = nullptr;

	//fprintf(test_res, "%s %3.2f %d %d", file_name, rot_angle, irt_top->irt_desc[0].image_par[OIMAGE].H, irt_top->irt_desc[0].image_par[OIMAGE].W);
	IRT_TRACE_TO_RES_UTILS(test_res, " image %d %dx%d stripe %dx%d", image,
		irt_top->irt_desc[image_first_desc].Wo, irt_top->irt_desc[image_first_desc].Ho,
		irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);

	if (error_count==0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Image %d %dx%d stripe %dx%d", image,
			irt_top->irt_desc[image_first_desc].Wo, irt_top->irt_desc[image_first_desc].Ho,
			irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);
		IRT_TRACE_UTILS (IRT_TRACE_LEVEL_INFO, " passed\n");
		IRT_TRACE_TO_RES_UTILS (test_res, " passed");
	} else {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Image %d %dx%d stripe %dx%d", image,
			irt_top->irt_desc[image_first_desc].Wo, irt_top->irt_desc[image_first_desc].Ho,
			irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);
		IRT_TRACE_UTILS (IRT_TRACE_LEVEL_ERROR, " error count = %d\n", error_count);
		IRT_TRACE_TO_RES_UTILS (test_res, " error count = %d", error_count);
	}

	//fclose (test_res);
}

void generate_plane_dump(const char* filename, char* dirname, uint32_t width, uint32_t height, Eirt_bmp_order vert_order, uint16_t** image_ptr, uint8_t image, uint8_t plane) {

	FILE* f = nullptr;
	uint32_t row = 0;
	char fname[150] = { '\0' };

	strncpy(fname, dirname, strlen(dirname));
	strncat(fname, filename, strlen(filename));

	if (num_of_images > 1) {
		sprintf(fname, fname, image, plane);
	} else {
		sprintf(fname, fname, plane);
	}

	f = fopen(fname, "w");

	for (uint32_t i = 0; i < height; i++) {
		if (vert_order == e_irt_bmp_H20)
			row = height - 1 - i;
		else
			row = i;

		for (uint32_t j = 0; j < width; j++) {
			fprintf(f, "%04x ", (uint32_t)image_ptr[row][j]);
		}
		fprintf(f, "// row %d\n", i);
	}

	fclose(f);
}

void generate_image_dump(const char* filename, char* dirname, uint32_t width, uint32_t height, Eirt_bmp_order vert_order, uint16_t*** image_ptr, uint8_t image) {

	for (uint8_t idx = 0; idx < PLANES; idx++) {
		generate_plane_dump(filename, dirname, width, height, vert_order, image_ptr[idx], image, idx);
	}
}

#endif

void irt_mems_modes_check(const rotation_par &rot_pars, const rotation_par &rot_pars0, uint8_t desc) {

	//rot memory check IRT_RM_CLIENTS
	for (uint8_t mem_client = 0; mem_client < IRT_RM_CLIENTS; mem_client++) {
		if (rot_pars.buf_format[mem_client] != rot_pars0.buf_format[mem_client]) { 
			//buf format is non consistant
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: desc %d %s memory %s buffer format is different from desc 0 %s buffer format\n",
				desc, irt_mem_type_s[mem_client], irt_buf_format_s[rot_pars.buf_format[mem_client]], irt_buf_format_s[rot_pars0.buf_format[mem_client]]);
			IRT_TRACE_TO_RES_UTILS(test_res, " was not run, desc %d %s memory %s buffer format is different from desc 0 %s buffer format\n",
				desc, irt_mem_type_s[mem_client], irt_buf_format_s[rot_pars.buf_format[mem_client]], irt_buf_format_s[rot_pars0.buf_format[mem_client]]);
			IRT_CLOSE_FAILED_TEST(0);
		}

		if (rot_pars0.buf_format[mem_client] == e_irt_buf_format_static) {
			//buf format is static, buf auto select is not allowed and all buffer modes must be same
			if (rot_pars0.buf_select[mem_client] == e_irt_buf_select_auto) {
				//buf auto select is not allowed
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: desc %d %s memory buf_select is set to auto that is not allowed in static buf_format\n", desc, irt_mem_type_s[mem_client]);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, desc %d %s memory buf_select is set to auto that is not allowed in static buf_format\n", desc, irt_mem_type_s[mem_client]);
				IRT_CLOSE_FAILED_TEST(0);
			}

			if (rot_pars.buf_mode[mem_client] != rot_pars0.buf_mode[mem_client]) {
				//buf mode is non consistant
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: desc %d %s memory buffer mode %d is different from desc 0 buffer format %d in static buf_format\n",
					desc, irt_mem_type_s[mem_client], rot_pars.buf_mode[mem_client], rot_pars0.buf_mode[mem_client]);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, desc %d %s memory buffer mode %d is different from desc 0 buffer format %d in static buf_format\n",
					desc, irt_mem_type_s[mem_client], rot_pars.buf_mode[mem_client], rot_pars0.buf_mode[mem_client]);
				IRT_CLOSE_FAILED_TEST(0);
			}
		}
	}
}

void irt_params_analize(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {


	rot_pars.affine_flags[e_irt_aff_rotation]   = (strchr(rot_pars.affine_mode, 'R') != nullptr);
	rot_pars.affine_flags[e_irt_aff_scaling]    = (strchr(rot_pars.affine_mode, 'S') != nullptr);
	rot_pars.affine_flags[e_irt_aff_reflection] = (strchr(rot_pars.affine_mode, 'M') != nullptr);
	rot_pars.affine_flags[e_irt_aff_shearing]   = (strchr(rot_pars.affine_mode, 'T') != nullptr);

	rot_pars.affine_rotation            = strcmp(rot_pars.affine_mode, "R") == 0;
	rot_pars.projection_direct_rotation = rot_pars.proj_mode == e_irt_rotation;
	rot_pars.projection_affine_rotation = rot_pars.proj_mode == e_irt_affine && rot_pars.affine_rotation;
	rot_pars.projection_rotation		= rot_pars.projection_direct_rotation | rot_pars.projection_affine_rotation;
	rot_pars.mesh_direct_rotation		= rot_pars.mesh_mode == e_irt_rotation;
	rot_pars.mesh_affine_rotation		= rot_pars.mesh_mode == e_irt_affine && rot_pars.affine_rotation;
	rot_pars.mesh_projection_rotation	= rot_pars.mesh_mode == e_irt_projection && rot_pars.projection_rotation;
	rot_pars.mesh_rotation				= rot_pars.mesh_direct_rotation | rot_pars.mesh_affine_rotation | rot_pars.mesh_projection_rotation;

	rot_pars.projection_affine			= rot_pars.proj_mode == e_irt_affine;
	rot_pars.mesh_direct_affine			= rot_pars.mesh_mode == e_irt_affine;
	rot_pars.mesh_projection_affine		= rot_pars.mesh_mode == e_irt_projection && rot_pars.projection_affine;
	rot_pars.mesh_affine				= rot_pars.mesh_direct_affine | rot_pars.mesh_projection_affine;

	rot_pars.mesh_dist_r0_Sh1_Sv1 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv == 1;
	//rot_pars.mesh_dist_r0_Sh1_Sv1_fp32 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv == 1;//&& irt_desc.mesh_format == e_irt_mesh_fp32;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "affine_rotation            = %d\n", rot_pars.affine_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "projection_direct_rotation = %d\n", rot_pars.projection_direct_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "projection_affine_rotation = %d\n", rot_pars.projection_affine_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "projection_rotation        = %d\n", rot_pars.projection_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_direct_rotation       = %d\n", rot_pars.mesh_direct_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_affine_rotation       = %d\n", rot_pars.mesh_affine_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_projection_rotation   = %d\n", rot_pars.mesh_projection_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_rotation              = %d\n", rot_pars.mesh_rotation);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "projection_affine          = %d\n", rot_pars.projection_affine);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_direct_affine         = %d\n", rot_pars.mesh_direct_affine);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_projection_affine     = %d\n", rot_pars.mesh_projection_affine);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "mesh_affine                = %d\n", rot_pars.mesh_affine);

	rot_pars.irt_rotate = irt_desc.irt_mode == e_irt_rotation	||
						 (irt_desc.irt_mode == e_irt_affine		&& rot_pars.affine_rotation)	 ||
						 (irt_desc.irt_mode == e_irt_projection && rot_pars.projection_rotation) ||
						 (irt_desc.irt_mode == e_irt_mesh		&& rot_pars.mesh_rotation && rot_pars.mesh_dist_r0_Sh1_Sv1);

	rot_pars.irt_affine = irt_desc.irt_mode == e_irt_affine		||
						 (irt_desc.irt_mode == e_irt_projection && rot_pars.projection_affine)	 ||
						 (irt_desc.irt_mode == e_irt_mesh		&& rot_pars.mesh_affine   && rot_pars.mesh_dist_r0_Sh1_Sv1);

	double cotx = rot_pars.shear_mode & 1 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_x] * M_PI / 180.0) : 0;
	double coty = rot_pars.shear_mode & 2 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_y] * M_PI / 180.0) : 0;

	rot_pars.irt_affine_hscaling = rot_pars.affine_flags[e_irt_aff_scaling] == 1 && rot_pars.Sx != 1;
	rot_pars.irt_affine_vscaling = rot_pars.affine_flags[e_irt_aff_scaling] == 1 && rot_pars.Sy != 1;
	rot_pars.irt_affine_shearing = rot_pars.affine_flags[e_irt_aff_shearing] == 1 && ((cotx != 0 && cotx != 1) || (coty != 0 && coty != 1));
	rot_pars.irt_affine_st_inth  = (rot_pars.irt_affine_hscaling || rot_pars.irt_affine_shearing) && rot_pars.irt_affine;
	rot_pars.irt_affine_st_intv  = (rot_pars.irt_affine_vscaling || rot_pars.irt_affine_shearing) && rot_pars.irt_affine;
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "irt_affine_hscaling        = %d\n", rot_pars.irt_affine_hscaling);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "irt_affine_vscaling        = %d\n", rot_pars.irt_affine_vscaling);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "irt_affine_shearing        = %d\n", rot_pars.irt_affine_shearing);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "irt_affine_st_inth         = %d\n", rot_pars.irt_affine_st_inth);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "irt_affine_st_intv         = %d\n", rot_pars.irt_affine_st_intv);
}

void irt_rot_angle_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	double irt_angle_tmp[e_irt_angle_shr_y + 1];
	for (uint8_t angle_type = e_irt_angle_roll; angle_type <= e_irt_angle_shr_y; angle_type++) {
		switch (angle_type) {
		case e_irt_angle_roll:
		case e_irt_angle_pitch:
		case e_irt_angle_yaw:
		case e_irt_angle_rot:
		case e_irt_angle_shr_x:
		case e_irt_angle_shr_y:
			if (rot_pars.irt_angles[angle_type] > 360 || rot_pars.irt_angles[angle_type] < -360) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "%s angle %f is not supported, provide angle in [-360:360] range\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, %s angle %f is not supported\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
				IRT_CLOSE_FAILED_TEST(0);
			}
			if (rot_pars.irt_angles[angle_type] < 0)
				irt_angle_tmp[angle_type] = rot_pars.irt_angles[angle_type] + 360; //converting angles to positive range
			else
				irt_angle_tmp[angle_type] = rot_pars.irt_angles[angle_type];
			if (irt_angle_tmp[angle_type] >= 270) irt_angle_tmp[angle_type] -= 360;//converting [270:360] to [-90:0]
			break;
		}
	}

	if ( irt_desc.irt_mode == e_irt_rotation ||
		(irt_desc.irt_mode == e_irt_affine  &&  strchr(rot_pars.affine_mode, 'R') != nullptr) ||
		(irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_rotation) ||
		(irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine && strchr(rot_pars.affine_mode, 'R') != nullptr) ||
		(irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_rotation) ||
		(irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_affine && strchr(rot_pars.affine_mode, 'R') != nullptr) ||
		(irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection && rot_pars.proj_mode == e_irt_rotation) ||
		(irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine && strchr(rot_pars.affine_mode, 'R') != nullptr)) {
		if (-MAX_ROT_ANGLE <= irt_angle_tmp[e_irt_angle_rot] && irt_angle_tmp[e_irt_angle_rot] <= MAX_ROT_ANGLE) {
			irt_desc.read_vflip = 0;
			irt_desc.read_hflip = 0;
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
		} else if ((180.0 - MAX_ROT_ANGLE) <= irt_angle_tmp[e_irt_angle_rot] && irt_angle_tmp[e_irt_angle_rot] <= (180.0 + MAX_ROT_ANGLE)) {
			irt_desc.read_hflip = 1;
			irt_desc.read_vflip = 1;
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot] - 180.0;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Converting rotation angle to %f with pre-rotation of 180 degree\n", rot_pars.irt_angles_adj[e_irt_angle_rot]);
		} else if (irt_angle_tmp[e_irt_angle_rot] == 90 || irt_angle_tmp[e_irt_angle_rot] == -90.0) {
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
			irt_desc.rot90 = 1;
			if ((irt_desc.image_par[OIMAGE].Xc & 1) ^ (irt_desc.image_par[IIMAGE].Yc & 1)) { //interpolation over input lines is required
				irt_desc.rot90_intv = 1;
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Rotation 90 degree will require input lines interpolation\n");
			}
			if ((irt_desc.image_par[OIMAGE].Yc & 1) ^ (irt_desc.image_par[IIMAGE].Xc & 1)) { //interpolation input pixels is required
				irt_desc.rot90_inth = 1;
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Rotation 90 degree will require input pixels interpolation\n");
			}
		} else {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
			IRT_TRACE_TO_RES_UTILS(test_res, " was not run, rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
			IRT_CLOSE_FAILED_TEST(0);
		}
	} else if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection)  ||
		       (irt_desc.irt_mode == e_irt_mesh       && rot_pars.mesh_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection)) {

		irt_desc.read_vflip = 0;
		irt_desc.read_hflip = 0;

		for (uint8_t angle_type = e_irt_angle_roll; angle_type <= e_irt_angle_yaw; angle_type++) {
			if (-MAX_ROT_ANGLE <= irt_angle_tmp[angle_type] && irt_angle_tmp[angle_type] <= MAX_ROT_ANGLE) {
				rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type];
			} else if ((180.0 - MAX_ROT_ANGLE) <= irt_angle_tmp[angle_type] && irt_angle_tmp[angle_type] <= (180.0 + MAX_ROT_ANGLE)) {
				switch (angle_type) {
				case e_irt_angle_roll:  irt_desc.read_hflip = !irt_desc.read_hflip;
										irt_desc.read_vflip = !irt_desc.read_vflip; break;
				case e_irt_angle_pitch: irt_desc.read_vflip = !irt_desc.read_vflip; break;
				case e_irt_angle_yaw:   irt_desc.read_hflip = !irt_desc.read_hflip; break;
				}
				rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type] - 180.0;
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Converting %s angle to %f with %s flip\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles_adj[angle_type], irt_flip_mode_s[angle_type]);
			} else if ((irt_angle_tmp[angle_type] == 90 || irt_angle_tmp[angle_type] == -90.0) && angle_type == e_irt_angle_roll) {
				rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type];
				//irt_desc.rot90 = 1;
				if ((irt_desc.image_par[OIMAGE].Xc & 1) ^ (irt_desc.image_par[IIMAGE].Yc & 1)) { //interpolation over input lines is required
					irt_desc.rot90_intv = 1;
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Roll rotation 90 degree will require input lines interpolation\n");
				}
				if ((irt_desc.image_par[OIMAGE].Yc & 1) ^ (irt_desc.image_par[IIMAGE].Xc & 1)) { //interpolation input pixels is required
					irt_desc.rot90_inth = 1;
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Roll rotation 90 degree will require input pixels interpolation\n");
				}
			} else {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "%s angle %f is not supported\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
				IRT_TRACE_TO_RES_UTILS(test_res, " was not run, %s angle %f is not supported\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
				IRT_CLOSE_FAILED_TEST(0);
			}

		}
	}
	//rot_pars.irt_angles[e_irt_angle_rot] = rot_pars.irt_angles_adj[e_irt_angle_rot];
	irt_desc.rot_dir = (rot_pars.irt_angles_adj[e_irt_angle_rot] >= 0) ? IRT_ROT_DIR_POS : IRT_ROT_DIR_NEG;
}

void irt_aff_coefs_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	rot_pars.affine_Si_factor = fabs((fabs(rot_pars.M11d * rot_pars.M22d) - rot_pars.M12d * rot_pars.M21d) / rot_pars.M22d);

	if (rot_pars.M22d < 0) { //resulted in Yi decrease when Yo increase (embedded rotation is [90:270], vertical flip)
		irt_desc.read_vflip = 1;
		rot_pars.M22d = -rot_pars.M22d;
	}

	if (rot_pars.M11d < 0) { //resulted in Xi decrease when Xo increase (embedded horizontal flip)
		irt_desc.read_hflip = 1;
		rot_pars.M11d = -rot_pars.M11d;
	}

	if (rot_pars.M12d >= 0) { //if (rot_pars.M21d <= 0) { //resulted in Yi decrease when Xo increase (embedded clockwise rotation)
		irt_desc.rot_dir = IRT_ROT_DIR_POS;
	} else {
		irt_desc.rot_dir = IRT_ROT_DIR_NEG;
	}
}

void irt_affine_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	double cosd = cos(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180);
	double sind = sin(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180);
	if (rot_pars.irt_angles[e_irt_angle_rot] != rot_pars.irt_angles_adj[e_irt_angle_rot]) //positive rotation definition (clockwise) is opposite to angle definition
		sind = -sind;
	double cotx = rot_pars.shear_mode & 1 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_x] * M_PI / 180.0) : 0;
	double coty = rot_pars.shear_mode & 2 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_y] * M_PI / 180.0) : 0;
	double refx = rot_pars.reflection_mode & 1 ? -1 : 1;
	double refy = rot_pars.reflection_mode & 2 ? -1 : 1;
	if (rot_pars.irt_angles[e_irt_angle_rot] == 90 || rot_pars.irt_angles[e_irt_angle_rot] == -90)
		cosd = 0;

	double aff_basic_matrix[5][2][2] = {
		{ { cosd, sind }, { -sind, cosd } }, //rotation
		{ { 1.0 / rot_pars.Sx, 0.0 }, { 0.0, 1.0 / rot_pars.Sy} }, //scaling
		{ { refx, 0.0}, { 0.0, refy} }, //reflection
		{ { 1.0, cotx}, { coty, 1.0 } }, //shearing
		{ { 1.0, 0.0 }, { 0.0, 1.0 } }, //unit
	};
	uint8_t aff_func, acc_idx = 1;

	double acc_matrix[2][2][2] = {
		{ { 1.0, 0.0 }, { 0.0, 1.0 } }, //unit
		{ { 0.0, 0.0 }, { 0.0, 0.0 } } //accumulated
	};

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "---------------------------------------------------------------------\n");
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Generating Affine coefficients for %s mode with:\n", desc, rot_pars.affine_mode);

	for (uint8_t i = 0; i < strlen(rot_pars.affine_mode); i++) {
		switch (rot_pars.affine_mode[i]) {
		case 'R': //rotation
			aff_func = 0;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %3.2f rotation angle and rotation matrix:\t\t", rot_pars.irt_angles[e_irt_angle_rot]);
			break;
		case 'S': //scaling
			aff_func = 1;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Sx/Sy %3.2f/%3.2f scaling factors and scaling matrix:\t", rot_pars.Sx, rot_pars.Sy);
			break;
		case 'M': //reflection
			aff_func = 2;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %s reflection and reflection matrix:\t\t", irt_refl_mode_s[rot_pars.reflection_mode]);
			break;
		case 'T': //shearing
			aff_func = 3;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %s shearing and shearing matrix:\t\t", irt_shr_mode_s[rot_pars.shear_mode]);
			break;
		default:  aff_func = 4; break;//non
		}
		if (aff_func != 4) {
			rot_pars.affine_flags[aff_func] = 1;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "[%5.2f %5.2f]\n", aff_basic_matrix[aff_func][0][0], aff_basic_matrix[aff_func][0][1]);
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\t\t\t\t\t\t\t[%5.2f %5.2f]\n", aff_basic_matrix[aff_func][1][0], aff_basic_matrix[aff_func][1][1]);
		}
		for (uint8_t r = 0; r < 2; r++) {
			for (uint8_t c = 0; c < 2; c++) {
				acc_matrix[acc_idx & 1][r][c] = 0;
				for (uint8_t idx = 0; idx < 2; idx++) {
					//M1[r][c] += M[r][idx] * shearing_matrix[idx][c];
					acc_matrix[acc_idx & 1][r][c] += acc_matrix[(acc_idx + 1) & 1][r][idx] * aff_basic_matrix[aff_func][idx][c];
				}
			}
		}
		acc_idx = (acc_idx + 1) & 1;
	}

	rot_pars.M11d = acc_matrix[(acc_idx + 1) % 2][0][0];
	rot_pars.M12d = acc_matrix[(acc_idx + 1) % 2][0][1];
	rot_pars.M21d = acc_matrix[(acc_idx + 1) % 2][1][0];
	rot_pars.M22d = acc_matrix[(acc_idx + 1) % 2][1][1];

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Affine matrix:\t\t\t\t\t[%5.2f %5.2f]\n", rot_pars.M11d, rot_pars.M12d);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "*\t\t\t\t\t\t\t[%5.2f %5.2f]\n", rot_pars.M21d, rot_pars.M22d);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "---------------------------------------------------------------------\n");

	if (irt_desc.irt_mode == e_irt_affine ||
	   (irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine) ||
	   (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_affine) ||
	   (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine))
		irt_aff_coefs_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
}

void irt_projection_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	//double Xoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Xc;
	//double Yoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Yc;
	double Xci = (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yci = (double)irt_desc.image_par[IIMAGE].Yc / 2;
	double tanx = rot_pars.shear_mode & 1 ? -1.0 / tan((90.0 - rot_pars.irt_angles[e_irt_angle_shr_x]) * M_PI / 180.0) : 0;
	double tany = rot_pars.shear_mode & 2 ? -1.0 / tan((90.0 - rot_pars.irt_angles[e_irt_angle_shr_y]) * M_PI / 180.0) : 0;
	//reflection in projection mode done by rotation around relative axis

	double proj_sin[e_irt_angle_yaw + 1], proj_cos[e_irt_angle_yaw + 1];//, proj_tan[3];
	for (uint8_t i = e_irt_angle_roll; i <= e_irt_angle_yaw; i++) {
		proj_sin[i] = sin(rot_pars.irt_angles_adj[i] * M_PI / 180.0);
		proj_cos[i] = cos(rot_pars.irt_angles_adj[i] * M_PI / 180.0);
		if (rot_pars.irt_angles_adj[i] == 90 || rot_pars.irt_angles_adj[i] == -90)
			proj_cos[i] = 0;//fix C++ bug that cos(90) is not zero
		//proj_tan[i] = tan(rot_pars.proj_angle[i] * M_PI / 180.0);
	}

	double Wd = rot_pars.proj_Wd;
	double Zd = rot_pars.proj_Zd;
	//double Sx = rot_pars.Sx;
	//double Sy = rot_pars.Sy;
	double proj_R[e_irt_angle_shear + 2][3][3];

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Generating Projection coefficients for %s mode with: ", desc, irt_proj_mode_s[rot_pars.proj_mode]);

	switch (rot_pars.proj_mode) {
	case e_irt_rotation: //rotation
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
		rot_pars.prj_Ad[0] = rot_pars.cosd;
		rot_pars.prj_Ad[1] = rot_pars.sind;
		rot_pars.prj_Ad[2] = /*-rot_pars.proj_A[0] * Xoc - rot_pars.proj_A[1] * Yoc*/ + Xci;
		rot_pars.prj_Cd[0] = 0;
		rot_pars.prj_Cd[1] = 0;
		rot_pars.prj_Cd[2] = 1;
		rot_pars.prj_Bd[0] = -rot_pars.sind;
		rot_pars.prj_Bd[1] = rot_pars.cosd;
		rot_pars.prj_Bd[2] = /*-rot_pars.proj_B[0] * Xoc - rot_pars.proj_B[1] * Yoc*/ + Yci;
		rot_pars.prj_Dd[0] = 0;
		rot_pars.prj_Dd[1] = 0;
		rot_pars.prj_Dd[2] = 1;
		break;
	case e_irt_affine: //affine
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s affine mode with: \n", rot_pars.affine_mode);
		if (rot_pars.affine_flags[e_irt_aff_rotation])	 IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
		if (rot_pars.affine_flags[e_irt_aff_scaling])	 IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %3.2f/%3.2f scaling factors\n", rot_pars.Sx, rot_pars.Sy);
		if (rot_pars.affine_flags[e_irt_aff_reflection]) IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %s reflection mode\n", irt_refl_mode_s[rot_pars.reflection_mode]);
		if (rot_pars.affine_flags[e_irt_aff_shearing])	 IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* %s shearing mode\n", irt_shr_mode_s[rot_pars.shear_mode]);
		rot_pars.prj_Ad[0] = rot_pars.M11d;
		rot_pars.prj_Ad[1] = rot_pars.M12d;
		rot_pars.prj_Ad[2] = /*-rot_pars.proj_A[0] * Xoc - rot_pars.proj_A[1] * Yoc +*/ Xci;
		rot_pars.prj_Cd[0] = 0;
		rot_pars.prj_Cd[1] = 0;
		rot_pars.prj_Cd[2] = 1;
		rot_pars.prj_Bd[0] = rot_pars.M21d;
		rot_pars.prj_Bd[1] = rot_pars.M22d;
		rot_pars.prj_Bd[2] = /*-rot_pars.proj_B[0] * Xoc - rot_pars.proj_B[1] * Yoc + */Yci;
		rot_pars.prj_Dd[0] = 0;
		rot_pars.prj_Dd[1] = 0;
		rot_pars.prj_Dd[2] = 1;
		break;
	case e_irt_projection:
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\n* %s order, roll %3.2f, pitch %3.2f, yaw %3.2f, Sx %3.2f, Sy %3.2f, Zd %3.2f, Wd %3.2f\n* %s shear mode, X shear %3.2f, Y shear %3.2f\n",
			rot_pars.proj_order, rot_pars.irt_angles[e_irt_angle_roll], rot_pars.irt_angles[e_irt_angle_pitch], rot_pars.irt_angles[e_irt_angle_yaw], rot_pars.Sx, rot_pars.Sy, rot_pars.proj_Zd, rot_pars.proj_Wd,
			irt_shr_mode_s[rot_pars.shear_mode],rot_pars.irt_angles[e_irt_angle_shr_x], rot_pars.irt_angles[e_irt_angle_shr_y]);
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* adjusted: roll %3.2f, pitch %3.2f, yaw %3.2f\n", rot_pars.irt_angles_adj[e_irt_angle_roll], rot_pars.irt_angles_adj[e_irt_angle_pitch], rot_pars.irt_angles_adj[e_irt_angle_yaw]);

		//defining rotation matrix for each rotation axis
		proj_R[e_irt_angle_pitch][0][0] = 1.0; proj_R[e_irt_angle_pitch][0][1] = 0.0;						  proj_R[e_irt_angle_pitch][0][2] = 0.0;
		proj_R[e_irt_angle_pitch][1][0] = 0.0; proj_R[e_irt_angle_pitch][1][1] = proj_cos[e_irt_angle_pitch]; proj_R[e_irt_angle_pitch][1][2] = -proj_sin[e_irt_angle_pitch];
		proj_R[e_irt_angle_pitch][2][0] = 0.0; proj_R[e_irt_angle_pitch][2][1] = proj_sin[e_irt_angle_pitch]; proj_R[e_irt_angle_pitch][2][2] = proj_cos[e_irt_angle_pitch];

		proj_R[e_irt_angle_yaw][0][0] =  proj_cos[e_irt_angle_yaw]; proj_R[e_irt_angle_yaw][0][1] = 0.0; proj_R[e_irt_angle_yaw][0][2] = proj_sin[e_irt_angle_yaw];
		proj_R[e_irt_angle_yaw][1][0] =  0.0;						proj_R[e_irt_angle_yaw][1][1] = 1.0; proj_R[e_irt_angle_yaw][1][2] = 0.0;
		proj_R[e_irt_angle_yaw][2][0] = -proj_sin[e_irt_angle_yaw]; proj_R[e_irt_angle_yaw][2][1] = 0.0; proj_R[e_irt_angle_yaw][2][2] = proj_cos[e_irt_angle_yaw];

		proj_R[e_irt_angle_roll][0][0] = proj_cos[e_irt_angle_roll]; proj_R[e_irt_angle_roll][0][1] = -proj_sin[e_irt_angle_roll]; proj_R[e_irt_angle_roll][0][2] = 0.0;
		proj_R[e_irt_angle_roll][1][0] = proj_sin[e_irt_angle_roll]; proj_R[e_irt_angle_roll][1][1] =  proj_cos[e_irt_angle_roll]; proj_R[e_irt_angle_roll][1][2] = 0.0;
		proj_R[e_irt_angle_roll][2][0] = 0.0;						proj_R[e_irt_angle_roll][2][1] =  0.0;						   proj_R[e_irt_angle_roll][2][2] = 1.0;

		proj_R[e_irt_angle_shear][0][0] = 1.0;  proj_R[e_irt_angle_shear][0][1] = tanx; proj_R[e_irt_angle_shear][0][2] = 0.0;
		proj_R[e_irt_angle_shear][1][0] = tany; proj_R[e_irt_angle_shear][1][1] = 1.0;  proj_R[e_irt_angle_shear][1][2] = 0.0;
		proj_R[e_irt_angle_shear][2][0] = 0.0;  proj_R[e_irt_angle_shear][2][1] = 0.0;	proj_R[e_irt_angle_shear][2][2] = 1.0;

		//reflection in projection mode done by rotation around relative axis
		proj_R[4][0][0] = 1.0;  proj_R[4][0][1] = 0.0;  proj_R[4][0][2] = 0.0;
		proj_R[4][1][0] = 0.0;  proj_R[4][1][1] = 1.0;  proj_R[4][1][2] = 0.0;
		proj_R[4][2][0] = 0.0;  proj_R[4][2][1] = 0.0;	proj_R[4][2][2] = 1.0;

		rot_pars.proj_T[0] = -rot_pars.proj_R[0][2] * rot_pars.proj_Wd;
		rot_pars.proj_T[1] = -rot_pars.proj_R[1][2] * rot_pars.proj_Wd;
		rot_pars.proj_T[2] = 0;

		//calculating total rotation matrix
		uint8_t aff_func = 0, acc_idx = 1;

		double acc_matrix[2][3][3] = {
			{ { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } }, //unit
			{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } //accumulated
		};
		for (uint8_t i = 0; i < strlen(rot_pars.proj_order); i++) {
			switch (rot_pars.proj_order[i]) {
			case 'Y': aff_func = e_irt_angle_yaw; break;
			case 'R': aff_func = e_irt_angle_roll; break;
			case 'P': aff_func = e_irt_angle_pitch; break;
			case 'S': aff_func = e_irt_angle_shear; break;
			default: aff_func = 4; break;
			}
			for (uint8_t r = 0; r < 3; r++) {
				for (uint8_t c = 0; c < 3; c++) {
					acc_matrix[acc_idx & 1][r][c] = 0;
					for (uint8_t idx = 0; idx < 3; idx++) {
						//M1[r][c] += M[r][idx] * shearing_matrix[idx][c];
						acc_matrix[acc_idx & 1][r][c] += acc_matrix[(acc_idx + 1) & 1][r][idx] * proj_R[aff_func][idx][c];
					}
				}
			}
			acc_idx = (acc_idx + 1) & 1;
		}


		for (uint8_t r = 0; r < 3; r++) {
			for (uint8_t c = 0; c < 3; c++) {
				rot_pars.proj_R[r][c] = acc_matrix[(acc_idx + 1) % 2][r][c];
			}
		}

		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
		for (uint8_t matrix = e_irt_angle_roll; matrix <= e_irt_angle_shear; matrix++) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "|   %s matrix     ", irt_prj_matrix_s[matrix]);
		}
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "|  Rotation matrix  |\n");

		for (uint8_t row = 0; row < 3; row++) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "|");
			for (uint8_t matrix = e_irt_angle_roll; matrix <= e_irt_angle_shear; matrix++) {
				for (uint8_t col = 0; col < 3; col++) {
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%5.2f ", proj_R[matrix][row][col]);
				}
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, " | ");
			}
			for (uint8_t col = 0; col < 3; col++) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%5.2f ", rot_pars.proj_R[row][col]);
			}
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "|\n");
		}
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");

		//A0 = Zd((R21*Xic+R22*Yic)*R32-(R31*Xic+R32*Yic-R33*Wd)*R22), A1 = Zd((R31*Xic+R32*Yic)*R12-(R11*Xic+R12*Yic)*R32), A2 = Zd^2((R11*Xic+R12*Yic)*R22-(R21*Xic+R22*Yic)*R12)
		//B0 = Zd((R31*Xic+R32*Yic)*R21-(R21*Xic+R22*Yic-R33*Wd)*R31), B1 = Zd((R11*Xic+R12*Yic)*R31-(R31*Xic+R32*Yic)*R11), B2 = Zd^2((R21*Xic+R22*Yic)*R11-(R11*Xic+R12*Yic)*R21)
		//C0 = Zd(R21*R32-R22*R31),									   C1 = Zd(R31*R12-R32*R11),							 C2 = Zd^2(R11*R22-R12*R21)
		//applying scaling
		double R11 = rot_pars.proj_R[0][0] * rot_pars.Sx;
		double R12 = rot_pars.proj_R[0][1] * rot_pars.Sy;
		//double R13 = rot_pars.proj_R[0][2];
		double R21 = rot_pars.proj_R[1][0] * rot_pars.Sx;
		double R22 = rot_pars.proj_R[1][1] * rot_pars.Sy;
		//double R23 = rot_pars.proj_R[1][2];
		double R31 = rot_pars.proj_R[2][0] * rot_pars.Sx;
		double R32 = rot_pars.proj_R[2][1] * rot_pars.Sy;
		double R33 = rot_pars.proj_R[2][2];

		//calculating projection coefficients
		rot_pars.prj_Ad[0] = ((R21 * Xci + R22 * Yci) * R32 - (R31 * Xci + R32 * Yci - R33 * Wd) * R22) / rot_pars.Sy;
		rot_pars.prj_Ad[1] = ((R31 * Xci + R32 * Yci - R33 * Wd) * R12 - (R11 * Xci + R12 * Yci) * R32) / rot_pars.Sy;
		rot_pars.prj_Ad[2] = Zd * ((R11 * Xci + R12 * Yci) * R22 - (R21 * Xci + R22 * Yci) * R12) / rot_pars.Sy;

		rot_pars.prj_Bd[0] = ((R31 * Xci + R32 * Yci - R33 * Wd) * R21 - (R21 * Xci + R22 * Yci) * R31) / rot_pars.Sx;
		rot_pars.prj_Bd[1] = ((R11 * Xci + R12 * Yci) * R31 - (R31 * Xci + R32 * Yci - R33 * Wd) * R11) / rot_pars.Sx;
		rot_pars.prj_Bd[2] = Zd * ((R21 * Xci + R22 * Yci) * R11 - (R11 * Xci + R12 * Yci) * R21) / rot_pars.Sx;

		rot_pars.prj_Cd[0] = (R32 * R21 - R31 * R22) / rot_pars.Sy;
		rot_pars.prj_Cd[1] = (R31 * R12 - R32 * R11) / rot_pars.Sy;
		rot_pars.prj_Cd[2] = Zd * (R11 * R22 - R12 * R21) / rot_pars.Sy;

		rot_pars.prj_Dd[0] = (R32 * R21 - R31 * R22) / rot_pars.Sx;
		rot_pars.prj_Dd[1] = (R31 * R12 - R32 * R11) / rot_pars.Sx;
		rot_pars.prj_Dd[2] = Zd * (R11 * R22 - R12 * R21) / rot_pars.Sx;

#ifdef IRT_PROJ_COEFS_NORMALIZATION
		rot_pars.prj_Ad[0] /= rot_pars.prj_Cd[2];
		rot_pars.prj_Ad[1] /= rot_pars.prj_Cd[2];
		rot_pars.prj_Ad[2] /= rot_pars.prj_Cd[2];
		rot_pars.prj_Bd[0] /= rot_pars.prj_Dd[2];
		rot_pars.prj_Bd[1] /= rot_pars.prj_Dd[2];
		rot_pars.prj_Bd[2] /= rot_pars.prj_Dd[2];
		rot_pars.prj_Cd[0] /= rot_pars.prj_Cd[2];
		rot_pars.prj_Cd[1] /= rot_pars.prj_Cd[2];
		rot_pars.prj_Cd[2] = 1;
		rot_pars.prj_Dd[0] /= rot_pars.prj_Dd[2];
		rot_pars.prj_Dd[1] /= rot_pars.prj_Dd[2];
		rot_pars.prj_Dd[2] = 1;
#endif
		break;
	}

}

void irt_mesh_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, bool trace) {

	irt_desc.image_par[MIMAGE].W = irt_desc.image_par[OIMAGE].W;
	irt_desc.image_par[MIMAGE].S = irt_desc.image_par[OIMAGE].S;
	irt_desc.image_par[MIMAGE].H = irt_desc.image_par[OIMAGE].H;

	if (rot_pars.mesh_Sh != 1.0) {
		irt_desc.mesh_sparse_h = 1;
		irt_desc.mesh_Gh = (uint32_t)rint(pow(2.0, IRT_MESH_G_PREC) / rot_pars.mesh_Sh);
		irt_desc.image_par[MIMAGE].W = (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].W - 1) / rot_pars.mesh_Sh) + 1;
		irt_desc.image_par[MIMAGE].S = (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].S - 1) / rot_pars.mesh_Sh) + 1;
	}
	if (rot_pars.mesh_Sv != 1.0) {
		irt_desc.mesh_sparse_v = 1;
		irt_desc.mesh_Gv = (uint32_t)rint(pow(2.0, IRT_MESH_G_PREC) / rot_pars.mesh_Sv);
		irt_desc.image_par[MIMAGE].H = (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].H - 1) / rot_pars.mesh_Sv) + 1;
	}


	if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) { //stripe height is multiple of 2
		irt_desc.image_par[MIMAGE].H = (uint16_t)(ceil((double)irt_desc.image_par[MIMAGE].H / IRT_MM_DYN_MODE_LINE_RELEASE) * IRT_MM_DYN_MODE_LINE_RELEASE);
		if (trace)
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Mesh matrix calculation: adjusting Hm to %d be multiple of 2 because of mesh buffer dynamic format\n", irt_desc.image_par[MIMAGE].H);
	}

	if (trace) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Generating mesh matrix for %s mode with ", desc, irt_mesh_mode_s[rot_pars.mesh_mode]);
		switch (rot_pars.mesh_mode) {
		case e_irt_rotation: IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]); break;
		case e_irt_affine:
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s affine mode with\n", rot_pars.affine_mode);
			if (rot_pars.affine_flags[e_irt_aff_rotation])	  IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
			if (rot_pars.affine_flags[e_irt_aff_scaling])	  IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%3.2f/%3.2f scaling factors\n", rot_pars.Sx, rot_pars.Sy);
			if (rot_pars.affine_flags[e_irt_aff_reflection])  IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s reflection mode\n", irt_refl_mode_s[rot_pars.reflection_mode]);
			if (rot_pars.affine_flags[e_irt_aff_shearing])	  IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s shearing mode\n", irt_shr_mode_s[rot_pars.shear_mode]);
			break;
		case e_irt_projection:
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\n* %s order, roll %3.2f, pitch %3.2f, yaw %3.2f, Sx %3.2f, Sy %3.2f, Zd %3.2f, Wd %3.2f\n* %s shear mode, X shear %3.2f, Y shear %3.2f\n",
				rot_pars.proj_order, rot_pars.irt_angles[e_irt_angle_roll], rot_pars.irt_angles[e_irt_angle_pitch], rot_pars.irt_angles[e_irt_angle_yaw], rot_pars.Sx, rot_pars.Sy, rot_pars.proj_Zd, rot_pars.proj_Wd,
				irt_shr_mode_s[rot_pars.shear_mode], rot_pars.irt_angles[e_irt_angle_shr_x], rot_pars.irt_angles[e_irt_angle_shr_y]);
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "\n* adjusted: roll %3.2f, pitch %3.2f, yaw %3.2f\n", rot_pars.irt_angles_adj[e_irt_angle_roll], rot_pars.irt_angles_adj[e_irt_angle_pitch], rot_pars.irt_angles_adj[e_irt_angle_yaw]);
			break;
		case e_irt_mesh: IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "%s order %3.2f distortion\n", irt_mesh_order_s[rot_pars.mesh_order], rot_pars.dist_r); break;
		}
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "--------------------------------------------------------------------------------------------------------\n");
	}

	irt_mesh_full_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
	irt_mesh_sparse_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
	irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
}

void irt_mesh_full_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	FILE *fptr_full = nullptr, *fptr_full2 = nullptr;
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	char fptr_full_name[50], fptr_rel_name[50];
	sprintf(fptr_full_name, "Mesh_matrix_full_%d.txt", desc);
	sprintf(fptr_rel_name, "Mesh_matrix_rel_%d.txt", desc);
	FILE* fptr_rel = nullptr;
	if (print_out_files) {
		fptr_full = fopen(fptr_full_name, "w");
		fptr_rel  = fopen(fptr_rel_name, "w");
		IRT_TRACE_TO_RES_UTILS(fptr_full, "[row, col] = [Xi, Yi]\n");
		IRT_TRACE_TO_RES_UTILS(fptr_rel, "[row, col] = [Xi, Yi]\n");
	}
#endif

	//memset(irt_cfg.mesh_images.mesh_image_full, 0, sizeof(irt_cfg.mesh_images.mesh_image_full));

	uint16_t Ho = irt_desc.image_par[OIMAGE].H;
	uint16_t So = irt_desc.image_par[OIMAGE].S;
	double Xoc = (double)irt_desc.image_par[OIMAGE].Xc / 2;
	double Yoc = (double)irt_desc.image_par[OIMAGE].Yc / 2;
	//uint16_t Hi = irt_desc.image_par[IIMAGE].H;
	//uint16_t Wi = irt_desc.image_par[IIMAGE].W;
	double Xic = (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yic = (double)irt_desc.image_par[IIMAGE].Yc / 2;

	double xo, yo, r, xi1, yi1, xi2, yi2, dist_ratio;

	double Ymax_line;

#if 0
	double M11, M12, M21, M22, COS, SIN;
	if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
		M11 = (double)irt_desc.M11f;
		M12 = (double)irt_desc.M12f;
		M21 = (double)irt_desc.M21f;
		M22 = (double)irt_desc.M22f;
		COS = (double)irt_desc.cosf;
		SIN = (double)irt_desc.sinf;
	}
	else {
		M11 = (double)irt_desc.M11i / pow(2.0, rot_pars.TOTAL_PREC);
		M12 = (double)irt_desc.M12i / pow(2.0, rot_pars.TOTAL_PREC);
		M21 = (double)irt_desc.M21i / pow(2.0, rot_pars.TOTAL_PREC);
		M22 = (double)irt_desc.M22i / pow(2.0, rot_pars.TOTAL_PREC);
		COS = (double)irt_desc.cosi / pow(2.0, rot_pars.TOTAL_PREC);
		SIN = (double)irt_desc.sini / pow(2.0, rot_pars.TOTAL_PREC);
	}
#endif

	for (uint16_t row = 0; row < Ho; row++) {
		for (uint16_t col = 0; col < So; col++) {
			xo = (double)col - Xoc;
			yo = (double)row - Yoc;
			r = sqrt(pow(rot_pars.dist_x * xo, 2) + pow(rot_pars.dist_y * yo, 2));
			dist_ratio = 1 + rot_pars.dist_r * r / sqrt(pow(So / 2, 2) + pow(Ho / 2, 2));

			if (rot_pars.mesh_order == 0) {//predistortion
				xi1 = xo * dist_ratio;// *rot_pars.dist_x;
				yi1 = yo * dist_ratio;// *rot_pars.dist_y;
			}
			else {
				xi1 = xo;
				yi1 = yo;
			}
			switch (rot_pars.mesh_mode) {
			case e_irt_rotation:
				xi2 =  cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * xi1 + sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * yi1;
				yi2 = -sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * xi1 + cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * yi1;
				break;
			case e_irt_affine:
				xi2 = rot_pars.M11d * xi1 + rot_pars.M12d * yi1;
				yi2 = rot_pars.M21d * xi1 + rot_pars.M22d * yi1;
				break;
			case e_irt_projection:
				xi2 = (rot_pars.prj_Ad[0] * xi1 + rot_pars.prj_Ad[1] * yi1 + rot_pars.prj_Ad[2]) /
					  (rot_pars.prj_Cd[0] * xi1 + rot_pars.prj_Cd[1] * yi1 + rot_pars.prj_Cd[2]) - Xic;
				yi2 = (rot_pars.prj_Bd[0] * xi1 + rot_pars.prj_Bd[1] * yi1 + rot_pars.prj_Bd[2]) /
					  (rot_pars.prj_Dd[0] * xi1 + rot_pars.prj_Dd[1] * yi1 + rot_pars.prj_Dd[2]) - Yic;
				break;
			default: //distortion
				xi2 = xi1;
				yi2 = yi1;
				break;
			}

			if (rot_pars.mesh_order == 1) {//post distortion
				irt_cfg.mesh_images.mesh_image_full[row][col].x = (float)(xi2 * dist_ratio/* rot_pars.dist_x*/ + Xic);
				irt_cfg.mesh_images.mesh_image_full[row][col].y = (float)(yi2 * dist_ratio/* rot_pars.dist_y*/ + Yic);
			} else {
				irt_cfg.mesh_images.mesh_image_full[row][col].x = (float)(xi2 + Xic);
				irt_cfg.mesh_images.mesh_image_full[row][col].y = (float)(yi2 + Yic);
			}

			//relative matrix
			irt_cfg.mesh_images.mesh_image_rel[row][col].x = (float)irt_cfg.mesh_images.mesh_image_full[row][col].x;
			irt_cfg.mesh_images.mesh_image_rel[row][col].y = (float)irt_cfg.mesh_images.mesh_image_full[row][col].y;

			if (irt_desc.mesh_rel_mode) {
				irt_cfg.mesh_images.mesh_image_rel[row][col].x -= (float)col;
				irt_cfg.mesh_images.mesh_image_rel[row][col].y -= (float)row;
			}

			//IRT_TRACE_UTILS("Mesh_image_full[%d, %d] = (%f, %f)\n", row, col, irt_cfg.mesh_images.mesh_image_full[row][col].x, irt_cfg.mesh_images.mesh_image_full[row][col].y);

			irt_map_image_pars_update < mesh_xy_fp32_meta, double > (irt_cfg, irt_desc, irt_cfg.mesh_images.mesh_image_full, row, col, Ymax_line, fptr_full, fptr_full2);
			if (print_out_files)
				IRT_TRACE_TO_RES_UTILS(fptr_rel, "[%d, %d] = [%4.2f, %4.2f]\n", row, col, (double)irt_cfg.mesh_images.mesh_image_rel[row][col].x, (double)irt_cfg.mesh_images.mesh_image_rel[row][col].y);

		}
	}

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	if (print_out_files) {
		fclose(fptr_full);
		fclose(fptr_rel);
	}
#endif
}

void irt_mesh_sparse_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	FILE *f2 = nullptr, *f3 = nullptr, *f4 = nullptr;
	char f2_name[50], f3_name[50], f4_name[50];
	sprintf(f2_name, "Mesh_matrix_fi16_%d.txt", desc);
	sprintf(f3_name, "Mesh_matrix_ui32_%d.txt", desc);
	sprintf(f4_name, "Mesh_matrix_fp32_%d.txt", desc);
	if (print_out_files) {
		f2 = fopen(f2_name, "w");
		f3 = fopen(f3_name, "w");
		f4 = fopen(f4_name, "w");
	}
#endif
	//generating sparse matrix & checking precision
	int32_t i0, j0, i1, j1;
	double xo, yo, xf, yf;
	int32_t xi32, yi32;

	uint16_t Ho = irt_desc.image_par[OIMAGE].H;
	uint16_t So = irt_desc.image_par[OIMAGE].S;

	for (int row = 0; row < irt_desc.image_par[MIMAGE].H; row++) {
		for (int col = 0; col < irt_desc.image_par[MIMAGE].S; col++) {

			yo = (double)row * rot_pars.mesh_Sv;
			xo = (double)col * rot_pars.mesh_Sh;

			i0 = (int32_t)floor(yo);
			i1 = i0 + 1;
			yf = yo - (double)i0;
			j0 = (int32_t)floor(xo);
			j1 = j0 + 1;
			xf = xo - (double)j0;

			i0 = std::max(0, std::min(i0, (int32_t)Ho - 1));
			i1 = std::max(0, std::min(i1, (int32_t)Ho - 1));
			j0 = std::max(0, std::min(j0, (int32_t)So - 1));
			j1 = std::max(0, std::min(j1, (int32_t)So - 1));

			irt_cfg.mesh_images.mesh_image_fp32[row][col].x =
				(float)(((double)irt_cfg.mesh_images.mesh_image_rel[i0][j0].x * (1.0 - yf) +
						 (double)irt_cfg.mesh_images.mesh_image_rel[i1][j0].x * yf) * (1.0 - xf) +
						((double)irt_cfg.mesh_images.mesh_image_rel[i0][j1].x * (1.0 - yf) +
						 (double)irt_cfg.mesh_images.mesh_image_rel[i1][j1].x * yf) * xf);

			irt_cfg.mesh_images.mesh_image_fp32[row][col].y =
				(float)(((double)irt_cfg.mesh_images.mesh_image_rel[i0][j0].y * (1.0 - yf) +
						 (double)irt_cfg.mesh_images.mesh_image_rel[i1][j0].y * yf) * (1.0 - xf) +
						((double)irt_cfg.mesh_images.mesh_image_rel[i0][j1].y * (1.0 - yf) +
						 (double)irt_cfg.mesh_images.mesh_image_rel[i1][j1].y * yf) * xf);

			xi32 = (int32_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x * pow(2.0, irt_desc.mesh_point_location));
			yi32 = (int32_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y * pow(2.0, irt_desc.mesh_point_location));

			float M_max = std::fmax(fabs(irt_cfg.mesh_images.mesh_image_fp32[row][col].x), fabs(irt_cfg.mesh_images.mesh_image_fp32[row][col].y));
			int M_max_int = (int)floor(M_max); //integer part of max value
			uint8_t M_I_width = (uint8_t)ceil(log2(M_max_int + 1));
			uint8_t M_F_width = 15 - M_I_width; //remained bits for fraction

			if (irt_desc.irt_mode == e_irt_mesh && irt_desc.mesh_format == e_irt_mesh_flex) { //relevant only in mesh_flex format and mesh mode
				if (M_F_width < irt_desc.mesh_point_location) { ////mesh_point_location > bits allowed for fraction presentation of M
					if (rot_pars.mesh_prec_auto_adj == 0) { //correction is not allowed
						if (xi32 >= (int32_t)pow(2, 15) || xi32 <= -(int32_t)pow(2, 15)) {
							IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: Mesh X value %f for pixel[%d][%d] cannot fit FI16 format with fixed point format S%d.%d as %d\n",
								(double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x, row, col, 15 - irt_desc.mesh_point_location, irt_desc.mesh_point_location, xi32);
							IRT_TRACE_TO_RES_UTILS(test_res, "Error: Mesh X value %f for pixel[%d][%d] cannot fit FI16 format with fixed point format S%d.%d as %d\n",
								(double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x, row, col, 15 - irt_desc.mesh_point_location, irt_desc.mesh_point_location, xi32);
						}
						if (yi32 >= (int32_t)pow(2, 15) || yi32 <= -(int32_t)pow(2, 15)) {
							IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: Mesh Y value %f for pixel[%d][%d] cannot fit FI16 format with fixed point format S%d.%d as %d\n",
								(double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y, row, col, 15 - irt_desc.mesh_point_location, irt_desc.mesh_point_location, yi32);
							IRT_TRACE_TO_RES_UTILS(test_res, "Error: Mesh Y value %f for pixel[%d][%d] cannot fit FI16 format with fixed point format S%d.%d as %d\n",
								(double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y, row, col, 15 - irt_desc.mesh_point_location, irt_desc.mesh_point_location, yi32);
						}
						IRT_CLOSE_FAILED_TEST(0);

					} else {
						IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Adjusting mesh_point_location to %d bits\n", desc, M_F_width);
						irt_desc.mesh_point_location = M_F_width;
					}
				}
			}
		}
	}

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
	uint64_t mimage_addr_start = irt_desc.image_par[MIMAGE].addr_start;
	uint64_t mimage_row_start, mimage_col_start, mimage_pxl_start;
#endif

	//generating fixed point matrix based on final precision
	for (int row = 0; row < irt_desc.image_par[MIMAGE].H; row++) {
		for (int col = 0; col < irt_desc.image_par[MIMAGE].S; col++) {

			irt_cfg.mesh_images.mesh_image_fi16[row][col].x = (int16_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x * pow(2.0, irt_desc.mesh_point_location));
			irt_cfg.mesh_images.mesh_image_fi16[row][col].y = (int16_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y * pow(2.0, irt_desc.mesh_point_location));

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
			mimage_row_start = (uint64_t)row * irt_desc.image_par[MIMAGE].S * (irt_desc.mesh_format ? 8 : 4);
			mimage_col_start = (uint64_t)col * (irt_desc.mesh_format ? 8 : 4);
			mimage_pxl_start = mimage_addr_start + mimage_row_start + mimage_col_start;
			for (uint8_t byte = 0; byte < (irt_desc.mesh_format ? 4 : 2); byte++) {
				if (irt_desc.mesh_format == 0)
					ext_mem[mimage_pxl_start + byte] = (irt_cfg.mesh_images.mesh_image_fi16[row][col].x >> (8 * byte)) & 0xff;
				else
					ext_mem[mimage_pxl_start + byte] = (IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].x) >> (8 * byte)) & 0xff;
			}
			mimage_pxl_start += (irt_desc.mesh_format ? 4 : 2);
			for (uint8_t byte = 0; byte < (irt_desc.mesh_format ? 4 : 2); byte++) {
				if (irt_desc.mesh_format == 0)
					ext_mem[mimage_pxl_start + byte] = (irt_cfg.mesh_images.mesh_image_fi16[row][col].y >> (8 * byte)) & 0xff;
				else
					ext_mem[mimage_pxl_start + byte] = (IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].y) >> (8 * byte)) & 0xff;
			}
			if (print_out_files) {
				IRT_TRACE_TO_RES_UTILS(f2, "%04x%04x\n", ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].y & 0xffff), ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].x & 0xffff));
				IRT_TRACE_TO_RES_UTILS(f3, "%08x%08x\n", IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].y), IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].x));
				IRT_TRACE_TO_RES_UTILS(f4, "[%d, %d] = [%4.2f, %4.2f]\n", row, col, (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x, (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y);
			}
#endif
		}
	}
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	if (print_out_files) {
		fclose(f2);
		fclose(f3);
		fclose(f4);
	}
#endif
}

void irt_mesh_interp_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	//return;
	FILE* fptr_vals = nullptr, * fptr_pars = nullptr;
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	char fptr_vals_name[50], fptr_pars_name[50];
	sprintf(fptr_vals_name, "Mesh_matrix_intr_%d.txt", desc);
	sprintf(fptr_pars_name, "Mesh_matrix_intr_pars_%d.txt", desc);
	if (print_out_files) {
		fptr_vals = fopen(fptr_vals_name, "w");
		fptr_pars = fopen(fptr_pars_name, "w");
		IRT_TRACE_TO_RES_UTILS(fptr_vals, "[row, col] = [Xi, Yi]\n");
	}
#endif
	double Ymax_line = 0;

	//interpolation of full matrix from sparse
	int64_t xm, ym;
	uint32_t xf, yf;
	float mp_value[2][4], mi[2], xw0, xw1, yw0, yw1;
	int16_t i0, j0, i1, j1;
	bool p_valid[4];
	double mesh_matrix_acc_err = 0, mesh_matrix_max_err = 0, mesh_matrix_max_err_left = 0, mesh_matrix_max_err_right = 0, mesh_matrix_error = 0;

	for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {
		for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].S; col++) {

			ym = (int64_t)row;
			if (irt_desc.mesh_sparse_v) { //sparse in v direction
				ym = ym * (int64_t)irt_desc.mesh_Gv;
			}
			else {
				ym = ym << IRT_MESH_G_PREC;
			}

			xm = (int64_t)col;
			if (irt_desc.mesh_sparse_h) {//sparse in h direction
				xm = xm * (int64_t)irt_desc.mesh_Gh;
			}
			else {
				xm = xm << IRT_MESH_G_PREC;
			}

			j0 = (int16_t)(xm >> IRT_MESH_G_PREC);
			j1 = j0 + 1;
			xf = (uint32_t)(xm - ((int64_t)j0 << IRT_MESH_G_PREC));

			i0 = (int16_t)(ym >> IRT_MESH_G_PREC);
			i1 = i0 + 1;
			yf = (uint32_t)(ym - ((int64_t)i0 << IRT_MESH_G_PREC));

			if (j1 >= irt_desc.image_par[MIMAGE].S) { //reset weight for oob
				xf = 0;
			}
			if (i1 >= irt_desc.image_par[MIMAGE].H) {//reset weight for oob
				yf = 0;
			}

			i1 = IRT_top::IRT_UTILS::irt_min_int16(i1, irt_desc.image_par[MIMAGE].H - 1);
			j1 = IRT_top::IRT_UTILS::irt_min_int16(j1, irt_desc.image_par[MIMAGE].S - 1);

			p_valid[0] = i0 >= 0 && i0 < irt_desc.image_par[MIMAGE].H && j0 >= 0 && j0 < irt_desc.image_par[MIMAGE].S;
			p_valid[1] = i0 >= 0 && i0 < irt_desc.image_par[MIMAGE].H && j1 >= 0 && j1 < irt_desc.image_par[MIMAGE].S;
			p_valid[2] = i1 >= 0 && i1 < irt_desc.image_par[MIMAGE].H && j0 >= 0 && j0 < irt_desc.image_par[MIMAGE].S;
			p_valid[3] = i1 >= 0 && i1 < irt_desc.image_par[MIMAGE].H && j1 >= 0 && j1 < irt_desc.image_par[MIMAGE].S;

			////selecting X for interpolation assign pixel
			mp_value[0][0] = (float)(p_valid[0] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i0][j0].x : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j0].x / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[0][1] = (float)(p_valid[1] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i0][j1].x : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j1].x / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[0][2] = (float)(p_valid[2] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i1][j0].x : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j0].x / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[0][3] = (float)(p_valid[3] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i1][j1].x : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j1].x / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);

			////selecting Y for interpolation assign pixel
			mp_value[1][0] = (float)(p_valid[0] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i0][j0].y : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j0].y / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[1][1] = (float)(p_valid[1] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i0][j1].y : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j1].y / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[1][2] = (float)(p_valid[2] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i1][j0].y : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j0].y / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);
			mp_value[1][3] = (float)(p_valid[3] ? (irt_desc.mesh_format ? irt_cfg.mesh_images.mesh_image_fp32[i1][j1].y : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j1].y / (float)pow(2.0, irt_desc.mesh_point_location)) : 0);

			xw0 = (float)((double)xf / pow(2.0, 31));
			xw1 = (float)((pow(2.0, 31) - (double)xf) / pow(2.0, 31));
			yw0 = (float)((double)yf / pow(2.0, 31));
			yw1 = (float)((pow(2.0, 31) - (double)yf) / pow(2.0, 31));

			for (uint8_t j = 0; j < 2; j++) {
				//vertical followed by horizontal
				//as in arch model
				mi[j] = (mp_value[j][0] * yw1 + mp_value[j][2] * yw0) * xw1 + (mp_value[j][1] * yw1 + mp_value[j][3] * yw0) * xw0;
				mi[j] = ((int64_t)rint(mi[j] * (float)pow(2, 31))) / ((float)pow(2, 31));
			}

			irt_cfg.mesh_images.mesh_image_intr[row][col].x = mi[0];  irt_cfg.mesh_images.mesh_image_intr[row][col].y = mi[1];
			if (irt_desc.mesh_rel_mode) {
				irt_cfg.mesh_images.mesh_image_intr[row][col].x += (double)col;
				irt_cfg.mesh_images.mesh_image_intr[row][col].y += (double)row;
			}

			mesh_matrix_error = abs(floor((double)irt_cfg.mesh_images.mesh_image_full[row][col].x) - floor((double)irt_cfg.mesh_images.mesh_image_intr[row][col].x));
			mesh_matrix_acc_err += mesh_matrix_error;
			mesh_matrix_max_err = fmax(mesh_matrix_max_err, mesh_matrix_error);
			if (col == 0)
				mesh_matrix_max_err_left = fmax(mesh_matrix_max_err_left, mesh_matrix_error);
			if (col == irt_desc.image_par[OIMAGE].S - 1)
				mesh_matrix_max_err_right = fmax(mesh_matrix_max_err_right, mesh_matrix_error);

			irt_map_image_pars_update < mesh_xy_fp64_meta, double > (irt_cfg, irt_desc, irt_cfg.mesh_images.mesh_image_intr, row, col, Ymax_line, fptr_vals, fptr_pars);

#if 0
			if (row > 0)
				if (((irt_cfg.mesh_images.mesh_image_intr[row][col].Ymin < irt_cfg.mesh_images.mesh_image_intr[row - 1][col].Ymin) && irt_desc.read_vflip == 0) ||
					((irt_cfg.mesh_images.mesh_image_intr[row][col].Ymin > irt_cfg.mesh_images.mesh_image_intr[row - 1][col].Ymin) && irt_desc.read_vflip == 1)) {
					IRT_TRACE_UTILS("Desc gen error: at mesh matrix interpolation, Ymin %f of line %d less than Ymin %f of line %d for output image width %d\n",
						irt_cfg.mesh_images.mesh_image_intr[row][col].Ymin, row, irt_cfg.mesh_images.mesh_image_intr[row - 1][col].Ymin, row - 1, col + 1);
					IRT_TRACE_TO_RES_UTILS(test_res, "was not run, at mesh matrix interpolation, Ymin %f of line %d less than Ymin %f of line %d for output image width %d\n",
						irt_cfg.mesh_images.mesh_image_intr[row][col].Ymin, row, irt_cfg.mesh_images.mesh_image_intr[row - 1][col].Ymin, row - 1, col + 1);
					IRT_CLOSE_FAILED_TEST(0);
				}
#endif

		}
	}

	//rot_pars.IBufH_req = irt_cfg.mesh_images.mesh_image_intr[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].W - 1].IBufH_req;
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Matrix accumulated error %f\n", mesh_matrix_acc_err);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Matrix maximum error %f\n", mesh_matrix_max_err);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Matrix average error %f\n", mesh_matrix_acc_err / ((double)irt_desc.image_par[OIMAGE].W * (double)irt_desc.image_par[OIMAGE].H));
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Matrix maximum error left %f\n", mesh_matrix_max_err_left);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Matrix maximum error right %f\n", mesh_matrix_max_err_right);
	const double mesh_matrix_acc_err_thr = 0.1; //0.71
	if (mesh_matrix_acc_err / ((double)irt_desc.image_par[OIMAGE].W * (double)irt_desc.image_par[OIMAGE].H) > mesh_matrix_acc_err_thr) { //0.71
		//making threshold 0 does not allow mesh as rotation to work in good resolution. Error is never zero in fixed mesh coord. Sometime output stripe even cannot be found
		rot_pars.mesh_matrix_error = 1;
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Interpolated mesh matrix error > %f, rotation and affine optimization will be disabled\n", mesh_matrix_acc_err_thr);
	}
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	if (print_out_files) {
		fclose(fptr_vals);
		fclose(fptr_pars);
	}
#endif
}


template < class mimage_type, class coord_type >
void irt_map_image_pars_update(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, mimage_type** mesh_image, uint16_t row, uint16_t col, coord_type& Ymax_line, FILE* fptr_vals, FILE* fptr_pars) {

	bool XiYi_inf = std::isinf(mesh_image[row][col].y) | std::isinf(mesh_image[row][col].x) | std::isnan(mesh_image[row][col].y) | std::isnan(mesh_image[row][col].x);

	if (col == 0) {
		mesh_image[row][col].Ymin = mesh_image[row][col].y;
		Ymax_line = mesh_image[row][col].y;
	} else {
		mesh_image[row][col].Ymin = std::fmin(mesh_image[row][col - 1].Ymin, mesh_image[row][col].y); //updating from left
		Ymax_line = std::fmax(Ymax_line, mesh_image[row][col].y);									  //updating from left
	}
	if (row > 0)
		mesh_image[row][col].Ymin_dir = mesh_image[row][col].Ymin >= mesh_image[row - 1][col].Ymin;
	else
		mesh_image[row][col].Ymin_dir = 1;

	if (col > 0)
		mesh_image[row][col].Ymin_dir_swap = mesh_image[row][col - 1].Ymin_dir_swap | (mesh_image[row][col].Ymin_dir != mesh_image[row][col - 1].Ymin_dir);
	else
		mesh_image[row][col].Ymin_dir_swap = 0;

	if (row == 0)
		mesh_image[row][col].IBufH_req = (uint32_t)(ceil((double)Ymax_line) - floor((double)mesh_image[row][col].Ymin));	//updating from top
	else
		mesh_image[row][col].IBufH_req = (uint32_t)(ceil((double)Ymax_line) - floor((double)mesh_image[row - 1][col].Ymin));//updating from top
	mesh_image[row][col].IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, mesh_image[row][col].IBufH_req);

	if (col == 0) {
		mesh_image[row][col].Xi_first = mesh_image[row][col].x;
		mesh_image[row][col].Xi_last  = mesh_image[row][col].x;
		mesh_image[row][col].Yi_first = mesh_image[row][col].y;
		mesh_image[row][col].Yi_last  = mesh_image[row][col].y;
		mesh_image[row][col].XiYi_inf = XiYi_inf;
	} else {
		mesh_image[row][col].Xi_first = std::fmin(mesh_image[row][col - 1].Xi_first, mesh_image[row][col].x);  //updating from left
		mesh_image[row][col].Xi_last  = std::fmax(mesh_image[row][col - 1].Xi_last,  mesh_image[row][col].x);  //updating from left
		mesh_image[row][col].Yi_first = std::fmin(mesh_image[row][col - 1].Yi_first, mesh_image[row][col].y);  //updating from left
		mesh_image[row][col].Yi_last  = std::fmax(mesh_image[row][col - 1].Yi_last,  mesh_image[row][col].y);  //updating from left
		mesh_image[row][col].XiYi_inf = mesh_image[row][col - 1].XiYi_inf | XiYi_inf;						   //updating from left
	}
	if (row > 0) {
		mesh_image[row][col].Xi_first = std::fmin(mesh_image[row][col].Xi_first, mesh_image[row - 1][col].Xi_first); //updating from top
		mesh_image[row][col].Xi_last  = std::fmax(mesh_image[row][col].Xi_last,  mesh_image[row - 1][col].Xi_last);	 //updating from top
		mesh_image[row][col].Yi_first = std::fmin(mesh_image[row][col].Yi_first, mesh_image[row - 1][col].Yi_first); //updating from top
		mesh_image[row][col].Yi_last  = std::fmax(mesh_image[row][col].Yi_last,  mesh_image[row - 1][col].Yi_last);	 //updating from top
		mesh_image[row][col].IBufH_req = (uint32_t)std::max((int32_t)mesh_image[row][col].IBufH_req, (int32_t)mesh_image[row - 1][col].IBufH_req);
		mesh_image[row][col].Ymin_dir_swap = mesh_image[row][col].Ymin_dir_swap | mesh_image[row - 1][col].Ymin_dir_swap | (mesh_image[row][col].Ymin_dir != mesh_image[row - 1][col].Ymin_dir);
		mesh_image[row][col].XiYi_inf = mesh_image[row - 1][col].XiYi_inf | XiYi_inf;
	}

	mesh_image[row][col].Si = (uint32_t)(ceil(mesh_image[row][col].Xi_last) - floor(mesh_image[row][col].Xi_first)) + 2; //updating from left
	mesh_image[row][col].IBufW_req = irt_IBufW_req_calc(irt_cfg, irt_desc, mesh_image[row][col].Si);

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	if (fptr_vals != nullptr)
		IRT_TRACE_TO_RES_UTILS(fptr_vals, "[%u, %u] = [%4.8f, %4.8f]\n", row, col, (double)mesh_image[row][col].x, (double)mesh_image[row][col].y);
	if (fptr_pars != nullptr)
		IRT_TRACE_TO_RES_UTILS(fptr_pars, "[%u, %u] = Xi_first %4.8f, Xi_last %4.8f, Si %u, Yi_first %f, Yi_last %f, BufW %u, BufH %u, Ymin %f, Ymin_dir %d, Ymin_dir_swap %d, XiYi_inf %d\n",
			row, col, (double)mesh_image[row][col].Xi_first, (double)mesh_image[row][col].Xi_last, mesh_image[row][col].Si,
			(double)mesh_image[row][col].Yi_first, (double)mesh_image[row][col].Yi_last, mesh_image[row][col].IBufW_req, mesh_image[row][col].IBufH_req,
			(double)mesh_image[row][col].Ymin, mesh_image[row][col].Ymin_dir, mesh_image[row][col].Ymin_dir_swap, mesh_image[row][col].XiYi_inf);
#endif

}

template < class mimage_type >
void irt_map_oimage_res_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, mimage_type** mesh_image) {

	uint16_t So = 0, Ho = 0;
	uint32_t OSize = 0;
	uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
	uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;

	bool Ymin_dir_cur, Ymin_dir_prv, Ymin_dir_swap = 0;

	for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {
		for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].W; col++) {

			if (row > 1) {
				Ymin_dir_cur = mesh_image[row - 0][col].Ymin > mesh_image[row - 1][col].Ymin;
				Ymin_dir_prv = mesh_image[row - 1][col].Ymin > mesh_image[row - 2][col].Ymin;
			} else {
				Ymin_dir_cur = 1; Ymin_dir_prv = 1;
			}

			Ymin_dir_swap = Ymin_dir_swap | (Ymin_dir_cur != Ymin_dir_prv);

			if ((mesh_image[row][col].IBufW_req <= BufW_act) && (mesh_image[row][col].IBufH_req <= BufH_act) &&
				!Ymin_dir_swap && !mesh_image[row][col].Ymin_dir_swap && mesh_image[row][col].Ymin_dir && mesh_image[row][col].XiYi_inf == 0) {
				if (((uint32_t)row + 1) * ((uint32_t)col + 1) > OSize) {
					So = col + 1;
					Ho = row + 1;
					OSize = (uint32_t)So * Ho;
				}
			}
		}
	}

	irt_desc.image_par[OIMAGE].S = So;
	irt_desc.image_par[OIMAGE].H = Ho;

	if (So > 0 && Ho > 0)
		irt_map_iimage_stripe_adj <mimage_type> (irt_cfg, rot_pars, irt_desc, desc, mesh_image);
}

template < class mimage_type >
void irt_map_iimage_stripe_adj(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, mimage_type** mesh_image) {

	rot_pars.IBufW_req = mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].IBufW_req;
	rot_pars.IBufH_req = mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].IBufH_req;
	rot_pars.Yi_first  = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Yi_first;
	rot_pars.Yi_last   = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Yi_last;
	rot_pars.Xi_first  = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Xi_first;
	rot_pars.Xi_last   = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Xi_last;
	irt_desc.image_par[IIMAGE].S = mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Si;

	switch (rot_pars.mesh_mode) {
	case e_irt_rotation:
	case e_irt_affine:
	case e_irt_projection:
		if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {
			rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
		}
		break;
	case e_irt_mesh:
		irt_desc.rot_dir = IRT_ROT_DIR_NEG; //IRT_ROT_DIR_POS;
		irt_desc.rot90 = 0;
		//rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
		break;
	}

	rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first);// -irt_desc.image_par[IIMAGE].S;
}

uint8_t irt_proc_size_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 }, Yi_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 };
	int XL_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 }, XR_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 }, YT_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 }, YB_fixed[IRT_ROT_MAX_PROC_SIZE] = { 0 };
	int int_win_w[IRT_ROT_MAX_PROC_SIZE], int_win_h[IRT_ROT_MAX_PROC_SIZE];
	uint8_t adj_proc_size = 1;

	for (uint8_t k = 0; k < IRT_ROT_MAX_PROC_SIZE; k++) { //calculating corners for 8 proc sizes
		switch (irt_desc.irt_mode) {
		case e_irt_rotation:
			Xi_fixed[k] = (int64_t)irt_desc.cosi * k;
			Yi_fixed[k] = (int64_t)irt_desc.sini * k;
			break;
		case e_irt_affine:
			Xi_fixed[k] = (int64_t)irt_desc.M11i * k;
			Yi_fixed[k] = (int64_t)irt_desc.M21i * k;
			break;
		case e_irt_projection:
			switch (rot_pars.proj_mode) {
			case e_irt_rotation:
				Xi_fixed[k] = (int64_t)irt_desc.cosi * k;
				Yi_fixed[k] = (int64_t)irt_desc.sini * k;
				break;
			case e_irt_affine:
				Xi_fixed[k] = (int64_t)irt_desc.M11i * k;
				Yi_fixed[k] = (int64_t)irt_desc.M21i * k;
				break;
			default:
				break;
			}
			break;
		case e_irt_mesh:
			switch (rot_pars.mesh_mode) {
			case e_irt_rotation:
				Xi_fixed[k] = (int64_t)irt_desc.cosi * k;
				Yi_fixed[k] = (int64_t)irt_desc.sini * k;
				break;
			case e_irt_affine:
				Xi_fixed[k] = (int64_t)irt_desc.M11i * k;
				Yi_fixed[k] = (int64_t)irt_desc.M21i * k;
				break;
			default:
				break;
			}
			break;
		}
	}

	//calculating interpolation region corners for 8 proc sizes
	for (uint8_t k = 0; k < IRT_ROT_MAX_PROC_SIZE; k++) { //calculating corners for 8 proc sizes

		XL_fixed[k] = std::min(0, (int)(Xi_fixed[k] >> rot_pars.TOTAL_PREC));
		XR_fixed[k] = std::max(0, (int)(Xi_fixed[k] >> rot_pars.TOTAL_PREC)) + 1;
		YT_fixed[k] = std::min(0, (int)(Yi_fixed[k] >> rot_pars.TOTAL_PREC));
		YB_fixed[k] = std::max(0, (int)(Yi_fixed[k] >> rot_pars.TOTAL_PREC)) + 1;

		int_win_w[k] = XR_fixed[k] - XL_fixed[k] + 2;
		int_win_h[k] = YB_fixed[k] - YT_fixed[k] + 2;
		switch (irt_desc.irt_mode) {
		case e_irt_rotation:
//			IRT_TRACE_UTILS("Interpolation window size: k=%d: %dx%d %dx%d\n", k + 1, int_win_h[k], int_win_w[k], (irt_desc.sind[k] >> IRT_TOTAL_PREC) + 2, (irt_desc.cosd[k] >> IRT_TOTAL_PREC) + 2);
			break;
		default: break;
		}

		if ((int_win_h[k] <= IRT_INT_WIN_H) && (int_win_w[k] <= IRT_INT_WIN_W)) {//fit interpolation window
			adj_proc_size = k + 1; //set for number of already processed pixels in group
		}
	}

	if (adj_proc_size > IRT_ROT_MAX_PROC_SIZE) {
		adj_proc_size = IRT_ROT_MAX_PROC_SIZE;
	}

	if (adj_proc_size > 2) {
		//adj_proc_size--;
	}

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: processing rate is limited to %d pixels/cycle\n", desc, adj_proc_size);

	return adj_proc_size;
}

void irt_mesh_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {
	//IBufW_req is required bytes per mesh lines to be stored in memory, equal to Wm multiplied by pixel size:
	//BufW (byte)						32 64 128 256 512 1024 2048 4096
	//BufW (pixels) for 4 byte/pixel -	 8 16  32  64 128  256  512 1024
	//BufW (pixels) for 8 byte/pixel -	 2  4   8  16  32   64  128  256
	uint16_t Si_log2 = (uint16_t)ceil(log2((double)irt_desc.image_par[MIMAGE].S)); //for 90 degree rotation Si+1 is not needed
	uint16_t Si_pxls = (uint16_t)pow(2.0, Si_log2);
	rot_pars.MBufW_req = Si_pxls << irt_desc.image_par[MIMAGE].Ps;
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Mesh mem fit check: Si_log2 %u, Si_pxls %u, Ps %u, MBufW_req %u\n", Si_log2, Si_pxls, irt_desc.image_par[MIMAGE].Ps, rot_pars.MBufW_req);

#if 0
	if (hl_only == 0 && rot_pars.MBufW_req > MAX_SM_WIDTH) {
		IRT_TRACE_UTILS("Mesh mem fit check: mesh stripe is not supported: required mesh stripe width %d pixels (%d byte) exceeds %d bytes\n",
			Si_pxls, rot_pars.MBufW_req, MAX_SM_WIDTH);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, mesh stripe is not supported: required mesh stripe width %d pixels (%d byte) exceeds %d bytes\n",
			Si_pxls, rot_pars.MBufW_req, MAX_SM_WIDTH);
		IRT_CLOSE_FAILED_TEST(0);
	}
	else if (rot_pars.MBufW_req < MIN_SM_WIDTH) {
		rot_pars.MBufW_req = MIN_SM_WIDTH;
	}
#endif

	uint8_t buf_mode = (uint8_t)(log2((double)rot_pars.MBufW_req / MIN_SM_WIDTH));

	if (irt_cfg.buf_select[e_irt_block_mesh] == e_irt_buf_select_auto) { //auto
		irt_desc.image_par[MIMAGE].buf_mode = buf_mode;

		if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_static)
			irt_desc.image_par[MIMAGE].buf_mode = (uint8_t)std::min((int32_t)irt_desc.image_par[MIMAGE].buf_mode, IRT_MM_MAX_STAT_MODE); //mode 7 is not supported
		else
			irt_desc.image_par[MIMAGE].buf_mode = (uint8_t)std::min((int32_t)irt_desc.image_par[MIMAGE].buf_mode, IRT_MM_MAX_DYN_MODE); //mode 6 and 7 is not supported

	} else { //manual
		irt_desc.image_par[MIMAGE].buf_mode = irt_cfg.buf_mode[e_irt_block_mesh];
	}

	// IBufH_req is required lines to be stored in memory
	rot_pars.MBufH_req = 2 + (irt_desc.image_par[MIMAGE].H > 2); //need 1 more line for interpolation for sparse matrix
	if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic && (irt_desc.image_par[MIMAGE].H > 2))
		rot_pars.MBufH_req += IRT_MM_DYN_MODE_LINE_RELEASE;  //incase of dynamic mode we discard 2 lines at a time

	//BufH_act - number of input lines that can be stored in rotation memory if we store BufW_req pixels per line
	uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufW;
	uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufH;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc %u: Mesh memory fit check: mesh stripe of %u pixels requires %ux%u mesh buffer\n",
		desc, irt_desc.image_par[MIMAGE].S, rot_pars.MBufW_req, rot_pars.MBufH_req);

	if (hl_only == 0) {
		if (rot_pars.MBufH_req > BufH_act || rot_pars.MBufW_req > BufW_act) {
			if (rot_pars.oimg_auto_adj == 0) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc %d: Mesh memory fit check: mesh stripe is not supported: required mesh stripe width %d pixels (%d bytes) and mesh memory size %ux%u for buf_mode %d exceeds %ux%u\n",
					desc, irt_desc.image_par[MIMAGE].S, irt_desc.image_par[MIMAGE].S << irt_desc.image_par[MIMAGE].Ps,
					rot_pars.MBufW_req, rot_pars.MBufH_req, irt_desc.image_par[MIMAGE].buf_mode, BufW_act, BufH_act);
				IRT_TRACE_TO_RES_UTILS(test_res, "was not run, mesh stripe is not supported: required mesh stripe width %d pixels (%d bytes) and mesh memory size %ux%u for buf_mode %d exceeds %ux%u\n",
					irt_desc.image_par[MIMAGE].S, irt_desc.image_par[MIMAGE].S << irt_desc.image_par[MIMAGE].Ps,
					rot_pars.MBufW_req, rot_pars.MBufH_req, irt_desc.image_par[MIMAGE].buf_mode, BufW_act, BufH_act);
				IRT_CLOSE_FAILED_TEST(0);
			} else {
				if (rot_pars.MBufH_req > BufH_act) {
					if (irt_cfg.buf_select[e_irt_block_mesh] == e_irt_buf_select_auto) {
						irt_desc.image_par[MIMAGE].buf_mode--;
						BufW_act = BufW_act >> 1;
					} else {
						IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc %u: Mesh memory fit check: mesh stripe is not supported: required mesh stripe height %u lines and mesh memory size %ux%u for buf_mode %u exceeds %ux%u\n",
							desc, rot_pars.MBufH_req, rot_pars.MBufW_req, rot_pars.MBufH_req, irt_desc.image_par[MIMAGE].buf_mode, BufW_act, BufH_act);
						IRT_TRACE_TO_RES_UTILS(test_res, "was not run, mesh stripe is not supported: required mesh stripe height %u lines and mesh memory size %ux%u for buf_mode %u exceeds %ux%u\n",
							rot_pars.MBufH_req, rot_pars.MBufW_req, rot_pars.MBufH_req, irt_desc.image_par[MIMAGE].buf_mode, BufW_act, BufH_act);
						IRT_CLOSE_FAILED_TEST(0);
					}
				}
				if (rot_pars.MBufW_req > BufW_act) {
					irt_desc.image_par[MIMAGE].S = BufW_act >> irt_desc.image_par[MIMAGE].Ps;
					if (rot_pars.mesh_Sh == 1) //non-sparse
						irt_desc.image_par[OIMAGE].S = std::min(irt_desc.image_par[OIMAGE].S, irt_desc.image_par[MIMAGE].S);
					else
						irt_desc.image_par[OIMAGE].S = std::min(irt_desc.image_par[OIMAGE].S, (uint16_t)floor(((double)irt_desc.image_par[MIMAGE].S - 1) * rot_pars.mesh_Sh));

					irt_oimage_corners_calc(irt_cfg, rot_pars, irt_desc, desc);

					irt_map_iimage_stripe_adj <mesh_xy_fp64_meta> (irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
				}
			}
		}
	}

	if (desc == 0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %u: Required mesh memory size for Sm = %u is (%ux%u), mode %d (%ux%u) is used\n",
			desc, irt_desc.image_par[MIMAGE].S, rot_pars.MBufW_req, rot_pars.MBufH_req,
			irt_desc.image_par[MIMAGE].buf_mode,
			irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufW,
			irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufH);
	}
}

void irt_oimage_res_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	uint16_t Si_log2;
	uint32_t Si_pxls;
	int32_t Si_adj, So_from_IBufW = 0, So_from_IBufH = 0, Si;

#if 0
	double M11, M12, M21, M22;

	if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
		M11 = (double)irt_desc.M11f;
		M12 = (double)irt_desc.M12f;
		M21 = (double)irt_desc.M21f;
		M22 = (double)irt_desc.M22f;
	} else {
		M11 = (double)irt_desc.M11i / pow(2.0, rot_pars.TOTAL_PREC);
		M12 = (double)irt_desc.M12i / pow(2.0, rot_pars.TOTAL_PREC);
		M21 = (double)irt_desc.M21i / pow(2.0, rot_pars.TOTAL_PREC);
		M22 = (double)irt_desc.M22i / pow(2.0, rot_pars.TOTAL_PREC);
	}
#endif

	uint32_t BufW_act = (uint32_t)irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
	uint32_t BufH_act = (uint32_t)irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;
	bool IBufW_fit = rot_pars.IBufW_req <= BufW_act;
	bool IBufH_fit = rot_pars.IBufH_req <= BufH_act;
	bool fit_width = 0, fit_height = 0;
	uint16_t Ho = irt_desc.image_par[OIMAGE].H;
	//uint16_t Wo = irt_desc.image_par[OIMAGE].W;
	uint16_t So = irt_desc.image_par[OIMAGE].S;
	uint16_t Hm = irt_desc.image_par[MIMAGE].H;
	//uint16_t Wo = irt_desc.image_par[OIMAGE].W;
	uint16_t Sm = irt_desc.image_par[MIMAGE].S;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc %d: Rotation memory fit check adjusts output stripe resolution to fit rotation memory buffer mode %d for IRT %s mode\n", desc, irt_desc.image_par[IIMAGE].buf_mode, irt_irt_mode_s[irt_desc.irt_mode]);

	Si_pxls = BufW_act >> irt_desc.image_par[IIMAGE].Ps;
	Si_log2 = (uint16_t)floor(log2((double)Si_pxls));
	Si_adj = (int32_t)pow(2.0, Si_log2);
	if (irt_desc.rot90 == 0) {
		Si_adj -= irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
		Si_adj -= (int32_t)ceil(2 * rot_pars.Si_delta) + 2; //(uint16_t)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
	}
	Si = Si_adj - 1 + irt_desc.rot90 - (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);

#if 0
	bool affine_rotation				 = strcmp(rot_pars.affine_mode, "R") == 0;
	bool projection_direct_rotation		 = rot_pars.proj_mode == e_irt_rotation;
	bool projection_affine_rotation		 = rot_pars.proj_mode == e_irt_affine && affine_rotation;
	bool projection_rotation			 = projection_direct_rotation | projection_affine_rotation;
	bool mesh_direct_rotation			 = rot_pars.mesh_mode == e_irt_rotation;
	bool mesh_affine_rotation			 = rot_pars.mesh_mode == e_irt_affine && affine_rotation;
	bool mesh_projection_rotation		 = rot_pars.mesh_mode == e_irt_projection && projection_rotation;
	bool mesh_rotation					 = mesh_direct_rotation | mesh_affine_rotation | mesh_projection_rotation;

	bool projection_affine				 = rot_pars.proj_mode == e_irt_affine;
	bool mesh_direct_affine				 = rot_pars.mesh_mode == e_irt_affine;
	bool mesh_projection_affine			 = rot_pars.mesh_mode == e_irt_projection && projection_affine;
	bool mesh_affine					 = mesh_direct_affine | mesh_projection_affine;

	IRT_TRACE_UTILS("affine_rotation			= %d\n", affine_rotation);
	IRT_TRACE_UTILS("projection_direct_rotation = %d\n", projection_direct_rotation);
	IRT_TRACE_UTILS("projection_affine_rotation = %d\n", projection_affine_rotation);
	IRT_TRACE_UTILS("projection_rotation		= %d\n", projection_rotation);
	IRT_TRACE_UTILS("mesh_direct_rotation		= %d\n", mesh_direct_rotation);
	IRT_TRACE_UTILS("mesh_affine_rotation		= %d\n", mesh_affine_rotation);
	IRT_TRACE_UTILS("mesh_projection_rotation	= %d\n", mesh_projection_rotation);
	IRT_TRACE_UTILS("mesh_rotation				= %d\n", mesh_rotation);
	IRT_TRACE_UTILS("projection_affine			= %d\n", projection_affine);
	IRT_TRACE_UTILS("mesh_direct_affine			= %d\n", mesh_direct_affine);
	IRT_TRACE_UTILS("mesh_projection_affine		= %d\n", mesh_projection_affine);
	IRT_TRACE_UTILS("mesh_affine				= %d\n", mesh_affine);

	bool irt_do_rotate = irt_desc.irt_mode == e_irt_rotation || 
						(irt_desc.irt_mode == e_irt_affine && affine_rotation) ||
						(irt_desc.irt_mode == e_irt_projection && projection_rotation) ||
						(irt_desc.irt_mode == e_irt_mesh && mesh_rotation && rot_pars.mesh_dist_r0_Sh1_Sv1 && rot_pars.mesh_matrix_error == 0);

	bool irt_do_affine = irt_desc.irt_mode == e_irt_affine ||
						(irt_desc.irt_mode == e_irt_projection && projection_affine) ||
						(irt_desc.irt_mode == e_irt_mesh && mesh_affine && rot_pars.mesh_dist_r0_Sh1_Sv1 && rot_pars.mesh_matrix_error == 0);
#endif

	bool irt_do_rotate = rot_pars.irt_rotate && rot_pars.mesh_matrix_error == 0;

	bool irt_do_affine = rot_pars.irt_affine && rot_pars.mesh_matrix_error == 0;

	if (irt_do_rotate) {
		if (irt_desc.rot90) {
			irt_desc.image_par[OIMAGE].H = (uint16_t)std::min(Si_pxls, (uint32_t)irt_desc.image_par[OIMAGE].H);
			So_from_IBufW = (int32_t)irt_desc.image_par[OIMAGE].S;
			So_from_IBufH = BufH_act - (/*irt_desc.rot90_inth |*/ irt_desc.rot90_intv);
		} else {
			if (rot_pars.use_rectangular_input_stripe == 0)
				So_from_IBufW = (int32_t)floor(((double)Si - 1) * cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0));
			else
				So_from_IBufW = (int32_t)floor(((double)Si + ceil(2 * rot_pars.Si_delta) - 1 - fabs(sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)) * (irt_desc.image_par[OIMAGE].H - 1)) / cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0) - 1);
			So_from_IBufH = (int32_t)floor(((double)BufH_act - 2 - (irt_cfg.buf_format[e_irt_block_rot] ? IRT_RM_DYN_MODE_LINE_RELEASE : 1)) / sin(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
		}
		if (So_from_IBufW > 0 && So_from_IBufH > 0) {
			fit_width = 1; fit_height = 1; IBufW_fit = 1; IBufH_fit = 1;
		}
	} else if (irt_do_affine) {
		if (irt_desc.rot90) {
			//irt_desc.image_par[OIMAGE].H = irt_desc.image_par[IIMAGE].S;
			irt_desc.image_par[OIMAGE].H = (uint16_t)std::min(Si_pxls, (uint32_t)irt_desc.image_par[OIMAGE].H) - rot_pars.irt_affine_st_inth;
			So_from_IBufW = (int32_t)irt_desc.image_par[OIMAGE].S;
			So_from_IBufH = BufH_act - (/*irt_desc.rot90_inth |*/ irt_desc.rot90_intv);
		} else {
			if (rot_pars.use_rectangular_input_stripe == 0)
				So_from_IBufW = (int32_t)floor(((double)Si - 1) / rot_pars.affine_Si_factor);
			else
				So_from_IBufW = (int32_t)floor(((double)Si + ceil(2 * rot_pars.Si_delta) - 1 - fabs(rot_pars.M12d) * (irt_desc.image_par[OIMAGE].H - 1)) / fabs(rot_pars.M11d) - 1);
			if (rot_pars.M21d != 0) {
				So_from_IBufH = (int32_t)floor(((double)BufH_act - 2 - rot_pars.IBufH_delta - (irt_cfg.buf_format[e_irt_block_rot] ? IRT_RM_DYN_MODE_LINE_RELEASE : 1)) / fabs(rot_pars.M21d));
			} else {
				So_from_IBufH = So_from_IBufW;
			}
		}
		if (So_from_IBufW > 0 && So_from_IBufH > 0) {
			fit_width = 1; fit_height = 1; IBufW_fit = 1; IBufH_fit = 1;
		}
	} else if  ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) || (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection) ||
				(irt_desc.irt_mode == e_irt_mesh /*&& rot_pars.mesh_mode == e_irt_mesh*/)) {

		//next fit approach is taken:
		//fit width -> fit height
		//else fit height -> fit width
		//else fir both

		//fit width -> fit height
		//trying to reduce output image width to fit rotation memory

		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Trying to reduce output image width to fit rotation memory, it takes time, be patient...\n");
		if (irt_desc.irt_mode == e_irt_mesh)
			irt_map_oimage_res_adj_calc <mesh_xy_fp64_meta> (irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
		else
			irt_map_oimage_res_adj_calc <mesh_xy_fp64_meta> (irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.proj_image_full);
		if (irt_desc.image_par[OIMAGE].S > 0 && irt_desc.image_par[OIMAGE].H > 0) {
			fit_width = 1; fit_height = 1;
			IBufW_fit = 1; IBufH_fit = 1;
		}

		So_from_IBufW = (uint32_t)irt_desc.image_par[OIMAGE].S;
		So_from_IBufH = (uint32_t)irt_desc.image_par[OIMAGE].S;
	} else {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc %d: Output image resolution auto adjust does not supported in this irt mode\n", desc);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, desc %d Output image resolution auto adjust does not supported in this irt mode\n", desc);
		IRT_CLOSE_FAILED_TEST(0);
	}

	irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufW);
	irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufH);
	if (irt_desc.irt_mode == e_irt_mesh) {
		irt_desc.image_par[MIMAGE].S = (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].S - 1) / rot_pars.mesh_Sh) + 1;
		irt_desc.image_par[MIMAGE].H = (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].H - 1) / rot_pars.mesh_Sv) + 1;
		if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) { //stripe height is multiple of 2
			irt_desc.image_par[MIMAGE].H = (uint16_t)(ceil((double)irt_desc.image_par[MIMAGE].H / IRT_MM_DYN_MODE_LINE_RELEASE) * IRT_MM_DYN_MODE_LINE_RELEASE);
		}
	}


	if (fit_width && fit_height && IBufW_fit && IBufH_fit && irt_desc.image_par[OIMAGE].S > 0 && irt_desc.image_par[OIMAGE].H > 0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image resolution auto adjust: for buffer mode %u with %ux%u buffer size, So from BufW is %d, So from BufH is %d, selected So is %u\n",
			irt_desc.image_par[IIMAGE].buf_mode, rot_pars.IBufW_req, rot_pars.IBufH_req, So_from_IBufW, So_from_IBufH, irt_desc.image_par[OIMAGE].S);
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image resolution auto adjust: for buffer mode %u with %ux%u buffer size, output image size is reduced from %ux%u to %ux%u in %s mode\n",
			irt_desc.image_par[IIMAGE].buf_mode, rot_pars.IBufW_req, rot_pars.IBufH_req, So, Ho, irt_desc.image_par[OIMAGE].S, irt_desc.image_par[OIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image resolution auto adjust: input image stripe is %ux%u\n", irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].Hs);
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image resolution auto adjust: mesh image size is reduced from %ux%u to %ux%u in %s mode\n",
			Sm, Hm, irt_desc.image_par[MIMAGE].S, irt_desc.image_par[MIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
	} else {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc %u: Output image resolution auto adjust cannot find output stripe resolution for defined %s parameters for buffer mode %u\n",
			desc, irt_irt_mode_s[irt_desc.irt_mode], irt_desc.image_par[IIMAGE].buf_mode);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, desc %u Output image resolution auto adjust cannot find output stripe resolution for defined %s parameters for buffer mode %u\n",
			desc, irt_irt_mode_s[irt_desc.irt_mode], irt_desc.image_par[IIMAGE].buf_mode);
		IRT_CLOSE_FAILED_TEST(0);
	}

#if 0
	rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
	if (irt_h5_mode) {
		if ((irt_desc.image_par[OIMAGE].S % 8) != 0) {//not multiple of 8, round down to multiple of 8
			irt_desc.image_par[OIMAGE].S -= 8;
			rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
			rot_pars.So8 += (8 - (irt_desc.image_par[OIMAGE].S % 8));
		}
	}

	//irt_desc.image_par[OIMAGE].W = irt_desc.image_par[OIMAGE].S;
	rot_pars.Xo_first = -(double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Xo_last  = (double)rot_pars.So8 - 1 - (double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Yo_first = -(double)irt_desc.image_par[OIMAGE].Yc / 2;
	rot_pars.Yo_last  = (double)irt_desc.image_par[OIMAGE].H - 1 - (double)irt_desc.image_par[OIMAGE].Yc / 2;
#endif

	irt_oimage_corners_calc(irt_cfg, rot_pars, irt_desc, desc);

	if (irt_do_rotate) {
		irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
		irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
	} else if (irt_do_affine) {
		irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
		irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc);
	}
	rot_pars.IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, rot_pars.IBufH_req);
}

void irt_oimage_corners_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
	if (irt_h5_mode) {
		if ((irt_desc.image_par[OIMAGE].S % 8) != 0) {//not multiple of 8, round down to multiple of 8
			irt_desc.image_par[OIMAGE].S -= 8;
			rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
			rot_pars.So8 += (8 - (irt_desc.image_par[OIMAGE].S % 8));
		}
	}

	//irt_desc.image_par[OIMAGE].W = irt_desc.image_par[OIMAGE].S;
	rot_pars.Xo_first = -(double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Xo_last = (double)rot_pars.So8 - 1 - (double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Yo_first = -(double)irt_desc.image_par[OIMAGE].Yc / 2;
	rot_pars.Yo_last = (double)irt_desc.image_par[OIMAGE].H - 1 - (double)irt_desc.image_par[OIMAGE].Yc / 2;

}

uint32_t irt_IBufW_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t Si) {

	uint16_t Si_log2;
	uint32_t Si_adj, Si_pxls;
	Si_adj = Si + 1 - irt_desc.rot90 + (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/); //for 90 degree rotation Si+1 is not needed
	if (irt_desc.rot90 == 0) {
		//Si_adj += (uint16_t)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
		Si_adj += irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
	}
	Si_log2 = (uint16_t)ceil(log2((double)Si_adj));
	Si_pxls = (uint32_t)pow(2.0, Si_log2);
	return Si_pxls << irt_desc.image_par[IIMAGE].Ps;
}

uint32_t irt_IBufH_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t IBufH) {

	uint32_t IBufH_req;

	if (irt_desc.rot90 == 0) {
		IBufH_req = IBufH + 1 + 1 + 1; //1 because of interpolation, 1 because of Y is released based on previous line
		if (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic)
			IBufH_req += IRT_RM_DYN_MODE_LINE_RELEASE;  //incase of dynamic mode we discard 8 lines at a time
	} else {
		IBufH_req = IBufH + (irt_desc.rot90_intv /*| irt_desc.rot90_inth*/);
	}

	return IBufH_req;
}

void irt_rotation_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	//IBufW_req is required bytes per input lines to be stored in memory, equal to Si round up to power of 2 and multiplied by pixel size:
	//BufW (byte)						32 64 128 256 512 1024 2048 4096
	//BufW (pixels) for 1 byte/pixel -	32 64 128 256 512 1024 2048 4096
	//BufW (pixels) for 2 byte/pixel -	16 32  64 128 256  512 1024 2048

	rot_pars.IBufW_req = irt_IBufW_req_calc(irt_cfg, irt_desc, irt_desc.image_par[IIMAGE].S);

	uint8_t buf_mode = (uint8_t)(log2((double)rot_pars.IBufW_req / MIN_SI_WIDTH));
	//BufH_available - number of input lines that can be stored in rotation memory if we store IBufW_req pixels per line

	if (irt_cfg.buf_select[0] == 1 /*auto*/ && irt_h5_mode == 0) {// && irt_desc.rot90 == 0) {
		irt_desc.image_par[IIMAGE].buf_mode = buf_mode;
	} else {
		irt_desc.image_par[IIMAGE].buf_mode = irt_cfg.buf_mode[0];
	}

	if (irt_cfg.buf_format[0] == e_irt_buf_format_dynamic)
		irt_desc.image_par[IIMAGE].buf_mode = (uint8_t)std::min((int32_t)irt_desc.image_par[IIMAGE].buf_mode, IRT_RM_MAX_DYN_MODE); //buffer mode 7 is not supported in e_irt_buf_format_dynamic
	else
		irt_desc.image_par[IIMAGE].buf_mode = (uint8_t)std::min((int32_t)irt_desc.image_par[IIMAGE].buf_mode, IRT_RM_MAX_STAT_MODE);

	uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
	uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc %u: Rotation memory fit check: output stripe of %d pixels requires input stripe width %d pixels (%d bytes) and %ux%u rotation buffer, input memory size for buf_mode %d is %ux%u\n",
		desc, irt_desc.image_par[OIMAGE].S, irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].S << irt_desc.image_par[IIMAGE].Ps, rot_pars.IBufW_req, rot_pars.IBufH_req,
		irt_desc.image_par[IIMAGE].buf_mode, BufW_act, BufH_act);

	if (hl_only == 0) {
		if ((rot_pars.IBufH_req > BufH_act || rot_pars.IBufW_req > BufW_act) && /*irt_desc.rot90 == 0 &&*/ irt_h5_mode == 0) {
			if (rot_pars.oimg_auto_adj == 0) {
				//IRT_TRACE_UTILS("Desc %d: IRT rotation memory fit check: Rotation angle and output stripe are not supported: required rotation memory size %dx%d exceeds %dx%d of buf_mode %d\n", desc, rot_pars.IBufW_req, rot_pars.IBufH_req, BufW_act, BufH_act, irt_desc.image_par[IIMAGE].buf_mode);
				//IRT_TRACE_TO_RES_UTILS(test_res, "was not run, desc %d rotation angle and output stripe are not supported: required rotation memory size %dx%d exceeds %dx%d of buf_mode %d\n", desc, rot_pars.IBufW_req, rot_pars.IBufH_req, BufW_act, BufH_act, irt_desc.image_par[IIMAGE].buf_mode);
				//IRT_CLOSE_FAILED_TEST(0);

				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc %d: Rotation memory fit check: output stripe is not supported: required input stripe width %d pixels (%d bytes) and input memory size %ux%u for buf_mode %d exceeds %ux%u\n",
					desc, irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].S << irt_desc.image_par[IIMAGE].Ps,
					rot_pars.IBufW_req, rot_pars.IBufH_req, irt_desc.image_par[IIMAGE].buf_mode, BufW_act, BufH_act);
				IRT_TRACE_TO_RES_UTILS(test_res, "was not run, output stripe is not supported: required input stripe width %d pixels (%d bytes) and memory size %ux%u for buf_mode %d exceeds %ux%u\n",
					irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].S << irt_desc.image_par[IIMAGE].Ps,
					rot_pars.IBufW_req, rot_pars.IBufH_req, irt_desc.image_par[IIMAGE].buf_mode, BufW_act, BufH_act);
				IRT_CLOSE_FAILED_TEST(0);

			} else {
				//rot_pars.IBufW_req = (uint32_t)BufW_act;
				//rot_pars.IBufH_req = (uint32_t)BufH_act;
				irt_oimage_res_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
			}
		} else { //in rotation/affine/projection with rotation/affine and mesh with rotation/affine w/o distortion and w/o sparsing we can rely on IBufW_req and IBufH_req linear calculation and not use map
			if ((irt_desc.irt_mode == e_irt_projection &&  rot_pars.proj_mode == e_irt_projection) ||
				(irt_desc.irt_mode == e_irt_mesh       && (rot_pars.mesh_mode == e_irt_projection || rot_pars.mesh_mode == e_irt_mesh || rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)))
				irt_oimage_res_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
		}
	}

	if (desc == 0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Required rotation memory size for Si = %d and angle %f is (%ux%u), mode %d (%dx%d) is used\n",
			desc, irt_desc.image_par[IIMAGE].S, rot_pars.irt_angles_adj[e_irt_angle_rot], rot_pars.IBufW_req, rot_pars.IBufH_req,
			irt_desc.image_par[IIMAGE].buf_mode, irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW,
			irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH);
	}
}

void irt_rotation_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	double cosD, sinD;

	if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
		cosD = (double)irt_desc.cosf;
		sinD = (double)irt_desc.sinf;
	} else {
		cosD = (double)irt_desc.cosi / pow(2.0, rot_pars.TOTAL_PREC);
		sinD = (double)irt_desc.sini / pow(2.0, rot_pars.TOTAL_PREC);
	}

	rot_pars.Si_delta = tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0);

	//if ((rot_pars.Si_delta - floor(rot_pars.Si_delta)) >= 0.75) //ceiling if >= x.5
	//	rot_pars.Si_delta = ceil(rot_pars.Si_delta);
	if (rot_pars.Si_delta > 50.0) rot_pars.Si_delta += 3;
	else if (rot_pars.use_Si_delta_margin && rot_pars.Si_delta != 0) rot_pars.Si_delta += 1;
	else if (rot_pars.Si_delta > 6.0) rot_pars.Si_delta += 0.73;
	else if (rot_pars.Si_delta > 4.0) rot_pars.Si_delta += 0.68;
	else if (rot_pars.Si_delta > 3.0) rot_pars.Si_delta += 0.64;
	else if (rot_pars.Si_delta > 2.0) rot_pars.Si_delta += 0.56;
	else if (rot_pars.Si_delta > 1.0) rot_pars.Si_delta += 0.55;
	else if (rot_pars.Si_delta != 0)  rot_pars.Si_delta += 0.40;//0.26;

	if (irt_desc.rot90) {
		irt_desc.image_par[IIMAGE].S = irt_desc.image_par[OIMAGE].H + (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);
	} else {
		irt_desc.image_par[IIMAGE].S = (uint16_t)ceil((double)rot_pars.So8 / cos(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0) + 2.0 * rot_pars.Si_delta) + 2;
	}

	rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.So8 * sin(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0)));

	if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { //positive rotation angle

		rot_pars.Xi_first = ( (double)irt_desc.cosi * rot_pars.Xo_last  + (double)irt_desc.sini * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (-(double)irt_desc.sini * rot_pars.Xo_last  + (double)irt_desc.cosi * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Yi_last  = (-(double)irt_desc.sini * rot_pars.Xo_first + (double)irt_desc.cosi * rot_pars.Yo_last)  / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Yc / 2;

		rot_pars.Xi_first = ( cosD * rot_pars.Xo_last  + sinD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (-sinD * rot_pars.Xo_last  + cosD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Yi_last  = (-sinD * rot_pars.Xo_first + cosD * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;
#if 1
		rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first) - irt_desc.image_par[IIMAGE].S;

		if (irt_desc.rot90 == 0)
			rot_pars.Xi_first = rot_pars.Xi_first + rot_pars.Si_delta + 2;

		if (irt_desc.rot90 == 1) {
			rot_pars.Xi_start += irt_desc.image_par[IIMAGE].S;
			//rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
		}
#endif
	} else {

		rot_pars.Xi_first = ( (double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (-(double)irt_desc.sini * rot_pars.Xo_first + (double)irt_desc.cosi * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Yi_last  = (-(double)irt_desc.sini * rot_pars.Xo_last  + (double)irt_desc.cosi * rot_pars.Yo_last)  / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Yc / 2;

		rot_pars.Xi_first = ( cosD * rot_pars.Xo_first + sinD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (-sinD * rot_pars.Xo_first + cosD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Yi_last  = (-sinD * rot_pars.Xo_last  + cosD * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;

#if 1
		rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first - rot_pars.Si_delta) - irt_desc.read_hflip - 1;

		if (irt_desc.rot90 == 0)
			rot_pars.Xi_first = rot_pars.Xi_first - rot_pars.Si_delta - 1;

		if (irt_desc.rot90 == 1) {
			rot_pars.Xi_start -= irt_desc.image_par[IIMAGE].S;
		}
#endif
	}

	if (irt_desc.rot90 == 0)
		rot_pars.im_read_slope = tan((double)rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0);
	else
		rot_pars.im_read_slope = 0;

	if (irt_desc.crd_mode == e_irt_crd_mode_fp32 && irt_desc.rot90 == 0) {
		rot_pars.IBufH_req++;
		rot_pars.Yi_first--;
	}

#if 1
	double Xi_TL = ((double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Xi_TR = ((double)irt_desc.cosi * rot_pars.Xo_last  + (double)irt_desc.sini * rot_pars.Yo_first) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Xi_BL = ((double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_last ) / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Xi_BR = ((double)irt_desc.cosi * rot_pars.Xo_last  + (double)irt_desc.sini * rot_pars.Yo_last)  / pow(2.0, rot_pars.TOTAL_PREC) + (double)irt_desc.image_par[IIMAGE].Xc / 2;

	//cos is always > 0 for [-90:90] rotation range, Xo_last > Xo_first => Xi_TR > Xi_TL
	//for same reason Xi_BR > Xi_BL
	// Xi_TL < Xi_TR ; Xi_BL < Xi_BR
	// Yo_first < Yo_last
	// if rot_dir positive and sin > 0 => Xi_TL < Xi_BL and Xi_TR < Xi_BR

	/*
	because Xoc and Yoc do not change Xi_XX relation, we can indentify relations as:
	Xi_TL = cosi * 0		+ sini * 0		= 0
	Xi_TR = cosi * (So-1)	+ sini * 0		= cosi * (So-1)
	Xi_BL = cosi * 0		+ sini * (Ho-1)	= sini * (Ho-1)
	Xi_BR = cosi * (So-1)	+ sini * (Ho-1)	= cosi * (So-1)	+ sini * (Ho-1)

	if rot_dir > 0, then sin and cos are > 0, then Xi_TL < Xi_TR, Xi_TL < Xi_BL, Xi_TR < Xi_BR, Xi_BL < Xi_BR and at the end Xi_TL is min value and Xi_BR is max value
	then rectangular input stripe Si = cosi * (So-1)	+ sini * (Ho-1)

	if rot_dir < 0, then sin < 0 and cos > 0, then Xi_TL < Xi_TR, Xi_TL > Xi_BL, Xi_TR > Xi_BR, Xi_BL < Xi_BR and at the end Xi_BL is min value and Xi_TR is max value
	then rectangular input stripe Si = cosi * (So-1) - sini * (Ho-1)

	And then input stripe Si = cosi * (So-1) + abs(sini * (Ho-1))
	Then So = (Si - abs(sini * (Ho-1))) / cosi + 1 
	*/

	double XiL = fmin(fmin(fmin(Xi_TL, Xi_TR), Xi_BL), Xi_BR);
	double XiR = fmax(fmax(fmax(Xi_TL, Xi_TR), Xi_BL), Xi_BR);

	uint16_t Si = (uint16_t)(ceil(XiR) - floor(XiL) + 2.0 * rot_pars.Si_delta * 0) + 2;
	Si = (uint16_t)ceil(((double)irt_desc.cosi * ((double)irt_desc.image_par[OIMAGE].S - 1) + fabs((double)irt_desc.sini * ((double)irt_desc.image_par[OIMAGE].H - 1))) / pow(2.0, rot_pars.TOTAL_PREC)) + 3;
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Xi_L = %f, Xi_R = %f, Si = %d %d\n", XiL, XiR, Si, (uint16_t)ceil(((double)irt_desc.cosi* ((double)irt_desc.image_par[OIMAGE].S - 1) + fabs((double)irt_desc.sini * ((double)irt_desc.image_par[OIMAGE].H - 1))) / pow(2.0, rot_pars.TOTAL_PREC)));

	rot_pars.use_rectangular_input_stripe = 0;
	if (Si < irt_desc.image_par[IIMAGE].S && irt_desc.rot90 == 0) { //we can read less
		rot_pars.use_rectangular_input_stripe = 1;
		irt_desc.image_par[IIMAGE].S = Si;
		rot_pars.im_read_slope = 0;
		if (irt_desc.rot_dir == IRT_ROT_DIR_POS)
			rot_pars.Xi_first = XiR + 2;
		else
			rot_pars.Xi_first = XiL - 1;
	}
#endif

}

void irt_affine_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	double M11, M12, M21, M22;

	if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
		M11 = (double)irt_desc.M11f;
		M12 = (double)irt_desc.M12f;
		M21 = (double)irt_desc.M21f;
		M22 = (double)irt_desc.M22f;
	} else {
		M11 = (double)irt_desc.M11i / pow(2.0, rot_pars.TOTAL_PREC);
		M12 = (double)irt_desc.M12i / pow(2.0, rot_pars.TOTAL_PREC);
		M21 = (double)irt_desc.M21i / pow(2.0, rot_pars.TOTAL_PREC);
		M22 = (double)irt_desc.M22i / pow(2.0, rot_pars.TOTAL_PREC);
	}

	rot_pars.Si_delta = ceil(fabs(rot_pars.M12d / rot_pars.M22d));
	//rot_pars.Si_delta = ceil(fabs(M12 / M22)) + 1;
	rot_pars.Si_delta = fabs(M12 / M22);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Affine descriptor generator: Si_delta = %f\n", rot_pars.Si_delta);
	if ((rot_pars.Si_delta - floor(rot_pars.Si_delta)) >= 0.5) //ceiling if >= x.5
		rot_pars.Si_delta = ceil(rot_pars.Si_delta) + 1;
	else
		rot_pars.Si_delta = ceil(rot_pars.Si_delta);
	rot_pars.Si_delta = fabs(M12 / M22);
	//if (rot_pars.Si_delta != 0) rot_pars.Si_delta += 0.5;
	if (rot_pars.use_Si_delta_margin && rot_pars.Si_delta != 0) rot_pars.Si_delta += 1;
	else if (rot_pars.Si_delta > 1.0) rot_pars.Si_delta += 0.7;
	else if (rot_pars.Si_delta != 0) rot_pars.Si_delta += 0.65; //0.61; //0.45

	double IBufH_delta = rot_pars.M22d - rot_pars.M21d;
	if (IBufH_delta <= 1) //delta <= 1 will not increase IBufH_req
		IBufH_delta = 0;
	else 
		IBufH_delta = ceil(fabs(rot_pars.M22d - rot_pars.M21d)) - 1;

	rot_pars.IBufH_delta = (uint32_t)IBufH_delta;

	if (irt_desc.rot90) {
		irt_desc.image_par[IIMAGE].S = irt_desc.image_par[OIMAGE].H + (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);
	} else {
		irt_desc.image_par[IIMAGE].S = (uint16_t)ceil((double)rot_pars.So8 * rot_pars.affine_Si_factor + 2 * rot_pars.Si_delta) + 2;
		irt_desc.image_par[IIMAGE].S = (uint16_t)ceil(rot_pars.So8 * fabs((M11 * M22 - M12 * M21) / M22) + 2 * rot_pars.Si_delta) + 2;
	}

	rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.So8 * fabs(rot_pars.M21d))) + rot_pars.IBufH_delta;
	//rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.So8 * fabs(M21)));

	if (rot_pars.M21d <= 0) { //embedded clockwise rotation
		rot_pars.Xi_first = (M11 * rot_pars.Xo_last  + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (M21 * rot_pars.Xo_last  + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Xi_last  = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_last  = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;

	} else { //embedded counterclockwise rotation

		rot_pars.Xi_first = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_first = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
		rot_pars.Xi_last  = (M11 * rot_pars.Xo_last  + M12 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Xc / 2;
		rot_pars.Yi_last  = (M21 * rot_pars.Xo_last  + M22 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;
	}

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Xo_first/last  %15.8f / %15.8f *\n", (double)rot_pars.Xo_first, (double)rot_pars.Xo_last);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Yo_first/last  %15.8f / %15.8f *\n", (double)rot_pars.Yo_first, (double)rot_pars.Yo_last);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Xi_first/last  %15.8f / %15.8f *\n", (double)rot_pars.Xi_first, (double)rot_pars.Xi_last);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "* Yi_first/last  %15.8f / %15.8f *\n", (double)rot_pars.Yi_first, (double)rot_pars.Yi_last);

	double Xi_TL = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yi_TL = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
	double Xi_TR = (M11 * rot_pars.Xo_last  + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yi_TR = (M21 * rot_pars.Xo_last  + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
	double Xi_BL = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yi_BL = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;
	double Xi_BR = (M11 * rot_pars.Xo_last  + M12 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Xc / 2;
	double Yi_BR = (M21 * rot_pars.Xo_last  + M22 * rot_pars.Yo_last)  + (double)irt_desc.image_par[IIMAGE].Yc / 2;

	double Y_on_line;
	if (rot_pars.M21d > 0) { //top input corner is from output top-left corner of output, need to detect where top-right output corner is mapped
		//TL-to-BR equation is y = (Yi_TL - Yi_BR) / (Xi_TL - Xi_BR) * x + Yi_BR
		Y_on_line = (Yi_TL - Yi_BR) / (Xi_TL - Xi_BR) * (Xi_TR - Xi_BR) + Yi_BR;
		if ((Xi_TL < Xi_BR && Yi_TR <= Y_on_line) || (Xi_TL > Xi_BR && Yi_TR >= Y_on_line)) // top-right output corner is above the line
			irt_desc.rot_dir = IRT_ROT_DIR_NEG;
		else // // top-right output corner is below the line
			irt_desc.rot_dir = IRT_ROT_DIR_POS;
	} else {//top input corner is from output top-right corner of output, need to detect where top-left output corner is mapped
		//TR-to-BL equation is y = (Yi_TR - Yi_BL) / (Xi_TR - Xi_BL) * x + Yi_BL
		Y_on_line = (Yi_TR - Yi_BL) / (Xi_TR - Xi_BL) * (Xi_TL - Xi_BL) + Yi_BL;
		if ((Xi_TR < Xi_BL && Yi_TL < Y_on_line) || (Xi_TR > Xi_BL && Yi_TL > Y_on_line)) // top-left output corner is above the line
			irt_desc.rot_dir = IRT_ROT_DIR_NEG;
		else // // top-left output corner is below the line
			irt_desc.rot_dir = IRT_ROT_DIR_POS;
	}

#if 1
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "[Xi_TL = %f, Yi_TL = %f], [Xi_TR = %f, Yi_TR = %f]\n", Xi_TL, Yi_TL, Xi_TR, Yi_TR);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "[Xi_BL = %f, Yi_BL = %f], [Xi_BR = %f, Yi_BR = %f]\n", Xi_BL, Yi_BL, Xi_BR, Yi_BR);
	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Y_on_line = %f\n", Y_on_line);
#endif

	if (rot_pars.M22d != 0)
		rot_pars.im_read_slope = rot_pars.M12d / rot_pars.M22d;
	else
		rot_pars.im_read_slope = rot_pars.M11d / rot_pars.M21d;
	//rot_pars.im_read_slope = M12 / M22;

	if (irt_desc.rot_dir) {
		rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first) - irt_desc.image_par[IIMAGE].S;

		if (irt_desc.rot90 == 0)
			rot_pars.Xi_first = rot_pars.Xi_first + rot_pars.Si_delta + 2;
	} else {
		rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first);

		if (irt_desc.rot90 == 0) {
			rot_pars.Xi_first = rot_pars.Xi_first - rot_pars.Si_delta - (rot_pars.im_read_slope == 0 ? 0 : 1);
		} else {
			if (!rot_pars.irt_rotate && rot_pars.affine_flags[e_irt_aff_reflection])
				rot_pars.Xi_first -= 2;
		}
	}

	/*
		because Xoc and Yoc do not change Xi_XX relation, we can indentify relations as:
		Xi_TL = M11 * 0			+ M12 * 0		= 0
		Xi_TR = M11 * (So-1)	+ M12 * 0		= M11 * (So-1)
		Xi_BL = M11 * 0			+ M12 * (Ho-1)	= M12 * (Ho-1)
		Xi_BR = M11 * (So-1)	+ M12 * (Ho-1)	= M11 * (So-1)	+ M12 * (Ho-1)

		if M11 > 0 & M12 > 0 then Xi_TL is min value and Xi_BR is max value and Si = M11 * (So-1) + M12 * (Ho-1)
		if M11 > 0 & M12 < 0 then Xi_BL is min value and Xi_TR is max value and Si = M11 * (So-1) + abs(M12) * (Ho-1)
		if M11 < 0 & M12 > 0 then Xi_TR is min value and Xi_BL is max value and Si = M12 * (Ho-1) + abs(M11) * (So-1)
		if M11 < 0 & M12 < 0 then Xi_BR is min value and Xi_TL is max value and Si = abs(M12) * (Ho-1) + (abs(M11)) * (So-1)
		Then rectangular input stripe Si = abs(M11) * (So-1) + abs(M12) * (Ho-1)

		Then So = (Si - abs(M12) * (Ho-1)) / abs(M11) + 1
	*/

	uint16_t Si = (uint16_t)ceil(fabs(M11) * (irt_desc.image_par[OIMAGE].S - 1) + fabs(M12) * (irt_desc.image_par[OIMAGE].H - 1)) + 3;
	double XiL = fmin(fmin(fmin(Xi_TL, Xi_TR), Xi_BL), Xi_BR);
	double XiR = fmax(fmax(fmax(Xi_TL, Xi_TR), Xi_BL), Xi_BR);

	rot_pars.use_rectangular_input_stripe = 0;
	if (Si < irt_desc.image_par[IIMAGE].S && irt_desc.rot90 == 0) { //we can read less
		rot_pars.use_rectangular_input_stripe = 1;
		irt_desc.image_par[IIMAGE].S = Si;
		rot_pars.im_read_slope = 0;
		if (irt_desc.rot_dir == IRT_ROT_DIR_POS)
			rot_pars.Xi_first = XiR + 2;
		else
			rot_pars.Xi_first = XiL - 1;
	}
}

void irt_projection_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV) || defined (HABANA_SIMULATION)
	float coord_out[IRT_ROT_MAX_PROC_SIZE], YTL, YTR, YBL, YBR;

	XiYi_float_struct iTL_corner, iTR_corner, iBL_corner, iBR_corner;
	XiYi_float_struct top_corner, bot_corner, left_corner, right_corner;
	//uint8_t top_corner_src, bot_corner_src, left_corner_src, right_corner_src;

	//output image top left corner map
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Af, irt_desc.prj_Cf, (float)rot_pars.Xo_first, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	iTL_corner.X = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_first, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	iTL_corner.Y = coord_out[0];

	//output image top right corner map
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Af, irt_desc.prj_Cf, (float)rot_pars.Xo_last, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	iTR_corner.X = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_last, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	iTR_corner.Y = coord_out[0];

	//output image bottom left corner map
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Af, irt_desc.prj_Cf, (float)rot_pars.Xo_first, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	iBL_corner.X = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_first, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	iBL_corner.Y = coord_out[0];

	//output image bottom right corner map
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Af, irt_desc.prj_Cf, (float)rot_pars.Xo_last, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	iBR_corner.X = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_last, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	iBR_corner.Y = coord_out[0];

	//IRT_TRACE_UTILS("Projection corners (%.2f, %.2f), (%.2f, %.2f)\n", (double)iTL_corner.X, (double)iTL_corner.Y, (double)iTR_corner.X, (double)iTR_corner.Y);
	//IRT_TRACE_UTILS("                   (%.2f, %.2f), (%.2f, %.2f)\n", (double)iBL_corner.X, (double)iBL_corner.Y, (double)iBR_corner.X, (double)iBR_corner.Y);

	//input image top corner finding
	if (iTL_corner.Y < iTR_corner.Y) {
		top_corner.Y = iTL_corner.Y; top_corner.X = iTL_corner.X; //top_corner_src = 0;
	} else {
		top_corner.Y = iTR_corner.Y; top_corner.X = iTR_corner.X; //top_corner_src = 1;
	}
	if (iBL_corner.Y < top_corner.Y) {
		top_corner.Y = iBL_corner.Y; top_corner.X = iBL_corner.X; //top_corner_src = 2;
	}
	if (iBR_corner.Y < top_corner.Y) {
		top_corner.Y = iBR_corner.Y; top_corner.X = iBR_corner.X; //top_corner_src = 3;
	}

	//input image bottom corner finding
	if (iTL_corner.Y > iTR_corner.Y) {
		bot_corner.Y = iTL_corner.Y; bot_corner.X = iTL_corner.X; //bot_corner_src = 0;
	} else {
		bot_corner.Y = iTR_corner.Y; bot_corner.X = iTR_corner.X; //bot_corner_src = 1;
	}
	if (iBL_corner.Y > bot_corner.Y) {
		bot_corner.Y = iBL_corner.Y; bot_corner.X = iBL_corner.X; //bot_corner_src = 2;
	}
	if (iBR_corner.Y > bot_corner.Y) {
		bot_corner.Y = iBR_corner.Y; bot_corner.X = iBR_corner.X; //bot_corner_src = 3;
	}

	//input image left corner finding
	if (iTL_corner.X < iTR_corner.X) {
		left_corner.Y = iTL_corner.Y; left_corner.X = iTL_corner.X; //left_corner_src = 0;
	} else {
		left_corner.Y = iTR_corner.Y; left_corner.X = iTR_corner.X; //left_corner_src = 1;
	}
	if (iBL_corner.X < left_corner.X) {
		left_corner.Y = iBL_corner.Y; left_corner.X = iBL_corner.X; //left_corner_src = 2;
	}
	if (iBR_corner.X < left_corner.X) {
		left_corner.Y = iBR_corner.Y; left_corner.X = iBR_corner.X; //left_corner_src = 3;
	}

	//input image right corner finding
	if (iTL_corner.X > iTR_corner.X) {
		right_corner.Y = iTL_corner.Y; right_corner.X = iTL_corner.X; //right_corner_src = 0;
	} else {
		right_corner.Y = iTR_corner.Y; right_corner.X = iTR_corner.X; //right_corner_src = 1;
	}
	if (iBL_corner.X > right_corner.X) {
		right_corner.Y = iBL_corner.Y; right_corner.X = iBL_corner.X; //right_corner_src = 2;
	}
	if (iBR_corner.X > right_corner.X) {
		right_corner.Y = iBR_corner.Y; right_corner.X = iBR_corner.X; //right_corner_src = 3;
	}


	rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.Yi_last - rot_pars.Yi_first)) + 1;
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_first, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	YTL = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_last, (float)rot_pars.Yo_first, 0, 0, 1, 1.0, e_irt_projection);
	YTR = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_first, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	YBL = coord_out[0];
	IRT_top::irt_iicc_float(coord_out, irt_desc.prj_Bf, irt_desc.prj_Df, (float)rot_pars.Xo_last, (float)rot_pars.Yo_last, 0, 0, 1, 1.0, e_irt_projection);
	YBR = coord_out[0];
	rot_pars.IBufH_req = (uint32_t)ceil(std::fmax(fabs(YTR - YTL), fabs(YBR - YBL)));

	rot_pars.Yi_first = (double) top_corner.Y;
	rot_pars.Yi_last  = (double) bot_corner.Y;
	rot_pars.Xi_first = (double) left_corner.X;
	rot_pars.Xi_last  = (double) right_corner.X;

	if (irt_desc.irt_mode == e_irt_projection) {
		if (std::min(iTL_corner.Y, iTR_corner.Y) > std::max(iBL_corner.Y, iBR_corner.Y)) { //top and bottom output image line are swapped in input image
			//irt_desc.read_vflip = 1;
		}
		if (std::min(iTL_corner.X, iBL_corner.X) > std::max(iTR_corner.X, iBR_corner.X)) { //left and right output image line are swapped in input image
			//irt_desc.read_hflip = 1;
		}
	}

	rot_pars.im_read_slope = 0;
	irt_desc.image_par[IIMAGE].S = (int16_t)fabs(ceil(rot_pars.Xi_last) - floor(rot_pars.Xi_first)) + 2;
	rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first);// -irt_desc.image_par[IIMAGE].S;

	//calculating input image coordinates according to actual implemetation
	float yo, xos, xis, xie, yis, yie, xi_slope, yi_slope, xi, yi;
	double Ymax_line;
	double Ymin = 100000, Ymax = -100000;
	rot_pars.Yi_first = 100000;
	rot_pars.Xi_first = 100000;
	rot_pars.Yi_last = -100000;
	rot_pars.Xi_last = -100000;
	rot_pars.IBufH_req = 0;

	FILE *fptr_vals = nullptr, *fptr_pars = nullptr;
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	char fptr_vals_name[50], fptr_pars_name[50];
	sprintf(fptr_vals_name, "Proj_matrix_intr_%d.txt", desc);
	sprintf(fptr_pars_name, "Proj_matrix_intr_pars_%d.txt", desc);
	if (print_out_files) {
		fptr_vals = fopen(fptr_vals_name, "w");
		fptr_pars = fopen(fptr_pars_name, "w");
		IRT_TRACE_TO_RES_UTILS(fptr_pars, "[row, col] = [Xi, Yi]\n");
	}
#endif

	for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {

		Ymin = 100000; Ymax = -100000;

		for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].S; col += irt_desc.proc_size) {

			xos = (float)col - (float)irt_desc.image_par[OIMAGE].Xc / 2;
			yo = (float)row - (float)irt_desc.image_par[OIMAGE].Yc / 2;
			xis = ((irt_desc.prj_Af[0] * xos + irt_desc.prj_Af[1] * yo) + irt_desc.prj_Af[2]) / ((irt_desc.prj_Cf[0] * xos + irt_desc.prj_Cf[1] * yo) + irt_desc.prj_Cf[2]);
			xie = ((irt_desc.prj_Af[0] * xos + irt_desc.prj_Af[1] * yo) + irt_desc.prj_Af[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Af[2]) /
				  ((irt_desc.prj_Cf[0] * xos + irt_desc.prj_Cf[1] * yo) + irt_desc.prj_Cf[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Cf[2]);
			yis = ((irt_desc.prj_Bf[0] * xos + irt_desc.prj_Bf[1] * yo) + irt_desc.prj_Bf[2]) / ((irt_desc.prj_Df[0] * xos + irt_desc.prj_Df[1] * yo) + irt_desc.prj_Df[2]);
			yie = ((irt_desc.prj_Bf[0] * xos + irt_desc.prj_Bf[1] * yo) + irt_desc.prj_Bf[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Bf[2]) /
				  ((irt_desc.prj_Df[0] * xos + irt_desc.prj_Df[1] * yo) + irt_desc.prj_Df[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Df[2]);
			xi_slope = (xie - xis) * (float)(1.0 / (irt_desc.proc_size - 1));
			yi_slope = (yie - yis) * (float)(1.0 / (irt_desc.proc_size - 1));

			for (uint8_t pixel = 0; pixel < irt_desc.proc_size; pixel++) {
				//xo = (float)col + (float)pixel - (float)irt_desc.image_par[OIMAGE].Xc / 2;

				if (irt_desc.proc_size == 1) {
					xi = xis;
					yi = yis;
				} else {
					xi = xis + xi_slope * pixel;
					yi = yis + yi_slope * pixel;
				}

				irt_cfg.mesh_images.proj_image_full[row][col + pixel].x = xi;
				irt_cfg.mesh_images.proj_image_full[row][col + pixel].y = yi;

				irt_map_image_pars_update < mesh_xy_fp64_meta, double > (irt_cfg, irt_desc, irt_cfg.mesh_images.proj_image_full, row, col + pixel, Ymax_line, fptr_vals, fptr_pars);

#if 1
				Ymin = std::fmin(Ymin, yi);
				Ymax = std::fmax(Ymax, yi);

				rot_pars.Yi_first = std::fmin(rot_pars.Yi_first, yi);
				rot_pars.Yi_last  = std::fmax(rot_pars.Yi_last, yi);
				rot_pars.Xi_first = std::fmin(rot_pars.Xi_first, xi);
				rot_pars.Xi_last  = std::fmax(rot_pars.Xi_last, xi);
#endif
			}
		}
		rot_pars.IBufH_req = (uint32_t)std::max((int32_t)rot_pars.IBufH_req, (int32_t)(ceil(Ymax) - floor(Ymin)) + 1);
	}

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
	if (print_out_files) {
		fclose(fptr_vals);
		fclose(fptr_pars);
	}
#endif

	irt_map_iimage_stripe_adj <mesh_xy_fp64_meta> (irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.proj_image_full);
	rot_pars.im_read_slope = 0;
	irt_desc.rot_dir = (rot_pars.irt_angles_adj[e_irt_angle_roll] >= 0) ? IRT_ROT_DIR_POS : IRT_ROT_DIR_NEG;

#endif

}

void irt_mesh_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

#if 0
	rot_pars.Yi_first = (double)irt_cfg.mesh_images.mesh_image_intr[0][0].y;
	rot_pars.Xi_first = (double)irt_cfg.mesh_images.mesh_image_intr[0][0].x;
	rot_pars.Yi_last  = (double)irt_cfg.mesh_images.mesh_image_intr[0][0].y;
	rot_pars.Xi_last  = (double)irt_cfg.mesh_images.mesh_image_intr[0][0].x;

	double Ymin = irt_cfg.mesh_images.mesh_image_intr[0][0].y, Ymax = irt_cfg.mesh_images.mesh_image_intr[0][0].y;
	rot_pars.IBufH_req = 0;

	for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {
		Ymin = irt_cfg.mesh_images.mesh_image_intr[row][0].y;
		Ymax = irt_cfg.mesh_images.mesh_image_intr[row][0].y;
		for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].S; col++) {
			Ymin = std::fmin(Ymin, irt_cfg.mesh_images.mesh_image_intr[row][col].y);
			Ymax = std::fmax(Ymax, irt_cfg.mesh_images.mesh_image_intr[row][col].y);

			rot_pars.Yi_first = std::fmin(rot_pars.Yi_first, (double)irt_cfg.mesh_images.mesh_image_intr[row][col].y);
			rot_pars.Yi_last  = std::fmax(rot_pars.Yi_last,  (double)irt_cfg.mesh_images.mesh_image_intr[row][col].y);
			rot_pars.Xi_first = std::fmin(rot_pars.Xi_first, (double)irt_cfg.mesh_images.mesh_image_intr[row][col].x);
			rot_pars.Xi_last  = std::fmax(rot_pars.Xi_last,  (double)irt_cfg.mesh_images.mesh_image_intr[row][col].x);
		}
		rot_pars.IBufH_req = (uint32_t)std::max((int32_t)rot_pars.IBufH_req, (int32_t)(ceil(Ymax) - floor(Ymin)) + 1);
		//IRT_TRACE_UTILS("Desc %d: irt_mesh_desc_gen row %d, Ymin %f, Ymax %f, BufH %d, IBufH_req %d\n", desc, row, Ymin, Ymax, (int32_t)(ceil(Ymax) - floor(Ymin)), rot_pars.IBufH_req);
	}

	//IRT_TRACE_UTILS("Desc %d: irt_mesh_desc_gen IBufH_req %d\n", desc, rot_pars.IBufH_req);
#endif
#if 0
	rot_pars.IBufH_req = IBufH_req + 1 + 1;//1 because of interpolation, 1 because of Y is released based on previous line;
	if (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic)
		rot_pars.IBufH_req += IRT_RM_DYN_MODE_LINE_RELEASE;  //incase of dynamic mode we discard 8 lines at a time
#endif

	irt_map_iimage_stripe_adj <mesh_xy_fp64_meta> (irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
	rot_pars.im_read_slope = 0;

}

void irt_xi_first_adj_rot(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 59) {
		irt_desc.Xi_start_offset = 0;
		irt_desc.Xi_start_offset_flip = 0;

		if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {//positive rotation angle
//			irt_desc.Xi_start_offset = 0;
		} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 45) {
			//irt_desc.Xi_start_offset = 0; //1
			//rot_pars.Xi_first -= 1;
		} else {
			//irt_desc.Xi_start_offset = 0;//2;
			//rot_pars.Xi_first -= 2;
		}

		if (irt_desc.read_hflip) {
			if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {//positive rotation angle
			//irt_desc.Xi_start_offset_flip = 0;
			} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 26) {
				//irt_desc.Xi_start_offset_flip = 0;//1;
				//rot_pars.Xi_first -= 1;
			} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 56) {
				//irt_desc.Xi_start_offset_flip = 0;//2;//3;
				//rot_pars.Xi_first -= 2;
			} else {
				//irt_desc.Xi_start_offset_flip = 0;//3;//3;
				//rot_pars.Xi_first -= 3;
			}
		}
	} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 80) {
		rot_pars.Xi_first -= 0;//7;//(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)));
		irt_desc.Xi_start_offset = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 6;//0;
		irt_desc.Xi_start_offset_flip = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 3;//3;//5;
	} else {

		if (irt_desc.rot90 == 0) {
			//rot_pars.Xi_first -= /*60 +*/ ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)));
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Xi_start_adj: decreasing Xi_first by %d for adjusted rotation angle %f\n", (int)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]))), rot_pars.irt_angles_adj[e_irt_angle_rot]);
		}
		irt_desc.Xi_start_offset = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 60;
		if (irt_desc.rot_dir == IRT_ROT_DIR_NEG && irt_desc.rot90 == 1) {
			//irt_desc.Xi_start_offset = 2;
		}

		if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {
			if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 85) {
				irt_desc.Xi_start_offset_flip = 5;//70;//4;//55;
			} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 89) {
				irt_desc.Xi_start_offset_flip = 7;//70;//4;//55;
			} else {
				irt_desc.Xi_start_offset_flip = 58;
			}
		} else {
			if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 85) {
				irt_desc.Xi_start_offset_flip = -47;//78;//4;//55;
			} else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 89) {
				irt_desc.Xi_start_offset_flip = -36;
			} else {
				irt_desc.Xi_start_offset_flip = 56;
			}
		}
	}

	irt_desc.Xi_start_offset = 0;
	irt_desc.Xi_start_offset_flip = 0;
	if (irt_desc.rot90) {
		if (/*irt_desc.rot90_intv ||*/ irt_desc.rot90_inth) { //interpolation will be required, rot90 will be reset for model
			if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {
				rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
			} else {
				rot_pars.Xi_first -= irt_desc.image_par[IIMAGE].S - 1;
			}
		} else { //interpolation is not required, working regulary as rot90
			if (irt_desc.rot_dir == IRT_ROT_DIR_NEG) {
				//irt_desc.Xi_start_offset = 2;
				rot_pars.Xi_first -= 2;
			}
		}
	}
}

void irt_xi_first_adj_aff(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	bool rot90 = irt_desc.rot90 == 1 && rot_pars.affine_flags[e_irt_aff_rotation] == 1; //affine has rotation of +/- 90
	if (rot90 == 0) {
		if (rot_pars.affine_flags[e_irt_aff_shearing] || rot_pars.affine_flags[e_irt_aff_reflection] || rot_pars.affine_flags[e_irt_aff_scaling]) //(!strcmp(rot_pars.affine_mode, "T"))
			rot_pars.Xi_first -= 0;
		else if (rot_pars.affine_flags[e_irt_aff_rotation] == 0) { //strcmp(rot_pars.affine_mode, "R")
			rot_pars.Xi_first -= 5;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Decreasing Xi_start by 5 in affine mode\n");
		} else
			irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
	} else { //affine has rotation of +/- 90
		if (rot_pars.affine_flags[e_irt_aff_shearing] || rot_pars.affine_flags[e_irt_aff_reflection])
			rot_pars.Xi_first -= 0;
		else if (rot_pars.affine_flags[e_irt_aff_scaling] == 0) //rot90 w/o scaling
			irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
		else { //rot90 w/ scaling
			if (rot_pars.Sy == 1)
				irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
			else
				rot_pars.Xi_first -= 0;
		}
	}
}

void irt_descriptor_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint32_t Hsi, uint8_t desc) {

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Task %d: Generating descriptor for %s mode\n", desc, irt_irt_mode_s[irt_desc.irt_mode]);
	irt_desc.Ho = irt_desc.image_par[OIMAGE].H;
	irt_desc.Wo = irt_desc.image_par[OIMAGE].W;

	if (irt_desc.image_par[OIMAGE].W > OIMAGE_W || irt_desc.image_par[OIMAGE].H > OIMAGE_H) {
		if (rot_pars.oimg_auto_adj == 0) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen error: output image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
				irt_desc.image_par[OIMAGE].W, irt_desc.image_par[OIMAGE].H, OIMAGE_W, OIMAGE_H);
			IRT_TRACE_TO_RES_UTILS(test_res, "was not run, output image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
				irt_desc.image_par[OIMAGE].W, irt_desc.image_par[OIMAGE].H, OIMAGE_W, OIMAGE_H);
			IRT_CLOSE_FAILED_TEST(0);
		} else {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc %d gen: adjusting output image %dx%d to maximum supported size", desc, irt_desc.image_par[OIMAGE].W, irt_desc.image_par[OIMAGE].H);
			irt_desc.image_par[OIMAGE].W = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].W, OIMAGE_W);
			irt_desc.image_par[OIMAGE].H = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].H, OIMAGE_H);
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, " %dx%d\n", irt_desc.image_par[OIMAGE].W, irt_desc.image_par[OIMAGE].H);
		}
	}

	irt_params_analize(irt_cfg, rot_pars, irt_desc, desc);

	irt_rot_angle_adj_calc(irt_cfg, rot_pars, irt_desc, desc);

	rot_pars.WEIGHT_PREC = rot_pars.MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;// +1 because of 0.5 error, +2 because of interpolation, -1 because of weigths multiplication
	rot_pars.COORD_PREC = rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT; // +1 because of polynom in Xi/Yi calculation
	rot_pars.COORD_ROUND = (1 << (rot_pars.COORD_PREC - 1));
	rot_pars.TOTAL_PREC = rot_pars.COORD_PREC + rot_pars.WEIGHT_PREC;
	rot_pars.TOTAL_ROUND = (1 << (rot_pars.TOTAL_PREC - 1));
	rot_pars.PROJ_NOM_PREC = rot_pars.TOTAL_PREC;
	rot_pars.PROJ_DEN_PREC = (rot_pars.TOTAL_PREC + 10);
	rot_pars.PROJ_NOM_ROUND = (1 << (rot_pars.PROJ_NOM_PREC - 1));
	rot_pars.PROJ_DEN_ROUND = (1 << (rot_pars.PROJ_DEN_PREC - 1));
	irt_desc.prec_align = 31 - rot_pars.TOTAL_PREC;

	irt_desc.image_par[OIMAGE].Ps = rot_pars.Pwo <= 8 ? 0 : 1;
	irt_desc.image_par[IIMAGE].Ps = rot_pars.Pwi <= 8 ? 0 : 1;
	irt_desc.image_par[MIMAGE].Ps = irt_desc.mesh_format == 0 ? 2 : 3;
	irt_desc.Msi = ((1 << rot_pars.Pwi) - 1) << rot_pars.Ppi;
	irt_desc.bli_shift = rot_pars.Pwo - (rot_pars.Pwi + rot_pars.Ppi);
	irt_desc.MAX_VALo = (1 << rot_pars.Pwo) - 1; //output pixel max value, equal to 2^pixel_width - 1

	//precision overflow prevension checking
	if (rot_pars.Pwi + (rot_pars.TOTAL_PREC - rot_pars.WEIGHT_SHIFT) * 2 + 2 > 66) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen: Input pixel width %d, coordinate width %d, max pixel width %d and weight shift %d are not supported: exceeds 64 bits calculation in BLI\n",
			rot_pars.Pwi, rot_pars.MAX_COORD_WIDTH, rot_pars.MAX_PIXEL_WIDTH, rot_pars.WEIGHT_SHIFT);
		IRT_TRACE_TO_RES_UTILS(test_res, "Input pixel width %d, coordinate width %d, max pixel width %d and weight shift %d are not supported: exceeds 64 bits calculation in BLI\n",
			rot_pars.Pwi, rot_pars.MAX_COORD_WIDTH, rot_pars.MAX_PIXEL_WIDTH, rot_pars.WEIGHT_SHIFT);
		IRT_CLOSE_FAILED_TEST(0);
	}

#if 0
	rot_pars.irt_angles[e_irt_angle_rot] = rot_pars.irt_angles_adj[e_irt_angle_rot];
	irt_desc.rot_dir = (rot_pars.irt_angles[e_irt_angle_rot] >= 0) ? IRT_ROT_DIR_POS : IRT_ROT_DIR_NEG;
	if (rot_pars.irt_angles[e_irt_angle_rot] == (double)90.0 || rot_pars.irt_angles[e_irt_angle_rot] == (double)-90.0) {
		irt_desc.rot90 = 1;
		rot_pars.im_read_slope = 0;
	} else {
		irt_desc.rot90 = 0;
	}
#endif

	irt_desc.image_par[OIMAGE].S = irt_desc.image_par[OIMAGE].W;

#if 0
	rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
	if (irt_h5_mode) {
		if ((irt_desc.image_par[OIMAGE].S % 8) != 0) {//not multiple of 8, round up to multiple of 8
			rot_pars.So8 += (8 - (irt_desc.image_par[OIMAGE].S % 8));
		}
	}

	rot_pars.Xo_first = -(double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Xo_last  =  (double)rot_pars.So8 - 1 - (double)irt_desc.image_par[OIMAGE].Xc / 2;
	rot_pars.Yo_first = -(double)irt_desc.image_par[OIMAGE].Yc / 2;
	rot_pars.Yo_last  =  (double)irt_desc.image_par[OIMAGE].H - 1 - (double)irt_desc.image_par[OIMAGE].Yc / 2;
#endif

	irt_oimage_corners_calc(irt_cfg, rot_pars, irt_desc, desc);

	rot_pars.cosd = cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0);
	if (irt_desc.rot90) {//fix C++ bug that cos(90) is not zero
		rot_pars.cosd = 0;
	}
	rot_pars.sind = sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0);
	irt_affine_coefs_calc(irt_cfg, rot_pars, irt_desc, desc);
	irt_projection_coefs_calc(irt_cfg, rot_pars, irt_desc, desc);

	if (irt_desc.irt_mode == e_irt_mesh) {
		irt_mesh_matrix_calc(irt_cfg, rot_pars, irt_desc, desc, 1);
	}

	//precision overflow prevension checking
	//affine coefficient in fixed point is presented as SI.F of 32 bits.
	//F width = 32 - 1 - I_width
	double M_max = std::max((double)fabs(rot_pars.M11d), fabs(rot_pars.M12d));
	M_max = std::max((double)M_max, fabs(rot_pars.M21d));
	M_max = std::max((double)M_max, fabs(rot_pars.M22d)); //maximum value of affine coefficients
	int M_max_int = (int)floor(M_max); //integer part of max value
	uint8_t M_I_width = (uint8_t)ceil(log2(M_max_int+1));
	uint8_t M_F_width = 31 - M_I_width; //remained bits for fraction

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc gen: Affine matrix max coef %f (%d), M_I_width %d, M_F_width %d\n", (double)M_max, M_max_int, M_I_width, M_F_width);
	if (M_F_width < rot_pars.TOTAL_PREC) { //TOTAL selected precision > bits allowed for fraction presentation of M
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen: Coordinate width %d, max pixel width %d and total precision %d are not supported: affine matrix coefficients exceed 32 bits by %d bits\n",
			rot_pars.MAX_COORD_WIDTH, rot_pars.MAX_PIXEL_WIDTH, rot_pars.TOTAL_PREC, rot_pars.TOTAL_PREC - M_F_width);
		if (rot_pars.rot_prec_auto_adj == 0) {
			IRT_TRACE_TO_RES_UTILS(test_res, "Coordinate width %d, max pixel width %d and total precision %d are not supported: affine matrix coefficients exceed 32 bits by %d bits\n",
				rot_pars.MAX_COORD_WIDTH, rot_pars.MAX_PIXEL_WIDTH, rot_pars.TOTAL_PREC, rot_pars.TOTAL_PREC - M_F_width);
			IRT_CLOSE_FAILED_TEST(0);
		} else {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Adjusting precision to %d bits\n", M_F_width);
			rot_pars.TOTAL_PREC = M_F_width;
			rot_pars.TOTAL_ROUND = (1 << (rot_pars.TOTAL_PREC - 1));
		}
	}
	rot_pars.bli_shift_fix = (2 * rot_pars.TOTAL_PREC) - (rot_pars.Pwo - (rot_pars.Pwi + rot_pars.Ppi)) - 1; //bi-linear interpolation shift
	irt_desc.prec_align = 31 - rot_pars.TOTAL_PREC;

	rot_pars.cos16 = (int)rint(rot_pars.cosd * pow(2.0, 16));
	rot_pars.sin16 = (int)rint(rot_pars.sind * pow(2.0, 16));
	irt_desc.cosf = (float)rot_pars.cosd;
	irt_desc.sinf = (float)rot_pars.sind;
	irt_desc.cosi = (int)rint(rot_pars.cosd * pow(2.0, rot_pars.TOTAL_PREC));
	irt_desc.sini = (int)rint(rot_pars.sind * pow(2.0, rot_pars.TOTAL_PREC));

	//used for HL model w/o flip (with rot_angle and not rot_angle_adj)
	rot_pars.cosd_hl = irt_desc.rot90 ? 0 : cos(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180.0); //fix C++ bug that cos(90) is not zero
	rot_pars.sind_hl = sin(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180.0);
	rot_pars.cosf_hl = (float)rot_pars.cosd_hl;
	rot_pars.sinf_hl = (float)rot_pars.sind_hl;
	rot_pars.cosi_hl = (int)rint(rot_pars.cosd_hl * pow(2.0, rot_pars.TOTAL_PREC));
	rot_pars.sini_hl = (int)rint(rot_pars.sind_hl * pow(2.0, rot_pars.TOTAL_PREC));

	irt_desc.M11f = (float)rot_pars.M11d;
	irt_desc.M12f = (float)rot_pars.M12d;
	irt_desc.M21f = (float)rot_pars.M21d;
	irt_desc.M22f = (float)rot_pars.M22d;
	irt_desc.M11i = (int)rint(rot_pars.M11d * pow(2.0, rot_pars.TOTAL_PREC));
	irt_desc.M12i = (int)rint(rot_pars.M12d * pow(2.0, rot_pars.TOTAL_PREC));
	irt_desc.M21i = (int)rint(rot_pars.M21d * pow(2.0, rot_pars.TOTAL_PREC));
	irt_desc.M22i = (int)rint(rot_pars.M22d * pow(2.0, rot_pars.TOTAL_PREC));
	for (uint8_t i = 0; i < 3; i++) {
		irt_desc.prj_Af[i] = (float)(rot_pars.prj_Ad[i]);
		irt_desc.prj_Bf[i] = (float)(rot_pars.prj_Bd[i]);
		irt_desc.prj_Cf[i] = (float)(rot_pars.prj_Cd[i]);
		irt_desc.prj_Df[i] = (float)(rot_pars.prj_Dd[i]);
		rot_pars.prj_Ai[i] = (int64_t)rint(rot_pars.prj_Ad[i] * pow(2.0, rot_pars.PROJ_NOM_PREC));
		rot_pars.prj_Bi[i] = (int64_t)rint(rot_pars.prj_Bd[i] * pow(2.0, rot_pars.PROJ_NOM_PREC));
		rot_pars.prj_Ci[i] = (int64_t)rint(rot_pars.prj_Cd[i] * pow(2.0, rot_pars.PROJ_DEN_PREC));
		rot_pars.prj_Di[i] = (int64_t)rint(rot_pars.prj_Dd[i] * pow(2.0, rot_pars.PROJ_DEN_PREC));
	}

	//calculation input image read stripe parameters
	switch (irt_desc.irt_mode) {
	case e_irt_rotation: irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
	case e_irt_affine: irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
	case e_irt_projection:
		switch (rot_pars.proj_mode) {
		case e_irt_rotation: irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_affine: irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_projection: irt_projection_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		default: break;
		}
		break;
	case e_irt_mesh:
		switch (rot_pars.mesh_mode) {
		case e_irt_rotation: irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_affine: irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_projection: 
			switch (rot_pars.proj_mode) {
			case e_irt_rotation: irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
			case e_irt_affine: irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
			case e_irt_projection: irt_projection_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
			default: break;
			}
			break;
		case e_irt_mesh: irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
		}
		if ((rot_pars.mesh_mode != e_irt_mesh) && (rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)) {
			irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
		}
		//irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
		break;
	}

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Desc %d: input stripe width Si = %d, sliding window height = %u\n", desc, irt_desc.image_par[IIMAGE].S, rot_pars.IBufH_req);

	//calculating processing size and offsets

	if (rot_pars.proc_auto && irt_desc.rate_mode == e_irt_rate_fixed) { //fixed rate mode set by proc_auto for projection/mesh does not work in all cases
		if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) ||
			(irt_desc.irt_mode == e_irt_mesh && (rot_pars.mesh_mode == e_irt_projection || rot_pars.mesh_mode == e_irt_mesh || rot_pars.mesh_dist_r0_Sh1_Sv1 == 0))) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: proc_auto = 1 is not supported in Mesh mode with provided parameters\n");
			IRT_TRACE_TO_RES_UTILS(test_res, "Error: proc_auto = 1 is not supported in Mesh mode with provided parameters\n");
			IRT_CLOSE_FAILED_TEST(0);
		}
		irt_desc.proc_size = irt_proc_size_calc(irt_cfg, rot_pars, irt_desc, desc);
	}

	switch (irt_desc.irt_mode) {
	case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
	case e_irt_affine:   irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
	case e_irt_projection:
		switch (rot_pars.proj_mode) {
		case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_affine:	 irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
		default: break;
		}
		break;
	case e_irt_mesh:
		switch (rot_pars.mesh_mode) {
		case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_affine:	 irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
		case e_irt_projection:
			switch (rot_pars.proj_mode) {
			case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
			case e_irt_affine:	 irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
			default: break;
			}
			break;
		default: break;
		}
		break;
	}

	//calculating required rotation memory view
	//IBufW_req is pixels per input lines stored in memory, equal to Si round up to multiple of 128
	//IBufW_req = (int)(pow(2.0, ceil(log(((double)irt_desc.image_par[IIMAGE].S + 1) / 256) / log(2.0))) * 256);
	//IBufW_req = (int)(ceil(((float)irt_desc.image_par[IIMAGE].S + 1) / 128) * 128);
	rot_pars.IBufW_req = (uint32_t)(ceil(((float)irt_desc.image_par[IIMAGE].S + (irt_desc.rot90 == 0 ? 1 + irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip : 0)) / 128) * 128);
	//BufW_entries - entries per input image line
	rot_pars.IBufW_entries_req = rot_pars.IBufW_req / 16;
	//2.0*IRT_ROT_MEM_HEIGHT1 = 256 - entries in bank
	//irt_desc.BufW_entries - entries per input image line
	//2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries - # of input image lines in single bank
	//2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8 - # of input image lines in all banks
	//log2(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8) - bits to present # of input image lines in all banks
	//floor(log(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8)/log(2.0)) - rounding down to integer value
	//pow(2.0,floor(log(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8)/log(2.0))) - makes BufH to be power of 2 to simplify buffer management
	//BufH_available = (int)(floor(2.0 * IRT_ROT_MEM_HEIGHT1 / irt_top->irt_cfg.BufW_entries) * 8);
	//BufH_available - numbe_fitr of input lines that can be stored in rotation memory if we store IBufW_req pixels per line
	//uint16_t BufH_available = (uint16_t)(pow(2.0, floor(log(2.0 * IRT_ROT_MEM_BANK_HEIGHT / rot_pars.IBufW_entries_req * 8))));

	rot_pars.IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, rot_pars.IBufH_req);

	irt_rotation_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);

	if (irt_desc.irt_mode == e_irt_mesh) {

		uint32_t Wo = irt_desc.image_par[OIMAGE].W;
		uint32_t So = irt_desc.image_par[OIMAGE].S;
		uint32_t Ho = irt_desc.image_par[OIMAGE].H;

		irt_mesh_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);
		if (Wo != irt_desc.image_par[OIMAGE].W || So != irt_desc.image_par[OIMAGE].S || Ho != irt_desc.image_par[OIMAGE].H) {
			switch (rot_pars.mesh_mode) {
			case e_irt_rotation:
				irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
				irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
				break;
			case e_irt_affine:
				irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
				irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc);
				break;
			case e_irt_projection:
				switch (rot_pars.proj_mode) {
				case e_irt_rotation: 
					irt_rotation_desc_gen(irt_cfg, rot_pars, irt_desc, desc); 
					irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); 
					break;
				case e_irt_affine: 
					irt_affine_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
					irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc);
					break;
				case e_irt_projection: 
					irt_projection_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
				default: break;
				}
				break;
			case e_irt_mesh: irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
			}
			if ((rot_pars.mesh_mode != e_irt_mesh) && (rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)) {
				irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
			}

			//irt_rotation_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc); //used to update buf mode after previous irt_rotation_memory_fit_check adjusted Wo
		}
	}

	if (irt_desc.rot90) {
		//rot_pars.im_read_slope = 0;
		if (irt_desc.rot90_inth || irt_desc.rot90_intv) { //work as regular rotation because of interpolation
			//irt_desc.rot90 = 0;
			if (/*rot_pars.rot90_intv &&*/ irt_desc.rate_mode == e_irt_rate_fixed && irt_desc.proc_size == 8 && irt_desc.rot90_intv) { //vertical interpolation is required
				irt_desc.proc_size = 7;
			}
		}
	}

	if (irt_desc.rot90 && (!rot_pars.irt_rotate && (rot_pars.irt_affine_st_inth || rot_pars.irt_affine_st_intv))) {
		irt_desc.rot90 = 0;
		if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { //positive rotation angle
			rot_pars.Xi_first += (double)irt_desc.image_par[IIMAGE].S;
		} else {
			rot_pars.Xi_first -= ((double)irt_desc.image_par[IIMAGE].S - 3);
		}
		if (irt_desc.rate_mode == e_irt_rate_fixed && irt_desc.proc_size == 8) { //vertical interpolation is required
			irt_desc.proc_size = 7;
		}
	}

	if ((fabs(rot_pars.Xi_first) > (double)IRT_XiYi_first_max) || (fabs(rot_pars.Xi_last) > (double)IRT_XiYi_first_max) ||
		(fabs(rot_pars.Yi_first) > (double)IRT_XiYi_first_max) || (fabs(rot_pars.Yi_last) > (double)IRT_XiYi_first_max)) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen error: transformation parameters are not supported, input coordinates range exceeds [-%d:%d]: Xi_first = %f, Xi_last = %f, Yi_first = %f, Yi_last = %f\n",
			IRT_XiYi_first_max, IRT_XiYi_first_max, rot_pars.Xi_first, rot_pars.Xi_last, rot_pars.Yi_first, rot_pars.Yi_last);
		IRT_TRACE_TO_RES_UTILS(test_res, "was not run, input coordinate range exceeds [-%d:%d]: Xi_first = %f, Xi_last = %f, Yi_first = %f, Yi_last = %f\n",
			IRT_XiYi_first_max, IRT_XiYi_first_max, rot_pars.Xi_first, rot_pars.Xi_last, rot_pars.Yi_first, rot_pars.Yi_last);
		IRT_CLOSE_FAILED_TEST(0);
	}

	//irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
	//rot_pars.IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, rot_pars.IBufH_req);

#ifdef IRT_USE_FLIP_FOR_MINUS1
	irt_desc.Xi_first_fixed = (int64_t)floor((rot_pars.Xi_first - irt_desc.read_hflip) * pow(2.0, IRT_SLOPE_PREC));
	irt_desc.Yi_first_fixed = (int64_t)rint((rot_pars.Yi_first - irt_desc.read_vflip) * pow(2.0, IRT_SLOPE_PREC));
#else
	irt_desc.Xi_first_fixed = (int64_t)floor((rot_pars.Xi_first) * pow(2.0, IRT_SLOPE_PREC));
	irt_desc.Yi_first_fixed = (int64_t)rint((rot_pars.Yi_first) * pow(2.0, IRT_SLOPE_PREC));
#endif

	//irt_desc.Yi_start = (int16_t)(irt_desc.Yi_first_fixed >> IRT_SLOPE_PREC) - irt_desc.read_vflip;
#ifdef IRT_USE_FLIP_FOR_MINUS1
	irt_desc.Yi_start = (int16_t)floor(rot_pars.Yi_first) - irt_desc.read_vflip;
#else
	irt_desc.Yi_start = (int16_t)floor(rot_pars.Yi_first);
#endif

	irt_desc.Yi_end   = (int16_t)ceil(rot_pars.Yi_last) + 1;
	if (irt_desc.bg_mode == e_irt_bg_frame_repeat) {
		//irt_desc.Yi_start = IRT_top::IRT_UTILS::irt_max_int16(irt_desc.Yi_start, 0);
		//irt_desc.Yi_start = IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_start, (int16_t)irt_desc.image_par[IIMAGE].H - 1);
		//irt_desc.Yi_end = IRT_top::IRT_UTILS::irt_max_int16(irt_desc.Yi_end, 1);
		//irt_desc.Yi_end = IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_end, (int16_t)irt_desc.image_par[IIMAGE].H - 1);
	}
	//if (irt_desc.bg_mode == e_irt_bg_frame_repeat)
	//	irt_desc.Yi_end = IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_end, (int16_t)irt_desc.image_par[IIMAGE].H - 1);

	//irt_desc.im_read_slope = (int)floor(rot_pars.im_read_slope * pow(2.0, rot_pars.COORD_PREC) + 0.5 * 1);
	irt_desc.im_read_slope = (int)rint(rot_pars.im_read_slope * pow(2.0, IRT_SLOPE_PREC));

	if (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic) { //stripe height is multiple of 8
		irt_desc.Yi_end = irt_desc.Yi_start + (uint16_t)(ceil(((double)irt_desc.Yi_end - irt_desc.Yi_start + 1) / IRT_RM_DYN_MODE_LINE_RELEASE) * IRT_RM_DYN_MODE_LINE_RELEASE) - 1;
	}

	irt_desc.image_par[IIMAGE].Hs = Hsi;//irt_desc.image_par[IIMAGE].W;
	irt_desc.image_par[OIMAGE].Size = irt_desc.image_par[OIMAGE].S * irt_desc.image_par[OIMAGE].H;
	irt_desc.image_par[OIMAGE].Hs = irt_desc.image_par[OIMAGE].S << irt_desc.image_par[OIMAGE].Ps;
	irt_desc.image_par[MIMAGE].Hs = irt_desc.image_par[MIMAGE].W << irt_desc.image_par[MIMAGE].Ps;

	if (irt_h5_mode || irt_desc.rot90 == 1) {
		//irt_desc.proc_size = 8;
	}

	//checking Xi_start overflow
	rot_pars.Xi_start = IRT_top::xi_start_calc(irt_desc, irt_desc.Yi_end, e_irt_xi_start_calc_caller_desc_gen, desc);
	rot_pars.Xi_start = IRT_top::xi_start_calc(irt_desc, irt_desc.Yi_start, e_irt_xi_start_calc_caller_desc_gen, desc);

}

uint16_t irt_out_stripe_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {

	double irt_angle_tmp[e_irt_angle_shr_y + 1];
	for (uint8_t angle_type = e_irt_angle_roll; angle_type <= e_irt_angle_shr_y; angle_type++) {
		switch (angle_type) {
//		case e_irt_angle_roll:
//		case e_irt_angle_pitch:
//		case e_irt_angle_yaw:
		case e_irt_angle_rot:
//		case e_irt_angle_shr_x:
//		case e_irt_angle_shr_y:
			if (rot_pars.irt_angles[angle_type] > 360 || rot_pars.irt_angles[angle_type] < -360) {
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "%s angle %f is not supported, provide angle in [-360:360] range\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
				IRT_CLOSE_FAILED_TEST(0);
			}
			if (rot_pars.irt_angles[angle_type] < 0)
				irt_angle_tmp[angle_type] = rot_pars.irt_angles[angle_type] + 360; //converting angles to positive range
			else
				irt_angle_tmp[angle_type] = rot_pars.irt_angles[angle_type];
			if (irt_angle_tmp[angle_type] >= 270) irt_angle_tmp[angle_type] -= 360;//converting [270:360] to [-90:0]
			break;
		}
	}

	if (irt_desc.irt_mode == e_irt_rotation) {
		if (-MAX_ROT_ANGLE <= irt_angle_tmp[e_irt_angle_rot] && irt_angle_tmp[e_irt_angle_rot] <= MAX_ROT_ANGLE) {
			irt_desc.read_vflip = 0;
			irt_desc.read_hflip = 0;
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
		} else if ((180.0 - MAX_ROT_ANGLE) <= irt_angle_tmp[e_irt_angle_rot] && irt_angle_tmp[e_irt_angle_rot] <= (180.0 + MAX_ROT_ANGLE)) {
			irt_desc.read_hflip = 1;
			irt_desc.read_vflip = 1;
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot] - 180.0;
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Converting rotation angle to %f with pre-rotation of 180 degree\n", rot_pars.irt_angles_adj[e_irt_angle_rot]);
		} else if (irt_angle_tmp[e_irt_angle_rot] == 90 || irt_angle_tmp[e_irt_angle_rot] == -90.0) {
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
			irt_desc.rot90 = 1;
			if ((irt_desc.image_par[OIMAGE].Xc & 1) ^ (irt_desc.image_par[IIMAGE].Yc & 1)) { //interpolation over input lines is required
				irt_desc.rot90_intv = 1;
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Rotation 90 degree will require input lines interpolation\n");
			}
			if ((irt_desc.image_par[OIMAGE].Yc & 1) ^ (irt_desc.image_par[IIMAGE].Xc & 1)) { //interpolation input pixels is required
				irt_desc.rot90_inth = 1;
				IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Rotation 90 degree will require input pixels interpolation\n");
			}
		} else {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
			IRT_CLOSE_FAILED_TEST(0);
		}
	}

	/* 
	Must be set before call this function
	irt_desc.image_par[OIMAGE].H
	irt_desc.image_par[OIMAGE].S
	irt_desc.image_par[IIMAGE].Ps
	rot_pars.irt_angles_adj[e_irt_angle_rot]
	irt_cfg.buf_format[e_irt_block_rot]
	irt_desc.irt_mode
	*/

	uint16_t Si_log2;
	uint32_t Si_pxls;
	int32_t Si_adj, So_from_IBufW = 0, So_from_IBufH = 0, Si;

	uint32_t BufW = 256;
	uint32_t BufH = 128;
	bool IBufW_fit = 0;
	bool IBufH_fit = 0;
	bool fit_width = 0, fit_height = 0;
	uint16_t Ho = irt_desc.image_par[OIMAGE].H;
	//uint16_t Wo = irt_desc.image_par[OIMAGE].W;
	irt_desc.image_par[OIMAGE].S = irt_desc.image_par[OIMAGE].W;
	uint16_t So = irt_desc.image_par[OIMAGE].S;
	uint16_t Hm = irt_desc.image_par[MIMAGE].H;
	//uint16_t Wo = irt_desc.image_par[OIMAGE].W;
	uint16_t Sm = irt_desc.image_par[MIMAGE].S;

	IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output stripe width calculation to fit rotation memory\n");

	double Si_delta = tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0);

	if (Si_delta > 50.0) Si_delta += 3;
	else if (Si_delta != 0) Si_delta += 1;

	Si_pxls = BufW >> irt_desc.image_par[IIMAGE].Ps;
	Si_log2 = (uint16_t)floor(log2((double)Si_pxls));
	Si_adj = (int32_t)pow(2.0, Si_log2);
	if (irt_desc.rot90 == 0) {
		Si_adj -= (int32_t)ceil(2 * Si_delta) + 2;
	}
	Si = Si_adj - 1 + irt_desc.rot90 - (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);

	rot_pars.irt_rotate = irt_desc.irt_mode == e_irt_rotation;
	bool irt_do_rotate = rot_pars.irt_rotate;

	if (irt_do_rotate) {
		if (irt_desc.rot90) {
			irt_desc.image_par[OIMAGE].H = (uint16_t)std::min(Si_pxls, (uint32_t)irt_desc.image_par[OIMAGE].H);
			So_from_IBufW = (int32_t)irt_desc.image_par[OIMAGE].S;
			So_from_IBufH = BufH - (/*irt_desc.rot90_inth |*/ irt_desc.rot90_intv);
		} else {
			if (rot_pars.use_rectangular_input_stripe == 0)
				So_from_IBufW = (int32_t)floor(((double)Si - 1) * cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0));
			else
				So_from_IBufW = (int32_t)floor(((double)Si + ceil(2 * Si_delta) - 1 - fabs(sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)) * (irt_desc.image_par[OIMAGE].H - 1)) / cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0) - 1);
			So_from_IBufH = (int32_t)floor(((double)BufH - 2 - (irt_cfg.buf_format[e_irt_block_rot] ? IRT_RM_DYN_MODE_LINE_RELEASE : 1)) / sin(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
		}
		if (So_from_IBufW > 0 && So_from_IBufH > 0) {
			fit_width = 1; fit_height = 1; IBufW_fit = 1; IBufH_fit = 1;
		}
	} 

	irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufW);
	irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufH);

	if (fit_width && fit_height && IBufW_fit && IBufH_fit && irt_desc.image_par[OIMAGE].S > 0 && irt_desc.image_par[OIMAGE].H > 0) {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image stripe calculation: So from BufW is %d, So from BufH is %d, selected So is %u\n", So_from_IBufW, So_from_IBufH, irt_desc.image_par[OIMAGE].S);
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "Output image stripe calculation: output image size is reduced from %ux%u to %ux%u in %s mode\n",
			So, Ho, irt_desc.image_par[OIMAGE].S, irt_desc.image_par[OIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
	} else {
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Output image stripe calculation cannot find output stripe resolution\n");
		IRT_CLOSE_FAILED_TEST(0);
	}

	return irt_desc.image_par[OIMAGE].S;
}