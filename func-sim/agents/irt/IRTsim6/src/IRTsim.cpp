//============================================================================
// Name        : IRTsim.cpp
// Author      : Evgeny Spektor
// Version     :
// Copyright   :
// Description : Hello World in C++, Ansi-style
//============================================================================

#define _USE_MATH_DEFINES

#include "stdio.h"
#include <iostream>
#include <algorithm>
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include <cmath>
//#include "IRTsim.h"
#include "IRT.h"
#include "IRTutils.h"
#include "time.h"
using namespace std;

FILE* test_res;
#define IRT_DESC_STR_LENGTH 1000

IRT_top *IRT_top0, *IRT_top1;

uint8_t* ext_mem;
uint16_t*** input_image;
uint16_t**** output_image;

extern void ext_memory_wrapper(uint64_t addr, bool read, bool write, bus128B_struct wr_data, bus128B_struct& rd_data, meta_data_struct meta_in, meta_data_struct& meta_out, int lsb_pad, int msb_pad);
extern void IRThl(IRT_top* irt_top, uint8_t image, uint8_t desc, uint16_t i_width, uint16_t i_height, char* input_file, char* outfilename, char* outfiledir);

extern int cycle;
extern uint8_t hl_only;
extern Eirt_bmp_order irt_bmp_rd_order, irt_bmp_wr_order;

extern FILE* log_file;
extern bool test_failed;
extern bool irt_multi_image;
extern bool print_out_files;
extern bool test_file_flag;
extern uint8_t num_of_images;

int main (int argc, char *argv[]){

	char input_file[150] = { '\0' }, input_file_name[50] = { '\0' }, input_file_name_no_ext[50] = { '\0' }, input_file_dir[100] = { "" }, test_file_name[50] = { '\0' };
	char output_file[150] = { '\0' }, output_file_name[50] = { '\0' }, output_file_name_no_ext[50] = { '\0' }, output_file_dir[100] = { "" };

	num_of_images = 0;
	uint64_t timeout_time = 0;
	int n;
	bool done = 0;

	time_t my_time = time(nullptr);
	struct tm* ti;

	ti = localtime(&my_time);
	printf("%s", asctime(ti));
#if 0
	if ((ti->tm_hour > 22) || (ti->tm_hour < 10)) {
		printf("IRTsim cannot run during this hours");
		exit(0);
	}
#endif

	for (int argcnt = 1; argcnt < argc; argcnt++) {
		if (!strcmp(argv[argcnt], "-tf")) {
			n = sprintf(test_file_name, "%s", argv[argcnt + 1]);
			test_file_flag = 1;
			irt_multi_image = 1;
		}
	}
	if (argc < 3 && !test_file_flag) {
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-----------------------------------\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "Use: IRTsim\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-f or -fi input_file\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-fo output_file (if not provided, input_file used as prefix)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-di input directory including / or \\ at the end (if not provided, current directory is used)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-do output directory including / or \\ at the end (if not provided, current directory is used)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-num num_of_images (default 1)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-irt_h5_mode enable irt_h5_mode\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-read_iimage_BufW enable read_iimage_BufW\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-read_mimage_BufW enable read_mimage_BufW\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-hl_only high level sim only (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-bmp_rd_order bmp image lines order (default 0 (from H-1 to 0))\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-bmp_wr_order bmp image lines order (default 0 (from H-1 to 0))\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-flow_mode 0 - new with Adaptive2 rate ctrl, default\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "           1 - new with Fixed or Adaptive1 rate ctrl\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "           2 - old with Fixed or Adaptive1 rate ctrl\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-multi_image enable multiple images\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-tf test_file_name\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_buf_format	rotation memory buffer mode format (0 - fixed, 1 - per task) (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_buf_select	rotation memory buffer mode select (0 - manual, 1 - auto) (default 1 - auto)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_buf_mode	rotation memory buffer mode manual (default 3)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_buf_1b_mode	rotation memory buffer 1byte mode (0 - 0...31, 1 - {16,0}, ...{31, 15}) (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_buf_format		mesh memory buffer mode format (0 - fixed, 1 - per task) (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_buf_select		mesh memory buffer mode select (0 - manual, 1 - auto) (default 1 - auto)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_buf_mode		mesh memory buffer mode manual (default 3)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_buf_1b_mode	mesh memory buffer 1byte mode (0 - 0...31, 1 - {16,0}, ...{31, 15}) (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Ho  output image vertical   height (default - 64)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Wo  output image horizontal width  (default - 128)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Hi  input  image vertical   height (default - detected from input image file)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Wi  input  image horizontal width  (default - detected from input image file)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Yco output image vertical   center .5 precision (default - output image vertical   height / 2)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Xco output image horizontal center .5 precision (default - output image horizontal width / 2)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Yci input  image vertical   center .5 precision (default - input  image vertical   height / 2)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Xci input  image horizontal center .5 precision(default - input  image horizontal width / 2)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Pwo output image pixel width (0 - 8 bits, 1 - 10 bits, 2 - 12 bits, 3 - 14 bits, 4 - 16 bits) (default - 0 - 8 bits)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Pwi input  image pixel width (0 - 8 bits, 1 - 10 bits, 2 - 12 bits, 3 - 14 bits, 4 - 16 bits) (default - 0 - 8 bits)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Ppo output image pixel LSB padding (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Ppi input  image pixel LSB padding (default - 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-int_mode interpolation  mode (0 - bilinear, 1 - nearest neigbour) (default - bilinear)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-irt_mode transformation mode (0 - rotation, 1 - affine, 2 - projection, 3 - mesh) (default - rotation)\n\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-bg_mode  background     mode (0 - programmable background, 1 - frame boundary repetition) (default - programmable background)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-crd_mode rotation/affine coordinate mode (0 - fixed point, 1 - fp32) (default - fixed point)\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_angle rotation angle (default 30)\n\n");

		//IRT_TRACE("-aff_mode affine mode (0 - rotation, 1 - scaling, 2 - rotation*scaling, 3 - scaling*rotation) (default - rotaion)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-aff_mode affine order (any composition and order of RMST, R-rotation, M-mirror/reflection, S-scaling, T-shear/translation)) (default - R, rotation)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rfl_mode reflection mode (0 - none, 1 - about Y axis, 2 - about X axis, 3 - about original) (default - none)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-shr_mode shearing mode (0 - none, 1 - horizontal, 2 - vertical, 3 - both) (default - none)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-shr_xang horizontal shearing angle (default - 90)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-shr_yang vertical shearing angle (default - 90)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Sx       affine/projection horizontal scaling factor (default 1)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Sy       affine/projection vertical   scaling factor (default 1)\n\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-prj_mode  projection mode (0 - rotation, 1 - affine, 2 - projection) (default - rotation)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-prj_order projection order (any composition and order of YRPS (Y-yaw, R-roll, P-pitch, S-shearing) (default - YRP)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-roll      projection roll angle  (default 50)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-pitch     projection pitch angle (default -60)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-yaw       projection yaw angle   (default 60)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Wd        projection plane distance (default 400)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-Zd        projection focal distance (default 1000)\n\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_mode   mesh mode (0 - rotation, 1 - affine, 2 - projection, 3 - distortion) (default - rotation)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_format mesh image format (0 - fixed point, 1 - FP32) (default - fixed point)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_point  mesh fixed point location (0: S15, ..., 15: S.15) (default - 0, S15)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_order  mesh order (0 - pre-distortion, 1 - post-distortion) (default - predistortion)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_rel    mesh image relative mode (0 - absolute, 1 - relative) (default - absolute)\n\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_Sh     mesh image horizontal scaling (default 1)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_Sv     mesh image vertical   scaling (default 1)\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-dist_x    image distortion horizontal factor (default 1)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-dist_y    image distortion vertical   factor (default 1)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-dist_r    image distortion radius     factor (default 0)\n\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rate_mode pixel processing rate mode (0 - fixed, equal to processing size, 1 - adaptive) (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-proc_size pixel / cycle (default 8)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-proc_auto processing size is autogenerated by descriptor generator for fixed rate mode (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-lwf oimage_line_wr_format (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-bg  pixel background      (default 0)\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-pw  PIXEL_WIDTH (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-cw  COORD_WIDTH (default 16)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-ws  WEIGHT_SHIFT (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rot_prec_auto_adj enable rotation precision auto adjust to fit 31 bits\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-mesh_prec_auto_adj enable mesh precision auto adjust to fit 16 bits\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-oimg_auto_adj out enable image auto adjust to buffer mode\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-oimg_auto_adj_rate enable image auto adjust rate in procents\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-use_Si_delta_margin enable Si_delta large margin\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-dbg_mode debug mode (0 - functional, 1 - bg flag) (default 0)\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-print_log enable print_log_file\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-print_images enable print_image_files\n");
		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-rand_input_image enable randomizion of input image data\n");

		IRT_TRACE(IRT_TRACE_LEVEL_HELP, "-----------------------------------\n");
		exit (0);

	}

	bool input_file_name_flag = false, output_file_name_flag = false, input_file_dir_flag = false, output_file_dir_flag = false;
	for (int argcnt = 1; argcnt < argc; argcnt++) {
		if (!strcmp(argv[argcnt], "-f")) {
			n = sprintf(input_file_name, "%s", argv[argcnt + 1]);
			input_file_name_flag = true;
		}
		if (!strcmp(argv[argcnt], "-fi")) {
			n = sprintf(input_file_name, "%s", argv[argcnt + 1]);
			input_file_name_flag = true;
		}
		if (!strcmp(argv[argcnt], "-fo")) {
			n = sprintf(output_file_name, "%s", argv[argcnt + 1]);
			output_file_name_flag = true;
		}
		if (!strcmp(argv[argcnt], "-di")) {
			n = sprintf(input_file_dir, "%s", argv[argcnt + 1]);
			input_file_dir_flag = true;
		}
		if (!strcmp(argv[argcnt], "-do")) {
			n = sprintf(output_file_dir, "%s", argv[argcnt + 1]);
			output_file_dir_flag = true;
		}

		if (!strcmp(argv[argcnt], "-bmp_wr_order")) irt_bmp_wr_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-bmp_rd_order")) irt_bmp_rd_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-multi_image"))  irt_multi_image = 1;
		if (!strcmp(argv[argcnt], "-num"))			num_of_images = (uint8_t)atoi(argv[argcnt + 1]);
		if (!strcmp(argv[argcnt], "-print_images"))	print_out_files = 1;
	}

	if (input_file_name_flag == 0) { //file not provided
		IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Input file name is not provided\n");
		exit(0);
	}

	strncpy(input_file, input_file_dir, strlen(input_file_dir));
	strncat(input_file, input_file_name, strlen(input_file_name));
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Input file is %s\n", input_file);

	if (!output_file_name_flag) { //not provided, used as input name
		strncpy(output_file_name, input_file_name, strlen(input_file_name));
	}
	strncpy(output_file_name_no_ext, output_file_name, strlen(output_file_name) - 4);
	strncpy(output_file, output_file_dir, strlen(output_file_dir));
	strncat(output_file, output_file_name_no_ext, strlen(output_file_name_no_ext));
	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Output file is %s.bmp\n", output_file);

	log_file = fopen("log_file.txt", "w");
	test_res = fopen("Test_results.txt", "a");

	uint64_t iimage_addr=0, oimage_addr=0, mimage_addr=0;
	meta_data_struct iimage_meta_in, iimage_meta_out, mimage_meta_in, mimage_meta_out;
	bus128B_struct iimage_rd_data, oimage_wr_data, mimage_rd_data;
	uint32_t i_width,  i_height;
	//int o_width,  o_height;

	IRT_top0 = new IRT_top();
	IRT_top0->setInstanceName("IRT_TOP");
	IRT_top0->pars_reset();
	IRT_top0->desc_reset();
	IRT_top0->reset();
	create_mems(&ext_mem, &input_image, &output_image);

	ReadBMP (IRT_top0, input_file, output_file_dir, i_width, i_height, argc, argv);
	//n = sprintf(input_file_name, "%s_out.bmp", input_file_name);
	IRT_top0->rot_pars[0].Pwo = 8;
	IRT_top0->irt_desc[0].Ppo = 0;

	char irt_desc_strs[MAX_TASKS][IRT_DESC_STR_LENGTH];
	char* irt_desc_str[IRT_DESC_STR_LENGTH];
	int argcnt = 0;
	FILE* test_file;

	if (test_file_flag == 1) {

		test_file = fopen(test_file_name, "r");
		if (test_file == nullptr) {
			IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Could not open test file %s", test_file_name);
			return 1;
		}

		while (fgets(irt_desc_strs[num_of_images], IRT_DESC_STR_LENGTH, test_file) != nullptr) {
			num_of_images++;
		}
		fclose(test_file);
		if (num_of_images > OUT_NUM_IMAGES_MT || num_of_images > OUT_NUM_IMAGES) {
			IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Error: number of input images in test file (%d) > OUT_NUM_IMAGES_MT (%d) or OUT_NUM_IMAGES (%d)\n", num_of_images, OUT_NUM_IMAGES_MT, OUT_NUM_IMAGES);
			IRT_CLOSE_FAILED_TEST(0);
		}
		IRT_TRACE_UTILS(IRT_TRACE_LEVEL_INFO, "PARSING COMPLETED TEST STARTS num_descriptors=%d\n", num_of_images);
	} else {
		num_of_images = 1;
		irt_params_parsing(IRT_top0, 0, argc, argv, (uint16_t)i_width, (uint16_t)i_height);

		IRT_TRACE(IRT_TRACE_LEVEL_INFO, "-----------------------------------\n");
		IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Rotating %s by %f degree\n", input_file, IRT_top0->rot_pars[0].irt_angles[e_irt_angle_rot]);
		IRT_TRACE(IRT_TRACE_LEVEL_INFO, "-----------------------------------\n");
	}

	/***************************************************/
	/* define rotation parameters for each stripe task */
	/***************************************************/
	uint8_t desc;
	IRT_top0->num_of_tasks = 0;
	for (uint8_t image = 0; image < num_of_images; image++) { //SJAGADALE
		//Parse parameter string for each descriptor and call irt_descriptor_gen() here.
		//If not, few arguments line buf_mode for all descriptors are override by the last descriptor from test file
		if (test_file_flag == 1) {
			char* pch;
			argcnt = 0;
			pch = strtok(irt_desc_strs[image], " ");
			irt_desc_str[argcnt++] = pch;

			while (pch != nullptr)	{ // convert single string into array of strings per parameters i.e. char* argv[] format
				pch = strtok(nullptr, " ");
				irt_desc_str[argcnt] = pch;
				argcnt++;
			}
			irt_params_parsing(IRT_top0, image, (argcnt - 1), irt_desc_str, (uint16_t)i_width, (uint16_t)i_height);//52 based on test generated by verif env

			for (uint8_t plane = 0; plane < PLANES; plane++) {
				desc = image * PLANES + plane; //SJAGADALE
				if (IRT_top0->irt_desc[desc].image_par[IIMAGE].W > IIMAGE_W || IRT_top0->irt_desc[desc].image_par[IIMAGE].H > IIMAGE_H) {
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Params parsing error: desc %d read image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
						desc, IRT_top0->irt_desc[desc].image_par[IIMAGE].W, IRT_top0->irt_desc[desc].image_par[IIMAGE].H, IIMAGE_W, IIMAGE_H);
					IRT_CLOSE_FAILED_TEST(0);
				}
				if (IRT_top0->irt_desc[desc].image_par[OIMAGE].W > OIMAGE_W || IRT_top0->irt_desc[desc].image_par[OIMAGE].H > OIMAGE_H) {
					IRT_TRACE_UTILS(IRT_TRACE_LEVEL_ERROR, "Desc gen error: desc %d output image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
						desc, IRT_top0->irt_desc[desc].image_par[OIMAGE].W, IRT_top0->irt_desc[desc].image_par[OIMAGE].H, OIMAGE_W, OIMAGE_H);
					IRT_CLOSE_FAILED_TEST(0);
				}
				irt_mems_modes_check(IRT_top0->rot_pars[desc], IRT_top0->rot_pars[0], desc);
			}
		}

		//--------------
		for (uint8_t stripe = 0; stripe < 1/*irt_desc[0].image_par[OIMAGE].Ns*/; stripe++) {
			for (uint8_t plane = 0; plane < PLANES; plane++) {
				desc = image * PLANES + plane; //SJAGADALE
				if (plane == 0)
					irt_descriptor_gen(IRT_top0->irt_cfg, IRT_top0->rot_pars[desc], IRT_top0->irt_desc[desc], (uint16_t)i_width << IRT_top0->irt_desc[desc].image_par[IIMAGE].Ps, desc);
				else {
					memcpy(&IRT_top0->rot_pars[desc], &IRT_top0->rot_pars[image * PLANES], sizeof(rotation_par));
					memcpy(&IRT_top0->irt_desc[desc], &IRT_top0->irt_desc[image * PLANES], sizeof(irt_desc_par));
					IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_start = (uint64_t)IIMAGE_REGION_START + (uint64_t)plane * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL;
					IRT_top0->irt_desc[desc].image_par[OIMAGE].addr_start = (uint64_t)OIMAGE_REGION_START + (uint64_t)desc  * OIMAGE_W * OIMAGE_H * BYTEs4PIXEL;
					IRT_top0->irt_desc[desc].image_par[MIMAGE].addr_start = (irt_multi_image ? (uint64_t)MIMAGE_REGION_START_MT : (uint64_t)MIMAGE_REGION_START) + (uint64_t)image * OIMAGE_W * OIMAGE_H * BYTEs4MESH;
				}
				IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_end = IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_start + (uint64_t)IRT_top0->irt_desc[desc].image_par[IIMAGE].Hs * IRT_top0->irt_desc[desc].image_par[IIMAGE].H - 1;
				//updating Xc and start address of output image to include (stripe number * stripe width) offset
				IRT_top0->irt_desc[desc].image_par[OIMAGE].Xc -= (stripe) * 2 * IRT_top0->irt_desc[desc].image_par[OIMAGE].S;
				IRT_top0->irt_desc[desc].image_par[OIMAGE].addr_start += ((uint64_t)stripe) * IRT_top0->irt_desc[desc].image_par[OIMAGE].S;
				IRT_top0->num_of_tasks++;
				if (test_failed)
					goto label_IRT_SIM_END;
			}
		}

		uint32_t proc_time_o = IRT_top0->irt_desc[image].image_par[OIMAGE].H * IRT_top0->irt_desc[image].image_par[OIMAGE].W / IRT_top0->irt_desc[image].proc_size;
		uint32_t proc_time_i = ((IRT_top0->irt_desc[image].image_par[IIMAGE].H * IRT_top0->irt_desc[image].image_par[IIMAGE].W) << IRT_top0->irt_desc[image].image_par[IIMAGE].Ps) >> 8;
		uint32_t proc_time_r = ((IRT_top0->irt_desc[image].Yi_end - IRT_top0->irt_desc[image].Yi_start + 1) * (IRT_top0->irt_desc[image].image_par[IIMAGE].S << IRT_top0->irt_desc[image].image_par[IIMAGE].Ps)) >> 8;
		uint32_t proc_time_m = (IRT_top0->irt_desc[image].image_par[MIMAGE].H * IRT_top0->irt_desc[image].image_par[MIMAGE].S * (uint32_t)ceil(IRT_top0->rot_pars->mesh_Sv) * (uint32_t)ceil(IRT_top0->rot_pars->mesh_Sh)) >> 2;

		//IRT_TRACE_UTILS("proc_time_o %d\n", proc_time_o);
		//IRT_TRACE_UTILS("proc_time_i %d\n", proc_time_i);
		//IRT_TRACE_UTILS("proc_time_r %d\n", proc_time_r);
		//IRT_TRACE_UTILS("proc_time_m %d\n", proc_time_m);

		timeout_time += (uint64_t)10 * PLANES * num_of_images * (1 + max(proc_time_m, max(proc_time_r, max(proc_time_o, proc_time_i))));
	}

	for (uint8_t image = 0; image < num_of_images; image++)
		IRThl (IRT_top0, image, PLANES, (uint16_t)i_width, (uint16_t)i_height, input_file, output_file_name_no_ext, output_file_dir);
	bool iimage_read, oimage_write, mimage_read;
	uint16_t ilsb_pad_rd, imsb_pad_rd;
	uint16_t mlsb_pad_rd, mmsb_pad_rd;

	memset(&iimage_meta_in, 0, sizeof(iimage_meta_in));
	memset(&mimage_meta_in, 0, sizeof(mimage_meta_in));
	memset(&iimage_rd_data, 0, sizeof(iimage_rd_data));
	memset(&mimage_rd_data, 0, sizeof(mimage_rd_data));

	IRT_TRACE(IRT_TRACE_LEVEL_INFO, "First run with timeout of %lld\n", timeout_time);

#if 1
	IRT_top0->reset();

	if (hl_only == 0) {
		do {

			//printf ("Cycle %d\n", cycle);
			cycle++;

			done = IRT_top0->run(iimage_addr, iimage_read, ilsb_pad_rd, imsb_pad_rd, iimage_meta_out, iimage_rd_data, iimage_meta_in,
				oimage_addr, oimage_write, oimage_wr_data,
				mimage_addr, mimage_read, mlsb_pad_rd, mmsb_pad_rd, mimage_meta_out, mimage_rd_data, mimage_meta_in);

			ext_memory_wrapper(mimage_addr, mimage_read, 0, oimage_wr_data, mimage_rd_data, mimage_meta_out, mimage_meta_in, mlsb_pad_rd, mmsb_pad_rd);
			ext_memory_wrapper(iimage_addr, iimage_read, 0, oimage_wr_data, iimage_rd_data, iimage_meta_out, iimage_meta_in, ilsb_pad_rd, imsb_pad_rd);
			ext_memory_wrapper(oimage_addr, 0, oimage_write, oimage_wr_data, iimage_rd_data, iimage_meta_out, iimage_meta_in, ilsb_pad_rd, imsb_pad_rd);

		} while (done == 0 && (cycle < timeout_time) && test_failed == 0);

		for (desc = 0; desc < IRT_top0->num_of_tasks; desc++) {

			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Task %d IRT %s processing rate statistics\n", desc, irt_irt_mode_s[IRT_top0->irt_desc[desc].irt_mode]);
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Min/Max/Avg: %d/%d/%3.2f\n", IRT_top0->rot_pars[desc].min_proc_rate, IRT_top0->rot_pars[desc].max_proc_rate, (double)IRT_top0->rot_pars[desc].acc_proc_rate / IRT_top0->rot_pars[desc].cycles);
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Min/Max/Avg: ");
			for (uint8_t bin = 1; bin < 9; bin++) {
				if (IRT_top0->rot_pars[desc].rate_hist[bin - 1] != 0) {
					IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%d/", bin);
					break;
				}
			}
			for (uint8_t bin = 8; bin > 0; bin--) {
				if (IRT_top0->rot_pars[desc].rate_hist[bin - 1] != 0) {
					IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%d/", bin);
					break;
				}
			}
			IRT_top0->rot_pars[desc].acc_proc_rate = 0;
			uint32_t bin_sum = 0;
			for (uint8_t bin = 1; bin < 9; bin++) {
				IRT_top0->rot_pars[desc].acc_proc_rate += ((uint32_t)bin * IRT_top0->rot_pars[desc].rate_hist[bin - 1]);
				bin_sum += IRT_top0->rot_pars[desc].rate_hist[bin - 1];
			}
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "%3.2f\n", (double)IRT_top0->rot_pars[desc].acc_proc_rate / bin_sum);
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Bin:    1    2    3    4    5    6    7    8\n");
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "Val:");
			for (uint8_t bin = 1; bin < 9; bin++) {
				IRT_TRACE(IRT_TRACE_LEVEL_INFO, " %4d", IRT_top0->rot_pars[desc].rate_hist[bin - 1]);
			}
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "\n");
		}
	}
#endif

	for (uint8_t image = 0; image < num_of_images; image++)
		irt_quality_comp(IRT_top0, image);

	if (hl_only == 0) {
		if (done == 1) {
			IRT_TRACE(IRT_TRACE_LEVEL_INFO, "All tasks are done at cycle %d, comparing %d images\n", cycle, num_of_images);
			for (uint8_t image = 0; image < num_of_images; image++)
				output_image_dump(IRT_top0, image, PLANES, output_file_name_flag ? output_file_name_no_ext : (char*)"IRT_llsim_out_image", output_file_dir);
			IRT_TRACE_TO_RES(test_res, "\n");
			//fclose(test_res);
		} else if (test_failed == 0){
			IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Stopped on timeout at cycle %d\n", cycle);
			IRT_TRACE_TO_RES(test_res, " stopped on timeout at cycle %d\n", cycle);
			//fclose(test_res);
		} else {
			IRT_TRACE(IRT_TRACE_LEVEL_ERROR, "Failed at cycle %d\n", cycle);
			IRT_TRACE_TO_RES(test_res, " failed at cycle %d\n", cycle);
		}
	}
	fflush(test_res);
#if 0
	printf("-------------------------\n");

	printf("Second run\n");
//	for (int i = 0; i < MAX_TASKS; i++) {
//		rm_first_line[i] = 0; rm_last_line[i] = 0; rm_lines_valid[i] = 0; rm_wr_done[i] = 0; rm_start_line[i] = 0;
//	}
//	rm_top_ptr = 0; rm_bot_ptr = 0; rm_fullness = 0;

//	for (int i = 0; i < 4; i++)
//		task[i] = 0;
	IRT_top1 = new IRT_top();
	IRT_top1->pars_reset();
	IRT_top1->desc_reset();
	params_parsing(IRT_top1, num_of_images, argc, argv, i_width, i_height, file_name, rot_angle, rot_angle_adj);
	IRT_top1->num_of_tasks = 0;

	for (int image = 0; image < num_of_images; image++) {
		for (int stripe = 0; stripe < 1/*irt_desc[0].image_par[OIMAGE].Ns*/; stripe++) {
			for (int plane = 0; plane < 3; plane++) {
				irt_descriptor_gen(IRT_top1, image, stripe, 3, plane, rot_angle_adj[image], read_flip[image], i_width);
				IRT_top1->num_of_tasks++;
			}
		}
	}

	cycle = 0;
	done = 0;
	IRT_top1->reset();
	do {

		//printf ("Cycle %d\n", cycle);
		cycle++;

		done = IRT_top1->run(iimage_addr, irt_read, lsb_pad_rd, msb_pad_rd, iimage_meta_out, iimage_rd_data, iimage_meta_in, oimage_addr, irt_write, oimage_wr_data);

		ext_memory_wrapper(iimage_addr, irt_read, 0, oimage_wr_data, iimage_rd_data, iimage_meta_out, iimage_meta_in, lsb_pad_rd, msb_pad_rd);
		ext_memory_wrapper(oimage_addr, 0, irt_write, oimage_wr_data, iimage_rd_data, iimage_meta_out, iimage_meta_in, lsb_pad_rd, msb_pad_rd);

	} while (done == 0 && (cycle < timeout_time));

	if (done == 1) {
		printf("All tasks are done at cycle %d, comparing %d images\n", cycle, num_of_images);
		for (int image = 0; image < num_of_images; image++)
			output_image_dump(IRT_top1, image, 3, file_name, rot_angle[image]);
		fprintf(test_res, "\n");
		fclose(test_res);
	}
	else {
		printf("Stopped on timeout at cycle %d\n", cycle);
		fprintf(test_res, " stopped on timeout at cycle %d\n", cycle);
		fclose(test_res);
	}

#endif

label_IRT_SIM_END:

	fclose(log_file);
	fclose(test_res);
	delete_mems(ext_mem, input_image, output_image);

	delete IRT_top0;
	IRT_top0 = nullptr;

	return 0;
}
