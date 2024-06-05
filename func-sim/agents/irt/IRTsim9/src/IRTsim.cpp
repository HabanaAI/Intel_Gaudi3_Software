
//============================================================================
// Name        : IRTsim.cpp
// Author      : Evgeny Spektor
// Version     :
// Copyright   :
// Description : Hello World in C++, Ansi-style
//============================================================================

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <iostream>
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
//#include "IRTsim.h"
#include "IRT.h"
#include "IRTutils.h"
#include "time.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Woverflow"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
using namespace std;
using namespace irt_utils_gaudi3;

// FILE* test_res;
#define IRT_DESC_STR_LENGTH 1000

// IRT_top *IRT_top0, *IRT_top1;

namespace irt_utils_gaudi3
{
uint8_t*     ext_mem;
uint64_t***  input_image;
uint64_t**** output_image;

extern FILE* test_res;
extern IRT_top *IRT_top0, *IRT_top1;

extern float***    m_warp_data;
extern float***    m_warp_grad_data;
extern float***    input_im_grad_data;
extern float***    input_im_data;
extern float***    output_im_data;
extern float***    output_im_grad_data;
extern void ext_memory_wrapper(uint64_t          addr,
                               bool              read,
                               bool              write,
                               bus128B_struct    wr_data,
                               bus128B_struct&   rd_data,
                               meta_data_struct  meta_in,
                               meta_data_struct& meta_out,
                               int               lsb_pad,
                               int               msb_pad,
                               bool&             rd_data_valid);
extern void IRThl(IRT_top* irt_top, uint8_t image, uint8_t desc, uint16_t i_width, uint16_t i_height, char* file_name);
extern void IRThl_resampler(IRT_top* irt_top, uint8_t image, uint8_t planes);
// extern void IRThl_resampler(IRT_top* irt_top, uint8_t desc) {

extern int            cycle;
extern int            cycle_per_task;
extern uint8_t        hl_only;
extern uint8_t        descr_gen_only;
extern Eirt_bmp_order irt_bmp_rd_order, irt_bmp_wr_order;

extern FILE* log_file;
extern bool  test_failed;
extern bool  irt_multi_image;
extern bool  print_out_files;
extern int   warp_gen_mode;
extern bool  test_file_flag;
extern char  file_warp_x[100];
extern char  file_warp_y[100];
extern char  file_descr_out[100];
extern float lanczos2_array[2048];
// FILE*        IRTLOG;
} // namespace irt_utils_gaudi3

int main(int argc, char* argv[])
{
    char file_name[50] = {'\0'}, input_file_name[50] = {'\0'}, test_file_name[50] = {'\0'};

    uint8_t  num_of_images = 0;
    uint64_t timeout_time  = 0;
    int      n;
    bool     done = 0;

    time_t     my_time = time(NULL);
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
            n               = sprintf(test_file_name, "%s", argv[argcnt + 1]);
            test_file_flag  = 1;
            irt_multi_image = 1;
        }
    }
    if (argc < 3 && !test_file_flag) {
        IRT_TRACE("-----------------------------------\n");
        IRT_TRACE("Use: IRTsim\n");
        IRT_TRACE("-f input_file\n");
        IRT_TRACE("-num num_of_images (default 1)\n");
        IRT_TRACE("-irt_h5_mode enable irt_h5_mode\n");
        IRT_TRACE("-read_iimage_BufW enable read_iimage_BufW\n");
        IRT_TRACE("-read_mimage_BufW enable read_mimage_BufW\n");
        IRT_TRACE("-hl_only high level sim only (default 0)\n");
        IRT_TRACE("-descr_gen_only Descriptor Generation only (default 0)\n");
        IRT_TRACE("-bmp_rd_order bmp image lines order (default 0 (from H-1 to 0))\n");
        IRT_TRACE("-bmp_wr_order bmp image lines order (default 0 (from H-1 to 0))\n");
        IRT_TRACE("-flow_mode 0 - new with Adaptive2 rate ctrl, default\n");
        IRT_TRACE("           1 - new with Fixed or Adaptive1 rate ctrl\n");
        IRT_TRACE("           2 - old with Fixed or Adaptive1 rate ctrl\n");
        IRT_TRACE("-multi_image enable multiple images\n");
        IRT_TRACE("-tf test_file_name\n");
        IRT_TRACE("\n");

        IRT_TRACE("-rot_buf_format	rotation memory buffer mode format (0 - fixed, 1 - per task) (default - 0)\n");
        IRT_TRACE("-rot_buf_select	rotation memory buffer mode select (0 - manual, 1 - auto) (default 1 - auto)\n");
        IRT_TRACE("-rot_buf_mode	rotation memory buffer mode manual (default 3)\n");
        IRT_TRACE("-rot_buf_1b_mode	rotation memory buffer 1byte mode (0 - 0...31, 1 - {16,0}, ...{31, 15}) (default - "
                  "0)\n");
        IRT_TRACE("\n");

        IRT_TRACE("-mesh_buf_format		mesh memory buffer mode format (0 - fixed, 1 - per task) (default - 0)\n");
        IRT_TRACE("-mesh_buf_select		mesh memory buffer mode select (0 - manual, 1 - auto) (default 1 - auto)\n");
        IRT_TRACE("-mesh_buf_mode		mesh memory buffer mode manual (default 3)\n");
        IRT_TRACE(
            "-mesh_buf_1b_mode	mesh memory buffer 1byte mode (0 - 0...31, 1 - {16,0}, ...{31, 15}) (default - 0)\n");
        IRT_TRACE("\n");

        IRT_TRACE("-Ho  output image vertical   height (default - 64)\n");
        IRT_TRACE("-Wo  output image horizontal width  (default - 128)\n");
        IRT_TRACE("-Hi  input  image vertical   height (default - detected from input image file)\n");
        IRT_TRACE("-Wi  input  image horizontal width  (default - detected from input image file)\n");
        IRT_TRACE("-Yco output image vertical   center .5 precision (default - output image vertical   height / 2)\n");
        IRT_TRACE("-Xco output image horizontal center .5 precision (default - output image horizontal width / 2)\n");
        IRT_TRACE("-Yci input  image vertical   center .5 precision (default - input  image vertical   height / 2)\n");
        IRT_TRACE("-Xci input  image horizontal center .5 precision(default - input  image horizontal width / 2)\n");
        IRT_TRACE("-Pwo output image pixel width (0 - 8 bits, 1 - 10 bits, 2 - 12 bits, 3 - 14 bits, 4 - 16 bits) "
                  "(default - 0 - 8 bits)\n");
        IRT_TRACE("-Pwi input  image pixel width (0 - 8 bits, 1 - 10 bits, 2 - 12 bits, 3 - 14 bits, 4 - 16 bits) "
                  "(default - 0 - 8 bits)\n");
        IRT_TRACE("-Ppo output image pixel LSB padding (default - 0)\n");
        IRT_TRACE("-Ppi input  image pixel LSB padding (default - 0)\n");
        IRT_TRACE("\n");

        IRT_TRACE("-int_mode interpolation  mode (0 - bilinear, 1 - nearest neigbour) (default - bilinear)\n");
        IRT_TRACE("-irt_mode transformation mode (0 - rotation, 1 - affine, 2 - projection, 3 - mesh, 4- Resamp FWD "
                  "5-Resamp BWD1 6-Resamp BWD2 7-Rescale) (default - "
                  "rotation)\n\n");
        IRT_TRACE("-bg_mode  background     mode (0 - programmable background, 1 - rame boundary repeatition) (default "
                  "- programmable background)\n");
        IRT_TRACE("-crd_mode rotation/affine coordinate mode (0 - fixed point, 1 - fp32) (default - fixed point)\n");

        IRT_TRACE("-rot_angle rotation angle (default 30)\n\n");

        // IRT_TRACE("-aff_mode affine mode (0 - rotation, 1 - scaling, 2 - rotation*scaling, 3 - scaling*rotation)
        // (default - rotaion)\n");
        IRT_TRACE("-aff_mode affine order (any composition and order of RMST, R-rotation, M-mirror/reflection, "
                  "S-scaling, T-shear/translation)) (default - R, rotaion)\n");
        IRT_TRACE("-rfl_mode reflection mode (0 - none, 1 - about Y axis, 2 - about X axis, 3 - about original) "
                  "(default - none)\n");
        IRT_TRACE("-shr_mode shearing mode (0 - none, 1 - horizontal, 2 - vertical, 3 - both) (default - none)\n");
        IRT_TRACE("-shr_xang horizontal shearing angle (default - 90)\n");
        IRT_TRACE("-shr_yang vertical shearing angle (default - 90)\n");
        IRT_TRACE("-Sx       affine/projection horizontal scaling factor (default 1)\n");
        IRT_TRACE("-Sy       affine/projection vertical   scaling factor (default 1)\n\n");

        IRT_TRACE("-prj_mode  projection mode (0 - rotation, 1 - affine, 2 - projection) (default - rotation)\n");
        IRT_TRACE("-prj_order projection order (any composition and order of YRPS (Y-yaw, R-roll, P-pitch, S-shearing) "
                  "(default - YRP)\n");
        IRT_TRACE("-roll      projection roll angle  (default 50)\n");
        IRT_TRACE("-pitch     projection pitch angle (default -60)\n");
        IRT_TRACE("-yaw       projection yaw angle   (default 60)\n");
        IRT_TRACE("-Wd        projection plane distance (default 400)\n\n");
        IRT_TRACE("-Zd        projection focal distance (default 1000)\n\n");

        IRT_TRACE(
            "-mesh_mode   mesh mode (0 - rotation, 1 - affine, 2 - projection, 3 - distortion) (default - rotaion)\n");
        IRT_TRACE("-mesh_format mesh image format (0 - fixed point, 1 - FP32) (default - fixed point)\n");
        IRT_TRACE("-mesh_point  mesh fixed point location (0: S15, ..., 15: S.15) (default - 0, S15)\n");
        IRT_TRACE("-mesh_order  mesh order (0 - pre-distortio, 1 - post-distortion) (default - predistortion)\n");
        IRT_TRACE("-mesh_rel    mesh image relative mode (0 - absolute, 1 - relative) (default - absolute)\n\n");
        IRT_TRACE("-mesh_Sh     mesh image horizontal scaling (default 1)\n");
        IRT_TRACE("-mesh_Sv     mesh image vertical   scaling (default 1)\n");

        IRT_TRACE("-dist_x    image distortion horizontal factor (default 1)\n");
        IRT_TRACE("-dist_y    image distortion vertical   factor (default 1)\n");
        IRT_TRACE("-dist_r    image distortion radious    factor (default 0)\n\n");

        IRT_TRACE(
            "-rate_mode pixel processing rate mode (0 - fixed, equal to processing size, 1 - adaptive) (default 0)\n");
        IRT_TRACE("-proc_size pixel / cycle (default 8)\n");
        IRT_TRACE(
            "-proc_auto processing size is autogenerated by descriptor generator for fixed rate mode (default 0)\n");
        IRT_TRACE("-lwf oimage_line_wr_format (default 0)\n");
        IRT_TRACE("-bg  pixel background      (default 0)\n");

        IRT_TRACE("-pw  PIXEL_WIDTH (default 0)\n");
        IRT_TRACE("-cw  COORD_WIDTH (default 16)\n");
        IRT_TRACE("-ws  WEIGHT_SHIFT (default 0)\n");
        IRT_TRACE("-rot_prec_auto_adj enable rotation precision auto adjust to fit 31 bits\n");
        IRT_TRACE("-mesh_prec_auto_adj enable mesh precision auto adjust to fit 16 bits\n");
        IRT_TRACE("-oimg_auto_adj out enable image auto adjust to buffer mode\n");
        IRT_TRACE("-oimg_auto_adj_rate out enable image auto adjust rate in procents\n");

        IRT_TRACE("-dbg_mode debug mode (0 - functional, 1 - bg flag) (default 0)\n");
        IRT_TRACE("-print_log enable print_log_file\n");
        IRT_TRACE("-print_images enable print_image_files\n");
        IRT_TRACE("-rand_input_image enable randomizion of input image data\n");
        IRT_TRACE(
            "-Ws Warp stride Supported Values are 0,1,3 (default 0) actual_stride => 0, 2, 4 respectively   \n\n");

        IRT_TRACE("-----------------------------------\n");
        exit(0);
    }

    int file_name_flag = 0;
    for (int argcnt = 1; argcnt < argc; argcnt++) {
        if (!strcmp(argv[argcnt], "-f")) {
            n              = sprintf(input_file_name, "%s", argv[argcnt + 1]);
            file_name_flag = 1;
        }

        if (!strcmp(argv[argcnt], "-bmp_wr_order"))
            irt_bmp_wr_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-bmp_rd_order"))
            irt_bmp_rd_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-multi_image"))
            irt_multi_image = 1;
        if (!strcmp(argv[argcnt], "-num"))
            num_of_images = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-print_images"))
            print_out_files = 1;
        if (!strcmp(argv[argcnt], "-file_warp_x"))
            sprintf(file_warp_x, "%s", argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-file_warp_y"))
            sprintf(file_warp_y, "%s", argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-file_descr_out"))
            sprintf(file_descr_out, "%s", argv[argcnt + 1]);
    }

    if (file_name_flag == 0) { // file not provided
        IRT_TRACE("Input file name is not provided\n");
        exit(0);
    }

    strncpy(file_name, input_file_name, strlen(input_file_name) - 4);
    // IRT_TRACE("Input file name is %s %s\n", file_name, input_file_name);

    log_file = fopen("log_file.txt", "w");
    test_res = fopen("Test_results.txt", "a");
    IRTLOG   = fopen("irt_log.txt", "w");

    uint64_t         iimage_addr = 0, oimage_addr = 0, mimage_addr = 0, gimage_addr = 0;
    meta_data_struct iimage_meta_in, iimage_meta_out, mimage_meta_in, mimage_meta_out, gimage_meta_out, gimage_meta_in;
    bus128B_struct   iimage_rd_data, oimage_wr_data, mimage_rd_data, gimage_rd_data;
    uint32_t         i_width, i_height;
    // int o_width,  o_height;

    IRT_top0 = new IRT_top(true);
    IRT_top0->setInstanceName("IRT_TOP");
    IRT_top0->pars_reset();
    IRT_top0->desc_reset();
    IRT_top0->reset();
    create_mems(&ext_mem, &input_image, &output_image);

    ReadBMP(IRT_top0, input_file_name, i_width, i_height, argc, argv);
    n                         = sprintf(input_file_name, "%s_out.bmp", input_file_name);
    IRT_top0->rot_pars[0].Pwo = 8;
    IRT_top0->irt_desc[0].Ppo = 0;

    char  irt_desc_strs[MAX_TASKS][IRT_DESC_STR_LENGTH];
    char* irt_desc_str[IRT_DESC_STR_LENGTH];
    int   argcnt = 0;
    FILE* test_file;

    if (test_file_flag == 1) {

        test_file = fopen(test_file_name, "r");
        if (test_file == NULL) {
            printf("Could not open test file %s", test_file_name);
            return 1;
        }

        while (fgets(irt_desc_strs[num_of_images], IRT_DESC_STR_LENGTH, test_file) != NULL) {
            num_of_images++;
        }
        fclose(test_file);
        if (num_of_images > OUT_NUM_IMAGES_MT || num_of_images > OUT_NUM_IMAGES) {
            IRT_TRACE_UTILS(
                "Error: number of input images in test file (%d) > OUT_NUM_IMAGES_MT (%d) or OUT_NUM_IMAGES (%d)\n",
                num_of_images,
                OUT_NUM_IMAGES_MT,
                OUT_NUM_IMAGES);
            IRT_CLOSE_FAILED_TEST(0);
        }
        IRT_TRACE_UTILS("PARSING COMPLETED TEST STARTS num_descriptors=%d\n", num_of_images);
    } else {
        num_of_images = 1;
        irt_params_parsing(IRT_top0, 0, argc, argv, (uint16_t)i_width, (uint16_t)i_height);

        IRT_TRACE("-----------------------------------\n");
        IRT_TRACE("Rotating %s by %f degree\n", file_name, IRT_top0->rot_pars[0].irt_angles[e_irt_angle_rot]);
        IRT_TRACE("-----------------------------------\n");
    }

    /***************************************************/
    /* define rotation parameters for each stripe task */
    /***************************************************/
    uint8_t desc;
    IRT_top0->num_of_tasks = 0;
    // IRT_TRACE_UTILS("num_of_images %d PLANES %d stripe 1 \n", num_of_images, PLANES);
    for (uint8_t image = 0; image < num_of_images; image++) { // SJAGADALE
        // Parse parameter string for each descriptor and call irt_descriptor_gen() here.
        // If not, few arguments line buf_mode for all descriptors are override by the last descriptor from test file
        if (test_file_flag == 1) {
            char* pch;
            argcnt                 = 0;
            pch                    = strtok(irt_desc_strs[image], " ");
            irt_desc_str[argcnt++] = pch;

            while (pch != NULL) { // convert single string into array of strings per parameters i.e. char* argv[] format
                pch                  = strtok(NULL, " ");
                irt_desc_str[argcnt] = pch;
                argcnt++;
            }
            irt_params_parsing(IRT_top0,
                               image,
                               (argcnt - 1),
                               irt_desc_str,
                               (uint16_t)i_width,
                               (uint16_t)i_height); // 52 based on test generated by verif env

            for (uint8_t plane = 0; plane < PLANES; plane++) {
                desc = image * PLANES + plane; // SJAGADALE
                if (IRT_top0->irt_desc[desc].image_par[IIMAGE].W > IIMAGE_W ||
                    IRT_top0->irt_desc[desc].image_par[IIMAGE].H > IIMAGE_H) {
                    IRT_TRACE_UTILS("Params parsing error: desc %d read image size %dx%d is not supported, exceeds "
                                    "%dx%d maximum supported resolution\n",
                                    desc,
                                    IRT_top0->irt_desc[desc].image_par[IIMAGE].W,
                                    IRT_top0->irt_desc[desc].image_par[IIMAGE].H,
                                    IIMAGE_W,
                                    IIMAGE_H);
                    IRT_CLOSE_FAILED_TEST(0);
                }
                if (IRT_top0->irt_desc[desc].image_par[OIMAGE].W > OIMAGE_W ||
                    IRT_top0->irt_desc[desc].image_par[OIMAGE].H > OIMAGE_H) {
                    IRT_TRACE_UTILS("Desc gen error: desc %d output image size %dx%d is not supported, exceeds %dx%d "
                                    "maximum supported resolution\n",
                                    desc,
                                    IRT_top0->irt_desc[desc].image_par[OIMAGE].W,
                                    IRT_top0->irt_desc[desc].image_par[OIMAGE].H,
                                    OIMAGE_W,
                                    OIMAGE_H);
                    IRT_CLOSE_FAILED_TEST(0);
                }
                irt_mems_modes_check(IRT_top0->rot_pars[desc], IRT_top0->rot_pars[0], desc);
            }
        }

        //--------------
        for (uint8_t stripe = 0; stripe < 1 /*irt_desc[0].image_par[OIMAGE].Ns*/; stripe++) {
            for (uint8_t plane = 0; plane < PLANES; plane++) {
                desc = image * PLANES + plane; // SJAGADALE
                if (plane == 0) {
                    irt_descriptor_gen(IRT_top0->irt_cfg,
                                       IRT_top0->rot_pars[desc],
                                       IRT_top0->irt_desc[desc],
                                       (uint16_t)i_width << IRT_top0->irt_desc[desc].image_par[IIMAGE].Ps,
                                       desc);
                    // IRT_top0->irt_desc_print(IRT_top0, desc);
                } else {
                    memcpy(&IRT_top0->rot_pars[desc], &IRT_top0->rot_pars[image * PLANES], sizeof(rotation_par));
                    memcpy(&IRT_top0->irt_desc[desc], &IRT_top0->irt_desc[image * PLANES], sizeof(irt_desc_par));
                    IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_start =
                        (uint64_t)IIMAGE_REGION_START + (uint64_t)plane * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL;
                    IRT_top0->irt_desc[desc].image_par[OIMAGE].addr_start =
                        (uint64_t)OIMAGE_REGION_START + (uint64_t)desc * OIMAGE_W * OIMAGE_H * BYTEs4PIXEL;
                    IRT_top0->irt_desc[desc].image_par[MIMAGE].addr_start =
                        (irt_multi_image ? (uint64_t)MIMAGE_REGION_START_MT : (uint64_t)MIMAGE_REGION_START) +
                        (uint64_t)image * OIMAGE_W * OIMAGE_H * BYTEs4MESH;
                }
                IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_end =
                    IRT_top0->irt_desc[desc].image_par[IIMAGE].addr_start +
                    (uint64_t)IRT_top0->irt_desc[desc].image_par[IIMAGE].Hs *
                        IRT_top0->irt_desc[desc].image_par[IIMAGE].H -
                    1;
                // updating Xc and start address of output image to include (stripe number * stripe width) offset
                IRT_top0->irt_desc[desc].image_par[OIMAGE].Xc -=
                    (stripe)*2 * IRT_top0->irt_desc[desc].image_par[OIMAGE].S;
                IRT_top0->irt_desc[desc].image_par[OIMAGE].addr_start +=
                    ((uint64_t)stripe) * IRT_top0->irt_desc[desc].image_par[OIMAGE].S;
                IRT_top0->num_of_tasks++;
                if (test_failed)
                    goto label_IRT_SIM_END;
            }
        }

        uint32_t proc_time_o = IRT_top0->irt_desc[image].image_par[OIMAGE].H *
                               IRT_top0->irt_desc[image].image_par[OIMAGE].W / IRT_top0->irt_desc[image].proc_size;
        uint32_t proc_time_i =
            ((IRT_top0->irt_desc[image].image_par[IIMAGE].H * IRT_top0->irt_desc[image].image_par[IIMAGE].W)
             << IRT_top0->irt_desc[image].image_par[IIMAGE].Ps) >>
            8;
        uint32_t proc_time_r =
            ((IRT_top0->irt_desc[image].Yi_end - IRT_top0->irt_desc[image].Yi_start + 1) *
             (IRT_top0->irt_desc[image].image_par[IIMAGE].S << IRT_top0->irt_desc[image].image_par[IIMAGE].Ps)) >>
            8;
        uint32_t proc_time_m =
            (IRT_top0->irt_desc[image].image_par[MIMAGE].H * IRT_top0->irt_desc[image].image_par[MIMAGE].S *
             (uint32_t)ceil(IRT_top0->rot_pars->mesh_Sv) * (uint32_t)ceil(IRT_top0->rot_pars->mesh_Sh)) >>
            2;

        // IRT_TRACE_UTILS("proc_time_o %d\n", proc_time_o);
        // IRT_TRACE_UTILS("proc_time_i %d\n", proc_time_i);
        // IRT_TRACE_UTILS("proc_time_r %d\n", proc_time_r);
        // IRT_TRACE_UTILS("proc_time_m %d\n", proc_time_m);

        timeout_time +=
            (uint64_t)200 * PLANES * num_of_images * max(proc_time_m, max(proc_time_r, max(proc_time_o, proc_time_i)));
    }
    //---------------------------------------
    // resampler HL model mem allocations
    m_warp_data         = mycalloc3(OIMAGE_H, OIMAGE_W, 2);
    m_warp_grad_data    = mycalloc3(OIMAGE_H, OIMAGE_W, 2);
    output_im_data      = mycalloc3(PLANES, OIMAGE_H, OIMAGE_W);
    output_im_grad_data = mycalloc3(PLANES, OIMAGE_H, OIMAGE_W);
    input_im_data       = mycalloc3(PLANES, IIMAGE_H, IIMAGE_W);
    input_im_grad_data  = mycalloc3(PLANES, IIMAGE_H, IIMAGE_W);
    if (descr_gen_only == 0) {
        for (uint8_t image = 0; image < num_of_images; image++) {
            if (IRT_top0->irt_desc[image].irt_mode >= e_irt_resamp_fwd) {
                uint16_t Ho = IRT_top0->irt_desc[image * PLANES].image_par[OIMAGE].H;
                uint16_t Wo = IRT_top0->irt_desc[image * PLANES].image_par[OIMAGE].W;
                uint16_t Hi = IRT_top0->irt_desc[image * PLANES].image_par[IIMAGE].H;
                uint16_t Wi = IRT_top0->irt_desc[image * PLANES].image_par[IIMAGE].W;
                uint16_t Hm = IRT_top0->irt_desc[image * PLANES].image_par[MIMAGE].H;
                uint16_t Wm = IRT_top0->irt_desc[image * PLANES].image_par[MIMAGE].W;
                //---------------------------------------
                // Init warp/grad/In images
                if (IRT_top0->irt_desc[image].irt_mode != e_irt_rescale) {
                    InitWarp(image * PLANES, IRT_top0->irt_desc[image * PLANES], warp_gen_mode, 0, 1);
                    InitImage /*<e_irt_grm>*/ (image * PLANES,
                                               IRT_top0->irt_desc[image * PLANES],
                                               IRT_top0->rot_pars[image * PLANES],
                                               0,
                                               0,
                                               e_irt_grm,
                                               0);
                }
                InitImage /*<e_irt_irm>*/ (image * PLANES,
                                           IRT_top0->irt_desc[image * PLANES],
                                           IRT_top0->rot_pars[image * PLANES],
                                           0,
                                           0,
                                           e_irt_irm,
                                           0);
                //---------------------------------------
                // InitGraddata(IRT_top0->irt_desc[image*PLANES] ,0);
                // printf("Sample Half precision test");
                // half op1;
                // float tmp=-4.567;
                // uint32_t tmp_uint;
                // op1=half(tmp);
                // myhalf op2;
                // op2.val_hf = op1;
                // uint16_t f2h=0;
                // float h2f = half2float(op2.val_uint);
                // printf("value : %f %x f2h: %x h2f: %f\n",tmp,op2.val_uint, h2f );
                /*bf16 op4 = tmp;
                  mybf16 op3;
                  op3.val_hf = tmp;
                  uint64_t bf_h;
                  uint16_t f2bf = float_to_bf16(tmp);
                  float bf2f = bf16_to_float(f2bf);
                  memcpy(&tmp_uint , &tmp, sizeof(tmp));
                  printf("vale : original : %f - hex : %lx , f2bf : %x  bf2f : %f , bf2f_floatx : %f   \n", tmp,
                  tmp_uint,f2bf,bf2f,float(bf16(tmp)));
                  */
#if 1
                printf("calling HL mode\n");
                IRThl_resampler(IRT_top0, image, PLANES);
#endif
                printf("IRThl_resampler-Done\n");
            } else
                break;
        }
        for (uint8_t image = 0; image < num_of_images; image++) {
            if (IRT_top0->irt_desc[image].irt_mode < 4)
                IRThl(IRT_top0, image, PLANES, (uint16_t)i_width, (uint16_t)i_height, file_name);
        }
    }
    bool     iimage_read, /*oimage_write,*/ mimage_read, gimage_read;
    uint8_t  oimage_write;
    bool     imem_rd_data_valid, mmem_rd_data_valid, gmem_rd_data_valid;
    uint16_t ilsb_pad_rd, imsb_pad_rd;
    uint16_t mlsb_pad_rd, mmsb_pad_rd;
    uint16_t glsb_pad_rd, gmsb_pad_rd;

    memset(&iimage_meta_in, 0, sizeof(iimage_meta_in));
    memset(&mimage_meta_in, 0, sizeof(mimage_meta_in));
    memset(&gimage_meta_in, 0, sizeof(gimage_meta_in));
    memset(&iimage_rd_data, 0, sizeof(iimage_rd_data));
    memset(&mimage_rd_data, 0, sizeof(mimage_rd_data));
    memset(&gimage_rd_data, 0, sizeof(gimage_rd_data));

    printf("First run with timeout of %lld hl_only %d\n", timeout_time, hl_only);

#if 1
    IRT_top0->reset();

    // IRT_TRACE("num_of_tasks %d \n", IRT_top0->num_of_tasks);
    if (hl_only == 0 && descr_gen_only == 0) {
        do {

            // printf ("Cycle %d\n", cycle);
            cycle++;
            cycle_per_task++;

            done = IRT_top0->run(iimage_addr,
                                 iimage_read,
                                 ilsb_pad_rd,
                                 imsb_pad_rd,
                                 iimage_meta_out,
                                 iimage_rd_data,
                                 iimage_meta_in,
                                 imem_rd_data_valid,
                                 mimage_addr,
                                 mimage_read,
                                 mlsb_pad_rd,
                                 mmsb_pad_rd,
                                 mimage_meta_out,
                                 mimage_rd_data,
                                 mimage_meta_in,
                                 mmem_rd_data_valid,
                                 gimage_addr,
                                 gimage_read,
                                 glsb_pad_rd,
                                 gmsb_pad_rd,
                                 gimage_meta_out,
                                 gimage_rd_data,
                                 gimage_meta_in,
                                 gmem_rd_data_valid,
                                 oimage_addr,
                                 oimage_write,
                                 oimage_wr_data);

            ext_memory_wrapper(mimage_addr,
                               mimage_read,
                               0,
                               oimage_wr_data,
                               mimage_rd_data,
                               mimage_meta_out,
                               mimage_meta_in,
                               mlsb_pad_rd,
                               mmsb_pad_rd,
                               mmem_rd_data_valid);
            ext_memory_wrapper(iimage_addr,
                               iimage_read,
                               0,
                               oimage_wr_data,
                               iimage_rd_data,
                               iimage_meta_out,
                               iimage_meta_in,
                               ilsb_pad_rd,
                               imsb_pad_rd,
                               imem_rd_data_valid);
            // ext_memory_wrapper(oimage_addr, 0          , oimage_write, oimage_wr_data, iimage_rd_data,
            // iimage_meta_out, iimage_meta_in, ilsb_pad_rd, imsb_pad_rd, gmem_rd_data_valid);
            ext_memory_wrapper(oimage_addr,
                               0,
                               oimage_write,
                               oimage_wr_data,
                               iimage_rd_data,
                               iimage_meta_out,
                               iimage_meta_in,
                               ilsb_pad_rd,
                               imsb_pad_rd,
                               gmem_rd_data_valid);
            ext_memory_wrapper(gimage_addr,
                               gimage_read,
                               0,
                               oimage_wr_data,
                               gimage_rd_data,
                               gimage_meta_out,
                               gimage_meta_in,
                               glsb_pad_rd,
                               gmsb_pad_rd,
                               gmem_rd_data_valid);

        } while (done == 0 && (cycle < timeout_time) && test_failed == 0);

        for (desc = 0; desc < IRT_top0->num_of_tasks; desc++) {

            IRT_TRACE(
                "Task %d IRT %s processing rate statistics\n", desc, irt_irt_mode_s[IRT_top0->irt_desc[desc].irt_mode]);
            IRT_TRACE("Min/Max/Avg: %d/%d/%3.2f\n",
                      IRT_top0->rot_pars[desc].min_proc_rate,
                      IRT_top0->rot_pars[desc].max_proc_rate,
                      (double)IRT_top0->rot_pars[desc].acc_proc_rate / IRT_top0->rot_pars[desc].cycles);
            IRT_TRACE("Min/Max/Avg: ");
            for (uint8_t bin = 1; bin < 9; bin++) {
                if (IRT_top0->rot_pars[desc].rate_hist[bin - 1] != 0) {
                    IRT_TRACE("%d/", bin);
                    break;
                }
            }
            for (uint8_t bin = 8; bin > 0; bin--) {
                if (IRT_top0->rot_pars[desc].rate_hist[bin - 1] != 0) {
                    IRT_TRACE("%d/", bin);
                    break;
                }
            }
            IRT_top0->rot_pars[desc].acc_proc_rate = 0;
            uint32_t bin_sum                       = 0;
            for (uint8_t bin = 1; bin < 9; bin++) {
                IRT_top0->rot_pars[desc].acc_proc_rate += ((uint32_t)bin * IRT_top0->rot_pars[desc].rate_hist[bin - 1]);
                bin_sum += IRT_top0->rot_pars[desc].rate_hist[bin - 1];
            }
            IRT_TRACE("%3.2f\n", (double)IRT_top0->rot_pars[desc].acc_proc_rate / bin_sum);
            IRT_TRACE("Bin:    1    2    3    4    5    6    7    8\n");
            IRT_TRACE("Val:");
            for (uint8_t bin = 1; bin < 9; bin++) {
                IRT_TRACE(" %4d", IRT_top0->rot_pars[desc].rate_hist[bin - 1]);
            }
            IRT_TRACE("\n");
        }
    }
#endif

    for (uint8_t image = 0; image < num_of_images; image++)
        irt_quality_comp(IRT_top0, image);

    if (hl_only == 0) {
        if (done == 1) {
            IRT_TRACE("All tasks are done at cycle %d, comparing %d images\n", cycle, num_of_images);
            for (uint8_t image = 0; image < num_of_images; image++)
                if (IRT_top0->irt_desc[image].irt_mode < 4)
                    output_image_dump(IRT_top0, image, PLANES, 0);
                else
                    output_image_dump(IRT_top0, image, 1, 0);
            IRT_TRACE_TO_RES(test_res, "\n");
            // fclose(test_res);
        } else if (test_failed == 0) {
            IRT_TRACE("Stopped on timeout at cycle %d\n", cycle);
            IRT_TRACE_TO_RES(test_res, " stopped on timeout at cycle %d\n", cycle);
            // fclose(test_res);
        } else {
            IRT_TRACE("Failed at cycle %d\n", cycle);
            IRT_TRACE_TO_RES(test_res, " failed at cycle %d\n", cycle);
        }
    }
    fflush(test_res);
    for (int i = 0; i < PLANES; i++) {
        for (int j = 0; j < IIMAGE_H; j++) {
            delete[] input_im_data[i][j];
            delete[] input_im_grad_data[i][j];
        }
        delete[] input_im_data[i];
        delete[] input_im_grad_data[i];
    }
    delete[] input_im_data;
    delete[] input_im_grad_data;
    // deleting warp mem data
    for (int i = 0; i < OIMAGE_H; i++) {
        for (int j = 0; j < OIMAGE_W; j++) {
            delete[] m_warp_data[i][j];
            delete[] m_warp_grad_data[i][j];
        }
        delete[] m_warp_data[i];
        delete[] m_warp_grad_data[i];
    }
    delete[] m_warp_data;
    delete[] m_warp_grad_data;
    // deleting output mem data
    for (int i = 0; i < PLANES; i++) {
        for (int j = 0; j < OIMAGE_H; j++) {
            delete[] output_im_data[i][j];
            delete[] output_im_grad_data[i][j];
        }
        delete[] output_im_data[i];
        delete[] output_im_grad_data[i];
    }
    delete[] output_im_data;
    delete[] output_im_grad_data;
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
    IRT_top0 = NULL;

    return 0;
}
#pragma GCC diagnostic pop
