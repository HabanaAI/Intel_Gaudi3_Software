/*
 * IRTutils.h
 *
 *  Created on: Jun 8, 2019
 *      Author: espektor
 */

#ifndef IRTUTILS_H_
#define IRTUTILS_H_

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <vector>
#include "IRT.h"

namespace irt_utils_gaudi3
{
#ifdef HABANA_SIMULATION
#include "fs_assert.h"
#endif

#define UVM_LOW 100 // defining verbosity levels for DBG trace
#define UVM_MEDIUM 200
#define UVM_HIGH 300
#define UVM_FULL 400
extern FILE*    IRTLOG;
extern int      run_failed; // = 0;
extern int      max_log_trace_level;
extern uint16_t h9taskid;

#define CALC_FORMATS 9
#define PROJ_ERR_BINS 16
#define PIXEL_ERR_BINS 64

#define MIN_NORM pow(2, -126)

extern int coord_err[CALC_FORMATS];

#if defined(STANDALONE_ROTATOR) || defined(HABANA_SIMULATION) // ROTSIM_DV_INTGR
void create_mems(uint8_t** ext_mem_l, uint64_t**** input_image_l, uint64_t***** output_image_l);
void delete_mems(uint8_t* ext_mem_l, uint64_t*** input_image_l, uint64_t**** output_image_l);

#endif

#if defined(STANDALONE_ROTATOR)
extern bool     test_failed;
extern IRT_top *IRT_top0, *IRT_top1;
#endif

typedef enum
{
    e_irt_crdf_arch  = 0,
    e_irt_crdf_fpt64 = 1,
    e_irt_crdf_fpt32 = 2,
    e_irt_crdf_fixed = 3,
    e_irt_crdf_fix16 = 4,
    e_irt_mesh_fpt64 = 5,
    e_irt_mesh_fpt32 = 6,
    e_irt_mesh_fixed = 7,
    e_irt_mesh_arch  = 8,
} Eirt_coord_format_type;

#define NUM_OF_COLORS 16
typedef enum
{
    e_irt_Black   = 0,
    e_irt_White   = 1,
    e_irt_Red     = 2,
    e_irt_Lime    = 3,
    e_irt_Blue    = 4,
    e_irt_Yellow  = 5,
    e_irt_Cyan    = 6,
    e_irt_Magenta = 7,
    e_irt_Silver  = 8,
    e_irt_Gray    = 9,
    e_irt_Maroon  = 10,
    e_irt_Olive   = 11,
    e_irt_Green   = 12,
    e_irt_Purple  = 13,
    e_irt_Teal    = 14,
    e_irt_Navy    = 15,
} Eirt_colors;

enum Eirt_bmp_header
{
    e_irt_bmp_header_id_field      = 0,
    e_irt_bmp_header_bmp_file_size = 2,
    e_irt_bmp_header_array_offset  = 10,
};

enum Eirt_dib_header
{
    irt_dib_header_header_size  = 0,
    irt_dib_header_image_width  = 4,
    irt_dib_header_image_height = 8,
    irt_dib_header_image_planes = 12,
    irt_dib_header_image_bpp    = 14,
    irt_dib_header_image_bi_rgb = 16,
    irt_dib_header_bitmap_size  = 20,
};

#ifdef STANDALONE_ROTATOR
#define IRT_TRACE(...) printf(__VA_ARGS__)
#define IRT_TRACE_DBG(...) printf(__VA_ARGS__)
#define IRT_TRACE_TO_LOG(trace_level, file, ...)                                                                       \
    if (trace_level <= max_log_trace_level)                                                                            \
    fprintf(file, __VA_ARGS__)
#define IRT_TRACE_TO_RES(file, ...) fprintf(file, __VA_ARGS__)
#define IRT_CLOSE_FAILED_TEST(...)                                                                                     \
    fclose(log_file);                                                                                                  \
    fclose(test_res);                                                                                                  \
    delete_mems(ext_mem, input_image, output_image);                                                                   \
    delete IRT_top0;                                                                                                   \
    IRT_top0 = NULL;                                                                                                   \
    exit(__VA_ARGS__);
//#define IRT_CLOSE_PASSED_TEST(...) fclose (log_file); fclose (test_res); delete_mems(ext_mem, input_image,
// output_image); exit(__VA_ARGS__);
#define IRT_TRACE_UTILS(...) printf(__VA_ARGS__)
#define IRT_TRACE_TO_RES_UTILS(file, ...) fprintf(file, __VA_ARGS__)
#define IRT_TRACE_TO_RES_UTILS_LOG(trace_level, file, ...)                                                             \
    if (trace_level <= max_log_trace_level)                                                                            \
    fprintf(file, __VA_ARGS__)
#ifdef DISABLE_IRT_TRACE_UTILS_PRINT
#undef IRT_TRACE_UTILS
#define IRT_TRACE_UTILS(...) void(0)
#endif
#else
#ifdef RUN_WITH_SV
//#define IRT_TRACE(...)                    print_to_sv(UVM_LOW,__VA_ARGS__)
//#define IRT_TRACE_DBG(verbosity,...)      print_to_sv(verbosity, __VA_ARGS__)
//#define IRT_TRACE_UTILS(...)              print_to_sv(UVM_LOW,__VA_ARGS__)
//#define IRT_TRACE_TO_RES_UTILS(file,...)  fatal_to_sv(__VA_ARGS__)
//#define IRT_TRACE_TO_RES(file,...)        fatal_to_sv(__VA_ARGS__)

#define IRT_TRACE(...)                                                                                                 \
    fprintf(IRTLOG, __VA_ARGS__);                                                                                      \
    fflush(IRTLOG)
#define IRT_TRACE_DBG(verbosity, ...)                                                                                  \
    fprintf(IRTLOG, __VA_ARGS__);                                                                                      \
    fflush(IRTLOG)
#define IRT_TRACE_UTILS(...)                                                                                           \
    fprintf(IRTLOG, __VA_ARGS__);                                                                                      \
    fflush(IRTLOG)

//#define IRT_TRACE(...)                      void(0)
//#define IRT_TRACE_DBG(verbosity,...)        void(0)
//#define IRT_TRACE_UTILS(...)                void(0)

#define IRT_TRACE_TO_RES_UTILS(file, ...)                                                                              \
    fprintf(file, __VA_ARGS__);                                                                                        \
    fflush(file)
#define IRT_TRACE_TO_RES_UTILS_LOG(trace_level, file, ...)                                                             \
    if (trace_level <= max_log_trace_level)                                                                            \
        fprintf(file, __VA_ARGS__);                                                                                    \
    fflush(file)
#define IRT_TRACE_TO_RES(file, ...) void(0) //   fprintf(file,__VA_ARGS__); fflush(file)
#define IRT_CLOSE_PASSED_TEST(...) void(0)
#define IRT_TRACE_TO_LOG(trace_level, file, ...)                                                                       \
    if (trace_level <= max_log_trace_level)                                                                            \
        fprintf(file, __VA_ARGS__);                                                                                    \
    fflush(file)
#define IRT_CLOSE_FAILED_TEST(...)                                                                                     \
    run_failed = 1;                                                                                                    \
    fprintf(IRTLOG, "IRT_CLOSE_FAILED_TEST\n")
#else
#define ROT_LOG_NAME "ROT"
#define TEST_ROT_LOG_NAME "TEST"

void IRT_TRACE_HABANA_SIM(bool useTestLogger, const std::string& name, const char* msg, ...)
    __attribute__((format(printf, 3, 4)));
#define IRT_CLOSE_FAILED_TEST(...) FS_ASSERT(0)
#define IRT_CLOSE_PASSED_TEST(...) void(0)

#define IRT_TRACE(msg, ...) IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
#define IRT_TRACE_TO_LOG(trace_level, file, msg, ...)                                                                  \
    IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
#define IRT_TRACE_TO_RES(file, msg, ...) IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
#define IRT_TRACE_UTILS(msg, ...) IRT_TRACE_HABANA_SIM(true, std::string("Rotator utils"), (msg), ##__VA_ARGS__)
#define IRT_TRACE_TO_RES_UTILS(file, msg, ...)                                                                         \
    IRT_TRACE_HABANA_SIM(true, std::string("Rotator utils"), (msg), ##__VA_ARGS__)
#define IRT_TRACE_TO_RES_UTILS_LOG(trace_level, file, msg, ...)                                                        \
    IRT_TRACE_HABANA_SIM(true, std::string("Rotator utils"), (msg), ##__VA_ARGS__)
#endif
#endif

void irt_rot_angle_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_aff_coefs_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_oimage_res_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_oimage_corners_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_rotation_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_affine_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_projection_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_xi_first_adj_rot(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_xi_first_adj_aff(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_descriptor_gen(irt_cfg_pars& irt_cfg,
                        rotation_par& rot_pars,
                        irt_desc_par& irt_desc,
                        uint32_t      Hsi,
                        uint8_t       desc);
void irt_affine_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_projection_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_matrix_calc(irt_cfg_pars& irt_cfg,
                          rotation_par& rot_pars,
                          irt_desc_par& irt_desc,
                          uint8_t       desc,
                          bool          trace);
void irt_mesh_full_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_sparse_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_interp_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_resamp_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_rescale_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_BiLiGrad_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void rescale_coeff_gen(Coeff_LUT_t* myLUT,
                       int          dec_factor,
                       int          interpol_factor,
                       int          filter_type,
                       int          max_phases,
                       int          lanczos_prec,
                       int16_t      op_xc,
                       int16_t      ip_xc);

template <class mimage_type, class coord_type>
void irt_map_image_pars_update(irt_cfg_pars& irt_cfg,
                               irt_desc_par& irt_desc,
                               mimage_type** mesh_image,
                               uint16_t      row,
                               uint16_t      col,
                               coord_type&   Ymax_line,
                               FILE*         fptr_vals,
                               FILE*         fptr_pars);

template <class mimage_type>
void irt_map_oimage_res_adj_calc(irt_cfg_pars& irt_cfg,
                                 rotation_par& rot_pars,
                                 irt_desc_par& irt_desc,
                                 uint8_t       desc,
                                 mimage_type** mesh_image);

template <class mimage_type>
void irt_map_iimage_stripe_adj(irt_cfg_pars& irt_cfg,
                               rotation_par& rot_pars,
                               irt_desc_par& irt_desc,
                               uint8_t       desc,
                               mimage_type** mesh_image);

void irt_rotation_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
uint8_t  irt_proc_size_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
uint32_t irt_IBufW_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t Si);
uint32_t irt_IBufH_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t IBufH);
void     irt_cfg_init(irt_cfg_pars& irt_cfg);
void     irt_par_init(rotation_par& rot_pars);
void     irt_mesh_mems_create(irt_mesh_images& mesh_images);
void     irt_mesh_mems_delete(irt_mesh_images& mesh_images);
void     irt_mems_modes_check(const rotation_par& rot_pars, const rotation_par& rot_pars0, uint8_t desc);

// resampler
// Find maximum between two numbers.
int max(int num1, int num2);
// Find minimum between two numbers.
int min(int num1, int num2);
// three dimensional memory allocation.
float*** mycalloc3(int num_ch, int im_h, int im_w);
// return a uniformly distributed random number
double RandomGenerator();
// return a normally distributed random number
double normalRandom();
// Initilizing warp coordinates.
void InitWarp(uint8_t task_idx, irt_desc_par& irt_desc, uint8_t random1, uint8_t verif, float warpincr);

// template < Eirt_blocks_enum block_type >
void InitImage(uint8_t          task_idx,
               irt_desc_par&    irt_desc,
               rotation_par&    rot_par,
               uint8_t          mode,
               uint8_t          verif,
               Eirt_blocks_enum block_type,
               uint8_t          num_range);

void InitGraddata(irt_desc_par& irt_desc, uint8_t random1);
#if defined(STANDALONE_ROTATOR) || defined(CREATE_IMAGE_DUMPS) || defined(RUN_WITH_SV)

void     ReadBMP(IRT_top* irt_top, char* filename, uint32_t& width, uint32_t& height, int argc, char* argv[]);
void     WriteBMP(IRT_top*    irt_top,
                  uint8_t     image,
                  uint8_t     planes,
                  char*       filename,
                  uint32_t    width,
                  uint32_t    height,
                  uint64_t*** out_image);
void     generate_bmp_header(FILE* file, uint32_t width, uint32_t height, uint32_t row_padded, uint8_t planes);
uint32_t output_image_dump(IRT_top* irt_top, uint8_t image, uint8_t planes, uint8_t verif);
void     irt_quality_comp(IRT_top* irt_top, uint8_t image);
void     generate_plane_dump(const char*    fname,
                             uint32_t       width,
                             uint32_t       height,
                             Eirt_bmp_order vert_order,
                             uint64_t**     image_ptr,
                             uint8_t        image,
                             uint8_t        plane,
                             uint8_t        img_format,
                             uint64_t       image_mask,
                             uint8_t        ip_img);
void     generate_image_dump(const char*    fname,
                             uint32_t       width,
                             uint32_t       height,
                             Eirt_bmp_order vert_order,
                             uint64_t***    image_ptr,
                             uint8_t        image,
                             uint64_t       image_mask,
                             uint8_t        ip_img);
#endif

void ext_memory_wrapper(uint64_t          addr,
                        bool              read,
                        uint8_t           write,
                        bus128B_struct    wr_data,
                        bus128B_struct&   rd_data,
                        meta_data_struct  meta_in,
                        meta_data_struct& meta_out,
                        uint16_t          lsb_pad,
                        uint16_t          msb_pad,
                        bool&             rd_data_valid);

uint8_t irt_get_max_proc_size(Eirt_tranform_type irt_mode, Eirt_resamp_dtype_enum DTi);
void    irt_check_max_proc_size(Eirt_tranform_type irt_mode, uint8_t proc_size, Eirt_resamp_dtype_enum DTi);
void irt_params_parsing(IRT_top* irt_top, uint8_t image, int argc, char* argv[], uint16_t i_width, uint16_t i_height);
void irt_params_analize(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void print_filter_coeff(float* lanczos_array, int rows, int cols, char filename[100]);
// void rescale_coeff_gen (Coeff_LUT_t *myLUT,int dec_factor, int interpol_factor, int filter_type,int max_phases,int
// lanczos_prec);
void print_descriptor(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& descFullData, const char* filename);
uint16_t irt_out_stripe_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
} // namespace irt_utils_gaudi3
#endif /* IRTUTILS_H_ */
