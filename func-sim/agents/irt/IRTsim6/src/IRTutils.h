/*
 * IRTutils.h
 *
 *  Created on: Jun 8, 2019
 *      Author: espektor
 */

#ifndef IRTUTILS_H_
#define IRTUTILS_H_

#include <inttypes.h>
#include <stdarg.h>
#include <float.h>
#include "IRT.h"

#ifdef HABANA_SIMULATION
#include "fs_assert.h"
#endif

#if defined(STANDALONE_ROTATOR) || defined (HABANA_SIMULATION) || defined(GC_BUILD)

#define CALC_FORMATS 9
#define PROJ_ERR_BINS 16
#define PIXEL_ERR_BINS 64

extern int coord_err[CALC_FORMATS];

void create_mems(uint8_t** ext_mem_l, uint16_t**** input_image_l, uint16_t***** output_image_l);
void delete_mems(uint8_t* ext_mem_l, uint16_t*** input_image_l, uint16_t**** output_image_l);

#endif

#if defined(STANDALONE_ROTATOR)
extern bool test_failed;
extern IRT_top* IRT_top0, * IRT_top1;
#endif

typedef enum {
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
typedef enum {
    e_irt_Black = 0,
    e_irt_White = 1,
    e_irt_Red = 2,
    e_irt_Lime = 3,
    e_irt_Blue = 4,
    e_irt_Yellow = 5,
    e_irt_Cyan = 6,
    e_irt_Magenta = 7,
    e_irt_Silver = 8,
    e_irt_Gray = 9,
    e_irt_Maroon = 10,
    e_irt_Olive = 11,
    e_irt_Green = 12,
    e_irt_Purple = 13,
    e_irt_Teal = 14,
    e_irt_Navy = 15,
} Eirt_colors;

enum Eirt_bmp_header {
    e_irt_bmp_header_id_field = 0,
    e_irt_bmp_header_bmp_file_size = 2,
    e_irt_bmp_header_array_offset = 10,
};

enum Eirt_dib_header {
    irt_dib_header_header_size = 0,
    irt_dib_header_image_width = 4,
    irt_dib_header_image_height = 8,
    irt_dib_header_image_planes = 12,
    irt_dib_header_image_bpp    = 14,
    irt_dib_header_image_bi_rgb = 16,
    irt_dib_header_bitmap_size  = 20,
};

#ifdef STANDALONE_ROTATOR
    #define IRT_TRACE(trace_level, ...) if (trace_level <= IRT_MAX_TRACE_LEVEL) printf(__VA_ARGS__)
    //#define IRT_TRACE_DBG(...) printf(__VA_ARGS__)
    #define IRT_TRACE_TO_LOG(trace_level, file,...) if (trace_level <= MAX_LOG_TRACE_LEVEL) fprintf(file, __VA_ARGS__)
    #define IRT_TRACE_TO_RES(file,...) fprintf(file, __VA_ARGS__)
    #define IRT_CLOSE_FAILED_TEST(...) fclose (log_file); fclose (test_res); delete_mems(ext_mem, input_image, output_image); delete IRT_top0; IRT_top0 = NULL; exit(__VA_ARGS__);
    //#define IRT_CLOSE_PASSED_TEST(...) fclose (log_file); fclose (test_res); delete_mems(ext_mem, input_image, output_image); exit(__VA_ARGS__);
    #define IRT_TRACE_UTILS(trace_level, ...) if (trace_level <= IRT_MAX_TRACE_LEVEL) printf(__VA_ARGS__)
    #define IRT_TRACE_TO_RES_UTILS(file,...) fprintf(file, __VA_ARGS__)
#else
    #ifdef RUN_WITH_SV
        #define IRT_TRACE(...) printf(__VA_ARGS__)
        #define IRT_TRACE_TO_LOG(file,...) void(0)
        #define IRT_TRACE_TO_RES(file,...) fatal_to_sv(__VA_ARGS__)
        #define IRT_CLOSE_FAILED_TEST(...) void(0)
        #define IRT_CLOSE_PASSED_TEST(...) void(0)
        #define IRT_TRACE_UTILS(...) printf(__VA_ARGS__)
        #define IRT_TRACE_TO_RES_UTILS(file...) fatal_to_sv(__VA_ARGS__)
    #else
        #ifdef HABANA_SIMULATION
            #define ROT_LOG_NAME "ROT"
            #define TEST_ROT_LOG_NAME "TEST"

            void IRT_TRACE_HABANA_SIM(bool useTestLogger, const std::string& name, const char* msg, ...) __attribute__((format(printf, 3, 4)));
            #define IRT_CLOSE_FAILED_TEST(...) FS_ASSERT(0)
            #define IRT_CLOSE_PASSED_TEST(...) void(0)

            #define IRT_TRACE(msg, ...) IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
            #define IRT_TRACE_TO_LOG(trace_level, file, msg,...) IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
            #define IRT_TRACE_TO_RES(file, msg,...) IRT_TRACE_HABANA_SIM(false, this->getInstanceName(), (msg), ##__VA_ARGS__)
            #define IRT_TRACE_UTILS(msg, ...) IRT_TRACE_HABANA_SIM(true, std::string("Rotator utils"), (msg), ##__VA_ARGS__)
            #define IRT_TRACE_TO_RES_UTILS(file, msg, ...) IRT_TRACE_HABANA_SIM(true, std::string("Rotator utils"), (msg), ##__VA_ARGS__)
        #else
            #ifdef GC_BUILD
                #define IRT_TRACE(...)
                #define IRT_TRACE_DBG(...)
                #define IRT_TRACE_TO_LOG(trace_level, file,...)
                #define IRT_TRACE_TO_RES(file,...)
                #define IRT_CLOSE_FAILED_TEST(...)
                #define IRT_CLOSE_PASSED_TEST(...)
                #define IRT_TRACE_UTILS(...)
                #define IRT_TRACE_TO_RES_UTILS(file,...)
            #else
            #endif
        #endif
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
void irt_descriptor_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint32_t Hsi, uint8_t desc);
void irt_affine_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_projection_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, bool trace);
void irt_mesh_full_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_sparse_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_interp_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);

uint16_t irt_out_stripe_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);

template < class mimage_type, class coord_type >
void irt_map_image_pars_update(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, mimage_type** mesh_image, uint16_t row, uint16_t col, coord_type& Ymax_line, FILE* fptr_vals, FILE* fptr_pars);

template < class mimage_type >
void irt_map_oimage_res_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, mimage_type** mesh_image);

template < class mimage_type >
void irt_map_iimage_stripe_adj(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc, mimage_type** mesh_image);

void irt_rotation_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
void irt_mesh_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
uint8_t irt_proc_size_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);
uint32_t irt_IBufW_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t Si);
uint32_t irt_IBufH_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t IBufH);
void irt_cfg_init(irt_cfg_pars& irt_cfg);
void irt_par_init(rotation_par& rot_pars);
void irt_mesh_mems_create(irt_mesh_images& mesh_images);
void irt_mesh_mems_delete(irt_mesh_images& mesh_images);
void irt_mems_modes_check(const rotation_par& rot_pars, const rotation_par& rot_pars0, uint8_t desc);

#if defined(STANDALONE_ROTATOR) || defined(CREATE_IMAGE_DUMPS)

void ReadBMP(IRT_top* irt_top, char* filename, char* outdirname, uint32_t& width, uint32_t& height, int argc, char* argv[]);
void WriteBMP(IRT_top* irt_top, uint8_t image, uint8_t planes, char* filename, uint32_t width, uint32_t height, uint16_t*** out_image);
void generate_bmp_header (FILE* file, uint32_t width, uint32_t height, uint32_t row_padded, uint8_t planes);
void output_image_dump (IRT_top* irt_top, uint8_t image, uint8_t planes, char* output_file, char* dirname);
void irt_quality_comp(IRT_top* irt_top, uint8_t image);
void generate_plane_dump(const char* filename, char* dirname, uint32_t width, uint32_t height, Eirt_bmp_order vert_order, uint16_t** image_ptr, uint8_t image, uint8_t plane);
void generate_image_dump(const char* filename, char* dirname, uint32_t width, uint32_t height, Eirt_bmp_order vert_order, uint16_t*** image_ptr, uint8_t image);
#endif

void ext_memory_wrapper (uint64_t addr, bool read, bool write, bus128B_struct wr_data, bus128B_struct &rd_data,
        meta_data_struct meta_in, meta_data_struct &meta_out, uint16_t lsb_pad, uint16_t msb_pad);

void irt_params_parsing(IRT_top *irt_top, uint8_t image, int argc, char* argv[], uint16_t i_width, uint16_t i_height);
void irt_params_analize(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc);


#endif /* IRTUTILS_H_ */
