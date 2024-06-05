#define _USE_MATH_DEFINES

#include <stdio.h>
#include <cmath>
#include <iostream>

#include <algorithm>
#include "IRTutils.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
//#include "IRTsim.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Woverflow"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

namespace irt_utils_gaudi3
{
// extern uint8_t*     ext_mem;
// extern uint64_t***  input_image;
// extern uint64_t**** output_image;
// extern FILE* log_file;

FILE* test_res;
FILE* IRTLOG;
IRT_top *IRT_top0, *IRT_top1;
float*** m_warp_data;
float*** m_warp_grad_data;
float*** input_im_grad_data;
float*** input_im_data;
float*** output_im_data;
float*** output_im_grad_data;

#if defined(RUN_WITH_SV) // ROTSIM_DV_INTGR
mesh_xy_fp32** mesh_image_full;
mesh_xy_fp32** mesh_image_fp32;
mesh_xy_fi16** mesh_image_fi16;
// extern bool super_irt;
FILE*        test_res;
float***     m_warp_data;
float***     m_warp_grad_data;
float***     input_im_grad_data;
float***     input_im_data;
float***     output_im_data;
float***     output_im_grad_data;
#endif
// uint8_t*     ext_mem;
// uint16_t***  input_image;
// uint64_t**** output_image;
// FILE*        IRTLOG;

//#if defined(STANDALONE_ROTATOR) || defined (HABANA_SIMULATION)

extern bool           irt_h5_mode;
extern bool           read_iimage_BufW;
extern bool           read_mimage_BufW;
extern uint8_t        hl_only;
extern uint8_t        descr_gen_only;
extern Eirt_bmp_order irt_bmp_rd_order, irt_bmp_wr_order;
int                   run_failed = 0;
extern char           file_warp_x[100];
extern char           file_warp_y[100];
extern char           file_descr_out[100];
extern float          lanczos2_array[512];

int         coord_err[CALC_FORMATS];
extern bool print_log_file;
extern bool print_out_files;
extern int  warp_gen_mode;
extern bool rand_input_image;
extern bool irt_multi_image;
extern bool test_file_flag;
extern int  range_bound;

// resampler
#if defined(STANDALONE_ROTATOR)
#if !defined(RUN_WITH_SV) // ROTSIM_DV_INTGR
extern FILE*        IRTLOG;
extern FILE*        test_res;
extern uint8_t*     ext_mem;
extern uint64_t***  input_image;
extern uint64_t**** output_image;
extern float***     m_warp_data;
extern float***     m_warp_grad_data;
extern float***     input_im_grad_data;
extern float***     input_im_data; // input image
extern float***     output_im_data;
extern float***     output_im_grad_data; // backward pass input
#endif
#else
uint8_t*     ext_mem;
uint64_t***  input_image;
uint64_t**** output_image;
FILE*        IRTLOG;
#endif

#if defined(STANDALONE_ROTATOR)
const uint8_t irt_colors[NUM_OF_COLORS][3] = {
    {0, 0, 0},
    {255, 255, 255},
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {0, 255, 255},
    {255, 0, 255},
    {192, 192, 192},
    {128, 128, 128},
    {128, 0, 0},
    {128, 128, 0},
    {0, 128, 0},
    {128, 0, 128},
    {0, 128, 128},
    {0, 0, 128},
};
#endif
// void IRT_UTILS_FATAL(char[400] msg){
//   IRT_TRACE_UTILS(msg);
//   IRT_TRACE_TO_RES_UTILS(msg);
//   IRT_CLOSE_FAILED_TEST(0);
//}

void create_mems(uint8_t** ext_mem_l, uint64_t**** input_image_l, uint64_t***** output_image_l)
{

    uint32_t emem_size, omem_num_images;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
    emem_size       = irt_multi_image ? EMEM_SIZE_MT : EMEM_SIZE;
    omem_num_images = irt_multi_image ? OMEM_NUM_IMAGES_MT : OMEM_NUM_IMAGES;
#else
    emem_size       = EMEM_SIZE;
    omem_num_images = OMEM_NUM_IMAGES;
#endif

    // memories definition
    *ext_mem_l = new uint8_t[emem_size]; // input, output, mesh

    *input_image_l = new uint64_t**[PLANES];
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        (*input_image_l)[plain] = new uint64_t*[IIMAGE_H];
        for (uint16_t row = 0; row < IIMAGE_H; row++) {
            (*input_image_l)[plain][row] = new uint64_t[IIMAGE_W];
        }
    }
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        for (uint16_t row = 0; row < IIMAGE_H; row++) {
            for (uint16_t col = 0; col < IIMAGE_W; col++) {
                (*input_image_l)[plain][row][col] = 0;
            }
        }
    }

    *output_image_l = new uint64_t***[omem_num_images];
    for (uint8_t image = 0; image < omem_num_images; image++) {
        (*output_image_l)[image] = new uint64_t**[PLANES];
        for (uint8_t plain = 0; plain < PLANES; plain++) {
            (*output_image_l)[image][plain] = new uint64_t*[OIMAGE_H];
            for (uint16_t row = 0; row < OIMAGE_H; row++) {
                (*output_image_l)[image][plain][row] = new uint64_t[OIMAGE_W];
            }
        }
    }
}

void delete_mems(uint8_t* ext_mem_l, uint64_t*** input_image_l, uint64_t**** output_image_l)
{

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

void irt_cfg_init(irt_cfg_pars& irt_cfg)
{

    // init buffer modes parameters
    // for rotation memory
    irt_cfg.buf_format[e_irt_block_rot]  = e_irt_buf_format_static;
    irt_cfg.buf_1b_mode[e_irt_block_rot] = 0;
    irt_cfg.buf_select[e_irt_block_rot]  = e_irt_buf_select_auto; // auto
    irt_cfg.buf_mode[e_irt_block_rot]    = 3;
    for (uint8_t mode = 0; mode < IRT_RM_MODES; mode++) {
        irt_cfg.rm_cfg[e_irt_block_rot][mode].BufW     = (2 * IRT_ROT_MEM_BANK_WIDTH) << mode;
        irt_cfg.rm_cfg[e_irt_block_rot][mode].Buf_EpL  = (1 << mode);
        irt_cfg.rm_cfg[e_irt_block_rot][mode].BufH     = (8 * IRT_ROT_MEM_BANK_HEIGHT) >> mode;
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
    irt_cfg.Hb[e_irt_block_rot]     = IRT_ROT_MEM_BANK_HEIGHT;
    irt_cfg.Hb_mod[e_irt_block_rot] = irt_cfg.Hb[e_irt_block_rot] - 1;

    // for mesh memory
    irt_cfg.buf_format[e_irt_block_mesh]  = e_irt_buf_format_static;
    irt_cfg.buf_1b_mode[e_irt_block_mesh] = 0;
    irt_cfg.buf_select[e_irt_block_mesh]  = e_irt_buf_select_auto; // auto
    irt_cfg.buf_mode[e_irt_block_mesh]    = 3;
    for (uint8_t mode = 0; mode < IRT_RM_MODES; mode++) {
        irt_cfg.rm_cfg[e_irt_block_mesh][mode].BufW     = (2 * IRT_MESH_MEM_BANK_WIDTH) << mode;
        irt_cfg.rm_cfg[e_irt_block_mesh][mode].Buf_EpL  = (1 << mode);
        irt_cfg.rm_cfg[e_irt_block_mesh][mode].BufH     = (2 * IRT_MESH_MEM_BANK_HEIGHT) >> mode;
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
    irt_cfg.Hb[e_irt_block_mesh]     = IRT_MESH_MEM_BANK_HEIGHT;
    irt_cfg.Hb_mod[e_irt_block_mesh] = irt_cfg.Hb[e_irt_block_mesh] - 1;

    irt_cfg.lanczos_max_phases_h = LANCZOS_MAX_PHASES;
    irt_cfg.lanczos_max_phases_v = LANCZOS_MAX_PHASES;
}

void irt_par_init(rotation_par& rot_pars)
{

    rot_pars.MAX_PIXEL_WIDTH = IRT_PIXEL_WIDTH;
    rot_pars.MAX_COORD_WIDTH = IRT_COORD_WIDTH;
    sprintf(rot_pars.proj_order, "YRP");
    rot_pars.min_proc_rate   = IRT_ROT_MAX_PROC_SIZE;
    rot_pars.WEIGHT_PREC     = rot_pars.MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;
    rot_pars.WEIGHT_SHIFT    = 0; // rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
    rot_pars.COORD_PREC      = rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
    rot_pars.TOTAL_PREC      = rot_pars.WEIGHT_PREC + rot_pars.COORD_PREC;
    rot_pars.TOTAL_ROUND     = (1 << (rot_pars.TOTAL_PREC - 1));
    rot_pars.PROJ_NOM_PREC   = rot_pars.TOTAL_PREC;
    rot_pars.PROJ_DEN_PREC   = rot_pars.TOTAL_PREC + 10;
    rot_pars.PROJ_NOM_ROUND  = (1 << (rot_pars.PROJ_NOM_PREC - 1));
    rot_pars.PROJ_DEN_ROUND  = (1 << (rot_pars.PROJ_DEN_PREC - 1));
    rot_pars.rescale_Gx_prec = RESCALE_MAX_GX_PREC;
    // rot_pars.nudge = 0;  //TODO - till we connect centers
}

void irt_mesh_mems_create(irt_mesh_images& mesh_images)
{

    // init mesh arrays pointer
    mesh_images.mesh_image_full = new mesh_xy_fp32_meta*[OIMAGE_H];
    mesh_images.mesh_image_rel  = new mesh_xy_fp32*[OIMAGE_H];
    mesh_images.mesh_image_fp32 = new mesh_xy_fp32*[OIMAGE_H];
    mesh_images.mesh_image_fi16 = new mesh_xy_fi16*[OIMAGE_H];
    mesh_images.mesh_image_intr = new mesh_xy_fp64_meta*[OIMAGE_H];
    mesh_images.proj_image_full = new mesh_xy_fp64_meta*[OIMAGE_H];
    for (int row = 0; row < OIMAGE_H; row++) {
        (mesh_images.mesh_image_full)[row] = new mesh_xy_fp32_meta[OIMAGE_W];
        (mesh_images.mesh_image_rel)[row]  = new mesh_xy_fp32[OIMAGE_W];
        (mesh_images.mesh_image_fp32)[row] = new mesh_xy_fp32[OIMAGE_W];
        (mesh_images.mesh_image_fi16)[row] = new mesh_xy_fi16[OIMAGE_W];
        (mesh_images.mesh_image_intr)[row] = new mesh_xy_fp64_meta[OIMAGE_W];
        (mesh_images.proj_image_full)[row] = new mesh_xy_fp64_meta[OIMAGE_W];
    }
}

void irt_mesh_mems_delete(irt_mesh_images& mesh_images)
{

    for (int row = OIMAGE_H - 1; row >= 0; row--) {
        delete[] mesh_images.mesh_image_full[row];
        delete[] mesh_images.mesh_image_rel[row];
        delete[] mesh_images.mesh_image_fp32[row];
        delete[] mesh_images.mesh_image_fi16[row];
        delete[] mesh_images.mesh_image_intr[row];
        delete[] mesh_images.proj_image_full[row];

        mesh_images.mesh_image_full[row] = nullptr;
        mesh_images.mesh_image_rel[row]  = nullptr;
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
//#ifdef STANDALONE_ROTATOR

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
extern FILE* log_file;

// addr - address to the image
// read - do read
// write - do write
// wr_data - data to write to the memory
// rd_data - data read from the memory. Use 0 for padding.
// meta_in - should be equal to meta_out
void ext_memory_read(uint64_t          addr,
                     bus128B_struct&   rd_data,
                     meta_data_struct  meta_in,
                     meta_data_struct& meta_out,
                     uint16_t          lsb_pad,
                     uint16_t          msb_pad,
                     bool&             rd_data_valid)
{
    int      byte;
    uint32_t emem_size;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
    emem_size = irt_multi_image ? EMEM_SIZE_MT : EMEM_SIZE;
#else
    emem_size       = EMEM_SIZE;
#endif
    rd_data_valid = 1; // TODO - BACKPRESSURE INJECTION
    for (byte = 0; byte < 128; byte++) {
        if ((addr + byte) >= 0 && (addr + byte) <= emem_size - 1) { //(int) sizeof(ext_mem))
            rd_data.pix[byte] = ext_mem[addr + byte];
#if 1
            if (byte < lsb_pad) {
                rd_data.pix[byte] = 0x0;
            }
            if (byte > (127 - msb_pad)) { // 128-byte <  msb_pad    byte >  128 - msb_pad
                rd_data.pix[byte] = 0x0;
            }
#endif
        } else {
            rd_data.pix[byte] = 0xee; //(char)0xaa;
        }
        // IRT_TRACE("Read ext mem byte %d addr %lld, data %d pads %d %d\n", byte, addr + byte, rd_data.pix[byte],
        // lsb_pad, msb_pad);
        meta_out = meta_in;
    }
}
void ext_memory_write(uint64_t addr, bus128B_struct wr_data)
{
    int byte;

    uint32_t emem_size;
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
    emem_size = irt_multi_image ? EMEM_SIZE_MT : EMEM_SIZE;
#else
    emem_size       = EMEM_SIZE;
#endif
    // IRT_TRACE_UTILS("Ext mem: writting to addr %x emem_size %x\n", addr,emem_size);
    for (byte = 0; byte < 128; byte++) {
        if ((addr + byte) >= 0 && (addr + byte) <= emem_size - 1) {
            if (wr_data.en[byte] == 1) {
                ext_mem[addr + byte] = wr_data.pix[byte];
                // IRT_TRACE_UTILS("MEM_WRITE :: ext_mem[%x] = %x \n",addr + byte,ext_mem[addr + byte]);
            }
            // IRT_TRACE_UTILS("MEM_WRITE :: ext_mem[%x] = %x en %d \n",addr + byte,ext_mem[addr + byte],
            // wr_data.en[byte]);
        } else {
            // IRT_TRACE_UTILS("External memory of %d size write address error: %llx\n", emem_size, addr + byte);
        }
    }
}
void ext_memory_wrapper(uint64_t          addr,
                        bool              read,
                        uint8_t           write,
                        bus128B_struct    wr_data,
                        bus128B_struct&   rd_data,
                        meta_data_struct  meta_in,
                        meta_data_struct& meta_out,
                        uint16_t          lsb_pad,
                        uint16_t          msb_pad,
                        bool&             rd_data_valid)
{

    //---------------------------------------
    // IRT_TRACE_UTILS("ext_memory_wrapper :: addr %x read %d write %d\n", addr,read, write);
    rd_data_valid = 0;
    if (read) { // read
        ext_memory_read(addr, rd_data, meta_in, meta_out, lsb_pad, msb_pad, rd_data_valid);
    }
    //---------------------------------------
    //	IRT_TRACE_UTILS("ext_memory_wrapper: before\n");
    if (write == 1) { // normal write
        ext_memory_write(addr, wr_data);
    }
    //---------------------------------------
    // IRT_TRACE_UTILS("ext_memory_wrapper-write:: write= %d \n",write);
    if (write == 2) { // reduction write
        bus128B_struct rdata;
        bus128B_struct wdata;
        uint32_t       tr, tw;
        float          tf, tf1;
        memset(&rdata, 0, sizeof(bus128B_struct));
        ext_memory_read(addr, rdata, meta_in, meta_out, 0, 0, rd_data_valid);
        // IRT_TRACE_UTILS("------------\nreduction-write:: addr %x \n",addr);
        for (int i = 0; i < 32; i++) {
            tr = 0;
            tw = 0;
            for (int j = 0; j < 4; j++) {
                tr |= rdata.pix[i * 4 + j] << (j * 8);
                tw |= ((wr_data.en[i * 4 + j] == 1) ? wr_data.pix[i * 4 + j] : 0) << (j * 8);
            }
            tf = IRT_top::IRT_UTILS::conversion_bit_float(tr, e_irt_fp32, 0);
            // IRT_TRACE_UTILS("data %f + ",tf);
            tf1 = IRT_top::IRT_UTILS::conversion_bit_float(tw, e_irt_fp32, 0);
            tf += tf1;
            tw = IRT_top::IRT_UTILS::conversion_float_bit(tf, e_irt_fp32, 0, 0, 0, 0, 0, 0, 0);
            // IRT_TRACE_UTILS("%f = %f :: %x\n",tf1,tf,tw);
            for (int j = 0; j < 4; j++) {
                wdata.pix[i * 4 + j] = ((tw >> (j * 8)) & 0xff);
                wdata.en[i * 4 + j]  = 1;
            }
        }
        ext_memory_write(addr, wdata);
    }
#if 0
		IRT_TRACE("ext_memory_wrapper-write::Addr %x data : ",addr);
		for (byte = 127; byte >=0; byte--)
			IRT_TRACE("%02x", (unsigned int) ((unsigned char) wr_data.pix[byte]));
		IRT_TRACE("\nEN : ");
		for (byte = 127; byte >=0; byte--)
			IRT_TRACE("%x", wr_data.en[byte]);
		IRT_TRACE("\n");
#endif
}
#endif

// void ext_memory_reduction_write(uint64_t addr, bool read, uint8_t write, bus128B_struct wr_data, bus128B_struct
// &rd_data, meta_data_struct meta_in, meta_data_struct &meta_out, uint16_t lsb_pad, uint16_t msb_pad, bool&
// rd_data_valid) {
//   if (write==1) {
//      for (byte = 0; byte < 128; byte++) {
//         if ((addr + byte) >= 0 && (addr + byte) <= emem_size - 1) {
//            if (wr_data.en[byte] == 1) {
//               ext_mem[addr + byte] = wr_data.pix[byte];
//            }
//         } else {
//            IRT_TRACE_UTILS("External memory of %d size write address error: %llx\n", emem_size, addr + byte);
//         }
//      }
//   }
//}

#ifdef STANDALONE_ROTATOR
uint8_t irt_get_max_proc_size(Eirt_tranform_type irt_mode, Eirt_resamp_dtype_enum DTi)
{
    uint8_t proc_size;
    switch (irt_mode) {
        case e_irt_rotation:
        case e_irt_affine:
        case e_irt_projection: proc_size = 8; break;
        case e_irt_mesh: proc_size = 6; break;
        case e_irt_resamp_fwd: (DTi == e_irt_fp32) ? proc_size = 4 : proc_size = 8; break;
        case e_irt_resamp_bwd1: (DTi == e_irt_fp32) ? proc_size = 4 : proc_size = 6; break;
        case e_irt_resamp_bwd2: proc_size = 4; break;
        case e_irt_rescale: proc_size = 8; break;
    }
    return proc_size;
}

void irt_check_max_proc_size(Eirt_tranform_type irt_mode, uint8_t proc_size, Eirt_resamp_dtype_enum DTi)
{
    uint8_t max_proc_size = irt_get_max_proc_size(irt_mode, DTi);
    if (proc_size > max_proc_size) {
        IRT_TRACE_UTILS("IRTutils : SW proc size %d is not supported for irt_mode %s, provide in [1:%d] range\n",
                        proc_size,
                        irt_irt_mode_s[irt_mode],
                        max_proc_size);
        IRT_TRACE_TO_RES_UTILS(
            test_res,
            " was not run, SW proc size %d is not supported for irt_mode %s, provide in [1:%d] range\n",
            proc_size,
            irt_irt_mode_s[irt_mode],
            max_proc_size);
        IRT_CLOSE_FAILED_TEST(0);
    }
}

void irt_params_parsing(IRT_top* irt_top, uint8_t image, int argc, char* argv[], uint16_t i_width, uint16_t i_height)
{

    uint8_t task;

    // IRT_TRACE_UTILS("Parsing %d input parameters\n", argc);
    // num_of_images = 1;
    // super_irt = 0;

    irt_top->irt_cfg.bwd2_err_margin = 8; // TODO - REVIEW ONCE ULP CHECKS ENABLED
    for (int argcnt = 1; argcnt < argc; argcnt++) {
        // IRT_TRACE_UTILS("ARG %10s, value %s\n", argv[argcnt], argv[argcnt + 1]);
        // if (!strcmp(argv[argcnt], "-num"))			num_of_images = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-irt_h5_mode"))
            irt_h5_mode = 1;
        if (!strcmp(argv[argcnt], "-read_iimage_BufW"))
            read_iimage_BufW = 1;
        if (!strcmp(argv[argcnt], "-read_mimage_BufW"))
            read_mimage_BufW = 1;
        if (!strcmp(argv[argcnt], "-hl_only"))
            hl_only = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-descr_gen_only"))
            descr_gen_only = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-bmp_wr_order"))
            irt_bmp_wr_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-bmp_rd_order"))
            irt_bmp_rd_order = (Eirt_bmp_order)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-flow_mode"))
            irt_top->irt_cfg.flow_mode = (Eirt_flow_type)atoi(argv[argcnt + 1]);

        if (!strcmp(argv[argcnt], "-rot_buf_format"))
            irt_top->irt_cfg.buf_format[e_irt_block_rot] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-rot_buf_select"))
            irt_top->irt_cfg.buf_select[e_irt_block_rot] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-rot_buf_mode"))
            irt_top->irt_cfg.buf_mode[e_irt_block_rot] = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-rot_buf_1b_mode"))
            irt_top->irt_cfg.buf_1b_mode[e_irt_block_rot] = (bool)atoi(argv[argcnt + 1]);

        if (!strcmp(argv[argcnt], "-mesh_buf_format"))
            irt_top->irt_cfg.buf_format[e_irt_block_mesh] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-mesh_buf_select"))
            irt_top->irt_cfg.buf_select[e_irt_block_mesh] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-mesh_buf_mode"))
            irt_top->irt_cfg.buf_mode[e_irt_block_mesh] = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-mesh_buf_1b_mode"))
            irt_top->irt_cfg.buf_1b_mode[e_irt_block_mesh] = (bool)atoi(argv[argcnt + 1]);

        if (!strcmp(argv[argcnt], "-dbg_mode"))
            irt_top->irt_cfg.debug_mode = (Eirt_debug_type)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-print_log"))
            print_log_file = 1;
        if (!strcmp(argv[argcnt], "-print_images"))
            print_out_files = 1;
        if (!strcmp(argv[argcnt], "-warp_gen_mode"))
            warp_gen_mode = (int)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-range_bound"))
            range_bound = (int)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-bwd2_err_margin"))
            irt_top->irt_cfg.bwd2_err_margin = (int)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-verbose"))
            max_log_trace_level = (int)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-PHh"))
            irt_top->irt_cfg.lanczos_max_phases_h = (int)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-PHv"))
            irt_top->irt_cfg.lanczos_max_phases_v = (int)atoi(argv[argcnt + 1]);
        //---------------------------------------
        // random C seed input. If not provided, C seeded TIME based
        if (!strcmp(argv[argcnt], "-cseed")) {
            cseed = (int)atoi(argv[argcnt + 1]);
        } else {
            srand((unsigned)time(0));
        }
    }

    for (uint8_t plane = 0; plane < PLANES; plane++) {
        task = image * PLANES + plane;

        // setting defaults
        irt_top->rot_pars[task].MAX_PIXEL_WIDTH = IRT_PIXEL_WIDTH;
        irt_top->rot_pars[task].MAX_COORD_WIDTH = IRT_COORD_WIDTH;
        irt_top->rot_pars[task].COORD_PREC      = irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
        irt_top->rot_pars[task].COORD_ROUND     = (1 << (irt_top->rot_pars[task].COORD_PREC - 1));
        irt_top->rot_pars[task].WEIGHT_PREC     = irt_top->rot_pars[task].MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT;
        irt_top->rot_pars[task].WEIGHT_SHIFT    = 0; // irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT;
        irt_top->rot_pars[task].TOTAL_PREC  = irt_top->rot_pars[task].WEIGHT_PREC + irt_top->rot_pars[task].COORD_PREC;
        irt_top->rot_pars[task].TOTAL_ROUND = (1 << (irt_top->rot_pars[task].TOTAL_PREC - 1));
        irt_top->rot_pars[task].rot_prec_auto_adj   = 0;
        irt_top->rot_pars[task].mesh_prec_auto_adj  = 0;
        irt_top->rot_pars[task].oimg_auto_adj       = 0;
        irt_top->rot_pars[task].oimg_auto_adj_rate  = 0;
        irt_top->rot_pars[task].use_Si_delta_margin = 0;

        irt_top->rot_pars[task].PROJ_NOM_PREC  = irt_top->rot_pars[task].TOTAL_PREC;
        irt_top->rot_pars[task].PROJ_DEN_PREC  = irt_top->rot_pars[task].TOTAL_PREC + 10;
        irt_top->rot_pars[task].PROJ_NOM_ROUND = (1 << (irt_top->rot_pars[task].PROJ_NOM_PREC - 1));
        irt_top->rot_pars[task].PROJ_DEN_ROUND = (1 << (irt_top->rot_pars[task].PROJ_DEN_PREC - 1));

        irt_top->irt_desc[task].image_par[IIMAGE].addr_start =
            (uint64_t)IIMAGE_REGION_START + (uint64_t)plane * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL;
        irt_top->irt_desc[task].image_par[OIMAGE].addr_start =
            (uint64_t)OIMAGE_REGION_START + (uint64_t)task * OIMAGE_W * OIMAGE_H * BYTEs4PIXEL;
        irt_top->irt_desc[task].image_par[MIMAGE].addr_start =
            (irt_multi_image ? (uint64_t)MIMAGE_REGION_START_MT : (uint64_t)MIMAGE_REGION_START) +
            (uint64_t)image * OIMAGE_W * OIMAGE_H * BYTEs4MESH;
        irt_top->irt_desc[task].image_par[GIMAGE].addr_start =
            (irt_multi_image ? (uint64_t)GIMAGE_REGION_START_MT : (uint64_t)GIMAGE_REGION_START) +
            (uint64_t)image * OIMAGE_W * OIMAGE_H * BYTEs4PIXEL;

        irt_top->irt_desc[task].image_par[IIMAGE].H = i_height;
        irt_top->irt_desc[task].image_par[IIMAGE].W = i_width;
        irt_top->irt_desc[task].image_par[OIMAGE].H = 64;
        irt_top->irt_desc[task].image_par[OIMAGE].W = 128;
        irt_top->irt_desc[task].image_par[MIMAGE].H = 64;
        irt_top->irt_desc[task].image_par[MIMAGE].W = 128;
        irt_top->irt_desc[task].image_par[GIMAGE].H = 64;
        irt_top->irt_desc[task].image_par[GIMAGE].W = 128;
        irt_top->rot_pars[task].Pwi                 = 8;
        irt_top->rot_pars[task].Pwo                 = 8;
        irt_top->rot_pars[task].Ppi                 = 0;
        irt_top->irt_desc[task].Ppo                 = 0;

        irt_top->irt_desc[task].int_mode = e_irt_int_bilinear; // bilinear
        irt_top->irt_desc[task].irt_mode = e_irt_rotation; // rotation
        irt_top->irt_desc[task].crd_mode = e_irt_crd_mode_fixed; // fixed point

        irt_top->rot_pars[task].irt_angles[e_irt_angle_rot] = 30;

        sprintf(irt_top->rot_pars[task].affine_mode, "R");
        irt_top->rot_pars[task].reflection_mode               = 0;
        irt_top->rot_pars[task].shear_mode                    = 0;
        irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_x] = 90;
        irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_y] = 90;
        irt_top->rot_pars[task].Sx                            = 1;
        irt_top->rot_pars[task].Sy                            = 1;

        irt_top->rot_pars[task].proj_mode = 0;
        sprintf(irt_top->rot_pars[task].proj_order, "YRP");
        irt_top->rot_pars[task].irt_angles[e_irt_angle_roll]  = 50.0;
        irt_top->rot_pars[task].irt_angles[e_irt_angle_pitch] = -60.0;
        irt_top->rot_pars[task].irt_angles[e_irt_angle_yaw]   = 60.0;
        irt_top->rot_pars[task].proj_Zd                       = 400.0;
        irt_top->rot_pars[task].proj_Wd                       = 1000.0;

        irt_top->irt_desc[task].oimage_line_wr_format = 0;
        irt_top->irt_desc[task].bg                    = 0x0;
        irt_top->irt_desc[task].bg_mode               = e_irt_bg_prog_value;

        irt_top->rot_pars[task].dist_x = 1;
        irt_top->rot_pars[task].dist_y = 1;
        irt_top->rot_pars[task].dist_r = 0;

        irt_top->rot_pars[task].mesh_mode           = 0;
        irt_top->rot_pars[task].mesh_order          = 0;
        irt_top->rot_pars[task].mesh_Sh             = 1.0;
        irt_top->rot_pars[task].mesh_Sv             = 1.0;
        irt_top->irt_desc[task].mesh_format         = e_irt_mesh_flex;
        irt_top->irt_desc[task].mesh_point_location = 0;
        irt_top->irt_desc[task].mesh_rel_mode       = e_irt_mesh_absolute;

        irt_top->irt_desc[task].rate_mode = e_irt_rate_fixed;
        irt_top->irt_desc[task].proc_size = IRT_ROT_MAX_PROC_SIZE;
        irt_top->rot_pars[task].proc_auto = 0;

        //---------------------------------------
        // resampler fields default values
        irt_top->irt_desc[task].image_par[OIMAGE].S = irt_top->irt_desc[task].image_par[OIMAGE].W;
        IRT_TRACE_UTILS("1. So %d Wo %d\n",
                        irt_top->irt_desc[task].image_par[OIMAGE].S,
                        irt_top->irt_desc[task].image_par[OIMAGE].W);
        irt_top->irt_desc[task].image_par[IIMAGE].DataType = static_cast<Eirt_resamp_dtype_enum>(4);
        irt_top->irt_desc[task].image_par[OIMAGE].DataType = static_cast<Eirt_resamp_dtype_enum>(4);
        irt_top->irt_desc[task].image_par[MIMAGE].DataType = static_cast<Eirt_resamp_dtype_enum>(4);
        irt_top->irt_desc[task].image_par[GIMAGE].DataType = static_cast<Eirt_resamp_dtype_enum>(4);
        //---------------------------------------
        irt_top->irt_desc[task].warp_stride = 0;
        irt_top->rot_pars[task].filter_type = e_irt_lanczos2;

        // parsing main parameters
        for (int argcnt = 1; argcnt < argc; argcnt++) {

            if (!strcmp(argv[argcnt], "-pw"))
                irt_top->rot_pars[task].MAX_PIXEL_WIDTH = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-cw"))
                irt_top->rot_pars[task].MAX_COORD_WIDTH = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-ws"))
                irt_top->rot_pars[task].WEIGHT_SHIFT = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-rot_prec_auto_adj"))
                irt_top->rot_pars[task].rot_prec_auto_adj = 1;
            if (!strcmp(argv[argcnt], "-mesh_prec_auto_adj"))
                irt_top->rot_pars[task].mesh_prec_auto_adj = 1;
            if (!strcmp(argv[argcnt], "-oimg_auto_adj"))
                irt_top->rot_pars[task].oimg_auto_adj = 1;
            if (!strcmp(argv[argcnt], "-oimg_auto_adj_rate"))
                irt_top->rot_pars[task].oimg_auto_adj_rate = (float)atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-use_Si_delta_margin"))
                irt_top->rot_pars[task].use_Si_delta_margin = 1;
            if (!strcmp(argv[argcnt], "-Ho"))
                irt_top->irt_desc[task].image_par[OIMAGE].H = (uint16_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Wo"))
                irt_top->irt_desc[task].image_par[OIMAGE].W = (uint16_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Pwo"))
                irt_top->rot_pars[task].Pwo = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Ppo"))
                irt_top->irt_desc[task].Ppo = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Hi"))
                irt_top->irt_desc[task].image_par[IIMAGE].H = (uint16_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Wi"))
                irt_top->irt_desc[task].image_par[IIMAGE].W = (uint16_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Pwi"))
                irt_top->rot_pars[task].Pwi = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Ppi"))
                irt_top->rot_pars[task].Ppi = (uint8_t)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-int_mode"))
                irt_top->irt_desc[task].int_mode = (Eirt_int_mode_type)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-irt_mode"))
                irt_top->irt_desc[task].irt_mode = (Eirt_tranform_type)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-crd_mode"))
                irt_top->irt_desc[task].crd_mode = (Eirt_coord_mode_type)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-rot_angle"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_rot] = (float)atof(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-aff_mode"))
                sprintf(irt_top->rot_pars[task].affine_mode, "%s", argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-rfl_mode"))
                irt_top->rot_pars[task].reflection_mode = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-shr_mode"))
                irt_top->rot_pars[task].shear_mode = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-shr_xang"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_x] = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-shr_yang"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_shr_y] = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Sx"))
                irt_top->rot_pars[task].Sx = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Sy"))
                irt_top->rot_pars[task].Sy = atof(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-prj_mode"))
                irt_top->rot_pars[task].proj_mode = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-prj_order"))
                sprintf(irt_top->rot_pars[task].proj_order, "%s", argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-roll"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_roll] = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-pitch"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_pitch] = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-yaw"))
                irt_top->rot_pars[task].irt_angles[e_irt_angle_yaw] = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Zd"))
                irt_top->rot_pars[task].proj_Zd = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Wd"))
                irt_top->rot_pars[task].proj_Wd = atof(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-mesh_mode"))
                irt_top->rot_pars[task].mesh_mode = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_order"))
                irt_top->rot_pars[task].mesh_order = (bool)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_Sh"))
                irt_top->rot_pars[task].mesh_Sh = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_Sv"))
                irt_top->rot_pars[task].mesh_Sv = atof(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-mesh_format"))
                irt_top->irt_desc[task].mesh_format = (Eirt_mesh_format_type)atoi(argv[argcnt + 1]);
            //---------------------------------------
            Eirt_mesh_format_type mesh_format_chk = irt_top->irt_desc[task].mesh_format;
            if (mesh_format_chk == e_irt_mesh_reserv0 || mesh_format_chk == e_irt_mesh_reserv2 ||
                mesh_format_chk == e_irt_mesh_reserv3) {
                IRT_TRACE_UTILS("Desc gen error: Mesh Image Date format Unsupported :: mesh_format = %0d :: Valid "
                                "mesh_format = 1 [Flex] 4 [FP32]\n",
                                (int)mesh_format_chk);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       "Desc gen error: Mesh Image Date format Unsupported :: mesh_format = %0d :: "
                                       "Valid mesh_format = 1 [Flex] 4 [FP32]\n",
                                       (int)mesh_format_chk);
                IRT_CLOSE_FAILED_TEST(0);
            }
            //---------------------------------------
            if (!strcmp(argv[argcnt], "-mesh_point"))
                irt_top->irt_desc[task].mesh_point_location = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_rel"))
                irt_top->irt_desc[task].mesh_rel_mode = (Eirt_mesh_rel_mode_type)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-dist_x"))
                irt_top->rot_pars[task].dist_x = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-dist_y"))
                irt_top->rot_pars[task].dist_y = atof(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-dist_r"))
                irt_top->rot_pars[task].dist_r = atof(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-proc_auto"))
                irt_top->rot_pars[task].proc_auto = (bool)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-lwf"))
                irt_top->irt_desc[task].oimage_line_wr_format = (bool)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-bg_mode"))
                irt_top->irt_desc[task].bg_mode = (Eirt_bg_mode_type)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-bg"))
                irt_top->irt_desc[task].bg = (uint16_t)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-rot_buf_format"))
                irt_top->rot_pars[task].buf_format[e_irt_block_rot] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-rot_buf_select"))
                irt_top->rot_pars[task].buf_select[e_irt_block_rot] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-rot_buf_mode"))
                irt_top->rot_pars[task].buf_mode[e_irt_block_rot] = (uint8_t)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-mesh_buf_format"))
                irt_top->rot_pars[task].buf_format[e_irt_block_mesh] = (Eirt_buf_format)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_buf_select"))
                irt_top->rot_pars[task].buf_select[e_irt_block_mesh] = (Eirt_buf_select)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-mesh_buf_mode"))
                irt_top->rot_pars[task].buf_mode[e_irt_block_mesh] = (uint8_t)atoi(argv[argcnt + 1]);

            //---------------------------------------
            // RESAMP ARGUMENTS
            if (!strcmp(argv[argcnt], "-So"))
                irt_top->irt_desc[task].image_par[OIMAGE].S = (uint16_t)atoi(argv[argcnt + 1]);
            // if (!strcmp(argv[argcnt], "-resamp_dtype"))		irt_top->rot_pars[task].resamp_dtype =
            // (Eirt_resamp_dtype_enum)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-DTi"))
                irt_top->irt_desc[task].image_par[IIMAGE].DataType = (Eirt_resamp_dtype_enum)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-DTo"))
                irt_top->irt_desc[task].image_par[OIMAGE].DataType = (Eirt_resamp_dtype_enum)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-DTm"))
                irt_top->irt_desc[task].image_par[MIMAGE].DataType = (Eirt_resamp_dtype_enum)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-DTg"))
                irt_top->irt_desc[task].image_par[GIMAGE].DataType = (Eirt_resamp_dtype_enum)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-Ws"))
                irt_top->irt_desc[task].warp_stride = (uint8_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-FltType"))
                irt_top->rot_pars[task].filter_type = (filter_type_t)atoi(argv[argcnt + 1]);
            if (!strcmp(argv[argcnt], "-resize_bli_grad_en"))
                irt_top->irt_desc[task].resize_bli_grad_en = (bool)atoi(argv[argcnt + 1]);
            // if (task == 0 && argv[argcnt][0] == '-') {
            // IRT_TRACE_UTILS("Switch %10s, value %s\n", argv[argcnt], argv[argcnt + 1]);
            //}

            //---------------------------------------
            if (!strcmp(argv[argcnt], "-rate_mode"))
                irt_top->irt_desc[task].rate_mode = (Eirt_rate_type)atoi(argv[argcnt + 1]);

            if (!strcmp(argv[argcnt], "-proc_size")) {
                irt_top->irt_desc[task].proc_size = (uint8_t)atoi(argv[argcnt + 1]);
                irt_check_max_proc_size(irt_top->irt_desc[task].irt_mode,
                                        irt_top->irt_desc[task].proc_size,
                                        irt_top->irt_desc[task].image_par[IIMAGE].DataType);
            } else {
                irt_top->irt_desc[task].proc_size = irt_get_max_proc_size(
                    irt_top->irt_desc[task].irt_mode, irt_top->irt_desc[task].image_par[IIMAGE].DataType);
            }
        }
        //---------------------------------------
        irt_top->irt_desc[task].image_par[MIMAGE].H = irt_top->irt_desc[task].image_par[OIMAGE].H;
        irt_top->irt_desc[task].image_par[MIMAGE].W = irt_top->irt_desc[task].image_par[OIMAGE].W;

        if (irt_top->irt_desc[task].irt_mode > 3) {
            irt_top->irt_desc[task].image_par[MIMAGE].S = irt_top->irt_desc[task].image_par[OIMAGE].S;
        } else {
            irt_top->irt_desc[task].image_par[MIMAGE].S = irt_top->irt_desc[task].image_par[MIMAGE].W;
        }
        //---------------------------------------
        irt_top->irt_desc[task].image_par[GIMAGE].H = irt_top->irt_desc[task].image_par[OIMAGE].H;
        irt_top->irt_desc[task].image_par[GIMAGE].W = irt_top->irt_desc[task].image_par[OIMAGE].W;

        if (irt_top->irt_desc[task].irt_mode > 3) {
            irt_top->irt_desc[task].image_par[GIMAGE].S = irt_top->irt_desc[task].image_par[OIMAGE].S;
        } else {
            irt_top->irt_desc[task].image_par[GIMAGE].S = irt_top->irt_desc[task].image_par[GIMAGE].W;
        }
        //---------------------------------------

        // calculating image centers as default
        irt_top->irt_desc[task].image_par[OIMAGE].Yc = irt_top->irt_desc[task].image_par[OIMAGE].H - 1;
        irt_top->irt_desc[task].image_par[OIMAGE].Xc = irt_top->irt_desc[task].image_par[OIMAGE].W - 1;
        irt_top->irt_desc[task].image_par[IIMAGE].Yc = irt_top->irt_desc[task].image_par[IIMAGE].H - 1;
        irt_top->irt_desc[task].image_par[IIMAGE].Xc = irt_top->irt_desc[task].image_par[IIMAGE].W - 1;

        // parsing image centers
        for (int argcnt = 1; argcnt < argc; argcnt++) {
            if (!strcmp(argv[argcnt], "-Yco"))
                irt_top->irt_desc[task].image_par[OIMAGE].Yc = (int16_t)(atof(argv[argcnt + 1]) * 2);
            if (!strcmp(argv[argcnt], "-Xco"))
                irt_top->irt_desc[task].image_par[OIMAGE].Xc = (int16_t)(atof(argv[argcnt + 1]) * 2);
            if (!strcmp(argv[argcnt], "-Yci"))
                irt_top->irt_desc[task].image_par[IIMAGE].Yc = (int16_t)(atof(argv[argcnt + 1]) * 2);
            if (!strcmp(argv[argcnt], "-Xci"))
                irt_top->irt_desc[task].image_par[IIMAGE].Xc = (int16_t)(atof(argv[argcnt + 1]) * 2);
        }

        irt_top->rot_pars[task].WEIGHT_PREC =
            irt_top->rot_pars[task].MAX_PIXEL_WIDTH +
            IRT_WEIGHT_PREC_EXT; // +1 because of 0.5 error, +2 because of interpolation, -1 because of weigths
                                 // multiplication
        irt_top->rot_pars[task].COORD_PREC =
            irt_top->rot_pars[task].MAX_COORD_WIDTH + IRT_COORD_PREC_EXT; // +1 because of polynom in Xi/Yi calculation
        irt_top->rot_pars[task].COORD_ROUND = (1 << (irt_top->rot_pars[task].COORD_PREC - 1));
        irt_top->rot_pars[task].TOTAL_PREC  = irt_top->rot_pars[task].COORD_PREC + irt_top->rot_pars[task].WEIGHT_PREC;
        irt_top->rot_pars[task].TOTAL_ROUND = (1 << (irt_top->rot_pars[task].TOTAL_PREC - 1));
        irt_top->rot_pars[task].PROJ_NOM_PREC  = irt_top->rot_pars[task].TOTAL_PREC;
        irt_top->rot_pars[task].PROJ_DEN_PREC  = (irt_top->rot_pars[task].TOTAL_PREC + 10);
        irt_top->rot_pars[task].PROJ_NOM_ROUND = (1 << (irt_top->rot_pars[task].PROJ_NOM_PREC - 1));
        irt_top->rot_pars[task].PROJ_DEN_ROUND = (1 << (irt_top->rot_pars[task].PROJ_DEN_PREC - 1));

        if (irt_top->irt_desc[task].irt_mode <= 3) {
            irt_top->irt_desc[task].image_par[OIMAGE].Ps = irt_top->rot_pars[task].Pwo <= 8 ? 0 : 1;
        } else {
            switch (irt_top->irt_desc[task].image_par[OIMAGE].DataType) {
                case e_irt_int8: irt_top->irt_desc[task].image_par[OIMAGE].Ps = 0; break;
                case e_irt_int16: irt_top->irt_desc[task].image_par[OIMAGE].Ps = 1; break;
                case e_irt_fp16: irt_top->irt_desc[task].image_par[OIMAGE].Ps = 1; break;
                case e_irt_bfp16: irt_top->irt_desc[task].image_par[OIMAGE].Ps = 1; break;
                case e_irt_fp32: irt_top->irt_desc[task].image_par[OIMAGE].Ps = 2; break;
            }
        }

        if (irt_top->irt_desc[task].irt_mode <= 3) {
            irt_top->irt_desc[task].image_par[IIMAGE].Ps = irt_top->rot_pars[task].Pwi <= 8 ? 0 : 1;
        } else {
            switch (irt_top->irt_desc[task].image_par[IIMAGE].DataType) {
                case e_irt_int8: irt_top->irt_desc[task].image_par[IIMAGE].Ps = 0; break;
                case e_irt_int16: irt_top->irt_desc[task].image_par[IIMAGE].Ps = 1; break;
                case e_irt_fp16: irt_top->irt_desc[task].image_par[IIMAGE].Ps = 1; break;
                case e_irt_bfp16: irt_top->irt_desc[task].image_par[IIMAGE].Ps = 1; break;
                case e_irt_fp32: irt_top->irt_desc[task].image_par[IIMAGE].Ps = 2; break;
            }
        }

        if (irt_top->irt_desc[task].irt_mode <= 3) {
            irt_top->irt_desc[task].image_par[MIMAGE].Ps =
                irt_top->irt_desc[task].mesh_format == e_irt_mesh_flex ? 2 : 3;
        } else {
            switch (irt_top->irt_desc[task].image_par[MIMAGE].DataType) {
                case e_irt_int8: irt_top->irt_desc[task].image_par[MIMAGE].Ps = 1; break;
                case e_irt_int16: irt_top->irt_desc[task].image_par[MIMAGE].Ps = 2; break;
                case e_irt_fp16: irt_top->irt_desc[task].image_par[MIMAGE].Ps = 2; break;
                case e_irt_bfp16: irt_top->irt_desc[task].image_par[MIMAGE].Ps = 2; break;
                case e_irt_fp32: irt_top->irt_desc[task].image_par[MIMAGE].Ps = 3; break;
            }
            if (irt_top->irt_desc[task].irt_mode == e_irt_rescale) {
                irt_top->irt_desc[task].image_par[MIMAGE].Ps--;
                // IRT_TRACE_UTILS("->>>>inside rescale Ps %d\n",irt_top->irt_desc[task].image_par[MIMAGE].Ps);
            }
        }
        if (irt_top->irt_desc[task].irt_mode > 3) {
            switch (irt_top->irt_desc[task].image_par[GIMAGE].DataType) {
                case e_irt_int8: irt_top->irt_desc[task].image_par[GIMAGE].Ps = 0; break;
                case e_irt_int16: irt_top->irt_desc[task].image_par[GIMAGE].Ps = 1; break;
                case e_irt_fp16: irt_top->irt_desc[task].image_par[GIMAGE].Ps = 1; break;
                case e_irt_bfp16: irt_top->irt_desc[task].image_par[GIMAGE].Ps = 1; break;
                case e_irt_fp32: irt_top->irt_desc[task].image_par[GIMAGE].Ps = 2; break;
            }
        }
        //---------------------------------------
        irt_top->irt_desc[task].image_par[IIMAGE].PsBytes = (1 << irt_top->irt_desc[task].image_par[IIMAGE].Ps);
        irt_top->irt_desc[task].image_par[OIMAGE].PsBytes = (1 << irt_top->irt_desc[task].image_par[OIMAGE].Ps);
        irt_top->irt_desc[task].image_par[MIMAGE].PsBytes = (1 << irt_top->irt_desc[task].image_par[MIMAGE].Ps);
        irt_top->irt_desc[task].image_par[GIMAGE].PsBytes = (1 << irt_top->irt_desc[task].image_par[GIMAGE].Ps);
        //---------------------------------------
        irt_top->irt_desc[task].Msi = ((1 << irt_top->rot_pars[task].Pwi) - 1) << irt_top->rot_pars[task].Ppi;
        irt_top->rot_pars[task].bli_shift_fix =
            (2 * irt_top->rot_pars[task].TOTAL_PREC) -
            (irt_top->rot_pars[task].Pwo - (irt_top->rot_pars[task].Pwi + irt_top->rot_pars[task].Ppi)) -
            1; // bi-linear interpolation shift
        irt_top->irt_desc[task].bli_shift =
            irt_top->rot_pars[task].Pwo - (irt_top->rot_pars[task].Pwi + irt_top->rot_pars[task].Ppi);
        irt_top->irt_desc[task].MAX_VALo =
            (1 << irt_top->rot_pars[task].Pwo) - 1; // output pixel max value, equal to 2^pixel_width - 1

        if (task == 0) {
            IRT_TRACE_UTILS("-----------------------------------\n");
            for (int argcnt = 0; argcnt < argc; argcnt++) {
                IRT_TRACE_TO_RES_UTILS(test_res, "%s ", argv[argcnt]);
                IRT_TRACE_UTILS("%s ", argv[argcnt]);
            }
            IRT_TRACE_TO_RES_UTILS(test_res, " - ");
            IRT_TRACE_UTILS("\n");
            IRT_TRACE_UTILS("-----------------------------------\n");
        }

        // irt_rotation_memory_fit_check(irt_top->irt_cfg, irt_top->rot_pars[task], irt_top->irt_desc[task], task);

        if (irt_top->irt_cfg.flow_mode == e_irt_flow_wCFIFO_fixed_adaptive_2x2 &&
                irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_wxh ||
            irt_top->irt_cfg.flow_mode == e_irt_flow_wCFIFO_fixed_adaptive_wxh &&
                irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_2x2 ||
            irt_top->irt_cfg.flow_mode == e_irt_flow_nCFIFO_fixed_adaptive_wxh &&
                irt_top->irt_desc[task].rate_mode == e_irt_rate_adaptive_2x2) {
            IRT_TRACE_UTILS("%s flow and %s rate control are not supported together\n",
                            irt_flow_mode_s[irt_top->irt_cfg.flow_mode],
                            irt_rate_mode_s[irt_top->irt_desc[task].rate_mode]);
            IRT_TRACE_TO_RES_UTILS(test_res,
                                   " was not run, %s flow and %s rate control are not supported together\n",
                                   irt_flow_mode_s[irt_top->irt_cfg.flow_mode],
                                   irt_rate_mode_s[irt_top->irt_desc[task].rate_mode]);
            IRT_CLOSE_FAILED_TEST(0);
        }
    }
    if (!(irt_top->irt_cfg.lanczos_max_phases_h == 1 || irt_top->irt_cfg.lanczos_max_phases_h == 2 ||
          irt_top->irt_cfg.lanczos_max_phases_h == 4 || irt_top->irt_cfg.lanczos_max_phases_h == 8 ||
          irt_top->irt_cfg.lanczos_max_phases_h == 16 || irt_top->irt_cfg.lanczos_max_phases_h == 32 ||
          irt_top->irt_cfg.lanczos_max_phases_h == 64 || irt_top->irt_cfg.lanczos_max_phases_h == 128)) {
        IRT_TRACE_TO_RES_UTILS(
            test_res, "Rescale H phase is %d which is not in power of 2\n", irt_top->irt_cfg.lanczos_max_phases_h);
        IRT_CLOSE_FAILED_TEST(0);
    }
    if (!(irt_top->irt_cfg.lanczos_max_phases_v == 1 || irt_top->irt_cfg.lanczos_max_phases_v == 2 ||
          irt_top->irt_cfg.lanczos_max_phases_v == 4 || irt_top->irt_cfg.lanczos_max_phases_v == 8 ||
          irt_top->irt_cfg.lanczos_max_phases_v == 16 || irt_top->irt_cfg.lanczos_max_phases_v == 32 ||
          irt_top->irt_cfg.lanczos_max_phases_v == 64 || irt_top->irt_cfg.lanczos_max_phases_v == 128)) {
        IRT_TRACE_TO_RES_UTILS(
            test_res, "Rescale V phase is %d which is not in power of 2\n", irt_top->irt_cfg.lanczos_max_phases_v);
        IRT_CLOSE_FAILED_TEST(0);
    }
    //}
    if (test_file_flag) {
        if (irt_top->rot_pars[image].Pwi != 8 || irt_top->rot_pars[image].Ppi != 0) {
            IRT_TRACE_UTILS("Pwi = %d and Ppi = %d for image %d are not supported together, must be 8 and 0 in "
                            "multi-descriptor test\n",
                            irt_top->rot_pars[image].Pwi,
                            irt_top->rot_pars[image].Ppi,
                            image);
            IRT_TRACE_TO_RES_UTILS(test_res,
                                   " was not run, Pwi = %d and Ppi = %d for image %d are not supported together, must "
                                   "be 8 and 0 in multi-descriptor test",
                                   irt_top->rot_pars[image].Pwi,
                                   irt_top->rot_pars[image].Ppi,
                                   image);
            IRT_CLOSE_FAILED_TEST(0);
        }
    }

    IRT_TRACE_UTILS("--->> max_log_trace_level = %d\n", max_log_trace_level);
    IRT_TRACE_UTILS("--->> RANGE BOUND  = %d\n", range_bound);
}

#endif

#if defined(STANDALONE_ROTATOR) || defined(CREATE_IMAGE_DUMPS) || defined(RUN_WITH_SV)
void generate_bmp_header(FILE* file, uint32_t width, uint32_t height, uint32_t row_padded, uint8_t planes)
{

    uint32_t filesize = 54 + row_padded * height;

    uint8_t bmpfilehdr[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    uint8_t bmpinfohdr[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0,
                              0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0};

    bmpfilehdr[2] = (uint8_t)(filesize);
    bmpfilehdr[3] = (uint8_t)(filesize >> 8);
    bmpfilehdr[4] = (uint8_t)(filesize >> 16);
    bmpfilehdr[5] = (uint8_t)(filesize >> 24);

    bmpinfohdr[irt_dib_header_image_width]      = (uint8_t)(width);
    bmpinfohdr[irt_dib_header_image_width + 1]  = (uint8_t)(width >> 8);
    bmpinfohdr[irt_dib_header_image_width + 2]  = (uint8_t)(width >> 16);
    bmpinfohdr[irt_dib_header_image_width + 3]  = (uint8_t)(width >> 24);
    bmpinfohdr[irt_dib_header_image_height]     = (uint8_t)(height);
    bmpinfohdr[irt_dib_header_image_height + 1] = (uint8_t)(height >> 8);
    bmpinfohdr[irt_dib_header_image_height + 2] = (uint8_t)(height >> 16);
    bmpinfohdr[irt_dib_header_image_height + 3] = (uint8_t)(height >> 24);
    bmpinfohdr[irt_dib_header_image_bpp]        = 8 * planes;
    bmpinfohdr[irt_dib_header_bitmap_size]      = (uint8_t)(row_padded * height);
    bmpinfohdr[irt_dib_header_bitmap_size + 1]  = (uint8_t)((row_padded * height) >> 8);

    fwrite(bmpfilehdr, 1, 14, file);
    fwrite(bmpinfohdr, 1, 40, file);
}

void WriteBMP(IRT_top*    irt_top,
              uint8_t     image,
              uint8_t     planes,
              char*       filename,
              uint32_t    width,
              uint32_t    height,
              uint64_t*** out_image)
{
    uint8_t desc = image * planes;
    FILE*   f    = fopen(filename, "wb");

    // if(f == NULL)
    //  throw "Argument Exception";

    IRT_TRACE_UTILS("Writing file %s %dx%d\n", filename, width, height);

    uint32_t row_padded = (width * 3 + 3) & (~3);
    uint8_t* data       = new uint8_t[row_padded];

    generate_bmp_header(f, width, height, row_padded, 3);

    uint32_t row = 0;

    // for(int32_t i = height-1; i >= 0; i--) //image array is swapped in BMP file
    for (uint32_t i = 0; i < height; i++) { // image array is swapped in BMP file
        if (irt_bmp_wr_order == e_irt_bmp_H20)
            row = height - 1 - i;
        else
            row = i;

        for (uint32_t j = 0; j < width; j++) {
            for (uint8_t idx = 0; idx < planes; idx++) {
                data[j * 3 + idx] = (uint8_t)(out_image[idx][row][j] >>
                                              (irt_top->rot_pars[desc].Pwo - 8 + irt_top->irt_desc[desc].Ppo));
            }
#if 1
            if (planes < 3) {
                for (uint8_t idx = planes; idx < 3; idx++) {
                    data[j * 3 + idx] = (uint8_t)(out_image[0][row][j] >>
                                                  (irt_top->rot_pars[desc].Pwo - 8 + irt_top->irt_desc[desc].Ppo));
                }
            }
#endif
        }
        fwrite(data, sizeof(uint8_t), row_padded, f);
    }

    fclose(f);
}

#endif

#if defined(STANDALONE_ROTATOR)
void ReadBMP(IRT_top* irt_top, char* filename, uint32_t& width, uint32_t& height, int argc, char* argv[])
{
    uint8_t pixel_width = 8, pixel_padding = 0, irt_mode = 0;
    FILE*   f = fopen(filename, "rb");

    // printf ("ReadBMP: %s, %x\n", filename, f);
    // if(f == NULL)
    //  throw "Argument Exception";

    uint8_t bmpfilehdr[14], bmpinfohdr[40];
    fread(bmpfilehdr, sizeof(uint8_t), 14, f); // read the 54-byte header
    fread(bmpinfohdr, sizeof(uint8_t), 40, f); // read the 54-byte header

    // extract image height and width from header
    width           = *(uint32_t*)&bmpinfohdr[irt_dib_header_image_width];
    height          = *(uint32_t*)&bmpinfohdr[irt_dib_header_image_height];
    uint32_t planes = (*(uint16_t*)&bmpinfohdr[irt_dib_header_image_bpp]) >> 3;

    uint32_t Wi = width;
    uint32_t Hi = height;

    // IRT_TRACE_UTILS("ReadBMP : W %d H %d Planes %d\n", width, height, planes);

    for (int argcnt = 1; argcnt < argc; argcnt++) {
        if (!strcmp(argv[argcnt], "-Pwi"))
            pixel_width = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-Ppi"))
            pixel_padding = (uint8_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-rand_input_image"))
            rand_input_image = 1;
        if (!strcmp(argv[argcnt], "-Hi"))
            Hi = (uint16_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-Wi"))
            Wi = (uint16_t)atoi(argv[argcnt + 1]);
        if (!strcmp(argv[argcnt], "-irt_mode"))
            irt_mode = (uint16_t)atoi(argv[argcnt + 1]);
    }

    // printf ("Reading file %s %dx%d\n", filename, width, height);

    if (width > IIMAGE_W || height > IIMAGE_H) {
        IRT_TRACE_UTILS(
            "Desc gen error: read image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
            width,
            height,
            IIMAGE_W,
            IIMAGE_H);
        IRT_TRACE_TO_RES_UTILS(
            test_res,
            "was not run, read image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
            width,
            height,
            IIMAGE_W,
            IIMAGE_H);
        IRT_CLOSE_FAILED_TEST(0);
    }
    if (Wi > IIMAGE_W || Hi > IIMAGE_H) {
        IRT_TRACE_UTILS(
            "Desc gen error: input image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
            Wi,
            Hi,
            IIMAGE_W,
            IIMAGE_H);
        IRT_TRACE_TO_RES_UTILS(
            test_res,
            "was not run, input image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
            Wi,
            Hi,
            IIMAGE_W,
            IIMAGE_H);
        IRT_CLOSE_FAILED_TEST(0);
    }

    if (irt_mode > 3) {
        return;
    }

    uint32_t row_padded = (width * planes + 3) & (~3);
    uint8_t* data       = new uint8_t[row_padded];
    // IRT_TRACE_UTILS("ReadBMP : row_padded %d\n", row_padded);

    uint8_t  Psi = pixel_width <= 8 ? 1 : 2;
    uint64_t pixel_addr;

    uint8_t  bits_to_add = pixel_width - 8;
    uint32_t row         = 0;

    // filling input memory array and ext mem array with initial data
    uint32_t Himax = (uint32_t)std::max((int32_t)Hi, (int32_t)height);
    uint32_t Wimax = (uint32_t)std::max((int32_t)Wi, (int32_t)width);
    for (uint32_t i = 0; i < Himax; i++) {
        row = i;
        for (uint32_t j = 0; j < Wimax; j++) {
            for (uint8_t idx = 0; idx < PLANES; idx++) {
                pixel_addr = irt_top->irt_desc[0].image_par[IIMAGE].addr_start +
                             (uint64_t)idx * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL; // component base address
                pixel_addr += (uint64_t)row * Wimax * Psi; // line base address
                pixel_addr += (uint64_t)j * Psi; // pixel address
                data[idx] = rand() % 255; // irt_colors[i % NUM_OF_COLORS][idx];
                // data[j + idx] = idx * 50 + 50 + rand() % 10;
                input_image[idx][row][j] = (((uint64_t)data[idx] << bits_to_add) + (rand() % (1 << bits_to_add)))
                                           << pixel_padding;
                input_image[idx][row][j] |= rand() % (1 << pixel_padding); // garbage on LSB
                input_image[idx][row][j] |= (rand() % (1 << (16 - pixel_width - pixel_padding)))
                                            << (pixel_width + pixel_padding);

                // input_image[idx][i][j / 3] = j / 3;

                ext_mem[pixel_addr] = (uint8_t)(input_image[idx][row][j] & 0xff);
                // ext_mem[pixel_addr] = (unsigned char)(((i - (height - 1)) * 100 + ((j/3)%100)));
                if (Psi == 2) {
                    ext_mem[pixel_addr + 1] =
                        (uint8_t)((input_image[idx][row][j] >> 8) & 0xff); // store 2nd byte at next address
                }
            }
        }
    }
    // for(int32_t i = height-1; i >= 0; i--) //image array is swapped in BMP file
    for (uint32_t i = 0; i < height; i++) // image array is swapped in BMP file
    {
        if (irt_bmp_rd_order == e_irt_bmp_H20)
            row = height - 1 - i;
        else
            row = i;

        fread(data, sizeof(uint8_t), row_padded, f);
        // IRT_TRACE_UTILS("ReadBMP : Initializing Line -  %d\n", i);

        for (uint32_t j = 0; j < width; j++) {

            for (uint8_t idx = 0; idx < PLANES; idx++) {
                pixel_addr = irt_top->irt_desc[0].image_par[IIMAGE].addr_start +
                             (uint64_t)idx * IIMAGE_W * IIMAGE_H * BYTEs4PIXEL; // component base address
                pixel_addr += (uint64_t)row * Wimax * Psi; // line base address
                pixel_addr += (uint64_t)j * Psi; // pixel address
                if (rand_input_image) {
                    data[j * planes + idx] = rand() % 255; // irt_colors[i % NUM_OF_COLORS][idx];
                }
                // data[j + idx] = idx * 50 + 50 + rand() % 10;
                input_image[idx][row][j] =
                    (((uint64_t)data[j * planes + idx] << bits_to_add) + (rand() % (1 << bits_to_add)))
                    << pixel_padding;
                input_image[idx][row][j] |= rand() % (1 << pixel_padding); // garbage on LSB
                input_image[idx][row][j] |= (rand() % (1 << (16 - pixel_width - pixel_padding)))
                                            << (pixel_width + pixel_padding);

                // input_image[idx][i][j / 3] = j / 3;

                ext_mem[pixel_addr] = (uint8_t)(input_image[idx][row][j] & 0xff);

                // IRT_TRACE_UTILS("ReadBMP : ext_mem[%x] = %x input_image[%d][%d][%d] = %x\n", pixel_addr,
                // ext_mem[pixel_addr], idx, row, j ,input_image[idx][row][j]);

                // ext_mem[pixel_addr] = (unsigned char)(((i - (height - 1)) * 100 + ((j/3)%100)));
                if (Psi == 2) {
                    ext_mem[pixel_addr + 1] =
                        (uint8_t)((input_image[idx][row][j] >> 8) & 0xff); // store 2nd byte at next address
                }
            }

            // input_image16[i * width*3 + j]   = (unsigned short)(data[j]);
            // input_image16[i * width*3 + j+1] = (unsigned short)(data[j+1]);
            // input_image16[i * width*3 + j+2] = (unsigned short)(data[j+2]);

            // input_image[0][i][j/3] = 0xaa; 	ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = 0xaa;
            // input_image[1][i][j/3] = 0xbb;   ext_mem[image_pars[IIMAGE].ADDR+1*IMAGE_H*IMAGE_W+i*width+j/3] = 0xbb;
            // input_image[2][i][j/3] = 0xcc;   ext_mem[image_pars[IIMAGE].ADDR+2*IMAGE_H*IMAGE_W+i*width+j/3] = 0xcc;

            //			input_image[0][i][j/3] = (i)&0xff;//((i&0xf)<<4)|((j/3)&0xf);
            //			ext_mem[irt_top->irt_desc[0].image_par[IIMAGE].addr_start+0*IMAGE_H*IMAGE_W+i*width+j/3] =
            //(i)&0xff;//((i&0xf)<<4)|((j/3)&0xf); 			input_image[2][i][j/3] =
            //(j/3)&0xff;//((i&0xf)<<4)|((j/3)&0xf);
            // ext_mem[image_pars[IIMAGE].ADDR+2*IMAGE_H*IMAGE_W+i*width+j/3] = (j/3)&0xff;//((i&0xf)<<4)|((j/3)&0xf);

            /*
                        if (j%2 == 0)
                            ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = (unsigned char)i;
                        else
                            ext_mem[image_pars[IIMAGE].ADDR+i*width+j/3] = (unsigned char) (j/3);
            */
        }
    }

    fclose(f);
    width  = Wimax;
    height = Himax;
    if (print_out_files)
        generate_image_dump("IRT_dump_in_image%d_plane%d.txt",
                            width,
                            height,
                            (Eirt_bmp_order)1,
                            input_image,
                            0, // input_image stored according to irt_bmp_rd_order, dumped as is
                            ((Psi == 2) ? 0xFFFF : 0xFF),
                            1); // IP image
}

void irt_quality_comp(IRT_top* irt_top, uint8_t image)
{

    uint8_t  desc = image * PLANES;
    uint16_t Ho   = irt_top->irt_desc[desc].image_par[OIMAGE].H;
    uint16_t So   = irt_top->irt_desc[desc].image_par[OIMAGE].S;

    // 0 - fixed2double, 1 - fixed2float, 2 - double2float, 3 - float2fix16
    int non_bg_pxl_cnt[CALC_FORMATS] = {0}, pixel_err_cnt[CALC_FORMATS] = {0}, pixel_err_max[CALC_FORMATS] = {0},
        pixel_err_hist[CALC_FORMATS][PIXEL_ERR_BINS] = {0}, pixel_err_max_idx[CALC_FORMATS][2] = {0};
    int  pxl_error;
    bool non_bg_flag;

    double mse[CALC_FORMATS] = {0};
    for (uint8_t format = e_irt_crdf_arch; format <= e_irt_crdf_fix16; format++) {
        if (format != e_irt_crdf_fpt64) {
            for (uint16_t row = 0; row < Ho; row++) {
                for (uint16_t col = 0; col < So; col++) {
                    for (uint8_t idx = 0; idx < PLANES; idx++) {
                        non_bg_flag = output_image[image * CALC_FORMATS_ROT + e_irt_crdf_fpt64][idx][row][col] !=
                                      irt_top->irt_desc[image].bg;
                        pxl_error =
                            abs((int)(output_image[image * CALC_FORMATS_ROT + format][idx][row][col] -
                                      output_image[image * CALC_FORMATS_ROT + e_irt_crdf_fpt64][idx][row][col]));
                        if (non_bg_flag) { // not count bg pixels
                            non_bg_pxl_cnt[format]++;
                            mse[format] += pow((double)pxl_error, 2);
                            if (pxl_error > 0) {
                                pixel_err_cnt[format]++;
                                pixel_err_hist[format][std::min(pxl_error, PIXEL_ERR_BINS - 1)]++;
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
        IRT_TRACE_UTILS("----------------------------------------------------------------------------------------------"
                        "----------\n");
        IRT_TRACE_UTILS("| Image | Format | Coordinates selection |  Pixels errors  |  PSNR  |     Max Error    | Non "
                        "bg pixels |\n");
        IRT_TRACE_UTILS("----------------------------------------------------------------------------------------------"
                        "----------\n");
    }

    double psnr[CALC_FORMATS];

    uint8_t last_format = e_irt_crdf_fixed;
    switch (irt_top->irt_desc[desc].irt_mode) {
        case e_irt_rotation: last_format = e_irt_crdf_fix16; break;
        case e_irt_affine: last_format = e_irt_crdf_fixed; break;
        case e_irt_projection:
            switch (irt_top->rot_pars[desc].proj_mode) {
                case e_irt_rotation: last_format = e_irt_crdf_fpt32; break;
                case e_irt_affine: last_format = e_irt_crdf_fpt32; break;
                case e_irt_projection: last_format = e_irt_crdf_fpt32; break;
            }
            break;
        case e_irt_mesh:
            switch (irt_top->rot_pars[desc].mesh_mode) {
                case e_irt_rotation: last_format = e_irt_crdf_fix16; break;
                case e_irt_affine: last_format = e_irt_crdf_fixed; break;
                case e_irt_projection: last_format = e_irt_crdf_fpt32; break;
                case e_irt_mesh: last_format = e_irt_crdf_fpt32; break;
            }
            if (irt_top->rot_pars[desc].dist_r == 1)
                last_format = e_irt_crdf_fpt32;
            break;
    }

    for (uint8_t format = e_irt_crdf_arch; format <= last_format; format++) {
        mse[format]  = mse[format] / ((double)non_bg_pxl_cnt[format]);
        psnr[format] = 10.0 * log10(pow((double)irt_top->irt_desc[desc].MAX_VALo, 2) / mse[format]);
        if (format != e_irt_crdf_fpt64) {
            IRT_TRACE_UTILS("|   %d   | %s  |", image, irt_hl_format_s[format]);
            if (coord_err[format] != 0) {
                // IRT_TRACE_UTILS("Image %d double and %s coordinate difference %d (%2.2f%%)\n", image,
                // irt_hl_format_s[format], coord_err[format], 100.0 * (float)coord_err[format] / (Ho * Wo));
                IRT_TRACE_UTILS(
                    " Error %5d (%6.2f%%) |", coord_err[format], 100.0 * (float)coord_err[format] / (Ho * So));
            } else {
                // IRT_TRACE_UTILS("Image %d double and %s coordinate selection match\n", image,
                // irt_hl_format_s[format]);
                IRT_TRACE_UTILS("         Match         |");
            }

            if (pixel_err_cnt[format] == 0) {
                // IRT_TRACE_UTILS("Image %d double and %s results are bit exact\n", image, irt_hl_format_s[format]);
                IRT_TRACE_UTILS("    Bit exact    |  High  |         0        |");
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
                IRT_TRACE_UTILS(" %7d(%5.2f%%) | %6.2f | %5d[%4d,%4d] |",
                                pixel_err_cnt[format],
                                100.0 * (double)pixel_err_cnt[format] / ((double)non_bg_pxl_cnt[format]),
                                psnr[format],
                                pixel_err_max[format],
                                pixel_err_max_idx[format][0],
                                pixel_err_max_idx[format][1]);
            }

            IRT_TRACE_UTILS(
                " %8d(%3.0f%%)|\n", non_bg_pxl_cnt[format], (float)100.0 * non_bg_pxl_cnt[format] / (3 * Ho * So));
        }
    }
    IRT_TRACE_UTILS(
        "--------------------------------------------------------------------------------------------------------\n");
}

#endif

#if defined(STANDALONE_ROTATOR) || defined(CREATE_IMAGE_DUMPS) || defined(RUN_WITH_SV)
uint32_t output_image_dump(IRT_top* irt_top, uint8_t image, uint8_t planes, uint8_t verif)
{

    char out_file_name[50];
    int  n                    = sprintf(out_file_name, "IRT_out_image_%d.bmp", image);
    n                         = n;
    uint8_t  image_first_desc = image * planes;
    FILE*    f                = nullptr;
    uint16_t stripe_o         = (irt_top->irt_desc[image_first_desc].irt_mode < 4)
                            ? irt_top->irt_desc[image_first_desc].image_par[OIMAGE].S
                            : ((irt_top->irt_desc[image_first_desc].irt_mode == 6)
                                   ? irt_top->irt_desc[image_first_desc].image_par[IIMAGE].W
                                   : irt_top->irt_desc[image_first_desc].image_par[OIMAGE].W);
    uint16_t height_o =
        ((irt_top->irt_desc[image_first_desc].irt_mode == 6) ? irt_top->irt_desc[image_first_desc].image_par[IIMAGE].H
                                                             : irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);

    uint64_t*** out_image = nullptr;
    out_image             = new uint64_t**[PLANES];
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        (out_image)[plain] = new uint64_t*[height_o];
        for (uint16_t row = 0; row < height_o; row++) {
            (out_image)[plain][row] = new uint64_t[stripe_o];
        }
    }
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        for (uint16_t row = 0; row < height_o; row++) {
            for (uint16_t col = 0; col < stripe_o; col++) {
                (out_image)[plain][row][col] = 0;
            }
        }
    }

    uint16_t row_padded = (stripe_o * 3 + 3) & (~3);
    IRT_TRACE_UTILS("output_image_dump :: image_first_desc %d stripe_o %d irtmode %d O.H %d \n",
                    image_first_desc,
                    stripe_o,
                    irt_top->irt_desc[image_first_desc].irt_mode,
                    height_o);

    if (print_out_files) {
        f = fopen(out_file_name, "wb");
        generate_bmp_header(f, stripe_o, irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H, row_padded, 3);
    }

    uint32_t error_count = 0;
    uint64_t pixel_val[3];
    uint64_t pixel_addr;
    bool     pixel_error;

    uint16_t row = 0;
    for (uint16_t i = 0; i < height_o; i++) { // image array is swapped in BMP file
        if (irt_bmp_wr_order == e_irt_bmp_H20)
            row = height_o - 1 - i;
        else
            row = i;

        for (uint16_t j = 0; j < stripe_o; j++) {
            for (uint8_t idx = 0; idx < planes; idx++) {
                pixel_addr = (verif == 1)
                                 ? VERIF_OIMAGE_OFFSET
                                 : irt_top->irt_desc[image_first_desc + idx]
                                       .image_par[OIMAGE]
                                       .addr_start; // +(uint64_t)idx * IMAGE_W * IMAGE_H; //component base address
                pixel_addr += ((uint64_t)row * stripe_o)
                              << irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps; // line base address
                pixel_addr +=
                    (((uint64_t)j) << irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps); // pixel address
                pixel_val[idx] = 0;
                for (int s = 0; s < (1 << irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps); s++) {
                    pixel_val[idx] += (((uint64_t)ext_mem[pixel_addr + s]) << (8 * s));
                }
                (out_image)[idx][row][j] = pixel_val[idx];
            }
            uint16_t k = j;
            // compare output image with reference image
            pixel_error = 0;
            for (uint8_t idx = 0; idx < planes; idx++) {
#ifndef RUN_WITH_SV
                // data[k * 3 + idx] = (uint8_t)(pixel_val[idx] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 +
                // irt_top->irt_desc[image_first_desc].Ppo));
#endif
                if (pixel_val[idx] != output_image[image * CALC_FORMATS_ROT][idx][row][j]) {
                    if (irt_top->irt_desc[image_first_desc].irt_mode == 6) { // BWD2 - allow single bit error
                        uint32_t pix_diff =
                            (pixel_val[idx] > output_image[image * CALC_FORMATS_ROT][idx][row][j])
                                ? (pixel_val[idx] - output_image[image * CALC_FORMATS_ROT][idx][row][j])
                                : (output_image[image * CALC_FORMATS_ROT][idx][row][j] - pixel_val[idx]);
                        if (pix_diff <= irt_top->irt_cfg.bwd2_err_margin) {
                            IRT_TRACE_UTILS("Pixel WARNING at [%d, %d][%d]: HL [0x%lx] LL [0x%lx]\n",
                                            row,
                                            j,
                                            idx,
                                            output_image[image * CALC_FORMATS_ROT][idx][row][j],
                                            pixel_val[idx]);
                        } else {
                            IRT_TRACE_UTILS("Pixel Error at [%d, %d][%d]: HL [0x%lx] LL [0x%lx]\n",
                                            row,
                                            j,
                                            idx,
                                            output_image[image * CALC_FORMATS_ROT][idx][row][j],
                                            pixel_val[idx]);
                            pixel_error = 1;
                        }
                    } else {
                        pixel_error = 1;
                        IRT_TRACE_UTILS("Pixel Error at [%d, %d][%d]: HL [0x%lx] LL [0x%lx]\n",
                                        row,
                                        j,
                                        idx,
                                        output_image[image * CALC_FORMATS_ROT][idx][row][j],
                                        pixel_val[idx]);
                    }
                } /*else{
                    IRT_TRACE_UTILS("Pixel Match at [%d, %d][%d]: HL [0x%lx] LL [0x%lx]\n", row, j, idx,
                 output_image[image * CALC_FORMATS_ROT][idx][row][j], pixel_val[idx]);
                 }*/
            }

#ifndef RUN_WITH_SV
            // if (planes < 3) {
            //   for (uint8_t idx = planes; idx < 3; idx++) {
            //      data[j * 3 + idx] = (uint8_t)(pixel_val[0] >> (irt_top->rot_pars[image_first_desc].Pwo - 8 +
            //      irt_top->irt_desc[image_first_desc].Ppo));
            //   }
            //}
#endif
            error_count += pixel_error;
        }
#ifndef RUN_WITH_SV
        // if (print_out_files)
        //   fwrite(data, sizeof(uint8_t), row_padded, f);
#endif
    }
    if (print_out_files)
        fclose(f);

    // fprintf(test_res, "%s %3.2f %d %d", file_name, rot_angle, irt_top->irt_desc[0].image_par[OIMAGE].H,
    // irt_top->irt_desc[0].image_par[OIMAGE].W);
    IRT_TRACE_UTILS("Image %d %dx%d stripe %dx%d",
                    image,
                    irt_top->irt_desc[image_first_desc].Wo,
                    irt_top->irt_desc[image_first_desc].Ho,
                    stripe_o,
                    irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);
    IRT_TRACE_TO_RES_UTILS(test_res,
                           " image %d %dx%d stripe %dx%d",
                           image,
                           irt_top->irt_desc[image_first_desc].Wo,
                           irt_top->irt_desc[image_first_desc].Ho,
                           stripe_o,
                           irt_top->irt_desc[image_first_desc].image_par[OIMAGE].H);

    if (error_count == 0) {
        IRT_TRACE_UTILS(" passed\n");
        IRT_TRACE_TO_RES_UTILS(test_res, " passed");
        IRT_TRACE_UTILS("IRTSIM :: TEST PASSED \n");
    } else {
        IRT_TRACE_UTILS(" error count = %d\n", error_count);
        IRT_TRACE_TO_RES_UTILS(test_res, " error count = %d", error_count);
        IRT_TRACE_UTILS("IRTSIM :: TEST FAILED \n");
    }

    if (print_out_files) {
        uint64_t image_mask = 0xFF;
        uint8_t  Ps         = 8 * (1 << (irt_top->irt_desc[image_first_desc].image_par[OIMAGE].Ps +
                                (irt_top->irt_desc[image_first_desc].irt_mode == 5)));
        for (uint16_t bc = 0; bc < Ps; bc++) {
            image_mask = (image_mask << 1) | 1;
        }
        generate_image_dump("IRT_dump_out_image%d_plane%d.txt",
                            stripe_o,
                            height_o,
                            (Eirt_bmp_order)0,
                            out_image,
                            image, // out_image stored according to irt_bmp_wr_order, dumped as is
                            image_mask,
                            0); // OP image
    }
    for (uint8_t plain = 0; plain < PLANES; plain++) {
        for (uint16_t row = 0; row < height_o; row++) {
            delete[] out_image[plain][row];
            out_image[plain][row] = nullptr;
        }
        delete[] out_image[plain];
        out_image[plain] = nullptr;
    }
    delete[] out_image;
    out_image = nullptr;
    return error_count;

    // fclose (test_res);
}

void generate_plane_dump(const char*    fname,
                         uint32_t       width,
                         uint32_t       height,
                         Eirt_bmp_order vert_order,
                         uint64_t**     image_ptr,
                         uint8_t        image,
                         uint8_t        plane,
                         uint8_t        img_format, // 0-Full Row per Line format   1-Each element per line format
                         uint64_t       image_mask,
                         uint8_t        ip_img)
{

    FILE*    f   = nullptr;
    FILE*    f1  = nullptr;
    uint32_t row = 0;
    char     fname1[50], fileName[50];

    sprintf(fname1, fname, image, plane);

    f = fopen(fname1, "w");
    IRT_TRACE_UTILS("generate_plane_dump: vert_order[%d] image_mask[%d] \n", (uint8_t)vert_order, image_mask);

    if ((ip_img == 1) && (plane == 0)) {
        sprintf(fileName, "sival_input_image_task%d.txt", image);
        f1 = fopen(fileName, "w");
        fprintf(
            f1, "// IMAGE DIMs H x W : %0d x %0d  \t data-type : %0d \n", height, width, (image_mask == 0xFF) ? 0 : 1);
    }

    fprintf(f, "// Output Image Plane %d : H x W - %d x %d \n", plane, height, width);
    for (uint32_t i = 0; i < height; i++) {
        if (vert_order == e_irt_bmp_H20)
            row = height - 1 - i;
        else
            row = i;

        if (img_format == 0) {
            for (uint32_t j = 0; j < width; j++) {
                fprintf(f, "%llx ", (uint64_t)image_ptr[row][j] & image_mask);
                if ((ip_img == 1) && (plane == 0)) {
                    fprintf(f1, "%llx ", (uint64_t)image_ptr[row][j] & image_mask);
                }
            }
            fprintf(f, "// row %d\n", i);
        } else {
            for (uint32_t j = 0; j < width; j++) {
                fprintf(f, "%llx \n", (uint64_t)image_ptr[row][j] & image_mask);
                if ((ip_img == 1) && (plane == 0)) {
                    fprintf(f1, "%llx \n", (uint64_t)image_ptr[row][j] & image_mask);
                    // IRT_TRACE_UTILS("generate_plane_dump: image_ptr[%d][%d] = %x  data =
                    // %x\n",row,j,image_ptr[row][j], (uint64_t)image_ptr[row][j] & image_mask);
                }
            }
            // fprintf(f, "// row %d\n", i);
        }
    }

    fclose(f);
}
void generate_image_dump_per_datatype(const char*            fname,
                                      uint32_t               width,
                                      uint32_t               height,
                                      float***               image_ptr,
                                      Eirt_resamp_dtype_enum DataType)
{

    FILE*    f   = nullptr;
    uint32_t row = 0;
    char     fname1[50];

    sprintf(fname1, fname);

    f = fopen(fname1, "w");
    fprintf(f, "// IMAGE DIMs H x W : %0d x %0d  \t data-type : %0d \n", height, width, DataType);
    for (uint32_t i = 0; i < height; i++) {
        row = i;
        for (uint32_t j = 0; j < width; j++) {
            fprintf(f,
                    "%llx\n",
                    (uint32_t)IRT_top::IRT_UTILS::conversion_float_bit(
                        image_ptr[0][row][j], DataType, 0, ((DataType == 0) ? 0xFF : 0xFFFF), 0, 0, 0, 0, 0));
        }
        // fprintf(f, "// row %d\n", i);
    }
    fclose(f);
}

void generate_image_dump(const char*    fname,
                         uint32_t       width,
                         uint32_t       height,
                         Eirt_bmp_order vert_order,
                         uint64_t***    image_ptr,
                         uint8_t        image,
                         uint64_t       image_mask,
                         uint8_t        ip_img)
{

    for (uint8_t idx = 0; idx < PLANES; idx++) {
        generate_plane_dump(fname, width, height, vert_order, image_ptr[idx], image, idx, 1, image_mask, ip_img);
    }
}

#endif

void irt_mems_modes_check(const rotation_par& rot_pars, const rotation_par& rot_pars0, uint8_t desc)
{

    // rot memory check IRT_RM_CLIENTS
    for (uint8_t mem_client = 0; mem_client < IRT_RM_CLIENTS; mem_client++) {
        if (rot_pars.buf_format[mem_client] != rot_pars0.buf_format[mem_client]) {
            // buf format is non consistant
            IRT_TRACE_UTILS("Error: desc %d %s memory %s buffer format is different from desc 0 %s buffer format\n",
                            desc,
                            irt_mem_type_s[mem_client],
                            irt_buf_format_s[rot_pars.buf_format[mem_client]],
                            irt_buf_format_s[rot_pars0.buf_format[mem_client]]);
            IRT_TRACE_TO_RES_UTILS(
                test_res,
                " was not run, desc %d %s memory %s buffer format is different from desc 0 %s buffer format\n",
                desc,
                irt_mem_type_s[mem_client],
                irt_buf_format_s[rot_pars.buf_format[mem_client]],
                irt_buf_format_s[rot_pars0.buf_format[mem_client]]);
            IRT_CLOSE_FAILED_TEST(0);
        }

        if (rot_pars0.buf_format[mem_client] == e_irt_buf_format_static) {
            // buf format is static, buf auto select is not allowed and all buffer modes must be same
            if (rot_pars0.buf_select[mem_client] == e_irt_buf_select_auto) {
                // buf auto select is not allowed
                IRT_TRACE_UTILS(
                    "Error: desc %d %s memory buf_select is set to auto that is not allowed in static buf_format\n",
                    desc,
                    irt_mem_type_s[mem_client]);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       " was not run, desc %d %s memory buf_select is set to auto that is not allowed "
                                       "in static buf_format\n",
                                       desc,
                                       irt_mem_type_s[mem_client]);
                IRT_CLOSE_FAILED_TEST(0);
            }

            if (rot_pars.buf_mode[mem_client] != rot_pars0.buf_mode[mem_client]) {
                // buf mode is non consistant
                IRT_TRACE_UTILS("Error: desc %d %s memory buffer mode %d is different from desc 0 buffer format %d in "
                                "static buf_format\n",
                                desc,
                                irt_mem_type_s[mem_client],
                                rot_pars.buf_mode[mem_client],
                                rot_pars0.buf_mode[mem_client]);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       " was not run, desc %d %s memory buffer mode %d is different from desc 0 buffer "
                                       "format %d in static buf_format\n",
                                       desc,
                                       irt_mem_type_s[mem_client],
                                       rot_pars.buf_mode[mem_client],
                                       rot_pars0.buf_mode[mem_client]);
                IRT_CLOSE_FAILED_TEST(0);
            }
        }
    }
}

void irt_params_analize(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{
    /*

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
        rot_pars.mesh_rotation				= rot_pars.mesh_direct_rotation | rot_pars.mesh_affine_rotation |
       rot_pars.mesh_projection_rotation;

        rot_pars.projection_affine			= rot_pars.proj_mode == e_irt_affine;
        rot_pars.mesh_direct_affine			= rot_pars.mesh_mode == e_irt_affine;
        rot_pars.mesh_projection_affine		= rot_pars.mesh_mode == e_irt_projection && rot_pars.projection_affine;
        rot_pars.mesh_affine				= rot_pars.mesh_direct_affine | rot_pars.mesh_projection_affine;

        rot_pars.mesh_dist_r0_Sh1_Sv1 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv == 1;
        //rot_pars.mesh_dist_r0_Sh1_Sv1_fp32 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv ==
       1;//&& irt_desc.mesh_format == e_irt_mesh_fp32;

        IRT_TRACE_UTILS("affine_rotation            = %d\n", rot_pars.affine_rotation);
        IRT_TRACE_UTILS("projection_direct_rotation = %d\n", rot_pars.projection_direct_rotation);
        IRT_TRACE_UTILS("projection_affine_rotation = %d\n", rot_pars.projection_affine_rotation);
        IRT_TRACE_UTILS("projection_rotation        = %d\n", rot_pars.projection_rotation);
        IRT_TRACE_UTILS("mesh_direct_rotation       = %d\n", rot_pars.mesh_direct_rotation);
        IRT_TRACE_UTILS("mesh_affine_rotation       = %d\n", rot_pars.mesh_affine_rotation);
        IRT_TRACE_UTILS("mesh_projection_rotation   = %d\n", rot_pars.mesh_projection_rotation);
        IRT_TRACE_UTILS("mesh_rotation              = %d\n", rot_pars.mesh_rotation);
        IRT_TRACE_UTILS("projection_affine          = %d\n", rot_pars.projection_affine);
        IRT_TRACE_UTILS("mesh_direct_affine         = %d\n", rot_pars.mesh_direct_affine);
        IRT_TRACE_UTILS("mesh_projection_affine     = %d\n", rot_pars.mesh_projection_affine);
        IRT_TRACE_UTILS("mesh_affine                = %d\n", rot_pars.mesh_affine);

        rot_pars.irt_rotate = irt_desc.irt_mode == e_irt_rotation	||
                             (irt_desc.irt_mode == e_irt_affine		&& rot_pars.affine_rotation)	 ||
                             (irt_desc.irt_mode == e_irt_projection && rot_pars.projection_rotation) ||
                             (irt_desc.irt_mode == e_irt_mesh		&& rot_pars.mesh_rotation &&
       rot_pars.mesh_dist_r0_Sh1_Sv1);

        rot_pars.irt_affine = irt_desc.irt_mode == e_irt_affine		||
                             (irt_desc.irt_mode == e_irt_projection && rot_pars.projection_affine)	 ||
                             (irt_desc.irt_mode == e_irt_mesh		&& rot_pars.mesh_affine   &&
       rot_pars.mesh_dist_r0_Sh1_Sv1);

        double cotx = rot_pars.shear_mode & 1 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_x] * M_PI / 180.0) : 0;
        double coty = rot_pars.shear_mode & 2 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_y] * M_PI / 180.0) : 0;

        rot_pars.irt_affine_hscaling = rot_pars.affine_flags[e_irt_aff_scaling] == 1 && rot_pars.Sx != 1;
        rot_pars.irt_affine_vscaling = rot_pars.affine_flags[e_irt_aff_scaling] == 1 && rot_pars.Sy != 1;
        rot_pars.irt_affine_shearing = rot_pars.affine_flags[e_irt_aff_shearing] == 1 && ((cotx != 0 && cotx != 1) ||
       (coty != 0 && coty != 1)); rot_pars.irt_affine_st_inth  = (rot_pars.irt_affine_hscaling ||
       rot_pars.irt_affine_shearing) && rot_pars.irt_affine; rot_pars.irt_affine_st_intv  =
       (rot_pars.irt_affine_vscaling || rot_pars.irt_affine_shearing) && rot_pars.irt_affine;
        IRT_TRACE_UTILS("irt_affine_hscaling        = %d\n", rot_pars.irt_affine_hscaling);
        IRT_TRACE_UTILS("irt_affine_vscaling        = %d\n", rot_pars.irt_affine_vscaling);
        IRT_TRACE_UTILS("irt_affine_shearing        = %d\n", rot_pars.irt_affine_shearing);
        IRT_TRACE_UTILS("irt_affine_st_inth         = %d\n", rot_pars.irt_affine_st_inth);
        IRT_TRACE_UTILS("irt_affine_st_intv         = %d\n", rot_pars.irt_affine_st_intv);
    */
}

void irt_rot_angle_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

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
                    IRT_TRACE_UTILS("%s angle %f is not supported, provide angle in [-360:360] range\n",
                                    irt_prj_matrix_s[angle_type],
                                    rot_pars.irt_angles[angle_type]);
                    IRT_TRACE_TO_RES_UTILS(test_res,
                                           " was not run, %s angle %f is not supported\n",
                                           irt_prj_matrix_s[angle_type],
                                           rot_pars.irt_angles[angle_type]);
                    IRT_CLOSE_FAILED_TEST(0);
                }
                if (rot_pars.irt_angles[angle_type] < 0)
                    irt_angle_tmp[angle_type] =
                        rot_pars.irt_angles[angle_type] + 360; // converting angles to positive range
                else
                    irt_angle_tmp[angle_type] = rot_pars.irt_angles[angle_type];
                if (irt_angle_tmp[angle_type] >= 270)
                    irt_angle_tmp[angle_type] -= 360; // converting [270:360] to [-90:0]
                break;
        }
    }

    if (irt_desc.irt_mode == e_irt_rotation ||
        (irt_desc.irt_mode == e_irt_affine && strchr(rot_pars.affine_mode, 'R') != nullptr) ||
        (irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_rotation) ||
        (irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine &&
         strchr(rot_pars.affine_mode, 'R') != nullptr) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_rotation) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_affine &&
         strchr(rot_pars.affine_mode, 'R') != nullptr) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection &&
         rot_pars.proj_mode == e_irt_rotation) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection &&
         rot_pars.proj_mode == e_irt_affine && strchr(rot_pars.affine_mode, 'R') != nullptr)) {
        if (-MAX_ROT_ANGLE <= irt_angle_tmp[e_irt_angle_rot] && irt_angle_tmp[e_irt_angle_rot] <= MAX_ROT_ANGLE) {
            irt_desc.read_vflip                      = 0;
            irt_desc.read_hflip                      = 0;
            rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
        } else if ((180.0 - MAX_ROT_ANGLE) <= irt_angle_tmp[e_irt_angle_rot] &&
                   irt_angle_tmp[e_irt_angle_rot] <= (180.0 + MAX_ROT_ANGLE)) {
            irt_desc.read_hflip                      = 1;
            irt_desc.read_vflip                      = 1;
            rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot] - 180.0;
            IRT_TRACE_UTILS("Converting rotation angle to %f with pre-rotation of 180 degree\n",
                            rot_pars.irt_angles_adj[e_irt_angle_rot]);
        } else if (irt_angle_tmp[e_irt_angle_rot] == 90 || irt_angle_tmp[e_irt_angle_rot] == -90.0) {
            rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
            irt_desc.rot90                           = 1;
            if ((irt_desc.image_par[OIMAGE].Xc & 1) ^
                (irt_desc.image_par[IIMAGE].Yc & 1)) { // interpolation over input lines is required
                irt_desc.rot90_intv = 1;
                IRT_TRACE_UTILS("Rotation 90 degree will require input lines interpolation\n");
            }
            if ((irt_desc.image_par[OIMAGE].Yc & 1) ^
                (irt_desc.image_par[IIMAGE].Xc & 1)) { // interpolation input pixels is required
                irt_desc.rot90_inth = 1;
                IRT_TRACE_UTILS("Rotation 90 degree will require input pixels interpolation\n");
            }
        } else {
            IRT_TRACE_UTILS("Rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
            IRT_TRACE_TO_RES_UTILS(
                test_res, " was not run, rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
            IRT_CLOSE_FAILED_TEST(0);
        }
    } else if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) ||
               (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection &&
                rot_pars.proj_mode == e_irt_projection)) {

        irt_desc.read_vflip = 0;
        irt_desc.read_hflip = 0;

        for (uint8_t angle_type = e_irt_angle_roll; angle_type <= e_irt_angle_yaw; angle_type++) {
            if (-MAX_ROT_ANGLE <= irt_angle_tmp[angle_type] && irt_angle_tmp[angle_type] <= MAX_ROT_ANGLE) {
                rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type];
            } else if ((180.0 - MAX_ROT_ANGLE) <= irt_angle_tmp[angle_type] &&
                       irt_angle_tmp[angle_type] <= (180.0 + MAX_ROT_ANGLE)) {
                switch (angle_type) {
                    case e_irt_angle_roll:
                        irt_desc.read_hflip = !irt_desc.read_hflip;
                        irt_desc.read_vflip = !irt_desc.read_vflip;
                        break;
                    case e_irt_angle_pitch: irt_desc.read_vflip = !irt_desc.read_vflip; break;
                    case e_irt_angle_yaw: irt_desc.read_hflip = !irt_desc.read_hflip; break;
                }
                rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type] - 180.0;
                IRT_TRACE_UTILS("Converting %s angle to %f with %s flip\n",
                                irt_prj_matrix_s[angle_type],
                                rot_pars.irt_angles_adj[angle_type],
                                irt_flip_mode_s[angle_type]);
            } else if ((irt_angle_tmp[angle_type] == 90 || irt_angle_tmp[angle_type] == -90.0) &&
                       angle_type == e_irt_angle_roll) {
                rot_pars.irt_angles_adj[angle_type] = irt_angle_tmp[angle_type];
                // irt_desc.rot90 = 1;
                if ((irt_desc.image_par[OIMAGE].Xc & 1) ^
                    (irt_desc.image_par[IIMAGE].Yc & 1)) { // interpolation over input lines is required
                    irt_desc.rot90_intv = 1;
                    IRT_TRACE_UTILS("Roll rotation 90 degree will require input lines interpolation\n");
                }
                if ((irt_desc.image_par[OIMAGE].Yc & 1) ^
                    (irt_desc.image_par[IIMAGE].Xc & 1)) { // interpolation input pixels is required
                    irt_desc.rot90_inth = 1;
                    IRT_TRACE_UTILS("Roll rotation 90 degree will require input pixels interpolation\n");
                }
            } else {
                IRT_TRACE_UTILS(
                    "%s angle %f is not supported\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       " was not run, %s angle %f is not supported\n",
                                       irt_prj_matrix_s[angle_type],
                                       rot_pars.irt_angles[angle_type]);
                IRT_CLOSE_FAILED_TEST(0);
            }
        }
    }
    // rot_pars.irt_angles[e_irt_angle_rot] = rot_pars.irt_angles_adj[e_irt_angle_rot];
    irt_desc.rot_dir = (rot_pars.irt_angles_adj[e_irt_angle_rot] >= 0) ? IRT_ROT_DIR_POS : IRT_ROT_DIR_NEG;
}

void irt_aff_coefs_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    rot_pars.affine_Si_factor =
        fabs((fabs(rot_pars.M11d * rot_pars.M22d) - rot_pars.M12d * rot_pars.M21d) / rot_pars.M22d);

    if (rot_pars.M22d < 0) { // resulted in Yi decrease when Yo increase (embedded rotation is [90:270], vertical flip)
        irt_desc.read_vflip = 1;
        rot_pars.M22d       = -rot_pars.M22d;
    }

    if (rot_pars.M11d < 0) { // resulted in Xi decrease when Xo increase (embedded horizontal flip)
        irt_desc.read_hflip = 1;
        rot_pars.M11d       = -rot_pars.M11d;
    }

    if (rot_pars.M12d >=
        0) { // if (rot_pars.M21d <= 0) { //resulted in Yi decrease when Xo increase (embedded clockwise rotation)
        irt_desc.rot_dir = IRT_ROT_DIR_POS;
    } else {
        irt_desc.rot_dir = IRT_ROT_DIR_NEG;
    }
}

void irt_affine_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    double cosd = cos(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180);
    double sind = sin(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180);
    if (rot_pars.irt_angles[e_irt_angle_rot] !=
        rot_pars.irt_angles_adj[e_irt_angle_rot]) // positive rotation definition (clockwise) is opposite to angle
                                                  // definition
        sind = -sind;
    double cotx = rot_pars.shear_mode & 1 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_x] * M_PI / 180.0) : 0;
    double coty = rot_pars.shear_mode & 2 ? 1.0 / tan(rot_pars.irt_angles[e_irt_angle_shr_y] * M_PI / 180.0) : 0;
    double refx = rot_pars.reflection_mode & 1 ? -1 : 1;
    double refy = rot_pars.reflection_mode & 2 ? -1 : 1;
    if (rot_pars.irt_angles[e_irt_angle_rot] == 90 || rot_pars.irt_angles[e_irt_angle_rot] == -90)
        cosd = 0;

    double aff_basic_matrix[5][2][2] = {
        {{cosd, sind}, {-sind, cosd}}, // rotation
        {{1.0 / rot_pars.Sx, 0.0}, {0.0, 1.0 / rot_pars.Sy}}, // scaling
        {{refx, 0.0}, {0.0, refy}}, // reflection
        {{1.0, cotx}, {coty, 1.0}}, // shearing
        {{1.0, 0.0}, {0.0, 1.0}}, // unit
    };
    uint8_t aff_func, acc_idx = 1;

    double acc_matrix[2][2][2] = {
        {{1.0, 0.0}, {0.0, 1.0}}, // unit
        {{0.0, 0.0}, {0.0, 0.0}} // accumulated
    };

    // IRT_TRACE_UTILS("---------------------------------------------------------------------\n");
    // IRT_TRACE_UTILS("Task %d: Generating Affine coefficients for %s mode with:\n", desc, rot_pars.affine_mode);

    for (uint8_t i = 0; i < strlen(rot_pars.affine_mode); i++) {
        switch (rot_pars.affine_mode[i]) {
            case 'R': // rotation
                aff_func = 0;
                // IRT_TRACE_UTILS("* %3.2f rotation angle and rotation matrix:\t\t",
                // rot_pars.irt_angles[e_irt_angle_rot]);
                break;
            case 'S': // scaling
                aff_func = 1;
                // IRT_TRACE_UTILS("* Sx/Sy %3.2f/%3.2f scaling factors and scaling matrix:\t", rot_pars.Sx,
                // rot_pars.Sy);
                break;
            case 'M': // reflection
                aff_func = 2;
                // IRT_TRACE_UTILS("* %s reflection and reflection matrix:\t\t",
                // irt_refl_mode_s[rot_pars.reflection_mode]);
                break;
            case 'T': // shearing
                aff_func = 3;
                // IRT_TRACE_UTILS("* %s shearing and shearing matrix:\t\t", irt_shr_mode_s[rot_pars.shear_mode]);
                break;
            default: aff_func = 4; break; // non
        }
        if (aff_func != 4) {
            rot_pars.affine_flags[aff_func] = 1;
            // IRT_TRACE_UTILS("[%5.2f %5.2f]\n", aff_basic_matrix[aff_func][0][0], aff_basic_matrix[aff_func][0][1]);
            // IRT_TRACE_UTILS("\t\t\t\t\t\t\t[%5.2f %5.2f]\n", aff_basic_matrix[aff_func][1][0],
            // aff_basic_matrix[aff_func][1][1]);
        }
        for (uint8_t r = 0; r < 2; r++) {
            for (uint8_t c = 0; c < 2; c++) {
                acc_matrix[acc_idx & 1][r][c] = 0;
                for (uint8_t idx = 0; idx < 2; idx++) {
                    // M1[r][c] += M[r][idx] * shearing_matrix[idx][c];
                    acc_matrix[acc_idx & 1][r][c] +=
                        acc_matrix[(acc_idx + 1) & 1][r][idx] * aff_basic_matrix[aff_func][idx][c];
                }
            }
        }
        acc_idx = (acc_idx + 1) & 1;
    }

    rot_pars.M11d = acc_matrix[(acc_idx + 1) % 2][0][0];
    rot_pars.M12d = acc_matrix[(acc_idx + 1) % 2][0][1];
    rot_pars.M21d = acc_matrix[(acc_idx + 1) % 2][1][0];
    rot_pars.M22d = acc_matrix[(acc_idx + 1) % 2][1][1];

    IRT_TRACE_UTILS("* Affine matrix:\t\t\t\t\t[%5.2f %5.2f]\n", rot_pars.M11d, rot_pars.M12d);
    IRT_TRACE_UTILS("*\t\t\t\t\t\t\t[%5.2f %5.2f]\n", rot_pars.M21d, rot_pars.M22d);
    IRT_TRACE_UTILS("---------------------------------------------------------------------\n");

    if (irt_desc.irt_mode == e_irt_affine ||
        (irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_affine) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_affine) ||
        (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection &&
         rot_pars.proj_mode == e_irt_affine))
        irt_aff_coefs_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
}

void irt_projection_coefs_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    // double Xoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Xc;
    // double Yoc = (double)irt_top->irt_desc[desc].image_par[OIMAGE].Yc;
    double Xci = (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Yci = (double)irt_desc.image_par[IIMAGE].Yc / 2;
    double tanx =
        rot_pars.shear_mode & 1 ? -1.0 / tan((90.0 - rot_pars.irt_angles[e_irt_angle_shr_x]) * M_PI / 180.0) : 0;
    double tany =
        rot_pars.shear_mode & 2 ? -1.0 / tan((90.0 - rot_pars.irt_angles[e_irt_angle_shr_y]) * M_PI / 180.0) : 0;
    // reflection in projection mode done by rotation around relative axis

    double proj_sin[e_irt_angle_yaw + 1], proj_cos[e_irt_angle_yaw + 1]; //, proj_tan[3];
    for (uint8_t i = e_irt_angle_roll; i <= e_irt_angle_yaw; i++) {
        proj_sin[i] = sin(rot_pars.irt_angles_adj[i] * M_PI / 180.0);
        proj_cos[i] = cos(rot_pars.irt_angles_adj[i] * M_PI / 180.0);
        if (rot_pars.irt_angles_adj[i] == 90 || rot_pars.irt_angles_adj[i] == -90)
            proj_cos[i] = 0; // fix C++ bug that cos(90) is not zero
        // proj_tan[i] = tan(rot_pars.proj_angle[i] * M_PI / 180.0);
    }

    double Wd = rot_pars.proj_Wd;
    double Zd = rot_pars.proj_Zd;
    // double Sx = rot_pars.Sx;
    // double Sy = rot_pars.Sy;
    double proj_R[e_irt_angle_shear + 2][3][3];

    // IRT_TRACE_UTILS("Task %d: Generating Projection coefficients for %s mode with: ", desc,
    // irt_proj_mode_s[rot_pars.proj_mode]);

    switch (rot_pars.proj_mode) {
        case e_irt_rotation: // rotation
            // IRT_TRACE_UTILS("%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
            rot_pars.prj_Ad[0] = rot_pars.cosd;
            rot_pars.prj_Ad[1] = rot_pars.sind;
            rot_pars.prj_Ad[2] = /*-rot_pars.proj_A[0] * Xoc - rot_pars.proj_A[1] * Yoc*/ +Xci;
            rot_pars.prj_Cd[0] = 0;
            rot_pars.prj_Cd[1] = 0;
            rot_pars.prj_Cd[2] = 1;
            rot_pars.prj_Bd[0] = -rot_pars.sind;
            rot_pars.prj_Bd[1] = rot_pars.cosd;
            rot_pars.prj_Bd[2] = /*-rot_pars.proj_B[0] * Xoc - rot_pars.proj_B[1] * Yoc*/ +Yci;
            rot_pars.prj_Dd[0] = 0;
            rot_pars.prj_Dd[1] = 0;
            rot_pars.prj_Dd[2] = 1;
            break;
        case e_irt_affine: // affine
            // IRT_TRACE_UTILS("%s affine mode with: \n", rot_pars.affine_mode);
            if (rot_pars.affine_flags[e_irt_aff_rotation])
                IRT_TRACE_UTILS("* %3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
            if (rot_pars.affine_flags[e_irt_aff_scaling])
                IRT_TRACE_UTILS("* %3.2f/%3.2f scaling factors\n", rot_pars.Sx, rot_pars.Sy);
            if (rot_pars.affine_flags[e_irt_aff_reflection])
                IRT_TRACE_UTILS("* %s reflection mode\n", irt_refl_mode_s[rot_pars.reflection_mode]);
            if (rot_pars.affine_flags[e_irt_aff_shearing])
                IRT_TRACE_UTILS("* %s shearing mode\n", irt_shr_mode_s[rot_pars.shear_mode]);
            rot_pars.prj_Ad[0] = rot_pars.M11d;
            rot_pars.prj_Ad[1] = rot_pars.M12d;
            rot_pars.prj_Ad[2] = /*-rot_pars.proj_A[0] * Xoc - rot_pars.proj_A[1] * Yoc +*/ Xci;
            rot_pars.prj_Cd[0] = 0;
            rot_pars.prj_Cd[1] = 0;
            rot_pars.prj_Cd[2] = 1;
            rot_pars.prj_Bd[0] = rot_pars.M21d;
            rot_pars.prj_Bd[1] = rot_pars.M22d;
            rot_pars.prj_Bd[2] = /*-rot_pars.proj_B[0] * Xoc - rot_pars.proj_B[1] * Yoc + */ Yci;
            rot_pars.prj_Dd[0] = 0;
            rot_pars.prj_Dd[1] = 0;
            rot_pars.prj_Dd[2] = 1;
            break;
        case e_irt_projection:
            // IRT_TRACE_UTILS("\n* %s order, roll %3.2f, pitch %3.2f, yaw %3.2f, Sx %3.2f, Sy %3.2f, Zd %3.2f, Wd
            // %3.2f\n* %s shear mode, X shear %3.2f, Y shear %3.2f\n", 	rot_pars.proj_order,
            // rot_pars.irt_angles[e_irt_angle_roll], rot_pars.irt_angles[e_irt_angle_pitch],
            // rot_pars.irt_angles[e_irt_angle_yaw], rot_pars.Sx, rot_pars.Sy, rot_pars.proj_Zd, rot_pars.proj_Wd,
            //	irt_shr_mode_s[rot_pars.shear_mode],rot_pars.irt_angles[e_irt_angle_shr_x],
            // rot_pars.irt_angles[e_irt_angle_shr_y]); IRT_TRACE_UTILS("* adjusted: roll %3.2f, pitch %3.2f, yaw
            // %3.2f\n", rot_pars.irt_angles_adj[e_irt_angle_roll], rot_pars.irt_angles_adj[e_irt_angle_pitch],
            // rot_pars.irt_angles_adj[e_irt_angle_yaw]);

            // defining rotation matrix for each rotation axis
            proj_R[e_irt_angle_pitch][0][0] = 1.0;
            proj_R[e_irt_angle_pitch][0][1] = 0.0;
            proj_R[e_irt_angle_pitch][0][2] = 0.0;
            proj_R[e_irt_angle_pitch][1][0] = 0.0;
            proj_R[e_irt_angle_pitch][1][1] = proj_cos[e_irt_angle_pitch];
            proj_R[e_irt_angle_pitch][1][2] = -proj_sin[e_irt_angle_pitch];
            proj_R[e_irt_angle_pitch][2][0] = 0.0;
            proj_R[e_irt_angle_pitch][2][1] = proj_sin[e_irt_angle_pitch];
            proj_R[e_irt_angle_pitch][2][2] = proj_cos[e_irt_angle_pitch];

            proj_R[e_irt_angle_yaw][0][0] = proj_cos[e_irt_angle_yaw];
            proj_R[e_irt_angle_yaw][0][1] = 0.0;
            proj_R[e_irt_angle_yaw][0][2] = proj_sin[e_irt_angle_yaw];
            proj_R[e_irt_angle_yaw][1][0] = 0.0;
            proj_R[e_irt_angle_yaw][1][1] = 1.0;
            proj_R[e_irt_angle_yaw][1][2] = 0.0;
            proj_R[e_irt_angle_yaw][2][0] = -proj_sin[e_irt_angle_yaw];
            proj_R[e_irt_angle_yaw][2][1] = 0.0;
            proj_R[e_irt_angle_yaw][2][2] = proj_cos[e_irt_angle_yaw];

            proj_R[e_irt_angle_roll][0][0] = proj_cos[e_irt_angle_roll];
            proj_R[e_irt_angle_roll][0][1] = -proj_sin[e_irt_angle_roll];
            proj_R[e_irt_angle_roll][0][2] = 0.0;
            proj_R[e_irt_angle_roll][1][0] = proj_sin[e_irt_angle_roll];
            proj_R[e_irt_angle_roll][1][1] = proj_cos[e_irt_angle_roll];
            proj_R[e_irt_angle_roll][1][2] = 0.0;
            proj_R[e_irt_angle_roll][2][0] = 0.0;
            proj_R[e_irt_angle_roll][2][1] = 0.0;
            proj_R[e_irt_angle_roll][2][2] = 1.0;

            proj_R[e_irt_angle_shear][0][0] = 1.0;
            proj_R[e_irt_angle_shear][0][1] = tanx;
            proj_R[e_irt_angle_shear][0][2] = 0.0;
            proj_R[e_irt_angle_shear][1][0] = tany;
            proj_R[e_irt_angle_shear][1][1] = 1.0;
            proj_R[e_irt_angle_shear][1][2] = 0.0;
            proj_R[e_irt_angle_shear][2][0] = 0.0;
            proj_R[e_irt_angle_shear][2][1] = 0.0;
            proj_R[e_irt_angle_shear][2][2] = 1.0;

            // reflection in projection mode done by rotation around relative axis
            proj_R[4][0][0] = 1.0;
            proj_R[4][0][1] = 0.0;
            proj_R[4][0][2] = 0.0;
            proj_R[4][1][0] = 0.0;
            proj_R[4][1][1] = 1.0;
            proj_R[4][1][2] = 0.0;
            proj_R[4][2][0] = 0.0;
            proj_R[4][2][1] = 0.0;
            proj_R[4][2][2] = 1.0;

            rot_pars.proj_T[0] = -rot_pars.proj_R[0][2] * rot_pars.proj_Wd;
            rot_pars.proj_T[1] = -rot_pars.proj_R[1][2] * rot_pars.proj_Wd;
            rot_pars.proj_T[2] = 0;

            // calculating total rotation matrix
            uint8_t aff_func = 0, acc_idx = 1;

            double acc_matrix[2][3][3] = {
                {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, // unit
                {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}} // accumulated
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
                            // M1[r][c] += M[r][idx] * shearing_matrix[idx][c];
                            acc_matrix[acc_idx & 1][r][c] +=
                                acc_matrix[(acc_idx + 1) & 1][r][idx] * proj_R[aff_func][idx][c];
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

#if 0
		IRT_TRACE_UTILS("--------------------------------------------------------------------------------------------------------\n");
		for (uint8_t matrix = e_irt_angle_roll; matrix <= e_irt_angle_shear; matrix++) {
			IRT_TRACE_UTILS("|   %s matrix     ", irt_prj_matrix_s[matrix]);
		}
		IRT_TRACE_UTILS("|  Rotation matrix  |\n");

		for (uint8_t row = 0; row < 3; row++) {
			IRT_TRACE_UTILS("|");
			for (uint8_t matrix = e_irt_angle_roll; matrix <= e_irt_angle_shear; matrix++) {
				for (uint8_t col = 0; col < 3; col++) {
					IRT_TRACE_UTILS("%5.2f ", proj_R[matrix][row][col]);
				}
				IRT_TRACE_UTILS(" | ");
			}
			for (uint8_t col = 0; col < 3; col++) {
				IRT_TRACE_UTILS("%5.2f ", rot_pars.proj_R[row][col]);
			}
			IRT_TRACE_UTILS("|\n");
		}
		IRT_TRACE_UTILS("--------------------------------------------------------------------------------------------------------\n");
#endif
            // A0 = Zd((R21*Xic+R22*Yic)*R32-(R31*Xic+R32*Yic-R33*Wd)*R22), A1 =
            // Zd((R31*Xic+R32*Yic)*R12-(R11*Xic+R12*Yic)*R32), A2 = Zd^2((R11*Xic+R12*Yic)*R22-(R21*Xic+R22*Yic)*R12)
            // B0 = Zd((R31*Xic+R32*Yic)*R21-(R21*Xic+R22*Yic-R33*Wd)*R31), B1 =
            // Zd((R11*Xic+R12*Yic)*R31-(R31*Xic+R32*Yic)*R11), B2 = Zd^2((R21*Xic+R22*Yic)*R11-(R11*Xic+R12*Yic)*R21)
            // C0 = Zd(R21*R32-R22*R31),									   C1 = Zd(R31*R12-R32*R11),
            // C2 = Zd^2(R11*R22-R12*R21) applying scaling
            double R11 = rot_pars.proj_R[0][0] * rot_pars.Sx;
            double R12 = rot_pars.proj_R[0][1] * rot_pars.Sy;
            // double R13 = rot_pars.proj_R[0][2];
            double R21 = rot_pars.proj_R[1][0] * rot_pars.Sx;
            double R22 = rot_pars.proj_R[1][1] * rot_pars.Sy;
            // double R23 = rot_pars.proj_R[1][2];
            double R31 = rot_pars.proj_R[2][0] * rot_pars.Sx;
            double R32 = rot_pars.proj_R[2][1] * rot_pars.Sy;
            double R33 = rot_pars.proj_R[2][2];

            // calculating projection coefficients
            rot_pars.prj_Ad[0] =
                ((R21 * Xci + R22 * Yci) * R32 - (R31 * Xci + R32 * Yci - R33 * Wd) * R22) / rot_pars.Sy;
            rot_pars.prj_Ad[1] =
                ((R31 * Xci + R32 * Yci - R33 * Wd) * R12 - (R11 * Xci + R12 * Yci) * R32) / rot_pars.Sy;
            rot_pars.prj_Ad[2] = Zd * ((R11 * Xci + R12 * Yci) * R22 - (R21 * Xci + R22 * Yci) * R12) / rot_pars.Sy;

            rot_pars.prj_Bd[0] =
                ((R31 * Xci + R32 * Yci - R33 * Wd) * R21 - (R21 * Xci + R22 * Yci) * R31) / rot_pars.Sx;
            rot_pars.prj_Bd[1] =
                ((R11 * Xci + R12 * Yci) * R31 - (R31 * Xci + R32 * Yci - R33 * Wd) * R11) / rot_pars.Sx;
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

void irt_mesh_matrix_calc(irt_cfg_pars& irt_cfg,
                          rotation_par& rot_pars,
                          irt_desc_par& irt_desc,
                          uint8_t       desc,
                          bool          trace)
{

    irt_desc.image_par[MIMAGE].W = irt_desc.image_par[OIMAGE].W;
    irt_desc.image_par[MIMAGE].S = irt_desc.image_par[OIMAGE].S;
    irt_desc.image_par[MIMAGE].H = irt_desc.image_par[OIMAGE].H;

    if (rot_pars.mesh_Sh != 1.0) {
        irt_desc.mesh_sparse_h = 1;
        irt_desc.mesh_Gh       = (uint32_t)rint(pow(2.0, IRT_MESH_G_PREC) / rot_pars.mesh_Sh);
        irt_desc.image_par[MIMAGE].W =
            (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].W - 1) / rot_pars.mesh_Sh) + 1;
        irt_desc.image_par[MIMAGE].S =
            (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].S - 1) / rot_pars.mesh_Sh) + 1;
    }
    if (rot_pars.mesh_Sv != 1.0) {
        irt_desc.mesh_sparse_v = 1;
        irt_desc.mesh_Gv       = (uint32_t)rint(pow(2.0, IRT_MESH_G_PREC) / rot_pars.mesh_Sv);
        irt_desc.image_par[MIMAGE].H =
            (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].H - 1) / rot_pars.mesh_Sv) + 1;
    }

    if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) { // stripe height is multiple of 2
        irt_desc.image_par[MIMAGE].H = (uint16_t)(
            ceil((double)irt_desc.image_par[MIMAGE].H / IRT_MM_DYN_MODE_LINE_RELEASE) * IRT_MM_DYN_MODE_LINE_RELEASE);
        if (trace)
            IRT_TRACE_UTILS(
                "Mesh matrix calculation: adjusting Hm to %d be multiple of 2 because of mesh buffer dynamic format\n",
                irt_desc.image_par[MIMAGE].H);
    }

    if (trace) {
        IRT_TRACE_UTILS("----------------------------------------------------------------------------------------------"
                        "----------\n");
        IRT_TRACE_UTILS("Task %d: Generating mesh matrix for %s mode with ", desc, irt_mesh_mode_s[rot_pars.mesh_mode]);
        switch (rot_pars.mesh_mode) {
            case e_irt_rotation: IRT_TRACE_UTILS("%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]); break;
            case e_irt_affine:
                IRT_TRACE_UTILS("%s affine mode with\n", rot_pars.affine_mode);
                if (rot_pars.affine_flags[e_irt_aff_rotation])
                    IRT_TRACE_UTILS("%3.2f rotation angle\n", rot_pars.irt_angles[e_irt_angle_rot]);
                if (rot_pars.affine_flags[e_irt_aff_scaling])
                    IRT_TRACE_UTILS("%3.2f/%3.2f scaling factors\n", rot_pars.Sx, rot_pars.Sy);
                if (rot_pars.affine_flags[e_irt_aff_reflection])
                    IRT_TRACE_UTILS("%s reflection mode\n", irt_refl_mode_s[rot_pars.reflection_mode]);
                if (rot_pars.affine_flags[e_irt_aff_shearing])
                    IRT_TRACE_UTILS("%s shearing mode\n", irt_shr_mode_s[rot_pars.shear_mode]);
                break;
            case e_irt_projection:
                IRT_TRACE_UTILS("\n* %s order, roll %3.2f, pitch %3.2f, yaw %3.2f, Sx %3.2f, Sy %3.2f, Zd %3.2f, Wd "
                                "%3.2f\n* %s shear mode, X shear %3.2f, Y shear %3.2f\n",
                                rot_pars.proj_order,
                                rot_pars.irt_angles[e_irt_angle_roll],
                                rot_pars.irt_angles[e_irt_angle_pitch],
                                rot_pars.irt_angles[e_irt_angle_yaw],
                                rot_pars.Sx,
                                rot_pars.Sy,
                                rot_pars.proj_Zd,
                                rot_pars.proj_Wd,
                                irt_shr_mode_s[rot_pars.shear_mode],
                                rot_pars.irt_angles[e_irt_angle_shr_x],
                                rot_pars.irt_angles[e_irt_angle_shr_y]);
                IRT_TRACE_UTILS("\n* adjusted: roll %3.2f, pitch %3.2f, yaw %3.2f\n",
                                rot_pars.irt_angles_adj[e_irt_angle_roll],
                                rot_pars.irt_angles_adj[e_irt_angle_pitch],
                                rot_pars.irt_angles_adj[e_irt_angle_yaw]);
                break;
            case e_irt_mesh:
                IRT_TRACE_UTILS("%s order %3.2f distortion\n", irt_mesh_order_s[rot_pars.mesh_order], rot_pars.dist_r);
                break;
        }
        IRT_TRACE_UTILS("----------------------------------------------------------------------------------------------"
                        "----------\n");
    }

    if (irt_desc.irt_mode == e_irt_mesh) {
        irt_mesh_full_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
        irt_mesh_sparse_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
        irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
    }
}

void irt_mesh_full_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

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

    // memset(irt_cfg.mesh_images.mesh_image_full, 0, sizeof(irt_cfg.mesh_images.mesh_image_full));

    uint16_t Ho  = irt_desc.image_par[OIMAGE].H;
    uint16_t So  = irt_desc.image_par[OIMAGE].S;
    double   Xoc = (double)irt_desc.image_par[OIMAGE].Xc / 2;
    double   Yoc = (double)irt_desc.image_par[OIMAGE].Yc / 2;
    // uint16_t Hi = irt_desc.image_par[IIMAGE].H;
    // uint16_t Wi = irt_desc.image_par[IIMAGE].W;
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
            xo         = (double)col - Xoc;
            yo         = (double)row - Yoc;
            r          = sqrt(pow(rot_pars.dist_x * xo, 2) + pow(rot_pars.dist_y * yo, 2));
            dist_ratio = 1 + rot_pars.dist_r * r / sqrt(pow(So / 2, 2) + pow(Ho / 2, 2));

            if (rot_pars.mesh_order == 0) { // predistortion
                xi1 = xo * dist_ratio; // *rot_pars.dist_x;
                yi1 = yo * dist_ratio; // *rot_pars.dist_y;
            } else {
                xi1 = xo;
                yi1 = yo;
            }
            switch (rot_pars.mesh_mode) {
                case e_irt_rotation:
                    xi2 = cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * xi1 +
                          sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * yi1;
                    yi2 = -sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * xi1 +
                          cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180) * yi1;
                    break;
                case e_irt_affine:
                    xi2 = rot_pars.M11d * xi1 + rot_pars.M12d * yi1;
                    yi2 = rot_pars.M21d * xi1 + rot_pars.M22d * yi1;
                    break;
                case e_irt_projection:
                    xi2 = (rot_pars.prj_Ad[0] * xi1 + rot_pars.prj_Ad[1] * yi1 + rot_pars.prj_Ad[2]) /
                              (rot_pars.prj_Cd[0] * xi1 + rot_pars.prj_Cd[1] * yi1 + rot_pars.prj_Cd[2]) -
                          Xic;
                    yi2 = (rot_pars.prj_Bd[0] * xi1 + rot_pars.prj_Bd[1] * yi1 + rot_pars.prj_Bd[2]) /
                              (rot_pars.prj_Dd[0] * xi1 + rot_pars.prj_Dd[1] * yi1 + rot_pars.prj_Dd[2]) -
                          Yic;
                    break;
                default: // distortion
                    xi2 = xi1;
                    yi2 = yi1;
                    break;
            }

            if (rot_pars.mesh_order == 1) { // post distortion
                irt_cfg.mesh_images.mesh_image_full[row][col].x = (float)(xi2 * dist_ratio /* rot_pars.dist_x*/ + Xic);
                irt_cfg.mesh_images.mesh_image_full[row][col].y = (float)(yi2 * dist_ratio /* rot_pars.dist_y*/ + Yic);
            } else {
                irt_cfg.mesh_images.mesh_image_full[row][col].x = (float)(xi2 + Xic);
                irt_cfg.mesh_images.mesh_image_full[row][col].y = (float)(yi2 + Yic);
            }

            // relative matrix
            irt_cfg.mesh_images.mesh_image_rel[row][col].x = (float)irt_cfg.mesh_images.mesh_image_full[row][col].x;
            irt_cfg.mesh_images.mesh_image_rel[row][col].y = (float)irt_cfg.mesh_images.mesh_image_full[row][col].y;

            if (irt_desc.mesh_rel_mode) {
                irt_cfg.mesh_images.mesh_image_rel[row][col].x -= (float)col;
                irt_cfg.mesh_images.mesh_image_rel[row][col].y -= (float)row;
            }

            //			IRT_TRACE_UTILS("Mesh_image_full[%d, %d] = (%f, %f)\n", row, col,
            // irt_cfg.mesh_images.mesh_image_full[row][col].x, irt_cfg.mesh_images.mesh_image_full[row][col].y);
            irt_map_image_pars_update<mesh_xy_fp32_meta, double>(
                irt_cfg, irt_desc, irt_cfg.mesh_images.mesh_image_full, row, col, Ymax_line, fptr_full, fptr_full2);
            if (print_out_files)
                IRT_TRACE_TO_RES_UTILS(fptr_rel,
                                       "[%d, %d] = [%4.2f, %4.2f]\n",
                                       row,
                                       col,
                                       (double)irt_cfg.mesh_images.mesh_image_rel[row][col].x,
                                       (double)irt_cfg.mesh_images.mesh_image_rel[row][col].y);
        }
    }

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
    if (print_out_files) {
        fclose(fptr_full);
        fclose(fptr_rel);
    }
#endif
}

void irt_mesh_sparse_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
    FILE *f2 = nullptr, *f3 = nullptr, *f4 = nullptr;
    char  f2_name[50], f3_name[50], f4_name[50];
    FILE* f5 = nullptr;
    char  f5_name[50];

    sprintf(f2_name, "Mesh_matrix_fi16_%d.txt", desc);
    sprintf(f3_name, "Mesh_matrix_ui32_%d.txt", desc);
    sprintf(f4_name, "Mesh_matrix_fp32_%d.txt", desc);
    sprintf(f5_name, "sival_warp_image_task%d.txt", desc);
    if (print_out_files) {
        f2 = fopen(f2_name, "w");
        f3 = fopen(f3_name, "w");
        f4 = fopen(f4_name, "w");
        f5 = fopen(f5_name, "w");
    }
#endif
    // generating sparse matrix & checking precision
    int32_t i0, j0, i1, j1;
    double  xo, yo, xf, yf;
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
                         (double)irt_cfg.mesh_images.mesh_image_rel[i1][j0].x * yf) *
                            (1.0 - xf) +
                        ((double)irt_cfg.mesh_images.mesh_image_rel[i0][j1].x * (1.0 - yf) +
                         (double)irt_cfg.mesh_images.mesh_image_rel[i1][j1].x * yf) *
                            xf);

            irt_cfg.mesh_images.mesh_image_fp32[row][col].y =
                (float)(((double)irt_cfg.mesh_images.mesh_image_rel[i0][j0].y * (1.0 - yf) +
                         (double)irt_cfg.mesh_images.mesh_image_rel[i1][j0].y * yf) *
                            (1.0 - xf) +
                        ((double)irt_cfg.mesh_images.mesh_image_rel[i0][j1].y * (1.0 - yf) +
                         (double)irt_cfg.mesh_images.mesh_image_rel[i1][j1].y * yf) *
                            xf);

            xi32 = (int32_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x *
                                 pow(2.0, irt_desc.mesh_point_location));
            yi32 = (int32_t)rint((double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y *
                                 pow(2.0, irt_desc.mesh_point_location));

            float   M_max     = std::fmax(fabs(irt_cfg.mesh_images.mesh_image_fp32[row][col].x),
                                    fabs(irt_cfg.mesh_images.mesh_image_fp32[row][col].y));
            int     M_max_int = (int)floor(M_max); // integer part of max value
            uint8_t M_I_width = (uint8_t)ceil(log2(M_max_int + 1));
            uint8_t M_F_width = 15 - M_I_width; // remained bits for fraction

            if (irt_desc.irt_mode == e_irt_mesh &&
                irt_desc.mesh_format == e_irt_mesh_flex) { // relevant only in mesh_flex format and mesh mode
                if (M_F_width <
                    irt_desc
                        .mesh_point_location) { ////mesh_point_location > bits allowed for fraction presentation of M
                    if (rot_pars.mesh_prec_auto_adj == 0) { // correction is not allowed
                        if (xi32 >= (int32_t)pow(2, 15) || xi32 <= -(int32_t)pow(2, 15)) {
                            IRT_TRACE_UTILS("Error: Mesh X value %f for pixel[%d][%d] cannot fit FI16 format with "
                                            "fixed point format S%d.%d as %d\n",
                                            (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x,
                                            row,
                                            col,
                                            15 - irt_desc.mesh_point_location,
                                            irt_desc.mesh_point_location,
                                            xi32);
                            IRT_TRACE_TO_RES_UTILS(test_res,
                                                   "Error: Mesh X value %f for pixel[%d][%d] cannot fit FI16 format "
                                                   "with fixed point format S%d.%d as %d\n",
                                                   (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x,
                                                   row,
                                                   col,
                                                   15 - irt_desc.mesh_point_location,
                                                   irt_desc.mesh_point_location,
                                                   xi32);
                        }
                        if (yi32 >= (int32_t)pow(2, 15) || yi32 <= -(int32_t)pow(2, 15)) {
                            IRT_TRACE_UTILS("Error: Mesh Y value %f for pixel[%d][%d] cannot fit FI16 format with "
                                            "fixed point format S%d.%d as %d\n",
                                            (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y,
                                            row,
                                            col,
                                            15 - irt_desc.mesh_point_location,
                                            irt_desc.mesh_point_location,
                                            yi32);
                            IRT_TRACE_TO_RES_UTILS(test_res,
                                                   "Error: Mesh Y value %f for pixel[%d][%d] cannot fit FI16 format "
                                                   "with fixed point format S%d.%d as %d\n",
                                                   (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y,
                                                   row,
                                                   col,
                                                   15 - irt_desc.mesh_point_location,
                                                   irt_desc.mesh_point_location,
                                                   yi32);
                        }
                        IRT_CLOSE_FAILED_TEST(0);

                    } else {
                        IRT_TRACE_UTILS("Task %d: Adjusting mesh_point_location to %d bits\n", desc, M_F_width);
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

    if (print_out_files) {
        IRT_TRACE_TO_RES_UTILS(f5,
                               "// IMAGE DIMs H x W : %0d x %0d  \t data-type : %0d \n",
                               irt_desc.image_par[MIMAGE].H,
                               irt_desc.image_par[MIMAGE].S,
                               (irt_desc.mesh_format == e_irt_mesh_fp32) ? 4 : 2);
    }

    // generating fixed point matrix based on final precision
    for (int row = 0; row < irt_desc.image_par[MIMAGE].H; row++) {
        for (int col = 0; col < irt_desc.image_par[MIMAGE].S; col++) {

            irt_cfg.mesh_images.mesh_image_fi16[row][col].x = (int16_t)rint(
                (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x * pow(2.0, irt_desc.mesh_point_location));
            irt_cfg.mesh_images.mesh_image_fi16[row][col].y = (int16_t)rint(
                (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y * pow(2.0, irt_desc.mesh_point_location));

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
            mimage_row_start =
                (uint64_t)row * irt_desc.image_par[MIMAGE].S * ((irt_desc.mesh_format == e_irt_mesh_fp32) ? 8 : 4);
            mimage_col_start = (uint64_t)col * ((irt_desc.mesh_format == e_irt_mesh_fp32) ? 8 : 4);
            mimage_pxl_start = mimage_addr_start + mimage_row_start + mimage_col_start;
#if defined(STANDALONE_ROTATOR) || defined(HABANA_SIMULATION) // ROTSIM_DV_INTGR
            for (uint8_t byte = 0; byte < ((irt_desc.mesh_format == e_irt_mesh_fp32) ? 4 : 2); byte++) {
                if (irt_desc.mesh_format == e_irt_mesh_flex)
                    ext_mem[mimage_pxl_start + byte] =
                        (irt_cfg.mesh_images.mesh_image_fi16[row][col].x >> (8 * byte)) & 0xff;
                else
                    ext_mem[mimage_pxl_start + byte] =
                        (IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].x) >>
                         (8 * byte)) &
                        0xff;
            }
            mimage_pxl_start += ((irt_desc.mesh_format == e_irt_mesh_fp32) ? 4 : 2);
            for (uint8_t byte = 0; byte < ((irt_desc.mesh_format == e_irt_mesh_fp32) ? 4 : 2); byte++) {
                if (irt_desc.mesh_format == e_irt_mesh_flex)
                    ext_mem[mimage_pxl_start + byte] =
                        (irt_cfg.mesh_images.mesh_image_fi16[row][col].y >> (8 * byte)) & 0xff;
                else
                    ext_mem[mimage_pxl_start + byte] =
                        (IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].y) >>
                         (8 * byte)) &
                        0xff;
            }
#endif
            if (print_out_files) {
                IRT_TRACE_TO_RES_UTILS(f2,
                                       "%04x%04x\n",
                                       ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].y & 0xffff),
                                       ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].x & 0xffff));
                IRT_TRACE_TO_RES_UTILS(
                    f3,
                    "%08x%08x\n",
                    IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].y),
                    IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].x));
                IRT_TRACE_TO_RES_UTILS(f4,
                                       "[%d, %d] = [%4.2f, %4.2f]\n",
                                       row,
                                       col,
                                       (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].x,
                                       (double)irt_cfg.mesh_images.mesh_image_fp32[row][col].y);

                if (irt_desc.mesh_format == e_irt_mesh_fp32) {
                    IRT_TRACE_TO_RES_UTILS(
                        f5,
                        "%08x%08x\n",
                        IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].y),
                        IRT_top::IRT_UTILS::irt_float_to_fp32(irt_cfg.mesh_images.mesh_image_fp32[row][col].x));
                } else {
                    IRT_TRACE_TO_RES_UTILS(f5,
                                           "%04x%04x\n",
                                           ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].y & 0xffff),
                                           ((uint32_t)irt_cfg.mesh_images.mesh_image_fi16[row][col].x & 0xffff));
                }
            }
#endif
        }
    }
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
    if (print_out_files) {
        fclose(f2);
        fclose(f3);
        fclose(f4);
        fclose(f5);
    }
#endif
}

void irt_mesh_interp_matrix_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    // return;
    FILE *fptr_vals = nullptr, *fptr_pars = nullptr;
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

    // interpolation of full matrix from sparse
    int64_t  xm, ym;
    uint32_t xf, yf;
    float    mp_value[2][4], mi[2], xw0, xw1, yw0, yw1;
    int16_t  i0, j0, i1, j1;
    bool     p_valid[4];
    double   mesh_matrix_acc_err = 0, mesh_matrix_max_err = 0, mesh_matrix_max_err_left = 0,
           mesh_matrix_max_err_right = 0, mesh_matrix_error = 0;

    for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {
        for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].S; col++) {

            ym = (int64_t)row;
            if (irt_desc.mesh_sparse_v) { // sparse in v direction
                ym = ym * (int64_t)irt_desc.mesh_Gv;
            } else {
                ym = ym << IRT_MESH_G_PREC;
            }

            xm = (int64_t)col;
            if (irt_desc.mesh_sparse_h) { // sparse in h direction
                xm = xm * (int64_t)irt_desc.mesh_Gh;
            } else {
                xm = xm << IRT_MESH_G_PREC;
            }

            j0 = (int16_t)(xm >> IRT_MESH_G_PREC);
            j1 = j0 + 1;
            xf = (uint32_t)(xm - ((int64_t)j0 << IRT_MESH_G_PREC));

            i0 = (int16_t)(ym >> IRT_MESH_G_PREC);
            i1 = i0 + 1;
            yf = (uint32_t)(ym - ((int64_t)i0 << IRT_MESH_G_PREC));

            if (j1 >= irt_desc.image_par[MIMAGE].S) { // reset weight for oob
                xf = 0;
            }
            if (i1 >= irt_desc.image_par[MIMAGE].H) { // reset weight for oob
                yf = 0;
            }

            i1 = IRT_top::IRT_UTILS::irt_min_int16(i1, irt_desc.image_par[MIMAGE].H - 1);
            j1 = IRT_top::IRT_UTILS::irt_min_int16(j1, irt_desc.image_par[MIMAGE].S - 1);

            p_valid[0] = i0 >= 0 && i0 < irt_desc.image_par[MIMAGE].H && j0 >= 0 && j0 < irt_desc.image_par[MIMAGE].S;
            p_valid[1] = i0 >= 0 && i0 < irt_desc.image_par[MIMAGE].H && j1 >= 0 && j1 < irt_desc.image_par[MIMAGE].S;
            p_valid[2] = i1 >= 0 && i1 < irt_desc.image_par[MIMAGE].H && j0 >= 0 && j0 < irt_desc.image_par[MIMAGE].S;
            p_valid[3] = i1 >= 0 && i1 < irt_desc.image_par[MIMAGE].H && j1 >= 0 && j1 < irt_desc.image_par[MIMAGE].S;

            ////selecting X for interpolation assign pixel
            mp_value[0][0] = (float)(p_valid[0] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i0][j0].x
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j0].x /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[0][1] = (float)(p_valid[1] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i0][j1].x
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j1].x /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[0][2] = (float)(p_valid[2] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i1][j0].x
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j0].x /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[0][3] = (float)(p_valid[3] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i1][j1].x
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j1].x /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);

            ////selecting Y for interpolation assign pixel
            mp_value[1][0] = (float)(p_valid[0] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i0][j0].y
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j0].y /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[1][1] = (float)(p_valid[1] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i0][j1].y
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i0][j1].y /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[1][2] = (float)(p_valid[2] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i1][j0].y
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j0].y /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);
            mp_value[1][3] = (float)(p_valid[3] ? ((irt_desc.mesh_format == e_irt_mesh_fp32)
                                                       ? irt_cfg.mesh_images.mesh_image_fp32[i1][j1].y
                                                       : (float)irt_cfg.mesh_images.mesh_image_fi16[i1][j1].y /
                                                             (float)pow(2.0, irt_desc.mesh_point_location))
                                                : 0);

            xw0 = (float)((double)xf / pow(2.0, 31));
            xw1 = (float)((pow(2.0, 31) - (double)xf) / pow(2.0, 31));
            yw0 = (float)((double)yf / pow(2.0, 31));
            yw1 = (float)((pow(2.0, 31) - (double)yf) / pow(2.0, 31));

            for (uint8_t j = 0; j < 2; j++) {
                // vertical followed by horizontal
                // as in arch model
                mi[j] = (mp_value[j][0] * yw1 + mp_value[j][2] * yw0) * xw1 +
                        (mp_value[j][1] * yw1 + mp_value[j][3] * yw0) * xw0;
                mi[j] = ((int64_t)rint(mi[j] * (float)pow(2, 31))) / ((float)pow(2, 31));
            }

            irt_cfg.mesh_images.mesh_image_intr[row][col].x = mi[0];
            irt_cfg.mesh_images.mesh_image_intr[row][col].y = mi[1];
            if (irt_desc.mesh_rel_mode) {
                irt_cfg.mesh_images.mesh_image_intr[row][col].x += (double)col;
                irt_cfg.mesh_images.mesh_image_intr[row][col].y += (double)row;
            }

            mesh_matrix_error = fabs(floor((double)irt_cfg.mesh_images.mesh_image_full[row][col].x) -
                                     floor((double)irt_cfg.mesh_images.mesh_image_intr[row][col]
                                               .x)); // ROTSIM_DV_INTGR -- local fix for fabs for float data types
            mesh_matrix_acc_err += mesh_matrix_error;
            mesh_matrix_max_err = fmax(mesh_matrix_max_err, mesh_matrix_error);
            if (col == 0)
                mesh_matrix_max_err_left = fmax(mesh_matrix_max_err_left, mesh_matrix_error);
            if (col == irt_desc.image_par[OIMAGE].S - 1)
                mesh_matrix_max_err_right = fmax(mesh_matrix_max_err_right, mesh_matrix_error);
            irt_map_image_pars_update<mesh_xy_fp64_meta, double>(
                irt_cfg, irt_desc, irt_cfg.mesh_images.mesh_image_intr, row, col, Ymax_line, fptr_vals, fptr_pars);

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

    // rot_pars.IBufH_req = irt_cfg.mesh_images.mesh_image_intr[irt_desc.image_par[OIMAGE].H -
    // 1][irt_desc.image_par[OIMAGE].W - 1].IBufH_req;
    IRT_TRACE_UTILS("Matrix accumulated error %f\n", mesh_matrix_acc_err);
    IRT_TRACE_UTILS("Matrix maximum error %f\n", mesh_matrix_max_err);
    IRT_TRACE_UTILS("Matrix average error %f\n",
                    mesh_matrix_acc_err /
                        ((double)irt_desc.image_par[OIMAGE].W * (double)irt_desc.image_par[OIMAGE].H));
    IRT_TRACE_UTILS("Matrix maximum error left %f\n", mesh_matrix_max_err_left);
    IRT_TRACE_UTILS("Matrix maximum error right %f\n", mesh_matrix_max_err_right);
    if (mesh_matrix_acc_err / ((double)irt_desc.image_par[OIMAGE].W * (double)irt_desc.image_par[OIMAGE].H) > 0.71) {
        rot_pars.mesh_matrix_error = 1;
        IRT_TRACE_UTILS("Interpolated mesh matrix error > 0.5, rotation and affine optimization will be disabled\n");
    }
#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
    if (print_out_files) {
        fclose(fptr_vals);
        fclose(fptr_pars);
    }
#endif
}

template <class mimage_type, class coord_type>
void irt_map_image_pars_update(irt_cfg_pars& irt_cfg,
                               irt_desc_par& irt_desc,
                               mimage_type** mesh_image,
                               uint16_t      row,
                               uint16_t      col,
                               coord_type&   Ymax_line,
                               FILE*         fptr_vals,
                               FILE*         fptr_pars)
{
    bool XiYi_inf = std::isinf(mesh_image[row][col].y) | std::isinf(mesh_image[row][col].x) |
                    std::isnan(mesh_image[row][col].y) | std::isnan(mesh_image[row][col].x);

    if (col == 0) {
        mesh_image[row][col].Ymin = mesh_image[row][col].y;
        Ymax_line                 = mesh_image[row][col].y;
    } else {
        mesh_image[row][col].Ymin =
            std::fmin(mesh_image[row][col - 1].Ymin, mesh_image[row][col].y); // updating from left
        Ymax_line = std::fmax(Ymax_line, mesh_image[row][col].y); // updating from left
    }
    if (row > 0)
        mesh_image[row][col].Ymin_dir = mesh_image[row][col].Ymin >= mesh_image[row - 1][col].Ymin;
    else
        mesh_image[row][col].Ymin_dir = 1;

    if (col > 0)
        mesh_image[row][col].Ymin_dir_swap = mesh_image[row][col - 1].Ymin_dir_swap |
                                             (mesh_image[row][col].Ymin_dir != mesh_image[row][col - 1].Ymin_dir);
    else
        mesh_image[row][col].Ymin_dir_swap = 0;

    if (row == 0)
        mesh_image[row][col].IBufH_req =
            (uint32_t)(ceil((double)Ymax_line) - floor((double)mesh_image[row][col].Ymin)); // updating from top
    else
        mesh_image[row][col].IBufH_req =
            (uint32_t)(ceil((double)Ymax_line) - floor((double)mesh_image[row - 1][col].Ymin)); // updating from top
    mesh_image[row][col].IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, mesh_image[row][col].IBufH_req);

    if (col == 0) {
        mesh_image[row][col].Xi_first = mesh_image[row][col].x;
        mesh_image[row][col].Xi_last  = mesh_image[row][col].x;
        mesh_image[row][col].Yi_first = mesh_image[row][col].y;
        mesh_image[row][col].Yi_last  = mesh_image[row][col].y;
        mesh_image[row][col].XiYi_inf = XiYi_inf;
    } else {
        mesh_image[row][col].Xi_first =
            std::fmin(mesh_image[row][col - 1].Xi_first, mesh_image[row][col].x); // updating from left
        mesh_image[row][col].Xi_last =
            std::fmax(mesh_image[row][col - 1].Xi_last, mesh_image[row][col].x); // updating from left
        mesh_image[row][col].Yi_first =
            std::fmin(mesh_image[row][col - 1].Yi_first, mesh_image[row][col].y); // updating from left
        mesh_image[row][col].Yi_last =
            std::fmax(mesh_image[row][col - 1].Yi_last, mesh_image[row][col].y); // updating from left
        mesh_image[row][col].XiYi_inf = mesh_image[row][col - 1].XiYi_inf | XiYi_inf; // updating from left
    }
    if (row > 0) {
        mesh_image[row][col].Xi_first =
            std::fmin(mesh_image[row][col].Xi_first, mesh_image[row - 1][col].Xi_first); // updating from top
        mesh_image[row][col].Xi_last =
            std::fmax(mesh_image[row][col].Xi_last, mesh_image[row - 1][col].Xi_last); // updating from top
        mesh_image[row][col].Yi_first =
            std::fmin(mesh_image[row][col].Yi_first, mesh_image[row - 1][col].Yi_first); // updating from top
        mesh_image[row][col].Yi_last =
            std::fmax(mesh_image[row][col].Yi_last, mesh_image[row - 1][col].Yi_last); // updating from top
        mesh_image[row][col].IBufH_req =
            (uint32_t)std::max((int32_t)mesh_image[row][col].IBufH_req, (int32_t)mesh_image[row - 1][col].IBufH_req);
        mesh_image[row][col].Ymin_dir_swap = mesh_image[row][col].Ymin_dir_swap |
                                             mesh_image[row - 1][col].Ymin_dir_swap |
                                             (mesh_image[row][col].Ymin_dir != mesh_image[row - 1][col].Ymin_dir);
        mesh_image[row][col].XiYi_inf = mesh_image[row - 1][col].XiYi_inf | XiYi_inf;
    }

    mesh_image[row][col].Si =
        (uint32_t)(ceil(mesh_image[row][col].Xi_last) - floor(mesh_image[row][col].Xi_first)) + 2; // updating from left
    mesh_image[row][col].IBufW_req = irt_IBufW_req_calc(irt_cfg, irt_desc, mesh_image[row][col].Si);

#if (defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV))
    if (fptr_vals != nullptr)
        IRT_TRACE_TO_RES_UTILS(fptr_vals,
                               "[%u, %u] = [%4.8f, %4.8f]\n",
                               row,
                               col,
                               (double)mesh_image[row][col].x,
                               (double)mesh_image[row][col].y);
    if (fptr_pars != nullptr)
        IRT_TRACE_TO_RES_UTILS(fptr_pars,
                               "[%u, %u] = Xi_first %4.8f, Xi_last %4.8f, Si %u, Yi_first %f, Yi_last %f, BufW %u, "
                               "BufH %u, Ymin %f, Ymin_dir %d, Ymin_dir_swap %d, XiYi_inf %d\n",
                               row,
                               col,
                               (double)mesh_image[row][col].Xi_first,
                               (double)mesh_image[row][col].Xi_last,
                               mesh_image[row][col].Si,
                               (double)mesh_image[row][col].Yi_first,
                               (double)mesh_image[row][col].Yi_last,
                               mesh_image[row][col].IBufW_req,
                               mesh_image[row][col].IBufH_req,
                               (double)mesh_image[row][col].Ymin,
                               mesh_image[row][col].Ymin_dir,
                               mesh_image[row][col].Ymin_dir_swap,
                               mesh_image[row][col].XiYi_inf);
#endif
}

template <class mimage_type>
void irt_map_oimage_res_adj_calc(irt_cfg_pars& irt_cfg,
                                 rotation_par& rot_pars,
                                 irt_desc_par& irt_desc,
                                 uint8_t       desc,
                                 mimage_type** mesh_image)
{

    uint16_t So = 0, Ho = 0;
    uint32_t OSize    = 0;
    uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
    uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;

    bool Ymin_dir_cur, Ymin_dir_prv, Ymin_dir_swap = 0;

    for (uint16_t row = 0; row < irt_desc.image_par[OIMAGE].H; row++) {
        for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].W; col++) {

            if (row > 1) {
                Ymin_dir_cur = mesh_image[row - 0][col].Ymin > mesh_image[row - 1][col].Ymin;
                Ymin_dir_prv = mesh_image[row - 1][col].Ymin > mesh_image[row - 2][col].Ymin;
            } else {
                Ymin_dir_cur = 1;
                Ymin_dir_prv = 1;
            }

            Ymin_dir_swap = Ymin_dir_swap | (Ymin_dir_cur != Ymin_dir_prv);

            if ((mesh_image[row][col].IBufW_req <= BufW_act) && (mesh_image[row][col].IBufH_req <= BufH_act) &&
                !Ymin_dir_swap && !mesh_image[row][col].Ymin_dir_swap && mesh_image[row][col].Ymin_dir &&
                mesh_image[row][col].XiYi_inf == 0) {
                if (((uint32_t)row + 1) * ((uint32_t)col + 1) > OSize) {
                    So    = col + 1;
                    Ho    = row + 1;
                    OSize = (uint32_t)So * Ho;
                }
            }
        }
    }

    irt_desc.image_par[OIMAGE].S = So;
    IRT_TRACE_UTILS("2. So %d\n", irt_desc.image_par[OIMAGE].S);
    irt_desc.image_par[OIMAGE].H = Ho;

    if (So > 0 && Ho > 0)
        irt_map_iimage_stripe_adj<mimage_type>(irt_cfg, rot_pars, irt_desc, desc, mesh_image);
}

template <class mimage_type>
void irt_map_iimage_stripe_adj(irt_cfg_pars& irt_cfg,
                               rotation_par& rot_pars,
                               irt_desc_par& irt_desc,
                               uint8_t       desc,
                               mimage_type** mesh_image)
{

    rot_pars.IBufW_req = mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].IBufW_req;
    rot_pars.IBufH_req = mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].IBufH_req;
    rot_pars.Yi_first = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Yi_first;
    rot_pars.Yi_last  = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Yi_last;
    rot_pars.Xi_first = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Xi_first;
    rot_pars.Xi_last  = (double)mesh_image[irt_desc.image_par[OIMAGE].H - 1][irt_desc.image_par[OIMAGE].S - 1].Xi_last;
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
            irt_desc.rot_dir = IRT_ROT_DIR_NEG; // IRT_ROT_DIR_POS;
            irt_desc.rot90   = 0;
            // rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
            break;
    }

    rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first); // -irt_desc.image_par[IIMAGE].S;
}

uint8_t irt_proc_size_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    int64_t Xi_fixed[IRT_ROT_MAX_PROC_SIZE] = {0}, Yi_fixed[IRT_ROT_MAX_PROC_SIZE] = {0};
    int     XL_fixed[IRT_ROT_MAX_PROC_SIZE] = {0}, XR_fixed[IRT_ROT_MAX_PROC_SIZE] = {0},
        YT_fixed[IRT_ROT_MAX_PROC_SIZE] = {0}, YB_fixed[IRT_ROT_MAX_PROC_SIZE] = {0};
    int     int_win_w[IRT_ROT_MAX_PROC_SIZE], int_win_h[IRT_ROT_MAX_PROC_SIZE];
    uint8_t adj_proc_size = 1;
    uint8_t phases_in_parallel, num_tap_cycle, max_proc_size_hcoeff;

    for (uint8_t k = 0; k < IRT_ROT_MAX_PROC_SIZE; k++) { // calculating corners for 8 proc sizes
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
                    default: break;
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
                    default: break;
                }
                break;
            default: break; // Resamp & Rescale mode this proc size calc function wont be used
        }
    }

    // calculating interpolation region corners for 8 proc sizes
    if (irt_desc.irt_mode == e_irt_rescale) {
        // TODO Use mesh precision
        IRT_TRACE_UTILS(
            "num_taps %0d and scaling factor %0f \n", irt_desc.rescale_LUT_x->num_taps, irt_desc.rescale_LUT_x->Gf);
        // irt_desc.rescale_LUT_x->Gf --> irt_desc.mesh_Gh   (S7.24)/2^24 Xi --> S16.31 --> rnd2phase_prec ->

        // adj_proc_size =   (floor(((17 - (irt_desc.rescale_LUT_x->num_taps)*num_phases_h)/irt_desc.rescale_LUT_x->Gh))
        // + 1); //set for number of already processed pixels in group
        adj_proc_size = (floor(((16 - (irt_desc.rescale_LUT_x->num_taps)) / irt_desc.rescale_LUT_x->Gf)) +
                         1); // TODO - Ambili, need to consider orignal format for Gf//set for number of already
                             // processed pixels in group
        if (adj_proc_size == 0) {
            adj_proc_size = 1;
            IRT_TRACE_UTILS("Check - adj_proc_size from eq is 0, making it 1 \n");
        }
        IRT_TRACE_UTILS("Original adj_proc_size %0d \n", adj_proc_size);
        num_tap_cycle        = std::max(irt_desc.rescale_LUT_x->num_taps / 2, irt_desc.rescale_LUT_y->num_taps / 2);
        phases_in_parallel   = floor(16 / irt_desc.rescale_LUT_x->num_taps);
        max_proc_size_hcoeff = num_tap_cycle * phases_in_parallel;

        IRT_TRACE_UTILS("max_proc_size_hcoeff %0d \n", max_proc_size_hcoeff);

        adj_proc_size = std::min(adj_proc_size, max_proc_size_hcoeff);

        if (adj_proc_size > 8)
            adj_proc_size = 8;
        IRT_TRACE_UTILS("Proc_size_calc: adj_proc_size=%d: num_taps-h:%d Gx_prec:%d mesh_Gh:%u\n",
                        adj_proc_size,
                        irt_desc.rescale_LUT_x->num_taps,
                        rot_pars.rescale_Gx_prec,
                        irt_desc.mesh_Gh);
    } else {
        for (uint8_t k = 0; k < IRT_ROT_MAX_PROC_SIZE; k++) { // calculating corners for 8 proc sizes

            XL_fixed[k] = std::min(0, (int)(Xi_fixed[k] >> rot_pars.TOTAL_PREC));
            XR_fixed[k] = std::max(0, (int)(Xi_fixed[k] >> rot_pars.TOTAL_PREC)) + 1;
            YT_fixed[k] = std::min(0, (int)(Yi_fixed[k] >> rot_pars.TOTAL_PREC));
            YB_fixed[k] = std::max(0, (int)(Yi_fixed[k] >> rot_pars.TOTAL_PREC)) + 1;

            int_win_w[k] = XR_fixed[k] - XL_fixed[k] + 2;
            int_win_h[k] = YB_fixed[k] - YT_fixed[k] + 2;
            switch (irt_desc.irt_mode) {
                case e_irt_rotation:
                    //			IRT_TRACE_UTILS("Interpolation window size: k=%d: %dx%d %dx%d\n", k + 1, int_win_h[k],
                    // int_win_w[k], (irt_desc.sind[k] >> IRT_TOTAL_PREC) + 2, (irt_desc.cosd[k] >> IRT_TOTAL_PREC) +
                    // 2);
                    break;
                default: break;
            }

            if ((int_win_h[k] <= IRT_INT_WIN_H) && (int_win_w[k] <= IRT_INT_WIN_W)) { // fit interpolation window
                adj_proc_size = k + 1; // set for number of already processed pixels in group
            }
        }
    }

    if (adj_proc_size > IRT_ROT_MAX_PROC_SIZE) {
        adj_proc_size = IRT_ROT_MAX_PROC_SIZE;
    }

    if (adj_proc_size > 2) {
        // adj_proc_size--;
    }

    // IRT_TRACE_UTILS("Task %d: processing rate is limited to %d pixels/cycle\n", desc, adj_proc_size);

    return adj_proc_size;
}

void irt_mesh_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{
    // IBufW_req is required bytes per mesh lines to be stored in memory, equal to Wm multiplied by pixel size:
    // BufW (byte)						32 64 128 256 512 1024 2048 4096
    // BufW (pixels) for 4 byte/pixel -	 8 16  32  64 128  256  512 1024
    // BufW (pixels) for 8 byte/pixel -	 2  4   8  16  32   64  128  256
    uint16_t Si_log2 =
        (uint16_t)ceil(log2((double)irt_desc.image_par[MIMAGE].S)); // for 90 degree rotation Si+1 is not needed
    uint16_t Si_pxls = (uint16_t)pow(2.0, Si_log2);
    if ((irt_desc.irt_mode == e_irt_rescale) & (irt_desc.image_par[MIMAGE].Ps == 0)) {
        rot_pars.MBufW_req =
            Si_pxls << (irt_desc.image_par[MIMAGE].Ps +
                        1); //+1 is required as Mesh 8 bit data is also used as 16 bit data(padding with 8'b00)
    } else if ((irt_desc.irt_mode == e_irt_resamp_fwd || irt_desc.irt_mode == e_irt_resamp_bwd1 ||
                irt_desc.irt_mode == e_irt_resamp_bwd2) &&
               (irt_desc.mesh_rel_mode == 1 || irt_desc.resize_bli_grad_en == 1)) {
        rot_pars.MBufW_req = Si_pxls << 3; // With relative mode - Mesh-core always writes FP32 coord to mesh mem..
                                           // irrespective of incoming mesh datatype type
    } else {
        rot_pars.MBufW_req = Si_pxls << irt_desc.image_par[MIMAGE].Ps;
    }
    IRT_TRACE_UTILS("Mesh mem fit check: Sm %u Si_log2 %u, Si_pxls %u, Ps %u, MBufW_req %u\n",
                    irt_desc.image_par[MIMAGE].S,
                    Si_log2,
                    Si_pxls,
                    irt_desc.image_par[MIMAGE].Ps,
                    rot_pars.MBufW_req);

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

    uint8_t buf_mode = (uint8_t)(log2((double)rot_pars.MBufW_req /
                                      ((irt_desc.irt_mode == e_irt_rescale) ? RESCALE_MIN_SM_WIDTH : MIN_SM_WIDTH)));
    if (buf_mode > 7)
        buf_mode = 7; // Max buffer mode for mesh mem is 7
    if (irt_cfg.buf_select[e_irt_block_mesh] == e_irt_buf_select_auto) { // auto
        irt_desc.image_par[MIMAGE].buf_mode = buf_mode;

        if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_static)
            irt_desc.image_par[MIMAGE].buf_mode = (uint8_t)std::min((int32_t)irt_desc.image_par[MIMAGE].buf_mode,
                                                                    IRT_MM_MAX_STAT_MODE); // mode 7 is not supported
        else
            irt_desc.image_par[MIMAGE].buf_mode = (uint8_t)std::min(
                (int32_t)irt_desc.image_par[MIMAGE].buf_mode, IRT_MM_MAX_DYN_MODE); // mode 6 and 7 is not supported

    } else { // manual
        irt_desc.image_par[MIMAGE].buf_mode = irt_cfg.buf_mode[e_irt_block_mesh];
    }
    irt_cfg.buf_mode[e_irt_block_mesh] = irt_desc.image_par[MIMAGE].buf_mode;

    // IBufH_req is required lines to be stored in memory
    uint16_t mm_min_H = (irt_desc.irt_mode == e_irt_rescale) ? irt_desc.rescale_LUT_y->num_taps : 2;
    // rot_pars.MBufH_req = 2 + (irt_desc.image_par[MIMAGE].H > 2); //need 1 more line for interpolation for sparse
    // matrix rot_pars.MBufH_req = mm_min_H + (irt_desc.image_par[MIMAGE].H > mm_min_H); //need 1 more line for
    // interpolation for sparse matrix
    //  IRT_TRACE_UTILS("check\n");
    if (irt_desc.irt_mode == e_irt_rescale) {
        // TODO - Design optimization pending to look for less lines, update scaling factor with 1 instead of 3
        rot_pars.MBufH_req = mm_min_H + ceil(irt_desc.rescale_LUT_y->Gf * 3) + 1;
        if (rot_pars.filter_type == e_irt_bicubic)
            rot_pars.MBufH_req =
                2 * rot_pars.MBufH_req; // Allowing double buffering for bicubic as no of TAPs are always 4.
    } else {
        rot_pars.MBufH_req = mm_min_H + (irt_desc.image_par[MIMAGE].H > mm_min_H);
    }
    if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic && (irt_desc.image_par[MIMAGE].H > 2))
        rot_pars.MBufH_req += IRT_MM_DYN_MODE_LINE_RELEASE; // incase of dynamic mode we discard 2 lines at a time

    // BufH_act - number of input lines that can be stored in rotation memory if we store BufW_req pixels per line
    uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufW;
    uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufH;
    if (irt_desc.irt_mode == e_irt_rescale)
        BufW_act = BufW_act >> 1; // Need ot reduce buf width by half for rescale

    IRT_TRACE_UTILS("Desc %u: Mesh memory fit check: mesh stripe of %u pixels requires %ux%u mesh buffer\n",
                    desc,
                    irt_desc.image_par[MIMAGE].S,
                    rot_pars.MBufW_req,
                    rot_pars.MBufH_req);

    if (hl_only == 0) {
        if (rot_pars.MBufH_req > BufH_act || rot_pars.MBufW_req > BufW_act) {
            if (rot_pars.oimg_auto_adj == 0) {
                IRT_TRACE_UTILS("Desc %d: Mesh memory fit check: mesh stripe is not supported: required mesh stripe "
                                "width %d pixels (%d bytes) and mesh memory size %ux%u for buf_mode %d exceeds %ux%u\n",
                                desc,
                                irt_desc.image_par[MIMAGE].S,
                                irt_desc.image_par[MIMAGE].S << irt_desc.image_par[MIMAGE].Ps,
                                rot_pars.MBufW_req,
                                rot_pars.MBufH_req,
                                irt_desc.image_par[MIMAGE].buf_mode,
                                BufW_act,
                                BufH_act);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       "was not run, mesh stripe is not supported: required mesh stripe width %u "
                                       "pixels (%u bytes) and mesh memory size %ux%u for buf_mode %u exceeds %ux%u\n",
                                       irt_desc.image_par[MIMAGE].S,
                                       irt_desc.image_par[MIMAGE].S << irt_desc.image_par[MIMAGE].Ps,
                                       rot_pars.MBufW_req,
                                       rot_pars.MBufH_req,
                                       irt_desc.image_par[MIMAGE].buf_mode,
                                       BufW_act,
                                       BufH_act);
                IRT_CLOSE_FAILED_TEST(0);
            } else {
                IRT_TRACE_UTILS("Desc %d: H req %d actual %d W req %d and actual %d. Mesh buf %d\n",
                                desc,
                                rot_pars.MBufH_req,
                                BufH_act,
                                rot_pars.MBufW_req,
                                BufW_act,
                                irt_cfg.buf_select[e_irt_block_mesh]);
                if (rot_pars.MBufH_req > BufH_act) {
                    if (irt_cfg.buf_select[e_irt_block_mesh] == e_irt_buf_select_auto) {
                        while (rot_pars.MBufH_req > BufH_act) {
                            IRT_TRACE_UTILS("Desc %u: Reducing Mesh buf mode by 1 from %d\n",
                                            desc,
                                            irt_desc.image_par[MIMAGE].buf_mode);
                            irt_desc.image_par[MIMAGE].buf_mode--;
                            BufW_act = BufW_act >> 1;
                            BufH_act = BufH_act << 1;
                            IRT_TRACE_UTILS("Desc %u: New Mesh buf mode %u, BufW_act %u and BufH_act %u\n",
                                            desc,
                                            irt_desc.image_par[MIMAGE].buf_mode,
                                            BufW_act,
                                            BufH_act);
                        }
                    } else {
                        IRT_TRACE_UTILS(
                            "Desc %u: Mesh memory fit check: mesh stripe is not supported: required mesh stripe height "
                            "%u lines and mesh memory size %ux%u for buf_mode %u exceeds %ux%u\n",
                            desc,
                            rot_pars.MBufH_req,
                            rot_pars.MBufW_req,
                            rot_pars.MBufH_req,
                            irt_desc.image_par[MIMAGE].buf_mode,
                            BufW_act,
                            BufH_act);
                        IRT_TRACE_TO_RES_UTILS(test_res,
                                               "was not run, mesh stripe is not supported: required mesh stripe height "
                                               "%u lines and mesh memory size %ux%u for buf_mode %u exceeds %ux%u\n",
                                               rot_pars.MBufH_req,
                                               rot_pars.MBufW_req,
                                               rot_pars.MBufH_req,
                                               irt_desc.image_par[MIMAGE].buf_mode,
                                               BufW_act,
                                               BufH_act);
                        IRT_CLOSE_FAILED_TEST(0);
                    }
                }
                if (rot_pars.MBufW_req > BufW_act) {
                    irt_desc.image_par[MIMAGE].S =
                        BufW_act >>
                        (irt_desc.image_par[MIMAGE].Ps +
                         (((irt_desc.irt_mode == e_irt_rescale) & (irt_desc.image_par[MIMAGE].Ps == 0)) ? 1 : 0));
                    IRT_TRACE_UTILS("Desc %u: Mesh stripe %d and mesh_Sh %f\n",
                                    desc,
                                    irt_desc.image_par[MIMAGE].S,
                                    rot_pars.mesh_Sh);
                    if (irt_desc.irt_mode == e_irt_rescale) { // rescale
                        if (rot_pars.mesh_Sh == 1) { // non-sparse
                            irt_desc.image_par[OIMAGE].S =
                                std::min(irt_desc.image_par[OIMAGE].S,
                                         (uint16_t)(irt_desc.image_par[MIMAGE].S - (irt_desc.rescale_LUT_x->num_taps)));
                            IRT_TRACE_UTILS("3. So %d\n", irt_desc.image_par[OIMAGE].S);
                        } else {
                            irt_desc.image_par[OIMAGE].S =
                                std::min(irt_desc.image_par[OIMAGE].S,
                                         (uint16_t)((uint16_t)floor(((double)irt_desc.image_par[MIMAGE].S -
                                                                     irt_desc.rescale_LUT_x->num_taps) *
                                                                    rot_pars.mesh_Sh) +
                                                    1));
                            // Xr = floor((So-1)*Gh)+taps/2
                            // Xl = -(taps/2-1)
                            // Si = (xr -xl)+1
                            //    = floor((So-1)*Gh) + taps
                            // So = floor((Si -(taps))*Sh) +1
                            IRT_TRACE_UTILS("4. So %d\n", irt_desc.image_par[OIMAGE].S);
                        }
                    } else { // mesh mode
                        if (rot_pars.mesh_Sh == 1) { // non-sparse
                            irt_desc.image_par[OIMAGE].S =
                                std::min(irt_desc.image_par[OIMAGE].S, irt_desc.image_par[MIMAGE].S);
                            IRT_TRACE_UTILS("3. So %d\n", irt_desc.image_par[OIMAGE].S);
                        } else {
                            irt_desc.image_par[OIMAGE].S = std::min(
                                irt_desc.image_par[OIMAGE].S,
                                (uint16_t)floor(((double)irt_desc.image_par[MIMAGE].S - 1) * rot_pars.mesh_Sh));
                            IRT_TRACE_UTILS("4. So %d\n", irt_desc.image_par[OIMAGE].S);
                        }
                    }

                    irt_oimage_corners_calc(irt_cfg, rot_pars, irt_desc, desc);
                    if (irt_desc.irt_mode != e_irt_rescale) {
                        irt_map_iimage_stripe_adj<mesh_xy_fp64_meta>(
                            irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
                    }
                    IRT_TRACE_UTILS("Desc %u: done with mesh_memory_fit_check\n", desc);
                }
            }
        }
    }

    if (desc == 0) {
        IRT_TRACE_UTILS("Task %u: Required mesh memory size for Sm = %u is (%ux%u), mode %d (%ux%u) is used\n",
                        desc,
                        irt_desc.image_par[MIMAGE].S,
                        rot_pars.MBufW_req,
                        rot_pars.MBufH_req,
                        irt_desc.image_par[MIMAGE].buf_mode,
                        irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufW,
                        irt_cfg.rm_cfg[e_irt_block_mesh][irt_desc.image_par[MIMAGE].buf_mode].BufH);
    }
}

void irt_oimage_res_adj_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    uint16_t Si_log2;
    uint32_t Si_pxls;
    int32_t  Si_adj, So_from_IBufW = 0, So_from_IBufH = 0, Si;

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

    uint32_t BufW_act  = (uint32_t)irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
    uint32_t BufH_act  = (uint32_t)irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;
    bool     IBufW_fit = rot_pars.IBufW_req <= BufW_act;
    bool     IBufH_fit = rot_pars.IBufH_req <= BufH_act;
    bool     fit_width = 0, fit_height = 0;
    // uint16_t Ho = irt_desc.image_par[OIMAGE].H;
    // uint16_t Wo = irt_desc.image_par[OIMAGE].W;
    // uint16_t So = irt_desc.image_par[OIMAGE].S;
    // uint16_t Hm = irt_desc.image_par[MIMAGE].H;
    // uint16_t Wo = irt_desc.image_par[OIMAGE].W;
    // uint16_t Sm = irt_desc.image_par[MIMAGE].S;

    // IRT_TRACE_UTILS("Desc %d: Rotation memory fit check adjusts output stripe resolution to fit rotation memory
    // buffer mode %d for IRT %s mode\n", desc, irt_desc.image_par[IIMAGE].buf_mode, irt_irt_mode_s[irt_desc.irt_mode]);

    Si_pxls = BufW_act >> irt_desc.image_par[IIMAGE].Ps;
    Si_log2 = (uint16_t)floor(log2((double)Si_pxls));
    Si_adj  = (int32_t)pow(2.0, Si_log2);
    if (irt_desc.rot90 == 0) {
        Si_adj -= irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
        Si_adj -= (int32_t)ceil(2 * rot_pars.Si_delta) +
                  2; //(uint16_t)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
    }
    Si = Si_adj - 1 + irt_desc.rot90 - (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);

    bool affine_rotation            = strcmp(rot_pars.affine_mode, "R") == 0;
    bool projection_direct_rotation = rot_pars.proj_mode == e_irt_rotation;
    bool projection_affine_rotation = rot_pars.proj_mode == e_irt_affine && affine_rotation;
    bool projection_rotation        = projection_direct_rotation | projection_affine_rotation;
    bool mesh_direct_rotation       = rot_pars.mesh_mode == e_irt_rotation;
    bool mesh_affine_rotation       = rot_pars.mesh_mode == e_irt_affine && affine_rotation;
    bool mesh_projection_rotation   = rot_pars.mesh_mode == e_irt_projection && projection_rotation;
    bool mesh_rotation              = mesh_direct_rotation | mesh_affine_rotation | mesh_projection_rotation;

    bool projection_affine      = rot_pars.proj_mode == e_irt_affine;
    bool mesh_direct_affine     = rot_pars.mesh_mode == e_irt_affine;
    bool mesh_projection_affine = rot_pars.mesh_mode == e_irt_projection && projection_affine;
    bool mesh_affine            = mesh_direct_affine | mesh_projection_affine;

    // IRT_TRACE_UTILS("affine_rotation			= %d\n", affine_rotation);
    // IRT_TRACE_UTILS("projection_direct_rotation = %d\n", projection_direct_rotation);
    // IRT_TRACE_UTILS("projection_affine_rotation = %d\n", projection_affine_rotation);
    // IRT_TRACE_UTILS("projection_rotation		= %d\n", projection_rotation);
    // IRT_TRACE_UTILS("mesh_direct_rotation		= %d\n", mesh_direct_rotation);
    // IRT_TRACE_UTILS("mesh_affine_rotation		= %d\n", mesh_affine_rotation);
    // IRT_TRACE_UTILS("mesh_projection_rotation	= %d\n", mesh_projection_rotation);
    // IRT_TRACE_UTILS("mesh_rotation				= %d\n", mesh_rotation);
    // IRT_TRACE_UTILS("projection_affine			= %d\n", projection_affine);
    // IRT_TRACE_UTILS("mesh_direct_affine			= %d\n", mesh_direct_affine);
    // IRT_TRACE_UTILS("mesh_projection_affine		= %d\n", mesh_projection_affine);
    // IRT_TRACE_UTILS("mesh_affine				= %d\n", mesh_affine);

    bool irt_do_rotate = irt_desc.irt_mode == e_irt_rotation ||
                         (irt_desc.irt_mode == e_irt_affine && affine_rotation) ||
                         (irt_desc.irt_mode == e_irt_projection && projection_rotation) ||
                         (irt_desc.irt_mode == e_irt_mesh && mesh_rotation && rot_pars.mesh_dist_r0_Sh1_Sv1 &&
                          rot_pars.mesh_matrix_error == 0);

    bool irt_do_affine = irt_desc.irt_mode == e_irt_affine ||
                         (irt_desc.irt_mode == e_irt_projection && projection_affine) ||
                         (irt_desc.irt_mode == e_irt_mesh && mesh_affine && rot_pars.mesh_dist_r0_Sh1_Sv1 &&
                          rot_pars.mesh_matrix_error == 0);

    if (irt_do_rotate) {
        if (irt_desc.rot90) {
            irt_desc.image_par[OIMAGE].H = (uint16_t)std::min(Si_pxls, (uint32_t)irt_desc.image_par[OIMAGE].H);
            So_from_IBufW                = (int32_t)irt_desc.image_par[OIMAGE].S;
            So_from_IBufH                = BufH_act - (/*irt_desc.rot90_inth |*/ irt_desc.rot90_intv);
        } else {
            if (rot_pars.use_rectangular_input_stripe == 0)
                So_from_IBufW =
                    (int32_t)floor(((double)Si - 1) * cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0));
            else
                So_from_IBufW = (int32_t)floor(((double)Si + ceil(2 * rot_pars.Si_delta) - 1 -
                                                fabs(sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)) *
                                                    (irt_desc.image_par[OIMAGE].H - 1)) /
                                                   cos(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0) -
                                               1);
            So_from_IBufH = (int32_t)floor(
                ((double)BufH_act - 2 - (irt_cfg.buf_format[e_irt_block_rot] ? IRT_RM_DYN_MODE_LINE_RELEASE : 1)) /
                sin(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
        }
        if (So_from_IBufW > 0 && So_from_IBufH > 0) {
            fit_width  = 1;
            fit_height = 1;
            IBufW_fit  = 1;
            IBufH_fit  = 1;
        }
    } else if (irt_do_affine) {
        if (irt_desc.rot90) {
            // irt_desc.image_par[OIMAGE].H = irt_desc.image_par[IIMAGE].S;
            irt_desc.image_par[OIMAGE].H = (uint16_t)std::min(Si_pxls, (uint32_t)irt_desc.image_par[OIMAGE].H);
            So_from_IBufW                = (int32_t)irt_desc.image_par[OIMAGE].S;
            So_from_IBufH                = BufH_act - (/*irt_desc.rot90_inth |*/ irt_desc.rot90_intv);
        } else {
            if (rot_pars.use_rectangular_input_stripe == 0)
                So_from_IBufW = (int32_t)floor(((double)Si - 1) / rot_pars.affine_Si_factor);
            else
                So_from_IBufW = (int32_t)floor(((double)Si + ceil(2 * rot_pars.Si_delta) - 1 -
                                                fabs(rot_pars.M12d) * (irt_desc.image_par[OIMAGE].H - 1)) /
                                                   fabs(rot_pars.M11d) -
                                               1); // ROTSIM_DV_INTGR -- local fix for fabs for float data types
            if (rot_pars.M21d != 0) {
                So_from_IBufH =
                    (int32_t)floor(((double)BufH_act - 2 - rot_pars.IBufH_delta -
                                    (irt_cfg.buf_format[e_irt_block_rot] ? IRT_RM_DYN_MODE_LINE_RELEASE : 1)) /
                                   fabs(rot_pars.M21d));
            } else {
                So_from_IBufH = So_from_IBufW;
            }
        }
        if (So_from_IBufW > 0 && So_from_IBufH > 0) {
            fit_width  = 1;
            fit_height = 1;
            IBufW_fit  = 1;
            IBufH_fit  = 1;
        }
    } else if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) ||
               (irt_desc.irt_mode == e_irt_mesh && rot_pars.mesh_mode == e_irt_projection) ||
               (irt_desc.irt_mode == e_irt_mesh /*&& rot_pars.mesh_mode == e_irt_mesh*/)) {

        // next fit approach is taken:
        // fit width -> fit height
        // else fit height -> fit width
        // else fir both

        // fit width -> fit height
        // trying to reduce output image width to fit rotation memory

        IRT_TRACE_UTILS("Trying to reduce output image width to fit rotation memory, it takes time, be patient...\n");
        if (irt_desc.irt_mode == e_irt_mesh)
            irt_map_oimage_res_adj_calc<mesh_xy_fp64_meta>(
                irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
        else
            irt_map_oimage_res_adj_calc<mesh_xy_fp64_meta>(
                irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.proj_image_full);
        if (irt_desc.image_par[OIMAGE].S > 0 && irt_desc.image_par[OIMAGE].H > 0) {
            fit_width  = 1;
            fit_height = 1;
            IBufW_fit  = 1;
            IBufH_fit  = 1;
        }

        So_from_IBufW = (uint32_t)irt_desc.image_par[OIMAGE].S;
        So_from_IBufH = (uint32_t)irt_desc.image_par[OIMAGE].S;
    } else {
        IRT_TRACE_UTILS("Desc %d: Output image resolution auto adjust does not supported in this irt mode\n", desc);
        IRT_TRACE_TO_RES_UTILS(
            test_res,
            "was not run, desc %d Output image resolution auto adjust does not supported in this irt mode\n",
            desc);
        IRT_CLOSE_FAILED_TEST(0);
    }

    irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufW);
    irt_desc.image_par[OIMAGE].S = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].S, So_from_IBufH);
    IRT_TRACE_UTILS("5. So %d\n", irt_desc.image_par[OIMAGE].S);
    if (irt_desc.irt_mode == e_irt_rotation || // H6-transforms need output.W & S to be same
        irt_desc.irt_mode == e_irt_affine || irt_desc.irt_mode == e_irt_projection || irt_desc.irt_mode == e_irt_mesh) {
        irt_desc.image_par[OIMAGE].W = irt_desc.image_par[OIMAGE].S;
    }
    if (irt_desc.irt_mode == e_irt_mesh) {
        irt_desc.image_par[MIMAGE].S =
            (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].S - 1) / rot_pars.mesh_Sh) + 1;
        irt_desc.image_par[MIMAGE].H =
            (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].H - 1) / rot_pars.mesh_Sv) + 1;
        if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) { // stripe height is multiple of 2
            irt_desc.image_par[MIMAGE].H =
                (uint16_t)(ceil((double)irt_desc.image_par[MIMAGE].H / IRT_MM_DYN_MODE_LINE_RELEASE) *
                           IRT_MM_DYN_MODE_LINE_RELEASE);
        }
    }

    if (fit_width && fit_height && IBufW_fit && IBufH_fit && irt_desc.image_par[OIMAGE].S > 0 &&
        irt_desc.image_par[OIMAGE].H > 0) {
#if 0
		IRT_TRACE_UTILS("Output image resolution auto adjust: for buffer mode %u with %ux%u buffer size, So from BufW is %d, So from BufH is %d, selected So is %u\n",
			irt_desc.image_par[IIMAGE].buf_mode, rot_pars.IBufW_req, rot_pars.IBufH_req, So_from_IBufW, So_from_IBufH, irt_desc.image_par[OIMAGE].S);
		IRT_TRACE_UTILS("Output image resolution auto adjust: for buffer mode %u with %ux%u buffer size, output image size is reduced from %ux%u to %ux%u in %s mode\n",
			irt_desc.image_par[IIMAGE].buf_mode, rot_pars.IBufW_req, rot_pars.IBufH_req, So, Ho, irt_desc.image_par[OIMAGE].S, irt_desc.image_par[OIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
		IRT_TRACE_UTILS("Output image resolution auto adjust: input image stripe is %ux%u\n", irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].Hs);
		IRT_TRACE_UTILS("Output image resolution auto adjust: mesh image size is reduced from %ux%u to %ux%u in %s mode\n",
			Sm, Hm, irt_desc.image_par[MIMAGE].S, irt_desc.image_par[MIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
#endif
    } else {
        IRT_TRACE_UTILS("Desc %u: Output image resolution auto adjust cannot find output stripe resolution for defined "
                        "%s parameters for buffer mode %u\n",
                        desc,
                        irt_irt_mode_s[irt_desc.irt_mode],
                        irt_desc.image_par[IIMAGE].buf_mode);
        IRT_TRACE_TO_RES_UTILS(test_res,
                               "was not run, desc %u Output image resolution auto adjust cannot find output stripe "
                               "resolution for defined %s parameters for buffer mode %u\n",
                               desc,
                               irt_irt_mode_s[irt_desc.irt_mode],
                               irt_desc.image_par[IIMAGE].buf_mode);
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
void irt_oimage_corners_calc(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{
    rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
    if (irt_h5_mode) {
        if ((irt_desc.image_par[OIMAGE].S % 8) != 0) { // not multiple of 8, round down to multiple of 8
            irt_desc.image_par[OIMAGE].S -= 8;
            rot_pars.So8 = irt_desc.image_par[OIMAGE].S;
            rot_pars.So8 += (8 - (irt_desc.image_par[OIMAGE].S % 8));
            IRT_TRACE_UTILS("6. So %d\n", irt_desc.image_par[OIMAGE].S);
        }
    }
    rot_pars.Xo_first = -(double)irt_desc.image_par[OIMAGE].Xc / 2;
    rot_pars.Xo_last  = (double)rot_pars.So8 - 1 - (double)irt_desc.image_par[OIMAGE].Xc / 2;
    rot_pars.Yo_first = -(double)irt_desc.image_par[OIMAGE].Yc / 2;
    rot_pars.Yo_last  = (double)irt_desc.image_par[OIMAGE].H - 1 - (double)irt_desc.image_par[OIMAGE].Yc / 2;
}

uint32_t irt_IBufW_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t Si)
{

    uint16_t Si_log2;
    uint32_t Si_adj, Si_pxls;
    Si_adj = Si + 1 - irt_desc.rot90 +
             (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/); // for 90 degree rotation Si+1 is not needed
    if (irt_desc.rot90 == 0) {
        // Si_adj += (uint16_t)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0));
        Si_adj += irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip * irt_desc.read_hflip;
    }
    Si_log2 = (uint16_t)ceil(log2((double)Si_adj));
    Si_pxls = (uint32_t)pow(2.0, Si_log2);
    return Si_pxls << irt_desc.image_par[IIMAGE].Ps;
}

uint32_t irt_IBufH_req_calc(irt_cfg_pars& irt_cfg, irt_desc_par& irt_desc, uint32_t IBufH)
{

    uint32_t IBufH_req;

    if (irt_desc.rot90 == 0) {
        IBufH_req = IBufH + 1 + 1 + 1; // 1 because of interpolation, 1 because of Y is released based on previous line
        if (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic)
            IBufH_req += IRT_RM_DYN_MODE_LINE_RELEASE; // incase of dynamic mode we discard 8 lines at a time
    } else {
        IBufH_req = IBufH + (irt_desc.rot90_intv /*| irt_desc.rot90_inth*/);
    }

    return IBufH_req;
}

void irt_rotation_memory_fit_check(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    // IBufW_req is required bytes per input lines to be stored in memory, equal to Si round up to power of 2 and
    // multiplied by pixel size: BufW (byte)						32 64 128 256 512 1024 2048 4096 BufW (pixels) for 1
    // byte/pixel -	32 64 128 256 512 1024 2048 4096 BufW (pixels) for 2 byte/pixel -	16 32  64 128 256  512 1024 2048

    rot_pars.IBufW_req = irt_IBufW_req_calc(irt_cfg, irt_desc, irt_desc.image_par[IIMAGE].S);

    uint8_t buf_mode = (uint8_t)(log2((double)rot_pars.IBufW_req / MIN_SI_WIDTH));
    // BufH_available - number of input lines that can be stored in rotation memory if we store IBufW_req pixels per
    // line

    if (irt_cfg.buf_select[0] == 1 /*auto*/ && irt_h5_mode == 0) { // && irt_desc.rot90 == 0) {
        irt_desc.image_par[IIMAGE].buf_mode = buf_mode;
    } else {
        irt_desc.image_par[IIMAGE].buf_mode = irt_cfg.buf_mode[0];
    }
    irt_cfg.buf_mode[0] = irt_desc.image_par[IIMAGE].buf_mode;

    if (irt_cfg.buf_format[0] == e_irt_buf_format_dynamic)
        irt_desc.image_par[IIMAGE].buf_mode =
            (uint8_t)std::min((int32_t)irt_desc.image_par[IIMAGE].buf_mode,
                              IRT_RM_MAX_DYN_MODE); // buffer mode 7 is not supported in e_irt_buf_format_dynamic
    else
        irt_desc.image_par[IIMAGE].buf_mode =
            (uint8_t)std::min((int32_t)irt_desc.image_par[IIMAGE].buf_mode, IRT_RM_MAX_STAT_MODE);

    uint32_t BufW_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW;
    uint32_t BufH_act = irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH;

    // IRT_TRACE_UTILS("Desc %u: Rotation memory fit check: output stripe of %d pixels requires input stripe width %d
    // pixels (%d bytes) and %ux%u rotation buffer, input memory size for buf_mode %d is %ux%u\n", 	desc,
    // irt_desc.image_par[OIMAGE].S, irt_desc.image_par[IIMAGE].S, irt_desc.image_par[IIMAGE].S <<
    // irt_desc.image_par[IIMAGE].Ps, rot_pars.IBufW_req, rot_pars.IBufH_req, 	irt_desc.image_par[IIMAGE].buf_mode,
    // BufW_act, BufH_act);

    if (hl_only == 0) {
        if ((rot_pars.IBufH_req > BufH_act || rot_pars.IBufW_req > BufW_act) &&
            /*irt_desc.rot90 == 0 &&*/ irt_h5_mode == 0) {
            if (rot_pars.oimg_auto_adj == 0) {
                // IRT_TRACE_UTILS("Desc %d: IRT rotation memory fit check: Rotation angle and output stripe are not
                // supported: required rotation memory size %dx%d exceeds %dx%d of buf_mode %d\n", desc,
                // rot_pars.IBufW_req, rot_pars.IBufH_req, BufW_act, BufH_act, irt_desc.image_par[IIMAGE].buf_mode);
                // IRT_TRACE_TO_RES_UTILS(test_res, "was not run, desc %d rotation angle and output stripe are not
                // supported: required rotation memory size %dx%d exceeds %dx%d of buf_mode %d\n", desc,
                // rot_pars.IBufW_req, rot_pars.IBufH_req, BufW_act, BufH_act, irt_desc.image_par[IIMAGE].buf_mode);
                // IRT_CLOSE_FAILED_TEST(0);

                IRT_TRACE_UTILS(
                    "Desc %d: Rotation memory fit check: output stripe is not supported: required input stripe width "
                    "%d pixels (%d bytes) and input memory size %ux%u for buf_mode %d exceeds %ux%u\n",
                    desc,
                    irt_desc.image_par[IIMAGE].S,
                    irt_desc.image_par[IIMAGE].S << irt_desc.image_par[IIMAGE].Ps,
                    rot_pars.IBufW_req,
                    rot_pars.IBufH_req,
                    irt_desc.image_par[IIMAGE].buf_mode,
                    BufW_act,
                    BufH_act);
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       "was not run, output stripe is not supported: required input stripe width %d "
                                       "pixels (%d bytes) and memory size %ux%u for buf_mode %d exceeds %ux%u\n",
                                       irt_desc.image_par[IIMAGE].S,
                                       irt_desc.image_par[IIMAGE].S << irt_desc.image_par[IIMAGE].Ps,
                                       rot_pars.IBufW_req,
                                       rot_pars.IBufH_req,
                                       irt_desc.image_par[IIMAGE].buf_mode,
                                       BufW_act,
                                       BufH_act);
                IRT_CLOSE_FAILED_TEST(0);

            } else {
                // rot_pars.IBufW_req = (uint32_t)BufW_act;
                // rot_pars.IBufH_req = (uint32_t)BufH_act;
                irt_oimage_res_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
            }
        } else { // in rotation/affine/projection with rotation/affine and mesh with rotation/affine w/o distortion and
                 // w/o sparsing we can rely on IBufW_req and IBufH_req linear calculation and not use map
            if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) ||
                (irt_desc.irt_mode == e_irt_mesh &&
                 (rot_pars.mesh_mode == e_irt_projection || rot_pars.mesh_mode == e_irt_mesh ||
                  rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)))
                irt_oimage_res_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
        }
    }

    if (desc == 0) {
        IRT_TRACE_UTILS(
            "Task %d: Required rotation memory size for Si = %d and angle %f is (%ux%u), mode %d (%dx%d) is used\n",
            desc,
            irt_desc.image_par[IIMAGE].S,
            rot_pars.irt_angles_adj[e_irt_angle_rot],
            rot_pars.IBufW_req,
            rot_pars.IBufH_req,
            irt_desc.image_par[IIMAGE].buf_mode,
            irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufW,
            irt_cfg.rm_cfg[e_irt_block_rot][irt_desc.image_par[IIMAGE].buf_mode].BufH);
    }
}

void irt_rotation_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    double cosD, sinD;

    if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
        cosD = (double)irt_desc.cosf;
        sinD = (double)irt_desc.sinf;
    } else {
        cosD = (double)irt_desc.cosi / pow(2.0, rot_pars.TOTAL_PREC);
        sinD = (double)irt_desc.sini / pow(2.0, rot_pars.TOTAL_PREC);
    }

    rot_pars.Si_delta = tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0);
    // if ((rot_pars.Si_delta - floor(rot_pars.Si_delta)) >= 0.75) //ceiling if >= x.5
    //	rot_pars.Si_delta = ceil(rot_pars.Si_delta);
    if (rot_pars.Si_delta > 50.0)
        rot_pars.Si_delta += 3;
    else if (rot_pars.use_Si_delta_margin && rot_pars.Si_delta != 0)
        rot_pars.Si_delta += 1;
    else if (rot_pars.Si_delta > 6.0)
        rot_pars.Si_delta += 0.73;
    else if (rot_pars.Si_delta > 4.0)
        rot_pars.Si_delta += 0.68;
    else if (rot_pars.Si_delta > 3.0)
        rot_pars.Si_delta += 0.64;
    else if (rot_pars.Si_delta > 2.0)
        rot_pars.Si_delta += 0.56;
    else if (rot_pars.Si_delta > 1.0)
        rot_pars.Si_delta += 0.55;
    else if (rot_pars.Si_delta != 0)
        rot_pars.Si_delta += 0.40; // 0.26;

    if (irt_desc.rot90) {
        irt_desc.image_par[IIMAGE].S = irt_desc.image_par[OIMAGE].H + (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);
    } else {
        irt_desc.image_par[IIMAGE].S =
            (uint16_t)ceil((double)rot_pars.So8 / cos(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0) +
                           2.0 * rot_pars.Si_delta) +
            2;
    }

    rot_pars.IBufH_req =
        (uint32_t)fabs(ceil(rot_pars.So8 * sin(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) * M_PI / 180.0)));

    if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle

        rot_pars.Xi_first = ((double)irt_desc.cosi * rot_pars.Xo_last + (double)irt_desc.sini * rot_pars.Yo_first) /
                                pow(2.0, rot_pars.TOTAL_PREC) +
                            (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first = (-(double)irt_desc.sini * rot_pars.Xo_last + (double)irt_desc.cosi * rot_pars.Yo_first) /
                                pow(2.0, rot_pars.TOTAL_PREC) +
                            (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Yi_last = (-(double)irt_desc.sini * rot_pars.Xo_first + (double)irt_desc.cosi * rot_pars.Yo_last) /
                               pow(2.0, rot_pars.TOTAL_PREC) +
                           (double)irt_desc.image_par[IIMAGE].Yc / 2;

        rot_pars.Xi_first =
            (cosD * rot_pars.Xo_last + sinD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first =
            (-sinD * rot_pars.Xo_last + cosD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Yi_last =
            (-sinD * rot_pars.Xo_first + cosD * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
#if 1
        rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first) - irt_desc.image_par[IIMAGE].S;

        if (irt_desc.rot90 == 0)
            rot_pars.Xi_first = rot_pars.Xi_first + rot_pars.Si_delta + 2;

        if (irt_desc.rot90 == 1) {
            rot_pars.Xi_start += irt_desc.image_par[IIMAGE].S;
            // rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
        }
#endif
    } else {

        rot_pars.Xi_first = ((double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_first) /
                                pow(2.0, rot_pars.TOTAL_PREC) +
                            (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first = (-(double)irt_desc.sini * rot_pars.Xo_first + (double)irt_desc.cosi * rot_pars.Yo_first) /
                                pow(2.0, rot_pars.TOTAL_PREC) +
                            (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Yi_last = (-(double)irt_desc.sini * rot_pars.Xo_last + (double)irt_desc.cosi * rot_pars.Yo_last) /
                               pow(2.0, rot_pars.TOTAL_PREC) +
                           (double)irt_desc.image_par[IIMAGE].Yc / 2;

        rot_pars.Xi_first =
            (cosD * rot_pars.Xo_first + sinD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first =
            (-sinD * rot_pars.Xo_first + cosD * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Yi_last =
            (-sinD * rot_pars.Xo_last + cosD * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;

#if 1
        rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first - rot_pars.Si_delta) - irt_desc.read_hflip - 1;

        if (irt_desc.rot90 == 0)
            rot_pars.Xi_first = rot_pars.Xi_first - rot_pars.Si_delta - 1;

        if (irt_desc.rot90 == 1) {
            rot_pars.Xi_start -= irt_desc.image_par[IIMAGE].S;
        }
#endif
    }
    rot_pars.im_read_slope = tan((double)rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0);
    if (irt_desc.crd_mode == e_irt_crd_mode_fp32) {
        rot_pars.IBufH_req++;
        rot_pars.Yi_first--;
    }
#if 1
    double Xi_TL = ((double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_first) /
                       pow(2.0, rot_pars.TOTAL_PREC) +
                   (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Xi_TR = ((double)irt_desc.cosi * rot_pars.Xo_last + (double)irt_desc.sini * rot_pars.Yo_first) /
                       pow(2.0, rot_pars.TOTAL_PREC) +
                   (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Xi_BL = ((double)irt_desc.cosi * rot_pars.Xo_first + (double)irt_desc.sini * rot_pars.Yo_last) /
                       pow(2.0, rot_pars.TOTAL_PREC) +
                   (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Xi_BR = ((double)irt_desc.cosi * rot_pars.Xo_last + (double)irt_desc.sini * rot_pars.Yo_last) /
                       pow(2.0, rot_pars.TOTAL_PREC) +
                   (double)irt_desc.image_par[IIMAGE].Xc / 2;
    // cos is always > 0 for [-90:90] rotation range, Xo_last > Xo_first => Xi_TR > Xi_TL
    // for same reason Xi_BR > Xi_BL
    // Xi_TL < Xi_TR ; Xi_BL < Xi_BR
    // Yo_first < Yo_last
    // if rot_dir positive and sin > 0 => Xi_TL < Xi_BL and Xi_TR < Xi_BR

    /*
    because Xoc and Yoc do not change Xi_XX relation, we can indentify relations as:
    Xi_TL = cosi * 0		+ sini * 0		= 0
    Xi_TR = cosi * (So-1)	+ sini * 0		= cosi * (So-1)
    Xi_BL = cosi * 0		+ sini * (Ho-1)	= sini * (Ho-1)
    Xi_BR = cosi * (So-1)	+ sini * (Ho-1)	= cosi * (So-1)	+ sini * (Ho-1)

    if rot_dir > 0, then sin and cos are > 0, then Xi_TL < Xi_TR, Xi_TL < Xi_BL, Xi_TR < Xi_BR, Xi_BL < Xi_BR and at the
    end Xi_TL is min value and Xi_BR is max value then rectangular input stripe Si = cosi * (So-1)	+ sini * (Ho-1)

    if rot_dir < 0, then sin < 0 and cos > 0, then Xi_TL < Xi_TR, Xi_TL > Xi_BL, Xi_TR > Xi_BR, Xi_BL < Xi_BR and at the
    end Xi_BL is min value and Xi_TR is max value then rectangular input stripe Si = cosi * (So-1) - sini * (Ho-1)

    And then input stripe Si = cosi * (So-1) + abs(sini * (Ho-1))
    Then So = (Si - abs(sini * (Ho-1))) / cosi + 1
    */
    double   XiL = fmin(fmin(fmin(Xi_TL, Xi_TR), Xi_BL), Xi_BR);
    double   XiR = fmax(fmax(fmax(Xi_TL, Xi_TR), Xi_BL), Xi_BR);
    uint16_t Si  = (uint16_t)(ceil(XiR) - floor(XiL) + 2.0 * rot_pars.Si_delta * 0) + 2;
    Si           = (uint16_t)ceil(((double)irt_desc.cosi * ((double)irt_desc.image_par[OIMAGE].S - 1) +
                         fabs((double)irt_desc.sini * ((double)irt_desc.image_par[OIMAGE].H - 1))) /
                        pow(2.0, rot_pars.TOTAL_PREC)) +
         3;
    IRT_TRACE_UTILS("Xi_L = %f, Xi_R = %f, Si = %d %d\n",
                    XiL,
                    XiR,
                    Si,
                    (uint16_t)ceil(((double)irt_desc.cosi * ((double)irt_desc.image_par[OIMAGE].S - 1) +
                                    fabs((double)irt_desc.sini * ((double)irt_desc.image_par[OIMAGE].H - 1))) /
                                   pow(2.0, rot_pars.TOTAL_PREC)));
    rot_pars.use_rectangular_input_stripe = 0;
    if (Si < irt_desc.image_par[IIMAGE].S && irt_desc.rot90 == 0) { // we can read less
        rot_pars.use_rectangular_input_stripe = 1;
        irt_desc.image_par[IIMAGE].S          = Si;
        rot_pars.im_read_slope                = 0;
        if (irt_desc.rot_dir == IRT_ROT_DIR_POS)
            rot_pars.Xi_first = XiR + 2;
        else
            rot_pars.Xi_first = XiL - 1;
    }
#endif
}

void irt_affine_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

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
    // rot_pars.Si_delta = ceil(fabs(M12 / M22)) + 1;
    rot_pars.Si_delta = fabs(M12 / M22);
    // IRT_TRACE_UTILS("Affine descriptor generator: Si_delta = %f\n", rot_pars.Si_delta);
    if ((rot_pars.Si_delta - floor(rot_pars.Si_delta)) >= 0.5) // ceiling if >= x.5
        rot_pars.Si_delta = ceil(rot_pars.Si_delta) + 1;
    else
        rot_pars.Si_delta = ceil(rot_pars.Si_delta);
    rot_pars.Si_delta = fabs(M12 / M22);
    // if (rot_pars.Si_delta != 0) rot_pars.Si_delta += 0.5;
    if (rot_pars.use_Si_delta_margin && rot_pars.Si_delta != 0)
        rot_pars.Si_delta += 1;
    else if (rot_pars.Si_delta > 1.0)
        rot_pars.Si_delta += 0.7;
    else if (rot_pars.Si_delta != 0)
        rot_pars.Si_delta += 0.65; // 0.61; //0.45

    double IBufH_delta = rot_pars.M22d - rot_pars.M21d;
    if (IBufH_delta <= 1) // delta <= 1 will not increase IBufH_req
        IBufH_delta = 0;
    else
        IBufH_delta = ceil(fabs(rot_pars.M22d - rot_pars.M21d)) - 1;

    rot_pars.IBufH_delta = (uint32_t)IBufH_delta;

    if (irt_desc.rot90) {
        irt_desc.image_par[IIMAGE].S = irt_desc.image_par[OIMAGE].H + (irt_desc.rot90_inth /*| irt_desc.rot90_intv*/);
    } else {
        irt_desc.image_par[IIMAGE].S =
            (uint16_t)ceil((double)rot_pars.So8 * rot_pars.affine_Si_factor + 2 * rot_pars.Si_delta) + 2;
        irt_desc.image_par[IIMAGE].S =
            (uint16_t)ceil(rot_pars.So8 * fabs((M11 * M22 - M12 * M21) / M22) + 2 * rot_pars.Si_delta) + 2;
    }

    rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.So8 * fabs(rot_pars.M21d))) + rot_pars.IBufH_delta;
    // rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.So8 * fabs(M21)));

    if (rot_pars.M21d <= 0) { // embedded clockwise rotation
        rot_pars.Xi_first =
            (M11 * rot_pars.Xo_last + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first =
            (M21 * rot_pars.Xo_last + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Xi_last =
            (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_last =
            (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;

    } else { // embedded counterclockwise rotation

        rot_pars.Xi_first =
            (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_first =
            (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
        rot_pars.Xi_last =
            (M11 * rot_pars.Xo_last + M12 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
        rot_pars.Yi_last =
            (M21 * rot_pars.Xo_last + M22 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
    }

    double Xi_TL = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Yi_TL = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
    double Xi_TR = (M11 * rot_pars.Xo_last + M12 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Yi_TR = (M21 * rot_pars.Xo_last + M22 * rot_pars.Yo_first) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
    double Xi_BL = (M11 * rot_pars.Xo_first + M12 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Yi_BL = (M21 * rot_pars.Xo_first + M22 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;
    double Xi_BR = (M11 * rot_pars.Xo_last + M12 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Xc / 2;
    double Yi_BR = (M21 * rot_pars.Xo_last + M22 * rot_pars.Yo_last) + (double)irt_desc.image_par[IIMAGE].Yc / 2;

    double Y_on_line;
    if (rot_pars.M21d > 0) { // top input corner is from output top-left corner of output, need to detect where
                             // top-right output corner is mapped
        // TL-to-BR equation is y = (Yi_TL - Yi_BR) / (Xi_TL - Xi_BR) * x + Yi_BR
        Y_on_line = (Yi_TL - Yi_BR) / (Xi_TL - Xi_BR) * (Xi_TR - Xi_BR) + Yi_BR;
        if ((Xi_TL < Xi_BR && Yi_TR <= Y_on_line) ||
            (Xi_TL > Xi_BR && Yi_TR >= Y_on_line)) // top-right output corner is above the line
            irt_desc.rot_dir = IRT_ROT_DIR_NEG;
        else // // top-right output corner is below the line
            irt_desc.rot_dir = IRT_ROT_DIR_POS;
    } else { // top input corner is from output top-right corner of output, need to detect where top-left output corner
             // is mapped
        // TR-to-BL equation is y = (Yi_TR - Yi_BL) / (Xi_TR - Xi_BL) * x + Yi_BL
        Y_on_line = (Yi_TR - Yi_BL) / (Xi_TR - Xi_BL) * (Xi_TL - Xi_BL) + Yi_BL;
        if ((Xi_TR < Xi_BL && Yi_TL < Y_on_line) ||
            (Xi_TR > Xi_BL && Yi_TL > Y_on_line)) // top-left output corner is above the line
            irt_desc.rot_dir = IRT_ROT_DIR_NEG;
        else // // top-left output corner is below the line
            irt_desc.rot_dir = IRT_ROT_DIR_POS;
    }

#if 0
	IRT_TRACE_UTILS("[Xi_TL = %f, Yi_TL = %f], [Xi_TR = %f, Yi_TR = %f]\n", Xi_TL, Yi_TL, Xi_TR, Yi_TR);
	IRT_TRACE_UTILS("[Xi_BL = %f, Yi_BL = %f], [Xi_BR = %f, Yi_BR = %f]\n", Xi_BL, Yi_BL, Xi_BR, Yi_BR);
	IRT_TRACE_UTILS("Y_on_line = %f\n", Y_on_line);
#endif

    rot_pars.im_read_slope = rot_pars.M12d / rot_pars.M22d;
    // rot_pars.im_read_slope = M12 / M22;

    if (irt_desc.rot_dir) {
        rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first) - irt_desc.image_par[IIMAGE].S;

        if (irt_desc.rot90 == 0)
            rot_pars.Xi_first = rot_pars.Xi_first + rot_pars.Si_delta + 2;
    } else {
        rot_pars.Xi_start = (int16_t)floor(rot_pars.Xi_first);

        if (irt_desc.rot90 == 0) {
            rot_pars.Xi_first = rot_pars.Xi_first - rot_pars.Si_delta - (rot_pars.im_read_slope == 0 ? 0 : 1);
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
        if M11 < 0 & M12 < 0 then Xi_BR is min value and Xi_TL is max value and Si = abs(M12) * (Ho-1) + (abs(M11)) *
       (So-1) Then rectangular input stripe Si = abs(M11) * (So-1) + abs(M12) * (Ho-1)

        Then So = (Si - abs(M12) * (Ho-1)) / abs(M11) + 1
    */

    uint16_t Si = (uint16_t)ceil(fabs(M11) * (irt_desc.image_par[OIMAGE].S - 1) +
                                 fabs(M12) * (irt_desc.image_par[OIMAGE].H - 1)) +
                  3;
    double XiL = fmin(fmin(fmin(Xi_TL, Xi_TR), Xi_BL), Xi_BR);
    double XiR = fmax(fmax(fmax(Xi_TL, Xi_TR), Xi_BL), Xi_BR);

    rot_pars.use_rectangular_input_stripe = 0;
    if (Si < irt_desc.image_par[IIMAGE].S && irt_desc.rot90 == 0) { // we can read less
        rot_pars.use_rectangular_input_stripe = 1;
        irt_desc.image_par[IIMAGE].S          = Si;
        rot_pars.im_read_slope                = 0;
        if (irt_desc.rot_dir == IRT_ROT_DIR_POS)
            rot_pars.Xi_first = XiR + 2;
        else
            rot_pars.Xi_first = XiL - 1;
    }
}

void irt_projection_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV) || defined(HABANA_SIMULATION)
    float             coord_out[IRT_ROT_MAX_PROC_SIZE], YTL, YTR, YBL, YBR;
    XiYi_float_struct iTL_corner, iTR_corner, iBL_corner, iBR_corner;
    XiYi_float_struct top_corner, bot_corner, left_corner, right_corner;
    // uint8_t top_corner_src, bot_corner_src, left_corner_src, right_corner_src;

    // output image top left corner map
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Af,
                            irt_desc.prj_Cf,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iTL_corner.X = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iTL_corner.Y = coord_out[0];

    // output image top right corner map
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Af,
                            irt_desc.prj_Cf,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iTR_corner.X = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iTR_corner.Y = coord_out[0];

    // output image bottom left corner map
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Af,
                            irt_desc.prj_Cf,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iBL_corner.X = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iBL_corner.Y = coord_out[0];

    // output image bottom right corner map
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Af,
                            irt_desc.prj_Cf,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iBR_corner.X = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    iBR_corner.Y = coord_out[0];

    // IRT_TRACE_UTILS("Projection corners (%.2f, %.2f), (%.2f, %.2f)\n", (double)iTL_corner.X, (double)iTL_corner.Y,
    // (double)iTR_corner.X, (double)iTR_corner.Y); IRT_TRACE_UTILS("                   (%.2f, %.2f), (%.2f, %.2f)\n",
    // (double)iBL_corner.X, (double)iBL_corner.Y, (double)iBR_corner.X, (double)iBR_corner.Y);

    // input image top corner finding
    if (iTL_corner.Y < iTR_corner.Y) {
        top_corner.Y = iTL_corner.Y;
        top_corner.X = iTL_corner.X; // top_corner_src = 0;
    } else {
        top_corner.Y = iTR_corner.Y;
        top_corner.X = iTR_corner.X; // top_corner_src = 1;
    }
    if (iBL_corner.Y < top_corner.Y) {
        top_corner.Y = iBL_corner.Y;
        top_corner.X = iBL_corner.X; // top_corner_src = 2;
    }
    if (iBR_corner.Y < top_corner.Y) {
        top_corner.Y = iBR_corner.Y;
        top_corner.X = iBR_corner.X; // top_corner_src = 3;
    }

    // input image bottom corner finding
    if (iTL_corner.Y > iTR_corner.Y) {
        bot_corner.Y = iTL_corner.Y;
        bot_corner.X = iTL_corner.X; // bot_corner_src = 0;
    } else {
        bot_corner.Y = iTR_corner.Y;
        bot_corner.X = iTR_corner.X; // bot_corner_src = 1;
    }
    if (iBL_corner.Y > bot_corner.Y) {
        bot_corner.Y = iBL_corner.Y;
        bot_corner.X = iBL_corner.X; // bot_corner_src = 2;
    }
    if (iBR_corner.Y > bot_corner.Y) {
        bot_corner.Y = iBR_corner.Y;
        bot_corner.X = iBR_corner.X; // bot_corner_src = 3;
    }

    // input image left corner finding
    if (iTL_corner.X < iTR_corner.X) {
        left_corner.Y = iTL_corner.Y;
        left_corner.X = iTL_corner.X; // left_corner_src = 0;
    } else {
        left_corner.Y = iTR_corner.Y;
        left_corner.X = iTR_corner.X; // left_corner_src = 1;
    }
    if (iBL_corner.X < left_corner.X) {
        left_corner.Y = iBL_corner.Y;
        left_corner.X = iBL_corner.X; // left_corner_src = 2;
    }
    if (iBR_corner.X < left_corner.X) {
        left_corner.Y = iBR_corner.Y;
        left_corner.X = iBR_corner.X; // left_corner_src = 3;
    }

    // input image right corner finding
    if (iTL_corner.X > iTR_corner.X) {
        right_corner.Y = iTL_corner.Y;
        right_corner.X = iTL_corner.X; // right_corner_src = 0;
    } else {
        right_corner.Y = iTR_corner.Y;
        right_corner.X = iTR_corner.X; // right_corner_src = 1;
    }
    if (iBL_corner.X > right_corner.X) {
        right_corner.Y = iBL_corner.Y;
        right_corner.X = iBL_corner.X; // right_corner_src = 2;
    }
    if (iBR_corner.X > right_corner.X) {
        right_corner.Y = iBR_corner.Y;
        right_corner.X = iBR_corner.X; // right_corner_src = 3;
    }

    rot_pars.IBufH_req = (uint32_t)fabs(ceil(rot_pars.Yi_last - rot_pars.Yi_first)) + 1;
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    YTL = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_first,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    YTR = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_first,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    YBL = coord_out[0];
    IRT_top::irt_iicc_float(coord_out,
                            irt_desc.prj_Bf,
                            irt_desc.prj_Df,
                            (float)rot_pars.Xo_last,
                            (float)rot_pars.Yo_last,
                            0,
                            0,
                            1,
                            1.0,
                            e_irt_projection);
    YBR                = coord_out[0];
    rot_pars.IBufH_req = (uint32_t)ceil(std::fmax(fabs(YTR - YTL), fabs(YBR - YBL)));

    rot_pars.Yi_first = (double)top_corner.Y;
    rot_pars.Yi_last  = (double)bot_corner.Y;
    rot_pars.Xi_first = (double)left_corner.X;
    rot_pars.Xi_last  = (double)right_corner.X;

    if (irt_desc.irt_mode == e_irt_projection) {
        if (std::min(iTL_corner.Y, iTR_corner.Y) >
            std::max(iBL_corner.Y, iBR_corner.Y)) { // top and bottom output image line are swapped in input image
            // irt_desc.read_vflip = 1;
        }
        if (std::min(iTL_corner.X, iBL_corner.X) >
            std::max(iTR_corner.X, iBR_corner.X)) { // left and right output image line are swapped in input image
            // irt_desc.read_hflip = 1;
        }
    }

    rot_pars.im_read_slope       = 0;
    irt_desc.image_par[IIMAGE].S = (int16_t)fabs(ceil(rot_pars.Xi_last) - floor(rot_pars.Xi_first)) + 2;
    rot_pars.Xi_start            = (int16_t)floor(rot_pars.Xi_first); // -irt_desc.image_par[IIMAGE].S;

    // calculating input image coordinates according to actual implemetation
    float  yo, xos, xis, xie, yis, yie, xi_slope, yi_slope, xi, yi;
    double Ymax_line;
    double Ymin = 100000, Ymax = -100000;
    rot_pars.Yi_first  = 100000;
    rot_pars.Xi_first  = 100000;
    rot_pars.Yi_last   = -100000;
    rot_pars.Xi_last   = -100000;
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

        Ymin = 100000;
        Ymax = -100000;

        for (uint16_t col = 0; col < irt_desc.image_par[OIMAGE].S; col += irt_desc.proc_size) {

            xos = (float)col - (float)irt_desc.image_par[OIMAGE].Xc / 2;
            yo  = (float)row - (float)irt_desc.image_par[OIMAGE].Yc / 2;
            xis = ((irt_desc.prj_Af[0] * xos + irt_desc.prj_Af[1] * yo) + irt_desc.prj_Af[2]) /
                  ((irt_desc.prj_Cf[0] * xos + irt_desc.prj_Cf[1] * yo) + irt_desc.prj_Cf[2]);
            xie = ((irt_desc.prj_Af[0] * xos + irt_desc.prj_Af[1] * yo) +
                   irt_desc.prj_Af[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Af[2]) /
                  ((irt_desc.prj_Cf[0] * xos + irt_desc.prj_Cf[1] * yo) +
                   irt_desc.prj_Cf[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Cf[2]);
            yis = ((irt_desc.prj_Bf[0] * xos + irt_desc.prj_Bf[1] * yo) + irt_desc.prj_Bf[2]) /
                  ((irt_desc.prj_Df[0] * xos + irt_desc.prj_Df[1] * yo) + irt_desc.prj_Df[2]);
            yie = ((irt_desc.prj_Bf[0] * xos + irt_desc.prj_Bf[1] * yo) +
                   irt_desc.prj_Bf[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Bf[2]) /
                  ((irt_desc.prj_Df[0] * xos + irt_desc.prj_Df[1] * yo) +
                   irt_desc.prj_Df[0] * (irt_desc.proc_size - 1) + irt_desc.prj_Df[2]);
            xi_slope = (xie - xis) * (float)(1.0 / (irt_desc.proc_size - 1));
            yi_slope = (yie - yis) * (float)(1.0 / (irt_desc.proc_size - 1));

            for (uint8_t pixel = 0; pixel < irt_desc.proc_size; pixel++) {
                // xo = (float)col + (float)pixel - (float)irt_desc.image_par[OIMAGE].Xc / 2;

                if (irt_desc.proc_size == 1) {
                    xi = xis;
                    yi = yis;
                } else {
                    xi = xis + xi_slope * pixel;
                    yi = yis + yi_slope * pixel;
                }

                irt_cfg.mesh_images.proj_image_full[row][col + pixel].x = xi;
                irt_cfg.mesh_images.proj_image_full[row][col + pixel].y = yi;

                irt_map_image_pars_update<mesh_xy_fp64_meta, double>(irt_cfg,
                                                                     irt_desc,
                                                                     irt_cfg.mesh_images.proj_image_full,
                                                                     row,
                                                                     col + pixel,
                                                                     Ymax_line,
                                                                     fptr_vals,
                                                                     fptr_pars);

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

    irt_map_iimage_stripe_adj<mesh_xy_fp64_meta>(
        irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.proj_image_full);
    rot_pars.im_read_slope = 0;
    irt_desc.rot_dir       = (rot_pars.irt_angles_adj[e_irt_angle_roll] >= 0) ? IRT_ROT_DIR_POS : IRT_ROT_DIR_NEG;

#endif
}

void irt_mesh_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

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

    irt_map_iimage_stripe_adj<mesh_xy_fp64_meta>(
        irt_cfg, rot_pars, irt_desc, desc, irt_cfg.mesh_images.mesh_image_intr);
    rot_pars.im_read_slope = 0;
}

void irt_resamp_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{
    if (irt_desc.resize_bli_grad_en == 0) {
        rot_pars.im_read_slope        = 0;
        irt_desc.image_par[OIMAGE].Xc = 0;
        irt_desc.image_par[OIMAGE].Yc = 0;
        irt_desc.image_par[IIMAGE].Xc = 0;
        irt_desc.image_par[IIMAGE].Yc = 0;
    }
    irt_desc.image_par[IIMAGE].buf_mode = 3;
    irt_desc.image_par[IIMAGE].S        = 128;
    //---------------------------------------
    irt_desc.image_par[MIMAGE].W = irt_desc.image_par[OIMAGE].W;
    irt_desc.image_par[MIMAGE].S = irt_desc.image_par[OIMAGE].S;
    irt_desc.image_par[MIMAGE].H = irt_desc.image_par[OIMAGE].H;
    irt_desc.mesh_Gh             = 1;
    irt_desc.mesh_Gv             = 1;
    irt_desc.mesh_sparse_h       = 0;
    irt_desc.mesh_sparse_v       = 0;
    //---------------------------------------
    irt_desc.image_par[GIMAGE].W = irt_desc.image_par[OIMAGE].W;
    irt_desc.image_par[GIMAGE].S = irt_desc.image_par[OIMAGE].S;
    irt_desc.image_par[GIMAGE].H = irt_desc.image_par[OIMAGE].H;
    //---------------------------------------
    // TODO - review wqith arch
    if (irt_desc.irt_mode == e_irt_resamp_bwd2) {
        irt_desc.crd_mode                   = e_irt_crd_mode_fp32;
        irt_cfg.buf_format[e_irt_block_rot] = e_irt_buf_format_static;
    }
}

void irt_BiLiGrad_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc) {}

void irt_rescale_desc_gen(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{
    IRT_TRACE_UTILS("Inside irt_rescale_desc_gen func\n");
    rot_pars.im_read_slope = 0;
    // irt_desc.image_par[MIMAGE].addr_start = irt_desc.image_par[IIMAGE].addr_start; //TODO - InitIMage preloads the
    // image to IIMAGE.star_addr whle warp_read reads from MIMAGE.start_addr, hence a hack for now
    //	memset(&irt_desc.image_par[IIMAGE], 0, sizeof(image_par));
    //	memset(&irt_desc.image_par[GIMAGE], 0, sizeof(image_par));
    irt_desc.image_par[MIMAGE].W = irt_desc.image_par[IIMAGE].W;
    //	irt_desc.image_par[MIMAGE].S = irt_desc.image_par[IIMAGE].W;
    irt_desc.image_par[MIMAGE].H = irt_desc.image_par[IIMAGE].H;

    // To make sure RTL is not using IP image params randomizing IP image parameters
    // TODO - Protect address randomization for SW
    //#ifdef RUN_WITH_SV
    //  irt_desc.image_par[IIMAGE].addr_start = rand();
    //  irt_desc.image_par[IIMAGE].W = irt_desc.image_par[IIMAGE].W + (rand() % 200);
    //  irt_desc.image_par[IIMAGE].H = irt_desc.image_par[IIMAGE].H + (rand() % 200);
    //#endif
    // TODO - nudge to be transalated to be translated to center
    ////format S15.1
    // TODO - PLAHOTI Review with Ambili, commented center =0
    // irt_desc.image_par[OIMAGE].Xc = 0;
    // irt_desc.image_par[OIMAGE].Yc = 0;
    irt_desc.mesh_sparse_h = 0;
    irt_desc.mesh_sparse_v = 0;
    //---------------------------------------
    irt_desc.rescale_LUT_x              = (Coeff_LUT_t*)malloc(sizeof(Coeff_LUT_t));
    irt_desc.rescale_LUT_x->Coeff_table = (float*)malloc(128 * 16 * sizeof(float)); // 8K table
    irt_desc.rescale_LUT_y              = (Coeff_LUT_t*)malloc(sizeof(Coeff_LUT_t));
    irt_desc.rescale_LUT_y->Coeff_table = (float*)malloc(128 * 16 * sizeof(float)); // 8K table
    rot_pars.filter_type                = rot_pars.filter_type; //  e_irt_lanczos2;//TODO
    IRT_TRACE_UTILS("Calling rescale_coeff_gen with OP X center %d and IP X center %d\n",
                    irt_desc.image_par[OIMAGE].Xc / 2,
                    irt_desc.image_par[IIMAGE].Xc / 2);
    rescale_coeff_gen(irt_desc.rescale_LUT_x,
                      irt_desc.image_par[MIMAGE].W,
                      irt_desc.image_par[OIMAGE].W,
                      (int)rot_pars.filter_type,
                      irt_cfg.lanczos_max_phases_h,
                      irt_desc.rescale_prec,
                      irt_desc.image_par[OIMAGE].Xc / 2,
                      irt_desc.image_par[IIMAGE].Xc / 2);
    rescale_coeff_gen(irt_desc.rescale_LUT_y,
                      irt_desc.image_par[MIMAGE].H,
                      irt_desc.image_par[OIMAGE].H,
                      (int)rot_pars.filter_type,
                      irt_cfg.lanczos_max_phases_v,
                      irt_desc.rescale_prec,
                      irt_desc.image_par[OIMAGE].Yc / 2,
                      irt_desc.image_par[IIMAGE].Yc / 2);
    if (irt_cfg.ovrld_rescale_coeff == 1) {
#ifdef RUN_WITH_SV
        srand(1);
#endif
        IRT_TRACE_UTILS("Overloading rescale co-efficients\n");
        for (int i = 0; i < irt_cfg.lanczos_max_phases_h; i++) {
            for (int j = 0; j < irt_desc.rescale_LUT_x->num_taps; j++) {
                irt_desc.rescale_LUT_x->Coeff_table[(irt_desc.rescale_LUT_x->num_taps * i) + j] = RandomGenerator();
                // IRT_TRACE_UTILS("Table X : phase %d tap %d co-efficient[%d]
                // %f\n",i,j,(irt_desc.rescale_LUT_x->num_taps*i)+j,irt_desc.rescale_LUT_x->Coeff_table[(irt_desc.rescale_LUT_x->num_taps*i)+j]);
            }
        }
        for (int i = 0; i < irt_cfg.lanczos_max_phases_v; i++) {
            for (int j = 0; j < irt_desc.rescale_LUT_y->num_taps; j++) {
                irt_desc.rescale_LUT_y->Coeff_table[(irt_desc.rescale_LUT_y->num_taps * i) + j] = RandomGenerator();
                // IRT_TRACE_UTILS("Table Y : phase %d tap %d co-efficient[%d]
                // %f\n",i,j,(irt_desc.rescale_LUT_y->num_taps*i)+j,irt_desc.rescale_LUT_y->Coeff_table[(irt_desc.rescale_LUT_y->num_taps*i)+j]);
            }
        }
    }
    //---------------------------------------
    irt_desc.prec_align            = 31 - rot_pars.rescale_Gx_prec; // TODO - Review
    irt_desc.rescale_phases_prec_H = 31 - (int)log2(irt_desc.rescale_LUT_x->num_phases); // TODO - Review
    irt_desc.rescale_phases_prec_V = 31 - (int)log2(irt_desc.rescale_LUT_y->num_phases); // TODO - Review
    int fixed_prec_factor          = 1 << rot_pars.rescale_Gx_prec;
    irt_desc.mesh_Gh = round(irt_desc.rescale_LUT_x->Gf * fixed_prec_factor); // TODO - FLOAT TO HEX CONVERSION -
    irt_desc.mesh_Gv = round(irt_desc.rescale_LUT_y->Gf * fixed_prec_factor);
    //---------------------------------------
    // Input stripe stripe width & Inpu
    irt_desc.image_par[MIMAGE].S =
        (uint16_t)ceil(((double)irt_desc.image_par[OIMAGE].S - 1) * irt_desc.rescale_LUT_x->Gf) +
        (float)irt_desc.rescale_LUT_x->num_taps /* + 1*/; // TODO : +1 REQUIRED ??
    if ((irt_desc.image_par[MIMAGE].S > irt_desc.image_par[MIMAGE].W) && (irt_desc.bg_mode == 1)) {
        irt_desc.image_par[MIMAGE].S = irt_desc.image_par[MIMAGE].W;
    }

    irt_desc.mesh_stripe_stride =
        irt_desc.image_par[MIMAGE].S -
        ceil((irt_desc.rescale_LUT_x->num_taps / 2)); // TO ALLOW LEFT & RIGHT SIDE NUM_TAPS/2 PIXELS read
    //---------------------------------------
    // TODO = Saturate for bd_mode
    float Yi = (irt_desc.rescale_LUT_y->Gf * (0 - (float)irt_desc.image_par[OIMAGE].Yc / 2)) +
               ((float)irt_desc.image_par[IIMAGE].Yc / 2);
    rot_pars.Yi_first = floor(Yi) - ceil(((float)irt_desc.rescale_LUT_y->num_taps / 2) - 1);
    // bg_mode=1 saturation
    if ((irt_desc.bg_mode == e_irt_bg_frame_repeat) && (rot_pars.Yi_first < 0)) {
        rot_pars.Yi_first = 0;
    }

    Yi = (irt_desc.rescale_LUT_y->Gf *
          ((irt_desc.image_par[OIMAGE].H - 1) - ((float)irt_desc.image_par[OIMAGE].Yc / 2))) +
         ((float)irt_desc.image_par[IIMAGE].Yc / 2);
    // Need rounding for phase=1. This mode can't support any fraction.
    Yi               = round(Yi * irt_desc.rescale_LUT_y->num_phases) / irt_desc.rescale_LUT_y->num_phases;
    rot_pars.Yi_last = floor(Yi) + ceil((float)irt_desc.rescale_LUT_y->num_taps / 2);
    // bg_mode=1 saturation
    if ((irt_desc.bg_mode == e_irt_bg_frame_repeat) && (rot_pars.Yi_last > (irt_desc.image_par[MIMAGE].H - 1))) {
        rot_pars.Yi_last = irt_desc.image_par[MIMAGE].H - 1;
    }
    irt_desc.Yi_start = (int16_t)floor(rot_pars.Yi_first);
    irt_desc.Yi_end   = (int16_t)ceil(rot_pars.Yi_last);
    IRT_TRACE_UTILS("Before RESCALE :: Yi_first:last [%f , %f]  Yi_start:end [%hi , %hi]\n",
                    rot_pars.Yi_first,
                    rot_pars.Yi_last,
                    irt_desc.Yi_start,
                    irt_desc.Yi_end);
    if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) {
        IRT_TRACE_UTILS("RESCALE: Inside dynamic buf_format_mode\n");
        irt_desc.Yi_end =
            irt_desc.Yi_start +
            (uint16_t)(ceil(((double)irt_desc.Yi_end - irt_desc.Yi_start + 1) / IRT_MM_DYN_MODE_LINE_RELEASE) *
                       IRT_MM_DYN_MODE_LINE_RELEASE) -
            1;
    }
    IRT_TRACE_UTILS("RESCALE :: Yi_first:last [%f , %f]  Yi_start:end [%f , %f]\n",
                    rot_pars.Yi_first,
                    rot_pars.Yi_last,
                    irt_desc.Yi_start,
                    irt_desc.Yi_end);

    // TODO - To be supported for bg_mode=1 cases
    float Xi = (irt_desc.rescale_LUT_x->Gf * (0 - ((float)irt_desc.image_par[OIMAGE].Xc / 2))) +
               ((float)irt_desc.image_par[IIMAGE].Xc / 2);
    rot_pars.Xi_first = floor(Xi) - ceil((float)irt_desc.rescale_LUT_x->num_taps / 2) + 1;
    // bg_mode=1 saturation
    if ((irt_desc.bg_mode == e_irt_bg_frame_repeat) && (rot_pars.Xi_first < 0)) {
        rot_pars.Xi_first = 0;
    }
    Xi = (irt_desc.rescale_LUT_x->Gf *
          ((irt_desc.image_par[OIMAGE].W - 1) - ((float)irt_desc.image_par[OIMAGE].Xc / 2))) +
         ((float)irt_desc.image_par[IIMAGE].Xc / 2);
    rot_pars.Xi_last = floor(Xi) + ceil((float)irt_desc.rescale_LUT_x->num_taps / 2);
    // bg_mode=1 saturation
    if ((irt_desc.bg_mode == e_irt_bg_frame_repeat) && (rot_pars.Xi_last > (irt_desc.image_par[IIMAGE].W - 1))) {
        rot_pars.Xi_last = irt_desc.image_par[IIMAGE].W - 1;
    }

    irt_desc.Xi_start_offset = (int16_t)floor(rot_pars.Xi_first);
    rot_pars.Xi_start        = (int16_t)floor(rot_pars.Xi_first);
    irt_desc.Xi_last_fixed   = (int16_t)ceil(rot_pars.Xi_last);
    //---------------------------------------
    // if (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) { //stripe height is multiple of 2
    //	//irt_desc.image_par[MIMAGE].H = (uint16_t)(ceil((double)irt_desc.image_par[MIMAGE].H /
    // IRT_MM_DYN_MODE_LINE_RELEASE) * IRT_MM_DYN_MODE_LINE_RELEASE); 	irt_desc.image_par[MIMAGE].H =
    // irt_desc.image_par[IIMAGE].H;
    //	//IRT_TRACE_UTILS("Mesh matrix calculation: adjusting Hm to %d be multiple of 2 because of mesh buffer dynamic
    // format\n", irt_desc.image_par[MIMAGE].H);
    //}
    irt_mesh_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);
    if (irt_desc.image_par[OIMAGE].W != irt_desc.image_par[OIMAGE].S) {
        IRT_TRACE_UTILS("RESCALE : OP image W is %d and S is %d hence reducing OP W to %d\n",
                        irt_desc.image_par[OIMAGE].W,
                        irt_desc.image_par[OIMAGE].S,
                        irt_desc.image_par[OIMAGE].S);
        irt_desc.image_par[OIMAGE].W = irt_desc.image_par[OIMAGE].S;
    }
    // Set correct mesh_buf_mode

    //
    //---------------------------------------
    char FxFile[150];
    sprintf(FxFile, "filter_x_%d.txt", desc);
    char FyFile[150];
    sprintf(FyFile, "filter_y_%d.txt", desc);
    print_filter_coeff(irt_desc.rescale_LUT_x->Coeff_table,
                       irt_desc.rescale_LUT_x->num_phases,
                       irt_desc.rescale_LUT_x->num_taps,
                       FxFile);
    print_filter_coeff(irt_desc.rescale_LUT_y->Coeff_table,
                       irt_desc.rescale_LUT_y->num_phases,
                       irt_desc.rescale_LUT_y->num_taps,
                       FyFile);
    //---------------------------------------
    // DUMP RESCALE COOEF IMAGE DUMP FOR SIVAL FORMAT
#ifndef RUN_WITH_SV
    FILE*    fpy = nullptr;
    char     fx[50], fy[50];
    uint64_t coord_pair;
    sprintf(fy, "sival_rescale_coefficient_Y_task%d.txt", desc);
    fpy = fopen(fy, "w");
    //---------------------------------------
    fprintf(fpy,
            "// NUM_OF_PHASES : %0d \tNUM_OF_TAPS :  %0d \n",
            irt_desc.rescale_LUT_y->num_phases,
            irt_desc.rescale_LUT_y->num_taps);
    for (uint32_t i = 0; i < irt_desc.rescale_LUT_y->num_phases; i++) {
        for (uint32_t j = 0; j < irt_desc.rescale_LUT_y->num_taps; j++) {
            fprintf(fpy,
                    "%04x\n",
                    (uint32_t)IRT_top::IRT_UTILS::irt_float_to_fp32(
                        irt_desc.rescale_LUT_y->Coeff_table[i * irt_desc.rescale_LUT_y->num_taps + j]));
        }
    }
    fclose(fpy);
    //---------------------------------------
    sprintf(fx, "sival_rescale_coefficient_X_task%d.txt", desc);
    FILE* fpx = nullptr;
    fpx       = fopen(fx, "w");
    fprintf(fpx,
            "// NUM_OF_PHASES : %0d \tNUM_OF_TAPS :  %0d \n",
            irt_desc.rescale_LUT_x->num_phases,
            irt_desc.rescale_LUT_x->num_taps);
    for (uint32_t i = 0; i < irt_desc.rescale_LUT_x->num_phases; i++) {
        for (uint32_t j = 0; j < irt_desc.rescale_LUT_x->num_taps; j++) {
            fprintf(fpx,
                    "%04x\n",
                    (uint32_t)IRT_top::IRT_UTILS::irt_float_to_fp32(
                        irt_desc.rescale_LUT_x->Coeff_table[i * irt_desc.rescale_LUT_x->num_taps + j]));
        }
    }
    fclose(fpx);
#endif
}

void irt_xi_first_adj_rot(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 59) {
        irt_desc.Xi_start_offset      = 0;
        irt_desc.Xi_start_offset_flip = 0;

        if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle
            //			irt_desc.Xi_start_offset = 0;
        } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 45) {
            // irt_desc.Xi_start_offset = 0; //1
            // rot_pars.Xi_first -= 1;
        } else {
            // irt_desc.Xi_start_offset = 0;//2;
            // rot_pars.Xi_first -= 2;
        }

        if (irt_desc.read_hflip) {
            if (irt_desc.rot_dir == IRT_ROT_DIR_POS) { // positive rotation angle
                // irt_desc.Xi_start_offset_flip = 0;
            } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 26) {
                // irt_desc.Xi_start_offset_flip = 0;//1;
                // rot_pars.Xi_first -= 1;
            } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) <= 56) {
                // irt_desc.Xi_start_offset_flip = 0;//2;//3;
                // rot_pars.Xi_first -= 2;
            } else {
                // irt_desc.Xi_start_offset_flip = 0;//3;//3;
                // rot_pars.Xi_first -= 3;
            }
        }
    } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 80) {
        rot_pars.Xi_first -= 0; // 7;//(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)));
        irt_desc.Xi_start_offset      = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 6; // 0;
        irt_desc.Xi_start_offset_flip = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 3; // 3;//5;
    } else {

        if (irt_desc.rot90 == 0) {
            // rot_pars.Xi_first -= /*60 +*/ ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0)));
            IRT_TRACE_UTILS("Xi_start_adj: decreasing Xi_first by %d for adjusted rotation angle %f\n",
                            (int)ceil(tan(fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]))),
                            rot_pars.irt_angles_adj[e_irt_angle_rot]);
        }
        irt_desc.Xi_start_offset = irt_desc.rot_dir == IRT_ROT_DIR_POS ? 0 : 60;
        if (irt_desc.rot_dir == IRT_ROT_DIR_NEG && irt_desc.rot90 == 1) {
            // irt_desc.Xi_start_offset = 2;
        }

        if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {
            if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 85) {
                irt_desc.Xi_start_offset_flip = 5; // 70;//4;//55;
            } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 89) {
                irt_desc.Xi_start_offset_flip = 7; // 70;//4;//55;
            } else {
                irt_desc.Xi_start_offset_flip = 58;
            }
        } else {
            if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 85) {
                irt_desc.Xi_start_offset_flip = -47; // 78;//4;//55;
            } else if (fabs(rot_pars.irt_angles_adj[e_irt_angle_rot]) < 89) {
                irt_desc.Xi_start_offset_flip = -36;
            } else {
                irt_desc.Xi_start_offset_flip = 56;
            }
        }
    }

    irt_desc.Xi_start_offset      = 0;
    irt_desc.Xi_start_offset_flip = 0;
    if (irt_desc.rot90) {
        if (/*irt_desc.rot90_intv ||*/ irt_desc
                .rot90_inth) { // interpolation will be required, rot90 will be reset for model
            if (irt_desc.rot_dir == IRT_ROT_DIR_POS) {
                rot_pars.Xi_first += irt_desc.image_par[IIMAGE].S;
            } else {
                rot_pars.Xi_first -= irt_desc.image_par[IIMAGE].S - 1;
            }
        } else { // interpolation is not required, working regulary as rot90
            if (irt_desc.rot_dir == IRT_ROT_DIR_NEG) {
                // irt_desc.Xi_start_offset = 2;
                rot_pars.Xi_first -= 2;
            }
        }
    }
}

void irt_xi_first_adj_aff(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& irt_desc, uint8_t desc)
{

    if (rot_pars.affine_flags[e_irt_aff_shearing] || rot_pars.affine_flags[e_irt_aff_reflection] ||
        rot_pars.affine_flags[e_irt_aff_scaling]) //(!strcmp(rot_pars.affine_mode, "T"))
        rot_pars.Xi_first -= 0;
    else if (rot_pars.affine_flags[e_irt_aff_rotation] == 0) { // strcmp(rot_pars.affine_mode, "R")
        rot_pars.Xi_first -= 5;
        IRT_TRACE_UTILS("Decreasing Xi_start by 5 in affine mode\n");
    } else
        irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc);
}

void irt_descriptor_gen(irt_cfg_pars& irt_cfg,
                        rotation_par& rot_pars,
                        irt_desc_par& irt_desc,
                        uint32_t      Hsi,
                        uint8_t       desc)
{

    IRT_TRACE_UTILS("Task %d: Generating descriptor for %s mode\n", desc, irt_irt_mode_s[irt_desc.irt_mode]);
    irt_desc.Ho = irt_desc.image_par[OIMAGE].H;
    irt_desc.Wo = irt_desc.image_par[OIMAGE].W;

    if (irt_desc.image_par[OIMAGE].W > OIMAGE_W || irt_desc.image_par[OIMAGE].H > OIMAGE_H) {
        if (rot_pars.oimg_auto_adj == 0) {
            IRT_TRACE_UTILS("Desc gen error: output image size %dx%d is not supported, exceeds %dx%d maximum supported "
                            "resolution\n",
                            irt_desc.image_par[OIMAGE].W,
                            irt_desc.image_par[OIMAGE].H,
                            OIMAGE_W,
                            OIMAGE_H);
            IRT_TRACE_TO_RES_UTILS(
                test_res,
                "was not run, output image size %dx%d is not supported, exceeds %dx%d maximum supported resolution\n",
                irt_desc.image_par[OIMAGE].W,
                irt_desc.image_par[OIMAGE].H,
                OIMAGE_W,
                OIMAGE_H);
            IRT_CLOSE_FAILED_TEST(0);
        } else {
            IRT_TRACE_UTILS("Desc %d gen: adjusting output image %dx%d to maximum supported size",
                            desc,
                            irt_desc.image_par[OIMAGE].W,
                            irt_desc.image_par[OIMAGE].H);
            irt_desc.image_par[OIMAGE].W = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].W, OIMAGE_W);
            irt_desc.image_par[OIMAGE].H = (uint16_t)std::min((int32_t)irt_desc.image_par[OIMAGE].H, OIMAGE_H);
            IRT_TRACE_UTILS(" %dx%d\n", irt_desc.image_par[OIMAGE].W, irt_desc.image_par[OIMAGE].H);
        }
    }

    if (irt_desc.irt_mode <= 3) {
        irt_params_analize(irt_cfg, rot_pars, irt_desc, desc);
        irt_rot_angle_adj_calc(irt_cfg, rot_pars, irt_desc, desc);
    }

    rot_pars.WEIGHT_PREC =
        rot_pars.MAX_PIXEL_WIDTH + IRT_WEIGHT_PREC_EXT; // +1 because of 0.5 error, +2 because of interpolation, -1
                                                        // because of weigths multiplication
    rot_pars.COORD_PREC  = rot_pars.MAX_COORD_WIDTH + IRT_COORD_PREC_EXT; // +1 because of polynom in Xi/Yi calculation
    rot_pars.COORD_ROUND = (1 << (rot_pars.COORD_PREC - 1));
    rot_pars.TOTAL_PREC  = rot_pars.COORD_PREC + rot_pars.WEIGHT_PREC;
    rot_pars.TOTAL_ROUND = (1 << (rot_pars.TOTAL_PREC - 1));
    rot_pars.PROJ_NOM_PREC  = rot_pars.TOTAL_PREC;
    rot_pars.PROJ_DEN_PREC  = (rot_pars.TOTAL_PREC + 10);
    rot_pars.PROJ_NOM_ROUND = (1 << (rot_pars.PROJ_NOM_PREC - 1));
    rot_pars.PROJ_DEN_ROUND = (1 << (rot_pars.PROJ_DEN_PREC - 1));
    irt_desc.prec_align     = 31 - rot_pars.TOTAL_PREC;

    //---------------------------------------
    // TODO = add assertion for bgmode=0 is invalid for resamp case
    //---------------------------------------
    // TODO - Need to merge below logic for standalone setup Vs RUN_SV setup. Same code replicated twice here
    if (irt_desc.irt_mode <= 3) {
        irt_desc.image_par[OIMAGE].Ps = rot_pars.Pwo <= 8 ? 0 : 1;
        irt_desc.image_par[IIMAGE].Ps = rot_pars.Pwi <= 8 ? 0 : 1;
        irt_desc.image_par[MIMAGE].Ps = irt_desc.mesh_format == e_irt_mesh_flex ? 2 : 3;
    } else {
        switch (irt_desc.image_par[OIMAGE].DataType) {
            case e_irt_int8: irt_desc.image_par[OIMAGE].Ps = 0; break;
            case e_irt_int16: irt_desc.image_par[OIMAGE].Ps = 1; break;
            case e_irt_fp16: irt_desc.image_par[OIMAGE].Ps = 1; break;
            case e_irt_bfp16: irt_desc.image_par[OIMAGE].Ps = 1; break;
            case e_irt_fp32: irt_desc.image_par[OIMAGE].Ps = 2; break;
        }
        switch (irt_desc.image_par[IIMAGE].DataType) {
            case e_irt_int8: irt_desc.image_par[IIMAGE].Ps = 0; break;
            case e_irt_int16: irt_desc.image_par[IIMAGE].Ps = 1; break;
            case e_irt_fp16: irt_desc.image_par[IIMAGE].Ps = 1; break;
            case e_irt_bfp16: irt_desc.image_par[IIMAGE].Ps = 1; break;
            case e_irt_fp32: irt_desc.image_par[IIMAGE].Ps = 2; break;
        }
        switch (irt_desc.image_par[MIMAGE].DataType) {
            case e_irt_int8: irt_desc.image_par[MIMAGE].Ps = 1; break;
            case e_irt_int16: irt_desc.image_par[MIMAGE].Ps = 2; break;
            case e_irt_fp16: irt_desc.image_par[MIMAGE].Ps = 2; break;
            case e_irt_bfp16: irt_desc.image_par[MIMAGE].Ps = 2; break;
            case e_irt_fp32: irt_desc.image_par[MIMAGE].Ps = 3; break;
        }
        switch (irt_desc.image_par[GIMAGE].DataType) {
            case e_irt_int8: irt_desc.image_par[GIMAGE].Ps = 0; break;
            case e_irt_int16: irt_desc.image_par[GIMAGE].Ps = 1; break;
            case e_irt_fp16: irt_desc.image_par[GIMAGE].Ps = 1; break;
            case e_irt_bfp16: irt_desc.image_par[GIMAGE].Ps = 1; break;
            case e_irt_fp32: irt_desc.image_par[GIMAGE].Ps = 2; break;
        }
        if (irt_desc.irt_mode == e_irt_rescale) {
            irt_desc.image_par[MIMAGE].Ps--; // TODO - PLAHOTI Why Ps is reduced by 1?
            // IRT_TRACE_UTILS("->>inside rescale Ps %d\n",irt_desc.image_par[MIMAGE].Ps);
        }
        if (irt_desc.irt_mode == e_irt_resamp_bwd1) { // BWD1 has 2 elements per output pix
            irt_desc.image_par[OIMAGE].Ps++;
        }
    }
    if (irt_desc.image_par[IIMAGE].DataType == e_irt_int16 || irt_desc.image_par[IIMAGE].DataType == e_irt_int8) {
        irt_desc.Msi = ((1 << rot_pars.Pwi) - 1) << rot_pars.Ppi;
    } else {
        irt_desc.Msi = 0xFFFF;
    }
    irt_desc.bli_shift    = rot_pars.Pwo - (rot_pars.Pwi + rot_pars.Ppi);
    irt_desc.MAX_VALo     = (1 << rot_pars.Pwo) - 1; // output pixel max value, equal to 2^pixel_width - 1
    irt_desc.rescale_prec = LANCZOS_MAX_PREC; // sjagadale - TODO  - REVIEW WITH ARCH

    // precision overflow prevension checking
    if (rot_pars.Pwi + (rot_pars.TOTAL_PREC - rot_pars.WEIGHT_SHIFT) * 2 + 2 > 66) {
        IRT_TRACE_UTILS("Desc gen: Input pixel width %d, coordinate width %d, max pixel width %d and weight shift %d "
                        "are not supported: exceeds 64 bits calculation in BLI\n",
                        rot_pars.Pwi,
                        rot_pars.MAX_COORD_WIDTH,
                        rot_pars.MAX_PIXEL_WIDTH,
                        rot_pars.WEIGHT_SHIFT);
        IRT_TRACE_TO_RES_UTILS(test_res,
                               "Input pixel width %d, coordinate width %d, max pixel width %d and weight shift %d are "
                               "not supported: exceeds 64 bits calculation in BLI\n",
                               rot_pars.Pwi,
                               rot_pars.MAX_COORD_WIDTH,
                               rot_pars.MAX_PIXEL_WIDTH,
                               rot_pars.WEIGHT_SHIFT);
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

    if ((irt_desc.irt_mode <= 3) ||
        (irt_desc.irt_mode == 6 && irt_desc.resize_bli_grad_en == 1)) { // TODO - REVIEW WITH Predeep
        if (irt_desc.irt_mode <= 3)
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
        if (irt_desc.rot90) { // fix C++ bug that cos(90) is not zero
            rot_pars.cosd = 0;
        }
        rot_pars.sind = sin(rot_pars.irt_angles_adj[e_irt_angle_rot] * M_PI / 180.0);
        irt_affine_coefs_calc(irt_cfg, rot_pars, irt_desc, desc);
        irt_projection_coefs_calc(irt_cfg, rot_pars, irt_desc, desc);

        rot_pars.mesh_dist_r0_Sh1_Sv1 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv == 1;
        // rot_pars.mesh_dist_r0_Sh1_Sv1_fp32 = rot_pars.dist_r == 0 && rot_pars.mesh_Sh == 1 && rot_pars.mesh_Sv ==
        // 1;//&& irt_desc.mesh_format == e_irt_mesh_fp32;

        if (irt_desc.irt_mode == e_irt_mesh) {
            irt_mesh_matrix_calc(irt_cfg, rot_pars, irt_desc, desc, 1);
        }

        // precision overflow prevension checking
        // affine coefficient in fixed point is presented as SI.F of 32 bits.
        // F width = 32 - 1 - I_width
        double M_max      = std::max((double)fabs(rot_pars.M11d), fabs(rot_pars.M12d));
        M_max             = std::max((double)M_max, fabs(rot_pars.M21d));
        M_max             = std::max((double)M_max, fabs(rot_pars.M22d)); // maximum value of affine coefficients
        int     M_max_int = (int)floor(M_max); // integer part of max value
        uint8_t M_I_width = (uint8_t)ceil(log2(M_max_int + 1));
        uint8_t M_F_width = 31 - M_I_width; // remained bits for fraction

        // IRT_TRACE_UTILS("Desc gen: Affine matrix max coef %f (%d), M_I_width %d, M_F_width %d\n", (double)M_max,
        // M_max_int, M_I_width, M_F_width);
        if (M_F_width < rot_pars.TOTAL_PREC) { // TOTAL selected precision > bits allowed for fraction presentation of M
            IRT_TRACE_UTILS("Desc gen: Coordinate width %d, max pixel width %d and total precision %d are not "
                            "supported: affine matrix coefficients exceed 32 bits by %d bits\n",
                            rot_pars.MAX_COORD_WIDTH,
                            rot_pars.MAX_PIXEL_WIDTH,
                            rot_pars.TOTAL_PREC,
                            rot_pars.TOTAL_PREC - M_F_width);
            if (rot_pars.rot_prec_auto_adj == 0) {
                IRT_TRACE_TO_RES_UTILS(test_res,
                                       "Coordinate width %d, max pixel width %d and total precision %d are not "
                                       "supported: affine matrix coefficients exceed 32 bits by %d bits\n",
                                       rot_pars.MAX_COORD_WIDTH,
                                       rot_pars.MAX_PIXEL_WIDTH,
                                       rot_pars.TOTAL_PREC,
                                       rot_pars.TOTAL_PREC - M_F_width);
                IRT_CLOSE_FAILED_TEST(0);
            } else {
                IRT_TRACE_UTILS("Adjusting precision to %d bits\n", M_F_width);
                rot_pars.TOTAL_PREC  = M_F_width;
                rot_pars.TOTAL_ROUND = (1 << (rot_pars.TOTAL_PREC - 1));
            }
        }
    } /*else{
        irt_mesh_matrix_calc(irt_cfg, rot_pars, irt_desc, desc, 1);
     }*/
    rot_pars.bli_shift_fix =
        (2 * rot_pars.TOTAL_PREC) - (rot_pars.Pwo - (rot_pars.Pwi + rot_pars.Ppi)) - 1; // bi-linear interpolation shift
    irt_desc.prec_align = 31 - rot_pars.TOTAL_PREC;

    if (irt_desc.irt_mode <= 3 || (irt_desc.irt_mode == 6 && irt_desc.resize_bli_grad_en)) {
        rot_pars.cos16 = (int)rint(rot_pars.cosd * pow(2.0, 16));
        rot_pars.sin16 = (int)rint(rot_pars.sind * pow(2.0, 16));
        irt_desc.cosf  = (float)rot_pars.cosd;
        irt_desc.sinf  = (float)rot_pars.sind;
        irt_desc.cosi  = (int)rint(rot_pars.cosd * pow(2.0, rot_pars.TOTAL_PREC));
        irt_desc.sini  = (int)rint(rot_pars.sind * pow(2.0, rot_pars.TOTAL_PREC));

        // used for HL model w/o flip (with rot_angle and not rot_angle_adj)
        rot_pars.cosd_hl =
            irt_desc.rot90
                ? 0
                : cos(rot_pars.irt_angles[e_irt_angle_rot] * M_PI / 180.0); // fix C++ bug that cos(90) is not zero
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
    }

    // calculation input image read stripe parameters
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
            if ((rot_pars.mesh_mode != e_irt_mesh) &&
                (rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)) {
                irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
            }
            // irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
            break;
        case e_irt_resamp_fwd: irt_resamp_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_resamp_bwd1: irt_resamp_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_resamp_bwd2: irt_resamp_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_rescale: irt_rescale_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_BiLinearGrad: irt_BiLiGrad_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
    }

    // IRT_TRACE_UTILS("Desc %d: input stripe width Si = %d, sliding window height = %u\n", desc,
    // irt_desc.image_par[IIMAGE].S, rot_pars.IBufH_req);

    // calculating processing size and offsets

    if (rot_pars.proc_auto &&
        (irt_desc.rate_mode == e_irt_rate_fixed |
         irt_desc.irt_mode ==
             e_irt_rescale)) { // fixed rate mode set by proc_auto for projection/mesh does not work in all cases
        if ((irt_desc.irt_mode == e_irt_projection && rot_pars.proj_mode == e_irt_projection) ||
            (irt_desc.irt_mode == e_irt_mesh &&
             (rot_pars.mesh_mode == e_irt_projection || rot_pars.mesh_mode == e_irt_mesh ||
              rot_pars.mesh_dist_r0_Sh1_Sv1 == 0))) {
            IRT_TRACE_UTILS("Error: proc_auto = 1 is not supported in Mesh mode with provided parameters\n");
            IRT_TRACE_TO_RES_UTILS(test_res,
                                   "Error: proc_auto = 1 is not supported in Mesh mode with provided parameters\n");
            IRT_CLOSE_FAILED_TEST(0);
        }
        irt_desc.proc_size = irt_proc_size_calc(irt_cfg, rot_pars, irt_desc, desc);
    }

    switch (irt_desc.irt_mode) {
        case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_affine: irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
        case e_irt_projection:
            switch (rot_pars.proj_mode) {
                case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
                case e_irt_affine: irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
                default: break;
            }
            break;
        case e_irt_mesh:
            switch (rot_pars.mesh_mode) {
                case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
                case e_irt_affine: irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
                case e_irt_projection:
                    switch (rot_pars.proj_mode) {
                        case e_irt_rotation: irt_xi_first_adj_rot(irt_cfg, rot_pars, irt_desc, desc); break;
                        case e_irt_affine: irt_xi_first_adj_aff(irt_cfg, rot_pars, irt_desc, desc); break;
                        default: break;
                    }
                    break;
                default: break;
            }
        default: break;
    }

    if (irt_desc.irt_mode <= 3) {
        // calculating required rotation memory view
        // IBufW_req is pixels per input lines stored in memory, equal to Si round up to multiple of 128
        // IBufW_req = (int)(pow(2.0, ceil(log(((double)irt_desc.image_par[IIMAGE].S + 1) / 256) / log(2.0))) * 256);
        // IBufW_req = (int)(ceil(((float)irt_desc.image_par[IIMAGE].S + 1) / 128) * 128);
        rot_pars.IBufW_req =
            (uint32_t)(ceil(((float)irt_desc.image_par[IIMAGE].S +
                             (irt_desc.rot90 == 0 ? 1 + irt_desc.Xi_start_offset + irt_desc.Xi_start_offset_flip : 0)) /
                            128) *
                       128);
        // BufW_entries - entries per input image line
        rot_pars.IBufW_entries_req = rot_pars.IBufW_req / 16;
        // 2.0*IRT_ROT_MEM_HEIGHT1 = 256 - entries in bank
        // irt_desc.BufW_entries - entries per input image line
        // 2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries - # of input image lines in single bank
        // 2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8 - # of input image lines in all banks
        // log2(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8) - bits to present # of input image lines in all banks
        // floor(log(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8)/log(2.0)) - rounding down to integer value
        // pow(2.0,floor(log(2.0*IRT_ROT_MEM_HEIGHT1/irt_desc.BufW_entries*8)/log(2.0))) - makes BufH to be power of 2
        // to simplify buffer management BufH_available = (int)(floor(2.0 * IRT_ROT_MEM_HEIGHT1 /
        // irt_top->irt_cfg.BufW_entries) * 8); BufH_available - numbe_fitr of input lines that can be stored in
        // rotation memory if we store IBufW_req pixels per line uint16_t BufH_available = (uint16_t)(pow(2.0,
        // floor(log(2.0 * IRT_ROT_MEM_BANK_HEIGHT / rot_pars.IBufW_entries_req * 8))));

        rot_pars.IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, rot_pars.IBufH_req);

        irt_rotation_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);

        if (irt_desc.irt_mode == e_irt_mesh) {

            uint32_t Wo = irt_desc.image_par[OIMAGE].W;
            uint32_t So = irt_desc.image_par[OIMAGE].S;
            uint32_t Ho = irt_desc.image_par[OIMAGE].H;

            irt_mesh_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);
            if (Wo != irt_desc.image_par[OIMAGE].W || So != irt_desc.image_par[OIMAGE].S ||
                Ho != irt_desc.image_par[OIMAGE].H) {
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
                            case e_irt_projection: irt_projection_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
                            default: break;
                        }
                        break;
                    case e_irt_mesh: irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc); break;
                }
                if ((rot_pars.mesh_mode != e_irt_mesh) &&
                    (rot_pars.mesh_dist_r0_Sh1_Sv1 == 0 || rot_pars.mesh_matrix_error == 1)) {
                    irt_mesh_desc_gen(irt_cfg, rot_pars, irt_desc, desc);
                }

                // irt_rotation_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc); //used to update buf mode after
                // previous irt_rotation_memory_fit_check adjusted Wo
            }
        }

        if (irt_desc.rot90) {
            rot_pars.im_read_slope = 0;
            if (irt_desc.rot90_inth || irt_desc.rot90_intv) { // work as regular rotation because of interpolation
                // irt_desc.rot90 = 0;
                if (/*rot_pars.rot90_intv &&*/ irt_desc.rate_mode == e_irt_rate_fixed && irt_desc.proc_size == 8 &&
                    irt_desc.rot90_intv) { // vertical interpolation is required
                    irt_desc.proc_size = 7;
                }
            }
        }

        // irt_mesh_interp_matrix_calc(irt_cfg, rot_pars, irt_desc, desc);
        // rot_pars.IBufH_req = irt_IBufH_req_calc(irt_cfg, irt_desc, rot_pars.IBufH_req);

        if ((fabs(rot_pars.Xi_first) > (double)IRT_XiYi_first_max) ||
            (fabs(rot_pars.Xi_last) > (double)IRT_XiYi_first_max) ||
            (fabs(rot_pars.Yi_first) > (double)IRT_XiYi_first_max) ||
            (fabs(rot_pars.Yi_last) > (double)IRT_XiYi_first_max)) {
            IRT_TRACE_UTILS("Desc gen error: transformation parameters are not supported, input coordinates range "
                            "exceeds [-%d:%d]: Xi_first = %f, Xi_last = %f, Yi_first = %f, Yi_last = %f\n",
                            IRT_XiYi_first_max,
                            IRT_XiYi_first_max,
                            rot_pars.Xi_first,
                            rot_pars.Xi_last,
                            rot_pars.Yi_first,
                            rot_pars.Yi_last);
            IRT_TRACE_TO_RES_UTILS(test_res,
                                   "was not run, input coordinate range exceeds [-%d:%d]: Xi_first = %f, Xi_last = %f, "
                                   "Yi_first = %f, Yi_last = %f\n",
                                   IRT_XiYi_first_max,
                                   IRT_XiYi_first_max,
                                   rot_pars.Xi_first,
                                   rot_pars.Xi_last,
                                   rot_pars.Yi_first,
                                   rot_pars.Yi_last);
            IRT_CLOSE_FAILED_TEST(0);
        }

#ifdef IRT_USE_FLIP_FOR_MINUS1
        irt_desc.Xi_first_fixed = (int64_t)floor((rot_pars.Xi_first - irt_desc.read_hflip) * pow(2.0, IRT_SLOPE_PREC));
        irt_desc.Yi_first_fixed = (int64_t)rint((rot_pars.Yi_first - irt_desc.read_vflip) * pow(2.0, IRT_SLOPE_PREC));
#else
        irt_desc.Xi_first_fixed = (int64_t)floor((rot_pars.Xi_first) * pow(2.0, IRT_SLOPE_PREC));
        irt_desc.Yi_first_fixed = (int64_t)rint((rot_pars.Yi_first) * pow(2.0, IRT_SLOPE_PREC));
#endif

        // irt_desc.Yi_start = (int16_t)(irt_desc.Yi_first_fixed >> IRT_SLOPE_PREC) - irt_desc.read_vflip;
#ifdef IRT_USE_FLIP_FOR_MINUS1
        irt_desc.Yi_start = (int16_t)floor(rot_pars.Yi_first) - irt_desc.read_vflip;
#else
        irt_desc.Yi_start       = (int16_t)floor(rot_pars.Yi_first);
#endif

        irt_desc.Yi_end = (int16_t)ceil(rot_pars.Yi_last) + 1;
        if (irt_desc.bg_mode == e_irt_bg_frame_repeat) {
            // irt_desc.Yi_start = IRT_top::IRT_UTILS::irt_max_int16(irt_desc.Yi_start, 0);
            // irt_desc.Yi_start = IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_start,
            // (int16_t)irt_desc.image_par[IIMAGE].H - 1); irt_desc.Yi_end =
            // IRT_top::IRT_UTILS::irt_max_int16(irt_desc.Yi_end, 1); irt_desc.Yi_end =
            // IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_end, (int16_t)irt_desc.image_par[IIMAGE].H - 1);
        }
        // if (irt_desc.bg_mode == e_irt_bg_frame_repeat)
        //	irt_desc.Yi_end = IRT_top::IRT_UTILS::irt_min_int16(irt_desc.Yi_end, (int16_t)irt_desc.image_par[IIMAGE].H -
        // 1);

        // irt_desc.im_read_slope = (int)floor(rot_pars.im_read_slope * pow(2.0, rot_pars.COORD_PREC) + 0.5 * 1);
        irt_desc.im_read_slope = (int)rint(rot_pars.im_read_slope * pow(2.0, IRT_SLOPE_PREC));

        // if (irt_desc.irt_mode == 7 ?  (irt_cfg.buf_format[e_irt_block_mesh] == e_irt_buf_format_dynamic) :
        // (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic)) { //stripe height is multiple of 8
        //   if(irt_desc.irt_mode == 7) irt_desc.Yi_end = irt_desc.Yi_start + (uint16_t)(ceil(((double)irt_desc.Yi_end -
        //   irt_desc.Yi_start + 1) / IRT_MM_DYN_MODE_LINE_RELEASE) * IRT_MM_DYN_MODE_LINE_RELEASE) - 1; else
        //   irt_desc.Yi_end = irt_desc.Yi_start + (uint16_t)(ceil(((double)irt_desc.Yi_end - irt_desc.Yi_start + 1) /
        //   IRT_RM_DYN_MODE_LINE_RELEASE) * IRT_RM_DYN_MODE_LINE_RELEASE) - 1;
        //}
        if (irt_cfg.buf_format[e_irt_block_rot] == e_irt_buf_format_dynamic) { // stripe height is multiple of 8
            irt_desc.Yi_end =
                irt_desc.Yi_start +
                (uint16_t)(ceil(((double)irt_desc.Yi_end - irt_desc.Yi_start + 1) / IRT_RM_DYN_MODE_LINE_RELEASE) *
                           IRT_RM_DYN_MODE_LINE_RELEASE) -
                1;
        }
    } else {
        // irt_rotation_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);
        irt_desc.Xi_first_fixed = (int64_t)floor((rot_pars.Xi_first) * pow(2.0, IRT_SLOPE_PREC));
        irt_mesh_memory_fit_check(irt_cfg, rot_pars, irt_desc, desc);
    }
    //---------------------------------------
    /*
    if(irt_desc.irt_mode <= 3){
       irt_desc.image_par[OIMAGE].Ps = rot_pars.Pwo <= 8 ? 0 : 1;
    }else{
       switch(irt_desc.image_par[OIMAGE].DataType) {
          case	e_irt_int8   :	 irt_desc.image_par[OIMAGE].Ps = 0; break;
          case	e_irt_int16  :  irt_desc.image_par[OIMAGE].Ps = 1; break;
          case	e_irt_fp16   :  irt_desc.image_par[OIMAGE].Ps = 1; break;
          case	e_irt_bfp16  :  irt_desc.image_par[OIMAGE].Ps = 1; break;
          case	e_irt_fp32   :  irt_desc.image_par[OIMAGE].Ps = 2; break;
       }
    }

    if(irt_desc.irt_mode <= 3){
       irt_desc.image_par[IIMAGE].Ps = rot_pars.Pwi <= 8 ? 0 : 1;
    }else{
       IRT_TRACE("Sagar DATA_TYPE = %s *\n", irt_resamp_dtype_s[irt_desc.image_par[IIMAGE].DataType]);
       switch(irt_desc.image_par[IIMAGE].DataType) {
          case	e_irt_int8   :	 irt_desc.image_par[IIMAGE].Ps = 0; break;
          case	e_irt_int16  :  irt_desc.image_par[IIMAGE].Ps = 1; break;
          case	e_irt_fp16   :  irt_desc.image_par[IIMAGE].Ps = 1; break;
          case	e_irt_bfp16  :  irt_desc.image_par[IIMAGE].Ps = 1; break;
          case	e_irt_fp32   :  irt_desc.image_par[IIMAGE].Ps = 2; break;
       }
       IRT_TRACE("Sagar Ps = %d *\n", irt_desc.image_par[IIMAGE].Ps);
    }

    if(irt_desc.irt_mode <= 3){
       irt_desc.image_par[MIMAGE].Ps = irt_desc.mesh_format == e_irt_mesh_flex ? 2 : 3;
    } else{
       switch(irt_desc.image_par[MIMAGE].DataType) {
          case	e_irt_int8   :	 irt_desc.image_par[MIMAGE].Ps = 1; break;
          case	e_irt_int16  :  irt_desc.image_par[MIMAGE].Ps = 2; break;
          case	e_irt_fp16   :  irt_desc.image_par[MIMAGE].Ps = 2; break;
          case	e_irt_bfp16  :  irt_desc.image_par[MIMAGE].Ps = 2; break;
          case	e_irt_fp32   :  irt_desc.image_par[MIMAGE].Ps = 3; break;
       }
    }
    */
    //---------------------------------------
    if (irt_desc.irt_mode < 4) {
        irt_desc.image_par[IIMAGE].Hs = Hsi; // irt_desc.image_par[IIMAGE].W;
    } else {
        irt_desc.image_par[IIMAGE].Hs = irt_desc.image_par[IIMAGE].W
                                        << irt_desc.image_par[IIMAGE].Ps; // irt_desc.image_par[IIMAGE].W;
    }
    //---------------------------------------
    if (irt_desc.irt_mode < 4) {
        irt_desc.image_par[OIMAGE].Hs   = irt_desc.image_par[OIMAGE].S << irt_desc.image_par[OIMAGE].Ps;
        irt_desc.image_par[OIMAGE].Size = irt_desc.image_par[OIMAGE].S * irt_desc.image_par[OIMAGE].H;
    } else {
        if (irt_desc.irt_mode == 6) { // BWD-2
            irt_desc.image_par[OIMAGE].Hs =
                irt_desc.image_par[IIMAGE].W
                << irt_desc.image_par[IIMAGE].Ps; // TODO -- SJAGADALE - REVIEW WITH ARCH TEAM -- Use IIMAGE.W for O.Hs
            IRT_TRACE_UTILS("Sagar OIMAGE.Hs = %x *\n", irt_desc.image_par[OIMAGE].Hs);
        } else {
            irt_desc.image_par[OIMAGE].Hs = irt_desc.image_par[OIMAGE].W << irt_desc.image_par[OIMAGE].Ps;
        }
        irt_desc.image_par[OIMAGE].Size = irt_desc.image_par[OIMAGE].W * irt_desc.image_par[OIMAGE].H;
        //---------------------------------------
        // SET MESH POINT for warp type == INT16
        if ((irt_desc.irt_mode < 7) && (irt_desc.image_par[MIMAGE].DataType == e_irt_int16)) { // only for RESAMP mode
            int     M_max_int            = std::max(irt_desc.image_par[IIMAGE].W, irt_desc.image_par[IIMAGE].H);
            uint8_t M_I_width            = (uint8_t)ceil(log2(M_max_int + 1));
            uint8_t M_F_width            = 15 - M_I_width; // remained bits for fraction
            irt_desc.mesh_point_location = M_F_width;
        }
    }
    //---------------------------------------
    irt_desc.image_par[MIMAGE].Hs =
        (irt_desc.image_par[MIMAGE].W +
         ((irt_desc.irt_mode == e_irt_rescale) && (irt_desc.bg_mode == 2) ? (irt_desc.rescale_LUT_x->num_taps - 1) : 0))
        << irt_desc.image_par[MIMAGE].Ps;
    irt_desc.image_par[GIMAGE].Hs = irt_desc.image_par[GIMAGE].W << irt_desc.image_par[GIMAGE].Ps;
    //---------------------------------------
    if (irt_h5_mode || irt_desc.rot90 == 1) {
        // irt_desc.proc_size = 8;
    }

    if (irt_desc.irt_mode <= 3) {
        // checking Xi_start overflow
        rot_pars.Xi_start =
            IRT_top::xi_start_calc(irt_desc, irt_desc.Yi_end, e_irt_xi_start_calc_caller_desc_gen, desc);
        rot_pars.Xi_start =
            IRT_top::xi_start_calc(irt_desc, irt_desc.Yi_start, e_irt_xi_start_calc_caller_desc_gen, desc);
    }

    #ifdef DISABLE_IRT_TRACE_UTILS_PRINT
        // no print info
    #else
        print_descriptor(irt_cfg, rot_pars, irt_desc, file_descr_out);
    #endif
}

//======================resampler functions=========================
// Find maximum between two numbers.
int max(int num1, int num2)
{
    return (num1 > num2) ? num1 : num2;
}
// Find minimum between two numbers.
int min(int num1, int num2)
{
    return (num1 > num2) ? num2 : num1;
}
// three dimensional memory allocation.
float*** mycalloc3(int num_ch, int im_h, int im_w)
{
    int      k, l;
    float*** m3_out;

    // m3_out=(float ***)calloc(num_ch,sizeof(float **));
    m3_out = new float**[num_ch];
    for (k = 0; k < num_ch; k++) {
        // m3_out[k]=(float **)calloc(im_h,sizeof(float *));
        m3_out[k] = new float*[im_h];
        for (l = 0; l < im_h; l++) {
            m3_out[k][l] = new float[im_w];
            // m3_out[k][l]=(float *)calloc(im_w,sizeof(float));
        }
    }
    return (m3_out);
}

// return a uniformly distributed random number
double RandomGenerator()
{
    return ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.);
}

// return a normally distributed random number
double normalRandom()
{
    double y1 = RandomGenerator();
    double y2 = RandomGenerator();
    return cos(2 * 3.14 * y2) * sqrt(-2. * log(y1));
}
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
// Initilizing warp coordinates.
//---------------------------------------
// random1
// 0 -> input file feed
// 1 -> C based random warp coordinates
// 2 -> sigma*normalRandom() based warp image
// 3 -> Incremental warp coordinate - Debug mode for flow flush
// 4 -> Incremental x2 warp coordinate - avoids conficts
// 5 -> rotation warps
// 7 -> denorm warp random
// 8 all denorm warps
void InitWarp(uint8_t       task_idx,
              irt_desc_par& irt_desc,
              uint8_t       random1,
              uint8_t       verif,
              float         warpincr = 1.0) // warpincr is only for debug purpose
{
    // unsigned int numElementsPerBatch = m_warp.h * m_warp.w * m_warp.c;
    uint16_t               Wm             = irt_desc.image_par[MIMAGE].W; //(inputWidth - 1);
    uint16_t               Hm             = irt_desc.image_par[MIMAGE].H; //(inputHeight - 1);
    uint16_t               c              = 1;
    uint16_t               Wi             = irt_desc.image_par[IIMAGE].W; //(inputWidth - 1);
    uint16_t               Hi             = irt_desc.image_par[IIMAGE].H; //(inputHeight - 1);
    uint16_t               Psi            = irt_desc.image_par[IIMAGE].Ps; //(inputHeight - 1);
    double                 internalScale1 = max(Wi - 1, c) / (1.0 + (double)RAND_MAX);
    double                 internalScale2 = max(Hi - 1, c) / (1.0 + (double)RAND_MAX);
    uint64_t               wimage_addr    = (verif == 1) ? VERIF_MIMAGE_OFFSET : irt_desc.image_par[MIMAGE].addr_start;
    Eirt_resamp_dtype_enum warp_dtype     = irt_desc.image_par[MIMAGE].DataType;

    FILE *   ipFileX, *ipFileXf;
    FILE *   ipFileY, *ipFileYf;
    double   min_rx, min_ry, max_rx, max_ry;
    int32_t  wc_int32;
    uint32_t wc_uint32;
    uint8_t  byte_per_coord = 1 << (irt_desc.image_par[MIMAGE].Ps - 1); //(inputHeight - 1);
    char     fileName[50];

    // resize bli grad cfg
    uint16_t resize_bli_grad_en = irt_desc.resize_bli_grad_en;
    uint16_t relative_mode_en   = irt_desc.mesh_rel_mode;
    float    gx                 = irt_desc.M11f;
    float    gy                 = irt_desc.M22f;
    float    Xc_o               = ((float)irt_desc.image_par[OIMAGE].Xc) / 2;
    float    Yc_o               = ((float)irt_desc.image_par[OIMAGE].Yc) / 2;
    float    Xc_i               = ((float)irt_desc.image_par[IIMAGE].Xc) / 2;
    float    Yc_i               = ((float)irt_desc.image_par[IIMAGE].Yc) / 2;
    uint8_t  bw2_en             = (irt_desc.irt_mode == e_irt_resamp_bwd2) ? 1 : 0;
    // end of resize bli grad cfg
    // m_warp_data = mycalloc3(BATCH_SIZE, irt_desc.image_par[MIMAGE].H, irt_desc.image_par[MIMAGE].W, 2);
    IRT_TRACE_TO_RES_UTILS_LOG(10,
                               test_res,
                               "InitWarp :: warp_gen_mode = %d internalScale %lf:%lf Wi:Hi %d:%d RAND_MAX %d",
                               random1,
                               internalScale1,
                               internalScale2,
                               Wi,
                               Hi,
                               RAND_MAX);
    //   IRT_TRACE_TO_RES_UTILS(fptr,"WARP_GEN::index :: [Yf:Xf]  :: [Yh:Xh] ");
    //---------------------------------------
    if (random1 == 0) {
        if ((ipFileX = fopen(file_warp_x, "r")) == NULL) {
            printf("Error: can't open warp x input file ");
            exit(1);
        }
        if ((ipFileY = fopen(file_warp_y, "r")) == NULL) {
            printf("Error: can't open warp y input file ");
            exit(1);
        }
    }
    if (random1 != 0) {
        // printf("open test input file\n");
        sprintf(fileName, "warp_x_task%d.txt", task_idx);
        if ((ipFileX = fopen(fileName, "w")) == NULL) {
            printf("Error: can't open warp x input file ");
            exit(1);
        }
        sprintf(fileName, "warp_y_task%d.txt", task_idx);
        if ((ipFileY = fopen(fileName, "w")) == NULL) {
            printf("Error: can't open warp y input file ");
            exit(1);
        }
        sprintf(fileName, "warp_xf_task%d.txt", task_idx);
        if ((ipFileXf = fopen(fileName, "w")) == NULL) {
            printf("Error: can't open warp x input file ");
            exit(1);
        }
        sprintf(fileName, "warp_yf_task%d.txt", task_idx);
        if ((ipFileYf = fopen(fileName, "w")) == NULL) {
            printf("Error: can't open warp y input file ");
            exit(1);
        }
    }
    //---------------------------------------
#if !defined(RUN_WITH_SV)
    // srand((unsigned) time(0));
    srand(cseed);
#endif

    //---------------------------------------
    for (int j = 0; j < Hm; j++) {
        for (int k = 0; k < Wm; k++) {
            if (random1 == 1) {
                double x = (double)rand() * internalScale1;
                double y = (double)rand() * internalScale2;
                // if((((((int)floor(x)+1)<<Psi)) % 128) == 0) x=x+1;
                m_warp_data[j][k][0] = (float)y;
                m_warp_data[j][k][1] = (float)x;
            } else if (random1 == 0) {
                // fscanf(ipFileY,"%f ",&m_warp_data[j][k][0]);
                // fscanf(ipFileX,"%f ",&m_warp_data[j][k][1]);
                // IRT_TRACE_UTILS("READING FILE [%d][%d]\n",j,k);
                fscanf(ipFileY, "%x ", &wc_uint32);
                m_warp_data[j][k][0] = IRT_top::IRT_UTILS::conversion_bit_float(wc_uint32, e_irt_fp32, 0);
                // IRT_TRACE_UTILS("%f:%x  ",m_warp_data[j][k][0],wc_uint32);
                fscanf(ipFileX, "%x ", &wc_uint32);
                m_warp_data[j][k][1] = IRT_top::IRT_UTILS::conversion_bit_float(wc_uint32, e_irt_fp32, 0);
                // IRT_TRACE_UTILS("%f:%x  ",m_warp_data[j][k][1],wc_uint32);
            } else if (random1 == 3) {
                double x             = (double)(k * warpincr) /*+ 0.5*/;
                double y             = (double)(j * warpincr) /*+ 0.5*/;
                m_warp_data[j][k][0] = (float)y;
                m_warp_data[j][k][1] = (float)x;
            } else {
                break;
            }
        }
    }

    if (random1 == 0) {
        fclose(ipFileX);
        fclose(ipFileY);
    }

    double sigma = (double)range_bound;
    // double Mi = 40.;
    if (random1 == 2 || random1 == 7) {
#if !defined(RUN_WITH_SV)
        // srand((unsigned) time(0));
        srand(cseed);
#endif

        for (int j = 0; j < Hm; j++) {
            for (int k = 0; k < Wm; k++) {
                double x1i = sigma * normalRandom() + k;
                double y1i = sigma * normalRandom() + j;
                //---------------------------------------
                m_warp_data[j][k][0] = (float)y1i;
                m_warp_data[j][k][1] = (float)x1i;
            }
        }
    }
    //---------------------------------------
    // generate warp as rotation
    if (random1 == 5) {
        // float rot_angle = (rand() % 45) + 1 ; //rotation angle upto 90
        float rot_angle = 0; //(rand() % 10) + 1 ; //rotation angle upto 90
        float cosf      = (float)(cos(rot_angle * M_PI / 180.0));
        float sinf      = (float)(sin(rot_angle * M_PI / 180.0));
        for (int j = 0; j < Hm; j++) {
            for (int k = 0; k < Wm; k++) {
                double x1i = (cosf * k) + (sinf * j);
                double y1i = (-sinf * k) + (cosf * j);
                //---------------------------------------
                m_warp_data[j][k][0] = (float)y1i;
                m_warp_data[j][k][1] = (float)x1i;
            }
        }
    }
    //---------------------------------------
    if (random1 != 0) {
        for (int j = 0; j < Hm; j++) {
            for (int k = 0; k < Wm; k++) {
                IRT_TRACE_TO_RES_UTILS_LOG(10, ipFileYf, "%f\n", m_warp_data[j][k][0]);
                IRT_TRACE_TO_RES_UTILS_LOG(10, ipFileXf, "%f\n", m_warp_data[j][k][1]);
                IRT_TRACE_TO_RES_UTILS_LOG(
                    10,
                    ipFileY,
                    "%x\n",
                    IRT_top::IRT_UTILS::conversion_float_bit(m_warp_data[j][k][0], e_irt_fp32, 0, 0, 0, 0, 0, 0, 0));
                IRT_TRACE_TO_RES_UTILS_LOG(
                    10,
                    ipFileX,
                    "%x\n",
                    IRT_top::IRT_UTILS::conversion_float_bit(m_warp_data[j][k][1], e_irt_fp32, 0, 0, 0, 0, 0, 0, 0));
            }
        }
        fclose(ipFileXf);
        fclose(ipFileYf);
    }

    //===========================================
    // Warp coord gen block : This is applicable
    // only when resize bli grad enable is there and
    // it will used only in HL model.
    // similarly LL model we generate internally.
    //===========================================
    if ((resize_bli_grad_en == 1) & (bw2_en == 1)) {
        for (int j = 0; j < Hm; j++) {
            for (int k = 0; k < Wm; k = k + 4) {
                float Xo                 = ((float)k) - Xc_o;
                float Yo                 = ((float)j) - Yc_o;
                m_warp_data[j][k][1]     = (gx * Xo) + Xc_i;
                m_warp_data[j][k + 1][1] = m_warp_data[j][k][1] + gx;
                m_warp_data[j][k + 2][1] = m_warp_data[j][k][1] + (2 * gx);
                m_warp_data[j][k + 3][1] = m_warp_data[j][k][1] + (3 * gx);

                m_warp_data[j][k][0]     = (gy * Yo) + Yc_i;
                m_warp_data[j][k + 1][0] = m_warp_data[j][k][0];
                m_warp_data[j][k + 2][0] = m_warp_data[j][k][0];
                m_warp_data[j][k + 3][0] = m_warp_data[j][k][0];

                // printf("dddddddddddddd warp print %f %f %f %f\n",
                // m_warp_data[j][k][1],m_warp_data[j][k+1][1],m_warp_data[j][k+2][1],m_warp_data[j][k+3][1]);

                // for(int ii=0;ii<4;ii++){
                //  printf("warp aaaaa gx:%f gy:%f Xo:%f, Yo:%f, (%f,%f) ",gx, gy, Xo,Yo,Xi[(i*4)+ii],Yi[(i*4)+ii]);
                //}
                // printf("\n");
            }
        }
    }
    //===========================================
    // end of warp coord gen block
    //===========================================
    //---------------------------------------
    // preload ext_mem and generate warp image file
    sprintf(fileName, "warp_image_task%d.txt", task_idx);
    FILE* fptr = fopen(fileName, "w");
    IRT_TRACE_TO_RES_UTILS_LOG(10, fptr, "WARP_GEN::data_dtype = %d range_bound %d \n", warp_dtype, range_bound);
    IRT_TRACE_TO_RES_UTILS_LOG(
        10, fptr, "----------------------------------------------------------------------------------------------\n");
    IRT_TRACE_TO_RES_UTILS_LOG(10, fptr, "pix-index \t\t\t::\t\t\t [Yf:Xf] \t\t\t::\t\t\t Xint \t\t\t::\t\t\t Yint\n");
    IRT_TRACE_TO_RES_UTILS_LOG(
        10, fptr, "----------------------------------------------------------------------------------------------\n");
    for (int j = 0; j < Hm; j++) {
        for (int k = 0; k < Wm; k++) {
            IRT_TRACE_TO_RES_UTILS_LOG(10,
                                       fptr,
                                       "[%d:%d] \t\t\t::\t\t\t [%f:%f] \t\t\t::\t\t\t ",
                                       j,
                                       k,
                                       m_warp_data[j][k][0],
                                       m_warp_data[j][k][1]);
            for (uint8_t coord = 2; coord > 0; coord--) {
                // IMP NOTE -- random float MUST be re-converted into matching precision float based on IIMAGE.dtype ..
                // to avoid HL - LL output precision issue
                if (irt_desc.image_par[MIMAGE].DataType == e_irt_int16) {
                    //(int16_t)rint(((double)m_warp_data[j][k][coord-1]) * pow(2.0, irt_desc.mesh_point_location));
                    wc_int32 = (int32_t)IRT_top::IRT_UTILS::conversion_float_fxd16(m_warp_data[j][k][coord - 1],
                                                                                   irt_desc.mesh_point_location);
                    // IRT_top::IRT_UTILS::conversion_float_bit(m_warp_data[j][k][coord-1], warp_dtype, 0, 0, 0,0,0, 0);
                    // //TODO -- WARP IMAGE INT8/16 SUPPORT FIX
                } else {
                    wc_int32 = IRT_top::IRT_UTILS::conversion_float_bit(m_warp_data[j][k][coord - 1],
                                                                        warp_dtype,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0,
                                                                        0); // TODO -- WARP IMAGE INT8/16 SUPPORT FIX
                }
                // float tmp = ((resize_bli_grad_en==1)&(bw2_en==1)) ? m_warp_data[j][k][coord-1]
                // :IRT_top::IRT_UTILS::conversion_bit_float(wc_int32, warp_dtype,0,irt_desc.mesh_point_location);
                //---------------------------------------
                if (random1 == 7) {
                    uint32_t dt_mask = (warp_dtype == 2) ? ROT_FP16_EXP_MASK
                                                         : ((warp_dtype == 3) ? ROT_BFP16_EXP_MASK : ROT_FP32_EXP_MASK);
                    if (rand() % 100 > 50)
                        wc_int32 &= ~dt_mask;
                }

                float tmp;
                if ((resize_bli_grad_en == 1) & (bw2_en == 1)) {
                    tmp = m_warp_data[j][k][coord - 1];
                } else {
                    if (warp_dtype == e_irt_int16)
                        tmp = IRT_top::IRT_UTILS::conversion_fxd16_float(wc_int32, irt_desc.mesh_point_location, 0);
                    else
                        tmp = IRT_top::IRT_UTILS::conversion_bit_float(wc_int32, warp_dtype, 0);
                }
                // TODO - assertion to ensure warp datatype should be fp32 or ignored in rtl
                m_warp_data[j][k][coord - 1] = tmp;
                IRT_TRACE_TO_RES_UTILS_LOG(10, fptr, "[%x][%f] \t\t\t::\t\t\t  ", wc_int32, tmp);
                // IRT_TRACE_UTILS("[%x][%f] \t\t\t::\t\t\t  ", wc_int32, tmp);
#if !defined(RUN_WITH_SV) // ROTSIM_DV_INTGR
                for (uint8_t byte = 0; byte < byte_per_coord; byte++) {
                    ext_mem[wimage_addr] = (wc_int32 >> (8 * byte)) & 0xff;
                    // IRT_TRACE_TO_RES_UTILS_LOG(10,fptr,"ext_mem[%x] = %x ",wimage_addr, ext_mem[wimage_addr]);
                    wimage_addr++;
                }
#endif
            }
            IRT_TRACE_TO_RES_UTILS_LOG(10, fptr, "\n");
        }
    }
    fclose(fptr);
    printf("  InitWarp -- Done \n");
    //---------------------------------------
    // DUMP WARP IMAGE FOR SIVAL FORMAT
#ifndef RUN_WITH_SV
    FILE*    f = nullptr;
    char     fname1[50];
    uint64_t coord_pair;

    sprintf(fname1, "sival_warp_image_task%d.txt", task_idx);

    f = fopen(fname1, "w");

    fprintf(f, "// IMAGE DIMs H x W : %0d x %0d  \t data-type : %0d \n", Hm, Wm, warp_dtype);
    for (uint32_t i = 0; i < Hm; i++) {
        for (uint32_t j = 0; j < Wm; j++) {
            if (irt_desc.image_par[MIMAGE].DataType == e_irt_int16) {
                uint32_t w_x = (int32_t)IRT_top::IRT_UTILS::conversion_float_fxd16(m_warp_data[i][j][0],
                                                                                   irt_desc.mesh_point_location);
                uint32_t w_y = (int32_t)IRT_top::IRT_UTILS::conversion_float_fxd16(m_warp_data[i][j][1],
                                                                                   irt_desc.mesh_point_location);
                fprintf(f, "%04x\n", (w_x << 16) | (w_y & 0xFFFF));
            } else {
                fprintf(f,
                        "%04x%04x\n",
                        (uint32_t)IRT_top::IRT_UTILS::conversion_float_bit(
                            m_warp_data[i][j][0], warp_dtype, 0, 0, 0, 0, 0, 0, 0),
                        (uint32_t)IRT_top::IRT_UTILS::conversion_float_bit(
                            m_warp_data[i][j][1], warp_dtype, 0, 0, 0, 0, 0, 0, 0));
            }
        }
    }
    fclose(f);
#endif
}
#endif

#define FP16_MAX_VAL 65502
//#define FP32_MAX_VAL 338953139000000000000000000000000000000
#define FP32_MAX_VAL (3.4028235 * (pow(10, 38)))
// template < Eirt_blocks_enum block_type >
// num_range - 0 [image DIMs range]    1 [FUll number system range]  2 [-1 to +1 range - ML- training workload range]
// mode - 0 [no denorm] 1 [ rand denorm] 2 [all denorm]
#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
void InitImage(uint8_t          task_idx,
               irt_desc_par&    irt_desc,
               rotation_par&    rot_par,
               uint8_t          mode,
               uint8_t          verif,
               Eirt_blocks_enum block_type,
               uint8_t          num_range)
{
    // unsigned int numElementsPerBatch = m_warp.h * m_warp.w * m_warp.c;
    //---------------------------------------
    uint16_t Hi = (irt_desc.irt_mode == e_irt_rescale)
                      ? irt_desc.image_par[MIMAGE].H
                      : ((block_type == e_irt_grm) ? irt_desc.image_par[GIMAGE].H : irt_desc.image_par[IIMAGE].H);
    uint16_t Wi = (irt_desc.irt_mode == e_irt_rescale)
                      ? irt_desc.image_par[MIMAGE].W
                      : ((block_type == e_irt_grm) ? irt_desc.image_par[GIMAGE].W : irt_desc.image_par[IIMAGE].W);
    uint64_t image_addr = (irt_desc.irt_mode == e_irt_rescale)
                              ? irt_desc.image_par[MIMAGE].addr_start
                              : ((block_type == e_irt_grm)
                                     ? ((verif == 1) ? VERIF_GIMAGE_OFFSET : irt_desc.image_par[GIMAGE].addr_start)
                                     : ((verif == 1) ? VERIF_IIMAGE_OFFSET : irt_desc.image_par[IIMAGE].addr_start));
    uint8_t PsByte = (irt_desc.irt_mode == e_irt_rescale)
                         ? irt_desc.image_par[MIMAGE].Ps
                         : ((block_type == e_irt_grm) ? (1 << (irt_desc.image_par[GIMAGE].Ps))
                                                      : (1 << (irt_desc.image_par[IIMAGE].Ps))); //(inputHeight - 1);
    uint                   PsByte_lp = (irt_desc.irt_mode == e_irt_rescale) ? PsByte + 1 : PsByte;
    Eirt_resamp_dtype_enum dtype =
        (irt_desc.irt_mode == e_irt_rescale)
            ? irt_desc.image_par[MIMAGE].DataType
            : ((block_type == e_irt_grm) ? irt_desc.image_par[GIMAGE].DataType : irt_desc.image_par[IIMAGE].DataType);
    // Eirt_resamp_dtype_enum dtype  = irt_desc.image_par[IIMAGE].DataType;
    float*** im_data = (block_type == e_irt_grm) ? output_im_grad_data : input_im_data;
    char     fileName[50];
    int      Pwi       = rot_par.Pwi;
    int16_t  bli_shift = irt_desc.bli_shift;
    uint16_t MAX_VALo  = irt_desc.MAX_VALo;
    uint16_t Ppo       = irt_desc.Ppo;
    //---------------------------------------
    const double nsystem_range = (dtype == e_irt_fp16)
                                     ? FP16_MAX_VAL
                                     : FP32_MAX_VAL; // generate numbers within full range of given number system
    const int    ML_training_range = 1; // generate numbers inside -1 to +1
    const double a                 = (dtype == e_irt_int8)
                         ? pow(2, 8)
                         : ((dtype == e_irt_int16)
                                ? pow(2, 16)
                                : (num_range == 0) ? Wi - 1 : (num_range == 1) ? 1 : nsystem_range); //(inputWidth - 1);
    const double b =
        (dtype == e_irt_int8)
            ? pow(2, 8)
            : ((dtype == e_irt_int16)
                   ? pow(2, 16)
                   : (num_range == 0) ? Hi - 1 : (num_range == 1) ? 1 : nsystem_range); //(inputHeight - 1);
    // const double c                   = PLANES;
    int32_t wc_int32;
    double  internalScale1 = fmax(a, b) / (1.0 + (double)RAND_MAX);
    // double    internalScale2      = fmax(b, c) / (1.0 + (double)RAND_MAX);
    double min_rx, min_ry, max_rx, max_ry;
    //---------------------------------------
    // input_im.data = mycalloc4(BATCH_SIZE, input_im.c, input_im.h, input_im.w);
    if (block_type == e_irt_grm) {
        sprintf(fileName, "grad_image_task%d.txt", task_idx);
    } else {
        sprintf(fileName, "input_image_task%d.txt", task_idx);
    }
    FILE* fptr = fopen(fileName, "w");
    //---------------------------------------
#if !defined(RUN_WITH_SV)
    // srand((unsigned)time(0));
    srand(cseed);
#endif
    IRT_TRACE_TO_RES_UTILS_LOG(
        22,
        fptr,
        "Generating IMage as [Hi=%d:Wi=%d:PsByte=%d:Pwi=%d:dtype=%d] internalScale1 %f mode %d\n",
        Hi,
        Wi,
        PsByte,
        Pwi,
        dtype,
        internalScale1,
        mode);
    // for(int ch=0; ch < PLANES; ch++)
    for (int ch = 0; ch < 1; ch++) // H9 ONLY used 1 plane
    {
        for (int j = 0; j < Hi; j++) {
            for (int k = 0; k < Wi; k++) {
                double x = (double)rand() * internalScale1;
                // IRT_TRACE_TO_RES_UTILS_LOG(20,fptr,"x = %6.64f ",x);
                // if(dtype == e_irt_int8 || dtype == e_irt_int16) x *= internalScale1;
                im_data[ch][j][k] = (float)x;
                //---------------------------------------
                // convert to appropriate bit format from float
                // IMP NOTE -- random float MUST be re-converted into matching precision float based on IIMAGE.dtype ..
                // to avoid HL - LL output precision issue
                if (dtype == e_irt_int8 || dtype == e_irt_int16) {
                    wc_int32 = round(x);
                } else {
                    wc_int32 = IRT_top::IRT_UTILS::conversion_float_bit(
                        im_data[ch][j][k], dtype, bli_shift, MAX_VALo, Ppo, 0, 0, 0, 0);
                }
                if (dtype == 2 && ((wc_int32 & 0x7c00 == 0x7c00)))
                    wc_int32 &= 0xF7FF; // flip 1 bit to aviod nan/inf
                if (dtype == 3 && ((wc_int32 & 0x7F80 == 0x7F80)))
                    wc_int32 &= 0xF7FF; // flip 1 bit to aviod nan/inf
                //---------------------------------------
                if (mode > 0) {
                    uint32_t dt_mask =
                        (dtype == 2) ? ROT_FP16_EXP_MASK : ((dtype == 3) ? ROT_BFP16_EXP_MASK : ROT_FP32_EXP_MASK);
                    if (mode == 2) { // all denorms
                        wc_int32 &= ~dt_mask;
                    } else if (rand() % 100 > 50) { // 50% chance of denorm numbers
                        wc_int32 &= ~dt_mask;
                    }
                }
                //---------------------------------------

                float tmp         = IRT_top::IRT_UTILS::conversion_bit_float(wc_int32, dtype, 0);
                im_data[ch][j][k] = tmp; // IRT_top::IRT_UTILS::conversion_bit_float(wc_int32, dtype);
                // memcpy(&wc_int32, &im_data[ch][j][k], sizeof(float));
                IRT_TRACE_TO_RES_UTILS_LOG(22,
                                           fptr,
                                           "im_data[%d:%d:%d] = Float[%f] Hex[%x] addr=%x FP32 [%0x]",
                                           ch,
                                           j,
                                           k,
                                           im_data[ch][j][k],
                                           wc_int32,
                                           image_addr,
                                           IRT_top::IRT_UTILS::irt_float_to_fp32(tmp));
                // IRT_TRACE_TO_RES_UTILS_LOG(11,fptr," :: x = %f :: wc_int32 = %x : %f ",x,wc_int32,tmp);
                // IRT_TRACE_TO_RES_UTILS_LOG(11,fptr,":: INPUT IMAGE ",ch,j,k,im_data[ch][j][k],x,wc_int32,tmp);
#if !defined(RUN_WITH_SV) // ROTSIM_DV_INTGR
                for (uint8_t byte = 0; byte < PsByte_lp; byte++) {
                    ext_mem[image_addr] = (uint8_t)((wc_int32 >> (8 * byte)) & 0xff);
                    IRT_TRACE_TO_RES_UTILS_LOG(
                        24, fptr, "byte=%d ext_mem[%d] = %x ", byte, image_addr, ext_mem[image_addr]);
                    image_addr++;
                }
#endif
                IRT_TRACE_TO_RES_UTILS_LOG(22, fptr, "\n");
#if defined(RUN_WITH_SV) // ROTSIM_DV_INTGR
                image_addr += PsByte;
#endif
                //---------------------------------------
            }
        }
    }
    fclose(fptr);
    printf("  InitImage -- Done \n");
    //---------------------------------------
    // SIVAL IMAGE DUMP
#ifndef RUN_WITH_SV
    if (block_type == e_irt_grm) {
        sprintf(fileName, "sival_grad_image_task%d.txt", task_idx);
    } else {
        sprintf(fileName, "sival_input_image_task%d.txt", task_idx);
    }

    generate_image_dump_per_datatype(fileName, Wi, Hi, im_data, dtype);
#endif
    //---------------------------------------
}
#endif

#if defined(STANDALONE_ROTATOR) || defined(RUN_WITH_SV)
void InitGraddata(irt_desc_par& irt_desc, uint8_t random1)
{
    uint16_t               Ho    = irt_desc.image_par[OIMAGE].H;
    uint16_t               Wo    = irt_desc.image_par[OIMAGE].W;
    Eirt_resamp_dtype_enum dtype = irt_desc.image_par[GIMAGE].DataType;
    //---------------------------------------
    const int a              = Wo - 1; //(inputWidth - 1);
    const int b              = Ho - 1; //(inputHeight - 1);
    const int c              = PLANES;
    double    internalScale1 = max(a, c) / (1.0 + (double)RAND_MAX);
    double    internalScale2 = max(b, c) / (1.0 + (double)RAND_MAX);
    double    min_rx, min_ry, max_rx, max_ry;

    // output_im_grad_data = mycalloc4(BATCH_SIZE, input_im.c, m_warp.h, m_warp.w);
#if !defined(RUN_WITH_SV)
    // srand((unsigned)time(0));
    srand(cseed);
#endif
    for (int ch = 0; ch < PLANES; ch++) {
        for (int j = 0; j < Ho; j++) {
            for (int k = 0; k < Wo; k++) {
                double x                   = (double)rand() * internalScale1;
                m_warp_grad_data[ch][j][k] = (float)x;
            }
        }
    }
}
#endif

void rescale_coeff_gen(Coeff_LUT_t* myLUT,
                       int          dec_factor,
                       int          interpol_factor,
                       int          filter_type,
                       int          max_phases,
                       int          lanczos_prec,
                       int16_t      op_xc,
                       int16_t      ip_xc)
{

    //---------------------------------------
    // TODO  - convert num_phases in the form simple ratio
    // int num_phases          = min(interpol_factor,max_phases);//TODO - Fix Independent X & Y num_Phases generation
    // from utility + Support in rtl
    int num_phases = max_phases;
    //---------------------------------------
    int phase_prec    = ceil(log2(num_phases));
    myLUT->num_phases = 1 << phase_prec;
    IRT_TRACE_UTILS("Inside rescale_coeff_gen dec_factor %d interpol_factor %d op center %d and ip center %d\n",
                    dec_factor,
                    interpol_factor,
                    op_xc,
                    ip_xc);
    float N = (filter_type == e_irt_lanczos3) ? 3 : 2; // TODO - REVIEW with Arc
    // float nudge =0.5;
    float Gx                = (float)dec_factor / (float)interpol_factor;
    float Sx                = (float)interpol_factor / (float)dec_factor;
    int   window            = ceil(max(N * Gx, N));
    myLUT->num_taps         = (filter_type == e_irt_bicubic) ? 4 : 2 * window;
    myLUT->Gf               = Gx;
    int   fixed_prec_factor = 1 << 24;
    int   G_fxd_pt_int      = round(Gx * fixed_prec_factor); // TODO - FLOAT TO HEX CONVERSION -
    float G_fx_pt           = (float)G_fxd_pt_int / fixed_prec_factor;
    float xi                = (float)ip_xc - (Gx * ((float)op_xc));
    float xi_inc            = (1 / (float)num_phases);
    // float xi_inc = (Gx);
    IRT_TRACE_UTILS("rescale_coeff_gen:filter=%d phases=%d phase_prec=%d taps %d Gx=%f  Sx=%f window=%d xi=%f "
                    "xi_inc=%f fxd_pt Gx=%f\n",
                    filter_type,
                    num_phases,
                    phase_prec,
                    myLUT->num_taps,
                    myLUT->Gf,
                    Sx,
                    window,
                    xi,
                    xi_inc,
                    G_fx_pt);

    for (int xo = 0; xo < (num_phases); xo++) {
        // float xi = (Gx*(xo+nudge))-nudge;
        int          left        = floor(xi) - (window - 1);
        int          right       = floor(xi) + window;
        int          tap         = 0;
        float        sumlanczos  = 0;
        unsigned int phase_index = rint((xi - floor(xi)) * (1 << phase_prec));
        // IRT_TRACE_UTILS("phase_index=%d left:right=%d:%d\n",phase_index,left,right);
        for (int y = left; y <= right; y++) {
            float Lancostaps = ((float)(((float)y - xi)) * fminf(1, Sx));
            // IRT_TRACE_UTILS("Lancostaps=%f y=%d xi=%f Sx=%f\n",Lancostaps,y,xi,Sx);
            if (filter_type == e_irt_lanczos2) { // Filter Type 0 -Lanczos2
                // myLUT->Coeff_table[phase_index,tap] =   lanczos(Lancostaps,2);
                float quant_Lancostaps = round(fabs(Lancostaps) * (1 << lanczos_prec));
                if (quant_Lancostaps >= 512) {
                    myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] = 0;
                } else {
                    myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] = lanczos2_array
                        [(unsigned int)
                             quant_Lancostaps]; // sagar
                                                // IRT_TRACE_UTILS("Coeff_table[%d]=lanczos2_array[%d]=%f:%f\n",phase_index*myLUT->num_taps+tap,(unsigned
                                                // int)quant_Lancostaps,lanczos2_array[(unsigned
                                                // int)quant_Lancostaps],myLUT->Coeff_table[phase_index*myLUT->num_taps+tap]);
                }
            } else {
                // TODO Add Bicubic
                myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] = 0;
            }
            sumlanczos = sumlanczos + myLUT->Coeff_table[phase_index * myLUT->num_taps + tap];
            tap++;
        }
        // Normalising filter coefficients
        tap = 0;
        for (int y = left; y <= right; y++) {
            myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] =
                myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] / sumlanczos;
            if (fabs(myLUT->Coeff_table[phase_index * myLUT->num_taps + tap]) < MIN_NORM)
                myLUT->Coeff_table[phase_index * myLUT->num_taps + tap] = 0;
            // IRT_TRACE_UTILS("Coeff_table[%d:%d]==%6.32f\n",phase_index,tap,myLUT->Coeff_table[phase_index*myLUT->num_taps+tap]);
            tap++;
        }
        xi = xi + xi_inc;
    }
    // return;
}

void print_filter_coeff(float* lanczos_array, int rows, int cols, char filename[100])
{
    FILE* fptr = fopen(filename, "w");
    fprintf(fptr, "num_taps %d num_phases %d\n", cols, rows);
    fprintf(fptr, "row:col \t\t::\t\t lanczos_array[index] \t\t::\t\t value\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fptr,
                    "[%d][%d] \t\t::\t\t\t\t [%d] \t\t\t\t::\t\t %f FP32 %x\n",
                    i,
                    j,
                    i * cols + j,
                    lanczos_array[i * cols + j],
                    IRT_top::IRT_UTILS::irt_float_to_fp32(lanczos_array[i * cols + j]));
            // fprintf(fptr,"[%d][%d] \t\t::\t\t\t\t [%d] \t\t\t\t::\t\t %f\n",i,j,i*cols + j,lanczos_array[i*cols +
            // j]);
        }
    }
    fclose(fptr);
}

void print_descriptor(irt_cfg_pars& irt_cfg, rotation_par& rot_pars, irt_desc_par& descFullData, const char* filename)
{

    // print_hlp_descriptors(irt_cfg,rot_pars, descFullData,filename);

    FILE* fp;
    fp = fopen(filename, "w");
    if (fp != NULL) {
        fprintf(fp, "********************H6 CMD BUF creator ***************************************************** \n");
        fprintf(fp, "irt_desc_par dump \n");
        fprintf(fp, "/*************************************************************************/ \n");
        fprintf(fp, "oimage_line_wr_format=%x\n", descFullData.oimage_line_wr_format);
        fprintf(fp, "bg=%x\n", descFullData.bg);
        fprintf(fp, "int_mode=%x\n", descFullData.int_mode);
        fprintf(fp, "bg_mode=%x\n", descFullData.bg_mode);
        fprintf(fp, "irt_mode=%x\n", descFullData.irt_mode);
        fprintf(fp, "rate_mode=%x\n", descFullData.rate_mode);
        fprintf(fp, "crd_mode=%x\n", descFullData.crd_mode);
        fprintf(fp, "proc_size=%x\n", descFullData.proc_size);
        fprintf(fp, "IIMAGE addr_start= %ld\n", descFullData.image_par[IIMAGE].addr_start);
        fprintf(fp, "IIMAGE S=%x\n", descFullData.image_par[IIMAGE].S);
        fprintf(fp, "IIMAGE W=%x\n", descFullData.image_par[IIMAGE].W);
        fprintf(fp, "IIMAGE H=%x\n", descFullData.image_par[IIMAGE].H);
        fprintf(fp, "IIMAGE Xc=%x\n", descFullData.image_par[IIMAGE].Xc);
        fprintf(fp, "IIMAGE Yc=%x\n", descFullData.image_par[IIMAGE].Yc);
        fprintf(fp, "IIMAGE Hs=%x\n", descFullData.image_par[IIMAGE].Hs);
        fprintf(fp, "IIMAGE DataType=%x\n", descFullData.image_par[IIMAGE].DataType);
        fprintf(fp, "IIMAGE addr_end = %ld\n", descFullData.image_par[IIMAGE].addr_end);
        fprintf(fp, "OIMAGE addr_start=%ld\n", descFullData.image_par[OIMAGE].addr_start);
        fprintf(fp, "OIMAGE S=%x\n", descFullData.image_par[OIMAGE].S);
        fprintf(fp, "OIMAGE W=%x\n", descFullData.image_par[OIMAGE].W);
        fprintf(fp, "OIMAGE H=%x\n", descFullData.image_par[OIMAGE].H);
        fprintf(fp, "OIMAGE Xc=%x\n", descFullData.image_par[OIMAGE].Xc);
        fprintf(fp, "OIMAGE Yc=%x\n", descFullData.image_par[OIMAGE].Yc);
        fprintf(fp, "OIMAGE Hs=%x\n", descFullData.image_par[OIMAGE].Hs);
        fprintf(fp, "OIMAGE DataType=%x\n", descFullData.image_par[OIMAGE].DataType);
        fprintf(fp, "OIMAGE addr_end =%ld\n", descFullData.image_par[OIMAGE].addr_end);
        fprintf(fp, "MIMAGE addr_start=%ld\n", descFullData.image_par[MIMAGE].addr_start);
        fprintf(fp, "MIMAGE S=%x\n", descFullData.image_par[MIMAGE].S);
        fprintf(fp, "MIMAGE W=%x\n", descFullData.image_par[MIMAGE].W);
        fprintf(fp, "MIMAGE H=%x\n", descFullData.image_par[MIMAGE].H);
        fprintf(fp, "MIMAGE Xc=%x\n", descFullData.image_par[MIMAGE].Xc);
        fprintf(fp, "MIMAGE Yc=%x\n", descFullData.image_par[MIMAGE].Yc);
        fprintf(fp, "MIMAGE Hs=%x\n", descFullData.image_par[MIMAGE].Hs);
        fprintf(fp, "MIMAGE DataType=%x\n", descFullData.image_par[MIMAGE].DataType);
        fprintf(fp, "MIMAGE addr_end =%ld\n", descFullData.image_par[MIMAGE].addr_end);
        fprintf(fp, "GIMAGE addr_start=%ld\n", descFullData.image_par[GIMAGE].addr_start);
        fprintf(fp, "GIMAGE S=%x\n", descFullData.image_par[GIMAGE].S);
        fprintf(fp, "GIMAGE W=%x\n", descFullData.image_par[GIMAGE].W);
        fprintf(fp, "GIMAGE H=%x\n", descFullData.image_par[GIMAGE].H);
        fprintf(fp, "GIMAGE Xc=%x\n", descFullData.image_par[GIMAGE].Xc);
        fprintf(fp, "GIMAGE Yc=%x\n", descFullData.image_par[GIMAGE].Yc);
        fprintf(fp, "GIMAGE Hs=%x\n", descFullData.image_par[GIMAGE].Hs);
        fprintf(fp, "GIMAGE DataType=%x\n", descFullData.image_par[GIMAGE].DataType);
        fprintf(fp, "GIMAGE addr_end =%ld\n", descFullData.image_par[GIMAGE].addr_end);
        fprintf(fp, "Ho=%x\n", descFullData.Ho);
        fprintf(fp, "Wo=%x\n", descFullData.Wo);
        fprintf(fp, "Msi=%x\n", descFullData.Msi);
        fprintf(fp, "bli_shift=%x\n", descFullData.bli_shift);
        fprintf(fp, "MAX_VALo=%x\n", descFullData.MAX_VALo);
        fprintf(fp, "Ppo=%x\n", descFullData.Ppo);
        fprintf(fp, "rot_dir=%x\n", descFullData.rot_dir);
        fprintf(fp, "read_hflip=%x\n", descFullData.read_hflip);
        fprintf(fp, "read_vflip=%x\n", descFullData.read_vflip);
        fprintf(fp, "rot90=%x\n", descFullData.rot90);
        fprintf(fp, "rot90_intv=%x\n", descFullData.rot90_intv);
        fprintf(fp, "rot90_inth=%x\n", descFullData.rot90_inth);
        fprintf(fp, "im_read_slope=%x\n", descFullData.im_read_slope);
        fprintf(fp, "cosi=%x\n", descFullData.cosi);
        fprintf(fp, "sini=%x\n", descFullData.sini);
        fprintf(fp, "cosf=%f\n", descFullData.cosf);
        fprintf(fp, "sinf=%f\n", descFullData.sinf);
        fprintf(fp, "Yi_start=%x\n", descFullData.Yi_start); // Yi
        fprintf(fp, "Yi_end=%x\n", descFullData.Yi_end); // Yi
        fprintf(fp, "Xi_first_fixed=%x\n", descFullData.Xi_first_fixed); // Xi_first
        fprintf(fp, "Yi_first_fixed=%x\n", descFullData.Yi_first_fixed); // Yi_first
        fprintf(fp, "Xi_start_offset=%x\n", descFullData.Xi_start_offset); // Xi_start_off
        fprintf(fp, "Xi_start_offset_flip=%x\n", descFullData.Xi_start_offset_flip); // Xi_start_offset_flip
        fprintf(fp, "prec_align=%x\n", descFullData.prec_align);
        fprintf(fp, "M11i=%x\n", descFullData.M11i);
        fprintf(fp, "M12i=%x\n", descFullData.M12i);
        fprintf(fp, "M21i=%x\n", descFullData.M21i);
        fprintf(fp, "M22i=%x\n", descFullData.M22i);
        fprintf(fp, "M11f=%x\n", *(unsigned int*)(&descFullData.M11f));
        fprintf(fp, "M12f=%x\n", *(unsigned int*)(&descFullData.M12f));
        fprintf(fp, "M21f=%x\n", *(unsigned int*)(&descFullData.M21f));
        fprintf(fp, "M22f=%x\n", *(unsigned int*)(&descFullData.M22f));
        fprintf(fp, "prj_Af[0]=%x\n", *(unsigned int*)(&descFullData.prj_Af[0]));
        fprintf(fp, "prj_Af[1]=%x\n", *(unsigned int*)(&descFullData.prj_Af[1]));
        fprintf(fp, "prj_Af[2]=%x\n", *(unsigned int*)(&descFullData.prj_Af[2]));
        fprintf(fp, "prj_Bf[0]=%x\n", *(unsigned int*)(&descFullData.prj_Bf[0]));
        fprintf(fp, "prj_Bf[1]=%x\n", *(unsigned int*)(&descFullData.prj_Bf[1]));
        fprintf(fp, "prj_Bf[2]=%x\n", *(unsigned int*)(&descFullData.prj_Bf[2]));
        fprintf(fp, "prj_Cf[0]=%x\n", *(unsigned int*)(&descFullData.prj_Cf[0]));
        fprintf(fp, "prj_Cf[1]=%x\n", *(unsigned int*)(&descFullData.prj_Cf[1]));
        fprintf(fp, "prj_Cf[2]=%x\n", *(unsigned int*)(&descFullData.prj_Cf[2]));
        fprintf(fp, "prj_Df[0]=%x\n", *(unsigned int*)(&descFullData.prj_Df[0]));
        fprintf(fp, "prj_Df[1]=%x\n", *(unsigned int*)(&descFullData.prj_Df[1]));
        fprintf(fp, "prj_Df[2]=%x\n", *(unsigned int*)(&descFullData.prj_Df[2]));
        fprintf(fp, "mesh_format=%x\n", descFullData.mesh_format);
        fprintf(fp, "mesh_point_location=%x\n", descFullData.mesh_point_location);
        fprintf(fp, "mesh_sparse_h=%x\n", descFullData.mesh_sparse_h);
        fprintf(fp, "mesh_sparse_v=%x\n", descFullData.mesh_sparse_v);
        fprintf(fp, "mesh_Gh=%x\n", descFullData.mesh_Gh);
        fprintf(fp, "mesh_Gv=%x\n", descFullData.mesh_Gv);
        fprintf(fp, "mesh_rel_mode=%x\n", descFullData.mesh_rel_mode);
        fprintf(fp, "warp_stride=%x\n", descFullData.warp_stride);
        fprintf(fp, "resize_bli_grad_en=%x\n", descFullData.resize_bli_grad_en);
        if (descFullData.irt_mode == 7) {
            fprintf(fp, "rescale_phases_prec_H=%x\n", descFullData.rescale_phases_prec_H);
            fprintf(fp, "rescale_phases_prec_V=%x\n", descFullData.rescale_phases_prec_V);
            fprintf(fp, "rescale_LUT_x->num_phases=%x\n", descFullData.rescale_LUT_x->num_phases);
            fprintf(fp, "rescale_LUT_y->num_phases=%x\n", descFullData.rescale_LUT_y->num_phases);
            fprintf(fp, "rescale_LUT_x->num_taps=%x\n", descFullData.rescale_LUT_x->num_taps);
            fprintf(fp, "rescale_LUT_y->num_taps=%x\n", descFullData.rescale_LUT_y->num_taps);
        }
        fprintf(fp, "************************************************************************* \n");

        fprintf(fp, "************************************************************************ \n");
        fprintf(fp, "irt_cfg dump \n");
        fprintf(fp, "/*************************************************************************/ \n");
        for (int i = 0; i < 2; i++) {
            fprintf(fp, "buf_1b_mode[%x]=%x\n", i, irt_cfg.buf_1b_mode[i]);
            fprintf(fp, "buf_format[%x]=%x\n", i, irt_cfg.buf_format[i]);
            fprintf(fp, "buf_select[%x]=%x\n", i, irt_cfg.buf_select[i]);
            fprintf(fp, "buf_mode[%x]=%x\n", i, irt_cfg.buf_mode[i]);
            fprintf(fp, "Hb[%x]=%x\n", i, irt_cfg.Hb[i]);
            fprintf(fp, "Hb_mod[%x]=%x\n", i, irt_cfg.Hb_mod[i]);
        }


        fclose(fp);
    } else {
        printf("Unable to open the file %s for writing\n", filename);
    }
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
				IRT_TRACE_UTILS("%s angle %f is not supported, provide angle in [-360:360] range\n", irt_prj_matrix_s[angle_type], rot_pars.irt_angles[angle_type]);
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
			IRT_TRACE_UTILS("Converting rotation angle to %f with pre-rotation of 180 degree\n", rot_pars.irt_angles_adj[e_irt_angle_rot]);
		} else if (irt_angle_tmp[e_irt_angle_rot] == 90 || irt_angle_tmp[e_irt_angle_rot] == -90.0) {
			rot_pars.irt_angles_adj[e_irt_angle_rot] = irt_angle_tmp[e_irt_angle_rot];
			irt_desc.rot90 = 1;
			if ((irt_desc.image_par[OIMAGE].Xc & 1) ^ (irt_desc.image_par[IIMAGE].Yc & 1)) { //interpolation over input lines is required
				irt_desc.rot90_intv = 1;
				IRT_TRACE_UTILS("Rotation 90 degree will require input lines interpolation\n");
			}
			if ((irt_desc.image_par[OIMAGE].Yc & 1) ^ (irt_desc.image_par[IIMAGE].Xc & 1)) { //interpolation input pixels is required
				irt_desc.rot90_inth = 1;
				IRT_TRACE_UTILS("Rotation 90 degree will require input pixels interpolation\n");
			}
		} else {
			IRT_TRACE_UTILS("Rotation angle %f is not supported\n", rot_pars.irt_angles[e_irt_angle_rot]);
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

	IRT_TRACE_UTILS("Output stripe width calculation to fit rotation memory\n");

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

	//rot_pars.irt_rotate = irt_desc.irt_mode == e_irt_rotation;
	//bool irt_do_rotate = rot_pars.irt_rotate;
    bool irt_do_rotate = irt_desc.irt_mode == e_irt_rotation;

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
		IRT_TRACE_UTILS("Output image stripe calculation: So from BufW is %d, So from BufH is %d, selected So is %u\n", So_from_IBufW, So_from_IBufH, irt_desc.image_par[OIMAGE].S);
		IRT_TRACE_UTILS("Output image stripe calculation: output image size is reduced from %ux%u to %ux%u in %s mode\n",
			So, Ho, irt_desc.image_par[OIMAGE].S, irt_desc.image_par[OIMAGE].H, irt_irt_mode_s[irt_desc.irt_mode]);
	} else {
		IRT_TRACE_UTILS("Output image stripe calculation cannot find output stripe resolution\n");
		IRT_CLOSE_FAILED_TEST(0);
	}

	return irt_desc.image_par[OIMAGE].S;
}
} // namespace irt_utils_gaudi3
#pragma GCC diagnostic pop
