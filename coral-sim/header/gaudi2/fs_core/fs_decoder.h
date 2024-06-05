#pragma once

#include <stdint.h>

#include <atomic>
#include <mutex>

#include "gaudi2/asic_reg_structs/vdec_brdg_ctrl_regs.h"
#include "gaudi2/asic_reg_structs/vdec_ctrl_regs.h"
#include "gaudi2/asic_reg_structs/vsi_cmd_regs.h"
#include "gaudi2/asic_reg_structs/vsi_dec_regs.h"
#include "gaudi2/asic_reg_structs/vsi_l2c_regs.h"
#include "mstr_if_params.h"

#include "cbb.h"
#include "fs_common.h"
// TODO:: Add API expose header
#include "fs_coral_regspace.h"
#include "lbw_hub.h"

class FS_Decoder : public Cbb_Base
{
   public:
    static constexpr unsigned c_jpeg_progressive_table_size = 888; // JPEGDEC_PROGRESSIVE_TABLE_SIZE
    static constexpr unsigned c_scaling_lanczos_table_size  = 1280 * 1024;
    static constexpr unsigned c_cbx_size                    = 8;

    enum EModuleType
    {
        e_type_vc8000e = 0,
        e_type_cutree  = 1,
        e_type_vc8000d = 2,
        e_type_jpege   = 3,
        e_type_jpegd   = 4,
    };

    enum EClientType
    {
        e_client_type_h264_dec = 1, // DWL_CLIENT_TYPE_H264_DEC
        e_client_type_jpeg_dec = 3, // DWL_CLIENT_TYPE_JPEG_DEC
        e_client_type_vp9_dec  = 11, // DWL_CLIENT_TYPE_VP9_DEC
        e_client_type_hevc_dec = 12, // DWL_CLIENT_TYPE_HEVC_DEC
        e_client_type_pp_dec   = 14, // DWL_CLIENT_TYPE_PP
    };

    enum EDecMode
    {
        e_dec_mode_undefined = 0,
        e_dec_mode_jpeg      = 3, // Jpg/ProgressiveJpg DEC_MODE_JPEG
        e_dec_mode_hevc      = 12, // DEC_MODE_HEVC
        e_dec_mode_vp9       = 13, // DEC_MODE_VP9
        e_dec_mode_pp        = 14, // DEC_MODE_RAW_PP
        e_dec_mode_h264      = 15, // DEC_MODE_H264_H10P
    };

    enum EPriority
    {
        e_priority_normal = 0,
        e_priority_high   = 1,
    };

    enum EJpegMode
    {
        e_jpeg_mode_yuv400 = 0, // JPEGDEC_YUV400
        e_jpeg_mode_yuv420 = 2, // JPEGDEC_YUV420
        e_jpeg_mode_yuv422 = 3, // JPEGDEC_YUV422
        e_jpeg_mode_yuv444 = 4, // JPEGDEC_YUV444
        e_jpeg_mode_yuv440 = 5, // JPEGDEC_YUV440
        e_jpeg_mode_yuv411 = 6, // JPEGDEC_YUV411
    };

    enum EDecoderAxUserType
    {
        e_decoder_axuser_read,
        e_decoder_axuser_write,
    };

    struct FS_CmdBufAttributes
    {
        uint32_t sw_pic_width_in_cbs;
        uint32_t sw_pic_height_in_cbs;
        uint32_t sw_stream_len;
        uint32_t sw_bit_depth_c_minus8;
        uint32_t sw_bit_depth_y_minus8;
        uint32_t sw_num_tile_cols_8k;
        uint32_t sw_lref_height;
        uint32_t sw_lref_width;
        uint32_t sw_gref_height;
        uint32_t sw_gref_width;
        uint32_t sw_aref_height;
        uint32_t sw_aref_width;
        uint32_t sw_lref_c_stride;
        uint32_t sw_lref_y_stride;
        uint32_t sw_gref_c_stride;
        uint32_t sw_gref_y_stride;
        uint32_t sw_aref_c_stride;
        uint32_t sw_aref_y_stride;
        uint32_t sw_strm_buffer_len;
        uint32_t sw_dec_out_y_stride;
        uint32_t sw_dec_out_c_stride;
        uint32_t sw_pp_rgb_planar;
        uint32_t sw_pp_out_format;
        uint32_t sw_min_cb_size;
        uint32_t sw_max_cb_size;
        uint32_t sw_pp_out_y_stride;
        uint32_t sw_pp_out_c_stride;
        uint32_t sw_pp_out_height;
        uint32_t sw_pp1_rgb_planar;
        uint32_t sw_pp1_out_format;
        uint32_t sw_pp1_out_y_stride;
        uint32_t sw_pp1_out_c_stride;
        uint32_t sw_pp1_out_height;

        uint32_t sw_crop_starty;
        uint32_t sw_pp_in_height;
        uint32_t sw_pp1_crop_starty;
        uint32_t sw_pp1_in_height;
        uint32_t sw_pp_in_y_stride;
        uint32_t sw_pp_in_c_stride;

        int sw_dec_mode;
        int sw_jpeg_mode;

        uint64_t sw_rlc_vlc_base_orig;
        uint64_t sw_rlc_vlc_base_modified;
    };

    class FS_CmdBuf
    {
       public:
        enum EOpcode
        {
            e_opcode_wreg   = 0x1,
            e_opcode_end    = 0x2,
            e_opcode_nop    = 0x3,
            e_opcode_stall  = 0x9,
            e_opcode_rreg   = 0x16,
            e_opcode_jmp    = 0x19,
            e_opcode_clrint = 0x1A,
        };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
        struct packet_wreg
        {
            union
            {
                struct
                {
                    uint32_t StartRegAddr : 16;
                    uint32_t Length : 10;
                    uint32_t Fix : 1;
                    uint32_t OpCode : 5;

                    uint32_t WDATA[0];
                };
                uint32_t _raw[2];
            };
        };

        struct packet_end
        {
            struct
            {
                uint32_t : 27;
                uint32_t OpCode : 5;

                uint32_t : 32;
            };
        };
#pragma GCC diagnostic pop

        struct packet_nop
        {
            uint32_t : 27;
            uint32_t OpCode : 5;

            uint32_t : 32;
        };

        struct packet_stall
        {
            uint32_t InterruptMask : 16;
            uint32_t : 10;
            uint32_t IM : 1;
            uint32_t OpCode : 5;

            uint32_t : 32;
        };

        struct packet_rreg
        {
            uint32_t StartRegAddr : 16;
            uint32_t Length : 10;
            uint32_t Fix : 1;
            uint32_t OpCode : 5;

            uint32_t RegValueBufferAddr_31_0;

            uint32_t RegValueBufferAddr_63_32;

            uint32_t : 32;
        };

        struct packet_jmp
        {
            uint32_t NextCmdBufLength : 16;
            uint32_t : 9;
            uint32_t IE : 1;
            uint32_t RDY : 1;
            uint32_t OpCode : 5;

            uint32_t NextCmdBufAddr_31_0;

            uint32_t NextCmdBufAddr_63_32;

            uint32_t NextCmdBufId;
        };

        struct packet_clrint
        {
            uint32_t IntrRegAddr : 16;
            uint32_t : 9;
            uint32_t OpType : 2;
            uint32_t OpCode : 5;

            uint32_t BitMask;
        };

        class FS_PostProcess
        {
           public:
            enum EOperation
            {
                e_operation_copy_to_hbw     = 1 << 0,
                e_operation_copy_from_hbw   = 1 << 1,
                e_operation_copy_to_lbw     = 1 << 2,
                e_operation_fill_local_regs = 1 << 3,
                e_operation_jmp             = 1 << 4,
            };

            struct pp_params_copy_hbw
            {
                pp_params_copy_hbw() : deviceAddr(0), hostAddr(nullptr), buffer(nullptr), length(0) {}

                uint64_t deviceAddr;
                uint8_t* hostAddr; // host address to use
                uint8_t* buffer; // actual allocated buffer
                uint32_t length;
            };

            struct pp_params_copy_lbw
            {
                pp_params_copy_lbw() : deviceAddr(0), hostAddr(nullptr), buffer(nullptr), length(0) {}

                uint64_t  deviceAddr;
                uint8_t*  hostAddr; // host address to use
                uint32_t* buffer; // actual allocated buffer
                uint32_t  length;
            };

            struct pp_params_jmp
            {
                pp_params_jmp() : nextCmdBufAddr(0), nextCmdBufId(0), nextCmdBufLength(0), ie(false) {}

                uint64_t nextCmdBufAddr;
                uint32_t nextCmdBufId;
                uint16_t nextCmdBufLength;
                bool     ie;
            };

            struct pp_params_fill_regs
            {
                pp_params_fill_regs() : hostAddr(nullptr), offset(0), length(0) {}

                uint32_t* hostAddr;
                uint32_t  offset;
                uint32_t  length;
            };

            struct pp_params
            {
                pp_params() : op(0) {}

                int op;

                pp_params_copy_hbw  copy_hbw;
                pp_params_copy_lbw  copy_lbw;
                pp_params_jmp       jmp;
                pp_params_fill_regs fill_regs;
            };
        };

        static void fillPPCopyHbw(FS_Decoder*                decoder,
                                  FS_PostProcess::pp_params* pp,
                                  FS_CmdBufAttributes*       cmdBufAttributes,
                                  uint32_t                   length,
                                  packet_wreg*               cmd,
                                  unsigned                   i,
                                  int                        op,
                                  bool                       dmvFix = false);

        static void fillCmdBufAttributes(FS_Decoder*          decoder,
                                         uint8_t*             buf,
                                         unsigned             maxCmdBufSize,
                                         FS_CmdBufAttributes* cmdBufAttributes);

        static void fixupJpegWregBuffer(FS_Decoder*                             decoder,
                                        std::vector<FS_PostProcess::pp_params>& ppParams,
                                        FS_CmdBufAttributes*                    cmdBufAttributes,
                                        packet_wreg*                            cmd,
                                        unsigned                                i,
                                        uint32_t                                curOffsetInBlock);
        static void fixupPpWregBuffer(FS_Decoder*                             decoder,
                                      std::vector<FS_PostProcess::pp_params>& ppParams,
                                      FS_CmdBufAttributes*                    cmdBufAttributes,
                                      packet_wreg*                            cmd,
                                      unsigned                                i,
                                      uint32_t                                curOffsetInBlock);
        static void fixupH264WregBuffer(FS_Decoder*                             decoder,
                                        std::vector<FS_PostProcess::pp_params>& ppParams,
                                        FS_CmdBufAttributes*                    cmdBufAttributes,
                                        packet_wreg*                            cmd,
                                        unsigned                                i,
                                        uint32_t                                curOffsetInBlock);
        static void fixupH265WregBuffer(FS_Decoder*                             decoder,
                                        std::vector<FS_PostProcess::pp_params>& ppParams,
                                        FS_CmdBufAttributes*                    cmdBufAttributes,
                                        packet_wreg*                            cmd,
                                        unsigned                                i,
                                        uint32_t                                curOffsetInBlock);
        static void fixupVp9WregBuffer(FS_Decoder*                             decoder,
                                       std::vector<FS_PostProcess::pp_params>& ppParams,
                                       FS_CmdBufAttributes*                    cmdBufAttributes,
                                       packet_wreg*                            cmd,
                                       unsigned                                i,
                                       uint32_t                                curOffsetInBlock);

        static void dumpBuffer(FS_Decoder*                             decoder,
                               uint8_t*                                buf,
                               unsigned                                maxCmdBufSize,
                               std::vector<FS_PostProcess::pp_params>& ppParams,
                               packet_jmp**                            jmp,
                               FS_CmdBufAttributes*                    cmdBufAttributes);
    };

    FS_Decoder(coral_module_name name);
    ~FS_Decoder() override;

    void               setInstanceName(const std::string& str);
    const std::string& getInstanceName() const { return m_name; }

    void connect_to_lbw(LBWHub* lbw, Specs* specs, uint32_t itr) override;
    void tcl_update_register(Addr_t addr_offset, uint32_t& value, Lbw_Protection_e prot, bool write) override;
    void tcl_update_mstrif_register(Addr_t           addr_offset,
                                    uint32_t&        value,
                                    Lbw_Protection_e prot,
                                    bool             write = true) override; // unit to add its base to offset

    void registerBlock(LBWHub*  lbwHub,
                       uint64_t baseAddressCtrl,
                       uint64_t baseAddressBrdgCtrl,
                       uint64_t baseAddressVsi,
                       uint64_t baseAddressL2c,
                       uint64_t baseAddressCmd);
    void cycle() override final;
    void reset();

   private:
    std::string m_name;

    Coral_RegSpace m_regSpaceCtrl;
    Coral_RegSpace m_regSpaceBrdgCtrl;
    Coral_RegSpace m_regSpaceVsi;
    Coral_RegSpace m_regSpaceL2c;
    Coral_RegSpace m_regSpaceCmd;

    gaudi2::block_vdec_ctrl*      m_blockCtrl;
    gaudi2::block_vdec_brdg_ctrl* m_blockBrdgCtrl;
    gaudi2::block_vsi_dec*        m_blockVsi;
    gaudi2::block_vsi_l2c*        m_blockL2c;
    gaudi2::block_vsi_cmd*        m_blockCmd;
    MstrIfParams_t                mstrif_params;

    std::vector<uint8_t>                     m_cmdBuffer;
    FS_CmdBuf::packet_jmp*                   m_jmp        = nullptr;
    uint64_t                                 m_jmpPktAddr = 0;
    uint32_t                                 m_jmpAxUser  = 0;
    bool                                     m_terminate;
    bool                                     m_reset;

    void* m_configParams;
    void* m_cmdbufMemParams;

    static std::atomic<int> m_libCounter;

    bool wreg32PrivateCtrl(uint32_t offset, uint32_t value);
    bool wreg32PrivateBrdgCtrl(uint32_t offset, uint32_t value);
    bool wreg32PrivateVsi(uint32_t offset, uint32_t value);
    bool wreg32PrivateL2c(uint32_t offset, uint32_t value);
    bool wreg32PrivateCmd(uint32_t offset, uint32_t value);

    uint32_t    getAxUser(enum EDecoderAxUserType axuser_type) const;
    void        threadMain();
    void        processCommandBuffer();
    void        executeResetCmd();

    void lbwWrite32(uint64_t addr, uint32_t data);
};
