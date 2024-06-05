#ifndef _SCAL_TEST_UTILS_H_
#define _SCAL_TEST_UTILS_H_

#include <string>
#include <variant>

#include "scal_internal/pkt_macros.hpp"
#include "scal.h"
#include "../tests/scal_basic_test.h"
#include "scal_internal/sched_pkts.hpp"


//////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
std::string getConfigFilePath(const char* configFileName);

//////////////////////////////////////////////////////////////////////////////////////////////
// SchedCmd class is a wrapper on top of the scheduler commands buffer
// it handles the buffer PI index, and takes care of the command crossing/spanning/wrapping limitation
//  (where commands cannot cross the dccm buffer size points)
//   so if our host cmd buffer is 2K and dccm buffer is 512 bytes
//     our command cannot cross  512,1024,1536,2048
//     and must use padding to fill the gaps
class SchedCmd {
    public:
        SchedCmd() {}
        SchedCmd(char *cmdBufferBase, unsigned cmdBufferSize, unsigned stream_dccm_buf_size):
            m_cmdBufferBase(cmdBufferBase),m_cmdBufferSize(cmdBufferSize), m_offset(0),
            m_streamDccmBufSize(stream_dccm_buf_size),m_canReset(false)
            {}
        void Init(char *cmdBufferBase, unsigned cmdBufferSize, unsigned stream_dccm_buf_size, scalDeviceType device);
        void PadReset(); // pad with nops to the end of the buffer and reset the buffer offset to 0
        void AllowBufferReset() { m_canReset=true;}// if true, assume you can start over from the beginning of the buffer once filled
        unsigned getPi() { return m_offset;} // in gaudi2 pi is always an offset into the command buffer (e.g. it wraps around to 0)
        uint64_t getPi3() { return m_pi3;} // in gaudi3, especially in auto fetcher mode, pi is an absolute counter (e.g. it does not wrap around)
        //
        bool NopCmd(uint32_t padding);
        bool FenceCmd(uint32_t fence_id, uint32_t target, bool forceFwFence, bool isDirectMode);
        bool PdmaTransferCmd(bool isDirectMode, uint64_t dst, uint64_t src, uint32_t size,
                             uint32_t engine_group_type = SCAL_PDMA_TX_DATA_GROUP,
                             int32_t workload_type = -1, uint32_t payload = 0, uint64_t pay_addr = 0,
                             uint32_t signal_to_cg = 0, uint32_t compGrpIdx = 0, bool wr_comp = false,
                             scal_completion_group_infoV2_t* cgInfo = nullptr);
        bool AllocBarrier(uint32_t comp_group_index, uint32_t target_value, uint32_t rel_so_set = 0);
        bool DispatchBarrier(uint32_t num_engine_group_type, uint8_t *engine_group_type);
        bool AllocAndDispatchBarrier(uint32_t comp_group_index, uint32_t target_value, uint32_t rel_so_set,uint32_t num_engine_group_type, uint8_t *engine_group_type);
        bool DispatchComputeEcbList(uint32_t engine_group_type,
                                bool single_static_chunk, bool single_dynamic_chunk,
                                uint32_t static_ecb_list_offset, uint32_t dynamic_ecb_list_addr);
        bool UpdateRecipeBase(const uint16_t *recipe_base_indexes,
                              uint32_t num_engine_group_type,
                              uint8_t* engine_group_type,
                              const uint64_t* recipe_base_addrsses,
                              uint32_t num_recipe_base_elements);

        bool FenceIncImmediate(uint32_t fence_count, uint8_t *arr_fence_id, bool forceFwFence);

        bool lbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool isDirectMode = false);
        bool lbwBurstWrite(uint32_t dst_addr, uint32_t* data, bool block_stream);
        bool lbwRead(uint32_t dst_addr, uint32_t src_addr, uint32_t size);
        bool memFence(bool arc, bool dup_eng, bool arc_dma); // wait till arc,dup_eng,arc_dma are done

        static uint64_t getArcAcpEngBaseAddr(unsigned smIndex);

        void setPi(uint64_t pi);

    protected:
        unsigned CheckCmdWrap(unsigned newCmdSize);
        bool Check(unsigned newCmdSize, bool isDirectMode = false);
        //
        scalDeviceType deviceType = dtOther;
        std::variant<G2Packets , G3Packets> m_buildPkt;
        char* m_cmdBufferBase = nullptr;
        unsigned m_cmdBufferSize = 0;
        unsigned m_offset = 0;
        uint64_t m_pi3 = 0;
        unsigned m_streamDccmBufSize = 0;
        bool m_canReset = false;
};


// EcbListCmd wrapper around ARC cmds in EcbList
// takes care of the command crossing/spanning/wrapping limitation
class EcbListCmd {
    public:
        EcbListCmd() {}
        EcbListCmd(char *bufferBase, unsigned bufferSize, scalDeviceType device);
        void Pad(uint32_t until, bool switchCQ); // pad with nops
        bool NopCmd(uint32_t padding, bool switchCQ = false, bool yield = false, uint32_t dma_completion = 0);
        bool ArcListSizeCmd(uint32_t list_size, bool yield = false, uint32_t dma_completion = 0);
        bool staticDescCmd(uint32_t cpu_index, uint32_t size, uint32_t addr_offset, uint32_t addr_index, bool yield = false);

    protected:
        bool Check(unsigned newCmdSize);
        //
        std::variant<G2Packets , G3Packets> m_buildPkt;
        char*    m_bufferBase =  nullptr;
        unsigned m_bufferSize = 0;
        unsigned m_offset     = 0;
        unsigned m_streamDccmBufSize = 0;
};

std::variant<G2Packets , G3Packets> createBuildPkts(scalDeviceType device);

#endif
