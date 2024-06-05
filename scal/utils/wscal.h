#pragma once
#include <iostream>
#include "scal.h"
//#include "hlthunk.h"
//#include "scal_utilities.h"
//#include "logger.h"
//#include "scal_assert.h"
#include "scal_test_utils.h"
#include "../tests/scal_basic_test.h"

//#define USE_DEBUG
#ifdef USE_DEBUG
#include "scal_base.h"
extern unsigned g_wscal_debug; // set it from outside for more debug printouts
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_STREAMS 32

// bind together the stream + command buffer + cg (completion group) + cluster
class streamBundle
{
    public:
        scal_stream_handle_t h_stream;
        scal_stream_info_t h_streamInfo;
        scal_buffer_handle_t h_cmdBuffer;
        scal_buffer_info_t h_cmdBufferInfo;
        scal_comp_group_handle_t h_cg;
        scal_completion_group_infoV2_t h_cgInfo;
        scal_cluster_handle_t h_cluster;
        scal_cluster_info_t h_clusterInfo;
        SchedCmd h_cmd;// handles cmdBuffer

        // methods
        void AllocAndDispatchBarrier(uint32_t num_engine_group_type, uint8_t* engine_group_type,
                                     uint32_t targetValue = (uint32_t)-1, bool rel_so_set = false)
        {
            if (targetValue == (uint32_t)-1) targetValue = h_clusterInfo.numCompletions;
            assert(targetValue>=0);
            if (h_cgInfo.force_order) targetValue++; // due to a force-order initializing so's to 1, according to Amos, should be done by the user
            h_cmd.AllocAndDispatchBarrier(h_cgInfo.index_in_scheduler, targetValue, (uint32_t)rel_so_set,
                                          num_engine_group_type, engine_group_type);
        }

        void WaitOnHostOverPdmaStream(uint32_t engine_group_type, bool isDirectMode)
        {
            assert((engine_group_type >= SCAL_PDMA_TX_CMD_GROUP) && (engine_group_type <= SCAL_PDMA_RX_DEBUG_GROUP));
            h_cmd.PdmaTransferCmd(isDirectMode,
                                  0 /* dst */,
                                  0 /* src */,
                                  0 /* size */,
                                  engine_group_type,
                                  -1 /* workload_type */,
                                  0 /* payload */,
                                  0 /* pay_addr */,
                                  1 /* signal_to_cg */,
                                  h_cgInfo.index_in_scheduler);
        }

        int stream_submit()
        {
            return scal_stream_submit(h_stream, (unsigned)h_cmd.getPi3(), h_streamInfo.submission_alignment);
        }

        int completion_group_wait(const uint64_t target, const uint64_t timeout = SCAL_FOREVER)
        {
#ifdef USE_DEBUG
            if (g_wscal_debug)
            {
                const Scal::CompletionGroup* cq        = (const Scal::CompletionGroup*)h_cg;
                unsigned                     mon_depth = cq->monNum / cq->actualNumberOfMonitors;
                //printf("stream %s will wait on CQ (%s) #%u of (main) scheduler %s (%u) interrupt 0x%x target=%lu mon [%u-%u] longSoIdx=%u so range [%u-%u]\n",
                LOG_DEBUG(SCAL,"stream {} will wait on CQ ({}) #{} of (main) scheduler {} ({}) interrupt {:#x} target={:#x} mon [{}-{}] longSoIdx={} so range [{}-{}]",
                       h_streamInfo.name, cq->name.c_str(),
                       cq->cqIdx, cq->scheduler->name.c_str(), cq->scheduler->cpuId, cq->isrIdx, target, cq->monBase, cq->monBase + mon_depth - 1, cq->longSoIndex, cq->sosBase,
                       cq->sosBase + cq->sosNum - 1);
                for (unsigned i = 0; i < h_cgInfo.num_slave_schedulers; i++)
                {
                    const Scal::Core* sched = cq->slaveSchedulers[i].scheduler;
                    LOG_DEBUG(SCAL,"CQ {} slave #{} scheduler {} ({}) cqIdx={}",
                        cq->name.c_str(), i, sched->name.c_str(), sched->cpuId, cq->slaveSchedulers[i].idxInScheduler);
                }
            }
#endif
            return  scal_completion_group_wait(h_cg, target, timeout);
        }

};

typedef struct _bufferBundle_t
{
    scal_buffer_handle_t h_Buffer;
    scal_buffer_info_t h_BufferInfo;
} bufferBundle_t;

typedef std::vector<std::string> strVec;

class WScal
{
    public:
        enum PoolType { HostSharedPool,GlobalHBMPool,devSharedPool};
        WScal(const char* config, strVec streams, strVec cgs, strVec clusters, bool skipIfDirectMode = false,
        strVec directModeCgs = {""}, int fd=-1);
        ~WScal();
        int getBufferX(PoolType pType, unsigned size,bufferBundle_t* buf);// get both buffer handle and info
        streamBundle* getStreamX(unsigned index);
        scal_handle_t getScalHandle() { return m_scalHandle; }
        scalDeviceType getScalDeviceType() { return ::getScalDeviceType(m_hw_ip.device_id);}
        unsigned getNumStreams() { return m_numStreams;}
        int getFd() { return m_scalFd;}
        int getStatus(){ return m_status;}
        //
    protected:
        int InitMemoryPools();
        int InitStreams(strVec streams, strVec cgs, strVec directModeCg, strVec clusters);

        //
        // Handles
        //
        int m_scalFd;// hlthunk fd
        struct hlthunk_hw_ip_info m_hw_ip = {};
        bool m_releaseFd;
        scal_handle_t m_scalHandle = nullptr;                        // main scal handle
        scal_pool_handle_t m_hostMemPoolHandle = nullptr;                 // handle for the "host shared" pool for the ctrl cmd buffer (memory is shared between host and arcs)
        scal_pool_handle_t m_deviceSharedMemPoolHandle = nullptr;         // handle for the "hbm_shared" pool for the user data on the device
        scal_pool_handle_t m_deviceMemPoolHandle = nullptr;               // handle for the "global_hbm" pool for the user data on the device
        //
        streamBundle m_streamsX[MAX_STREAMS]= {};
        unsigned m_numStreams = 0;
        int      m_status;
        bool     m_skipInDirectMode = false;


};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
