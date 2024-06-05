#ifndef SCAL_SUBMIT_TRACER_H
#define SCAL_SUBMIT_TRACER_H


#ifdef STAND_ALONE_CPP
#include <assert.h>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include "scal.h"
#include "scal_utilities.h"
#include "scal_base.h"
#include "logger.h"

#include "gaudi2/asic_reg_structs/qman_arc_aux_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"
#include "gaudi2/asic_reg/gaudi2_regs.h"

#include "scal_internal_data.h"
#endif


#include "gaudi2_arc_sched_packets.h"

/*
    To Use:

    in scal_rt.cpp

    * add at the top
            #define SCAL_DEBUG_TOOLS
            #include "../utils/scal_debug_tools.h"

    * inside Scal::streamSubmit(..)
      add
            #ifdef SCAL_DEBUG_TOOLS
                SubmitTracer(stream,pi);
            #endif
     just before the call to writeLbwReg(...)

    * you can also add (but the information is already printed into scal_log.txt)
          so only if you want it on the console together with the submit info above (helps debugging)
      inside Scal::completionGroupWait(...)
        #ifdef SCAL_DEBUG_TOOLS
                waitCQTracer(completionGroup, target);
        #endif
        just before the call to hlthunk_wait_for_interrupt(...)

*/

struct StreamTracerInfo {
    Scal::Stream *stream;
    unsigned pi;
};
class ScalDebugTool
{
    public:
        ScalDebugTool()
        {

        }
        void streamSubmit(Scal::Stream *stream, const unsigned pi);
        void waitCQ(const Scal::CompletionGroup* cq, const uint64_t target);
    protected:
        void parseBuf(Scal::Stream *stream, unsigned pPi,const unsigned pi);
        std::vector<StreamTracerInfo> m_streamTracers;
};

void ScalDebugTool::parseBuf(Scal::Stream *stream,unsigned pPi,const unsigned pi)
{
    if (stream.scheduler->name != "compute_media_scheduler")
    {
        LOG_ERR(SCAL, "ScalDebugTool is only usable for compute_media_scheduler");
        return;
    }
    uint64_t pBuffaddr = (uint64_t)(stream->coreBuffer->pool->hostBase) + stream->coreBuffer->base;
    uint8_t* pBuff = (uint8_t*)(pBuffaddr + pPi);
    uint8_t* pBuffEnd = (uint8_t*)(pBuffaddr+pi);
    // uint32_t opcode:5;
    while (pBuff < pBuffEnd)
    {
        uint8_t op = *pBuff & 0x1F;
        switch (op)
        {
        case SCHED_COMPUTE_ARC_CMD_FENCE_WAIT:
            {
                struct sched_arc_cmd_fence_wait_t* fence = (struct sched_arc_cmd_fence_wait_t*)pBuff;
                printf ("\tSCHED_COMPUTE_ARC_CMD_FENCE_WAIT fence_id=%u target=%u\n", fence->fence_id, fence->target);
                pBuff += sizeof(struct sched_arc_cmd_fence_wait_t);
            }
            break;
        case SCHED_COMPUTE_ARC_CMD_LBW_WRITE:
            printf ("\tSCHED_COMPUTE_ARC_CMD_LBW_WRITE\n");
            pBuff += sizeof(struct sched_arc_cmd_lbw_write_t);
            break;
        case SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE:
            printf ("\tSCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE\n");
            pBuff += sizeof(struct sched_arc_cmd_lbw_burst_write_t);
            break;
        case SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER:
            {
                struct sched_arc_cmd_dispatch_barrier_t* t = (struct sched_arc_cmd_dispatch_barrier_t*)pBuff;
                printf ("\tSCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER num_groups=%u (%u,%u,%u,%u)\n", t->num_engine_group_type,
                        t->engine_group_type[0], t->engine_group_type[1],
                        t->engine_group_type[2], t->engine_group_type[3]);
                pBuff += sizeof(struct sched_arc_cmd_dispatch_barrier_t);
            }
            break;
        case SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE:
            {
                struct sched_arc_cmd_fence_inc_immediate_t* cmd = (struct sched_arc_cmd_fence_inc_immediate_t*)pBuff;
                printf ("\tSCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE fence_index=%u\n", cmd->fence_index);
                pBuff += sizeof(struct sched_arc_cmd_fence_inc_immediate_t);
            }
            break;
        case SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER:
            {
                struct sched_arc_cmd_pdma_batch_transfer_t* trans = (struct sched_arc_cmd_pdma_batch_transfer_t*)pBuff;
                printf ("\tSCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER size=0x%x has_payload=%u\n", trans->transfer_size, trans->has_payload);
                // move the buffer pointer to point after the batch pdma command and its parameters.
                // calculating size of the batch struct itself, and the structs of parameters coming after.
                pBuff += sizeof(struct sched_arc_cmd_pdma_batch_transfer_t) +
                        trans->batch_count * sizeof(sched_arc_pdma_commands_params_t);
            }
            break;
        case SCHED_COMPUTE_ARC_CMD_LBW_READ:
            printf ("\tSCHED_COMPUTE_ARC_CMD_LBW_READ\n");
            pBuff += sizeof(struct sched_arc_cmd_lbw_read_t);
            break;
        case SCHED_COMPUTE_ARC_CMD_MEM_FENCE:
            printf ("\tSCHED_COMPUTE_ARC_CMD_MEM_FENCE\n");
            pBuff += sizeof(struct sched_arc_cmd_mem_fence_t);
            break;
        case SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2:
            printf ("\tSCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2\n");
            pBuff += sizeof(struct sched_arc_cmd_update_recipe_base_v2_t);
            break;
        case SCHED_COMPUTE_ARC_CMD_NOP:
            {
                struct sched_arc_cmd_nop_t* nop = (struct sched_arc_cmd_nop_t*)pBuff;
                printf ("\tSCHED_COMPUTE_ARC_CMD_NOP  padding=%u\n",nop->padding_count);
                pBuff += sizeof(struct sched_arc_cmd_nop_t) + nop->padding_count * sizeof(uint32_t);
            }
            break;
        default:
            printf ("\nOOPS undefined op=%u\n", op);
            pBuff = pBuffEnd;
            break;
        };
    }
}

void ScalDebugTool::streamSubmit(Scal::Stream *stream, const unsigned pi)
{
    unsigned pPi = 0;
    unsigned idx = 0;
    for (auto& X : m_streamTracers)
    {
        if (X.stream == stream)
        {
            pPi = X.pi;
            X.pi = pi;
            printf("stream %s (%u) pPi=%u\n",stream->name.c_str(), stream->id, pPi);
            break;
        }
        idx++;
    }
    if (pPi == 0)
    {
        struct StreamTracerInfo Y;
        Y.stream = stream;
        Y.pi = pi;
        printf("stream %s (%u) pi=%u\n",stream->name.c_str(), stream->id, pi);
        m_streamTracers.push_back(Y);
    }
    parseBuf(stream,pPi,pi);
}

void ScalDebugTool::waitCQ(const Scal::CompletionGroup* cq, const uint64_t target)
{
    unsigned mon_depth = cq->monNum / cq->actualNumberOfMonitors;
    printf("will wait on CQ (%s) #%u of (main) scheduler %s (%u) interrupt 0x%x target=%lu mon [%u-%u] longSoIdx=%u so range [%u-%u] forceOrder=%u\n",
            cq->name.c_str(), cq->cqIdx, cq->scheduler->name.c_str(), cq->scheduler->id, cq->isrIdx, target,
            cq->monBase, cq->monBase + mon_depth - 1, cq->longSoIndex, cq->sosBase, cq->sosBase + cq->sosNum - 1, cq->force_order);
    unsigned i=0;
    for (auto& sched : cq->slaveSchedulers)
    {
        printf("\tCQ %s slave #%u scheduler %s (%u) cqIdx=%u\n",
            cq->name.c_str(), i, sched.scheduler->name.c_str(), sched.scheduler->id, cq->slaveSchedulers[i].idxInScheduler);
        i++;
    }
}

//
//  The global object! (not thread safe ..)
//
ScalDebugTool g_ScalDebugTool;
//
//
//


void SubmitTracer(Scal::Stream *stream, const unsigned pi)
{
    g_ScalDebugTool.streamSubmit(stream,pi);
}

void waitCQTracer(const Scal::CompletionGroup* cq, const uint64_t target)
{
    g_ScalDebugTool.waitCQ(cq, target);
}
#endif