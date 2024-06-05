#include <assert.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <sys/mman.h>
#include "scal.h"
#include "scal_utilities.h"
#include "scal_base.h"
#include "logger.h"

#include "scal_data_gaudi2.h"
#include "scal_qman_program_gaudi2.h"
#include "gaudi2/asic_reg_structs/qman_arc_aux_regs.h"
#include "gaudi2/asic_reg_structs/qman_regs.h"

#include "gaudi2/asic_reg/dcore0_mme_qm_arc_acp_eng_regs.h"
#include "gaudi2/asic_reg/dcore1_mme_qm_arc_acp_eng_regs.h"
#include "gaudi2/asic_reg/dcore2_mme_qm_arc_acp_eng_regs.h"
#include "gaudi2/asic_reg/dcore3_mme_qm_arc_acp_eng_regs.h"

#include "scal_macros.h"
#include "gaudi2_info.hpp"

int Scal_Gaudi2::isLocal(const ArcCore* arcCore, bool & localMode)
{
    // localMode:  detect if we're running locally, meaning, qman and core are on the same address space
    // or are they on different locations (example:  qman of TPC configures DCCM of arcFarm )
    uint64_t qmanDccmAddr;
    if (!qmanId2DccmAddr(arcCore->qmanID, qmanDccmAddr))
    {
        LOG_ERR(SCAL,"{}, qmanId2DccmAddr() failed for core id {}", __FUNCTION__, arcCore->qmanID);
        assert(0);
        return SCAL_FAILURE;
    }
    localMode = (qmanDccmAddr == arcCore->dccmDevAddress);

    return SCAL_SUCCESS;
}

Scal_Gaudi2::Scal_Gaudi2(
    const int fd,
    const struct hlthunk_hw_ip_info & hw_ip,
    scal_arc_fw_config_handle_t fwCfg)
    : Scal(fd, hw_ip, c_cores_nr, c_sync_managers_nr, SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT,
           std::unique_ptr<DevSpecificInfo>(new Gaudi2Info))
{
    m_schedulerNr = c_scheduler_nr;
    if (fwCfg)
    {
        m_arc_fw_synapse_config_t = *(struct arc_fw_synapse_config_t*)fwCfg;
    }
    else
    {
        memset(&m_arc_fw_synapse_config_t,0,sizeof(m_arc_fw_synapse_config_t));
    }
}

Scal_Gaudi2::~Scal_Gaudi2()
{
    int ret=0;

    ret |= haltCores();
    scal_assert(ret == SCAL_SUCCESS,"in Scal destructor, haltCores failed");

    if(m_dmaBuff4tpcNopKernel)
    {
        delete m_dmaBuff4tpcNopKernel;
    }

    if (ret != SCAL_SUCCESS)
    {
        LOG_ERR(SCAL,"{}: fd={} errors in scal destructor", __FUNCTION__, m_fd);
    }
}

uint64_t Scal_Gaudi2::getCoreAuxOffset(const ArcCore * core)
{
    if (const Scheduler * scheduler = core->getAs<Scheduler>())
    {
        if (scheduler->arcFarm)
        {
            return mmARC_FARM_ARC0_AUX_BASE - mmARC_FARM_ARC0_DCCM0_BASE;
        }
        else
        {
            return mmDCORE0_TPC0_QM_ARC_AUX_BASE - mmDCORE0_TPC0_QM_DCCM_BASE;
        }
    }
    else
    {
        return mmDCORE0_TPC0_QM_ARC_AUX_BASE - mmDCORE0_TPC0_QM_DCCM_BASE;
    }
}

int Scal_Gaudi2::haltCores()
{
    // halt all the active cores

    // get the indices of the relevant cores to halt
    uint32_t coreIds[c_cores_nr] = {};
    uint32_t counter = 0;
    for (uint32_t idx = 0; idx < c_cores_nr; idx++)
    {
        Core * core = m_cores[idx];
        if (core)
        {
            coreIds[counter++] = idx;
        }
    }

    // request halt core
    int rc = hlthunk_engines_command(m_fd, coreIds, counter, HL_ENGINE_CORE_HALT);
    if (rc != 0)
    {
        LOG_ERR(SCAL,"{}: fd={} Failed to halt cores (amount {}) rc {}", __FUNCTION__, m_fd, counter, rc);
        assert(0);
        return SCAL_FAILURE;
    }

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::streamSubmit(Stream *stream, const unsigned pi, const unsigned submission_alignment)
{
    assert(submission_alignment !=0); // currently we support only buffer size of 65536

    //  Todo: streamSubmit is on the critical path. if needed, change to assert only
    LOG_TRACE(SCAL,"submit stream ({}) #{} pi={} {:#x} on core {}", stream->name, stream->id, pi, pi, stream->scheduler->name);
    if (pi % submission_alignment != 0)
    {
        LOG_ERR(SCAL, "{}, pi is not submission_alignemnt bytes aligned {}", __FUNCTION__, pi);
        assert(0);
        return SCAL_INVALID_PARAM;
    }
    uint32_t g2_pi = pi % stream->coreBuffer->size;// this will allow the user to treat Gaudi3 and Gaudi2 the same way
    writeLbwReg(stream->pi, (g2_pi/submission_alignment));

    return SCAL_SUCCESS;
}

int Scal_Gaudi2::setStreamBuffer(Stream *stream, Buffer *buffer) const
{
    /* check for buffer size restrictions:
     minimal size is 2^16
     max size 2^24
     must be power of 2
     so quant must be ( TODO add assert for that 1/2/4/8/16/32/64/128/256)
     first stage FW supports only 2^16
     */
    if (!buffer->pool->coreBase)
    {
        LOG_ERR(SCAL,"{}, buffer not mapped to the cores {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    if (buffer->size < c_core_counter_max_value)
    {
        LOG_ERR(SCAL,"{}, buffer size too small {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    if (!isPowerOf2Unsigned(buffer->size))
    {
        LOG_ERR(SCAL,"{}, buffer size is not power of 2 {}", __FUNCTION__, buffer->size);
        assert(0);
        return SCAL_INVALID_PARAM;
    }

    // compute address on device for the cyclic comman buffer information for this stream
    // scheduler registers starts at the base of the scheduler dccm

    uint64_t baseAddr      = (uint64_t)(stream->scheduler->dccmHostAddress) + offsetof(sched_registers_t, ccb) + sizeof(struct ccb_t) * stream->id;
    uint64_t ccbAddressPtr = baseAddr + offsetof(ccb_t, ccb_addr);

    uint64_t ccbSizePtr = baseAddr + offsetof(ccb_t, ccb_size);

    writeLbwReg((volatile uint32_t*)ccbAddressPtr, (buffer->base + buffer->pool->coreBase));
    writeLbwReg((volatile uint32_t*)ccbSizePtr, buffer->size);
    stream->coreBuffer = buffer;


    return SCAL_SUCCESS;
}

int Scal_Gaudi2::setStreamPriority(Stream *stream, unsigned priority) const
{
    // TODO update the priority also in the ccb_t struct when FW team adds support for that.
    /* from user perspective there are 32 streams. they are implemented with 64 internal stream:
       [0-31]  dma stream - highest priority to DMA commands into the device
       [32-63] execution stream - include the execution commands themselves.
       within the execution stream the user can define 2 different priorities high/low
       when configuring priorities the dma stream priority must always be higher then the
       its execution stream [ 0 max priority, 3 lowest priority]
       0 - high dma priority
       1 - low dma priority
       2 - high exec priority  ( from user perspective SCAL_HIGH_PRIORITY_STREAM)
       3 - low exec priority ( from user perspective SCAL_LOW_PRIORITY_STREAM)
       illustration of the stream information priorities: split to "DMA stream" and "EXE stream"
       (priority 0: dma stream ( for example, stream #0) would get priority of
       its exe stream ( stream $32)  would be get priority 2)

                      DMA                     EXE
                     ______________________________________________
       priority     |0|1|0|0|...............|2|3|2|
       PI
       CI
       MASK

       */

    uint64_t dmaStreamPriorityAddress;
    uint64_t exeStreamPriorityAddress;

    if ((priority != SCAL_HIGH_PRIORITY_STREAM) && (priority != SCAL_LOW_PRIORITY_STREAM))
    {
        LOG_ERR(SCAL,"{}, invalid priority {}", __FUNCTION__, priority);
        assert(0);
        return SCAL_FAILURE;
    }
    // The entire LBW address space is 27 bits (mmARC_FARM_ARC0_ACP_ENG_ACP_PR_REG_0 for example is 27 bits)
    // the ACP block is 4 KB - can look at it in gaudi2_blocks.h:
    // #define DCORE0_MME_QM_ARC_ACP_ENG_MAX_OFFSET 0x1000
    // we want to figure out the priority register offset relative to its size, we can do that by modulo as seen below
    if(stream->scheduler->arcFarm)
    {
        dmaStreamPriorityAddress = (uint64_t)stream->scheduler->acpHostAddress +
            (mmARC_FARM_ARC0_ACP_ENG_ACP_PR_REG_0 % ARC_FARM_ARC0_ACP_ENG_MAX_OFFSET) +
            stream->id*sizeof(uint32_t);
    }
    else
    {
        dmaStreamPriorityAddress = (uint64_t)stream->scheduler->acpHostAddress +
            (mmDCORE0_MME_QM_ARC_ACP_ENG_ACP_PR_REG_0 % DCORE0_MME_QM_ARC_ACP_ENG_MAX_OFFSET) +
            stream->id * sizeof(uint32_t);
    }

    exeStreamPriorityAddress = dmaStreamPriorityAddress + c_num_max_user_streams*sizeof(uint32_t);

    writeLbwReg((volatile uint32_t*) exeStreamPriorityAddress, priority);

    stream->priority = priority;

    return SCAL_SUCCESS;
}
