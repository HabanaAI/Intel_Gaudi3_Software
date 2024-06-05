#include <assert.h>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include "scal.h"
#include "scal_utilities.h"
#include "scal_base.h"
#include "logger.h"

#include "gaudi3/asic_reg_structs/arc_acp_eng_regs.h"
#include "gaudi3/asic_reg/gaudi3_blocks.h"
#include "gaudi3/asic_reg_structs/arc_af_eng_regs.h"

#include "scal_macros.h"
#include "scal_data_gaudi3.h"
#include "gaudi3_info.hpp"

#include <infiniband/hlibdv.h>
#include <dlfcn.h>

using namespace Qman;

static constexpr char c_scal_sync_scheme_env_var_name[] = "SCAL_SYNC_SCHEME";

int Scal_Gaudi3::isLocal(const ArcCore* core, bool & localMode)
{
    // localMode:  detect if we're running locally, meaning, qman and core are on the same address space
    // or are they on different locations (example:  qman of TPC configures DCCM of arcFarm )
    uint64_t qmanDccmAddr;
    if (!queueId2DccmAddr(core->qmanID, qmanDccmAddr))
    {
        LOG_ERR(SCAL,"{}, queueId2DccmAddr() failed for queue id {}", __FUNCTION__, core->qmanID);
        assert(0);
        return SCAL_FAILURE;
    }
    localMode = (qmanDccmAddr == core->dccmDevAddress);

    return SCAL_SUCCESS;
}

Scal_Gaudi3::Scal_Gaudi3(
    const int fd,
    const struct hlthunk_hw_ip_info & hw_ip,
    scal_arc_fw_config_handle_t fwCfg)
: Scal(fd, hw_ip, c_cores_nr, c_sync_managers_nr, SCHED_ARC_VIRTUAL_SOB_INDEX_COUNT,
       std::unique_ptr<DevSpecificInfo>(new Gaudi3Info))
, m_configChannel(fd)
{
    m_schedulerNr = c_scheduler_nr;
    if (fwCfg)
    {
        m_arc_fw_synapse_config = *(struct arc_fw_synapse_config_t*)fwCfg;
    }
    else
    {
        memset(&m_arc_fw_synapse_config, 0, sizeof(m_arc_fw_synapse_config));
        m_arc_fw_synapse_config.sync_scheme_mode = ARC_FW_GAUDI3_SYNC_SCHEME;
        const char * envVarValue = getenv(c_scal_sync_scheme_env_var_name);
        if (envVarValue)
        {
            m_arc_fw_synapse_config.sync_scheme_mode = (arc_fw_sync_scheme_t)std::stoul(envVarValue);
        }
        // enable cme by default
        // if in json cme is not enabled - cme_enable will be set to 0
        m_arc_fw_synapse_config.cme_enable = 1;
    }
}

Scal_Gaudi3::~Scal_Gaudi3()
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
    }

    delete m_configChannel.m_stream;
    for (auto& core : m_nicCores)
    {
        if (core)
        {
            delete core;
        }
    }
}

uint64_t Scal_Gaudi3::getCoreAuxOffset(const ArcCore * core)
{
    if (core->getAs<Scheduler>())
    {
        return mmHD0_ARC_FARM_ARC0_AUX_BASE - mmHD0_ARC_FARM_ARC0_DCCM0_BASE;
    }
    else
    {
        return mmHD0_TPC0_QM_ARC_AUX_BASE - mmHD0_TPC0_QM_DCCM_BASE;
    }
}

uint64_t Scal_Gaudi3::getCoreQmOffset()
{
    return mmHD0_TPC0_QM_BASE - mmHD0_TPC0_QM_DCCM_BASE;
}

int Scal_Gaudi3::streamSubmit(Stream *stream, const unsigned pi, const unsigned submission_alignment)
{
    // currently we support only buffer size of 65536

    //  Todo: streamSubmit is on the critical path. if needed, change to assert only
    LOG_TRACE(SCAL,"submit stream ({}) #{} pi={} {:#x} buffer_size={} submission_alignment={} on core {}",
        stream->name, stream->id, pi, pi, stream->coreBuffer->size, submission_alignment, stream->scheduler->name);

    // Gaudi3 auto fetcher requires pi to be continous, e.g not wrap around on buffer size
    if (!m_use_auto_fetcher)
    {
        uint32_t g2_pi = pi % stream->coreBuffer->size;// this will allow the user to treat Gaudi3 and Gaudi2 the same way
        writeLbwReg(stream->pi, (g2_pi/submission_alignment));
    }
    else
    {
        writeLbwReg(stream->pi, pi);
    }
    return SCAL_SUCCESS;
}

int Scal_Gaudi3::setStreamBuffer(Stream *stream, Buffer *buffer) const
{
    /* check for buffer size restrictions:
     minimal size is 2^16
     max size 2^24
     must be power of 2
     so quant must be ( TODO add assert for that 1/2/4/8/16/32/64/128/256)
     first stage FW supports only 2^16
     */
    if (!m_use_auto_fetcher && !buffer->pool->coreBase)
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

    if (buffer->size > c_core_counter_max_ccb_size)
    {
        LOG_ERR(SCAL,"{}, buffer size too big {}", __FUNCTION__, buffer->size);
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
    if (m_use_auto_fetcher)
    {
        // set up the auto fetcher registers
        uint64_t afBaseAddr = (uint64_t)stream->scheduler->afHostAddress;
        uint32_t offset = varoffsetof(gaudi3::block_arc_af_eng, af_host_stream_size[stream->id]);
        // stream size is calculated in DW (rather than bytes) and programmed as 1 lesser value
        // e.g.  CB_QUEUE_SIZE=2^(N+1) DWs   AF_HOST_CB_STREAM_Q_LOG_SIZE
        uint32_t logSize = (uint32_t) log2(buffer->size >> 2);
        writeLbwReg((volatile uint32_t*)(afBaseAddr + offset), (logSize - 1));// AF_HOST_STREAM_SIZE  (soconline HDCORE0-ARCFARM_CFG-HD0_ARC_FARM_ARC0_AF-AF_HOST_STREAM_SIZE)
        offset = varoffsetof(gaudi3::block_arc_af_eng, af_host_stream_base_l[stream->id]);
        uint64_t buf_addr = buffer->base + (uint64_t)buffer->pool->deviceBase;
        if (buf_addr % c_ccb_buffer_alignment != 0)
        {
            LOG_ERR(SCAL,"{}, buffer mapped address is not 128B aligned {}", __FUNCTION__, buf_addr);
            assert(0);
            return SCAL_INVALID_PARAM;
        }
        writeLbwReg((volatile uint32_t*)(afBaseAddr + offset), lower_32_bits(buf_addr));// AF_HOST_STREAM_BASE_L
        offset = varoffsetof(gaudi3::block_arc_af_eng, af_host_stream_base_h[stream->id]);
        writeLbwReg((volatile uint32_t*)(afBaseAddr + offset), upper_32_bits(buf_addr));// AF_HOST_STREAM_BASE_H
        //
        for(unsigned i=0;i<4;i++)
        {
            offset = varoffsetof(gaudi3::block_arc_af_eng, af_prio_weight[i]);
            writeLbwReg((volatile uint32_t*)(afBaseAddr + offset), 0x1);// af_prio_weight
        }
    }

    stream->coreBuffer = buffer;

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::setStreamPriority(Stream *stream, unsigned priority) const
{

    if ((priority != SCAL_HIGH_PRIORITY_STREAM) && (priority != SCAL_LOW_PRIORITY_STREAM))
    {
        LOG_ERR(SCAL,"{}, invalid priority {}", __FUNCTION__, priority);
        assert(0);
        return SCAL_FAILURE;
    }

    auto acpBlock = (gaudi3::block_arc_acp_eng*)stream->scheduler->acpHostAddress;
    writeLbwReg((volatile uint32_t*)&acpBlock->qsel_prio[stream->id], priority);
    stream->priority = priority;

    return SCAL_SUCCESS;
}

int Scal_Gaudi3::allocAndSetupPortDBFifo(ibv_context* ibv_ctxt)
{
    for (auto& cluster : m_nicClusters)
    {
        for (auto& core : cluster->engines)
        {
            G3NicCore * nicCore = core->getAs<G3NicCore>();
            for (int i = 0; i < 2; i++)
            {
                if (nicCore->portsMask.test(i))
                {
                    for (auto& [queueIndex, queue] : cluster->queues)
                    {
                        (void)queueIndex;
                        unsigned port = nicCore->portsMask.test(0) ? nicCore->ports[0] : nicCore->ports[1];
                        unsigned dbFifoMaxValue = port % 2 ? ODD_PORT_MAX_DB_FIFO : EVEN_PORT_MAX_DB_FIFO;
                        auto db_fifo_id = dbFifoMaxValue - queue.index;
                        LOG_TRACE(SCAL, "hlthunk_nic_alloc_user_db_fifo: port {} db_fifo_id {}", port, db_fifo_id);
                        int rc = hlthunk_nic_alloc_user_db_fifo(m_fd, port, &db_fifo_id);
                        // scal_assert_return(rc == 0, SCAL_FAILURE, "hlthunk_nic_alloc_user_db_fifo fd {} cluster {} nic {} port {} db_fifo_id {} faild rc = {} errno = {} {}", m_fd, cluster->name, nicCore->qman, port, db_fifo_id, rc, errno, std::strerror(errno));
                        if (rc != 0) LOG_WARN(SCAL, "hlthunk_nic_alloc_user_db_fifo fd {} cluster {} nic {} port {} db_fifo_id {} faild rc = {} errno = {} {}", m_fd, cluster->name, nicCore->qman, port, db_fifo_id, rc, errno, std::strerror(errno));
                        LOG_INFO(SCAL, "cluster {} queue {} core {} port {} : db_fifo_id {}", cluster->name, queue.index, nicCore->qman, port, db_fifo_id);

                        struct hlthunk_nic_user_db_fifo_set_in fifo_set_in;
                        fifo_set_in.port = port;
                        fifo_set_in.id = db_fifo_id;
                        fifo_set_in.base_sob_addr = queue.sobjBaseAddr;
                        fifo_set_in.num_sobs = c_sos_for_completion_group_credit_management;
                        fifo_set_in.mode = HL_NIC_DB_FIFO_TYPE_COLL_DIR_OPS_SHORT;
                        fifo_set_in.dir_dup_ports_mask = nicCore->portsMask.to_ulong();
                        LOG_INFO(SCAL, "hlthunk_nic_user_db_fifo_set_in: port {} id {} base_sob_addr {:#x}  queue.sobjBaseAddr {:#x} queue.sob_start_id {} num_sobs {} mode {} dir_dup_ports_mask 0b{:b}",
                            fifo_set_in.port, fifo_set_in.id, fifo_set_in.base_sob_addr, queue.sobjBaseAddr, queue.sobjBaseIndex ,fifo_set_in.num_sobs, fifo_set_in.mode, fifo_set_in.dir_dup_ports_mask);
                        struct hlthunk_nic_user_db_fifo_set_out fifo_set_out;
                        rc = hlthunk_nic_user_db_fifo_set(m_fd, &fifo_set_in, &fifo_set_out);
                        // scal_assert_return(rc == 0, SCAL_FAILURE, "hlthunk_nic_user_db_fifo_set fd {} faild rc = {} errno = {} {}", m_fd, rc, errno, std::strerror(errno));
                        if (rc != 0) LOG_WARN(SCAL, "hlthunk_nic_user_db_fifo_set fd {} faild rc = {} errno = {} {}", m_fd, rc, errno, std::strerror(errno));
                        LOG_TRACE(SCAL, "nic_user_db_fifo_set_out : ci_handle {} regs_handle {} regs_offset {} fifo_size {} fifo_bp_thresh {}", fifo_set_out.ci_handle, fifo_set_out.regs_handle, fifo_set_out.regs_offset, fifo_set_out.fifo_size, fifo_set_out.fifo_bp_thresh);
                        // scal_assert_return(fifo_set_out.fifo_size == c_nic_db_fifo_size, SCAL_FAILURE, "hlthunk_nic_user_db_fifo_set fd {} faild fifo_set_out.fifo_size {}", m_fd, fifo_set_out.fifo_size);
                        if (fifo_set_out.fifo_size != c_nic_db_fifo_size) LOG_WARN(SCAL, "hlthunk_nic_user_db_fifo_set fd {} faild fifo_set_out.fifo_size {}", m_fd, fifo_set_out.fifo_size);
                        // scal_assert_return(fifo_set_out.fifo_bp_thresh == c_nic_db_fifo_bp_treshold, SCAL_FAILURE, "hlthunk_nic_user_db_fifo_set fd {} faild fifo_set_out.fifo_bp_thresh {}", m_fd, fifo_set_out.fifo_bp_thresh);
                        if (fifo_set_out.fifo_bp_thresh != c_nic_db_fifo_bp_treshold) LOG_WARN(SCAL, "hlthunk_nic_user_db_fifo_set fd {} faild fifo_set_out.fifo_bp_thresh {}", m_fd, fifo_set_out.fifo_bp_thresh);
                    }
                    break;
                }
            }
        }
    }
    return SCAL_SUCCESS;
}

static bool shouldCreateDbFifo(int port_num, int dir_dup_mask, uint64_t nicsMask)
{
    // skip HCL disabled ports that are enabled by LKD
    unsigned port = port_num - 1;
    unsigned nicMacro = port / 2;
    unsigned port0 = nicMacro * 2;

    // get the current nic macro HCL mask
    unsigned nicMacroStatusMask = (nicsMask & (3 << port0)) >> port0 & 3;

    unsigned nicMacroDbFifosMask = dir_dup_mask;
    LOG_TRACE(SCAL, "port={}, nicMacro={}, nicMacroDbFifosMask={}", port, nicMacro, nicMacroDbFifosMask);

    // 2 ports that share the same db_fifo must be enabled or disabled together
    if ((nicMacroStatusMask & nicMacroDbFifosMask) != nicMacroDbFifosMask && (nicMacroStatusMask & nicMacroDbFifosMask) != 0)
    {
        LOG_ERR(SCAL, "ports that share the same db_fifo must be enabled or disabled together. port = {} hcl mask = {} db fifo mask = {}",
                port, nicMacroStatusMask, nicMacroDbFifosMask);
    }

    return nicMacroStatusMask & nicMacroDbFifosMask;
}

typedef struct hlibdv_usr_fifo* (*hlibdv_create_usr_fifo_tmp_fn)(struct ibv_context*          context,
                                                             struct hlibdv_usr_fifo_attr_tmp* attr);

int Scal_Gaudi3::allocAndSetupPortDBFifoV2(const scal_ibverbs_init_params* ibvInitParams,
                                           struct hlibdv_usr_fifo       ** createdFifoBuffers,
                                           uint32_t                      * createdFifoBuffersCount)
{
    unsigned nicUserDbFifoParamsCount = 0;
    getDbFifoParams_tmp(nullptr, &nicUserDbFifoParamsCount);
    if (*createdFifoBuffersCount == 0)
    {
        LOG_INFO(SCAL, "Set nic user db_fifo params count = {}", nicUserDbFifoParamsCount);
        *createdFifoBuffersCount = nicUserDbFifoParamsCount;
        return 0;
    }
    std::vector<hlibdv_usr_fifo_attr_tmp> attrs(nicUserDbFifoParamsCount);
    getDbFifoParams_tmp(attrs.data(), &nicUserDbFifoParamsCount);
    uint32_t fifoBuffersCount = 0;
    hlibdv_create_usr_fifo_tmp_fn hlibdv_create_usr_fifo_tmp = nullptr;
    hlibdv_create_usr_fifo_tmp = (hlibdv_create_usr_fifo_tmp_fn) dlsym(ibvInitParams->ibverbsLibHandle, "hlibdv_create_usr_fifo_tmp");
    if (hlibdv_create_usr_fifo_tmp == nullptr)
    {
        LOG_ERR(SCAL, "cannot load 'hlibdv_create_usr_fifo' from ibverbs library");
        return SCAL_INVALID_PARAM;
    }
    for (auto & attr : attrs)
    {
        if (shouldCreateDbFifo(attr.port_num, attr.dir_dup_mask, ibvInitParams->nicsMask))
        {
            fifoBuffersCount++;
            if (fifoBuffersCount > *createdFifoBuffersCount)
            {
                LOG_ERR(SCAL, "provided createdFifoBuffersCount {} is less than the actual number of fifoBuffers", *createdFifoBuffersCount);
                return SCAL_INVALID_PARAM;
            }

            LOG_DEBUG(SCAL, "hlibdv_create_usr_fifo nic: {}, sob_addr: 0x{:x}, num_sobs: {}, hint: {}, mode: {}, dup_mask: 0x{:x}",
                            attr.port_num - 1,
                            attr.base_sob_addr,
                            attr.num_sobs,
                            attr.usr_fifo_num_hint,
                            attr.usr_fifo_type,
                            attr.dir_dup_mask);
            const unsigned fifoBuffersIdx = fifoBuffersCount - 1;
            /* On success API returns a handle. Compare it with hlthunk_nic_user_db_fifo_set_out */
            createdFifoBuffers[fifoBuffersIdx] = hlibdv_create_usr_fifo_tmp(ibvInitParams->ibv_ctxt, &attr);

            if (createdFifoBuffers[fifoBuffersIdx] == nullptr)
            {
                LOG_ERR(SCAL, "hlibdv_create_usr_fifo() failed. nic: {} errno: {} - {}", attr.port_num - 1, errno, std::strerror(errno));
                typedef int (*hlibdv_destroy_usr_fifo_fn)(struct hlibdv_usr_fifo* usr_fifo);
                hlibdv_destroy_usr_fifo_fn hlibdv_destroy_usr_fifo = nullptr;
                hlibdv_destroy_usr_fifo = (hlibdv_destroy_usr_fifo_fn) dlsym(ibvInitParams->ibverbsLibHandle, "hlibdv_destroy_usr_fifo");
                if (hlibdv_destroy_usr_fifo == nullptr)
                {
                    LOG_ERR(SCAL, "failed to load hlibdv_destroy_usr_fifo");
                    return SCAL_FAILURE;
                }
                for (unsigned i = 0 ; i < fifoBuffersIdx; ++i)
                {
                    int err = hlibdv_destroy_usr_fifo(createdFifoBuffers[i]);
                    if (err)
                    {
                         LOG_ERR(SCAL, "hlibdv_destroy_usr_fifo failed. i: {} buffer: {} errno: {} - {}", i, createdFifoBuffers[i], errno, std::strerror(errno));
                    }
                    createdFifoBuffers[i] = nullptr;
                }
                return SCAL_FAILURE;
            }
        }
    }
    *createdFifoBuffersCount = fifoBuffersCount;
    return 0;
}

int Scal_Gaudi3::getDbFifoParams_tmp(hlibdv_usr_fifo_attr_tmp* nicUserDbFifoParams, unsigned* nicUserDbFifoParamsCount)
{
    unsigned dbFifos = 0;
    for (auto& cluster : m_nicClusters)
    {
        for (auto& core : cluster->engines)
        {
            G3NicCore * nicCore = core->getAs<G3NicCore>();
            for (int i = 0; i < 2; i++)
            {
                if (nicCore->portsMask.test(i))
                {
                    for (auto& [queueIndex, queue] : cluster->queues)
                    {
                        (void)queueIndex;
                        dbFifos++;
                        if (*nicUserDbFifoParamsCount == 0)
                        {
                            // only count the number of db_fifos
                            continue;
                        }
                        if (dbFifos > *nicUserDbFifoParamsCount)
                        {
                            // nicUserDbFifoParams is not enough
                            LOG_ERR(SCAL, "provided nicUserDbFifoParamsCount is less than the actual number");
                            return SCAL_INVALID_PARAM;
                        }
                        struct hlibdv_usr_fifo_attr_tmp* usrFifoAttr = nicUserDbFifoParams + dbFifos - 1;
                        unsigned port = nicCore->portsMask.test(0) ? nicCore->ports[0] : nicCore->ports[1];
                        unsigned dbFifoMaxValue = port % 2 ? ODD_PORT_MAX_DB_FIFO : EVEN_PORT_MAX_DB_FIFO;
                        auto db_fifo_id = dbFifoMaxValue - queue.index;
                        usrFifoAttr->port_num = port + 1; // port 0 is reserved
                        usrFifoAttr->usr_fifo_type = HLIBDV_USR_FIFO_TYPE_COLL_DIR_OPS_SHORT;
                        usrFifoAttr->base_sob_addr = queue.sobjBaseAddr;
                        usrFifoAttr->num_sobs = c_sos_for_completion_group_credit_management;
                        usrFifoAttr->usr_fifo_num_hint = db_fifo_id;
                        usrFifoAttr->dir_dup_mask = nicCore->portsMask.to_ulong();
                        LOG_TRACE(SCAL, "getDbFifoParams fd {} cluster {} nic {} port {} "
                                        "usr_fifo_attr.port_num {} usr_fifo_attr.mode {} "
                                        "usr_fifo_attr.base_sob_addr {:#x} usr_fifo_attr.num_sobs {}"
                                        "usr_fifo_num_hint {} dir_dup_mask {}",
                                        m_fd, cluster->name, nicCore->qman, port,
                                        usrFifoAttr->port_num, usrFifoAttr->usr_fifo_type,
                                        usrFifoAttr->base_sob_addr, usrFifoAttr->num_sobs,
                                        usrFifoAttr->usr_fifo_num_hint, usrFifoAttr->dir_dup_mask);
                    }
                    break;
                }
            }
        }
    }
    *nicUserDbFifoParamsCount = dbFifos;
    return SCAL_SUCCESS;
}
