#include <cstdint>
#include <bitset>

#include <infiniband/verbs.h>
#include <infiniband/hlibdv.h>

#include "scal_test_nic_basic.hpp"
#include "gaudi3_arc_sched_packets.h"
#include <limits>
#define UNSUPPORTED_CMD 0x1F
#define WRAPS 2

const uint32_t alloc_barrier_nic_opcode[] =
{
    [g3fw::SCHED_TYPE_COMPUTE] = UNSUPPORTED_CMD,
    [g3fw::SCHED_TYPE_GARBAGE_REDUCTION] = g3fw::SCHED_GC_REDUCTION_ARC_CMD_ALLOC_NIC_BARRIER,
    [g3fw::SCHED_TYPE_SCALE_OUT_RECV] = g3fw::SCHED_SCALEOUT_RECV_ARC_CMD_ALLOC_NIC_BARRIER,
    [g3fw::SCHED_TYPE_SCALE_OUT_SEND] = g3fw::SCHED_SCALEOUT_SEND_ARC_CMD_ALLOC_NIC_BARRIER,
    [g3fw::SCHED_TYPE_SCALE_UP_RECV] = g3fw::SCHED_SCALEUP_RECV_ARC_CMD_ALLOC_NIC_BARRIER,
    [g3fw::SCHED_TYPE_SCALE_UP_SEND] = g3fw::SCHED_SCALEUP_SEND_ARC_CMD_ALLOC_NIC_BARRIER
};

const uint32_t lbw_write_opcode[] =
{
    [g3fw::SCHED_TYPE_COMPUTE] = g3fw::SCHED_COMPUTE_ARC_CMD_LBW_WRITE,
    [g3fw::SCHED_TYPE_GARBAGE_REDUCTION] = g3fw::SCHED_GC_REDUCTION_ARC_CMD_LBW_WRITE,
    [g3fw::SCHED_TYPE_SCALE_OUT_RECV] = g3fw::SCHED_SCALEOUT_RECV_ARC_CMD_LBW_WRITE,
    [g3fw::SCHED_TYPE_SCALE_OUT_SEND] = g3fw::SCHED_SCALEOUT_SEND_ARC_CMD_LBW_WRITE,
    [g3fw::SCHED_TYPE_SCALE_UP_RECV] = g3fw::SCHED_SCALEUP_RECV_ARC_CMD_LBW_WRITE,
    [g3fw::SCHED_TYPE_SCALE_UP_SEND] = g3fw::SCHED_SCALEUP_SEND_ARC_CMD_LBW_WRITE
};

const uint32_t nop_opcode[] =
{
    [g3fw::SCHED_TYPE_COMPUTE] = g3fw::SCHED_COMPUTE_ARC_CMD_NOP,
    [g3fw::SCHED_TYPE_GARBAGE_REDUCTION] = g3fw::SCHED_GC_REDUCTION_ARC_CMD_NOP,
    [g3fw::SCHED_TYPE_SCALE_OUT_RECV] = g3fw::SCHED_SCALEOUT_RECV_ARC_CMD_NOP,
    [g3fw::SCHED_TYPE_SCALE_OUT_SEND] = g3fw::SCHED_SCALEOUT_SEND_ARC_CMD_NOP,
    [g3fw::SCHED_TYPE_SCALE_UP_RECV] = g3fw::SCHED_SCALEUP_RECV_ARC_CMD_NOP,
    [g3fw::SCHED_TYPE_SCALE_UP_SEND] = g3fw::SCHED_SCALEUP_SEND_ARC_CMD_NOP
};

uint8_t SCALGaudi3NicTest::getSchedNopOpcode(uint32_t sched_type)
{
    return nop_opcode[sched_type];
}

uint8_t SCALGaudi3NicTest::getSchedAllocNicBarrierOpcode(uint32_t sched_type)
{
    return alloc_barrier_nic_opcode[sched_type];
}

uint8_t SCALGaudi3NicTest::getSchedLbwWriteOpcode(uint32_t sched_type)
{
    return lbw_write_opcode[sched_type];
}

void SCALGaudi3NicTest::initNics()
{
    ASSERT_EQ(m_ibvLibFuncs.load(), true) << "ibv lib loading failed";
    // get ibverbs device
    struct ibv_device** dev_list;
    int num_of_device = -1;
    dev_list = m_ibvLibFuncs.hlibv_get_device_list(&num_of_device);
    ASSERT_GE(num_of_device, 1) << "hlibv_get_device_list failed: num_of_device=" << num_of_device;
    struct ibv_device* ibdev = nullptr;
    char ibdev_name[16];
    snprintf(ibdev_name, sizeof(ibdev_name), "hlib_%d", m_hw_ip.module_id);
    for (int i = 0; i < num_of_device; i++)
    {
        if (strstr(m_ibvLibFuncs.hlibv_get_device_name(dev_list[i]), ibdev_name))
        {
            ibdev = dev_list[i];
            break;
        }
    }
    ASSERT_NE(ibdev, nullptr) << "dev_list[0] is null";

    struct hlthunk_nic_get_ports_masks_out portsMask;
    int ret = hlthunk_nic_get_ports_masks(m_fd, &portsMask);
    ASSERT_EQ(ret, 0) << "hlthunk_nic_get_ports_masks fail, return: " << ret << " errno: " << errno << " - " << std::strerror(errno);
    struct hlibdv_ucontext_attr attr = {.ports_mask = portsMask.ports_mask << 1, .core_fd = m_fd};
    m_ibctx = m_ibvLibFuncs.hlibdv_open_device(ibdev, &attr);
    ASSERT_NE(m_ibctx, nullptr) << "hlibdv_open_device failed: ibctx is null";
    m_ibvLibFuncs.hlibv_free_device_list(dev_list);

    // set ports ex
    for (unsigned port = 0; port < 24; port++)
    {
        struct hlibdv_port_ex_attr_tmp port_attr;
        memset(&port_attr, 0, sizeof(port_attr));
        port_attr.port_num = port + 1;
        port_attr.caps |= HLIBDV_PORT_CAP_ADVANCED;
        std::vector<hlibdv_wq_array_type> hlibdv_wq_array_types = {
            hlibdv_wq_array_type::HLIBDV_WQ_ARRAY_TYPE_GENERIC,
            hlibdv_wq_array_type::HLIBDV_WQ_ARRAY_TYPE_COLLECTIVE,
            hlibdv_wq_array_type::HLIBDV_WQ_ARRAY_TYPE_SCALE_OUT_COLLECTIVE
        };
        for (hlibdv_wq_array_type array_type : hlibdv_wq_array_types)
        {
            auto& port_wq_arr_attr = port_attr.wq_arr_attr[array_type];
            port_wq_arr_attr.max_num_of_wqs = 16;
            port_wq_arr_attr.max_num_of_wqes_in_wq = 128;
            port_wq_arr_attr.mem_id = m_hw_ip.dram_enabled ? HLIBDV_MEM_DEVICE : HLIBDV_MEM_HOST;
            port_wq_arr_attr.swq_granularity = HLIBDV_SWQE_GRAN_32B; //hlna->ms_en ? HLIBDV_SWQE_GRAN_64B : HLIBDV_SWQE_GRAN_32B;
        }
        int rc = m_ibvLibFuncs.hlibdv_set_port_ex_tmp(m_ibctx, &port_attr);
        ASSERT_EQ(rc, 0) << "hlibdv_set_port_ex failed on port=" << port << " with rc=" << rc << " errno=" << errno << " error=" << std::strerror(errno);
    }
    ASSERT_EQ(scal_nics_db_fifos_init_and_allocV2(m_scalHandle, 0, m_hlibdvUsrFifos.data(), &m_nicUserDbFifoParamsCount), SCAL_SUCCESS) << "scal_nics_db_fifos_init_and_allocV2 failed to acquire number of db_fifo params";
    ASSERT_LE(m_nicUserDbFifoParamsCount, 24u * 2 * 2 /*24 ports * 2 (up/out) * 2 (send/receive))*/);

    m_hlibdvUsrFifos.resize(m_nicUserDbFifoParamsCount);
    scal_ibverbs_init_params initParams;
    initParams.ibv_ctxt = m_ibctx;
    initParams.ibverbsLibHandle = m_ibvLibFuncs.libHandle();
    initParams.nicsMask = std::numeric_limits<uint64_t>::max();
    ASSERT_EQ(scal_nics_db_fifos_init_and_allocV2(m_scalHandle, &initParams, m_hlibdvUsrFifos.data(), &m_nicUserDbFifoParamsCount), SCAL_SUCCESS) << "scal_nics_db_fifos_init_and_allocV2 failed";
    m_hlibdvUsrFifos.resize(m_nicUserDbFifoParamsCount);
}

void SCALGaudi3NicTest::releaseNics()
{
    int rc = 0;
    if (m_ibvLibFuncs.isLoaded())
    {
        for (unsigned i = 0; i < m_nicUserDbFifoParamsCount; i++)
        {
            rc = m_ibvLibFuncs.hlibdv_destroy_usr_fifo(m_hlibdvUsrFifos[i]);
            ASSERT_EQ(rc, 0) << "hlibdv_destroy_usr_fifo failed with rc " << rc << " errno " << errno << " " << std::strerror(errno);
        }
        rc = m_ibvLibFuncs.hlibv_close_device(m_ibctx);
        ASSERT_EQ(rc, 0) << "hlibv_close_device failed with rc " << rc << " errno " << errno << " " << std::strerror(errno);
    }
}

const TestNicConfig testConfigs[] =
{
    {"scaleup_receive", "scaleup_receive0", g3fw::SCHED_TYPE_SCALE_UP_RECV},
    {"scaleup_send", "scaleup_send0", g3fw::SCHED_TYPE_SCALE_UP_SEND},
    {"scaleout_receive", "scaleout_receive0", g3fw::SCHED_TYPE_SCALE_OUT_RECV},
    {"scaleout_send", "scaleout_send0", g3fw::SCHED_TYPE_SCALE_OUT_SEND},
    {"network_garbage_collector_and_reduction", "network_garbage_collector_and_reduction0", g3fw::SCHED_TYPE_GARBAGE_REDUCTION}
};

INSTANTIATE_TEST_SUITE_P(, SCALGaudi3NicTest, testing::Values(
    TestNicConfig(testConfigs[4])
)
);

TEST_P_CHKDEV(SCALGaudi3NicTest, basic_test, {GAUDI3})
{
    nicBasicTest();
}

TEST_P_CHKDEV(SCALGaudi3NicTest, wrap_test, {GAUDI3})
{
    nicBasicTest(WRAPS);
}

TEST_F_CHKDEV(SCALGaudi3NicTest, multi_sched_test, {GAUDI3})
{
    nicMultiSchedTest(testConfigs);
}
