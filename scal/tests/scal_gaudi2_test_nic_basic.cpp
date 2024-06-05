#include "scal_test_nic_basic.hpp"
#include "gaudi2_arc_sched_packets.h"
#include <cstdint>

#define UNSUPPORTED_CMD 0x1F
#define WRAPS 2

const uint32_t alloc_barrier_nic_opcode[] =
{
    [g2fw::SCHED_TYPE_COMPUTE] = UNSUPPORTED_CMD,
    [g2fw::SCHED_TYPE_GARBAGE_REDUCTION] = g2fw::SCHED_GC_REDUCTION_ARC_CMD_ALLOC_NIC_BARRIER,
    [g2fw::SCHED_TYPE_SCALE_OUT_RECV] = g2fw::SCHED_SCALEOUT_RECV_ARC_CMD_ALLOC_NIC_BARRIER,
    [g2fw::SCHED_TYPE_SCALE_OUT_SEND] = g2fw::SCHED_SCALEOUT_SEND_ARC_CMD_ALLOC_NIC_BARRIER,
    [g2fw::SCHED_TYPE_SCALE_UP_RECV] = g2fw::SCHED_SCALEUP_RECV_ARC_CMD_ALLOC_NIC_BARRIER,
    [g2fw::SCHED_TYPE_SCALE_UP_SEND] = g2fw::SCHED_SCALEUP_SEND_ARC_CMD_ALLOC_NIC_BARRIER
};

const uint32_t lbw_write_opcode[] =
{
    [g2fw::SCHED_TYPE_COMPUTE] = g2fw::SCHED_COMPUTE_ARC_CMD_LBW_WRITE,
    [g2fw::SCHED_TYPE_GARBAGE_REDUCTION] = g2fw::SCHED_GC_REDUCTION_ARC_CMD_LBW_WRITE,
    [g2fw::SCHED_TYPE_SCALE_OUT_RECV] = g2fw::SCHED_SCALEOUT_RECV_ARC_CMD_LBW_WRITE,
    [g2fw::SCHED_TYPE_SCALE_OUT_SEND] = g2fw::SCHED_SCALEOUT_SEND_ARC_CMD_LBW_WRITE,
    [g2fw::SCHED_TYPE_SCALE_UP_RECV] = g2fw::SCHED_SCALEUP_RECV_ARC_CMD_LBW_WRITE,
    [g2fw::SCHED_TYPE_SCALE_UP_SEND] = g2fw::SCHED_SCALEUP_SEND_ARC_CMD_LBW_WRITE
};

const uint32_t nop_opcode[] =
{
    [g2fw::SCHED_TYPE_COMPUTE] = g2fw::SCHED_COMPUTE_ARC_CMD_NOP,
    [g2fw::SCHED_TYPE_GARBAGE_REDUCTION] = g2fw::SCHED_GC_REDUCTION_ARC_CMD_NOP,
    [g2fw::SCHED_TYPE_SCALE_OUT_RECV] = g2fw::SCHED_SCALEOUT_RECV_ARC_CMD_NOP,
    [g2fw::SCHED_TYPE_SCALE_OUT_SEND] = g2fw::SCHED_SCALEOUT_SEND_ARC_CMD_NOP,
    [g2fw::SCHED_TYPE_SCALE_UP_RECV] = g2fw::SCHED_SCALEUP_RECV_ARC_CMD_NOP,
    [g2fw::SCHED_TYPE_SCALE_UP_SEND] = g2fw::SCHED_SCALEUP_SEND_ARC_CMD_NOP
};

uint8_t SCALGaudi2NicTest::getSchedNopOpcode(uint32_t sched_type)
{
    return nop_opcode[sched_type];
}

uint8_t SCALGaudi2NicTest::getSchedAllocNicBarrierOpcode(uint32_t sched_type)
{
    return alloc_barrier_nic_opcode[sched_type];
}

uint8_t SCALGaudi2NicTest::getSchedLbwWriteOpcode(uint32_t sched_type)
{
    return lbw_write_opcode[sched_type];
}

const TestNicConfig testConfigs[] =
{
    {"scaleup_receive", "scaleup_receive0", g2fw::SCHED_TYPE_SCALE_UP_RECV},
    {"scaleup_send", "scaleup_send0", g2fw::SCHED_TYPE_SCALE_UP_SEND},
    {"scaleout_receive", "scaleout_receive0", g2fw::SCHED_TYPE_SCALE_OUT_RECV},
    {"scaleout_send", "scaleout_send0", g2fw::SCHED_TYPE_SCALE_OUT_SEND},
    {"network_garbage_collector_and_reduction", "network_garbage_collector_and_reduction0", g2fw::SCHED_TYPE_GARBAGE_REDUCTION}
};

INSTANTIATE_TEST_SUITE_P(, SCALGaudi2NicTest, testing::Values(
    TestNicConfig(testConfigs[4])
)
);

TEST_P_CHKDEV(SCALGaudi2NicTest, basic_test, {GAUDI2})
{
    nicBasicTest();
}

TEST_P_CHKDEV(SCALGaudi2NicTest, wrap_test, {GAUDI2})
{
    nicBasicTest(WRAPS);
}

TEST_F_CHKDEV(SCALGaudi2NicTest, multi_sched_test, {GAUDI2})
{
    nicMultiSchedTest(testConfigs);
}