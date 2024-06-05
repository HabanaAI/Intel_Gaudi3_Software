#include "scal_test_nic_basic.hpp"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <thread>

#define NUM_NIC_SCHEDS 5
#define DWORD_SIZE 4 // in bytes
#define WL_SIZE 256 // in bytes
#define BUFFER_MIN_SIZE (64*1024)

void SCALNicTest::SetUp()
{
    SCALTestDevice::SetUp();
    int rc;
    const char* confFile    = ""; // use default json
    printf("Loading scal with config=%s\n",confFile);
    rc = scal_init(m_fd, confFile, &m_scalHandle, nullptr);
    ASSERT_EQ(rc, 0);
    initNics();
}

void SCALNicTest::TearDown()
{
    releaseNics();
    if (m_scalHandle != 0)
    {
        scal_destroy(m_scalHandle);
    }
    SCALTestDevice::TearDown();
}

void SCALNicTest::nicBasicTest(uint32_t wrapCount)
{

    /*
     * Send workload to a single nic scheduler.
     * We expect a single signal for each completion.
     */
    TestNicConfig testConfig = GetParam();
    sendNopToNicSched(&testConfig, m_scalHandle, 1, "network_completion_queue_external_00",  wrapCount, true);

}

void SCALNicTest::nicMultiSchedTest(const TestNicConfig testConfigs[])
{
    bool isMaster = true;
    std::vector<std::thread> threads;
    for(uint32_t i = 0; i < NUM_NIC_SCHEDS ; i++)
    {
        /*
         * Send workload to NUM_NIC_SCHEDS nic schedulers simultaneously.
         * We expect NUM_NIC_SCHEDS signals for each completion.
         */
        threads.push_back(
            std::thread(&SCALNicTest::sendNopToNicSched, this, &testConfigs[i], std::ref(m_scalHandle), NUM_NIC_SCHEDS, "network_completion_queue_internal_00", 10, isMaster));
        isMaster = false;
    }

    for(auto& thread : threads)
    {
        thread.join();
    }
}

void SCALNicTest::sendNopToNicSched(const TestNicConfig *testConfig, const scal_handle_t &scalHandle, uint32_t expectedSignals, std::string cgName, uint32_t wrapCount, bool isMaster)
{
    int rc;

    /*********************************************************************************************
     * Allocate buffer, get stream, get cg info                                                  *
     ********************************************************************************************/

    scal_pool_handle_t memPool;
    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memPool);
    ASSERT_EQ(rc, 0);

    scal_buffer_handle_t ctrlBuff;
    uint32_t hostCyclicBufferSize = BUFFER_MIN_SIZE;
    rc = scal_allocate_buffer(memPool, hostCyclicBufferSize, &ctrlBuff);
    ASSERT_EQ(rc, 0);

    scal_stream_handle_t stream;
    rc = scal_get_stream_handle_by_name(scalHandle, testConfig->stream_name, &stream);
    ASSERT_EQ(rc, 0);

    rc = scal_stream_set_commands_buffer(stream, ctrlBuff);
    ASSERT_EQ(rc, 0);

    scal_stream_info_t streamInfo;
    rc = scal_stream_get_info(stream, &streamInfo);
    ASSERT_EQ(rc, 0);

    scal_buffer_info_t bufferInfo;
    rc = scal_buffer_get_info(ctrlBuff, &bufferInfo);
    ASSERT_EQ(rc, 0);

    scal_comp_group_handle_t cgHandle;
    rc = scal_get_completion_group_handle_by_name(scalHandle, cgName.c_str(), &cgHandle);
    ASSERT_EQ(rc, 0);

    scal_completion_group_infoV2_t cgInfo;
    rc = scal_completion_group_get_infoV2(cgHandle, &cgInfo);
    ASSERT_EQ(rc, 0);
    if (cgInfo.force_order)
    {
        expectedSignals += 1; // when the cq is configured as force_order, the so's are initialized as 1, not 0
    }

    /*********************************************************************************************
     * WorkLoad:
     * - NOP + padding
     * - ALLOC NIC BARRIER with requested sob = 1 since we want to use each time a single sob for a completion
     * - LBW WRITE to the SOB *instead of nic) in order to reach C_MAX for completion
     *
     * SOBs:
     * - For each iteration we use a dedicated sob, starting from sob_base in cg info
     * - For next iteration we increse sob id by 1
     * - All Schedulers (master + slaves) will signal to the same sob
     * - target completion will increased by 1 each iteration
     *
     * Distributed completion group:
     * - A single master with slaves, all use the same completion group
     * - Each scheduler holds a copy of the CG object, therefore all schedulers should send alloc nic barrier
     *   in order to be sync with the current credit amount
     * - Requested sob in alloc nic is decremented from the total credits number
     *   For example: total credits is 512 and we get requested sob = 10 then credits should be set to 502
     *   In order for slaves to be sync with the value we must send alloc nic barrier also from them
     ********************************************************************************************/

    // Fill commands in CCB
    auto buildPkt = createBuildPkts(getScalDeviceType());

    // Create CCB
    char* currCommandBufferLoc = (char*)bufferInfo.host_address;
    uint32_t sobLbwBaseAddr    = cgInfo.sm_base_addr + cgInfo.sos_base * 4;
    uint32_t padding           = (WL_SIZE - getPktSize<AllocNicBarrier>(buildPkt) - getPktSize<LbwWritePkt>(buildPkt));
    uint32_t sobLbwAddr        = 0;
    uint64_t target            = 0;
    uint32_t pi                = 0;

    do
    {
        // Signal to the next sob for each completion
        sobLbwAddr = sobLbwBaseAddr + ((target % cgInfo.sos_num) * 4);

        fillPktForceOpcodeNoSize<NopCmdPkt>(buildPkt, currCommandBufferLoc, getSchedNopOpcode(testConfig->schedType), padding);
        currCommandBufferLoc += padding;
        currCommandBufferLoc += fillPkt<AllocNicBarrier>(buildPkt, currCommandBufferLoc, getSchedAllocNicBarrierOpcode(testConfig->schedType), cgInfo.index_in_scheduler, 1);

        // Master will increment by CMAX - number of ecpected signals and also increment by 1
        if (isMaster)
        {
            currCommandBufferLoc += fillPktForceOpcode<LbwWritePkt>(buildPkt, currCommandBufferLoc ,getSchedLbwWriteOpcode(testConfig->schedType), sobLbwAddr, 0x80000001 + COMP_SYNC_GROUP_CMAX_TARGET - expectedSignals, false);
        }
        else
        {
            currCommandBufferLoc += fillPktForceOpcode<LbwWritePkt>(buildPkt, currCommandBufferLoc, getSchedLbwWriteOpcode(testConfig->schedType), sobLbwAddr, 0x80000001, false);
        }

        // Submit work to NIC SCHEDULER on selected stream
        pi += WL_SIZE;
        rc = scal_stream_submit(stream, pi, streamInfo.submission_alignment);
        ASSERT_EQ(rc, 0);

        // Wait for complition
        target++;
        rc = scal_completion_group_wait(cgHandle, target, SCAL_FOREVER);
        ASSERT_EQ(rc, 0);

        if ((pi % hostCyclicBufferSize) == 0)
        {
            currCommandBufferLoc = (char*)bufferInfo.host_address;
            wrapCount--;
        }
    } while (wrapCount > 0);

    rc = scal_free_buffer(ctrlBuff);
    ASSERT_EQ(rc, 0);
}
