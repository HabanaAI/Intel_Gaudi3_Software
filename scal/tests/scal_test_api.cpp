#include <gtest/gtest.h>
#include <math.h>
#include "scal.h"
#include "scal_basic_test.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"

class ScalApiTests : public SCALTest {};

void testAllocFree(scal_pool_handle_t pool, unsigned poolSize,unsigned alignment = 0)
{
    // Allocate Buffer on the Host
    int rc = 0;
    #define KB 1024
    #define MB (1024*1024)
    std::vector<unsigned> sizes={100,1000,25356,78999,154666,256*KB,784*KB,MB, 10*MB,52*MB,186*MB,546*MB};
    std::vector<scal_buffer_handle_t> buffHandlesArr;
    unsigned szArr = sizes.size();

    for (unsigned loop=0;loop<1000;loop++)
    {
        scal_memory_pool_info info;
        rc = scal_pool_get_info(pool, &info);
        ASSERT_EQ(rc, 0);
        LOG_TRACE(SCAL,"Before loop {} on pool {} totalSize {} freeSize {}", loop, info.name, info.totalSize, info.freeSize);
        std::vector<unsigned> sSizes = sizes;
        std::random_shuffle ( sSizes.begin(), sSizes.end() );
        buffHandlesArr.resize(szArr);
        uint64_t totalMemory =  info.totalSize - info.freeSize;// on some pools, there were allocations before we got here (scal_init, etc.)
        uint64_t numAllocated = 0;
        for(unsigned i=0;i<szArr;i++)
        {
            totalMemory += sSizes[i];
            if (totalMemory > poolSize)
                break;
            LOG_TRACE(SCAL,"Allocating {} from pool",sSizes[i]);
            if (!alignment)
                rc = scal_allocate_buffer(pool, sSizes[i], &buffHandlesArr[i]);
            else
                rc = scal_allocate_aligned_buffer(pool, sSizes[i], alignment,&buffHandlesArr[i]);
            ASSERT_EQ(rc, 0) << "allocating " << sSizes[i] << " when pool free size is " << info.freeSize << " failed";
            numAllocated++;
            // get device data buffer
            scal_buffer_info_t deviceDataBuffInfo;
            rc = scal_buffer_get_info(buffHandlesArr[i], &deviceDataBuffInfo);
            ASSERT_EQ(rc, 0);
            if(alignment)
            {
                // device_address is always set
                if (deviceDataBuffInfo.device_address % alignment != 0)
                {
                    LOG_ERR(SCAL,"scal_allocate_aligned_buffer with alignment={:#x} returned non aligned address={:#x}",
                        alignment,deviceDataBuffInfo.device_address);
                    assert(0);
                }
            }
            rc = scal_pool_get_info(pool, &info);
            ASSERT_EQ(rc, 0);
            LOG_TRACE(SCAL,"totalMemory {} pool totalSize {} freeSize {}", totalMemory, info.totalSize, info.freeSize);
            ASSERT_EQ(totalMemory, (unsigned)(info.totalSize - info.freeSize));
        }
    #ifdef NDEBUG
        auto oldLevel = hl_logger::getLoggingLevel(scal::LoggerTypes::SCAL);
        hl_logger::setLoggingLevel(scal::LoggerTypes::SCAL, HLLOG_LEVEL_CRITICAL);
        scal_buffer_handle_t hostBuffHandle = nullptr;
    // this suppose to fail (no more memory left), but in debug it will just fail on assert and we can't continue
        LOG_TRACE(SCAL,"Trying to allocate more than pool size");
        rc = scal_allocate_buffer(pool, info.freeSize + sSizes[0], &hostBuffHandle);
        ASSERT_EQ(!rc, 0) << "allocating " << info.freeSize << " + " << sSizes[0] << " should have failed";
        LOG_TRACE(SCAL,"Trying to allocate 0 bytes");
        rc = scal_allocate_buffer(pool, 0, &hostBuffHandle);
        ASSERT_EQ(!rc, 0) << "allocating 0 should have failed";
        hl_logger::setLoggingLevel(scal::LoggerTypes::SCAL, oldLevel);
    #endif
        for(unsigned i=0;i<numAllocated;i++)
        {
            LOG_TRACE(SCAL,"Deallocating {} from pool",sSizes[i]);
            rc = scal_free_buffer(buffHandlesArr[i]);
            ASSERT_EQ(rc, 0);
            buffHandlesArr[i] = nullptr;
        }
    }
}

TEST_F_CHKDEV(ScalApiTests, allocFree,{ALL})
{
    /*
        test scal_allocate_buffer
             scal_allocate_aligned_buffer
             scal_buffer_get_info
             scal_free_buffer
    */
    scal_handle_t      scalHandle, scalHandle1;
    scal_pool_handle_t memHostPoolHandle,deviceMemPoolHandle;
    int rc = 0;

    srand((unsigned) time(nullptr));

    int scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    const char  configFilePath[] = ":/default.json";
    std::string confFileStr      = getConfigFilePath(configFilePath);
    const char* confFile         = confFileStr.c_str();
    printf("Loading scal with config=%s\n",confFile);
    //
    //   Init Scal
    //
    rc = scal_init(scalFd, confFile, &scalHandle, nullptr);
    ASSERT_EQ(rc, 0);
    // try again with the same fd, should return the same handle
    rc = scal_init(scalFd, confFile, &scalHandle1, nullptr);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(scalHandle, scalHandle1);
    // test the get_handle_from_fd api
    scalHandle1 = nullptr;
    rc          = scal_get_handle_from_fd(scalFd, &scalHandle1);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(scalHandle, scalHandle1);

    //  Get Host Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "host_shared", &memHostPoolHandle);
    ASSERT_EQ(rc, 0);

    scal_memory_pool_info hostPoolInfo;
    rc = scal_pool_get_info(memHostPoolHandle, &hostPoolInfo);

    //  Get Device Memory Pool handle
    rc = scal_get_pool_handle_by_name(scalHandle, "global_hbm", &deviceMemPoolHandle);
    ASSERT_EQ(rc, 0);

    LOG_DEBUG(SCAL,"Allocate Buffers on the HostShared pool, size {}", hostPoolInfo.freeSize);
    testAllocFree(memHostPoolHandle, hostPoolInfo.freeSize);

    LOG_DEBUG(SCAL,"Allocate Buffers on the device global hbm pool");
    testAllocFree(deviceMemPoolHandle,1024*MB);

    // test alignment
    std::vector<unsigned> alignments={0x100,0x1000,0x10000,0x100000};
    for (unsigned i=0;i<alignments.size();i++)
    {
        unsigned alignment = alignments[i];
        LOG_DEBUG(SCAL,"Allocate Buffers on the HostShared pool. Alignment={}",alignment);
        testAllocFree(memHostPoolHandle, hostPoolInfo.freeSize);

        LOG_DEBUG(SCAL,"Allocate Buffers on the device global hbm pool. Alignment={}",alignment);
        testAllocFree(deviceMemPoolHandle,1024*MB,alignment);
    }



   // tear down

   scal_destroy(scalHandle);
   hlthunk_close(scalFd);
}

void testHostFenceCounters(scal_handle_t scalHandle, bool enableIsr)
{
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            std::string host_counter_name =  "host_fence_counters_" + std::to_string(i) + std::to_string(j);
            scal_host_fence_counter_handle_t hostFenceCounterHandle;
            ASSERT_EQ(scal_get_host_fence_counter_handle_by_name(scalHandle, host_counter_name.c_str(), &hostFenceCounterHandle), 0);

            ASSERT_EQ(scal_host_fence_counter_enable_isr(hostFenceCounterHandle, enableIsr), 0);

            scal_host_fence_counter_info_t hostFenceCounterInfo;
            ASSERT_EQ(scal_host_fence_counter_get_info(hostFenceCounterHandle, &hostFenceCounterInfo), 0);
            ASSERT_EQ(std::string(hostFenceCounterInfo.name),host_counter_name);
            ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, 0, 0), 0);


            scal_sm_info_t smInfo;
            ASSERT_EQ(scal_get_sm_info(scalHandle, hostFenceCounterInfo.sm, &smInfo), 0);
            const unsigned soIdx = hostFenceCounterInfo.so_index;
            scal_write_mapped_reg(&smInfo.objs[soIdx], 3);
            scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000001);
            scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000001);

            ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, 5, SCAL_FOREVER), 0);
            scal_write_mapped_reg(&smInfo.objs[soIdx], 50);
            scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000050);
            scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000001);

            ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, 101, SCAL_FOREVER), 0);
        }
    }
}

TEST_F_CHKDEV(ScalApiTests, hostFenceCounter,{ALL})
{
    scal_handle_t      scalHandle;
    int scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    const char  configFilePath[] = ":/default.json";
    printf("Loading scal with config=%s\n",configFilePath);
    ASSERT_EQ(scal_init(scalFd, configFilePath, &scalHandle, nullptr), 0);

    testHostFenceCounters(scalHandle, false);
    testHostFenceCounters(scalHandle, true);
    testHostFenceCounters(scalHandle, false);
    testHostFenceCounters(scalHandle, true);

    // tear down
    scal_destroy(scalHandle);
    hlthunk_close(scalFd);
}


void testHostFenceCountersExtreme(scal_handle_t scalHandle, bool enableIsr)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            std::string host_counter_name =  "host_fence_counters_" + std::to_string(i) + std::to_string(j);
            scal_host_fence_counter_handle_t hostFenceCounterHandle;
            ASSERT_EQ(scal_get_host_fence_counter_handle_by_name(scalHandle, host_counter_name.c_str(), &hostFenceCounterHandle), 0);

            ASSERT_EQ(scal_host_fence_counter_enable_isr(hostFenceCounterHandle, enableIsr), 0);

            scal_host_fence_counter_info_t hostFenceCounterInfo;
            ASSERT_EQ(scal_host_fence_counter_get_info(hostFenceCounterHandle, &hostFenceCounterInfo), 0);
            ASSERT_EQ(std::string(hostFenceCounterInfo.name),host_counter_name);
            ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, 0, 0), 0);


            scal_sm_info_t smInfo;
            ASSERT_EQ(scal_get_sm_info(scalHandle, hostFenceCounterInfo.sm, &smInfo), 0);
            const unsigned soIdx = hostFenceCounterInfo.so_index;
            scal_write_mapped_reg(&smInfo.objs[soIdx], 0);
            for (unsigned n = 0; n < 2048; n++)
            {
                scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000001);
            }
            ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, 2048, SCAL_FOREVER), 0);
            for (unsigned n = 0; n < 2048; n++)
            {
                for (unsigned m = 0; m < n; m++)
                {
                    scal_write_mapped_reg(&smInfo.objs[soIdx], 0x80000001);
                }
                ASSERT_EQ(scal_host_fence_counter_wait(hostFenceCounterHandle, n, SCAL_FOREVER), 0);
            }
        }
    }
}

TEST_F_CHKDEV(ScalApiTests, DISABLED_ASIC_hostFenceCounter,{ALL})
{
    scal_handle_t      scalHandle;
    int scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    ASSERT_GE(scalFd, 0);
    const char  configFilePath[] = ":/default.json";
    printf("Loading scal with config=%s\n",configFilePath);
    ASSERT_EQ(scal_init(scalFd, configFilePath, &scalHandle, nullptr), 0);

    testHostFenceCountersExtreme(scalHandle, false);
    testHostFenceCountersExtreme(scalHandle, true);

    // tear down
    scal_destroy(scalHandle);
    hlthunk_close(scalFd);
}