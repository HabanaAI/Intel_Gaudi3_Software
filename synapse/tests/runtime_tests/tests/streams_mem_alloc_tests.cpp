#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "habana_global_conf_runtime.h"
#include "global_conf_test_setter.h"
#include "runtime/scal/common/recipe_launcher/mapped_mem_mgr.hpp"

class SynStreamsMemAllocTests : public SynBaseTest
{
public:
    SynStreamsMemAllocTests() : SynBaseTest()
    {
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }
};

REGISTER_SUITE(SynStreamsMemAllocTests, ALL_TEST_PACKAGES);

TEST_F_SYN(SynStreamsMemAllocTests, get_dev_attribute_mem_size)
{
    uint64_t        expectedMappedSize;
    uint64_t        streamsReturnTotalSize;
    const uint32_t  numOfCreationStreams = 4;
    synStreamHandle genericStreams[numOfCreationStreams];

    const synDeviceAttribute attributes[]  = {DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE};
    const size_t             nofAttributes = 1;

    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            expectedMappedSize = GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP.value() * 1024 *
                                     GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP.value() +
                                 GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP.value() * 1024 *
                                     GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP.value();
            break;
        }
        case synDeviceGaudi2:
        case synDeviceGaudi3:
        {
            const uint32_t numOfComputeStreams = GCFG_GAUDI3_SINGLE_DIE_CHIP.value() ? 2 : 3;
            expectedMappedSize                 = numOfComputeStreams * MappedMemMgr::testingOnlyInitialMappedSize();
            break;
        }
        default:
        {
            ASSERT_TRUE(false) << "Unsupported device type " << m_deviceType;
            break;
        }
    }

    TestDevice device(m_deviceType);

    ASSERT_EQ(synDeviceGetAttribute(&streamsReturnTotalSize, attributes, nofAttributes, device.getDeviceId()),
              synSuccess)
        << "Failed to get device attribute";
    ASSERT_EQ(expectedMappedSize, streamsReturnTotalSize);

    // create generic streams, make sure used mapped memory is not changed
    for (uint32_t idx = 0; idx < numOfCreationStreams; idx++)
    {
        ASSERT_EQ(synStreamCreateGeneric(&genericStreams[idx], device.getDeviceId(), 0), synSuccess)
            << "Failed to create stream compute (streamIndex = " << idx << ")";
        ASSERT_EQ(synDeviceGetAttribute(&streamsReturnTotalSize, attributes, nofAttributes, device.getDeviceId()),
                  synSuccess)
            << "Failed to get device attribute";
        ASSERT_EQ(expectedMappedSize, streamsReturnTotalSize);
    }

    // Clean up
    for (uint32_t idx = 0; idx < numOfCreationStreams; idx++)
    {
        ASSERT_EQ(synStreamDestroy(genericStreams[idx]), synSuccess) << "Failed to destroy stream";
    }
}

TEST_F_SYN(SynStreamsMemAllocTests, compute_stream_allocation_address_high_restriction, {synDeviceGaudi2, synDeviceGaudi3})
{
    GlobalConfTestSetter expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GlobalConfTestSetter glblMemSize("HBM_GLOBAL_MEM_SIZE_MEGAS", "1400");

    synDeviceId deviceId;
    // The test is setting the compute stream HBM allocation to 1.4GB, we have 3 compute streams,
    // meaning we will try to allocate more than 4GB(0x1 0000 0000) for compute streams and the high 32bits will change.
    // On this case we will try to allocate again with 4GB alignment by padding the previous 4GB address range.
    ASSERT_EQ(synDeviceAcquireByDeviceType(&deviceId, m_deviceType), synSuccess) << "Failed to acquire device";
}