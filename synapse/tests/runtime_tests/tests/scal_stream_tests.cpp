#include "syn_base_test.hpp"
#include "synapse_api.h"
#include "syn_singleton.hpp"
#include "global_conf_test_setter.h"

#include "runtime/common/device/device_mem_alloc.hpp"

#include "runtime/scal/common/entities/scal_dev.hpp"
#include "runtime/scal/common/entities/scal_stream_base.hpp"
#include "runtime/scal/common/stream_base_scal.hpp"
#include "runtime/scal/common/stream_compute_scal.hpp"
#include "runtime/common/device/device_common.hpp"
#include "runtime/common/streams/stream.hpp"
#include "scal_monitor_base.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_recipe_tpc.hpp"

class ScalStreamTest : public SynBaseTest
{
public:
    ScalStreamTest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }
    void testStreamCcbOccupancyWatermark(synDeviceType deviceType);
    void testUserTxPdmaStreamCcbOccupancy(synDeviceType deviceType);
};

REGISTER_SUITE(ScalStreamTest, ALL_TEST_PACKAGES);

static void testDmaSyncInSingleStream(synDeviceId deviceId, synDeviceType deviceType)
{
    enum HostAddrEnum
    {
        Source1 = 0,
        Source2,
        Destination1,
        Destination2,
        HostAddrMax
    };

    // Execution
    uint64_t* pHostAddress[HostAddrMax] {nullptr};
    uint32_t  bufferSize = 1000;
    uint32_t  dataSize   = bufferSize * sizeof(uint64_t);
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        synStatus status = synHostMalloc(deviceId, dataSize, 0, (void**)&pHostAddress[iter]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for iter (" << iter << ")";
    }

    for (uint32_t i = 0; i < bufferSize; i++)
    {
        pHostAddress[Source1][i]      = 0x1;
        pHostAddress[Source2][i]      = 0x2;
        pHostAddress[Destination1][i] = 0x0;
        pHostAddress[Destination2][i] = 0x0;
    }

    uint64_t  deviceAddress = 0;
    synStatus status        = synDeviceMalloc(deviceId, dataSize, 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";

    DevMemoryAlloc* pAllocator = DevMemoryAllocScal::debugGetLastConstructedAllocator();
    ASSERT_NE(pAllocator, nullptr) << "failed to get allocator";

    uint64_t pMappedAddress[HostAddrMax] {0};
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        const eMappingStatus translateStatus =
            pAllocator->getDeviceVirtualAddress(true, pHostAddress[iter], dataSize, &pMappedAddress[iter], nullptr);
        ASSERT_EQ(translateStatus, HATVA_MAPPING_STATUS_FOUND) << "Failed to translate iter (" << iter << ")";
    }

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();
    ASSERT_NE(pDev, nullptr) << "failed to get device";

    ScalStreamBaseInterface* pStreamBaseDmaDownUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_DOWN_USER);
    ScalStreamCopyInterface* pStreamDmaDownUser     = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDownUser);
    ASSERT_NE(pStreamDmaDownUser, nullptr) << "Device stream dma-down (user) is nullptr";
    //
    ScalStreamBaseInterface* pStreamBaseDmaUpUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_UP);
    ScalStreamCopyInterface* pStreamDmaUpUser     = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaUpUser);
    ASSERT_NE(pStreamDmaUpUser, nullptr) << "Device stream dma-up (user) is nullptr";

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};

    ScalLongSyncObject longSo(LongSoEmpty);
    status = pStreamDmaDownUser->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                         {{pMappedAddress[Source1], deviceAddress, dataSize}},
                                         false,
                                         false,
                                         0,
                                         longSo,
                                         0,
                                         memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUpUser->longSoWaitOnDevice(longSo, true);
    ASSERT_EQ(status, synSuccess) << "fenceWait failed";

    status = pStreamDmaUpUser->memcopy(ResourceStreamType::USER_DMA_UP,
                                       {{deviceAddress, pMappedAddress[Destination1], dataSize}},
                                       false,
                                       false,
                                       0,
                                       longSo,
                                       0,
                                       memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaDownUser->longSoWaitOnDevice(longSo, true);
    ASSERT_EQ(status, synSuccess) << "fenceWait failed";

    status = pStreamDmaDownUser->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                         {{pMappedAddress[Source2], deviceAddress, dataSize}},
                                         false,
                                         true,
                                         0,
                                         longSo,
                                         0,
                                         memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUpUser->longSoWaitOnDevice(longSo, true);
    ASSERT_EQ(status, synSuccess) << "fenceWait failed";

    status = pStreamDmaUpUser->memcopy(ResourceStreamType::USER_DMA_UP,
                                       {{deviceAddress, pMappedAddress[Destination2], dataSize}},
                                       false,
                                       true,
                                       0,
                                       longSo,
                                       0,
                                       memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    const uint64_t timeoutMicroSec = (uint64_t)((int64_t)SCAL_FOREVER);
    status                         = pStreamDmaUpUser->longSoWait(longSo, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "syncStreamEng failed";

    status = synDeviceFree(deviceId, deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory (" << deviceAddress << ")";

    // Validation
    for (uint32_t i = 0; i < bufferSize; i++)
    {
        ASSERT_EQ(pHostAddress[Source1][i], pHostAddress[Destination1][i])
            << "Compare failed i = " << i << " *pDestination1 (" << pHostAddress[Destination1][i] << ")";
        ASSERT_EQ(pHostAddress[Source2][i], pHostAddress[Destination2][i])
            << "Compare failed i = " << i << " *pDestination2 (" << pHostAddress[Destination2][i] << ")";
    }

    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        status = synHostFree(deviceId, pHostAddress[iter], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Host memory (" << iter << ")";
    }
}

void ScalStreamTest::testStreamCcbOccupancyWatermark(synDeviceType deviceType)
{
    GlobalConfTestSetter expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");
    GlobalConfTestSetter ccbSampleFlag("ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK", "true");

    TestDevice device(deviceType);
    TestStream testStream = device.createStream();

    TestRecipeTpc tpcRecipe(deviceType);
    tpcRecipe.generateRecipe();

    TestLauncher       launcher(device);
    RecipeLaunchParams tpcLaunchParams =
        launcher.createRecipeLaunchParams(tpcRecipe, {TensorInitOp::RANDOM_WITH_NEGATIVE, 0});

    // Copy 2GB to device and then execute many jobs.
    // Since the stream is waiting for the dma job to finish, the compute queue CCB will fill up
    uint64_t              dmaSize     = 1024 * 1024 * 1024 * 2ULL;  // 2GB
    TestHostBufferMalloc  hostAlloc   = device.allocateHostBuffer(dmaSize, 0);
    TestDeviceBufferAlloc deviceAlloc = device.allocateDeviceBuffer(dmaSize, 0);

    uint64_t devAddr  = deviceAlloc.getBuffer();
    void*    hostAddr = hostAlloc.getBuffer();
    testStream.memcopyAsync((uint64_t)hostAddr, dmaSize, devAddr, HOST_TO_DRAM);

    for (int i = 1; i < 1000; ++i)
    {
        testStream.launch(tpcLaunchParams.getSynLaunchTensorInfoVec().data(),
                          tpcRecipe.getTensorInfoVecSize(),
                          tpcLaunchParams.getWorkspace(),
                          tpcRecipe.getRecipe(),
                          0);
    }

    // Validate the CCB occupancy watermark
    DeviceCommon* deviceCommon = (DeviceCommon*)(_SYN_SINGLETON_INTERNAL->getDevice().get());
    Stream*       streamHandle {};
    {
        auto streamSptr = deviceCommon->loadAndValidateStream(testStream, __FUNCTION__);
        ASSERT_NE(streamSptr, nullptr) << "Failed to load Stream";
        streamHandle = streamSptr.get();
    }
    QueueInterface* pQueueInterface;
    streamHandle->testGetQueueInterface(QUEUE_TYPE_COMPUTE, pQueueInterface);
    ScalStreamCopyInterface* scalStreamInterface =
        reinterpret_cast<QueueBaseScalCommon*>(pQueueInterface)->getScalStream();
    ScalStreamBase* scalStreamBase = reinterpret_cast<ScalStreamBase*>(scalStreamInterface);

    uint64_t expectedCcbOccupancyWatermark = 15000;
    uint64_t ccbOccupancyWatermark         = scalStreamBase->getStreamCyclicBufferOccupancyWatermark();
    ASSERT_GE(ccbOccupancyWatermark, expectedCcbOccupancyWatermark)
        << "Stream CCB occupancy watermark is " << ccbOccupancyWatermark << " bytes, expected to be greater than "
        << expectedCcbOccupancyWatermark << " bytes";
}

void ScalStreamTest::testUserTxPdmaStreamCcbOccupancy(synDeviceType deviceType)
{
    GlobalConfTestSetter expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");

    // Define very small CCB-chunk size, so a submission will result multiple chunks usage
    GlobalConfTestSetter ccbBufferSize("SET_HOST_CYCLIC_BUFFER_SIZE", "64"); // 1KB
    GlobalConfTestSetter ccbChunksAmount("SET_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT", "128");

    TestDevice device(deviceType);
    TestStream testStream = device.createStream();

    // Get device
    DeviceCommon* deviceCommon = (DeviceCommon*)(_SYN_SINGLETON_INTERNAL->getDevice().get());
    // Get Synapse-Stream
    Stream* synapseStream {};
    auto streamSptr = deviceCommon->loadAndValidateStream(testStream, __FUNCTION__);
    ASSERT_NE(streamSptr, nullptr) << "Failed to load Stream";
    synapseStream = streamSptr.get();
    // Get TX Queue
    QueueInterface* pTxUserQueueInterface;
    synapseStream->testGetQueueInterface(QUEUE_TYPE_COPY_HOST_TO_DEVICE, pTxUserQueueInterface);
    ScalStreamCopyInterface* txUserScalStreamInterface =
        reinterpret_cast<QueueBaseScalCommon*>(pTxUserQueueInterface)->getScalStream();
    ScalStreamBase* txUserScalStreamBase = reinterpret_cast<ScalStreamBase*>(txUserScalStreamInterface);

    // Intentionally requesting the same copy multiple times
    unsigned copyAmount = 10000;
    uint64_t dmaSize    = 1024 * 2ULL; // 2KB
    std::vector<TestHostBufferMalloc>  hostAllocDB;
    std::vector<TestDeviceBufferAlloc> deviceAllocDB;
    std::vector<uint64_t>              hostAddrDB;
    std::vector<uint64_t>              devAddrDB;
    std::vector<uint64_t>              copySizeDB;
    for (unsigned i = 0; i < copyAmount; i++)
    {
        hostAllocDB.emplace_back(device.allocateHostBuffer(dmaSize, 0));
        deviceAllocDB.emplace_back(device.allocateDeviceBuffer(dmaSize, 0));
        hostAddrDB.emplace_back((uint64_t) hostAllocDB.back().getBuffer());
        devAddrDB.emplace_back(deviceAllocDB.back().getBuffer());
        copySizeDB.emplace_back(dmaSize);
    }

    testStream.memcopyAsyncMultiple(hostAddrDB.data(), copySizeDB.data(), devAddrDB.data(), HOST_TO_DRAM, copyAmount);
    StreamCyclicBufferBase* txUserCcb = txUserScalStreamBase->getStreamCyclicBuffer();
    ASSERT_EQ(txUserCcb->testOnlyCheckCcbConsistency(), true) << "TX-User CCB long-SO is not consistent";
}

TEST_F_SYN(ScalStreamTest, testDmaSyncInSingleStream)
{
    TestDevice device(m_deviceType);
    testDmaSyncInSingleStream(device.getDeviceId(), m_deviceType);
}

TEST_F_SYN(ScalStreamTest, testDmaCompletionTargetInSingleStream)
{
    enum HostAddrEnum
    {
        Source1 = 0,
        Source2,
        Destination1,
        Destination2,
        HostAddrMax
    };

    // Execution
    TestDevice device(m_deviceType);

    uint64_t* pHostAddress[HostAddrMax] {nullptr};
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        auto status = synHostMalloc(device.getDeviceId(), sizeof(uint64_t), 0, (void**)&pHostAddress[iter]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for iter (" << iter << ")";
    }

    *pHostAddress[Source1]      = 0x1;
    *pHostAddress[Source2]      = 0x2;
    *pHostAddress[Destination1] = 0x0;
    *pHostAddress[Destination2] = 0x0;

    uint64_t deviceAddress = 0;
    auto     status        = synDeviceMalloc(device.getDeviceId(), sizeof(uint64_t), 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";

    DevMemoryAlloc* pAllocator = DevMemoryAllocScal::debugGetLastConstructedAllocator();
    ASSERT_NE(pAllocator, nullptr) << "failed to get allocator";

    uint64_t pMappedAddress[HostAddrMax] {0};
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        const eMappingStatus translateStatus = pAllocator->getDeviceVirtualAddress(true,
                                                                                   pHostAddress[iter],
                                                                                   sizeof(uint64_t),
                                                                                   &pMappedAddress[iter],
                                                                                   nullptr);
        ASSERT_EQ(translateStatus, HATVA_MAPPING_STATUS_FOUND) << "Failed to translate iter (" << iter << ")";
    }

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();
    ASSERT_NE(pDev, nullptr) << "failed to get device";
    ScalStreamBaseInterface* pStreamBaseDmaDownUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_DOWN_USER);
    ScalStreamCopyInterface* pStreamDmaDownUser     = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDownUser);
    ASSERT_NE(pStreamDmaDownUser, nullptr) << "Device stream dma-down (user) is nullptr";
    //
    ScalStreamBaseInterface* pStreamBaseDmaUpUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_UP);
    ScalStreamCopyInterface* pStreamDmaUpUser     = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaUpUser);
    ASSERT_NE(pStreamDmaUpUser, nullptr) << "Device stream dma-up (user) is nullptr";

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};

    ScalLongSyncObject longSoDmaDown(LongSoEmpty);
    ScalLongSyncObject longSoDmaUp(LongSoEmpty);
    status = pStreamDmaDownUser->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                         {{pMappedAddress[Source1], deviceAddress, sizeof(uint64_t)}},
                                         false,
                                         false,
                                         0,
                                         longSoDmaDown,
                                         0,
                                         memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUpUser->memcopy(ResourceStreamType::USER_DMA_UP,
                                       {{deviceAddress, pMappedAddress[Destination1], sizeof(uint64_t)}},
                                       false,
                                       false,
                                       0,
                                       longSoDmaUp,
                                       0,
                                       memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaDownUser->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                         {{pMappedAddress[Source2], deviceAddress, sizeof(uint64_t)}},
                                         false,
                                         true,
                                         0,
                                         longSoDmaDown,
                                         0,
                                         memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUpUser->memcopy(ResourceStreamType::USER_DMA_UP,
                                       {{deviceAddress, pMappedAddress[Destination2], sizeof(uint64_t)}},
                                       false,
                                       true,
                                       0,
                                       longSoDmaUp,
                                       0,
                                       memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    const uint64_t timeoutMicroSec = (uint64_t)((int64_t)SCAL_FOREVER);
    //
    status = pStreamDmaDownUser->longSoWait(longSoDmaDown, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "syncStreamEng failed";
    status = pStreamDmaUpUser->longSoWait(longSoDmaUp, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "syncStreamEng failed";

    // We do not validate the values since we do not sync. We do check that completionTarget reaches the expected value

    status = synDeviceFree(device.getDeviceId(), deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory (" << deviceAddress << ")";

    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        status = synHostFree(device.getDeviceId(), pHostAddress[iter], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Host memory (" << iter << ")";
    }
}

/**
 * Copy from host to device, then from device to device, and then from device back to host, and then validates the
 * values match
 */
static void copyAndValidateData(unsigned int                  batchSize,
                                std::vector<uint64_t*>&       hostPtrs,
                                const std::vector<uint64_t*>& devPtrs1,
                                const std::vector<uint64_t*>& devPtrs2,
                                const std::vector<uint64_t>&  sizes,
                                const unsigned char           initialTestValue,
                                const std::vector<uint64_t>&  pMappedAddress,
                                ScalStreamCopyInterface*      pUploadStream,
                                ScalStreamCopyInterface*      pDownloadStream,
                                ScalStreamCopyInterface*      pDevToDevStream,
                                ResourceStreamType            uploadResourceStreamType,
                                ResourceStreamType            downloadResourceStreamType,
                                ResourceStreamType            devToDevResourceStreamType)
{
    synStatus status;

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};

    // setup host to device copy params
    internalMemcopyParams params;
    for (unsigned i = 0; i < batchSize; ++i)
    {
        params.push_back({.src = (uint64_t)pMappedAddress[i], .dst = (uint64_t)devPtrs1[i], .size = sizes[i]});
    }
    // copy host to device
    ScalLongSyncObject longSo(LongSoEmpty);
    status = pDownloadStream->memcopy(downloadResourceStreamType, params, false, true, 0, longSo, 0, memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    // wait on device for copy to complete
    const uint64_t timeoutMicroSec = (uint64_t)SCAL_FOREVER;
    //
    status = pDownloadStream->longSoWait(longSo, timeoutMicroSec, __FUNCTION__);

    // setup mem copy device to device params
    params.clear();
    for (unsigned i = 0; i < batchSize; ++i)
    {
        params.push_back({.src = (uint64_t)devPtrs1[i], .dst = (uint64_t)devPtrs2[i], .size = sizes[i]});
    }

    // copy device to device
    status = pDevToDevStream->memcopy(devToDevResourceStreamType, params, false, true, 0, longSo, 0, memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    // wait for copy to complete
    status = pDevToDevStream->longSoWait(longSo, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "syncStreamEng failed";

    // scrub host memory
    for (unsigned i = 0; i < batchSize; ++i)
    {
        memset(hostPtrs[i], 0xFF, sizes[i]);
    }

    // setup mem copy device to host params
    params.clear();
    // on simulator d2h is very slow - limit it to 1 MB
    const unsigned maxD2HSize = 1024 * 1024;
    for (unsigned i = 0; i < batchSize; ++i)
    {
        uint64_t sizeToCopy = sizes[i];
        if (sizes[i] > 2 * maxD2HSize)
        {
            sizeToCopy = maxD2HSize;
        }
        params.push_back({.src = (uint64_t)devPtrs2[i], .dst = (uint64_t)pMappedAddress[i], .size = sizeToCopy});
        if (sizes[i] > 2 * maxD2HSize)
        {
            uint64_t offset = sizes[i] - maxD2HSize;
            params.push_back({.src  = (uint64_t)devPtrs2[i] + offset,
                              .dst  = (uint64_t)pMappedAddress[i] + offset,
                              .size = maxD2HSize});
        }
    }
    // copy device to host
    status = pUploadStream->memcopy(uploadResourceStreamType, params, false, true, 0, longSo, 0, memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    // wait for copy to complete
    status = pUploadStream->longSoWait(longSo, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "syncStreamEng failed";

    // compare results with expected value
    unsigned char  cmpBuf[1024];
    const uint64_t cmpBufSize = sizeof(cmpBuf);
    for (unsigned i = 0; i < batchSize; ++i)
    {
        memset(cmpBuf, initialTestValue + i, cmpBufSize);
        uint64_t sizeToCmpInBeginning = std::min(cmpBufSize, sizes[i]);
        ASSERT_EQ(memcmp(cmpBuf, hostPtrs[i], sizeToCmpInBeginning), 0);
        if (sizes[i] < sizeToCmpInBeginning)
        {
            uint64_t sizeToCmpInEnd = std::min(cmpBufSize, sizes[i]);
            ASSERT_EQ(memcmp(cmpBuf, hostPtrs[i] + sizes[i] - sizeToCmpInEnd, sizeToCmpInEnd), 0);
        }
    }
}

static void testDmaCommandsTransfer(synDeviceType deviceType,
                                    unsigned      batchSize,
                                    unsigned      largeItemIdx  = -1,
                                    uint64_t      largeItemSize = 0x100000000ull)
{
    // Execution
    TestDevice device(deviceType);

    std::vector<uint64_t*> hostPtrs(batchSize, nullptr);
    std::vector<uint64_t*> devPtrs1(batchSize, nullptr);
    std::vector<uint64_t*> devPtrs2(batchSize, nullptr);
    std::vector<uint64_t>  sizes;
    std::vector<uint64_t>  pMappedAddress(batchSize, 0);

    DevMemoryAlloc* pAllocator = DevMemoryAllocScal::debugGetLastConstructedAllocator();
    ASSERT_NE(pAllocator, nullptr) << "failed to get allocator";

    const unsigned char initialTestValue = 55;
    // uint64_t         val              = initialTestValue;
    // allocating memory on host for each copy param
    for (unsigned i = 0; i < batchSize; ++i)
    {
        auto allocSize = (i == largeItemIdx ? largeItemSize : sizeof(uint64_t));
        sizes.push_back(allocSize);

        auto status = synHostMalloc(device.getDeviceId(), allocSize, 0, (void**)&hostPtrs[i]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory";
        memset(hostPtrs[i], initialTestValue + i, allocSize);
        // allocating memory on device for each copy param
        uint64_t ptr;
        status      = synDeviceMalloc(device.getDeviceId(), allocSize, 0, 0, &ptr);
        devPtrs1[i] = (uint64_t*)ptr;
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";
        // allocating additional memory on device for each copy param

        status      = synDeviceMalloc(device.getDeviceId(), allocSize, 0, 0, &ptr);
        devPtrs2[i] = (uint64_t*)ptr;
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";

        // mapping host memory
        const eMappingStatus translateStatus =
            pAllocator->getDeviceVirtualAddress(true, hostPtrs[i], allocSize, &pMappedAddress[i], nullptr);
        ASSERT_EQ(translateStatus, HATVA_MAPPING_STATUS_FOUND) << "Failed to translate iter (" << i << ")";
    }

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();
    ASSERT_NE(pDev, nullptr) << "failed to get device";

    ScalStreamBaseInterface* pStreamBaseDmaDownUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_DOWN_USER);
    ScalStreamCopyInterface* pStreamDmaDown         = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDownUser);
    ASSERT_NE(pStreamDmaDown, nullptr) << "Device stream dma-down is nullptr";
    //
    ScalStreamBaseInterface* pStreamBaseDmaUpUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_UP);
    ScalStreamCopyInterface* pStreamDmaUp         = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaUpUser);
    ASSERT_NE(pStreamDmaUp, nullptr) << "Device stream dma-up is nullptr";
    //
    ScalStreamBaseInterface* pStreamBaseDmaDevToDev = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DEV_TO_DEV);
    ScalStreamCopyInterface* pStreamDmaDevToDev     = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDevToDev);
    ASSERT_NE(pStreamDmaUp, nullptr) << "Device stream dma-up is nullptr";
    //
    const ComputeCompoundResources* pComputeCompoundResources = nullptr;
    pDev->debugGetCreatedComputeResources(pComputeCompoundResources);
    ASSERT_NE(pComputeCompoundResources, nullptr) << "Device Compute's compound resources is nullptr";
    ASSERT_NE(pComputeCompoundResources->m_pTxCommandsStream, nullptr) << "Device stream synapse dma-down is nullptr";
    ASSERT_NE(pComputeCompoundResources->m_pDev2DevCommandsStream, nullptr)
        << "Device stream synapse dev-to-dev is nullptr";

    ResourceStreamType dmaUpTypes[]       = {ResourceStreamType::SYNAPSE_DMA_UP, ResourceStreamType::USER_DMA_UP};
    ResourceStreamType dmaDownTypes[]     = {ResourceStreamType::SYNAPSE_DMA_DOWN, ResourceStreamType::USER_DMA_DOWN};
    ResourceStreamType dmaDevToDevTypes[] = {ResourceStreamType::SYNAPSE_DEV_TO_DEV,
                                             ResourceStreamType::USER_DEV_TO_DEV};

    // test for each combination of down, d2d & up DMA-types
    for (auto dmaUp : dmaUpTypes)
    {
        for (auto dmaDown : dmaDownTypes)
        {
            for (auto dmaDevToDev : dmaDevToDevTypes)
            {
                copyAndValidateData(
                    batchSize,
                    hostPtrs,
                    devPtrs1,
                    devPtrs2,
                    sizes,
                    initialTestValue,
                    pMappedAddress,
                    (dmaUp == ResourceStreamType::SYNAPSE_DMA_UP) ? pComputeCompoundResources->m_pRxCommandsStream
                                                                  : pStreamDmaUp,
                    (dmaDown == ResourceStreamType::SYNAPSE_DMA_DOWN) ? pComputeCompoundResources->m_pTxCommandsStream
                                                                      : pStreamDmaDown,
                    (dmaDevToDev == ResourceStreamType::SYNAPSE_DEV_TO_DEV)
                        ? pComputeCompoundResources->m_pDev2DevCommandsStream
                        : pStreamDmaDevToDev,
                    dmaUp,
                    dmaDown,
                    dmaDevToDev);
            }
        }
    }

    // We do not validate the values since we do not sync. We do check that completionTarget reaches the expected value

    for (uint64_t* devPtr : devPtrs1)
    {
        auto status = synDeviceFree(device.getDeviceId(), (uint64_t)devPtr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    }

    for (uint64_t* devPtr : devPtrs2)
    {
        auto status = synDeviceFree(device.getDeviceId(), (uint64_t)devPtr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    }

    for (uint64_t* hostPtr : hostPtrs)
    {
        auto status = synHostFree(device.getDeviceId(), hostPtr, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Host memory";
    }
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransfer2Items)
{
    testDmaCommandsTransfer(m_deviceType, 2);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransfer5Items)
{
    testDmaCommandsTransfer(m_deviceType, 5);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransfer5ItemsWithLarge_0)
{
    testDmaCommandsTransfer(m_deviceType, 5, 0, 0x80000);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransfer5ItemsWithLarge_4)
{
    testDmaCommandsTransfer(m_deviceType, 5, 4, 0x80000);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransfer5ItemsWithLarge_2)
{
    testDmaCommandsTransfer(m_deviceType, 5, 2, 0x80000);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransferOneMore4GB_ASIC_CI, {synTestPackage::ASIC})
{
    testDmaCommandsTransfer(m_deviceType, 10, 1, 0x100000800ull);
}

TEST_F_SYN(ScalStreamTest, testDmaBatchTransferManyItems)
{
    testDmaCommandsTransfer(m_deviceType, 1000, 2, 0x800);
}

TEST_F_SYN(ScalStreamTest, testDmaSyncBetweenTwoStreams)
{
    TestDevice device(m_deviceType);

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_MECH_LONG_SO,
                                                                .m_workCompletionAddress = 0,
                                                                .m_workCompletionValue   = 0};

    enum HostAddrEnum
    {
        Source1 = 0,
        Source2,
        Destination1,
        Destination2,
        HostAddrMax
    };

    // Execution
    uint64_t* pHostAddress[HostAddrMax] {nullptr};
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        auto status = synHostMalloc(device.getDeviceId(), sizeof(uint64_t), 0, (void**)&pHostAddress[iter]);
        ASSERT_EQ(status, synSuccess) << "Could not allocate host memory for iter (" << iter << ")";
    }

    *pHostAddress[Source1]      = 0x1;
    *pHostAddress[Source2]      = 0x2;
    *pHostAddress[Destination1] = 0x0;
    *pHostAddress[Destination2] = 0x0;

    uint64_t deviceAddress = 0;
    auto     status        = synDeviceMalloc(device.getDeviceId(), sizeof(uint64_t), 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";

    DevMemoryAlloc* pAllocator = DevMemoryAllocScal::debugGetLastConstructedAllocator();
    ASSERT_NE(pAllocator, nullptr) << "failed to get allocator";

    uint64_t pMappedAddress[HostAddrMax] {0};
    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        const eMappingStatus translateStatus = pAllocator->getDeviceVirtualAddress(true,
                                                                                   pHostAddress[iter],
                                                                                   sizeof(uint64_t),
                                                                                   &pMappedAddress[iter],
                                                                                   nullptr);
        ASSERT_EQ(translateStatus, HATVA_MAPPING_STATUS_FOUND) << "Failed to translate iter (" << iter << ")";
    }

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();

    ScalStreamBaseInterface* pStreamBaseDmaDownUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_DOWN_USER);
    ScalStreamCopyInterface* pStreamDmaDown         = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDownUser);
    ASSERT_NE(pStreamDmaDown, nullptr) << "Device stream dma-down is nullptr";
    //
    ScalStreamBaseInterface* pStreamBaseDmaUpUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_UP);
    ScalStreamCopyInterface* pStreamDmaUp         = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaUpUser);
    ASSERT_NE(pStreamDmaUp, nullptr) << "Device stream dma-up is nullptr";

    ScalLongSyncObject longSoDown(LongSoEmpty);
    ScalLongSyncObject longSoUp(LongSoEmpty);
    status = pStreamDmaDown->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                     {{pMappedAddress[Source1], deviceAddress, sizeof(uint64_t)}},
                                     false,
                                     false,
                                     0,
                                     longSoDown,
                                     0,
                                     memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUp->longSoWaitOnDevice(longSoDown, true);
    ASSERT_EQ(status, synSuccess) << "longSoWaitOnDevice failed";

    status = pStreamDmaUp->memcopy(ResourceStreamType::USER_DMA_UP,
                                   {{deviceAddress, pMappedAddress[Destination1], sizeof(uint64_t)}},
                                   false,
                                   false,
                                   0,
                                   longSoUp,
                                   0,
                                   memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaDown->longSoWaitOnDevice(longSoUp, true);
    ASSERT_EQ(status, synSuccess) << "longSoWaitOnDevice failed";

    status = pStreamDmaDown->memcopy(ResourceStreamType::USER_DMA_DOWN,
                                     {{pMappedAddress[Source2], deviceAddress, sizeof(uint64_t)}},
                                     false,
                                     true,
                                     0,
                                     longSoDown,
                                     0,
                                     memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    status = pStreamDmaUp->longSoWaitOnDevice(longSoDown, true);
    ASSERT_EQ(status, synSuccess) << "longSoWaitOnDevice failed";

    status = pStreamDmaUp->memcopy(ResourceStreamType::USER_DMA_UP,
                                   {{deviceAddress, pMappedAddress[Destination2], sizeof(uint64_t)}},
                                   false,
                                   true,
                                   0,
                                   longSoUp,
                                   0,
                                   memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    const uint64_t timeoutMicroSec = (uint64_t)((int64_t)SCAL_FOREVER);
    status                         = pStreamDmaUp->longSoWait(longSoUp, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "longSoWait failed";

    status = synDeviceFree(device.getDeviceId(), deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory (" << deviceAddress << ")";

    // Validation
    ASSERT_EQ(*pHostAddress[Source1], *pHostAddress[Destination1])
        << "Compare failed *pDestination1 (" << *pHostAddress[Destination1] << ")";
    ASSERT_EQ(*pHostAddress[Source2], *pHostAddress[Destination2])
        << "Compare failed *pDestination2 (" << *pHostAddress[Destination2] << ")";

    for (unsigned iter = Source1; iter < HostAddrMax; iter++)
    {
        status = synHostFree(device.getDeviceId(), pHostAddress[iter], 0);
        ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Host memory (" << iter << ")";
    }
}

TEST_F_SYN(ScalStreamTest, monitor_basic_test)
{
    TestDevice device(m_deviceType);

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();
    ASSERT_NE(pDev, nullptr) << "failed to get device";

    ScalStreamBaseInterface* pStreamBaseDmaDownUser = pDev->debugGetCreatedStream(INTERNAL_STREAM_TYPE_DMA_DOWN_USER);
    ScalStreamCopyInterface* pStreamDmaDown         = dynamic_cast<ScalStreamCopyInterface*>(pStreamBaseDmaDownUser);
    ASSERT_NE(pStreamDmaDown, nullptr) << "Device stream dma-down is nullptr";

    ScalMonitorBase* pScalMonitor = pStreamDmaDown->testGetScalMonitor();
    ASSERT_NE(pScalMonitor, nullptr) << "failed to get stream's scal-monitor";

    MonitorAddressesType addr;
    MonitorValuesType    value;
    uint8_t              numRegs = 0;
    uint8_t              fenceId = 0;

    unsigned expectedRegs = sizeof(MonitorAddressesType) / sizeof(uint64_t);
    if (m_deviceType != synDeviceGaudi2)
    {
        // Gaudi2 requires dummy-message due to some WA (SW-67146)
        expectedRegs -= 3;
    }
    bool isValid = pScalMonitor->getConfigRegsForLongSO(fenceId, numRegs, addr, value);
    ASSERT_EQ(isValid, true);
    ASSERT_EQ(numRegs, expectedRegs);

    ScalLongSyncObject longSO;
    longSO.m_index      = 192;
    uint64_t& targetVal = longSO.m_targetValue;
    targetVal           = 2;

    expectedRegs = 1;
    // as long we configure only the 15 lsb only one regiester should be return
    while (targetVal < (1 << 15))
    {
        pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);

        ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

        unsigned lower      = (2 << 10);
        unsigned upper      = (2 << 15);
        unsigned randNumber = (rand() % (upper - lower + 1)) + lower;
        targetVal += randNumber;
    }

    targetVal    = (1 << 15);
    expectedRegs = 2;  // second monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ", targetVal=" << targetVal;

    targetVal    = (1UL << 30) - 1UL;
    expectedRegs = 2;  // second monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

    targetVal    = (1UL << 30);
    expectedRegs = 3;  // second and third monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

    targetVal    = (1UL << 30) + (1UL << 30);
    expectedRegs = 2;  // third monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

    targetVal    = (1UL << 45) - 1UL;
    expectedRegs = 3;  // second and third monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

    targetVal    = (1UL << 45);
    expectedRegs = 4;  // second,third and forth monitor changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;

    expectedRegs = 1;  // only the first changed
    pScalMonitor->getArmRegsForLongSO(longSO, fenceId, numRegs, addr, value);
    ASSERT_EQ(numRegs, expectedRegs) << "expectedRegs not equal to " << expectedRegs << ",targetVal=" << targetVal;
}

TEST_F_SYN(ScalStreamTest, testDmaSyncTowardsComputeStream)
{
    TestDevice device(m_deviceType);

    uint64_t* pHostAddress = nullptr;
    auto      status       = synHostMalloc(device.getDeviceId(), sizeof(uint64_t), 0, (void**)&pHostAddress);
    ASSERT_EQ(status, synSuccess) << "Could not allocate host memory";
    //
    *pHostAddress = 0x1;

    uint64_t deviceAddress = 0;
    //
    status = synDeviceMalloc(device.getDeviceId(), sizeof(uint64_t), 0, 0, &deviceAddress);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace in device memory";

    DevMemoryAlloc* pAllocator = DevMemoryAllocScal::debugGetLastConstructedAllocator();
    ASSERT_NE(pAllocator, nullptr) << "failed to get allocator";

    uint64_t             mappedAddress = 0;
    const eMappingStatus translateStatus =
        pAllocator->getDeviceVirtualAddress(true, pHostAddress, sizeof(uint64_t), &mappedAddress, nullptr);
    ASSERT_EQ(translateStatus, HATVA_MAPPING_STATUS_FOUND) << "Failed to translate (" << (uint64_t)pHostAddress << ")";

    ScalDev* pDev = ScalDev::debugGetLastConstuctedDevice();
    ASSERT_NE(pDev, nullptr) << "failed to get device";

    const ComputeCompoundResources* pComputeCompoundResources;
    ScalStreamBaseInterface* pScalStreamCompute    = pDev->debugGetCreatedComputeResources(pComputeCompoundResources);
    ScalStreamCopyInterface* pScalStreamTxCommands = pComputeCompoundResources->m_pTxCommandsStream;
    ASSERT_NE(pScalStreamCompute, nullptr) << "Device stream compute is nullptr";
    ASSERT_NE(pScalStreamTxCommands, nullptr) << "Device stream synapse dma-down is nullptr";

    if (!pScalStreamTxCommands->isDirectMode())
    {
        return;
    }

    LOG_DEBUG(SYN_RT_TEST, "Test direct-mode DMA Stream's external-signal towards Compute Stream");

    uint64_t payloadAddress = 0;
    uint32_t payloadData    = 0;
    ASSERT_EQ(pScalStreamCompute->getStaticMonitorPayloadInfo(payloadAddress, payloadData), true)
        << "Failed to get Payload-Info";

    ScalStreamCopyInterface::MemcopySyncInfo memcopySyncInfo = {.m_pdmaSyncMechanism =
                                                                    ScalStreamCopyInterface::PDMA_TX_SYNC_FENCE_ONLY,
                                                                .m_workCompletionAddress = payloadAddress,
                                                                .m_workCompletionValue   = payloadData};
    ScalLongSyncObject                       longSo(LongSoEmpty);
    status = pScalStreamTxCommands->memcopy(ResourceStreamType::SYNAPSE_DMA_DOWN,
                                            {{mappedAddress, deviceAddress, sizeof(uint64_t)}},
                                            false,
                                            true,
                                            0,
                                            longSo,
                                            0,
                                            memcopySyncInfo);
    ASSERT_EQ(status, synSuccess) << "memcopy failed";

    uint32_t target = 1;
    status = pScalStreamCompute->addStreamFenceWait(target, false /* isUserReq */, true /* isInternalComputeSync */);
    ASSERT_EQ(status, synSuccess) << "addStreamFenceWait failed";

    ScalLongSyncObject longSoBarrier(LongSoEmpty);
    status = pScalStreamCompute->addBarrierOrEmptyPdma(longSoBarrier);
    ASSERT_EQ(status, synSuccess) << "addBarrierOrEmptyPdma failed";

    const uint64_t timeoutMicroSec = (uint64_t)((int64_t)SCAL_FOREVER);
    status                         = pScalStreamCompute->longSoWaitForLast(false, timeoutMicroSec, __FUNCTION__);
    ASSERT_EQ(status, synSuccess) << "longSoWaitForLast failed";

    status = synDeviceFree(device.getDeviceId(), deviceAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory (" << deviceAddress << ")";

    status = synHostFree(device.getDeviceId(), pHostAddress, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Host memory";
}

TEST_F_SYN(ScalStreamTest, testStreamCcbOccupancyWatermark)
{
    testStreamCcbOccupancyWatermark(m_deviceType);
}

TEST_F_SYN(ScalStreamTest, testTxUserStreamCcbOccupancy)
{
    testUserTxPdmaStreamCcbOccupancy(m_deviceType);
}