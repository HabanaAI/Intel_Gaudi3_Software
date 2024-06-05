#include "syn_base_test.hpp"
#include "synapse_api_types.h"
#include "synapse_api.h"
#include "runtime/common/osal/buffer_allocator.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

class APITest : public SynBaseTest
{
public:
    APITest() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); }
    synTensor createTrainingTensor(unsigned             dims,
                                   synDataType          data_type,
                                   const unsigned*      tensor_size,
                                   bool                 is_presist,
                                   const char*          name,
                                   const synGraphHandle graph_handle)
    {
        synStatus           status;
        synTensorDescriptor desc {};

        // input
        desc.m_dataType = data_type;
        desc.m_dims     = dims;
        desc.m_name     = name;
        memset(desc.m_strides, 0, sizeof(desc.m_strides));

        for (unsigned i = 0; i < dims; ++i)
        {
            desc.m_sizes[i] = tensor_size[dims - 1 - i];
        }

        synSectionHandle pSectionHandle = nullptr;
        if (is_presist)
        {
            synSectionCreate(&pSectionHandle, 0, graph_handle);
            m_sections.push_back(pSectionHandle);
        }
        synTensor tensor;
        status = synTensorCreate(&tensor, &desc, pSectionHandle, 0);
        assert(status == synSuccess && "Create tensor failed!");

        UNUSED(status);

        return tensor;
    }
    virtual void TearDown() override
    {
        for (auto section : m_sections)
        {
            ASSERT_EQ(synSuccess, synSectionDestroy(section));
        }
        m_sections.clear();
        SynBaseTest::TearDown();
    }
    std::vector<synSectionHandle> m_sections;
};

REGISTER_SUITE(APITest, ALL_TEST_PACKAGES);

TEST_F_SYN(APITest, gaudi_acquring_test)
{
    TestDevice device(m_deviceType);

    uint32_t count  = 0;
    auto     status = synDeviceGetCount(&count);
    ASSERT_EQ(status, synSuccess) << "Failed on synDeviceGetCount ";
    ASSERT_GT(count, 0);

    status = synDeviceGetCountByDeviceType(&count, synDeviceEmulator);
    ASSERT_EQ(status, synSuccess) << "Failed on synDeviceGetCountByDeviceType";

    char version[100] = "";

    status = synDriverGetVersion(version, 100);
    ASSERT_EQ(status, synSuccess) << "Failed on synDriverGetVersion";
    ASSERT_NE(0, strlen(version));

    char name[100] = "";

    status = synDeviceGetName(name, 100, device.getDeviceId());
    ASSERT_EQ(status, synSuccess) << "Failed on synDeviceGetName";
    ASSERT_NE(0, strlen(name));
}

TEST_F_SYN(APITest, gaudi_mem_test)
{
    uint64_t       free                 = 0;
    uint64_t       total                = 0;
    const uint64_t numOfBytesToAllocate = 100;
    synDeviceInfo  deviceInfo;

    TestDevice device(m_deviceType);

    auto status = synDeviceGetInfo(device.getDeviceId(), &deviceInfo);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    LOG_TRACE(SYN_API,
              "dramSize {} m_deviceId {} deviceType {} dramBaseAddress {} dramEnabled {} sramSize {} tpcEnabledMask {} "
              "dramBaseAddress {}",
              deviceInfo.dramSize,
              deviceInfo.deviceId,
              deviceInfo.deviceType,
              deviceInfo.dramBaseAddress,
              deviceInfo.dramEnabled,
              deviceInfo.sramSize,
              deviceInfo.tpcEnabledMask,
              deviceInfo.dramBaseAddress);

    status = synDeviceGetMemoryInfo(device.getDeviceId(), &free, &total);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    uint64_t input_a1_mem, input_a2_mem;
    void *   input_a1_host = nullptr, *input_a2_host = nullptr;

    status = synDeviceMalloc(device.getDeviceId(), numOfBytesToAllocate, 0, 0, &input_a1_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    status = synDeviceMalloc(device.getDeviceId(), numOfBytesToAllocate * 2, 0, 0, &input_a2_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    status = synHostMalloc(device.getDeviceId(), numOfBytesToAllocate, 0, &input_a1_host);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Host memory";
    status = synHostMalloc(device.getDeviceId(), numOfBytesToAllocate, 0, &input_a2_host);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Host memory";

    status = synHostMap(device.getDeviceId(), numOfBytesToAllocate, input_a2_host);
    ASSERT_EQ(status, synSuccess) << "Failed to Map host to Device HBM memory";
    // cleanup
    status = synHostUnmap(device.getDeviceId(), input_a2_host);
    ASSERT_EQ(status, synSuccess) << "Failed to Map host to Device HBM memory";

    status = synDeviceFree(device.getDeviceId(), input_a1_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    status = synDeviceFree(device.getDeviceId(), input_a2_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";

    status = synHostFree(device.getDeviceId(), input_a1_host, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate Host memory";
}

// The test examines the basic malloc / free API'S and also the MEMINFO api correctness
TEST_F_SYN(APITest, device_malloc_test_ASIC_CI, {synTestPackage::ASIC_CI, synTestPackage::ASIC})
{
    const uint64_t numOfBytesToAllocate     = 128;
    uint64_t       numOfBytesToAllocateBuf1 = 0;
    uint64_t       numOfBytesToAllocateBuf2 = 0;
    uint64_t       dummyBuffer1             = 0;
    uint64_t       dummyBuffer2             = 0;

    uint64_t freeMemAtStart = 0;
    uint64_t total          = 0;

    synDeviceId deviceId;
    synStatus   status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to acquire device";

    status = synDeviceGetMemoryInfo(deviceId, &freeMemAtStart, &total);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    const unsigned numOfIterations = 2;
    for (unsigned iterationNum = 0; iterationNum < numOfIterations; iterationNum++)
    {
        uint64_t free = 0;

        status = synDeviceRelease(deviceId);
        ASSERT_EQ(status, synSuccess) << "synDeviceRelease failed! (iterationNum " << iterationNum << ")";

        status = synDeviceAcquireByDeviceType(&deviceId, m_deviceType);
        ASSERT_EQ(status, synSuccess) << "synDeviceAcquire (first) failed! (iterationNum " << iterationNum << ")";

        status = synDeviceMalloc(deviceId, numOfBytesToAllocate, 0, 0, &dummyBuffer1);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate device buffer (iterationNum " << iterationNum << ")";

        status = synDeviceGetMemoryInfo(deviceId, &free, &total);
        ASSERT_EQ(status, synSuccess) << "Failed to get memory usage (iterationNum " << iterationNum << ")";
        numOfBytesToAllocateBuf1 = freeMemAtStart - free;

        status = synDeviceMalloc(deviceId, numOfBytesToAllocate, 0, 0, &dummyBuffer2);
        ASSERT_EQ(status, synSuccess) << "Failed to allocate device buffer (iterationNum " << iterationNum << ")";

        status = synDeviceGetMemoryInfo(deviceId, &free, &total);
        ASSERT_EQ(status, synSuccess) << "Failed to get memory usage (iterationNum " << iterationNum << ")";

        numOfBytesToAllocateBuf2 = freeMemAtStart - free - numOfBytesToAllocateBuf1;

        status = synDeviceGetMemoryInfo(deviceId, &free, &total);
        ASSERT_EQ(status, synSuccess) << "Failed to get memory usage (iterationNum " << iterationNum << ")";
        ASSERT_EQ(free, (freeMemAtStart - numOfBytesToAllocateBuf1 - numOfBytesToAllocateBuf2));

        status = synDeviceFree(deviceId, dummyBuffer1, 0);
        ASSERT_EQ(status, synSuccess) << "Failed to free buffer (iterationNum " << iterationNum << ")";

        status = synDeviceGetMemoryInfo(deviceId, &free, &total);
        ASSERT_EQ(status, synSuccess) << "Failed to get memory usage (iterationNum " << iterationNum << ")";
        ASSERT_EQ(free, (freeMemAtStart - numOfBytesToAllocateBuf2));

        // On the first iteration we will free upon release-device
        if (iterationNum != 0)
        {
            status = synDeviceFree(deviceId, dummyBuffer2, 0);
            ASSERT_EQ(status, synSuccess) << "Failed to free buffer (iterationNum " << iterationNum << ")";

            status = synDeviceGetMemoryInfo(deviceId, &free, &total);
            ASSERT_EQ(status, synSuccess) << "Failed to get memory usage (iterationNum " << iterationNum << ")";
            ASSERT_EQ(free, freeMemAtStart);
        }
    }
}

// The test examines the basic malloc / free API'S and also the MEMINFO api correctness
TEST_F_SYN(APITest, device_malloc_all_memory)
{
    unsigned allocationIndex = 0;
    uint64_t freeMemAtStart  = 0;
    uint64_t total           = 0;

    TestDevice device(m_deviceType);

    auto status = synDeviceGetMemoryInfo(device.getDeviceId(), &freeMemAtStart, &total);
    ASSERT_EQ(status, synSuccess) << "Failed to get memory usage";

    const unsigned numOfAllocations      = 65;
    const unsigned numOfValidAllocations = (numOfAllocations - 1);
    uint64_t       alignmentBuffer       = (numOfValidAllocations * ManagedBufferAllocator::m_defaultAlignment);
    uint64_t       numOfBytesToAllocate  = (freeMemAtStart - alignmentBuffer) / numOfValidAllocations;
    uint64_t       dummyBuffer[numOfAllocations];

    do
    {
        status = synDeviceMalloc(device.getDeviceId(), numOfBytesToAllocate, 0, 0, &dummyBuffer[allocationIndex]);
        allocationIndex++;
    } while ((status == synSuccess) && (allocationIndex < numOfAllocations));

    ASSERT_EQ(allocationIndex, numOfAllocations) << "Failed to fully allocate all memory";
    ASSERT_NE(status, synSuccess) << "Last faulty allocation succeeded";

    unsigned freeIndex = 0;

    do
    {
        status = synDeviceFree(device.getDeviceId(), dummyBuffer[freeIndex], 0);
        freeIndex++;
    } while ((status == synSuccess) && (freeIndex < numOfValidAllocations));
}

TEST_F_SYN(APITest, gaudi_tensor_create)
{
    uint64_t  input_a1_mem, input_b1_mem, input_x1_mem, input_x2_mem;
    synStatus status;

    // Prepare some descriptors

    const unsigned x1TensorSizes[] = {2, 2};
    const unsigned x2TensorSizes[] = {3};
    const unsigned a1TensorSizes[] = {2, 2};
    const unsigned b1TensorSizes[] = {3, 3};

    const unsigned x1TensorTotalSize = x1TensorSizes[0] * x1TensorSizes[1];
    const unsigned x2TensorTotalSize = x2TensorSizes[0];
    const unsigned a1TensorTotalSize = a1TensorSizes[0] * a1TensorSizes[1];
    const unsigned b1TensorTotalSize = b1TensorSizes[0] * b1TensorSizes[1];

    synTensor firstInputTensor;
    synTensor secondInputTensor;
    synTensor firstOutputTensor;
    synTensor secondOutputTensor;

    synGraphHandle graphHandle;
    status = synGraphCreate(&graphHandle, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to create graph";

    TestDevice device(m_deviceType);

    status = synDeviceMalloc(device.getDeviceId(), x1TensorTotalSize * sizeof(uint16_t), 0, 0, &input_x1_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    status = synDeviceMalloc(device.getDeviceId(), x2TensorTotalSize * sizeof(uint16_t), 0, 0, &input_x2_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    status = synDeviceMalloc(device.getDeviceId(), a1TensorTotalSize * sizeof(uint16_t), 0, 0, &input_a1_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    status = synDeviceMalloc(device.getDeviceId(), b1TensorTotalSize * sizeof(uint16_t), 0, 0, &input_b1_mem);
    ASSERT_EQ(status, synSuccess) << "Failed to allocate Device HBM memory";

    // Use descriptors to prepare tensor definition

    firstInputTensor   = createTrainingTensor(2U, syn_type_bf16, x1TensorSizes, false, "X1", graphHandle);
    secondInputTensor  = createTrainingTensor(1U, syn_type_bf16, x2TensorSizes, false, "X2", graphHandle);
    firstOutputTensor  = createTrainingTensor(2U, syn_type_bf16, a1TensorSizes, false, "A_0_1", graphHandle);
    secondOutputTensor = createTrainingTensor(2U, syn_type_bf16, b1TensorSizes, true, "B_0_1", graphHandle);

    status = synTensorDestroy(firstInputTensor);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Tensor";

    status = synTensorDestroy(secondInputTensor);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Tensor";

    status = synTensorDestroy(secondOutputTensor);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Tensor";

    status = synTensorDestroy(firstOutputTensor);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Tensor";

    status = synDeviceFree(device.getDeviceId(), input_x1_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    status = synDeviceFree(device.getDeviceId(), input_x2_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    status = synDeviceFree(device.getDeviceId(), input_a1_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";
    status = synDeviceFree(device.getDeviceId(), input_b1_mem, 0);
    ASSERT_EQ(status, synSuccess) << "Failed to Deallocate (free) Device memory";

    status = synGraphDestroy(graphHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to Destroy Graph";
}

TEST_F_SYN(APITest, DISABLED_check_module_ids)
{
    synStatus status;

    uint32_t size = 0;
    status        = synDeviceGetCount(&size);
    ASSERT_EQ(status, synSuccess) << "Failed to get device count";

    uint32_t allocateSize = size;
    uint32_t moduleIDsArray[size];
    status = synDeviceGetModuleIDs(moduleIDsArray, &allocateSize);
    ASSERT_EQ(status, synSuccess) << "Failed to get device module ID's";
    ASSERT_EQ(size, allocateSize) << "Got wrong module ID's sizes";

    uint32_t firstModule = moduleIDsArray[size - 1];
    status               = synDeviceGetModuleIDs(moduleIDsArray, &size);
    ASSERT_EQ(status, synSuccess) << "Failed to get device module ID's";
    ASSERT_EQ(moduleIDsArray[size - 1], firstModule) << "Failed to get device module ID's sizes";
}