#include "runtime/common/osal/osal.hpp"

#include "syn_base_test.hpp"
#include "test_device.hpp"
#include "test_recipe_gemm.hpp"
#include "scoped_configuration_change.h"
#include "../infra/test_types.hpp"
#include "../infra/test_tensor_init.hpp"
#include "synapse_api.h"
#include <memory>

class SynApiTests : public SynBaseTest
{
public:
    SynApiTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3}); };

    void memcopyTest(uint64_t copySize);
    void getDeviceParentNameTest();
};

REGISTER_SUITE(SynApiTests, ALL_TEST_PACKAGES);

void SynApiTests::memcopyTest(uint64_t copySize)
{
    synStatus status(synSuccess);

    TestDevice device(m_deviceType);

    TestHostBufferMalloc  dwlHostBuffer(device.allocateHostBuffer(copySize, 0 /* flags */));
    TestHostBufferMalloc  uplHostBuffer(device.allocateHostBuffer(copySize, 0 /* flags */));
    TestDeviceBufferAlloc deviceBuffer(device.allocateDeviceBuffer(copySize, 0 /* flags */));

    // Init buffers value
    OneTensorInitInfo dwlHostBufferInfo = {TensorInitOp::CONST, 0xEE, true};
    OneTensorInitInfo uplHostBufferInfo = {TensorInitOp::CONST, 0xFF, true};

    initBufferValues(dwlHostBufferInfo, syn_type_uint8, copySize, dwlHostBuffer.getBuffer());
    initBufferValues(uplHostBufferInfo, syn_type_uint8, copySize, uplHostBuffer.getBuffer());

    // Create streams
    TestStream      stream       = device.createStream();
    synStreamHandle streamHandle = stream;

    // DWL
    status = synMemCopyAsync(streamHandle,
                             (uint64_t)dwlHostBuffer.getBuffer(),
                             copySize,
                             deviceBuffer.getBuffer(),
                             HOST_TO_DRAM);
    ASSERT_EQ(status, synSuccess) << "Failed copy to the device";

    // UPL
    status = synMemCopyAsync(streamHandle,
                             deviceBuffer.getBuffer(),
                             copySize,
                             (uint64_t)uplHostBuffer.getBuffer(),
                             DRAM_TO_HOST);
    ASSERT_EQ(status, synSuccess) << "Failed copy from the device";
    //
    // Synchronize
    status = synStreamSynchronize(streamHandle);
    ASSERT_EQ(status, synSuccess) << "Failed synchronize-stream (upload)";

    // Validate results
    uint64_t  numOfValues   = copySize / sizeof(uint32_t);
    uint32_t* pCurrDownload = (uint32_t*)dwlHostBuffer.getBuffer();
    uint32_t* pCurrUpload   = (uint32_t*)uplHostBuffer.getBuffer();
    for (uint64_t i = 0; i < numOfValues; i++, pCurrUpload++, pCurrDownload++)
    {
        ASSERT_EQ(*pCurrUpload, *pCurrDownload)
            << "Invalid value read from device i = " << i << " out of " << numOfValues;
    }
}

void SynApiTests::getDeviceParentNameTest()
{
    std::string deviceParentName;
    EXPECT_EQ(OSAL::getInstance().getDeviceParentName(deviceParentName, 0), synSuccess);
    EXPECT_NE(deviceParentName, "");
}

TEST_F_SYN(SynApiTests, basic_memcopy_validation)
{
    uint64_t copySize = 10000;
    memcopyTest(copySize);
}

TEST_F_SYN(SynApiTests, DISABLED_huge_memcopy_validation_ASIC_CI, {synTestPackage::ASIC_CI})
{
    ScopedEnvChange timeoutDisEnv("SCAL_DISABLE_TIMEOUT", "1");

    uint64_t copySize = (uint64_t)7 * 1024 * 1024 * 1024;  // 7GB (which is >> std::numeric_limits<uint32_t>::max())
    memcopyTest(copySize);
}

TEST_F_SYN(SynApiTests, get_device_parent_name)
{
    getDeviceParentNameTest();
}