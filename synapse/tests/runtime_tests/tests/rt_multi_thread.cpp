#include "syn_base_test.hpp"
#include "test_event.hpp"
#include "test_stream.hpp"
#include "test_device.hpp"
#include "synapse_api.h"
#include <thread>

class SynGaudi2EventMtTests : public SynBaseTest
{
public:
    SynGaudi2EventMtTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }
    virtual ~SynGaudi2EventMtTests() = default;

    void testEventMt();
};

void anotherRecord(TestEvent& event, TestStream& stream)
{
    try
    {
        stream.eventRecord(event);
        event.query();
    }
    catch (...)
    {
        LOG_DEBUG(SYN_RT_TEST, "{}: caught expected exception on event.record", __func__);
    }
}

REGISTER_SUITE(SynGaudi2EventMtTests, synTestPackage::CI);

// try to change an event in another thread while the original tries to synchronize on it
void SynGaudi2EventMtTests::testEventMt()
{
    const uint64_t SIZE = 1024ULL * 1024ULL;

    assert(SIZE % sizeof(uint64_t) == 0);

    TestDevice device(m_deviceType);

    // create streams (up/down)
    TestStream streamDown = device.createStream();
    TestStream streamUp   = device.createStream();

    TestHostBufferMalloc  inBuff  = device.allocateHostBuffer(SIZE, 0);
    TestDeviceBufferAlloc devBuff = device.allocateDeviceBuffer(SIZE, 0);

    TestEvent      event          = device.createEvent();
    const unsigned NUM_ITERATIONS = 10000;
    unsigned       counter        = NUM_ITERATIONS;

    // Copy the buffer to device
    uint64_t inAddress  = (uint64_t)inBuff.getBuffer();
    uint64_t dstAddress = (uint64_t)devBuff.getBuffer();

    while (counter--)
    {
        // loop until the sync problem arises ...

        // we expect (at least until we fix it) to get an error like:
        // longSoWait: SCAL: Unrecognized longSo value detected m_cgHndl 0x55c4ca361ad0 longSo.m_index 528
        // longSo.m_targetValue 0x0
        streamDown.memcopyAsync(inAddress, SIZE, dstAddress, HOST_TO_DRAM);

        try
        {
            streamDown.eventRecord(event);  // take status of streamDown into a struct
            event.query();
        }
        catch (...)
        {
            LOG_DEBUG(SYN_RT_TEST, "{}: caught expected exception on event.record", __func__);
        }

        // create a thread that will try to change the event (by recording from streamUp)
        // while streamDown tries to synchronize on it
        std::thread func(anotherRecord, std::ref(event), std::ref(streamUp));
        ASSERT_EQ(synEventSynchronize(event), synSuccess) << "Failed to synchronize on event";
        func.join();
    }
}

TEST_F_SYN(SynGaudi2EventMtTests, test_event_mt)
{
    testEventMt();
}