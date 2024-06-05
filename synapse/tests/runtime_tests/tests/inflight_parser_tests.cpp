#include "syn_base_test.hpp"

#include "runtime/common/common_types.hpp"

#include "runtime/common/device/device_common.hpp"

#include "runtime/common/streams/stream.hpp"

#include "syn_singleton.hpp"

#include "runtime/common/recipe/recipe_handle_impl.hpp"

#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/queue_compute_qman.hpp"

#include "runtime/qman/gaudi/master_qmans_definition.hpp"

#include "synapse_common_types.h"

#include "test_recipe_dma.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"

// Specs
#include "gaudi/gaudi_packets.h"

class SynGaudiInflightParserTests : public SynBaseTest
{
public:
    SynGaudiInflightParserTests() : SynBaseTest() { setSupportedDevices({synDeviceGaudi}); };

    virtual ~SynGaudiInflightParserTests() {};

    bool parseCommandSubmission(TestDevice& rDevice, CommandSubmissionDataChunks* pCsDataChunks);
    CommandSubmissionDataChunks* getTestedCommandSubmission(TestDevice& rDevice, synRecipeHandle& recipeHandle);

    QueueComputeQman* getTestedComputeQueue(TestDevice& rDevice);

    // Retrieves the host-buffer and internal-queue of any of the arb-slaves
    // and internal-queue for both arb-master and arb-slave
    // This is used for corrupting the CS content, as part of the test
    void retrieveQueuesAndArbSlaveHostBuffer(synInternalQueue*&           pArbMasterInternalQueue,
                                             synInternalQueue*&           pArbSlaveInternalQueue,
                                             void*&                       arbSlaveHostBuffer,
                                             CommandSubmissionDataChunks* pCsDataChunks);
};

REGISTER_SUITE(SynGaudiInflightParserTests, ALL_TEST_PACKAGES);

bool SynGaudiInflightParserTests::parseCommandSubmission(TestDevice&                  rDevice,
                                                         CommandSubmissionDataChunks* pCsDataChunks)
{
    if (pCsDataChunks == nullptr)
    {
        return false;
    }

    QueueComputeQman* pComputeQueue = getTestedComputeQueue(rDevice);
    return pComputeQueue->_parseSingleCommandSubmission(pCsDataChunks);
}

CommandSubmissionDataChunks* SynGaudiInflightParserTests::getTestedCommandSubmission(TestDevice&      rDevice,
                                                                                     synRecipeHandle& recipeHandle)
{
    // We will update the executionBlobsBuffer, as the parser uses this parser when parsing the device parts
    QueueComputeQman* pComputeQueue = getTestedComputeQueue(rDevice);

    auto recipeProcessorItr    = pComputeQueue->m_recipeProcessorsDB.find(recipeHandle);
    auto recipeProcessorEndItr = pComputeQueue->m_recipeProcessorsDB.end();
    if (recipeProcessorItr == recipeProcessorEndItr)
    {
        return nullptr;
    }

    // The CS should have been completed (due to stream-sync)
    // In any case, we only care about the CS content
    return recipeProcessorItr->second->getAvailableCommandSubmissionDataChunks(EXECUTION_STAGE_ENQUEUE, true);
}

QueueComputeQman* SynGaudiInflightParserTests::getTestedComputeQueue(TestDevice& rDevice)
{
    // Create streams
    TestStream      stream       = rDevice.createStream();
    synStreamHandle streamHandle = stream;

    std::shared_ptr<DeviceInterface> deviceInterface = _SYN_SINGLETON_INTERNAL->getDevice();

    DeviceCommon*   pDevice = (DeviceCommon*)(deviceInterface.get());
    QueueInterface* pQueueInterface;
    Stream*         pStream;

    {
        auto streamSptr = pDevice->loadAndValidateStream(streamHandle, __FUNCTION__);
        pStream         = streamSptr.get();
    }

    pStream->testGetQueueInterface(QUEUE_TYPE_COMPUTE, pQueueInterface);

    return dynamic_cast<QueueComputeQman*>(pQueueInterface);
}

void SynGaudiInflightParserTests::retrieveQueuesAndArbSlaveHostBuffer(synInternalQueue*& pArbMasterInternalQueue,
                                                                      synInternalQueue*& pArbSlaveInternalQueue,
                                                                      void*&             arbSlaveHostBuffer,
                                                                      CommandSubmissionDataChunks* pCsDataChunks)
{
    pArbSlaveInternalQueue  = nullptr;
    arbSlaveHostBuffer      = nullptr;
    pArbMasterInternalQueue = nullptr;

    // Find an arb-slave's host-buffer on submitted CS
    CommandSubmission* pCommandSubmission  = pCsDataChunks->getCommandSubmissionInstance();
    uint32_t           numOfInternalQueues = pCommandSubmission->getNumExecuteInternalQueue();
    synInternalQueue*  pInternalQueue = const_cast<synInternalQueue*>(pCommandSubmission->getExecuteInternalQueueCb());
    //
    bool isArbSlaveFound  = false;
    bool isArbMasterFound = false;
    for (unsigned i = 0; i < numOfInternalQueues; i++, pInternalQueue++)
    {
        if ((!isArbSlaveFound) &&
            gaudi::QmansDefinition::getInstance()->isComputeArbSlaveQueueId(pInternalQueue->queueIndex))
        {
            pArbSlaveInternalQueue = pInternalQueue;
            isArbSlaveFound        = true;
        }
        else if ((!isArbMasterFound) &&
                 gaudi::QmansDefinition::getInstance()->isArbMasterForComputeAndNewGaudiSyncScheme(
                     pInternalQueue->queueIndex))
        {
            pArbMasterInternalQueue = pInternalQueue;
            isArbMasterFound        = true;
        }
    }
    //
    cpDmaDataChunksDB& upperCpDataChunk = pCsDataChunks->getCpDmaDataChunks();

    isArbSlaveFound = false;
    for (DataChunk* pDataChunk : upperCpDataChunk)
    {
        void*    hostBuffer = pDataChunk->getChunkBuffer();
        uint64_t handle     = pDataChunk->getHandle();

        if ((!isArbSlaveFound) && (pArbSlaveInternalQueue->address == handle))
        {
            arbSlaveHostBuffer = hostBuffer;
            isArbSlaveFound    = true;
        }
    }
}

TEST_F_SYN(SynGaudiInflightParserTests, inflight_parser)
{
    TURN_ON_TRACE_MODE_LOGGING();

    TestRecipeDma recipe(m_deviceType, 16 * 1024U, 1024U, 0xEE, false, syn_type_uint8);
    recipe.generateRecipe();

    TestDevice device(m_deviceType);

    TestStream stream = device.createStream();

    TestLauncher launcher(device);

    RecipeLaunchParams recipeLaunchParams =
        launcher.createRecipeLaunchParams(recipe, {TensorInitOp::RANDOM_POSITIVE, 0});

    TestLauncher::execute(stream, recipe, recipeLaunchParams);

    stream.synchronize();

    // The Parser uses the execution-blobs buffer, for parsing the program-code parts that are at the device
    synRecipeHandle recipeHandle         = recipe.getRecipe();
    recipe_t*       pRecipe              = recipeHandle->basicRecipeHandle.recipe;
    uint64_t*       executionBlobsBuffer = pRecipe->execution_blobs_buffer;

    CommandSubmissionDataChunks* pCsDataChunks = getTestedCommandSubmission(device, recipeHandle);
    ASSERT_NE(pCsDataChunks, nullptr) << "CS DC is nullptr";

    synInternalQueue* pArbMasterInternalQueue = nullptr;
    synInternalQueue* pArbSlaveInternalQueue  = nullptr;
    void*             arbSlaveHostBuffer      = nullptr;
    retrieveQueuesAndArbSlaveHostBuffer(pArbMasterInternalQueue,
                                        pArbSlaveInternalQueue,
                                        arbSlaveHostBuffer,
                                        pCsDataChunks);

    // Test-0: Parse original CS-DC
    bool parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, true) << "Parsing unexpectedly failed";

    // Test-1: Invalid opcode
    packet_nop* pPacket = (packet_nop*)executionBlobsBuffer;
    ASSERT_EQ(pPacket->opcode, PACKET_MSG_SHORT) << "Expected different opcode (origin execution blobs buffer)";
    //
    static const uint32_t INVLAID_OPCODE = 0x10;
    pPacket->opcode                      = INVLAID_OPCODE;
    //
    parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, false) << "Parsing unexpectedly succeed (Invalid opcode)";

    // Test-2: Invalid address
    packet_cp_dma* pCpDmaPacket = (packet_cp_dma*)((uint8_t*)arbSlaveHostBuffer + sizeof(packet_arb_point));
    ASSERT_EQ(pCpDmaPacket->opcode, PACKET_CP_DMA) << "Expected CP_DMA opcode";
    //
    // Test-2a: Due to base-address
    pCpDmaPacket->src_addr -= 0x1;
    parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, false) << "Parsing unexpectedly succeed (Invalid base-address)";
    pCpDmaPacket->src_addr += 0x1;  // back to original address
    //
    // Test-2b: Due to Size
    pCpDmaPacket->tsize += 1;
    parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, false) << "Parsing unexpectedly succeed (Invalid size)";
    pCpDmaPacket->tsize -= 1;

    // Test-3: Invalid stream structure
    // Test-3a: ARB-Slave should start with ARB-Point
    pArbSlaveInternalQueue->address += sizeof(packet_arb_point);
    parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, false) << "Parsing unexpectedly succeed (Invalid ARB-Slave commands-structure)";
    pArbSlaveInternalQueue->address -= sizeof(packet_arb_point);
    //
    // Test-3b: ARB-Master should start with Fence
    pArbMasterInternalQueue->address += sizeof(packet_fence);
    parseStatus = parseCommandSubmission(device, pCsDataChunks);
    ASSERT_EQ(parseStatus, false) << "Parsing unexpectedly succeed (Invalid ARB-Master commands-structure)";
    pArbMasterInternalQueue->address -= sizeof(packet_fence);

    TURN_OFF_TRACE_MODE_LOGGING();
}