#include <sstream>
#include "hal_conventions.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "command_queue.h"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "infra/defs.h"

using namespace gaudi;

#define validateEngine(engine, numValidEngines)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (engine >= numValidEngines)                                                                                 \
        {                                                                                                              \
            LOG_CRITICAL(GC, "{}: engine number {} is above the limit", HLLOG_FUNC, engine);                           \
            HB_ASSERT(engine < numValidEngines, "invalid engine");                                                     \
        }                                                                                                              \
    } while (0)

static gaudi_queue_id getTPCEngQueueId( unsigned engine )
{
    validateEngine(engine, CompilationHalReader::getHalReader()->getNumTpcEngines());
    return (gaudi_queue_id) (GAUDI_QUEUE_ID_TPC_0_0 + engine * CompilationHalReader::getHalReader()->getNumEngineStreams());
}

static gaudi_queue_id getMMEEngQueueId( unsigned engine )
{
    validateEngine(engine, CompilationHalReader::getHalReader()->getNumMmeEngines());
    return (gaudi_queue_id) (GAUDI_QUEUE_ID_MME_0_0 + engine * CompilationHalReader::getHalReader()->getNumEngineStreams());
}

static gaudi_queue_id getDMAHostDeviceEngQueueId()
{
    return (gaudi_queue_id) (GAUDI_QUEUE_ID_DMA_0_0);
}

static gaudi_queue_id getDMASramEngQueueId( unsigned engine )
{
    validateEngine(engine, CompilationHalReader::getHalReader()->getNumInternalDmaEngines());
    return (gaudi_queue_id) (GAUDI_QUEUE_ID_DMA_2_0 + engine * CompilationHalReader::getHalReader()->getNumEngineStreams());
}


gaudi_queue_id gaudi::getQueueID( HabanaDeviceType type, unsigned id )
{
    switch (type)
    {
    case DEVICE_TPC:
        return getTPCEngQueueId(id);

    case DEVICE_MME:
        return getMMEEngQueueId(id);

    case DEVICE_DMA_HOST_DEVICE:
        return getDMAHostDeviceEngQueueId();

    case DEVICE_COMPLETION_QUEUE:
        return (gaudi_queue_id)QmansDefinition::getInstance()->getWorkCompletionQueueId();

    case DEVICE_DMA_DRAM_SRAM:
    case DEVICE_DMA_SRAM_DRAM:
    case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
        return getDMASramEngQueueId(id);

    default:
        break;
    }
    LOG_ERR(GC, "Invalid queue requested. Type = {} ID = {}", type, id);
    return GAUDI_QUEUE_ID_SIZE;
}

std::string gaudi::getEngineName(gaudi_queue_id id, HalReader* pHalReader)
{
    if( id == GAUDI_QUEUE_ID_SIZE )
    {
        HB_ASSERT(false, "Unknown engine name");
        return "UNKNOWN";
    }
    else if (id == gaudi::QmansDefinition::getInstance()->getWorkCompletionQueueId())
    {
        return "WORK_COMPLETION";
    }

    std::stringstream   engineNameStr;
    unsigned            engineNumber = 0;
    unsigned            streamId     = 0;
    unsigned            numEngineStreams = 0;
    if (pHalReader == nullptr)
    {
        numEngineStreams = CompilationHalReader::getHalReader()->getNumEngineStreams();
    }
    else
    {
        numEngineStreams = pHalReader->getNumEngineStreams();
    }

    if(   ( id >= GAUDI_QUEUE_ID_DMA_0_0 )
       && ( id <= GAUDI_QUEUE_ID_DMA_1_3 ) )
    {
        engineNumber = id / numEngineStreams;
        streamId     = id - ( GAUDI_QUEUE_ID_DMA_0_0 + ( engineNumber * numEngineStreams ) );

        engineNameStr << "DMA";
    }

    if(   ( id >= GAUDI_QUEUE_ID_DMA_2_0 )
       && ( id <= GAUDI_QUEUE_ID_DMA_7_3 ) )
    {
        engineNumber = 2 + ( ( id - GAUDI_QUEUE_ID_DMA_2_0 ) / numEngineStreams );
        streamId     = id - ( GAUDI_QUEUE_ID_DMA_2_0 + ( (engineNumber - 2) * numEngineStreams ) );

        engineNameStr << "DMA";
    }

    if(   ( id >= GAUDI_QUEUE_ID_MME_0_0 )
       && ( id <= GAUDI_QUEUE_ID_MME_1_3 ) )
    {
        engineNumber = ( ( id - GAUDI_QUEUE_ID_MME_0_0 ) / numEngineStreams );
        streamId     = id - ( GAUDI_QUEUE_ID_MME_0_0 + ( engineNumber * numEngineStreams ) );

        engineNameStr << "MME";
    }

    if(   ( id >= GAUDI_QUEUE_ID_TPC_0_0 )
       && ( id <= GAUDI_QUEUE_ID_TPC_7_3 ) )
    {
        engineNumber = ( ( id - GAUDI_QUEUE_ID_TPC_0_0 ) / numEngineStreams );
        streamId     = id - ( GAUDI_QUEUE_ID_TPC_0_0 + ( engineNumber * numEngineStreams ) );

        engineNameStr << "TPC";
    }

    if(   ( id >= GAUDI_QUEUE_ID_NIC_0_0 )
       && ( id <= GAUDI_QUEUE_ID_NIC_9_3 ) )
    {
        engineNumber = ( ( id - GAUDI_QUEUE_ID_NIC_0_0 ) / numEngineStreams );
        streamId     = id - ( GAUDI_QUEUE_ID_NIC_0_0 + ( engineNumber * numEngineStreams ) );

        engineNameStr << "NIC";
    }

    engineNameStr << "-" << engineNumber << " STREAM-" << streamId;

    return engineNameStr.str();
}

unsigned gaudi::baseDmaLogicalQueue(unsigned numOfEngines)
{
    static constexpr unsigned numOfInternalEngines = 6;
    unsigned baseLogicalQueue = DEVICE_DMA_1_1_DRAM_SRAM_LOGICAL_QUEUE;
    for (unsigned i = 1; i < numOfEngines; ++i)
    {
        baseLogicalQueue += numOfInternalEngines / i;
    }
    return baseLogicalQueue;
}

unsigned gaudi::deviceTypeToLogicalQueue(const pNode& node, HabanaDeviceType deviceType)
{
    switch (deviceType)
    {
        case DEVICE_MME:                         return DEVICE_MME_LOGICAL_QUEUE;
        case DEVICE_TPC:                         return DEVICE_TPC_LOGICAL_QUEUE;
        case DEVICE_DMA_HOST_DEVICE:             return DEVICE_DMA_HOST_DEVICE_LOGICAL_QUEUE;
        case DEVICE_COMPLETION_QUEUE:            return DEVICE_COMPLETION_LOGICAL_QUEUE;
        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
        {
            std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);
            return baseDmaLogicalQueue(dmaNode->parallelLevel()) + dmaNode->dispatcherIndex();
        }
        default:
            HB_ASSERT(false, "Device type is not supported");
            break;
    }
    return LOGICAL_QUEUE_MAX_ID;
}

std::string_view gaudi::getQmanIdName(uint32_t id)
{
    // Aligned with enum gaudi_engine_id
    static constexpr std::string_view qmanNames[] = {"GAUDI_QMAN_ID_DMA0",
                                                     "GAUDI_QMAN_ID_DMA1",
                                                     "GAUDI_QMAN_ID_DMA2",
                                                     "GAUDI_QMAN_ID_DMA3",
                                                     "GAUDI_QMAN_ID_DMA4",
                                                     "GAUDI_QMAN_ID_DMA5",
                                                     "GAUDI_QMAN_ID_DMA6",
                                                     "GAUDI_QMAN_ID_DMA7",

                                                     "GAUDI_QMAN_ID_MME0",
                                                     "GAUDI_QMAN_ID_MME1",
                                                     "GAUDI_QMAN_ID_MME2",
                                                     "GAUDI_QMAN_ID_MME3",

                                                     "GAUDI_QMAN_ID_TPC0",
                                                     "GAUDI_QMAN_ID_TPC1",
                                                     "GAUDI_QMAN_ID_TPC2",
                                                     "GAUDI_QMAN_ID_TPC3",
                                                     "GAUDI_QMAN_ID_TPC4",
                                                     "GAUDI_QMAN_ID_TPC5",
                                                     "GAUDI_QMAN_ID_TPC6",
                                                     "GAUDI_QMAN_ID_TPC7",

                                                     "GAUDI_QMAN_ID_NIC0_QM0",
                                                     "GAUDI_QMAN_ID_NIC0_QM1",
                                                     "GAUDI_QMAN_ID_NIC1_QM0",
                                                     "GAUDI_QMAN_ID_NIC1_QM1",
                                                     "GAUDI_QMAN_ID_NIC2_QM0",
                                                     "GAUDI_QMAN_ID_NIC2_QM1",
                                                     "GAUDI_QMAN_ID_NIC3_QM0",
                                                     "GAUDI_QMAN_ID_NIC3_QM1",
                                                     "GAUDI_QMAN_ID_NIC4_QM0",
                                                     "GAUDI_QMAN_ID_NIC4_QM1",
                                                     "UNKNOWN"};

    HB_ASSERT(id < sizeof(qmanNames)/sizeof(qmanNames[0]), "invalid queue id");
    return qmanNames[id];
}
