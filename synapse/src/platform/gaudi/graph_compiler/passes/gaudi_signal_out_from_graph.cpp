#include "gaudi_signal_out_from_graph.h"

#include "graph_compiler/compilation_hal_reader.h"
#include "habana_device_types.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "node_annotation.h"
#include "node.h"
#include "passes.h"
#include "syn_logging.h"
#include "sync_scheme_manager.h"
#include "sync_types.h"
#include "types_exception.h"
#include "types.h"

#include <algorithm>
#include <memory>
#include <vector>

constexpr auto MAX_NUM_OF_MONITOR_OBJECTS_FOR_SFG = 400;
constexpr auto MIN_NUM_OF_MONITOR_OBJECTS_FOR_SFG = 8;

void SigOutSyncGroup::init(const HabanaGraph&            g,
                           SyncObjectManager::SyncId     firstSyncId,
                           uint32_t                      groupSize,
                           std::vector<HabanaDeviceType> supportedDevices,
                           const HalReader&              halReader)
{
    m_firstSyncId = firstSyncId;

    auto current = firstSyncId;
    for (auto& device : supportedDevices)
    {
        auto deviceGroups = div_round_up(getNumEnginesForDeviceType(device, halReader), groupSize);
        std::vector<SyncObjectManager::SyncId> ids;
        ids.reserve(deviceGroups);
        for (int i = 0; i < deviceGroups; i++)
        {
            ids.push_back(current++);
        }
        m_habanaDeviceToSyncId[device] = ids;
    }
    m_total = current - firstSyncId + 1;
    HB_ASSERT(m_total <= g.getCodeGenerator()->getSyncObjectManager()->getSyncConventions().getNumOfSignalGroups(),
              "Too many groups for sigout: {}",
              current - firstSyncId);
}

void GaudiSignalOutFromGraph::setInitialSyncValues(HabanaGraph& g)
{
    for (auto& keyValue : m_deviceToSignals)
    {
        auto startSignal = keyValue.second[0];
        if (startSignal == 0)
        {
            // Init value for SFG is 0 - no need to set it up
            continue;
        }

        LOG_DEBUG(SFG, "Init value for device: {} is: {}", getDeviceName(keyValue.first), keyValue.second[0]);

        SyncObject s;
        s.operation = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getSyncSetOp();
        s.value     = startSignal;
        SyncOrMonitor som;
        som.type = SyncOrMonitor::SYNC_OBJ;
        som.sync = s;

        for (auto& sync : m_sigOutGroup.getSyncsForDevice(keyValue.first))
        {
            som.sync.id = sync;

            unsigned engineIdx = 0;
            if (keyValue.first == DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL && GCFG_MEMSET_PARALLEL_LEVEL.value())
            {
                // When working in 1+4 mode (first DMA used for memset) engineIdx 1 represents the
                // first memcpy DMA engine
                engineIdx = 1;
            }
            m_codeGenerator.getInitialSyncInstructionsByQueueId()[keyValue.first][engineIdx].push_back(som);
        }
    }
}

unsigned GaudiSignalOutFromGraph::getAvailableMonitor()
{
    HB_ASSERT(!m_monitorObjects.empty(), "Insufficient monitor objects for signaling from graph");

    unsigned monitorId = m_monitorObjects.front();
    m_monitorObjects.pop();

    return monitorId;
}

void GaudiSignalOutFromGraph::dumpPatchableMonitor(HabanaGraph& g, const TensorPtr& t, int armValue)
{
    auto     groupSize = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getGroupSize();
    unsigned syncId    = m_sigOutGroup.getFirstSyncId() / groupSize;

    LOG_TRACE(SFG,
              "Patchable monitor created for tensor: {}, tensorId: {}. syncId to wait for: {}, armVal: {}, setupVal: "
              "{}, mask: {}",
              t->getName(),
              t->getId(),
              syncId * groupSize,
              armValue,
              1,
              m_sigOutGroup.getMask());
}

PatchableMonitor
GaudiSignalOutFromGraph::getPatchableMonitor(HabanaGraph& g, const TensorPtr& t, int armValue, int setupValue)
{
    auto      groupSize = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getGroupSize();
    MonObject patchableMonitorTemplate;
    patchableMonitorTemplate.operation = MONITOR_SO_OP_GREQ;
    // ArmMonitor always multiply the syncId by 8 in case we have mask, so we divide here by 8
    patchableMonitorTemplate.syncId         = m_sigOutGroup.getFirstSyncId() / groupSize;
    patchableMonitorTemplate.setupValue     = setupValue;
    patchableMonitorTemplate.shouldInc      = true;
    patchableMonitorTemplate.signalSyncId   = m_codeGenerator.getSyncObjectManager()->getDummySyncId();  // to be patched
    patchableMonitorTemplate.fenceId        = WaitID::ID_0;
    patchableMonitorTemplate.fenceTargetVal = 0;
    patchableMonitorTemplate.mask           = m_sigOutGroup.getMask();
    PatchableMonitor patchable {patchableMonitorTemplate};

    patchable.monObject.id = getAvailableMonitor();

    LOG_DEBUG(SFG,
              "Patchable monitor id: {} created for tensor: {}, tensorId: {}. syncId to wait for: {}, armVal: {}, "
              "setupVal: {}, mask: {}",
              patchable.monObject.id,
              t->getName(),
              t->getId(),
              patchableMonitorTemplate.syncId * groupSize,
              armValue,
              setupValue,
              patchableMonitorTemplate.mask.value());

    patchable.monObject.armValue = armValue;
    patchable.tensorId           = t->getId();
    return patchable;
}

void GaudiSignalOutFromGraph::dumpSigOutGroupMonitor(HabanaGraph&     g,
                                                     const NodePtr&   producer,
                                                     uint32_t         setupValue,
                                                     const TensorPtr& t)
{
    auto     device       = g.getNodeUtility().getNodeDeviceType(producer);
    auto     groupSize    = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getGroupSize();
    auto     numEngines   = producer->getNodeAnnotation().syncScheme.size();
    auto     firstSyncPtr = producer->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().sync;
    unsigned syncId       = firstSyncPtr->id / groupSize;
    unsigned armValue     = producer->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().syncTotalValue;

    auto     syncsForDevice = m_sigOutGroup.getSyncsForDevice(device);
    unsigned signalSyncId   = syncsForDevice[0];

    unsigned maskable = (numEngines <= groupSize) ? numEngines : groupSize;
    unsigned mask     = (1 << maskable) - 1;

    LOG_TRACE(SFG,
              "Monitor created for node: {}. syncId to wait for: {}, armVal: {}, syncId to signal: {}, "
              "inc: {}, setupVal: {}, mask: {}",
              producer->getNodeName(),
              syncId,
              armValue,
              signalSyncId,
              true,
              setupValue,
              mask);
}

void GaudiSignalOutFromGraph::dumpSigOutGroupMonitors(HabanaGraph&     g,
                                                      const NodeSet&   producers,
                                                      const TensorPtr& t,
                                                      int              index)
{
    // Normal patch points
    for (auto& producer : producers)
    {
        auto device = g.getNodeUtility().getNodeDeviceType(producer);
        dumpSigOutGroupMonitor(g, producer, m_tensorsToSignals[index][device], t);
    }
}

SyncOrMonitor GaudiSignalOutFromGraph::getMonitorsToUpdateTheSigOutGroup(HabanaGraph&     g,
                                                                         const NodePtr&   producer,
                                                                         uint32_t         setupValue,
                                                                         const TensorPtr& t)
{
    auto                   device       = g.getNodeUtility().getNodeDeviceType(producer);
    auto                   groupSize    = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getGroupSize();
    auto                   numEngines   = producer->getNodeAnnotation().syncScheme.size();
    auto                   firstSyncPtr = producer->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().sync;
    MonObject              monitor;
    monitor.operation      = MONITOR_SO_OP_GREQ;
    monitor.setupValue     = setupValue;
    monitor.shouldInc      = true;
    monitor.fenceId        = WaitID::ID_0;
    monitor.fenceTargetVal = 0;
    auto syncsForDevice    = m_sigOutGroup.getSyncsForDevice(device);

    unsigned maskable = (numEngines <= groupSize) ? numEngines : groupSize;

    monitor.id = getAvailableMonitor();

    monitor.mask         = (1 << maskable) - 1;
    monitor.armValue     = producer->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().syncTotalValue;
    monitor.syncId       = firstSyncPtr->id / groupSize;
    monitor.signalSyncId = syncsForDevice[0];

    LOG_DEBUG(SFG,
              "Monitor id: {} created for node: {}. syncId to wait for: {}, armVal: {}, syncId to signal: {}, "
              "inc: {}, setupVal: {}, mask: {}",
              monitor.id,
              producer->getNodeName(),
              monitor.syncId,
              monitor.armValue,
              monitor.signalSyncId,
              monitor.shouldInc,
              monitor.setupValue,
              monitor.mask.value());

    // Register the monitor
    SyncOrMonitor som;
    som.monitor = monitor;
    som.type    = SyncOrMonitor::MONITOR_OBJ;

    return som;
}

void GaudiSignalOutFromGraph::init(HabanaGraph& g)
{
    commonInit(g);

    auto groupSize = m_codeGenerator.getSyncObjectManager()->getSyncConventions().getGroupSize();

    m_sigOutGroup.init(g,
                       m_codeGenerator.getSyncObjectManager()->getSyncConventions().getSignalOutGroup(),
                       groupSize,
                       m_supportedDevices,
                       *g.getHALReader());

    setInitialSyncValues(g);

    unsigned availableMonitors = GCFG_SFG_MAX_NUM_OF_MONITORS.value();
    if (availableMonitors > MAX_NUM_OF_MONITOR_OBJECTS_FOR_SFG ||
        availableMonitors < MIN_NUM_OF_MONITOR_OBJECTS_FOR_SFG)
    {
        availableMonitors = MAX_NUM_OF_MONITOR_OBJECTS_FOR_SFG;
    }

    unsigned monId;

    while (m_monitorObjects.size() < availableMonitors &&
           (monId = m_codeGenerator.getSyncObjectManager()->getFreeMonitorId()) != -1)
    {
        m_monitorObjects.push(monId);
    }

    // At the very least we need 8 monitors - 2 monitors (left/right) per engine (TPC/MME/DMA/ROT)
    if (m_monitorObjects.size() < MIN_NUM_OF_MONITOR_OBJECTS_FOR_SFG)
    {
        throw SynapseException(
            fmt::format("Insufficient monitor objects for signaling from graph. Allocated: {}, minimum required: {}",
                        m_monitorObjects.size(),
                        MIN_NUM_OF_MONITOR_OBJECTS_FOR_SFG));
    }

    if (m_monitorObjects.size())
    {
        LOG_DEBUG(
            SFG,
            "Num of signal-out tensors: {}, Num of reserved monitors: {}, first monitor Id: {}, last monitor Id: {}",
            m_outTensorsSorted.size(),
            m_monitorObjects.size(),
            m_monitorObjects.front(),
            m_monitorObjects.back());

        allocateBuckets(m_monitorObjects.size());
    }
}

////////////////////////////////////////////// allocateBuckets ///////////////////////////////////////
// General idea:
//
// 1. Calculate for each engine the number of left monitors - meaning the number of external tensors this engine type is
// producing
// 2. Allocate the reserved 400 monitors proportionally between engine type. Each allocation is split to 2 - left/right
// 3. Generate buckets list per engine
// 4. Avoid "starvation" - no monitors allocated for engine(s)

// For example:

// There are total of 275 external tensors. We have 400 monitors to split between them

// TPC nodes are responsible to produce 177 of them - 64.3%  --> 257 monitors (128 / 128)
// MME nodes are responsible to produce 81 of them  - 29.4%  --> 118 monitors (59 / 59)
// DMA nodes are responsible to produce 17 of them  - 6.2%   --> 25  monitors (12 / 12)

// DMA buckets list: 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1   (12 buckets for 17 monitors)

// Tests examples:
// ===============

// Num of signal-out tensors: 12, Num of reserved monitors: 50, first monitor Id: 47, last monitor Id: 96
// Num of external tensors produced by engine type: DEVICE_TPC is: 11
// Num of external tensors produced by engine type: DEVICE_DMA is: 1
// Total number of monitors needed: 24
// Ratio for engine DEVICE_TPC: 0.916667, allocated monitors: 46
// Ratio for engine DEVICE_DMA: 0.0833333, allocated monitors: 4
// Engine DEVICE_TPC: available left/right monitors: 23/23
// Engine DEVICE_TPC: actual left monitors: 11
// Engine DEVICE_DMA: available left/right monitors: 2/2
// Engine DEVICE_DMA: actual left monitors: 1
// Total number of actual monitors used: 24
// Num of buckets for engine DEVICE_TPC is: 11. Buckets: 1 1 1 1 1 1 1 1 1 1 1
// Num of buckets for engine DEVICE_DMA is: 1. Buckets: 1

// Num of signal-out tensors: 12, Num of reserved monitors: 8, first monitor Id: 47, last monitor Id: 54
// Num of external tensors produced by engine type: DEVICE_TPC is: 11
// Num of external tensors produced by engine type: DEVICE_DMA is: 1
// Total number of monitors needed: 24
// Ratio for engine DEVICE_TPC: 0.916667, allocated monitors: 7
// Ratio for engine DEVICE_DMA: 0.0833333, allocated monitors: 2
// Total num of monitors allocated (9) is bigger than the number of available monitors (8) - adjusting the allocation
// Decrement engine DEVICE_TPC allocated monitors (7) by 1
// Engine DEVICE_TPC: available left/right monitors: 3/3
// Engine DEVICE_TPC: actual left monitors: 3
// Engine DEVICE_DMA: available left/right monitors: 1/1
// Engine DEVICE_DMA: actual left monitors: 1
// Total number of actual monitors used: 8
// Num of buckets for engine DEVICE_TPC is: 3. Buckets: 4 4 3
// Num of buckets for engine DEVICE_DMA is: 1. Buckets: 1

// Num of signal-out tensors: 401, Num of reserved monitors: 400, first monitor Id: 47, last monitor Id: 446
// Num of external tensors produced by engine type: DEVICE_TPC is: 301
// Num of external tensors produced by engine type: DEVICE_DMA is: 100
// Total number of monitors needed: 802
// Ratio for engine DEVICE_TPC: 0.750623, allocated monitors: 300
// Ratio for engine DEVICE_DMA: 0.249377, allocated monitors: 100
// Engine DEVICE_TPC: available left/right monitors: 150/150
// Engine DEVICE_TPC: actual left monitors: 150
// Engine DEVICE_DMA: available left/right monitors: 50/50
// Engine DEVICE_DMA: actual left monitors: 50
// Total number of actual monitors used: 400
// Num of buckets for engine DEVICE_TPC is: 150. Buckets: 3 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 2 2 2 2 2 2 2 2 2 2 2 2 2 2
// Num of buckets for engine DEVICE_DMA is: 50. Buckets: 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 2 2 2 2 2 2 2 2 2 2 2 2 2 2

//////////////////////////////////////////////////////////////////////////////////////////////////////
void GaudiSignalOutFromGraph::allocateBuckets(unsigned availableMonitors)
{
    // find the total number of left monitors needed for each engine type
    unsigned totalLeftMonitorsNeeded = 0;
    unsigned engineExtTensorCount    = 0;

    std::map<HabanaDeviceType, unsigned> engineLeftMonitors;

    for (auto producer : m_deviceToSignals)
    {
        engineExtTensorCount = producer.second.size() - 1;
        totalLeftMonitorsNeeded += engineExtTensorCount;
        engineLeftMonitors[producer.first] = engineExtTensorCount;
        LOG_DEBUG(SFG,
                  "Num of external tensors produced by engine type: {} is: {}",
                  getDeviceName(producer.first),
                  engineExtTensorCount);
    }

    LOG_DEBUG(SFG, "Total number of monitors needed: {}", totalLeftMonitorsNeeded + m_outTensorsSorted.size());

    // split the total monitors proportionally according to device type usage
    std::map<HabanaDeviceType, unsigned> engineAllocatedMonitors;
    splitMonitors(engineAllocatedMonitors, engineLeftMonitors, totalLeftMonitorsNeeded, availableMonitors);

    unsigned numOfActualMonitorsUsed = 0;

    for (auto engineMon : engineAllocatedMonitors)
    {
        unsigned leftMonitors = engineMon.second / 2;

        LOG_DEBUG(SFG,
                  "Engine {}: available left/right monitors: {}/{}",
                  getDeviceName(engineMon.first),
                  leftMonitors,
                  leftMonitors);

        generateBucketForEngine(engineMon.first, leftMonitors, engineLeftMonitors[engineMon.first]);

        numOfActualMonitorsUsed += leftMonitors + leftMonitors;  // left=right

        LOG_DEBUG(SFG, "Engine {}: actual left monitors: {}", getDeviceName(engineMon.first), leftMonitors);
    }

    LOG_DEBUG(SFG, "Total number of actual monitors used: {}", numOfActualMonitorsUsed);

    for (auto engBucket : m_engineToBuckets)
    {
        m_currentBucketIndex[engBucket.first]       = 0;
        m_currentIndexInBucket[engBucket.first]     = 0;
        m_bucketSetupValue[engBucket.first]         = 0;
        m_bucketMultiProdTensorDec[engBucket.first] = 0;

        if (LOG_LEVEL_AT_LEAST_DEBUG(SFG))
        {
            std::stringstream s;
            for (auto it = engBucket.second.begin(); it != engBucket.second.end(); ++it)
            {
                s << *it << " ";
            }
            LOG_DEBUG(SFG,
                      "Num of buckets for engine {} is: {}. Buckets: {}",
                      getDeviceName(engBucket.first),
                      engBucket.second.size(),
                      s.str());
        }
    }
}

void GaudiSignalOutFromGraph::splitMonitors(std::map<HabanaDeviceType, unsigned>& engineMonitors,
                                            std::map<HabanaDeviceType, unsigned>& engineLeftMonitorsNeeded,
                                            unsigned                              totalMonitorsNeeded,
                                            unsigned                              availableMonitors)
{
    unsigned totalMonitorsAllocated = 0;

    for (auto engineMon : engineLeftMonitorsNeeded)
    {
        // Calculate the total number of left monitors needed for each engine type
        float ratio = float(engineMon.second) / float(totalMonitorsNeeded);

        // Rounding up to even number will result in losing less monitors when splitting the total number to left/right
        unsigned allocatedMonitorsPerEngine = round(ratio * availableMonitors);
        allocatedMonitorsPerEngine += allocatedMonitorsPerEngine % 2;

        if (allocatedMonitorsPerEngine < 2)
        {
            // Regardless of how the monitors split between engines, need to make sure each engine has at least 2
            allocatedMonitorsPerEngine = 2;
        }

        totalMonitorsAllocated += allocatedMonitorsPerEngine;
        engineMonitors[engineMon.first] = allocatedMonitorsPerEngine;

        LOG_DEBUG(SFG,
                  "Ratio for engine {}: {}, allocated monitors: {}",
                  getDeviceName(engineMon.first),
                  ratio,
                  allocatedMonitorsPerEngine);
    }
    // In rare cases we may allocate more monitors then we actually reserved, so we need to adjust the allocation
    // For example: we allocated: (400, 2, 2, 2) = 406  OR  (2, 380, 2, 20) = 404
    // The desired allocation is: (394, 2, 2, 2) = 400  OR  (2, 378, 2, 18) = 400

    while (totalMonitorsAllocated > availableMonitors)
    {
        LOG_DEBUG(
            SFG,
            "Total num of monitors allocated ({}) is bigger than the number of available monitors ({}) - adjusting "
            "the allocation",
            totalMonitorsAllocated,
            availableMonitors);

        unsigned diff = totalMonitorsAllocated - availableMonitors;
        while (diff)
        {
            for (auto engineMon : engineMonitors)
            {
                if (engineMon.second > 2)
                {
                    LOG_DEBUG(SFG,
                              "Decrement engine {} allocated monitors ({}) by 2",
                              getDeviceName(engineMon.first),
                              engineMon.second);
                    engineMonitors[engineMon.first] -= 2;
                    diff -= 2;
                    totalMonitorsAllocated -= 2;
                    if (diff == 0)
                    {
                        break;
                    }
                }
            }
            break;
        }
    }
}

void GaudiSignalOutFromGraph::generateBucketForEngine(HabanaDeviceType engType,
                                                      unsigned&        leftMonitors,
                                                      unsigned         numSigOutTensors)
{
    // Determine the bucket size and generate the list of buckets per engine type
    if (numSigOutTensors < leftMonitors)
    {
        // Avoid 0 size buckets in case the initial left monitors is greater then the actual number of sig-out tensors
        // Practically it means that all buckets has the size 1 - similar to working in legacy (non-bucketing) mode
        leftMonitors = numSigOutTensors;
    }
    unsigned baseBucketSize = numSigOutTensors / leftMonitors;

    std::vector<unsigned> buckets;

    for (int i = 0; i < leftMonitors; i++)
    {
        buckets.push_back(baseBucketSize);
    }

    unsigned diff = numSigOutTensors - (leftMonitors * baseBucketSize);

    for (int i = 0; i < diff; i++)
    {
        buckets[i]++;
    }

    m_engineToBuckets[engType] = buckets;
}

////////////////////////////////////////////// addBucketsMonitors ///////////////////////////////////////
// General flow:
//
// We iterate on the list of external tensors. We maintain DB of buckets per engine type. We generate left/right
// monitors only at the end of the bucket. Otherwise we just aggregate the current open bucket setup value in
// m_bucketSetupValue[device].
// When a bucket is closed we generate left monitor and right monitor:

// The left monitor represent monitor that is linked to the graph hence the armValue is being taken from the
// nodeAnnotation syncScheme. Its setup value represent the value we want to increase the engine SOB - it was
// aggregated for all external tensors in the same bucket

// The right (patchable) monitor is used to notify the user that external tensor(s) is ready. Its arm value is simply
// the index of the current tensor while its setup value is the bucket size (since we want to notify the user that
// several tensors are ready)

// Special handling for multiple-producers tensors. It affects only the right monitor setup value. In case tensor
// is produced by more than one node (different engine type) we want to make sure that only one bucket will "report"
// the readiness of the tensor

// std::map<HabanaDeviceType, std::vector<unsigned>> m_engineToBuckets      -> bucket list per engine (4,4,3,3,3)
// std::map<HabanaDeviceType, unsigned>              m_currentBucketIndex   -> current bucket index   (0/1/2/3/4)
// std::map<HabanaDeviceType, unsigned>              m_currentIndexInBucket -> current index in current bucket (1/2/3/4)
// std::map<HabanaDeviceType, unsigned>              m_bucketSetupValue     -> aggregated setupValue for left monitor

// For multi-produced tensors
// std::set<unsigned>                   m_multiProducedTensorIndex --> set of tensors with multiple producers
// std::map<HabanaDeviceType, unsigned> m_bucketMultiProdTensorDec --> value to be decremented for right monitor setup

/////////////////////////////////////////////////////////////////////////////////////////////////////////
void GaudiSignalOutFromGraph::addBucketsMonitors(HabanaGraph& g)
{
    unsigned aggregatedRightSetupVal = 0;

    for (unsigned index = 0; index < m_outTensorsSorted.size(); index++)
    {
        auto& t         = m_outTensorsSorted[index];
        auto  producers = getLastProducers(g, t);
        HB_ASSERT(!producers.empty(), "Tensor must be created by at least one node");

        for (auto& producer : producers)
        {
            auto device = g.getNodeUtility().getNodeDeviceType(producer);

            // bucket handling
            m_currentIndexInBucket[device]++;
            unsigned currentBucketSize = m_engineToBuckets[device][m_currentBucketIndex[device]];

            // handle multiple producers external tensors
            if (producers.size() > 1)
            {
                if (m_multiProducedTensorIndex.find(index) == m_multiProducedTensorIndex.end())
                {
                    // First time - insert into set so other producers will decrease their setup count
                    LOG_DEBUG(SFG,
                              "Tensor: {}, index: {} is being produced by multiple producers({}). current device: {}",
                              t->getName(),
                              index,
                              producers.size(),
                              getDeviceName(device));
                    m_multiProducedTensorIndex.insert(index);
                }
                else
                {
                    // Some other producer is already handling this multi-produced tensor - just decrease the count
                    LOG_DEBUG(SFG,
                              "Tensor: {}, index: {} is being handled by other producer - decrease setup count. "
                              "current device: {})",
                              t->getName(),
                              index,
                              getDeviceName(device));
                    m_bucketMultiProdTensorDec[device]++;
                }
            }

            if (m_currentIndexInBucket[device] == currentBucketSize)  // reached end of bucket - create monitors
            {
                // Create right (patchable) monitor
                unsigned rightArmVal   = index + 1;
                unsigned rightSetupVal = currentBucketSize - m_bucketMultiProdTensorDec[device];

                if (rightSetupVal > 0)
                {
                    PatchableMonitor patchable = getPatchableMonitor(g, t, rightArmVal, rightSetupVal);
                    producer->getNodeAnnotation().syncScheme[0].patchableMonitors.push_back(patchable);
                    aggregatedRightSetupVal += rightSetupVal;
                }

                // Create left monitor
                unsigned leftArmVal = producer->getNodeAnnotation().syncScheme[0].pipelineSyncs.back().syncTotalValue;

                // Aggregated setupValue for left monitor
                m_bucketSetupValue[device] += m_tensorsToSignals[index][device];
                unsigned leftSetupVal = m_bucketSetupValue[device];

                auto som = getMonitorsToUpdateTheSigOutGroup(g, producer, leftSetupVal, t);
                producer->getNodeAnnotation().syncScheme[0].preSyncsAndMon.push_back(som);

                LOG_DEBUG(
                    SFG,
                    "Device: {}, bucket: {}, size: {}, left mon: arm: {}, setup: {}, right mon: arm: {}, setup: {}",
                    getDeviceName(device),
                    m_currentBucketIndex[device],
                    currentBucketSize,
                    leftArmVal,
                    leftSetupVal,
                    rightArmVal,
                    rightSetupVal);

                // Move to next bucket. Reset necessary fields for new bucket
                m_currentBucketIndex[device]++;
                m_currentIndexInBucket[device]     = 0;
                m_bucketSetupValue[device]         = 0;
                m_bucketMultiProdTensorDec[device] = 0;
            }
            else
            {
                // Aggregated setupValue for left monitor
                m_bucketSetupValue[device] += m_tensorsToSignals[index][device];
            }
        }
    }

    HB_ASSERT(aggregatedRightSetupVal == m_outTensorsSorted.size(),
              "Patchable aggregated setup value ({}) is not equal to num of external tensors ({})",
              aggregatedRightSetupVal,
              m_outTensorsSorted.size());
}

void GaudiSignalOutFromGraph::dumpLegacySFGMonitors(HabanaGraph& g)
{
    // legacy mode (no bucketing) dump
    LOG_TRACE(SFG, "Legacy SFG Dump:");

    for (int i = 0; i < m_outTensorsSorted.size(); i++)
    {
        auto& t         = m_outTensorsSorted[i];
        auto  producers = getLastProducers(g, t);
        HB_ASSERT(!producers.empty(), "Tensor must be created by at least one node");
        auto lastProducer =
            *std::max_element(producers.begin(), producers.end(), [](const NodePtr& lhs, const NodePtr& rhs) {
                return lhs->getExecutionOrderedIndex() < rhs->getExecutionOrderedIndex();
            });

        dumpPatchableMonitor(g, t, i + 1);

        dumpSigOutGroupMonitors(g, producers, t, i);
    }
}

bool GaudiSignalOutFromGraph::executePass(HabanaGraph& g)
{
    try
    {
        init(g);

        addBucketsMonitors(g);

        if (LOG_LEVEL_AT_LEAST_TRACE(SFG))
        {
            dumpLegacySFGMonitors(g);
        }

        return true;
    }
    catch (const SynapseException& e)
    {
        LOG_CRITICAL(GC, "Signaling from graph pass failed: {}", e.what());
        return false;
    }
}

namespace gaudi
{
bool signalOutFromGraph(GaudiGraph& g)
{
    GaudiSignalOutFromGraph sigOut(*g.getCodeGenerator());

    return sigOut.executePass(g);
}
}  // namespace gaudi
