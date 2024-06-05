#include "sync_scheme_manager_base.h"

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "desc_gen/desc_gen_hal.h"
#include "desc_gen/node2desc.h"

namespace eager_mode
{
void SyncSchemeManagerBase::generateNodesArcSyncScheme(SingleNode2Desc& singleNode2Desc) const
{
    DescGeneratorBase& descGen         = singleNode2Desc.getDescGen();
    const size_t       logicalRoisSize = descGen.getLogicalRoiNr();
    EAGER_ASSERT(logicalRoisSize != 0, "Invalid logical ROIs");
    auto& arcSyncSchemeVec = descGen.getNode().getNodeAnnotation().arcSyncScheme;
    arcSyncSchemeVec.resize(logicalRoisSize);
    bool     breakpointMode = GCFG_ENABLE_BREAKPOINT_MODE.value();
    uint64_t breakpointCtr  = 0;

    if (logicalRoisSize > 1)
    {
        // Update emitting signal and pipeline dependencies
        const unsigned logicalId = m_descGenHal.deviceTypeToLogicalQueue(descGen.getEngineType(), descGen.getNode());
        unsigned       signalCnt = 0;
        for (ArcSyncInteraction& arcSyncScheme : arcSyncSchemeVec)
        {
            arcSyncScheme.dependencies[logicalId] = signalCnt;
            signalCnt                             = m_descGenHal.safeIncrement(logicalId, signalCnt, 1);
            arcSyncScheme.emittedSigVal.set(signalCnt);
            if (unlikely(breakpointMode))
            {
                arcSyncScheme.breakpoint.set(++breakpointCtr);
            }
        }
    }
}

void SyncSchemeManagerBase::generateNodesArcSyncScheme(Node2DescContainer& multiNode2Desc) const
{
    auto& execSequence = multiNode2Desc.getExecSequence();
    if (execSequence.size() == 1) return generateNodesArcSyncScheme(execSequence.front());
    bool                                                         breakpointMode = GCFG_ENABLE_BREAKPOINT_MODE.value();
    uint64_t                                                     breakpointCtr  = 0;
    std::array<unsigned, DescGeneratorHal::LOGICAL_QUEUE_MAX_ID> sigArray = {0};
    for (SingleNode2Desc& singleNode : execSequence)
    {
        DescGeneratorBase& descGen = singleNode.getDescGen();
        Node&              node    = descGen.getNode();
        EAGER_ASSERT(!node.isLogicalOperation(), "Logical nodes should not be executed on device");

        const size_t logicalRoisSize = descGen.getLogicalRoiNr();
        EAGER_ASSERT(logicalRoisSize == 1, "Multiple logical ROIs are not supported");
        node.getNodeAnnotation().arcSyncScheme.resize(logicalRoisSize);
        ArcSyncInteraction& arcSyncScheme = node.getNodeAnnotation().arcSyncScheme.back();

        // Update dependency upon previous physical producer node
        const EagerNode* prevNode = singleNode.getLatestPhysicalProducer();
        if (prevNode != nullptr)
        {
            EAGER_ASSERT((*prevNode)->isLogicalOperation() == false,
                         "Physical node should not rely on logical nodes for device signaling");
            // Dependency signals array is accumulative
            EAGER_ASSERT((*prevNode)->getNodeAnnotation().arcSyncScheme.empty() == false, "Invalid ARC sync scheme");
            const ArcSyncInteraction& prevArcSyncScheme = (*prevNode)->getNodeAnnotation().arcSyncScheme.back();
            EAGER_ASSERT(prevArcSyncScheme.emittedSigVal.is_set(), "Node depends on another one that succeeds it");
            const unsigned prevLogicalId = m_descGenHal.deviceTypeToLogicalQueue(prevNode->getEngineType(), *prevNode);
            arcSyncScheme.dependencies[prevLogicalId] += prevArcSyncScheme.emittedSigVal.value();
        }

        // Update emitting signal and pipeline dependencies
        const unsigned logicalId = m_descGenHal.deviceTypeToLogicalQueue(descGen.getEngineType(), descGen.getNode());
        arcSyncScheme.dependencies[logicalId] = sigArray[logicalId];
        sigArray[logicalId]                   = m_descGenHal.safeIncrement(logicalId, sigArray[logicalId], 1);
        arcSyncScheme.emittedSigVal.set(sigArray[logicalId]);

        if (unlikely(breakpointMode))
        {
            arcSyncScheme.breakpoint.set(++breakpointCtr);
        }
    }
}

}  // namespace eager_mode
