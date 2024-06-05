#include "sync_scheme_manager_arc.h"

#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "sync_utils.h"
#include "types_exception.h"

#include <sstream>

SyncSchemeManagerArcGaudi2::SyncSchemeManagerArcGaudi2(Gaudi2Graph* graph)
: SyncSchemeManagerArc(graph),
  m_pOverlap(std::make_shared<OverlapArcGaudi2>()),
  m_pConverter(std::make_shared<OverlapSigToArcSigGaudi2>())
{
    m_sobResetMngr.bindSafeIncrement(&OverlapSigToArcSigGaudi2::safeIncrement);
}

void SyncSchemeManagerArcGaudi2::resetOverlap()
{
    m_pOverlap.reset(new OverlapArcGaudi2());
    m_pConverter.reset(new OverlapSigToArcSigGaudi2());
}

unsigned SyncSchemeManagerArcGaudi2::getLogicalId(const NodePtr& node) const
{
    return gaudi2::deviceTypeToLogicalQueue(m_graph->getNodeUtility().getNodeDeviceType(node));
}

const SyncSchemeManagerArcGaudi2::NumEngsMap& SyncSchemeManagerArcGaudi2::getNumEngsPerLogicalId() const
{
    static SyncSchemeManagerArcGaudi2::NumEngsMap ret;
    if (ret.empty())
    {
        ret[gaudi2::DEVICE_MME_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumMmeEngines();
        ret[gaudi2::DEVICE_TPC_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumTpcEngines();
        ret[gaudi2::DEVICE_DMA_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumInternalDmaEngines();
        ret[gaudi2::DEVICE_ROT_LOGICAL_QUEUE] = m_graph->getHALReader()->getNumRotatorEngines();
    }
    return ret;
}

void SyncSchemeManagerArcGaudi2::createRoiPipelineSyncs(const NodePtr&             node,
                                                        const std::list<NodeROI*>& rois,
                                                        const DependencyMap&       inputDependencies,
                                                        DependencyMap&             outputDependencies,
                                                        unsigned&                  emittedSigVal)
{
    OverlapDescriptor desc;

    desc.numSignals        = 0;
    desc.engineID          = getLogicalId(node);
    desc.engineIDForDepCtx = desc.engineID;

    // TPC operates in Shared SOB mode
    if (desc.engineID == gaudi2::DEVICE_TPC_LOGICAL_QUEUE)
    {
        desc.minSelfWaitForSharedSob = OverlapSigToArcSigGaudi2::c_tpcSharedSobSetSize;
    }

    for (auto roi : rois)
    {
        desc.numSignals += roi->numSignals;
        generateOverlapRois(roi->inputRois, desc.inputRois);
        generateOverlapRois(roi->outputRois, desc.outputRois);
    }

    OverlapArcGaudi2::DependencyCtx overlapDep = {0};

    // Fill in input dependencies
    for (const auto& inDep : inputDependencies) // inDep.first is logical ID, and inDep.second is signal value
    {
        overlapDep.valid[inDep.first] = true;
        overlapDep.signalIdx[inDep.first] = convertSigValToOverlapIdx(inDep.first, inDep.second);
    }

    // Run the overlap
    m_pOverlap->addDescriptor(desc, overlapDep, getMaxOverlapSigIdxForNodeToDependOn(node));

    OverlapArcGaudi2::DependencyCtx arcDep = {0};
    m_pConverter->convert(desc, overlapDep, arcDep, emittedSigVal);
    validateSigVal(desc.engineID, emittedSigVal);

    for (unsigned logicalId = 0; logicalId < gaudi2::LOGICAL_QUEUE_MAX_ID; logicalId++)
    {
        if (arcDep.valid[logicalId])
        {
            validateSigVal(logicalId, arcDep.signalIdx[logicalId]);
            outputDependencies[logicalId] = arcDep.signalIdx[logicalId];
        }
    }
}

void SyncSchemeManagerArcGaudi2::validateSigVal(unsigned logicalId, unsigned sigVal) const
{
    if (sigVal > GCFG_ARC_SYNC_SCHEME_SIGNAL_LIMIT.value())
    {
        LOG_ERR(SYNC_SCHEME, "Signal value {} exceeds the maximum (logical engine={})", sigVal, logicalIdToStr(logicalId));
        throw PassFailedException();
    }
}

unsigned SyncSchemeManagerArcGaudi2::convertSigValToOverlapIdx(unsigned logicalId, unsigned sigVal) const
{
    return m_pConverter->reverseConvert(logicalId, sigVal);
}

std::string SyncSchemeManagerArcGaudi2::logicalIdToStr(unsigned logicalId) const
{
    if (logicalId == gaudi2::DEVICE_TPC_LOGICAL_QUEUE)
    {
        return std::string("TPC");
    }
    else if (logicalId == gaudi2::DEVICE_MME_LOGICAL_QUEUE)
    {
        return std::string("MME");
    }
    else if (logicalId == gaudi2::DEVICE_DMA_LOGICAL_QUEUE)
    {
        return std::string("DMA");
    }
    else if (logicalId == gaudi2::DEVICE_ROT_LOGICAL_QUEUE)
    {
        return std::string("ROT");
    }
    else
    {
        HB_ASSERT(0, "Unsupported logical ID");
        return std::string("Unknown");
    }
}

std::string SyncSchemeManagerArcGaudi2::sigValToStr(unsigned logicalId, Settable<unsigned> sigVal) const
{
    if (!sigVal.is_set()) return std::string("none");

    std::stringstream ret;
    unsigned          sigValRaw = sigVal.value();

    ret << std::to_string(sigValRaw);

    ret << " [";
    if (logicalId == gaudi2::DEVICE_TPC_LOGICAL_QUEUE)
    {
        ret << "sob_id_offset: " << (sigValRaw & 0x1F)
            << ", sob_val: "     << ((sigValRaw & OverlapSigToArcSigGaudi2::SOB_VALUE_MASK_TPC) >> 5) << " (x32)"
            << ", cycle_bit: "   << ((sigValRaw & 0x8000) >> 15);
    }
    else if (logicalId == gaudi2::DEVICE_MME_LOGICAL_QUEUE)
    {
        ret << "sob_val: "   << (sigValRaw & OverlapSigToArcSigGaudi2::SOB_VALUE_MASK_MME)
            << ", pair_id: " << ((sigValRaw & 0x6000) >> 13)
            << ", smg_id: "  << ((sigValRaw & 0x8000) >> 15);
    }
    else if (logicalId == gaudi2::DEVICE_DMA_LOGICAL_QUEUE)
    {
        ret << "sob_val: "    << (sigValRaw & OverlapSigToArcSigGaudi2::SOB_VALUE_MASK_DMA)
            << ", tuple_id: " << ((sigValRaw & 0x8000) >> 15);
    }
    else if (logicalId == gaudi2::DEVICE_ROT_LOGICAL_QUEUE)
    {
        ret << "sob_val: "  << (sigValRaw & OverlapSigToArcSigGaudi2::SOB_VALUE_MASK_ROT)
            << ", set_id: " << ((sigValRaw & 0xE000) >> 13);
    }
    else
    {
        HB_ASSERT(0, "Unsupported logical ID");
    }
    ret << "]";

    return ret.str();
}

//----------------------------------------------------------------------------
//                          OverlapSigToArcSigGaudi2
//----------------------------------------------------------------------------

OverlapSigToArcSigGaudi2::OverlapSigToArcSigGaudi2()
{
    for (unsigned logicalId = 0; logicalId < gaudi2::LOGICAL_QUEUE_MAX_ID; logicalId++)
    {
        m_sigOffsets[logicalId].emplace_back(0, 0);
    }
}

unsigned OverlapSigToArcSigGaudi2::safeIncrement(unsigned logicalId, unsigned sigVal, unsigned incAmount)
{
    // It is prohibited to have virtual SOB value that all its threshold bits are 0. Also, in the case of the MME, it
    // is prohibited to change the pair-ID and/or the SoSet in the middle of the increment (this can only happen in the
    // MME which may signal more than 1 in a single activation; the problem is that we cannot change SOB set in the
    // middle of an activation). This function increments sigVal by incAmount while avoiding the prohibited values by:
    //   1. If after the increment all the threshold bits are 0, add 1 to the threshold bits (DMA and TPC).
    //   2. For MME, if the increment amount is chaning the 3 MSBs, perform the increment but keep only the 3 MSBs
    //      and then perform the increment again; this ensures the full increment occurs on the new SOB set.
    //
    //  DMA Virtual Sync Object Breakdown:
    //
    //          |SoSet|                                                                               |
    //          | ID  |                                threashold                                     |
    //          |_____|_______________________________________________________________________________|
    //             1                                      15
    //
    //  MME Virtual Sync Object Breakdown:
    //
    //          |SoSet|           |                                                                   |
    //          | ID  |  pair ID  |                          threashold                               |
    //          |_____|_____|_____|___________________________________________________________________|
    //             1        2                                  13
    //
    //  TPC Virtual Sync Object Breakdown:
    //
    //          |SoSet|                                                 |                             |
    //          | ID  |                   threashold                    |         sync-object ID      |
    //          |_____|_________________________________________________|_____|_____|_____|_____|_____|
    //             1                          10                                       5
    //
    //  ROT Virtual Sync Object Breakdown:
    //
    //          |                 |                                                                   |
    //          |     set ID      |                          threashold                               |
    //          |_____|_____|_____|___________________________________________________________________|
    //                   3                                       13
    //

    if (incAmount == 0) return sigVal;

    if (logicalId == gaudi2::DEVICE_TPC_LOGICAL_QUEUE)
    {
        HB_ASSERT(incAmount == 1, "bad increment value for TPC");
        if (sigVal == 0) return c_tpcSharedSobSetSize;  // de-facto this means +1
        sigVal += incAmount;
        // at least one bit between bits #5 and #14 (inclusive) must be on
        if ((sigVal & SOB_VALUE_MASK_TPC) == 0) sigVal |= 0x20;  // de-facto this means +1
    }
    else if (logicalId == gaudi2::DEVICE_MME_LOGICAL_QUEUE)
    {
        HB_ASSERT(incAmount <= SOB_VALUE_MASK_MME, "bad increment value for MME");
        // check if SOB set changes
        if ((sigVal & 0xE000) != ((sigVal + incAmount) & 0xE000))
        {
            sigVal = ((sigVal + incAmount) & 0xE000);  // set only the SOB set and pair ID bits
        }
        sigVal += incAmount;
    }
    else if (logicalId == gaudi2::DEVICE_DMA_LOGICAL_QUEUE)
    {
        HB_ASSERT(incAmount == 1, "bad increment value for DMA");
        sigVal += incAmount;
        // at least one bit between bits #0 and #14 (inclusive) must be on
        if ((sigVal & SOB_VALUE_MASK_DMA) == 0) sigVal |= 1;
    }
    else if (logicalId == gaudi2::DEVICE_ROT_LOGICAL_QUEUE)
    {
        HB_ASSERT(incAmount == 1, "bad increment value for ROT");
        // check if SOB set changes
        if ((sigVal & 0xE000) != ((sigVal + incAmount) & 0xE000))
        {
            sigVal = ((sigVal + incAmount) & 0xE000);  // set only the SOB set
        }
        sigVal += incAmount;
    }
    else
    {
        HB_ASSERT(0, "unsupported logical ID in safeIncrement");
    }
    return sigVal;
}

void OverlapSigToArcSigGaudi2::convert(const OverlapDescriptor&               overlapDesc,   // input overlap descriptor
                                       const OverlapArcGaudi2::DependencyCtx& overlapDep,    // input overlap dependencies
                                       OverlapArcGaudi2::DependencyCtx&       arcDep,        // output arc dependencies
                                       unsigned&                              emittedSignal) // output emitted signal
{
    unsigned logicalId           = overlapDesc.engineID;
    unsigned naivePhysicalSigVal = m_physicalSigVal[logicalId] + overlapDesc.numSignals;
    m_physicalSigVal[logicalId]  = safeIncrement(logicalId, m_physicalSigVal[logicalId], overlapDesc.numSignals);
    unsigned gap                 = m_physicalSigVal[logicalId] - naivePhysicalSigVal;
    emittedSignal                = m_physicalSigVal[logicalId]; // set the output

    // If needed, add a new starting point of signal-offset in the overlap realm
    if (gap)
    {
        m_sigOffsets[logicalId].emplace_back(m_overlapSigIdx[logicalId], m_sigOffsets[logicalId].back().offset + gap);
    }

    // Update the accumulated overlap signal
    m_overlapSigIdx[logicalId] += overlapDesc.numSignals;

    // Convert the overlap dependency to ARC physical dependency
    for (logicalId = 0; logicalId < gaudi2::LOGICAL_QUEUE_MAX_ID; logicalId++)
    {
        arcDep.valid[logicalId] = overlapDep.valid[logicalId];
        if (arcDep.valid[logicalId])
        {
            auto rit = m_sigOffsets[logicalId].rbegin(); // note the reverse iterator
            while (overlapDep.signalIdx[logicalId] < rit->start) rit++; // find the last entry that covers the dependency
            arcDep.signalIdx[logicalId] = overlapDep.signalIdx[logicalId] + rit->offset; // rit will never be rend()
            arcDep.signalIdx[logicalId] += 1; // convert from overlap's 0-based index-realm to 1-based signal-realm
        }
    }
}

unsigned OverlapSigToArcSigGaudi2::reverseConvert(unsigned logicalId, unsigned sigVal) const
{
    HB_ASSERT(sigVal > 0, "unexpected signal value 0");
    unsigned sigIdx = sigVal - 1; // convert from 1-based signal-realm to overlap's 0-based index-realm
    auto rit = m_sigOffsets[logicalId].rbegin(); // note the reverse iterator
    while (sigIdx < rit->start + rit->offset) rit++; // find the last entry that covers the signal index
    return sigIdx - rit->offset; // rit will never be rend()
}
