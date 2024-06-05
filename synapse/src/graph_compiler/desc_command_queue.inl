#pragma once

#include "cache_types.h"
#include "command_queue.h"
#include <cstdint>
#include <memory>
#include "habana_global_conf.h"
#include "habana_nodes/node.h"
#include "graph_compiler/habana_nodes/mme_node.h"
#include "graph_compiler/habana_nodes/dma_node.h"
#include "graph_compiler/habana_nodes/habana_nodes.h"
#include "node_annotation.h"
#include "queue_command.h"
#include "queue_command_factory.h"
#include "infra/defs.h"
#include "address_fields_container_info.h"
#include "define_synapse_common.hpp"

//---------------------------------------------------------
//--------------------- DescCommandQueue-------------------
//------------ Base for all descriptor-based queues -------
//---------------------------------------------------------

template <typename DescType>
DescCommandQueue<DescType>::DescCommandQueue(const QueueCommandFactory& cmdFactory, unsigned queId, HabanaDeviceType devType)
  : CommandQueue(cmdFactory, queId, devType),
    m_allPatchpointsDropped(false)
{}

template<typename DescType>
DescCommandQueue<DescType>::~DescCommandQueue() = default;

// addLoadDesc -> Adds a list of commands, returns whether added atleast one.
template<typename DescType>
bool DescCommandQueue<DescType>::addLoadDesc(const NodePtr&            n,
                                             DescSection               descSection,
                                             BasicFieldsContainerInfo* pBasicFieldsContainerInfo,
                                             uint32_t                  predicate,
                                             DescriptorShadow*         pDescriptorShadow)
{
    static BasicFieldsContainerInfo emptyContainer;
    if (pBasicFieldsContainerInfo == nullptr) pBasicFieldsContainerInfo = &emptyContainer;
    DescriptorShadow& descriptorShadow = (pDescriptorShadow == nullptr) ? getDescShadow(n) : *pDescriptorShadow;

    // Calculate minimal bulks of registers that we must write.
    // The groupby returns a list of segments where all the entries in a segment are either of type WriteExecute or
    // WritePatching but they are not mixed, meaning, in a given segment we will not have one WriteExecute entry and
    // another WritePatching entry.
    Segments coalescedSegments = groupby(
        descSection.sizeInRegs(),
        [&](uint32_t i) {
            return descriptorShadow.getWriteType(descSection.regArray()[i], i + descSection.offsetInRegs());
        },
        0,
        /* ignore=*/DescriptorShadow::WriteType::NoWrite);

    if (coalescedSegments.empty())
    {
        return false; // nothing will be written
    }

    // create a write_bulk command for a given segment
    auto createWriteBulkCommand = [&](Segment segment)
    {
        return m_commandFactory.getWriteManyRegisters(
            m_commandFactory.getRegForLoadDesc(GetDeviceType(), GetEngineID(), n) + descSection.offset + segment.start() * sizeof(uint32_t),
            segment.size(),
            descSection.regArray() + segment.start(),
            predicate);
    };

    QueueCommandPtr               cache;
    std::vector<QueueCommandPtr>  bulkCommands;
    uint32_t                      i = 0;

    while(i < coalescedSegments.size())
    {
        Segment currentSegment = coalescedSegments[i];
        auto currentCommand = cache ? std::move(cache) : createWriteBulkCommand(currentSegment);
        i++;

        while (i < coalescedSegments.size())
        {
            QueueCommandPtr nextCommand = createWriteBulkCommand(coalescedSegments[i]);
            auto combinedSegment = coalescedSegments[i].join(currentSegment);
            auto combinedCommand = createWriteBulkCommand(combinedSegment);
            // We check if it's more efficient to join bulk writes to one than to leave them separate.
            // descriptorShadow.canJoin is slower atm so it's second
            if ((nextCommand->GetBinarySize() + currentCommand->GetBinarySize()) >= combinedCommand->GetBinarySize() &&
                descriptorShadow.canJoin(currentSegment, coalescedSegments[i], descSection.offsetInRegs()))
            {
                currentSegment = combinedSegment;
                currentCommand = std::move(combinedCommand);
                i++;
            }
            else
            {
                cache = std::move(nextCommand); // for next iteration.
                break;
            }
        }
        // Here we've join the segments as much as we can, currentCommand is the largest most efficient bulk write.
        // i currently points to the next segment that needs to be written.

        currentCommand->SetContainerInfo(
            pBasicFieldsContainerInfo->retrieveAndDropSegment(descSection.offsetInRegs() + currentSegment.start(),
                                                              descSection.offsetInRegs() + currentSegment.end() - 1));

        descriptorShadow.updateLoadedSegment(currentSegment.start() + descSection.offsetInRegs(),
                                             currentSegment.end() + descSection.offsetInRegs(),
                                             descSection.regArray() + currentSegment.start());

        bulkCommands.emplace_back(std::move(currentCommand));
    }

    // push commands to queue
    for (auto& ld : bulkCommands)
    {
        PushBack(std::move(ld), false);
    }

    return true;
}

template<typename DescType>
bool DescCommandQueue<DescType>::loadDescWithPredicates(pNode                                               n,
                                                        DescSection                                         descSection,
                                                        std::vector<Settable<DescriptorWrapper<DescType>>>* pPipeDescs)
{
    HB_ASSERT_PTR(pPipeDescs);

    std::vector<Settable<DescriptorWrapper<DescType>>>& pipeDescs = *pPipeDescs;
    unsigned sectionOffset = descSection.offset;

    // Construct history pointers for all descriptors if they aren't exist already
    if (m_descriptorShadowWithPred.size() < pipeDescs.size())
    {
        m_descriptorShadowWithPred.resize(getMaxEngineCount());
    }

    bool updated = false;
    bool ret = false;
    Settable<DescriptorShadow::AllRegistersProperties> properties;
    // Duplicate and predicate each descriptor
    for (unsigned descIdx = 0; descIdx < pipeDescs.size(); ++descIdx)
    {
        if (!pipeDescs[descIdx].is_set()) continue;
        DescriptorWrapper<DescType> descWrapper    = pipeDescs[descIdx].value();
        DescType&                   desc           = descWrapper.getDescriptor();
        unsigned                    predicate      = descWrapper.getExecutionInfo().predicate;
        unsigned                    engId          = predicate - 1;
        HB_ASSERT(engId <= getMaxEngineCount(), "Engine ID is out of bounds - {}", engId);

        DescSection engSection = descSection;  // create section initialized from the original
        engSection.addr = ((char*)(&desc)) + sectionOffset;   // point to this engine's descriptor
        if (!properties.is_set())
        {
            properties.set(registersPropertiesForDesc(n, descWrapper));
        }
        m_descriptorShadowWithPred[engId].setAllRegProperties(properties.value());

        // finally, add commands to queue with diff
        updated = addLoadDesc(n,
                              engSection,
                              &descWrapper.getBasicFieldsContainerInfo(),
                              predicate,
                              &m_descriptorShadowWithPred[engId]);

        if (descIdx == GetEngineIndex()) ret = updated; // record the updated flag only for this queue
    }

    return ret;
}

template<typename DescType>
std::vector<DescSection> DescCommandQueue<DescType>::getPredicatedSections(pNode n, const DescType& desc) const
{
    return {};
}

template<typename DescType>
std::vector<DescSection> DescCommandQueue<DescType>::getUnpredicatedSections(pNode n, const DescType& desc) const
{
    return {DescSection(desc)};
}

template<typename DescType>
QueueCommandPtr DescCommandQueue<DescType>::createSkipSignal(SyncObject& syncObj, NodeROI* roi)
{
    // Always have an engine barrier, it prevent a race condition that the QMAN signal for the next
    // activation (if it is disabled) before the engine finished the current activation
    return m_commandFactory.getSignalSemaphoreWithPredicate(syncObj.id,
                                                            syncObj.value * roi->numSignals,
                                                            SKIP_PREDICATE,
                                                            syncObj.operation,
                                                            syncObj.barrier | ENGINE_BARRIER);
}

template<typename DescType>
QueueCommandPtr DescCommandQueue<DescType>::createExecuteCommand(const pNode&                       n,
                                                                 const DescriptorWrapper<DescType>& descWrap,
                                                                 std::shared_ptr<SyncObject>        syncPtr,
                                                                 NodeROI*                           roi,
                                                                 bool                               isLastDescForRoi,
                                                                 bool                               isLastInPipeline)
{
    BypassType enableBypass = ENABLE_BYPASS;
    // execute the operation
    QueueCommandPtr ex = getExeCmd(n, descWrap, isLastInPipeline);
    // Null ROI == empty job
    // Only Gaudi and GaudiM have signaling from the queue
    // Gaudi2/3 do the signaling from the ARC
    if (isSignalingFromQman() && roi && n->isROIDynamic(roi))
    {
        BasicFieldsContainerInfo exBfci;
        exBfci.setRoi(roi);

        const auto mmeNode = std::dynamic_pointer_cast<MmeNode>(n);
        size_t signalCount = DynamicExecuteFieldInfo::NO_SIGNAL;
        FieldType fieldType = FieldType::FIELD_DYNAMIC_EXECUTE_NO_SIGNAL;
        std::vector<QueueCommandPtr> sigVector;

        if (mmeNode != nullptr && syncPtr && isLastInPipeline)
        {
            // In the mme in dynamic shapes we have a single ROI that there are 2 descriptors
            // pointing to (One per mme engine).
            enableBypass = isLastDescForRoi ? ENABLE_BYPASS : DISABLE_BYPASS;
            signalCount = DynamicExecuteFieldInfo::MME_SIGNAL_COUNT;
            fieldType = FieldType::FIELD_DYNAMIC_EXECUTE_MME;

            SyncObject secondSync = *syncPtr;
            secondSync.id++;
            sigVector.push_back(createSkipSignal(*syncPtr, roi));
            sigVector.push_back(createSkipSignal(secondSync, roi));
        }
        else if (syncPtr && isLastInPipeline)
        {
            signalCount = DynamicExecuteFieldInfo::SINGLE_SIGNAL;
            fieldType = FieldType::FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL;
            sigVector.push_back(createSkipSignal(*syncPtr, roi));
        }

        const auto dmaNode = std::dynamic_pointer_cast<DMANode>(n);
        if (dmaNode != nullptr)
        {
            if (dmaNode->isDynamicMemoryOp()) {
                enableBypass = ENABLE_BYPASS_ONLY_COMPLETLY_OUT;
            }
        }
        auto executePredField = std::make_shared<DynamicExecuteFieldInfo>(fieldType, signalCount, n, roi);
        exBfci.add(executePredField);

        ex = m_commandFactory.getDynamicExecute(std::move(ex), sigVector, enableBypass);
        ex->SetContainerInfo(exBfci);
    }

    ex->setAsExe();
    return ex;
}

template<typename DescType>
void DescCommandQueue<DescType>::AddPartialNode(
    pNode                                               n,
    DescriptorWrapper<DescType>&                        descWrapper,
    std::vector<Settable<DescriptorWrapper<DescType>>>* pPipeDescs, // all descriptors of current pipeline stage
    unsigned                                            pipeStage,
    bool                                                isSetup,
    const std::vector<uint64_t>&                        baseRegsCache, // list of cache resident section IDs
    bool                                                isLastPipelineLevel,
    std::vector<QueueCommandPtr>*                       preSyncCmds,
    std::vector<QueueCommandPtr>*                       postSyncCmds,
    bool                                                isFirstInEnginePerLevel,
    bool                                                isLastInEnginePerLevel)
{
    HB_ASSERT_PTR(n);
    HB_ASSERT((preSyncCmds == nullptr) == (postSyncCmds == nullptr),
              "either preSyncCmds and postSyncCmds are both null or both non-null");
    DescType desc = descWrapper.getDescriptor();
    BasicFieldsContainerInfo *pBasicFieldsContainerInfo = &descWrapper.getBasicFieldsContainerInfo();
    std::list<MonObject> monitors;
    std::list<SyncObject> cpSyncs;
    std::shared_ptr<SyncObject> syncPtr = nullptr;

    // Fill property mask for current descriptor's registers
    getDescShadow(n).setAllRegProperties(registersPropertiesForDesc(n, descWrapper));

    std::vector<SyncInteraction> &syncScheme = n->getNodeAnnotation().syncScheme;

    if (GetEngineIndex() < syncScheme.size())  // legacy sync scheme
    {
        SyncInteraction &engineSyncScheme = syncScheme[GetEngineIndex()];
        if (pipeStage < engineSyncScheme.pipelineSyncs.size())
        {
            PipelineSyncScheme &pipelineSyncs = engineSyncScheme.pipelineSyncs[pipeStage];
            // Add pipeline stage monitor after descriptor is loaded
            monitors = pipelineSyncs.monitors;
            cpSyncs = pipelineSyncs.cpSyncs;
            syncPtr = pipelineSyncs.sync;
        }
    }

    if (isLastInEnginePerLevel)
    {
        // only last engine per level should signal, previous rounds should avoid it
        setDescriptorSignaling(desc, syncPtr);
    }

    bool terminateBlobBeforeDesc = false;
    bool terminateBlobBeforeExe = false;
    QueueCommand *lastDescWriteCmd = nullptr;

    // only first engine activation in pipeline level should monitor, next rounds can skip it
    if (isFirstInEnginePerLevel)
    {
        if (m_armMonBeforeDesc)
        {
            for (const auto &monitor : monitors)
            {
                addMonitor(monitor, isSetup, MonitorCommandParts::SetupArm);
            }
            if (monitors.size()) terminateBlobBeforeDesc = true;
        }
    }

    // wait on mutex between the QMAN and its engine if required (goya2 HW bug WA)
    if (m_qmanMutexRequired)
    {
        PushBack(std::move(m_commandFactory.getWaitForQmanMutex()), false);
        terminateBlobBeforeDesc = true;
    }

    if (terminateBlobBeforeDesc) delimitBlobOnCmd(m_queueExe.back().get(), n->getExecutionOrderedIndex());

    bool updated = false;

    // Drop address patch-points when you can and update their property mask accordingly
    optimizePatchpoints(n, desc, pBasicFieldsContainerInfo, baseRegsCache);

    // load the descriptor which might be a non-consecutive desriptor broken into sections
    for (auto &descSection : getPredicatedSections(n, desc))
    {
        bool newCommandsAdded = loadDescWithPredicates(n, descSection, pPipeDescs);
        updated = updated || newCommandsAdded;
        if (newCommandsAdded && descSection.isCommitter)
        {
            delimitBlobOnCmd(m_queueExe.back().get(), n->getExecutionOrderedIndex());
        }
    }
    for (auto &descSection : getUnpredicatedSections(n, desc))
    {
        bool newCommandsAdded = addLoadDesc(n, descSection, pBasicFieldsContainerInfo);
        updated = updated || newCommandsAdded;
        if (newCommandsAdded && descSection.isCommitter)
        {
            delimitBlobOnCmd(m_queueExe.back().get(), n->getExecutionOrderedIndex());
        }
    }
    HB_ASSERT(!m_queueExe.empty(), "not expecting to have empty queue at this point");
    HB_ASSERT(updated || allowNoDescUpdates(n), "Descriptor is identical to previous descriptor");
    lastDescWriteCmd = m_queueExe.back().get();

    // only first engine activation in pipeline level should monitor and do pre sync cmds, next rounds can skip it
    if (isFirstInEnginePerLevel)
    {
        if (preSyncCmds)
        {
            for (auto &cmd : *preSyncCmds)
            {
                PushBack(std::move(cmd), isSetup);
                terminateBlobBeforeExe = true;
            }
            preSyncCmds->clear();
        }

        auto monParts = m_armMonBeforeDesc ? MonitorCommandParts::Fence : MonitorCommandParts::SetupArmFence;
        for (const auto &monitor : monitors)
        {
            addMonitor(monitor, isSetup, monParts);
            terminateBlobBeforeExe = !m_armMonBeforeDesc; // if we separated the arm, don't separate the fence and execute from the main descriptor blob
        }
    }

    // only last engine activation in pipeline level should signal and do post syncs, previous rounds should skip it
    if (isLastInEnginePerLevel)
    {
        // add cpSyncs before the descriptor is executed
        for (const auto &sync : cpSyncs)
        {
            addSignal(sync, isSetup);
            terminateBlobBeforeExe = true;
        }

        if (postSyncCmds)
        {
            for (auto &cmd : *postSyncCmds)
            {
                PushBack(std::move(cmd), isSetup);
                terminateBlobBeforeExe = true;
            }
            postSyncCmds->clear();
        }
    }

    if (terminateBlobBeforeExe) delimitBlobOnCmd(lastDescWriteCmd, n->getExecutionOrderedIndex());

    bool isLastDescForRoi = true;
    if (pPipeDescs != nullptr)
    {
        unsigned setElements = std::count_if(pPipeDescs->begin(), pPipeDescs->end(),
                                             [](Settable<DescriptorWrapper<DescType>> k) { return k.is_set(); });

        isLastDescForRoi = &descWrapper == &((pPipeDescs->data()[setElements - 1]).value());
    }

    // Supporting ARC architecture - Force Static Configuration:
    // The MME (and currently also the DMA) is operating in static mode even in mode 3 (WD with new Sync Scheme).
    // As a result it will have multiple queues, one for each physical engine. To stay out of trouble with the CQ
    // switching sequence, we want to ensure we will always have static configuration between engine execution for
    // all engines; otherwise, if only one of the engines got back-to-back execution without static config in-between
    // (due to the diff mechanism), then we won't have identical CQ switching scheme across the engines and we won't
    // be able to share the same dynamic ECB list - which we *must* share. The function forceStaticConfig() will push
    // NOP to the queue (if needed) to avoid this undesired situation (note that it is queue-specific overloaded).
    forceStaticConfig();

    bool cqSwitchRequired = GCFG_ARC_ARCHITECTURE.value();

    // If the top of the queue is Execute, which means we have back-to-back executes with no static config
    // in between, we need to cancel the switch of the previous execute. And if the top of the queue is static
    // config, we need to switch the CQ just before the upcoming Execute. So toggling works either way.
    if (cqSwitchRequired && !m_queueExe.empty()) m_queueExe.back()->toggleSwitchCQ();

    // From here and downward we are pushing dynamic commands only

    QueueCommandPtr exeCmd = createExecuteCommand(n,
                                                  descWrapper,
                                                  syncPtr,
                                                  pBasicFieldsContainerInfo->getRoi(),
                                                  isLastDescForRoi,
                                                  isLastInEnginePerLevel);

    PushBack(std::move(exeCmd), false); // execute the operation
    pushAdditionalDynamicCmds(n, pipeStage, isLastPipelineLevel); // push virtual commands (e.g: SFG, ResetSobs, etc.)
    delimitBlobOnCmd(m_queueExe.back().get(), n->getExecutionOrderedIndex());

    // Assume we will have static configuration after this execution, so switch. If eventually we won't have, we will
    // cancel the switch in the next iteration (and if this is the last command of the workload, we will handle
    // the last switch to the dynamic CQ in the ECB command generation using a Nop with SwitchCQ bit on).
    if (cqSwitchRequired) m_queueExe.back()->setSwitchCQ();

    if (m_queueExe.back()->invalidateHistory())
    {
        getDescShadow(n).invalidateRegs(getInvalidRegsIndices());
    }

    getDescShadow(n).movePresentPropertiesToPastProperties();
    updateQueueStateAfterPush(n);
}

template <typename DescType>
QueueCommandPtr
DescCommandQueue<DescType>::getExeCmd(pNode n, const DescriptorWrapper<DescType>& descWrap, bool enableSignal)
{
    return m_commandFactory.getExecute(GetDeviceType(), GetEngineID());
}

template <typename DescType>
void DescCommandQueue<DescType>::updateQueueStateAfterPush(pNode n)
{
    // Give sub-classes the opportunity to do something at the end of engine activation by overwriting this function
}

template <typename DescType>
bool DescCommandQueue<DescType>::allowNoDescUpdates(pNode n)
{
    // if we succeeded to drop all patchpoints, it is possible that a descriptor will have no update at all
    return m_allPatchpointsDropped;
}

template<typename DescType>
DescriptorShadow::AllRegistersProperties DescCommandQueue<DescType>::registersPropertiesForDesc(
    pNode                               n,
    const DescriptorWrapper<DescType>&  descWrapper)
{
    std::vector<DescriptorShadow::RegisterProperties> registerProperties(
        MASK_SIZE(DescType),
        DescriptorShadow::RegisterProperties::createFromHandling(DescriptorShadow::RegisterDataHandling::Data));

    if (descWrapper.getMask().has_value())
    {
        ValidityMask<DescType> mask = descWrapper.getMask().value();

        for(size_t i=0; i < mask.size(); i++)
        {
            if (!mask[i])
            {
                registerProperties[i].ignore = true;
            }
        }
    }
    return std::make_shared<std::vector<DescriptorShadow::RegisterProperties>>(registerProperties);
}

template <typename DescType>
void DescCommandQueue<DescType>::delimitBlobOnCmd(QueueCommand* cmd, unsigned nodeExeIdx)
{
    HB_ASSERT_PTR(cmd);
    cmd->setAsBlobCommitter(nodeExeIdx);
}

template <typename DescType>
void DescCommandQueue<DescType>::finalizeQueue(bool isSetup)
{
    CommandQueue::finalizeQueue(isSetup);
}

template <typename DescType>
void DescCommandQueue<DescType>::finalizeInitQueue(bool isSetup)
{
    CommandQueue::finalizeInitQueue(isSetup);
}

// Try to drop patchpoints and set the shadow mask for Patching
template<typename DescType>
void DescCommandQueue<DescType>::optimizePatchpoints(const NodePtr&               n,
                                                     const DescType&              desc,
                                                     BasicFieldsContainerInfo*    bfci,
                                                     const std::vector<uint64_t>& baseRegsCache)
{
    if (bfci == nullptr) return;
    optimizeDsdPatchpoints(n, bfci);
    optimizeMcidPatchpoints(n, desc, bfci, baseRegsCache);
    optimizeAddressPatchpoints(n, desc, bfci, baseRegsCache);
}

// Try to drop address patchpoints and set the shadow mask for Patching
template<typename DescType>
void DescCommandQueue<DescType>::optimizeAddressPatchpoints(const NodePtr&               n,
                                                            const DescType&              desc,
                                                            BasicFieldsContainerInfo*    bfci,
                                                            const std::vector<uint64_t>& baseRegsCache)
{
    if (bfci == nullptr) return;

    DescriptorShadow& shadow = getDescShadow(n);
    auto afciItr = bfci->retrieveAddressFieldInfoSet().begin();

    while (afciItr != bfci->retrieveAddressFieldInfoSet().end())
    {
        const AddressFieldInfoPair& addressInfoPair = *afciItr++; // move itr as we may be deleting the current element
        AddressFieldInfoSharedPtr   patchPoint      = addressInfoPair.second;

        // Low part will be processed with its high counterpart
        if (patchPoint->getAddressPart() == FIELD_ADDRESS_PART_LOW) continue;

        uint32_t                  fieldOffsetLow  = 0;
        uint32_t                  fieldOffsetHigh = 0;
        AddressFieldInfoSharedPtr lowPartSharedPtr;

        // Lambda to delete the current patchpoint (full or high+low) and update its shadow mask
        auto patchPointEraser = [&](DescriptorShadow::RegisterProperties prop)
        {
            shadow.setPropertiesAt(fieldOffsetLow,  prop);
            shadow.setPropertiesAt(fieldOffsetHigh, prop);
            bfci->retrieveAddressFieldInfoSet().erase(addressInfoPair);
            if (patchPoint->getAddressPart() == FIELD_ADDRESS_PART_HIGH)
            {
                if (fieldOffsetLow == (*afciItr).first) afciItr++; // in case we are just about to delete our itr
                bfci->retrieveAddressFieldInfoSet().erase({fieldOffsetLow, lowPartSharedPtr});
            }
        };

        // Get the patchpoint positions (i.e. offsets within descriptor).
        // If we process now high-part patchpoint, get also its low counterpart.
        std::tie(fieldOffsetLow, fieldOffsetHigh, lowPartSharedPtr) = getLowAndHighPositions(patchPoint);

        // Validity Check: low and high handling must be identical
        HB_ASSERT(shadow.getRegHandling(fieldOffsetLow) == shadow.getRegHandling(fieldOffsetHigh), "current handling mismatch");
        HB_ASSERT(shadow.getPastRegHandling(fieldOffsetLow) == shadow.getPastRegHandling(fieldOffsetHigh), "past handling mismatch");

        // Get the current handling
        DescriptorShadow::RegisterDataHandling lowPartHandling = shadow.getRegHandling(fieldOffsetLow);

        // Skip over dynamic-shape patchpoints, they ae handled elsewhere
        if (lowPartHandling == DescriptorShadow::RegisterDataHandling::DynamicShapePatching) continue;

        // Optimization #1:
        // If the shadow mask is 'ignore', erase the patchpoint (this case probably never happens in real-life)
        if (lowPartHandling == DescriptorShadow::RegisterDataHandling::Ignore)
        {
            patchPointEraser(DescriptorShadow::RegisterProperties::getIgnore());
            continue; // move on to next patch point
        }

        // Extract the 64bit address from the descriptor
        uint32_t* pDesc = (uint32_t*)&desc;
        ptrToInt  descVal;
        descVal.u32[0] = *(pDesc + fieldOffsetLow);
        descVal.u32[1] = *(pDesc + fieldOffsetHigh);

        // Optimization #2:
        // Compare the 64bit address from the descriptor to the one from the shadow and if they are identical,
        // erase the patchpoints. Do this optimization only if:
        //   1. the DIFF mechanism is enabled and
        //   2. the register is NOT marked with AlwaysWritePatching and
        //   3. the previous register's handling was patching
        if (!GCFG_DISABLE_LOAD_DIFF_DESC.value() &&
            lowPartHandling != DescriptorShadow::RegisterDataHandling::AlwaysWritePatching &&
            shadow.isPatching(shadow.getPastRegHandling(fieldOffsetLow)))
        {
            Settable<uint32_t> shadowValLow  = shadow.getDataAt(fieldOffsetLow);
            Settable<uint32_t> shadowValHigh = shadow.getDataAt(fieldOffsetHigh);
            HB_ASSERT(shadowValLow.is_set() == shadowValHigh.is_set(), "either both parts of address are set or both not set");
            if (shadowValLow.is_set())
            {
                ptrToInt shadowVal;
                shadowVal.u32[0] = shadowValLow.value();
                shadowVal.u32[1] = shadowValHigh.value();

                if (shadowVal.u64 == descVal.u64)
                {
                    patchPointEraser(DescriptorShadow::RegisterProperties::getOptOutPatching());
                    continue; // move on to next patch point
                }
            }
        }

        // Optimization #3:
        // If the memory section address of these patchpoints is loaded in the Base Registers Cache, push to queue
        // WriteReg64 command (wreg64), update the shadow history and erase the patchpoints.
        auto itr = std::find(baseRegsCache.begin(), baseRegsCache.end(), patchPoint->getMemorySectionId());
        if (itr != baseRegsCache.end())
        {
            unsigned cacheIndex = itr - baseRegsCache.begin();
            pushWriteReg64Commands(n, cacheIndex, fieldOffsetLow, fieldOffsetHigh, descVal);
            shadow.updateLoadedReg(fieldOffsetLow, descVal.u32[0]);
            shadow.updateLoadedReg(fieldOffsetHigh, descVal.u32[1]);
            patchPointEraser(DescriptorShadow::RegisterProperties::getOptOutPatching());
            LOG_TRACE(BASE_REGS_CACHE,
                      "Found sectionID {} in Base Regs Cache at entry {}, dropping patchpoint and using wreg64 instead",
                      patchPoint->getMemorySectionId(),
                      cacheIndex);
            continue; // move on to next patch point
        }

        // No choice left, set the mask to 'Patching' for standard run-time patching.
        shadow.addRegHandlingAt(fieldOffsetLow,  DescriptorShadow::RegisterDataHandling::Patching);
        shadow.addRegHandlingAt(fieldOffsetHigh, DescriptorShadow::RegisterDataHandling::Patching);
    }

    m_allPatchpointsDropped = bfci->retrieveAddressFieldInfoSet().empty();
    validateAllAddressPatchpointsDropped(n);
}

template<typename DescType>
void DescCommandQueue<DescType>::optimizeMcidPatchpoints(const NodePtr&               n,
                                                         const DescType&              desc,
                                                         BasicFieldsContainerInfo*    bfci,
                                                         const std::vector<uint64_t>& baseRegsCache)
{
    if (bfci == nullptr) return;

    DescriptorShadow& shadow = getDescShadow(n);
    BasicFieldInfoSet& ppSet = bfci->retrieveBasicFieldInfoSet();
    auto mcidPPItr = ppSet.begin();

    unsigned degradeBaseRegIndex = getMcidBaseRegsFirstIndex();
    unsigned discardBaseRegIndex = degradeBaseRegIndex + 2; // The interface with the firmware dictates that the discard entry is two entries after the degrade

    while (mcidPPItr != ppSet.end())
    {
        auto currIterator = mcidPPItr++; // move itr as we may be deleting the current element

        const BasicFieldInfoPair& basicFieldInfoPair = *currIterator;
        BasicFieldInfoSharedPtr patchPoint = basicFieldInfoPair.second;

        // Skip if this is not an MCID patch point
        if (!patchPoint->isMcidPatchPoint())
        {
            continue; // move on to next patch point
        }

        // Get register offset in descriptor
        uint32_t offset = patchPoint->getFieldIndexOffset();

        // Get the current handling
        DescriptorShadow::RegisterDataHandling ppHandling = shadow.getRegHandling(offset);

        // Get register value
        uint32_t* pDesc = (uint32_t*)&desc;
        uint32_t  regVal;
        regVal = *(pDesc + offset);

        /******************************************** MCID Patch Point Optimizations **********************************************/

        // Optimization #1:
        // If the shadow mask is 'ignore', erase the patchpoint (this case probably never happens in real-life)
        if (ppHandling == DescriptorShadow::RegisterDataHandling::Ignore)
        {
            ppSet.erase(currIterator);
            continue; // move on to next patch point
        }

        // Optimization #2:
        // Compare the 32bit reg value from the descriptor to the one from the shadow and if they are identical,
        // erase the patchpoints. Do this optimization only if:
        //   1. the DIFF mechanism is enabled and
        //   2. the register is NOT marked with AlwaysWritePatching and
        //   3. the previous register's handling was patching
        if (!GCFG_DISABLE_LOAD_DIFF_DESC.value() &&
            ppHandling != DescriptorShadow::RegisterDataHandling::AlwaysWritePatching &&
            shadow.isPatching(shadow.getPastRegHandling(offset)))
        {
            Settable<uint32_t> shadowVal = shadow.getDataAt(offset);
            if (shadowVal.is_set())
            {
                if (shadowVal.value() == regVal)
                {
                    ppSet.erase(currIterator);
                    shadow.setPropertiesAt(offset, DescriptorShadow::RegisterProperties::getOptOutPatching());
                    continue; // move on to next patch point
                }
            }
        }

        // Convert patchpoint to wreg64:
        // Create WriteReg64 command (wreg64), update the shadow history and erase the patchpoints.
        auto mcidPP = std::dynamic_pointer_cast<McidFieldInfo>(patchPoint);
        HB_ASSERT_PTR(mcidPP);
        unsigned cacheIndex = mcidPP->getCmAction() == DEGRADE ? degradeBaseRegIndex : discardBaseRegIndex;

        unsigned target = m_commandFactory.getRegForLoadDesc(GetDeviceType(), GetEngineID(), n) + (offset * sizeof(uint32_t));
        PushBack(m_commandFactory.getWriteReg64(cacheIndex, regVal, target, true, false), false);

        shadow.updateLoadedReg(offset, regVal);
        ppSet.erase(currIterator);
        shadow.setPropertiesAt(offset, DescriptorShadow::RegisterProperties::getOptOutPatching());
    }
}

template<typename DescType>
void DescCommandQueue<DescType>::optimizeDsdPatchpoints(const NodePtr& n, BasicFieldsContainerInfo* bfci)
{
    BasicFieldInfoSet& ppSet = bfci->retrieveBasicFieldInfoSet();
    auto dynamicPPItr = ppSet.begin();

    while (dynamicPPItr != ppSet.end())
    {
        auto currIterator = dynamicPPItr++; // move itr as we may be deleting the current element
        const BasicFieldInfoPair& basicFieldInfoPair = *currIterator;
        BasicFieldInfoSharedPtr patchPoint = basicFieldInfoPair.second;

        if (!patchPoint->isDynamicShape())
        {
            continue;
        }

        // We cannot shadow context fields, only descriptror fields
        if (patchPoint->isContextField())
        {
            continue;
        }

        DynamicShapeFieldInfoSharedPtr dynamicPP = std::dynamic_pointer_cast<DynamicShapeFieldInfo>(patchPoint);
        auto handling = DescriptorShadow::RegisterDataHandling::DynamicShapePatching;

        // If a dynamic shape pp is marked as maskable - it means it can be optimized out. A former PP already
        // patched the field to the correct dynamic value. Now this PP can be deleted, and the field should be
        // marked as banned. This will prevent overriding the correct value, and will keep the shadow history empty
        // (As it was cleaned by the first dynamic patch point in this field).
        if (dynamicPP->isMaskable())
        {
            handling = DescriptorShadow::RegisterDataHandling::Banned;
            ppSet.erase(currIterator);
        }

        for (size_t i = 0; i < dynamicPP->getSize(); i++)
        {
            getDescShadow(n).addRegHandlingAt(dynamicPP->getFieldIndexOffset() + i, handling);
        }
    }
}

// High patch point has a weak pointer to its Low counterpart, but Full and Low patch points have nullptr in that field,
// so the tuple returned by getLowAndHighPositions is: <low_position, high_position, low_patchpoint (if exist)>
// Regardlessly, the input patch point to getLowAndHighPositions is either Full or High, but never Low.
template<typename DescType>
std::tuple<uint32_t, uint32_t, AddressFieldInfoSharedPtr>
DescCommandQueue<DescType>::getLowAndHighPositions(AddressFieldInfoSharedPtr patchPoint)
{
    uint32_t fieldOffsetLow  = 0;
    uint32_t fieldOffsetHigh = 0;

    if (patchPoint->getAddressPart() == FIELD_ADDRESS_PART_HIGH)
    {
        fieldOffsetHigh = patchPoint->getFieldIndexOffset();

        if (auto spt = patchPoint->getOtherAddressFieldInfo())
        {
            fieldOffsetLow = spt->getFieldIndexOffset();
            HB_ASSERT(patchPoint->getEngineFieldId() == spt->getEngineFieldId(), "mismatched low/high patch points");
        }
        else
        {
            HB_ASSERT(0, "low part patch-point has expired? impossible!");
        }
    }
    else if (patchPoint->getAddressPart() == FIELD_ADDRESS_PART_FULL)
    {
        fieldOffsetLow  = patchPoint->getFieldIndexOffset();
        fieldOffsetHigh = fieldOffsetLow + 1;
    }
    else
    {
        HB_ASSERT(0, "expecting to get only high or full patch-point types here");
    }
    return std::make_tuple(fieldOffsetLow, fieldOffsetHigh, patchPoint->getOtherAddressFieldInfo());
}

template<typename DescType>
void DescCommandQueue<DescType>::pushWriteReg64Commands(const NodePtr& n,
                                                        unsigned       cacheIndex,
                                                        uint32_t       fieldOffsetLow,
                                                        uint32_t       fieldOffsetHigh,
                                                        ptrToInt       descVal)
{
    unsigned target = m_commandFactory.getRegForLoadDesc(GetDeviceType(), GetEngineID(), n) + (fieldOffsetLow * sizeof(uint32_t));

    // We keep only the offset in section (by masking out the section ID from the high bits). Later, the HW will
    // add the section base while it executes the wreg64 packet which is abstracted by the WriteReg64 command.
    uint64_t offsetInSection = maskOutMemoryID(descVal.u64);

    if (fieldOffsetLow + 1 == fieldOffsetHigh) // either FULL patchpoint or *consecutive* LOW and HIGH
    {
        PushBack(m_commandFactory.getWriteReg64(cacheIndex, offsetInSection, target), false);
    }
    else
    {
        PushBack(m_commandFactory.getWriteReg64(cacheIndex, offsetInSection, target, true, false), false);
        target = m_commandFactory.getRegForLoadDesc(GetDeviceType(), GetEngineID(), n) + (fieldOffsetHigh * sizeof(uint32_t));
        PushBack(m_commandFactory.getWriteReg64(cacheIndex, offsetInSection, target, false, true), false);
    }
}

template<typename DescType>
void DescCommandQueue<DescType>::validateAllAddressPatchpointsDropped(const NodePtr& n) const
{
    // The default expectation is that all legacy patchpoints will get dropped since we are using the base-register
    // mechanism. Gaudi1 and goya2 overwrite this implementation and they are not validate anything. TPC of gaudi2
    // also overwrites because it has some corner case exception. Dynamic shape and rotator nodes have some exceptions
    // so we don't check them.
    if (n == nullptr || n->isDynamicShape() || n->isRotate()) return;
    HB_ASSERT(m_allPatchpointsDropped, "found unexpected legacy patchpoints");
}

//---------------------------------------------------------
//------------------------ MmeQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
MmeQueue<DescType>::MmeQueue(const QueueCommandFactory& cmdFactory, unsigned queId)
  : DescCommandQueue<DescType>(cmdFactory, queId, DEVICE_MME)
{}

template <typename DescType>
MmeQueue<DescType>::~MmeQueue()
{}

template <typename DescType>
unsigned MmeQueue<DescType>::getMaxEngineCount()
{
    return getMaxMmeEnginesCount();
}

template <typename DescType>
void MmeQueue<DescType>::AddNode(const pNode& n, HabanaGraph* g, bool isSetup)
{
    HB_ASSERT_PTR(n);
    HB_ASSERT_PTR(g);

    CommandQueue::addPreSyncScheme(n, isSetup);

    std::vector< DescriptorWrapper<DescType> > descriptorsWrappers;
    getDescriptorsWrappers(n, g, descriptorsWrappers);
    unsigned int numDescs = (unsigned)descriptorsWrappers.size();
    unsigned int pipelineLevel = 0;

    for (unsigned int descIndex = 0; descIndex < numDescs; ++descIndex)
    {
        DescriptorWrapper<DescType>& wrapper = descriptorsWrappers[descIndex];

        DescCommandQueue<DescType>::AddPartialNode(n, wrapper, nullptr, pipelineLevel, isSetup, {},
                                                   (descIndex == numDescs - 1),
                                                   nullptr, //preSyncCmds
                                                   nullptr, //postSyncCmds
                                                   true,
                                                   true);
        if (canSignal(wrapper.getDescriptor()))
        {
            ++pipelineLevel;
        }
    }

    CommandQueue::addPostExeSyncs(n, isSetup);
}

//---------------------------------------------------------
//------------------------ TpcQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
TpcQueue<DescType>::TpcQueue(const QueueCommandFactory& cmdFactory, unsigned queId)
  : DescCommandQueue<DescType>(cmdFactory, queId, DEVICE_TPC)
{}

template <typename DescType>
TpcQueue<DescType>::~TpcQueue()
{}

template <typename DescType>
unsigned TpcQueue<DescType>::getMaxEngineCount()
{
    return getMaxTpcEnginesCount();
}

//---------------------------------------------------------
//------------------------ RotatorQueue -----------------------
//---------------------------------------------------------

template <typename DescType>
RotatorQueue<DescType>::RotatorQueue(const QueueCommandFactory& cmdFactory, unsigned queId)
        : DescCommandQueue<DescType>(cmdFactory, queId, DEVICE_ROTATOR)
{}

template <typename DescType>
RotatorQueue<DescType>::~RotatorQueue()
{}

template <typename DescType>
unsigned RotatorQueue<DescType>::getMaxEngineCount()
{
    return getMaxRotatorEnginesCount();
}
//---------------------------------------------------------
//---------------------- DmaDescQueue ---------------------
//---------------------------------------------------------

template <typename DescType>
DmaDescQueue<DescType>::DmaDescQueue(const QueueCommandFactory& cmdFactory, unsigned queId)
  : DescCommandQueue<DescType>(cmdFactory, queId, DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL)
{}

template <typename DescType>
DmaDescQueue<DescType>::~DmaDescQueue()
{}

template <typename DescType>
unsigned DmaDescQueue<DescType>::getMaxEngineCount()
{
    return getMaxDmaEnginesCount();
}
