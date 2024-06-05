#ifndef MME__OVERLAP_INL
#define MME__OVERLAP_INL

template<unsigned N>
Overlap<N>::Overlap()
{
}

template<unsigned N>
void Overlap<N>::handleOutputRois(
    const OverlapDescriptor &desc,
    DependencyCtx &dependencyList,
    uint32_t engineMaxSignalIdx)
{
    // iterate over output ROIs
    for (auto & roi : desc.outputRois)
    {
        const uint64_t offset = roi.offset;
        auto & segSpace = roi.isSram ? m_sram : m_vmem;

        // iterate over sub-ROIs
        for (auto & subRoi : *roi.subRois)
        {
            unsigned signalIdx = m_signalCtrs[desc.engineID] + subRoi.relSoIdx;
            bool isCyclic = !subRoi.cyclicRanges.empty();

            SyncInfo newRangeSyncInfo;
            if (isCyclic)
            {
                assert(subRoi.ranges.size() == subRoi.cyclicRanges.size());
                CyclicRangeAccess cyclicAccess;
                cyclicAccess.accessType = roi.isReduction? AccessType::RMW : AccessType::WRITE;
                cyclicAccess.engine = desc.engineID;
                cyclicAccess.signal = signalIdx;
                newRangeSyncInfo.cyclicDependency.emplace_back(cyclicAccess);
            }
            else
            {
                if (roi.isReduction)
                {
                    newRangeSyncInfo.isReduction = true;
                    newRangeSyncInfo.reductionDependency.valid[desc.engineID] = true;
                    newRangeSyncInfo.reductionDependency.signalIdx[desc.engineID] = signalIdx;
                }
                else
                {
                    newRangeSyncInfo.producerValid = true;
                    newRangeSyncInfo.producerEngine = desc.engineID;
                    newRangeSyncInfo.producerSignalIdx = signalIdx;
                }
            }

            // iterate over the linear ranges
            for (int i = 0; i < subRoi.ranges.size(); i++)
            {
                auto& range = subRoi.ranges[i];
                if (isCyclic)
                {
                    newRangeSyncInfo.cyclicDependency.back().cyclicParams = subRoi.cyclicRanges[i];
                    newRangeSyncInfo.cyclicDependency.back().cyclicParams.shift(offset);
                }

                std::vector<std::tuple<uint64_t, uint64_t, SyncInfo*>> rangesSyncInfo;
                bool allValid = segSpace.getSegments(range.start() + offset, range.size(), rangesSyncInfo);

                // iterate over the segments
                for (auto & rangeSyncInfo : rangesSyncInfo)
                {
                    if (isCyclic)
                    {
                        const CyclicRangeAccess& cyclicAccess = newRangeSyncInfo.cyclicDependency.back();
                        handleCyclicAccess(dependencyList, rangeSyncInfo, cyclicAccess, engineMaxSignalIdx);
                    }
                    else
                    {
                        handleLinearWrite(dependencyList, rangeSyncInfo, desc.engineID, signalIdx, engineMaxSignalIdx, roi.isReduction);
                    }
                }

                // insert a new sync info to the segments map
                if (!roi.isReduction && !isCyclic)
                {
                    segSpace.addSegment(range.start() + offset, range.size(), newRangeSyncInfo);
                }
                else if (!allValid)
                {
                    segSpace.addSegment(range.start() + offset, range.size(), newRangeSyncInfo, false);
                }
            }
        }
    }
}

template<unsigned N>
void Overlap<N>::handleInputRois(
    const OverlapDescriptor &desc,
    DependencyCtx &dependencyList,
    uint32_t engineMaxSignalIdx)
{
    // iterate over input ROIs
    for (auto & roi : desc.inputRois)
    {
        const uint64_t offset = roi.offset;
        auto & segSpace = roi.isSram ? m_sram : m_vmem;

        // iterate over sub-ROIs
        for (auto & subRoi : *roi.subRois)
        {
            unsigned signalIdx = m_signalCtrs[desc.engineID] + subRoi.relSoIdx;

            SyncInfo defaultReaderSyncInfo;
            bool isCyclic = !subRoi.cyclicRanges.empty();
            if (isCyclic)
            {
                assert(subRoi.ranges.size() == subRoi.cyclicRanges.size());
                CyclicRangeAccess cyclicAccess;
                cyclicAccess.accessType = READ;
                cyclicAccess.engine = desc.engineID;
                cyclicAccess.signal = signalIdx;
                defaultReaderSyncInfo.cyclicDependency.emplace_back(cyclicAccess);
            }
            else
            {
                defaultReaderSyncInfo.consumersDependency.valid[desc.engineID] = true;
                defaultReaderSyncInfo.consumersDependency.signalIdx[desc.engineID] = signalIdx;
            }

            // iterate over the linear ranges
            for (int i = 0; i < subRoi.ranges.size(); i++)
            {
                auto& range = subRoi.ranges[i];
                if (isCyclic)
                {
                    defaultReaderSyncInfo.cyclicDependency.back().cyclicParams = subRoi.cyclicRanges[i];
                    defaultReaderSyncInfo.cyclicDependency.back().cyclicParams.shift(offset);
                }

                std::vector<std::tuple<uint64_t, uint64_t, SyncInfo*>> rangesSyncInfo;
                bool allValid = segSpace.getSegments(range.start() + offset, range.size(), rangesSyncInfo);

                // iterate over the segments
                for (auto& rangeSyncInfo : rangesSyncInfo)
                {
                    if (isCyclic)
                    {
                        const CyclicRangeAccess& cyclicAccess = defaultReaderSyncInfo.cyclicDependency.back();
                        handleCyclicAccess(dependencyList, rangeSyncInfo, cyclicAccess, engineMaxSignalIdx);
                    }
                    else
                    {
                        handleLinearRead(dependencyList, rangeSyncInfo, desc.engineID, signalIdx, engineMaxSignalIdx);
                    }
                }

                // create new segments instead of the non-valid segments
                if (!allValid)
                {
                    segSpace.addSegment(range.start() + offset,
                                        range.size(),
                                        defaultReaderSyncInfo,
                                        false); /* don't overwrite existing segments*/
                }
            }
        }
    }
}

template<unsigned N>
void Overlap<N>::handleLinearWrite(
    DependencyCtx &dependencyList,
    std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
    unsigned engineID,
    unsigned signalIdx,
    uint32_t engineMaxSignalIdx,
    bool isReductionWrite)
{
    SyncInfo* rangeSyncInfo;
    uint64_t start, end;
    std::tie(start, end, rangeSyncInfo) = rangeInfo;
    getReadDependencies(dependencyList, rangeSyncInfo, engineID, engineMaxSignalIdx);              // write after read
    getWriteDependencies(dependencyList, rangeSyncInfo, engineID, engineMaxSignalIdx, signalIdx, false);  // write after write
    for (const CyclicRangeAccess& cyclicOverlappingAccess : rangeSyncInfo->cyclicDependency)
    {
        if (isReductionWrite && cyclicOverlappingAccess.accessType == AccessType::RMW) continue; // RMW after RMW
        if (!cyclicOverlappingAccess.cyclicParams.isOverlap(start, end)) continue;  // no real overlap

        getCyclicRangeDependencies(dependencyList, cyclicOverlappingAccess, engineID, engineMaxSignalIdx, signalIdx,
                                    isReductionWrite? AccessType::RMW : AccessType::WRITE);
    }

    // reduction writes
    if (isReductionWrite)
    {
        rangeSyncInfo->isReduction = true;
        rangeSyncInfo->reductionDependency.valid[engineID] = true;
        rangeSyncInfo->reductionDependency.signalIdx[engineID] = signalIdx;
    }
    else
    {
        getRMWDependencies(dependencyList, rangeSyncInfo, engineID, engineMaxSignalIdx, false);        // write after RMW
        rangeSyncInfo->cyclicDependency.clear();
    }
}

template<unsigned N>
void Overlap<N>::handleLinearRead(
    DependencyCtx &dependencyList,
    std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
    unsigned engineID,
    unsigned signalIdx,
    uint32_t engineMaxSignalIdx)
{
    SyncInfo* rangeSyncInfo;
    uint64_t start, end;
    std::tie(start, end, rangeSyncInfo) = rangeInfo;
    getWriteDependencies(dependencyList, rangeSyncInfo, engineID, engineMaxSignalIdx, signalIdx, true);
    getRMWDependencies(dependencyList, rangeSyncInfo, engineID, engineMaxSignalIdx, true);

    for (const CyclicRangeAccess& cyclicOverlappingAccess : rangeSyncInfo->cyclicDependency)
    {
        if (cyclicOverlappingAccess.accessType == AccessType::READ) continue;       // read after read
        if (!cyclicOverlappingAccess.cyclicParams.isOverlap(start, end)) continue;  // no real overlap

        getCyclicRangeDependencies(dependencyList, cyclicOverlappingAccess, engineID, engineMaxSignalIdx, signalIdx, READ);
    }

    // register the engine as consumer of the segment
    addDependency(rangeSyncInfo->consumersDependency, engineID, signalIdx);
}

template<unsigned N>
void Overlap<N>::handleCyclicAccess(
    DependencyCtx &dependencyList,
    std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
    const CyclicRangeAccess& cyclicAccess,
    uint32_t engineMaxSignalIdx)
{
    SyncInfo* rangeSyncInfo;
    uint64_t start, end;
    std::tie(start, end, rangeSyncInfo) = rangeInfo;
    AccessType accessType = cyclicAccess.accessType;

    if (!cyclicAccess.cyclicParams.isOverlap(start, end)) return; // no real overlap with segment

    // all wait for writers
    getWriteDependencies(dependencyList, rangeSyncInfo, cyclicAccess.engine, engineMaxSignalIdx, cyclicAccess.signal, accessType == AccessType::READ);
    if (accessType != AccessType::RMW) // no RMW after RMW
    {
        getRMWDependencies(dependencyList, rangeSyncInfo, cyclicAccess.engine, engineMaxSignalIdx, accessType == AccessType::READ);
    }
    if (accessType != AccessType::READ) // no READ after READ
    {
        getReadDependencies(dependencyList, rangeSyncInfo, cyclicAccess.engine, engineMaxSignalIdx);
    }

    bool updatedDep = false;
    for (CyclicRangeAccess& cyclicOverlappingAccess : rangeSyncInfo->cyclicDependency)
    {
        if (!(cyclicOverlappingAccess.accessType == AccessType::READ && accessType == AccessType::READ) &&  // not read after read
            !(cyclicOverlappingAccess.accessType == AccessType::RMW && accessType == AccessType::RMW) &&    // not RMW after RMW
            cyclicAccess.cyclicParams.isOverlap(cyclicOverlappingAccess.cyclicParams, start, end))          // real overlap
        {
            getCyclicRangeDependencies(dependencyList, cyclicOverlappingAccess, cyclicAccess.engine, engineMaxSignalIdx, cyclicAccess.signal, accessType);
        }

        // override existing cyclic range if possible (instead of adding)
        if (!updatedDep &&
            accessType == cyclicOverlappingAccess.accessType &&
            (accessType == AccessType::WRITE || cyclicAccess.engine == cyclicOverlappingAccess.engine) &&
            cyclicAccess.cyclicParams == cyclicOverlappingAccess.cyclicParams)
        {
            cyclicOverlappingAccess.signal = cyclicAccess.signal;
            cyclicOverlappingAccess.engine = cyclicAccess.engine;
            updatedDep = true;
        }
    }

    // add cyclic access to existing segment
    if (!updatedDep)
    {
        rangeSyncInfo->cyclicDependency.push_back(cyclicAccess);
    }
}

template<unsigned N>
void Overlap<N>::getReadDependencies(
    DependencyCtx &dependencyList,
    const SyncInfo* rangeSyncInfo,
    unsigned currentEngineID,
    uint32_t engineMaxSignalIdx)
{
    // iterate over the consumer engines
    for (unsigned consumerEngine = 0; consumerEngine < c_engines_nr; consumerEngine++)
    {
        if (rangeSyncInfo->consumersDependency.valid[consumerEngine])
        {
            auto value = rangeSyncInfo->consumersDependency.signalIdx[consumerEngine];
            if (consumerEngine == currentEngineID && value >= engineMaxSignalIdx)
            {
                // We want to ignore this value, but we can't, since the signal above engineMaxSignalIdx might
                // have overridden a value lower than him. Therefore, the safest way to continue is
                // to wait on the maximal signal available.
                value = engineMaxSignalIdx - 1;
                if (engineMaxSignalIdx == 0)
                {
                    continue;
                }
            }
            addDependency(dependencyList, consumerEngine, value);
        }
    }
}

template<unsigned N>
void Overlap<N>::getWriteDependencies(
    DependencyCtx &dependencyList,
    const SyncInfo* rangeSyncInfo,
    unsigned currentEngineID,
    uint32_t engineMaxSignalIdx,
    unsigned currentSignalIdx,
    bool isReadAccess)
{
    // make sure the the segment is not updated in-place (for read after write only).
    if (isReadAccess &&
        rangeSyncInfo->producerValid &&
        (rangeSyncInfo->producerEngine == currentEngineID) &&
        (rangeSyncInfo->producerSignalIdx >= m_signalCtrs[currentEngineID]))
    {
        return;
    }

    // add dependency on the producer
    if (rangeSyncInfo->producerValid)
    {
        if (rangeSyncInfo->producerEngine != currentEngineID ||
            (rangeSyncInfo->producerSignalIdx < engineMaxSignalIdx &&
                /* Protects self coverlapping ranges */
                rangeSyncInfo->producerSignalIdx < currentSignalIdx) /*(Should always be true for Read after write */)
        {
            addDependency(dependencyList, rangeSyncInfo->producerEngine, rangeSyncInfo->producerSignalIdx);
        }
    }
}

template<unsigned N>
void Overlap<N>::getRMWDependencies(
    DependencyCtx &dependencyList,
    const SyncInfo* rangeSyncInfo,
    unsigned currentEngineID,
    uint32_t engineMaxSignalIdx,
    bool isReadAccess)
{
    if (rangeSyncInfo->isReduction)
    {
        // iterate over the producer engines
        for (unsigned producerEngine = 0; producerEngine < c_engines_nr; producerEngine++)
        {
            if ((rangeSyncInfo->reductionDependency.valid[producerEngine]) &&
                // Protecting TPC in-place read-write (for read after RMW only)
                (!isReadAccess ||
                (producerEngine != currentEngineID) ||
                (rangeSyncInfo->reductionDependency.signalIdx[producerEngine] < m_signalCtrs[currentEngineID])) &&
                // Protecting TPC internal dependencies
                (producerEngine != currentEngineID ||
                rangeSyncInfo->reductionDependency.signalIdx[producerEngine] < engineMaxSignalIdx))
            {
                addDependency(dependencyList, producerEngine, rangeSyncInfo->reductionDependency.signalIdx[producerEngine]);
            }
        }
    }
}

template<unsigned N>
void Overlap<N>::addDependency(
    DependencyCtx &dependencyList,
    unsigned engine,
    unsigned signal)
{
    // Check if needs to updated the dependency
    if (!dependencyList.valid[engine] || (signal > dependencyList.signalIdx[engine]))
    {
        dependencyList.valid[engine] = true;
        dependencyList.signalIdx[engine] = signal;
    }
}

template<unsigned N>
void Overlap<N>::getCyclicRangeDependencies(
    DependencyCtx &dependencyList,
    const CyclicRangeAccess& cyclicAccess,
    unsigned currentEngineID,
    uint32_t engineMaxSignalIdx,
    unsigned currentSignalIdx,
    AccessType currentAccess)
{
    // make sure the the segment is not updated in-place (for read after write only).
    if (currentAccess == AccessType::READ &&
        (cyclicAccess.engine == currentEngineID) &&
        (cyclicAccess.signal >= m_signalCtrs[currentEngineID]))
    {
        return;
    }
    // add dependency on operation
    unsigned value = cyclicAccess.signal;
    if (cyclicAccess.engine == currentEngineID && cyclicAccess.signal >= engineMaxSignalIdx)
    {
        if (currentAccess == AccessType::WRITE && cyclicAccess.accessType == AccessType::READ)
        {
            // see getReadDependencies
            if (engineMaxSignalIdx == 0) return;
            value = engineMaxSignalIdx - 1;
        }
        else  // all other cases just ignore this dependency.
        {
            return;
        }
    }

    /* Protects self coverlapping ranges */
    if (currentAccess == WRITE && cyclicAccess.accessType == AccessType::WRITE &&
        cyclicAccess.engine == currentEngineID &&
        cyclicAccess.signal >= currentSignalIdx) return;

    addDependency(dependencyList, cyclicAccess.engine, value);
}

template<unsigned N>
void Overlap<N>::addSelfDependencyForSharedSob(
    const OverlapDescriptor& desc,
    DependencyCtx&           dependencyList) const
{
    if (desc.minSelfWaitForSharedSob == 0 || m_signalCtrs[desc.engineID] < desc.minSelfWaitForSharedSob) return;

    unsigned currentMinSelfWait = m_signalCtrs[desc.engineID] - desc.minSelfWaitForSharedSob;

    // If I don't depend on myself at all, add dependency on myself with the value currentMinSelfWait
    // If I depend on myself, but on value which is smaller (older) than currentMinSelfWait, update the target value
    // If I depend on myself and on value which is greater (newer) than currentMinSelfWait, do nothing
    if (!dependencyList.valid[desc.engineID])
    {
        dependencyList.valid[desc.engineID] = true;
        dependencyList.signalIdx[desc.engineID] = currentMinSelfWait;
    }
    else if (dependencyList.signalIdx[desc.engineID] < currentMinSelfWait)
    {
        dependencyList.signalIdx[desc.engineID] = currentMinSelfWait;
    }
}

template<unsigned N>
void Overlap<N>::updateEngineCtxAndReduceWaitList(
    const OverlapDescriptor &desc,
    DependencyCtx &dependencyList)
{
    auto &engineCtx = m_engineSyncCtx[desc.engineIDForDepCtx];

    // compare the list of reduced signals with the current engine context
    for (unsigned engineIdx = 0; engineIdx < c_engines_nr; engineIdx++)
    {
        if ((dependencyList.valid[engineIdx]) &&
            (engineCtx.valid[engineIdx]) &&
            (dependencyList.signalIdx[engineIdx] <= engineCtx.signalIdx[engineIdx]))
        {
            dependencyList.valid[engineIdx] = false;
        }
    }

    // iterate over the signals in the reduced list
    for (unsigned engineIdx = 0; engineIdx < c_engines_nr; engineIdx++)
    {
        if (dependencyList.valid[engineIdx])
        {
            assert(dependencyList.signalIdx[engineIdx] < m_signalsCtx[engineIdx].size());
            const auto &waitSignalCtx = m_signalsCtx[engineIdx][dependencyList.signalIdx[engineIdx]];

            // check if the reduced list could be further reduced
            for (unsigned reducedIdx = 0; reducedIdx < c_engines_nr; reducedIdx++)
            {
                if ((reducedIdx != engineIdx) &&
                    (dependencyList.valid[reducedIdx]) &&
                    (waitSignalCtx.valid[reducedIdx]) &&
                    (dependencyList.signalIdx[reducedIdx] <= waitSignalCtx.signalIdx[reducedIdx]))
                {
                    dependencyList.valid[reducedIdx] = false;
                }
            }

            // check if the engine already waited for this signal
            if ((!engineCtx.valid[engineIdx]) ||
                (engineCtx.signalIdx[engineIdx] < dependencyList.signalIdx[engineIdx]))
            {
                // update the engine context according to the new signal context
                for (unsigned engine = 0; engine < c_engines_nr; engine++)
                {
                    if ((waitSignalCtx.valid[engine]) &&
                        (!engineCtx.valid[engine] ||
                        (engineCtx.signalIdx[engine] < waitSignalCtx.signalIdx[engine])))
                    {
                        engineCtx.valid[engine] = true;
                        engineCtx.signalIdx[engine] = waitSignalCtx.signalIdx[engine];
                    }
                }
            }
        }
    }
}

template<unsigned N>
void Overlap<N>::updateSyncObjectsCtx(const OverlapDescriptor &desc)
{
    DependencyCtx newSigCtx = m_engineSyncCtx[desc.engineIDForDepCtx];
    for (unsigned i = 0; i < desc.numSignals; i++)
    {
        newSigCtx.valid[desc.engineID] = true;
        newSigCtx.signalIdx[desc.engineID] = m_signalCtrs[desc.engineID] + i;
        m_signalsCtx[desc.engineID].push_back(newSigCtx);
    }
}

template<unsigned N>
void Overlap<N>::addDescriptor(
    const OverlapDescriptor& desc, // input descriptor (must be added in the submission order)
    DependencyCtx& dependency,     // input/output reduced dependency list
    uint32_t engineMaxSignalIdx)   // max signal index (non-inclusive) to consider in self dependency prior Shared SOB consideration
{
    // add the descriptor to the maps and get the reduced dependency list.
    // handle input and output rois and update the segments maps
    handleOutputRois(desc, dependency, engineMaxSignalIdx);
    handleInputRois(desc, dependency, engineMaxSignalIdx);

    addSelfDependencyForSharedSob(desc, dependency);

    // reduce the dependency list and update the engines sync contexts
    updateEngineCtxAndReduceWaitList(desc, dependency);

    // update the signal contexts with the descriptors
    updateSyncObjectsCtx(desc);

    // update the signal counters
    m_signalCtrs[desc.engineID] += desc.numSignals;
}

#endif //MME__OVERLAP_INL
