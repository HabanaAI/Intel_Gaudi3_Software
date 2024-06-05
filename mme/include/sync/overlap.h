#ifndef MME__OVERLAP_H
#define MME__OVERLAP_H

#include <list>
#include <memory>
#include "include/sync/data_range.h"
#include "include/sync/segments_space.h"

struct OverlapSubRoi
{
    std::vector<DataRange<uint64_t>> ranges;
    // cyclic ranges
    std::vector<CyclicDataRange> cyclicRanges;
    unsigned relSoIdx;
    OverlapSubRoi() = default;
    OverlapSubRoi(const OverlapSubRoi& other)
    : ranges(other.ranges), cyclicRanges(other.cyclicRanges), relSoIdx(other.relSoIdx){}

    bool operator==(const OverlapSubRoi& other) const
    {
        return (relSoIdx == other.relSoIdx && ranges == other.ranges && cyclicRanges == other.cyclicRanges);
    }
};
using SubRoiVecPtr = std::shared_ptr<std::vector<OverlapSubRoi>>;

struct OverlapRoi
{
    OverlapRoi()
        : subRois(std::make_shared<std::vector<OverlapSubRoi>>())
        , isSram(false)
        , isL0(false)
        , isReduction(false)
        , isLocalSignal(false)
        , offset(0)

    {}

    OverlapRoi(const OverlapRoi& other)
    : subRois(std::make_shared<std::vector<OverlapSubRoi>>(*other.subRois)),
      isSram(other.isSram),
      isL0(other.isL0),
      isReduction(other.isReduction),
      isLocalSignal(other.isLocalSignal),
      offset(other.offset)
    {}

    OverlapRoi& operator = (const OverlapRoi &other)
    {
        subRois = other.subRois;
        isSram = other.isSram;
        isL0 = other.isL0;
        isReduction = other.isReduction;
        isLocalSignal = other.isLocalSignal;
        offset = other.offset;
        return *this;
    }
    bool operator==(const OverlapRoi& other) const
    {
        return (isL0 == other.isL0 &&
         isLocalSignal == other.isLocalSignal
         && isReduction == other.isReduction
         && isSram == other.isSram
         && offset == other.offset
         && (*subRois == (*other.subRois)));
    }

    std::shared_ptr<std::vector<OverlapSubRoi>> subRois;
    bool isSram;
    bool isL0 ;
    bool isReduction;
    bool isLocalSignal;
    uint64_t offset;
};

struct OverlapDescriptor
{
    std::list<OverlapRoi> inputRois;
    std::list<OverlapRoi> outputRois;
    unsigned numSignals = 0;
    unsigned numLocalSignals = 0;
    unsigned engineID = 0xFFFFFFFF;
    unsigned engineIDForDepCtx = 0xFFFFFFFF;  // allow using different engine ID for the engine's dependency context
    unsigned minSelfWaitForSharedSob = 0;  // minimum self dependency for Shared SOB (0 means Shared SOB is disabled)
};

// Template param is number of logical engines
template<unsigned N>
class Overlap
{
public:
    static const unsigned c_engines_nr = N;

    struct DependencyCtx
    {
        unsigned signalIdx[c_engines_nr] = {0}; // the index of the engine's signal. (zero based)
        unsigned valid[c_engines_nr] = {0};
    };

    enum AccessType
    {
        WRITE = 0,
        READ,
        RMW
    };

    struct CyclicRangeAccess
    {
        unsigned engine;
        unsigned signal;
        CyclicDataRange cyclicParams {0, 0, 1};
        AccessType accessType;
    };

    Overlap();

    void addDescriptor(const OverlapDescriptor& desc,     // input descriptor (must be added in the submission order)
                       DependencyCtx& dependency,         // input/output reduced dependency list
                       uint32_t engineMaxSignalIdx = -1); // max signal index (non-inclusive) to consider in self
                                                          // dependency prior Shared SOB consideration

    inline const DependencyCtx& getSignalCtx(unsigned engineID, unsigned signalIdx) const
    {
        assert(engineID < c_engines_nr);
        return m_signalsCtx[engineID].at(signalIdx);
    }

private:
    struct SyncInfo
    {
        SyncInfo()
        : producerValid(0),
          producerSignalIdx(0),
          producerEngine(0),
          isReduction(0),
          reductionDependency({0}),
          consumersDependency({0}) {};
        bool producerValid;
        unsigned producerSignalIdx;
        unsigned producerEngine;
        bool isReduction;
        DependencyCtx reductionDependency;
        DependencyCtx consumersDependency;
        std::vector<CyclicRangeAccess> cyclicDependency;
    };

    void handleOutputRois(const OverlapDescriptor& desc, DependencyCtx& dependencyList, uint32_t engineMaxSignalIdx);

    void handleInputRois(const OverlapDescriptor& desc, DependencyCtx& dependencyList, uint32_t engineMaxSignalIdx);

    void handleLinearWrite(DependencyCtx& dependencyList,
                           std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
                           unsigned engineID,
                           unsigned signalIdx,
                           uint32_t engineMaxSignalIdx,
                           bool isReductionWrite);

    void handleLinearRead(DependencyCtx& dependencyList,
                          std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
                          unsigned engineID,
                          unsigned signalIdx,
                          uint32_t engineMaxSignalIdx);

    void handleCyclicAccess(DependencyCtx& dependencyList,
                            std::tuple<uint64_t, uint64_t, SyncInfo*>& rangeInfo,
                            const CyclicRangeAccess& cyclicAccess,
                            uint32_t engineMaxSignalIdx);

    void getWriteDependencies(DependencyCtx& dependencyList,
                              const SyncInfo* rangeSyncInfo,
                              unsigned currentEngineID,
                              uint32_t engineMaxSignalIdx,
                              unsigned currentSignalIdx,
                              bool isReadAccess);

    void getReadDependencies(DependencyCtx& dependencyList,
                             const SyncInfo* rangeSyncInfo,
                             unsigned currentEngineID,
                             uint32_t engineMaxSignalIdx);

    void getRMWDependencies(DependencyCtx& dependencyList,
                            const SyncInfo* rangeSyncInfo,
                            unsigned currentEngineID,
                            uint32_t engineMaxSignalIdx,
                            bool isReadAccess);

    void getCyclicRangeDependencies(DependencyCtx& dependencyList,
                                    const CyclicRangeAccess& cyclicAccess,
                                    unsigned currentEngineID,
                                    uint32_t engineMaxSignalIdx,
                                    unsigned currentSignalIdx,
                                    AccessType currentAccess);

    void addDependency(DependencyCtx& dependencyList, unsigned engine, unsigned signal);

    void updateEngineCtxAndReduceWaitList(
        const OverlapDescriptor &desc,
        DependencyCtx &dependencyList);

    void addSelfDependencyForSharedSob(const OverlapDescriptor& desc, DependencyCtx& dependencyList) const;

    void updateSyncObjectsCtx(const OverlapDescriptor &desc);

    SegmentsSpace<SyncInfo> m_sram;
    SegmentsSpace<SyncInfo> m_vmem;

    unsigned m_signalCtrs[c_engines_nr] = {0};
    DependencyCtx m_engineSyncCtx[c_engines_nr] = {0};
    std::vector<DependencyCtx> m_signalsCtx[c_engines_nr];
};

#include "overlap.inl"

#endif //MME__OVERLAP_H
