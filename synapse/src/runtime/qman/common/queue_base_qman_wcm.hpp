#pragma once

#include "runtime/common/device/dfa_observer.hpp"
#include "queue_base_qman.hpp"
#include "runtime/qman/common/wcm/wcm_observer_interface.hpp"

#include <condition_variable>
#include <mutex>
#include <dfa_defines.hpp>

class WorkCompletionManagerInterface;
class AddressRangeMapper;

class QueueBaseQmanWcm
: public QueueBaseQman
, public WcmObserverInterface
{
public:
    QueueBaseQmanWcm(const BasicQueueInfo&           rBasicQueueInfo,
                     uint32_t                        physicalQueueOffset,
                     synDeviceType                   deviceType,
                     PhysicalQueuesManagerInterface* pPhysicalStreamsManager,
                     WorkCompletionManagerInterface& rWorkCompletionManager);

    virtual ~QueueBaseQmanWcm() = default;

    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) override;

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override;

    virtual void finalize() override;

    virtual bool parseCsDc(uint64_t suspectedCsHandle);
    void         dfaLogCsDescription(const uint64_t csSeq, DfaReq dfaReq);
    virtual void dfaInfo(DfaReq dfaReq, uint64_t csSeq) override;

    uint64_t getCsDcDataBaseSize() const;

protected:
    // Todo remove the unused method updateDataChunkCacheSize and the relevant GCFGs
    static void updateDataChunkCacheSize(uint64_t& maximalCacheAmount,
                                         uint64_t& minimalCacheAmount,
                                         uint64_t& maximalFreeCacheAmount,
                                         bool      ignoreInitialCacheSizeConfiguration = false);

    virtual bool isRecipeHasInflightCsdc(InternalRecipeHandle* pRecipeHandle) = 0;

    void _waitForWCM(bool queryAllCsdcs, InternalRecipeHandle* pRecipeHandle = nullptr);

    void _wcmReleaseThreadIfNeeded();

    bool _parseSingleCommandSubmission(CommandSubmissionDataChunks* pCsDc) const;

    void _addCsdcToDb(CommandSubmissionDataChunks* pCsDataChunks);

    bool _addDataChunksMapping(AddressRangeMapper& addressRangeMap, CommandSubmissionDataChunks& csDc) const;

    // Static mappings are those which are alocated by the StreamDcDownloader,
    // are common to all CS-DCs of that stream, and are for the content of Gaudi-1 Compute-Stream's exteranl HW-Stream
    virtual bool _addStaticMapping(AddressRangeMapper& addressRangeMap) const { return true; };

    WorkCompletionManagerInterface& m_rWorkCompletionManager;

    bool                            m_cvFlag;
    mutable std::mutex              m_condMutex;
    mutable std::condition_variable m_condVar;

    bool                  m_waitForAllCsdcs;
    InternalRecipeHandle* m_waitForInternalRecipeHandle;

    std::deque<CommandSubmissionDataChunks*> m_csDataChunksDb;

    mutable std::mutex m_mutex;

    mutable std::mutex m_DBMutex;

private:
    virtual void _dfaLogCsDcInfo(CommandSubmissionDataChunks* csPtr, int logLevel, bool errorCsOnly) = 0;
};
