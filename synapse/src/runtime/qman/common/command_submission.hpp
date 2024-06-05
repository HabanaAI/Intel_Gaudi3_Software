#pragma once

#include "synapse_common_types.h"
#include "recipe_package_types.hpp"
#include "defs.h"

#include <deque>
#include <memory>
#include <mutex>
#include "global_statistics.hpp"
// TODO - Refactor this class to use only this struct
struct PrimeQueueEntry
{
    uint32_t queueIndex;
    uint32_t size;
    uint64_t address;
};
typedef std::deque<PrimeQueueEntry>       PrimeQueueEntries;
typedef PrimeQueueEntries::const_iterator PrimeQueueEntriesConstIterator;

enum ePrimeQueueEntryType
{
    PQ_ENTRY_TYPE_INTERNAL_EXECUTION,
    PQ_ENTRY_TYPE_EXTERNAL_EXECUTION
};

struct OffsetAndSize
{
    OffsetAndSize();
    uint64_t offset;
    uint32_t size;
};

struct StagedInfo
{
    StagedInfo();
    void addDataChunkInfo(unsigned dcIndex, uint64_t offset, uint32_t size);

    std::vector<OffsetAndSize> offsetSizeInDc;

    bool isFirstSubmission;
    bool isLastSubmission;
    bool hasWork;
};

class CommandSubmission
{
public:
    CommandSubmission(bool isExternalCbRequired = true);
    ~CommandSubmission();

    void addPrimeQueueEntry(ePrimeQueueEntryType primeQueueType, uint32_t queueIndex, uint32_t size, uint64_t address);

    void clearPrimeQueueEntries(ePrimeQueueEntryType primeQueueType);

    void copyPrimeQueueEntries(ePrimeQueueEntryType primeQueueType, PrimeQueueEntries& primeQueueEnries);

    void getPrimeQueueEntries(ePrimeQueueEntryType primeQueueType, const PrimeQueueEntries*& pPrimeQueueEnries) const;

    uint32_t getPrimeQueueEntriesAmount(ePrimeQueueEntryType primeQueueType) const;

    synCommandBuffer* getExecuteExternalQueueCb() const;

    const synInternalQueue* getExecuteInternalQueueCb() const;

    void setExecuteExternalQueueCb(synCommandBuffer* extQueueCb);

    void copyExtQueueCb(synCommandBuffer* extQueueCb, uint32_t numOfCb);

    void setExecuteInternalQueueCb(synInternalQueue* intQueueCb);

    bool setExecuteIntQueueCbIndex(uint32_t index, uint32_t queueIndex, uint32_t size, uint64_t address);

    void copyIntQueueCb(synInternalQueue* intQueueCb, uint32_t numOfInternal);

    uint32_t getNumExecuteInternalQueue() const;

    void setNumExecuteExternalQueue(uint32_t numExtQueue);

    uint32_t getNumExecuteExternalQueue() const;

    void setNumExecuteInternalQueue(uint32_t numIntQueue);

    void        setCalledFrom(const char* msg);

    const char* getCalledFrom();

    void setFirstStageCSHandle(uint64_t csHandle) { m_firstStageCSHandle = csHandle; }

    synStatus prepareForSubmission(void*&            pExecuteChunkArgs,
                                   uint32_t&         executeChunksAmount,
                                   bool              isRequireExternalChunk,
                                   uint32_t          queueOffset,
                                   const StagedInfo* pStagedInfo);

    synStatus submitCommandBuffers(uint64_t*            csHandle,
                                   uint64_t*            mappedBuff,
                                   uint32_t             queueOffset,
                                   const StagedInfo*    pStagedInfo,
                                   globalStatPointsEnum point);

    void setEncapsHandleId(uint64_t encapsHandleId) { m_encapsHandleId = encapsHandleId; }

    std::mutex& getMutex();

    void dump() const;

private:
    CommandSubmission& operator=(const CommandSubmission& curr) = delete;

    PrimeQueueEntries& _getPqEntries(ePrimeQueueEntryType primeQueueType);

    const PrimeQueueEntries& _getPqEntries(ePrimeQueueEntryType primeQueueType) const;

    synStatus _createExecuteChunks(void*&            pExecuteChunkArgs,
                                   uint32_t&         executeChunksAmount,
                                   bool              isRequireExternalChunk,
                                   const StagedInfo* pStagedInfo,
                                   uint32_t          queueOffset);

    PrimeQueueEntries m_executeInternalPqEntries;
    PrimeQueueEntries m_executeExternalPqEntries;  // for staged

    synCommandBuffer* m_executeExternalQueueCb;
    synInternalQueue* m_executeInternalQueueCb;  // for staged

    uint32_t m_numExecuteExternalCbs;
    uint32_t m_numExecuteInternalCbs;  // For staged

    uint64_t m_firstStageCSHandle;

    const char* m_calledFrom;  // for debug, shows who is trying to submit
    uint64_t    m_encapsHandleId;

    bool m_isExternalCbRequired;

    // Mutexes
    std::mutex m_commandSubmissionManagerMutex;
};
