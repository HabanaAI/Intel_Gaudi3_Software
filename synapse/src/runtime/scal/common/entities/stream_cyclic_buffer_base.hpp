#pragma once

#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "dfa_defines.hpp"

#include "scal_completion_group_base.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"
#include "log_manager.h"

#include <string>
#include <variant>

class ScalMemoryPool;

struct CommandSubmissionData
{
    uint64_t    pi;
    uint64_t    offsetInBuff;
    const char* desc;
    bool        valid;
};

class StreamCyclicBufferBase
{
public:
    friend class ScalStreamTest;
    friend class SynScalLaunchDummyRecipe;

    StreamCyclicBufferBase(std::string streamName);

    virtual ~StreamCyclicBufferBase()
    {
        if (GCFG_ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK.value() == true)
        {
            auto currLogLevel = hl_logger::getLoggingLevel(synapse::LogManager::LogType::SYN_STREAM);
            hl_logger::setLoggingLevel(synapse::LogManager::LogType::SYN_STREAM, HLLOG_LEVEL_DEBUG);

            uint32_t ccbSize      = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
            uint32_t ccbChunkSize = ccbSize / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
            LOG_DEBUG(SYN_STREAM,
                    "CCB occupancy watermark of stream {} is: {} bytes out of total CCB size of {} bytes ({} out of {} CCB chunks were occupied)",
                    m_streamName,
                    m_ccbOccupancyWatermark,
                    ccbSize,
                    m_ccbOccupancyWatermark / ccbChunkSize,
                    GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value());

            hl_logger::setLoggingLevel(synapse::LogManager::LogType::SYN_STREAM, currLogLevel);
        }
    }

    void doneChunkOfCommands(ScalLongSyncObject& rLongSo);

    inline bool isFirstJobInChunk() const { return m_isFirstJobInChunk; };

    const std::string dfaInfo() const;

    // Template methods
    template<class TPacketBuildFunc>
    synStatus addCommand(uint32_t                            cmdSize,
                         bool                                send,
                         TPacketBuildFunc&                   packetBuildFunc,
                         std::vector<CommandSubmissionData>& commandSubmissionDataList);

    uint64_t getStreamCyclicBufferOccupancyWatermark() { return m_ccbOccupancyWatermark; };

protected:
    void init(ScalCompletionGroupBase* pScalCompletionGroup,
              uint8_t*                 cyclicBufferBaseAddress,
              uint64_t                 streamHndl,
              uint16_t                 cmdAlign);

    uint8_t* getBufferOffsetAddr(uint64_t pi) { return m_cyclicBufferBaseAddress + pi; };

    // pure virtual methods
    virtual void addAlignmentPackets(uint64_t alignSize) = 0;
    virtual void dumpSubmission(const char* desc) = 0;

    // Members
    const std::string m_streamName;
    uint64_t          m_streamHndl;

    uint64_t m_offsetInBuffer;
    uint64_t m_prevOffsetInBuffer;
    uint64_t m_pi;

private:
    void incPi(size_t size);

    void putSchedulerBarrier(const ScalLongSyncObject& rLongSo);

    synStatus preSubmit(CommandSubmissionData& commandSubmissionData,
                        const char*            description);

    void logAddCmd(uint32_t cmdSize, bool send);

    uint64_t getCommandsAlignmentSize(uint32_t cmdSize);

    synStatus handleCommandsAlignment(CommandSubmissionData& csDataAlignmentInfo,
                                      uint32_t               alignmentSize);

    synStatus handleCommandsBarrier();

    const ScalLongSyncObject& getBarrierToCompletionHandle(uint32_t waitIdx, uint32_t currentBarrierIdx);

    void sampleCcbOccupancy();

    bool testOnlyCheckCcbConsistency();

    uint64_t getCcbChunkSize();
    inline uint32_t getBarrierIndex() { return m_offsetInBuffer / getCcbChunkSize(); };

    // Members
    std::vector<ScalLongSyncObject> m_barrierToCompeltionHandle;

    const ScalCompletionGroupBase* m_pScalCompletionGroup    = nullptr;
    uint8_t*                       m_cyclicBufferBaseAddress = nullptr;

    uint16_t m_cmdAlign;

    bool m_isFirstJobInChunk;
    bool m_debugSendEachPacket;
    bool m_isInitialized;

    uint64_t m_ccbOccupancyWatermark = 0;
};

/*
 ***************************************************************************************************
 *   @brief addCommand() -
 *      Add a command to the cyclic buffer
 *      A command shouldn't cross a chunk boundary, add alignment-packets if needed
 *      Verify it doesn't overrides buffer's parts that had not yet been executed
 *
 *   @param  cmd poitner, size
 *   @param  send - if true, we also do submit (update the pi in scal)
 *   @param  commandSubmissionDataList - the list of command data for scal stream submission
 *   @return status
 *
 ***************************************************************************************************
 */
template<class TPacketBuildFunc>
synStatus StreamCyclicBufferBase::addCommand(
    uint32_t                            cmdSize,
    bool                                send,
    TPacketBuildFunc&                   packetBuildFunc,
    std::vector<CommandSubmissionData>& commandSubmissionDataList)
{
    if (!m_isInitialized)
    {
        LOG_ERR(SYN_STREAM, "{} uninitialized element", HLLOG_FUNC);
        return synFail;
    }

    CommandSubmissionData& commandSubmissionDataAlign = commandSubmissionDataList[0];
    CommandSubmissionData& commandSubmissionDataCmd   = commandSubmissionDataList[1];

    logAddCmd(cmdSize, send);

    uint64_t alignmentSize = getCommandsAlignmentSize(cmdSize);

    synStatus status = handleCommandsAlignment(commandSubmissionDataAlign, alignmentSize);
    if (status != synSuccess)
    {
        return status;
    }

    status = handleCommandsBarrier();
    if (status != synSuccess)
    {
        return status;
    }

    packetBuildFunc(getBufferOffsetAddr(m_offsetInBuffer));
    incPi(cmdSize);

    if (send || ((m_offsetInBuffer % m_cmdAlign) == 0) || m_debugSendEachPacket)
    {
        synStatus status = preSubmit(commandSubmissionDataCmd, "cmd");

        if (status != synSuccess)
        {
            return status;
        }
    }

    return synSuccess;
}