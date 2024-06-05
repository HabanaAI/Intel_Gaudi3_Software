#pragma once

#include "scal_stream_copy_interface.hpp"
#include "global_statistics.hpp"
#include "stream_cyclic_buffer_base.hpp"
#include "runtime/scal/common/entities/scal_monitor.hpp"
#include "scal_internal/pkt_macros.hpp"

class ScalStreamBase : virtual public ScalStreamCopyInterface
{
public:
    friend class ScalStreamTest;
    friend class SynScalLaunchDummyRecipe;

    ScalStreamBase(const ScalStreamCtorInfoBase* pScalStreamCtorInfo);

    virtual ~ScalStreamBase();

    const std::string getName() const override { return m_name; };

    void longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const override;

    synStatus eventRecord(bool isUserReq, ScalEvent& scalEvent) const override;

    synStatus longSoQuery(const ScalLongSyncObject& rLongSo, bool alwaysWaitForInterrupt = false) const override;

    // Wait on host for last longSo (e.g. see isUserReq + longSo @ addDispatchBarrier) to complete on the device
    synStatus longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec, const char* caller) const override;

    synStatus longSoWaitOnDevice(const ScalLongSyncObject& rLongSo, bool isUserReq) override;

    ScalLongSyncObject getIncrementedLongSo(bool isUserReq, uint64_t targetOffset = 1) override;

    ScalLongSyncObject getTargetLongSo(uint64_t targetOffset) const override;

    // Wait on host for longSo (e.g. see longSo @ addDispatchBarrier) to complete on the device
    synStatus
    longSoWait(const ScalLongSyncObject& rLongSo, uint64_t timeoutMicroSec, const char* caller) const override;

    synStatus addStreamFenceWait(uint32_t target, bool isUserReq, bool isInternalComputeSync) override;

    synStatus getStreamInfo(std::string& info, uint64_t& devLongSo) override;
    /**
     * If last command in the stream is 'wait', then the next event record will need to add a barrier, and it requires a
     * lock.
     * @return true if the last command added to the stream is 'wait'
     */
    bool prevCmdIsWait() override { return m_prevCmdIsWait; }

    void dfaDumpScalStream() override;

    // Dispatch to the Completion-Group
    TdrRtn tdr(TdrType tdrType) override;

    virtual void printCgTdrInfo(bool tdr) const override;

    virtual bool getStaticMonitorPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData) const override;

    ResourceStreamType getResourceType() const override { return m_resourceType; };

    bool isComputeStream() override { return m_resourceType == ResourceStreamType::COMPUTE; };

    synStatus init();

    uint64_t getStreamCyclicBufferOccupancyWatermark() { return getStreamCyclicBuffer()->getStreamCyclicBufferOccupancyWatermark(); };

protected:
    synStatus releaseResources();

    inline const scal_handle_t getDeviceHandle() const { return m_devHndl; };
    inline uint32_t            getStreamIndex() const  { return m_streamIdx; };

    virtual synStatus initInternal() = 0;
    virtual ScalMonitorBase* getMonitor() = 0;
    virtual const ScalMonitorBase* getMonitor() const = 0;
    virtual std::string getSchedulerInfo(unsigned& schedulerIdx) const = 0;
    virtual synStatus addFenceWait(uint32_t target, FenceIdType fenceId, bool send, bool isGlobal) = 0;
    virtual synStatus addEmptyPdmaPacket() = 0;
    virtual StreamCyclicBufferBase* getStreamCyclicBuffer() = 0;

    synStatus initStaticMonitor();

    bool retrievStreamHandle();
    bool allocateAndSetCommandsBuffer();
    bool retrievCommandsBufferInfo();

    synStatus addEmptyPdma(ScalLongSyncObject& rLongSo);

    uint64_t getCompletionTarget() const { return m_pScalCompletionGroup->getCompletionTarget(); }

    template<class Pkt, typename... Args>
    synStatus addPacket(bool send, Args&&... args);

    template<class Pkt, typename... Args>
    synStatus addPacketCommon(bool send, size_t cmdSize, Args&&... args);

    template<class TPacketBuildFunc>
    synStatus addCmd(uint32_t cmdSize, bool send, TPacketBuildFunc packetBuildFunc);

    uint8_t* getCommandBufferHostAddress() { return static_cast<uint8_t*>(m_cmdBufferInfo.host_address); };

    synStatus doneChunkOfCommands(bool isUserReq, ScalLongSyncObject& rLongSo);

    inline bool checkSubmissionQueueAlignment(uint16_t alignment)
    {
        return ((uint64_t)m_cmdBufferInfo.host_address % alignment) == 0;
    };

    uint32_t getFenceId() const { return m_fenceId; }

    uint32_t getFenceIdForCompute() const { return m_fenceIdForCompute; }

    scal_stream_handle_t m_streamHndl;

    uint16_t m_cmdAlign;
    uint16_t m_submitAlign;

    uint64_t m_consecutiveWaitCommands;

    const globalStatPointsEnum m_submitStatPoint;

    // will be true if the last command submitted to this stream is longSoWaitOnDevice
    std::atomic<bool> m_prevCmdIsWait = false;
    bool              m_isFirstJobInChunk;

    ScalCompletionGroupBase* m_pScalCompletionGroup;

    const std::string m_name;

    std::variant<G2Packets, G3Packets> m_gxPackets;
    const DevStreamInfo& m_devStreamInfo;
    uint32_t m_workCompletionValue;

private:
    const scal_handle_t  m_devHndl;
    const uint32_t       m_streamIdx;
    ScalMemoryPool&      m_mpHostShared;

    // Buffer-handle (and its buffer-info) which is used for the Submission-Queue
    scal_buffer_handle_t m_ctrlBuffHndl;
    scal_buffer_info_t   m_cmdBufferInfo;

    FenceIdType m_fenceId;
    FenceIdType m_fenceIdForCompute;

    ResourceStreamType m_resourceType;
};

class ScalStreamCopyBase : public ScalStreamBase
{
public:
    ScalStreamCopyBase(const ScalStreamCtorInfoBase* pScalStreamCtorInfo) : ScalStreamBase(pScalStreamCtorInfo) {}

protected:
    static enum PdmaDirCtx getDir(ResourceStreamType resourceType);
    static internalStreamType getInternalStreamType(ResourceStreamType resourceType);
    static uint8_t            getContextId(enum PdmaDirCtx dir, ResourceStreamType resourceType, uint32_t index);
};

/*
 ***************************************************************************************************
 *   @brief addPacket(), addPacketCommon() add a packet to the cyclic buffer
 *
 *   @param  send - do submit after adding the packet
 *   @param  args - arguments needed for building the packet
 *
 ***************************************************************************************************
 */

template<class Pkt, typename... Args>
synStatus ScalStreamBase::addPacket(bool send, Args&&... args)
{
    const size_t cmdSize = Pkt::getSize();

    return addPacketCommon<Pkt>(send, cmdSize, std::forward<Args>(args)...);
}

template<class Pkt, typename... Args>
synStatus ScalStreamBase::addPacketCommon(bool send, size_t cmdSize, Args&&... args)
{
    const synStatus status =
        addCmd(cmdSize, send, [&](uint8_t* buffer) { Pkt::build(buffer, std::forward<Args>(args)...); });

    return status;
}

/*
 ***************************************************************************************************
 *   @brief addCmd() - add a command to the cyclic buffer
 *                     a command shouldn't cross a chunk boundary, add nop if needed
 *                     verify we don't step over the cyclic buffer that wasn't executed yet
 *
 *   @param  cmd poitner, size
 *   @param  send - if true, we also do submit (update the pi in scal)
 *   @return status
 *
 ***************************************************************************************************
 */
template<class TPacketBuildFunc>
synStatus ScalStreamBase::addCmd(uint32_t cmdSize, bool send, TPacketBuildFunc packetBuildFunc)
{
    std::vector<CommandSubmissionData> commandSubmissionDataList(2, {0, 0, nullptr, false});
    synStatus status = getStreamCyclicBuffer()->addCommand(cmdSize, send, packetBuildFunc, commandSubmissionDataList);
    if (status != synSuccess)
    {
        return status;
    }

    for (auto& commandSubmissionDataElement : commandSubmissionDataList)
    {
        if (commandSubmissionDataElement.valid)
        {
            STAT_GLBL_START(scalSubmit);
            // for Gaudi3 auto fetcher mode we need the unwrapped pi counter ccb m_pi
            // Gaudi2 scal code will wrap it if needed
            ScalRtn rc = scal_stream_submit(m_streamHndl, commandSubmissionDataElement.pi, m_submitAlign);
            STAT_GLBL_COLLECT_TIME(scalSubmit, m_submitStatPoint);
            if (rc != SCAL_SUCCESS)
            {
                LOG_CRITICAL(SYN_STREAM, "stream {}: {} failed to submit rc {}",
                                  m_name, commandSubmissionDataElement.desc, rc);
                return synFailedToSubmitWorkload;
            }
        }
    }

    return synSuccess;
}