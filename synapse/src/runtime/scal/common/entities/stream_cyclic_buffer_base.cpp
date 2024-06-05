#include "stream_cyclic_buffer_base.hpp"

#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"

StreamCyclicBufferBase::StreamCyclicBufferBase(std::string streamName)
: m_streamName(streamName),
  m_streamHndl(0),
  m_offsetInBuffer(0),
  m_prevOffsetInBuffer(0),
  m_pi(0),
  m_cmdAlign(0),
  m_isFirstJobInChunk(false),
  m_isInitialized(false)
{
    m_barrierToCompeltionHandle.resize(GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value());
    fill(m_barrierToCompeltionHandle.begin(), m_barrierToCompeltionHandle.end(), LongSoEmpty);
    m_debugSendEachPacket = (GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value() & SEND_EACH_PACKET) == SEND_EACH_PACKET;
}

/*
 ***************************************************************************************************
 *   @brief incPi() increment pi, wrap around if needed
 *
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
void StreamCyclicBufferBase::incPi(size_t size)
{
    m_offsetInBuffer += size;
    m_pi             += size;  // does not wrap

    uint32_t ccbSize = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
    if (m_offsetInBuffer >= ccbSize)
    {
        m_offsetInBuffer = 0;
    }
}

/*
 ***************************************************************************************************
 *   @brief doneChunkOfCommands() add a barrier or empty PDMA in the beginning of each CCB chunk
 *
 *   @param  longSo
 *   @return None
 *
 ***************************************************************************************************
 */
void StreamCyclicBufferBase::doneChunkOfCommands(ScalLongSyncObject& rLongSo)
{
    if (m_isFirstJobInChunk)
    {
        putSchedulerBarrier(rLongSo);
        m_isFirstJobInChunk = false;
    }
}

void StreamCyclicBufferBase::sampleCcbOccupancy()
{
    uint32_t ccbChunkSize      = getCcbChunkSize();
    uint32_t currentBarrierIdx = m_offsetInBuffer / ccbChunkSize;

    scal_completion_group_infoV2_t cgInfo {};
    const_cast<ScalCompletionGroupBase*>(m_pScalCompletionGroup)->getCurrentCgInfo(cgInfo);

    uint64_t currentCcbOccupancy = 0;
    uint64_t completedTarget     = cgInfo.current_value;
    uint32_t barrierIdx          = currentBarrierIdx;

    do
    {
        if (m_barrierToCompeltionHandle[barrierIdx].m_targetValue > completedTarget)
        {
            currentCcbOccupancy++;
        }
        barrierIdx = (barrierIdx + 1) % GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    }
    while (barrierIdx != currentBarrierIdx);

    currentCcbOccupancy *= ccbChunkSize;
    if (m_ccbOccupancyWatermark < currentCcbOccupancy)
    {
        m_ccbOccupancyWatermark = currentCcbOccupancy;
    }
}

void StreamCyclicBufferBase::putSchedulerBarrier(const ScalLongSyncObject& rLongSo)
{
    // we assume that all the scheduler commands between the alloc and dispatch barrier
    // will fit into single barrier chunk (16k when the split ratio is 4),
    // resulting in that the currentBarrierIdx is the same
    // for all the added commands in the same job
    uint32_t currentBarrierIdx = getBarrierIndex();

    LOG_TRACE(SYN_STREAM,
                   "PutSchedulerBarrier for {} in completionTarget 0x{:x} pi {:x}",
                   currentBarrierIdx,
                   rLongSo.m_targetValue,
                   m_pi);

    m_barrierToCompeltionHandle[currentBarrierIdx] = rLongSo;

    if (GCFG_ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK.value() == true)
    {
        sampleCcbOccupancy();
    }
}

const std::string StreamCyclicBufferBase::dfaInfo() const
{
    const std::string dfaInfo = fmt::format("bufOffset: 0x{:x}, PI: 0x{:x}",
                                            m_offsetInBuffer,
                                            m_pi);
    return dfaInfo;
}

/*
 ***************************************************************************************************
 *   @brief init() init the CCB. If fails, release all resources.
 *
 *   @param  ScalCompletionGroup
 *   @return status
 *
 ***************************************************************************************************
 */
void StreamCyclicBufferBase::init(ScalCompletionGroupBase* pScalCompletionGroup,
                                  uint8_t*                 cyclicBufferBaseAddress,
                                  uint64_t                 streamHndl,
                                  uint16_t                 cmdAlign)
{
    m_pScalCompletionGroup    = pScalCompletionGroup;
    m_cyclicBufferBaseAddress = cyclicBufferBaseAddress;
    m_streamHndl              = streamHndl;
    m_cmdAlign                = cmdAlign;

    m_isInitialized = true;
}

/*
 ***************************************************************************************************
 *   @brief preSubmit() -
 *      Dumps new packets (from prev-PI until current-PI)
 *      Incerements PI
 *      Prepares CS-Data information
 *
 *   @param  description (for logging only)
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus StreamCyclicBufferBase::preSubmit(CommandSubmissionData& csDataInfo,
                                            const char*            description)
{
    dumpSubmission(description);
    m_prevOffsetInBuffer = m_offsetInBuffer;

    csDataInfo.pi           = m_pi;
    csDataInfo.offsetInBuff = m_offsetInBuffer;
    csDataInfo.desc         = description;
    csDataInfo.valid        = true;

    return synSuccess;
}

void StreamCyclicBufferBase::logAddCmd(uint32_t cmdSize, bool send)
{
    LOG_TRACE(SYN_STREAM,
                   "Device stream {}: add cmd with size {:x} in offsetInbuffer {:x}. align {:x} send {}",
                   m_streamName,
                   cmdSize,
                   m_offsetInBuffer,
                   m_cmdAlign,
                   send);
}

uint64_t StreamCyclicBufferBase::getCommandsAlignmentSize(uint32_t cmdSize)
{
    const uint64_t startByte = m_offsetInBuffer;
    const uint64_t endByte   = startByte + cmdSize - 1;

    const uint64_t startChunk = startByte / m_cmdAlign;
    const uint64_t endChunk   = endByte / m_cmdAlign;

    return (startChunk == endChunk) ? 0 : (m_cmdAlign - (startByte % m_cmdAlign));
}

synStatus StreamCyclicBufferBase::handleCommandsAlignment(CommandSubmissionData& csDataAlignmentInfo,
                                                          uint32_t               alignmentSize)
{
    if (alignmentSize == 0)
    {
        return synSuccess;
    }

    addAlignmentPackets(alignmentSize);
    incPi(alignmentSize);

    return preSubmit(csDataAlignmentInfo, "align");
}

synStatus StreamCyclicBufferBase::handleCommandsBarrier()
{
    uint32_t ccbChunkSize = getCcbChunkSize();
    if (m_offsetInBuffer % ccbChunkSize == 0)
    {
        const uint32_t currentBarrierIdx = m_offsetInBuffer / ccbChunkSize;

        // waiting for the next next chunk long-so to avoid corner case of ci=pi
        // that caused FW to stuck if read part of 256 bytes.
        uint32_t                  waitIdx = 0;
        const ScalLongSyncObject& rlongSo = getBarrierToCompletionHandle(waitIdx, currentBarrierIdx);

        if (rlongSo != LongSoEmpty)
        {
            // wait on the next chunk barrier
            LOG_DEBUG(SYN_STREAM,
                      "stream {} waiting on the next chunk barrier to start on {}. wait for {} longSo (index {}, value "
                      "{:#x})",
                      m_streamName,
                      currentBarrierIdx,
                      waitIdx,
                      rlongSo.m_index,
                      rlongSo.m_targetValue);
            synStatus status = m_pScalCompletionGroup->longSoWait(rlongSo, 0);
            if (status == synBusy)
            {
                if (GCFG_ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK.value() == true)
                {
                    m_ccbOccupancyWatermark = ccbChunkSize * (GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value() - 2);
                }

                STAT_GLBL_START(ccbBusyTime);
                status = m_pScalCompletionGroup->longSoWait(rlongSo, (int64_t)SCAL_FOREVER);
                STAT_GLBL_COLLECT_TIME(ccbBusyTime, globalStatPointsEnum::ccbBusyTime);

                LOG_DEBUG(SYN_STREAM,
                          "stream {} finished waiting on the next chunk barrier. longSo (index {}, value {:#x})",
                          m_streamName,
                          rlongSo.m_index,
                          rlongSo.m_targetValue);

                if (status != synSuccess)
                {
                    LOG_ERR(SYN_STREAM,
                            "stream {} wait on the next chunk barrier, return with error {}. longSo (index {}, value "
                            "{:#x})",
                            m_streamName,
                            status,
                            rlongSo.m_index,
                            rlongSo.m_targetValue);

                    return status;
                }
            }
        }

        m_isFirstJobInChunk = true;
    }

    return synSuccess;
}

const ScalLongSyncObject& StreamCyclicBufferBase::getBarrierToCompletionHandle(uint32_t waitIdx,
                                                                               uint32_t currentBarrierIdx)
{
    waitIdx = (currentBarrierIdx + 2) % GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
    return m_barrierToCompeltionHandle[waitIdx];
}

bool StreamCyclicBufferBase::testOnlyCheckCcbConsistency()
{
    uint32_t currentBarrierIdx = getBarrierIndex();
    if (currentBarrierIdx >= m_barrierToCompeltionHandle.size())
    {
        LOG_ERR(SYN_RT_TEST,
                "Invalid barrier index {} DB-size {}",
                currentBarrierIdx, m_barrierToCompeltionHandle.size());
        return false;
    }

    if (m_barrierToCompeltionHandle[currentBarrierIdx + 1].m_targetValue != 0)
    {
        for (uint32_t index = currentBarrierIdx + 1; index < m_barrierToCompeltionHandle.size() - 1; index++)
        {
            if (m_barrierToCompeltionHandle[index].m_targetValue >=
                m_barrierToCompeltionHandle[index + 1].m_targetValue)
            {
                return false;
            }
        }
    }

    for (uint32_t index = 0; index < currentBarrierIdx; index++)
    {
        if (m_barrierToCompeltionHandle[index].m_targetValue >= m_barrierToCompeltionHandle[index + 1].m_targetValue)
        {
            return false;
        }
    }

    return true;
}

uint64_t StreamCyclicBufferBase::getCcbChunkSize()
{
    return GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024 / GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT.value();
}