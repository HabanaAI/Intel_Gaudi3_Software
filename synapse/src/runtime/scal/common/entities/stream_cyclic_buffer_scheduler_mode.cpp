#include "stream_cyclic_buffer_scheduler_mode.hpp"
#include "scal_internal/pkt_macros.hpp"

#include "habana_global_conf_runtime.h"

void StreamCyclicBufferSchedulerMode::init(ScalCompletionGroupBase*           pScalCompletionGroup,
                                           uint8_t*                           cyclicBufferBaseAddress,
                                           uint64_t                           streamHndl,
                                           uint16_t                           cmdAlign,
                                           uint16_t                           submitAlign,
                                           std::variant<G2Packets,G3Packets>& gxPackets)
{
    StreamCyclicBufferBase::init(pScalCompletionGroup,
                                 cyclicBufferBaseAddress,
                                 streamHndl,
                                 cmdAlign);

    m_submitAlign = submitAlign;
    m_gxPackets   = gxPackets;
}

void StreamCyclicBufferSchedulerMode::addAlignmentPackets(uint64_t alignSize)
{
    void* piBuff = getBufferOffsetAddr(m_offsetInBuffer);

    std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            NopCmdPkt<T>::build(piBuff, alignSize);
        },
        m_gxPackets);
}

void StreamCyclicBufferSchedulerMode::dumpSubmission(const char* desc)
{
    const int  logLevel = HLLOG_LEVEL_DEBUG;
    if (HLLOG_UNLIKELY(hl_logger::logLevelAtLeast(synapse::LogManager::LogType::SYN_STREAM, logLevel)))
    {
        const auto spLogger = hl_logger::getLogger(synapse::LogManager::LogType::SYN_STREAM);

        if (HLLOG_UNLIKELY(hl_logger::logLevelAtLeast(synapse::LogManager::LogType::SYN_STREAM, logLevel)))
        {
            HLLOG_UNTYPED(spLogger,
                          logLevel,
                          "Device stream {}: sending {} on streamHndl {:x} offsetInBuff {:x} m_pi {:x} m_submitAlign {:x}",
                          m_streamName,
                          desc,
                          m_streamHndl,
                          m_offsetInBuffer,
                          m_pi,
                          m_submitAlign);
        }

        uint32_t ccbSize = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
        while (m_prevOffsetInBuffer != m_offsetInBuffer)
        {
            uint64_t inc = std::visit(
                [&](auto pkts) {
                    using T = decltype(pkts);
                    return dumpPacket<T>(getBufferOffsetAddr(m_prevOffsetInBuffer), spLogger, logLevel);
                },
                m_gxPackets);

            if (inc == 0) break;

            m_prevOffsetInBuffer = (m_prevOffsetInBuffer + inc) % ccbSize;
        }
    }
}