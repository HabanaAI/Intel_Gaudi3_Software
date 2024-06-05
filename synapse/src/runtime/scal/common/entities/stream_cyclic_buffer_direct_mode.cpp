#include "stream_cyclic_buffer_direct_mode.hpp"
#include "runtime/scal/gaudi3/direct_mode_packets/pqm_packets.hpp"

#include "habana_global_conf_runtime.h"

void StreamCyclicBufferDirectMode::init(ScalCompletionGroupBase*           pScalCompletionGroup,
                                        uint8_t*                           cyclicBufferBaseAddress,
                                        uint64_t                           streamHndl,
                                        uint16_t                           cmdAlign)
{
    StreamCyclicBufferBase::init(pScalCompletionGroup,
                                 cyclicBufferBaseAddress,
                                 streamHndl,
                                 cmdAlign);
}

void StreamCyclicBufferDirectMode::addAlignmentPackets(uint64_t alignSize)
{
    static const uint64_t nopSize = pqm::Nop::getSize();

    uint8_t* piBuff = getBufferOffsetAddr(m_offsetInBuffer);

    uint64_t amountOfNopPackets = alignSize / nopSize;

    for (uint64_t i = 0; i < amountOfNopPackets; i++)
    {
        pqm::Nop::build((void*) piBuff);
        piBuff += pqm::Nop::getSize();
    }
}

void StreamCyclicBufferDirectMode::dumpSubmission(const char* desc)
{
    const int  logLevel = HLLOG_LEVEL_DEBUG;

    if (HLLOG_UNLIKELY(hl_logger::logLevelAtLeast(synapse::LogManager::LogType::SYN_DM_STREAM, logLevel)))
    {
        const auto spLogger = hl_logger::getLogger(synapse::LogManager::LogType::SYN_DM_STREAM);

        HLLOG_UNTYPED(spLogger,
                      logLevel,
                      "Device stream {}: sending {} on m_streamHndl {:x} offsetInBuff {:x} m_pi {:x}",
                      m_streamName,
                      desc,
                      m_streamHndl,
                      m_offsetInBuffer,
                      m_pi);

        uint32_t ccbSize = GCFG_HOST_CYCLIC_BUFFER_SIZE.value() * 1024;
        while (m_prevOffsetInBuffer != m_offsetInBuffer)
        {
            uint64_t inc = pqm::dumpPqmPacket(getBufferOffsetAddr(m_prevOffsetInBuffer));
            if (inc == 0)
            {
                break;
            }

            m_prevOffsetInBuffer = (m_prevOffsetInBuffer + inc) % ccbSize;
        }
    }
}