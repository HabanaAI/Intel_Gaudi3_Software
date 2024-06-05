#pragma once

#include "stream_cyclic_buffer_base.hpp"

#include "scal_internal/pkt_macros.hpp"

class StreamCyclicBufferSchedulerMode : public StreamCyclicBufferBase
{
public:
    StreamCyclicBufferSchedulerMode(std::string streamName)
    : StreamCyclicBufferBase(streamName)
    {};

    virtual ~StreamCyclicBufferSchedulerMode() = default;

    void init(ScalCompletionGroupBase*           pScalCompletionGroup,
              uint8_t*                           cyclicBufferBaseAddress,
              uint64_t                           streamHndl,
              uint16_t                           cmdAlign,
              uint16_t                           submitAlign,
              std::variant<G2Packets,G3Packets>& gxPackets);

protected:
    virtual void addAlignmentPackets(uint64_t alignSize) override;
    virtual void dumpSubmission(const char* desc) override;

private:
    uint16_t m_submitAlign;

    std::variant<G2Packets,G3Packets> m_gxPackets;
};