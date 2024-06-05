#pragma once

#include "stream_cyclic_buffer_base.hpp"

#include "scal_internal/pkt_macros.hpp"

class StreamCyclicBufferDirectMode : public StreamCyclicBufferBase
{
public:
    StreamCyclicBufferDirectMode(std::string streamName)
    : StreamCyclicBufferBase(streamName)
    {};

    virtual ~StreamCyclicBufferDirectMode() = default;

    void init(ScalCompletionGroupBase* pScalCompletionGroup,
              uint8_t*                 cyclicBufferBaseAddress,
              uint64_t                 streamHndl,
              uint16_t                 cmdAlign);

protected:
    virtual void addAlignmentPackets(uint64_t alignSize) override;
    virtual void dumpSubmission(const char* desc) override;

private:
    uint16_t m_submitAlign;
};