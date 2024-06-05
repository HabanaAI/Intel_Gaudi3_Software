#pragma once

class StreamMasterHelperInterface
{
public:
    virtual ~StreamMasterHelperInterface() = default;

    virtual bool createStreamMasterJobBuffer(uint64_t arbMasterBaseQmanId) = 0;

    virtual uint32_t getStreamMasterBufferSize() const        = 0;
    virtual uint64_t getStreamMasterBufferHandle() const      = 0;
    virtual uint64_t getStreamMasterBufferHostAddress() const = 0;

    virtual uint32_t getStreamMasterFenceBufferSize() const   = 0;
    virtual uint64_t getStreamMasterFenceBufferHandle() const = 0;
    virtual uint64_t getStreamMasterFenceHostAddress() const  = 0;

    virtual uint32_t getStreamMasterFenceClearBufferSize() const   = 0;
    virtual uint64_t getStreamMasterFenceClearBufferHandle() const = 0;
    virtual uint64_t getStreamMasterFenceClearHostAddress() const  = 0;
};
