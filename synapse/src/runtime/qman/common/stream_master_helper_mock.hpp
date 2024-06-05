#pragma once

#include "stream_master_helper_interface.hpp"

class StreamMasterHelperMock : public StreamMasterHelperInterface
{
public:
    StreamMasterHelperMock() : StreamMasterHelperInterface() {}

    virtual ~StreamMasterHelperMock() override = default;

    virtual bool createStreamMasterJobBuffer(uint64_t arbMasterBaseQmanId) override { return true; }

    virtual uint32_t getStreamMasterBufferSize() const override { return 0; }
    virtual uint64_t getStreamMasterBufferHandle() const override { return 0; }
    virtual uint64_t getStreamMasterBufferHostAddress() const override { return 0; }

    virtual uint32_t getStreamMasterFenceBufferSize() const override { return 0; }
    virtual uint64_t getStreamMasterFenceBufferHandle() const override { return 0; }
    virtual uint64_t getStreamMasterFenceHostAddress() const override { return 0; }

    virtual uint32_t getStreamMasterFenceClearBufferSize() const override { return 0; }
    virtual uint64_t getStreamMasterFenceClearBufferHandle() const override { return 0; }
    virtual uint64_t getStreamMasterFenceClearHostAddress() const override { return 0; }
};