#pragma once

#include "synapse_common_types.h"
#include "stream_master_helper_interface.hpp"

class CommandBuffer;

namespace generic
{
class CommandBufferPktGenerator;
}

class StreamMasterHelper : public StreamMasterHelperInterface
{
public:
    StreamMasterHelper(synDeviceType deviceType, uint32_t physicalQueueOffset);

    virtual ~StreamMasterHelper();

    virtual bool createStreamMasterJobBuffer(uint64_t arbMasterBaseQmanId) override;

    virtual uint32_t getStreamMasterBufferSize() const override;
    virtual uint64_t getStreamMasterBufferHandle() const override;
    virtual uint64_t getStreamMasterBufferHostAddress() const override;

    virtual uint32_t getStreamMasterFenceBufferSize() const override;
    virtual uint64_t getStreamMasterFenceBufferHandle() const override;
    virtual uint64_t getStreamMasterFenceHostAddress() const override;

    virtual uint32_t getStreamMasterFenceClearBufferSize() const override;
    virtual uint64_t getStreamMasterFenceClearBufferHandle() const override;
    virtual uint64_t getStreamMasterFenceClearHostAddress() const override;

private:
    const synDeviceType                 m_deviceType;
    const uint32_t                      m_physicalQueueOffset;
    CommandBuffer*                      m_pCommandBuffer {nullptr};
    CommandBuffer*                      m_pFenceCommandBuffer {nullptr};
    CommandBuffer*                      m_pFenceClearCommandBuffer {nullptr};
    generic::CommandBufferPktGenerator* m_packetGenerator {nullptr};
};
