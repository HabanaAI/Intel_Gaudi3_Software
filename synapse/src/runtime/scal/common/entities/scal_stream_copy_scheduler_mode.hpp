#pragma once

#include "scal_stream_base.hpp"

#include "define_synapse_common.hpp"
#include "scal_monitor.hpp"
#include "stream_cyclic_buffer_scheduler_mode.hpp"
#include "synapse_common_types.h"

#include "infra/settable.h"

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "runtime/scal/common/infra/scal_types.hpp"

#include "runtime/scal/common/packets/define.hpp"

#include <string>
#include <queue>
#include <array>
#include <variant>

namespace common
{
class ArcSchedPacketsInterface;
class DeviceInfoInterface;
}  // namespace common

class ScalMemoryPool;
class ScalMonitorBase;
struct ScalEvent;

class ScalStreamCopySchedulerMode : public ScalStreamCopyBase
{
public:
    ScalStreamCopySchedulerMode(const ScalStreamCtorInfoBase* pScalStreamCtorInfo);

    virtual ~ScalStreamCopySchedulerMode();

    virtual bool isDirectMode() const override { return false; };

    virtual synStatus addFenceInc(FenceIdType fenceId, bool send);

    virtual ScalMonitorBase* testGetScalMonitor() override { return &m_scalMonitor; };

    virtual synStatus memcopy(ResourceStreamType           resourceType,
                              const internalMemcopyParams& memcpyParams,
                              bool                         isUserRequest,
                              bool                         send,
                              uint8_t                      apiId,
                              ScalLongSyncObject&          longSo,
                              uint64_t                     overrideMemsetVal,
                              MemcopySyncInfo&             memcopySyncInfo) override;

    // Adds a packet, which writes data to an LBW-address
    virtual synStatus addLbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool send, bool isInSyncMgr) override;

    bool setStreamPriority(uint32_t priority);

protected:
    virtual synStatus memcopyImpl(ResourceStreamType               resourceType,
                                  const internalMemcopyParamEntry* memcpyParams,
                                  uint32_t                         params_count,
                                  bool                             send,
                                  uint8_t                          apiId,
                                  bool                             memsetMode,
                                  bool                             sendUnfence,
                                  uint32_t                         completionGroupIndex,
                                  MemcopySyncInfo&                 memcopySyncInfo) = 0;

    synStatus addPdmaBatchTransfer(ResourceStreamType  resourceType,
                                   const void* const&& params,
                                   uint32_t            param_count,
                                   bool                send,
                                   uint8_t             apiId,
                                   bool                bMemset,
                                   uint32_t            payload,
                                   uint32_t            pay_addr,
                                   uint32_t            completionGroupIndex);

    virtual synStatus addFenceWait(uint32_t target, FenceIdType fenceId, bool send, bool isGlobal) override;

    virtual ScalMonitorBase* getMonitor() override { return &m_scalMonitor; };

    virtual const ScalMonitorBase* getMonitor() const { return &m_scalMonitor; };

    virtual std::string getSchedulerInfo(unsigned& schedulerIdx) const override;

    uint32_t m_qmanEngineGroupsAmount;
    uint32_t m_schedPdmaCommandsTransferMaxParamCount;
    uint64_t m_schedPdmaCommandsTransferMaxCopySize;

private:
    struct PdmaParams
    {
        uint8_t  engGrp;
        uint32_t workloadType;
        enum PdmaDirCtx dir;
    };

    PdmaParams getPdmaParams(ResourceStreamType operationType,
                             uint8_t            qmanEngineGroupsAmount);

    virtual synStatus addEmptyPdmaPacket() override;

    virtual synStatus addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo) override;

    virtual StreamCyclicBufferBase* getStreamCyclicBuffer() override { return &m_streamCyclicBuffer; };

    virtual synStatus initInternal() override;

    bool retrievStreamInfo();

    uint16_t m_maxBatchesInChunk;  // how many pdmas we can do before we have to add a signal_to_cq

    ScalMonitor m_scalMonitor;

    scal_core_handle_t   m_schedulerHandle;

    StreamCyclicBufferSchedulerMode m_streamCyclicBuffer;
};
