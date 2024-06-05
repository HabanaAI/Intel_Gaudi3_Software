#pragma once

#include "scal_stream_base.hpp"

#include "define_synapse_common.hpp"
#include "global_statistics.hpp"
#include "scal_monitor_direct_mode.hpp"
#include "stream_cyclic_buffer_direct_mode.hpp"
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
struct ScalStreamCtorInfo;

class ScalStreamCopyDirectMode : public ScalStreamCopyBase
{
public:
    ScalStreamCopyDirectMode(const ScalStreamCtorInfoBase* pScalStreamCtorInfo);

    virtual ~ScalStreamCopyDirectMode();

    virtual bool isDirectMode() const override { return true; };

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

protected:
    virtual ScalMonitorBase* getMonitor() override { return &m_scalMonitor; };

    virtual const ScalMonitorBase* getMonitor() const { return &m_scalMonitor; };

    virtual std::string getSchedulerInfo(unsigned& schedulerIdx) const { return "Not supported"; };

private:
    // PQM commands' creation methods
    synStatus addPqmMsgLong(uint32_t value, uint64_t address, bool send);

    synStatus addPqmMsgShort(uint32_t value, uint64_t address, bool send);

    synStatus addPqmFence(uint8_t fenceId, uint32_t decVal, uint32_t targetVal, bool send);

    synStatus addPqmNopCmd(bool send);

    synStatus addPqmLinPdmaCmd(uint64_t           src,
                               uint64_t           dst,
                               uint32_t           size,
                               bool               bMemset,
                               PdmaDir            direction,
                               LinPdmaBarrierMode barrierMode,
                               uint64_t           barrierAddress,
                               uint32_t           barrierData,
                               uint32_t           fenceDecVal,
                               uint32_t           fenceTarget,
                               uint32_t           fenceId,
                               bool               send,
                               uint8_t            apiId);

    virtual synStatus addFenceWait(uint32_t target, FenceIdType fenceId, bool send, bool isGlobal) override;

    virtual synStatus addEmptyPdmaPacket() override;

    virtual synStatus addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo) override;

    virtual synStatus initInternal() override;

    virtual StreamCyclicBufferBase* getStreamCyclicBuffer() override { return &m_streamCyclicBuffer; };

    bool retrievStreamInfo();

    uint64_t getCqLongSoAddress();
    uint32_t getCqLongSoValue();

    bool initPqmMsgShortBaseAddrs();

    uint16_t m_maxLinPdmasInChunk;  // how many pdmas we can do before we have to add a signal_to_cq

    ScalMonitorDirectMode m_scalMonitor;

    StreamCyclicBufferDirectMode m_streamCyclicBuffer;

    uint8_t m_apiId;

    // This map will contain all base addreses and the coresponding index for this address in the SPDMA msg base addr.
    // The pair of base address and the coresponding index are by the sync manager ID.
    // When composing a msgShort, the 'base' field get the index of the desired base addr from this map.
    // The 'msg_addr_offset' is relative to the base addr described above.
    std::unordered_map<unsigned, std::pair<uint64_t, unsigned>> m_pqmMsgShortBaseAddrs;
};
