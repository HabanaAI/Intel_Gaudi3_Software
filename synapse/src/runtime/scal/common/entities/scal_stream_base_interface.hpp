#pragma once

#include "synapse_common_types.h"
#include "runtime/scal/common/infra/scal_types.hpp"
#include <string>

namespace common
{
class DeviceInfoInterface;
}

struct ScalEvent;
struct ScalLongSyncObject;
class ScalCompletionGroupBase;
class ScalMemoryPool;
class ScalMonitorBase;

struct ScalStreamCtorInfoBase
{
    const std::string&                 name;
    ScalMemoryPool&                    mpHostShared;
    scal_handle_t                      devHndl              = nullptr;
    const DevStreamInfo*               devStreamInfo        = nullptr;
    const common::DeviceInfoInterface* deviceInfoInterface  = nullptr;
    ScalCompletionGroupBase*           pScalCompletionGroup = nullptr;
    synDeviceType                      devType              = synDeviceTypeInvalid;
    // fenceId is the ID used between different streams
    FenceIdType fenceId = -1;
    // fenceIdForCompute is the ID used to sync between a compute stream and its PDMA stream
    FenceIdType        fenceIdForCompute = -1;
    uint32_t           streamIdx         = -1;
    MonitorIdType      syncMonitorId     = -1;
    ResourceStreamType resourceType      = ResourceStreamType::AMOUNT;
};

struct PreAllocatedStreamMemory
{
    uint32_t             AddrCore  = 0;
    uint64_t             AddrDev   = 0;
    uint64_t             Size      = 0;
    scal_buffer_handle_t BufHandle = nullptr;
};

struct PreAllocatedStreamMemoryAll
{
    struct PreAllocatedStreamMemory global;
    struct PreAllocatedStreamMemory shared;
};

class ScalStreamBaseInterface
{
public:
    virtual ~ScalStreamBaseInterface() = default;

    virtual bool isDirectMode() const = 0;

    virtual const std::string getName() const = 0;

    virtual synStatus addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo) = 0;

    virtual void longSoRecord(bool isUserReq, ScalLongSyncObject& rLongSo) const = 0;

    virtual synStatus eventRecord(bool isUserReq, ScalEvent& scalEvent) const = 0;

    virtual synStatus longSoQuery(const ScalLongSyncObject& rLongSo, bool alwaysWaitForInterrupt = false) const = 0;

    // Wait on host for last longSo (e.g. see isUserReq + longSo @ addDispatchBarrier) to complete on the device
    virtual synStatus longSoWaitForLast(bool isUserReq, uint64_t timeoutMicroSec, const char* caller) const = 0;

    virtual synStatus longSoWaitOnDevice(const ScalLongSyncObject& rLongSo, bool isUserReq) = 0;

    virtual ScalLongSyncObject getIncrementedLongSo(bool isUserReq, uint64_t targetOffset = 1) = 0;

    virtual ScalLongSyncObject getTargetLongSo(uint64_t targetOffset) const = 0;

    // Wait on host for longSo (e.g. see longSo @ addDispatchBarrier) to complete on the device
    virtual synStatus
    longSoWait(const ScalLongSyncObject& rLongSo, uint64_t timeoutMicroSec, const char* caller) const = 0;

    virtual synStatus addStreamFenceWait(uint32_t target, bool isUserReq, bool isInternalComputeSync) = 0;

    virtual ScalMonitorBase* testGetScalMonitor() = 0;

    // Adds a packet, which writes data to an LBW-address
    virtual synStatus addLbwWrite(uint64_t dst_addr, uint32_t data, bool block_stream, bool send, bool isInSyncMgr) = 0;

    virtual synStatus getStreamInfo(std::string& info, uint64_t& devLongSo) = 0;

    virtual bool prevCmdIsWait() = 0;

    virtual void dfaDumpScalStream() = 0;

    virtual TdrRtn tdr(TdrType tdrType) = 0;

    virtual void printCgTdrInfo(bool tdr) const = 0;

    virtual bool getStaticMonitorPayloadInfo(uint64_t& payloadAddress, uint32_t& payloadData) const = 0;

    virtual ResourceStreamType getResourceType() const = 0;

    virtual bool isComputeStream() = 0;
};
