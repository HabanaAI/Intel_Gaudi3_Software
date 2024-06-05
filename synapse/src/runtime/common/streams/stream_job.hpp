#pragma once

#include "runtime/common/queues/queue_compute_utils.hpp"
#include "define_synapse_common.hpp"

class QueueInterface;

enum class JobType
{
    MEMCOPY_H2D,
    MEMCOPY_D2H,
    MEMCOPY_D2D,
    MEMSET,
    COMPUTE,
    NETWORK,
    RECORD_EVENT,
    WAIT_FOR_EVENT
};

class StreamJob
{
public:
    StreamJob(JobType jobType);

    virtual ~StreamJob() = default;

    inline JobType getType() const { return m_jobType; }

    virtual synStatus run(QueueInterface* pStreamInterface) = 0;

    virtual std::string getJobParams() const { return ""; };

    virtual std::string getDescription() const;

protected:
    const JobType m_jobType;
};

class ComputeJob : public StreamJob
{
public:
    ComputeJob(const synLaunchTensorInfoExt* launchTensorsInfo,
               uint32_t                      launchTensorsAmount,
               uint64_t                      workspaceAddress,
               InternalRecipeHandle*         pRecipeHandle,
               uint64_t                      assertAsyncMappedAddress,
               uint32_t                      flags,
               EventWithMappedTensorDB&      events,
               uint8_t                       apiId);

    virtual synStatus run(QueueInterface* pStreamInterface) override;

    virtual std::string getJobParams() const override;

protected:
    const synLaunchTensorInfoExt* m_launchTensorsInfo;
    uint32_t                      m_launchTensorsAmount;
    uint64_t                      m_workspaceAddress;
    InternalRecipeHandle*         m_pRecipeHandle;
    uint64_t                      m_assertAsyncMappedAddress;
    uint32_t                      m_flags;
    EventWithMappedTensorDB&      m_events;
    const uint8_t                 m_apiId;
};

class MemcopyJob : public StreamJob
{
public:
    MemcopyJob(internalMemcopyParams& memcpyParams, const internalDmaDir direction, bool isUserRequest, uint8_t apiId);
    virtual synStatus run(QueueInterface* pStreamInterface) override;

    virtual std::string getJobParams() const override;

protected:
    static JobType         getMemcopyJobType(internalDmaDir direction);
    internalMemcopyParams& m_memcpyParams;
    const internalDmaDir   m_direction;
    const bool             m_isUserRequest;
    const uint8_t          m_apiId;
};

class MemsetJob : public StreamJob
{
public:
    MemsetJob(uint64_t       pDeviceMem,
              const uint32_t value,
              const size_t   numOfElements,
              const size_t   elementSize,
              uint8_t        apiId);
    virtual synStatus run(QueueInterface* pStreamInterface) override;

protected:
    uint64_t       m_pDeviceMem;
    const uint32_t m_value;
    const size_t   m_numOfElements;
    const size_t   m_elementSize;
    const uint8_t  m_apiId;
};

class EventRecordJob : public StreamJob
{
public:
    EventRecordJob(EventInterface& rEventInterface, synStreamHandle streamHandle);
    virtual synStatus run(QueueInterface* pStreamInterface) override;

    virtual std::string getJobParams() const override;

protected:
    EventInterface& m_rEventInterface;
    synStreamHandle m_streamHandleHcl;
};

class EventNetwork : public StreamJob
{
public:
    EventNetwork();
    virtual synStatus run(QueueInterface* pStreamInterface) override;
};