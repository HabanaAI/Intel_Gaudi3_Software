#include "stream_job.hpp"
#include "queues/queue_interface.hpp"
#include "defs.h"

StreamJob::StreamJob(JobType jobType) : m_jobType(jobType) {}

static const std::string getJobStr(JobType jobType)
{
    switch (jobType)
    {
        case JobType::MEMCOPY_H2D:
            return "MEMCOPY_H2D";
        case JobType::MEMCOPY_D2H:
            return "MEMCOPY_D2H";
        case JobType::MEMCOPY_D2D:
            return "MEMCOPY_D2D";
        case JobType::MEMSET:
            return "MEMSET";
        case JobType::COMPUTE:
            return "COMPUTE";
        case JobType::NETWORK:
            return "NETWORK";
        case JobType::RECORD_EVENT:
            return "RECORD_EVENT";
        case JobType::WAIT_FOR_EVENT:
            return "WAIT_FOR_EVENT";
        default:
            return "";
    }
}

std::string StreamJob::getDescription() const
{
    return getJobStr(m_jobType) + ": " + getJobParams();
}

ComputeJob::ComputeJob(const synLaunchTensorInfoExt* launchTensorsInfo,
                       uint32_t                      launchTensorsAmount,
                       uint64_t                      workspaceAddress,
                       InternalRecipeHandle*         pRecipeHandle,
                       uint64_t                      assertAsyncMappedAddress,
                       uint32_t                      flags,
                       EventWithMappedTensorDB&      events,
                       uint8_t                       apiId)
: StreamJob(JobType::COMPUTE),
  m_launchTensorsInfo(launchTensorsInfo),
  m_launchTensorsAmount(launchTensorsAmount),
  m_workspaceAddress(workspaceAddress),
  m_pRecipeHandle(pRecipeHandle),
  m_assertAsyncMappedAddress(assertAsyncMappedAddress),
  m_flags(flags),
  m_events(events),
  m_apiId(apiId)
{
}

synStatus ComputeJob::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);
    return pStreamInterface->launch(m_launchTensorsInfo,
                                    m_launchTensorsAmount,
                                    m_workspaceAddress,
                                    m_pRecipeHandle,
                                    m_assertAsyncMappedAddress,
                                    m_flags,
                                    m_events,
                                    m_apiId);
}

std::string ComputeJob::getJobParams() const
{
    char buff[100];
    snprintf(buff, sizeof(buff), "recipeHandle=%p, m_launchTensorsAmount=%d", m_pRecipeHandle, m_launchTensorsAmount);
    return buff;
}

MemcopyJob::MemcopyJob(internalMemcopyParams& memcpyParams,
                       const internalDmaDir   direction,
                       bool                   isUserRequest,
                       uint8_t                apiId)
: StreamJob(getMemcopyJobType(direction)),
  m_memcpyParams(memcpyParams),
  m_direction(direction),
  m_isUserRequest(isUserRequest),
  m_apiId(apiId)
{
}

synStatus MemcopyJob::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);
    return pStreamInterface->memcopy(m_memcpyParams,
                                     m_direction,
                                     m_isUserRequest,
                                     nullptr,
                                     0 /* overrideMemsetVal */,
                                     false /* inspectCopiedContent */,
                                     nullptr /* pRecipeProgramBuffer */,
                                     m_apiId);
}

std::string MemcopyJob::getJobParams() const
{
    char buff[250];
    if (m_memcpyParams.size() > 0)
    {
        uint64_t totalSize = 0;
        for (const auto &param : m_memcpyParams)
        {
            totalSize += param.size;
        }

        snprintf(buff,
                 sizeof(buff),
                 "direction=%d, memcpyParamsSize=%lu totalSize=0x%lx firstCopySrc=0x%lx firstCopyDest=0x%lx firstCopySize=0x%lx",
                 m_direction,
                 m_memcpyParams.size(),
                 totalSize,
                 m_memcpyParams[0].src,
                 m_memcpyParams[0].dst,
                 m_memcpyParams[0].size);
    }
    else
    {
        snprintf(buff, sizeof(buff), "direction=%d, memcpyParamsSize=%lu", m_direction, m_memcpyParams.size());
    }

    return buff;
}

JobType MemcopyJob::getMemcopyJobType(internalDmaDir direction)
{
    JobType jobType;
    switch (direction)
    {
        case MEMCOPY_HOST_TO_DRAM:
        {
            jobType = JobType::MEMCOPY_H2D;
            break;
        }
        case MEMCOPY_DRAM_TO_HOST:
        {
            jobType = JobType::MEMCOPY_D2H;
            break;
        }
        case MEMCOPY_DRAM_TO_DRAM:
        {
            jobType = JobType::MEMCOPY_D2D;
            break;
        }
        default:
        {
            jobType = JobType::MEMCOPY_H2D;
            HB_ASSERT(false, "Got unsupported memcopy direction: {}", direction);
        }
    }

    return jobType;
}

MemsetJob::MemsetJob(uint64_t       pDeviceMem,
                     const uint32_t value,
                     const size_t   numOfElements,
                     const size_t   elementSize,
                     uint8_t        apiId)
: StreamJob(JobType::MEMSET),
  m_pDeviceMem(pDeviceMem),
  m_value(value),
  m_numOfElements(numOfElements),
  m_elementSize(elementSize),
  m_apiId(apiId)
{
}

synStatus MemsetJob::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);

    uint64_t data;

    switch (m_elementSize)
    {
        case sizeof(uint8_t):
        {
            uint8_t val = m_value;
            std::fill_n((uint8_t*)&data, (sizeof(data) / sizeof(val)), val);
            break;
        }
        case sizeof(uint16_t):
        {
            uint16_t val = m_value;
            std::fill_n((uint16_t*)&data, (sizeof(data) / sizeof(val)), val);
            break;
        }
        case sizeof(uint32_t):
        {
            uint32_t val = m_value;
            std::fill_n((uint32_t*)&data, (sizeof(data) / sizeof(val)), val);
            break;
        }
        default:
        {
            LOG_ERR(SYN_API, "{}:Unsupported size", HLLOG_FUNC);
            return synInvalidArgument;
        }
    }

    internalMemcopyParams memcpyParams {{.src = 0, .dst = m_pDeviceMem, .size = (m_numOfElements * m_elementSize)}};
    return pStreamInterface->memcopy(memcpyParams,
                                     MEMCOPY_DRAM_TO_DRAM,
                                     true,
                                     nullptr,
                                     data,
                                     false /* inspectCopiedContent */,
                                     nullptr /* pRecipeProgramBuffer */,
                                     m_apiId);
}

EventRecordJob::EventRecordJob(EventInterface& rEventInterface, synStreamHandle streamHandle)
: StreamJob(JobType::RECORD_EVENT), m_rEventInterface(rEventInterface), m_streamHandleHcl(streamHandle)
{
}

synStatus EventRecordJob::run(QueueInterface* pStreamInterface)
{
    HB_ASSERT_PTR(pStreamInterface);
    return pStreamInterface->eventRecord(m_rEventInterface, m_streamHandleHcl);
}

std::string EventRecordJob::getJobParams() const
{
    char buff[100];
    snprintf(buff, sizeof(buff), "last recorded event %s", m_rEventInterface.toString().c_str());
    return buff;
}

EventNetwork::EventNetwork() : StreamJob(JobType::NETWORK) {}

synStatus EventNetwork::run(QueueInterface* pStreamInterface)
{
    return synSuccess;  // We do execute any HCL work here as it is done later from the HCL context
}