#pragma once

#include "synapse_common_types.h"
#include "define_synapse_common.hpp"
#include "containers/slot_map.hpp"
#include "synapse_common_types.hpp"
#include "synapse_api_types.h"
#include "dfa_defines.hpp"
#include "hcl_public_streams.h"

#include <memory>

struct tensor_info_t;
struct BasicQueueInfo;
class EventInterface;
class EventWithMappedTensor;
using EventWithMappedTensorSptr = SlotMapItemSptr<EventWithMappedTensor>;
using EventWithMappedTensorDB   = std::vector<EventWithMappedTensorSptr>;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

class QueueInterface
{
public:
    virtual ~QueueInterface() = default;

    virtual const BasicQueueInfo& getBasicQueueInfo() const = 0;

    // HCL
    virtual synStatus createHclStream() = 0;

    virtual synStatus destroyHclStream() = 0;

    virtual hcl::hclStreamHandle getHclStreamHandle() const = 0;

    virtual uint32_t getPhysicalQueueOffset() const = 0;

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const = 0;

    // In the case of QueueCollectiveNetwork the streamHandle is the stream owner handle in other cases N/A
    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) = 0;

    // In the case of QueueCollectiveNetwork the streamHandle is the stream owner handle in other cases N/A
    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) = 0;

    virtual synStatus query() = 0;

    // In the case of QueueCollectiveNetwork the streamHandle is the stream owner handle in other cases N/A
    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) = 0;

    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest,
                              QueueInterface*        pPreviousStream,
                              const uint64_t         overrideMemsetVal,
                              bool                   inspectCopiedContent,
                              SpRecipeProgramBuffer* pRecipeProgramBuffer,
                              uint8_t                apiId) = 0;

    virtual synStatus launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             uint64_t                      assertAsyncMappedAddress,
                             uint32_t                      flags,
                             EventWithMappedTensorDB&      events,
                             uint8_t                       apiId) = 0;

    virtual void finalize() = 0;

    /**
     * get stream status upon any kind of failure
     * @param logForUser if true, log minimized info, user-friendly.
     */
    virtual void dfaInfo(DfaReq dfaReq, uint64_t csSeq) = 0;

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const = 0;
};
