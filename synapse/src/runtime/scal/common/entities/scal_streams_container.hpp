#pragma once

#include "define_synapse_common.hpp"
#include "device_info_interface.hpp"

#include <memory>
#include <mutex>
#include <vector>
#include <map>

class ScalCompletionGroupBase;
class ScalMemoryPool;
class ScalStreamsMonitors;
class ScalStreamsFences;

struct ScalStreamCtorInfo;
struct ScalStreamCtorInfoBase;

namespace common
{
class DeviceInfoInterface;
}

class ScalStreamBaseInterface;

struct StreamAndIndex
{
    ScalStreamBaseInterface* pStream;
    unsigned                idx;
};

namespace common
{
enum class StreamModeType
{
    SCHEDULER,
    DIRECT
};

struct StreamsSetInfo
{
    std::string_view m_streamsSetName;
    uint16_t         m_amount;
};

class ScalStreamsContainer
{
public:

    ScalStreamsContainer(const common::DeviceInfoInterface* deviceInfoInterface);
    virtual ~ScalStreamsContainer();

    synStatus createStreams(scal_handle_t                devHndl,
                            ScalMemoryPool&              memoryPoolHostShared,
                            ScalStreamsMonitors&         streamsMonitors,
                            ScalStreamsFences&           streamsFences);

    bool getFreeStream(ResourceStreamType resourceType, StreamAndIndex& streamInfo);
    ScalStreamBaseInterface* getFreeComputeResources(const ComputeCompoundResources*& pCompoundResourceInfo);

    ScalStreamBaseInterface* debugGetCreatedStream(ResourceStreamType resourceType);
    ScalStreamBaseInterface*
    debugGetCreatedComputeResources(const ComputeCompoundResources*& pComputeCompoundResources);

    synStatus releaseStream(ScalStreamBaseInterface* scalStream);
    synStatus releaseComputeResources(ScalStreamBaseInterface* pComputeScalStream);

    synStatus releaseAllDeviceStreams();

    unsigned getResourcesAmount(ResourceStreamType type);
    bool getResourcesStreamMode(StreamModeType& streamModeType, ResourceStreamType type);

    inline scal_arc_fw_config_handle_t& getArcFwConfigHandle() { return m_scalArcFwConfigHandle; };

protected:
    // A given resource may have an array of clusters, which they are using
    // We may change the clusters used and hence the primary & secondary clusters type
    // In case the primary is not defined, we will use the secondary instead
    struct ResourceClusters
    {
        std::vector<unsigned> primaryClusterType;
        std::vector<unsigned> secondaryClusterType;
    };

    ScalStreamBaseInterface* createStream(scal_handle_t            devHndl,
                                          ScalMemoryPool&          memoryPoolHostShared,
                                          ScalStreamsMonitors&     streamsMonitors,
                                          ScalStreamsFences&       streamsFences,
                                          StreamModeType           queueType,
                                          ResourceStreamType       resourceType,
                                          unsigned                 streamIndex,
                                          const std::string&       streamName,
                                          ScalCompletionGroupBase* pCompletionGroup,
                                          FenceIdType              globalFenceId) const;

    ScalCompletionGroupBase* createCompletionGroup(scal_handle_t      devHndl,
                                                   StreamModeType     queueType,
                                                   const std::string& completionGroupName) const;

    bool updateNumOfCompletions(scal_handle_t devHandle);

    synStatus releaseAllDeviceStreamsL();

    bool getStreamSetInfo(const StreamsSetInfo*& pStreamSetInfo, ResourceStreamType resourceType) const;

    bool createStreamResource(const std::string& streamSetName,
                              ResourceStreamType resourceType,
                              unsigned           streamIndex);

    bool updateResourceUsage(ResourceStreamType type, unsigned index, bool isUsed);

    ScalStreamBaseInterface* getScalStream(ResourceStreamType type, unsigned index);

    synStatus initComputeStream();

    virtual const std::map<ResourceStreamType, ResourceClusters>& getResourcesClusters() const = 0;

    virtual ScalStreamBaseInterface*
    createCopySchedulerModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const = 0;

    virtual ScalStreamBaseInterface* createCopyDirectModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const = 0;

    virtual ScalStreamBaseInterface* createComputeStream(ScalStreamCtorInfo* pScalStreamCtorInfo) const = 0;

    DevStreamInfo m_deviceStreamInfo;

    scal_arc_fw_config_handle_t m_scalArcFwConfigHandle;

    const DeviceInfoInterface* m_apDeviceInfoInterface;

    std::mutex m_mutex;

private:
    struct SingleScalStreamInfo
    {
        std::string streamName;
        std::string completionName;
        uint32_t    streamIdx;

        ScalStreamBaseInterface* pScalStream;
        ScalCompletionGroupBase* pScalCompletionGroup;

        bool used = false;
    };

    struct SingleResourceTypeInfo
    {
        StreamModeType                    streamModeType = StreamModeType::SCHEDULER;
        std::vector<SingleScalStreamInfo> scalStreamsInfo;
    };

    bool initStreamsResources(scal_handle_t& devHndl);
    bool initSingleStreamsSetResources(unsigned&          streamsAmount,
                                       StreamModeType&    streamModeType,
                                       const std::string& streamSetName,
                                       scal_handle_t&     devHndl);

    synStatus createSingleScalResource(SingleScalStreamInfo&        scalStreamInfo,
                                       ResourceStreamType           resourceType,
                                       StreamModeType               streamModeType,
                                       unsigned                     streamIndex,
                                       scal_handle_t                devHndl,
                                       ScalMemoryPool&              memoryPoolHostShared,
                                       ScalStreamsMonitors&         streamsMonitors,
                                       ScalStreamsFences&           streamsFences,
                                       FenceIdType                  globalFenceId);

    bool isValidStreamSetIndex(ResourceStreamType type, unsigned index);

    bool updateClustersSetNumOfCompletion(scal_handle_t devHandle,
                                          const std::vector<unsigned>& clustersSet);
    int updateSingleClustersNumOfCompletions(scal_handle_t      devHandle,
                                             uint32_t           clusterType,
                                             const std::string& clusterName);
    void setResourceInfo(ResourceStreamType           resourceType,
                         const std::vector<unsigned>& resourceClustersType);

    bool createCompoundResources();

    // NOTE: changes to the vector below effects test stream_restriction_ASIC
    // streamId is used also for streamId to the pdma.

    SingleResourceTypeInfo m_streamsResources[(uint8_t)ResourceStreamType::AMOUNT] {};

    std::vector<ComputeCompoundResources> m_computeCompoundResources;
};
}  // namespace common
