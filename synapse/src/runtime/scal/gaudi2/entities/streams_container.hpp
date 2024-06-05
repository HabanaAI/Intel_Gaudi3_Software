#pragma once

#include "synapse_common_types.h"
#include "runtime/scal/common/entities/scal_streams_container.hpp"

struct g2_arc_fw_synapse_config_t;

namespace gaudi2
{

class StreamsContainer : public common::ScalStreamsContainer
{
public:
    StreamsContainer(const common::DeviceInfoInterface* deviceInfoInterface);
    virtual ~StreamsContainer() = default;

protected:
    typedef std::map<ResourceStreamType, common::ScalStreamsContainer::ResourceClusters> ResourcesTypeToClustersDB;
    virtual const ResourcesTypeToClustersDB& getResourcesClusters() const override;

    virtual ScalStreamBaseInterface*
    createCopySchedulerModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const override;

    virtual ScalStreamBaseInterface*
    createCopyDirectModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const override;

    virtual ScalStreamBaseInterface* createComputeStream(ScalStreamCtorInfo* pScalStreamCtorInfo) const override;

private:
    std::unique_ptr<g2_arc_fw_synapse_config_t> m_arcFwSynapseConfig_t;
};
}  // namespace gaudi2
