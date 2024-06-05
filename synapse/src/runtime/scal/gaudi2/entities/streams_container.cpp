#include "streams_container.hpp"

#include "defs.h"
#include "habana_global_conf_runtime.h"
#include "scal_stream.hpp"

#include "log_manager.h"

#include "runtime/scal/gaudi2/entities/device_info.hpp"

// engines-arc
#include "gaudi2_arc_common_packets.h"  // QMAN_ENGINE_GROUP_TYPE_COUNT

struct g2_arc_fw_synapse_config_t : g2fw::arc_fw_synapse_config_t
{
};

using namespace gaudi2;

StreamsContainer::StreamsContainer(const common::DeviceInfoInterface* deviceInfoInterface)
: common::ScalStreamsContainer(deviceInfoInterface)
{
    m_arcFwSynapseConfig_t  = std::make_unique<g2_arc_fw_synapse_config_t>();

    m_arcFwSynapseConfig_t->sync_scheme_mode = 1;
    LOG_INFO(SYN_DEVICE,
                  "Initializing... newSchemeMode={}",
                  m_arcFwSynapseConfig_t->sync_scheme_mode);

    m_scalArcFwConfigHandle = (scal_arc_fw_config_handle_t)m_arcFwSynapseConfig_t.get();
}

const StreamsContainer::ResourcesTypeToClustersDB& StreamsContainer::getResourcesClusters() const
{
    static std::map<ResourceStreamType, ResourceClusters> resourcesClusters =
    {
        {ResourceStreamType::USER_DMA_UP,         {{SCAL_PDMA_RX_GROUP},      {}}},
        {ResourceStreamType::USER_DMA_DOWN,       {{SCAL_PDMA_TX_DATA_GROUP}, {}}},
        {ResourceStreamType::USER_DEV_TO_DEV,     {{SCAL_PDMA_RX_GROUP},      {}}},
        {ResourceStreamType::SYNAPSE_DMA_UP,      {{SCAL_PDMA_RX_GROUP},      {}}},
        {ResourceStreamType::SYNAPSE_DMA_DOWN,    {{SCAL_PDMA_TX_CMD_GROUP},  {}}},
        {ResourceStreamType::SYNAPSE_DEV_TO_DEV,  {{SCAL_PDMA_RX_GROUP},      {}}},
        {ResourceStreamType::COMPUTE,             {{SCAL_MME_COMPUTE_GROUP,
                                                    SCAL_TPC_COMPUTE_GROUP,
                                                    SCAL_EDMA_COMPUTE_GROUP,
                                                    SCAL_RTR_COMPUTE_GROUP},  {}}},
    };
    return resourcesClusters;
}

ScalStreamBaseInterface*
StreamsContainer::createCopySchedulerModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const
{
    pScalStreamCtorInfo->deviceInfoInterface = m_apDeviceInfoInterface;
    pScalStreamCtorInfo->devStreamInfo       = &m_deviceStreamInfo;
    pScalStreamCtorInfo->devType             = synDeviceGaudi2;

    return new ScalStreamCopyGaudi2(pScalStreamCtorInfo);
}

ScalStreamBaseInterface* StreamsContainer::createCopyDirectModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const
{
    LOG_ERR(SYN_DEVICE,
                 "Direct-Mode is not supported");
    return nullptr;
}

ScalStreamBaseInterface* StreamsContainer::createComputeStream(ScalStreamCtorInfo* pScalStreamCtorInfo) const
{
    pScalStreamCtorInfo->deviceInfoInterface = m_apDeviceInfoInterface;
    pScalStreamCtorInfo->devStreamInfo       = &m_deviceStreamInfo;
    pScalStreamCtorInfo->devType             = synDeviceGaudi2;

    return new ScalStreamComputeGaudi2(pScalStreamCtorInfo);
}