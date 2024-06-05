#include "streams_container.hpp"
#include "device_info.hpp"
#include "scal_stream.hpp"
#include "habana_global_conf.h"

#include "runtime/scal/common/entities/scal_stream_copy_direct_mode.hpp"

#include "log_manager.h"

// engines-arc
#include "gaudi3_arc_common_packets.h"
#include "gaudi3_arc_host_packets.h"

using namespace gaudi3;

struct g3fw_arc_fw_synapse_config_t : g3fw::arc_fw_synapse_config_t
{
};

StreamsContainer::StreamsContainer(const common::DeviceInfoInterface* deviceInfoInterface) : common::ScalStreamsContainer(deviceInfoInterface)
{
    m_arcFwSynapseConfig_t  = std::make_unique<g3fw_arc_fw_synapse_config_t>();

    m_arcFwSynapseConfig_t->sync_scheme_mode = g3fw::ARC_FW_GAUDI3_SYNC_SCHEME;
    m_arcFwSynapseConfig_t->cme_enable = true;

    LOG_INFO(SYN_DEVICE,
                  "Initializing... newSchemeMode={} cme_enable={}",
                  m_arcFwSynapseConfig_t->sync_scheme_mode, m_arcFwSynapseConfig_t->cme_enable);

    m_scalArcFwConfigHandle = (scal_arc_fw_config_handle_t)m_arcFwSynapseConfig_t.get();
}

const StreamsContainer::ResourcesTypeToClustersDB& StreamsContainer::getResourcesClusters() const
{
    static std::map<ResourceStreamType, ResourceClusters> resourcesClusters =
    {
        {ResourceStreamType::USER_DMA_UP,         {{SCAL_PDMA_RX_GROUP},            {}}},
        {ResourceStreamType::USER_DMA_DOWN,       {{SCAL_PDMA_TX_DATA_GROUP},       {}}},
        {ResourceStreamType::USER_DEV_TO_DEV,     {{SCAL_PDMA_RX_GROUP},            {}}},
        {ResourceStreamType::SYNAPSE_DMA_UP,      {{SCAL_PDMA_RX_DEBUG_GROUP},      {SCAL_PDMA_RX_GROUP}}},
        {ResourceStreamType::SYNAPSE_DMA_DOWN,    {{SCAL_PDMA_TX_CMD_GROUP},        {}}},
        {ResourceStreamType::SYNAPSE_DEV_TO_DEV,  {{SCAL_PDMA_DEV2DEV_DEBUG_GROUP}, {SCAL_PDMA_RX_GROUP}}},
        {ResourceStreamType::COMPUTE,             {{SCAL_MME_COMPUTE_GROUP,
                                                    SCAL_TPC_COMPUTE_GROUP,
                                                    SCAL_RTR_COMPUTE_GROUP,
                                                    SCAL_CME_GROUP},                {}}},
    };
    return resourcesClusters;
}

ScalStreamBaseInterface*
StreamsContainer::createCopySchedulerModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const
{
    pScalStreamCtorInfo->deviceInfoInterface = m_apDeviceInfoInterface;
    pScalStreamCtorInfo->devStreamInfo       = &m_deviceStreamInfo;
    pScalStreamCtorInfo->devType             = synDeviceGaudi3;

    return new ScalStreamCopyGaudi3(pScalStreamCtorInfo);
}

ScalStreamBaseInterface* StreamsContainer::createCopyDirectModeStream(ScalStreamCtorInfoBase* pScalStreamCtorInfo) const
{
    pScalStreamCtorInfo->deviceInfoInterface = m_apDeviceInfoInterface;
    pScalStreamCtorInfo->devStreamInfo       = &m_deviceStreamInfo;
    pScalStreamCtorInfo->devType             = synDeviceGaudi3;

    return new ScalStreamCopyDirectMode(pScalStreamCtorInfo);
}

ScalStreamBaseInterface* StreamsContainer::createComputeStream(ScalStreamCtorInfo* pScalStreamCtorInfo) const
{
    pScalStreamCtorInfo->deviceInfoInterface = m_apDeviceInfoInterface;
    pScalStreamCtorInfo->devStreamInfo       = &m_deviceStreamInfo;
    pScalStreamCtorInfo->devType             = synDeviceGaudi3;

    return new ScalStreamComputeGaudi3(pScalStreamCtorInfo);
}