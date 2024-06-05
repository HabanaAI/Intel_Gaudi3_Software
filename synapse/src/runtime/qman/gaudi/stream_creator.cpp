#include "stream_creator.hpp"

#include "master_qmans_definition.hpp"
#include "habana_global_conf.h"
#include "drm/habanalabs_accel.h"
#include "defs.h"

using namespace gaudi;

StreamCreator* StreamCreator::s_streamCreator = nullptr;

StreamCreator* StreamCreator::getInstance()
{
    if (s_streamCreator == nullptr)
    {
        s_streamCreator = new StreamCreator();
    }

    return s_streamCreator;
};

bool StreamCreator::getPhysicalQueueId(PhysicalQueuesId&   physicalQueuesId,
                                       const TrainingQueue logicalQueue,
                                       uint32_t            physicalQueueOffset) const
{
    switch (logicalQueue)
    {
        case TRAINING_QUEUE_DMA_UP:
            physicalQueuesId = gaudi::QmansDefinition::getInstance()->getStreamsMasterQueueIdForMemcopyFromDevice();
            break;

        case TRAINING_QUEUE_DMA_DOWN_USER:
            physicalQueuesId = gaudi::QmansDefinition::getInstance()->getStreamsMasterQueueIdForMemcopyToDevice();
            break;

        case TRAINING_QUEUE_DMA_DOWN_SYNAPSE:
            physicalQueuesId =
                gaudi::QmansDefinition::getInstance()->getStreamsMasterQueueIdForSynapseMemcopyToDevice();
            break;

        case TRAINING_QUEUE_COMPUTE_0:
        case TRAINING_QUEUE_COMPUTE_1:
            physicalQueuesId =
                gaudi::QmansDefinition::getInstance()->getStreamMasterQueueIdForCompute() + physicalQueueOffset;
            break;

        case TRAINING_QUEUE_COLLECTIVE_0:
        case TRAINING_QUEUE_COLLECTIVE_1:
        case TRAINING_QUEUE_COLLECTIVE_2:
            physicalQueuesId =
                gaudi::QmansDefinition::getInstance()->getStreamsMasterQueueIdForCollective() + physicalQueueOffset;
            break;

        case TRAINING_QUEUE_DEV_TO_DEV_SYNAPSE:
            return false;

        case TRAINING_QUEUE_NUM:
            HB_ASSERT(false, "Unexpected logical-queue type");
            return false;
    }

    return true;
}

uint32_t StreamCreator::getStreamsElementsRestriction(internalStreamType queueType) const
{
    uint32_t streamsElementsRestriction;
    switch (queueType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
        case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
        case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
        {
            streamsElementsRestriction = 1;
            break;
        }
        case INTERNAL_STREAM_TYPE_COMPUTE:
        {
            // There are two different H/W qman for COMPUTE
            streamsElementsRestriction = 2;
            break;
        }
        case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
        {
            streamsElementsRestriction = 1;
            break;
        }
        default:
        {
            streamsElementsRestriction = 0;
            HB_ASSERT(false, "Unexpected logical-queue type");
        }
    }

    return streamsElementsRestriction;
}
