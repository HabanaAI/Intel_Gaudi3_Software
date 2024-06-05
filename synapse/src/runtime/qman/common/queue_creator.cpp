#include "queue_creator.hpp"

using namespace common;

TrainingRetCode StreamCreator::selectTrainingQueue(const internalStreamType& queueType,
                                                   TrainingQueue&            type,
                                                   uint32_t                  elementIndex) const
{
    uint16_t   numOfTrainingQueueElements = 2;
    const bool needToSwitch               = ((elementIndex % numOfTrainingQueueElements) == 0);

    switch (queueType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
            type = TRAINING_QUEUE_DMA_UP;
            break;

        case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
            type = TRAINING_QUEUE_DMA_UP;
            break;

        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            type = TRAINING_QUEUE_DMA_UP;
            break;

        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            type = TRAINING_QUEUE_DMA_DOWN_USER;
            break;

        case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
            type = TRAINING_QUEUE_DMA_DOWN_SYNAPSE;
            break;

        case INTERNAL_STREAM_TYPE_COMPUTE:
            type = needToSwitch ? TRAINING_QUEUE_COMPUTE_0 : TRAINING_QUEUE_COMPUTE_1;
            break;

        case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
        {
            numOfTrainingQueueElements = 3;
            uint16_t switchOffset      = (elementIndex % numOfTrainingQueueElements);
            type                       = (TrainingQueue)(TRAINING_QUEUE_COLLECTIVE_0 + switchOffset);
        }
        break;

        case INTERNAL_STREAM_TYPE_NUM:
            return TRAINING_RET_CODE_INVALID_REQUEST;
    }

    return TRAINING_RET_CODE_SUCCESS;
}