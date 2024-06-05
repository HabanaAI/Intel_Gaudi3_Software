#include "basic_queue_info.hpp"
#include "syn_logging.h"

std::string BasicQueueInfo::getDescription() const
{
    return fmt::format("Queue-ID {} Queue-Type {}-{} User-Queue-Index {}",
                       queueIndex,
                       internalStreamTypeToString(queueType),
                       queueType,
                       userQueueIndex);
}
