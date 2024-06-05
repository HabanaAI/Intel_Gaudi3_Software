#include <memory>
#include "dma_dispatchers_creator.h"
#include "platform/gaudi2/graph_compiler/dma_dispatcher.h"
#include "habana_global_conf.h"

QueueDispatcherPtr gaudi2::DmaDispatcherCreator::makeDmaDispatcher(uint8_t mask,
                                                                   unsigned dispatcherIndex,
                                                                   HabanaGraph* g) const
{
    return std::make_shared<gaudi2::DmaDispatcher>(mask, GCFG_DMA_SYNC_TRACE_EN_MASK.value(), dispatcherIndex, g);
}

bool gaudi2::DmaDispatcherCreator::isParallelLevelsValid(std::set<QueueDispatcherParams> allParallelLevels) const
{
    return (allParallelLevels.size() == 1);  // only one parallel level is allowed in gaudi2
}
