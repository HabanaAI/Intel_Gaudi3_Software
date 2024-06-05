#include "dma_dispatchers_creator.h"
#include "platform/gaudi/graph_compiler/dma_dispatcher.h"
#include "habana_global_conf.h"

QueueDispatcherPtr gaudi::DmaDispatcherCreator::makeDmaDispatcher(uint8_t mask,
                                                                  unsigned dispatcherIndex,
                                                                  HabanaGraph* g) const
{
    return std::make_shared<gaudi::DMADispatcher>(mask, GCFG_DMA_SYNC_TRACE_EN_MASK.value(), dispatcherIndex, g);
}
