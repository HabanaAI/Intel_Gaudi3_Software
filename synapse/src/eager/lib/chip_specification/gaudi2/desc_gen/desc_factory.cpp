#include "desc_factory.h"

// eager includes (relative to src/eager/lib/)
#include "chip_specification/gaudi2/desc_gen/dma_desc.h"
#include "chip_specification/gaudi2/desc_gen/mme_desc.h"
#include "chip_specification/gaudi2/desc_gen/tpc_desc.h"
#include "eager_graph.h"

namespace eager_mode::gaudi2_spec_info
{
DescGeneratorBasePtr DescFactory::createDescGenerator(EagerGraph& graph, const EagerNode& node, EngineType engineType)
{
    switch (engineType)
    {
        case EngineType::DMA:
            return std::make_unique<DmaDescGenerator>(graph, node);
        case EngineType::TPC:
            return std::make_unique<TpcDescGenerator>(graph, node);
        case EngineType::MME:
            return std::make_unique<MmeDescGenerator>(graph, node);
        default:
            break;
    }
    EAGER_ASSERT(false, "unsupported engine type");
    return nullptr;
}
}  // namespace eager_mode::gaudi2_spec_info
