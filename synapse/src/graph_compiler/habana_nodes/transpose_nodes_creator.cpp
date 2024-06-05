#include "node_factory.h"
#include "transpose_utils.h"
#include "graph_traits.h"
#include "transpose_nodes_creator.h"
#include "habana_graph.h"
#include "compilation_hal_reader.h"

#include "transpose_via_mme.h"
#include "transpose_via_dma.h"
#include "transpose_via_tpc.h"

NodeVector TransposeNodeStrategy::handleAliasedOutputTranspose(const TransposeNodeParams& transposeNodeParams,
                                                               const HalReaderPtr&        hal) const
{
    const TensorPtr& output = transposeNodeParams.output;
    TensorPtr        middle = output->cloneGeometry();
    middle->setName(fmt::format("{}_dense_copy", output->getName()));

    TransposeNodeParams newTransposeParams = {transposeNodeParams.input,
                                              middle,
                                              transposeNodeParams.permutation,
                                              transposeNodeParams.nodeName,
                                              transposeNodeParams.preferLogicalBeforePhysical,
                                              transposeNodeParams.preferTransposeOnlyOnce};

    NodePtr    memcpyNode = NodeFactory::createInternalNode({middle},
                                                         {output},
                                                         nullptr,
                                                         NodeFactory::memcpyNodeTypeName,
                                                         fmt::format("{}_internal_memcpy", output->getName()));
    NodeVector ret        = extract(newTransposeParams, hal);
    ret.push_back(memcpyNode);
    return ret;
}

NodeVector TransposeNodeStrategy::createReshapeWithExtractTranspose(const TensorPtr&                 in,
                                                                    const TensorPtr&                 out,
                                                                    const TransposePermutationArray& permutation,
                                                                    const std::string&               name)
{
    bool      enforceLogical = true;
    TensorPtr shapeOut       = out->cloneGeometry();
    shapeOut->setTensorInWorkspace();
    shapeOut->setShapeTensor(SHAPE_TENSOR);

    synTransposeParamsNDims params = permutationToParams(permutation);

    NodePtr exctractTranspose = NodeFactory::createInternalNode({in},
                                                                {shapeOut},
                                                                &params,
                                                                NodeFactory::transposedShapeNodeTypeName,
                                                                fmt::format("{}_internal_transpose_shape", name));

    NodePtr reshapeNode = NodeFactory::createInternalNode(TensorVector {in, shapeOut},
                                                          TensorVector {out},
                                                          &enforceLogical,
                                                          "reshape",
                                                          fmt::format("{}_via_reshape", name));
    LOG_DEBUG(GC, "adding reshape transpose node - {}", name);
    return {exctractTranspose, reshapeNode};
}

using TransposeStrategiesArray = std::array<const TransposeNodeStrategy*, 5>;

static LogicTransposeStrategies getLogicTransposeStrategies()
{
    static const TransposeViaTransposeShape transposeViaTransposeShape;
    static const TransposeViaReshape        transposeViaReshape;
    static const TransposeViaLogical        transposeViaLogical;
    static const TransposeWithStaticShape   transposeWithStaticShape;
    // Logic transpose strategies in descending priority
    return {&transposeViaTransposeShape, &transposeViaReshape, &transposeViaLogical, &transposeWithStaticShape};
}

// Return physical strategy according to Hal
static const TransposeNodeStrategy* getPhysicalStrategy(const HalReaderPtr&             hal,
                                                        const LogicTransposeStrategies& logicStrategies)
{
    HabanaDeviceType             devType = hal->getTransposeEngine();
    static const TransposeViaDMA transposeViaDma;
    static const TransposeViaTPC transposeViaTpc;
    static const TransposeViaMME transposeViaMme(logicStrategies);

    switch (devType)
    {
        case HabanaDeviceType::DEVICE_EDMA:
            return &transposeViaDma;
        case HabanaDeviceType::DEVICE_MME:
            return &transposeViaMme;
        case HabanaDeviceType::DEVICE_TPC:
            return &transposeViaTpc;
        default:
            HB_ASSERT(false, "Unsupported transpose engine");
            return &transposeViaDma;
    }
}


static TransposeStrategiesArray getTransposeStrategies(const HalReaderPtr& hal)
{
    TransposeStrategiesArray strategies;
    const auto               logicStrategies = getLogicTransposeStrategies();
    static_assert(strategies.size() == logicStrategies.size() /*logic strategies*/ + 1 /*physical strategy*/);
    for (auto idx = 0; idx < logicStrategies.size(); ++idx)
    {
        strategies.at(idx) = logicStrategies.at(idx);
    }
    strategies.at(strategies.size() - 1) = getPhysicalStrategy(hal, logicStrategies);
    return strategies;
}

const TransposeNodeStrategy* TransposeNodesCreator::getWinningStrategy(const TransposeNodeParams& transposeNodeParams,
                                                                       TransposeStrategyID        strategyToSkip) const
{
    const auto& strategies = getTransposeStrategies(m_halReader);
    auto strategyIt = std::find_if(strategies.begin(), strategies.end(), [&](const TransposeNodeStrategy* strategy) {
        return strategy->getStrategyID() != strategyToSkip && strategy->canBeUsed(transposeNodeParams, m_halReader);
    });
    HB_ASSERT(strategyIt != strategies.end(),
              "No available strategy to transpose: {}, inSizes: {}, permutation: ({}), outSizes: {}",
              transposeNodeParams.nodeName.value_or("NoName"),
              transposeNodeParams.input->getDimSizesStr(),
              toString(transposeNodeParams.permutation, ','),
              transposeNodeParams.output->getDimSizesStr());
    return *strategyIt;
}

NodeVector TransposeNodesCreator::getTransposeNodesByParams(const TransposeNodeParams& transposeNodeParams,
                                                            TransposeStrategyID        strategyToSkip) const
{
    const auto& strategy = getWinningStrategy(transposeNodeParams, strategyToSkip);
    LOG_INFO(HABANA_NODE, "Selected \"{}\", for transposing", strategy->strategyName());
    return strategy->extract(transposeNodeParams, m_halReader);
}

uint64_t TransposeNodesCreator::getTransposeCostByParams(const TransposeNodeParams& transposeNodeParams,
                                                         TransposeStrategyID        strategyToSkip) const
{
    const auto& strategy = getWinningStrategy(transposeNodeParams, strategyToSkip);
    return strategy->calculateCost(transposeNodeParams, m_halReader);
}

NodeVector TransposeNodesCreator::getTransposeNodes(const TransposeNode& transpose,
                                                    TransposeStrategyID  strategyToSkip) const
{
    return getTransposeNodesByParams(TransposeNodeParams::fromNode(transpose), strategyToSkip);
}

std::pair<NodeVector, uint64_t>
TransposeNodesCreator::getTransposeNodesAndCost(const TransposeNode& transpose,
                                                TransposeStrategyID  strategyToSkip) const
{
    const auto& nodes     = getTransposeNodes(transpose, strategyToSkip);
    const auto& costModel = getCostModel();
    HB_ASSERT_PTR(costModel);

    return {nodes, costModel->getCost(nodes)};
}

std::shared_ptr<TransposeCostModel> TransposeNodesCreator::getCostModel() const
{
    HabanaDeviceType devType = m_halReader->getTransposeEngine();

    switch (devType)
    {
        case HabanaDeviceType::DEVICE_EDMA:
            return std::make_shared<DmaTransposeCostModel>();
        case HabanaDeviceType::DEVICE_MME:
            return std::make_shared<MmeTransposeCostModel>();
        case HabanaDeviceType::DEVICE_TPC:
            return std::make_shared<TpcTransposeCostModel>();
        default:
            HB_ASSERT(false, "Unsupported transpose engine");
            return nullptr;
    }
}
