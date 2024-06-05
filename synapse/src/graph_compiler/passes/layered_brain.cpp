#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "brain_conf.h"
#include "runner.h"
#include "habana_norms_handler.h"
#include "pair_grads.h"
#include "node_factory.h"

using namespace gc::layered_brain;

class LayeredBrain
{
public:
    explicit LayeredBrain(HabanaGraph& graph)
    : m_graph(graph),
      m_runner(std::make_unique<Runner>(graph)),
      m_normsHandler(std::make_unique<HabanaNormsHandler>(graph, std::make_shared<PatternNodesCollector>())),
      m_gradPairTransformer(std::make_unique<GradAReshapedGradBPairTransformer>(graph))
    {
    }

    bool run();

private:
    void preExecutionHandling();
    void postExecutionHandling();
    // TODO: SW-97127 - Temp WA to avoid undirected cycle in Resnet50 BWD layer.
    void handleSharedBundleInputs();

    HabanaGraph&                                             m_graph;
    const std::unique_ptr<Runner>                            m_runner;
    const std::unique_ptr<HabanaNormsHandler>                m_normsHandler;
    const std::unique_ptr<GradAReshapedGradBPairTransformer> m_gradPairTransformer;
};

void LayeredBrain::preExecutionHandling()
{
    // Break normalization undirected cycles as they are not supported atm
    m_normsHandler->findAndRemoveSliceNormNodes();

    // Pair more dedx/dedw to improve bwd pass bundles
    {
        const bool success = m_gradPairTransformer->optimizeGradPairs();
        HB_ASSERT(success, "Grad pair optimization failed unexpectedly.");
    }

    // TODO: SW-97127 - Temp WA to avoid undirected cycle in Resnet50 BWD layer.
    handleSharedBundleInputs();
}

void LayeredBrain::postExecutionHandling()
{
    // re-connect normalization undirected cycles post slicing
    {
        const bool success = m_normsHandler->handleRemovedSliceNormNodes();
        HB_ASSERT(success, "Failed to reconnect undirected normalization cycles.");
    }
}

void LayeredBrain::handleSharedBundleInputs()
{
    if (!GCFG_ENABLE_LB_DUPLICATE_SHARED_BUNDLE_INPUTS.value()) return;

    NodeSet nodes = m_graph.getNodes();  // Create a copy since the graph nodes are changing during iteration
    for (const auto& node : nodes)
    {
        if (node->getNodeType() == Node::TYPE_DEDW)
        {
            const auto& sharedBundleInput = node->getInput(1);
            HB_ASSERT_PTR(sharedBundleInput);
            const auto& in0Consumers = m_graph.getTensorConsumers(node->getInput(0));
            const auto  in0ConsumerIt =
                std::find_if(in0Consumers.begin(), in0Consumers.end(), [](const NodePtr& consumer) {
                    return consumer && Node::isDedxNode(consumer);
                });
            if (in0ConsumerIt != in0Consumers.end())  // Shared input DEDX
            {
                for (const auto& dedxConsumer : m_graph.getRealConsumers((*in0ConsumerIt)->getOutput(0)))
                {
                    const auto& consumerInputs = dedxConsumer->getInputs();
                    if (std::find(consumerInputs.begin(), consumerInputs.end(), sharedBundleInput) !=
                        consumerInputs.end())
                    {
                        LOG_DEBUG(LAYERED_BRAIN,
                                  "Duplicate tensor {} - shared by DEDW node {} and DEDX consumer {}",
                                  sharedBundleInput->getName(),
                                  node->getNodeName(),
                                  dedxConsumer->getNodeName());
                        const TensorPtr copy = sharedBundleInput->clone(false, false, false);
                        copy->setName(fmt::format("{}_copy", sharedBundleInput->getName()));
                        const NodePtr identityNode =
                            NodeFactory::createNode({sharedBundleInput},
                                                    {copy},
                                                    nullptr,
                                                    NodeFactory::identityNodeTypeName,
                                                    fmt::format("{}_copy", sharedBundleInput->getName()));
                        GraphEditor::addNode(m_graph, identityNode);
                        GraphEditor::replaceInput(m_graph, node, 1, copy);
                        break;
                    }
                }
            }
        }
    }
}

bool LayeredBrain::run()
{
    bool success = true;
    try
    {
        preExecutionHandling();
        // Deploy layered brain runner
        m_runner->run();
        postExecutionHandling();
    }
    catch (const std::exception& e)
    {
        LOG_ERR(LAYERED_BRAIN, "{} failed with exception: {}", HLLOG_FUNC, e.what());
        success = false;
    }
    return success;
}

bool runLayeredBrain(HabanaGraph& graph)
{
    if (!GCFG_ENABLE_LAYERED_PIPELINE_BRAIN.value()) return true;
    return LayeredBrain(graph).run();
}
