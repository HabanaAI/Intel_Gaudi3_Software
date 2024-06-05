#include "bundle_memory_preprocessor.h"

#include "tensor_view_node.h"

using namespace gc::layered_brain;

POCBundleMemoryPreProcessor::POCBundleMemoryPreProcessor(HabanaGraph& graph, const BundleNodes& bundle)
: m_graph(graph), m_nodes(bundle)
{
}

MemoryUsageDB POCBundleMemoryPreProcessor::buildMemUsageDB()
{
    for (size_t step = 0; step < m_nodes.size(); step++)
    {
        addStep(step);
    }
    processExternalInteractions();
    processAliases();
    logDB();
    return m_db;
}

void POCBundleMemoryPreProcessor::addStep(size_t step)
{
    addStepEntry(step);
    updateSliceEntries(step);
}

void POCBundleMemoryPreProcessor::addStepEntry(size_t step)
{
    const NodePtr& node = m_nodes[step];
    m_db.steps.push_back(MemoryUsageDB::BundleStepEntry {step, node});
}

// Adds slice information of the current step
void POCBundleMemoryPreProcessor::updateSliceEntries(size_t step)
{
    const NodePtr& node = m_nodes[step];
    for (const TensorPtr& input : node->getInputs())
    {
        processInput(step, input);
    }
    for (const TensorPtr& output : node->getOutputs())
    {
        processOutput(step, output);
    }
}

// Add consuming information to the slice
void POCBundleMemoryPreProcessor::processInput(size_t step, const TensorPtr& input)
{
    if (!input) return;
    if (isJoin(m_nodes[step]))  // join is not considered an internal consumer
    {
        HB_ASSERT(properties(input).joinedBy == nullptr,
                  "Second join node (first: {}, second: {}) found for slice tensor {}",
                  properties(input).joinedBy->getNodeName(),
                  m_nodes[step]->getNodeName(),
                  input->getName());

        properties(input).joinedBy = m_nodes[step];
    }
    else
    {
        properties(input).consumingSteps.insert(step);
        properties(input).immediateConsumingSteps.insert(step);
    }
}

// Add producing information to the slice.
void POCBundleMemoryPreProcessor::processOutput(size_t step, const TensorPtr& output)
{
    if (!output) return;
    if (isFork(m_nodes[step]))  // fork is not considered an internal producer
    {
        // No need to check for previous fork, since it will imply multiple producers
        properties(output).forkedBy = m_nodes[step];
    }
    else
    {
        properties(output).producingStep = step;
    }
}

// Adds external consumer/producer information to each slice. Separate pass in order not to re-check each slice both as
// an output and as an input of each of its internal consumers.
void POCBundleMemoryPreProcessor::processExternalInteractions()
{
    for (auto& sliceAndEntry : m_db.slices)
    {
        processExternalInteractions(sliceAndEntry.first);
    }
}

void POCBundleMemoryPreProcessor::processExternalInteractions(const TensorPtr& slice)
{
    const NodePtr& producer = m_graph.getTensorProducer(slice);
    if (producer && !isInBundle(producer))
    {
        processExternalProducer(slice, producer);
    }
    for (const NodePtr& consumer : m_graph.getTensorConsumers(slice))
    {
        if (!isInBundle(consumer))
        {
            processExternalConsumer(slice, consumer);
        }
    }
}

void POCBundleMemoryPreProcessor::processExternalProducer(const TensorPtr& slice, const NodePtr& producer)
{
    if (isFork(producer))
    {
        // If the fork is external to the bundle, it is still important to mark it for the alias
        // processing
        properties(slice).forkedBy = producer;
    }
}

void POCBundleMemoryPreProcessor::processExternalConsumer(const TensorPtr& slice, const NodePtr& consumer)
{
    properties(slice).consumedExternally = true;
    if (isJoin(consumer))
    {
        // If the join is external to the bundle, it is still important to mark it for the alias
        // processing.
        properties(slice).joinedBy = consumer;
    }
}

// Aliases create a forst (group of trees) which are not topologically sorted in the graph. E.g
//             +-------alias-------+
//             |                   v
// [t0]->n1->[t1]->n2->[t2]->n3->[t3]->n4->[t4]
//   |        ^^         |
//   +-alias--++--alias--+
//
// t3 is the root, it has a single offspring t1, which in turn has 2 offsprings: t0 and t2.
//
// Process aliases does a sort of DFS scan of the alias tree to aggregate consumers.
void POCBundleMemoryPreProcessor::processAliases()
{
    InboundAliasesPerSlice aliasForest = buildAliasForest();
    logAliasForest(aliasForest);
    while (!aliasForest.empty())
    {
        // If the children of the "root" are not leaves, they will be aggregated recursively and taken out of their
        // tree. If the "root" has a parent, it will be processed in a different iteration. The joins are saved
        // for each processed slice, making this a DFS using dynamic programming without finding actual roots that don't
        // have parents.
        Slice root = aliasForest.begin()->first;
        aggregateConsumersAndAliases(root, aliasForest);
        aliasForest.erase(root);
    }

    // At this point, the aliases list is full for all 'real' slices (slices that are not aliases of other intermediate
    // slices), but the 'realSlice' field still needs setting. The alias slices will appear in exactly one list, so this
    // loop actually sets each realSlice field of alias slices exactly once and doesn't touch 'real' slices.
    for (const auto& sliceAndEntry : m_db.slices)
    {
        for (const auto& alias : sliceAndEntry.second.properties.aliases)
        {
            HB_ASSERT(!properties(alias).realSlice, "Slice appears to be an alias to 2 different 'real' slices");
            properties(alias).realSlice = sliceAndEntry.first;
        }
    }
}

// Collect all the inbound aliases for each slice (which has them). Inbound aliases == all the tensors which are an
// alias of the processed slice.
POCBundleMemoryPreProcessor::InboundAliasesPerSlice POCBundleMemoryPreProcessor::buildAliasForest()
{
    InboundAliasesPerSlice forest;
    for (const auto& node : m_nodes)
    {
        // Since only intermediate slices are interesting, it's enough to go over the outputs. Inputs that are
        // interesting will be process when they are outputs.
        for (const auto& slice : node->getOutputs())
        {
            if (slice->isAliasedTensor())
            {
                const TensorPtr& target = slice->getAliasTensor();
                if (isIntermediate(slice) && isIntermediate(target))
                {
                    // only interested in aliases between intermediate slices
                    forest[target].push_back(slice);
                }
            }
        }
    }
    return forest;
}

// recursively aggregates the consumers and aliases of real slices.
void POCBundleMemoryPreProcessor::aggregateConsumersAndAliases(const Slice& slice, InboundAliasesPerSlice& aliasTrees)
{
    if (aliasTrees.find(slice) == aliasTrees.end()) return;  // No un-processed inbound aliases
    for (const Slice& alias : aliasTrees[slice])
    {
        aggregateConsumersAndAliases(alias, aliasTrees);

        accumulateConsumers(alias, slice);
        accumulateAndMoveAliases(alias, slice);

        aliasTrees.erase(alias);
    }
}

// Adds all the inbound aliases consumers to the given slice consumers
void POCBundleMemoryPreProcessor::accumulateConsumers(const Slice& alias, const Slice& targetSlice)
{
    auto& targetConsumers = properties(targetSlice).consumingSteps;
    auto& aliasConsumers  = properties(alias).consumingSteps;
    targetConsumers.insert(aliasConsumers.begin(), aliasConsumers.end());
}

// Moves all the inbound aliases to the given slice aliases (recursive)
void POCBundleMemoryPreProcessor::accumulateAndMoveAliases(const Slice& alias, const Slice& targetSlice)
{
    auto& tsAliases = properties(targetSlice).aliases;
    tsAliases.push_back(alias);
    tsAliases.splice(tsAliases.end(), properties(alias).aliases);
}

bool POCBundleMemoryPreProcessor::isJoin(const NodePtr& node) const
{
    switch (node->getNodeType())
    {
        case Node::TYPE_INTERNAL_CONCAT:
            return true;
        case Node::TYPE_TENSOR_VIEW:
            return !static_cast<TensorViewNode*>(node.get())->realTensorIsInput();
        default:
            return false;
    }
}

bool POCBundleMemoryPreProcessor::isFork(const NodePtr& node) const
{
    switch (node->getNodeType())
    {
        case Node::TYPE_INTERNAL_SPLIT:
        case Node::TYPE_SPLIT_SHAPE:
            return true;
        case Node::TYPE_TENSOR_VIEW:
            return static_cast<TensorViewNode*>(node.get())->realTensorIsInput();
        default:
            return false;
    }
}

// Is the node part of the currently handled bundle
bool POCBundleMemoryPreProcessor::isInBundle(const NodePtr& node) const
{
    return std::find(m_nodes.begin(), m_nodes.end(), node) != m_nodes.end();
}

bool POCBundleMemoryPreProcessor::isIntermediate(const Slice& slice) const
{
    if (m_db.slices.count(slice) == 0) return false;
    const auto& props = properties(slice);
    // Intermediate if it's a slice and has a producing step and some consuming steps (for this definition, a fork is
    // considered a producer and a join a consumer)
    return (!props.consumingSteps.empty() || props.joinedBy) && (props.producingStep || props.forkedBy);
}

// Printing
void POCBundleMemoryPreProcessor::logDB() const
{
    if (!log_level_at_least(synapse::LogManager::LogType::LB_CACHE_MNGR, 1)) return;

    LOG_DEBUG(LB_CACHE_MNGR, "Memory Usage DB:");
    int stepIdx = 0;
    for (const auto& step : m_db.steps)
    {
        LOG_DEBUG(LB_CACHE_MNGR, "Step {}", stepIdx++);
        LOG_DEBUG(LB_CACHE_MNGR, "  Node: {} [{}]", step.sliceNode->getNodeName(), step.sliceNode->getNodeTypeStr());
        LOG_DEBUG(LB_CACHE_MNGR, "  Input Entries:");
        for (int i = 0; i < step.sliceNode->getNumInputs(); i++)
        {
            logSlice(step.sliceNode->getInput(i), i);
        }
        LOG_DEBUG(LB_CACHE_MNGR, "  Output Entries:");
        for (int i = 0; i < step.sliceNode->getNumOutputs(); i++)
        {
            logSlice(step.sliceNode->getOutput(i), i);
        }
        LOG_DEBUG(LB_CACHE_MNGR, "");
    }
}
void POCBundleMemoryPreProcessor::logSlice(const TensorPtr& t, int index) const
{
    if (!t)
    {
        LOG_DEBUG(LB_CACHE_MNGR, "    [{}] null", index);
        return;
    }

    const auto& sliceProp = properties(t);
    LOG_DEBUG(
        LB_CACHE_MNGR,
        "    [{:2}] name: {}, size: {}, producing step: {}, consuming steps: {}, consumed externally: {}, joined: {}, "
        "forked: {}, #aliases: {}, real: {}",
        index,
        t->getName(),
        t->getDimSizesStr(),
        sliceProp.producingStep ? *sliceProp.producingStep : -1,
        toString(sliceProp.consumingSteps, ','),
        sliceProp.consumedExternally,
        bool(sliceProp.joinedBy),
        bool(sliceProp.forkedBy),
        sliceProp.aliases.size(),
        sliceProp.realSlice ? sliceProp.realSlice->getName() : "nullptr");
}

void POCBundleMemoryPreProcessor::logAliasForest(const InboundAliasesPerSlice& forest)
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(LB_CACHE_MNGR)) return;
    LOG_TRACE(LB_CACHE_MNGR, "Alias clusters:{}", forest.empty() ? " empty" : "");
    for (const auto& [slice, aliases] : forest)
    {
        if (aliases.empty()) continue;
        LOG_TRACE(LB_CACHE_MNGR, "  Slice: {}, aliases:", slice->getName());
        for (const auto& alias : aliases) LOG_TRACE(LB_CACHE_MNGR, "    {}", alias->getName());
    }
}