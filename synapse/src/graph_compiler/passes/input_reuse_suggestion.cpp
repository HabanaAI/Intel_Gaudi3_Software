#include "input_reuse_suggestion.h"
#include "habana_graph.h"

// In case the input is aliased, it is possible to apply inplace reuse as long as the actual memory that will be written
// by the node is not read after writing it (otherwise the readers will read corrupt memory).
// The method of checking this condition is by going over the alias chain, and if a consumer of the alias tensor doesn't
// have a path to the current node, then reuse is not allowed (no path -> runs after current node).
bool InputInplaceReuseSuggestion::isMemoryConsumedAfterNode(const HabanaGraph& g,
                                                            const NodePtr&     node,
                                                            const TensorPtr&   input) const
{
    TensorPtr tensorInAliasChain = input;
    while (tensorInAliasChain)
    {
        const auto& consumers                 = g.getTensorConsumers(tensorInAliasChain);
        // memoryIsConsumedAfterNode is a time-consuming check, so this condition will be checked last
        bool memoryIsConsumedAfterNode = g.isUserManagedDram(tensorInAliasChain) ||
                                         tensorInAliasChain->isStaticParam() ||
                                         std::any_of(consumers.begin(), consumers.end(), [&](const NodePtr& consumer) {
                                             return g.getNumberOfPaths(consumer, node) == 0;
                                         });
        if (memoryIsConsumedAfterNode)
        {
            return true;
        }
        tensorInAliasChain = tensorInAliasChain->getAliasTensor();
    }
    return false;
}

bool InputInplaceReuseSuggestion::outputMemcopyRequired(const HabanaGraph& g,
                                                        const NodePtr&     node,
                                                        const TensorPtr&   output) const
{
    if (!output->isDenseLayout()) return true;
    if (output->isAliasedTensor()) return true;
    if (g.isUserManagedDram(output)) return true;
    return false;
}

bool InputInplaceReuseSuggestion::inputMemcopyRequired(const HabanaGraph& g,
                                                       const NodePtr&     node,
                                                       const TensorPtr&   input,
                                                       const TensorPtr&   output) const
{
    if (!input->isDenseLayout()) return true;
    if (g.isUserManagedDram(input)) return true;
    if (input->location() != output->location()) return true;
    if (isMemoryConsumedAfterNode(g, node, input)) return true;
    return false;
}

// In case input/output are multibuffered, the reuse can change the liveness of the input so that it overlaps all the
// other tensors in the multibuffer, which will cause a failure when allocating the multibuffer.
// Since *currently* most of the multibuffered tensors are bundle-tensors, and the memory consumption was already calculated,
// this optimization is not mandatory so we can block those tensors.
bool InputInplaceReuseSuggestion::areTensorsMultibuffered(const TensorPtr& input, const TensorPtr& output) const
{
    if (input->getTensorAnnotation().nonPersistentSectionInfo.sectionId.is_set()) return true;
    if (output->getTensorAnnotation().nonPersistentSectionInfo.sectionId.is_set()) return true;
    return false;
}

ReusePairsMap InputInplaceReuseSuggestion::getReusePairs(const NodePtr& node)
{
    return node->getReusableInputs();
}

bool InputInplaceReuseSuggestion::isDiscarded(const HabanaGraph& g, const TensorPtr& t) const
{
    auto getWrittenAlias = [&](const TensorPtr& tensor) {
        // getRealTensor always provides the end of the aliasing chain, but due to in-placing (input reuse) the tensor
        // may be written several times throughout the chain.
        // This loop finds the real tensor or the first alias that is written to by a physical operation.
        TensorPtr written = tensor;
        while (written->isAliasedTensor())
        {
            const auto& producer = g.getTensorProducer(written);
            if (producer == nullptr) break;
            if (!producer->isLogicalOperation()) break;
            written = written->getAliasTensor();
        }
        return written;
    };

    TensorPtr tWritten = getWrittenAlias(t);
    for (const NodePtr& realConsumer : g.getRealConsumers(tWritten))
    {
        if (realConsumer->getNodeAnnotation().inputsCacheMetaData.empty())
            continue;  // No cache metadata => no discards

        for (size_t inIdx = 0; inIdx < realConsumer->getNumInputs(); inIdx++)
        {
            const TensorPtr& consumerInput = realConsumer->getInput(inIdx);
            if (!consumerInput) continue;
            // output tensor is not an alias, or else input reuse is not possible
            if (getWrittenAlias(consumerInput) == tWritten &&
                realConsumer->getNodeAnnotation().inputsCacheMetaData.at(inIdx).cmAction ==
                    CacheMaintenanceAction::DISCARD)
            {
                return true;
            }
        }
    }
    return false;
}

bool InputInplaceReuseSuggestion::outputViableForInplace(const HabanaGraph& g,
                                                         const NodePtr&     node,
                                                         const TensorPtr&   nodeOutput) const
{
    if (outputMemcopyRequired(g, node, nodeOutput))
    {
        LOG_DEBUG(GC,
                  "Output in-place suggestion not viable: node = {}, output = {}. Output requires memcpy.",
                  node->getNodeName(),
                  nodeOutput->getName());
        return false;
    }
    if (isDiscarded(g, nodeOutput))
    {
        LOG_DEBUG(GC,
                  "Output in-place suggestion not viable: node = {}, output = {}. Output is discarded.",
                  node->getNodeName(),
                  nodeOutput->getName());
        return false;
    }
    return true;
}

bool InputInplaceReuseSuggestion::viableInputCandidate(const HabanaGraph& g,
                                                       const NodePtr&     node,
                                                       const TensorPtr&   input,
                                                       const TensorPtr&   output) const
{
    if (isAlreadyReused(g, input, *node))
    {
        LOG_DEBUG(GC,
                  "Input reuse suggestion not viable: node = {}, output = {}, input = {}. Input is already reused.",
                  node->getNodeName(),
                  output->getName(),
                  input->getName());
        return false;
    }

    if (areTensorsMultibuffered(input, output))
    {
        LOG_DEBUG(GC,
                  "Input reuse suggestion not viable: node = {}, output = {}, input = {}. Tensors are multibuffered.",
                  node->getNodeName(),
                  output->getName(),
                  input->getName());
        return false;
    }

    if (isDiscarded(g, input))
    {
        LOG_DEBUG(GC,
                  "Input reuse suggestion not viable: node = {}, output = {}, input = {}. Input is discarded.",
                  node->getNodeName(),
                  output->getName(),
                  input->getName());
        return false;
    }

    // This check is expensive. Should be performed last.
    if (inputMemcopyRequired(g, node, input, output))
    {
        LOG_DEBUG(GC,
                  "Input reuse suggestion not viable: node = {}, output = {}, input = {}. Requires memcpy.",
                  node->getNodeName(),
                  output->getName(),
                  input->getName());
        return false;
    }

    return true;
}

bool InputInplaceReuseSuggestion::applyReuse(HabanaGraph&        g,
                                             const NodePtr&      node,
                                             const TensorPtr&    nodeOutput,
                                             const TensorVector& reuseCandidates)
{
    for (const TensorPtr& t : reuseCandidates)
    {
        if (!viableInputCandidate(g, node, t, nodeOutput)) continue;

        LOG_INFO(GC,
                 "Apply inplace reuse for suggestion: node = {}, set tensor {} as aliased by tensor {}",
                 node->getNodeName(),
                 t->getName(),
                 nodeOutput->getName());

        // Reaching this point implies gc will reuse one of the TPC node input tensors
        // to which the output will be written, hence we set the output tensor as an alias of the input tensor.
        nodeOutput->setAsAliasSubTensor(t);
        t->setIsRealInLogical(true);
        break;
    }
    return true;
}

bool inPlaceInputReuseSuggestion(HabanaGraph& g)
{
    if (!GCFG_ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS.value())
    {
        return true;
    }
    InputInplaceReuseSuggestion inplaceReuse;
    return inplaceReuse.runInputInplaceReuse(g);
}
