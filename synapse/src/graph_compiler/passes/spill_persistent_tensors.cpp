#include "defs.h"
#include "graph.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "handle_memory_reuse.h"
#include "node_factory.h"
#include "register_memory_coherence.h"
#include "identity_node.h"

void spillTensor(HabanaGraph& g, const TensorPtr& spilledTensor, bool keepSpilledTensor = true)
{
    // Do not spill user outputs
    if (keepSpilledTensor && g.getNumberOfTensorConsumers(spilledTensor) == 0)
    {
        LOG_TRACE(GC, "Skipping on spilling tensor: {} - tensor is user output.", spilledTensor->getName());
        return;
    }

    TensorPtr spilledTensorReplace = spilledTensor->clone();
    spilledTensorReplace->setName(spilledTensor->getName() + "_replaced");

    GraphEditor::replaceTensor(g, spilledTensor, spilledTensorReplace);

    if (keepSpilledTensor)
    {
        NodePtr identityNode = NodeFactory::createNode({spilledTensorReplace},
                                                       {spilledTensor},
                                                       nullptr,
                                                       NodeFactory::identityNodeTypeName,
                                                       spilledTensor->getName() + "_planted_identity");

        // To avoid optimizations that mark these identity nodes as redundant nodes and remove them
        std::static_pointer_cast<IdentityNode>(identityNode)->markPersistent();

        GraphEditor::addNode(g, identityNode);
        LOG_DEBUG(GC,
                  "Planting identity node:{}. Replace tensor: {}, Spilled tensor: {}",
                  identityNode->getNodeName(),
                  spilledTensorReplace->getName(),
                  spilledTensor->getName());
    }
};

bool isTensorPartOfReuseableBinding(const HabanaGraph& g, const TensorPtr& spilledTensorCandidate)
{
    const NodePtr& tensorProducer          = g.getTensorProducer(spilledTensorCandidate);
    bool           producerHasBindingReuse = tensorProducer->hasBindingInputReuse();

    const NodeList& tensorConsumers = g.getTensorConsumers(spilledTensorCandidate);
    bool consumersHaveBindingReuse  = std::any_of(tensorConsumers.begin(), tensorConsumers.end(), [](const NodePtr& n) {
        return n->hasBindingInputReuse();
    });

    return producerHasBindingReuse || consumersHaveBindingReuse;
}

bool shouldSpillTensor(const HabanaGraph&  g,
                       const TensorPtr&    spilledTensorCandidate,
                       const TensorVector& nextCoherencyTensors)
{
    // Do not spill tensors if they don't overlap with following persistent tensors
    if (nextCoherencyTensors.empty())
    {
        LOG_TRACE(GC,
                  "Skipping on spilling tensor: {}, tensor does not overlap with following persistent tensor",
                  spilledTensorCandidate->getName());
        return false;
    }

    // Do not spill user inputs
    if (g.getNumberOfTensorProducers(spilledTensorCandidate) == 0)
    {
        LOG_TRACE(GC, "Skipping on spilling tensor: {} - tensor is user input.", spilledTensorCandidate->getName());
        return false;
    }

    // Do not spill tensors that are part of reusable binding. Doing so for outputs of reusable bindings will force a
    // memcpy, because a tensor can't be an alias of two different tensors.
    // Doing so for inputs of reusable bindings can cause an illegal aliasing from persistent to intermediate tensor.
    if (isTensorPartOfReuseableBinding(g, spilledTensorCandidate))
    {
        LOG_TRACE(GC,
                  "Skipping on spilling tensor: {} - tensor is part of reusable binding.",
                  spilledTensorCandidate->getName());
        return false;
    }

    return true;
}

bool spillPersistentTensors(HabanaGraph& g)
{
    if (!GCFG_SPILL_PERSISTENT_TENSORS.value())
    {
        LOG_TRACE(GC, "Skipping spillPersistentTensors pass");
        return true;
    }

    const auto& memoryCoherence = g.getGraphAnnotation().memoryCoherence;
    HB_ASSERT(memoryCoherence != nullptr, "Graph doesn't have a tensor coherence mapping");

    const auto& consistentMemorySections =
        memoryCoherence->getAllSectionsTensorCoherence(TensorCoherenceMapping::SectionType::USER_ALLOCATED_SECTIONS);

    // always keep spilled tensors in this case
    bool doReadAfterWriteDependenciesExist = memoryCoherence->doReadAfterWriteExternalDependenciesExist();

    // Outer and Inside loops iterate over all the persistent tensors in the graph
    for (const auto& consistentMemorySection : consistentMemorySections)
    {
        const TensorVector& persistentTensors = consistentMemorySection.second;
        for (const TensorPtr& persistentTensor : persistentTensors)
        {
            const TensorVector& nextCoherencyTensors = memoryCoherence->findNextCoherencyTensors(persistentTensor);
            if (shouldSpillTensor(g, persistentTensor, nextCoherencyTensors))
            {
                // If there is a full overlap with a following persistent tensor - no need to keep it in the graph
                bool noExactOverlap          = std::none_of(nextCoherencyTensors.begin(),
                                                   nextCoherencyTensors.end(),
                                                   [&persistentTensor](const TensorPtr& t) {
                                                       return MemoryReuseHandler::isExactOverlap(persistentTensor, t);
                                                   });
                bool shouldKeepSpilledTensor = noExactOverlap || doReadAfterWriteDependenciesExist;
                spillTensor(g, persistentTensor, shouldKeepSpilledTensor);
            }
        }
    }
    // turning predicate to run again registerMemoryCoherence pass as persistent tensors removed.
    g.turnOnPredicate(PREDICATE_ID_MEMORY_SECTION_TENSOR_CREATED);

    return true;
}