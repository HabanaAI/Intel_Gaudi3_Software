#include "add_serialize_deserialize.h"

#include "physical_memory_ops_nodes.h"
#include "node_factory.h"
#include "transpose_node.h"

NodePtr DynamicTensorSerializer::createDeserializeNode(const TensorPtr& deserializedTensor,
                                                       const TensorPtr& persistentTensor) const
{
    std::string nodeName = fmt::format("Deserialize_{}", persistentTensor->getName());
    auto        desNode =
        NodeFactory::createNode({persistentTensor},
                                {deserializedTensor},
                                nullptr,
                                NodeFactory::getDeserializeNodeGUID(),
                                nodeName);
    return desNode;
}

NodePtr DynamicTensorSerializer::createSerializeNode(const TensorPtr& deserializedTensor,
                                                     const TensorPtr& persistentTensor) const
{
    std::string nodeName = fmt::format("Serialize_{}", persistentTensor->getName());
    auto        serNode =
        NodeFactory::createNode({deserializedTensor},
                                {persistentTensor},
                                nullptr,
                                NodeFactory::getSerializeNodeGUID(),
                                nodeName);
    return serNode;
}

bool DynamicTensorSerializer::isDynamicPersistent(const TensorPtr& t)
{
    return t->isDynamicShape() && t->isPersistent();
}

bool DynamicTensorSerializer::isDynamicShapePersistentTranspose(const NodePtr& node, bool checkInput)
{
    if (!node || node->getNodeType() != Node::TYPE_LOGICAL_TRANSPOSE) return false;

    const auto* transpose = dynamic_cast<LogicalTransposeNode*>(node.get());
    HB_ASSERT_PTR(transpose);
    if (!transpose->isUserPermutationTranspose()) return false;
    if (transpose->getAliasDirection() == (checkInput ? OUTPUT_TO_INPUT : INPUT_TO_OUTPUT))
    {
        const TensorPtr& t = checkInput ? transpose->getInput(0) : transpose->getOutput(0);
        HB_ASSERT_PTR(t);
        return isDynamicPersistent(t);
    }

    return false;
}

bool DynamicTensorSerializer::shouldInsertDeserialize(const TensorPtr& tensor) const
{
    if (tensor == nullptr) return false;
    if (tensor->isShapeTensor()) return false;
    if (tensor->isZeroSizedDataTensor()) return false;

    // We only need to insert deserialize when the tensor is serialized (dense) but shouldn't be.
    // This happens when the tensor is dynamic (The actual size may be smaller then the max), and is a persistent
    // input to the graph (The frameworks always works with dense tensors).
    if (!isDynamicPersistent(tensor)) return false;

    if (m_g.getNumberOfTensorConsumers(tensor) == 0) return false;
    return true;
}

bool DynamicTensorSerializer::shouldInsertSerialize(const TensorPtr& tensor) const
{
    if (tensor == nullptr) return false;
    if (tensor->isShapeTensor()) return false;
    if (tensor->isZeroSizedDataTensor()) return false;

    // We only need to insert serialize when the user expect dense layout, and internally we have strided layout.
    // This happens when the tensor is dynamic, and is a persistent output (or intermediate) to the graph
    // (The frameworks always works with dense tensors).
    // No SCD optimization for serialize to prevent corrupting internal tensors when working with PyTorch
    // because it allocates persistent memory based on actual size and not on max size - i.e. we always have to
    // serialize.
    if (!isDynamicPersistent(tensor)) return false;

    NodePtr producer = m_g.getTensorProducer(tensor);
    if (!producer) return false;

    return true;
}

/* logical transpose must be connected directly to the persistent (permuted) tensor.
    in this case we need to serialize the input of that transpose.
    (producer) -> [T] -> (logicalTranspose) -> [persistent]
    should turn into:
    (producer) -> [T] -> (serialize) -> [..] -> (logicalTranspose) -> [persistent]. */
void DynamicTensorSerializer::insertSerializeForPermutationTranspose(const NodePtr& transpose)
{
    const TensorPtr& tensor        = transpose->getInput(0);
    TensorPtr        copy          = tensor->clone(false, false);
    NodePtr          serializeNode = createSerializeNode(tensor, copy);
    GraphEditor::replaceTensor(m_g, transpose, tensor, copy);
    GraphEditor::addNode(m_g, serializeNode);
}

/* logical transpose must be connected directly to the persistent (permuted) tensor.
    in this case we need to deserialize the output of that transpose.
    ->[persistent]-> (logicalTranspose) -> [T] -> (consumer)
    should turn into:
    ->[persistent]-> (logicalTranspose) -> [] -> (deserialize) -> [T] -> (consumer). */
void DynamicTensorSerializer::insertDeserializeForPermutationTranspose(const NodePtr& transpose)
{
    const TensorPtr& tensor          = transpose->getOutput(0);
    TensorPtr        copy            = tensor->clone(false, false);
    NodePtr          deserializeNode = createDeserializeNode(tensor, copy);
    GraphEditor::replaceTensor(m_g, transpose, tensor, copy);
    GraphEditor::addNode(m_g, deserializeNode);
}

void DynamicTensorSerializer::insertDeserializeNode(const TensorPtr& tensor)
{
    HB_ASSERT_PTR(tensor);
    for (const NodePtr& consumer : m_g.getTensorConsumers(tensor))
    {
        if (isDynamicShapePersistentTranspose(consumer, /* checkInput */ true))
        {
            HB_ASSERT(m_g.getNumberOfTensorConsumers(tensor),
                      "expecting a single logical transpose for deserializing tensor {}!",
                      tensor->getName());
            insertDeserializeForPermutationTranspose(consumer);
            return;
        }
    }

    TensorPtr deserializedTensor = tensor->clone(false, false);
    NodePtr   deserializeNode    = createDeserializeNode(deserializedTensor, tensor);
    for (const NodePtr& consumer : m_g.getTensorConsumers(tensor))
    {
        GraphEditor::replaceTensor(m_g, consumer, tensor, deserializedTensor);
    }

    bool status = GraphEditor::addNode(m_g, deserializeNode);
    HB_ASSERT(status == true, "failed inserting deserialize node");
}

void DynamicTensorSerializer::insertSerializeNode(const TensorPtr& tensor)
{
    HB_ASSERT_PTR(tensor);
    NodePtr producer = m_g.getTensorProducer(tensor);
    HB_ASSERT_PTR(producer);

    if (isDynamicShapePersistentTranspose(producer, /* checkInput */ false))
    {
        insertSerializeForPermutationTranspose(producer);
        return;
    }

    TensorPtr deserializedTensor = tensor->clone(false, false);
    NodePtr   serializeNode      = createSerializeNode(deserializedTensor, tensor);

    // shortcut the consumers of the dynamic tensor, so that they will consume deserialized tensor
    for (const NodePtr& consumer : m_g.getTensorConsumers(tensor))
    {
        GraphEditor::replaceTensor(m_g, consumer, tensor, deserializedTensor);
    }

    GraphEditor::replaceTensor(m_g, producer, tensor, deserializedTensor);

    bool status = GraphEditor::addNode(m_g, serializeNode);
    HB_ASSERT(status == true, "failed inserting serialize node");
}

void DynamicTensorSerializer::serializeDynamicTensors()
{
    TensorVector dynamicTensors;
    const auto&  allTensors = m_g.getTensors();
    std::copy_if(allTensors.begin(), allTensors.end(), std::back_inserter(dynamicTensors), [this](const TensorPtr& t) {
        return this->shouldInsertSerialize(t) || this->shouldInsertDeserialize(t);
    });

    for (const TensorPtr& t : dynamicTensors)
    {
        if (shouldInsertSerialize(t))
        {
            insertSerializeNode(t);
        }
        if (shouldInsertDeserialize(t))
        {
            insertDeserializeNode(t);
        }
    }
}

bool insertSerializeDeserialize(HabanaGraph& g)
{
    if (!g.isDynamicShape())
    {
        return true;
    }

    if (!GCFG_GAUDI_ENABLE_SERIALIZE_DESERIALIZE_PASS.value())
    {
        LOG_DEBUG(GC, "DSD : Skipping serialize/deserialize pass. Feature disabled by configuration");
        return true;
    }

    DynamicTensorSerializer(g).serializeDynamicTensors();
    return true;
}
