#pragma once

#include "habana_graph.h"

class DynamicTensorSerializer
{
public:  // methods:
    /** c'tor*/
    DynamicTensorSerializer(HabanaGraph& g) : m_g(g) {}

    void serializeDynamicTensors();

private:  // methods:
    bool shouldInsertDeserialize(const TensorPtr& tensor) const;
    bool shouldInsertSerialize(const TensorPtr& tensor) const;

    void insertDeserializeNode(const TensorPtr& persistentTensor);
    void insertSerializeNode(const TensorPtr& persistentTensor);
    void insertSerializeForPermutationTranspose(const NodePtr& transpose);
    void insertDeserializeForPermutationTranspose(const NodePtr& transpose);

    static bool isDynamicPersistent(const TensorPtr& t);
    static bool isDynamicShapePersistentTranspose(const NodePtr& node, bool checkInput);

    NodePtr createSerializeNode(const TensorPtr& deserializedTensor, const TensorPtr& persistentTensor) const;
    NodePtr createDeserializeNode(const TensorPtr& deserializedTensor, const TensorPtr& persistentTensor) const;

    HabanaGraph& m_g;
};

bool insertSerializeDeserialize(HabanaGraph& g);