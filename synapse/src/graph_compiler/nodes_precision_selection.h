#pragma once

#include "types.h"

class HabanaGraph;

class NodesPrecisionSelection
{
public:
    NodesPrecisionSelection();
    bool        runSetNodesPrecision(HabanaGraph& g);
    bool        ignoreUserDataType(const NodePtr& node);
    synDataType getNodePrecisionFromSuccessors(HabanaGraph& g, const NodePtr& node);
    synDataType getTensorDataTypeFromConsumers(HabanaGraph& g, const TensorPtr& tensor);
    synDataType getNodePrecisionFromPredecessor(HabanaGraph& g, const NodePtr& node);
    synDataType getProfilePrecision() { return m_profilePrecision; };

    virtual synDataType getPrecisionToRaise() = 0;

protected:
    synDataType m_minNodePrecision;
    synDataType m_precisionToRaise;

private:
    synDataType m_profilePrecision;
    bool        setNodePrecision(HabanaGraph& g, const NodePtr& node, bool toRaiseLayer);
};