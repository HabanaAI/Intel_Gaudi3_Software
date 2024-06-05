#pragma once

#include <map>
#include <list>

#include "synapse_types.h"
#include "types.h"

#include <memory>

class HabanaGraph;
class Tensor;
class Node;
class TPCNode;

class CastNodeHandler
{
public:
    static NodePtr createCastNode(const TensorPtr&      inputTensor,
                                  const TensorPtr&      outputTensor,
                                  std::string_view      nodeName,
                                  tpc_lib_api::DeviceId deviceId = tpc_lib_api::DEVICE_ID_MAX);

    // Return true if cast nodes created successfully
    bool createCastNodes(const pNode& node, tpc_lib_api::DeviceId deviceId);

    // Return true if cast nodes were added
    bool plantCastNodes(HabanaGraph& g);

    void clear();

    unsigned getTotalCreatedCasts() { return m_totalCreatedCasts; }

private:
    bool _createCastNodes(const pNode& node, bool castInput, unsigned& createdCasts, tpc_lib_api::DeviceId deviceId);

    unsigned m_totalCreatedCasts;

    struct CastInfoKey
    {
        pTensor tensor;
        synDataType requiredType = syn_type_na;
        bool castInput = false;

        bool operator<(const CastInfoKey& rhs) const;
    };

    struct CastInfo
    {
        pNode castNode;
        std::list<pNode> castNodeConsumers;
        pNode castNodeProducer;
    };

    std::map<CastInfoKey, CastInfo> m_castInfoMap;
};

pTensor createCastTensor(pTensor castFrom, synDataType toType, const std::string& name);