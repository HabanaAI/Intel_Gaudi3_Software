#include "duplication_map.h"

// eager includes (relative to src/eager/lib/)
#include "node_info/tensor_info.h"

// synapse-internal includes (relative to src/)
#include "include/tensor.h"

namespace eager_mode
{
void DuplicationMap::constructTensorMapping(const EagerTensorsSetBuilder& orgTensors,
                                            const EagerTensorsSetBuilder& otherOrgTensors)
{
    const auto& dstGraphTensorsVec = orgTensors.getTensors();
    const auto& srcGraphTensorsVec = otherOrgTensors.getTensors();

    HB_ASSERT(dstGraphTensorsVec.size() == srcGraphTensorsVec.size(),
              "original eager graph and duplicate eager graph have a different tensor count");
    m_origTensorsToCurrentMapping.reserve(dstGraphTensorsVec.size());
    for (int i = 0; i < dstGraphTensorsVec.size(); i++)
    {
        m_origTensorsToCurrentMapping.emplace_back(srcGraphTensorsVec[i].get(), dstGraphTensorsVec[i]);
    }
}

TensorPtr DuplicationMap::getNewTensor(Tensor* origTensor) const
{
    auto newTensorIter =
        std::find_if(m_origTensorsToCurrentMapping.begin(),
                     m_origTensorsToCurrentMapping.end(),
                     [&origTensor](const auto& tensorMapping) { return tensorMapping.first == origTensor; });
    if (newTensorIter != m_origTensorsToCurrentMapping.end())
    {
        return newTensorIter->second.lock();
    }
    return {};
}

}  // namespace eager_mode