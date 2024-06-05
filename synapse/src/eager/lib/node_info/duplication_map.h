#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/node_info_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// relative to <3rd-parties>/
#include "llvm/small_vector.h"

class Tensor;
class HabanaGraph;

namespace eager_mode
{
class EagerTensorsSetBuilder;

// Used in graph duplication to map original tensors to new ones.
class DuplicationMap
{
public:
    void      constructTensorMapping(const EagerTensorsSetBuilder& orgTensors,
                                     const EagerTensorsSetBuilder& otherOrgTensors);
    TensorPtr getNewTensor(Tensor* origTensor) const;
    void      clear() { m_origTensorsToCurrentMapping.clear(); }

private:
    using TensorPtrMapping = llvm_vecsmall::SmallVector<std::pair<Tensor*, TensorWeakPtr>, defaultTensorsPerGraph>;

    // m_origTensorsToCurrentMapping is required to keep track of already cloned tensors during clone
    // operation, as a given Tensor may appear several times as a node input and once as a node output.
    // This is being kept as a member to allow us to utilize it for the Synapse Duplicate API where
    // we need to return the user the original to duplicate graph tensors mapping.
    TensorPtrMapping m_origTensorsToCurrentMapping;
};

}  // namespace eager_mode