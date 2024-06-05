#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/node_info_defs.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerTensorsSet
///////////////////////////////////////////////////////////////////////////////////////////////////

// Tensor-related information that is available at early stage when nodes are added
class EagerTensorsSet
{
public:
    EagerTensorsSet() = default;
    explicit EagerTensorsSet(const EagerTensorsSet& other);
    virtual ~EagerTensorsSet() = default;

    const VecTensors<TensorPtr>& getTensors() const { return m_tensors; }
    bool                         isGraphInput(size_t idx) const { return checkAndEcho(m_isGraphInput[idx]); }
    size_t                       getPersistentNr() const { return checkAndEcho(m_persistentNr); }
    size_t                       getNamesSizeOfPersistentTensors() const { return m_persistentTensorNamesSize; }

private:
    template<class T>
    const T& checkAndEcho(const T& val) const;

protected:
    bool                  m_isAddingNewTensorEnabled = true;   // Allow adding new tensors
    VecTensors<TensorPtr> m_tensors;                           // Pointers to all unique tensors that are not NULL
    VecTensors<bool>      m_isGraphInput;                      // Is a tensor input or output of the graph
    size_t                m_persistentNr              = 0;     // Number of persistent tensors
    size_t                m_persistentTensorNamesSize = 0;     // Size of all names strings (used for recipe allocation)
    bool                  m_areAllTensorsSupported    = true;  // optional, marking the fallback reason
};

// Util to do some checks and return same parameter it receives
template<class T>
const T& EagerTensorsSet::checkAndEcho(const T& val) const
{
    // Note that both m_tensors and m_isGraphInput might be empty in cae of ZST
    EAGER_ASSERT(!m_isAddingNewTensorEnabled, "Tensors set is not ready");
    return val;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerTensorsSetBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////

class EagerNode;
class EagerNodes;

class EagerTensorsSetBuilder final : public EagerTensorsSet
{
public:
    explicit EagerTensorsSetBuilder(uint64_t tensorSizeThresholdForParallelExecution);
    explicit EagerTensorsSetBuilder(const EagerTensorsSetBuilder& other);
    void disableAddingNewTensors(bool isOriginalGraph);
    void addTensors(const EagerNode& node);
    bool addTensorsWithParallelExecCheck(const EagerNode& node);
    bool checkForTensorExceedParallelExecutionThreshold(const TensorVector& tensors);
    bool   shouldUtilizeParallelExecution() const { return m_hasTensorSizeExceedsParallelExecutionThreshold; }
    bool   allowParallelExecHandling() const { return m_tensorSizeThresholdForParallelExecution != -1; }
    size_t getIndex(const TensorPtr& tensor) const;
    void   setGraphInputs(const EagerNodes& nodes);
    bool   isRoot(const EagerNode& node) const;

    bool areAllTensorsSupported() const { return m_areAllTensorsSupported; }

private:
    void                 makeSortedSet(bool isOriginalGraph);
    static const Tensor* getRealTensor(const TensorPtr& tensor);

private:
    // This threshold used to check if there is any tensor size that exceeds it
    const uint64_t m_tensorSizeThresholdForParallelExecution;
    // Flag to track if there is at least one tensor with size exceeds the threshold that is determined by
    // the per-chip TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION
    bool m_hasTensorSizeExceedsParallelExecutionThreshold = false;
};

}  // namespace eager_mode