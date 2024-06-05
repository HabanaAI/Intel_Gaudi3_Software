#include "tensor_info.h"

// eager includes (relative to src/eager/lib/)
#include "node_info/eager_node.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"
#include "include/tensor.h"

// std includes
#include <algorithm>

namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerTensorsSet
///////////////////////////////////////////////////////////////////////////////////////////////////

EagerTensorsSet::EagerTensorsSet(const EagerTensorsSet& other)
: m_isAddingNewTensorEnabled(other.m_isAddingNewTensorEnabled),
  m_isGraphInput(other.m_isGraphInput),
  m_persistentNr(other.m_persistentNr),
  m_persistentTensorNamesSize(other.m_persistentTensorNamesSize),
  m_areAllTensorsSupported(other.m_areAllTensorsSupported)
{
    m_tensors.reserve(other.m_tensors.size());
    for (const auto& origTensor : other.m_tensors)
    {
        m_tensors.emplace_back(origTensor->clone(false /*copyAddress*/,
                                                 true /*copyData*/,
                                                 true /*keepPersistent*/,
                                                 TensorNameClonePolicy::COPY_NAME));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// EagerTensorsSetBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////

EagerTensorsSetBuilder::EagerTensorsSetBuilder(uint64_t tensorSizeThresholdForParallelExecution)
: m_tensorSizeThresholdForParallelExecution(tensorSizeThresholdForParallelExecution)
{
}

EagerTensorsSetBuilder::EagerTensorsSetBuilder(const EagerTensorsSetBuilder& other)
: EagerTensorsSet(other),
  m_tensorSizeThresholdForParallelExecution(other.m_tensorSizeThresholdForParallelExecution),
  m_hasTensorSizeExceedsParallelExecutionThreshold(other.m_hasTensorSizeExceedsParallelExecutionThreshold)
{
}

void EagerTensorsSetBuilder::disableAddingNewTensors(bool isOriginalGraph)
{
    EAGER_ASSERT(m_isAddingNewTensorEnabled, "Wrong flow");
    m_isAddingNewTensorEnabled = false;
    // Note: m_tensors.empty() is possible in case of ZST

    makeSortedSet(isOriginalGraph);
}

// Sort tensors vector and exclude duplicates
void EagerTensorsSetBuilder::makeSortedSet(bool isOriginalGraph)
{
    // Sort the vector and keep one pointer for each tensor
    std::sort(m_tensors.begin(), m_tensors.end(), TensorComparator());
    auto last = std::unique(m_tensors.begin(), m_tensors.end());
    m_tensors.resize(last - m_tensors.begin());

    // Increment the persistence counter if the given tensor is persistent
    auto updatePersistenceInfo = [&](const TensorPtr& tensor) {
        EAGER_ASSERT_PTR(tensor);
        if (tensor->isPersistent())
        {
            ++m_persistentNr;
            // CPU branch predictor should mitigate the overhead of the 'if'
            if (!isOriginalGraph) return;

            m_persistentTensorNamesSize += tensor->getName().length() + /*for the \0*/ 1;
            if (m_areAllTensorsSupported)
            {
                if (tensor->isDynamicShape() || tensor->isHost2DeviceTensor())
                {
                    EAGER_LOG_WARN("{}: Tensor '{}' has Dynamic-Shape which isn't supported in Eager mode",
                                   HLLOG_FUNC,
                                   tensor->getName());
                    m_areAllTensorsSupported = false;
                }
                else if (tensor->inSram())
                {
                    EAGER_LOG_WARN("{}: Tensor '{}' is in SRAM which isn't supported in Eager mode",
                                   HLLOG_FUNC,
                                   tensor->getName());
                    m_areAllTensorsSupported = false;
                }
            }
        }
    };
    std::for_each(m_tensors.begin(), m_tensors.end(), updatePersistenceInfo);
}

// Add input and output tensors of a given node, excluding NULL tensors.
void EagerTensorsSetBuilder::addTensors(const EagerNode& node)
{
    EAGER_ASSERT(m_isAddingNewTensorEnabled, "Wrong flow");
    for (const TensorVector* tensors : {&node->getInputs(), &node->getOutputs()})
    {
        for (const TensorPtr& tensor : *tensors)
        {
            if (unlikely(tensor == nullptr)) continue;
            m_tensors.push_back(tensor);
        }
    }
}

// Same as addTensors(...) except it return true if there is at least one tensor its size exceeds
// threshold for parallel execution.
bool EagerTensorsSetBuilder::addTensorsWithParallelExecCheck(const EagerNode& node)
{
    EAGER_ASSERT(m_tensorSizeThresholdForParallelExecution != -1, "Wrong flow");
    EAGER_ASSERT(m_isAddingNewTensorEnabled, "Wrong flow");
    if (m_hasTensorSizeExceedsParallelExecutionThreshold)
    {
        addTensors(node);
        return true;
    }
    for (const TensorVector* tensors : {&node->getInputs(), &node->getOutputs()})
    {
        for (const TensorPtr& tensor : *tensors)
        {
            if (unlikely(tensor == nullptr)) continue;
            m_tensors.push_back(tensor);
            if (!m_hasTensorSizeExceedsParallelExecutionThreshold)
            {
                m_hasTensorSizeExceedsParallelExecutionThreshold =
                    (tensor->getDenseSizeInBytes() >= m_tensorSizeThresholdForParallelExecution);
            }
        }
    }
    return m_hasTensorSizeExceedsParallelExecutionThreshold;
}

// Check if the at least one of the given tensors exceeds the threshold that is determined by per-chip
// TENSOR_SIZE_THRESHOLD_FOR_EAGER_PARALLEL_EXECUTION or there is already such a tensor detected at past
// processing.
bool EagerTensorsSetBuilder::checkForTensorExceedParallelExecutionThreshold(const TensorVector& tensors)
{
    EAGER_ASSERT(!m_hasTensorSizeExceedsParallelExecutionThreshold, "Wrong flow");
    EAGER_ASSERT(m_tensorSizeThresholdForParallelExecution != -1, "Wrong flow");
    for (const TensorPtr& tensor : tensors)
    {
        if (unlikely(tensor == nullptr)) continue;
        if (tensor->getDenseSizeInBytes() >= m_tensorSizeThresholdForParallelExecution)
        {
            m_hasTensorSizeExceedsParallelExecutionThreshold = true;
            return true;
        }
    }
    return false;
}

// Return index in the sorted tensors set that match the given pointer.
// if tensor is not found return -1.
// Complexity: O(Log(number of tensors)).
size_t EagerTensorsSetBuilder::getIndex(const TensorPtr& tensor) const
{
    EAGER_ASSERT(!m_isAddingNewTensorEnabled, "Wrong flow");
    const auto result = std::lower_bound(m_tensors.begin(), m_tensors.end(), tensor, TensorComparator());
    if ((result == m_tensors.end()) || (*result != tensor)) return -1;
    return result - m_tensors.begin();
}

// Fill graph inputs info for the given nodes
void EagerTensorsSetBuilder::setGraphInputs(const EagerNodes& nodes)
{
    // Note: Both m_isGraphInput and nodes may be empty in case of ZST

    m_isGraphInput.resize(m_tensors.size(), true);  // Default: all tensors are graph inputs
    for (const EagerNode& node : nodes)
    {
        for (const TensorPtr& tensor : node->getOutputs())
        {
            if (unlikely(tensor == nullptr)) continue;
            const size_t index = getIndex(tensor);
            EAGER_ASSERT(index < m_isGraphInput.size(), "Input tensor does not found");
            m_isGraphInput[index] = false;
        }
    }
}

// Check if all inputs of the given node are graph inputs
bool EagerTensorsSetBuilder::isRoot(const EagerNode& node) const
{
    for (const TensorPtr& tensor : node->getInputs())
    {
        if (unlikely(tensor == nullptr)) continue;
        const size_t index = getIndex(tensor);
        EAGER_ASSERT(index < m_isGraphInput.size(), "Input tensor does not found");
        if (m_isGraphInput[index] == false) return false;
    }
    return true;
}

}  // namespace eager_mode