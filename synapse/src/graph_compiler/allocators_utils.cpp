#include "allocators_utils.h"

#include "alloc_utils.h"
#include "graph_annotation.h"
#include "habana_graph.h"
#include "infra/settable.h"
#include "math_utils.h"
#include "memory_management/heap_allocator.h"
#include "memory_management/memory_allocator.h"
#include "range.h"
#include "tensor.h"
#include "tensor_annotation.h"

#include <memory>

void setIOsMemAllocationError(HabanaGraph &graph)
{
    setMemAllocationError(graph);
    graph.getGraphAnnotation().errors.IOsMemAllocationError = true;
}

bool allocateTensorInSram(HabanaGraph&             graph,
                          pTensor                  tensor,
                          bool                     allocateRealTensor /* = true */,
                          bool                     allowFailure /* = False */,
                          MemoryAllocator*         alloc /* = nullptr */,
                          NonPersistentSectionAllocTracker* allocTracker /* = nullptr */)
{
    if (alloc == nullptr)
    {
        alloc = &graph.getCodeGenerator()->getSramAllocator();
    }
    return allocateTensor(tensor, true, allocateRealTensor, allowFailure, *alloc, allocTracker);
}

bool allocateTensorInDram(HabanaGraph&                      graph,
                          pTensor                           tensor,
                          bool                              allocateRealTensor /* = true */,
                          bool                              allowFailure /* = false */,
                          MemoryAllocator*                  alloc /* = nullptr */,
                          NonPersistentSectionAllocTracker* allocTracker /* = nullptr */,
                          Lifetime                          tensorLifetime)
{
    if (alloc == nullptr)
    {
        alloc = &graph.getCodeGenerator()->getWorkspaceAllocator();
    }

    if (tensor->isStaticParam() && tensor->hasDramAllocatedTensor())
    {
        if (!tensor->getDramAllocatedTensor()->isDramOffsetSet())
        {
            LOG_TRACE(GC,"- {}: Allocating Tensor {} in DRAM", HLLOG_FUNC, tensor->getName());
            if (!allocateTensorInDram(graph,
                                      tensor->getDramAllocatedTensor(),
                                      allocateRealTensor,
                                      allowFailure,
                                      alloc,
                                      allocTracker))
            {
                LOG_ERR(GC,
                        "{}: Failed to allocate DRAM allocated tensor {} of tensor {}",
                        HLLOG_FUNC,
                        tensor->getDramAllocatedTensor()->getName(),
                        tensor->getName());
                return false;
            }
        }
        tensor->setDramOffset(tensor->getDramAllocatedTensor()->getDramOffset() + tensor->getDramAllocatedTensorOffset());
        return true;
    }

    return allocateTensor(tensor, false, allocateRealTensor, allowFailure, *alloc, allocTracker, tensorLifetime);
}

bool freeTensorFromSram(HabanaGraph&             graph,
                        const TensorPtr&         tensor,
                        bool                     freeRealTensor /* = true */,
                        MemoryAllocator*         alloc /* = nullptr */,
                        bool                     rollback /* = false */,
                        NonPersistentSectionAllocTracker* allocTracker /* = nullptr */)
{
    if (alloc == nullptr)
    {
        alloc = &graph.getCodeGenerator()->getSramAllocator();
    }
    return freeTensor(tensor, true, freeRealTensor, *alloc, rollback, allocTracker);
}

bool freeTensorFromDram(HabanaGraph&             graph,
                        pTensor                  tensor,
                        bool                     freeRealTensor /* = true */,
                        MemoryAllocator*         alloc /* = nullptr */,
                        NonPersistentSectionAllocTracker* allocTracker /* = nullptr */)
{
    if (alloc == nullptr)
    {
        alloc = &graph.getCodeGenerator()->getWorkspaceAllocator();
    }
    return freeTensor(tensor, false, freeRealTensor, *alloc, /*rollback*/ false, allocTracker);
}

void setMemoryAllocError(GraphAnnotation &ann,
                         bool isIO /* = false */)
{
    ann.errors.memoryAllocationError = true;
    ann.errors.IOsMemAllocationError = isIO;
}

void resetMemoryAllocError(GraphAnnotation &ann)
{
    ann.errors.memoryAllocationError = false;
    ann.errors.IOsMemAllocationError = false;
}

bool isMemoryAllocErrorSet(GraphAnnotation &ann,
                           bool isIO /* = false */)
{
    return (ann.errors.memoryAllocationError ||
            (isIO && ann.errors.IOsMemAllocationError));
}

bool validateTensorComp(const HabanaGraph& graph, pTensor t, allocationType allocType)
{
    GET_REAL_TENSOR_IF_NULL_RETURN_VAL(t, false);
    bool forceDram = isAllocInDramForced(t);

    if (forceDram && (allocType != ALLOC_TYPE_FORCED_DRAM))
    {
        return false;
    }

    if (t->isShapeTensor()) return false;

    switch(allocType)
    {
        case ALLOC_TYPE_PERSISTENT_IO:
            return !t->isModelParameter() && (graph.isInputTensor(t) || graph.isOutputTensor(t)) &&
                   graph.getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs;
        case ALLOC_TYPE_STATIC_TENSORS:
            return (t->isStaticParam());
        case ALLOC_TYPE_ACTIVATIONS:
            return (!t->isModelParameter());
        case ALLOC_TYPE_FORCED_DRAM:
            return forceDram;
        case ALLOC_TYPE_ALL:
            return true;
        default:
            return false;
    }
}

bool validateTensorComp(HabanaGraph *graph,
                        pTensor t,
                        allocationType allocType)
{
    return validateTensorComp(*graph, t, allocType);
}

static bool shouldBeAllocated(const TensorPtr& tensor)
{
    return !tensor->isShapeTensor() && !tensor->isPersistent() && !tensor->isZeroSizedDataTensor();
}

bool isAllocInDramForced(const pTensor &tensor)
{
    return (shouldBeAllocated(tensor) && tensor->inDram());
}


uint64_t getWriteSpaceForTensor(const pTensor &tensor)
{
    // Compute space needed for tensor (tensor size + alignment + offset)
    uint64_t tensorSize  = getSizeToAllocate(tensor) + tensor->getTensorAnnotation().memory.offset;
    uint64_t alignedSize = round_to_multiple(tensorSize, tensor->getTensorAnnotation().memory.alignment);

    return alignedSize;
}

bool isAnyTensor(const pTensor &tensor)
{
    return true;
}

bool isStaticTensor(const pTensor &tensor)
{
    return tensor->isStaticParam();
}

bool isActivationTensor(const pTensor &tensor)
{
    return !tensor->isModelParameter();
}

bool isNonPersistentTensor(HabanaGraph* g, const pTensor &tensor)
{
    return !isAllocInDramForced(tensor) &&
           (tensor->isModelParameter() ||
            (!g->isInputTensor(tensor) && !g->isOutputTensor(tensor)) ||
            (!g->getGraphAnnotation().memoryStrategyParams.sramRegionsInfo.persistentIOs));
}

bool isSramIndicatedTensor(const pTensor& tensor)
{
    return tensor->inSram();
}

bool isNonPersistentActivationTensor(HabanaGraph* g, const pTensor &tensor)
{
    return isActivationTensor(tensor) && isNonPersistentTensor(g, tensor);
}

static bool inputWithMultipleConsumers(pTensor cand, const HabanaGraph& g)
{
    return g.getNumberOfTensorConsumers(cand) > 1;
}

bool canApplyReuse(pTensor reused, pTensor reusing, const HabanaGraph& g)
{
    LOG_TRACE(GC, "{}: test if tensor {} can reuse tensor {}", HLLOG_FUNC, reusing->getName(), reused->getName());
    //check that tensors memory access is equal
    if (!reused->isDenseLayout() || !reusing->isDenseLayout())
    {
        LOG_TRACE(GC,
                  "{}: can not apply inplace reuse because one of {}, {} is strided tensor",
                  HLLOG_FUNC,
                  reusing->getName(),
                  reused->getName());
        return false;
    }
    if (reused->isAliasedTensor() || reusing->isAliasedTensor())
    {
        LOG_TRACE(GC,
                  "{}: can not apply inplace reuse because one of {}, {} is alias tensor",
                  HLLOG_FUNC,
                  reusing->getName(),
                  reused->getName());
        return false;
    }
    if (g.isInputTensor(reused))
    {
        LOG_TRACE(GC, "{}: {} is graph input and can not be reused", HLLOG_FUNC, reused->getName());
        return false;
    }
    if (g.isOutputTensor(reusing))
    {
        LOG_TRACE(GC, "{}: {} is graph output and can not reuse", HLLOG_FUNC, reusing->getName());
        return false;
    }
    // check that the reused tensor does'nt have multiple consumers
    if (inputWithMultipleConsumers(reused, g))
    {
        LOG_TRACE(GC, "{}: can not apply reuse because {} has multiple consumers", HLLOG_FUNC, reused->getName());
        return false;
    }
    // check if reused tensor forced to dram
    if (reused->inDram() || reusing->inDram())
    {
        return false;
    }
    // check if isStaticParam
    if (reused->isStaticParam())
    {
        return false;
    }
    return true;
}
