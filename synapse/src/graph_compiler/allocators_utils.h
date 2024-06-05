#pragma once

#include "types.h"

#include <cstdint>
#include <functional>
#include <memory>

struct GraphAnnotation;
class HabanaGraph;
class NonPersistentSectionAllocTracker;
class Tensor;

namespace SynapseInternals
{
    class MemoryAllocator;
}

using SynapseInternals::MemoryAllocator;

#define ALL_TENSORS_COMPATIBILTY           std::bind(&isAnyTensor, std::placeholders::_1)
#define STATIC_TENSORS_COMPATIBILTY        std::bind(&isStaticTensor, std::placeholders::_1)
#define ACTIVATIONS_TENSORS_COMPATIBILTY   std::bind(&isActivationTensor, std::placeholders::_1)
#define NON_PERSISTENT_TENSORS             std::bind(&isNonPersistentTensor, &g, std::placeholders::_1)
#define NON_PERSISTENT_ACTIVATIONS_TENSORS std::bind(&isNonPersistentActivationTensor, &g, std::placeholders::_1)
#define FORCE_DRAM_TENSORS_COMPATIBILTY    std::bind(&isAllocInDramForced, std::placeholders::_1)
#define SRAM_TENSORS_COMPATIBILTY          std::bind(&isSramIndicatedTensor, std::placeholders::_1)

typedef std::function<bool(const pTensor& t)> TensorsCompatibility;

enum allocationType
{
    ALLOC_TYPE_PERSISTENT_IO,
    ALLOC_TYPE_STATIC_TENSORS,
    ALLOC_TYPE_ACTIVATIONS,
    ALLOC_TYPE_FORCED_DRAM,
    ALLOC_TYPE_ALL
};

struct Lifetime
{
    uint32_t m_start;  // first generation
    uint32_t m_end;    // last generation (inclusive)
};

bool allocateTensorInSram(HabanaGraph&                      graph,
                          pTensor                           tensor,
                          bool                              allocateRealTensor = true,
                          bool                              allowFailure       = false,
                          MemoryAllocator*                  alloc              = nullptr,
                          NonPersistentSectionAllocTracker* allocTracker       = nullptr);
bool allocateTensorInDram(HabanaGraph&                      graph,
                          pTensor                           tensor,
                          bool                              allocateRealTensor = true,
                          bool                              allowFailure       = false,
                          MemoryAllocator*                  alloc              = nullptr,
                          NonPersistentSectionAllocTracker* allocTracker       = nullptr,
                          Lifetime                          tensorLifetime     = {});

bool freeTensorFromSram(HabanaGraph&                      graph,
                        const TensorPtr&                  tensor,
                        bool                              freeRealTensor,  // = true
                        MemoryAllocator*                  alloc,           // = nullptr
                        bool                              rollback,        // = false
                        NonPersistentSectionAllocTracker* allocTracker);   // = nullptr

bool freeTensorFromDram(HabanaGraph&                      graph,
                        pTensor                           tensor,
                        bool                              freeRealTensor = true,
                        MemoryAllocator*                  alloc          = nullptr,
                        NonPersistentSectionAllocTracker* allocTracker   = nullptr);

void setMemoryAllocError(GraphAnnotation& ann, bool isIO = false);

void resetMemoryAllocError(GraphAnnotation& ann);

bool isMemoryAllocErrorSet(GraphAnnotation& ann, bool isIO = false);

bool validateTensorComp(HabanaGraph* graph, std::shared_ptr<Tensor> t, allocationType allocType);

bool validateTensorComp(const HabanaGraph& graph, std::shared_ptr<Tensor> t, allocationType allocType);

bool isAllocInDramForced(const pTensor& tensor);

bool isSramIndicatedTensor(const pTensor& tensor);

uint64_t getWriteSpaceForTensor(const pTensor& tensor);

bool isAnyTensor(const pTensor& tensor);
bool isStaticTensor(const pTensor& tensor);
bool isActivationTensor(const pTensor& tensor);
bool isNonPersistentTensor(HabanaGraph* g, const pTensor& tensor);
bool isNonPersistentActivationTensor(HabanaGraph* g, const pTensor& tensor);

bool canApplyReuse(pTensor reused, pTensor reusing, const HabanaGraph& g);
