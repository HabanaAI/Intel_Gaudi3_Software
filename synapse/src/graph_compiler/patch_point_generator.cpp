#include "patch_point_generator.h"
#include "utils.h"

void PatchPointGenerator::createDescriptorPatchPoint(const TensorPtr&          gcTensor,
                                                     const uint32_t*           pAddressFieldLow,
                                                     const uint32_t*           pAddressFieldHigh,
                                                     BasicFieldsContainerInfo& descFieldsInfoContainer,
                                                     uint64_t                  descriptorBaseAddress,
                                                     const NodePtr&            node)
{
    if ((gcTensor != nullptr) && gcTensor->tensorAllocatedInSram()) return;

    ptrToInt fullAddr;
    uint64_t memId;

    fullAddr.u32[0] = *pAddressFieldLow;
    fullAddr.u32[1] = *pAddressFieldHigh;
    memId           = getMemoryIDFromVirtualAddress(fullAddr.u64);

    descFieldsInfoContainer.addAddressEngineFieldInfo(node,
                                                      getMemorySectionNameForMemoryID(memId, gcTensor),
                                                      memId,
                                                      (uint64_t) pAddressFieldLow,
                                                      (uint64_t) pAddressFieldHigh,
                                                      descriptorBaseAddress);
}
