#pragma once

#include "log_manager.h"
#include "tensor.h"

namespace ConstSectionReplacer
{
template<typename TensorsContainer>
void replace(const TensorPtr& tensor, const TensorsContainer& tensors, bool unbindReplacedTensor = false)
{
    uint64_t                   offset           = tensor->getMemorySectionOffset();
    uint64_t                   sectionId        = tensor->getMemorySectionID();
    const auto&                sectionHandle    = tensor->getSectionHandle();
    const synMemoryDescriptor& memoryDescriptor = tensor->getMemoryDescriptor();

    for (const auto& t : tensors)
    {
        LOG_DEBUG(GC,
                  "Updating tensor {} memory section info to, section ID: {}, "
                  "section offset: {}, isPersistent: {}",
                  t->getName(),
                  sectionId,
                  offset,
                  memoryDescriptor.m_isPersistent);

        t->setMemoryDescriptor(memoryDescriptor);
        t->setSectionHandle(sectionHandle);
        t->setMemorySectionID(sectionId);
        t->setMemorySectionOffset(offset);
        offset += t->getTotalSizeInBytes();
    }

    // Original const buffer unneeded, only use already has been const folded.
    if (unbindReplacedTensor)
    {
        LOG_TRACE(GC, "Unbinding const tensor {} after const folding", tensor->getName());
        tensor->unbind();
    }

}
}  // namespace ConstSectionReplacer