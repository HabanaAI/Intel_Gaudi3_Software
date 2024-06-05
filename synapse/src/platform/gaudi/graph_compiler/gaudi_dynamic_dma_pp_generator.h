#pragma once

#include "gaudi_types.h"
#include <dynamic_dma_pp_generator.h>

namespace gaudi
{
class DynamicDMAPatchPointGenerator : public ::DynamicDMAPatchPointGenerator<gaudi::DmaDesc>
{
    using ::DynamicDMAPatchPointGenerator<gaudi::DmaDesc>::DynamicDMAPatchPointGenerator;

    // This function in Gaudi/GaudiM does nothing
    // because dynamic execution PPs are insertrd by the qman instead
    virtual void addDynamicExecutionPatchPoints(const DMANode& node) override {}

    virtual DynamicShapeFieldInfoSharedPtr getDynamicAddressInfo(const DMAPhysicalMemoryOpNode& node,
                                                                 ShapeFuncID                    smf) override;

    virtual uint32_t                    fieldTypeAndDimToOffset(FieldType fieldType, uint32_t dim) override;
    virtual std::pair<uint32_t, BlockT> fieldTypeAndDimToOffsetAndBlock(FieldType fieldType, uint32_t dim) override;
    virtual uint64_t                    getAddressPtrForPhysicalMemOp(const DMAPhysicalMemoryOpNode& node) override;

    virtual uint32_t addressOffsetHi(bool isSrc) override;
    virtual uint32_t addressOffsetLo(bool isSrc) override;
    virtual uint32_t addressValueHi(bool isSrc) override;
    virtual uint32_t addressValueLo(bool isSrc) override;
};
}  // namespace gaudi
