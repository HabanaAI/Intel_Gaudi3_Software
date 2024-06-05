#pragma once

#include "gaudi_types.h"
#include "include/gaudi/mme_descriptor_generator.h"

namespace gaudi
{
class DynamicMMEPatchPointGenerator
{
public:
    DynamicMMEPatchPointGenerator() = default;

    void generateDynamicShapesPatchPoints(const MmeNode& node, DescriptorWrapper<gaudi::MmeDesc>& descWrapper);

private:
    void generateDynamicPatchPointsForOperand(const MmeNode&                     node,
                                              const pTensor&                     tensor,
                                              MmeCommon::EMmeOperand             op,
                                              DescriptorWrapper<gaudi::MmeDesc>& descWrapper);

    void addValidElementsPatchPoint(const MmeNode&                     node,
                                    const pTensor&                     tensor,
                                    const Mme::MmeTensorDesc*          tensorDesc,
                                    int                                dim,
                                    DescriptorWrapper<gaudi::MmeDesc>& descWrapper);

    int getTensorIndex(const TensorVector& tensorVector, const pTensor& tensor);

    void generatePaddingPatchPoints(const ConvBaseNode& node, DescriptorWrapper<gaudi::MmeDesc>& descWrapper);
    void generateOneAguPaddingPatchPoint(const ConvBaseNode&                node,
                                         DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                                         const int32_t*                     roiOffsets);
    void generateOneDimPaddingPatchPoint(const ConvBaseNode&                node,
                                         DescriptorWrapper<gaudi::MmeDesc>& descWrapper,
                                         uint32_t                           dim,
                                         const int32_t*                     roiOffsets);
};

}  // namespace gaudi
