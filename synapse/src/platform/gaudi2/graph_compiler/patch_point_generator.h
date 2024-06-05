#pragma once
#include "graph_compiler/patch_point_generator.h"
#include "graph_compiler/patch_point_generator.inl"
#include "gaudi2_types.h"

namespace gaudi2
{
typedef RotatorPatchPointGenerator<gaudi2::RotatorDesc> Gaudi2RotatorPatchPointGenerator;

class Gaudi2MMEPatchPointGenerator : public MMEPatchPointGenerator<MmeDesc>
{
public:
    Gaudi2MMEPatchPointGenerator() : MMEPatchPointGenerator<MmeDesc>() {}

    void generateMmePatchPoints(const MmeNode& node,
                                const MmeDescriptorGenerator& descGenerator,
                                DescriptorWrapper<MmeDesc>& descWrapper);

private:
    void createPatchPointIfNotNull(const TensorPtr& tensor, EMmeOperand operand,
                                   const MmeDescriptorGenerator& descGenerator,
                                   DescriptorWrapper<MmeDesc>& descWrapper);
};

class Gaudi2DMAPatchPointGenerator : public PatchPointGenerator
{
public:
    Gaudi2DMAPatchPointGenerator() : PatchPointGenerator() {}

    void generateDmaPatchPoints(const DMANode& node, DescriptorWrapper<gaudi2::DmaDesc>& descWrapper);
};

class Gaudi2TPCPatchPointGenerator : public TPCPatchPointGenerator<gaudi2::TpcDesc>
{
public:
    Gaudi2TPCPatchPointGenerator() : TPCPatchPointGenerator<gaudi2::TpcDesc>()     {}

    void generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi2::TpcDesc>& descWrapper) override;

protected:
    uint32_t* getBaseAddrHigh(gaudi2::TpcDesc& desc) override;
    uint32_t* getBaseAddrLow(gaudi2::TpcDesc& desc) override;
};

}