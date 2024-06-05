#pragma once

#include "graph_compiler/patch_point_generator.h"
#include "graph_compiler/patch_point_generator.inl"
#include "platform/gaudi3/graph_compiler/tpc_descriptor_generator.h"
#include "gaudi3_types.h"

namespace gaudi3
{
typedef RotatorPatchPointGenerator<gaudi3::RotatorDesc> Gaudi3RotatorPatchPointGenerator;

class Gaudi3MMEPatchPointGenerator : public MMEPatchPointGenerator<MmeDesc>
{
public:
    Gaudi3MMEPatchPointGenerator() : MMEPatchPointGenerator<MmeDesc>() {}

    void generateMmePatchPoints(const MmeNode&                node,
                                const MmeDescriptorGenerator& descGenerator,
                                DescriptorWrapper<MmeDesc>&   descWrapper,
                                const McidMmeUsage&           mcidMmeUsage,
                                unsigned                      engineIdx);

private:
    void createPatchPointIfNotNull(const TensorPtr&              tensor,
                                   EMmeOperand                   operand,
                                   const MmeDescriptorGenerator& descGenerator,
                                   DescriptorWrapper<MmeDesc>&   descWrapper);

    void generateMcidMmePatchPoints(const MmeNode&              node,
                                    DescriptorWrapper<MmeDesc>& descWrapper,
                                    const McidMmeUsage&         mcidMmeUsage);
};

class Gaudi3TPCPatchPointGenerator : public TPCPatchPointGenerator<gaudi3::TpcDesc>
{
public:
    Gaudi3TPCPatchPointGenerator() : TPCPatchPointGenerator<gaudi3::TpcDesc>() {}

    void generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi3::TpcDesc>& descWrapper) override;
    void generateTpcMcidPatchPoints(const TPCNode& node, DescriptorWrapper<gaudi3::TpcDesc>& descWrapper);
    void setMcidTpcUsage(TpcDescriptorGenerator::McidTpcUsage& mcidTpcUsage) { m_mcidTpcUsage = mcidTpcUsage; };

protected:
    uint32_t* getBaseAddrHigh(gaudi3::TpcDesc& desc) override;
    uint32_t* getBaseAddrLow(gaudi3::TpcDesc& desc) override;

private:
    TpcDescriptorGenerator::McidTpcUsage m_mcidTpcUsage;
};

}  // namespace gaudi3
