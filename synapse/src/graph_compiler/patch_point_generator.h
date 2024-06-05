#pragma once

#include "address_fields_container_info.h"
#include "descriptor_wrapper.h"
#include "tpc_node.h"
#include "habana_nodes.h"

class PatchPointGenerator
{
public:
    PatchPointGenerator() = default;

    static void createDescriptorPatchPoint(const TensorPtr&          gcTensor,
                                           const uint32_t*           pAddressFieldLow,
                                           const uint32_t*           pAddressFieldHigh,
                                           BasicFieldsContainerInfo& descFieldsInfoContainer,
                                           uint64_t                  descriptorBaseAddress,
                                           const NodePtr&            node = nullptr);
};

template<class Descriptor>
class TPCPatchPointGenerator : public PatchPointGenerator
{
public:
    TPCPatchPointGenerator() : m_tensorIdx(0) {}

    virtual void generateTpcPatchPoints(const TPCNode& node, DescriptorWrapper<Descriptor>& descWrapper);

protected:
    void generatePrintfPatchPoints(const TPCNode& node, DescriptorWrapper<Descriptor>& descWrapper);
    virtual uint32_t* getBaseAddrHigh(Descriptor& desc) = 0;
    virtual uint32_t* getBaseAddrLow(Descriptor& desc)  = 0;
    size_t m_tensorIdx;
};

template<class Descriptor>
class DMAPatchPointGenerator : public PatchPointGenerator
{
public:
    DMAPatchPointGenerator() = default;

    virtual void generateDmaPatchPoints(const DMANode& node, DescriptorWrapper<Descriptor>& descWrapper);
};

template<class Descriptor>
class MMEPatchPointGenerator : public PatchPointGenerator
{
public:
    MMEPatchPointGenerator() = default;
};

template<class Descriptor>
class RotatorPatchPointGenerator : public PatchPointGenerator
{
public:
    RotatorPatchPointGenerator() = default;

    virtual void generateRotatorPatchPoints(const RotateNode& node, DescriptorWrapper<Descriptor>& descWrapper);

    virtual void generateEmptyJobRotatorPatchPoints(DescriptorWrapper<Descriptor>& descWrapper);
};
