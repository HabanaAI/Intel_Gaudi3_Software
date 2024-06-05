#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "recipe_gen/recipe_hal_base.h"

namespace eager_mode
{
class EagerGraph;

class MmeDescGeneratorBase : public DescGeneratorBase
{
public:
    virtual void
    copyDescToBlob(Byte* out, unsigned descIdx, StructSizeType offsetInDescriptor, BlobSizeType sizeToCopy) const = 0;

protected:
    MmeDescGeneratorBase(EagerGraph& graph, const EagerNode& node, const MmeCommon::MmeLayerParams& params)
    : DescGeneratorBase(graph, node), m_params(params)
    {
        EAGER_ASSERT(node.getEngineType() == EngineType::MME, "Invalid engine type");
        m_patchableTensorsNr = node->getInputs().size() + node->getOutputs().size();
    }

    std::array<deviceAddrOffset, RecipeHalBase::maxMmeTensorsNr> m_operandVirtualAddress = {};
    MmeCommon::MmeLayerParams                                    m_params;
};

}  // namespace eager_mode
