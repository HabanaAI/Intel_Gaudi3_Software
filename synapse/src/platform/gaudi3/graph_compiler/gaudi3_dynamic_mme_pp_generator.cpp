#include "mme/mme_brain_ifc.h"
#include "mme/mme_desc_gen_utils.h"
#include "gaudi3_dynamic_mme_pp_generator.h"
#include "recipe_metadata.h"
#include "platform/gaudi3/graph_compiler/descriptor_generator.h"
#include "platform/gaudi/graph_compiler/smf/smf.h"
#include "platform/common/graph_compiler/arc_mme_field_infos.h"
#include "platform/common/graph_compiler/arc_dynamic_mme_pp_generator.inl"


namespace gaudi3
{

// The function from Gaudi2 doesn't work for Gaudi3, for reasons that are stil unknown
// We leave it empty for the time being. This leaves null descriptor
// functional (valid_elements PP still works) but noit as performant as it could be
// with the dynamic execution PP.
void DynamicMMEPatchPointGenerator::generateDynamicExecutionPatchPoint(const MmeNode& node, DescriptorWrapper<MmeDesc>& descWrapper)
{
}

DynamicMMEPatchPointGenerator::TensorTile DynamicMMEPatchPointGenerator::getTensorTileFromEngine(const MmeNode&                node,
                                                                                                 const TensorPtr&              tensor,
                                                                                                 unsigned                      engineIdx,
                                                                                                 bool&                         haveTile)
{
    if (node.getNodeAnnotation().m_dcoreROIs.empty())
    {
        // This node works on the old-fashioned way the entire tensor is the tile

        static SizeArray zeros{0};
        TensorTile ret(tensor->getDim(), tensor->getShape().getMaxSizes(), zeros);
        haveTile = false;
        return ret;
    }

    unsigned dcoreIdx = engineIdx / 2;
    // Copy this dcore ROI because we need to modify it
    auto dcoreRoi = node.getNodeAnnotation().m_dcoreROIs[dcoreIdx];
    auto& firstDcoreRoi = node.getNodeAnnotation().m_dcoreROIs[0];

    // re-offset to get small tensor offsets, see MmeDescriptorBuilder::getDcoreOffsetBasedOnSmallTensor
    for (unsigned i = 0; i < HABANA_DIM_MAX; ++i)
        dcoreRoi.baseOffset[i] -= firstDcoreRoi.baseOffset[i];

    auto accessPattern = node.getNodeAccessPattern();
    llvm_vecsmall::SmallVector<uint64_t, tpc_lib_api::MAX_TENSOR_DIM> geometry(std::begin(dcoreRoi.size),
                                                                               std::end(dcoreRoi.size));
    llvm_vecsmall::SmallVector<uint64_t, tpc_lib_api::MAX_TENSOR_DIM> offset(std::begin(dcoreRoi.baseOffset),
                                                                             std::end(dcoreRoi.baseOffset));

    geometry.resize(accessPattern->getNodeResolution().size());
    offset.resize(accessPattern->getNodeResolution().size());

    auto nodeTile = gc::access_pattern::NodeTile(geometry, offset);

    auto outTile   = accessPattern->getTensorTile(tensor, nodeTile);

    haveTile = true;

    return outTile;
}


} // namespace gaudi3

namespace arc_platforms
{

// explicit instantiation for gaudi3
template class DynamicMMEPatchPointGenerator<gaudi3::MmeTypes>;

};


