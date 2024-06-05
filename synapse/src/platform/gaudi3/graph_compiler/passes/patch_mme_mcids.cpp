#include "gaudi3_graph.h"
#include "../descriptor_generator.h"

using namespace MmeCommon;

namespace gaudi3
{

static PhysicalMcid getPhysicalMcid(McidConverter& mcidConverter, const CacheMetaData& cacheMD)
{
    PhysicalMcid ret = 0;
    switch (cacheMD.cmAction)
    {
        case NOP:
            return ret;
        case DEGRADE:
            mcidConverter.convertDegrade(cacheMD.mcid, ret);
            return ret;
        case DISCARD:
            unsigned dummyRolloverIndication;
            mcidConverter.convertDiscard(cacheMD.mcid, ret, dummyRolloverIndication);
            return ret;
        default:
            HB_ASSERT(false, "Cache Maintenance Action not supported");
    }
    return ret;
}

bool patchMmeMcids(Gaudi3Graph& g)
{
    HB_ASSERT(g.getCompilationMode() != CompilationMode::Eager, "No reason to run patchMmeMcids pass for eager");

    CacheMetaDataArray cacheMetaDataArray;
    for (NodePtr node : g.getExeSortedNodes())
    {
        if (g.runsOnMME(node))
        {
            MMENodePtr              mmeNode = std::static_pointer_cast<MmeNode>(node);
            MmeDescriptorGenerator& descGenerator = g.getMmeNodeDescriptorGenerator(node);
            McidConverter           mcidConverter = g.getCodeGenerator()->getMcidConverter();

            if (mmeNode->isCdPerforated())
            {
                DescriptorGenerator::getTensorCacheMetaDataForCDParallel(mmeNode, cacheMetaDataArray);
            }
            else
            {
                DescriptorGenerator::getTensorCacheMetaData(mmeNode, cacheMetaDataArray);
            }

            PhysicalMcid aMcid = getPhysicalMcid(mcidConverter, cacheMetaDataArray[INPUT_TENSOR_A]);
            PhysicalMcid bMcid = getPhysicalMcid(mcidConverter, cacheMetaDataArray[INPUT_TENSOR_B]);
            PhysicalMcid cMcid = getPhysicalMcid(mcidConverter, cacheMetaDataArray[OUTPUT_TENSOR_C]);

            LOG_DEBUG(CACHE_MAINT,
                      "Patching MCIDs for MME node {}: OpA mcid = {}, OpB mcid = {}, OpCout mcid = {}",
                      mmeNode->getNodeName(),
                      aMcid,
                      bMcid,
                      cMcid);

            // CD Parallel - mcid for aux tensors
            std::optional<PhysicalMcid> auxScratchpadMcid = std::nullopt;
            std::optional<PhysicalMcid> auxReductionMcid  = std::nullopt;
            if (mmeNode->isCdPerforated())
            {
                auxScratchpadMcid = getPhysicalMcid(mcidConverter, cacheMetaDataArray[AUX_TENSOR_SCRATCHPAD]);
                auxReductionMcid  = getPhysicalMcid(mcidConverter, cacheMetaDataArray[AUX_TENSOR_REDUCTION]);
                LOG_DEBUG(CACHE_MAINT,
                          "CD Parallel: aux scratchpad mcid = {}, aux reduction mcid = {}",
                          auxScratchpadMcid.value(),
                          auxReductionMcid.value());
            }

            descGenerator.patchMcids(aMcid, bMcid, cMcid, auxScratchpadMcid, auxReductionMcid);
            McidMmeUsage mcidMmeUsage;

            NodeROI& firstRoi = g.GetNodeROIs(node)->front();
            NodeROI& lastRoi  = g.GetNodeROIs(node)->back();
            NodeROI* roi;

            MmeCommon::TensorRoles tensorRoleA;
            MmeCommon::TensorRoles tensorRoleB;
            MmeCommon::TensorRoles tensorRoleC;
            // register the descriptors in the graph with their mcid usage
            for (const auto& activation : descGenerator.getMmeActivations())
            {
                tensorRoleA = activation.operandRoles[MmeCommon::INPUT_TENSOR_A];
                tensorRoleB = activation.operandRoles[MmeCommon::INPUT_TENSOR_B];
                tensorRoleC = activation.operandRoles[MmeCommon::OUTPUT_TENSOR_C];

                mcidMmeUsage.operandA = cacheMetaDataArray[tensorRoleA].cmAction;
                mcidMmeUsage.operandB = cacheMetaDataArray[tensorRoleB].cmAction;
                mcidMmeUsage.operandC = cacheMetaDataArray[tensorRoleC].cmAction;

                if (tensorRoleA == MmeCommon::AUX_TENSOR_REDUCTION && tensorRoleB == MmeCommon::AUX_TENSOR_SCRATCHPAD)
                {
                    // CD parallel - reductionAdd activation (matches the last roi)
                    roi = &lastRoi;
                }
                else
                {
                    roi = &firstRoi;
                }

                for (unsigned descIdx = 0; descIdx < g.getHALReader()->getNumMmeEngines(); descIdx++)
                {
                    g.updateMmeNodeDescriptorWrapper(*mmeNode, activation.getDesc(descIdx), mcidMmeUsage, *roi, descIdx);
                }
            }
        }
    }
    return true;
}

}  // namespace gaudi3
