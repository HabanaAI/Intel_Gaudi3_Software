#include "brain_conf.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "hal_reader/hal_reader.h"
#include "mme_brain_ifc.h"

struct McidCmAction
{
    LogicalMcid            mcid     = 0;
    CacheMaintenanceAction cmAction = NOP;
};

class LitePerforation
{
private:
    bool areSameBundle(const Settable<BundleInfo>& info1, const Settable<BundleInfo>& info2)
    {
        return (info1.is_set() == info2.is_set()) && (!info1.is_set() || (info2->bundleIndex == info1->bundleIndex));
    }

    // return true if both nodes are not in any bundle or both in same bundle
    bool nodesInSameBundle(const NodePtr& n1, const NodePtr& n2)
    {
        return areSameBundle(n1->getNodeAnnotation().bundleInfo, n2->getNodeAnnotation().bundleInfo);
    }

    // return true only if both nodes in same bundle. False if either (or the 2 nodes) are unbundeled or in different
    // bundles
    bool areNodesBundled(const Settable<BundleInfo>& info1, const Settable<BundleInfo>& info2)
    {
        return (info1.is_set() && info2.is_set() && info1->bundleIndex == info2->bundleIndex);
    }

    bool isNodeBundled(HabanaGraph& g, const NodePtr& node)
    {
        // Ignore this check if we want to treat all nodes as bundles (including "single" nodes)
        if (GCFG_LITE_PERFORATION_SKIP_BUNDLE_CHECK.value()) return true;

        for (const TensorPtr& tensor : node->getInputs())
        {
            if (!tensor) continue;

            for (auto& n : g.getRealProducers(tensor))
            {
                if (areNodesBundled(node->getNodeAnnotation().bundleInfo, n->getNodeAnnotation().bundleInfo))
                {
                    return true;
                }
            }
        }

        for (const TensorPtr& tensor : node->getOutputs())
        {
            if (!tensor) continue;

            for (auto& n : g.getRealConsumers(tensor))
            {
                if (areNodesBundled(node->getNodeAnnotation().bundleInfo, n->getNodeAnnotation().bundleInfo))
                {
                    return true;
                }
            }
        }

        return false;
    }

    bool isBPT(HabanaGraph& g, const NodePtr& node, const TensorPtr& tensor, bool bIsInput)
    {
        // if at least one of the input tensor producers or output tensor consumers is outside the bundle - it is
        // considered BPT
        if (bIsInput)
        {
            for (auto& n : g.getRealProducers(tensor))
            {
                if (!nodesInSameBundle(node, n))
                {
                    return true;
                }
            }
        }
        else
        {
            for (auto& n : g.getRealConsumers(tensor))
            {
                if (!nodesInSameBundle(node, n))
                {
                    return true;
                }
            }
        }

        return false;
    }

    void setTensorCacheMaintenanceAction(HabanaGraph&     g,
                                         const NodePtr&   node,
                                         const TensorPtr& tensor,
                                         CacheMetaData&   tensorCacheMetaData,
                                         bool             bIsInput)
    {
        // Logic: for each non-persistent tensor AND in bundled node:

        // 1. if non-BPT – meaning not consumed outside the bundle --> DISCARD
        //    else --> DEGRADE
        // 2. Set also MCID using 2 running counters

        if (Tensor::getRealTensor(tensor)->isPersistent() || !isNodeBundled(g, node)) return;

        uint64_t tensorId = Tensor::getRealTensor(tensor)->getId();

        bool isNewTensor = m_tensorIdToMcidCmAction.find(tensorId) == m_tensorIdToMcidCmAction.end();

        int64_t liteCMEMode = GCFG_LITE_CME_MODE.value();  // 0 = disabled, 1 = discard, 2 = degrade, 3 = both

        if (isNewTensor)
        {
            if (isBPT(g, node, tensor, bIsInput))
            {
                if (liteCMEMode == 2 || liteCMEMode == 3)
                {
                    tensorCacheMetaData.mcid     = g.getCodeGenerator()->getNextMCID(MCIDGenerator::MCIDType::DEGRADE);
                    tensorCacheMetaData.cmAction = DEGRADE;
                    m_tensorIdToMcidCmAction[tensorId] = {tensorCacheMetaData.mcid, tensorCacheMetaData.cmAction};
                }
            }
            else
            {
                if (liteCMEMode == 1 || liteCMEMode == 3)
                {
                    tensorCacheMetaData.mcid     = g.getCodeGenerator()->getNextMCID(MCIDGenerator::MCIDType::DISCARD);
                    tensorCacheMetaData.cmAction = DISCARD;
                    m_tensorIdToMcidCmAction[tensorId] = {tensorCacheMetaData.mcid, tensorCacheMetaData.cmAction};
                }
            }
        }
        else
        {
            tensorCacheMetaData.mcid     = m_tensorIdToMcidCmAction[tensorId].mcid;
            tensorCacheMetaData.cmAction = m_tensorIdToMcidCmAction[tensorId].cmAction;
        }

        LOG_DEBUG(CACHE_MAINT,
                  "Setting action {} and MCID: {} to tensor: {}, id: {} in node: {}",
                  tensorCacheMetaData.cmAction,
                  tensorCacheMetaData.mcid,
                  tensor->getName(),
                  tensorId,
                  node->getNodeName());
    }

    void setTensorPerforationIndication(const TensorPtr& tensor, CacheMetaData& tensorCacheMetaData)
    {
        auto& perforationData = tensor->getTensorAnnotation().perforation;
        if (!perforationData.has_value()) return;

        if (perforationData->sliced && perforationData->cached)
        {
            tensorCacheMetaData.cacheDirective = DcoreAllocate;
        }
    }

    static bool balanceDcoreSplit(unsigned size, unsigned granularity, unsigned dcoreNr, std::vector<unsigned>& sizes)
    {
        std::fill(sizes.begin(), sizes.end(), 0);
        while (size)
        {
            for (int dcore = 0; dcore < dcoreNr; dcore++)
            {
                sizes[dcore] += granularity;
                size -= granularity;
                if (size == 0) break;
            }
        }
        if (!GCFG_ENABLE_UNEVEN_PERFORATION_IN_MME.value())
        {
            // make sure all splits are even to avoid strange corner cases - SW-136623
            unsigned dcoreSize = sizes[0];
            for (unsigned dcore = 1; dcore < dcoreNr; dcore++)
            {
                if (sizes[dcore] != dcoreSize)
                {
                    return false;
                }
            }
        }
        return true;
    }

    std::map<uint64_t, McidCmAction> m_tensorIdToMcidCmAction;

public:
    void perforateSingleNode(const NodePtr& node, HabanaGraph& g)
    {
        if (node->isLogicalOperation()) return;                          // can only perforate physical nodes
        if (!node->getNodeAnnotation().perforation.has_value()) return;  // cant perforate unsliced nodes
        LOG_TRACE(DCORE_SPLITTER, "trying to perforate node {}", node->getNodeName());
        bool isMmeNode = HabanaGraph::runsOnMME(node);
        if (isMmeNode)
        {
            const auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
            HB_ASSERT_PTR(mmeNode);

            if (mmeNode->isCdIndexSpaceDim(mmeNode->getNodeAnnotation().perforation->indexSpaceDim))
            {
                for (const TensorPtr& tensor : node->getOutputs())
                {
                    if (!tensor) continue;
                    // TODO: replace with - (!tensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled &&
                    // !GCFG_ENABLE_CD_PARALLEL.value()) once cd parallel implementation is done
                    if (!tensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled ||
                        GCFG_ENABLE_CD_PARALLEL.value())
                    {
                        LOG_TRACE(DCORE_SPLITTER, "perforating on CD requires reduction - aborting");
                        return;
                    }
                }
            }
        }

        auto roiList = g.GetNodeROIs(node);
        HB_ASSERT(roiList->size() == 1, "dcore split is only supported for a single logical ROI");
        NodeROI& nodeRoi            = roiList->front();
        unsigned perforationDim     = node->getNodeAnnotation().perforation->indexSpaceDim;
        unsigned granularity        = node->getNodeAnnotation().perforation->granularity;
        unsigned dcoreNr            = g.getHALReader()->getNumDcores();
        unsigned perforationDimSize = nodeRoi.size[perforationDim];

        if ((perforationDimSize / granularity) < dcoreNr)
        {
            LOG_TRACE(DCORE_SPLITTER, "aborting split - not enough work to split among all dcores");
            return;
        }

        if ((perforationDimSize % granularity) != 0)
        {
            LOG_TRACE(DCORE_SPLITTER, "aborting split - cant split with leftover smaller than granularity");
            return;
        }

        std::vector<unsigned> dcoreSizes(dcoreNr);
        if (!balanceDcoreSplit(perforationDimSize, granularity, dcoreNr, dcoreSizes) && isMmeNode)
        {
            LOG_TRACE(DCORE_SPLITTER,
                      "Aborting split - was not able to divide MME node evenly between dcores on index space dim {}, "
                      "sizes - {}",
                      perforationDim,
                      toString(dcoreSizes, ','));
            return;
        }

        if (!GCFG_ENABLE_DCORE_LOCALITY_SPLIT.value()) return;

        LOG_DEBUG(DCORE_SPLITTER,
                  "perforating node {} on index space dim {} to sizes {}",
                  node->getNodeName(),
                  perforationDim,
                  toString(dcoreSizes.begin(), dcoreSizes.end(), ','));

        node->getNodeAnnotation().perforation->isPerforated = true;
        node->getNodeAnnotation().splitToLogicalROIs = false;  // logical and dcore split aren't supported together
        nodeRoi.dcoreROIs.resize(dcoreNr);
        unsigned dcoreOffset = 0;
        for (int dcore = 0; dcore < dcoreNr; dcore++)
        {
            std::copy_n(nodeRoi.size, HABANA_DIM_MAX, nodeRoi.dcoreROIs[dcore].size);
            std::copy_n(nodeRoi.baseOffset, HABANA_DIM_MAX, nodeRoi.dcoreROIs[dcore].baseOffset);
            nodeRoi.dcoreROIs[dcore].size[perforationDim] = dcoreSizes[dcore];
            nodeRoi.dcoreROIs[dcore].baseOffset[perforationDim] += dcoreOffset;
            dcoreOffset += dcoreSizes[dcore];
        }

        // indicate for each sliced tensors that it is not also perforated, this will cause it to be allocated in dcore
        // in case it is cached
        unsigned tensorIdx = 0;
        for (const TensorPtr& tensor : node->getInputs())
        {
            if (!tensor || tensor->isShapeTensor() || tensorIdx >= nodeRoi.inputsCacheMetaData.size()) continue;
            setTensorPerforationIndication(tensor, nodeRoi.inputsCacheMetaData[tensorIdx]);
            tensorIdx++;
        }
        tensorIdx = 0;
        for (const TensorPtr& tensor : node->getOutputs())
        {
            if (!tensor || tensor->isShapeTensor() || tensorIdx >= nodeRoi.outputsCacheMetaData.size()) continue;
            setTensorPerforationIndication(tensor, nodeRoi.outputsCacheMetaData[tensorIdx]);
            tensorIdx++;
        }
    }

    void liteCMESingleNode(const NodePtr& node, HabanaGraph& g)
    {
        if (node->isLogicalOperation()) return;

        auto     roiList = g.GetNodeROIs(node);
        NodeROI& nodeRoi = roiList->front();

        unsigned tensorIdx = 0;
        for (const TensorPtr& tensor : node->getInputs())
        {
            if (!tensor || tensor->isShapeTensor() || tensor->isReductionEnabled() ||
                tensorIdx >= nodeRoi.inputsCacheMetaData.size())
                continue;
            setTensorCacheMaintenanceAction(g, node, tensor, nodeRoi.inputsCacheMetaData[tensorIdx], true);
            tensorIdx++;
        }
        tensorIdx = 0;
        for (const TensorPtr& tensor : node->getOutputs())
        {
            if (!tensor || tensor->isShapeTensor() || tensor->isReductionEnabled() ||
                tensorIdx >= nodeRoi.outputsCacheMetaData.size())
                continue;
            setTensorCacheMaintenanceAction(g, node, tensor, nodeRoi.outputsCacheMetaData[tensorIdx], false);
            tensorIdx++;
        }
    }
};

bool splitToDcoreROIs(HabanaGraph& g)
{
    LitePerforation litePer;

    if (GCFG_LITE_CME_MODE.value() > 0 && !GCFG_ENABLE_LAYERED_PIPELINE_BRAIN.value())  // 0 = disabled
    {
        for (const NodePtr& node : g.getExeSortedNodes())
        {
            litePer.liteCMESingleNode(node, g);
        }
    }

    // without locality hints the perforation pass cant operate
    if (GCFG_ENABLE_BRAIN_LOCALITY_HINTS_ANNOTATION.value())
    {
        for (const NodePtr& node : g.getExeSortedNodes())
        {
            litePer.perforateSingleNode(node, g);
        }
    }

    return true;
}
