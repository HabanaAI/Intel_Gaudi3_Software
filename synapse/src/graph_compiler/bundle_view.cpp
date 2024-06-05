#include "bundle_view.h"
#include "node.h"

BundleViewContainer::BundleViewContainer(uint32_t numOfBundleViews)
{
    m_bundleViews.resize(numOfBundleViews);
    for (auto i = 0; i < m_bundleViews.size(); i++)
    {
        m_bundleViews[i].id = i;
    }
}

void BundleViewContainer::logBundleViews() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(LB_SLICER)) return;

    for (const auto& bvd : m_bundleViews)
    {
        LOG_DEBUG(LB_SLICER, "###### BVD {} #####", bvd.id);
        LOG_DEBUG(LB_SLICER, "\t BVD resolution (max multiplier): {}", bvd.resolution);
        LOG_DEBUG(LB_SLICER, "\t Tensor dims and granularity: ");
        for (const auto& [tensorDim, granularity] : bvd.tensorDimsGranularity)
        {
            LOG_DEBUG(LB_SLICER,
                      "\t\t [{} , {}] : granularity = {} tensor dim size = {}",
                      tensorDim.first->getName(),
                      tensorDim.second,
                      granularity,
                      tensorDim.first->getSizeInElements(tensorDim.second));
        }
        LOG_DEBUG(LB_SLICER, "\t Node dims and granularity: ");
        for (const auto& [nodeDim, granularity] : bvd.nodeDimsGranularity)
        {
            LOG_DEBUG(LB_SLICER,
                      "\t\t [{} , {}] : granularity = {} node dim size = {}",
                      nodeDim.first->getNodeName(),
                      nodeDim.second,
                      granularity,
                      nodeDim.first->getNodeAccessPattern()->getNodeResolution()[nodeDim.second]);
        }
    }
}

uint32_t BundleViewContainer::getNumOfBundleViews() const
{
    return m_bundleViews.size();
}

const BundleView& BundleViewContainer::getBundleView(BundleViewId bvdId) const
{
    HB_ASSERT(bvdId < m_bundleViews.size(), "Invalid BVD id {}", bvdId);
    return m_bundleViews[bvdId];
}

BundleViewId BundleViewContainer::getBVDForTensorDim(const TensorPtr& tensor, Dim tensorDim) const
{
    auto it = m_tensorDimToBvd.find({tensor, tensorDim});
    HB_ASSERT(it != m_tensorDimToBvd.end(), "Missing BVD for tensor {} dim {}", tensor->getName(), tensorDim);
    return it->second;
}

BundleViewId BundleViewContainer::getBVDForNodeDim(const NodePtr& node, Dim nodeDim) const
{
    auto it = m_nodeDimToBvd.find({node, nodeDim});
    HB_ASSERT(it != m_nodeDimToBvd.end(), "Missing BVD for node {} dim {}", node->getNodeName(), nodeDim);
    return it->second;
}

bool BundleViewContainer::isNodeDimMappedToBVD(const NodePtr& node, Dim nodeDim) const
{
    return m_nodeDimToBvd.find({node, nodeDim}) != m_nodeDimToBvd.end();
}

bool BundleViewContainer::isTensorMappedToBVD(const TensorPtr& tensor, BundleViewId bvdId) const
{
    for (Dim dim = 0; dim < tensor->getDim(); ++dim)
    {
        const auto it = m_tensorDimToBvd.find({tensor, dim});
        if (it != m_tensorDimToBvd.end() && it->second == bvdId)
        {
            return true;
        }
    }
    return false;
}

std::vector<Dim> BundleViewContainer::getNodeDimsInBVD(BundleViewId bvdId, const NodePtr& node) const
{
    std::vector<Dim> nodeDimsInBVD;
    for (const auto& nodeDimGranularity : getBundleView(bvdId).nodeDimsGranularity)
    {
        if (nodeDimGranularity.first.first == node)
        {
            nodeDimsInBVD.push_back(nodeDimGranularity.first.second);
        }
    }
    return nodeDimsInBVD;
}

BVDSet BundleViewContainer::getNodesBVDs(const NodeVector& nodes) const
{
    BVDSet bvds;
    for (const auto& node : nodes)
    {
        const auto& ap = node->getNodeAccessPattern();
        HB_ASSERT_PTR(ap);
        for (Dim nodeDim = 0; nodeDim < ap->getNodeResolution().size(); nodeDim++)
        {
            if (isNodeDimMappedToBVD(node, nodeDim))
            {
                bvds.insert(getBVDForNodeDim(node, nodeDim));
            }
        }
    }
    return bvds;
}

TensorTile::Size BundleViewContainer::getGranularityForTensorDim(const TensorPtr& tensor, Dim tensorDim) const
{
    BundleViewId bvdId = getBVDForTensorDim(tensor, tensorDim);
    HB_ASSERT(bvdId < m_bundleViews.size(), "Invalid BVD {} for tensor {} dim {}", bvdId, tensor->getName(), tensorDim);
    auto it = m_bundleViews[bvdId].tensorDimsGranularity.find({tensor, tensorDim});
    HB_ASSERT(it != m_bundleViews[bvdId].tensorDimsGranularity.end(),
              "Missing granularity for tensor {} dim {}",
              tensor->getName(),
              tensorDim);
    return it->second;
}

NodeTile::Size BundleViewContainer::getGranularityForNodeDim(const NodePtr& node, Dim nodeDim) const
{
    BundleViewId bvdId = getBVDForNodeDim(node, nodeDim);
    HB_ASSERT(bvdId < m_bundleViews.size(), "Invalid BVD {} for node {} dim {}", bvdId, node->getNodeName(), nodeDim);
    auto it = m_bundleViews[bvdId].nodeDimsGranularity.find({node, nodeDim});
    HB_ASSERT(it != m_bundleViews[bvdId].nodeDimsGranularity.end(),
              "Missing granularity for node {} dim {}",
              node->getNodeName(),
              nodeDim);
    return it->second;
}

void BundleViewContainer::mapTensorDimToBVD(const TensorPtr& tensor,
                                            Dim              tensorDim,
                                            BundleViewId     bvdId,
                                            TensorTile::Size granularity)
{
    HB_ASSERT(bvdId < m_bundleViews.size(), "Invalid BVD {} for tensor {} dim {}", bvdId, tensor->getName(), tensorDim);
    m_bundleViews[bvdId].tensorDimsGranularity[{tensor, tensorDim}] = granularity;
    m_tensorDimToBvd[{tensor, tensorDim}]                           = bvdId;
}

void BundleViewContainer::mapNodeDimToBVD(const NodePtr& node,
                                          Dim            nodeDim,
                                          BundleViewId   bvdId,
                                          NodeTile::Size granularity)
{
    HB_ASSERT(bvdId < m_bundleViews.size(), "Invalid BVD {} for node {} dim {}", bvdId, node->getNodeName(), nodeDim);
    const auto& nodeAccessPattern = node->getNodeAccessPattern();
    HB_ASSERT_PTR(nodeAccessPattern);
    uint64_t resolution = div_round_up(nodeAccessPattern->getNodeResolution()[nodeDim], granularity);
    if (m_bundleViews[bvdId].nodeDimsGranularity.empty())
    {
        // First node dim in BVD - update BVD resolution
        m_bundleViews[bvdId].resolution = resolution;
    }
    else
    {
        // Make sure all node dims for all nodes mapped to this BVD have the same resolution
        HB_ASSERT(m_bundleViews[bvdId].resolution == resolution,
                  "Invalid resolution for node {} dim {} (BVD id {})",
                  node->getNodeName(),
                  nodeDim,
                  bvdId);
    }
    m_bundleViews[bvdId].nodeDimsGranularity[{node, nodeDim}] = granularity;
    m_nodeDimToBvd[{node, nodeDim}]                           = bvdId;
}

BVDSet BundleViewContainer::getBvdsForNodeDims(const NodePtr& node, const std::unordered_set<Dim>& nodeDims) const
{
    BVDSet bvds;
    for (auto nodeDim : nodeDims)
    {
        if (isNodeDimMappedToBVD(node, nodeDim))
        {
            bvds.insert(getBVDForNodeDim(node, nodeDim));
        }
    }
    return bvds;
}