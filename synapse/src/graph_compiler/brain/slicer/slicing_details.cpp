#include "slicing_details.h"
#include "compilation_hal_reader.h"

namespace gc::layered_brain
{
SolutionParamsPtr SlicingDetails::getQOR(const NodePtr& mme) const
{
    return m_strategy->getNodeQORs(mme);
}

uint64_t SlicingDetails::getNofSlices() const
{
    uint64_t nofSlices = 1;
    for (auto bvdId = 0; bvdId < m_bundleViews->getNumOfBundleViews(); ++bvdId)
    {
        uint64_t nofBvdSlices = m_strategy->getNumOfSlicesForBVD(bvdId, m_bundleViews);
        HB_ASSERT(nofBvdSlices >= 1, "Expecting nofBvdSlices {} >= 1", std::to_string(nofBvdSlices));
        nofSlices *= nofBvdSlices;
    }
    return nofSlices;
}

float SlicingDetails::getMmeUtil(const NodePtr& mme) const
{
    float      mmeUtil = 0;
    const auto qor     = getQOR(mme);
    HB_ASSERT_PTR(qor);
    mmeUtil = qor->perfAttr.mmeUtilization;
    return mmeUtil;
}

float SlicingDetails::getMmeBw(const NodePtr& mme) const
{
    const auto qor = getQOR(mme);
    HB_ASSERT_PTR(qor);
    const auto  optionalInflateForBwBvd = qor->solutionRequirements.bwInflationDim;
    const auto& perfAttr                = qor->perfAttr;
    if (!optionalInflateForBwBvd.has_value())
    {
        // no inflation bvd, bw stays const
        return perfAttr.memoryAttrA.accessBW + perfAttr.memoryAttrB.accessBW + perfAttr.memoryAttrC.accessBW +
               perfAttr.memoryAttrAux.accessBW;
    }
    // optionalInflateForBwBvd.has_value()
    unsigned inflateFactor = m_strategy->getBVDMultiplier(optionalInflateForBwBvd.value()).getInflationFactor();
    HB_ASSERT(inflateFactor > 0,
              "Expecting inflation factor {} > 0 for {}[{}]",
              inflateFactor,
              mme->getNodeName(),
              mme->getNodeTypeStr());

    float mmeBw = 0;
    // TODO [SW-147424]: take into account MMEs max reuse limit
    if (m_bundleViews->isTensorMappedToBVD(mme->getInput(0), optionalInflateForBwBvd.value()))
    {
        // operand a inflated
        mmeBw += perfAttr.memoryAttrA.accessBW;
        mmeBw += (perfAttr.memoryAttrB.accessBW / inflateFactor);
    }
    else
    {
        // operand b inflated
        mmeBw += (perfAttr.memoryAttrA.accessBW / inflateFactor);
        mmeBw += perfAttr.memoryAttrB.accessBW;
    }
    mmeBw += perfAttr.memoryAttrC.accessBW;
    mmeBw += perfAttr.memoryAttrAux.accessBW;
    return mmeBw;
}

std::optional<uint64_t> SlicingDetails::getNodePerforationBvdMultiplier(const NodePtr& n) const
{
    const auto bvd = m_strategy->getPerforationBVDForNode(n);
    if (!bvd.has_value()) return std::nullopt;
    const auto bvdMultiplier = m_strategy->getBVDMultiplier(bvd.value());
    return bvdMultiplier.isSliced() ? bvdMultiplier.getMultiplier()
                                    : m_bundleViews->getBundleView(bvd.value()).resolution;
}

std::optional<float> SlicingDetails::getNodePerforationUtil(const NodePtr& n) const
{
    const auto perforationMultiplier = getNodePerforationBvdMultiplier(n);
    if (!perforationMultiplier.has_value()) return std::nullopt;
    const auto perforationUtil =
        static_cast<float>(perforationMultiplier.value()) /
        round_to_multiple(perforationMultiplier.value(), CompilationHalReader::getHalReader()->getNumDcores());
    return perforationUtil;
}

}  // namespace gc::layered_brain