#include "dynamic_patch_point_optimizer.h"
#include <memory>

DynamicPatchPointOptimizer::DynamicPatchPointOptimizer()
{
    for (unsigned int& patchPointPriority : m_patchPointPriorities)
    {
        patchPointPriority = MIN_PRIORITY;
    }

    // TPC sizes and Addresses are prioritized over dynamic execution because even if the ROI is fully inside,
    // and the dynamic execute is going to bypass - these values have to be patched to the correct value.
    m_patchPointPriorities[FIELD_DYNAMIC_TPC_SIZE] = MAX_PRIORITY;
    m_patchPointPriorities[FIELD_DYNAMIC_ADDRESS] = MAX_PRIORITY;

    // Prioritize the dynamic execute next, as it determines if the other PPs should even be run.
    m_patchPointPriorities[FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL] = MAX_PRIORITY - 1;
    m_patchPointPriorities[FIELD_DYNAMIC_EXECUTE_NO_SIGNAL] = m_patchPointPriorities[FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL];
    m_patchPointPriorities[FIELD_DYNAMIC_EXECUTE_MME] = m_patchPointPriorities[FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL];
}

void DynamicPatchPointOptimizer::optimizePatchPoints(std::vector<DynamicPatchPointPtr>& patchPoints)
{
    if (patchPoints.empty())
    {
        return;
    }

    std::vector<std::pair<uint32_t,uint32_t>> roiOffsets = getRoiOffsets(patchPoints);

    for (auto& pair : roiOffsets)
    {
        sortPatchPoints(patchPoints, pair.first, pair.second);
    }
};

std::vector<std::pair<uint32_t, uint32_t>> DynamicPatchPointOptimizer::getRoiOffsets(std::vector<DynamicPatchPointPtr>& patchPoints)
{
    std::vector<std::pair<uint32_t,uint32_t>> roiOffsets;
    uint32_t offsetStart = 0;
    uint32_t currRoi = patchPoints[0]->getRoiIndex();

    for (int i = 0; i < patchPoints.size(); i++)
    {
        if (patchPoints[i]->getRoiIndex() != currRoi)
        {
            roiOffsets.emplace_back(offsetStart, i - 1);
            offsetStart = i;
            currRoi = patchPoints[i]->getRoiIndex();
        }
    }
    roiOffsets.emplace_back(offsetStart, patchPoints.size() - 1);

    return roiOffsets;
}

void DynamicPatchPointOptimizer::sortPatchPoints(std::vector<DynamicPatchPointPtr>& patchPoints, uint32_t start_index, uint32_t end_index)
{
    std::stable_sort(patchPoints.begin() + start_index, patchPoints.begin() + end_index + 1,
         ([&](const DynamicPatchPointPtr& first, const DynamicPatchPointPtr& second)
         {
             return m_patchPointPriorities[first->getFieldType()] > m_patchPointPriorities[second->getFieldType()];
         }));
}