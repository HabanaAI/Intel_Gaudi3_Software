#pragma once
#include "recipe_patch_point.h"


class DynamicPatchPointOptimizer
{
public:
    DynamicPatchPointOptimizer();
    void optimizePatchPoints(std::vector<DynamicPatchPointPtr>& patchPoints);
private:

    std::vector<std::pair<uint32_t, uint32_t>> getRoiOffsets(std::vector<DynamicPatchPointPtr>& patchPoints);
    void sortPatchPoints(std::vector<DynamicPatchPointPtr>& patchPoints, uint32_t start_index, uint32_t end_index);

    uint32_t m_patchPointPriorities[EFieldType::_LAST];

    static constexpr uint32_t MAX_PRIORITY = 100;
    static constexpr uint32_t MIN_PRIORITY = 0;
};
