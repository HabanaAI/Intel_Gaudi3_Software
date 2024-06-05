#include "device_agnostic_recipe_info.hpp"

uint32_t SignalFromGraphInfo::getExtTensorExeOrderByExtTensorIdx(uint32_t tensorIdx) const
{
    auto itr = m_sfgExtTensorIdxToExeOrder.find(tensorIdx);
    if (itr == m_sfgExtTensorIdxToExeOrder.end())
    {
        return TENSOR_EXE_ORDER_INVALID;
    }
    return itr->second;
}