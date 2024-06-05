#include "scal_types.hpp"

bool ScalLongSyncObject::operator!=(const ScalLongSyncObject& o) const
{
    return ((m_index != o.m_index) || (m_targetValue != o.m_targetValue));
}
