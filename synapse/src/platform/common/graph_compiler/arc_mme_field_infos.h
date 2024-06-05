#pragma once

#include "arc_dynamic_mme_pp_generator.h"

// TODO move to common smf
#include "platform/gaudi/graph_compiler/smf/smf.h"

namespace arc_platforms
{

class DynamicMmeFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicMmeFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi, ShapeFuncID smfId);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicMmeFieldInfo>(*this); }
};

inline DynamicMmeFieldInfo::DynamicMmeFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi, ShapeFuncID smfId)
: DynamicShapeFieldInfo(fieldIndexOffset,
                        FIELD_DYNAMIC_MME_VALID_ELEMENTS,
                        smfId,
                        origin,
                        roi)
{
    m_size = 1;
}

class MmeCommitFieldInfo : public DynamicShapeFieldInfo
{
public:
    MmeCommitFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<MmeCommitFieldInfo>(*this); }
};

inline MmeCommitFieldInfo::MmeCommitFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi)
: DynamicShapeFieldInfo(fieldIndexOffset, FIELD_DYNAMIC_MME_COMMIT, ShapeFuncID::SMF_GAUDI2_MME_NULL_DESC, origin, roi)
{
    m_size = 1;
}

class MmeSyncFieldInfo : public DynamicShapeFieldInfo
{
public:
    MmeSyncFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<MmeSyncFieldInfo>(*this); }
};

inline MmeSyncFieldInfo::MmeSyncFieldInfo(uint32_t fieldIndexOffset, NodePtr origin, NodeROI* roi)
: DynamicShapeFieldInfo(fieldIndexOffset, FIELD_DYNAMIC_MME_SYNC, ShapeFuncID::SMF_GAUDI2_MME_SYNC, origin, roi)
{
    m_size          = 1;
    m_isUnskippable = true;
}

} // arc_platforms

