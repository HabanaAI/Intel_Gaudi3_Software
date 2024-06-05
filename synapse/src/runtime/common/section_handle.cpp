#include "section_handle.hpp"
#include "defs.h"
#include "log_manager.h"

#include <functional>

bool InternalSectionHandle::setGroup(InternalSectionHandle::GraphIDType group)
{
    return safeModify([&]() { m_group = group; }, __FUNCTION__);
}

bool InternalSectionHandle::setPersistent(bool isPersistent)
{
    return safeModify([&]() { m_persistent = isPersistent; }, __FUNCTION__);
}

bool InternalSectionHandle::setConst(bool isConst)
{
    return safeModify([&]() { m_const = isConst; }, __FUNCTION__);
}

bool InternalSectionHandle::setRMW(bool isRMW)
{
    return safeModify([&]() { m_rmw = isRMW; }, __FUNCTION__);
}

bool InternalSectionHandle::setSectionHandle(synSectionHandle sectionHandle)
{
    return safeModify([&]() { m_sectionHandle = sectionHandle; }, __FUNCTION__);
}

bool InternalSectionHandle::setIDAndLock(InternalSectionHandle::SectionIDType id)
{
    return safeModify([&]() { m_sectionId = id; }, __FUNCTION__);
}

bool InternalSectionHandle::safeModify(std::function<void()> modifier, const std::string& actionName)
{
    if (isLocked())
    {
        LOG_ERR(SYN_API,
                "{}: Section Handle 0x{:x} was already used to create a tensor and can not be modified",
                actionName,
                m_sectionId.value());
        return false;
    }
    modifier();
    return true;
}