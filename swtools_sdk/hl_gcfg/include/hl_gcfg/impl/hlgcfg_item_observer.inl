#pragma once

#include "../hlgcfg_item_observer.hpp"

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{

inline
GcfgItemObserver::GcfgItemObserver(GcfgItem*                              observer,
                                       std::map<std::string, std::string> subjectToObserverMap,
                                       bool                               supportUnknownValue)
: m_observer(observer)
, m_subjectToObserverMap(std::move(subjectToObserverMap))
, m_supportUnknownValue(supportUnknownValue)
{
}

inline
const std::string& GcfgItemObserver::primaryName() const
{
    return m_observer->primaryName();
}

inline
VoidOutcome GcfgItemObserver::updateValue(const std::string& subjectValue)
{
    auto it = m_subjectToObserverMap.find(subjectValue);
    if (it != m_subjectToObserverMap.end())
    {
        if (m_observer->isSetFromDefault() || m_observer->isSetFromObserver())
        {
            return m_observer->setFromString(it->second, true);
        }
        else
        {
            HLGCFG_LOG_WARN("{}: value is not set because is already set from {}",
                            m_observer->primaryName(),
                            m_observer->getSourceStr());
        }
    }
    else if (!m_supportUnknownValue)
    {
        HLGCFG_LOG_CRITICAL("Got invalid string value \"{}\" for observer global conf {}.",
                            subjectValue,
                            m_observer->primaryName());
        HLGCFG_RETURN_ERR(invalidString, "invalid string value \"{}\" for observer global conf {}.",
                          subjectValue,
                          m_observer->primaryName());
    }
    return {};
}
}}