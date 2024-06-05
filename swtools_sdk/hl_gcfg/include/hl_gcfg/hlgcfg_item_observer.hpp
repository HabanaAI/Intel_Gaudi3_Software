#pragma once

#include <map>
#include <string>
#include "hlgcfg_defs.hpp"

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{
class GcfgItem;

class GcfgItemObserver
{
public:
    GcfgItemObserver() = delete;
    GcfgItemObserver(GcfgItem*                          observer,
                     std::map<std::string, std::string> subjectToObserverMap,
                     bool                               supportUnknownValue = false);

    VoidOutcome      updateValue(const std::string& subjectValue);
    const std::string& primaryName() const;

private:
    GcfgItem*                          m_observer;
    std::map<std::string, std::string> m_subjectToObserverMap;
    bool                               m_supportUnknownValue;
};
}}