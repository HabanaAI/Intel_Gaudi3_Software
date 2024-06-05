#pragma once

#include <string>
#include <vector>
#include <list>
#include "hlgcfg_default_item.hpp"
#include "hlgcfg_defs.hpp"

#include "size_param.hpp"
#include "hlgcfg_item_interface.hpp"

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{
class GcfgItemObserver;
using ObserversList = std::vector<GcfgItemObserver*>;

class GcfgItem : public GcfgItemInterface
{
public:
    GcfgItem(const std::string&              name,
             const std::vector<std::string>& aliases,
             const std::string&              description,
             bool                            isPublic  = false,
             ObserversList                   observers = {});
    GcfgItem(const std::string& name,
             const std::string& description,
             bool               isPublic  = false,
             ObserversList      observers = {});

    virtual ~GcfgItem();

    // No copy for global configuration
    GcfgItem(const GcfgItem&) = delete;
    void operator=(const GcfgItem&) = delete;

    const std::string& primaryName() const override
    {
        return m_names.front();
    }
    const std::vector<std::string>& aliases() const override
    {
        return m_names;
    }
    const std::string& description() const override
    {
        return m_description;
    }

    // Get the env variable used or "" if none was used
    const std::string& getUsedEnvAlias() const override
    {
        return m_usedEnvAliasName;
    }
    using GcfgItemInterface::setFromString;
    VoidOutcome setFromString(const std::string& str, GcfgSource setFrom, bool fromObserver = false) override
    {
        auto ret = setFromString(str, fromObserver);
        m_setFrom = setFrom;
        return ret;
    }

    bool isPublic() const override
    {
        return m_isPublic;
    }

    bool isSetFromDefault() const override
    {
        return m_setFrom == GcfgSource::default_src;
    }

    bool isSetFromObserver() const
    {
        return m_setFrom == GcfgSource::observer;
    }

    bool isSetFromUserConfig() const
    {
        return m_setFrom == GcfgSource::env || m_setFrom == GcfgSource::file;
    }

    std::string getSourceStr() const override
    {
        return toString(m_setFrom);
    }

    VoidOutcome updateFromEnv(bool enableExperimental) override;
protected:
    VoidOutcome updateObservers(const std::string& value);
    GcfgSource m_setFrom;

private:
    const std::vector<std::string>   m_names;  // A primary name followed by optional aliases
    const std::string                m_description;
    std::string                      m_usedEnvAliasName;  // Env variable alias used
    std::string                      m_usedEnvValue;      // Env variable value used
    const bool                       m_isPublic;
    bool                             m_shouldUnregister;
protected:
    std::vector<GcfgItemObserver*> m_observers;
};

template<class T, class R = T>
class GcfgItemImpl : public GcfgItem
{
public:
    GcfgItemImpl(const std::string&              name,
                 const std::vector<std::string>& names,
                 const std::string&              description,
                 const GcfgDefaultItem<T>&       defaultValue,
                 bool                            isPublic  = false,
                 ObserversList                   observers = {});
    GcfgItemImpl(const std::string&        name,
                 const std::string&        description,
                 const GcfgDefaultItem<T>& defaultValue,
                 bool                      isPublic  = false,
                 ObserversList             observers = {});

    const R& value() const;

    const R& getDefaultValue(uint32_t device) const;

    VoidOutcome setValue(const T& value);
    using GcfgItem::setFromString;
    VoidOutcome setFromString(const std::string& str, bool fromObserver = false) override;

    std::string getValueStr() const override;

    std::string getDefaultValuesStr() const override;

private:
    template<class U>
    VoidOutcome setValue(const U& value, bool fromObserver);

    T m_value;
    GcfgDefaultItem<T> m_defaultValue;

    virtual void reset() override;
};

typedef GcfgItemImpl<int64_t>                      GcfgItemInt64;
typedef GcfgItemImpl<uint64_t>                     GcfgItemUint64;
typedef GcfgItemImpl<bool>                         GcfgItemBool;
typedef GcfgItemImpl<float>                        GcfgItemFloat;
typedef GcfgItemImpl<std::string>                  GcfgItemString;
typedef GcfgItemImpl<hl_gcfg::SizeParam, uint64_t> GcfgItemSize;

template<>
const uint64_t& GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::value() const;

template<>
const uint64_t& GcfgItemImpl<hl_gcfg::SizeParam, uint64_t>::getDefaultValue(uint32_t device) const;

constexpr bool MakePublic  = true;
constexpr bool MakePrivate = !MakePublic;
}}
#include "impl/hlgcfg_item.inl"
