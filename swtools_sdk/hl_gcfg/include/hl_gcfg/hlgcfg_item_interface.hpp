#pragma once

#include <string>
#include <vector>
#include "hlgcfg_defs.hpp"

namespace hl_gcfg{
HLGCFG_NAMESPACE{
enum class GcfgSource
{
    default_src = 0,
    env,
    file,
    observer,
    runtime,
    MAX_SIZE
};

HLGCFG_API const std::string& toString(GcfgSource src);

class GcfgItemInterface
{
public:
    virtual const std::string& primaryName() const             = 0;
    virtual const std::vector<std::string>& aliases() const    = 0;
    virtual const std::string& description() const             = 0;
    virtual VoidOutcome setFromString(const std::string& str, bool fromObserver = false)                     = 0;
    virtual VoidOutcome setFromString(const std::string& str, GcfgSource setFrom, bool fromObserver = false) = 0;
    virtual std::string getValueStr() const                    = 0;
    virtual std::string getDefaultValuesStr() const            = 0;
    virtual std::string getSourceStr() const                   = 0;
    virtual const std::string& getUsedEnvAlias() const         = 0;
    virtual VoidOutcome updateFromEnv(bool enableExperimental) = 0;
    virtual bool isPublic() const                              = 0;
    virtual bool isSetFromDefault() const                      = 0;
protected:
    virtual void reset() = 0;
    friend void reset();
};
}}
