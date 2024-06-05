#pragma once
#include <array>
#include <string>
#include <optional>
#include "size_param.hpp"
#include "hlgcfg_defs.hpp"

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{

// MUST BE NOT LESS THAN THE NUMBER OF SUPPORTED DEVICES IN SYNAPSE
const uint32_t maxDeviceType = 10;

template<class T>
class GcfgDefaultItem
{
public:
    // Global default CTOR
    GcfgDefaultItem(const T& defaultVal);
    virtual ~GcfgDefaultItem() = default;

    GcfgDefaultItem& operator<<(std::pair<uint32_t, T> const & value);

    // Get default value for device
    const T& value(uint32_t device) const;

    std::string getValueStr() const;

private:
    std::optional<T>  m_default;
    std::array<std::optional<T>, maxDeviceType> m_defaultsPerDevice;
};

using DfltInt64  = GcfgDefaultItem<int64_t>;
using DfltUint64 = GcfgDefaultItem<uint64_t>;
using DfltBool   = GcfgDefaultItem<bool>;
using DfltFloat  = GcfgDefaultItem<float>;
using DfltString = GcfgDefaultItem<std::string>;
using DfltSize   = GcfgDefaultItem<hl_gcfg::SizeParam>;

template <class T>
std::pair<uint32_t, T> deviceValue(uint32_t deviceType, T value)
{
    return std::pair(deviceType, std::move(value));
}
}}
#include "impl/hlgcfg_default_item.inl"

