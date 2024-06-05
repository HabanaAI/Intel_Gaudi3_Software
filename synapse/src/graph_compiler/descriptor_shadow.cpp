#include <memory>
#include <cstdint>
#include <utility>
#include <algorithm>
#include "descriptor_shadow.h"
#include "habana_global_conf.h"
#include "infra/defs.h"


//-----------------------------------------------------------------------------
//    DescriptorShadow::RegisterProperties implementation
//-----------------------------------------------------------------------------

DescriptorShadow::RegisterProperties DescriptorShadow::RegisterProperties::createFromHandling(RegisterDataHandling handling)
{
    if (handling == RegisterDataHandling::Ignore)               return getIgnore();
    if (handling == RegisterDataHandling::Data)                 return getGeneralData();
    if (handling == RegisterDataHandling::AlwaysWrite)          return getAlwaysWrite();
    if (handling == RegisterDataHandling::Patching)             return getPatching();
    if (handling == RegisterDataHandling::DynamicShapePatching) return getDynamicPatching();
    if (handling == RegisterDataHandling::AlwaysWritePatching)  return getAlwaysWritePatching();
    if (handling == RegisterDataHandling::OptOutPatching)       return getOptOutPatching();
    HB_ASSERT(handling == RegisterDataHandling::Banned, "handling must be Banned");
    return getBanned();
}

DescriptorShadow::RegisterDataHandling DescriptorShadow::RegisterProperties::getHandling() const
{
    HB_ASSERT(!banned || (!alwaysWrite && !patching && !dynamicPatching && !alwaysWritePatching && !optOutPatching),
              "Banned can't co-exist with AlwaysWrite, Patching, DynamicPatching, alwaysWritePatching or optOutPatching");

    // Order matters! Banned is "stronger" than AlwaysWritePatching, which is "stronger" than OptOutPatching, which is...
    if (banned)              return RegisterDataHandling::Banned;
    if (alwaysWritePatching) return RegisterDataHandling::AlwaysWritePatching;
    if (optOutPatching)      return RegisterDataHandling::OptOutPatching;
    if (dynamicPatching)     return RegisterDataHandling::DynamicShapePatching;
    if (patching)            return RegisterDataHandling::Patching;
    if (alwaysWrite)         return RegisterDataHandling::AlwaysWrite;
    if (ignore)              return RegisterDataHandling::Ignore;
    return RegisterDataHandling::Data;
}

void DescriptorShadow::RegisterProperties::addHandling(RegisterDataHandling handling)
{
    switch (handling)
    {
    case RegisterDataHandling::Ignore:
        ignore = true;
        break;
    case RegisterDataHandling::DynamicShapePatching:
        HB_ASSERT(!banned, "DynamicShapePatching can't co-exist with Banned");
        dynamicPatching = true;
        patching = true;
        break;
    case RegisterDataHandling::AlwaysWritePatching:
        HB_ASSERT(!banned, "AlwaysWritePatching can't co-exist with Banned");
        alwaysWritePatching = true;
        patching            = true;
        break;
    case RegisterDataHandling::AlwaysWrite:
        HB_ASSERT(!banned, "AlwaysWrite can't co-exist with Banned");
        alwaysWrite = true;
        break;
    case RegisterDataHandling::Patching:
        HB_ASSERT(!banned, "Patching can't co-exist with Banned");
        patching = true;
        break;
    case RegisterDataHandling::OptOutPatching:
        HB_ASSERT(!banned && !alwaysWrite && !alwaysWritePatching, "OptOutPatching can't co-exist with Banned or forced write");
        optOutPatching = true;
        break;
    case RegisterDataHandling::Banned:
        HB_ASSERT(!alwaysWrite && !patching, "Banned can't co-exist with AlwaysWrite or Patching");
        banned = true;
        break;
    default:
        break; // Data
    }
}

std::ostream& operator<<(std::ostream& stream, const DescriptorShadow::RegisterProperties& rhs)
{
    stream << (rhs.banned ?              "b" : "-");
    stream << (rhs.patching ?            "p" : "-");
    stream << (rhs.dynamicPatching ?     "d" : "-");
    stream << (rhs.alwaysWrite ?         "w" : "-");
    stream << (rhs.ignore ?              "i" : "-");
    stream << (rhs.alwaysWritePatching ? "a" : "-");
    stream << (rhs.optOutPatching ?      "o" : "-");
    return stream;
}

bool DescriptorShadow::RegisterProperties::operator==(const RegisterProperties& rhs) const
{
    return (ignore == rhs.ignore) && (alwaysWrite == rhs.alwaysWrite) && (banned == rhs.banned) &&
           (patching == rhs.patching) && (dynamicPatching == rhs.dynamicPatching) &&
           (alwaysWritePatching == rhs.alwaysWritePatching) && (optOutPatching == rhs.optOutPatching);
}


//-----------------------------------------------------------------------------
//    DescriptorShadow implementation
//-----------------------------------------------------------------------------

void DescriptorShadow::setRegisterPropertyOnSegment(
    std::vector<RegisterProperties>& props,
    const Segment&                   seg,
    const RegisterProperties&        prop)
{
    if (seg.end() > props.size())
    {
        props.resize(seg.end());
    }
    std::fill(props.begin() + seg.start(), props.begin() + seg.end(), prop);
}

DescriptorShadow::AllRegistersProperties DescriptorShadow::createPropertiesToAllRegs(
    uint32_t                                   totalsize,
    std::initializer_list<StartEndAndHandling> listHandling)
{
    auto data = std::make_shared<std::vector<RegisterProperties>>(totalsize);
    for (auto& it : listHandling)
    {
        uint32_t             start, end;
        RegisterDataHandling handling;
        std::tie(start, end, handling) = it;
        auto prop = RegisterProperties::createFromHandling(handling);
        setRegisterPropertyOnSegment(*data, Segment{start, end}, prop);
    }
    return data;
}

DescriptorShadow::AllRegistersProperties DescriptorShadow::createPropertiesToAllRegs(
    uint32_t                                                         totalsize,
    std::initializer_list<std::tuple<Segment, RegisterDataHandling>> listHandling)
{
    auto data = std::make_shared<std::vector<RegisterProperties>>(totalsize);
    for (auto& it: listHandling)
    {
        Segment seg;
        RegisterDataHandling handling;
        std::tie(seg, handling) = it;
        auto prop = RegisterProperties::createFromHandling(handling);
        setRegisterPropertyOnSegment(*data, seg, prop);
    }
    return data;
}

DescriptorShadow::AllRegistersProperties DescriptorShadow::createPropertiesToAllRegsByEnds(
    uint32_t                              totalsize,
    std::initializer_list<EndAndHandling> listHandling)
{
    auto data = std::make_shared<std::vector<RegisterProperties>>(totalsize);
    uint32_t prev = 0;
    for (auto& pair : listHandling)
    {
        auto prop = RegisterProperties::createFromHandling(pair.second);
        setRegisterPropertyOnSegment(*data, Segment{prev, pair.first}, prop);
        prev = pair.first;
    }
    return data;
}

DescriptorShadow::RegisterDataHandling DescriptorShadow::getRegHandling(uint32_t index) const
{
    return (!m_allRegProperties || index >= m_allRegProperties->size()) ?
        RegisterDataHandling::Undefined : m_allRegProperties->at(index).getHandling();
}

DescriptorShadow::RegisterDataHandling DescriptorShadow::getPastRegHandling(uint32_t index) const
{
    return (!m_loadedRegProperties || index >= m_loadedRegProperties->size()) ?
        RegisterDataHandling::Undefined : m_loadedRegProperties->at(index).getHandling();
}

DescriptorShadow::WriteType DescriptorShadow::getWriteType(uint32_t val, uint32_t regIndex) const
{
    auto handling     = getRegHandling(regIndex);
    auto pastHandling = getPastRegHandling(regIndex);

    if (handling == RegisterDataHandling::Patching)
    {
        return WriteType::WritePatching;
    }
    if (handling == RegisterDataHandling::DynamicShapePatching)
    {
        return WriteType::WritePatching;
    }
    if (handling == RegisterDataHandling::AlwaysWritePatching)
    {
        return WriteType::WritePatching;
    }
    if (handling == RegisterDataHandling::AlwaysWrite)
    {
        return WriteType::WriteExecute;
    }
    if (handling == RegisterDataHandling::Data && (isPatching(pastHandling) || isSkippable(pastHandling)))
    {
        return WriteType::WriteExecute;
    }
    if (isSkippable(handling))
    {
        return WriteType::NoWrite;
    }
    if (val == getDataAt(regIndex) && !GCFG_DISABLE_LOAD_DIFF_DESC.value()) // note that here we compare to the history
    {
        return WriteType::NoWrite;
    }
    return WriteType::WriteExecute;
}

void DescriptorShadow::addRegHandlingAt(uint32_t index, RegisterDataHandling handling) const
{
    HB_ASSERT(index < m_allRegProperties->size(), "Index out of range");
    return m_allRegProperties->at(index).addHandling(handling);
}

void DescriptorShadow::invalidateRegs(std::vector<uint32_t>& indicies)
{
    for (auto index : indicies)
    {
        m_loadedRegsValidityMask[index] = false;
    }
}

void DescriptorShadow::updateLoadedReg(uint32_t index, uint32_t data)
{
    ensureVectorBigEnough(index);
    m_data[index] = data;

    // If the handling is dynamic shapes patching - we can't relay on the value in the descriptor
    // because it may change in runtime. In this case we "flush" the history.
    m_loadedRegsValidityMask[index] = (getRegHandling(index) != RegisterDataHandling::DynamicShapePatching);
}

// end is non-inclusive
void DescriptorShadow::updateLoadedSegment(uint32_t start, uint32_t end, const uint32_t* data)
{
    ensureVectorBigEnough(end - 1);
    std::copy(data, data + (end - start), m_data.begin() + start);

    for (size_t i = start; i < end; i++)
    {
        // If the handling is dynamic shapes patching - we can't relay on the value in the descriptor
        // because it may change in runtime. In this case we "flush" the history.
        m_loadedRegsValidityMask[i] = (getRegHandling(i) != RegisterDataHandling::DynamicShapePatching);
    }
}

bool DescriptorShadow::canJoin(const Segment& seg1, const Segment& seg2, unsigned offset) const
{
    // Assumes ranges have consistent property patching/none-patching wise, so it's enough to test only one register
    if (isPatching(getRegHandling(offset + seg1.end() - 1)) != isPatching(getRegHandling(offset + seg2.end() - 1)))
    {
        return false;
    }
    HB_ASSERT(!seg1.isOverlap(seg2) && seg1.end() <= seg2.start(), "Invalid segments");
    return std::all_of(
        m_allRegProperties->begin() + offset + seg1.end(),
        m_allRegProperties->begin() + offset + seg2.start(),
        [](RegisterProperties maskValue) {
            // Banned and OptOutPatching should not be written
            return (maskValue.getHandling() != RegisterDataHandling::Banned &&
                    maskValue.getHandling() != RegisterDataHandling::OptOutPatching); });
}

bool DescriptorShadow::isPatching(RegisterDataHandling handling) const
{
    return (handling == RegisterDataHandling::Patching) ||
           (handling == RegisterDataHandling::DynamicShapePatching) ||
           (handling == RegisterDataHandling::AlwaysWritePatching) ||
           (handling == RegisterDataHandling::OptOutPatching); // register is skipped, still it was patched
}

bool DescriptorShadow::isSkippable(RegisterDataHandling handling) const
{
    return (handling == RegisterDataHandling::Ignore) ||
           (handling == RegisterDataHandling::Banned) ||
           (handling == RegisterDataHandling::OptOutPatching); // OptOutPatching marks a dropped patching register
}

void DescriptorShadow::flush()
{
    m_data.clear();
    m_loadedRegsValidityMask.clear();
}

Settable<uint32_t> DescriptorShadow::getDataAt(uint32_t index) const
{
    if (m_data.size() <= index) return nullset;
    if (!m_loadedRegsValidityMask[index]) return nullset;
    return m_data[index];
}

void DescriptorShadow::ensureVectorBigEnough(uint32_t index)
{
    if (m_data.size() <= index)
    {
        m_data.resize(index + 1);
        m_loadedRegsValidityMask.resize(index + 1);
    }
}

DescriptorShadow::RegisterProperties DescriptorShadow::propertiesAt(uint32_t index) const
{
    HB_ASSERT(m_allRegProperties != nullptr && index < m_allRegProperties->size(), "bad input");
    return m_allRegProperties->at(index);
}

void DescriptorShadow::setPropertiesAt(uint32_t index, const RegisterProperties& prop)
{
    HB_ASSERT(m_allRegProperties != nullptr && index < m_allRegProperties->size(), "bad input");
    m_allRegProperties->at(index) = prop;
}

void DescriptorShadow::printAllHandling(std::ostream& stream)
{
    unsigned i = 0;
    for (auto h : *m_allRegProperties)
    {
        stream << "[" << i++ << "]" << h << ",";
    }
    stream << std::endl;
}
