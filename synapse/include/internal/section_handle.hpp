#pragma once

#include "synapse_api_types.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

struct InternalSectionHandle
{
public:
    using SectionIDType     = uint32_t;
    using GraphIDType       = uint32_t;
    using GroupType         = uint8_t;

    explicit InternalSectionHandle(GraphIDType graphID) : m_graphID(graphID) {}
    InternalSectionHandle(GraphIDType graphID, const InternalSectionHandle& other)
    : m_sectionId(other.m_sectionId),
      m_graphID(graphID),
      m_group(other.m_group),
      m_persistent(other.m_persistent),
      m_const(other.m_const),
      m_rmw(other.m_rmw)
    {
    }

    SectionIDType getID() const
    {
        return m_sectionId.value_or((SectionIDType)-1);
    }

    GraphIDType      getGraphID() const { return m_graphID; }
    GroupType        getGroup() const { return m_group; }
    bool             getPersistent() const { return m_persistent; }
    bool             getConst() const { return m_const; }
    bool             getRMW() const { return m_rmw; }
    synSectionHandle getSectionHandle() const { return m_sectionHandle; }

    bool isLocked() const { return m_sectionId.has_value(); }

    // All setters return success value (fail if the section is already locked)
    bool setIDAndLock(SectionIDType id);
    bool setGroup(GraphIDType group);
    bool setPersistent(bool isPersistent);
    bool setConst(bool isConst);
    bool setRMW(bool isRMW);
    bool setSectionHandle(synSectionHandle sectionHandle);

private:
    // Sequential running index, when set it should not change to prevent inconsistencies
    std::optional<SectionIDType> m_sectionId {};

    const GraphIDType m_graphID;                  // The graph handle that this section is assigned to.
    GroupType         m_group         = 0;        // The section group
    bool              m_persistent    = true;     // The section is used to hold persistent tensors
    bool              m_const         = false;    // The section used to hold the weights
    bool              m_rmw           = false;    // Writes to the section tensors can use RMW writes
    synSectionHandle  m_sectionHandle = nullptr;  // The section handle of this section.

    bool safeModify(std::function<void()> modifier, const std::string& actionName);
};