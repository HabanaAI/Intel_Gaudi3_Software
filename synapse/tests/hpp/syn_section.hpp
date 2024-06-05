#pragma once

#include "syn_object.hpp"

namespace syn
{
class Section : public SynObject<synSectionHandle>
{
public:
    Section() : SynObject(nullptr) {}

    void setPersistent(bool persistent) const { SYN_CHECK(synSectionSetPersistent(handle(), persistent)); }

    bool isPersistent() const
    {
        bool persistent = false;
        SYN_CHECK(synSectionGetPersistent(handle(), &persistent));
        return persistent;
    }

    void setRMW(bool rmw) const { SYN_CHECK(synSectionSetRMW(handle(), rmw)); }

    void setGroup(const uint64_t group) const { SYN_CHECK(synSectionSetGroup(handle(), group)); }

    uint64_t getGroup() const
    {
        uint64_t group;
        SYN_CHECK(synSectionGetGroup(handle(), &group));
        return group;
    }

    bool isRmw() const
    {
        bool rmw;
        SYN_CHECK(synSectionGetRMW(handle(), &rmw));
        return rmw;
    }

    void setConst(bool isConst) const { SYN_CHECK(synSectionSetConst(handle(), isConst)); }

    bool isConst() const
    {
        bool isConst;
        SYN_CHECK(synSectionGetConst(handle(), &isConst));
        return isConst;
    }

    uint64_t getId() const { return reinterpret_cast<uint64_t>(handle()); }

protected:
    Section(std::shared_ptr<synSectionHandle> handle) : SynObject(handle) {}

    friend class GraphBase;  // GraphBase class requires access to Section private constructor
    friend class Tensor;     // Tensor class requires access to Section private constructor
};
}  // namespace syn
