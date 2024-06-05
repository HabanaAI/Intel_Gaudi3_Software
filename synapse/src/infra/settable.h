#ifndef _SETTABLE_H_
#define _SETTABLE_H_

#include "defs.h"

#include <utility>


struct unsetType{};
constexpr unsetType nullset;

template<class T>
class Settable
{
public:
    Settable() : m_set(false), m_value(T{}) {}

    Settable(unsetType) : Settable() {}

    Settable(const T& value) : m_set(true), m_value(value) {}

    T* operator->()
    {
        HB_ASSERT(m_set, "settable value not set but {} was called", __func__);
        return &m_value;
    }

    const T* operator->() const
    {
        HB_ASSERT(m_set, "settable value not set but {} was called", __func__);
        return &m_value;
    }

    const T& operator*() const
    {
        HB_ASSERT(m_set, "settable value not set but {} was called", __func__);
        return m_value;
    }

    // default copy-ctor and assignment operator

    bool is_set() const
    {
        return m_set;
    }

    void set(T&& value)
    {
        m_set = true;
        m_value = std::move(value);
    }

    void set(const T& value)
    {
        m_set = true;
        m_value = value;
    }

    void unset()
    {
        m_set = false;
    }

    const T& value() const &
    {
        HB_ASSERT(m_set, "settable value not set");
        return m_value;
    }

    T& value() &
    {
        HB_ASSERT(m_set, "settable value not set");
        return m_value;
    }

    T value() &&
    {
        HB_ASSERT(m_set, "settable value not set");
        return std::move(m_value);
    }

    Settable<T>& operator=(T&& other)
    {
        set(std::move(other));
        return *this;
    }

    Settable<T>& operator=(const T& other)
    {
        set(other);
        return *this;
    }

    friend bool operator==(const Settable<T>& lhs, const Settable<T>& rhs)
    {
        return lhs.m_set == rhs.m_set && (lhs.m_set == false || lhs.m_value == rhs.m_value);
    }

    friend bool operator==(const T& lhs, const Settable<T>& rhs)
    {
        return rhs.is_set() && lhs == rhs.value();
    }

    friend bool operator==(const Settable<T>& lhs, const T& rhs)
    {
        return rhs == lhs;
    }

    friend bool operator!=(const Settable<T>& lhs, const T& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator!=(const T& lhs, const Settable<T>& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator!=(const Settable<T>& lhs, const Settable<T>& rhs)
    {
        return !(lhs == rhs);
    }
private:

    bool m_set;
    T m_value;
};

#endif // _SETTABLE_H_
