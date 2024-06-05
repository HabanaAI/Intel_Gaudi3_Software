#pragma once
// This class may be used when using a vector, just in case of resizing the vector
template<typename T>
class MovableAtomic
{
public:
    MovableAtomic() = default;

    MovableAtomic(std::atomic<T>&& a) : m_val(a.load()) {}

    MovableAtomic(MovableAtomic&& other) : m_val(other.m_a.load()) {}

    MovableAtomic& operator=(MovableAtomic&& other) { m_val.store(other.m_val.load()); }

    MovableAtomic& operator=(T other)
    {
        m_val = other;
        return *this;
    }

    MovableAtomic& operator+=(T val)
    {
        m_val += val;
        return *this;
    }

    operator T() const noexcept
    {
        return m_val;
    }

    MovableAtomic& operator-=(T val)
    {
        m_val -= val;
        return *this;
    }
private:

    std::atomic<T> m_val;
};
