#pragma once

#include "syn_infra.hpp"
#include "synapse_api.h"

namespace syn
{
template<class T>
inline std::shared_ptr<T> createHandle()
{
    return std::make_shared<T>();
}

template<class T, class D, typename... Args>
inline std::shared_ptr<T> createHandle(D deleter, Args&&... args)
{
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...), [deleter](const T* p) {
        deleter(*p);
        delete p;
    });
}

template<class T, class D, typename... Args>
inline std::shared_ptr<T> createHandleWithCustomDeleter(D deleter, Args&&... args)
{
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...), [deleter](const T* p) {
        deleter();
        delete p;
    });
}

template<class T, class V>
inline std::vector<T> getHandles(const V& vec)
{
    std::vector<T> handles;
    for (const auto& e : vec)
    {
        handles.push_back(e.handle());
    }
    return handles;
}

template<class T>
class SynObject
{
public:
    SynObject() = default;
    SynObject(std::shared_ptr<T> handle) : m_handle(std::move(handle)) {}
    virtual ~SynObject() = default;

    T  handle() const { return *m_handle.get(); }
    T* handlePtr() const { return m_handle.get(); }

    void reset() { m_handle = nullptr; }

    explicit operator bool() const { return m_handle != nullptr; }

protected:
    std::shared_ptr<T> m_handle = nullptr;
};
}  // namespace syn