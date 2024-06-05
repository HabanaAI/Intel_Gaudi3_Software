#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/general_defs.h"
#include "utils/numeric_utils.h"

// std includes
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>

namespace eager_mode
{
template<typename T>
inline std::enable_if_t<std::is_trivially_copyable_v<T>, T> readAs(const void* src)
{
    T res;
    std::memcpy(&res, src, sizeof(T));
    return res;
}

template<typename T>
inline void writeAs(void* dst, const T& val)
{
    std::memcpy(dst, &val, sizeof(T));
}

// TODO: There's a constexpr variant of this in c++20
template<typename To, typename From>
inline std::enable_if_t<sizeof(To) == sizeof(From) &&
                            (std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To> &&
                             std::is_trivially_constructible_v<To>),
                        To>
bit_cast(const From& src)
{
    To res;
    std::memcpy(&res, &src, sizeof(To));
    return res;
}

// Advance `offset` to fit `count` items of type T with specified alignment requirements
// NOP if `count` is 0.
template<typename T, size_t BaseAlignment = alignof(T), size_t PerItemAlignment = alignof(T)>
static constexpr void planAlloc(size_t& offset, size_t count)
{
    static_assert(BaseAlignment % PerItemAlignment == 0);
    if (count > 0)  // 0 sized allocs will not result in a unique address nor contrib padding
    {
        offset = alignUpTo<BaseAlignment>(offset) + alignUpTo<PerItemAlignment>(sizeof(T)) * count;
    }
}

// Placement-new `count` items of type T with a given alignment at `ptr` and advance `ptr` to point past them.
// NOP if `count` is 0 and return nullptr.
template<typename T, size_t BaseAlignment = alignof(T)>
static std::add_pointer_t<T> doPlacement(/*IN, OUT*/ std::byte*& ptr, size_t count)
{
    if (count > 0)  // 0 sized allocs will not result in a unique address nor contrib padding
    {
        std::byte* base = reinterpret_cast<std::byte*>(alignUpTo<BaseAlignment>(reinterpret_cast<std::uintptr_t>(ptr)));
        ptr             = base + sizeof(T) * count;
        using ResType   = typename std::remove_reference_t<T>;
        return count > 1 ? new (base) ResType[count] : new (base) ResType;
    }
    return {};
}

// Cast-up buf ito object with type T.
// In order to avoid hitting the static assert, buf must be correctly aligned once allocating it (statically or dynamically).
template<typename T>
inline T& doExactPlacement(std::byte* buf)
{
    assert(alignFor<T>(buf) == buf);
    auto& obj = *(new (buf) T);
    return obj;
}

template<typename T>
class OwnedAlignedBuf
{
public:
    struct AlignedDeleter
    {
        void operator()(std::byte* b) { ::operator delete[](b, std::align_val_t {alignof(T)}, std::nothrow); }
    };
    using AlignUniquePtr = std::unique_ptr<std::byte[], AlignedDeleter>;

    constexpr OwnedAlignedBuf() noexcept = default;

    constexpr explicit OwnedAlignedBuf(AlignUniquePtr&& buf) noexcept
    : m_buf(std::move(buf)), m_ptr(m_buf ? reinterpret_cast<T*>(alignFor<T>(m_buf.get())) : nullptr)
    {
        // TODO: this isn't terribly clean and might still result in UB:
        //       What if the T object wasn't constructed there at all?
        //       What if the cast is done before it's constructed?
    }

    constexpr explicit OwnedAlignedBuf(OwnedAlignedBuf&& o) noexcept : OwnedAlignedBuf() { swap(*this, o); }

    constexpr OwnedAlignedBuf& operator=(OwnedAlignedBuf&& o) noexcept
    {
        swap(*this, o);
        return *this;
    }

    constexpr friend void swap(OwnedAlignedBuf& a, OwnedAlignedBuf& b) noexcept
    {
        using std::swap;         // Good habit
        swap(a.m_buf, b.m_buf);  // a.m_buf.swap(b.m_buf); ??? // TODO: c++23: swap(a.m_buf, b.m_buf); became constepxr
        swap(a.m_ptr, b.m_ptr);
    }

    constexpr T* get() const noexcept { return m_ptr; }

private:
    AlignUniquePtr m_buf {};
    T*             m_ptr {};
};

// Movable but not Copyable
// static_assert(std::is_copy_constructible_v<OwnedAlignedBuf<int>> == false);
// static_assert(std::is_copy_assignable_v<OwnedAlignedBuf<int>> == false);
// static_assert(std::is_move_constructible_v<OwnedAlignedBuf<int>> == true);
// static_assert(std::is_move_assignable_v<OwnedAlignedBuf<int>> == true);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utilities to help incrementally initialize a buffer
////////////////////////////////////////////////////////////////////////////////////////////////////

// A representation of buffer with its size, it can be replaced with std::span once we have C++20
class DataBuf
{
public:
    DataBuf() : m_isInitialized(false), m_bufBase(nullptr), m_size(0) {}
    DataBuf(std::byte* buffer, size_t size) : m_isInitialized(true), m_bufBase(buffer), m_size(size) {}
    // Note that the following copy constructor allows producing multiple objects that point on same buffer
    DataBuf(const DataBuf& bufAlloc) : m_isInitialized(true), m_bufBase(bufAlloc.m_bufBase), m_size(bufAlloc.m_size) {}
    virtual ~DataBuf() = default;

    void init(std::byte* buffer, size_t size)
    {
        EAGER_ASSERT(!m_isInitialized, "Buffer had been initialized before");
        m_isInitialized = true;
        m_bufBase       = buffer;
        m_size          = size;
    }

    size_t size() const { return m_size; }

protected:
    bool       m_isInitialized;  // Does this object initialized before
    std::byte* m_bufBase;        // Pointer to buffer
    size_t     m_size;           // Actual size of the buffer
};

template<typename T>
class DataBufAllocator : public DataBuf
{
public:
    DataBufAllocator(T* buffer, size_t size) : DataBuf(reinterpret_cast<std::byte*>(buffer), size) {}
    DataBufAllocator(const DataBuf& bufAlloc) : DataBuf(bufAlloc) {}

    std::byte* allocate(size_t size)
    {
        const size_t curPos = m_pos;
        m_pos += size;
        EAGER_ASSERT(m_pos <= m_size, "Invalid blobs data buffer allocation");
        return m_bufBase + curPos;
    }

    size_t getPos() const { return m_pos; }
    bool   isAllocationCompleted() const { return (m_pos == m_size); }

private:
    size_t m_pos {};  // Current pointer in buffer to be assigned to next data pointer
};

class StringBufAllocator final : public DataBufAllocator<char>
{
public:
    StringBufAllocator(const DataBuf& strBufAlloc) : DataBufAllocator<char>(strBufAlloc) {}

    char* cloneAllocStr(std::string_view strView)
    {
        const size_t length = strView.size();
        auto*        buf    = reinterpret_cast<char*>(allocate(length + /*for the \0*/ 1));
        strView.copy(buf, length);
        buf[length] = 0;  // the \0
        return buf;
    }
};

}  // namespace eager_mode
