#pragma once

#include <mutex>
#include <memory>
#include <cstdint>
#include <vector>
#include <array>
#include <functional>
#include <atomic>
#include "intrusive_stack.hpp"
#include "infra/crc16.hpp"
#include <thread>
#include "../math_utils.h"

using SMHandle = uint64_t;

enum class SlotMapItemSptrOwnership
{
    shared,
    exclusive
};

/**
 * SlotMapItemSptr shared pointer to a SlotMap item
 * provides a simpler version of std::shared_ptr with similar functionality
 * difference with std::shared_ptr:
 *  -> atomic operations are lock-free (not like in std::shared_ptr)
 *  -> has no weak_ptr support
 *  -> not all the functionality is supported (e.g. no aliasing ctor)
 * @tparam T SlotMap value type
 */
template<class T>
class SlotMapItemSptr
{
public:
    using element_type = T;
    SlotMapItemSptr() : m_refCounter(nullptr), m_item(nullptr) {}
    SlotMapItemSptr(std::nullptr_t) : SlotMapItemSptr() {}
    SlotMapItemSptr(SlotMapItemSptr const& other) : SlotMapItemSptr(other.m_item, other.m_refCounter) {}

    template<class U>
    SlotMapItemSptr(SlotMapItemSptr<U> const& other) : SlotMapItemSptr(other.m_item, other.m_refCounter)
    {
        static_assert(std::is_same<U, T>::value || std::is_base_of<T, U>::value, "other must be a derived from T");
    }

    SlotMapItemSptr(SlotMapItemSptr<T>&& other) : m_refCounter(other.m_refCounter), m_item(other.m_item)
    {
        other.m_refCounter = nullptr;
    }

    template<class U>
    SlotMapItemSptr(SlotMapItemSptr<U>&& other) : m_refCounter(other.m_refCounter), m_item(other.m_item)
    {
        static_assert(std::is_same<std::remove_const_t<U>, std::remove_const_t<T>>::value || std::is_base_of<std::remove_const_t<T>, std::remove_const_t<U>>::value, "other must be a derived from T");
        other.m_refCounter = nullptr;
    }

    SlotMapItemSptr<T>& operator=(SlotMapItemSptr const& other)
    {
        if (&other == this) return *this;
        decrementRefCounter();
        new (this) SlotMapItemSptr<T>(other.m_item, other.m_refCounter);
        return *this;
    }

    template<class U>
    SlotMapItemSptr<T>& operator=(SlotMapItemSptr<U> const& other)
    {
        static_assert(std::is_same<U, T>::value || std::is_base_of<T, U>::value, "other must be a derived from T");
        decrementRefCounter();
        new (this) SlotMapItemSptr<T>(other.m_item, other.m_refCounter);
        return *this;
    }

    SlotMapItemSptr<T>& operator=(SlotMapItemSptr&& other)
    {
        if (&other == this) return *this;
        decrementRefCounter();

        m_item       = other.m_item;
        m_refCounter = other.m_refCounter;

        other.resetHard();
        return *this;
    }

    template<class U>
    SlotMapItemSptr<T>& operator=(SlotMapItemSptr<U>&& other)
    {
        static_assert(std::is_same<U, T>::value || std::is_base_of<T, U>::value, "other must be a derived from T");
        decrementRefCounter();
        m_item       = other.m_item;
        m_refCounter = other.m_refCounter;
        other.resetHard();
        return *this;
    }

    explicit operator bool() const { return m_item != nullptr; }

    T* operator->() { return m_item; }
    const T* operator->() const { return m_item; }

    T* get() { return m_item; }
    const T* get() const { return m_item; }

    T&       operator*() { return *m_item; }
    const T& operator*() const { return *m_item; }

    bool operator==(T* other) const { return m_item == other; }
    bool operator!=(T* other) const { return !(*this == other); }

    void reset()
    {
        decrementRefCounter();
        resetHard();
    }

    uint32_t use_count() const { return m_refCounter ? m_refCounter->load() : 0; }

    ~SlotMapItemSptr() { decrementRefCounter(); }

private:
    SlotMapItemSptr(T* item, std::atomic<uint32_t>* refCounter) : m_refCounter(refCounter), m_item(item) { init(); }
    SlotMapItemSptr(T* item, std::atomic<uint32_t>* refCounter, SlotMapItemSptrOwnership ownership)
    : m_refCounter(refCounter), m_item(item)
    {
        if (refCounter && ownership == SlotMapItemSptrOwnership::exclusive)
        {
            uint32_t curRefCount = 1;
            if (!m_refCounter->compare_exchange_strong(curRefCount, 2, std::memory_order_acq_rel))
            {
                resetHard();
            }
        }
        else
        {
            init();
        }
    }

    void init()
    {
        if (m_refCounter != nullptr)
        {
            uint32_t curRefCount = m_refCounter->load(std::memory_order_acquire);
            // if we are reaching max uint32 with a threshold - fail
            const uint32_t threshold = 10000;
            if (curRefCount > std::numeric_limits<uint32_t>::max() - threshold)
            {
                resetHard();
                return;
            }
            while (curRefCount != 0 &&
                   !m_refCounter->compare_exchange_weak(curRefCount, curRefCount + 1, std::memory_order_acq_rel))
            {
            };
            if (curRefCount == 0)
            {
                resetHard();
            }
        }
    }
    void decrementRefCounter()
    {
        if (m_refCounter != nullptr)
        {
            --(*m_refCounter);
        }
    }
    void resetHard()
    {
        m_refCounter = nullptr;
        m_item       = nullptr;
    }

    std::atomic<uint32_t>* m_refCounter;
    T*                     m_item;

    template<class U>
    friend class SlotMapItemSptr;
    template<class, uint32_t>
    friend class ConcurrentSlotMap;
    template<class TTo, class TFrom>
    friend SlotMapItemSptr<TTo> SlotMapItemDynamicCast(SlotMapItemSptr<TFrom> from);
};

template<class TTo, class TFrom>
SlotMapItemSptr<TTo> SlotMapItemDynamicCast(SlotMapItemSptr<TFrom> from)
{
    TTo* to = dynamic_cast<TTo*>(from.get());
    if (to != nullptr)
    {
        return SlotMapItemSptr<TTo>(to, from.m_refCounter);
    }
    return SlotMapItemSptr<TTo>();
}

// Handle structure : index : 32bit; payload : 16bit; crc : 16bit
constexpr unsigned handleCrcOffset     = 48;
constexpr unsigned handlePayloadOffset = 32;

constexpr SMHandle makeSMHandle(uint32_t index, uint16_t payload, uint16_t crc)
{
    // add 1 in order to prevent handle to become 0
    return uint64_t(index + 1) | (uint64_t(payload) << handlePayloadOffset) | ((uint64_t)(crc) << handleCrcOffset);
}

constexpr SMHandle makeSMHandle(uint32_t index)
{
    return makeSMHandle(index, 0, 0);
}

constexpr uint32_t getSMHandleIndex(SMHandle handle)
{
    return (handle & 0xFFFFFFFF) - 1;
}

constexpr uint16_t getSMHandleCrc(SMHandle handle)
{
    return handle >> handleCrcOffset;
}

constexpr uint16_t getSMHandlePayload(SMHandle handle)
{
    return handle >> handlePayloadOffset;
}

enum class SlotMapChecks
{
    full,
    noRefCounting
};

/**
 * ConcurrentSlotMap provides key-value functionality where a unique key is generated on value insertion
 * the container DOES NOT provide value ownership
 * if ownership is required pls use ConcurrentSlotMapAllocor or ConcurrentSlotPrecache
 *
 * all the operations are O(1)
 *
 * @tparam T value data type
 * @tparam EntryArraySize entry array size. must be a power of 2
 */
template<class T, uint32_t EntryArraySize = 1024>
class ConcurrentSlotMap
{
public:
    /**
     * ctor
     * @param maxCapacity max number of elements that the map can hold
     */
    ConcurrentSlotMap(uint32_t maxCapacity = 1024 * EntryArraySize, SlotMapChecks checks = SlotMapChecks::full)
    : m_insertIndex(0),
      m_allocatedCapacity(0),
      m_maxCapacity(maxCapacity),
      m_checks(checks),
      m_itemsDB((maxCapacity + EntryArraySize - 1) / EntryArraySize)
    {
        const uint16_t initialCrcValue = 0x7f7f;  // some random value
        m_crcStart                     = crc16(initialCrcValue, this);
    }
    /**
     * insert a new value into map.
     * complexity : O(1)
     * @param newItem pointer to a new value to insert
     * @param handlePayload payload that will be a part of Handle and can be used by a user
     * @return pair of a unique handle of the inserted value and a shared pointer to the value
     *         if insertion failed - shared pointer contains nullptr
     */
    std::pair<SMHandle, SlotMapItemSptr<T>> insert(T* newItem, uint16_t handlePayload)
    {
        return insertImpl(newItem, handlePayload);
    }
    /**
     * find an element by handle and return its value
     * complexity : O(1), lock-free.
     *
     * @param handle
     * @return shared pointer to value or nullptr if the value not found
     */
    SlotMapItemSptr<T> operator[](SMHandle handle) { return getItem(handle, CrcCheck::on); }
    SlotMapItemSptr<const T> operator[](SMHandle handle) const { return const_cast<ConcurrentSlotMap*>(this)->getItem(handle, CrcCheck::on); }

    /**
     * delete an element by handle. the freed memory is reused by insert
     * fails in the following cases:
     *  -> if an element is in use (is held by a shared pointer)
     *  -> handle is incorrect or an element was erased
     * if erase is called simultaneously from several threads - only one of them succeed and the others fail
     * complexity : O(1)
     *
     * @param handle
     * @return true if succeeded
     */
    bool erase(SMHandle handle) { return eraseImpl(handle, CrcCheck::on); }

    /**
     * execute a function for all the elements
     * @tparam UnaryFunction functor type. must have an operator void (T*)
     * @param f              functor that is invoked for all items
     */
    template<class UnaryFunction>
    void forEach(UnaryFunction&& f)
    {
        for (uint32_t i = 0; i < m_insertIndex.load(); ++i)
        {
            auto item = getItem(makeSMHandle(i), CrcCheck::off);
            if (item)
            {
                f(item.get());
            }
        }
    }
    /**
     * erase all the elements
     */
    void eraseAll()
    {
        for (uint32_t i = 0; i < m_insertIndex.load(); ++i)
        {
            eraseImpl(makeSMHandle(i), CrcCheck::off);
        }
    }

    ConcurrentSlotMap(ConcurrentSlotMap const&) = delete;
    ConcurrentSlotMap(ConcurrentSlotMap&&)      = delete;

private:
    struct Entry : IntrusiveStackNodeBase<Entry>
    {
        std::atomic<uint32_t> refCount {0};
        T*                    item {nullptr};
        // protection against the following case: delete item - create item with the same index.
        // abaProtectionCounter gives a different crc to a new entry with the same index
        uint32_t              abaProtectionCounter {0};
        std::atomic<uint16_t> crc {0};
        uint32_t              itemIndex {0};
    };
    enum class CrcCheck
    {
        on,
        off
    };

    SlotMapItemSptr<T> getItem(SMHandle                 handle,
                               CrcCheck                 crcCheckPolicy,
                               SlotMapItemSptrOwnership ownership = SlotMapItemSptrOwnership::shared,
                               Entry**                  outEntry  = nullptr)
    {
        Entry* entry = getEntry(handle);
        if (outEntry)
        {
            *outEntry = entry;
        }

        if (entry != nullptr)
        {
            // if crc check succeed - return a shared_ptr
            if (isCrcMatch(entry, handle, crcCheckPolicy))
            {
                SlotMapItemSptr<T> item(entry->item,
                                        m_checks == SlotMapChecks::full ? &entry->refCount : nullptr,
                                        ownership);
                // double check of crc - might be the item was deleted and re-inserted after the last check
                if (isCrcMatch(entry, handle, crcCheckPolicy))
                {
                    return item;
                }
            }
        }

        return SlotMapItemSptr<T>();
    }

    bool eraseImpl(SMHandle handle, CrcCheck crcCheck)
    {
        return eraseImpl(handle, crcCheck, [](Entry*) {});
    }

    template<class TUninitFunc>
    bool eraseImpl(SMHandle handle, CrcCheck crcCheck, TUninitFunc uninitFunc)
    {
        Entry*             entry;
        SlotMapItemSptr<T> itemSptr = getItem(handle, crcCheck, SlotMapItemSptrOwnership::exclusive, &entry);

        if (itemSptr == nullptr)
        {
            return false;
        }

        uint16_t curCrc  = entry->crc.load(std::memory_order_acquire);
        bool     success = false;
        // fail if the item is in use or the item was already erased
        unsigned expectedRefCounter = m_checks == SlotMapChecks::full ? 2 : 1;
        // even if there is no reference counting (SlotMapChecks::noRefCounting)
        // in case of simultaneous deletion only one thread will succeed
        if (entry->refCount.compare_exchange_strong(expectedRefCounter, 0))
        {
            // now we have exclusive access
            entry->crc.store(curCrc + 1, std::memory_order_release);
            itemSptr.resetHard();
            uninitFunc(entry);
            m_deletedItems.push(entry);
            success = true;
        }

        return success;
    }
    std::pair<SMHandle, SlotMapItemSptr<T>> insertImpl(T* preAllocatedItem, uint16_t handlePayload)
    {
        return insertImpl(preAllocatedItem, handlePayload, [](Entry*) {});
    }

    template<class TInitFunc>
    std::pair<SMHandle, SlotMapItemSptr<T>> insertImpl(T* preAllocatedItem, uint16_t handlePayload, TInitFunc initFunc)
    {
        Entry* entry = m_deletedItems.pop();
        if (!entry)
        {
            entry = addNewEntry();
        }
        if (!entry)
        {
            return {};
        }
        // at this point this 'insert' has exclusive access to the entry.
        // other threads can access this entry only to read or erase. both will fail
        ++entry->abaProtectionCounter;

        if (preAllocatedItem)
        {
            entry->item = preAllocatedItem;
        }
        initFunc(entry);
        uint16_t crc = crc16(m_crcStart, entry->item);
        crc          = crc16(crc, entry->abaProtectionCounter);
        crc          = crc16(crc, entry->itemIndex);
        crc          = crc16(crc, handlePayload);
        entry->crc   = crc;
        entry->refCount.store(1, std::memory_order_release);

        SMHandle handle = makeSMHandle(entry->itemIndex, handlePayload, entry->crc);

        return std::make_pair(
            handle,
            SlotMapItemSptr<T>(entry->item, m_checks == SlotMapChecks::full ? &entry->refCount : nullptr));
    }

    static bool isCrcMatch(const Entry* entry, uint64_t handle, CrcCheck crcCheck)
    {
        return crcCheck == CrcCheck::off ||
               (entry != nullptr && getSMHandleCrc(handle) == entry->crc.load(std::memory_order_acquire));
    }

    Entry* getEntry(SMHandle handle)
    {
        uint32_t index = getSMHandleIndex(handle);
        if (index >= m_insertIndex.load(std::memory_order_acquire))
        {
            return nullptr;
        }

        uint32_t entryArrayIndex   = index / EntryArraySize;
        uint32_t inEntryArrayIndex = index % EntryArraySize;
        return &(*m_itemsDB[entryArrayIndex])[inEntryArrayIndex];
    }

    Entry* addNewEntry()
    {
        if (m_insertIndex.load(std::memory_order_acquire) >= m_maxCapacity)
        {
            return nullptr;
        }

        uint32_t newItemIndex = m_insertIndex.fetch_add(1, std::memory_order_acq_rel);

        if (newItemIndex >= m_maxCapacity)
        {
            m_insertIndex.store(m_maxCapacity, std::memory_order_release);

            return nullptr;
        }

        // add a new entry array if all the entry arrays are full
        if (newItemIndex >= m_allocatedCapacity.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lck(m_addChunkMutex);
            uint32_t                    maxIndex = m_allocatedCapacity.load(std::memory_order_acquire);
            while (newItemIndex >= maxIndex)
            {
                auto newEntryArray = std::unique_ptr<EntryArray>(new EntryArray);
                for (auto& entry : *newEntryArray)
                {
                    entry.itemIndex = maxIndex++;
                }
                size_t newEntryArrayIndex = (maxIndex - 1) / EntryArraySize;

                assert(newEntryArrayIndex < m_itemsDB.capacity());

                m_itemsDB[newEntryArrayIndex] = std::move(newEntryArray);
                m_allocatedCapacity.store(maxIndex, std::memory_order_release);
            }
        }

        assert(newItemIndex < m_allocatedCapacity.load(std::memory_order_acquire));

        uint32_t entryArrayIndex   = newItemIndex / EntryArraySize;
        uint32_t inEntryArrayIndex = newItemIndex % EntryArraySize;
        return &(*m_itemsDB[entryArrayIndex])[inEntryArrayIndex];
    }

    using EntryArray    = std::array<Entry, EntryArraySize>;
    using EntryArrayPtr = std::unique_ptr<EntryArray>;

    static_assert(isPowerOf2(EntryArraySize), "EntryArraySize must be power of 2");

    ConcurrentIntrusiveStack<Entry> m_deletedItems;
    std::atomic<uint32_t>      m_insertIndex;  // item index where a new item is inserted if m_deletedItems is empty
    std::atomic<uint32_t>      m_allocatedCapacity;
    uint32_t                   m_maxCapacity;
    SlotMapChecks              m_checks;
    std::vector<EntryArrayPtr> m_itemsDB;
    std::mutex                 m_addChunkMutex;
    uint16_t                   m_crcStart;
    template<class, uint32_t>
    friend class ConcurrentSlotMapAlloc;
    template<class, uint32_t>
    friend class ConcurrentSlotMapPrecache;
};
