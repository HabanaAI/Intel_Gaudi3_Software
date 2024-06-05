#pragma once
#include "slot_map.hpp"

/**
 * ConcurrentSlotMapPrecache
 * similar to ConcurrentSlotAlloc but with precaching mechanism
 * ctor allocates maxCapacity elements that are deallocated in dtor
 * @tparam T value data type
 * @tparam EntryArraySize entry array size. must be a power of 2
 */
template<class T, uint32_t EntryArraySize = 1024>
class ConcurrentSlotMapPrecache : private ConcurrentSlotMap<T, EntryArraySize>
{
    using Base = ConcurrentSlotMap<T, EntryArraySize>;

public:
    using UninitFunc = std::function<bool(T*)>;
    /**
     * ctor
     * allocate maxCapacity elements for further use. so no more memory allocations happens at runtime
     * @param maxCapacity max number of elements that the map can hold.
     * @param uninitFunc a user-provided function that is used to uninitialize a value
     * @param ctorArgs ctor arguments
     */
    template<class... CtorArgs>
    ConcurrentSlotMapPrecache(uint32_t      maxCapacity,
                              SlotMapChecks checks     = SlotMapChecks::full,
                              UninitFunc    uninitFunc = nullptr,
                              CtorArgs const&... ctorArgs)
    : Base(maxCapacity, checks), m_uninitFunc(uninitFunc)
    {
        if (initCache(ctorArgs...) == false)
        {
            throw std::runtime_error("failed to initialize ConcurrentSlotMapPrecache. failed to get a new entry");
        }
    }

    /**
     * insert a new value
     * complexity : O(1)
     *
     * @tparam CtorArgs
     * @param handlePayload user value that is injected into handle
     * @return pair of a unique handle of the inserted value and a shared pointer to the value
     *         if insertion failed - shared pointer contains nullptr
     */
    std::pair<SMHandle, SlotMapItemSptr<T>> insert(uint16_t handlePayload)
    {
        return Base::insertImpl(nullptr, handlePayload);
    }

    /**
     * find an element by handle and return its value
     * complexity : O(1), lock-free.
     *
     * @param handle
     * @return shared pointer to value or nullptr if the value not found
     */
    SlotMapItemSptr<T> operator[](SMHandle handle) { return Base::operator[](handle); }
    SlotMapItemSptr<const T> operator[](SMHandle handle) const { return Base::operator[](handle); }
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
    bool erase(SMHandle handle) { return eraseImpl(handle, Base::CrcCheck::on); }

    /**
     * execute a function for all the elements
     * @tparam UnaryFunction functor type. must have an operator void (T*)
     * @param f              functor that is invoked for all items
     */
    template<class UnaryFunction>
    void forEach(UnaryFunction&& f)
    {
        Base::forEach(std::forward<UnaryFunction>(f));
    }

    /**
     * erase all elements
     */
    void eraseAll()
    {
        for (uint32_t i = 0; i < Base::m_insertIndex.load(); ++i)
        {
            // erase by index - so no need for crc check
            eraseImpl(makeSMHandle(i), Base::CrcCheck::off);
        }
    }

    ~ConcurrentSlotMapPrecache()
    {
        for (uint32_t i = 0; i < Base::m_maxCapacity; ++i)
        {
            auto* entry = Base::getEntry(makeSMHandle(i));
            if (entry)
            {
                if (m_uninitFunc)
                {
                    m_uninitFunc(entry->item);
                }
                delete entry->item;
            }
        }
    }

private:
    bool eraseImpl(SMHandle handle, typename Base::CrcCheck crcCheck)
    {
        return Base::eraseImpl(handle, crcCheck, [this](typename Base::Entry* entry) {
            if (m_uninitFunc)
            {
                m_uninitFunc(entry->item);
            }
        });
    }

    template<class... CtorArgs>
    bool initCache(CtorArgs const&... ctorArgs)
    {
        for (uint32_t i = 0; i < Base::m_maxCapacity; ++i)
        {
            typename Base::Entry* entry = Base::addNewEntry();
            if (entry)
            {
                entry->item = new T(ctorArgs...);
                Base::m_deletedItems.push(entry);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    UninitFunc m_uninitFunc;
};
