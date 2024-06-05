#pragma once

#include "slot_map.hpp"

/**
 * ConcurrentSlotMapAlloc
 * similar to ConcurrentSlotMap but with full value ownership
 * insert allocates a new Item with operator new
 * erase deallocates an item with operator delete
 * @tparam T value data type
 * @tparam EntryArraySize entry array size. must be a power of 2
 */
template<class T, uint32_t EntryArraySize = 1024>
class ConcurrentSlotMapAlloc : private ConcurrentSlotMap<T, EntryArraySize>
{
    using Base = ConcurrentSlotMap<T, EntryArraySize>;

public:
    using BeforeDeleteFunc = std::function<bool(T*)>;

    /**
     * ctor
     * @param maxCapacity max number of elements that the map can hold
     * @param beforeDeleteFunc a user-provided function that is called before item deletion
     */
    ConcurrentSlotMapAlloc(uint32_t         maxCapacity      = 1024 * EntryArraySize,
                           SlotMapChecks    checks           = SlotMapChecks::full,
                           BeforeDeleteFunc beforeDeleteFunc = nullptr)
    : Base(maxCapacity, checks), m_beforeDeleteFunc(beforeDeleteFunc)
    {
    }

    /**
     * insert a new value
     * @tparam CtorArgs
     * @param handlePayload user value that is injected into handle
     * @param ctorArgs      T ctor arguments
     * @return pair of a unique handle of the inserted value and a shared pointer to the value
     *         if insertion failed - shared pointer contains nullptr
     */
    template<class... CtorArgs>
    std::pair<SMHandle, SlotMapItemSptr<T>> insert(uint16_t handlePayload, CtorArgs&&... ctorArgs)
    {
        return Base::insertImpl(nullptr, handlePayload, [&ctorArgs...](typename Base::Entry* entry) {
            entry->item = new T(std::forward<CtorArgs>(ctorArgs)...);
        });
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
     * erase all
     */
    void eraseAll()
    {
        for (uint32_t i = 0; i < Base::m_insertIndex.load(); ++i)
        {
            // erase by index - so no need for crc check
            eraseImpl(makeSMHandle(i), Base::CrcCheck::off);
        }
    }
    ~ConcurrentSlotMapAlloc() { eraseAll(); }

private:
    bool eraseImpl(SMHandle handle, typename Base::CrcCheck crcCheck)
    {
        return Base::eraseImpl(handle, crcCheck, [this](typename Base::Entry* entry) {
            if (m_beforeDeleteFunc)
            {
                m_beforeDeleteFunc(entry->item);
            }
            delete entry->item;
            entry->item = nullptr;
        });
    }
    BeforeDeleteFunc m_beforeDeleteFunc;
};
