#pragma once
#include <optional>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>

namespace containers{
inline namespace hllog_1_0{

#ifdef likely
#undef likely
#endif
#define likely(x) __builtin_expect(!!(x), 1)

enum class ElementsOrder
{
    chronological,
    reverseChronological
};

/**
 * mostly lock-free queue that keeps QueueSize recent elements
 * new elements overwrite old ones
 *
 * push/emplace usually lock-free
 * not lock-free scenarios:
 *  1. QueueSize is not sufficient and 2 (or more) threads are writing into the same internal item
 *     e.g. QueueSize == 2
 *     T0-T3 are pushing at the same time
 *     in this case at least 2 threads will try to write into the same place.
 *     one thread will write and the other will be locked
 *  2. UseDoubleBuffer == false and processAndClear is working
 *     a new item can overwrite an existing one and then the item is processed by processAndClear
 *
 * @usage storing last N values for logging purposes
 *
 * @tparam TItem           element type of the queue
 * @param  UseDoubleBuffer use double buffer to prevent items overwrite during processAndClear/clear
 *                         if new items are being added while processAndClear/clear is in progress
 *                         set it to false if:
 *                           * there is a guarantee that no push/emplace is called while processAndClear is running
 *                           * double memory usage is not acceptable
 *
 */
template<class TItem>
class ConcurrentRecentElementsQueue
{
    // disable double buffering
    static constexpr bool UseDoubleBuffer = false;
    struct InternalItem
    {
        mutable std::atomic<bool>    locked {false};
        uint8_t              epoch {0xFF};
        std::optional<TItem> value;
    };

    struct Bucket
    {
        Bucket(size_t size)
        : items(size)
        {}
        std::atomic<uint64_t>     index {0};
        std::vector<InternalItem> items;
    };
public:
    // queueSize amount of recent elements of the queue. must be power of 2
    ConcurrentRecentElementsQueue(size_t queueSize)
    : m_size{toPowerOf2(queueSize)}
    , m_sizeBits{0}
    , m_buckets{m_size}
    {
        auto sz = m_size;
        while (sz > 1)
        {
            sz >>= 1;
            m_sizeBits ++;
        }
    }
    size_t capacity() const  { return m_size; };

    // construct a new element in-place
    // lock-free in most cases (see class desc)
    template<class... TArgs>
    void emplace(TArgs&&... args)
    {
        Bucket&        bucket       = acquireBucket();
        const uint64_t fullIndex    = bucket.index.fetch_add(1, std::memory_order_acq_rel);
        const uint8_t  newEpoch     = fullIndex >> m_sizeBits;
        const uint64_t index        = fullIndex & (m_size - 1);
        InternalItem&  internalItem = bucket.items[index];
        ItemUnlocker   itemUnlocker = acquireItem(internalItem);

        if (likely(uint8_t(newEpoch - internalItem.epoch) < 4))
        {
            internalItem.value.emplace(std::forward<TArgs>(args)...);
            internalItem.epoch = newEpoch;
        }
    }

    // add a new element
    // lock-free in most cases (see class desc)
    void push(TItem&& newItem)
    {
        addItemImpl([&](std::optional<TItem>& item) { item.emplace(std::move(newItem)); });
    }
    template <class TOther>
    void push(TOther&& newItem)
    {
        addItemImpl([&](std::optional<TItem>& item) { item.emplace(std::forward<TOther>(newItem)); });
    }

    // add a new element
    // lock-free in most cases (see class desc)
    void push(TItem const& newItem)
    {
        addItemImpl([&](std::optional<TItem>& item) { item = newItem; });
    }

    // 1. apply processItemFunc to all the recent items
    // 2. clear all the processed items from the queue
    // if new items are pushed while processAndClear is working,
    // they will be processed by the next processAndClear
    // if UseDoubleBuffer is false then new items can overwrite old ones and one should use the next overload
    void processAndClear(std::function<void(TItem& item)> processItemFunc,
                         ElementsOrder                    order = ElementsOrder::chronological)
    {
        processAndClear(
            [&](TItem& item, bool epochMatch) {
                if (epochMatch)
                {
                    processItemFunc(item);
                }
            },
            order);
    }

    class REQIterator
    {
    public:
        REQIterator() = default;
        REQIterator(std::unique_lock<std::mutex> lock, Bucket & bucket, uint64_t startIdx, uint64_t endIdx)
        : m_lock(std::move(lock))
        , m_bucket(&bucket)
        , m_startIdx(startIdx)
        , m_endIdx(endIdx)
        {
            if (m_bucket)
            {
                // it's possible that in the beginning items are empty - skip empty items
                skipEmptyItems();
            }
        }

        REQIterator & operator++()
        {
            if (m_startIdx != m_endIdx)
            {
                auto accessIndex = m_startIdx & (m_bucket->items.size() - 1);
                ItemUnlocker itemUnlocker = acquireItem(m_bucket->items[accessIndex]);
                m_bucket->items[accessIndex].value.reset();
            }
            if (m_startIdx < m_endIdx)
            {
                m_startIdx++;
            }
            else if (m_startIdx > m_endIdx)
            {
                m_startIdx--;
            }
            if (m_startIdx == m_endIdx)
            {
                m_lock.unlock();
            }
            return *this;
        }
        REQIterator(REQIterator && ) = default;
        // c++17 does not have move_only_function in order to keep this iterator
        // add a fake copy ctor to make it compile
        REQIterator(REQIterator const & )
        {
            throw std::runtime_error("REQIterator copy ctor should not be invoked");
        }
        explicit operator bool() const
        {
            return m_startIdx != m_endIdx;
        }
        const std::optional<TItem> & operator*() const
        {
            if (m_bucket == nullptr || m_startIdx == m_endIdx)
            {
                throw std::runtime_error("REQIterator - dereference an empty item");
            }
            auto accessIndex = m_startIdx & (m_bucket->items.size() - 1);
            return m_bucket->items[accessIndex].value;
        }

    private:
        void skipEmptyItems()
        {
            while (m_startIdx != m_endIdx)
            {
                auto accessIndex = m_startIdx & (m_bucket->items.size() - 1);
                ItemUnlocker itemUnlocker = acquireItem(m_bucket->items[accessIndex]);
                if (!m_bucket->items[accessIndex].value.has_value())
                {
                    m_startIdx++;
                }
                else
                {
                    break;
                }
            }
        }

        std::unique_lock<std::mutex> m_lock;
        Bucket * m_bucket   = nullptr;
        uint64_t m_startIdx = 0;
        uint64_t m_endIdx   = 0;
    };

    REQIterator getIteratorToProcessAndClean()
    {
        if (empty()) return REQIterator();

        std::unique_lock lock(m_dumpMtx);
        Bucket& bucket = acquireBucket();
        switchBucket();
        size_t itemsToProcess = std::min(bucket.index.load(), m_size);
        const uint64_t start = bucket.index - itemsToProcess;
        const uint64_t end = start + itemsToProcess;
        return REQIterator(std::move(lock), bucket, start, end);
    }

    // the same as the previous but provides an ability to handle epoch mismatch:
    // UseDoubleBuffer is false and an item is written after processAndClear started and before
    // the item is processed by processAndClear
    void processAndClear(std::function<void(TItem& item, bool epochMatch)> processItemFunc,
                         ElementsOrder                                     order = ElementsOrder::chronological)
    {
        std::lock_guard guard(m_dumpMtx);
        Bucket&         bucket = acquireBucket();
        switchBucket();

        auto processItem = [&](uint64_t i) {
            const uint8_t epoch = i >> m_sizeBits;
            i                   = i & (m_size - 1);

            ItemUnlocker itemUnlocker = acquireItem(bucket.items[i]);
            if (bucket.items[i].value)
            {
                processItemFunc(*bucket.items[i].value, epoch == bucket.items[i].epoch);
                bucket.items[i].value.reset();
            }
        };

        size_t itemsToProcess = std::min(bucket.index.load(), m_size);
        if (itemsToProcess)
        {
            if (order == ElementsOrder::chronological)
            {
                const uint64_t start = bucket.index - itemsToProcess;
                for (uint64_t i = start; i != start + itemsToProcess; ++i)
                {
                    processItem(i);
                }
            }
            else
            {
                const uint64_t start = bucket.index - 1;
                for (uint64_t i = start; i != start - itemsToProcess; --i)
                {
                    processItem(i);
                }
            }
        }
    }

    void clear()
    {
        std::lock_guard guard(m_dumpMtx);
        Bucket&         bucket = acquireBucket();
        switchBucket();

        const uint64_t start = bucket.index;
        for (uint64_t i = start; i != start + m_size; ++i)
        {
            auto&        InternalItem = bucket.items[i & (m_size - 1)];
            ItemUnlocker itemUnlocker = acquireItem(InternalItem);
            InternalItem.value.reset();
        }
        bucket.index = 0;
    }

    uint64_t getTotalNumItems() const { return acquireBucket().index; }
    bool     empty() const
    {
        if (acquireBucket().index == 0)
        {
            return true;
        }
        const Bucket&  bucket       = acquireBucket();
        const uint64_t fullIndex    = bucket.index.load() - 1;
        const uint64_t index        = fullIndex & (m_size - 1);
        const InternalItem&  internalItem = bucket.items[index];
        ItemUnlocker   itemUnlocker = acquireItem(internalItem);
        return ! internalItem.value.has_value();
    }

private:
    static constexpr unsigned toPowerOf2(size_t v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v;
    }
    template<class AddFunc>
    void addItemImpl(AddFunc addFunc)
    {
        Bucket&        bucket       = acquireBucket();
        const uint64_t fullIndex    = bucket.index.fetch_add(1, std::memory_order_acq_rel);
        const uint8_t  newEpoch     = fullIndex >> m_sizeBits;
        const uint64_t index        = fullIndex & (m_size - 1);
        InternalItem&  internalItem = bucket.items[index];
        ItemUnlocker   itemUnlocker = acquireItem(internalItem);

        if (likely(uint8_t(newEpoch - internalItem.epoch) < 4))
        {
            addFunc(internalItem.value);
            internalItem.epoch = newEpoch;
        }
    }

    Bucket& acquireBucket()
    {
        uint32_t bucketIdx = UseDoubleBuffer ? m_curBucketIdx.load(std::memory_order_acquire) : 0;
        return m_buckets[bucketIdx];
    }

    const Bucket& acquireBucket() const
    {
        uint32_t bucketIdx = UseDoubleBuffer ? m_curBucketIdx.load(std::memory_order_acquire) : 0;
        return m_buckets[bucketIdx];
    }

    void switchBucket() { m_curBucketIdx = UseDoubleBuffer ? (1 - m_curBucketIdx) : 0; }
    class ItemUnlocker
    {
    public:
        ItemUnlocker(std::atomic<bool>& locked) : m_locked(locked) {}
        ~ItemUnlocker() { m_locked.store(false, std::memory_order_release); }

    private:
        std::atomic<bool>& m_locked;
    };

    [[nodiscard]] static ItemUnlocker acquireItem(const InternalItem& item)
    {
        bool expected = false;
        while (!item.locked.compare_exchange_weak(expected, true, std::memory_order_acquire))
        {
            std::this_thread::yield();
            expected = false;
        }
        return ItemUnlocker(item.locked);
    }

    static constexpr unsigned          m_bucketsCount = UseDoubleBuffer ? 2 : 1;
    size_t                             m_size;
    uint8_t                            m_sizeBits;
    std::atomic<uint32_t>              m_curBucketIdx {0};
    std::array<Bucket, m_bucketsCount> m_buckets;

    std::mutex m_dumpMtx;
};

}
}