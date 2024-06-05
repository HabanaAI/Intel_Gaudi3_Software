#ifndef MME__DESCRIPTOR_CACHE_H
#define MME__DESCRIPTOR_CACHE_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace MmeCommon
{
// utility class to combine key with it's corresponding hash, so that
// the hash is only computed once and then used for following searches.
// The API intentionally does not allow the user to change the key\hash
// after creation. It is possible to extend it to allow changing the key
// and re-calculating the hash implicitly, but we do not wish to allow
// the user supplying the hash to avoid pitfalls where the supplies hash
// diverges from the key.
template<typename Key>
class KeyAndHash
{
public:
    KeyAndHash() = default;
    template<typename... Args>
    KeyAndHash(Args&&... args) noexcept(std::is_nothrow_constructible_v<Key, Args...>)
    : m_key(std::forward<Args>(args)...), m_hash(std::hash<Key> {}(m_key))
    {
    }
    const Key& getKey() const { return m_key; }
    size_t getHash() const { return m_hash; }
    bool operator==(const KeyAndHash& other) const { return m_key == other.m_key; }
private:
    Key m_key = {};
    size_t m_hash = 0;
};

// this class is a generic static LRU map.
// it is designed as generic template Key (Key implements getHash)
// and template Value that would be a vector of Values in the cache
template<typename KeyType, typename ValueType>
class DescriptorsCache
{
public:
    static DescriptorsCache& getCacheInstance()
    {
        static DescriptorsCache instance;
        return instance;
    }

    DescriptorsCache(DescriptorsCache& other) = delete;
    void operator=(const DescriptorsCache& other) = delete;
    virtual ~DescriptorsCache() = default;

    using DescriptorsCacheValue = std::shared_ptr<const std::vector<ValueType>>;

    DescriptorsCacheValue get(const KeyAndHash<KeyType>& key)
    {
        std::unique_lock lock(m_mutex);
        const auto cachedDescIter = m_descCacheMap.find(key);
        if (cachedDescIter == m_descCacheMap.end())
        {
            return nullptr;
        }
        m_cachesHit++;
        const auto& [nodeIdx, valuePtr] = cachedDescIter->second;
        m_descCacheItList.updateToRecentlyUsed(nodeIdx);
        return valuePtr;
    }

    bool contains(const KeyAndHash<KeyType>& key)
    {
        std::unique_lock lock(m_mutex);
        return m_descCacheMap.count(key) > 0;
    }

    bool add(const KeyAndHash<KeyType>& key, const std::vector<ValueType>& value)
    {
        std::unique_lock lock(m_mutex);
        bool addedPair = false;
        if (likely(isEnabled() && m_descCacheMap.count(key) == 0))
        {
            if (size() == m_cacheSizeLimit)
            {
                reuseLastUsed(key, value);
            }
            else
            {
                addNewPair(key, value);
            }
            addedPair = true;
        }
        return addedPair;
    }

    std::vector<std::string> getDebugInfo();

protected:
    // API for derived test class and internal use (unsafe as lock is not taken)
    bool isEnabled() const { return m_cacheSizeLimit > 0; }
    size_t size() const { return m_descCacheMap.size(); }
    DescriptorsCache() : DescriptorsCache(getLRUSize()) {}
    DescriptorsCache(size_t cacheSizeLimit) : m_cacheSizeLimit(cacheSizeLimit), m_descCacheItList(m_cacheSizeLimit)
    {
        m_descCacheMap.reserve(m_cacheSizeLimit);
    }

private:
    using DescriptorMap = std::unordered_map<KeyAndHash<KeyType>, std::pair<size_t, DescriptorsCacheValue>>;
    using DescriptorMapCacheIt = typename DescriptorMap::iterator;
    using DescriptorMapCacheConstIt = typename DescriptorMap::const_iterator;

    // LRU tailored linked list supporting operations required by the LRU cache.
    // We need it for two reasons:
    // 1. we want to preallocate all the nodes at once to reduce allocations in
    //    hot path + reduce cache misses, as this way if we move around adjacently allocated
    //    nodes, they have better chances to reside in the same cache line and memory page.
    //    This is not guaranteed with an std list even if we allocate nodes one after the other
    //    as this depends upon the free pools and concurrent calls to new\malloc\delete\free from
    //    other threads.
    // 2. We want to have a two directional pointing between the std list and map so that accessing
    //    one element given the other is O(1), but with iterators to std list and std map this results
    //    in a cyclic template definition.
    class LRUList
    {
    public:
        LRUList(size_t size) { m_nodes.resize(size); }

        void updateToRecentlyUsed(size_t nodeIdx)
        {
            LRUNode& node = m_nodes[nodeIdx];
            if (m_first != &node)
            {
                removeNode(node);
                addToFront(node);
            }
        }

        size_t addNewNode(DescriptorMapCacheIt iter)
        {
            LRUNode& node = m_nodes[m_nextFreeIdx];
            node.mapIter = iter;
            if (unlikely(m_first == nullptr))
            {
                m_first = &node;
                m_last = &node;
            }
            else
            {
                addToFront(node);
            }
            return m_nextFreeIdx++;
        }

        DescriptorMapCacheIt reuseLastUsed()
        {
            updateToRecentlyUsed(std::distance(m_nodes.data(), m_last));
            return m_first->mapIter;
        }

    private:
        struct LRUNode
        {
            LRUNode* prev = nullptr;
            LRUNode* next = nullptr;
            DescriptorMapCacheIt mapIter;
        };

        void removeNode(LRUNode& node)
        {
            if (node.next)
            {
                node.prev->next = node.next;
                node.next->prev = node.prev;
            }
            else
            {
                m_last = node.prev;
                node.prev->next = nullptr;
            }
        }

        void addToFront(LRUNode& node)
        {
            m_first->prev = &node;
            node.next = m_first;
            m_first = &node;
        }

        LRUNode* m_first = nullptr;
        LRUNode* m_last = nullptr;
        size_t m_nextFreeIdx = 0;
        std::vector<LRUNode> m_nodes;
    };

    void reuseLastUsed(const KeyAndHash<KeyType>& key, const std::vector<ValueType>& value)
    {
        // re-use the map entry for the new (key,value) pair
        DescriptorMapCacheConstIt mapIter = m_descCacheItList.reuseLastUsed();
        auto mapEntry = m_descCacheMap.extract(mapIter);
        mapEntry.key() = key;
        mapEntry.mapped().second = std::make_shared<const std::vector<ValueType>>(value);
        m_descCacheMap.insert(std::move(mapEntry));
    }

    void addNewPair(const KeyAndHash<KeyType>& key, const std::vector<ValueType>& value)
    {
        auto iterPair =
            m_descCacheMap.emplace(key, std::make_pair(0, std::make_shared<const std::vector<ValueType>>(value)));
        DescriptorMapCacheIt mapIter = iterPair.first;
        auto& mapValue = mapIter->second;
        auto& nodeIdx = mapValue.first;
        nodeIdx = m_descCacheItList.addNewNode(mapIter);
        m_descsGenerated++;
    }

    static size_t getLRUSize()
    {
        static constexpr size_t DEFAULT_CACHE_LIMIT = 238;
        const char* descCacheSize = getenv("MME_DESCRIPTORS_CACHE_SIZE");
        return (descCacheSize != nullptr) ? std::stoi(descCacheSize) : DEFAULT_CACHE_LIMIT;
    }

    size_t m_cacheSizeLimit = 0;
    DescriptorMap m_descCacheMap = {};
    LRUList m_descCacheItList;
    std::mutex m_mutex;
    // Stats variables
    size_t m_cachesHit = 0;
    size_t m_descsGenerated = 0;
};
}  // namespace MmeCommon

#endif //MME__DESCRIPTOR_CACHE_H
