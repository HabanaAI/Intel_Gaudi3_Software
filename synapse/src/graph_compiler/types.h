#pragma once

#include "synapse_api_types.h"
#include "eager/lib/node_info/node_info_defs.h"

#include "tpc_kernel_lib_interface_private.h"
#include "tpc_kernel_lib_interface.h"

#include "chromium/small_map.h"
#include "chromium/small_set.h"
#include "llvm/small_vector.h"

#include <array>
#include <vector>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

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
    template<typename... Args, std::enable_if_t<std::is_constructible_v<Key, Args...>, int> = 0>
    KeyAndHash(Args&&... args) noexcept(std::is_nothrow_constructible_v<Key, Args...>)
    : m_key(std::forward<Args>(args)...), m_hash(std::hash<Key> {}(m_key))
    {
    }
    const Key& getKey() const { return m_key; }
    size_t     getHash() const { return m_hash; }
    bool       operator==(const KeyAndHash& other) const { return m_key == other.m_key; }
    // allow casts to pass the cached hash
    template<typename Key2>
    friend class KeyAndHash;

private:
    KeyAndHash(const Key& key, size_t hash) : m_key(key), m_hash(hash) {}
    KeyAndHash(Key&& key, size_t hash) : m_key(std::move(key)), m_hash(hash) {}

    Key    m_key  = {};
    size_t m_hash = 0;
};

template<typename Key>
struct std::hash<KeyAndHash<Key>>
{
    std::size_t operator()(const KeyAndHash<Key>& keyAndHash) const noexcept { return keyAndHash.getHash(); }
};

using StringViewWithHash = KeyAndHash<std::string_view>;
using StringWithHash     = KeyAndHash<std::string>;

// specialization for string_view to make a cheaper comparison
// since guids often point to same pointer + supplying a cast
// operation from StringViewWithHash to StringWithHash.

template<>
class KeyAndHash<std::string_view>
{
public:
    KeyAndHash() = default;
    KeyAndHash(const char* key) : m_key(key), m_hash(std::hash<std::string_view> {}(m_key)) {}
    KeyAndHash(std::string_view key) : m_key(key), m_hash(std::hash<std::string_view> {}(m_key)) {}
    KeyAndHash(const std::string& key) : m_key(key), m_hash(std::hash<std::string_view> {}(m_key)) {}
    KeyAndHash(const StringWithHash& other) : m_key(other.m_key), m_hash(other.m_hash) {}
    std::string_view getKey() const { return m_key; }
    size_t           getHash() const { return m_hash; }
    bool             operator==(const KeyAndHash& other) const
    {
        if (m_key.size() != other.m_key.size()) return false;
        if (m_key.data() == other.m_key.data()) return true;
        return m_key == other.m_key;
    }
    explicit operator StringWithHash() const { return StringWithHash(std::string(m_key), m_hash); }

private:
    std::string_view m_key  = {};
    size_t           m_hash = 0;
};

template<typename T, unsigned N>
using SmallVector = llvm_vecsmall::SmallVector<T, N>;

template<typename NormalSet, size_t ArraySize>
using SmallSet = chromium_small_set::small_set<NormalSet, ArraySize>;

template<typename NormalMap, size_t ArraySize>
using SmallMap = chromium_small_map::small_map<NormalMap, ArraySize>;

using deviceAddrOffset                         = uint64_t;

using UserParams      = void*;
using NSizeArray      = std::array<TSize, tpc_lib_api::MAX_TENSOR_DIM>;
using SizeArray       = std::array<TSize, SYN_MAX_TENSOR_DIM>;
using OffsetArray     = std::array<int, SYN_MAX_TENSOR_DIM>;
using CoordArray      = std::array<int, SYN_MAX_TENSOR_DIM>;
using NCoordArray     = std::array<int, tpc_lib_api::MAX_TENSOR_DIM>;
using StrideArray     = std::array<TStride, tpc_lib_api::MAX_TENSOR_DIM + 1>;
using NStrideArray    = std::array<TStride, tpc_lib_api::MAX_TENSOR_DIM + 1>;
using SifParams       = tpc_lib_api::ShapeInferenceParams;
using SifOutputs      = tpc_lib_api::ShapeInferenceOutput;
using TensorShapeInfo = tpc_lib_api::TensorShapeInfo;
using SifReturn       = tpc_lib_api::GlueCodeReturn;
using SifNodeParams   = UserParams;
using DimVector       = llvm_vecsmall::SmallVector<uint8_t, tpc_lib_api::MAX_TENSOR_DIM>;
using SizeVector      = llvm_vecsmall::SmallVector<TSize, tpc_lib_api::MAX_TENSOR_DIM>;
using SifPermutation  = tpc_lib_api::NodeTensorPermutation;

class Node;
class Tensor;
class NodeIOManager;
class TPCNode;
class MmeNode;
class DMANode;
class CommandQueue;
class Quantizer;
class QuantizationData;

struct TensorComparator;
struct NodeComparator;

using NodePtr                 = std::shared_ptr<Node>;
using ConstNodePtr            = std::shared_ptr<const Node>;
using TensorWeakPtr           = std::weak_ptr<Tensor>;
using TensorPtr               = std::shared_ptr<Tensor>;
using ConstTensorPtr          = std::shared_ptr<const Tensor>;
using DMANodePtr              = std::shared_ptr<DMANode>;
using TPCNodePtr              = std::shared_ptr<TPCNode>;
using MMENodePtr              = std::shared_ptr<MmeNode>;
using NodeList                = std::list<NodePtr>;
using NodeVector              = SmallVector<NodePtr, eager_mode::defaultMaxNodesPerGraph>;
using NodesMap                = std::map<NodePtr, NodePtr>;
using NodesByIDMap            = std::map<synNodeId, NodePtr>;
using TensorList              = std::list<TensorPtr>;
using TensorVector            = SmallVector<TensorPtr, MAX_TENSOR_NR>;
using TensorIndicesVector     = SmallVector<uint8_t, HABANA_DIM_MAX>;
using TensorSizesVector       = SmallVector<TSize, HABANA_DIM_MAX>;
using TensorStridesVector     = SmallVector<TStride, HABANA_DIM_MAX>;
using TensorMap               = std::map<TensorPtr, TensorPtr, TensorComparator>;
using TensorVectorPtr         = std::shared_ptr<TensorVector>;
using CommandQueuePtr         = std::shared_ptr<CommandQueue>;
using ConstCommandQueuePtr    = std::shared_ptr<const CommandQueue>;
using QuantizationMap         = SmallMap<std::map<uint32_t, QuantizationData>, 2>;
using QuantizationConflictMap = std::map<synNodeId, QuantizationMap>;
using QuantizerPtr            = std::shared_ptr<Quantizer>;
using QuantizersMap           = std::unordered_map<StringViewWithHash, QuantizerPtr>;
using kernelID                = uint32_t;
using NodesIDs                = SmallSet<std::set<synNodeId>, 4>;
using RawParamsData           = SmallVector<uint8_t, 64>;

template<typename T>
using TensorToItemOrderedMap = std::map<TensorPtr, T, TensorComparator>;
template<typename T>
using NodeToItemOrderedMap = std::map<NodePtr, T, NodeComparator>;

// TODO: Real deprecation or remove
typedef NodePtr          pNode          /*__attribute__((deprecated))*/; // Please use NodePtr
typedef TensorPtr        pTensor        /*__attribute__((deprecated))*/; // Please use TensorPtr

enum BypassType
{
    DISABLE_BYPASS,
    ENABLE_BYPASS,
    ENABLE_BYPASS_ONLY_COMPLETLY_OUT,
};

enum BundleType
{
    UNDEFINED,
    MME,
    TPC,
    SCALAR_PIPE,
    COMPLEX_GUID,  // TODO SW-40695 rename to RMW section
    DMA_TRANSPOSE
};

enum BundleEngine
{
    ENGINE_UNDEFINED,
    ENGINE_MME,
    ENGINE_TPC,
    ENGINE_MME_TPC,
    ENGINE_DMA
};

enum utilPacketType
{
    UTIL_PKT_TYPE_CP_DMA,
    UTIL_PKT_TYPE_LDMA,
    UTIL_PKT_TYPE_ARB_SET,
    UTIL_PKT_TYPE_ARB_CLEAR,
    UTIL_PKT_TYPE_OTHER,
};

enum ePacketValidationLoggingMode
{
    PKT_VAIDATION_LOGGING_MODE_DISABLED,
    PKT_VAIDATION_LOGGING_MODE_UPON_FAILURE,
    PKT_VAIDATION_LOGGING_MODE_ENABLED
};