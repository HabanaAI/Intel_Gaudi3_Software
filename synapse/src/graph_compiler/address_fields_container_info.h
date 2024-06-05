#pragma once

#include <exception>
#include <map>
#include <memory>
#include <cstdint>
#include <utility>
#include <vector>
#include <utility>
#include <set>

#include "cache_types.h"
#include "types.h"
#include "infra/settable.h"
#include "node_roi.h"
#include "smf/shape_func_registry.h"

constexpr uint32_t FIRST_EXE_NODE_ID = 0;

using FieldType = EFieldType;

enum FieldMemoryType
{
    // Values are set to match the gc_recipe's AddressSpace enum
    FIELD_MEMORY_TYPE_HOST = 1,
    FIELD_MEMORY_TYPE_DRAM = 2,
    FIELD_MEMORY_TYPE_SRAM = 3
};

class FieldIndexIsNotSupportedException : public std::exception
{
};

class FieldHostAddressIsNotSupportedException : public std::exception
{
};

class FieldDeviceAddressIsNotSupportedException : public std::exception
{
};

static const uint64_t INVALID_ENGINE_FIELD_ID = 0;
static const uint64_t INVALID_CONTAINER_ID    = 0;

static const uint32_t SINGLE_FEILD_INDEX = 0;

class BasicFieldsContainerInfo;

class AddressFieldInfo;
typedef std::shared_ptr<AddressFieldInfo> AddressFieldInfoSharedPtr;
typedef std::weak_ptr<AddressFieldInfo>   AddressFieldInfoWeakPtr;

class BasicFieldInfo;
typedef std::shared_ptr<BasicFieldInfo> BasicFieldInfoSharedPtr;
typedef std::weak_ptr<BasicFieldInfo>   BasicFieldInfoWeakPtr;

class DynamicShapeFieldInfo;
typedef std::shared_ptr<DynamicShapeFieldInfo> DynamicShapeFieldInfoSharedPtr;
typedef std::weak_ptr<DynamicShapeFieldInfo>   DynamicShapeFieldInfoWeakPtr;

// This structs specifies an engine's field that could be patched.
class BasicFieldInfo
{
public:
    BasicFieldInfo(uint32_t  fieldIndexOffset,
                   uint64_t  fieldEngineId,
                   FieldType fieldType,
                   uint64_t  containerId = INVALID_CONTAINER_ID,
                   NodeROI*  roi = nullptr)
    : m_fieldIndexOffset(fieldIndexOffset),
      m_fieldEngineId(fieldEngineId),
      m_type(fieldType),
      m_fieldContainerId(containerId),
      m_roi(roi)
    {
    }

    virtual ~BasicFieldInfo() = default;

    virtual BasicFieldInfoSharedPtr clone() const = 0;

    virtual bool isDynamicShape() const = 0;

    virtual bool isMcidPatchPoint() const = 0;

    inline void setFieldIndexOffset(const uint32_t& fieldIndexOffset)
    {
        m_fieldIndexOffset = fieldIndexOffset;
        m_fieldHostAddress.unset();
    }

    void         setFieldAddress(const uint64_t& fieldAddress) { m_fieldHostAddress.set(fieldAddress); }
    void         setDeviceAddress(const uint64_t& deviceAddress) { m_deviceAddress.set(deviceAddress); }
    void         setEngineFieldId(unsigned newValue) { m_fieldEngineId = newValue; }
    void         setContainerId(const uint64_t& containerId) { m_fieldContainerId = containerId; }
    virtual void reconnectWithNewContainer(const BasicFieldsContainerInfo& bfci) {}

    uint32_t                     getFieldIndexOffset() const;
    uint64_t                     getFieldHostAddress() const;
    uint64_t                     getFieldDeviceAddress() const;
    inline uint64_t              getEngineFieldId() const { return m_fieldEngineId; }
    inline uint64_t              getFieldContainerId() const { return m_fieldContainerId; }

    FieldType getType() const { return m_type; }

    NodeROI* getRoi() const       { return m_roi; }
    void     setRoi(NodeROI* roi) { m_roi = roi; }

    const pNode& getOrigin() const { return m_origin; }
    void         setOrigin(pNode value) { m_origin = std::move(value); }

    uint32_t getNodeExecutionIndex() const;

    bool isContextField() { return m_isContextField; }
    void setIsContextField(bool flag) { m_isContextField = flag; }

    struct PatchedTensorInfo
    {
        unsigned m_tensorId;
        unsigned m_operandIndex;
        unsigned m_dim;

        friend bool operator==(const PatchedTensorInfo& lhs, const PatchedTensorInfo& rhs)
        {
            return lhs.m_tensorId == rhs.m_tensorId && lhs.m_operandIndex == rhs.m_operandIndex &&
                   lhs.m_dim == rhs.m_dim;
        }
    };

    virtual const PatchedTensorInfo* getPatchedTensorInfo() const { return nullptr; }
    bool                             hasPatchedTensorInfo() const { return getPatchedTensorInfo() != nullptr; }

private:
    uint32_t           m_fieldIndexOffset = 0;  // mutual exclusive with m_fieldHostAddress, only one can be valid
    Settable<uint64_t> m_fieldHostAddress;  // mutual exclusive with m_fieldIndexOffset, only one can be valid
    Settable<uint64_t> m_deviceAddress;     // this field is being used when de-serializing for specifying the
                                            // device-address of the patch-point

    // unique incremntal ID for the change. In case of an address that is separated to two instances, each will have the same field id.
    uint64_t  m_fieldEngineId = 0;
    FieldType m_type;

    uint64_t m_fieldContainerId = 0;  // usually blank
    NodeROI* m_roi              = nullptr;
    pNode    m_origin           = nullptr;

    bool m_isContextField = false;
};

class DynamicShapeFieldInfo : public BasicFieldInfo
{
public:
    DynamicShapeFieldInfo(uint32_t fieldIndexOffset, FieldType FieldType, ShapeFuncID smf, pNode origin, NodeROI* roi);

    virtual bool isDynamicShape() const override { return true; }
    virtual bool isMcidPatchPoint() const override { return false; }
    virtual bool isAddressPatchPoint() const { return false; }
    size_t       getSize() const { return m_size; }
    void         setSize(size_t size) { m_size = size; }

    const std::vector<uint8_t>& getMetadata() const { return m_metadata; }
    void                        setMetadata(std::vector<uint8_t> value) { m_metadata = std::move(value); }

    bool         isMaskable() const { return m_isMaskable; }
    bool         isUnskippable() const { return m_isUnskippable; }

    ShapeFuncID getSmfID() const { return m_smf; }

protected:
    size_t               m_size; // The amount of uint32_t registers that this patch point affects
    std::vector<uint8_t> m_metadata;
    pNode                m_origin;
    ShapeFuncID          m_smf;
    bool                 m_isMaskable;
    bool                 m_isUnskippable;
};

class  DynamicAddressFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicAddressFieldInfo(uint32_t                  fieldIndexOffset,
                            FieldType                 fieldType,
                            ShapeFuncID               smf,
                            pNode                     origin,
                            NodeROI*                  roi,
                            AddressFieldInfoSharedPtr addressFieldLow,
                            AddressFieldInfoSharedPtr addressFieldHigh)
    : DynamicShapeFieldInfo(fieldIndexOffset, fieldType, smf, std::move(origin), roi),
      m_addressFieldLow(std::move(addressFieldLow)),
      m_addressFieldHigh(std::move(addressFieldHigh))
    {
        m_size = 2;
    }
    bool isAddressPatchPoint() const override { return true; }
    void reconnectWithNewContainer(const BasicFieldsContainerInfo& bfci) override;

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicAddressFieldInfo>(*this); }

    AddressFieldInfoSharedPtr  m_addressFieldLow;
    AddressFieldInfoSharedPtr  m_addressFieldHigh;
};

class DynamicExecuteFieldInfo : public DynamicShapeFieldInfo
{
public:
    DynamicExecuteFieldInfo(FieldType FieldType, size_t signalCount, pNode origin, NodeROI* roi);

    BasicFieldInfoSharedPtr clone() const final { return std::make_shared<DynamicExecuteFieldInfo>(*this); }

    static constexpr size_t MME_SIGNAL_COUNT = 2;
    static constexpr size_t SINGLE_SIGNAL = 1;
    static constexpr size_t NO_SIGNAL = 0;
private:
    size_t m_signalCount;
};

// An Engine address might be defined by two separate descriptor's registers (like on current Goya's MME & TPC).
// This struct defines the Engine-Information for a given Descriptor-Field-Address
class AddressFieldInfo : public BasicFieldInfo
{
public:
    AddressFieldInfo(std::string_view                 sectionName,
                     uint64_t                         memorySectionId,
                     uint64_t                         targetAddress,
                     uint32_t                         fieldIndexOffset,
                     FieldType                        addressPart,
                     FieldMemoryType                  fieldMemoryType,
                     uint64_t                         fieldEngineId,
                     const AddressFieldInfoSharedPtr& pOtherAddressFieldInfo = nullptr,
                     uint64_t                         containerId            = INVALID_CONTAINER_ID)
    : BasicFieldInfo(fieldIndexOffset, fieldEngineId, addressPart, containerId),
      m_sectionName(sectionName),
      m_memorySectionId(memorySectionId),
      m_targetAddress(targetAddress),
      m_fieldMemoryType(fieldMemoryType),
      m_relatedFieldInfo(pOtherAddressFieldInfo)
    {
    }

    BasicFieldInfoSharedPtr clone() const override;

    bool isDynamicShape() const override { return false; }

    bool isMcidPatchPoint() const override { return false; }

    uint64_t getTargetAddress() const { return m_targetAddress; }

    inline const std::string& getSectionName() const { return m_sectionName; }

    inline uint64_t getMemorySectionId() const { return m_memorySectionId; }

    inline FieldType getAddressPart() const { return getType(); }

    inline FieldMemoryType getFieldMemoryType() const { return m_fieldMemoryType; }

    void setOtherAddressFieldInfo(const AddressFieldInfoSharedPtr& relatedFieldInfo)
    {
        m_relatedFieldInfo = relatedFieldInfo;
    }
    inline AddressFieldInfoSharedPtr getOtherAddressFieldInfo() const { return m_relatedFieldInfo.lock(); }

    uint64_t getIndex() const { return m_index; }
    void     setIndex(uint64_t index) { m_index = index; }
    void     reconnectWithNewContainer(const BasicFieldsContainerInfo& bfci) override;

private:
    std::string m_sectionName;  // Section name, for debug purposes and slight identification.
    uint64_t    m_memorySectionId;
    uint64_t    m_targetAddress;

    FieldMemoryType m_fieldMemoryType;

    // link to some other address field info relevant to it. For high address part, points to low.
    AddressFieldInfoWeakPtr m_relatedFieldInfo;
    uint64_t m_index = -1;
};

class SyncObjectAddressFieldInfo : public BasicFieldInfo
{
public:
    SyncObjectAddressFieldInfo(uint32_t tensorId, NodePtr origin);
    virtual BasicFieldInfoSharedPtr clone() const override { return std::make_shared<SyncObjectAddressFieldInfo>(*this); }
    virtual bool                    isDynamicShape() const override { return false; }
    virtual bool                    isMcidPatchPoint() const override { return false; }

    uint32_t getTensorId() const { return m_tensorId; }

private:
    uint32_t m_tensorId;
};

template<typename T1, typename T2>
struct PairComparator
{
    bool operator() (const std::pair<T1, T2>& lhs,
                     const std::pair<T1, T2>& rhs) const
    {
        if (lhs.first == rhs.first)
            return lhs.second < rhs.second;
        return lhs.first < rhs.first;
    }
};

class McidFieldInfo : public BasicFieldInfo
{
public:
    McidFieldInfo(uint32_t               fieldIndexOffset,
                  NodePtr                origin,
                  CacheMaintenanceAction cmAction);
    virtual BasicFieldInfoSharedPtr clone() const override { return std::make_shared<McidFieldInfo>(*this); }
    virtual bool                    isDynamicShape() const override { return false; }
    virtual bool                    isMcidPatchPoint() const override { return true; }
    CacheMaintenanceAction          getCmAction() const { return m_cmAction; }

private:
    CacheMaintenanceAction m_cmAction;
};

// Pair containing the field-offset, and its address-field's information
using AddressFieldInfoPair = std::pair<uint32_t, AddressFieldInfoSharedPtr>;

// A DB for the Container's AddressFieldInfoPair-s
using AddressFieldInfoSet = std::set<AddressFieldInfoPair, PairComparator<uint32_t, AddressFieldInfoSharedPtr>>;
using AddressFieldInfoSetIter = AddressFieldInfoSet::iterator;

// Pair containing the field-offset, and its address-field's information
using BasicFieldInfoPair = std::pair<uint32_t, BasicFieldInfoSharedPtr>;
using BasicFieldInfoSet = std::set<BasicFieldInfoPair, PairComparator<uint32_t, BasicFieldInfoSharedPtr>>;

void transferAddressToBasic(AddressFieldInfoSet& addressFieldsInfoSet, BasicFieldInfoSet& basicFieldsInfoSet);
void transferBasicToAddress(BasicFieldInfoSet& basicFieldsInfoSet, AddressFieldInfoSet& addressFieldsInfoSet);

// Describes information about address-fields that some container (like descriptor) contains
//
// This container represents information for a single consumer of those contained
// EngineFields (pair of AddressFieldInfo that represent an address on an Engine) elements.
// The EngineFields's container "producer" (AKA - that there address is relative to) might be different,
// and it is represented by uint64_t at the field instance
class BasicFieldsContainerInfo
{
public:
    static uint64_t getNextUniqueId();

    BasicFieldsContainerInfo() : m_roi(nullptr) {}

    void addAddressEngineFieldInfo(const NodePtr&   node,
                                   std::string_view sectionName,
                                   uint64_t         memorySectionId,
                                   uint64_t         fieldLowAddress,
                                   uint64_t         fieldHighAddress,
                                   uint64_t         containerBaseAddress,
                                   FieldMemoryType  fieldMemoryType  = FIELD_MEMORY_TYPE_DRAM,
                                   uint64_t         fieldContainerId = INVALID_CONTAINER_ID);

    void addAddressEngineFieldInfo(const NodePtr&   node,
                                   std::string_view sectionName,
                                   uint64_t         memorySectionId,
                                   uint32_t         fieldLowAddress,
                                   uint32_t         fieldHighAddress,
                                   uint32_t         fieldLowIndexOffset,
                                   uint32_t         fieldHighIndexOffset,
                                   FieldMemoryType  fieldMemoryType = FIELD_MEMORY_TYPE_DRAM);

    void addAddressEngineFieldInfo(const NodePtr&   node,
                                   std::string_view sectionName,
                                   uint64_t         memorySectionId,
                                   uint64_t         fieldFullAddress,
                                   uint64_t         containerBaseAddress,
                                   FieldMemoryType  fieldMemoryType  = FIELD_MEMORY_TYPE_DRAM,
                                   uint64_t         fieldContainerId = INVALID_CONTAINER_ID);

    void addAddressEngineFieldInfo(const NodePtr&   node,
                                   std::string_view sectionName,
                                   uint64_t         memorySectionId,
                                   uint64_t         targetAddr,
                                   uint32_t         fieldIndexOffset,
                                   FieldMemoryType  fieldMemoryType);

    void add(const AddressFieldInfoPair& addressFieldInfoPair);
    void add(const BasicFieldInfoPair& basicFieldInfoPair);
    void add(const AddressFieldInfoSharedPtr& addressFieldInfoSharedPtr);
    void add(const BasicFieldInfoSharedPtr& basicFieldInfoSharedPtr);

    void retrieveAndDropSegment(uint32_t                  firstFieldIndexOffset,
                                uint32_t                  lastFieldIndexOffset,
                                BasicFieldsContainerInfo& partialAddrFieldContInfo);

    BasicFieldsContainerInfo
                         retrieveAndDropSegment(uint32_t firstFieldIndexOffset, uint32_t lastFieldIndexOffset);
    BasicFieldsContainerInfo
                         retrieveSegment(uint32_t firstFieldIndexOffset, uint32_t lastFieldIndexOffset);

    BasicFieldInfoSet&   retrieveBasicFieldInfoSet() { return m_basicFieldInfoSet; }
    AddressFieldInfoSet& retrieveAddressFieldInfoSet() { return m_addressFieldInfoSet; }

    const AddressFieldInfoSet& retrieveAddressFieldInfoSet() const { return m_addressFieldInfoSet; }
    const BasicFieldInfoSet&   retrieveBasicFieldInfoSet() const { return m_basicFieldInfoSet; }

    void updateContainerId(const uint64_t& fieldContainerId);

    void clear() { m_addressFieldInfoSet.clear(); }

    unsigned size() const { return m_addressFieldInfoSet.size() + m_basicFieldInfoSet.size(); }

    bool empty() const { return m_addressFieldInfoSet.empty() && m_basicFieldInfoSet.empty(); }

    bool find(uint32_t fieldIndex) const;
    AddressFieldInfoSharedPtr findAddress(uint32_t fieldIndex) const;

    bool hasDynamicPatchPoints() const
    {
        // Currently the m_basicFieldInfoSet only contains dynamic patch points, and all dynamic patch points are saved in that struct.
        // Thought: Maybe create a container for dynamic patch points?
        return !m_basicFieldInfoSet.empty();
    }

    bool hasOnlyCompressibleDynamicPatchPoints() const
    {
        for (const auto& fld : m_basicFieldInfoSet)
        {
            if (!fld.second->hasPatchedTensorInfo()) return false;
        }
        return true;
    }

    void     setRoi(NodeROI* roi) { m_roi = roi; }
    NodeROI* getRoi() const       { return m_roi; }

private:
    void retrieveSegment(uint32_t                  firstFieldIndexOffset,
                         uint32_t                  lastFieldIndexOffset,
                         BasicFieldsContainerInfo& partialAddrFieldContInfo);

    template<typename T>
    std::pair<typename T::iterator, typename T::iterator>
    retrieveSegment(uint32_t                  firstFieldIndexOffset,
                    uint32_t                  lastFieldIndexOffset,
                    BasicFieldsContainerInfo& partialAddrFieldContInfo,
                    T&                        container,
                    bool                      clone);

    template<typename T>
    void retrieveAndDropSegment(uint32_t                  firstFieldIndexOffset,
                                uint32_t                  lastFieldIndexOffset,
                                BasicFieldsContainerInfo& partialAddrFieldContInfo,
                                T& container);

    AddressFieldInfoSet m_addressFieldInfoSet = {};
    BasicFieldInfoSet   m_basicFieldInfoSet   = {};
    NodeROI*            m_roi                 = nullptr;
};
