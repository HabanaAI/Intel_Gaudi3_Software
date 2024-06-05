#include <atomic>
#include <memory>

#include "address_fields_container_info.h"
#include "cache_types.h"
#include "utils.h"
#include "node.h"

static std::atomic<uint64_t> stEngineFieldIdentifier(0);

void transferAddressToBasic(AddressFieldInfoSet& addressFieldsInfoSet, BasicFieldInfoSet& basicFieldsInfoSet)
{
    basicFieldsInfoSet.clear();
    for (const auto& addressPair : addressFieldsInfoSet)
    {
        basicFieldsInfoSet.insert(std::make_pair(addressPair.first, addressPair.second));
    }
}

void transferBasicToAddress(BasicFieldInfoSet& basicFieldsInfoSet, AddressFieldInfoSet& addressFieldsInfoSet)
{
    addressFieldsInfoSet.clear();
    for (const auto& basicPair : basicFieldsInfoSet)
    {
        addressFieldsInfoSet.insert(
            std::make_pair(basicPair.first, std::static_pointer_cast<AddressFieldInfo>(basicPair.second)));
    }
}
void AddressFieldInfo::reconnectWithNewContainer(const BasicFieldsContainerInfo& bfci)
{
    auto other = getOtherAddressFieldInfo();
    if (other == nullptr)
    {
        return;
    }
    auto        it        = bfci.findAddress(other->getFieldIndexOffset());
    HB_ASSERT(it != nullptr, "We can not find the other address in the new continer: {}", other->getFieldIndexOffset());
    setOtherAddressFieldInfo(it);
}

uint32_t BasicFieldInfo::getNodeExecutionIndex() const
{
    if (getOrigin().get() == nullptr)
    {
        return FIRST_EXE_NODE_ID;
    }

    return getOrigin()->getExecutionOrderedIndex() + 1;  // index 0 is reserved, so return real execution index + 1.
}

BasicFieldInfoSharedPtr AddressFieldInfo::clone() const
{
    AddressFieldInfoSharedPtr pClone = std::make_shared<AddressFieldInfo>(*this);
    return pClone;
}

uint32_t BasicFieldInfo::getFieldIndexOffset() const
{
    if (m_fieldHostAddress.is_set())
    {
        throw FieldHostAddressIsNotSupportedException();
    }
    return m_fieldIndexOffset;
}

uint64_t BasicFieldInfo::getFieldHostAddress() const
{
    if (!m_fieldHostAddress.is_set())
    {
        throw FieldHostAddressIsNotSupportedException();
    }
    return m_fieldHostAddress.value();
}

uint64_t BasicFieldInfo::getFieldDeviceAddress() const
{
    if (!m_deviceAddress.is_set())
    {
        throw FieldDeviceAddressIsNotSupportedException();
    }
    return m_deviceAddress.value();
}

void BasicFieldsContainerInfo::addAddressEngineFieldInfo(const NodePtr&   node,
                                                         std::string_view sectionName,
                                                         uint64_t         memorySectionId,
                                                         uint64_t         fieldLowAddress,
                                                         uint64_t         fieldHighAddress,
                                                         uint64_t         containerBaseAddress,
                                                         FieldMemoryType  fieldMemoryType,
                                                         uint64_t         fieldContainerId)
{
    const uint32_t fieldLowIndexOffset   = ((uint64_t) fieldLowAddress - containerBaseAddress) / sizeof(uint32_t);
    const uint32_t fieldHighIndexOffset  = ((uint64_t) fieldHighAddress - containerBaseAddress) / sizeof(uint32_t);
    addAddressEngineFieldInfo(node,
                              sectionName,
                              memorySectionId,
                              *(uint32_t*) fieldLowAddress,
                              *(uint32_t*) fieldHighAddress,
                              fieldLowIndexOffset,
                              fieldHighIndexOffset,
                              fieldMemoryType);
}

void BasicFieldsContainerInfo::addAddressEngineFieldInfo(const NodePtr&   node,
                                                         std::string_view sectionName,
                                                         uint64_t         memorySectionId,
                                                         uint64_t         fieldFullAddress,
                                                         uint64_t         containerBaseAddress,
                                                         FieldMemoryType  fieldMemoryType,
                                                         uint64_t         fieldContainerId)
{
    const uint32_t fieldFullIndexOffset = ((uint64_t) fieldFullAddress - containerBaseAddress) / sizeof(uint32_t);
    uint64_t       dramAddress;
    memcpy(&dramAddress, (void *)fieldFullAddress, sizeof(uint64_t));

    addAddressEngineFieldInfo(node, sectionName, memorySectionId, dramAddress, fieldFullIndexOffset, fieldMemoryType);
}

void BasicFieldsContainerInfo::addAddressEngineFieldInfo(const NodePtr&   node,
                                                         std::string_view sectionName,
                                                         uint64_t         memorySectionId,
                                                         uint64_t         targetAddr,
                                                         uint32_t         fieldIndexOffset,
                                                         FieldMemoryType  fieldMemoryType)
{
    uint64_t engineFieldIdentifier = getNextUniqueId();

    auto& pFullAddressField = m_addressFieldInfoSet
                                  .emplace(fieldIndexOffset,
                                           new AddressFieldInfo(std::move(sectionName),
                                                                memorySectionId,
                                                                maskOutMemoryID(targetAddr),
                                                                fieldIndexOffset,
                                                                FIELD_ADDRESS_PART_FULL,
                                                                fieldMemoryType,
                                                                engineFieldIdentifier))
                                  .first->second;

    pFullAddressField->setOrigin(node);
    pFullAddressField->setRoi(m_roi);
}

void BasicFieldsContainerInfo::addAddressEngineFieldInfo(const NodePtr&   node,
                                                         std::string_view sectionName,
                                                         uint64_t         memorySectionId,
                                                         uint32_t         fieldLowAddress,
                                                         uint32_t         fieldHighAddress,
                                                         uint32_t         fieldLowIndexOffset,
                                                         uint32_t         fieldHighIndexOffset,
                                                         FieldMemoryType  fieldMemoryType)
{
    uint64_t engineFieldIdentifier = getNextUniqueId();
    uint64_t       dramAddress = getAddressFromSplitParts(fieldHighAddress, fieldLowAddress);
    auto&          pLowAddressField      = m_addressFieldInfoSet
                                 .emplace(fieldLowIndexOffset,
                                          new AddressFieldInfo(sectionName,
                                                               memorySectionId,
                                                               maskOutMemoryID(dramAddress),
                                                               fieldLowIndexOffset,
                                                               FIELD_ADDRESS_PART_LOW,
                                                               fieldMemoryType,
                                                               engineFieldIdentifier))
                                 .first->second;

    auto& pHighAddressField = m_addressFieldInfoSet
                                  .emplace(fieldHighIndexOffset,
                                           new AddressFieldInfo(sectionName,
                                                                memorySectionId,
                                                                maskOutMemoryID(dramAddress),
                                                                fieldHighIndexOffset,
                                                                FIELD_ADDRESS_PART_HIGH,
                                                                fieldMemoryType,
                                                                engineFieldIdentifier,
                                                                pLowAddressField))
                                  .first->second;

    pLowAddressField->setOrigin(node);
    pHighAddressField->setOrigin(node);

    pLowAddressField->setOtherAddressFieldInfo(pHighAddressField);
    pLowAddressField->setRoi(m_roi);
    pHighAddressField->setRoi(m_roi);
}

template<typename T>
std::pair<typename T::iterator, typename T::iterator>
BasicFieldsContainerInfo::retrieveSegment(uint32_t                  firstFieldIndexOffset,
                                          uint32_t                  lastFieldIndexOffset,
                                          BasicFieldsContainerInfo& partialAddrFieldContInfo,
                                          T&                        container,
                                          bool                      clone)

{
    using PairType     = typename T::value_type;
    using PtrFieldType = typename PairType::second_type;
    using FieldType    = typename PtrFieldType::element_type;
    auto first         = container.end();
    auto last          = container.end();

    for (auto it = container.begin(); it != container.end(); it++)
    {
        uint32_t fieldIndexOffset = it->first;

        if (fieldIndexOffset > lastFieldIndexOffset)
        {
            break;
        }
        if (fieldIndexOffset >= firstFieldIndexOffset)
        {
            uint32_t updatedFieldIndexOffset = fieldIndexOffset - firstFieldIndexOffset;

            // If the original AFCI is not needed anymore no need to clone
            auto modifiedInfo = clone ? std::dynamic_pointer_cast<FieldType>(it->second->clone()) : it->second;

            // set new relative/partial offset and add to new list
            modifiedInfo->setFieldIndexOffset(updatedFieldIndexOffset);
            partialAddrFieldContInfo.add(modifiedInfo);

            if (first == container.end())
            {
                first = it;
            }
            last = it;
        }
    }

    return std::make_pair(first, last);
}

template<typename T>
void BasicFieldsContainerInfo::retrieveAndDropSegment(uint32_t                  firstFieldIndexOffset,
                                                      uint32_t                  lastFieldIndexOffset,
                                                      BasicFieldsContainerInfo& partialAddrFieldContInfo,
                                                      T&                        container)
{
    typename T::iterator first;
    typename T::iterator last;
    std::tie(first, last) =
        retrieveSegment(firstFieldIndexOffset, lastFieldIndexOffset, partialAddrFieldContInfo, container, false);

    if (first != container.end())
    {
        // erase from original set to speedup compilation time
        container.erase(first, last);
    }
}
void BasicFieldsContainerInfo::retrieveAndDropSegment(uint32_t                  firstFieldIndexOffset,
                                                      uint32_t                  lastFieldIndexOffset,
                                                      BasicFieldsContainerInfo& partialAddrFieldContInfo)
{
    retrieveAndDropSegment(firstFieldIndexOffset,
                           lastFieldIndexOffset,
                           partialAddrFieldContInfo,
                           m_addressFieldInfoSet);
    retrieveAndDropSegment(firstFieldIndexOffset,
                           lastFieldIndexOffset,
                           partialAddrFieldContInfo,
                           m_basicFieldInfoSet);

    partialAddrFieldContInfo.setRoi(m_roi);
}

void BasicFieldsContainerInfo::retrieveSegment(uint32_t                  firstFieldIndexOffset,
                                               uint32_t                  lastFieldIndexOffset,
                                               BasicFieldsContainerInfo& partialAddrFieldContInfo)
{
    partialAddrFieldContInfo.setRoi(m_roi);
    retrieveSegment(firstFieldIndexOffset, lastFieldIndexOffset, partialAddrFieldContInfo, m_addressFieldInfoSet, true);
    retrieveSegment(firstFieldIndexOffset, lastFieldIndexOffset, partialAddrFieldContInfo, m_basicFieldInfoSet, true);
}

void BasicFieldsContainerInfo::add(const BasicFieldInfoPair& basicFieldInfoPair)
{
    basicFieldInfoPair.second->setRoi(m_roi);
    m_basicFieldInfoSet.emplace(basicFieldInfoPair);
}

BasicFieldsContainerInfo BasicFieldsContainerInfo::retrieveAndDropSegment(uint32_t firstFieldIndexOffset,
                                                                          uint32_t lastFieldIndexOffset)
{
    BasicFieldsContainerInfo newContainer;
    newContainer.setRoi(m_roi);
    retrieveAndDropSegment(firstFieldIndexOffset, lastFieldIndexOffset, newContainer);
    return newContainer;
}

BasicFieldsContainerInfo BasicFieldsContainerInfo::retrieveSegment(uint32_t firstFieldIndexOffset,
                                                                   uint32_t lastFieldIndexOffset)
{
    BasicFieldsContainerInfo newContainer;
    retrieveSegment(firstFieldIndexOffset, lastFieldIndexOffset, newContainer);
    for (const auto& item : newContainer.m_addressFieldInfoSet)
    {
        item.second->reconnectWithNewContainer(newContainer);
    }
    for (const auto& item : newContainer.m_basicFieldInfoSet)
    {
        item.second->reconnectWithNewContainer(newContainer);
    }
    return newContainer;
}

void BasicFieldsContainerInfo::add(const AddressFieldInfoSharedPtr& pAddressFieldInfo)
{
    pAddressFieldInfo->setRoi(m_roi);
    m_addressFieldInfoSet.emplace(pAddressFieldInfo->getFieldIndexOffset(), pAddressFieldInfo);
}

void BasicFieldsContainerInfo::add(const AddressFieldInfoPair& addressFieldInfoPair)
{
    addressFieldInfoPair.second->setRoi(m_roi);
    m_addressFieldInfoSet.emplace(addressFieldInfoPair);
}

void BasicFieldsContainerInfo::updateContainerId(const uint64_t& fieldContainerId)
{
    for (const auto& basicFieldInfo : m_addressFieldInfoSet)
    {
        basicFieldInfo.second->setContainerId(fieldContainerId);
    }

    for (const auto& basicFieldInfo : m_basicFieldInfoSet)
    {
        basicFieldInfo.second->setContainerId(fieldContainerId);
    }
}

AddressFieldInfoSharedPtr BasicFieldsContainerInfo::findAddress(uint32_t fieldIndex) const
{
    // nullptr is less than any real value, so upperbound is being used.
    auto it = m_addressFieldInfoSet.upper_bound({fieldIndex, nullptr});
    if (it == m_addressFieldInfoSet.end() || it->first != fieldIndex)
    {
        return nullptr;
    }
    return it->second;
}

bool BasicFieldsContainerInfo::find(uint32_t fieldIndex) const
{
    return findAddress(fieldIndex) != nullptr;
}

uint64_t BasicFieldsContainerInfo::getNextUniqueId()
{
    return ++stEngineFieldIdentifier;
}
void BasicFieldsContainerInfo::add(const BasicFieldInfoSharedPtr& pBasicFieldInfo)
{
    pBasicFieldInfo->setRoi(m_roi);
    m_basicFieldInfoSet.emplace(pBasicFieldInfo->getFieldIndexOffset(), pBasicFieldInfo);
}

DynamicShapeFieldInfo::DynamicShapeFieldInfo(uint32_t    fieldIndexOffset,
                                             FieldType   fieldType,
                                             ShapeFuncID smf,
                                             pNode       origin,
                                             NodeROI*    roi)
: BasicFieldInfo(fieldIndexOffset, BasicFieldsContainerInfo::getNextUniqueId(), fieldType, INVALID_CONTAINER_ID, roi),
  m_size(0),
  m_smf(smf),
  m_isMaskable(false),
  m_isUnskippable(false)
{
    setOrigin(origin);
}

DynamicExecuteFieldInfo::DynamicExecuteFieldInfo(FieldType fieldType, size_t signalCount, pNode origin, NodeROI* roi)
: DynamicShapeFieldInfo(0, fieldType, ShapeFuncID::SMF_DYNAMIC_EXE, std::move(origin), roi), m_signalCount(signalCount)
{}

void DynamicAddressFieldInfo::reconnectWithNewContainer(const BasicFieldsContainerInfo& bfci)
{
    if (m_addressFieldHigh)
    {
        m_addressFieldHigh = bfci.findAddress(m_addressFieldHigh->getFieldIndexOffset());
        HB_ASSERT(m_addressFieldHigh != nullptr, "couldn't find address in new bfci");
    }
    if (m_addressFieldLow)
    {
        m_addressFieldLow = bfci.findAddress(m_addressFieldLow->getFieldIndexOffset());
        HB_ASSERT(m_addressFieldLow != nullptr, "couldn't find address in new bfci");
    }
}

SyncObjectAddressFieldInfo::SyncObjectAddressFieldInfo(uint32_t tensorId, NodePtr origin)
: BasicFieldInfo(0, BasicFieldsContainerInfo::getNextUniqueId(), EFieldType::FIELD_SYNC_OBJECT_ADDRESS),
  m_tensorId(tensorId)
{
    setOrigin(origin);
}

McidFieldInfo::McidFieldInfo(uint32_t               fieldIndexOffset,
                             NodePtr                origin,
                             CacheMaintenanceAction cmAction)
: BasicFieldInfo(fieldIndexOffset, BasicFieldsContainerInfo::getNextUniqueId(), EFieldType::FIELD_CM_MCID, INVALID_CONTAINER_ID),
  m_cmAction(cmAction)
{
    setOrigin(origin);
}