#include "blob_to_desc_map_structs.h"

// eager includes (relative to src/eager/lib/)
#include "recipe_hal_base.h"
#include "utils/general_defs.h"

// std includes
#include <algorithm>
#include <climits>


namespace eager_mode
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// Positions map creators for all commands
///////////////////////////////////////////////////////////////////////////////////////////////////

// Check if {reg1Start, regs1Nr} range fully contain {reg2Start, regs2Nr} or not overlap it.
// This is not a generic util. reg1 range should represent a descriptor.
bool AsicRange::AsicRange::isSubRange(AsicRegType    reg1Start,
                                      StructSizeType regs1Nr,
                                      AsicRegType    reg2Start,
                                      StructSizeType regs2Nr)
{
    // Check if {reg1Start, regs1Nr} and {reg2Start, regs2Nr} ranges have at least one common register
    if (reg1Start > reg2Start) return false;
    const AsicRegType reg2End = reg2Start + sizeOfAsicRegVal * (regs2Nr * asicRegsPerEntry - 1);
    if (reg1Start > reg2End) return false;
    const AsicRegType reg1End = reg1Start + sizeOfAsicRegVal * (regs1Nr * asicRegsPerEntry - 1);
    if (reg1End < reg2Start) return false;
    EAGER_ASSERT(reg1End >= reg2End, "Range1 must contain range 2 or not overlap it");
    return true;
}

Blob2StructPosMapCreator::Blob2StructPosMapCreator(Blob2StructPosMap& blob2StructPosMap,
                                                   AsicRegType        firstRegPos,
                                                   StructSizeType     sizeOfStruct)
: StructPosMapCreator(firstRegPos, sizeOfStruct), m_blob2StructPosMap(blob2StructPosMap)
{
    // Reserve enough memory to avoid additional extensions
    m_blob2StructPosMap.reserve(sizeOfStruct / sizeOfAsicRegVal);
}

bool Blob2StructPosMapCreator::addNewMapping(BlobsNrType  blobIdx,
                                             BlobSizeType blobPos,
                                             AsicRegType  regStart,
                                             DataNrType   regsNr)
{
    if (regsNr == 0)
    {
        return true;  // Nothing to be added
    }
    // If first reg is not included, it's an indication that it doesn't belong to struct associated to this instance
    if (!m_range.contains(regStart))
    {
        return false;
    }
    // Last registers can be outside the range this instance mapper covers. In this case exclude them.
    m_blob2StructPosMap.push_back({.blobIdx   = blobIdx,
                                   .blobPos   = blobPos,
                                   .size64Pos = 0,
                                   .structPos = m_range.calcPosInStruct(regStart),
                                   .regsNr    = regsNr});
    return true;
}

BaseRegOffsetsMapCreator::BaseRegOffsetsMapCreator(Blob2StructPosMap& blob2StructPosMap, const RecipeHalBase& recipeHal)
: Blob2StructPosMapCreator(blob2StructPosMap, recipeHal.getRegForBaseAddress(0), recipeHal.getBaseRegistersCacheSize())
{
}

ExecDescMapCreator::ExecDescMapCreator(Blob2StructPosMap&   blob2StructPosMap,
                                       EngineType           engineType,
                                       SecondaryEngineType  secondaryEngineType,
                                       const RecipeHalBase& recipeHal)
: Blob2StructPosMapCreator(blob2StructPosMap,
                           recipeHal.getFirstRegPosInDesc(engineType, secondaryEngineType),
                           recipeHal.getDescSize(engineType))
{
}

constexpr auto asicRegSizeInBits = static_cast<BitPosInBlobType>(sizeOfAsicRegVal) * CHAR_BIT;

UnalignedPosType::UnalignedPosType(BitPosInBlobType posInBits, AsicRegValType fieldMaxVal)
: base(posInBits / CHAR_BIT), offset(posInBits % asicRegSizeInBits), mask(~(fieldMaxVal << offset))
{
    EAGER_ASSERT((base % sizeOfAsicRegVal) == 0, "Unsupported command");
}

UnalignedPosType::UnalignedPosType(const UnalignedPosType& unalignedPos, BlobSizeType cmdPos)
: base(unalignedPos.base + cmdPos), offset(unalignedPos.offset), mask(unalignedPos.mask)
{
    EAGER_ASSERT((cmdPos % sizeOfAsicRegVal) == 0, "Invalid command position");
}

WrAddrMapPosMapCreator::WrAddrMapPosMapCreator(WrAddrMapPosMap&     wrAddrMapPosMap,
                                               EngineType           engineType,
                                               SecondaryEngineType  secondaryEngineType,
                                               const RecipeHalBase& recipeHal)
: StructPosMapCreator(recipeHal.getFirstRegPosInDesc(engineType, secondaryEngineType),
                      recipeHal.getDescSize(engineType)),
  m_wrAddrMapPosMap(wrAddrMapPosMap)
{
}

StructSizeType WrAddrMapPosMapCreator::calcPosInDesc(AsicRegType asicDwRegOfAddr) const
{
    const AsicRegType reg = asicDwRegOfAddr * sizeof(uint32_t);  // It's stored in DWords
    EAGER_ASSERT(m_range.contains(reg), "Address field is not in the descriptor");
    return m_range.calcPosInStruct(reg);
}

void WrAddrMapPosMapCreator::addNewMapping(BlobsNrType             blobIdx,
                                           BlobSizeType            offsetPos,
                                           const UnalignedPosType& baseRegPos,
                                           PatchPointNrType        tensorIdx,
                                           bool                    isShortCmd)
{
    m_wrAddrMapPosMap.push_back({.blobIdx    = blobIdx,
                                 .offsetPos  = offsetPos,
                                 .baseRegPos = std::move(baseRegPos),
                                 .tensorIdx  = tensorIdx,
                                 .isQWord    = !isShortCmd});
}

void Blob2EngineIdsMapCreator::init(BlobsNrType blobsNr)
{
    EAGER_ASSERT(!m_isInitialized, "Blobs to engine ids map is already initialized");
    m_isInitialized = true;
    m_blob2EngineIdsMap.resize(blobsNr, -1);  // Initialize with invalid value
}

void Blob2EngineIdsMapCreator::addNewMapping(BlobsNrType blobIdx, EngineIdType engineId)
{
    EAGER_ASSERT(m_isInitialized, "Blobs to engine ids map was not initialized");
    EAGER_ASSERT(!m_isLocked, "Blobs to engine ids map is locked");
    EAGER_ASSERT(blobIdx < m_blob2EngineIdsMap.size(), "Blob index is out of bound");
    m_blob2EngineIdsMap[blobIdx] = std::min(m_blob2EngineIdsMap[blobIdx], engineId);
}

void Blob2EngineIdsMapCreator::lock()
{
    EAGER_ASSERT(m_isInitialized, "Blobs to engine ids map was not initialized");
    EAGER_ASSERT(!m_isLocked, "Blobs to engine ids map is already locked");
    // Verify that all blobs got their mapping
    auto                        isValInvalid = [](EngineIdType engineId) { return (engineId == -1); };
    [[maybe_unused]] const bool isValid =
        !std::any_of(m_blob2EngineIdsMap.cbegin(), m_blob2EngineIdsMap.cend(), isValInvalid);
    EAGER_ASSERT(isValid, "Blobs to engine ids map is not fully initialized");
    m_isLocked = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Black (and White) List
///////////////////////////////////////////////////////////////////////////////////////////////////

// Add new register to the black list while it's unlocked
void AsicRegsBlackList::addReg(AsicRegType newReg)
{
    EAGER_ASSERT(!m_isLocked, "Black list is locked. It's illegal to add new registers");
    m_registers.push_back(newReg);
}

// Add new a new range to either black or white lists
void AsicRegsBlackList::addRange(bool isBlack, const AsicRange& newRange)
{
    EAGER_ASSERT(!m_isLocked, "Black list is locked. It's illegal to add new bases");
    auto& rangesList = isBlack ? m_blackRanges : m_whiteRanges;
    rangesList.push_back(newRange);
}

// Lock the list to prevent adding new registers.
// Sort the outcome as this is the assumption of the find operation.
void AsicRegsBlackList::lock()
{
    if (!m_isLocked)
    {
        m_isLocked = true;
        std::sort(m_registers.begin(), m_registers.end());
    }
}

// Check if certain register is found in the list
bool AsicRegsBlackList::isFound(AsicRegType reg) const
{
    EAGER_ASSERT(m_isLocked, "Black list needs to be locked");
    return !isFound(m_whiteRanges, reg) && (isFound(m_blackRanges, reg) || isFound(m_registers, reg));
}

bool AsicRegsBlackList::isFound(const AsicRegList& asicRegList, AsicRegType reg)
{
    static constexpr unsigned threshold = 500;  // Binary or iterative search
    if (asicRegList.size() == 1)
    {
        return (asicRegList.back() == reg);
    }
    else if (asicRegList.size() < threshold)
    {
        for (unsigned i = 0; i < asicRegList.size(); ++i)
        {
            if (asicRegList[i] >= reg)
            {
                return (asicRegList[i] == reg);
            }
        }
    }
    return (std::binary_search(asicRegList.begin(), asicRegList.end(), reg));
}

bool AsicRegsBlackList::isFound(const AsicRangeList& asicRangeList, AsicRegType reg)
{
    if (asicRangeList.size() == 1)
    {
        return (asicRangeList.back().contains(reg));
    }
    for (unsigned i = 0; i < asicRangeList.size(); ++i)
    {
        if (asicRangeList[i].contains(reg))
        {
            return true;
        }
    }
    return false;
}

}  // namespace eager_mode