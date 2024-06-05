#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/recipe_defs.h"
#include "utils/general_defs.h"

// std includes
#include <vector>

namespace eager_mode
{
class RecipeHalBase;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Positions map creators for "WREG_32" and "WREG_BULK" commands
///////////////////////////////////////////////////////////////////////////////////////////////////

// Inclusive start and end to represent range of ASIC registers
class AsicRange
{
public:
    AsicRange(AsicRegType inclusiveFirst, AsicRegType inclusiveLast) : m_first(inclusiveFirst), m_last(inclusiveLast) {}

    AsicRegType    getFirst() const { return m_first; }
    AsicRegType    getLast() const { return m_last; }
    bool           contains(const AsicRegType& reg) const { return (reg >= m_first) && (reg <= m_last); }
    StructSizeType calcPosInStruct(AsicRegType reg) const { return reg - m_first; }
    static bool
    isSubRange(AsicRegType reg1Start, StructSizeType regs1Nr, AsicRegType reg2Start, StructSizeType regs2Nr);

private:
    AsicRegType m_first = 0;
    AsicRegType m_last  = 0;
};

class StructPosMapCreator
{
public:
    StructPosMapCreator(AsicRegType firstRegPos, StructSizeType sizeOfStruct)
    : m_range(firstRegPos, firstRegPos + (sizeOfStruct - 1) * sizeOfAsicRegVal)
    {
        EAGER_ASSERT((firstRegPos % sizeOfAsicRegVal) == 0, "Invalid structure");
        EAGER_ASSERT((sizeOfStruct % sizeOfAsicRegVal) == 0, "Invalid structure");
    }

    const AsicRange& getRange() const { return m_range; }

protected:
    // Descriptors lay within array of ASIC registers of 32bit each. In order to do the mapping between ASIC register
    // and its position in software struct, we need the ASIC id of first bit at software struct.
    //  - First: Id of first ASIC register laid in software struct.
    //  -  Last: Ditto but inclusive last id. Used to check if a register belongs to the struct
    const AsicRange m_range;
};

// Representation of mapping between position in blob and position in software struct or descriptor.
// It's associated to "REG_OFFSET" and "VALUE" fields of "WREG_32" and "WREG_BULK" commands.
// All positions are in bytes.
struct Blob2StructMappingElement
{
    BlobsNrType    blobIdx;    // Blob index to which the mapping belongs
    BlobSizeType   blobPos;    // Position in blob
    BlobSizeType   size64Pos;  // Position of "SIZE64" field in blob. It's set for TPC "WREG_BULK" commands only
    StructSizeType structPos;  // Position of the ASIC reg in software struct
    DataNrType     regsNr;     // Number of ASIC registers values to be mapped in a single chunk (WREG BULK)
};
// Contains all mapping elements between blobs array and a software structure
using Blob2StructPosMap = std::vector<Blob2StructMappingElement>;

// Logic to create position map between blob and software structs.
// Each instance should be associated to a single engine.
class Blob2StructPosMapCreator : public StructPosMapCreator
{
public:
    Blob2StructPosMapCreator(Blob2StructPosMap& blob2StructPosMap,
                             AsicRegType        firstRegPos,
                             StructSizeType     sizeOfStruct);
    bool addNewMapping(BlobsNrType blobIdx, BlobSizeType blobPos, AsicRegType regStart, DataNrType regsNr);

protected:
    Blob2StructPosMap& m_blob2StructPosMap;
};

// Creator of positions map between execution blobs and descriptor
struct ExecDescMapCreator : public Blob2StructPosMapCreator
{
    ExecDescMapCreator(Blob2StructPosMap&   blob2StructPosMap,
                       EngineType           engineType,
                       SecondaryEngineType  secondaryEngineType,
                       const RecipeHalBase& recipeHal);
};

// Creator of positions map between execution blobs and base register
struct BaseRegOffsetsMapCreator : public Blob2StructPosMapCreator
{
    BaseRegOffsetsMapCreator(Blob2StructPosMap& blob2StructPosMap, const RecipeHalBase& recipeHal);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Positions map creators for "WREG_64_LONG" commands
///////////////////////////////////////////////////////////////////////////////////////////////////

// Representation of fields that are not aligned to bytes
struct UnalignedPosType
{
    BlobSizeType    base;    // Field position in blobs (in bytes and aligned to sizeOfAsicRegVal)
    QmanCommandSizeType offset;  // Number of bits to shift left in order to write to the actual position
    AsicRegValType  mask;    // Mask of actual bytes (zeros on bits overlap with the field)

    UnalignedPosType(BitPosInBlobType posInBits, AsicRegValType fieldMaxVal);
    UnalignedPosType(const UnalignedPosType& unalignedPos, BlobSizeType cmdPos);
};

// Representation of addresses wrote relative to base register. All positions are in bytes.
// It's associated to "DREG_OFFSET", "OFFSET" and "REG" fields of "WREG_64_LONG" commands.
// All positions are in bytes.
struct WrAddrMappingElement
{
    BlobsNrType      blobIdx;     // Blob index to which the mapping belongs
    BlobSizeType     offsetPos;   // Position of "OFFSET" field in blob (in AsicRegValType units)
    UnalignedPosType baseRegPos;  // Position of "BASE" field in blob (it can be unaligned field)

    // The following will be calculated in two stages:
    //  1. Calculate structPos (of tensor address) while parsing commands
    //  2. Calculate tensorIdx by mapping structPos to tensor index (depends on engine)
    PatchPointNrType tensorIdx;  // Index of the tensor that produced this mapping element

    bool isQWord;  // If "true" then "OFFSET" points to 64bit data, otherwise 32bit
};
using WrAddrMapPosMap =
    std::vector<WrAddrMappingElement>;  // Contains all mapping elements between blob and a software structure

class WrAddrMapPosMapCreator : private StructPosMapCreator
{
public:
    WrAddrMapPosMapCreator(WrAddrMapPosMap&     wrAddrMapPosMap,
                           EngineType           engineType,
                           SecondaryEngineType  secondaryEngineType,
                           const RecipeHalBase& recipeHal);

    StructSizeType calcPosInDesc(AsicRegType asicDwRegOfAddr) const;
    void           addNewMapping(BlobsNrType             blobIdx,
                                 BlobSizeType            offsetPos,
                                 const UnalignedPosType& baseRegPos,
                                 PatchPointNrType        tensorIdx,
                                 bool                    isShortCmd);

private:
    WrAddrMapPosMap& m_wrAddrMapPosMap;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Map each blob to engine id
///////////////////////////////////////////////////////////////////////////////////////////////////

// Size of the vector is equal to blobs number. Content is engine ids.
// There are blobs that are shared between multiple engines, in this case the engine id will be the smallest.
using Blob2EngineIdsMap = std::vector<EngineIdType>;

class Blob2EngineIdsMapCreator
{
public:
    Blob2EngineIdsMapCreator(Blob2EngineIdsMap& blob2EngineIdsMap) : m_blob2EngineIdsMap(blob2EngineIdsMap) {}
    void init(BlobsNrType blobsNr);
    void addNewMapping(BlobsNrType blobIdx, EngineIdType engineId);
    void lock();

private:
    bool               m_isInitialized = false;
    bool               m_isLocked      = false;
    Blob2EngineIdsMap& m_blob2EngineIdsMap;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Black (and White) List
///////////////////////////////////////////////////////////////////////////////////////////////////

// Sorted list of hardware registers that are copied as is from recipe template into actual recipe
class AsicRegsBlackList
{
public:
    void addReg(AsicRegType reg);
    void addBlackRange(const AsicRange& newRange) { addRange(true, newRange); }
    void addWhiteRange(const AsicRange& newRange) { addRange(false, newRange); }
    void lock();
    bool isFound(AsicRegType reg) const;

private:
    void addRange(bool isBlack, const AsicRange& newRange);
    using AsicRegList = std::vector<AsicRegType>;
    static bool isFound(const AsicRegList& asicRegList, AsicRegType reg);

    using AsicRangeList = std::vector<AsicRange>;
    static bool isFound(const AsicRangeList& asicRangeList, AsicRegType reg);

private:
    bool          m_isLocked = false;  // Once setting this flag, adding new register becomes illegal
    AsicRegList   m_registers;         // Blacklist of ASIC registers
    AsicRangeList m_blackRanges;       // Blacklist of ranges
    AsicRangeList m_whiteRanges;       // Whitelist of ranges (ranges that are excluded from blacklist)
};

}  // namespace eager_mode