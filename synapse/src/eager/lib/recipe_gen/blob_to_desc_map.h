#pragma once

// eager includes (relative to src/eager/lib/)
#include "recipe_gen/blob_to_desc_map_structs.h"
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "internal/recipe.h"

namespace eager_mode
{
class RecipeHalBase;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Mapping result
///////////////////////////////////////////////////////////////////////////////////////////////////

// This struct is used to fill all mappings between blobs array and software structs.
// Each instance is associated to single (actual) descriptor.
struct Blob2DescMaps
{
    Blob2EngineIdsMap blob2EngineIdsMap;  // Blobs to engine ids
    Blob2StructPosMap execDescMap;        // Execution blobs to descriptor
    Blob2StructPosMap baseRegOffsetsMap;  // Base registers offsets to their ids
    WrAddrMapPosMap   wrAddrMapPosMap;    // Offsets of sections to base register ids
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Positions map creator base
///////////////////////////////////////////////////////////////////////////////////////////////////

// Blob2DescMaps builder to fill all mappings for blobs of a specific descriptor
class Blob2DescMapCreatorBase
{
public:
    void fillMaps(const recipe_t& recipe);

protected:
    Blob2DescMapCreatorBase(Blob2DescMaps&       blob2DescMaps,
                            EngineType           engineType,
                            SecondaryEngineType  secondaryEngineType,
                            const RecipeHalBase& recipeHal);
    virtual void             initEngineBlackList()                  = 0;
    virtual PatchPointNrType calcTensorId(StructSizeType structPos) = 0;
    void                     fillBlob2EngineIdsMap(const recipe_t& recipe);
    void                     fillBlob2DescMaps(const blob_t* blobs, BlobsNrType blobsNr);

private:
    void                initBlackList();
    void                addNewMapping(blob_t::EBlobType blobTypte,
                                      BlobsNrType       blobIdx,
                                      BlobSizeType      posInBlob,
                                      AsicRegType       reg,
                                      DataNrType        regsNr = 1);
    QmanCommandSizeType fillMapsForCommand(BlobsNrType blobIdx, const blob_t& blob, BlobSizeType cmdOffset);

protected:
    AsicRegsBlackList    m_asicRegsBlackList;
    const RecipeHalBase& m_recipeHal;

private:
    bool m_areMapsFilled = false;
    Blob2EngineIdsMapCreator m_blob2EngineIdsMap;
    ExecDescMapCreator       m_execDescMap;        // Descriptor
    BaseRegOffsetsMapCreator m_baseRegOffsetsMap;  // Cache base register
    WrAddrMapPosMapCreator   m_wrAddrMapPosMap;    // "WREG_64_LONG" commands
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Positions map creator template
///////////////////////////////////////////////////////////////////////////////////////////////////

template<EngineType ENGINE_TYPE>
class Blob2DescMapCreator final : public Blob2DescMapCreatorBase
{
public:
    Blob2DescMapCreator(Blob2DescMaps&       blob2DescMaps,
                        std::size_t          tensorsNr,
                        const RecipeHalBase& recipeHal,
                        SecondaryEngineType  secondaryEngineType = SecondaryEngineType::NONE)
    : Blob2DescMapCreatorBase(blob2DescMaps, ENGINE_TYPE, secondaryEngineType, recipeHal),
      m_tensorsNr(tensorsNr),
      m_secondaryEngineType(secondaryEngineType)
    {
    }

protected:
    // Optional ability to add descriptor registers to black list.
    // While adding register need to be aware of reg-bulk fragmentation. Those are registers may create gaps
    // when doing memcpy. In sparse case it's worth to copy the register than to split the bulk into two memcpy's.
    virtual void initEngineBlackList() override {}
    // Convert tensor HW regs into numerical index. Inputs first then outputs
    virtual PatchPointNrType calcTensorId(StructSizeType structPos) override;

private:
    const std::size_t         m_tensorsNr;
    const SecondaryEngineType m_secondaryEngineType;
};

}  // namespace eager_mode
