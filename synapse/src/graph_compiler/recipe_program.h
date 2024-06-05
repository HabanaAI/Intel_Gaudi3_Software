#pragma once

#include <cstdint>
#include <vector>
#include "settable.h"
#include "habana_global_conf.h"
#include "tensor.h"
#include "habana_device_types.h"
#include "recipe_blob.h"

struct program_t;
class RecipeAllocator;

class RecipeProgram
{
public:
    RecipeProgram() = delete;

    RecipeProgram(unsigned engineId, HabanaDeviceType devType, bool isSetup = false);

    void             insertBlobIndex(uint64_t blobIdx);
    void             insertBlobIndex(uint64_t blobIdx, Settable<BlobMetaData> blobMD);
    bool             isSetup() const;
    unsigned         getEngineId() const;
    void             calcHash() const;
    uint64_t         getHash() const;
    void             serialize(program_t* pProgram, RecipeAllocator* pRecipeAlloc) const;
    void             print() const;
    HabanaDeviceType getDeviceType() const { return m_deviceType; }

    bool     isInitSfg() const { return m_sfgSyncObjInitValue.is_set(); }
    unsigned getInitSfgValue() const { return m_sfgSyncObjInitValue.value(); }
    void     setInitSfgValue(unsigned value) { m_sfgSyncObjInitValue.set(value); }
    void     setPreNodesRolloverIds(const std::set<unsigned>& rolloverIds) { m_preNodesRolloverIds = rolloverIds; }
    const std::set<unsigned>& getPreNodesRolloverIds() const { return m_preNodesRolloverIds; }

    const std::vector<uint64_t>& blobIndicies() const
    {
        return m_blobIndices;
    }

    const std::vector<Settable<BlobMetaData>>& getBlobsMetaData() const { return m_blobMetaData; }

private:
    std::vector<uint64_t>               m_blobIndices;
    std::vector<Settable<BlobMetaData>> m_blobMetaData;
    unsigned                            m_engineId;
    HabanaDeviceType                    m_deviceType;
    mutable Settable<uint64_t>          m_hash;
    bool                                m_isSetup;
    Settable<unsigned>                  m_sfgSyncObjInitValue;
    std::set<unsigned>                  m_preNodesRolloverIds;
};


class RecipeProgramContainer
{
public:
    RecipeProgramContainer();

    RecipeProgram&
    getProgram(unsigned engineId, HabanaDeviceType devType, unsigned& programIndex, bool isSetup = false);

    unsigned             getNumPrograms() const;
    RecipeProgram&       getProgramByIndex(unsigned idx);
    const RecipeProgram& getProgramByIndex(unsigned idx) const;
    void                 eraseProgramByIndex(unsigned idx);
    void                 serialize(uint32_t* pNumPrograms, program_t** ppPrograms, RecipeAllocator* pRecipeAlloc) const;
    void                 print() const;

private:

    std::vector<RecipeProgram> m_programs;
};

class ShapePlaneInfoContainer
{
public:
    ShapePlaneInfoContainer() {};

    void addTensor(const pTensor& tensor);

    size_t getAmountOfTensors() const { return m_indexToTensorMap.size(); }

    pTensor getTensorByIndex(size_t index) const { return m_indexToTensorMap.at(index); }

    size_t getTensorIndexByID(size_t tensorID) const { return m_tensorIDToIndex.at(tensorID); }
private:
    // These maps are initialized in the shape plane recipe generation.
    // They are used to create the global tensor array, and to make sure there is a single
    // copy of each tensor there.
    std::unordered_map<uint32_t, uint32_t> m_tensorIDToIndex;
    std::unordered_map<uint32_t, pTensor>  m_indexToTensorMap;
};
