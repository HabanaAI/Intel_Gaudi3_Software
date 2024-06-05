#include <atomic>
#include "recipe_program.h"
#include "infra/defs.h"
#include "recipe.h"
#include "infra/fasthash.h"
#include "graph_compiler/utils.h"
#include "recipe_allocator.h"
#include "habana_global_conf.h"
#include "hal_conventions.h"
#include "sstream"

static const bool debugProgramIndices = false; // set to true to log program indices

RecipeProgram::RecipeProgram(unsigned engineId, HabanaDeviceType devType, bool isSetup /* = false*/)

: m_engineId(engineId), m_deviceType(devType), m_isSetup(isSetup)
{
}

void RecipeProgram::insertBlobIndex(uint64_t blobIdx)
{
    m_blobIndices.push_back(blobIdx);
    m_blobMetaData.push_back(Settable<BlobMetaData>());
    m_hash.unset();
}

void RecipeProgram::insertBlobIndex(uint64_t blobIdx, Settable<BlobMetaData> blobMD)
{
    m_blobIndices.push_back(blobIdx);
    m_blobMetaData.push_back(blobMD);
    m_hash.unset();
}

unsigned RecipeProgram::getEngineId() const
{
    return m_engineId;
}

void RecipeProgram::calcHash() const
{
    m_hash.set(fasthash(m_blobIndices.data(),  m_blobIndices.size() *sizeof(m_blobIndices.at(0))));
}

uint64_t RecipeProgram::getHash() const
{
    if (!m_hash.is_set())
    {
        calcHash();
    }

    return m_hash.value();
}

bool RecipeProgram::isSetup() const
{
    return m_isSetup;
}

void RecipeProgram::serialize(program_t* pProgram, RecipeAllocator* pRecipeAlloc) const
{
    HB_ASSERT_PTR(pProgram);
    uint64_t numOfIndices = m_blobIndices.size();

    pProgram->program_length = numOfIndices ;
    pProgram->blob_indices   = (uint64_t*)pRecipeAlloc->allocate(numOfIndices * sizeof(uint64_t));

    if (numOfIndices != 0)
    {
        memcpy(pProgram->blob_indices, m_blobIndices.data(), sizeof(uint64_t) * numOfIndices);
    }
}

void RecipeProgram::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN)) return;

    LOG_DEBUG(RECIPE_GEN, "      engine ID = {}, {} stage", m_engineId, m_isSetup ? "Activate" : "Execute");
    LOG_DEBUG(RECIPE_GEN, "      num indices = {}", m_blobIndices.size());

    // this dump a lot of indices to the log file
    if (debugProgramIndices)
    {
        std::stringstream indices;
        std::string sp = "";
        std::for_each(m_blobIndices.begin(), m_blobIndices.end(), [&](uint64_t v){indices << sp << v; sp = std::string(",");});
        LOG_DEBUG(RECIPE_GEN, "      blob indices = {}", indices.str());
    }
}

RecipeProgramContainer::RecipeProgramContainer()
{
}

unsigned RecipeProgramContainer::getNumPrograms() const
{
    return m_programs.size();
}

RecipeProgram&
RecipeProgramContainer::getProgram(unsigned engineId, HabanaDeviceType devType, unsigned& programIndex, bool isSetup)
{
    for (unsigned i = 0; i < m_programs.size(); ++i)
    {
        if (m_programs[i].getEngineId() == engineId && m_programs[i].isSetup() == isSetup)
        {
            programIndex = i;
            return m_programs[i];
        }
    }

    // No existing program for 'engineId' and 'isSetup', create new program
    m_programs.emplace_back(RecipeProgram(engineId, devType, isSetup));

    programIndex = m_programs.size() - 1;

    return m_programs[programIndex];
}

const RecipeProgram& RecipeProgramContainer::getProgramByIndex(unsigned idx) const
{
    HB_ASSERT(idx < m_programs.size(), "Invalid program index (Smaller than program vector size)");
    return m_programs[idx];
}

RecipeProgram& RecipeProgramContainer::getProgramByIndex(unsigned idx)
{
    HB_ASSERT(idx < m_programs.size(), "Invalid program index (Smaller than program vector size)");
    return m_programs[idx];
}

void RecipeProgramContainer::eraseProgramByIndex(unsigned idx)
{
    HB_ASSERT(idx < m_programs.size(), "Invalid program index (Smaller than program vector size)");
    m_programs.erase(m_programs.begin() + idx);
}

void RecipeProgramContainer::serialize(uint32_t*        pNumPrograms,
                                       program_t**      ppPrograms,
                                       RecipeAllocator* pRecipeAlloc) const
{
    HB_ASSERT_PTR(pNumPrograms);
    HB_ASSERT_PTR(ppPrograms);

    *pNumPrograms = m_programs.size();
    *ppPrograms        = (program_t*)pRecipeAlloc->allocate(*pNumPrograms * sizeof(program_t));
    program_t* pFiller = *ppPrograms;

    for (unsigned i = 0; i < m_programs.size(); ++i)
    {
        m_programs[i].serialize(pFiller, pRecipeAlloc);
        pFiller++;
    }
}

void RecipeProgramContainer::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN)) return;

    LOG_DEBUG(RECIPE_GEN, "  Program Container Dump:");
    LOG_DEBUG(RECIPE_GEN, "    Number of programs = {}", m_programs.size());
    uint64_t i = 0;
    for (auto program : m_programs)
    {
        LOG_DEBUG(RECIPE_GEN, "    Program at index {}:", i++);
        program.print();
    }
}

void ShapePlaneInfoContainer::addTensor(const pTensor& tensor)
{
    // Add the tensor to the tensors array if it isn't there already.
    if (tensor != nullptr && m_tensorIDToIndex.find(tensor->getId()) == m_tensorIDToIndex.end())
    {
        size_t index = getAmountOfTensors();
        m_tensorIDToIndex[tensor->getId()] = index;
        tensor->setShapePlaneIndex(index);
        m_indexToTensorMap[index] = tensor;
    }
}
