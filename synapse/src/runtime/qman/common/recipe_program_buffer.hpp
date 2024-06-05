#pragma once

#include <stdint.h>
#include <string>

// Description: RecipeProgramBuffer
//      Holds a buffer and states (should delete, should map)
//      The un/mapping is done by user
//      The deletion is done by this class DTR

class RecipeProgramBuffer
{
public:
    RecipeProgramBuffer(uint64_t recipeId, char* buffer, uint64_t size, bool shouldDelete);
    ~RecipeProgramBuffer();

    void setMappingInformation(const char* mappingDescription);

    const char* getBuffer() const { return m_buffer; };

    const char* getMappingDescription() const { return m_mappingDescription.c_str(); };

    uint64_t getSize() const { return m_size; };

    bool shouldMap() const { return m_shouldMap; };

    uint64_t getRecipeId() const { return m_recipeId; };

    void     setMappedAddr(uint64_t addr) { m_mappedAddr = addr; }
    uint64_t getMappedAddr() const { return m_mappedAddr; }

private:
    char*       m_buffer;
    uint64_t    m_mappedAddr;
    std::string m_mappingDescription;
    uint64_t    m_size;
    bool        m_shouldDelete;
    bool        m_shouldMap;
    uint64_t    m_recipeId;
};