#include "recipe_program_buffer.hpp"

#include "log_manager.h"

RecipeProgramBuffer::RecipeProgramBuffer(uint64_t recipeId, char* buffer, uint64_t size, bool shouldDelete)
: m_buffer(buffer),
  m_mappedAddr(0),
  m_mappingDescription(),
  m_size(size),
  m_shouldDelete(shouldDelete),
  m_shouldMap(false),
  m_recipeId(recipeId)
{
}

RecipeProgramBuffer::~RecipeProgramBuffer()
{
    LOG_DEBUG(SYN_STREAM, "RecipeProgramBuffer DTR, shouldDelete {} {:#x}", m_shouldDelete, (uint64_t)m_buffer);
    if (m_shouldDelete)
    {
        delete[] m_buffer;
    }
}

void RecipeProgramBuffer::setMappingInformation(const char* mappingDescription)
{
    m_shouldMap          = true;
    m_mappingDescription = mappingDescription;
}