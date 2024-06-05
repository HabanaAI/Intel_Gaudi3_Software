#include "params_file_manager.h"

#include "infra/defs.h"
#include "infra/filesystem.h"

#include "utils.h"

#include <algorithm>

void ParamsSizeManager::append(const char* data, uint64_t size)
{
    if (size == 0) return;
    m_CurrentSize += size;
}

uint32_t ParamsSizeManager::getCurrentSize() const
{
    return m_CurrentSize;
}

bool ParamsSizeManager::finalize()
{
    //Doing nothing as it is currently irrelevant for this class
    return true;
}

void ParamsSizeManager::getCurrentData(uint64_t size, char* dataPtr)
{
    //Doing nothing as it is currently irrelevant for this class
}

ParamsManager::ParamsManager()
    : m_offset(0),
      m_crc32(0),
      m_outputInitialized(false)
{
}

bool ParamsManager::finalize()
{
    return saveToDisk();
}

void ParamsManager::setFileName(const char* fileName, bool addSuffix)
{
    HB_ASSERT(((fileName != nullptr) && (fileName[0] != '\0')), "Params file name cannot be empty or null");

    if ((fileName != nullptr) && (fileName[0] != '\0'))
    {
        m_fileName = std::string(fileName);
        if (addSuffix == true)
        {
            m_fileName.append(".bin");
        }
        LOG_DEBUG(GC, "Opened params-file (name: {})", m_fileName);
    }
}

//TODO: Make smart saving to disk when appending >X data
void ParamsManager::append(const char* data, uint64_t size)
{
    if (!m_outputInitialized)
    {
        // Define the path to the desired file
        fs::path filePath = m_fileName;

        std::error_code errCode;
        // Create the directories if they don't exist
        fs::path parentFilePath = filePath.parent_path();
        if ((!parentFilePath.empty()) &&
            (!fs::create_directories(parentFilePath, errCode)) &&
            // Ensure failure not due to - Directory already exists
            (errCode.value() != 0))
        {
            LOG_ERR(GC, "Failed to create directory {} due to {}", parentFilePath, errCode);
            return;
        }

        m_output.open(m_fileName, std::ios::binary);
        m_outputInitialized = true;
    }

    if (size == 0 ) return;
    HB_ASSERT((data != nullptr), "Cannot append empty data to params");
    //insert copy of the data
    m_output.write(data, size);
    internalCalculateCrc32(data, size);
    m_output.flush();
    m_offset += size ;
}


uint64_t ParamsManager::getLastOffset()
{
    return m_offset;
}

void ParamsManager::internalCalculateCrc32(const char* ptr, uint64_t size)
{
    m_crc32 = crc32(ptr, size, m_crc32);
}

uint32_t ParamsManager::calculateCrc32()
{
    return m_crc32;
}

uint32_t ParamsManager::getCrc32()
{
    return m_crc32;
}

bool ParamsManager::saveToDisk()
{
    uint32_t crc32 = calculateCrc32();

    //add calculated crc32 to the end of the binary params file
    append((char*)&crc32, sizeof(crc32));
    LOG_DEBUG(GC, "Added to params file crc32: {}, number of bytes saved: {}", crc32, sizeof(crc32));

    HB_ASSERT(!m_fileName.empty(), "Params file was not initialized");
    try
    {
        m_output.close();
        LOG_DEBUG(GC, "Saved params file {} to disk", m_fileName);
    }
    catch(std::ofstream::failure &e) //How can we get here if exceptions are not set?
    {
        LOG_ERR(GC, "Cannot Open/Write file: {}", m_fileName);
        return false;
    }

    return true;
}

void ParamsManager::getData(uint64_t offset, uint64_t size, char* dataPtr)
{
    m_input.seekg(offset, std::ios::beg);
    m_input.read(dataPtr, size);
}

void ParamsManager::getCurrentData(uint64_t size, char* dataPtr)
{
    if (size == 0)
    {
        return;
    }

    if (!dataPtr)
    {
        char* dummy_ptr = new char[size];
        m_input.read(dummy_ptr, size);
        delete[] dummy_ptr;
        return;
    }
    m_input.read(dataPtr, size);
}

bool ParamsManager::isEof()
{
    return m_input && m_input.peek() == EOF;
}

bool ParamsManager::loadFromDisk()
{
    HB_ASSERT(!m_fileName.empty(), "Params file was not initialized");
    try
    {
        if (!m_input.is_open())
        {
            m_input.open(m_fileName, std::ios::binary);
        }

        if(m_input.fail())
        {
            LOG_ERR(GC, "param file {} cannot be accessed, errno {} {}", m_fileName, errno, strerror(errno));
            return false;
        }

        m_input.unsetf(std::ios::skipws);

        // Calculate file's size:
        std::streampos fileSize;
        m_input.seekg(0, std::ios::end);
        fileSize = m_input.tellg();

        m_input.seekg((-1) * sizeof(m_crc32), std::ios_base::end);
        m_input.read ((char*)&m_crc32,sizeof(m_crc32));

        m_input.seekg(0, std::ios::beg);

        if(fileSize == 0)
        {
            LOG_ERR(GC, "param file {} is empty, with no crc32", m_fileName);
            return false;
        }
        LOG_DEBUG(GC, "param file {} loaded, file size: {}, crc32: {}", m_fileName, fileSize, m_crc32);
    }
    catch (const std::ifstream::failure& e) //How can we get here if exceptions are not set?
    {
        LOG_ERR(GC, "Cannot Open/Read file: {} err {} errno {} {}", m_fileName, e.what(), errno, strerror(errno));
        return false;
    }
    return true;
}
