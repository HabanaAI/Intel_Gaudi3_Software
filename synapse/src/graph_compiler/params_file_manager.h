#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>

class ParamsManagerBase
{
public:
    virtual void append(const char* data, uint64_t size) = 0;
    virtual ~ParamsManagerBase() {}
    virtual bool finalize() = 0;
    virtual void getCurrentData(uint64_t size, char* dataPtr) = 0;

};

class ParamsSizeManager : public ParamsManagerBase
{
public:
    void append(const char *data, uint64_t size) override;
    uint32_t getCurrentSize() const;
    bool finalize() override;
    void getCurrentData(uint64_t size, char* dataPtr) override;
private:
    uint32_t m_CurrentSize = 0;
};



class ParamsManager: public ParamsManagerBase
{

public:

    ParamsManager();
    ~ParamsManager() {};

    void     setFileName(const char* fileName, bool addSuffix);
    void     append(const char* data, uint64_t size) override;
    uint64_t getLastOffset();
    uint32_t calculateCrc32();
    uint32_t getCrc32();
    bool     finalize() override;
    bool     saveToDisk();
    bool     loadFromDisk();
    void     getData(uint64_t offset, uint64_t size, char* dataPtr);
    void     getCurrentData(uint64_t size, char* dataPtr) override;
    bool     isEof();

private:

    void internalCalculateCrc32(const char* ptr, uint64_t size);

    uint64_t      m_offset;
    uint32_t      m_crc32;
    std::string   m_fileName;

    std::ofstream m_output;
    bool          m_outputInitialized;

    std::ifstream m_input;
};
