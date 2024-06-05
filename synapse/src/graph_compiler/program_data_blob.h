#pragma once

#include "types.h"

class ProgramDataBlob
{
public:
    ProgramDataBlob() : deviceAddr(0), hostAddrPtr(nullptr), hostAddrSharedPtr(nullptr), binSize(0)
    {}

    virtual ~ProgramDataBlob() = default;

    ProgramDataBlob(deviceAddrOffset deviceAddr, char* hostAddr, uint64_t size)
    : deviceAddr(deviceAddr),
        hostAddrPtr(hostAddr),
        hostAddrSharedPtr(nullptr),
        binSize(size)
    {}

    ProgramDataBlob(deviceAddrOffset deviceAddr, std::shared_ptr<char> hostAddr, uint64_t size)
    : deviceAddr(deviceAddr),
        hostAddrPtr(nullptr),
        hostAddrSharedPtr(hostAddr),
        binSize(size)
    {}

    // Default copy c'tor is okay cause we don't need deep copy for the pointers here

    deviceAddrOffset        deviceAddr;
    char*                   hostAddrPtr;
    std::shared_ptr<char>   hostAddrSharedPtr;
    uint64_t                binSize;
};

class TPCProgramDataBlob : public ProgramDataBlob
{
public:
    TPCProgramDataBlob() : ProgramDataBlob()
    {}

    virtual ~TPCProgramDataBlob() = default;

    TPCProgramDataBlob(deviceAddrOffset deviceAddr, char* hostAddr, uint64_t size, kernelID kid)
        : ProgramDataBlob(deviceAddr, hostAddr, size), kid(kid){}

    TPCProgramDataBlob(deviceAddrOffset deviceAddr, std::shared_ptr<char> hostAddr, uint64_t size, kernelID kid)
        : ProgramDataBlob(deviceAddr, hostAddr, size), kid(kid){}

    kernelID kid;
};

struct ProgramDataBlobComparator
{
    bool operator() (const std::shared_ptr<ProgramDataBlob>& lhs, const std::shared_ptr<ProgramDataBlob>& rhs) const
    {
        return lhs->deviceAddr < rhs->deviceAddr;
    }
};

using ProgramDataBlobSet = std::set<std::shared_ptr<ProgramDataBlob>, ProgramDataBlobComparator>;
