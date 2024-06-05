#pragma once

static uint32_t _deviceId;

static std::map<std::string, uint64_t> dramMap;


static inline uint32_t _getDeviceId()
{
    return _deviceId;
}


static synStatus hbmAlloc_s(uint64_t size, uint64_t* addr, std::string name)
{
    if (dramMap.find(name) != dramMap.end())
    {
        *addr = dramMap[name];
        return synSuccess;
    }
    else
    {
        synStatus status = synDeviceMalloc(_getDeviceId(), size, 0, 0, addr);
        dramMap[name] = *addr;
        dramMap[name + "_wu"] = *addr;
        dramMap[name + "_wu_out"] = *addr;
        return status;
    }
}

static synStatus hbmFree_s(uint64_t addr, std::string name)
{
    if (dramMap.find(name) != dramMap.end())
    {
        dramMap.erase(name);
        dramMap.erase(name + "_wu");
        dramMap.erase(name + "_wu_out");
        return synDeviceFree(_getDeviceId(), addr, 0);
    }
    return synSuccess;
}

#define hbmAlloc hbmAlloc_s
#define hbmFree hbmFree_s

static inline void checkResult(float result, int i)
{
    if (std::isinf(result) || std::isnan(result))
    {
        ASSERT_TRUE(false && "Invalid loss in iteration");
    }
}


static void memcpyTensorData(uint64_t src, uint64_t dst, unsigned sizeBytes, synDmaDir direction, synStreamHandle streamHandle)
{
    synStatus status;
    bool needToDestroy = false;

    if (streamHandle == 0)
    {
        status = synStreamCreateGeneric(&streamHandle, _getDeviceId(), 0);
        ASSERT_TRUE((status == synSuccess));
        needToDestroy = true;
    }

    status = synMemCopyAsync(streamHandle, src, sizeBytes, dst, direction);
    ASSERT_TRUE((status == synSuccess));

    status = synStreamSynchronize(streamHandle);
    ASSERT_TRUE((status == synSuccess));

    if (needToDestroy)
    {
        synStreamDestroy(streamHandle);
    }
}

static void downloadTensorData(void* data, uint64_t tensorAddr, unsigned sizeBytes, synStreamHandle streamHandle)
{
    memcpyTensorData((uint64_t)data, tensorAddr, sizeBytes, HOST_TO_DRAM, streamHandle);
}

static void uploadTensorData(uint64_t tensorAddr, void* data, unsigned sizeBytes, synStreamHandle streamHandle)
{
    memcpyTensorData(tensorAddr, (uint64_t)data, sizeBytes, DRAM_TO_HOST, streamHandle);
}

template<typename T>
static void setSingleValueInDram(unsigned deviceId, T value, uint64_t dram_address, synStreamHandle streamHandle = 0)
{
    synStatus status;
    void* data;
    status = synHostMalloc(deviceId, sizeof(T), 0, &data);
    assert((status == synSuccess));
    UNUSED(status);
    *(static_cast<T*>(data)) = value;
    downloadTensorData(data, dram_address, sizeof(T), streamHandle);
    synHostFree(deviceId, data, 0);
}


template<typename T>
static  T readSingleValueInDram(unsigned deviceId, uint64_t dram_address, synStreamHandle streamHandle = 0)
{
    synStatus status;
    void* data;
    status = synHostMalloc(deviceId, sizeof(T), 0, &data);
    assert((status == synSuccess));
    UNUSED(status);
    uploadTensorData(dram_address, data, sizeof(T), streamHandle);
    T val = *(static_cast<T*>(data));
    synHostFree(deviceId, data, 0);
    return val;
}