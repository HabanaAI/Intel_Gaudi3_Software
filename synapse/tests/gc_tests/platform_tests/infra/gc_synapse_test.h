/*****************************************************************************
* Copyright (C) 2016 HabanaLabs, Ltd.
* All Rights Reserved.
*
* Unauthorized copying of this file, via any medium is strictly prohibited.
* Proprietary and confidential.
*
******************************************************************************
*/

#ifndef TESTS_SYNAPSE_TEST_HPP_
#define TESTS_SYNAPSE_TEST_HPP_

#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <gtest/gtest.h>
#include <linux/types.h>
#include <condition_variable>
#include "hpp/syn_context.hpp"
#include "infra/test_device_manager.h"
#include "infra/gc_tests_types.h"
#include "test_utils.h"
#include "syn_logging.h"
#include "perf_lib_layer_params.h"
#include "infra/settable.h"
#include "synapse_api.h"
#include "supported_devices_macros.h"
#include "infra/gc_test_configuration.h"

using namespace gc_tests;

struct SpatialReduction2DDef : public ns_AveragePooling::Params
{
    SpatialReduction2DDef()
    {
        pad_w_begin        = 0;
        pad_w_end          = 0;
        pad_h_begin        = 0;
        pad_h_end          = 0;
        kernel_w           = 1;
        kernel_h           = 1;
        stride_w           = 1;
        stride_h           = 1;
        dilation_w         = 1;
        dilation_h         = 1;
        pooling_convention = POOLING_CONVENTION_VALID;
        includePadding     = 0;
    }
};

struct RaggedSoftmaxParams
{
    int dim;
    RaggedSoftmaxParams(): dim(3) {}

};

struct IterationsCnt
{
    unsigned value;
};

inline IterationsCnt operator"" _iterations(unsigned long long value)
{
    return IterationsCnt {value};
}

struct ThreadsCnt
{
    unsigned value;
};

inline ThreadsCnt operator"" _threads(unsigned long long value)
{
    return ThreadsCnt {value};
}

using namespace std::literals::chrono_literals;
//
struct MultiThread
{
    std::chrono::milliseconds maxDuration;  // execute a test in a loop until execution time exceeds maxDuration
    unsigned                  iterations;   // execute a test in a loop this number of iteration
    unsigned                  nbThreads;    // execute a test in this number of threads

    MultiThread() : MultiThread(0ms) {}

    explicit MultiThread(ThreadsCnt nbThreads) : MultiThread(0ms, nbThreads) {}

    explicit MultiThread(IterationsCnt iterationsCnt, ThreadsCnt nbThreads = ThreadsCnt {})
    : maxDuration(0ms),
      iterations(iterationsCnt.value ? iterationsCnt.value : 1),
      nbThreads(std::max(nbThreads.value == 0 ? std::thread::hardware_concurrency() : nbThreads.value, 1u))
    {
    }

    explicit MultiThread(std::chrono::milliseconds maxDuration, ThreadsCnt nbThreads = ThreadsCnt {})
    : maxDuration(maxDuration),
      iterations(maxDuration == 0ms ? 1 : 0),
      nbThreads(std::max(nbThreads.value == 0 ? std::thread::hardware_concurrency() : nbThreads.value, 1u))
    {
    }
};

struct SingleThread : MultiThread
{
    SingleThread() : MultiThread(1_iterations, 1_threads) {}
    SingleThread(std::chrono::milliseconds maxDuration) : MultiThread(maxDuration, 1_threads) {}
    SingleThread(IterationsCnt iterationsCnt) : MultiThread(iterationsCnt, 1_threads) {}
};

void initGlobalConfManager();

class supportedDeviceType
{
public:
    enum eParsingState
    {
        INIT,
        READ_ACTION,
        MODIFY_SUPPORTED_DEVICES,
        SET_SUPPORTED_DEVICES,
        ERROR
    };

    enum eAction
    {
        INVALID = 1,
        READ_ANNOTATION,
        ADD_DEVICE,
        REMOVE_DEVICE
    };

    using SupportedDeviceConf = std::vector<std::variant<char, synDeviceType>>;

    supportedDeviceType(std::vector<synDeviceType>& confSuportedDevice) : m_confSuportedDevice(confSuportedDevice) {}
    void parseSupportedDeviceConf(SupportedDeviceConf& devicesToSupport);

private:
    std::vector<synDeviceType>& m_confSuportedDevice;
};

class SynTest :
    public ::testing::Test
{
public:
    static const char* pooling_in_layouts[];
    static const char* pooling_out_layouts[];
    static const char* conv2D_in_layouts[];
    static const char* conv2D_out_layouts[];
    static const char* conv3D_in_layouts[];
    static const char* conv3D_out_layouts[];

    static void ReleaseDevice();

    virtual void AcquireDevices();
    void ReleaseDevices();

    void SetTestFileName(std::string testName="");
    std::string GetTestFileName();
    void CleanTestIntermediatesFiles();
    void CleanTestIntermediatesFiles(const std::string& testName,
                                     bool               isRecipeExtRequired);
    static synStatus callIoctlMemory(void *arg);

    synTensor createTrainingTensor( unsigned               dims,
                                    synDataType            dataType,
                                    const unsigned*        tensorSize,
                                    bool                   isPersist,
                                    const char*            name,
                                    const synGraphHandle   graphHandle,
                                    synSectionHandle      *pGivenSectionHandle = nullptr,
                                    uint64_t               offset = 0);

protected:
    SynTest();
    ~SynTest()
    {
        DROP_LOGGER(SYN_TEST);
    }

    void EnableSimultaneousDeviceTest() { m_simultaneousDeviceExecution = true; }
    void SyncTestThreads();

    virtual unsigned _getNumOfDevices() const { return m_deviceCount; };

    virtual void SetUp();
    virtual void TearDown();

    virtual void SetUpTest() { SynTest::SetUp(); }
    virtual void TearDownTest() { SynTest::TearDown(); }

    virtual void setSupportedDevices(supportedDeviceType::SupportedDeviceConf supportedDeviceTypes);
    virtual void setTestPackage(TestPackage package);
    virtual bool shouldRunTest();

    unsigned _getDeviceId() const
    {
        assert((_getNumOfDevices() > 0) && "Unexpected number of devices");

        auto deviceId = m_threadDeviceId;
        if (deviceId == -1)
        {
            // for simultaneous multidevice test _getDeviceId returns thread_local device_id
            // this initialization happens in macros from supported_devices_macros.h
            // if those macros are not used - then m_threadDeviceId is not initialized
            // in this case get device_id using _getDeviceId(0)
            deviceId = _getDeviceId(0);
        }

        return deviceId;
    }

    unsigned _getDeviceId(unsigned which) const
    {
        assert( ( which < _getNumOfDevices() ) && "Unexpected number of devices" );
        return m_deviceIds[which];
    }
    virtual void afterSynInitialize();
    virtual void beforeSynDestroy();

    synStatus SwitchPacketToDeviceVA(__u64& addr, __u32& bufferSize);

    synStatus SwitchBufferToDeviceVA(uint64_t& addr, uint32_t& bufferSize);

    synStatus GetDramMemory(uint32_t size, uint64_t& handle, uint64_t& addr, bool contiguous);

    synStatus ReleaseDramMemory(uint64_t handle, uint64_t addr);

    synStatus AllocDramMemory(uint32_t size, uint64_t& handle, bool contiguous);

    synStatus FreeDramMemory(uint64_t handle);

    synStatus MapHostMemory(uint64_t host_addr, uint32_t size, uint64_t& device_addr);

    synStatus MapDramMemory(uint64_t handle, uint64_t& addr);

    synStatus UnmapDramMemory(uint64_t addr);

    synStatus UnmapHostMemory(uint64_t addr);

    synStatus UnmapMemory(uint64_t addr, uint32_t flags);

    synStatus retryDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName);

    void printProfileInformation();

    long double getRunningTime();

    void initSynapse();

    unsigned getNumOfAcquiredDevices() const { return m_deviceIds.size(); }

    // DATA
    // Keeping a vector at this level, for cases we will have SynTests that are using more than one device.
    bool                       m_isInitialized;
    std::vector<unsigned>      m_deviceIds;
    synDeviceType              m_deviceType;
    bool                       m_setupStatus;
    Settable<std::string>      m_testName;
    TestConfiguration          m_testConfig;
    std::optional<std::string> m_experimentalOrigVal;

    static constexpr const char* m_experimental_str = "ENABLE_EXPERIMENTAL_FLAGS";
    // multithreading support (used by TestClassWithSetTestConfiguration)
    MultiThread m_MultiThreadConf;
    // if several devices are connected - run test on all of them (used by TestClassWithSetTestConfiguration)
    bool                                m_simultaneousDeviceExecution;
    unsigned                            m_deviceCount;
    unsigned                            m_nbTestThreads        = 1;
    bool                                m_threadSyncInProgress = false;
    std::atomic<unsigned>               m_nbSyncWaitingThreads = 0;
    std::condition_variable             m_threadsSyncCV;
    std::mutex                          m_threadsSyncMtx;
    inline static thread_local unsigned m_threadDeviceId = -1;
    supportedDeviceType                 m_supportedDevices;

    std::vector<synSectionHandle>       m_sections;
    std::mutex                          m_sectionsMutex;

    TestConfig                                                  m_testGlobalConfig;
    std::vector<std::shared_ptr<TestDeviceManager::DeviceReleaser>> m_devices;

    std::optional<syn::Context> m_ctx;
};

template<typename T>
void prepareTensorInfo(synRecipeHandle recipe, T* tensorInfo, uint32_t totalNumOfTensors)
{
    const char* tensorNames[totalNumOfTensors] = {};
    uint64_t    tensorIds[totalNumOfTensors];
    uint32_t    i = 0;

    for (i = 0; i < totalNumOfTensors; ++i)
    {
        tensorNames[i] = tensorInfo[i].tensorName;
    }
    ASSERT_EQ(synTensorRetrieveIds(recipe, tensorNames, tensorIds, totalNumOfTensors), synSuccess);
    for (i = 0; i < totalNumOfTensors; i++)
    {
        tensorInfo[i].tensorId = tensorIds[i];
    }
}

/******************************************************/
void check_ok(synStatus status, const std::string& msg);

class testStreamHandle
{
public:
    testStreamHandle(const testStreamHandle&) = delete;
    testStreamHandle(const synDeviceId deviceId, std::string name);
    testStreamHandle(synStreamHandle handle, std::string name);
    ~testStreamHandle();
    synStreamHandle& get()     {return m_handle;}
    std::string      getName() {return m_name;}

private:
    synStreamHandle m_handle;
    std::string     m_name;
    const bool      m_shouldDestroy;
};

class testEventHandle
{
public:
    testEventHandle(const testEventHandle&) = delete;
    testEventHandle(testEventHandle &&rhs) noexcept;
    testEventHandle() {};
    ~testEventHandle();
    void record(testStreamHandle& stream);
    void wait(  testStreamHandle& stream);
    void init(std::string name, synDeviceId deviceId, const uint32_t flags = 0);
    synEventHandle get() {return m_handle;}

private:
    synEventHandle m_handle = nullptr;
    std::string    m_name;
};

class testTensor
{
public:
    testTensor(const testTensor&) = delete;
    testTensor(testTensor &&rhs) noexcept;
    testTensor(synTensor tensor, std::string name, bool shouldDestroy)
    : m_tensor(tensor), m_name(name), m_shouldDestroy(shouldDestroy)
    {
    }
    ~testTensor();
    bool isNull()                {return m_tensor == nullptr;}
    const std::string& getName() {return m_name;}
    synTensor getTensor()        {return m_tensor;}

private:
    synTensor   m_tensor = nullptr;
    std::string m_name;
    bool        m_shouldDestroy = false;
};

class testGraphHandle
{
public:

    struct CreatedTensorInfo
    {
        synTensor        tensor;
        synSectionHandle section;
    };

    testGraphHandle(const testGraphHandle&) = delete;
    testGraphHandle(std::string name, synDeviceType deviceType = synDeviceGaudi);
    testGraphHandle() : testGraphHandle("") {}
    testGraphHandle(synDeviceType deviceType) : testGraphHandle("", deviceType) {}
    ~testGraphHandle();

    synGraphHandle& get() {return m_handle;}
    synRecipeHandle& getRecipeHandle() { return m_recipeHandle; }

    void set_name(std::string name) {m_name = name;}
    synTensor getTensorHandle(std::string name);
    synStatus compile(synDeviceId deviceId);
    synStatus reallocateWorkspace(synDeviceId deviceId);
    synStatus
    launch(testStreamHandle& stream, synLaunchTensorInfo tensors[], uint32_t totalNumOfTensors, uint32_t flags);
    synStatus launchNoCheck(testStreamHandle& stream, synLaunchTensorInfo tensors[], uint32_t totalNumOfTensors);

    synStatus launchNoCheckWithFlags(testStreamHandle&   stream,
                                     synLaunchTensorInfo tensors[],
                                     uint32_t            totalNumOfTensors,
                                     uint32_t            flags);

    synStatus addTensor(unsigned          dims,
                        synDataType       dataType,
                        const unsigned   *tensorSize,
                        bool              isPersist,
                        std::string       name,
                        synSectionHandle *pSectionHandle = nullptr,
                        uint64_t          offset         = 0);

   synStatus getWorkspaceInfo(synDeviceId deviceId, uint64_t& workspaceAddr, uint64_t& workspaceSize);


private:

    synGraphHandle                m_handle = nullptr;
    std::string                   m_name;
    std::vector<testTensor>       m_tensorVec;
    synRecipeHandle               m_recipeHandle    = nullptr;
    uint64_t                      m_workspaceBuffer = 0;
    std::vector<synSectionHandle> m_sections;
};

testGraphHandle::CreatedTensorInfo createTrainingTensorAndSection(  unsigned               dims,
                                                                    synDataType            dataType,
                                                                    const unsigned*        tensorSize,
                                                                    bool                   isPersist,
                                                                    const char*            name,
                                                                    const synGraphHandle   graphHandle,
                                                                    synSectionHandle      *pGivenSectionHandle = nullptr,
                                                                    uint64_t               offset = 0);


#endif /* TESTS_SYNAPSE_TEST_HPP_ */
