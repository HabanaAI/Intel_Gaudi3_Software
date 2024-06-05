#include "synapse_test.hpp"

#include "runtime/common/osal/buffer_allocator.hpp"
#include "habana_global_conf.h"
#include "supported_devices_macros.h"
#include "synapse_common_types.h"

#include "runtime/common/syn_singleton.hpp"
#include "runtime/common/osal/osal.hpp"
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <string>
#include <sstream>
#include <variant>

const char* SynTest::pooling_in_layouts[]  = {"CWHN"};
const char* SynTest::pooling_out_layouts[] = {"CWHN"};
const char* SynTest::conv2D_in_layouts[]   = {"CWHN", "KCSR", "", "CWHN", ""};
const char* SynTest::conv2D_out_layouts[]  = {"CWHN"};
const char* SynTest::conv3D_in_layouts[]   = {"CWHDN", "KCSRQ", "", "CWHDN", ""};
const char* SynTest::conv3D_out_layouts[]  = {"CWHDN"};

typedef eHostAddrToVirtualAddrMappingStatus eMappingStatus;

SynTest::SynTest()
: m_isInitialized(false),
  m_deviceType(synDeviceTypeInvalid),
  m_MultiThreadConf(SingleThread()),
  m_simultaneousDeviceExecution(false),
  m_deviceCount(1),
  m_supportedDevices(m_testConfig.m_supportedDeviceTypes)
{
    if (getenv("SYN_DEVICE_TYPE") != nullptr)
    {
        m_deviceType = (synDeviceType)std::stoi(getenv("SYN_DEVICE_TYPE"));
    }
    if (getenv("NUM_OF_DEVICES") != nullptr)
    {
        m_testConfig.m_numOfTestDevices = (uint16_t)std::stoi(getenv("NUM_OF_DEVICES"));
    }
    m_setupStatus = false;
    CREATE_LOGGER(SYN_TEST, SYNAPSE_LOG_SINK_FILE, 1024 * 1024, 5);
}

void SynTest::SetTestFileName(std::string testName)
{
    if (m_testName.is_set())
    {
        LOG_ERR(SYN_API, "Test name already set with {}", m_testName.value());
        return;
    }

    // If empty string get the name from gtest infra
    if (testName == "")
    {
        testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    }

    // remove / char in case it appears in string
    std::string toReplace   = "/";
    std::string replaceWith = "_";
    std::size_t pos         = testName.find(toReplace);
    if (pos != std::string::npos)
    {
        testName = testName.replace(pos, toReplace.length(), replaceWith);
    }

    m_testName.set(testName);
}

std::string SynTest::GetTestFileName()
{
    if (!m_testName.is_set())
    {
        return "";
    }
    else
    {
        return (m_testName.value() + ".recipe");
    }
}

void SynTest::CleanTestIntermediatesFiles(const std::string& testName, bool isRecipeExtRequired)
{
    std::string topologyName = testName;
    if (isRecipeExtRequired)
    {
        topologyName += ".recipe";
    }

    std::string topologyBinFileName = topologyName + ".bin";
    std::string gcfgUsedFileName    = topologyName + ".used";

    if (!GCFG_PRESERVE_TESTS_RECIPE.value())
    {
        std::remove(topologyName.c_str());
    }
    std::remove(topologyBinFileName.c_str());
    std::remove(gcfgUsedFileName.c_str());
}

void SynTest::CleanTestIntermediatesFiles()
{
    if (m_testName.is_set())
    {
        CleanTestIntermediatesFiles(m_testName.value(), true);
    }
}

void SynTest::initSynapse()
{
    synStatus status;
    TURN_ON_TRACE_MODE_LOGGING();

    // destroy previous synapse instance (if it was not destroyed by mistake)
    status = synDestroy();
    if (status != synUninitialized)
    {
        std::cerr << "synDestroy succeeded but should fail! synInitialize was not paired "
                     "with synDestroy. check the previous test!";
    }
    TURN_OFF_TRACE_MODE_LOGGING();
    status = synInitialize();
    ASSERT_EQ(status, synSuccess) << "synInitialize failed!";

    afterSynInitialize();

    m_isInitialized = true;

    if (m_simultaneousDeviceExecution)
    {
        uint32_t deviceCount = 0;
        ASSERT_EQ(synDeviceGetCountByDeviceType(&deviceCount, m_deviceType), synSuccess);

        m_deviceCount = std::min(uint32_t(m_testConfig.m_numOfTestDevices), deviceCount);
        if (m_deviceCount == 0)
        {
            m_deviceCount = 1;
        }
        EXPECT_EQ(m_deviceCount, m_testConfig.m_numOfTestDevices)
            << "number of actual devices does not match NUM_OF_DEVICES envvar";
    }
    else
    {
        m_deviceCount = 1;
    }
}

void SynTest::SyncTestThreads()
{
    if (m_nbTestThreads == 1)
    {
        return;
    }

    // protection against 2 sequential syncs
    // when some threads try to start the second sync while some threads did not exit the previous sync
    if (m_threadSyncInProgress)
    {
        // wait till the end of sync
        std::unique_lock lck(m_threadsSyncMtx);
        m_threadsSyncCV.wait(lck, [this]() { return !m_threadSyncInProgress; });
    }

    if (++m_nbSyncWaitingThreads == m_nbTestThreads)
    {
        // only one thread will wake up all the rest
        std::unique_lock lck(m_threadsSyncMtx);
        m_threadSyncInProgress = true;
        m_threadsSyncCV.notify_all();
    }
    else
    {
        std::unique_lock lck(m_threadsSyncMtx);
        m_threadsSyncCV.wait_for(lck, std::chrono::seconds(2), [this]() { return m_threadSyncInProgress; });
    }
    if (--m_nbSyncWaitingThreads == 0)
    {
        std::unique_lock lck(m_threadsSyncMtx);
        m_threadSyncInProgress = false;
        m_threadsSyncCV.notify_all();
    }
}
void SynTest::SetUp()
{
    const char* experimentalVal = getenv(m_experimental_str);

    if (!experimentalVal)
    {
        setenv(m_experimental_str, "1", true);
        m_experimentalOrigVal = "0";  // indicate that env var was created by the test
    }

    initSynapse();

    AcquireDevices();
    if (!shouldRunTest()) GTEST_SKIP() << m_testConfig.skipReason();

    SetTestFileName();

    m_setupStatus = true;
}

void SynTest::TearDown()
{
    if (m_setupStatus)
    {
        // Remove the env var when it was created by the test
        if (m_experimentalOrigVal.has_value())
        {
            unsetenv(m_experimental_str);
        }

        printProfileInformation();

        CleanTestIntermediatesFiles();
        ReleaseDevices();
    }

    for (auto section: m_sections)
    {
        ASSERT_EQ(synSuccess, synSectionDestroy(section));
    }
    m_sections.clear();

    if (m_isInitialized)
    {
        beforeSynDestroy();

        auto status = synDestroy();
        ASSERT_EQ(status, synSuccess) << "Failed to destroy synapse";
    }
}

void supportedDeviceType::parseSupportedDeviceConf(SupportedDeviceConf& devicesToSupport)
{
    eParsingState                 nextState(INIT);
    eAction                       action(INVALID);
    std::vector<synDeviceType>    supportedDevices;
    SupportedDeviceConf::iterator iter = devicesToSupport.begin();

    do
    {
        switch (nextState)
        {
            case INIT:
                ASSERT_FALSE(devicesToSupport.empty());
                if (std::holds_alternative<char>(devicesToSupport.front()))
                {
                    nextState = READ_ACTION;
                    action    = READ_ANNOTATION;
                }
                else
                {
                    m_confSuportedDevice.clear();
                    nextState = SET_SUPPORTED_DEVICES;
                    action    = ADD_DEVICE;
                }
                break;
            case READ_ACTION:
                ASSERT_TRUE(READ_ANNOTATION);
                action    = (std::get<char>(*iter) == '+') ? ADD_DEVICE : REMOVE_DEVICE;
                iter      = std::next(iter);
                nextState = MODIFY_SUPPORTED_DEVICES;
                break;
            case MODIFY_SUPPORTED_DEVICES:
            {
                ASSERT_TRUE((action == ADD_DEVICE) || (action == REMOVE_DEVICE));
                auto deviceType = std::get<synDeviceType>(*iter);
                iter            = std::next(iter);
                if (action == ADD_DEVICE)
                {
                    m_confSuportedDevice.push_back(deviceType);
                }
                else if (action == REMOVE_DEVICE)
                {
                    m_confSuportedDevice.erase(
                        std::remove(m_confSuportedDevice.begin(), m_confSuportedDevice.end(), deviceType),
                        m_confSuportedDevice.end());
                }
                else
                {
                    LOG_ERR(SYN_TEST, "Action {} not supported in state MODIFY_SUPPORTED_DEVICE ", action);
                    ASSERT_TRUE(false);
                }
                nextState = READ_ACTION;
                break;
            }
            case SET_SUPPORTED_DEVICES:
            {
                auto deviceType = std::get<synDeviceType>(*iter);
                iter            = std::next(iter);
                m_confSuportedDevice.push_back(deviceType);
                break;
            }
            case ERROR:
                LOG_ERR(SYN_TEST, "Illegal supported device configuration");
                ASSERT_TRUE(false);
                break;
            default:
                ASSERT_TRUE(false);
        }
    } while (iter != devicesToSupport.end());
}

void SynTest::setSupportedDevices(supportedDeviceType::SupportedDeviceConf supportedDeviceTypes)
{
    m_supportedDevices.parseSupportedDeviceConf(supportedDeviceTypes);
}

void SynTest::setTestPackage(TestPackage package)
{
    m_testConfig.m_testPackage = package;
}

bool SynTest::shouldRunTest()
{
    return m_testConfig.shouldRunTest(m_deviceType, _getNumOfDevices());
}

void SynTest::AcquireDevices()
{
    m_deviceIds.clear();
    synStatus status = synSuccess;

    for (unsigned deviceCntr = 0; deviceCntr < _getNumOfDevices(); deviceCntr++)
    {
        uint32_t    tmpDeviceId;
        std::string pciAddress = "";
        if (getenv("PCI_DOMAIN") != nullptr)
        {
            pciAddress = getenv("PCI_DOMAIN");
        }

        if ((pciAddress.compare("") == 0) || (synDeviceAcquire(&tmpDeviceId, pciAddress.c_str()) != synSuccess))
        {
            LOG_DEBUG(SYN_TEST, "Acquiring a specific Synapse device aborted, allocating any...");

            uint32_t requestedDeviceTypeCount = 0;
            uint32_t deviceCount[synDeviceTypeSize];
            status = synDeviceCount(deviceCount);
            ASSERT_EQ(status, synSuccess) << "Failed to get device count (" << status << ")";
            requestedDeviceTypeCount = deviceCount[m_deviceType];

            if (requestedDeviceTypeCount == 0)
            {
                for (auto deviceType : m_testConfig.m_supportedDeviceTypes)
                {
                    requestedDeviceTypeCount = deviceCount[deviceType];
                    if (requestedDeviceTypeCount > 0)
                    {
                        LOG_DEBUG(SYN_TEST,
                                  "Cannot find any device with the requsted type - {}, "
                                  "switching to another supported device - {}",
                                  m_deviceType,
                                  deviceType);
                        m_deviceType = deviceType;
                        break;
                    }
                }
                if (requestedDeviceTypeCount == 0)
                {
                    m_deviceType = synDeviceTypeInvalid;
                    for (uint32_t device = 0; device < synDeviceTypeSize; ++device)
                    {
                        if (deviceCount[device] > 0)
                        {
                            m_deviceType = synDeviceType(device);
                            break;
                        }
                    }
                    // Not supported device, return
                    return;
                }
            }

            if (!shouldRunTest())
            {
                return;
            }

            status = synDeviceAcquireByDeviceType(&tmpDeviceId, m_deviceType);
            ASSERT_EQ(status, synSuccess)
                << "Failed to acquire any device of type " << m_deviceType << " (Err:" << status << ")";
        }

        m_deviceIds.push_back(tmpDeviceId);
        ASSERT_LT(m_deviceIds.size(), m_testConfig.m_numOfTestDevices + 1);  // equal less than
    }
    if (!m_deviceIds.empty())
    {
        m_threadDeviceId = m_deviceIds[0];
    }
}

void SynTest::ReleaseDevices()
{
    for (unsigned deviceCntr = 0; deviceCntr < m_deviceIds.size(); deviceCntr++)
    {
        auto status = synDeviceRelease(m_deviceIds[deviceCntr]);
        ASSERT_EQ(status, synSuccess) << "Failed to release device " << deviceCntr << " idx "
                                      << m_deviceIds[deviceCntr];
    }
    m_deviceIds.clear();
}

void initGlobalConfManager()
{
    GlobalConfManager::instance().init(synSingleton::getConfigFilename());
}

void SynTest::afterSynInitialize() {}
void SynTest::beforeSynDestroy() {}

synStatus SynTest::SwitchPacketToDeviceVA(__u64& addr, __u32& bufferSize)
{
    uint64_t  addr2       = addr;
    uint32_t  bufferSize2 = bufferSize;
    synStatus status;

    status = SwitchBufferToDeviceVA(addr2, bufferSize2);
    if (status != synSuccess) return status;

    addr = addr2;

    return synSuccess;
}

synStatus SynTest::SwitchBufferToDeviceVA(uint64_t& addr, uint32_t& bufferSize)
{
    BufferAllocator* pBufferAllocator;
    synStatus        status;

    uint64_t         hostVA;
    std::string mappingDesc("Test's buffer");

    if (_SYN_SINGLETON_INTERNAL->m_deviceManager.mapBufferToDevice(bufferSize, (void*)addr, true, 0, mappingDesc) !=
        synSuccess)
    {
        return synFail;
    }

    eMappingStatus mappingStatus =
        _SYN_SINGLETON_INTERNAL->_getDeviceVirtualAddress(true, (void*)addr, bufferSize, &hostVA);
    if (mappingStatus != HATVA_MAPPING_STATUS_FOUND)
    {
        return synFail;
    }

    status = OSAL::getInstance().getBufferAllocator(hostVA, &pBufferAllocator);
    if (status != synSuccess) return status;

    addr = (uint64_t)pBufferAllocator->getDeviceVa();

    return synSuccess;
}

synStatus SynTest::GetDramMemory(uint32_t size, uint64_t& handle, uint64_t& addr, bool contiguous)
{
    synStatus status;

    status = AllocDramMemory(size, handle, contiguous);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "failed to allocate DRAM memory, size: {}", size);
        return status;
    }

    status = MapDramMemory(handle, addr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "failed to map DRAM memory, size: {}", size);
        return status;
    }

    return status;
}

synStatus SynTest::ReleaseDramMemory(uint64_t handle, uint64_t addr)
{
    synStatus status;

    status = UnmapDramMemory(addr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "failed to unmap DRAM memory, addr: {}", (void*)addr);
        return status;
    }

    status = FreeDramMemory(handle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_API, "failed to free DRAM memory, handle {}", handle);
        return status;
    }

    return status;
}

synStatus SynTest::AllocDramMemory(uint32_t size, uint64_t& handle, bool contiguous)
{
    hl_mem_args args;
    memset(&args, 0, sizeof(hl_mem_args));

    args.in.op             = HL_MEM_OP_ALLOC;
    args.in.alloc.mem_size = size;
    if (contiguous) args.in.flags = HL_MEM_CONTIGUOUS;

    int res = callIoctlMemory(&args);

    if (res != 0) return synFail;

    handle = args.out.handle;

    return synSuccess;
}

synStatus SynTest::FreeDramMemory(uint64_t handle)
{
    hl_mem_args args;
    memset(&args, 0, sizeof(hl_mem_args));

    args.in.op          = HL_MEM_OP_FREE;
    args.in.free.handle = handle;

    int res = callIoctlMemory(&args);

    return res != 0 ? synFail : synSuccess;
}

synStatus SynTest::MapHostMemory(uint64_t host_addr, uint32_t size, uint64_t& device_addr)
{
    hl_mem_args args;
    memset(&args, 0, sizeof(hl_mem_args));

    args.in.op                      = HL_MEM_OP_MAP;
    args.in.map_host.host_virt_addr = host_addr;
    args.in.map_host.mem_size       = size;
    args.in.flags                   = HL_MEM_USERPTR;

    int res = callIoctlMemory(&args);

    if (res != 0) return synFail;

    device_addr = args.out.device_virt_addr;

    return synSuccess;
}

synStatus SynTest::MapDramMemory(uint64_t handle, uint64_t& addr)
{
    hl_mem_args args;
    memset(&args, 0, sizeof(hl_mem_args));

    args.in.op                = HL_MEM_OP_MAP;
    args.in.map_device.handle = handle;

    int res = callIoctlMemory(&args);

    if (res != 0) return synFail;

    addr = args.out.device_virt_addr;

    return synSuccess;
}

synStatus SynTest::UnmapDramMemory(uint64_t addr)
{
    return UnmapMemory(addr, 0);
}

synStatus SynTest::UnmapHostMemory(uint64_t addr)
{
    return UnmapMemory(addr, HL_MEM_USERPTR);
}

synStatus SynTest::UnmapMemory(uint64_t addr, uint32_t flags)
{
    hl_mem_args args;
    memset(&args, 0, sizeof(hl_mem_args));

    args.in.op                     = HL_MEM_OP_UNMAP;
    args.in.unmap.device_virt_addr = addr;
    args.in.flags                  = flags;

    int res = callIoctlMemory(&args);

    return res != 0 ? synFail : synSuccess;
}

synTensor SynTest::createTrainingTensor(unsigned             dims,
                                        synDataType          dataType,
                                        const unsigned*      tensorSize,
                                        bool                 isPersist,
                                        const char*          name,
                                        const synGraphHandle graphHandle,
                                        synSectionHandle*    pGivenSectionHandle /*optional*/,
                                        uint64_t             offset)
{
    testGraphHandle::CreatedTensorInfo createdTensorAndSection = {nullptr, nullptr};

    createdTensorAndSection = createTrainingTensorAndSection(dims,
                                                            dataType,
                                                            tensorSize,
                                                            isPersist,
                                                            name,
                                                            graphHandle,
                                                            pGivenSectionHandle,
                                                            offset);

    std::unique_lock<std::mutex> lock(m_sectionsMutex);
    if (createdTensorAndSection.section != nullptr)
    {
        m_sections.push_back(createdTensorAndSection.section);
    }

    return createdTensorAndSection.tensor;
}

synStatus SynTest::callIoctlMemory(void* arg)
{
    int fd = OSAL::getInstance().getFd();

    union hl_mem_args* mem_args = (union hl_mem_args*)arg;
    int                ret      = 0;

    if (mem_args->in.op == HL_MEM_OP_ALLOC)
    {
        mem_args->out.handle = hlthunk_device_memory_alloc(fd,
                                                           mem_args->in.alloc.mem_size,
                                                           0,
                                                           mem_args->in.flags & HL_MEM_CONTIGUOUS,
                                                           mem_args->in.flags & HL_MEM_SHARED);
        if (!mem_args->out.handle) ret = -1;
    }
    else if (mem_args->in.op == HL_MEM_OP_FREE)
    {
        ret = hlthunk_device_memory_free(fd, mem_args->in.free.handle);
    }
    else if (mem_args->in.op == HL_MEM_OP_MAP)
    {
        if (mem_args->in.flags & HL_MEM_USERPTR)
        {
            mem_args->out.device_virt_addr = hlthunk_host_memory_map(fd,
                                                                     (void*)mem_args->in.map_host.host_virt_addr,
                                                                     mem_args->in.map_host.hint_addr,
                                                                     mem_args->in.map_host.mem_size);
            if (!mem_args->out.device_virt_addr) ret = -1;
        }
        else
        {
            mem_args->out.device_virt_addr =
                hlthunk_device_memory_map(fd, mem_args->in.map_device.handle, mem_args->in.map_device.hint_addr);
            if (!mem_args->out.device_virt_addr) ret = -1;
        }
    }
    else
    {
        ret = hlthunk_memory_unmap(fd, mem_args->in.unmap.device_virt_addr);
    }

    return ((ret >= 0) ? synSuccess : synFail);
}

// Very rarely the deserialize fails in CI, looks like on some file issue. This code is trying to
// open the file and log out to stdout the error it gets.
synStatus SynTest::retryDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName)
{
    {
        std::ifstream          input;
        std::ios_base::iostate exceptionMask = input.exceptions() | std::ios::failbit;
        input.exceptions(exceptionMask);

        try
        {
            if (!input.is_open())
            {
                input.open(recipeFileName, std::ios::binary);
            }

            input.unsetf(std::ios::skipws);

            // Calculate file's size:
            std::streampos fileSize;
            input.seekg(0, std::ios::end);
            fileSize = input.tellg();

            uint32_t crc32;
            input.seekg((-1) * sizeof(crc32), std::ios_base::end);
            input.read((char*)&crc32, sizeof(crc32));

            input.seekg(0, std::ios::beg);

            if (fileSize == 0)
            {
                std::cout << "param file" << recipeFileName << " is empty, with no crc32";
            }
            else
            {
                std::cout << "param file " << recipeFileName << " file size: " << fileSize << " crc32 " << crc32
                          << "\n";
            }
        }
        catch (const std::ifstream::failure& e)
        {
            std::cout << "Cannot Open/Read file: " << recipeFileName << " errno " << errno << ", " << strerror(errno)
                      << "\n";
        }
    }
    return synRecipeDeSerialize(pRecipeHandle, recipeFileName);
}

testGraphHandle::CreatedTensorInfo createTrainingTensorAndSection(  unsigned             dims,
                                                                    synDataType          dataType,
                                                                    const unsigned*      tensorSize,
                                                                    bool                 isPersist,
                                                                    const char*          name,
                                                                    const synGraphHandle graphHandle,
                                                                    synSectionHandle*    pGivenSectionHandle /*optional*/,
                                                                    uint64_t             offset)
{
    synStatus           status;
    synTensorDescriptor desc;

    // input
    desc.m_dataType = dataType;
    desc.m_dims     = dims;
    desc.m_name     = name;
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensorSize[dims - 1 - i];
    }

    synSectionHandle sectionHandle = pGivenSectionHandle ? *pGivenSectionHandle : nullptr;
    if (isPersist && !sectionHandle)
    {
        uint64_t sectionDescriptor = 0;
        synSectionCreate(&sectionHandle, sectionDescriptor, graphHandle);
    }

    synTensor tensor;
    status = synTensorCreate(&tensor, &desc, sectionHandle, offset);
    assert(status == synSuccess && "Create tensor failed!");
    UNUSED(status);

    if (pGivenSectionHandle != nullptr)
    {
        // return null in case we didn't create a new section
        sectionHandle = nullptr;
    }

    return {tensor, sectionHandle};
}

/*************************************************************/
void check_ok(synStatus status, const std::string& msg)
{
    if (status != synSuccess)
    {
        std::cout << "BAD\n";  // just to have an easy point to break on
    }
    ASSERT_EQ(synSuccess, status) << msg.c_str();
}

/********************* synStreamHandleW ****************************************/
testStreamHandle::testStreamHandle(const synDeviceId deviceId, std::string name) : m_name(name), m_shouldDestroy(true)
{
    synStatus status = synStreamCreateGeneric(&m_handle, deviceId, 0);
    check_ok(status, "Failed to create stream. Name:" + m_name);
}

testStreamHandle::testStreamHandle(synStreamHandle handle, std::string name)
: m_handle(handle), m_name(name), m_shouldDestroy(false)
{
}

testStreamHandle::~testStreamHandle()
{
    if (m_shouldDestroy)
    {
        synStatus status = synStreamDestroy(m_handle);
        check_ok(status, "Failed to destroy stream. Name:" + m_name);
    }
    m_handle = nullptr;
}

/******************* synEventHandleW ******************************************/
testEventHandle::testEventHandle(testEventHandle&& rhs) noexcept
{
    m_handle     = rhs.m_handle;
    rhs.m_handle = nullptr;
    m_name       = std::move(rhs.m_name);
}

testEventHandle::~testEventHandle()
{
    if (m_handle == nullptr) return;
    synStatus status = synEventDestroy(m_handle);
    check_ok(status, "Failed to destroy" + m_name);
    m_handle = nullptr;
}

void testEventHandle::record(testStreamHandle& stream)
{
    synStatus status = synEventRecord(m_handle, stream.get());
    check_ok(status, "Failed to record an event " + m_name + "on stream " + stream.getName());
}

void testEventHandle::wait(testStreamHandle& stream)
{
    synStatus status = synStreamWaitEvent(stream.get(), m_handle, 0);
    check_ok(status, "Failed to wait on event " + m_name + "on stream " + stream.getName());
}

void testEventHandle::init(std::string name, synDeviceId deviceId, const uint32_t flags)
{
    synStatus status = synEventCreate(&m_handle, deviceId, flags);
    check_ok(status, "Failed to create event " + m_name);
    m_name = name;
}

/*********************** synTensorW ******************************************/
testTensor::testTensor(testTensor&& rhs) noexcept
{
    m_tensor            = rhs.m_tensor;
    rhs.m_tensor        = nullptr;
    m_name              = std::move(rhs.m_name);
    m_shouldDestroy     = rhs.m_shouldDestroy;
    rhs.m_shouldDestroy = false;
}

testTensor::~testTensor()
{
    if (m_tensor == nullptr) return;
    if (m_shouldDestroy)
    {
        synStatus status = synTensorDestroy(m_tensor);
        check_ok(status, "Failed to destroy tensor " + m_name);
    }
    m_shouldDestroy = false;
    m_tensor = nullptr;
}

/********************* synGraphHandleW *******************************************/

testGraphHandle::testGraphHandle(std::string name, synDeviceType deviceType) : m_name(name)
{
    synStatus status = synGraphCreate(&m_handle, deviceType);
    check_ok(status, "Failed to create graph " + m_name);
};

testGraphHandle::~testGraphHandle()
{
    synStatus status = synGraphDestroy(m_handle);
    check_ok(status, "Failed to destroy graph " + m_name);
    m_handle = nullptr;
    if (m_recipeHandle)
    {
        synRecipeDestroy(m_recipeHandle);
    }

    for (auto section: m_sections)
    {
        synSectionDestroy(section);
    }
    m_sections.clear();
}

synTensor testGraphHandle::getTensorHandle(std::string name)
{
    for (auto& in : m_tensorVec)
    {
        if (in.getName() == name)
        {
            return in.getTensor();
        }
    }
    return nullptr;
}

synStatus testGraphHandle::compile(synDeviceId deviceId)
{
    uint64_t WorkspaceSize = 0;

    // synGraphCompile - compile the graph specified
    synStatus status = synGraphCompile(&m_recipeHandle, m_handle, (m_name + ".recipe").c_str(), 0);
    check_ok(status, "Failed to compile graph " + m_name);

    // synWorkspaceGetSize - Gets the size of the workspace which is required to execute a given recipe
    status = synWorkspaceGetSize(&WorkspaceSize, m_recipeHandle);
    check_ok(status, "Failed to get workspace size " + m_name);

    // synDeviceMalloc - Creates a memory allocation on a specific device
    status = synDeviceMalloc(deviceId, WorkspaceSize, 0, 0, &m_workspaceBuffer);
    check_ok(status, "Failed to allocate workspace buffer " + m_name);

    return status;
}

synStatus testGraphHandle::reallocateWorkspace(synDeviceId deviceId)
{
    uint64_t  WorkspaceSize = 0;
    synStatus status        = synWorkspaceGetSize(&WorkspaceSize, m_recipeHandle);
    check_ok(status, "Failed to get workspace size " + m_name);

    uint64_t ws;
    status = synDeviceMalloc(deviceId, WorkspaceSize, 0, 0, &ws);
    check_ok(status, "Failed to allocate workspace buffer " + m_name);

    status = synDeviceFree(deviceId, m_workspaceBuffer, 0);
    check_ok(status, "failed to release previous work space");
    m_workspaceBuffer = ws;
    return status;
}

synStatus testGraphHandle::launch(testStreamHandle&   stream,
                                  synLaunchTensorInfo tensors[],
                                  uint32_t            totalNumOfTensors,
                                  uint32_t            flags)
{
    synStatus status = synSuccess;
    if (SYN_FLAGS_TENSOR_NAME != flags)
    {
        prepareTensorInfo(m_recipeHandle, tensors, totalNumOfTensors);
    }
    status = synLaunch(stream.get(), tensors, totalNumOfTensors, m_workspaceBuffer, m_recipeHandle, flags);
    check_ok(status, "Failed to launch " + m_name);
    return status;
}

synStatus
testGraphHandle::launchNoCheck(testStreamHandle& stream, synLaunchTensorInfo tensors[], uint32_t totalNumOfTensors)
{
    prepareTensorInfo(m_recipeHandle, tensors, totalNumOfTensors);

    return synLaunch(stream.get(), tensors, totalNumOfTensors, m_workspaceBuffer, m_recipeHandle, 0);
}

synStatus testGraphHandle::launchNoCheckWithFlags(testStreamHandle&   stream,
                                                  synLaunchTensorInfo tensors[],
                                                  uint32_t            totalNumOfTensors,
                                                  uint32_t            flags)
{
    prepareTensorInfo(m_recipeHandle, tensors, totalNumOfTensors);

    synStatus status = synLaunch(stream.get(), tensors, totalNumOfTensors, m_workspaceBuffer, m_recipeHandle, flags);

    return status;
}

synStatus testGraphHandle::addTensor(unsigned          dims,
                                     synDataType       dataType,
                                     const unsigned*   tensorSize,
                                     bool              isPersist,
                                     std::string       name,
                                     synSectionHandle* pSectionHandle,
                                     uint64_t          offset)
{
    testGraphHandle::CreatedTensorInfo createdTensorAndSection = {nullptr, nullptr};

    createdTensorAndSection =
        createTrainingTensorAndSection(dims, dataType, tensorSize, isPersist, name.c_str(), m_handle, pSectionHandle, offset);

    if (createdTensorAndSection.section != nullptr)
    {
        m_sections.push_back(createdTensorAndSection.section);
    }

    if (createdTensorAndSection.tensor != nullptr)
    {
        // since tensor was created with graph context, the tensor will
        // be destroyed as part of synGraphDestroy.
        m_tensorVec.emplace_back(createdTensorAndSection.tensor, name, false);
        return synSuccess;
    }
    else
    {
        EXPECT_TRUE(false) << "Failed to create tensor " << name;
        return synFail;
    }
}

synStatus testGraphHandle::getWorkspaceInfo(synDeviceId deviceId, uint64_t& workspaceAddr, uint64_t& workspaceSize)
{
    if (m_workspaceBuffer == 0)
    {
        return synFail;
    }

    synStatus status = synWorkspaceGetSize(&workspaceSize, m_recipeHandle);
    check_ok(status, "Failed to get workspace size " + m_name);

    workspaceAddr = m_workspaceBuffer;

    return status;
}

long double SynTest::getRunningTime()
{
    size_t bufSize = 0;
    size_t entries = 0;
    synProfilerGetTrace(synTraceType::synTraceDevice, _getDeviceId(), synTraceFormatSize, nullptr, &bufSize, &entries);
    long double minValue = 0;
    long double maxValue = 0;
    if (bufSize == 0)
    {
        return 0;
    }
    std::vector<char> buf(bufSize, 0);
    synProfilerGetTrace(synTraceType::synTraceDevice,
                        _getDeviceId(),
                        synTraceFormatTEF,
                        buf.data(),
                        &bufSize,
                        &entries);
    auto events = reinterpret_cast<synTraceEvent*>(buf.data());
    for (int i = 0; i < entries; i++)
    {
        if (events[i].timestamp == 0) continue;
        if (events[i].type != 'E' && events[i].type != 'B') continue;
        if (events[i].arguments.operation == std::string()) continue;
        if (minValue == 0)
        {
            minValue = events[i].timestamp;
            maxValue = events[i].timestamp;
        }
        minValue = std::min(minValue, events[i].timestamp);
        maxValue = std::max(maxValue, events[i].timestamp);
    }
    return maxValue - minValue;
}

void SynTest::printProfileInformation()
{
    if (!GCFG_ENABLE_PROFILER.value() || !LOG_LEVEL_AT_LEAST_INFO(PERF))
    {
        return;
    }
    auto              runningTime = getRunningTime();
    std::stringstream data;
    data.precision(15);
    data << runningTime;
    LOG_INFO(PERF, "Time for {}: {}", ::testing::UnitTest::GetInstance()->current_test_info()->name(), data.str());
}

void prepareTensorInfo(synRecipeHandle recipe, synLaunchTensorInfo* tensorInfo, uint32_t totalNumOfTensors)
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