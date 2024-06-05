#include "habana_global_conf_runtime.h"

#include "global_conf_test_setter.h"

#include "recipe.h"

#include "synapse_api.h"
#include "syn_base_test.hpp"
#include "syn_singleton.hpp"

#include "test_device.hpp"
#include "test_dummy_recipe.hpp"

#include "runtime/common/common_types.hpp"

#include "runtime/common/device/device_common.hpp"

#include "runtime/common/queues/queue_interface.hpp"

#include "runtime/common/recipe/device_agnostic_recipe_processor.hpp"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

#include "runtime/common/streams/stream.hpp"

#include "runtime/scal/common/entities/scal_stream_base.hpp"

#include "runtime/scal/common/recipe_launcher/recipe_launcher.hpp"

#include "runtime/scal/common/stream_compute_scal.hpp"

#include <thread>

class SynScalLaunchDummyRecipe
: public SynBaseTest
, public ::testing::WithParamInterface<DummyRecipeType>
{
public:
    SynScalLaunchDummyRecipe() { setSupportedDevices({synDeviceGaudi2, synDeviceGaudi3}); }

protected:
    void runAllSteps(const TestDevice&                              rDevice,
                     const TestStream&                              rStream,
                     const TestDeviceBufferAlloc&                   rWorkspace,
                     std::vector<std::unique_ptr<TestDummyRecipe>>& recipes);
    void prepareToRun(bool memProtect, std::vector<std::unique_ptr<TestDummyRecipe>>& recipes, bool perfTest);
    void postRun(const TestStream& rStream);
    void createAndRunDummyAlmostAligned(const TestDevice&            rDevice,
                                        const TestStream&            rStream,
                                        const TestDeviceBufferAlloc& rWorkspace,
                                        bool                         random,
                                        bool                         protectMem,
                                        int                          numThreads);
    void runEagerRecipes(const TestDevice& rDevice, const TestStream& rStream, const TestDeviceBufferAlloc& rWorkspace);
    void
    hugeNonPatchable(const TestDevice& rDevice, const TestStream& rStream, const TestDeviceBufferAlloc& rWorkspace);
    void hugePatchable(
        bool                   processOnly,
        uint64_t               patchSize,
        TestDevice*            pDevice      = nullptr,
        TestStream*            pStream      = nullptr,
        TestDeviceBufferAlloc* pWorkspace   = nullptr,
        uint64_t               execSize     = 0x1000,
        uint64_t               dynamicSize  = 0x1000,
        uint64_t               prgDataSize  = 0x1000,
        uint64_t               ecbListsSize = 0x200);

    uint64_t ecbListsSize;
    void outOfMmm();

    static const uint64_t TENSOR_TO_ADDR_FACTOR = 0x10000;

    uint64_t getDynamicComputeEcbListBuffSize()
    {
        uint64_t dynamicComputeEcbListBuffSize = 0;
        if (m_deviceType == synDeviceGaudi2)
        {
            dynamicComputeEcbListBuffSize = g2fw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        }
        else if (m_deviceType == synDeviceGaudi3)
        {
            dynamicComputeEcbListBuffSize = g3fw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
        }
        else
        {
            HB_ASSERT(false, " Device-type not supported");
        }
        return dynamicComputeEcbListBuffSize;
    }

private:
    void runRecipes(const TestDevice*                              pDevice,
                    const TestStream*                              pStream,
                    const TestDeviceBufferAlloc*                   pWorkspace,
                    std::vector<std::unique_ptr<TestDummyRecipe>>& recipes,
                    bool                                           random,
                    bool                                           perfTest);
};

REGISTER_SUITE(SynScalLaunchDummyRecipe, ALL_TEST_PACKAGES);

INSTANTIATE_TEST_SUITE_P(,
                         SynScalLaunchDummyRecipe,
                         ::testing::Values(RECIPE_TYPE_NORMAL, RECIPE_TYPE_DSD, RECIPE_TYPE_DSD_AND_IH2D),
                         [](const ::testing::TestParamInfo<DummyRecipeType>& info) {
                             // Test name suffix - either DSD, DSD and IH2D or regular version
                             const DummyRecipeType recipeType = info.param;
                             std::string           name;
                             switch (recipeType)
                             {
                                 case RECIPE_TYPE_NORMAL:
                                     name = "regular_recipe";
                                     break;
                                 case RECIPE_TYPE_DSD:
                                     name = "dsd_recipe";
                                     break;
                                 case RECIPE_TYPE_DSD_AND_IH2D:
                                     name = "dsd_ih2d_recipe";
                                     break;
                                 default:
                                     name = "NULL";
                                     break;
                             }

                             return name;
                         });

// create recipes with random sizes, multiple of dcSize / 2
TEST_P(SynScalLaunchDummyRecipe, dummy_recipes_aligned_sizes)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);

    const uint64_t dynamicComputeEcbListBuffSize = getDynamicComputeEcbListBuffSize();
    uint64_t       dcSize                        = MappedMemMgr::getDcSize();
    const uint8_t  maxDc                         = 10;  // not too big, so it won't take too long

    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    // add a special case where all sizes are dcSize
    {
        uint64_t patchSize   = dcSize;
        uint64_t execSize    = dcSize;
        uint64_t dynamicSize = dcSize;
        uint64_t prgDataSize = dcSize;

        uint64_t ecbListsSize = 0x800;

        recipes.emplace_back(new TestDummyRecipe(GetParam(),
                                                 patchSize,
                                                 execSize,
                                                 dynamicSize,
                                                 prgDataSize,
                                                 ecbListsSize,
                                                 0,
                                                 m_deviceType));

        TestDummyRecipe* dummyRecipe = recipes.back().get();
        dummyRecipe->createValidEcbLists();
    }

    auto randomFunc = [&]() { return (1 + std::rand() % (maxDc * 2)) * dcSize / 2; };

    for (int i = 0; i < 20; i++)
    {
        uint64_t patchSize   = randomFunc();
        uint64_t execSize    = randomFunc();
        uint64_t dynamicSize = randomFunc();
        uint64_t prgDataSize = randomFunc();

        uint64_t ecbListsSize = (1 + std::rand() % 0x1FF) * dynamicComputeEcbListBuffSize;  // aligned to 0x100

        recipes.emplace_back(new TestDummyRecipe(GetParam(),
                                                 patchSize,
                                                 execSize,
                                                 dynamicSize,
                                                 prgDataSize,
                                                 ecbListsSize,
                                                 0,
                                                 m_deviceType));

        TestDummyRecipe* dummyRecipe = recipes.back().get();
        dummyRecipe->createValidEcbLists();
    }
    runAllSteps(device, stream, workspace, recipes);
}

// create recipes with random sizes, almost aligned to  multiple of dcSize / 2
// This function has 3 options:
// random:     when running, pick a random recipe from the recipe-pool
// protectMem: run with the debug option that protects the mapped-memory-manager mapped memory
// numThreads: how many thread run recipes (same stream)
void SynScalLaunchDummyRecipe::createAndRunDummyAlmostAligned(const TestDevice&            rDevice,
                                                              const TestStream&            rStream,
                                                              const TestDeviceBufferAlloc& rWorkspace,
                                                              bool                         random,
                                                              bool                         protectMem,
                                                              int                          numThreads)
{
    const uint64_t dynamicComputeEcbListBuffSize = getDynamicComputeEcbListBuffSize();
    uint64_t       dcSize                        = MappedMemMgr::getDcSize();
    const uint8_t  maxDc                         = 10;  // not too big, so it won't take too long

    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    auto randomFunc = [&]() { return (1 + std::rand() % (maxDc * 2)) * dcSize / 2 - 0x20 + (std::rand() % 9) * 8; };

    for (int i = 0; i < 24; i++)
    {
        uint64_t patchSize   = randomFunc();
        uint64_t execSize    = randomFunc();
        uint64_t dynamicSize = randomFunc();
        uint64_t prgDataSize = randomFunc();

        uint64_t ecbListsSize = (1 + std::rand() % 0x1FF) * dynamicComputeEcbListBuffSize;  // aligned to 0x100

        recipes.emplace_back(new TestDummyRecipe(GetParam(),
                                                 patchSize,
                                                 execSize,
                                                 dynamicSize,
                                                 prgDataSize,
                                                 ecbListsSize,
                                                 0,
                                                 m_deviceType));

        TestDummyRecipe* dummyRecipe = recipes.back().get();
        dummyRecipe->createValidEcbLists();

        // patching does not support protection becuase it caches the GCFG value. In this test we set the protection
        // during the run but the patch code already cached the original value and doesn't un-protect before patching
        // and throws a seg-fault
        dummyRecipe->getRecipe()->patch_points_nr = 0;
    }

    prepareToRun(protectMem, recipes, false);

    std::vector<std::thread> threadVector;
    for (unsigned thread = 0; thread < numThreads; thread++)
    {
        std::thread th(&SynScalLaunchDummyRecipe::runRecipes,
                       this,
                       &rDevice,
                       &rStream,
                       &rWorkspace,
                       std::ref(recipes),
                       random,
                       false);
        threadVector.push_back(std::move(th));
    }

    for (auto& th : threadVector)
    {
        th.join();
    }

    postRun(rStream);
}

// ran recipes with sizes almost aligned to dcSize / 2. Run all in sequence, one thread, no memory protection
TEST_P(SynScalLaunchDummyRecipe, dummy_recipes_almost_aligned)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);
    createAndRunDummyAlmostAligned(device, stream, workspace, false, false, 1);
}

// ran recipes with sizes almost aligned to dcSize / 2. Run all in sequence, 4 threads, no memory protection
TEST_P(SynScalLaunchDummyRecipe, dummy_recipes_almost_aligned_multi_thread)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);
    createAndRunDummyAlmostAligned(device, stream, workspace, true, false, 4);
}

// ran recipes with sizes almost aligned to dcSize / 2. Run all in sequence, 4 threads, with memory protection
TEST_P(SynScalLaunchDummyRecipe, dummy_recipes_almost_aligned_multi_thread_protect_mem)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);
    createAndRunDummyAlmostAligned(device, stream, workspace, true, true, 4);
}

// create very small recipes
TEST_P(SynScalLaunchDummyRecipe, dummy_recipes_small)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);

    const uint64_t dynamicComputeEcbListBuffSize = getDynamicComputeEcbListBuffSize();

    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    auto randomFunc = []() { return (1 + std::rand() % 0x80) * 8; };

    for (int i = 0; i < 20; i++)
    {
        uint64_t patchSize   = randomFunc();
        uint64_t execSize    = randomFunc();
        uint64_t dynamicSize = randomFunc();
        uint64_t prgDataSize = randomFunc();

        uint64_t ecbListsSize = (1 + std::rand() % 0x1FF) * dynamicComputeEcbListBuffSize;  // Aligned to 0x100

        recipes.emplace_back(new TestDummyRecipe(GetParam(),
                                                 patchSize,
                                                 execSize,
                                                 dynamicSize,
                                                 prgDataSize,
                                                 ecbListsSize,
                                                 0,
                                                 m_deviceType));

        TestDummyRecipe* dummyRecipe = recipes.back().get();
        dummyRecipe->createValidEcbLists();
    }
    runAllSteps(device, stream, workspace, recipes);
}

// run commands needed before any synLaunch (acquire device, create stream, etc.)
void SynScalLaunchDummyRecipe::prepareToRun(bool                                           memProtect,
                                            std::vector<std::unique_ptr<TestDummyRecipe>>& recipes,
                                            bool                                           perfTest)
{
    for (auto& dummyRecipe : recipes)
    {
        InternalRecipeHandle* internalRecipeHandle = dummyRecipe->getInternalRecipeHandle();

        GCFG_HBM_GLOBAL_MEM_SIZE_MEGAS.setValue(256);
        GCFG_MAX_CONST_TENSOR_SIZE_BYTES.setValue(104851000);
        synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle->basicRecipeHandle,
                                                                  internalRecipeHandle->deviceAgnosticRecipeHandle);
        ASSERT_EQ(status, synSuccess);
    }

    GCFG_CHECK_SECTION_OVERLAP.setValue(false);
    if (perfTest)
    {
        ;  // set nothing
    }
    else if (memProtect)
    {
        GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD | PROTECT_MAPPED_MEM);
    }
    else
    {
        GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD);
    }
}

// run commands needed after all synLaunchs are done (release resource)
void SynScalLaunchDummyRecipe::postRun(const TestStream& rStream)
{
    LOG_INFO(SYN_API, "-----sync device");
    rStream.synchronize();
}

void SynScalLaunchDummyRecipe::runAllSteps(const TestDevice&                              rDevice,
                                           const TestStream&                              rStream,
                                           const TestDeviceBufferAlloc&                   rWorkspace,
                                           std::vector<std::unique_ptr<TestDummyRecipe>>& recipes)
{
    prepareToRun(false, recipes, false);
    runRecipes(&rDevice, &rStream, &rWorkspace, recipes, false, false);
    postRun(rStream);
}

void SynScalLaunchDummyRecipe::runRecipes(const TestDevice*                              pDevice,
                                          const TestStream*                              pStream,
                                          const TestDeviceBufferAlloc*                   pWorkspace,
                                          std::vector<std::unique_ptr<TestDummyRecipe>>& recipes,
                                          bool                                           random,
                                          bool                                           perfTest)
{
    int numToRun = recipes.size();

    if (random) numToRun *= 1;

    uint64_t orgVal = GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.value();

    for (int i = 0; i < numToRun; i++)
    {
        int idx = random ? std::rand() % recipes.size() : i;

        TestDummyRecipe* dummyRecipe = recipes[idx].get();

        recipe_t*             recipe               = dummyRecipe->getRecipe();
        InternalRecipeHandle* internalRecipeHandle = dummyRecipe->getInternalRecipeHandle();

        std::vector<synLaunchTensorInfo> tensors;

        uint64_t numTensors = recipe->persist_tensors_nr;
        tensors.resize(numTensors);
        for (uint64_t t = 0; t < numTensors; t++)
        {
            tensors[t].tensorName = recipe->tensors[t].name;
            tensors[t].pTensorAddress =
                (t + 1) * TENSOR_TO_ADDR_FACTOR + pWorkspace->getBuffer();  // just to get an in hbm addr
        }

        if (perfTest)
        {
            ;  // nothing to se
        }
        else if (!random || (idx % 8 == 0))
        {
            GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(orgVal | COMPARE_RECIPE_ON_DEVICE_AFTER_DOWNLOAD);
        }
        else
        {
            GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(orgVal);
        }

        ASSERT_EQ(synLaunch(*pStream,
                            tensors.data(),
                            tensors.size(),
                            pWorkspace->getBuffer(),
                            internalRecipeHandle,
                            SYN_FLAGS_TENSOR_NAME),
                  synSuccess)
            << "Failed to synLaunch";
    }

    GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE.setValue(orgVal);
}

TEST_F_SYN(SynScalLaunchDummyRecipe, dsd_dummy)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);

    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    uint64_t dcSize      = MappedMemMgr::getDcSize();
    uint64_t patchSize   = dcSize;
    uint64_t execSize    = dcSize;
    uint64_t dynamicSize = dcSize;
    uint64_t prgDataSize = dcSize;

    uint64_t ecbListsSize = 0x800;
    recipes.emplace_back(new TestDummyRecipe(RECIPE_TYPE_DSD,
                                             patchSize,
                                             execSize,
                                             dynamicSize,
                                             prgDataSize,
                                             ecbListsSize,
                                             0,
                                             m_deviceType));

    TestDummyRecipe* dummyRecipe = recipes.back().get();
    dummyRecipe->createValidEcbLists();

    recipes.emplace_back(new TestDummyRecipe(RECIPE_TYPE_DSD_AND_IH2D,
                                             patchSize,
                                             execSize,
                                             dynamicSize,
                                             prgDataSize,
                                             ecbListsSize,
                                             0,
                                             m_deviceType));
    dummyRecipe = recipes.back().get();
    dummyRecipe->createValidEcbLists();

    runAllSteps(device, stream, workspace, recipes);
}

// This test is running a lot of different small recipes to try and simulate an eager scenario
// It helps in development, no need to run it in CI. You should run it on a real device
TEST_F_SYN(SynScalLaunchDummyRecipe, DISABLED_eager_perf)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);
    runEagerRecipes(device, stream, workspace);
}

void SynScalLaunchDummyRecipe::runEagerRecipes(const TestDevice&            rDevice,
                                               const TestStream&            rStream,
                                               const TestDeviceBufferAlloc& rWorkspace)
{
    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    for (int i = 0; i < 10000; i++)
    {
        // Numbers are taken from an exaple I got
        uint64_t patchSize   = 0x28;
        uint64_t execSize    = 0x1f8;
        uint64_t dynamicSize = 0x50;
        uint64_t prgDataSize = 0x900;

        uint64_t ecbListsSize = 0x100;  // Aligned to 0x100

        recipes.emplace_back(new TestDummyRecipe(RECIPE_TYPE_NORMAL,
                                                 patchSize,
                                                 execSize,
                                                 dynamicSize,
                                                 prgDataSize,
                                                 ecbListsSize,
                                                 0,
                                                 m_deviceType));

        TestDummyRecipe* dummyRecipe = recipes.back().get();
        dummyRecipe->createValidEcbLists();
    }

    prepareToRun(false, recipes, true);
    runRecipes(&rDevice, &rStream, &rWorkspace, recipes, false, true);
    postRun(rStream);
}

void SynScalLaunchDummyRecipe::hugeNonPatchable(const TestDevice&            rDevice,
                                                const TestStream&            rStream,
                                                const TestDeviceBufferAlloc& rWorkspace)
{
    uint64_t dcSize = MappedMemMgr::getDcSize();

    std::vector<std::unique_ptr<TestDummyRecipe>>
        recipes;  // only one is used, but easier in a vector because of the infrastructure

    uint64_t patchSize = 0x1000;
    uint64_t execSize  = dcSize * 0x76;  // make sure we don't limit the size (we limit the patchable size to 0x100
                                         // DC because of the patching code
    uint64_t dynamicSize = 0x1000;
    uint64_t prgDataSize = 0x1000;

    uint64_t ecbListsSize = 0x200;  // aligned to 0x100

    recipes.emplace_back(
        new TestDummyRecipe(GetParam(), patchSize, execSize, dynamicSize, prgDataSize, ecbListsSize, 0, m_deviceType));

    TestDummyRecipe* dummyRecipe = recipes.back().get();
    dummyRecipe->createValidEcbLists();

    prepareToRun(false, recipes, true);
    runRecipes(&rDevice, &rStream, &rWorkspace, recipes, false, true);
    postRun(rStream);
}

// processOnly - if falls, run the recipe
void SynScalLaunchDummyRecipe::hugePatchable(
    bool                   processOnly,
    uint64_t               patchSize,
    TestDevice*            pDevice,
    TestStream*            pStream,
    TestDeviceBufferAlloc* pWorkspace,
    uint64_t               execSize,
    uint64_t               dynamicSize,
    uint64_t               prgDataSize,
    uint64_t               ecbListsSize)
{
    // only one is used, but easier in a vector because of the infrastructure
    std::vector<std::unique_ptr<TestDummyRecipe>> recipes;

    recipes.emplace_back(
        new TestDummyRecipe(GetParam(), patchSize, execSize, dynamicSize, prgDataSize, ecbListsSize, 0, m_deviceType));

    TestDummyRecipe* dummyRecipe = recipes.back().get();
    dummyRecipe->createValidEcbLists();

    if (processOnly)
    {
        InternalRecipeHandle* internalRecipeHandle = dummyRecipe->getInternalRecipeHandle();

        synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle->basicRecipeHandle,
                                                                internalRecipeHandle->deviceAgnosticRecipeHandle);
        ASSERT_EQ(status, synFail) << "Should fail, patchable too big";
    }
    else
    {
        HB_ASSERT_PTR(pDevice);
        HB_ASSERT_PTR(pStream);
        HB_ASSERT_PTR(pWorkspace);

        prepareToRun(false, recipes, true);
        runRecipes(pDevice, pStream, pWorkspace, recipes, false, true);
        postRun(*pStream);

        // Get Synapse-Stream
        Stream*       synapseStream {};
        DeviceCommon* deviceCommon = (DeviceCommon*)(_SYN_SINGLETON_INTERNAL->getDevice().get());
        auto          streamSptr   = deviceCommon->loadAndValidateStream(*pStream, __FUNCTION__);
        ASSERT_NE(streamSptr, nullptr) << "Failed to load Stream";
        synapseStream = streamSptr.get();

        // Check CCB consistency
        QueueInterface* pComputeQueueInterface;
        synapseStream->testGetQueueInterface(QUEUE_TYPE_COMPUTE, pComputeQueueInterface);
        ScalStreamCopyInterface* txSynapseScalStreamInterface =
            reinterpret_cast<QueueComputeScal*>(pComputeQueueInterface)->m_computeResources.m_pTxCommandsStream;
        ScalStreamBase* txSynapseScalStreamBase = reinterpret_cast<ScalStreamBase*>(txSynapseScalStreamInterface);
        StreamCyclicBufferBase* txSynapseCcb = txSynapseScalStreamBase->getStreamCyclicBuffer();
        ASSERT_EQ(txSynapseCcb->testOnlyCheckCcbConsistency(), true) << "TX-Synapse CCB long-SO is not consistent";
    }
}

void SynScalLaunchDummyRecipe::outOfMmm()
{
    uint64_t dcSize = MappedMemMgr::getDcSize();
    int      numDc  = MappedMemMgr::getNumDc();

    uint64_t numPatch = std::min(numDc / 2, 0xFF);  // must be less than 0x100

    uint64_t patchSize = dcSize * numPatch;  // make sure we limit the size (we limit the patchable size to 0x100
    uint64_t execSize  = dcSize * (numDc - numPatch);  // With patchable we need exactly the number of DC. Adding other
                                                       // parts (ecb list, dynamic, etc.) we should be out of memory
    // DC because of the patching code
    uint64_t dynamicSize = 0x1000;
    uint64_t prgDataSize = 0x1000;

    uint64_t ecbListsSize = 0x200;  // aligned to 0x100

    TestDummyRecipe
        dummyRecipe(GetParam(), patchSize, execSize, dynamicSize, prgDataSize, ecbListsSize, 0, m_deviceType);

    InternalRecipeHandle* internalRecipeHandle = dummyRecipe.getInternalRecipeHandle();

    synStatus status = DeviceAgnosticRecipeProcessor::process(internalRecipeHandle->basicRecipeHandle,
                                                              internalRecipeHandle->deviceAgnosticRecipeHandle);
    ASSERT_EQ(status, synFail) << "Should fail, recipe too big to fit in mapped-memory-manager";
}

TEST_P(SynScalLaunchDummyRecipe, hugeNonPatchable)
{
    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);
    hugeNonPatchable(device, stream, workspace);
}

TEST_P(SynScalLaunchDummyRecipe, hugePatchable)
{
    uint64_t dcSize = MappedMemMgr::getDcSize();

    uint64_t patchSize = dcSize * 0x101;  // make sure we limit the size (we limit the patchable size to 0x100
    hugePatchable(true /* processOnly */, patchSize);
}

TEST_P(SynScalLaunchDummyRecipe, outOfMmm)
{
    outOfMmm();
}

TEST_P(SynScalLaunchDummyRecipe, testTxSynapsePdmaStreamCcbOccupancy)
{
    GlobalConfTestSetter expFlags("ENABLE_EXPERIMENTAL_FLAGS", "true");

    // Define very small CCB-chunk size, so a submission will result multiple chunks usage
    GlobalConfTestSetter ccbBufferSize("SET_HOST_CYCLIC_BUFFER_SIZE", "64"); // 1KB
    GlobalConfTestSetter ccbChunksAmount("SET_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT", "128");

    uint64_t dcSize = MappedMemMgr::getDcSize();

    TestDevice            device(m_deviceType);
    TestStream            stream    = device.createStream();
    TestDeviceBufferAlloc workspace = device.allocateDeviceBuffer(0x1000, 0);

    bool     processOnly  = false;
    uint64_t patchSize    = dcSize * 0x90;
    uint64_t execSize     = 0x1000000;
    uint64_t dynamicSize  = 0x100000;
    uint64_t prgDataSize  = 0x100000;
    uint64_t ecbListsSize = 0x200000;  // aligned to 0x100

    hugePatchable(processOnly, patchSize, &device, &stream, &workspace,
                  execSize, dynamicSize, prgDataSize, ecbListsSize);
}