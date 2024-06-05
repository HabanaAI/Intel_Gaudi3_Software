#include "gaudi_mem_test.hpp"
#include "test_device.hpp"
#include "test_launcher.hpp"
#include "test_recipe_relu_conv.hpp"
#include <locale>

REGISTER_SUITE(SynFlowMemTests, synTestPackage::ASIC);

void SynFlowMemTests::allocateWorkspace(unsigned&   err,
                                        uint64_t    freemem,
                                        uint64_t    topologyWorkspaceSize,
                                        synDeviceId deviceId)
{
    synStatus status = synSuccess;
    err              = 0;

    uint64_t topologyWorkspaceBuffer = 0;

    status = synDeviceMalloc(deviceId, topologyWorkspaceSize, 0, 0, &topologyWorkspaceBuffer);
    if (freemem < topologyWorkspaceSize)
    {
        if (status != synSuccess)
        {
            err = memTestErrorCodeDeviceMallocFail;
            std::cout << "As expected, allocation of workspace size=" << topologyWorkspaceSize << " fail when only "
                      << freemem << " left." << std::endl;
            return;
        }
        else
        {
            ASSERT_EQ(status, synFail) << "Allocate workspace buffer should have failed here";
        }
    }
    else
    {
        ASSERT_EQ(status, synSuccess) << "Failed to allocate workspace buffer";
    }
    m_workspaceAddeVec.push_back(topologyWorkspaceBuffer);
}

void SynFlowMemTests::_checkStatus(unsigned& err, synStatus status)
{
    err = 0;

    if (status != synSuccess)
    {
        if (status == synBusy)
        {
            err = memTestErrorCodeLaunchFail;
            return;
        }
        err = memTestErrorCodeGeneral;
    }
    ASSERT_EQ(status, synSuccess) << "Failed to enqueue";
}
/**************************************************************************************************/
//#define LOGGER_DEMO
void SynFlowMemTests::mem_test_internal(int passOnErr)
{
    synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
    synConfigurationSet("ENABLE_PERSISTENT_OUTPUT_REUSE", "false");

    synStatus  status         = synSuccess;
    uint64_t   freeMemAtStart = 0, total = 0, free = 0;
    TestDevice device(m_deviceType);

    {
        // allocate 1 time tensors
        TestRecipeReluConv recipe(m_deviceType);
        recipe.generateRecipe();

        TestStream   stream = device.createStream();
        TestLauncher launcher(device);

        std::cout.imbue(std::locale(""));  // to print commas between thousands

#ifdef LOGGER_DEMO
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::SYN_API, 1);
        auto consoleHolder = hl_logger::addConsole(synapse::LogManager::LogType::SYN_API);
#endif

        device.getDeviceMemoryInfo(freeMemAtStart, total);
        std::cout << "memory at start:  Total=" << total << "  Free=" << freeMemAtStart << std::endl;

        unsigned err;

        // allocates device memory for the Workspace - once,
        // and on other loops set the new allocated WS
        unsigned           num_alloc          = 1;
        RecipeLaunchParams recipeLaunchParams = launcher.createRecipeLaunchParams(recipe, {TensorInitOp::ALL_ZERO, 0});

        TestLauncher::download(stream, recipe, recipeLaunchParams);

        m_workspaceAddeVec.push_back(recipeLaunchParams.getWorkspace());

        //
        //  find how many (allocate workspace + synLaunch( )) can we do
        //

        device.getDeviceMemoryInfo(free, total);
        std::cout << "memory before launch loop: " << free << " tensors memory=" << freeMemAtStart - free << std::endl;

        unsigned loop          = 0;
        uint64_t workspaceSize = recipe.getWorkspaceSize();

        while (true)
        {
            auto launchTensorsInfo = recipeLaunchParams.getSynLaunchTensorInfoVec().data();
            status                 = synLaunchExt(stream.operator synStreamHandle(),
                                  launchTensorsInfo,
                                  recipe.getTensorInfoVecSize(),  // numberTensors
                                  m_workspaceAddeVec[loop],
                                  recipe.getRecipe(),
                                  0);  // flags
            _checkStatus(err, status);
            if (err) break;
            device.getDeviceMemoryInfo(free, total);
            // allocate for next loop
            allocateWorkspace(err, free, workspaceSize, device.getDeviceId());
            if (err) break;
            num_alloc++;
            loop++;
        }
        ASSERT_EQ(err, passOnErr) << "Received wrong error code - failed on wrong method";
        unsigned maxN = loop + 1;
        std::cout << "maximum loops " << maxN << std::endl;

        stream.synchronize();  // wait for ALL works to be done
        ASSERT_EQ(status, synSuccess) << "synStreamSynchronize fails waiting on compute stream";

        // cleanup - delete all workspaces, first WS is freed by test Infra
        for (unsigned int it = 1; it < num_alloc; it++)
        {
            status = synDeviceFree(device.getDeviceId(), m_workspaceAddeVec[it], 0);
            ASSERT_EQ(status, synSuccess) << "Failed to Deallocate Device memory for workspace (loop=" << it << ")";
        }
    }  // end of test body, recipe should be freed here

    device.getDeviceMemoryInfo(free, total);
    std::cout << "memory at end of test: " << free << " diff(total-free)=" << freeMemAtStart - free << std::endl;
    ASSERT_EQ(freeMemAtStart, free) << "not all memory was released at the end of the test (total-free)";

#ifdef LOGGER_DEMO
    synapse::LogManager::instance().disableConsole(synapse::LogManager::LogType::SYN_API,
                                                   console);  // turn console logging off
#endif
}
/**************************************************************************************************/
// following 2 test cases are only relevant for simulator - we don't want them to run on CI
TEST_F_SYN(SynFlowMemTests, DISABLED_mem_test)
{
    // needed for simulator in Ci, since it can be VERY slow
    GCFG_DATACHUNK_LOOP_NUM_RETRIES.setValue(10000);
    mem_test_internal(memTestErrorCodeDeviceMallocFail);
}

TEST_F_SYN(SynFlowMemTests, DISABLED_mem_test_busy)
{
    // need to fail on launch
    GCFG_DATACHUNK_LOOP_NUM_RETRIES.setValue(500);
    mem_test_internal(memTestErrorCodeLaunchFail);
}
/**************************************************************************************************/
TEST_F_SYN(SynFlowMemTests, mem_test)
{
    mem_test_internal(memTestErrorCodeDeviceMallocFail);
}