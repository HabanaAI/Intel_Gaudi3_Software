#include "graph_compiler/sif/shape_inference_functions.h"
#include "infra/timer.h"
#include "log_manager.h"
#include "recipe.h"
#include "smf/shape_func_registry.h"
#include "tpc_kernel_lib_interface.h"
#include "utils.h"

#include <gtest/gtest.h>


// This test is used to measure the time it takes to run SIF functions (you need a test case for each function)
class sifTime : public ::testing::Test
{
public:
    int RUNS = 1000;

    struct SifGivenData
    {
        unsigned                     numIn;
        std::vector<TensorShapeInfo> in;
        unsigned                     numParams;
        std::vector<uint8_t>         params;
        unsigned                     numOut;
    };

    tpc_lib_api::GlueCodeReturn runSif(sif_t sif, SifGivenData sifGivenData);

    struct TestCase
    {
        sif_t         sif;
        std::string   description;
        SifGivenData  sifGivenData;
    };

    struct TestCaseTpc
    {
        sm_function_id_t sif_id;
        std::string      description;
        SifGivenData     sifGivenData;
    };

    std::vector<TestCase> testCases;

    void addCase(TestCase testCase);
    void addCase(TestCaseTpc testCase);
    void addCases();
    void addCasesTpc();
};

// This function builds the needed params for the sif function and runs it
tpc_lib_api::GlueCodeReturn sifTime::runSif(sif_t sif, SifGivenData sifGivenData)
{
    auto numIn     = sifGivenData.numIn;
    auto numOut    = sifGivenData.numOut;
    auto numParams = sifGivenData.numParams;

    TensorShapeInfo out[numOut];

    TensorShapeInfo* pIn[numIn];
    TensorShapeInfo* pOut[numOut];

    for (int i = 0; i < numIn; i++)  pIn[ i] = &sifGivenData.in[i];
    for (int i = 0; i < numOut; i++) pOut[i] = &out[i];

    SifParams sifParams;
    sifParams.inputTensors    = pIn;
    sifParams.inputTensorsNr  = numIn;
    sifParams.outputTensorsNr = numOut;
    sifParams.nodeParams.nodeParamsSize = numParams;
    sifParams.nodeParams.nodeParams     = sifGivenData.params.data();

    uint64_t invalidMaskSize = numOut != 0 ? 1 : div_round_up(numOut, BITS_IN_UNSIGNED);
    unsigned invalidMask[invalidMaskSize];
    memset(invalidMask, 0, sizeof(invalidMask));

    // Output params
    SifOutputs sifOutputs;
    sifOutputs.outputTensors = pOut;
    sifOutputs.invalidMask   = invalidMask;

    tpc_lib_api::GlueCodeReturn sifRes = sif(tpc_lib_api::DEVICE_ID_MAX, &sifParams, &sifOutputs);

    return sifRes;
}

// This function adds a test case (GC sif)
void sifTime::addCase(TestCase testCase)
{
    testCases.push_back(testCase);
}

// This function adds a test case (TPC sif).
void sifTime::addCase(TestCaseTpc testCase)
{
    TestCase temp;

    auto sfr = ShapeFuncRegistry::instance();
    auto sif = sfr.getSIF(testCase.sif_id);
    if( sif == nullptr)
    {
        printf("Failed to find id %lX in ShapeFuncRegistry\n", TO64(testCase.sif_id.sm_func_index));
        return;
    }

    temp.sif          = sif;
    temp.description  = testCase.description;
    temp.sifGivenData = testCase.sifGivenData;

    testCases.push_back(temp);
}

// This function runs the sif time test
// It first prints all the functions in the ShapeFuncRegistry with an indication if the function has a test or not
// Then it runs all the test cases and shows the results
TEST_F(sifTime, DISABLED_sifTimeTest)
{
    initShapeFuncRegistry();

    addCases();
    addCasesTpc();

    for (auto testCase : testCases)
    {
        printf("Have test case for %16lX\n", TO64(testCase.sif));
    }

    auto       sfr     = ShapeFuncRegistry::instance();
    const auto sifBank = sfr.getAllSifTestingOnly();

    // Build unique set of pointers
    // Print if has test or not
    for (auto entry : sifBank)
    {
        bool hasTest = false;
        sif_t sifFunc = entry.second.func;
        for (auto& testCase : testCases)
        {
            if (testCase.sif == sifFunc)
            {
                hasTest = true;
                break;
            }
        }
        printf("SIF %lX id 0x%lX ", TO64(sifFunc), entry.first);
        printf("%s", hasTest ? "has     " : "***NO***");
        printf(" test.\n");
    }

    printf("\n");

    // Run each test case RUNS time and measure
    auto numCases = testCases.size();
    std::vector<uint64_t> measure(numCases, 0);
    std::vector<bool>     runOK(numCases, true);
    for (int run = 0; run < RUNS; run++)
    {
        for (size_t i = 0; i < numCases; i++)
        {
            if (runOK[i])
            {
                auto &testCase = testCases[i];
                auto start = TimeTools::timeNow();
                auto res = runSif(testCase.sif, testCase.sifGivenData);
                measure[i] += TimeTools::timeFromNs(start);
                if (res != tpc_lib_api::GLUE_SUCCESS)
                {
                    printf("SIF %lX failed in run %d with res %d\n", TO64(testCase.sif), run, res);
                    runOK[i] = false;
                }
            }
        }
    }

    // Output measure results
    printf("\nTime measure for %d runs\n================================\n", RUNS);
    for (size_t i = 0; i < numCases; i++)
    {
        auto &testCase = testCases[i];
        if(runOK[i] == true)
        {
            printf("ran %-40s average (ns): %ld\n", testCase.description.c_str(), measure[i] / RUNS);
        }
        else
        {
            printf("-----%-40s failed\n", testCase.description.c_str());
        }
    }

}

// This function builds some test cases (GC SIFs) and add them to the testCases vector
void sifTime::addCases()
{
    {
        TestCase testCase {};
        testCase.sif                    = dmaMemcpyShapeInferenceFunction;
        testCase.description            = "dmaMemcpyShapeInferenceFunction";
        testCase.sifGivenData.numIn     = 1;
        testCase.sifGivenData.in        = { {{3, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_F32}, 0, {}} };
        testCase.sifGivenData.numOut    = 1;
        testCase.sifGivenData.numParams = 0;
        testCase.sifGivenData.params    = {};
        addCase(testCase);
    }

    {
        TestCase testCase {};
        testCase.sif                    = concatenateShapeInferenceFunction;
        testCase.description            = "concatenateShapeInferenceFunction1";
        testCase.sifGivenData.numIn     = 1;
        testCase.sifGivenData.in        = { {{3, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_F32}, 0, {}} };
        testCase.sifGivenData.numOut    = 1;
        testCase.sifGivenData.numParams = 4;
        testCase.sifGivenData.params    = {2};
        addCase(testCase);
    }
    {
        TestCase testCase {};
        testCase.sif                    = concatenateShapeInferenceFunction;
        testCase.description            = "concatenateShapeInferenceFunction2";
        testCase.sifGivenData.numIn     = 2;
        testCase.sifGivenData.in        = { {{3, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_F32}, 0, {}},
                                            {{3, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_F32}, 0, {}} };
        testCase.sifGivenData.numOut    = 1;
        testCase.sifGivenData.numParams = 4;
        testCase.sifGivenData.params    = {2};
        addCase(testCase);
    }
}


// This function builds some test cases (TPC SIFs) and add them to the testCases vector
void sifTime::addCasesTpc()
{
    {
        TestCaseTpc testCase{};
        testCase.sif_id.sm_func_index = 0x02000006;
        testCase.description = "batch_norm_stage2_dynamic_add_relu_fwd_bf16";
        testCase.sifGivenData.numIn = 3;
        testCase.sifGivenData.in = { {{3, { 10, 10, 10, 10, 10 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_BF16}, 0, {}},
                                     {{3, { 10, 10, 10, 10, 10 }, { 0, 0, 0, 0, 0 }, tpc_lib_api::DATA_BF16}, 0, {}} };
        testCase.sifGivenData.numOut = 1;
        testCase.sifGivenData.numParams = 0;
        testCase.sifGivenData.params = {};
        addCase(testCase);
    }
}
