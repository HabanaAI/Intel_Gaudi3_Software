// mmeTest.cpp : Defines the entry point for the console application.
//

#undef NDEBUG
#include "mme_test.h"
#include "mme_reference.h"
#include "mme_test_cfg.h"
#include "mme_test_chip.h"
#include "tensor_utils.h"
#include <assert.h>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifdef WIN32
#include "Shlwapi.h"
#include <windows.h>
#endif

std::mutex printMutex;
using namespace MmeCommon;

void printDiffMessage(int* diff, int dim, uint32_t ref, uint32_t res, const char* name, bool fp)
{
    char buff[1024];

    sprintf(buff, "Failure in test %s\nFailed at element [", name);
    for (int i = 0; i < dim - 1; i++)
    {
        sprintf(buff + strlen(buff), "%d, ", diff[i]);
    }
    sprintf(buff + strlen(buff), "%d] - ", diff[dim - 1]);
    if (fp)
    {
        sprintf(buff + strlen(buff),
                "ref value: %f (%08x), mme value :%f (%08x)\n",
                *(float*) &ref,
                ref,
                *(float*) &res,
                res);
    }
    else
    {
        sprintf(buff + strlen(buff), "ref value: %d, mme value :%d\n", ref, res);
    }
    atomicColoredPrint(COLOR_RED, "%s", buff);
}

void dumpTestInfo(std::vector<MmeTestParams_t>* testsParams, const std::string& dumpDir)
{
    std::string fileName = dumpDir + "/tests_info.txt";

    std::ofstream of(fileName.c_str());
    if (!of.is_open())
    {
        std::cerr << "ERROR: Failed to open the tests info file. (" << fileName << ")\n";
        exit(1);
    }

    for (unsigned i = 0; i < testsParams->size(); i++)
    {
        of << "//test #" << i << std::endl;
        of << test2text(&(*testsParams)[i]) << "\n";
        of << "///////////////////////////////////////////\n\n\n";
    }

    of.close();
}

void getLfsrState(unsigned tid, unsigned seed, const std::string& lfsrDir, testLfsrState_t* state)
{
    if (!lfsrDir.empty())
    {
        for (unsigned core = 0; core < Mme::MME_CORES_NR; core++)
        {
            const char* coreStr = 0;
            switch (core)
            {
                case Mme::EMmeCore::MME_CORE_NW:
                    coreStr = "north_master";
                    break;
                case Mme::EMmeCore::MME_CORE_SW:
                    coreStr = "south_master";
                    break;
                case Mme::EMmeCore::MME_CORE_NE:
                    coreStr = "north_slave";
                    break;
                case Mme::EMmeCore::MME_CORE_SE:
                    coreStr = "south_slave";
                    break;
            }

            char core_tid[64];
            sprintf(core_tid, "/%s_%04x_lfsr.csv", coreStr, tid);
            std::string fileName = lfsrDir + core_tid;

            FILE* lfsrFile = fopen(fileName.c_str(), "r");
            if (!lfsrFile)
            {
                std::cerr << "ERROR: Failed to open input LFSR file. (" << fileName << ")\n";
                exit(1);
            }

            for (unsigned j = 0; j <= Mme::c_mme_lfsr_seeds_nr; j++)
            {
                uint32_t val;
                int ret = fscanf(lfsrFile, "%" SCNx32 ", ", &val);
                if (j > 0)
                {
                    state->seeds[core][j - 1] = val;
                }
                else
                {
                    state->poly[core] = val;
                }

                if ((ret == EOF) && (j != Mme::c_mme_lfsr_seeds_nr))
                {
                    std::cerr << "ERROR: LFSR file parsing error. (" << fileName << ")\n";
                    exit(1);
                }
            }
            fclose(lfsrFile);
        }
    }
    else
    {
        srand(~seed);
        uint32_t poly = rand();
        for (unsigned core = 0; core < Mme::MME_CORES_NR; core++)
        {
            for (unsigned j = 0; j < Mme::c_mme_lfsr_seeds_nr; j++)
            {
                state->seeds[core][j] = rand();
            }
            state->poly[core] = poly;
        }
    }
}

void allocTestHostTensors(const MmeTestParams_t* params,
                          MmeSimTensor** in,
                          MmeSimTensor** weights,
                          MmeSimTensor** out,
                          MmeSimTensor** ref)
{
    int inputShape[Mme::c_mme_max_tensor_dims];
    int weightsShape[Mme::c_mme_max_tensor_dims];
    int outputShape[Mme::c_mme_max_tensor_dims];

    memcpy(inputShape, params->inputShape, sizeof(inputShape));
    memcpy(weightsShape, params->weightsShape, sizeof(weightsShape));
    memcpy(outputShape, params->outputShape, sizeof(outputShape));

    if ((params->adjustFloatShapes) && (params->inputType == e_type_fp32))
    {
        inputShape[0] = (inputShape[0] / 2) + (inputShape[0] % 2);
        inputShape[1] = (inputShape[1] / 2) + (inputShape[1] % 2);
        weightsShape[0] = (weightsShape[0] / 2) + (weightsShape[0] % 2);
        weightsShape[1] = (weightsShape[1] / 2) + (weightsShape[1] % 2);
        outputShape[0] = (outputShape[0] / 2) + (outputShape[0] % 2);
        outputShape[1] = (outputShape[1] / 2) + (outputShape[1] % 2);
    }

    auto inputType = ((params->op == E_CONV_TEST_DEDX)) ? params->outputType : params->inputType;
    auto weightsType = ((params->op == E_CONV_TEST_DEDW)) ? params->outputType : params->inputType;
    auto outputType =
        ((params->op == E_CONV_TEST_FWD) || (params->op == E_CONV_TEST_AB) || (params->op == E_CONV_TEST_ABT) ||
         (params->op == E_CONV_TEST_ATB) || (params->op == E_CONV_TEST_ATBT))
            ? params->outputType
            : params->inputType;

    *in = new MmeSimTensor(inputShape, params->ioDim, inputType);
    *weights = new MmeSimTensor(weightsShape, params->conv.dim + 2, weightsType);
    *out = new MmeSimTensor(outputShape, params->ioDim, outputType);

    if (params->skipReference)
    {
        *ref = 0;
    }
    else
    {
        switch (params->op)
        {
            case E_CONV_TEST_FWD:
            case E_CONV_TEST_AB:
            case E_CONV_TEST_ABT:
            case E_CONV_TEST_ATB:
            case E_CONV_TEST_ATBT:
                *ref = new MmeSimTensor(outputShape, (*out)->getDim(), (*out)->getElementType());
                break;
            case E_CONV_TEST_DEDX:
                *ref = new MmeSimTensor(inputShape, (*in)->getDim(), (*in)->getElementType());
                break;
            case E_CONV_TEST_DEDW:
                *ref = new MmeSimTensor(weightsShape, (*weights)->getDim(), (*weights)->getElementType());
                break;
            default:
                assert(0);
        }
    }
}

std::string getFileName(std::string& str)
{
#ifdef WIN32
    size_t i = str.rfind('\\', str.length());
#else
    size_t i = str.rfind('/', str.length());
#endif
    if (i != std::string::npos)
    {
        return (str.substr(i + 1, str.length() - i));
    }
    else
    {
        return str;
    }
}

static const char dumpDirStr[] = "out_dir=";
static const char cfgFileStr[] = "cfg=";
static const char testTypeStr[] = "test_type=";
static const char seedStr[] = "seed=";
static const char lfsrDirStr[] = "lfsr_dir=";
static const char verifModeStr[] = "verif_mode";
static const char devAIdxStr[] = "a_idx=";
static const char devBIdxStr[] = "b_idx=";
static const char chipTypeStr[] = "--gaudiM";

void printUsageMes(const char* argv0)
{
    std::string arg0Str = argv0;
    std::cerr << "Usage: " << getFileName(arg0Str) << " " << testTypeStr << "<type> " << dumpDirStr << "path "
              << cfgFileStr << "path [" << seedStr << "seed] [" << lfsrDirStr << "path] [" << verifModeStr << "] ["
              << devAIdxStr << "idx] [" << devBIdxStr << "idx] [--gaudiM]\n";
    std::cerr << "Test types:\n";
    //    std::cerr << "\tqman_cluster\n";
    //    std::cerr << "\tcluster\n";
    //    std::cerr << "\tunit\n";
    std::cerr << "\tdevice_sim_sim\n";
    std::cerr << "\tdevice_sim_chip\n";
    std::cerr << "\tdevice_sim_null\n";
    std::cerr << "\tdevice_chip_null\n";
}

int executeMMEGaudiTest(int argc, char** argv)
{
#ifdef WIN32
    const char* testArgv[] = {
        "program\\prog",
        //"test_type=device_sim_sim",
        //"test_type=device_sim_chip",
        "test_type=device_sim_null",
        //"test_type=device_chip_null",
        //"test_type=qman_cluster",
        //"test_type=cluster",
        //"test_type=unit",
        "cfg=C:/Users/agoldman.HABANA-LABS/Documents/h3/mme_verif/mme_sim/"
        "configs/gemm_tests.cfg",
        //"cfg=C:/Users/agoldman.HABANA-LABS/Documents/h3/mme_verif/mme_sim/configs/sbreuse_tests.cfg",
        // cfg=C:/Users/agoldman.HABANA-LABS/Documents/h3/mme_verif/mme_sim/configs/topologies.cfg",
        //"cfg=C:/Users/agoldman.HABANA-LABS/Documents/h3/mme_verif/mme_sim/configs/tmp.cfg",
        //"out_dir=C:/temp/gemm_dump",
        "seed=12345678",
        //"lfsr_dir=C:/temp/gemm_dump",
    };
    argc = sizeof(testArgv) / sizeof(testArgv[0]);
    argv = testArgv;
#endif

    std::string dumpDir;
    std::string cfgFile;
    std::string lfsrDir;
    std::string testType;

    bool exitStatus = true;
    bool verifMode = false;
    unsigned seed = time(0);
    unsigned devAIdx = 0;
    unsigned devBIdx = 0;
    bool gaudiM = false;
    try
    {
        for (unsigned i = 1; i < argc; i++)
        {
            if (!strncmp(dumpDirStr, argv[i], strlen(dumpDirStr)))
            {
                dumpDir = argv[i] + strlen(dumpDirStr);
            }
            else if (!strncmp(testTypeStr, argv[i], strlen(testTypeStr)))
            {
                testType = argv[i] + strlen(testTypeStr);
            }
            else if (!strncmp(lfsrDirStr, argv[i], strlen(lfsrDirStr)))
            {
                lfsrDir = argv[i] + strlen(lfsrDirStr);
            }
            else if (!strncmp(cfgFileStr, argv[i], strlen(cfgFileStr)))
            {
                cfgFile = argv[i] + strlen(cfgFileStr);
            }
            else if (!strncmp(seedStr, argv[i], strlen(seedStr)))
            {
                seed = atoi(argv[i] + strlen(seedStr));
            }
            else if (!strncmp(verifModeStr, argv[i], strlen(verifModeStr)))
            {
                verifMode = true;
            }
            else if (!strncmp(devAIdxStr, argv[i], strlen(devAIdxStr)))
            {
                devAIdx = atoi(argv[i] + strlen(devAIdxStr));
            }
            else if (!strncmp(devBIdxStr, argv[i], strlen(devBIdxStr)))
            {
                devBIdx = atoi(argv[i] + strlen(devBIdxStr));
            }
            else if (!strncmp(chipTypeStr, argv[i], strlen(chipTypeStr)))
            {
                gaudiM = true;
            }
            else
            {
                std::cerr << "ERROR: Unknown config - " << argv[i] << "\n";
                printUsageMes(argv[0]);
                return EXIT_FAILURE;
            }
        }

        if (cfgFile.empty())
        {
            std::cerr << "ERROR: Config file not set\n";
            printUsageMes(argv[0]);
            return EXIT_FAILURE;
        }

        if (testType.empty())
        {
            std::cerr << "ERROR: Test type is not set\n";
            printUsageMes(argv[0]);
            return EXIT_FAILURE;
        }

        atomicColoredPrint(COLOR_YELLOW, "INFO: Test type - %s\n", testType.c_str());
        atomicColoredPrint(COLOR_YELLOW, "INFO: Config file - %s\n", cfgFile.c_str());
        atomicColoredPrint(COLOR_YELLOW, "INFO: Configuration seed - %u\n", seed);

        if (!lfsrDir.empty())
        {
            atomicColoredPrint(COLOR_YELLOW, "INFO: LFSR path - %s\n", lfsrDir.c_str());
        }

        if (!dumpDir.empty())
        {
            atomicColoredPrint(COLOR_YELLOW, "INFO: Output directory - %s\n", dumpDir.c_str());
#ifdef WIN32
            if (-1 == GetFileAttributes(dumpDir))
            {
                if (!CreateDirectory(dumpDir, 0))
#else
            struct stat sb;
            if (stat(dumpDir.c_str(), &sb))
            {
                if (mkdir(dumpDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
#endif
                {
                    std::cerr << "ERROR: Failed to create the requested output path. (" << dumpDir << ")\n";
                    return EXIT_FAILURE;
                }
            }
        }

        if (gaudiM)
        {
            atomicColoredPrint(COLOR_YELLOW, "INFO: chipType = gaudiM\n");
        }

        srand(seed);
        std::vector<MmeTestParams_t> testsParams;
        cfgFile2Tests(cfgFile, seed, &testsParams);

        if (!testsParams.size())
        {
            std::cerr << "ERROR: Config file parsing error. (" << cfgFile << ")\n";
            return EXIT_FAILURE;
        }

        //    if (!strcmp(testType, "qman_cluster"))
        //    {
        //        runQmanClusterTests(&testsParams, dumpDir, lfsrDir);
        //    }
        //    else if (!strcmp(testType, "cluster"))
        //    {
        //        runClusterTests(&testsParams, dumpDir, lfsrDir);
        //    }
        //    if (!strcmp(testType, "unit"))
        //    {
        //        runUnitTests(&testsParams, dumpDir, lfsrDir);
        //    }
        if (testType == "device_sim_sim")
        {
            exitStatus = runChipTests(&testsParams, dumpDir, lfsrDir, e_sim, e_sim, devAIdx, devBIdx, verifMode, gaudiM);
        }
        else if (testType == "device_sim_chip")
        {
            exitStatus = runChipTests(&testsParams, dumpDir, lfsrDir, e_sim, e_chip, devAIdx, devBIdx, verifMode, gaudiM);
        }
        else if (testType == "device_sim_null")
        {
            exitStatus = runChipTests(&testsParams, dumpDir, lfsrDir, e_sim, e_null, devAIdx, devBIdx, verifMode, gaudiM);
        }
        else if (testType == "device_chip_null")
        {
            exitStatus = runChipTests(&testsParams, dumpDir, lfsrDir, e_chip, e_null, devAIdx, devBIdx, verifMode, gaudiM);
        }
        else
        {
            std::cerr << "ERROR: Unknown test type. (" << testType << ")\n";
            printUsageMes(argv[0]);
            return EXIT_FAILURE;
        }
    }
    catch (...)
    {
        exitStatus = false;
    }

    if (exitStatus)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
