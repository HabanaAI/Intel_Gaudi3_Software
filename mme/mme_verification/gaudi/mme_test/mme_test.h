#pragma once
#include <mutex>
#include "sim_tensor.h"
#include "mme_test_gen.h"
#include "gaudi/mme.h"
#include "print_utils.h"

// main test function
int executeMMEGaudiTest(int argc, char** argv);

typedef struct
{
    int soValue;
    convTestData_t data;
    MmeTestParams_t params;
} testInfo_t;

void allocTestHostTensors(const MmeTestParams_t* params,
                          MmeSimTensor** in,
                          MmeSimTensor** weights,
                          MmeSimTensor** out,
                          MmeSimTensor** ref);

bool diffTestOutput(
    int *diff,
    const testInfo_t *ti,
    uint32_t *ref,
    uint32_t *res,
    uint32_t *resDims);

std::string test2text(const MmeTestParams_t *tp);
void printDiffMessage(int *diff, int dim, uint32_t ref, uint32_t res, const char *name, bool fp);
bool compareResults(testInfo_t& ti, bool firstDeviceIsChip, char* devBAddr, unsigned testCounter);

void dumpTestInfo(
    std::vector<MmeTestParams_t> *testsParams,
    const std::string& dumpDir);

typedef struct
{
    uint32_t poly[Mme::MME_CORES_NR];
    uint32_t seeds[Mme::MME_CORES_NR][Mme::c_mme_lfsr_seeds_nr];
} testLfsrState_t;

void getLfsrState(
    unsigned tid,
    unsigned seed,
    const std::string& lfsrDir,
    testLfsrState_t *state);


