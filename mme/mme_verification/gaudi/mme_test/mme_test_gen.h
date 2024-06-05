#pragma once
#include "convolution_params.h"
#include "gaudi/mme.h"
#include "mme_test_gen_interface.h"
#include "gaudi/headers/mme_user.h"
#include "sim_tensor.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include <stdint.h>

using namespace gaudi;

static const unsigned c_cl_size = 128;

#define _IN_
#define _OUT_
#define _IO_

enum EConvTestOp
{
    E_CONV_TEST_FWD = 0,
    E_CONV_TEST_DEDX = 1,
    E_CONV_TEST_DEDW = 2,
    E_CONV_TEST_AB = 3,
    E_CONV_TEST_ABT = 4,
    E_CONV_TEST_ATB = 5,
    E_CONV_TEST_ATBT = 6,
};

enum POLE
{
    NORTH_POLE = 0,
    SOUTH_POLE = 1,
};

typedef struct
{
    char name[1024];
    int tid;
    ConvolutionParams conv;
    BGemm bgemm;
    MmeCommon::EMmeDataType inputType;
    MmeCommon::EMmeDataType outputType;
    int xMinVal;
    int xMaxVal;
    int wMinVal;
    int wMaxVal;
    int yMinVal;
    int yMaxVal;
    int ioDim;
    int wDim;
    int inputShape[Mme::c_mme_max_tensor_dims];
    int weightsShape[Mme::c_mme_max_tensor_dims];
    int outputShape[Mme::c_mme_max_tensor_dims];
    uint32_t soAddrLow[4];
    uint32_t soAddrHigh;
    bool skipReference;
    bool fp;
    uint64_t sramBase;
    uint64_t sramSize;
    uint64_t hbmBase;
    uint64_t smBase;
    bool stochsaticConversion;
    MmeCommon::RoundingMode rm;
    EConvTestOp op;
    bool shuffle;
    unsigned seed;
    bool setId;
    bool multipleTests;
    bool sbReuse;
    bool sbReuseInStripes;
    bool unrollEn;
    bool dedwAsBgemmEn;
    bool recurringMisalignmentOptEn;
    bool lower;
    unsigned repeats;
    MmeCommon::EMmeGeometry geometry;
    MmeCommon::EMmePattern pattern;
    // TODO [SW-100319] add support for reduction tests in Gaudi mme test
    MmeCommon::EMmeReductionOp reductionOp = MmeCommon::e_mme_reduction_none;
    MmeCommon::EMmeReductionRm reductionRm = MmeCommon::e_mme_reduction_round_half_to_nearest_even;
    POLE pole;
    bool xInSram;
    bool yInSram;
    bool wInSram;
    bool programInSram;
    int sramStreamingBudget;
    bool dumpEn;
    bool randomMD;
    bool signalPartial;
    bool fullDesc;
    bool memsetOutput;
    bool memsetVoidPixels;
    bool incDec;
    bool loop;
    bool adjustFloatShapes;
    bool scaledRandomValues;
    bool verifMode;
} MmeTestParams_t;

typedef struct
{
    MmeSimTensor in;
    MmeSimTensor weights;
    MmeSimTensor out;
    MmeSimTensor* hostIn;
    MmeSimTensor* hostWeights;
    MmeSimTensor* hostOut;
    MmeSimTensor* hostRef;
} convTestData_t;

void genConvTest(const MmeTestParams_t* params,
                 const convTestData_t* data,
                 int* soValue,
                 std::list<MmeRegWriteCmd>* cmds,
                 const Mme::Desc* prevDesc,
                 Mme::Desc* lastDesc,
                 const bool verifMode,
                 unsigned testCounter);