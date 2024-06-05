#pragma once

#include "graph_compiler/utils.h"
#include "synapse_common_types.h"
#include "test_utils.h"
#include "syn_singleton.hpp"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "perf_lib_layer_params.h"

class SynTrainingNodeOperations : public SynTrainingTestInfra
{
public:
    SynTrainingNodeOperations();
    void SetUpTest() override;
    void TearDownTest() override;

    void runMmeTest(const TestSizes&            xSize,
                    const TestSizes&            wSize,
                    const TestSizes&            ySize,
                    const synConvolutionParams& params,
                    ERepefenceOp                op,
                    synDataType                 dtype             = syn_type_single,
                    synDataType                 outputDataType    = syn_type_na,
                    bool                        usePearsonCompare = false);
    void runMmeTest(const TestSizes&            xSize,
                    const unsigned int          yChannels,
                    const synConvolutionParams& convParams,
                    ERepefenceOp                op,
                    synDataType                 dataType          = syn_type_single,
                    synDataType                 outputDataType    = syn_type_na,
                    bool                        usePearsonCompare = false);
};

class SynTrainingStridedOperations : public SynTrainingNodeOperations
{
};

class SynTrainingBatchGemmTest : public SynTrainingTestInfra
{
public:
    SynTrainingBatchGemmTest() { setTestPackage(TEST_PACKAGE_GEMM); }
    void doBatchGemmTest(const TestSizes& xSize,
                         const TestSizes& wSize,
                         const unsigned   rank,
                         ERepefenceOp     op,
                         synDataType      inputDataType  = syn_type_float,
                         synDataType      outputDataType = syn_type_float,
                         TestSizes*       optYSizes      = nullptr);

    void doBatchGemmTest(const TestSizes& xSize,
                         const TestSizes& wSize,
                         const unsigned   ifmRank,
                         const unsigned   weightRank,
                         ERepefenceOp     op,
                         synDataType      inputDataType  = syn_type_float,
                         synDataType      outputDataType = syn_type_float,
                         TestSizes*       optYSizes      = nullptr);
};

class SynTrainingGemmTest : public SynTrainingTestInfra
{
public:
    void doGemmTest(std::array<unsigned, 2> aSize,
                    std::array<unsigned, 2> bSize,
                    bool                    transposeA,
                    bool                    transposeB,
                    synDataType             dtype = syn_type_single);
};
