#include "graph_optimizer_test.h"
#include "synapse_common_types.h"
#include "type_utils.h"
#include "utils.h"
#include "node_factory.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <chrono>
#include <numeric>
#include <sstream>
#include "op_validator.h"

using namespace gc::ops;

class OpValidatorTest : public GraphOptimizerTest
{
protected:
};

using ValidationTestContext = std::tuple<std::string,
                                         std::vector<synDataType>,
                                         std::vector<unsigned>,
                                         std::vector<synDataType>,
                                         std::vector<unsigned>,
                                         synDeviceType>;
// context: guid, input types, input ranks, output types, output ranks, device type
class SynOpValidatorTest
: public OpValidatorTest
, public ::testing::WithParamInterface<ValidationTestContext>
{
public:
    SynOpValidatorTest()
    : m_guid(std::get<0>(GetParam())),
      m_inputTypes(std::get<1>(GetParam())),
      m_inputRanks(std::get<2>(GetParam())),
      m_outputTypes(std::get<3>(GetParam())),
      m_outputRanks(std::get<4>(GetParam())),
      m_device(std::get<5>(GetParam()))
    {
        EXPECT_TRUE(!m_guid.empty());
        EXPECT_EQ(m_inputTypes.size(), m_inputRanks.size());
        EXPECT_EQ(m_outputTypes.size(), m_outputRanks.size());
    }

protected:
    const std::string              m_guid;
    const std::vector<synDataType> m_inputTypes;
    const std::vector<unsigned>    m_inputRanks;
    const std::vector<synDataType> m_outputTypes;
    const std::vector<unsigned>    m_outputRanks;
    const synDeviceType            m_device;

    OpValidationContext makeValidationContext()
    {
        OpValidationContext ovc;
        for (bool isInput : {true, false})
        {
            const auto& types = isInput ? m_inputTypes : m_outputTypes;
            const auto& ranks = isInput ? m_inputRanks : m_outputRanks;
            EXPECT_EQ(types.size(), ranks.size());
            auto& operandCtx = isInput ? ovc.getInputs() : ovc.getOutputs();
            operandCtx.reserve(types.size());
            for (auto idx = 0; idx < types.size(); ++idx)
            {
                TensorValidationContext tvc(ranks.at(idx), types.at(idx), DATA_TENSOR);
                operandCtx.push_back(tvc);
            }
        }
        return ovc;
    }
};

class SynOpValidatorPerfTest : public SynOpValidatorTest
{
};

TEST_P(SynOpValidatorPerfTest, DISABLED_perf_test)
{
    using namespace std::chrono;
    using Resolution               = nanoseconds;
    constexpr unsigned   N_SKIP    = 5;                // skip first 5 iterations to filter out static data init
    constexpr unsigned   NUM_ITERS = 1000 + N_SKIP;    // >>1 samples to reduce noise
    constexpr Resolution MAX_VALIDATION_TIME_NS(300);  // spec demands 500ns so leaving slack for top layer overhead

    // run several tests since other processes also introduce noise to the test as well
    constexpr unsigned N_TEST_ITERS = 10;
    using Clock                     = steady_clock;
    const auto params               = makeValidationContext();
    for (auto testIter = 0; testIter < N_TEST_ITERS; ++testIter)
    {
        std::vector<uint64_t> samples;
        samples.reserve(NUM_ITERS - N_SKIP);
        for (auto iter = 0; iter < NUM_ITERS; ++iter)
        {
            const auto start = Clock::now();
            const auto ret   = OpValidator::validateOp(m_guid.c_str(), params, m_device);
            const auto end   = Clock::now();
            ASSERT_EQ(ret, ValidationResult::SUCCESS);
            if (iter < N_SKIP) continue;
            samples.push_back(duration_cast<Resolution>(end - start).count());
        }

        uint64_t meanValidationTimeNs = std::accumulate(samples.begin(), samples.end(), 0ull) / samples.size();
        EXPECT_LT(meanValidationTimeNs, MAX_VALIDATION_TIME_NS.count());  // Expecting mean time less than 0.5usec
        std::cout << fmt::format("Mean validation time <{} samples, after discarding {}>: {} ns",
                                 samples.size(),
                                 N_SKIP,
                                 meanValidationTimeNs)
                  << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(conv3d,
                         SynOpValidatorPerfTest,
                         ::testing::Combine(::testing::Values(NodeFactory::convolution3DNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 5)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 5)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(conv2d,
                         SynOpValidatorPerfTest,
                         ::testing::Combine(::testing::Values(NodeFactory::convolutionNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(reshape,
                         SynOpValidatorPerfTest,
                         ::testing::Combine(::testing::Values(NodeFactory::reshapeNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 2)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(split,
                         SynOpValidatorPerfTest,
                         ::testing::Combine(::testing::Values(NodeFactory::splitNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(std::vector<synDataType>(6, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(6, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(concat,
                         SynOpValidatorPerfTest,
                         ::testing::Combine(::testing::Values(NodeFactory::concatenateNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(7, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(7, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));

class SynOpValidatorSuccessTest : public SynOpValidatorTest
{
};

TEST_P(SynOpValidatorSuccessTest, pass_validation)
{
    std::vector<uint64_t> samples;
    const auto            params = makeValidationContext();
    const auto            ret    = OpValidator::validateOp(m_guid.c_str(), params, m_device);
    ASSERT_EQ(ret, ValidationResult::SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(conv3d,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::convolution3DNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 5)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 5)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(conv2d,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::convolutionNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(reshape,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::reshapeNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(2, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(2, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 2)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(split,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::splitNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(std::vector<synDataType>(6, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(6, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));
INSTANTIATE_TEST_SUITE_P(concat,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::concatenateNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(7, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(7, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));

// Edge case of concat with varying num of input operands + dynamic shape.
// To WA us not knowning whether a tensor is a shape tensor, last input operand is ignored
// during type check. Hence test passes although last input type is different from the other operands.
INSTANTIATE_TEST_SUITE_P(concat_dynamic_shape,
                         SynOpValidatorSuccessTest,
                         ::testing::Combine(::testing::Values(NodeFactory::concatenateNodeTypeName),
                                            ::testing::Values(std::vector<synDataType> {syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_uint32}),
                                            ::testing::Values(std::vector<unsigned>(7, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));

class SynOpValidatorDatatypeValidationFailTest : public SynOpValidatorTest
{
};

TEST_P(SynOpValidatorDatatypeValidationFailTest, fail_dtype_validation)
{
    std::vector<uint64_t> samples;
    const auto            params = makeValidationContext();
    const auto            ret    = OpValidator::validateOp(m_guid.c_str(), params, m_device);
    ASSERT_EQ(ret, ValidationResult::INCOMPATIBLE_DATA_TYPE);
}

// Should fail due to inconsistent datatype of the last output operand (should be same as other outputs)
INSTANTIATE_TEST_SUITE_P(split,
                         SynOpValidatorDatatypeValidationFailTest,
                         ::testing::Combine(::testing::Values(NodeFactory::splitNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(std::vector<synDataType> {syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_float}),
                                            ::testing::Values(std::vector<unsigned>(6, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));

// Should fail due to inconsistent datatype of the output operand (different than inputs)
INSTANTIATE_TEST_SUITE_P(concat,
                         SynOpValidatorDatatypeValidationFailTest,
                         ::testing::Combine(::testing::Values(NodeFactory::concatenateNodeTypeName),
                                            ::testing::Values(std::vector<synDataType>(7, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(7, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_fp16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));

// Case of concat (with varying num of input operands) + dynamic shape.
// To WA us not knowning whether a tensor is a shape tensor from gc shared layer, last input operand is ignored
// during type check. Last TWO input operand datatypes differ from others hence test should fail.
INSTANTIATE_TEST_SUITE_P(concat_dynamic_shape,
                         SynOpValidatorDatatypeValidationFailTest,
                         ::testing::Combine(::testing::Values(NodeFactory::concatenateNodeTypeName),
                                            ::testing::Values(std::vector<synDataType> {syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_bf16,
                                                                                        syn_type_uint32,
                                                                                        syn_type_uint32}),
                                            ::testing::Values(std::vector<unsigned>(7, 4)),
                                            ::testing::Values(std::vector<synDataType>(1, syn_type_bf16)),
                                            ::testing::Values(std::vector<unsigned>(1, 4)),
                                            ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3)));