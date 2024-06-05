#pragma once


#include "data_container.h"
#include "json_utils.h"
#include "layout.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

class DataComparator
{
public:
    struct Config
    {
        Config() {};
        Config(const std::string& configFilePath, const nlohmann_hcl::json& graph)
        {
            nlohmann_hcl::json jsonData = json_utils::jsonFromFile(configFilePath);

            bitAccurate              = json_utils::get(jsonData, "bitAccurate", bitAccurate);
            breakOnFirstError        = json_utils::get(jsonData, "breakOnFirstError", breakOnFirstError);
            pearsonThreshold         = json_utils::get(jsonData, "pearsonThreshold", pearsonThreshold);
            l2NormMinRatio           = json_utils::get(jsonData, "l2NormMinRatio", l2NormMinRatio);
            fp8EpsilonAbsoluteError  = json_utils::get(jsonData, "fp8EpsilonAbsoluteError", fp8EpsilonAbsoluteError);
            fp8EpsilonRelativeError  = json_utils::get(jsonData, "fp8EpsilonRelativeError", fp8EpsilonRelativeError);
            bf16EpsilonAbsoluteError = json_utils::get(jsonData, "bf16EpsilonAbsoluteError", bf16EpsilonAbsoluteError);
            bf16EpsilonRelativeError = json_utils::get(jsonData, "bf16EpsilonRelativeError", bf16EpsilonRelativeError);
            fp16EpsilonAbsoluteError = json_utils::get(jsonData, "fp16EpsilonAbsoluteError", fp16EpsilonAbsoluteError);
            fp16EpsilonRelativeError = json_utils::get(jsonData, "fp16EpsilonRelativeError", fp16EpsilonRelativeError);
            f32EpsilonAbsoluteError  = json_utils::get(jsonData, "f32EpsilonAbsoluteError", f32EpsilonAbsoluteError);
            f32EpsilonRelativeError  = json_utils::get(jsonData, "f32EpsilonRelativeError", f32EpsilonRelativeError);
            maxLenAbsError           = json_utils::get(jsonData, "maxLenAbsError", maxLenAbsError);
            maxLenAverageVsStdev     = json_utils::get(jsonData, "maxLenAverageVsStdev", maxLenAverageVsStdev);

            dumpTensorsData = json_utils::get_or_default<std::vector<std::string>>(jsonData, "dumpTensorsData");
            ignoredTensors  = json_utils::get_or_default<std::vector<std::string>>(jsonData, "ignoredTensors");
            ignoredGuids    = json_utils::get_or_default<std::vector<std::string>>(jsonData, "ignoredGuids");

            if (!graph.empty())
            {
                auto outputTensors = getOutputTensors(ignoredGuids, graph);
                ignoredTensors.insert(ignoredTensors.end(), outputTensors.begin(), outputTensors.end());
            }
        }

        bool                     bitAccurate              = false;
        bool                     breakOnFirstError        = true;
        float                    pearsonThreshold         = 0.95f;
        float                    l2NormMinRatio           = 1.0f - 0.2f;
        float                    fp8EpsilonAbsoluteError  = 1e-2;
        float                    fp8EpsilonRelativeError  = 1e-2;
        float                    bf16EpsilonAbsoluteError = 1e-2;
        float                    bf16EpsilonRelativeError = 1e-2;
        float                    fp16EpsilonAbsoluteError = 1e-2;
        float                    fp16EpsilonRelativeError = 1e-2;
        float                    f32EpsilonAbsoluteError  = 1e-3;
        float                    f32EpsilonRelativeError  = 1e-3;
        unsigned                 maxLenAbsError           = 1;
        unsigned                 maxLenAverageVsStdev     = 15;
        std::vector<std::string> dumpTensorsData;
        std::vector<std::string> ignoredTensors;
        std::vector<std::string> ignoredGuids;

    private:
        std::vector<std::string> getOutputTensors(const std::vector<std::string>& guids,
                                                  const nlohmann_hcl::json&       graph)
        {
            const auto& nodes = json_utils::get(graph, "nodes");

            std::vector<std::string> ret;
            for (const auto& n : nodes)
            {
                const std::string& guid = json_utils::get(n, "guid");
                if (!matchesPattern(guid, guids)) continue;
                auto outputTensors = json_utils::get(n, "output_tensors", std::vector<std::string>());
                ret.insert(ret.end(), outputTensors.begin(), outputTensors.end());
            }
            return ret;
        }
    };
    struct Status
    {
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    DataComparator(const std::shared_ptr<DataContainer>& actualData,
                   const std::shared_ptr<DataContainer>& referenceData,
                   Config                                config = Config())
    : m_actualData(actualData), m_referenceData(referenceData), m_config(std::move(config))
    {
    }

    Status compare(const std::string& tensorName) const
    {
        Status      ret;
        std::string sts;
        switch (m_referenceData->getDataType(tensorName))
        {
            case syn_type_int8:
                sts = compare<int8_t>(tensorName);
                break;
            case syn_type_int16:
                sts = compare<int16_t>(tensorName);
                break;
            case syn_type_int32:
                sts = compare<int32_t>(tensorName);
                break;
            case syn_type_uint4:
            case syn_type_uint8:
                sts = compare<uint8_t>(tensorName);
                break;
            case syn_type_uint16:
                sts = compare<uint16_t>(tensorName);
                break;
            case syn_type_uint32:
                sts = compare<uint32_t>(tensorName);
                break;
            case syn_type_float:
                sts = compare<float>(tensorName);
                break;
            case syn_type_bf16:
                sts = compare<bfloat16>(tensorName);
                break;
            case syn_type_fp16:
                sts = compare<fp16_t>(tensorName);
                break;
            case syn_type_int64:
                sts = compare<int64_t>(tensorName);
                break;
            case syn_type_uint64:
                sts = compare<uint64_t>(tensorName);
                break;
            case syn_type_fp8_143:
                sts = compare<fp8_143_t>(tensorName);
                break;
            case syn_type_fp8_152:
                sts = compare<fp8_152_t>(tensorName);
                break;
            case syn_type_na:
            case syn_type_int4:
            case syn_type_tf32:
            case syn_type_hb_float:
            case syn_type_ufp16:
            case syn_type_max:
                throw std::runtime_error("unsupported data type: " +
                                         std::to_string(m_referenceData->getDataType(tensorName)));
        }
        if (!sts.empty())
        {
            if (matchesPattern(tensorName, m_config.ignoredTensors))
            {
                ret.warnings.push_back(
                    fmt::format("{}, warning (mismatch requested to be ignored): {}", tensorName, sts));
            }
            else
            {
                ret.errors.push_back(fmt::format("{}, error: {}", tensorName, sts));
            }
        }
        return ret;
    }

    Status compare() const
    {
        Status ret;
        auto   dataProviderTensorsNames  = m_referenceData->getTensorsNames();
        auto   dataCollectorTensorsNames = m_actualData->getTensorsNames();
        for (const auto& tn : dataProviderTensorsNames)
        {
            if (std::find(dataCollectorTensorsNames.begin(), dataCollectorTensorsNames.end(), tn) ==
                dataCollectorTensorsNames.end())
            {
                continue;
            }

            auto currRet = compare(tn);
            ret.errors.insert(ret.errors.end(), currRet.errors.begin(), currRet.errors.end());
            if (!currRet.errors.empty() && m_config.breakOnFirstError) break;
        }
        return ret;
    }

private:
    static bool matchesPattern(const std::string& item, const std::vector<std::string>& patterns)
    {
        return std::any_of(patterns.begin(), patterns.end(), [&](const std::string& i) {
            return item.find(i) != std::string::npos;
        });
    }

    template<class T>
    void dumpToFile(const std::string& filePath, const std::vector<T>& ref, const std::vector<T>& actual) const
    {
        std::ofstream f(filePath);
        f << fmt::format("Index,Refernece (float),Actual (float)\n");
        for (size_t i = 0; i < ref.size(); ++i)
        {
            f << fmt::format("{},{},{}\n", i, float(ref[i]), float(actual[i]));
        }
    }

    template<class T>
    void transposeBuffer(const gc::Permutation&    permutation,
                         synDataType               dataType,
                         const std::vector<TSize>& outputShape,
                         std::vector<T>&           buffer) const
    {
        if (permutation.size() == 0 || permutation.isIdentity()) return;

        std::vector<TSize> inputShape = outputShape;
        permutation.permuteShape(inputShape);
        auto invPermute = permutation.getInversePermutation();

        synTransposeParamsNDims params {};
        params.tensorDim = invPermute.size();
        const auto& vals = invPermute.getValues();
        std::copy(vals.begin(), vals.end(), params.permutation);

        char*     data    = reinterpret_cast<char*>(buffer.data());
        TensorPtr IFM     = std::make_shared<Tensor>(inputShape.size(), inputShape.data(), dataType, data);
        TensorPtr OFM_ref = std::make_shared<Tensor>(outputShape.size(), outputShape.data(), dataType);

        NodePtr ref_n = NodeFactory::createNode({IFM}, {OFM_ref}, &params, NodeFactory::transposeNodeTypeName, "");
        ref_n->RunOnCpu();

        const T* ref_resultArray = (T*)OFM_ref->map();
        uint64_t numElements     = multiplyElements(outputShape);
        std::copy_n(ref_resultArray, numElements, buffer.data());
    }

    template<class T>
    std::string compare(const std::string& name) const
    {
        auto expected = m_referenceData->getElements<T>(name);
        auto actual   = m_actualData->getElements<T>(name);

        auto dataType = m_referenceData->getDataType(name);
        auto shape    = m_referenceData->getShape(name);

        auto perAct      = m_actualData->getPermutation(name);
        auto compilePerm = gc::Permutation(DimVector(perAct.begin(), perAct.end()));

        auto perRef  = m_referenceData->getPermutation(name);
        auto recPerm = gc::Permutation(DimVector(perRef.begin(), perRef.end()));

        transposeBuffer(compilePerm, dataType, shape, actual);
        transposeBuffer(recPerm, dataType, shape, expected);

        if (matchesPattern(name, m_config.dumpTensorsData))
        {
            dumpToFile<T>(fmt::format("{}.csv", sanitizeFileName(name)), expected, actual);
        }

        return m_config.bitAccurate ? validateResultBitAccurate(expected, actual) : validateResult(expected, actual);
    }

    float getErrorThreshold(fp8_143_t ref) const
    {
        return m_config.bf16EpsilonAbsoluteError + m_config.fp8EpsilonAbsoluteError * std::abs(float(ref));
    }

    float getErrorThreshold(fp8_152_t ref) const
    {
        return m_config.bf16EpsilonAbsoluteError + m_config.fp8EpsilonAbsoluteError * std::abs(float(ref));
    }

    float getErrorThreshold(bfloat16 ref) const
    {
        return m_config.bf16EpsilonAbsoluteError + m_config.bf16EpsilonRelativeError * std::abs(float(ref));
    }

    float getErrorThreshold(fp16_t ref) const
    {
        return m_config.fp16EpsilonAbsoluteError + m_config.fp16EpsilonRelativeError * std::abs(float(ref));
    }

    float getErrorThreshold(float ref) const
    {
        return m_config.f32EpsilonAbsoluteError + m_config.f32EpsilonRelativeError * std::abs(ref);
    }

    template<typename T>
    std::string hasNanOrInf(const std::vector<T>& values) const
    {
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            float v = static_cast<float>(values[idx]);
            if (std::isnan(v))
            {
                return fmt::format("nan found at index: {} | ", idx);
            }
            if (std::isinf(v))
            {
                return fmt::format("inf found at index: {} | ", idx);
            }
        }

        return "";
    }

    template<typename T>
    bool isMonotonic(const std::vector<T>& values) const
    {
        if (values.empty()) return false;
        auto  res            = std::minmax_element(values.begin(), values.end());
        float min            = float(*res.first);
        float max            = float(*res.second);
        float errorThreshold = getErrorThreshold(*res.first);

        return std::abs(max - min) <= errorThreshold;
    }

    template<typename T>
    static float calculateL2Norm(const std::vector<T>& values)
    {
        float sumSquared = 0;
        for (unsigned idx = 0; idx < values.size(); idx++)
        {
            sumSquared += (float)values[idx] * (float)values[idx];
        }
        return sqrt(sumSquared);
    };

    template<typename T, typename R = float>
    std::string validateL2Norm(const std::vector<T>& expected, const std::vector<R>& result) const
    {
        float l2NormExpected = calculateL2Norm(expected);
        float l2NormResult   = calculateL2Norm(result);

        if (l2NormExpected == 0 && l2NormResult == 0)
        {
            return "";
        }

        if (l2NormExpected > l2NormResult)
        {
            std::swap(l2NormExpected, l2NormResult);
        }
        if (m_config.l2NormMinRatio > l2NormExpected / l2NormResult)
        {
            return fmt::format("l2Norm min ratio: {} is greater than L2Norm actual ration: {} | ",
                               m_config.l2NormMinRatio,
                               l2NormExpected / l2NormResult);
        }
        return "";
    }

    template<typename T, typename R = float>
    std::string validatePearson(const std::vector<T>& expected, const std::vector<R>& result) const
    {
        double sumResult = 0, sumExpected = 0, sumResultXSumExpected = 0;
        double squareSumResult = 0, squareSumExpected = 0;

        for (int i = 0; i < expected.size(); i++)
        {
            sumResult = sumResult + (double)result[i];

            sumExpected = sumExpected + (double)expected[i];

            sumResultXSumExpected = sumResultXSumExpected + (double)result[i] * (double)expected[i];

            squareSumResult   = squareSumResult + (double)result[i] * (double)result[i];
            squareSumExpected = squareSumExpected + (double)expected[i] * (double)expected[i];
        }

        // Cannot compare all-zeros cases.
        if (sumResult == 0 || sumExpected == 0)
        {
            return "";
        }

        // use formula for calculating correlation coefficient.
        double corr = (double)(expected.size() * sumResultXSumExpected - sumResult * sumExpected) /
                        sqrt((expected.size() * squareSumResult - sumResult * sumResult) *
                            (expected.size() * squareSumExpected - sumExpected * sumExpected));
        if (m_config.pearsonThreshold > corr)
        {
            return fmt::format("pearson threshold: {} is greater than correlation coefficient: {} | ",
                                m_config.pearsonThreshold,
                                corr);
        }
        return "";
    }

    template<typename T, typename R = float>
    static std::string validateAvgErrorVsStdev(const std::vector<T>& expected, const std::vector<R>& result)
    {
        float var     = 0.0f;
        float mean    = 0.0f;
        float avg_err = 0.0f;

        for (unsigned idx = 0; idx < expected.size(); idx++)
        {
            float ref = expected[idx];
            float x   = result[idx];
            var += ref * ref / expected.size();
            mean += ref / expected.size();
            avg_err += std::abs(ref - x) / ((float)expected.size());
        }
        var -= mean * mean;
        float rms = std::sqrt(var) * 0.5;
        if (avg_err > rms)
        {
            return fmt::format("average error: {} is greater than rms: {} | ", avg_err, rms);
        }
        return "";
    };

    template<typename T, typename R = float>
    std::string validateAbsoluteErrorVsEpsilon(const std::vector<T>& expected, const std::vector<R>& result) const
    {
        for (unsigned idx = 0; idx < expected.size(); idx++)
        {
            float ref = expected[idx];
            float res = result[idx];

            float absoluteError  = std::abs(ref - res);
            float errorThreshold = getErrorThreshold(expected[idx]);

            if (absoluteError > errorThreshold)
            {
                std::stringstream ss;
                ss << "absolute error: " << absoluteError << " is greater than error threshold: " << errorThreshold
                   << " | ";
                return ss.str();
            }
        }
        return "";
    };

    template<typename T, typename R = float>
    std::string validateResult(const std::vector<T>& expected, const std::vector<R>& actual) const
    {
        if (expected.size() != actual.size())
        {
            throw std::runtime_error(
                fmt::format("validate result expected/actual size mismatch, expected size: {}, actual size: {}",
                            expected.size(),
                            actual.size()));
        }
        std::string ret = hasNanOrInf(actual);
        if (expected.size() <= m_config.maxLenAbsError || isMonotonic(expected))
        {
            ret += validateAbsoluteErrorVsEpsilon(expected, actual);
        }
        else if (expected.size() <= m_config.maxLenAverageVsStdev)
        {
            ret += validateAvgErrorVsStdev(expected, actual);
        }
        else
        {
            ret += validatePearson(expected, actual);
            ret += validateL2Norm(expected, actual);
        }
        return ret;
    }

    template<typename T>
    std::string toHex(const T& val) const
    {
        auto*                               b = reinterpret_cast<const uint8_t*>(&val);
        std::array<char, sizeof(T) * 2 + 1> ret;
        ret.back() = 0;
        for (size_t i = 0; i < sizeof(T); ++i)
        {
            fmt::format_to_n(&ret[(sizeof(T) - 1 - i) * 2], 2, "{:02X}", b[i]);
        }

        return ret.data();
    }

    template<typename T>
    std::string validateResultBitAccurate(const std::vector<T>& expected, const std::vector<T>& actual) const
    {
        if (expected.size() != actual.size())
        {
            return fmt::format("ref data size is different from actual data size, expected: {}, actual: {}",
                               expected.size(),
                               actual.size());
        }
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (expected[i] != actual[i])
            {
                if (std::isnan(float(expected[i])) && std::isnan(float(actual[i]))) continue;
                return fmt::format("data mismatch on index: {}, expected: {} (0x{}), actual: {} (0x{})",
                                   i,
                                   float(expected[i]),
                                   toHex(expected[i]),
                                   float(actual[i]),
                                   toHex(actual[i]));
            }
        }
        return "";
    }

private:
    std::shared_ptr<DataContainer> m_actualData;
    std::shared_ptr<DataContainer> m_referenceData;
    Config                         m_config;
};