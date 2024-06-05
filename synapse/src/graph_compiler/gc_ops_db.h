#pragma once
#include "op_validator.h"
#ifndef _GC_OPS_DB_H_
#define _GC_OPS_DB_H_

#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <memory>
#include <utility>

#include "synapse_common_types.h"

namespace gc::ops
{
struct OpInfo;
class OpValidationContext;
using DeviceTypeToOpInfoMap = std::unordered_map<synDeviceType, std::shared_ptr<OpInfo>>;

struct OpInfo
{
    typedef bool (*ValidationFunction)(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);

    using OperandCount   = unsigned;
    using DatatypesMask  = unsigned;
    using DatatypesMasks = std::vector<DatatypesMask>;
    using DimRange       = std::pair<unsigned, unsigned>;
    using DimRanges      = std::vector<DimRange>;

    /**
     * @brief Depicts a group of API node operands expected to have the same element datatype
     */
    struct TypeGroup
    {
        TypeGroup(OpInfo::DatatypesMask mask, const std::vector<unsigned>& input, const std::vector<unsigned>& output);
        OpInfo::DatatypesMask mask;
        std::vector<unsigned> inputIndices;
        std::vector<unsigned> outputIndices;
    };

    OpInfo(OperandCount                      nInputs,
           OperandCount                      nOutputs,
           const DatatypesMasks&             supportedInputTypes,
           const DatatypesMasks&             supportedOutputTypes,
           const DimRanges&                  inputDims,
           const DimRanges&                  outputDims,
           const std::vector<TypeGroup>&     typeGroups               = {},
           bool                              isVaryingNumInputs       = false,
           bool                              isVaryingNumOutputs      = false,
           std::optional<ValidationFunction> optionalCustomValidation = std::nullopt)
    : maxInputTensors(nInputs),
      maxOutputTensors(nOutputs),
      supportedInputDatatypes(supportedInputTypes),
      supportedOutputDatatypes(supportedOutputTypes),
      supportedInputRanks(inputDims),
      supportedOutputRanks(outputDims),
      operandTypeGroups(typeGroups),
      isVaryingNumInput(isVaryingNumInputs),
      isVaryingNumOutput(isVaryingNumOutputs),
      optionalCustomValidation(optionalCustomValidation)
    {
    }

    OperandCount                      maxInputTensors;
    OperandCount                      maxOutputTensors;
    DatatypesMasks                    supportedInputDatatypes;
    DatatypesMasks                    supportedOutputDatatypes;
    DimRanges                         supportedInputRanks;
    DimRanges                         supportedOutputRanks;
    std::vector<TypeGroup>            operandTypeGroups;
    bool                              isVaryingNumInput;
    bool                              isVaryingNumOutput;
    std::optional<ValidationFunction> optionalCustomValidation;
};

class GCOpsDB final
{
public:
    GCOpsDB(const GCOpsDB&) = delete;
    GCOpsDB(GCOpsDB&&)      = delete;
    GCOpsDB& operator=(const GCOpsDB&) = delete;
    GCOpsDB& operator=(GCOpsDB&&) = delete;
    ~GCOpsDB()                    = default;

    /**
     * @brief Get Ops DB singleton instance
     *
     * @return GCOpsDB&
     */
    static GCOpsDB& instance();

    /**
     * @brief Query db for op info corresponding to {guid, device}
     *
     * @return nullptr if corresponding op info is not found
     */
    const std::shared_ptr<const OpInfo> getGCOpInfo(const std::string& guid, synDeviceType deviceType) const;

    /**
     * @brief Get initializer list of supported device types
     *
     */
    static constexpr std::initializer_list<synDeviceType> getSupportedDevices() { return m_supportedDevices; }

    /**
     * @brief Get pointer to const set of supported guids per input device
     *
     * @return nullptr if map for input device is not found
     */
    const std::shared_ptr<const std::unordered_set<std::string>> getSupportedGuids(synDeviceType device) const;

private:
    std::unordered_map<synDeviceType, std::shared_ptr<std::unordered_set<std::string>>> m_supportedGuidsPerDevice {};
    std::unordered_map<std::string, std::unique_ptr<DeviceTypeToOpInfoMap>>             m_guidToOpInfoMap {};
    static constexpr std::initializer_list<synDeviceType> m_supportedDevices = {synDeviceGaudi,
                                                                                synDeviceGaudi2,
                                                                                synDeviceGaudi3};

    GCOpsDB();
    void registerOp(const std::string& guid, const DeviceTypeToOpInfoMap& opInfoMap);
    void initSupportedGuidsSet();
};
}  // namespace gc::ops
#endif  // _GC_OPS_DB_H_