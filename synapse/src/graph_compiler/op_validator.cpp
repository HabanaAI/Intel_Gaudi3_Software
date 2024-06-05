#include "op_validator.h"
#include "data_type_utils.h"
#include "gc_ops_db.h"
#include "habana_global_conf.h"
#include "synapse_common_types.h"

namespace gc::ops
{
bool                    OpValidator::s_enabled = true;  // enabled by default
static std::string_view toString(ValidationResult status);

static std::string_view operandRoleToStr(bool isInput)
{
    return isInput ? "input" : "output";
}

const std::vector<std::pair<OpValidator::ValidationFunction, std::string_view>> OpValidator::s_validations = {
    {OpValidator::validateNumOperands, "number of operand validation"},
    {OpValidator::validateRank, "operands rank validation"},
    {OpValidator::validateDatatypes, "operands datatype validation"},
    {OpValidator::validateCustom, "operands datatype validation"}};

OpValidator::ScopedSkipValidation::ScopedSkipValidation() : m_prevMode(enabled())
{
    OpValidator::setEnabled(false);
}

OpValidator::ScopedSkipValidation::~ScopedSkipValidation()
{
    OpValidator::setEnabled(m_prevMode);
}

OpValidator::OpValidator(const std::string& guid, const OpValidationContext& ctx, synDeviceType deviceType)
: m_guid(guid), m_opCtx(ctx), m_device(deviceType)
{
}

void OpValidator::setEnabled(bool mode)
{
    s_enabled = mode;
}

bool OpValidator::enabled()
{
    return s_enabled;
}

ValidationResult
OpValidator::validateOp(const std::string& guid, const OpValidationContext& ovc, synDeviceType deviceType)
{
    if (!enabled()) return ValidationResult::SUCCESS;
    OpValidator validator(guid, ovc, deviceType);
    return validator.runValidations();
}

ValidationResult OpValidator::runValidations() const
{
    const auto pGCOpInfo = GCOpsDB::instance().getGCOpInfo(getGuid(), getDevice());
    if (!pGCOpInfo)
    {
        // Don't fail gc node validation on guids not supported by a certain device
        LOG_DEBUG(OP_VALIDATOR, "{}: Empty op info for input guid: {}", HLLOG_FUNC, getGuid());
        return ValidationResult::GUID_NOT_FOUND;
    }

    HB_ASSERT(!getTestedOpCtx().empty(),
              "Expecting non-empty op validation context, guid={}, device={}",
              getGuid(),
              getDevice());

    for (const auto& [validationFunc, validationStr] : s_validations)
    {
        LOG_TRACE(OP_VALIDATOR, "{}: running {}", HLLOG_FUNC, validationStr);
        const auto rc = validationFunc(*pGCOpInfo, getTestedOpCtx());
        if (rc != ValidationResult::SUCCESS)
        {
            LOG_ERR(OP_VALIDATOR,
                    "{}: Failed {} for guid={}, device={}, rc={}",
                    HLLOG_FUNC,
                    validationStr,
                    getGuid(),
                    getDevice(),
                    toString(rc));
            return rc;
        }
    }
    return ValidationResult::SUCCESS;
}

bool OpValidator::isSupportedDatatype(synDataType testedType, unsigned supportedTypesMask)
{
    return ((supportedTypesMask & testedType) != 0) ||
           (GCFG_SYNAPSE_DATA_TYPE_SELECTION.value() && testedType == syn_type_na);
}

ValidationResult OpValidator::validateRank(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx)
{
    const auto validateOperandsRank = [&gcOpInfo, &opValCtx](bool isInput) {
        const auto& testedOpCtx       = isInput ? opValCtx.getInputs() : opValCtx.getOutputs();
        const auto& expectedDimRanges = isInput ? gcOpInfo.supportedInputRanks : gcOpInfo.supportedOutputRanks;
        bool        isVaryingOperands = isInput ? gcOpInfo.isVaryingNumInput : gcOpInfo.isVaryingNumOutput;

        unsigned idx = 0;
        for (const auto& ctx : testedOpCtx)
        {
            if (!ctx.empty() && ctx.getRank() > 0)
            {
                const auto& [minDim, maxDim] = isVaryingOperands ? expectedDimRanges.at(0) : expectedDimRanges.at(idx);

                if (ctx.getRank() < minDim || maxDim < ctx.getRank())
                {
                    LOG_ERR(OP_VALIDATOR,
                            "Incompatible rank {} for {} tensor index {}, min dim: {}, max dim: {}",
                            ctx.getRank(),
                            operandRoleToStr(isInput),
                            idx,
                            minDim,
                            maxDim);
                    return isInput ? ValidationResult::INCOMPATIBLE_INPUT_DIMENSION
                                   : ValidationResult::INCOMPATIBLE_OUTPUT_DIMENSION;
                }
            }
            ++idx;
        }
        return ValidationResult::SUCCESS;
    };

    for (const auto& isInput : {true, false})
    {
        auto sts = validateOperandsRank(isInput);
        if (sts != ValidationResult::SUCCESS)
        {
            return sts;
        }
    }
    return ValidationResult::SUCCESS;
}

ValidationResult OpValidator::validateDatatypes(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx)
{
    if (gcOpInfo.operandTypeGroups.empty())
    {
        // Ignore type validation
        return ValidationResult::SUCCESS;
    }
    const auto validateOperandsTypeGroup =
        [&](const auto& operandIndices, const auto& operandsCtx, auto& supportedTypes, bool input) {
            for (const auto& idx : operandIndices)
            {
                if (idx >= operandsCtx.size()) continue;
                const auto& ctx = operandsCtx.at(idx);
                if (ctx.empty()) continue;
                LOG_DEBUG(OP_VALIDATOR,
                          "{}[{}] supportedTypesMask: 0x{:08x}, operand.datatype: 0x{:08x} <{}>",
                          operandRoleToStr(input),
                          idx,
                          supportedTypes,
                          ctx.getDatatype(),
                          getStringFromSynDataType(ctx.getDatatype()));
                if (!isSupportedDatatype(ctx.getDatatype(), supportedTypes))
                {
                    LOG_ERR(OP_VALIDATOR,
                            "Incompatible datatype {} for {} tensor idx {}",
                            getStringFromSynDataType(ctx.getDatatype()),
                            operandRoleToStr(input),
                            idx);
                    return ValidationResult::INCOMPATIBLE_DATA_TYPE;
                }
                supportedTypes = ctx.getDatatype();
            }
            return ValidationResult::SUCCESS;
        };

    // test operands for type validity & consistency
    for (const auto& typeGroup : gcOpInfo.operandTypeGroups)
    {
        OpInfo::DatatypesMask dtypeMask = typeGroup.mask;  // updated inside the validation lambda
        auto                  validateInputsResult =
            validateOperandsTypeGroup(typeGroup.inputIndices, opValCtx.getInputs(), dtypeMask, true /*input*/);
        if (validateInputsResult != ValidationResult::SUCCESS) return validateInputsResult;
        auto validateOutputsResult =
            validateOperandsTypeGroup(typeGroup.outputIndices, opValCtx.getOutputs(), dtypeMask, false /*input*/);
        if (validateOutputsResult != ValidationResult::SUCCESS) return validateOutputsResult;
    }

    // Specific handling for split and concat which differ due to them having varying operands.
    // Split can have N outputs and Concat can have N inputs, hence workaround is different for inputs/outputs.
    // In case of a dynamic shape concat, shape tensor is the last input tensor.
    // Due to lack of knowledge from shared_layer validation context whether a tensor is a shape,
    // this workaround skips validation of the last input tensor in case of varying outputs while
    // requiring all tensors have the same dtype.
    // In case of a dynamic shape split, shape tensor is an optional input hence in that case
    // all outputs are requiring to have the same dtype.
    if (gcOpInfo.isVaryingNumInput ^ gcOpInfo.isVaryingNumOutput)
    {
        HB_ASSERT(!gcOpInfo.operandTypeGroups.empty(), "Expecting at least one type group");
        OpInfo::DatatypesMask supportedType = gcOpInfo.operandTypeGroups.at(0).mask;
        const auto&           operandsCtx   = gcOpInfo.isVaryingNumInput ? opValCtx.getInputs() : opValCtx.getOutputs();
        HB_ASSERT(!operandsCtx.empty(), "");

        unsigned maxOperandIdx = gcOpInfo.isVaryingNumInput ? operandsCtx.size() - 1 : operandsCtx.size();
        for (unsigned idx = 0; idx < maxOperandIdx; ++idx)
        {
            const auto& tensorCtx = operandsCtx.at(idx);
            if (tensorCtx.empty()) continue;
            if (!isSupportedDatatype(tensorCtx.getDatatype(), supportedType))
            {
                LOG_ERR(OP_VALIDATOR,
                        "Incompatible datatype {} for {} tensor idx {}",
                        getStringFromSynDataType(tensorCtx.getDatatype()),
                        operandRoleToStr(gcOpInfo.isVaryingNumInput),
                        idx);
                return ValidationResult::INCOMPATIBLE_DATA_TYPE;
            }
            supportedType = tensorCtx.getDatatype();
        }
    }
    return ValidationResult::SUCCESS;
}

ValidationResult OpValidator::validateNumOperands(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx)
{
    if (!gcOpInfo.isVaryingNumInput)
    {
        if (opValCtx.getInputs().size() > gcOpInfo.maxInputTensors)
        {
            LOG_ERR(OP_VALIDATOR,
                    "Tested input tensor num {} > max expected tensor num {}",
                    opValCtx.getInputs().size(),
                    gcOpInfo.maxInputTensors);
            return ValidationResult::INCOMPATIBLE_INPUT_COUNT;
        }
    }

    if (!gcOpInfo.isVaryingNumOutput)
    {
        if (opValCtx.getOutputs().size() > gcOpInfo.maxOutputTensors)
        {
            LOG_ERR(OP_VALIDATOR,
                    "Tested output tensor num {} > max expected tensor num {}",
                    opValCtx.getOutputs().size(),
                    gcOpInfo.maxOutputTensors);
            return ValidationResult::INCOMPATIBLE_OUTPUT_COUNT;
        }
    }

    return ValidationResult::SUCCESS;
}

ValidationResult OpValidator::validateCustom(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx)
{
    bool valid = true;
    if (gcOpInfo.optionalCustomValidation.has_value())
    {
        valid = gcOpInfo.optionalCustomValidation.value()(gcOpInfo, opValCtx);
    }
    return valid ? ValidationResult::SUCCESS : ValidationResult::FAILURE;
}

static std::string_view toString(ValidationResult status)
{
    switch (status)
    {
        case ValidationResult::SUCCESS:
            return "SUCCESS";
        case ValidationResult::GUID_NOT_FOUND:
            return "GUID_NOT_FOUND";
        case ValidationResult::INCOMPATIBLE_INPUT_COUNT:
            return "INCOMPATIBLE_INPUT_COUNT";
        case ValidationResult::INCOMPATIBLE_INPUT_DIMENSION:
            return "INCOMPATIBLE_INPUT_DIMENSION";
        case ValidationResult::INCOMPATIBLE_INPUT_SIZE:
            return "INCOMPATIBLE_INPUT_SIZE";
        case ValidationResult::INCOMPATIBLE_OUTPUT_COUNT:
            return "INCOMPATIBLE_OUTPUT_COUNT";
        case ValidationResult::INCOMPATIBLE_OUTPUT_DIMENSION:
            return "INCOMPATIBLE_OUTPUT_DIMENSION";
        case ValidationResult::INCOMPATIBLE_OUTPUT_SIZE:
            return "INCOMPATIBLE_OUTPUT_SIZE";
        case ValidationResult::INCOMPATIBLE_DATA_TYPE:
            return "INCOMPATIBLE_DATA_TYPE";
        case ValidationResult::FAILURE:
            return "FAILED";
        default:
            LOG_ERR(OP_VALIDATOR, "Encountered unknown gc ops retcode {}", status);
            return "UNDEFINED";
    }
}

const std::vector<TensorValidationContext>& OpValidationContext::getInputs() const
{
    return const_cast<const std::vector<TensorValidationContext>&>(const_cast<OpValidationContext*>(this)->getInputs());
}
const std::vector<TensorValidationContext>& OpValidationContext::getOutputs() const
{
    return const_cast<const std::vector<TensorValidationContext>&>(
        const_cast<OpValidationContext*>(this)->getOutputs());
}
std::vector<TensorValidationContext>& OpValidationContext::getInputs()
{
    return m_input;
}
std::vector<TensorValidationContext>& OpValidationContext::getOutputs()
{
    return m_output;
}

bool OpValidationContext::empty() const
{
    return (m_input.empty() && m_output.empty());
}

}  // namespace gc::ops