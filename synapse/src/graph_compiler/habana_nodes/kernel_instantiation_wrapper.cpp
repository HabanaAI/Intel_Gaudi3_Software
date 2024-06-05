#include "kernel_instantiation_wrapper.h"

#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "habana_global_conf.h"
#include "kernel_db.h"
#include "log_manager.h"
#include "tpc_node.h"

#include <cstring>
#include <string_view>

const int MAX_TPC_INSTANTIATE_RETRIES = 5;

static inline unsigned getGCParamsNumUsedTensorEntries(const Tensor& gcTensor)
{
    return div_round_up(gcTensor.getDim(), SYN_MAX_TENSOR_DIM);
}

// Returns the number of tensors used
static uint32_t createUserTensorFromGCTensor(tpc_lib_api::Tensor*   userTensor,
                                             const TensorPtr&       gcTensor,
                                             const NSizeArray&      sizes,
                                             const NSizeArray&      minSizes,
                                             const gc::Permutation& perm,
                                             bool                   setReducible)
{
    // Additional dimensions are handled as if they are additional tensors
    const int      usedTensors   = getGCParamsNumUsedTensorEntries(*gcTensor);
    const auto     dataType      = toTpcLibDataType(gcTensor->getElementType());
    const auto&    quantParams   = gcTensor->getQuantizationParams().getChannelParams();
    const bool     reducible     = setReducible || gcTensor->isReductionEnabled() || gcTensor->isPartOfRMWSection();
    const auto&    permValues    = perm.getValues();
    const bool     isIdentity    = perm.isIdentity();
    const bool     isShapeTensor = gcTensor->isShapeTensor();
    const void*    pData         = nullptr;
    const unsigned tensorDims    = gcTensor->getDim();
    if (!isShapeTensor)
    {
        if (gcTensor->isStaticParam()) pData = gcTensor->getAddress();
        else if (gcTensor->hasHostData())
            pData = gcTensor->getHostMaxData();
    }

    unsigned tensorEntryDimOffset = 0;
    for (int i = 0; i < usedTensors; i++)
    {
        userTensor[i].geometry.dataType = dataType;
        userTensor[i].geometry.dims = std::min<unsigned>(tensorDims - tensorEntryDimOffset, SYN_MAX_TENSOR_DIM);

        // Min sizes will be equal to the sizes in case of static tensor.
        for (unsigned dimOffset = 0; dimOffset < SYN_MAX_TENSOR_DIM; ++dimOffset)
        {
            unsigned dim = dimOffset + tensorEntryDimOffset;
            if (dim >= tensorDims)
            {
                userTensor[i].geometry.maxSizes[dimOffset] = 1;
                userTensor[i].geometry.minSizes[dimOffset] = 1;
            }
            else
            {
                userTensor[i].geometry.maxSizes[dimOffset] = sizes[dim];
                userTensor[i].geometry.minSizes[dimOffset] = minSizes[dim];
            }
            if (isIdentity || dim >= tensorDims)  // treat dims outside the tensor size as identity as well
            {
                userTensor[i].permutation[dimOffset] = dimOffset;
            }
            else
            {
                userTensor[i].permutation[dimOffset] = std::min<unsigned>(permValues[dim], SYN_MAX_TENSOR_DIM);
            }
        }
        tensorEntryDimOffset += SYN_MAX_TENSOR_DIM;
        if (isShapeTensor) continue;

        userTensor[i].quantizationParam.scale     = quantParams.scale;
        userTensor[i].quantizationParam.zeroPoint = quantParams.zp;
        if (gcTensor->getElementType() == syn_type_fp8_143)
        {
            userTensor[i].quantizationParam.fp8bias = quantParams.expBias;
        }
        userTensor[i].Reducible                   = reducible;
        userTensor[i].pData                       = pData;
    }
    return usedTensors;
}

static inline uint32_t createUserTensorFromGCTensor(tpc_lib_api::Tensor*   userTensor,
                                                    const TensorPtr&       gcTensor,
                                                    const gc::Permutation& perm,
                                                    bool                   setReducible)
{
    // avoid copy if type width is not 4 bit
    if (!gcTensor->isType4Bit())
    {
        const NSizeArray& sizes    = gcTensor->getAllNSizesInElements();
        const NSizeArray& minSizes = gcTensor->getNMinimalSizesInElements();
        return createUserTensorFromGCTensor(userTensor,
                                            gcTensor,
                                            sizes,
                                            minSizes,
                                            perm,
                                            setReducible);
    }
    NSizeArray sizes;
    NSizeArray minSizes;
    gcTensor->getAllSizesInElementsCondensed(sizes);
    gcTensor->getAllMinimalSizesInElementsCondensed(minSizes);
    return createUserTensorFromGCTensor(userTensor,
                                        gcTensor,
                                        sizes,
                                        minSizes,
                                        perm,
                                        setReducible);
}

static std::string_view ParseTPCReturnValue(TpcElfTools::TpcElfStatus ret)
{
#if MAGIC_ENUM_SUPPORTED
    return magic_enum::enum_name(ret);
#else
    switch (ret)
    {
        TRANSLATE_ENUM_TO_STRING(TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS)
        TRANSLATE_ENUM_TO_STRING(TpcElfTools::TpcElfStatus::TPC_ELF_INVALID_ELF_BUFFER)
        TRANSLATE_ENUM_TO_STRING(TpcElfTools::TpcElfStatus::TPC_ELF_SECTION_NOT_FOUND)
        TRANSLATE_ENUM_TO_STRING(TpcElfTools::TpcElfStatus::TPC_ELF_UNSUPPORTED)

        default:
            return "UNKNOWN RETURN VALUE";
    }
#endif
}

KernelInstantiationWrapper::KernelInstantiationWrapper()
{
    m_instance.auxiliaryTensors          = m_auxTensors.data();
    m_instance.auxiliaryTensorNr         = m_auxBuffers.size();
    for (int i = 0; i < m_auxBuffers.size(); ++i)
    {
        m_instance.auxiliaryTensors[i].pData = m_auxBuffers[i].buffer.get();
        m_instance.auxiliaryTensors[i].bufferSize = m_auxBuffers[i].size;
    }
}

KernelInstantiationWrapper::KernelInstantiationWrapper(const KernelInstantiationWrapper& other)
: m_kernelBinary(other.m_kernelBinary),
  m_kernelElf(other.m_validTpcHeader ? other.m_kernelElf : nullptr),
  m_kernelSize(other.m_kernelSize),
  m_instantiated(other.m_instantiated),
  m_validTpcHeader(other.m_validTpcHeader),
  m_instance(other.m_instance),
  m_glueParams(other.m_glueParams),
  m_programHeader(other.m_programHeader),
  m_auxTensors(other.m_auxTensors),
  m_auxBuffers(other.m_auxBuffers),
  m_tensorOperands(other.m_tensorOperands),
  m_tensorOperandsAccessPattern(other.m_tensorOperandsAccessPattern)
{
    m_glueParams.inputTensors            = m_tensorOperands.getInputs();
    m_glueParams.outputTensors           = m_tensorOperands.getOutputs();
    m_instance.inputTensorAccessPattern  = m_tensorOperandsAccessPattern.getInputs();
    m_instance.outputTensorAccessPattern = m_tensorOperandsAccessPattern.getOutputs();
    m_instance.auxiliaryTensors          = m_auxTensors.data();
}

void KernelInstantiationWrapper::setKernelBinary(void *kernelBinary, uint32_t binarySize)
{
    if (kernelBinary >= m_instance.kernel.kernelElf &&
        ((uint64_t)kernelBinary + binarySize) <= ((uint64_t)m_instance.kernel.kernelElf + m_instance.kernel.elfSize))
    {
        // binary is inside the elf, don't delete (empty deleter).
        m_kernelBinary.reset((char*)kernelBinary, [](char*) {});
    }
    else
    {
        m_kernelBinary.reset(new char[binarySize], ArrayDeletor<char>());
        memcpy(m_kernelBinary.get(), kernelBinary, binarySize);
    }
    m_kernelSize = binarySize;
}

void KernelInstantiationWrapper::setKernelElf(void* kernelElf, uint32_t elfSize)
{
    prepareElfBuffer(elfSize);
    memcpy(m_kernelElf.get(), kernelElf, elfSize);
    m_validTpcHeader = false;
}

const std::shared_ptr<char>& KernelInstantiationWrapper::getAuxTensor(size_t i)
{
    return m_auxBuffers[i].buffer;
}

void KernelInstantiationWrapper::deleteAuxTensorBuffer(size_t i)
{
    m_auxBuffers[i].buffer.reset();
}

void KernelInstantiationWrapper::initParams(const TPCNode& tpcNode, tpc_lib_api::DeviceId deviceId, bool setReducible)
{
    if (m_glueParamsInitialized) return;
    updateGlueCodeParamsAndTensorAccessPatternPointers(tpcNode);
    initParams(m_glueParams,
               tpcNode,
               deviceId,
               setReducible);
    m_glueParamsInitialized = true;
}

KernelInstantiationWrapper::TensorOperandCounts
KernelInstantiationWrapper::updateGlueCodeParamsTensorPointers(const TPCNode&                   tpcNode,
                                                               tpc_lib_api::HabanaKernelParams& glueParams,
                                                               TPCLibTensorOperandsVector&      tensorOperands)
{
    KernelInstantiationWrapper::TensorOperandCounts operandCounts = {};
    // Clear containers old content if exists
    tensorOperands.clear();

    for (const auto& tensor : tpcNode.getInputs())
    {
        if (tensor == nullptr) continue;
        if (tensor->isShapeTensor())
        {
            operandCounts.shapeTensorCount++;
        }
        // In essence we can skip here over both shape tensors and auxiliary tensors
        // (isTensorAuxOrShapeOutput method) as is being done at instantiation time.
        // But we have various place in the code which attempt to access entries for shape tensors.
        // And keeping a zero filled entry seems to make them work properly or at least hide additional
        // issues. So unlike the instantiation case we only skip over auxilary tensors.
        if (tensor->isAuxTensor()) continue;
        if (likely(tensor->getDim() > 0))
        {
            operandCounts.inputTensorCount += getGCParamsNumUsedTensorEntries(*tensor);
        }
        else
        {
            // protection against rouge tests creating zero dim tensors
            operandCounts.inputTensorCount += 1;
        }
    }

    for (const auto& tensor : tpcNode.getOutputs())
    {
        if (tensor == nullptr) continue;
        if (tensor->isShapeTensor())
        {
            operandCounts.shapeTensorCount++;
        }
        if (likely(tensor->getDim() > 0))
        {
            operandCounts.outputTensorCount += getGCParamsNumUsedTensorEntries(*tensor);
        }
        else
        {
            // protection against rouge tests creating zero dim tensors
            operandCounts.outputTensorCount += 1;
        }
    }

    if (unlikely(operandCounts.inputTensorCount == 0 && tpcNode.getGUID() == TPCNode::NOP_KERNEL_NAME))
    {
        operandCounts.inputTensorCount++;
    }

    // resize and initialize to 0s
    tensorOperands.resize(operandCounts.inputTensorCount, operandCounts.outputTensorCount);

    // update pointers
    glueParams.inputTensors   = tensorOperands.getInputs();
    glueParams.outputTensors  = tensorOperands.getOutputs();

    return operandCounts;
}

void KernelInstantiationWrapper::updateGlueCodeParamsAndTensorAccessPatternPointers(const TPCNode& tpcNode)
{
    // Clear containers old content if exists
    m_tensorOperandsAccessPattern.clear();

    KernelInstantiationWrapper::TensorOperandCounts operandCounts =
        updateGlueCodeParamsTensorPointers(tpcNode, m_glueParams, m_tensorOperands);

    // resize and initialize to 0s
    m_tensorOperandsAccessPattern.resize(operandCounts.inputTensorCount, operandCounts.outputTensorCount);

    // update pointers
    m_instance.inputTensorAccessPattern  = m_tensorOperandsAccessPattern.getInputs();
    m_instance.outputTensorAccessPattern = m_tensorOperandsAccessPattern.getOutputs();
}

void KernelInstantiationWrapper::updateGlueCodeParamsAndLayoutsPointers(
    const TPCNode&                        tpcNode,
    tpc_lib_api::HabanaKernelParams&      glueParams,
    TPCLibTensorOperandsVector&           tensorOperands,
    tpc_lib_api::NodeDataLayouts&         nodeLayouts,
    TPCLibTensorOperandsDataLayoutVector& operandTensorLayouts,
    TPCLibTensorDataLayoutVector&         shapeTensorLayouts)
{
    // Clear containers old content if exists
    operandTensorLayouts.clear();
    shapeTensorLayouts.clear();

    KernelInstantiationWrapper::TensorOperandCounts operandCounts =
        updateGlueCodeParamsTensorPointers(tpcNode, glueParams, tensorOperands);

    // resize and initialize to 0s
    operandTensorLayouts.resize(operandCounts.inputTensorCount, operandCounts.outputTensorCount);
    shapeTensorLayouts.resize(operandCounts.shapeTensorCount);

    // update pointers
    nodeLayouts.inputs       = operandTensorLayouts.getInputs();
    nodeLayouts.outputs      = operandTensorLayouts.getOutputs();
    nodeLayouts.shapeTensors = shapeTensorLayouts.data();
}

void KernelInstantiationWrapper::initParams(tpc_lib_api::HabanaKernelParams& outParams,
                                            const TPCNode&                   tpcNode,
                                            tpc_lib_api::DeviceId            deviceId,
                                            bool                             setReducible)
{
    // Prepare params for glue code call
    outParams.deviceId      = deviceId;
    outParams.debugFlags    = 0;
    outParams.apiVersion    = 1;
    outParams.inputTensorNr = 0;
    outParams.uniqueNodeId  = tpcNode.getId();

    switch (GCFG_DETERMINISTIC_MODE.value())
    {
        case 0:
            outParams.useDeterministic = false;
            break;
        case 1:
            outParams.useDeterministic = true;
            break;
        default:
            outParams.useDeterministic = tpcNode.getDeterministic();
            break;
    }

    const HalReader* reader = CompilationHalReader::getHalReader(true).get();
    bool             bReductionSupportedAnywhere = reader &&
                                                   reader->isDRAMReductionSupported() &&
                                                   reader->isSRAMReductionSupported();

    int tensorIndex          = 0;
    auto& annotation = tpcNode.getNodeAnnotation();
    for (const auto& tensor: tpcNode.getInputs())
    {
        if (tensor == nullptr) continue;

        // ignore aux tensors  and shape tensors (as they're not real inputs and they don't have inputPermutations),
        // relevant when being called after TPC::init
        // Shape tensors should be discarded if they describe the output sizes (In this case its used for shape inference only)
        // H2D tensors should be passed unless host only flag is set (also used for shape inference/patching).
        if (tensor->isTensorAuxOrShapeOutput()) continue;

        int usedTensors = createUserTensorFromGCTensor(&outParams.inputTensors[outParams.inputTensorNr],
                                                       tensor,
                                                       annotation.inputPermutations[tensorIndex++],
                                                       bReductionSupportedAnywhere);
        outParams.inputTensorNr += usedTensors;
    }
    tensorIndex = 0;

    outParams.outputTensorNr = 0;

    for (const auto& tensor : tpcNode.getOutputs())
    {
        if (tensor == nullptr) continue;

        bool tensorReducible = setReducible && tensor->getTotalSizeInBytes() <= GCFG_MAX_RMW_TENSOR_BYTES.value();

        int usedTensors = createUserTensorFromGCTensor(&outParams.outputTensors[outParams.outputTensorNr],
                                                       tensor,
                                                       gc::Permutation(),  // permutations are irrelevant for outputs
                                                       tensorReducible || bReductionSupportedAnywhere);
        for (int i = 0; i < usedTensors; i++)
        {
            outParams.outputTensors[outParams.outputTensorNr].pData = nullptr;
            outParams.outputTensorNr++;
        }
    }
    const StringWithHash& guidAndHash = tpcNode.getGUIDAndHash();
    const std::string&    guid        = guidAndHash.getKey();
    static constexpr std::string_view LOG_SIGMA_EXP_GUID("logsigmaexp_st2_i32");
    static constexpr std::string_view AVG_POOL_GUID("average_pooling_i8");

    if (unlikely(guid == LOG_SIGMA_EXP_GUID))
    {
        outParams.outputTensors[0].geometry.dims = 1;
    }
    else if (unlikely(guid == AVG_POOL_GUID))
    {
        uint32_t newFlags = 0;
        // Turns on the forceAsmVersion flag
        newFlags |= 1;
        outParams.debugFlags = newFlags;
    }
    else if (unlikely((guid == TPCNode::NOP_KERNEL_NAME) && (outParams.inputTensorNr == 0)))
    {
        outParams.inputTensorNr = 1;
        // TODO: remove when new entry points are implemented (not converting old to new)
        outParams.inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
    }
    // Init node name
    strncpy(outParams.guid.name, guid.c_str(), tpc_lib_api::MAX_NODE_NAME - 1);
    outParams.nodeParams.nodeParams     = tpcNode.getParams();
    outParams.nodeParams.nodeParamsSize = tpcNode.getParamsSize();

    // Set guid name hash
    outParams.guid.nameHash.Value = KernelDB::instance().getKernelHashByName(guidAndHash, deviceId);

    if (reader)
    {
        outParams.maxAvailableTpc = TPCNode::getMaxAvailableTpc(reader);
    }
    else
    {
        outParams.maxAvailableTpc = TPCNode::getMaxAvailableTpc(deviceId);
    }
}

bool KernelInstantiationWrapper::checkAndIncreaseAuxBuffer()
// check if any of the tensors is small and increase it
{
    bool sizeChanged = false;
    m_auxBuffers.resize(m_instance.auxiliaryTensorNr);
    HB_ASSERT(m_instance.auxiliaryTensorNr <= MAX_TENSOR_NR, "size mismatch");
    for (unsigned i = 0; i < m_instance.auxiliaryTensorNr; i++)
    {
        unsigned bufferSize = m_instance.auxiliaryTensors[i].bufferSize;
        if (bufferSize != m_auxBuffers[i].size)
        {
            m_auxBuffers[i].buffer.reset(new char[bufferSize], ArrayDeletor<char>());
            m_auxBuffers[i].size                 = bufferSize;
            m_instance.auxiliaryTensors[i].pData = m_auxBuffers[i].buffer.get();
            sizeChanged = true;
        }
    }
    return sizeChanged;
}

void KernelInstantiationWrapper::deleteUnneededAuxBuffers()
{
    if (m_auxBuffers.size() == m_instance.auxiliaryTensorNr) return;
    m_auxBuffers.resize(m_instance.auxiliaryTensorNr);
    for (unsigned i = m_instance.auxiliaryTensorNr; i < MAX_TENSOR_NR; i++)
    {
        m_instance.auxiliaryTensors[i].pData = nullptr;
        m_instance.auxiliaryTensors[i].bufferSize = 0;
    }
}

void KernelInstantiationWrapper::validateAuxTensorsSize()
{
    // Validate that the size of the aux tensor buffer reported by glue code matches the actual geometry/dataType.
    // This is to avoid a case where number of bytes is mixed up with number of elements
    for (unsigned i = 0; i < m_instance.auxiliaryTensorNr; i++)
    {
        uint64_t calculatedBufSize =
            getActualTensorSize<uint64_t>(m_instance.auxiliaryTensors[i].geometry.dims,
                                          m_instance.auxiliaryTensors[i].geometry.maxSizes,
                                          translateTensorDataType(m_instance.auxiliaryTensors[i].geometry.dataType));

        HB_ASSERT(m_instance.auxiliaryTensors[i].bufferSize == calculatedBufSize,
                  "Aux buff size does not match actual size");
    }
}

void KernelInstantiationWrapper::prepareElfBuffer(uint64_t bufferSize)
{
    m_instance.kernel.elfSize = bufferSize;
    m_kernelElf.reset(new char[m_instance.kernel.elfSize], ArrayDeletor<char>());
    m_instance.kernel.kernelElf = m_kernelElf.get();
}

void KernelInstantiationWrapper::resetKernelInstantiationParams()
{
    // save and restore all IN fields of the structure before memset
    unsigned givenElfSize                      = m_instance.kernel.elfSize;
    void*    givenElfPointer                   = m_instance.kernel.kernelElf;
    uint32_t auxTensorsNr                      = m_instance.auxiliaryTensorNr;
    void*    AuxTensorsPointers[MAX_TENSOR_NR] = {};
    unsigned AuxTensorsSizes[MAX_TENSOR_NR]    = {};

    for (unsigned i = 0; i < auxTensorsNr; i++)
    {
        AuxTensorsPointers[i] = m_instance.auxiliaryTensors[i].pData;
        AuxTensorsSizes[i]    = m_instance.auxiliaryTensors[i].bufferSize;
    }

    tpc_lib_api::TensorAccessPattern* inputTensorAccessPattern  = m_instance.inputTensorAccessPattern;
    tpc_lib_api::TensorAccessPattern* outputTensorAccessPattern = m_instance.outputTensorAccessPattern;

    memset(&m_instance, 0, sizeof(tpc_lib_api::HabanaKernelInstantiation));
    memset(m_auxTensors.data(), 0, sizeof(tpc_lib_api::AuxTensor) * auxTensorsNr);
    m_instance.kernel.elfSize            = givenElfSize;
    m_instance.kernel.kernelElf          = givenElfPointer;
    m_instance.auxiliaryTensorNr         = auxTensorsNr;
    m_instance.auxiliaryTensors          = m_auxTensors.data();
    m_instance.inputTensorAccessPattern  = inputTensorAccessPattern;
    m_instance.outputTensorAccessPattern = outputTensorAccessPattern;
    for (unsigned i = 0; i < auxTensorsNr; i++)
    {
        m_instance.auxiliaryTensors[i].pData      = AuxTensorsPointers[i];
        m_instance.auxiliaryTensors[i].bufferSize = AuxTensorsSizes[i];
    }
}

tpc_lib_api::GlueCodeReturn KernelInstantiationWrapper::instantiate(const StringWithHash& guidAndHash)
{
    // Since the glue code can fail for various reasons, we gonna call it several times - while handling previous error.
    // Currently there are only 3 errors that we can recover from - on all others we fail the kernel instantiation
    tpc_lib_api::GlueCodeReturn ret      = tpc_lib_api::GLUE_FAILED;
    int numTries = 0;
    while (ret != tpc_lib_api::GLUE_SUCCESS && numTries < MAX_TPC_INSTANTIATE_RETRIES)
    {
        if (numTries > 0)
        {
            // clear stale information from the previously failed instantiation
            resetKernelInstantiationParams();
        }

        ret = KernelDB::instance().GetKernelInstance(&m_glueParams, &m_instance, guidAndHash);

        switch(ret)
        {
            case tpc_lib_api::GLUE_SUCCESS:
            {
                break;
            }
            case tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE:
            {
                if (checkAndIncreaseAuxBuffer())
                {
                    numTries++;
                }
                else
                {
                    LOG_ERR(GC,
                            "TPC instantiation failed({}) - unable to increase aux buffers",
                            KernelDB::parseReturnValue(ret));
                    numTries = MAX_TPC_INSTANTIATE_RETRIES;
                }
                break;
            }
            case tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER:
            {
                // Need to allocate a larger buffer and retry
                prepareElfBuffer(m_instance.kernel.elfSize);
                numTries++;
                break;
            }
            default:
            {
                numTries = MAX_TPC_INSTANTIATE_RETRIES;
                break;
            }
        }
    }

    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        std::string kernelName(m_glueParams.guid.name);
        LOG_WARN(GC, "{} Glue code returned {}", kernelName, KernelDB::parseReturnValue(ret));
        m_instance.auxiliaryTensorNr = 0;
    }
    deleteUnneededAuxBuffers();

    validateAuxTensorsSize();

    return ret;

}

TpcElfTools::TpcElfStatus KernelInstantiationWrapper::extractElf(const std::string& name)
{
    if (m_validTpcHeader)
    {
        return TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS;
    }
    void*    kernelBinary;
    uint32_t binarySize;
    TpcElfTools::TpcElfStatus result = TpcElfTools::ExtractTpcBinaryAndHeaderFromElf(m_instance.kernel.kernelElf,
                                                                                     m_instance.kernel.elfSize,
                                                                                     kernelBinary,
                                                                                     binarySize,
                                                                                     m_programHeader);
    if (result != TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS)
    {
        LOG_ERR(GC, "{} Failed to extract binary from Elf {}", name, ParseTPCReturnValue(result));
    }
    else
    {
        LOG_INFO(GC, "{}: extracted binary from Elf: {} , Size {}", name, ((void *) kernelBinary), binarySize);
        // Since the returned kernelBinary is just a pointer to the ELF, we deep copy it
        setKernelBinary(kernelBinary, binarySize);

        m_validTpcHeader = true;
    }
    return result;
}
