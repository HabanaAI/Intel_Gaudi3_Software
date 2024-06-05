#include "tpc_node.h"

// synapse graph_compiler
#include "access_pattern.h"
#include "graph_traits.h"
#include "habana_global_conf.h"         // for GCFG_TPC_PRINTF_TENSOR_SIZE
#include "habana_graph.h"
#include "habana_nodes/node_visitor.h"
#include "kernel_db.h"
#include "node.h"
#include "node_io_manager.h"            // for NodeIOManagerFactory
#include "synapse_common_types.h"
#include "tpc_kernel_loader.h"          // for deviceTypeToDeviceID
#include "types.h"
#include "utils.h"                      // for TRANSLATE_ENUM_TO_STRING, UNUSED
#include "tpc_kernel_names.h"           // for getBNxGuid
#include "stitchable_tpc_kernels.h"
#include "compilation_hal_reader.h"
#include "tpc_slice.h"
#include "access_pattern_generator.h"
#include "node_factory.h"

// synapse
#include "infra/log_manager.h"          // for GC
#include "allocators_utils.h"
#include "llvm/small_vector.h"

#include <algorithm>
#include <bitset>
#include <optional>
#include <utility>
#include <vector>
#include "register_memory_coherence.h"
#include "physical_memory_ops_nodes.h"

using namespace gc;

constexpr std::string_view ASSERT_ASYNC_KERNEL_NAME = "assert_async";

constexpr std::array<std::string_view, 20> m_broadcastableKernels = {
    "add", "and",  "div", "equal", "greater", "greater_equal", "less",  "less_equal", "max", "mean",
    "min", "mult", "or",  "xor",   "pow",     "sub",           "prelu", "where",      "mod", "binary_with_alpha"};

const std::set<std::string_view> TPCNode::m_seperableBlockList = {"logsoftmax_fwd",
                                                                  "one_hot",
                                                                  "resize_fwd",
                                                                  "batch_norm_fwd",
                                                                  "batch_norm_bwd"};

const std::set<std::string_view> TPCNode::m_stridedBlockList = {"lowering_pack_2_w77_s22"};

// remove once [SW-108990] is completed
const std::set<std::string_view> TPCNode::m_lfsrRandomGuidList = {"random_normal_fwd",
                                                                  "random_normal",
                                                                  "random_bernoulli_fwd",
                                                                  "random_bernoulli",
                                                                  "random_multinomial_fwd",
                                                                  "random_multinomial",
                                                                  "random_exponential_fwd",
                                                                  "random_exponential",
                                                                  "random_negative_binomial_fwd",
                                                                  "random_negative_binomial",
                                                                  "random_gamma_fwd",
                                                                  "random_gamma",
                                                                  "random_uniform_fwd",
                                                                  "random_uniform",
                                                                  "random_poisson_fwd",
                                                                  "random_poisson",
                                                                  "xavier_fill_fwd",
                                                                  "dropout_fwd",
                                                                  "philox_random_uniform_fwd",
                                                                  "philox_random_uniform"};

thread_local unsigned TPCNode::s_printfAllocatedSize = 0;

static bool isSizeCoveredByIndexSpace(const TSize*                            tensorSize,
                                      unsigned int                            dimNum,
                                      const uint64_t*                         indexSpaceGeometry,
                                      const tpc_lib_api::TensorAccessPattern& accessPattern)
{
    if (accessPattern.allRequired)
        return true;

    for (unsigned int dim = 0; dim < dimNum; ++dim)
    {
        const auto& tran = accessPattern.mapping[dim];
        if (tran.start_b > 0)
        {
            return false;
        }
        if (((double)tran.a * (indexSpaceGeometry[tran.indexSpaceDim] - 1) + tran.end_b + 1) < tensorSize[dim])
        {
            return false;
        }
    }
    return true;
}

static inline void addAssertAsyncOutput(TPCNode& tpc)
{
    if (startsWith(tpc.getGUID(), ASSERT_ASYNC_KERNEL_NAME))
    {
        // add output tensor to assert async node for msg id and node id
        const TSize sizes[] = {2, 0, 0, 0, 0};
        TensorPtr   output  = std::make_shared<Tensor>(1, sizes, syn_type_uint32);
        output->setMemorySectionID(MEMORY_ID_RESERVED_FOR_ASSERT_ASYNC);
        output->setName(tpc.getNodeName() + "_msg_id");

        tpc.addOutput(output);

        // insert node id before message id
        UserParams          params  = tpc.getParams();
        uint32_t*           node_id = (uint32_t*)params;
        uint32_t*           msg_id  = (uint32_t*)params + 1;
        *msg_id                     = *node_id;
        *node_id                    = (uint32_t)tpc.getId();
    }
}

// When a tensor has more than MAX_TENSOR_DIM, it is spread across multiple Tensor Descriptors,
// Thus, multiple access pattern entries in the access pattern arrays.
// Only one of these access pattern entries contain the actual access pattern base on index, and it's marked with
// allRequired=false. getAccessPatternDescIdx function finds the index of that entry.
struct AccessPatternDescIdxInfo
{
    unsigned descIdx  = 0;
    unsigned numDescs = 1;
    unsigned baseDim  = 0;
};

static unsigned getFirstDescIdx(unsigned tensorIdx, const TensorVector& tensors)
{
    // calculate the first descriptor index corresponding to tensorIdx
    unsigned firstDescriptorIdx = 0;
    for (unsigned i = 0; i < tensorIdx; i++)
    {
        // for each previous tensor, calculate how many descriptors it has, and add them up.
        if (!tensors[i]) continue;
        firstDescriptorIdx += TPCNode::numTensorGlueCodeAccessPatternEntries(tensors[i]);
    }
    return firstDescriptorIdx;
}

static AccessPatternDescIdxInfo getAccessPatternDescIdx(const Tensor&                           tensor,
                                                        const tpc_lib_api::TensorAccessPattern* tensorAccessPatterns,
                                                        unsigned                                firstDescriptorIdx)
{
    AccessPatternDescIdxInfo result;
    result.descIdx = firstDescriptorIdx;
    // find the descriptor that has an access pattern that isn't marked "allRequired"
    if (tensor.getDim() > SYN_MAX_TENSOR_DIM)
    {
        result.numDescs = TPCNode::numTensorDescriptors(tensor);
        bool entryFound = false;
        for (unsigned descIdx = firstDescriptorIdx; descIdx < firstDescriptorIdx + result.numDescs; descIdx++)
        {
            if (!tensorAccessPatterns[descIdx].allRequired)
            {
                HB_ASSERT(!entryFound, "Exactly one desc should be mapped to the index space.");
                result.descIdx = descIdx;
                result.baseDim = (descIdx - firstDescriptorIdx) * SYN_MAX_TENSOR_DIM;
                entryFound     = true;
            }
        }
    }
    return result;
}

static AccessPatternDescIdxInfo getAccessPatternDescIdx(unsigned                                tensorIdx,
                                                        const TensorVector&                     tensors,
                                                        const tpc_lib_api::TensorAccessPattern* tensorAccessPatterns,
                                                        std::optional<unsigned> firstDescriptorIdx = std::nullopt)
{
    if (!firstDescriptorIdx.has_value())
    {
        firstDescriptorIdx.emplace(getFirstDescIdx(tensorIdx, tensors));
    }
    const TensorPtr& gcTensor = tensors[tensorIdx];
    HB_ASSERT_PTR(gcTensor);
    return getAccessPatternDescIdx(*gcTensor, tensorAccessPatterns, *firstDescriptorIdx);
}

HabanaDeviceType TPCNode::getNodeDeviceType() const
{
    return DEVICE_TPC;
}

bool TPCNode::isSpecialFunctionsUsed() const
{
    if (!m_instanceWrapper.isValidElfProgramHeader())
    {
        LOG_ERR(TPC_NODE, "TPC ELF header is not valid");
        return false;
    }
    return m_instanceWrapper.getElfProgramHeader().specialFunctionUsed;
}

bool TPCNode::isSmallVLMRequired() const
{
    auto deviceId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());

    HB_ASSERT(m_instanceWrapper.isValidElfProgramHeader(),"{} TPC ELF header is not valid",m_name);

    if (!isSpecialFunctionsUsed()) return false;

    auto supportedFeature = KernelDB::instance().getSupportedFeatures();

    if (!supportedFeature.isFeatureSupported(deviceId, KernelDB::Feature::SMALL_VLM)) return true;

    if (m_instanceWrapper.getElfProgramHeader().version < KernelDB::MIN_SUPPORTED_VERSION_SMALL_VLM) return true;

    auto smallVlmRequired = !(m_instanceWrapper.getElfProgramHeader().unsetSmallVLM);
    LOG_INFO(TPC_NODE, "Kernel {} requires SmallVLM {}", m_name, smallVlmRequired);
    return smallVlmRequired;
}

bool TPCNode::isPrintfUsed() const
{
    if (!m_instanceWrapper.isValidElfProgramHeader())
    {
        LOG_ERR(TPC_NODE, "TPC ELF header is not valid");
        return false;
    }
    return m_instanceWrapper.getElfProgramHeader().printfUsed;
}

bool TPCNode::is44bitMode() const
{
    if (!m_instanceWrapper.isValidElfProgramHeader())
    {
        LOG_ERR(TPC_NODE, "TPC ELF header is not valid");
        return false;
    }
    return !(m_instanceWrapper.getElfProgramHeader().irf32Mode);
}

bool TPCNode::isAccessingSharedMem() const
{
    if (!m_instanceWrapper.isValidElfProgramHeader())
    {
        LOG_ERR(TPC_NODE, "TPC ELF header is not valid");
        return false;
    }
    return m_instanceWrapper.getElfProgramHeader().mmioUse;
}

uint16_t TPCNode::getRmwOutputMask(tpc_lib_api::DeviceId deviceId) const
{
    KernelInstantiationWrapper instance;
    if (getInfoInstance(instance, deviceId, true /*extract elf*/, true /*setReducible*/))
    {
        const auto RMW_STORE_SIZE_BYTES = sizeof(instance.getElfProgramHeader().rmwStore);
        std::bitset<BITS_PER_BYTE * RMW_STORE_SIZE_BYTES> rmwStoreMaskElf = instance.getElfProgramHeader().rmwStore;
        std::bitset<BITS_PER_BYTE * RMW_STORE_SIZE_BYTES> rmwStoreMaskUser = rmwStoreMaskElf;
        if (rmwStoreMaskElf.any())
        {
            //Clear mask depending on actual kernel rmw usage
            const auto& kernelInstantiationParams = instance.getInstance();
            for (int i = 0; i < m_inputs.size(); ++i)
            {
                if (kernelInstantiationParams.inputTensorAccessPattern[i].noRmwAccess)
                {
                    rmwStoreMaskUser.reset(i);
                }
            }

            int numInputs = m_inputs.size();
            for (int i = 0; i < m_outputs.size(); ++i)
            {
                if (kernelInstantiationParams.outputTensorAccessPattern[i].noRmwAccess)
                {
                    rmwStoreMaskUser.reset(numInputs + i);
                }
            }
        }

        if (rmwStoreMaskElf != rmwStoreMaskUser)
        {
            LOG_DEBUG(GC, "RMW mask for node {}, mask reported by ELF header: 0x{:x}, mask after clearing bits reported by user TensorAccessPattern: 0x{:x}",
                      getNodeName(),
                      rmwStoreMaskElf.to_ulong(),
                      rmwStoreMaskUser.to_ulong());
        }

        return (uint16_t)rmwStoreMaskUser.to_ulong();
    }
    // In case elf header is not valid, we take no reduction as default
    return 0;
}

unsigned TPCNode::getLlvmTensorIdFromMap(unsigned tensorIdx, uint64_t dupTensorsMap)
{
    constexpr auto TENSOR_ID_SIZE_BITS = 4;
    auto           tensorId            = (dupTensorsMap >> (tensorIdx * TENSOR_ID_SIZE_BITS) & 0xf);
    return tensorId;
}

std::map<unsigned, unsigned> TPCNode::getDuplicateTensorsFromElf(void*& elf, unsigned int elfSize)
{
    TpcElfTools::TPCProgramHeader programHeader;
    auto                          status = TpcElfTools::ExtractTpcProgramHeaderFromElf(elf, elfSize, programHeader);
    std::map<unsigned, unsigned>  origIdToDupId;
    if (status == TpcElfTools::TPC_ELF_SUCCESS)
    {
        uint64_t dupTensorsMap       = programHeader.duplicateTensors;
        for (unsigned i = 0; i < MAX_TENSOR_NR; i++)
        {
            auto id    = getLlvmTensorIdFromMap(i, dupTensorsMap);
            auto dupId = getLlvmTensorIdFromMap(++i, dupTensorsMap);
            if (id == dupId) break;
            origIdToDupId.insert({id, dupId});
        }
        return origIdToDupId;
    }
    HB_ASSERT(0, "Failed to extract program header from elf");
    return origIdToDupId;
}

bool TPCNode::isOutputTensorRmw(unsigned tensorOutputIndex, tpc_lib_api::DeviceId deviceId) const
{
    std::bitset<BITS_PER_BYTE * sizeof(uint16_t)> rmwMask      = getRmwOutputMask(deviceId);
    size_t                                        outputBitPos = getNumInputsToKernel() + tensorOutputIndex;
    return rmwMask.test(outputBitPos);
}

bool TPCNode::isOutputTensorAllRequired(unsigned tensorOutputIndex) const
{
    KernelInstantiationWrapper instance;
    auto deviceId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());
    if (getInfoInstance(instance,
                        deviceId,
                        true /*extract elf*/,
                        true /*setReducible*/))
    {
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(tensorOutputIndex, getOutputs(), instance.getInstance().outputTensorAccessPattern);
        return instance.getInstance().outputTensorAccessPattern[info.descIdx].allRequired;
    }
    return false;
}

bool TPCNode::isOutputTensorFullyWritten(unsigned tensorOutputIndex) const
{
    KernelInstantiationWrapper instance;
    auto                       deviceId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());
    if (getInfoInstance(instance, deviceId, true /*extract elf*/, true /*setReducible*/))
    {
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(tensorOutputIndex, getOutputs(), instance.getInstance().outputTensorAccessPattern);
        return instance.getInstance().outputTensorAccessPattern[info.descIdx].fullyWritten;
    }
    return false;
}

bool TPCNode::isOutputTensorPartialWrites(unsigned tensorOutputIndex) const
{
    KernelInstantiationWrapper instance;
    auto deviceId = deviceTypeToDeviceID(CompilationHalReader::getHalReader()->getDeviceType());
    if (getInfoInstance(instance,
                        deviceId,
                        true /*extract elf*/,
                        true /*setReducible*/))
    {
        std::bitset<BITS_PER_BYTE * sizeof(uint16_t)> partialStore = instance.getElfProgramHeader().partialStore;

        // index of the bit represents output index within all node tensors
        size_t outputBitPos = getNumInputsToKernel() + tensorOutputIndex;
        return partialStore.test(outputBitPos);
    }
    return false;
}

bool TPCNode::isOutputTensorMemset(unsigned tensorOutputIndex, tpc_lib_api::DeviceId deviceId) const
{
    KernelInstantiationWrapper instance;
    if (getInfoInstance(instance,
                        deviceId,
                        true /*extract elf*/,
                        true /*setReducible*/))
    {
        const auto& kernelInstantiationParams = instance.getInstance();
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(tensorOutputIndex, getOutputs(), instance.getInstance().outputTensorAccessPattern);
        return kernelInstantiationParams.outputTensorAccessPattern[info.descIdx].memsetBeforeExecution;
    }
    return false;
}

TPCNode::TPCNode(const TensorVector& inputs,
                 const TensorVector& outputs,
                 std::string_view    name,
                 UserParams          params /*=nullptr*/,
                 unsigned            paramsSize /*=0*/)
: GenericParametersNode(inputs, outputs, name, TYPE_USER, params, paramsSize, false),
  MultiSifNodeInfoHelper(),
  m_optimized(false),
  m_uniqueID(0),
  m_printfTensor(nullptr),
  m_shapeInferenceFunctionVersion(0)
{
    m_io = std::make_unique<TPCNodeIOManager>(this);
}

NodePtr TPCNode::createNode(const TensorVector& inputs,
                            const TensorVector& outputs,
                            UserParams          userParams,
                            unsigned            userParamsSize,
                            std::string_view    guid,
                            std::string_view    name)
{
    auto tpcNode = std::make_shared<TPCNode>(inputs, outputs, name, userParams, userParamsSize);
    tpcNode->setGUID(guid);
    addAssertAsyncOutput(*tpcNode);
    return tpcNode;
}

TPCNode::TPCNode(const TPCNode& other)
: GenericParametersNode(other),
  m_instanceWrapper(other.m_instanceWrapper),
  m_optimized(other.m_optimized),
  m_uniqueID(other.m_uniqueID),
  m_printfTensor(nullptr),
  m_shapeInferenceFunctionVersion(other.m_shapeInferenceFunctionVersion),
  m_dynamicShapeProjectionTensors(other.m_dynamicShapeProjectionTensors)
{
    m_GUID = other.m_GUID;

    // Init the printfTesnor - if needed
    if (other.m_printfTensor != nullptr)
    {
        initPrintfTensor();
    }
    // Make a copy of m_multiSifNodeInfo to avoid all the slices pointing to the same struct
    if (other.m_multiSifNodeInfo)
    {
        m_multiSifNodeInfo = std::make_shared<MultiSifNodeInfo>(*other.m_multiSifNodeInfo);
    }
    m_io = std::make_unique<TPCNodeIOManager>(this);
}

TPCNode::~TPCNode() = default;

TPCNode& TPCNode::operator=(const TPCNode& other)
{
    if (this != &other)
    {
        GenericParametersNode::operator=(other);
        m_instanceWrapper              = other.m_instanceWrapper;

        m_optimized = other.m_optimized;
        m_uniqueID  = other.m_uniqueID;
        m_GUID      = other.m_GUID;

        m_shapeInferenceFunctionVersion = other.m_shapeInferenceFunctionVersion;
        m_dynamicShapeProjectionTensors = other.m_dynamicShapeProjectionTensors;
        // Make a copy of m_multiSifNodeInfo to avoid all the slices pointing to the same struct
        if (other.m_multiSifNodeInfo)
        {
            m_multiSifNodeInfo = std::make_shared<MultiSifNodeInfo>(*other.m_multiSifNodeInfo);
        }
        m_io                 = std::make_unique<TPCNodeIOManager>(this);
        m_printfTensor       = other.m_printfTensor;
    }
    return *this;
}

void TPCNode::validateTensorsIndexSpace(const TensorVector&                     tensors,
                                        const tpc_lib_api::TensorAccessPattern* accessPattern) const
{
    for (unsigned int tensorIdx = 0; tensorIdx < tensors.size(); ++tensorIdx)
    {
        if (tensors[tensorIdx]->isAuxTensor()) continue;
        const NSizeArray& tensorSize = tensors[tensorIdx]->getAllNSizesInElements();
        if (!isSizeCoveredByIndexSpace(tensorSize.data(),
                                       tensors[tensorIdx]->getDim(),
                                       m_instanceWrapper.getInstance().indexSpaceGeometry,
                                       accessPattern[tensorIdx]))
        {
            LOG_WARN(TPC_NODE,
                     "Node: {} Kernel {} index space geometry doesn't cover sizes [{}]",
                     getNodeName(),
                     getGUID(),
                     fmt::join(&tensorSize[0], &tensorSize[0] + tensors[tensorIdx]->getDim(), ","));
        }
    }
}

NodePtr TPCNode::clone() const
{
    TPCNodePtr clonedNode(new TPCNode(*this));

    if (m_instanceWrapper.isInstantiated())
    {
        if (m_printfTensor != nullptr)
        {
            clonedNode->initPrintfTensor();
        }
    }

    return clonedNode;
}

NodePtr TPCNode::getSlice() const
{
    auto slice = std::make_shared<TPCSlice>(this);
    return slice;
}

std::string TPCNode::getUpgradedGUID(const std::string& guid)
{
    std::string newGUID = guid;
    std::string guidWithoutDtype = std::string(extractGUIDFromFullGUID(guid));
    bool kernelExist = false;
    while (!kernelExist && !newGUID.empty())
    {
        std::string_view suffix = extractDtypeFromGUID(newGUID);
        LOG_WARN(TPC_NODE, "TPC kernel {} not found, trying to upgrade {}", newGUID, guid);
        if (suffix == "i4" || suffix == "u4")
        {
            newGUID = guidWithoutDtype + "_hf8";
        }
        else if (suffix == "hf8")
        {
            newGUID = guidWithoutDtype + "_f8";
        }
        else if (suffix == "f8")
        {
            newGUID = guidWithoutDtype + "_i8";
        }
        else if (suffix == "i8" || suffix == "u8")
        {
            newGUID = guidWithoutDtype + "_i16";
        }
        else if (suffix == "i16" || suffix == "u16" || suffix == "f16" || suffix == "bf16")
        {
            newGUID = guidWithoutDtype + "_f32";
        }
        else if (suffix == "u32")
        {
            newGUID = guidWithoutDtype + "_i32";
        }
        else
        {
            newGUID = "";
        }
        kernelExist = KernelDB::instance().isKernelExist(newGUID, getGraphTraits()->getDeviceId());
    }
    return newGUID;
}


std::string_view TPCNode::getGUIDWithoutDtype() const
{
    const std::string& guid = getGUID();
    std::string_view newGUID = extractGUIDFromFullGUID(guid);
    if (newGUID.empty())
    {
        if (!guid.empty())
        {
            LOG_ERR(TPC_NODE, "GUID of node {} starts with a data type name: {}", getNodeName(), guid);
        }
        return guid;
    }
    return newGUID;
}

std::string TPCNode::getNodeTypeStr() const
{
    return getGUID();  // GUID is more informative than Type in TPC node
}

void TPCNode::accept(NodeVisitor* visitor)
{
    visitor->visit(this);
}

void TPCNode::registerKernelToCodeGen(const std::unique_ptr<CodeGenerator>& codeGen, const deviceAddrOffset& addr) const
{
    const auto& cachedBinary = m_instanceWrapper.getCacheKernel();
    if (cachedBinary == nullptr)
    {
        codeGen->registerTPCProgramDataBlobForDownload((char*)m_instanceWrapper.getKernelBinary(),
                                                       addr,
                                                       getKernelSize(),
                                                       getUniqueID());
        if (m_instanceWrapper.shouldCacheElf())
        {
            codeGen->cacheBlobBuffer(m_instanceWrapper.getElfBuffer());
        }
    }
    else
    {
        codeGen->registerTPCProgramDataBlobForDownload(cachedBinary,
                                                       addr,
                                                       getKernelSize(),
                                                       getUniqueID());
    }
}

unsigned TPCNode::getNumInputsToKernel() const
{
    unsigned n = 0;
    for (auto tensor : m_inputs)
    {
        if (tensor->isTensorAuxOrShapeOutput()) n++;
    }
    return m_inputs.size() - n;
}

NodeROI TPCNode::getWorkROI(const tpc_lib_api::TensorAccessPattern& accessPattern,
                            const NodeROI&                          roi,
                            const TensorPtr&                        tensorPtr,
                            unsigned                                startDim) const
{
    HB_ASSERT_PTR(tensorPtr);
    const Tensor& tensor = *tensorPtr;

    if (tensor.isAuxTensor() || accessPattern.allRequired)
    {
        return generateFullWorkROI(tensorPtr);
    }

    const Tensor* clipTensor = &tensor;
    NodeROI workRoi = generateInitialWorkROIForTensor(tensorPtr, roi);
    // Override the previous data for the appropriate dimensions, according to the index space
    for (unsigned dim = startDim; dim < std::min(startDim + SYN_MAX_TENSOR_DIM, tensor.getDim()); ++dim)
    {
        unsigned    accessDim        = dim - startDim;
        const auto& dimAccessPattern = accessPattern.mapping[accessDim];
        unsigned    mappedDim        = dimAccessPattern.indexSpaceDim;

        TOffset startPixel = (double)dimAccessPattern.a * roi.baseOffset[mappedDim] + dimAccessPattern.start_b +
                             getSliceOffset(tensorPtr, dim, dimAccessPattern);
        startPixel = clip(startPixel, (int64_t)0, (int64_t)clipTensor->getSizeInElements(dim));

        TOffset endPixel = (double)dimAccessPattern.a * (roi.baseOffset[mappedDim] + roi.size[mappedDim] - 1) +
                           dimAccessPattern.end_b + getSliceOffset(tensorPtr, dim, dimAccessPattern) + 1;
        endPixel = clip(endPixel, (int64_t)0, (int64_t)clipTensor->getSizeInElements(dim));

        workRoi.baseOffset[dim] = startPixel;
        workRoi.size[dim]       = endPixel - startPixel;
    }
    return workRoi;
}

NodeROI TPCNode::generateFullWorkROI(const TensorPtr& tensor)
{
    NodeROI ret;
    auto    elements = tensor->getNSizesInElements();
    std::copy(elements.begin(), elements.end(), ret.size);
    return ret;
}

NodeROI TPCNode::generateInitialWorkROIForTensor(const TensorPtr& tensor, const NodeROI& baseROI)
{
    NodeROI workROI(baseROI);
    // Initialize all baseOffsets and dim size as allRequired
    // note that the 'roi' parameter sizes are of dimension 5, and refer to the node's index space geometery.
    // the workRoi refers to the node's dimension and can have a larger dimension.
    for (unsigned dim = 0; dim < tensor->getDim(); dim++)
    {
        workROI.baseOffset[dim] = 0;
        workROI.size[dim]       = tensor->getSizeInElements(dim);
    }
    return workROI;
}

// Offset in elements for the slice (TPCSlice in gaudis)
TOffset TPCNode::getSliceOffset(const TensorPtr&                         tensor,
                                unsigned                                 dim,
                                const tpc_lib_api::DimIndexSpaceMapping& dimAccessPattern) const
{
    const SizeArray& sliceBase = getNodeAnnotation().baseOffset;
    return (double)dimAccessPattern.a * sliceBase[dimAccessPattern.indexSpaceDim];
}

gc::access_pattern::NodeAccessPatternPtr TPCNode::generateNodeAccessPattern() const
{
    HB_ASSERT(isInstantiated(), "Trying to get access pattern for uninstantiated TPC node: {}", getNodeName());

    // Note: caching is done in Node. No need to worry about it here.
    return gc::access_pattern::AccessPatternFromGlueCodeGenerator::generate(this);
}

NodeROI TPCNode::generateRoi() const
{
    NodeROI fullRoi;

    const auto& instance = getInstance();
    for (unsigned dim = 0; dim < instance.indexSpaceRank; ++dim)
    {
        fullRoi.size[dim] = instance.indexSpaceGeometry[dim];
    }
    std::fill(fullRoi.size + instance.indexSpaceRank, fullRoi.size + ARRAY_SIZE(fullRoi.size), 1ULL);

    return fullRoi;
}

Settable<NodeROI> TPCNode::getInputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    AccessPatternDescIdxInfo info =
        getAccessPatternDescIdx(tensorIdx, getInputs(), getInstance().inputTensorAccessPattern);
    const TensorPtr& input         = getInput(tensorIdx);
    HB_ASSERT_PTR(input);
    return getWorkROI(getInstance().inputTensorAccessPattern[info.descIdx], roi, input, info.baseDim);
}

Settable<NodeROI> TPCNode::getOutputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    AccessPatternDescIdxInfo info =
        getAccessPatternDescIdx(tensorIdx, getOutputs(), getInstance().outputTensorAccessPattern);
    const TensorPtr& output        = getOutput(tensorIdx);
    HB_ASSERT_PTR(output);
    return getWorkROI(getInstance().outputTensorAccessPattern[info.descIdx], roi, output, info.baseDim);
}

synDataType TPCNode::getRequiredInputType(uint32_t tensorIdx) const
{
    bool inference = getGraphTraits()->inferenceGraph();
    synDataType inputType = getInput(tensorIdx)->getElementType();
    if (inference && is8BitFloat(inputType) && !isCast())
    {
        // Currently tpc kernels does not support fp8 tensors.
        return getNodePrecision();
    }
    if (inference && !m_instanceWrapper.isInstantiated())
    {
        return inputType;
    }
    auto& glueParams = m_instanceWrapper.getGlueParams();
    if (tensorIdx >= glueParams.inputTensorNr &&
        (tensorIdx - glueParams.inputTensorNr) < m_instanceWrapper.getInstance().auxiliaryTensorNr)
    {
        return inputType;
    }
    HB_ASSERT(tensorIdx < glueParams.inputTensorNr,
              "Tensor index {} exceeds available tensors {}",
              tensorIdx,
              glueParams.inputTensorNr);
    return translateTensorDataType(glueParams.inputTensors[tensorIdx].geometry.dataType, syn_type_single);
}

synDataType TPCNode::getRequiredOutputType(uint32_t tensorIdx) const
{
    bool inference = getGraphTraits()->inferenceGraph();
    synDataType outputType = getOutput(tensorIdx)->getElementType();
    if (inference && is8BitFloat(outputType) && !isCast())
    {
        // Currently tpc kernels does not support fp8 tensors.
        return getNodePrecision();
    }
    if (inference && !m_instanceWrapper.isInstantiated())
    {
        return outputType;
    }
    auto& glueParams = m_instanceWrapper.getGlueParams();
    HB_ASSERT(tensorIdx < glueParams.outputTensorNr,
              "Tensor index {} exceeds avaialable tensors {}",
              tensorIdx,
              glueParams.outputTensorNr);
    return translateTensorDataType(glueParams.outputTensors[tensorIdx].geometry.dataType, syn_type_single);
}

void TPCNode::initPrintfTensor()
{
    if (isPrintfUsed())
    {
        unsigned newSize = s_printfAllocatedSize + GCFG_TPC_PRINTF_TENSOR_SIZE.value();

        if (newSize < GCFG_TPC_PRINTF_MAX_BUFFER_SIZE.value())
        {
            TSize size[2] = {1, 0};
            unsigned dim1Size = GCFG_TPC_PRINTF_TENSOR_SIZE.value() / 4;
            size[1] = dim1Size;
            m_printfTensor = std::make_shared<Tensor>(2U, size, syn_type_int32);
            s_printfAllocatedSize = newSize;
            if (m_instanceWrapper.isValidElfProgramHeader())
            {
                m_printfPosition = m_instanceWrapper.getElfProgramHeader().printTensorNum;
            }
        }
        else
        {
            LOG_ERR(TPC_NODE, "TPC printf tensors allocation size reached its limit: {}. Did not allocate new tensor",
                    GCFG_TPC_PRINTF_MAX_BUFFER_SIZE.value());
            m_printfTensor = nullptr;
        }
    }
    else
    {
        m_printfTensor = nullptr;
    }
}

uint64_t TPCNode::getShapeInferenceFunctionVersion() const
{
    return m_shapeInferenceFunctionVersion;
}

void TPCNode::setSparseAccessTensorsAnnotation()
{
    const auto& instance = m_instanceWrapper.getInstance();
    unsigned    currentTensorFirstDescriptorIdx = 0;
    for (unsigned tensorIdx = 0; tensorIdx < m_inputs.size(); ++tensorIdx)
    {
        const TensorPtr& input = m_inputs[tensorIdx];
        if (input->isShapeTensor() || input->isAuxTensor()) continue;
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(*input, instance.inputTensorAccessPattern, currentTensorFirstDescriptorIdx);
        // A tensor is marked as sparseAccess once at least one node accesses it sparsely
        input->getTensorAnnotation().sparseAccess |= instance.inputTensorAccessPattern[info.descIdx].sparseAccess;
        currentTensorFirstDescriptorIdx += info.numDescs;
    }

    currentTensorFirstDescriptorIdx = 0;
    for (unsigned tensorIdx = 0; tensorIdx < m_outputs.size(); ++tensorIdx)
    {
        const TensorPtr&         output = m_outputs[tensorIdx];
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(*output, instance.outputTensorAccessPattern, currentTensorFirstDescriptorIdx);
        // A tensor is marked as sparseAccess once at least one node accesses it sparsely
        output->getTensorAnnotation().sparseAccess |= instance.outputTensorAccessPattern[info.descIdx].sparseAccess;
        currentTensorFirstDescriptorIdx += info.numDescs;
    }
}

// This function sets the prefetch stride value per tensor, as part of the HW pre-fetcher configuration
void TPCNode::setTensorsPrefetchStride(const HalReader& reader)
{
    const TpcElfTools::TPCProgramHeader& programHeader  = m_instanceWrapper.getElfProgramHeader();
    // Nothing to do if there's no scalar loading
    if (programHeader.scalarLoad == 0) return;

    // first collect all tensors
    using TensorRawPtrVector = llvm_vecsmall::SmallVector<Tensor*, MAX_TENSOR_NR>;
    TensorRawPtrVector allTensors;
    auto               fillRawTensorPtrs = [&allTensors](const TensorVector& src) {
        std::transform(src.begin(), src.end(), std::back_inserter(allTensors), [](const TensorPtr& t) {
            return t.get();
        });
    };
    fillRawTensorPtrs(m_inputs);
    fillRawTensorPtrs(m_outputs);

    using TensorSizesInBytesVec = llvm_vecsmall::SmallVector<uint64_t, MAX_TENSOR_NR>;
    TensorSizesInBytesVec                tensorSizes;
    uint64_t                             tensorNr       = allTensors.size();
    unsigned                             tensorSizesSum = 0;

    const uint32_t cacheLineSize  = reader.getDCacheLineSize();
    const uint32_t cacheLineNr    = reader.getDCacheLineNr();
    const uint32_t cacheMaxStride = reader.getDCacheMaxStride();
    const uint32_t cacheMinStride = reader.getDCacheMinStride();

    uint32_t remTensorsToFill = 0;
    uint32_t remCacheLines    = cacheLineNr;

    for (int idx = 0; idx < tensorNr; idx++)
    {
        uint64_t sizeInBytes = allTensors[idx]->getTotalSizeInBytes();

        tensorSizes.push_back(sizeInBytes);

        if (isBitSelected(programHeader.scalarLoad, idx) && (sizeInBytes > cacheLineSize))
        {
            LOG_DEBUG(TPC_NODE, "scalarLoad is on for tensor: {}. Size: {}", allTensors[idx]->getName(), sizeInBytes);
            tensorSizesSum += sizeInBytes;
        }
    }

    if (tensorSizesSum == 0)
    {
        LOG_DEBUG(TPC_NODE, "scalarLoad is not being used in tpc");
        return;
    }

    using TensorPrefStride = llvm_vecsmall::SmallVector<uint32_t, MAX_TENSOR_NR>;
    TensorPrefStride tensorPrefStride;
    using TensorSizeInCacheLines = llvm_vecsmall::SmallVector<uint32_t, MAX_TENSOR_NR>;
    TensorSizeInCacheLines tensorSizeInCacheLines;

    // Divide D$ cache-lines between the tensors according to the sizes ratio.
    for (unsigned i = 0; i < tensorNr; i++)
    {
        uint32_t prefStride       = 0;
        uint32_t sizeInCacheLines = (tensorSizes[i] >> 7) + std::max(1, int(tensorSizes[i] / cacheLineSize));
        tensorSizeInCacheLines.push_back(sizeInCacheLines);
        if (isBitSelected(programHeader.scalarLoad, i) && (tensorSizes[i] > cacheLineSize))
        {
            prefStride = (tensorSizes[i] / tensorSizesSum) * (cacheLineNr / cacheMinStride) * cacheMinStride;
            prefStride = std::max(prefStride, cacheMinStride);
            prefStride = std::min(prefStride, cacheMaxStride);
            prefStride = std::min(prefStride, sizeInCacheLines);
            remCacheLines -= prefStride;
            if ((prefStride < cacheMaxStride) && (prefStride < sizeInCacheLines))
            {
                remTensorsToFill++;
            }
        }
        tensorPrefStride.push_back(prefStride);
    }

    // Divide the remaining cache lines between the tensors evenly
    uint64_t i = 0;
    while (remTensorsToFill && remCacheLines)
    {
        if (isBitSelected(programHeader.scalarLoad, i) && (tensorSizes[i] > cacheLineSize) &&
            (tensorPrefStride[i] < cacheMaxStride) && (tensorPrefStride[i] < tensorSizeInCacheLines[i]))
        {
            tensorPrefStride[i]++;
            remCacheLines--;
            if ((tensorPrefStride[i] == cacheMaxStride) || (tensorPrefStride[i] == tensorSizeInCacheLines[i]))
            {
                remTensorsToFill--;
            }
        }
        i = (i + 1) % tensorNr;
    }

    // Finally set the prefetch stride for each tensor
    for (i = 0; i < tensorNr; i++)
    {
        LOG_DEBUG(TPC_NODE,
                  "Assigning prefetch stride of: {} to tensor: {}",
                  tensorPrefStride[i],
                  allTensors[i]->getName());
        allTensors[i]->setPrefetchStride(tensorPrefStride[i]);
    }
}

TensorPtr TPCNode::getAuxTensor(const tpc_lib_api::AuxTensor* auxiliaryTensor,
                                const std::shared_ptr<char>&  data,
                                AuxiliaryTensors*             cachedAuxiliaryTensors)
{
    const unsigned int dims = auxiliaryTensor->geometry.dims;

    TSize tSizes[tpc_lib_api::MAX_TENSOR_DIM];
    castNcopy(tSizes, auxiliaryTensor->geometry.maxSizes, tpc_lib_api::MAX_TENSOR_DIM);

    synDataType dataType = translateTensorDataType(auxiliaryTensor->geometry.dataType);

    if (data == nullptr || cachedAuxiliaryTensors == nullptr)
    {
        return std::make_shared<Tensor>(dims, tSizes, dataType, data.get());
    }

    uint32_t            crc = crc32(data.get(), auxiliaryTensor->bufferSize);
    std::string         sizesStr = toString(tSizes, tSizes + dims, 'x');
    AuxiliaryTensorsKey key(dataType, crc, sizesStr);

    auto it = cachedAuxiliaryTensors->insert(std::make_pair(key, std::make_pair(nullptr, data)));
    if(it.second)
    {
        it.first->second.first = std::make_shared<Tensor>(dims, tSizes, dataType, static_cast<char*>(data.get()));
    }

    return it.first->second.first;
}

void TPCNode::createAuxTensors(AuxiliaryTensors* cachedAuxiliaryTensors)
{
    auto& instance = m_instanceWrapper.getInstance();

    // In case this node has been initialized before, remove existing aux tensor in case of incompatability with the
    // current initialization.
    auto iter = std::remove_if(m_inputs.begin(), m_inputs.end(), [](const TensorPtr& t) { return t->isAuxTensor(); });
    m_inputs.erase(iter, m_inputs.end());

    tpc_lib_api::AuxTensor* auxiliaryTensor = instance.auxiliaryTensors;
    for (unsigned auxIdx = 0; auxIdx < instance.auxiliaryTensorNr; auxIdx++)
    {
        bool  isScratchPad = auxiliaryTensor->noInit;
        char* pData = (char*)auxiliaryTensor->pData;
        if (isScratchPad)
        {
            pData = nullptr;
        }
        if (pData == nullptr)
        {
            m_instanceWrapper.deleteAuxTensorBuffer(auxIdx);
        }

        TensorPtr auxTensor =
            getAuxTensor(auxiliaryTensor, m_instanceWrapper.getAuxTensor(auxIdx), cachedAuxiliaryTensors);
        m_inputs.push_back(auxTensor);

        m_inputs.back()->setAsAuxTensor(isScratchPad);
        if (pData == nullptr)
        {
            uint64_t size = getWriteSpaceForTensor(auxTensor);
            if (size < GCFG_SCRATCHPAD_SRAM_MAX_SIZE.value())
            {
                m_inputs.back()->setTensorInSram();
                LOG_DEBUG(TPC_NODE, "Set auxTensor {} in SRAM", m_inputs.back()->getName());
            }
            else
            {
                m_inputs.back()->setTensorInWorkspace();
                LOG_DEBUG(TPC_NODE, "Set auxTensor {} in DRAM", m_inputs.back()->getName());
            }
        }
        else
        {
            m_inputs.back()->setAsStaticParam();
            LOG_DEBUG(TPC_NODE, "Set auxTensor {} as static param", m_inputs.back()->getName());
        }

        auxiliaryTensor++;
    }
}

bool TPCNode::validateNodeLayout() const
{
    if (m_instanceWrapper.isInstantiated())
    {
        const auto& instance = getInstance();
        unsigned    nDims    = instance.indexSpaceRank;
        if (nDims == 0)
        {
            LOG_WARN(TPC_NODE, "Kernel {} returned zero-dimensional index space", getGUID());
        }
        for (unsigned dim = 0; dim < nDims; ++dim)
        {
            if (instance.indexSpaceGeometry[dim] == 0)
            {
                LOG_WARN(TPC_NODE,
                         "Kernel {} returned {}-dimensional index space but size is set to 0 at index {}",
                         getGUID(),
                         nDims,
                         dim);
            }
        }
        validateTensorsIndexSpace(m_inputs, instance.inputTensorAccessPattern);
        validateTensorsIndexSpace(m_outputs, instance.outputTensorAccessPattern);
    }
    return Node::validateNodeLayout();
}

bool TPCNode::requiresOutputMaxDimInfer() const
{
    if (getGUID() == TPCNode::NOP_KERNEL_NAME) return false;

    return BaseClass::requiresOutputMaxDimInfer();
}

void TPCNode::permuteParams(const PermutationVector& inputPermutations)
{
    // TPC node permute parameters in the instantiation function
}

void TPCNode::isAccessPatternExceedsTensor(const TensorPtr&                        inputTensor,
                                           unsigned                                baseDim,
                                           const uint64_t*                         indexSpaceGeometry,
                                           const tpc_lib_api::TensorAccessPattern& accessPattern) const
{
    if (accessPattern.allRequired)
    {
        LOG_DEBUG(GC, "Skipping validation in case access pattern states all tensor is required.");
        return;
    }

    const NSizeArray& tensorSize = inputTensor->getAllNSizesInElements();
    const unsigned    maxDim     = std::min(inputTensor->getDim(), baseDim + tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE);
    for (unsigned int dim = baseDim; dim < maxDim; ++dim)
    {
        const auto& tran       = accessPattern.mapping[dim - baseDim];
        auto        indexSpace = indexSpaceGeometry[tran.indexSpaceDim];
        if (unlikely(indexSpace == 0))
        {
            LOG_DEBUG(GC,
                      "According to the geometry of the kernel {}, the index space is equal to zero for tensor {}. "
                      "Skipping kernel.",
                      getGUID(),
                      inputTensor->getName());
            break;
        }
        if (unlikely(tensorSize[dim] == 0))
        {
            LOG_DEBUG(GC,
                      "According to the geometry of the kernel {}, the index space is greater than zero for tensor {} "
                      "with size 0."
                      "The TPC Kernel would handle zero-sized tensor.",
                      getGUID(),
                      inputTensor->getName());
            continue;
        }

        DataRange<int64_t> rangeFirstIndex(std::min((tran.start_b), (tran.end_b + 1)),
                                           std::max((tran.start_b), (tran.end_b + 1)));

        double             offset    = static_cast<double>(tran.a) * (indexSpace - 1);
        int64_t            startLast = offset + tran.start_b;
        int64_t            endLast   = offset + tran.end_b + 1;
        DataRange<int64_t> rangeLastIndex(std::min(startLast, endLast), std::max(startLast, endLast));

        DataRange<int64_t> tensorRange(0, tensorSize[dim]);
        if (unlikely(!tensorRange.isOverlap(rangeFirstIndex) || !tensorRange.isOverlap(rangeLastIndex)))
        {
            LOG_DEBUG(GC,
                      "For kernel: {}, the access pattern is completely out of input tensors boundaries. Tensor range: "
                      "[{}-{}] while index space mapped ranges are:[{},{}] and [{}, {}] ",
                      getGUID(),
                      tensorRange.start(),
                      tensorRange.end(),
                      rangeFirstIndex.start(),
                      rangeFirstIndex.end(),
                      rangeLastIndex.start(),
                      rangeLastIndex.end());
            LOG_ERR(GC, "For kernel: {}, the access pattern is completely out of input tensors boundaries.", getGUID());
            HB_ASSERT(false, "the tensor access pattern is completely out of input tensor boundaries.");
        }
    }
}

bool TPCNode::shouldSkipAccessPatternValidation() const
{
    if (getGUID() == TPCNode::NOP_KERNEL_NAME) return true;
    bool                              shouldSkip = false;
    static constexpr std::string_view padGuid    = "pad";
    static constexpr std::string_view filterGuid = "filter";

    if (startsWith(getGUID(), padGuid) || startsWith(getGUID(), filterGuid))
    {
        shouldSkip = true;
    }
    else
    {
        shouldSkip = (getGUID().find("pool") != std::string::npos);
    }

    if (shouldSkip)
    {
        LOG_DEBUG(GC, "Skipping validation for pad, pool, reduce_sum_square, and batch_norm kernels");
        return true;
    }
    return false;
}

// This function validates the access pattern defined in the kernel is not exceeding the input tensor boundaries
// entirely. Reading partially from tensor data and from out of boundaries is allowed. The kernel families: pad and pool
// are allowed to access entirely outside of tensors boundaries
void TPCNode::validateAccessPattern() const
{
    if (shouldSkipAccessPatternValidation())
    {
        return;
    }
    const auto& instance                        = m_instanceWrapper.getInstance();
    unsigned    currentTensorFirstDescriptorIdx = 0;
    int lastAccessPatternIdx = -1;
    for (unsigned int tensorIdx = 0; tensorIdx < m_inputs.size(); ++tensorIdx)
    {
        const TensorPtr& input = m_inputs[tensorIdx];
        if (input->isShapeTensor() || input->isAuxTensor())
        {
            // Aux and shape tensors do not have access pattern entries
            LOG_DEBUG(GC,
                      "Skipping validation of {} tensor: node:{}, tensor:{}.",
                      input->isAuxTensor() ? "aux" : "shape",
                      m_name,
                      input->getName());
            continue;
        }

        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(*input, instance.inputTensorAccessPattern, currentTensorFirstDescriptorIdx);
        currentTensorFirstDescriptorIdx += info.numDescs;
        HB_ASSERT(info.baseDim < input->getDim(),
                  "access pattern refers to illegal dimension in tensor {} in node {} (type: {})",
                  input->getName(),
                  m_name,
                  getGUID());
        // We want to verify there's at least one access pattern entry for each non-aux and non-shape tensor.
        // Without casting to int, this may check if accessPatternIdx > ((unsigned)-1)
        HB_ASSERT((int)info.descIdx > lastAccessPatternIdx,
                  "illegal access pattern index for input[{}] ({}) in node {} (type: {}): accessPatternIdx: {} "
                  "expected to be bigger than {}",
                  tensorIdx,
                  input->getName(),
                  m_name,
                  getGUID(),
                  info.descIdx,
                  lastAccessPatternIdx);
        lastAccessPatternIdx = info.descIdx;

        isAccessPatternExceedsTensor(input,
                                     info.baseDim,
                                     instance.indexSpaceGeometry,
                                     instance.inputTensorAccessPattern[info.descIdx]);
    }
    currentTensorFirstDescriptorIdx = 0;
    lastAccessPatternIdx = -1;
    for (unsigned int tensorIdx = 0; tensorIdx < m_outputs.size(); ++tensorIdx)
    {
        const TensorPtr& output = m_outputs[tensorIdx];
        AccessPatternDescIdxInfo info =
            getAccessPatternDescIdx(*output, instance.outputTensorAccessPattern, currentTensorFirstDescriptorIdx);
        currentTensorFirstDescriptorIdx += info.numDescs;
        HB_ASSERT(info.baseDim < output->getDim(),
                  "access pattern refers to illegal dimension in tensor {} in node {} (type: {})",
                  output->getName(),
                  m_name,
                  getGUID());
        // We want to verify there's at least one access pattern entry for each non-aux and non-shape tensor.
        // Without casting to int, this may check if accessPatternIdx > ((unsigned)-1)
        HB_ASSERT((int)info.descIdx > lastAccessPatternIdx,
                  "illegal access pattern index for output[{}] ({}) in node {} (type: {}): accessPatternIdx: {} "
                  "expected to be bigger than {}",
                  tensorIdx,
                  output->getName(),
                  m_name,
                  getGUID(),
                  info.descIdx,
                  lastAccessPatternIdx);
        lastAccessPatternIdx = info.descIdx;
    }
}

void TPCNode::validatePreferredSplitDim() const
{
    const tpc_lib_api::HabanaKernelInstantiation& kernelInstance = getInstance();

    if (kernelInstance.preferredSplitDim == 0) return; // no preferred split dim, nothing to validate

    unsigned preferredSplitDim = kernelInstance.preferredSplitDim - 1;  // see comment in API H file declaration

    // Verify that preferredSplitDim is marked as allRequired
    auto verifier = [&](const TensorVector& tensorVec, bool isInput, tpc_lib_api::TensorAccessPattern* tensorAPs) {
        unsigned currentTensorFirstDescriptorIdx = 0;
        for (unsigned tensorIdx = 0; tensorIdx < tensorVec.size(); ++tensorIdx)
        {
            const TensorPtr& tensor = tensorVec.at(tensorIdx);
            if (tensor == nullptr) continue;
            AccessPatternDescIdxInfo info =
                getAccessPatternDescIdx(*tensor, tensorAPs, currentTensorFirstDescriptorIdx);
            currentTensorFirstDescriptorIdx += info.numDescs;
            if (preferredSplitDim >= tensor->getDim()) continue;
            const tpc_lib_api::TensorAccessPattern& tensorAP = tensorAPs[info.descIdx];
            if (tensorAP.allRequired) continue; // we are good
            bool foundMapping = false;
            for (unsigned i = 0; !foundMapping && i < tensor->getDim(); ++i)
            {
                if (tensorAP.mapping[i].indexSpaceDim == preferredSplitDim)
                {
                    HB_ASSERT(
                        tensorAP.mapping[i].allRequired,
                        "Dimension {} of tensor {} is preferredSplitDim but was not marked as allRequired",
                        preferredSplitDim,
                        tensor->getName());
                    foundMapping = true;
                }
            }
            HB_ASSERT(foundMapping, "didn't find index space mapping for dim {}", preferredSplitDim);
        }
    };

    verifier(m_inputs, true, kernelInstance.inputTensorAccessPattern);
    verifier(m_outputs, false, kernelInstance.outputTensorAccessPattern);
}

tpc_lib_api::GlueCodeReturn TPCNode::init(tpc_lib_api::DeviceId   deviceId,
                                          AuxiliaryTensors*       cachedAuxiliaryTensors,
                                          std::optional<uint32_t> kernelUniqueId)
{
    LOG_DEBUG(TPC_NODE, "Initializing TPC node - {}", getNodeName());

    if (m_instanceWrapper.isInstantiated())
    {
        return tpc_lib_api::GLUE_SUCCESS;
    }

    m_inputs.erase(std::remove(begin(m_inputs), end(m_inputs), nullptr), end(m_inputs));
    m_outputs.erase(std::remove(begin(m_outputs), end(m_outputs), nullptr), end(m_outputs));

    llvm_vecsmall::SmallVector<uint32_t, MAX_TENSOR_NR> inputsCrc;
    for (const TensorPtr& tensor : m_inputs)
    {
        if (tensor->isStaticParam() && tensor->getAddress())
        {
            inputsCrc.push_back(crc32(tensor->getAddress(), tensor->getTotalSizeInBytes()));
        }
        else
        {
            // Inserted to align indexes, will be ignored
            inputsCrc.push_back(0);
        }
    }

    m_instanceWrapper.initParams(*this, deviceId);
    if (tpc_lib_api::GlueCodeReturn ret = instantiate(m_instanceWrapper); ret != tpc_lib_api::GLUE_SUCCESS)
    {
        m_instanceWrapper.resetParams();
        LOG_WARN(TPC_NODE, "Can't instantiate TPC kernel (GlueCode return code = {})", ret);
        return ret;
    }

    TpcElfTools::TpcElfStatus result = m_instanceWrapper.extractElf(m_name);
    if (result != TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS)
    {
        return tpc_lib_api::GLUE_FAILED;
    }

    if (m_instanceWrapper.isValidElfProgramHeader())
    {
        // For kernels supporting SMT4, the value of numberOfThreads should be set to 4. Currently we are not supporting
        // any number of threads other than 1 (single thread)
        HB_ASSERT(m_instanceWrapper.getElfProgramHeader().numberOfThreads == 1,
                  "SMT4 not supported. Num of threads: {}", m_instanceWrapper.getElfProgramHeader().numberOfThreads);
    }

    const bool dynamicNode = isDynamicShape();
    if (dynamicNode && KernelDB::instance().isDynamicShapeKernel(getGUIDAndHash(), deviceId))
    {
        auto getSifId = [deviceId](const std::string& name,
                                   const std::string& guid,
                                   uint64_t&          sifId,
                                   uint64_t&          sifVersion) {
            tpc_lib_api::UniqueShapeInferenceHash sifHash = {};

            if (!KernelDB::instance().GetKernelShapeInferenceFunctionID(deviceId, guid, &sifHash))
            {
                LOG_ERR(GC, "Can't get shape inference function for dynamic shape tpc node {}", guid);
                return tpc_lib_api::GLUE_FAILED;
            }
            sifId = sifHash.Value;
            sifVersion = KernelDB::instance().GetLibraryVersion(deviceId, guid);
            LOG_TRACE(GC,
                      "TPC node name {} (guid {}) got shape inference function id {} version {}",
                      name,
                      guid,
                      sifHash.Value,
                      sifVersion);

            return tpc_lib_api::GLUE_SUCCESS;
        };

        auto multiSifInfo = getMultiSifInfo();
        if (multiSifInfo != nullptr)
        {
            for (auto& subnode : multiSifInfo->m_nodes)
            {
                if (KernelDB::instance().isDynamicShapeKernel(subnode.m_nodeGUID, deviceId))
                {
                    auto ret = getSifId(subnode.m_nodeName, subnode.m_nodeGUID, subnode.m_sifID.sm_func_index, subnode.m_sifVersion);
                    if (ret != tpc_lib_api::GLUE_SUCCESS)
                    {
                        LOG_ERR(GC,
                                "Can't get shape inference function for dynamic shape tpc node {}",
                                subnode.m_nodeGUID);
                        return ret;
                    }
                }
            }
        }
        else
        {
            auto ret = getSifId(m_name, getGUID(), m_shapeInferenceFunctionID, m_shapeInferenceFunctionVersion);
            if (ret != tpc_lib_api::GLUE_SUCCESS)
            {
                LOG_ERR(GC, "Can't get shape inference function for dynamic shape tpc node {}", getGUID());
                return ret;
            }

            LOG_TRACE(GC,
                      "TPC node name {} (guid {}) got shape inference function id {} version {}",
                      m_name,
                      getGUID(),
                      m_shapeInferenceFunctionID,
                      m_shapeInferenceFunctionVersion);
        }
    }

    // Validate that sent data was not modified
    for (unsigned t = 0; t < m_inputs.size(); ++t)
    {
        const TensorPtr& inputTensor = getInput(t);
        if (!inputTensor) continue;
        if (!(inputTensor->isStaticParam() && inputTensor->getAddress())) continue;

        if (inputsCrc[t] != crc32(inputTensor->getAddress(), inputTensor->getTotalSizeInBytes()))
        {
            LOG_ERR(TPC_NODE,
                    "Tensor data of tensor index {} was changed in call to {} kernel glue code",
                    t,
                    getGUID());
            return tpc_lib_api::GLUE_FAILED;
        }
    }

    createAuxTensors(cachedAuxiliaryTensors);

    initPrintfTensor();
    setSparseAccessTensorsAnnotation();

    m_uniqueID = kernelUniqueId ? *kernelUniqueId
                                : crc32(m_instanceWrapper.getKernelBinary(), m_instanceWrapper.getKernelSize());
    m_instanceWrapper.setInstantiated(true);

    if (dynamicNode)
    {
        generateDynamicShapeProjectionsTensors();
    }

    validateAccessPattern();
    validatePreferredSplitDim();

    // setTensorsPrefetchStride only if we're using DCache Prefetcher (DCacheLineNr > 0)
    const HalReader* reader = CompilationHalReader::getHalReader(true).get();
    if (reader && reader->getDCacheLineNr())
    {
        setTensorsPrefetchStride(*reader);
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

const tpc_lib_api::HabanaKernelParams& TPCNode::getSucceededGlueParams() const
{
    HB_ASSERT(m_instanceWrapper.isInstantiated(),
              "Request for glue params from uninstantiated tpc node: {}",
              getNodeName());
    return m_instanceWrapper.getGlueParams();
}

bool TPCNode::hasMandatorySplitDim() const
{
    // Although TPC Lib interface uses the word "preferred", it's actually mandatory. GC must split on this dim first.
    return m_instanceWrapper.isInstantiated() && getInstance().preferredSplitDim > 0;
}

unsigned TPCNode::getMandatorySplitDim() const
{
    HB_ASSERT(hasMandatorySplitDim(), "getMandatorySplitDim() was called for TPCNode that has no mandatory split dim");
    // Although TPC Lib interface uses the word "preferred", it's actually mandatory. GC must split on this dim first.
    return getInstance().preferredSplitDim - 1; // see comment in API H file declaration for the -1 explanation
}

std::optional<TPCNode::CostModelResult> TPCNode::getCostModelResult() const
{
    TpcElfTools::CostModelResult result;
    auto maxAvailableTpc = getMaxAvailableTpc(deviceTypeToDeviceID(m_graphTraits->getHalReader()->getDeviceType()));
    TpcElfTools::TpcElfStatus    status =
        TpcElfTools::GetTpcProgramCostModelValuesV4(&getSucceededGlueParams(), &getInstance(), maxAvailableTpc, result);
    if (TpcElfTools::TPC_ELF_SUCCESS != status)
    {
        LOG_WARN(COST_MODEL, "Request for cost model result failed for tpc node: {}", getNodeName());
        return std::nullopt;
    }
    return TPCNode::CostModelResult(result);
}

tpc_lib_api::GlueCodeReturn
TPCNode::getSuggestedTensorManipulation(tpc_lib_api::TensorManipulationSuggestion* suggestion)
{
    m_instanceWrapper.initParams(*this, getGraphTraits()->getDeviceId());

    return KernelDB::instance().GetSuggestedTensorManipulation(&m_instanceWrapper.getGlueParams(),
                                                               suggestion,
                                                               getGUIDAndHash());
}

static bool hasTransposeOp(const tpc_lib_api::TensorOperation* operations, uint32_t numOps)
{
    for (uint32_t index = 0; index < numOps; ++index)
    {
        if (operations[index].opType == tpc_lib_api::TENSOR_OP_TRANSPOSE) return true;
    }
    return false;
}

bool TPCNode::hasTransposeOptimization(tpc_lib_api::DeviceId deviceId) const
{
    KernelInstantiationWrapper instance;

    if (!getInfoInstance(instance, deviceId, false))
    {
        LOG_WARN(TPC_NODE,
                 "Cannot instantiate TPC node ({}) GUID ({}) in order to determine if it has transpose optimization",
                 getNodeName(),
                 getGUID());
        return false;
    }
    tpc_lib_api::TensorManipulationSuggestion suggestion;
    tpc_lib_api::TensorOperation              inputTensors[MAX_TENSOR_NR];
    tpc_lib_api::TensorOperation              outputTensors[MAX_TENSOR_NR];
    suggestion.inputTensors  = inputTensors;
    suggestion.outputTensors = outputTensors;
    KernelDB::instance().GetSuggestedTensorManipulation(&instance.getGlueParams(), &suggestion, getGUIDAndHash());

    return hasTransposeOp(suggestion.inputTensors, getNumInputs()) ||
           hasTransposeOp(suggestion.outputTensors, getNumOutputs());
}

bool TPCNode::runShapeInferenceFunction(synDeviceType deviceType,
                                        SifParams*    params,
                                        SifOutputs*   outputs,
                                        bool          inferMax,
                                        bool          skipStatic)
{
    auto multiSifInfo = getMultiSifInfo();

    tpc_lib_api::GlueCodeReturn ret;

    if (multiSifInfo != nullptr)
    {
        ret = multiSifRun(deviceType, multiSifInfo.get(), params, outputs, inferMax);
    }
    else
    {
        auto deviceIdGlueFormat = deviceTypeToDeviceID(deviceType);

        params->maxAvailableTpc = TPCNode::getMaxAvailableTpc(deviceIdGlueFormat);

        ret = KernelDB::instance().RunShapeInferenceFunction(deviceIdGlueFormat, getGUID(), params, outputs);
    }

    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_ERR(GC, "Running shape inference for tpc node {} guid {} error {}", m_name, getGUID(), enumToString(ret));

        return false;
    }

    return true;
}

bool TPCNode::isTensorCoveringIndexSpaceDim(unsigned                                indexSpaceDim,
                                            TSize*                                  tensorDimsSize,
                                            unsigned int                            dimNum,
                                            const uint64_t*                         geometry,
                                            const tpc_lib_api::TensorAccessPattern& accessPattern,
                                            unsigned&                               tensorDim,
                                            unsigned&                               missingElements) const
{
    tensorDim = -1;
    LOG_TRACE(TPC_NODE, "TPC node ({}) GUID ({}) isTensorCoveringIndexSpaceDim", getNodeName(), getGUID());
    if (accessPattern.allRequired)
    {
        return false;
    }

    // Projections are mappings from a tensor dimension to an index space dimension.
    // An interval of an index space is projected to an interval of a tensor space.
    // The reverse mapping (from index space to tensor) is stored in DimIndexSpaceMapping
    // and consists of 3 numbers a, start_b, end_b.
    //
    // In TPC we can use a single tensor dimension projection to describe a
    // index-space dimension as long as it has a one-to-one correspondence
    // relation to that index space dimension.
    // In order to check that a tensor fulfills these conditions:
    // 1. The index space dimension covers the entire tensor dimension.
    // 2. There is no overlap in the tensor index space dimension.
    for (unsigned int dim = 0; dim < dimNum; ++dim)
    {
        const auto& tran = accessPattern.mapping[dim];
        if (tran.indexSpaceDim != indexSpaceDim) continue;

        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) dim {} found index space dim {}",
                  getNodeName(),
                  getGUID(),
                  dim,
                  indexSpaceDim);
        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) a {}, start_b {}, end_b {}",
                  getNodeName(),
                  getGUID(),
                  tran.a,
                  tran.start_b,
                  tran.end_b);

        if (tran.indexSpaceDim == 0 && tran.a == 0 && tran.start_b == 0 && tran.end_b == 0)
        {
            // index space mapping == 0 ignoring mapping.
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) dim {} index space mapping == 0 ignoring mapping",
                      getNodeName(),
                      getGUID(),
                      dim);
            return false;
        }
        tensorDim = dim;

        if (tran.start_b != 0)
        {
            // index space does not start at the start of the tensor.
            // And so, the index space either doesn't cover the entire tensor
            // or is outside the tensor indicating pad.
            LOG_TRACE(TPC_NODE, "TPC node ({}) GUID ({}) tran.start_b != 0", getNodeName(), getGUID());
            return false;
        }

        auto indexSpaceDimSize = (double)tran.a * (geometry[tran.indexSpaceDim] - 1) + tran.end_b + 1;
        auto tensorDimSize = tensorDimsSize[dim];
        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) geometry.sizes[tran.dim] {} indexSpaceDimSize {}, tensorDimSize {}",
                  getNodeName(),
                  getGUID(),
                  geometry[tran.indexSpaceDim],
                  indexSpaceDimSize,
                  tensorDimSize);
        if (tensorDimSize > indexSpaceDimSize ||
            (tran.a != 0 && geometry[tran.indexSpaceDim] > 1 && tensorDimSize <= indexSpaceDimSize - tran.a))
        {
            // tensor size does not end inside the last index space step.
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) tensor size does not end inside the last index space step.",
                      getNodeName(),
                      getGUID());
            return false;
        }

        if (geometry[tran.indexSpaceDim] > 1 && tran.end_b != tran.a - 1)
        {
            // either index space does not cover the tensor as there is a hole in the between
            // adjacent activations (tran.end_b > tran.a - 1) or there are overlaps
            // (tran.end_b < tran.a - 1)
            LOG_TRACE(TPC_NODE, "TPC node ({}) GUID ({}) tran.end_b != tran.a - 1", getNodeName(), getGUID());
            return false;
        }
        missingElements = indexSpaceDimSize - tensorDimSize;
        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) dim {} covering index space dim {}, missing elements {}",
                  getNodeName(),
                  getGUID(),
                  dim,
                  indexSpaceDim,
                  missingElements);
        return true;
    }
    LOG_TRACE(TPC_NODE,
              "TPC node ({}) GUID ({}) didn't find index space dim {}",
              getNodeName(),
              getGUID(),
              indexSpaceDim);
    return false;
}

bool TPCNode::findProjectionBetterCoveringIndexSpaceDimension(unsigned                          indexSpaceDim,
                                                              TensorVector                      tensors,
                                                              bool                              isOutput,
                                                              unsigned&                         minMissingElements,
                                                              unsigned&                         maxRank,
                                                              bool&                             foundMappedDim,
                                                              Node::NodeDynamicShapeProjection& projection) const
{
    auto tensorAccessPattern =
        isOutput? getInstance().outputTensorAccessPattern: getInstance().inputTensorAccessPattern;
    bool foundProjection = false;
    for (unsigned i = 0 ; i < (unsigned)tensors.size(); ++i)
    {
        auto tensorPtr = tensors[i];
        if (tensorPtr->isAuxTensor() || tensorPtr->isShapeTensor()) continue;
        NSizeArray tensorSize              = tensorPtr->getNSizesInElements();
        unsigned   tensorDim               = -1;
        unsigned   missingElements         = 0;
        bool       foundPossibleProjection = isTensorCoveringIndexSpaceDim(indexSpaceDim,
                                                                     tensorSize.data(),
                                                                     tensorPtr->getDim(),
                                                                     getInstance().indexSpaceGeometry,
                                                                     tensorAccessPattern[i],
                                                                     tensorDim,
                                                                     missingElements);

        if (foundPossibleProjection)
        {
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) found a possible projection for index space dim {}, "
                      "isOutput: {} index: {} dim: {} dynamic: {}",
                      getNodeName(),
                      getGUID(),
                      indexSpaceDim,
                      isOutput,
                      i,
                      tensorDim,
                      tensorPtr->isDynamicDim(tensorDim));
        }

        if (foundPossibleProjection && ((minMissingElements > missingElements) ||
                                        (minMissingElements == missingElements && tensorPtr->getDim() > maxRank)))

        {
            projection.indexSpaceDim = indexSpaceDim;
            projection.tensorDim = tensorDim;
            projection.tensorIdx = i;
            projection.isOutput = isOutput;
            foundProjection = true;
            minMissingElements = missingElements;
            maxRank              = tensorPtr->getDim();
        }

        if (tensorDim != -1 && tensorPtr->isDynamicDim(tensorDim))
        {
            foundMappedDim = true;
        }
    }
    return foundProjection;
}

void TPCNode::generateDynamicShapeProjectionsTensors()
{
    auto& instance = getInstance();
    // Since this vector is copied in TPC node clone, we need to clear it here for nodes that are re-instantiated to
    // avoid duplications.
    m_dynamicShapeProjectionTensors.clear();
    for (uint32_t indexSpaceDim = 0; indexSpaceDim < instance.indexSpaceRank; ++indexSpaceDim)
    {
        bool foundProjection = false;
        bool foundMappedDim = false;
        unsigned minMissingElements = std::numeric_limits<unsigned>::max();
        unsigned                         maxRank            = 0;
        Node::NodeDynamicShapeProjection projection;

        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) index space dim {} going over outputs",
                  getNodeName(),
                  getGUID(),
                  indexSpaceDim);
        foundProjection |= findProjectionBetterCoveringIndexSpaceDimension(indexSpaceDim,
                                                                           getOutputs(),
                                                                           true,
                                                                           minMissingElements,
                                                                           maxRank,
                                                                           foundMappedDim,
                                                                           projection);

        LOG_TRACE(TPC_NODE,
                  "TPC node ({}) GUID ({}) index space dim {} going over inputs",
                  getNodeName(),
                  getGUID(),
                  indexSpaceDim);
        foundProjection |= findProjectionBetterCoveringIndexSpaceDimension(indexSpaceDim,
                                                                           getInputs(),
                                                                           false,
                                                                           minMissingElements,
                                                                           maxRank,
                                                                           foundMappedDim,
                                                                           projection);

        if (foundProjection)
        {
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) found a valid projection for index space dim {}, "
                      "isOutput: {} index: {} dim: {} mapped: {} ",
                      getNodeName(),
                      getGUID(),
                      indexSpaceDim,
                      projection.isOutput,
                      projection.tensorIdx,
                      projection.tensorDim,
                      foundMappedDim);
            m_dynamicShapeProjectionTensors.push_back(projection);
            continue;
        }

        if (!foundMappedDim)
        {
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) Cannot find a tensor dim mapping index space dim {}, "
                      "marking it as static",
                      getNodeName(),
                      getGUID(),
                      indexSpaceDim);
            continue;
        }

        if (!foundProjection)
        {
            LOG_TRACE(TPC_NODE,
                      "TPC node ({}) GUID ({}) Cannot find a valid projection for index space dim {}, "
                      "using all tensors",
                      getNodeName(),
                      getGUID(),
                      indexSpaceDim);
            m_dynamicShapeProjectionTensors.clear();
            return;
        }
    }
    LOG_TRACE(TPC_NODE,
              "TPC node ({}) GUID ({}) valid projection, size {}",
              getNodeName(),
              getGUID(),
              m_dynamicShapeProjectionTensors.size());
}

std::vector<Node::NodeDynamicShapeProjection> TPCNode::getDynamicShapeProjectionsTensors() const
{
    return m_dynamicShapeProjectionTensors;
}

unsigned TPCNode::getKernelSize() const
{
    if (!m_instanceWrapper.isInstantiated())
    {
        LOG_ERR(TPC_NODE, "Attempt to query the size of a kernel that wasn't instantiated");
        return 0;
    }
    return m_instanceWrapper.getKernelSize();
}

KernelInfo TPCNode::getKernelInfo() const
{
    KernelInfo result;
    if (!m_instanceWrapper.isInstantiated())
    {
        LOG_ERR(TPC_NODE, "Attempt to query the size of a kernel that wasn't instantiated");
        return result;
    }
    const auto& cachedBinary = m_instanceWrapper.getCacheKernel();
    if (cachedBinary == nullptr)
    {
        result.kernelBinary = (char*)m_instanceWrapper.getKernelBinary();
        if (m_instanceWrapper.shouldCacheElf())
        {
            result.cachedBinary = m_instanceWrapper.getElfBuffer();
        }
    }
    else
    {
        result.kernelBinary = cachedBinary.get();
        result.cachedBinary = cachedBinary;
    }
    result.kernelId   = getUniqueID();
    result.kernelSize = m_instanceWrapper.getKernelSize();
    return result;
}

unsigned TPCNode::getPrintfPosition(unsigned int descTensorCount) const
{
    if (!m_instanceWrapper.isValidElfProgramHeader())
    {
        LOG_WARN(TPC_NODE, "TPC ELF header is not valid");
        return descTensorCount;
    }
    return m_printfPosition;
}

std::string_view TPCNode::getEngineTypeStr() const
{
    return "TPC";
}

bool TPCNode::isCast() const
{
    return isCastGUID(getGUID());
}

TensorSemanticType TPCNode::getParamSemanticType(const TensorPtr& param) const
{
    const char* EMBEDDING_GUID = "embedding";
    bool isEmbedding           = (strncmp(getGUID().c_str(), EMBEDDING_GUID, strlen(EMBEDDING_GUID)) == 0);
    if (isEmbedding)
    {
        if (param == getInput(TENSOR_IFM))
        {
            return TYPE_WEIGHTS;
        }
    }
    return Node::getParamSemanticType(param);
}

std::map<TensorPtr, TensorVector, TensorComparator>
TPCNode::getReusableInputs(const KernelInstantiationWrapper& instanceWrapper, bool isSuggestedBinding) const
{
    std::map<TensorPtr, TensorVector, TensorComparator> ret;
    if (m_inputs.empty()) return ret;  // early exit
    const TPCPhysicalMemoryOpNode* memcopyNode = dynamic_cast<const TPCPhysicalMemoryOpNode*>(this);
    if (memcopyNode != nullptr)
        // cannot have reusability with physical memory operation nodes
        return ret;

    for (unsigned outIdx = 0; outIdx < m_outputs.size(); ++outIdx)
    {
        const auto& outputTensorAccessPattern = instanceWrapper.getInstance().outputTensorAccessPattern[outIdx];
        if (outputTensorAccessPattern.inputsReusability == 0) continue;

        const TensorPtr& output = m_outputs[outIdx];
        HB_ASSERT_PTR(output);
        std::bitset<MAX_TENSOR_NR> inputsReusability(outputTensorAccessPattern.inputsReusability);

        for (unsigned i = 0; i < m_inputs.size(); ++i)
        {
            if (!inputsReusability.test(i)) continue;
            const TensorPtr& input = m_inputs[i];
            if (!input) continue;
            if (input->isAuxTensor()) continue;

            // If isSuggestedBinding is true, return tensors which CAN be reused,
            // otherwise return tensors that MUST be reused; their inputReusabilityBinding flag is on.
            bool isInputReusability = (isSuggestedBinding || outputTensorAccessPattern.inputReusabilityBinding);

            if (isInputReusability)
            {
                ret[output].push_back(input);
            }
        }
    }
    return ret;
}

std::map<TensorPtr, TensorVector, TensorComparator> TPCNode::getReusableInputs(bool isSuggestedBinding) const
{
    if (m_instanceWrapper.isInstantiated())
    {
        return getReusableInputs(m_instanceWrapper, isSuggestedBinding);
    }
    // The function might be used by passes that are running before the kernel instantiation.
    // For example: Common sub-expression elimination (CSE)
    HB_ASSERT_PTR(m_graphTraits);
    synDeviceType              deviceType      = m_graphTraits->getHalReader()->getDeviceType();
    KernelInstantiationWrapper instanceWrapper = m_instanceWrapper;
    getInfoInstance(instanceWrapper, deviceTypeToDeviceID(deviceType), false);
    return getReusableInputs(instanceWrapper, isSuggestedBinding);
}

std::map<TensorPtr, TensorVector, TensorComparator> TPCNode::getReusableInputs() const
{
    return getReusableInputs(true);
}

std::map<TensorPtr, TensorVector, TensorComparator> TPCNode::getReusableInputBinding() const
{
    return getReusableInputs(false);
}

TensorVector TPCNode::getMemsetBeforeExecTensors() const
{
    TensorVector ret;
    for (unsigned idx = 0; idx < m_outputs.size(); ++idx)
    {
        if (m_instanceWrapper.getInstance().outputTensorAccessPattern[idx].memsetBeforeExecution)
        {
            ret.push_back(getOutput(idx));
        }
    }
    return ret;
}

// TODO - remove this awkward function and use the method from [SW-27229] instead
bool TPCNode::isBroadcastableOperation() const
{
    std::string_view op = getGUIDWithoutDtype();
    if (op.find("_fwd") != std::string::npos) // the _fwd suffix doesn't prevent from being broadcast
    {
        op = op.substr(0, op.find("_fwd"));
    }

    return std::find(m_broadcastableKernels.begin(), m_broadcastableKernels.end(), op) != m_broadcastableKernels.end();
}

bool TPCNode::isSuggestedOptimizationDone() const
{
    return m_optimized;
}

bool TPCNode::validateNodeForGraph(const HabanaGraph &g) const
{
    // in inference TPCNodes can be upgraded - validation is done without precision
    if (g.getInferenceMode())
    {
        return validateNodeWithoutPrecision();
    }
    else if (KernelDB::instance().isKernelExist(getGUIDAndHash(), getGraphTraits()->getDeviceId()) ||
             getGUID() == TPCNode::NOP_KERNEL_NAME)
    {
        if (isDynamicShape() &&
            !KernelDB::instance().isDynamicShapeKernel(getGUIDAndHash(), getGraphTraits()->getDeviceId()))
        {
            LOG_ERR(HABANA_NODE, "{}: TPC kernel with guid \"{}\" doesn't support DS", getNodeName(), getGUID());
            return false;
        }
    }
    else
    {
        LOG_ERR(HABANA_NODE, "{}: TPC kernel with guid \"{}\" doesn't exist", getNodeName(), getGUID());
        return false;
    }
    return true;
}

bool TPCNode::validateNodeWithoutPrecision() const
{
    bool valid = KernelDB::instance().isKernelExist(getGUIDAndHash(), getGraphTraits()->getDeviceId());
    if (!valid)
    {
        const auto isCast = this->isCast();
        for (synDataType dtype = (synDataType)1; dtype < syn_type_max; dtype = (synDataType)((unsigned)dtype << 1))
        {
            const std::string_view dtypeSuffix = getDtypeSuffixFromSynDataType(dtype);
            StringWithHash         guidAndHash(isCast ? fmt::format("cast_{}_to_{}", dtypeSuffix, getDtypeFromGUID())
                                                      : fmt::format("{}_{}", getGUIDWithoutDtype(), dtypeSuffix));

            if (KernelDB::instance().isKernelExist(guidAndHash, getGraphTraits()->getDeviceId()))
            {
                // If this is dynamic shape node but the kernel doesn't support dynamic shape.
                if (isDynamicShape() &&
                    !KernelDB::instance().isDynamicShapeKernel(guidAndHash, getGraphTraits()->getDeviceId()))
                {
                    continue;
                }

                return true;
            }
        }
    }
    return valid;
}

void TPCNode::upgradeNodePrecisionIfMissingKernel(bool forceUpgrade /*= false*/)
{
    bool kernelExists = KernelDB::instance().isKernelExist(getGUIDAndHash(), getGraphTraits()->getDeviceId());
    if (!kernelExists || forceUpgrade)
    {
        std::string guidToSet = getUpgradedGUID(getGUID());
        if (guidToSet.empty())
        {
            if (forceUpgrade)
            {
                LOG_WARN(TPC_NODE, "TPC kernel {} not found, nor a suitable replacement", getGUID());
            }
            else
            {
                LOG_WARN(TPC_NODE, "TPC kernel {} not found", getGUID());
            }
            guidToSet = getGUID();
        }
        else
        {
            LOG_INFO(TPC_NODE, "TPC kernel internal change from {} to {}", getGUID(), guidToSet);

        }
        m_GUID = guidToSet;
    }
}

// Check if the tensor sizes are equal in dimension dim
static bool tensorSizesEqual(TensorPtr input, TensorPtr output, unsigned dim)
{
    if (input->getDim() != output->getDim())
    {
        return false;
    }
    return (input->getSizeInElements(dim) == output->getSizeInElements(dim));
}

static bool indicesOverlap(unsigned                                      tensorIdx,
                           TensorPtr                                     tensor,
                           const tpc_lib_api::HabanaKernelInstantiation& instance,
                           unsigned                                      dim,
                           bool                                          isInput)
{
    const auto& tensorAccessPattern =
        isInput ? instance.inputTensorAccessPattern[tensorIdx] : instance.outputTensorAccessPattern[tensorIdx];

    // Check if TPC kernel is allowed to be sliced
    if (tensorAccessPattern.allRequired) return true;

    // If dimension size is one, it always use the same element
    // So it can be considered as separable
    if (tensor->getSizeInElements(dim) == 1) return false;

    unsigned indexSpaceDim = tensorAccessPattern.mapping[dim].indexSpaceDim;
    if (instance.indexSpaceGeometry[indexSpaceDim] == 1) return false;
    double a       = tensorAccessPattern.mapping[dim].a;
    double start_b = tensorAccessPattern.mapping[dim].start_b;
    double end_b   = tensorAccessPattern.mapping[dim].end_b;

    // Check for an overlap between ranges 0 and 1.
    // Check the integer range, as for non integer a or b the float value might not overlap, while the integer
    // elements do
    DataRange<double> range0(std::floor(std::min(start_b, end_b)), std::ceil(std::max(start_b, end_b) + 1));
    DataRange<double> range1(std::floor(std::min(a + start_b, a + end_b)),
                             std::ceil(std::max(a + start_b, a + end_b) + 1));

    return (range0.isOverlap(range1));
}

// specific kernels that are exceptions to the isSeparable() function - they are "ElementWise" by definition
bool TPCNode::isSeparableException() const
{
    std::string_view GUIDWithoutDType = extractGUIDFromFullGUID(getGUID());

    if (GUIDWithoutDType == "gelu_fwd") return true;

    for (const auto direction : {Direction::FWD, Direction::BWD})
    {
        std::string bn1FullGUID(getBN1Guid(direction, syn_type_float));
        std::string_view bn1GUID = extractGUIDFromFullGUID(bn1FullGUID);
        if (GUIDWithoutDType == bn1GUID) return true;

        for (const auto operation : {BN_OPS_BN, BN_OPS_BN_ACTIVATION, BN_OPS_BN_ADD_ACTIVATION})
        {
            std::string bn2FullGUID(getBN2Guid(operation, direction, syn_type_float));
            std::string_view bn2GUID = extractGUIDFromFullGUID(bn2FullGUID);
            if (GUIDWithoutDType == bn2GUID) return true;
        }
    }
    return false;
}

// kernels that use LFSR random generators will output different results when shape is manipulated.
bool TPCNode::isRestrictedShapeRandomNode() const
{
    return !GCFG_ENABLE_LFSR_KERNEL_SHAPE_MANIPULATION.value() &&
           (m_lfsrRandomGuidList.find(extractGUIDFromFullGUID(getGUID())) != m_lfsrRandomGuidList.end());
}

void TPCNode::setNodePrecision(synDataType precision)
{
    if (precision == getNodePrecision()) return;

    Node::setNodePrecision(precision);

    bool isCastGuid = isCast();
    // TODO - simplify this when SW-40016 is resolved
    synDataType guidPrecision =
        getSynDataTypeFromDtypeSuffix(isCastGuid ? extractDtypeFromCastGUID(getGUID()) : getDtypeFromGUID());

    // Change a suffix-less guid only if it's missing from the kernel DB
    if (guidPrecision == syn_type_na &&
        KernelDB::instance().isKernelExist(getGUIDAndHash(), getGraphTraits()->getDeviceId()))
        return;

    // If the precision was taken from the GUID, it's redundant to re-set it
    if (precision == guidPrecision) return;

    if (isCastGuid)
    {
        std::string_view castFrom = getDtypeSuffixFromSynDataType(precision);
        std::string_view castTo   = getDtypeFromGUID();
        if (castFrom.empty() || castTo.empty())
        {
            return;
        }

        setGUID(fmt::format("cast_{}_to_{}", castFrom, castTo));
        return;
    }

    std::string_view suffix   = getDtypeSuffixFromSynDataType(precision);
    std::string_view bareGuid = getGUIDWithoutDtype();
    if (suffix.empty())
    {
        setGUID(bareGuid);
    }
    else
    {
        setGUID(fmt::format("{}_{}", bareGuid, suffix));
    }
}

bool TPCNode::isGuidBlocked(const std::string& guid)
{
    return (m_seperableBlockList.find(extractGUIDFromFullGUID(guid)) != m_seperableBlockList.end());
}

// Heuristic check if the kernel behaves like element-wise, and can be sliced to any size
bool TPCNode::isSeparable(tpc_lib_api::DeviceId deviceId) const
{
    return isSeparableOnAllOrSingleDim(deviceId, true);
}

bool TPCNode::isSeparable(tpc_lib_api::DeviceId deviceId, unsigned dimension) const
{
    return isSeparableOnAllOrSingleDim(deviceId, false, dimension);
}

bool TPCNode::isSeparableOnAllOrSingleDim(tpc_lib_api::DeviceId deviceId, bool checkAllDimensions, unsigned dim) const
{
    if (isSeparableException()) return true;

    KernelInstantiationWrapper instance;

    if (!getInfoInstance(instance, deviceId, false))
    {
        LOG_WARN(TPC_NODE,
                 "Cannot instantiate TPC node ({}) GUID ({}) in order to determine if elementwise",
                 getNodeName(),
                 getGUID());
        return false;
    }

    if (isGuidBlocked(getGUID()))
    {
        LOG_WARN(TPC_NODE, "TPC node ({}) GUID ({}) is in blocked list and not seprable ", getNodeName(), getGUID());
        return false;
    }

    TensorPtr output = getOutputs().front();
    for (const auto& otherOut : getOutputs())
    {
        if (output == otherOut) continue;
        if (output->is1DAndSameDepthAsOther(otherOut))
        {
            output = otherOut;
            continue;
        }

        if (!checkAllDimensions && output->getDim() <= dim) return false;
        else if (output->getDim() != otherOut->getDim())
        {
            if (otherOut->is1DAndSameDepthAsOther(output)) continue;
            else return false;
        }
        else
        {
            if (checkAllDimensions)
            {
                if (output->getNSizesInElements() != otherOut->getNSizesInElements()) return false;
            }
            else
            {
                if (output->getSizeInElements(dim) != otherOut->getSizeInElements(dim)) return false;
            }
        }
    }
    unsigned  tensorIdx = 0;
    for (TensorPtr input: getInputs())
    {
        // Scalar is always considered - so can be acounted as separable
        if (input->getTotalElements() == 1)
        {
            ++tensorIdx;
            continue;
        }
        bool bIsSameDepthAndInput1D = input->is1DAndSameDepthAsOther(output);
        if (checkAllDimensions)
        {
            for (unsigned i = 0; i < input->getDim(); i++)
            {
                if (!(bIsSameDepthAndInput1D || tensorSizesEqual(input, output, i))) return false;
                if (indicesOverlap(tensorIdx, input, instance.getInstance(), i, true /*isInput*/)) return false;
            }
        }
        else
        {
            if (dim >= input->getDim()) return bIsSameDepthAndInput1D;
            if (!(bIsSameDepthAndInput1D || tensorSizesEqual(input, output, dim))) return false;
            if (indicesOverlap(tensorIdx, input, instance.getInstance(), dim, true /*isInput*/)) return false;
        }
        tensorIdx++;
    }
    return true;
}

std::vector<bool> TPCNode::getInputsScalarPipeStatus(tpc_lib_api::DeviceId deviceId) const
{
    std::vector<bool> scalarPipeStatus;
    KernelInstantiationWrapper instance;

    if (!getInfoInstance(instance, deviceId, true, true) || !instance.isValidElfProgramHeader())
    {
        LOG_WARN(TPC_NODE, "Cannot instantiate TPC node ({}) in order to get scalarLoad", getNodeName());

        return scalarPipeStatus;
    }

    const TpcElfTools::TPCProgramHeader& programHeader = instance.getElfProgramHeader();
    for (int idx = 0; idx < getNumInputs(); idx++)
    {
        scalarPipeStatus.push_back((programHeader.scalarLoad & (1 << idx)) ? true : false);
    }

    return scalarPipeStatus;
}

uint64_t TPCNode::getScalarPipeInputsSize(tpc_lib_api::DeviceId deviceId) const
{
    std::vector<bool> scalarPipeStatus = getInputsScalarPipeStatus(deviceId);
    if (scalarPipeStatus.empty())
    {
        return 0;
    }
    HB_ASSERT(scalarPipeStatus.size() == getNumInputs(), "inputs number mismatch");
    uint64_t scalarPipeInputsSizeBytes = 0;
    for (auto idx = 0; idx < getNumInputs(); idx++)
    {
        if (!getInput(idx)) continue;
        if (scalarPipeStatus[idx])
        {
            scalarPipeInputsSizeBytes += getInput(idx)->getTotalSizeInBytes();
        }
    }
    return scalarPipeInputsSizeBytes;
}

bool TPCNode::getInfoInstance(KernelInstantiationWrapper& instance,
                              tpc_lib_api::DeviceId       deviceId,
                              bool                        extractElf,
                              bool                        setReducible) const
{
    if (m_instanceWrapper.isInstantiated())
    {
        // use the existing instance if TPC node was already initialized
        instance = m_instanceWrapper;
    }
    else
    {
        /* At this level we are defining all tensors as in
         * SRAM in order to get the binary kernel of the in SRAM
         * version of the kernel
         */
        instance.initParams(*this, deviceId, setReducible);
        if (auto ret = instantiate(instance); ret != tpc_lib_api::GLUE_SUCCESS)
        {
            LOG_WARN(TPC_NODE, "Cannot instantiate TPC node ({}) (GlueCode return {})", getNodeName(), ret);
            return false;
        }
    }

    return !extractElf || instance.extractElf(m_name) == TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS;
}

void TPCNode::addDoubleStoreAccessPattern(unsigned tensorIdx, bool isInput)
{
    const auto& allOperands = getOperands();
    auto        isNdim      = std::any_of(allOperands.begin(), allOperands.end(), [](const TensorPtr& t) {
        return (t && (t->getDim() > SYN_MAX_TENSOR_DIM));
    });
    HB_ASSERT(!isNdim, "Adding double store for N-dim node {} is unsupported", getNodeName());
    auto& accessPattern = isInput ? m_instanceWrapper.getInstance().inputTensorAccessPattern
                                  : m_instanceWrapper.getInstance().outputTensorAccessPattern;
    memcpy(&m_instanceWrapper.getInstance().outputTensorAccessPattern[m_instanceWrapper.getGlueParams().outputTensorNr],
           &accessPattern[tensorIdx],
           sizeof(accessPattern[tensorIdx]));
    m_instanceWrapper.getGlueParams().outputTensorNr++;
}

void TPCNode::setKernelElf(void *kernelElf, uint32_t elfSize)
{
    m_instanceWrapper.setKernelElf(kernelElf, elfSize);
    if (m_instanceWrapper.extractElf(m_name) != TpcElfTools::TpcElfStatus::TPC_ELF_SUCCESS)
    {
        HB_ASSERT(0, "{}: {} Failed to extract binary from Elf", __FUNCTION__, m_name);
    }
    // uniqueId is used in allocateTpcKernels, a unique node should have a unique identifier
    m_uniqueID = crc32(m_instanceWrapper.getKernelBinary(), getKernelSize());
}

void TPCNode::setDoubleStore(void* kernelElf, uint32_t elfSize, unsigned tensorIdx, bool isInput)
{
    setKernelElf(kernelElf, elfSize);
    addDoubleStoreAccessPattern(tensorIdx, isInput);
}

void TPCNode::resetInstantiated()
{
    m_instanceWrapper.resetParams();
    m_instanceWrapper.setInstantiated(false);
}

bool TPCNode::isFusedKernel() const
{
    return ::isFusedKernel(getGUID());
}

void TPCNode::setAllowedForStitching(bool allowedForStitching)
{
    m_isAllowedForStitching = allowedForStitching;
}

bool TPCNode::isAllowedForStitching(const HabanaGraph& graph) const
{
    const auto& memoryCoherence = graph.getGraphAnnotation().memoryCoherence;
    if (memoryCoherence)
    {
        for (const auto& tensor : getOutputs())
        {
            if (memoryCoherence->overlapsWithOthersInSection(tensor))
            {
                LOG_DEBUG(SRAM_SLICE,
                        "Tensor {} has multiple producers in section and can't be stitched",
                        tensor->getName());
                return false;
            }
        }
    }

    if (!m_isAllowedForStitching.has_value())
    {
        const auto& allowedForStitchingList = getAllowedForStitchingList();
        m_isAllowedForStitching.emplace(allowedForStitchingList.find(getGUIDWithoutDtype()) != allowedForStitchingList.end());
    }
    return m_isAllowedForStitching.value();
}

bool TPCNode::isNode64BitCompatible() const
{
    return true;
}

unsigned TPCNode::getMaxAvailableTpc(const HalReader* reader)
{
    unsigned maxNumOfTPCs = reader->getNumTpcEngines();
    return countSetBits(GCFG_TPC_ENGINES_ENABLED_MASK.value() & reader->getTpcEnginesMask(), maxNumOfTPCs);
}

unsigned TPCNode::getMaxAvailableTpc(tpc_lib_api::DeviceId deviceId)
{
    if (!CompilationHalReader::isHalReaderSet())
    {
        unsigned maxNumOfTPCs =
            (deviceId == tpc_lib_api::DEVICE_ID_GAUDI2) ? 24 : ((deviceId == tpc_lib_api::DEVICE_ID_GAUDI3) ? 64 : 8);
        LOG_WARN(TPC_NODE, "Can't access halReader, setting maxNumOfTPCS to {} ", maxNumOfTPCs);
        return maxNumOfTPCs;
    }
    return TPCNode::getMaxAvailableTpc(CompilationHalReader::getHalReader().get());
}

unsigned TPCNode::getTotalNumDescriptors()
{
    unsigned totalDesc = isPrintfUsed() ? 1 : 0;
    for (const auto& op : getOperands())
    {
        totalDesc += TPCNode::numTensorDescriptors(*op);
    }
    return totalDesc;
}

// Not all tensors appear in the input/output access pattern arrays in the glue code return value. When traversing these
// arrays, use this method to advance.
unsigned TPCNode::numTensorGlueCodeAccessPatternEntries(const TensorPtr& tensor)
{
    if (tensor->isAuxTensor()) return 0;
    return numTensorDescriptors(*tensor);
}

void TPCNode::updateCache()
{
    if (isInstantiated())
    {
        Node::updateCache();
    }
    else
    {
        m_nodeAccessPatternCache.reset();
    }
}

bool TPCNode::isTranspose() const
{
    return Node::isTranspose() || getGUIDWithoutDtype() == "transpose";
}

bool TPCNode::canHandleStridedInput(synDeviceType device) const
{
    const bool isInBlockList = m_stridedBlockList.find(extractGUIDFromFullGUID(getGUID())) != m_stridedBlockList.end();

    return !isInBlockList;
}
