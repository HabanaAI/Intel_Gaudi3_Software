#include "tpc_kernel_loader.h"

#include "code_generator.h"
#include "graph_editor.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "habana_nodes/node_factory.h"
#include "habana_nodes/tpc_node.h"
#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "node.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "utils.h"

#include <algorithm>

static const unsigned s_dataTypedInitializer[] = {synDataType::syn_type_int8,
                                                  synDataType::syn_type_int16,
                                                  synDataType::syn_type_int32,
                                                  synDataType::syn_type_uint8,
                                                  synDataType::syn_type_int4,
                                                  synDataType::syn_type_uint4,
                                                  synDataType::syn_type_uint16,
                                                  synDataType::syn_type_uint32,
                                                  synDataType::syn_type_int64,
                                                  synDataType::syn_type_uint64,
                                                  synDataType::syn_type_fp8_152,
                                                  synDataType::syn_type_fp8_143,
                                                  synDataType::syn_type_bf16,
                                                  synDataType::syn_type_fp16,
                                                  synDataType::syn_type_ufp16,
                                                  synDataType::syn_type_tf32,
                                                  synDataType::syn_type_hb_float,
                                                  synDataType::syn_type_single};

static std::map<unsigned, unsigned> s_dataTypesHierarchy {{s_dataTypedInitializer[0], 0},
                                                          {s_dataTypedInitializer[1], 0},
                                                          {s_dataTypedInitializer[2], 0},
                                                          {s_dataTypedInitializer[3], 0},
                                                          {s_dataTypedInitializer[4], 0},
                                                          {s_dataTypedInitializer[5], 0},
                                                          {s_dataTypedInitializer[6], 0},
                                                          {s_dataTypedInitializer[7], 0},
                                                          {s_dataTypedInitializer[8], 0},
                                                          {s_dataTypedInitializer[9], 0},
                                                          {s_dataTypedInitializer[10], 1},
                                                          {s_dataTypedInitializer[11], 2},
                                                          {s_dataTypedInitializer[12], 3},
                                                          {s_dataTypedInitializer[13], 3},
                                                          {s_dataTypedInitializer[14], 4},
                                                          {s_dataTypedInitializer[15], 5},
                                                          {s_dataTypedInitializer[16], 6},
                                                          {s_dataTypedInitializer[17], 6}};

TpcKernelLoader::TpcKernelLoader(HabanaGraph* graph)
: m_graph(graph),
  m_reRunTpcInit(false)
{
}

TpcKernelLoader::~TpcKernelLoader()
{
}

void assertInputDataTypes(const pNode& node, TensorPtr const& tensor, const unsigned tensorIndex)
{
    if (tensor == nullptr || !tensor->isDataTensor())
    {
        // the check only makes sense for data tensors
        return;
    }

    static_assert((1 << ((sizeof(s_dataTypedInitializer) / sizeof(s_dataTypedInitializer[0])) - 1)) + 1 ==
                      synDataType::syn_type_max,
                  "Missing syn data type; Update the table s_dataTypedInitializer");

    auto typeGivenByTensor  = s_dataTypesHierarchy.find(tensor->getElementType());
    auto typeRequiredByNode = s_dataTypesHierarchy.find(node->getRequiredInputType(tensorIndex));
    if (typeGivenByTensor == s_dataTypesHierarchy.end() || typeRequiredByNode == s_dataTypesHierarchy.end())
    {
        HB_ASSERT(false,
                  "Failed to find data type in the data type hierarchy table for node {} and input type: {} with "
                  "input tensor type:{} ",
                  node->getNodeName(),
                  node->getRequiredInputType(tensorIndex),
                  tensor->getElementType());
    }

    if (typeGivenByTensor->second > typeRequiredByNode->second)
    {
        LOG_ERR(GC,
                "During the compilation of a training graph, got a glue code error: INCOMPATIBLE_DATA_TYPE; "
                "Since the node's required data type is less precise than the given tensor type, "
                "no cast node will be planted to not lose precision.");
        HB_ASSERT(false,
                  "In training flow, the node {} has a required data type that is less precise than the data type "
                  "of the input tensor.",
                  node->getNodeName());
    }
}

void assertTypesCanBeCasted(const pNode& node)
{
    const TensorVector& inputs = node->getInputs();
    for (unsigned i = 0; i < inputs.size(); ++i)
    {
        assertInputDataTypes(node, inputs[i], i);
    }
}

bool TpcKernelLoader::tryRecover(const pNode& node, tpc_lib_api::GlueCodeReturn initRes)
{
    bool recoverSuccessful = false;
    bool tryRecover = GCFG_RECOVER_INCOMPATIBLE_DATA_TYPE.value();

    if (initRes == tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE && tryRecover)
    {
        assertTypesCanBeCasted(node);
        recoverSuccessful = m_castNodeHandler.createCastNodes(node, m_graph->getDeviceId());
        if (!(recoverSuccessful && m_castNodeHandler.getTotalCreatedCasts() != 0) && _isLastNode(node))
        {
            // Try to proceed with no cast
            recoverSuccessful = _adaptNodeOutputTensor(node);
        }
    }

    m_reRunTpcInit = recoverSuccessful;

    return recoverSuccessful;
}

bool TpcKernelLoader::finalizeRecover()
{
    m_castNodeHandler.plantCastNodes(*m_graph);
    return m_reRunTpcInit;
}

bool TpcKernelLoader::_adaptNodeOutputTensor(const std::shared_ptr<Node>& node) const
{
    bool        tensorDataTypeChanged = false;
    const auto& nodeOutputs           = node->getOutputs();
    for (unsigned int index = 0; index < nodeOutputs.size(); ++index)
    {
        synDataType requiredType = node->getRequiredOutputType(index);
        if (requiredType != nodeOutputs[index]->getElementType())
        {
            nodeOutputs[index]->setElementType(requiredType);
            tensorDataTypeChanged = true;
        }
    }
    return tensorDataTypeChanged;
}

bool TpcKernelLoader::load()
{
    return _loadFullGraph();
}

bool TpcKernelLoader::load(pNode& node)
{
    return _loadSingleNode(node);
}

bool TpcKernelLoader::allocate(bool bAddNOPKernel)
{
    return _allocateFullGraph(bAddNOPKernel);
}

bool TpcKernelLoader::allocate(TPCNode& node)
{
    return _allocateSingleNode(node);
}

bool TpcKernelLoader::_isLastNode(const std::shared_ptr<Node>& node) const
{
    for (const auto& output : node->getOutputs())
    {
        const auto consumers = m_graph->getTensorConsumers(output);
        for (const auto& consumer : consumers)
        {
            if (!consumer->isDma())
            {
                return false;
            }
        }
    }
    return true;
}

bool TpcKernelLoader::_loadFullGraph()
{
    _reset();

    NodeVector sortedNodes = m_graph->getExeSortedNodes();

    for (pNode node : sortedNodes)
    {
        if( !_loadAndRecover(node) )
        {
            return false;
        }
    }

    if (finalizeRecover())
    {
        // Cast nodes were added
        return _loadFullGraph();
    }

    LOG_DEBUG(GC,
              "{}: Aggregate binary size for TPC kernels: {} bytes",
              HLLOG_FUNC,
              m_graph->getCodeGenerator()->getKernelsBinarySize());

    m_graph->turnOnPredicate(PREDICATE_ID_TPC_NODE_INITIALIZED);

    return true;
}

bool TpcKernelLoader::_loadSingleNode(pNode& node)
{
    _reset();

    if( !_loadAndRecover(node) )
    {
        return false;
    }

    if (finalizeRecover())
    {
        // Cast nodes were added
        return _loadSingleNode(node);
    }

    LOG_DEBUG(GC,
              "{}: Aggregate binary size for TPC kernels: {} bytes",
              HLLOG_FUNC,
              m_graph->getCodeGenerator()->getKernelsBinarySize());

    return true;
}

bool TpcKernelLoader::_loadAndRecover(pNode& node)
{
    if (m_graph->runsOnTPC(node))
    {
        tpc_lib_api::GlueCodeReturn initRes = tpc_lib_api::GLUE_FAILED;
        TPCNode* tpcNode = dynamic_cast<TPCNode*>(node.get());
        HB_ASSERT(tpcNode != nullptr, "invalid node type");

        if (!tpcNode->isInstantiated())
        {
            GraphEditor::editNode(*m_graph, node, [&]() {
                initRes = tpcNode->init(deviceTypeToDeviceID(m_graph->getDeviceType()),
                                        &m_graph->getGraphAnnotation().cachedAuxiliaryTensors,
                                        m_graph->getNextTPCKernelUniqueId());
            });

            if (initRes != tpc_lib_api::GLUE_SUCCESS)
            {
                if (tryRecover(node, initRes))
                {
                    // Recover successful
                    return true;
                }
                LOG_ERR(GC,
                        "Failed to load tpc kernel for node: {}, GUID: {}. Got error: {}",
                        tpcNode->getNodeName(),
                        tpcNode->getGUID(),
                        KernelDB::parseReturnValue(initRes));
                return false; // should be throw
            }
        }
    }

    return true;
}

bool TpcKernelLoader::_allocateFullGraph(bool bAddNOPKernel)
{
    NodeVector sortedNodes = m_graph->getExeSortedNodes();

    for (pNode node : sortedNodes)
    {
        if (HabanaGraph::runsOnTPC(node))
        {
            auto& tpcNode = static_cast<TPCNode&>(*node);

            if (!_allocateSingleNode(tpcNode))
            {
                return false;
            }
        }
    }

    bool retVal = true;
    if (bAddNOPKernel)
    {
        retVal = _addNOPKernel();
    }
    return retVal;
}

bool TpcKernelLoader::_addNOPKernel()
{
    // Not using TPC nop if not initialized - Case of tests which not initialize kernelDB
    if (!KernelDB::instance().initialized()) return true;
    // create nop kernel and update its offset/section in graph so later it can be serialized
    const TSize inputSizes[] = {1};
    pTensor     inTensor     = std::make_shared<Tensor>(1, inputSizes, syn_type_float);

    pNode nopTPCNode = NodeFactory::createGenericTPCNode({inTensor}, {}, nullptr, TPCNode::NOP_KERNEL_NAME);
    std::unique_ptr<CodeGenerator>& codeGenerator = m_graph->getCodeGenerator();

    TPCNode* tpcNode = dynamic_cast<TPCNode*>(nopTPCNode.get());
    HB_ASSERT(tpcNode != nullptr, "invalid node type");

    bool res =
        tpcNode->init(deviceTypeToDeviceID(m_graph->getDeviceType()), nullptr, m_graph->getNextTPCKernelUniqueId());
    if (res)
    {
        LOG_ERR(GC, "Failed to init NOP kernel");
        return false;
    }

    kernelID kernelHash = tpcNode->getUniqueID();

    Settable<deviceAddrOffset> addr = allocateKernelWorkspace(tpcNode->getKernelSize());
    if (!addr.is_set())
    {
        m_graph->getGraphAnnotation().errors.memoryAllocationError = true;
        LOG_ERR(GC, "Failed to allocate tpc kernel: {}", tpcNode->getGUID());
        return false;
    }

    LOG_DEBUG(GC,
              "{} kernel with ID 0x{:x} and size {} allocated at address 0x{:x}",
              tpcNode->getGUID(),
              kernelHash,
              tpcNode->getKernelSize(),
              addr.value());

    tpcNode->registerKernelToCodeGen(codeGenerator, addr.value());
    codeGenerator->configNOPKernel(addr.value(), MEMORY_ID_RESERVED_FOR_PROGRAM_DATA, tpcNode->getKernelSize());

    return true;
}

bool TpcKernelLoader::_allocateSingleNode(TPCNode& tpcNode)
{
    kernelID                        kernelHash = tpcNode.getUniqueID();
    bool                            wasFound;
    std::unique_ptr<CodeGenerator>& codeGenerator = m_graph->getCodeGenerator();
    deviceAddrOffset                addr          = codeGenerator->getKernelAddress(kernelHash, wasFound);
    if (wasFound)
    {
        LOG_DEBUG(GC,
                  "{} kernel with ID 0x{:x} already allocated at address 0x{:x}",
                  tpcNode.getGUID(),
                  kernelHash,
                  addr);
    }
    else
    {
        LOG_DEBUG(GC,
                  "Allocating {} bytes for {} kernel with ID 0x{:x}",
                  tpcNode.getKernelSize(),
                  tpcNode.getGUID(),
                  kernelHash);
        Settable<deviceAddrOffset> addr = allocateKernelWorkspace(tpcNode.getKernelSize());
        if (!addr.is_set())
        {
            m_graph->getGraphAnnotation().errors.memoryAllocationError = true;
            LOG_ERR(GC, "Failed to allocate tpc kernel");
            return false;
        }
        tpcNode.setKernelOffsetInSection(addr.value());
        tpcNode.registerKernelToCodeGen(codeGenerator, addr.value());
    }

    const TensorPtr& printfTensor = tpcNode.getPrintfTensor();
    if (printfTensor != nullptr)
    {
        Settable<deviceAddrOffset> addr = allocatePrintfWorkspace(printfTensor->getTotalSizeInBytes());
        if (!addr.is_set())
        {
            m_graph->getGraphAnnotation().errors.memoryAllocationError = true;
            LOG_ERR(GC, "Failed to allocate printf tensor kernel");
            return false;
        }
        setPrintfTensorOffset(*printfTensor, addr.value());
        codeGenerator->addKernelPrintf(addr.value());
        codeGenerator->setUsingPrintf();
    }

    return true;
}

void TpcKernelLoader::_reset()
{
    m_reRunTpcInit = false;
    m_castNodeHandler.clear();
}

Settable<deviceAddrOffset> TpcKernelLoader::allocateKernelWorkspace(unsigned requiredSize)
{
    return m_graph->getCodeGenerator()->getAllocatorForProgramData().Allocate(requiredSize, m_graph->getHALReader()->getCacheLineSizeInBytes());
}

Settable<deviceAddrOffset> TpcKernelLoader::allocatePrintfWorkspace(unsigned requiredSize)
{
    if (m_printfBuffer == nullptr)
    {
        m_printfBuffer.reset(new char[GCFG_TPC_PRINTF_TENSOR_SIZE.value()], ArrayDeletor<char>());
        memset(m_printfBuffer.get(), 0, GCFG_TPC_PRINTF_TENSOR_SIZE.value());
    }
    Settable<deviceAddrOffset> addr = m_graph->getCodeGenerator()->getAllocatorForProgramData().Allocate(requiredSize, m_graph->getHALReader()->getCacheLineSizeInBytes());
    m_graph->getCodeGenerator()->registerProgramDataBlobForDownload(m_printfBuffer, addr.value(), requiredSize);
    return addr;
}

void TpcKernelLoader::setPrintfTensorOffset(Tensor& printfTensor, deviceAddrOffset addr)
{
    LOG_DEBUG(GC,"kernel printf tensor allocated at virtual addr 0x{:x}, size {}", addr, printfTensor.getTotalSizeInBytes());
    printfTensor.setDramOffset(addr);
}
