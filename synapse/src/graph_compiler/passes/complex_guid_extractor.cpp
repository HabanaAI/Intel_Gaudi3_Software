#include "complex_guid_extractor.h"
#include "flash_attention_scheduler.h"
#include "flash_attention_slicer.h"
#include "habana_graph.h"
#include "habana_global_conf.h"
#include "utils.h"
#include "synapse_graph_wrapper.hpp"
#include "ir_to_synapse_translator.hpp"
#include "gc_protocol_utils.hpp"
#include "physical_memory_ops_nodes.h"

/*
 * ComplexGuidExtractorSharedObject
 */

void ComplexGuidExtractorSharedObject::init()
{
    if (!m_isInitialized)
    {
        LOG_DEBUG(GC_COMPLEX_GUID, "Loading complex GUID extractor shared object");
        uint64_t mode = GCFG_COMPLEX_GUID_EXTRACTOR_MODE.value();
        if (mode == ComplexGUIDExtractorModeDisabled || mode == ComplexGUIDExtractorModeEnabled)
        {
            // Extraction logic and entry points are part of tpcFuser lib
            m_complexGuidExtractorLibName = GCFG_COMPLEX_GUID_LIB_NAME.value();
        }
        else if (mode == ComplexGUIDExtractorModeEnabledDummy)
        {
            // TODO: what if the env is not present and it's nullptr??
            m_dummyLibName                = fmt::format("{}/libdummyComplexGuid.so", getenv("BUILD_ROOT_LATEST"));
            m_complexGuidExtractorLibName = m_dummyLibName;
        }
        else
        {
            LOG_ERR(GC_COMPLEX_GUID, "Unknown complex GUID extractor mode - {}. failed to init.", mode);
            return;
        }
        // handle is acquired in the destructor using dlopen with appropriate
        // flags, so there is no resource leak of handle.
        void* handle = LoadSharedObject(m_complexGuidExtractorLibName.c_str());
        if (!handle)
        {
            LOG_ERR(GC_COMPLEX_GUID,
                    "Cannot load complex GUID extractor lib {}, error: {}",
                    m_complexGuidExtractorLibName,
                    dlerror());
            return;
        }
        // Load the symbols
        *(void**)(&m_pfnExtractFunctionalComplexGuid) = GetFunction(handle, COMPLEX_GUID_FUNCTIONAL_ENTRY_POINT_NAME);
        *(void**)(&m_pfnExtractPerformanceComplexGuid) = GetFunction(handle, COMPLEX_GUID_PERFORMANCE_ENTRY_POINT_NAME);
        if (!(m_pfnExtractFunctionalComplexGuid && m_pfnExtractPerformanceComplexGuid))
        {
            LOG_ERR(GC_COMPLEX_GUID, "Failed loading Complex GUID (protocol-IR) lib entry point.");
            UnloadSharedObject(handle);
            return;
        }
        LOG_DEBUG(GC_COMPLEX_GUID, "Complex GUID extractor lib entry points were loaded successfully.");
        m_isInitialized = true;
    }
    LOG_DEBUG(GC_COMPLEX_GUID, "Complex GUID extractor shared object is initialized.");
}

ComplexGuidExtractorSharedObject& ComplexGuidExtractorSharedObject::instance()
{
    static ComplexGuidExtractorSharedObject complexGuidExtractorSharedObject;

    return complexGuidExtractorSharedObject;
}

void ComplexGuidExtractorSharedObject::destroy()
{
    if (!m_isInitialized) return;
    // test if the object is loaded.
    void* handle = dlopen(m_complexGuidExtractorLibName.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle)
    {
        UnloadSharedObject(handle);
    }
    m_isInitialized = false;
}

tpc_lib_api::GlueCodeReturn
ComplexGuidExtractorSharedObject::extractFunctionalComplexGuid(const ProtocolGraph* graphIn,
                                                               ProtocolGraph**      graphOut) const
{
    HB_ASSERT(m_pfnExtractFunctionalComplexGuid != nullptr, "functional CGUID extract function is not initialized");
    auto ret = m_pfnExtractFunctionalComplexGuid(graphIn, graphOut);
    LOG_DEBUG(GC_COMPLEX_GUID, "functional complex GUID extract function return code - {}", ret);
    return ret;
}

tpc_lib_api::GlueCodeReturn
ComplexGuidExtractorSharedObject::extractPerformanceComplexGuid(const ProtocolGraph* graphIn,
                                                                ProtocolGraph**      graphOut) const
{
    HB_ASSERT(m_pfnExtractPerformanceComplexGuid != nullptr, "performance CGUID extract function is not initialized");
    auto ret = m_pfnExtractPerformanceComplexGuid(graphIn, graphOut);
    LOG_DEBUG(GC_COMPLEX_GUID, "performance complex GUID extract function return code - {}", ret);
    return ret;
}

/*
 * ComplexGuidExtractor
 */

ComplexGuidExtractor::ComplexGuidExtractor(tpc_lib_api::DeviceId deviceID, ComplexGUIDType type)
: m_sharedObject(ComplexGuidExtractorSharedObject::instance()), m_deviceID(deviceID)
{
    if (type == FUNCTIONAL_COMPLEX_GUID)
    {
        LOG_TRACE(GC_COMPLEX_GUID, "Using functional complex guid entry points");
        m_extractFunc     = &ComplexGuidExtractorSharedObject::extractFunctionalComplexGuid;
        m_isSupportedFunc = &KernelDB::isSupportedFunctionalComplexGuid;
    }
    else if (type == PERFORMANCE_COMPLEX_GUID)
    {
        LOG_TRACE(GC_COMPLEX_GUID, "Using performance complex guid entry points");
        m_extractFunc     = &ComplexGuidExtractorSharedObject::extractPerformanceComplexGuid;
        m_isSupportedFunc = &KernelDB::isSupportedPerformanceComplexGuid;
    }
    else
    {
        HB_ASSERT(false, "invalid Complex GUID type");
    }
}

ComplexGuidExtractor::~ComplexGuidExtractor()
{
    clearProtocolGraphData();
}

tpc_lib_api::GlueCodeReturn ComplexGuidExtractor::extractComplexGuid(const SynapseNodeWrapper& nodeWrapper)
{
    auto returnCode = (m_sharedObject.*m_extractFunc)(&nodeWrapper, &m_extractedGraph);
    if (returnCode == tpc_lib_api::GLUE_SUCCESS && m_extractedGraph == nullptr)
    {
        LOG_WARN(GC_COMPLEX_GUID, "Extraction was successful but extracted graph is empty!");
    }
    return returnCode;
}

bool ComplexGuidExtractor::isNodeNeedsExtract(const NodePtr& node) const
{
    // Cannot extract serialize and deserialize nodes
    if (std::dynamic_pointer_cast<SerializeNode<TPCMemcpyNode>>(node) != nullptr ||
        std::dynamic_pointer_cast<DeserializeNode<TPCMemcpyNode>>(node) != nullptr)
        return false;

    return ((KernelDB::instance()).*m_isSupportedFunc)(node->getGUID(), m_deviceID);
}

bool ComplexGuidExtractor::validateExtractedGraph(const ir_translation_defs::NewTensorMap& gcTensors) const
{
    if (!GCFG_COMPLEX_GUID_VALIDATE_EXTRACTED_GRAPH.value())
    {
        LOG_DEBUG(GC_COMPLEX_GUID, "CGUID extracted graph validation is disabled, skipping");
        return true;
    }
    LOG_DEBUG(GC_COMPLEX_GUID, "Verifying CGUID extracted nodes");
    ProtocolIRGraphValidator validator(m_extractedGraph, gcTensors);
    return validator.validateNodes() && validator.validateGraphInputsAndOutputs();
}

void ComplexGuidExtractor::clearProtocolGraphData()
{
    if (m_extractedGraph)
    {
        delete (m_extractedGraph);
        m_extractedGraph = nullptr;
    }
}

bool ComplexGuidExtractor::handleInternalControlDependencies(HabanaGraph&                 g,
                                                             const IRToSynapseTranslator& translator) const
{
    // handle internal ctrl dep set in the extracted graph
    const auto& blockingNodesMap = translator.getIrNodeIdsToBlockingNodesMap();
    for (const auto& [irNodeId, blockingNodesIds] : blockingNodesMap)
    {
        const NodePtr& gcNode = translator.getCreatedNodeFromIrId(irNodeId);
        if (blockingNodesIds.empty()) continue;
        LOG_TRACE(GC_COMPLEX_GUID, "Handling internal control dependencies for ir node with ID {}", irNodeId);
        for (auto blockingIrNodeId : blockingNodesIds)
        {
            auto it = blockingNodesMap.find(blockingIrNodeId);
            CHECK_RET_FALSE(it != blockingNodesMap.end(),
                            "Protocol node id {} is found in node {} control dependency set but doesn't have known id",
                            blockingIrNodeId,
                            irNodeId);
            g.addControlDependency(translator.getCreatedNodeFromIrId(blockingIrNodeId), gcNode);
        }
    }
    return true;
}

bool ComplexGuidExtractor::isComplexGuidExtractorEnabled() const
{
    return GCFG_COMPLEX_GUID_EXTRACTOR_MODE.value() != 0;
}

bool ComplexGuidExtractor::canRunComplexGuidExtractor() const
{
    if (!ComplexGuidExtractorSharedObject::instance().isInitialized())
    {
        LOG_WARN(GC_COMPLEX_GUID, "Complex GUID extractor is not initialized, can't run it");
        return false;
    }
    return isComplexGuidExtractorEnabled();
}

void ComplexGuidExtractor::flashAttentionNodePreExtract(HabanaGraph& g, const NodePtr& node)
{
    if (GCFG_ENABLE_FLASH_ATTENTION_MEMORY_ORIENTED_SCHEDULE.value())
    {
        if (node->getGUID().find("sdpa") != std::string::npos && !g.getGraphAnnotation().flashAttentionDb.isRegistered(node->getParentId()))
        {
            g.getGraphAnnotation().flashAttentionDb.registerId(node->getId());
        }
    }
}

// Remove FA parentId's that weren't sliced by cguid, because they don't need special handling by GC
void ComplexGuidExtractor::flashAttentionNodePostExtract(HabanaGraph& g)
{
    for (const auto& node : g.getNodes())
    {
        FlashAttentionScheduler::initFlashAttentionInfo(g, node);
    }
    g.getGraphAnnotation().flashAttentionDb.removeUnslicedFlashAttentionNodes();
}

/*
 * Passes functions.
 */

bool extractComplexGuidNodes(HabanaGraph& g, ComplexGUIDType type)
{
    ComplexGuidExtractor extractor(g.getDeviceId(), type);
    if (!extractor.canRunComplexGuidExtractor())
    {
        LOG_DEBUG(GC_COMPLEX_GUID, "Complex GUID extractor is disabled");
        return true;
    }
    auto             synToIrNodeWrapper = SynapseNodeWrapper(g.getDeviceId(), false /*isEagerMode*/);

    auto fn    = [&extractor](const NodePtr& node) { return extractor.isNodeNeedsExtract(node); };
    auto nodes = g.getNodesCond(fn);

    for (auto& node : nodes)
    {
        extractor.flashAttentionNodePreExtract(g, node);
        synToIrNodeWrapper.setNode(node);
        LOG_TRACE(GC_COMPLEX_GUID, "Extracting node {} using the complex GUID lib", node->getNodeName());
        auto extractionReturnCode = extractor.extractComplexGuid(synToIrNodeWrapper);
        if (extractionReturnCode == tpc_lib_api::GLUE_CGUID_GRAPH_UNCHANGED)
        {
            LOG_DEBUG(GC_COMPLEX_GUID, "Graph was unchanged, skipping extraction for node {}", node->getNodeName());
            continue;
        }
        else if (extractionReturnCode != tpc_lib_api::GLUE_SUCCESS)
        {
            LOG_ERR(GC_COMPLEX_GUID, "Extracting node {} failed", node->getNodeName());
            return false;
        }
        // Translate extracted graph
        auto protocolToSynapseTranslator = IRToSynapseTranslator(*extractor.getExtractedGraph());
        if (!protocolToSynapseTranslator.startNodeTranslationToSynapse(&g, node))
        {
            LOG_ERR(GC_COMPLEX_GUID, "Translation of CGUID graph extracted from node {} failed", node->getNodeName());
            return false;
        }
        if (!extractor.validateExtractedGraph(protocolToSynapseTranslator.getCreatedTensors()))
        {
            LOG_ERR(GC_COMPLEX_GUID, "Validation of graph extracted from node {} failed", node->getNodeName());
            return false;
        }
        if (!extractor.handleInternalControlDependencies(g, protocolToSynapseTranslator))
        {
            LOG_ERR(GC_COMPLEX_GUID, "Handling control dependencies created by complex GUID lib failed");
            return false;
        }
        extractor.clearProtocolGraphData();
        LOG_TRACE(GC_COMPLEX_GUID, "Finished Extracting node {}", node->getNodeName());
    }
    extractor.flashAttentionNodePostExtract(g);
    return true;
}

bool extractFunctionalComplexGuidNodes(HabanaGraph& g)
{
    FlashAttentionSlicer(g).sliceFlashAttentionNodes();  // Temp solution until it will be implemented in cguid

    bool res = extractComplexGuidNodes(g, FUNCTIONAL_COMPLEX_GUID);

    if (g.getDeviceType() != synDeviceGaudi)  // [SW-159339] WA - remove when root caused and fixed
    {
        // Complex guid node extraction can be resulted in nodes that are not being used in data paths consumed by the
        // user, for example that happened for the Linear CGUID in this ticket: [SW-154635]
        g.turnOnPredicate(PREDICATE_ID_ELIMINATE_REDUNDANT_NODES);
    }

    return res;
}

bool extractPerformanceComplexGuidNodes(HabanaGraph& g)
{
    if (auto flagVal = GCFG_COMPLEX_GUID_SKIP_PERF_PASS.value(); flagVal)
    {
        LOG_WARN(GC_COMPLEX_GUID, "Skipping CGUID performance pass, COMPLEX_GUID_SKIP_PERF_PASS = {}", flagVal);
        return true;
    }
    return extractComplexGuidNodes(g, PERFORMANCE_COMPLEX_GUID);
}