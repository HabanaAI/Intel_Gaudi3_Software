#pragma once

#include "tpc_fuser.h"
#include "kernel_db.h"
#include "ir_translation/ir_to_synapse_translator_defs.hpp"

class HabanaGraph;
class IRToSynapseTranslator;
class EagerIRToSynapseTranslator;
class SynapseNodeWrapper;

enum ComplexGUIDExtractorModes
{
    ComplexGUIDExtractorModeDisabled,
    ComplexGUIDExtractorModeEnabled,
    ComplexGUIDExtractorModeEnabledDummy
};

enum ComplexGUIDType
{
    FUNCTIONAL_COMPLEX_GUID,
    PERFORMANCE_COMPLEX_GUID,
};

class ComplexGuidExtractorSharedObject
{
private:
    friend class KernelDB;
    void init();
    void destroy();

public:
    static ComplexGuidExtractorSharedObject& instance();

    bool                        isInitialized() const { return m_isInitialized; };
    tpc_lib_api::GlueCodeReturn extractFunctionalComplexGuid(const ProtocolGraph* graphIn,
                                                             ProtocolGraph**      graphOut) const;
    tpc_lib_api::GlueCodeReturn extractPerformanceComplexGuid(const ProtocolGraph* graphIn,
                                                              ProtocolGraph**      graphOut) const;

private:
    bool        m_isInitialized {false};
    std::string m_complexGuidExtractorLibName;
    std::string m_dummyLibName;
    // function pointers to complex GUID methods
    pfnExtractFunctionalComplexGUID  m_pfnExtractPerformanceComplexGuid {nullptr};
    pfnExtractPerformanceComplexGUID m_pfnExtractFunctionalComplexGuid {nullptr};
};

namespace
{
using FuserSectionId = uint32_t;
using GCSectionId    = uint32_t;
using FuserTensorId  = uint32_t;
using GCTensorId     = uint32_t;
using FuserNodeId    = uint32_t;
}  // namespace

class ComplexGuidExtractor
{
public:
    ComplexGuidExtractor(tpc_lib_api::DeviceId deviceID, ComplexGUIDType type);
    ~ComplexGuidExtractor();

    bool isNodeNeedsExtract(const NodePtr& node) const;
    bool isComplexGuidExtractorEnabled() const;
    bool canRunComplexGuidExtractor() const;
    bool validateExtractedGraph(const ir_translation_defs::NewTensorMap&) const;
    bool handleInternalControlDependencies(HabanaGraph& g, const IRToSynapseTranslator& translator) const;
    void clearProtocolGraphData();
    // return code is used to identify early exists (not only failures)
    tpc_lib_api::GlueCodeReturn extractComplexGuid(const SynapseNodeWrapper& nodeWrapper);
    const ProtocolGraph*        getExtractedGraph() { return m_extractedGraph; }
    void                        flashAttentionNodePreExtract(HabanaGraph& g, const NodePtr& node);
    void                        flashAttentionNodePostExtract(HabanaGraph& g);

protected:
    ComplexGuidExtractorSharedObject& m_sharedObject;
    // A pointer to the extraction function (member function of ComplexGuidExtractorSharedObject)
    tpc_lib_api::GlueCodeReturn (ComplexGuidExtractorSharedObject::*m_extractFunc)(const ProtocolGraph*,
                                                                                   ProtocolGraph**) const;
    // A pointer to the function that decides if a given guid is supported for the defined extractor type.
    bool (KernelDB::*m_isSupportedFunc)(const StringWithHash&, tpc_lib_api::DeviceId) const;
    // Extracted graph by CGUID lib
    ProtocolGraph*              m_extractedGraph = nullptr;
    const tpc_lib_api::DeviceId m_deviceID;
};