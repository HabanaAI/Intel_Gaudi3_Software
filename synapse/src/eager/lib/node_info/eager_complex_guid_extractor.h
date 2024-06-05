#pragma once

// eager includes (relative to src/eager/lib/)
#include "node_info/const_tensor_optimizer.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/passes/complex_guid_extractor.h"
#include "graph_compiler/passes/ir_translation/ir_to_synapse_translator.hpp"

#include <optional>

namespace eager_mode
{
class EagerIRToSynapseTranslator final : public IRToSynapseTranslatorBase
{
public:
    EagerIRToSynapseTranslator(const gc_protocol::ProtocolGraph& graphProvider,
                               ConstantTensorOptimizer&          constantTensorOptimizer)
    : IRToSynapseTranslatorBase(graphProvider), m_constantTensorOptimizer(constantTensorOptimizer) {};

    bool handleNode(const gc_protocol::ProtocolNode& node) override;
    // Iterates on graph tensors to store them before.
    bool startTranslationToSynapse(HabanaGraph* graph = nullptr) override;
    // Iterates only on node's inputs
    bool startNodeTranslationToSynapse(HabanaGraph* graph, const NodePtr& origNode) override;

protected:
    bool createGCNode(const gc_protocol::ProtocolNode& node, NodePtr& createdNode) override;
    void setGCTensorName(const gc_protocol::ProtocolTensor& irTensor, Tensor& gcTensor) override;

private:
    ConstantTensorOptimizer& m_constantTensorOptimizer;  // utility class to optimize out constant\cast
};

class EagerComplexGuidExtractor final : public ComplexGuidExtractor
{
public:
    EagerComplexGuidExtractor(tpc_lib_api::DeviceId deviceID, ConstantTensorOptimizer& constantTensorOptimizer)
    : ComplexGuidExtractor(deviceID, FUNCTIONAL_COMPLEX_GUID), m_constantTensorOptimizer(constantTensorOptimizer)
    {
    }
    // For eager mode, specific return codes for some cases (e.g. early exits).
    tpc_lib_api::GlueCodeReturn calcExtract(HabanaGraph* g, const NodePtr& node);
    const NodeVector&           extract()
    {
        EAGER_ASSERT(m_eagerProtocolToSynapseTranslator.has_value(), "m_eagerProtocolToSynapseTranslator is empty");
        return m_eagerProtocolToSynapseTranslator->getExecutionSortedNodes();
    }
    // This function is static for eager mode optimizations.
    static bool isNodeNeedsExtract(const NodePtr& node, ComplexGUIDType type, tpc_lib_api::DeviceId deviceId);

private:
    ConstantTensorOptimizer&                  m_constantTensorOptimizer;  // utility class to optimize out constant\cast
    std::optional<EagerIRToSynapseTranslator> m_eagerProtocolToSynapseTranslator;
};

}  // namespace eager_mode