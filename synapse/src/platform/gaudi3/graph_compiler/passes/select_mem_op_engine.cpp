#include "code_generation/tensor_size_validator.h"
#include "compilation_hal_reader.h"
#include "memcpy_engine_manager.h"
#include "gaudi3_graph.h"
#include "types.h"
#include "habana_graph.h"
#include "data_type_utils.h"
#include "cast_nodes_handler.h"
#include "node_factory.h"

namespace gaudi3
{
class Gaudi3MemcpyEngineManager : public MemcpyEngineManager
{
public:
    virtual ~Gaudi3MemcpyEngineManager() = default;

protected:
    NodePtr
    getDefaultCopyNode(const TensorVector& inputs, const TensorVector& outputs, const std::string& name) const override;

private:
    NodePtr createConcreteSetNode(const HabanaGraph& graph, const NodePtr& semanticNode) const override;
    bool    validateOperandsSizes(const TensorVector& operands) const;
};

bool Gaudi3MemcpyEngineManager::validateOperandsSizes(const TensorVector& operands) const
{
    return std::all_of(operands.begin(), operands.end(), [](const TensorPtr& operand) {
        return !operand->isDataTensor() || TensorSizeValidator(static_cast<unsigned>(SPDLOG_LEVEL_TRACE))
                                               .validateTensor(operand,
                                                               operand->getNSizesInElements(),
                                                               operand->getNStridesInElements(),
                                                               DEVICE_TPC,
                                                               false);
    });
}

NodePtr Gaudi3MemcpyEngineManager::getDefaultCopyNode(const TensorVector& inputs,
                                                      const TensorVector& outputs,
                                                      const std::string&  name) const
{
    const bool isValidInputsSize  = validateOperandsSizes(inputs);
    const bool isValidOutputsSize = validateOperandsSizes(outputs);

    const bool shouldHandleWithIRF44 = !(isValidInputsSize && isValidOutputsSize);

    ns_Memcpy::Params memcpyParams;
    memcpyParams.mode = shouldHandleWithIRF44 ? MEMCPY_IRF44 : MEMCPY_DEFAULT;

    // Default concrete copy node for Gaudi3 is tpc memcpy
    HB_ASSERT(!CompilationHalReader::getHalReader()->isGcDmaSupported(), "Expecting GC to support DMA engine");

    const auto tpcMemcpy =
        NodeFactory::createNode(inputs, outputs, &memcpyParams, NodeFactory::tpcMemcpyNodeTypeName, name);
    HB_ASSERT_PTR(tpcMemcpy);
    return tpcMemcpy;
}

NodePtr Gaudi3MemcpyEngineManager::createConcreteSetNode(const HabanaGraph& graph, const NodePtr& semanticNode) const
{
    const auto elementType = semanticNode->getOutput(0)->getElementType();
    HB_ASSERT(graph.getHALReader()->isTPCMemsetSupportedDataType(elementType),
              "TPC memset not implemented for {}",
              getStringFromSynDataType(elementType));
    return createTpcMemset(semanticNode->getInputs(), semanticNode->getOutputs(), semanticNode->getNodeName());
}

bool selectMemcpyEngine(Gaudi3Graph& g)
{
    return Gaudi3MemcpyEngineManager().selectEngine(g);
}
}  // namespace gaudi3