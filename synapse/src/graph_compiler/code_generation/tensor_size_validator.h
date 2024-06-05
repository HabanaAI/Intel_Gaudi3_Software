#pragma once
#include "habana_graph.h"
#include "compilation_hal_reader.h"

// A class for validating the tensors size according to each engine for the codeGeneration
// All the specs and rules can be found in confluence page 'CodeGen Tensors Size Validation'
class TensorSizeValidator
{
public:
    explicit TensorSizeValidator(unsigned logLevel = SPDLOG_LEVEL_ERROR)
    : m_halReader(CompilationHalReader::getHalReader()), m_logLevel(logLevel)
    {
    }

    TensorSizeValidator(const HabanaGraph& graph, unsigned logLevel = SPDLOG_LEVEL_ERROR)
    : m_halReader(graph.getHALReader()), m_logLevel(logLevel)
    {
    }
    bool validateTensors(const HabanaGraph& graph) const;

    // Validate that tensor with potential sizes and strides fit to the node,
    // since some nodes are not run on specific engine (i.e. multi nodes) it possible to specify the node engine,
    // so using the default value of "engineType" is possible only for nodes that tied with an engine type (i.e. gemm).
    bool validateTensor(const NodePtr&      node,
                        const TensorPtr&    tensor,
                        const NSizeArray&   sizes,
                        const NStrideArray& strides,
                        HabanaDeviceType    engineType = HabanaDeviceType::LAST_HABANA_DEVICE) const;

    bool validateTensor(const TensorPtr&    tensor,
                        const NSizeArray&   sizes,
                        const NStrideArray& strides,
                        HabanaDeviceType    engineType,
                        bool                isIRF44 = false /* relevant only for tpc engine type */) const;

private:
    void logTensorSizeOverflow(const TensorPtr& tensor, unsigned dim, const uint64_t offset) const;
    void logTensorSizeOverflow(const TensorPtr& tensor,
                               TSize            size,
                               TStride          stride,
                               unsigned         dim,
                               const uint64_t   offset) const;

    bool validatePerSizeAndStrideForDMA(const TensorPtr&  tensor,
                                        const NSizeArray& sizes,
                                        const TStride*    strides,
                                        const uint64_t    maxRegVal) const;
    bool validatePerSizeAndStrideForMME(const TensorPtr& tensor, const uint64_t maxRegVal) const;
    bool validatePerSizeAndStrideForMME(const TensorPtr& tensor,
                                        const TStride*   sizes,
                                        const TStride*   strides,
                                        const uint64_t   maxRegVal) const;

    bool validateDMANodeTensors(const TensorROIVector& tensorRois, const uint64_t maxRegVal) const;
    bool validateMMENodeTensors(const TensorVector& tensors) const;
    bool validateTPCNodeTensors(const TensorVector& tensors, bool isIRF44) const;

    bool validateTensorIRF44ModeForTPC(const TensorPtr& tensor, const bool supportIRF44 = false) const;
    bool validateTensorIRF44ModeForTPC(const TensorPtr& tensor,
                                       const TStride*   sizes,
                                       const TStride*   strides,
                                       const bool       supportIRF44 = false) const;
    bool validateTensorForTPC(const TensorPtr& tensor) const;
    bool validateTensorForTPC(const TensorPtr& tensor, const TStride* sizes, const TStride* strides) const;

    const HalReaderPtr& m_halReader;
    unsigned            m_logLevel;
};
