#include "graph_compiler/habana_nodes/node_factory.h"
#include "habana_global_conf.h"
#include "tpc_node.h"
#include "data_type_utils.h"
#include "quantizer_factory.h"

QuantizerFactory& QuantizerFactory::getInstance()
{
    static QuantizerFactory instance;
    return instance;
}

QuantizerFactory::QuantizerFactory()
{
    m_defaultQunatizer = std::make_shared<Quantizer>();
    // backward
    m_quantizersMap.emplace(NodeFactory::concatenateNodeTypeName, std::make_shared<BackwardQuantizer>());
    // boolean output
    m_quantizersMap.emplace("less", std::make_shared<BooleanOutputQuantizer>());
    m_quantizersMap.emplace("equal", std::make_shared<BooleanOutputQuantizer>());
    m_quantizersMap.emplace("greater", std::make_shared<BooleanOutputQuantizer>());

    //boolean input
    m_quantizersMap.emplace(NodeFactory::andNodeTypeName, std::make_shared<BooleanInputQuantizer>());
    m_quantizersMap.emplace(NodeFactory::notNodeTypeName, std::make_shared<BooleanInputQuantizer>());
    m_quantizersMap.emplace(NodeFactory::orNodeTypeName, std::make_shared<BooleanInputQuantizer>());
    m_quantizersMap.emplace(NodeFactory::xorNodeTypeName, std::make_shared<BooleanInputQuantizer>());

    // forward
    m_quantizersMap.emplace("take", std::make_shared<ForwardQuantizer>(0));
    m_quantizersMap.emplace(NodeFactory::splitNodeTypeName, std::make_shared<ForwardQuantizer>());
    m_quantizersMap.emplace(NodeFactory::sliceNodeTypeName, std::make_shared<ForwardQuantizer>());

    // don't care
    m_quantizersMap.emplace(NodeFactory::reluNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::transposeNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::reshapeNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::expandDimsNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::flattenNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::upsampleNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::clipNodeTypeName, std::make_shared<DontCareQuantizer>());
    m_quantizersMap.emplace(NodeFactory::maxPool2dNodeTypeName, std::make_shared<DontCareQuantizer>());

    // special
    m_quantizersMap.emplace("cast", std::make_shared<CastQuantizer>());
    m_quantizersMap.emplace(NodeFactory::beamSearchNodeTypeName, std::make_shared<TopKQuantizer>());
    m_quantizersMap.emplace(NodeFactory::sequenceReverseNodeTypeName, std::make_shared<SequenceReverseQuantizer>());
    m_quantizersMap.emplace(NodeFactory::sequenceLengthNodeTypeName, std::make_shared<SequenceLengthQuantizer>());
    m_quantizersMap.emplace(NodeFactory::embeddingNodeTypeName, std::make_shared<EmbeddingQuantizer>());
}

const QuantizerPtr& QuantizerFactory::getDefaultNodeQuantizer()
{
    return QuantizerFactory::getInstance().m_defaultQunatizer;
}

const QuantizerPtr& QuantizerFactory::getNodeQuantizer(const StringViewWithHash& guidWithoutDType)
{
    const auto& instance      = QuantizerFactory::getInstance();
    const auto& quantizersMap = instance.m_quantizersMap;
    auto        iter          = quantizersMap.find(guidWithoutDType);
    if (iter == quantizersMap.end())
    {
        return instance.m_defaultQunatizer;
    }
    return iter->second;
}
