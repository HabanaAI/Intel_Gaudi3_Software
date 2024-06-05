#include "serialize_utils.h"
#include "common_type_utils.h"
#include "data_type_utils.h"
#include "lz4/lz4.h"

#include "gaudi3_graph.h"

namespace graph_serializer
{
std::vector<uint64_t> getStrides(const Tensor* tensor)
{
    std::vector<uint64_t> strides(tensor->c_numOfNStrides);
    tensor->getNStridesInBytes(strides.data());
    strides.resize(tensor->getDim() + 1);
    if (tensor->isStridedOnFCD())
    {
        std::reverse(strides.begin(), strides.end());
    }
    return strides;
}

std::vector<uint8_t> getPermutation(const Tensor* tensor)
{
    const auto& perm = tensor->getPermutation();

    if (!perm) return {};

    const auto& data = perm->getValues();

    std::vector<uint8_t> ret;
    ret.reserve((perm->size()));
    for (const auto& v : data)
    {
        ret.emplace_back(v);
    }
    return ret;
}

std::vector<char> getConstData(const Tensor* tensor)
{
    std::vector<char> ret;
    const char*       data         = tensor->getData();
    const uint64_t    dataSize     = tensor->getBufferSizeInBytes();
    int               maxDstSize   = 0;
    int               compDataSize = 0;

    if (dataSize <= std::numeric_limits<int>::max())
    {
        maxDstSize = LZ4_compressBound(dataSize);
        if (maxDstSize > 0)
        {
            ret.resize(maxDstSize);
            compDataSize = LZ4_compress_default(data, ret.data(), dataSize, maxDstSize);
        }
    }

    // Avoid compression in case of failure.
    if (compDataSize <= 0 || dataSize <= compDataSize)
    {
        ret = std::vector<char>(data, data + dataSize);
    }
    else
    {
        ret.resize(compDataSize);
    }

    return ret;
}

void serializeQuantParams(nlohmann_hcl::json& tensor, const Tensor* t)
{
    uint64_t tensorProperty = synTensorPropUnknown;

    if (t->isPropSet(synTensorPropDynamicRange))
    {
        DynamicRange drange  = t->getDynamicRange();
        tensor["drange_min"] = drange.min;
        tensor["drange_max"] = drange.max;

        tensorProperty |= synTensorPropDynamicRange;
    }
    if (t->isPropSet(synTensorPropPCDynamicRange))
    {
        auto                pcDynamicRange = t->getPerChannelDynamicRange();
        unsigned            numChannels    = pcDynamicRange.numChannels;
        auto                ranges         = pcDynamicRange.ranges;
        std::vector<double> pcDrangeMinRanges;
        std::vector<double> pcDrangeMaxRanges;

        for (unsigned i = 0; i < numChannels; i++)
        {
            pcDrangeMinRanges.push_back(ranges[i].min);
            pcDrangeMaxRanges.push_back(ranges[i].max);
        }

        tensor["pc_drange_min_ranges"]   = pcDrangeMinRanges;
        tensor["pc_drange_max_ranges"]   = pcDrangeMaxRanges;
        tensor["pc_drange_num_channels"] = numChannels;

        tensorProperty |= synTensorPropPCDynamicRange;
    }
    if (t->isPropSet(synTensorPropQuantMetadata))
    {
        QuantizationData userMetadata = t->getQuantizationParams();

        tensor["quant_md_data_type"] = userMetadata.getSynDataType();
        uint64_t zpSize              = userMetadata.getZpVector().size();
        tensor["quant_md_zp_data"]   = userMetadata.getZpVector();
        tensor["quant_md_zp_size"]   = zpSize;

        uint64_t scaleSize            = userMetadata.getScaleVector().size();
        tensor["quant_md_scale_data"] = userMetadata.getScaleVector();
        tensor["quant_md_scale_size"] = scaleSize;

        tensorProperty |= synTensorPropQuantMetadata;
    }
    if (t->isPropSet(synTensorPropFpQuantMetadata))
    {
        QuantizationData userMetadata = t->getQuantizationParams();

        tensor["quant_md_data_type"] = userMetadata.getSynDataType();

        uint64_t scaleSize            = userMetadata.getScaleVector().size();
        tensor["quant_md_scale_data"] = userMetadata.getScaleVector();
        tensor["quant_md_scale_size"] = scaleSize;

        uint64_t expBiasSize            = userMetadata.getExpBiasVector().size();
        tensor["quant_md_expBias_data"] = userMetadata.getExpBiasVector();
        tensor["quant_md_expBias_size"] = expBiasSize;

        tensorProperty |= synTensorPropFpQuantMetadata;
    }
    if (t->isPropSet(synTensorPropFlags))
    {
        tensor["quant_flags_enable_per_channel_quant"] = t->isPerChannelQuant();
        tensor["quant_flags_is_sparsified_weights"]    = t->isSparsityWeights();
        tensor["quant_flags_is_weights"]               = t->isWeights();

        tensorProperty |= synTensorPropFlags;
    }

    tensor["tensor_property"] = tensorProperty;
}

// TODO: move to synapse/src/graph_serialize/graph_serializer.cpp when removing DUMP_PRE_GRAPHS
nlohmann_hcl::json serializeTensor(const Tensor* t)
{
    nlohmann_hcl::json tensor;
    synTensorType      tensorType = t->getTensorType();

    tensor["graph_index"] = t->getGraphID();
    tensor["name"]        = t->getName();
    tensor["type"]        = tensorTypeToString(tensorType);
    tensor["dtype"]       = std::string(getStringFromSynDataType(t->getElementType()));
    tensor["is_const"]    = t->isStaticParam();
    tensor["persistent"]  = t->isPersistent();
    tensor["rmw_section"] = t->isPartOfRMWSection();
    tensor["external"]    = t->getTensorIsExternal();

    NSizeArray maxShape = t->getAllNSizesInElements();
    NSizeArray minShape = t->getNMinimalSizesInElements();
    tensor["max_shape"] = std::vector<TSize>(maxShape.begin(), maxShape.begin() + t->getDim());
    tensor["min_shape"] = std::vector<TSize>(minShape.begin(), minShape.begin() + t->getDim());

    tensor["strides"]     = graph_serializer::getStrides(t);
    tensor["permutation"] = graph_serializer::getPermutation(t);
    tensor["allow_permutation"] = t->getTensorAnnotation().memory.allowPermutation;

    if (t->isPersistent())
    {
        tensor["user_mem_offset"]        = t->getMemorySectionOffset();
        tensor["user_mem_section_index"] = {t->getMemorySectionID()};
    }
    else if (t->isPartOfRMWSection())
    {
        const auto& sectionInfo          = t->getTensorAnnotation().nonPersistentSectionInfo;
        tensor["user_mem_offset"]        = sectionInfo.offsetFromBase.value();
        tensor["user_mem_section_index"] = {sectionInfo.sectionId.value()};
    }

    if (t->inConstSection())
    {
        tensor["is_const_section"] = true;
    }

    if (t->isStaticParam() || t->inConstSection())
    {
        auto data = graph_serializer::getConstData(t);
        if (data.size() < t->getBufferSizeInBytes())
        {
            tensor["comp_type"] = "lz4";
        }
        tensor["data"] = std::move(data);
    }

    if (tensorType == HOST_SHAPE_TENSOR || tensorType == HOST_TO_DEVICE_TENSOR)
    {
        const char*    data     = t->getData();
        const uint64_t dataSize = t->getBufferSizeInBytes();
        tensor["data"]          = std::vector<char>(data, data + dataSize);
    }

    serializeQuantParams(tensor, t);

    return tensor;
}

std::set<std::string> getNodeNames(const NodeSet& nodes)
{
    std::set<std::string> ret;
    for (const NodePtr& n : nodes)
    {
        if (n)
        {
            ret.insert(n->getNodeName());
        }
    }
    return ret;
}

std::string getPerforationDebugInfo(const NodePtr& node)
{
    const auto& rois = node->getNodeAnnotation().m_dcoreROIs;
    if (rois.size() <= 1) return "Monolithic";
    auto ap = node->getNodeAccessPattern();
    if (!ap) return "Monolithic";

    const auto& indexSpace = ap->getNodeResolution();

    gc::access_pattern::NodeTile roi0(indexSpace.size());
    gc::access_pattern::NodeTile roi1(indexSpace.size());
    std::copy_n(rois[0].size, indexSpace.size(), roi0.geometry.begin());
    std::copy_n(rois[1].size, indexSpace.size(), roi1.geometry.begin());
    std::copy_n(rois[0].baseOffset, indexSpace.size(), roi0.offset.begin());
    std::copy_n(rois[1].baseOffset, indexSpace.size(), roi1.offset.begin());

    std::vector<size_t> splittedInputs {};
    for (size_t inIdx = 0; inIdx < node->getNumInputs(); inIdx++)
    {
        const auto& input = node->getInput(inIdx);
        if (!input) continue;
        auto tensorTile0 = ap->getTensorTile(input, roi0);
        auto tensorTile1 = ap->getTensorTile(input, roi1);
        if (tensorTile0.offset != tensorTile1.offset) splittedInputs.push_back(inIdx);
    }
    std::vector<size_t> splittedOutputs {};
    for (size_t outIdx = 0; outIdx < node->getNumOutputs(); outIdx++)
    {
        const auto& output = node->getOutput(outIdx);
        if (!output) continue;
        auto tensorTile0 = ap->getTensorTile(output, roi0);
        auto tensorTile1 = ap->getTensorTile(output, roi1);
        if (tensorTile0.offset != tensorTile1.offset) splittedOutputs.push_back(outIdx);
    }

    return fmt::format("ROIs: {}, Splitted inputs: [{}], Splitted outputs: [{}]",
                       rois.size(),
                       fmt::join(splittedInputs.begin(), splittedInputs.end(), ","),
                       fmt::join(splittedOutputs.begin(), splittedOutputs.end(), ","));
}

std::vector<std::string> getMmeRecipeDebugInfo(const HabanaGraph& graph, const NodePtr& node)
{
    if (!graph.runsOnMME(node)) return {};

    Gaudi3Graph* gaudi3Graph = dynamic_cast<Gaudi3Graph*>(const_cast<HabanaGraph*>(&graph));
    if (!gaudi3Graph) return {};

    auto& descGen = gaudi3Graph->getMmeNodeDescriptorGenerator(node);
    return descGen.getRecipeDebugInfo(true);
}

}  // namespace graph_serializer
