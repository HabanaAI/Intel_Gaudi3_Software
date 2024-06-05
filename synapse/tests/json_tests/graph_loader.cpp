#include "graph_loader.h"
#include "base_test.h"
#include "habana_global_conf.h"
#include "hpp/syn_graph.hpp"
#include "hpp/syn_section.hpp"
#include "hpp/syn_tensor.hpp"
#include "json_utils.h"
#include "lz4/lz4.h"
#include "synapse_common_types.h"
#include <optional>

synDataType JsonGraphLoader::dataTypeFromString(const std::string& str)
{
    if (str == "int8" || str == "syn_type_int8") return syn_type_int8;
    if (str == "bf16" || str == "syn_type_bf16") return syn_type_bf16;
    if (str == "float32" || str == "syn_type_single") return syn_type_single;
    if (str == "int16" || str == "syn_type_int16") return syn_type_int16;
    if (str == "int32" || str == "syn_type_int32") return syn_type_int32;
    if (str == "uint8" || str == "syn_type_uint8") return syn_type_uint8;
    if (str == "int4" || str == "syn_type_int4") return syn_type_int4;
    if (str == "uint4" || str == "syn_type_uint4") return syn_type_uint4;
    if (str == "float16" || str == "fp16_t" || str == "syn_type_fp16") return syn_type_fp16;
    if (str == "uint16" || str == "syn_type_uint16") return syn_type_uint16;
    if (str == "uint32" || str == "syn_type_uint32") return syn_type_uint32;
    if (str == "hfloat8" || str == "fp8_143_t" || str == "syn_type_fp8_143") return syn_type_fp8_143;
    if (str == "float8" || str == "fp8_152_t" || str == "syn_type_fp8_152") return syn_type_fp8_152;
    if (str == "int64" || str == "syn_type_int64") return syn_type_int64;
    if (str == "uint64" || str == "syn_type_uint64") return syn_type_uint64;
    if (str == "invalid") return syn_type_na;
    throw std::runtime_error("Unknown data type string: " + str);
}

synTensorType JsonGraphLoader::tensorTypeFromString(const std::string& str)
{
    if (str == "DATA_TENSOR") return DATA_TENSOR;
    if (str == "OUTPUT_DESCRIBING_SHAPE_TENSOR") return OUTPUT_DESCRIBING_SHAPE_TENSOR;
    if (str == "INPUT_DESCRIBING_SHAPE_TENSOR") return INPUT_DESCRIBING_SHAPE_TENSOR;
    if (str == "DATA_TENSOR_DYNAMIC") return DATA_TENSOR_DYNAMIC;
    if (str == "DEVICE_SHAPE_TENSOR") return DEVICE_SHAPE_TENSOR;
    if (str == "HOST_SHAPE_TENSOR") return HOST_SHAPE_TENSOR;
    if (str == "HOST_TO_DEVICE_TENSOR") return HOST_TO_DEVICE_TENSOR;
    throw std::runtime_error("Invalid tensor type string: " + str);
}

std::vector<uint8_t> JsonGraphLoader::decompress(const std::vector<uint8_t>& src, int dstSize)
{
    std::vector<uint8_t> dst(dstSize);

    const int decompressedSize = LZ4_decompress_safe(reinterpret_cast<const char*>(src.data()),
                                                     reinterpret_cast<char*>(dst.data()),
                                                     src.size(),
                                                     dst.size());
    if (decompressedSize != dstSize)
    {
        throw std::runtime_error("Decompressed size: " + std::to_string(decompressedSize) +
                                 " is different than expected size: " + std::to_string(dstSize));
    }
    return dst;
}

JsonGraphLoader::JsonGraphLoader(syn::Context&                  ctx,
                                 synDeviceType                  deviceType,
                                 std::optional<CompilationMode> compilationMode,
                                 const nlohmann_hcl::json&      jsonGraph,
                                 const std::string&             constTensorsFilePath)
: m_jsonGraph(jsonGraph), m_constTensorsFilePath(constTensorsFilePath)
{
    // If compilationMode has value - force graph compilation in the specified compilation mode
    // Else - compile the graph according to the recording
    bool isEager = compilationMode.has_value() ? compilationMode == CompilationMode::Eager : this->isEager();

    m_graph = isEager ? std::make_unique<syn::GraphBase>(ctx.createEagerGraph(deviceType))
                      : std::make_unique<syn::GraphBase>(ctx.createGraph(deviceType));
    allocateTensors();
    loadGraphAttributes();
    generateModel();
}

JsonGraphLoader::~JsonGraphLoader() = default;

uint64_t JsonGraphLoader::getGroup() const
{
    return json_utils::get<uint64_t>(m_jsonGraph, "group", 0);
}

std::string JsonGraphLoader::getName() const
{
    return json_utils::get(m_jsonGraph, "name", std::string());
}

uint16_t JsonGraphLoader::getRecipeId() const
{
    return json_utils::get(m_jsonGraph, "recipe_id", -1);
}

bool JsonGraphLoader::isEager() const
{
    return json_utils::get<CompilationMode>(m_jsonGraph, "compilation_mode", CompilationMode::Graph) ==
           CompilationMode::Eager;
}

const syn::GraphBase& JsonGraphLoader::getGraph() const
{
    return *m_graph;
}

const std::map<std::string, syn::Tensor>& JsonGraphLoader::getTensors() const
{
    return m_tensors;
}

const nlohmann_hcl::json& JsonGraphLoader::getJsonGraph() const
{
    return m_jsonGraph;
};

void JsonGraphLoader::generateModel()
{
    std::map<std::string, syn::Node>              nodes;
    std::map<synNodeId, syn::Node>                nodesIds;
    std::map<synNodeId, std::vector<std::string>> blockingNodesMap;

    const auto& jsonNodes = json_utils::get(m_jsonGraph, "nodes");
    for (const auto& n : jsonNodes)
    {
        std::string                    name          = json_utils::get(n, "name");
        std::string                    guid          = json_utils::get(n, "guid");
        std::vector<uint8_t>           params        = json_utils::get(n, "params");
        std::optional<synRoundingMode> roundingMode  = json_utils::get_opt<synRoundingMode>(n, "rounding_mode");
        std::optional<bool>            deterministic = json_utils::get_opt<bool>(n, "deterministic");
        std::vector<syn::Tensor>       inputTensors;
        std::vector<syn::Tensor>       ouputTensors;
        for (const std::string& t : json_utils::get(n, "input_tensors", std::vector<std::string>()))
        {
            inputTensors.push_back(t.empty() ? syn::Tensor() : m_tensors.at(t));
        }
        for (const std::string& t : json_utils::get(n, "output_tensors", std::vector<std::string>()))
        {
            ouputTensors.push_back(t.empty() ? syn::Tensor() : m_tensors.at(t));
        }

        auto inputLayouts  = json_utils::get(n, "input_layouts", std::vector<std::string>());
        auto outputLayouts = json_utils::get(n, "output_layouts", std::vector<std::string>());

        auto blockingNodes = json_utils::get(n, "blocking_nodes", std::vector<std::string>());

        auto node = m_graph->createNode(inputTensors, ouputTensors, params, guid, name, inputLayouts, outputLayouts);

        if (deterministic.has_value())
        {
            m_graph->setDeterministic(node, deterministic.value());
        }

        if (roundingMode.has_value() && GCFG_ENABLE_ROUNDING_MODE_PLAYBACK.value())
        {
            m_graph->setRoundingMode(node, roundingMode.value());
        }

        nodes[name]             = node;
        nodesIds[node.handle()] = node;
        if (!blockingNodes.empty())
        {
            blockingNodesMap[node.handle()] = blockingNodes;
        }
    }

    for (auto& e : blockingNodesMap)
    {
        syn::Nodes blockingNodes;
        for (const auto& n : e.second)
        {
            blockingNodes.push_back(nodes.at(n));
        }
        m_graph->setNodeDependency(blockingNodes, {nodesIds.at(e.first)});
    }
}

void JsonGraphLoader::loadTensorQuantParams(const nlohmann_hcl::json& t, syn::Tensor& tensor)
{
    uint64_t tensorProperty = json_utils::get<uint64_t>(t, "tensor_property", 0);

    if (tensorProperty & synTensorPropDynamicRange)
    {
        double               drange_min = json_utils::get<double>(t, "drange_min", 2);
        double               drange_max = json_utils::get<double>(t, "drange_max", 1);
        synQuantDynamicRange drange {drange_min, drange_max};

        tensor.setQuantizationData(SYN_QUANT_DYNAMIC_RANGE, &drange, sizeof(synQuantDynamicRange));
    }

    if (tensorProperty & synTensorPropPCDynamicRange)
    {
        std::vector<double> pc_drange_min_ranges = json_utils::get(t, "pc_drange_min_ranges", std::vector<double>());
        std::vector<double> pc_drange_max_ranges = json_utils::get(t, "pc_drange_max_ranges", std::vector<double>());
        unsigned            numChannels          = json_utils::get(t, "pc_drange_num_channels");

        synPerChannelDynamicRange pcDynamicRange;
        pcDynamicRange.numChannels = numChannels;

        std::vector<synQuantDynamicRange> rangesArray;
        for (unsigned i = 0; i < numChannels; i++)
        {
            rangesArray.push_back({pc_drange_min_ranges[i], pc_drange_max_ranges[i]});
        }
        pcDynamicRange.ranges = rangesArray.data();

        tensor.setQuantizationData(SYN_QUANT_PC_DYNAMIC_RANGE, &pcDynamicRange, sizeof(synPerChannelDynamicRange));
    }

    if (tensorProperty & synTensorPropQuantMetadata)
    {
        synDataType quant_md_data_type = json_utils::get(t, "quant_md_data_type");
        uint64_t    zpSize             = json_utils::get(t, "quant_md_zp_size");
        if (zpSize > 0)
        {
            std::vector<double> zpData    = json_utils::get(t, "quant_md_zp_data", std::vector<double>());
            std::vector<double> scaleData = json_utils::get(t, "quant_md_scale_data", std::vector<double>());

            synQuantMetadata userMetadata;

            userMetadata.dataType    = quant_md_data_type;
            userMetadata.numZPScales = zpSize;

            synQuantZPScale* zpScale = new synQuantZPScale[zpSize];

            for (int i = 0; i < zpSize; i++)
            {
                zpScale[i].zp    = zpData[i];
                zpScale[i].scale = scaleData[i];
            }
            userMetadata.zpScales = zpScale;

            tensor.setQuantizationData(SYN_QUANT_METADATA, &userMetadata, sizeof(synQuantMetadata));
        }
    }

    if (tensorProperty & synTensorPropFpQuantMetadata)
    {
        synDataType quant_md_data_type = json_utils::get(t, "quant_md_data_type");
        uint64_t    scaleSize          = json_utils::get(t, "quant_md_scale_size");
        if (scaleSize > 0)
        {
            std::vector<double> scaleData   = json_utils::get(t, "quant_md_scale_data", std::vector<double>());
            std::vector<double> expBiasData = json_utils::get(t, "quant_md_expBias_data", std::vector<double>());

            synFpQuantMetadata userMetadata;

            userMetadata.dataType         = quant_md_data_type;
            userMetadata.numFpQuantParams = scaleSize;

            synFpQuantParam* fpQuantParam = new synFpQuantParam[scaleSize];

            for (int i = 0; i < scaleSize; i++)
            {
                fpQuantParam[i].scale   = scaleData[i];
                fpQuantParam[i].expBias = expBiasData[i];
            }
            userMetadata.fpQuantParams = fpQuantParam;

            tensor.setQuantizationData(SYN_FP_QUANT_METADATA, &userMetadata, sizeof(synFpQuantMetadata));
        }
    }

    if (tensorProperty & synTensorPropFlags)
    {
        bool flags_enable_per_channel_quant = json_utils::get(t, "quant_flags_enable_per_channel_quant", false);
        bool flags_is_sparsified_weights    = json_utils::get(t, "quant_flags_is_sparsified_weights", false);
        bool flags_is_weights               = json_utils::get(t, "quant_flags_is_weights", false);

        synQuantFlags userFlags {flags_enable_per_channel_quant, flags_is_sparsified_weights, flags_is_weights};

        tensor.setQuantizationData(SYN_QUANT_FLAGS, &userFlags, sizeof(synQuantFlags));
    }
}

void JsonGraphLoader::allocateTensors()
{
    const auto& tensors = json_utils::get(m_jsonGraph, "tensors");
    m_persistentSectionsMap.clear();
    m_nonPersistentSectionsMap.clear();

    auto dataProvider =
        m_constTensorsFilePath.empty()
            ? nullptr
            : std::make_unique<CapturedDataProvider>(m_constTensorsFilePath, getName(), getRecipeId(), getGroup());

    for (const auto& t : tensors)
    {
        std::string   name          = json_utils::get(t, "name");
        bool          isPersistent  = json_utils::get(t, "persistent");
        bool          isRmwSection  = json_utils::get(t, "rmw_section", false);
        bool          isConst       = json_utils::get(t, "is_const");
        bool          isExternal    = json_utils::get(t, "external", false);
        synDataType   dataType      = dataTypeFromString(json_utils::get(t, "dtype"));
        uint64_t      userMemOffset = json_utils::get<uint64_t>(t, "user_mem_offset", 0);
        synTensorType tensorType    = tensorTypeFromString(json_utils::get(t, "type"));

        std::vector<TSize>    maxShape         = json_utils::get(t, "max_shape");
        std::vector<TSize>    minShape         = json_utils::get(t, "min_shape");
        std::vector<unsigned> sectionsIndices  = json_utils::get(t, "user_mem_section_index", std::vector<unsigned>());
        std::vector<uint8_t>  permutation      = json_utils::get(t, "permutation", std::vector<uint8_t>());
        bool                  allowPermutation = json_utils::get(t, "allow_permutation", false);
        bool                  isConstSection   = json_utils::get(t, "is_const_section", false);

        m_tensors.emplace(name, m_graph->createTensor(tensorType, name));
        syn::Tensor& tensor = m_tensors.at(name);

        tensor.setExternal(isExternal);
        tensor.setAllowPermutation(allowPermutation);

        if (!permutation.empty())
        {
            tensor.setPermutation(permutation);
        }

        if (!maxShape.empty())
        {
            tensor.setGeometry(maxShape, synGeometryMaxSizes);
            tensor.setGeometry(minShape, synGeometryMinSizes);
            tensor.setDeviceDataType(dataType);
        }

        loadTensorQuantParams(t, tensor);

        if (isConst || isConstSection)
        {
            if (isConstSection)
            {
                syn::Section section = getSection(sectionsIndices, isPersistent, isRmwSection, isConstSection);
                tensor.assignToSection(section, userMemOffset);
            }

            std::vector<uint8_t> constData = json_utils::get(t, "data", std::vector<uint8_t>());
            if (constData.empty() && dataProvider)
            {
                constData = dataProvider->getBuffer(name);
            }

            uint64_t tensorSize = tensor.getSizeInBytes();
            uint64_t dataSize   = constData.size();
            if (dataSize > 0 && dataSize < tensorSize)
            {
                constData = decompress(constData, tensorSize);
            }
            if (constData.empty())
            {
                std::vector<uint8_t> buffer(tensorSize, 0);
                tensor.setHostPtr(buffer, true);
            }
            else
            {
                tensor.setHostPtr(constData, true);
            }
            continue;
        }

        if (tensorType == HOST_SHAPE_TENSOR || tensorType == HOST_TO_DEVICE_TENSOR)
        {
            std::vector<uint8_t> data = json_utils::get(t, "data", std::vector<uint8_t>());
            if (data.empty())
            {
                throw std::runtime_error(fmt::format("host tensor: {} is missing required data", name));
            }
            tensor.setHostPtr(data, true);
        }

        if (!isPersistent && !isRmwSection)
        {
            continue;
        }

        if (sectionsIndices.empty())
        {
            syn::Section section = createSection(isPersistent, isRmwSection, isConstSection);
            tensor.assignToSection(section, userMemOffset);
            continue;
        }

        syn::Section section = getSection(sectionsIndices, isPersistent, isRmwSection, isConstSection);
        tensor.assignToSection(section, userMemOffset);
    }
}

syn::Section JsonGraphLoader::createSection(bool isPersistent, bool isRmwSection, bool isConstSection)
{
    syn::Section section = m_graph->createSection();
    section.setPersistent(isPersistent);
    section.setRMW(isRmwSection);
    section.setConst(isConstSection);
    return section;
}

syn::Section JsonGraphLoader::getSection(std::vector<unsigned>& sectionsIndices,
                                         bool                   isPersistent,
                                         bool                   isRmwSection,
                                         bool                   isConstSection)
{
    uint32_t sectionIndex = sectionsIndices.front();
    auto&    sectionMap   = isPersistent ? m_persistentSectionsMap : m_nonPersistentSectionsMap;
    auto     it           = sectionMap.find(sectionIndex);
    if (it == sectionMap.end())
    {
        syn::Section section     = createSection(isPersistent, isRmwSection, isConstSection);
        sectionMap[sectionIndex] = section;
    }
    return sectionMap.at(sectionIndex);
}

nlohmann_hcl::json JsonGraphLoader::getConfig() const
{
    return json_utils::get(m_jsonGraph, "config", nlohmann_hcl::json());
}

nlohmann_hcl::json JsonGraphLoader::getGraphAttributes() const
{
    return json_utils::get(m_jsonGraph, "attributes", nlohmann_hcl::json());
}

synGraphAttribute stringToEnum(const std::string& attr)
{
    static const std::unordered_map<std::string_view, synGraphAttribute> translationMap {
        std::make_pair("GRAPH_ATTRIBUTE_INFERENCE", synGraphAttribute::GRAPH_ATTRIBUTE_INFERENCE),
        std::make_pair("GRAPH_ATTRIBUTE_QUANTIZATION", synGraphAttribute::GRAPH_ATTRIBUTE_QUANTIZATION),
        std::make_pair("GRAPH_ATTRIBUTE_BACKOFF_FACTOR", synGraphAttribute::GRAPH_ATTRIBUTE_BACKOFF_FACTOR),
        std::make_pair("GRAPH_ATTRIBUTE_MAX", synGraphAttribute::GRAPH_ATTRIBUTE_MAX)};
    HB_ASSERT(translationMap.size() == synGraphAttribute::GRAPH_ATTRIBUTE_MAX + 1,
              "Expecting graphAttrTranslationMap contains all enums");
    const auto it = translationMap.find(attr);
    HB_ASSERT(it != translationMap.end(), "Expecting attr in stringToEnum");
    return it->second;
}

void JsonGraphLoader::loadGraphAttributes()
{
    const auto graphAttributes = getGraphAttributes();
    if (graphAttributes.empty()) return;
    std::vector<synGraphAttribute> attributes;
    attributes.reserve(graphAttributes.size());
    std::vector<synGraphAttributeVal> values;
    values.reserve(graphAttributes.size());
    for (auto it = graphAttributes.begin(); it != graphAttributes.end(); ++it)
    {
        JT_LOG_INFO("Set attribue: " << it.key() << ", value: " << it.value());
        synGraphAttribute graphAttr;
#if MAGIC_ENUM_SUPPORTED
        const auto graphAttribute = magic_enum::enum_cast<synGraphAttribute>(it.key());
        HB_ASSERT(graphAttribute.has_value(), "Expecting successful cast from string to enum");
        graphAttr = graphAttribute.value();
#else
        graphAttr = stringToEnum(it.key());
#endif
        attributes.push_back(graphAttr);
        synGraphAttributeVal val;
        switch (graphAttr)
        {
            case GRAPH_ATTRIBUTE_INFERENCE:
            case GRAPH_ATTRIBUTE_QUANTIZATION:
            {
                val.iAttrVal = it.value();
                break;
            }
            case GRAPH_ATTRIBUTE_BACKOFF_FACTOR:
            {
                val.dAttrVal = it.value();
                break;
            }
            default:
            {
                JT_LOG_ERR(fmt::format("Unrecognized Graph Attribute in json ({})", graphAttr));
            }
        }
        values.push_back(val);
    }
    // set attributes for the graph.
    m_graph->setGraphAttributesV2(attributes, values, attributes.size());
}