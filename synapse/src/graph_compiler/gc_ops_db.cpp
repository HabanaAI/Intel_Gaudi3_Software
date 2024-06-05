#include <algorithm>

#include "defs.h"
#include "node_factory.h"
#include "gc_ops_db.h"
#include "synapse_common_types.h"

static const auto isInputToStr = [](bool isInput) { return isInput ? "input" : "output"; };

// Device to gc op info getter forward declaration
namespace gc::ops::details
{
// Training API ops:

// Matrix operations
static DeviceTypeToOpInfoMap getConv2DOpInfoMap();
static DeviceTypeToOpInfoMap getConv3DOpInfoMap();
static DeviceTypeToOpInfoMap getGEMMOpInfoMap();
static DeviceTypeToOpInfoMap getBatchGemmOpInfoMap();
static DeviceTypeToOpInfoMap getMaskedBatchGemmOpInfoMap();

// Data movement
static DeviceTypeToOpInfoMap getTransposeOpInfoMap();
static DeviceTypeToOpInfoMap getSplitShapeOpInfoMap();
static DeviceTypeToOpInfoMap getStridedViewOpInfoMap();
static DeviceTypeToOpInfoMap getStridedInsertOpInfoMap();
static DeviceTypeToOpInfoMap getStridedSliceGradOpInfoMap();
static DeviceTypeToOpInfoMap getSliceInsertOpInfoMap();
static DeviceTypeToOpInfoMap getSqueezeOpInfoMap();
static DeviceTypeToOpInfoMap getSplitOpInfoMap();
static DeviceTypeToOpInfoMap getSliceAxisOpInfoMap();
static DeviceTypeToOpInfoMap getSliceFwdOpInfoMap();
static DeviceTypeToOpInfoMap getFlattenOpInfoMap();
static DeviceTypeToOpInfoMap getExpandDimsOpInfoMap();
static DeviceTypeToOpInfoMap getIdentityOpInfoMap();
static DeviceTypeToOpInfoMap getMemcpyOpInfoMap();
static DeviceTypeToOpInfoMap getMemsetOpInfoMap();
static DeviceTypeToOpInfoMap getReinterpretCastOpInfoMap();
static DeviceTypeToOpInfoMap getReshapeOpInfoMap();
static DeviceTypeToOpInfoMap getBroadcastOpInfoMap();
static DeviceTypeToOpInfoMap getConcatOpInfoMap();

// Normalization
static DeviceTypeToOpInfoMap getNormMomentsOpInfoMap();
static DeviceTypeToOpInfoMap getFrobeniusNormOpInfoMap();

// Misc
static DeviceTypeToOpInfoMap getEinsumOpInfoMap();

}  // namespace gc::ops::details

namespace gc::ops
{
GCOpsDB::GCOpsDB()
{
    initSupportedGuidsSet();

    // clang-format off
    registerOp(NodeFactory::transposeNodeTypeName,        details::getTransposeOpInfoMap());
    registerOp(NodeFactory::splitShapeNodeTypeName,       details::getSplitShapeOpInfoMap());
    registerOp(NodeFactory::stridedViewNodeTypeName,      details::getStridedViewOpInfoMap());
    registerOp(NodeFactory::stridedInsertNodeTypeName,    details::getStridedInsertOpInfoMap());
    registerOp(NodeFactory::stridedSliceGradNodeTypeName, details::getStridedSliceGradOpInfoMap());
    registerOp(NodeFactory::sliceInsertNodeTypeName,      details::getSliceInsertOpInfoMap());
    registerOp(NodeFactory::squeezeNodeTypeName,          details::getSqueezeOpInfoMap());
    registerOp(NodeFactory::splitNodeTypeName,            details::getSplitOpInfoMap());
    registerOp(NodeFactory::sliceAxisNodeTypeName,        details::getSliceAxisOpInfoMap());
    registerOp(NodeFactory::sliceNodeTypeName,            details::getSliceFwdOpInfoMap());
    registerOp(NodeFactory::flattenNodeTypeName,          details::getFlattenOpInfoMap());
    registerOp(NodeFactory::expandDimsNodeTypeName,       details::getExpandDimsOpInfoMap());
    registerOp(NodeFactory::identityNodeTypeName,         details::getIdentityOpInfoMap());
    registerOp(NodeFactory::memcpyNodeTypeName,           details::getMemcpyOpInfoMap());
    registerOp(NodeFactory::memsetNodeTypeName,           details::getMemsetOpInfoMap());
    registerOp(NodeFactory::reinterpretCastNodeTypeName,  details::getReinterpretCastOpInfoMap());
    registerOp(NodeFactory::reshapeNodeTypeName,          details::getReshapeOpInfoMap());
    registerOp(NodeFactory::broadcastNodeTypeName,        details::getBroadcastOpInfoMap());
    registerOp(NodeFactory::concatenateNodeTypeName,      details::getConcatOpInfoMap());
    registerOp(NodeFactory::momentsFwdNodeTypeName,       details::getNormMomentsOpInfoMap());
    registerOp(NodeFactory::FrobeniusNormTypeName,        details::getFrobeniusNormOpInfoMap());
    registerOp(NodeFactory::einsumTypeName,               details::getEinsumOpInfoMap());
    registerOp(NodeFactory::batchGemmNodeTypeName,        details::getBatchGemmOpInfoMap());
    registerOp(NodeFactory::batchGemmDeDxNodeTypeName,    details::getBatchGemmOpInfoMap());
    registerOp(NodeFactory::batchGemmDeDwNodeTypeName,    details::getBatchGemmOpInfoMap());
    registerOp(NodeFactory::convolutionNodeTypeName,      details::getConv2DOpInfoMap());
    registerOp(NodeFactory::convolution3DNodeTypeName,    details::getConv3DOpInfoMap());
    registerOp(NodeFactory::deDxNodeTypeName,             details::getConv2DOpInfoMap());
    registerOp(NodeFactory::deDx3DNodeTypeName,           details::getConv3DOpInfoMap());
    registerOp(NodeFactory::deDwNodeTypeName,             details::getConv2DOpInfoMap());
    registerOp(NodeFactory::deDw3DNodeTypeName,           details::getConv3DOpInfoMap());
    registerOp(NodeFactory::gemmNodeTypeName,             details::getGEMMOpInfoMap());
    registerOp(NodeFactory::gemmDeDxNodeTypeName,         details::getGEMMOpInfoMap());
    registerOp(NodeFactory::gemmDeDwNodeTypeName,         details::getGEMMOpInfoMap());
    registerOp(NodeFactory::maskedBatchGemmNodeTypeName,  details::getMaskedBatchGemmOpInfoMap());
    // clang-format on

    HB_ASSERT(m_guidToOpInfoMap.size() == NodeFactory::getNumApiNodes(),
              "Expecting all API nodes registered to GC ops database");
}

const std::shared_ptr<const std::unordered_set<std::string>> GCOpsDB::getSupportedGuids(synDeviceType device) const
{
    std::shared_ptr<std::unordered_set<std::string>> ret {};

    const auto it = m_supportedGuidsPerDevice.find(device);
    if (it != m_supportedGuidsPerDevice.end())
    {
        ret = it->second;
    }
    return ret;
}

GCOpsDB& GCOpsDB::instance()
{
    static GCOpsDB opdb;
    return opdb;
}

void GCOpsDB::registerOp(const std::string& guid, const DeviceTypeToOpInfoMap& opInfoMap)
{
    HB_ASSERT(!guid.empty(), "Expecting a non-empty guid");
    HB_ASSERT(NodeFactory::getInstance().isApiNode(guid), "Internal guid {} is not supported", guid);
    HB_ASSERT(!opInfoMap.empty(), "Empty deviceToOpInfoMap for guid: {}", guid);

    for (const auto device : getSupportedDevices())
    {
        const auto it = opInfoMap.find(device);
        if (it == opInfoMap.end())
        {
            LOG_DEBUG(GC, "{}: No op info for guid {}, deviceType: {}", HLLOG_FUNC, guid, device);
            continue;
        }

        HB_ASSERT_PTR(it->second);
        for (const auto& isInput : {true, false})
        {
            const auto& supportedDatatypes =
                isInput ? it->second->supportedInputDatatypes : it->second->supportedOutputDatatypes;
            const auto& operandsRank = isInput ? it->second->supportedInputRanks : it->second->supportedOutputRanks;
            HB_ASSERT(operandsRank.size() == supportedDatatypes.size(),
                      "initSupportedGuidsSet(): Expecting {} operands supported data types size {} == "
                      "operandsranks size {} for guid {}, deviceType {}",
                      isInputToStr(isInput),
                      supportedDatatypes.size(),
                      operandsRank.size(),
                      guid,
                      device);
        }

        HB_ASSERT(m_supportedGuidsPerDevice.find(device) != m_supportedGuidsPerDevice.end(), "");
        m_supportedGuidsPerDevice[device]->insert(guid);
    }
    m_guidToOpInfoMap.insert(std::make_pair(guid, std::make_unique<DeviceTypeToOpInfoMap>(opInfoMap)));
}

const std::shared_ptr<const OpInfo> GCOpsDB::getGCOpInfo(const std::string& guid, synDeviceType deviceType) const
{
    const auto guidToOpInfoMapIt = m_guidToOpInfoMap.find(guid);
    if (guidToOpInfoMapIt != m_guidToOpInfoMap.end() && guidToOpInfoMapIt->second != nullptr)
    {
        const auto& pOpInfoMap       = guidToOpInfoMapIt->second;
        const auto  deviceToOpInfoIt = pOpInfoMap->find(deviceType);
        if (deviceToOpInfoIt != pOpInfoMap->end())
        {
            return deviceToOpInfoIt->second;
        }
    }
    return nullptr;
}

void GCOpsDB::initSupportedGuidsSet()
{
    for (const auto device : getSupportedDevices())
    {
        m_supportedGuidsPerDevice.insert({device, std::make_shared<std::unordered_set<std::string>>()});
    }
}

OpInfo::TypeGroup::TypeGroup(OpInfo::DatatypesMask        supportedTypes,
                             const std::vector<unsigned>& input,
                             const std::vector<unsigned>& output)
: mask(supportedTypes), inputIndices(input), outputIndices(output)
{
}

}  // namespace gc::ops

namespace gc::ops::details
{
// Ignore shape tensors datatypes
static constexpr OpInfo::DatatypesMask shapeDatatypes =
    syn_type_int8 | syn_type_uint8 | syn_type_int16 | syn_type_uint16 | syn_type_uint32 | syn_type_int32 |
    syn_type_bf16 | syn_type_fp16 | syn_type_float | syn_type_fp8_143 | syn_type_fp8_152 | syn_type_uint64 |
    syn_type_int64 | syn_type_tf32 | syn_type_int4 | syn_type_uint4 | syn_type_hb_float | syn_type_ufp16;
static constexpr OpInfo::DatatypesMask mmeGaudiTypes = syn_type_bf16 | syn_type_single;
static constexpr OpInfo::DatatypesMask mmeGaudi2Types =
    mmeGaudiTypes | syn_type_int8 | syn_type_uint8 | syn_type_int16 | syn_type_uint16 | syn_type_fp16 | syn_type_tf32 |
    syn_type_hb_float | syn_type_fp8_143 | syn_type_fp8_152 | syn_type_int32 | syn_type_uint32;

static constexpr OpInfo::DatatypesMask mmeGaudi3Types = syn_type_bf16 | syn_type_fp16 | syn_type_tf32 |
                                                        syn_type_single | syn_type_hb_float | syn_type_fp8_143 |
                                                        syn_type_fp8_152 | syn_type_ufp16;

// Device to OpInfo getters for each synapse API node
static DeviceTypeToOpInfoMap getBroadcastOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_int16 | syn_type_uint16 |
                                                     syn_type_uint32 | syn_type_bf16 | syn_type_fp16 | syn_type_int32 |
                                                     syn_type_float | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi2Types = supportedTypes | syn_type_fp8_143 | syn_type_fp8_152;

    const auto gaudi2Info = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi2Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi2Types, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;

    const auto gaudi3Info = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getConcatOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_uint64 | syn_type_int64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})},
        true /* is varying input operand num */,
        false /* is varying output operand num */);

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})},
        true /* is varying input operand num */,
        false /* is varying output operand num */);

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getIdentityOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type group with node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type group with node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getMemcpyOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges(1, OpInfo::DimRange(1, HABANA_DIM_MAX)) /* supported input dim ranges */,
        OpInfo::DimRanges(1, OpInfo::DimRange(1, HABANA_DIM_MAX)) /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges(1, OpInfo::DimRange(1, HABANA_DIM_MAX)) /* supported input dim ranges */,
        OpInfo::DimRanges(1, OpInfo::DimRange(1, HABANA_DIM_MAX)) /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});
    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getMemsetOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(shapeDatatypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types =
        syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 | syn_type_uint32 | syn_type_float |
        syn_type_fp8_143 | syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 | syn_type_uint16 | syn_type_int64 |
        syn_type_uint64;
    const auto gaudi3Info = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(shapeDatatypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getReinterpretCastOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getFlattenOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getSplitOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_uint64 | syn_type_int64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})},
        false /* is varying input num */,
        true /* is varying output num */);

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})},
        false /* is varying input num */,
        true /* is varying output num */);

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getExpandDimsOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getSliceAxisOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes =
        syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 | syn_type_uint32 | syn_type_float |
        syn_type_fp8_143 | syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 | syn_type_uint16;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-147735]: Align to synapse node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-147735]: Align to synapse node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getReshapeOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | syn_type_hb_float | syn_type_tf32;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        2 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types, shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getSliceFwdOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136551][SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-136551][SW-136998]: once ticket is done, align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getStridedSliceGradOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getSliceInsertOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_uint64 | syn_type_int64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getSplitShapeOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {shapeDatatypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {},  // ignore types as they are ignored in shape tensors
        false /* is varying input num */,
        true /* is varying output num  */);

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudiInfo});
    return map;
}

static DeviceTypeToOpInfoMap getSqueezeOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_uint64 | syn_type_int64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported output dim ranges */,
        // TODO [SW-148611] - align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX - 1)} /* supported output dim ranges */,
        // TODO [SW-148611] - align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getStridedInsertOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getStridedViewOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_int64 | syn_type_uint64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                shapeDatatypes,
                                shapeDatatypes,
                                shapeDatatypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX),
                           OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getTransposeOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_bf16 | syn_type_int32 |
                                                     syn_type_uint32 | syn_type_float | syn_type_fp8_143 |
                                                     syn_type_fp8_152 | syn_type_fp16 | syn_type_int16 |
                                                     syn_type_uint16 | syn_type_uint64 | syn_type_int64;
    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-148611] - align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, HABANA_DIM_MAX)} /* supported output dim ranges */,
        // TODO [SW-148611] - align type groups to node docs
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0}, {}),
                                        OpInfo::TypeGroup(supportedGaudi3Types, {}, {0})});
    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getConv2DOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudiTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudiTypes, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudiTypes, {}, {0})});

    const auto gaudi2Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi2Types, mmeGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi2Types, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudi2Types, {}, {0, 1})});

    const auto gaudi3Info = std::make_shared<OpInfo>(
        6 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi3Types, mmeGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi3Types, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudi3Types, {}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getConv3DOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudiTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudiTypes, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudiTypes, {}, {0})});

    const auto gaudi2Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi2Types, mmeGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi2Types, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudi2Types, {}, {0, 1})});

    const auto gaudi3Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi3Types, mmeGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(3, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi3Types, {0, 1}, {}),
                                        OpInfo::TypeGroup(mmeGaudi3Types, {}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getGEMMOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudiTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudiTypes, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudiTypes, {}, {0})});

    const auto gaudi2Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi2Types, mmeGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi2Types, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudi2Types, {}, {0, 1})});

    const auto gaudi3Info = std::make_shared<OpInfo>(
        6 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi3Types, mmeGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi3Types, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudi3Types, {}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getBatchGemmOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudiTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudiTypes, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudiTypes, {}, {0})});

    const auto gaudi2Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi2Types, mmeGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi2Types, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudi2Types, {}, {0, 1})});

    const auto gaudi3Info = std::make_shared<OpInfo>(
        6 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi3Types, mmeGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi3Types, {0, 1, 2, 3}, {}),
                                        OpInfo::TypeGroup(mmeGaudi3Types, {}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getMaskedBatchGemmOpInfoMap()
{
    DeviceTypeToOpInfoMap map;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes,
                                mmeGaudiTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudiTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudiTypes, {0, 1, 2, 3}, {0})});

    const auto gaudi2Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types,
                                mmeGaudi2Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi2Types, mmeGaudi2Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi2Types, {0, 1, 2, 3}, {0, 1})});

    const auto gaudi3Info = std::make_shared<OpInfo>(
        4 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types,
                                mmeGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {mmeGaudi3Types, mmeGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(2, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(mmeGaudi3Types, {0, 1, 2, 3}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudi2Info});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}

static DeviceTypeToOpInfoMap getNormMomentsOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int32 | syn_type_uint32 | syn_type_float | syn_type_bf16;

    const auto gaudiInfo = std::make_shared<OpInfo>(
        1 /* max inputs*/,
        2 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes, supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        // TODO: Align type groups with synapse docs once [SW-136615] is done
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {}),
                                        OpInfo::TypeGroup(supportedTypes, {}, {0, 1})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudiInfo});
    return map;
}

static DeviceTypeToOpInfoMap getFrobeniusNormOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_int32 | syn_type_uint32 | syn_type_float | syn_type_bf16;

    const auto gaudiInfo =
        std::make_shared<OpInfo>(1 /* max inputs*/,
                                 1 /* max outputs*/,
                                 OpInfo::DatatypesMasks {supportedTypes} /* supported input datatypes */,
                                 OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
                                 OpInfo::DimRanges {OpInfo::DimRange(2, 4)} /* supported input dim ranges */,
                                 OpInfo::DimRanges {OpInfo::DimRange(1, 1)} /* supported output dim ranges */,
                                 std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudiInfo});
    return map;
}

static DeviceTypeToOpInfoMap getEinsumOpInfoMap()
{
    DeviceTypeToOpInfoMap           map;
    constexpr OpInfo::DatatypesMask supportedTypes = syn_type_bf16 | syn_type_float | syn_type_int32;
    const auto                      gaudiInfo      = std::make_shared<OpInfo>(
        3 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedTypes, supportedTypes, supportedTypes} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedTypes} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedTypes, {0, 1, 2}, {0})});

    constexpr OpInfo::DatatypesMask supportedGaudi3Types = supportedTypes | mmeGaudi3Types;
    const auto                      gaudi3Info           = std::make_shared<OpInfo>(
        3 /* max inputs*/,
        1 /* max outputs*/,
        OpInfo::DatatypesMasks {supportedGaudi3Types,
                                supportedGaudi3Types,
                                supportedGaudi3Types} /* supported input datatypes */,
        OpInfo::DatatypesMasks {supportedGaudi3Types} /* supported output datatypes */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM),
                           OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported input dim ranges */,
        OpInfo::DimRanges {OpInfo::DimRange(1, SYN_MAX_TENSOR_DIM)} /* supported output dim ranges */,
        std::vector<OpInfo::TypeGroup> {OpInfo::TypeGroup(supportedGaudi3Types, {0, 1, 2}, {0})});

    map.insert({synDeviceGaudi, gaudiInfo});
    map.insert({synDeviceGaudi2, gaudiInfo});
    map.insert({synDeviceGaudi3, gaudi3Info});
    return map;
}
}  // namespace gc::ops::details
