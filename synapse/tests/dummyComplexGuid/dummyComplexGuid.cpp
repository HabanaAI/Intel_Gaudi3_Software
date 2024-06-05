#include <string>
#include <cstring>
#include <algorithm>
#include "dummyComplexGuid.h"
#include "dummy_translator.h"

using namespace tpc_lib_api;

typedef std::vector<std::string> GuidsVector;

const GuidsVector supportedFunctionalGUIDs = {
    "dummy_functional_guid",     // used to test performance pass
    "complex_add_f32",           // complex_add(x,y) = add(x,neg(neg(y))
    "greco_complex_add_f32",     // like complex_add, but extracted to non-fwd ops
    "nested_complex_add_f32",    // nested_complex_add(x,y) = complex_add(x,neg(neg(y))
    "infinite_complex_add_f32",  // expands to infinite loop :
                                 // infinite_complex_add_f32(x,y) = infinite_complex_add_f32(x, neg(neg(y))
    "ctrl_dep_complex_add_f32",  // complex_add with internal ctrl dep
    "complex_abs_f32",           // complex_abs(x) = abs(abs(x))
    "two_RMW_complex_add_f32",   // input and output of outer neg are RMW with different sections
    "dropped_external_tensor_complex_add_f32",        // input and output of outer neg are RMW with different sections
    "invalid_new_RMW_output_complex_add_f32",         // new RMW output for add with no alias (different section)
    "invalid_new_RMW_output_offset_complex_add_f32",  // new RMW output for add no alias (different offset)
    "invalid_new_input_complex_add_f32",              // new input tensor with no static data
    "valid_nDims_complex_add_f32",                    // output of inner neg has too many dims
    "invalid_prop_complex_add_f32",                   // invalid tensor property
    "invalid_new_persistent_output_complex_add_f32",  // new RMW output for add with no alias (different section)
    "invalid_new_persistent_output_offset_complex_add_f32",  // new RMW output for add no alias (different offset)
    "complex_nop_strides_f32",  // used only to check non-default strides in CommonIR Tensor. no actual operation done
    "norm_moments_f32",         // only node guid is updated
    "norm_moments_fwd_f32_unconnected_f32",                // unconnected cluster is extracted.
    "invalid_nDims_norm_moments_fwd_f32_unconnected_f32",  // output of gemm has too many dims
    "graph_unchanged_dummy",
    "logical_shape_manipulation"  // a node with isShapeManipulation set to true
};

const GuidsVector supportedPerformanceGUIDs = {"dummy_performance_guid",  // used to test functional pass
                                               // GUIDS identical to those in functional
                                               "greco_complex_add_f32",
                                               "complex_add_f32",
                                               "graph_unchanged_dummy",
                                               "logical_shape_manipulation"};

const GuidsVector supportedDynamicShapeGUIDs = {"logical_shape_manipulation"};

GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params,
                                       NodeDataLayouts* supportedLayouts,
                                       uint32_t* layoutCount)
{
    if (layoutCount == nullptr) return GLUE_FAILED;
    if (supportedLayouts != nullptr)
    {
        for (unsigned i = 0; i < *layoutCount; i++)
        {
            std::memcpy(supportedLayouts->inputs[i].layout, "xxxxx", MAX_TENSOR_DIM);
            std::memcpy(supportedLayouts->outputs[i].layout, "xxxxx", MAX_TENSOR_DIM);
        }
    }
    else
    {
        *layoutCount = 1;
    }
    return GLUE_SUCCESS;
}

GlueCodeReturn getGuids(GuidInfo* guids, unsigned* guidCount, const GuidsVector& guidList)
{
    if (guidCount == nullptr) return GLUE_FAILED;
    // if GUIDs is null, update GUIDCount
    if (guids == nullptr)
    {
        *guidCount = guidList.size();
    }
    else
    {
        guids = new GuidInfo[*guidCount];
        // if GUIDs is allocated, update GUIDs
        for (unsigned i = 0; i < *guidCount; i++)
        {
            size_t len = guidList[i].size();
            strncpy(guids[i].name, guidList[i].c_str(), len);
            guids[i].name[len] = '\0';
            if (std::find(supportedDynamicShapeGUIDs.begin(),
                          supportedDynamicShapeGUIDs.end(),
                          guidList[i].c_str()) != supportedDynamicShapeGUIDs.end())
            {
                guids[i].supportsDynamicShapes = 1;
            }
        }
    }
    return GLUE_SUCCESS;
}

GlueCodeReturn GetFunctionalComplexGuids(DeviceId deviceId, unsigned* guidCount, GuidInfo* guids)
{
    return getGuids(guids, guidCount, supportedFunctionalGUIDs);
}

GlueCodeReturn GetPerformanceComplexGuids(DeviceId deviceId, unsigned* guidCount, GuidInfo* guids)
{
    return getGuids(guids, guidCount, supportedPerformanceGUIDs);
}

GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, TensorManipulationSuggestion* suggestion)
{
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance)
{
    return GLUE_SUCCESS;
}

GlueCodeReturn GetShapeInference(DeviceId deviceId, ShapeInferenceParams* inputParams, ShapeInferenceOutput* outputData)
{
    return GLUE_SUCCESS;
}

uint64_t GetLibVersion()
{
    return 1;
}

GlueCodeReturn ExtractFunctionalComplexGUID(const gc_protocol::ProtocolGraph* inputGraph,
                                            gc_protocol::ProtocolGraph**      outputGraph)
{
    DefaultGraphTranslator* defaultGraphTranslator = new DefaultGraphTranslator(*inputGraph);
    defaultGraphTranslator->convertToDefault();
    DefaultGraphWrapper* defaultGraphWrapper =
        new DefaultGraphWrapper(defaultGraphTranslator->m_translatorIRNodes,
                                defaultGraphTranslator->m_translatorIRNodesIdsAndIRTensors);
    *outputGraph = reinterpret_cast<gc_protocol::ProtocolGraph*>(defaultGraphWrapper);
    return GLUE_SUCCESS;
}

GlueCodeReturn ExtractPerformanceComplexGUID(const gc_protocol::ProtocolGraph* inputGraph,
                                             gc_protocol::ProtocolGraph**      outputGraph)
{
    const std::function<bool(const gc_protocol::ProtocolNode&)> x;
    inputGraph->foreachNode([&](const gc_protocol::ProtocolNode inputNode) {
        printf("ExtractPerformanceComplexGUIDP was called\n");
        return true;
    });

    return GLUE_SUCCESS;
}
