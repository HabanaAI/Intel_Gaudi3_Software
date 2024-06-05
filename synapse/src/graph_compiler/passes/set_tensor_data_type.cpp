
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "data_type_utils.h"
#include "quantization_utils.h"
#include "cast_nodes_handler.h"
#include "synapse_common_types.h"
#include "types.h"

synDataType getTensorDataTypeFromConsumers(HabanaGraph& g, const TensorPtr& tensor)
{
    synDataType precision = syn_type_na;

    for (const NodePtr& consumer : g.getTensorConsumers(tensor))
    {
        precision = getHighestGUIDDataType({precision, getQuantDataType(g, tensor, consumer)});
    }

    return precision;
}

static bool canTensorHaveInvalidDtype(TensorPtr tensor)
{
    // A tensor can have invalid (syn_type_na) data type at the beginning of the pass if it is either :
    // 1. non-persistent
    // 2. persistent and assigned to const section (static tensor)
    return !tensor->isPersistent() || tensor->isAssignedToConstSection();
}

bool ignoreUserDataType(const HabanaGraph& g, const TensorPtr& tensor)
{
    if (tensor->isPersistent() && !tensor->inConstSection())
    {
        return false;
    }

    for (const NodePtr& consumer : g.getTensorConsumers(tensor))
    {
        if (HabanaGraph::runsOnTPC(consumer))
        {
            return false;
        }
    }
    return GCFG_IGNORE_USER_DATA_TYPE.value();
}

void setNodeOutputTensorsDataType(HabanaGraph& g)
{
    for (const NodePtr& node : g.getTopoSortedNodes())
    {
        const bool isMmeNode = HabanaGraph::runsOnMME(node);
        for (const TensorPtr& output : node->getOutputs())
        {
            if (output == nullptr) continue;

            const bool ignoreUserDtype = ignoreUserDataType(g, output);

            synDataType dtype = output->getElementType();
            if (dtype == syn_type_na || ignoreUserDtype)
            {
                HB_ASSERT(canTensorHaveInvalidDtype(output),
                          "Persistent output tensor {} has invalid data type",
                          output->getName());

                if (dtype == syn_type_na)
                {
                    LOG_TRACE(DATA_TYPES,
                              "Data type for tensor {} wasn't found, trying to get by index",
                              output->getName());
                }

                if (dtype != syn_type_na && ignoreUserDtype)
                {
                    LOG_TRACE(DATA_TYPES,
                              "Ignore user data type for tensor {}",
                              output->getName());
                }

                // TODO - SW-136400 remove this WA for MME nodes after supporting the cguid convert_f8 kernel
                if (isMmeNode)
                {
                    LOG_TRACE(DATA_TYPES,
                              "Set BF16 type for MME node's output tensor {}",
                              output->getName());
                    dtype = syn_type_bf16;
                }
                else
                {
                    dtype = getRequiredOutputDataTypeByIndex(node->getGUID(), node->getOutputIndexOfTensor(output));
                }

                if (dtype == syn_type_na)
                {
                    LOG_TRACE(DATA_TYPES,
                              "Data type for tensor {} wasn't found by index, checking consumers",
                              output->getName());
                    dtype = getTensorDataTypeFromConsumers(g, output);
                }
                // if got syn_type_na from consumer then use producer node precision
                // if producer node precision was not set, then use its node type minimum precision
                dtype = (dtype == syn_type_na) ? node->getNodePrecision() : dtype;
                dtype = (dtype == syn_type_na) ? g.getNodeTypeMinPrecision(node) : dtype;
            }

            output->changeDefaultElementType(dtype, ignoreUserDtype);
            LOG_TRACE(DATA_TYPES,
                      "{}'s output tensor {} chosen dtype is {}",
                      node->getNodeName(),
                      output->getName(),
                      getStringFromSynDataType(dtype));
        }
    }
}

void convertBiasDataTo32bit(TensorPtr bias)
{
    char* data = bias->getData();
    unsigned numElements = bias->getBufferSizeInBytes() / bias->getElementSizeInBytes();
    switch (bias->getBufferDataType())
    {
        case syn_type_float:
        case syn_type_int32:
        {
            // No need to convert
            break;
        }
        case syn_type_bf16:
        {
            LOG_TRACE(DATA_TYPES, "converting bias tensor {} data from bf16 to float32",
                      bias->getName());
            float* convertedData = bf16BufferTofloatBuffer((bf16_t*)data, numElements);
            bias->setTensorBuffer(convertedData, numElements * sizeof(float), syn_type_float, true);
            delete convertedData;
            break;
        }
        case syn_type_fp16:
        {
            LOG_TRACE(DATA_TYPES, "converting bias tensor {} data from fp16 to float32",
                      bias->getName());
            float* convertedData = float16BufferToFloatBuffer((fp16_t*)data, numElements);
            bias->setTensorBuffer(convertedData, numElements * sizeof(float), syn_type_float, true);
            delete convertedData;
            break;
        }
        default:
        {
            // todo: in case of int8/16, need to be dequanted to int32
            LOG_WARN(DATA_TYPES, "model tensor {} is static bias with unsupported data type {}",
                     bias->getName(), bias->getElementType());
            HB_ASSERT(false, "unsupported bias buffer data type");
        }
    }
}

bool setGraphTensorsDataType(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(DATA_TYPES,
                  "Data type selection is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  HLLOG_FUNC);
        return true;
    }

    if (!GCFG_SYNAPSE_DATA_TYPE_SELECTION.value())
    {
        LOG_DEBUG(DATA_TYPES, "Data type selection is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    setNodeOutputTensorsDataType(g);

    for (const TensorPtr& tensor : g.getTensors())
    {
        // If tensor has a producer then its type was set by setNodeOutputTensorsDataType function
        if (g.getTensorProducer(tensor) != nullptr) continue;
        const bool ignoreUserDType = ignoreUserDataType(g, tensor);

        synDataType dtype = tensor->getElementType();
        if (dtype == syn_type_na)
        {
            HB_ASSERT(canTensorHaveInvalidDtype(tensor),
                      "Persistent input tensor {} has invalid data type",
                      tensor->getName());
            LOG_TRACE(DATA_TYPES, "Data type for tensor {} wasn't found, checking consumers", tensor->getName());
            dtype = getTensorDataTypeFromConsumers(g, tensor);
        }
        else if (ignoreUserDType)
        {
            LOG_TRACE(DATA_TYPES,
                      "Ignoring user provided data type {} for tensor {}, checking consumers",
                      dtype, tensor->getName());
            dtype = getTensorDataTypeFromConsumers(g, tensor);
        }

        if (dtype == syn_type_na)
        {
            dtype = syn_type_bf16;
            LOG_TRACE(DATA_TYPES,
                      "Data type for tensor {} wasn't found in consumers, picking default value {}",
                      tensor->getName(),
                      dtype);
        }
        // If GCFG_IGNORE_USER_DATA_TYPE is true, we may have static tensors with element and buffer type fp32
        // provided from the user. Therefore we must ignore data type match data to set their element type
        // correctly.
        tensor->changeDefaultElementType(dtype, ignoreUserDType);
        LOG_TRACE(DATA_TYPES, "model tensor {} chosen dtype is {}", tensor->getName(), getStringFromSynDataType(dtype));
    }

    // After the pass ends - adding typeless tensors to the graph is disallowed
    g.finishDataTypeSelection();

    return true;
}
