#include <habana_nodes/node_factory.h>
#include "habana_pass.h"
#include "quantizer_factory.h"
#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "synapse_common_types.h"

bool shouldEnforceBf16(const TensorPtr& tensor)
{
    // syn_type_fp8_152 supporting one expBias value, therefore it could be treated as non-QuantType
    synDataType dType = tensor->getElementType();
    bool isQuantType = dType != syn_type_fp8_152 && isQuantDtype(getDtypeSuffixFromSynDataType(tensor->getElementType()));

    return  isQuantType && !tensor->getDynamicRange().isSet &&
           !tensor->getQuantizationParams().m_isUserQuantInfo && !tensor->getQuantizationParams().m_isUserPCQuantInfo &&
           !tensor->isDataTypeMatchData();
}

bool enforceNodePrecision(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT, "Graph is not in inference mode, enforce node precision won't run.");
        return true;
    }

    if (!GCFG_SYNAPSE_DATA_TYPE_SELECTION.value())
    {
        LOG_DEBUG(QUANT, "Data type selection is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    if (GCFG_ALLOW_DEFAULT_QUANT_PARAMS.value())
    {
        LOG_DEBUG(QUANT, "GC is allowed to use default quantization params. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    // SW-141552 define the default data type in a common location for usage of all data type related passes
    constexpr synDataType typeToEnforce = syn_type_bf16;
    // enforce node precision to be bf16 for input/output tensors lacking Dynamic range/Quantization data.
    for (const TensorPtr& t : g.getTensors())
    {
        if (shouldEnforceBf16(t))
        {
            if (t->isPersistent() && !t->inConstSection())
            {
                LOG_ERR(GC,
                        "{}: Cannot update dtype for non-const persistent tensor {}, but the current type {} is missing "
                        "quantization info so it can't be used",
                        HLLOG_FUNC,
                        t->getName(),
                        getStringFromSynDataType(t->getElementType()));
                return false;
            }
            t->changeDefaultElementType(typeToEnforce);

            QuantizationData quantInfo(typeToEnforce);
            t->setQuantizationParams(quantInfo);
            LOG_DEBUG(GC,
                      "{}: setting precision for tensor {} to be {} by default - due to missing quantization info",
                      HLLOG_FUNC,
                      t->getName(),
                      getStringFromSynDataType(typeToEnforce));

            NodePtr producer = g.getTensorProducer(t);
            if (producer != nullptr && producer->getNodePrecision() != typeToEnforce)
            {
                producer->setNodePrecision(typeToEnforce);
                LOG_DEBUG(GC,
                          "{}: setting node {} precision to {} - due to missing output quantization info",
                          HLLOG_FUNC,
                          producer->getNodeName(),
                          getStringFromSynDataType(typeToEnforce));
            }

            for (const NodePtr& consumer : g.getTensorConsumers(t))
            {
                if (consumer != nullptr && consumer->getNodePrecision() != typeToEnforce)
                {
                    consumer->setNodePrecision(typeToEnforce);
                    LOG_DEBUG(GC,
                              "{}: setting node {} precision to {} - due to missing input quantization info",
                              HLLOG_FUNC,
                              consumer->getNodeName(),
                              getStringFromSynDataType(typeToEnforce));
                }
            }
        }
    }

    return true;
}

