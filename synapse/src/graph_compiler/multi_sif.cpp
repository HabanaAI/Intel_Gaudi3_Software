#include <unordered_map>
#include <algorithm>

#include "tpc_kernel_lib_interface.h"
#include "habana_graph.h"
#include "kernel_db.h"
#include "smf/shape_func_registry.h"
#include "multi_sif.h"
#include "tpc_fuser.h"

tpc_lib_api::GlueCodeReturn multiSifRun(synDeviceType     deviceType,
                                        MultiSifNodeInfo* multiSifNodeInfo,
                                        SifParams*        sifParams,
                                        SifOutputs*       sifOutputs,
                                        bool              inferMax)
{
    std::vector<TensorShapeInfo> internalTensors(multiSifNodeInfo->m_internalTensorsNr);
    std::vector<bool>            tensorSet(multiSifNodeInfo->m_internalTensorsNr);

    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    unsigned subnodeIdx = 0;

    for (auto& node : multiSifNodeInfo->m_nodes)
    {
        inputs.resize(node.m_inputs.size());
        outputs.resize(node.m_outputs.size());

        // build the inputs

        SifParams nodeInputParams;

        nodeInputParams.inputTensorsNr            = inputs.size();
        nodeInputParams.inputTensors              = inputs.data();
        nodeInputParams.outputTensorsNr           = outputs.size();
        nodeInputParams.nodeParams.nodeParams     = node.m_nodeParams.data();
        nodeInputParams.nodeParams.nodeParamsSize = node.m_nodeParams.size();

        nodeInputParams.inputPermutations =
            node.m_inputPermutations.empty() ? nullptr : node.m_inputPermutations.data();
        nodeInputParams.outputPermutations =
            node.m_outputPermutations.empty() ? nullptr : node.m_outputPermutations.data();

        for (unsigned i = 0; i < node.m_inputs.size(); ++i)
        {
            auto& inputDescription = node.m_inputs[i];
            if (inputDescription.m_isInternal)
            {
                if (!tensorSet[inputDescription.m_index])
                {
                    LOG_CRITICAL(GC, "Internal input tensor {} for fused node not set", i);
                }

                nodeInputParams.inputTensors[i] = &internalTensors[inputDescription.m_index];

                // update internal input tensors in multi sif info
                // from internalTensors[inputDescription.m_index]
                // this tensor is set (see tensorSet above), that is, it was an output
                // of some other internal node that has already ran its SIF
                // so its size is up to date

                TSize tSizes[tpc_lib_api::MAX_TENSOR_DIM];
                memcpy(tSizes, nodeInputParams.inputTensors[i]->geometry.maxSizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));
                if (inferMax)
                {
                    inputDescription.m_shape.setMaxSize(tSizes);
                }
                else
                {
                    inputDescription.m_shape.setMinSize(tSizes);
                }
            }
            else
            {
                // see FuserGraphToMultiSifData in tpc_fuser.cpp
                if (inputDescription.m_takeFromOutput)
                {
                    nodeInputParams.inputTensors[i] = sifOutputs->outputTensors[inputDescription.m_index];
                }
                else
                {
                    nodeInputParams.inputTensors[i] = sifParams->inputTensors[inputDescription.m_index];
                }
            }
        }

        // build the outputs

        SifOutputs nodeOutputData;

        nodeOutputData.outputTensors = outputs.data();
        size_t invalidMaskSize       = 1;

        if (!outputs.empty())
        {
            invalidMaskSize = div_round_up(outputs.size(), BITS_IN_UINT32);
        }
        std::vector<uint32_t> nodeInvalidMask(invalidMaskSize);
        nodeOutputData.invalidMask = nodeInvalidMask.data();

        for (unsigned i = 0; i < outputs.size(); ++i)
        {
            const auto& outputDescription = node.m_outputs[i];
            if (outputDescription.m_isInternal)
            {
                internalTensors[outputDescription.m_index].geometry.dims = outputDescription.m_shape.getDim();
                auto sizes                                               = outputDescription.m_shape.getMaxSizes();
                std::copy(std::cbegin(sizes),
                          std::cend(sizes),
                          std::begin(internalTensors[outputDescription.m_index].geometry.maxSizes));

                nodeOutputData.outputTensors[i]      = &internalTensors[outputDescription.m_index];
                tensorSet[outputDescription.m_index] = true;
            }
            else
            {
                nodeOutputData.outputTensors[i] = sifOutputs->outputTensors[outputDescription.m_index];
            }
        }

        sif_t                       sif;
        tpc_lib_api::GlueCodeReturn sifRet;

        if (node.m_sifID.sm_tableid == ShapeFuncOrigin::LIB_ID_RESERVED_FOR_GC_SIF)
        {
            // retrieve and run the sif (not a TPC node)
            sif    = ShapeFuncRegistry::instance().getSIF(node.m_sifID);
            sifRet = sif(deviceTypeToDeviceID(deviceType), &nodeInputParams, &nodeOutputData);
        }
        else
        {
            // run the sif (TPC node)
            nodeInputParams.maxAvailableTpc = TPCNode::getMaxAvailableTpc(deviceTypeToDeviceID(deviceType));
            sifRet = KernelDB::instance().RunShapeInferenceFunction(deviceTypeToDeviceID(deviceType),
                                                                    node.m_nodeGUID,
                                                                    &nodeInputParams,
                                                                    &nodeOutputData);
        }

        if (sifRet != tpc_lib_api::GLUE_SUCCESS)
        {
            LOG_ERR(GC, "SIF for subnode idx {} returned failure: {}", subnodeIdx, enumToString(sifRet));
            return sifRet;
        }

        // update internal output tensors in multi sif info
        for (unsigned i = 0; i < outputs.size(); ++i)
        {
            auto& outputDescription = node.m_outputs[i];

            // convert glue-code size array type to tensors' size array type
            TSize tSizes[tpc_lib_api::MAX_TENSOR_DIM];
            memcpy(tSizes, nodeOutputData.outputTensors[i]->geometry.maxSizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));

            if (outputDescription.m_isInternal)
            {
                if (inferMax)
                {
                    outputDescription.m_shape.setMaxSize(tSizes);
                }
                else
                {
                    outputDescription.m_shape.setMinSize(tSizes);
                }
            }
        }

        // collect invalid tensor mask
        for (unsigned i = 0; i < outputs.size(); ++i)
        {
            const auto& outputDescription = node.m_outputs[i];
            if (!outputDescription.m_isInternal)
            {
                unsigned inputOffset = i / BITS_IN_UNSIGNED;
                unsigned inputBit    = i % BITS_IN_UNSIGNED;
                if ((nodeOutputData.invalidMask[inputOffset] & (1 << inputBit)) != 0)
                {
                    const auto& outputDescription   = node.m_outputs[i];
                    auto        externalOutputIndex = outputDescription.m_index;
                    // set bit in sifOutputs.
                    unsigned outputOffset = externalOutputIndex / BITS_IN_UNSIGNED;
                    unsigned outputBit    = externalOutputIndex % BITS_IN_UNSIGNED;

                    sifOutputs->invalidMask[outputOffset] |= 1 << outputBit;
                }
            }
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
