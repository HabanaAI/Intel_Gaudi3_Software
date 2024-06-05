#pragma once
#include "synapse_api.h"
#include "graph_manager.h"

template<class T_PARAM>
static void createNode(const std::list<pManagedTensor>& inputs, const std::list<pManagedTensor>& outputs,
                       const std::string& nodeName, const std::string& operationName, synGraphHandle handle,
                       T_PARAM &params)
{
    ManagedNodeWithParams<T_PARAM> node(inputs, outputs, nodeName, operationName, handle, params);
    node.createNode();
}

static void createNode(const std::list<pManagedTensor>& inputs, const std::list<pManagedTensor>& outputs,
                       const std::string& nodeName, const std::string& operationName, synGraphHandle handle)
{
    ManagedNode node(inputs, outputs, nodeName, operationName, handle);
    node.createNode();
}


static void setConvolutionParams(synConvolutionParams &params, unsigned dH, unsigned dW, unsigned kH, unsigned kW, int padT,
                                 int padB, int padL, int padR, unsigned dilW, unsigned dilH)
{
    params.dH = dH;
    params.dW = dW;
    params.kH = kH;
    params.kW = kW;
    params.padT = padT;
    params.padB = padB;
    params.padL = padL;
    params.padR = padR;
    params.dilH = dilH;
    params.dilW = dilW;
}
