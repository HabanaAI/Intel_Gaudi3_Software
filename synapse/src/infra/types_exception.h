#pragma once

#include "synapse_common_types.h"  // for synStatus

#include <exception>
#include <string>
#include <utility>

class SynapseException : public std::exception
{
    // TODO: call-stack may be printed as part of the object constructor
public:
    SynapseException(std::string exceptionString) : m_exceptionString(std::move(exceptionString)) {}
    const char* what() const noexcept { return m_exceptionString.c_str(); }

private:
    std::string m_exceptionString;
};

class SynapseStatusException : public SynapseException
{
public:
    SynapseStatusException(std::string exceptionString, synStatus status = synFail)
    : SynapseException(std::move(exceptionString)), m_status(status)
    {
    }

    synStatus status() const noexcept { return m_status; }

private:
    synStatus m_status;
};

class IllegalGroupParams : public SynapseException
{
public:
    IllegalGroupParams() : SynapseException("Illegal Group Params!") {}

    IllegalGroupParams(std::string nodeName) : SynapseException("Illegal Group Params! Node name: " + nodeName) {}
};

class CannotResizeTensor : public SynapseException
{
public:
    CannotResizeTensor() : SynapseException("Cannot Resize Tensor") {}
};

class NoDmaNodesException : public SynapseException
{
public:
    NoDmaNodesException() : SynapseException("No Dma Nodes Exception!") {}
};

class PassFailedException : public SynapseException
{
public:
    PassFailedException() : SynapseException("Pass Failed!") {}
};

class InvalidNodeParamsException : public SynapseException
{
public:
    InvalidNodeParamsException() : SynapseException("Invalid Node Params!") {}

    InvalidNodeParamsException(std::string nodeName) : SynapseException("Invalid Node Params! Node name: " + nodeName)
    {
    }

    InvalidNodeParamsException(std::string nodeName, std::string paramName)
    : SynapseException("Invalid Node Params! Node name: " + nodeName + ", param name : " + paramName)
    {
    }
};

class InvalidNodeParamsSizeException : public SynapseException
{
public:
    InvalidNodeParamsSizeException(std::string nodeName)
    : SynapseException("Invalid Size for Node Params! Node name: " + nodeName)
    {
    }

    InvalidNodeParamsSizeException(std::string nodeName, unsigned paramSizeOrig, unsigned paramSize)
    : SynapseException("Invalid Size for Node Params! Node name: " + nodeName + ", user param size is :" +
                       std::to_string(paramSizeOrig) + ", should be :" + std::to_string(paramSize))
    {
    }
};

class NodeHasNoInput : public SynapseException
{
public:
    NodeHasNoInput() : SynapseException("The node has no input at a given index.") {}

    NodeHasNoInput(std::string nodeName)
    : SynapseException("The node has no input at a given index. Node name: " + nodeName)
    {
    }
};

class InvalidSplitFactor : public SynapseException
{
public:
    InvalidSplitFactor() : SynapseException("Invalid slice factor given to range") {}
};

class InvalidPipelineParamsException : public SynapseException
{
public:
    InvalidPipelineParamsException() : SynapseException("Invalid Pipeline Params!") {}
};

class SyncValueOverflow : public SynapseException
{
public:
    SyncValueOverflow() : SynapseException("Sync value overflow") {}
};

class InvalidTensorParamsException : public SynapseException
{
public:
    InvalidTensorParamsException() : SynapseException("Invalid Tensor Params!") {}
};

class InvalidTensorSizeException : public SynapseException
{
public:
    InvalidTensorSizeException() : SynapseException("Invalid Tensor Size!") {}
    InvalidTensorSizeException(std::string tensorName) : SynapseException("Invalid Tensor " + tensorName + " Size!") {}
};

class DeviceLimitationFP32Exception : public SynapseException
{
public:
    DeviceLimitationFP32Exception() : SynapseException("Device Limitation - float32 operation is not supported") {}
    DeviceLimitationFP32Exception(const std::string& nodeName)
    : SynapseException("Device Limitation - float32 operation in node " + nodeName + " is not supported")
    {
    }
};
