#pragma once

#include <sstream>
#include <string>
#include "types.h"

class HabanaGraph;

class GraphComparator
{
public:
    bool compareGraphs(const HabanaGraph& originalGraph, HabanaGraph& newGraph);

private:
    template<typename T>
    void checkEqual(T origValue, T newValue, const std::string& valueName);
    template<typename T>
    void checkNonEqual(T origValue, T newValue, const std::string& valueName);
    template<typename T>
    void checkEqualBuffer(T* origBuffer, T* newBuffer, const std::string& bufferName, unsigned bufferSize);
    template<typename T>
    bool checkPointers(std::shared_ptr<T> origPointer, std::shared_ptr<T> newPointer, const std::string& pointerName);

    void compareConvNodesParams(const NodePtr origNode, const NodePtr newNode);
    void compareTensors(const TensorPtr origTensor, const TensorPtr newTensor);
    void compareNodes(const NodePtr origNode, const NodePtr newNode);

    void postCompareFailure();

    bool              m_localCompareResult  = true; // compare result of specific tensors / nodes pair
    bool              m_globalCompareResult = true; // compare result of the entire graph and all tensors / nodes pairs
    std::stringstream m_errString;                  // stores collected error strings for local compare
};
