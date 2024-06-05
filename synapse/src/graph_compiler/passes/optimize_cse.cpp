#include "graph_editor.h"
#include "habana_graph.h"
#include "passes.h"

/*
    Common Sub-Expression Elimination

    we use dynamic programming to create a hash for every tensor.
    this hash should represent the actual data in the tensor, it is made of all the operations producing that tensor.
    method:
    starting condition - if a tensor doesn't have a producer, the entire data is represented by its uniue tensor ID.
    otherwise - for every node, we calculate a hash from its calculative operation, and its inputs.
    2 tensors will contain the same data if their producers have the same hash. in that case, we will remove the first
   producer.
*/

namespace CSEElimiator
{
using OperandHash = uint64_t;  // hash should represent the actual data in every tensor
using NodeHash    = uint64_t;  // hash should represent the node calculation method

// apparently std::hash is the identity function for integers, which is really bad.
uint64_t hash(uint64_t x)
{
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9LLU;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebLLU;
    x = x ^ (x >> 31);
    return x;
}

OperandHash hash(const std::string& x)
{
    return std::hash<std::string> {}(x);
}

template<class T>
inline void hashCombine(uint64_t& s, const T& v)
{
    // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
    s ^= hash(v) + 0x9e3779b97f4a7c15LLU + (s << 12) + (s >> 4);
}

template<class T>
inline void hashCombine(OperandHash& s, const std::vector<T>& vector)
{
    for (const auto& v : vector)
    {
        hashCombine(s, v);
    }
}

inline void hashCombine(OperandHash& s, const RawParamsData& data)
{
    for (const auto& v : data)
    {
        hashCombine(s, v);
    }
}

// data to be used for calculating node hash
struct NodeCSEData
{
    NodeCSEData(const HabanaGraph& g, const std::unordered_map<NodePtr, NodeCSEData>& nodeDataMap, const NodePtr& node);
    bool operator==(const NodeCSEData& other)
    {
        return (operationHash == other.operationHash) && (inputsHash == other.inputsHash);
    }

    std::vector<OperandHash> inputsHash;     // hash for every input
    NodeHash                 operationHash;  // hash for the node operation
    NodePtr                  originalNode;   // node
};
using NodeDataMap = std::unordered_map<NodePtr, NodeCSEData>;

struct TensorCSEData
{
    TensorCSEData(const NodeDataMap& nodeDataMap, const TensorPtr& t, const NodePtr& producer);
    OperandHash getTensorHash() const;  // calculate the tensor hash - should reflect the data of the tensor.
    std::string toString() const;       // debug function - get string to represent the tensor hash

    std::optional<NodeCSEData> producerData;       // NodeData of producer
    unsigned                   producerOutputIdx;  // output index of producer
    TensorPtr                  originalTensor;     // tensor
};
using HashedTensorsMap = std::unordered_map<OperandHash, TensorPtr>;

// debug function - get string to represent the tensor hash
std::string TensorCSEData::toString() const
{
    std::stringstream ss;
    const TensorPtr&  t = originalTensor;

    if (producerData.has_value())
    {
        ss << "node type: " << producerData->originalNode->getNodeType() << ", ";
        ss << "node guid: " << producerData->originalNode->getGUID() << ", ";
        ss << "node hash: " << producerData->operationHash << ", ";
        ss << "outputIdx: " << producerOutputIdx << ", ";
        ss << "inputsHash: ";
        for (const OperandHash& input : producerData->inputsHash)
        {
            ss << input << ",\t";
        }
    }
    else if (t->isShapeTensor() && !t->isDynamicShape())
    {
        ss << "static shape tensor: " << t->getDimSizesStr();
    }
    else if (t->isStaticParam() && t->getData() && t->getDim() == 1) // we only hash on staticTensor with 1Dim
    {
        char* buffer = t->getData();
        ss << "static tensor start buffer: " << buffer;
    }
    else
    {
        ss << "unique " << originalTensor->getId();
        return ss.str();
    }
    return ss.str();
}

// calculate a hash based on the node operation.
NodeHash getNodeOperationHash(const NodePtr& node)
{
    NodeHash hash = 0;
    hashCombine(hash, node->getGUID());
    hashCombine(hash, node->getNodeType());
    hashCombine(hash, node->getParamsRawData());
    return hash;
}

OperandHash getShapeHash(const TensorPtr& t)
{
    OperandHash hash = 0;
    for (unsigned i = 0; i < t->getDim(); i++)
    {
        hashCombine(hash, t->getSizeInElements(i));
        hashCombine(hash, t->getMinimalSizeInElements(i));
    }
    return hash;
}

// calculate the tensor hash - should reflect the data of the tensor.
OperandHash TensorCSEData::getTensorHash() const
{
    OperandHash      hash = 0;
    const TensorPtr& t    = originalTensor;

    if (producerData.has_value())
    {
        // tensor hash contains:
        hashCombine(hash, getShapeHash(t));                        // tensor shape
        hashCombine(hash, producerData->operationHash);            // producer operation hash
        hashCombine(hash, producerOutputIdx);                      // producer output index
        hashCombine(hash, static_cast<uint64_t>(t->getElementType()));  // data type
        for (const OperandHash& input : producerData->inputsHash)  // producer inputsHash
        {
            hashCombine(hash, input);
        }
    }
    else if (t->isShapeTensor() && !t->isDynamicShape())
    {
        hashCombine(hash, getShapeHash(t));  // tensor shape
    }
    else if (t->isStaticParam() && t->getData() && t->getDim() == 1) // we only hash on staticTensor with 1Dim
    {
        char* buffer = t->getData();
        std::vector<char> data(buffer, buffer + t->getTotalSizeInBytes());
        hashCombine<char>(hash, data);  // hash on the tensor data

    }
    else  // no producer and not a static shape tensor
    {
        hashCombine(hash, t->getId());  // starting condition of the dynamic programming
    }
    return hash;
}

bool isInputReuseBinding(const HabanaGraph& g, const TensorPtr& input)
{
    // if a tensor is consumed by a node with "binding reuse" then eliminating that tensor will result in a memcopy for
    // each additional consumer
    for (const NodePtr& consumer : g.getTensorConsumers(input))
    {
        for (auto bindingReuse : consumer->getReusableInputBinding())
        {
            if (std::find(bindingReuse.second.begin(), bindingReuse.second.end(), input) != bindingReuse.second.end())
            {
                LOG_DEBUG(CSE_OPTIMIZATION,
                          "tensor {}, marked unique since it is written over by a tpc node (binding reuse)",
                          input->getName());
                return true;
            }
        }
    }
    return false;
}

bool isReductionInput(const HabanaGraph& g, const TensorPtr& t)
{
    const auto& consumers = g.getTensorConsumers(t);
    return consumers.size() == 1 && consumers.front()->getNodeType() == Node::TYPE_INTERNAL_REDUCTION;
}

bool isConstantProducerAndSingleTpcConsumer(const HabanaGraph& g, const TensorPtr& t)
{
    const auto producer = std::dynamic_pointer_cast<TPCNode>(g.getTensorProducer(t));
    if (!producer) return false;
    if (producer->getGUIDWithoutDtype() == "constant")
    {
        const auto consumer = g.getTensorSingleConsumer(t);
        if (!consumer) return false;
        if (HabanaGraph::runsOnTPC(consumer))
        {
            LOG_TRACE(CSE_OPTIMIZATION, "prevent CSE from deleting {}, so const can be fused", t->getName());
            return true;
        }
    }
    return false;
}

// check if we are allowed to remove the tensor
bool isUnique(const HabanaGraph& g, const TensorPtr& t)
{
    if (t->isUserManagedDram()) return true;     // user expects this specific tensor to exist
    if (isReductionInput(g, t)) return true;     // internal reduction must be the single consumer of all its inputs
    if (isInputReuseBinding(g, t)) return true;  // tensor is needed for a write-over by another node.
    // fusing this pattern: (constant) --> (tpc node) leads to better performance (even though const is recomputed)
    // because const can be done in fuser code (meanning its output will be read from the registers).
    if (isConstantProducerAndSingleTpcConsumer(g, t)) return true;
    return false;
}

// data to be used for calculating tensor has
TensorCSEData::TensorCSEData(const NodeDataMap& nodeDataMap, const TensorPtr& t, const NodePtr& producer)
{
    originalTensor = t;
    if (producer != nullptr)
    {
        producerData      = nodeDataMap.at(producer);
        producerOutputIdx = producerData->originalNode->getOutputIndexOfTensor(t);
    }
}

NodeCSEData::NodeCSEData(const HabanaGraph& g, const NodeDataMap& nodeDataMap, const NodePtr& node)
{
    originalNode  = node;
    operationHash = getNodeOperationHash(node);
    for (const TensorPtr& in : node->getInputs())
    {
        if (in == nullptr) continue;
        TensorCSEData tensorData(nodeDataMap, in, g.getTensorProducer(in));
        inputsHash.push_back(tensorData.getTensorHash());
    }
}

void eliminateCSE(HabanaGraph& g)
{
    HashedTensorsMap tensorDataMap;  // map: TensorHash -> Tensor
    NodeDataMap      nodeDataMap;    // map: Node -> NodeData

    tensorDataMap.reserve(g.getTensors().size());
    nodeDataMap.reserve(g.getNumNodes());

    auto topoSortedNodes = g.getTopoSortedNodes();  // copy by value because it may change
    for (const NodePtr& n : topoSortedNodes)
    {
        HB_ASSERT_PTR(n);
        // save the current node data in map for later use
        NodeCSEData nodeData(g, nodeDataMap, n);
        nodeDataMap.emplace(std::make_pair(n, nodeData));

        // for every output, check if it can be replaced by another producer
        unsigned               numberOfDuplicatedOutputs = 0;
        std::optional<NodePtr> newProducer;
        for (const TensorPtr& output : n->getOutputs())
        {
            if (!output) continue;
            // calculate tensor hash for 'output'
            TensorCSEData tensorData(nodeDataMap, output, n);
            OperandHash   hash = tensorData.getTensorHash();
            LOG_TRACE(CSE_OPTIMIZATION,
                      "tensor {}, hash: {}, data: {}",
                      output->getName(),
                      hash,
                      tensorData.toString());

            // check if a tensor with identical hash already exists
            auto res = tensorDataMap.emplace(std::make_pair(hash, output));
            if (res.second == false && !isUnique(g, output))
            {
                LOG_DEBUG(CSE_OPTIMIZATION, "detected common sub-expression for tensor {}", output->getName());
                const TensorPtr& substituteTensor = res.first->second;
                LOG_DEBUG(CSE_OPTIMIZATION, "substitue tensor {}", substituteTensor->getName());
                const NodePtr& producer = g.getTensorProducer(substituteTensor);
                if (!newProducer.has_value())
                {
                    newProducer = producer;
                }
                numberOfDuplicatedOutputs++;

                // if the hash function is ok, this has the same chance of randomly picking the same ant twice.
                HB_ASSERT(nodeDataMap.at(newProducer.value()) == nodeData, "{}, invalid hash conflict", __FUNCTION__);
                HB_ASSERT(newProducer == producer, "{}, invalid new producer", __FUNCTION__);
            }
        }

        // if all outputs can be replaced
        if (numberOfDuplicatedOutputs > 0 && numberOfDuplicatedOutputs == n->getNumOutputs())
        {
            if (n->getNumOutputs(Node::TENSOR_TYPE_CONTROL) > 0 ||
                newProducer.value()->getNumOutputs(Node::TENSOR_TYPE_CONTROL) > 0)
            {
                // [CID: 42199] False positive - Uninitialized scalar variable defects caused by usage of std::optional,
                // link:
                // https://community.synopsys.com/s/article/FP-Uninitialized-scalar-variable-defects-caused-by-usage-of-std-optional
                continue;  // for now do not replace CSE that have control dependencies. can be removed after [SW-76501]
            }
            LOG_INFO(CSE_OPTIMIZATION,
                     "elmination common sub-expression of node {} and replacing with {}",
                     n->getNodeName(),
                     newProducer.value()->getNodeName());
            GraphEditor::removeNode(g, n, newProducer.value());
        }
    }
}

};  // namespace CSEElimiator

bool commonSubExpressionElimination(HabanaGraph& g)
{
    if (!GCFG_ENABLE_CSE_OPTIMIZATION.value()) return true;
    CSEElimiator::eliminateCSE(g);
    return true;
}