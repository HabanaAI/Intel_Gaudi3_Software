#pragma once

#include "syn_object.hpp"
#include "syn_tensor.hpp"

namespace syn
{
class Node;
using Nodes      = std::vector<Node>;
using UserParams = std::vector<uint8_t>;

class Node : public SynObject<synNodeId>
{
public:
    Node() = default;

    synNodeId getId() const { return handle(); }

private:
    Node(const std::shared_ptr<synNodeId>& handle) : SynObject(handle) {}

    friend class GraphBase;  // GraphBase class requires access to Node private constructor
};
}  // namespace syn