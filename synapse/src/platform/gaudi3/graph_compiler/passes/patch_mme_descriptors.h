#pragma once

#include "graph_compiler/habana_nodes/mme_node.h"
#include "include/gaudi3/mme_descriptor_generator.h"

namespace gaudi3
{
class MMETensorPatcher
{
public:
    static void patchTensors(const MmeNode& mmeNode, MmeDescriptorGenerator& descGenerator);
};

};  // namespace gaudi3
