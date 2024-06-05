#pragma once

class TPCNode;

namespace eager_mode
{
class EagerGraph;
}  // namespace eager_mode

namespace eager_mode::glue
{
bool loadKernelAndAllocAuxTensors(EagerGraph& g, TPCNode& n);

}  // namespace eager_mode::glue