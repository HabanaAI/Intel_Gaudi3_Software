#include "habana_graph.h"
#include "mme_node.h"

bool validateMMENodes(HabanaGraph& g)
{
    for (pNode& node : g.getSortedMMENodes())
    {
        std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        HB_ASSERT(mmeNode, "could not downcast Node to MME Node");
        CHECK_RET_FALSE(!mmeNode->hasBias() && !mmeNode->hasCin(),
                        "MME Node {} cannot have bias or Cin at this stage.", mmeNode->getNodeName());
    }
    return true;
}
