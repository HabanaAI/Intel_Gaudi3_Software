#include "habana_pass.h"
#include "habana_graph.h"
#include "defs.h"

bool validateExecutionScheduleBundles(HabanaGraph& g)
{
    Settable<unsigned int> currentBundle;
    std::set<unsigned int> previousBundles;
    unsigned currentOperation = 0;
    for (NodePtr node : g.getExeSortedNodes())
    {
        //We don't check logical operations
        if (node->isLogicalOperation())
        {
            continue;
        }
        if (node->getNodeAnnotation().bundleInfo.is_set())
        {
            //To see if we are before the first bundle
            if (currentBundle.is_set())
            {
                //We are in the same bundle check operation index increased
                if (currentBundle.value() == node->getNodeAnnotation().bundleInfo->bundleIndex)
                {
                    HB_ASSERT(currentOperation <= node->getNodeAnnotation().bundleInfo->operationIndex,
                          "operation {} was scheduled after operation {} in bundle {}",
                          node->getNodeAnnotation().bundleInfo->operationIndex,
                          currentOperation,
                          currentBundle.value());
                }
                //We are in a new bundle. Make sure we weren't in it already
                else
                {
                    HB_ASSERT(previousBundles.find(node->getNodeAnnotation().bundleInfo->bundleIndex)
                          == previousBundles.end(),
                          "validateExecution bundle id {} in the middle of bundle {}",
                          node->getNodeAnnotation().bundleInfo->bundleIndex,
                          currentBundle.value());
                    previousBundles.insert(currentBundle.value());
                }
            }
            currentOperation = node->getNodeAnnotation().bundleInfo->operationIndex;
            currentBundle.set(node->getNodeAnnotation().bundleInfo->bundleIndex);
        }
    }
    return true;
}
