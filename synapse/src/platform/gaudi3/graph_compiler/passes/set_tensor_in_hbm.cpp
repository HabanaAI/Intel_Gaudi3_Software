#include "set_tensor_in_hbm.h"
#include "cache_types.h"

namespace gaudi3
{
bool setTensorInHbm(Gaudi3Graph& g)
{
    TensorInHbmSetter tensorInHbmSetter(&g);
    return tensorInHbmSetter.setInHbm();
}

TensorInHbmSetter::TensorInHbmSetter(Gaudi3Graph* graph) : m_graph(graph) {}

bool TensorInHbmSetter::setInHbm()
{
    const bool bCompilerDecidesDirectives = (GCFG_DEFAULT_CACHE_DIRECTIVE.value() == 0);

    CacheMetaData cacheMetaData;

    for (auto& node : m_graph->getNodes())
    {
        // if someone (e.g. some test) already filled the inputs cache metadata, then skip
        if (node->getNodeAnnotation().inputsCacheMetaData.empty())
        {
            node->getNodeAnnotation().inputsCacheMetaData.reserve(node->getNumInputs());
            for (const TensorPtr& tensor : node->getInputs())
            {
                if (tensor == nullptr)
                {
                    node->getNodeAnnotation().inputsCacheMetaData.emplace_back();  // ignored, but should have an entry.
                    continue;
                }

                if (tensor->inSram() && !tensor->isAliasedTensor())
                {
                    cacheMetaData.cacheDirective = CacheDirective::HomeAllocate;
                    tensor->setTensorInWorkspace();
                }
                else
                {
                    cacheMetaData.cacheDirective = CacheDirective::NoAllocate;
                }

                if (bCompilerDecidesDirectives)
                {
                    node->getNodeAnnotation().inputsCacheMetaData.push_back(cacheMetaData);
                }
                else
                {
                    node->getNodeAnnotation().inputsCacheMetaData.emplace_back();  // default
                }
            }
        }

        // if someone (e.g. some test) already filled the outputs cache metadata, then skip
        if (node->getNodeAnnotation().outputsCacheMetaData.empty())
        {
            node->getNodeAnnotation().outputsCacheMetaData.reserve(node->getNumOutputs());
            for (const TensorPtr& tensor : node->getOutputs())
            {
                if (tensor == nullptr)
                {
                    node->getNodeAnnotation()
                        .outputsCacheMetaData.emplace_back();  // ignored, but should have an entry.
                    continue;
                }

                if (tensor->inSram() && !tensor->isAliasedTensor())
                {
                    cacheMetaData.cacheDirective = CacheDirective::HomeAllocate;
                    tensor->setTensorInWorkspace();
                }
                else
                {
                    cacheMetaData.cacheDirective = CacheDirective::NoAllocate;
                }

                if (bCompilerDecidesDirectives)
                {
                    node->getNodeAnnotation().outputsCacheMetaData.push_back(cacheMetaData);
                }
                else
                {
                    node->getNodeAnnotation().outputsCacheMetaData.emplace_back();  // default
                }
            }
        }
    }

    return true;
}
}  // namespace gaudi3