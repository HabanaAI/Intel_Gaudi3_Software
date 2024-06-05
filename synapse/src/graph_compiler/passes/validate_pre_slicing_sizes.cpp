#include <habana_graph.h>
#include <log_manager.h>
#include <habana_global_conf.h>

static bool cmpSizes (const SizeArray& sz1, const SizeArray& sz2, std::size_t dim)
{
    auto beg = std::begin(sz1);
    auto end = beg + std::min(std::size(sz1), dim);
    return std::equal(beg, end, std::begin(sz2));
}

bool validatePreSlicingSizes(HabanaGraph& graph)
{
    if (!graph.isDynamicShape())
    {
        return true;
    }

    const auto& nodes = graph.getNodes();

    bool forceFailure = GCFG_ENFORCE_POST_SLICING_SHAPE_CHECK.value();

    LOG_DEBUG (GC, "Validating pre-slicing sizes");
    for (const auto& node: nodes)
    {
        const auto& outputs = node->getOutputs();

        for (const auto& output: outputs)
        {
            if (output->isDynamicShape())
            {
                if (output->getTensorAnnotation().m_preSlicingSize)
                {
                    if (!cmpSizes(*output->getTensorAnnotation().m_preSlicingSize, output->getShape().getMinSizes(), output->getDim()))
                    {
                        // Do NOT assert/throw and do NOT return false here if validation fails.
                        // We are just gathering info at this stage, not failing compilations.
                        // return false will be added in a separate commit if necessary
                        SYN_LOG_TYPE(GC,
                                (forceFailure ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_WARN),
                                "Validating pre-slicing sizes: node {} tensor {} ({}) changed size after slicing!"
                                "Old size [{}] new size [{}]",
                                node->getNodeName(), output->getName(), output->getId(),
                                fmt::join(output->getTensorAnnotation().m_preSlicingSize.value(), ","),
                                fmt::join(output->getShape().getMinSizes(), ","));
                        if (forceFailure)
                        {
                            return false;
                        }
                    }
                }
            }
        }
        LOG_DEBUG (GC, "Validating pre-slicing sizes: node {} OK", node->getNodeName());
    }

    return true;
}
