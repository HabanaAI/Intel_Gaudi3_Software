#include <habana_graph.h>

bool snapshotPreSlicingSizes(HabanaGraph& graph)
{
    if (!graph.isDynamicShape())
    {
        return true;
    }

    const auto& nodes = graph.getNodes();

    for (const auto& node: nodes)
    {
        const auto& outputs = node->getOutputs();

        for (const auto& output: outputs)
        {
            if (output->isDynamicShape())
            {
                output->getTensorAnnotation().m_preSlicingSize.emplace(output->getShape().getMinSizes());
            }
        }
    }

    return true;
}
