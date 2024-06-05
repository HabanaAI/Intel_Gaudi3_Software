#include <habana_graph.h>

bool forceExplicitPadding(HabanaGraph& g)
{
    // TODO SW-89319 -- remove this check when Gaudi2 support is added
    bool isGaudi1Platform = g.getDeviceType() == synDeviceGaudi;

    for (const auto& node : g.getNodes())
    {
        const auto& convNode = std::dynamic_pointer_cast<ConvBaseNode>(node);
        if (convNode != nullptr)
        {
            auto& params = convNode->getConvolutionParams();
            // force padding to be explicit if we can get away with it
            if (params.paddingType == PADDING_SAME)
            {
                if (!convNode->isDynamicPaddingConvolution())
                {
                    params.paddingType = PADDING_EXPLICIT;
                }
                else if (!isGaudi1Platform)
                {
                    LOG_ERR(GC,
                            "Node {}: PADDING_SAME not convertible to PADDING_EXPLICIT found. This is only supported "
                            "on the Gaudi platform at this time",
                            convNode->getNodeName());
                    return false;
                }
            }
        }
    }
    return true;
}
