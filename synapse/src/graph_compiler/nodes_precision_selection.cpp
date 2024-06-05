#include "habana_graph.h"
#include "data_type_utils.h"
#include "nodes_precision_selection.h"

NodesPrecisionSelection::NodesPrecisionSelection()
{
    m_minNodePrecision = syn_type_na;

    m_profilePrecision = getSynDataTypeFromString(GCFG_PROFILE_PRECISION.value());
    HB_ASSERT(m_profilePrecision != syn_type_na, "{} is an invalid profile precision", GCFG_PROFILE_PRECISION.value());

    m_precisionToRaise = getSynDataTypeFromString(GCFG_PRECISION_TO_RAISE.value());
    HB_ASSERT(GCFG_PRECISION_TO_RAISE.value() == "" || m_precisionToRaise != syn_type_na,
              "{} is an invalid precision to raise value",
              GCFG_PRECISION_TO_RAISE.value());
}

bool NodesPrecisionSelection::runSetNodesPrecision(HabanaGraph& g)
{
    if (!GCFG_SYNAPSE_DATA_TYPE_SELECTION.value())
    {
        LOG_DEBUG(DATA_TYPES, "Data type selection is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    /* There are 6 factors that affect the chosen node precision:
     *   1) User node precision         user-provided per node name
     *   2) User node type precision    user-provided per node type (guid)
     *   3) Cast node                   whether the node is a cast or not
     *   4) Node type min precision     algo's recommended minimum precision
     *   5) Profile precision           defined in habana_global_conf.h, default value is int8
     *   6) Number of layers to raise   defined in habana_global_conf.h, default value is 2
     */

    int numOfLayersToRaise = GCFG_NUM_OF_LAYERS_TO_RAISE.value();
    // go through the graph nodes and call setNodePrecision() for each of them
    const NodeVector& nodes = g.getTopoSortedNodes();
    for (const NodePtr& node : nodes)
    {
        bool layerRaised = setNodePrecision(g, node, (numOfLayersToRaise > 0));
        if (layerRaised)
        {
            numOfLayersToRaise -= 1;
        }
        LOG_TRACE(DATA_TYPES,
                  "{}: Node '{}' precision set to: {}",
                  HLLOG_FUNC,
                  node->getNodeName(),
                  getStringFromSynDataType(node->getNodePrecision()));
    }

    return true;
}

bool NodesPrecisionSelection::ignoreUserDataType(const NodePtr& node)
{
    if (!HabanaGraph::runsOnTPC(node))
    {
        return GCFG_IGNORE_USER_DATA_TYPE.value();
    }

    return false;
}

bool NodesPrecisionSelection::setNodePrecision(HabanaGraph& g, const NodePtr& node, bool toRaiseLayer)
{
    LOG_TRACE(DATA_TYPES, "{}: Starting precision selection for node '{}'", HLLOG_FUNC, node->getNodeName());

    bool isLayerRaised = false;

    // 1) if user provided a node precision - take it and end
    synDataType userNodePrecision = node->getNodePrecision();
    if (userNodePrecision != syn_type_na && !ignoreUserDataType(node))
    {
        LOG_TRACE(DATA_TYPES,
                  "{}: User node precision is provided ({}) - taking it",
                  HLLOG_FUNC,
                  getStringFromSynDataType(userNodePrecision));
        return isLayerRaised;
    }

    // 2) if user provided a node type precision - take it and end
    synDataType userNodeTypePrecision;
    if (g.getUserNodeTypePrecision(node->getGUID(), userNodeTypePrecision))
    {
        LOG_TRACE(DATA_TYPES,
                  "{}: User node type precision is provided ({}) - taking it",
                  HLLOG_FUNC,
                  getStringFromSynDataType(userNodeTypePrecision));
        node->setNodePrecision(userNodeTypePrecision);
        return isLayerRaised;
    }

    // 3) cast node - predict node precision based on its predecessor
    if (node->isCast())
    {
        LOG_TRACE(DATA_TYPES, "{}: Node is of type cast - taking precision from predecessor", HLLOG_FUNC);
        node->setNodePrecision(getNodePrecisionFromPredecessor(g, node));
        return isLayerRaised;
    }

    // 4) handle the node type minimum precision
    synDataType precision = g.getNodeTypeMinPrecision(node);
    // handle nodes that don't have minimum precision ("don't care")
    if (precision == syn_type_na)
    {
        LOG_TRACE(DATA_TYPES, "{}: Node is of type \"don't care\" - taking min precision from successors", HLLOG_FUNC);
        precision = getNodePrecisionFromSuccessors(g, node);
    }
    LOG_TRACE(DATA_TYPES, "{}: Min node precision is: {}", HLLOG_FUNC, getStringFromSynDataType(precision));

    // 5) get max between factors 4 and 5
    precision = getHighestGUIDDataType({precision, m_profilePrecision});
    LOG_TRACE(DATA_TYPES,
              "{}: Max between the min node precision and the profile precision ({}) is: {}",
              HLLOG_FUNC,
              getStringFromSynDataType(m_profilePrecision),
              getStringFromSynDataType(precision));

    // 6) raise layer precision if needed
    if (toRaiseLayer && (precision == syn_type_fp8_143 ||
                         precision == syn_type_fp8_152 ||
                         precision == syn_type_int8    ||
                         precision == syn_type_uint8))
    {
        LOG_TRACE(DATA_TYPES,
                  "{}: Raising node precision from {} to {}",
                  HLLOG_FUNC,
                  getStringFromSynDataType(precision),
                  getStringFromSynDataType(getPrecisionToRaise()));

        precision     = getPrecisionToRaise();
        isLayerRaised = true;
    }

    node->setNodePrecision(precision);
    LOG_TRACE(DATA_TYPES,
              "{}: Chosen precision for node '{}' is: {}",
              HLLOG_FUNC,
              node->getNodeName(),
              getStringFromSynDataType(precision));

    return isLayerRaised;
}

synDataType NodesPrecisionSelection::getNodePrecisionFromSuccessors(HabanaGraph& g, const NodePtr& node)
{
    // if the node is a graph output, take the precision from its predecessors
    if (!g.hasConsumer(*node))
    {
        return getNodePrecisionFromPredecessor(g, node);
    }

    std::vector<synDataType> precisions;

    for (const TensorPtr& outputTensor : node->getOutputs())
    {
        precisions.push_back(getTensorDataTypeFromConsumers(g, outputTensor));
    }

    // return the highest precision we got from the successors
    return getHighestGUIDDataType(precisions);
}

synDataType NodesPrecisionSelection::getTensorDataTypeFromConsumers(HabanaGraph& g, const TensorPtr& tensor)
{
    std::vector<synDataType> dtypes;

    for (const NodePtr& consumer : g.getTensorConsumers(tensor))
    {
        synDataType precision;
        synDataType consumerQuantDataType = getQuantDataType(g, tensor, consumer);
        if (consumerQuantDataType != syn_type_na)
        {
            precision = consumerQuantDataType;
        }
        else
        {
            precision = getNodePrecisionFromSuccessors(g, consumer);
        }
        dtypes.push_back(precision);
    }

    return getHighestGUIDDataType(dtypes);
}

synDataType NodesPrecisionSelection::getNodePrecisionFromPredecessor(HabanaGraph& g, const NodePtr& node)
{
    std::vector<synDataType> precisions;
    precisions.push_back(m_minNodePrecision);

    NodeSet producers = g.getNodeProducers(node);
    if (producers.empty())
    {
        return m_profilePrecision;
    }

    for (const NodePtr& producer : producers)
    {
        synDataType producerPrecision    = producer->getNodePrecision();
        synDataType producerMinPrecision = g.getNodeTypeMinPrecision(producer);
        if (producerPrecision != syn_type_na)
        {
            precisions.push_back(producerPrecision);
        }
        else if (producerMinPrecision != syn_type_na)
        {
            precisions.push_back(producerMinPrecision);
        }
        else
        {
            precisions.push_back(getNodePrecisionFromPredecessor(g, producer));
        }
    }

    return getHighestGUIDDataType(precisions);
}