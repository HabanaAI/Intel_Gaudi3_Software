#include "habana_pass.h"

#include "habana_graph.h"
#include "node_factory.h"
#include "types_exception.h"

#include <utility>

Pass::Pass(std::string_view name, PassId id, PassPriority priority) : Pass(name, id, priority, {}, {}) {}

Pass::Pass(std::string_view name,
           PassId           id,
           PassPriority     priority,
           PredicateIDSet   predicateSet,
           PassIDSet        dependencySet)
: m_name(name),
  m_id(id),
  m_predicateSet(std::move(predicateSet)),
  m_dependencySet(std::move(dependencySet)),
  m_priority(priority)
{}

Pass::~Pass()
{}

const std::string& Pass::getName() const
{
    return m_name;
}

PassId Pass::getId() const
{
    return m_id;
}

const PassIDSet& Pass::getDependencySet() const
{
    return m_dependencySet;
}

const PredicateIDSet& Pass::getPredicateSet() const
{
    return m_predicateSet;
}

Pass* Pass::addPredicate(PredicateId predId)
{

    if (!canRunMultipleTimes())
    {
        LOG_ERR(GC, "Pass {}: Can't add predicate - pass may not be executed more than once", getName());
        throw PassFailedException();
    }

    m_predicateSet.insert(predId);
    return this;
}
