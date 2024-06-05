#pragma once

#include "bundle_scheduler_interfaces.h"

namespace gc::layered_brain
{
class SimpleThreadsCreator : public ThreadsCreator
{
public:
    SimpleThreadsCreator() {}
    ThreadsSequence getThreadsSequeceBySliceSets(const std::vector<TensorVector>& sliceSets,
                                                 const RoutesTable&               routesPerSlice) override;

protected:
    NodesThread             getThreadPerSliceSet(const TensorVector& sliceSet, const RoutesTable& routesPerSlice);
    std::vector<NodesRoute> sortSubRoutes(const TensorVector& sliceSet, const RoutesTable& routesPerSlice);
};

}  // namespace gc::layered_brain
