#pragma once
#include "bundle_seed_collector.h"
#include "bundler.h"

namespace gc::layered_brain
{
class MmeBundler : public Bundler
{
public:
    MmeBundler(HabanaGraph& graph, const std::vector<bundler::SeedCollectorPtr>& seedCollectors = {});
    virtual ~MmeBundler() = default;

protected:
    static std::vector<bundler::SeedCollectorPtr>
    getSupportedCollectors(const std::vector<bundler::SeedCollectorPtr>& derivedCollectors, HabanaGraph& graph);
};
}  // namespace gc::layered_brain