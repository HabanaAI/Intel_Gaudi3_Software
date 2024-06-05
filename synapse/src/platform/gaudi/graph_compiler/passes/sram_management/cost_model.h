#pragma once

#include "graph_compiler/passes/sram_management/slice_mapping.h"

namespace gaudi
{

// Interface for the cost model to be used to evaluate the cost of a single operation.
class CostModel
{
public:
    CostModel() = default;
    virtual ~CostModel() = default;

    // Cost of an operation or aggregated cost of a series of operations.
    struct Cost
    {
        enum class Engine
        {
            MME,
            TPC,
            DMA,

            AGGREGATION, // Cost from several engines
        };

        // Default constructor
        Cost() : engine(Engine::AGGREGATION) {}

        // Create a zero cost for the specified engine
        explicit Cost(Engine engine_) : engine(engine_) {}

        Engine   engine;
        uint64_t timeNano        = 0ull;
        uint64_t hbmTrafficBytes = 0ull;

        Cost& operator+=(const Cost& other)
        {
            timeNano += other.timeNano;
            hbmTrafficBytes += other.hbmTrafficBytes;
            return *this;
        }

        Cost& operator=(const Cost& other) = default;

        std::string toString() const
        {
            return fmt::format("Time: {} nano, HBM Traffic: {} bytes", timeNano, hbmTrafficBytes);
        }
    };

    virtual Cost calcCost(const pNode& node,
                          const SliceReferenceList& inputs,
                          const SliceReferenceList& outputs) const = 0;
};

} // namespace gaudi