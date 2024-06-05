#pragma once

#include "hal_reader/hal_reader.h"
#include "passes/sram_management/cost_model.h"
#include "settable.h"
#include "strategy_slicing_data.h"
#include "strategy_visitor.h"

class SlicingStrategy;
using SlicingStrategyPtr = std::shared_ptr<SlicingStrategy>;
using SlicingStrategyList = std::list<SlicingStrategyPtr>;

class SlicingStrategy
{
public:
    // metrics struct - holds all the relevant calculated parameters.
    struct Metrics
    {
        uint64_t SRAMCapacity = 0;
        float MMEUtilization = 0;
        double HBMBandwidth = 0;
        unsigned SBReuse = 0;
        bool isDoubleBuffered = true;
        bool valid = false;
    };

    static SlicingStrategyPtr createStrategy(const HalReader& halReader, const pNode& node);

    SlicingStrategy(const HalReader& halReader, const StrategySlicingDataPtr& m_slicingData);
    SlicingStrategy(const SlicingStrategy& rhs, bool resetAlignment);

    virtual ~SlicingStrategy() = default;

    virtual SlicingStrategyPtr clone(bool resetAlignment);

    virtual void accept(StrategyVisitor& visitor) {visitor.visit(*this);}

    virtual Metrics& getMetrics();
    virtual const Metrics& getMetrics() const;
    virtual Metrics& calculateMetrics();
    virtual Metrics& recalculateSramCapacity();

    virtual void printLog(int logLevel, const synapse::LogManager::LogType& logname) const;
    virtual void printLog() const {printLog(0, synapse::LogManager::LogType::SRAM_SLICE);}

    virtual StrategySlicingData& getSlicingData();
    virtual const StrategySlicingData& getSlicingData() const;

    void setDoubleBuffer(bool val);
    bool sramSlicedOperandsDoubleBuffered() const;
    void setAllowUpdateNumOfBuffers(bool state) { m_allowUpdateNumOfBuffers = state; }
    bool allowUpdateNumOfBuffers() const { return m_allowUpdateNumOfBuffers; }

    // If the double buffer flag is set, any sliced operand residing in SRAM needs to have 2 buffers.
    // If no operand is sliced or in SRAM, the double buffer flag should be set to false.
    // This method align these 2 requirements.
    void alignNumBuffers();

    void resetAlignment();

    // In case the strategy has trivially sliced master operand, we want it to be considered LTR, both for metrics
    // and for partially sliced shared MME input stitching.
    void alignWalkingPattern();

    // try to align sram inputs to cache line if possible given SRAM capacity (mainly
    // for improving MME performance, implement in MME strategy subclass)
    virtual void tryAlignToCacheLine() {}

    // align the shape tensors slicing to the relevant output tensor slicing
    void alignShapeTensorsSlicing(const pBundle& bundle);

    bool compareInitialStrategy(const SlicingStrategy& other, bool exactMatch = true) const;

    SlicingStrategy& setInputIsInSRAM(unsigned idx, bool val);

    SlicingStrategy& setOutputIsInSRAM(bool val);

    virtual std::string getSlicingDataString(bool exactMatch = true) const;

    void setGraphSizeOptimized(bool graphSizeOptimized) {  m_graphSizeOptimized = graphSizeOptimized;}
    bool getGraphSizeOptimized() const { return m_graphSizeOptimized;}

    void                             setCost(const gaudi::CostModel::Cost& cost) { m_cost = cost; }
    Settable<gaudi::CostModel::Cost> getCost() { return m_cost; }

    struct Hasher
    {
        Hasher() = default;
        Hasher(bool exactMatch) : m_exactMatch(exactMatch) {}
        size_t operator()(const SlicingStrategyPtr& s) const
        {
            return std::hash<std::string>()(s->getSlicingDataString(m_exactMatch));
        }
        const bool m_exactMatch = true;
    };

    struct IsEqual
    {
        IsEqual() = default;
        IsEqual(bool exactMatch) : m_exactMatch(exactMatch) {}
        bool operator()(const SlicingStrategyPtr& obj1, const SlicingStrategyPtr& obj2) const
        {
            return (obj1->compareInitialStrategy(*obj2, m_exactMatch));
        }
        const bool m_exactMatch = true;
    };

protected:
    const HalReader&                 m_halReader;
    StrategySlicingDataPtr           m_slicingData;
    Metrics                          m_metrics;
    Settable<gaudi::CostModel::Cost> m_cost;
    bool                             m_graphSizeOptimized;
    bool                             m_allowUpdateNumOfBuffers;
};
