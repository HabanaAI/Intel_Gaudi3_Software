#pragma once

#include "tensor.h"
#include "graph_optimizer_test.h"
#include "graph_factory.h"
#include "test_utils.h"

class GenericGraphTest
: public GraphOptimizerTest
, public testing::WithParamInterface<synDeviceType>
{
protected:
    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        synDeviceType deviceType = GetParam();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
    }

    void TearDown() override
    {
        m_graph.reset();
        GraphOptimizerTest::TearDown();
    }

    std::unique_ptr<HabanaGraph> m_graph;

public:
    struct GetName
    {
        std::string operator()(const ::testing::TestParamInfo<synDeviceType>& deviceType) const
        {
            ::std::stringstream ss;
            ss << "_" << deviceTypeToString(deviceType.param);
            return ss.str();
        }
    };
};

template<typename... Ts>
class GenericTupleGraphTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<Ts..., synDeviceType>>
{
public:
    using ParamType = typename GenericTupleGraphTest::ParamType;

protected:
    synDeviceType getDeviceType()
    {
        return std::get<std::tuple_size<ParamType>::value - 1>(GenericTupleGraphTest::GetParam());
    }

    void SetUp() override
    {
        GraphOptimizerTest::SetUp();
        synDeviceType deviceType = getDeviceType();
        m_graph                  = GraphFactory::createGraph(deviceType, CompilationMode::Graph);
    }

    void TearDown() override
    {
        m_graph.reset();
        GraphOptimizerTest::TearDown();
    }

    std::unique_ptr<HabanaGraph> m_graph;

public:
    template<size_t I = 0, typename... Tp>
    static void createName(const std::tuple<Tp...>& t, ::std::stringstream& ss)
    {
        ss << "_" << std::get<I>(t);
        if constexpr (I + 1 != sizeof...(Tp) - 1)
        {
            createName<I + 1>(t, ss);
        }
        // Last one.. device type location..
        else if constexpr (I + 1 == sizeof...(Tp) - 1)
        {
            ss << "_" << deviceTypeToString(std::get<I + 1>(t));
        }
    }
    struct GetName
    {
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            ::std::stringstream ss;
            createName(info.param, ss);
            return ss.str();
        }
    };
};

inline const std::string
getGUIDByDevice(synDeviceType deviceType, const std::string& guid, const std::string& precision = "f32")
{
    return guid + (deviceType == synDeviceGreco ? "" : "_fwd") + "_" + precision;
}