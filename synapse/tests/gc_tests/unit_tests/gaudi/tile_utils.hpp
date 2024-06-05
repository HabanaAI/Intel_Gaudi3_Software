#pragma once

#include "perf_lib_layer_params.h"
#include "gtest/gtest.h"

template<class T>
class TileTestUtilsBase : public testing::WithParamInterface<::testing::tuple<T, T>>
{
public:
    struct GetName
    {
        template<class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            ::std::stringstream ss;
            ss << "sizein_" << toString(::testing::get<0>(info.param), 'x') << "_"
               << "sizeout_" << toString(::testing::get<1>(info.param), 'x');
            return ss.str();
        }
    };

    void calculateTileParams(const T& in, const T& out, ns_TileKernel::ParamsV2& params)
    {
        auto& repeat = params.repeat;
        for (int i = 0; i < out.size(); i++)
        {
            if ((in.size() <= i) || (in[i] <= 1))
            {
                repeat[i] = out[i];
            }
            else
            {
                repeat[i] = out[i] / in[i];
                ASSERT_TRUE(out[i] == in[i] * repeat[i]);
            }
        }
    }
};