#pragma once

#include <vector>

struct DimSizes
{
    DimSizes() : DimSizes(0) {}
    DimSizes(unsigned size) : DimSizes(size, size, size) {}
    DimSizes(unsigned min, unsigned max, unsigned actual) : min(min), max(max), actual(actual) {}
    unsigned min;
    unsigned max;
    unsigned actual;
};

struct ShapeSizes
{
    std::vector<unsigned> min;
    std::vector<unsigned> max;
    std::vector<unsigned> actual;
};
