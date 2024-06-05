#include "utils.h"

#include "defs.h"
#include "filesystem.h"
#include "spdlog/fmt/bundled/core.h"

#include <cmath>

void validateFilePath(const std::string& filePath)
{
    fs::path path = filePath;

    const std::string fileName = path.filename();

    if (fileName.length() >= NAME_MAX || path.native().length() >= PATH_MAX)
    {
        throw std::runtime_error(fmt::format(
            "the file path doesn't match the file system limits, file path length: max: {}, actual: {}, file name "
            "length: max: {}, actual: {}",
            PATH_MAX,
            path.native().length(),
            NAME_MAX,
            fileName.length()));
    }
}

void realToFixedPoint(double realVal, int32_t& scale, int32_t& exponent)
{
    if (realVal == 0)
    {
        scale    = 0;
        exponent = 0;
        return;
    }
    double log2ofRealVal = log(fabs(realVal)) / log(2.0);

    double realShift = ceil(log2ofRealVal);

    double realScale      = realVal / pow(2, realShift);
    exponent              = (int32_t)realShift;
    double scaleDiffFrom1 = fabs(realScale - 1.0);
    if (scaleDiffFrom1 < std::numeric_limits<double>::epsilon() * (1 << 22))
    {
        scale = 0;
    }
    else
    {
        double dblScale = round(realScale * pow(2, 31));
        dblScale        = dblScale < INT32_MAX ? dblScale : INT32_MAX;
        dblScale        = dblScale > INT32_MIN ? dblScale : INT32_MAX;
        scale           = (int32_t)dblScale;
        HB_ASSERT(scale != 0, "zero scale is not supported with gemmlowp");
    }
}