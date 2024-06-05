#pragma once

#include <string>

void realToFixedPoint(double realVal, int32_t& scale, int32_t& exponent);
void validateFilePath(const std::string& filePath);
