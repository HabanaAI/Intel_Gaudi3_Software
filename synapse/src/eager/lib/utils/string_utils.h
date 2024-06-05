#pragma once

// std includes
#include <string_view>
#include <string>

namespace eager_mode
{

// TODO[c++20]: use std::string_view::starts_with
constexpr bool startsWith(std::string_view str, std::string_view prefix)
{
    return str.substr(0, prefix.size()) == prefix;
}

// TODO[c++20]: use std::string_view::ends_with
constexpr bool endsWith(std::string_view str, std::string_view suffix)
{
    auto s = str.size();
    auto p = suffix.size();
    return s >= p && str.substr(s - p, p) == suffix;
}

// Split a string into a container of string_view.
//
// Example use:
//  auto res = splitStringIntoContainer<std::vector<std::string_view>>(":ABC:::DE:", ':');
//  auto ref = std::vector<std::string_view>{"", "ABC", "", "", "DE", ""};
//  assert(res == ref);
template<typename Container>
constexpr Container splitStringIntoContainer(std::string_view text, char delim)
{
    Container res;
    if (!text.empty())
    {
        while (true)
        {
            auto idx = text.find(delim);
            res.push_back(text.substr(0, idx));
            if (idx == std::string_view::npos) break;
            text = text.substr(idx + 1);
        }
    }
    return res;
}

/** * Sanitize filename.
 *
 * https://en.wikipedia.org/wiki/Filename#Comparison_of_file_name_limitations
 * POSIX "Fully portable filenames" allows [0-9A-Za-z._-] in filenames so that
 * every other char is replaced by this function with '_'. Leading '-' is also
 * replaced. But the max 14 char length is ignored by this function.
 *
 * @param[in] str - Input string to sanitize
 * @return Sanitized string
 */
inline std::string sanitizeFileName(std::string_view str)
{
    std::string fileName {str};
    if (!fileName.empty() && fileName[0] == '-')
    {
        fileName[0] = '_';
    }

    for (char& c : fileName)
    {
        c = ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == '.') ? c
                                                                                                                 : '_';
    }

    return fileName;
}

}  // namespace eager_mode