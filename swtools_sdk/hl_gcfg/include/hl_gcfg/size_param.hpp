#pragma once

#include <string>
#include <algorithm>
#include "hlgcfg_defs.hpp"

namespace hl_gcfg{
HLGCFG_INLINE_NAMESPACE{
/*
 ***************************************************************************************************
 *   class SizeParam
 *
 *   @brief Class that holds a string representation and size in bytes
 *          * read the size with optional units
 *          -- default is bytes
 *          -- 1K == 1024 (not 1000)
 *          -- example valid inputs: 64K, 512kb, 32mb, 16g, 40GB
 ***************************************************************************************************
 */
class SizeParam
{
public:
    SizeParam()
    : m_byteVal(0)
    , m_inputStr("0B")
    {}
    SizeParam(uint64_t byteVal)
    : m_byteVal(byteVal)
    , m_inputStr(std::to_string(byteVal) + "B")
    {}
    SizeParam(std::string const & str)
    : m_inputStr(str)
    {
        constexpr unsigned int KB_SIZE = (1024);
        constexpr unsigned int MB_SIZE = (KB_SIZE * KB_SIZE);
        constexpr unsigned int GB_SIZE = (KB_SIZE * KB_SIZE * KB_SIZE);

        std::string::size_type nextCharIdx = 0;

        uint64_t inputVal = std::stoull(str, &nextCharIdx, 0);
        m_byteVal         = inputVal;
        std::string units = str.substr(nextCharIdx);
        if (units.empty())
        {
            // no unit from user - default is Byte
            m_inputStr += "B";
        }
        else
        {
            // transform to uppercase to simplify comparisons
            std::transform(units.begin(), units.end(), units.begin(), ::toupper);
            if (units == "K" || units == "KB")
            {
                m_byteVal *= KB_SIZE;
            }
            else if (units == "M" || units == "MB")
            {
                m_byteVal *= MB_SIZE;
            }
            else if (units == "G" || units == "GB")
            {
                m_byteVal *= GB_SIZE;
            }
            else if (units != "B")
            {
                m_isValid = false;
            }
        }
    }

    // copy ctors
    SizeParam& operator=(const SizeParam&) = default;
    SizeParam(const SizeParam& s)          = default;

    // operators
    operator std::string() const { return toString(); }

    // methods
    bool isValid() const
    {
        return m_isValid;
    }

    std::string toString() const
    {
        return std::to_string(m_byteVal) + " (" + m_inputStr + ")";
    }

    std::string getString() const
    {
        return toString();
    }

    const uint64_t & getByteVal() const
    {
        return m_byteVal;
    }
    bool operator==(SizeParam const & other) const
    {
        return m_byteVal == other.m_byteVal
               && m_isValid && other.m_isValid;
    }
private:
    uint64_t    m_byteVal;
    std::string m_inputStr;
    bool        m_isValid = true;
};

inline std::string toString(SizeParam const & sizeParam)
{
    return sizeParam.toString();
}
}}
