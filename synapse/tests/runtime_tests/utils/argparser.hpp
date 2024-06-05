#pragma once
#include <algorithm>
#include <vector>
#include <string>

class InputParser
{
public:
    InputParser(int& argc, char** argv)
    {
        for (int i = 1; i < argc; ++i)
            m_tokens.push_back(argv[i]);
    }

    // this handles --option VALUE
    const std::vector<std::string> getCmdOption(const std::string& option) const
    {
        std::vector<std::string> argvals;
        auto                     itr = std::find(m_tokens.begin(), m_tokens.end(), option);

        while (itr != m_tokens.end() && itr + 1 != m_tokens.end() && (*(itr + 1)).rfind("-", 0) != 0)
        {
            argvals.push_back(*(++itr));  // return the VALUE part
        }

        return argvals;
    }

    // this handles --option=VALUE
    const std::vector<std::string> getCmdOptionEQ(const std::string& option) const
    {
        std::vector<std::string> argvals;
        auto                     itr = m_tokens.begin();
        while (itr != m_tokens.end())
        {
            std::string token    = *itr;
            size_t      foundPos = token.find("=");
            if (foundPos != -1)
            {
                std::string lpart = token.substr(0, foundPos);
                if (lpart == option)
                {
                    std::string rpart = token.substr(foundPos + 1);
                    argvals.push_back(rpart);  // return the VALUE part
                }
            }
            ++itr;
        }
        return argvals;
    }

    bool cmdOptionExists(const std::string& option) const
    {
        return std::find(m_tokens.begin(), m_tokens.end(), option) != m_tokens.end();
    }

private:
    std::vector<std::string> m_tokens;
};