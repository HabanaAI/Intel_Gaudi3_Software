#pragma once

#include "base_test.h"
#include "playback_tests.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>


namespace json_tests
{
class DbParser : public BaseTest
{
public:
    DbParser(const ArgParser& args);
    virtual ~DbParser() = default;

    void run() override;

private:
    std::string m_dataFilePath;
    std::string m_graphName;
    std::string m_tensorName;
    std::string m_outputFileName;
    uint64_t    m_iteration;
    uint64_t    m_elementLimit;
    uint64_t    m_group;
    bool        m_binary;
    bool        m_splitFiles;
    bool        m_findNans;
    bool        m_findInfs;

    std::ofstream m_outputStream;

    std::ostream& getStream(const std::string& tensorName);

    struct IdentityCaster
    {
        template<typename T> T operator()(T t) { return t; }
    };

    struct CharCaster
    {
        template<typename T> int operator()(T t) { return t; }
    };

    struct BfloatCaster
    {
        float operator()(uint16_t t)
        {
            uint32_t r = t << 16;
            float ret;
            memcpy(&ret, &r, sizeof(r));
            return ret;
        }
    };

    template <typename T, typename Caster> void findInfsNans (const T* data, std::size_t size, Caster caster)
    {
        std::size_t toCheck = std::min(size, m_elementLimit);
        bool foundNan = false;
        bool foundInf = false;
        for (std::size_t i = 0; i < toCheck; ++i)
        {
            if (m_findNans && caster(std::isnan(data[i])))
            {
                foundNan = true;
            }
            if (m_findInfs && caster(std::isinf(data[i])))
            {
                foundInf = true;
            }

            if (foundNan)
            {
                JT_LOG_ERR("Found NaN at position " << i);
                return;
            }
            if (foundInf)
            {
                JT_LOG_ERR("Found Inf at position " << i);
                return;
            }
        }
    }

    template <typename T, typename Caster> void printElements(const std::string& tensorName, CapturedDataProvider& dataProvider,
              Caster caster)
    {

        if (m_findNans || m_findInfs)
        {
            if constexpr (std::is_same_v<T, float> ||
                          std::is_same_v<T, double> ||
                          std::is_same_v<Caster, BfloatCaster>)
            {
                auto buffer = dataProvider.getBuffer(tensorName);
                findInfsNans(reinterpret_cast<T*>(buffer.data()), buffer.size() / sizeof(T), caster);
            }
            return;
        }

        auto buffer = dataProvider.getBuffer(tensorName);
        if (m_binary)
        {
            std::string filename = sanitizeFileName(tensorName);
            std::ofstream of(filename, std::ios::binary | std::ios::trunc | std::ios::out);
            if (!of)
            {
                JT_LOG_ERR(fmt::format("Could not open file {}", filename));
                return;
            }

            std::size_t toWrite = std::min(buffer.size(), m_elementLimit * sizeof(T));

            of.write(reinterpret_cast<const char*>(buffer.data()), toWrite);
            if (!of)
            {
                JT_LOG_ERR(fmt::format("Error writing file {}", filename));
                return;
            }
        }
        else
        {
            T* elements = reinterpret_cast<T*>(buffer.data());
            auto nElements = buffer.size() / sizeof(T);
            std::ostream& os = getStream(tensorName);
            if (!m_splitFiles)
            {
                os << "Tensor " << tensorName << "\n";
            }

            auto lim = std::min(nElements, m_elementLimit);

            for (std::size_t i = 0; i < lim; ++i)
            {
                os << caster(elements[i]) << "\n";
            }
        }

    }

    template <typename T> void printElements(const std::string& tensorName, CapturedDataProvider& dataProvider)
    {
        printElements<T>(tensorName, dataProvider, IdentityCaster());
    }

    void runGroups();
    void runGraphs(const data_serialize::DataDeserializerPtr& dataDeserializer, uint64_t group);
    void runIterations(const data_serialize::GraphInfo& graphInfo);
    void runTensors(const data_serialize::GraphInfo& graphInfo, uint64_t iteration);

    std::ostream* m_realStream = nullptr;
};
}  // namespace json_tests
