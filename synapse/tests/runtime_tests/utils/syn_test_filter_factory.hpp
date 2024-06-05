#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "../infra/test_types.hpp"

#define TEST_CONCAT_(a, b) a##b
#define TEST_CONCAT(a, b)  TEST_CONCAT_(a, b)
#define CREATE_SUITE_REGISTRATION(suite)                                                                               \
    static bool TEST_CONCAT(registrar_, __LINE__) = SynTestFilterFactory::createSuiteRegistration(#suite);
#define REGISTER_SUITE(suite, ...)                                                                                     \
    static bool TEST_CONCAT(registrar_, __LINE__) = SynTestFilterFactory::registerSuite(#suite, {__VA_ARGS__});
#define REGISTER_SUITE_UNIQUE(suite, ...)                                                                              \
    static bool TEST_CONCAT(registrar_, __LINE__) = SynTestFilterFactory::registerSuiteUnique(#suite, {__VA_ARGS__});

class SynTestFilterFactory
{
public:
    SynTestFilterFactory() = delete;

    static bool registerSuite(const std::string& name, std::initializer_list<synTestPackage> packages)
    {
        s_suiteRegistrationMap[name] = true;
        for (auto pkg : packages)
        {
            s_packageMap[pkg].insert(name);
            s_suiteDefaultPackageMap[name].push_back(pkg);
        }
        return true;
    }

    static bool registerSuiteUnique(const std::string& name, std::initializer_list<synTestPackage> packages)
    {
        s_suiteRegistrationMap[name] = true;
        for (auto pkg : packages)
        {
            s_packageMap[pkg].insert(name);
        }
        return true;
    }

    static void printFactory()
    {
        for (auto package : s_packageMap)
        {
            std::cout << (int)(package.first) << std::endl;
            for (const auto& suite : package.second)
            {
                std::cout << "  " << suite << std::endl;
            }
        }
    }

    static std::string buildFilter(const std::vector<std::string>& includePackages,
                                   const std::vector<std::string>& excludePackages)
    {
        // In case no packages are included by-request, then all packages will be included by-filter
        // Hence, we will want to ensure exclude by-request will be part of the filter definition

        // 1) Define which package is included by-filter (According to given includePackages &
        //    excludePackages paramaters)
        //
        // 2) In case includePackages had been defined (not empty), while suitNames is empty:
        //    a) Each included pkg had been excluded      - nothing should run
        //    b) Nothing had been included at first place - handle as if includePackages had been empty (see 3 below)
        //
        // 3) In case includePackages had NOT been defined:
        //    Define filter by excluding packages according to the excludePackages
        //
        // 4) In case both parameters are empty or no package had been excluded (and none had been included),
        //    filter is empty => run all tests

        std::unordered_set<std::string> suitNames;
        std::string                     prefix = "--gtest_filter=";
        synTestPackage                  pkg    = synTestPackage::SIZE;

        bool defineFilterByExclusion = includePackages.empty();
        bool fallbackRunAllPkgs      = true;

        do
        {
            if (!defineFilterByExclusion)
            {
                for (auto suitSet : includePackages)
                {
                    pkg = getPackageFromName(suitSet);
                    if (pkg == synTestPackage::SIZE)
                    {
                        continue;
                    }
                    suitNames.insert(s_packageMap[pkg].begin(), s_packageMap[pkg].end());
                }

                if (suitNames.empty())
                {
                    // See (2b)
                    defineFilterByExclusion = true;
                    fallbackRunAllPkgs      = false;
                    break;
                }

                if (!excludePackages.empty())
                {
                    for (auto suitSet : excludePackages)
                    {
                        pkg = getPackageFromName(suitSet);
                        if (pkg == synTestPackage::SIZE)
                        {
                            continue;
                        }
                        for (auto itr = s_packageMap[pkg].begin(); itr != s_packageMap[pkg].end(); itr++)
                        {
                            suitNames.erase(*itr);
                        }
                    }
                }

                if (suitNames.empty())
                {
                    // See (2a)
                    prefix             = "--gtest_filter=-:*";
                    fallbackRunAllPkgs = false;
                }
            }
        } while (0);  // Do once

        // See (2b & 3)
        if (defineFilterByExclusion && !excludePackages.empty())
        {
            for (auto suitSet : excludePackages)
            {
                pkg = getPackageFromName(suitSet);
                if (pkg == synTestPackage::SIZE)
                {
                    continue;
                }
                suitNames.insert(s_packageMap[pkg].begin(), s_packageMap[pkg].end());
            }

            if (suitNames.empty())
            {
                prefix = "--gtest_filter=-:*";
            }
            else
            {
                prefix = "--gtest_filter=-";
            }
        }

        // See (4)
        if (fallbackRunAllPkgs && suitNames.empty())
        {
            return "";
        }

        std::stringstream gtestFilter;
        std::copy(suitNames.begin(), suitNames.end(), std::ostream_iterator<std::string>(gtestFilter, ".*:"));
        return prefix + gtestFilter.str();
    }

    static inline synTestPackage getPackageFromName(std::string packageName)
    {
        if (s_nameToSuitMap.find(packageName) == s_nameToSuitMap.end())
        {
            return synTestPackage::SIZE;
        }
        return s_nameToSuitMap[packageName];
    }

    static inline std::vector<synTestPackage>& getSuiteDefaultPackages(std::string suiteName)
    {
        return s_suiteDefaultPackageMap[suiteName];
    }

    static inline bool createSuiteRegistration(std::string suiteName) { return s_suiteRegistrationMap[suiteName]; }

    static inline std::unordered_map<std::string, bool>& getSuiteRegistrationMap() { return s_suiteRegistrationMap; }

private:
    static inline std::unordered_map<synTestPackage, std::unordered_set<std::string>> s_packageMap;
    static inline std::unordered_map<std::string, std::vector<synTestPackage>>        s_suiteDefaultPackageMap;
    static inline std::unordered_map<std::string, bool>                               s_suiteRegistrationMap;
    static inline std::unordered_map<std::string, synTestPackage>                     s_nameToSuitMap = {
        {"DEFAULT", synTestPackage::DEFAULT},
        {"CI", synTestPackage::CI},
        {"ASIC", synTestPackage::ASIC},
        {"ASIC-CI", synTestPackage::ASIC_CI},
        {"SIM", synTestPackage::SIM},
        {"DEATH", synTestPackage::DEATH}};
};