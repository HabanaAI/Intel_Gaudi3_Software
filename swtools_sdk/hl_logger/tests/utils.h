#pragma once
#include <string>
#include <vector>
/*
 * test crash handlers
 * can be tested only manually and one test at a time
 * each test crashes and in the log file it should add the log message from the test
 * */
int recursion(int i);
int findStringsCountInLog(const std::string& strToFind, const std::string& filename);
std::vector<std::string> findStringsInLog(const std::string& strToFind, const std::string& filename, const std::string& logDir);
std::vector<std::string> findStringsInLog(const std::string& strToFind, const std::string& filename);
std::string filenameFromPath(const std::string& path);