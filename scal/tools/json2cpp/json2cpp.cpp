#include <iostream>
#include <string>
#include <vector>
#include "../../src/common/json.hpp"
#include <fstream>
#include <array>

std::string filenameToVarName(std::string const & filename)
{
    std::string res;
    std::transform(filename.begin(), filename.end(), std::back_inserter(res), [](char c){
        return (c == '/' || c == '.' || c == '\\' || c == '-')  ? '_' : c;
    });
    return res;
}
const unsigned line_length = 20;

const unsigned xor_length = 40;
const std::array<uint8_t, xor_length> xor_values = [](){
    std::srand(std::time(nullptr));
    std::array<uint8_t, xor_length> values{};
    for (auto & v : values)
        v = unsigned(rand()) % 255;
    return values;
}();

void addContentToCpp(std::string const & var_name, std::vector<uint8_t> const & cbor, std::ostream & ofs)
{
    ofs << "const unsigned char " << var_name << "[] ={\n";
    unsigned line_pos = 0;
    unsigned xor_pos = 0;
    for (uint8_t c : cbor)
    {
        if (line_pos == 0) ofs << "   ";
        ofs << std::hex << " 0x" << std::setw(2) << std::setfill('0') << unsigned(c ^ xor_values[xor_pos]) << ",";
        if (++line_pos >= line_length)
        {
            line_pos = 0;
            ofs << "\n";
        }
        if (++xor_pos >= xor_length)
        {
            xor_pos = 0;
        }
    }
    ofs << "};\n\n";
}
int main(int argc, char * argv [])
try{
    if (argc < 3)
    {
        std::cout << "json2cpp output_cpp_filename input_folder input_json_files ...\n";
        std::cout << "   convert json files into binary format, encrypt and embed them into cpp file\n";
        std::cout << "   output_cpp_filename : .cpp file with embedded jsonfiles. .h file is created in thesame folder\n";
        std::cout << "   input_folder : todo\n";
        std::cout << "   input_json_files : json files for conversion\n";
        return 0;
    }
    std::string cpp_filename = argv[1];
    std::string input_folder = argv[2];
    if (!input_folder.empty())
    {
        if (input_folder.back() != '/')
        {
            input_folder += '/';
        }
    }
    std::ofstream ocppfs(cpp_filename);
    if (ocppfs.bad() || ocppfs.fail())
    {
        throw std::runtime_error("failed to open file " + cpp_filename);
    }
    ocppfs.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    std::vector<std::pair<std::string, std::string>> varname2path;
    for (int i = 3; i <argc; ++i)
    {
        std::string filename = argv[i];
        std::ifstream ifs(filename);
        if (ifs.bad() || ifs.fail())
        {
            throw std::runtime_error("failed to open file " + filename);
        }

        if (filename.find(input_folder) == 0)
        {
            filename = filename.substr(input_folder.size());
        }
        try {
            auto json = scaljson::json::parse(ifs, nullptr, true, true);
            std::vector<uint8_t> cbor = scaljson::json::to_cbor(json);
            std::string var_name = filenameToVarName(filename);
            addContentToCpp(var_name, cbor, ocppfs);
            varname2path.push_back({var_name, filename});
        }
        catch(std::exception & e)
        {
            throw std::runtime_error(std::string("failed to parse ") + argv[i] + " error: " + e.what());
        }
    }

    ocppfs << "#include <unordered_map>\n#include <vector>\n#include <string>\n";
    ocppfs << "static const std::unordered_map<std::string, std::string_view> internalFiles = {\n";

    for (auto const & varname_path: varname2path)
    {
        ocppfs << "    { \"" << varname_path.second << "\", std::string_view((const char *)" << varname_path.first << ", sizeof(" << varname_path.first << "))},\n";
    }
    ocppfs << R"(};
std::string getInternalFile(std::string const & filename)
{
    const unsigned char xor_values[] = {)";
    for (auto v : xor_values)
    {
        ocppfs << std::hex << " 0x" << std::setw(2) << std::setfill('0') << unsigned(v) << ",";
    }
    ocppfs << "};\n";
    ocppfs << R"(
    const auto it = internalFiles.find(filename);
    if (it == internalFiles.end())
    {
        return "";
    }
    std::string val;
    val.reserve(it->second.size());
    unsigned xor_pos = 0;
    for (auto c : it->second)
    {
        val += char((unsigned char)(c) ^ xor_values[xor_pos++]);
        if (xor_pos >= sizeof(xor_values)) xor_pos = 0;
    }
    return val;
}
std::vector<std::string> getInternalFilenames()
{
    std::vector<std::string> res;
    for (auto const & item : internalFiles)
    {
        res.push_back(item.first);
    }
    return res;
}
)";
    unsigned dot_pos = cpp_filename.rfind('.');
    std::string h_filename = cpp_filename.substr(0, dot_pos) + ".h";
    std::ofstream ohfs(h_filename);
    if (ohfs.bad() || ohfs.fail())
    {
        throw std::runtime_error("failed to open file " + h_filename);
    }
    ohfs << R"(#pragma once
#include <string>
#include <vector>
std::string getInternalFile(std::string const & filename);
std::vector<std::string> getInternalFilenames();
)";
    return 0;
}
catch(std::exception & e)
{
    std::cerr << "error: " << e.what() << "\n";
    exit(1);
}
