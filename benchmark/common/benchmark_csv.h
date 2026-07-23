#ifndef CVH_BENCHMARK_COMMON_BENCHMARK_CSV_H
#define CVH_BENCHMARK_COMMON_BENCHMARK_CSV_H

#include <initializer_list>
#include <ostream>
#include <string>

namespace cvh_bench {
namespace common {

inline bool csv_needs_quotes(const std::string& value)
{
    return value.find_first_of(",\"\n\r") != std::string::npos;
}

inline std::string csv_escape(const std::string& value)
{
    if (!csv_needs_quotes(value))
    {
        return value;
    }

    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char ch : value)
    {
        if (ch == '"')
        {
            out.push_back('"');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

inline void write_csv_row(std::ostream& os, std::initializer_list<std::string> cells)
{
    bool first = true;
    for (const auto& cell : cells)
    {
        if (!first)
        {
            os << ",";
        }
        first = false;
        os << csv_escape(cell);
    }
    os << "\n";
}

}  // namespace common
}  // namespace cvh_bench

#endif  // CVH_BENCHMARK_COMMON_BENCHMARK_CSV_H
