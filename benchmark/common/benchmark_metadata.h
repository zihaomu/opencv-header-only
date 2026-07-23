#ifndef CVH_BENCHMARK_COMMON_BENCHMARK_METADATA_H
#define CVH_BENCHMARK_COMMON_BENCHMARK_METADATA_H

#include <ostream>
#include <string>

namespace cvh_bench {
namespace common {

struct BenchmarkMetadata
{
    std::string mode;
    std::string suite;
    std::string profile;
    std::string implementation;
    std::string output_csv;
    std::string cvh_commit = "unknown";
    std::string opencv_commit;
    std::string opencv_source;
    std::string opencv_build_dir;
    std::string cmake_build_type;
    std::string runtime = "single_thread";
    std::string cpu_model = "unknown";
    int warmup = 0;
    int iters = 0;
    int repeats = 0;
    int threads = 1;
};

inline std::string compiler_id()
{
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

inline std::string compiler_version()
{
#if defined(__clang__)
    return __clang_version__;
#elif defined(__GNUC__)
    return __VERSION__;
#elif defined(_MSC_VER)
    return std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

inline std::string target_arch()
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

inline std::string target_os()
{
#if defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#else
    return "unknown";
#endif
}

inline std::string build_config()
{
#if defined(NDEBUG)
    return "release";
#else
    return "debug";
#endif
}

inline void write_json_string(std::ostream& os, const std::string& value)
{
    os << '"';
    for (const char ch : value)
    {
        switch (ch)
        {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << ch; break;
        }
    }
    os << '"';
}

inline void write_metadata_json(std::ostream& os, const BenchmarkMetadata& meta)
{
    os << "{\n";
    os << "  \"mode\": ";
    write_json_string(os, meta.mode);
    os << ",\n  \"suite\": ";
    write_json_string(os, meta.suite);
    os << ",\n  \"profile\": ";
    write_json_string(os, meta.profile);
    os << ",\n  \"implementation\": ";
    write_json_string(os, meta.implementation);
    os << ",\n  \"output_csv\": ";
    write_json_string(os, meta.output_csv);
    os << ",\n  \"cvh_commit\": ";
    write_json_string(os, meta.cvh_commit);
    os << ",\n  \"opencv_commit\": ";
    write_json_string(os, meta.opencv_commit);
    os << ",\n  \"opencv_source\": ";
    write_json_string(os, meta.opencv_source);
    os << ",\n  \"opencv_build_dir\": ";
    write_json_string(os, meta.opencv_build_dir);
    os << ",\n  \"cmake_build_type\": ";
    write_json_string(os, meta.cmake_build_type);
    os << ",\n  \"runtime\": ";
    write_json_string(os, meta.runtime);
    os << ",\n  \"cpu_model\": ";
    write_json_string(os, meta.cpu_model);
    os << ",\n  \"warmup\": " << meta.warmup;
    os << ",\n  \"iters\": " << meta.iters;
    os << ",\n  \"repeats\": " << meta.repeats;
    os << ",\n  \"threads\": " << meta.threads;
    os << ",\n  \"compiler_id\": ";
    write_json_string(os, compiler_id());
    os << ",\n  \"compiler_version\": ";
    write_json_string(os, compiler_version());
    os << ",\n  \"target_arch\": ";
    write_json_string(os, target_arch());
    os << ",\n  \"target_os\": ";
    write_json_string(os, target_os());
    os << ",\n  \"build_config\": ";
    write_json_string(os, build_config());
    os << "\n}\n";
}

}  // namespace common
}  // namespace cvh_bench

#endif  // CVH_BENCHMARK_COMMON_BENCHMARK_METADATA_H
