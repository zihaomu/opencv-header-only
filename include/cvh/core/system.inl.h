#ifndef CVH_SYSTEM_INL_H
#define CVH_SYSTEM_INL_H

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

namespace cvh
{

namespace detail
{

inline const char* errorStr(int status)
{
    static char buf[256];

    switch (status)
    {
        case Error::StsOk:             return "No Error";
        case Error::StsBackTrace:      return "Backtrace";
        case Error::StsError:          return "Unspecified error";
        case Error::StsInternal:       return "Internal error";
        case Error::StsNoMem:          return "Insufficient memory";
        case Error::StsBadArg:         return "Bad argument";
        case Error::StsNullPtr:        return "Null pointer";
        case Error::StsBadSize:        return "Incorrect size of input array";
        case Error::StsNotImplemented: return "The function/feature is not implemented";
        case Error::StsAssert:         return "Assertion failed";
        default:
            break;
    }

    std::snprintf(buf, sizeof(buf), "Unknown %s code %d", status >= 0 ? "status" : "error", status);
    return buf;
}

inline int m_vsnprintf(char* buf, int len, const char* fmt, va_list args)
{
#if defined(_MSC_VER)
    if (len <= 0)
    {
        return len == 0 ? 1024 : -1;
    }
    int res = _vsnprintf_s(buf, len, _TRUNCATE, fmt, args);
    if (res >= 0 && res < len)
    {
        buf[res] = 0;
        return res;
    }
    buf[len - 1] = 0;
    return res >= len ? res : (len * 2);
#else
    return std::vsnprintf(buf, len, fmt, args);
#endif
}

} // namespace detail

inline int m_snprintf(char* buf, int len, const char* fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    int res = detail::m_vsnprintf(buf, len, fmt, va);
    va_end(va);
    return res;
}

inline void cvprintf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
}

inline std::string format(const char* fmt, ...)
{
    std::vector<char> buf(1024);

    for (;;)
    {
        va_list va;
        va_start(va, fmt);
        const int bsize = static_cast<int>(buf.size());
        const int len = detail::m_vsnprintf(buf.data(), bsize, fmt, va);
        va_end(va);

        CV_Assert(len >= 0 && "Check format string for errors");
        if (len >= bsize)
        {
            buf.resize(static_cast<size_t>(len) + 1);
            continue;
        }
        buf[static_cast<size_t>(bsize) - 1] = 0;
        return std::string(buf.data(), static_cast<size_t>(len));
    }
}

inline Exception::Exception()
    : code(0), line(0)
{
}

inline Exception::Exception(int _code, const std::string& _err, const std::string _func,
                            const std::string& _file, int _line)
    : code(_code), err(_err), func(_func), file(_file), line(_line)
{
    formatMessage();
}

inline Exception::~Exception() noexcept = default;

inline const char* Exception::what() const noexcept
{
    return msg.c_str();
}

inline void Exception::formatMessage()
{
    size_t pos = err.find('\n');
    const bool multiline = pos != std::string::npos;

    if (multiline)
    {
        std::stringstream ss;
        size_t prev_pos = 0;

        while (pos != std::string::npos)
        {
            ss << "> " << err.substr(prev_pos, pos - prev_pos) << std::endl;
            prev_pos = pos + 1;
            pos = err.find('\n', prev_pos);
        }

        ss << "> " << err.substr(prev_pos);
        if (!err.empty() && err.back() != '\n')
        {
            ss << std::endl;
        }
        err = ss.str();
    }

    if (!func.empty())
    {
        if (multiline)
        {
            msg = format("cvh(%s) %s:%d: error (%d:%s) in function '%s' \n %s",
                         CV_VERSION, file.c_str(), line, code, detail::errorStr(code), func.c_str(), err.c_str());
        }
        else
        {
            msg = format("cvh(%s) %s:%d: error (%d:%s) %s in function '%s' \n",
                         CV_VERSION, file.c_str(), line, code, detail::errorStr(code), err.c_str(), func.c_str());
        }
    }
    else
    {
        msg = format("cvh(%s) %s:%d: error (%d:%s) %s%s",
                     CV_VERSION, file.c_str(), line, code, detail::errorStr(code), err.c_str(), multiline ? "" : "\n");
    }
}

inline void error(const Exception& exc)
{
    throw exc;
}

inline void error(int code, const std::string& err, const std::string func,
                  const std::string& file, int line)
{
    error(Exception(code, err, func, file, line));
}

inline void error(const std::string& err, const std::string func,
                  const std::string& file, int line)
{
    error(Exception(Error::StsError, err, func, file, line));
}

inline void warning(int code, const std::string& msg,
                    const std::string& func,
                    const std::string& file,
                    int line)
{
    std::cerr << "[WARNING][" << file << ":" << line
              << "][" << func << "] Code=" << code
              << ": " << msg << std::endl;
}

inline void warning(const std::string& msg,
                    const std::string& func,
                    const std::string& file,
                    int line)
{
    warning(0, msg, func, file, line);
}

inline void debug(int code,
                  const std::string& msg,
                  const std::string& func,
                  const std::string& file,
                  int line)
{
#if CV_DEBUG
    std::cout << "[DEBUG][" << file << ":" << line
              << "][" << func << "] Code=" << code
              << ": " << msg << std::endl;
#else
    (void)code;
    (void)msg;
    (void)func;
    (void)file;
    (void)line;
#endif
}

inline void debug(const std::string& msg,
                  const std::string& func,
                  const std::string& file,
                  int line)
{
    debug(0, msg, func, file, line);
}

} // namespace cvh

#endif // CVH_SYSTEM_INL_H
