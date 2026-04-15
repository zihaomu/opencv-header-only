#ifndef CVH_PARALLEL_RUNTIME_H
#define CVH_PARALLEL_RUNTIME_H

#include "../mat.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvh {

enum class ParallelBackend
{
    Auto = 0,
    StdThread,
    OpenMP,
    Serial,
};

inline const char* parallel_backend_name(ParallelBackend backend)
{
    switch (backend)
    {
        case ParallelBackend::StdThread:
            return "std_thread";
        case ParallelBackend::OpenMP:
            return "openmp";
        case ParallelBackend::Auto:
            return "auto";
        case ParallelBackend::Serial:
        default:
            return "serial";
    }
}

namespace detail {
namespace parallel {

inline int detect_hardware_threads()
{
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

inline int max_thread_cap()
{
    const int hw = detect_hardware_threads();
    const int expanded = std::max(1, hw * 4);
    return std::min(256, expanded);
}

inline int clamp_thread_count(int requested)
{
    if (requested < 1)
    {
        requested = 1;
    }
    return std::min(requested, max_thread_cap());
}

inline std::atomic<int>& requested_threads_storage()
{
    static std::atomic<int> g_requested_threads {detect_hardware_threads()};
    return g_requested_threads;
}

inline std::atomic<int>& requested_backend_storage()
{
    static std::atomic<int> g_requested_backend {static_cast<int>(ParallelBackend::Auto)};
    return g_requested_backend;
}

inline std::atomic<bool>& strict_mode_storage()
{
    static std::atomic<bool> g_parallel_strict {false};
    return g_parallel_strict;
}

inline thread_local ParallelBackend g_last_parallel_backend = ParallelBackend::Serial;
inline thread_local int g_last_parallel_chunks = 1;

inline void set_num_threads(int nthreads)
{
    requested_threads_storage().store(clamp_thread_count(nthreads), std::memory_order_relaxed);
}

inline int get_num_threads()
{
    const int stored = requested_threads_storage().load(std::memory_order_relaxed);
    return clamp_thread_count(stored);
}

inline bool is_openmp_available()
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

inline ParallelBackend sanitize_backend(ParallelBackend backend)
{
    switch (backend)
    {
        case ParallelBackend::Auto:
        case ParallelBackend::StdThread:
        case ParallelBackend::OpenMP:
        case ParallelBackend::Serial:
            return backend;
        default:
            return ParallelBackend::Auto;
    }
}

inline void set_parallel_backend(ParallelBackend backend)
{
    requested_backend_storage().store(static_cast<int>(sanitize_backend(backend)), std::memory_order_relaxed);
}

inline ParallelBackend get_parallel_backend()
{
    const int raw = requested_backend_storage().load(std::memory_order_relaxed);
    return sanitize_backend(static_cast<ParallelBackend>(raw));
}

inline void set_parallel_strict(bool strict)
{
    strict_mode_storage().store(strict, std::memory_order_relaxed);
}

inline bool get_parallel_strict()
{
    return strict_mode_storage().load(std::memory_order_relaxed);
}

inline ParallelBackend last_parallel_backend()
{
    return g_last_parallel_backend;
}

inline int last_parallel_chunks()
{
    return g_last_parallel_chunks;
}

inline void set_last_parallel_execution(ParallelBackend backend, int chunks)
{
    g_last_parallel_backend = backend;
    g_last_parallel_chunks = std::max(1, chunks);
}

inline int compute_chunk_count(int length, int thread_count, double nstripes)
{
    CV_Assert(length >= 0);
    if (length == 0)
    {
        return 1;
    }

    if (nstripes > 0.0)
    {
        const int stripes = static_cast<int>(std::ceil(nstripes));
        return std::max(1, std::min(stripes, length));
    }

    return std::max(1, std::min(thread_count, length));
}

inline int compute_stdthread_chunk_count(int chunk_count, int thread_count)
{
    CV_Assert(chunk_count >= 1);
    CV_Assert(thread_count >= 1);
    const int max_chunks = std::max(1, thread_count * 4);
    return std::max(1, std::min(chunk_count, max_chunks));
}

inline Range compute_chunk_range(const Range& whole, int chunk_idx, int chunk_count)
{
    const int64_t begin = static_cast<int64_t>(whole.start);
    const int64_t len = static_cast<int64_t>(whole.end) - static_cast<int64_t>(whole.start);
    const int64_t s = begin + (len * static_cast<int64_t>(chunk_idx)) / static_cast<int64_t>(chunk_count);
    const int64_t e = begin + (len * static_cast<int64_t>(chunk_idx + 1)) / static_cast<int64_t>(chunk_count);
    return Range(static_cast<int>(s), static_cast<int>(e));
}

inline ParallelBackend resolve_execution_backend(ParallelBackend requested, int chunk_count, int thread_count)
{
    if (chunk_count <= 1 || thread_count <= 1)
    {
        return ParallelBackend::Serial;
    }

    if (requested == ParallelBackend::Auto)
    {
        return ParallelBackend::StdThread;
    }

    if (requested == ParallelBackend::OpenMP && !is_openmp_available())
    {
        return ParallelBackend::StdThread;
    }

    return requested;
}

template <class Fn>
inline void run_serial(const Range& range, Fn&& fn)
{
    set_last_parallel_execution(ParallelBackend::Serial, 1);
    fn(range);
}

template <class Fn>
inline void run_stdthread(const Range& range, int chunk_count, Fn&& fn)
{
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::atomic<bool> aborted {false};
    const bool strict = get_parallel_strict();
    const int worker_count = std::max(1, std::min(get_num_threads(), chunk_count));
    std::atomic<int> next_chunk {0};

    auto record_failure = [&](std::exception_ptr eptr) {
        aborted.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (!first_exception)
        {
            first_exception = eptr;
        }
    };

    auto run_chunk = [&](int chunk_idx) {
        const Range sub = compute_chunk_range(range, chunk_idx, chunk_count);
        if (sub.end <= sub.start)
        {
            return;
        }
        try
        {
            fn(sub);
        }
        catch (...)
        {
            record_failure(std::current_exception());
            throw;
        }
    };

    auto run_worker = [&]() {
        while (!aborted.load(std::memory_order_relaxed))
        {
            const int chunk_idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (chunk_idx >= chunk_count)
            {
                return;
            }
            try
            {
                run_chunk(chunk_idx);
            }
            catch (...)
            {
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(std::max(0, worker_count - 1)));

    for (int worker_idx = 1; worker_idx < worker_count; ++worker_idx)
    {
        try
        {
            workers.emplace_back(run_worker);
        }
        catch (const std::system_error&)
        {
            if (strict)
            {
                record_failure(std::current_exception());
                break;
            }
        }
    }

    run_worker();

    for (std::thread& worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    if (first_exception)
    {
        std::rethrow_exception(first_exception);
    }
}

template <class Fn>
inline void run_openmp(const Range& range, int chunk_count, Fn&& fn)
{
#ifdef _OPENMP
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::atomic<bool> aborted {false};

    const int threads = get_num_threads();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx)
    {
        if (aborted.load(std::memory_order_relaxed))
        {
            continue;
        }

        const Range sub = compute_chunk_range(range, chunk_idx, chunk_count);
        if (sub.end <= sub.start)
        {
            continue;
        }

        try
        {
            fn(sub);
        }
        catch (...)
        {
            aborted.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!first_exception)
            {
                first_exception = std::current_exception();
            }
        }
    }

    if (first_exception)
    {
        std::rethrow_exception(first_exception);
    }
#else
    run_stdthread(range, chunk_count, std::forward<Fn>(fn));
#endif
}

template <class Fn>
inline void parallel_for_impl(const Range& range, Fn&& fn, double nstripes)
{
    if (range.end < range.start)
    {
        CV_Error_(Error::StsBadArg,
                  ("parallel_for_: invalid range, start=%d end=%d", range.start, range.end));
    }
    if (std::isnan(nstripes) || nstripes == 0.0)
    {
        CV_Error_(Error::StsBadArg,
                  ("parallel_for_: invalid nstripes=%g, expected -1 or positive", nstripes));
    }

    const int length = range.end - range.start;
    if (length <= 0)
    {
        set_last_parallel_execution(ParallelBackend::Serial, 1);
        return;
    }

    const int thread_count = get_num_threads();
    const int chunk_count = compute_chunk_count(length, thread_count, nstripes);
    const ParallelBackend backend = resolve_execution_backend(get_parallel_backend(), chunk_count, thread_count);
    const int stdthread_chunk_count =
        compute_stdthread_chunk_count(chunk_count, thread_count);
    const int dispatch_chunk_count =
        backend == ParallelBackend::StdThread ? stdthread_chunk_count : chunk_count;

    if (backend == ParallelBackend::Serial)
    {
        run_serial(range, std::forward<Fn>(fn));
        return;
    }

    set_last_parallel_execution(backend, dispatch_chunk_count);
    if (backend == ParallelBackend::OpenMP)
    {
        run_openmp(range, chunk_count, std::forward<Fn>(fn));
        return;
    }

    run_stdthread(range, dispatch_chunk_count, std::forward<Fn>(fn));
}

}  // namespace parallel
}  // namespace detail

namespace cpu {

inline bool should_parallelize_1d_loop(std::size_t outer_work_units,
                                       std::size_t inner_work_units,
                                       long long min_total_work,
                                       int min_threads)
{
    if (min_threads <= 1)
    {
        min_threads = 2;
    }

    const int thread_count = detail::parallel::get_num_threads();
    if (thread_count < min_threads)
    {
        return false;
    }

    const std::uint64_t total_work = static_cast<std::uint64_t>(outer_work_units) *
                                     static_cast<std::uint64_t>(inner_work_units);
    if (total_work < static_cast<std::uint64_t>(min_total_work))
    {
        return false;
    }

    return outer_work_units >= static_cast<std::size_t>(min_threads);
}

}  // namespace cpu

}  // namespace cvh

#endif  // CVH_PARALLEL_RUNTIME_H
