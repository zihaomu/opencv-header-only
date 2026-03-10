#ifndef CVH_OPENMP_UTILS_H
#define CVH_OPENMP_UTILS_H

#include <cstddef>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvh {
namespace cpu {

inline bool should_parallelize_1d_loop(std::size_t outer_work_units,
                                       std::size_t inner_work_units,
                                       long long min_total_work,
                                       int min_threads)
{
#ifdef _OPENMP
    if (min_threads <= 1)
    {
        min_threads = 2;
    }

    const int thread_count = omp_get_max_threads();
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
#else
    (void)outer_work_units;
    (void)inner_work_units;
    (void)min_total_work;
    (void)min_threads;
    return false;
#endif
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_OPENMP_UTILS_H
