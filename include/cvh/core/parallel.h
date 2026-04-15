#ifndef CVH_PARALLEL_H
#define CVH_PARALLEL_H

#include "mat.h"
#include "detail/parallel_runtime.h"

#include <utility>

namespace cvh {

class ParallelLoopBody
{
public:
    virtual ~ParallelLoopBody() = default;
    virtual void operator()(const Range& range) const = 0;
};

inline void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes = -1.0)
{
    detail::parallel::parallel_for_impl(
        range,
        [&body](const Range& subrange) {
            body(subrange);
        },
        nstripes);
}

template <class Fn>
inline void parallel_for_(const Range& range, Fn&& body, double nstripes = -1.0)
{
    detail::parallel::parallel_for_impl(range, std::forward<Fn>(body), nstripes);
}

inline void setNumThreads(int nthreads)
{
    detail::parallel::set_num_threads(nthreads);
}

inline int getNumThreads()
{
    return detail::parallel::get_num_threads();
}

inline void setParallelBackend(ParallelBackend backend)
{
    detail::parallel::set_parallel_backend(backend);
}

inline ParallelBackend getParallelBackend()
{
    return detail::parallel::get_parallel_backend();
}

inline bool isOpenMPBackendAvailable()
{
    return detail::parallel::is_openmp_available();
}

inline void setParallelStrict(bool strict)
{
    detail::parallel::set_parallel_strict(strict);
}

inline bool getParallelStrict()
{
    return detail::parallel::get_parallel_strict();
}

inline const char* last_parallel_backend()
{
    return parallel_backend_name(detail::parallel::last_parallel_backend());
}

inline int last_parallel_chunks()
{
    return detail::parallel::last_parallel_chunks();
}

}  // namespace cvh

#endif  // CVH_PARALLEL_H
