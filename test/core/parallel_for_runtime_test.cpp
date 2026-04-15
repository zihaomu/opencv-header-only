#include "cvh.h"
#include "gtest/gtest.h"

#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace {

int expected_thread_cap()
{
    const unsigned int hw = std::thread::hardware_concurrency();
    const int hw_threads = hw > 0 ? static_cast<int>(hw) : 1;
    return std::min(256, std::max(1, hw_threads * 4));
}

struct ParallelRuntimeScope
{
    int threads = 1;
    cvh::ParallelBackend backend = cvh::ParallelBackend::Auto;
    bool strict = false;

    ParallelRuntimeScope()
        : threads(cvh::getNumThreads()),
          backend(cvh::getParallelBackend()),
          strict(cvh::getParallelStrict())
    {
    }

    ~ParallelRuntimeScope()
    {
        cvh::setNumThreads(threads);
        cvh::setParallelBackend(backend);
        cvh::setParallelStrict(strict);
    }
};

class ParallelSumBody : public cvh::ParallelLoopBody
{
public:
    explicit ParallelSumBody(std::atomic<long long>* sum) : sum_(sum) {}

    void operator()(const cvh::Range& range) const override
    {
        long long local = 0;
        for (int i = range.start; i < range.end; ++i)
        {
            local += static_cast<long long>(i);
        }
        sum_->fetch_add(local, std::memory_order_relaxed);
    }

private:
    std::atomic<long long>* sum_;
};

}  // namespace

TEST(ParallelRuntime_TEST, set_num_threads_is_clamped)
{
    ParallelRuntimeScope scope;

    cvh::setNumThreads(0);
    EXPECT_EQ(cvh::getNumThreads(), 1);

    cvh::setNumThreads(-7);
    EXPECT_EQ(cvh::getNumThreads(), 1);

    cvh::setNumThreads(1000000);
    EXPECT_LE(cvh::getNumThreads(), expected_thread_cap());
    EXPECT_GE(cvh::getNumThreads(), 1);
}

TEST(ParallelRuntime_TEST, single_thread_runs_serial)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(1);
    cvh::setParallelBackend(cvh::ParallelBackend::StdThread);

    std::atomic<int> callback_count {0};
    cvh::parallel_for_(
        cvh::Range(0, 128),
        [&](const cvh::Range& range) {
            ++callback_count;
            for (int i = range.start; i < range.end; ++i)
            {
                (void)i;
            }
        });

    EXPECT_EQ(callback_count.load(), 1);
    EXPECT_STREQ(cvh::last_parallel_backend(), "serial");
    EXPECT_EQ(cvh::last_parallel_chunks(), 1);
}

TEST(ParallelRuntime_TEST, stdthread_backend_covers_each_index_once)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(4);
    cvh::setParallelBackend(cvh::ParallelBackend::StdThread);

    constexpr int kLength = 257;
    std::vector<std::atomic<int>> hits(static_cast<std::size_t>(kLength));
    for (auto& hit : hits)
    {
        hit.store(0, std::memory_order_relaxed);
    }

    cvh::parallel_for_(
        cvh::Range(0, kLength),
        [&](const cvh::Range& range) {
            for (int i = range.start; i < range.end; ++i)
            {
                hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
            }
        },
        8.0);

    for (const auto& hit : hits)
    {
        EXPECT_EQ(hit.load(std::memory_order_relaxed), 1);
    }
    EXPECT_STREQ(cvh::last_parallel_backend(), "std_thread");
    EXPECT_EQ(cvh::last_parallel_chunks(), 8);
}

TEST(ParallelRuntime_TEST, serial_backend_ignores_requested_stripes)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(8);
    cvh::setParallelBackend(cvh::ParallelBackend::Serial);

    std::atomic<int> callback_count {0};
    std::atomic<int> callback_start {std::numeric_limits<int>::max()};
    std::atomic<int> callback_end {std::numeric_limits<int>::min()};
    cvh::parallel_for_(
        cvh::Range(0, 257),
        [&](const cvh::Range& range) {
            callback_start.store(range.start, std::memory_order_relaxed);
            callback_end.store(range.end, std::memory_order_relaxed);
            callback_count.fetch_add(1, std::memory_order_relaxed);
        },
        64.0);

    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(callback_start.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(callback_end.load(std::memory_order_relaxed), 257);
    EXPECT_STREQ(cvh::last_parallel_backend(), "serial");
    EXPECT_EQ(cvh::last_parallel_chunks(), 1);
}

TEST(ParallelRuntime_TEST, openmp_backend_falls_back_when_unavailable)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(4);
    cvh::setParallelBackend(cvh::ParallelBackend::OpenMP);

    std::atomic<int> chunks {0};
    cvh::parallel_for_(
        cvh::Range(0, 64),
        [&](const cvh::Range& range) {
            if (range.end > range.start)
            {
                chunks.fetch_add(1, std::memory_order_relaxed);
            }
        },
        4.0);

    ASSERT_GE(chunks.load(std::memory_order_relaxed), 1);
    if (cvh::isOpenMPBackendAvailable())
    {
        EXPECT_STREQ(cvh::last_parallel_backend(), "openmp");
    }
    else
    {
        EXPECT_STREQ(cvh::last_parallel_backend(), "std_thread");
    }
}

TEST(ParallelRuntime_TEST, worker_exception_propagates)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(4);
    cvh::setParallelBackend(cvh::ParallelBackend::StdThread);
    cvh::setParallelStrict(false);

    EXPECT_THROW(
        cvh::parallel_for_(
            cvh::Range(0, 100),
            [](const cvh::Range& range) {
                if (range.start <= 42 && 42 < range.end)
                {
                    CV_Error(cvh::Error::StsError, "parallel worker failure");
                }
            },
            4.0),
        cvh::Exception);
}

TEST(ParallelRuntime_TEST, parallel_loop_body_overload_matches_expected_sum)
{
    ParallelRuntimeScope scope;
    cvh::setNumThreads(4);
    cvh::setParallelBackend(cvh::ParallelBackend::StdThread);

    std::atomic<long long> sum {0};
    ParallelSumBody body(&sum);

    cvh::parallel_for_(cvh::Range(1, 101), body, 5.0);

    EXPECT_EQ(sum.load(std::memory_order_relaxed), 5050);
}
