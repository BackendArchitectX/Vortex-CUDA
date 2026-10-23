#define main vortexBenchmarkCliMain
#include "../src/benchmark_m15.cu"
#undef main

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct RecallCase
{
    const char* name;
    std::vector<Neighbor> expected;
    std::vector<Neighbor> actual;
    int queryCount;
    int topK;
    int depth;
    double expectedRecall;
};

bool runCase(const RecallCase& testCase)
{
    try
    {
        const double recall =
            recallAtDepth(
                testCase.expected,
                testCase.actual,
                testCase.queryCount,
                testCase.topK,
                testCase.depth
            );

        const bool inRange =
            std::isfinite(recall) &&
            recall >= 0.0 &&
            recall <= 1.0;
        const bool matchesExpected =
            std::abs(
                recall -
                testCase.expectedRecall
            ) <= 1.0e-12;

        if (inRange && matchesExpected)
        {
            std::cout
                << "[PASS] "
                << testCase.name
                << " = "
                << recall
                << '\n';
            return true;
        }

        std::cerr
            << "[FAIL] "
            << testCase.name
            << ": expected "
            << testCase.expectedRecall
            << ", got "
            << recall
            << ", in-range="
            << (inRange ? "yes" : "no")
            << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "[FAIL] "
            << testCase.name
            << ": threw "
            << error.what()
            << '\n';
    }

    return false;
}

} // namespace

int main()
{
    const std::vector<RecallCase> cases{
        {
            "duplicate actual indices",
            {{0, 0.0F}, {1, 1.0F}},
            {{0, 0.0F}, {0, 0.5F}},
            1,
            2,
            2,
            0.5
        },
        {
            "distinct correct indices",
            {{0, 0.0F}, {1, 1.0F}},
            {{0, 0.0F}, {1, 1.0F}},
            1,
            2,
            2,
            1.0
        },
        {
            "reordered neighbors",
            {{0, 0.0F}, {1, 1.0F}},
            {{1, 1.0F}, {0, 0.0F}},
            1,
            2,
            2,
            1.0
        },
        {
            "zero matches",
            {{0, 0.0F}, {1, 1.0F}},
            {{2, 2.0F}, {3, 3.0F}},
            1,
            2,
            2,
            0.0
        },
        {
            "topK equals one",
            {{5, 0.0F}},
            {{5, 0.0F}},
            1,
            1,
            1,
            1.0
        },
        {
            "depth less than topK",
            {{10, 0.0F}, {11, 1.0F}, {12, 2.0F}},
            {{11, 1.0F}, {10, 0.0F}, {12, 2.0F}},
            1,
            3,
            2,
            1.0
        },
        {
            "multiple queries",
            {{0, 0.0F}, {1, 1.0F}, {5, 0.0F}, {6, 1.0F}},
            {{1, 1.0F}, {0, 0.0F}, {99, 9.0F}, {6, 1.0F}},
            2,
            2,
            2,
            0.75
        },
        {
            "duplicate indices isolated by query",
            {{0, 0.0F}, {1, 1.0F}, {0, 0.0F}, {1, 1.0F}},
            {{0, 0.0F}, {0, 0.5F}, {0, 0.0F}, {1, 1.0F}},
            2,
            2,
            2,
            0.75
        },
        {
            "duplicate indices within partial depth",
            {{10, 0.0F}, {11, 1.0F}, {12, 2.0F}},
            {{10, 0.0F}, {10, 0.5F}, {12, 2.0F}},
            1,
            3,
            2,
            0.5
        }
    };

    std::size_t passed = 0;

    for (const RecallCase& testCase : cases)
    {
        if (runCase(testCase))
        {
            ++passed;
        }
    }

    const std::size_t failed =
        cases.size() -
        passed;

    std::cout
        << "Recall cases: "
        << passed
        << " passed, "
        << failed
        << " failed, "
        << cases.size()
        << " total\n";

    return failed == 0
        ? 0
        : 1;
}
