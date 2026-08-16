package com.vortex.cuda;

import java.util.Arrays;
import java.util.SplittableRandom;

public final class VortexJniSmokeTest {

    private static final int VECTOR_COUNT =
            25_000;

    private static final int DIMENSION =
            128;

    private static final int QUERY_COUNT =
            4;

    private static final int TOP_K =
            10;

    private static final long DATABASE_SEED =
            42L;

    private static final long QUERY_SEED =
            1337L;

    private VortexJniSmokeTest() {
    }

    public static void main(
            String[] args
    ) {
        System.out.println(
                "============================================"
        );

        System.out.println(
                " Vortex CUDA M16 JNI Smoke Test"
        );

        System.out.println(
                "============================================"
        );

        float[] database =
                generateRandomFloats(
                        Math.multiplyExact(
                                VECTOR_COUNT,
                                DIMENSION
                        ),
                        DATABASE_SEED
                );

        float[] queries =
                generateRandomFloats(
                        Math.multiplyExact(
                                QUERY_COUNT,
                                DIMENSION
                        ),
                        QUERY_SEED
                );

        try (
                VortexIndex index =
                        new VortexIndex(
                                database,
                                VECTOR_COUNT,
                                DIMENSION
                        )
        ) {
            System.out.println(
                    "Native library: " +
                            index.nativeVersion()
            );

            System.out.println(
                    "Index:          " +
                            index.vectorCount() +
                            " x " +
                            index.dimension()
            );

            IndexStats stats =
                    index.stats();

            System.out.printf(
                    "FP32 bytes:     %,d%n",
                    stats.fp32IndexBytes()
            );

            System.out.printf(
                    "FP16 bytes:     %,d%n",
                    stats.fp16IndexBytes()
            );

            System.out.printf(
                    "Reduction:      %.3f%%%n",
                    stats.logicalMemoryReductionPercent()
            );

            BatchSearchResult fp32 =
                    index.searchBatch(
                            queries,
                            QUERY_COUNT,
                            TOP_K,
                            SearchPrecision
                                    .FP32_EXACT
                    );

            BatchSearchResult fp16 =
                    index.searchBatch(
                            queries,
                            QUERY_COUNT,
                            TOP_K,
                            SearchPrecision
                                    .FP16_STORAGE_FP32_ACCUMULATION
                    );

            int[][] cpuExact =
                    cpuExactTopK(
                            database,
                            queries,
                            VECTOR_COUNT,
                            DIMENSION,
                            QUERY_COUNT,
                            TOP_K
                    );

            verifyFp32AgainstCpu(
                    fp32,
                    cpuExact
            );

            verifySingleQueryMatchesBatch(
                    index,
                    queries,
                    fp32
            );

            double fp16Recall =
                    recallAtK(
                            fp32,
                            fp16
                    );

            if (fp16Recall <
                    0.98) {

                throw new AssertionError(
                        "FP16 Recall@K below 98%: " +
                                fp16Recall
                );
            }

            verifySortedAndValid(
                    fp32,
                    VECTOR_COUNT
            );

            verifySortedAndValid(
                    fp16,
                    VECTOR_COUNT
            );

            System.out.println(
                    "JNI FP32 vs CPU exact: PASS"
            );

            System.out.println(
                    "JNI single vs batch:   PASS"
            );

            System.out.printf(
                    "JNI FP16 Recall@%d:    %.6f%%%n",
                    TOP_K,
                    fp16Recall *
                            100.0
            );

            System.out.println(
                    "JNI result validation: PASS"
            );

            SearchResult first =
                    fp16.resultForQuery(
                            0
                    );

            System.out.println(
                    "\nQuery 0 FP16 Top-K"
            );

            for (int rank = 0;
                 rank <
                         first.size();
                 ++rank) {

                Neighbor neighbor =
                        first.neighbor(
                                rank
                        );

                System.out.printf(
                        "#%2d vector=%6d squared_l2=%.6f%n",
                        rank + 1,
                        neighbor.index(),
                        neighbor.squaredL2Distance()
                );
            }
        }

        System.out.println(
                "\nM16 JNI smoke test completed successfully."
        );
    }

    private static float[] generateRandomFloats(
            int count,
            long seed
    ) {
        SplittableRandom random =
                new SplittableRandom(
                        seed
                );

        float[] values =
                new float[
                        count
                ];

        for (int index = 0;
             index <
                     values.length;
             ++index) {

            values[index] =
                    (float) (
                            random.nextDouble(
                                    -1.0,
                                    1.0
                            )
                    );
        }

        return values;
    }

    private static int[][] cpuExactTopK(
            float[] database,
            float[] queries,
            int vectorCount,
            int dimension,
            int queryCount,
            int topK
    ) {
        int[][] results =
                new int[
                        queryCount
                ][
                        topK
                ];

        for (int queryIndex = 0;
             queryIndex <
                     queryCount;
             ++queryIndex) {

            float[] bestDistances =
                    new float[
                            topK
                    ];

            int[] bestIndices =
                    new int[
                            topK
                    ];

            Arrays.fill(
                    bestDistances,
                    Float.POSITIVE_INFINITY
            );

            Arrays.fill(
                    bestIndices,
                    Integer.MAX_VALUE
            );

            int queryBase =
                    Math.multiplyExact(
                            queryIndex,
                            dimension
                    );

            for (int vectorIndex = 0;
                 vectorIndex <
                         vectorCount;
                 ++vectorIndex) {

                int vectorBase =
                        Math.multiplyExact(
                                vectorIndex,
                                dimension
                        );

                float distance =
                        0.0F;

                for (int d = 0;
                     d <
                             dimension;
                     ++d) {

                    float difference =
                            database[
                                    vectorBase +
                                            d
                            ] -
                                    queries[
                                            queryBase +
                                                    d
                                    ];

                    distance =
                            Math.fma(
                                    difference,
                                    difference,
                                    distance
                            );
                }

                int insertion =
                        topK;

                for (int rank = 0;
                     rank <
                             topK;
                     ++rank) {

                    if (distance <
                            bestDistances[
                                    rank
                            ] ||
                            (
                                    distance ==
                                            bestDistances[
                                                    rank
                                            ] &&
                                            vectorIndex <
                                                    bestIndices[
                                                            rank
                                                    ]
                            )) {

                        insertion =
                                rank;
                        break;
                    }
                }

                if (insertion <
                        topK) {

                    for (int rank =
                                 topK - 1;
                         rank >
                                 insertion;
                         --rank) {

                        bestDistances[
                                rank
                        ] =
                                bestDistances[
                                        rank - 1
                                ];

                        bestIndices[
                                rank
                        ] =
                                bestIndices[
                                        rank - 1
                                ];
                    }

                    bestDistances[
                            insertion
                    ] =
                            distance;

                    bestIndices[
                            insertion
                    ] =
                            vectorIndex;
                }
            }

            results[
                    queryIndex
            ] =
                    bestIndices;
        }

        return results;
    }

    private static void verifyFp32AgainstCpu(
            BatchSearchResult nativeFp32,
            int[][] cpuExact
    ) {
        for (int queryIndex = 0;
             queryIndex <
                     nativeFp32.queryCount();
             ++queryIndex) {

            int[] nativeIndices =
                    nativeFp32
                            .resultForQuery(
                                    queryIndex
                            )
                            .indices();

            if (!Arrays.equals(
                    nativeIndices,
                    cpuExact[
                            queryIndex
                    ]
            )) {

                throw new AssertionError(
                        "FP32 JNI result differs from CPU exact reference for query " +
                                queryIndex +
                                "\nNative: " +
                                Arrays.toString(
                                        nativeIndices
                                ) +
                                "\nCPU:    " +
                                Arrays.toString(
                                        cpuExact[
                                                queryIndex
                                        ]
                                )
                );
            }
        }
    }

    private static void verifySingleQueryMatchesBatch(
            VortexIndex index,
            float[] queries,
            BatchSearchResult batchFp32
    ) {
        float[] firstQuery =
                Arrays.copyOfRange(
                        queries,
                        0,
                        DIMENSION
                );

        SearchResult single =
                index.search(
                        firstQuery,
                        TOP_K,
                        SearchPrecision
                                .FP32_EXACT
                );

        SearchResult batch =
                batchFp32
                        .resultForQuery(
                                0
                        );

        if (!Arrays.equals(
                single.indices(),
                batch.indices()
        )) {

            throw new AssertionError(
                    "Single-query JNI result differs from batch query 0."
            );
        }
    }

    private static double recallAtK(
            BatchSearchResult expected,
            BatchSearchResult actual
    ) {
        if (expected.queryCount() !=
                actual.queryCount() ||
                expected.topK() !=
                        actual.topK()) {

            throw new IllegalArgumentException(
                    "Recall inputs have incompatible shapes."
            );
        }

        int matches = 0;

        for (int queryIndex = 0;
             queryIndex <
                     expected.queryCount();
             ++queryIndex) {

            int[] expectedIndices =
                    expected
                            .resultForQuery(
                                    queryIndex
                            )
                            .indices();

            int[] actualIndices =
                    actual
                            .resultForQuery(
                                    queryIndex
                            )
                            .indices();

            for (int actualIndex :
                    actualIndices) {

                for (int expectedIndex :
                        expectedIndices) {

                    if (actualIndex ==
                            expectedIndex) {

                        ++matches;
                        break;
                    }
                }
            }
        }

        int denominator =
                Math.multiplyExact(
                        expected.queryCount(),
                        expected.topK()
                );

        return
                (double) matches /
                        (double) denominator;
    }

    private static void verifySortedAndValid(
            BatchSearchResult result,
            int vectorCount
    ) {
        for (int queryIndex = 0;
             queryIndex <
                     result.queryCount();
             ++queryIndex) {

            SearchResult query =
                    result.resultForQuery(
                            queryIndex
                    );

            float previous =
                    Float.NEGATIVE_INFINITY;

            for (int rank = 0;
                 rank <
                         query.size();
                 ++rank) {

                Neighbor neighbor =
                        query.neighbor(
                                rank
                        );

                if (neighbor.index() <
                        0 ||
                        neighbor.index() >=
                                vectorCount) {

                    throw new AssertionError(
                            "Invalid vector index in JNI result."
                    );
                }

                if (!Float.isFinite(
                        neighbor.squaredL2Distance()
                )) {

                    throw new AssertionError(
                            "Non-finite distance in JNI result."
                    );
                }

                if (neighbor.squaredL2Distance() <
                        previous) {

                    throw new AssertionError(
                            "JNI Top-K distances are not sorted."
                    );
                }

                previous =
                        neighbor.squaredL2Distance();
            }
        }
    }
}
