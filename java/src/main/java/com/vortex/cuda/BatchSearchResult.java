package com.vortex.cuda;

import java.util.Arrays;

public final class BatchSearchResult {

    private final int queryCount;
    private final int topK;
    private final int[] indices;
    private final float[] distances;

    BatchSearchResult(
            int queryCount,
            int topK,
            int[] indices,
            float[] distances
    ) {
        if (queryCount <= 0) {
            throw new IllegalArgumentException(
                    "queryCount must be positive."
            );
        }

        if (topK <= 0) {
            throw new IllegalArgumentException(
                    "topK must be positive."
            );
        }

        int expected =
                Math.multiplyExact(
                        queryCount,
                        topK
                );

        if (indices == null ||
                distances == null ||
                indices.length != expected ||
                distances.length != expected) {

            throw new IllegalArgumentException(
                    "Batch result arrays do not match queryCount * topK."
            );
        }

        this.queryCount =
                queryCount;

        this.topK =
                topK;

        this.indices =
                indices.clone();

        this.distances =
                distances.clone();
    }

    public int queryCount() {
        return queryCount;
    }

    public int topK() {
        return topK;
    }

    public SearchResult resultForQuery(
            int queryIndex
    ) {
        if (queryIndex < 0 ||
                queryIndex >=
                        queryCount) {

            throw new IndexOutOfBoundsException(
                    "queryIndex=" +
                            queryIndex
            );
        }

        int from =
                Math.multiplyExact(
                        queryIndex,
                        topK
                );

        int to =
                from +
                        topK;

        return new SearchResult(
                Arrays.copyOfRange(
                        indices,
                        from,
                        to
                ),
                Arrays.copyOfRange(
                        distances,
                        from,
                        to
                )
        );
    }

    public int[] flattenedIndices() {
        return indices.clone();
    }

    public float[] flattenedSquaredL2Distances() {
        return distances.clone();
    }
}
