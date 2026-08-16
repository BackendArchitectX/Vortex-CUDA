package com.vortex.cuda;

import java.util.Arrays;

public final class SearchResult {

    private final int[] indices;
    private final float[] distances;

    SearchResult(
            int[] indices,
            float[] distances
    ) {
        if (indices == null ||
                distances == null) {

            throw new IllegalArgumentException(
                    "indices and distances must not be null."
            );
        }

        if (indices.length !=
                distances.length) {

            throw new IllegalArgumentException(
                    "indices and distances must have equal lengths."
            );
        }

        this.indices =
                indices.clone();

        this.distances =
                distances.clone();
    }

    public int size() {
        return indices.length;
    }

    public Neighbor neighbor(
            int rank
    ) {
        if (rank < 0 ||
                rank >=
                        indices.length) {

            throw new IndexOutOfBoundsException(
                    "rank=" + rank
            );
        }

        return new Neighbor(
                indices[rank],
                distances[rank]
        );
    }

    public int[] indices() {
        return indices.clone();
    }

    public float[] squaredL2Distances() {
        return distances.clone();
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "indices=" +
                Arrays.toString(indices) +
                ", distances=" +
                Arrays.toString(distances) +
                '}';
    }
}
