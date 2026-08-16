package com.vortex.cuda;

import java.util.Random;

public final class VortexDemo {

    public static void main(String[] args) {

        int vectorCount = 10_000;
        int dimension = 128;
        int topK = 10;

        // Database: 10,000 vectors × 128 dimensions
        float[] database =
                new float[vectorCount * dimension];

        // One query vector
        float[] query =
                new float[dimension];

        Random random =
                new Random(42);

        for (int i = 0; i < database.length; i++) {
            database[i] =
                    random.nextFloat();
        }

        for (int i = 0; i < query.length; i++) {
            query[i] =
                    random.nextFloat();
        }

        try (VortexIndex index =
                     new VortexIndex(
                             database,
                             vectorCount,
                             dimension
                     )) {

            SearchResult result =
                    index.searchFp16(
                            query,
                            topK
                    );

            System.out.println(
                    "Top-" + topK + " results:"
            );

            for (int rank = 0;
                 rank < result.size();
                 rank++) {

                Neighbor neighbor =
                        result.neighbor(rank);

                System.out.printf(
                        "#%d vector=%d squared_l2=%.6f%n",
                        rank + 1,
                        neighbor.index(),
                        neighbor.squaredL2Distance()
                );
            }
        }
    }
}