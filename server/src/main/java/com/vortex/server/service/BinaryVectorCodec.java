package com.vortex.server.service;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class BinaryVectorCodec {

    private BinaryVectorCodec() {
    }

    public static float[] decodeLittleEndianFloat32(
            byte[] body,
            int vectorCount,
            int dimension
    ) {
        if (body == null) {
            throw new IllegalArgumentException(
                    "Binary vector payload must not be null."
            );
        }

        if (vectorCount <= 0) {
            throw new IllegalArgumentException(
                    "vectorCount must be positive."
            );
        }

        if (dimension <= 0) {
            throw new IllegalArgumentException(
                    "dimension must be positive."
            );
        }

        long elementCount =
                Math.multiplyExact(
                        (long) vectorCount,
                        (long) dimension
                );

        long expectedBytes =
                Math.multiplyExact(
                        elementCount,
                        (long) Float.BYTES
                );

        if (expectedBytes >
                Integer.MAX_VALUE) {

            throw new IllegalArgumentException(
                    "Binary payload is too large for a single request."
            );
        }

        if (body.length !=
                (int) expectedBytes) {

            throw new IllegalArgumentException(
                    "Binary payload length must equal " +
                            expectedBytes +
                            " bytes for vectorCount=" +
                            vectorCount +
                            " and dimension=" +
                            dimension +
                            "."
            );
        }

        float[] vectors =
                new float[
                        Math.toIntExact(
                                elementCount
                        )
                ];

        ByteBuffer.wrap(body)
                .order(
                        ByteOrder.LITTLE_ENDIAN
                )
                .asFloatBuffer()
                .get(vectors);

        return vectors;
    }
}
