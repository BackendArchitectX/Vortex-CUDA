package com.vortex.server.service;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BinaryVectorCodecTest {

    @Test
    void decodesLittleEndianFloat32Payload() {
        float[] expected = {
                1.25F,
                -2.5F,
                3.75F,
                4.5F,
                -5.25F,
                6.0F
        };

        byte[] bytes =
                encode(
                        expected
                );

        float[] actual =
                BinaryVectorCodec
                        .decodeLittleEndianFloat32(
                                bytes,
                                2,
                                3
                        );

        assertEquals(
                expected.length,
                actual.length
        );

        for (int index = 0;
             index < expected.length;
             index++) {

            assertEquals(
                    expected[index],
                    actual[index],
                    0.0F
            );
        }
    }

    @Test
    void rejectsPayloadWithWrongLength() {
        byte[] bytes =
                new byte[
                        4
                ];

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        BinaryVectorCodec
                                .decodeLittleEndianFloat32(
                                        bytes,
                                        2,
                                        3
                                )
        );
    }

    private static byte[] encode(
            float[] values
    ) {
        ByteBuffer buffer =
                ByteBuffer.allocate(
                                values.length *
                                        Float.BYTES
                        )
                        .order(
                                ByteOrder.LITTLE_ENDIAN
                        );

        for (float value : values) {
            buffer.putFloat(value);
        }

        return buffer.array();
    }
}
