from __future__ import annotations

import unittest

import numpy as np

from tests.support import NO_READER, SharedBuffer, cleanup_buffer, make_name, release_mem_views


class SharedBufferBehaviorTests(unittest.TestCase):
    def _make_pair(self, *, size: int = 64, num_readers: int = 1) -> tuple[SharedBuffer, SharedBuffer]:
        name = make_name("buffer")
        writer = SharedBuffer(
            name=name,
            create=True,
            size=size,
            num_readers=num_readers,
            reader=NO_READER,
        )
        reader = SharedBuffer(
            name=name,
            create=False,
            size=size,
            num_readers=num_readers,
            reader=0,
        )
        self.addCleanup(cleanup_buffer, writer, unlink=True)
        self.addCleanup(cleanup_buffer, reader, unlink=False)
        return writer, reader

    @staticmethod
    def _view_len(view: tuple[memoryview, memoryview | None, int, bool]) -> int:
        mv1, mv2, _, _ = view
        return mv1.nbytes + (mv2.nbytes if mv2 is not None else 0)

    def test_simple_copy_helpers_preserve_bytes(self) -> None:
        writer, reader = self._make_pair(size=64)
        payload = b"uci-rpl-liquids"

        writer_view = writer.expose_writer_mem_view(len(payload))
        self.assertEqual(writer_view[2], len(payload))
        self.assertEqual(self._view_len(writer_view), len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], len(payload))
        self.assertEqual(self._view_len(reader_view), len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(reader_view, dst)
        reader.inc_reader_pos(len(payload))
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), payload)

    def test_simple_write_does_not_publish_until_writer_position_advances(self) -> None:
        writer, reader = self._make_pair(size=32)
        payload = b"not-published-yet"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], 0)
        release_mem_views(reader_view[0], reader_view[1])

        writer.inc_writer_pos(len(payload))
        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(reader_view, dst)
        release_mem_views(reader_view[0], reader_view[1])
        self.assertEqual(bytes(dst), payload)

    def test_simple_read_does_not_consume_until_reader_position_advances(self) -> None:
        writer, reader = self._make_pair(size=32)
        payload = b"read-twice"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        first_view = reader.expose_reader_mem_view(len(payload))
        first_dst = bytearray(len(payload))
        reader.simple_read(first_view, first_dst)
        release_mem_views(first_view[0], first_view[1])

        second_view = reader.expose_reader_mem_view(len(payload))
        second_dst = bytearray(len(payload))
        reader.simple_read(second_view, second_dst)
        release_mem_views(second_view[0], second_view[1])

        self.assertEqual(bytes(first_dst), payload)
        self.assertEqual(bytes(second_dst), payload)

        reader.inc_reader_pos(len(payload))
        exhausted_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(exhausted_view[2], 0)
        release_mem_views(exhausted_view[0], exhausted_view[1])

    def test_multiple_sequential_writes_and_reads_preserve_order(self) -> None:
        writer, reader = self._make_pair(size=64)
        payloads = [b"alpha", b"beta123", b"gamma!"]

        for payload in payloads:
            writer_view = writer.expose_writer_mem_view(len(payload))
            writer.simple_write(writer_view, payload)
            writer.inc_writer_pos(len(payload))
            release_mem_views(writer_view[0], writer_view[1])

        for payload in payloads:
            reader_view = reader.expose_reader_mem_view(len(payload))
            dst = bytearray(len(payload))
            reader.simple_read(reader_view, dst)
            reader.inc_reader_pos(len(payload))
            release_mem_views(reader_view[0], reader_view[1])
            self.assertEqual(bytes(dst), payload)

    def test_space_is_reusable_after_partial_consumption_preserves_order(self) -> None:
        writer, reader = self._make_pair(size=16)
        first_payload = b"ABCDEFGHIJKL"
        second_payload = b"MNOPQR"
        reader.set_reader_active(True)

        writer_view = writer.expose_writer_mem_view(len(first_payload))
        self.assertEqual(writer_view[2], len(first_payload))
        writer.simple_write(writer_view, first_payload)
        writer.inc_writer_pos(len(first_payload))
        release_mem_views(writer_view[0], writer_view[1])

        first_read_view = reader.expose_reader_mem_view(10)
        self.assertEqual(first_read_view[2], 10)
        first_dst = bytearray(10)
        reader.simple_read(first_read_view, first_dst)
        reader.inc_reader_pos(10)
        release_mem_views(first_read_view[0], first_read_view[1])
        self.assertEqual(bytes(first_dst), first_payload[:10])

        writer.compute_max_amount_writable(force_rescan=True)
        second_write_view = writer.expose_writer_mem_view(len(second_payload))
        self.assertEqual(second_write_view[2], len(second_payload))
        writer.simple_write(second_write_view, second_payload)
        writer.inc_writer_pos(len(second_payload))
        release_mem_views(second_write_view[0], second_write_view[1])

        remaining = first_payload[10:] + second_payload
        reader_view = reader.expose_reader_mem_view(len(remaining))
        self.assertEqual(reader_view[2], len(remaining))
        dst = bytearray(len(remaining))
        reader.simple_read(reader_view, dst)
        reader.inc_reader_pos(len(remaining))
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), remaining)

    def test_write_array_and_read_array_preserve_dtype_and_values(self) -> None:
        cases = (
            np.array([4, 8, 15, 16, 23, 42], dtype=np.int16),
            np.array([1.25, 2.5, 5.0, 10.0], dtype=np.float32),
            np.array([0, 127, 255], dtype=np.uint8),
        )

        for arr in cases:
            with self.subTest(dtype=str(arr.dtype), nbytes=arr.nbytes):
                writer, reader = self._make_pair(size=128)
                bytes_written = writer.write_array(arr)
                self.assertEqual(bytes_written, arr.nbytes)

                roundtrip = reader.read_array(arr.nbytes, arr.dtype)
                np.testing.assert_array_equal(roundtrip, arr)

    def test_write_array_returns_zero_when_full_payload_does_not_fit(self) -> None:
        writer, reader = self._make_pair(size=16)
        arr = np.arange(8, dtype=np.int16)

        reader.update_reader_pos(0)
        reader.set_reader_active(True)
        writer.update_write_pos(4)
        writer.compute_max_amount_writable(force_rescan=True)

        bytes_written = writer.write_array(arr)
        self.assertEqual(bytes_written, 0)

    def test_read_array_returns_empty_when_not_enough_bytes_are_available(self) -> None:
        writer, reader = self._make_pair(size=64)
        arr = np.array([1, 2, 3], dtype=np.int16)

        writer.write_array(arr)

        roundtrip = reader.read_array(arr.nbytes + arr.dtype.itemsize, arr.dtype)
        self.assertEqual(roundtrip.size, 0)
        self.assertEqual(roundtrip.dtype, arr.dtype)

    def test_space_reuse_with_arrays_preserves_order_and_values(self) -> None:
        writer, reader = self._make_pair(size=16)
        first = np.array([10, 20, 30, 40, 50, 60], dtype=np.int16)
        second = np.array([70, 80, 90, 100], dtype=np.int16)

        reader.set_reader_active(True)

        bytes_written = writer.write_array(first)
        self.assertEqual(bytes_written, first.nbytes)

        consumed_prefix = reader.read_array(8, first.dtype)
        np.testing.assert_array_equal(consumed_prefix, first[:4])

        writer.compute_max_amount_writable(force_rescan=True)
        bytes_written = writer.write_array(second)
        self.assertEqual(bytes_written, second.nbytes)

        remaining = reader.read_array(first[4:].nbytes + second.nbytes, first.dtype)
        np.testing.assert_array_equal(remaining, np.concatenate((first[4:], second)))

    def test_writer_buffer_request_clamps_when_space_is_insufficient(self) -> None:
        writer, reader = self._make_pair(size=16)
        reader.update_reader_pos(0)
        reader.set_reader_active(True)
        writer.update_write_pos(12)
        writer.compute_max_amount_writable(force_rescan=True)

        view = writer.expose_writer_mem_view(8)
        self.assertEqual(view[2], 4)
        self.assertEqual(self._view_len(view), 4)
        release_mem_views(view[0], view[1])

    def test_reader_buffer_request_clamps_when_data_is_insufficient(self) -> None:
        writer, reader = self._make_pair(size=16)
        writer.update_write_pos(3)

        view = reader.expose_reader_mem_view(8)
        self.assertEqual(view[2], 3)
        self.assertEqual(self._view_len(view), 3)
        release_mem_views(view[0], view[1])

    def test_zero_length_requests_return_empty_views(self) -> None:
        writer, reader = self._make_pair(size=16)

        writer_view = writer.expose_writer_mem_view(0)
        self.assertEqual(writer_view[2], 0)
        self.assertEqual(self._view_len(writer_view), 0)
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(0)
        self.assertEqual(reader_view[2], 0)
        self.assertEqual(self._view_len(reader_view), 0)
        release_mem_views(reader_view[0], reader_view[1])

    def test_simple_write_truncates_to_exposed_destination_length(self) -> None:
        writer, reader = self._make_pair(size=16)
        payload = b"abcdef"

        writer_view = writer.expose_writer_mem_view(4)
        self.assertEqual(writer_view[2], 4)
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(writer_view[2])
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], 4)
        dst = bytearray(4)
        reader.simple_read(reader_view, dst)
        reader.inc_reader_pos(reader_view[2])
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), payload[:4])

    def test_simple_read_into_larger_destination_preserves_tail(self) -> None:
        writer, reader = self._make_pair(size=64)
        payload = b"buffer"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        dst = bytearray([0xAA] * 12)
        reader.simple_read(reader_view, dst)
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst[:len(payload)]), payload)
        self.assertEqual(bytes(dst[len(payload):]), b"\xaa" * (len(dst) - len(payload)))

    def test_simple_read_into_smaller_destination_truncates(self) -> None:
        writer, reader = self._make_pair(size=64)
        payload = b"truncate-me"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        dst = bytearray(4)
        reader.simple_read(reader_view, dst)
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), payload[:4])

    def test_exact_buffer_sized_write_is_fully_exposed_when_unconstrained(self) -> None:
        writer, _ = self._make_pair(size=16)
        writer_view = writer.expose_writer_mem_view(16)
        self.assertEqual(writer_view[2], 16)
        self.assertEqual(self._view_len(writer_view), 16)
        release_mem_views(writer_view[0], writer_view[1])

    def test_exact_buffer_sized_read_is_allowed_at_retention_boundary(self) -> None:
        writer, reader = self._make_pair(size=16)
        payload = bytes(range(16))

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(reader_view, dst)
        reader.inc_reader_pos(len(payload))
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), payload)
