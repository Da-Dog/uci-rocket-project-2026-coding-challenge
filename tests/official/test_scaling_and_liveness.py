from __future__ import annotations

import unittest

from tests.support import NO_READER, SharedBuffer, cleanup_buffer, make_name, release_mem_views


class SharedBufferScalingAndLivenessTests(unittest.TestCase):
    def _make_writer(self, *, size: int = 64, num_readers: int = 1) -> SharedBuffer:
        ring = SharedBuffer(
            name=make_name("liveness"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=NO_READER,
        )
        self.addCleanup(cleanup_buffer, ring, unlink=True)
        return ring

    def _open_reader(
        self,
        writer: SharedBuffer,
        *,
        size: int,
        num_readers: int,
        reader: int,
    ) -> SharedBuffer:
        ring = SharedBuffer(
            name=writer.name,
            create=False,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_buffer, ring, unlink=False)
        return ring

    def test_default_inactive_readers_do_not_reduce_writer_capacity(self) -> None:
        writer = self._make_writer(size=32, num_readers=8)
        writer.update_write_pos(20)
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 32)

    def test_all_active_readers_caught_up_leave_full_writer_capacity(self) -> None:
        writer = self._make_writer(size=64, num_readers=8)
        readers = [self._open_reader(writer, size=64, num_readers=8, reader=index) for index in range(8)]

        writer.update_write_pos(50)
        for reader in readers:
            reader.update_reader_pos(50)
            reader.set_reader_active(True)

        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 64)

    def test_large_reader_count_supports_independent_consumption(self) -> None:
        writer = self._make_writer(size=256, num_readers=64)
        readers = [self._open_reader(writer, size=256, num_readers=64, reader=index) for index in range(64)]
        payload = b"many-readers"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        for index, reader in enumerate(readers):
            with self.subTest(reader=index):
                view = reader.expose_reader_mem_view(len(payload))
                self.assertEqual(view[2], len(payload))
                dst = bytearray(len(payload))
                reader.simple_read(view, dst)
                self.assertEqual(bytes(dst), payload)
                reader.inc_reader_pos(len(payload))
                release_mem_views(view[0], view[1])

    def test_many_active_readers_use_slowest_reader_for_backpressure(self) -> None:
        writer = self._make_writer(size=64, num_readers=32)
        readers = [self._open_reader(writer, size=64, num_readers=32, reader=index) for index in range(32)]

        writer.update_write_pos(50)
        for reader in readers:
            reader.update_reader_pos(50)
            reader.set_reader_active(True)
        readers[7].update_reader_pos(10)

        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 24)

    def test_force_rescan_observes_external_reader_progress(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        reader = self._open_reader(writer, size=32, num_readers=1, reader=0)

        writer.update_write_pos(20)
        reader.update_reader_pos(0)
        reader.set_reader_active(True)
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 12)

        reader.update_reader_pos(12)
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 24)

    def test_active_stalled_reader_limits_writer_capacity(self) -> None:
        writer = self._make_writer(size=16, num_readers=2)
        fast_reader = self._open_reader(writer, size=16, num_readers=2, reader=0)
        stalled_reader = self._open_reader(writer, size=16, num_readers=2, reader=1)

        writer.update_write_pos(12)
        fast_reader.update_reader_pos(12)
        stalled_reader.update_reader_pos(0)
        fast_reader.set_reader_active(True)
        stalled_reader.set_reader_active(True)

        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 4)
        view = writer.expose_writer_mem_view(8)
        self.assertEqual(view[2], 4)
        release_mem_views(view[0], view[1])

    def test_inactive_stalled_reader_does_not_limit_writer_capacity(self) -> None:
        writer = self._make_writer(size=16, num_readers=2)
        active_reader = self._open_reader(writer, size=16, num_readers=2, reader=0)
        inactive_reader = self._open_reader(writer, size=16, num_readers=2, reader=1)

        writer.update_write_pos(12)
        active_reader.update_reader_pos(12)
        inactive_reader.update_reader_pos(0)
        active_reader.set_reader_active(True)
        inactive_reader.set_reader_active(False)

        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 16)
        view = writer.expose_writer_mem_view(8)
        self.assertEqual(view[2], 8)
        release_mem_views(view[0], view[1])

    def test_reactivated_reader_applies_backpressure_from_its_current_position(self) -> None:
        writer = self._make_writer(size=16, num_readers=2)
        active_reader = self._open_reader(writer, size=16, num_readers=2, reader=0)
        rejoining_reader = self._open_reader(writer, size=16, num_readers=2, reader=1)

        writer.update_write_pos(12)
        active_reader.update_reader_pos(12)
        rejoining_reader.update_reader_pos(0)
        active_reader.set_reader_active(True)
        rejoining_reader.set_reader_active(False)
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 16)

        rejoining_reader.set_reader_active(True)
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 4)

    def test_exactly_one_buffer_of_unread_data_is_still_retained(self) -> None:
        writer = self._make_writer(size=16, num_readers=1)
        reader = self._open_reader(writer, size=16, num_readers=1, reader=0)
        payload = bytes(range(16))

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader.update_reader_pos(0)
        view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(view[2], len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(view, dst)
        reader.inc_reader_pos(len(payload))
        release_mem_views(view[0], view[1])

        self.assertEqual(bytes(dst), payload)

    def test_reader_that_falls_behind_is_resynced_to_current_writer(self) -> None:
        writer = self._make_writer(size=16, num_readers=1)
        reader = self._open_reader(writer, size=16, num_readers=1, reader=0)

        reader.update_reader_pos(0)
        writer.update_write_pos(33)

        stale_view = reader.expose_reader_mem_view(4)
        self.assertEqual(stale_view[2], 0)
        release_mem_views(stale_view[0], stale_view[1])

        payload = b"next"
        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        fresh_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(fresh_view[2], len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(fresh_view, dst)
        self.assertEqual(bytes(dst), payload)
        reader.inc_reader_pos(len(payload))
        release_mem_views(fresh_view[0], fresh_view[1])

    def test_jump_to_writer_discards_old_data_and_only_exposes_future_writes(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        reader = self._open_reader(writer, size=32, num_readers=1, reader=0)
        old_payload = b"legacy"
        new_payload = b"fresh-data"

        old_view = writer.expose_writer_mem_view(len(old_payload))
        writer.simple_write(old_view, old_payload)
        writer.inc_writer_pos(len(old_payload))
        release_mem_views(old_view[0], old_view[1])

        reader.update_reader_pos(0)
        reader.jump_to_writer()

        empty_view = reader.expose_reader_mem_view(len(old_payload))
        self.assertEqual(empty_view[2], 0)
        release_mem_views(empty_view[0], empty_view[1])

        new_view = writer.expose_writer_mem_view(len(new_payload))
        writer.simple_write(new_view, new_payload)
        writer.inc_writer_pos(len(new_payload))
        release_mem_views(new_view[0], new_view[1])

        reader_view = reader.expose_reader_mem_view(len(new_payload))
        self.assertEqual(reader_view[2], len(new_payload))
        dst = bytearray(len(new_payload))
        reader.simple_read(reader_view, dst)
        reader.inc_reader_pos(len(new_payload))
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), new_payload)

    def test_calculate_pressure_tracks_reader_progress(self) -> None:
        writer = self._make_writer(size=64, num_readers=1)
        reader = self._open_reader(writer, size=64, num_readers=1, reader=0)

        reader.update_reader_pos(0)
        reader.set_reader_active(True)
        writer.update_write_pos(48)
        self.assertEqual(writer.calculate_pressure(), 75)

        reader.update_reader_pos(32)
        self.assertEqual(writer.calculate_pressure(), 25)
