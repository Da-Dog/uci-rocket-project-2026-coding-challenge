from __future__ import annotations

from multiprocessing import shared_memory
import unittest

from tests.support import NO_READER, SharedBuffer, cleanup_buffer, make_name, release_mem_views


class SharedBufferContractTests(unittest.TestCase):
    def _make_writer(self, *, size: int = 64, num_readers: int = 1, **kwargs) -> SharedBuffer:
        ring = SharedBuffer(
            name=make_name("contract"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=NO_READER,
            **kwargs,
        )
        self.addCleanup(cleanup_buffer, ring, unlink=True)
        return ring

    def _open_reader(
        self,
        name: str,
        *,
        size: int,
        num_readers: int,
        reader: int = 0,
        **kwargs,
    ) -> SharedBuffer:
        ring = SharedBuffer(
            name=name,
            create=False,
            size=size,
            num_readers=num_readers,
            reader=reader,
            **kwargs,
        )
        self.addCleanup(cleanup_buffer, ring, unlink=False)
        return ring

    def test_shared_buffer_inherits_shared_memory(self) -> None:
        ring = self._make_writer()
        self.assertIsInstance(ring, shared_memory.SharedMemory)

    def test_constructor_records_size_and_reader_count(self) -> None:
        ring = self._make_writer(size=128, num_readers=3)
        self.assertEqual(ring.buffer_size, 128)
        self.assertEqual(ring.num_readers, 3)

    def test_constructor_rejects_reader_index_out_of_range(self) -> None:
        writer = self._make_writer(size=32, num_readers=2)
        for bad_reader in (-2, 2, 100):
            with self.subTest(reader=bad_reader):
                with self.assertRaises(ValueError):
                    SharedBuffer(
                        name=writer.name,
                        create=False,
                        size=32,
                        num_readers=2,
                        reader=bad_reader,
                    )

    def test_constructor_rejects_non_power_of_two_cache_size_when_aligned(self) -> None:
        with self.assertRaises(ValueError):
            self._make_writer(size=64, num_readers=1, cache_align=True, cache_size=48)

    def test_writer_only_instances_reject_reader_methods(self) -> None:
        ring = self._make_writer()
        for method_name, args in (
            ("update_reader_pos", (1,)),
            ("inc_reader_pos", (1,)),
            ("expose_reader_mem_view", (1,)),
            ("jump_to_writer", ()),
            ("set_reader_active", (True,)),
            ("is_reader_active", ()),
        ):
            with self.subTest(method=method_name):
                with self.assertRaises(RuntimeError):
                    getattr(ring, method_name)(*args)

    def test_reader_starts_inactive(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        reader = self._open_reader(writer.name, size=32, num_readers=1, reader=0)
        self.assertFalse(reader.is_reader_active())

    def test_reader_context_manager_marks_active_only_inside_context(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        reader = self._open_reader(writer.name, size=32, num_readers=1, reader=0)

        self.assertFalse(reader.is_reader_active())
        with reader as active_reader:
            self.assertTrue(active_reader.is_reader_active())

        probe = self._open_reader(writer.name, size=32, num_readers=1, reader=0)
        self.assertFalse(probe.is_reader_active())

    def test_new_buffer_starts_empty_for_readers(self) -> None:
        writer = self._make_writer(size=16, num_readers=1)
        reader = self._open_reader(writer.name, size=16, num_readers=1, reader=0)

        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 16)
        reader_view = reader.expose_reader_mem_view(4)
        self.assertEqual(reader_view[2], 0)
        release_mem_views(reader_view[0], reader_view[1])

    def test_calculate_pressure_tracks_active_readers_only(self) -> None:
        writer = self._make_writer(size=64, num_readers=2)
        active_reader = self._open_reader(writer.name, size=64, num_readers=2, reader=0)
        inactive_reader = self._open_reader(writer.name, size=64, num_readers=2, reader=1)

        active_reader.update_reader_pos(20)
        active_reader.set_reader_active(True)
        inactive_reader.update_reader_pos(0)
        inactive_reader.set_reader_active(False)
        writer.update_write_pos(52)

        self.assertEqual(writer.calculate_pressure(), 50)

    def test_jump_to_writer_discards_currently_unread_bytes(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        reader = self._open_reader(writer.name, size=32, num_readers=1, reader=0)
        payload = b"discard-me"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        reader.update_reader_pos(0)
        reader.jump_to_writer()

        reader_view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(reader_view[2], 0)
        release_mem_views(reader_view[0], reader_view[1])
