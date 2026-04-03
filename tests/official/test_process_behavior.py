from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
import unittest
import uuid

from tests.support import NO_READER, SharedBuffer, cleanup_buffer, make_name, release_mem_views


def _child_read_payload(
    name: str,
    size: int,
    num_readers: int,
    reader_index: int,
    nbytes: int,
    output_path: str,
) -> None:
    output = Path(output_path)
    ring = SharedBuffer(
        name=name,
        create=False,
        size=size,
        num_readers=num_readers,
        reader=reader_index,
    )
    try:
        ring.update_reader_pos(0)
        view = ring.expose_reader_mem_view(nbytes)
        dst = bytearray(view[2])
        ring.simple_read(view, dst)
        ring.inc_reader_pos(view[2])
        release_mem_views(view[0], view[1])
        output.write_bytes(bytes(dst))
    except BaseException as exc:  # pragma: no cover - propagated into parent assertion
        output.write_text(f"ERROR:{type(exc).__name__}:{exc}", encoding="utf-8")
    finally:
        cleanup_buffer(ring, unlink=False)


def _child_write_payload(
    name: str,
    size: int,
    num_readers: int,
    payload: bytes,
    output_path: str,
) -> None:
    output = Path(output_path)
    ring = SharedBuffer(
        name=name,
        create=False,
        size=size,
        num_readers=num_readers,
        reader=NO_READER,
    )
    try:
        ring.update_write_pos(0)
        view = ring.expose_writer_mem_view(len(payload))
        ring.simple_write(view, payload)
        ring.inc_writer_pos(len(payload))
        release_mem_views(view[0], view[1])
        output.write_text("ok", encoding="utf-8")
    except BaseException as exc:  # pragma: no cover - propagated into parent assertion
        output.write_text(f"ERROR:{type(exc).__name__}:{exc}", encoding="utf-8")
    finally:
        cleanup_buffer(ring, unlink=False)


def _child_read_prefix_and_leave_active(
    name: str,
    size: int,
    num_readers: int,
    reader_index: int,
    nbytes: int,
    output_path: str,
) -> None:
    output = Path(output_path)
    ring = SharedBuffer(
        name=name,
        create=False,
        size=size,
        num_readers=num_readers,
        reader=reader_index,
    )
    try:
        ring.update_reader_pos(0)
        ring.set_reader_active(True)
        view = ring.expose_reader_mem_view(nbytes)
        dst = bytearray(view[2])
        ring.simple_read(view, dst)
        ring.inc_reader_pos(view[2])
        release_mem_views(view[0], view[1])
        output.write_bytes(bytes(dst))
    except BaseException as exc:  # pragma: no cover - propagated into parent assertion
        output.write_text(f"ERROR:{type(exc).__name__}:{exc}", encoding="utf-8")
    finally:
        cleanup_buffer(ring, unlink=False)


class SharedBufferProcessTests(unittest.TestCase):
    def _make_writer(self, *, size: int = 64, num_readers: int = 2) -> SharedBuffer:
        ring = SharedBuffer(
            name=make_name("proc"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=NO_READER,
        )
        self.addCleanup(cleanup_buffer, ring, unlink=True)
        return ring

    def _make_output_path(self, stem: str, suffix: str) -> Path:
        path = Path.cwd() / f".{stem}_{uuid.uuid4().hex}{suffix}"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_multiple_readers_can_consume_the_same_written_payload_independently(self) -> None:
        writer = self._make_writer(size=64, num_readers=2)
        reader0 = SharedBuffer(name=writer.name, create=False, size=64, num_readers=2, reader=0)
        reader1 = SharedBuffer(name=writer.name, create=False, size=64, num_readers=2, reader=1)
        self.addCleanup(cleanup_buffer, reader0, unlink=False)
        self.addCleanup(cleanup_buffer, reader1, unlink=False)

        payload = b"multi-reader"
        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        for reader in (reader0, reader1):
            view = reader.expose_reader_mem_view(len(payload))
            dst = bytearray(len(payload))
            reader.simple_read(view, dst)
            self.assertEqual(bytes(dst), payload)
            reader.inc_reader_pos(len(payload))
            release_mem_views(view[0], view[1])

    def test_data_written_in_one_process_is_readable_in_another(self) -> None:
        writer = self._make_writer(size=64, num_readers=1)
        payload = b"cross-process-shared-memory"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        ctx = get_context("spawn")
        output_path = self._make_output_path("cross_process_result", ".bin")
        proc = ctx.Process(
            target=_child_read_payload,
            args=(writer.name, 64, 1, 0, len(payload), str(output_path)),
        )
        proc.start()
        proc.join(timeout=10)
        self.assertFalse(proc.is_alive(), "child process timed out")
        self.assertEqual(proc.exitcode, 0)
        self.assertTrue(output_path.exists(), "child process produced no output")

        result = output_path.read_bytes()
        self.assertFalse(result.startswith(b"ERROR:"), result.decode("utf-8", errors="replace"))
        self.assertEqual(result, payload)

    def test_data_written_in_child_process_is_readable_in_parent(self) -> None:
        writer = self._make_writer(size=64, num_readers=1)
        reader = SharedBuffer(name=writer.name, create=False, size=64, num_readers=1, reader=0)
        self.addCleanup(cleanup_buffer, reader, unlink=False)

        payload = b"child-to-parent"
        output_path = self._make_output_path("child_writer_result", ".txt")

        ctx = get_context("spawn")
        proc = ctx.Process(
            target=_child_write_payload,
            args=(writer.name, 64, 1, payload, str(output_path)),
        )
        proc.start()
        proc.join(timeout=10)
        self.assertFalse(proc.is_alive(), "child process timed out")
        self.assertEqual(proc.exitcode, 0)
        self.assertTrue(output_path.exists(), "child writer produced no output")

        child_status = output_path.read_text(encoding="utf-8")
        self.assertEqual(child_status, "ok")

        view = reader.expose_reader_mem_view(len(payload))
        self.assertEqual(view[2], len(payload))
        dst = bytearray(len(payload))
        reader.simple_read(view, dst)
        reader.inc_reader_pos(len(payload))
        release_mem_views(view[0], view[1])
        self.assertEqual(bytes(dst), payload)

    def test_multiple_child_process_readers_can_consume_same_payload_independently(self) -> None:
        writer = self._make_writer(size=64, num_readers=2)
        payload = b"independent-cross-process-readers"

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        ctx = get_context("spawn")
        output0 = self._make_output_path("child_reader0", ".bin")
        output1 = self._make_output_path("child_reader1", ".bin")

        proc0 = ctx.Process(
            target=_child_read_payload,
            args=(writer.name, 64, 2, 0, len(payload), str(output0)),
        )
        proc1 = ctx.Process(
            target=_child_read_payload,
            args=(writer.name, 64, 2, 1, len(payload), str(output1)),
        )
        proc0.start()
        proc1.start()
        proc0.join(timeout=10)
        proc1.join(timeout=10)

        self.assertFalse(proc0.is_alive(), "reader 0 child process timed out")
        self.assertFalse(proc1.is_alive(), "reader 1 child process timed out")
        self.assertEqual(proc0.exitcode, 0)
        self.assertEqual(proc1.exitcode, 0)
        self.assertEqual(output0.read_bytes(), payload)
        self.assertEqual(output1.read_bytes(), payload)

    def test_child_reader_progress_updates_parent_writer_capacity(self) -> None:
        writer = self._make_writer(size=32, num_readers=1)
        payload = bytes(range(20))

        writer_view = writer.expose_writer_mem_view(len(payload))
        writer.simple_write(writer_view, payload)
        writer.inc_writer_pos(len(payload))
        release_mem_views(writer_view[0], writer_view[1])

        ctx = get_context("spawn")
        output_path = self._make_output_path("child_reader_progress", ".bin")
        proc = ctx.Process(
            target=_child_read_prefix_and_leave_active,
            args=(writer.name, 32, 1, 0, 12, str(output_path)),
        )
        proc.start()
        proc.join(timeout=10)

        self.assertFalse(proc.is_alive(), "child process timed out")
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(output_path.read_bytes(), payload[:12])
        self.assertEqual(writer.compute_max_amount_writable(force_rescan=True), 24)
