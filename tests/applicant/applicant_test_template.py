from __future__ import annotations

import unittest

from tests.support import SharedBuffer, cleanup_buffer, make_name, release_mem_views


class ApplicantSharedBufferTests(unittest.TestCase):
    def _make_buffer(self, *, size: int = 16, num_readers: int = 1, reader: int = 0):
        buffer_obj = SharedBuffer(
            name=make_name("applicant"),
            create=True,
            size=size,
            num_readers=num_readers,
            reader=reader,
        )
        self.addCleanup(cleanup_buffer, buffer_obj)
        return buffer_obj

    def test_example_roundtrip(self):
        buffer_obj = self._make_buffer(size=16)
        buffer_obj.update_reader_pos(0)
        buffer_obj.update_write_pos(0)

        writer_view = buffer_obj.expose_writer_mem_view(4)
        buffer_obj.simple_write(writer_view, b"UCI!")
        buffer_obj.inc_writer_pos(writer_view[2])
        release_mem_views(writer_view[0], writer_view[1])

        reader_view = buffer_obj.expose_reader_mem_view(4)
        dst = bytearray(4)
        buffer_obj.simple_read(reader_view, dst)
        release_mem_views(reader_view[0], reader_view[1])

        self.assertEqual(bytes(dst), b"UCI!")
