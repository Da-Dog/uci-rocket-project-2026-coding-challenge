from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.

    The intended contract is:
    - one writer and one or more readers
    - shared state visible across processes
    - bounded storage with reusable space after readers advance
    - reads and writes report how many bytes are actually available
    """

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        """
        Open or create the shared buffer.

        Expected behavior:
        - validate constructor arguments
        - allocate or attach to shared memory
        - initialize any shared metadata needed to track writer and reader state
        - set up local views/fields used by the rest of the methods

        Parameters:
        - `name`: shared memory block name
        - `create`: `True` for the creator/owner, `False` to attach to an existing block
        - `size`: logical payload capacity in bytes
        - `num_readers`: number of reader slots to support
        - `reader`: reader index for this instance, or `_NO_READER` for the writer instance
        - `cache_align` / `cache_size`: optional metadata-layout knobs; you may ignore
          them internally as long as validation and behavior remain correct
        """
        if cache_size & (cache_size - 1) != 0:
            raise ValueError(f"Cache size is not a power of two")

        self._active_offset = 1
        self._reader_offset = 1 + num_readers
        self._writer_offset = 1 + num_readers + (num_readers * 8)
        self.payload_offset = self._writer_offset + 8

        if create:
            super().__init__(name, create, size + self.payload_offset)
            self.num_readers = num_readers
            self.is_active = [False] * num_readers
            self.reader_pos = [0] * num_readers
            self.writer_pos = 0

            self.buf[0] = num_readers
            for i in range(num_readers):
                self.buf[self._active_offset + i] = 0
                r_idx = self._reader_offset + (i * 8)
                self.buf[r_idx: r_idx + 8] = (0).to_bytes(8, 'little')

            self.buf[self._writer_offset: self._writer_offset + 8] = (0).to_bytes(8, 'little')
        else:
            super().__init__(name, create)
            self.num_readers = self.buf[0]
            self.is_active = [bool(self.buf[self._active_offset + i]) for i in
                              range(self.num_readers)]
            self.reader_pos = [
                int.from_bytes(
                    self.buf[self._reader_offset + (i * 8): self._reader_offset + (i * 8) + 8],
                    'little')
                for i in range(self.num_readers)
            ]
            self.writer_pos = int.from_bytes(self.buf[self._writer_offset: self._writer_offset + 8],
                                             'little')

        if reader != self._NO_READER and (reader < 0 or reader >= self.num_readers):
            raise ValueError(f"Invalid reader index {reader}, total readers {self.num_readers}")

        self.reader = reader
        if reader != self._NO_READER:
            r_idx = self._reader_offset + (self.reader * 8)
            self.buf[r_idx: r_idx + 8] = self.reader_pos[self.reader].to_bytes(8, 'little')
        self.buffer_size = size

    def close(self) -> None:
        """
        Release local views and close this process's handle to the shared memory.

        This should not destroy the buffer for other attached processes.
        """
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """
        Enter the context manager.

        Reader instances are expected to mark themselves active while inside the
        context. Writer-only instances can simply return `self`.
        """
        if self.reader != self._NO_READER:
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        """
        Exit the context manager.

        Reader instances are expected to mark themselves inactive on exit, then
        close local resources.
        """
        if self.reader != self._NO_READER:
            self.set_reader_active(False)
        self.close()

    def _sync_memory(self):
        self.is_active = [bool(self.buf[self._active_offset + i]) for i in range(self.num_readers)]
        self.reader_pos = [
            int.from_bytes(self.buf[self._reader_offset + (i * 8) : self._reader_offset + (i * 8) + 8], 'little')
            for i in range(self.num_readers)
        ]

    def calculate_pressure(self) -> int:
        """
        Return current writer pressure as an integer percentage.

        Pressure is based on how much of the bounded storage is currently in use
        relative to the slowest active reader.
        """
        self._sync_memory()
        if not any(self.is_active):
            return 0
        min_reader_pos = min([self.reader_pos[i] for i in range(self.num_readers) if self.is_active[i]])
        used = self.writer_pos - min_reader_pos
        return int(used / self.buffer_size * 100)

    def int_to_pos(self, value: int) -> int:
        """
        Convert an absolute position counter into a position inside the bounded payload area.

        If your design does not use modulo arithmetic internally, you may still
        keep this helper as the mapping from logical positions to buffer offsets.
        """
        return value % self.buffer_size

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """
        Store this reader's absolute read position in shared state.

        This must fail clearly when called on a writer-only instance.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot update reader position")

        self.reader_pos[self.reader] = new_reader_pos
        r_idx = self._reader_offset + (self.reader * 8)
        self.buf[r_idx: r_idx + 8] = self.reader_pos[self.reader].to_bytes(8, 'little')

    def set_reader_active(self, active: bool) -> None:
        """
        Mark this reader as active or inactive in shared state.

        Active readers apply backpressure. Inactive readers should not reduce
        writer capacity.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot set reader active")

        self.is_active[self.reader] = active
        self.buf[self._active_offset + self.reader] = int(active)

    def is_reader_active(self) -> bool:
        """
        Return whether this reader is currently marked active.

        This must fail clearly when called on a writer-only instance.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot check reader active")

        self.is_active[self.reader] = bool(self.buf[self._active_offset + self.reader])
        return self.is_active[self.reader]

    def update_write_pos(self, new_writer_pos: int) -> None:
        """
        Store the writer's absolute write position in shared state.

        The write position is what makes newly written bytes visible to readers.
        """
        self.writer_pos = new_writer_pos
        self.buf[self._writer_offset: self._writer_offset + 8] = self.writer_pos.to_bytes(8,
                                                                                          'little')

    def inc_writer_pos(self, inc_amount: int) -> None:
        """
        Advance the writer's absolute position by `inc_amount` bytes.

        This is how a writer publishes bytes after copying them into the buffer.
        """
        self.writer_pos += inc_amount
        self.buf[self._writer_offset: self._writer_offset + 8] = self.writer_pos.to_bytes(8,
                                                                                          'little')

    def inc_reader_pos(self, inc_amount: int) -> None:
        """
        Advance this reader's absolute position by `inc_amount` bytes.

        This is how a reader consumes bytes after reading them.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot advance reader position")

        self.reader_pos[self.reader] += inc_amount
        r_idx = self._reader_offset + (self.reader * 8)
        self.buf[r_idx: r_idx + 8] = self.reader_pos[self.reader].to_bytes(8, 'little')

    def get_write_pos(self) -> int:
        """
        Return the current absolute writer position.

        Readers can use this to resynchronize or compute how much data is available.
        """
        self.writer_pos = int.from_bytes(self.buf[self._writer_offset : self._writer_offset + 8], 'little')
        return self.writer_pos

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """
        Return how many bytes the writer can safely expose right now.

        This should take active readers into account. `force_rescan=True` is used
        by the tests to ensure externally updated reader positions are observed.
        """
        if force_rescan:
            self._sync_memory()

        if not any(self.is_active):
            return self.buffer_size
        min_reader_pos = min([self.reader_pos[i] for i in range(self.num_readers) if self.is_active[i]])
        return self.buffer_size - (self.writer_pos - min_reader_pos)

    def jump_to_writer(self) -> None:
        """
        Move this reader directly to the current writer position.

        Use this when a reader has fallen too far behind and old unread data is
        no longer retained.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot jump to writer position")

        self.writer_pos = int.from_bytes(self.buf[self._writer_offset: self._writer_offset + 8],
                                         'little')
        self.reader_pos[self.reader] = self.writer_pos
        r_idx = self._reader_offset + (self.reader * 8)
        self.buf[r_idx: r_idx + 8] = self.reader_pos[self.reader].to_bytes(8, 'little')

    def expose_writer_mem_view(self, size: int) -> RingView:
        """
        Return a writable view tuple for up to `size` bytes.

        The return shape is:
        - `mv1`: first writable view
        - `mv2`: optional second writable view if the exposed region is split
        - `actual_size`: how many bytes are actually writable right now
        - `split`: whether the writable region is split across two views

        If less than `size` bytes are currently writable, clamp to the amount
        available rather than raising.
        """
        amount = min(size, self.compute_max_amount_writable())
        start = self.payload_offset + self.int_to_pos(self.writer_pos)

        if self.int_to_pos(self.writer_pos) + amount > self.buffer_size:
            mv1 = memoryview(self.buf)[start: self.payload_offset + self.buffer_size]
            mv2 = memoryview(self.buf)[self.payload_offset: self.payload_offset + (
                        amount - (self.buffer_size - self.int_to_pos(self.writer_pos)))]
            return mv1, mv2, amount, True
        else:
            return memoryview(self.buf)[start: start + amount], None, amount, False

    def expose_reader_mem_view(self, size: int) -> RingView:
        """
        Return a readable view tuple for up to `size` bytes.

        The shape matches `expose_writer_mem_view()`. If less than `size` bytes
        are currently readable, clamp to the amount available rather than raising.
        """
        if self.reader == self._NO_READER:
            raise RuntimeError("Writer instance cannot expose reader view")

        readable_bytes = self.get_write_pos() - self.reader_pos[self.reader]

        if readable_bytes > self.buffer_size:
            readable_bytes = 0
            self.update_reader_pos(self.get_write_pos())


        amount = min(size, readable_bytes)
        start = self.payload_offset + self.int_to_pos(self.reader_pos[self.reader])

        if self.int_to_pos(self.reader_pos[self.reader]) + amount > self.buffer_size:
            mv1 = memoryview(self.buf)[start : self.payload_offset + self.buffer_size]
            mv2 = memoryview(self.buf)[self.payload_offset : self.payload_offset + (amount - (self.buffer_size - self.int_to_pos(self.reader_pos[self.reader])))]
            return mv1, mv2, amount, True
        else:
            return memoryview(self.buf)[start : start + amount], None, amount, False

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """
        Copy bytes from `src` into the exposed writer view(s).

        If `src` is larger than the destination region, copy only the prefix that fits.
        This helper should not publish data by itself; publishing happens when the
        writer position is advanced.
        """
        if not hasattr(src, "__getitem__") and not hasattr(src, "__len__"):
            raise ValueError("Source must support item access and length")

        src_size = len(src)

        mv1, mv2, _, split = writer_mem_view

        copy1 = min(src_size, len(mv1))
        mv1[:copy1] = src[:copy1]

        if split and src_size > copy1:
            copy2 = min(src_size - copy1, len(mv2))
            mv2[:copy2] = src[copy1: copy1 + copy2]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """
        Copy bytes from the exposed reader view(s) into `dst`.

        If `dst` is smaller than the readable region, copy only the prefix that fits.
        This helper should not consume data by itself; consumption happens when the
        reader position is advanced.
        """
        if not hasattr(dst, "__setitem__") and not hasattr(dst, "__len__"):
            raise ValueError("Destination must support item assignment and length")

        dst_size = len(dst)

        mv1, mv2, _, split = reader_mem_view

        copy1 = min(dst_size, len(mv1))
        dst[:copy1] = mv1[:copy1]

        if split and dst_size > copy1:
            copy2 = min(dst_size - copy1, len(mv2))
            dst[copy1: copy1 + copy2] = mv2[:copy2]

    def write_array(self, arr: np.ndarray) -> int:
        """
        Write a NumPy array's raw bytes into the shared buffer.

        Return the number of bytes written. If the full array does not fit, the
        contract used by the tests expects this method to return `0`.
        """
        if arr.nbytes > self.compute_max_amount_writable():
            return 0

        writer_view, writer_view2, _, _ = self.expose_writer_mem_view(arr.nbytes)
        if writer_view2 is not None:
            first_part_size = len(writer_view)
            writer_view[:first_part_size] = arr.tobytes()[:first_part_size]
            writer_view2[:arr.nbytes - first_part_size] = arr.tobytes()[first_part_size:]
        else:
            writer_view[:arr.nbytes] = arr.tobytes()
        self.inc_writer_pos(arr.nbytes)

        return arr.nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """
        Read `nbytes` from the shared buffer and interpret them as `dtype`.

        Return a NumPy array view/copy of the requested bytes when enough data is
        available. If there are not enough readable bytes, return an empty array
        with the requested dtype.
        """
        if nbytes > self.get_write_pos() - self.reader_pos[self.reader]:
            return np.array([], dtype=dtype)

        reader_view, reader_view2, _, _ = self.expose_reader_mem_view(nbytes)
        if reader_view2 is not None:
            data = bytes(reader_view) + bytes(reader_view2)
        else:
            data = bytes(reader_view)
        self.inc_reader_pos(nbytes)

        return np.frombuffer(data, dtype=dtype)
