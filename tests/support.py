from __future__ import annotations

import gc
import importlib
import os
import uuid


SUBMISSION_MODULE_NAME = os.environ.get("SHARED_BUFFER_MODULE", "solution")
submission_module = importlib.import_module(SUBMISSION_MODULE_NAME)
SharedBuffer = getattr(submission_module, "SharedBuffer")
NO_READER = getattr(SharedBuffer, "_NO_READER", -1)


def make_name(prefix: str = "buf") -> str:
    return f"{prefix}{uuid.uuid4().hex[:20]}"


def reader_slot(reader: int) -> int:
    return 6 + (reader * 3)


def release_mem_views(*views: memoryview | None) -> None:
    for view in views:
        if view is None:
            continue
        try:
            view.release()
        except Exception:
            pass


def drop_local_views(buffer_obj) -> None:
    if buffer_obj is None:
        return
    try:
        buffer_obj.buffer.release()
    except Exception:
        pass
    try:
        del buffer_obj.buffer
    except Exception:
        pass
    try:
        buffer_obj.header = None
    except Exception:
        pass
    try:
        del buffer_obj.header
    except Exception:
        pass


def cleanup_buffer(buffer_obj, *, unlink: bool = True) -> None:
    if buffer_obj is None:
        return
    drop_local_views(buffer_obj)
    gc.collect()
    try:
        buffer_obj.close()
    except Exception:
        pass
    if unlink:
        try:
            buffer_obj.unlink()
        except FileNotFoundError:
            pass


def set_reader_state(buffer_obj, reader: int, *, pos: int | None = None, alive: int | None = None) -> None:
    slot = reader_slot(reader)
    if pos is not None:
        buffer_obj.header[slot] = pos
    if alive is not None:
        buffer_obj.header[slot + 1] = alive
    if hasattr(buffer_obj, "_reader_positions_dirty"):
        buffer_obj._reader_positions_dirty = True


def mark_reader_alive(buffer_obj, reader_index: int | None = None) -> None:
    if reader_index is None:
        reader_index = buffer_obj.reader
    set_reader_state(buffer_obj, reader_index, alive=1)
