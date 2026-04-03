from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution import SharedBuffer  # noqa: E402
from tests.support import NO_READER, cleanup_buffer, release_mem_views  # noqa: E402


def run_benchmark(*, buffer_size: int, chunk_size: int, seconds: float, verify: bool) -> dict[str, float]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be > 0")

    name = f"bench{uuid.uuid4().hex[:12]}"
    writer = SharedBuffer(
        name=name,
        create=True,
        size=buffer_size,
        num_readers=1,
        reader=NO_READER,
    )
    reader = SharedBuffer(
        name=name,
        create=False,
        size=buffer_size,
        num_readers=1,
        reader=0,
    )
    payload = bytes((i % 251) for i in range(chunk_size))
    scratch = bytearray(chunk_size)

    try:
        reader.update_reader_pos(0)
        reader.set_reader_active(True)
        writer.update_write_pos(0)

        deadline = time.perf_counter() + seconds
        iterations = 0
        bytes_transferred = 0

        while time.perf_counter() < deadline:
            writer_view = writer.expose_writer_mem_view(chunk_size)
            n = writer_view[2]
            if n == 0:
                release_mem_views(writer_view[0], writer_view[1])
                continue

            writer.simple_write(writer_view, payload)
            writer.inc_writer_pos(n)
            release_mem_views(writer_view[0], writer_view[1])

            reader_view = reader.expose_reader_mem_view(n)
            reader.simple_read(reader_view, scratch)
            if verify and bytes(scratch[:n]) != payload[:n]:
                raise RuntimeError("benchmark data verification failed")
            reader.inc_reader_pos(n)
            release_mem_views(reader_view[0], reader_view[1])

            bytes_transferred += n
            iterations += 1

        mb_per_s = (bytes_transferred / seconds) / (1024 * 1024)
        return {
            "seconds": seconds,
            "iterations": float(iterations),
            "bytes_transferred": float(bytes_transferred),
            "mb_per_s": mb_per_s,
        }
    finally:
        cleanup_buffer(reader, unlink=False)
        cleanup_buffer(writer, unlink=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a rough shared-buffer throughput benchmark.")
    parser.add_argument("--buffer-size", type=int, default=1 << 20, help="Logical shared buffer size in bytes.")
    parser.add_argument("--chunk-size", type=int, default=64 << 10, help="Bytes transferred per iteration.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Benchmark duration in seconds.")
    parser.add_argument("--verify", action="store_true", help="Verify payload integrity on every iteration.")
    args = parser.parse_args()

    results = run_benchmark(
        buffer_size=args.buffer_size,
        chunk_size=args.chunk_size,
        seconds=args.seconds,
        verify=args.verify,
    )

    print("Shared Buffer Throughput Benchmark")
    print(f"buffer_size_bytes: {args.buffer_size}")
    print(f"chunk_size_bytes: {args.chunk_size}")
    print(f"seconds: {results['seconds']:.2f}")
    print(f"iterations: {int(results['iterations'])}")
    print(f"bytes_transferred: {int(results['bytes_transferred'])}")
    print(f"throughput_mib_per_s: {results['mb_per_s']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
