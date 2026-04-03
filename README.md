# UCI Rocket Project Liquids

Spring 2026 recruitment coding challenge: implement a cross-process shared-memory buffer in Python.

## Important Links

- Fork this repository and complete the challenge in your own fork.
- Submission Form: https://forms.gle/KJM6GWKq28sT7MgN9
- Questions / Help Discord: https://discord.gg/3wNUqmGT5p

## What Applicants Work On

Your task is to replace the starter in `solution.py` with a correct implementation of:

- `SharedBuffer`

The reference contract is defined by the official tests in `tests/official/`.
A circular implementation is one valid solution, but it is not the only valid solution.
This challenge assumes a single writer and one or more readers. Multi-writer support is not required and won't be tested.

## Repo Layout

- `solution.py`: starter submission file. It is intentionally incomplete.
- `tests/official/`: official grading tests.
- `tests/applicant/applicant_test_template.py`: example file you can copy to write your own tests.
- `score.py`: simple local scoring script.

## Getting Started

1. Create a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the official score against your submission:

```bash
python score.py
```

## Applicant Workflow

- Fork this repository and do your work in your own fork.
- Edit `solution.py`.
- Keep the existing file names and public bindings unchanged.
- Put your entire implementation in `solution.py`.
- Leave `tests/official/` alone when you want a real score.
- Copy `tests/applicant/applicant_test_template.py` to something like `tests/applicant/test_my_solution.py`.
- Add your own cases there as you build.

To run your own tests too:

```bash
python score.py --include-applicant-tests
```

When you are finished, submit your work here:

https://forms.gle/KJM6GWKq28sT7MgN9

## Challenge Contract

Your implementation should preserve the public API used by the tests, including:

- backing storage built on `multiprocessing.shared_memory`
- a single-writer, multi-reader shared buffer class
- cross-process visibility of written data
- buffered reads and writes that report the actual available amount
- simple copy-helper integrity
- multiple independent readers
- context-manager behavior for reader lifetimes

The official tests intentionally avoid enforcing one internal algorithm.
A circular implementation is acceptable, but not required.

The official suite is behavior-first. It checks:

- constructor validation and `SharedMemory` inheritance
- single-writer / multi-reader semantics
- write publication and read consumption semantics
- byte and numpy-array integrity across normal use and after space is reused
- clamped partial exposure when less data or space is currently available
- multiple independent readers in one process and across processes
- stalled-reader/backpressure behavior, including inactive readers and readers that fall behind retention

The official tests do not require a specific circular layout. They only require bounded storage, ordered delivery, and the ability to reuse space after readers advance.

## Stalled Readers

The official contract treats stalled readers like this:

- an active stalled reader still counts as a consumer and can reduce the amount a writer can safely expose
- an inactive reader should not apply backpressure to active producers
- if a reader falls behind beyond the data retention window, it may be resynchronized to the current writer position and only see new data from that point forward

## Benchmark

A rough throughput benchmark is included for local experimentation:

```bash
python benchmarks/throughput_benchmark.py --seconds 2 --chunk-size 65536 --verify
```

It is not part of grading. It is only there to estimate how much data your implementation can move on your machine.
