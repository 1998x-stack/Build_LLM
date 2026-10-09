# Contributing

## Development setup

    python -m pip install -e ".[dev]"

Before opening a pull request:

    make check

## Design rules

1. Correctness changes require regression tests.
2. Canonical model code belongs under src/build_llm.
3. Chapter files should explain or wrap canonical behavior, not maintain another production copy.
4. Model modules must not store a device string; placement is controlled by model.to(device).
5. Optimized paths must remain numerically comparable to a readable reference path.
6. Keep training, generation and evaluation separated from model forward logic.

## Pull request size

Prefer small, independently verifiable changes. Large architecture migrations should be split so each commit keeps tests runnable.
