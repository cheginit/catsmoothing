# Reviewing CatSmoothing

For every reviewer of a change here, a person or an agent. Read `AGENTS.md` first, the
`docs/design.md` entries the change touches, and for Rust code
`.agents/skills/rust-style/SKILL.md`.

## What failure looks like here

1. **The curve is wrong.** Resampled points stop being equally spaced by arc length,
    `distances` stops capping the gap, a vertex or its tangent moves under a continuity
    split, a ring stops closing exactly, or pre-smoothing moves a straight run (D-04,
    D-05, D-10, D-12, D-13).
1. **A panic reaches Python.** Input a caller controls reaches `panic!`, `unwrap`,
    `expect`, or an out-of-bounds index, and Python sees `PanicException` where it
    should see `ValueError` (D-07).
1. **The public surface moves silently.** A function, argument, default, or return shape
    in `catsmoothing/api.py` changes without a deprecation release (D-06).
1. **It gets slower.** The GIL stays held during Rust work, a batch copies per line, or
    quadrature work grows with input it should not (D-03, D-14). CodSpeed comments on
    every PR.
1. **A claim outruns its test.** A docstring, changelog entry, notebook sentence, or
    design-record entry states behavior that no test or recorded output pins (L-04).

## Out of scope

- `pixi.lock`, `Cargo.lock` bumps, notebook outputs in `docs/examples/*.ipynb`, and
    `docs/examples/data/v041.npz`.
- Anything the gate already enforces: ruff with `select = ["ALL"]`, pyright strict on
    `catsmoothing/`, `cargo fmt`, the clippy lints in `Cargo.toml`, mdformat, codespell.
- Accepted entries in `docs/design.md`. Challenge one only with evidence or a
    consequence the entry does not account for.

## How to report

- Each finding names the file and line, the input or state that triggers it, and what
    goes wrong. A guess from a name is no finding.
- Mark each finding verified (you read the code or ran a probe) or inferred.
- Tag each one important, nit, or pre-existing (present before this change). Report at
    most three nits.
- Report the class of a bug: when one site has it, list every site that does, so one fix
    round covers them.
- Open with the count, "N findings (I important, P pre-existing)", and say "No findings"
    when there are none.
- On a re-review, give a fixed or still-open verdict on each earlier finding first,
    review only the new diff, and repeat no finding already answered.

## Depth

Match the depth to what a missed bug would cost:

| Change touches                                                                                                                                     | Review                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Only docs, lock files, pre-commit revisions, or tests                                                                                              | CI only                                     |
| Other code in `catsmoothing/`, `src/`, or `scripts/`                                                                                               | A review of the diff                        |
| `src/splines.rs`, `src/presmooth.rs`, `src/continuity.rs`, `src/python.rs`, the public functions in `catsmoothing/api.py`, or `.github/workflows/` | A review of the whole project at the change |

After three review rounds on one change, stop and reconsider the approach before a
fourth.
