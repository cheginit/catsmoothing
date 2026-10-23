Closes #

## What changed

<!-- What the change does and why, in a few sentences. Name the D-, L-, or Q- entries in
docs/design.md it touches. -->

## How it was checked

<!-- Tick what ran. Add any manual check, with what it showed. -->

- [ ] `pixi r lint`
- [ ] `pixi r typecheck`
- [ ] `pixi r -e test313 test`
- [ ] `pixi r -e test315 test`
- [ ] `cargo fmt --check`,
    `cargo clippy --features python --all-targets -- -D warnings`, and `cargo test`
    (needed when `src/`, `tests/`, or `Cargo.toml` changed)
- [ ] `pixi r build-docs` (needed when `docs/`, `README.md`, or `mkdocs.yml` changed)

## Risk and revert

<!-- What this could break and for whom, the review depth REVIEW.md asks for, and how to
undo it: a revert of this PR, or, after a release, a fix release. -->

## Design record

- [ ] `docs/design.md` is updated, or this change makes, reverses, or answers nothing
    there
