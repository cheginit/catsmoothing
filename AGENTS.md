# Contributing

Conventions for human contributors and AI coding agents working in this repository. The
docs site renders this file as its contributing guide.

## Project

CatSmoothing smooths Shapely geometries with Catmull-Rom splines and computes tangent
angles at vertices. The spline math is a Rust crate in `src/`, exposed to Python through
PyO3 bindings in `src/python.rs` as `catsmoothing._catsmoothing`; `catsmoothing/api.py`
wraps them with the public functions `smooth_linestrings`, `smooth_polygon`,
`linestrings_tangent_angles`, and the `CatmullRom` class.

## Design record

`docs/design.md` holds every design decision (`D-`), the lessons from past mistakes
(`L-`), and the open questions (`Q-`), each with its rationale.

- Before designing a component or proposing an alternative to a rule below, search it.
    Do not reopen an accepted decision without evidence it does not already account for.
- When a change makes or reverses a decision, answers a question, or exposes a mistake
    worth not repeating, update `docs/design.md` in the same change, following the rules
    at its top.

## Design rules

One-line summaries of the decisions any code change can break. The IDs point into
`docs/design.md`.

- Smoothing functions fit centripetal splines (`alpha=0.5`) (D-02).
- Resampled points are equally spaced by arc length, and `distances` caps the gap in
    coordinate units (D-04, D-05).
- Caller input errors return `SplineError` in Rust and raise `ValueError` in Python;
    Rust never panics on input a caller controls (D-07).
- Repeated consecutive vertices are dropped when the grid is generated, and vertex
    arrays must have two columns (D-08, D-09).
- Gaussian pre-smoothing point-reflects open ends and wraps rings; Taubin and Gaussian
    smoothing do not combine (D-10, D-11).
- Continuity splits keep every vertex and its tangent (D-12).
- The last point of a curve is its last vertex, so rings close exactly (D-13).
- Batch functions release the GIL, pass coordinates in one stacked array, and use
    Rayon's global pool, running serially when it has one thread (D-14).
- A public argument is removed only after a release in which passing it raises a
    `DeprecationWarning` (D-06).
- Rust follows the Google Rust style guide: the clippy lints in `Cargo.toml` plus
    `.agents/skills/rust-style/SKILL.md` (D-16).

## Commands

Install [pixi](https://pixi.sh) and run everything through it; each environment is
created on first use. Never call bare `pytest`, `python`, or `pyright`. Rust checks run
through `cargo`.

| Task                               | Command                                                       |
| ---------------------------------- | ------------------------------------------------------------- |
| Lint (pre-commit)                  | `pixi r lint`                                                 |
| Dependency audit (Python and Rust) | `pixi r audit`                                                |
| Type check                         | `pixi r typecheck`                                            |
| Test, Python 3.13                  | `pixi r -e test313 test`                                      |
| Test, Python 3.15                  | `pixi r -e test315 test`                                      |
| Coverage report / HTML             | `pixi r -e test315 report` / `html`                           |
| Rust format check                  | `cargo fmt --check`                                           |
| Rust lint                          | `cargo clippy --features python --all-targets -- -D warnings` |
| Rust tests                         | `cargo test`                                                  |
| Benchmarks                         | `pixi r benchmark`                                            |
| Run Python                         | `pixi r -e dev python ...`                                    |
| Spell check (writes fixes)         | `pixi r spell`                                                |
| Changelog preview                  | `pixi r changelog`                                            |
| Docs, strict build                 | `pixi r docs`                                                 |
| Docs, live preview                 | `pixi r docs-serve`                                           |
| Notebooks, sync and run            | `pixi r nb-run`                                               |
| Notebooks, pair a new one          | `pixi r nb-pair`                                              |

Environments: `dev`, `test313`, `test315`, `benchmark`, `typecheck`, `lint`, `docs`.

A change is done when lint, typecheck, and the tests on both test313 and test315 pass,
plus the three Rust checks when it touches `src/`, `tests/`, or `Cargo.toml`, and
`pixi r docs` when it touches `docs/`, `README.md`, `CHANGELOG.md`, or `zensical.toml`.
`.github/workflows/test.yml` runs the tests on Linux, macOS, and Windows with the Rust,
typecheck, lint, and audit jobs; `docs.yml` builds the site on every PR and deploys
`main` to GitHub Pages; `release.yml` builds the wheels on every PR; and `codspeed.yml`
runs the benchmarks.

## Code style

- Rust: `cargo fmt` defaults and the `[lints.clippy]` table in `Cargo.toml`. Before
    writing or reviewing Rust, read `.agents/skills/rust-style/SKILL.md`, the Google
    style rules no lint checks.
- Python >= 3.13, with `from __future__ import annotations` in every module.
- ruff with 100-character lines and `select = ["ALL"]`; pyright in strict mode on
    `catsmoothing/`. Both are configured in `pyproject.toml`.
- NumPy-convention docstrings.
- Rust in `src/` with integration tests in `tests/`; Python in `catsmoothing/` with
    tests in `catsmoothing/tests/`.

## Gotchas

- `test`, `typecheck`, and `benchmark` first run `preinstall` (`maturin develop --uv`),
    which builds the optimized `release` profile that `[tool.maturin]` names (L-03).
- The `test` task appends coverage across runs. Delete `.coverage` before trusting a
    number.
- pyright checks only `catsmoothing/` and skips its tests.
- `pixi r lint` runs vulture at 80% confidence, which catches only unused imports,
    unused arguments, and unreachable code. pre-commit's `--all-files` sees only tracked
    files, so `pixi r lint` then runs `scripts/lint_untracked.py` on untracked,
    non-ignored files.
- mdformat rewraps markdown to 88 columns. Keep `mdformat-frontmatter` in its hook, or
    it mangles the frontmatter of skills and rules (L-02).
- Each example in `docs/examples/` is a notebook paired with a jupytext `py:percent`
    script; edit either and commit both, the notebook with its outputs. `pixi r nb-run`
    syncs them and executes the notebooks against a release build. ssort skips this
    folder, since it can move code between cells (L-04).
- `pixi r docs` converts each notebook into a Markdown page beside it without running
    it, since zensical renders no notebooks. The generated `.md` files and `*_files/`
    folders are git-ignored, apart from `index.md`. The site also publishes the `.ipynb`
    and `.py` files, so a link to `other.ipynb` downloads the notebook; link `other.md`
    to reach its page.
- Each notebook gets a card in `docs/examples/index.md` with a thumbnail in
    `docs/examples/images/`. The strict build fails if a card links to a notebook that
    does not exist.
- `docs/index.md`, `docs/changelog.md`, `docs/contributing.md`, and `docs/license.md`
    only include `README.md`, `CHANGELOG.md`, this file, and `LICENSE`. Edit the
    sources.
- `docs/examples/whats_new.ipynb` reads `data/v041.npz`. Regenerate it only with the
    command in `docs/examples/data/make_v041_data.py`, then rerun the notebook (D-21).
- Links in the included files must be absolute URLs: a relative link resolves against
    the site, where its target does not exist, and the strict build does not catch it.
- `release.yml` comes from `maturin generate-ci github`. After regenerating it, restore
    the `test` and `release-notes` jobs, `release-notes` in the release job's `needs`,
    pushes to `main` only, trusted publishing without a token, the `interpreter` entries
    of the 32-bit Linux targets, the commit-SHA pins, and `persist-credentials: false`
    on each checkout (D-23).
- The version lives in `pyproject.toml` and `Cargo.toml`; bump both in the release PR.
- Skills live in `.agents/skills/`. `.claude/skills` is a symlink to it for Claude Code,
    and `.claude/rules/rust-style.md` points Claude Code to the Rust skill when it opens
    a `.rs` file. `CLAUDE.md` is git-ignored because Claude Code reads `AGENTS.md` only
    when no `CLAUDE.md` exists (D-17).
- The ruleset on `main` names its required checks by job: `Test (test313, <os>)` and
    `Test (test315, <os>)` for `ubuntu-latest`, `macos-latest`, and `windows-latest`,
    plus `Rust`, `Typecheck`, and `Lint` from `test.yml`, and `Docs` from `docs.yml`.
    Renaming a job or a matrix entry leaves the old name unreported, so the renaming PR
    and every PR after it wait forever. Swap the name under
    `Settings > Rules > Rulesets` as that PR merges.

## Git

- Conventional Commits: `<type>[optional scope]: <description>`, with type one of
    `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
- Subject under 72 characters, imperative mood.
- Every change reaches `main` through a PR from its own branch, named
    `<type>/<short-description>` (`fix/ring-closure`). A ruleset blocks direct pushes,
    force pushes, and deletion of `main` (D-22).
- PRs fill in `.github/PULL_REQUEST_TEMPLATE.md` and name the issue they finish with
    `Closes #N`.
- `CHANGELOG.md` is hand-written through 0.5.0, whose release PR replaces the
    `20XX-XX-XX` placeholder in its heading with the date. Later releases get their
    notes from git-cliff and the tag message; after each one, `pixi r changelog-update`
    prepends them to `CHANGELOG.md` (D-19).
- `pixi r release vX.Y.Z [notes-file]` tags and pushes that tag, which publishes to
    PyPI. Run it only when a maintainer asks.

## Writing style

These rules cover code comments, docstrings, commit messages, PR descriptions, issues,
and docs.

- American spelling. Fix British spellings on sight.
- Active voice, present tense, concise.
- Use an em or en dash only when no comma, colon, semicolon, parentheses, or new
    sentence works.
- Be concrete. A number, name, or mechanism beats an abstraction: "handles bad input"
    becomes "raises `ValueError` for vertex arrays without two columns".
- No contrastive framing. Say what a thing is and stop, without a negated counterpart
    added for rhythm or emphasis: "the launch is checked rather than trusted" reads "the
    launch is checked". The ban covers "rather than", "not X but Y", "instead of", and
    "X, not Y" in any position. A real either/or, where both options are live and the
    reader needs to know which was picked, stays.
- Keep terms consistent. Once something has a name, reuse it, because a reader takes a
    new word for a new thing. If the text calls it a `grid`, later sentences do not
    switch to "knots" or "parameters".
- Cut filler ("it's worth noting", "in order to", "when it comes to"), puffery ("plays a
    vital role", "stands as a testament"), and closing summaries that restate what came
    before.
- Banned words: delve, foster, empower, streamline, facilitate, cutting-edge, seamless,
    multifaceted, nuanced, intricate, meticulous, paramount, transformative, elevate,
    embark, supercharge.

Comments and commits:

- Comment only what the code does not say: a decision a reader would not guess, or
    behavior a reader would not see from reading it.
- Keep a comment or commit body to a line or two. Cut the rationale to the sentence that
    stops the next reader from undoing the change.
- Comments describe the code as it stands. History ("used to", "the old code", the bug
    that prompted the change) goes in the commit message.

Reviews and reports:

- A code review follows `REVIEW.md`: what failure looks like here, what to skip, how to
    report, and how deep to go.
- Say which claims you checked against the source and which you infer.
- Reserve prose for what changes the conclusion. Minor issues go in a numbered list at
    the end.
