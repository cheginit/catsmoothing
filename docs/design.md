# Design decisions

This page records the design decisions behind CatSmoothing, the mistakes that shaped
them, and the questions still open. Contributors and coding agents check it before
proposing a design change and update it in the same change that makes or reverses a
decision.

## Maintaining this page

- Search here before deciding. If a decision (`D-`) or lesson (`L-`) covers the case,
    follow it, or argue against it with evidence it does not already account for.
- A new decision gets a `D-` entry with its date. A mistake worth not repeating gets an
    `L-` entry that names the check catching it.
- To reverse a decision, set the old entry's status to `superseded by D-NN` and keep its
    text. The record of what was tried is the point of this page.
- An answered question (`Q-`) becomes a `D-` entry, and the question's status becomes
    `answered by D-NN`.
- Some decisions have a one-line summary under "Design rules" in `AGENTS.md`. Update
    that line in the same change.
- Never renumber. New entries take the next free number.
- D-01 and D-02 predate this page and carry no dates.

Each decision lists its status, the decision, why, and what was rejected or its limits.

## Scope

CatSmoothing smooths Shapely `LineString`, `LinearRing`, `Polygon`, and `MultiPolygon`
geometries with Catmull-Rom splines and computes tangent angles at vertices. A Rust
crate in `src/` holds the spline math, the PyO3 bindings in `src/python.rs` expose it as
`catsmoothing._catsmoothing`, and `catsmoothing/api.py` wraps the bindings with
Shapely-aware functions. NumPy and Shapely are the only runtime dependencies.

## Decisions

### D-01: Rust core behind PyO3 bindings

- **Status:** accepted.
- **Decision:** The spline math is a Rust crate that maturin builds into
    `catsmoothing._catsmoothing`. The crate is not on crates.io; only the bindings call
    it.
- **Why:** Speed, and Rayon runs batches of lines in parallel.
- **Limits:** The Rust API has one caller, so Google's library API rules apply to it as
    recommendations (D-16).

### D-02: Centripetal parameterization by default

- **Status:** accepted.
- **Decision:** `smooth_linestrings`, `smooth_polygon`, and `linestrings_tangent_angles`
    fit with `alpha=0.5`. `CatmullRom` takes any `alpha` or an explicit `grid`.
- **Why:** Centripetal Catmull-Rom curves have no cusps or self-intersections within a
    segment (Yuksel, Schaefer, and Keyser, 2011).

### D-03: Inline segment coefficients, serial evaluation for small requests

- **Status:** accepted (2026-10-22).
- **Decision:** Each segment stores its cubic as power-basis coefficients and evaluates
    with Horner's rule. A request below 4096 points (`PARALLEL_MIN_POINTS` in
    `src/splines.rs`) evaluates serially.
- **Why:** Building a 2,000-vertex spline got about 6x faster, and smoothing 2,000 lines
    of 100 vertices about 2x faster (`CHANGELOG.md`, 0.5.0).

### D-04: Resample at equal arc length

- **Status:** accepted (2026-10-22).
- **Decision:** `uniform_distances`, `smooth_linestrings`, and `smooth_polygon` place
    points at equal arc length. Arc length comes from 16-point Gauss-Legendre
    quadrature, bisected until each piece agrees with its halves to 1e-6 relative; the
    inverse takes Newton steps inside a shrinking bisection bracket.
- **Why:** 0.4.1 iterated toward equal chords and, within 100 iterations, converged on
    none of 2,000 noisy test lines.
- **Rejected:** The fixed-point chord iteration, and with it the `tolerance` and
    `max_iterations` arguments (D-06).
- **Limits:** Spacing is equal along the curve; chords are shorter wherever the curve
    bends between two points.

### D-05: `distances` caps the gap in coordinate units

- **Status:** accepted (2026-10-22).
- **Decision:** `distances` (and `distance` in `smooth_polygon`) is the largest allowed
    arc length between output points, in the coordinates' units. A line of length `L`
    gets `ceil(L / d) + 1` evenly spaced points.
- **Rejected:** 0.4.1's comparison against the centripetal parameter, which counted
    intervals as points and returned 2 points for `distances=10` on a 100-unit line.

### D-06: Deprecated arguments stay accepted and warn

- **Status:** accepted (2026-10-22; keyword-only since 2026-10-23).
- **Decision:** `tolerance` and `max_iterations` are keyword-only, ignored, and raise a
    `DeprecationWarning` when passed. They go in a later release.
- **Why:** D-04 removed what they controlled, and the warning gives callers a release to
    drop them.

### D-07: Caller input errors raise ValueError, never a panic

- **Status:** accepted (2026-10-22).
- **Decision:** Rust validates every input a caller controls and returns
    `Err(SplineError)`, which the bindings map to `ValueError`. `panic!` and `expect`
    guard internal invariants only.
- **Why:** A Rust panic reaches Python as `PanicException`, which callers cannot catch
    as a normal error, and 0.4.1 panicked or returned NaN on several bad inputs.

### D-08: Repeated consecutive vertices are dropped

- **Status:** accepted (2026-10-22).
- **Decision:** When the grid is generated (no explicit `grid`, or `alpha` overriding
    it), consecutive duplicate vertices are dropped before fitting, without a warning.
- **Rejected:** Raising `ValueError`, and keeping them, which produced all-NaN output in
    0.4.1.

### D-09: Vertices are 2D

- **Status:** accepted (2026-10-22).
- **Decision:** Vertex arrays must have exactly two columns; anything else raises
    `ValueError`.
- **Rejected:** Dropping extra columns, as 0.4.1 did without a word.

### D-10: Gaussian pre-smoothing by point reflection and ring wrapping

- **Status:** accepted (2026-10-22).
- **Decision:** An open line is extended past each end by point reflection through its
    end vertex; a closed ring wraps around. A sigma of at least one period of the
    extended line returns the kernel's limit (the chord for open lines, the centroid for
    rings). `sigma_units="length"` measures sigma along the input line and weights each
    vertex by the length it stands for.
- **Why:** Point reflection keeps evenly spaced straight runs in place, and wrapping
    smooths the seam vertex like any other.
- **Rejected:** 0.4.1's mirror reflection, which moved vertices near the ends along the
    line, and its pinned seam vertex.

### D-11: Taubin smoothing with lambda 0.5 and mu -0.53

- **Status:** accepted (2026-10-22).
- **Decision:** `taubin_iterations` runs that many lambda/mu step pairs on the vertices
    before fitting. It cannot be combined with a Gaussian sigma.
- **Why:** Gaussian smoothing shrinks a curve; Taubin's inflating step keeps the size of
    bends drawn with enough vertices.
- **Limits:** A shape drawn with about a dozen or fewer vertices per turn still shrinks,
    and shapes grow slowly as iterations add up. The what's new notebook measures both.

### D-12: G2 and C2 continuity through continuity-enhancing splits

- **Status:** accepted (2026-10-22).
- **Decision:** `continuity="g2"` or `"c2"` splits each piece into three cubics at a
    quarter of its span (Yuksel, "Continuity-Enhancing Degree Elevation and Splits",
    SIGGRAPH 2026). Every vertex and its tangent stay. The default is `"c1"`.
- **Why:** C1 Catmull-Rom curves can jump in curvature at every vertex.

### D-13: The last point is the last vertex

- **Status:** accepted (2026-10-22).
- **Decision:** Evaluation at the last knot returns the last vertex exactly. A closed
    curve treats its last vertex as a repeat of the first when the two lie within 1e-9
    of the ring's bounding-box diagonal.
- **Why:** Rounding at projected-CRS magnitudes left smoothed rings open, and Shapely
    then appended an extra closing vertex.

### D-14: Batches cross the boundary once, without the GIL

- **Status:** accepted (2026-10-22).
- **Decision:** Batch functions pass every line's coordinates as one stacked array plus
    offsets, release the GIL while Rust works, and run on Rayon's global pool, so
    `RAYON_NUM_THREADS` applies.
- **Rejected:** One array per line, and a custom thread pool built on first use, which
    ignored `RAYON_NUM_THREADS`.

### D-15: Python 3.13 to 3.15

- **Status:** accepted (2026-10-23).
- **Decision:** `requires-python = ">=3.13"`, tests run on 3.13 and 3.15, and wheels
    build for 3.13 to 3.15 with PyO3 0.30.

### D-16: Style comes from Google's Rust guide, enforced by tools first

- **Status:** accepted (2026-10-23).
- **Decision:** The `[lints.clippy]` table in `Cargo.toml` and `clippy.toml` enforce the
    rules of the
    [Google Rust style guide](https://google.github.io/styleguide/rustguide.html) that
    clippy can check; `.agents/skills/rust-style/SKILL.md` holds the rest. Python uses
    ruff with `select = ["ALL"]` and NumPy docstrings.
- **Why:** A lint costs no agent context and fails CI. The skill loads only when an
    agent works on Rust: Claude Code through the path-scoped rule in
    `.claude/rules/rust-style.md`, Codex and Gemini through the skill listing.
- **Rejected:** Google's Python guide, which would replace every NumPy docstring, and
    copying the full Rust guide (about 18,000 tokens) into agent context.

### D-17: AGENTS.md is the single instruction file

- **Status:** accepted (2026-10-23).
- **Decision:** `AGENTS.md` is the contributing guide for people and agents, and the
    docs render it. Skills live in `.agents/skills/`, with `.claude/skills` a symlink to
    it. `CLAUDE.md` is git-ignored.
- **Why:** Codex and Gemini read `AGENTS.md`, and Claude Code reads it natively when no
    `CLAUDE.md` exists. A probe on 2026-10-23 showed Claude Code and Codex both listing
    the `rust-style` skill.

### D-18: Development tools live in dependency groups

- **Status:** accepted (2026-10-23).
- **Decision:** Test, lint, docs, typecheck, benchmark, and dev tools are PEP 735
    dependency groups, which pixi maps to features.
- **Rejected:** Optional dependencies, which published them as installable extras.

### D-19: Releases tag a matching version; git-cliff writes the notes

- **Status:** accepted (2026-10-23).
- **Decision:** The version is set in both `pyproject.toml` and `Cargo.toml`, and
    `pixi r release vX.Y.Z` refuses a tag that does not match both or a commit not on
    `origin/main`. The release workflow publishes a release's `CHANGELOG.md` section
    when one exists and git-cliff's notes, with the tag summary, otherwise.
    `CHANGELOG.md` is hand-written through 0.5.0; after each later release,
    `pixi r changelog-update` prepends that release's git-cliff entries below its
    preamble.
- **Rejected:** Rewriting `CHANGELOG.md` from commit history, which would replace the
    hand-written entries.

### D-20: Example notebooks are paired with scripts

- **Status:** accepted (2026-10-23).
- **Decision:** Each notebook in `docs/examples/` is paired with a jupytext `py:percent`
    script and committed with its outputs. `pixi r nb-run` syncs the pair and executes
    the notebooks against a release build. ssort skips the folder, and
    pretty-format-json skips every notebook.
- **Why:** Agents and reviewers edit and diff the script. The docs show stored outputs
    and never execute a notebook.

### D-21: The what's new notebook compares against recorded 0.4.1 outputs

- **Status:** accepted (2026-10-23).
- **Decision:** `docs/examples/data/make_v041_data.py` runs the released 0.4.1 wheel and
    saves its outputs to `v041.npz`. It refuses to run on a newer build, and the
    notebook refuses to run on 0.4.1.
- **Rejected:** Importing both versions in one process, and prose-only comparisons.

### D-22: Pull requests are the only way into main

- **Status:** accepted (2026-10-23).
- **Decision:** A GitHub ruleset blocks direct pushes, force pushes, and deletion of
    `main`, and requires the CI jobs named in `AGENTS.md` to pass before a merge.

### D-23: Actions are pinned to commit SHAs

- **Status:** accepted (2026-10-23).
- **Decision:** Every `uses:` in `.github/workflows/` names a full commit SHA with its
    version in a comment, which Dependabot updates, and every checkout sets
    `persist-credentials: false`.
- **Why:** A moved tag cannot change the code that runs with release permissions, and no
    later step can push with the checkout's token.

## Lessons

### L-01: A probe of the released wheel can import the checkout

`python -c` puts the current directory on `sys.path`, so a probe of the 0.4.1 wheel run
from the repository root imported the checkout's `catsmoothing/` and reported the new
behavior. **Check:** run such probes from outside the repository and print
`catsmoothing.__file__`.

### L-02: mdformat without its frontmatter plugin breaks skills

mdformat turned the YAML frontmatter of a skill and a Claude rule into a heading and a
list, and both silently stop loading that way. **Check:** `mdformat-frontmatter` is in
`.pre-commit-config.yaml`; re-parse frontmatter after any automated rewrite.

### L-03: The default build is a debug build

`preinstall` runs `maturin develop --uv`, which builds without optimizations, so timings
taken after `pixi r test` overstate run time. **Check:** time against
`maturin develop --uv --release`, which `pixi r nb-execute` runs first.

### L-04: A claim about 0.4.1 needs the 0.4.1 wheel

The changelog describes a 0.4.1 bug in which rings at projected-CRS magnitudes lost
vertices near the start, yet the 0.4.1 wheel on PyPI keeps them, including on the fix's
regression input. **Check:** reproduce "0.4.1 did X" against the released wheel before
writing it (Q-02).

### L-05: ssort can move code between notebook cells

ssort sorts a jupytext script as one module, so a function in one cell that reads a
variable from a later cell pulls that statement up. **Check:** ssort excludes
`docs/examples/`.

## Open questions

### Q-01: Move the docs to zensical and GitHub Pages

- **Status:** open; planned as its own pull request.
- water-ogcapi builds with zensical and deploys to GitHub Pages. CatSmoothing builds
    with mkdocs and mkdocs-jupyter on Read the Docs.

### Q-02: Does the ring-vertex changelog entry describe 0.4.1?

- **Status:** open; settle before releasing 0.5.0.
- The v0.4.1 source compares ring endpoints per coordinate with a relative tolerance of
    1e-5, which at UTM magnitudes is 5 m in easting and 40 m in northing and should drop
    a vertex that close to the start, but the 0.4.1 wheel keeps it (L-04).
