---
name: rust-style
description: Google Rust style rules for this repository that clippy does not check. Use before writing, editing, or reviewing Rust code in src/ or tests/.
---

# Rust style

Distilled from the
[Google Rust style guide](https://google.github.io/styleguide/rustguide.html) for this
crate. `cargo fmt` and the `[lints.clippy]` table in `Cargo.toml` enforce the mechanical
rules; this file holds the judgment calls. The crate is not on crates.io and only the
Python bindings in `src/python.rs` call it, so the guide's API-design rules apply as
strong recommendations, and its `#[non_exhaustive]` and Chromium or Fuchsia rules do not
apply.

## Modules and imports

- `src/lib.rs` holds module declarations and re-exports only. Name a new module after
    its primary type (`foo.rs` for `Foo`).
- One `use` statement imports from one module:
    `use crate::splines::{CatmullRom, SplineOptions};` is fine,
    `use crate::{splines::CatmullRom, error::SplineError};` is not.
- Import types and traits unqualified. Call a free function from another module through
    its module name: `presmooth::gaussian(...)`. `io::Error`-style qualification is fine
    when a bare name would be ambiguous.
- Import a trait inside the function that needs it, unless the whole file uses it.

## Ownership and signatures

- Borrow before you `.clone()`. Use `Rc` or `Arc` only when plain ownership would be
    unreasonably complex.
- Take `&[T]`, `&str`, and `Option<&T>`. Return a `Vec` when callers collect anyway;
    return an iterator when it saves an allocation on a hot path.
- Replace a `bool` parameter with an enum when a call site like `fit(v, true)` does not
    say what `true` means.

## Errors and panics

- A Rust panic reaches Python as `PanicException`. Every input a caller can get wrong
    returns `Err(SplineError::...)`, which the bindings turn into `ValueError`.
- `Option` fits a failure whose cause is obvious. `panic!` and `expect` are for broken
    internal invariants only, and `expect` says why the invariant holds:
    `.expect("grid has at least two knots")`.
- An `assert!` message says why the condition must hold or what the caller did wrong,
    never the condition in words.

## Types and traits

- Use a newtype over a type alias when two quantities could be mixed up (a curve
    parameter and an arc length are both `f64`).
- Prefer named fields to tuples, apart from newtypes and tuples used within a few lines.
- Implement the standard trait (`FromStr`, `From`, `Default`, `Display`) in place of an
    ad hoc method, derive when you can, and give options structs a `Default`.
- Define a trait only for generics or `dyn`, never to group methods. Name an extension
    trait `...Ext`. Prefer generics to `dyn Trait` in numeric loops.
- Overload operators only for math, as `Vec2D` does.
- Write a function, type, or module before a `macro_rules!` macro, and a `macro_rules!`
    macro before a procedural one.
- Never reach into a poisoned mutex; let the panic that poisoned it propagate.

## Naming

- Follow the standard library: `as_`, `to_`, `into_` conversions; `new`, with `try_new`
    for a new fallible constructor (`CatmullRom::new` returns `Result` and predates this
    rule); `from_foo` beside a `From` impl; getters `foo()`, `foo_mut()`, `set_foo()`,
    `take_foo()`.
- Spell words out (`vertices`, `tolerance`, `distance`). Keep standard terms such as
    `alpha`, `sigma`, `bc`, and math symbols that match a cited paper's notation.
- Name a lifetime after what it borrows once a signature needs more than two. A generic
    parameter gets a short name hinting at its main bound; `T` fits a parameter that
    takes almost any type.

## Documentation

- Document public functions and types with `///`, and give each new module a `//!`
    header. Fields and enum variants with self-explanatory names can go undocumented.
    Say what the code does not: units, invariants, the parameter domain, the paper an
    equation comes from.
- Add `# Errors` and `# Panics` sections; clippy flags a public function that returns
    `Result` or can panic without them.

## Implementation

- Annotate the binding when inference hides a type: results of `.into()`, `.parse()`,
    and long iterator chains. Use `_` in a turbofish the compiler forces.
- Match or `if let` over checking and then unwrapping.
- Write a `for` loop over a long or stateful combinator chain, and bind a chain to a
    named variable before looping over it. Short chains like `.iter().any(...)` stay.
- Use `format!` or `write!` to join more than two strings.
- No `unsafe` unless no safe version exists; then a `// SAFETY:` comment, a stated
    reason, and a benchmark if the reason is speed. No unstable features, no code before
    `main`, no `panic::catch_unwind`.
- Mark a function `const` only if it will stay `const`. Use `#[rustfmt::skip]` only to
    keep deliberate alignment, such as a coefficient table.
